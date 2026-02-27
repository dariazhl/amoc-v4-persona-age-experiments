from typing import TYPE_CHECKING, Optional, List, Set, Tuple
import copy
import logging
import re
import networkx as nx

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge
    from amoc.graph.per_sentence_graph import PerSentenceGraph
    from spacy.tokens import Span

class SentenceOps:

    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        story_lemmas: Set[str],
        max_distance_from_active_nodes: int,
        edge_visibility: int,
        strict_attachment_constraint: bool = True,
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._story_lemmas = story_lemmas
        self._max_distance = max_distance_from_active_nodes
        self._edge_visibility = edge_visibility
        self._strict_attachment_constraint = strict_attachment_constraint

        self._anchor_nodes: Set["Node"] = set()
        self._explicit_nodes_current_sentence: Set["Node"] = set()
        self._per_sentence_view: Optional["PerSentenceGraph"] = None
        self._triplet_intro = {}
        self._current_sentence_index = None
        self._current_sentence_text = ""

    def set_state_refs(
        self,
        anchor_nodes: Set["Node"],
        explicit_nodes: Set["Node"],
        triplet_intro: dict,
    ):
        self._anchor_nodes = anchor_nodes
        self._explicit_nodes_current_sentence = explicit_nodes
        self._triplet_intro = triplet_intro

    def set_current_sentence(self, idx: int, text: str):
        self._current_sentence_index = idx
        self._current_sentence_text = text

    def snapshot_sentence_state(
        self,
        anchor_nodes: Set["Node"],
        triplet_intro: dict,
        per_sentence_view: Optional["PerSentenceGraph"],
        recently_deactivated: Optional[Set["Node"]],
        prev_active_nodes: Optional[Set["Node"]],
    ) -> Tuple:
        return (
            copy.deepcopy(self._graph),
            set(anchor_nodes),
            copy.deepcopy(triplet_intro),
            copy.deepcopy(per_sentence_view),
            copy.deepcopy(recently_deactivated),
            copy.deepcopy(prev_active_nodes),
        )

    def restore_sentence_state(
        self,
        snapshot: Tuple,
    ) -> Tuple:
        return snapshot

    def _clean_resolved_sentence(self, orig_text: str, candidate: str) -> str:
        if not isinstance(candidate, str) or not candidate.strip():
            return orig_text
        cleaned = re.sub(r"<[^>]+>", " ", candidate)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        prompt_scaffolding_patterns = [
            r"(?i)^the text is:\s*",
            r"(?i)^here is the text:\s*",
            r"(?i)^the sentence is:\s*",
            r"(?i)^text:\s*",
            r"(?i)^sentence:\s*",
            r"(?i)^replace the pronouns.*?:\s*",
        ]
        for pattern in prompt_scaffolding_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()

        llm_meta_patterns = [
            r"(?i)the text does not contain pronouns?",
            r"(?i)there are no pronouns? to replace",
            r"(?i)no pronouns? (?:to replace|found|present|in (?:the|this) (?:text|sentence))",
            r"(?i)does not (?:have|contain) (?:any )?pronouns?",
            r"(?i)without (?:any )?pronouns?",
            r"(?i)pronouns? (?:have been |are |were )?(?:already )?replaced",
        ]
        for pattern in llm_meta_patterns:
            if re.search(pattern, cleaned):
                return orig_text

        orig_doc = self._spacy_nlp(orig_text)
        orig_tokens = {t.lemma_.lower() for t in orig_doc if t.is_alpha}
        best_sent = None
        best_overlap = -1
        cand_doc = self._spacy_nlp(cleaned)
        for sent in cand_doc.sents:
            toks = {t.lemma_.lower() for t in sent if t.is_alpha}
            overlap = len(orig_tokens & toks)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sent = sent.text.strip()

        chosen = best_sent or cleaned

        max_len = max(len(orig_text) * 2 + 40, 400)
        if len(chosen) > max_len:
            chosen = chosen[:max_len].rstrip(" ,.;") + "..."
        return chosen or orig_text

    def resolve_sentences(
        self,
        story_text: str,
        replace_pronouns: bool,
        resolve_pronouns_fn: callable = None,
    ) -> Tuple[List[Tuple["Span", str, str]], Set[str]]:
        doc = self._spacy_nlp(story_text)
        story_lemma_set = {t.lemma_.lower() for t in doc if t.is_alpha}
        resolved_sentences: List[Tuple["Span", str, str]] = []

        for orig_sent in doc.sents:
            resolved_text = orig_sent.text
            if replace_pronouns and resolve_pronouns_fn:
                candidate = resolve_pronouns_fn(orig_sent.text)
                if isinstance(candidate, str) and candidate.strip():
                    resolved_text = self._clean_resolved_sentence(orig_sent.text, candidate)
            if resolved_text and resolved_text.strip().startswith("{"):
                logging.error(
                    "LLM JSON contamination detected — reverting to original sentence."
                )
                resolved_text = orig_sent.text
            resolved_doc = self._spacy_nlp(resolved_text)
            if not resolved_doc:
                resolved_text = orig_sent.text
                resolved_doc = self._spacy_nlp(resolved_text)

            resolved_span = resolved_doc[0 : len(resolved_doc)]
            if resolved_text.lower().startswith(("user", "assistant", "system")):
                continue
            resolved_sentences.append((resolved_span, resolved_text, orig_sent.text))

        return resolved_sentences, story_lemma_set

    def reset_sentence_state(self, original_text: str) -> Set["Node"]:
        self._graph.deactivate_all_edges()
        self._current_sentence_text = original_text

        nodes_before_sentence = set(self._graph.nodes)
        self._explicit_nodes_current_sentence = set()
        return nodes_before_sentence

    def build_per_sentence_view(
        self,
        explicit_nodes: List["Node"],
        sentence_index: int,
        build_per_sentence_graph_fn: callable,
    ) -> Optional["PerSentenceGraph"]:
        if not self._strict_attachment_constraint:
            self._per_sentence_view = None
            return None

        admitted_nodes = []

        for node in explicit_nodes:
            if node is None:
                continue

            label = node.get_text_representer()
            if not label:
                continue

            lemma = label.lower().strip()

            if lemma not in self._story_lemmas:
                continue

            canonical_node = self._graph.get_node(node.lemmas)
            if node.get_text_representer().lower() in {
                "has", "is", "was", "were", "be",
            }:
                continue

            if canonical_node is None:
                canonical_node = self._graph.add_or_get_node(
                    node.lemmas,
                    label,
                    node.node_type,
                    node.source,
                    provenance=node.provenance,
                    origin_sentence=sentence_index,
                )

            if canonical_node:
                admitted_nodes.append(canonical_node)

        view = build_per_sentence_graph_fn(
            cumulative_graph=self._graph,
            explicit_nodes=admitted_nodes,
            max_distance=self._max_distance,
            anchor_nodes=self._anchor_nodes,
            sentence_index=sentence_index,
            repair_callback=None,
        )

        active_nodes = set(view.explicit_nodes) | set(view.carryover_nodes)

        connected_nodes = set()
        for e in view.active_edges:
            connected_nodes.add(e.source_node)
            connected_nodes.add(e.dest_node)

        if connected_nodes:
            for node in active_nodes:
                has_edge = any(
                    e.active and (e.source_node == node or e.dest_node == node)
                    for e in view.active_edges
                )

                if not has_edge:
                    anchor_candidates = sorted(
                        connected_nodes,
                        key=lambda n: sum(
                            1
                            for e in self._graph.edges
                            if e.active and (e.source_node == n or e.dest_node == n)
                        ),
                        reverse=True,
                    )

                    anchor = next((n for n in anchor_candidates if n != node), None)

                    if anchor:
                        edge = self._graph.add_edge(
                            anchor,
                            node,
                            "relates_to",
                            self._edge_visibility,
                            persona_influenced=False,
                            inferred=False,
                        )
                        if edge:
                            edge.active = True
                            edge.asserted_this_sentence = False
                            edge.reactivated_this_sentence = False

            view = build_per_sentence_graph_fn(
                cumulative_graph=self._graph,
                explicit_nodes=admitted_nodes,
                max_distance=self._max_distance,
                anchor_nodes=self._anchor_nodes,
                sentence_index=sentence_index,
                repair_callback=None,
            )

        self._per_sentence_view = view
        return view

    def get_attachable_nodes_for_sentence(
        self,
        get_nodes_with_active_edges_fn: callable,
    ) -> Set["Node"]:
        from amoc.graph.node import NodeType

        if self._per_sentence_view is not None:
            return (
                set(self._per_sentence_view.explicit_nodes)
                | set(self._per_sentence_view.carryover_nodes)
                | set(self._per_sentence_view.anchor_nodes)
            )

        attachable = (
            self._explicit_nodes_current_sentence
            | self._anchor_nodes
            | get_nodes_with_active_edges_fn()
        )

        for edge in self._graph.edges:
            if (
                edge.source_node in attachable
                and edge.dest_node.node_type == NodeType.EVENT
            ):
                attachable.add(edge.dest_node)
            if (
                edge.dest_node in attachable
                and edge.source_node.node_type == NodeType.EVENT
            ):
                attachable.add(edge.source_node)

        return attachable

    def compute_explicit_nodes(
        self,
        sent: "Span",
        text_based_nodes: List["Node"],
    ) -> Set["Node"]:
        from amoc.graph.node import NodeType

        sentence_lemma_set = {token.lemma_.lower() for token in sent}

        return {
            n
            for n in text_based_nodes
            if n.node_type in {NodeType.CONCEPT, NodeType.PROPERTY}
            and any(lemma in sentence_lemma_set for lemma in n.lemmas)
        }

    def activate_explicit_nodes(
        self,
        explicit_nodes: Set["Node"],
        activation_max_distance: int,
    ) -> None:
        for node in explicit_nodes:
            node.activation_score = activation_max_distance
            node.active = True

            if node not in self._graph.nodes:
                self._graph.nodes.add(node)
