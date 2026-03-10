from typing import TYPE_CHECKING, Optional, List, Set, Tuple, Dict
import copy
import logging
import re
import networkx as nx
from amoc.core.node import NodeType

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge
    from amoc.graph_views.per_sentence import PerSentenceGraph
    from spacy.tokens import Span


class SentenceRuntime:

    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        story_lemmas: Set[str],
        max_distance_from_active_nodes: int,
        edge_visibility: int,
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._story_lemmas = story_lemmas
        self._max_distance = max_distance_from_active_nodes
        self._edge_visibility = edge_visibility

        self._explicit_nodes_current_sentence: Set["Node"] = set()
        self._per_sentence_view: Optional["PerSentenceGraph"] = None
        self._triplet_intro = {}
        self._current_sentence_index = None
        self._current_sentence_text = ""

    def set_runtime_state_refs(
        self,
        anchor_nodes: Set["Node"],
        explicit_nodes: Set["Node"],
        triplet_intro: dict,
    ):
        self._explicit_nodes_current_sentence = explicit_nodes
        self._triplet_intro = triplet_intro

    def configure_graph_for_sentence(self, idx: int, lemmas: Set[str]) -> None:
        self._graph.set_current_sentence(idx)
        self._graph.set_current_sentence_lemmas(lemmas)

    # create deep copy of all relevant states for rollback (if needed)
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
            copy.deepcopy(triplet_intro),
            copy.deepcopy(per_sentence_view),
            copy.deepcopy(recently_deactivated),
            copy.deepcopy(prev_active_nodes),
        )

    def clean_resolved_sentence(self, orig_text: str, candidate: str) -> str:
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
        resolve_pronouns_wrapper_fn: callable = None,
    ) -> Tuple[List[Tuple["Span", str, str]], Set[str]]:
        doc = self._spacy_nlp(story_text)
        story_lemma_set = {t.lemma_.lower() for t in doc if t.is_alpha}
        resolved_sentences: List[Tuple["Span", str, str]] = []

        # Build context from previous sentences
        context_sentences = []

        for orig_sent in doc.sents:
            resolved_text = orig_sent.text

            if replace_pronouns and resolve_pronouns_wrapper_fn and context_sentences:
                context = " ".join(context_sentences[-5:])  # Last 5 sentences

                # Get pronoun mapping
                mapping = resolve_pronouns_wrapper_fn(context, orig_sent.text)

                if mapping and isinstance(mapping, dict):
                    resolved_text = self.apply_pronoun_mapping(orig_sent.text, mapping)
                    logging.info(f"Pronoun mapping: {mapping} -> '{resolved_text}'")

            # Validate no contamination
            if resolved_text.strip().startswith("{"):
                logging.error("JSON contamination detected — reverting to original.")
                resolved_text = orig_sent.text

            # Parse resolved text
            resolved_doc = self._spacy_nlp(resolved_text)
            if not resolved_doc:
                resolved_text = orig_sent.text
                resolved_doc = self._spacy_nlp(resolved_text)

            resolved_span = resolved_doc[0 : len(resolved_doc)]

            # Store original for context (not resolved)
            context_sentences.append(orig_sent.text)
            resolved_sentences.append((resolved_span, resolved_text, orig_sent.text))

        return resolved_sentences, story_lemma_set

    # Apply pronoun mapping to a sentence
    def apply_pronoun_mapping(self, sentence: str, mapping: Dict[str, str]) -> str:
        doc = self._spacy_nlp(sentence)
        tokens = []

        for token in doc:
            if token.pos_ == "PRON" and token.text in mapping:
                # Replace pronoun with its referent
                tokens.append(mapping[token.text])
                logging.debug(f"Replaced '{token.text}' with '{mapping[token.text]}'")
            else:
                tokens.append(token.text)

        # Reconstruct sentence
        return " ".join(tokens)

    # Active graph requires a reset of the sentence state
    # Method deactivates all edges, clears explicit nodes, keeps track of carryovers
    def reset_sentence_state(self, original_text: str) -> Set["Node"]:
        self._graph.deactivate_all_edges_wrapper()
        self._current_sentence_text = original_text

        nodes_before_sentence = set(self._graph.nodes)
        self._explicit_nodes_current_sentence = set()
        return nodes_before_sentence

    # Creates a frozen snapshot of the active subgraph for the current sentence
    def build_per_sentence_view(
        self,
        explicit_nodes: List["Node"],
        sentence_index: int,
        build_per_sentence_graph_fn: callable,
    ) -> Optional["PerSentenceGraph"]:
        admitted_nodes = []

        for node in explicit_nodes:
            if node is None:
                continue

            label = node.get_text_representer()
            if not label:
                continue

            lemma = label.lower().strip()
            # check that the lemma exists in the story lemma
            # ie. knight rode through forest -> ["knight"]
            if lemma not in self._story_lemmas:
                continue
            # look up a canonical node in the graph
            # this ensures that nodes with the same lemma but different provenance or types are still merged
            # in the per-sentence view, and that we don't end up with multiple nodes representing the same
            # concept in the same sentence
            canonical_node = self._graph.get_node(node.lemmas)

            if canonical_node is None:
                canonical_node = self._graph.add_or_get_node(
                    node.lemmas,
                    label,
                    node.node_type,
                    node.node_source,
                    provenance=node.provenance,
                    origin_sentence=sentence_index,
                )

            if canonical_node:
                admitted_nodes.append(canonical_node)
        # call the graph building function with the cumulative graph, the admitted nodes, max distance etc.
        view = build_per_sentence_graph_fn(
            cumulative_graph=self._graph,
            explicit_nodes=admitted_nodes,
            max_distance=self._max_distance,
            anchor_nodes=set(),
            sentence_index=sentence_index,
            repair_callback=None,
        )
        self._per_sentence_view = view
        return view

    def get_attachable_nodes_for_sentence(
        self,
        get_nodes_with_active_edges_fn: callable,
    ) -> Set["Node"]:
        if self._per_sentence_view is not None:
            return set(self._per_sentence_view.explicit_nodes) | set(
                self._per_sentence_view.carryover_nodes
            )

        return self._explicit_nodes_current_sentence | get_nodes_with_active_edges_fn()

    def extract_sentence_lemmas(self, text: str) -> Set[str]:
        return {w.lower() for w in re.findall(r"[a-zA-Z]+", text)}
