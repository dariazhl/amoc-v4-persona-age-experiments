from typing import TYPE_CHECKING, Optional, List, Set
import logging
from amoc.utils.spacy_utils import (
    get_concept_lemmas,
    canonicalize_node_text,
    get_content_words_from_sent,
)
from amoc.core.node import NodeType, NodeSource, NodeProvenance

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from spacy.tokens import Span


class NodeAdmission:
    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        story_lemmas: Set[str],
        persona_only_lemmas: Set[str],
        max_distance_from_active_nodes: int,
        debug: bool = False,
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._story_lemmas = story_lemmas
        self._persona_only_lemmas = persona_only_lemmas
        self._max_distance = max_distance_from_active_nodes
        self._debug = debug
        self._ever_admitted_nodes = set()
        self._layout_depth = 3
        self._has_active_attachment_fn = None
        self._canonicalize_and_classify_fn = None

    def set_callbacks(
        self,
        has_active_attachment_fn: callable,
        canonicalize_and_classify_fn: callable,
    ):
        self._has_active_attachment_fn = has_active_attachment_fn
        self._canonicalize_and_classify_fn = canonicalize_and_classify_fn

    def admit_node(
        self,
        lemma: str,
        node_type: "NodeType",
        provenance: str = "STORY_EXPLICIT",
        sent: Optional["Span"] = None,
    ) -> bool:
        lemma = (lemma or "").lower().strip()
        if not lemma:
            return False

        if not self._graph._provenance_ops.passes_length_policy(lemma):
            return False

        # Reject nodes from internal provenance
        if provenance in {
            "LLM_PROMPT",
            "GRAPH_SERIALIZATION",
            "CSV",
            "PLOTTING",
            "META",
        }:
            return False

        if provenance == "STORY_EXPLICIT":
            if sent is None:
                return False
            token_matches = [tok for tok in sent if tok.lemma_.lower() == lemma]
            if not token_matches:
                return False
            for tok in token_matches:
                if tok.pos_ in {"VERB", "AUX"}:
                    return False

        if provenance == "STORY_EXPLICIT":
            if lemma not in self._story_lemmas:
                if not (lemma.endswith("s") and lemma[:-1] in self._story_lemmas):
                    return False

        if node_type == NodeType.PROPERTY:
            if sent is None:
                return False
            grounded = any(
                tok.lemma_.lower() == lemma
                and tok.pos_ == "ADJ"
                and tok.dep_ in {"amod", "acomp", "attr"}
                for tok in sent
            )
            if not grounded:
                return False

        # Inference admission
        is_story_grounded = lemma in self._story_lemmas
        is_allowed_inference = (
            provenance in {"INFERRED_RELATION", "INFERENCE_BASED"}
            and is_story_grounded
            and self._has_active_attachment_fn
            and self._has_active_attachment_fn(lemma)
        )

        if provenance != "STORY_EXPLICIT" and not is_allowed_inference:
            return False

        # Track new node admission
        is_new = lemma not in self._ever_admitted_nodes
        self._ever_admitted_nodes.add(lemma)

        # Set limits max no. nodes
        if is_new:
            total_nodes = len(self._ever_admitted_nodes)
            if total_nodes > 40:
                self._layout_depth = max(self._layout_depth, 6)
            elif total_nodes > 25:
                self._layout_depth = max(self._layout_depth, 5)
            elif total_nodes > 12:
                self._layout_depth = max(self._layout_depth, 4)

        return True

    def validate_node_provenance(
        self,
        lemma: str,
        current_sentence_text: Optional[str] = None,
        *,
        allow_bootstrap: bool = False,
    ) -> bool:
        lemma_lower = lemma.lower()

        # Reject persona-only lemmas
        if lemma_lower in self._persona_only_lemmas:
            return False

        # Must appear in story text
        if lemma_lower in self._story_lemmas:
            return True
        if current_sentence_text:
            sent_doc = self._spacy_nlp(current_sentence_text)
            sent_lemmas = {tok.lemma_.lower() for tok in sent_doc if tok.is_alpha}
            if lemma_lower in sent_lemmas:
                return True

        # Allow concepts that already exist
        existing_node = self._graph.get_node([lemma_lower])
        if existing_node is not None:
            return True

        if allow_bootstrap:
            return True

        return False

    def resolve_node_from_sentence_text(
        self,
        text: str,
        curr_sentences_nodes: List["Node"],
        curr_sentences_words: List[str],
        node_source: "NodeSource",
        create_node: bool,
    ) -> Optional["Node"]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]

        if not create_node or not self._canonicalize_and_classify_fn:
            return None

        canon, inferred_type = self._canonicalize_and_classify_fn(text)
        if inferred_type is None:
            return None

        lemmas = get_concept_lemmas(self._spacy_nlp, canon)
        if not self.admit_node(
            lemma=canon,
            node_type=inferred_type,
            provenance="TEXT_FALLBACK",
        ):
            return None

        return self._graph.add_or_get_node(lemmas, canon, inferred_type, node_source)

    def resolve_node_from_relationship_text(
        self,
        text: str,
        graph_active_nodes: List["Node"],
        curr_sentences_nodes: List["Node"],
        curr_sentences_words: List[str],
        node_source: "NodeSource",
        create_node: bool,
    ) -> Optional["Node"]:
        # 1. Exact sentence match
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]

        # 2. Canonicalize
        if not self._canonicalize_and_classify_fn:
            return None

        canon, inferred_type = self._canonicalize_and_classify_fn(text)
        if inferred_type is None:
            return None

        lemmas = get_concept_lemmas(self._spacy_nlp, canon)
        if not lemmas:
            return None
        # 3. Try match active graph
        for node in graph_active_nodes:
            if lemmas == node.lemmas:
                return node
        # 4. Create node if allowed
        if not create_node:
            return None

        if canon in {"subject", "object", "relation", "properties"}:
            return None

        if not self.admit_node(
            lemma=lemmas[0],
            node_type=inferred_type,
            provenance=NodeProvenance.STORY_TEXT,
        ):
            return None

        return self._graph.add_or_get_node(
            lemmas,
            canon,
            inferred_type,
            node_source,
        )

    def find_node_by_text(
        self,
        text: str,
        candidates,
    ) -> Optional["Node"]:
        canon = canonicalize_node_text(self._spacy_nlp, text)
        lemmas = tuple(get_concept_lemmas(self._spacy_nlp, canon))
        for node in candidates:
            if lemmas == tuple(node.lemmas):
                return node
        return None

    def node_token_for_matrix(self, node: "Node") -> str:
        return (node.get_text_representer() or "").strip().lower()

    def passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List["Node"],
        graph_active_nodes: List["Node"],
        explicit_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
        get_nodes_with_active_edges_fn: callable,
        graph_active_edge_nodes: Optional[Set["Node"]] = None,
        allow_inference_bridge: bool = False,
    ) -> bool:

        # Canonicalize
        subject = canonicalize_node_text(self._spacy_nlp, subject)
        obj = canonicalize_node_text(self._spacy_nlp, obj)

        subj_key = tuple(get_concept_lemmas(self._spacy_nlp, subject))
        obj_key = tuple(get_concept_lemmas(self._spacy_nlp, obj))

        if not self._graph.nodes:
            return True

        active_nodes = set(get_nodes_with_active_edges_fn())
        frontier_nodes = active_nodes | explicit_nodes | carryover_nodes
        frontier_keys = {tuple(n.lemmas) for n in frontier_nodes}

        # Preserve already-connected relationships
        if subj_key in frontier_keys and obj_key in frontier_keys:
            return True

        # Allow if at least one endpoint touches frontier
        if subj_key in frontier_keys or obj_key in frontier_keys:
            return True

        # Otherwise reject
        return False

    def extract_sentence_text_based_nodes(
        self,
        previous_sentences: List["Span"],
        current_sentence_index: int,
        create_unexistent_nodes: bool = True,
    ) -> tuple:
        # garbage words that are generated by the LLM that have nothing to do with the story
        META_LEMMAS = {"subject", "object", "entity", "concept", "property", "thing"}

        text_based_nodes = []
        text_based_words = []

        for sent in previous_sentences:
            content_words = get_content_words_from_sent(self._spacy_nlp, sent)
            for word in content_words:
                lemma = word.lemma_.lower().strip()
                if not lemma:
                    continue

                if word.pos_ in {"NOUN", "PROPN"}:
                    if lemma in META_LEMMAS:
                        continue
                    node = self._graph.get_node([lemma])
                    if node is None and create_unexistent_nodes:
                        node = self._graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.CONCEPT,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            origin_sentence=current_sentence_index,
                            mark_explicit=False,
                        )
                    if node is not None:
                        node.mark_explicit_in_sentence(current_sentence_index)
                        text_based_nodes.append(node)
                        text_based_words.append(lemma)

                elif word.pos_ == "ADJ" or (
                    word.pos_ == "VERB"
                    and word.tag_ == "VBN"
                    and word.dep_ in {"acomp", "attr", "amod", "ROOT"}
                ):
                    node = self._graph.get_node([lemma])
                    if node is None and create_unexistent_nodes:
                        node = self._graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.PROPERTY,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            origin_sentence=current_sentence_index,
                            mark_explicit=False,
                        )
                    if node is not None:
                        node.mark_explicit_in_sentence(current_sentence_index)
                        text_based_nodes.append(node)
                        text_based_words.append(lemma)

        seen = set()
        unique_nodes = []
        unique_words = []
        for node, word in zip(text_based_nodes, text_based_words):
            if node not in seen:
                seen.add(node)
                unique_nodes.append(node)
                unique_words.append(word)

        return unique_nodes, unique_words

    def get_phrase_level_concepts(
        self,
        sent: "Span",
        admit_node_fn: callable,
    ) -> List["Node"]:
        phrase_nodes = []

        # spaCy noun chunks = adjective + noun phrases
        for chunk in sent.noun_chunks:
            # Extract the head noun from the chunk
            head_noun = None
            for tok in chunk:
                if tok.pos_ in {"NOUN", "PROPN"}:
                    head_noun = tok
                    break
            if head_noun is None:
                continue

            # AMoC paper: nodes are "country" not "the country"
            lemma = head_noun.lemma_.lower()
            if not admit_node_fn(
                lemma=lemma,
                node_type=NodeType.CONCEPT,
                provenance="STORY_TEXT",
            ):
                continue
            node = self._graph.add_or_get_node(
                lemmas=[lemma],
                actual_text=lemma,
                node_type=NodeType.CONCEPT,
                node_source=NodeSource.TEXT_BASED,
                provenance=NodeProvenance.STORY_TEXT,
            )
            if node is not None:
                phrase_nodes.append(node)
        return phrase_nodes
