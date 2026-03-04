from typing import TYPE_CHECKING, Optional, List, Set, Callable
from amoc.graph.node import NodeType, NodeSource, NodeProvenance
from amoc.nlp.spacy_utils import (
    extract_adjectival_modifiers,
    extract_deterministic_relation_candidates,
)

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from spacy.tokens import Span


class LinguisticProcessing:
    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        llm_extractor,
        story_lemmas: Set[str],
        story_text: str = "",
        persona: str = "",
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._llm = llm_extractor
        self._story_lemmas = story_lemmas
        self._story_text = story_text
        self._persona = persona
        self._add_edge_fn: Optional[Callable] = None
        self._classify_relation_fn: Optional[Callable] = None
        self._edge_visibility: int = 3
        self._current_sentence_index: int = 0

    def set_callbacks(
        self,
        add_edge_fn: Callable,
        classify_relation_fn: Callable,
    ):
        self._add_edge_fn = add_edge_fn
        self._classify_relation_fn = classify_relation_fn

    def set_sentence_context(self, sentence_index: int, edge_visibility: int):
        self._current_sentence_index = sentence_index
        self._edge_visibility = edge_visibility

    def resolve_pronouns(self, text: str) -> str:
        resolved = self._llm.resolve_pronouns(text, self._persona)
        if not isinstance(resolved, str) or not resolved.strip():
            return text
        low = resolved.lower()
        if "does not mention any pronouns" in low or "no pronouns to replace" in low:
            return text
        return resolved

    def append_adjectival_hints(
        self,
        nodes_from_text: str,
        sent: "Span",
    ) -> str:
        adj_mods = extract_adjectival_modifiers(sent)

        for mod in adj_mods:
            nodes_from_text += (
                f" - ({mod['adjective']}, PROPERTY) [describes {mod['head_noun']}]\n"
            )

        return nodes_from_text

    # Old code: all edges + nodes are extracted by the LLM
    # New code: same principle applied on new code creates great variability
    # Current design: persona should not influence the LLM extraction of nodes + edges, therefore we extract nodes determinically and we let inference handle variability between personas
    def extract_deterministic_structure(
        self,
        sent,
        sentence_nodes: List["Node"],
        sentence_words: List[str],
    ) -> None:

        if not self._add_edge_fn or not self._classify_relation_fn:
            return

        sentence_node_map = {tuple(node.lemmas): node for node in sentence_nodes}

        def get_node_by_lemma(lemma: str):
            if not lemma:
                return None
            from_sentence = sentence_node_map.get((lemma,))
            if from_sentence is not None:
                return from_sentence
            return self._graph.get_node([lemma])

        def assert_edge(src, dst, label):
            edge = self._add_edge_fn(
                src,
                dst,
                label=label,
                edge_forget=self._edge_visibility,
                created_at_sentence=self._current_sentence_index,
            )
            if edge:
                edge.mark_as_asserted(reset_score=True)

        for candidate in extract_deterministic_relation_candidates(sent):
            subj_node = get_node_by_lemma(candidate.subject_lemma)
            if subj_node is None:
                continue

            obj_node = get_node_by_lemma(candidate.object_lemma)
            if obj_node is None and candidate.object_is_property:
                obj_node = self._graph.add_or_get_node(
                    [candidate.object_lemma],
                    candidate.object_lemma,
                    NodeType.PROPERTY,
                    NodeSource.TEXT_BASED,
                    provenance=NodeProvenance.STORY_TEXT,
                    origin_sentence=self._current_sentence_index,
                )

            if obj_node is None:
                continue

            assert_edge(subj_node, obj_node, candidate.relation_label)
