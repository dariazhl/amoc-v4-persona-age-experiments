import logging
from typing import TYPE_CHECKING, Optional, List, Set, Callable, Dict
from amoc.core.node import NodeType, NodeSource, NodeProvenance
from amoc.utils.spacy_utils import (
    extract_adjectival_modifiers,
    extract_deterministic_relation_candidates,
)

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
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
        self._edge_visibility: int = 3
        self._current_sentence_index: int = 0

    def set_callbacks(
        self,
        add_edge_fn: Callable,
    ):
        self._add_edge_fn = add_edge_fn

    def set_sentence_context(self, sentence_index: int, edge_visibility: int):
        self._current_sentence_index = sentence_index
        self._edge_visibility = edge_visibility

    def resolve_pronouns_wrapper(self, context: str, sentence: str) -> Dict[str, str]:
        result = self._llm.resolve_pronouns(
            sentence=sentence, context=context, persona=self._persona
        )

        if not isinstance(result, dict):
            logging.warning(f"Pronoun resolution returned non-dict: {result}")
            return {}

        return result

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

        if not self._add_edge_fn:
            return

        sentence_node_map = {tuple(node.lemmas): node for node in sentence_nodes}

        # list of nodes that are explicitly mentioned in the sentence
        def get_node(lemma: str):
            if not lemma:
                return None
            node = sentence_node_map.get((lemma,))
            if node is not None:
                return node
            return self._graph.get_node([lemma])

        # creates an edge between two nodes using the _add_edge_fn callback
        #  _add_edge_fn calls edge_admission.add_edge that handles duplication, visibility etc and returns edge
        #  if successful, it marks edge as part of the current sentence with full activation
        def assert_edge(src, dst, label):
            edge = self._add_edge_fn(
                src,
                dst,
                label=label,
                edge_forget=self._edge_visibility,
                created_at_sentence=self._current_sentence_index,
            )
            if edge:
                edge.mark_as_current_sentence(reset_score=True)

        # iterate through candidates and create edges for those that have both subject and object nodes present in the graph
        # create S-V-O triplet structure
        for candidate in extract_deterministic_relation_candidates(sent):
            subj_node = get_node(candidate.subject_lemma)
            if subj_node is None:
                continue

            obj_node = get_node(candidate.object_lemma)
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
