from typing import TYPE_CHECKING, Optional, List, Tuple

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge
    from spacy.tokens import Span


class Inference:
    def __init__(
        self,
        graph_ref: "Graph",
        llm_extractor,
        spacy_nlp,
        max_new_concepts: int,
        max_new_properties: int,
        persona: str,
    ):
        self._graph = graph_ref
        self._llm = llm_extractor
        self._spacy_nlp = spacy_nlp
        self._max_new_concepts = max_new_concepts
        self._max_new_properties = max_new_properties
        self._persona = persona
        self._append_adjectival_hints_fn = None
        self._get_sentences_text_based_nodes_fn = None

    def set_callbacks(
        self,
        append_adjectival_hints_fn: callable,
        get_sentences_text_based_nodes_fn: callable,
    ):
        self._append_adjectival_hints_fn = append_adjectival_hints_fn
        self._get_sentences_text_based_nodes_fn = get_sentences_text_based_nodes_fn

    def infer_new_relationships_step_0(
        self, sent: "Span"
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        if not self._get_sentences_text_based_nodes_fn:
            return [], []

        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._get_sentences_text_based_nodes_fn(
                [sent], create_unexistent_nodes=False
            )
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        if self._append_adjectival_hints_fn:
            nodes_from_text = self._append_adjectival_hints_fn(nodes_from_text, sent)

        for _ in range(3):
            try:
                object_properties_dict = (
                    self._llm.infer_objects_and_properties_first_sentence(
                        nodes_from_text, sent.text, self._persona
                    )
                )
                break
            except:
                continue
        else:
            return [], []

        for _ in range(3):
            try:
                new_relationships = (
                    self._llm.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        object_properties_dict["concepts"][: self._max_new_concepts],
                        object_properties_dict["properties"][
                            : self._max_new_properties
                        ],
                        sent.text,
                        self._persona,
                    )
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List["Node"],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        doc = self._spacy_nlp(text)
        sent_span = doc[0 : len(doc)]
        if self._append_adjectival_hints_fn:
            nodes_from_text = self._append_adjectival_hints_fn(
                nodes_from_text, sent_span
            )

        object_properties_dict = None
        for _ in range(3):
            try:
                object_properties_dict = self._llm.infer_objects_and_properties(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    text,
                    self._persona,
                )
                break
            except:
                continue

        if object_properties_dict is None:
            return [], []

        for _ in range(3):
            try:
                new_relationships = self._llm.generate_new_inferred_relationships(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    object_properties_dict["concepts"][: self._max_new_concepts],
                    object_properties_dict["properties"][: self._max_new_properties],
                    text,
                    self._persona,
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []
