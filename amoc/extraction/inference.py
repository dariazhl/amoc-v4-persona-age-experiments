from typing import TYPE_CHECKING, Optional, List, Tuple
import logging

from amoc.llm.parsing import coerce_triplet_list

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

    def _relaxed(self) -> bool:
        return bool(getattr(self._llm, "relaxed_inference", False))

    @staticmethod
    def _has_concepts_or_properties(result) -> bool:
        return isinstance(result, dict) and bool(
            result.get("concepts") or result.get("properties")
        )

    def _relaxed_relationships(self, result):
        if not isinstance(result, dict):
            return None
        concepts = coerce_triplet_list(result.get("concept_relationships", []))
        properties = coerce_triplet_list(result.get("property_relationships", []))
        if not concepts and not properties:
            logging.warning(
                f"Relaxed inference: no usable triplets in relationship response: {str(result)[:300]}"
            )
            return None
        return concepts, properties

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
                if object_properties_dict:
                    logging.info(
                        f"Inference attempt succeeded: {object_properties_dict}"
                    )
                    break
            except Exception as e:
                logging.error(f"Inference attempt failed: {e}", exc_info=True)
                continue
        else:
            logging.warning("All 3 inference attempts failed for first sentence")
            return [], []

        for _ in range(3):
            try:
                new_relationships = (
                    self._llm.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        (
                            object_properties_dict.get("concepts") or []
                            if self._relaxed()
                            else object_properties_dict["concepts"]
                        )[: self._max_new_concepts],
                        (
                            object_properties_dict.get("properties") or []
                            if self._relaxed()
                            else object_properties_dict["properties"]
                        )[: self._max_new_properties],
                        sent.text,
                        self._persona,
                    )
                )
                if not new_relationships:
                    continue
                if self._relaxed():
                    coerced = self._relaxed_relationships(new_relationships)
                    if coerced is None:
                        continue
                    logging.info(f"Relationship generation attempt succeeded")
                    return coerced
                logging.info(f"Relationship generation attempt succeeded")
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except Exception as e:
                logging.error(
                    f"Relationship generation attempt failed: {e}",
                    exc_info=True,
                )
                continue
        logging.warning(
            "All 3 relationship generation attempts failed for first sentence"
        )
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
                if self._relaxed() and not self._has_concepts_or_properties(
                    object_properties_dict
                ):
                    logging.warning(
                        "Relaxed inference: unparseable concepts/properties response, retrying"
                    )
                    object_properties_dict = None
                    continue
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
                    (
                        object_properties_dict.get("concepts") or []
                        if self._relaxed()
                        else object_properties_dict["concepts"]
                    )[: self._max_new_concepts],
                    (
                        object_properties_dict.get("properties") or []
                        if self._relaxed()
                        else object_properties_dict["properties"]
                    )[: self._max_new_properties],
                    text,
                    self._persona,
                )
                if self._relaxed():
                    coerced = self._relaxed_relationships(new_relationships)
                    if coerced is None:
                        continue
                    return coerced
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []
