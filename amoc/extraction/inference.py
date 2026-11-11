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

    def _text_lemmas(self, sent_span, text_words) -> set:
        lemmas = {w.lower() for w in text_words if w}
        lemmas |= {tok.lemma_.lower() for tok in sent_span if tok.is_alpha}
        lemmas |= {tok.text.lower() for tok in sent_span if tok.is_alpha}
        for node in getattr(self._graph, "nodes", ()):
            if getattr(node.node_source, "name", "") == "TEXT_BASED":
                lemmas.update(node.lemmas)
        return lemmas

    def _is_text_item(self, item, text_lemmas) -> bool:
        if not isinstance(item, str) or not item.strip():
            return False
        doc = self._spacy_nlp(item.strip())
        words = [tok for tok in doc if tok.is_alpha and not tok.is_stop]
        if not words:
            return item.strip().lower() in text_lemmas
        def in_text(tok):
            forms = {tok.lemma_.lower(), tok.text.lower()}
            forms |= {f[:-1] for f in list(forms) if f.endswith("s")}
            return bool(forms & text_lemmas)

        return all(in_text(tok) for tok in words)

    def _select_new_items(self, result, sent_span, text_words):
        if not self._relaxed():
            return (
                result["concepts"][: self._max_new_concepts],
                result["properties"][: self._max_new_properties],
            )
        concepts = list(result.get("concepts") or []) if isinstance(result, dict) else []
        properties = list(result.get("properties") or []) if isinstance(result, dict) else []
        text_lemmas = self._text_lemmas(sent_span, text_words)
        new_concepts = [c for c in concepts if not self._is_text_item(c, text_lemmas)]
        new_properties = [p for p in properties if not self._is_text_item(p, text_lemmas)]
        logging.info(
            f"INFERENCE_PROPOSALS: concepts={concepts} properties={properties} "
            f"kept_concepts={new_concepts[: self._max_new_concepts]} "
            f"kept_properties={new_properties[: self._max_new_properties]}"
        )
        return new_concepts[: self._max_new_concepts], new_properties[: self._max_new_properties]

    def _nudge_if_no_new_items(self, result, sent_span, text_words, ask):
        if not self._relaxed():
            return result
        current = result if isinstance(result, dict) else {}
        concepts, properties = self._select_new_items(current, sent_span, text_words)
        if concepts or properties:
            return result
        proposed = list(current.get("concepts") or []) + list(current.get("properties") or [])
        previous = ", ".join(str(x) for x in proposed) or "empty"
        try:
            nudged = ask(previous)
        except Exception as e:
            logging.error(f"INFERENCE_NUDGE failed: {e}", exc_info=True)
            return result
        logging.info(f"INFERENCE_NUDGE: previous=[{previous}] response={nudged}")
        return nudged if isinstance(nudged, dict) else result

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

        object_properties_dict = self._nudge_if_no_new_items(
            object_properties_dict,
            sent,
            current_sentence_text_based_words,
            lambda previous: self._llm.infer_objects_and_properties_first_sentence(
                nodes_from_text, sent.text, self._persona, nudge_previous=previous
            ),
        )

        for _ in range(3):
            try:
                concepts, properties = self._select_new_items(
                    object_properties_dict, sent, current_sentence_text_based_words
                )
                new_relationships = (
                    self._llm.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        concepts,
                        properties,
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
                if self._relaxed() and not isinstance(object_properties_dict, dict):
                    logging.warning(
                        "Relaxed inference: unparseable concepts/properties response, retrying"
                    )
                    object_properties_dict = None
                    continue
                break
            except:
                continue

        object_properties_dict = self._nudge_if_no_new_items(
            object_properties_dict,
            sent_span,
            current_sentence_text_based_words,
            lambda previous: self._llm.infer_objects_and_properties(
                nodes_from_text,
                graph_nodes_representation,
                graph_edges_representation,
                text,
                self._persona,
                nudge_previous=previous,
            ),
        )

        if object_properties_dict is None:
            return [], []

        for _ in range(3):
            try:
                concepts, properties = self._select_new_items(
                    object_properties_dict, sent_span, current_sentence_text_based_words
                )
                new_relationships = self._llm.generate_new_inferred_relationships(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    concepts,
                    properties,
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
