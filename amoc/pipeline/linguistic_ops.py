from typing import TYPE_CHECKING, Optional, List, Dict, Set, Callable, Tuple
import re
import logging
from amoc.nlp.spacy_utils import extract_prepositional_objects as extract_prep
from amoc.graph.node import NodeType, NodeSource, NodeProvenance

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node, NodeType, NodeSource, NodeProvenance
    from spacy.tokens import Span


class LinguisticOps:
    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        client_ref,
        story_lemmas: Set[str],
        story_text: str = "",
        persona: str = "",
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._client = client_ref
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
        resolved = self._client.resolve_pronouns(text, self._persona)
        if not isinstance(resolved, str) or not resolved.strip():
            return text
        low = resolved.lower()
        if "does not mention any pronouns" in low or "no pronouns to replace" in low:
            return text
        return resolved

    def extract_adjectival_modifiers(self, sent: "Span") -> Dict[str, List[str]]:
        result: Dict[str, List[str]] = {}

        for token in sent:
            if token.pos_ == "ADJ" and token.dep_ in {"amod", "acomp", "attr"}:
                head = token.head
                if head.pos_ in {"NOUN", "PROPN"}:
                    head_lemma = head.lemma_.lower()
                    adj_lemma = token.lemma_.lower()

                    if head_lemma not in result:
                        result[head_lemma] = []

                    if adj_lemma not in result[head_lemma]:
                        result[head_lemma].append(adj_lemma)

        return result

    def append_adjectival_hints(
        self,
        nodes_from_text: str,
        sent: "Span",
    ) -> str:
        adj_mods = self.extract_adjectival_modifiers(sent)

        for head, adjs in adj_mods.items():
            for adj in adjs:
                nodes_from_text += f" - ({adj}, PROPERTY) [describes {head}]\n"

        return nodes_from_text

    def get_content_words_from_sentence(
        self,
        sent: "Span",
        pos_filter: Optional[Set[str]] = None,
    ) -> List[str]:
        if pos_filter is None:
            pos_filter = {"NOUN", "PROPN", "VERB", "ADJ"}

        words = []
        for token in sent:
            if token.pos_ in pos_filter and not token.is_stop:
                words.append(token.lemma_.lower())

        return words

    def extract_prepositional_objects(
        self,
        sent: "Span",
    ) -> List[tuple]:
        return extract_prep(sent)

    def extract_deterministic_structure(
        sentence_nodes: List["Node"],
        sentence_words: List[str],
    ) -> None:
        if not self._add_edge_fn or not self._classify_relation_fn:
            return

        def get_node(token):
            return self._graph.get_node([token.lemma_.lower()])

        for token in sent:
            if token.dep_ in {"acomp", "attr", "ROOT"} and (
                token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.tag_ == "VBN")
            ):
                subj = None

                if token.head.lemma_ == "be":
                    subj = next(
                        (
                            c
                            for c in token.head.children
                            if c.dep_ in {"nsubj", "nsubjpass"}
                        ),
                        None,
                    )

                if token.dep_ == "ROOT":
                    subj = next(
                        (c for c in token.children if c.dep_ == "nsubjpass"),
                        None,
                    )

                if subj:
                    subj_node = get_node(subj)

                    prop_node = self._graph.add_or_get_node(
                        [token.lemma_.lower()],
                        token.lemma_.lower(),
                        NodeType.PROPERTY,
                        NodeSource.TEXT_BASED,
                        provenance=NodeProvenance.STORY_TEXT,
                        origin_sentence=self._current_sentence_index,
                    )

                    if subj_node and prop_node:
                        edge = self._add_edge_fn(
                            subj_node,
                            prop_node,
                            label="is",
                            edge_forget=self._edge_visibility,
                            created_at_sentence=self._current_sentence_index,
                        )
                        if edge:
                            edge.mark_as_asserted(reset_score=True)

                    for prep in (c for c in token.children if c.dep_ == "prep"):
                        pobj = next(
                            (c for c in prep.children if c.dep_ == "pobj"),
                            None,
                        )
                        if not pobj:
                            continue

                        obj_node = get_node(pobj)
                        if not obj_node or not subj_node:
                            continue

                        label = f"{token.lemma_}_{prep.lemma_}"

                        edge = self._add_edge_fn(
                            subj_node,
                            obj_node,
                            label=label,
                            edge_forget=self._edge_visibility,
                            created_at_sentence=self._current_sentence_index,
                        )
                        if edge:
                            edge.mark_as_asserted(reset_score=True)

            if token.dep_ == "amod" and token.pos_ == "ADJ":
                head_node = get_node(token.head)
                prop_node = get_node(token)

                if head_node and prop_node:
                    edge = self._add_edge_fn(
                        head_node,
                        prop_node,
                        label="is",
                        edge_forget=self._edge_visibility,
                        created_at_sentence=self._current_sentence_index,
                    )
                    if edge:
                        edge.mark_as_asserted(reset_score=True)

            if token.pos_ == "VERB" and token.lemma_ != "be":
                subj = next(
                    (c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}),
                    None,
                )

                if not subj:
                    continue

                subj_node = get_node(subj)
                if not subj_node:
                    continue

                for obj in (c for c in token.children if c.dep_ in {"dobj", "attr"}):

                    obj_node = get_node(obj)

                    if obj_node:
                        edge = self._add_edge_fn(
                            subj_node,
                            obj_node,
                            label=token.lemma_,
                            edge_forget=self._edge_visibility,
                            created_at_sentence=self._current_sentence_index,
                        )
                        if edge:
                            edge.mark_as_asserted(reset_score=True)

                    for conj in (c for c in obj.children if c.dep_ == "conj"):
                        conj_node = get_node(conj)
                        if conj_node:
                            edge2 = self._add_edge_fn(
                                subj_node,
                                conj_node,
                                label=token.lemma_,
                                edge_forget=self._edge_visibility,
                                created_at_sentence=self._current_sentence_index,
                            )
                            if edge2:
                                edge2.mark_as_asserted(reset_score=True)

                for prep in (c for c in token.children if c.dep_ == "prep"):
                    pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
                    if not pobj:
                        continue

                    if not obj_node:
                        continue

                    label = f"{token.lemma_}_{prep.lemma_}"

                    edge = self._add_edge_fn(
                        subj_node,
                        obj_node,
                        label=label,
                        edge_forget=self._edge_visibility,
                        created_at_sentence=self._current_sentence_index,
                    )
                    if edge:
                        edge.mark_as_asserted(reset_score=True)

                    for conj in (c for c in pobj.children if c.dep_ == "conj"):
                        conj_node = get_node(conj)
                        if conj_node:
                            edge2 = self._add_edge_fn(
                                subj_node,
                                conj_node,
                                label=label,
                                edge_forget=self._edge_visibility,
                                created_at_sentence=self._current_sentence_index,
                            )
                            if edge2:
                                edge2.mark_as_asserted(reset_score=True)
