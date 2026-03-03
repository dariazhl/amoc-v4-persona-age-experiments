from typing import TYPE_CHECKING, Optional, List, Set, Callable
from amoc.graph.node import NodeType, NodeSource, NodeProvenance
from amoc.nlp.spacy_utils import canonicalize_node_text, extract_adjectival_modifiers

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
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
        self._canonicalize_node_text = lambda text: canonicalize_node_text(
            self._spacy_nlp, text
        )

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

        def get_node(token):
            return self._graph.get_node([token.lemma_.lower()])

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

        for token in sent:
            # knight (nsubj) ← is (be) → brave (acomp)
            # knight - is - brave
            if token.dep_ in {"acomp", "attr"} and (
                token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.tag_ == "VBN")
            ):
                if token.head.lemma_ != "be":
                    continue

                subj = next(
                    (
                        c
                        for c in token.head.children
                        if c.dep_ in {"nsubj", "nsubjpass"}
                    ),
                    None,
                )
                if not subj:
                    continue

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
                    assert_edge(subj_node, prop_node, "is")

            # Adjectival modifier: dark (amod) → forest (NOUN)
            # forest - is - dark
            if token.dep_ == "amod" and token.pos_ == "ADJ":
                head_node = get_node(token.head)
                prop_node = get_node(token)

                if head_node and prop_node:
                    assert_edge(head_node, prop_node, "is")

            # Verb SVO
            # knight (nsubj) → kills (VERB) → dragon (dobj)
            # knight - kill - dragon
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

                # Direct objects
                for obj in (c for c in token.children if c.dep_ in {"dobj", "attr"}):
                    obj_node = get_node(obj)
                    if obj_node:
                        assert_edge(subj_node, obj_node, token.lemma_)

                    for conj in (c for c in obj.children if c.dep_ == "conj"):
                        conj_node = get_node(conj)
                        if conj_node:
                            assert_edge(subj_node, conj_node, token.lemma_)

                # Prepositional objects
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    pobj = next(
                        (c for c in prep.children if c.dep_ == "pobj"),
                        None,
                    )
                    if not pobj:
                        continue

                    obj_node = get_node(pobj)
                    if not obj_node:
                        continue

                    label = f"{token.lemma_}_{prep.lemma_}"
                    assert_edge(subj_node, obj_node, label)

                    for conj in (c for c in pobj.children if c.dep_ == "conj"):
                        conj_node = get_node(conj)
                        if conj_node:
                            assert_edge(subj_node, conj_node, label)

            # ROOT copular with prep
            # knight -- ride_through --> forest
            if token.dep_ == "ROOT" and token.lemma_ == "be":
                subj = next(
                    (c for c in token.children if c.dep_ in {"nsubj", "nsubjpass"}),
                    None,
                )
                if not subj:
                    continue

                subj_node = get_node(subj)
                if not subj_node:
                    continue

                # Handle prepositional phrases attached to "be" ROOT
                # book (nsubj) ← is (ROOT, be) → on (prep) → table (pobj)
                # book --> is_on --> table
                for prep in (c for c in token.children if c.dep_ == "prep"):
                    pobj = next(
                        (c for c in prep.children if c.dep_ == "pobj"),
                        None,
                    )
                    if not pobj:
                        continue

                    obj_node = get_node(pobj)
                    if not obj_node:
                        continue

                    label = f"is_{prep.lemma_}"
                    assert_edge(subj_node, obj_node, label)

                    for conj in (c for c in pobj.children if c.dep_ == "conj"):
                        conj_node = get_node(conj)
                        if conj_node:
                            assert_edge(subj_node, conj_node, label)
