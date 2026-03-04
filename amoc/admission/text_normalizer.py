from typing import TYPE_CHECKING, Optional, Set
from amoc.nlp.spacy_utils import canonicalize_node_text
from amoc.nlp.spacy_utils import canonicalize_edge_label
from amoc.graph.node import NodeType
from amoc.nlp.spacy_utils import get_concept_lemmas
import re

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node


class TextNormalizer:
    def __init__(
        self,
        spacy_nlp,
        graph_ref: "Graph",
        story_lemmas: Set[str],
        persona_only_lemmas: Set[str],
    ):
        self._spacy_nlp = spacy_nlp
        self._graph = graph_ref
        self._story_lemmas = story_lemmas
        self._persona_only_lemmas = persona_only_lemmas

    def normalize_label(self, label: str) -> str:
        if not label:
            return ""
        return label.lower().strip()

    def normalize_edge_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return label

        cleaned = label.strip()
        if not cleaned:
            return cleaned

        result = canonicalize_edge_label(self._spacy_nlp, cleaned)
        if not result:
            return result
        return result

    def is_valid_relation_label(self, label: str) -> bool:
        # verbs cannot become nodes
        label = (label or "").strip()
        if not label:
            return False

        if not self.is_verb_relation(label):
            return False

        return True

    def is_verb_relation(self, label: str) -> bool:
        if not label:
            return False

        doc = self._spacy_nlp(label)
        has_verb = False
        has_copula = False
        has_adj_after_copula = False
        prev_was_copula = False

        for tok in doc:
            if not tok.is_alpha:
                continue

            pos = tok.pos_
            lemma = tok.lemma_.lower()

            if pos in {"VERB", "AUX"}:
                has_verb = True
                if lemma in {"be", "is", "was", "were", "been", "being", "am", "are"}:
                    has_copula = True
                    prev_was_copula = True
                else:
                    prev_was_copula = False

            elif pos == "ADJ":
                if prev_was_copula or has_copula:
                    has_adj_after_copula = True
                prev_was_copula = False

            elif pos in {"NOUN", "PROPN", "ADP", "PART", "ADV"}:
                prev_was_copula = False

        if has_verb:
            return True
        if has_copula and has_adj_after_copula:
            return True

        return False

    def classify_relation(self, label: str) -> str:
        label = (label or "").strip().lower()
        if not label:
            return "stative"

        doc = self._spacy_nlp(label)

        for tok in doc:
            if not tok.is_alpha:
                continue

            # Copula or auxiliary → attributive
            if tok.lemma_ in {"be", "have"} and tok.pos_ in {"AUX", "VERB"}:
                return "attributive"

            # Main verb → eventive
            if tok.pos_ == "VERB":
                return "eventive"

        return "stative"

    def normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        if not text:
            return None

        doc = self._spacy_nlp(text)
        if not doc:
            return None

        for tok in doc:
            if not tok.is_alpha:
                continue

            pos = tok.pos_
            if is_subject and pos not in {"NOUN", "PROPN", "PRON"}:
                continue
            if not is_subject and pos not in {"NOUN", "PROPN", "PRON"}:
                continue

            lemma = (tok.lemma_ or "").strip().lower()

            if not lemma or lemma in self._spacy_nlp.Defaults.stop_words:
                continue

            return lemma

        return None

    def classify_canonical_node_text(self, canon: str) -> Optional["NodeType"]:

        if not canon:
            return None

        doc = self._spacy_nlp(canon)
        if not doc:
            return None

        token = next((t for t in doc if t.is_alpha), None) or doc[0]

        if token.pos_ in {"NOUN", "PROPN"}:
            return NodeType.CONCEPT
        if token.pos_ == "ADJ":
            return NodeType.PROPERTY

        return None

    def canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional["NodeType"]]:

        canon = canonicalize_node_text(self._spacy_nlp, text)
        return canon, self.classify_canonical_node_text(canon)

    def appears_in_story(self, text: str, *, check_graph: bool = False) -> bool:
        if not text:
            return False

        doc = self._spacy_nlp(text)

        for tok in doc:
            if tok.is_alpha and tok.lemma_.lower() in self._story_lemmas:
                return True

        if check_graph:

            lemmas = get_concept_lemmas(self._spacy_nlp, text)
            if self._graph.get_node(lemmas) is not None:
                return True

        return False

    def canonicalize_edge_direction(
        self, label: str, source_text: str, dest_text: str
    ) -> tuple:

        if not label or not isinstance(label, str):
            return (label, source_text, dest_text, False)

        label_lower = label.strip().lower()

        passive_patterns = [
            (r"^(was|is|were|been|being)\s+(\w+ed)\s+by$", 2),
            (r"^(was|is|were|been|being)\s+(\w+en)\s+by$", 2),
            (r"^(was|is|were|been|being)\s+(\w+)\s+by$", 2),
        ]

        for pattern, verb_group in passive_patterns:
            match = re.match(pattern, label_lower)
            if match:
                verb = match.group(verb_group)
                return (verb, dest_text, source_text, True)

        # No inverse lexical overrides anymore
        return (label, source_text, dest_text, False)

def canonicalize_relation_label(label: str) -> str:
    if not label or not isinstance(label, str):
        return ""

    label = label.strip()
    if not label:
        return ""

    prefixes_to_remove = [
        "nsubj:",
        "dobj:",
        "pobj:",
        "prep:",
        "amod:",
        "advmod:",
        "ROOT:",
        "VERB:",
        "NOUN:",
        "ADJ:",
        "dep:",
        "compound:",
        "agent:",
        "xcomp:",
        "ccomp:",
        "aux:",
        "auxpass:",
    ]
    for prefix in prefixes_to_remove:
        if label.lower().startswith(prefix.lower()):
            label = label[len(prefix) :]

    label = re.sub(r"[^\w\s]+$", "", label)
    label = label.strip()
    label = re.sub(r"\s+", " ", label)

    if len(label) > 0:
        if re.search(r"(.)\1{2,}", label):
            label = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+$", r"\1", label)

        words = label.split()
        cleaned_words = []
        for word in words:
            if len(word) <= 2:
                cleaned_words.append(word.lower())
                continue
            if not re.search(r"[aeiou]", word.lower()):
                continue
            cleaned_words.append(word.lower())

        if not cleaned_words:
            return ""
        label = " ".join(cleaned_words)

    label = label.lower().strip()

    if len(label) < 2:
        return ""

    return label
