from typing import TYPE_CHECKING, Optional, Set
from amoc.utils.spacy_utils import canonicalize_node_text
from amoc.utils.spacy_utils import canonicalize_edge_label
from amoc.core.node import NodeType
from amoc.utils.spacy_utils import get_concept_lemmas
import re

if TYPE_CHECKING:
    from amoc.core.graph import Graph


class TextNormalizer:
    def __init__(
        self,
        spacy_nlp,
        graph_ref: "Graph",
        story_lemmas: Set[str],
    ):
        self._spacy_nlp = spacy_nlp
        self._graph = graph_ref
        self._story_lemmas = story_lemmas

    def normalize_edge_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return ""
        cleaned = label.strip()
        if not cleaned:
            return ""
        # Basic cleanup: remove "(edge)" if present, lowercase, replace spaces with underscores
        cleaned = re.sub(r"\s*\(edge\)\s*", "", cleaned)
        cleaned = cleaned.lower().replace(" ", "_")
        # Remove any non‑alphanumeric characters except underscore
        cleaned = re.sub(r"[^\w]", "", cleaned)
        return cleaned

    # verbs cannot become nodes
    def is_valid_relation_label(self, label: str) -> bool:
        label = (label or "").strip()
        if not label:
            return False
        # Reject if label consists only of punctuation or numbers
        if not re.search(r"[a-zA-Z]", label):
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

    def extract_canonical_node_lemma(
        self, text: str, is_subject: bool
    ) -> Optional[str]:
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
            # allow adjectives such that inferred nodes can pop up ie. knight - is - brave
            if not is_subject and pos not in {"NOUN", "PROPN", "PRON", "ADJ"}:
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

    def normalize_and_classify_node(
        self, text: str
    ) -> tuple[str, Optional["NodeType"]]:

        canon = canonicalize_node_text(self._spacy_nlp, text)
        return canon, self.classify_canonical_node_text(canon)

    def is_grounded_in_story(self, text: str, *, check_graph: bool = False) -> bool:
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

    @classmethod
    def clean_label(cls, label: str) -> str:
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
