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

    def extract_meaning(self, text: str) -> str:
        text = text.replace("_", " ")

        # Likes/enjoys patterns
        if re.search(r"\b(like|likes?)\b", text):
            return "enjoys"

        # |Is" patterns - prevents triplets such as: charlegmagne - is_type_of - king OR biographer - is_kind_of - knowledgeable
        if re.search(r"\b(is|are|was|were|be|being|been)\b", text):
            if re.search(r"kind|type|sort|form|variant|example|instance", text):
                return "is"
            return "is"

        # "Has" patterns
        if re.search(r"\b(has|have|possess|own)\b", text):
            return "has"

        # If no pattern matched, return cleaned version
        return text

    def normalize_edge_label(self, label: str) -> str:
        if not label or not isinstance(label, str):
            return ""
        cleaned = label.strip()
        if not cleaned:
            return ""

        # Remove (edge) tag
        cleaned = re.sub(r"\s*\(edge\)\s*", "", cleaned)
        original_lower = cleaned.lower()

        # Step 1: Apply meaning extraction
        meaning = self.extract_meaning(original_lower)

        # Step 2: Clean up for final output
        if meaning != original_lower:
            cleaned = meaning
        else:
            # Standard cleaning
            cleaned = re.sub(r"[^\w]", "", original_lower.replace(" ", "_"))

        # Remove double underscores
        cleaned = re.sub(r"_+", "_", cleaned)

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
