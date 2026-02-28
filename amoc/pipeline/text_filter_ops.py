from typing import TYPE_CHECKING, Optional, Set
import re


if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node, NodeType

class TextFilterOps:

    BLACKLISTED_RELATIONS: Set[str] = {
        "has",
        "have",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
    }

    GENERIC_RELATIONS: Set[str] = {
        "relates_to",
        "is_related_to",
        "associated_with",
        "connected_to",
        "involves",
        "concerns",
        "pertains_to",
    }

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
        from amoc.nlp.spacy_utils import canonicalize_edge_label

        if not label or not isinstance(label, str):
            return label

        label = label.strip()
        if not label:
            return label

        result = canonicalize_edge_label(self._spacy_nlp, label)

        if len(result) > 0:
            if re.search(r"(.)\1{2,}", result):
                return ""
            words = result.split()
            for word in words:
                if len(word) > 3 and not re.search(r"[aeiou]", word):
                    return ""

        return result

    def is_valid_relation_label(self, label: str) -> bool:
        if not label or not isinstance(label, str):
            return False

        label_stripped = label.strip()
        if not label_stripped:
            return False

        if self.is_generic_relation(label_stripped):
            return False

        if self.is_blacklisted_relation(label_stripped):
            return False

        if not self.is_verb_relation(label_stripped):
            return False

        return True

    def is_generic_relation(self, label: str) -> bool:
        return self.normalize_label(label) in self.GENERIC_RELATIONS

    def is_blacklisted_relation(self, label: str) -> bool:
        return self.normalize_label(label) in self.BLACKLISTED_RELATIONS

    def is_verb_relation(self, label: str) -> bool:
        if not label:
            return False

        doc = self._spacy_nlp(label)
        has_verb = False
        has_copula = False
        has_adj_after_copula = False
        prev_was_copula = False

        for tok in doc:
            if not getattr(tok, "is_alpha", False):
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
        label = label.lower()

        EVENTIVE_VERBS = {
            "attack",
            "kill",
            "destroy",
            "build",
            "ride",
            "run",
            "eat",
            "strike",
            "burn",
            "move",
        }

        if label in {"is", "has", "belongs_to"}:
            return "attributive"

        if label in EVENTIVE_VERBS:
            return "eventive"

        return "stative"

    def normalize_endpoint_text(
        self, text: str, is_subject: bool
    ) -> Optional[str]:
        META_LEMMAS = {
            "subject", "object", "entity", "concept",
            "property", "someone", "something",
        }

        if not text:
            return None

        doc = self._spacy_nlp(text)
        if not doc:
            return None

        for tok in doc:
            if not getattr(tok, "is_alpha", False):
                continue

            pos = tok.pos_
            if is_subject and pos not in {"NOUN", "PROPN", "PRON"}:
                continue
            if not is_subject and pos not in {"NOUN", "PROPN", "PRON"}:
                continue

            lemma = (getattr(tok, "lemma_", "") or "").strip().lower()

            if lemma in META_LEMMAS:
                return None

            if not lemma or lemma in self._spacy_nlp.Defaults.stop_words:
                continue

            return lemma

        return None

    def classify_canonical_node_text(self, canon: str) -> Optional["NodeType"]:
        from amoc.graph.node import NodeType

        if not canon:
            return None

        doc = self._spacy_nlp(canon)
        if not doc:
            return None

        token = next((t for t in doc if getattr(t, "is_alpha", False)), None) or doc[0]

        if token.pos_ in {"NOUN", "PROPN"}:
            return NodeType.CONCEPT
        if token.pos_ == "ADJ":
            return NodeType.PROPERTY

        return None

    def canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional["NodeType"]]:
        from amoc.nlp.spacy_utils import canonicalize_node_text

        META_LEMMAS = {"subject", "object", "entity", "concept", "property"}

        canon = canonicalize_node_text(self._spacy_nlp, text)
        if canon in META_LEMMAS:
            return canon, None

        return canon, self.classify_canonical_node_text(canon)

    def appears_in_story(self, text: str, *, check_graph: bool = False) -> bool:
        if not text:
            return False

        doc = self._spacy_nlp(text)

        for tok in doc:
            if tok.is_alpha and tok.lemma_.lower() in self._story_lemmas:
                return True

        if check_graph:
            from amoc.nlp.spacy_utils import get_concept_lemmas

            lemmas = get_concept_lemmas(self._spacy_nlp, text)
            if self._graph.get_node(lemmas) is not None:
                return True

        return False

    def canonicalize_edge_direction(
        self, label: str, source_text: str, dest_text: str
    ) -> tuple:
        import logging

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
                logging.debug(
                    "[EdgeDirection] Passive detected: %r → active: %r (swapped)",
                    label_lower,
                    verb,
                )
                return (verb, dest_text, source_text, True)

        inverse_mappings = {
            "is threatened by": ("threatens", True),
            "was threatened by": ("threatens", True),
            "is loved by": ("loves", True),
            "was loved by": ("loves", True),
            "is hated by": ("hates", True),
            "was hated by": ("hates", True),
            "is owned by": ("owns", True),
            "was owned by": ("owns", True),
            "belongs to": ("owns", True),
        }

        if label_lower in inverse_mappings:
            active_label, should_swap = inverse_mappings[label_lower]
            if should_swap:
                logging.debug(
                    "[EdgeDirection] Inverse relation: %r → %r (swapped)",
                    label_lower,
                    active_label,
                )
                return (active_label, dest_text, source_text, True)
            return (active_label, source_text, dest_text, False)

        return (label, source_text, dest_text, False)

    # ==========================================================================
    # RELATION LABEL CANONICALIZATION (moved from Graph)
    # ==========================================================================

    @staticmethod
    def canonicalize_relation_label(label: str) -> str:
        """
        Canonicalize relation labels before edge creation.

        Removes dependency prefixes, trailing punctuation, repeated characters,
        and words without vowels (likely garbage).

        Args:
            label: Raw relation label

        Returns:
            Canonicalized label, or empty string if invalid
        """
        if not label or not isinstance(label, str):
            return ""

        label = label.strip()
        if not label:
            return ""

        prefixes_to_remove = [
            "nsubj:", "dobj:", "pobj:", "prep:", "amod:", "advmod:",
            "ROOT:", "VERB:", "NOUN:", "ADJ:", "dep:", "compound:",
            "agent:", "xcomp:", "ccomp:", "aux:", "auxpass:",
        ]
        for prefix in prefixes_to_remove:
            if label.lower().startswith(prefix.lower()):
                label = label[len(prefix):]

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
