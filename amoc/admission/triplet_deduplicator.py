import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional


class TripletDeduplicator:
    def __init__(self, spacy_nlp):
        self.nlp = spacy_nlp
        self.vector_cache = {}

    _VAGUE_PATTERNS = (
        "relat",
        "associat",
        "connect",
        "link",
        "involv",
        "concern",
        "regard",
        "pertain",
        "correspond",
        "tie",
        "bind",
    )

    _VAGUE_EXACT = frozenset(
        {
            "related",
            "relates",
            "relates_to",
            "related_to",
            "associated",
            "associated_with",
            "connected",
            "connected_to",
            "linked",
            "linked_to",
            "involves",
            "concerns",
            "regarding",
            "pertains_to",
            "corresponds_to",
            "tied_to",
            "bound_to",
            "relating_to",
            "associating_with",
        }
    )

    def deduplicate(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        if not triplets:
            return []

        original_count = len(triplets)

        exact_deduped = self.remove_exact_duplicates(triplets)
        negation_filtered = self.remove_negation_relations(exact_deduped)
        vague_filtered = self.remove_vague_relations(negation_filtered)
        semantic_deduped = self.remove_semantic_duplicates(vague_filtered)
        final = self.remove_exact_duplicates(semantic_deduped)

        removed = original_count - len(final)
        if removed > 0:
            logging.info(f"deduplication removed {removed} triplets")

        return final

    def is_vague_relation(self, relation: str) -> bool:
        if not relation:
            return False

        # Normalize for checking
        r_check = relation.lower().replace("_", " ")

        vague_patterns = [
            "relat",
            "associat",
            "connect",
            "link",
            "involv",
            "concern",
            "regard",
            "pertain",
            "correspond",
            "tie",
            "bind",
        ]

        return any(pattern in r_check for pattern in vague_patterns)

    _NEGATION_PHRASES = (
        "not connected",
        "not_connected",
        "not connected to",
        "not_connected_to",
        "not related",
        "not associated",
        "not linked",
        "not involved",
        "not applicable",
        "not present",
        "not exist",
        "no connection",
        "no_connection",
        "no relation",
        "no link",
        "no association",
        "no involvement",
        "without connection",
        "without relation",
        "unconnected",
        "unrelated",
        "disconnected",
    )

    def is_negation_relation(self, relation: str) -> bool:
        if not relation:
            return False

        r_lower = relation.lower().strip().replace("_", " ")

        for phrase in self._NEGATION_PHRASES:
            if phrase in r_lower:
                return True

        return r_lower.startswith(("not ", "no "))

    def remove_negation_relations(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        if not triplets:
            return []

        result = []
        removed_count = 0

        for s, r, o in triplets:
            if self.is_negation_relation(r):
                removed_count += 1
                logging.debug(f"removed negation relation: ({s}, {r}, {o})")
            else:
                result.append((s, r, o))

        if removed_count > 0:
            logging.info(f"removed {removed_count} negation triplets")

        return result

    def remove_vague_relations(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        if not triplets:
            return []

        # Core vague patterns - these capture all variations
        vague_patterns = [
            "relat",  # related, relates, relating, relationship
            "associat",  # associated, associates, association
            "connect",  # connected, connects, connection
            "involv",  # involves, involved, involvemen
        ]

        result = []
        removed_count = 0

        for s, r, o in triplets:
            # Convert to lowercase and replace underscores with spaces for checking
            r_check = r.lower().replace("_", " ")

            # Check if any vague pattern appears
            is_vague = any(pattern in r_check for pattern in vague_patterns)

            if is_vague:
                removed_count += 1
                logging.debug(f"Removed vague relation: ({s}, {r}, {o})")
            else:
                result.append((s, r, o))

        if removed_count > 0:
            logging.info(f"Removed {removed_count} triplets with vague relations")

        return result

    def remove_exact_duplicates(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        seen = set()
        result = []

        for s, r, o in triplets:
            key = (s.lower().strip(), r.lower().strip(), o.lower().strip())
            if key not in seen:
                seen.add(key)
                result.append((s, r, o))

        return result

    def remove_semantic_duplicates(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        if len(triplets) < 2:
            return triplets

        keep = [True] * len(triplets)
        subject_groups = self.group_by_subject(triplets)

        for subject, group_indices in subject_groups.items():
            self.process_subject_group(triplets, group_indices, keep)

        return [triplets[i] for i in range(len(triplets)) if keep[i]]

    def group_by_subject(
        self, triplets: List[Tuple[str, str, str]]
    ) -> Dict[str, List[int]]:
        groups = {}
        for i, (s, r, o) in enumerate(triplets):
            s_norm = s.lower().strip()
            groups.setdefault(s_norm, []).append(i)
        return groups

    def process_subject_group(
        self, triplets: List[Tuple[str, str, str]], indices: List[int], keep: List[bool]
    ) -> None:
        if len(indices) < 2:
            return

        parsed_data = []
        for idx in indices:
            if not keep[idx]:
                continue
            data = self.parse_triplet(triplets[idx], idx)
            if data:
                parsed_data.append(data)

        for i, data1 in enumerate(parsed_data):
            if not keep[data1["idx"]]:
                continue
            for j, data2 in enumerate(parsed_data[i + 1 :], i + 1):
                if not keep[data2["idx"]]:
                    continue
                self.check_adj_noun_pair(data1, data2, keep)

    def parse_triplet(
        self, triplet: Tuple[str, str, str], idx: int
    ) -> Optional[Dict[str, Any]]:
        s, r, o = triplet
        doc = self.nlp(o)
        if not doc or len(doc) == 0:
            return None

        token = doc[0]

        return {
            "idx": idx,
            "subject": s.lower().strip(),
            "relation": r.lower().strip(),
            "object": o,
            "object_lower": o.lower().strip(),
            "pos": token.pos_,
            "lemma": token.lemma_.lower(),
            "vector": token.vector if token.has_vector else None,
            "morph": str(token.morph) if token.morph else "",
            "text": token.text,
        }

    def check_adj_noun_pair(
        self, data1: Dict[str, Any], data2: Dict[str, Any], keep: List[bool]
    ) -> None:
        is_adj_noun = (data1["pos"] == "ADJ" and data2["pos"] == "NOUN") or (
            data1["pos"] == "NOUN" and data2["pos"] == "ADJ"
        )

        if not is_adj_noun:
            return

        adj_data = data1 if data1["pos"] == "ADJ" else data2
        noun_data = data2 if data1["pos"] == "ADJ" else data1

        if self.are_semantically_related(adj_data, noun_data):
            if adj_data["relation"] == "is" and noun_data["relation"] == "has":
                keep[noun_data["idx"]] = False
                logging.debug(
                    f"removed duplicate ({noun_data['subject']}, {noun_data['relation']}, {noun_data['object']})"
                )
            elif adj_data["relation"] == "has" and noun_data["relation"] == "is":
                keep[adj_data["idx"]] = False
                logging.debug(
                    f"removed duplicate ({adj_data['subject']}, {adj_data['relation']}, {adj_data['object']})"
                )

    def are_semantically_related(
        self, adj_data: Dict[str, Any], noun_data: Dict[str, Any]
    ) -> bool:
        # Strategy 1: Vector similarity
        if adj_data["vector"] is not None and noun_data["vector"] is not None:
            sim = self.vector_similarity(adj_data["vector"], noun_data["vector"])
            if sim > 0.7:
                return True

        # Strategy 2: Lemma relationship
        if self.check_lemma_relationship(adj_data["lemma"], noun_data["lemma"]):
            return True

        # Strategy 3: Morphological patterns
        if self.check_morphological_pattern(adj_data, noun_data):
            return True

        # Strategy 4: String containment
        if self.check_string_relationship(
            adj_data["object_lower"], noun_data["object_lower"]
        ):
            return True

        return False

    def vector_similarity(self, vec1, vec2) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def check_lemma_relationship(self, lemma1: str, lemma2: str) -> bool:
        if lemma1 in lemma2 or lemma2 in lemma1:
            if abs(len(lemma1) - len(lemma2)) <= 4:
                return True

        transformations = [
            (lambda x: x + "ed", lambda x: x),
            (lambda x: x + "ing", lambda x: x),
            (lambda x: x + "tion", lambda x: x),
            (lambda x: x + "ity", lambda x: x),
            (lambda x: x + "ness", lambda x: x),
            (lambda x: x[:-2] + "y", lambda x: x),
        ]

        for trans, _ in transformations:
            if trans(lemma1) == lemma2 or trans(lemma2) == lemma1:
                return True

        return False

    def check_morphological_pattern(
        self, adj_data: Dict[str, Any], noun_data: Dict[str, Any]
    ) -> bool:
        adj_suffixes = ["ed", "ing", "al", "ive", "ous", "ful", "ic", "ish", "y"]
        noun_suffixes = [
            "tion",
            "ity",
            "ness",
            "ence",
            "ance",
            "dom",
            "hood",
            "ship",
            "th",
        ]

        adj_text = adj_data["object_lower"]
        noun_text = noun_data["object_lower"]

        for suffix in noun_suffixes:
            if noun_text.endswith(suffix):
                stem = noun_text[: -len(suffix)]
                for adj_suffix in adj_suffixes:
                    if adj_text == stem + adj_suffix:
                        return True
                    if adj_text == stem + adj_suffix + "e":
                        return True

        for suffix in adj_suffixes:
            if adj_text.endswith(suffix):
                stem = adj_text[: -len(suffix)]
                for noun_suffix in noun_suffixes:
                    if noun_text == stem + noun_suffix:
                        return True
                    if noun_text == stem + noun_suffix + "e":
                        return True

        return False

    def check_string_relationship(self, str1: str, str2: str) -> bool:
        if str1 in str2 or str2 in str1:
            if abs(len(str1) - len(str2)) <= 4:
                return True

        if abs(len(str1) - len(str2)) <= 2:
            if self.levenshtein_distance(str1, str2) <= 2:
                return True

        return False

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def get_word_vector(self, word: str):
        if word in self.vector_cache:
            return self.vector_cache[word]

        doc = self.nlp(word)
        if doc and len(doc) == 1 and doc[0].has_vector:
            self.vector_cache[word] = doc[0].vector
            return doc[0].vector

        self.vector_cache[word] = None
        return None
