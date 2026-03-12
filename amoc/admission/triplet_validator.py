from typing import Dict, List, Optional, Tuple, Any
import logging
from difflib import SequenceMatcher

from amoc.core.node import Node, NodeSource
from amoc.core.edge import Edge
from amoc.extraction.linguistic import LinguisticProcessing
from amoc.admission.text_normalizer import TextNormalizer
from amoc.llm.vllm_client import VLLMClient
from amoc.core.edge import _maybe_embed
from amoc.admission.triplet_deduplicator import TripletDeduplicator


class TripletValidator:
    def __init__(
        self,
        linguistic_ops: LinguisticProcessing,
        extract_deterministic_fn,
        text_normalizer: TextNormalizer,
        client: VLLMClient,
        persona: str = "",
        similarity_threshold: float = 0.8,
        spacy_nlp=None,
    ):
        self.linguistic_ops = linguistic_ops
        self.extract_deterministic_fn = extract_deterministic_fn
        self.text_normalizer = text_normalizer
        self.client = client
        self.persona = persona
        self.similarity_threshold = similarity_threshold
        self._embedding_cache = {}
        self.spacy_nlp = spacy_nlp or (
            linguistic_ops._spacy_nlp if linguistic_ops else None
        )
        self.deduplicator = TripletDeduplicator(self.spacy_nlp)

    def deduplicate_triplets(
        self, triplets: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        return self.deduplicator.deduplicate(triplets)

    def normalize_llm_triplets(
        self, new_relationships: List[Any]
    ) -> List[Tuple[str, str, str]]:
        normalized = []
        for rel in new_relationships or []:
            triple = self.normalize_single_triplet(rel)
            if triple:
                normalized.append(triple)
        return normalized

    def normalize_single_triplet(
        self, relationship: Any
    ) -> Optional[Tuple[str, str, str]]:
        if not relationship:
            return None

        if isinstance(relationship, dict):
            subj = relationship.get("subject") or relationship.get("head")
            rel = relationship.get("relation") or relationship.get("predicate")
            obj = relationship.get("object") or relationship.get("tail")
            if subj and rel and obj:
                return (str(subj), str(rel), str(obj))
            return None

        if isinstance(relationship, (list, tuple)):
            if len(relationship) == 3:
                return tuple(relationship)
            if len(relationship) == 4:
                subj, rel, _, obj = relationship
                return (subj, rel, obj)

        return None

    # extract deterministic candidates and build lookup dictionaries for validation
    def build_deterministic_lookup(
        self, sentence_nodes: List[Node], sentence_words: List[str], current_sentence
    ) -> Dict[str, Dict]:
        candidates = (
            self.extract_deterministic_fn(
                current_sentence,
                sentence_nodes,
                sentence_words,
            )
            or []
        )

        forward = {}
        reverse = {}

        for cand in candidates:
            key_forward = (cand.subject_lemma, cand.object_lemma)
            forward[key_forward] = cand.relation_label
            key_reverse = (cand.object_lemma, cand.subject_lemma)
            reverse[key_reverse] = cand.relation_label

        return {"forward": forward, "reverse": reverse}

    # check if an llm triple conflicts with deterministic extraction
    def validate_against_deterministic(
        self, subj: str, rel: str, obj: str, det_lookup: Dict, sentence: str = None
    ) -> Dict:
        forward = det_lookup["forward"]
        reverse = det_lookup["reverse"]

        # If we have deterministic info, use it strictly
        if forward or reverse:
            forward_key = (subj, obj)
            reverse_key = (obj, subj)

            # case 1: exact match exists
            if forward_key in forward:
                det_rel = forward[forward_key]
                if self.labels_are_similar(rel, det_rel):
                    return {"valid": True, "subj": subj, "obj": obj, "rel": rel}
                else:
                    logging.info(
                        f"llm edge ({subj},{rel},{obj}) conflicts with deterministic ({det_rel})"
                    )
                    return {"valid": False}

            # case 2: reverse match exists (likely swapped)
            if reverse_key in reverse:
                det_rel = reverse[reverse_key]
                logging.info(
                    f"llm edge ({subj},{rel},{obj}) appears reversed - swapping to ({obj},{rel},{subj})"
                )
                if self.labels_are_similar(rel, det_rel):
                    return {"valid": True, "subj": obj, "obj": subj, "rel": rel}
                else:
                    logging.info(
                        f"swapped edge relation ({rel}) conflicts with deterministic ({det_rel})"
                    )
                    return {"valid": False}

            # case 3: deterministic info exists but no match
            logging.info(
                f"llm edge ({subj},{rel},{obj}) has no deterministic match - rejecting"
            )
            return {"valid": False}

        # case 4: NO deterministic info – apply simple heuristics
        # "thing" should never be the subject of an action verb
        if subj.lower() == "thing" and rel.lower() in [
            "writes",
            "knows",
            "describes",
            "says",
            "tells",
            "wrote",
            "knew",
            "described",
            "said",
            "told",
        ]:
            logging.info(f"rejecting - 'thing' cannot be the subject of '{rel}'")
            return {"valid": False}

        # If object is "thing", subject should be a person/entity
        if obj.lower() == "thing" and subj.lower() in ["thing", "it", "this"]:
            logging.info(f"rejecting - vague subject '{subj}' with 'thing' as object")
            return {"valid": False}

        # No deterministic info and heuristics passed – let LLM decide
        logging.info(f"no deterministic info for ({subj},{rel},{obj}) - passing to LLM")
        return {"valid": True, "subj": subj, "obj": obj, "rel": rel}

    # use llm to validate semantic plausibility of a triple
    # TROUBLE - remove the story_context param
    def validate_with_llm(
        self, subj: str, rel: str, obj: str, sentence: str, story_context: str = ""
    ) -> Dict:
        llm_validation = self.client.validate_triplet(
            sentence=sentence,
            subject=subj,
            relation=rel,
            object=obj,
            persona=self.persona,
        )

        if not llm_validation.get("valid", False):
            logging.info(
                f"llm rejected triple ({subj},{rel},{obj}): {llm_validation.get('reason', '')}"
            )

        return llm_validation

    def labels_are_similar(self, label1: str, label2: str) -> bool:
        if not label1 or not label2:
            return False

        # exact match
        if label1.strip().lower() == label2.strip().lower():
            return True

        # string similarity
        if (
            SequenceMatcher(None, label1.lower(), label2.lower()).ratio()
            >= self.similarity_threshold
        ):
            return True

        # embedding similarity (if available)
        emb1 = self.get_label_embedding(label1)
        emb2 = self.get_label_embedding(label2)
        if emb1 is not None and emb2 is not None:
            import numpy as np

            cos = float(np.dot(emb1, emb2))
            if cos >= self.similarity_threshold:
                return True

        return False

    def get_label_embedding(self, label: str):
        if label not in self._embedding_cache:
            self._embedding_cache[label] = _maybe_embed(label)

        return self._embedding_cache[label]

    def normalize_endpoints(
        self, subj: str, obj: str
    ) -> Tuple[Optional[str], Optional[str]]:
        subj_norm = self.text_normalizer.extract_canonical_node_lemma(
            subj, is_subject=True
        )
        obj_norm = self.text_normalizer.extract_canonical_node_lemma(
            obj, is_subject=False
        )
        return subj_norm, obj_norm

    def clean_and_validate_relation(self, rel: str) -> Optional[str]:
        edge_label = self.text_normalizer.normalize_edge_label(
            rel.replace("(edge)", "").strip()
        )
        if not self.text_normalizer.is_valid_relation_label(edge_label):
            return None
        return edge_label

    def prioritize_hub(
        self, triplets: List[Tuple[str, str, str]], explicit_nodes: List[str]
    ) -> List[Tuple[str, str, str]]:
        if not triplets or not explicit_nodes:
            return triplets

        # Find the most specific entity (usually a person name)
        generic_terms = {
            "thing",
            "something",
            "it",
            "this",
            "that",
            "they",
            "he",
            "she",
        }
        specific_nodes = [n for n in explicit_nodes if n.lower() not in generic_terms]

        if specific_nodes:
            # Use the first specific node as hub candidate
            hub_candidate = specific_nodes[0]

            # Split triples into those with hub as subject and others
            hub_as_subject = [t for t in triplets if t[0] == hub_candidate]
            other_triples = [t for t in triplets if t[0] != hub_candidate]

            # Return hub-first triples followed by others
            return hub_as_subject + other_triples

        return triplets

    def validate_relation_is_verb(
        self, triplet: Tuple[str, str, str]
    ) -> Tuple[bool, Optional[str], Optional[Tuple[str, str, str]]]:
        subj, relation, obj = triplet

        if not relation or not isinstance(relation, str):
            return False, "empty or invalid relation", None

        relation_clean = relation.strip().lower()
        if not relation_clean:
            return False, "empty relation after cleaning", None

        if self.spacy_nlp is None:
            return True, None, None

        rel_doc = self.spacy_nlp(relation_clean)
        if not rel_doc or len(rel_doc) == 0:
            return False, f"could not parse relation '{relation}'", None

        # Check multiple signals for verb-ness
        has_verb = False

        for token in rel_doc:
            # Signal 1: POS tag
            if token.pos_ in {"VERB", "AUX"}:
                has_verb = True
                break

            # Signal 2: Morphological features
            morph_str = str(token.morph)
            verb_features = ["Tense=", "VerbForm=", "Mood=", "Voice="]
            if any(f in morph_str for f in verb_features):
                has_verb = True
                break

            # Signal 3: Dependency relations typical of verbs
            if token.dep_ in {"ROOT", "aux", "auxpass", "xcomp", "ccomp"}:
                has_verb = True
                break

            # Signal 4: Word shape and probability (for imperative/infinitive)
            if len(rel_doc) == 1 and token.is_alpha:
                # Check if this word has verb-like characteristics in the model
                if hasattr(token, "prob") and token.prob < -3:
                    # Low probability words are more likely to be content words like verbs
                    has_verb = True
                    break

        if not has_verb:
            pos_tags = [f"{t.text}({t.pos_})" for t in rel_doc]
            return (
                False,
                f"relation '{relation}' has no verb. pos: {', '.join(pos_tags)}",
                None,
            )

        return True, None, None

    _COPULAR_VERBS = frozenset(
        {
            "be",
            "is",
            "am",
            "are",
            "was",
            "were",
            "been",
            "being",
            "become",
            "became",
            "seem",
            "appear",
            "feel",
            "look",
            "sound",
            "taste",
            "smell",
            "remain",
            "stay",
        }
    )

    def is_negation_relation(self, relation: str) -> bool:
        if not relation or not isinstance(relation, str):
            return False

        r_lower = relation.lower().strip().replace("_", " ")

        negation_phrases = (
            "not connected",
            "not related",
            "not associated",
            "not linked",
            "not involved",
            "not applicable",
            "not present",
            "not exist",
            "no connection",
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

        for phrase in negation_phrases:
            if phrase in r_lower:
                return True

        if r_lower.startswith(("not ", "no ")):
            return True

        return False

    def validate_triplet_relation(
        self, triplet: Tuple[str, str, str]
    ) -> Dict[str, Any]:
        subj, relation, obj = triplet

        # negation check before spacy parsing
        if self.is_negation_relation(relation):
            return {
                "valid": False,
                "reason": f"negation relation '{relation}' adds no semantic value",
                "corrected_triple": None,
                "action": "reject_negation",
            }

        if self.spacy_nlp is None:
            return {
                "valid": True,
                "reason": None,
                "corrected_triple": None,
                "action": "accept",
            }

        # parse all three slots once
        subj_doc = self.spacy_nlp(subj) if subj else None
        rel_doc = self.spacy_nlp(relation.strip().lower()) if relation else None
        obj_doc = self.spacy_nlp(obj) if obj else None

        subj_pos = subj_doc[0].pos_ if subj_doc and len(subj_doc) > 0 else None
        obj_pos = obj_doc[0].pos_ if obj_doc and len(obj_doc) > 0 else None

        # --- verb-presence check FIRST (includes morphological fallback) ---
        is_valid_verb, verb_reason, verb_corrected = self.validate_relation_is_verb(
            triplet
        )
        rel_has_verb = is_valid_verb

        # find the relation's verb lemma
        rel_lemma = None
        if rel_doc:
            # Pass 1: Look for explicit VERB/AUX tags
            for t in rel_doc:
                if t.pos_ in {"VERB", "AUX"}:
                    rel_lemma = t.lemma_.lower()
                    break

            # Pass 2: If no explicit verb but rel_has_verb is True, use morphological features
            if rel_lemma is None and rel_has_verb:
                for t in rel_doc:
                    morph_str = str(t.morph)
                    # Check for any verb-related morphological features
                    if any(
                        f in morph_str
                        for f in ["Tense=", "VerbForm=", "Mood=", "Voice="]
                    ):
                        rel_lemma = t.lemma_.lower()
                        break

            # Pass 3: Check dependency relations that indicate verbs
            if rel_lemma is None:
                for t in rel_doc:
                    if t.dep_ in {"ROOT", "aux", "auxpass", "xcomp", "ccomp"}:
                        rel_lemma = t.lemma_.lower()
                        # If we found a verb-like dependency, ensure rel_has_verb is True
                        rel_has_verb = True
                        break

            # Pass 4: Use word shape heuristics for common verb patterns
            if rel_lemma is None:
                for t in rel_doc:
                    word = t.text.lower()
                    # Common verb endings (linguistic patterns, not hardcoded list)
                    if (
                        t.is_alpha
                        and word.endswith(("ing", "ed", "en", "s"))
                        and len(word) > 3
                    ):
                        # Check if this word has verb-like properties in the model
                        if hasattr(t, "prob") and t.prob < -4:
                            rel_lemma = t.lemma_.lower()
                            rel_has_verb = True
                            break

            # Pass 5: Final fallback - use first token's lemma
            if rel_lemma is None and len(rel_doc) > 0:
                rel_lemma = rel_doc[0].lemma_.lower()

                # If we got here, do one last check for verb-likeness
                first_token = rel_doc[0]
                word = first_token.text.lower()
                if first_token.is_alpha and word.endswith(("ing", "ed", "en", "s")):
                    # This looks like a verb even if spaCy missed it
                    rel_has_verb = True

        # --- RULE: Reject "has" + adjective ---
        if rel_lemma == "have" and obj_pos == "ADJ":
            return {
                "valid": False,
                "reason": f"'has' cannot take adjective '{obj}' as object — 'has' requires a noun",
                "corrected_triple": None,
                "action": "reject",
            }

        # --- RULE: Reject circular "is" + noun where noun is same as subject or action ---
        if rel_lemma == "be" and obj_pos in {"NOUN", "PROPN"}:
            obj_lower = obj.lower().strip()
            subj_lower = subj.lower().strip()

            # Check if object is the same as subject (circular)
            if obj_lower == subj_lower:
                return {
                    "valid": False,
                    "reason": f"circular relation: '{subj} is {obj}' says nothing",
                    "corrected_triple": None,
                    "action": "reject",
                }

            # Check if object is the nominalized form of an action (victory is battle)
            action_nouns = {"battle", "war", "fight", "victory", "defeat", "conquest"}
            if obj_lower in action_nouns and subj_lower in action_nouns:
                return {
                    "valid": False,
                    "reason": f"'{subj} is {obj}' is circular — both are events/actions",
                    "corrected_triple": None,
                    "action": "reject",
                }

        # --- structural POS checks ---

        # ADJ-*-ADJ: always reject
        if subj_pos == "ADJ" and obj_pos == "ADJ":
            return {
                "valid": False,
                "reason": f"both subject '{subj}' and object '{obj}' are adjectives",
                "corrected_triple": None,
                "action": "reject",
            }

        # ADJ-VERB-NOUN with action verb: adjectives can't perform actions
        if (
            subj_pos == "ADJ"
            and obj_pos in {"NOUN", "PROPN"}
            and rel_has_verb
            and rel_lemma not in self._COPULAR_VERBS
        ):
            return {
                "valid": False,
                "reason": (
                    f"adjective '{subj}' cannot be the subject of "
                    f"action verb '{relation}'"
                ),
                "corrected_triple": None,
                "action": "reject",
            }

        # NOUN/PRON-VERB-ADJ with non-copular verb: action verbs need noun objects
        # This catches both "charlemagne values educational" and "he values traditional"
        if (
            subj_pos in {"NOUN", "PROPN", "PRON"}
            and obj_pos == "ADJ"
            and rel_has_verb
            and rel_lemma not in self._COPULAR_VERBS
        ):
            return {
                "valid": False,
                "reason": (
                    f"action verb '{relation}' cannot take adjective "
                    f"'{obj}' as object — needs a noun"
                ),
                "corrected_triple": None,
                "action": "reject",
            }
        # PRON-VERB-NOUN is usually fine (keep existing acceptance)
        # No need for explicit rule as it will pass through

        # --- verb-presence result ---
        if is_valid_verb:
            return {
                "valid": True,
                "reason": None,
                "corrected_triple": None,
                "action": "accept",
            }

        # --- correction attempts for missing verb ---

        # verb might be in the object slot (common swap)
        if obj_doc and any(t.pos_ in {"VERB", "AUX"} for t in obj_doc):
            return {
                "valid": False,
                "reason": f"relation '{relation}' has no verb, but object '{obj}' does",
                "corrected_triple": (subj, obj, relation),
                "action": "swap",
            }

        # single adjective as relation → suggest copula
        if rel_doc and len(rel_doc) == 1 and rel_doc[0].pos_ == "ADJ":
            return {
                "valid": False,
                "reason": f"relation '{relation}' is an adjective, not a verb",
                "corrected_triple": (subj, "is", relation),
                "action": "add_copula",
            }

        return {
            "valid": False,
            "reason": verb_reason,
            "corrected_triple": None,
            "action": "reject",
        }

    def is_likely_verb(self, word: str) -> bool:
        if self.spacy_nlp is None:
            return False

        doc = self.spacy_nlp(word)
        if not doc or len(doc) != 1:
            return False

        token = doc[0]
        if token.pos_ in {"VERB", "AUX"}:
            return True

        morph_str = str(token.morph)
        return any(f in morph_str for f in ["VerbForm", "Tense", "Mood", "Voice"])
