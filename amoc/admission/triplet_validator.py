from typing import Dict, List, Optional, Tuple, Any
import logging
from difflib import SequenceMatcher

from amoc.core.node import Node, NodeSource
from amoc.core.edge import Edge
from amoc.extraction.linguistic import LinguisticProcessing
from amoc.admission.text_normalizer import TextNormalizer
from amoc.llm.vllm_client import VLLMClient
from amoc.core.edge import _maybe_embed


class TripletValidator:
    def __init__(
        self,
        linguistic_ops: LinguisticProcessing,
        text_normalizer: TextNormalizer,
        client: VLLMClient,
        persona: str = "",
        similarity_threshold: float = 0.8,
    ):
        self.linguistic_ops = linguistic_ops
        self.text_normalizer = text_normalizer
        self.client = client
        self.persona = persona
        self.similarity_threshold = similarity_threshold
        self._embedding_cache = {}

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
            self.linguistic_ops.extract_deterministic_structure_fn(
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
        self, subj: str, rel: str, obj: str, det_lookup: Dict
    ) -> Dict:
        forward = det_lookup["forward"]
        reverse = det_lookup["reverse"]

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

        # case 3: no deterministic info – accept llm triple as is
        return {"valid": True, "subj": subj, "obj": obj, "rel": rel}

    # use llm to validate semantic plausibility of a triple
    def validate_with_llm(self, subj: str, rel: str, obj: str, sentence: str) -> Dict:
        llm_validation = self.client.validate_triple(
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
