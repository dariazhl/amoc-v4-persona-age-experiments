from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from difflib import SequenceMatcher
from amoc.core.node import NodeType
from amoc.config.constants import DEFAULT_ACTIVATION_SCORE, DECAY_STEP


try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _EMB_MODEL: Optional[SentenceTransformer] = None
except Exception:
    SentenceTransformer = None
    np = None
    _EMB_MODEL = None

if TYPE_CHECKING:
    from amoc.core.node import Node


def _maybe_embed(text: str) -> Optional["np.ndarray"]:
    global _EMB_MODEL
    if SentenceTransformer is None or np is None:
        return None
    if _EMB_MODEL is None:
        try:
            _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMB_MODEL = None
            return None
    try:
        vecs = _EMB_MODEL.encode([text], normalize_embeddings=True)
        return vecs[0]
    except Exception:
        return None


class Edge:
    def __init__(
        self,
        source_node: "Node",
        dest_node: "Node",
        label: str,
        visibility_score: int,
        *,
        persona_influenced: bool = False,
        inferred: bool = False,
        active: bool = True,
        created_at_sentence: Optional[int] = None,
        activation_score: Optional[int] = None,
        relation_class=None,
        justification=None,
    ) -> None:
        self.source_node: "Node" = source_node
        self.dest_node: "Node" = dest_node
        self.active: bool = active
        self.label: str = label
        self.visibility_score: int = visibility_score

        self.persona_influenced: bool = persona_influenced
        self.inferred: bool = inferred

        self.activation_score: int = (
            activation_score
            if activation_score is not None
            else DEFAULT_ACTIVATION_SCORE
        )
        self.origin_sentence: Optional[int] = created_at_sentence
        self.created_at_sentence: Optional[int] = created_at_sentence

        self.similarity_threshold = 0.8
        self.embedding: Optional["np.ndarray"] = _maybe_embed(label)

        self.asserted_this_sentence: bool = False
        self.reactivated_this_sentence: bool = False
        self.activation_role: Optional[str] = None

        self.forced_connection: bool = False
        self.checkpoint: bool = False

    def mark_as_reactivated(self, reset_score: bool = True) -> None:
        self.active = True
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = True
        self.activation_role = "reactivated"

        if reset_score:
            self.activation_score = max(self.activation_score, 1)

    def mark_as_asserted(self, reset_score: bool = True) -> None:
        self.active = True
        self.asserted_this_sentence = True
        self.reactivated_this_sentence = False
        self.activation_role = "asserted"
        if reset_score:
            self.activation_score = DEFAULT_ACTIVATION_SCORE

    def reset_for_sentence_start(self) -> None:
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False
        self.activation_role = None
        self.active = False

    def reduce_visibility(self) -> None:
        if self.visibility_score <= 1:
            self.visibility_score = 1
            return
        self.visibility_score -= DECAY_STEP
        if self.visibility_score <= 1:
            self.visibility_score = 1

    def is_property_edge(self) -> bool:
        return (
            self.source_node.node_type == NodeType.PROPERTY
            or self.dest_node.node_type == NodeType.PROPERTY
        )

    def is_asserted(self) -> bool:
        return self.asserted_this_sentence

    def is_reactivated(self) -> bool:
        return self.reactivated_this_sentence

    def is_similar(self, other_edge: "Edge") -> bool:
        a = (self.label or "").strip().lower()
        b = (other_edge.label or "").strip().lower()
        if a == b:
            return True

        if self.embedding is not None and other_edge.embedding is not None:
            cos = float(np.dot(self.embedding, other_edge.embedding))
            if cos >= self.similarity_threshold:
                return True

        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold

    def __eq__(self, other: "Edge") -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        if self.active:
            if self.asserted_this_sentence:
                status = "asserted"
            elif self.reactivated_this_sentence:
                status = "reactivated"
            else:
                status = "active"
        else:
            status = "inactive"
        return f"{self.source_node.get_text_representer()} --{self.label} ({status}, act={self.activation_score})--> {self.dest_node.get_text_representer()} (vis={self.visibility_score})"

    def __repr__(self) -> str:
        return self.__str__()
