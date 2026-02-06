from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from difflib import SequenceMatcher

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    _EMB_MODEL: Optional[SentenceTransformer] = None
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None
    np = None
    _EMB_MODEL = None

if TYPE_CHECKING:
    from amoc.graph.node import Node


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
    # Default activation score for new edges
    DEFAULT_ACTIVATION_SCORE: int = 3

    def __init__(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        forget_score: int,
        active: bool = True,
        created_at_sentence: Optional[int] = None,
        activation_score: Optional[int] = None,
    ) -> None:
        self.source_node: Node = source_node
        self.dest_node: Node = dest_node
        self.active: bool = active
        self.label: str = label
        self.forget_score: int = forget_score
        # activation_score: sentence-local activation counter (distinct from forget_score)
        # Edges are activated when asserted/inferred; decays when inactive
        self.activation_score: int = (
            activation_score
            if activation_score is not None
            else self.DEFAULT_ACTIVATION_SCORE
        )
        # origin_sentence: the sentence where this edge was first created (immutable)
        self.origin_sentence: Optional[int] = created_at_sentence
        self.similarity_threshold = 0.8
        self.embedding: Optional["np.ndarray"] = _maybe_embed(label)
        self.created_at_sentence: Optional[int] = created_at_sentence
        self.metadata: dict[str, bool] = {}

        # === EDGE ASSERTION STATES (per AMoC v4 paper) ===
        # Each edge has exactly one state per sentence:
        # - asserted_this_sentence: edge was explicitly created/inferred this sentence
        # - reactivated_this_sentence: edge was reactivated from memory this sentence
        # - inactive: edge exists in memory but not active this sentence
        # These are mutually exclusive and reset at each sentence start
        # Default is False - mark_as_asserted() must be called explicitly for new edges
        self.asserted_this_sentence: bool = False
        self.reactivated_this_sentence: bool = False

        # activation_role: tracks how edge became active
        # "asserted" - created/inferred this sentence
        # "reactivated" - reactivated from memory
        # "connector" - promoted to preserve active graph connectivity (not asserted/reactivated)
        # Default is None - mark_as_* methods set the appropriate role
        self.activation_role: Optional[str] = None

        # ==========================================================================
        # TASK 2: FORCED CONNECTIVITY EDGES
        # ==========================================================================
        # forced_connection: True if this edge was created by secondary LLM call
        # specifically to restore graph connectivity when no existing edges could
        # connect disconnected components.
        #
        # KEY DISTINCTION:
        # - connector edges: EXISTING edges promoted to active for connectivity
        # - forced_connection edges: NEW edges created by LLM to fix disconnection
        #
        # These edges are marked so they can be:
        # 1. Distinguished in debugging/analysis
        # 2. Styled differently in plots if needed
        # 3. Excluded from certain semantic analyses
        self.forced_connection: bool = False

    def fade_away(self) -> None:
        """
        AMoC v4 STEP 7: Edge decay mechanism.

        Per paper Section 3.1:
        - Decrement forget_score by 1
        - When forget_score reaches 0, edge becomes inactive
        - This implements the gradual "fading" of memory traces
        """
        self.forget_score -= 1
        if self.forget_score <= 0:
            self.active = False

    def reset_for_sentence_start(self) -> None:
        """
        Reset edge state at the start of a new sentence.
        Per AMoC v4 paper: all edges become inactive at sentence start,
        then selectively activated through assertion or reactivation.
        """
        self.active = False
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False
        self.activation_role = None

    def deactivate(self) -> None:
        """Deactivate edge (sentence-local reset)."""
        self.active = False
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False
        self.activation_role = None

    def mark_as_asserted(self, reset_score: bool = True) -> None:
        """
        Mark edge as asserted this sentence (created/inferred from current sentence).
        Asserted edges are active and have their activation_score reset.
        """
        self.active = True
        self.asserted_this_sentence = True
        self.reactivated_this_sentence = False
        self.activation_role = "asserted"
        if reset_score:
            self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def mark_as_reactivated(self, reset_score: bool = True) -> None:
        """
        Mark edge as reactivated this sentence (brought back from memory).
        PROPERTY edges should NEVER be reactivated - caller must check.
        """
        if self.activation_role == "connector":
            return
        self.active = True
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = True
        self.activation_role = "reactivated"
        if reset_score:
            self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def mark_as_connector(self) -> None:
        """
        Mark edge as a connector (promoted to preserve active graph connectivity).
        Connector edges:
        - Must already exist in the cumulative graph
        - Must NOT be PROPERTY edges (caller must check)
        - Do NOT count as asserted or reactivated
        - Do NOT increase activation scores
        - Are NOT eligible for inference
        - Exist only to preserve structural connectivity
        """
        self.active = True
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False
        self.activation_role = "connector"
        # Do NOT reset activation_score - connectors don't boost activation

    def activate(self, reset_score: bool = True) -> None:
        """Activate edge and optionally reset activation_score (legacy method)."""
        self.active = True
        if reset_score:
            self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def decay_activation(self) -> None:
        if self.activation_role == "connector":
            return
        if not self.active:
            self.activation_score -= 1

    def is_connector(self) -> bool:
        """Check if this edge is serving as a connector (for connectivity only)."""
        return self.activation_role == "connector"

    def mark_as_forced_connection(self) -> None:
        """
        TASK 2: Mark edge as a forced connectivity edge.

        Forced connection edges are NEW edges created by a secondary LLM call
        specifically to restore graph connectivity when:
        1. The graph becomes disconnected
        2. No existing edges in the cumulative graph can connect components

        Unlike connector edges (which promote existing edges), forced connection
        edges are semantically new and created with minimal semantic commitment.

        Properties:
        - Created by secondary LLM call for connectivity only
        - Marked for traceability and debugging
        - Can be styled differently in visualizations
        - Should be excluded from certain semantic analyses
        """
        self.forced_connection = True
        self.active = True
        self.asserted_this_sentence = True  # It IS newly asserted
        self.reactivated_this_sentence = False
        self.activation_role = "asserted"  # Semantically it's a new edge
        self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def is_forced_connection(self) -> bool:
        """
        TASK 2: Check if this edge was created by secondary LLM call for connectivity.

        Returns True if this edge was force-created to connect disconnected
        components, rather than being inferred from semantic content.
        """
        return self.forced_connection

    def is_asserted(self) -> bool:
        """Check if this edge was asserted this sentence."""
        return self.asserted_this_sentence

    def is_reactivated(self) -> bool:
        """Check if this edge was reactivated this sentence."""
        return self.reactivated_this_sentence

    def is_property_edge(self) -> bool:
        """
        Check if this edge connects a concept to a property (attribute edge).
        Attribute edges should only attach in their origin sentence.
        """
        from amoc.graph.node import NodeType

        src_type = self.source_node.node_type
        dst_type = self.dest_node.node_type
        return (src_type == NodeType.CONCEPT and dst_type == NodeType.PROPERTY) or (
            src_type == NodeType.PROPERTY and dst_type == NodeType.CONCEPT
        )

    def is_similar(self, other_edge: "Edge") -> bool:
        """
        Check if two edges have similar labels.
        NOTE: This only checks label similarity, NOT direction.
        Use matches_directed() for full direction-aware comparison.
        """
        a = (self.label or "").strip().lower()
        b = (other_edge.label or "").strip().lower()
        if a == b:
            return True

        if self.embedding is not None and other_edge.embedding is not None:
            cos = float(np.dot(self.embedding, other_edge.embedding))
            if cos >= self.similarity_threshold:
                return True

        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold

    def matches_directed(self, other_edge: "Edge") -> bool:
        """
        Check if this edge matches another edge considering DIRECTION.

        Per AMoC v4 paper: (A → B, label) ≠ (B → A, label)
        Two edges match only if:
        1. Same source node
        2. Same destination node
        3. Similar labels
        """
        same_direction = (
            self.source_node == other_edge.source_node
            and self.dest_node == other_edge.dest_node
        )
        return same_direction and self.is_similar(other_edge)

    def is_inverse_of(self, other_edge: "Edge") -> bool:
        """
        Check if this edge is the directional inverse of another edge.

        (A → B) is inverse of (B → A) if labels are similar.
        This is used to detect and canonicalize passive voice constructions.
        """
        opposite_direction = (
            self.source_node == other_edge.dest_node
            and self.dest_node == other_edge.source_node
        )
        return opposite_direction and self.is_similar(other_edge)

    def get_directed_key(self) -> tuple:
        """
        Get a unique key for this directed edge.

        Returns (source_lemmas, dest_lemmas, label) tuple that uniquely
        identifies this directed edge. Used for deduplication.
        """
        src_key = tuple(self.source_node.lemmas)
        dst_key = tuple(self.dest_node.lemmas)
        label_key = (self.label or "").strip().lower()
        return (src_key, dst_key, label_key)

    def get_undirected_key(self) -> tuple:
        """
        Get a direction-agnostic key for this edge.

        Returns a tuple where node order is normalized (sorted by lemmas).
        Used to find edges between the same pair of nodes regardless of direction.
        """
        src_key = tuple(self.source_node.lemmas)
        dst_key = tuple(self.dest_node.lemmas)
        label_key = (self.label or "").strip().lower()
        # Sort to make order-independent
        if src_key <= dst_key:
            return (src_key, dst_key, label_key)
        return (dst_key, src_key, label_key)

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
            elif self.activation_role == "connector":
                status = "connector"
            else:
                status = "active"
        else:
            status = "inactive"
        return f"{self.source_node.get_text_representer()} --{self.label} ({status}, act={self.activation_score})--> {self.dest_node.get_text_representer()} (forget={self.forget_score})"

    def __repr__(self) -> str:
        return self.__str__()

    def violates_property_sentence_constraint(self, current_sentence: int) -> bool:
        """
        PROPERTY edges must only be active in their origin sentence.
        """
        if self.is_property_edge() and self.origin_sentence is None:
            raise RuntimeError("PROPERTY edge missing origin_sentence")
        if not self.is_property_edge():
            return False
        if self.origin_sentence is None:
            return True
        return current_sentence != self.origin_sentence
