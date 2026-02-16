from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from difflib import SequenceMatcher
from enum import Enum, auto

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


# =============================================================================
# ONTOLOGY-INVARIANT ENUMS (RECOMMENDATION 1)
# =============================================================================
# These enums define structural ontology that is INVARIANT to persona.
# Persona may affect labels (expression), NOT these ontology classes.


class RelationClass(Enum):
    """Structural classification of relations - invariant to persona."""

    EVENTIVE = auto()  # actions, events
    STATIVE = auto()  # states, properties
    ATTRIBUTIVE = auto()  # has / is / describes
    CONNECTIVE = auto()  # relates to / involves / concerns


class Justification(Enum):
    """How the edge was derived - invariant to persona."""

    TEXTUAL = auto()  # explicit in text
    IMPLIED = auto()  # strongly implied
    CONNECTIVE = auto()  # added for connectivity


class EventiveRole(Enum):
    PARTICIPATION = auto()  # concept → event
    EFFECT = auto()  # event → concept


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
    """
    Edge with ontology-invariant metadata (Recommendation 1).

    CRITICAL RULE:
    - relation_class is structural ontology - invariant to persona
    - justification is derivation source - invariant to persona
    - persona_influenced only affects the label (expression)
    - Persona must NEVER change relation_class, justification, or inferred
    """

    # Default activation score for new edges
    DEFAULT_ACTIVATION_SCORE: int = 2
    DECAY_STEP = 1

    def __init__(
        self,
        source_node: "Node",
        dest_node: "Node",
        label: str,
        visibility_score: int,
        *,
        relation_class: Optional[RelationClass] = None,
        justification: Optional[Justification] = None,
        persona_influenced: bool = False,
        inferred: bool = False,
        active: bool = True,
        created_at_sentence: Optional[int] = None,
        activation_score: Optional[int] = None,
        eventive_role: Optional[EventiveRole] = None,
    ) -> None:
        self.source_node: "Node" = source_node
        self.dest_node: "Node" = dest_node
        self.active: bool = active
        self.label: str = label
        self.visibility_score: int = visibility_score

        # ==========================================================================
        # ONTOLOGY-INVARIANT METADATA (Recommendation 1)
        # ==========================================================================
        # These are structural properties that persona CANNOT modify.
        # relation_class determines structural behavior, NOT the label.
        self.relation_class: Optional[RelationClass] = relation_class
        self.justification: Optional[Justification] = justification
        self.persona_influenced: bool = persona_influenced
        self.inferred: bool = inferred
        self.eventive_role: Optional[EventiveRole] = eventive_role

        # activation_score: sentence-local activation counter (distinct from visibility_score)
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
        # FORCED CONNECTIVITY EDGES
        # ==========================================================================
        # forced_connection: True if this edge was created by secondary LLM call
        # specifically to restore graph connectivity when no existing edges could
        # connect disconnected components.
        self.forced_connection: bool = False
        self.structural: bool = False

    def mark_as_reactivated(self, reset_score: bool = True) -> None:
        self.active = True
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = True
        self.activation_role = "reactivated"

        if reset_score:
            # Weak boost instead of full reset (paper-aligned)
            self.activation_score = max(self.activation_score, 1)

    def set_relation_class(self, new_class: RelationClass) -> None:
        """
        DISCOURAGED: relation_class should be immutable after edge creation.

        Per Recommendation 1: Ontology mutation after creation is discouraged.
        Records violation in metadata instead of raising.
        """
        self.metadata.setdefault("ontology_violations", []).append(
            "REC1: relation_class mutated after creation"
        )
        self.relation_class = new_class

    def is_property_edge(self) -> bool:
        """
        Check if this edge connects a concept to a property (attribute edge).

        Ontology-invariant: determined by relation_class, not by label.
        """
        return self.relation_class == RelationClass.ATTRIBUTIVE

    def reduce_visibility(self) -> None:
        if self.visibility_score <= 0:
            return

        self.visibility_score -= self.DECAY_STEP

        # Activation decays with visibility
        # if self.activation_score > 0:
        #     self.activation_score -= 1

        if self.visibility_score <= 0:
            self.visibility_score = 0
            self.active = False

    def reset_for_sentence_start(self) -> None:
        """
        Reset edge state at the start of a new sentence.
        Per AMoC v4 paper: all edges become inactive at sentence start,
        then selectively activated through assertion or reactivation.
        """
        self.asserted_this_sentence = False
        self.reactivated_this_sentence = False
        self.activation_role = None
        self.active = False

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

    def activate(self, reset_score: bool = True) -> None:
        """Activate edge and optionally reset activation_score (legacy method)."""
        self.active = True
        if reset_score:
            self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def decay_activation(self) -> None:
        """
        Activation decay disabled (replication mode uses visibility only).
        """
        if self.activation_score > 0:
            self.activation_score -= 1

        # Synchronize active with activation_score
        self.active = self.activation_score > 0

    def mark_as_forced_connection(self) -> None:
        """
        Mark edge as a forced connectivity edge.

        Forced connection edges should be CREATED with correct ontology.
        This method only marks activation state, NOT ontology.

        Records violations in metadata instead of raising/asserting.
        """
        # Record violations as metadata instead of asserting
        violations = []
        if self.relation_class != RelationClass.CONNECTIVE:
            violations.append(
                f"Forced connection edge should have relation_class=CONNECTIVE, got {self.relation_class}"
            )
        if self.justification != Justification.CONNECTIVE:
            violations.append(
                f"Forced connection edge should have justification=CONNECTIVE, got {self.justification}"
            )
        if not self.inferred:
            violations.append(
                "Forced connection edge should be inferred=True at creation"
            )

        if violations:
            self.metadata.setdefault("ontology_violations", []).extend(violations)

        self.forced_connection = True
        self.active = True
        self.asserted_this_sentence = True
        self.reactivated_this_sentence = False
        self.activation_role = "asserted"
        self.activation_score = self.DEFAULT_ACTIVATION_SCORE

    def is_forced_connection(self) -> bool:
        """Check if this edge was created by secondary LLM call for connectivity."""
        return self.forced_connection

    def is_asserted(self) -> bool:
        """Check if this edge was asserted this sentence."""
        return self.asserted_this_sentence

    def is_reactivated(self) -> bool:
        """Check if this edge was reactivated this sentence."""
        return self.reactivated_this_sentence

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
            else:
                status = "active"
        else:
            status = "inactive"
        rel_class = self.relation_class.name if self.relation_class else "NONE"
        return f"{self.source_node.get_text_representer()} --{self.label} ({status}, {rel_class}, act={self.activation_score})--> {self.dest_node.get_text_representer()} (vis={self.visibility_score})"

    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# CENTRALIZED ONTOLOGY DIAGNOSTICS (Recommendation 1 + 2)
# =============================================================================
# This is the ONLY place where ontology rules are checked.
# This function is persona-blind and must NOT inspect the relation label.
# CRITICAL: This function NEVER raises or blocks - it only records violations.


def enforce_ontology_invariants(edge: Edge) -> None:
    """
    Check ontology invariants on an edge and record violations as metadata.

    This function:
    - Is persona-blind
    - Must be called immediately after every edge creation
    - Is the ONLY place where ontology rules are checked
    - Must NOT inspect the relation label
    - Must NOT mutate ontology fields - only validate and record
    - NEVER raises exceptions or blocks execution

    Violations are recorded in edge.metadata["ontology_violations"].
    """
    from amoc.graph.node import NodeType

    violations = []

    # INVARIANT 1: relation_class must be explicitly set (no implicit derivation)
    if edge.relation_class is None:
        violations.append("REC1: missing relation_class")

    # INVARIANT 2: justification must be explicitly set
    if edge.justification is None:
        violations.append("REC1: missing justification")

    # INVARIANT 3: ATTRIBUTIVE relations target PROPERTY nodes only
    if edge.relation_class == RelationClass.ATTRIBUTIVE:
        if edge.dest_node.node_type != NodeType.PROPERTY:
            violations.append(
                f"REC1: ATTRIBUTIVE relation must target PROPERTY node, got {edge.dest_node.node_type}"
            )

    # INVARIANT 4: CONNECTIVE edges must be inferred at creation time
    if edge.relation_class == RelationClass.CONNECTIVE and not edge.inferred:
        violations.append("REC1: CONNECTIVE edges must be inferred at creation time")

    # ==========================================================================
    # RECOMMENDATION 2: EVENT MEDIATION INVARIANTS
    # ==========================================================================

    # INVARIANT 5: EVENTIVE relations must be mediated by an EVENT node
    if edge.relation_class == RelationClass.EVENTIVE:
        if (
            edge.source_node.node_type != NodeType.EVENT
            and edge.dest_node.node_type != NodeType.EVENT
        ):
            violations.append(
                f"REC2: EVENTIVE edge not mediated by EVENT node "
                f"({edge.source_node.node_type} -> {edge.dest_node.node_type})"
            )

    # INVARIANT 5b: EVENTIVE relations should declare their eventive role
    if edge.relation_class == RelationClass.EVENTIVE:
        if getattr(edge, "eventive_role", None) is None:
            violations.append("REC2: EVENTIVE edge missing eventive_role")

    # INVARIANT 6: EVENT nodes may only connect via EVENTIVE or CONNECTIVE relations
    is_event_involved = (
        edge.source_node.node_type == NodeType.EVENT
        or edge.dest_node.node_type == NodeType.EVENT
    )
    if is_event_involved:
        if edge.relation_class not in {
            RelationClass.EVENTIVE,
            RelationClass.CONNECTIVE,
        }:
            violations.append(
                f"REC2: EVENT node has invalid relation_class {edge.relation_class}"
            )

    # Record violations as metadata (NEVER raise)
    if violations:
        edge.metadata.setdefault("ontology_violations", []).extend(violations)


def assert_persona_did_not_modify_ontology(edge: Edge) -> None:
    """
    Check that persona did not modify structural ontology.

    Per Recommendation 1: Persona may affect expression (labels),
    NOT ontology (relation_class, justification).

    Records violations in metadata instead of asserting.
    NEVER raises or blocks execution.
    """
    if edge.persona_influenced and getattr(edge, "_relation_class_changed", False):
        edge.metadata.setdefault("ontology_violations", []).append(
            "REC1: Persona modified ontology (relation_class changed)"
        )


def collect_edge_violations(edge: Edge) -> list[str]:
    """
    Collect all ontology violations recorded on an edge.

    Returns:
        List of violation strings, empty if no violations.
    """
    return edge.metadata.get("ontology_violations", [])
