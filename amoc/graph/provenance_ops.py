"""
Provenance operations for Graph.

CANONICAL VALIDATION AUTHORITY for node creation.

All validation/filtering logic lives here:
- Semantic category validation (system terms, narration artifacts, generic terms)
- Story grounding checks
- Persona leakage prevention
- Provenance sanity checks

Graph.add_or_get_node() delegates validation to this class.

SEMANTIC CATEGORIES (for structural validation):
1. SYSTEM_TERMINOLOGY: Graph/system technical terms (node, edge, property, etc.)
2. NARRATIVE_META: Terms describing narrative structure (story, sentence, text, etc.)
3. CONTEXT_REFERENCES: Terms referencing processing context (student, persona, user, etc.)
4. LINGUISTIC_CATEGORIES: Parts of speech names (pronoun, noun, etc.)
5. GENERIC_ABSTRACTIONS: Highly abstract/vague terms (thing, kind, type, etc.)
"""

from typing import TYPE_CHECKING, Set, Optional, List, Tuple
import logging
import re

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import NodeProvenance, NodeType


# =============================================================================
# SEMANTIC CATEGORY DEFINITIONS (Structural Validation)
# =============================================================================
# These replace hardcoded blocklists with semantic categories.
# Each category defines a TYPE of term that should be filtered.

class SemanticCategory:
    """
    Semantic category for lemma classification.

    Replaces hardcoded blocklists with typed categories that can be:
    - Extended programmatically
    - Configured externally
    - Tested for category membership
    """

    # Category 1: Graph/system technical terms
    SYSTEM_TERMINOLOGY: Set[str] = {
        "node", "edge", "property", "label", "target", "source", "relation",
    }

    # Category 2: Terms describing narrative structure
    NARRATIVE_META: Set[str] = {
        "story", "narrative", "sentence", "text", "paragraph",
        "mention", "mentions", "narration",
    }

    # Category 3: Terms referencing processing context
    CONTEXT_REFERENCES: Set[str] = {
        "student", "persona", "user", "context",
    }

    # Category 4: Parts of speech names
    LINGUISTIC_CATEGORIES: Set[str] = {
        "pronoun", "noun",
    }

    # Category 5: Highly abstract/vague terms
    GENERIC_ABSTRACTIONS: Set[str] = {
        "thing", "kind", "type", "approach", "person", "object", "place",
    }

    # Category 6: Garbage tokens (too short, single chars, technical artifacts)
    GARBAGE_TOKENS: Set[str] = {
        "t",  # Single character artifact
    }

    @classmethod
    def get_category(cls, lemma: str) -> Optional[str]:
        """Classify a lemma into its semantic category."""
        lemma_lower = lemma.lower()

        if lemma_lower in cls.SYSTEM_TERMINOLOGY:
            return "system_terminology"
        if lemma_lower in cls.NARRATIVE_META:
            return "narrative_meta"
        if lemma_lower in cls.CONTEXT_REFERENCES:
            return "context_references"
        if lemma_lower in cls.LINGUISTIC_CATEGORIES:
            return "linguistic_categories"
        if lemma_lower in cls.GENERIC_ABSTRACTIONS:
            return "generic_abstractions"
        if lemma_lower in cls.GARBAGE_TOKENS:
            return "garbage_tokens"
        return None

    @classmethod
    def is_blocked(cls, lemma: str) -> bool:
        """Check if a lemma belongs to any blocked category."""
        return cls.get_category(lemma) is not None

    @classmethod
    def all_blocked_lemmas(cls) -> Set[str]:
        """Get the union of all blocked lemmas across categories."""
        return (
            cls.SYSTEM_TERMINOLOGY
            | cls.NARRATIVE_META
            | cls.CONTEXT_REFERENCES
            | cls.LINGUISTIC_CATEGORIES
            | cls.GENERIC_ABSTRACTIONS
            | cls.GARBAGE_TOKENS
        )

    # =========================================================================
    # BLOCKLIST EQUIVALENTS (for exact parity during migration)
    # =========================================================================
    # These sets exactly match the original blocklists for parity testing.
    # After migration is complete, these can be removed.

    # Equivalent to GARBAGE_LEMMAS (checked ONLY for primary lemma)
    GARBAGE_LEMMAS_EQUIVALENT: Set[str] = {
        "edge", "node", "relation", "property", "label", "target",
        "source", "t", "type", "person", "approach", "kind", "thing",
    }

    # Equivalent to FORBIDDEN_NODE_LEMMAS (checked for ALL lemmas)
    FORBIDDEN_LEMMAS_EQUIVALENT: Set[str] = {
        "student", "persona", "relation", "context", "object", "place",
        "story", "narrative", "sentence", "edge", "node", "property",
        "label", "target", "source", "pronoun", "noun", "user",
    }

    # Equivalent to NARRATION_ARTIFACT_LEMMAS (checked for ALL lemmas)
    NARRATION_ARTIFACT_EQUIVALENT: Set[str] = {
        "text", "sentence", "paragraph", "mention", "mentions", "narration", "story",
    }


class ProvenanceOps:
    """
    Canonical validation authority for node creation.

    Responsibilities:
    - Validate lemmas using semantic category classification
    - Ensure story grounding
    - Prevent persona leakage
    - Provenance sanity checks

    MIGRATION COMPLETE: Hardcoded blocklists replaced with SemanticCategory.
    """

    # =========================================================================
    # LEGACY ALIASES (for backwards compatibility)
    # =========================================================================
    # These provide access to the blocked terms via the old API.
    # They delegate to SemanticCategory which is now the source of truth.

    @property
    def FORBIDDEN_NODE_LEMMAS(self) -> set:
        """Legacy alias for SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT."""
        return SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT

    @property
    def NARRATION_ARTIFACT_LEMMAS(self) -> set:
        """Legacy alias for SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT."""
        return SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT

    @property
    def GARBAGE_LEMMAS(self) -> set:
        """Legacy alias for SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT."""
        return SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def validate_node_creation(
        self,
        lemmas: List[str],
        node_type: "NodeType",
        provenance: Optional["NodeProvenance"] = None,
        is_new_node: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate node creation using semantic category classification.

        Uses SemanticCategory to classify and filter terms instead of
        hardcoded blocklists.

        Args:
            lemmas: List of lemmas for the node (expected to be lowercased)
            node_type: Type of node being created
            provenance: Optional provenance information
            is_new_node: Whether this is a new node or existing

        Returns:
            (valid, reason) tuple where valid is True if allowed
        """
        from amoc.graph.node import NodeType, NodeProvenance

        if not lemmas or not lemmas[0]:
            return (False, "empty_lemmas")

        primary_lemma = lemmas[0].lower()

        if len(primary_lemma) <= 1:
            return (False, "lemma_too_short")

        # Check PRIMARY lemma against garbage terms
        if primary_lemma in SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT:
            return (False, "garbage_lemma")

        # Check for invalid characters
        if node_type != NodeType.EVENT and not re.match(r"^[a-zA-Z]+$", lemmas[0]):
            return (False, "invalid_characters")

        # Check ALL lemmas against forbidden terms
        for lemma in lemmas:
            if lemma in SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT:
                return (False, "forbidden_lemma")

        # Check ALL lemmas against narration artifacts
        for lemma in lemmas:
            if lemma in SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT:
                return (False, "narration_artifact")

        # Persona leakage check
        if self._graph._persona_only_lemmas and primary_lemma in self._graph._persona_only_lemmas:
            return (False, "persona_only_lemma")

        # Story grounding check
        if is_new_node and self._graph._story_lemmas is not None:
            ed_stem = (
                primary_lemma[:-2]
                if primary_lemma.endswith("ed") and len(primary_lemma) > 2
                else None
            )
            ing_stem = (
                primary_lemma[:-3]
                if primary_lemma.endswith("ing") and len(primary_lemma) > 3
                else None
            )
            is_story_grounded = (
                primary_lemma in self._graph._story_lemmas
                or (ed_stem and ed_stem in self._graph._story_lemmas)
                or (ing_stem and ing_stem in self._graph._story_lemmas)
            )
            is_inferred = provenance == NodeProvenance.INFERRED_FROM_STORY

            if (
                node_type != NodeType.PROPERTY
                and not is_story_grounded
                and not is_inferred
            ):
                return (False, "not_story_grounded")

        return (True, None)

    def is_forbidden_lemma(self, lemma: str) -> bool:
        """
        Check if a lemma is blocked by any semantic category.

        Uses SemanticCategory.is_blocked() for classification.
        """
        return SemanticCategory.is_blocked(lemma)

    def is_story_grounded(self, lemma: str) -> bool:
        if self._graph._story_lemmas is None:
            return True

        lemma_lower = lemma.lower()
        ed_stem = (
            lemma_lower[:-2]
            if lemma_lower.endswith("ed") and len(lemma_lower) > 2
            else None
        )
        ing_stem = (
            lemma_lower[:-3]
            if lemma_lower.endswith("ing") and len(lemma_lower) > 3
            else None
        )

        return (
            lemma_lower in self._graph._story_lemmas
            or (ed_stem and ed_stem in self._graph._story_lemmas)
            or (ing_stem and ing_stem in self._graph._story_lemmas)
        )

    def validate_explicit_marking(self, primary_lemma: str) -> bool:
        """
        Validate whether an existing node can be marked as explicit in the current sentence.

        For existing nodes, we only allow explicit marking if:
        - No sentence lemmas filter is set, OR
        - The primary lemma appears in the current sentence's lemmas

        This prevents "hallucinating" explicit mentions of concepts
        that aren't actually mentioned in the current sentence text.

        Args:
            primary_lemma: The primary lemma of the node to mark

        Returns:
            True if explicit marking is allowed, False otherwise
        """
        if self._graph._current_sentence_lemmas is None:
            return True
        return primary_lemma.lower() in self._graph._current_sentence_lemmas

    def set_provenance_gate(
        self,
        story_lemmas: Set[str],
        persona_only_lemmas: Optional[Set[str]] = None,
    ) -> None:
        """
        Configure the provenance gate for node creation.

        Args:
            story_lemmas: Set of lemmas from story text (valid for node creation)
            persona_only_lemmas: Set of lemmas ONLY in persona (must be blocked)
        """
        self._graph._story_lemmas = {s.lower() for s in story_lemmas}
        self._graph._persona_only_lemmas = (
            {s.lower() for s in persona_only_lemmas} if persona_only_lemmas else set()
        )

    def sanity_check_provenance(
        self,
        story_lemmas: set,
        persona_only_lemmas: set,
    ) -> list:
        """
        AMoC v4 PROVENANCE SANITY CHECK: Detect potential persona leakage.

        Per AMoC v4 paper: Nodes must come from STORY TEXT only.
        Persona influences salience (weights), never content (nodes/edges).

        Args:
            story_lemmas: Set of lemmas from the story text
            persona_only_lemmas: Set of lemmas unique to persona (not in story)

        Returns:
            List of warning strings for any detected violations.
            Empty list if all nodes pass provenance check.
        """
        from amoc.graph.node import NodeProvenance

        warnings = []

        for node in self._graph.nodes:
            for lemma in node.lemmas:
                lemma_lower = lemma.lower()

                if lemma_lower in persona_only_lemmas:
                    warnings.append(
                        f"PROVENANCE VIOLATION: Node '{node.get_text_representer()}' "
                        f"contains lemma '{lemma_lower}' which appears ONLY in persona, "
                        f"not in story text. Provenance: {node.provenance}"
                    )

                elif lemma_lower not in story_lemmas:
                    if node.provenance == NodeProvenance.STORY_TEXT:
                        if any(lemma not in story_lemmas for lemma in node.lemmas):
                            logging.warning(
                                "PROVENANCE WARNING: Node '%s' lemma(s) %s not found in story lemma set.",
                                node.get_text_representer(),
                                node.lemmas,
                            )
        return warnings
