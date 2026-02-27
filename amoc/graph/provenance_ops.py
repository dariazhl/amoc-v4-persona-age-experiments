"""
Provenance operations for Graph.

CANONICAL VALIDATION AUTHORITY for node creation.

All validation/filtering logic lives here:
- Blocklists (forbidden lemmas, garbage lemmas, narration artifacts)
- Story grounding checks
- Persona leakage prevention
- Provenance sanity checks

Graph.add_or_get_node() delegates validation to this class.
"""

from typing import TYPE_CHECKING, Set, Optional, List, Tuple
import logging
import re

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import NodeProvenance, NodeType


class ProvenanceOps:
    """
    Canonical validation authority for node creation.

    Responsibilities:
    - Validate lemmas against blocklists
    - Ensure story grounding
    - Prevent persona leakage
    - Provenance sanity checks
    """

    FORBIDDEN_NODE_LEMMAS: set[str] = {
        "student", "persona", "relation", "context", "object", "place",
        "story", "narrative", "sentence", "edge", "node", "property",
        "label", "target", "source", "pronoun", "noun", "user",
    }

    NARRATION_ARTIFACT_LEMMAS: set[str] = {
        "text", "sentence", "paragraph", "mention", "mentions", "narration", "story",
    }

    GARBAGE_LEMMAS: set[str] = {
        "edge", "node", "relation", "property", "label", "target",
        "source", "t", "type", "person", "approach", "kind", "thing",
    }

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def validate_node_creation(
        self,
        lemmas: List[str],
        node_type: "NodeType",
        provenance: Optional["NodeProvenance"] = None,
        is_new_node: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        from amoc.graph.node import NodeType, NodeProvenance

        if not lemmas or not lemmas[0]:
            return (False, "empty_lemmas")

        primary_lemma = lemmas[0].lower()

        if len(primary_lemma) <= 1:
            return (False, "lemma_too_short")

        if primary_lemma in self.GARBAGE_LEMMAS:
            return (False, "garbage_lemma")

        if node_type != NodeType.EVENT and not re.match(r"^[a-zA-Z]+$", lemmas[0]):
            return (False, "invalid_characters")

        if any(lemma in self.FORBIDDEN_NODE_LEMMAS for lemma in lemmas):
            return (False, "forbidden_lemma")

        if any(lemma in self.NARRATION_ARTIFACT_LEMMAS for lemma in lemmas):
            return (False, "narration_artifact")

        if self._graph._persona_only_lemmas and primary_lemma in self._graph._persona_only_lemmas:
            return (False, "persona_only_lemma")

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
        lemma_lower = lemma.lower()
        return (
            lemma_lower in self.FORBIDDEN_NODE_LEMMAS
            or lemma_lower in self.NARRATION_ARTIFACT_LEMMAS
            or lemma_lower in self.GARBAGE_LEMMAS
        )

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
