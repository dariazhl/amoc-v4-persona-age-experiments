"""
Provenance operations for Graph.

Contains methods for provenance gate configuration and sanity checking.
Moved from Graph class to separate topology from provenance logic.
"""

from typing import TYPE_CHECKING, Set, Optional
import logging

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import NodeProvenance


class ProvenanceOps:
    """Provenance operations that can be applied to a Graph."""

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

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
