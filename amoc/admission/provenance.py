from typing import TYPE_CHECKING, Set, Optional, List, Tuple
import logging
import re
from amoc.core.node import NodeType, NodeProvenance

if TYPE_CHECKING:
    from amoc.core.graph import Graph


class ProvenanceValidation:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def validate_node_creation(
        self,
        lemmas: List[str],
        node_type: "NodeType",
        provenance: Optional["NodeProvenance"] = None,
        is_new_node: bool = True,
    ) -> Tuple[bool, Optional[str]]:

        if not lemmas or not lemmas[0]:
            return (False, "empty_lemmas")

        primary_lemma = lemmas[0].lower()
        all_lemmas = [l.lower() for l in lemmas if l]

        if len(primary_lemma) <= 1:
            return (False, "lemma_too_short")

        if not re.match(r"^[a-zA-Z]+$", lemmas[0]):
            return (False, "invalid_characters")

        if (
            self._graph._persona_only_lemmas
            and primary_lemma in self._graph._persona_only_lemmas
        ):
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

    def passes_length_policy(
        self,
        lemma: str,
        node_type: Optional["NodeType"] = None,
        provenance: Optional["NodeProvenance"] = None,
    ) -> bool:
        if not lemma:
            return False
        lemma_lower = lemma.lower()
        if len(lemma_lower) <= 1:
            return False
        return True

    def validate_explicit_marking(self, primary_lemma: str) -> bool:
        if self._graph._current_sentence_lemmas is None:
            return True
        return primary_lemma.lower() in self._graph._current_sentence_lemmas

    def sanity_check_provenance(
        self,
        story_lemmas: set,
        persona_only_lemmas: set,
    ) -> list:
        warnings = []

        for node in self._graph.nodes:
            for lemma in node.lemmas:
                lemma_lower = lemma.lower()

                if lemma_lower in persona_only_lemmas:
                    warnings.append(
                        f"Node '{node.get_text_representer()}' "
                        f"contains lemma '{lemma_lower}' which appears ONLY in persona, "
                        f"not in story text. Provenance: {node.provenance}"
                    )

                elif lemma_lower not in story_lemmas:
                    if node.provenance == NodeProvenance.STORY_TEXT:
                        if any(lemma not in story_lemmas for lemma in node.lemmas):
                            logging.warning(
                                "Node '%s' lemma(s) %s not found in story lemma set.",
                                node.get_text_representer(),
                                node.lemmas,
                            )

        return warnings
