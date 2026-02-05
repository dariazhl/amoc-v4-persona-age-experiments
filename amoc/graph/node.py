from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from amoc.graph.node import Node

from enum import Enum
from typing import List, Dict


class NodeType(Enum):
    CONCEPT = 1
    PROPERTY = 2
    # NOTE: RELATION type removed for AMoCv4 surface-relation format compliance.
    # Verbs are represented as direct labeled edges, not intermediate nodes.


class NodeSource(Enum):
    TEXT_BASED = 1
    INFERENCE_BASED = 2


class Node:
    def __init__(
        self,
        lemmas: List[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
        score: int,
    ) -> None:
        self.lemmas: List[str] = [lemma.lower() for lemma in lemmas]
        actual_text_l = (actual_text or "").lower()
        self.actual_texts: Dict[str, int] = {actual_text_l: 1}
        self.node_type: NodeType = node_type
        self.node_source: NodeSource = node_source
        self.score = score
        self.edges: List["Edge"] = []

    def __eq__(self, other: "Node") -> bool:
        return self.lemmas == other.lemmas

    def __hash__(self) -> int:
        return hash(tuple(self.lemmas))

    def add_actual_text(self, actual_text: str) -> None:
        actual_text_l = (actual_text or "").lower()
        if actual_text_l in self.actual_texts:
            self.actual_texts[actual_text_l] += 1
        else:
            self.actual_texts[actual_text_l] = 1

    def get_text_representer(self) -> str:
        """
        Get the most frequent text representation for this node.

        Per AMoC v4 paper: Node labels are single lowercase lemmas like "country",
        NEVER "the country". This method strips leading determiners as a safety net.
        """
        DETERMINERS = {"the", "a", "an"}
        best = max(self.actual_texts, key=self.actual_texts.get)
        # Safety: strip leading determiners (should already be canonicalized, but ensure)
        words = best.split()
        while words and words[0].lower() in DETERMINERS:
            words.pop(0)
        return " ".join(words) if words else best

    def __str__(self) -> str:
        return f"{self.get_text_representer()} ({self.node_type.name}, {self.node_source.name}, {self.score})"

    def __repr__(self) -> str:
        return str(self)
