from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from amoc.graph.node import Node

from enum import Enum
from typing import List, Dict, Set


class NodeType(Enum):
    CONCEPT = 1
    PROPERTY = 2
    EVENT = 3  # EVENT/PROCESS mediation nodes


class NodeRole(Enum):
    ACTOR = 1  # Persons, agents (typically nsubj)
    OBJECT = 2  # Things (typically dobj)
    PROPERTY = 3  # Adjectives (attributes)
    SETTING = 4  # Locations, environments (typically pobj)


class NodeSource(Enum):
    TEXT_BASED = 1
    INFERENCE_BASED = 2


class NodeProvenance(Enum):
    STORY_TEXT = 1  # Node derived from story sentence tokens
    INFERRED_FROM_STORY = 2  # Node inferred by LLM but validated against story


class Node:
    def __init__(
        self,
        lemmas: List[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
        score: int,
        origin_sentence: Optional[int] = None,
        provenance: Optional[NodeProvenance] = None,
        node_role: Optional[NodeRole] = None,
    ) -> None:
        self.lemmas: List[str] = [lemma.lower() for lemma in lemmas]
        actual_text_l = (actual_text or "").lower()
        self.actual_texts: Dict[str, int] = {actual_text_l: 1}
        self.node_type: NodeType = node_type
        self.node_source: NodeSource = node_source
        self.score = score
        self.edges: List["Edge"] = []
        self.origin_sentence: Optional[int] = origin_sentence
        self.provenance: NodeProvenance = provenance or NodeProvenance.STORY_TEXT
        self.first_seen_sentence: Optional[int] = origin_sentence
        self.explicit_sentences: Set[int] = (
            {origin_sentence} if origin_sentence is not None else set()
        )
        # ever_explicit: True if the node has ever been explicit in any sentence
        self.ever_explicit: bool = origin_sentence is not None

        if self.origin_sentence is not None:
            self.explicit_sentences.add(self.origin_sentence)
            self.ever_explicit = True

        self.node_role: Optional[NodeRole] = node_role

        if self.node_type == NodeType.PROPERTY and self.node_role is None:
            self.node_role = NodeRole.PROPERTY

        self.activation_score: int = 0
        self.active: bool = False

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

    def is_setting(self) -> bool:
        return self.node_role == NodeRole.SETTING

    def mark_explicit_in_sentence(self, sentence_id: int) -> None:
        self.explicit_sentences.add(sentence_id)
        self.ever_explicit = True

    def is_explicit_in_sentence(self, sentence_id: int) -> bool:
        return sentence_id in self.explicit_sentences

    def is_carryover_in_sentence(self, sentence_id: int) -> bool:
        return sentence_id not in self.explicit_sentences

    def get_text_representer(self) -> str:
        DETERMINERS = {"the", "a", "an"}
        best = max(self.actual_texts, key=self.actual_texts.get)
        # Safety: strip leading determiners
        words = best.split()
        while words and words[0].lower() in DETERMINERS:
            words.pop(0)
        return " ".join(words) if words else best

    def __str__(self) -> str:
        return f"{self.get_text_representer()} ({self.node_type.name}, {self.node_source.name}, {self.score})"

    def __repr__(self) -> str:
        return str(self)
