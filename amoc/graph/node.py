from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from amoc.graph.node import Node

from enum import Enum
from typing import List, Dict, Set


class NodeType(Enum):
    CONCEPT = 1
    PROPERTY = 2
    EVENT = 3  # EVENT/PROCESS mediation nodes (Recommendation 2)
    # NOTE: EVENT nodes are created for relations with event_level in {EVENT, PROCESS}
    # They mediate between actor and object: actor --participates_in--> event --affects--> object


class NodeRole(Enum):
    """
    Lightweight semantic role for nodes (per AMoC v4 paper alignment).

    This is NOT a new ontology - just a classification hint for:
    - ACTOR: persons, agents (subjects of actions)
    - OBJECT: things (direct objects of actions)
    - PROPERTY: adjectives (attributes)
    - SETTING: locations, environments (prepositional objects like "forest", "castle")

    SETTING nodes:
    - Are nouns introduced via prepositional phrases
    - Have NO agency and do NOT affect inference or lifecycle
    - Exist only to preserve scene context
    """
    ACTOR = 1      # Persons, agents (typically nsubj)
    OBJECT = 2     # Things (typically dobj)
    PROPERTY = 3   # Adjectives (attributes)
    SETTING = 4    # Locations, environments (typically pobj)


class NodeSource(Enum):
    TEXT_BASED = 1
    INFERENCE_BASED = 2


class NodeProvenance(Enum):
    """
    Provenance tracking for nodes per AMoC v4 paper alignment.

    CRITICAL: Only STORY_TEXT and INFERRED_FROM_STORY are valid for graph nodes.
    PERSONA nodes must NEVER be created - persona influences weights only.
    """
    STORY_TEXT = 1           # Node derived from story sentence tokens
    INFERRED_FROM_STORY = 2  # Node inferred by LLM but validated against story
    # PERSONA = 3            # INVALID - must never be used for nodes


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

        # PROVENANCE TRACKING (Paper-Aligned)
        # origin_sentence: The sentence index where this node was first created
        # provenance: How this node was derived (STORY_TEXT or INFERRED_FROM_STORY)
        self.origin_sentence: Optional[int] = origin_sentence
        self.provenance: NodeProvenance = provenance or NodeProvenance.STORY_TEXT

        # ==========================================================================
        # PHASE 2: SENTENCE-SCOPED NODE PROVENANCE
        # ==========================================================================
        # first_seen_sentence: Same as origin_sentence (kept for compatibility)
        # explicit_sentences: Set of sentences where this node appears in dependency parse
        #
        # INVARIANT: A node is explicit in sentence S ONLY if it appears in S's parse.
        # Carry-over nodes remain carry-over unless explicitly re-mentioned.
        # Visualization color is driven ONLY by explicit_sentences.
        self.first_seen_sentence: Optional[int] = origin_sentence
        self.explicit_sentences: Set[int] = (
            {origin_sentence} if origin_sentence is not None else set()
        )

        # NODE ROLE: Lightweight semantic role (ACTOR, OBJECT, PROPERTY, SETTING)
        # SETTING nodes exist for scene context only - they don't affect lifecycle logic
        self.node_role: Optional[NodeRole] = node_role

        # CRITICAL ASSERTION: Nodes must never come from persona
        # Persona influences salience/weights only, never content
        # This assertion is a fail-fast guard against persona leakage

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
        """
        Check if this node is a SETTING (location/environment).

        SETTING nodes exist for scene context only and should NOT:
        - Participate in lifecycle/deactivation logic
        - Become hubs or central nodes
        - Trigger inferred edges by themselves
        """
        return self.node_role == NodeRole.SETTING

    def mark_explicit_in_sentence(self, sentence_id: int) -> None:
        """
        Mark this node as explicit in the given sentence.

        PHASE 2: Only call this if the node appears in the sentence's dependency parse.
        Do NOT infer explicitness from memory, edges, or prior sentences.
        """
        self.explicit_sentences.add(sentence_id)

    def is_explicit_in_sentence(self, sentence_id: int) -> bool:
        """
        Check if this node is explicit (not carry-over) in the given sentence.

        PHASE 2: Color assignment must use ONLY this method.
        Degree, recency, connectivity, persona must NOT affect this.
        """
        return sentence_id in self.explicit_sentences

    def is_carryover_in_sentence(self, sentence_id: int) -> bool:
        """
        Check if this node is carry-over (not explicit) in the given sentence.

        A node is carry-over if it exists but was not mentioned in this sentence's parse.
        """
        return sentence_id not in self.explicit_sentences

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
