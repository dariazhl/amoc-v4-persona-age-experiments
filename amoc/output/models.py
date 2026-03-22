from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DecayDecision:
    triplet: Tuple[str, str, str]
    score: int  # 0=remove, 1=decay, 2=maintain, 3=reinforce
    action: str  # "removed", "decayed", "maintained", "reinforced"
    was_connectivity_critical: bool
    reasoning: str = ""


@dataclass
class SentenceTripletRecord:
    sentence_index: int
    sentence_text: str
    explicit_text_triplets: List[Tuple[str, str, str]]
    explicit_inferred_triplets: List[Tuple[str, str, str]]
    carryover_text_triplets: List[Tuple[str, str, str]]
    carryover_inferred_triplets: List[Tuple[str, str, str]]
    inactive_text_triplets: List[Tuple[str, str, str]]
    inactive_inferred_triplets: List[Tuple[str, str, str]]
    decay_decisions: List[DecayDecision] = field(default_factory=list)
