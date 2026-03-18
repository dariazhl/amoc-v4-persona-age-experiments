import re
from typing import List, Dict

_POSITIVE_WORDS: List[str] = [
    "good",
    "great",
    "happy",
    "love",
    "enjoy",
    "excited",
    "optimistic",
    "positive",
    "amazing",
    "fantastic",
    "wonderful",
    "proud",
    "satisfied",
    "confident",
    "hopeful",
]

_NEGATIVE_WORDS: List[str] = [
    "bad",
    "sad",
    "hate",
    "angry",
    "upset",
    "worried",
    "anxious",
    "negative",
    "terrible",
    "awful",
    "depressed",
    "lonely",
    "frustrated",
    "guilty",
    "ashamed",
]


def simple_sentiment_score(text: str) -> float:
    if not text:
        return 0.0

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0

    pos = sum(t in _POSITIVE_WORDS for t in tokens)
    neg = sum(t in _NEGATIVE_WORDS for t in tokens)
    return (pos - neg) / len(tokens)


def compute_lexical_metrics(text: str) -> Dict[str, float]:
    if not text:
        return {"lexical_ttr": 0.0, "lexical_avg_word_len": 0.0}

    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return {"lexical_ttr": 0.0, "lexical_avg_word_len": 0.0}

    unique_tokens = len(set(tokens))
    return {
        "lexical_ttr": unique_tokens / len(tokens),
        "lexical_avg_word_len": sum(len(t) for t in tokens) / len(tokens),
    }
