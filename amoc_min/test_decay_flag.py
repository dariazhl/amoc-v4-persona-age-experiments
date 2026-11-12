"""The --llm-decay flag selects how a relation's age is reset.

    text       (default)  the relation's own word appears in the sentence
    relevance  (--llm-decay)  NARRATIVE_RELEVANCE scored the edge 2

Only activation reads this, never extraction, so both rules share one
story_analysis.json and two runs into the same --out differ by the rule alone.

    cd .. && venv/bin/python3 -m pytest amoc_min/test_decay_flag.py -q
"""
import pytest

from . import config as C
from .graph import Graph


@pytest.fixture
def ageing():
    """Restore the module-level default, which run.py sets per process."""
    before = C.RELATION_AGEING
    yield lambda mode: setattr(C, "RELATION_AGEING", mode)
    C.RELATION_AGEING = before


def ride_at_s3(relevance, drop_to=None):
    """Activation of the relation label `ride` at S3.

    Steps S1..S3 carrying `history` forward exactly as engine.read does. That
    matters: activation snaps a token back to 5.0 when its PREVIOUS value was
    0, so a fresh history per sentence hides ageing entirely.

    S1 states the relation's own word; S2 and S3 do not.
    """
    g = Graph()
    e = g.add("knight", "rode through", "forest", "text", 1)
    e.relevance = relevance
    hist, col = {}, {}
    for i, words in enumerate([["knight", "ride", "forest"],
                               ["knight", "country"],
                               ["knight", "dragon"]], 1):
        if drop_to is not None and i == 2:
            e.relevance = drop_to          # the model lets the edge go
        col = g.activation([("knight",)], i, hist, sentence_lemmas=words)
        for t in set(hist) | set(col):
            hist.setdefault(t, []).append(col.get(t, 0.0))
    return col.get("ride", 0.0)


def test_text_rule_ages_when_the_word_is_absent(ageing):
    ageing(C.AGEING_TEXT)
    # `ride` is stated at S1 and not after, so by S3 it has aged two
    # sentences: 5 - SENIORITY_WEIGHT * 2 = 1.0
    assert ride_at_s3(relevance=2) == pytest.approx(1.0)


def test_llm_rule_keeps_a_relation_the_model_still_calls_relevant(ageing):
    ageing(C.AGEING_LLM)
    # the word is gone from the text, but the model still scores the edge 2,
    # so it does not age at all -- the opposite of the text rule
    assert ride_at_s3(relevance=2) == pytest.approx(5.0)


def test_llm_rule_ages_a_relation_the_model_has_let_go(ageing):
    ageing(C.AGEING_LLM)
    # scored 2 at S1, dropped to FADING at S2: it ages from S1 like any other
    assert ride_at_s3(relevance=2, drop_to=1) == pytest.approx(1.0)


def test_the_two_rules_disagree(ageing):
    """If they ever agree everywhere the flag is pointless, so pin it."""
    out = {}
    for mode in (C.AGEING_TEXT, C.AGEING_LLM):
        ageing(mode)
        out[mode] = ride_at_s3(relevance=2)
    assert out[C.AGEING_TEXT] != out[C.AGEING_LLM], out


def test_flag_names_exist_for_run_py_to_select_between():
    assert {C.AGEING_TEXT, C.AGEING_LLM} == {"text", "relevance"}
    assert C.RELATION_AGEING == C.AGEING_TEXT, "text must stay the default"
