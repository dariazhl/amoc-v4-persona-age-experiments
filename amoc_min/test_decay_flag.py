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
    # sentences: 5 - SENIORITY_WEIGHT*2 = 3.0 at distance 0. Carry-over does
    # not reach it here -- S2 was 4.0, and 4.0*0.4 = 1.6 is under the 2.0
    # floor -- so ageing is what sets the value.
    assert ride_at_s3(relevance=2) == pytest.approx(3.0)


def test_llm_rule_relevance_keeps_a_relation_but_does_not_make_it_fresh(ageing):
    ageing(C.AGEING_LLM)
    # amoc-v5 semantics (LLM_AGE_RETAIN = 1.0, 2026-11-20): the model scoring
    # the edge 2 keeps it ACTIVE, but it still ages from its last mention --
    # it used to be held at 5.0, which is what made --llm-decay matrices 0/5.
    assert ride_at_s3(relevance=2) == pytest.approx(3.0)


def test_llm_rule_resets_a_relation_its_own_word_restates(ageing):
    """v5 resets seniority on an explicit re-mention, under either rule."""
    ageing(C.AGEING_LLM)
    g = Graph()
    e = g.add("knight", "rode through", "forest", "text", 1)
    e.relevance = 2
    hist, col = {}, {}
    for i, words in enumerate([["knight", "ride", "forest"],
                               ["knight", "country"],
                               ["knight", "ride"]], 1):   # S3 says `ride` again
        col = g.activation([("knight",)], i, hist, sentence_lemmas=words)
        for t in set(hist) | set(col):
            hist.setdefault(t, []).append(col.get(t, 0.0))
    assert col.get("ride", 0.0) == pytest.approx(5.0)


def test_llm_rule_ages_a_relation_the_model_has_let_go(ageing):
    ageing(C.AGEING_LLM)
    # scored 2 at S1, dropped to FADING at S2: it ages from S1 like any
    # other, landing where the text rule does
    assert ride_at_s3(relevance=2, drop_to=1) == pytest.approx(3.0)


def test_the_two_rules_disagree(ageing):
    """If they ever agree everywhere the flag is pointless, so pin it.

    Since ageing is shared (v5), the rules now differ in what the model's
    judgement REMOVES: a FADING (1) edge is deactivated under --llm-decay, as
    v5 decay.py does, and left alone under the text rule. A triangle, so no
    edge is a bridge that PROTECT_BRIDGES would keep."""
    import json
    from .engine import _reactivate

    class Fading:
        def ask(self, prompt):
            ns = [int(l.split(".")[0]) for l in prompt.splitlines()
                  if l.strip()[:1].isdigit() and "." in l]
            return json.dumps({"scores": [{"n": n, "score": 1} for n in ns]})

    forget = {}
    for mode in (C.AGEING_TEXT, C.AGEING_LLM):
        ageing(mode)
        g = Graph()
        e = g.add("knight", "rode through", "forest", "text", 1)
        g.add("knight", "rides", "horse", "text", 1)
        g.add("horse", "runs through", "forest", "text", 1)
        _reactivate(Fading(), g, list(g.nodes.values()), "ctx", [],
                    sentence_text="The knight met a dragon.")
        forget[mode] = e.forget             # fade() steps it down by one
    assert forget[C.AGEING_LLM] < forget[C.AGEING_TEXT], forget


def test_flag_names_exist_for_run_py_to_select_between():
    assert {C.AGEING_TEXT, C.AGEING_LLM} == {"text", "relevance"}
    assert C.RELATION_AGEING == C.AGEING_TEXT, "text must stay the default"


# -- the flag must move CONCEPT rows, not just relation rows ---------------
def forest_at_s3(relevance, mode):
    """Activation of the CONCEPT `forest` at S3.

    `forest` is stated at S1 and never again. S2 and S3 state only `knight`.
    Under the text rule it therefore ages out. Under the llm rule it is held
    up for as long as the model keeps calling the knight-forest relation
    CONTINUOUS -- that is the whole point of the flag.
    """
    C.RELATION_AGEING = mode
    g = Graph()
    e = g.add("knight", "rode through", "forest", "text", 1)
    e.relevance = relevance
    hist, col = {}, {}
    for i, stated in enumerate([[("knight",), ("forest",)],
                                [("knight",)], [("knight",)]], 1):
        col = g.activation(stated, i, hist, sentence_lemmas=["knight"])
        for k in set(hist) | set(col):
            hist.setdefault(k, []).append(col.get(k, 0.0))
    return col.get("forest", 0.0)


def test_llm_rule_ages_a_concept_the_model_still_calls_relevant(ageing):
    ageing(C.AGEING_LLM)
    # v5 semantics: relevance keeps the knight-forest edge traversable, but
    # `forest` still ages from its last mention, exactly as under the text
    # rule. (Pre-2026-11-20 it was held at 3.5, age 0, for as long as the
    # model said yes.)
    held = forest_at_s3(relevance=2, mode=C.AGEING_LLM)
    assert held == pytest.approx(forest_at_s3(relevance=2, mode=C.AGEING_TEXT))
    assert held < 3.5


def test_llm_rule_lets_a_concept_go_when_the_model_does(ageing):
    ageing(C.AGEING_LLM)
    # FADING, not continuous: the model is no longer holding it up
    assert forest_at_s3(relevance=1, mode=C.AGEING_LLM) < 5.0


def test_a_partial_age_retain_still_moves_concept_rows(ageing, monkeypatch):
    """The older regression, kept: whenever relevance is allowed to slow
    ageing (LLM_AGE_RETAIN < 1), it must reach CONCEPT rows, not only the 7
    relation labels among the reference's 20 tokens."""
    monkeypatch.setattr(C, "LLM_AGE_RETAIN", 0.0)
    out = {}
    for mode in (C.AGEING_TEXT, C.AGEING_LLM):
        ageing(mode)
        out[mode] = forest_at_s3(relevance=2, mode=mode)
    assert out[C.AGEING_TEXT] != out[C.AGEING_LLM], out


def test_the_llm_brakes_cannot_reach_the_score_rule(ageing, monkeypatch):
    """The two rules have to stay independently comparable.

    Every brake added to make the LLM rule standalone -- score-1 removal,
    the carry-over cap, partial age retention and bridge protection -- is
    gated on RELATION_AGEING being the LLM one.
    Moving any of them must leave a text-rule run bit-identical, otherwise
    an A/B between the two rules is measuring the brakes as well.
    """
    ageing(C.AGEING_TEXT)
    base = forest_at_s3(relevance=2, mode=C.AGEING_TEXT)
    for name, value in (("LLM_DROP_FADING", False),
                        ("LLM_CARRYOVER_CAP", 1),
                        ("LLM_AGE_RETAIN", 0.0),
                        ("PROTECT_BRIDGES", False)):
        monkeypatch.setattr(C, name, value)
        assert forest_at_s3(relevance=2, mode=C.AGEING_TEXT) == base, name


def test_llm_score_reads_the_matrix_off_the_ratings(ageing):
    """--llm-score: the 0-5 rating is the activation, for the relationship's
    own row and for both concepts; what the sentence states is 5; a rating
    of 0 switches the relationship off and takes its concepts with it."""
    ageing(C.AGEING_SCORE)
    g = Graph()
    e = g.add("knight", "rode through", "forest", "text", 1)
    e.rate(3, g.edge_forget)
    col = g.rated_activation([("knight",)])
    assert col["knight"] == 5.0
    assert col["forest"] == 3.0 and col["ride"] == 3.0
    e.rate(0, g.edge_forget)
    assert not e.active
    assert "forest" not in g.rated_activation([("dragon",)])


def test_persona_reaches_the_decay_call_only_when_asked(monkeypatch):
    """--persona-decay puts the reader in front of the what-stays-in-mind
    prompt, for both the 0/1/2 relevance call and the 0-5 rating. With the
    flag off, or for the baseline reader, the prompt is the same for all."""
    from . import engine
    seen = []

    class Spy:
        def ask(self, prompt):
            seen.append(prompt)
            return "{}"

    def prompts(persona):
        g = Graph()
        g.add("knight", "rode through", "forest", "text", 1)
        engine._rate(Spy(), g, "ctx", [], "A sentence.", persona=persona)
        engine._reactivate(Spy(), g, list(g.nodes.values()), "ctx", [],
                           "A sentence.", persona=persona)
        return seen[-2:]

    monkeypatch.setattr(C, "PERSONA_IN_DECAY", False)
    assert not any("a 9-year-old" in p for p in prompts("a 9-year-old"))
    monkeypatch.setattr(C, "PERSONA_IN_DECAY", True)
    assert all(p.startswith("Reader: a 9-year-old") for p in prompts("a 9-year-old"))
    assert not any(p.startswith("Reader:") for p in prompts(None))
