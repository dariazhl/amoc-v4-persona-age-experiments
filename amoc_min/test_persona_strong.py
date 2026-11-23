"""The stronger-persona switches (branch feature/persona-effect-stronger).

Offline: a fake model answers each prompt by its wording, so every step of
the new reader path can be checked without a server.

    python -m pytest amoc_min/test_persona_strong.py
"""
import json

import pytest

from . import config as C
from . import nlp as N
from .engine import read
from .graph import Graph

PERSONA = "A lively ten-year-old who reviews books on her YouTube channel"


def passes():
    """Two sentences, analysed the way analyze_story stores them. The shared
    (persona-free) list offers one inference: knight -- carries -- sword."""
    out = []
    for i, (sent, trips, inferred) in enumerate([
            ("A young knight rode through the forest.",
             [["knight", "is", "young"], ["knight", "rode through", "forest"]],
             [{"triplet": ["knight", "carries", "sword"], "from": []}]),
            ("The knight saw a dragon.",
             [["knight", "saw", "dragon"]], [])], 1):
        words = N.content_words(sent)
        out.append({"sentence": sent, "words": words, "rejected": [],
                    "stated": [list(k) for _, k, _ in words],
                    "triplets": trips, "inferred": inferred, "context": sent})
    return out


class Fake:
    """Answers by prompt wording and records every prompt it was sent."""
    def __init__(self, license_ok=True):
        self.prompts, self.license_ok = [], license_ok

    def ask(self, p):
        self.prompts.append(p)
        if "checking whether candidate inferences follow" in p:        # NEUTRAL_LICENSE
            n = p.count(" -- ")
            return json.dumps({"judgements": [{"n": i, "inferable": self.license_ok}
                                              for i in range(n)]})
        if "additional concepts and properties" in p:                  # INFER_RELATIONSHIPS
            return json.dumps({"inferences": [
                {"triplet": ["knight", "rides", "horse"], "from": []},
                {"triplet": ["knight", "watches", "youtube"], "from": []}]})
        if "Name concepts and properties" in p:                        # INFER_OBJECTS
            return json.dumps({"concepts": ["horse", "youtube"], "properties": []})
        if '"makes"' in p:                                             # selection prompts
            n = p.count(" -- ")
            return json.dumps({"judgements": [{"n": i, "makes": True} for i in range(n)]})
        if "scores" in p or "SCORE" in p:                              # NARRATIVE_RELEVANCE
            return json.dumps({"scores": [], "context_shift": False})
        return "{}"


@pytest.fixture
def switches(monkeypatch):
    for k in ("PERSONA_GENERATES", "STRICT_SELECTION", "READER_CAPACITY", "PERSONA_IN_DECAY"):
        monkeypatch.setattr(C, k, False)
    return lambda **kw: [monkeypatch.setattr(C, k, v) for k, v in kw.items()]


def inferred_edges(g):
    return {(e.src.lemma, e.label, e.dst.lemma) for e in g.edges if e.source == "inferred"}


def test_default_reader_only_picks_from_the_shared_list(switches):
    llm = Fake()
    _, counts, _, g = read(passes(), llm=llm, persona=PERSONA)
    assert not any("Name concepts and properties" in p for p in llm.prompts)
    assert {t[2] for t in inferred_edges(g)} == {"sword"}
    assert "proposed" not in counts[0]


def test_generating_reader_proposes_its_own_inferences(switches):
    switches(PERSONA_GENERATES=True)
    llm = Fake()
    _, counts, _, g = read(passes(), llm=llm, persona=PERSONA)
    gen = [p for p in llm.prompts if "Name concepts and properties" in p]
    assert gen and all(PERSONA in p for p in gen)                    # step 2: persona-conditioned
    objs = {t[2] for t in inferred_edges(g)}
    assert "horse" in objs                                            # the reader's own inference
    assert "sword" not in objs                                        # not the shared list
    assert counts[0]["proposed"] == 2


def test_persona_word_ban_drops_concepts_from_the_persona(switches):
    switches(PERSONA_GENERATES=True)
    _, counts, _, g = read(passes(), llm=Fake(), persona=PERSONA)
    assert "youtube" not in {t[2] for t in inferred_edges(g)}         # 'youtube' is only in the persona
    assert counts[0]["banned"] == 1


def test_neutral_judge_never_sees_the_persona_and_can_veto(switches):
    switches(PERSONA_GENERATES=True)
    llm = Fake(license_ok=False)
    _, counts, _, g = read(passes(), llm=llm, persona=PERSONA)
    lic = [p for p in llm.prompts if "checking whether candidate inferences follow" in p]
    assert lic and not any(PERSONA in p or "ten-year-old" in p for p in lic)
    assert inferred_edges(g) == set()
    assert counts[0]["licensed"] == 0


def test_selection_is_skipped_after_generation_unless_strict(switches):
    switches(PERSONA_GENERATES=True)
    llm = Fake()
    read(passes(), llm=llm, persona=PERSONA)
    assert not any('"makes"' in p for p in llm.prompts)
    switches(STRICT_SELECTION=True)
    llm = Fake()
    read(passes(), llm=llm, persona=PERSONA)
    sel = [p for p in llm.prompts if '"makes"' in p]
    assert sel and all("SOME readers make" in p for p in sel)


def test_strict_selection_replaces_the_default_prompt(switches):
    switches(STRICT_SELECTION=True)
    llm = Fake()
    read(passes(), llm=llm, persona=PERSONA)
    sel = [p for p in llm.prompts if '"makes"' in p]
    assert sel and all("SOME readers make" in p for p in sel)
    assert not any("The text supports these inferred relationships" in p for p in sel)


def test_baseline_reader_keeps_the_shared_list_in_every_mode(switches):
    switches(PERSONA_GENERATES=True, STRICT_SELECTION=True)
    llm = Fake()
    _, _, _, g = read(passes(), llm=llm, persona=None)
    assert {t[2] for t in inferred_edges(g)} == {"sword"}
    assert not any("Name concepts and properties" in p for p in llm.prompts)


def test_capacity_defaults_reproduce_the_model(switches):
    g = Graph()
    assert g.capacity == 1.0
    assert g.max_carryover == C.MAX_CARRYOVER_NODES
    assert g.carryover_decay == C.CARRYOVER_DECAY
    assert g.llm_carryover_cap == C.LLM_CARRYOVER_CAP
    assert C.capacity_for("primary") == 1.0                           # switch off -> no effect


def test_capacity_shrinks_with_regime_when_on(switches):
    switches(READER_CAPACITY=True)
    caps = [C.capacity_for(r) for r in ("primary", "secondary", "highschool", "university")]
    assert caps == sorted(caps) and caps[-1] == 1.0
    assert C.capacity_for(None) == 1.0                                # baseline reader
    small = Graph(capacity=C.capacity_for("primary"))
    assert small.max_carryover < C.MAX_CARRYOVER_NODES
    assert small.carryover_decay < C.CARRYOVER_DECAY


def test_apply_persona_cfg_reaches_a_fresh_worker(switches):
    C.apply_persona_cfg({"persona_generates": True, "reader_capacity": True,
                         "strict_selection": False, "persona_decay": True})
    assert C.PERSONA_GENERATES and C.READER_CAPACITY and C.PERSONA_IN_DECAY
    assert not C.STRICT_SELECTION
