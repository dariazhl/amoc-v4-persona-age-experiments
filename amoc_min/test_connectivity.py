"""The graph is a connected unit by construction.

Connectivity is not repaired after the fact: it follows from what `add` and
the attachment rule admit. These tests pin that down, and they pin the one
accepted exception, so neither is undone by accident. No LLM: everything here
is structural.

    cd .. && venv/bin/python3 -m pytest amoc_min/test_connectivity.py -q
"""
import json
import os

import pytest

from . import config as C
from . import engine
from .graph import Graph

RUN = ("results/Qwen3-30b/final_plots/nov_11/amoc-min-2/run_298395"
       "/story_analysis.json")


def tri(s, r, o, frm=()):
    return {"triplet": [s, r, o], "from": list(frm)}


def components(g):
    return [sorted(n.lemma for n in c) for c in g.components()]


# -- add() ---------------------------------------------------------------
def test_self_loop_does_not_mint_an_orphan():
    g = Graph()
    assert g.add("dragon", "appeared as", "dragon", "text", 1) is None
    assert g.nodes == {}, "a rejected self-loop must leave no node behind"


# -- the attachment rule -------------------------------------------------
def test_floating_pair_is_never_admitted():
    """Both endpoints new and nothing joins them to the graph: dropped.

    This is the S7 regression. `pursuit -is a-> chase` used to be admitted
    because its `from` was empty and `all([])` is True.
    """
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    out = engine._attaching([tri("pursuit", "is a", "chase")], set(g.nodes))
    assert out == []


def test_anchored_inference_is_admitted():
    """`knight -is on-> pursuit` used to be REJECTED, because its `from` was
    ['hurried after'] -- a relation label, never a node."""
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    out = engine._attaching(
        [tri("knight", "is on", "pursuit", ["hurried after"])], set(g.nodes))
    assert [c["triplet"] for c in out] == [["knight", "is on", "pursuit"]]


def test_chain_extends_once_its_anchor_lands():
    """Order must not matter: the floating pair is listed first, and is still
    admitted after the triplet that anchors it."""
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    out = engine._attaching([tri("pursuit", "is a", "chase"),
                             tri("knight", "is on", "pursuit")],
                            set(g.nodes))
    assert [c["triplet"][0] for c in out] == ["knight", "pursuit"]


def test_declined_anchor_prunes_its_subtree():
    """A reader who does not draw `knight -> pursuit` cannot receive
    `pursuit -> chase`, so no reader ends up with a satellite."""
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    drawn = [tri("pursuit", "is a", "chase")]          # anchor declined
    assert engine._attaching(drawn, set(g.nodes)) == []


def test_attachment_is_to_the_main_component_not_any_node():
    """A node off the main graph must not anchor a cluster onto itself.

    This is the S3 regression: the stranded `dragon` grew
    {cave, dragon, fire, legend} when "some node" was enough.
    """
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    g.node("dragon", "text", 3)                        # stated, no relation
    main = {n.key for n in g.components()[0]}
    assert engine._attaching([tri("dragon", "lives in", "cave")], main) == []
    # ... whereas "any node" would have let it in
    assert engine._attaching([tri("dragon", "lives in", "cave")],
                             set(g.nodes)) != []


# -- carry-over ----------------------------------------------------------
def test_a_concept_survives_a_sentence_that_cannot_reach_it():
    """"Suddenly, a dragon appeared." states one peripheral word.

    Activation is seeded only from what a sentence states, so without
    carry-over every other concept drops to 0 the moment a sentence names
    something far from them -- `knight` fell 5.0 -> 0.0 at S3 where the
    reference holds it at 2.5. Carry-over is what the reader still has in
    mind, and it is the only thing that can produce that 2.5: no distance
    rule reaches `knight` from `dragon`.
    """
    g = Graph()
    g.add("knight", "rode through", "forest", "text", 1)
    hist = {}
    # S1 and S2 state `knight`; S3 states only `dragon`, which is not even
    # in this graph, so nothing is reachable at all
    for i, stated in enumerate([[("knight",)], [("knight",)], [("dragon",)]], 1):
        col = g.activation(stated, i, hist)
        for k in set(hist) | set(col):
            hist.setdefault(k, []).append(col.get(k, 0.0))
    assert col.get("knight") == pytest.approx(2.5), col
    assert C.CARRYOVER_DECAY > 0, "set CARRYOVER_DECAY=0 to disable"


# -- the invariant, on the real run --------------------------------------
@pytest.mark.skipif(not os.path.exists(RUN), reason="run artefacts not synced")
def test_real_run_is_connected_except_for_the_stated_singleton():
    passes = json.load(open(RUN))
    snaps = []
    engine.read(passes, llm=None, persona=None, snapshots=snaps)

    broken = {}
    for i, snap in enumerate(snaps, 1):
        adj = {}
        for e in snap["edges"]:                        # every edge: a decayed
            adj.setdefault(e["from"], set()).add(e["to"])   # edge is drawn in
            adj.setdefault(e["to"], set()).add(e["from"])   # grey, so it is
        seen, comps = set(), []                             # not an island
        for n in [x["lemma"] for x in snap["nodes"]]:
            if n in seen:
                continue
            stack, c = [n], []
            seen.add(n)
            while stack:
                x = stack.pop()
                c.append(x)
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            comps.append(sorted(c))
        if len(comps) > 1:
            broken[i] = sorted(comps, key=lambda c: -len(c))[1:]

    # A multi-node island is always a bug. A single STATED node may stand
    # alone -- "Suddenly, a dragon appeared." -- which carry-over and the
    # hub-first prompt are there to fix at generation time, so this must not
    # assert that it still happens. See Graph.components.
    for i, islands in broken.items():
        stated = {" ".join(k) for k in passes[i - 1]["stated"]}
        for isl in islands:
            assert len(isl) == 1, f"S{i}: multi-node island {isl}"
            assert isl[0] in stated, f"S{i}: unstated node {isl[0]} is alone"
