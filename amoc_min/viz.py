"""Per-sentence figures for one reader.

The layout is computed once from every triplet the story can produce --
stated and inferred alike -- so a node sits in the same place in all frames
whether or not this particular reader ever activates it. Positions are keyed
by lemma, which is what the activation columns are keyed by.
"""
import os

from . import nlp as N
from .plot import draw, layout


def _lemma(surface):
    return " ".join(N.lemmas(surface))


def make_plotter(out_dir, passes, persona_text, tag, model="", regime=None,
                 age=None):
    """Returns an on_sentence callback for engine.read()."""
    union = []
    for p in passes:
        for s, _, o in p["triplets"]:
            union.append((_lemma(s), "", _lemma(o)))
        for c in p["inferred"]:
            s, _, o = c["triplet"]
            union.append((_lemma(s), "", _lemma(o)))
    pos = layout(union)
    if persona_text is None:
        who = "baseline reader (all candidate inferences)"
    else:
        who = persona_text + (f"  ·  {regime}" if regime else "") + \
              (f", age {age}" if age else "")
    paths = []

    def plot(i, step, g, col):
        stated = {" ".join(k) for k in step["stated"]}
        nodes = {}
        for n in g.nodes.values():
            score = col.get(n.lemma, 0.0)
            if score <= 0:
                kind = "off"
            elif n.source == "inferred" and n.lemma not in stated:
                kind = "inferred"
            else:
                kind = "text"
            nodes[n.lemma] = (kind, score)

        edges = [(e.src.lemma, e.label, e.dst.lemma, e.active) for e in g.edges]

        active = {l for l, (k, _) in nodes.items() if k != "off"}
        paths.append(draw(
            os.path.join(out_dir, f"{tag}_s{i:02d}.png"),
            pos, nodes, edges,
            sentence_no=i, sentence=step["sentence"], persona=who, model=model,
            explicit=sorted(stated & set(nodes)),
            carryover=sorted(active - stated),
            inactive=sorted(set(nodes) - active)))
        return paths[-1]

    plot.paths = paths
    return plot
