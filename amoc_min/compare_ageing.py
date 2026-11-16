"""Decide between the two ageing rules by measuring what the flag changes.

    python3 -m amoc_min.compare_ageing RUN_TEXT [RUN_LLM ...]

RUN is a directory holding summary.json (or the summary.json itself). Each is
one arm of the A/B. Two questions decide the flag, and this prints both:

  ALIGNMENT   mean Spearman vs the Landscape reference -- the Study 1 number.
              The text rule should win here; the reference is built on surface
              mentions, so this quantifies what llm_decay costs.

  DISPERSION  how much readers differ FROM EACH OTHER. Under the text rule
              persona only picks inferences, so matrices barely move. If
              dispersion does not rise under llm_decay, option C buys nothing
              and the persona story has no mechanism -- that is the finding.

Readers are compared on the 20x13 ref20 matrix, baseline excluded from the
persona statistics (it is persona-free) and reported separately.
"""
import json
import os
import sys
from itertools import combinations

from scipy.stats import spearmanr


def load(path):
    p = path if path.endswith(".json") else os.path.join(path, "summary.json")
    recs = json.load(open(p))
    return p, recs if isinstance(recs, list) else [recs]


def flat(rec, order):
    return [v for t in order for v in rec["ref20"][t]["ours"]]


def arm(path):
    p, recs = load(path)
    order = list(recs[0]["ref20"])
    base = next((r for r in recs if r["tag"] == "baseline"), None)
    ps = [r for r in recs if r["tag"] != "baseline"]
    if not ps:
        sys.exit(f"{p}: no persona readers")

    vecs = {r["tag"]: flat(r, order) for r in ps}
    pairs = [spearmanr(vecs[a], vecs[b]).correlation
             for a, b in combinations(vecs, 2)]
    pairs = [x for x in pairs if x == x]
    rhos = [r["mean_rho"] for r in ps]
    inf = [r["n_inferred"] for r in ps]
    cells = sum(1 for i in range(len(order) * 13)
                if len({v[i] for v in vecs.values()}) > 1)

    return {
        "path": p,
        "ageing": recs[0].get("relation_ageing") or "unstamped",
        "llm_decay": recs[0].get("llm_decay"),
        "n": len(ps),
        "base_rho": base and base["mean_rho"],
        "align": sum(rhos) / len(rhos),
        "align_sd": (sum((x - sum(rhos) / len(rhos)) ** 2
                         for x in rhos) / len(rhos)) ** .5,
        "align_min": min(rhos), "align_max": max(rhos),
        "distinct": len({tuple(v) for v in vecs.values()}),
        "cells": cells, "n_cells": len(order) * 13,
        "pair_rho": sum(pairs) / len(pairs) if pairs else float("nan"),
        "inf": sum(inf) / len(inf),
        "inf_min": min(inf), "inf_max": max(inf),
        "cand": ps[0]["n_candidates"],
    }


def show(a):
    print(f"\n{'=' * 72}\n{a['path']}")
    print(f"  ageing rule      {a['ageing']}  (llm_decay={a['llm_decay']})")
    print(f"  readers          {a['n']} personas"
          + (f" + baseline" if a["base_rho"] is not None else ""))
    print("\n  ALIGNMENT vs Landscape reference")
    if a["base_rho"] is not None:
        print(f"    baseline            {a['base_rho']:+.4f}")
    print(f"    personas  mean      {a['align']:+.4f}  sd {a['align_sd']:.4f}"
          f"  range {a['align_max'] - a['align_min']:.4f}"
          f"  [{a['align_min']:+.3f}, {a['align_max']:+.3f}]")
    print("\n  DISPERSION across readers (does persona bite?)")
    print(f"    distinct matrices   {a['distinct']} of {a['n']}")
    print(f"    cells that vary     {a['cells']} of {a['n_cells']}"
          f"  ({100 * a['cells'] / a['n_cells']:.1f}%)")
    print(f"    mean pairwise rho   {a['pair_rho']:+.4f}"
          f"   (1.0 = readers identical)")
    print(f"    inferences drawn    {a['inf']:.1f} of {a['cand']}"
          f"  [{a['inf_min']}, {a['inf_max']}]")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arms = [arm(p) for p in sys.argv[1:]]
    for a in arms:
        show(a)
    if len(arms) < 2:
        print(f"\n{'=' * 72}\nOne arm only -- pass both to get the verdict.\n")
        return

    t = next((a for a in arms if a["ageing"] == "text"), arms[0])
    l = next((a for a in arms if a["ageing"] == "relevance"), arms[-1])
    if t is l:
        print("\nBoth arms carry the same ageing rule -- not an A/B.")
        return

    print(f"\n{'=' * 72}\nVERDICT   text -> llm_decay\n")
    print(f"{'':22}{'text':>10}{'llm_decay':>12}{'delta':>10}")
    for lbl, k, f in (("alignment (mean rho)", "align", "{:+.4f}"),
                      ("reader spread (sd)", "align_sd", "{:.4f}"),
                      ("cells varying", "cells", "{:.0f}"),
                      ("distinct matrices", "distinct", "{:.0f}"),
                      ("pairwise rho", "pair_rho", "{:+.4f}"),
                      ("inferences drawn", "inf", "{:.1f}")):
        print(f"  {lbl:20}{f.format(t[k]):>10}{f.format(l[k]):>12}"
              f"{f.format(l[k] - t[k]):>10}")
    print(f"\n  Q1 what llm_decay costs Study 1:"
          f"  {l['align'] - t['align']:+.4f} mean rho")
    print(f"  Q2 what llm_decay buys the persona story:"
          f"  pairwise rho {t['pair_rho']:+.4f} -> {l['pair_rho']:+.4f},"
          f"  varying cells {t['cells']} -> {l['cells']}")
    print("\n  Read it this way: Q2 must move substantially for option C to be"
          "\n  worth a section. If pairwise rho stays near 1.0 under llm_decay,"
          "\n  the rule changed the matrix but not what distinguishes readers.\n")


if __name__ == "__main__":
    main()
