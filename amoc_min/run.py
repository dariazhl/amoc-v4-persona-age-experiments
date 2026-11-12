"""CLI: story analysis, then one reader pass per persona, then scoring.

  python3 -m amoc_min.run --server-url http://127.0.0.1:8000 \
      --personas personas_dfs/personas_refined_age/chunks_balanced/primary_000.csv \
      --out runs/primary_000 --score
"""
import argparse
import csv
import json
import os

from . import config as C
from .engine import LLM, analyze_story, read, to_matrix


def load_personas(paths, limit=None):
    """All persona files at once. limit applies PER FILE, so a smoke test
    still gets a spread across regimes rather than 2 primary personas."""
    out = []
    for path in paths:
        regime = os.path.basename(path).split("_")[0]
        rows = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                text = (r.get("persona_text") or "").strip()
                if text:
                    rows.append({"idx": r.get("idx"), "regime": regime,
                                 "age": r.get("age_refined") or r.get("age"),
                                 "persona_text": text})
        out += rows[:limit] if limit else rows
    return out


def reference(fix_killed=False):
    ref, order = {}, []
    for i, line in enumerate(open(C.REFERENCE).read().strip().split("\n")):
        if i == 0:
            continue
        p = line.split()
        ref[p[0]] = [float(x) for x in p[1:]]
        order.append(p[0])
    if fix_killed:                       # Fig.8 has killed starting at S10,
        ref["killed"] = [0] * 9 + [5, 3, 1.5, 0]   # the CSV starts it at S9
    return ref, order


def score(columns, ref, order):
    """Per-sentence Spearman against the reference, using the vendored TOKEN_MAP."""
    from scipy.stats import spearmanr

    from .tokens import TOKEN_MAP

    out = []
    for s, col in enumerate(columns):
        canon = {}
        for tok, val in col.items():
            key = TOKEN_MAP.get(tok, tok)
            canon[key] = max(canon.get(key, 0.0), val)
        mine = [canon.get(t, 0.0) for t in order]
        theirs = [ref[t][s] for t in order]
        rho, p = spearmanr(mine, theirs)
        out.append({"sentence": s + 1, "rho": None if rho != rho else round(rho, 3),
                    "p": None if p != p else round(p, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--personas", nargs="*", default=None,
                    help="one or more persona CSVs; omit for baseline only")
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--reader-guidance", default=C.READER_GUIDANCE,
                    choices=["neutral", "directional", "none"],
                    help="how the reader prompt describes differences "
                         "between readers; see config.py")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the baseline reader. It is persona-free and "
                         "identical in every array task, so compute it once.")
    ap.add_argument("--story-analysis", default=None,
                    help="path to the shared story-analysis cache. Point every "
                         "array task at one file so EXTRACT/LICENSE run once "
                         "for the story instead of once per chunk.")
    ap.add_argument("--graphs", action="store_true",
                    help="also dump per-sentence graph snapshots and GraphViz")
    ap.add_argument("--limit", type=int, help="personas per file")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--fix-killed", action="store_true",
                    help="correct the killed row to match Fig. 8 (off by default)")
    ap.add_argument("--allow-persona-words", action="store_true",
                    help="let readers reword concepts. OFF by default: 10 of the "
                         "20 reference tokens match by exact string only, so "
                         "rewording zeroes those rows in Study 1 scoring.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    C.READER_GUIDANCE = a.reader_guidance
    llm = LLM(a.server_url, a.model, a.temperature)
    print(f"temperature={a.temperature}  "
          f"reader_guidance={a.reader_guidance}", flush=True)
    ref, order = reference(a.fix_killed)

    cache = a.story_analysis or os.path.join(a.out, "story_analysis.json")
    passes = analyze_story(llm, cache=cache)
    print(f"candidate inferred relationships per sentence: "
          f"{[len(p['inferred']) for p in passes]}", flush=True)
    chained = sum(1 for p in passes for c in p["inferred"] if c.get("from"))
    print(f"of which chained (build on an earlier inference): {chained}", flush=True)

    runs = [] if a.no_baseline else [
        {"tag": "baseline", "persona": None, "regime": "none",
         "idx": "-", "age": 0}]
    if a.personas:
        runs += [{"tag": f"{p['regime']}_{p['idx']}", "persona": p["persona_text"],
                  "regime": p["regime"], "idx": p["idx"], "age": p["age"]}
                 for p in load_personas(a.personas, a.limit)]

    summary = []
    for r in runs:
        snaps = []
        plotter = None
        if a.graphs:
            from .viz import make_plotter
            gd = os.path.join(a.out, "graphs")
            os.makedirs(gd, exist_ok=True)
            plotter = make_plotter(gd, passes, r["persona"], r["tag"],
                                   model=a.model, regime=r.get("regime"),
                                   age=r.get("age"))
        cols, counts, records, graph = read(
            passes, llm=llm, persona=r["persona"],
            allow_persona_words=a.allow_persona_words,
            snapshots=snaps, on_sentence=plotter,
            meta={"original_index": r.get("idx", -1),
                  "age_refined": r.get("age") or -1,
                  "regime": r.get("regime", ""),
                  "persona_text": r.get("persona") or "",
                  "model_name": a.model})
        with open(os.path.join(a.out, f"matrix_{r['tag']}.csv"), "w") as f:
            f.write(to_matrix(cols))
        if records:                                  # the triplet log
            with open(os.path.join(a.out, f"triplets_{r['tag']}.csv"),
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0]))
                w.writeheader()
                w.writerows(records)
        if a.graphs:
            with open(os.path.join(a.out, "graphs", f"{r['tag']}.json"), "w") as f:
                json.dump(snaps, f, indent=1)
        rec = {k: r[k] for k in ("tag", "regime", "idx")}
        rec["reader inferences"] = counts
        rec["n_candidates"] = sum(c["candidates"] for c in counts)
        rec["n_inferred"] = sum(c["drawn"] for c in counts)
        inferred_nodes = {n["lemma"] for n in graph.snapshot()["nodes"]
                          if n["source"] == "inferred"}
        rec["n_inferred_nodes"] = len(inferred_nodes)
        rec["inference_rate"] = round(
            len(inferred_nodes) / max(1, len(graph.nodes)), 4)
        if a.score:
            rec["spearman"] = score(cols, ref, order)
            rec["mean_rho"] = round(
                sum(x["rho"] or 0 for x in rec["spearman"]) / len(rec["spearman"]), 3)
            print(f"{r['tag']:<22} mean rho={rec['mean_rho']:+.3f}  "
                  f"drew {rec['n_inferred']}/{rec['n_candidates']} candidates  "
                  f"inf_rate={rec['inference_rate']:.3f}",
                  flush=True)
        summary.append(rec)

    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {len(runs)} matrices to {a.out}")


if __name__ == "__main__":
    main()
