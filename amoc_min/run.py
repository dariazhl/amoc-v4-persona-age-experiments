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
from .engine import LLM, analyze_story, read, resolve, to_matrix


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


def _one_reader(args):
    """One reader, start to finish. Module level and taking only plain data,
    so it can run in a worker process.

    Each reader builds its own Graph from the shared, already-extracted
    `passes` and writes files named after its own tag, so nothing is shared
    between workers but the read-only analysis.
    """
    r, cfg, passes, ref, order = args
    # A spawned process re-imports config at its default, so the ageing rule
    # has to be re-applied here or every worker would silently run `text`.
    C.RELATION_AGEING = cfg["relation_ageing"]
    llm = LLM(cfg["server_url"], cfg["model"], cfg["temperature"])
    snaps = []
    plotter = None
    if cfg["graphs"]:
        from .viz import make_plotter
        gd = os.path.join(cfg["out"], "graphs")
        os.makedirs(gd, exist_ok=True)
        plotter = make_plotter(gd, passes, r["persona"], r["tag"],
                               model=cfg["model"], regime=r.get("regime"),
                               age=r.get("age"))
    cols, counts, records, graph = read(
        passes, llm=llm, persona=r["persona"],
        allow_persona_words=cfg["allow_persona_words"],
        snapshots=snaps, on_sentence=plotter,
        meta={"original_index": r.get("idx", -1),
              "age_refined": r.get("age") or -1,
              "regime": r.get("regime", ""),
              "persona_text": r.get("persona") or "",
              "model_name": cfg["model"]})
    with open(os.path.join(cfg["out"], f"matrix_{r['tag']}.csv"), "w") as f:
        f.write(to_matrix(cols))
    if order:                            # v4-style: query the 20 reference
        from .tokens import TOKEN_MAP    # tokens, one row each
        with open(os.path.join(cfg["out"],
                               f"matrix_{r['tag']}_ref20.csv"), "w") as f:
            f.write(to_matrix(cols, vocab=order, token_map=TOKEN_MAP))
    if records:                                  # the triplet log
        with open(os.path.join(cfg["out"], f"triplets_{r['tag']}.csv"),
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0]))
            w.writeheader()
            w.writerows(records)
    if cfg["graphs"]:
        with open(os.path.join(cfg["out"], "graphs", f"{r['tag']}.json"), "w") as f:
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
    if order:
        # Per-token behaviour against the reference, so a matrix can be
        # read row by row without opening two CSVs side by side.
        # `on` is how many of the 13 sentences agree on merely being
        # active vs zero -- the shape of the trace, before its values.
        from .tokens import TOKEN_MAP
        rec["ref20"] = {}
        for t in order:
            ours = [resolve(c, TOKEN_MAP.get(t, t), TOKEN_MAP)
                    for c in cols]
            theirs = ref[t]
            rec["ref20"][t] = {
                "ref": theirs, "ours": ours,
                "on": sum(1 for x, y in zip(theirs, ours)
                          if (x > 0) == (y > 0))}
    if cfg["score"]:
        rec["spearman"] = score(cols, ref, order)
        rec["mean_rho"] = round(
            sum(x["rho"] or 0 for x in rec["spearman"]) / len(rec["spearman"]), 3)
        print(f"{r['tag']:<22} mean rho={rec['mean_rho']:+.3f}  "
              f"drew {rec['n_inferred']}/{rec['n_candidates']} candidates  "
              f"inf_rate={rec['inference_rate']:.3f}",
              flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--personas", nargs="*", default=None,
                    help="one or more persona CSVs; omit for baseline only")
    ap.add_argument("--out", default=str(C.OUT))
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
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
    ap.add_argument("--workers", type=int, default=1,
                    help="readers processed concurrently. The story analysis "
                         "runs once up front either way; this parallelises "
                         "the per-persona pass, whose requests vLLM batches, "
                         "so throughput scales with it. Processes, not "
                         "threads -- spaCy is not thread-safe.")
    ap.add_argument("--llm-decay", action="store_true",
                    help="age a relation from when the model last judged it "
                         "RELEVANT (NARRATIVE_RELEVANCE score 2) instead of "
                         "from when its own word last appeared in the text. "
                         "Lets the model decide what is still in mind rather "
                         "than keying off surface mentions, and is the only "
                         "ageing rule the persona could ever influence. "
                         "Mechanically furthest from the Landscape reference: "
                         "one LLM judgement then drives BOTH traversal cost "
                         "and age, so the matrix becomes largely a function "
                         "of the model's opinion of its own graph.")
    ap.add_argument("--allow-persona-words", action="store_true",
                    help="let readers reword concepts. OFF by default: 10 of the "
                         "20 reference tokens match by exact string only, so "
                         "rewording zeroes those rows in Study 1 scoring.")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # Set before anything builds a graph: graph.activation reads it per edge.
    C.RELATION_AGEING = C.AGEING_LLM if a.llm_decay else C.AGEING_TEXT
    llm = LLM(a.server_url, a.model, a.temperature)
    print(f"temperature={a.temperature} "
          f"relation_ageing={C.RELATION_AGEING}"
          f"{' (--llm-decay)' if a.llm_decay else ''}", flush=True)
    # The reference matrix is only needed for --score. Load it lazily and
    # degrade to generation-only if it is missing, rather than throwing away
    # a 90-second model load over an optional file.
    ref = order = None
    if True:                                 # also needed for the ref20 matrix
        try:
            ref, order = reference(a.fix_killed)
        except FileNotFoundError:
            print(f"WARNING: no reference matrix at {C.REFERENCE} -- "
                  f"continuing without scoring. Sync amoc_min/reference/ "
                  f"and re-run with --score to score these matrices.",
                  flush=True)
            a.score = False

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

    cfg = {k: getattr(a, k) for k in
           ("graphs", "model", "out", "allow_persona_words", "score",
            "server_url", "temperature")}
    cfg["relation_ageing"] = C.RELATION_AGEING
    tasks = [(r, cfg, passes, ref, order) for r in runs]

    summary = []
    workers = min(a.workers, len(tasks)) or 1
    if workers > 1:
        # vLLM batches concurrent requests, so throughput scales with this.
        # Processes, not threads: spaCy's Language object is not documented
        # thread-safe and every reader lemmatises constantly.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"{len(tasks)} readers across {workers} workers", flush=True)
        done = 0
        with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_one_reader, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                r = futures[fut]
                try:
                    summary.append(fut.result())
                except Exception as exc:        # one reader must not sink the run
                    print(f"FAILED {r['tag']}: {exc}", flush=True)
                done += 1
                print(f"  {done}/{len(tasks)} readers done", flush=True)
        # as_completed yields in finishing order, so restore the run order --
        # otherwise summary.json is shuffled differently on every run.
        pos = {r["tag"]: n for n, r in enumerate(runs)}
        summary.sort(key=lambda rec: pos.get(rec["tag"], 0))
    else:
        for t in tasks:
            summary.append(_one_reader(t))

    # Stamp the ageing rule on every run, so a matrix can never be read
    # without knowing which decay produced it.
    for rec in summary:
        rec["relation_ageing"] = C.RELATION_AGEING
        rec["llm_decay"] = bool(a.llm_decay)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {len(runs)} matrices to {a.out}")


if __name__ == "__main__":
    main()
