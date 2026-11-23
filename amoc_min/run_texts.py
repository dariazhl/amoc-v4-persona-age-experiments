"""Keefe & McDaniel (1993) and O'Brien et al. (1998) on amoc_min.

The Study 1 method applied to many short texts instead of one story: each
text is analysed once, persona-free, and every reader then reads it under the
chosen decay rule (--llm-score, --llm-decay, or the text rule by default).
What differs is the measurement taken when the text ends.

  keefe   one sentence per text, in a control, predictive or explicit
          version, and a target word. Two measures: `activation`, the model's
          own activation of the target (0 if this reader never built it), and
          `score`, the 1-4 relevance rating from the AMoC v4 paper's prompt.
  obrien  an introduction plus one elaboration (consistent, inconsistent or
          one of four qualified versions) and a target sentence. The model
          lists which relationships support and which contradict the target;
          counted once over the relationships still ACTIVE at the end, which
          is where a decay rule shows, and once over ALL relationships built,
          which is what AMoC v4 counted.

Both scoring prompts are persona-free, so a difference between readers comes
from the graph each one built, not from who is asked to judge it.

  python3 -m amoc_min.run_texts --experiment keefe \\
      --server-url http://127.0.0.1:8000 --personas <chunks>/*.csv --limit 5 \\
      --workers 10 --out runs/keefe_score
"""
import argparse
import csv
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

from . import config as C
from . import nlp as N
from .engine import LLM, analyze_story, read, resolve
from .run import load_personas, write_proposals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = {"keefe": os.path.join(ROOT, "amoc", "keefe_exp", "keefe_ready.json"),
         "obrien": os.path.join(ROOT, "amoc", "obrien_exp", "obrien_combined.json")}
KEEFE_CONDITIONS = ("control", "predictive", "explicit")
OBRIEN_CONDITIONS = ("consistent", "inconsistent", "qualified1", "qualified2",
                     "qualified3", "qualified4")

# The wording used for the v5 runs of the same two experiments, unchanged.
KEEFE_PROBE = """You have the following edges from a knowledge graph in the format: node - edge - node.
{edges}
For the word "{probe_word}", assign a score from 1 to 4 using these definitions:
- 4: Strong connection or relevance to the story. The story told by the edges clearly implies or revolves around this word, even if the word itself is not written among the edges.
- 3: A clear connection or relevance, but the word is not the main focus of the story.
- 2: Little connection or relevance.
- 1: No connection or relevance.
Return ONLY the number (1,2,3,4). Do not add any extra text or punctuation."""

SUPPORT_CONTRADICT = """You are evaluating a knowledge graph against a target sentence.

Here are the edges from the graph, numbered:
{edges}

Target sentence: "{target_sentence}"

Classify each edge as SUPPORT, CONTRADICT:
- SUPPORT: the edge describes a trait, event, or circumstance that makes the target more likely or expected.
- CONTRADICT: the edge describes a trait, event, or circumstance that conflicts with the target — even indirectly. For example, if a character fears heights, any edge expressing that fear contradicts a target involving high-altitude activities, because a person afraid of heights would be unlikely to seek them out.

Only flag an edge as CONTRADICT if it directly states or strongly implies something incompatible with the target. Do NOT flag edges just because they describe traits that might make the target less likely.

Return ONLY a JSON object with "support" and "contradict" lists containing edge numbers.
Example: {{"support": [1, 3], "contradict": [2, 5]}}"""


def texts_of(experiment, items):
    """One entry per text the readers will read."""
    out = []
    for it in items:
        if experiment == "keefe":
            for cond in KEEFE_CONDITIONS:
                out.append({"text_id": f"{it['id']}_{cond}", "item_id": it["id"],
                            "condition": cond, "text": it[f"{cond}_text"],
                            "target": it["target_lemma"]})
        else:
            for cond in OBRIEN_CONDITIONS:
                elab = it.get(cond)
                if isinstance(elab, str) and elab.strip():    # not every text has every version
                    out.append({"text_id": f"{it['id']}_{cond}", "item_id": it["id"],
                                "condition": cond,
                                "text": it["introduction"].strip() + " " + elab.strip(),
                                "target": it["target_sentence_1"]})
    return out


def _edges(g, only_active):
    return [(e.src.text, e.label, e.dst.text) for e in g.edges
            if str(e.label).strip() and (e.active or not only_active)]


def score_keefe(llm, g, column, target):
    edges = _edges(g, only_active=False)
    score = 1
    if edges:
        reply = llm.ask(KEEFE_PROBE.format(
            edges="\n".join(f"{s} - {r} - {o}" for s, r, o in edges),
            probe_word=target))
        m = re.search(r"\b([1-4])\b", reply)
        score = int(m.group(1)) if m else 1
    return {"score": score, "activation": resolve(column, target),
            "n_edges": len(edges), "n_active": len(_edges(g, only_active=True))}


def _support_contradict(llm, edges, target):
    if not edges:
        return 0, 0
    reply = llm.ask(SUPPORT_CONTRADICT.format(
        edges="\n".join(f"{i}. {s} - {r} - {o}" for i, (s, r, o) in enumerate(edges, 1)),
        target_sentence=target)).strip()
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    counts = []
    for key in ("support", "contradict"):
        m = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', reply, re.DOTALL)
        nums = {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()
        counts.append(len({n for n in nums if 1 <= n <= len(edges)}))
    return tuple(counts)


def score_obrien(llm, g, column, target):
    active, every = _edges(g, only_active=True), _edges(g, only_active=False)
    sa, ca = _support_contradict(llm, active, target)
    # one sentence in, nothing has decayed: do not pay for the same call twice
    se, ce = (sa, ca) if active == every else _support_contradict(llm, every, target)
    return {"support_active": sa, "contradict_active": ca, "difference_active": sa - ca,
            "support_all": se, "contradict_all": ce, "difference_all": se - ce,
            "n_edges": len(every), "n_active": len(active)}


SCORERS = {"keefe": score_keefe, "obrien": score_obrien}


def _apply(cfg):
    """A spawned process re-imports config at its defaults."""
    C.RELATION_AGEING = cfg["relation_ageing"]
    C.apply_persona_cfg(cfg)
    return LLM(cfg["server_url"], cfg["model"], cfg["temperature"])


def _analyse(args):
    text, cfg = args
    llm = _apply(cfg)
    sentences = [s.text.strip() for s in N.nlp()(text["text"]).sents if s.text.strip()]
    return text["text_id"], analyze_story(llm, sentences=sentences)


def _one_reader(args):
    r, cfg, texts, analyses = args
    llm = _apply(cfg)
    rows, proposals = [], []
    for t in texts:
        passes = analyses.get(t["text_id"])
        if not passes:
            continue
        try:
            cols, counts, _, g = read(passes, llm=llm, persona=r["persona"],
                                      capacity=C.capacity_for(r["regime"] if r["persona"] else None))
            measured = SCORERS[cfg["experiment"]](llm, g, cols[-1], t["target"])
        except Exception as exc:              # one text must not sink the reader
            print(f"FAILED {r['tag']} {t['text_id']}: {exc}", flush=True)
            continue
        rows.append({"tag": r["tag"], "regime": r["regime"],
                     "global_persona_index": r["idx"], "age": r["age"],
                     "persona_text": r["persona"] or "baseline",
                     "item_id": t["item_id"], "condition": t["condition"],
                     **measured, "relation_ageing": cfg["relation_ageing"],
                     "persona_decay": cfg["persona_decay"],
                     "persona_generates": cfg.get("persona_generates", False),
                     "strict_selection": cfg.get("strict_selection", False),
                     "reader_capacity": cfg.get("reader_capacity", False),
                     "loose_license": cfg.get("loose_license", False),
                     "n_drawn": sum(c["drawn"] for c in counts),
                     "n_proposed": sum(c.get("proposed", 0) for c in counts),
                     "n_licensed": sum(c.get("licensed", 0) for c in counts)})
        proposals += [
            {"tag": r["tag"], "regime": r["regime"], "item_id": t["item_id"],
             "condition": t["condition"], "sentence": i,
             "source": p["triplet"][0], "verb": p["triplet"][1],
             "object": p["triplet"][2], "status": p["status"]}
            for i, c in enumerate(counts, 1) for p in c.get("proposals", [])]
    write_proposals(os.path.join(cfg["out"], f"proposals_{r['tag']}.csv"), proposals)
    return r["tag"], rows


def _pool(fn, tasks, workers, label):
    """Run tasks across worker processes against the one vLLM server."""
    workers = max(1, min(workers, len(tasks)))
    if workers == 1:
        for t in tasks:
            yield fn(t)
        return
    print(f"{label}: {len(tasks)} tasks across {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        for fut in as_completed([pool.submit(fn, t) for t in tasks]):
            try:
                yield fut.result()
            except Exception as exc:
                print(f"FAILED {label} task: {exc}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(ITEMS))
    ap.add_argument("--items", help="items JSON (default: the one under amoc/)")
    ap.add_argument("--server-url", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--personas", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=C.TEMPERATURE)
    ap.add_argument("--limit", type=int, help="personas per file")
    ap.add_argument("--max-items", type=int, help="first N items only (smoke test)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--llm-decay", action="store_true")
    ap.add_argument("--llm-score", action="store_true")
    ap.add_argument("--persona-decay", action="store_true")
    ap.add_argument("--persona-generates", action="store_true")
    ap.add_argument("--strict-selection", action="store_true")
    ap.add_argument("--loose-license", action="store_true",
                    help="the neutral judge accepts what is plausible given "
                         "the text (needs --persona-generates)")
    ap.add_argument("--reader-capacity", action="store_true")
    ap.add_argument("--persona-strong", action="store_true",
                    help="= --persona-generates --reader-capacity --persona-decay")
    a = ap.parse_args()
    if a.persona_strong:
        a.persona_generates = a.reader_capacity = a.persona_decay = True
    if a.loose_license and not a.persona_generates:
        ap.error("--loose-license only applies with --persona-generates")
    if a.llm_decay and a.llm_score:
        ap.error("--llm-decay and --llm-score are different rules; pick one")

    os.makedirs(a.out, exist_ok=True)
    cfg = {"experiment": a.experiment, "out": a.out, "loose_license": a.loose_license, "server_url": a.server_url, "model": a.model,
           "temperature": a.temperature, "persona_decay": a.persona_decay,
           "persona_generates": a.persona_generates,
           "strict_selection": a.strict_selection, "reader_capacity": a.reader_capacity,
           "relation_ageing": (C.AGEING_SCORE if a.llm_score else
                               C.AGEING_LLM if a.llm_decay else C.AGEING_TEXT)}
    items = json.load(open(a.items or ITEMS[a.experiment], encoding="utf-8"))
    texts = texts_of(a.experiment, items[:a.max_items] if a.max_items else items)
    print(f"{a.experiment}: {len(texts)} texts | relation_ageing={cfg['relation_ageing']} "
          f"persona_decay={a.persona_decay} persona_generates={a.persona_generates} "
          f"loose_license={a.loose_license} strict_selection={a.strict_selection} reader_capacity={a.reader_capacity} "
          f"| model={a.model}", flush=True)

    # 1. persona-free analysis of every text, once, shared by every reader
    cache = os.path.join(a.out, "text_analysis.json")
    analyses = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [(t, cfg) for t in texts if t["text_id"] not in analyses]
    for n, (text_id, passes) in enumerate(_pool(_analyse, todo, a.workers, "analysis"), 1):
        analyses[text_id] = passes
        if n % 20 == 0 or n == len(todo):
            print(f"  analysed {n}/{len(todo)} texts", flush=True)
            json.dump(analyses, open(cache + ".tmp", "w"))
            os.replace(cache + ".tmp", cache)

    # 2. every reader reads every text; finished readers survive a restart
    readers = [] if a.no_baseline else [
        {"tag": "baseline", "persona": None, "regime": "none", "idx": "-", "age": 0}]
    if a.personas:
        readers += [{"tag": f"{p['regime']}_{p['idx']}", "persona": p["persona_text"],
                     "regime": p["regime"], "idx": p["idx"], "age": p["age"]}
                    for p in load_personas(a.personas, a.limit)]
    out_csv = os.path.join(a.out, f"{a.experiment}_scores.csv")
    done = set()
    if os.path.exists(out_csv):
        done = {row["tag"] for row in csv.DictReader(open(out_csv, newline=""))}
        print(f"resuming: {len(done)} readers already in {out_csv}", flush=True)
    tasks = [(r, cfg, texts, analyses) for r in readers if r["tag"] not in done]
    for n, (tag, rows) in enumerate(_pool(_one_reader, tasks, a.workers, "readers"), 1):
        if rows:
            new = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                if new:
                    w.writeheader()
                w.writerows(rows)
        print(f"  {n}/{len(tasks)} readers done ({tag}: {len(rows)} texts)", flush=True)

    # 3. the means by condition, so the log already says whether it worked
    if os.path.exists(out_csv):
        rows = list(csv.DictReader(open(out_csv, newline="")))
        measures = [k for k in ("score", "activation", "difference_active", "difference_all")
                    if k in rows[0]]
        print(f"\n{len(rows)} rows, {len({r['tag'] for r in rows})} readers -> {out_csv}")
        print("condition".ljust(14) + "".join(m.rjust(20) for m in measures))
        conds = KEEFE_CONDITIONS if a.experiment == "keefe" else OBRIEN_CONDITIONS
        for c in conds:
            sub = [r for r in rows if r["condition"] == c]
            if sub:
                print(c.ljust(14) + "".join(
                    f"{sum(float(r[m]) for r in sub) / len(sub):20.3f}" for m in measures))


if __name__ == "__main__":
    main()
