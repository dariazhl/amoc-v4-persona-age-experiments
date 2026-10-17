import argparse
import glob
import hashlib
import itertools
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from amoc.metrics.aggregation import build_persona_record

warnings.filterwarnings("ignore")

REGIME_ORDER = ["primary", "secondary", "high_school", "university"]
REGIME_LEVEL = {r: i for i, r in enumerate(REGIME_ORDER)}
METRICS = {
    "num_triplets": "Triplets",
    "num_unique_concepts": "Concepts",
    "graph_density": "Density",
    "graph_avg_degree": "AvgDeg",
    "abstract_concept_ratio": "AbsCon",
    "graph_largest_component_ratio": "LCC",
}
STOP = set("""a an the and or of in on at to for with by from as is are was were be been being who which that this these
those his her their its it he she they them him i my me we our you your has have had do does did not no but so than then
into about over under very more most also whom whose what when where why how all any some each other such own same
can will would should could may might must just only out up down off again further once here there both few many much
year years old age aged student students school grade level interested love loves enjoys enjoy passionate""".split())
ALPHA = 0.05


def toks(s):
    out = set()
    for w in re.findall(r"[a-z]+", str(s).lower()):
        if len(w) < 3 or w in STOP:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)
    return out


def _norm(x):
    return " ".join(str(x).lower().replace("_", " ").split())


def _jac(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


def _read(files):
    frames = [pd.read_csv(f, engine="python", on_bad_lines="skip") for f in files]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_dir(d):
    ps = _read(sorted(glob.glob(os.path.join(d, "**", "triplets_per_sentence", "*.csv"), recursive=True)))
    fin = _read(sorted(glob.glob(os.path.join(d, "**", "triplets_final_state", "*final_triplets*.csv"), recursive=True)))
    if ps.empty or fin.empty:
        raise FileNotFoundError(f"no triplets_per_sentence / triplets_final_state CSVs under {d}")
    first = ps.sort_values("sentence_index")["sentence_text"].dropna().iloc[0]
    text_id = hashlib.sha1(_norm(first).encode()).hexdigest()[:8]
    out = []
    for df in (ps, fin):
        df = df[df["regime"].isin(REGIME_ORDER)].copy()
        df["persona_id"] = df["persona_text"].astype(str).map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:12])
        df["text"] = text_id
        df["folder"] = os.path.basename(os.path.normpath(d))
        out.append(df)
    story = set().union(*[toks(s) for s in ps["sentence_text"].dropna().unique()])
    return out[0], out[1], story


def persona_rows(set_name, ps, fin, story, seed):
    rng = np.random.default_rng(seed)
    pwords = {pid: toks(t) - story for pid, t in ps.drop_duplicates("persona_id").set_index("persona_id")["persona_text"].items()}
    cum = ps.drop_duplicates(["persona_id", "subject", "relation", "object"])
    rows = []
    for view, df in (("cumulative", cum), ("final", fin)):
        for pid, g in df.groupby("persona_id"):
            tw = [toks(f"{s} {r} {o}") for s, r, o in zip(g["subject"], g["relation"], g["object"])]
            own = pwords.get(pid, set())
            others = [p for p in pwords if p != pid and pwords[p]]
            ctrl = [pwords[p] for p in rng.choice(others, size=min(10, len(others)), replace=False)] if others else []
            inf_mask = (g["subject_source"] != "TEXT_BASED") | (g["object_source"] != "TEXT_BASED")
            inf_nodes = set(g.loc[g["subject_source"] != "TEXT_BASED", "subject"].map(_norm)) | \
                set(g.loc[g["object_source"] != "TEXT_BASED", "object"].map(_norm))
            rows.append({
                "set": set_name, "text": g["text"].iloc[0], "folder": g["folder"].iloc[0], "view": view,
                "persona_id": pid, "regime": g["regime"].iloc[0], "n_triplets": len(g),
                "leak_own": np.mean([bool(w & own) for w in tw]) if own else np.nan,
                "leak_other": np.mean([np.mean([bool(w & c) for w in tw]) for c in ctrl]) if own and ctrl else np.nan,
                "inferred_share": inf_mask.mean(),
                "inferred_word_len": np.mean([len(x) for x in inf_nodes]) if inf_nodes else np.nan,
                "inferred_nodes": "|".join(sorted(inf_nodes)),
                "triplets": "|".join(sorted({f"{_norm(s)};{_norm(r)};{_norm(o)}" for s, r, o in zip(g["subject"], g["relation"], g["object"])})),
            })
    return rows


def graph_rows(set_name, fin):
    recs = []
    for pid, g in fin.groupby("persona_id"):
        rec = build_persona_record(g)
        if rec is None:
            continue
        recs.append({"set": set_name, "text": g["text"].iloc[0], "persona_id": pid, "regime": g["regime"].iloc[0],
                     **{m: rec.get(m, np.nan) for m in METRICS}})
    return recs


def _bh(df, col):
    ok = df[col].notna()
    if ok.any():
        df.loc[ok, col + "_bh"] = multipletests(df.loc[ok, col], method="fdr_bh")[1]
    return df


def test_leakage(pr):
    rows = []
    for (s, view), g in pr.groupby(["set", "view"]):
        g = g.dropna(subset=["leak_own", "leak_other"])
        d = g["leak_own"] - g["leak_other"]
        p = wilcoxon(g["leak_own"], g["leak_other"], alternative="greater").pvalue if (d != 0).any() else 1.0
        rows.append({"set": s, "view": view, "n": len(g), "own_pct": 100 * g["leak_own"].mean(),
                     "other_pct": 100 * g["leak_other"].mean(), "excess_pp": 100 * d.mean(),
                     "personas_with_hit_pct": 100 * (g["leak_own"] > 0).mean(), "p_own_gt_other": p})
    return pd.DataFrame(rows)


def test_content_fingerprint(pr, field):
    rows = []
    for (s, view), g in pr.groupby(["set", "view"]):
        same, diff = [], []
        for _, gt in g.groupby("text"):
            items = [(set(filter(None, str(x).split("|"))), r) for x, r in zip(gt[field], gt["regime"])]
            for (a, ra), (b, rb) in itertools.combinations(items, 2):
                j = _jac(a, b)
                if not np.isnan(j):
                    (same if ra == rb else diff).append(j)
        p = mannwhitneyu(same, diff, alternative="greater").pvalue if same and diff else np.nan
        rows.append({"set": s, "view": view, "J_same_regime": np.mean(same) if same else np.nan,
                     "J_diff_regime": np.mean(diff) if diff else np.nan,
                     "diff": (np.mean(same) - np.mean(diff)) if same and diff else np.nan, "p_same_gt_diff": p})
    return pd.DataFrame(rows)


def test_regime(df, cols, label_fn=lambda c: c):
    rows = []
    for s, g in df.groupby("set"):
        g = g.copy()
        for c in cols:
            g["r"] = g.groupby("text")[c].rank(pct=True)
            pm = g.dropna(subset=["r"]).groupby(["persona_id", "regime"])["r"].mean().reset_index()
            groups = [pm.loc[pm.regime == r, "r"] for r in REGIME_ORDER if (pm.regime == r).sum() > 1]
            if len(groups) < 2:
                continue
            h, p = kruskal(*groups)
            rho = spearmanr(pm["regime"].map(REGIME_LEVEL), pm["r"])
            rows.append({"set": s, "measure": label_fn(c),
                         **{f"rank_{r}": pm.loc[pm.regime == r, "r"].mean() for r in REGIME_ORDER},
                         "p_kw": p, "trend_rho": rho.statistic, "p_trend": rho.pvalue})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = _bh(_bh(out, "p_kw"), "p_trend")
    return out


def icc1(w):
    w = w.dropna()
    n, k = w.shape
    if n < 3 or k < 2:
        return np.nan, n
    gm = w.values.mean()
    msb = k * ((w.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((w.values - w.mean(1).values[:, None]) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), n


def test_cross_text(gr, pr):
    rows = []
    cum = pr[pr.view == "cumulative"]
    for s in gr["set"].unique():
        g = gr[gr.set == s].copy()
        c = cum[cum.set == s].copy()
        row = {"set": s, "n_texts": g["text"].nunique()}
        for m, lab in METRICS.items():
            g["r"] = g.groupby("text")[m].rank(pct=True)
            row[lab], row["n_personas"] = icc1(g.pivot_table(index="persona_id", columns="text", values="r"))
        for m in ["inferred_share", "leak_own"]:
            c["r"] = c.groupby("text")[m].rank(pct=True)
            row[m], _ = icc1(c.pivot_table(index="persona_id", columns="text", values="r"))
        rows.append(row)
    return pd.DataFrame(rows)


def test_mode(pr, gr, base, test):
    rows = []
    cum = pr[pr.view == "cumulative"].copy()
    cum["leak_excess"] = cum["leak_own"] - cum["leak_other"]
    shared = set(cum.loc[cum.set == base, "text"]) & set(cum.loc[cum.set == test, "text"])
    measures = [("leak_excess", cum), ("inferred_share", cum), ("inferred_word_len", cum)]
    gm = gr.copy()
    measures += [(m, gm) for m in METRICS]
    for m, df in measures:
        a = df.loc[df.set == base].groupby("persona_id")[m].mean()
        b = df.loc[df.set == test].groupby("persona_id")[m].mean()
        row = {"measure": METRICS.get(m, m), "baseline": base, "test": test,
               "mean_baseline": a.mean(), "mean_test": b.mean(),
               "p_unpaired": mannwhitneyu(a.dropna(), b.dropna()).pvalue if a.notna().sum() > 1 and b.notna().sum() > 1 else np.nan}
        if shared:
            pa = df[(df.set == base) & df.text.isin(shared)].set_index(["text", "persona_id"])[m]
            pb = df[(df.set == test) & df.text.isin(shared)].set_index(["text", "persona_id"])[m]
            j = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
            row["n_paired"] = len(j)
            row["p_paired"] = wilcoxon(j["a"], j["b"]).pvalue if len(j) > 5 and (j["a"] != j["b"]).any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out = _bh(out, "p_unpaired")
    if "p_paired" in out:
        out = _bh(out, "p_paired")
    return out


def verdict(leak, fp_inf, fp_trip, reg_graph, reg_cont, cross):
    lines = []
    for s in leak["set"].unique():
        l = leak[(leak.set == s) & (leak.view == "cumulative")].iloc[0]
        fi = fp_inf[(fp_inf.set == s) & (fp_inf.view == "cumulative")].iloc[0]
        ft = fp_trip[(fp_trip.set == s) & (fp_trip.view == "cumulative")].iloc[0]
        rg = reg_graph[reg_graph.set == s]
        rc = reg_cont[reg_cont.set == s]
        cr = cross[cross.set == s]
        n_reg = int(((rg["p_kw_bh"] < ALPHA) | (rg["p_trend_bh"] < ALPHA)).sum()) if not rg.empty else 0
        n_cont = int(((rc["p_kw_bh"] < ALPHA) | (rc["p_trend_bh"] < ALPHA)).sum()) if not rc.empty else 0
        icc = cr.drop(columns=["set", "n_texts", "n_personas"]).max(axis=1).iloc[0] if not cr.empty else np.nan
        checks = {
            "A persona words in own graph > other personas' words (>=1 pp, p<.05)": l.excess_pp >= 1 and l.p_own_gt_other < ALPHA,
            "B inferred concepts more similar within regime than across (p<.05)": fi.p_same_gt_diff < ALPHA,
            "C triplets more similar within regime than across (p<.05)": ft.p_same_gt_diff < ALPHA,
            "D regime differs on >=1 graph metric after BH": n_reg > 0,
            "E regime differs on inferred share / inferred word length after BH": n_cont > 0,
            "F same persona behaves alike across texts (max ICC >= .30)": bool(icc >= 0.30) if not np.isnan(icc) else False,
        }
        lines.append(f"[{s}] blueprint checks passed: {sum(checks.values())}/6")
        lines += [f"   {'PASS' if v else 'fail'}  {k}" for k, v in checks.items()]
    return lines


def parse_sets(items):
    out = {}
    for it in items:
        name, dirs = it.split("=", 1)
        paths = []
        for d in dirs.split(","):
            paths += sorted(glob.glob(d)) or [d]
        out[name] = [p for p in paths if os.path.isdir(p)]
        if not out[name]:
            raise FileNotFoundError(f"no directories for set {name}: {dirs}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Is there a persona blueprint in the triplets? Compares sets (e.g. legacy vs reader).")
    ap.add_argument("--set", action="append", required=True,
                    help="NAME=DIR[,DIR...] ; each DIR holds one text's run (globs allowed)")
    ap.add_argument("--baseline", help="set name used as the reference in the mode comparison (e.g. legacy)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sets = parse_sets(args.set)
    prs, grs = [], []
    for name, dirs in sets.items():
        for d in dirs:
            ps, fin, story = load_dir(d)
            prs += persona_rows(name, ps, fin, story, args.seed)
            grs += graph_rows(name, fin)
            print(f"loaded {name}: {d} ({ps['persona_id'].nunique()} personas)", flush=True)
    pr, gr = pd.DataFrame(prs), pd.DataFrame(grs)

    leak = test_leakage(pr)
    fp_inf = test_content_fingerprint(pr, "inferred_nodes")
    fp_trip = test_content_fingerprint(pr, "triplets")
    reg_graph = test_regime(gr, list(METRICS), lambda c: METRICS[c])
    reg_cont = test_regime(pr[pr.view == "cumulative"], ["inferred_share", "inferred_word_len", "leak_own"])
    cross = test_cross_text(gr, pr)
    tables = {"1_persona_word_leakage": leak, "2a_inferred_concept_fingerprint": fp_inf,
              "2b_triplet_fingerprint": fp_trip, "3a_regime_graph_metrics": reg_graph,
              "3b_regime_inferred_content": reg_cont, "4_cross_text_consistency_icc": cross}
    if args.baseline:
        for s in sets:
            if s != args.baseline:
                tables[f"5_mode_{s}_vs_{args.baseline}"] = test_mode(pr, gr, args.baseline, s)

    pr.drop(columns=["triplets"]).to_csv(os.path.join(args.out_dir, "persona_rows.csv"), index=False)
    gr.to_csv(os.path.join(args.out_dir, "graph_metrics.csv"), index=False)
    out = []
    with pd.option_context("display.width", 220, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        for name, t in tables.items():
            t.to_csv(os.path.join(args.out_dir, name + ".csv"), index=False)
            out += [f"=== {name} ===", t.to_string(index=False), ""]
    out += ["=== VERDICT ==="] + verdict(leak, fp_inf, fp_trip, reg_graph, reg_cont, cross)
    out += ["", "A blueprint needs A or B/C (content) plus D/E (systematic by regime) and F (stable per persona).",
            "Texts are matched across sets by their first sentence; table 5 reports paired tests when sets share texts."]
    txt = "\n".join(out)
    open(os.path.join(args.out_dir, "summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
