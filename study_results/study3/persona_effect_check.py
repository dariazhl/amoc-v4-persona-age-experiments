import argparse
import glob
import hashlib
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr, wilcoxon
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
ICC_PASS = 0.5
ALPHA = 0.05


def _norm(x):
    return " ".join(str(x).lower().split())


def load_rep(root):
    files = sorted(glob.glob(os.path.join(root, "**", "triplets_final_state", "*final_triplets*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no triplets_final_state/*final_triplets*.csv under {root}")
    frames = []
    for path in files:
        rel = os.path.relpath(path, root)
        text = rel.split(os.sep)[0]
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        need = {"persona_text", "regime", "subject", "relation", "object", "original_index", "age_refined"}
        if df.empty or not need <= set(df.columns):
            continue
        df["text"] = text
        df["source_file"] = os.path.basename(path)
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["regime"].isin(REGIME_ORDER)].copy()
    raw["persona_id"] = raw["persona_text"].astype(str).map(lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()[:12])
    return raw


def persona_metrics(raw):
    recs = []
    for (text, pid), g in raw.groupby(["text", "persona_id"]):
        rec = build_persona_record(g)
        if rec is None:
            continue
        row = {"text": text, "persona_id": pid, "regime": g["regime"].iloc[0]}
        row.update({m: rec.get(m, np.nan) for m in METRICS})
        recs.append(row)
    df = pd.DataFrame(recs)
    for m in METRICS:
        df["r_" + m] = df.groupby("text")[m].rank(pct=True)
    return df


def triplet_sets(raw):
    out = {}
    for (text, pid), g in raw.groupby(["text", "persona_id"]):
        trip = set(zip(g["subject"].map(_norm), g["relation"].map(_norm), g["object"].map(_norm)))
        conc = set(g["subject"].map(_norm)) | set(g["object"].map(_norm))
        out[(text, pid)] = (trip, conc, g["regime"].iloc[0])
    return out


def icc_2_1(x, y):
    data = np.column_stack([x, y])
    n, k = data.shape
    if n < 3:
        return np.nan
    gm = data.mean()
    ssr = k * ((data.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((data.mean(axis=0) - gm) ** 2).sum()
    sst = ((data - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / den if den > 0 else np.nan


def icc_ci(x, y, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(icc_2_1(x[idx], y[idx]))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def check1_reliability(pm_a, pm_b):
    rows = []
    merged = pm_a.merge(pm_b, on=["text", "persona_id"], suffixes=("_a", "_b"))
    for m, label in METRICS.items():
        a = merged["r_" + m + "_a"].to_numpy()
        b = merged["r_" + m + "_b"].to_numpy()
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        icc = icc_2_1(a, b)
        lo, hi = icc_ci(a, b)
        ma = pm_a.groupby("persona_id")["r_" + m].mean()
        mb = pm_b.groupby("persona_id")["r_" + m].mean()
        common = ma.index.intersection(mb.index)
        r_mean = spearmanr(ma[common], mb[common]).statistic if len(common) > 2 else np.nan
        rows.append({
            "metric": label,
            "n_pairs": int(ok.sum()),
            "icc_within_text": round(icc, 3),
            "icc_ci_low": round(lo, 3),
            "icc_ci_high": round(hi, 3),
            "persona_mean_rho": round(r_mean, 3),
            "pass": bool(icc >= ICC_PASS),
        })
    return pd.DataFrame(rows)


def _jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def check2_fingerprint(sets_a, sets_b=None):
    cross = sets_b is not None
    sets_b = sets_b if cross else sets_a
    rows = []
    for kind, idx in (("triplets", 0), ("concepts", 1)):
        per = []
        for (text, pid), va in sets_a.items():
            same_self = np.nan
            if cross and (text, pid) in sets_b:
                same_self = _jac(va[idx], sets_b[(text, pid)][idx])
            same_reg, diff_reg = [], []
            for (t2, p2), vb in sets_b.items():
                if t2 != text or p2 == pid:
                    continue
                j = _jac(va[idx], vb[idx])
                (same_reg if vb[2] == va[2] else diff_reg).append(j)
            per.append({"text": text, "persona_id": pid, "self": same_self,
                        "same_regime": np.nanmean(same_reg) if same_reg else np.nan,
                        "diff_regime": np.nanmean(diff_reg) if diff_reg else np.nan})
        per = pd.DataFrame(per).groupby("persona_id")[["self", "same_regime", "diff_regime"]].mean()
        row = {"unit": kind, "n_personas": len(per),
               "J_self": round(per["self"].mean(), 3) if cross else np.nan,
               "J_same_regime": round(per["same_regime"].mean(), 3),
               "J_diff_regime": round(per["diff_regime"].mean(), 3)}
        if cross:
            d = per.dropna(subset=["self", "same_regime"])
            row["p_self_gt_same"] = wilcoxon(d["self"], d["same_regime"], alternative="greater").pvalue
        else:
            row["p_self_gt_same"] = np.nan
        d = per.dropna(subset=["same_regime", "diff_regime"])
        row["p_same_gt_diff"] = wilcoxon(d["same_regime"], d["diff_regime"], alternative="greater").pvalue
        row["pass_persona"] = bool(cross and row["p_self_gt_same"] < ALPHA)
        row["pass_regime"] = bool(row["p_same_gt_diff"] < ALPHA)
        rows.append(row)
    return pd.DataFrame(rows)


def _multi_regime(pm):
    keep = pm.groupby("text")["regime"].transform("nunique") >= 2
    return pm[keep]


def check3_regime(pm):
    pm = _multi_regime(pm)
    rows = []
    for m, label in METRICS.items():
        pmean = pm.groupby(["persona_id", "regime"])["r_" + m].mean().reset_index()
        groups = [pmean.loc[pmean.regime == r, "r_" + m].to_numpy() for r in REGIME_ORDER]
        groups = [g for g in groups if len(g) > 1]
        n = sum(len(g) for g in groups)
        if len(groups) < 2:
            rows.append({"metric": label, "n_personas": n})
            continue
        h, p = kruskal(*groups)
        eps2 = (h - len(groups) + 1) / (n - len(groups))
        rho = spearmanr(pmean["regime"].map(REGIME_LEVEL), pmean["r_" + m])
        rows.append({"metric": label, "n_personas": n,
                     **{f"mean_rank_{r}": round(pmean.loc[pmean.regime == r, "r_" + m].mean(), 3) for r in REGIME_ORDER},
                     "H": round(h, 2), "eps2": round(max(eps2, 0), 3), "p_kw": p,
                     "trend_rho": round(rho.statistic, 3), "p_trend": rho.pvalue})
    df = pd.DataFrame(rows)
    if "p_kw" in df:
        ok = df["p_kw"].notna()
        df.loc[ok, "p_kw_bh"] = multipletests(df.loc[ok, "p_kw"], method="fdr_bh")[1]
        df.loc[ok, "p_trend_bh"] = multipletests(df.loc[ok, "p_trend"], method="fdr_bh")[1]
        df["sig"] = (df["p_kw_bh"] < ALPHA) | (df["p_trend_bh"] < ALPHA)
    return df


def check4_mode(pm_test, pm_base):
    import statsmodels.formula.api as smf
    rows = []
    a = _multi_regime(pm_test).assign(mode=1)
    b = _multi_regime(pm_base).assign(mode=0)
    both = pd.concat([a, b], ignore_index=True)
    both["level"] = both["regime"].map(REGIME_LEVEL)
    for m, label in METRICS.items():
        d = both.dropna(subset=["r_" + m]).rename(columns={"r_" + m: "y"})
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm("y ~ level * mode", d, groups=d["persona_id"]).fit(reml=False)
            rows.append({"metric": label, "n_obs": len(d),
                         "slope_base": round(fit.params["level"], 4),
                         "slope_test": round(fit.params["level"] + fit.params["level:mode"], 4),
                         "interaction": round(fit.params["level:mode"], 4),
                         "p_interaction": fit.pvalues["level:mode"]})
        except Exception as e:
            rows.append({"metric": label, "n_obs": len(d), "error": str(e)[:80]})
    df = pd.DataFrame(rows)
    ok = df["p_interaction"].notna()
    df.loc[ok, "p_interaction_bh"] = multipletests(df.loc[ok, "p_interaction"], method="fdr_bh")[1]
    df["pass"] = df["p_interaction_bh"] < ALPHA
    return df


def _parse_reps(items):
    reps = {}
    for it in items:
        name, root = it.split("=", 1)
        reps[name.strip()] = os.path.expanduser(root.strip())
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", action="append", required=True)
    ap.add_argument("--test", nargs="+")
    ap.add_argument("--baseline")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    reps = _parse_reps(args.rep)
    os.makedirs(args.out_dir, exist_ok=True)
    raw, pm, sets = {}, {}, {}
    for name, root in reps.items():
        raw[name] = load_rep(root)
        pm[name] = persona_metrics(raw[name])
        sets[name] = triplet_sets(raw[name])
        pm[name].to_csv(os.path.join(args.out_dir, f"persona_metrics_{name}.csv"), index=False)
        cov = pm[name].groupby(["text", "regime"]).size().unstack(fill_value=0)
        print(f"\n[{name}] {root}\n  texts={pm[name].text.nunique()} personas={pm[name].persona_id.nunique()} rows={len(pm[name])}")
        print("  " + cov.to_string().replace("\n", "\n  "))

    test = args.test or list(reps)[:2]
    a = test[0]
    b = test[1] if len(test) > 1 and test[1] in reps else None
    summary = []
    pd.set_option("display.width", 200)

    print("\n" + "-" * 80 + "\nCHECK 1  Is a persona's graph reproducible?  (within-text rank, ICC(2,1))")
    if b:
        c1 = check1_reliability(pm[a], pm[b])
        c1.to_csv(os.path.join(args.out_dir, "check1_reliability.csv"), index=False)
        print(c1.to_string(index=False))
        n_pass = int(c1["pass"].sum())
        summary.append(f"CHECK 1 reliability ({a} vs {b}): {n_pass}/{len(c1)} metrics with ICC >= {ICC_PASS} -> {'PASS' if n_pass >= len(c1) / 2 else 'FAIL'}")
    else:
        print("  skipped: needs two runs of the same mode")
        summary.append("CHECK 1 reliability: SKIPPED (one run only)")

    print("\n" + "-" * 80 + "\nCHECK 2  Do personas differ more than reruns?  (Jaccard overlap, per persona, Wilcoxon)")
    c2 = check2_fingerprint(sets[a], sets[b] if b else None)
    c2.to_csv(os.path.join(args.out_dir, "check2_fingerprint.csv"), index=False)
    print(c2.to_string(index=False))
    t = c2[c2.unit == "triplets"].iloc[0]
    summary.append(f"CHECK 2 fingerprint (triplets): J self={t.J_self:.3f} same-regime={t.J_same_regime:.3f} "
                   f"diff-regime={t.J_diff_regime:.3f}; self>same-regime {'PASS' if t.pass_persona else 'FAIL'} "
                   f"(p={t.p_self_gt_same:.3g}); same-regime>diff-regime {'PASS' if t.pass_regime else 'FAIL'} "
                   f"(p={t.p_same_gt_diff:.3g})")

    print("\n" + "-" * 80 + "\nCHECK 3  Does education level shift the graph?  (persona mean within-text rank, KW + trend, BH)")
    sig_sets = {}
    for name in [x for x in (a, b) if x]:
        c3 = check3_regime(pm[name])
        c3.to_csv(os.path.join(args.out_dir, f"check3_regime_{name}.csv"), index=False)
        print(f"[{name}]\n" + c3.to_string(index=False))
        sig_sets[name] = c3
    if b:
        ca, cb = sig_sets[a], sig_sets[b]
        rep_ok = ca["sig"] & cb["sig"] & (np.sign(ca["trend_rho"]) == np.sign(cb["trend_rho"]))
        rep_metrics = ca.loc[rep_ok, "metric"].tolist()
        summary.append(f"CHECK 3 regime effect replicated in both runs: {rep_metrics or 'none'} -> {'PASS' if rep_metrics else 'FAIL'}")
    else:
        sm = sig_sets[a].loc[sig_sets[a]["sig"].fillna(False), "metric"].tolist()
        summary.append(f"CHECK 3 regime effect ({a}, single run, unreplicated): {sm or 'none'}")

    print("\n" + "-" * 80 + "\nCHECK 4  Is it caused by the prompt change?  (rank ~ level * mode, persona random effect)")
    if args.baseline and args.baseline in pm:
        c4 = check4_mode(pm[a], pm[args.baseline])
        c4.to_csv(os.path.join(args.out_dir, "check4_mode.csv"), index=False)
        print(c4.to_string(index=False))
        pm_ = c4.loc[c4["pass"].fillna(False), "metric"].tolist()
        summary.append(f"CHECK 4 mode x regime interaction ({a} vs {args.baseline}): {pm_ or 'none'} -> {'PASS' if pm_ else 'FAIL'}")
    else:
        print("  skipped: pass --baseline <rep name>")
        summary.append("CHECK 4 mode contrast: SKIPPED (no --baseline)")

    print("\n" + "=" * 80 + "\nSUMMARY")
    for s in summary:
        print("  " + s)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary) + "\n")
    print(f"\nsaved to {args.out_dir}")


if __name__ == "__main__":
    main()
