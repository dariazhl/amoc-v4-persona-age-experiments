import os
import re
import glob
import hashlib
import argparse
import itertools
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
from statsmodels.stats.multitest import multipletests

from amoc.metrics.aggregation import build_persona_record


METRICS = [
    "num_triplets",
    "num_unique_concepts",
    "graph_density",
    "triplets_per_100_tokens",
    "graph_avg_degree",
    "abstract_relation_ratio",
    "abstract_concept_ratio",
    "graph_largest_component_ratio",
]

TEXT_ORDER_HINT = ["primary", "secondary", "highschool", "high_school", "college", "university"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _load_text_labels(text_dir: Optional[str]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not text_dir or not os.path.isdir(text_dir):
        return labels
    for path in glob.glob(os.path.join(text_dir, "*.txt")):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as fh:
                labels[_norm(fh.read())] = stem
        except OSError:
            continue
    return labels


def _story_label(story_text: str, label_map: Dict[str, str]) -> str:
    norm = _norm(story_text)
    if norm in label_map:
        return label_map[norm]
    for known_norm, stem in label_map.items():
        if norm.startswith(known_norm[:80]) or known_norm.startswith(norm[:80]):
            return stem
    return "story_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


def _run_dir_of(path: str) -> str:
    p = os.path.dirname(os.path.abspath(path))
    parts = p.split(os.sep)
    if parts[-2:] == ["triplets", "triplets_final_state"]:
        return os.sep.join(parts[:-2])
    return p


def _recover_story_from_run_dir(run_dir: str) -> str:
    for mp in sorted(glob.glob(os.path.join(run_dir, "matrix", "amoc_matrix_*.csv"))):
        try:
            mdf = pd.read_csv(mp, index_col=0, nrows=1)
        except Exception:
            continue
        if len(mdf.index) and str(mdf.index[0]) == "story_text" and mdf.shape[1] > 0:
            val = str(mdf.iloc[0, 0])
            if val.strip():
                return val
    return ""


def _run_dir_label(run_dir: str, label_map: Dict[str, str],
                   overrides: Optional[Dict[str, str]] = None) -> str:
    base = os.path.basename(os.path.normpath(run_dir))
    if overrides:
        if run_dir in overrides:
            return overrides[run_dir]
        if base in overrides:
            return overrides[base]
    story = _recover_story_from_run_dir(run_dir)
    if story:
        label = _story_label(story, label_map)
        if not label.startswith("story_"):
            return label
    return base


def parse_label_overrides(items: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--label expects run_dir=label, got: {item!r}")
        key, val = item.split("=", 1)
        out[os.path.basename(os.path.normpath(key.strip()))] = val.strip()
    return out


def _as_dir_list(input_dir: Union[str, List[str]]) -> List[str]:
    dirs = [input_dir] if isinstance(input_dir, (str, os.PathLike)) else list(input_dir)
    seen, out = set(), []
    for d in dirs:
        d = os.fspath(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def discover_files(input_dir: Union[str, List[str]], model_name: str) -> List[str]:
    raw_safe_tag = model_name.replace("/", "-").replace(":", "-").replace(" ", "_")
    files: List[str] = []
    for d in _as_dir_list(input_dir):
        pattern = os.path.join(
            d,
            "triplets",
            "triplets_final_state",
            f"model_{raw_safe_tag}_paper_final_triplets_*.csv",
        )
        found = sorted(glob.glob(pattern))
        if not found:
            found = sorted(
                p for p in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
                if raw_safe_tag.lower() in os.path.basename(p).lower()
                and "final_triplets" in os.path.basename(p).lower()
            )
        if not found:
            found = sorted(
                p for p in glob.glob(os.path.join(d, "**", "*.csv"), recursive=True)
                if raw_safe_tag.lower() in os.path.basename(p).lower()
                and "quantile_trimmed" in os.path.basename(p).lower()
            )
        files.extend(found)
    seen, out = set(), []
    for p in files:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


EMPTY_LONG_COLS = ["persona_id", "regime", "text", "model_name"] + METRICS


def build_long(input_dir: Union[str, List[str]], model_name: str,
               text_dir: Optional[str], require_min_texts: int = 3,
               label_overrides: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    files = discover_files(input_dir, model_name)
    if not files:
        raise FileNotFoundError(
            f"No triplet CSVs found for {model_name} under {input_dir}"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path, engine="python", on_bad_lines="warn")
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        required = {"original_index", "age_refined", "persona_text",
                    "subject", "object", "regime"}
        if not required <= set(df.columns):
            continue
        df["source_file"] = os.path.basename(path)
        df["run_dir"] = _run_dir_of(path)
        if "model_name" not in df.columns:
            df["model_name"] = model_name
        frames.append(df)

    if not frames:
        raise RuntimeError("Triplet files found but none had the required columns "
                           "(need original_index, age_refined, persona_text, "
                           "subject, object, regime).")

    raw = pd.concat(frames, ignore_index=True)
    raw["persona_id"] = raw["persona_text"].astype(str).map(
        lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest()
    )
    raw["story_id"] = raw["run_dir"]

    n_stories = raw["story_id"].nunique()
    if n_stories < require_min_texts:
        print(f"[repeated-measures] only {n_stories} distinct text(s)/run-dir(s) "
              f"found under {input_dir}; need >= {require_min_texts} for a "
              f"repeated-measures test. Skipping (expected for a single-text run).")
        return pd.DataFrame(columns=EMPTY_LONG_COLS)

    label_map = _load_text_labels(text_dir)
    text_by_run_dir = {
        rd: _run_dir_label(rd, label_map, label_overrides)
        for rd in raw["story_id"].unique()
    }

    group_cols = ["persona_id", "story_id", "regime", "model_name"]
    records = []
    for _, g in raw.groupby(group_cols, dropna=False):
        rec = build_persona_record(g)
        if rec is None:
            continue
        rec["text"] = text_by_run_dir[g["story_id"].iloc[0]]
        records.append(rec)

    long = pd.DataFrame(records)
    keep = ["persona_id", "regime", "text", "model_name"] + \
           [m for m in METRICS if m in long.columns]
    return long[keep]


def _ordered_texts(texts: List[str]) -> List[str]:
    rank = {t: i for i, t in enumerate(TEXT_ORDER_HINT)}
    return sorted(texts, key=lambda t: (rank.get(t, len(rank)), t))


def _rank_biserial_paired(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    t_plus = ranks[d > 0].sum()
    t_minus = ranks[d < 0].sum()
    return float((t_plus - t_minus) / ranks.sum())


def run(long: pd.DataFrame, alpha: float = 0.05):
    omnibus: List[Dict] = []
    posthoc: List[Dict] = []

    for regime, g_reg in long.groupby("regime"):
        for metric in METRICS:
            if metric not in g_reg.columns:
                continue

            wide = g_reg.pivot_table(index="persona_id", columns="text",
                                     values=metric, aggfunc="mean")
            wide = wide.dropna(axis=0, how="any")
            texts = _ordered_texts(list(wide.columns))
            n_blocks = len(wide)

            if len(texts) < 3 or n_blocks < 3:
                omnibus.append({
                    "regime": regime, "metric": metric, "n_texts": len(texts),
                    "n_personas": n_blocks, "friedman_chi2": np.nan,
                    "p_value": np.nan, "kendall_w": np.nan,
                    "significant": False,
                    "note": "skipped: need >=3 texts and >=3 complete personas",
                })
                continue

            cols = [wide[t].values for t in texts]
            chi2, p = friedmanchisquare(*cols)
            kendall_w = chi2 / (n_blocks * (len(texts) - 1))
            omnibus.append({
                "regime": regime, "metric": metric, "n_texts": len(texts),
                "n_personas": n_blocks, "friedman_chi2": round(float(chi2), 4),
                "p_value": float(p), "kendall_w": round(float(kendall_w), 4),
                "significant": bool(p < alpha), "note": "",
            })

            pair_rows: List[Dict] = []
            for t1, t2 in itertools.combinations(texts, 2):
                a, b = wide[t1].values, wide[t2].values
                if np.allclose(a, b):
                    w_stat, p_w = np.nan, 1.0
                else:
                    try:
                        w_stat, p_w = wilcoxon(a, b)
                    except ValueError:
                        w_stat, p_w = np.nan, 1.0
                pair_rows.append({
                    "regime": regime, "metric": metric,
                    "text_1": t1, "text_2": t2, "n_personas": n_blocks,
                    "W": (round(float(w_stat), 4) if not np.isnan(w_stat) else np.nan),
                    "p_raw": float(p_w),
                    "effect_size_rb": round(_rank_biserial_paired(a, b), 4),
                })

            pvals = [r["p_raw"] for r in pair_rows]
            if pvals:
                _, p_corr, _, _ = multipletests(pvals, method="fdr_bh")
                for r, pc in zip(pair_rows, p_corr):
                    r["p_corrected"] = round(float(pc), 6)
                    r["significant"] = bool(pc < alpha)
            posthoc.extend(pair_rows)

    omnibus_cols = ["regime", "metric", "n_texts", "n_personas",
                    "friedman_chi2", "p_value", "kendall_w", "significant", "note"]
    posthoc_cols = ["regime", "metric", "text_1", "text_2", "n_personas",
                    "W", "p_raw", "p_corrected", "effect_size_rb", "significant"]
    omnibus_df = pd.DataFrame(omnibus, columns=omnibus_cols)
    posthoc_df = (pd.DataFrame(posthoc, columns=posthoc_cols)
                  if posthoc else pd.DataFrame(columns=posthoc_cols))
    return omnibus_df, posthoc_df


def _safe_tag(model_name: str) -> str:
    return (model_name.replace("/", "-").replace(":", "-")
            .replace(" ", "_").lower())


def analyze(input_dir: Union[str, List[str]], model_name: str, output_dir: str,
            text_dir: Optional[str] = "tusa_text/min_drp_texts",
            alpha: float = 0.05, model_tag: Optional[str] = None,
            require_min_texts: int = 3,
            label_overrides: Optional[Dict[str, str]] = None):
    long = build_long(input_dir, model_name, text_dir,
                      require_min_texts=require_min_texts,
                      label_overrides=label_overrides)
    if long.empty:
        return pd.DataFrame(columns=["regime", "metric"]), pd.DataFrame()

    print(f"[repeated-measures] {len(long)} persona x text rows | "
          f"regimes={sorted(long['regime'].unique())} | "
          f"texts={sorted(long['text'].unique())}")

    omnibus_df, posthoc_df = run(long, alpha=alpha)

    os.makedirs(output_dir, exist_ok=True)
    tag = model_tag or _safe_tag(model_name)
    o_path = os.path.join(output_dir, f"{tag}_friedman_omnibus.csv")
    p_path = os.path.join(output_dir, f"{tag}_wilcoxon_posthoc.csv")
    omnibus_df.to_csv(o_path, index=False)
    posthoc_df.to_csv(p_path, index=False)
    print(f"[repeated-measures] saved omnibus -> {o_path}")
    print(f"[repeated-measures] saved post-hoc -> {p_path}")

    sig = omnibus_df[omnibus_df["significant"]]
    print(f"[repeated-measures] {len(sig)}/{len(omnibus_df)} (regime,metric) "
          f"omnibus tests significant after Friedman (alpha={alpha})")
    if not sig.empty:
        print(sig[["regime", "metric", "friedman_chi2", "p_value",
                   "kendall_w"]].to_string(index=False))
    return omnibus_df, posthoc_df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, nargs="+",
                    help="One or more dirs, each containing "
                         "triplets/triplets_final_state/*.csv.")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--text-dir", default="tusa_text/min_drp_texts",
                    help="Dir of *.txt reading passages, used to label texts")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--label", nargs="*", default=None,
                    help="Explicit run_dir=label overrides for text names, "
                         "e.g. --label run_228474=primary")
    args = ap.parse_args()
    analyze(args.input_dir, args.model, args.output_dir,
            text_dir=args.text_dir, alpha=args.alpha,
            label_overrides=parse_label_overrides(args.label))


if __name__ == "__main__":
    main()
