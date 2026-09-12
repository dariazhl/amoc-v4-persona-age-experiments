
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEARMAN_DIR = os.path.join(
    os.path.dirname(_HERE), "study_results", "study1_landscape"
)
if _SPEARMAN_DIR not in sys.path:
    sys.path.insert(0, _SPEARMAN_DIR)

from spearman_correlation import (
    TOKEN_MAP,
    load_amoc_matrix,
    load_landscape_matrix,
    wide_to_long,
    align_to_landscape,
    age_from_filename,
    age_to_regime,
)

REGIME_META = {
    "primary":     ("Primary",     "P1--P40",    40),
    "secondary":   ("Secondary",   "P41--P74",   34),
    "high_school": ("High school", "P81--P120",  40),
    "university":  ("University",  "P121--P159", 39),
}
REGIME_ORDER = ["primary", "secondary", "high_school", "university"]

N_SENTENCES = 13
S_COLS = [str(i) for i in range(1, N_SENTENCES + 1)]



def compute_per_sentence_spearman(
    amoc_csv: str,
    landscape_wide: pd.DataFrame,
    landscape_tokens: set,
) -> pd.DataFrame | None:
    try:
        amoc_wide = load_amoc_matrix(amoc_csv)
    except (ValueError, Exception) as e:
        print(f"  SKIP {os.path.basename(amoc_csv)}: {e}")
        return None

    _, merged = align_to_landscape(amoc_wide, landscape_wide, landscape_tokens)
    if merged.empty:
        print(f"  SKIP {os.path.basename(amoc_csv)}: no overlapping tokens.")
        return None

    rows = []
    for s in range(1, N_SENTENCES + 1):
        sub = merged[merged["sentence"] == s]
        if len(sub) < 2:
            r, p = float("nan"), float("nan")
        else:
            r, p = spearmanr(sub["score_amoc"], sub["score_land"])
        rows.append({
            "sentence":    s,
            "spearman_r":  float(r) if not np.isnan(float(r)) else float("nan"),
            "p_value":     float(p) if not np.isnan(float(p)) else float("nan"),
            "significant": (not np.isnan(float(p))) and (float(p) < 0.05),
        })
    return pd.DataFrame(rows)



def aggregate_by_regime(
    per_file: list[tuple[str, pd.DataFrame]]
) -> pd.DataFrame:
    records = []
    for fname, df in per_file:
        regime = age_to_regime(age_from_filename(fname))
        df = df.copy()
        df["regime"] = regime
        df["persona_file"] = fname
        records.append(df)

    if not records:
        return pd.DataFrame()

    all_rows = pd.concat(records, ignore_index=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        valid = g["spearman_r"].dropna()
        mean_r = float(valid.mean()) if len(valid) > 0 else float("nan")
        return pd.Series({
            "mean_spearman_r": round(mean_r, 2) if not np.isnan(mean_r) else float("nan"),
            "n_significant":   int(g["significant"].sum()),
            "n_personas":      int(g["persona_file"].nunique()),
        })

    grouped = (
        all_rows.groupby(["regime", "sentence"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    grouped["n_significant"] = grouped["n_significant"].astype(int)
    grouped["n_personas"] = grouped["n_personas"].astype(int)

    regime_rank = {r: i for i, r in enumerate(REGIME_ORDER)}
    grouped["_rank"] = grouped["regime"].map(lambda r: regime_rank.get(r, 99))
    grouped = grouped.sort_values(["_rank", "sentence"]).drop(columns="_rank")
    return grouped.reset_index(drop=True)



def fmt_r(val) -> str:
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return "---"


def build_latex_table(df: pd.DataFrame) -> str:
    pivot_r = df.pivot(index="regime", columns="sentence", values="mean_spearman_r")
    pivot_n = df.pivot(index="regime", columns="sentence", values="n_significant")

    sentences = list(range(1, N_SENTENCES + 1))
    col_spec = "lll l " + " ".join(["c"] * N_SENTENCES)
    sent_headers = " & ".join(f"\\textbf{{S{s}}}" for s in sentences)

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        (r"\caption{Mean Spearman correlations between AMoC v5.0 and the "
         r"Landscape Model by sentence and regime}"),
        r"\label{tab:mean-spearman-by-sentence-regime}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        (r"\textbf{Regime} & \textbf{Personas} & \textbf{Nr.} & \textbf{Statistic}"
         f"\n  & {sent_headers} \\\\"),
        r"\midrule",
    ]

    for i, regime_key in enumerate(REGIME_ORDER):
        display_label, persona_range, n_total = REGIME_META[regime_key]

        r_vals = " & ".join(
            fmt_r(pivot_r.loc[regime_key, s])
            if regime_key in pivot_r.index else "---"
            for s in sentences
        )
        n_vals = " & ".join(
            (str(int(pivot_n.loc[regime_key, s]))
             if pd.notna(pivot_n.loc[regime_key, s]) else "---")
            if regime_key in pivot_n.index else "---"
            for s in sentences
        )

        lines += [
            (rf"\multirow{{2}}{{*}}{{{display_label}}}"
             f"\n  & \\multirow{{2}}{{*}}{{{persona_range}}}"
             f"\n  & \\multirow{{2}}{{*}}{{{n_total}}}"
             f"\n  & Mean $\\rho$"
             f"\n  & {r_vals} \\\\"),
            (f"&\n&\n& Significant $p$ count"
             f"\n  & {n_vals} \\\\"),
        ]
        if i < len(REGIME_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"{\footnotesize",
        (r"Mean $\rho$ is the average persona-level Spearman correlation for each "
         r"sentence within regime. Significant $p$ count reports the number of "
         r"persona-level Spearman correlations with $p < .05$ out of the personas "
         r"available in that regime (40 for Primary and High school, "
         r"34 for Secondary, and 39 for University)."),
        r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "AMoC v5.0 Landscape Study: compute Spearman correlations per persona, "
            "aggregate by educational regime, and render the LaTeX table."
        )
    )
    p.add_argument(
        "--matrix-dir", required=True,
        help="Directory containing formatted_amoc_matrix_*.csv files (fixC output).",
    )
    p.add_argument(
        "--landscape",
        default="study_results/study1_landscape/matrix/landscape_paper_no_inference.csv",
        help="Path to the 17-token landscape reference (space-separated). "
             "Default: study_results/study1_landscape/matrix/landscape_paper_no_inference.csv",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Directory for outputs: regime_means.csv + landscape_spearman.tex.",
    )
    p.add_argument(
        "--pattern", default="formatted_amoc_matrix_*.csv",
        help="Glob pattern within --matrix-dir. Default: formatted_amoc_matrix_*.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isfile(args.landscape):
        sys.exit(f"ERROR: landscape file not found: {args.landscape}")
    landscape_wide = load_landscape_matrix(args.landscape)
    landscape_tokens = set(landscape_wide["token"])
    print(f"Landscape loaded: {len(landscape_tokens)} tokens from {args.landscape}")

    files = sorted(glob.glob(os.path.join(args.matrix_dir, args.pattern)))
    if not files:
        sys.exit(f"ERROR: no files matching '{args.pattern}' in {args.matrix_dir}")
    print(f"Processing {len(files)} matrices from {args.matrix_dir} …\n")

    per_file = []
    for f in files:
        result_df = compute_per_sentence_spearman(f, landscape_wide, landscape_tokens)
        if result_df is not None:
            per_file.append((os.path.basename(f), result_df))

    print(f"\n{len(per_file)}/{len(files)} matrices processed successfully.")

    if not per_file:
        sys.exit("ERROR: no matrices produced valid correlations.")

    regime_df = aggregate_by_regime(per_file)
    regime_csv = os.path.join(args.output_dir, "regime_means.csv")
    regime_df.to_csv(regime_csv, index=False)
    print(f"\nRegime means written to: {regime_csv}")
    print(regime_df.to_string(index=False))

    latex = build_latex_table(regime_df)
    tex_path = os.path.join(args.output_dir, "landscape_spearman.tex")
    with open(tex_path, "w") as fh:
        fh.write(latex + "\n")
    print(f"\nLaTeX table written to: {tex_path}")


if __name__ == "__main__":
    main()
