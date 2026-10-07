"""Turn raw Keefe per-item score CSVs into per-persona E/C, P/C, P/E LME stats.

The scoring loop in keefe_original_paper_personas.py writes one row per
(persona, item, condition) with a 1-4 score. Its own stats step pools every
persona in the chunk into a single LME fit, which is not what Appendix E's
per-persona table needs. This script re-fits the same model
(score ~ C(condition, Treatment('control')), random intercept by item_id)
separately for each persona and reports the same three contrasts:
Explicit/Control, Predictive/Control, Predictive/Explicit.

Usage:
    python compute_persona_lme_stats.py chunk_a.csv chunk_b.csv ... -o stats.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.formula.api import mixedlm


def lme_pairwise(df, item_col="item_id", condition_col="condition", score_col="score"):
    df = df.copy()
    df[condition_col] = pd.Categorical(
        df[condition_col],
        categories=["control", "predictive", "explicit"],
        ordered=False,
    )
    model = mixedlm(
        f"{score_col} ~ C({condition_col}, Treatment('control'))", df, groups=df[item_col]
    )
    result = model.fit()

    pred_coef = result.params.get("C(condition, Treatment('control'))[T.predictive]", np.nan)
    pred_se = result.bse.get("C(condition, Treatment('control'))[T.predictive]", np.nan)
    expl_coef = result.params.get("C(condition, Treatment('control'))[T.explicit]", np.nan)
    expl_se = result.bse.get("C(condition, Treatment('control'))[T.explicit]", np.nan)
    cov = result.cov_params().loc[
        "C(condition, Treatment('control'))[T.predictive]",
        "C(condition, Treatment('control'))[T.explicit]",
    ]
    pred_vs_expl_est = pred_coef - expl_coef
    pred_vs_expl_se = np.sqrt(pred_se**2 + expl_se**2 - 2 * cov)

    def z_p(est, se):
        if np.isnan(est) or se == 0:
            return np.nan, np.nan
        z = est / se
        p = 2 * (1 - norm.cdf(abs(z)))
        return z, p

    expl_z, expl_p = z_p(expl_coef, expl_se)
    pred_z, pred_p = z_p(pred_coef, pred_se)
    pvex_z, pvex_p = z_p(pred_vs_expl_est, pred_vs_expl_se)

    return {
        "EC_z": expl_z, "EC_p": expl_p,
        "PC_z": pred_z, "PC_p": pred_p,
        "PE_z": pvex_z, "PE_p": pvex_p,
    }


def format_p(p: float) -> str:
    if np.isnan(p):
        return "--"
    if p < 0.001:
        return r"<.001^{**}"
    return f"{p:.4f}"


def format_z(z: float) -> str:
    if np.isnan(z):
        return "--"
    return f"{z:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv", nargs="+", help="One or more raw score CSVs")
    ap.add_argument("-o", "--output", default=None, help="Optional path to save the results as CSV")
    ap.add_argument("--latex", action="store_true", help="Also print rows formatted for a LaTeX table")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    rows = []
    for path in args.input_csv:
        df = pd.read_csv(path)
        required = {"persona_text", "age", "item_id", "condition", "score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        for persona_text, g in df.groupby("persona_text", sort=False):
            stats = lme_pairwise(g)
            rows.append({
                "source_file": path,
                "persona_text": persona_text,
                "age": g["age"].iloc[0],
                **stats,
            })

    out_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 50)
    print(out_df.to_string(index=False))

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(out_df)} rows to {args.output}")

    if args.latex:
        print("\n% LaTeX rows (ID left blank -- fill in manually)")
        for _, r in out_df.iterrows():
            print(
                f"P?? & {format_z(r['EC_z'])} & ${format_p(r['EC_p'])}$ & "
                f"{format_z(r['PC_z'])} & ${format_p(r['PC_p'])}$ & "
                f"${format_z(r['PE_z'])}$ & {format_p(r['PE_p'])} "
                f"% {r['persona_text'][:60]}"
                r" \\"
            )


if __name__ == "__main__":
    main()
