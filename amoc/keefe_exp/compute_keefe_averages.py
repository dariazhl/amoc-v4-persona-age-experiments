"""Compute the grand-average E/C, P/C, P/E contrasts (tab:keefe-averages) from
the per-persona Appendix E table (tab:lme-persona-comparisons).

Method: take the arithmetic mean of each contrast's z-statistic across every
filled persona row (blank rows, e.g. personas with no data yet, are skipped),
then convert that single mean z into a p-value with the same two-sided
standard-normal formula used for each individual persona:
    p = 2 * (1 - norm.cdf(|z|))
This is NOT an average of p-values (which is not statistically meaningful);
it is the p-value implied by the averaged z-statistic.

Accepts either:
  - the raw Appendix E .tex file (rows are parsed with a regex), or
  - a CSV with EC_z, PC_z, PE_z columns (e.g. the output of
    compute_persona_lme_stats.py, after combining all chunks).

Usage:
    python compute_keefe_averages.py appendix_e.tex
    python compute_keefe_averages.py stats.csv
"""

import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import norm

CONTRASTS = [
    ("EC", "Explicit/Control"),
    ("PC", "Predictive/Control"),
    ("PE", "Predictive/Explicit"),
]


def _clean_number(raw: str):
    """Strip LaTeX wrapping ($, \\phantom{-}, ^{**}, <, whitespace) from a
    table cell and return a float, or None if the cell is blank / not a
    plain number (e.g. "<.001^{**}")."""
    s = raw.strip()
    s = s.replace(r"\phantom{-}", "")
    s = s.strip("$").strip()
    if not s:
        return None
    if s.startswith("<"):
        return None  # e.g. <.001^{**} -- not a bare number, caller uses z instead
    s = re.sub(r"\^\{.*?\}", "", s)  # drop trailing ^{**} etc.
    try:
        return float(s)
    except ValueError:
        return None


def parse_tex_rows(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.endswith(r"\\") or "&" not in line:
                continue
            if line.startswith("%") or "toprule" in line or "midrule" in line:
                continue
            body = line[: -len(r"\\")].strip()
            fields = [c.strip() for c in body.split("&")]
            if len(fields) != 8:
                continue  # not a data row (ID, Regime, EC_z, EC_p, PC_z, PC_p, PE_z, PE_p)
            _id, regime, ec_z, ec_p, pc_z, pc_p, pe_z, pe_p = fields
            ec_z_val = _clean_number(ec_z)
            pc_z_val = _clean_number(pc_z)
            pe_z_val = _clean_number(pe_z)
            if ec_z_val is None or pc_z_val is None or pe_z_val is None:
                continue  # blank row, e.g. no data for this persona yet
            rows.append({
                "id": _id, "regime": regime,
                "EC_z": ec_z_val, "PC_z": pc_z_val, "PE_z": pe_z_val,
            })
    return pd.DataFrame(rows)


def format_p(p: float) -> str:
    if p < 0.001:
        return r"<.001^{**}"
    return f"{p:.3f}".lstrip("0") or "0"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_path", help="Appendix E .tex file, or a CSV with EC_z/PC_z/PE_z columns")
    args = ap.parse_args()

    if args.input_path.endswith(".csv"):
        df = pd.read_csv(args.input_path)
    else:
        df = parse_tex_rows(args.input_path)

    print(f"Parsed {len(df)} filled persona rows\n")

    print(f"{'Contrast':22s} {'mean z':>8s} {'p':>10s}")
    results = {}
    for key, label in CONTRASTS:
        col = f"{key}_z"
        mean_z = df[col].mean()
        p = 2 * (1 - norm.cdf(abs(mean_z)))
        results[key] = (mean_z, p)
        print(f"{label:22s} {mean_z:8.2f} {format_p(p):>10s}")

    print("\n% --- LaTeX table (tab:keefe-averages) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\caption{Between-group comparisons of the LME model predictions for AMoC~v5.0.}")
    print(r"\label{tab:keefe-averages}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"\textbf{Between-Group Comparisons} & \multicolumn{2}{c}{\textbf{AMoC~v5.0}} \\")
    print(r"\cmidrule(lr){2-3}")
    print(r"& $z$ & $p$ \\")
    print(r"\midrule")
    for key, label in CONTRASTS:
        mean_z, p = results[key]
        print(f"{label:20s} & ${mean_z:.2f}$  & ${format_p(p)}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\begin{tablenotes}")
    print(r"\footnotesize")
    print(r"\item $^{**}p < .001$")
    print(r"\end{tablenotes}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
