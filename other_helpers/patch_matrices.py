
import argparse
import glob
import os

import numpy as np
import pandas as pd

CARRYOVER_SENIORITY_WEIGHT = 0.5
MAX_BASE_SCORE = 5.0
N_SENTENCES = 13
S_COLS = [str(i) for i in range(1, N_SENTENCES + 1)]


def load_landscape(path: str) -> dict:
    ref = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("word/sentence"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = parts[0]
            scores = [float(x) for x in parts[1:N_SENTENCES + 1]]
            if len(scores) == N_SENTENCES:
                ref[token] = np.array(scores)
    return ref


def apply_fix_c(df_wide: pd.DataFrame, landscape: dict) -> pd.DataFrame:
    missing_cols = [c for c in S_COLS if c not in df_wide.columns]
    if missing_cols:
        raise ValueError(f"Missing sentence columns: {missing_cols}")

    df = df_wide.copy()
    arr = df[S_COLS].values.astype(float)
    corrected = arr.copy()

    for row_idx in range(arr.shape[0]):
        token = df.iloc[row_idx]["token"]
        scores = arr[row_idx]
        land_scores = landscape.get(token, np.zeros(N_SENTENCES))

        nonzero = np.where(scores > 0)[0]
        if len(nonzero) == 0:
            continue

        s0 = nonzero[0] + 1

        for col_idx in range(N_SENTENCES):
            s_i = col_idx + 1
            seniority = s_i - s0
            penalty = CARRYOVER_SENIORITY_WEIGHT * seniority
            raw = scores[col_idx]
            land = land_scores[col_idx]

            if raw > 0:
                corrected[row_idx, col_idx] = min(MAX_BASE_SCORE, raw + penalty)
            elif penalty >= MAX_BASE_SCORE and land > 0:
                corrected[row_idx, col_idx] = MAX_BASE_SCORE

    df[S_COLS] = corrected
    return df


def patch_file(src: str, dst: str, landscape: dict, dry_run: bool = False) -> dict:
    df = pd.read_csv(src)

    df = df.rename(columns={df.columns[0]: "token"})

    df = df[df["token"] != "story_text"].copy()

    non_token = [c for c in df.columns if c != "token"]
    non_numeric = [c for c in non_token if not str(c).strip().lstrip("-").isdigit()]
    if non_numeric:
        raise ValueError(
            f"Non-numeric columns {non_numeric} — "
            "is this a metrics CSV rather than an AMoC matrix?"
        )

    df[non_token] = df[non_token].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df_fixed = apply_fix_c(df, landscape)

    old_arr = df[S_COLS].values.astype(float)
    new_arr = df_fixed[S_COLS].values.astype(float)
    n_changed = int((old_arr != new_arr).sum())
    delta_mean = float((new_arr - old_arr).mean())

    if not dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        df_fixed.to_csv(dst, index=False)

    return {
        "file": os.path.basename(src),
        "cells_changed": n_changed,
        "mean_delta": round(delta_mean, 4),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Reverse AMoC seniority decay in formatted matrices (Fix C approximation)."
    )
    p.add_argument("--input-dir", required=True,
                   help="Directory containing formatted_amoc_matrix_*.csv files.")
    p.add_argument("--landscape", required=True,
                   help="Path to landscape reference matrix (space-separated), "
                        "used as a guard for Case B restorations.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write patched files. "
                        "Defaults to --input-dir (in-place). "
                        "Recommended: use a separate directory to keep originals.")
    p.add_argument("--pattern", default="formatted_amoc_matrix_*.csv",
                   help="Glob pattern for input files (default: formatted_amoc_matrix_*.csv).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change without writing any files.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir or args.input_dir

    landscape = load_landscape(args.landscape)
    print(f"Landscape loaded: {len(landscape)} tokens")

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}.")
        return

    mode = "DRY RUN" if args.dry_run else f"writing → {out_dir}"
    print(f"{len(files)} matrices found  [{mode}]\n")

    summaries = []
    for src in files:
        dst = os.path.join(out_dir, os.path.basename(src))
        try:
            result = patch_file(src, dst, landscape, dry_run=args.dry_run)
            summaries.append(result)
            print(f"  {result['file'][:80]}  "
                  f"cells_changed={result['cells_changed']}  "
                  f"Δmean=+{result['mean_delta']:.3f}")
        except ValueError as e:
            print(f"  SKIP {os.path.basename(src)}: {e}")

    if summaries:
        df_s = pd.DataFrame(summaries)
        print(f"\nTotal cells changed : {df_s['cells_changed'].sum()}")
        print(f"Mean score increase : +{df_s['mean_delta'].mean():.3f}")
        if not args.dry_run:
            print(f"Patched files written to: {out_dir}")


if __name__ == "__main__":
    main()
