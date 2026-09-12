
import argparse
import os
import shutil

import pandas as pd


def clamp_run_onsets(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    df = df.copy()
    token_col = df.columns[0]
    sent_cols = list(df.columns[1:])
    changed = 0

    for idx, row in df.iterrows():
        if str(row[token_col]).strip() == "story_text":
            continue
        prev = 0.0
        for col in sent_cols:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                prev = 0.0
                continue
            if val > 0.0 and prev == 0.0 and val != 5.0:
                df.at[idx, col] = 5.0
                changed += 1
            prev = val
    return df, changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--input-dir", default="study_results/study1_landscape/matrix")
    ap.add_argument("--output-dir", default="study_results/study1_landscape/matrix_corrected")
    args = ap.parse_args()

    n_files, n_cells = 0, 0
    for root, _dirs, files in os.walk(args.input_dir):
        rel = os.path.relpath(root, args.input_dir)
        out_root = args.output_dir if rel == "." else os.path.join(args.output_dir, rel)
        os.makedirs(out_root, exist_ok=True)
        for name in sorted(files):
            if not name.endswith(".csv"):
                continue
            src = os.path.join(root, name)
            dst = os.path.join(out_root, name)
            if name.startswith("landscape_"):
                shutil.copy2(src, dst)
                continue
            try:
                df = pd.read_csv(src)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"SKIP {src}: {e}")
                continue
            corrected, changed = clamp_run_onsets(df)
            corrected.to_csv(dst, index=False)
            n_files += 1
            n_cells += changed
            print(f"{os.path.join(rel, name) if rel != '.' else name}: {changed} onset cells -> 5.0")

    print(f"\nDone: {n_files} matrices corrected, {n_cells} onset cells clamped, "
          f"output in {args.output_dir}")


if __name__ == "__main__":
    main()
