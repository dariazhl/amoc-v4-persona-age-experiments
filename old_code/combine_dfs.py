import csv
import os
import pandas as pd


def safe_combine_csv(csv1: str, csv2: str, output_csv: str):
    # ---- Safety checks ----
    if not os.path.exists(csv1):
        raise FileNotFoundError(f"File not found: {csv1}")
    if not os.path.exists(csv2):
        raise FileNotFoundError(f"File not found: {csv2}")
    if os.path.abspath(output_csv) in {
        os.path.abspath(csv1),
        os.path.abspath(csv2),
    }:
        raise ValueError("output_csv must be different from both input files.")

    print(f"Loading:\n  {csv1}\n  {csv2}")

    # Load as text; let pandas parse CSV structure
    df1 = pd.read_csv(csv1, dtype=str, encoding="utf-8", engine="python")
    df2 = pd.read_csv(csv2, dtype=str, encoding="utf-8", engine="python")

    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")

    # Align columns (union of both sets)
    all_columns = sorted(set(df1.columns) | set(df2.columns))
    df1 = df1.reindex(columns=all_columns)
    df2 = df2.reindex(columns=all_columns)

    print(f"Final column set ({len(all_columns)}): {all_columns}")

    # ---- Prepare rows as pure strings, sanitize weird bytes ----
    def sanitize(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return ""
        s = str(value)
        # Remove null bytes and normalize newlines
        s = s.replace("\x00", "").replace("\r", " ").replace("\n", " ")
        return s

    # Convert to list-of-lists of strings
    rows = []
    for df in (df1, df2):
        for _, row in df.iterrows():
            rows.append([sanitize(row[col]) for col in all_columns])

    print(f"Total rows to write: {len(rows)}")

    # ---- Write using csv.writer (plain text) ----
    # This is the most “dumb” and reliable way to get a real text CSV.
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(all_columns)
        # Data
        writer.writerows(rows)

    print(f"Combined CSV saved to: {output_csv}")


if __name__ == "__main__":
    safe_combine_csv(
        "balanced_dfs/university_llm_judge.csv",
        "personas_dfs/high_school_students.csv",
        "balanced_dfs/university.csv",
    )
