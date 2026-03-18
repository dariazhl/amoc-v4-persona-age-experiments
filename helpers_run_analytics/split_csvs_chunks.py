import os
import pandas as pd
from math import ceil

INPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs/personas_refined_age"
OUTPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs/personas_refined_age/chunks"
ROWS_PER_CHUNK = 50


def extract_regime(filename: str) -> str:
    return filename.split("_FINAL_")[0]


def split_csv_into_chunks():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        regime = extract_regime(filename)

        print(f"Processing {filename} (regime={regime})")

        df = pd.read_csv(input_path)
        total_rows = len(df)

        if total_rows == 0:
            print(f"  [Skip] Empty file")
            continue

        num_chunks = ceil(total_rows / ROWS_PER_CHUNK)

        for i in range(num_chunks):
            start = i * ROWS_PER_CHUNK
            end = start + ROWS_PER_CHUNK

            chunk_df = df.iloc[start:end]

            chunk_filename = f"{regime}_{i:03d}.csv"
            chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)

            chunk_df.to_csv(chunk_path, index=False, encoding="utf-8")

            print(
                f"  → Wrote {chunk_filename} "
                f"(rows {start}–{min(end, total_rows) - 1})"
            )


if __name__ == "__main__":
    split_csv_into_chunks()
