# python merge_csv_triplet_chunks.py \
#   --regime primary

# Merge chunked triplet CSVs into per-model, per-regime outputs.
# Expected chunk naming: model_<model>_triplets_<regime>_partXXXX.csv
# Outputs: model_<model>_triplets_<regime>.csv

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from typing import Optional

INPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/10_example_output"
OUTPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/10_example_output/merged_files"

KNOWN_REGIMES = {"primary", "secondary", "highschool", "high_school", "university"}


def parse_model_and_regime(filename: str) -> Tuple[str, str]:
    m = re.match(r"model_(.+?)_triplets_(.+)\.csv$", filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    model = m.group(1)
    tail = m.group(2)
    tokens = tail.split("_")
    regime = tokens[0]

    if regime.lower() in KNOWN_REGIMES:
        regime = regime.lower()
    return model, regime


def merge_chunks(
    input_dir: Path, output_dir: Path, regime_filter: Optional[str]
) -> None:
    csv_files = sorted(input_dir.glob("model_*_triplets_*.csv"))
    if not csv_files:
        raise RuntimeError(f"No triplet CSV files found in {input_dir}")

    grouped: Dict[Tuple[str, str], List[Path]] = {}
    for path in csv_files:
        try:
            model, regime = parse_model_and_regime(path.name)
        except ValueError:
            continue
        if regime_filter and regime != regime_filter:
            continue
        grouped.setdefault((model, regime), []).append(path)

    if not grouped:
        raise RuntimeError("No files matched the given regime pattern.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for (model, regime), files in grouped.items():
        dfs = [pd.read_csv(f) for f in files]
        merged = pd.concat(dfs, ignore_index=True)
        out_name = f"model_{model}_triplets_{regime}.csv"
        out_path = output_dir / out_name
        merged.to_csv(out_path, index=False)
        print(f"Merged {len(files)} files → {out_path} ({len(merged)} rows)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Merge chunked triplet CSVs into per-model, per-regime outputs."
    )
    ap.add_argument(
        "--regime",
        default=None,
        choices=["primary", "highschool", "secondary", "university"],
        help="Educational regime (default: highschool).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    merge_chunks(
        Path(INPUT_DIR),
        Path(OUTPUT_DIR),
        args.regime,
    )


if __name__ == "__main__":
    main()
