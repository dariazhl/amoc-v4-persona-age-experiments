import re
import pandas as pd
from typing import List
from amoc.metrics.aggregation import process_triplets_file

REGIME_RE = re.compile(
    r"_triplets_(primary|secondary|highschool|university)(?:_.*)?\.csv$"
)


def build_persona_stats(triplet_files: List[str]) -> pd.DataFrame:
    stats_dfs = []

    for path in triplet_files:
        df = process_triplets_file(path)
        if df is None or df.empty:
            continue

        m = REGIME_RE.search(path)
        if not m:
            raise RuntimeError(f"Cannot infer regime from {path}")

        df["regime"] = m.group(1)
        stats_dfs.append(df)

    if not stats_dfs:
        raise RuntimeError("No usable stats produced")

    return pd.concat(stats_dfs, ignore_index=True)
