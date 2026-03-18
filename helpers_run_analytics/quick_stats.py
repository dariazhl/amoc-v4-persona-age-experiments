#!/usr/bin/env python3
from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import List

import pandas as pd

INPUT_CSV = Path(
    "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/final_plots/jan_20/statistics/_persona_stats_quantile_trimmed.csv"
)

OUTPUT_CSV = Path(
    "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/final_plots/jan_20/statistics/regime_summary_stats.csv"
)

GROUP_COL = "regime"

METRIC_COLS: List[str] = [
    # demographics
    "age_refined",
    # abstraction
    "lexical_ttr",
    "lexical_avg_word_len",
    # graph structure
    "graph_density",
    "graph_num_components",
    "graph_avg_degree",
    "graph_num_edges",
    # scale
    "num_triplets",
    "triplets_per_100_tokens",
    "graph_num_nodes",
]

ROUND_DIGITS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def iqr(series: pd.Series) -> float:
    return series.quantile(0.75) - series.quantile(0.25)


def with_stat(df_stat: pd.DataFrame, stat_name: str) -> pd.DataFrame:
    out = df_stat.copy()
    out["stat"] = stat_name
    return out.reset_index()


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------


def main() -> None:
    logger.info("Loading dataset")
    df = pd.read_csv(INPUT_CSV)

    logger.info("Validating schema")
    validate_columns(df, [GROUP_COL] + METRIC_COLS)

    logger.info("Casting metrics to numeric")
    df[METRIC_COLS] = df[METRIC_COLS].apply(pd.to_numeric, errors="coerce")

    logger.info("Dropping rows with missing regime")
    df = df.dropna(subset=[GROUP_COL])

    logger.info("Grouping by regime")
    grouped = df.groupby(GROUP_COL)[METRIC_COLS]

    logger.info("Computing sample size by regime")
    n_by_regime = df.groupby(GROUP_COL).size().rename("n").reset_index()

    logger.info("Computing statistics")
    mean_df = grouped.mean()
    median_df = grouped.median()
    std_df = grouped.std()
    iqr_df = grouped.apply(lambda g: g.apply(iqr))

    logger.info("Stacking statistics into one table")
    summary = pd.concat(
        [
            with_stat(mean_df, "mean"),
            with_stat(median_df, "median"),
            with_stat(std_df, "std"),
            with_stat(iqr_df, "iqr"),
        ],
        ignore_index=True,
    )

    logger.info("Merging sample size")
    summary = summary.merge(n_by_regime, on=GROUP_COL, how="left")

    logger.info("Reordering columns")

    ordered_cols = [GROUP_COL, "stat", "n"] + [
        c for c in summary.columns if c not in {GROUP_COL, "stat", "n"}
    ]

    summary = summary[ordered_cols]

    logger.info("Rounding values")
    summary[METRIC_COLS] = summary[METRIC_COLS].round(ROUND_DIGITS)

    logger.info("Sorting output")

    STAT_ORDER = ["mean", "median", "std", "iqr"]

    summary["stat"] = pd.Categorical(
        summary["stat"],
        categories=STAT_ORDER,
        ordered=True,
    )

    summary = summary.sort_values(["stat", GROUP_COL]).reset_index(drop=True)

    logger.info("Saving output")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    logger.info("Analysis complete")
    logger.info("Output written to %s", OUTPUT_CSV.resolve())


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
