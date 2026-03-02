import os
import pandas as pd
from typing import List
import re


def find_triplet_files(input_dir: str, safe_tag: str) -> List[str]:
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().startswith(f"model_{safe_tag.lower()}")
        and "_final_triplets_" in f.lower()
        and f.lower().endswith(".csv")
    ]


def save_persona_outputs(
    df_trimmed: pd.DataFrame,
    df_clean: pd.DataFrame,
    removed_personas: set,
    model_tag: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df_trimmed.to_csv(
        os.path.join(out_dir, f"{model_tag}_persona_stats_quantile_trimmed.csv"),
        index=False,
    )

    df_clean.to_csv(
        os.path.join(out_dir, f"{model_tag}_persona_stats_iqr_winsorized.csv"),
        index=False,
    )

    pd.DataFrame({"original_index": list(removed_personas)}).to_csv(
        os.path.join(out_dir, f"{model_tag}_removed_personas.csv"),
        index=False,
    )


def filter_bins_by_min_n(df, bin_col="age_bin", min_n=1):
    counts = df[bin_col].value_counts()
    valid = counts[counts >= min_n].index
    dropped = set(counts.index) - set(valid)

    if dropped:
        print(f"Dropping age bins with n < {min_n}: {sorted(dropped)}")

    return df[df[bin_col].isin(valid)]
