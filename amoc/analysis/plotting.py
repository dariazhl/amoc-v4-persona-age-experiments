import numpy as np
import pandas as pd

LOG_SCALE_METRICS = {
    "num_triplets",
    "num_unique_concepts",
    "graph_num_nodes",
    "graph_num_edges",
    "triplets_per_100_tokens",
}


def annotate_stats(ax, pearson_r, pearson_p, spearman_r, spearman_p):
    text = (
        f"Pearson r = {pearson_r:.3f} (p={pearson_p:.3g})\n"
        f"Spearman r = {spearman_r:.3f} (p={spearman_p:.3g})"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="black"),
    )


def maybe_log_transform(
    df: pd.DataFrame,
    metric: str,
):
    if metric in LOG_SCALE_METRICS:
        return np.log10(df[metric] + 1), f"log10({metric} + 1)"
    else:
        return df[metric], metric
