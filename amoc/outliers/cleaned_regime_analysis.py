import os
import pandas as pd
from scipy.stats import kruskal
import glob
from typing import Iterable
from amoc.outliers.io import filter_bins_by_min_n
from amoc.analysis.age_regimes import assign_age_bin, coarse_age_bin

from amoc.analysis.regime_plots import (
    plot_violin_box,
    plot_ecdf,
    plot_boxplot,
    plot_median_ci,
    plot_pairwise_median_diff,
    plot_age_ecdf,
    plot_age_violin,
    plot_age_box,
    plot_age_median_ci,
    plot_regime_age_heatmap,
    plot_discrete_age_frequencies,
    plot_boxplot_by_age,
    plot_violin_box_by_age,
    plot_boxplot_by_age_bin,
    plot_violin_box_by_age_bin,
)

QUANTILE_TRIM_METRICS = [
    "num_triplets",
    "num_unique_concepts",
    "triplets_per_100_tokens",
    "graph_num_nodes",
    "graph_num_edges",
]

IQR_WINSORIZED_METRICS = [
    "graph_density",
    "graph_avg_degree",
]

# AGE_BIN_ORDER = [
#     "5-6",
#     "7-8",
#     "9-10",
#     "11-12",
#     "13-14",
#     "15-16",
#     "17-18",
# ]
AGE_BIN_ORDER = [
    "3–10",
    "11–14",
    "15–18",
]


def normalize_age_bin(x: str | None) -> str | None:
    if pd.isna(x):
        return None
    return str(x).strip().replace("-", "–").replace("—", "–")


def normalize_regime(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("_", "", regex=False)
    )


def winsorize_by_age(
    df: pd.DataFrame,
    age_col: str,
    metric_col: str,
    k: float = 1.5,
    min_n: int = 5,
) -> pd.DataFrame:
    df = df.copy()

    if age_col not in df.columns:
        raise KeyError(f"Missing age column: {age_col}")
    if metric_col not in df.columns:
        raise KeyError(f"Missing metric column: {metric_col}")

    for age, g in df.groupby(age_col, observed=True):
        if len(g) < min_n:
            # too few samples - skip
            continue

        q1 = g[metric_col].quantile(0.25)
        q3 = g[metric_col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 or pd.isna(iqr):
            continue

        lo = q1 - k * iqr
        hi = q3 + k * iqr

        mask = df[age_col] == age
        df.loc[mask, metric_col] = df.loc[mask, metric_col].clip(lo, hi)

    return df


def winsorize_by_group(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    k: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()

    for group, g in df.groupby(group_col, observed=True):
        q1 = g[metric_col].quantile(0.25)
        q3 = g[metric_col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lo = q1 - k * iqr
        hi = q3 + k * iqr

        mask = df[group_col] == group
        df.loc[mask, metric_col] = df.loc[mask, metric_col].clip(lo, hi)

    return df


# Kruskal-Willis test = 1-way ANOVA
def _run_kruskal(df: pd.DataFrame, metric: str) -> None:
    groups = [
        g[metric].dropna().values
        for _, g in df.groupby("regime", observed=True)
        if g[metric].notna().any()
    ]

    if len(groups) >= 2:
        h, p = kruskal(*groups)
        print(f"  Kruskal–Wallis H={h:.3f}, p={p:.4g}")


# Attach persona attributes to persona stats
def attach_persona_attributes(
    df_stats: pd.DataFrame,
    refined_age_dir: str,
    required_cols: Iterable[str] = ("persona_text", "age_refined"),
) -> pd.DataFrame:
    required_cols = tuple(required_cols)

    if "persona_text" not in df_stats.columns:
        raise RuntimeError(
            "df_stats is missing 'persona_text'. "
            f"Available columns: {list(df_stats.columns)}"
        )

    pattern = os.path.join(refined_age_dir, "*_age_refined.csv")
    print("Searching refined-age files with pattern:", pattern)

    files = glob.glob(pattern)
    if not files:
        raise RuntimeError(f"No refined-age files found in {refined_age_dir}")

    dfs = []
    for path in files:
        df = pd.read_csv(path)

        missing = set(required_cols) - set(df.columns)
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {missing}")

        dfs.append(df[list(required_cols)])

    all_attrs = pd.concat(dfs, ignore_index=True)

    # warn on duplicate persona ids
    vc = all_attrs["persona_text"].value_counts()
    if (vc > 1).any():
        print("[WARN] Duplicate persona_text entries in refined-age files:")
        print(vc[vc > 1].head())

    attrs = all_attrs.drop_duplicates(subset="persona_text")

    # merge
    df_merged = df_stats.merge(
        attrs[["persona_text", "age_refined"]],
        on="persona_text",
        how="left",
    )

    n_missing = df_merged["age_refined"].isna().sum()
    if n_missing > 0:
        print(
            f"[WARN] {n_missing} personas missing age_refined "
            f"({100 * n_missing / len(df_merged):.1f}%)"
        )

    return df_merged


def run_cleaned_regime_analysis(
    input_dir: str,
    model: str,
    output_dir: str,
    plots: bool = False,
    plots_age: bool = False,
    refined_age_dir: str | None = None,
) -> None:
    print("\n" + "=" * 60)
    print(f"AMoC CLEANED REGIME ANALYSIS — MODEL: {model}")
    print("=" * 60)

    if plots_age and refined_age_dir is None:
        raise RuntimeError("--plots-age requires refined_age_dir")

    stats_dir = os.path.join(input_dir, "persona_stats")

    quantile_path = os.path.join(
        stats_dir,
        f"{model}_persona_stats_quantile_trimmed.csv",
    )
    winsorized_path = os.path.join(
        stats_dir,
        f"{model}_persona_stats_iqr_winsorized.csv",
    )

    if not os.path.exists(quantile_path):
        raise FileNotFoundError(f"Missing: {quantile_path}")
    if not os.path.exists(winsorized_path):
        raise FileNotFoundError(f"Missing: {winsorized_path}")

    df_q = pd.read_csv(quantile_path)
    df_w = pd.read_csv(winsorized_path)

    REGIME_ORDER = ["primary", "secondary", "highschool", "university"]

    for df in (df_q, df_w):
        if "regime" not in df.columns:
            raise RuntimeError("Missing 'regime' column")

        df["regime"] = normalize_regime(df["regime"])

        unexpected = set(df["regime"].unique()) - set(REGIME_ORDER)
        if unexpected:
            raise RuntimeError(f"Unexpected regimes found: {unexpected}")

        df["regime"] = pd.Categorical(
            df["regime"],
            categories=REGIME_ORDER,
            ordered=True,
        )

    for name, df in [("quantile", df_q), ("winsorized", df_w)]:
        if "regime" not in df.columns:
            raise RuntimeError(f"Missing 'regime' column in {name} stats")

    os.makedirs(output_dir, exist_ok=True)

    if plots:
        for metric in QUANTILE_TRIM_METRICS:
            if metric not in df_q.columns:
                continue

            plot_violin_box(
                df=df_q,
                metric=metric,
                output_dir=output_dir,
                model_tag=f"{model}_quantile_trimmed",
                order=REGIME_ORDER,
            )
            plot_ecdf(df_q, metric, output_dir, model)
            plot_boxplot(df_q, metric, output_dir, model)
            plot_median_ci(df_q, metric, output_dir, model)
            plot_pairwise_median_diff(df_q, metric, output_dir, model)
            _run_kruskal(df_q, metric)

        for metric in IQR_WINSORIZED_METRICS:
            if metric not in df_w.columns:
                continue

            plot_violin_box(
                df=df_w,
                metric=metric,
                output_dir=output_dir,
                model_tag=f"{model}_iqr_winsorized",
                order=REGIME_ORDER,
            )
            _run_kruskal(df_w, metric)

    # Age plots
    if plots_age:
        if "age_refined" not in df_q.columns:
            raise RuntimeError("age_refined missing from df_q")

        if "age_refined" not in df_w.columns:
            raise RuntimeError("age_refined missing from df_w")

        print("Age_refined summary:")
        print(df_q["age_refined"].describe())

        print("Unique ages:")
        print(sorted(df_q["age_refined"].dropna().unique())[:20], "...")

        print("Age_refined value counts:")
        print(df_q["age_refined"].value_counts().sort_index())

        df_q["age_bin"] = (
            df_q["age_refined"].apply(coarse_age_bin).apply(normalize_age_bin)
        )
        df_q = df_q.dropna(subset=["age_bin"])

        df_q["age_bin"] = pd.Categorical(
            df_q["age_bin"],
            categories=AGE_BIN_ORDER,
            ordered=True,
        )

        # df_q = df_q.dropna(subset=["age_bin"])
        unexpected = set(df_q["age_bin"].unique()) - set(AGE_BIN_ORDER)
        if unexpected:
            raise RuntimeError(f"Unexpected age_bin labels: {unexpected}")

        print("Coarse age-bin counts:")
        print(df_q["age_bin"].value_counts().sort_index())

        df_q_age = df_q.copy()
        df_q_wins_age = winsorize_by_group(
            df_q_age,
            group_col="age_bin",
            metric_col="triplets_per_100_tokens",
        )
        df_q_wins_age = winsorize_by_age(
            df_q_age,
            age_col="age_bin",
            metric_col="triplets_per_100_tokens",
            k=1.5,
            min_n=1,
        )

        print(
            "Age coverage (quantile-trimmed):",
            df_q["age_refined"].notna().mean(),
        )
        print(
            "Age coverage (winsorized):",
            df_w["age_refined"].notna().mean(),
        )

        age_out_dir = os.path.join(output_dir, "age")
        os.makedirs(age_out_dir, exist_ok=True)

        plot_regime_age_heatmap(
            df_q,
            output_dir=age_out_dir,
            model_tag=f"{model}_age_quantile_trimmed",
            normalize=True,
        )

        plot_discrete_age_frequencies(
            df_q,
            output_dir=age_out_dir,
            model_tag=f"{model}_age_quantile_trimmed",
        )
        plot_age_ecdf(df_q, age_out_dir, f"{model}_age_quantile_trimmed")

        age_metric_out = os.path.join(output_dir, "age_metrics")
        os.makedirs(age_metric_out, exist_ok=True)
        for metric in QUANTILE_TRIM_METRICS + IQR_WINSORIZED_METRICS:
            if metric not in df_q.columns:
                continue

            plot_violin_box_by_age(
                df=df_q,
                metric=metric,
                model_tag=f"{model}_quantile_trimmed",
                output_dir=age_metric_out,
            )

            plot_boxplot_by_age(
                df=df_q,
                metric=metric,
                model_tag=f"{model}_quantile_trimmed",
                output_dir=age_metric_out,
            )

            plot_violin_box_by_age_bin(
                df=df_q,
                metric=metric,
                model_tag=f"{model}_quantile_trimmed",
                output_dir=age_metric_out,
            )

            plot_boxplot_by_age_bin(
                df=df_q,
                metric=metric,
                model_tag=f"{model}_quantile_trimmed",
                output_dir=age_metric_out,
            )
