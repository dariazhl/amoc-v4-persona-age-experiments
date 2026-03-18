import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import numpy as np
from amoc.analysis.plotting import maybe_log_transform


def bootstrap_ci(values, n_boot=5000, ci=(2.5, 97.5)):
    vals = np.array(values)
    boots = np.random.choice(vals, (n_boot, len(vals)), replace=True).mean(axis=1)
    return np.percentile(boots, ci)


def plot_violin_box(
    df: pd.DataFrame,
    metric: str,
    model_tag: str,
    regime_col: str = "regime",
    order: list | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    figsize: tuple = (8, 5),
    show_n: bool = True,
    output_dir: str | None = None,
):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in dataframe.")

    if regime_col not in df.columns:
        raise ValueError(f"Regime column '{regime_col}' not found in dataframe.")

    plot_df = df[[metric, regime_col]].dropna()
    if plot_df.empty:
        raise ValueError("No data left after dropping NaNs.")

    if order is None:
        order = sorted(plot_df[regime_col].unique())

    plt.figure(figsize=figsize)

    # --- Violin plot (distribution shape)
    sns.violinplot(
        data=plot_df,
        x=regime_col,
        y=metric,
        order=order,
        inner=None,
        cut=0,
    )

    # --- Boxplot overlay (median + IQR)
    sns.boxplot(
        data=plot_df,
        x=regime_col,
        y=metric,
        order=order,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 2},
        showfliers=False,
        whiskerprops={"linewidth": 1.5},
        zorder=2,
    )

    # --- Sample size annotation
    # n = sample sizes per educational regime.
    if show_n:
        counts = plot_df.groupby(regime_col)[metric].count()
        ymax = plot_df[metric].max()
        y_offset = 0.05 * (ymax if ymax != 0 else 1)

        for i, regime in enumerate(order):
            n = counts.get(regime, 0)
            plt.text(
                i,
                ymax + y_offset,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.xlabel("Educational regime")
    plt.ylabel(ylabel or metric)
    plt.title(title or f"{metric} by regime — {model_tag}", pad=30)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"violin_{metric}_{model_tag}.png"
        out_path = os.path.join(output_dir, fname)
        plt.savefig(out_path, dpi=300)

    plt.close()


def plot_ecdf(
    df,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    plt.figure(figsize=(6, 4))

    for regime, g in df.groupby("regime", observed=True):
        values = np.sort(g[metric].dropna().values)
        if len(values) == 0:
            continue
        y = np.arange(1, len(values) + 1) / len(values)
        plt.step(values, y, where="post", label=regime)

    plt.xlabel(metric)
    plt.ylabel("ECDF")
    plt.title(f"ECDF — {metric}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_ecdf.png"),
        dpi=150,
    )
    plt.close()


def plot_boxplot(
    df,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    data = []
    labels = []

    for regime, g in df.groupby("regime", observed=True):
        vals = g[metric].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(regime)

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )

    plt.ylabel(metric)
    plt.title(f"Box plot — {metric}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_boxplot.png"),
        dpi=150,
    )
    plt.close()


def plot_median_ci(
    df,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    regimes = []
    medians = []
    lowers = []
    uppers = []

    for regime, g in df.groupby("regime", observed=True):
        vals = g[metric].dropna().values
        if len(vals) < 5:
            continue
        lo, hi = bootstrap_ci(vals)
        med = np.median(vals)

        regimes.append(regime)
        medians.append(med)
        lowers.append(lo)
        uppers.append(hi)

    x = np.arange(len(regimes))
    plt.figure(figsize=(6, 4))

    # plot medians
    plt.scatter(x, medians, zorder=3)

    # plot CI as vertical segments
    for xi, lo, hi in zip(x, lowers, uppers):
        plt.vlines(xi, lo, hi, linewidth=2)

    plt.xticks(x, regimes)
    plt.ylabel(metric)
    plt.title(f"Median ± 95% CI — {metric}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_median_ci.png"),
        dpi=150,
    )
    plt.close()


def plot_pairwise_median_diff(
    df,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    regimes = sorted(df["regime"].unique())
    diffs = []

    for r1, r2 in itertools.combinations(regimes, 2):
        v1 = df[df["regime"] == r1][metric].dropna().values
        v2 = df[df["regime"] == r2][metric].dropna().values
        if len(v1) == 0 or len(v2) == 0:
            continue
        diff = np.median(v1) - np.median(v2)
        diffs.append((f"{r1} − {r2}", diff))

    labels, values = zip(*diffs)

    plt.figure(figsize=(7, 4))
    plt.barh(labels, values)
    plt.axvline(0, color="black", linestyle="--")

    plt.xlabel(f"Median difference ({metric})")
    plt.title(f"Pairwise regime differences — {metric}")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_pairwise_diff.png"),
        dpi=150,
    )
    plt.close()


def plot_age_ecdf(
    df,
    output_dir: str,
    model_tag: str,
):
    plt.figure(figsize=(6, 4))

    for regime, g in df.groupby("regime", observed=True):
        ages = np.sort(g["age_refined"].dropna().values)
        if len(ages) == 0:
            continue
        y = np.arange(1, len(ages) + 1) / len(ages)
        plt.step(ages, y, where="post", label=regime)

    plt.xlabel("Age")
    plt.ylabel("ECDF")
    plt.title("ECDF of age by regime")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_age_ecdf.png"),
        dpi=150,
    )
    plt.close()


def plot_age_violin(
    df: pd.DataFrame,
    output_dir: str,
    model_tag: str,
    regime_col: str = "regime",
    order: list | None = None,
    figsize: tuple = (8, 5),
    show_n: bool = True,
):
    if "age_refined" not in df.columns:
        raise ValueError("Column 'age' not found in dataframe.")

    if regime_col not in df.columns:
        raise ValueError(f"Regime column '{regime_col}' not found in dataframe.")

    plot_df = df[[regime_col, "age_refined"]].dropna()
    if plot_df.empty:
        raise ValueError("No age data left after dropping NaNs.")

    if order is None:
        order = sorted(plot_df[regime_col].unique())

    plt.figure(figsize=figsize)

    # --- Violin (distribution shape)
    sns.violinplot(
        data=plot_df,
        x=regime_col,
        y="age_refined",
        order=order,
        inner=None,
        cut=0,
    )

    # --- Boxplot overlay (median + IQR)
    sns.boxplot(
        data=plot_df,
        x=regime_col,
        y="age_refined",
        order=order,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 2},
        showfliers=False,
        whiskerprops={"linewidth": 1.5},
        zorder=2,
    )

    # --- Sample size annotation
    if show_n:
        counts = plot_df.groupby(regime_col)["age_refined"].count()
        ymax = plot_df["age_refined"].max()
        y_offset = 0.05 * (ymax if ymax != 0 else 1)

        for i, regime in enumerate(order):
            n = counts.get(regime, 0)
            plt.text(
                i,
                ymax + y_offset,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.xlabel("Educational regime")
    plt.ylabel("Age")
    plt.title("Age distribution by regime", pad=30)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_age_violin.png"),
        dpi=300,
    )
    plt.close()


def plot_age_box(
    df,
    output_dir: str,
    model_tag: str,
):
    data = []
    labels = []

    for regime, g in df.groupby("regime", observed=True):
        ages = g["age_refined"].dropna().values
        if len(ages) == 0:
            continue
        data.append(ages)
        labels.append(regime)

    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.ylabel("Age")
    plt.title("Age by regime (box plot)")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_age_box.png"),
        dpi=150,
    )
    plt.close()


def plot_age_median_ci(
    df,
    output_dir: str,
    model_tag: str,
):
    regimes = []
    medians = []
    lowers = []
    uppers = []

    for regime, g in df.groupby("regime", observed=True):
        ages = g["age_refined"].dropna().values
        if len(ages) < 5:
            continue

        lo, hi = bootstrap_ci(ages)
        med = np.median(ages)

        regimes.append(regime)
        medians.append(med)
        lowers.append(lo)
        uppers.append(hi)

    x = np.arange(len(regimes))
    plt.figure(figsize=(6, 4))

    plt.scatter(x, medians, zorder=3)
    for xi, lo, hi in zip(x, lowers, uppers):
        plt.vlines(xi, lo, hi, linewidth=2)

    plt.xticks(x, regimes)
    plt.ylabel("Age")
    plt.title("Median age ± 95% CI by regime")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_age_median_ci.png"),
        dpi=150,
    )
    plt.close()


# Bar plot of discrete age counts per regime.
def plot_discrete_age_frequencies(
    df: pd.DataFrame,
    output_dir: str,
    model_tag: str,
):
    if "regime" not in df.columns or "age_refined" not in df.columns:
        raise ValueError("df must contain 'regime' and 'age_refined'")

    plot_df = df[["regime", "age_refined"]].dropna()

    regimes = sorted(plot_df["regime"].unique())
    ages = sorted(plot_df["age_refined"].unique())

    plt.figure(figsize=(10, 5))

    width = 0.8 / len(regimes)
    x = np.arange(len(ages))

    for i, regime in enumerate(regimes):
        counts = (
            plot_df[plot_df["regime"] == regime]["age_refined"]
            .value_counts()
            .reindex(ages, fill_value=0)
        )

        plt.bar(
            x + i * width,
            counts.values,
            width=width,
            label=regime,
        )

    plt.xticks(
        x + width * (len(regimes) - 1) / 2,
        [int(a) for a in ages],
    )

    plt.xlabel("Age")
    plt.ylabel("Number of personas")
    plt.title("Discrete age frequencies by regime")
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{model_tag}_age_frequency.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200)
    plt.close()


# heatmap regime vs age
def plot_regime_age_heatmap(
    df: pd.DataFrame,
    output_dir: str,
    model_tag: str,
    normalize: bool = True,  # True = proportions, False = raw counts
):
    if "regime" not in df.columns or "age_refined" not in df.columns:
        raise ValueError("df must contain 'regime' and 'age_refined'")

    plot_df = df[["regime", "age_refined"]].dropna()

    # build contingency table
    table = pd.crosstab(
        plot_df["regime"],
        plot_df["age_refined"],
        normalize="index" if normalize else False,
    )

    plt.figure(figsize=(10, 4))
    plt.imshow(table.values, aspect="auto")

    plt.colorbar(
        label="Proportion" if normalize else "Count",
        fraction=0.03,
        pad=0.04,
    )

    plt.xticks(
        ticks=np.arange(len(table.columns)),
        labels=table.columns.astype(int),
    )
    plt.yticks(
        ticks=np.arange(len(table.index)),
        labels=table.index,
    )

    plt.xlabel("Age")
    plt.ylabel("Educational regime")
    plt.title("Regime × Age distribution")

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{model_tag}_regime_age_heatmap.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=200)
    plt.close()


def plot_boxplot_by_age(
    df: pd.DataFrame,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    if "age_refined" not in df.columns:
        raise ValueError("Column 'age_refined' not found")

    plot_df = df[[metric, "age_refined"]].dropna()

    if plot_df.empty:
        raise ValueError("No data left after dropping NaNs")
        
    ages = sorted(plot_df["age_refined"].unique())

    data = []
    labels = []

    for age in ages:
        vals = plot_df.loc[plot_df["age_refined"] == age, metric].values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(int(age))

    plt.figure(figsize=(8, 4))
    plt.boxplot(
        data,
        labels=labels,
        showfliers=True,
        patch_artist=True,
    )

    plt.xlabel("Age")
    plt.ylabel(metric)
    plt.title(f"{metric} by age — {model_tag}")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_by_age_boxplot.png"),
        dpi=150,
    )
    plt.close()


def plot_violin_box_by_age(
    df: pd.DataFrame,
    metric: str,
    model_tag: str,
    output_dir: str,
    figsize: tuple = (9, 5),
    show_n: bool = True,
):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    if "age_refined" not in df.columns:
        raise ValueError("Column 'age_refined' not found")

    y_vals, y_label = maybe_log_transform(df, metric)

    plot_df = pd.DataFrame(
        {
            metric: y_vals,
            "age_refined": df["age_refined"],
        }
    ).dropna()

    if plot_df.empty:
        raise ValueError("No data left after dropping NaNs")

    ages = sorted(plot_df["age_refined"].unique())

    plt.figure(figsize=figsize)

    sns.violinplot(
        data=plot_df,
        x="age_refined",
        y=metric,
        order=ages,
        inner=None,
        cut=0,
    )

    sns.boxplot(
        data=plot_df,
        x="age_refined",
        y=metric,
        order=ages,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 2},
        showfliers=False,
        whiskerprops={"linewidth": 1.5},
        zorder=2,
    )

    if show_n:
        counts = plot_df.groupby("age_refined", observed=True)[metric].count()
        ymax = plot_df[metric].max()
        y_offset = 0.05 * (ymax if ymax != 0 else 1)

        for i, age in enumerate(ages):
            n = counts.get(age, 0)
            plt.text(
                i,
                ymax + y_offset,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xlabel("Age")
    plt.ylabel(metric)
    plt.title(f"{metric} by age — {model_tag}", pad=20)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"violin_{metric}_{model_tag}_by_age.png"),
        dpi=300,
    )
    plt.close()


def plot_boxplot_by_age_bin(
    df: pd.DataFrame,
    metric: str,
    output_dir: str,
    model_tag: str,
):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    if "age_bin" not in df.columns:
        raise ValueError("Column 'age_bin' not found")

    plot_df = df[[metric, "age_bin"]].dropna()
    if plot_df.empty:
        raise ValueError("No data left after dropping NaNs")

    order = list(plot_df["age_bin"].unique())

    data = [plot_df.loc[plot_df["age_bin"] == b, metric].values for b in order]

    plt.figure(figsize=(9, 4))
    plt.boxplot(
        data,
        labels=order,
        showfliers=True,
        patch_artist=True,
    )

    plt.xlabel("Age bin (years)")
    plt.ylabel(metric)
    plt.title(f"{metric} by age bin — {model_tag}")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_by_age_bin_boxplot.png"),
        dpi=150,
    )
    plt.close()


def plot_violin_box_by_age_bin(
    df: pd.DataFrame,
    metric: str,
    model_tag: str,
    output_dir: str,
    figsize=(10, 5),
    show_n=True,
):
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found")

    if "age_bin" not in df.columns:
        raise ValueError("Column 'age_bin' not found")

    plot_df = df[[metric, "age_bin"]].dropna()
    if plot_df.empty:
        raise ValueError("No data left after dropping NaNs")

    # preserve bin order as defined
    order = list(plot_df["age_bin"].unique())

    plt.figure(figsize=figsize)

    sns.violinplot(
        data=plot_df,
        x="age_bin",
        y=metric,
        order=order,
        inner=None,
        cut=0,
    )

    sns.boxplot(
        data=plot_df,
        x="age_bin",
        y=metric,
        order=order,
        width=0.25,
        showcaps=True,
        boxprops={"facecolor": "none"},
        showfliers=False,
    )

    if show_n:
        counts = plot_df.groupby("age_bin", observed=True)[metric].count()
        ymax = plot_df[metric].max()
        offset = 0.05 * (ymax if ymax != 0 else 1)

        for i, b in enumerate(order):
            plt.text(
                i,
                ymax + offset,
                f"n={counts.get(b, 0)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xlabel("Age bin (years)")
    plt.ylabel(metric)
    plt.title(f"{metric} by age bin — {model_tag}", pad=20)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{model_tag}_{metric}_by_age_bin_violin.png"),
        dpi=300,
    )
    plt.close()
