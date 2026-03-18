from amoc.analysis.statistics import run_statistical_analysis
from amoc.analysis.age_regimes import assign_age_bin
from amoc.analysis.plotting import maybe_log_transform
from amoc.analysis.regime_plots import (
    plot_violin_box,
    plot_ecdf,
    plot_boxplot,
    plot_median_ci,
    plot_pairwise_median_diff,
    plot_age_box,
    plot_age_ecdf,
    plot_age_median_ci,
    plot_age_violin,
    plot_regime_age_heatmap,
    plot_discrete_age_frequencies,
    plot_violin_box_by_age,
    plot_boxplot_by_age,
    plot_boxplot_by_age_bin,
    plot_violin_box_by_age_bin,
)

__all__ = [
    "run_statistical_analysis",
    "plot_violin_box",
    "plot_ecdf",
    "plot_boxplot",
    "plot_median_ci",
    "plot_pairwise_median_diff",
    "plot_age_box",
    "plot_age_ecdf",
    "plot_age_median_ci",
    "plot_age_violin",
    "plot_regime_age_heatmap",
    "plot_discrete_age_frequencies",
    "maybe_log_transform",
    "plot_violin_box_by_age",
    "plot_boxplot_by_age",
    "plot_boxplot_by_age_bin",
    "plot_violin_box_by_age_bin",
    "assign_age_bin",
    "coarse_age_bin",
]
