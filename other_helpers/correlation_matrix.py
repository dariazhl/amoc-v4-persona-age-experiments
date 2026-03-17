#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------
# Input
# ------------------------------------------------------------------

INPUT_CSV = Path(
    "/Users/dariazahaleanu/Documents/Coding_Projects/"
    "amoc-v4-persona-age-experiments/results/Qwen3-30b/"
    "final_plots/jan_20/statistics/_persona_stats_quantile_trimmed.csv"
)

COLS = [
    "age_refined",
    "graph_avg_degree",
    "graph_num_edges",
    "triplets_per_100_tokens",
    "lexical_avg_word_len",
]


df = pd.read_csv(INPUT_CSV)

df = df[COLS].apply(pd.to_numeric, errors="coerce")
df = df.dropna()


def corr_pval_matrix(df, method="pearson"):
    corr = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    pval = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)

    for i in df.columns:
        for j in df.columns:
            if method == "pearson":
                r, p = pearsonr(df[i], df[j])
            elif method == "spearman":
                r, p = spearmanr(df[i], df[j])
            else:
                raise ValueError("Unknown method")

            corr.loc[i, j] = r
            pval.loc[i, j] = p

    return corr, pval


pearson_corr, pearson_p = corr_pval_matrix(df, method="pearson")
spearman_corr, spearman_p = corr_pval_matrix(df, method="spearman")

print("\n=== Pearson correlation ===")
print(pearson_corr.round(3))
print("\n=== Pearson p-values ===")
print(pearson_p.round(4))

print("\n=== Spearman correlation ===")
print(spearman_corr.round(3))
print("\n=== Spearman p-values ===")
print(spearman_p.round(4))


INPUT_CSV = Path(
    "/Users/dariazahaleanu/Documents/Coding_Projects/"
    "amoc-v4-persona-age-experiments/results/Qwen3-30b/"
    "final_plots/jan_20/statistics/_persona_stats_quantile_trimmed.csv"
)

VARS = [
    "age_refined",
    "graph_avg_degree",
    "graph_num_edges",
    "triplets_per_100_tokens",
    "lexical_avg_word_len",
]

POSSIBLE_REGIME_COLS = ["regime", "education_regime", "regime_bin"]


df = pd.read_csv(INPUT_CSV)


def find_regime_col(df):
    for col in POSSIBLE_REGIME_COLS:
        if col in df.columns:
            return col
    raise ValueError("No regime column found in input CSV")


GROUP_COL = find_regime_col(df)


df = df[[GROUP_COL] + VARS]
df[VARS] = df[VARS].apply(pd.to_numeric, errors="coerce")
df = df.dropna()


residuals = pd.DataFrame(index=df.index)

for var in VARS:
    model = smf.ols(f"{var} ~ C({GROUP_COL})", data=df).fit()
    residuals[var] = model.resid


corr = pd.DataFrame(index=VARS, columns=VARS, dtype=float)
pval = pd.DataFrame(index=VARS, columns=VARS, dtype=float)

for v1 in VARS:
    for v2 in VARS:
        r, p = pearsonr(residuals[v1], residuals[v2])
        corr.loc[v1, v2] = r
        pval.loc[v1, v2] = p

print("\n=== Regime-controlled Pearson correlations ===")
print(corr.round(3))

print("\n=== Regime-controlled Pearson p-values ===")
print(pval.round(4))

##heatmap
OUTPUT_FIG = Path(
    "/Users/dariazahaleanu/Documents/Coding_Projects/"
    "amoc-v4-persona-age-experiments/results/Qwen3-30b/"
    "final_plots/jan_20/statistics/correlation_heatmap.png"
)
df = pd.read_csv(INPUT_CSV)[COLS]
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# ------------------------------------------------------------------
# Compute correlation and p-values
# ------------------------------------------------------------------

corr = df.corr(method="pearson")

pvals = pd.DataFrame(index=COLS, columns=COLS)
for i in COLS:
    for j in COLS:
        _, p = pearsonr(df[i], df[j])
        pvals.loc[i, j] = p

# Mask non-significant correlations
mask = pvals.astype(float) >= 0.05

plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    mask=mask,
    linewidths=0.5,
    cbar_kws={"label": "Pearson r"},
)

plt.title(
    "Correlation between age, lexical, and graph metrics\n(non-significant cells masked)"
)
plt.tight_layout()
plt.savefig(OUTPUT_FIG, dpi=300)
plt.close()

print(f"Saved heatmap to {OUTPUT_FIG}")

# heatmap by regime
INPUT_CSV = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/final_plots/jan_20/statistics/_persona_stats_quantile_trimmed.csv"

METRICS = [
    "graph_avg_degree",
    "graph_num_edges",
    "graph_num_nodes",
    "triplets_per_100_tokens",
    "lexical_avg_word_len",
]

df = pd.read_csv(INPUT_CSV)

df = df[["regime"] + METRICS].dropna()

# Regime-level median (robust)
regime_median = df.groupby("regime")[METRICS].median()
scaler = StandardScaler()
regime_z = pd.DataFrame(
    scaler.fit_transform(regime_median),
    index=regime_median.index,
    columns=regime_median.columns,
)

plt.figure(figsize=(10, 5))

sns.heatmap(
    regime_z,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    linewidths=0.5,
)

plt.title(
    "Educational Regime Effects Across Structural and Lexical Metrics\n(Z-scored regime medians)"
)
plt.xlabel("Metric")
plt.ylabel("Educational Regime")
plt.tight_layout()
plt.show()
