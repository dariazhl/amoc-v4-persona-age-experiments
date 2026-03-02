import os
import glob
import pandas as pd
from scipy.stats import kruskal

from amoc.metrics.aggregation import process_triplets_file
from amoc.config.paths import OUTPUT_DIR, OUTPUT_ANALYSIS_DIR
from amoc.analysis.regime_plots import plot_violin_box


METRICS_TO_PLOT = [
    "num_triplets",
    "num_unique_concepts",
    "graph_density",
    "triplets_per_100_tokens",
    "graph_avg_degree",
]


def canonicalize_model_name(name: str) -> str:
    name = name.lower().strip()

    if "gemma3" in name:
        name = name.replace("gemma3", "gemma-3")
    elif "phi4" in name:
        name = name.replace("phi4", "phi-4")
    elif "llama3.3" in name:
        name = name.replace("llama3.3", "Llama-3.3")
    elif "qwen3" in name:
        name = name.replace("qwen3", "Qwen3")

    return name


def run_statistical_analysis(model_name: str):
    print("\n" + "=" * 60)
    print(f"AMoC REGIME-BASED ANALYSIS — MODEL: {model_name}")
    print("=" * 60)

    model_name = canonicalize_model_name(model_name)
    safe_tag = model_name.replace("/", "-").replace(":", "-").replace(" ", "_")
    # Apply Qwen-specific capitalization rules
    safe_tag = safe_tag.lower()

    if "qwen" in safe_tag:
        parts = safe_tag.split("-")
        new_parts = []
        for p in parts:
            if p.startswith("qwen"):
                # qwen -> Qwen, qwen3 -> Qwen3
                new_parts.append("Qwen" + p[4:])
            elif p.endswith("b") and any(c.isdigit() for c in p):
                # 30b -> 30B, a3b -> A3B
                new_parts.append(p.upper())
            elif p.isalpha():
                # instruct -> Instruct
                new_parts.append(p.capitalize())
            else:
                # numbers etc. unchanged
                new_parts.append(p)

        safe_tag = "-".join(new_parts)

    is_llama = model_name.lower().startswith("meta-llama")

    pattern = f"model_{safe_tag}_triplets_*.csv"
    search_path = os.path.join(OUTPUT_DIR, pattern)

    print(f"Looking for CSV files with pattern: {search_path}")
    files_to_analyze = glob.glob(search_path)

    if not files_to_analyze and is_llama:
        print("No files found with glob")

        candidates = [
            os.path.join(OUTPUT_DIR, f)
            for f in os.listdir(OUTPUT_DIR)
            if f.lower().startswith("model_")
            and "triplets_" in f.lower()
            and f.lower().endswith(".csv")
        ]

        safe_tag_lc = safe_tag.lower()

        files_to_analyze = [
            f for f in candidates if safe_tag_lc in os.path.basename(f).lower()
        ]

    if not files_to_analyze:
        print(f"No triplet CSVs found for model {model_name}")
        return

    all_metrics = []
    for path in files_to_analyze:
        df = process_triplets_file(path)
        if df is not None and not df.empty:
            df["model"] = model_name
            all_metrics.append(df)

    if not all_metrics:
        print("No usable metric data.")
        return

    df_master = pd.concat(all_metrics, ignore_index=True)

    if "regime" not in df_master.columns:
        raise RuntimeError("Missing 'regime' column ")

    print(f"Total personas analyzed: {len(df_master)}")
    print("Regime counts:")
    print(df_master["regime"].value_counts())

    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

    model_tag = safe_tag
    stats_records = []

    for metric in METRICS_TO_PLOT:
        if metric not in df_master.columns:
            continue

        print(f"\n[Plotting] {metric} by regime")

        plot_violin_box(
            df=df_master,
            metric=metric,
            output_dir=OUTPUT_ANALYSIS_DIR,
            model_tag=model_tag,
        )

        groups = [
            g[metric].dropna().values
            for _, g in df_master.groupby("regime")
            if g[metric].notna().any()
        ]

        if len(groups) >= 2:
            h, p = kruskal(*groups)
            stats_records.append(
                {
                    "metric": metric,
                    "test": "kruskal",
                    "H": h,
                    "p_value": p,
                }
            )
            print(f"  Kruskal–Wallis H={h:.3f}, p={p:.4g}")

    if stats_records:
        stats_df = pd.DataFrame(stats_records)
        stats_path = os.path.join(OUTPUT_ANALYSIS_DIR, f"{model_tag}_regime_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"\nSaved statistical test results to {stats_path}")
