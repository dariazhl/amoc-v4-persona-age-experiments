import os
import sys
import time
import glob as glob_module
import logging
import argparse
import re
from typing import List
import multiprocessing

if __name__ == "__main__" and os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-m", "amoc.cli.main", *sys.argv[1:]])

multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

from amoc.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_ANALYSIS_DIR,
    BLUE_NODES,
    STORY_TEXT,
)
from amoc.pipeline.runner import process_persona_csv
from amoc.llm.vllm_client import ServerUnavailableError
from amoc.config.constants import PERSONA_MODES, allow_inferred_chains, allow_persona_words, persona_mode
from amoc.analysis.statistics import run_statistical_analysis, canonicalize_model_name
from amoc.utils.spacy_utils import load_spacy
from amoc.utils.highlights import blue_nodes_from_text
from amoc.outliers.io import save_persona_outputs
from amoc.outliers.stats import build_persona_stats
from amoc.outliers.trimming import quantile_trim, iqr_cap
from amoc.outliers.triplets import filter_triplets_by_persona
from amoc.outliers.cleaned_regime_analysis import run_cleaned_regime_analysis

LOWER_Q = 0.05
UPPER_Q = 0.95
MIN_PERSONAS_FOR_TRIMMING = 14
TRIM_METRICS = [
    "num_triplets",
    "num_unique_concepts",
    "triplets_per_100_tokens",
    "graph_num_nodes",
    "graph_num_edges",
]
CAP_METRICS = ["graph_density", "graph_avg_degree"]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run AMoCv4 over persona CSVs using age-aware and persona-aware prompts"
        )
    )

    p.add_argument(
        "--models",
        required=True,
        help=(
            "Comma-separated list of vLLM model names "
            "(e.g. 'Qwen/Qwen3-30B-A3B-Instruct-2507,openai/gpt-oss-120b')"
        ),
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit on rows per CSV",
    )

    p.add_argument(
        "--no-replace-pronouns",
        dest="replace_pronouns",
        action="store_false",
        help="Disable pronoun resolution",
    )
    p.set_defaults(replace_pronouns=True)

    p.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp_size",
        help="Tensor parallel size for vLLM",
    )

    p.add_argument(
        "--server-url",
        default=None,
        help=(
            "URL of a running `vllm serve` instance (e.g. http://127.0.0.1:8000). "
            "If set, the model is not loaded in-process"
        ),
    )

    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of personas processed concurrently (requires --server-url). "
            "vLLM batches their requests, so throughput scales with this"
        ),
    )

    p.add_argument(
        "--resume-only",
        action="store_true",
        help="Only process personas not yet completed",
    )

    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for extracted triplets",
    )

    p.add_argument(
        "--plot-after-each-sentence",
        action="store_true",
        help="Plot a graph after each sentence for a specific persona",
    )

    p.add_argument(
        "--no-reverse-plots",
        action="store_true",
        help="Skip the per-persona reverse plots",
    )

    p.add_argument(
        "--plot-final-graph",
        action="store_true",
        help="Plot a single final graph per persona",
    )

    p.add_argument(
        "--plot-largest-component-only",
        action="store_true",
        dest="plot_largest_component_only",
        help="Keep only the largest connected component when plotting",
    )
    p.add_argument(
        "--plot-all-components",
        action="store_false",
        dest="plot_largest_component_only",
        help="Plot all connected components",
    )
    p.set_defaults(plot_largest_component_only=False)

    p.add_argument(
        "--strict-reactivate-function",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the stricter reactivation logic (default). Disable to use the legacy "
            "reactivation behavior from the original paper code."
        ),
    )

    p.add_argument(
        "--single-anchor-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Keep a single anchor hub that every edge must touch "),
    )

    p.add_argument(
        "--edge-visibility",
        type=int,
        default=None,
        help="Override edge visibility score (default uses value from amoc.config.constants",
    )

    p.add_argument(
        "--include-inactive-edges",
        action="store_true",
        help="Include inactive edges in CSV export ",
    )

    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single persona CSV chunk file to process (required without --post-process)",
    )

    p.add_argument(
        "--story-text",
        type=str,
        default=None,
        help="Override the default AMoC story text; defaults to configured knight STORY_TEXT ",
    )

    p.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable checkpoint mode for edges",
    )

    p.add_argument(
        "--post-process",
        action="store_true",
        help=(
            "After extraction, wait for all array tasks to finish then run "
            "outlier removal and plots across all run_* output folders"
        ),
    )

    p.add_argument(
        "--plots",
        action="store_true",
        help="Generate statistical (violin, etc.) plots per regime after outlier removal",
    )

    p.add_argument(
        "--plots-age",
        action="store_true",
        help="Generate plots per regime after outlier removal for age",
    )

    p.add_argument(
        "--persona-mode",
        choices=PERSONA_MODES,
        default=None,
        help=(
            "legacy: prompts as in the published runs (persona 'for focus only'); "
            "reader: the graph is built as the persona would understand the text. "
            "Defaults to $AMOC_PERSONA_MODE, else reader."
        ),
    )

    p.add_argument(
        "--allow-inferred-chains",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Admit inferred-to-inferred edges (validator rule 6 and edge admission). "
            "Defaults to $AMOC_ALLOW_INFERRED_CHAINS, else off."
        ),
    )

    return p.parse_args(argv)


def is_leader() -> bool:
    return os.environ.get("SLURM_ARRAY_TASK_ID") in (None, "0")


def run_post_processing(model_name: str, base_dir: str, plots: bool = False, plots_age: bool = False) -> None:
    csv_pattern = os.path.join(
        base_dir, "triplets", "triplets_final_state", "*_final_triplets_*.csv"
    )

    triplet_files = glob_module.glob(csv_pattern)
    if not triplet_files:
        raise RuntimeError(f"[post-process] No final triplet CSVs found under {base_dir}")

    print(f"\n[post-process] Aggregating {len(triplet_files)} CSVs")

    model_tag = canonicalize_model_name(model_name).replace("/", "-")
    statistics_dir = os.path.join(base_dir, "statistics")
    os.makedirs(statistics_dir, exist_ok=True)

    df_stats = build_persona_stats(triplet_files)
    if "original_index" in df_stats.columns:
        df_stats = df_stats.rename(columns={"original_index": "idx"})

    print(df_stats.groupby("regime").size())

    df_stats.to_csv(os.path.join(statistics_dir, "persona_stats_raw.csv"), index=False)

    if len(df_stats) < MIN_PERSONAS_FOR_TRIMMING:
        print(f"[post-process] Only {len(df_stats)} personas — skipping trimming (need >= {MIN_PERSONAS_FOR_TRIMMING})")
        df_trimmed = df_stats.copy()
        df_clean = df_stats.copy()
    else:
        df_trimmed = quantile_trim(df_stats, TRIM_METRICS, LOWER_Q, UPPER_Q)
        df_clean = iqr_cap(df_trimmed, CAP_METRICS)

    kept_personas = set(df_trimmed["idx"])
    removed_personas = set(df_stats["idx"]) - kept_personas

    save_persona_outputs(
        df_trimmed,
        df_clean,
        removed_personas,
        model_tag,
        out_dir=os.path.join(statistics_dir, "persona_stats"),
    )

    filter_triplets_by_persona(
        triplet_files,
        kept_personas,
        model_tag,
        out_dir=os.path.join(statistics_dir, "triplets_processed"),
    )

    print(f"[post-process] Kept {len(df_trimmed)} of {len(df_stats)} personas")

    run_cleaned_regime_analysis(
        input_dir=statistics_dir,
        model=model_tag,
        output_dir=statistics_dir,
        plots=plots,
        plots_age=plots_age,
    )


def load_story_text_from_arg(story_text_arg: str) -> str:
    if story_text_arg is None:
        return None

    if os.path.isfile(story_text_arg):
        if not story_text_arg.lower().endswith(".txt"):
            raise ValueError(f"Story text file must be a .txt file: {story_text_arg}")

        with open(story_text_arg, "r", encoding="utf-8") as f:
            text = f.read()

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    return story_text_arg.strip()


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--models must contain at least one model")

    if not args.file:
        raise SystemExit("--file is required for extraction")

    if args.workers > 1:
        if not args.server_url:
            raise SystemExit("--workers > 1 requires --server-url")
        if len(model_names) != 1:
            raise SystemExit("--workers > 1 supports a single model (the one the server serves)")

    if args.post_process and not args.output_dir:
        raise SystemExit("--output-dir is required when --post-process is set")

    if args.persona_mode is not None:
        os.environ["AMOC_PERSONA_MODE"] = args.persona_mode
    if args.allow_inferred_chains is not None:
        os.environ["AMOC_ALLOW_INFERRED_CHAINS"] = "1" if args.allow_inferred_chains else "0"
    print(f"Persona mode: {persona_mode()}  allow inferred chains: {allow_inferred_chains()}")

    spacy_nlp = load_spacy()

    files_to_process = [args.file]

    if not files_to_process:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    print(f"Discovered {len(files_to_process)} persona CSV files")
    print(f"Models: {model_names}")
    print(f"Output directory: {OUTPUT_DIR}")

    total_start = time.time()

    if args.story_text is not None:
        story_text = load_story_text_from_arg(args.story_text)
    else:
        story_text = STORY_TEXT

    story_is_default = (story_text or "").strip() == (STORY_TEXT or "").strip()
    highlight_nodes = (
        BLUE_NODES if story_is_default else blue_nodes_from_text(story_text, spacy_nlp)
    )

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = args.output_dir
        print(f"Overriding output directory to: {output_dir}")
    else:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "persona_mode.txt"), "w") as f:
        f.write(f"persona_mode={persona_mode()}\nallow_inferred_chains={int(allow_inferred_chains())}\nallow_persona_words={int(allow_persona_words())}\n")

    if args.output_dir:
        graphs_output_dir = os.path.join(args.output_dir, "graphs")
    else:
        graphs_output_dir = os.path.join(OUTPUT_ANALYSIS_DIR, "graphs")

    server_failed = False
    try:
        for filename in files_to_process:
            print(f"\n=== Processing file: {os.path.basename(filename)} ===")
            process_persona_csv(
                filename=filename,
                model_names=model_names,
                spacy_nlp=spacy_nlp,
                output_dir=output_dir,
                max_rows=args.max_rows,
                replace_pronouns=args.replace_pronouns,
                tensor_parallel_size=args.tp_size,
                resume_only=args.resume_only,
                plot_after_each_sentence=args.plot_after_each_sentence,
                graphs_output_dir=graphs_output_dir,
                highlight_nodes=highlight_nodes,
                plot_final_graph=args.plot_final_graph,
                plot_largest_component_only=args.plot_largest_component_only,
                include_inactive_edges=args.include_inactive_edges,
                strict_reactivate_function=args.strict_reactivate_function,
                single_anchor_hub=args.single_anchor_hub,
                edge_visibility=args.edge_visibility,
                story_text=story_text,
                force_node=True,
                checkpoint=args.checkpoint,
                generate_reverse_plots=not args.no_reverse_plots,
                reverse_plot_mode="both",
                server_url=args.server_url,
                workers=args.workers,
            )
    except ServerUnavailableError:
        server_failed = True
        print("\nERROR: vLLM server failed; run aborted, skipping post-processing")
        raise
    finally:
        elapsed = time.time() - total_start
        print(f"\nExtraction phase finished in {elapsed:.2f} seconds")

        if args.post_process and not server_failed:
            sentinel_dir = os.path.join(output_dir, ".sentinels")
            os.makedirs(sentinel_dir, exist_ok=True)
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
            open(os.path.join(sentinel_dir, f"done_{task_id}"), "w").close()

            total = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
            done = len(glob_module.glob(os.path.join(sentinel_dir, "done_*")))
            print(f"[post-process] {done}/{total} tasks finished")

            if done >= total:
                for model in model_names:
                    try:
                        run_statistical_analysis(
                            model,
                            output_dir=output_dir,
                            analysis_dir=os.path.join(output_dir, "statistics"),
                        )
                    except Exception as e:
                        logging.error(f"Statistical analysis failed for {model}: {e}", exc_info=True)
                        print(f"Statistics failed for {model}")

                try:
                    run_post_processing(
                        model_names[0],
                        output_dir,
                        plots=args.plots,
                        plots_age=args.plots_age,
                    )
                except Exception as e:
                    logging.error(f"Post-processing failed: {e}", exc_info=True)
                    print(f"Post-processing failed: {e}")
        else:
            if is_leader():
                for model in model_names:
                    try:
                        run_statistical_analysis(model, output_dir=output_dir)
                    except Exception as e:
                        logging.error(
                            f"Statistical analysis failed for {model}: {e}",
                            exc_info=True,
                        )
                        print(f"Statistics failed for {model}")


if __name__ == "__main__":
    main(sys.argv[1:])
