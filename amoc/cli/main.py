import os
import sys
import time
import logging
import argparse
import re
from typing import List

# --- Multiprocessing safety (vLLM + CUDA) ---
import multiprocessing

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
from amoc.analysis.statistics import run_statistical_analysis
from amoc.nlp import load_spacy
from amoc.nlp.highlights import blue_nodes_from_text


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
        "--strict-attachament-constraint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled new edges must touch the current sentence AND "
            "the active neighborhood and anchor"
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
        required=True,
        help="Path to a single persona CSV chunk file to process",
    )

    p.add_argument(
        "--story-text",
        type=str,
        default=None,
        help="Override the default AMoC story text; defaults to configured knight STORY_TEXT ",
    )

    p.add_argument(
        "--allow-multi-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("Allow multiple edges between the same ordered node pair"),
    )

    p.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable checkpoint mode for edges",
    )

    return p.parse_args(argv)


def is_leader() -> bool:
    return os.environ.get("SLURM_ARRAY_TASK_ID") in (None, "0")


def load_story_text_from_arg(story_text_arg: str) -> str:
    if story_text_arg is None:
        return None

    if os.path.isfile(story_text_arg):
        if not story_text_arg.lower().endswith(".txt"):
            raise ValueError(f"Story text file must be a .txt file: {story_text_arg}")

        with open(story_text_arg, "r", encoding="utf-8") as f:
            text = f.read()

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    return story_text_arg.strip()


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--models must contain at least one model")

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
                graphs_output_dir=os.path.join(OUTPUT_ANALYSIS_DIR, "graphs"),
                highlight_nodes=highlight_nodes,
                plot_final_graph=args.plot_final_graph,
                plot_largest_component_only=args.plot_largest_component_only,
                include_inactive_edges=args.include_inactive_edges,
                strict_reactivate_function=args.strict_reactivate_function,
                strict_attachament_constraint=args.strict_attachament_constraint,
                single_anchor_hub=args.single_anchor_hub,
                edge_visibility=args.edge_visibility,
                story_text=story_text,
                force_node=True,
                allow_multi_edges=args.allow_multi_edges,
                checkpoint=args.checkpoint,
            )
    finally:
        elapsed = time.time() - total_start
        print(f"\nExtraction phase finished in {elapsed:.2f} seconds")

        if is_leader():
            for model in model_names:
                try:
                    run_statistical_analysis(model)
                except Exception as e:
                    logging.error(
                        f"Statistical analysis failed for {model}: {e}",
                        exc_info=True,
                    )
                    print(f"Statistics failed for {model}")


if __name__ == "__main__":
    main(sys.argv[1:])
