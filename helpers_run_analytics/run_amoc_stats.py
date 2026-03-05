# Run AMoC age-related statistical analysis for a single model.
# This is a thin wrapper around `run_statistical_analysis(model_tag)` defined in `vllm_train_minimal_v4.py`.
# Usage example (Qwen):
#     python run_amoc_stats.py \
#         --model "Qwen/Qwen3-30B-A3B-Instruct-2507"

import argparse
import sys

# Import your existing analysis function from the big script
from vllm_train_minimal_v4 import run_statistical_analysis


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run AMoC statistical analysis for a given model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model identifier, e.g. "
            '"Qwen/Qwen3-30B-A3B-Instruct-2507" or "openai/gpt-oss-120b". '
            "Must match the string used in --models when generating triplets."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    model_tag = args.model
    print(f"\n=== Running AMoC statistical analysis for model: {model_tag} ===\n")

    # This function already:
    #  - builds the safe model tag
    #  - globs only CSV files for this model
    #  - computes correlations
    #  - saves plots and master table in OUTPUT_ANALYSIS_DIR
    run_statistical_analysis(model_tag)


if __name__ == "__main__":
    main(sys.argv[1:])
