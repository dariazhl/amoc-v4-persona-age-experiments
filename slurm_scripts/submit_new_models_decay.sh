#!/bin/bash
# Submit the persona-modularity run for the three new models, each under BOTH
# decay rules:
#   score decay -> LLM_SCORE=1  (--llm-score: the model's 0-5 rating is the activation)
#   llm decay   -> LLM_DECAY=1  (--llm-decay: age from NARRATIVE_RELEVANCE score 2)
#
# Per model, the llm-decay job waits for the score job (afterok) and reuses
# its story_analysis.json, so the two runs differ by the decay rule only and
# not by extraction too (see run_llama70b_persona_modularity.sh).
#
# Run on the cluster login node, NOT with sbatch:
#   bash slurm_scripts/submit_new_models_decay.sh
#   LIMIT=1 TAG=smoke bash slurm_scripts/submit_new_models_decay.sh   # smoke test
#   MODELS="qwen38_27b" bash slurm_scripts/submit_new_models_decay.sh # one model
set -euo pipefail

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODELS="${MODELS:-llama4scout gemma4_31b qwen38_27b}"

declare -A HF_ID=(
    [llama4scout]="meta-llama/Llama-4-Scout-17B-16E-Instruct"
    [gemma4_31b]="google/gemma-4-31B-it"
    [qwen38_27b]="Qwen/Qwen3.8-27B"
)

for m in ${MODELS}; do
    [[ -n "${HF_ID[$m]:-}" ]] || { echo "ERROR: unknown model key '$m'"; exit 1; }
    script="${PROJECT_ROOT}/slurm_scripts/run_${m}_persona_modularity.sh"
    [[ -f "${script}" ]] || { echo "ERROR: missing ${script}"; exit 1; }

    # Same slug rule as amoc-min-persona-modularity-body.sh
    slug="$(basename "${HF_ID[$m]}" | tr '[:upper:]' '[:lower:]')"
    base="${OUTPUT_ROOT}/amoc_min/${slug}"
    score_dir="${base}/${TAG}_llm_score"
    decay_dir="${base}/${TAG}_llm_decay"

    score_id=$(RUN_OUTPUT_DIR="${score_dir}" LLM_SCORE=1 \
        sbatch --parsable --job-name="pm_${m}_score" "${script}")

    decay_id=$(RUN_OUTPUT_DIR="${decay_dir}" LLM_DECAY=1 \
        STORY_ANALYSIS="${score_dir}/story_analysis.json" \
        sbatch --parsable --dependency=afterok:"${score_id}" \
               --job-name="pm_${m}_decay" "${script}")

    echo "${m}: score=${score_id} -> ${score_dir}"
    echo "${m}: decay=${decay_id} (after ${score_id}) -> ${decay_dir}"
done
