#!/bin/bash
# Submit the persona-modularity run for the new models: ONE job per model,
# which runs BOTH decay rules back to back against one vLLM server. The
# default pair is the one behind mdpi_draft1/models/ (checked in each run's
# summary.json "relation_ageing"):
#   text      -> mdpi *_score  (default rule: age from the relation's own word)
#   llm_score -> mdpi *_llm    (--llm-score: the model's 0-5 rating is the activation)
# llm_decay (--llm-decay, NARRATIVE_RELEVANCE) was never used for the paper;
# add it with DECAY_RULES="text llm_score llm_decay" if wanted.
#
# The first rule writes story_analysis.json and the second reuses it, so the
# two runs differ by the decay rule only (not by extraction too) -- with no
# second job, no --dependency and no second model load.
# See DECAY_RULES in amoc-min-persona-modularity-body.sh.
#
# Run on the cluster login node, NOT with sbatch:
#   bash slurm_scripts/submit_new_models_decay.sh
#   LIMIT=1 TAG=smoke bash slurm_scripts/submit_new_models_decay.sh     # smoke test
#   MODELS="qwen38_27b" bash slurm_scripts/submit_new_models_decay.sh   # one model
#   DECAY_RULES="llm_score" bash slurm_scripts/submit_new_models_decay.sh  # one rule
set -euo pipefail

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
TAG="${TAG:-$(date +%Y%m%d)}"
MODELS="${MODELS:-llama4scout gemma4_31b qwen38_27b}"
export DECAY_RULES="${DECAY_RULES:-text llm_score}"

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
    base="${OUTPUT_ROOT}/amoc_min/${slug}/${TAG}"

    job_id=$(RUN_OUTPUT_DIR="${base}" sbatch --parsable --job-name="pm_${m}" "${script}")

    echo "${m}: job=${job_id} | rules: ${DECAY_RULES}"
    for rule in ${DECAY_RULES}; do
        echo "    ${rule} -> ${base}_${rule}"
    done
done
