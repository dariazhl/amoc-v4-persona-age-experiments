#!/bin/bash
# Keefe & McDaniel (1993) on amoc_min (amoc_min.run_texts --experiment keefe)
# for the models below: ONE independent sbatch job per model x decay rule.
#
# This is how mdpi_draft1/experiments/keefe_<model>_{score,llm} were produced:
# 81 readers (80 personas + baseline) x 120 items, text rule (_score) and
# --llm-score (_llm) as separate jobs, each building its own
# text_analysis.json. No --dependency between jobs.
#
# Each job reuses the model's own wrapper, run_<model>_persona_modularity.sh,
# which sets MODEL, TP, the GPU request and any model-specific setting (the
# HF_HUB_CACHE override for the hub/ models, Qwen3.8's thinking-off flag,
# Gemma-4's transformers overlay), then hands EXPERIMENT=keefe to the body.
#
# Run on the cluster login node, NOT with sbatch:
#   bash slurm_scripts/submit_keefe_models.sh                       # all 5 models, text + llm_score
#   MODELS="gemma4_31b" bash slurm_scripts/submit_keefe_models.sh   # one model
#   RULES="llm_score llm_decay" bash slurm_scripts/submit_keefe_models.sh
#   LIMIT=1 MAX_ITEMS=6 TAG=smoke bash slurm_scripts/submit_keefe_models.sh   # smoke test
#   DRY_RUN=1 bash slurm_scripts/submit_keefe_models.sh             # print, submit nothing
#
# EXPERIMENT=obrien runs O'Brien et al. (1998) through the same path instead;
# submit_obrien_models.sh is that, ready-made. Jobs and folders are named
# after the experiment: <experiment>_<model>_<rule>, <experiment>_<TAG>_<rule>.
set -euo pipefail

EXPERIMENT="${EXPERIMENT:-keefe}"
case "${EXPERIMENT}" in keefe|obrien) ;; *) echo "ERROR: EXPERIMENT must be keefe or obrien"; exit 1 ;; esac

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
MODELS="${MODELS:-gemma4_31b llama4scout qwen38_27b llama70b phi4}"
RULES="${RULES:-text llm_score}"     # text | llm_score | llm_decay
TAG="${TAG:-p80}"
LIMIT="${LIMIT:-10}"                 # personas per file; 8 files x 10 = 80
WORKERS="${WORKERS:-10}"

hf_id() {   # model key -> Hugging Face id (must match the wrapper's MODEL)
    case "$1" in
        gemma4_31b)  echo "google/gemma-4-31B-it" ;;
        llama4scout) echo "meta-llama/Llama-4-Scout-17B-16E-Instruct" ;;
        qwen38_27b)  echo "Qwen/Qwen3.8-27B" ;;
        llama70b)    echo "meta-llama/Llama-3.3-70B-Instruct" ;;
        phi4)        echo "microsoft/phi-4" ;;
        *)           return 1 ;;
    esac
}

for m in ${MODELS}; do
    id="$(hf_id "$m")" || { echo "ERROR: unknown model '$m' (gemma4_31b llama4scout qwen38_27b llama70b phi4)"; exit 1; }
    script="${PROJECT_ROOT}/slurm_scripts/run_${m}_persona_modularity.sh"
    [[ -f "${script}" ]] || { echo "ERROR: missing ${script}"; exit 1; }
    # same slug rule as amoc-min-persona-modularity-body.sh
    slug="$(basename "${id}" | tr '[:upper:]' '[:lower:]')"

    for rule in ${RULES}; do
        case "${rule}" in
            text)      flags=() ;;
            llm_score) flags=(LLM_SCORE=1) ;;
            llm_decay) flags=(LLM_DECAY=1) ;;
            *) echo "ERROR: unknown rule '${rule}' (text|llm_score|llm_decay)"; exit 1 ;;
        esac
        out="${OUTPUT_ROOT}/amoc_min/${slug}/${EXPERIMENT}_${TAG}_${rule}"
        name="${EXPERIMENT}_${m}_${rule}"
        envs=(EXPERIMENT="${EXPERIMENT}" RUN_OUTPUT_DIR="${out}" LIMIT="${LIMIT}" WORKERS="${WORKERS}"
              ${MAX_ITEMS:+MAX_ITEMS="${MAX_ITEMS}"} ${flags[@]+"${flags[@]}"})
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "[dry] env ${envs[*]} sbatch --job-name=${name} ${script}"
            continue
        fi
        job_id=$(env "${envs[@]}" sbatch --parsable --job-name="${name}" "${script}")
        echo "${EXPERIMENT} ${m} ${rule}: job=${job_id} -> ${out}"
    done
done
