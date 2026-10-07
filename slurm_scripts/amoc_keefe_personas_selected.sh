#!/bin/bash
#SBATCH --job-name=keefe_persona
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-7%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced"
KEEFE_JSON="${PROJECT_ROOT}/amoc/keefe_exp/keefe_ready.json"
BASE_OUTPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/keefe_llama"
RUN_ID="keefe_${SLURM_ARRAY_JOB_ID}"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_OUTPUT_DIR}"

mapfile -t CHUNK_FILES < <(ls "${CHUNKS_DIR}"/*.csv | sort -V)
NUM_CHUNKS=${#CHUNK_FILES[@]}

if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_CHUNKS}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds number of chunks (${NUM_CHUNKS})"
    exit 1
fi

INPUT_FILE="${CHUNK_FILES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_CSV="${RUN_OUTPUT_DIR}/keefe_scores_chunk_${SLURM_ARRAY_TASK_ID}.csv"
STATS_OUTPUT="${RUN_OUTPUT_DIR}/lme_stats_chunk_${SLURM_ARRAY_TASK_ID}.csv"

echo "Processing chunk: ${INPUT_FILE}"
echo "Output: ${OUTPUT_CSV}"
echo "Stats output: ${STATS_OUTPUT}"

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
apptainer exec --nv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${CHUNKS_DIR}:${CHUNKS_DIR}" \
    -B "$(dirname "${OUTPUT_CSV}"):$(dirname "${OUTPUT_CSV}")" \
    -B "${HF_HOME}:${HF_HOME}" \
    "$SIF_IMAGE" \
    bash -c "
        cd '$PROJECT_ROOT'
        python3 -m amoc.keefe_exp.keefe_original_paper_personas \
            --persona-csv '$INPUT_FILE' \
            --keefe-json '$KEEFE_JSON' \
            --output-csv '$OUTPUT_CSV' \
            --stats-output '$STATS_OUTPUT' \
            --model meta-llama/Llama-3.3-70B-Instruct \
            --max-rows 10 \
            --tp 4
    "