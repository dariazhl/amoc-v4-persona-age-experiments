#!/bin/bash
#SBATCH --job-name=amoc_llama70b_small_example
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-7%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN  
export VLLM_USE_V1=1 

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced"
STORY_DIR="${1:-tusa_text/min_drp_texts}"
if [[ "${STORY_DIR}" != /* ]]; then
    STORY_DIR="${PROJECT_ROOT}/${STORY_DIR}"
fi

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_ID="run_${SLURM_ARRAY_JOB_ID}"
BASE_OUTPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/small_example_output_llama"
RUN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_ID}"
mkdir -p "${RUN_OUTPUT_DIR}"

SINGLE_FILE="${SINGLE_FILE:-}"
if [[ -n "${SINGLE_FILE}" ]]; then
    if [[ "${SINGLE_FILE}" != /* ]]; then
        SINGLE_FILE="${PROJECT_ROOT}/${SINGLE_FILE}"
    fi
    if [[ ! -f "${SINGLE_FILE}" ]]; then
        echo "SINGLE_FILE not found: ${SINGLE_FILE}"
        exit 1
    fi
    INPUT_FILE="${SINGLE_FILE}"
    echo "SINGLE_FILE override -> processing only: ${INPUT_FILE}"
else
    mapfile -t CHUNK_FILES < <(ls "${CHUNKS_DIR}"/*.csv | sort)
    NUM_CHUNKS=${#CHUNK_FILES[@]}

    if [[ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_CHUNKS}" ]]; then
        echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds number of chunks (${NUM_CHUNKS})"
        exit 1
    fi

    INPUT_FILE="${CHUNK_FILES[$SLURM_ARRAY_TASK_ID]}"
fi

echo "Running Llama-3.3-70B"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing chunk file: ${INPUT_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

shopt -s nullglob
STORY_FILES=("${STORY_DIR}"/*.txt)
shopt -u nullglob

if [[ ${#STORY_FILES[@]} -eq 0 ]]; then
    echo "No .txt files found in ${STORY_DIR}"
    exit 1
fi

echo "Story dir: ${STORY_DIR}"
echo "Stories to process: ${#STORY_FILES[@]}"

FAILED=()
INDEX=0

for STORY_FILE in "${STORY_FILES[@]}"; do
    INDEX=$((INDEX + 1))
    STORY_TAG="$(basename "${STORY_FILE}" .txt)"
    STORY_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${STORY_TAG}"
    mkdir -p "${STORY_OUTPUT_DIR}"

    echo "[${INDEX}/${#STORY_FILES[@]}] story=${STORY_TAG} chunk=$(basename "${INPUT_FILE}")"

    if bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
        --models "meta-llama/Llama-3.3-70B-Instruct" \
        --tp 4 \
        --max-rows 10 \
        --plot-after-each-sentence \
        --output-dir "${STORY_OUTPUT_DIR}" \
        --file "${INPUT_FILE}" \
        --strict-reactivate-function \
        --post-process \
        --story-text "${STORY_FILE}"; then
        echo "[${INDEX}/${#STORY_FILES[@]}] ${STORY_TAG} OK"
    else
        echo "[${INDEX}/${#STORY_FILES[@]}] ${STORY_TAG} FAILED"
        FAILED+=("${STORY_TAG}")
    fi
done

echo "Task ${SLURM_ARRAY_TASK_ID} complete: $(( ${#STORY_FILES[@]} - ${#FAILED[@]} ))/${#STORY_FILES[@]} stories succeeded"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed stories: ${FAILED[*]}"
    exit 1
fi
