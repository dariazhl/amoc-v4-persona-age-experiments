#!/bin/bash
#SBATCH --job-name=amoc_llama70b_small_example
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --exclude=dgxa100-ncit-wn04  # GPU 3 faulted ("GPU requires reset"), 2026-10-11
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
STORY_FILE="${1:-}"
if [[ -n "${STORY_FILE}" && "${STORY_FILE}" != /* ]]; then
    STORY_FILE="${PROJECT_ROOT}/${STORY_FILE}"
fi

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
# Do NOT hardcode CUDA_VISIBLE_DEVICES: Slurm sets it to the GPUs allocated to
# this array task. Overriding it breaks tasks that share a node.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN_ID="run_${SLURM_ARRAY_JOB_ID}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
BASE_OUTPUT_DIR="${OUTPUT_ROOT}/extracted_triplets/small_example_output_llama"
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
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

# Fail fast with a clear message if this task can't see its GPUs
# (instead of vLLM's misleading "Device string must not be empty").
if ! nvidia-smi -L; then
    echo "ERROR: nvidia-smi failed on host $(hostname); GPU allocation/node is broken"
    exit 2
fi
if ! apptainer exec --nv /export/projects/nlp/containers/daria-vllm-updated.sif nvidia-smi -L; then
    echo "ERROR: GPUs visible on host $(hostname) but not inside the container"
    exit 3
fi

if [[ -n "${STORY_FILE}" ]]; then
    echo "Using story file: ${STORY_FILE}"
else
    echo "No story file provided"
fi

STORY_ARG=""
if [[ -n "${STORY_FILE}" ]]; then
    STORY_ARG="--story-text ${STORY_FILE}"
fi

MODEL="meta-llama/Llama-3.3-70B-Instruct"

WORKERS="${WORKERS:-10}"
VLLM_PORT=$((20000 + SLURM_JOB_ID % 20000))
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_LOG="/export/home/acs/stud/a/ana_daria.zahaleanu/exports/vllm_server_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "Starting vLLM server on ${VLLM_URL} (log: ${VLLM_LOG})"
bash "${PROJECT_ROOT}/slurm_scripts/vllm-serve.sh" "${MODEL}" "${VLLM_PORT}" 4 > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 360); do
    if curl -sf "${VLLM_URL}/health" > /dev/null; then
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited during startup; last lines of ${VLLM_LOG}:"
        tail -n 50 "${VLLM_LOG}"
        exit 4
    fi
    sleep 10
done
if ! curl -sf "${VLLM_URL}/health" > /dev/null; then
    echo "ERROR: vLLM server not healthy after 60 minutes"
    exit 4
fi
echo "vLLM server ready"

OMP_NUM_THREADS=1 bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "${MODEL}" \
    --server-url "${VLLM_URL}" \
    --workers "${WORKERS}" \
    --max-rows 10 \
    --plot-after-each-sentence \
    --output-dir "${RUN_OUTPUT_DIR}" \
    --file "${INPUT_FILE}" \
    --strict-reactivate-function \
    --post-process \
    ${STORY_ARG}
