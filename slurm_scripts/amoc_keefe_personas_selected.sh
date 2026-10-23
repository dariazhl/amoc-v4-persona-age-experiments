#!/bin/bash
#SBATCH --job-name=keefe_persona
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --exclude=dgxa100-ncit-wn01
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
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
OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
BASE_OUTPUT_DIR="${OUTPUT_ROOT}/extracted_triplets/keefe_llama"
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
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"

if ! nvidia-smi -L; then
    echo "ERROR: nvidia-smi failed on host $(hostname); GPU allocation/node is broken"
    exit 2
fi
if ! apptainer exec --nv "$SIF_IMAGE" nvidia-smi -L; then
    echo "ERROR: GPUs visible on host $(hostname) but not inside the container"
    exit 3
fi

MODEL="meta-llama/Llama-3.3-70B-Instruct"

WORKERS="${WORKERS:-10}"
PORT_BASE=$((20000 + (SLURM_JOB_ID % 1000) * 40))
VLLM_PORT=""
for p in $(seq "${PORT_BASE}" $((PORT_BASE + 39))); do
    if ! (echo > "/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then
        VLLM_PORT="${p}"
        break
    fi
done
if [[ -z "${VLLM_PORT}" ]]; then
    echo "ERROR: no free port in ${PORT_BASE}-$((PORT_BASE + 39))"
    exit 4
fi
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_LOG="/export/home/acs/stud/a/ana_daria.zahaleanu/exports/vllm_server_keefe_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

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
MODELS_JSON="$(curl -sf "${VLLM_URL}/v1/models" || true)"
if ! kill -0 "${VLLM_PID}" 2>/dev/null || [[ "${MODELS_JSON}" != *"\"${MODEL}\""* ]]; then
    echo "ERROR: server on ${VLLM_URL} is not serving ${MODEL}"
    exit 4
fi
echo "vLLM server ready"

apptainer exec --nv \
    --env PYTHONHASHSEED=0 \
    --env OMP_NUM_THREADS=1 \
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
            --model '$MODEL' \
            --server-url '$VLLM_URL' \
            --workers '$WORKERS' \
            --max-rows 10
    "
