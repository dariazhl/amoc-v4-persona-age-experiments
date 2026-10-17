#!/bin/bash
#SBATCH --job-name=amoc_llama70b_small_example
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-159%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

PROJECT_ROOT="${PROJECT_ROOT:-/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced}"
STORY_DIR="${1:-tusa_text/min_drp_texts}"
if [[ "${STORY_DIR}" != /* ]]; then
    STORY_DIR="${PROJECT_ROOT}/${STORY_DIR}"
fi

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=0
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
BASE_OUTPUT_DIR="${OUTPUT_ROOT}/extracted_triplets/small_example_output_llama"
SWEEP_TAG="${SWEEP_TAG:-sweep_$(basename "${STORY_DIR}")}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${SWEEP_TAG}}"

SINGLE_FILE="${SINGLE_FILE:-}"
if [[ -n "${SINGLE_FILE}" ]]; then
    if [[ "${SINGLE_FILE}" != /* ]]; then
        SINGLE_FILE="${PROJECT_ROOT}/${SINGLE_FILE}"
    fi
    if [[ ! -f "${SINGLE_FILE}" ]]; then
        echo "SINGLE_FILE not found: ${SINGLE_FILE}"
        exit 1
    fi
    CHUNK_FILES=("${SINGLE_FILE}")
    echo "SINGLE_FILE override -> only chunk: ${SINGLE_FILE}"
else
    CHUNK_FILES=()
    while IFS=$'\t' read -r _ _ f; do CHUNK_FILES+=("$f"); done < <(
        for f in "${CHUNKS_DIR}"/*.csv; do
            [[ -e "$f" ]] || continue
            b="$(basename "$f" .csv)"
            printf '%s\t%s\t%s\n' "${b##*_}" "${b}" "${f}"
        done | LC_ALL=C sort
    )
fi
STORY_FILES=()
if [[ -n "${STORY_LIST:-}" ]]; then
    IFS=',' read -r -a _stories <<< "${STORY_LIST}"
    for s in "${_stories[@]}"; do
        f="${STORY_DIR}/${s%.txt}.txt"
        if [[ ! -f "${f}" ]]; then
            echo "STORY_LIST entry not found: ${f}"
            exit 1
        fi
        STORY_FILES+=("${f}")
    done
else
    while IFS= read -r f; do STORY_FILES+=("$f"); done < <(ls "${STORY_DIR}"/*.txt 2>/dev/null | LC_ALL=C sort)
fi

NUM_CHUNKS=${#CHUNK_FILES[@]}
NUM_STORIES=${#STORY_FILES[@]}
if [[ "${NUM_CHUNKS}" -eq 0 ]]; then
    echo "No .csv chunk files found in ${CHUNKS_DIR}"
    exit 1
fi
if [[ "${NUM_STORIES}" -eq 0 ]]; then
    echo "No .txt files found in ${STORY_DIR}"
    exit 1
fi
TOTAL_TASKS=$((NUM_CHUNKS * NUM_STORIES))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set (run via sbatch --array or set it for DRY_RUN)}"
if [[ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]]; then
    echo "Task ${TASK_ID} >= ${TOTAL_TASKS} (chunks=${NUM_CHUNKS} x stories=${NUM_STORIES}); nothing to do"
    exit 0
fi

CHUNK_IDX=$((TASK_ID / NUM_STORIES))
STORY_IDX=$((TASK_ID % NUM_STORIES))
INPUT_FILE="${CHUNK_FILES[$CHUNK_IDX]}"
STORY_FILE="${STORY_FILES[$STORY_IDX]}"
CHUNK_TAG="$(basename "${INPUT_FILE}" .csv)"
STORY_TAG="$(basename "${STORY_FILE}" .txt)"

TASK_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${STORY_TAG}/${CHUNK_TAG}"
DONE_MARKER="${TASK_OUTPUT_DIR}/.done"

echo "Running Llama-3.3-70B"
echo "SLURM ARRAY TASK ID: ${TASK_ID} / ${TOTAL_TASKS} (chunks=${NUM_CHUNKS} x stories=${NUM_STORIES})"
echo "Chunk [${CHUNK_IDX}]: ${INPUT_FILE}"
echo "Story [${STORY_IDX}]: ${STORY_FILE}"
echo "Output dir: ${TASK_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -f "${DONE_MARKER}" ]]; then
    echo "Already done ($(cat "${DONE_MARKER}")); skipping"
    exit 0
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN set; not running"
    exit 0
fi

mkdir -p "${TASK_OUTPUT_DIR}"
START_TS=$(date +%s)

MODEL="meta-llama/Llama-3.3-70B-Instruct"
WORKERS="${WORKERS:-10}"
if [[ "${PLOTS:-1}" == "1" ]]; then
    PLOT_ARGS=(--plot-after-each-sentence)
else
    PLOT_ARGS=(--no-reverse-plots)
fi

if ! nvidia-smi -L; then
    echo "ERROR: nvidia-smi failed on host $(hostname); GPU allocation/node is broken"
    exit 2
fi

if [[ "${WORKERS}" -gt 1 ]]; then
    PORT_BASE=$((20000 + (${SLURM_JOB_ID:-0} % 1000) * 40))
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
    VLLM_LOG="${VLLM_LOG_DIR:-/export/home/acs/stud/a/ana_daria.zahaleanu/exports}/vllm_server_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.log"

    echo "Starting vLLM server on ${VLLM_URL} (log: ${VLLM_LOG}), workers=${WORKERS}"
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

    OMP_NUM_THREADS=1 bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
        --models "${MODEL}" \
        --server-url "${VLLM_URL}" \
        --workers "${WORKERS}" \
        --max-rows 10 \
        "${PLOT_ARGS[@]}" \
        --output-dir "${TASK_OUTPUT_DIR}" \
        --file "${INPUT_FILE}" \
        --strict-reactivate-function \
        --post-process \
        --story-text "${STORY_FILE}"
else
    bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
        --models "${MODEL}" \
        --tp 4 \
        --max-rows 10 \
        "${PLOT_ARGS[@]}" \
        --output-dir "${TASK_OUTPUT_DIR}" \
        --file "${INPUT_FILE}" \
        --strict-reactivate-function \
        --post-process \
        --story-text "${STORY_FILE}"
fi

ELAPSED=$(( $(date +%s) - START_TS ))
echo "job=${SLURM_ARRAY_JOB_ID:-local} task=${TASK_ID} finished=$(date -Is) elapsed_s=${ELAPSED}" > "${DONE_MARKER}"
echo "Task ${TASK_ID} complete: story=${STORY_TAG} chunk=${CHUNK_TAG} in ${ELAPSED}s"
