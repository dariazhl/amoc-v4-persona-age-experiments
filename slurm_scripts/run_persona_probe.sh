#!/bin/bash
#SBATCH --job-name=amoc_persona_probe
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --exclude=dgxa100-ncit-wn04
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments}"
SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
STORY="${STORY:-tusa_text/min_drp_texts/college_4.txt}"
OUT_DIR="${OUT_DIR:-/export/projects/nlp/daria_amoc_output/persona_probe/${SLURM_JOB_ID:-local}}"

nvidia-smi -L

PORT_BASE=$((20000 + (${SLURM_JOB_ID:-0} % 1000) * 40))
VLLM_PORT=""
for p in $(seq "${PORT_BASE}" $((PORT_BASE + 39))); do
    if ! (echo > "/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then
        VLLM_PORT="${p}"
        break
    fi
done
[[ -n "${VLLM_PORT}" ]] || { echo "ERROR: no free port"; exit 4; }
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_LOG="${VLLM_LOG_DIR:-/export/home/acs/stud/a/ana_daria.zahaleanu/exports}/vllm_probe_${SLURM_JOB_ID:-local}.log"

bash "${PROJECT_ROOT}/slurm_scripts/vllm-serve.sh" "${MODEL}" "${VLLM_PORT}" 4 > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 360); do
    curl -sf "${VLLM_URL}/health" > /dev/null && break
    kill -0 "${VLLM_PID}" 2>/dev/null || { echo "ERROR: vLLM exited"; tail -n 50 "${VLLM_LOG}"; exit 4; }
    sleep 10
done
curl -sf "${VLLM_URL}/health" > /dev/null || { echo "ERROR: vLLM not healthy"; exit 4; }
echo "vLLM ready on ${VLLM_URL}; writing to ${OUT_DIR}"

mkdir -p "${OUT_DIR}"
export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/export/projects/nlp/.cache}"
apptainer exec --nv \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_TOKEN_PATH="${HF_HOME}/token" \
    --env PYTHONHASHSEED=0 \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    -B "${OUT_DIR}:${OUT_DIR}" \
    "${SIF_IMAGE}" \
    bash -c 'cd "$1" && shift && exec python3 study_results/study3/persona_llm_probe.py "$@"' bash "${PROJECT_ROOT}" \
        --server-url "${VLLM_URL}" \
        --model "${MODEL}" \
        --story "${STORY}" \
        --n-personas "${N_PERSONAS:-5}" \
        --sentences "${SENTENCES:-6}" \
        --reps "${REPS:-3}" \
        --out-dir "${OUT_DIR}"
