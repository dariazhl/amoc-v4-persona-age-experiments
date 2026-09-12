#!/bin/bash
#SBATCH --job-name=model_download
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
PROJECT_ROOT="$HOME/to_transfer/amoc-v4-persona-age-experiments/other_helpers"
MODEL_ID="${1:-}"
CACHE_DIR="${2:-${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"

if [ -z "${MODEL_ID}" ]; then
    echo "Usage: HF_TOKEN=hf_... sbatch $0 <model/id> [cache_dir]"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set."
    echo "Submit with: HF_TOKEN=hf_... sbatch $0 ${MODEL_ID}"
    exit 1
fi

if [ ! -f "${SIF_IMAGE}" ]; then
    echo "ERROR: Container image not found at ${SIF_IMAGE}"
    exit 1
fi

mkdir -p "${CACHE_DIR}"

echo "Model:     ${MODEL_ID}"
echo "Cache dir: ${CACHE_DIR}"
df -h "${CACHE_DIR}" | tail -1

apptainer exec \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HUB_CACHE="${CACHE_DIR}" \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${HOME}:${HOME}" \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "${SIF_IMAGE}" \
    python3 "${PROJECT_ROOT}/download_model_not_in_cache.py" \
    --model_name "${MODEL_ID}" \
    --cache-dir "${CACHE_DIR}"

echo "Download finished into: ${CACHE_DIR}"
du -sh "${CACHE_DIR}"
