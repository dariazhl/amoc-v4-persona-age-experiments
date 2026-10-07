#!/bin/bash
#SBATCH --job-name=keefe_original
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A.err

set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"

DEFAULT_JSON="${PROJECT_ROOT}/amoc/keefe_exp/keefe_ready.json"
DEFAULT_OUTPUT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/keefe_llama/keefe_exp_original.csv"

JSON_FILE="${KEEFE_JSON:-$DEFAULT_JSON}"
OUTPUT_FILE="${KEEFE_OUTPUT:-$DEFAULT_OUTPUT}"

OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

if [ ! -f "$JSON_FILE" ]; then
    echo "ERROR: JSON file not found at $JSON_FILE"
    exit 1
fi

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "Starting Keefe Exp ..."
echo "JSON file: $JSON_FILE"
echo "Output CSV: $OUTPUT_FILE"

apptainer exec --nv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "$(dirname "$JSON_FILE"):$(dirname "$JSON_FILE")" \
    -B "$(dirname "$OUTPUT_FILE"):$(dirname "$OUTPUT_FILE")" \
    -B "${HF_HOME}:${HF_HOME}" \
    "$SIF_IMAGE" \
    bash -c "
        cd '$PROJECT_ROOT'
        exec python3 -m amoc.keefe_exp.keefe_original_paper \"\$@\"
    " bash "$@"