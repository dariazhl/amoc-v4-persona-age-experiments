#!/bin/bash
#SBATCH --job-name=amoc_llama70b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-13%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks"

# list of chunk files
CHUNK_FILES=($(ls ${CHUNKS_DIR}/*.csv | sort))
NUM_CHUNKS=${#CHUNK_FILES[@]}

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${NUM_CHUNKS}" ]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} exceeds number of chunks (${NUM_CHUNKS})"
    exit 1
fi

INPUT_FILE="${CHUNK_FILES[$SLURM_ARRAY_TASK_ID]}"

echo "Running LLama 70B"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing chunk file: ${INPUT_FILE}"

bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "meta-llama/Llama-3.3-70B-Instruct" \
    --include-inactive-edges \
    --tp 4 \
    --file "${INPUT_FILE}"