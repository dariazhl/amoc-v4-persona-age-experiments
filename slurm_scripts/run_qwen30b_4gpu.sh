#!/bin/bash
#SBATCH --job-name=amoc_qwen30b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
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

echo "Running Qwen 30B"
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing chunk file: ${INPUT_FILE}"

bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --tp 2 \
    --max-rows 10 \
    --plot-after-each-sentence \
    --output-dir "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/10_example_output" \
    --file "${INPUT_FILE}" \
    --strict-reactivate-function \
    --strict-attachament-constraint 