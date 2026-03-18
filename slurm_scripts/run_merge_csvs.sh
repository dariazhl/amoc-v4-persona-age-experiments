#!/bin/bash
#SBATCH --job-name=amoc_merge_triplets
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err


set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"


if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

echo "Running triplet file merge script"
echo "Project root : $PROJECT_ROOT"

apptainer exec \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    "$SIF_IMAGE" \
    bash -c "
        cd '${PROJECT_ROOT}' || exit 1
        python3 merge_csv_triplet_chunks.py
    "
