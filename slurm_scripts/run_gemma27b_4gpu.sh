#!/bin/bash
#SBATCH --job-name=amoc_gemma3_27b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --array=0-31%1
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

export HUGGING_FACE_HUB_TOKEN="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"

PERSONAS_PER_JOB=50
START_INDEX=$((SLURM_ARRAY_TASK_ID * PERSONAS_PER_JOB))
END_INDEX=$((START_INDEX + PERSONAS_PER_JOB))

echo "Running Gemma 3 27B..."
echo "SLURM ARRAY TASK ID: ${SLURM_ARRAY_TASK_ID}"
echo "Processing personas ${START_INDEX} → ${END_INDEX}"

bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-run.sh" \
    --models "google/gemma-3-27b-it" \
    --tp 4 \
    --replace-pronouns \
    --start-index ${START_INDEX} \
    --end-index ${END_INDEX}

unset HUGGING_FACE_HUB_TOKEN