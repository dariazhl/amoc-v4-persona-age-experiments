#!/bin/bash
#SBATCH --job-name=amoc_phi4
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

echo "Running Phi-4..."

bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/amoc-run.sh" \
    "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/vllm_train_minimal_v4.py" \
    --models "microsoft/phi-4" \
    --tp 1 \
    --replace-pronouns
