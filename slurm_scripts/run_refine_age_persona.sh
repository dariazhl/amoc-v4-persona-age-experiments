#!/bin/bash
#SBATCH --job-name=amoc_refine_age_persona
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

OUT_FILE="$1"
SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
echo "Running Llama 70b..."

apptainer exec --nv \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    bash "$HOME/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc_sh_simplified.sh" \
        "$HOME/to_transfer/amoc-v4-persona-age-experiments/refine_age_personas.py" \
        --regime "$OUT_FILE" \
        --model "Qwen/Qwen3-30B-A3B-Instruct-2507" \
        --batch-size 8 \
        --min-confidence 60 \
        --tensor-parallel-size 2
