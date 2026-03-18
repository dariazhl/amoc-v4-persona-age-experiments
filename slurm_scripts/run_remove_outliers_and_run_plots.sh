#!/bin/bash
#SBATCH --job-name=amoc_remove_outliers
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

MODEL="$1"
SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
echo "Running model: $MODEL"

apptainer exec --nv \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    bash "$HOME/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc_sh_simplified.sh" \
        "$HOME/to_transfer/amoc-v4-persona-age-experiments/remove_outliers.py" \
        --input-dir "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/10_example_output" \
        --model "$MODEL" \
        --refined-age-dir "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs/personas_refined_age" \
        --plots-age \
        --plots

#works to run on CPU locally 
# apptainer exec \
#   -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
#   -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
#   /export/projects/nlp/containers/daria-vllm.sif \
#   python "$HOME/to_transfer/amoc-v4-persona-age-experiments/remove_outliers.py" \
#         --input-dir "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets/triplets_16_dec_test" \
#         --model "Qwen/Qwen3-30B-A3B-Instruct-2507"