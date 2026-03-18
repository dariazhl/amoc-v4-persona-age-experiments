#!/bin/bash
#SBATCH --job-name=amoc_generate_dfs
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=8          
#SBATCH --mem=64G                  
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
echo "Running Llama 70b..."

apptainer exec --nv \
  -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
  -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
  "$SIF_IMAGE" \
  python "$HOME/to_transfer/amoc-v4-persona-age-experiments/normalize_dfs.py" \
    --model "meta-llama/Llama-3.3-70B-Instruct" \
    --tensor_parallel_size 4 \
    --batch_size 16
