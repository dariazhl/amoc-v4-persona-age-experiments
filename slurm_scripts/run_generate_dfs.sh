#!/bin/bash
#SBATCH --job-name=amoc_generate_dfs
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4  
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --array=0-15
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
        "$HOME/to_transfer/amoc-v4-persona-age-experiments/generate_dfs_edu_all_personas.py" \
        --model "meta-llama/Llama-3.3-70B-Instruct" \
        --file "$OUT_FILE" \
        --tensor_parallel_size 4 \
        --min_confidence 80 \
        --shard-id $SLURM_ARRAY_TASK_ID \
        --num-shards 16
