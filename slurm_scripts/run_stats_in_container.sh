#!/bin/bash
# Wrapper to execute a Python script inside the AMoC container
#!/bin/bash
#SBATCH --job-name=amoc_stats_run
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
PROJECT_DIR="/export/projects/nlp"

apptainer exec --nv \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    bash "$HOME/to_transfer/amoc-v4-persona-age-experiments/run_amoc_stats.py" \
        --model "$MODEL"