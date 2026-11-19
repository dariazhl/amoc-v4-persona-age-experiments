#!/bin/bash
#SBATCH --job-name=amoc_min_pm_gemma4_31b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err
#
# amoc_min on the knight story, google/gemma-4-31B-it.
# Same knobs as run_llama70b_persona_modularity.sh (LLM_DECAY, LLM_SCORE,
# RUN_OUTPUT_DIR, STORY_ANALYSIS, LIMIT, WORKERS). To launch both decay rules
# for all three new models: slurm_scripts/submit_new_models_decay.sh
#
# Dense 31B, ~62 GB in bf16 -> TP=2 on the 80 GB A100s.
# Weights live in .cache/hub/, hence the HF_HUB_CACHE override; OFFLINE so a
# missing snapshot fails loudly instead of downloading inside the job.

export MODEL="google/gemma-4-31B-it"
export TP=2
export WORKERS="${WORKERS:-16}"
export HF_HUB_CACHE="/export/projects/nlp/.cache/hub"
export HF_HUB_OFFLINE=1

exec bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-min-persona-modularity-body.sh"
