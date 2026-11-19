#!/bin/bash
#SBATCH --job-name=amoc_min_pm_llama4scout
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err
#
# amoc_min on the knight story, meta-llama/Llama-4-Scout-17B-16E-Instruct.
# Same knobs as run_llama70b_persona_modularity.sh (LLM_DECAY, LLM_SCORE,
# RUN_OUTPUT_DIR, STORY_ANALYSIS, LIMIT, WORKERS). To launch both decay rules
# for all three new models: slurm_scripts/submit_new_models_decay.sh
#
# MoE: 17B active but 109B total -- all experts must be resident, ~218 GB in
# bf16. Fits TP=4 on the 80 GB A100s at --gpu-memory-utilization 0.85
# (~54 GB weights + KV per GPU). If it OOMs, go to gpu:tesla_a100:8 + TP=8.
#
# The weights live in .cache/hub/ (not .cache/ like Llama-3.3), hence the
# HF_HUB_CACHE override. OFFLINE so a missing snapshot fails loudly instead
# of starting a 200 GB download inside the job.

export MODEL="meta-llama/Llama-4-Scout-17B-16E-Instruct"
export TP=4
export WORKERS="${WORKERS:-16}"
export HF_HUB_CACHE="/export/projects/nlp/.cache/hub"
export HF_HUB_OFFLINE=1

exec bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-min-persona-modularity-body.sh"
