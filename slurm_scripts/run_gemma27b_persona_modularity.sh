#!/bin/bash
#SBATCH --job-name=amoc_min_pm_gemma27b
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err
#
# amoc_min on the knight story, google/gemma-3-27b-it. ALL personas in ONE job.
#
# The story analysis runs once (persona-free, ~26 calls) and every reader then
# costs 13 calls. Those are parallelised with WORKERS processes against the one
# vLLM server, which batches them -- so this is a single job, not an array:
# an array would reload the model per task to parallelise the same work.
#
#   sbatch slurm_scripts/run_gemma27b_persona_modularity.sh
#   WORKERS=16 LIMIT=1 sbatch slurm_scripts/run_gemma27b_persona_modularity.sh
#
# LLM_DECAY=1 ages a relation from when the model last judged it RELEVANT
# (NARRATIVE_RELEVANCE score 2) rather than from when its own word last
# appeared. Unset keeps the text rule, the one comparable with the reference.
# An A/B must reuse one story_analysis.json or the runs differ by extraction
# too, and the decay rule is no longer the only variable:
#
#   RUN_OUTPUT_DIR=.../ab_text LIMIT=1 \
#     sbatch slurm_scripts/run_gemma27b_persona_modularity.sh
#   RUN_OUTPUT_DIR=.../ab_llm  LIMIT=1 LLM_DECAY=1 \
#     STORY_ANALYSIS=.../ab_text/story_analysis.json \
#     sbatch slurm_scripts/run_gemma27b_persona_modularity.sh
#
# Everything below is shared with the other models.

export MODEL="google/gemma-3-27b-it"
export TP=4
export WORKERS="${WORKERS:-16}"

exec bash "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/slurm_scripts/amoc-min-persona-modularity-body.sh"
