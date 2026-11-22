#!/bin/bash
# O'Brien et al. (1998) on amoc_min (amoc_min.run_texts --experiment obrien)
# for Gemma-4, Llama-4-Scout, Qwen3.8, Llama-3.3 70B and Phi-4: ONE independent
# sbatch job per model x decay rule, 80 personas + baseline, all 22 items.
#
# Same launcher as Keefe (submit_keefe_models.sh), with EXPERIMENT=obrien, so
# the two experiments cannot drift apart. mdpi_draft1 has O'Brien results only
# for Llama-3.3 70B (obrien_llama70b_{score,llm}: text rule and --llm-score).
#
# Run on the cluster login node, NOT with sbatch:
#   RULES="llm_score" bash slurm_scripts/submit_obrien_models.sh   # score decay, 5 models
#   RULES="llm_decay" bash slurm_scripts/submit_obrien_models.sh   # LLM decay, 5 models
#   RULES="text"      bash slurm_scripts/submit_obrien_models.sh   # text rule (draft1 *_score)
#   MODELS="gemma4_31b" RULES="llm_score" bash slurm_scripts/submit_obrien_models.sh
#   RULES="llm_score" LIMIT=1 MAX_ITEMS=3 TAG=smoke bash slurm_scripts/submit_obrien_models.sh
#   DRY_RUN=1 RULES="llm_score" bash slurm_scripts/submit_obrien_models.sh
#
# Output: /export/projects/nlp/daria_amoc_output/amoc_min/<model>/obrien_<TAG>_<rule>/
#         obrien_scores.csv + text_analysis.json
# Logs:   ~/exports/obrien_<model>_<rule>_<jobid>.{out,err}, vllm_min_<jobid>.log
set -euo pipefail
EXPERIMENT=obrien exec bash "$(dirname "$0")/submit_keefe_models.sh"
