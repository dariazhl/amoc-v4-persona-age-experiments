#!/bin/bash
#SBATCH --job-name=amoc_persona_pilot
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --exclude=dgxa100-ncit-wn04
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-31%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments}"

export AMOC_PERSONA_MODE="${PERSONA_MODE:-reader}"
export AMOC_ALLOW_INFERRED_CHAINS="${INFERRED_CHAINS:-0}"
REP="${REP:-1}"

TAG="pilot_${AMOC_PERSONA_MODE}"
[[ "${AMOC_ALLOW_INFERRED_CHAINS}" == "1" ]] && TAG="${TAG}_chains"
export SWEEP_TAG="${TAG}_rep${REP}"
export STORY_LIST="${STORY_LIST:-primary_4,secondary_3,highschool_3,college_4}"

echo "Persona pilot: mode=${AMOC_PERSONA_MODE} chains=${AMOC_ALLOW_INFERRED_CHAINS} rep=${REP} stories=${STORY_LIST}"
exec bash "${PROJECT_ROOT}/slurm_scripts/run_story_sweep.sh" tusa_text/min_drp_texts
