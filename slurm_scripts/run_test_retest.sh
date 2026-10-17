#!/bin/bash
#SBATCH --job-name=amoc_test_retest
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --exclude=dgxa100-ncit-wn04
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-1
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments}"

export AMOC_PERSONA_MODE="${PERSONA_MODE:-reader}"
export AMOC_ALLOW_PERSONA_WORDS="${PERSONA_WORDS:-0}"
REP="${REP:-1}"
export PLOTS=0
export STORY_LIST="${STORY_LIST:-college_4}"

SRC="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced"
export CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/retest_chunks"
mkdir -p "${CHUNKS_DIR}"
for c in ${RETEST_CHUNKS:-primary_000 university_000}; do
    ln -sfn "${SRC}/${c}.csv" "${CHUNKS_DIR}/${c}.csv"
done

TAG="retest_${AMOC_PERSONA_MODE}"
[[ "${AMOC_ALLOW_PERSONA_WORDS}" == "1" ]] && TAG="${TAG}_pwords"
export SWEEP_TAG="${TAG}_rep${REP}"

echo "Test-retest: mode=${AMOC_PERSONA_MODE} persona_words=${AMOC_ALLOW_PERSONA_WORDS} rep=${REP} story=${STORY_LIST} tag=${SWEEP_TAG}"
exec bash "${PROJECT_ROOT}/slurm_scripts/run_story_sweep.sh" tusa_text/min_drp_texts
