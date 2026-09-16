#!/bin/bash
#SBATCH --job-name=amoc_llama70b_small_example
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-159%2
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%A_%a.err

# Flat story sweep: one array task = one (persona chunk, story) pair.
#
#   task id  ->  chunk = id / NUM_STORIES,  story = id % NUM_STORIES
#
# With 8 chunks x 20 stories the array is 0-159. Each task runs ~2-3 h
# (one model load + 10 personas), so every task fits the 24 h limit.
# Output goes to a fixed tree per story dir, so resubmitting the same
# array only runs the pairs that have no .done marker yet:
#
#   <BASE_OUTPUT_DIR>/sweep_<story_dir>/<story>/<chunk>/{triplets,graphs,matrix}
#
# Usage:
#   sbatch slurm_scripts/run_story_sweep.sh                       # tusa_text/min_drp_texts
#   sbatch slurm_scripts/run_story_sweep.sh tusa_text/other_dir   # different story set
#   sbatch --array=0-159%4 slurm_scripts/run_story_sweep.sh       # different throttle
#   DRY_RUN=1 SLURM_ARRAY_TASK_ID=17 bash slurm_scripts/run_story_sweep.sh   # show mapping only
#
# Final statistics are NOT run per task (the array is too large for the
# sentinel logic in amoc.cli.main); aggregate once after the sweep finishes.

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

PROJECT_ROOT="${PROJECT_ROOT:-/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments}"
CHUNKS_DIR="${CHUNKS_DIR:-${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced}"
STORY_DIR="${1:-tusa_text/min_drp_texts}"
if [[ "${STORY_DIR}" != /* ]]; then
    STORY_DIR="${PROJECT_ROOT}/${STORY_DIR}"
fi

export HF_HOME="/export/projects/nlp/.cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=0
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
BASE_OUTPUT_DIR="${OUTPUT_ROOT}/extracted_triplets/small_example_output_llama"
# Fixed (not per-job) so that repeated submissions fill in the same tree.
SWEEP_TAG="${SWEEP_TAG:-sweep_$(basename "${STORY_DIR}")}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${SWEEP_TAG}}"

# --- Build the (chunk, story) grid with a locale-independent order ---------
SINGLE_FILE="${SINGLE_FILE:-}"
if [[ -n "${SINGLE_FILE}" ]]; then
    if [[ "${SINGLE_FILE}" != /* ]]; then
        SINGLE_FILE="${PROJECT_ROOT}/${SINGLE_FILE}"
    fi
    if [[ ! -f "${SINGLE_FILE}" ]]; then
        echo "SINGLE_FILE not found: ${SINGLE_FILE}"
        exit 1
    fi
    CHUNK_FILES=("${SINGLE_FILE}")
    echo "SINGLE_FILE override -> only chunk: ${SINGLE_FILE}"
else
    CHUNK_FILES=()
    while IFS= read -r f; do CHUNK_FILES+=("$f"); done < <(ls "${CHUNKS_DIR}"/*.csv 2>/dev/null | LC_ALL=C sort)
fi
STORY_FILES=()
while IFS= read -r f; do STORY_FILES+=("$f"); done < <(ls "${STORY_DIR}"/*.txt 2>/dev/null | LC_ALL=C sort)

NUM_CHUNKS=${#CHUNK_FILES[@]}
NUM_STORIES=${#STORY_FILES[@]}
if [[ "${NUM_CHUNKS}" -eq 0 ]]; then
    echo "No .csv chunk files found in ${CHUNKS_DIR}"
    exit 1
fi
if [[ "${NUM_STORIES}" -eq 0 ]]; then
    echo "No .txt files found in ${STORY_DIR}"
    exit 1
fi
TOTAL_TASKS=$((NUM_CHUNKS * NUM_STORIES))

TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set (run via sbatch --array or set it for DRY_RUN)}"
if [[ "${TASK_ID}" -ge "${TOTAL_TASKS}" ]]; then
    echo "Task ${TASK_ID} >= ${TOTAL_TASKS} (chunks=${NUM_CHUNKS} x stories=${NUM_STORIES}); nothing to do"
    exit 0
fi

CHUNK_IDX=$((TASK_ID / NUM_STORIES))
STORY_IDX=$((TASK_ID % NUM_STORIES))
INPUT_FILE="${CHUNK_FILES[$CHUNK_IDX]}"
STORY_FILE="${STORY_FILES[$STORY_IDX]}"
CHUNK_TAG="$(basename "${INPUT_FILE}" .csv)"
STORY_TAG="$(basename "${STORY_FILE}" .txt)"

# One directory per (story, chunk): the pipeline names reverse plots and
# matrices by the chunk-local persona index (0..9), so chunks sharing a
# directory would overwrite each other.
TASK_OUTPUT_DIR="${RUN_OUTPUT_DIR}/${STORY_TAG}/${CHUNK_TAG}"
DONE_MARKER="${TASK_OUTPUT_DIR}/.done"

echo "Running Llama-3.3-70B"
echo "SLURM ARRAY TASK ID: ${TASK_ID} / ${TOTAL_TASKS} (chunks=${NUM_CHUNKS} x stories=${NUM_STORIES})"
echo "Chunk [${CHUNK_IDX}]: ${INPUT_FILE}"
echo "Story [${STORY_IDX}]: ${STORY_FILE}"
echo "Output dir: ${TASK_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -f "${DONE_MARKER}" ]]; then
    echo "Already done ($(cat "${DONE_MARKER}")); skipping"
    exit 0
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY_RUN set; not running"
    exit 0
fi

mkdir -p "${TASK_OUTPUT_DIR}"
START_TS=$(date +%s)

# --post-process is kept deliberately: it makes amoc.cli.main write a
# per-task sentinel and skip its leader-only statistics pass, which would
# otherwise run on a single (story, chunk) directory for task 0.
bash "${PROJECT_ROOT}/slurm_scripts/amoc-run.sh" \
    --models "meta-llama/Llama-3.3-70B-Instruct" \
    --tp 4 \
    --max-rows 10 \
    --plot-after-each-sentence \
    --output-dir "${TASK_OUTPUT_DIR}" \
    --file "${INPUT_FILE}" \
    --strict-reactivate-function \
    --post-process \
    --story-text "${STORY_FILE}"

ELAPSED=$(( $(date +%s) - START_TS ))
echo "job=${SLURM_ARRAY_JOB_ID:-local} task=${TASK_ID} finished=$(date -Is) elapsed_s=${ELAPSED}" > "${DONE_MARKER}"
echo "Task ${TASK_ID} complete: story=${STORY_TAG} chunk=${CHUNK_TAG} in ${ELAPSED}s"
