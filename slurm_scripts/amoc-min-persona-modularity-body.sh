#!/bin/bash
# Shared body for the amoc_min persona-modularity runs. NOT an sbatch script:
# each run_<model>_persona_modularity.sh carries its own #SBATCH header, then
# sets MODEL and TP and execs this.
#
# Split out so the env knobs -- WORKERS, LLM_DECAY, STORY_ANALYSIS,
# RUN_OUTPUT_DIR, LIMIT -- are defined once. Five copies of this would drift
# apart the first time one of them gained a flag.
#
# Required from the caller:  MODEL, TP
# Optional:  WORKERS (default 16)  LIMIT  TEMPERATURE  LLM_DECAY
#            STORY_ANALYSIS  RUN_OUTPUT_DIR  OUTPUT_ROOT
#            FIX_KILLED  ALLOW_PERSONA_WORDS

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/export/projects/nlp/.cache"

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced"
: "${MODEL:?set MODEL in the calling script}"
: "${TP:?set TP (tensor-parallel size) in the calling script}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
# Overridable, so an A/B can name its own directory instead of taking the job
# id. Two runs that differ only by the decay rule want names that say so.
MODEL_SLUG="$(basename "${MODEL}" | tr '[:upper:]' '[:lower:]')"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_ROOT}/amoc_min/${MODEL_SLUG}/run_${SLURM_JOB_ID}}"
mkdir -p "${RUN_OUTPUT_DIR}"

mapfile -t PERSONA_FILES < <(ls "${CHUNKS_DIR}"/*.csv | sort)
[[ ${#PERSONA_FILES[@]} -gt 0 ]] || { echo "ERROR: no persona CSVs in ${CHUNKS_DIR}"; exit 1; }

echo "amoc_min | ${MODEL} | TP=${TP} | ${#PERSONA_FILES[@]} persona files"
echo "workers: ${WORKERS:-16} | out: ${RUN_OUTPUT_DIR} | node $(hostname)"
if [[ -n "${LLM_DECAY:-}" ]]; then
  echo "relation ageing: llm -- from NARRATIVE_RELEVANCE score 2 (--llm-decay)"
else
  echo "relation ageing: text -- from the relation's own word (default)"
fi
nvidia-smi -L || { echo "ERROR: nvidia-smi failed"; exit 2; }
apptainer exec --nv /export/projects/nlp/containers/daria-vllm-updated.sif nvidia-smi -L \
    || { echo "ERROR: GPUs not visible inside container"; exit 3; }

PORT_BASE=$((20000 + (SLURM_JOB_ID % 1000) * 40))
VLLM_PORT=""
for p in $(seq "${PORT_BASE}" $((PORT_BASE + 39))); do
    if ! (echo > "/dev/tcp/127.0.0.1/${p}") 2>/dev/null; then VLLM_PORT="${p}"; break; fi
done
[[ -n "${VLLM_PORT}" ]] || { echo "ERROR: no free port"; exit 4; }
VLLM_URL="http://127.0.0.1:${VLLM_PORT}"
VLLM_LOG="/export/home/acs/stud/a/ana_daria.zahaleanu/exports/vllm_min_${SLURM_JOB_ID}.log"

echo "starting vLLM on ${VLLM_URL} (log: ${VLLM_LOG})"
bash "${PROJECT_ROOT}/slurm_scripts/vllm-serve.sh" "${MODEL}" "${VLLM_PORT}" "${TP}" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 360); do
    curl -sf "${VLLM_URL}/health" > /dev/null && break
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM exited during startup; tail of ${VLLM_LOG}:"; tail -n 50 "${VLLM_LOG}"; exit 4
    fi
    sleep 10
done
curl -sf "${VLLM_URL}/health" > /dev/null || { echo "ERROR: vLLM unhealthy after 60m"; exit 4; }
MODELS_JSON="$(curl -sf "${VLLM_URL}/v1/models" || true)"
[[ "${MODELS_JSON}" == *"\"${MODEL}\""* ]] || { echo "ERROR: ${MODEL} not served"; exit 4; }
echo "vLLM ready"

OMP_NUM_THREADS=1 bash "${PROJECT_ROOT}/slurm_scripts/amoc-min-run.sh" \
    --server-url "${VLLM_URL}" \
    --model "${MODEL}" \
    --personas "${PERSONA_FILES[@]}" \
    --out "${RUN_OUTPUT_DIR}" \
    --temperature "${TEMPERATURE:-0.0}" \
    --graphs \
    --workers "${WORKERS:-16}" \
    ${LIMIT:+--limit ${LIMIT}} \
    ${FIX_KILLED:+--fix-killed} \
    ${ALLOW_PERSONA_WORDS:+--allow-persona-words} \
    ${LLM_DECAY:+--llm-decay} \
    ${STORY_ANALYSIS:+--story-analysis "${STORY_ANALYSIS}"} \
    --score
