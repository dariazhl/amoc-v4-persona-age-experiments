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
# Optional:  WORKERS (default 16)  LIMIT  TEMPERATURE  LLM_DECAY  LLM_SCORE  PERSONA_DECAY
#            EXPERIMENT (keefe|obrien)  MAX_ITEMS
#            STORY_ANALYSIS  RUN_OUTPUT_DIR  OUTPUT_ROOT
#            FIX_KILLED
#            PERSONA_STRONG (= PERSONA_GENERATES + READER_CAPACITY + PERSONA_DECAY)
#            PERSONA_GENERATES  READER_CAPACITY  STRICT_SELECTION  LOOSE_LICENSE

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
# DECAY_RULES="llm_score llm_decay" runs several decay rules back to back in
# THIS job, against one vLLM server: the first rule writes story_analysis.json
# and every later rule reuses it, so the A/B differs by the decay rule only --
# without a second job, a --dependency, or a second model load. Each rule
# writes to ${RUN_OUTPUT_DIR}_<rule>. Unset = the old single-rule behaviour.
# Rules: text | llm_decay | llm_score
DECAY_RULES="${DECAY_RULES:-}"
[[ -n "${DECAY_RULES}" ]] || mkdir -p "${RUN_OUTPUT_DIR}"

mapfile -t PERSONA_FILES < <(ls "${CHUNKS_DIR}"/*.csv | sort)
[[ ${#PERSONA_FILES[@]} -gt 0 ]] || { echo "ERROR: no persona CSVs in ${CHUNKS_DIR}"; exit 1; }

echo "amoc_min | ${MODEL} | TP=${TP} | ${#PERSONA_FILES[@]} persona files"
echo "workers: ${WORKERS:-16} | out: ${RUN_OUTPUT_DIR} | node $(hostname)"
echo "persona: strong=${PERSONA_STRONG:-0} generates=${PERSONA_GENERATES:-0} capacity=${READER_CAPACITY:-0} strict_selection=${STRICT_SELECTION:-0} loose_license=${LOOSE_LICENSE:-0} decay=${PERSONA_DECAY:-0}"
if [[ -n "${DECAY_RULES}" ]]; then
  echo "relation ageing: ${DECAY_RULES} -- in sequence, one shared story analysis"
elif [[ -n "${LLM_SCORE:-}" ]]; then
  echo "relation ageing: llm_score -- the model's 0-5 rating is the activation (--llm-score)"
elif [[ -n "${LLM_DECAY:-}" ]]; then
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

if [[ -n "${EXPERIMENT:-}" ]]; then
  # Keefe & McDaniel / O'Brien et al.: the same readers and decay rules as
  # the knight run, over many short texts. See amoc_min/run_texts.py.
  echo "experiment: ${EXPERIMENT} (amoc_min.run_texts)"
  AMOC_MIN_MODULE=amoc_min.run_texts OMP_NUM_THREADS=1 \
    bash "${PROJECT_ROOT}/slurm_scripts/amoc-min-run.sh" \
      --experiment "${EXPERIMENT}" \
      --server-url "${VLLM_URL}" \
      --model "${MODEL}" \
      --personas "${PERSONA_FILES[@]}" \
      --out "${RUN_OUTPUT_DIR}" \
      --temperature "${TEMPERATURE:-0.0}" \
      --workers "${WORKERS:-10}" \
      ${LIMIT:+--limit ${LIMIT}} \
      ${MAX_ITEMS:+--max-items ${MAX_ITEMS}} \
      ${LLM_DECAY:+--llm-decay} \
      ${LLM_SCORE:+--llm-score} \
      ${PERSONA_DECAY:+--persona-decay} \
      ${PERSONA_STRONG:+--persona-strong} \
      ${PERSONA_GENERATES:+--persona-generates} \
      ${READER_CAPACITY:+--reader-capacity} \
      ${STRICT_SELECTION:+--strict-selection} \
      ${LOOSE_LICENSE:+--loose-license}
  exit $?
fi

# One knight-story pass. Args: out_dir llm_decay(0/1) llm_score(0/1) [story_analysis]
run_knight() {
    local out="$1" use_decay="$2" use_score="$3" story="${4:-}"
    local flags=()
    [[ "${use_decay}" == 1 ]] && flags+=(--llm-decay)
    [[ "${use_score}" == 1 ]] && flags+=(--llm-score)
    [[ -n "${story}" ]] && flags+=(--story-analysis "${story}")
    mkdir -p "${out}"
    OMP_NUM_THREADS=1 bash "${PROJECT_ROOT}/slurm_scripts/amoc-min-run.sh" \
        --server-url "${VLLM_URL}" \
        --model "${MODEL}" \
        --personas "${PERSONA_FILES[@]}" \
        --out "${out}" \
        --temperature "${TEMPERATURE:-0.0}" \
        --graphs \
        --workers "${WORKERS:-16}" \
        ${LIMIT:+--limit ${LIMIT}} \
        ${FIX_KILLED:+--fix-killed} \
        ${PERSONA_DECAY:+--persona-decay} \
        ${PERSONA_STRONG:+--persona-strong} \
        ${PERSONA_GENERATES:+--persona-generates} \
        ${READER_CAPACITY:+--reader-capacity} \
        ${STRICT_SELECTION:+--strict-selection} \
        ${LOOSE_LICENSE:+--loose-license} \
        ${flags[@]+"${flags[@]}"} \
        --score
}

if [[ -z "${DECAY_RULES}" ]]; then
    run_knight "${RUN_OUTPUT_DIR}" \
        "$([[ -n "${LLM_DECAY:-}" ]] && echo 1 || echo 0)" \
        "$([[ -n "${LLM_SCORE:-}" ]] && echo 1 || echo 0)" \
        "${STORY_ANALYSIS:-}"
    exit $?
fi

# Several rules, one job. The first rule builds the story analysis (unless
# STORY_ANALYSIS points at an existing one); every later rule reuses it.
SHARED_STORY="${STORY_ANALYSIS:-}"
for rule in ${DECAY_RULES}; do
    out="${RUN_OUTPUT_DIR}_${rule}"
    case "${rule}" in
        text)      d=0; s=0 ;;
        llm_decay) d=1; s=0 ;;
        llm_score) d=0; s=1 ;;
        *) echo "ERROR: unknown decay rule '${rule}' (text|llm_decay|llm_score)"; exit 5 ;;
    esac
    echo "==== decay rule: ${rule} -> ${out} (story analysis: ${SHARED_STORY:-built here})"
    run_knight "${out}" "${d}" "${s}" "${SHARED_STORY}"
    SHARED_STORY="${SHARED_STORY:-${out}/story_analysis.json}"
done
