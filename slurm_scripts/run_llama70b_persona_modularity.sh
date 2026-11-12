#!/bin/bash
#SBATCH --job-name=amoc_min_persona_modularity
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err
#
# amoc_min on the knight story, Llama-3.3-70B. ALL personas in ONE job.
#
# No array. EXTRACT/LICENSE run once for the story (26 calls, persona-free)
# and each persona then costs 13 SELECT_READER_INFERENCES calls -- about 1040 calls for all 80.
# Splitting that across array tasks would load the 70B model once per task to
# parallelise ~20 minutes of inference, which costs more than it saves.
# Drop-in sibling of run_llama70b_4gpu_small_example.sh: same partition, GPUs,
# container and vLLM bootstrap. Only the final command differs -- it calls
# amoc_min.run instead of amoc.cli.main. The original script is untouched.
#
#   sbatch slurm_scripts/run_llama70b_persona_modularity.sh
#   DECAY_RATE=0.6 sbatch slurm_scripts/run_llama70b_persona_modularity.sh

set -euo pipefail

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME="/export/projects/nlp/.cache"

PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
CHUNKS_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks_balanced"
MODEL="meta-llama/Llama-3.3-70B-Instruct"

OUTPUT_ROOT="${OUTPUT_ROOT:-/export/projects/nlp/daria_amoc_output}"
export OUTPUT_DIR="${OUTPUT_ROOT}"
RUN_OUTPUT_DIR="${OUTPUT_ROOT}/amoc_min/run_${SLURM_JOB_ID}"
mkdir -p "${RUN_OUTPUT_DIR}"

mapfile -t PERSONA_FILES < <(ls "${CHUNKS_DIR}"/*.csv | sort)
[[ ${#PERSONA_FILES[@]} -gt 0 ]] || { echo "ERROR: no persona CSVs in ${CHUNKS_DIR}"; exit 1; }

echo "amoc_min | ${#PERSONA_FILES[@]} persona files | node $(hostname)"
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
bash "${PROJECT_ROOT}/slurm_scripts/vllm-serve.sh" "${MODEL}" "${VLLM_PORT}" 4 > "${VLLM_LOG}" 2>&1 &
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
    ${LIMIT:+--limit ${LIMIT}} \
    ${FIX_KILLED:+--fix-killed} \
    ${ALLOW_PERSONA_WORDS:+--allow-persona-words} \
    --score
