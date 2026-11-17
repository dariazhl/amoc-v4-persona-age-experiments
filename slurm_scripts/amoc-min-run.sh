#!/bin/bash
# Container wrapper for amoc_min (persona-modularity branch).
# Mirrors amoc-run.sh, except it passes NO AMOC_* env vars: amoc_min has no
# env-var defaults by design -- every parameter lives in amoc_min/config.py
# or on the command line, so nothing can be silently inherited.
set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
OUTPUT_DIR="${OUTPUT_DIR:-/export/projects/nlp/daria_amoc_output}"

# AMOC_MIN_MODULE picks the entry point: amoc_min.run (the knight story,
# default) or amoc_min.run_texts (Keefe / O'Brien). It selects a program,
# not a parameter, so the no-env-defaults rule above still holds.
[ -f "$SIF_IMAGE" ] || { echo "ERROR: container not found at $SIF_IMAGE"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/export/projects/nlp/.cache}"
export HF_TOKEN_PATH="${HF_HOME}/token"
unset TRANSFORMERS_CACHE || true
mkdir -p "${HF_HOME}"

apptainer exec --nv \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN_PATH="${HF_TOKEN_PATH}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    --env PYTHONHASHSEED=0 \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    "$SIF_IMAGE" \
    bash -c '
        cd "$1" || exit 1
        module="$2"
        shift 2
        exec python3 -m "$module" "$@"
    ' bash "${PROJECT_ROOT}" "${AMOC_MIN_MODULE:-amoc_min.run}" "$@"
