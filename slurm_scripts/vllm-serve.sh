#!/bin/bash
set -euo pipefail

MODEL="$1"
PORT="$2"
TP="$3"

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm-updated.sif"

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/export/projects/nlp/.cache}"
export HF_TOKEN_PATH="${HF_HOME}/token"
unset TRANSFORMERS_CACHE
mkdir -p "${HF_HOME}"

exec apptainer exec --nv \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN_PATH="${HF_TOKEN_PATH}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    "$SIF_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --tensor-parallel-size "${TP}" \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8200
