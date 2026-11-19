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

# vLLM's torch.compile / Triton caches default to ~/.cache/vllm and ~/.triton,
# i.e. the home quota -- which Llama-4-Scout filled ("Disk quota exceeded"
# during compile, job 302685). Keep them on the project filesystem instead.
COMPILE_CACHE="${COMPILE_CACHE:-/export/projects/nlp/daria_vllm_cache}"
mkdir -p "${COMPILE_CACHE}"/{vllm,triton,inductor}

# Optional per-model server flags, whitespace-separated, e.g. to switch a
# hybrid model into its non-thinking mode:
#   VLLM_EXTRA_ARGS='--default-chat-template-kwargs {"enable_thinking":false}'
# Empty for every model that does not set it, so older callers are unchanged.
read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS:-}"

exec apptainer exec --nv \
    --env HF_HOME="${HF_HOME}" \
    --env HF_TOKEN_PATH="${HF_TOKEN_PATH}" \
    --env HF_HUB_CACHE="${HF_HUB_CACHE}" \
    --env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    --env VLLM_CACHE_ROOT="${COMPILE_CACHE}/vllm" \
    --env TRITON_CACHE_DIR="${COMPILE_CACHE}/triton" \
    --env TORCHINDUCTOR_CACHE_DIR="${COMPILE_CACHE}/inductor" \
    -B "${HF_HOME}:${HF_HOME}" \
    -B "${HF_HUB_CACHE}:${HF_HUB_CACHE}" \
    -B "${COMPILE_CACHE}:${COMPILE_CACHE}" \
    "$SIF_IMAGE" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --tensor-parallel-size "${TP}" \
        --trust-remote-code \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8200 \
        ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
