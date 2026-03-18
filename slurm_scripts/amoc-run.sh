#!/bin/bash
set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_ROOT="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments"
INPUT_DIR="${PROJECT_ROOT}/personas_dfs/personas_refined_age/chunks/"
OUTPUT_DIR="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output"

mkdir -p "${OUTPUT_DIR}/extracted_triplets"
mkdir -p "${OUTPUT_DIR}/amoc_analysis"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

echo "Starting Container..."
echo "Project root: $PROJECT_ROOT"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

export HF_HOME="/export/projects/nlp/.cache"

apptainer exec --nv \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${INPUT_DIR}:${INPUT_DIR}" \
    -B "${OUTPUT_DIR}:${OUTPUT_DIR}" \
    -B "${HF_HOME}:${HF_HOME}" \
    "$SIF_IMAGE" \
    bash -c '
        cd "$1" || exit 1
        shift
        exec python3 -m amoc.cli.main "$@"
    ' bash "${PROJECT_ROOT}" "$@"
