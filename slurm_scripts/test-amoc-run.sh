#!/bin/bash
# Path to your container image on FEP
SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
# The python script you want to run (passed as an argument or hardcoded)
PYTHON_SCRIPT="test_vllm_minimal.py"
# Check if SIF image exists
if [ ! -f "$SIF_IMAGE" ]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    exit 1
fi

apptainer exec --nv \
       -B /export/projects/nlp:/export/projects/nlp \
       $SIF_IMAGE \
       python3 $PYTHON_SCRIPT