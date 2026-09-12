#!/bin/bash
#SBATCH --job-name=amoc_landscape_table
#SBATCH --partition=dgxa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla_a100:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.out
#SBATCH --error=/export/home/acs/stud/a/ana_daria.zahaleanu/exports/%x_%j.err

set -euo pipefail

SIF_IMAGE="/export/projects/nlp/containers/daria-vllm.sif"
PROJECT_DIR="$HOME/to_transfer/amoc-v4-persona-age-experiments"

MATRIX_DIR="${MATRIX_DIR:-${PROJECT_DIR}/study_results/study1_landscape/matrix_results/run_233630_fixC}"
LANDSCAPE="${LANDSCAPE:-${PROJECT_DIR}/study_results/study1_landscape/matrix/landscape_paper_no_inference.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/landscape_study}"

echo "Matrix dir : ${MATRIX_DIR}"
echo "Landscape  : ${LANDSCAPE}"
echo "Output dir : ${OUTPUT_DIR}"

apptainer exec --nv \
    --pwd "$PROJECT_DIR" \
    --env PYTHONPATH="$PROJECT_DIR" \
    -B /export/home/acs/stud/a/ana_daria.zahaleanu:/export/home/acs/stud/a/ana_daria.zahaleanu \
    -B /export/projects/nlp/.cache:/export/projects/nlp/.cache \
    "$SIF_IMAGE" \
    python other_helpers/generate_landscape_latex.py \
        --matrix-dir "$MATRIX_DIR" \
        --landscape  "$LANDSCAPE" \
        --output-dir "$OUTPUT_DIR"

echo ""
echo "Done. Include in your paper with:"
echo "  \\input{landscape_spearman}"
