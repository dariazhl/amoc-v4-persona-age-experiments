import os

# Root folders for data and outputs
INPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs/chunks"
OUTPUT_DIR = (
    "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output/extracted_triplets"
)
OUTPUT_ANALYSIS_DIR = os.path.join(
    "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/output", "amoc_analysis"
)

try:
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)
except PermissionError:
    OUTPUT_ANALYSIS_DIR = os.path.join(os.getcwd(), "output", "amoc_analysis")
    os.makedirs(OUTPUT_ANALYSIS_DIR, exist_ok=True)

VLLM_MODELS = {
    "qwen3:30b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-oss:120b": "openai/gpt-oss-120b",
}
