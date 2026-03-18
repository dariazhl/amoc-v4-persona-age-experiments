# download_phi.py
import os

os.environ["HF_HOME"] = "/export/projects/nlp/.cache"
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import sys


def main(argv):
    p = argparse.ArgumentParser(description=("Add parameters to the script"))
    p.add_argument(
        "--model_name",
        required=True,
        help=("Input the name of a LLM ie. microsoft/phi-4 (strigified))"),
    )
    args = p.parse_args(argv)

    MODEL_ID = args.model_name
    print(f"Starting download of {MODEL_ID} into cache...")


    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, low_cpu_mem_usage=True
        )

        print(f"Download complete.")
        print(
            f"Model files should be in the directory rooted at: {os.environ['HF_HOME']}"
        )
    except Exception as e:
        print(f"ERROR: Failed to download model {MODEL_ID}.")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
