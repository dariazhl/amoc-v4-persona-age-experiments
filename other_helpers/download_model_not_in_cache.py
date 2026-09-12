# # download_phi.py
# import os

# os.environ["HF_HOME"] = "/export/projects/nlp/.cache"
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import argparse
# import sys


# def main(argv):
#     p = argparse.ArgumentParser(description=("Add parameters to the script"))
#     p.add_argument(
#         "--model_name",
#         required=True,
#         help=("Input the name of a LLM ie. microsoft/phi-4 (strigified))"),
#     )
#     args = p.parse_args(argv)

#     MODEL_ID = args.model_name
#     print(f"Starting download of {MODEL_ID} into cache...")


#     try:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#         model = AutoModelForCausalLM.from_pretrained(
#             MODEL_ID, trust_remote_code=True, low_cpu_mem_usage=True
#         )

#         print(f"Download complete.")
#         print(
#             f"Model files should be in the directory rooted at: {os.environ['HF_HOME']}"
#         )
#     except Exception as e:
#         print(f"ERROR: Failed to download model {MODEL_ID}.")
#         print(f"Details: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main(sys.argv[1:])
# download_model_not_in_cache.py
import os

os.environ.setdefault("HF_HOME", "/export/projects/nlp/.cache")

from huggingface_hub import snapshot_download


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        default=os.environ.get("HF_HUB_CACHE"),
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN_DOWNLOAD")
    target = args.cache_dir or os.path.join(os.environ["HF_HOME"], "hub")

    print(f"HF_HOME={os.environ['HF_HOME']}")
    print(f"Downloading {args.model_name} -> {target}")
    if not token:
        print("WARNING: no HF_TOKEN set; gated repos will fail")

    path = snapshot_download(
        repo_id=args.model_name,
        cache_dir=args.cache_dir,
        token=token,
    )

    print(f"Download complete: {path}")


if __name__ == "__main__":
    main()
