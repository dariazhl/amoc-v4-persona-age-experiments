import os
import sys

# 1. CONFIGURATION
# Matches your provided file configuration
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"
MODEL_ID = "openai/gpt-oss-120b" 

# 2. GPU DETECTION
# We must correctly detect GPUs to set tensor_parallel_size.
# A 120B model will NOT fit on 1 GPU; it requires sharding across multiple cards.
try:
    N_GPUS = int(os.environ.get("SLURM_GPUS_ON_NODE", os.environ.get("SLURM_GPUS_PER_TASK", 1)))
except:
    # Fallback to counting visible devices if Slurm vars aren't set
    import torch
    N_GPUS = torch.cuda.device_count()

print(f"--- VLLM DIAGNOSTICS ---")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"GPUs Detected: {N_GPUS}")
print(f"Target Model: {MODEL_ID}")
print(f"------------------------")

# 3. IMPORT VLLM
try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import vllm.\nDetails: {e}")
    sys.exit(1)

def run_test():
    # 4. INITIALIZE MODEL
    try:
        print(f"Initializing LLM engine with tensor_parallel_size={N_GPUS}...")
        
        llm = LLM(
            model=MODEL_ID,
            # Critical: 120B models require sharding. This splits the model across N GPUs.
            tensor_parallel_size=N_GPUS, 
            trust_remote_code=True,
            # Adjust memory utilization if needed (0.9 is standard)
            gpu_memory_utilization=0.90,
            # Ensure we don't download to default home dir
            download_dir=os.environ["HF_HOME"]
        )
    except Exception as e:
        print(f"\nFAILED to initialize LLM engine.")
        print(f"Error: {e}")
        print("Note: If the error is 'Revision not found' or 'Entry not found', the model ID is incorrect or you do not have access to it.")
        sys.exit(1)

    # 5. RUN INFERENCE
    try:
        prompts = [
            "The capital of France is",
            "To be or not to be,"
        ]
        
        # Simple sampling params
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)

        print("Model loaded. Running generation...")
        outputs = llm.generate(prompts, sampling_params)

        print("\n--- GENERATION RESULTS ---")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Output: {generated_text!r}\n")
            
        print("SUCCESS: VLLM container and Model are working.")

    except Exception as e:
        print(f"FAILED during generation step: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_test()