# Refines the age of the personas from the raw dfs. It does the following:
# 1. Separates between rows with "age" = None and the rest.
# 2. For the rows where "age" = None calls the llm and gets the refined age.
# 3. Drops the rows where "age_refined" = None after the llm call.
# 4. Appends the rows with the "age" != None from the original df in "age_refined" and the newly populated ones.
# 5. Save the df as out_path = os.path.join(OUTPUT_FOLDER, filename.replace(".csv", "_age_refined.csv")).
# 6. Adds a flag called --overwrite, which calls the llm for all the ages in the "age" column.
# 7. Saves the df as "_age_refined.csv".

import argparse
import json
import math
import os
import multiprocessing
import re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
import csv
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np
from vllm import LLM, SamplingParams

multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

tokenizer = None
llm = None
sampling_params = None
INPUT_FOLDER = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs"


# ------------------------
# PROMPTS
# ------------------------
SYSTEM_PROMPT_PRIMARY_REFINEMENT = """
You are a constrained attribute annotator.

Task: Refine or validate the AGE of a PERSONA who is a PRIMARY SCHOOL student.

AGE CONSTRAINT:
For this task, we ONLY accept ages 11 or younger.
If the persona implies an age above 11, you MUST return null.

Authoritative rules (in order):
1. If the persona explicitly states a numeric age:
   - If age ≤ 11, return that age.
   - If age > 11, return null.
2. If no numeric age is stated:
   - Infer an age ONLY if it is clearly consistent with primary education
     (e.g., elementary school, young child in school).
3. Do NOT invent precision.
4. If there is insufficient evidence to infer an age ≤ 11, return null.

You MUST NOT:
- Assign an age above 11.
- Infer age based on maturity, tone, or narrative sophistication.

Output format (JSON ONLY):
{
  "age": <integer or null>,
  "confidence": <integer 0-100>,
  "reason": "<brief justification referencing explicit text cues>"
}
""".strip()

SYSTEM_PROMPT_SECONDARY_REFINEMENT = """
You are a constrained attribute annotator.

Task: Refine or validate the AGE of a PERSONA who is a SECONDARY SCHOOL student.

The regime label is authoritative.
You may assume the persona belongs to secondary school unless contradicted by explicit evidence.

AGE CONSTRAINT:
Only ages between 11 and 14 (inclusive) are allowed.
If the persona implies an age outside this range, you MUST return null.

Rules (in order):
1. If the persona explicitly states a numeric age:
   - If 11 ≤ age ≤ 14, return that age.
   - Otherwise, return null.
2. If no numeric age is stated:
   - If the persona is consistent with middle school / lower secondary education,
     infer a typical age between 11 and 14.
   - Prefer common ages (12–13).
3. Do NOT invent precision beyond the regime.
4. Use the confidence score to reflect uncertainty:
   - Explicit age → confidence 80–100
   - Inferred typical age → confidence 40–70
5. Return null ONLY if the persona contradicts secondary school
   or lacks sufficient evidence to confirm regime consistency.

You MUST NOT:
- Assign an age outside 11–14.
- Assume ages beyond this range.

Output format (JSON ONLY):
{
  "age": <integer or null>,
  "confidence": <integer 0-100>,
  "reason": "<brief justification referencing explicit or regime-consistent cues>"
}
""".strip()


SYSTEM_PROMPT_HIGHSCHOOL_REFINEMENT = """
You are a constrained attribute annotator.

Task: Refine or validate the AGE of a PERSONA who is a HIGH SCHOOL student.

The regime label is authoritative.
You may assume the persona belongs to high school unless contradicted by explicit evidence.

AGE CONSTRAINT:
Only ages between 14 and 18 (inclusive) are allowed.
If the persona implies an age outside this range, you MUST return null.

Rules (in order):
1. If the persona explicitly states a numeric age:
   - If 14 ≤ age ≤ 18, return that age.
   - Otherwise, return null.
2. If no numeric age is stated:
   - If the persona is consistent with current high school attendance,
     infer a typical age between 14 and 18.
   - Prefer common ages (15–17).
3. Do NOT invent precision beyond the regime.
4. Use the confidence score to reflect uncertainty:
   - Explicit age → confidence 80–100
   - Inferred typical age → confidence 40–70
5. Return null ONLY if the persona contradicts high school
   or lacks sufficient evidence to confirm regime consistency.

You MUST NOT:
- Assign an age below 14 or above 18.
- Assume post-secondary or adult status.

Output format (JSON ONLY):
{
  "age": <integer or null>,
  "confidence": <integer 0-100>,
  "reason": "<brief justification referencing explicit or regime-consistent cues>"
}
""".strip()


SYSTEM_PROMPT_UNIVERSITY_REFINEMENT = """
You are a constrained attribute annotator.

Task: Refine or validate the AGE of a PERSONA who is a UNIVERSITY student.

The regime label is authoritative.
You may assume the persona is a university student unless contradicted by explicit evidence.

IMPORTANT AGE CONSTRAINT:
Only ages 18 or younger are allowed.
If the persona clearly implies an age above 18, you MUST return null.

Rules (in order):
1. If the persona explicitly states a numeric age:
   - If age ≤ 18, return that age.
   - If age > 18, return null.
2. If no numeric age is stated:
   - If the persona is consistent with early or typical university entry,
     infer a plausible age ≤ 18.
   - Prefer common ages (17–18).
3. Do NOT invent precision beyond the constraint.
4. Use the confidence score to reflect uncertainty:
   - Explicit age → confidence 80–100
   - Inferred plausible age → confidence 40–70
5. Return null ONLY if the persona contradicts university enrollment
   or suggests a typical older undergraduate age.

You MUST NOT:
- Assign an age above 18.
- Assume typical adult undergraduate age ranges.

Output format (JSON ONLY):
{
  "age": <integer or null>,
  "confidence": <integer 0-100>,
  "reason": "<brief justification referencing explicit or regime-consistent cues>"
}
""".strip()

REFINEMENT_PROMPTS = {
    "primary": SYSTEM_PROMPT_PRIMARY_REFINEMENT,
    "secondary": SYSTEM_PROMPT_SECONDARY_REFINEMENT,
    "highschool": SYSTEM_PROMPT_HIGHSCHOOL_REFINEMENT,
    "university": SYSTEM_PROMPT_UNIVERSITY_REFINEMENT,
}


# ------------------------
# VALIDATION
# ------------------------
def age_valid_for_regime(age: int, regime: str) -> bool:
    if regime == "primary":
        return 3 <= age <= 11
    if regime == "secondary":
        return 11 <= age <= 14
    if regime == "highschool":
        return 14 <= age <= 18
    if regime == "university":
        return age <= 18
    return False


# ------------------------
# LLM CALL
# ------------------------
def call_llm_for_batch(
    llm: LLM,
    tokenizer,
    sampling_params: SamplingParams,
    personas: List[str],
    regime: str,
) -> List[Dict[str, Any]]:

    messages = [
        [
            {"role": "system", "content": REFINEMENT_PROMPTS[regime]},
            {
                "role": "user",
                "content": f"Regime: {regime}\nPersona:\n{p}\n\nReturn ONLY JSON.",
            },
        ]
        for p in personas
    ]

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            results.append(json.loads(text[start:end]))
        except Exception:
            results.append({"age": None, "confidence": 0, "reason": "parse failure"})
    return results


# ------------------------
# MAIN
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", required=True, choices=REFINEMENT_PROMPTS.keys())
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-confidence", type=int, default=80)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    # sets flag to fals if there is no overwrite
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Call LLM for all rows, ignoring existing ages",
    )
    args = parser.parse_args()

    input_path = os.path.join(INPUT_FOLDER, f"{args.regime}_FINAL.csv")
    df = pd.read_csv(input_path)

    print("\n=== RAW INPUT DIAGNOSTICS ===")
    print("Total rows:", len(df))
    print("age != NaN:", df["age"].notna().sum())
    print("age == NaN:", df["age"].isna().sum())

    # Ensure columns
    for col in ["age_refined", "age_refine_confidence", "age_refine_reason"]:
        if col not in df.columns:
            df[col] = None

    # ------------------------
    # SPLIT DATA
    # ------------------------
    if args.overwrite:
        df_to_refine = df.copy()
        df_passthrough = pd.DataFrame(columns=df.columns)
    else:
        df_to_refine = df[df["age"].isna()].copy()
        df_passthrough = df[df["age"].notna()].copy()
        df_passthrough["age_refined"] = df_passthrough["age"]

    print("\n=== AFTER SPLIT ===")
    print("df_to_refine (age is NaN):", len(df_to_refine))
    print("df_passthrough (age not NaN):", len(df_passthrough))

    if df_to_refine.empty:
        final_df = df_passthrough
    else:
        # ------------------------
        # INIT LLM
        # ------------------------
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.3, top_p=1)

        valid_rows = []

        # ------------------------
        # LLM LOOP
        # ------------------------
        indices = df_to_refine.index.tolist()

        accepted = 0
        rejected_null_age = 0
        rejected_low_conf = 0
        rejected_regime = 0

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            personas = df_to_refine.loc[batch_idx, "persona_text"].astype(str).tolist()

            results = call_llm_for_batch(
                llm, tokenizer, sampling_params, personas, args.regime
            )

            for idx, r in zip(batch_idx, results):
                age = r.get("age")
                conf = r.get("confidence", 0)

                if isinstance(age, int):
                    if not age_valid_for_regime(age, args.regime):
                        rejected_regime += 1
                    elif conf < args.min_confidence:
                        rejected_low_conf += 1
                    else:
                        accepted += 1
                        row = df_to_refine.loc[idx].copy()
                        row["age_refined"] = age
                        row["age_refine_confidence"] = conf
                        row["age_refine_reason"] = r.get("reason")
                        valid_rows.append(row)
                else:
                    rejected_null_age += 1

        refined_df = pd.DataFrame(valid_rows)
        final_df = pd.concat([df_passthrough, refined_df], ignore_index=True)

        print("\n=== LLM RESULTS ===")
        print("Accepted rows:", accepted)
        print("Rejected (age = null):", rejected_null_age)
        print("Rejected (low confidence):", rejected_low_conf)
        print("Rejected (regime mismatch):", rejected_regime)
        print(
            "Total LLM processed:",
            accepted + rejected_null_age + rejected_low_conf + rejected_regime,
        )

    # ------------------------
    # WRITE OUTPUT
    # ------------------------

    OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "personas_refined_age")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    filename = os.path.basename(input_path)
    out_path = os.path.join(OUTPUT_FOLDER, filename.replace(".csv", "_age_refined.csv"))

    before = len(final_df)
    final_df = final_df[final_df["age_refined"].notna()].reset_index(drop=True)
    print(f"Dropped {before - len(final_df)} rows with null age_refined")
    final_df.to_csv(out_path, index=False)
    print(f"Saved refined file to: {out_path}")


if __name__ == "__main__":
    main()
