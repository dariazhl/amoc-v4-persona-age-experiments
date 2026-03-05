import argparse
import multiprocessing
import os
import re
import json
import math
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from vllm import LLM, SamplingParams

# -------------------------------------------------------------------
# Multiprocessing & environment setup (same style as your working code)
# -------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Use the same HF_HOME as in your working setup
os.environ.setdefault("HF_HOME", "/export/projects/nlp/.cache")

# -------------------------------------------------------------------
# Regex patterns to detect educational groups from filenames
# -------------------------------------------------------------------
PATTERNS = {
    "primary": re.compile(r"(primary)", re.I),
    "secondary": re.compile(r"(secondary)", re.I),
    "highschool": re.compile(r"(high[_-]?school|highschool)", re.I),
    "university": re.compile(r"(university|uni)", re.I),
}


def detect_group(filename: str) -> Optional[str]:
    lower = filename.lower()
    for group, pattern in PATTERNS.items():
        if pattern.search(lower):
            return group
    return None


# -------------------------------------------------------------------
# Heuristic age extraction
# -------------------------------------------------------------------
AGE_REGEX = re.compile(
    r"(\d{1,2})\s*-*\s*(?:year[s]?\s*[-]?\s*old|y/o|yr[s]?)",
    re.IGNORECASE,
)

PHRASE_AGE_MAP = {
    "child": 5,
    "young child": 4,
    "younger child": 4,
    "kid": 6,
    "little kid": 5,
    "younger sibling": 6,
    "young student": 7,
    "schoolchild": 8,
    "school child": 8,
    "pupil": 8,
    "schoolboy": 8,
    "schoolgirl": 8,
    "primary school student": 8,
    "elementary school student": 9,
    "middle school student": 13,
    "junior high student": 13,
    "high school student": 16,
    "high-school student": 16,
    "college student": 19,
    "university student": 19,
    "student council president": 17,
    "student council": 17,
    "debate team captain": 18,
    "debate team": 17,
    "varsity": 17,
    "cheerleader": 16,
}
AGE_GROUP_DEFAULTS = {
    "young_child": 5,
    "primary_school_child": 9,
    "middle_school_student": 13,
    "high_school_student": 16,
    "university_student": 19,
}


def extract_age_heuristic(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None

    low = text.lower()

    # 1) Explicit age patterns like "16-year-old"
    m = AGE_REGEX.search(low)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass

    # 2) Generic phrase -> default age
    for phrase, age in PHRASE_AGE_MAP.items():
        if phrase in low:
            return age

    # 3) Extra school-level heuristics
    if "preschool" in low:
        return 3
    if "kindergarten" in low:
        return 5
    if "primary school" in low or "elementary school" in low:
        return 8
    if "middle school" in low or "junior high" in low or "secondary school" in low:
        return 13
    if "high school" in low or "highschool" in low:
        return 16
    if "university" in low or "college" in low:
        return 19

    return None


# -------------------------------------------------------------------
# LLM setup & prompt
# -------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a careful classifier.

Task: Given a persona description, infer the approximate age of the person and their age group.

Rules:
- If the persona clearly states an age (e.g. "I am a 16-year-old"), use that age.
- If the persona mentions only a school level (e.g. "high school student", "young child"), pick a reasonable average age for that level. Go with averages.
- If the persona is clearly an adult older than 18 years old, return null.
- If you truly cannot tell an approximate age, return null.

You must return a JSON object with these fields:

{
  "age": <integer or null>,
  "age_group": "<one of: young_child | primary_school_child | middle_school_student | high_school_student | university_student | adult | unknown>",
  "reason": "<short explanation>"
}

- "age" should be an integer if you can reasonably guess, otherwise null.
- "age_group" should always be one of the listed options.
- Do not include any other top-level keys.
- Do not add commentary outside JSON.
""".strip()


def build_prompt(persona_text: str) -> str:
    return (
        SYSTEM_PROMPT
        + "\n\nPersona description:\n"
        + persona_text
        + "\n\nReturn ONLY the JSON object:"
    )


def init_llm(model_name: str, tensor_parallel_size: int) -> Tuple[LLM, SamplingParams]:
    print(f"Loading model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size}\n")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        # you can uncomment these if you used them before:
        # trust_remote_code=True,
        # gpu_memory_utilization=0.80,
        # max_model_len=5000,
    )

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )
    return llm, sampling_params


def call_llm_for_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    personas: List[str],
) -> List[Dict[str, Any]]:
    prompts = [build_prompt(p) for p in personas]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            payload = text[start:end]
            data = json.loads(payload)
        except Exception:
            data = {
                "age": None,
                "age_group": "unknown",
                "reason": f"Could not parse JSON: {text[:120]}",
            }
        results.append(data)
    return results


def resolve_final_age(persona_text: str, llm_result: Dict[str, Any]) -> Optional[int]:
    age = llm_result.get("age", None)
    age_group = llm_result.get("age_group", "unknown")

    if isinstance(age, (int, float)) and not math.isnan(float(age)):
        return int(age)

    if isinstance(age_group, str):
        key = age_group.strip().lower()
        if key in AGE_GROUP_DEFAULTS:
            return AGE_GROUP_DEFAULTS[key]

    return extract_age_heuristic(persona_text)


def fill_missing_ages_in_df(
    df: pd.DataFrame,
    persona_col: str,
    age_col: str,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 16,
) -> pd.DataFrame:
    if persona_col not in df.columns:
        raise ValueError(f"Column '{persona_col}' missing in dataframe.")
    if age_col not in df.columns:
        raise ValueError(f"Column '{age_col}' missing in dataframe.")

    df = df.copy()
    df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    needs_age_mask = df[age_col].isna()
    print(f"  → rows with missing age: {needs_age_mask.sum()}")

    if needs_age_mask.sum() == 0:
        return df

    ages = df[age_col].copy()
    rows_needing_llm: List[int] = []

    # 1) Heuristics
    for idx in df.index[needs_age_mask]:
        persona = str(df.at[idx, persona_col])
        h_age = extract_age_heuristic(persona)
        if h_age is not None:
            ages.at[idx] = h_age
        else:
            rows_needing_llm.append(idx)

    print(f"  → after heuristics, need LLM for {len(rows_needing_llm)} rows.")

    # 2) LLM for remaining
    for start in range(0, len(rows_needing_llm), batch_size):
        batch_indices = rows_needing_llm[start : start + batch_size]
        personas_batch = [str(df.at[i, persona_col]) for i in batch_indices]

        results = call_llm_for_batch(llm, sampling_params, personas_batch)

        for idx, llm_result in zip(batch_indices, results):
            persona = str(df.at[idx, persona_col])
            final_age = resolve_final_age(persona, llm_result)
            if final_age is not None:
                ages.at[idx] = final_age
            # else: leave NaN

    df[age_col] = ages
    print(f"  → after LLM, remaining NaN ages: {df[age_col].isna().sum()}")
    return df


# -------------------------------------------------------------------
# Balancing + age completion combined
# -------------------------------------------------------------------
def balance_persona_csvs_separately_with_age_completion(
    folder: str,
    output_folder: str,
    persona_col: str,
    age_col: str,
    llm: LLM,
    sampling_params: SamplingParams,
    batch_size: int = 16,
):
    group_dfs: Dict[str, List[pd.DataFrame]] = {
        "primary": [],
        "secondary": [],
        "highschool": [],
        "university": [],
    }

    # ---- Step 1–3: load, complete ages, assign groups ----
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".csv"):
            continue

        group = detect_group(fname)
        if group is None:
            print(f"Ignoring file (no educational group match): {fname}")
            continue

        path = os.path.join(folder, fname)
        print(f"\nProcessing file: {fname} → detected group = {group}")
        df = pd.read_csv(path)

        # Fill missing ages
        df = fill_missing_ages_in_df(
            df,
            persona_col=persona_col,
            age_col=age_col,
            llm=llm,
            sampling_params=sampling_params,
            batch_size=batch_size,
        )

        # Optional: drop rows that *still* have no age
        df = df[~df[age_col].isna()].copy()

        if persona_col not in df.columns:
            raise ValueError(f"Column '{persona_col}' missing in {fname}")

        df["edu_group"] = group
        group_dfs[group].append(df)

    # ---- Step 4: merge per group ----
    final_groups: Dict[str, pd.DataFrame] = {}
    for group, dfs in group_dfs.items():
        if len(dfs) == 0:
            print(f"WARNING: No files found for group '{group}'")
            continue
        final_groups[group] = pd.concat(dfs, ignore_index=True)
        print(f"{group}: {len(final_groups[group])} samples after age completion")

    if len(final_groups) < 2:
        raise ValueError("Need at least two educational groups with data to balance.")

    # Some groups might still be empty
    sizes = {g: len(df) for g, df in final_groups.items() if len(df) > 0}
    print("\nGroup sizes after completion:", sizes)

    if len(sizes) == 0:
        raise ValueError("No data to balance after age completion.")

    target_n = min(sizes.values())
    print(f"Balancing to sample size: {target_n} per group\n")

    os.makedirs(output_folder, exist_ok=True)

    balanced_groups: Dict[str, pd.DataFrame] = {}

    # ---- Step 5: downsample & save per group ----
    for group, df in final_groups.items():
        if len(df) == 0:
            continue
        print(f"Balancing {group}...")
        df_bal = df.sample(n=target_n, random_state=42)
        balanced_groups[group] = df_bal

        out_path = os.path.join(output_folder, f"{group}_balanced.csv")
        df_bal.to_csv(out_path, index=False)
        print(f"  → saved to {out_path} ({len(df_bal)} rows)")

    # ---- Step 6: combined balanced dataset ----
    df_combined = pd.concat(list(balanced_groups.values()), ignore_index=True)
    combined_path = os.path.join(output_folder, "all_balanced_combined.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"\nCombined balanced dataset saved to: {combined_path}")
    print(f"Total rows = {len(df_combined)}")

    return balanced_groups, df_combined


# -------------------------------------------------------------------
# Argparse + main (same style as your working script)
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize persona CSVs by completing ages and balancing edu groups."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name or path for vLLM (default: meta-llama/Llama-3.3-70B-Instruct).",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM (default: 4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for LLM age completion (default: 16).",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/balanced_dfs",
        help="Folder with input CSVs to normalize.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/balanced_dfs/normalized",
        help="Folder to write normalized CSVs.",
    )
    parser.add_argument(
        "--persona_col",
        type=str,
        default="persona_text",
        help="Name of the column containing persona text.",
    )
    parser.add_argument(
        "--age_col",
        type=str,
        default="age",
        help="Name of the column containing age.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Init LLM once (same as your working script)
    llm, sampling_params = init_llm(args.model, args.tensor_parallel_size)

    # 2. Run normalization / balancing
    balance_persona_csvs_separately_with_age_completion(
        folder=args.input_folder,
        output_folder=args.output_folder,
        persona_col=args.persona_col,
        age_col=args.age_col,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
