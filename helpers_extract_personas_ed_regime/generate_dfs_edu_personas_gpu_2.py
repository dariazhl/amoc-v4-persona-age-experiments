import argparse
import json
import multiprocessing
import os
import re
from typing import List, Dict
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import csv
import math
from typing import Optional, Dict, Any, List, Tuple

tokenizer = None
OUTPUT_FOLDER = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs"
FINAL_HS_FILE = os.path.join(OUTPUT_FOLDER, "highschool_FINAL.csv")
# -------------------------------------------------------------------
# Multiprocessing & environment setup
# -------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

PERSONAHUB_CONFIGS = [
    "persona",
    "instruction",
    "knowledge",
    "math",
    "npc",
    "reasoning",
    "tool",
]

JUDGE_SYSTEM_PROMPT_PRIMARY = """
You are a strict classifier.

Task: Decide whether the persona describes a primary school student.

Definitions:
- Primary school = preschool, kindergarten, or elementary/primary education, typically up to around ages 11–12.
- The key requirement is that the PERSONA is a child who is currently attending primary/elementary school (or an equivalent early school level).
- Exclude:
  - Middle school or secondary school students.
  - High school students.
  - University/college students.
  - Teachers, parents, or other adults talking about primary school children.
  - Adults describing their past in primary school.
- If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a child currently attending primary/elementary school (or preschool/kindergarten), label "yes".
- If the persona clearly belongs to a different educational level (secondary, high school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_SECONDARY = """
You are a strict classifier.

Task: Decide whether the persona describes a secondary school student (middle school / lower secondary).

Definitions:
- Secondary school (for this task) = middle school, junior high, or lower secondary education that comes after primary/elementary school and before high school.
- Typical age range is around 11–14, but age alone is not sufficient; the persona must be clearly positioned in this middle / lower secondary stage.
- The key requirement is that the PERSONA is currently attending a secondary school level that is after primary but not yet high school.

Exclude:
- Primary / elementary / preschool children.
- High school students.
- University / college students.
- Teachers, parents, or other adults talking about secondary school children.
- Adults describing their past in secondary school.

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current middle school / junior high / lower secondary school student, label "yes".
- If the persona clearly belongs to a different educational level (primary, high school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_HIGHSCHOOL = """
You are a strict classifier.

Task: Decide whether the persona describes a CURRENT high school / upper secondary student.

Definition:
- High school = upper secondary education immediately prior to university or college.
- The persona MUST explicitly describe an individual who is currently enrolled in high school or an equivalent upper secondary institution.

MANDATORY CONDITIONS (ALL must be satisfied):

1. The persona explicitly identifies the individual as a high school / upper secondary student
   (e.g., “a high school student”, “a high school senior”, “a student in upper secondary school”).

2. The persona includes at least ONE explicit reference to a concrete school context,
   such as:
   - classes or coursework
   - teachers or classmates
   - exams, finals, or graduation
   - school life, school environment, or curriculum

Role labels WITHOUT school context are NOT sufficient.

NOT accepted as sufficient evidence (even if present):
- Mentions of age alone
- Generic mentions of being “a student”
- Mentions of exams or studying WITHOUT explicit reference to high school
- Statements about preparing for university without describing current high school attendance
- Mentions of “school” without specifying high school or upper secondary
- Statements about past high school experiences
- Third-person descriptions that only assign a demographic role

Explicit exclusions (automatic "no"):
- Primary, elementary, middle school, or lower secondary references
- University, college, or post-secondary references
- Vocational or professional training unless explicitly stated to be part of a high-school program
- Adults recalling or narrating past high school experiences
- Teachers, parents, or third parties describing high school students

Decision rules:
- Label "yes" ONLY if ALL mandatory conditions are met with explicit textual evidence.
- Label "no" if another educational level is explicitly stated.
- Label "uncertain" in ALL other cases.

Output format (JSON ONLY):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0–100>,
  "reason": "<direct quote or paraphrase of the explicit evidence>"
}

Return ONLY this JSON object.
"""

JUDGE_SYSTEM_PROMPT_UNI = """
You are a strict classifier.

Task: Decide whether the persona describes a university freshman (first-year university or college student).

Definitions:
- University freshman = a person who is currently in their first year of a university, college, or equivalent higher-education program (including community college, polytechnic, etc.).
- They may be described as "freshman", "first-year student", "first year at university/college", or similar.
- The key requirement is that the PERSONA is currently in their first year of a higher-education degree program.

Exclude:
- High school students.
- Middle school or primary/elementary students.
- University students who are not in their first year (e.g., second-year, third-year, senior, graduate student, master's student, PhD student).
- Teachers, professors, or other staff.
- Adults only mentioning that they attended university in the past (not currently enrolled).

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current first-year university/college student, label "yes".
- If the persona clearly belongs to a different educational level (high school, later-year university, graduate school, adult not in school), label "no".
- If the educational level or year of study is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

SYSTEM_PROMPT = """
You are a constrained attribute annotator.

Your task is to infer an APPROXIMATE AGE RANGE for a PERSONA.
This is NOT a creative task. You must follow the rules strictly.

Authoritative rules (in order):
1. If the persona explicitly states a numeric age (e.g. "I am 16 years old"):
   - Set age_min = age_max = that exact age.
2. If no numeric age is stated, infer an AGE RANGE based ONLY on the described school level or role.
3. If the persona is clearly an adult (18+), return null for both age_min and age_max.
4. If there is not enough information to infer an age range, return null for both age_min and age_max.

You MUST NOT:
- Invent precise ages when only school level is known.
- Guess outside typical educational age ranges.
- Use stereotypes, emotions, or narrative cues.
- Output age_min or age_max ≥ 18 for a student persona.
- Override explicit numeric ages.

Allowed school-level → age RANGE mappings (use ONLY these):

- young_child → age_min: 4, age_max: 6
- primary_school_child → age_min: 7, age_max: 11
- middle_school_student → age_min: 11, age_max: 14
- high_school_student → age_min: 14, age_max: 18
- university_student → age_min: 18, age_max: 22

Age group definitions:
- young_child: preschool or early childhood references
- primary_school_child: elementary / primary school
- middle_school_student: middle school / junior high / lower secondary
- high_school_student: high school / upper secondary
- university_student: college / university
- adult: clearly post-education or full-time work
- unknown: insufficient information

If the persona mentions high school but gives no grade or age, always use:
- age_min = 14
- age_max = 18

Output format (JSON ONLY, no extra text):
{
  "age_min": <integer or null>,
  "age_max": <integer or null>,
  "age_group": "<one of: young_child | primary_school_child | middle_school_student | high_school_student | university_student | adult | unknown>",
  "reason": "<brief factual justification>"
}

The "reason" must reference ONLY explicit text cues (e.g. "mentions high school student").
Do not add commentary, hedging, or speculation.
""".strip()


# -------------------------------------------------------------------
# Keyword lists for quick filtering
# -------------------------------------------------------------------
PRIMARY_EXCLUDE_KEYWORDS = [
    "teacher",
    "professor",
    "lecturer",
    "instructor",
    "parent",
    "mother",
    "father",
    "mom",
    "dad",
    "doctoral",
    "phd",
    "postdoc",
]

SECONDARY_EXCLUDE_KEYWORDS = [
    "professor",
    "lecturer",
    "postdoc",
    "phd",
    "kindergarten teacher",
    "preschool teacher",
]

HIGH_EXCLUDE_KEYWORDS = [
    "professor",
    "lecturer",
    "postdoc",
    "phd",
    "kindergarten teacher",
    "preschool teacher",
]

UNI_EXCLUDE_KEYWORDS = [
    "high school teacher",
    "middle school teacher",
    "elementary teacher",
    "primary teacher",
    "principal",
    "headmaster",
]

# These are used by is_relevant()
YOUNG_EDU_KEYWORDS = [
    # School Levels
    "primary school",
    "elementary school",
    "kindergarten",
    "preschool",
    "nursery school",
    "grade school",
    "middle school",
    "junior high",
    # Grades / Years (Specific formats)
    "1st grade",
    "2nd grade",
    "3rd grade",
    "4th grade",
    "5th grade",
    "grade 1",
    "grade 2",
    "grade 3",
    "grade 4",
    "grade 5",
    "year 1",
    "year 2",
    "year 3",
    "year 4",
    "year 5",
    # Roles & Activities
    "pupil",
    "young student",
    "young learner",
    "schoolboy",
    "schoolgirl",
    "kindergartener",
    "preschooler",
    "pre-schooler",
    "kid",
    "child",
    "learning to read",
    "learning to write",
    "alphabet",
    "reading",
    "storytime",
    "story time",
]

SECONDARY_EDU_KEYWORDS = [
    "secondary school",
    "middle school",
    "preparatory school",
    "college prep",
    "academy",
    "gymnasium",
    "lyceum",
    "comprehensive school",
    # Grades / Years
    "6th grade",
    "7th grade",
    "8th grade",
    "grade 6",
    "grade 7",
    "grade 8",
    "year 6",
    "year 7",
    "year 8",
    # Roles & Identity
    "teen student",
    "teenage student",
    "middle schooler",
    "adolescent student",
    "student council",
    "varsity team",
]

HIGH_SCHOOL_KEYWORDS = [
    # Institutions
    "high school",
    "senior high",
    "college prep",
    "preparatory school",
    "boarding school",
    "academy",
    # Grades / Years (US & International)
    "9th grade",
    "10th grade",
    "11th grade",
    "12th grade",
    "grade 9",
    "grade 10",
    "grade 11",
    "grade 12",
    "year 9",
    "year 10",
    "year 11",
    "year 12",
    "freshman high",
    "sophomore high",
    "junior high",
    "senior high",
    "high school freshman",
    "high school sophomore",
    "high school junior",
    "high school senior",
    # Roles & Identity
    "high school student",
    "high schooler",
    "teen student",
    "teenage student",
    "student council",
    "varsity",
    "yearbook club",
    "debate team",
]

# 1. UNIVERSITY KEYWORDS (18 + - including 18)
# =========================================================
UNIVERSITY_KEYWORDS = [
    # Institutions
    "university",
    "college",
    "campus",
    "medical school",
    "law school",
    "business school",
    "community college",
    "liberal arts college",
    "polytechnic",
    # Levels / Degrees
    "undergrad",
    "undergraduate",
    "bachelor's",
    # Years / Status
    "freshman",
    "sophomore",
    "junior",
    "senior",
    "first year",
    "second year",
    "third year",
    "final year",
    "major in",
    "minoring in",
    "studying for a degree",
    "thesis",
    "capstone",
    # Identity
    "college student",
    "university student",
    "med student",
    "law student",
    "engineering student",
    "art student",
]


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

HIGH_SCHOOL_ROLE_KEYWORDS = [
    "high school student",
    "high schooler",
    "high school senior",
    "high school junior",
    "high school sophomore",
    "high school freshman",
    "upper secondary student",
]


# -------------------------------------------------------------------
# Simple exclusion helpers
# -------------------------------------------------------------------
def should_exclude_primary(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in PRIMARY_EXCLUDE_KEYWORDS)


def should_exclude_secondary(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SECONDARY_EXCLUDE_KEYWORDS)


def should_exclude_highschool(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in HIGH_EXCLUDE_KEYWORDS)


def should_exclude_university(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in UNI_EXCLUDE_KEYWORDS)


def is_role_based_high_school(text: str) -> bool:
    t = text.lower()
    for k in HIGH_SCHOOL_ROLE_KEYWORDS:
        if re.search(r"\b" + re.escape(k) + r"\b", t):
            return True
    return False


def strict_high_school_filter(persona_text: str, age: int | None) -> list[str]:
    reasons = []
    text = persona_text.lower()

    # Age rules
    if age is None:
        reasons.append("missing age")
    elif age < 14 or age >= 18:
        reasons.append("age out of range")

    # Explicit HS grounding
    if "high school" not in text and not re.search(r"\bgrade\s*(9|10|11|12)\b", text):
        reasons.append("no explicit high school mention")

    # Hard exclusions
    if any(t in text for t in ["university", "college", "degree"]):
        reasons.append("mentions university/college")

    if any(t in text for t in ["full-time job", "working full time"]):
        reasons.append("mentions full-time work")

    if any(
        t in text
        for t in [
            "when i was in high school",
            "back in high school",
            "used to be in high school",
        ]
    ):
        reasons.append("adult reflection")

    return reasons


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


def resolve_final_age(persona_text: str, llm_result: Dict[str, Any]) -> Optional[int]:
    age = llm_result.get("age", None)
    age_group = llm_result.get("age_group", "unknown")

    if isinstance(age, (int, float)) and not math.isnan(float(age)):
        age = int(age)
        if age >= 18:
            return None
        return age

    if isinstance(age_group, str):
        key = age_group.strip().lower()
        if key in AGE_GROUP_DEFAULTS:
            candidate = AGE_GROUP_DEFAULTS[key]
            if candidate >= 18:
                return None
            return candidate

    return extract_age_heuristic(persona_text)


def call_llm_for_batch(llm, sampling_params, personas: List[str]) -> List[Dict]:
    messages_list = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Persona:\n{p}\n\nReturn ONLY the JSON object.",
            },
        ]
        for p in personas
    ]

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
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
            results.append(
                {"age": None, "age_group": "unknown", "reason": "parse failure"}
            )
    return results


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


def refine_highschool_ages_with_llm(
    csv_path: str,
    persona_col: str = "persona_text",
    age_col: str = "age",
    batch_size: int = 16,
):
    print("Running LLM-based age refinement for high school personas...")

    df = pd.read_csv(csv_path)

    # Preserve original ages for auditability
    if "age_heuristic" not in df.columns:
        df["age_heuristic"] = df[age_col]

    df = fill_missing_ages_in_df(
        df,
        persona_col=persona_col,
        age_col=age_col,
        llm=llm,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )

    # Optional: rename refined age
    df.rename(columns={age_col: "age_refined"}, inplace=True)
    df["age"] = df["age_refined"]

    df.to_csv(csv_path, index=False)
    print("Age refinement complete.")


# -------------------------------------------------------------------
# V2 loaders (LLM-centric)
# -------------------------------------------------------------------
def loading_filtering_young_learners(
    min_confidence: int = 80,
    num_rows: int = 100,
) -> pd.DataFrame:
    print("V2: Loading and Filtering for PRESCHOOL / PRIMARY (LLM-centric)...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "young"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No young learner candidates found.")
        return df

    if len(df) > num_rows:
        print(
            f"V2: Limiting PRIMARY candidates to first {num_rows} rows for LLM judging."
        )
        df = df.sample(n=num_rows, random_state=42).copy()

    # Step 2: normalize text + extract age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_primary)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: Candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for PRIMARY classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_PRIMARY)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep only LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
        & df["age"].notna()
    ].copy()

    print(f"V2: Final PRIMARY personas (LLM-approved): {len(df_final)}")
    return df_final


def loading_filtering_secondary_students(
    min_confidence: int = 80,
    num_rows: int = 100,
) -> pd.DataFrame:
    print("V2: Loading and Filtering for SECONDARY SCHOOL STUDENTS...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "secondary"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No secondary candidates found.")
        return df

    # if len(df) > num_rows:
    #     print(
    #         f"V2: Limiting PRIMARY candidates to first {num_rows} rows for LLM judging."
    #     )
    #     df = df.sample(n=num_rows, random_state=42).copy()
    print(f"V2: Initial secondary domain matches: {len(df)}")

    # Step 2: normalize + age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_secondary)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: Secondary candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for SECONDARY classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_SECONDARY)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
        & df["age"].notna()
    ].copy()

    print(f"V2: Final SECONDARY personas (LLM-approved): {len(df_final)}")

    return df_final


def loading_filtering_university_students(min_confidence: int = 80) -> pd.DataFrame:
    print("V2: Loading and Filtering for UNIVERSITY STUDENTS...")
    all_rows = []

    # Step 1: domain match
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if is_relevant(rec, "university"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No university candidates found.")
        return df

    # if len(df) > num_rows:
    #         print(f"V2: Limiting PRIMARY candidates to first {num_rows} rows for LLM judging.")
    #         df = df.head(num_rows).copy()
    print(f"V2: Initial university domain matches: {len(df)}")

    # Step 2: normalize + age for info
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # Step 3: minimalist keyword exclude
    df["is_excluded"] = df["persona_text"].apply(should_exclude_university)
    df = df[~df["is_excluded"] & (df["age"] <= 18)].copy()

    print(f"V2: University candidates after keyword exclude: {len(df)}")

    if df.empty:
        return df

    # Step 4: LLM judge
    print("V2: Running LLM judge for UNIVERSITY FRESHMAN classification...")
    llm_labels, llm_conf, llm_reasons = [], [], []

    for text in df["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_UNI)
        llm_labels.append(result.get("label"))
        llm_conf.append(result.get("confidence"))
        llm_reasons.append(result.get("reason"))

    df["llm_label"] = llm_labels
    df["llm_confidence"] = llm_conf
    df["llm_reason"] = llm_reasons

    # Step 5: keep LLM-approved
    df_final = df[
        (df["llm_label"] == "yes")
        & (df["llm_confidence"].astype(float) >= min_confidence)
        & df["age"].notna()
    ].copy()

    print(f"V2: Final UNIVERSITY FRESHMEN personas (LLM-approved): {len(df_final)}")

    return df_final


def loading_filtering_highschool_students(
    min_confidence: int = 80,
    num_rows: int = 100,
    strict: bool = False,
    shard_id: int = 0,
    num_shards: int = 1,
) -> pd.DataFrame:

    print("V2: Loading and Filtering for HIGH SCHOOL STUDENTS...")
    all_rows = []

    FINAL_OUT = os.path.join(OUTPUT_FOLDER, "highschool_FINAL.csv")

    # -------- STEP 1: DOMAIN FILTER --------
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                if not is_relevant(rec, "highschool"):
                    continue

                persona_text = preferred_persona_text(rec)
                age = extract_age(persona_text)

                # ---- HARD GATE ----
                if strict:
                    violations = strict_high_school_filter(persona_text, age)
                    if violations:
                        continue

                all_rows.append(
                    {
                        "idx": i,
                        "persona_text": persona_text,
                        "age": age,
                        "source_config": config_name,
                    }
                )
        except Exception:
            pass

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("V2: No high school candidates found.")
        return df

    # if len(df) > num_rows:
    #     print(
    #         f"V2: Limiting PRIMARY candidates to first {num_rows} rows for LLM judging."
    #     )
    #     df = df.sample(n=num_rows, random_state=42).copy()
    print(f"V2: Initial high school domain matches: {len(df)}")

    # -------- STEP 2: NORMALIZE TEXT + EXTRACT AGE (for info only) --------
    df["persona_text"] = df["persona_text"].astype(str)
    df["age"] = df["persona_text"].apply(extract_age)

    # -------- STEP 3: MINIMAL KEYWORD-BASED EXCLUDE --------
    df["is_excluded"] = df["persona_text"].apply(should_exclude_highschool)
    df = df[~df["is_excluded"]].copy()

    print(f"V2: High school candidates after keyword exclude: {len(df)}")
    df["persona_id"] = df["source_config"].astype(str) + "::" + df["idx"].astype(str)
    # -------- SHARDING (AFTER HARD GATE, BEFORE LLM) --------
    if num_shards > 1:
        df = df.sort_values("idx").reset_index(drop=True)
        df = df.iloc[shard_id::num_shards].copy()
        print(f"V2: Shard {shard_id}/{num_shards} — " f"{len(df)} personas to judge")

    already_done = set()
    if os.path.exists(FINAL_HS_FILE):
        done_df = pd.read_csv(FINAL_HS_FILE, usecols=["persona_id"])
        already_done = set(done_df["persona_id"].astype(str))
        print(
            f"Shard {shard_id}: {len(already_done)} personas already completed globally"
        )

    before = len(df)
    df = df[~df["persona_id"].isin(already_done)].copy()
    skipped = before - len(df)

    if skipped > 0:
        print(f"Shard {shard_id}: skipping {skipped} already-judged personas")

    if df.empty:
        return df

    # include only the ones that have an educational context
    df = df[df["persona_text"].apply(is_role_based_high_school)]
    print(f"V2: High school candidates after school context exclude: {len(df)}")

    # -------- STEP 4: LLM JUDGE FOR HIGH SCHOOL --------
    print("V2: Running LLM judge for HIGH SCHOOL classification...")

    write_header = not os.path.exists(FINAL_OUT)

    with open(FINAL_OUT, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "persona_id",
                "idx",
                "source_config",
                "persona_text",
                "age",
                "llm_label",
                "llm_confidence",
                "llm_reason",
                "shard_id",
            ],
        )
        if write_header:
            writer.writeheader()

        for _, row in df.iterrows():
            result = judge_persona(row["persona_text"], JUDGE_SYSTEM_PROMPT_HIGHSCHOOL)

            if (
                result.get("label") == "yes"
                and float(result.get("confidence", 0)) >= min_confidence
            ):
                writer.writerow(
                    {
                        "persona_id": row["persona_id"],
                        "idx": row["idx"],
                        "source_config": row["source_config"],
                        "persona_text": row["persona_text"],
                        "age": row["age"],
                        "llm_label": result["label"],
                        "llm_confidence": result["confidence"],
                        "llm_reason": result["reason"],
                        "shard_id": shard_id,
                    }
                )

    print(f"Shard {shard_id}: appended approved personas to highschool_FINAL.csv")

    return df


# -------------------------------------------------------------------
# Age extraction
# -------------------------------------------------------------------
AGE_REGEX = re.compile(
    r"(\d{1,2})\s*-*\s*(?:year[s]?\s*[-]?\s*old|y/o|yr[s]?)",
    re.IGNORECASE,
)


# -------------------------------------------------------------------
# Global LLM + sampling_params will be initialized in init_llm()
# -------------------------------------------------------------------
llm = None
sampling_params = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter PersonaHub for educational personas using gpt-oss."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name or path for vLLM (default: openai/gpt-oss-120b).",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="high-school.csv",
        help=(
            "Output filename. Category is inferred from its name.\n"
            "Must contain one of: primary, secondary, highschool, university."
        ),
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
        default=32,
        help="Batch size for LLM judging (default: 32).",
    )
    parser.add_argument(
        "--min_confidence",
        type=int,
        default=80,
        help="Min confidence (%) to accept 'yes' judgments (default: 80).",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        default=100,
        help="Max number of calls to the LLM.",
    )
    parser.add_argument(
        "--strict-high-school",
        action="store_true",
        help="Apply hard symbolic filtering for high school personas (AMoC strict mode)",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index for SLURM array jobs (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for SLURM array jobs.",
    )

    return parser.parse_args()


# for LLaMA-style models – add tokenizer because we should use a chat template
def init_llm(model_name: str, tensor_parallel_size: int):
    global llm, sampling_params, tokenizer

    print(f"Loading model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size}\n")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def text_fields(record):
    fields = []
    for k, v in record.items():
        if isinstance(v, str):
            fields.append(v)
        elif isinstance(v, list):
            fields.extend([str(x) for x in v if isinstance(x, (str, int, float))])
        elif isinstance(v, dict):
            fields.extend([f"{kk}: {vv}" for kk, vv in v.items()])
    return fields


def is_relevant(rec, keywords):
    blob = " ".join(text_fields(rec)).lower()
    if keywords == "young":
        return any(k in blob for k in YOUNG_EDU_KEYWORDS)
    elif keywords == "secondary":
        return any(k in blob for k in SECONDARY_EDU_KEYWORDS)
    elif keywords in ("high_school", "highschool"):
        return any(k in blob for k in HIGH_SCHOOL_KEYWORDS)
    elif keywords == "university":
        return any(k in blob for k in UNIVERSITY_KEYWORDS)
    return False


def preferred_persona_text(rec):
    for cand in [
        "persona_text",
        "persona",
        "description",
        "text",
        "profile",
        "traits",
        "input persona",
    ]:
        if cand in rec and isinstance(rec[cand], str) and rec[cand].strip():
            return rec[cand].strip()
    return " ".join(text_fields(rec))[:1500]


def extract_age(text):
    if not isinstance(text, str):
        return None

    text_low = text.lower()

    match = AGE_REGEX.search(text_low)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # Heuristic proxies
    if "preschool" in text_low:
        return 3
    if "kindergarten" in text_low:
        return 5
    if "primary school" in text_low:
        return 8
    if "elementary" in text_low:
        return 8
    if "middle school" in text_low:
        return 12
    if "high school" in text_low:
        return 16
    if "university" in text_low and "student" in text_low:
        return 19
    if "first grade" in text_low or "1st grade" in text_low:
        return 6
    if "second grade" in text_low or "2nd grade" in text_low:
        return 7
    if "third grade" in text_low or "3rd grade" in text_low:
        return 8
    if "fourth grade" in text_low or "4th grade" in text_low:
        return 9
    if "fifth grade" in text_low or "5th grade" in text_low:
        return 10
    if "sixth grade" in text_low or "6th grade" in text_low:
        return 11
    if "seventh grade" in text_low or "7th grade" in text_low:
        return 12
    if "eighth grade" in text_low or "8th grade" in text_low:
        return 13
    if (
        "ninth grade" in text_low
        or "9th grade" in text_low
        or "high school freshman" in text_low
    ):
        return 14
    if "sophomore" in text_low and "high" in text_low:
        return 15
    if "junior" in text_low and "high" in text_low:
        return 16
    if "senior" in text_low and "high" in text_low:
        return 17
    if "freshman" in text_low and "university" in text_low:
        return 18
    if "freshman" in text_low and "college" in text_low:
        return 18
    if "first year" in text_low and "college" in text_low:
        return 18
    if "first year" in text_low and "undergraduate" in text_low:
        return 18
    if "first year" in text_low and "undergrad" in text_low:
        return 18
    if "senior" in text_low and "university" in text_low:
        return 22

    return None


# -------------------------------------------------------------------
# LLM judging helpers
# -------------------------------------------------------------------
def judge_persona(persona: str, system_prompt: str) -> Dict:
    if llm is None or sampling_params is None or tokenizer is None:
        raise RuntimeError("LLM/tokenizer not initialized. Call init_llm() first.")

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": f"Persona:\n{persona}\n\nAnswer ONLY in JSON.",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text.strip()

    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
    except Exception:
        data = {
            "label": "uncertain",
            "confidence": 0,
            "reason": f"Could not parse JSON: {text[:120]}",
        }
    return data


def judge_batch_120b(personas: List[str]) -> List[Dict]:
    if llm is None or sampling_params is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")

    # This assumes you have a global JUDGE_SYSTEM_PROMPT defined;
    # or you can adapt it similarly to judge_persona with a parameter.
    messages_list = [
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT_HIGHSCHOOL.strip()},
            {
                "role": "user",
                "content": f"Persona:\n{p}\n\nAnswer ONLY in JSON.",
            },
        ]
        for p in personas
    ]

    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]

    outputs = llm.generate(prompts, sampling_params)

    results = []
    # Debug print first few
    for out in outputs[:10]:
        text = out.outputs[0].text.strip()
        print("RAW LLM OUTPUT EXAMPLE:")
        print(text)

    for out in outputs:
        text = out.outputs[0].text.strip()
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            data = json.loads(text[start:end])
        except Exception:
            data = {
                "label": "uncertain",
                "confidence": 0,
                "reason": f"Could not parse JSON: {text[:120]}",
            }
        results.append(data)

    return results


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    args = parse_args()
    out_file = args.file
    min_conf = args.min_confidence
    num_rows = args.num_rows

    # Initialize LLM once
    init_llm(args.model, args.tensor_parallel_size)

    filename = os.path.basename(out_file).lower()

    if "primary" in filename:
        print("Detected CATEGORY = PRIMARY from filename.")
        df = loading_filtering_young_learners(
            min_confidence=min_conf,
            num_rows=num_rows,
        )

    elif "secondary" in filename:
        print("Detected CATEGORY = SECONDARY from filename.")
        df = loading_filtering_secondary_students(
            min_confidence=min_conf,
            num_rows=num_rows,
        )

    elif (
        "highschool" in filename
        or "high_school" in filename
        or "high-school" in filename
    ):
        print("Detected CATEGORY = HIGH SCHOOL from filename.")
        df = loading_filtering_highschool_students(
            min_confidence=min_conf,
            num_rows=num_rows,
            strict=args.strict_high_school,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
        )

    elif "university" in filename or "uni" in filename:
        print("Detected CATEGORY = UNIVERSITY FRESHMEN from filename.")
        df = loading_filtering_university_students(
            min_confidence=min_conf,
            # num_rows=num_rows,
        )

    else:
        raise ValueError(
            "Filename must contain one of: primary, secondary, highschool, university"
        )

    # Save dataframe
    if df is not None and not df.empty:
        out_path = os.path.join(
            OUTPUT_FOLDER,
            f"{out_file}_shard{args.shard_id}.csv",
        )

        # -------- HIGH SCHOOL SPECIAL CASE --------
        if "highschool" in filename and args.strict_high_school:
            # Phase 1: sharded jobs
            if args.num_shards > 1:
                print(
                    f"Shard {args.shard_id} finished. "
                    "High school personas appended to highschool_FINAL.csv."
                )
            # Phase 2: single post-processing job → refine ages -> re-run job with --num-shards 1
            else:
                print("Running FINAL age refinement for high school personas...")
                refine_highschool_ages_with_llm(
                    csv_path=FINAL_HS_FILE,
                    batch_size=args.batch_size,
                )
                print("High school age refinement complete.")
        else:
            df.to_csv(out_path, index=False)
            print(f"\nSaved {len(df)} personas to: {out_path}")

    else:
        print("\nNo personas found for this category; nothing saved.")


if __name__ == "__main__":
    main()
