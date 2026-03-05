import argparse
import json
import multiprocessing
import os
import re
import pandas as pd
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import csv
import math
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import numpy as np


multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"


tokenizer = None
llm = None
sampling_params = None
OUTPUT_FOLDER = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs"

# -------------------------------------------------------------------
# Configs and Regexes
# -------------------------------------------------------------------
AGE_REGEX = re.compile(
    r"(\d{1,2})\s*-*\s*(?:year[s]?\s*[-]?\s*old|y/o|yr[s]?)",
    re.IGNORECASE,
)

PERSONAHUB_CONFIGS = [
    "persona",
    "instruction",
    "knowledge",
    "math",
    "npc",
    "reasoning",
    "tool",
]


# -------------------------------------------------------------------
# Regime configuration dataclass
# -------------------------------------------------------------------
@dataclass
class RegimeConfig:
    name: str
    relevance_key: str
    judge_prompt: str
    exclude_fn: Callable
    strict_filter_fn: Callable | None
    output_file: str | None = None


# -------------------------------------------------------------------
# Prompts
# -------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT_PRIMARY = """
You are a strict classifier.

Task: Decide whether the persona describes a primary school student.

Definitions:
- Primary school = preschool, kindergarten, or elementary/primary education, typically between ages 3 and 11.
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

Task: Decide whether the persona describes a high school student.

Definitions:
- High school = upper secondary education before university/college, typically around ages 14–18.
- The key requirement is that the PERSONA is currently attending high school (or an equivalent upper secondary level), such as "high school", "upper secondary", "gymnasium" (in some systems), etc.

Exclude:
- Primary / elementary / middle school / lower secondary students.
- University / college students.
- Teachers, parents, or other adults talking about high school students.
- Adults describing their past in high school.
- Vocational or professional training that is clearly post-secondary (unless it is explicitly described as a high school program).

If the text does not give enough information to decide, label it as "uncertain".

Decision rules:
- If the persona clearly describes a current high school / upper secondary student, label "yes".
- If the persona clearly belongs to a different educational level (primary, middle school, university, adult), label "no".
- If the age or educational level is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

JUDGE_SYSTEM_PROMPT_UNIVERSITY = """
You are a strict classifier.

Task: Decide whether the persona describes a university freshman (first-year university or college student). The persona's age must be less than or equal to 18 years old.

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
- If the persona clearly describes a current first-year university/college student less, label "yes".
- If the persona clearly belongs to a different educational level (high school, later-year university, graduate school, adult not in school), label "no".
- If the persona is an university student older than 18 years old or similarly, as inferred by the university level (ie. sophomore, junior, senior, second-year, third-year, fourth-year), label "no"
- If the educational level or year of study is unclear or ambiguous, label "uncertain".

Output format (JSON only):
{
  "label": "yes" | "no" | "uncertain",
  "confidence": <integer 0-100>,
  "reason": "<short explanation>"
}

Return ONLY this JSON object, with no additional text.
"""

# -------------------------------------------------------------------
# EXCLUSION KEYWORDS PER REGIME
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

# -------------------------------------------------------------------
# REGIME SPECIFIC KEYWORDS
# -------------------------------------------------------------------
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
    "college student": 18,
    "university student": 18,
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
    "university_student": 18,
}


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


# -------------------------------------------------------------------
# Filtering functions per regime
# -------------------------------------------------------------------
def strict_primary_filter(persona_text: str, age: int | None) -> list[str]:
    reasons = []
    text = persona_text.lower()

    # --- 1. Age sanity (PRIMARY ≈ 3–12) ---
    if age is not None:
        if age < 3:
            reasons.append("age too young (infant)")
        elif age > 11:
            reasons.append("age too old for primary")

    # --- 2. Fictional / narrative framing ---
    if any(
        kw in text
        for kw in [
            "fictional character",
            "storybook character",
            "children's storybook",
            "fairy tale",
        ]
    ):
        reasons.append("fictional character")

    # --- 3. Retrospective / historical framing ---
    if any(
        kw in text
        for kw in [
            "from the 80s",
            "in the 80s",
            "when i was a child",
            "back then",
            "used to be",
        ]
    ):
        reasons.append("retrospective framing")

    # --- 4. Not actually the student ---
    if any(
        kw in text
        for kw in [
            "my child",
            "their child",
            "his child",
            "her child",
            "someone else's child",
            "parent of",
            "attend with their toddler",
        ]
    ):
        reasons.append("not the persona themselves")

    # --- 5. Infant-only personas without school context ---
    if age is not None and age <= 2:
        if not any(
            kw in text
            for kw in [
                "preschool",
                "kindergarten",
                "nursery",
                "early education",
            ]
        ):
            reasons.append("infant without school context")

    return reasons


def strict_secondary_filter(persona_text: str, age: int | None) -> list[str]:
    reasons = []
    text = persona_text.lower()

    # --- 1. Age sanity (SECONDARY ≈ 11–14) ---
    if age is not None:
        if age < 11:
            reasons.append("age too young for secondary")
        elif age > 14:
            reasons.append("age too old for secondary")

    # --- 2. Primary school leakage ---
    if any(
        kw in text
        for kw in [
            "elementary school",
            "primary school",
            "kindergarten",
            "preschool",
            "grade school",
        ]
    ):
        reasons.append("primary school mention")

    # --- 3. High school leakage ---
    if any(
        kw in text
        for kw in [
            "high school",
            "upper secondary",
            "college prep",
            "sat",
            "act",
        ]
    ):
        reasons.append("high school mention")

    # --- 4. Retrospective framing ---
    if any(
        kw in text
        for kw in [
            "when i was in middle school",
            "back in middle school",
            "used to be in middle school",
            "former middle school student",
        ]
    ):
        reasons.append("retrospective framing")

    # --- 5. Not the student themselves ---
    if any(
        kw in text
        for kw in [
            "my child",
            "their child",
            "his child",
            "her child",
            "parent of",
        ]
    ):
        reasons.append("not the persona themselves")

    # --- 6. Fictional framing ---
    if any(
        kw in text
        for kw in [
            "fictional character",
            "storybook character",
            "novel character",
        ]
    ):
        reasons.append("fictional character")

    return reasons


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


def strict_university_filter(persona_text: str, age: int | None) -> list[str]:
    reasons = []
    text = persona_text.lower()

    # --- 1. Age sanity (UNIVERSITY freshman ≈ 16–21, intl.) ---
    if age is not None:
        if age < 15:
            reasons.append("age too young for university")
        elif age > 18:
            reasons.append("age exceeds maximum allowed for university regime")

    # --- 2. Non-freshman leakage ---
    if any(
        kw in text
        for kw in [
            "sophomore",
            "junior",
            "senior",
            "upperclassman",
            "second year",
            "third year",
            "final year",
        ]
    ):
        reasons.append("non-freshman undergraduate")

    # --- 3. Graduate / postgraduate leakage ---
    if any(
        kw in text
        for kw in [
            "graduate student",
            "postgraduate",
            "master's",
            "phd",
            "doctoral",
            "mba",
        ]
    ):
        reasons.append("graduate-level student")

    # --- 4. Alumni / retrospective framing ---
    if any(
        kw in text
        for kw in [
            "alumni",
            "former student",
            "when i was in college",
            "back in college",
            "after graduating college",
        ]
    ):
        reasons.append("retrospective framing")

    # --- 5. Not the student themselves ---
    if any(
        kw in text
        for kw in [
            "older sibling",
            "younger sibling",
            "parent of",
            "advisor to",
            "shares tips with",
            "offers advice to",
        ]
    ):
        reasons.append("not the persona themselves")

    # --- 6. Fictional / narrative framing ---
    if any(
        kw in text
        for kw in [
            "fictional character",
            "novel character",
            "story character",
        ]
    ):
        reasons.append("fictional character")

    # --- 7. Missing freshman signal entirely (soft structural guard) ---
    if not any(
        kw in text
        for kw in [
            "freshman",
            "first year",
            "first-year",
            "incoming",
            "just started university",
            "starting university",
        ]
    ):
        # Allow implicit cases only if clearly enrolled
        if not any(
            kw in text
            for kw in [
                "studying at",
                "enrolled at",
                "attends",
                "current college",
                "university student",
            ]
        ):
            reasons.append("no clear first-year university signal")

    return reasons


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


def extract_age(text: str) -> Optional[int]:
    if not isinstance(text, str):
        return None

    t = text.lower()

    m = AGE_REGEX.search(t)
    if m:
        try:
            age = int(m.group(1))
            if 0 < age < 100:
                return age
        except ValueError:
            pass

    GRADE_TO_AGE = {
        1: 6,
        2: 7,
        3: 8,
        4: 9,
        5: 10,
        6: 11,
        7: 12,
        8: 13,
        9: 14,
        10: 15,
        11: 16,
        12: 17,
    }

    grade_match = re.search(r"\b(1[0-2]|[1-9])(?:st|nd|rd|th)?\s+grade\b", t)
    if grade_match:
        grade = int(grade_match.group(1))
        return GRADE_TO_AGE.get(grade)

    ROLE_AGE_MAP = {
        # Primary
        "preschool": 3,
        "kindergarten": 5,
        # Middle school
        "middle school student": 12,
        "middle schooler": 12,
        # High school
        "high school freshman": 14,
        "high school sophomore": 15,
        "high school junior": 16,
        "high school senior": 17,
        # University (heuristic only, still refined later)
        "college freshman": 18,
        "university freshman": 18,
        "first-year college student": 18,
        "first year university student": 18,
    }

    for phrase, age in ROLE_AGE_MAP.items():
        if phrase in t:
            return age

    if "primary school" in t or "elementary school" in t:
        return 8

    if "secondary school" in t or "middle school" in t:
        return 12

    if "high school" in t:
        return 16

    if "university" in t or "college" in t:
        return 18

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


# -------------------------------------------------------------------
# Main function to load and filter per regime
# -------------------------------------------------------------------
def load_and_filter_regime(
    config: RegimeConfig,
    min_confidence: int,
    shard_id: int = 0,
    num_shards: int = 1,
) -> pd.DataFrame:

    print(f"Loading and filtering for {config.name.upper()}")

    rows = []

    # --- Domain filter ---
    for cfg in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset("proj-persona/PersonaHub", name=cfg, split="train")

            for i, rec in enumerate(ds):
                if not is_relevant(rec, config.relevance_key):
                    continue

                text = preferred_persona_text(rec)
                age = extract_age(text)

                rows.append(
                    {
                        "idx": i,
                        "persona_text": text,
                        "age": age,
                        "source_config": cfg,
                    }
                )
        except Exception:
            pass

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # --- Keyword exclude ---
    df = df[~df["persona_text"].apply(config.exclude_fn)].copy()

    # --- Strict regime filter  ---
    df["strict_reject_reasons"] = df.apply(
        lambda r: config.strict_filter_fn(r.persona_text, r.age),
        axis=1,
    )
    df = df[df["strict_reject_reasons"].str.len() == 0].copy()

    # --- Sharding (always, before LLM) ---
    if num_shards > 1:
        df = (
            df.sort_values(["source_config", "idx"])
            .reset_index(drop=True)
            .iloc[shard_id::num_shards]
            .copy()
        )
        print(
            f"Shard {shard_id}/{num_shards}: "
            f"{len(df)} personas to judge for {config.name}"
        )

    # --- LLM judge ---
    labels, confs, reasons = [], [], []
    for text in df["persona_text"]:
        r = judge_persona(text, config.judge_prompt)
        labels.append(r["label"])
        confs.append(r["confidence"])
        reasons.append(r["reason"])

    df["llm_label"] = labels
    df["llm_confidence"] = confs
    df["llm_reason"] = reasons

    return df[
        (df["llm_label"] == "yes") & (df["llm_confidence"] >= min_confidence)
    ].copy()


# -------------------------------------------------------------------
# LLM initialization
# -------------------------------------------------------------------
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
# Regime config dataclass and instances
# -------------------------------------------------------------------

REGIMES = {
    "primary": RegimeConfig(
        name="primary",
        relevance_key="young",
        judge_prompt=JUDGE_SYSTEM_PROMPT_PRIMARY,
        exclude_fn=should_exclude_primary,
        strict_filter_fn=strict_primary_filter,
    ),
    "secondary": RegimeConfig(
        name="secondary",
        relevance_key="secondary",
        judge_prompt=JUDGE_SYSTEM_PROMPT_SECONDARY,
        exclude_fn=should_exclude_secondary,
        strict_filter_fn=strict_secondary_filter,
    ),
    "highschool": RegimeConfig(
        name="highschool",
        relevance_key="highschool",
        judge_prompt=JUDGE_SYSTEM_PROMPT_HIGHSCHOOL,
        exclude_fn=should_exclude_highschool,
        strict_filter_fn=strict_high_school_filter,
    ),
    "university": RegimeConfig(
        name="university",
        relevance_key="university",
        judge_prompt=JUDGE_SYSTEM_PROMPT_UNIVERSITY,
        exclude_fn=should_exclude_university,
        strict_filter_fn=strict_university_filter,
    ),
}


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter PersonaHub for educational personas using gpt-oss."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name or path for vLLM (default: meta-llama/Llama-3.3-70B-Instruct).",
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
        "--min_confidence",
        type=int,
        default=80,
        help="Min confidence (%) to accept 'yes' judgments (default: 80).",
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


def main():
    args = parse_args()

    init_llm(args.model, args.tensor_parallel_size)

    filename = os.path.basename(args.file).lower()

    regime = None
    for key in REGIMES:
        if key in filename:
            regime = REGIMES[key]
            break
    if regime is None:
        raise ValueError("Filename must contain a valid regime")

    df = load_and_filter_regime(
        regime,
        min_confidence=args.min_confidence,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    # Restrict university student ages to <=18
    if regime.name == "university":
        df = df[(df["age"] <= 18)]

    out_path = os.path.join(OUTPUT_FOLDER, f"{regime.name}_FINAL.csv")

    # remove already existing personas if shard is re-tried
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, usecols=["persona_text"])
        df = df[~df["persona_text"].isin(existing["persona_text"])]

    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
