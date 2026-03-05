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

tokenizer = None  # will init later


# -------------------------------------------------------------------
# Multiprocessing & environment setup
# -------------------------------------------------------------------
multiprocessing.set_start_method("spawn", force=True)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["HF_HOME"] = "/export/projects/nlp/.cache"

OUTPUT_DIR = "/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/personas_dfs"

PERSONAHUB_CONFIGS = [
    "persona",
    "instruction",
    "knowledge",
    "math",
    "npc",
    "reasoning",
    "tool",
]

# 3. HIGH SCHOOL KEYWORDS (Ages 14-18)
# =========================================================
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

# 1. YOUNG LEARNER KEYWORDS (Targeting Kindergarten to Primary School) - Ages 5-10 or 11
# =========================================================
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

# 2. SECONDARY SCHOOL KEYWORDS (Ages 11 - 13)
# =========================================================
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


JUDGE_SYSTEM_PROMPT = """
You are a strict classifier.

Task: Decide if the persona describes a **high school student**.

Definitions:
- High school student = high-school, ages ~14–18, pre-university.
- Exclude: university/college students, middle school, primary/elementary, working adults.
- If the text is too vague, say "uncertain".

Examples (follow these patterns):

Persona: "I'm a 16-year-old in 11th grade studying for my SATs."
Output: {"label": "yes", "confidence": 97, "reason": "16-year-old 11th grade high school student."}

Persona: "I'm a first-year computer science student at the University of Copenhagen."
Output: {"label": "no", "confidence": 99, "reason": "University student, not high school."}

Persona: "I'm a math teacher."
Output: {"label": "uncertain", "confidence": 60, "reason": "Teacher, but level not specified."}

Now classify the next persona. Return ONLY JSON with fields label, confidence, and reason.
"""

JUDGE_SYSTEM_PROMPT_PRIMARY = """
You are a strict classifier.

Task: Decide if the persona describes a primary school student.

Definitions:
- Primary school = elementary/primary, typically ages ~4–11, pre-middle school.
- Exclude: middle school, high school, university/college students, teachers, parents, working adults.
- If the text is too vague, say "uncertain".

Examples (follow these patterns):

Persona: "I'm a 7-year-old in second grade who loves math and recess."
Output: {"label": "yes", "confidence": 96, "reason": "Elementary-aged student in 2nd grade."}

Persona: "I'm a 14-year-old in 8th grade preparing for high school."
Output: {"label": "no", "confidence": 98, "reason": "Middle school, not primary."}

Persona: "I'm a math teacher."
Output: {"label": "no", "confidence": 95, "reason": "Teacher, not a primary school student."}

Persona: "I like going to school and playing games."
Output: {"label": "uncertain", "confidence": 40, "reason": "Student status and level not clearly specified."}

Now classify the next persona. Return ONLY JSON with fields label, confidence, and reason.
"""

JUDGE_SYSTEM_PROMPT_SECONDARY = """
You are a strict classifier.

Task: Decide if the persona describes a secondary school student.

Definitions:
- Secondary school = middle school + high school, typically ages ~12–18, pre-university.
- Include: middle schoolers, junior high, high school.
- Exclude: primary/elementary pupils and university/college students, teachers, parents, working adults.
- If the text is too vague, say "uncertain".

Examples:

Persona: "I'm a 13-year-old in 8th grade who loves science."
Output: {"label": "yes", "confidence": 97, "reason": "8th grade is middle school."}

Persona: "I'm a 17-year-old high school senior applying to colleges."
Output: {"label": "yes", "confidence": 99, "reason": "Clearly high school student."}

Persona: "I'm a first-year computer science student at university."
Output: {"label": "no", "confidence": 99, "reason": "University student, not secondary."}

Persona: "I'm a 9-year-old in 4th grade."
Output: {"label": "no", "confidence": 98, "reason": "Primary school age, not secondary."}

Now classify the next persona. Return ONLY JSON with fields label, confidence, and reason.
"""

JUDGE_SYSTEM_PROMPT_UNI = """
You are a strict classifier.

Task: Decide if the persona describes a university freshman (first-year university or college student).

Definitions:
- University freshman = first-year university/college student (including community college, polytechnic, etc.).
- Exclude: high school students, upperclass university students (sophomore, junior, senior, 2nd/3rd/4th year), primary/secondary students, teachers, working adults.
- If the text is too vague, say "uncertain".

Examples:

Persona: "I'm a first-year computer science student at the University of Copenhagen."
Output: {"label": "yes", "confidence": 99, "reason": "First-year university student."}

Persona: "I'm starting my freshman year at Amherst College."
Output: {"label": "yes", "confidence": 98, "reason": "Freshman at college."}

Persona: "I'm a third-year biology major at UCLA."
Output: {"label": "no", "confidence": 99, "reason": "Third-year, not freshman."}

Persona: "I'm a 17-year-old senior in high school."
Output: {"label": "no", "confidence": 99, "reason": "High school, not university."}

Now classify the next persona. Return ONLY JSON with fields label, confidence, and reason.
"""


# put near the top of the file
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
        description="Filter PersonaHub for high school students using gpt-oss."
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
        default="high-school",
        help="Choose file from: primary, seconday, high school and university.",
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
        "--high_conf_threshold",
        type=int,
        default=80,
        help="Min confidence (%) to accept 'yes' judgments (default: 80).",
    )
    return parser.parse_args()


# for LLama - add tokenizer bc we should use a chat template via tokenizer.apply_chat_template
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

    # Llama 3 tokenizer for chat template
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
    elif keywords == "high_school":
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

    # Simple, safe regex – no lookbehind
    match = AGE_REGEX.search(text_low)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    # Proxies for very young children if no number is present
    if "preschool" in text_low:
        return 3
    if "kindergarten" in text_low:
        return 5
    if "primary school" in text_low:
        return 8
    if "middle school" in text_low:
        return 12
    if "high school" in text_low:
        return 16  # Average age
    if "freshman" in text_low and "college" in text_low:
        return 18
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
    if "junior" in text_low and "high school" in text_low:
        return 16
    if "senior" in text_low and "high school" in text_low:
        return 17
    if "senior" in text_low and "university" in text_low:
        return 22

    return None


def should_exclude_non_primary_st(text):
    if not isinstance(text, str):
        return True
    text_low = text.lower()

    # Strictly exclude parents
    bad_phrases = [
        # 1. Explicit Teachers & Educators
        "Indian expatriate",
        "primary school teacher",
        "preschool teacher",
        "kindergarten teacher",
        "preschool educator",
        "primary school educator",
        "english teacher",
        "science teacher",
        "history teacher",
        "dedicated teacher",
        "fellow teacher",
        "teaching experience",
        "teenager",
        "dad",
        # 2. School Administration & Leadership
        "principal",
        "headteacher",
        "director",
        "school nurse",
        "curriculum specialist",
        # 3. Parents & Family Roles
        "mother of",
        "father of",
        "mom of",
        "dad of",
        "parent",
        "single, working mother",
        "with her 4-year old",
        "with her child",
        "with his child",
        "preparing for the primary school",  # Catches "mother of 3... preparing for..."
        # 4. Professionals & Adults Interacting with Schools
        "programmer",
        "intern",
        "artist who",
        "comedian",
        "florist",
        "gallery owner",
        "bakery owner",
        "blogger",
        "villager who only finished",  # Catches adults reminiscing
        "city dweller",
        "educator",
        "child psychologistteaching assistant",
        "high school student",
        "outgoing childowner",
        "teacher",
        "single mom",
        "single dad",
        "single mother",
        "single father",
        "owner",
        "psychologist",
        # 5. Past Tense / Reminiscing (Adults)
        "adult who",
        "grown-up",
        "retired",
        "former",
        "grew up",
        "remember",
        "fondly remembers",
        # 6. Specific Contexts from your data
        "serves at the pta",
        "consistently comes up with",
        "advocating for",
        "incorporates the exhibit",
        "uses their sibling's content",
        "offers practical strategies",
        "a primary school art teacher",
        "optimist teaching kindergarten",
        "state senator who recognizes the importance of early childhood",
        "stay-at-home mom",
    ]
    if any(p in text_low for p in bad_phrases):
        return True

    # Exclude historical ages ("who was 10")
    if re.search(r"who\s*was\s*\d+\s*year[s]?\s*old", text_low):
        return True

    return False


def should_exclude_non_sec_school_student(text):
    if not isinstance(text, str):
        return True
    text_low = text.lower()

    # adjust bad phrases
    bad_phrases = [
        # 1. Educators
        "teacher",
        "professor",
        "educator",
        "principal",
        "headmaster",
        "counselor",
        "coach",
        "instructor",
        "tutor",
        "faculty",
        "teaching",
        "curriculum",
        "classroom management",
        "pre-med student",
        "local resident",
        # 2. Parents
        "mother",
        "father",
        "mom",
        "dad",
        "parent",
        "guardian",
        "with her son",
        "with his son",
        "with her daughter",
        "with his daughter",
        "raising a",
        "has a child",
        "my child",
        # 3. Adults / Alumni / Professionals
        "former",
        "retired",
        "graduated",
        "alumni",
        "alumnus",
        "grown-up",
        "adult",
        "looking back",
        "memories of",
        "fondly remembers",
        "writer",
        "author",
        "historian",
        "journalist",
        "nurse",
        "driver",
        "owner",
        "director",
        "manager",
        "biotech researcher",
        "military academy",
        "student council",
        "Air Force Academy",
        "United States Military Academy",
        "reputed art academy",
        "JROTC",
    ]

    if any(p in text_low for p in bad_phrases):
        return True

    # Exclude historical ages ("who was 15 in 1990")
    if re.search(r"who\s*was\s*\d+\s*year[s]?\s*old", text_low):
        return True

    return False


def should_exclude_non_hs_student(text):
    if not isinstance(text, str):
        return True

    text_low = text.lower()

    is_explicitly_student = False
    student_pattern = r"^\s*(a|an|the|i\'m|i am)\s+([\w\s-]*)?\b(student|pupil|freshman|sophomore|junior|senior|attendee)\b"
    if re.search(student_pattern, text_low):
        is_explicitly_student = True
        if "senior citizen" in text_low:
            return True
        if "former" in text_low:
            return True

    hard_excludes = [
        "teacher",
        "professor",
        "principal",
        "headmaster",
        "educator",
        "counselor",
        "faculty",
        "chaperone",
        "director",
        "coordinator",
        "mother",
        "father",
        "parent",
        "guardian",
        "mom ",
        "dad ",
        "veteran",
        "elderly",
        "husband",
        "wife",
        "grown-up",
        "millennial",
        "entrepreneur",
        "plumber",
        "doctor",
        "civil engineer",
        "professional",
        "working professional",
        "reunion",
        "prom night 19",
        "in the 90s",
        "years ago",
        "looking back",
        "memories of",
        "fondly remembers",
        "stumbled upon",
        "university student",
        "college student",
        "undergraduate",
        "old friend",
        "best friend",
        "friend from",
        "friend of",
        "recent graduate",
        "graduate from",
        "graduated",
        "dropped out",
        "dropout",
        "played",
        "used to play",
        "was a",
        "discovered",
        "started in",
        "began in",
        "remembers",
        "remember",
        "memories",
        "good old days",
        "commentator",
        "athletics director",
        "football coach",
        "head coach",
        "assistant coach",
        "moved out",
        "relocated",
        "living in",
        "awaiting",
        "applying to",
        "middle school",
        "junior high",
        "primary school",
        "kindergarten",
        "sophomore student of the University",
        "fan of Sonia Citron",
        "alumn",
        "alumna",
        "alumni",
        "successful doctor",
        "Greenup Countyhigh sch",
    ]
    if any(phrase in text_low for phrase in hard_excludes):
        return True

    if re.search(r"\b(alumna|alumnus|alumni|alma mater)\b", text_low):
        if re.search(
            r"^\s*(a|an|the)\s+([\w\s]*)\b(alumna|alumnus|alumni)\b", text_low
        ):
            return True

    if re.search(r"\bgraduat(e|ed)\b", text_low):
        if re.search(r"^\s*(a|an|the)\s+([\w\s]*)\bgraduat(e|ed)\b", text_low):
            if not any(
                x in text_low
                for x in ["will graduate", "to graduate", "soon graduate", "hopes to"]
            ):
                return True

    if re.search(r"class of 19\d{2}", text_low):
        return True
    if re.search(r"class of 20[0-1]\d", text_low):
        return True

    if "working professional" in text_low or "it professional" in text_low:
        return True

    if "sweetheart" in text_low:
        if any(x in text_low for x in ["engaged", "married", "wife", "husband"]):
            return True

    if (
        "dropout" in text_low
        or "dropped out" in text_low
        or "didn't finish" in text_low
    ):
        return True

    if "slice of life" in text_low or "girls und panzer" in text_low:
        if not is_explicitly_student:
            return True

    if "old friend" in text_low or "old classmate" in text_low:
        return True

    return False


def should_exclude_non_uni_student(text):
    if not isinstance(text, str):
        return True
    text_low = text.lower()

    bad_phrases = [
        # 1. Faculty & Staff (Expanded)
        "professor",
        "lecturer",
        "instructor",
        "faculty",
        "dean",
        "chancellor",
        "researcher",
        "fellow",
        "academic advisor",
        "teaching",
        "tenure",
        "adjunct",
        "visiting scholar",
        "administrator",
        "coordinator",
        "technician",
        "librarian",
        "counselor",
        "analyst",
        "specialist",
        "staff",
        "officer",
        # 2. Mentors & Guides (New Category)
        "guiding the",
        "mentoring",
        "advising",
        "assists the",
        "guides and trains",
        "offers advice",
        # 3. Family (Expanded)
        "mother",
        "father",
        "mom",
        "dad",
        "parent",
        "sibling",
        "brother",
        "sister",
        "with her son",
        "with his son",
        "with her daughter",
        "supporting my child",
        "paying tuition",
        # 4. Alumni / Past Tense (Expanded)
        "alumni",
        "alumnus",
        "alumna",
        "graduate of",
        "graduated",
        "former student",
        "was an undergrad",
        "was a student",
        "looking back",
        "memories of",
        "career in",
        "working as",
        "professional",
        "expert",
        "math problem",
        "-shot instruction data synthesis",
        "high school freshman",
        "high school sophomore",
        "high school junior",
        "high school senior",
        "sophomore in high school",
        "freshman in high school",
        "freshman high",
        "sophomore high",
        "junior high",
        "senior high",
        "high school freshman",
        "high school sophomore",
        "high school junior",
        "high school senior",
        "high school student",
        "high schooler",
        "junior in high school",
        "senior in high school",
        "primary school",
        "middle school",
        "kindergartenhigh school years",
    ]

    if any(p in text_low for p in bad_phrases):
        return True

    # Exclude historical ages ("who was 20 in 1990")
    if re.search(r"who\s*was\s*\d+\s*year[s]?\s*old", text_low):
        return True

    return False


def should_keep_uni_student(text):
    text_low = text.lower()
    good_phrases = [
        "high school graduate",
        "A college student studying abroad from the country the high school student wants to learn the language of",
        "A talented high school athlete with aspirations of playing at the college level, seeking guidance on academic eligibility requirements",
    ]
    if any(p in text_low for p in good_phrases):
        return False
    return True


# =========================================================
# 3. DATA LOADING & FILTERING
# =========================================================


def loading_filtering_young_learners(min_confidence: int = 70) -> pd.DataFrame:
    print("Loading and Filtering for PRESCHOOL, KINDERGARDEN AND PRI...")
    all_rows = []

    # Iterate through configs to find candidates
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                # Step 1: Domain Filter (Young Education Keywords)
                if is_relevant(rec, "young"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception as e:
            # You might log this if needed
            pass  # Skip errors for cleaner output

    df_young = pd.DataFrame(all_rows)

    if df_young.empty:
        print("No young learner personas found.")
        return df_young

    # Step 2: Extract Age
    df_young["persona_text"] = df_young["persona_text"].astype(str)
    df_young["age"] = df_young["persona_text"].apply(extract_age)

    # Step 3: Exclude Parents / Non-students
    df_young["is_excluded"] = df_young["persona_text"].apply(
        should_exclude_non_primary_st
    )

    # Step 4: Strict Age Limit (Max 11)
    # Age <= 11 AND Valid Age AND Not Excluded
    df_final = df_young[
        (df_young["age"].notna()) & (df_young["age"] <= 11) & (~df_young["is_excluded"])
    ].copy()

    df_final = df_final.sort_values(by="age")

    print(f"\nFound {len(df_final)} Young Learner Personas (Age <= 11) before LLM.")

    if df_final.empty:
        return df_final

    # Step 5: LLM judging step (primary-school classifier)
    print("Running LLM judge for PRIMARY classification...")
    labels = []
    confidences = []
    reasons = []

    for text in df_final["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_PRIMARY)
        labels.append(result.get("label"))
        confidences.append(result.get("confidence"))
        reasons.append(result.get("reason"))

    df_final["llm_label"] = labels
    df_final["llm_confidence"] = confidences
    df_final["llm_reason"] = reasons

    # Step 6: Keep only those judged as 'yes' with sufficient confidence
    df_primary_llm = df_final[
        (df_final["llm_label"] == "yes")
        & (df_final["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"After LLM judge: {len(df_primary_llm)} PRIMARY-like personas kept.\n")

    if not df_primary_llm.empty:
        print("\n--- Sample Primary / Young Learners ---")
        print(df_primary_llm[["age", "llm_confidence", "persona_text"]].head())

        # Optional: Save
        # output_file = os.path.join(OUT_DIR, "primary_school_students_llm.csv")
        # df_primary_llm.to_csv(output_file, index=False)
        # print(f"\nSaved to {output_file}")

    return df_primary_llm


def loading_filtering_secondary_students(min_confidence: int = 70) -> pd.DataFrame:
    print("Loading and Filtering for SECONDARY SCHOOL STUDENTS...")
    all_rows = []

    # Iterate through configs
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub", name=config_name, split="train"
            )
            for i in range(len(ds)):
                rec = ds[i]
                # Step 1: Domain Filter (Secondary Education)
                if is_relevant(rec, "secondary"):
                    all_rows.append(
                        {
                            "idx": i,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception as e:
            pass

    df_secondary = pd.DataFrame(all_rows)

    if df_secondary.empty:
        print("No secondary school personas found.")
        return df_secondary

    print(f"Initial domain match: {len(df_secondary)} personas.")

    # Step 2: Extract Age
    df_secondary["persona_text"] = df_secondary["persona_text"].astype(str)
    df_secondary["age"] = df_secondary["persona_text"].apply(extract_age)

    # Step 3: Exclude Non-Students
    df_secondary["is_excluded"] = df_secondary["persona_text"].apply(
        should_exclude_non_sec_school_student
    )

    # Step 4: Filter for Age Range
    # (you had 11–14; adjust if you want exactly 12–19 or similar)
    df_final = df_secondary[
        (df_secondary["age"].notna())
        & (df_secondary["age"] >= 11)
        & (df_secondary["age"] <= 14)
        & (~df_secondary["is_excluded"])
    ].copy()

    df_final = df_final.sort_values(by="age")

    print(
        f"\nFound {len(df_final)} Secondary School Student Personas (Age 11-14) before LLM."
    )

    if df_final.empty:
        return df_final

    # Step 5: LLM judging step (secondary-school classifier)
    print("Running LLM judge for SECONDARY classification...")
    labels = []
    confidences = []
    reasons = []

    for text in df_final["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_SECONDARY)
        labels.append(result.get("label"))
        confidences.append(result.get("confidence"))
        reasons.append(result.get("reason"))

    df_final["llm_label"] = labels
    df_final["llm_confidence"] = confidences
    df_final["llm_reason"] = reasons

    # Step 6: Keep only those judged as 'yes' with sufficient confidence
    df_secondary_llm = df_final[
        (df_final["llm_label"] == "yes")
        & (df_final["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"After LLM judge: {len(df_secondary_llm)} SECONDARY personas kept.\n")

    if not df_secondary_llm.empty:
        print("\n--- Sample Secondary Students ---")
        print(df_secondary_llm[["age", "llm_confidence", "persona_text"]].head())

        # Optional: Save
        # output_file = os.path.join(OUT_DIR, "secondary_students_llm.csv")
        # df_secondary_llm.to_csv(output_file, index=False)
        # print(f"\nSaved to {output_file}")

    return df_secondary_llm


def loading_filtering_university_students(min_confidence: int = 70) -> pd.DataFrame:
    print("Loading and Filtering for UNIVERSITY STUDENTS...")
    all_rows = []

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
        except Exception as e:
            pass

    df_uni = pd.DataFrame(all_rows)

    if df_uni.empty:
        print("No university personas found.")
        return df_uni

    print(f"Initial domain match: {len(df_uni)} personas.")

    # Step 2: Extract Age
    df_uni["persona_text"] = df_uni["persona_text"].astype(str)
    df_uni["age"] = df_uni["persona_text"].apply(extract_age)

    # Step 3: Exclude clear non-students
    df_uni["is_excluded"] = df_uni["persona_text"].apply(should_exclude_non_uni_student)
    df_uni["to_keep"] = df_uni["persona_text"].apply(should_keep_uni_student)

    # Step 4: Filter for Age Range (17–18 as proxy for first-year)
    df_final = df_uni[
        (df_uni["age"].notna())
        & (df_uni["age"] >= 17)
        & (df_uni["age"] <= 18)
        & (~df_uni["is_excluded"])
        & (df_uni["to_keep"])
    ].copy()

    df_final = df_final.sort_values(by="age")

    print(
        f"\nFound {len(df_final)} University Student Personas (Age 17-18) before LLM."
    )

    if df_final.empty:
        return df_final

    # Step 5: LLM judging step (university freshman classifier)
    print("Running LLM judge for UNIVERSITY FRESHMAN classification...")
    labels = []
    confidences = []
    reasons = []

    for text in df_final["persona_text"]:
        result = judge_persona(text, JUDGE_SYSTEM_PROMPT_UNI)
        labels.append(result.get("label"))
        confidences.append(result.get("confidence"))
        reasons.append(result.get("reason"))

    df_final["llm_label"] = labels
    df_final["llm_confidence"] = confidences
    df_final["llm_reason"] = reasons

    # Step 6: Keep only those judged as 'yes' with sufficient confidence
    df_uni_llm = df_final[
        (df_final["llm_label"] == "yes")
        & (df_final["llm_confidence"].astype(float) >= min_confidence)
    ].copy()

    print(f"After LLM judge: {len(df_uni_llm)} UNIVERSITY FRESHMEN personas kept.\n")

    if not df_uni_llm.empty:
        print("\n--- Sample University Freshmen ---")
        print(df_uni_llm[["age", "llm_confidence", "persona_text"]].head())

        # Optional: Save
        # output_file = os.path.join(OUT_DIR, "university_freshmen_llm.csv")
        # df_uni_llm.to_csv(output_file, index=False)
        # print(f"\nSaved to {output_file}")

    return df_uni_llm


def build_prompt(persona: str) -> str:
    if tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call init_llm() first.")

    messages = [
        {
            "role": "system",
            "content": JUDGE_SYSTEM_PROMPT.strip(),
        },
        {
            "role": "user",
            "content": f"Persona:\n{persona}\n\nAnswer ONLY in JSON.",
        },
    ]

    # Let the tokenizer apply the correct chat template for Llama 3
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # add assistant header
    )
    return prompt
    # return JUDGE_SYSTEM_PROMPT.strip() + "\n\nPersona:\n" + persona + "\n\nAnswer:"


def judge_batch_120b(personas: List[str]) -> List[Dict]:
    if llm is None or sampling_params is None:
        raise RuntimeError("LLM not initialized. Call init_llm() first.")

    prompts = [build_prompt(p) for p in personas]
    outputs = llm.generate(prompts, sampling_params)

    results = []
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


def loading_filtering_high_school_students(
    batch_size: int = 32,
    high_conf_threshold: int = 80,
):
    print("Loading and Filtering for HIGH SCHOOL STUDENTS...")
    all_rows = []

    # 1) Load & domain filter (cheap)
    for config_name in PERSONAHUB_CONFIGS:
        try:
            ds = load_dataset(
                "proj-persona/PersonaHub",
                name=config_name,
                split="train",
            )

            for idx, rec in enumerate(ds):
                if is_relevant(rec, "high_school"):
                    all_rows.append(
                        {
                            "idx": idx,
                            "persona_text": preferred_persona_text(rec),
                            "source_config": config_name,
                        }
                    )
        except Exception as e:
            print(f"[WARN] Failed to load config {config_name}: {e}")
            continue

    df_high_school = pd.DataFrame(all_rows)

    if df_high_school.empty:
        print("No high school personas found at domain-filter level.")
        return None

    df_high_school["persona_text"] = df_high_school["persona_text"].astype(str)

    # 2) Cheap filters (age + non-student) BEFORE LLM
    df_high_school["age"] = df_high_school["persona_text"].apply(extract_age)
    df_high_school["is_excluded"] = df_high_school["persona_text"].apply(
        should_exclude_non_hs_student
    )

    mask_age = (
        df_high_school["age"].notna()
        & (df_high_school["age"] >= 14)
        & (df_high_school["age"] <= 18)
    )
    mask_not_excluded = ~df_high_school["is_excluded"]

    df_candidates = df_high_school[mask_age & mask_not_excluded].copy()

    if df_candidates.empty:
        print("No candidates in age 14–18 after cheap filters.")
        return None

    print(f"Running LLM judge on {len(df_candidates)} candidates...")

    # 3) Batched LLM judging
    personas = df_candidates["persona_text"].tolist()
    judgments = []

    for i in range(0, len(personas), batch_size):
        batch = personas[i : i + batch_size]
        batch_results = judge_batch_120b(batch)
        judgments.extend(batch_results)

    assert len(judgments) == len(df_candidates)

    df_candidates["hs_label"] = [j["label"] for j in judgments]
    df_candidates["hs_confidence"] = [j["confidence"] for j in judgments]
    df_candidates["hs_reason"] = [j["reason"] for j in judgments]

    print("Label distribution:")
    print(df_candidates["hs_label"].value_counts(dropna=False))

    print("\nConfidence stats:")
    print(df_candidates["hs_confidence"].describe())

    print("\nA few 'yes' rows regardless of confidence:")
    print(
        df_candidates[df_candidates["hs_label"] == "yes"][
            ["age", "hs_confidence", "persona_text"]
        ].head(10)
    )

    print("\nA few 'uncertain' rows:")
    print(
        df_candidates[df_candidates["hs_label"] == "uncertain"][
            ["age", "hs_confidence", "persona_text"]
        ].head(10)
    )

    mask_llm = df_candidates["hs_label"] == "yes"
    # & (df_candidates["hs_confidence"] >= high_conf_threshold)

    df_final = df_candidates[mask_llm].copy()

    if df_final.empty:
        print("No high school student personas passed the LLM filter.")
        return None

    df_final = df_final.sort_values(by="age")

    print(f"\nFound {len(df_final)} High School Student Personas (Age 14–18).")
    print("\n--- Sample High School Students ---")
    print(df_final[["age", "persona_text"]].head(20))

    return df_final


def main():
    args = parse_args()
    init_llm(args.model, args.tensor_parallel_size)

    file_name = args.file
    if file_name == "high-school":
        df_final = loading_filtering_high_school_students(
            batch_size=args.batch_size,
            high_conf_threshold=args.high_conf_threshold,
        )
        if df_final is None:
            print("\nNo data to save.")
            return

        output_file = os.path.join(OUTPUT_DIR, "high_school_students.csv")
        df_final.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    elif file_name == "primary":
        df_primary = loading_filtering_young_learners(min_confidence=80)
        if df_primary is None:
            print("\nNo data to save.")
            return
        output_file = os.path.join(OUTPUT_DIR, "primary_students.csv")
        df_primary.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    elif file_name == "university":
        df_uni = loading_filtering_university_students(min_confidence=80)
        if df_uni is None:
            print("\nNo data to save.")
            return
        output_file = os.path.join(OUTPUT_DIR, "university_students.csv")
        df_uni.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")

    elif file_name == "secondary":
        df_secondary = loading_filtering_secondary_students(min_confidence=70)
        if df_secondary is None:
            print("\nNo data to save.")
            return
        output_file = os.path.join(OUTPUT_DIR, "secondary_students.csv")
        df_secondary.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
