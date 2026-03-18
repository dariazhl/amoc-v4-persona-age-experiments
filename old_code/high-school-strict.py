# PATCHED VERSION — STRICT HIGH SCHOOL FILTERING FOR AMoC
# Changes:
# 1. Hard symbolic filtering BEFORE LLM validation
# 2. Strict age + education constraints
# 3. Fail-fast assertions before saving

import argparse
import json
import multiprocessing
import os
import re
from datasets import load_dataset
import pandas as pd

############################################
# HARD FILTERS (RULE-FIRST, AMoC STYLE)
############################################


def hard_filter_high_school(persona_text: str, age: int | None) -> list[str]:
    reasons = []
    text = persona_text.lower()

    # Age must exist and be in range
    if age is None:
        reasons.append("missing age")
    elif age < 14 or age >= 18:
        reasons.append("age out of range")

    # Must explicitly be high school
    if "high school" not in text and not re.search(r"\bgrade\s*(9|10|11|12)\b", text):
        reasons.append("no explicit high school mention")

    # Hard exclusions
    if any(t in text for t in ["university", "college", "degree"]):
        reasons.append("mentions university")

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


############################################
# MAIN GENERATION LOGIC (SIMPLIFIED)
############################################


def build_high_school_df():
    all_rows = []

    ds = load_dataset("proj-persona/PersonaHub", split="train")

    for rec in ds:
        persona_text = rec.get("persona_text", "")
        age = rec.get("age", None)

        # HARD FILTER — EXCLUDE EARLY
        violations = hard_filter_high_school(persona_text, age)
        if violations:
            continue

        # OPTIONAL: LLM VALIDATION CAN STILL HAPPEN HERE
        # (kept minimal — LLM should only confirm, not decide)

        all_rows.append(
            {
                "persona_text": persona_text,
                "age": age,
                "edu_regime": "high_school",
            }
        )

    df = pd.DataFrame(all_rows)

    ############################################
    # FAIL-FAST ASSERTIONS (NO SILENT LEAKAGE)
    ############################################

    assert not df.empty, "High school DF is empty after filtering"

    assert df["age"].between(14, 17).all(), "Age leakage in high school personas"

    assert (
        df["persona_text"]
        .str.contains(
            r"high school|grade\s*(9|10|11|12)",
            case=False,
            regex=True,
        )
        .all()
    ), "Non–high school personas leaked"

    return df


############################################
# ENTRY POINT
############################################

if __name__ == "__main__":
    df_hs = build_high_school_df()
    out_path = "high_school_personas_strict.csv"
    df_hs.to_csv(out_path, index=False)
    print(f"Saved strict high school personas to {out_path} ({len(df_hs)} rows)")
