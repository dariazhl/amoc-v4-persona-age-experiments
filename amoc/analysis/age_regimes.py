import pandas as pd
from amoc.config.constants import AGE_REGIMES

AGE_BINS = [
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (15, 16),
    (17, 18),
]


def assign_age_bin(age):
    for lo, hi in AGE_BINS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return None


def coarse_age_bin(age):
    if pd.isna(age):
        return None
    age = int(age)
    if age <= 10:
        return "3–10"
    elif age <= 14:
        return "11–14"
    else:
        return "15–18"


def assign_age_regime(age: int) -> str | None:
    for name, (lo, hi) in AGE_REGIMES.items():
        if lo <= age <= hi:
            return name
    return None
