from typing import List
import pandas as pd


def quantile_trim(
    df: pd.DataFrame,
    value_cols: List[str],
    lower_q: float,
    upper_q: float,
) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for col in value_cols:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        keep &= (df[col] >= lo) & (df[col] <= hi)
    return df[keep].reset_index(drop=True)


def iqr_cap(
    df: pd.DataFrame,
    value_cols: List[str],
    k: float = 1.5,
) -> pd.DataFrame:
    df = df.copy()
    for col in value_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + k * iqr
        df[col] = df[col].clip(upper=upper)
    return df
