import os
import re
import pandas as pd
from typing import Set, List

REGIME_RE = re.compile(
    r"_triplets_(primary|secondary|highschool|university)(?:_.*)?\.csv$"
)


def filter_triplets_by_persona(
    triplet_files: List[str],
    good_personas: Set[int],
    model_tag: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for path in triplet_files:
        df = pd.read_csv(path)
        before = len(df)

        df_trimmed = df[df["original_index"].isin(good_personas)].reset_index(drop=True)

        m = REGIME_RE.search(path)
        if not m:
            raise RuntimeError(f"Cannot infer regime from {path}")

        regime = m.group(1)

        out_path = os.path.join(
            out_dir,
            f"{model_tag}_triplets_{regime}_quantile_trimmed.csv",
        )

        df_trimmed.to_csv(out_path, index=False)

        print(
            f"{os.path.basename(path)}: "
            f"{len(df_trimmed)} / {before} kept "
            f"({100 * len(df_trimmed) / before:.1f}%)"
        )
