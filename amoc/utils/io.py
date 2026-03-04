import os
import json
from datetime import datetime
from typing import Dict, Any

import pandas as pd


def robust_read_persona_csv(filename: str) -> pd.DataFrame:
    short_filename = os.path.basename(filename)

    try:
        df = pd.read_csv(filename, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding="cp1252")
    except Exception:
        if filename.endswith(".parquet"):
            df = pd.read_parquet(filename)
        elif filename.endswith(".pkl"):
            df = pd.read_pickle(filename)
        else:
            raise

    if "persona" in df.columns and "persona_text" not in df.columns:
        df["persona_text"] = df["persona"]

    return df


# Build a model+file specific checkpoint path.
def get_checkpoint_path(
    output_dir: str,
    short_filename: str,
    model_name: str,
    start_index: int | None = None,
    end_index: int | None = None,
) -> str:
    safe_model_name = model_name.replace(":", "-").replace("/", "-")
    base_name = os.path.splitext(short_filename)[0]

    # Slice tag only if slicing is actually used
    if start_index is not None or end_index is not None:
        slice_tag = f"slice_{start_index or 0}_{end_index or 'end'}"
        ckpt_name = f"{base_name}__{safe_model_name}__{slice_tag}.ckpt.json"
    else:
        ckpt_name = f"{base_name}__{safe_model_name}.ckpt.json"

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    return os.path.join(ckpt_dir, ckpt_name)


# Load checkpoint JSON if it exists, otherwise return empty structure
def load_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    if not os.path.isfile(ckpt_path):
        return {
            "personas_processed": 0,
            "processed_indices": [],
            "failures": [],
            "last_update": None,
        }

    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "personas_processed": 0,
            "processed_indices": [],
            "failures": [],
            "last_update": None,
        }


# Save checkpoint JSON. Overwrites previous.
def save_checkpoint(ckpt_path: str, ckpt: Dict[str, Any]) -> None:
    ckpt["last_update"] = datetime.utcnow().isoformat()
    tmp_path = ckpt_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp_path, ckpt_path)


def infer_regime_from_filename(filename: str) -> str:
    name = filename.lower()

    if "primary" in name:
        return "primary"
    if "secondary" in name or "sec" in name:
        return "secondary"
    if "high" in name:
        return "high_school"
    if "university" in name or "freshman" in name:
        return "university"

    return "unknown"
