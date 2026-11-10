import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_for_dict(response: str) -> Optional[Dict[str, Any]]:
    if not isinstance(response, str):
        return None

    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        return ast.literal_eval(response[start : end + 1])
    except Exception:
        return None


def extract_list_from_string(response: str) -> List[Any]:
    if not isinstance(response, str):
        return []

    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        return []

    try:
        result = ast.literal_eval(response[start : end + 1])
        return result if isinstance(result, list) else []
    except Exception:
        return []


def parse_for_dict_lenient(response: str) -> Optional[Dict[str, Any]]:
    result = parse_for_dict(response)
    if result is not None or not isinstance(response, str):
        return result

    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        result = json.loads(response[start : end + 1])
    except Exception:
        return None
    return result if isinstance(result, dict) else None


_TRIPLET_DICT_KEYS = (
    ("subject", "relation", "object"),
    ("subj", "rel", "obj"),
    ("source", "relation", "target"),
    ("head", "relation", "tail"),
    ("concept1", "relation", "concept2"),
)


def coerce_triplet(item: Any) -> Optional[Tuple[str, str, str]]:
    if isinstance(item, (list, tuple)):
        if len(item) == 3 and all(isinstance(x, (str, int, float)) for x in item):
            return tuple(str(x).strip() for x in item)
        return None

    if isinstance(item, dict):
        lowered = {str(k).lower(): v for k, v in item.items()}
        for keys in _TRIPLET_DICT_KEYS:
            if all(k in lowered for k in keys):
                return coerce_triplet([lowered[k] for k in keys])
        return None

    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        try:
            literal = ast.literal_eval(text)
            if not isinstance(literal, str):
                return coerce_triplet(literal)
        except Exception:
            pass
        text = text.strip("()[]{} ")
        for sep in (r"\s*->\s*", r"\s+-\s+", r"\s*\|\s*", r"\s*,\s*"):
            parts = [p.strip(" '\"") for p in re.split(sep, text)]
            if len(parts) == 3 and all(parts):
                return tuple(parts)
        return None

    return None


def coerce_triplet_list(items: Any) -> List[Tuple[str, str, str]]:
    if not isinstance(items, (list, tuple)):
        return []
    out = []
    for item in items:
        triplet = coerce_triplet(item)
        if triplet is not None:
            out.append(triplet)
    return out
