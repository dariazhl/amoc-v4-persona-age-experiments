import ast
from typing import Any, Dict, List, Optional


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
