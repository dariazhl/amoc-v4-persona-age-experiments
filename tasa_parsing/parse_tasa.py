import re
import csv
from pathlib import Path

TOP_PARAGRAPHS = 10
SCRIPT_DIR = Path(__file__).resolve().parent

#redesign 
def drp_to_grade(drp: float) -> str:
    if drp < 60:
        return "primary"
    elif drp < 67:
        return "secondary"
    elif drp <= 74:
        return "highschool"
    else:
        return "university"


def extract_paragraphs_with_drp(text: str, category: str = None):
    blocks = re.split(r'(?=DRP="[\d.]+")', text)

    results = []

    for block in blocks:
        drp_match = re.search(r'DRP="([\d.]+)"', block)
        if not drp_match:
            continue

        if category and f'{category}="Yes"' not in block:
            continue

        drp = float(drp_match.group(1))

        cleaned = re.sub(r"<[^>]+>", "", block)
        cleaned = re.sub(r'DRP="[\d.]+"', "", cleaned)

        paragraph = " ".join(
            line.strip() for line in cleaned.splitlines() if line.strip()
        )

        paragraph = re.sub(r'^[A-Za-z]+="Yes"\s*>\s*', '', paragraph)
        paragraph = re.sub(r'\s*<ID="[^"]*".*$', '', paragraph).strip()

        if paragraph:
            results.append((drp, paragraph))

    return results


def recreate_paragraphs(text: str):
    paragraphs = []
    current = []

    for line in text.splitlines():
        stripped = line.strip()

        # Skip XML / markup lines
        if stripped.startswith("<") or not stripped:
            continue

        # New paragraph starts with leading whitespace
        if re.match(r"^\s+", line):
            if current:
                paragraphs.append(" ".join(current).strip())
            current = [stripped]
        else:
            current.append(stripped)

    if current:
        paragraphs.append(" ".join(current).strip())

    return paragraphs


if __name__ == "__main__":
    input_path = SCRIPT_DIR / "tasa.txt"
    output_path = SCRIPT_DIR / "output.csv"

    # 1. Read full file
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Extract DRP and grade level
    paragraphs_with_drp = extract_paragraphs_with_drp(raw_text, category="SocialStudies")

    if not paragraphs_with_drp:
        raise ValueError("No DRP-tagged paragraphs found")

    # 4. Write full CSV
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drp", "level", "paragraph"])

        for drp, para in paragraphs_with_drp:
            level = drp_to_grade(drp)
            writer.writerow([f"{drp:.5f}", level, para])

    print(f"Saved {len(paragraphs_with_drp)} paragraphs to {output_path}")

    # 5. Write top 3 min-DRP per group
    groups = {}
    for drp, para in paragraphs_with_drp:
        level = drp_to_grade(drp)
        groups.setdefault(level, []).append((drp, para))

    min_output_path = output_path.with_name(output_path.stem + "_min_drp_per_group.csv")
    with open(min_output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drp", "level", "paragraph"])

        for level in ["primary", "secondary", "highschool", "university"]:
            if level not in groups:
                print(f"Warning: no paragraphs found for group '{level}'")
                continue
            top3 = sorted(groups[level], key=lambda x: x[0])[:TOP_PARAGRAPHS]
            for drp, para in top3:
                writer.writerow([f"{drp:.5f}", level, para])

    print(f"Saved min-DRP representatives to {min_output_path}")
