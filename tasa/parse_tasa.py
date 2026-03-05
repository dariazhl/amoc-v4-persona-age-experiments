import re
import csv


# ============================
# DRP → Grade Level Mapping
# ============================


def drp_to_grade(drp: float) -> str:
    if drp < 44:
        return "Grades K–1"
    elif 42 <= drp <= 54:
        return "Grades 2–3"
    elif 52 <= drp <= 60:
        return "Grades 4–5"
    elif 57 <= drp <= 67:
        return "Grades 6–8"
    elif 62 <= drp <= 72:
        return "Grades 9–10"
    elif 67 <= drp <= 74:
        return "Grades 11–12"
    else:
        return "College and Career Readiness (CCR)"


# ============================
# Extract DRP from input.txt
# ============================


def extract_paragraphs_with_drp(text: str):
    # Split *before* each DRP occurrence
    blocks = re.split(r'(?=DRP="[\d.]+")', text)

    results = []

    for block in blocks:
        drp_match = re.search(r'DRP="([\d.]+)"', block)
        if not drp_match:
            continue

        drp = float(drp_match.group(1))

        # Remove XML-like tags
        cleaned = re.sub(r"<[^>]+>", "", block)

        # Remove the DRP attribute itself
        cleaned = re.sub(r'DRP="[\d.]+"', "", cleaned)

        # Normalize whitespace
        paragraph = " ".join(
            line.strip() for line in cleaned.splitlines() if line.strip()
        )

        if paragraph:
            results.append((drp, paragraph))

    return results


# ============================
# Recreate paragraphs by indent
# ============================


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


# ============================
# MAIN: TXT → CSV
# ============================

if __name__ == "__main__":
    input_path = "tasa.txt"
    output_path = "output.csv"

    # 1. Read full file
    with open(input_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Extract DRP and grade level
    paragraphs_with_drp = extract_paragraphs_with_drp(raw_text)

    if not paragraphs_with_drp:
        raise ValueError("No DRP-tagged paragraphs found")

    # 4. Write CSV
    with open(output_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["drp", "level", "paragraph"])

        for drp, para in paragraphs_with_drp:
            level = drp_to_grade(drp)
            writer.writerow([f"{drp:.5f}", level, para])

    print(f"Saved {len(paragraphs_with_drp)} paragraphs to {output_path}")
