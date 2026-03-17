import os
import re
import math
from collections import defaultdict
from PIL import Image


INPUT_ROOT = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/graphs_per_sentence/18 y:o/"

OUTPUT_DIR = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/graphs_per_sentence/comparison_across_ages"

SENTENCES = list(range(1, 16))  # sent1–sent15
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

CELL_BG_COLOR = (255, 255, 255)
DEFAULT_CELL_SIZE = (512, 512)
GRID_COLS = 5  # e.g. 5 columns x 3 rows for 15 sentences


def extract_prefix_and_sentence(filename):
    match = re.search(r"^(.*)_sent(\d+)\.\w+$", filename)
    if not match:
        return None, None

    prefix = match.group(1)
    sent_id = int(match.group(2))
    return prefix, sent_id


def collect_by_prefix(root_dir):
    groups = defaultdict(dict)

    for fname in sorted(os.listdir(root_dir)):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue

        prefix, sent_id = extract_prefix_and_sentence(fname)
        if prefix is None or sent_id not in SENTENCES:
            continue

        groups[prefix][sent_id] = os.path.join(root_dir, fname)

    return groups


def infer_cell_size(groups):
    for sent_map in groups.values():
        for path in sent_map.values():
            with Image.open(path) as img:
                return img.size
    return DEFAULT_CELL_SIZE


def create_sentence_grid(sent_map, cell_w, cell_h):
    n_sent = len(SENTENCES)
    rows = math.ceil(n_sent / GRID_COLS)

    grid = Image.new(
        "RGB",
        (GRID_COLS * cell_w, rows * cell_h),
        CELL_BG_COLOR,
    )

    for idx, sent_id in enumerate(SENTENCES):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = col * cell_w
        y = row * cell_h

        if sent_id in sent_map:
            with Image.open(sent_map[sent_id]) as img:
                img = img.resize((cell_w, cell_h))
                grid.paste(img, (x, y))

    return grid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    groups = collect_by_prefix(INPUT_ROOT)
    cell_w, cell_h = infer_cell_size(groups)

    for prefix, sent_map in groups.items():
        grid = create_sentence_grid(sent_map, cell_w, cell_h)

        safe_name = prefix.replace(" ", "_")[:200]
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_grid.png")
        grid.save(out_path)

        print(f"[✓] Saved grid for prefix: {prefix}")


if __name__ == "__main__":
    main()
