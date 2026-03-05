import os
import re
from PIL import Image

# ================= CONFIG =================

INPUT_ROOT = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/failed_graphs_jan_6/v8-to-show-for-comparison-replace-pronouns"  # folder with all images flat
OUTPUT_DIR = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/results/Qwen3-30b/failed_graphs_jan_6/v8-to-show-for-comparison-replace-pronouns/grid"

SENTENCES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

GRID_ROWS = 3
GRID_COLS = 3
MAX_IMAGES = 8

CELL_BG_COLOR = (255, 255, 255)  # white
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# =========================================


def find_images_by_sentence(root_dir):
    sentence_map = {s: [] for s in SENTENCES}

    sent_pattern = re.compile(r"sent(\d+)")

    for fname in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, fname)
        if not os.path.isfile(full_path):
            continue
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue

        match = sent_pattern.search(fname)
        if not match:
            continue

        sent_id = int(match.group(1))
        if sent_id in SENTENCES:
            sentence_map[sent_id].append(full_path)

    return sentence_map


def create_grid(images, sentence_id):
    # Load first available image to get dimensions
    sample_img = Image.open(images[0]) if images else None

    if sample_img:
        cell_w, cell_h = sample_img.size
    else:
        # fallback size if no images exist at all
        cell_w, cell_h = 512, 512

    grid_w = GRID_COLS * cell_w
    grid_h = GRID_ROWS * cell_h

    grid = Image.new("RGB", (grid_w, grid_h), CELL_BG_COLOR)

    for idx in range(MAX_IMAGES):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        x = col * cell_w
        y = row * cell_h

        if idx < len(images):
            img = Image.open(images[idx]).resize((cell_w, cell_h))
            grid.paste(img, (x, y))

    return grid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sentence_map = find_images_by_sentence(INPUT_ROOT)

    for sent_id in sorted(SENTENCES):
        images = sentence_map.get(sent_id, [])

        grid = create_grid(images, sent_id)

        out_path = os.path.join(OUTPUT_DIR, f"{sent_id}.png")

        grid.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
