import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

LABELS = {
    "a1": "Primary\nschool",
    "a2": "Secondary\nschool",
    "a3": "High\nschool",
    "a4": "University",
}

BLUE   = "#9ec6e8"
YELLOW = "#f5dd9a"
GREY   = "#d9d9d9"


def autotrim(im, bg_thresh: int = 245):
    arr = np.asarray(im.convert("RGB"))
    mask = (arr < bg_thresh).any(axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def load_cell(path: str, crop_top: float, crop_right: float):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((0, int(h * crop_top), int(w * (1 - crop_right)), h))
    return autotrim(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", required=True, help="top-left:  Primary school")
    ap.add_argument("--a2", required=True, help="top-right: Secondary")
    ap.add_argument("--a3", required=True, help="bot-left:  High school")
    ap.add_argument("--a4", required=True, help="bot-right: University")
    ap.add_argument("--out",        default="grid.png")
    ap.add_argument("--title",      default="")
    ap.add_argument("--crop-top",   type=float, default=0.14)
    ap.add_argument("--crop-right", type=float, default=0.14)
    ap.add_argument("--cell-in",    type=float, default=4.6)
    args = ap.parse_args()

    cells = [["a1", "a2"], ["a3", "a4"]]
    paths = {"a1": args.a1, "a2": args.a2, "a3": args.a3, "a4": args.a4}

    imgs = {k: load_cell(p, args.crop_top, args.crop_right) for k, p in paths.items()}
    aspects = [im.size[0] / im.size[1] for im in imgs.values()]
    cell_aspect = sum(aspects) / len(aspects)
    cell_h = args.cell_in / cell_aspect

    panel_title_in = 0.85
    suptitle_in = 0.55 if args.title else 0.0
    legend_in = 0.55

    fig_h = cell_h * 2 + panel_title_in + suptitle_in + legend_in
    fig_w = args.cell_in * 2

    fig, axes = plt.subplots(2, 2,
                             figsize=(fig_w, fig_h),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.06})

    for row, cell_row in enumerate(cells):
        for col, key in enumerate(cell_row):
            ax = axes[row, col]
            ax.axis("off")
            img = imgs[key]
            ax.imshow(img, aspect="auto")
            w, h = img.size
            ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h, fill=False,
                                        edgecolor="0.75", linewidth=1.1,
                                        clip_on=False))
            ax.set_title(LABELS[key], fontsize=15, pad=7)

    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=c, markeredgecolor="0.4",
               markersize=11, label=lbl)
        for c, lbl in [(BLUE, "text-based"), (YELLOW, "inferred"), (GREY, "inactive")]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handletextpad=0.4, columnspacing=1.8)

    top_frac = 1 - (panel_title_in + suptitle_in) / fig_h
    bottom_frac = legend_in / fig_h
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac)
    if args.title:
        suptitle_y = 1 - (suptitle_in * 0.45) / fig_h
        fig.suptitle(args.title, fontsize=17, y=suptitle_y)

    fig.savefig(args.out, dpi=250, bbox_inches="tight")
    print(f"saved → {args.out}")


if __name__ == "__main__":
    main()
