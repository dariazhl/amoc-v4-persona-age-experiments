#!/usr/bin/env python3
import argparse
from PIL import Image, ImageDraw, ImageFont

BLUE, ORANGE, FADED = "#A0CBE2", "#FFE8A0", "#ECECEC"
STROKE = "#8899AA"
INK, MUTED = "#1B2A41", "#6B7583"

F_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
F_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
F_ITAL = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

LEGEND = [(BLUE, "Text-based"), (ORANGE, "Inferred"), (FADED, "Inactive / retained")]
CHIP_BG, CHIP_INK = "#E4EAF2", "#1B2A41"


def find_panels(fig):
    w, h = fig.size
    page = fig.getpixel((3, 3))

    def gutter(x):
        ys = range(int(h * 0.05), int(h * 0.95), max(1, h // 60))
        hit = sum(all(abs(fig.getpixel((x, y))[i] - page[i]) < 6 for i in range(3))
                  for y in ys)
        return hit / len(list(ys)) > 0.95

    cols = [x for x in range(0, w, 4) if gutter(x)]
    if not cols:
        return []
    runs, start, prev = [], cols[0], cols[0]
    for x in cols[1:]:
        if x - prev > 12:
            runs.append((start, prev))
            start = x
        prev = x
    runs.append((start, prev))
    runs = [r for r in runs if r[1] - r[0] > 20]
    edges = [0] + [(a + b) // 2 for a, b in runs] + [w]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
            if edges[i + 1] - edges[i] > w * 0.15]


def badge_panels(fig, labels, total, font):
    cards = find_panels(fig)
    if len(cards) != len(labels):
        print(f"  ! found {len(cards)} panels for {len(labels)} labels - skipping badges")
        return fig
    d = ImageDraw.Draw(fig)
    for (x0, x1), n in zip(cards, labels):
        txt = f"{n} / {total}"
        tw = d.textlength(txt, font=font)
        th = font.size
        padx, pady = int(th * 0.55), int(th * 0.34)
        bw, bh = tw + 2 * padx, th + 2 * pady
        bx = x1 - bw - int(th * 1.1)
        by = int(th * 0.9)
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=bh // 2, fill=CHIP_BG)
        d.text((bx + bw / 2, by + bh / 2), txt, font=font, fill=CHIP_INK, anchor="mm")
    return fig


def wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if draw.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def build(fig_path, out_path, persona, eyebrow="READER PERSONA",
          print_cm=76.1, pt_persona=26, pt_eyebrow=15, pt_legend=22,
          badges=None, total=None):
    fig = Image.open(fig_path).convert("RGB")
    W = fig.width
    bg = fig.getpixel((2, 2))

    px_cm = W / print_cm
    def px(pt):
        return max(1, int(pt / 72 * 2.54 * px_cm))

    f_eyebrow = ImageFont.truetype(F_BOLD, px(pt_eyebrow))
    f_persona = ImageFont.truetype(F_ITAL, px(pt_persona))
    f_legend = ImageFont.truetype(F_REG, px(pt_legend))

    if badges and total:
        fig = badge_panels(fig.copy(), badges, total,
                           ImageFont.truetype(F_BOLD, px(pt_legend * 0.82)))

    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = wrap(probe, persona, f_persona, W * 0.90)

    pad = int(px_cm * 0.55)
    line_h = int(px(pt_persona) * 1.35)
    head_h = pad + int(px(pt_eyebrow) * 1.9) + len(lines) * line_h + pad
    dot_r = int(px(pt_legend) * 0.42)
    leg_h = pad + int(px(pt_legend) * 1.7) + pad

    out = Image.new("RGB", (W, head_h + fig.height + leg_h), bg)
    d = ImageDraw.Draw(out)

    y = pad
    d.text((W // 2, y), " ".join(eyebrow), font=f_eyebrow, fill=MUTED, anchor="ma")
    y += int(px(pt_eyebrow) * 1.9)
    for ln in lines:
        d.text((W // 2, y), ln, font=f_persona, fill=INK, anchor="ma")
        y += line_h

    out.paste(fig, (0, head_h))

    gap, sp = int(px_cm * 0.28), int(px_cm * 1.5)
    widths = [dot_r * 2 + gap + probe.textlength(lab, font=f_legend) for _, lab in LEGEND]
    x = (W - (sum(widths) + sp * (len(LEGEND) - 1))) / 2
    cy = head_h + fig.height + pad + int(px(pt_legend) * 0.85)
    for (colour, lab), wdt in zip(LEGEND, widths):
        d.ellipse([x, cy - dot_r, x + 2 * dot_r, cy + dot_r],
                  fill=colour, outline=STROKE, width=max(1, int(px_cm * 0.02)))
        d.text((x + 2 * dot_r + gap, cy), lab, font=f_legend, fill=INK, anchor="lm")
        x += wdt + sp

    out.save(out_path)
    a = out.width / out.height
    print(f"{out_path}  {out.width}x{out.height}  {a:.2f}:1  -> {76.1/a:.1f} cm at 76 cm wide")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--persona", required=True)
    ap.add_argument("--eyebrow", default="READER PERSONA")
    ap.add_argument("--print-cm", type=float, default=76.1,
                    help="printed width of the figure on the poster (default 76.1)")
    ap.add_argument("--pt-persona", type=int, default=26)
    ap.add_argument("--pt-legend", type=int, default=22)
    ap.add_argument("--badges", help="per-panel sentence numbers, e.g. 1,2,13")
    ap.add_argument("--total", type=int, help="total sentences in the passage, e.g. 13")
    a = ap.parse_args()
    build(a.figure, a.out, a.persona, a.eyebrow,
          a.print_cm, a.pt_persona, pt_legend=a.pt_legend,
          badges=[x.strip() for x in a.badges.split(",")] if a.badges else None,
          total=a.total)
