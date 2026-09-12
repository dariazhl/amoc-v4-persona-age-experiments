#!/usr/bin/env python3
import argparse
from PIL import Image, ImageDraw, ImageFont

REG  = "/System/Library/Fonts/Supplemental/Arial.ttf"
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ITAL = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
MONO = "/System/Library/Fonts/Supplemental/Courier New.ttf"

PAGE, CARD, CODE = "#F2F3F6", "#FFFFFF", "#EFEFF2"
INK, MUTED, RULE = "#1B2A41", "#5A6472", "#DDE1E8"
TEXTBASED, INFERRED = "#3B6FA8", "#D99A2B"

CARDS = [
    (TEXTBASED, "Step 0 · spaCy — no model call",
     "Deterministic parse — identical for every persona.",
     ["knight, forest      young"]),
    (TEXTBASED, "Call 1 · relations among what the text states",
     "“I already extracted the nodes from the text. "
     "Generate the relationships between them.”",
     ['("knight","travels_through","forest"),', '("young","is","knight")']),
    (INFERRED, "Call 2 · what this reader would infer",
     "“Concepts coherent with the text, but not stated in it. "
     "Age and educational level should influence which edges you include.”",
     ['{"concepts": ["horse","castle"],', ' "properties": ["brave","skilled"]}']),
    (INFERRED, "Call 3 · attach the inferences",
     "“Hub-first attachment: prioritise connections to the most central "
     "explicit node.”",
     ['("knight","rides","horse"),', '("knight","is","brave")']),
]


def wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=font) <= maxw:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines


def build(width_cm, dpi, out, sentence=None, label="Sentence 1."):
    W = int(width_cm / 2.54 * dpi)
    px = W / width_cm
    def pt(p): return max(1, int(p / 72 * 2.54 * px))

    f_head  = ImageFont.truetype(BOLD, pt(16))
    f_headr = ImageFont.truetype(REG,  pt(16))
    f_title = ImageFont.truetype(BOLD, pt(17))
    f_body  = ImageFont.truetype(ITAL, pt(14))
    f_code  = ImageFont.truetype(MONO, pt(13))

    pad, gap, bar = int(px*0.55), int(px*0.35), int(px*0.13)
    lh_t, lh_b, lh_c = pt(17)*1.30, pt(14)*1.38, pt(13)*1.45
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    inner = W - 2*pad - bar

    head_h = 0
    if sentence:
        head_h = int(pt(16)*1.35 + px*0.55)

    laid, total = [], int(px*0.25) + head_h
    for colour, title, body, code in CARDS:
        tl = wrap(probe, title, f_title, inner)
        bl = wrap(probe, body, f_body, inner)
        h = int(pad*0.8 + len(tl)*lh_t + int(px*0.12) + len(bl)*lh_b
                + int(px*0.22) + len(code)*lh_c + int(px*0.30) + pad*0.8)
        laid.append((colour, tl, bl, code, h)); total += h + gap

    im = Image.new("RGB", (W, int(total)), PAGE)
    d = ImageDraw.Draw(im)

    y = int(px*0.25)
    if sentence:
        lw = d.textlength(label + " ", font=f_head)
        d.text((0, y), label, font=f_head, fill=INK)
        d.text((lw, y), sentence, font=f_headr, fill=INK)
        y += int(pt(16)*1.35 + px*0.18)
        d.line([(0, y), (W, y)], fill=RULE, width=max(1, int(px*0.02)))
        y += int(px*0.35)
    for colour, tl, bl, code, h in laid:
        d.rounded_rectangle([0, y, W-1, y+h], radius=int(px*0.20),
                            fill=CARD, outline=RULE, width=max(1, int(px*0.015)))
        d.rounded_rectangle([0, y, bar, y+h], radius=int(px*0.06), fill=colour)
        ty = y + pad*0.8
        for ln in tl:
            d.text((bar+pad, ty), ln, font=f_title, fill=INK); ty += lh_t
        ty += int(px*0.12)
        for ln in bl:
            d.text((bar+pad, ty), ln, font=f_body, fill=MUTED); ty += lh_b
        ty += int(px*0.22)
        ch = len(code)*lh_c + int(px*0.28)
        d.rounded_rectangle([bar+pad, ty, W-pad, ty+ch], radius=int(px*0.10), fill=CODE)
        cy = ty + int(px*0.14)
        for ln in code:
            d.text((bar+pad+int(px*0.30), cy), ln, font=f_code, fill=INK); cy += lh_c
        y += h + gap

    im.save(out)
    print(f"{out}  {im.width}x{im.height}  {im.width/im.height:.2f}:1  "
          f"-> {width_cm:.1f} x {im.height/px:.1f} cm")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--width-cm", type=float, default=33.5)
    a.add_argument("--dpi", type=int, default=300)
    a.add_argument("--out", default="paper/poster/images/calls_stack.png")
    a.add_argument("--sentence", default="A knight rode through the forest.")
    a.add_argument("--label", default="Sentence 1.")
    n = a.parse_args()
    build(n.width_cm, n.dpi, n.out, n.sentence, n.label)
