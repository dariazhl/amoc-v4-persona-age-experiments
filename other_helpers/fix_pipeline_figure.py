#!/usr/bin/env python3
import argparse
from PIL import Image, ImageDraw, ImageChops

CARD_X       = (40, 1310)
CARD_TITLE_B = 1256
CARD_BORDER  = 1461
CARD_NEWBORD = 1310
CLEAN_WHITE  = (1160, 1195)

SUB_TOP, SUB_BOT = 168, 330
PANEL_BOT        = 2540


def render(pdf, page, dpi, out):
    import subprocess, tempfile, os, glob
    d = tempfile.mkdtemp()
    subprocess.run(["pdftocairo", "-png", "-r", str(dpi), "-f", str(page),
                    "-l", str(page), pdf, f"{d}/s"], check=True)
    im = Image.open(glob.glob(f"{d}/s*.png")[0]).convert("RGB")
    w, h = im.size
    c = im.crop((0, int(h * 0.10), w, int(h * 0.95)))
    bg = Image.new("RGB", c.size, c.getpixel((4, 4)))
    b = ImageChops.difference(c, bg).convert("L").point(
        lambda v: 255 if v > 12 else 0).getbbox()
    p = 40
    c.crop((max(0, b[0] - p), max(0, b[1] - p),
            min(c.size[0], b[2] + p), min(c.size[1], b[3] + p))).save(out)


DETECT_Y = 2350


def panels(im, detect_y=None):
    W, H = im.size
    px = im.load()
    page = px[5, H - 5]

    def runs_at(y):
        out, cur = [], None
        for x in range(W):
            p = px[x, y]
            empty = all(abs(p[i] - page[i]) < 7 for i in range(3))
            if not empty and cur is None:
                cur = x
            if empty and cur is not None:
                if x - cur > 150:
                    out.append((cur, x))
                cur = None
        if cur is not None and W - cur > 150:
            out.append((cur, W))
        return out

    if detect_y is not None:
        return runs_at(detect_y)
    for y in range(int(H * 0.80), int(H * 0.30), -10):
        r = runs_at(y)
        if len(r) == 4:
            return r
    return []


def trim_text_card(im):
    XL, XR = CARD_X
    interior = im.crop((XL, *CLEAN_WHITE[:1], XR, CLEAN_WHITE[1])) \
        if False else im.crop((XL, CLEAN_WHITE[0], XR, CLEAN_WHITE[1]))
    bottom = im.crop((XL, CARD_BORDER - 26, XR, CARD_BORDER + 19))
    panel = im.getpixel((200, 1560))
    d = ImageDraw.Draw(im)
    d.rectangle([XL, CARD_TITLE_B, XR, CARD_BORDER + 25], fill=panel)
    for y in range(CARD_TITLE_B, CARD_NEWBORD - 26, interior.height):
        im.paste(interior, (XL, y))
    im.paste(bottom, (XL, CARD_NEWBORD - 26))
    return im


def drop_subtitles(im):
    px = im.load()
    d = ImageDraw.Draw(im)
    for a, b in panels(im):
        bg = px[a + 14 if a > 0 else 8, DETECT_Y]
        body = im.crop((a, SUB_BOT, b, PANEL_BOT))
        d.rectangle([a, SUB_TOP, b, PANEL_BOT], fill=bg)
        im.paste(body, (a, SUB_TOP))
    return im


def even_margins(im):
    W, H = im.size
    px = im.load()
    page = px[5, H - 5]
    cols = panels(im)
    inner = _card_x(px, *cols[1])[0] - cols[1][0]
    dl = max(0, inner - (_card_x(px, *cols[0])[0] - cols[0][0]))
    dr = max(0, inner - (cols[-1][1] - _card_x(px, *cols[-1])[1]))
    if not (dl or dr):
        return im

    def vspan(a, b):
        x = a + 14 if a > 0 else 8
        bg = px[x, 300]
        ys = [y for y in range(H)
              if all(abs(px[x, y][i] - bg[i]) < 8 for i in range(3))]
        return bg, min(ys), max(ys)

    out = Image.new("RGB", (W + dl + dr, H), page)
    out.paste(im, (dl, 0))
    d = ImageDraw.Draw(out)
    if dl:
        bg, t, bot = vspan(*cols[0])
        d.rectangle([0, t, dl, bot], fill=bg)
    if dr:
        bg, t, bot = vspan(*cols[-1])
        d.rectangle([W + dl - 1, t, W + dl + dr, bot], fill=bg)
    return out


def center_arrows(im, row=1):
    W, H = im.size
    px = im.load()
    page = px[5, H - 5]
    d = ImageDraw.Draw(im)
    cols = panels(im)
    cx = _card_x(px, *cols[0])
    y0, y1 = _cards(px, cx, H)[row]
    target = (y0 + y1) // 2
    x = cols[0][0] + 8
    bg = px[x, 300]
    pbot = max(y for y in range(H)
               if all(abs(px[x, y][i] - bg[i]) < 8 for i in range(3)))
    for (_, b0), (a1, _) in zip(cols, cols[1:]):
        ink = [(xx, yy) for yy in range(0, pbot) for xx in range(b0, a1, 2)
               if sum(px[xx, yy]) < 450]
        if not ink:
            continue
        xs = [xx for xx, _ in ink]; ys = [yy for _, yy in ink]
        ax0, ax1, ay0, ay1 = min(xs) - 6, max(xs) + 8, min(ys) - 6, max(ys) + 8
        glyph = im.crop((ax0, ay0, ax1, ay1))
        d.rectangle([ax0, ay0, ax1, ay1], fill=page)
        im.paste(glyph, (ax0, target - glyph.height // 2))
    return im


def shorten(im, pad=127, loop_extra=110, strip_h=45):
    W, H = im.size
    px = im.load()
    page = px[5, H - 5]
    d = ImageDraw.Draw(im)
    cols = panels(im)

    deepest = 0
    for a, b in cols:
        cx = _card_x(px, a, b)
        for y in range(H - 1, 0, -1):
            if any(px[x, y] == (255, 255, 255) for x in range(cx[0] + 20, cx[1] - 20, 6)):
                deepest = max(deepest, y)
                break
    new_bot = deepest + pad

    old_bot = None
    for a, b in cols:
        x = a + 14 if a > 0 else 8
        bg = px[x, 300]
        old_bot = max(y for y in range(H)
                      if all(abs(px[x, y][i] - bg[i]) < 8 for i in range(3)))
        strip = im.crop((a, old_bot - strip_h, b, old_bot + 18))
        d.rectangle([a, new_bot - strip_h, b, old_bot + 20], fill=page)
        im.paste(strip, (a, new_bot - strip_h))

    shift = (old_bot - new_bot) + loop_extra
    loop = im.crop((0, old_bot + 20, W, H))
    d.rectangle([0, old_bot + 20 - shift, W, H], fill=page)
    im.paste(loop, (0, old_bot + 20 - shift))
    return im.crop((0, 0, W, H - shift))


BOLD_LUM = 75


def _card_x(px, a, b):
    for y in range(150, 700, 10):
        runs, run = [], None
        for x in range(a, b):
            p = px[x, y]
            white = p[0] > 248 and p[1] > 248 and p[2] > 248
            if white and run is None:
                run = x
            if not white and run is not None:
                runs.append((run, x)); run = None
        if run is not None:
            runs.append((run, b))
        if runs:
            lo, hi = min(r[0] for r in runs), max(r[1] for r in runs)
            if hi - lo > (b - a) * 0.7:
                return lo, hi
    return None


def _cards(px, cx, H):
    out, cur = [], None
    for y in range(H):
        p = px[cx[0] + 12, y]
        white = p[0] > 248 and p[1] > 248 and p[2] > 248
        if white and cur is None:
            cur = y
        if not white and cur is not None:
            if y - cur > 60:
                out.append((cur, y))
            cur = None
    return out


def _title_end(px, cx, y0, y1):
    last_bold = None
    for y in range(y0, y1):
        row = [sum(px[x, y]) / 3 for x in range(cx[0] + 160, cx[1] - 20, 4)
               if sum(px[x, y]) / 3 < 200]
        if row and sum(row) / len(row) < BOLD_LUM:
            last_bold = y

    circle = [(x, y) for y in range(y0, y1)
              for x in range(cx[0] + 10, cx[0] + 400, 3)
              if max(px[x, y]) - min(px[x, y]) > 50]
    if not circle:
        return last_bold, None, None
    return last_bold, max(y for _, y in circle), max(x for x, _ in circle)


def drop_card_bodies(im, card_h=238, top_y=229, gap=120):
    W, H = im.size
    px = im.load()
    d = ImageDraw.Draw(im)
    for a, b in panels(im):
        cx = _card_x(px, a, b)
        if not cx:
            continue
        cards = _cards(px, cx, H)
        bg = px[a + 14 if a > 0 else 8, cards[0][0] + 20]
        cuts = []
        for y0, y1 in cards:
            text_end, circ_end, circ_x = _title_end(px, cx, y0, y1)
            if text_end is None:
                text_end = y1 - 58
            x0 = (circ_x + 12) if circ_x else (cx[0] + 20)
            d.rectangle([x0, text_end + 8, cx[1] - 6, y1 - 2],
                        fill=(255, 255, 255))
            cuts.append(max(text_end, circ_end or 0) + 6)

        FOOT, TOP_PAD = 22, 53
        pieces, gaps = [], []
        for i, ((y0, y1), cut) in enumerate(zip(cards, cuts)):
            tstart = next((y for y in range(y0, y1)
                           if any(sum(px[x, y]) / 3 < 120
                                  for x in range(cx[0] + 160, cx[1] - 20, 4))), y0)
            short = TOP_PAD - (tstart - y0)
            head = im.crop((a, y0 - 8, b, cut))
            if short > 0:
                pad_src = im.crop((a, y0 + 2, b, y0 + 2 + short))
                grown = Image.new("RGB", (head.width, head.height + short))
                grown.paste(head.crop((0, 0, head.width, 10)), (0, 0))
                grown.paste(pad_src, (0, 10))
                grown.paste(head.crop((0, 10, head.width, head.height)), (0, 10 + short))
                head = grown
            foot = im.crop((a, y1 - (FOOT - 8), b, y1 + 8))
            fill = im.crop((a, y0 + 10, b, y0 + 30))
            need = card_h - head.height - foot.height
            if need < 0:
                print(f"  ! card at y{y0} needs {-need}px more than card_h={card_h}")
                need = 0
            pieces.append((head, fill, need, foot))
        arrow = None
        if len(cards) > 1:
            strip = im.crop((a, cards[0][1] + 8, b, cards[1][0] - 8))
            if any(sum(strip.getpixel((x, y))) < 620
                   for y in range(strip.height) for x in range(0, strip.width, 7)):
                arrow = strip

        panel_bot = max(y for y in range(H)
                        if all(abs(px[a + 14 if a > 0 else 8, y][i] - bg[i]) < 8
                               for i in range(3)))
        d.rectangle([a, min(top_y, cards[0][0] - 8), b, panel_bot], fill=bg)
        for i, (head, fill, need, foot) in enumerate(pieces):
            y = top_y + i * (card_h + gap)
            if i and arrow is not None:
                im.paste(arrow, (a, y - gap + (gap - arrow.height) // 2))
            im.paste(head, (a, y)); y += head.height
            done = 0
            while done < need:
                h = min(fill.height, need - done)
                im.paste(fill.crop((0, 0, fill.width, h)), (a, y + done))
                done += h
            y += need
            im.paste(foot, (a, y))
    return im


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--page", type=int, default=5)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="paper/poster/images/fig_methods_slide5.png")
    ap.add_argument("--card-h", type=int, default=238,
                    help="uniform card height in px")
    ap.add_argument("--top-y", type=int, default=229,
                    help="y of the first card in every column")
    ap.add_argument("--gap", type=int, default=120,
                    help="vertical gap between cards")
    ap.add_argument("--arrow-row", type=int, default=1,
                    help="0-based card row the column arrows are centred on")
    ap.add_argument("--pad", type=int, default=127,
                    help="padding below the deepest card")
    ap.add_argument("--loop-extra", type=int, default=95,
                    help="extra px to raise the Next-sentence loop")
    a = ap.parse_args()
    render(a.pdf, a.page, a.dpi, "/tmp/_fig2_base.png")
    im = Image.open("/tmp/_fig2_base.png").convert("RGB")
    im = trim_text_card(im)
    im = drop_subtitles(im)
    im = drop_card_bodies(im, a.card_h, a.top_y, a.gap)
    im = shorten(im, a.pad, a.loop_extra)
    im = even_margins(im)
    im = center_arrows(im, a.arrow_row)
    im.save(a.out)
    print(f"{a.out}  {im.width}x{im.height}  {im.width/im.height:.2f}:1")


