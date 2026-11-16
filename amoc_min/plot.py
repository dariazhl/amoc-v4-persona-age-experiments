"""Figure rendering for amoc_min.

Carries the original AMoC plot's text -- persona, sentence, explicit /
carry-over / inactive nodes -- but one font,
four type sizes, a muted palette, thin links and no boxes or rotated text.

The layout is computed ONCE from the union of every triplet the story can
produce, so node positions never move between sentences and the thirteen
frames read as one animation.

    deep blue    concept stated in the text
    amber        concept the reader inferred
    pale grey    in memory but not active
"""
import math
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe      # noqa: E402
import matplotlib.pyplot as plt          # noqa: E402
import networkx as nx                    # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

FONT = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
INK, MUTED, FAINT = "#1F2933", "#7B8794", "#B6BFCA"
# v5's node colours: blue = from the text, amber = reader-inferred,
# grey = in memory. Labels sit INSIDE the circle as they do in v5, which
# is what stops them colliding once the graph passes ~20 nodes.
TEXT_FILL, TEXT_EDGE = "#A0CBE2", "#1F2933"
INFER_FILL, INFER_EDGE = "#FFE8A0", "#1F2933"
OFF_FILL, OFF_EDGE = "#D0D0D0", "#999999"
LINK, LINK_OFF = "#5B6B7F", "#E6E9ED"
RULE = "#E8EBEF"

# One multiplier on every type size, so the hierarchy between the sizes is
# preserved and only the weight of text against the graph changes. Lower it
# to give the drawing more of the canvas.
TYPE = 0.78


def _fs(points):
    return round(points * TYPE, 2)


# The footer runs the full page width, past the graph column and under the
# triplet panel, so it gets a size of its own rather than the body size.
FOOTER_PT, FOOTER_WRAP = 5.6, 385


def node_radius(n_nodes):
    """v5's curve (node_size = 16000*sqrt(8/n), clamped), in axes fractions."""
    return max(0.038, min(0.098, 0.082 * math.sqrt(8 / max(1, n_nodes))))


def overlaps(pos, sep):
    """Pairs still closer than `sep`. Empty means no circle touches another."""
    keys = list(pos)
    return [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
            if math.dist(pos[a], pos[b]) < sep - 1e-9]


def layout(all_triplets, seed=7):
    """One layout for the whole story, from every triplet that can appear."""
    g = nx.Graph()
    for s, _, o in all_triplets:
        g.add_edge(s, o)
    if not g:
        return {}
    pos = nx.spring_layout(g, seed=seed, k=2.7 / math.sqrt(max(len(g), 2)),
                           iterations=600)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    # Fit to a 16:9 box rather than a square, so the graph fills the width.
    # Scaling x and y independently is fine: aspect="equal" is applied to the
    # axes, so node circles stay circular whatever the layout aspect.
    sx = (max(xs) - min(xs)) or 1.0
    sy = (max(ys) - min(ys)) or 1.0
    out = {n: ((x - cx) / sx * 1.52, (y - cy) / sy * 1.12)
           for n, (x, y) in pos.items()}
    # Separate by twice the radius the frames will actually use, plus a
    # margin. node_radius() keys off len(pos) too, so the two always agree
    # and no two circles can touch in any frame.
    return _separate(out, sep=2.0 * node_radius(len(out)) * 1.12)


def _separate(pos, sep=0.088, rounds=4000):
    """Push overlapping nodes apart.

    spring_layout optimises edge lengths, not disc packing, so once the
    circles are big enough to hold their own label a few pairs always sit on
    top of one another. This relaxes any pair closer than `sep` without
    disturbing the overall shape. Deterministic, so frames stay aligned.
    """
    keys = list(pos)
    p = {k: list(v) for k, v in pos.items()}
    for _ in range(rounds):
        moved = False
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                dx, dy = p[b][0] - p[a][0], p[b][1] - p[a][1]
                d = math.hypot(dx, dy)
                if d >= sep:
                    continue
                if d < 1e-9:            # exactly coincident: split on x
                    dx, dy, d = 1e-3, 0.0, 1e-3
                push = (sep - d) / 2.0
                ux, uy = dx / d, dy / d
                p[a][0] -= ux * push; p[a][1] -= uy * push
                p[b][0] += ux * push; p[b][1] += uy * push
                moved = True
        if not moved:
            break
    return {k: tuple(v) for k, v in p.items()}


def _field(ax, y, label, items, width=FOOTER_WRAP):
    """'Label: a, b, c', wrapped and hanging-indented.

    Left edge stays on the graph axes, so it lines up with the header, but the
    wrap is wide enough that the lines run the whole page rather than stopping
    at the graph column.
    """
    value = ", ".join(items) if items else "none"
    lines = textwrap.wrap(f"{label}: {value}", width=width,
                          subsequent_indent=" " * (len(label) + 2)) or [label]
    for n, line in enumerate(lines):
        ax.annotate(line, (0.0, y - n * 0.016), xycoords="axes fraction",
                    fontsize=_fs(FOOTER_PT), color=MUTED, ha="left", va="top",
                    annotation_clip=False)
    return y - 0.016 * len(lines) - 0.006


def draw(path, pos, nodes, edges, sentence_no, sentence, persona, model,
         explicit, carryover, inactive, active_triplets=()):
    """nodes: {lemma: ("text"|"inferred"|"off", activation)}
       edges: [(src, label, dst, active)]

    The active-triplet list is no longer drawn: it duplicates triplets_<tag>.csv
    and cost the graph a fifth of the canvas."""
    plt.rcParams.update({"font.family": FONT, "axes.linewidth": 0})
    fig = plt.figure(figsize=(22.0, 12.0), dpi=160)
    gs = GridSpec(1, 2, width_ratios=[0.74, 0.26], figure=fig, wspace=0.03)
    ax = fig.add_subplot(gs[0])
    panel = fig.add_subplot(gs[1])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- graph ----------------------------------------------------------
    # Sized from the story's whole node set, not this sentence's, so a
    # circle keeps one size across all thirteen frames and the separation
    # computed in layout() holds everywhere.
    _base_r = node_radius(len(pos))

    def _radius(lemma):
        kind, score = nodes.get(lemma, ("off", 0.0))
        return _base_r * (0.82 + 0.18 * (score / 5.0))

    for s, label, o, active in edges:
        if s not in pos or o not in pos:
            continue
        (x1, y1), (x2, y2) = pos[s], pos[o]
        # shrink the arrow to the circle edges so the head is visible
        dx, dy = x2 - x1, y2 - y1
        n = math.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        a = (x1 + ux * _radius(s), y1 + uy * _radius(s))
        b = (x2 - ux * _radius(o), y2 - uy * _radius(o))
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11 if active else 8,
            lw=1.3 if active else 0.7,
            color=LINK if active else LINK_OFF,
            shrinkA=0, shrinkB=0, zorder=1))
        if active and label:
            dx, dy = x2 - x1, y2 - y1
            n = math.hypot(dx, dy) or 1.0
            ox, oy = -dy / n * 0.018, dx / n * 0.018
            ax.annotate(label, ((x1 + x2) / 2 + ox, (y1 + y2) / 2 + oy),
                        fontsize=_fs(6.6), color=MUTED, ha="center", va="center",
                        zorder=3, path_effects=[
                            pe.withStroke(linewidth=2.0, foreground="white")])

    for lemma, (kind, score) in nodes.items():
        if lemma not in pos:
            continue
        x, y = pos[lemma]
        fill, line, ink = {
            "text": (TEXT_FILL, TEXT_EDGE, INK),
            "inferred": (INFER_FILL, INFER_EDGE, INK),
            "off": (OFF_FILL, OFF_EDGE, FAINT),
        }[kind]
        r = _radius(lemma)
        on = kind != "off"
        ax.add_patch(plt.Circle((x, y), r, facecolor=fill, edgecolor=line,
                                linewidth=1.5 if on else 1.0,
                                alpha=1.0 if on else 0.45, zorder=4))
        # Inside the circle, wrapped and shrunk to fit -- never above it,
        # which is what made neighbouring labels overlap.
        lines = textwrap.wrap(lemma, 9, break_long_words=False) or [lemma]
        longest = max(len(l) for l in lines)
        ax.annotate("\n".join(lines), (x, y),
                    fontsize=_fs(11.0) * min(1.0, 7.5 / longest) *
                            (r / _base_r),
                    color=ink if on else "#777777", fontweight="bold" if on else "normal",
                    ha="center", va="center", zorder=5, linespacing=0.92)

    # ---- header ---------------------------------------------------------
    ax.annotate(f"AMoC knowledge graph · {model}", (0, 1.135),
                xycoords="axes fraction", fontsize=_fs(7.6), color=FAINT,
                ha="left", va="bottom")
    ax.annotate(f"Sentence {sentence_no}: {sentence}", (0, 1.060),
                xycoords="axes fraction", fontsize=_fs(12.5), color=INK,
                ha="left", va="bottom")
    ax.annotate(f"Persona: {persona}", (0, 1.018), xycoords="axes fraction",
                fontsize=_fs(8.4), color=MUTED, ha="left", va="bottom")

    for k, (lab, fill, line) in enumerate([("explicit", TEXT_FILL, TEXT_EDGE),
                                           ("inferred", INFER_FILL, INFER_EDGE),
                                           ("in memory", OFF_FILL, OFF_EDGE)]):
        x = 0.655 + k * 0.118
        ax.add_patch(plt.Circle((x, 1.030), 0.009, facecolor=fill,
                                edgecolor=line, linewidth=1.0,
                                transform=ax.transAxes, clip_on=False, zorder=6))
        ax.annotate(lab, (x + 0.013, 1.030), xycoords="axes fraction",
                    fontsize=_fs(7.2), color=MUTED, ha="left", va="center")

    # ---- footer ---------------------------------------------------------
    y = -0.045
    y = _field(ax, y, "Explicit this sentence", explicit)
    y = _field(ax, y, "Carry-over from previous sentences", carryover)
    y = _field(ax, y, "Inactive nodes (retained in memory)", inactive)

    px = [p[0] for p in pos.values()] or [0]
    py = [p[1] for p in pos.values()] or [0]
    m = 0.11
    ax.set_xlim(min(px) - m, max(px) + m)
    ax.set_ylim(min(py) - m, max(py) + m * 1.3)
    ax.set_aspect("equal")
    ax.axis("off")

    panel.axis("off")
    n = len(active_triplets)
    panel.annotate(f"Active triplets  ({n})", (0, 1.055),
                   xycoords="axes fraction", fontsize=_fs(8.4), color=INK,
                   ha="left", va="bottom")
    panel.plot([0, 1], [1.042, 1.042], transform=panel.transAxes,
               color=RULE, lw=0.9, clip_on=False)
    if not n:
        panel.annotate("none", (0, 1.010), xycoords="axes fraction",
                       fontsize=_fs(7.6), color=FAINT, ha="left", va="top")
    for i, (s, r_, o) in enumerate(active_triplets[:34]):
        yy = 1.010 - i * 0.024
        panel.annotate(f"{i + 1}", (0.0, yy), xycoords="axes fraction",
                       fontsize=_fs(6.8), color=FAINT, ha="left", va="top")
        panel.annotate(f"{s} \u2014{r_}\u2192 {o}", (0.06, yy),
                       xycoords="axes fraction", fontsize=_fs(7.2), color=MUTED,
                       ha="left", va="top")

    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
