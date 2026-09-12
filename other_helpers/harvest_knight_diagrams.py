import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNIGHT = os.path.join(ROOT, "results/Qwen3-30b/final_plots/jun_28/knight_july_6")
OUT = os.path.join(ROOT, "diagrams/text_diagrams_2")

SELECTED = {
    "primary": ("A primary school student who loves learning new", 8, 3),
    "secondary": ("A secondary school student who is just starting to learn", 12, 4),
    "highschool": ("A high school student on the baseball team", 16, 3),
    "college": ("a freshman business student at Oakland University", 18, 3),
}

FNAME_PAT = re.compile(
    r"amoc_graph_.*?_Age_ (?P<frag>.+?)_(?P<age>\d+)_reverse_paper_sent(?P<sent>\d+)_paper\.png$"
)


def _norm(s):
    return re.sub(r"[^0-9A-Za-z]+", " ", s).strip().lower()


def main():
    for tag_dir in ("sent1", "sent2", "last"):
        os.makedirs(os.path.join(OUT, tag_dir), exist_ok=True)
    n = 0
    for band, (prefix, age, oi) in SELECTED.items():
        d = f"{KNIGHT}/persona_{oi}/reverse_plots"
        target = _norm(f"{age} years old. {prefix}")
        sents = {}
        for fn in os.listdir(d):
            m = FNAME_PAT.match(fn)
            if not m or m.group("age") != str(age):
                continue
            frag = _norm(m.group("frag"))
            if target.startswith(frag) or frag.startswith(target):
                sent = int(m.group("sent"))
                if sent in sents and sents[sent] != os.path.join(d, fn):
                    sys.exit(f"collision for {band} sent{sent} in {d}")
                sents[sent] = os.path.join(d, fn)
        last = max(sents) if sents else None
        for tag_dir, sent in (("sent1", 1), ("sent2", 2), ("last", last)):
            if sent is None or sent not in sents:
                sys.exit(f"missing sent{sent} for band={band} in {d}")
            tag = "last" if tag_dir == "last" else f"sent{sent}"
            dst = os.path.join(OUT, tag_dir, f"knight_{band}_{tag}.png")
            shutil.copy2(sents[sent], dst)
            n += 1
        print(f"  {band:10s} last=sent{last}  ({os.path.basename(sents[1])[:60]}...)")
    print(f"copied {n} files -> {OUT}")


if __name__ == "__main__":
    main()
