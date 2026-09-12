import os
import re
import glob
import shutil
import sys
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUN28 = os.path.join(ROOT, "results/Qwen3-30b/final_plots/jun_28")
OUT = os.path.join(ROOT, "diagrams/text_diagrams_2")

TEXTS = ["primary", "secondary", "highschool"]
BANDS = {"primary": "primary", "secondary": "secondary",
         "highschool": "highschool", "university": "college"}

SELECTED_PREFIX = {
    "primary": "A primary school student who loves learning new",
    "secondary": "A curious and inquisitive middle school student",
    "highschool": "A high school student who volunteers alongside the",
    "university": "a first-year economy student at a less competitive",
}

FNAME_PAT = re.compile(
    r"amoc_graph_.*?_Age_ (?P<frag>.+?)_(?P<age>\d+)_reverse_paper_sent(?P<sent>\d+)_paper\.png$"
)


def _norm(s):
    return re.sub(r"[^0-9A-Za-z]+", " ", s).strip().lower()


def full_persona_records():
    out = {}
    for band, prefix in SELECTED_PREFIX.items():
        f = glob.glob(
            f"{JUN28}/big_run_primary/triplets/triplets_per_sentence/*_{band}_0*.csv"
        )
        for path in f:
            df = pd.read_csv(
                path, usecols=["persona_text", "age_refined", "original_index"]
            )
            hit = df[df.persona_text.str.startswith(prefix)]
            if len(hit):
                out[band] = (
                    hit.persona_text.iloc[0],
                    int(hit.age_refined.iloc[0]),
                    int(hit.original_index.iloc[0]),
                )
                break
        if band not in out:
            sys.exit(f"persona not found in CSVs for band {band}")
    return out


def persona_pngs(text, ptext, age, oi):
    target = _norm(f"{age} years old. {ptext}")
    for d in (
        f"{JUN28}/big_run_{text}/persona_{oi}/reverse_plots",
        f"{JUN28}/big_run_{text}/reverse_plots",
    ):
        if not os.path.isdir(d):
            continue
        sents = {}
        collisions = defaultdict(set)
        for fn in os.listdir(d):
            m = FNAME_PAT.match(fn)
            if not m:
                continue
            if m.group("age") != str(age) or not target.startswith(_norm(m.group("frag"))):
                continue
            sent = int(m.group("sent"))
            sents[sent] = os.path.join(d, fn)
            collisions[sent].add(m.group("frag"))
        if any(len(v) > 1 for v in collisions.values()):
            sys.exit(f"fragment collision in {d} for '{ptext[:40]}'")
        if sents:
            return sents
    return {}


def main():
    personas = full_persona_records()
    plan = {}
    for text in TEXTS:
        for band, (ptext, age, oi) in personas.items():
            sents = persona_pngs(text, ptext, age, oi)
            if not sents:
                sys.exit(f"no PNGs: text={text} band={band} '{ptext[:40]}'")
            plan[(text, band)] = sents

    last_per_text = {
        text: min(max(plan[(text, b)]) for b in personas) for text in TEXTS
    }
    print("last sentence per text:", last_per_text)

    for tag_dir in ("sent1", "sent2", "last"):
        os.makedirs(os.path.join(OUT, tag_dir), exist_ok=True)

    n = 0
    for (text, band), sents in plan.items():
        out_band = BANDS[band]
        for tag_dir, sent in (("sent1", 1), ("sent2", 2), ("last", last_per_text[text])):
            src = sents.get(sent)
            if src is None:
                sys.exit(f"missing sent{sent} for text={text} band={band}")
            tag = "last" if tag_dir == "last" else f"sent{sent}"
            dst = os.path.join(OUT, tag_dir, f"{text}_{out_band}_{tag}.png")
            shutil.copy2(src, dst)
            n += 1
    print(f"copied {n} files -> {OUT}")
    for band, (ptext, age, oi) in personas.items():
        print(f"  {BANDS[band]:10s} age {age} (idx {oi}): {ptext[:70]}")


if __name__ == "__main__":
    main()
