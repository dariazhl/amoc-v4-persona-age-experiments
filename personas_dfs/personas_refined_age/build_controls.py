"""Positive-control pilot: 12 story-relevant personas, 12 matched generic ones, 12 placebos.
Relevant and generic texts are copied unchanged from chunks_balanced (the pool the .sh reads).
Placebo = the matched generic text with its words in random order (fixed seed).
    python build_controls.py   ->  chunks_controls/<regime>_controls.csv, controls_key.csv
idx keeps the pool id with a letter in front: R relevant, G generic, P placebo."""
import csv, glob, os, random
H = os.path.dirname(os.path.abspath(__file__))
PAIRS = {"primary":    [("114691", "170391"), ("188617", "137880"), ("183033", "37956")],
         "secondary":  [("134631", "17374"),  ("69806", "90719"),   ("111683", "26653")],
         "highschool": [("40356", "22234"),   ("9781", "23635"),    ("26780", "28768")],
         "university": [("72382", "41919"),   ("29065", "29158"),   ("161077", "64610")]}
rng = random.Random(0)
key = []
for reg, pairs in PAIRS.items():
    pool = {}
    for f in sorted(glob.glob(f"{H}/chunks_balanced/{reg}_*.csv")):
        for i, r in enumerate(csv.DictReader(open(f, newline=""))):
            pool.setdefault(r["idx"], (r, os.path.basename(f), i))
    rows = []
    for n, (rel, gen) in enumerate(pairs, 1):
        for arm, src in (("R", rel), ("G", gen), ("P", gen)):
            r, f, i = pool[src]
            text = r["persona_text"].strip()
            if arm == "P":
                w = text.rstrip(".").split(); orig = w[:]
                while w == orig: rng.shuffle(w)
                text = " ".join(w)
            rows.append({"idx": arm + src, "persona_text": text, "age_refined": r["age_refined"] or r["age"]})
            key.append({"tag": f"{reg}_{arm}{src}", "arm": {"R": "relevant", "G": "generic", "P": "placebo"}[arm],
                        "pair": f"{reg}_{n}", "regime": reg, "age": rows[-1]["age_refined"], "words": len(text.split()),
                        "pool_file": f, "pool_row": i, "in_80": i < 10, "persona_text": text})
    with open(f"{H}/chunks_controls/{reg}_controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["idx", "persona_text", "age_refined"]); w.writeheader(); w.writerows(rows)
with open(f"{H}/controls_key.csv", "w", newline="") as f:
    w = csv.DictWriter(f, key[0].keys()); w.writeheader(); w.writerows(key)
