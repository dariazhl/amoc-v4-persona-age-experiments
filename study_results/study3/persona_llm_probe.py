import argparse
import glob
import itertools
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from amoc.llm.vllm_client import VLLMClient
from amoc.utils.spacy_utils import load_spacy

REGIME_FILES = {"primary": "primary_*.csv", "university": "university_*.csv"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ"}


def _norm(x):
    return " ".join(str(x).lower().replace("_", " ").split())


def _jac(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


def pick_personas(chunks_dir, n, seed):
    rows = []
    for regime, pat in REGIME_FILES.items():
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(chunks_dir, pat)))], ignore_index=True)
        df = df.drop_duplicates("persona_text").sample(n=n, random_state=seed)
        rows += [{"regime": regime, "persona_text": t} for t in df["persona_text"]]
    return pd.DataFrame(rows)


def nodes_block(nlp, sent):
    seen, out = set(), ""
    for tok in nlp(sent):
        lem = tok.lemma_.lower()
        if tok.pos_ in CONTENT_POS and tok.is_alpha and lem not in seen:
            seen.add(lem)
            out += f" - ({tok.text}, {tok.pos_})\n"
    return out


def run_calls(args):
    nlp = load_spacy()
    story = open(args.story).read()
    sents = [s.text.strip() for s in nlp(story).sents if s.text.strip()][: args.sentences]
    personas = pick_personas(args.chunks_dir, args.n_personas, args.seed)
    client = VLLMClient(model_name=args.model, server_url=args.server_url)
    rows = []
    for mode in args.modes:
        os.environ["AMOC_PERSONA_MODE"] = mode
        for pi, p in personas.iterrows():
            for si, sent in enumerate(sents):
                block = nodes_block(nlp, sent)
                for rep in range(args.reps):
                    d = client.infer_objects_and_properties_first_sentence(block, sent, p.persona_text) or {}
                    rows.append({"mode": mode, "persona_idx": pi, "regime": p.regime, "persona_text": p.persona_text,
                                 "sentence_idx": si, "sentence": sent, "rep": rep,
                                 "concepts": "|".join(_norm(c) for c in d.get("concepts", []) or []),
                                 "properties": "|".join(_norm(c) for c in d.get("properties", []) or [])})
                print(f"mode={mode} persona={pi + 1}/{len(personas)} done", flush=True)
    return pd.DataFrame(rows), story


def words(s):
    return {w for w in re.findall(r"[a-z]+", str(s).lower()) if len(w) > 2}


def analyse(df, story, out_dir):
    df = df.fillna("")
    df["items"] = [set(filter(None, c.split("|"))) | set(filter(None, p.split("|"))) for c, p in zip(df.concepts, df.properties)]
    story_words = words(story)
    summary = []
    for mode, g in df.groupby("mode"):
        within, same, diff = [], [], []
        for si, gs in g.groupby("sentence_idx"):
            cells = {pi: list(gp["items"]) for pi, gp in gs.groupby("persona_idx")}
            reg = gs.groupby("persona_idx")["regime"].first().to_dict()
            for pi, reps in cells.items():
                within += [_jac(a, b) for a, b in itertools.combinations(reps, 2)]
            for a, b in itertools.combinations(cells, 2):
                j = _jac(cells[a][0], cells[b][0])
                (same if reg[a] == reg[b] else diff).append(j)
        per = g.groupby(["persona_idx", "regime"]).agg(
            n_items=("items", lambda s: np.mean([len(x) for x in s])),
            word_len=("items", lambda s: np.mean([np.mean([len(w) for w in x]) for x in s if x] or [np.nan])),
            off_text=("items", lambda s: np.mean([np.mean([not (words(w) <= story_words) for w in x]) for x in s if x] or [np.nan])),
        ).reset_index()
        leak = []
        for pi, gp in g.groupby("persona_idx"):
            pw = words(gp.persona_text.iloc[0]) - story_words
            hits = [bool(words(" ".join(x)) & pw) for x in gp["items"]]
            leak.append(np.mean(hits))
        rec = {"mode": mode,
               "J_within_persona_reps": np.nanmean(within) if within else np.nan,
               "J_between_same_regime": np.nanmean(same),
               "J_between_diff_regime": np.nanmean(diff),
               "persona_word_hit_rate": np.mean(leak)}
        if within:
            rec["p_within_gt_between"] = mannwhitneyu(within, same + diff, alternative="greater").pvalue
        rec["p_same_gt_diff"] = mannwhitneyu(same, diff, alternative="greater").pvalue if same and diff else np.nan
        for m in ["n_items", "word_len", "off_text"]:
            a = per.loc[per.regime == "primary", m].dropna()
            b = per.loc[per.regime == "university", m].dropna()
            rec[f"{m}_primary"] = a.mean()
            rec[f"{m}_university"] = b.mean()
            rec[f"p_{m}"] = mannwhitneyu(a, b).pvalue if len(a) > 1 and len(b) > 1 else np.nan
        summary.append(rec)
        per.assign(mode=mode).to_csv(os.path.join(out_dir, f"per_persona_{mode}.csv"), index=False)
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        txt = s.set_index("mode").T.to_string()
    lines = [txt, "",
             "J_within_persona_reps: same persona, same sentence, repeated calls (1.0 = deterministic).",
             "J_between_*: different personas on the same sentence. Persona effect if within >> between and same_regime > diff_regime.",
             "off_text: share of inferred items not in the story; word_len: mean characters per item (proxy for vocabulary level).",
             "persona_word_hit_rate: share of calls whose output contains a word from the persona description that is not in the story."]
    if "legacy" in s["mode"].values and "reader" in s["mode"].values:
        pairs = df.pivot_table(index=["persona_idx", "sentence_idx", "rep"], columns="mode", values="items", aggfunc="first").dropna()
        same_mode_diff = np.mean([_jac(a, b) for a, b in zip(pairs["legacy"], pairs["reader"])])
        lines.append(f"J_legacy_vs_reader (same persona/sentence/rep): {same_mode_diff:.3f}")
    open(os.path.join(out_dir, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url")
    ap.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--story", default="tusa_text/min_drp_texts/college_4.txt")
    ap.add_argument("--chunks-dir", default="personas_dfs/personas_refined_age/chunks_balanced")
    ap.add_argument("--n-personas", type=int, default=5)
    ap.add_argument("--sentences", type=int, default=6)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["legacy", "reader"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="study_results/study3/persona_effect/llm_probe")
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw_calls.csv")
    if args.analyse_only:
        df = pd.read_csv(raw_path)
        story = open(args.story).read()
    else:
        if not args.server_url:
            ap.error("--server-url is required unless --analyse-only")
        df, story = run_calls(args)
        df.to_csv(raw_path, index=False)
    analyse(df, story, args.out_dir)


if __name__ == "__main__":
    main()
