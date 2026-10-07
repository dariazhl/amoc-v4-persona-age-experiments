import json
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.formula.api import mixedlm
import sys
import re

from amoc.pipeline.orchestrator import AMoCv4
from amoc.llm.vllm_client import VLLMClient
from amoc.output.recorder import graph_edges_to_triplets
from amoc.utils.spacy_utils import load_spacy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prompt exactly as in the paper (Appendix 1, Prompt 8)
PROMPT = """You have the following edges from a knowledge graph in the format: node - edge - node.

{edges}

Using the graph and the story that the graph tells, for the word "{probe_word}" assign a score between 1 and 4 with the following meaning:
1 – no connection or relevance to one or more ideas in the graph (pay attention to the story as well)
2 – little connection or relevance to one or more ideas in the graph
3 – a clear connection or relevance to one or more ideas in the graph
4 – a strong connection or relevance to one or more ideas in the graph

Return only the number (1,2,3,4). Do not add any extra text or punctuation."""

def score_probe_llm(amoc, probe_lemma):
    triplets = graph_edges_to_triplets(amoc.graph, only_active=False)
    if not triplets:
        print("No edges extracted.\n")
        return 1
    edges_str = "\n".join([f"{s} - {r} - {o}" for s, r, o in triplets])
    prompt = PROMPT.format(edges=edges_str, probe_word=probe_lemma)
    response = amoc.client.generate_raw(prompt, temperature=0.0)
    print(f"[DEBUG] LLM response for '{probe_lemma}': '{response.strip()}'")
    try:
        score = int(response.strip())
    except ValueError:
        match = re.search(r"\b([1-4])\b", response)
        score = int(match.group(1)) if match else 1

    # Print debug 
    print("\n=== Extracted edges ===")
    print(edges_str)
    print("="*50)
    print(f"LLM response: {response.strip()}")
    print(f"Parsed score: {score}\n")
    return score

def lme_pairwise(df, item_col="item_id", condition_col="condition", score_col="score"):
    df = df.copy()
    df[condition_col] = pd.Categorical(df[condition_col],
                                       categories=["control", "predictive", "explicit"],
                                       ordered=False)
    model = mixedlm(f"{score_col} ~ C({condition_col}, Treatment('control'))", df, groups=df[item_col])
    result = model.fit()
    pred_coef = result.params.get("C(condition, Treatment('control'))[T.predictive]", np.nan)
    pred_se = result.bse.get("C(condition, Treatment('control'))[T.predictive]", np.nan)
    expl_coef = result.params.get("C(condition, Treatment('control'))[T.explicit]", np.nan)
    expl_se = result.bse.get("C(condition, Treatment('control'))[T.explicit]", np.nan)
    cov = result.cov_params().loc[
        "C(condition, Treatment('control'))[T.predictive]",
        "C(condition, Treatment('control'))[T.explicit]"
    ]
    pred_vs_expl_est = pred_coef - expl_coef
    pred_vs_expl_se = np.sqrt(pred_se**2 + expl_se**2 - 2 * cov)

    def z_p(est, se):
        if np.isnan(est) or se == 0:
            return np.nan, np.nan
        z = est / se
        p = 2 * (1 - norm.cdf(abs(z)))
        return z, p

    expl_z, expl_p = z_p(expl_coef, expl_se)
    pred_z, pred_p = z_p(pred_coef, pred_se)
    pvex_z, pvex_p = z_p(pred_vs_expl_est, pred_vs_expl_se)

    return pd.DataFrame({
        "Comparison": ["Explicit/Control", "Predictive/Control", "Predictive/Explicit"],
        "Estimate": [expl_coef, pred_coef, pred_vs_expl_est],
        "z": [expl_z, pred_z, pvex_z],
        "p": [expl_p, pred_p, pvex_p]
    })

def main():
    parser = argparse.ArgumentParser(description="Keefe v4.0 style")
    parser.add_argument("--json", default="/export/home/acs/stud/a/ana_daria.zahaleanu/to_transfer/amoc-v4-persona-age-experiments/amoc/keefe_exp/keefe_ready.json", help="Path to JSON file")
    parser.add_argument("--output", default="keefe_original_results.csv", help="Output CSV file")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    args = parser.parse_args()

    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("Failed to load spaCy model. Run: python -m spacy download en_core_web_sm")

    client = VLLMClient(model_name=args.model, tp_size=args.tp, debug=False)

    with open(args.json, "r") as f:
        items = json.load(f)
    logger.info(f"Loaded {len(items)} items from {args.json}")

    rows = []
    for item in tqdm(items, desc="Processing items"):
        item_id = item["id"]
        probe_lemma = item["target_lemma"]
        for cond in ["control", "predictive", "explicit"]:
            sentence = item[f"{cond}_text"]

            amoc = AMoCv4(
                persona_description=" ",
                story_text=sentence,
                vllm_client=client,
                max_distance_from_active_nodes=2,
                max_new_concepts=0,
                max_new_properties=0,
                context_length=1,
                edge_visibility=2,
                nr_relevant_edges=10,
                spacy_nlp=spacy_nlp,
                debug=False,
                persona_age=-1,
                strict_reactivate_function=True,
                single_anchor_hub=True,
                matrix_dir_base=None,
                checkpoint=False,
            )
            # disable inference and activation matrix wrapper which interfere with the prompt above
            amoc.record_activation_matrix_wrapper = lambda *args, **kwargs: None
            amoc._inference_ops.infer_new_relationships_step_0 = lambda sent: ([], [])
            amoc._inference_ops.infer_new_relationships = lambda *a, **kw: ([], [])
            amoc._inference_ops._add_inferred_relationships_to_graph_step_0 = lambda *a, **kw: None
            amoc._inference_ops._add_inferred_relationships_to_graph = lambda *a, **kw: None
            amoc._infer_new_relationships_step_0_fn = lambda sent: ([], [])
            amoc._add_inferred_relationships_to_graph_step_0_fn = lambda *a, **kw: None
            amoc._infer_new_relationships_fn = lambda *a, **kw: ([], [])
            amoc._add_inferred_relationships_to_graph_fn = lambda *a, **kw: None
            amoc._activation_ops.record_sentence_activation_matrix = lambda *a, **kw: None
            amoc._activation_ops.export_activation_matrix_csv = lambda *a, **kw: None
            amoc._output_ops.finalize_outputs = lambda *a, **kw: (None, None, None)
            amoc._plot_ops.plot_sentence_views = lambda *a, **kw: None
            amoc._plot_ops.plot_graph_snapshot_full = lambda *a, **kw: None

            amoc.analyze(replace_pronouns=False, plot_after_each_sentence=False)
            score = score_probe_llm(amoc, probe_lemma)
            rows.append({
                "item_id": item_id,
                "condition": cond,
                "score": score
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print("\n=== Descriptive Statistics ===")
    print(df.groupby("condition")["score"].describe())
    print("\n=== LME Pairwise Comparisons ===")
    print(lme_pairwise(df))

if __name__ == "__main__":
    main()