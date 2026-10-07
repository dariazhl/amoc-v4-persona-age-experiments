import json
import argparse
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from statsmodels.formula.api import mixedlm
import re
import os

from amoc.pipeline.orchestrator import AMoCv4
from amoc.llm.vllm_client import VLLMClient
from amoc.output.recorder import graph_edges_to_triplets
from amoc.utils.spacy_utils import load_spacy
from amoc.utils.io import robust_read_persona_csv
from amoc.admission.node_admission import NodeAdmission as _NodeAdmission

# Fix: get_or_create_node_from_text calls admit_node with provenance="TEXT_FALLBACK",
# which hits the catch-all rejection at node_admission.py:127. Allow it through.
_orig_admit = _NodeAdmission.admit_node
def _permissive_admit(self, lemma, node_type, provenance="STORY_EXPLICIT", **kw):
    if provenance == "TEXT_FALLBACK":
        return True
    return _orig_admit(self, lemma, node_type, provenance=provenance, **kw)
_NodeAdmission.admit_node = _permissive_admit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPT = """You have the following edges from a knowledge graph in the format: node - edge - node.
{edges}
For the word "{probe_word}", assign a score from 1 to 4 using these definitions:
- 4: Strong connection or relevance to the story. The story told by the edges clearly implies or revolves around this word, even if the word itself is not written among the edges.
- 3: A clear connection or relevance, but the word is not the main focus of the story.
- 2: Little connection or relevance.
- 1: No connection or relevance.
Return ONLY the number (1,2,3,4). Do not add any extra text or punctuation."""

def score_probe_llm(amoc, probe_lemma):
    triplets = graph_edges_to_triplets(amoc.graph, only_active=False)
    #print(f"[DEBUG] Number of triplets for probe '{probe_lemma}': {len(triplets)}")
    # if triplets:
    #     print(f"[DEBUG] First triplet: {triplets[0]}")
    # else:
    #     print(f"[DEBUG] No triplets found. Graph edges: {list(amoc.graph.edges)}")
    if not triplets:
        return 1
    edges_str = "\n".join([f"{s} - {r} - {o}" for s, r, o in triplets])
    print(f"Edges for {probe_lemma}:\n{edges_str}")
    prompt = PROMPT.format(edges=edges_str, probe_word=probe_lemma)
    try:
        response = amoc.client.call_vllm(prompt, persona=amoc.persona)
        print(f"[DEBUG] LLM response: '{response}'", flush=True)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return 1

    try:
        score = int(response.strip())
    except ValueError:
        match = re.search(r"\b([1-4])\b", response)
        score = int(match.group(1)) if match else 1
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
    parser = argparse.ArgumentParser(description="Keefe experiment with personas (SLURM array compatible)")
    parser.add_argument("--persona-csv", required=True, help="CSV chunk file with personas (must have 'persona_text' and 'age_refined')")
    parser.add_argument("--keefe-json", required=True, help="JSON file with Keefe items")
    parser.add_argument("--output-csv", required=True, help="Output CSV file for this chunk (will include all personas in the chunk)")
    parser.add_argument("--stats-output", default=None, help="Optional file to save LME pairwise table")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="vLLM model name")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of personas in this chunk (for testing)")
    parser.add_argument("--start-row", type=int, default=0, help="Skip the first N personas in this chunk (0-based offset, e.g. resume after an earlier --max-rows run)")
    args = parser.parse_args()

    # Load texts
    with open(args.keefe_json, "r") as f:
        items = json.load(f)
    logger.info(f"Loaded {len(items)} Keefe items from {args.keefe_json}")

    # Load persona chunk
    df = robust_read_persona_csv(args.persona_csv)
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        raise ValueError(f"CSV must contain 'persona_text' and 'age_refined' columns. Found: {df.columns.tolist()}")
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")
    start = args.start_row or 0
    if args.max_rows:
        df = df.iloc[start:start + args.max_rows]
    elif start:
        df = df.iloc[start:]
    logger.info(f"Loaded {len(df)} personas from {args.persona_csv} (start_row={start})")

    # spaCy
    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("Failed to load spaCy model.")

    client = VLLMClient(model_name=args.model, tp_size=args.tp, debug=False)

    original_clean = client._clean_response
    def keefe_clean_response(raw_text: str) -> str:
        # First, look for a single digit 1
        match = re.search(r'\b([1-4])\b', raw_text)
        if match:
            return match.group(1)
        # Fall back to the original cleaning (which expects brackets)
        return original_clean(raw_text)
    client._clean_response = keefe_clean_response

    all_scores = []
    write_header = not os.path.exists(args.output_csv)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing personas"):
        persona_text = str(row["persona_text"])
        age = int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
        persona_description = f"Age: {age} years old.\n{persona_text}"
        persona_scores = []
        try:
            for item in items:
                item_id = item["id"]
                probe_lemma = item["target_lemma"]
                for cond in ["control", "predictive", "explicit"]:
                    sentence = item[f"{cond}_text"]
                    amoc = AMoCv4(
                        persona_description=persona_description,
                        story_text=sentence,
                        vllm_client=client,
                        max_distance_from_active_nodes=2,
                        max_new_concepts=10,
                        max_new_properties=10,
                        context_length=1,
                        edge_visibility=2,
                        nr_relevant_edges=10,
                        spacy_nlp=spacy_nlp,
                        debug=False,
                        persona_age=age,
                        strict_reactivate_function=True,
                        single_anchor_hub=True,
                        matrix_dir_base=None,
                        checkpoint=False,
                    )
                    amoc._activation_ops.record_sentence_activation_matrix = lambda *a, **kw: None
                    # amoc._inference_ops.infer_new_relationships_step_0 = lambda sent: ([], [])
                    # amoc._inference_ops.infer_new_relationships = lambda *a, **kw: ([], [])
                    # amoc._inference_ops._add_inferred_relationships_to_graph_step_0 = lambda *a, **kw: None
                    # amoc._inference_ops._add_inferred_relationships_to_graph = lambda *a, **kw: None
                    # amoc._infer_new_relationships_step_0_fn = lambda sent: ([], [])
                    # amoc._add_inferred_relationships_to_graph_step_0_fn = lambda *a, **kw: None
                    # amoc._infer_new_relationships_fn = lambda *a, **kw: ([], [])
                    # amoc._add_inferred_relationships_to_graph_fn = lambda *a, **kw: None
                    amoc.record_activation_matrix_wrapper = lambda *args, **kwargs: None
                    amoc._activation_ops.export_activation_matrix_csv = lambda *a, **kw: None
                    amoc._output_ops.finalize_outputs = lambda *a, **kw: (None, None, None)
                    amoc._plot_ops.plot_sentence_views = lambda *a, **kw: None
                    amoc._plot_ops.plot_graph_snapshot_full = lambda *a, **kw: None
                    # amoc.stabilize_connectivity_wrapper = lambda *a, **kw: False
                    # amoc._connectivity_ops.run_repair_pipeline = lambda *a, **kw: None
                    # amoc.is_attachable_wrapper = lambda *a, **kw: True
                    # amoc._edge_ops._get_attachable_nodes = lambda: set(amoc.graph.nodes)
                    # amoc._sentence_processing_ops._extract_deterministic_structure_fn = lambda *a, **kw: None

                    amoc.analyze(replace_pronouns=False, plot_after_each_sentence=False)
                    score = score_probe_llm(amoc, probe_lemma)
                    persona_scores.append({
                        "global_persona_index": row.get("original_index", idx),
                        "age": age,
                        "persona_text": persona_text,
                        "item_id": item_id,
                        "condition": cond,
                        "score": score,
                    })
        except Exception as e:
            logger.error(f"Persona idx={idx} failed: {e}", exc_info=True)
            if not persona_scores:
                continue

        if persona_scores:
            all_scores.extend(persona_scores)
            pd.DataFrame(persona_scores).to_csv(
                args.output_csv, mode="a", header=write_header, index=False
            )
            write_header = False
            logger.info(f"Saved {len(persona_scores)} rows for persona idx={idx} (running total: {len(all_scores)})")

    # After all personas in the chunk are processed, compute LME on the entire chunk
    if all_scores:
        df_out = pd.DataFrame(all_scores)
        print(f"Total {len(df_out)} scores written to {args.output_csv}")

        # Compute pairwise table
        stats_df = lme_pairwise(df_out)
        print("\n=== LME Pairwise Comparisons ===")
        print(stats_df)

        # Save to separate file 
        if args.stats_output:
            stats_df.to_csv(args.stats_output, index=False)
            print(f"Saved LME pairwise table to {args.stats_output}")

if __name__ == "__main__":
    main()