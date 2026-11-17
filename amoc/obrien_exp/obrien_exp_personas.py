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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from itertools import combinations

from amoc.pipeline.orchestrator import AMoCv4
from amoc.llm.vllm_client import ServerUnavailableError, VLLMClient
from amoc.output.recorder import graph_edges_to_triplets
from amoc.utils.spacy_utils import load_spacy
from amoc.utils.io import robust_read_persona_csv
from amoc.admission.node_admission import NodeAdmission as _NodeAdmission
from amoc.admission.triplet_validator import TripletValidator

# allow nodes created with TEXT_FALLBACK provenance -  same as in the Keefe experiment
_orig_admit = _NodeAdmission.admit_node
def _permissive_admit(self, lemma, node_type, provenance="STORY_EXPLICIT", **kw):
    if provenance == "TEXT_FALLBACK":
        return True
    return _orig_admit(self, lemma, node_type, provenance=provenance, **kw)
_NodeAdmission.admit_node = _permissive_admit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Scoring prompt — uses a neutral system role to avoid biasing Llama away from narrative reasoning.
SUPPORT_CONTRADICT_PROMPT = """You are evaluating a knowledge graph against a target sentence.

Here are the edges from the graph, numbered:
{edges}

Target sentence: "{target_sentence}"

Classify each edge as SUPPORT, CONTRADICT:
- SUPPORT: the edge describes a trait, event, or circumstance that makes the target more likely or expected.
- CONTRADICT: the edge describes a trait, event, or circumstance that conflicts with the target — even indirectly. For example, if a character fears heights, any edge expressing that fear contradicts a target involving high-altitude activities, because a person afraid of heights would be unlikely to seek them out.

Only flag an edge as CONTRADICT if it directly states or strongly implies something incompatible with the target. Do NOT flag edges just because they describe traits that might make the target less likely.

Return ONLY a JSON object with "support" and "contradict" lists containing edge numbers.
Example: {{"support": [1, 3], "contradict": [2, 5]}}"""


def _parse_scoring_response(response: str, raw_prompt: str) -> dict:
    clean = response.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # strip Llama <think> blocks
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    try:
        result = json.loads(clean)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    # Fallback: regex extraction
    support_match = re.search(r"\"support\"\s*:\s*\[(.*?)\]", clean, re.DOTALL)
    contradict_match = re.search(r"\"contradict\"\s*:\s*\[(.*?)\]", clean, re.DOTALL)
    if support_match and contradict_match:
        try:
            support_nums = [int(x.strip()) for x in support_match.group(1).split(",") if x.strip().isdigit()]
            contradict_nums = [int(x.strip()) for x in contradict_match.group(1).split(",") if x.strip().isdigit()]
            return {"support": support_nums, "contradict": contradict_nums}
        except Exception:
            pass
    logger.warning(f"Could not parse scoring response: {response[:200]}")
    return {"support": [], "contradict": []}


def get_support_contradict(amoc, target_sentence):
    triplets = graph_edges_to_triplets(amoc.graph, only_active=False)
    if not triplets:
        return {"support": [], "contradict": []}

    edges_str = "\n".join(f"{i}. {s} - {r} - {o}" for i, (s, r, o) in enumerate(triplets, 1))
    prompt = SUPPORT_CONTRADICT_PROMPT.format(
        edges=edges_str,
        target_sentence=target_sentence,
    )

    try:
        # Use score_edges (neutral role) instead of call_vllm (KG-builder role)
        if hasattr(amoc.client, "score_edges"):
            response = amoc.client.score_edges(prompt)
        else:
            response = amoc.client.call_vllm(prompt, persona=amoc.persona)
        print(f"[DEBUG] LLM response: '{response}'", flush=True)
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
        return {"support": [], "contradict": []}

    return _parse_scoring_response(response, prompt)


def lme_all_pairs(df, metric_col, condition_col="condition", item_col="item_id"):
    df = df.copy()
    df[condition_col] = pd.Categorical(df[condition_col])
    model = mixedlm(f"{metric_col} ~ C({condition_col}) - 1", df, groups=df[item_col])
    result = model.fit()

    levels = df[condition_col].cat.categories.tolist()
    # Build contrast vectors and apply t_test
    rows = []
    for cond1, cond2 in combinations(levels, 2):
        # Contrast vector: +1 for cond1, -1 for cond2, 0 otherwise
        contrast = np.zeros(len(result.params))
        param_names = result.params.index.tolist()
        idx1 = param_names.index(f"C({condition_col})[{cond1}]")
        idx2 = param_names.index(f"C({condition_col})[{cond2}]")
        contrast[idx1] = 1.0
        contrast[idx2] = -1.0
        t_res = result.t_test(contrast)
        est = t_res.effect[0]  # difference estimate
        se = t_res.sd[0]
        z = t_res.tvalue[0]    
        p = t_res.pvalue[0]
        rows.append({
            "Comparison": f"{cond1} - {cond2}",
            "Estimate": est,
            "z": z,
            "p": p
        })

    return pd.DataFrame(rows)


_WORKER_CLIENT = None
_WORKER_SPACY = None
_WORKER_ITEMS = None


def _init_worker(model, tp, server_url, items):
    """Per-process setup, as in the Keefe runner: with --server-url every
    worker talks to the one running vLLM server, which batches their requests."""
    global _WORKER_CLIENT, _WORKER_SPACY, _WORKER_ITEMS
    # invalidate triplet validator - let all triplets go through. Done here,
    # not in main(), so it also holds inside spawned worker processes.
    if not hasattr(TripletValidator, "_orig_validate_with_llm"):
        TripletValidator._orig_validate_with_llm = TripletValidator.validate_with_llm
        TripletValidator.validate_with_llm = lambda self, *args, **kwargs: {
            "valid": True,
            "reason": "bypassed for O'Brien experiment",
            "corrected_triple": None
        }
    spacy_nlp = load_spacy()
    if spacy_nlp is None:
        raise RuntimeError("Failed to load spaCy model.")
    _WORKER_CLIENT = VLLMClient(model_name=model, tp_size=tp, debug=False,
                                server_url=server_url)
    _WORKER_SPACY = spacy_nlp
    _WORKER_ITEMS = items


def score_persona(idx, global_index, age, persona_text):
    """Every item x condition for one persona. Returns (idx, rows)."""
    server_failures_before = getattr(_WORKER_CLIENT, "server_failures", 0)
    persona_description = f"Age: {age} years old.\n{persona_text}"
    rows = []
    try:
        for item in _WORKER_ITEMS:
            item_id = item["id"]
            intro = item["introduction"]
            target = item["target_sentence_1"]
            for cond in ["consistent", "inconsistent",
                         "qualified1", "qualified2", "qualified3", "qualified4"]:
                elaboration = item.get(cond, "")
                if not elaboration or not isinstance(elaboration, str) or not elaboration.strip():
                    continue  # skip missing conditions
                story = intro.strip() + " " + elaboration.strip()

                amoc = AMoCv4(
                    persona_description=persona_description,
                    story_text=story,
                    vllm_client=_WORKER_CLIENT,
                    max_distance_from_active_nodes=2,
                    max_new_concepts=15,
                    max_new_properties=15,
                    context_length=2,
                    edge_visibility=2,
                    nr_relevant_edges=15,
                    spacy_nlp=_WORKER_SPACY,
                    debug=False,
                    persona_age=age,
                    strict_reactivate_function=True,
                    single_anchor_hub=True,
                    matrix_dir_base=None,
                    checkpoint=False,
                )

                # Suppress unnecessary recordings and outputs 
                amoc.record_activation_matrix_wrapper = lambda *a, **kw: None
                amoc._activation_ops.record_sentence_activation_matrix = lambda *a, **kw: None
                amoc._activation_ops.export_activation_matrix_csv = lambda *a, **kw: None
                amoc._output_ops.finalize_outputs = lambda *a, **kw: (None, None, None)
                amoc._plot_ops.plot_sentence_views = lambda *a, **kw: None
                amoc._plot_ops.plot_graph_snapshot_full = lambda *a, **kw: None
                amoc.is_attachable_wrapper = lambda *a, **kw: True
                amoc._edge_ops._get_attachable_nodes = lambda: set(amoc.graph.nodes)
                amoc._activation_ops.apply_semantic_edge_decay = lambda: []
                amoc._activation_ops.reactivate_relevant_edges = lambda *a, **kw: None

                amoc.analyze(replace_pronouns=False, plot_after_each_sentence=False)

                triplets = graph_edges_to_triplets(amoc.graph, only_active=False)
                active_triplets = graph_edges_to_triplets(amoc.graph, only_active=True)

                print(f"\n{'='*60}")
                print(f"Item {item_id} | Condition: {cond} | Persona idx: {idx}")
                print(f"Story (intro + elaboration): {story[:200]}...")
                print(f"Target sentence: {target}")
                print(f"\nTotal nodes in graph: {len(amoc.graph.nodes)}")
                print(f"Total edges in graph: {len(amoc.graph.edges)}")
                print(f"Active edges: {len(active_triplets)}")
                print(f"All edges (active + inactive): {len(triplets)}")

                print(f"\nActive edges:")
                for i, (s, r, o) in enumerate(active_triplets, 1):
                    print(f"  {i}. {s} --{r}--> {o}")

                print(f"\nAll edges (including inactive):")
                for i, (s, r, o) in enumerate(triplets, 1):
                    print(f"  {i}. {s} --{r}--> {o}")

                print(f"{'='*60}\n")

                res = get_support_contradict(amoc, target)
                support_count = len(res.get("support", []))
                contradict_count = len(res.get("contradict", []))
                diff = support_count - contradict_count

                rows.append({
                    "global_persona_index": global_index,
                    "age": age,
                    "persona_text": persona_text,
                    "item_id": item_id,
                    "condition": cond,
                    "support": support_count,
                    "contradict": contradict_count,
                    "difference": diff
                })
    except Exception as e:
        logger.error(f"Persona idx={idx} failed: {e}", exc_info=True)

    if getattr(_WORKER_CLIENT, "server_failures", 0) > server_failures_before:
        raise ServerUnavailableError(f"vllm server failed while processing persona idx={idx}; aborting run")
    return idx, rows


def main():
    parser = argparse.ArgumentParser(description="O'Brien experiment with personas (SLURM array compatible)")
    parser.add_argument("--persona-csv", required=True, help="CSV chunk file with personas (must have 'persona_text' and 'age_refined')")
    parser.add_argument("--obrien-json", required=True, help="JSON file with O'Brien items (combined)")
    parser.add_argument("--output-csv", required=True, help="Output CSV file for this chunk (will include all personas)")
    parser.add_argument("--stats-output", default=None, help="Optional file to save LME pairwise tables")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct", help="vLLM model name")
    parser.add_argument("--tp", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of personas (for testing)")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of O'Brien items (for testing)")
    parser.add_argument("--start-row", type=int, default=0, help="Skip the first N personas in this chunk")
    parser.add_argument("--server-url", default=None, help="Use a running vLLM OpenAI server (e.g. http://127.0.0.1:8000) instead of loading the model in-process")
    parser.add_argument("--workers", type=int, default=1, help="Personas processed in parallel; >1 requires --server-url")
    args = parser.parse_args()

    if args.workers > 1 and not args.server_url:
        raise SystemExit("--workers > 1 requires --server-url")

    # Load O'Brien files
    with open(args.obrien_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    logger.info(f"Loaded {len(items)} O'Brien items from {args.obrien_json}")
    if args.max_items:
        items = items[:args.max_items]

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

    tasks = []
    for idx, row in df.iterrows():
        age = int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
        tasks.append((idx, row.get("original_index", idx), age, str(row["persona_text"])))

    # Collect rows; saved after every persona so a failed job keeps its work
    all_rows = []

    def save(idx, rows):
        if not rows:
            return
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(args.output_csv, index=False)
        logger.info(f"Saved persona idx={idx} ({len(all_rows)} rows so far)")

    init_args = (args.model, args.tp, args.server_url, items)
    if args.workers > 1:
        logger.info(f"Running {len(tasks)} personas with {args.workers} workers against {args.server_url}")
        pool = ProcessPoolExecutor(
            max_workers=min(args.workers, len(tasks)) or 1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=init_args,
        )
        with pool:
            futures = [pool.submit(score_persona, *task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing personas"):
                try:
                    idx, rows = future.result()
                except (BrokenProcessPool, ServerUnavailableError):
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                save(idx, rows)
    else:
        _init_worker(*init_args)
        for task in tqdm(tasks, desc="Processing personas"):
            save(*score_persona(*task))

    # compute LME on the entire chunk
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(args.output_csv, index=False)
        print(f"Saved {len(df_out)} observations to {args.output_csv}")

        # ---- Means by Condition (Paper Format) ----
        print("\n" + "=" * 80)
        print("  MEANS BY CONDITION (Paper Format)")
        print("=" * 80)

        all_items_list = sorted(df_out['item_id'].unique())
        items_with_q1q2 = sorted(df_out[df_out['condition'].isin(['qualified1', 'qualified2'])]['item_id'].unique())
        items_with_q3q4 = sorted(df_out[df_out['condition'].isin(['qualified3', 'qualified4'])]['item_id'].unique())

        ci_support = df_out[df_out['condition'] == 'consistent']['support'].mean()
        ci_contradict = df_out[df_out['condition'] == 'consistent']['contradict'].mean()
        i_support = df_out[df_out['condition'] == 'inconsistent']['support'].mean()
        i_contradict = df_out[df_out['condition'] == 'inconsistent']['contradict'].mean()

        ciq12 = df_out[df_out['item_id'].isin(items_with_q1q2)]
        cq12_support = ciq12[ciq12['condition'] == 'consistent']['support'].mean()
        cq12_contradict = ciq12[ciq12['condition'] == 'consistent']['contradict'].mean()
        iq12_support = ciq12[ciq12['condition'] == 'inconsistent']['support'].mean()
        iq12_contradict = ciq12[ciq12['condition'] == 'inconsistent']['contradict'].mean()
        q1_support = ciq12[ciq12['condition'] == 'qualified1']['support'].mean()
        q1_contradict = ciq12[ciq12['condition'] == 'qualified1']['contradict'].mean()
        q2_support = ciq12[ciq12['condition'] == 'qualified2']['support'].mean()
        q2_contradict = ciq12[ciq12['condition'] == 'qualified2']['contradict'].mean()

        ciq34 = df_out[df_out['item_id'].isin(items_with_q3q4)]
        cq34_support = ciq34[ciq34['condition'] == 'consistent']['support'].mean()
        cq34_contradict = ciq34[ciq34['condition'] == 'consistent']['contradict'].mean()
        iq34_support = ciq34[ciq34['condition'] == 'inconsistent']['support'].mean()
        iq34_contradict = ciq34[ciq34['condition'] == 'inconsistent']['contradict'].mean()
        q3_support = ciq34[ciq34['condition'] == 'qualified3']['support'].mean()
        q3_contradict = ciq34[ciq34['condition'] == 'qualified3']['contradict'].mean()
        q4_support = ciq34[ciq34['condition'] == 'qualified4']['support'].mean()
        q4_contradict = ciq34[ciq34['condition'] == 'qualified4']['contradict'].mean()

        print(f"{'Type':<6} {'C/I Support':<15} {'C/I Contradict':<17} {'C/I/Q1/Q2 Support':<20} {'C/I/Q1/Q2 Contradict':<22} {'C/I/Q3/Q4 Support':<20} {'C/I/Q3/Q4 Contradict'}")
        print(f"{'C':<6} {ci_support:<15.2f} {ci_contradict:<17.2f} {cq12_support:<20.2f} {cq12_contradict:<22.2f} {cq34_support:<20.2f} {cq34_contradict:.2f}")
        print(f"{'I':<6} {i_support:<15.2f} {i_contradict:<17.2f} {iq12_support:<20.2f} {iq12_contradict:<22.2f} {iq34_support:<20.2f} {iq34_contradict:.2f}")
        print(f"{'Q1':<6} {'–':<15} {'–':<17} {q1_support:<20.2f} {q1_contradict:<22.2f} {'–':<20} {'–'}")
        print(f"{'Q2':<6} {'–':<15} {'–':<17} {q2_support:<20.2f} {q2_contradict:<22.2f} {'–':<20} {'–'}")
        print(f"{'Q3':<6} {'–':<15} {'–':<17} {'–':<20} {'–':<22} {q3_support:<20.2f} {q3_contradict:.2f}")
        print(f"{'Q4':<6} {'–':<15} {'–':<17} {'–':<20} {'–':<22} {q4_support:<20.2f} {q4_contradict:.2f}")

        print(f"\nItems with Q1/Q2: {len(items_with_q1q2)}")
        print(f"Items with Q3/Q4: {len(items_with_q3q4)}")

        # ---- LME for each metric ----
        all_stats = []
        for metric in ['support', 'contradict', 'difference']:
            print(f"\n{'='*60}")
            print(f"  LME for {metric.upper()}")
            print(f"{'='*60}")

            sub = df_out[df_out[metric].notna()].copy()

            cond_map = {
                'consistent': 'C',
                'inconsistent': 'I',
                'qualified1': 'Q1',
                'qualified2': 'Q2',
                'qualified3': 'Q3',
                'qualified4': 'Q4'
            }
            sub['cond_clean'] = sub['condition'].map(cond_map)
            sub['cond_clean'] = pd.Categorical(sub['cond_clean'])

            print(f"  Observations: {len(sub)}")
            print(f"  Unique items: {sub['item_id'].nunique()}")

            try:
                model = mixedlm(
                    f"{metric} ~ C(cond_clean, Treatment('C'))",
                    sub,
                    groups=sub['item_id']
                )
                result = model.fit()

                coefs = result.params
                vcov = result.cov_params()

                rows = []
                all_levels = sorted(sub['cond_clean'].unique())
                ref_level = 'C'

                for i, c1 in enumerate(all_levels):
                    for c2 in all_levels[i+1:]:
                        contrast = np.zeros(len(coefs))
                        param_names = coefs.index.tolist()

                        if c1 == ref_level:
                            contrast[0] = 1.0
                        else:
                            param_name = f"C(cond_clean, Treatment('{ref_level}'))[T.{c1}]"
                            if param_name in param_names:
                                contrast[param_names.index(param_name)] = 1.0
                            else:
                                continue

                        if c2 == ref_level:
                            contrast[0] -= 1.0
                        else:
                            param_name = f"C(cond_clean, Treatment('{ref_level}'))[T.{c2}]"
                            if param_name in param_names:
                                contrast[param_names.index(param_name)] -= 1.0
                            else:
                                continue

                        est = contrast @ coefs.values
                        se = np.sqrt(contrast @ vcov.values @ contrast)

                        if se > 0:
                            t_val = est / se
                            p_val = 2 * (1 - norm.cdf(abs(t_val)))

                            rows.append({
                                "Comparison": f"{c1} – {c2}",
                                "Estimate": round(est, 3),
                                "t": round(t_val, 3),
                                "p": round(p_val, 6),
                                "metric": metric
                            })

                if rows:
                    tbl = pd.DataFrame(rows)
                    tbl['sig'] = tbl['p'].apply(
                        lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ''))
                    )
                    print(tbl.to_string(index=False))
                    all_stats.extend(rows)
                else:
                    print("  No comparisons computed")

            except Exception as e:
                print(f"  LME failed: {e}")
                import traceback
                traceback.print_exc()

        # Save all stats
        if args.stats_output and all_stats:
            final_stats = pd.DataFrame(all_stats)
            final_stats.to_csv(args.stats_output, index=False)
            print(f"\nSaved LME pairwise tables to {args.stats_output}")


if __name__ == "__main__":
    main()
