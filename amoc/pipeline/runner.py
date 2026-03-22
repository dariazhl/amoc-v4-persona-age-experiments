import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd

from amoc.pipeline import AgeAwareAMoCEngine
from amoc.llm.vllm_client import VLLMClient
from amoc.config.constants import DEBUG
from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from datetime import datetime
from amoc.utils.io import (
    robust_read_persona_csv,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    infer_regime_from_filename,
)
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.viz.reverse_plotter import ReverseGraphPlotter

VLLM_CLIENT_CACHE: Dict[str, VLLMClient] = {}

CONTROL_TOKENS = {
    "|eot_id|",
    "<|eot_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<s>",
    "</s>",
    "assistant",
    "user",
    "system",
}

CSV_HEADERS = [
    "original_index",
    "age_refined",
    "regime",
    "persona_text",
    "model_name",
    "sentence_index",
    "introduced_at",
    "sentence_text",
    "subject",
    "relation",
    "object",
    "triplet_origin",
    "triplet_state",
    "carryover_decision",
]


def format_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def is_bad(x: str) -> bool:
    if not isinstance(x, str):
        return False
    return any(tok in x for tok in CONTROL_TOKENS)


def repair_triplet(e1: str, e2: str, e3: str):
    e1, e2, e3 = str(e1).strip(), str(e2).strip(), str(e3).strip()

    # Fix subject
    if is_bad(e1):
        e1 = e3 if not is_bad(e3) else "UNKNOWN"

    # Fix object
    if is_bad(e3):
        e3 = e1 if not is_bad(e1) else "UNKNOWN"

    # Fix relation
    if is_bad(e2) or e2 == "":
        e2 = "related_to"

    return e1, e2, e3


def story_snippet(story_text: Optional[str], max_words: int = 5) -> str:
    if not story_text:
        return ""
    words = str(story_text).split()
    return " ".join(words[:max_words])


def process_persona_csv(
    filename: str,
    model_names: List[str],
    spacy_nlp,
    output_dir: str,
    max_rows: Optional[int] = None,
    replace_pronouns: bool = False,
    tensor_parallel_size: int = 1,
    resume_only: bool = False,
    plot_after_each_sentence: bool = False,
    graphs_output_dir: Optional[str] = None,
    highlight_nodes: Optional[List[str]] = None,
    plot_final_graph: bool = False,
    plot_largest_component_only: bool = False,
    include_inactive_edges: bool = False,
    strict_reactivate_function: bool = True,
    single_anchor_hub: bool = True,
    edge_visibility: Optional[int] = None,
    story_text: Optional[str] = None,
    force_node: bool = False,
    checkpoint: bool = False,  # deprecated, used to save progress in analysis run
    generate_reverse_plots: bool = False,
    reverse_plot_mode: str = "paper",
) -> None:
    short_filename = os.path.basename(filename)
    print(f"Processing: {short_filename}")
    story_excerpt = story_snippet(story_text)

    output_dir = Path(output_dir)

    # 1. Load data (entire chunk)
    df = robust_read_persona_csv(filename)

    # Ensure required columns
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        print(
            f"Skipping {short_filename}: missing persona_text or age_refined columns."
        )
        return

    regime = infer_regime_from_filename(filename)
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")

    # Optional row cap
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if df.empty:
        print(f"Skipping {short_filename}: no valid rows.")
        return

    # 2. Initialize engines
    engines: Dict[str, AgeAwareAMoCEngine] = {}

    for model_name in model_names:
        if model_name not in VLLM_CLIENT_CACHE:
            VLLM_CLIENT_CACHE[model_name] = VLLMClient(
                model_name=model_name,
                tp_size=tensor_parallel_size,
                debug=DEBUG,
            )

        engines[model_name] = AgeAwareAMoCEngine(
            vllm_client=VLLM_CLIENT_CACHE[model_name],
            spacy_nlp=spacy_nlp,
        )
    # 3. For each model, collect triplets into a table (incremental write)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Process per model
    for model_name, engine in engines.items():
        safe_model_name = model_name.replace(":", "-").replace("/", "-")
        triplets_base = output_dir / "triplets"
        triplets_per_sentence_dir = triplets_base / "triplets_per_sentence"
        triplets_per_sentence_dir.mkdir(parents=True, exist_ok=True)
        sentence_output_filename = (
            f"model_{safe_model_name}_paper_sentence_triplets_{short_filename}"
        )
        sentence_output_path = triplets_per_sentence_dir / sentence_output_filename
        triplet_final_state_dir = triplets_base / "triplets_final_state"
        triplet_final_state_dir.mkdir(parents=True, exist_ok=True)
        final_output_filename = (
            f"model_{safe_model_name}_paper_final_triplets_{short_filename}"
        )
        final_output_path = triplet_final_state_dir / final_output_filename

        if checkpoint:
            ckpt_path = get_checkpoint_path(
                output_dir=str(output_dir),
                short_filename=short_filename,
                model_name=model_name,
            )
            ckpt = load_checkpoint(ckpt_path)
            processed_indices = set(ckpt.get("processed_indices", []))
            failures = ckpt.get("failures", [])
            personas_processed = ckpt.get("personas_processed", 0)
        else:
            ckpt_path = None
            ckpt = {}
            processed_indices = set()
            failures = []
            personas_processed = 0

        print(f"[{model_name}] out={final_output_path}")

        if not sentence_output_path.exists():
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                sentence_output_path, index=False, encoding="utf-8"
            )
        if not final_output_path.exists():
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                final_output_path, index=False, encoding="utf-8"
            )

        start_model_time = time.time()
        total_rows = len(df)

        try:
            for idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
                if checkpoint and row_idx in processed_indices:
                    continue

                persona_text = str(row["persona_text"])
                age_refined_int = (
                    int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
                )

                try:
                    final_triplets, sentence_triplets, _ = engine.run(
                        persona_text=persona_text,
                        age_refined=age_refined_int,
                        replace_pronouns=replace_pronouns,
                        plot_after_each_sentence=plot_after_each_sentence,
                        graphs_output_dir=graphs_output_dir,
                        highlight_nodes=highlight_nodes,
                        largest_component_only=plot_largest_component_only,
                        strict_reactivate_function=strict_reactivate_function,
                        single_anchor_hub=single_anchor_hub,
                        story_text=story_text,
                        edge_visibility=edge_visibility,
                        matrix_dir_base=str(output_dir),
                        force_node=force_node,
                        checkpoint=False,
                        collect_plot_states=generate_reverse_plots,
                    )

                    def _make_record(origin, state, s, r, o, sent_idx, sent_text, carryover_decision=""):
                        s, r, o = repair_triplet(s, r, o)
                        return {
                            "original_index": row_idx,
                            "age_refined": age_refined_int,
                            "persona_text": persona_text,
                            "model_name": model_name,
                            "sentence_index": int(sent_idx),
                            "introduced_at": -1,
                            "sentence_text": sent_text,
                            "subject": s,
                            "relation": r,
                            "object": o,
                            "regime": regime,
                            "triplet_origin": origin,
                            "triplet_state": state,
                            "carryover_decision": carryover_decision,
                        }

                    sentence_records = []
                    for record in sentence_triplets:
                        # SentenceTripletRecord dataclass
                        if hasattr(record, "sentence_index"):
                            sent_idx = record.sentence_index
                            sent_text = record.sentence_text

                            # Build lookup: triplet → reasoning from decay decisions
                            decay_lookup = {}
                            for dd in record.decay_decisions:
                                decay_lookup[dd.triplet] = dd.reasoning

                            for origin, state, triplet_list in [
                                ("text", "explicit", record.explicit_text_triplets),
                                ("inferred", "explicit", record.explicit_inferred_triplets),
                                ("text", "carryover", record.carryover_text_triplets),
                                ("inferred", "carryover", record.carryover_inferred_triplets),
                                ("text", "inactive", record.inactive_text_triplets),
                                ("inferred", "inactive", record.inactive_inferred_triplets),
                            ]:
                                for s, r, o in triplet_list:
                                    # Carryover/inactive get decay reasoning; explicit stays blank
                                    if state in ("carryover", "inactive"):
                                        decision = decay_lookup.get((s, r, o), "")
                                    else:
                                        decision = ""
                                    sentence_records.append(
                                        _make_record(
                                            origin, state, s, r, o,
                                            sent_idx, sent_text, decision,
                                        )
                                    )
                        else:
                            # Legacy 8-tuple fallback
                            if len(record) == 8:
                                (
                                    sent_idx,
                                    sent_text,
                                    s,
                                    r,
                                    o,
                                    active,
                                    _anchor_kept,
                                    introduced_at,
                                ) = record
                            else:
                                continue
                            rec = _make_record(
                                "text",
                                "explicit" if active else "inactive",
                                s, r, o, sent_idx, sent_text,
                            )
                            rec["introduced_at"] = int(introduced_at)
                            sentence_records.append(rec)

                    if sentence_records:
                        seen_sent = set()
                        deduped_sent = []
                        for rec in sentence_records:
                            key = (
                                rec["sentence_index"],
                                rec["subject"],
                                rec["relation"],
                                rec["object"],
                                rec["triplet_origin"],
                                rec["triplet_state"],
                            )
                            if key in seen_sent:
                                continue
                            seen_sent.add(key)
                            deduped_sent.append(rec)
                        sentence_records = deduped_sent

                        pd.DataFrame(sentence_records).to_csv(
                            sentence_output_path,
                            mode="a",
                            header=False,
                            index=False,
                            columns=CSV_HEADERS,
                            encoding="utf-8",
                        )

                    # Final state = last sentence's explicit + carryover triplets (no inactive)
                    final_records = (
                        [
                            rec
                            for rec in sentence_records
                            if rec["triplet_state"] in ("explicit", "carryover")
                            and rec["sentence_index"]
                            == max(r["sentence_index"] for r in sentence_records)
                        ]
                        if sentence_records
                        else []
                    )

                    if final_records:
                        seen_final = set()
                        deduped_final = []
                        for rec in final_records:
                            key = (
                                rec["original_index"],
                                rec["subject"],
                                rec["relation"],
                                rec["object"],
                                rec["triplet_origin"],
                            )
                            if key in seen_final:
                                continue
                            seen_final.add(key)
                            deduped_final.append(rec)

                        if final_output_path.exists():
                            existing = pd.read_csv(final_output_path)
                            existing_keys = set(
                                zip(
                                    existing["original_index"],
                                    existing["subject"],
                                    existing["relation"],
                                    existing["object"],
                                    existing["triplet_origin"],
                                )
                            )
                        else:
                            existing_keys = set()

                        deduped_final = [
                            rec
                            for rec in deduped_final
                            if (
                                rec["original_index"],
                                rec["subject"],
                                rec["relation"],
                                rec["object"],
                                rec["triplet_origin"],
                            )
                            not in existing_keys
                        ]

                        if deduped_final:
                            pd.DataFrame(deduped_final).to_csv(
                                final_output_path,
                                mode="a",
                                header=False,
                                index=False,
                                columns=CSV_HEADERS,
                                encoding="utf-8",
                            )

                    personas_processed += 1
                    processed_indices.add(row_idx)

                    if checkpoint:
                        ckpt["personas_processed"] = personas_processed
                        ckpt["processed_indices"] = sorted(processed_indices)
                        save_checkpoint(ckpt_path, ckpt)

                except Exception as e:
                    logging.error(
                        f"Failure idx={row_idx}, model={model_name}",
                        exc_info=True,
                    )
                    if checkpoint:
                        failures.append(
                            {
                                "row_index": int(row_idx),
                                "persona_snippet": persona_text[:80],
                                "error": str(e),
                                "time": datetime.utcnow().isoformat(),
                            }
                        )
                        ckpt["failures"] = failures
                        save_checkpoint(ckpt_path, ckpt)
        finally:
            # Generate reverse plots from the AMoC plotter's collected states
            if generate_reverse_plots and engine.last_amoc is not None:
                plotter = getattr(engine.last_amoc, "_plot_ops", None)
                if plotter is not None:
                    try:
                        all_states = plotter.get_graph_states()

                        if len(all_states) >= 2:
                            logging.info(
                                f"generating reverse plots from {len(all_states)} states"
                            )

                            reverse_plotter = ReverseGraphPlotter(
                                output_dir=graphs_output_dir or output_dir
                            )

                            final_positions = plotter.get_viz_positions()

                            base_kwargs = {
                                "persona": persona_text,
                                "model_name": model_name,
                                "age": age_refined_int,
                                "blue_nodes": highlight_nodes,
                                "avoid_edge_overlap": True,
                                "layout_depth": 3,
                                "show_triplet_overlay": True,
                            }

                            # Only paper mode is supported for reverse plots
                            filtered_states = [
                                s
                                for s in all_states
                                if "paper" in s.get("step_tag", "")
                            ]

                            if len(filtered_states) >= 2:
                                png_paths = reverse_plotter.plot_reverse_sequence(
                                    filtered_states,
                                    base_kwargs,
                                    final_positions,
                                    mode="paper",
                                )
                                logging.info(
                                    f"made {len(png_paths)} reverse plots for paper mode"
                                )

                            reverse_dir = os.path.join(
                                graphs_output_dir or output_dir, "reverse_plots"
                            )
                            logging.info(f"reverse plots saved in: {reverse_dir}")

                        plotter.clear_graph_states()

                    except Exception as e:
                        logging.error(
                            f"Failed to generate reverse PNGs: {e}", exc_info=True
                        )

            if checkpoint:
                ckpt["elapsed_seconds"] = time.time() - start_model_time
                ckpt["failures"] = failures
                save_checkpoint(ckpt_path, ckpt)
            print(
                f"{model_name}: processed {personas_processed} "
                f"personas from {short_filename}"
            )
