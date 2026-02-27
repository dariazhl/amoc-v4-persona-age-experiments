import os
import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd

from amoc.pipeline import AgeAwareAMoCEngine
from amoc.llm.vllm_client import VLLMClient
from amoc.config.constants import DEBUG
from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.pipeline.io import (
    robust_read_persona_csv,
    get_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    infer_regime_from_filename,
)
from amoc.viz.graph_plots import plot_amoc_triplets

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
    "persona_text",
    "story_text",
    "model_name",
    "subject",
    "relation",
    "object",
    "sentence_index",
    "introduced_at",
    "regime",
    "active",
]

SENTENCE_CSV_HEADERS = [
    "original_index",
    "age_refined",
    "persona_text",
    "story_text",
    "model_name",
    "sentence_index",
    "introduced_at",
    "sentence_text",
    "subject",
    "relation",
    "object",
    "regime",
    "active",
    "anchor_kept",
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
    strict_attachament_constraint: bool = True,
    single_anchor_hub: bool = True,
    edge_visibility: Optional[int] = None,
    story_text: Optional[str] = None,
    force_node: bool = False,
    allow_multi_edges: bool = False,
    checkpoint: bool = False,
) -> None:
    short_filename = os.path.basename(filename)
    print(f"\n=== Processing File (chunk): {short_filename} ===")
    story_excerpt = story_snippet(story_text)

    # --- Path normalization (CRITICAL FIX) ---
    output_dir = Path(output_dir)

    # 1. Load data (entire chunk)
    df = robust_read_persona_csv(filename)

    # Ensure required columns
    if "persona_text" not in df.columns or "age_refined" not in df.columns:
        print(
            f"   [Skip] File {short_filename} missing "
            f"'persona_text' or 'age_refined' columns."
        )
        return

    regime = infer_regime_from_filename(filename)
    df["age_refined"] = pd.to_numeric(df["age_refined"], errors="coerce")

    # Optional row cap (debug / testing only)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    if df.empty:
        print(f"   [Skip] File {short_filename} has no valid rows.")
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
        cumulative_dir = output_dir / "cumulative"
        cumulative_dir.mkdir(parents=True, exist_ok=True)
        cumulative_output_filename = (
            f"model_{safe_model_name}_cumulative_triplets_{short_filename}"
        )
        cumulative_output_path = cumulative_dir / cumulative_output_filename
        active_dir = output_dir / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        sentence_output_filename = (
            f"model_{safe_model_name}_sentence_triplets_{short_filename}"
        )
        sentence_output_path = active_dir / sentence_output_filename
        final_output_filename = (
            f"model_{safe_model_name}_final_triplets_{short_filename}"
        )
        final_output_path = output_dir / final_output_filename

        ckpt_path = get_checkpoint_path(
            output_dir=str(output_dir),
            short_filename=short_filename,
            model_name=model_name,
        )

        ckpt = load_checkpoint(ckpt_path)
        processed_indices = set(ckpt.get("processed_indices", []))
        failures = ckpt.get("failures", [])
        personas_processed = ckpt.get("personas_processed", 0)

        print(f"[Model] {model_name}")
        print(f"  → Cumulative Output: {cumulative_output_path}")
        print(f"  → Per-sentence Output: {sentence_output_path}")
        print(f"  → Final active Output: {final_output_path}")
        print(f"  → Checkpoint: {ckpt_path}")

        if not cumulative_output_path.exists():
            pd.DataFrame([], columns=CSV_HEADERS).to_csv(
                cumulative_output_path, index=False, encoding="utf-8"
            )
        if not sentence_output_path.exists():
            pd.DataFrame([], columns=SENTENCE_CSV_HEADERS).to_csv(
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
                if row_idx in processed_indices:
                    continue

                persona_text = str(row["persona_text"])
                age_refined_int = (
                    int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
                )

                try:
                    final_triplets, sentence_triplets, cumulative_triplets = engine.run(
                        persona_text=persona_text,
                        age_refined=age_refined_int,
                        replace_pronouns=replace_pronouns,
                        plot_after_each_sentence=plot_after_each_sentence,
                        graphs_output_dir=graphs_output_dir,
                        highlight_nodes=highlight_nodes,
                        largest_component_only=plot_largest_component_only,
                        strict_reactivate_function=strict_reactivate_function,
                        strict_attachament_constraint=strict_attachament_constraint,
                        single_anchor_hub=single_anchor_hub,
                        story_text=story_text,
                        edge_visibility=edge_visibility,
                        matrix_dir_base=str(output_dir),
                        force_node=force_node,
                        allow_multi_edges=allow_multi_edges,
                        checkpoint=checkpoint,
                    )

                    records = []
                    for trip in final_triplets:
                        # Support legacy 3/4-tuple and new 6-tuple with intro/last-active
                        sentence_idx = -1
                        if len(trip) == 6:
                            s, r, o, active, introduced_at, last_active = trip
                            sentence_idx = introduced_at
                        elif len(trip) == 5:
                            s, r, o, active, sentence_idx = trip
                        elif len(trip) == 4:
                            s, r, o, active = trip
                        else:
                            s, r, o = trip
                            active = True
                        if not active:
                            continue
                        s, r, o = repair_triplet(s, r, o)
                        records.append(
                            {
                                "original_index": row_idx,
                                "age_refined": age_refined_int,
                                "persona_text": persona_text,
                                "story_text": story_excerpt,
                                "model_name": model_name,
                                "subject": s,
                                "relation": r,
                                "object": o,
                                "sentence_index": (
                                    int(sentence_idx)
                                    if sentence_idx is not None
                                    else -1
                                ),
                                "regime": regime,
                                "active": bool(active),
                            }
                        )

                    sentence_records = []
                    for trip in sentence_triplets:
                        if len(trip) == 8:
                            (
                                sent_idx,
                                sent_text,
                                s,
                                r,
                                o,
                                active,
                                anchor_kept,
                                introduced_at,
                            ) = trip
                        else:
                            # fallback: skip malformed
                            continue
                        s, r, o = repair_triplet(s, r, o)
                        sentence_records.append(
                            {
                                "original_index": row_idx,
                                "age_refined": age_refined_int,
                                "persona_text": persona_text,
                                "story_text": story_excerpt,
                                "model_name": model_name,
                                "sentence_index": int(sent_idx),
                                "introduced_at": int(introduced_at),
                                "sentence_text": sent_text,
                                "subject": s,
                                "relation": r,
                                "object": o,
                                "regime": regime,
                                "active": bool(active),
                                "anchor_kept": bool(anchor_kept),
                            }
                        )

                    if records:
                        # Deduplicate triplets (helps avoid repeats after pronoun replacement).
                        seen = set()
                        deduped = []
                        for rec in records:
                            key = (
                                rec["subject"],
                                rec["relation"],
                                rec["object"],
                                rec["active"],
                                rec["sentence_index"],
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            deduped.append(rec)
                        records = deduped

                    if cumulative_triplets:
                        cum_records = []
                        for trip in cumulative_triplets:
                            if len(trip) == 4:
                                s, r, o, introduced_at = trip
                            else:
                                continue
                            s, r, o = repair_triplet(s, r, o)
                            cum_records.append(
                                {
                                    "original_index": row_idx,
                                    "age_refined": age_refined_int,
                                    "persona_text": persona_text,
                                    "story_text": story_excerpt,
                                    "model_name": model_name,
                                    "subject": s,
                                    "relation": r,
                                    "object": o,
                                    "sentence_index": -1,
                                    "introduced_at": int(introduced_at),
                                    "regime": regime,
                                    "active": True,
                                }
                            )
                        seen_cum = set()
                        deduped_cum = []
                        for rec in cum_records:
                            key = (rec["subject"], rec["relation"], rec["object"])
                            if key in seen_cum:
                                continue
                            seen_cum.add(key)
                            deduped_cum.append(rec)
                        if deduped_cum:
                            pd.DataFrame(deduped_cum).to_csv(
                                cumulative_output_path,
                                mode="a",
                                header=False,
                                index=False,
                                columns=CSV_HEADERS,
                                encoding="utf-8",
                            )

                    if sentence_records:
                        seen_sent = set()
                        deduped_sent = []
                        for rec in sentence_records:
                            key = (
                                rec["sentence_index"],
                                rec["sentence_text"],
                                rec["subject"],
                                rec["relation"],
                                rec["object"],
                                rec["active"],
                                rec["anchor_kept"],
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
                            columns=SENTENCE_CSV_HEADERS,
                            encoding="utf-8",
                        )

                    # After processing all sentences for this persona, write final active edges
                    if final_triplets:
                        final_records = []
                        for trip in final_triplets:
                            # Expect shape (s, r, o, active, introduced_at, last_active)
                            if len(trip) == 6:
                                s, r, o, active, introduced_at, last_active = trip
                            elif len(trip) == 4:
                                s, r, o, active = trip
                                introduced_at = -1
                                last_active = -1
                            else:
                                # Skip malformed
                                continue
                            if not active:
                                continue
                            s, r, o = repair_triplet(s, r, o)
                            final_records.append(
                                {
                                    "original_index": row_idx,
                                    "age_refined": age_refined_int,
                                    "persona_text": persona_text,
                                    "story_text": story_excerpt,
                                    "model_name": model_name,
                                    "subject": s,
                                    "relation": r,
                                    "object": o,
                                    "sentence_index": (
                                        int(last_active)
                                        if last_active is not None
                                        else -1
                                    ),
                                    "introduced_at": int(introduced_at),
                                    "regime": regime,
                                    "active": True,
                                }
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
                                    rec["introduced_at"],
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
                                        existing["introduced_at"],
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
                                    rec["introduced_at"],
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

                    ckpt["personas_processed"] = personas_processed
                    ckpt["processed_indices"] = sorted(processed_indices)
                    save_checkpoint(ckpt_path, ckpt)

                    if plot_final_graph and records:
                        # Use cumulative graph for final plot to show full memory
                        trips = [
                            (t[0], t[1], t[2])
                            for t in cumulative_triplets
                            if len(t) >= 3
                        ]
                        if not trips:
                            trips = [
                                (rec["subject"], rec["relation"], rec["object"])
                                for rec in records
                                if include_inactive_edges or rec.get("active", True)
                            ]
                        if trips:
                            plot_dir = graphs_output_dir or os.path.join(
                                OUTPUT_ANALYSIS_DIR, "graphs"
                            )
                            plot_amoc_triplets(
                                triplets=trips,
                                persona=persona_text,
                                model_name=model_name,
                                age=age_refined_int,
                                blue_nodes=highlight_nodes,
                                output_dir=plot_dir,
                                step_tag="cumulative_graph_final",
                                largest_component_only=plot_largest_component_only,
                            )

                except Exception as e:
                    logging.error(
                        f"Failure idx={row_idx}, model={model_name}",
                        exc_info=True,
                    )
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
            ckpt["elapsed_seconds"] = time.time() - start_model_time
            ckpt["failures"] = failures
            save_checkpoint(ckpt_path, ckpt)

            print(
                f"[Model] {model_name}: processed {personas_processed} "
                f"personas from {short_filename}"
            )
