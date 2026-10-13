import os
import time
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
    "sentence_text",
    "sentence_index",
    "subject",
    "relation",
    "object",
    "subject_source",
    "object_source",
    "triplet_source",
    "edge_visibility",
    "edge_status",
    "decay_explanation",
    "context_shift",
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


def generate_reverse_plots_for_amoc(
    amoc,
    persona_text: str,
    age_refined_int: int,
    model_name: str,
    highlight_nodes,
    base_output_dir: str,
    persona_index,
) -> bool:
    if amoc is None:
        return False
    plotter = getattr(amoc, "_plot_ops", None)
    if plotter is None:
        return False
    produced = False
    try:
        all_states = plotter.get_graph_states()
        if len(all_states) >= 2:
            # One subdir per persona, keyed on the unique original_index.
            persona_dir = os.path.join(
                base_output_dir, f"persona_{persona_index}"
            )
            reverse_plotter = ReverseGraphPlotter(output_dir=persona_dir)
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
                s for s in all_states if "paper" in s.get("step_tag", "")
            ]
            if len(filtered_states) >= 2:
                png_paths = reverse_plotter.plot_reverse_sequence(
                    filtered_states, base_kwargs, final_positions, mode="paper"
                )
                produced = len(png_paths) > 0
                logging.info(
                    f"made {len(png_paths)} reverse plots for persona "
                    f"{persona_index} -> {os.path.join(persona_dir, 'reverse_plots')}"
                )
        # Free memory before the next persona regardless of state count.
        plotter.clear_graph_states()
    except Exception as e:
        logging.error(
            f"Failed to generate reverse PNGs for persona {persona_index}: {e}",
            exc_info=True,
        )
    return produced


def edge_records_to_dicts(edge_records) -> List[Dict[str, Any]]:
    record_dicts = []
    for record in edge_records:
        s, r, o = repair_triplet(record.subject, record.relation, record.object)
        record_dicts.append(
            {
                "original_index": record.original_index,
                "age_refined": record.age_refined,
                "regime": record.regime,
                "persona_text": record.persona_text,
                "model_name": record.model_name,
                "sentence_index": record.sentence_idx,
                "sentence_text": record.sentence_text,
                "subject": s,
                "relation": r,
                "object": o,
                "subject_source": record.subject_source,
                "object_source": record.object_source,
                "triplet_source": record.triplet_source,
                "edge_visibility": record.edge_visibility,
                "edge_status": record.edge_status,
                "decay_explanation": record.decay_explanation,
                "context_shift": record.context_shift,
            }
        )
    return record_dicts


def write_persona_records(
    record_dicts: List[Dict[str, Any]],
    sentence_output_path: Path,
    final_output_path: Path,
) -> None:
    if not record_dicts:
        return

    seen_sent = set()
    deduped_sent = []
    for rec in record_dicts:
        key = (
            rec["sentence_index"],
            rec["subject"],
            rec["relation"],
            rec["object"],
            rec["edge_status"],
        )
        if key in seen_sent:
            continue
        seen_sent.add(key)
        deduped_sent.append(rec)

    pd.DataFrame(deduped_sent).to_csv(
        sentence_output_path,
        mode="a",
        header=False,
        index=False,
        columns=CSV_HEADERS,
        encoding="utf-8",
    )

    max_sent_idx = max(rec["sentence_index"] for rec in deduped_sent)
    final_records = [
        rec
        for rec in deduped_sent
        if rec["edge_status"] in ("explicit", "asserted", "carryover")
        and rec["sentence_index"] == max_sent_idx
    ]
    if not final_records:
        return

    seen_final = set()
    deduped_final = []
    for rec in final_records:
        key = (
            rec["original_index"],
            rec["subject"],
            rec["relation"],
            rec["object"],
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


def run_single_persona(
    engine: AgeAwareAMoCEngine,
    row_idx,
    persona_text: str,
    age_refined_int: int,
    regime: str,
    model_name: str,
    run_kwargs: Dict[str, Any],
    generate_reverse_plots: bool,
    highlight_nodes: Optional[List[str]],
    reverse_plots_base_dir: str,
):
    engine.set_recorder_metadata(
        original_index=row_idx,
        age_refined=age_refined_int,
        regime=regime,
        persona_text=persona_text,
        model_name=model_name,
    )

    _, _, edge_records = engine.run(
        persona_text=persona_text,
        age_refined=age_refined_int,
        highlight_nodes=highlight_nodes,
        checkpoint=False,
        collect_plot_states=generate_reverse_plots,
        return_records=True,
        **run_kwargs,
    )
    record_dicts = edge_records_to_dicts(edge_records) if edge_records else []

    # Reverse plots are per-persona: generate them here, while
    # engine.last_amoc still holds THIS persona's run. Doing it
    # after the loop would only ever plot the final persona.
    made_reverse_plots = False
    if generate_reverse_plots:
        made_reverse_plots = generate_reverse_plots_for_amoc(
            amoc=engine.last_amoc,
            persona_text=persona_text,
            age_refined_int=age_refined_int,
            model_name=model_name,
            highlight_nodes=highlight_nodes,
            base_output_dir=reverse_plots_base_dir,
            persona_index=row_idx,
        )
    return record_dicts, made_reverse_plots


_WORKER_ENGINE: Optional[AgeAwareAMoCEngine] = None


def _init_persona_worker(model_name: str, server_url: str) -> None:
    global _WORKER_ENGINE
    from amoc.utils.spacy_utils import load_spacy

    client = VLLMClient(model_name=model_name, debug=DEBUG, server_url=server_url)
    _WORKER_ENGINE = AgeAwareAMoCEngine(vllm_client=client, spacy_nlp=load_spacy())


def _persona_worker_task(kwargs: Dict[str, Any]):
    return run_single_persona(engine=_WORKER_ENGINE, **kwargs)


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
    server_url: Optional[str] = None,
    workers: int = 1,
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

    parallel = workers > 1
    if parallel and not server_url:
        raise ValueError("workers > 1 requires a vLLM server URL (--server-url)")

    # 2. Initialize engines
    engines: Dict[str, Optional[AgeAwareAMoCEngine]] = {}

    for model_name in model_names:
        if parallel:
            engines[model_name] = None
            continue
        if model_name not in VLLM_CLIENT_CACHE:
            VLLM_CLIENT_CACHE[model_name] = VLLMClient(
                model_name=model_name,
                tp_size=tensor_parallel_size,
                debug=DEBUG,
                server_url=server_url,
            )

        engines[model_name] = AgeAwareAMoCEngine(
            vllm_client=VLLM_CLIENT_CACHE[model_name],
            spacy_nlp=spacy_nlp,
        )
    # 3. For each model, collect triplets into a table (incremental write)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_kwargs = dict(
        replace_pronouns=replace_pronouns,
        plot_after_each_sentence=plot_after_each_sentence,
        graphs_output_dir=graphs_output_dir,
        largest_component_only=plot_largest_component_only,
        strict_reactivate_function=strict_reactivate_function,
        single_anchor_hub=single_anchor_hub,
        story_text=story_text,
        edge_visibility=edge_visibility,
        matrix_dir_base=str(output_dir),
        force_node=force_node,
    )
    reverse_plots_base_dir = str(graphs_output_dir or output_dir)

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
        reverse_plot_folders = 0

        tasks = []
        for row_idx, row in df.iterrows():
            if checkpoint and row_idx in processed_indices:
                continue
            tasks.append(
                dict(
                    row_idx=row_idx,
                    persona_text=str(row["persona_text"]),
                    age_refined_int=(
                        int(row["age_refined"]) if pd.notna(row["age_refined"]) else -1
                    ),
                    regime=regime,
                    model_name=model_name,
                    run_kwargs=run_kwargs,
                    generate_reverse_plots=generate_reverse_plots,
                    highlight_nodes=highlight_nodes,
                    reverse_plots_base_dir=reverse_plots_base_dir,
                )
            )

        def on_success(task, record_dicts, made_reverse_plots):
            nonlocal personas_processed, reverse_plot_folders
            write_persona_records(record_dicts, sentence_output_path, final_output_path)
            if made_reverse_plots:
                reverse_plot_folders += 1
            personas_processed += 1
            processed_indices.add(task["row_idx"])
            if checkpoint:
                ckpt["personas_processed"] = personas_processed
                ckpt["processed_indices"] = sorted(processed_indices)
                save_checkpoint(ckpt_path, ckpt)

        def on_failure(task, e):
            logging.error(
                f"Failure idx={task['row_idx']}, model={model_name}",
                exc_info=e,
            )
            if checkpoint:
                failures.append(
                    {
                        "row_index": int(task["row_idx"]),
                        "persona_snippet": task["persona_text"][:80],
                        "error": str(e),
                        "time": datetime.utcnow().isoformat(),
                    }
                )
                ckpt["failures"] = failures
                save_checkpoint(ckpt_path, ckpt)

        try:
            if parallel:
                print(f"[{model_name}] running {len(tasks)} personas with {workers} workers")
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(tasks)) or 1,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_persona_worker,
                    initargs=(model_name, server_url),
                ) as pool:
                    futures = {
                        pool.submit(_persona_worker_task, task): task for task in tasks
                    }
                    for future in as_completed(futures):
                        task = futures[future]
                        try:
                            record_dicts, made_reverse_plots = future.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:
                            on_failure(task, e)
                            continue
                        on_success(task, record_dicts, made_reverse_plots)
                        print(
                            f"[{model_name}] persona {task['row_idx']} done "
                            f"({personas_processed}/{len(tasks)}, "
                            f"{format_seconds(time.time() - start_model_time)} elapsed)"
                        )
            else:
                for task in tasks:
                    try:
                        record_dicts, made_reverse_plots = run_single_persona(
                            engine=engine, **task
                        )
                    except Exception as e:
                        on_failure(task, e)
                        continue
                    on_success(task, record_dicts, made_reverse_plots)
        finally:
            # Reverse plots are now generated per-persona inside the loop above
            # (see generate_reverse_plots_for_amoc), so each persona gets its own
            # reverse_plots/persona_<idx>/ folder instead of only the last one.
            if generate_reverse_plots:
                logging.info(
                    f"{model_name}: produced reverse-plot folders for "
                    f"{reverse_plot_folders}/{personas_processed} personas "
                    f"under {graphs_output_dir or output_dir}/persona_<idx>/reverse_plots/"
                )
            if checkpoint:
                ckpt["elapsed_seconds"] = time.time() - start_model_time
                ckpt["failures"] = failures
                save_checkpoint(ckpt_path, ckpt)
            print(
                f"{model_name}: processed {personas_processed} "
                f"personas from {short_filename}"
            )
