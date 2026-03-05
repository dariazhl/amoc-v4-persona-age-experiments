from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Set
import os
import logging
import pandas as pd
import re

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node


class OutputFinalizer:
    def __init__(
        self,
        graph_ref: "Graph",
        model_name: str,
        persona: str,
        persona_age: Optional[int],
        story_text: str,
        matrix_dir_base: str,
    ):
        self._graph = graph_ref
        self._model_name = model_name
        self._persona = persona
        self._persona_age = persona_age
        self._story_text = story_text
        self._matrix_dir_base = matrix_dir_base

    def sanitize_filename_component(self, component: str, max_len: int = 80) -> str:
        component = (component or "").replace("\n", " ").strip()
        component = component[:max_len]
        component = re.sub(r"[\\/:*?\"<>|]", "_", component)
        component = re.sub(r"\s+", "_", component)
        return component or "unknown"

    # generates final output files:
    # 1. the activation matrix CSV
    # 2. the list of final active triplets
    # 3. the per‑sentence triplets from active graph
    # 4. the triplets from cumulative graph
    def finalize_outputs(
        self,
        amoc_matrix_records: List[Dict],
        triplet_intro: Dict[Tuple[str, str, str], int],
        explicit_nodes_current_sentence: Set["Node"],
        get_nodes_with_active_edges_fn: callable,
        reconstruct_semantic_triplets_fn: callable,
        current_sentence_index: Optional[int],
        sentence_triplets: List,
        matrix_suffix: Optional[str] = None,
    ) -> Tuple[List, List, List]:
        # Save score matrix
        df = pd.DataFrame(amoc_matrix_records)

        # Collapse duplicate token/sentence entries by mean to avoid pivot errors
        if not df.empty:
            df = (
                df.groupby(["token", "sentence"], as_index=False)["score"]
                .mean()
                .astype({"token": str})
            )
            matrix = (
                df.pivot(index="token", columns="sentence", values="score")
                .sort_index()
                .fillna(0.0)
            )
            salience_max = matrix.max(axis=1)
            salience_sum = matrix.sum(axis=1)
            ordering = (
                salience_max.to_frame("max")
                .assign(sum=salience_sum)
                .sort_values(by=["max", "sum", "token"], ascending=[False, False, True])
            )
            matrix = matrix.loc[ordering.index]

            if len(matrix.columns) > 0:
                story_row = pd.DataFrame(
                    [{col: "" for col in matrix.columns}], index=["story_text"]
                )
                story_row.iloc[0, 0] = self._story_text
                matrix = pd.concat([story_row, matrix])
        else:
            matrix = pd.DataFrame()

        matrix_dir = os.path.join(self._matrix_dir_base, "matrix")
        os.makedirs(matrix_dir, exist_ok=True)

        safe_model = self.sanitize_filename_component(self._model_name, max_len=60)
        safe_persona = self.sanitize_filename_component(self._persona, max_len=60)
        age_for_filename = self._persona_age if self._persona_age is not None else -1
        suffix = (
            f"_{self.sanitize_filename_component(matrix_suffix)}"
            if matrix_suffix
            else ""
        )
        matrix_filename = (
            f"amoc_matrix_{safe_model}_{safe_persona}_{age_for_filename}{suffix}.csv"
        )
        matrix_path = os.path.join(matrix_dir, matrix_filename)

        if not matrix.empty:
            matrix.to_csv(matrix_path)
            logging.info(
                "Saved activation matrix for persona '%s' to %s",
                self._persona,
                matrix_path,
            )
            logging.info("AMoC activation matrix:\n%s", matrix.to_string())

        final_sentence_idx = current_sentence_index
        final_triplets = []
        current_nodes = (
            explicit_nodes_current_sentence | get_nodes_with_active_edges_fn()
        )

        for subj, rel, obj in reconstruct_semantic_triplets_fn(
            only_active=True, restrict_nodes=current_nodes
        ):
            intro = triplet_intro.get((subj, rel, obj), -1)
            final_triplets.append(
                (
                    subj,
                    rel,
                    obj,
                    True,
                    int(intro),
                    int(final_sentence_idx) if final_sentence_idx else -1,
                )
            )

        cumulative_triplets = []
        for subj, rel, obj in reconstruct_semantic_triplets_fn():
            intro = triplet_intro.get((subj, rel, obj), -1)
            cumulative_triplets.append((subj, rel, obj, int(intro)))

        return final_triplets, sentence_triplets, cumulative_triplets
