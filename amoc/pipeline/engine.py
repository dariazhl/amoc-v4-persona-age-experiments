from typing import List, Tuple, Iterable, Optional

import pandas as pd

from amoc.pipeline.orchestrator import AMoCv4
from amoc.output.recorder import EdgeRecord
from amoc.llm.vllm_client import VLLMClient
from amoc.config.constants import (
    MAX_DISTANCE_FROM_ACTIVE_NODES,
    MAX_NEW_CONCEPTS,
    MAX_NEW_PROPERTIES,
    CONTEXT_LENGTH,
    EDGE_VISIBILITY,
    NR_RELEVANT_EDGES,
    DEBUG,
    STORY_TEXT,
)


class AgeAwareAMoCEngine:
    def __init__(self, vllm_client: VLLMClient, spacy_nlp):
        self.vllm_client = vllm_client
        self.spacy_nlp = spacy_nlp
        self.last_amoc: Optional["AMoCv4"] = None
        self._pending_metadata: Optional[dict] = None

    def build_analysis_text(self, persona_text: str, age_refined_int: int) -> str:
        return f"Age: {age_refined_int} years old.\n{persona_text}"

    def set_recorder_metadata(
        self,
        original_index: int,
        age_refined: int,
        regime: str,
        persona_text: str,
        model_name: str,
    ) -> None:
        self._pending_metadata = {
            "original_index": original_index,
            "age_refined": age_refined,
            "regime": regime,
            "persona_text": persona_text,
            "model_name": model_name,
        }

    def run(
        self,
        persona_text: str,
        age_refined,
        replace_pronouns: bool = True,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: str | None = None,
        highlight_nodes: Optional[Iterable[str]] = None,
        largest_component_only: bool = False,
        strict_reactivate_function: bool = True,
        single_anchor_hub: bool = True,
        edge_visibility: Optional[int] = None,
        story_text: Optional[str] = None,
        matrix_dir_base: Optional[str] = None,
        force_node: bool = False,
        checkpoint: bool = False,
        collect_plot_states: bool = False,
        return_records: bool = False,
    ) -> List[Tuple[str, str, str]]:

        try:
            age_refined_int = int(age_refined)
        except Exception:
            age_refined_int = int(float(age_refined)) if pd.notna(age_refined) else -1

        persona_description = self.build_analysis_text(persona_text, age_refined_int)

        amoc = AMoCv4(
            persona_description=persona_description,
            story_text=story_text if story_text is not None else STORY_TEXT,
            vllm_client=self.vllm_client,
            max_distance_from_active_nodes=MAX_DISTANCE_FROM_ACTIVE_NODES,
            max_new_concepts=MAX_NEW_CONCEPTS,
            max_new_properties=MAX_NEW_PROPERTIES,
            context_length=CONTEXT_LENGTH,
            edge_visibility=(
                edge_visibility if edge_visibility is not None else EDGE_VISIBILITY
            ),
            nr_relevant_edges=NR_RELEVANT_EDGES,
            spacy_nlp=self.spacy_nlp,
            debug=DEBUG,
            persona_age=age_refined_int,
            strict_reactivate_function=strict_reactivate_function,
            single_anchor_hub=single_anchor_hub,
            matrix_dir_base=matrix_dir_base,
            checkpoint=checkpoint,
        )
        self.last_amoc = amoc

        if self._pending_metadata:
            amoc.set_recorder_metadata(**self._pending_metadata)
            self._pending_metadata = None

        if collect_plot_states and hasattr(amoc, "_plot_ops"):
            amoc._plot_ops.enable_state_collection(True)
        final_triplets, sentence_triplets, cumulative_triplets = amoc.analyze(
            replace_pronouns=replace_pronouns,
            plot_after_each_sentence=plot_after_each_sentence,
            graphs_output_dir=graphs_output_dir,
            highlight_nodes=highlight_nodes,
            largest_component_only=largest_component_only,
            force_node=force_node,
        )

        edge_records = []
        if return_records:
            edge_records = amoc.get_edge_records()

        return final_triplets, sentence_triplets, edge_records
