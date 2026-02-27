import logging
import os
import re
from typing import List, Tuple, Optional, Iterable
import pandas as pd
from spacy.tokens import Span, Token
import networkx as nx
import warnings
import copy
import random

from amoc.graph.node import NodeType
from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph.node import NodeProvenance, NodeRole
from amoc.graph.edge import Edge
from amoc.graph.per_sentence_graph import (
    PerSentenceGraph,
    PerSentenceGraphBuilder,
    build_per_sentence_graph,
)
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.llm.vllm_client import VLLMClient
from amoc.nlp.spacy_utils import (
    get_concept_lemmas,
    canonicalize_node_text,
    get_content_words_from_sent,
    extract_prepositional_objects,
    canonicalize_edge_label,
    is_adverb_token,
    are_semantically_equivalent,
    get_semantic_class,
)
from collections import deque
from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.prompts.amoc_prompts import FORCED_CONNECTIVITY_EDGE_PROMPT
import json

from amoc.pipeline.connectivity_ops import ConnectivityOps
from amoc.pipeline.decay_ops import DecayOps
from amoc.pipeline.text_filter_ops import TextFilterOps
from amoc.pipeline.triplet_ops import TripletOps
from amoc.pipeline.edge_ops import EdgeOps
from amoc.pipeline.node_ops import NodeOps
from amoc.pipeline.sentence_ops import SentenceOps
from amoc.pipeline.inference_ops import InferenceOps
from amoc.pipeline.linguistic_ops import LinguisticOps
from amoc.pipeline.plot_ops import PlotOps
from amoc.pipeline.activation_ops import ActivationOps
from amoc.pipeline.output_ops import OutputOps
from amoc.pipeline.sentence_processing_ops import SentenceProcessingOps
from amoc.pipeline.relationship_graph_ops import RelationshipGraphOps
from amoc.pipeline.init_ops import InitOps
from amoc.pipeline.projection_bookkeeping_ops import ProjectionBookkeepingOps

def _sanitize_filename_component(component: str, max_len: int = 80) -> str:
    component = (component or "").replace("\n", " ").strip()
    component = component[:max_len]
    component = re.sub(r"[\\/:*?\"<>|]", "_", component)
    component = re.sub(r"\s+", "_", component)
    return component or "unknown"

class AMoCv4:
    GENERIC_RELATION_LABELS = {
        "contains",
        "includes",
        "include",
        "contain",
        "refers to",
        "involves",
        "describes",
    }

    ENFORCE_ATTACHMENT_CONSTRAINT = True
    ACTIVATION_MAX_DISTANCE = 2
    RELATION_BLACKLIST = {"describes", "is_at_stake"}

    def __init__(
        self,
        persona_description: str,
        story_text: str,
        vllm_client: VLLMClient,
        max_distance_from_active_nodes: int,
        max_new_concepts: int,
        max_new_properties: int,
        context_length: int,
        edge_visibility: int,
        nr_relevant_edges: int,
        spacy_nlp,
        debug: bool = False,
        persona_age: Optional[int] = None,
        strict_reactivate_function: bool = True,
        strict_attachament_constraint: bool = True,
        single_anchor_hub: bool = True,
        matrix_dir_base: Optional[str] = None,
        allow_multi_edges: bool = False,
        checkpoint: bool = False,
    ) -> None:
        self.persona = persona_description
        self.story_text = story_text
        self.matrix_dir_base = matrix_dir_base or OUTPUT_ANALYSIS_DIR
        self.client = vllm_client
        self.model_name = vllm_client.model_name
        self.persona_age = persona_age

        if not isinstance(story_text, str) or not story_text.strip():
            raise ValueError("story_text must be a non-empty string")

        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.max_new_concepts = max_new_concepts
        self.max_new_properties = max_new_properties
        self.context_length = context_length
        self.edge_visibility = edge_visibility
        self.nr_relevant_edges = nr_relevant_edges

        self.graph = Graph()
        self.graph._debug_no_filter = True
        self.spacy_nlp = spacy_nlp

        if self.spacy_nlp is None:
            raise RuntimeError("AMoCv4 requires a spaCy nlp object (spacy_nlp).")

        self.debug = debug
        story_doc = self.spacy_nlp(story_text)
        self.story_lemmas = {tok.lemma_.lower() for tok in story_doc if tok.is_alpha}

        persona_doc = self.spacy_nlp(persona_description)
        self._persona_only_lemmas = {
            tok.lemma_.lower() for tok in persona_doc if tok.is_alpha
        } - self.story_lemmas
        self._prev_active_nodes_for_plot: set[Node] = set()
        self._cumulative_deactivated_nodes_for_plot: set[Node] = set()
        self._viz_positions: dict[str, tuple[float, float]] = {}
        self._recently_deactivated_nodes_for_inference: set[Node] = set()
        self._anchor_nodes: set[Node] = set()
        self._explicit_nodes_current_sentence: set[Node] = set()
        self._carryover_nodes_current_sentence: set[Node] = set()
        self.strict_reactivate_function = strict_reactivate_function
        self.strict_attachament_constraint = strict_attachament_constraint
        self.single_anchor_hub = single_anchor_hub
        self.allow_multi_edges = allow_multi_edges
        self.checkpoint = checkpoint
        self._current_sentence_text: str = ""
        self.cumulative_graph = nx.MultiDiGraph()
        self.active_graph = nx.MultiDiGraph()
        self._triplet_intro: dict[tuple[str, str, str], int] = {}
        self._cumulative_triplet_records: list[dict] = []
        self._fixed_hub = None
        self._per_sentence_view: Optional[PerSentenceGraph] = None
        self._ever_admitted_nodes: set[str] = set()
        self._layout_depth = 3
        self._persistent_is_edges: set[tuple[str, str, str]] = set()

        self._setup_ops_classes()

    def _setup_ops_classes(self):
        self._connectivity_ops = ConnectivityOps(
            graph_ref=self.graph,
            get_explicit_nodes=lambda: self._explicit_nodes_current_sentence,
            get_carryover_nodes=lambda: getattr(self, "_carryover_nodes_current_sentence", set()),
            edge_visibility=self.edge_visibility,
            client_ref=self.client,
        )
        self._decay_ops = DecayOps(
            graph_ref=self.graph,
            get_explicit_nodes=lambda: self._explicit_nodes_current_sentence,
            max_distance=self.max_distance_from_active_nodes,
        )
        self._text_filter_ops = TextFilterOps(
            spacy_nlp=self.spacy_nlp,
            graph_ref=self.graph,
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
        )
        self._triplet_ops = TripletOps(
            graph_ref=self.graph,
            cumulative_graph_ref=self.cumulative_graph,
            active_graph_ref=self.active_graph,
            triplet_intro_ref=self._triplet_intro,
        )
        self._edge_ops = EdgeOps(
            graph_ref=self.graph,
            client_ref=self.client,
            spacy_nlp=self.spacy_nlp,
            get_explicit_nodes=lambda: self._explicit_nodes_current_sentence,
            get_carryover_nodes=lambda: getattr(self, "_carryover_nodes_current_sentence", set()),
            get_attachable_nodes=lambda: self._get_attachable_nodes_for_sentence(),
            edge_visibility=self.edge_visibility,
            allow_multi_edges=self.allow_multi_edges,
            debug=self.debug,
        )
        self._edge_ops.set_inference_callbacks(
            normalize_endpoint_text_fn=self._normalize_endpoint_text,
            normalize_edge_label_fn=self._normalize_edge_label,
            is_valid_relation_label_fn=self._is_valid_relation_label,
            find_node_by_text_fn=self._find_node_by_text,
            add_edge_fn=self._add_edge,
            classify_relation_fn=self._classify_relation,
            persona=self.persona,
        )
        self._edge_ops.set_state_refs(
            triplet_intro=self._triplet_intro,
            persistent_is_edges=self._persistent_is_edges,
        )
        self._node_ops = NodeOps(
            graph_ref=self.graph,
            spacy_nlp=self.spacy_nlp,
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
            max_distance_from_active_nodes=self.max_distance_from_active_nodes,
            debug=self.debug,
        )
        self._node_ops.set_callbacks(
            has_active_attachment_fn=self._has_active_attachment,
            canonicalize_and_classify_fn=self._canonicalize_and_classify_node_text,
        )
        self._sentence_ops = SentenceOps(
            graph_ref=self.graph,
            spacy_nlp=self.spacy_nlp,
            story_lemmas=self.story_lemmas,
            max_distance_from_active_nodes=self.max_distance_from_active_nodes,
            edge_visibility=self.edge_visibility,
            strict_attachment_constraint=self.strict_attachament_constraint,
        )
        self._sentence_ops.set_state_refs(
            anchor_nodes=self._anchor_nodes,
            explicit_nodes=self._explicit_nodes_current_sentence,
            triplet_intro=self._triplet_intro,
        )
        self._inference_ops = InferenceOps(
            graph_ref=self.graph,
            client_ref=self.client,
            spacy_nlp=self.spacy_nlp,
            max_new_concepts=self.max_new_concepts,
            max_new_properties=self.max_new_properties,
            persona=self.persona,
        )
        self._inference_ops.set_callbacks(
            append_adjectival_hints_fn=self._append_adjectival_hints,
            get_sentences_text_based_nodes_fn=lambda sents, create_unexistent_nodes=True: self.get_senteces_text_based_nodes(
                sents, create_unexistent_nodes=create_unexistent_nodes
            ),
        )
        self._linguistic_ops = LinguisticOps(
            graph_ref=self.graph,
            spacy_nlp=self.spacy_nlp,
            client_ref=self.client,
            story_lemmas=self.story_lemmas,
        )
        self._linguistic_ops.set_callbacks(
            add_edge_fn=self._add_edge,
            classify_relation_fn=self._classify_relation,
        )
        self._plot_ops = PlotOps(
            graph_ref=self.graph,
            output_dir=self.matrix_dir_base,
            model_name=self.model_name,
            persona=self.persona,
            persona_age=self.persona_age,
            layout_depth=self._layout_depth,
            allow_multi_edges=self.allow_multi_edges,
        )
        self._plot_ops.set_callbacks(
            get_explicit_nodes_fn=lambda: self._explicit_nodes_current_sentence,
            get_edge_activation_scores_fn=self._get_edge_activation_scores,
            graph_edges_to_triplets_fn=self._graph_edges_to_triplets,
            enforce_cumulative_connectivity_fn=self._enforce_cumulative_connectivity,
        )
        self._plot_ops.set_lemmas(
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
        )
        self._activation_ops = ActivationOps(
            graph_ref=self.graph,
            client_ref=self.client,
            get_explicit_nodes=lambda: self._explicit_nodes_current_sentence,
            max_distance=self.max_distance_from_active_nodes,
            edge_visibility=self.edge_visibility,
            nr_relevant_edges=self.nr_relevant_edges,
            strict_reactivate=self.strict_reactivate_function,
        )
        self._activation_ops.set_state_refs(
            anchor_nodes=self._anchor_nodes,
            record_edge_fn=self._record_edge_in_graphs,
        )
        self._output_ops = OutputOps(
            graph_ref=self.graph,
            model_name=self.model_name,
            persona=self.persona,
            persona_age=self.persona_age,
            story_text=self.story_text,
            matrix_dir_base=self.matrix_dir_base,
        )
        self._relationship_graph_ops = RelationshipGraphOps(
            graph_ref=self.graph,
            spacy_nlp=self.spacy_nlp,
            edge_visibility=self.edge_visibility,
            debug=self.debug,
        )
        self._init_ops = InitOps(
            graph_ref=self.graph,
            client_ref=self.client,
            spacy_nlp=self.spacy_nlp,
            edge_visibility=self.edge_visibility,
            debug=self.debug,
        )
        self._sentence_processing_ops = SentenceProcessingOps(
            graph_ref=self.graph,
            client_ref=self.client,
            spacy_nlp=self.spacy_nlp,
            max_distance_from_active_nodes=self.max_distance_from_active_nodes,
            edge_visibility=self.edge_visibility,
            context_length=self.context_length,
            debug=self.debug,
        )
        self._relationship_graph_ops.set_callbacks(
            normalize_endpoint_text_fn=self._normalize_endpoint_text,
            normalize_edge_label_fn=self._normalize_edge_label,
            is_valid_relation_label_fn=self._is_valid_relation_label,
            validate_node_provenance_fn=self._validate_node_provenance,
            admit_node_fn=self._admit_node,
            passes_attachment_constraint_fn=self._passes_attachment_constraint,
            canonicalize_edge_direction_fn=self._canonicalize_edge_direction,
            canonicalize_and_classify_node_text_fn=self._canonicalize_and_classify_node_text,
            classify_relation_fn=self._classify_relation,
            add_edge_fn=self._add_edge,
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges,
            get_node_from_text_fn=self.get_node_from_text,
            get_node_from_new_relationship_fn=self.get_node_from_new_relationship,
            get_concept_lemmas_fn=lambda text: get_concept_lemmas(self.spacy_nlp, text),
            appears_in_story_fn=self._appears_in_story,
        )
        self._relationship_graph_ops.set_state_refs(
            explicit_nodes_ref=lambda: self._explicit_nodes_current_sentence,
        )
        self._init_ops.set_callbacks(
            normalize_endpoint_text_fn=self._normalize_endpoint_text,
            normalize_edge_label_fn=self._normalize_edge_label,
            is_valid_relation_label_fn=self._is_valid_relation_label,
            passes_attachment_constraint_fn=self._passes_attachment_constraint,
            canonicalize_edge_direction_fn=self._canonicalize_edge_direction,
            classify_relation_fn=self._classify_relation,
            add_edge_fn=self._add_edge,
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges,
            get_node_from_text_fn=self.get_node_from_text,
            get_sentences_text_based_nodes_fn=lambda sents, create_unexistent_nodes=True: self.get_senteces_text_based_nodes(
                sents, create_unexistent_nodes=create_unexistent_nodes
            ),
            extract_deterministic_structure_fn=self._extract_deterministic_structure,
            infer_new_relationships_step_0_fn=self.infer_new_relationships_step_0,
            add_inferred_relationships_to_graph_step_0_fn=self.add_inferred_relationships_to_graph_step_0,
            restrict_active_to_current_explicit_fn=self._restrict_active_to_current_explicit,
        )
        self._init_ops.set_state_refs(
            explicit_nodes_ref=lambda: self._explicit_nodes_current_sentence,
            persona=self.persona,
        )
        self._sentence_processing_ops.set_callbacks(
            normalize_endpoint_text_fn=self._normalize_endpoint_text,
            normalize_edge_label_fn=self._normalize_edge_label,
            is_valid_relation_label_fn=self._is_valid_relation_label,
            passes_attachment_constraint_fn=self._passes_attachment_constraint,
            canonicalize_edge_direction_fn=self._canonicalize_edge_direction,
            classify_relation_fn=self._classify_relation,
            add_edge_fn=self._add_edge,
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges,
            extract_adjectival_modifiers_fn=self._extract_adjectival_modifiers,
            append_adjectival_hints_fn=self._append_adjectival_hints,
            extract_deterministic_structure_fn=self._extract_deterministic_structure,
            infer_edges_to_recently_deactivated_fn=self._infer_edges_to_recently_deactivated,
            propagate_activation_from_edges_fn=self._propagate_activation_from_edges,
            restrict_active_to_current_explicit_fn=self._restrict_active_to_current_explicit,
            get_node_from_new_relationship_fn=self.get_node_from_new_relationship,
            get_phrase_level_concepts_fn=self.get_phrase_level_concepts,
            get_sentences_text_based_nodes_fn=lambda sents, create_unexistent_nodes=True: self.get_senteces_text_based_nodes(
                sents, create_unexistent_nodes=create_unexistent_nodes
            ),
            infer_new_relationships_fn=self.infer_new_relationships,
            add_inferred_relationships_to_graph_fn=self.add_inferred_relationships_to_graph,
            reactivate_relevant_edges_fn=self.reactivate_relevant_edges,
        )
        self._sentence_processing_ops.set_state_refs(
            explicit_nodes_ref=lambda: self._explicit_nodes_current_sentence,
            anchor_nodes_ref=self._anchor_nodes,
            triplet_intro_ref=self._triplet_intro,
            carryover_nodes_ref=lambda: getattr(self, "_carryover_nodes_current_sentence", set()),
            persona=self.persona,
        )
        self._projection_bookkeeping_ops = ProjectionBookkeepingOps(
            graph_ref=self.graph,
            max_distance=self.max_distance_from_active_nodes,
            enforce_attachment_constraint=self.ENFORCE_ATTACHMENT_CONSTRAINT,
            debug=self.debug,
        )
        self._projection_bookkeeping_ops.set_callbacks(
            record_sentence_activation_fn=self._record_sentence_activation,
        )

    def _rebind_ops_graph_refs(self) -> None:
        for ops in [
            self._connectivity_ops,
            self._decay_ops,
            self._text_filter_ops,
            self._triplet_ops,
            self._edge_ops,
            self._node_ops,
            self._sentence_ops,
            self._inference_ops,
            self._linguistic_ops,
            self._plot_ops,
            self._activation_ops,
            self._output_ops,
            self._relationship_graph_ops,
            self._init_ops,
            self._sentence_processing_ops,
            self._projection_bookkeeping_ops,
        ]:
            if hasattr(ops, "_graph"):
                ops._graph = self.graph
            if hasattr(ops, "graph"):
                ops.graph = self.graph

    def _classify_relation(self, label: str) -> str:
        return self._text_filter_ops.classify_relation(label)

    def init_graph(self, sent: Span) -> None:
        self._init_ops.init_graph(sent)

    def repair_connectivity_callback(
        self,
        components,
        active_nodes,
        active_edges,
        sentence_index,
        temperature: float = 0.3,
        forced_pair=None,
    ):
        self._connectivity_ops.set_context(
            story_text=self.story_text,
            current_sentence_text=self._current_sentence_text,
        )
        return self._connectivity_ops.repair_connectivity_callback(
            components=components,
            active_nodes=active_nodes,
            active_edges=active_edges,
            sentence_index=sentence_index,
            temperature=temperature,
            forced_pair=forced_pair,
        )

    def _admit_node(
        self,
        lemma: str,
        node_type: NodeType,
        provenance: str,
        sent: Optional[Span] = None,
    ) -> bool:
        return self._node_ops.admit_node(lemma, node_type, provenance, sent)

    def _node_token_for_matrix(self, node: Node) -> str:
        return (node.get_text_representer() or "").strip().lower()

    def _validate_node_provenance(
        self,
        lemma: str,
        current_sentence_text: Optional[str] = None,
        *,
        allow_bootstrap: bool = False,
    ) -> bool:
        return self._node_ops.validate_node_provenance(
            lemma, current_sentence_text, allow_bootstrap=allow_bootstrap
        )

    def _build_per_sentence_view(
        self,
        explicit_nodes: List[Node],
        sentence_index: int,
    ) -> Optional[PerSentenceGraph]:
        view = self._sentence_ops.build_per_sentence_view(
            explicit_nodes=explicit_nodes,
            sentence_index=sentence_index,
            build_per_sentence_graph_fn=build_per_sentence_graph,
        )
        self._per_sentence_view = view
        return view

    def _get_attachable_nodes_for_sentence(self) -> set[Node]:
        return self._sentence_ops.get_attachable_nodes_for_sentence(
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges
        )

    def _extract_adjectival_modifiers(self, sent: Span) -> dict[str, list[str]]:
        return self._linguistic_ops.extract_adjectival_modifiers(sent)

    def _append_adjectival_hints(self, nodes_from_text: str, sent: Span) -> str:
        return self._linguistic_ops.append_adjectival_hints(nodes_from_text, sent)

    def _distances_from_sources_active_edges(
        self, sources: set[Node], max_distance: int
    ) -> dict[Node, int]:
        return self._activation_ops.distances_from_sources_active_edges(
            sources, max_distance
        )

    def _record_sentence_activation(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        self._activation_ops.record_sentence_activation_matrix(
            sentence_id=sentence_id,
            explicit_nodes=explicit_nodes,
            newly_inferred_nodes=newly_inferred_nodes,
            max_distance=self.ACTIVATION_MAX_DISTANCE,
            node_token_fn=self._node_token_for_matrix,
            append_record_fn=self._amoc_matrix_records.append,
        )

    def _infer_edges_to_recently_deactivated(
        self,
        current_sentence_nodes: List[Node],
        current_sentence_words: List[str],
        current_text: str,
    ) -> List[Edge]:
        return self._edge_ops.infer_edges_to_recently_deactivated(
            current_sentence_nodes=current_sentence_nodes,
            current_sentence_words=current_sentence_words,
            current_text=current_text,
            recently_deactivated_nodes=self._recently_deactivated_nodes_for_inference,
            enforce_attachment=self.ENFORCE_ATTACHMENT_CONSTRAINT,
        )

    def _passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
        allow_inference_bridge: bool = False,
    ) -> bool:
        return self._node_ops.passes_attachment_constraint(
            subject=subject,
            obj=obj,
            current_sentence_words=current_sentence_words,
            current_sentence_nodes=current_sentence_nodes,
            graph_active_nodes=graph_active_nodes,
            explicit_nodes=set(self._explicit_nodes_current_sentence),
            carryover_nodes=getattr(self, "_carryover_nodes_current_sentence", set()),
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges,
            graph_active_edge_nodes=graph_active_edge_nodes,
            allow_inference_bridge=allow_inference_bridge,
        )

    def _add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
        bypass_attachment_constraint: bool = False,
        skip_event_mediation: bool = False,
        relation_class=None,
        justification=None,
        persona_influenced: bool = False,
    ) -> Optional[Edge]:
        self._edge_ops.set_current_sentence(self._current_sentence_index)
        edge = self._edge_ops.add_edge(
            source_node=source_node,
            dest_node=dest_node,
            label=label,
            edge_forget=edge_forget,
            created_at_sentence=created_at_sentence,
            bypass_attachment_constraint=bypass_attachment_constraint,
            skip_event_mediation=skip_event_mediation,
            relation_class=relation_class,
            justification=justification,
            persona_influenced=persona_influenced,
        )
        if edge:
            self._record_edge_in_graphs(edge, self._current_sentence_index)
        return edge

    def _create_forced_connectivity_edges(
        self,
        story_context=None,
        current_sentence=None,
        mode="active",
    ):
        return self._edge_ops.create_forced_connectivity_edges(
            story_context=story_context,
            current_sentence=current_sentence,
            mode=mode,
            persona=self.persona,
            normalize_edge_label_fn=self._normalize_edge_label,
        )

    def resolve_pronouns(self, text: str) -> str:
        resolved = self.client.resolve_pronouns(text, self.persona)
        if not isinstance(resolved, str) or not resolved.strip():
            return text
        low = resolved.lower()
        if "does not mention any pronouns" in low or "no pronouns to replace" in low:
            return text
        return resolved

    def _graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
        return self._triplet_ops.graph_edges_to_triplets(only_active=only_active)

    def _reconstruct_semantic_triplets(
        self,
        *,
        only_active: bool = False,
        restrict_nodes: Optional[set[Node]] = None,
    ):
        return self._triplet_ops.reconstruct_semantic_triplets(
            only_active=only_active, restrict_nodes=restrict_nodes
        )

    def _restrict_active_to_current_explicit(self, explicit_nodes):
        self._activation_ops.restrict_active_to_current_explicit(explicit_nodes)

    def _get_nodes_with_active_edges(self) -> set[Node]:
        return self._connectivity_ops._get_nodes_with_active_edges()

    def _has_active_attachment(self, lemma: str) -> bool:
        active_nodes = {n for n in self.graph.nodes if n.active}
        active_nodes |= set(self._explicit_nodes_current_sentence)
        return bool(active_nodes)

    def _normalize_label(self, label: str) -> str:
        norm = (label or "").strip().lower()
        norm = re.sub(r"[\s\-]+", "_", norm)
        return norm

    def _edge_key(self, edge: Edge) -> tuple[str, str, str]:
        return (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )

    def _get_edge_activation_scores(self) -> dict[tuple[str, str, str], int]:
        return self._triplet_ops.get_edge_activation_scores()

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        self._edge_ops.record_edge_in_graphs(
            edge=edge,
            sentence_idx=sentence_idx,
            cumulative_graph=self.cumulative_graph,
            active_graph=self.active_graph,
            cumulative_triplet_records=self._cumulative_triplet_records,
        )

    def _is_generic_relation(self, label: str) -> bool:
        return self._text_filter_ops.is_generic_relation(label)

    def _is_blacklisted_relation(self, label: str) -> bool:
        return self._text_filter_ops.is_blacklisted_relation(label)

    def _is_verb_relation(self, label: str) -> bool:
        return self._text_filter_ops.is_verb_relation(label)

    def _canonicalize_edge_direction(
        self, label: str, source_text: str, dest_text: str
    ) -> tuple[str, str, str, bool]:
        return self._text_filter_ops.canonicalize_edge_direction(
            label, source_text, dest_text
        )

    def _normalize_edge_label(self, label: str) -> str:
        return self._text_filter_ops.normalize_edge_label(label)

    def _is_valid_relation_label(self, label: str) -> bool:
        return self._text_filter_ops.is_valid_relation_label(label)

    def _normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        return self._text_filter_ops.normalize_endpoint_text(text, is_subject)

    def _has_edge_between(
        self, a: Node, b: Node, relation_lemma: Optional[str] = None
    ) -> bool:
        return self._edge_ops.has_edge_between(a, b, relation_lemma)

    def _get_existing_edge_between_nodes(
        self, source_node: Node, dest_node: Node
    ) -> Optional[Edge]:
        return self._edge_ops.get_existing_edge_between_nodes(source_node, dest_node)

    def _find_node_by_text(
        self, text: str, candidates: Iterable[Node]
    ) -> Optional[Node]:
        return self._node_ops.find_node_by_text(text, candidates)

    def _appears_in_story(self, text: str, *, check_graph: bool = False) -> bool:
        return self._text_filter_ops.appears_in_story(text, check_graph=check_graph)

    def _classify_canonical_node_text(self, canon: str) -> Optional[NodeType]:
        return self._text_filter_ops.classify_canonical_node_text(canon)

    def _canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional[NodeType]]:
        return self._text_filter_ops.canonicalize_and_classify_node_text(text)

    def _enforce_cumulative_connectivity(self):
        self._connectivity_ops.enforce_cumulative_connectivity_simple()

    def _plot_graph_snapshot(
        self,
        sentence_index: int,
        sentence_text: str,
        output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        only_active: bool = False,
        largest_component_only: bool = False,
        mode: str = "sentence_active",
        triplets_override: Optional[List[Tuple[str, str, str]]] = None,
        active_edges: Optional[set[tuple[str, str]]] = None,
        explicit_nodes: Optional[List[str]] = None,
        salient_nodes: Optional[List[str]] = None,
        inactive_nodes: Optional[List[str]] = None,
        active_triplets_for_overlay: Optional[List[Tuple[str, str, str]]] = None,
        property_nodes: Optional[List[str]] = None,
    ) -> None:
        self._plot_ops.plot_graph_snapshot_full(
            sentence_index=sentence_index,
            sentence_text=sentence_text,
            output_dir=output_dir,
            highlight_nodes=highlight_nodes,
            only_active=only_active,
            largest_component_only=largest_component_only,
            mode=mode,
            triplets_override=triplets_override,
            active_edges=active_edges,
            explicit_nodes=explicit_nodes,
            salient_nodes=salient_nodes,
            inactive_nodes=inactive_nodes,
            active_triplets_for_overlay=active_triplets_for_overlay,
            property_nodes=property_nodes,
        )
        self._viz_positions = self._plot_ops.viz_positions

    def _decay_node_activation(self):
        self._activation_ops.decay_node_activation()

    def _initialize_run_state(self) -> None:
        if not hasattr(self, "_amoc_matrix_records"):
            self._amoc_matrix_records = []
        self._previous_active_triplets = []
        if not hasattr(self, "_viz_positions") or self._viz_positions is None:
            self._viz_positions = {}
        self._projection_bookkeeping_ops.reset_state()

    def _resolve_sentences(self, replace_pronouns: bool) -> list:
        resolved_sentences, story_lemma_set = self._sentence_ops.resolve_sentences(
            story_text=self.story_text,
            replace_pronouns=replace_pronouns,
            resolve_pronouns_fn=self.resolve_pronouns,
        )
        self._story_lemma_set = story_lemma_set
        return resolved_sentences

    def _snapshot_sentence_state(self) -> tuple:
        return self._sentence_ops.snapshot_sentence_state(
            anchor_nodes=self._anchor_nodes,
            triplet_intro=self._triplet_intro,
            per_sentence_view=getattr(self, "_per_sentence_view", None),
            recently_deactivated=getattr(self, "_recently_deactivated_nodes_for_inference", None),
            prev_active_nodes=getattr(self, "_prev_active_nodes_for_plot", None),
        )

    def _reset_sentence_state(self, original_text: str) -> set:
        nodes_before_sentence = self._sentence_ops.reset_sentence_state(original_text)
        self._current_sentence_text = original_text
        self._anchor_drop_log: list[tuple[int, str, str, str, str]] = []
        self._explicit_nodes_current_sentence.clear()
        return nodes_before_sentence

    def _handle_first_sentence(
        self,
        sent,
        resolved_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
    ) -> tuple:
        result = self._init_ops.handle_first_sentence(
            sent=sent,
            resolved_text=resolved_text,
            prev_sentences=prev_sentences,
            nodes_before_sentence=nodes_before_sentence,
        )
        nodes_before, should_skip, explicit_nodes, anchor_nodes = result
        self._explicit_nodes_current_sentence.clear()
        self._explicit_nodes_current_sentence.update(explicit_nodes)
        self._anchor_nodes.clear()
        self._anchor_nodes.update(anchor_nodes)
        return (nodes_before, should_skip)

    def _handle_nonfirst_sentence(
        self,
        i: int,
        sent,
        resolved_text: str,
        original_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
    ) -> tuple:
        return self._sentence_processing_ops.handle_nonfirst_sentence(
            i=i,
            sent=sent,
            resolved_text=resolved_text,
            original_text=original_text,
            prev_sentences=prev_sentences,
            nodes_before_sentence=nodes_before_sentence,
            current_sentence_index=self._current_sentence_index,
            current_sentence_text=resolved_text,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            anchor_nodes=self._anchor_nodes,
            triplet_intro=self._triplet_intro,
        )

    def _apply_global_edge_decay(self):
        self._activation_ops.apply_global_edge_decay()

    def _process_sentence_core(
        self,
        i: int,
        sent,
        resolved_text: str,
        original_text: str,
        prev_sentences: list,
    ) -> tuple:
        self._new_inferred_nodes_count = 0
        nodes_before_sentence = self._reset_sentence_state(original_text)
        logging.info("Processing sentence %d: %s", i, resolved_text)
        if resolved_text.strip().startswith("{"):
            logging.error(
                "LLM JSON contamination detected — reverting to original sentence."
            )
            resolved_text = original_text
            sent = self.spacy_nlp(original_text)[0 : len(self.spacy_nlp(original_text))]
        if i == 0:
            return self._handle_first_sentence(
                sent, resolved_text, prev_sentences, nodes_before_sentence
            )
        else:
            result = self._handle_nonfirst_sentence(
                i,
                sent,
                resolved_text,
                original_text,
                prev_sentences,
                nodes_before_sentence,
            )
            if not result[1]:
                self._apply_global_edge_decay()

                self.graph.enforce_cumulative_stability(
                    set(self._explicit_nodes_current_sentence)
                )
                self._decay_node_activation()
                self.graph.enforce_carryover_connectivity(
                    getattr(self, "_carryover_nodes_current_sentence", set())
                )
                if not self.graph.is_active_connected():
                    if self._enforce_connectivity(prev_sentences):
                        return (nodes_before_sentence, True)

            return result

    def _finalize_outputs(self, matrix_suffix: Optional[str]) -> tuple:
        return self._output_ops.finalize_outputs(
            amoc_matrix_records=self._amoc_matrix_records,
            triplet_intro=self._triplet_intro,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            get_nodes_with_active_edges_fn=self._get_nodes_with_active_edges,
            reconstruct_semantic_triplets_fn=self._reconstruct_semantic_triplets,
            current_sentence_index=getattr(self, "_current_sentence_index", None),
            sentence_triplets=self._sentence_triplets,
            matrix_suffix=matrix_suffix,
        )

    def _is_active_connected(self) -> bool:
        return self._connectivity_ops.is_active_connected()

    def _is_cumulative_connected(self) -> bool:
        return self._connectivity_ops.is_cumulative_connected()

    def _validate_sentence_state(self) -> bool:
        return self._connectivity_ops.validate_sentence_state()

    def _enforce_connectivity(self, prev_sentences: list) -> bool:
        rollback_needed = self._connectivity_ops.enforce_connectivity(
            prev_sentences=prev_sentences,
            current_sentence_text=self._current_sentence_text,
            create_forced_edges_fn=self._create_forced_connectivity_edges,
        )

        self._per_sentence_view = self._build_per_sentence_view(
            explicit_nodes=list(self._explicit_nodes_current_sentence),
            sentence_index=self._current_sentence_index,
        )

        if not rollback_needed:
            rollback_needed = self._connectivity_ops.repair_dangling_nodes(
                per_sentence_view=self._per_sentence_view,
                prev_sentences=prev_sentences,
                normalize_edge_label_fn=self._normalize_edge_label,
                persona=self.persona,
            )

        if not rollback_needed:
            rollback_needed = not self._validate_sentence_state()

        return rollback_needed

    def _build_projection(self, sentence_id: int):
        return self._projection_bookkeeping_ops.build_projection(
            sentence_id=sentence_id,
            per_sentence_view=self._per_sentence_view,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            previous_active_triplets=getattr(self, "_previous_active_triplets", []),
        )

    def _post_projection_bookkeeping(
        self, sentence_id: int, i: int, newly_inferred_nodes: set, per_sentence_view
    ):
        result = self._projection_bookkeeping_ops.post_projection_bookkeeping(
            sentence_id=sentence_id,
            sentence_index=i,
            newly_inferred_nodes=newly_inferred_nodes,
            per_sentence_view=per_sentence_view,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            persona=self.persona,
        )
        self._recently_deactivated_nodes_for_inference = (
            self._projection_bookkeeping_ops.get_recently_deactivated_nodes()
        )
        return result

    def _handle_sentence_rollback(
        self,
        i: int,
        original_text: str,
        plot_after_each_sentence: bool,
        graphs_output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        largest_component_only: bool,
        previous_graph_state,
        previous_anchor_nodes,
        previous_triplet_intro,
        previous_per_sentence_view,
        previous_recently_deactivated,
        previous_prev_active_nodes,
    ) -> None:
        logging.error(
            "[ROLLBACK] Sentence invalid — reverting to previous state."
        )

        if plot_after_each_sentence and hasattr(
            self, "_previous_active_triplets"
        ):
            try:
                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=original_text,
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    inactive_nodes=[],
                    explicit_nodes=[
                        n.get_text_representer()
                        for n in self._explicit_nodes_current_sentence
                    ],
                    salient_nodes=[],
                    only_active=True,
                    largest_component_only=largest_component_only,
                    mode="sentence_active",
                    triplets_override=self._previous_active_triplets,
                    active_edges=set(),
                    active_triplets_for_overlay=self._previous_active_triplets,
                    property_nodes=[],
                )
            except Exception as e:
                logging.warning(
                    f"[RollbackPlot] Failed to save previous plot: {e}"
                )

        self.graph = previous_graph_state
        self._rebind_ops_graph_refs()
        self._anchor_nodes.clear()
        self._anchor_nodes.update(previous_anchor_nodes)
        self._triplet_intro.clear()
        self._triplet_intro.update(previous_triplet_intro)
        self._per_sentence_view = previous_per_sentence_view
        self._recently_deactivated_nodes_for_inference = (
            previous_recently_deactivated
        )
        self._prev_active_nodes_for_plot = previous_prev_active_nodes
        if plot_after_each_sentence and previous_per_sentence_view is not None:
            self._per_sentence_view = previous_per_sentence_view

    def _capture_sentence_triplets(
        self,
        original_text: str,
    ) -> None:
        current_nodes = (
            self._explicit_nodes_current_sentence
            | self._get_nodes_with_active_edges()
        )
        for subj, rel, obj in self._reconstruct_semantic_triplets(
            only_active=False, restrict_nodes=current_nodes
        ):
            self._sentence_triplets.append(
                (
                    self._current_sentence_index,
                    original_text,
                    subj,
                    rel,
                    obj,
                    True,
                    True,
                    self._triplet_intro.get((subj, rel, obj), -1),
                )
            )
        for sent_idx, sent_text, subj, rel, obj in getattr(
            self, "_anchor_drop_log", []
        ):
            self._sentence_triplets.append(
                (
                    sent_idx,
                    sent_text,
                    subj,
                    rel,
                    obj,
                    False,
                    False,
                    -1,
                )
            )

    def _plot_sentence(
        self,
        i: int,
        original_text: str,
        graphs_output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        inactive_nodes_for_plot: list,
        salient_nodes_for_plot: list,
        largest_component_only: bool,
    ):
        self._plot_ops.plot_sentence_views(
            sentence_idx=i,
            original_text=original_text,
            graphs_output_dir=graphs_output_dir,
            highlight_nodes=highlight_nodes,
            inactive_nodes_for_plot=inactive_nodes_for_plot,
            salient_nodes_for_plot=salient_nodes_for_plot,
            largest_component_only=largest_component_only,
            per_sentence_view=self._per_sentence_view,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            reconstruct_semantic_triplets_fn=self._reconstruct_semantic_triplets,
        )
        self._viz_positions = self._plot_ops.viz_positions
        self._previous_active_triplets = self._plot_ops._previous_active_triplets

    def analyze(
        self,
        replace_pronouns: bool = True,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: Optional[str] = None,
        highlight_nodes: Optional[Iterable[str]] = None,
        matrix_suffix: Optional[str] = None,
        largest_component_only: bool = False,
        force_node: bool = False,
    ) -> List[Tuple[str, str, str]]:
        logging.info(
            "[AMoC] Story text (first 200 chars): %s",
            self.story_text[:200] if self.story_text else "NONE",
        )
        doc = self.spacy_nlp(self.story_text)
        sentences = list(doc.sents)
        logging.info("[AMoC] Number of sentences detected by spaCy: %d", len(sentences))
        for i, sent in enumerate(sentences):
            logging.info("[AMoC] Sentence %d: %s", i + 1, sent.text.strip()[:100])

        self._initialize_run_state()

        resolved_sentences = self._resolve_sentences(replace_pronouns)

        prev_sentences: list[str] = []
        current_sentence = ""
        self._sentence_triplets: list[
            tuple[int, str, str, str, str, bool, bool, int]
        ] = []
        sentence_counter = 0

        for i, (sent, resolved_text, original_text) in enumerate(resolved_sentences):
            if re.match(r"^\s*(user|system|assistant)\b", original_text.lower()):
                continue

            sentence_counter += 1
            original_text = re.sub(
                r"^The text is:\s*",
                "",
                original_text,
                flags=re.IGNORECASE,
            )

            resolved_text = re.sub(
                r"^The text is:\s*",
                "",
                resolved_text,
                flags=re.IGNORECASE,
            )
            self.active_graph = nx.MultiDiGraph()
            self._current_sentence_index = sentence_counter
            self.graph.set_current_sentence(self._current_sentence_index)

            current_sentence_lemmas = {
                w.lower() for w in re.findall(r"[a-zA-Z]+", original_text)
            }

            self.graph.set_current_sentence_lemmas(current_sentence_lemmas)

            (
                _previous_graph_state,
                _previous_anchor_nodes,
                _previous_triplet_intro,
                _previous_per_sentence_view,
                _previous_recently_deactivated,
                _previous_prev_active_nodes,
            ) = self._snapshot_sentence_state()

            nodes_before_sentence, should_skip_sentence = self._process_sentence_core(
                i, sent, resolved_text, original_text, prev_sentences
            )
            if should_skip_sentence:
                continue
            if self.debug:
                logging.info(
                    "Active graph after sentence %d:\n%s",
                    i,
                    self.graph.get_active_graph_repr(),
                )

            if self._anchor_nodes:
                self._anchor_nodes = {
                    n for n in self._anchor_nodes if n in self.graph.nodes
                }

            sentence_id = i + 1
            newly_inferred_nodes = {
                n
                for n in (set(self.graph.nodes) - nodes_before_sentence)
                if n.node_source == NodeSource.INFERENCE_BASED
            }

            rollback_needed = self._enforce_connectivity(prev_sentences)

            if rollback_needed:
                self._handle_sentence_rollback(
                    i=i,
                    original_text=original_text,
                    plot_after_each_sentence=plot_after_each_sentence,
                    graphs_output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    largest_component_only=largest_component_only,
                    previous_graph_state=_previous_graph_state,
                    previous_anchor_nodes=_previous_anchor_nodes,
                    previous_triplet_intro=_previous_triplet_intro,
                    previous_per_sentence_view=_previous_per_sentence_view,
                    previous_recently_deactivated=_previous_recently_deactivated,
                    previous_prev_active_nodes=_previous_prev_active_nodes,
                )
                continue

            temp_projection = self._build_projection(sentence_id)

            self.graph.enforce_carryover_connectivity(
                set(temp_projection.carryover_nodes)
            )

            per_sentence_view = self._build_projection(sentence_id)

            self._carryover_nodes_current_sentence.clear()
            self._carryover_nodes_current_sentence.update(per_sentence_view.carryover_nodes)

            (
                recently_deactivated_nodes,
                explicit_nodes_for_plot,
                salient_nodes_for_plot,
                inactive_nodes_for_plot,
            ) = self._post_projection_bookkeeping(
                sentence_id, i, newly_inferred_nodes, per_sentence_view
            )

            if plot_after_each_sentence:
                self._plot_sentence(
                    i,
                    original_text,
                    graphs_output_dir,
                    highlight_nodes,
                    inactive_nodes_for_plot,
                    salient_nodes_for_plot,
                    largest_component_only,
                )

            self._capture_sentence_triplets(original_text)

        return self._finalize_outputs(matrix_suffix)

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        return self._inference_ops.infer_new_relationships_step_0(sent)

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        return self._inference_ops.infer_new_relationships(
            text=text,
            current_sentence_text_based_nodes=current_sentence_text_based_nodes,
            current_sentence_text_based_words=current_sentence_text_based_words,
            graph_nodes_representation=graph_nodes_representation,
            graph_edges_representation=graph_edges_representation,
        )

    def reactivate_relevant_edges(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        self._activation_ops.set_current_sentence(self._current_sentence_index)
        self._activation_ops.reactivate_relevant_edges(
            active_nodes=active_nodes,
            prev_sentences_text=prev_sentences_text,
            newly_added_edges=newly_added_edges,
        )

    def _propagate_activation_from_edges(self):
        self._activation_ops.propagate_activation_from_edges()

    def _extract_deterministic_structure(
        self,
        sent: Span,
        sentence_nodes: List[Node],
        sentence_words: List[str],
    ) -> None:
        self._linguistic_ops.set_sentence_context(
            sentence_index=self._current_sentence_index,
            edge_visibility=self.edge_visibility,
        )
        self._linguistic_ops.extract_deterministic_structure(
            sent, sentence_nodes, sentence_words
        )

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )
        self._relationship_graph_ops.set_current_sentence(self._current_sentence_index)
        self._relationship_graph_ops.add_inferred_relationships_to_graph_step_0(
            inferred_relationships=inferred_relationships,
            node_type=node_type,
            sent=sent,
            current_sentence_text_based_nodes=current_sentence_text_based_nodes,
            current_sentence_text_based_words=current_sentence_text_based_words,
        )

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        active_graph_nodes: List[Node],
        added_edges: List[Edge],
    ) -> None:
        self._relationship_graph_ops.set_current_sentence(self._current_sentence_index)
        self._relationship_graph_ops.add_inferred_relationships_to_graph(
            inferred_relationships=inferred_relationships,
            node_type=node_type,
            curr_sentences_nodes=curr_sentences_nodes,
            curr_sentences_words=curr_sentences_words,
            active_graph_nodes=active_graph_nodes,
            added_edges=added_edges,
        )

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        return self._node_ops.get_node_from_text(
            text=text,
            curr_sentences_nodes=curr_sentences_nodes,
            curr_sentences_words=curr_sentences_words,
            node_source=node_source,
            create_node=create_node,
        )

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        return self._node_ops.get_node_from_new_relationship(
            text=text,
            graph_active_nodes=graph_active_nodes,
            curr_sentences_nodes=curr_sentences_nodes,
            curr_sentences_words=curr_sentences_words,
            node_source=node_source,
            create_node=create_node,
        )

    def is_content_word_and_non_stopword(
        self,
        token: Token,
        pos_list: List[str] = [
            "NOUN",
            "PROPN",
            "ADJ",
        ],
    ) -> bool:
        return (token.pos_ in pos_list) and (
            token.lemma_ not in self.spacy_nlp.Defaults.stop_words
        )

    def get_phrase_level_concepts(self, sent):
        return self._node_ops.get_phrase_level_concepts(
            sent=sent,
            admit_node_fn=self._admit_node,
        )

    def get_senteces_text_based_nodes(
        self,
        previous_sentences: List[Span],
        create_unexistent_nodes: bool = True,
    ) -> Tuple[List[Node], List[str]]:
        return self._node_ops.get_sentences_text_based_nodes(
            previous_sentences=previous_sentences,
            current_sentence_index=self._current_sentence_index,
            create_unexistent_nodes=create_unexistent_nodes,
        )
