import copy
import logging
import re
from typing import List, Tuple, Optional, Iterable

import networkx as nx
from spacy.tokens import Span

from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph.per_sentence_graph import PerSentenceGraph, build_per_sentence_graph
from amoc.llm.vllm_client import VLLMClient

from amoc.pipeline.connectivity_ops import ConnectivityOps
from amoc.pipeline.text_filter_ops import TextFilterOps
from amoc.pipeline.triplet_ops import TripletOps
from amoc.pipeline.edge_ops import EdgeOps
from amoc.pipeline.node_ops import NodeOps
from amoc.pipeline.sentence_ops import SentenceOps
from amoc.pipeline.inference_ops import InferenceOps
from amoc.pipeline.linguistic_ops import LinguisticOps
from amoc.pipeline.plot_ops import PlotOps
from amoc.pipeline.activation_scheduler import ActivationScheduler
from amoc.pipeline.output_ops import OutputOps
from amoc.pipeline.sentence_processing_ops import SentenceProcessingOps
from amoc.pipeline.relationship_graph_ops import RelationshipGraphOps
from amoc.pipeline.init_ops import InitOps
from amoc.pipeline.projection_bookkeeping_ops import ProjectionBookkeepingOps


class AMoCv4:
    ENFORCE_ATTACHMENT_CONSTRAINT = True
    ACTIVATION_MAX_DISTANCE = 2

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
        self._current_sentence_index: Optional[int] = None
        self.cumulative_graph = nx.MultiDiGraph()
        self.active_graph = nx.MultiDiGraph()
        self._triplet_intro: dict[tuple[str, str, str], int] = {}
        self._cumulative_triplet_records: list[dict] = []
        self._fixed_hub = None
        self._per_sentence_view: Optional[PerSentenceGraph] = None
        self._ever_admitted_nodes: set[str] = set()
        self._layout_depth = 3
        self._persistent_is_edges: set[tuple[str, str, str]] = set()
        self._amoc_matrix_records: list[dict] = []
        self._previous_active_triplets: list[tuple[str, str, str]] = []
        self._anchor_drop_log: list[tuple[int, str, str, str, str]] = []
        self._story_lemma_set: set[str] = set()
        self._sentence_triplets: list[
            tuple[int, str, str, str, str, bool, bool, int]
        ] = []
        self._new_inferred_nodes_count: int = 0

        self._get_explicit_nodes = lambda: self._explicit_nodes_current_sentence
        self._get_carryover_nodes = lambda: self._carryover_nodes_current_sentence
        self._get_active_edge_nodes = (
            lambda: self._connectivity_ops._get_nodes_with_active_edges()
        )

        self._setup_ops_classes()

    def _setup_ops_classes(self):
        self._connectivity_ops = ConnectivityOps(
            graph_ref=self.graph,
            get_explicit_nodes=self._get_explicit_nodes,
            get_carryover_nodes=self._get_carryover_nodes,
            edge_visibility=self.edge_visibility,
            client_ref=self.client,
        )
        self._text_filter_ops = TextFilterOps(
            spacy_nlp=self.spacy_nlp,
            graph_ref=self.graph,
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
        )

        self._normalize_edge_label = (
            lambda l: self._text_filter_ops.normalize_edge_label(l)
        )
        self._is_valid_relation_label = (
            lambda l: self._text_filter_ops.is_valid_relation_label(l)
        )
        self._classify_relation = lambda l: self._text_filter_ops.classify_relation(l)
        self._normalize_endpoint_text = (
            lambda text, is_subject: self._text_filter_ops.normalize_endpoint_text(
                text, is_subject
            )
        )
        self._canonicalize_edge_direction = (
            lambda l, s, d: self._text_filter_ops.canonicalize_edge_direction(l, s, d)
        )
        self._get_sentences_nodes = lambda sents, create_unexistent_nodes=True: self._collect_sentence_text_based_nodes(
            sents, create_unexistent_nodes=create_unexistent_nodes
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
            get_explicit_nodes=self._get_explicit_nodes,
            get_carryover_nodes=self._get_carryover_nodes,
            get_attachable_nodes=lambda: self._sentence_ops.get_attachable_nodes_for_sentence(
                self._get_active_edge_nodes
            ),
            edge_visibility=self.edge_visibility,
            allow_multi_edges=self.allow_multi_edges,
            debug=self.debug,
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
            has_active_attachment_fn=lambda l: self._activation_ops.has_active_attachment(
                l
            ),
            canonicalize_and_classify_fn=lambda t: self._text_filter_ops.canonicalize_and_classify_node_text(
                t
            ),
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
            append_adjectival_hints_fn=lambda n, s: self._linguistic_ops.append_adjectival_hints(
                n, s
            ),
            get_sentences_text_based_nodes_fn=self._get_sentences_nodes,
        )
        self._linguistic_ops = LinguisticOps(
            graph_ref=self.graph,
            spacy_nlp=self.spacy_nlp,
            client_ref=self.client,
            story_lemmas=self.story_lemmas,
            persona=self.persona,
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
            get_explicit_nodes_fn=self._get_explicit_nodes,
            get_edge_activation_scores_fn=lambda: self._triplet_ops.get_edge_activation_scores(),
            graph_edges_to_triplets_fn=lambda only_active=False: self._triplet_ops.graph_edges_to_triplets(
                only_active
            ),
            enforce_cumulative_connectivity_fn=lambda: self._connectivity_ops.warn_if_cumulative_disconnected(),
        )
        self._plot_ops.set_lemmas(
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
        )
        self._activation_ops = ActivationScheduler(
            graph_ref=self.graph,
            client_ref=self.client,
            get_explicit_nodes=self._get_explicit_nodes,
            max_distance=self.max_distance_from_active_nodes,
            edge_visibility=self.edge_visibility,
            nr_relevant_edges=self.nr_relevant_edges,
            strict_reactivate=self.strict_reactivate_function,
        )
        self._activation_ops.set_state_refs(
            anchor_nodes=self._anchor_nodes,
            record_edge_fn=lambda e, i: self._edge_ops.record_edge_in_graphs(
                edge=e,
                sentence_idx=i,
                cumulative_graph=self.cumulative_graph,
                active_graph=self.active_graph,
                cumulative_triplet_records=self._cumulative_triplet_records,
            ),
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
        # Configure ops classes
        self._edge_ops.configure_with_core(self)
        self._relationship_graph_ops.configure_with_core(self)
        self._init_ops.configure_with_core(self)
        self._sentence_processing_ops.configure_with_core(self)
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
        ]:
            ops._graph = self.graph

        for ops in [
            self._relationship_graph_ops,
            self._init_ops,
            self._sentence_processing_ops,
            self._projection_bookkeeping_ops,
        ]:
            ops.graph = self.graph

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

    def _construct_per_sentence_view(
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
            node_token_fn=lambda n: self._node_ops.node_token_for_matrix(n),
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
        allow_inference_bridge: bool = False,  # inference is low in early sentences, want to boost it
    ) -> bool:
        return self._node_ops.passes_attachment_constraint(
            subject=subject,
            obj=obj,
            current_sentence_words=current_sentence_words,
            current_sentence_nodes=current_sentence_nodes,
            graph_active_nodes=graph_active_nodes,
            explicit_nodes=set(self._explicit_nodes_current_sentence),
            carryover_nodes=self._carryover_nodes_current_sentence,
            get_nodes_with_active_edges_fn=self._get_active_edge_nodes,
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
            relation_class=relation_class,
            justification=justification,
            persona_influenced=persona_influenced,
        )
        if edge:
            self._record_edge_in_graphs(edge, self._current_sentence_index)
        return edge

    def _create_forced_connectivity_edges(
        self, story_context=None, current_sentence=None, mode="active"
    ):
        return self._edge_ops.create_forced_connectivity_edges(
            story_context=story_context,
            current_sentence=current_sentence,
            mode=mode,
            persona=self.persona,
            normalize_edge_label_fn=self._normalize_edge_label,
        )

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        self._edge_ops.record_edge_in_graphs(
            edge=edge,
            sentence_idx=sentence_idx,
            cumulative_graph=self.cumulative_graph,
            active_graph=self.active_graph,
            cumulative_triplet_records=self._cumulative_triplet_records,
        )

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

    def _initialize_run_state(self) -> None:
        self._previous_active_triplets = []
        self._viz_positions = {}
        self._projection_bookkeeping_ops.reset_state()

    def _resolve_story_sentences(self, replace_pronouns: bool) -> list:
        resolved_sentences, story_lemma_set = self._sentence_ops.resolve_sentences(
            story_text=self.story_text,
            replace_pronouns=replace_pronouns,
            resolve_pronouns_fn=self._linguistic_ops.resolve_pronouns,
        )
        self._story_lemma_set = story_lemma_set
        return resolved_sentences

    def _snapshot_sentence_runtime_state(self) -> tuple:
        return self._sentence_ops.snapshot_sentence_state(
            anchor_nodes=self._anchor_nodes,
            triplet_intro=self._triplet_intro,
            per_sentence_view=self._per_sentence_view,
            recently_deactivated=self._recently_deactivated_nodes_for_inference,
            prev_active_nodes=self._prev_active_nodes_for_plot,
        )

    def _prepare_sentence_runtime_state(self, original_text: str) -> set:
        nodes_before_sentence = self._sentence_ops.reset_sentence_state(original_text)
        self._current_sentence_text = original_text
        self._anchor_drop_log = []
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

    def _process_sentence_core(
        self,
        i: int,
        sent,
        resolved_text: str,
        original_text: str,
        prev_sentences: list,
    ) -> tuple:
        self._new_inferred_nodes_count = 0
        nodes_before_sentence = self._prepare_sentence_runtime_state(original_text)
        logging.info("Processing sentence %d: %s", i, resolved_text)

        resolved_text, sent = self._sentence_processing_ops.sanitize_json_contamination(
            resolved_text, original_text, sent
        )

        if i == 0:
            result = self._handle_first_sentence(
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
                # Connectivity enforcement is handled by _stabilize_sentence_connectivity() in analyze()
                self._sentence_processing_ops.apply_post_sentence_processing(
                    explicit_nodes=self._explicit_nodes_current_sentence,
                    carryover_nodes=self._carryover_nodes_current_sentence,
                    apply_global_edge_decay_fn=lambda: self._activation_ops.apply_global_edge_decay(),
                    decay_node_activation_fn=lambda: self._activation_ops.decay_node_activation(),
                )

        return result

    def _finalize_run_outputs(self, matrix_suffix: Optional[str]) -> tuple:
        return self._output_ops.finalize_outputs(
            amoc_matrix_records=self._amoc_matrix_records,
            triplet_intro=self._triplet_intro,
            explicit_nodes_current_sentence=self._explicit_nodes_current_sentence,
            get_nodes_with_active_edges_fn=self._get_active_edge_nodes,
            reconstruct_semantic_triplets_fn=lambda only_active=False, restrict_nodes=None: self._triplet_ops.reconstruct_semantic_triplets(
                only_active=only_active, restrict_nodes=restrict_nodes
            ),
            current_sentence_index=self._current_sentence_index,
            sentence_triplets=self._sentence_triplets,
            matrix_suffix=matrix_suffix,
        )

    def _stabilize_sentence_connectivity(self, prev_sentences: list) -> bool:
        # Step 1 — deterministic connectivity repair
        rollback_needed = self._connectivity_ops.run_connectivity_pipeline(
            prev_sentences=prev_sentences,
            current_sentence_text=self._current_sentence_text,
            create_forced_edges_fn=self._create_forced_connectivity_edges,
        )

        if rollback_needed:
            return True

        # introducing anchors = all concepts that were explicit
        explicit_concepts = {
            n
            for n in self._explicit_nodes_current_sentence
            if n.node_type == NodeType.CONCEPT
        }

        active_concepts = {
            n for n in self._get_active_edge_nodes() if n.node_type == NodeType.CONCEPT
        }

        self._anchor_nodes |= explicit_concepts
        self._anchor_nodes |= active_concepts

        # keep only the active nodes
        self._anchor_nodes = {n for n in self._anchor_nodes if n in self.graph.nodes}

        # Step 2 — rebuild per-sentence view
        self._per_sentence_view = self._construct_per_sentence_view(
            explicit_nodes=list(self._explicit_nodes_current_sentence),
            sentence_index=self._current_sentence_index,
        )

        # Step 3 — repair dangling nodes - they appear sometimes
        repair_needed = self._connectivity_ops.repair_dangling_nodes(
            per_sentence_view=self._per_sentence_view,
            prev_sentences=prev_sentences,
            normalize_edge_label_fn=self._normalize_edge_label,
            persona=self.persona,
        )

        if repair_needed:
            return True

        # Step 4 — validation
        if not self._connectivity_ops.validate_sentence_state():
            return True

        return False

    def _construct_sentence_projection(self, sentence_id: int):
        return self._projection_bookkeeping_ops.build_projection(
            sentence_id,
            self._per_sentence_view,
            self._explicit_nodes_current_sentence,
            self._previous_active_triplets,
        )

    def _update_post_projection_state(
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
        logging.error("Sentence invalid — reverting to previous state.")
        if plot_after_each_sentence:
            try:
                explicit = [
                    n.get_text_representer()
                    for n in self._explicit_nodes_current_sentence
                ]
                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=original_text,
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    inactive_nodes=[],
                    explicit_nodes=explicit,
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
                logging.warning(f"Rollback plot failed: {e}")

        self.graph = previous_graph_state
        self._rebind_ops_graph_refs()
        self._anchor_nodes.clear()
        self._anchor_nodes.update(previous_anchor_nodes)
        self._triplet_intro.clear()
        self._triplet_intro.update(previous_triplet_intro)
        self._recently_deactivated_nodes_for_inference = previous_recently_deactivated
        self._prev_active_nodes_for_plot = previous_prev_active_nodes
        self._per_sentence_view = previous_per_sentence_view

    def _record_sentence_triplets(self, original_text: str) -> None:
        self._triplet_ops.capture_sentence_triplets(
            original_text=original_text,
            current_sentence_index=self._current_sentence_index,
            explicit_nodes=self._explicit_nodes_current_sentence,
            nodes_with_active_edges=self._get_active_edge_nodes(),
            sentence_triplets=self._sentence_triplets,
            anchor_drop_log=self._anchor_drop_log,
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
            reconstruct_semantic_triplets_fn=lambda only_active=False, restrict_nodes=None: self._triplet_ops.reconstruct_semantic_triplets(
                only_active=only_active, restrict_nodes=restrict_nodes
            ),
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
            "Story text: %s", self.story_text[:200] if self.story_text else "NONE"
        )
        doc = self.spacy_nlp(self.story_text)
        sentences = list(doc.sents)
        logging.info("Number of sentences detected by spaCy: %d", len(sentences))
        for i, sent in enumerate(sentences):
            logging.info("[Sentence %d: %s", i + 1, sent.text.strip()[:100])

        self._initialize_run_state()

        resolved_sentences = self._resolve_story_sentences(replace_pronouns)

        prev_sentences: list[str] = []
        self._sentence_triplets = []
        sentence_counter = 0
        text_prefix_pattern = re.compile(r"^The text is:\s*", re.IGNORECASE)

        for i, (sent, resolved_text, original_text) in enumerate(resolved_sentences):
            if re.match(r"^\s*(user|system|assistant)\b", original_text.lower()):
                continue
            sentence_counter += 1
            original_text = text_prefix_pattern.sub("", original_text)
            resolved_text = text_prefix_pattern.sub("", resolved_text)
            self.active_graph = nx.MultiDiGraph()
            self._current_sentence_index = sentence_counter

            self._sentence_ops.configure_graph_for_sentence(
                self._current_sentence_index,
                self._sentence_ops.extract_sentence_lemmas(original_text),
            )

            (
                _previous_graph_state,
                _previous_anchor_nodes,
                _previous_triplet_intro,
                _previous_per_sentence_view,
                _previous_recently_deactivated,
                _previous_prev_active_nodes,
            ) = self._snapshot_sentence_runtime_state()

            nodes_before_sentence, should_skip_sentence = self._process_sentence_core(
                i, sent, resolved_text, original_text, prev_sentences
            )
            if should_skip_sentence:
                continue

            sentence_id = i + 1
            newly_inferred_nodes = (
                self._projection_bookkeeping_ops.compute_newly_inferred_nodes(
                    nodes_before_sentence
                )
            )

            # CONNECTIVITY
            self._connectivity_ops.set_anchor_nodes(self._anchor_nodes)
            rollback_needed = self._stabilize_sentence_connectivity(prev_sentences)

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

            # Build projection AFTER connectivity is guaranteed
            self._per_sentence_view = self._construct_sentence_projection(sentence_id)

            # Update carryover nodes from projection
            if self._per_sentence_view is not None:
                self._carryover_nodes_current_sentence.clear()
                self._carryover_nodes_current_sentence.update(
                    self._per_sentence_view.carryover_nodes
                )
            else:
                self._carryover_nodes_current_sentence.clear()

            (
                recently_deactivated_nodes,
                explicit_nodes_for_plot,
                salient_nodes_for_plot,
                inactive_nodes_for_plot,
            ) = self._update_post_projection_state(
                sentence_id, i, newly_inferred_nodes, self._per_sentence_view
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
            self._record_sentence_triplets(original_text)

        # Ensure cumulative graph is connected
        if not self._connectivity_ops.is_cumulative_connected():
            logging.warning(
                "Cumulative graph disconnected after sentence loop - repairing"
            )
            self._stabilize_sentence_connectivity(prev_sentences)

        return self._finalize_run_outputs(matrix_suffix)

    def _infer_new_relationships_bootstrap(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        return self._inference_ops.infer_new_relationships_step_0(sent)

    def _infer_new_relationships_for_sentence(
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

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._collect_sentence_text_based_nodes(
                [sent], create_unexistent_nodes=False
            )
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

    def _resolve_node_from_text(
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

    def _resolve_node_from_new_relationship(
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

    def _extract_phrase_level_concepts(self, sent):
        return self._node_ops.get_phrase_level_concepts(
            sent=sent,
            admit_node_fn=lambda lemma, node_type, provenance, sent=None: self._node_ops.admit_node(
                lemma, node_type, provenance, sent
            ),
        )

    def _collect_sentence_text_based_nodes(
        self,
        previous_sentences: List[Span],
        create_unexistent_nodes: bool = True,
    ) -> Tuple[List[Node], List[str]]:
        return self._node_ops.get_sentences_text_based_nodes(
            previous_sentences=previous_sentences,
            current_sentence_index=self._current_sentence_index,
            create_unexistent_nodes=create_unexistent_nodes,
        )
