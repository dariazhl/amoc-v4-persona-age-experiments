import copy
import logging
import re
from typing import List, Tuple, Optional, Iterable

from spacy.tokens import Span

from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.core import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph_views.per_sentence import (
    PerSentenceGraph,
    build_per_sentence_graph,
)
from amoc.graph_views.active import ActiveGraph
from amoc.graph_views.cumulative import CumulativeGraph
from amoc.graph_views.active import ActiveGraphBuilder
from amoc.graph_views.cumulative import CumulativeGraphBuilder
from amoc.llm.vllm_client import VLLMClient
from amoc.pipeline.wiring import wire_core_dependencies
from amoc.config.constants import MAX_DISTANCE_FROM_ACTIVE_NODES


class AMoCv4:
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
        single_anchor_hub: bool = True,
        matrix_dir_base: Optional[str] = None,
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

        self._prev_active_nodes_for_plot = set()
        self._cumulative_deactivated_nodes_for_plot = set()
        self._viz_positions = {}
        self._recently_deactivated_nodes_for_inference = set()
        self._explicit_nodes_current_sentence = set()
        self._carryover_nodes_current_sentence = set()
        self.strict_reactivate_function = strict_reactivate_function
        self.single_anchor_hub = single_anchor_hub
        self.checkpoint = checkpoint
        self._current_sentence_text = ""
        self._current_sentence_index = None
        self.cumulative_graph = CumulativeGraph()
        self.active_graph = ActiveGraph()
        self.cumulative_graph_builder = CumulativeGraphBuilder(self.cumulative_graph)
        self.active_graph_builder = ActiveGraphBuilder(self.active_graph)
        self._triplet_intro = {}
        self._cumulative_triplet_records = []
        self._per_sentence_view = None
        self._ever_admitted_nodes = set()
        self._layout_depth = 3
        self._amoc_matrix_records = []
        self._previous_active_triplets = []
        self._story_lemma_set = set()
        self._sentence_triplets = []
        self._new_inferred_nodes_count = 0

        self._get_explicit_nodes = lambda: self._explicit_nodes_current_sentence
        self._get_carryover_nodes = lambda: self._carryover_nodes_current_sentence
        self._get_active_edge_nodes = (
            lambda: self._connectivity_ops.get_nodes_with_active_edges()
        )

        self.setup_ops_classes()

    def setup_ops_classes(self):
        wire_core_dependencies(self)

    def rebind_ops_graph_refs(self) -> None:
        for ops in [
            self._connectivity_ops,
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

    def build_per_sentence_view_wrapper(
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

    def record_activation_matrix_wrapper(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        view = self._activation_ops.record_sentence_activation_matrix(
            sentence_id=sentence_id,
            explicit_nodes=explicit_nodes,
            newly_inferred_nodes=newly_inferred_nodes,
            max_distance=MAX_DISTANCE_FROM_ACTIVE_NODES,
            node_token_fn=lambda n: self._node_ops.node_token_for_matrix(n),
            append_record_fn=self._amoc_matrix_records.append,
        )

        # If view is None or has no active nodes, create a minimal view with explicit nodes only
        if view is None or (
            len(view.explicit_nodes) == 0 and len(view.carryover_nodes) == 0
        ):
            # Create a minimal view with just the explicit nodes (no edges)
            view = PerSentenceGraph(
                sentence_index=sentence_id,
                explicit_nodes=frozenset(explicit_nodes),
                carryover_nodes=frozenset(),
                active_nodes=frozenset(explicit_nodes),
                active_edges=frozenset(),
                anchor_nodes=frozenset(),
            )

        self._per_sentence_view = view
        return view

    def llm_attach_explicit_to_carryover_wrapper(
        self,
        current_sentence_nodes: List[Node],
        current_sentence_words: List[str],
        current_text: str,
    ) -> List[Edge]:
        return self._edge_ops.llm_attach_explicit_to_carryover(
            current_sentence_nodes=current_sentence_nodes,
            current_sentence_words=current_sentence_words,
            current_text=current_text,
            recently_deactivated_nodes=self._recently_deactivated_nodes_for_inference,
            enforce_attachment=True,
        )

    def is_attachable_wrapper(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
        allow_inference_bridge: bool = False,
    ) -> bool:
        return self._node_ops.is_attachable(
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

    def add_edge_wrapper(
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
        self._edge_ops.set_edge_sentence_context(self._current_sentence_index)
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
            self.record_edge_in_graphs_wrapper(edge, self._current_sentence_index)
        return edge

    def create_forced_connectivity_edges_wrapper(
        self, story_context=None, current_sentence=None, mode="active"
    ):
        return self._edge_ops.create_forced_connectivity_edges(
            story_context=story_context,
            current_sentence=current_sentence,
            mode=mode,
            persona=self.persona,
            normalize_edge_label_fn=self._normalize_edge_label,
        )

    def record_edge_in_graphs_wrapper(
        self, edge: Edge, sentence_idx: Optional[int]
    ) -> None:
        self._edge_ops.record_edge_in_graphs(
            edge=edge,
            sentence_idx=sentence_idx,
            cumulative_graph=self.cumulative_graph,
            active_graph=self.active_graph,
            cumulative_triplet_records=self._cumulative_triplet_records,
            cumulative_graph_builder=self.cumulative_graph_builder,
            active_graph_builder=self.active_graph_builder,
        )

    def plot_graph_snapshot_wrapper(
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
        self._viz_positions = self._plot_ops.get_viz_positions()

    def initialize_run_state(self) -> None:
        self._previous_active_triplets = []
        self._viz_positions = {}
        self._projection_bookkeeping_ops.reset_state()

    def resolve_story_sentences(self, replace_pronouns: bool) -> list:
        resolved_sentences, story_lemma_set = self._sentence_ops.resolve_sentences(
            story_text=self.story_text,
            replace_pronouns=replace_pronouns,
            resolve_pronouns_wrapper_fn=self._linguistic_ops.resolve_pronouns_wrapper,
        )
        self._story_lemma_set = story_lemma_set
        return resolved_sentences

    def snapshot_graph_state(self):
        return copy.deepcopy(self.graph)

    def reset_sentence_state_wrapper(self, original_text: str) -> set:
        nodes_before_sentence = self._sentence_ops.reset_sentence_state(original_text)
        self._current_sentence_text = original_text
        self._explicit_nodes_current_sentence.clear()
        return nodes_before_sentence

    def handle_first_sentence_wrapper(
        self,
        sent,
        resolved_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
    ) -> tuple:
        result = self._sentence_processing_ops.handle_first_sentence(
            sent=sent,
            resolved_text=resolved_text,
            prev_sentences=prev_sentences,
            nodes_before_sentence=nodes_before_sentence,
        )
        nodes_before, should_skip, explicit_nodes, _unused = result
        self._explicit_nodes_current_sentence.clear()
        self._explicit_nodes_current_sentence.update(explicit_nodes)
        return (nodes_before, should_skip)

    def handle_nonfirst_sentence_wrapper(
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
            anchor_nodes=set(),
            triplet_intro=self._triplet_intro,
        )

    def process_sentence_core_wrapper(
        self,
        i: int,
        sent,
        resolved_text: str,
        original_text: str,
        prev_sentences: list,
    ) -> tuple:
        self._new_inferred_nodes_count = 0
        nodes_before_sentence = self.reset_sentence_state_wrapper(original_text)

        resolved_text, sent = self._sentence_processing_ops.clean_llm_output(
            resolved_text, original_text, sent
        )

        if i == 0:
            result = self.handle_first_sentence_wrapper(
                sent, resolved_text, prev_sentences, nodes_before_sentence
            )
            logging.info(
                f"after sentence 1, graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
            )
        else:
            result = self.handle_nonfirst_sentence_wrapper(
                i,
                sent,
                resolved_text,
                original_text,
                prev_sentences,
                nodes_before_sentence,
            )

            if not result[1]:
                # Connectivity enforcement is handled by stabilize_connectivity_wrapper() in analyze()
                self._sentence_processing_ops.run_post_processing(
                    explicit_nodes=self._explicit_nodes_current_sentence,
                    carryover_nodes=self._carryover_nodes_current_sentence,
                    apply_edge_decay_fn=lambda: self._activation_ops.apply_semantic_edge_decay(),
                    enforce_node_limit_fn=lambda: self._activation_ops.enforce_node_limit(),
                )

        return result

    def finalize_run_outputs_wrapper(self, matrix_suffix: Optional[str]) -> tuple:
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

    def stabilize_connectivity_wrapper(self, prev_sentences: list) -> bool:
        explicit_nodes = self._explicit_nodes_current_sentence

        # Condition 1: No explicit nodes at all
        if len(explicit_nodes) == 0:
            logging.warning(
                f"ROLLBACK TRIGGER: No explicit nodes in sentence {self._current_sentence_index}"
            )
            return True

        # Run repair pipeline before checking for dangling nodes
        if self._per_sentence_view is not None:
            self._connectivity_ops.run_repair_pipeline(
                per_sentence_view=self._per_sentence_view,
                prev_sentences=prev_sentences,
                current_sentence_text=self._current_sentence_text,
                normalize_edge_label_fn=self._normalize_edge_label,
                create_forced_edges_fn=self.create_forced_connectivity_edges_wrapper,
                persona=self.persona,
            )

            # Rebuild the view to reflect any edges added by repair
            self._per_sentence_view = self.build_per_sentence_view_wrapper(
                explicit_nodes=list(explicit_nodes),
                sentence_index=self._current_sentence_index,
            )

        # Condition 2: Single dangling explicit node (no active edges after repair)
        if len(explicit_nodes) == 1:
            node = next(iter(explicit_nodes))
            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self.graph.edges
            )
            if not has_active_edge:
                logging.warning(
                    f"ROLLBACK TRIGGER: Single dangling node "
                    f"'{node.get_text_representer()}' in sentence "
                    f"{self._current_sentence_index}"
                )
                return True

        return False

    def build_projection_wrapper(self, sentence_id: int):
        return self._projection_bookkeeping_ops.build_projection(
            sentence_id,
            self._per_sentence_view,
            self._explicit_nodes_current_sentence,
            self._previous_active_triplets,
        )

    def update_post_projection_state_wrapper(
        self, sentence_id: int, i: int, newly_inferred_nodes: set, per_sentence_view
    ):
        result = self._projection_bookkeeping_ops.update_projection_state(
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

    def handle_sentence_rollback_wrapper(
        self,
        i: int,
        original_text: str,
        prev_sentences: list,
        previous_graph_state,
        previous_carryover_nodes,
    ) -> None:
        logging.warning(
            f"ROLLBACK_OCCURRED: At sentence {i+1} | "
            f"Before rollback: {len(self.graph.nodes)} nodes | "
            f"Restoring to: {len(previous_graph_state.nodes)} nodes"
        )

        # 1. Restore graph to previous state
        self.graph = previous_graph_state
        self.rebind_ops_graph_refs()

        # 2. Keep current sentence in prev_sentences so LLM context is aware
        if original_text not in prev_sentences:
            prev_sentences.append(original_text)

        # 3. Restore carryover nodes to pre-sentence state
        self._carryover_nodes_current_sentence = previous_carryover_nodes

        # 4. Clear per-sentence view
        self._per_sentence_view = None

        logging.info(
            f"rollback done for sentence {i+1}, "
            f"context preserved ({len(prev_sentences)} sentences in history)"
        )

    def capture_sentence_triplets_wrapper(self, original_text: str) -> None:
        self._triplet_ops.capture_sentence_triplets(
            original_text=original_text,
            current_sentence_index=self._current_sentence_index,
            explicit_nodes=self._explicit_nodes_current_sentence,
            nodes_with_active_edges=self._get_active_edge_nodes(),
            sentence_triplets=self._sentence_triplets,
        )

    def capture_state_only_wrapper(
        self,
        i: int,
        original_text: str,
        inactive_nodes_for_plot: list,
        salient_nodes_for_plot: list,
    ):
        """Capture graph state for reverse plots without generating per-sentence PNGs."""
        if not self._plot_ops._collect_states:
            return

        explicit_nodes_for_plot = [
            node.get_text_representer()
            for node in self._explicit_nodes_current_sentence
            if node.get_text_representer()
        ]

        inferred_nodes = [
            n.get_text_representer()
            for n in self.graph.nodes
            if n.node_source == NodeSource.INFERENCE_BASED
        ]

        # Cumulative view state
        cumulative_active_pairs = {
            (
                edge.source_node.get_text_representer(),
                edge.dest_node.get_text_representer(),
            )
            for edge in self.graph.edges
            if edge.active
        }
        cumulative_triplets = self._triplet_ops.reconstruct_semantic_triplets(
            only_active=True
        )

        active_node_names = set(explicit_nodes_for_plot) | set(salient_nodes_for_plot)
        self._plot_ops._ever_in_working_memory.update(active_node_names)
        inactive_nodes_recalc = sorted(
            self._plot_ops._ever_in_working_memory - active_node_names
        )

        self._plot_ops._capture_state(
            sentence_idx=i,
            sentence_text=original_text,
            mode="cumulative",
            triplets=cumulative_triplets,
            explicit_nodes=explicit_nodes_for_plot,
            inactive_nodes=inactive_nodes_recalc,
            salient_nodes=salient_nodes_for_plot,
            inferred_nodes=inferred_nodes,
            active_edges=cumulative_active_pairs,
        )

        # Paper view state: all triplets (active + inactive), active edges highlighted
        all_triplets = self._triplet_ops.reconstruct_semantic_triplets(
            only_active=False
        )
        active_triplets = self._triplet_ops.reconstruct_semantic_triplets(
            only_active=True
        )

        nodes_with_active_edges = set()
        for s, r, o in active_triplets:
            if s:
                nodes_with_active_edges.add(s)
            if o:
                nodes_with_active_edges.add(o)

        active_edge_set = {(s, o) for s, r, o in active_triplets if s and o}

        all_nodes = set()
        for s, _, o in all_triplets:
            if s:
                all_nodes.add(s)
            if o:
                all_nodes.add(o)

        carryover_nodes = sorted(nodes_with_active_edges - set(explicit_nodes_for_plot))

        inferred_node_names = {
            n.get_text_representer()
            for n in self.graph.nodes
            if n.node_source == NodeSource.INFERENCE_BASED and n.active
        }

        self._plot_ops._capture_state(
            sentence_idx=i,
            sentence_text=original_text,
            mode="paper",
            triplets=all_triplets,
            explicit_nodes=explicit_nodes_for_plot,
            inactive_nodes=sorted(all_nodes - nodes_with_active_edges),
            salient_nodes=carryover_nodes,
            inferred_nodes=sorted(inferred_node_names & nodes_with_active_edges),
            active_edges=active_edge_set,
        )

    def plot_sentence_views_wrapper(
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
        self._viz_positions = self._plot_ops.get_viz_positions()
        self._previous_active_triplets = self._plot_ops._previous_active_triplets

    def plot_paper_graph_style_wrapper(
        self,
        sentence_idx: int,
        original_text: str,
        graphs_output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
    ) -> None:
        # Paper plot: all edges for layout, active edges highlighted
        all_triplets = self._triplet_ops.reconstruct_semantic_triplets(
            only_active=False
        )
        active_triplets = self._triplet_ops.reconstruct_semantic_triplets(
            only_active=True
        )

        active_nodes, _ = self.graph.get_active_subgraph_wrapper()
        active_node_names = {n.get_text_representer() for n in active_nodes}

        # Only active inferred nodes
        inferred_node_names = {
            n.get_text_representer()
            for n in self.graph.nodes
            if n.node_source == NodeSource.INFERENCE_BASED and n.active
        }
        # Only active explicit nodes
        explicit_node_names = [
            n.get_text_representer()
            for n in self._explicit_nodes_current_sentence
            if n.get_text_representer()
        ]

        self._plot_ops.plot_paper_graph_style(
            sentence_index=sentence_idx,
            sentence_text=original_text,
            output_dir=graphs_output_dir,
            highlight_nodes=highlight_nodes,
            all_triplets=all_triplets,
            active_triplets=active_triplets,
            active_node_names=active_node_names,
            inferred_node_names=inferred_node_names,
            explicit_node_names=explicit_node_names,
        )
        self._viz_positions = self._plot_ops.get_viz_positions()

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
            "story text: %s", self.story_text[:200] if self.story_text else "none"
        )
        doc = self.spacy_nlp(self.story_text)
        sentences = list(doc.sents)
        logging.info("spacy found %d sentences", len(sentences))
        for i, sent in enumerate(sentences):
            logging.info("sentence %d: %s", i + 1, sent.text.strip()[:100])

        self.initialize_run_state()

        resolved_sentences = self.resolve_story_sentences(replace_pronouns)

        prev_sentences = []
        self._sentence_triplets = []
        sentence_counter = 0
        text_prefix_pattern = re.compile(r"^The text is:\s*", re.IGNORECASE)

        for i, (sent, resolved_text, original_text) in enumerate(resolved_sentences):
            if re.match(r"^\s*(user|system|assistant)\b", original_text.lower()):
                continue
            sentence_counter += 1
            original_text = text_prefix_pattern.sub("", original_text)
            resolved_text = text_prefix_pattern.sub("", resolved_text)
            self.active_graph.reset()
            self._current_sentence_index = sentence_counter

            logging.info(
                f"before sentence {sentence_counter} | "
                f"nodes: {len(self.graph.nodes)} | "
                f"edges: {len(self.graph.edges)} | "
                f"graph id: {id(self.graph)}"
            )

            self._sentence_ops.configure_graph_for_sentence(
                self._current_sentence_index,
                self._sentence_ops.extract_sentence_lemmas(original_text),
            )

            _previous_graph_state = self.snapshot_graph_state()
            _previous_carryover_nodes = set(self._carryover_nodes_current_sentence)

            nodes_before_sentence, should_skip_sentence = (
                self.process_sentence_core_wrapper(
                    i, sent, resolved_text, original_text, prev_sentences
                )
            )
            if should_skip_sentence:
                continue

            # At this point, all graph modifications for this sentence are complete
            # build the per-sentence view with the final graph state
            self._per_sentence_view = self.build_per_sentence_view_wrapper(
                explicit_nodes=list(self._explicit_nodes_current_sentence),
                sentence_index=self._current_sentence_index,
            )

            logging.info(
                f"after sentence {sentence_counter} | "
                f"nodes: {len(self.graph.nodes)} | "
                f"active edges: {sum(1 for e in self.graph.edges if e.active)} | "
                f"graph id: {id(self.graph)}"
            )

            sentence_id = i + 1
            newly_inferred_nodes = (
                self._projection_bookkeeping_ops.compute_newly_inferred_nodes(
                    nodes_before_sentence
                )
            )

            # CONNECTIVITY
            rollback_needed = self.stabilize_connectivity_wrapper(prev_sentences)

            if rollback_needed:
                self.handle_sentence_rollback_wrapper(
                    i=i,
                    original_text=original_text,
                    prev_sentences=prev_sentences,
                    previous_graph_state=_previous_graph_state,
                    previous_carryover_nodes=_previous_carryover_nodes,
                )
                # Nothing new was inferred since graph was restored
                newly_inferred_nodes = set()
                # Keep _per_sentence_view = None so plotting reads active edges
                # directly from the restored graph (fallback path)
                # Carryover already restored by rollback handler — don't overwrite
            else:
                # Update carryover nodes from projection (normal path only)
                if self._per_sentence_view is not None:
                    self._carryover_nodes_current_sentence.clear()
                    self._carryover_nodes_current_sentence.update(
                        self._per_sentence_view.carryover_nodes
                    )
                else:
                    self._carryover_nodes_current_sentence.clear()

            if rollback_needed:
                # Rollback: compute plot lists from the restored graph state.
                # Explicit = this sentence's nodes (may be empty or dangling).
                # Salient = all active nodes in restored graph minus explicit.
                # Inactive = all truly inactive nodes in restored graph.
                explicit_nodes_for_plot = [
                    n.get_text_representer()
                    for n in self._explicit_nodes_current_sentence
                    if n.get_text_representer()
                ]
                active_node_names = {
                    n.get_text_representer()
                    for n in self.graph.nodes
                    if n.active and n.get_text_representer()
                }
                explicit_set = set(explicit_nodes_for_plot)
                salient_nodes_for_plot = sorted(active_node_names - explicit_set)
                inactive_nodes_for_plot = sorted(
                    n.get_text_representer()
                    for n in self.graph.nodes
                    if not n.active and n.get_text_representer()
                )
                recently_deactivated_nodes = set()
            else:
                (
                    recently_deactivated_nodes,
                    explicit_nodes_for_plot,
                    salient_nodes_for_plot,
                    inactive_nodes_for_plot,
                ) = self.update_post_projection_state_wrapper(
                    sentence_id, i, newly_inferred_nodes, self._per_sentence_view
                )

            logging.info(
                f"after projection update, inactive nodes: {inactive_nodes_for_plot}"
            )

            if plot_after_each_sentence:
                self.plot_sentence_views_wrapper(
                    i,
                    original_text,
                    graphs_output_dir,
                    highlight_nodes,
                    inactive_nodes_for_plot,
                    salient_nodes_for_plot,
                    largest_component_only,
                )
                self.plot_paper_graph_style_wrapper(
                    i,
                    original_text,
                    graphs_output_dir,
                    highlight_nodes,
                )
            elif self._plot_ops._collect_states:
                self.capture_state_only_wrapper(
                    i,
                    original_text,
                    inactive_nodes_for_plot,
                    salient_nodes_for_plot,
                )
            self.capture_sentence_triplets_wrapper(original_text)

        # Ensure cumulative graph is connected
        if not self._connectivity_ops.is_cumulative_connected_wrapper():
            logging.warning(
                "Cumulative graph disconnected after sentence loop - repairing"
            )
            self.stabilize_connectivity_wrapper(prev_sentences)

        return self.finalize_run_outputs_wrapper(matrix_suffix)

    def infer_new_relationships_bootstrap_wrapper(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        return self._inference_ops.infer_new_relationships_step_0(sent)

    def infer_new_relationships_for_sentence_wrapper(
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

    def reactivate_relevant_edges_wrapper(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        self._activation_ops.set_decay_sentence_context(
            idx=self._current_sentence_index, text=self._current_sentence_text
        )
        self._activation_ops.reactivate_relevant_edges(
            active_nodes=active_nodes,
            prev_sentences_text=prev_sentences_text,
            newly_added_edges=newly_added_edges,
        )

    def add_inferred_relationships_to_graph_step_0_wrapper(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.collect_sentence_text_based_nodes_wrapper(
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

    def add_inferred_relationships_to_graph_wrapper(
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

    def resolve_node_from_text_wrapper(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        return self._node_ops.get_or_create_node_from_text(
            text=text,
            curr_sentences_nodes=curr_sentences_nodes,
            curr_sentences_words=curr_sentences_words,
            node_source=node_source,
            create_node=create_node,
        )

    def resolve_node_from_new_relationship_wrapper(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        return self._node_ops.get_or_create_node_from_relationship(
            text=text,
            graph_active_nodes=graph_active_nodes,
            curr_sentences_nodes=curr_sentences_nodes,
            curr_sentences_words=curr_sentences_words,
            node_source=node_source,
            create_node=create_node,
        )

    def extract_phrase_level_concepts_wrapper(self, sent):
        return self._node_ops.extract_main_nouns(
            sent=sent,
            admit_node_fn=lambda lemma, node_type, provenance, sent=None: self._node_ops.admit_node(
                lemma=lemma,
                node_type=node_type,
                provenance=provenance,
                sent=sent,
            ),
        )

    def collect_sentence_text_based_nodes_wrapper(
        self,
        previous_sentences: List[Span],
        create_unexistent_nodes: bool = True,
    ) -> Tuple[List[Node], List[str]]:
        return self._node_ops.extract_explicit_nodes(
            previous_sentences=previous_sentences,
            current_sentence_index=self._current_sentence_index,
            create_unexistent_nodes=create_unexistent_nodes,
        )
