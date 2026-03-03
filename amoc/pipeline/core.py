import logging
import re
from typing import List, Tuple, Optional, Iterable

import networkx as nx
from spacy.tokens import Span

from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph.per_sentence_graph import PerSentenceGraph
from amoc.llm.vllm_client import VLLMClient

from amoc.pipeline.text_filter_ops import TextFilterOps
from amoc.pipeline.triplet_ops import TripletOps
from amoc.pipeline.edge_ops import EdgeOps
from amoc.pipeline.node_ops import NodeOps
from amoc.pipeline.sentence_ops import SentenceOps
from amoc.pipeline.inference_ops import InferenceOps
from amoc.pipeline.linguistic_ops import LinguisticOps
from amoc.pipeline.plot_ops import PlotOps
from amoc.pipeline.output_ops import OutputOps
from amoc.pipeline.sentence_processing_ops import SentenceProcessingOps
from amoc.pipeline.relationship_graph_ops import RelationshipGraphOps
from amoc.pipeline.init_ops import InitOps


class AMoCv4:
    ENFORCE_ATTACHMENT_CONSTRAINT = True

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

        self.strict_reactivate_function = strict_reactivate_function
        self.strict_attachament_constraint = strict_attachament_constraint
        self.single_anchor_hub = single_anchor_hub
        self.allow_multi_edges = allow_multi_edges
        self.checkpoint = checkpoint

        self._current_sentence_text: str = ""
        self._current_sentence_index: Optional[int] = None

        self.cumulative_graph = nx.MultiDiGraph()

        self._explicit_nodes_current_sentence: set[Node] = set()
        self._carryover_nodes_current_sentence: set[Node] = set()
        self._historical_explicit_nodes: set[Node] = set()

        self._triplet_intro: dict[tuple[str, str, str], int] = {}
        self._cumulative_triplet_records: list[dict] = []
        self._fixed_hub = None
        self._per_sentence_view: Optional[PerSentenceGraph] = None
        self._ever_admitted_nodes: set[str] = set()
        self._layout_depth = 3
        self._persistent_is_edges: set[tuple[str, str, str]] = set()
        self._amoc_matrix_records: list[dict] = []
        self._previous_active_triplets: list[tuple[str, str, str]] = []
        self._story_lemma_set: set[str] = set()
        self._sentence_triplets: list[
            tuple[int, str, str, str, str, bool, bool, int]
        ] = []
        self._new_inferred_nodes_count: int = 0

        self._dummy_anchor_nodes: set[Node] = set()

        self._get_explicit_nodes = lambda: self._explicit_nodes_current_sentence
        self._get_carryover_nodes = lambda: self._carryover_nodes_current_sentence
        self._get_active_edge_nodes = self._compute_nodes_with_active_edges

        self._setup_ops_classes()

    def _setup_ops_classes(self):
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
            active_graph_ref=None,
            triplet_intro_ref=self._triplet_intro,
        )
        self._edge_ops = EdgeOps(
            graph_ref=self.graph,
            client_ref=self.client,
            spacy_nlp=self.spacy_nlp,
            get_explicit_nodes=self._get_explicit_nodes,
            get_carryover_nodes=self._get_carryover_nodes,
            get_attachable_nodes=lambda: (
                set(self._explicit_nodes_current_sentence)
                | set(self._carryover_nodes_current_sentence)
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
            has_active_attachment_fn=lambda l: any(
                l in n.lemmas for n in self.graph.nodes
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
            anchor_nodes=self._dummy_anchor_nodes,
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
            enforce_cumulative_connectivity_fn=lambda: None,
        )
        self._plot_ops.set_lemmas(
            story_lemmas=self.story_lemmas,
            persona_only_lemmas=self._persona_only_lemmas,
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

        self._edge_ops.configure_with_core(self)
        self._relationship_graph_ops.configure_with_core(self)

        self._init_ops.set_callbacks(
            normalize_endpoint_text_fn=self._normalize_endpoint_text,
            normalize_edge_label_fn=self._normalize_edge_label,
            is_valid_relation_label_fn=self._is_valid_relation_label,
            passes_attachment_constraint_fn=self._passes_attachment_constraint,
            canonicalize_edge_direction_fn=self._canonicalize_edge_direction,
            classify_relation_fn=self._classify_relation,
            add_edge_fn=self._add_edge,
            get_nodes_with_active_edges_fn=self._get_active_edge_nodes,
            get_node_from_text_fn=self._resolve_node_from_text,
            get_sentences_text_based_nodes_fn=self._get_sentences_nodes,
            extract_deterministic_structure_fn=lambda s, n, w: (
                self._linguistic_ops.set_sentence_context(
                    self._current_sentence_index, self.edge_visibility
                ),
                self._linguistic_ops.extract_deterministic_structure(s, n, w),
            )[-1],
            infer_new_relationships_step_0_fn=self._infer_new_relationships_bootstrap,
            add_inferred_relationships_to_graph_step_0_fn=self.add_inferred_relationships_to_graph_step_0,
            restrict_active_to_current_explicit_fn=lambda en: None,
        )
        self._init_ops.set_state_refs(
            explicit_nodes_ref=self._get_explicit_nodes,
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
            get_nodes_with_active_edges_fn=self._get_active_edge_nodes,
            append_adjectival_hints_fn=lambda n, s: self._linguistic_ops.append_adjectival_hints(
                n, s
            ),
            extract_deterministic_structure_fn=lambda s, n, w: (
                self._linguistic_ops.set_sentence_context(
                    self._current_sentence_index, self.edge_visibility
                ),
                self._linguistic_ops.extract_deterministic_structure(s, n, w),
            )[-1],
            infer_edges_to_recently_deactivated_fn=lambda *args, **kwargs: [],
            restrict_active_to_current_explicit_fn=lambda en: None,
            get_node_from_new_relationship_fn=self._resolve_node_from_new_relationship,
            get_phrase_level_concepts_fn=self._extract_phrase_level_concepts,
            get_sentences_text_based_nodes_fn=self._get_sentences_nodes,
            infer_new_relationships_fn=self._infer_new_relationships_for_sentence,
            add_inferred_relationships_to_graph_fn=self.add_inferred_relationships_to_graph,
        )
        self._sentence_processing_ops.set_state_refs(
            explicit_nodes_ref=self._get_explicit_nodes,
            anchor_nodes_ref=self._dummy_anchor_nodes,
            triplet_intro_ref=self._triplet_intro,
            carryover_nodes_ref=self._get_carryover_nodes,
            persona=self.persona,
        )

    def _compute_nodes_with_active_edges(self) -> set[Node]:
        nodes: set[Node] = set()
        for edge in self.graph.edges:
            if edge.active:
                nodes.add(edge.source_node)
                nodes.add(edge.dest_node)
        return nodes

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
            explicit_nodes=self._explicit_nodes_current_sentence,
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
            persona_influenced=False,
        )
        if edge:
            self._record_edge_in_graphs(edge, self._current_sentence_index)
        return edge

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        self._edge_ops.record_edge_in_graphs(
            edge=edge,
            sentence_idx=sentence_idx,
            cumulative_graph=self.cumulative_graph,
            active_graph=None,
            cumulative_triplet_records=self._cumulative_triplet_records,
        )

    def _initialize_run_state(self) -> None:
        self._previous_active_triplets = []
        self._plot_ops.viz_positions = {}
        self._historical_explicit_nodes.clear()

    def _resolve_story_sentences(self, replace_pronouns: bool) -> list:
        resolved_sentences, story_lemma_set = self._sentence_ops.resolve_sentences(
            story_text=self.story_text,
            replace_pronouns=replace_pronouns,
            resolve_pronouns_fn=self._linguistic_ops.resolve_pronouns,
        )
        self._story_lemma_set = story_lemma_set
        return resolved_sentences

    def _prepare_sentence_runtime_state(self, original_text: str) -> set:
        nodes_before_sentence = self._sentence_ops.reset_sentence_state(original_text)
        self._current_sentence_text = original_text
        self._explicit_nodes_current_sentence.clear()
        self._carryover_nodes_current_sentence.clear()
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
        nodes_before, should_skip, explicit_nodes, _anchor_nodes = result
        self._explicit_nodes_current_sentence.clear()
        self._explicit_nodes_current_sentence.update(explicit_nodes)
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
            anchor_nodes=self._dummy_anchor_nodes,
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

        return result

    def _apply_edge_decay_paper(self, prev_sentences_text: str) -> None:
        edges_text, edges = self.graph.get_edges_str(
            self.graph.nodes, only_active=False
        )

        raw_indices = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, None
        )

        valid_indices: set[int] = set()
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(edges):
                valid_indices.add(i)

        for idx, edge in enumerate(edges, start=1):
            if (
                edge.created_at_sentence == self._current_sentence_index
                or idx in valid_indices
            ):
                edge.visibility_score = self.edge_visibility
                continue

            edge.visibility_score -= 1
            if edge.visibility_score <= 0:
                edge.visibility_score = 0

    def _compute_carryover_nodes(self, current_explicit_nodes: set[Node]) -> set[Node]:
        if not current_explicit_nodes:
            return set()

        G_active = nx.Graph()
        for edge in self.graph.edges:
            if edge.active:
                G_active.add_edge(edge.source_node, edge.dest_node)

        for node in self._historical_explicit_nodes | current_explicit_nodes:
            G_active.add_node(node)

        carryover: set[Node] = set()
        for prev in self._historical_explicit_nodes:
            if prev in current_explicit_nodes:
                continue
            for curr in current_explicit_nodes:
                try:
                    if nx.has_path(G_active, prev, curr):
                        carryover.add(prev)
                        break
                except nx.NetworkXError:
                    continue

        return carryover

    def _build_paper_sentence_view(self, sentence_index: int) -> PerSentenceGraph:
        explicit_nodes = set(self._explicit_nodes_current_sentence)
        carryover_nodes = set(self._carryover_nodes_current_sentence)
        active_nodes = explicit_nodes | carryover_nodes

        active_edges = {
            edge
            for edge in self.graph.edges
            if edge.active
            and edge.source_node in active_nodes
            and edge.dest_node in active_nodes
        }

        return PerSentenceGraph(
            sentence_index=sentence_index,
            explicit_nodes=frozenset(explicit_nodes),
            carryover_nodes=frozenset(carryover_nodes),
            active_nodes=frozenset(active_nodes),
            active_edges=frozenset(active_edges),
            anchor_nodes=frozenset(),
        )

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

    def _record_sentence_triplets(self, original_text: str) -> None:
        self._triplet_ops.capture_sentence_triplets(
            original_text=original_text,
            current_sentence_index=self._current_sentence_index,
            explicit_nodes=self._explicit_nodes_current_sentence,
            nodes_with_active_edges=self._get_active_edge_nodes(),
            sentence_triplets=self._sentence_triplets,
            anchor_drop_log=None,
        )

    def _plot_sentence(
        self,
        i: int,
        original_text: str,
        graphs_output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        largest_component_only: bool,
    ):
        explicit_nodes_for_plot = sorted(
            {
                n.get_text_representer()
                for n in self._per_sentence_view.explicit_nodes
                if n.get_text_representer()
            }
        )
        salient_nodes_for_plot = sorted(
            {
                n.get_text_representer()
                for n in self._per_sentence_view.carryover_nodes
                if n.get_text_representer()
            }
        )

        all_nodes = {
            n.get_text_representer()
            for n in self.graph.nodes
            if n.get_text_representer()
        }
        active_nodes = set(explicit_nodes_for_plot) | set(salient_nodes_for_plot)
        inactive_nodes_for_plot = sorted(all_nodes - active_nodes)

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
            self._current_sentence_index = sentence_counter

            self._sentence_ops.configure_graph_for_sentence(
                self._current_sentence_index,
                self._sentence_ops.extract_sentence_lemmas(original_text),
            )

            nodes_before_sentence, should_skip_sentence = self._process_sentence_core(
                i, sent, resolved_text, original_text, prev_sentences
            )
            if should_skip_sentence:
                continue

            self._apply_edge_decay_paper(" ".join(prev_sentences))
            prev_sentences.append(original_text)

            self._historical_explicit_nodes |= set(
                self._explicit_nodes_current_sentence
            )
            carryover = self._compute_carryover_nodes(
                self._explicit_nodes_current_sentence
            )
            self._carryover_nodes_current_sentence.clear()
            self._carryover_nodes_current_sentence.update(carryover)

            self._per_sentence_view = self._build_paper_sentence_view(
                sentence_index=self._current_sentence_index
            )

            if plot_after_each_sentence:
                self._plot_sentence(
                    i,
                    original_text,
                    graphs_output_dir,
                    highlight_nodes,
                    largest_component_only,
                )
            self._record_sentence_triplets(original_text)

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
