from amoc.admission.edge_admission import EdgeAdmission
from amoc.admission.node_admission import NodeAdmission
from amoc.admission.text_normalizer import TextNormalizer
from amoc.construction.relationship_builder import RelationshipGraphBuilder
from amoc.construction.sentence_builder import SentenceGraphBuilder
from amoc.extraction.linguistic import LinguisticProcessing
from amoc.extraction.inference import Inference
from amoc.output.plotter import GraphPlotter
from amoc.output.finalizer import OutputFinalizer
from amoc.output.recorder import TripletRecorder
from amoc.projection.state_manager import ProjectionStateManager
from amoc.runtime.sentence_runtime import SentenceRuntime
from amoc.memory.decay import Decay
from amoc.connectivity.stabilizer import ConnectivityStabilizer


def wire_core_dependencies(core) -> None:
    core._connectivity_ops = ConnectivityStabilizer(
        graph_ref=core.graph,
        get_explicit_nodes=core._get_explicit_nodes,
        get_carryover_nodes=core._get_carryover_nodes,
        edge_visibility=core.edge_visibility,
        llm_extractor=core.client,
    )
    core._text_filter_ops = TextNormalizer(
        spacy_nlp=core.spacy_nlp,
        graph_ref=core.graph,
        story_lemmas=core.story_lemmas,
    )

    core._normalize_edge_label = lambda l: core._text_filter_ops.normalize_edge_label(l)
    core._is_valid_relation_label = (
        lambda l: core._text_filter_ops.is_valid_relation_label(l)
    )
    core._normalize_endpoint_text = (
        lambda text, is_subject: core._text_filter_ops.extract_canonical_node_lemma(
            text, is_subject
        )
    )
    core._get_sentences_nodes = lambda sents, create_unexistent_nodes=True: core.collect_sentence_text_based_nodes_wrapper(
        sents, create_unexistent_nodes=create_unexistent_nodes
    )

    core._triplet_ops = TripletRecorder(
        graph_ref=core.graph,
        cumulative_graph_ref=core.cumulative_graph,
        active_graph_ref=core.active_graph,
        triplet_intro_ref=core._triplet_intro,
    )
    core._edge_ops = EdgeAdmission(
        graph_ref=core.graph,
        llm_extractor=core.client,
        spacy_nlp=core.spacy_nlp,
        get_explicit_nodes=core._get_explicit_nodes,
        get_carryover_nodes=core._get_carryover_nodes,
        get_attachable_nodes=lambda: core._sentence_ops.get_attachable_nodes_for_sentence(
            core._get_active_edge_nodes
        ),
        edge_visibility=core.edge_visibility,
        debug=core.debug,
    )
    core._node_ops = NodeAdmission(
        graph_ref=core.graph,
        spacy_nlp=core.spacy_nlp,
        story_lemmas=core.story_lemmas,
        persona_only_lemmas=core._persona_only_lemmas,
        max_distance_from_active_nodes=core.max_distance_from_active_nodes,
        debug=core.debug,
    )
    core._node_ops.set_callbacks(
        has_active_attachment_fn=lambda l: core._activation_ops.has_active_attachment(
            l
        ),
        canonicalize_and_classify_fn=lambda t: core._text_filter_ops.normalize_and_classify_node(
            t
        ),
    )
    core._sentence_ops = SentenceRuntime(
        graph_ref=core.graph,
        spacy_nlp=core.spacy_nlp,
        story_lemmas=core.story_lemmas,
        max_distance_from_active_nodes=core.max_distance_from_active_nodes,
        edge_visibility=core.edge_visibility,
    )
    core._sentence_ops.set_runtime_state_refs(
        anchor_nodes=core._anchor_nodes,
        explicit_nodes=core._explicit_nodes_current_sentence,
        triplet_intro=core._triplet_intro,
    )
    core._inference_ops = Inference(
        graph_ref=core.graph,
        llm_extractor=core.client,
        spacy_nlp=core.spacy_nlp,
        max_new_concepts=core.max_new_concepts,
        max_new_properties=core.max_new_properties,
        persona=core.persona,
    )
    core._inference_ops.set_callbacks(
        append_adjectival_hints_fn=lambda n, s: core._linguistic_ops.append_adjectival_hints(
            n, s
        ),
        get_sentences_text_based_nodes_fn=core._get_sentences_nodes,
    )
    core._linguistic_ops = LinguisticProcessing(
        graph_ref=core.graph,
        spacy_nlp=core.spacy_nlp,
        llm_extractor=core.client,
        story_lemmas=core.story_lemmas,
        persona=core.persona,
    )
    core._linguistic_ops.set_callbacks(
        add_edge_fn=core.add_edge_wrapper,
    )
    core._plot_ops = GraphPlotter(
        graph_ref=core.graph,
        output_dir=core.matrix_dir_base,
        model_name=core.model_name,
        persona=core.persona,
        persona_age=core.persona_age,
        layout_depth=core._layout_depth,
    )
    core._plot_ops.set_callbacks(
        get_explicit_nodes_fn=core._get_explicit_nodes,
        get_edge_activation_scores_fn=lambda: core._triplet_ops.get_edge_activation_scores(),
        graph_edges_to_triplets_fn=lambda only_active=False: core._triplet_ops.graph_edges_to_triplets(
            only_active
        ),
        enforce_cumulative_connectivity_fn=lambda: core._connectivity_ops.warn_if_cumulative_disconnected(),
    )
    core._plot_ops.set_lemmas(
        story_lemmas=core.story_lemmas,
        persona_only_lemmas=core._persona_only_lemmas,
    )
    core._activation_ops = Decay(
        graph_ref=core.graph,
        llm_extractor=core.client,
        get_explicit_nodes=core._get_explicit_nodes,
        max_distance=core.max_distance_from_active_nodes,
        edge_visibility=core.edge_visibility,
        nr_relevant_edges=core.nr_relevant_edges,
        strict_reactivate=core.strict_reactivate_function,
    )
    core._activation_ops.set_decay_state_refs(
        anchor_nodes=core._anchor_nodes,
        record_edge_fn=lambda e, i: core._edge_ops.record_edge_in_graphs(
            edge=e,
            sentence_idx=i,
            cumulative_graph=core.cumulative_graph,
            active_graph=core.active_graph,
            cumulative_triplet_records=core._cumulative_triplet_records,
            cumulative_graph_builder=core.cumulative_graph_builder,
            active_graph_builder=core.active_graph_builder,
        ),
    )
    core._output_ops = OutputFinalizer(
        graph_ref=core.graph,
        model_name=core.model_name,
        persona=core.persona,
        persona_age=core.persona_age,
        story_text=core.story_text,
        matrix_dir_base=core.matrix_dir_base,
    )
    core._relationship_graph_ops = RelationshipGraphBuilder(
        graph_ref=core.graph,
        spacy_nlp=core.spacy_nlp,
        edge_visibility=core.edge_visibility,
        debug=core.debug,
    )
    core._sentence_processing_ops = SentenceGraphBuilder(
        graph_ref=core.graph,
        llm_extractor=core.client,
        spacy_nlp=core.spacy_nlp,
        max_distance_from_active_nodes=core.max_distance_from_active_nodes,
        edge_visibility=core.edge_visibility,
        context_length=core.context_length,
        text_normalizer=core._text_filter_ops,
        debug=core.debug,
    )
    core._edge_ops.configure_edge_admission_with_core(core)
    core._relationship_graph_ops.configure_with_core(core)
    core._sentence_processing_ops.configure_sentence_builder_with_core(core)
    core._projection_bookkeeping_ops = ProjectionStateManager(
        graph_ref=core.graph,
        max_distance=core.max_distance_from_active_nodes,
        debug=core.debug,
    )
    core._projection_bookkeeping_ops.set_callbacks(
        record_sentence_activation_fn=core.record_activation_matrix_wrapper,
    )
