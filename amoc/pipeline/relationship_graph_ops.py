import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from amoc.graph.node import NodeType, NodeSource, NodeProvenance
from amoc.nlp.spacy_utils import get_concept_lemmas

if TYPE_CHECKING:
    from amoc.pipeline.core import AMoCv4


class RelationshipGraphOps:
    def __init__(
        self,
        graph_ref,
        spacy_nlp,
        edge_visibility: int,
        debug: bool = False,
    ):
        self.graph = graph_ref
        self.spacy_nlp = spacy_nlp
        self.edge_visibility = edge_visibility
        self.debug = debug

        # Callbacks to be set by parent
        self._normalize_endpoint_text_fn = None
        self._normalize_edge_label_fn = None
        self._is_valid_relation_label_fn = None
        self._validate_node_provenance_fn = None
        self._admit_node_fn = None
        self._passes_attachment_constraint_fn = None
        self._canonicalize_edge_direction_fn = None
        self._canonicalize_and_classify_node_text_fn = None
        self._classify_relation_fn = None
        self._add_edge_fn = None
        self._get_nodes_with_active_edges_fn = None
        self._get_node_from_text_fn = None
        self._get_node_from_new_relationship_fn = None
        self._get_concept_lemmas_fn = None
        self._appears_in_story_fn = None

        self._explicit_nodes_ref = None
        self._current_sentence_index = 0
        self._new_inferred_nodes_count = 0

    def set_callbacks(
        self,
        normalize_endpoint_text_fn,
        normalize_edge_label_fn,
        is_valid_relation_label_fn,
        validate_node_provenance_fn,
        admit_node_fn,
        passes_attachment_constraint_fn,
        canonicalize_edge_direction_fn,
        canonicalize_and_classify_node_text_fn,
        classify_relation_fn,
        add_edge_fn,
        get_nodes_with_active_edges_fn,
        get_node_from_text_fn,
        get_node_from_new_relationship_fn,
        get_concept_lemmas_fn,
        appears_in_story_fn,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self._validate_node_provenance_fn = validate_node_provenance_fn
        self._admit_node_fn = admit_node_fn
        self._passes_attachment_constraint_fn = passes_attachment_constraint_fn
        self._canonicalize_edge_direction_fn = canonicalize_edge_direction_fn
        self._canonicalize_and_classify_node_text_fn = (
            canonicalize_and_classify_node_text_fn
        )
        self._classify_relation_fn = classify_relation_fn
        self._add_edge_fn = add_edge_fn
        self._get_nodes_with_active_edges_fn = get_nodes_with_active_edges_fn
        self._get_node_from_text_fn = get_node_from_text_fn
        self._get_node_from_new_relationship_fn = get_node_from_new_relationship_fn
        self._get_concept_lemmas_fn = get_concept_lemmas_fn
        self._appears_in_story_fn = appears_in_story_fn

    def set_state_refs(self, explicit_nodes_ref):
        self._explicit_nodes_ref = explicit_nodes_ref

    def configure_with_core(self, core: "AMoCv4") -> None:
        self.set_callbacks(
            normalize_endpoint_text_fn=core._normalize_endpoint_text,
            normalize_edge_label_fn=core._normalize_edge_label,
            is_valid_relation_label_fn=core._is_valid_relation_label,
            validate_node_provenance_fn=lambda l, t=None, allow_bootstrap=False: (
                core._node_ops.validate_node_provenance(
                    l, t, allow_bootstrap=allow_bootstrap
                )
            ),
            admit_node_fn=lambda l, nt, p, s=None: core._node_ops.admit_node(
                l, nt, p, s
            ),
            passes_attachment_constraint_fn=core._passes_attachment_constraint,
            canonicalize_edge_direction_fn=core._canonicalize_edge_direction,
            canonicalize_and_classify_node_text_fn=lambda t: (
                core._text_filter_ops.canonicalize_and_classify_node_text(t)
            ),
            classify_relation_fn=core._classify_relation,
            add_edge_fn=core._add_edge,
            get_nodes_with_active_edges_fn=core._get_active_edge_nodes,
            get_node_from_text_fn=core._resolve_node_from_text,
            get_node_from_new_relationship_fn=core._resolve_node_from_new_relationship,
            get_concept_lemmas_fn=lambda text: get_concept_lemmas(core.spacy_nlp, text),
            appears_in_story_fn=lambda t, check_graph=False: (
                core._text_filter_ops.appears_in_story(t, check_graph=check_graph)
            ),
        )
        self.set_state_refs(explicit_nodes_ref=core._get_explicit_nodes)

    def set_current_sentence(self, sentence_index: int):
        self._current_sentence_index = sentence_index
        self._new_inferred_nodes_count = 0

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent,
        current_sentence_text_based_nodes: List,
        current_sentence_text_based_words: List[str],
    ) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        for relationship in inferred_relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue

            norm_subj = self._normalize_endpoint_text_fn(
                relationship[0], is_subject=True
            )
            norm_obj = self._normalize_endpoint_text_fn(
                relationship[2], is_subject=False
            )
            if norm_subj is None or norm_obj is None:
                continue

            subj_node_existing = self.graph.get_node(
                self._get_concept_lemmas_fn(relationship[0])
            )
            obj_node_existing = self.graph.get_node(
                self._get_concept_lemmas_fn(relationship[2])
            )

            active_nodes_set = set(self._get_nodes_with_active_edges_fn())

            attachment_ok = self._passes_attachment_constraint_fn(
                relationship[0],
                relationship[2],
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges_fn(),
            )

            # SAFE RELAXATION: Allow if at least one endpoint is already active
            if not attachment_ok:
                existing_source = self.graph.get_node(
                    self._get_concept_lemmas_fn(relationship[0])
                )
                existing_dest = self.graph.get_node(
                    self._get_concept_lemmas_fn(relationship[2])
                )

                source_active = existing_source in active_nodes_set
                dest_active = existing_dest in active_nodes_set

                if not (source_active or dest_active):
                    continue

            subj, subj_type = self._canonicalize_and_classify_node_text_fn(
                relationship[0]
            )
            obj, obj_type = self._canonicalize_and_classify_node_text_fn(
                relationship[2]
            )
            if subj_type is None or obj_type is None:
                continue

            source_node = self._get_node_from_text_fn(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self._get_node_from_text_fn(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )

            edge_label = relationship[1].replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)
            if not self._is_valid_relation_label_fn(edge_label):
                continue

            if source_node is None:
                if not self._validate_node_provenance_fn(
                    subj,
                    allow_bootstrap=(
                        dest_node is not None
                        and (
                            dest_node in explicit_nodes
                            or dest_node in self._get_nodes_with_active_edges_fn()
                        )
                    ),
                ):
                    continue
                if not self._admit_node_fn(
                    lemma=subj,
                    node_type=subj_type,
                    provenance="INFERRED_RELATION",
                ):
                    continue

                source_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if dest_node is None:
                allow_bootstrap = source_node is not None and (
                    source_node in explicit_nodes
                    or source_node in self._get_nodes_with_active_edges_fn()
                )

                if not self._validate_node_provenance_fn(
                    obj,
                    allow_bootstrap=allow_bootstrap,
                ):
                    continue

                dest_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if source_node is None or dest_node is None:
                continue

            canon_label, canon_src, canon_dst, was_swapped = (
                self._canonicalize_edge_direction_fn(
                    edge_label,
                    source_node.get_text_representer(),
                    dest_node.get_text_representer(),
                )
            )
            if was_swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            self._add_edge_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List,
        curr_sentences_words: List[str],
        active_graph_nodes: List,
        added_edges: List,
    ) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        for relationship in inferred_relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue
            if not isinstance(relationship[0], str) or not isinstance(
                relationship[2], str
            ):
                continue

            norm_subj = self._normalize_endpoint_text_fn(
                relationship[0], is_subject=True
            )
            norm_obj = self._normalize_endpoint_text_fn(
                relationship[2], is_subject=False
            )

            if norm_subj is None or norm_obj is None:
                continue

            if not self._passes_attachment_constraint_fn(
                relationship[0],
                relationship[2],
                curr_sentences_words,
                curr_sentences_nodes,
                active_graph_nodes,
                self._get_nodes_with_active_edges_fn(),
            ):
                continue

            subj, subj_type = self._canonicalize_and_classify_node_text_fn(
                relationship[0]
            )
            obj, obj_type = self._canonicalize_and_classify_node_text_fn(
                relationship[2]
            )

            if subj_type is None or obj_type is None:
                continue

            source_node = self._get_node_from_new_relationship_fn(
                norm_subj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )

            dest_node = self._get_node_from_new_relationship_fn(
                norm_obj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )

            edge_label = relationship[1].replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)

            if not self._is_valid_relation_label_fn(edge_label):
                continue

            if source_node is None:
                if not self._validate_node_provenance_fn(
                    subj,
                    allow_bootstrap=(
                        dest_node is not None and dest_node in explicit_nodes
                    ),
                ):
                    continue

                source_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if dest_node is None:
                if not self._validate_node_provenance_fn(
                    obj, allow_bootstrap=(source_node is not None)
                ):
                    continue

                dest_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if source_node is None or dest_node is None:
                continue

            canon_label, canon_src, canon_dst, was_swapped = (
                self._canonicalize_edge_direction_fn(
                    edge_label,
                    source_node.get_text_representer(),
                    dest_node.get_text_representer(),
                )
            )

            if was_swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            source_active = source_node.active
            dest_active = dest_node.active

            if not (
                source_active
                or dest_active
                or self._appears_in_story_fn(relationship[0], check_graph=False)
                or self._appears_in_story_fn(relationship[2], check_graph=False)
            ):
                continue

            explicit_count = len(explicit_nodes)
            active_count = len(self._get_nodes_with_active_edges_fn())

            base_budget = max(3, explicit_count)
            bonus = 2 if active_count < explicit_count * 2 else 0
            MAX_NEW_INFERRED = min(base_budget + bonus, 8)

            explicit_keys = {tuple(n.lemmas) for n in explicit_nodes}

            new_node_created = False
            source_created_at_sentence = source_node.__dict__.get("created_at_sentence")
            dest_created_at_sentence = dest_node.__dict__.get("created_at_sentence")

            if (
                source_node.node_source == NodeSource.INFERENCE_BASED
                and source_created_at_sentence == self._current_sentence_index
                and tuple(source_node.lemmas) not in explicit_keys
            ):
                new_node_created = True

            if (
                dest_node.node_source == NodeSource.INFERENCE_BASED
                and dest_created_at_sentence == self._current_sentence_index
                and tuple(dest_node.lemmas) not in explicit_keys
            ):
                new_node_created = True

            if new_node_created:
                if self._new_inferred_nodes_count >= MAX_NEW_INFERRED:
                    continue
                self._new_inferred_nodes_count += 1

            explicit_set = set(explicit_nodes)

            if not (
                source_node.active
                or dest_node.active
                or source_node in explicit_set
                or dest_node in explicit_set
            ):
                continue

            potential_edge = self._add_edge_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

            if potential_edge:
                added_edges.append(potential_edge)
