import logging
from typing import TYPE_CHECKING, List, Optional, Tuple
from collections import defaultdict

from amoc.core.node import NodeType, NodeSource, NodeProvenance
from amoc.utils.spacy_utils import get_concept_lemmas
from amoc.config.constants import MAX_NEW_CONCEPTS, MAX_NEW_PROPERTIES
from amoc.admission.triplet_validator import TripletValidator

if TYPE_CHECKING:
    from amoc.pipeline.orchestrator import AMoCv4


class RelationshipGraphBuilder:
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
        self._is_node_allowed_fn = None
        self._admit_node_fn = None
        self.is_attachable_wrapper_fn = None
        self._canonicalize_and_classify_node_text_fn = None
        self.add_edge_wrapper_fn = None
        self.get_nodes_with_active_edges_fn = None
        self._get_node_from_text_fn = None
        self._get_node_from_new_relationship_fn = None
        self._get_concept_lemmas_fn = None
        self._appears_in_story_fn = None

        self._explicit_nodes_ref = None
        self._current_sentence_index = 0

        # set in configure_with_core
        self._triplet_validator = None

    def set_callbacks(
        self,
        normalize_endpoint_text_fn,
        normalize_edge_label_fn,
        is_valid_relation_label_fn,
        is_node_allowed_fn,
        admit_node_fn,
        is_attachable_fn,
        canonicalize_and_classify_node_text_fn,
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
        self._is_node_allowed_fn = is_node_allowed_fn
        self._admit_node_fn = admit_node_fn
        self.is_attachable_wrapper_fn = is_attachable_fn
        self._canonicalize_and_classify_node_text_fn = (
            canonicalize_and_classify_node_text_fn
        )
        self.add_edge_wrapper_fn = add_edge_fn
        self.get_nodes_with_active_edges_fn = get_nodes_with_active_edges_fn
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
            is_node_allowed_fn=lambda l, t=None, bypass=False: (
                core._node_ops.is_node_allowed(l, t, bypass=bypass)
            ),
            admit_node_fn=lambda l, nt, p, sent=None, is_first_sentence=False: core._node_ops.admit_node(
                lemma=l,
                node_type=nt,
                provenance=p,
                sent=sent,
                is_first_sentence=is_first_sentence,
            ),
            is_attachable_fn=core.is_attachable_wrapper,
            canonicalize_and_classify_node_text_fn=lambda t: (
                core._text_filter_ops.normalize_and_classify_node(t)
            ),
            add_edge_fn=core.add_edge_wrapper,
            get_nodes_with_active_edges_fn=core._get_active_edge_nodes,
            get_node_from_text_fn=core.resolve_node_from_text_wrapper,
            get_node_from_new_relationship_fn=core.resolve_node_from_new_relationship_wrapper,
            get_concept_lemmas_fn=lambda text: get_concept_lemmas(core.spacy_nlp, text),
            appears_in_story_fn=lambda t, check_graph=False: (
                core._text_filter_ops.is_grounded_in_story(t, check_graph=check_graph)
            ),
        )
        self.set_state_refs(explicit_nodes_ref=core._get_explicit_nodes)

        # issue: the triplet validation rules in triplet_validator do not apply to triplets with inferred nodes
        # Create a triplet validator for this class
        self._triplet_validator = TripletValidator(
            linguistic_ops=core._linguistic_ops,
            extract_deterministic_fn=core._linguistic_ops.extract_deterministic_structure,
            text_normalizer=core._text_filter_ops,
            client=core.client,
            persona=core.persona,
            spacy_nlp=self.spacy_nlp,
        )

    def validate_triplet(self, subj: str, rel: str, obj: str) -> tuple:
        if not self._triplet_validator:
            return True, subj, rel, obj

        # rel = not related, not associated etc. => reject
        if self._triplet_validator.is_negation_relation(rel):
            logging.debug(f"Rejected negation relation (raw): ({subj}, {rel}, {obj})")
            return False, None, None, None

        if self._triplet_validator.is_vague_relation(rel):
            logging.debug(f"Rejected vague relation (raw): ({subj}, {rel}, {obj})")
            return False, None, None, None

        # reject invalid labels
        if not self._triplet_validator.is_valid_relation_label(rel):
            logging.info(f"Inferred triplet rejected: invalid relation label '{rel}'")
            return False, None, None, None

        validation = self._triplet_validator.validate_triplet_relation((subj, rel, obj))

        if validation["action"] in ("reject", "reject_negation"):
            logging.info(
                f"Inferred triplet rejected: ({subj}, {rel}, {obj}) – {validation['reason']}"
            )
            return False, None, None, None

        # corrections such as swap of the subj/obj (ie. beautiful - is - king)
        if validation["action"] == "swap" and validation.get("corrected_triple"):
            corrected = validation["corrected_triple"]
            logging.info(
                f"Swapped inferred triplet: ({corrected[0]}, {corrected[1]}, {corrected[2]})"
            )
            return True, corrected[0], corrected[1], corrected[2]

        if validation["action"] == "add_copula" and validation.get("corrected_triple"):
            corrected = validation["corrected_triple"]
            logging.info(
                f"Added copula to inferred triplet: ({corrected[0]}, {corrected[1]}, {corrected[2]})"
            )
            return True, corrected[0], corrected[1], corrected[2]

        #  Normalize endpoints
        norm_subj, norm_obj = self._triplet_validator.normalize_endpoints(subj, obj)
        if norm_subj is None or norm_obj is None:
            logging.debug(
                f"Inferred triplet rejected: failed to normalize endpoints ({subj}, {obj})"
            )
            return False, None, None, None

        subj, obj = norm_subj, norm_obj

        # Clean and validate relation label
        edge_label = self._triplet_validator.clean_and_validate_relation(rel)
        if not edge_label:
            logging.debug(
                f"Inferred triplet rejected: failed to clean relation '{rel}'"
            )
            return False, None, None, None

        rel = edge_label

        return True, subj, rel, obj

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
        is_first_sentence: bool = False,
    ) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        # Per-node counters for first sentence
        concepts_per_node = defaultdict(int)
        properties_per_node = defaultdict(int)

        for relationship in inferred_relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue

            norm_subj = self._normalize_endpoint_text_fn(
                relationship[0], is_subject=True
            )
            norm_obj = self._normalize_endpoint_text_fn(
                relationship[2], is_subject=False
            )
            if norm_subj is None or norm_obj is None:
                continue

            active_nodes_set = set(self.get_nodes_with_active_edges_fn())

            attachment_ok = self.is_attachable_wrapper_fn(
                relationship[0],
                relationship[2],
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self.get_nodes_with_active_edges_fn(),
            )

            if not attachment_ok and not is_first_sentence:
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

            # Validate the triplet
            is_valid, subj, rel_validated, obj = self.validate_triplet(
                subj, relationship[1], obj
            )
            if not is_valid:
                continue

            # Check per-node limits before proceeding
            if subj_type == NodeType.CONCEPT:
                if concepts_per_node[subj] >= MAX_NEW_CONCEPTS:
                    logging.debug(f"Skipping - too many concepts for {subj}")
                    continue
            elif subj_type == NodeType.PROPERTY:
                if properties_per_node[subj] >= MAX_NEW_PROPERTIES:
                    logging.debug(f"Skipping - too many properties for {subj}")
                    continue

            if obj_type == NodeType.CONCEPT:
                if concepts_per_node[obj] >= MAX_NEW_CONCEPTS:
                    logging.debug(f"Skipping - too many concepts for {obj}")
                    continue
            elif obj_type == NodeType.PROPERTY:
                if properties_per_node[obj] >= MAX_NEW_PROPERTIES:
                    logging.debug(f"Skipping - too many properties for {obj}")
                    continue

            if is_first_sentence:
                source_node = None
                dest_node = None
            else:
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

            edge_label = rel_validated.replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)
            if not self._is_valid_relation_label_fn(edge_label):
                continue

            if source_node is None:
                if not self._is_node_allowed_fn(
                    subj,
                    bypass=is_first_sentence
                    or (
                        dest_node is not None
                        and (
                            dest_node in explicit_nodes
                            or dest_node in self.get_nodes_with_active_edges_fn()
                        )
                    ),
                ):
                    continue
                if not self._admit_node_fn(
                    subj,
                    subj_type,
                    "INFERRED_RELATION",
                    sent=sent,
                    is_first_sentence=is_first_sentence,
                ):
                    continue

                source_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                    mark_explicit=False,
                )

            if dest_node is None:
                bypass = is_first_sentence or (
                    source_node is not None
                    and (
                        source_node in explicit_nodes
                        or source_node in self.get_nodes_with_active_edges_fn()
                    )
                )

                if not self._is_node_allowed_fn(
                    obj,
                    bypass=bypass,
                ):
                    continue

                if not self._admit_node_fn(
                    obj,
                    obj_type,
                    "INFERRED_RELATION",
                    sent=sent,
                    is_first_sentence=is_first_sentence,
                ):
                    continue

                dest_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                    mark_explicit=False,
                )

            if source_node is None or dest_node is None:
                continue

            # Increment counters after successful node creation
            if subj_type == NodeType.CONCEPT:
                concepts_per_node[subj] += 1
            else:
                properties_per_node[subj] += 1

            if obj_type == NodeType.CONCEPT:
                concepts_per_node[obj] += 1
            else:
                properties_per_node[obj] += 1

            self.add_edge_wrapper_fn(
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
        is_first_sentence: bool = False,
    ) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        # Per-node counters
        concepts_per_node = defaultdict(int)
        properties_per_node = defaultdict(int)

        for relationship in inferred_relationships:
            if len(relationship) != 3:
                continue
            if not relationship[0] or not relationship[2]:
                continue
            if relationship[0] == relationship[2]:
                continue

            norm_subj = self._normalize_endpoint_text_fn(
                relationship[0], is_subject=True
            )
            norm_obj = self._normalize_endpoint_text_fn(
                relationship[2], is_subject=False
            )

            if norm_subj is None or norm_obj is None:
                continue

            if not self.is_attachable_wrapper_fn(
                relationship[0],
                relationship[2],
                curr_sentences_words,
                curr_sentences_nodes,
                active_graph_nodes,
                self.get_nodes_with_active_edges_fn(),
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

            # Validate the triplet
            is_valid, subj, rel_validated, obj = self.validate_triplet(
                subj, relationship[1], obj
            )
            if not is_valid:
                continue

            # Check per-node limits before proceeding
            if subj_type == NodeType.CONCEPT:
                if concepts_per_node[subj] >= MAX_NEW_CONCEPTS:
                    logging.debug(f"Skipping - too many concepts for {subj}")
                    continue
            elif subj_type == NodeType.PROPERTY:
                if properties_per_node[subj] >= MAX_NEW_PROPERTIES:
                    logging.debug(f"Skipping - too many properties for {subj}")
                    continue

            if obj_type == NodeType.CONCEPT:
                if concepts_per_node[obj] >= MAX_NEW_CONCEPTS:
                    logging.debug(f"Skipping - too many concepts for {obj}")
                    continue
            elif obj_type == NodeType.PROPERTY:
                if properties_per_node[obj] >= MAX_NEW_PROPERTIES:
                    logging.debug(f"Skipping - too many properties for {obj}")
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

            edge_label = rel_validated.replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)

            if not self._is_valid_relation_label_fn(edge_label):
                continue

            if source_node is None:
                if not self._is_node_allowed_fn(
                    subj,
                    bypass=(dest_node is not None and dest_node in explicit_nodes),
                ):
                    continue

                source_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                    mark_explicit=False,
                )

            if dest_node is None:
                if not self._is_node_allowed_fn(obj, bypass=(source_node is not None)):
                    continue

                dest_node = self.graph.add_or_get_node(
                    self._get_concept_lemmas_fn(obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                    mark_explicit=False,
                )

            if source_node is None or dest_node is None:
                continue

            # Increment counters after successful node creation
            if subj_type == NodeType.CONCEPT:
                concepts_per_node[subj] += 1
            else:
                properties_per_node[subj] += 1

            if obj_type == NodeType.CONCEPT:
                concepts_per_node[obj] += 1
            else:
                properties_per_node[obj] += 1

            potential_edge = self.add_edge_wrapper_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

            if potential_edge:
                added_edges.append(potential_edge)
