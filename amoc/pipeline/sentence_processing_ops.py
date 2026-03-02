import logging
import copy
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Set
import networkx as nx

from amoc.graph.node import NodeType, NodeSource, NodeProvenance

if TYPE_CHECKING:
    from amoc.pipeline.core import AMoCv4


class SentenceProcessingOps:
    def __init__(
        self,
        graph_ref,
        client_ref,
        spacy_nlp,
        max_distance_from_active_nodes: int,
        edge_visibility: int,
        context_length: int,
        debug: bool = False,
    ):
        self.graph = graph_ref
        self.client = client_ref
        self.spacy_nlp = spacy_nlp
        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.edge_visibility = edge_visibility
        self.context_length = context_length
        self.debug = debug

        self._normalize_endpoint_text_fn = None
        self._normalize_edge_label_fn = None
        self._is_valid_relation_label_fn = None
        self._passes_attachment_constraint_fn = None
        self._canonicalize_edge_direction_fn = None
        self._classify_relation_fn = None
        self._add_edge_fn = None
        self._get_nodes_with_active_edges_fn = None
        self._append_adjectival_hints_fn = None
        self._extract_deterministic_structure_fn = None
        self._infer_edges_to_recently_deactivated_fn = None
        self._propagate_activation_from_edges_fn = None
        self._restrict_active_to_current_explicit_fn = None
        self._get_node_from_new_relationship_fn = None
        self._get_phrase_level_concepts_fn = None
        self._get_sentences_text_based_nodes_fn = None
        self._infer_new_relationships_fn = None
        self._add_inferred_relationships_to_graph_fn = None
        self._reactivate_relevant_edges_fn = None

        self._explicit_nodes_ref = None
        self._anchor_nodes_ref = None
        self._triplet_intro_ref = None
        self._carryover_nodes_ref = None
        self.persona = None
        self.ENFORCE_ATTACHMENT_CONSTRAINT = True
        self.ACTIVATION_MAX_DISTANCE = 2

    def set_callbacks(
        self,
        normalize_endpoint_text_fn,
        normalize_edge_label_fn,
        is_valid_relation_label_fn,
        passes_attachment_constraint_fn,
        canonicalize_edge_direction_fn,
        classify_relation_fn,
        add_edge_fn,
        get_nodes_with_active_edges_fn,
        append_adjectival_hints_fn,
        extract_deterministic_structure_fn,
        infer_edges_to_recently_deactivated_fn,
        propagate_activation_from_edges_fn,
        restrict_active_to_current_explicit_fn,
        get_node_from_new_relationship_fn,
        get_phrase_level_concepts_fn,
        get_sentences_text_based_nodes_fn,
        infer_new_relationships_fn,
        add_inferred_relationships_to_graph_fn,
        reactivate_relevant_edges_fn,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self._passes_attachment_constraint_fn = passes_attachment_constraint_fn
        self._canonicalize_edge_direction_fn = canonicalize_edge_direction_fn
        self._classify_relation_fn = classify_relation_fn
        self._add_edge_fn = add_edge_fn
        self._get_nodes_with_active_edges_fn = get_nodes_with_active_edges_fn
        self._append_adjectival_hints_fn = append_adjectival_hints_fn
        self._extract_deterministic_structure_fn = extract_deterministic_structure_fn
        self._infer_edges_to_recently_deactivated_fn = (
            infer_edges_to_recently_deactivated_fn
        )
        self._propagate_activation_from_edges_fn = propagate_activation_from_edges_fn
        self._restrict_active_to_current_explicit_fn = (
            restrict_active_to_current_explicit_fn
        )
        self._get_node_from_new_relationship_fn = get_node_from_new_relationship_fn
        self._get_phrase_level_concepts_fn = get_phrase_level_concepts_fn
        self._get_sentences_text_based_nodes_fn = get_sentences_text_based_nodes_fn
        self._infer_new_relationships_fn = infer_new_relationships_fn
        self._add_inferred_relationships_to_graph_fn = (
            add_inferred_relationships_to_graph_fn
        )
        self._reactivate_relevant_edges_fn = reactivate_relevant_edges_fn

    def set_state_refs(
        self,
        explicit_nodes_ref,
        anchor_nodes_ref,
        triplet_intro_ref,
        carryover_nodes_ref,
        persona: str,
    ):
        self._explicit_nodes_ref = explicit_nodes_ref
        self._anchor_nodes_ref = anchor_nodes_ref
        self._triplet_intro_ref = triplet_intro_ref
        self._carryover_nodes_ref = carryover_nodes_ref
        self.persona = persona

    def configure_with_core(self, core: "AMoCv4") -> None:
        self.set_callbacks(
            normalize_endpoint_text_fn=core._normalize_endpoint_text,
            normalize_edge_label_fn=core._normalize_edge_label,
            is_valid_relation_label_fn=core._is_valid_relation_label,
            passes_attachment_constraint_fn=core._passes_attachment_constraint,
            canonicalize_edge_direction_fn=core._canonicalize_edge_direction,
            classify_relation_fn=core._classify_relation,
            add_edge_fn=core._add_edge,
            get_nodes_with_active_edges_fn=core._get_active_edge_nodes,
            append_adjectival_hints_fn=lambda n, s: (
                core._linguistic_ops.append_adjectival_hints(n, s)
            ),
            extract_deterministic_structure_fn=lambda s, n, w: (
                core._linguistic_ops.set_sentence_context(
                    core._current_sentence_index, core.edge_visibility
                ),
                core._linguistic_ops.extract_deterministic_structure(s, n, w),
            )[-1],
            infer_edges_to_recently_deactivated_fn=core._infer_edges_to_recently_deactivated,
            propagate_activation_from_edges_fn=lambda: (
                core._activation_ops.propagate_activation_from_edges()
            ),
            restrict_active_to_current_explicit_fn=lambda en: (
                core._activation_ops.restrict_active_to_current_explicit(en)
            ),
            get_node_from_new_relationship_fn=core.get_node_from_new_relationship,
            get_phrase_level_concepts_fn=core.get_phrase_level_concepts,
            get_sentences_text_based_nodes_fn=core._get_sentences_nodes,
            infer_new_relationships_fn=core.infer_new_relationships,
            add_inferred_relationships_to_graph_fn=core.add_inferred_relationships_to_graph,
            reactivate_relevant_edges_fn=core.reactivate_relevant_edges,
        )
        self.set_state_refs(
            explicit_nodes_ref=core._get_explicit_nodes,
            anchor_nodes_ref=core._anchor_nodes,
            triplet_intro_ref=core._triplet_intro,
            carryover_nodes_ref=core._get_carryover_nodes,
            persona=core.persona,
        )

    def handle_nonfirst_sentence(
        self,
        i: int,
        sent,
        resolved_text: str,
        original_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
        current_sentence_index: int,
        current_sentence_text: str,
        explicit_nodes_current_sentence: set,
        anchor_nodes: set,
        triplet_intro: dict,
    ) -> Tuple[set, bool]:
        self._current_sentence_index = current_sentence_index
        self._current_sentence_text = current_sentence_text

        for e in self.graph.edges:
            e.active_this_sentence = False
        added_edges = []

        _graph_snapshot = copy.deepcopy(self.graph)
        _anchor_snapshot = copy.deepcopy(anchor_nodes)
        _triplet_intro_snapshot = copy.deepcopy(triplet_intro)

        current_sentence = sent
        prev_sentences.append(resolved_text)
        if len(prev_sentences) > self.context_length:
            prev_sentences.pop(0)

        phrase_nodes = self._get_phrase_level_concepts_fn(current_sentence)

        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._get_sentences_text_based_nodes_fn(
                [current_sentence], create_unexistent_nodes=True
            )
        )

        sentence_lemma_set = {
            token.lemma_.lower() for token in self.spacy_nlp(original_text)
        }

        new_explicit_nodes = {
            n
            for n in current_sentence_text_based_nodes
            if n.node_type in {NodeType.CONCEPT, NodeType.PROPERTY}
            and any(lemma in sentence_lemma_set for lemma in n.lemmas)
        }
        explicit_nodes_current_sentence.clear()
        explicit_nodes_current_sentence.update(new_explicit_nodes)

        self._extract_deterministic_structure_fn(
            current_sentence,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
        )

        for node in explicit_nodes_current_sentence:
            node.activation_score = self.ACTIVATION_MAX_DISTANCE
            node.active = True

        current_all_text = resolved_text

        graph_active_nodes = self.graph.get_active_nodes(
            self.max_distance_from_active_nodes, only_text_based=True
        )

        active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
        active_nodes_edges_text, _ = self.graph.get_edges_str(
            graph_active_nodes, only_text_based=True
        )

        nodes_from_text = ""
        for idx, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"
            )

        nodes_from_text = self._append_adjectival_hints_fn(nodes_from_text, sent)

        new_relationships = self.client.get_new_relationships(
            nodes_from_text,
            active_nodes_text,
            active_nodes_edges_text,
            current_all_text,
            self.persona,
        )

        if explicit_nodes_current_sentence and (
            not new_relationships or len(new_relationships) == 0
        ):
            explicit = list(explicit_nodes_current_sentence)
            if len(explicit) >= 2:
                new_relationships = [
                    (
                        explicit[0].get_text_representer(),
                        "co_occurs_with",
                        explicit[1].get_text_representer(),
                    )
                ]

        text_based_activated_nodes = current_sentence_text_based_nodes
        sentence_lemma_keys = {
            tuple(n.lemmas) for n in current_sentence_text_based_nodes
        }

        added_edges = self._process_llm_relationships(
            new_relationships,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            graph_active_nodes,
            sentence_lemma_keys,
            explicit_nodes_current_sentence,
        )

        inferred_concept_relationships, inferred_property_relationships = (
            self._infer_new_relationships_fn(
                current_all_text,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                self.graph.get_nodes_str(
                    self.graph.get_active_nodes(
                        self.max_distance_from_active_nodes,
                        only_text_based=True,
                    )
                ),
                self.graph.get_edges_str(
                    self.graph.get_active_nodes(
                        self.max_distance_from_active_nodes,
                        only_text_based=True,
                    ),
                    only_text_based=True,
                )[0],
            )
        )

        graph_active_nodes = self.graph.get_active_nodes(
            self.max_distance_from_active_nodes,
            only_text_based=True,
        )

        self._add_inferred_relationships_to_graph_fn(
            inferred_concept_relationships,
            NodeType.CONCEPT,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            graph_active_nodes,
            added_edges,
        )
        self._add_inferred_relationships_to_graph_fn(
            inferred_property_relationships,
            NodeType.PROPERTY,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            graph_active_nodes,
            added_edges,
        )

        if self.ENFORCE_ATTACHMENT_CONSTRAINT:
            targeted_edges = self._infer_edges_to_recently_deactivated_fn(
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                current_all_text,
            )
            added_edges.extend(targeted_edges)

        reactivated_edges = self.graph.reactivate_memory_edges_within_distance(
            explicit_nodes=explicit_nodes_current_sentence,
            max_distance=self.max_distance_from_active_nodes,
            current_sentence=current_sentence_index,
        )
        self._reactivate_relevant_edges_fn(
            self.graph.get_active_nodes(
                self.max_distance_from_active_nodes, only_text_based=True
            ),
            " ".join(prev_sentences),
            added_edges,
        )
        self._propagate_activation_from_edges_fn()

        anchor_nodes.clear()
        anchor_nodes.update(
            anchor_nodes
            | {
                n
                for n in explicit_nodes_current_sentence
                if n.node_type == NodeType.CONCEPT
            }
            | {
                n
                for n in self._get_nodes_with_active_edges_fn()
                if n.node_type == NodeType.CONCEPT
            }
        )

        self._handle_single_explicit_bridge(
            explicit_nodes_current_sentence,
            anchor_nodes,
        )

        self._ensure_explicit_nodes_have_edges(
            explicit_nodes_current_sentence,
            current_sentence_text,
        )

        self._restrict_active_to_current_explicit_fn(
            list(explicit_nodes_current_sentence)
        )

        should_skip = self._handle_empty_projection_retry(
            explicit_nodes_current_sentence,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            graph_active_nodes,
            current_all_text,
            nodes_from_text,
            _graph_snapshot,
            _anchor_snapshot,
            _triplet_intro_snapshot,
            anchor_nodes,
            triplet_intro,
            nodes_before_sentence,
        )

        return (nodes_before_sentence, should_skip)

    def _process_llm_relationships(
        self,
        new_relationships,
        current_sentence_text_based_nodes,
        current_sentence_text_based_words,
        graph_active_nodes,
        sentence_lemma_keys,
        explicit_nodes_current_sentence,
    ) -> list:
        added_edges = []

        for idx, relationship in enumerate(new_relationships):
            if relationship is None or isinstance(relationship, (int, float, bool)):
                continue

            if isinstance(relationship, dict):
                subj = relationship.get("subject") or relationship.get("head")
                rel = relationship.get("relation") or relationship.get("predicate")
                obj = relationship.get("object") or relationship.get("tail")
                if not (subj and rel and obj):
                    continue
                relationship = (str(subj), str(rel), str(obj))

            if not isinstance(relationship, (list, tuple)):
                continue

            if isinstance(relationship, (list, tuple)) and len(relationship) == 4:
                subj, rel, _, obj = relationship
                relationship = (subj, rel, obj)

            if len(relationship) != 3:
                continue

            subj, rel, obj = relationship

            subj = self._normalize_endpoint_text_fn(subj, is_subject=True) or None
            obj = self._normalize_endpoint_text_fn(obj, is_subject=False) or None
            if subj is None or obj is None:
                continue
            if not subj or not obj:
                continue
            if subj == obj:
                continue
            if not isinstance(subj, str) or not isinstance(obj, str):
                continue

            if not self._passes_attachment_constraint_fn(
                subj,
                obj,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                graph_active_nodes,
                self._get_nodes_with_active_edges_fn(),
            ):
                continue

            source_node = self._get_node_from_new_relationship_fn(
                subj,
                graph_active_nodes,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=False,
            )

            dest_node = self._get_node_from_new_relationship_fn(
                obj,
                graph_active_nodes,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=False,
            )

            edge_label = rel.replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)
            if not self._is_valid_relation_label_fn(edge_label):
                continue
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

            if tuple(source_node.lemmas) in sentence_lemma_keys:
                source_node.node_source = NodeSource.TEXT_BASED
            if tuple(dest_node.lemmas) in sentence_lemma_keys:
                dest_node.node_source = NodeSource.TEXT_BASED

            if source_node is None:
                source_node = self._get_node_from_new_relationship_fn(
                    subj,
                    graph_active_nodes,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    node_source=NodeSource.TEXT_BASED,
                    create_node=True,
                )

            if dest_node is None:
                dest_node = self._get_node_from_new_relationship_fn(
                    obj,
                    graph_active_nodes,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    node_source=NodeSource.TEXT_BASED,
                    create_node=True,
                )

            edge = self._add_edge_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )
            if edge:
                added_edges.append(edge)

        return added_edges

    def _handle_single_explicit_bridge(
        self,
        explicit_nodes_current_sentence: set,
        anchor_nodes: set,
    ):
        if len(explicit_nodes_current_sentence) == 1:
            node = next(iter(explicit_nodes_current_sentence))

            if node not in self._get_nodes_with_active_edges_fn():
                anchor = next(iter(anchor_nodes), None)

                if anchor and anchor != node:
                    edge = self._add_edge_fn(
                        anchor,
                        node,
                        "appears",
                        self.edge_visibility,
                    )
                    if edge:
                        edge.mark_as_asserted(reset_score=True)

    def _ensure_active_connectivity(
        self,
        explicit_nodes_current_sentence: set,
        anchor_nodes: set,
    ):
        carryover = (
            self._carryover_nodes_ref()
            if callable(self._carryover_nodes_ref)
            else set()
        )
        required_nodes = explicit_nodes_current_sentence | carryover

        # Delegate to canonical authority for deterministic repair
        connected = self.graph.enforce_connectivity(
            required_nodes,
            allow_reactivation=True,
        )


    def _ensure_explicit_nodes_have_edges(
        self,
        explicit_nodes_current_sentence: set,
        current_sentence_text: str,
    ):
        active_nodes = self._get_nodes_with_active_edges_fn()

        for node in explicit_nodes_current_sentence:
            if node not in active_nodes:
                repair_relationships = self.client.get_new_relationships(
                    node.get_text_representer(),
                    self.graph.get_nodes_str(self.graph.nodes),
                    self.graph.get_edges_str(self.graph.nodes)[0],
                    current_sentence_text,
                    self.persona,
                )

                for relationship in repair_relationships:
                    if (
                        isinstance(relationship, (list, tuple))
                        and len(relationship) == 3
                    ):
                        subj, rel, obj = relationship

                        source_node = self._get_node_from_new_relationship_fn(
                            subj,
                            self.graph.nodes,
                            [],
                            [],
                            node_source=NodeSource.TEXT_BASED,
                            create_node=False,
                        )

                        dest_node = self._get_node_from_new_relationship_fn(
                            obj,
                            self.graph.nodes,
                            [],
                            [],
                            node_source=NodeSource.TEXT_BASED,
                            create_node=False,
                        )

                        if source_node and dest_node:
                            edge = self._add_edge_fn(
                                source_node,
                                dest_node,
                                rel,
                                self.edge_visibility,
                            )

                            if edge:
                                edge.mark_as_asserted(reset_score=True)
                                break

    def _handle_empty_projection_retry(
        self,
        explicit_nodes_current_sentence: set,
        current_sentence_text_based_nodes,
        current_sentence_text_based_words,
        graph_active_nodes,
        current_all_text: str,
        nodes_from_text: str,
        _graph_snapshot,
        _anchor_snapshot,
        _triplet_intro_snapshot,
        anchor_nodes: set,
        triplet_intro: dict,
        nodes_before_sentence: set,
    ) -> bool:
        current_active_nodes = self._get_nodes_with_active_edges_fn()

        if not current_active_nodes:
            logging.warning(
                "Empty projection at sentence %d. Retrying once.",
                self._current_sentence_index,
            )

            # Recompute active context
            graph_active_nodes = self.graph.get_active_nodes(
                self.max_distance_from_active_nodes,
                only_text_based=True,
            )

            active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
            active_nodes_edges_text, _ = self.graph.get_edges_str(
                graph_active_nodes,
                only_text_based=True,
            )

            retry_relationships = self.client.get_new_relationships(
                nodes_from_text,
                active_nodes_text,
                active_nodes_edges_text,
                current_all_text,
                self.persona,
            )

            for relationship in retry_relationships:
                if isinstance(relationship, (list, tuple)) and len(relationship) == 3:
                    subj, rel, obj = relationship
                    subj = self._normalize_endpoint_text_fn(subj, is_subject=True)
                    obj = self._normalize_endpoint_text_fn(obj, is_subject=False)

                    if not subj or not obj or subj == obj:
                        continue

                    source_node = self._get_node_from_new_relationship_fn(
                        subj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                    dest_node = self._get_node_from_new_relationship_fn(
                        obj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                    if source_node and dest_node:
                        edge = self._add_edge_fn(
                            source_node,
                            dest_node,
                            rel,
                            self.edge_visibility,
                        )
                        if edge:
                            break

            self._restrict_active_to_current_explicit_fn(
                list(explicit_nodes_current_sentence)
            )

            if not self._get_nodes_with_active_edges_fn():
                logging.error("Retry failed — reverting.")
                return True

        return False

    def sanitize_json_contamination(
        self, resolved_text: str, original_text: str, sent
    ) -> tuple:
        # sometimes, the parser returns sentences such as: "Processing sentence 0: answer is: A man very close to Charlemagne wrote most of the things we know about Charlemagne."
        if resolved_text.strip().startswith("{"):
            logging.error("JSON contamination in LLM output — reverting.")
            resolved_text = original_text

        # Strip bad prefix
        resolved_text = re.sub(
            r"^\s*(processing sentence \d+:)?\s*answer\s+is:\s*",
            "",
            resolved_text,
            flags=re.IGNORECASE,
        ).strip()

        # rebuild spaCy doc from cleaned text
        resolved_doc = self.spacy_nlp(resolved_text)
        sents = list(resolved_doc.sents)
        if not sents:
            # Fallback to original if parsing fails
            fallback_doc = self.spacy_nlp(original_text.strip())
            fallback_sents = list(fallback_doc.sents)
            return original_text.strip(), fallback_sents[0] if fallback_sents else sent

        return resolved_text, sents[0]

    def apply_post_sentence_processing(
        self,
        explicit_nodes: set,
        carryover_nodes: set,
        apply_global_edge_decay_fn: callable,
        decay_node_activation_fn: callable,
    ) -> None:
        apply_global_edge_decay_fn()
        self.graph.enforce_cumulative_stability(set(explicit_nodes))
        decay_node_activation_fn()
