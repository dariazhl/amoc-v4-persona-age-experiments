import logging
from typing import TYPE_CHECKING, List, Optional, Set

from amoc.graph.node import NodeType, NodeSource

if TYPE_CHECKING:
    from amoc.pipeline.core import AMoCv4


class InitOps:
    def __init__(
        self,
        graph_ref,
        client_ref,
        spacy_nlp,
        edge_visibility: int,
        debug: bool = False,
    ):
        self.graph = graph_ref
        self.client = client_ref
        self.spacy_nlp = spacy_nlp
        self.edge_visibility = edge_visibility
        self.debug = debug

        self._normalize_endpoint_text_fn = None
        self._normalize_edge_label_fn = None
        self._is_valid_relation_label_fn = None
        self._passes_attachment_constraint_fn = None
        self._canonicalize_edge_direction_fn = None
        self._classify_relation_fn = None
        self._add_edge_fn = None
        self._get_nodes_with_active_edges_fn = None
        self._get_node_from_text_fn = None
        self._get_sentences_text_based_nodes_fn = None
        self._extract_deterministic_structure_fn = None
        self._infer_new_relationships_step_0_fn = None
        self._add_inferred_relationships_to_graph_step_0_fn = None
        self._restrict_active_to_current_explicit_fn = None

        self._explicit_nodes_ref = None
        self.persona = None
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
        get_node_from_text_fn,
        get_sentences_text_based_nodes_fn,
        extract_deterministic_structure_fn,
        infer_new_relationships_step_0_fn=None,
        add_inferred_relationships_to_graph_step_0_fn=None,
        restrict_active_to_current_explicit_fn=None,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self._passes_attachment_constraint_fn = passes_attachment_constraint_fn
        self._canonicalize_edge_direction_fn = canonicalize_edge_direction_fn
        self._classify_relation_fn = classify_relation_fn
        self._add_edge_fn = add_edge_fn
        self._get_nodes_with_active_edges_fn = get_nodes_with_active_edges_fn
        self._get_node_from_text_fn = get_node_from_text_fn
        self._get_sentences_text_based_nodes_fn = get_sentences_text_based_nodes_fn
        self._extract_deterministic_structure_fn = extract_deterministic_structure_fn
        self._infer_new_relationships_step_0_fn = infer_new_relationships_step_0_fn
        self._add_inferred_relationships_to_graph_step_0_fn = (
            add_inferred_relationships_to_graph_step_0_fn
        )
        self._restrict_active_to_current_explicit_fn = (
            restrict_active_to_current_explicit_fn
        )

    def set_state_refs(self, explicit_nodes_ref, persona: str):
        self._explicit_nodes_ref = explicit_nodes_ref
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
            get_node_from_text_fn=core._resolve_node_from_text,
            get_sentences_text_based_nodes_fn=core._get_sentences_nodes,
            extract_deterministic_structure_fn=lambda s, n, w: (
                core._linguistic_ops.set_sentence_context(
                    core._current_sentence_index, core.edge_visibility
                ),
                core._linguistic_ops.extract_deterministic_structure(s, n, w),
            )[-1],
            infer_new_relationships_step_0_fn=core._infer_new_relationships_bootstrap,
            add_inferred_relationships_to_graph_step_0_fn=core.add_inferred_relationships_to_graph_step_0,
            restrict_active_to_current_explicit_fn=lambda en: None,
        )
        self.set_state_refs(
            explicit_nodes_ref=core._get_explicit_nodes,
            persona=core.persona,
        )

    def init_graph(self, sent) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._get_sentences_text_based_nodes_fn(
                [sent], create_unexistent_nodes=True
            )
        )

        self._extract_deterministic_structure_fn(
            sent,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        relationships = self.client.get_new_relationships_first_sentence(
            nodes_from_text, sent.text, self.persona
        )

        for relationship in relationships:
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
                norm_subj,
                norm_obj,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges_fn(),
            ):
                continue

            source_node = self._get_node_from_text_fn(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self._get_node_from_text_fn(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )

            edge_label = relationship[1].replace("(edge)", "").strip()
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

            self._add_edge_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

        current_all_text = sent.text
        for node in explicit_nodes:
            degree = sum(
                1
                for e in self.graph.edges
                if e.source_node == node or e.dest_node == node
            )

            if degree == 0:
                nodes_from_text = ""
                for i, n in enumerate(current_sentence_text_based_nodes):
                    nodes_from_text += (
                        f" - ({current_sentence_text_based_words[i]}, {n.node_type})\n"
                    )

                extra_relationships = self.client.get_new_relationships_first_sentence(
                    nodes_from_text,
                    current_all_text,
                    self.persona,
                )

                self._process_extra_relationships(
                    extra_relationships,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                )

    def _process_extra_relationships(
        self,
        extra_relationships: List,
        current_sentence_text_based_nodes: List,
        current_sentence_text_based_words: List[str],
    ) -> None:
        for relationship in extra_relationships:
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
                norm_subj,
                norm_obj,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges_fn(),
            ):
                continue

            source_node = self._get_node_from_text_fn(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self._get_node_from_text_fn(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )

            if source_node and dest_node:
                edge_label = relationship[1].replace("(edge)", "").strip()
                edge_label = self._normalize_edge_label_fn(edge_label)

                if self._is_valid_relation_label_fn(edge_label):
                    self._add_edge_fn(
                        source_node,
                        dest_node,
                        edge_label,
                        self.edge_visibility,
                    )

    def handle_first_sentence(
        self,
        sent,
        resolved_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
    ) -> tuple:
        for e in self.graph.edges:
            e.active_this_sentence = False

        prev_sentences.append(resolved_text)
        self.init_graph(sent)

        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._get_sentences_text_based_nodes_fn([sent], True)
        )

        self._extract_deterministic_structure_fn(
            sent,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
        )

        sentence_lemma_set = {token.lemma_.lower() for token in sent}

        explicit_nodes_current_sentence = {
            n
            for n in current_sentence_text_based_nodes
            if n.node_type in {NodeType.CONCEPT, NodeType.PROPERTY}
            and any(lemma in sentence_lemma_set for lemma in n.lemmas)
        }

        for node in explicit_nodes_current_sentence:
            node.activation_score = self.ACTIVATION_MAX_DISTANCE

        for node in explicit_nodes_current_sentence:
            if node not in self.graph.nodes:
                self.graph.nodes.add(node)

        explicit_nodes_current_sentence = set(explicit_nodes_current_sentence)

        anchor_nodes = {
            n
            for n in explicit_nodes_current_sentence
            if n.node_type == NodeType.CONCEPT
        } | {
            n
            for n in self.graph.nodes
            if n.node_type == NodeType.CONCEPT and any(e.active for e in n.edges)
        }

        inferred_concept_relationships, inferred_property_relationships = (
            self._infer_new_relationships_step_0_fn(sent)
        )

        self._add_inferred_relationships_to_graph_step_0_fn(
            inferred_concept_relationships, NodeType.CONCEPT, sent
        )
        self._add_inferred_relationships_to_graph_step_0_fn(
            inferred_property_relationships, NodeType.PROPERTY, sent
        )

        if not any(edge.active for edge in self.graph.edges):
            for edge in self.graph.edges:
                edge.mark_as_asserted(reset_score=True)

        self._restrict_active_to_current_explicit_fn(
            list(explicit_nodes_current_sentence)
        )

        return (
            nodes_before_sentence,
            False,
            explicit_nodes_current_sentence,
            anchor_nodes,
        )
