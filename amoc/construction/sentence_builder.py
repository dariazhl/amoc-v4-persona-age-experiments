import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Set
import networkx as nx

from amoc.core.node import NodeType, NodeSource, NodeProvenance
from amoc.admission.triplet_validator import TripletValidator

if TYPE_CHECKING:
    from amoc.pipeline.orchestrator import AMoCv4


class SentenceGraphBuilder:
    def __init__(
        self,
        graph_ref,
        llm_extractor,
        spacy_nlp,
        max_distance_from_active_nodes: int,
        edge_visibility: int,
        context_length: int,
        text_normalizer,
        debug: bool = False,
    ):
        self.graph = graph_ref
        self.llm = llm_extractor
        self.spacy_nlp = spacy_nlp
        self.max_distance_from_active_nodes = max_distance_from_active_nodes
        self.edge_visibility = edge_visibility
        self.context_length = context_length
        self.debug = debug

        self._normalize_endpoint_text_fn = None
        self._normalize_edge_label_fn = None
        self._is_valid_relation_label_fn = None
        self.is_attachable_wrapper_fn = None
        self.add_edge_wrapper_fn = None
        self.get_nodes_with_active_edges_fn = None
        self._append_adjectival_hints_fn = None
        self._extract_deterministic_structure_fn = None
        self.llm_attach_explicit_to_carryover_wrapper_fn = None
        self._propagate_activation_from_edges_fn = None
        self._restrict_active_nodes_fn = None
        self._get_node_from_new_relationship_fn = None
        self._get_node_from_text_fn = None
        self._extract_main_nouns_fn = None
        self._get_sentences_text_based_nodes_fn = None
        self._infer_new_relationships_fn = None
        self._add_inferred_relationships_to_graph_fn = None
        self._reactivate_relevant_edges_fn = None
        # step‑0 callbacks
        self._infer_new_relationships_step_0_fn = None
        self._add_inferred_relationships_to_graph_step_0_fn = None
        self._explicit_nodes_ref = None
        self._anchor_nodes_ref = lambda: set()
        self._triplet_intro_ref = None
        self._carryover_nodes_ref = None
        self.persona = None
        self.ACTIVATION_MAX_DISTANCE = 2
        self.text_normalizer = text_normalizer
        self.triple_validator = None

        self._current_sentence_span = None
        self._current_sentence_text = ""
        self._current_sentence_index = None

    def set_builder_callbacks(
        self,
        normalize_endpoint_text_fn,
        normalize_edge_label_fn,
        is_valid_relation_label_fn,
        is_attachable_fn,
        add_edge_fn,
        get_nodes_with_active_edges_fn,
        append_adjectival_hints_fn,
        extract_deterministic_structure_fn,
        llm_attach_explicit_to_carryover_fn,
        propagate_activation_from_edges_fn,
        restrict_active_nodes_fn,
        get_node_from_new_relationship_fn,
        get_node_from_text_fn,
        extract_main_nouns_fn,
        get_sentences_text_based_nodes_fn,
        infer_new_relationships_fn,
        add_inferred_relationships_to_graph_fn,
        reactivate_relevant_edges_fn,
        # New step‑0 callbacks
        infer_new_relationships_step_0_fn,
        add_inferred_relationships_to_graph_step_0_fn,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self.is_attachable_wrapper_fn = is_attachable_fn
        self.add_edge_wrapper_fn = add_edge_fn
        self.get_nodes_with_active_edges_fn = get_nodes_with_active_edges_fn
        self._append_adjectival_hints_fn = append_adjectival_hints_fn
        self._extract_deterministic_structure_fn = extract_deterministic_structure_fn
        self.llm_attach_explicit_to_carryover_wrapper_fn = (
            llm_attach_explicit_to_carryover_fn
        )
        self._propagate_activation_from_edges_fn = propagate_activation_from_edges_fn
        self._restrict_active_nodes_fn = restrict_active_nodes_fn
        self._get_node_from_new_relationship_fn = get_node_from_new_relationship_fn
        self._get_node_from_text_fn = get_node_from_text_fn
        self._extract_main_nouns_fn = extract_main_nouns_fn
        self._get_sentences_text_based_nodes_fn = get_sentences_text_based_nodes_fn
        self._infer_new_relationships_fn = infer_new_relationships_fn
        self._add_inferred_relationships_to_graph_fn = (
            add_inferred_relationships_to_graph_fn
        )
        self._reactivate_relevant_edges_fn = reactivate_relevant_edges_fn
        self._infer_new_relationships_step_0_fn = infer_new_relationships_step_0_fn
        self._add_inferred_relationships_to_graph_step_0_fn = (
            add_inferred_relationships_to_graph_step_0_fn
        )

    def set_builder_state_refs(
        self,
        explicit_nodes_ref,
        anchor_nodes_ref,
        triplet_intro_ref,
        carryover_nodes_ref,
        persona: str,
    ):
        self._explicit_nodes_ref = explicit_nodes_ref
        self._anchor_nodes_ref = lambda: set()
        self._triplet_intro_ref = triplet_intro_ref
        self._carryover_nodes_ref = carryover_nodes_ref
        self.persona = persona

    def configure_sentence_builder_with_core(self, core: "AMoCv4") -> None:
        self.text_normalizer = core._text_filter_ops
        self.set_builder_callbacks(
            normalize_endpoint_text_fn=core._normalize_endpoint_text,
            normalize_edge_label_fn=core._normalize_edge_label,
            is_valid_relation_label_fn=core._is_valid_relation_label,
            is_attachable_fn=core.is_attachable_wrapper,
            add_edge_fn=core.add_edge_wrapper,
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
            llm_attach_explicit_to_carryover_fn=core.llm_attach_explicit_to_carryover_wrapper,
            propagate_activation_from_edges_fn=lambda: (
                core._activation_ops.propagate_activation_from_edges()
            ),
            restrict_active_nodes_fn=lambda en: (
                core._activation_ops.restrict_active_nodes(en)
            ),
            get_node_from_new_relationship_fn=core.resolve_node_from_new_relationship_wrapper,
            get_node_from_text_fn=core.resolve_node_from_text_wrapper,
            extract_main_nouns_fn=core.extract_phrase_level_concepts_wrapper,
            get_sentences_text_based_nodes_fn=core._get_sentences_nodes,
            infer_new_relationships_fn=core.infer_new_relationships_for_sentence_wrapper,
            add_inferred_relationships_to_graph_fn=core.add_inferred_relationships_to_graph_wrapper,
            reactivate_relevant_edges_fn=core.reactivate_relevant_edges_wrapper,
            # New step‑0 callbacks
            infer_new_relationships_step_0_fn=core.infer_new_relationships_bootstrap_wrapper,
            add_inferred_relationships_to_graph_step_0_fn=core.add_inferred_relationships_to_graph_step_0_wrapper,
        )
        self.set_builder_state_refs(
            explicit_nodes_ref=core._get_explicit_nodes,
            anchor_nodes_ref=lambda: set(),
            triplet_intro_ref=core._triplet_intro,
            carryover_nodes_ref=core._get_carryover_nodes,
            persona=core.persona,
        )

    def format_nodes_for_prompt(self, nodes, words) -> str:
        result = ""
        for i, node in enumerate(nodes):
            result += f" - ({words[i]}, {node.node_type})\n"
        return result

    def add_edge_from_triplet(
        self,
        rel_tuple,
        current_nodes,
        current_words,
        bypass_attachment: bool = False,
    ) -> bool:
        if len(rel_tuple) != 3:
            return False
        subj, rel, obj = rel_tuple
        if not subj or not obj or subj == obj:
            return False
        if not isinstance(subj, str) or not isinstance(obj, str):
            return False

        norm_subj = self._normalize_endpoint_text_fn(subj, is_subject=True)
        norm_obj = self._normalize_endpoint_text_fn(obj, is_subject=False)
        if norm_subj is None or norm_obj is None:
            return False

        # Only perform attachability check if we are NOT bypassing attachment
        if not bypass_attachment:
            if not self.is_attachable_wrapper_fn(
                norm_subj,
                norm_obj,
                current_words,
                current_nodes,
                list(self.graph.nodes),
                self.get_nodes_with_active_edges_fn(),
            ):
                logging.warning(
                    f"FS_DEBUG: attachability check failed for ({norm_subj}, {norm_obj})"
                )
                return False
        # else: skip attachability check (first sentence)

        source_node = self._get_node_from_text_fn(
            norm_subj,
            current_nodes,
            current_words,
            node_source=NodeSource.TEXT_BASED,
            create_node=True,
        )
        dest_node = self._get_node_from_text_fn(
            norm_obj,
            current_nodes,
            current_words,
            node_source=NodeSource.TEXT_BASED,
            create_node=True,
        )
        if source_node is None or dest_node is None:
            return False

        edge_label = rel.replace("(edge)", "").strip()
        edge_label = self._normalize_edge_label_fn(edge_label)
        if not self._is_valid_relation_label_fn(edge_label):
            logging.warning(f"FS_DEBUG: invalid relation label: {edge_label}")
            return False

        edge = self.add_edge_wrapper_fn(
            source_node,
            dest_node,
            edge_label,
            self.edge_visibility,
            bypass_attachment_constraint=bypass_attachment,
        )
        if edge:
            logging.info(f"edge added: {source_node} --{edge_label}--> {dest_node}")
            return True
        else:
            logging.warning(
                f"FS_DEBUG: edge NOT added: {source_node} --{edge_label}--> {dest_node}"
            )
            return False

    def init_graph(self, sent) -> None:
        explicit_nodes = (
            self._explicit_nodes_ref() if callable(self._explicit_nodes_ref) else set()
        )

        current_nodes, current_words = self._get_sentences_text_based_nodes_fn(
            [sent], create_unexistent_nodes=True
        )

        # self._extract_deterministic_structure_fn(sent, current_nodes, current_words)

        nodes_from_text = self.format_nodes_for_prompt(current_nodes, current_words)

        relationships = self.llm.get_new_relationships_first_sentence(
            nodes_from_text, sent.text, self.persona
        )
        logging.info(f"raw llm relationships for first sentence: {relationships}")

        for rel in relationships:
            self.add_edge_from_triplet(
                rel, current_nodes, current_words, bypass_attachment=True
            )

        for node in explicit_nodes:
            degree = sum(
                1
                for e in self.graph.edges
                if e.source_node == node or e.dest_node == node
            )
            if degree > 0:
                continue

            nodes_from_text = self.format_nodes_for_prompt(current_nodes, current_words)
            extra_relationships = self.llm.get_new_relationships_first_sentence(
                nodes_from_text, sent.text, self.persona
            )
            for rel in extra_relationships:
                self.add_edge_from_triplet(rel, current_nodes, current_words)

    # first sentence must be handled differently because there is no active graph to compare to,
    # so we only extract explicit relationships from the text
    def handle_first_sentence(
        self,
        sent,
        resolved_text: str,
        prev_sentences: list,
        nodes_before_sentence: set,
    ) -> tuple:
        logging.info("starting first sentence processing")
        # add resolved sentence text
        prev_sentences.append(resolved_text)
        # initiliaze graph
        self.init_graph(sent)
        logging.info(
            f"after first sentence, graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges"
        )

        # extrat nodes
        current_nodes, current_words = self._get_sentences_text_based_nodes_fn(
            [sent], True
        )
        sentence_lemma_set = {token.lemma_.lower() for token in sent}

        # Get the validator
        validator = self.get_triplet_validator()
        current_sentence_text = self._current_sentence_text

        # Validate each edge that was just added in init_graph
        for edge in list(
            self.graph.edges
        ):  # Use list to avoid modification during iteration
            if not hasattr(edge, "validated") or not edge.validated:
                subj = edge.source_node.get_text_representer()
                rel = edge.label
                obj = edge.dest_node.get_text_representer()

                # Ask LLM if this triple makes sense
                result = validator.validate_with_llm(
                    subj, rel, obj, current_sentence_text
                )

                if not result.get("valid", False):
                    # Remove invalid edge
                    self.graph.remove_edge(edge)
                    logging.info(
                        f"first sentence: removed invalid edge ({subj}, {rel}, {obj})"
                    )

        # identify explict nodes
        explicit_nodes_current_sentence = {
            n
            for n in current_nodes
            if n.node_type in {NodeType.CONCEPT, NodeType.PROPERTY}
            and any(lemma in sentence_lemma_set for lemma in n.lemmas)
        }
        # add nodes to graph
        for node in explicit_nodes_current_sentence:
            if node not in self.graph.nodes:
                self.graph.nodes.add(node)
        # explicit nodes
        explicit_nodes_current_sentence = set(explicit_nodes_current_sentence)
        # define anchor nodes = explicit nodes for the first sent
        # inferred nodes
        inferred_concept_relationships, inferred_property_relationships = (
            self._infer_new_relationships_step_0_fn(sent)
        )
        self._add_inferred_relationships_to_graph_step_0_fn(
            inferred_concept_relationships,
            NodeType.CONCEPT,
            sent,
            is_first_sentence=True,
        )
        self._add_inferred_relationships_to_graph_step_0_fn(
            inferred_property_relationships,
            NodeType.PROPERTY,
            sent,
            is_first_sentence=True,
        )
        # ensure at least one active edge
        if not any(edge.active for edge in self.graph.edges):
            logging.info("no active edges found, marking all edges as current")
            for edge in self.graph.edges:
                edge.mark_as_current_sentence(reset_score=True)

        self._restrict_active_nodes_fn(list(explicit_nodes_current_sentence))
        return (
            nodes_before_sentence,
            False,
            explicit_nodes_current_sentence,
            set(),
        )

    def is_relation_valid(
        self,
        source_node,
        edge_label,
        dest_node,
    ) -> bool:
        # triplets such as "famous - knows - man" are incorrect
        # method validates that the triplet makes semantic sense
        doc = self.spacy_nlp(self._current_sentence_text)

        source_lemma = source_node.get_text_representer().lower()
        dest_lemma = dest_node.get_text_representer().lower()

        for tok in doc:
            if tok.lemma_.lower() == edge_label.lower():
                # find nominal subject
                subj = [c for c in tok.children if c.dep_ in {"nsubj", "nsubjpass"}]
                # find direct object
                obj = [c for c in tok.children if c.dep_ in {"dobj", "pobj", "attr"}]

                if subj and obj:
                    subj_lemma = subj[0].lemma_.lower()
                    obj_lemma = obj[0].lemma_.lower()

                    if subj_lemma == source_lemma and obj_lemma == dest_lemma:
                        return True
        # do not admit triplet that does not make sense
        return False

    # handle the rest of the sentences
    # extract explicit nodes, inferred nodes, reactivate memory
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
        self._current_sentence_span = sent
        self._story_context = " ".join(prev_sentences)

        for e in self.graph.edges:
            e.active_this_sentence = False
        added_edges = []
        # update sent context
        current_sentence = sent
        prev_sentences.append(resolved_text)
        if len(prev_sentences) > self.context_length:
            prev_sentences.pop(0)
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self._get_sentences_text_based_nodes_fn(
                [current_sentence], create_unexistent_nodes=True
            )
        )

        sentence_lemma_set = {
            token.lemma_.lower() for token in self.spacy_nlp(original_text)
        }
        # explicit nodes
        new_explicit_nodes = {
            n
            for n in current_sentence_text_based_nodes
            if n.node_type in {NodeType.CONCEPT, NodeType.PROPERTY}
            and any(lemma in sentence_lemma_set for lemma in n.lemmas)
        }
        explicit_nodes_current_sentence.clear()
        explicit_nodes_current_sentence.update(new_explicit_nodes)
        # add edges determined by text structure
        self._extract_deterministic_structure_fn(
            current_sentence,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
        )
        current_all_text = resolved_text
        # retrieve active subgraph within distance from explicit nodes
        graph_active_nodes = self.graph.get_active_nodes_wrapper(
            self.max_distance_from_active_nodes, only_text_based=True
        )

        active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
        active_nodes_edges_text, _ = self.graph.get_edges_str(
            graph_active_nodes, only_text_based=True
        )
        # inferred nodes
        nodes_from_text = ""
        for idx, node in enumerate(current_sentence_text_based_nodes):
            # pass it as list of nodes to the LLM to infer relationships
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"
            )

        nodes_from_text = self._append_adjectival_hints_fn(nodes_from_text, sent)
        # get new relationships from LLM
        new_relationships = self.llm.get_new_relationships(
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

        text_based_activated_nodes = current_sentence_text_based_nodes
        sentence_lemma_keys = {
            tuple(n.lemmas) for n in current_sentence_text_based_nodes
        }
        # process LLM relationships and add to graph, keep track of added edges to reactivate them later if needed
        added_edges = self.add_edges_from_llm(
            new_relationships,
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            graph_active_nodes,
            sentence_lemma_keys,
            explicit_nodes_current_sentence,
            prev_sentences,
        )
        # inferr new ceoncepts and properties
        inferred_concept_relationships, inferred_property_relationships = (
            self._infer_new_relationships_fn(
                current_all_text,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                self.graph.get_nodes_str(
                    self.graph.get_active_nodes_wrapper(
                        self.max_distance_from_active_nodes,
                        only_text_based=True,
                    )
                ),
                self.graph.get_edges_str(
                    self.graph.get_active_nodes_wrapper(
                        self.max_distance_from_active_nodes,
                        only_text_based=True,
                    ),
                    only_text_based=True,
                )[0],
            )
        )

        graph_active_nodes = self.graph.get_active_nodes_wrapper(
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
        # create edges between explicit nodes and carryover nodes
        targeted_edges = self.llm_attach_explicit_to_carryover_wrapper_fn(
            current_sentence_text_based_nodes,
            current_sentence_text_based_words,
            current_all_text,
        )
        added_edges.extend(targeted_edges)
        # use LLM to select which memory edges to reactivate based on the sentence context
        self._reactivate_relevant_edges_fn(
            self.graph.get_active_nodes_wrapper(
                self.max_distance_from_active_nodes, only_text_based=True
            ),
            " ".join(prev_sentences),
            added_edges,
        )
        self._propagate_activation_from_edges_fn()
        self._restrict_active_nodes_fn(list(explicit_nodes_current_sentence))

        return (nodes_before_sentence, False)

    def connect_isolated_explicit_node(
        self,
        explicit_nodes_current_sentence: set,
    ) -> None:
        if len(explicit_nodes_current_sentence) == 1:
            node = next(iter(explicit_nodes_current_sentence))

            active_nodes = self.get_nodes_with_active_edges_fn()
            if node not in active_nodes and active_nodes:
                anchor = min(
                    active_nodes,
                    key=lambda n: n.get_text_representer(),
                )

                if anchor != node:
                    edge = self.add_edge_wrapper_fn(
                        anchor,
                        node,
                        "appears",
                        self.edge_visibility,
                    )
                    if edge:
                        edge.mark_as_current_sentence(reset_score=True)

    def repair_isolated_explicit_nodes(
        self,
        explicit_nodes_current_sentence: set,
        current_sentence_text: str,
    ) -> None:
        active_nodes = self.get_nodes_with_active_edges_fn()

        for node in explicit_nodes_current_sentence:
            if node not in active_nodes:
                repair_relationships = self.llm.get_new_relationships(
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
                            edge = self.add_edge_wrapper_fn(
                                source_node,
                                dest_node,
                                rel,
                                self.edge_visibility,
                            )

                            if edge:
                                edge.mark_as_current_sentence(reset_score=True)
                                break

    # Normalize LLM output into (subj, rel, obj)
    def normalize_llm_triple(self, relationship):
        if not relationship:
            return None

        if isinstance(relationship, dict):
            subj = relationship.get("subject") or relationship.get("head")
            rel = relationship.get("relation") or relationship.get("predicate")
            obj = relationship.get("object") or relationship.get("tail")
            if subj and rel and obj:
                return str(subj), str(rel), str(obj)
            return None

        if isinstance(relationship, (list, tuple)):
            if len(relationship) == 3:
                return relationship
            if len(relationship) == 4:
                subj, rel, _, obj = relationship
                return subj, rel, obj

        return None

    # Take the raw list of triples returned by the LLM, process them and add edges to graph
    def add_edges_from_llm(
        self,
        new_relationships,
        current_sentence_text_based_nodes,
        current_sentence_text_based_words,
        graph_active_nodes,
        sentence_lemma_keys,
        explicit_nodes_current_sentence,
        prev_sentences,
    ) -> list:

        added_edges = []

        # get or create triple validator
        validator = self.get_triplet_validator()

        # step 1: normalize raw llm output to (subj, rel, obj) triples
        normalized_triples = validator.normalize_llm_triplets(new_relationships)
        if not normalized_triples:
            return added_edges

        # step 2: verb validation + LLM validation
        validated_triples = []
        for subj, rel, obj in normalized_triples:
            # verb check before calling LLM
            verb_result = validator.validate_triplet_relation((subj, rel, obj))
            if verb_result["action"] in ("reject", "reject_negation"):
                logging.info(
                    f"rejected ({verb_result['action']}): ({subj}, {rel}, {obj}) - {verb_result['reason']}"
                )
                continue
            elif verb_result["action"] == "swap" and verb_result.get(
                "corrected_triple"
            ):
                subj, rel, obj = verb_result["corrected_triple"]
                logging.info(f"swapped verb position: ({subj}, {rel}, {obj})")
            elif verb_result["action"] == "add_copula" and verb_result.get(
                "corrected_triple"
            ):
                subj, rel, obj = verb_result["corrected_triple"]
                logging.info(f"added copula: ({subj}, {rel}, {obj})")

            validation = validator.validate_with_llm(
                subj,
                rel,
                obj,
                self._current_sentence_text,
                story_context=self._story_context,
            )
            if not validation.get("valid", False):
                continue

            corrected = validation.get("corrected_triple")
            if (
                corrected
                and isinstance(corrected, (list, tuple))
                and len(corrected) == 3
            ):
                subj, rel, obj = corrected

            validated_triples.append((subj, rel, obj))

        if not validated_triples:
            return added_edges

        # step 3: deduplicate semantically similar triples
        validated_triples = validator.deduplicate_triplets(validated_triples)

        # step 4: prioritize hub ordering
        explicit_node_texts = [
            n.get_text_representer() for n in current_sentence_text_based_nodes
        ]
        prioritized_triples = validator.prioritize_hub(
            validated_triples, explicit_node_texts
        )

        # step 5: process each triple
        for subj, rel, obj in prioritized_triples:
            # Use raw strings directly – node lookup will handle them
            subj_final, obj_final = subj, obj
            rel_final = rel

            # check attachability
            if not self.check_attachable(
                subj_final,
                obj_final,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                graph_active_nodes,
            ):
                continue

            # SPECIAL CASE: For first sentence only, allow pronouns as nodes
            pronouns = {"he", "she", "it", "they", "him", "her", "i", "you", "we"}

            if self._current_sentence_index == 0:
                # Handle subject
                if subj_final.lower() in pronouns:
                    source_node = self._graph.add_or_get_node(
                        lemmas=[subj_final.lower()],
                        actual_text=subj_final,
                        node_type=NodeType.CONCEPT,
                        node_source=NodeSource.TEXT_BASED,
                        provenance=NodeProvenance.STORY_TEXT,
                        origin_sentence=self._current_sentence_index,
                        mark_explicit=True,
                    )
                else:
                    # Normal node creation path for subject
                    source_node = self.get_or_create_node_from_relationship(
                        subj_final,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                # Handle object
                if obj_final.lower() in pronouns:
                    dest_node = self._graph.add_or_get_node(
                        lemmas=[obj_final.lower()],
                        actual_text=obj_final,
                        node_type=NodeType.CONCEPT,
                        node_source=NodeSource.TEXT_BASED,
                        provenance=NodeProvenance.STORY_TEXT,
                        origin_sentence=self._current_sentence_index,
                        mark_explicit=True,
                    )
                else:
                    # Normal node creation path for object
                    dest_node = self.get_or_create_node_from_relationship(
                        obj_final,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        NodeSource.TEXT_BASED,
                        create_node=True,
                    )
            else:
                # Normal processing for sentences 2+
                source_node, dest_node = self.get_or_create_nodes(
                    subj_final,
                    obj_final,
                    graph_active_nodes,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                )

            if not source_node or not dest_node:
                continue

            # clean and validate relation label
            edge_label = validator.clean_and_validate_relation(rel_final)
            if not edge_label:
                continue

            # update node sources if they appear in current sentence
            self.update_node_source_if_in_sentence(source_node, sentence_lemma_keys)
            self.update_node_source_if_in_sentence(dest_node, sentence_lemma_keys)

            # create the edge
            edge = self.add_edge_wrapper_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

            if edge:
                added_edges.append(edge)
            else:
                logging.warning(
                    f"FS_DEBUG: edge NOT added: {source_node} --{edge_label}--> {dest_node}"
                )

        return added_edges

    def get_triplet_validator(self):
        if not hasattr(self, "_triple_validator") or self._triple_validator is None:
            self._triple_validator = TripletValidator(
                linguistic_ops=self,
                extract_deterministic_fn=self._extract_deterministic_structure_fn,
                text_normalizer=self.text_normalizer,
                client=self.llm,
                persona=self.persona,
                spacy_nlp=self.spacy_nlp,
            )
        return self._triple_validator

    def check_attachable(
        self,
        subj: str,
        obj: str,
        current_words: list,
        current_nodes: list,
        graph_active_nodes: list,
    ) -> bool:
        return self.is_attachable_wrapper_fn(
            subj,
            obj,
            current_words,
            current_nodes,
            graph_active_nodes,
            self.get_nodes_with_active_edges_fn(),
        )

    def get_or_create_nodes(
        self,
        subj: str,
        obj: str,
        graph_active_nodes: list,
        current_nodes: list,
        current_words: list,
    ) -> tuple:
        # Determine source for subject
        subj_source = (
            NodeSource.TEXT_BASED
            if subj in current_words
            else NodeSource.INFERENCE_BASED
        )
        source_node = self._get_node_from_new_relationship_fn(
            subj,
            graph_active_nodes,
            current_nodes,
            current_words,
            node_source=subj_source,
            create_node=True,
        )

        # Determine source for object
        obj_source = (
            NodeSource.TEXT_BASED
            if obj in current_words
            else NodeSource.INFERENCE_BASED
        )
        dest_node = self._get_node_from_new_relationship_fn(
            obj,
            graph_active_nodes,
            current_nodes,
            current_words,
            node_source=obj_source,
            create_node=True,
        )

        return source_node, dest_node

    def update_node_source_if_in_sentence(self, node, sentence_lemma_keys):
        if tuple(node.lemmas) in sentence_lemma_keys:
            node.node_source = NodeSource.TEXT_BASED

    # sometimes, the parser returns sentences such as: "Processing sentence 0: answer is: A man very close to Charlemagne wrote most of the things we know about Charlemagne."
    def clean_llm_output(self, resolved_text: str, original_text: str, sent) -> tuple:

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
            fallback_doc = self.spacy_nlp(original_text.strip())
            fallback_sents = list(fallback_doc.sents)
            return original_text.strip(), fallback_sents[0] if fallback_sents else sent

        return resolved_text, sents[0]

    # decay first, then connect
    def run_post_processing(
        self,
        explicit_nodes: set,
        carryover_nodes: set,
        apply_edge_decay_fn: callable,
        enforce_node_limit_fn: callable = None,
    ) -> None:
        apply_edge_decay_fn()
        if enforce_node_limit_fn:
            enforce_node_limit_fn()
        # self.graph.stabilize_cumulative_graph_wrapper(set(explicit_nodes))
