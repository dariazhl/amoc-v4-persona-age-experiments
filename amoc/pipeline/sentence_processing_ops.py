import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple, Set

from amoc.graph.node import NodeType, NodeSource

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
        self._restrict_active_to_current_explicit_fn = None
        self._get_node_from_new_relationship_fn = None
        self._get_phrase_level_concepts_fn = None
        self._get_sentences_text_based_nodes_fn = None
        self._infer_new_relationships_fn = None
        self._add_inferred_relationships_to_graph_fn = None

        self._explicit_nodes_ref = None
        self._anchor_nodes_ref = None
        self._triplet_intro_ref = None
        self._carryover_nodes_ref = None
        self.persona = None
        self.ENFORCE_ATTACHMENT_CONSTRAINT = True

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
        restrict_active_to_current_explicit_fn,
        get_node_from_new_relationship_fn,
        get_phrase_level_concepts_fn,
        get_sentences_text_based_nodes_fn,
        infer_new_relationships_fn,
        add_inferred_relationships_to_graph_fn,
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
            extract_deterministic_structure_fn=lambda *args, **kwargs: None,
            infer_edges_to_recently_deactivated_fn=core._infer_edges_to_recently_deactivated,
            restrict_active_to_current_explicit_fn=lambda en: None,
            get_node_from_new_relationship_fn=core._resolve_node_from_new_relationship,
            get_phrase_level_concepts_fn=lambda *args, **kwargs: [],
            get_sentences_text_based_nodes_fn=core._get_sentences_nodes,
            infer_new_relationships_fn=core._infer_new_relationships_for_sentence,
            add_inferred_relationships_to_graph_fn=core.add_inferred_relationships_to_graph,
        )
        self.set_state_refs(
            explicit_nodes_ref=core._get_explicit_nodes,
            anchor_nodes_ref=core._dummy_anchor_nodes,
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

        prev_sentences.append(resolved_text)
        if len(prev_sentences) > self.context_length:
            prev_sentences.pop(0)

        # llm node extraction
        llm_nodes = self.client.get_explicit_nodes_from_text(
            resolved_text,
            self.persona,
        )

        concepts = llm_nodes.get("concepts", [])
        properties = llm_nodes.get("properties", [])

        current_nodes = []

        for concept in concepts:
            node = self.graph.add_or_get_node(
                lemmas=[concept.lower()],
                actual_text=concept.lower(),
                node_type=NodeType.CONCEPT,
                node_source=NodeSource.TEXT_BASED,
                origin_sentence=current_sentence_index,
            )
            if node:
                current_nodes.append(node)

        for prop in properties:
            node = self.graph.add_or_get_node(
                lemmas=[prop.lower()],
                actual_text=prop.lower(),
                node_type=NodeType.PROPERTY,
                node_source=NodeSource.TEXT_BASED,
                origin_sentence=current_sentence_index,
            )
            if node:
                current_nodes.append(node)

        explicit_nodes_current_sentence.clear()
        explicit_nodes_current_sentence.update(current_nodes)

        # llm relationships
        nodes_from_text = ""
        for node in current_nodes:
            nodes_from_text += f" - ({node.get_text_representer()}, {node.node_type})\n"

        graph_nodes_text = self.graph.get_nodes_str(self.graph.nodes)
        graph_edges_text, _ = self.graph.get_edges_str(self.graph.nodes)

        new_relationships = self.client.get_new_relationships(
            nodes_from_text,
            graph_nodes_text,
            graph_edges_text,
            resolved_text,
            self.persona,
        )
        sentence_lemma_keys = {tuple(n.lemmas) for n in current_nodes}
        self._process_llm_relationships(
            new_relationships=new_relationships,
            current_sentence_text_based_nodes=current_nodes,
            current_sentence_text_based_words=[
                n.get_text_representer() for n in current_nodes
            ],
            graph_active_nodes=list(self.graph.nodes),
            sentence_lemma_keys=sentence_lemma_keys,
            explicit_nodes_current_sentence=explicit_nodes_current_sentence,
        )

        # anchors
        anchor_nodes.clear()
        anchor_nodes.update(
            n
            for n in explicit_nodes_current_sentence
            if n.node_type == NodeType.CONCEPT
        )

        return (nodes_before_sentence, False)

    def _normalize_relationship(self, relationship):
        # Normalize LLM output into (subj, rel, obj)
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

        normalized = []
        for rel in new_relationships or []:
            triple = self._normalize_relationship(rel)
            if triple:
                normalized.append(triple)

        for subj, rel, obj in normalized:

            subj = self._normalize_endpoint_text_fn(subj, is_subject=True)
            obj = self._normalize_endpoint_text_fn(obj, is_subject=False)

            if not subj or not obj or subj == obj:
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

            if not source_node or not dest_node:
                continue

            edge_label = self._normalize_edge_label_fn(
                rel.replace("(edge)", "").strip()
            )
            if not self._is_valid_relation_label_fn(edge_label):
                continue

            canon_label, _, _, swapped = self._canonicalize_edge_direction_fn(
                edge_label,
                source_node.get_text_representer(),
                dest_node.get_text_representer(),
            )

            if swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            if tuple(source_node.lemmas) in sentence_lemma_keys:
                source_node.node_source = NodeSource.TEXT_BASED
            if tuple(dest_node.lemmas) in sentence_lemma_keys:
                dest_node.node_source = NodeSource.TEXT_BASED

            edge = self._add_edge_fn(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

            if edge:
                added_edges.append(edge)

        return added_edges

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
        # B-state: global activation scheduler hooks are disabled.
        return None
