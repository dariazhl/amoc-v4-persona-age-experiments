import logging
import os
import re
from typing import List, Tuple, Optional, Iterable
import pandas as pd
from spacy.tokens import Span, Token
import networkx as nx

from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.graph.node import NodeProvenance, NodeRole
from amoc.graph.per_sentence_graph import (
    PerSentenceGraph,
    PerSentenceGraphBuilder,
    build_per_sentence_graph,
)
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.llm.vllm_client import VLLMClient
from amoc.nlp.spacy_utils import (
    get_concept_lemmas,
    canonicalize_node_text,
    get_content_words_from_sent,
    extract_adjectival_modifiers,
    extract_prepositional_objects,
    canonicalize_edge_label,
    is_adverb_token,
    are_semantically_equivalent,
    get_semantic_class,
)
from collections import deque
from amoc.config.paths import OUTPUT_ANALYSIS_DIR


def _sanitize_filename_component(component: str, max_len: int = 80) -> str:
    component = (component or "").replace("\n", " ").strip()
    component = component[:max_len]
    component = re.sub(r"[\\/:*?\"<>|]", "_", component)
    component = re.sub(r"\s+", "_", component)
    return component or "unknown"


class AMoCv4:
    # Generic relations to filter out - these are too vague to be useful
    # NOTE: Simple "is" and "be" are KEPT because the AMoC paper uses them
    # (e.g., "knight - is - brave" in Figure 7)
    GENERIC_RELATION_LABELS = {
        "appears",
        "contains",
        "includes",
        "include",
        "contain",
        "refers to",
        "involves",
        "describes",
    }

    ENFORCE_ATTACHMENT_CONSTRAINT = False
    ACTIVATION_MAX_DISTANCE = 2
    RELATION_BLACKLIST = {"describes", "is_at_stake"}

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
        self.spacy_nlp = spacy_nlp

        if self.spacy_nlp is None:
            raise RuntimeError("AMoCv4 requires a spaCy nlp object (spacy_nlp).")

        self.debug = debug
        # Cache story lemmas for quick membership checks (provenance validation)
        story_doc = self.spacy_nlp(story_text)
        self.story_lemmas = {tok.lemma_.lower() for tok in story_doc if tok.is_alpha}

        # PROVENANCE GATE: Extract persona lemmas to create blocklist
        # Per AMoC v4 paper: Persona influences SALIENCE only, never CONTENT
        # Nodes must come from story text, never from persona description
        persona_doc = self.spacy_nlp(persona_description)
        self._persona_only_lemmas = {
            tok.lemma_.lower() for tok in persona_doc if tok.is_alpha
        } - self.story_lemmas  # Only lemmas unique to persona (not in story)
        self._prev_active_nodes_for_plot: set[Node] = set()
        self._cumulative_deactivated_nodes_for_plot: set[Node] = set()
        self._viz_positions: dict[str, tuple[float, float]] = {}
        self._recently_deactivated_nodes_for_inference: set[Node] = set()
        self._anchor_nodes: set[Node] = set()
        self._explicit_nodes_current_sentence: set[Node] = set()
        self.strict_reactivate_function = strict_reactivate_function
        self.strict_attachament_constraint = strict_attachament_constraint
        self.single_anchor_hub = single_anchor_hub
        # ==========================================================================
        # PAPER-ALIGNED SINGLE-EDGE POLICY
        # ==========================================================================
        # Per AMoC paper (Figures 2–6): At most one edge exists between any ordered
        # node pair ⟨subject, object⟩ at any time. Later relations replace earlier ones.
        # When allow_multi_edges=False (default, paper-aligned):
        #   - New edges between existing node pairs REPLACE the old edge
        #   - This models memory abstraction via replacement + decay
        # When allow_multi_edges=True (experimental):
        #   - Multiple edges can exist between the same nodes
        #   - Used for debugging or alternative memory models
        self.allow_multi_edges = allow_multi_edges
        self._current_sentence_text: str = ""
        # Separate memory (cumulative) vs salience (active) graphs for auditing.
        self.cumulative_graph = nx.MultiDiGraph()
        self.active_graph = nx.MultiDiGraph()
        # Stable triplet introduction index: (subj, rel, obj) -> introduced_at_sentence
        self._triplet_intro: dict[tuple[str, str, str], int] = {}
        # Append-only cumulative records (one row per active episode)
        self._cumulative_triplet_records: list[dict] = []
        self._fixed_hub = None
        # Per-sentence graph view (rebuilt each sentence for isolation)
        self._per_sentence_view: Optional[PerSentenceGraph] = None

    def _admit_node(
        self,
        lemma: str,
        node_type: NodeType,
        provenance: str,
        sent: Optional[Span] = None,
    ) -> bool:
        lemma = lemma.lower().strip()

        # 1. Lexical grounding (hard gate)
        if lemma not in self.story_lemmas:
            pass

        if node_type == NodeType.PROPERTY:
            if sent is None:
                return False

            # Check if the adjective appears in the sentence as ADJ
            grounded = any(
                tok.lemma_.lower() == lemma
                and tok.pos_ == "ADJ"
                and tok.dep_ in {"amod", "acomp", "attr"}
                for tok in sent
            )

            if not grounded:
                return False

        # 3. Provenance guard (hard gate)
        if provenance in {
            "LLM_PROMPT",
            "GRAPH_SERIALIZATION",
            "CSV",
            "PLOTTING",
            "META",
        }:
            return False

        if lemma not in self.story_lemmas and provenance not in {
            "INFERRED_RELATION",
            "INFERENCE_BASED",
        }:
            return False

        return True

    def _node_token_for_matrix(self, node: Node) -> str:
        return (node.get_text_representer() or "").strip().lower()

    # ==========================================================================
    # PROVENANCE VALIDATION: Per AMoC v4 paper alignment
    # ==========================================================================
    # CRITICAL: Nodes must come from STORY TEXT only, never from persona.
    # Persona influences salience/weights, never graph content.
    # ==========================================================================

    def _validate_node_provenance(
        self,
        lemma: str,
        current_sentence_text: Optional[str] = None,
        *,
        allow_bootstrap: bool = False,
    ) -> bool:
        lemma_lower = lemma.lower()

        # HARD GATE 1: Reject persona-only lemmas (never bypassed)
        if lemma_lower in self._persona_only_lemmas:
            if self.debug:
                logging.debug(
                    f"PROVENANCE GATE: Rejected persona-only lemma '{lemma_lower}'"
                )
            return False

        # HARD GATE 2: Must appear in story text
        if lemma_lower in self.story_lemmas:
            return True

        # Additional validation: check current sentence if provided
        # This allows nodes from the current sentence being processed
        if current_sentence_text:
            sent_doc = self.spacy_nlp(current_sentence_text)
            sent_lemmas = {tok.lemma_.lower() for tok in sent_doc if tok.is_alpha}
            if lemma_lower in sent_lemmas:
                return True

        # GRAPH GROUNDING: Allow concepts that already exist in the graph
        # This enables semantic inference chains:
        # - Sentence 1: knight → danger (danger bootstrapped)
        # - Sentence 2: danger → threat (danger now exists, threat can bootstrap)
        # Per AMoC v4 Figures 2-4: abstract concepts like "danger", "threat", "goal"
        # can appear when semantically connected to grounded concepts
        existing_node = self.graph.get_node([lemma_lower])
        if existing_node is not None:
            if self.debug:
                logging.debug(
                    f"PROVENANCE GATE: Graph grounding for '{lemma_lower}' (exists in graph)"
                )
            return True

        candidate_class = get_semantic_class(lemma_lower)
        if candidate_class is not None:
            for node in self.graph.get_active_nodes(
                self.max_distance_from_active_nodes,
                only_text_based=False,
            ):
                node_class = get_semantic_class(node.get_text_representer())
                if node_class == candidate_class:
                    if self.debug:
                        logging.debug(
                            f"PROVENANCE GATE: Semantic-class grounding "
                            f"'{lemma_lower}' via class '{candidate_class}'"
                        )
                    return True

        # BOOTSTRAP PATH: Allow inferred nodes that will be connected by an edge
        # This enables semantic relations from LLM extraction (e.g., "knight" → "danger")
        # where "danger" may not be a direct token but is semantically inferred.
        if allow_bootstrap:
            if self.debug:
                logging.debug(
                    f"PROVENANCE GATE: BOOTSTRAP allowed for '{lemma_lower}' (will be connected)"
                )
            return True

        # STRICT: Not in story text and not grounded - REJECT
        if self.debug:
            logging.debug(
                f"PROVENANCE GATE: Rejected lemma '{lemma_lower}' - not grounded"
            )
        return False

    def _validate_node_provenance_strict(
        self,
        lemma: str,
        current_sentence_text: Optional[str] = None,
    ) -> bool:
        """
        Strict provenance validation - rejects any lemma not in story text.

        Use this for node creation paths that MUST be grounded in story text.
        Fails for any lemma not explicitly present in story_lemmas.
        """
        lemma_lower = lemma.lower()

        # HARD GATE: Reject persona-only lemmas
        if lemma_lower in self._persona_only_lemmas:
            if self.debug:
                logging.debug(
                    f"PROVENANCE GATE (strict): Rejected persona-only lemma '{lemma_lower}'"
                )
            return False

        # STRICT: Must appear in story text
        if lemma_lower in self.story_lemmas:
            return True

        # Check current sentence as fallback
        if current_sentence_text:
            sent_doc = self.spacy_nlp(current_sentence_text)
            sent_lemmas = {tok.lemma_.lower() for tok in sent_doc if tok.is_alpha}
            if lemma_lower in sent_lemmas:
                return True

        if self.debug:
            logging.debug(
                f"PROVENANCE GATE (strict): Rejected lemma '{lemma_lower}' - not in story"
            )
        return False

    def _build_per_sentence_view(
        self, explicit_nodes: List[Node], sentence_index: int
    ) -> Optional[PerSentenceGraph]:
        if not self.strict_attachament_constraint:
            self._per_sentence_view = None
            return None

        self._per_sentence_view = build_per_sentence_graph(
            cumulative_graph=self.graph,
            explicit_nodes=explicit_nodes,
            max_distance=self.max_distance_from_active_nodes,
            anchor_nodes=self._anchor_nodes,
            sentence_index=sentence_index,
        )
        return self._per_sentence_view

    def _get_attachable_nodes_for_sentence(self) -> set[Node]:
        if self._per_sentence_view is not None:
            return (
                set(self._per_sentence_view.explicit_nodes)
                | set(self._per_sentence_view.carryover_nodes)
                | set(self._per_sentence_view.anchor_nodes)
            )
        # Fallback when no per-sentence view exists yet
        return (
            self._explicit_nodes_current_sentence
            | self._anchor_nodes
            | self._get_nodes_with_active_edges()
        )

    def _extract_adjectival_modifiers(self, sent: Span) -> dict[str, list[str]]:
        if not isinstance(sent, Span):
            return {}

        prop_map: dict[str, list[str]] = {}

        for tok in sent:
            if tok.pos_ not in {"NOUN", "PROPN"}:
                continue

            head = tok.lemma_.lower()

            for child in tok.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    adj = child.lemma_.lower()

                    if not self._admit_node(
                        lemma=adj,
                        node_type=NodeType.PROPERTY,
                        provenance="SYNTACTIC_PROPERTY",
                        sent=sent,
                    ):
                        continue

                    prop_map.setdefault(head, []).append(adj)

        return prop_map

    def _append_adjectival_hints(self, nodes_from_text: str, sent: Span) -> str:
        adjectival_properties = self._extract_adjectival_modifiers(sent)
        if adjectival_properties:
            nodes_from_text += "\nAdjectival hints:\n"
            for noun, adjectives in adjectival_properties.items():
                nodes_from_text += f" - {noun}: {', '.join(adjectives)}\n"
        return nodes_from_text

    def _distances_from_sources_active_edges(
        self, sources: set[Node], max_distance: int
    ) -> dict[Node, int]:
        if not sources:
            return {}
        distances: dict[Node, int] = {s: 0 for s in sources}
        queue: deque[Node] = deque(sources)
        while queue:
            node = queue.popleft()
            dist = distances[node]
            if dist >= max_distance:
                continue
            for edge in node.edges:
                if not edge.active:
                    continue
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                # NOTE: NodeType.RELATION check removed - type no longer exists in AMoCv4
                if neighbor in distances:
                    continue
                distances[neighbor] = dist + 1
                queue.append(neighbor)
        return distances

    def _record_sentence_activation(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        def _to_landscape_score(raw_score: float) -> float:
            # Transform AMoC "distance" style (0=most active) into Landscape style (5=most active).
            val = 5.0 - float(raw_score)
            if val < 0.0:
                return 0.0
            if val > 5.0:
                return 5.0
            return val

        # NOTE: NodeType.RELATION filter removed - type no longer exists in AMoCv4
        explicit_set = set(explicit_nodes)
        distances = self._distances_from_sources_active_edges(
            explicit_set, max_distance=self.ACTIVATION_MAX_DISTANCE
        )

        token_to_raw_score: dict[str, int] = {}
        node_raw_score: dict[Node, int] = {}

        # Step 2: explicit nodes reset to 0 (non-negotiable)
        for node in explicit_set:
            token = self._node_token_for_matrix(node)
            if token:
                token_to_raw_score[token] = 0
                node_raw_score[node] = 0

        # Step 5: newly inferred nodes start at 1 (never 0)
        for node in newly_inferred_nodes:
            if node in explicit_set:
                continue
            token = self._node_token_for_matrix(node)
            if token:
                token_to_raw_score[token] = 1
                node_raw_score[node] = 1

        # Step 3/4: carried-over nodes within range, score = distance
        for node, dist in distances.items():
            if node in explicit_set:
                continue
            if dist <= 0:
                continue
            token = self._node_token_for_matrix(node)
            if not token:
                continue
            if token in token_to_raw_score:
                continue
            token_to_raw_score[token] = dist
            node_raw_score[node] = dist

        # Convert node scores to Landscape scale and record.
        for token, raw_score in token_to_raw_score.items():
            self._amoc_matrix_records.append(
                {
                    "sentence": sentence_id,
                    "token": token,
                    "score": _to_landscape_score(raw_score),
                }
            )

        # Add verb (edge-label) activations: take the max activation of connected nodes minus 0.5.
        verb_scores: dict[str, float] = {}
        for edge in self.graph.edges:
            if not edge.active:
                continue
            label = (edge.label or "").strip().lower()
            if not label:
                continue
            src_tok = self._node_token_for_matrix(edge.source_node)
            dst_tok = self._node_token_for_matrix(edge.dest_node)
            if not src_tok or not dst_tok:
                continue
            src_raw = node_raw_score.get(edge.source_node, edge.source_node.score)
            dst_raw = node_raw_score.get(edge.dest_node, edge.dest_node.score)
            src_act = _to_landscape_score(src_raw)
            dst_act = _to_landscape_score(dst_raw)
            verb_act = max(src_act, dst_act) - 0.5
            if verb_act < 0.0:
                verb_act = 0.0
            prev = verb_scores.get(label)
            if prev is None or verb_act > prev:
                verb_scores[label] = verb_act

        for token, score in verb_scores.items():
            self._amoc_matrix_records.append(
                {"sentence": sentence_id, "token": token, "score": score}
            )

    def _infer_edges_to_recently_deactivated(
        self,
        current_sentence_nodes: List[Node],
        current_sentence_words: List[str],
        current_text: str,
    ) -> List[Edge]:
        if not self.ENFORCE_ATTACHMENT_CONSTRAINT:
            return []
        recent = [
            n
            for n in self._recently_deactivated_nodes_for_inference
            if n in self.graph.nodes
        ]
        if not recent or not current_sentence_nodes:
            return []

        candidate_pairs: set[frozenset[Node]] = set()
        for node in current_sentence_nodes:
            for other in recent:
                if node == other:
                    continue
                # NOTE: Removed _has_edge_between pre-check.
                # Edge replacement is now handled in _add_edge (paper-aligned single-edge policy).
                candidate_pairs.add(frozenset((node, other)))
        if not candidate_pairs:
            return []

        nodes_for_prompt = {n for pair in candidate_pairs for n in pair}

        def _node_line(node: Node) -> str:
            return f" - ({node.get_text_representer()}, {node.node_type})\n"

        nodes_from_text = "".join(
            _node_line(n)
            for n in sorted(nodes_for_prompt, key=lambda x: x.get_text_representer())
        )
        graph_nodes_repr = self.graph.get_nodes_str(list(nodes_for_prompt))
        graph_edges_repr, _ = self.graph.get_edges_str(
            list(nodes_for_prompt), only_text_based=False
        )

        try:
            new_relationships = self.client.get_new_relationships(
                nodes_from_text,
                graph_nodes_repr,
                graph_edges_repr,
                current_text,
                self.persona,
            )
        except Exception:
            logging.error("Targeted LLM edge inference failed", exc_info=True)
            return []

        added: List[Edge] = []
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
            if not isinstance(relationship, (list, tuple)) or len(relationship) != 3:
                continue

            subj, rel, obj = relationship
            subj = self._normalize_endpoint_text(subj, is_subject=True) or None
            obj = self._normalize_endpoint_text(obj, is_subject=False) or None
            if subj is None or obj is None:
                continue
            if not subj or not obj:
                continue
            if not isinstance(subj, str) or not isinstance(obj, str):
                continue
            edge_label = rel.replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label(edge_label)
            if not self._is_valid_relation_label(edge_label):
                continue

            subj_node = self._find_node_by_text(subj, nodes_for_prompt)
            obj_node = self._find_node_by_text(obj, nodes_for_prompt)
            if subj_node is None or obj_node is None:
                continue
            pair_key = frozenset((subj_node, obj_node))
            if pair_key not in candidate_pairs:
                continue
            # NOTE: Removed _has_edge_between pre-check.
            # Edge replacement is now handled in _add_edge (paper-aligned single-edge policy).

            # AMoCv4 surface-relation format: direct edge between entities
            # ⟨entity, verb, entity⟩ - NO intermediate RELATION nodes
            edge = self._add_edge(
                subj_node,
                obj_node,
                edge_label,
                self.edge_visibility,
            )
            # Track added edge (for reactivation bookkeeping)
            if edge:
                added.append(edge)
        return added

    def _passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
    ) -> bool:
        # 1. Bootstrap: allow first edges
        if not self.graph.nodes:
            return True

        # 2. If strict mode is ON:
        #    do NOT reject here — let _add_edge enforce connectivity
        if self.strict_attachament_constraint:
            return True

        # 3. Permissive mode (legacy behavior):
        #    require that at least one endpoint touches memory
        subject = canonicalize_node_text(self.spacy_nlp, subject)
        obj = canonicalize_node_text(self.spacy_nlp, obj)

        def _lemma_key(text: str) -> tuple[str, ...]:
            return tuple(get_concept_lemmas(self.spacy_nlp, text))

        subj_key = _lemma_key(subject)
        obj_key = _lemma_key(obj)

        memory_lemma_keys = {tuple(n.lemmas) for n in self.graph.nodes}

        touches_memory = (
            subj_key in memory_lemma_keys
            or obj_key in memory_lemma_keys
            or subject in current_sentence_words
            or obj in current_sentence_words
        )

        return touches_memory

    def _add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
        bypass_attachment_constraint: bool = False,
    ) -> Optional[Edge]:
        # Canonicalize relation label before edge creation
        # Removes parser prefixes/artifacts and normalizes format
        label = Graph.canonicalize_relation_label(label)
        if not label:
            return None

        # STRICT MODE: enforce structural connectivity guarantee
        if (
            self.strict_attachament_constraint
            and self.graph.edges
            and not bypass_attachment_constraint
            and not (
                source_node.node_source == NodeSource.INFERENCE_BASED
                or dest_node.node_source == NodeSource.INFERENCE_BASED
                or source_node not in self.graph.nodes
                or dest_node not in self.graph.nodes
            )
        ):
            # Build undirected view of current graph
            G = nx.Graph()
            for e in self.graph.edges:
                G.add_edge(e.source_node, e.dest_node)

            # Find main component (prefer one containing anchors)
            main_component: set[Node] = set()
            for comp in nx.connected_components(G):
                if any(n in self._anchor_nodes for n in comp):
                    main_component = set(comp)
                    break

            # Fallback: largest component
            if not main_component and G.number_of_nodes() > 0:
                main_component = set(max(nx.connected_components(G), key=len))

            # Nodes allowed to attach new edges
            attachable = (
                main_component
                | self._anchor_nodes
                | getattr(self, "_explicit_nodes_current_sentence", set())
            )

            # HARD GUARD: at least one endpoint must be attachable
            if source_node not in attachable and dest_node not in attachable:
                return None

        use_sentence = (
            created_at_sentence
            if created_at_sentence is not None
            else getattr(self, "_current_sentence_index", None)
        )

        # ==========================================================================
        # SINGLE-EDGE POLICY WITH SEMANTIC EQUIVALENCE (AMoC v4 Paper-Aligned)
        # ==========================================================================
        # Per AMoC paper: graphs display a single semantic relation between entities.
        #
        # SEMANTIC REPLACEMENT RULES:
        # 1. If existing edge is SEMANTICALLY EQUIVALENT → REPLACE with new label
        #    (e.g., "traverses" and "rides_through" are both MOTION)
        # 2. If existing edge is SEMANTICALLY INCOMPATIBLE → REJECT new edge
        #    (e.g., "kidnaps" (CAPTURE) vs "fights" (COMBAT) are different)
        #
        # This prevents paraphrase accumulation while allowing semantic updates.
        if not self.allow_multi_edges:
            existing_edge = self._get_existing_edge_between_nodes(
                source_node, dest_node
            )
            if existing_edge is not None:
                old_label = existing_edge.label

                # Check semantic equivalence
                if are_semantically_equivalent(old_label, label):
                    # REPLACE: semantically equivalent, update to newer form
                    existing_edge.label = label
                    existing_edge.visibility_score = edge_forget
                    existing_edge.created_at_sentence = use_sentence
                    existing_edge.mark_as_asserted(reset_score=True)

                    if self.debug:
                        logging.debug(
                            "[SingleEdge] REPLACED equivalent edge: %s --%s--> %s (was: %s, class: %s)",
                            source_node.get_text_representer(),
                            label,
                            dest_node.get_text_representer(),
                            old_label,
                            get_semantic_class(
                                label.split("_")[0] if "_" in label else label
                            ),
                        )

                    # Update triplet intro tracking with new label
                    trip_id = (
                        existing_edge.source_node.get_text_representer(),
                        existing_edge.label,
                        existing_edge.dest_node.get_text_representer(),
                    )
                    if trip_id not in self._triplet_intro:
                        self._triplet_intro[trip_id] = (
                            use_sentence if use_sentence is not None else -1
                        )

                    self._record_edge_in_graphs(
                        existing_edge, self._current_sentence_index
                    )
                    return existing_edge
                else:
                    # ==========================================================================
                    # INCOMPATIBILITY DECAY: Both edges coexist, decay resolves conflict
                    # ==========================================================================
                    # Per AMoC v4 paper alignment: competing relations age out gradually.
                    # DO NOT reject the new edge or replace the old edge.
                    # Instead:
                    # 1. Decay the existing edge (reduce visibility)
                    # 2. Create new edge with NORMAL visibility (marked as asserted)
                    # 3. Let natural decay resolve which relation survives
                    #
                    # If existing edge becomes inactive (visibility <= 0), remove it
                    # and only then add the new edge as the sole edge.
                    # ==========================================================================
                    existing_edge.reduce_visibility()

                    # If old edge decayed to inactive, remove it and add the new edge
                    if not existing_edge.active:
                        self.graph.remove_edge(existing_edge)
                        # Fall through to create new edge below
                    else:
                        # BOTH EDGES COEXIST: old edge still active, add new edge alongside
                        # New edge has NORMAL visibility and is marked as asserted
                        new_edge = self.graph.add_edge(
                            source_node,
                            dest_node,
                            label,
                            edge_visibility=edge_forget,  # NORMAL visibility
                            created_at_sentence=use_sentence,
                        )

                        if new_edge:
                            # Mark as properly asserted with normal activation
                            new_edge.mark_as_asserted(reset_score=True)

                            trip_id = (
                                new_edge.source_node.get_text_representer(),
                                new_edge.label,
                                new_edge.dest_node.get_text_representer(),
                            )
                            if trip_id not in self._triplet_intro:
                                self._triplet_intro[trip_id] = (
                                    use_sentence if use_sentence is not None else -1
                                )

                            self._record_edge_in_graphs(
                                new_edge, self._current_sentence_index
                            )
                            return new_edge

                        return None  # Failed to create competing edge

        # No existing edge (or allow_multi_edges=True, or old edge was removed): create new edge
        edge = self.graph.add_edge(
            source_node,
            dest_node,
            label,
            edge_forget,
            created_at_sentence=use_sentence,
        )

        if edge:
            # Mark edge as ASSERTED this sentence (per AMoC v4 paper semantics)
            # New edges are explicitly asserted, not reactivated or connectors
            edge.mark_as_asserted(reset_score=True)

            trip_id = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            if trip_id not in self._triplet_intro:
                self._triplet_intro[trip_id] = (
                    use_sentence if use_sentence is not None else -1
                )

            self._record_edge_in_graphs(edge, self._current_sentence_index)

        return edge

    def reset_graph(self) -> None:
        self.graph = Graph()
        self._anchor_nodes = set()

    def _enforce_graph_connectivity(self) -> None:
        # No-op: do not prune edges after addition; connectivity is enforced at add time.
        return

    # ==========================================================================
    # TASK 2: FORCED CONNECTIVITY EDGE CREATION
    # ==========================================================================
    def _create_forced_connectivity_edges(
        self,
        story_context: str,
        current_sentence: str,
    ) -> List[Edge]:
        """
        TASK 2: Create forced connectivity edges when existing edges cannot restore connectivity.

        This is the SECONDARY LLM CALL that fires ONLY when:
        1. ensure_active_connectivity() has been called
        2. The graph is STILL disconnected (check_active_connectivity() returns False)
        3. No existing edges in cumulative memory can bridge the components

        The method:
        1. Identifies disconnected components using get_nodes_needing_connection()
        2. For each pair needing connection, calls LLM to generate a minimal edge
        3. Creates the edge and marks it as forced_connection

        Args:
            story_context: Previous sentences for LLM context
            current_sentence: Current sentence being processed

        Returns:
            List of newly created forced connectivity edges
        """
        # Step 1: Check if graph is still disconnected
        if self.graph.check_active_connectivity():
            return []  # Already connected, no forced edges needed

        # Step 2: Get pairs that need connecting
        pairs = self.graph.get_nodes_needing_connection(
            focus_nodes=self._explicit_nodes_current_sentence
        )

        if not pairs:
            logging.warning(
                "[Connectivity] Graph disconnected but no pairs identified for connection"
            )
            return []

        logging.info(
            "[Connectivity] SECONDARY LLM CALL: Creating %d forced connectivity edges",
            len(pairs),
        )

        forced_edges: List[Edge] = []

        for isolated_node, focus_node in pairs:
            # Step 3: Call LLM to generate edge label
            node_a_text = isolated_node.get_text_representer()
            node_b_text = focus_node.get_text_representer()

            result = self.client.get_forced_connectivity_edge_label(
                node_a=node_a_text,
                node_b=node_b_text,
                story_context=story_context,
                current_sentence=current_sentence,
                persona=self.persona,
            )

            edge_label = result.get("label", "relates to")
            explanation = result.get("explanation", "")

            logging.debug(
                "[Connectivity] Creating forced edge: %s --%s--> %s (reason: %s)",
                node_a_text,
                edge_label,
                node_b_text,
                explanation,
            )

            # Step 4: Normalize and create the edge through centralized path
            edge_label = self._normalize_edge_label(edge_label)
            if not edge_label:
                logging.warning(
                    "[Connectivity] Edge label rejected after normalization: %s --%s--> %s",
                    node_a_text,
                    result.get("label", "relates to"),
                    node_b_text,
                )
                continue

            edge = self._add_edge(
                isolated_node,
                focus_node,
                edge_label,
                self.edge_visibility,
                created_at_sentence=self._current_sentence_index,
            )

            if edge is not None:
                # Mark as forced connectivity edge
                edge.mark_as_forced_connection()
                forced_edges.append(edge)

                # Record in graphs for auditing
                self._record_edge_in_graphs(edge, self._current_sentence_index)

                logging.info(
                    "[Connectivity] Created forced edge: %s --%s--> %s",
                    node_a_text,
                    edge_label,
                    node_b_text,
                )
            else:
                logging.warning(
                    "[Connectivity] Failed to create forced edge: %s --%s--> %s",
                    node_a_text,
                    edge_label,
                    node_b_text,
                )

        # Verify connectivity after creating forced edges
        if not self.graph.check_active_connectivity():
            logging.error(
                "[Connectivity] Graph STILL disconnected after creating %d forced edges",
                len(forced_edges),
            )

        return forced_edges

    def resolve_pronouns(self, text: str) -> str:
        resolved = self.client.resolve_pronouns(text, self.persona)
        if not isinstance(resolved, str) or not resolved.strip():
            return text
        low = resolved.lower()
        if "does not mention any pronouns" in low or "no pronouns to replace" in low:
            return text
        return resolved

    def _graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
        """AMoCv4 surface-relation format: edges ARE the triplets."""
        triplets: List[Tuple[str, str, str]] = []
        for edge in self.graph.edges:
            if only_active and not edge.active:
                continue
            # NOTE: agent_of/target_of check removed - AMoCv4 format never creates these
            if not edge.label or not str(edge.label).strip():
                continue
            if edge.source_node == edge.dest_node:
                continue
            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return triplets

    def _reconstruct_semantic_triplets(
        self,
        *,
        only_active: bool = False,
        restrict_nodes: Optional[set[Node]] = None,
    ):
        """
        AMoCv4 surface-relation format: edges ARE the semantic triplets.
        No reconstruction needed - just filter and return.
        """
        trips = []

        for edge in self.graph.edges:
            if only_active and not edge.active:
                continue
            if not edge.label or not edge.label.strip():
                continue
            if edge.source_node == edge.dest_node:
                continue
            if restrict_nodes is not None:
                if (
                    edge.source_node not in restrict_nodes
                    or edge.dest_node not in restrict_nodes
                ):
                    continue
            trips.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return trips

    # STEP 5 from paper - HARD RESET: Only explicit nodes from current sentence are active
    def _restrict_active_to_current_explicit(self, explicit_nodes: List[Node]) -> None:
        # Start with explicit nodes from sentence text
        seed_nodes = set(explicit_nodes)

        for edge in self.graph.edges:
            if (
                edge.is_property_edge()
                and edge.source_node in seed_nodes
                and edge.visibility_score > 0
            ):
                seed_nodes.add(edge.dest_node)

        # The BFS function will compute distances from these seed nodes
        # Nodes unreachable or beyond max_distance get score=100 (inactive)
        self.graph.set_nodes_score_based_on_distance_from_active_nodes(list(seed_nodes))

    def _get_nodes_with_active_edges(self) -> set[Node]:
        active_nodes: set[Node] = set()
        for edge in self.graph.edges:
            if edge.active:
                active_nodes.add(edge.source_node)
                active_nodes.add(edge.dest_node)
        # Ensure explicit nodes are always considered attachable.
        return active_nodes | getattr(self, "_explicit_nodes_current_sentence", set())

    def _can_attach(self, node: Node) -> bool:
        attachable = (
            self._get_nodes_with_active_edges()
            | self._anchor_nodes
            | getattr(self, "_explicit_nodes_current_sentence", set())
        )
        return node in attachable

    def _normalize_label(self, label: str) -> str:
        norm = (label or "").strip().lower()
        norm = re.sub(r"[\s\-]+", "_", norm)
        return norm

    def _edge_key(self, edge: Edge) -> tuple[str, str, str]:
        return (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )

    def _get_edge_activation_scores(self) -> dict[tuple[str, str, str], int]:
        """Get activation scores for all edges, keyed by (source, dest, label)."""
        scores = {}
        for edge in self.graph.edges:
            key = (
                edge.source_node.get_text_representer(),
                edge.dest_node.get_text_representer(),
                edge.label,
            )
            scores[key] = edge.activation_score
            # Also add 2-tuple key for compatibility
            scores[(key[0], key[1])] = edge.activation_score
        return scores

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        u, v, lbl = self._edge_key(edge)
        # Safety check: skip recording edges with empty/whitespace labels
        if not lbl or not lbl.strip():
            logging.warning(
                "Skipping recording edge with empty label: %s -> %s",
                u,
                v,
            )
            return
        introduced = self._triplet_intro.get((u, lbl, v))
        if introduced is None:
            introduced = (
                edge.created_at_sentence if edge.created_at_sentence is not None else -1
            )
        self._triplet_intro[(u, lbl, v)] = int(introduced)
        edge_key = f"{lbl}__introduced_{introduced}"

        # Append-only cumulative record: one row per state observation when we touch an edge.
        if sentence_idx is not None:
            self._cumulative_triplet_records.append(
                {
                    "subject": u,
                    "relation": lbl,
                    "object": v,
                    "introduced_at": int(introduced),
                    "last_active": int(sentence_idx),
                    "currently_active": bool(edge.active),
                }
            )

        # Active projection (salience)
        if edge.active or edge.visibility_score > 0:
            self.active_graph.add_edge(
                u,
                v,
                key=edge_key,
                relation=lbl,
                introduced_at_sentence=int(introduced),
                last_active_sentence=int(
                    sentence_idx if sentence_idx is not None else -1
                ),
            )
        else:
            if self.active_graph.has_edge(u, v, key=edge_key):
                self.active_graph.remove_edge(u, v, key=edge_key)

    def _graph_to_triplets(self, graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
        trips: List[Tuple[str, str, str]] = []
        for u, v, key, data in graph.edges(keys=True, data=True):
            rel = data.get("relation") or key
            # Skip edges with empty/whitespace-only labels
            if not rel or not str(rel).strip():
                continue
            trips.append((u, rel, v))
        return trips

    def _cumulative_triplets_upto(
        self, sentence_idx: Optional[int] = None
    ) -> List[Tuple[str, str, str]]:
        trips = []
        for edge in self.graph.edges:
            if not edge.label:
                continue
            trips.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return trips

    def _is_generic_relation(self, label: str) -> bool:
        norm = self._normalize_label(label)
        return norm in self.GENERIC_RELATION_LABELS

    def _is_blacklisted_relation(self, label: str) -> bool:
        norm = self._normalize_label(label)
        return norm in self.RELATION_BLACKLIST

    def _is_verb_relation(self, label: str) -> bool:
        """
        Check if a relation label is verb-based per AMoC v4 paper.

        Per paper Figures 2-4, valid relations include:
        - Simple verbs: "rode", "kidnapped"
        - Verb + preposition: "rode through", "is unfamiliar with"
        - Auxiliary + verb: "was kidnapping", "wanted to free"
        - Copula + adjective: "is unfamiliar" (predicate adjective)

        Rejects:
        - Standalone adjectives/nouns (not verb-based)
        - Pure adverb phrases without verbs
        """
        doc = self.spacy_nlp(label)
        has_verb = False
        has_copula = False
        has_adj_after_copula = False
        has_standalone_noun = False

        prev_was_copula = False

        for tok in doc:
            if not getattr(tok, "is_alpha", False):
                continue

            pos = tok.pos_
            lemma = tok.lemma_.lower()

            # Track verbs and copulas
            if pos in {"VERB", "AUX"}:
                has_verb = True
                # Check if this is a copula (be verbs)
                if lemma in {"be", "is", "was", "were", "been", "being", "am", "are"}:
                    has_copula = True
                    prev_was_copula = True
                else:
                    prev_was_copula = False

            # Allow adjectives only after copulas (predicate adjectives)
            # e.g., "is unfamiliar" is valid
            elif pos == "ADJ":
                if prev_was_copula or has_copula:
                    has_adj_after_copula = True
                prev_was_copula = False

            # Nouns in relations are problematic unless they're part of a phrase
            elif pos in {"NOUN", "PROPN"}:
                # Check if this is a standalone noun (not part of verb phrase)
                # Skip this check for now - rely on context
                has_standalone_noun = True
                prev_was_copula = False

            # Prepositions, particles, adverbs are fine as modifiers
            elif pos in {"ADP", "PART", "ADV"}:
                prev_was_copula = False

        # Accept if has a verb
        # Also accept copula + adjective constructions ("is unfamiliar")
        if has_verb:
            return True
        if has_copula and has_adj_after_copula:
            return True

        return False

    def _canonicalize_edge_direction(
        self, label: str, source_text: str, dest_text: str
    ) -> tuple[str, str, str, bool]:
        """
        Canonicalize edge direction by detecting and normalizing passive voice.

        Per AMoC v4 paper: edges are directed semantic propositions.
        - Active voice: subject → object (e.g., "dragon kidnapped princess")
        - Passive voice must be normalized to active: swap direction

        Args:
            label: The edge label (verb phrase)
            source_text: Current source node text
            dest_text: Current destination node text

        Returns:
            (normalized_label, final_source, final_dest, was_swapped)

        Examples:
            "was kidnapped by" → ("kidnapped", dest, source, True)
            "is threatened by" → ("threatens", dest, source, True)
            "kidnapped" → ("kidnapped", source, dest, False)
        """
        if not label or not isinstance(label, str):
            return (label, source_text, dest_text, False)

        label_lower = label.strip().lower()

        # Passive voice patterns that require direction swap
        # Pattern: "was/is/were Xed by" → "X" with swapped direction
        passive_patterns = [
            (r"^(was|is|were|been|being)\s+(\w+ed)\s+by$", 2),  # "was kidnapped by"
            (r"^(was|is|were|been|being)\s+(\w+en)\s+by$", 2),  # "was taken by"
            (r"^(was|is|were|been|being)\s+(\w+)\s+by$", 2),  # "was hurt by"
        ]

        for pattern, verb_group in passive_patterns:
            match = re.match(pattern, label_lower)
            if match:
                # Extract the main verb and convert to active voice
                verb = match.group(verb_group)
                # Swap source and dest to reflect active voice direction
                logging.debug(
                    "[EdgeDirection] Passive detected: %r → active: %r (swapped)",
                    label_lower,
                    verb,
                )
                return (verb, dest_text, source_text, True)

        # Inverse relation patterns (semantic inverses)
        # These describe the same relation from opposite perspectives
        inverse_mappings = {
            "is threatened by": ("threatens", True),
            "was threatened by": ("threatens", True),
            "is loved by": ("loves", True),
            "was loved by": ("loves", True),
            "is hated by": ("hates", True),
            "was hated by": ("hates", True),
            "is owned by": ("owns", True),
            "was owned by": ("owns", True),
            "belongs to": ("owns", True),  # "X belongs to Y" → "Y owns X"
        }

        if label_lower in inverse_mappings:
            active_label, should_swap = inverse_mappings[label_lower]
            if should_swap:
                logging.debug(
                    "[EdgeDirection] Inverse relation: %r → %r (swapped)",
                    label_lower,
                    active_label,
                )
                return (active_label, dest_text, source_text, True)
            return (active_label, source_text, dest_text, False)

        # No passive/inverse detected - keep original direction
        return (label, source_text, dest_text, False)

    def _normalize_edge_label(self, label: str) -> str:
        """
        Normalize edge label per AMoC v4 paper.

        ==========================================================================
        AMoC v4 CANONICAL EDGE LABELS
        ==========================================================================
        - Converts all verbs to simple present tense (canonical form)
        - Uses underscore formatting for multi-word labels
        - REJECTS modal/intentional verbs entirely (returns "")
        - Removes copula from adjective constructions

        Examples:
        - "is walking" → "walks"
        - "was kidnapping" → "kidnaps"
        - "running through" → "runs_through"
        - "is unfamiliar with" → "unfamiliar_with"
        - "rode through" → "rides_through"

        REJECTIONS (returns ""):
        - "wants to free" → "" (modal verb)
        - "tries to escape" → "" (intentional verb)
        """
        if not label or not isinstance(label, str):
            return label

        label = label.strip()
        if not label:
            return label

        # Use the centralized canonicalization function
        result = canonicalize_edge_label(self.spacy_nlp, label)

        # Reject malformed labels (too short, repeated syllables, non-alphabetic noise)
        if len(result) > 0:
            # Check for repeated character sequences (sign of corruption)
            if re.search(r"(.)\1{2,}", result):  # 3+ repeated chars
                logging.debug("[EdgeLabel] Rejected repeated chars: %r", result)
                return ""
            # Check for non-word patterns (consonant clusters without vowels)
            words = result.split()
            for word in words:
                if len(word) > 3 and not re.search(r"[aeiou]", word):
                    logging.debug("[EdgeLabel] Rejected no-vowel word: %r", result)
                    return ""

        return result

    def _is_valid_relation_label(self, label: str) -> bool:
        # Explicitly handle None, empty string, and whitespace-only labels
        if not label or not isinstance(label, str):
            logging.debug("[EdgeFilter] Rejected empty/None label: %r", label)
            return False
        label_stripped = label.strip()
        if not label_stripped:
            logging.debug("[EdgeFilter] Rejected whitespace-only label: %r", label)
            return False
        if self._is_generic_relation(label_stripped):
            logging.debug("[EdgeFilter] Rejected generic relation: %r", label_stripped)
            return False
        if self._is_blacklisted_relation(label_stripped):
            logging.debug(
                "[EdgeFilter] Rejected blacklisted relation: %r", label_stripped
            )
            return False
        if not self._is_verb_relation(label_stripped):
            logging.debug("[EdgeFilter] Rejected non-verb relation: %r", label_stripped)
            return False
        logging.debug("[EdgeFilter] Accepted relation: %r", label_stripped)
        return True

    def _normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        if not text:
            return None
        doc = self.spacy_nlp(text)
        if not doc:
            return None
        allowed_subject = {"NOUN", "PROPN", "PRON"}
        allowed_object = {"NOUN", "PROPN", "PRON"}
        for tok in doc:
            if not getattr(tok, "is_alpha", False):
                continue
            pos = tok.pos_
            if is_subject and pos not in allowed_subject:
                continue
            if not is_subject and pos not in allowed_object:
                continue
            lemma = (getattr(tok, "lemma_", "") or "").strip().lower()
            if not lemma or lemma in self.spacy_nlp.Defaults.stop_words:
                continue
            return lemma
        return None

    def _has_edge_between(
        self, a: Node, b: Node, relation_lemma: Optional[str] = None
    ) -> bool:
        """
        AMoCv4 surface-relation format: check for direct edge between nodes.
        """
        for edge in a.edges:
            other = edge.dest_node if edge.source_node == a else edge.source_node
            if other == b:
                if relation_lemma is None:
                    return True
                # Check if edge label matches the relation lemma
                if relation_lemma in edge.label.lower():
                    return True
        return False

    def _get_existing_edge_between_nodes(
        self, source_node: Node, dest_node: Node
    ) -> Optional[Edge]:
        """
        Find an existing DIRECTED edge from source_node to dest_node.

        Per AMoC paper: edges are directed semantic propositions.
        (A → B) is distinct from (B → A).

        Returns the existing edge if found, None otherwise.
        """
        for edge in source_node.edges:
            if edge.source_node == source_node and edge.dest_node == dest_node:
                return edge
        return None

    def _find_node_by_text(
        self, text: str, candidates: Iterable[Node]
    ) -> Optional[Node]:
        canon = canonicalize_node_text(self.spacy_nlp, text)
        lemmas = tuple(get_concept_lemmas(self.spacy_nlp, canon))
        for node in candidates:
            if lemmas == tuple(node.lemmas):
                return node
        return None

    def _appears_in_story(self, text: str, *, check_graph: bool = False) -> bool:
        """
        Check if text has grounding in story or graph.

        Per AMoC v4 paper alignment:
        - Literal grounding: lemma appears in story_lemmas (strict)
        - Graph grounding: node already exists in cumulative graph (for inference chains)

        Args:
            text: The text to check
            check_graph: If True, also allow concepts that exist in the graph.
                        This enables semantic inference chains like:
                        knight → danger → threat (where danger bridges)

        Returns:
            True if the text is grounded (literal or graph-based), False otherwise
        """
        if not text:
            return False
        doc = self.spacy_nlp(text)

        # Check literal lemma presence in story
        for tok in doc:
            if tok.is_alpha and tok.lemma_.lower() in self.story_lemmas:
                return True

        # Check graph membership (allows inference chains)
        if check_graph:
            lemmas = get_concept_lemmas(self.spacy_nlp, text)
            if self.graph.get_node(lemmas) is not None:
                return True

        return False

    def _classify_canonical_node_text(self, canon: str) -> Optional[NodeType]:
        if not canon:
            return None
        doc = self.spacy_nlp(canon)
        if not doc:
            return None
        token = next((t for t in doc if getattr(t, "is_alpha", False)), None) or doc[0]
        lemma = (getattr(token, "lemma_", "") or "").lower()
        if lemma in self.spacy_nlp.Defaults.stop_words:
            return None
        if token.pos_ in {"NOUN", "PROPN"}:
            return NodeType.CONCEPT
        if token.pos_ == "ADJ":
            return NodeType.PROPERTY
        return None

    def _canonicalize_and_classify_node_text(
        self, text: str
    ) -> tuple[str, Optional[NodeType]]:
        canon = canonicalize_node_text(self.spacy_nlp, text)
        return canon, self._classify_canonical_node_text(canon)

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
        active_triplets_for_overlay: Optional[
            List[Tuple[str, str, str]]
        ] = None,  # TASK 2: Triplet overlay
        property_nodes: Optional[
            List[str]
        ] = None,  # AMoC-v4 Figure 7: PROPERTY nodes are blue
    ) -> None:
        # ==========================================================================
        # DEFENSIVE GUARD: Ensure sentence_text never contains prompt scaffolding
        # ==========================================================================
        # This assertion catches the bug where LLM prompt text leaks into
        # the sentence stream. If this fires, the bug is in the sentence
        # processing pipeline (likely resolve_pronouns or its cleanup).
        sentence_text_lower = (sentence_text or "").lower().strip()
        prompt_contamination_patterns = [
            "the text is:",
            "here is the text:",
            "the sentence is:",
            "replace the pronouns",
        ]
        for pattern in prompt_contamination_patterns:
            if sentence_text_lower.startswith(pattern):
                logging.error(
                    "[BUG] Prompt scaffolding leaked into sentence_text: %s",
                    sentence_text[:100],
                )
                # Strip the contamination as a fallback (but the bug should be fixed upstream)
                sentence_text = sentence_text[len(pattern) :].strip()
                break

        # Route per-sentence plots into mode-specific subfolders for clarity.
        plot_dir = output_dir
        if output_dir and mode in {"sentence_active", "sentence_cumulative"}:
            subdir = "active" if mode == "sentence_active" else "cumulative"
            plot_dir = os.path.join(output_dir, subdir)
        triplets = (
            triplets_override
            if triplets_override is not None
            else self._graph_edges_to_triplets(only_active=only_active)
        )
        age_for_filename = self.persona_age if self.persona_age is not None else -1
        try:
            # --- FIX: ensure every node that may be plotted has a concrete position ---
            nodes_to_plot = set()

            for u, _, v in triplets:
                if u:
                    nodes_to_plot.add(u)
                if v:
                    nodes_to_plot.add(v)

            for lst in (inactive_nodes, explicit_nodes, salient_nodes, highlight_nodes):
                if lst:
                    nodes_to_plot.update(lst)

            # Assign positions for any missing nodes
            if nodes_to_plot:
                # Build a temporary graph only for layout
                G_tmp = nx.Graph()
                G_tmp.add_nodes_from(nodes_to_plot)

                # Use existing positions as fixed anchors
                fixed = {
                    n: self._viz_positions[n]
                    for n in nodes_to_plot
                    if n in self._viz_positions
                }

                new_pos = nx.spring_layout(
                    G_tmp,
                    pos=fixed if fixed else None,
                    fixed=fixed.keys() if fixed else None,
                    seed=42,
                )

                # Merge back into persistent positions
                self._viz_positions.update(new_pos)
            # ------------------------------------------------------------------------

            # PROVENANCE SANITY CHECK: Detect persona leakage before plotting
            provenance_warnings = self.graph.sanity_check_provenance(
                story_lemmas=self.story_lemmas,
                persona_only_lemmas=self._persona_only_lemmas,
            )
            for warning in provenance_warnings:
                logging.warning(warning)

            # ==========================================================================
            # AMoC-v4 FIGURE 7 COMPLIANCE: PROPERTY nodes are blue
            # ==========================================================================
            # Per AMoC-v4 Figure 7: Adjective PROPERTY nodes (young, beautiful, scorched)
            # MUST appear as blue nodes. Combine property_nodes with any highlight_nodes.
            # ==========================================================================
            blue_nodes_combined = set()
            if highlight_nodes:
                blue_nodes_combined.update(highlight_nodes)
            if property_nodes:
                blue_nodes_combined.update(property_nodes)

            saved_path = plot_amoc_triplets(
                triplets=triplets,
                persona=self.persona,
                model_name=self.model_name,
                age=age_for_filename,
                blue_nodes=list(blue_nodes_combined) if blue_nodes_combined else None,
                output_dir=plot_dir,
                step_tag=(
                    f"sent{sentence_index+1}_{mode}"
                    if mode
                    else f"sent{sentence_index+1}"
                ),
                sentence_text=sentence_text,
                inactive_nodes=inactive_nodes,
                explicit_nodes=explicit_nodes,
                salient_nodes=salient_nodes,
                largest_component_only=largest_component_only,
                positions=self._viz_positions,
                active_edges=active_edges,
                # LAYOUT POLICY: Pass activation scores for edge thickness/alpha
                edge_activation_scores=self._get_edge_activation_scores(),
                layout_from_active_only=True,
                # PAPER-ALIGNED: Pass single-edge policy to plotting
                allow_multi_edges=self.allow_multi_edges,
                # TASK 2: Pass active triplets for overlay
                # Triplets originate from _graph_edges_to_triplets() or triplets_override
                active_triplets_for_overlay=active_triplets_for_overlay,
                show_triplet_overlay=True,
            )
            if triplets:
                logging.info(
                    "[Plot] Saved sentence %d graph to %s", sentence_index, saved_path
                )
            else:
                logging.info(
                    "[Plot] Sentence %d graph skipped (no active edges)", sentence_index
                )
        except Exception:
            logging.error("Failed to plot graph snapshot", exc_info=True)

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
        # Log story text and sentences for debugging
        logging.info(
            "[AMoC] Story text (first 200 chars): %s",
            self.story_text[:200] if self.story_text else "NONE",
        )
        doc = self.spacy_nlp(self.story_text)
        sentences = list(doc.sents)
        logging.info("[AMoC] Number of sentences detected by spaCy: %d", len(sentences))
        for i, sent in enumerate(sentences):
            logging.info("[AMoC] Sentence %d: %s", i + 1, sent.text.strip()[:100])

        if not hasattr(self, "_amoc_matrix_records"):
            self._amoc_matrix_records = []
        # Initialize persistent visualization positions ONCE per analyze run.
        # These positions must remain stable across sentences.
        if not hasattr(self, "_viz_positions") or self._viz_positions is None:
            self._viz_positions = {}
        self._cumulative_deactivated_nodes_for_plot = set()
        self._prev_active_nodes_for_plot = set()

        def _clean_resolved_sentence(orig_text: str, candidate: str) -> str:
            if not isinstance(candidate, str) or not candidate.strip():
                return orig_text
            cleaned = re.sub(r"<[^>]+>", " ", candidate)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            # ==========================================================================
            # BUG FIX: Strip prompt scaffolding that LLM may echo back
            # ==========================================================================
            # The REPLACE_PRONOUNS_PROMPT ends with "The text is:\n" and the LLM
            # sometimes echoes this prefix in its response. This contamination must
            # be stripped to prevent it from leaking into sentence titles.
            prompt_scaffolding_patterns = [
                r"(?i)^the text is:\s*",  # "The text is: ..."
                r"(?i)^here is the text:\s*",  # "Here is the text: ..."
                r"(?i)^the sentence is:\s*",  # "The sentence is: ..."
                r"(?i)^text:\s*",  # "Text: ..."
                r"(?i)^sentence:\s*",  # "Sentence: ..."
                r"(?i)^replace the pronouns.*?:\s*",  # Echo of prompt instruction
            ]
            for pattern in prompt_scaffolding_patterns:
                cleaned = re.sub(pattern, "", cleaned).strip()

            # Filter out common LLM meta-responses about pronoun resolution
            llm_meta_patterns = [
                r"(?i)the text does not contain pronouns?",
                r"(?i)there are no pronouns? to replace",
                r"(?i)no pronouns? (?:to replace|found|present|in (?:the|this) (?:text|sentence))",
                r"(?i)does not (?:have|contain) (?:any )?pronouns?",
                r"(?i)without (?:any )?pronouns?",
                r"(?i)pronouns? (?:have been |are |were )?(?:already )?replaced",
            ]
            for pattern in llm_meta_patterns:
                if re.search(pattern, cleaned):
                    # LLM returned a meta-response, use original text
                    return orig_text

            # Pick the candidate sentence with the highest content-word overlap
            # with the original to avoid prompt echoes in the header.
            orig_doc = self.spacy_nlp(orig_text)
            orig_tokens = {t.lemma_.lower() for t in orig_doc if t.is_alpha}
            best_sent = None
            best_overlap = -1
            cand_doc = self.spacy_nlp(cleaned)
            for sent in cand_doc.sents:
                toks = {t.lemma_.lower() for t in sent if t.is_alpha}
                overlap = len(orig_tokens & toks)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sent = sent.text.strip()

            chosen = best_sent or cleaned

            # Trim runaway echoes to a reasonable length.
            max_len = max(len(orig_text) * 2 + 40, 400)
            if len(chosen) > max_len:
                chosen = chosen[:max_len].rstrip(" ,.;") + "..."
            return chosen or orig_text

        doc = self.spacy_nlp(self.story_text)
        resolved_sentences: list[tuple[Span, str, str]] = []
        for orig_sent in doc.sents:
            resolved_text = orig_sent.text
            if replace_pronouns:
                candidate = self.resolve_pronouns(orig_sent.text)
                if isinstance(candidate, str) and candidate.strip():
                    resolved_text = _clean_resolved_sentence(orig_sent.text, candidate)
            resolved_doc = self.spacy_nlp(resolved_text)
            if not resolved_doc:
                resolved_text = orig_sent.text
                resolved_doc = self.spacy_nlp(resolved_text)
            resolved_span = resolved_doc[0 : len(resolved_doc)]
            resolved_sentences.append((resolved_span, resolved_text, orig_sent.text))

        prev_sentences: list[str] = []
        current_sentence = ""
        self._sentence_triplets: list[
            tuple[int, str, str, str, str, bool, bool, int]
        ] = (
            []
        )  # sentence_idx, sentence_text, subj, rel, obj, active, anchor_kept, introduced_at
        for i, (sent, resolved_text, original_text) in enumerate(resolved_sentences):
            # Working-memory projection is rebuilt each sentence.
            self.active_graph = nx.MultiDiGraph()
            self._current_sentence_index = i + 1
            self._current_sentence_text = original_text
            self._anchor_drop_log: list[tuple[int, str, str, str, str]] = (
                []
            )  # sent_idx, sent_text, subj, rel, obj
            nodes_before_sentence = set(self.graph.nodes)
            self._explicit_nodes_current_sentence = set()
            logging.info("Processing sentence %d: %s", i, resolved_text)
            if i == 0:
                current_sentence = sent
                prev_sentences.append(resolved_text)
                self.init_graph(sent)

                # phrase concepts - old code
                phrase_nodes = self.get_phrase_level_concepts(sent)

                (
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                ) = self.get_senteces_text_based_nodes(
                    [sent], create_unexistent_nodes=True
                )

                # union them as explicit nodes
                # CRITICAL FIX: Explicit nodes are ALL nodes from current sentence text
                # ==========================================================================
                # AMoC-v4 FIGURE 7 COMPLIANCE: PROPERTY nodes are explicit in origin sentence
                # ==========================================================================
                # Per AMoC-v4 paper Figure 7: adjectives (PROPERTY nodes) like "young",
                # "beautiful", "scorched" MUST be explicit nodes in their origin sentence.
                # They appear as blue nodes connected via "is" edges.
                #
                # PROPERTY nodes are sentence-explicit, just like CONCEPT nodes.
                # They are NOT latent or inferred - they appear directly in the text.
                # ==========================================================================
                self._explicit_nodes_current_sentence = {
                    n
                    for n in (
                        set(phrase_nodes) | set(current_sentence_text_based_nodes)
                    )
                    # INCLUDE both CONCEPT and PROPERTY nodes - per Figure 7 compliance
                }

                # ==========================================================================
                # PAPER-FAITHFUL EXTRACTION: Deterministic linguistic grounding for sentence 0
                # Per AMoC paper Figures 4-6: Extract adjectives and prepositional objects
                # BEFORE LLM enrichment. This ensures PROPERTY nodes get "is" edges.
                # ==========================================================================
                self._extract_deterministic_structure(
                    sent,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                )

                # Populate _anchor_nodes from first sentence's explicit nodes
                # Per AMoC paper: anchors are CONCEPT nodes only (not PROPERTY)
                # SETTING nodes (locations/environments) should NOT become anchors
                self._anchor_nodes = {
                    n
                    for n in current_sentence_text_based_nodes
                    if n.node_type == NodeType.CONCEPT and not n.is_setting()
                }
                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships_step_0(sent)
                )

                self.add_inferred_relationships_to_graph_step_0(
                    inferred_concept_relationships, NodeType.CONCEPT, sent
                )
                self.add_inferred_relationships_to_graph_step_0(
                    inferred_property_relationships, NodeType.PROPERTY, sent
                )
                # Enforce connectivity for first sentence - remove any disconnected edges
                self._enforce_graph_connectivity()
                self._restrict_active_to_current_explicit(
                    list(self._explicit_nodes_current_sentence)
                )
                # self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                #     current_sentence_text_based_nodes
                # )
            else:
                added_edges = []
                current_sentence = sent
                prev_sentences.append(resolved_text)
                if len(prev_sentences) > self.context_length:
                    prev_sentences.pop(0)

                # NOTE: Per AMoC v4 paper (old_code.py ground truth):
                # Edges are NOT deactivated at sentence start.
                # Instead, they remain active until faded via reduce_visibility() in STEP 7.
                # The gradual decay mechanism is: non-relevant edges call reduce_visibility()
                # which decrements visibility_score. When visibility_score reaches 0, edge becomes inactive.
                logging.debug(
                    "[Activation] Sentence %d: edges carry forward, will fade if not relevant",
                    i,
                )

                # phrase level nodes
                phrase_nodes = self.get_phrase_level_concepts(current_sentence)

                current_sentence_text_based_nodes, current_sentence_text_based_words = (
                    self.get_senteces_text_based_nodes(
                        [current_sentence], create_unexistent_nodes=True
                    )
                )

                # ==========================================================================
                # AMoC-v4 FIGURE 7 COMPLIANCE: PROPERTY nodes are explicit in origin sentence
                # ==========================================================================
                # CRITICAL FIX: Set explicit nodes BEFORE deterministic extraction
                # so that _add_edge() attachment check can find current sentence nodes.
                #
                # Per AMoC-v4 Figure 7: Both CONCEPT and PROPERTY nodes from the current
                # sentence are explicit. Adjectives like "young", "beautiful", "scorched"
                # must appear as explicit PROPERTY nodes in their origin sentence.
                # ==========================================================================
                self._explicit_nodes_current_sentence = {
                    n
                    for n in (
                        set(phrase_nodes) | set(current_sentence_text_based_nodes)
                    )
                    # INCLUDE both CONCEPT and PROPERTY nodes - per Figure 7 compliance
                }

                # ==========================================================================
                # PAPER-FAITHFUL EXTRACTION: Deterministic linguistic grounding
                # Per AMoC paper Figures 4-6: Extract adjectives and prepositional objects
                # BEFORE LLM enrichment. Must be called AFTER _explicit_nodes_current_sentence
                # is set, so _add_edge() attachment check works.
                # ==========================================================================
                self._extract_deterministic_structure(
                    current_sentence,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                )

                current_all_text = resolved_text
                # Step 3: build active subgraph using only explicit (text-based) nodes.
                graph_active_nodes = self.graph.get_active_nodes(
                    self.max_distance_from_active_nodes, only_text_based=True
                )
                active_nodes_text = self.graph.get_nodes_str(graph_active_nodes)
                active_nodes_edges_text, _ = self.graph.get_edges_str(
                    graph_active_nodes, only_text_based=True
                )

                adjectival_properties = self._extract_adjectival_modifiers(sent)

                nodes_from_text = ""
                for idx, node in enumerate(current_sentence_text_based_nodes):
                    nodes_from_text += f" - ({current_sentence_text_based_words[idx]}, {node.node_type})\n"

                nodes_from_text = self._append_adjectival_hints(nodes_from_text, sent)

                new_relationships = self.client.get_new_relationships(
                    nodes_from_text,  # 1. Nodes from Text
                    active_nodes_text,  # 2. Nodes from Graph (explicit only)
                    active_nodes_edges_text,  # 3. Edges from Graph (explicit only)
                    current_all_text,  # 4. Text
                    self.persona,  # 5. Persona
                )

                text_based_activated_nodes = current_sentence_text_based_nodes
                sentence_lemma_keys = {
                    tuple(n.lemmas) for n in current_sentence_text_based_nodes
                }
                for idx, relationship in enumerate(new_relationships):
                    # Skip None or scalar junk (int, float, bool, etc.)
                    if relationship is None or isinstance(
                        relationship, (int, float, bool)
                    ):
                        logging.error(
                            f"[AMoC] Skipping non-iterable relationship at {idx}: {relationship!r}"
                        )
                        continue

                    # If relationship is a dict, try to convert it
                    if isinstance(relationship, dict):
                        subj = relationship.get("subject") or relationship.get("head")
                        rel = relationship.get("relation") or relationship.get(
                            "predicate"
                        )
                        obj = relationship.get("object") or relationship.get("tail")
                        if not (subj and rel and obj):
                            logging.error(
                                f"[AMoC] Skipping malformed dict relationship at {idx}: {relationship!r}"
                            )
                            continue
                        relationship = (str(subj), str(rel), str(obj))

                    # Must be list/tuple from this point on
                    if not isinstance(relationship, (list, tuple)):
                        logging.error(
                            f"[AMoC] Skipping unexpected relationship type at {idx}: {type(relationship)} → {relationship!r}"
                        )
                        continue

                    # Must have exactly 3 elements
                    if len(relationship) != 3:
                        logging.error(
                            f"[AMoC] Skipping relationship with wrong length at {idx}: {relationship!r}"
                        )
                        continue

                    # Unpack
                    subj, rel, obj = relationship

                    subj = self._normalize_endpoint_text(subj, is_subject=True) or None
                    obj = self._normalize_endpoint_text(obj, is_subject=False) or None
                    if subj is None or obj is None:
                        continue
                    # Validate subject/object strings
                    if not subj or not obj:
                        continue
                    if subj == obj:
                        continue
                    if not isinstance(subj, str) or not isinstance(obj, str):
                        continue

                    if not self._passes_attachment_constraint(
                        subj,
                        obj,
                        current_sentence_text_based_words,
                        current_sentence_text_based_nodes,
                        graph_active_nodes,
                        self._get_nodes_with_active_edges(),
                    ):
                        continue

                    # Continue with your original code
                    source_node = self.get_node_from_new_relationship(
                        subj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )

                    dest_node = self.get_node_from_new_relationship(
                        obj,
                        graph_active_nodes,
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        node_source=NodeSource.TEXT_BASED,
                        create_node=True,
                    )
                    edge_label = rel.replace("(edge)", "").strip()
                    edge_label = self._normalize_edge_label(edge_label)
                    if not self._is_valid_relation_label(edge_label):
                        continue
                    if source_node is None or dest_node is None:
                        continue

                    # DIRECTION CANONICALIZATION: Normalize passive voice to active
                    # Per AMoC v4: edges are directed semantic propositions (subject → object)
                    canon_label, canon_src, canon_dst, was_swapped = (
                        self._canonicalize_edge_direction(
                            edge_label,
                            source_node.get_text_representer(),
                            dest_node.get_text_representer(),
                        )
                    )
                    if was_swapped:
                        # Swap nodes to reflect active voice direction
                        source_node, dest_node = dest_node, source_node
                        edge_label = canon_label
                        logging.debug(
                            "[EdgeDirection] Swapped direction: %s → %s (label: %s)",
                            canon_src,
                            canon_dst,
                            edge_label,
                        )

                    if tuple(source_node.lemmas) in sentence_lemma_keys:
                        source_node.node_source = NodeSource.TEXT_BASED
                    if tuple(dest_node.lemmas) in sentence_lemma_keys:
                        dest_node.node_source = NodeSource.TEXT_BASED

                    # AMoCv4 surface-relation format: direct edge between entities
                    # ⟨entity, verb, entity⟩ - NO intermediate RELATION nodes
                    # Direction: source (subject/agent) → dest (object/patient)
                    edge = self._add_edge(
                        source_node,
                        dest_node,
                        edge_label,
                        self.edge_visibility,
                    )

                    # NOTE: Removed node_source filtering - explicit nodes are determined
                    # by presence in sentence text, not by how they were introduced

                # infer new relationships logic...
                inferred_concept_relationships, inferred_property_relationships = (
                    self.infer_new_relationships(
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

                self.add_inferred_relationships_to_graph(
                    inferred_concept_relationships,
                    NodeType.CONCEPT,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )
                self.add_inferred_relationships_to_graph(
                    inferred_property_relationships,
                    NodeType.PROPERTY,
                    current_sentence_text_based_nodes,
                    current_sentence_text_based_words,
                    graph_active_nodes,
                    added_edges,
                )

                if self.ENFORCE_ATTACHMENT_CONSTRAINT:
                    targeted_edges = self._infer_edges_to_recently_deactivated(
                        current_sentence_text_based_nodes,
                        current_sentence_text_based_words,
                        current_all_text,
                    )
                    added_edges.extend(targeted_edges)

                # self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                #     text_based_activated_nodes
                # )

                # ACTIVATION LOGIC: Reactivate memory edges within MAX_DISTANCE of explicit nodes
                # Property/attribute edges only reactivate in their origin sentence
                reactivated_edges = self.graph.reactivate_memory_edges_within_distance(
                    explicit_nodes=self._explicit_nodes_current_sentence,
                    max_distance=self.max_distance_from_active_nodes,
                    current_sentence=self._current_sentence_index,
                )
                logging.debug(
                    "[Activation] Reactivated %d memory edges within distance %d",
                    len(reactivated_edges),
                    self.max_distance_from_active_nodes,
                )

                self.reactivate_relevant_edges(
                    self.graph.get_active_nodes(
                        self.max_distance_from_active_nodes, only_text_based=True
                    ),
                    " ".join(prev_sentences),
                    added_edges,
                )

                # Update anchor nodes to include current explicit nodes and
                # nodes with active edges to maintain connectivity across sentences
                # Per AMoC paper: anchors are CONCEPT nodes only (not PROPERTY)
                # SETTING nodes (locations/environments) should NOT become anchors
                self._anchor_nodes = (
                    self._anchor_nodes
                    | {
                        n
                        for n in current_sentence_text_based_nodes
                        if n.node_type == NodeType.CONCEPT and not n.is_setting()
                    }
                    | {
                        n
                        for n in self._get_nodes_with_active_edges()
                        if n.node_type == NodeType.CONCEPT and not n.is_setting()
                    }
                )

                # ACTIVE CONNECTIVITY PRESERVATION (per AMoC v4 paper)
                # The active graph must remain connected at all times.
                # If disconnected, promote minimum memory edges as connectors.
                # Connector edges preserve structure but don't count as asserted/reactivated.
                connector_edges = self.graph.ensure_active_connectivity(
                    focus_nodes=self._explicit_nodes_current_sentence,
                    carryover_focus_nodes=self._anchor_nodes,
                )
                if connector_edges:
                    logging.debug(
                        "[Connectivity] Promoted %d edges as connectors to preserve connectivity",
                        len(connector_edges),
                    )

                # ==========================================================================
                # TASK 2: SECONDARY LLM CALL FOR FORCED CONNECTIVITY
                # ==========================================================================
                # If ensure_active_connectivity() couldn't fully connect the graph
                # (no existing edges could bridge components), trigger secondary LLM call
                # to create minimal forced connectivity edges.
                if not self.graph.check_active_connectivity():
                    forced_edges = self._create_forced_connectivity_edges(
                        story_context=(
                            " ".join(prev_sentences[:-1])
                            if len(prev_sentences) > 1
                            else ""
                        ),
                        current_sentence=resolved_text,
                    )
                    if forced_edges:
                        logging.info(
                            "[Connectivity] Created %d forced connectivity edges via secondary LLM call",
                            len(forced_edges),
                        )
                        added_edges.extend(forced_edges)

                # ACTIVATION LOGIC: Decay activation_score for inactive edges
                self.graph.decay_inactive_edges()
                logging.debug(
                    "[Activation] Decayed activation scores for inactive edges"
                )
                self._restrict_active_to_current_explicit(
                    list(self._explicit_nodes_current_sentence)
                )
                # self.graph.set_nodes_score_based_on_distance_from_active_nodes(
                #     current_sentence_text_based_nodes
                # )

            if self.debug:
                logging.info(
                    "Active graph after sentence %d:\n%s",
                    i,
                    self.graph.get_active_graph_repr(),
                )

            sentence_id = i + 1
            newly_inferred_nodes = {
                n
                for n in (set(self.graph.nodes) - nodes_before_sentence)
                if n.node_source == NodeSource.INFERENCE_BASED
            }
            self._explicit_nodes_current_sentence |= newly_inferred_nodes

            # BUILD PER-SENTENCE VIEW (only when strict mode is enabled)
            # When enabled, enforces:
            # - Only explicit + carry-over nodes are visible
            # - Inactive nodes are completely excluded
            # - Only edges where BOTH endpoints are active are included
            # - The graph is guaranteed connected (or empty)
            per_sentence_view = self._build_per_sentence_view(
                explicit_nodes=list(self._explicit_nodes_current_sentence),
                sentence_index=sentence_id,
            )

            # Log per-sentence view invariants for debugging (only in strict mode)
            if self.debug and per_sentence_view is not None:
                logging.info(
                    "Per-sentence view for sentence %d: "
                    "%d explicit, %d carry-over, %d active edges, connected=%s",
                    sentence_id,
                    len(per_sentence_view.explicit_nodes),
                    len(per_sentence_view.carryover_nodes),
                    len(per_sentence_view.active_edges),
                    per_sentence_view.is_connected,
                )

            # Refresh active projection for this step
            self._record_sentence_activation(
                sentence_id=sentence_id,
                explicit_nodes=list(self._explicit_nodes_current_sentence),
                newly_inferred_nodes=newly_inferred_nodes,
            )

            current_active_nodes = self._get_nodes_with_active_edges()

            # Connectivity check using per-sentence view (only in strict mode)
            if (
                per_sentence_view is not None
                and not per_sentence_view.is_empty
                and not per_sentence_view.is_connected
            ):
                logging.error(
                    "Per-sentence graph disconnected at sentence %s for persona '%s' "
                    "(this should not happen with strict-attachment-constraint enabled)",
                    sentence_id,
                    self.persona,
                )
            # Legacy active_graph check (for backwards compatibility)
            if self.active_graph.number_of_nodes() > 1 and not nx.is_connected(
                nx.Graph(self.active_graph)
            ):
                logging.error(
                    "Legacy active_graph disconnected at sentence %s (per-sentence view governs correctness)",
                    sentence_id,
                )
            if i == 0:
                recently_deactivated_nodes: set[Node] = set()
            else:
                appeared = current_active_nodes - self._prev_active_nodes_for_plot
                gone = self._prev_active_nodes_for_plot - current_active_nodes
                self._cumulative_deactivated_nodes_for_plot.update(gone)
                self._cumulative_deactivated_nodes_for_plot.difference_update(
                    current_active_nodes
                )
                recently_deactivated_nodes = set(gone)
            # Use per-sentence view for node categorization (constraint-first design)
            # This ensures inactive nodes are properly excluded from all outputs
            if self._per_sentence_view is not None:
                # Explicit nodes from per-sentence view (guaranteed present)
                explicit_nodes_for_plot = sorted(
                    filter(
                        None,
                        {
                            node.get_text_representer()
                            for node in self._per_sentence_view.explicit_nodes
                        },
                    )
                )
                # Carry-over nodes (salient) from per-sentence view
                salient_nodes_for_plot = sorted(
                    filter(
                        None,
                        {
                            node.get_text_representer()
                            for node in self._per_sentence_view.carryover_nodes
                        },
                    )
                )
                # Inactive nodes: all cumulative nodes NOT in the per-sentence active set
                all_cumulative_node_texts = {
                    node.get_text_representer()
                    for node in self.graph.nodes
                    if node.get_text_representer()
                }
                active_node_texts = set(explicit_nodes_for_plot) | set(
                    salient_nodes_for_plot
                )
                inactive_nodes_for_plot = sorted(
                    all_cumulative_node_texts - active_node_texts
                )
            else:
                # Fallback to legacy behavior if no per-sentence view
                inactive_nodes_for_plot = sorted(
                    filter(
                        None,
                        {
                            node.get_text_representer()
                            for node in self._cumulative_deactivated_nodes_for_plot
                        },
                    )
                )
                explicit_nodes_for_plot = sorted(
                    filter(
                        None,
                        {
                            node.get_text_representer()
                            for node in current_sentence_text_based_nodes
                        },
                    )
                )
                salient_nodes_for_plot = sorted(
                    {
                        node.get_text_representer()
                        for node in current_active_nodes
                        if node.get_text_representer()
                    }
                    - set(explicit_nodes_for_plot)
                )
            # Only keep the deactivated set for targeted inference when the
            # attachment constraint is enforced.
            if self.ENFORCE_ATTACHMENT_CONSTRAINT:
                self._recently_deactivated_nodes_for_inference = (
                    recently_deactivated_nodes
                )
            else:
                self._recently_deactivated_nodes_for_inference = set()
            self._prev_active_nodes_for_plot = current_active_nodes

            if plot_after_each_sentence:
                # Active (salience) view - use per-sentence view for clean isolation
                # This guarantees only edges with BOTH endpoints active are shown
                active_nodes = (
                    set(self._per_sentence_view.explicit_nodes)
                    | set(self._per_sentence_view.carryover_nodes)
                    if self._per_sentence_view is not None
                    else None
                )

                active_triplets = self._reconstruct_semantic_triplets(
                    only_active=True,
                    restrict_nodes=active_nodes,
                )

                # ==========================================================================
                # AMoC-v4 FIGURE 7 COMPLIANCE: Collect PROPERTY nodes for blue coloring
                # ==========================================================================
                # Per AMoC-v4 Figure 7: Adjectives (young, beautiful, scorched) MUST appear
                # as blue PROPERTY nodes in their origin sentence. Extract PROPERTY node
                # texts from _explicit_nodes_current_sentence to pass as blue_nodes.
                # ==========================================================================
                property_nodes_for_plot = sorted(
                    filter(
                        None,
                        {
                            node.get_text_representer()
                            for node in self._explicit_nodes_current_sentence
                            if node.node_type == NodeType.PROPERTY
                        },
                    )
                )

                # AMoCv4 surface-relation format: edges ARE the semantic triplets
                # Just collect active edge pairs directly
                active_edge_pairs = set()
                for edge in self.graph.edges:
                    if not edge.active:
                        continue
                    active_edge_pairs.add(
                        (
                            edge.source_node.get_text_representer(),
                            edge.dest_node.get_text_representer(),
                        )
                    )

                # ==========================================================================
                # BUG FIX: Use original_text for plot titles, NOT sent.text
                # ==========================================================================
                # `sent` is a spaCy Span created from `resolved_text` which may contain
                # contamination from LLM prompt echoes (e.g., "The text is: ...").
                # We must use `original_text` (the clean, unprocessed sentence) for
                # all display purposes like plot titles.
                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=original_text,  # FIX: was sent.text (contaminated)
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    inactive_nodes=inactive_nodes_for_plot,
                    explicit_nodes=explicit_nodes_for_plot,
                    salient_nodes=salient_nodes_for_plot,
                    only_active=True,
                    largest_component_only=largest_component_only,
                    mode="sentence_active",
                    triplets_override=active_triplets,
                    active_edges=active_edge_pairs,
                    # TASK 2: Pass active triplets for overlay display
                    active_triplets_for_overlay=active_triplets,
                    # AMoC-v4 Figure 7: PROPERTY nodes are blue
                    property_nodes=property_nodes_for_plot,
                )
                # Cumulative memory view
                # AMoCv4 surface-relation format: edges ARE the semantic triplets
                cumulative_active_pairs = set()
                for edge in self.graph.edges:
                    if not edge.active:
                        continue
                    cumulative_active_pairs.add(
                        (
                            edge.source_node.get_text_representer(),
                            edge.dest_node.get_text_representer(),
                        )
                    )

                self._plot_graph_snapshot(
                    sentence_index=i,
                    sentence_text=original_text,  # FIX: was sent.text (contaminated)
                    output_dir=graphs_output_dir,
                    highlight_nodes=highlight_nodes,
                    inactive_nodes=inactive_nodes_for_plot,
                    explicit_nodes=explicit_nodes_for_plot,
                    salient_nodes=salient_nodes_for_plot,
                    only_active=False,
                    largest_component_only=largest_component_only,
                    mode="sentence_cumulative",
                    triplets_override=self._reconstruct_semantic_triplets(
                        only_active=False
                    ),
                    active_edges=cumulative_active_pairs,
                    # TASK 2: Pass active triplets for overlay display
                    # For cumulative view, still show only the current sentence's active triplets
                    active_triplets_for_overlay=active_triplets,
                    # AMoC-v4 Figure 7: PROPERTY nodes are blue
                    property_nodes=property_nodes_for_plot,
                )

            # Capture triplets for this sentence (all edges, with current active flag)
            # Capture semantic triplets for this sentence (relation-node reconstruction)
            current_nodes = (
                self._explicit_nodes_current_sentence
                | self._get_nodes_with_active_edges()
            )
            for subj, rel, obj in self._reconstruct_semantic_triplets(
                only_active=False, restrict_nodes=current_nodes
            ):
                self._sentence_triplets.append(
                    (
                        self._current_sentence_index,
                        original_text,
                        subj,
                        rel,
                        obj,
                        True,  # semantic triplet exists
                        True,  # anchor_kept (semantic-level; structural handled elsewhere)
                        self._triplet_intro.get((subj, rel, obj), -1),
                    )
                )
            for sent_idx, sent_text, subj, rel, obj in getattr(
                self, "_anchor_drop_log", []
            ):
                self._sentence_triplets.append(
                    (
                        sent_idx,
                        sent_text,
                        subj,
                        rel,
                        obj,
                        False,  # inactive/not added
                        False,  # anchor_kept flag
                        -1,  # introduced_at unknown/unused for dropped anchors
                    )
                )

        # save score matrix
        df = pd.DataFrame(self._amoc_matrix_records)
        # Collapse duplicate token/sentence entries by mean to avoid pivot errors.
        if not df.empty:
            df = (
                df.groupby(["token", "sentence"], as_index=False)["score"]
                .mean()
                .astype({"token": str})
            )
        matrix = (
            df.pivot(index="token", columns="sentence", values="score")
            .sort_index()
            .fillna(0.0)
        )
        # Order rows by salience: highest peak activation first, then total activation.
        salience_max = matrix.max(axis=1)
        salience_sum = matrix.sum(axis=1)
        ordering = (
            salience_max.to_frame("max")
            .assign(sum=salience_sum)
            .sort_values(by=["max", "sum", "token"], ascending=[False, False, True])
        )
        matrix = matrix.loc[ordering.index]
        # Prepend full story text as first row for traceability.
        if len(matrix.columns) > 0:
            story_row = pd.DataFrame(
                [{col: "" for col in matrix.columns}], index=["story_text"]
            )
            story_row.iloc[0, 0] = self.story_text
            matrix = pd.concat([story_row, matrix])

        matrix_dir = os.path.join(self.matrix_dir_base, "matrix")
        os.makedirs(matrix_dir, exist_ok=True)
        safe_model = _sanitize_filename_component(self.model_name, max_len=60)
        safe_persona = _sanitize_filename_component(self.persona, max_len=60)
        age_for_filename = self.persona_age if self.persona_age is not None else -1
        suffix = (
            f"_{_sanitize_filename_component(matrix_suffix)}" if matrix_suffix else ""
        )
        matrix_filename = (
            f"amoc_matrix_{safe_model}_{safe_persona}_{age_for_filename}{suffix}.csv"
        )
        matrix_path = os.path.join(matrix_dir, matrix_filename)

        matrix.to_csv(matrix_path)
        logging.info(
            "[Matrix] Saved activation matrix for persona '%s' to %s",
            self.persona,
            matrix_path,
        )
        logging.info("AMoC activation matrix:\n%s", matrix.to_string())
        # Collect final active triplets: edges active after the final sentence.
        final_sentence_idx = getattr(self, "_current_sentence_index", None)

        final_triplets = []
        current_nodes = (
            self._explicit_nodes_current_sentence | self._get_nodes_with_active_edges()
        )
        for subj, rel, obj in self._reconstruct_semantic_triplets(
            only_active=True, restrict_nodes=current_nodes
        ):
            intro = self._triplet_intro.get((subj, rel, obj), -1)
            final_triplets.append(
                (
                    subj,
                    rel,
                    obj,
                    True,
                    int(intro),
                    int(final_sentence_idx) if final_sentence_idx else -1,
                )
            )

        cumulative_triplets = []
        for subj, rel, obj in self._reconstruct_semantic_triplets():
            intro = self._triplet_intro.get((subj, rel, obj), -1)
            cumulative_triplets.append((subj, rel, obj, int(intro)))

        # AMoCv4 HARD CONSTRAINTS - Validate surface-relation format
        # Fail fast if any forbidden patterns exist
        self.graph.validate_amocv4_constraints()
        self.graph.sanity_check_readable_triplets()

        return final_triplets, self._sentence_triplets, cumulative_triplets

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )

        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        nodes_from_text = self._append_adjectival_hints(nodes_from_text, sent)

        for _ in range(3):
            try:
                object_properties_dict = (
                    self.client.infer_objects_and_properties_first_sentence(
                        nodes_from_text, sent.text, self.persona
                    )
                )
                break
            except:
                continue
        else:
            return [], []

        for _ in range(3):
            try:
                new_relationships = (
                    self.client.generate_new_inferred_relationships_first_sentence(
                        nodes_from_text,
                        object_properties_dict["concepts"][: self.max_new_concepts],
                        object_properties_dict["properties"][: self.max_new_properties],
                        sent.text,
                        self.persona,
                    )
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def infer_new_relationships(
        self,
        text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        graph_nodes_representation: str,
        graph_edges_representation: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        nodes_from_text = ""
        for i, node in enumerate(current_sentence_text_based_nodes):
            nodes_from_text += (
                f" - ({current_sentence_text_based_words[i]}, {node.node_type})\n"
            )

        doc = self.spacy_nlp(text)
        sent_span = doc[0 : len(doc)]
        nodes_from_text = self._append_adjectival_hints(nodes_from_text, sent_span)

        for _ in range(3):
            try:
                object_properties_dict = self.client.infer_objects_and_properties(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    text,
                    self.persona,
                )
                break
            except:
                continue

        for _ in range(3):
            try:
                new_relationships = self.client.generate_new_inferred_relationships(
                    nodes_from_text,
                    graph_nodes_representation,
                    graph_edges_representation,
                    object_properties_dict["concepts"][: self.max_new_concepts],
                    object_properties_dict["properties"][: self.max_new_properties],
                    text,
                    self.persona,
                )
                return (
                    new_relationships["concept_relationships"],
                    new_relationships["property_relationships"],
                )
            except:
                continue
        return [], []

    def reactivate_relevant_edges(
        self,
        active_nodes: List[Node],
        prev_sentences_text: str,
        newly_added_edges: List[Edge],
    ) -> None:
        edges_text, edges = self.graph.get_edges_str(
            self.graph.nodes, only_active=False
        )
        # Non-strict mode: accumulate salience monotonically (no fading/pruning).
        if not self.strict_reactivate_function:
            for edge in edges:
                # Use proper state management - mark as reactivated
                # PROPERTY edges should NOT be reactivated per paper rules
                if not edge.is_property_edge():
                    edge.active = True
                    self._record_edge_in_graphs(edge, self._current_sentence_index)
                else:
                    edge.mark_as_reactivated(reset_score=True)
                edge.visibility_score = self.edge_visibility
                self._record_edge_in_graphs(edge, self._current_sentence_index)
            # Enforce connectivity even in non-strict mode
            self._enforce_graph_connectivity()
            return

        raw_indices = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, self.persona
        )

        valid_indices: List[int] = []
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(edges):
                valid_indices.append(i)

        valid_indices = valid_indices[: self.nr_relevant_edges]
        active_node_set = set(active_nodes)
        if not valid_indices:
            # Improved fallback edge selection: keep edges that are newly added,
            # or active edges that touch at least one active/anchor node.
            # Uses only connected nodes to prevent selecting disconnected edges.
            selected = set()
            connected_nodes = active_node_set | self._anchor_nodes
            connected_lemma_keys = {tuple(n.lemmas) for n in connected_nodes}
            for idx, edge in enumerate(edges, start=1):
                if edge in newly_added_edges:
                    selected.add(idx)
                elif edge.active:
                    # Check both direct membership and lemma matching
                    source_connected = (
                        edge.source_node in connected_nodes
                        or tuple(edge.source_node.lemmas) in connected_lemma_keys
                    )
                    dest_connected = (
                        edge.dest_node in connected_nodes
                        or tuple(edge.dest_node.lemmas) in connected_lemma_keys
                    )
                    if source_connected or dest_connected:
                        selected.add(idx)
        else:
            selected = set(valid_indices)
            for i in selected:
                edge = edges[i - 1]
                edge.visibility_score = self.edge_visibility
                if not edge.is_property_edge():
                    continue
                self._record_edge_in_graphs(edge, self._current_sentence_index)

        # Preserve connectivity in the active projection.
        # If deactivating an edge would disconnect active nodes,
        # keep it active as a low-salience bridge (paper-consistent).
        def _active_subgraph_connected():
            G = nx.Graph()
            for e in edges:
                if e.active:
                    G.add_edge(e.source_node, e.dest_node)
            for n in active_nodes:
                G.add_node(n)
            return nx.is_connected(G) if G.number_of_nodes() > 1 else True

        # AMoC v4 STEP 7: Edge decay via reduce_visibility()
        # Per paper Section 3.1:
        # - Non-relevant edges fade away (visibility_score decrements)
        # - When visibility_score reaches 0, edge becomes inactive
        for j in range(1, len(edges) + 1):
            edge = edges[j - 1]
            if j not in selected and edges[j - 1] not in newly_added_edges:
                # Paper-faithful: use reduce_visibility() for gradual decay
                edge.reduce_visibility()

                if not _active_subgraph_connected():
                    # Keep as connectivity bridge - mark as CONNECTOR, not reactivated
                    # Connectors don't count as asserted/reactivated
                    edge.mark_as_connector()
                    edge.visibility_score = 0  # lowest salience
                    # Record bridge edges in active_graph to prevent disconnection in plots
                    self._record_edge_in_graphs(edge, self._current_sentence_index)
                else:
                    self._record_edge_in_graphs(edge, self._current_sentence_index)

        # Final connectivity sweep - ensure the entire graph remains connected
        self._enforce_graph_connectivity()

    # ==========================================================================
    # PAPER-FAITHFUL EXTRACTION: Deterministic linguistic extraction
    # Per AMoC paper Figures 4-6: Adjectives become PROPERTY nodes, prepositional
    # objects become CONCEPT nodes. All extraction is rule-based (no LLM).
    # ==========================================================================

    def _extract_deterministic_structure(
        self,
        sent: Span,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
    ) -> None:
        # 1. Extract adjectival modifiers → PROPERTY nodes
        adj_modifiers = extract_adjectival_modifiers(sent)
        for mod in adj_modifiers:
            adj_lemma = mod["adjective"]
            head_lemma = mod["head_noun"]

            # Find or skip if head noun is not already a CONCEPT node
            head_node = None
            for node in current_sentence_text_based_nodes:
                if head_lemma in node.lemmas:
                    head_node = node
                    break

            if head_node is None:
                # Head noun not in current sentence nodes - skip
                continue

            # Create PROPERTY node for the adjective
            if not self._admit_node(
                lemma=adj_lemma,
                node_type=NodeType.PROPERTY,
                provenance="SYNTACTIC_PROPERTY",
                sent=sent,
            ):
                continue

            property_node = self.graph.add_or_get_node(
                [adj_lemma],
                adj_lemma,
                NodeType.PROPERTY,
                NodeSource.TEXT_BASED,
                provenance=NodeProvenance.STORY_TEXT,
            )

            # Track that this is a text-based node from current sentence
            if property_node not in current_sentence_text_based_nodes:
                current_sentence_text_based_nodes.append(property_node)
                current_sentence_text_based_words.append(adj_lemma)

            # Create edge: concept --is--> property (through centralized path)
            edge = self._add_edge(
                head_node,
                property_node,
                "is",
                self.edge_visibility,
                created_at_sentence=self._current_sentence_index,
            )
            if edge is not None:
                logging.debug(
                    f"[Deterministic] Created PROPERTY edge: {head_lemma} --is--> {adj_lemma}"
                )

        # 2. Extract prepositional objects → CONCEPT nodes
        prep_objects = extract_prepositional_objects(sent)
        logging.debug(
            f"[Deterministic] Extracted {len(prep_objects)} prep objects from: {sent.text}"
        )
        for prep_obj in prep_objects:
            logging.debug(f"[Deterministic] Processing prep object: {prep_obj}")

        for prep_obj in prep_objects:
            subj_lemma = prep_obj["subject"]
            obj_lemma = prep_obj["object"]
            edge_label = prep_obj["label"]

            subj_node = None

            # First: check current sentence text-based nodes
            for node in current_sentence_text_based_nodes:
                if subj_lemma in node.lemmas:
                    subj_node = node
                    break

            # Second: check all existing graph nodes (for cross-sentence attachment)
            if subj_node is None:
                subj_node = self.graph.get_node([subj_lemma])

            if subj_node is None:
                # Subject not in graph at all - skip
                logging.debug(
                    f"[Deterministic] SKIP prep: subject '{subj_lemma}' not in graph"
                )
                continue

            # Create CONCEPT node for the prepositional object
            if not self._admit_node(
                lemma=obj_lemma,
                node_type=NodeType.CONCEPT,
                provenance="STORY_TEXT",
                sent=sent,
            ):
                logging.debug(
                    f"[Deterministic] SKIP prep: _admit_node rejected '{obj_lemma}'. "
                    f"In story_lemmas: {obj_lemma in self.story_lemmas}"
                )
                continue

            # Prepositional objects are SETTING nodes (locations/environments)
            obj_node = self.graph.add_or_get_node(
                [obj_lemma],
                obj_lemma,
                NodeType.CONCEPT,
                NodeSource.TEXT_BASED,
                provenance=NodeProvenance.STORY_TEXT,
                node_role=NodeRole.SETTING,
            )

            # Track that this is a text-based node from current sentence
            if obj_node not in current_sentence_text_based_nodes:
                current_sentence_text_based_nodes.append(obj_node)
                current_sentence_text_based_words.append(obj_lemma)

            # Normalize and create edge through centralized path
            original_label = edge_label
            norm_label = self._normalize_edge_label(edge_label)
            if not norm_label:
                logging.debug(
                    f"[Deterministic] Prep edge rejected, but node kept: "
                    f"{subj_lemma} --X--> {obj_lemma}"
                )
                continue
            edge = self._add_edge(
                subj_node,
                obj_node,
                norm_label,
                self.edge_visibility,
                created_at_sentence=self._current_sentence_index,
                bypass_attachment_constraint=True,
            )
            if edge is not None:
                logging.debug(
                    f"[Deterministic] Created prep edge: {subj_lemma} --{norm_label}--> {obj_lemma}"
                )

            else:
                logging.debug(
                    f"[Deterministic] SKIP prep: _add_edge returned None for "
                    f"{subj_lemma} --{edge_label}--> {obj_lemma}"
                )

    def init_graph(self, sent: Span) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=True)
        )

        # ==========================================================================
        # PAPER-FAITHFUL EXTRACTION: Deterministic linguistic grounding
        # Per AMoC paper Figures 4-6: Extract adjectives and prepositional objects
        # BEFORE LLM enrichment. This ensures "young" and "forest" appear in Sentence 1.
        # ==========================================================================
        self._extract_deterministic_structure(
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
        # print(f"First sentence edges:\n{relationships}")

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
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            if not self._passes_attachment_constraint(
                norm_subj,
                norm_obj,
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges(),
            ):
                continue
            source_node = self.get_node_from_text(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            dest_node = self.get_node_from_text(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.TEXT_BASED,
                create_node=True,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label(edge_label)
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None or dest_node is None:
                continue

            # DIRECTION CANONICALIZATION: Normalize passive voice to active
            canon_label, canon_src, canon_dst, was_swapped = (
                self._canonicalize_edge_direction(
                    edge_label,
                    source_node.get_text_representer(),
                    dest_node.get_text_representer(),
                )
            )
            if was_swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            # AMoCv4 surface-relation format: direct edge between entities
            # ⟨entity, verb, entity⟩ - NO intermediate RELATION nodes
            # Direction: source (subject/agent) → dest (object/patient)
            self._add_edge(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
    ) -> None:
        current_sentence_text_based_nodes, current_sentence_text_based_words = (
            self.get_senteces_text_based_nodes([sent], create_unexistent_nodes=False)
        )
        for relationship in inferred_relationships:
            # print(relationship)
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
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            # RELAXED: Allow semantic inferences where at least one endpoint is grounded
            # (literally in story text OR exists in graph from prior inference)
            # This enables abstract concepts like "danger", "threat", "goal" per paper Figures 2-4
            if not (
                self._appears_in_story(relationship[0], check_graph=True)
                or self._appears_in_story(relationship[2], check_graph=True)
            ):
                continue
            if not self._passes_attachment_constraint(
                relationship[0],
                relationship[2],
                current_sentence_text_based_words,
                current_sentence_text_based_nodes,
                list(self.graph.nodes),
                self._get_nodes_with_active_edges(),
            ):
                continue
            subj, subj_type = self._canonicalize_and_classify_node_text(relationship[0])
            obj, obj_type = self._canonicalize_and_classify_node_text(relationship[2])
            if subj_type is None or obj_type is None:
                continue
            source_node = self.get_node_from_text(
                norm_subj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_text(
                norm_obj,
                current_sentence_text_based_nodes,
                current_sentence_text_based_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label(edge_label)
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None:
                # PROVENANCE GATE: Validate LLM-inferred nodes against story text
                # BOOTSTRAP: allow if dest_node already exists (one endpoint grounded)
                if not self._validate_node_provenance(
                    subj, allow_bootstrap=(dest_node is not None)
                ):
                    continue
                if not self._admit_node(
                    lemma=subj,
                    node_type=subj_type,
                    provenance="INFERRED_RELATION",
                ):
                    continue

                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if dest_node is None:
                # PROVENANCE GATE: Validate LLM-inferred nodes against story text
                # BOOTSTRAP: allow if source_node already exists (one endpoint grounded)
                if not self._validate_node_provenance(
                    obj, allow_bootstrap=(source_node is not None)
                ):
                    continue
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            # DIRECTION CANONICALIZATION: Normalize passive voice to active
            canon_label, canon_src, canon_dst, was_swapped = (
                self._canonicalize_edge_direction(
                    edge_label,
                    source_node.get_text_representer(),
                    dest_node.get_text_representer(),
                )
            )
            if was_swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            # AMoCv4 surface-relation format: direct edge between entities
            # ⟨entity, verb, entity⟩ - NO intermediate RELATION nodes
            # Direction: source (subject/agent) → dest (object/patient)
            self._add_edge(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
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
        for relationship in inferred_relationships:
            # print(relationship)
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
            norm_subj = self._normalize_endpoint_text(relationship[0], is_subject=True)
            norm_obj = self._normalize_endpoint_text(relationship[2], is_subject=False)
            if norm_subj is None or norm_obj is None:
                continue
            # RELAXED: Allow semantic inferences where at least one endpoint is grounded
            # (literally in story text OR exists in graph from prior inference)
            # This enables abstract concepts like "danger", "threat", "goal" per paper Figures 2-4
            if not (
                self._appears_in_story(relationship[0], check_graph=True)
                or self._appears_in_story(relationship[2], check_graph=True)
            ):
                continue
            if not self._passes_attachment_constraint(
                relationship[0],
                relationship[2],
                curr_sentences_words,
                curr_sentences_nodes,
                active_graph_nodes,
                self._get_nodes_with_active_edges(),
            ):
                continue
            subj, subj_type = self._canonicalize_and_classify_node_text(relationship[0])
            obj, obj_type = self._canonicalize_and_classify_node_text(relationship[2])
            if subj_type is None or obj_type is None:
                continue
            source_node = self.get_node_from_new_relationship(
                norm_subj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            dest_node = self.get_node_from_new_relationship(
                norm_obj,
                active_graph_nodes,
                curr_sentences_nodes,
                curr_sentences_words,
                node_source=NodeSource.INFERENCE_BASED,
                create_node=False,
            )
            edge_label = relationship[1].replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label(edge_label)
            if not self._is_valid_relation_label(edge_label):
                continue
            if source_node is None:
                # PROVENANCE GATE: Validate LLM-inferred nodes against story text
                # BOOTSTRAP: allow if dest_node already exists (one endpoint grounded)
                if not self._validate_node_provenance(
                    subj, allow_bootstrap=(dest_node is not None)
                ):
                    continue
                source_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, subj),
                    subj,
                    subj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            if dest_node is None:
                # PROVENANCE GATE: Validate LLM-inferred nodes against story text
                # BOOTSTRAP: allow if source_node already exists (one endpoint grounded)
                if not self._validate_node_provenance(
                    obj, allow_bootstrap=(source_node is not None)
                ):
                    continue
                dest_node = self.graph.add_or_get_node(
                    get_concept_lemmas(self.spacy_nlp, obj),
                    obj,
                    obj_type,
                    NodeSource.INFERENCE_BASED,
                    provenance=NodeProvenance.INFERRED_FROM_STORY,
                )

            # DIRECTION CANONICALIZATION: Normalize passive voice to active
            canon_label, canon_src, canon_dst, was_swapped = (
                self._canonicalize_edge_direction(
                    edge_label,
                    source_node.get_text_representer(),
                    dest_node.get_text_representer(),
                )
            )
            if was_swapped:
                source_node, dest_node = dest_node, source_node
                edge_label = canon_label

            # AMoCv4 surface-relation format: direct edge between entities
            # ⟨entity, verb, entity⟩ - NO intermediate RELATION nodes
            # Direction: source (subject/agent) → dest (object/patient)
            potential_edge = self._add_edge(
                source_node,
                dest_node,
                edge_label,
                self.edge_visibility,
            )
            if potential_edge:
                added_edges.append(potential_edge)

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        if create_node:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            if not self._admit_node(
                lemma=canon,
                node_type=inferred_type,
                provenance="TEXT_FALLBACK",
            ):
                return None

            return self.graph.add_or_get_node(lemmas, canon, inferred_type, node_source)
        return None

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]
        else:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            for node in graph_active_nodes:
                if lemmas == node.lemmas:
                    return node
        if create_node:
            canon, inferred_type = self._canonicalize_and_classify_node_text(text)
            if inferred_type is None:
                return None
            lemmas = get_concept_lemmas(self.spacy_nlp, canon)
            if not self._admit_node(
                lemma=canon,
                node_type=inferred_type,
                provenance="TEXT_FALLBACK",
            ):
                return None

            return self.graph.add_or_get_node(lemmas, canon, inferred_type, node_source)

        return None

    def is_content_word_and_non_stopword(
        self,
        token: Token,
        pos_list: List[str] = [
            "NOUN",
            "PROPN",
            "ADJ",
        ],
    ) -> bool:
        return (token.pos_ in pos_list) and (
            token.lemma_ not in self.spacy_nlp.Defaults.stop_words
        )

    def get_phrase_level_concepts(self, sent):
        """
        Extract phrase-level concepts from a sentence per AMoC v4 paper.

        Per paper Figures 2-4:
        - Node labels are single lowercase lemmas (e.g., "country" not "the country")
        - Determiners are NEVER included in node labels
        - Each noun becomes a CONCEPT node, each adjective a PROPERTY node
        """
        phrase_nodes = []

        # spaCy noun chunks = adjective + noun phrases
        for chunk in sent.noun_chunks:
            # Extract the head noun from the chunk (ignore determiners completely)
            head_noun = None
            for tok in chunk:
                if tok.pos_ in {"NOUN", "PROPN"}:
                    head_noun = tok
                    break

            if head_noun is None:
                continue

            # CRITICAL FIX: Use single lemma as node key, not full phrase
            # Per AMoC paper: nodes are "country" not "the country"
            lemma = head_noun.lemma_.lower()

            # actual_text should also be the clean lemma (no determiners)
            # This ensures get_text_representer() returns the clean label
            if not self._admit_node(
                lemma=lemma,
                node_type=NodeType.CONCEPT,
                provenance="STORY_TEXT",
            ):
                continue

            node = self.graph.add_or_get_node(
                lemmas=[lemma],
                actual_text=lemma,
                node_type=NodeType.CONCEPT,
                node_source=NodeSource.TEXT_BASED,
                provenance=NodeProvenance.STORY_TEXT,
            )

            phrase_nodes.append(node)

        return phrase_nodes

    def get_senteces_text_based_nodes(
        self, previous_sentences: List[Span], create_unexistent_nodes: bool = True
    ) -> Tuple[List[Node], List[str]]:

        text_based_nodes: list[Node] = []
        text_based_words: list[str] = []

        for sent in previous_sentences:
            content_words = get_content_words_from_sent(self.spacy_nlp, sent)

            for word in content_words:
                lemma = word.lemma_.lower().strip()

                # ---------- CONCEPT NODES (nouns / proper nouns) ----------
                if word.pos_ in {"NOUN", "PROPN"}:
                    if not self._admit_node(
                        lemma=lemma,
                        node_type=NodeType.CONCEPT,
                        provenance="STORY_TEXT",
                        sent=sent,
                    ):
                        continue

                    # ==========================================================================
                    # NODE ROLE DETECTION (per AMoC v4 paper alignment)
                    # Detect semantic role based on syntactic dependency:
                    # - SETTING: nouns in prepositional phrases (pobj, obl) - locations/environments
                    # - ACTOR: subjects of actions (nsubj, nsubjpass)
                    # - OBJECT: direct objects and other nouns (dobj, attr, etc.)
                    # ==========================================================================
                    node_role = None
                    if word.dep_ in {"pobj", "obl"}:
                        # Prepositional objects are typically locations/settings
                        node_role = NodeRole.SETTING
                    elif word.dep_ in {"nsubj", "nsubjpass"}:
                        # Subjects are typically actors/agents
                        node_role = NodeRole.ACTOR
                    else:
                        # Default to OBJECT for other noun dependencies
                        node_role = NodeRole.OBJECT

                    node = self.graph.get_node([lemma])
                    if node is not None:
                        node.add_actual_text(lemma)
                        # Update role if not already set
                        if node.node_role is None:
                            node.node_role = node_role
                    elif create_unexistent_nodes:
                        node = self.graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.CONCEPT,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            node_role=node_role,
                        )
                    else:
                        continue

                    text_based_nodes.append(node)
                    text_based_words.append(lemma)

                # ---------- ADJECTIVE NODES (per AMoC-v4 Figure 7) ----------
                # Per AMoC-v4: adjectives are explicit nodes in their origin sentence.
                # They are treated as PROPERTY nodes (per AMoC v4 Figure 7)
                # only in the sentence they appear, becoming inactive in later sentences
                # unless reasserted.
                elif word.pos_ == "ADJ":
                    if not self._admit_node(
                        lemma=lemma,
                        node_type=NodeType.PROPERTY,
                        provenance="STORY_TEXT",
                        sent=sent,
                    ):
                        continue

                    node = self.graph.get_node([lemma])
                    if node is not None:
                        node.add_actual_text(lemma)
                    elif create_unexistent_nodes:
                        node = self.graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.PROPERTY,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            node_role=NodeRole.PROPERTY,
                        )
                    else:
                        continue

                    text_based_nodes.append(node)
                    text_based_words.append(lemma)

        return text_based_nodes, text_based_words
