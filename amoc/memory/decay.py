import logging
from typing import TYPE_CHECKING, Optional, List, Set, Dict
from collections import deque

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.node import NodeSource
    from amoc.core.edge import Edge


class Decay:
    def __init__(
        self,
        graph_ref: "Graph",
        llm_extractor,
        get_explicit_nodes: callable,
        get_story_context: callable,  # Added for semantic decay
        max_distance: int,
        edge_visibility: int,
        nr_relevant_edges: int,
        strict_reactivate: bool = True,  # uses LLM to select which edges to reactivate
    ):
        self._graph = graph_ref
        self._llm = llm_extractor
        self._get_explicit_nodes = get_explicit_nodes
        self._get_story_context = get_story_context  # Store for semantic decay
        self._max_distance = max_distance
        self._edge_visibility = edge_visibility
        self._nr_relevant_edges = nr_relevant_edges
        self._strict_reactivate = strict_reactivate
        self._current_sentence_index = None
        self._current_sentence_text = None
        self._persona = None
        self._record_edge_fn = None

    def set_decay_state_refs(
        self,
        anchor_nodes: Set["Node"],
        record_edge_fn: callable = None,
        persona: str = None,  # Added persona parameter
    ):
        self._record_edge_fn = record_edge_fn
        self._persona = persona  # Store persona for LLM calls

    def set_decay_sentence_context(
        self, idx: int, text: str = None
    ):  # Added text parameter
        self._current_sentence_index = idx
        if text:
            self._current_sentence_text = text

    # global decay = fade edges that are never reinforced in cumulative graph
    def apply_global_edge_decay(self) -> None:
        for edge in self._graph.edges:
            if edge.created_at_sentence == self._current_sentence_index:
                continue
            if not edge.asserted_this_sentence and not edge.reactivated_this_sentence:
                edge.reduce_visibility()
                if edge.visibility_score <= 0:
                    edge.visibility_score = 0
                    edge.active = False

    def apply_semantic_edge_decay(self) -> None:
        # Get all active edges that are candidates for decay
        decay_candidates = []
        candidate_strings = []
        edge_to_triplet = {}

        for edge in self._graph.edges:
            if edge.created_at_sentence == self._current_sentence_index:
                continue
            if edge.asserted_this_sentence or edge.reactivated_this_sentence:
                continue

            # Format for LLM
            triplet = f"({edge.source_node.get_text_representer()}, {edge.label}, {edge.dest_node.get_text_representer()})"
            candidate_strings.append(triplet)
            decay_candidates.append(edge)
            edge_to_triplet[edge] = triplet

        if not decay_candidates:
            return

        # Build story context (last few sentences)
        story_context = self._get_story_context() if self._get_story_context else ""

        # Get current sentence text
        current_sentence = self._current_sentence_text

        if not current_sentence:
            logging.warning("SEMANTIC_DECAY: No current sentence text, skipping")
            self._apply_fallback_decay(decay_candidates)
            return

        # Call LLM to evaluate narrative relevance
        try:
            result = self._llm.check_narrative_relevance(
                story_context=story_context,
                current_sentence=current_sentence,
                active_triplets="\n".join(candidate_strings),
                persona=self._persona,
            )
        except Exception as e:
            logging.error(f"SEMANTIC_DECAY: LLM call failed: {e}")
            self._apply_fallback_decay(decay_candidates)
            return

        if not result or "scores" not in result:
            # Fallback to simple decay
            self._apply_fallback_decay(decay_candidates)
            return

        # Get scores from LLM response
        scores = result.get("scores", {})
        reasoning = result.get("reasoning", "")

        if reasoning:
            logging.info(f"SEMANTIC_DECAY: LLM reasoning: {reasoning}")

        # Build connectivity map to check critical edges
        connectivity_map = self.build_connectivity_map()

        # Track stats for logging
        stats = {"reinforce": 0, "maintain": 0, "decay": 0, "protected": 0}

        for edge in decay_candidates:
            triplet_str = edge_to_triplet[edge]
            score = scores.get(triplet_str, 2)  # Default to maintain (2)

            # Ensure score is within 1-3 range
            try:
                score = int(score)
                if score < 1:
                    score = 1
                elif score > 3:
                    score = 3
            except:
                score = 2

            # Check if this edge is critical for connectivity
            is_critical = not self.can_remove_edge(edge, connectivity_map)

            if score == 1:  # LOW RELEVANCE - should decay
                if is_critical:
                    # Double decay
                    edge.reduce_visibility()
                    edge.reduce_visibility()
                    stats["protected"] += 1
                    logging.info(
                        f"SEMANTIC_DECAY: Kept connectivity-critical edge {triplet_str} (would break graph)"
                    )
                else:
                    # Safe to decay fully
                    edge.reduce_visibility()
                    if edge.visibility_score <= 0:
                        edge.visibility_score = 0
                        edge.active = False
                    stats["decay"] += 1
                    logging.info(
                        f"SEMANTIC_DECAY: Decayed edge {triplet_str} (score=1)"
                    )

            elif score == 2:  # MEDIUM RELEVANCE - maintain
                # Decay faster
                edge.visibility_score = max(0, edge.visibility_score - 1)
                stats["maintain"] += 1
                logging.info(f"SEMANTIC_DECAY: Maintained edge {triplet_str} (score=2)")

            elif score == 3:  # HIGH RELEVANCE - reinforce
                # Reactivate to full visibility
                edge.visibility_score = min(
                    edge.visibility_score + 1, self._edge_visibility
                )
                edge.mark_as_reactivated(reset_score=False)
                stats["reinforce"] += 1
                logging.info(
                    f"SEMANTIC_DECAY: Reactivated edge {triplet_str} to full visibility (score=3)"
                )

            else:
                # Fallback for any other values
                edge.reduce_visibility()
                logging.info(
                    f"SEMANTIC_DECAY: Fallback decay for edge {triplet_str} (unexpected score={score})"
                )

        logging.info(
            f"SEMANTIC_DECAY: Stats - Reinforce: {stats['reinforce']}, "
            f"Maintain: {stats['maintain']}, Decay: {stats['decay']}, "
            f"Protected: {stats['protected']}"
        )

    def build_connectivity_map(self) -> Dict["Node", Set["Node"]]:
        connectivity = {}
        for edge in self._graph.edges:
            if edge.active:
                connectivity.setdefault(edge.source_node, set()).add(edge.dest_node)
                connectivity.setdefault(edge.dest_node, set()).add(edge.source_node)
        return connectivity

    def can_remove_edge(self, edge, connectivity_map) -> bool:
        source = edge.source_node
        dest = edge.dest_node

        # If either node has multiple connections, safe to remove
        if (
            len(connectivity_map.get(source, set())) > 1
            and len(connectivity_map.get(dest, set())) > 1
        ):
            return True

        # If this is the only connection for either node, check if there's an alternative path
        if (
            len(connectivity_map.get(source, set())) == 1
            or len(connectivity_map.get(dest, set())) == 1
        ):

            # Temporarily remove edge and check if still connected
            # This is a simplified check - you might want to use BFS
            source_has_other = len(connectivity_map.get(source, set()) - {dest}) > 0
            dest_has_other = len(connectivity_map.get(dest, set()) - {source}) > 0

            return source_has_other and dest_has_other

        return False

    def calculate_inference_boost(self, edge) -> int:
        # deprecated
        return 0

    def identify_hub_nodes(self) -> Set:
        # deprecated
        return set()

    def _apply_fallback_decay(self, edges):
        for edge in edges:
            edge.reduce_visibility()
            if edge.visibility_score <= 0:
                edge.visibility_score = 0
                edge.active = False

    def reactivate_relevant_edges(
        self,
        active_nodes: List["Node"],
        prev_sentences_text: str,
        newly_added_edges: List["Edge"],
    ) -> None:
        edges_text, edges = self._graph.get_edges_str(
            self._graph.nodes, only_active=False
        )

        if not self._strict_reactivate:
            for edge in edges:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)
            return

        raw_indices = self._llm.get_relevant_edges(
            edges_text, prev_sentences_text, None
        )

        valid_indices = []
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(edges):
                valid_indices.append(i)

        valid_indices = valid_indices[: self._nr_relevant_edges]

        active_node_set = set(active_nodes)

        if valid_indices:
            selected = set(valid_indices)
            for i in selected:
                edge = edges[i - 1]
                edge.visibility_score = self._edge_visibility
                if not edge.is_property_edge():
                    continue
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)
        else:
            selected = set()  # Edge not selected by LLM
            logging.info("No relevant edges identified by LLM for reactivation")

        for idx, edge in enumerate(edges, start=1):
            if idx in selected or edge in newly_added_edges:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)

    def get_fallback_edges(
        self,
        edges: List["Edge"],
        newly_added_edges: List["Edge"],
        active_node_set: Set["Node"],
    ) -> Set["Edge"]:
        fallback = set()
        for idx, edge in enumerate(edges, start=1):
            if (
                edge in newly_added_edges
                or edge.source_node in active_node_set
                or edge.dest_node in active_node_set
            ):
                fallback.add(edge)
        return fallback

    def select_edges_for_reactivation(
        self,
        edges: List["Edge"],
        selected_indices: List[int],
        newly_added_edges: List["Edge"],
        active_node_set: Set["Node"],
    ) -> Set["Edge"]:
        edges_to_reactivate = set()
        for i in selected_indices:
            edge = edges[i - 1]
            edge.visibility_score = self._edge_visibility
            if edge.is_property_edge():
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)
            else:
                edges_to_reactivate.add(edge)

        for edge in newly_added_edges:
            if not edge.is_property_edge():
                edges_to_reactivate.add(edge)

        for idx, edge in enumerate(edges, start=1):
            if idx not in selected_indices:
                if (
                    edge.source_node in active_node_set
                    or edge.dest_node in active_node_set
                ) and not edge.is_property_edge():
                    edges_to_reactivate.add(edge)

        return edges_to_reactivate

    def process_edges(
        self, edges: List["Edge"], edges_to_reactivate: Set["Edge"]
    ) -> None:
        for edge in edges:
            if edge in edges_to_reactivate:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)
            # No decay here — apply_global_edge_decay handles it

    def propagate_activation_from_edges(self) -> None:
        pass

    def convert_to_landscape_score(self, raw_score: float) -> float:
        val = 5.0 - float(raw_score)
        if val < 0.0:
            return 0.0
        if val > 5.0:
            return 5.0
        return val

    # records the activation scores of all relevant nodes for the current sent
    # landscape model
    def record_sentence_activation_matrix(
        self,
        sentence_id: int,
        explicit_nodes: List["Node"],
        newly_inferred_nodes: Set["Node"],
        max_distance: int,
        node_token_fn: callable,
        append_record_fn: callable,
    ) -> None:
        explicit_set = set(explicit_nodes)
        distances = self.compute_distances_from_sources(
            explicit_set, max_distance=max_distance
        )

        token_to_raw_score = {}
        node_raw_score = {}

        for node in explicit_set:
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 0
                node_raw_score[node] = 0

        for node in newly_inferred_nodes:
            if node in explicit_set:
                continue
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 1
                node_raw_score[node] = 1

        for node, dist in distances.items():
            if node in explicit_set or dist <= 0:
                continue
            token = node_token_fn(node)
            if token and token not in token_to_raw_score:
                token_to_raw_score[token] = dist
                node_raw_score[node] = dist

        for token, raw_score in token_to_raw_score.items():
            append_record_fn(
                {
                    "sentence": sentence_id,
                    "token": token,
                    "score": self.convert_to_landscape_score(raw_score),
                }
            )

        # Add verb activations: take the max activation of connected nodes minus 0.5.
        # Paper: "Second, in the Landscape Model, verbs are treated as nodes. However, in AMoC v4.0, verbs are primarily found in the edges... Thus, verbs were assigned an activation score with the following procedure: the highest activation score of the nodes linked by the edge that the verb is part of, decayed by 0.5"
        verb_scores: Dict[str, float] = {}
        for edge in self._graph.edges:
            if not edge.active:
                continue
            label = (edge.label or "").strip().lower()
            if not label:
                continue
            src_tok = node_token_fn(edge.source_node)
            dst_tok = node_token_fn(edge.dest_node)
            if not src_tok or not dst_tok:
                continue
            src_raw = node_raw_score.get(edge.source_node, max_distance + 1)
            dst_raw = node_raw_score.get(edge.dest_node, max_distance + 1)
            src_act = self.convert_to_landscape_score(src_raw)
            dst_act = self.convert_to_landscape_score(dst_raw)
            verb_act = max(src_act, dst_act) - 0.5
            if verb_act < 0.0:
                verb_act = 0.0
            prev = verb_scores.get(label)
            if prev is None or verb_act > prev:
                verb_scores[label] = verb_act

        for token, score in verb_scores.items():
            append_record_fn({"sentence": sentence_id, "token": token, "score": score})

    def compute_distances_from_sources(
        self, sources: Set["Node"], max_distance: int
    ) -> Dict["Node", int]:
        if not sources:
            return {}
        distances = {s: 0 for s in sources}
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            dist = distances[node]
            if dist >= max_distance:
                continue
            for edge in node.edges:
                if not edge.active or edge.visibility_score <= 1:
                    continue
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor in distances:
                    continue
                distances[neighbor] = dist + 1
                queue.append(neighbor)
        return distances

    def restrict_active_nodes(self, explicit_nodes: List["Node"]) -> None:
        pass

    def has_active_attachment(self, lemma: str) -> bool:
        active_nodes = {n for n in self._graph.nodes if n.active}
        active_nodes |= self._get_explicit_nodes()
        return any(lemma in n.lemmas for n in active_nodes)
