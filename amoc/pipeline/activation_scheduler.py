from typing import TYPE_CHECKING, Optional, List, Set, Dict
from collections import deque

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class ActivationScheduler:
    def __init__(
        self,
        graph_ref: "Graph",
        client_ref,
        get_explicit_nodes: callable,
        max_distance: int,
        edge_visibility: int,
        nr_relevant_edges: int,
        strict_reactivate: bool = True,
    ):
        self._graph = graph_ref
        self._client = client_ref
        self._get_explicit_nodes = get_explicit_nodes
        self._max_distance = max_distance
        self._edge_visibility = edge_visibility
        self._nr_relevant_edges = nr_relevant_edges
        self._strict_reactivate = strict_reactivate
        self._anchor_nodes: Set["Node"] = set()
        self._current_sentence_index = None
        self._record_edge_fn = None

    def set_state_refs(
        self,
        anchor_nodes: Set["Node"],
        record_edge_fn: callable = None,
    ):
        self._anchor_nodes = anchor_nodes
        self._record_edge_fn = record_edge_fn

    def set_current_sentence(self, idx: int):
        self._current_sentence_index = idx

    def apply_global_edge_decay(self) -> None:
        for edge in self._graph.edges:
            # Don't decay edges created this sentence
            if edge.created_at_sentence == self._current_sentence_index:
                continue

            # Only decay edges reactivated this sentence
            if not edge.asserted_this_sentence and not edge.reactivated_this_sentence:
                edge.reduce_visibility()

                if edge.visibility_score <= 0:
                    edge.visibility_score = 0
                    edge.active = False

    def decay_node_activation(self) -> None:
        explicit_nodes = self._get_explicit_nodes()

        for node in self._graph.nodes:
            # Explicit nodes never decay this sentence
            if node in explicit_nodes:
                node.active = True
                continue

            # Step 1 — numeric decay
            if node.activation_score > 0:
                node.activation_score -= 1

            # Step 2 — cutoff by score
            if node.activation_score <= 0:
                node.active = False
                continue

            # Step 3 — active node must have ≥1 active edge
            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self._graph.edges
            )

            if not has_active_edge:
                node.active = False

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
                edge.mark_as_reactivated(reset_score=False)
                edge.visibility_score = self._edge_visibility
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)
            return

        raw_indices = self._client.get_relevant_edges(
            edges_text, prev_sentences_text, None
        )

        valid_indices: List[int] = []
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(edges):
                valid_indices.append(i)

        valid_indices = valid_indices[: self._nr_relevant_edges]

        active_node_set = set(active_nodes)

        if not valid_indices:
            # Fallback: keep edges that are newly added
            selected = set()
            connected_nodes = active_node_set
            for idx, edge in enumerate(edges, start=1):
                if edge in newly_added_edges:
                    selected.add(idx)
        else:
            selected = set(valid_indices)
            for i in selected:
                edge = edges[i - 1]
                edge.visibility_score = self._edge_visibility
                if not edge.is_property_edge():
                    continue
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)

        # Apply decay to edges
        for idx, edge in enumerate(edges, start=1):
            if idx in selected or edge in newly_added_edges:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(reset_score=False)
                edge.visibility_score = self._edge_visibility
            else:
                edge.reduce_visibility()

    def reactivate_memory_edges(
        self,
        current_sentence: int,
    ) -> Set["Edge"]:
        # Reactivate memory edges within distance from explicit nodes
        explicit_nodes = self._get_explicit_nodes()

        if not explicit_nodes:
            return set()

        return self._graph.reactivate_memory_edges_within_distance(
            explicit_nodes=explicit_nodes,
            max_distance=self._max_distance,
            current_sentence=current_sentence,
        )

    def propagate_activation_from_edges(self) -> None:
        for edge in self._graph.edges:
            if not edge.active:
                continue
            edge.source_node.activation_score = max(
                edge.source_node.activation_score, edge.activation_score
            )
            edge.dest_node.activation_score = max(
                edge.dest_node.activation_score, edge.activation_score
            )
            edge.source_node.active = True
            edge.dest_node.active = True

    def record_sentence_activation(
        self,
        sentence_idx: int,
        explicit_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
    ) -> dict:
        active_nodes, active_edges = self._graph.get_active_subgraph()

        return {
            "sentence_idx": sentence_idx,
            "explicit_count": len(explicit_nodes),
            "carryover_count": len(carryover_nodes),
            "active_node_count": len(active_nodes),
            "active_edge_count": len(active_edges),
            "explicit_nodes": [n.get_text_representer() for n in explicit_nodes],
            "carryover_nodes": [n.get_text_representer() for n in carryover_nodes],
        }

    def record_sentence_activation_matrix(
        self,
        sentence_id: int,
        explicit_nodes: List["Node"],
        newly_inferred_nodes: Set["Node"],
        max_distance: int,
        node_token_fn: callable,
        append_record_fn: callable,
    ) -> None:

        def _to_landscape_score(raw_score: float) -> float:
            # Transform AMoC "distance" style (0 -> most active) into Landscape style (5 -> most active).
            val = 5.0 - float(raw_score)
            if val < 0.0:
                return 0.0
            if val > 5.0:
                return 5.0
            return val

        explicit_set = set(explicit_nodes)
        distances = self.distances_from_sources_active_edges(
            explicit_set, max_distance=max_distance
        )

        token_to_raw_score: Dict[str, int] = {}
        node_raw_score: Dict["Node", int] = {}

        # explicit nodes reset to 0
        for node in explicit_set:
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 0
                node_raw_score[node] = 0

        # newly inferred nodes start at 1 (never 0)
        for node in newly_inferred_nodes:
            if node in explicit_set:
                continue
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 1
                node_raw_score[node] = 1

        # carried-over nodes within range, score = distance
        for node, dist in distances.items():
            if node in explicit_set:
                continue
            if dist <= 0:
                continue
            token = node_token_fn(node)
            if not token:
                continue
            if token in token_to_raw_score:
                continue
            token_to_raw_score[token] = dist
            node_raw_score[node] = dist

        # Convert node scores to Landscape scale and record
        for token, raw_score in token_to_raw_score.items():
            append_record_fn(
                {
                    "sentence": sentence_id,
                    "token": token,
                    "score": _to_landscape_score(raw_score),
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
            src_raw = node_raw_score.get(
                edge.source_node, edge.source_node.activation_score
            )
            dst_raw = node_raw_score.get(
                edge.dest_node, edge.dest_node.activation_score
            )
            src_act = _to_landscape_score(src_raw)
            dst_act = _to_landscape_score(dst_raw)
            verb_act = max(src_act, dst_act) - 0.5
            if verb_act < 0.0:
                verb_act = 0.0
            prev = verb_scores.get(label)
            if prev is None or verb_act > prev:
                verb_scores[label] = verb_act

        for token, score in verb_scores.items():
            append_record_fn({"sentence": sentence_id, "token": token, "score": score})

    def distances_from_sources_active_edges(
        self, sources: Set["Node"], max_distance: int
    ) -> Dict["Node", int]:
        if not sources:
            return {}
        distances: Dict["Node", int] = {s: 0 for s in sources}
        queue: deque["Node"] = deque(sources)
        while queue:
            node = queue.popleft()
            dist = distances[node]
            if dist >= max_distance:
                continue
            for edge in node.edges:
                if not edge.active:
                    continue

                # Ignore edges that are fully faded
                if edge.visibility_score <= 1:
                    continue
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor in distances:
                    continue
                distances[neighbor] = dist + 1
                queue.append(neighbor)
        return distances

    def restrict_active_to_current_explicit(self, explicit_nodes: List["Node"]) -> None:
        explicit_set = set(explicit_nodes)

        for node in self._graph.nodes:
            # Explicit nodes always stay active if they have active edges
            if node in explicit_set:
                has_active_edge = any(
                    e.active and (e.source_node == node or e.dest_node == node)
                    for e in self._graph.edges
                )
                node.active = has_active_edge
                continue

            # Non-explicit nodes only stay active if they have active edges
            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self._graph.edges
            )
            node.active = has_active_edge

    def has_active_attachment(self, lemma: str) -> bool:
        active_nodes = {n for n in self._graph.nodes if n.active}
        active_nodes |= self._get_explicit_nodes()
        return any(lemma in n.lemmas for n in active_nodes)
