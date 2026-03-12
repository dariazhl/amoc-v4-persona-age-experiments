import logging
from typing import TYPE_CHECKING, Set, Dict, List, Tuple, Optional
from collections import deque
from amoc.core.node import NodeType, NodeSource


if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge


class NodeActivationEngine:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def find_and_reinforce_similar_edge(
        self,
        edge: "Edge",
        edge_visibility: int,
    ) -> Optional["Edge"]:
        if edge.inferred:
            return None

        for other_edge in self._graph.edges:
            if (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
                and edge.label.strip().lower() == other_edge.label.strip().lower()
            ):
                other_edge.visibility_score = min(
                    edge_visibility, other_edge.visibility_score + 1
                )
                other_edge.active = True
                other_edge.mark_as_current_sentence(reset_score=False)
                return other_edge

        return None

    def get_active_subgraph(self) -> Tuple[Set["Node"], Set["Edge"]]:
        active_edges = {
            e for e in self._graph.edges if e.active and e.visibility_score > 0
        }
        active_nodes = {e.source_node for e in active_edges} | {
            e.dest_node for e in active_edges
        }
        return active_nodes, active_edges

    def deactivate_all_edges(self):
        active_before = sum(1 for e in self._graph.edges if e.active)
        for edge in self._graph.edges:
            edge.reset_for_sentence_start()
        active_after = sum(1 for e in self._graph.edges if e.active)
        logging.info(
            f"deactivated all edges | "
            f"active before: {active_before} | "
            f"active after: {active_after} | "
            f"total edges: {len(self._graph.edges)}"
        )

    def decay_inactive_edges(self) -> None:
        for edge in list(self._graph.edges):
            if not edge.active:
                edge.reduce_visibility()

    def bfs_from_activated_nodes(
        self,
        activated_nodes: List["Node"],
        direction: str = "both",
    ) -> Dict["Node", int]:
        distances = {}
        queue = deque(
            (node, 0) for node in activated_nodes if node.node_type != NodeType.PROPERTY
        )

        while queue:
            curr_node, curr_distance = queue.popleft()
            if curr_node in distances:
                continue
            distances[curr_node] = curr_distance
            for edge in curr_node.edges:
                # Only traverse edges that are active AND have positive visibility
                if not (edge.active and edge.visibility_score > 0):
                    continue

                next_node = None
                if direction == "both":
                    next_node = (
                        edge.dest_node
                        if edge.source_node == curr_node
                        else edge.source_node
                    )
                elif direction == "outgoing" and edge.source_node == curr_node:
                    next_node = edge.dest_node
                elif direction == "incoming" and edge.dest_node == curr_node:
                    next_node = edge.source_node

                if next_node is not None:
                    queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List["Node"]
    ) -> None:
        distances = self.bfs_from_activated_nodes(activated_nodes)

        for node in self._graph.nodes:
            if node.node_type == NodeType.PROPERTY:
                continue
            if node in distances:
                node.score = distances[node]
            else:
                node.score = min(node.score + 1, 100)

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List["Node"]:
        return [
            node
            for node in self._graph.nodes
            if node.score <= score_threshold
            and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)
        ]
