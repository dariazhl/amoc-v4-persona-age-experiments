from __future__ import annotations
from typing import TYPE_CHECKING, Set, List, Dict, Optional, Tuple, FrozenSet, Callable
from collections import deque
import networkx as nx
import logging

if TYPE_CHECKING:
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge
    from amoc.graph.graph import Graph

from amoc.graph.node import NodeType


class PerSentenceGraph:
    def __init__(
        self,
        sentence_index: int,
        explicit_nodes: FrozenSet[Node],
        carryover_nodes: FrozenSet[Node],
        active_nodes: FrozenSet[Node],
        active_edges: FrozenSet[Edge],
        anchor_nodes: FrozenSet[Node],
    ) -> None:
        self.sentence_index = sentence_index
        self.explicit_nodes = explicit_nodes
        self.carryover_nodes = carryover_nodes
        self.active_nodes = active_nodes
        self.active_edges = active_edges
        self.anchor_nodes = anchor_nodes

        adjacency = {n: set() for n in self.active_nodes}
        for edge in self.active_edges:
            if edge.source_node in adjacency and edge.dest_node in adjacency:
                adjacency[edge.source_node].add(edge.dest_node)
                adjacency[edge.dest_node].add(edge.source_node)
        self._adjacency: Dict[Node, Set[Node]] = adjacency

        # Compute degrees
        degrees = {n: len(neighbors) for n, neighbors in adjacency.items()}
        self._node_degrees: Dict[Node, int] = degrees

        # Build NetworkX graph for connectivity
        G = nx.Graph()
        for node in self.active_nodes:
            G.add_node(node)
        for edge in self.active_edges:
            G.add_edge(edge.source_node, edge.dest_node, edge=edge)
        self._nx_graph: Optional[nx.Graph] = G

    def is_connected(self) -> bool:
        if self._nx_graph.number_of_nodes() <= 1:
            return True
        return nx.is_connected(self._nx_graph)

    def is_empty(self) -> bool:
        return len(self.active_edges) == 0

    def get_node_degree(self, node: Node) -> int:
        return self._node_degrees.get(node, 0)

    def get_neighbors(self, node: Node) -> Set[Node]:
        return self._adjacency.get(node, set())

    def node_is_visible(self, node: Node) -> bool:
        return node in self.active_nodes

    def edge_is_visible(self, edge: Edge) -> bool:
        return edge in self.active_edges

    def get_triplets(self) -> List[Tuple[str, str, str]]:
        return [
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            for edge in self.active_edges
        ]

    def to_networkx(self) -> nx.Graph:
        return self._nx_graph.copy()


class PerSentenceGraphBuilder:
    def __init__(
        self,
        cumulative_graph: Graph,
        max_distance: int,
        anchor_nodes: Set[Node],
        repair_callback=None,
    ):
        self.cumulative_graph = cumulative_graph
        self.max_distance = max_distance
        self.anchor_nodes = frozenset(anchor_nodes)
        self.repair_callback = repair_callback

        self._explicit_nodes: Set[Node] = set()
        self._carryover_nodes: Set[Node] = set()
        self._sentence_index: Optional[int] = None

    def set_explicit_nodes(self, nodes: List[Node]) -> "PerSentenceGraphBuilder":
        self._explicit_nodes = set(nodes)
        return self

    def compute_carryover_nodes(self) -> "PerSentenceGraphBuilder":
        if not self._explicit_nodes:
            self._carryover_nodes = set()
            return self

        distances = {n: 0 for n in self._explicit_nodes}
        queue = deque(self._explicit_nodes)

        while queue:
            node = queue.popleft()
            current_dist = distances[node]

            if current_dist >= self.max_distance:
                continue

            for edge in node.edges:
                if not edge.active:
                    continue
                if (
                    edge.created_at_sentence is not None
                    and self._sentence_index is not None
                    and edge.created_at_sentence > self._sentence_index
                ):
                    continue

                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )

                if neighbor in distances:
                    continue

                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

        # Carry-over = reachable nodes excluding explicit
        self._carryover_nodes = set(distances.keys()) - self._explicit_nodes
        return self

    def get_active_nodes(self) -> Set[Node]:
        return self._explicit_nodes | self._carryover_nodes

    def get_attachable_nodes(self) -> Set[Node]:
        return self._explicit_nodes | self._carryover_nodes | set(self.anchor_nodes)

    def can_add_edge(self, source: Node, dest: Node) -> bool:
        attachable = self.get_attachable_nodes()
        return source in attachable or dest in attachable

    def build(self, sentence_index: int) -> PerSentenceGraph:
        self._sentence_index = sentence_index

        active_nodes, active_edges = self.cumulative_graph.get_active_subgraph_wrapper()
        active_nodes = set(active_nodes)
        active_edges = set(active_edges)

        nodes_with_edges = {e.source_node for e in active_edges} | {
            e.dest_node for e in active_edges
        }
        active_nodes = nodes_with_edges

        explicit_nodes = {n for n in self._explicit_nodes if n in active_nodes}
        carryover_nodes = {n for n in self._carryover_nodes if n in active_nodes}

        return PerSentenceGraph(
            sentence_index=sentence_index,
            explicit_nodes=frozenset(explicit_nodes),
            carryover_nodes=frozenset(carryover_nodes),
            active_nodes=frozenset(active_nodes),
            active_edges=frozenset(active_edges),
            anchor_nodes=self.anchor_nodes,
        )


def build_per_sentence_graph(
    cumulative_graph: Graph,
    explicit_nodes: List[Node],
    max_distance: int,
    anchor_nodes: Set[Node],
    sentence_index: int,
    repair_callback: Optional[
        Callable[
            [List[Set[Node]], Set[Node], Set[Edge], int],
            Optional[Set[Edge]],
        ]
    ] = None,
) -> PerSentenceGraph:

    builder = PerSentenceGraphBuilder(
        cumulative_graph=cumulative_graph,
        max_distance=max_distance,
        anchor_nodes=anchor_nodes,
        repair_callback=repair_callback,
    )

    return (
        builder.set_explicit_nodes(explicit_nodes)
        .compute_carryover_nodes()
        .build(sentence_index)
    )
