from typing import TYPE_CHECKING, Set, Dict, List, Tuple, Optional
from amoc.graph.node import NodeSource

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class NodeActivationEngine:
    MAX_REACTIVATION_COUNT: int = 6

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def find_and_reinforce_similar_edge(
        self,
        edge: "Edge",
        edge_visibility: int,
    ) -> Optional["Edge"]:
        # Attention simulation disabled: no edge reinforcement path.
        return None

    def get_active_subgraph(self) -> Tuple[Set["Node"], Set["Edge"]]:
        # Attention simulation disabled: treat whole graph as available.
        all_edges = set(self._graph.edges)
        all_nodes = set(self._graph.nodes)
        return all_nodes, all_edges

    def deactivate_all_edges(self) -> None:
        # Attention simulation disabled: no sentence-level deactivation.
        return None

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set["Node"],
        max_distance: int,
        current_sentence: int,
    ) -> Set["Edge"]:
        # Attention simulation disabled: no reactivation pass.
        return set()

    def bfs_from_activated_nodes(
        self,
        activated_nodes: List["Node"],
        direction: str = "both",
    ) -> Dict["Node", int]:
        return {}

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List["Node"]
    ) -> None:
        # Attention simulation disabled: node scores are no longer updated here.
        return None

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List["Node"]:
        if not only_text_based:
            return list(self._graph.nodes)

        return [n for n in self._graph.nodes if n.node_source == NodeSource.TEXT_BASED]
