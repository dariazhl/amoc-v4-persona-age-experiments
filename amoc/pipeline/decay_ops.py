# amoc/pipeline/decay_ops.py
from typing import TYPE_CHECKING, Set, Optional

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class DecayOps:
    """
    Encapsulates all decay and reactivation logic.
    Injected with references to parent state - does not own state.
    """

    def __init__(
        self,
        graph_ref: "Graph",
        get_explicit_nodes: callable,
        max_distance: int,
    ):
        self._graph = graph_ref
        self._get_explicit_nodes = get_explicit_nodes
        self._max_distance = max_distance

    # =========================================================
    # EDGE DECAY
    # =========================================================

    def apply_global_edge_decay(self) -> None:
        """
        Apply decay to all inactive edges.
        Paper-aligned: edges not active this sentence lose visibility.
        """
        for edge in list(self._graph.edges):
            if not edge.active:
                edge.reduce_visibility()

    # =========================================================
    # NODE DECAY
    # =========================================================

    def decay_node_activation(self) -> None:
        """
        Decay node activation scores.
        Explicit nodes are protected from decay.
        """
        explicit_nodes = self._get_explicit_nodes()

        for node in self._graph.nodes:
            if node in explicit_nodes:
                continue

            if not node.active:
                node.score = min(node.score + 1, 100)

    # =========================================================
    # REACTIVATION
    # =========================================================

    def reactivate_relevant_edges(
        self,
        current_sentence: int,
    ) -> Set["Edge"]:
        """
        Reactivate memory edges within distance from explicit nodes.
        Returns set of reactivated edges.
        """
        explicit_nodes = self._get_explicit_nodes()

        if not explicit_nodes:
            return set()

        return self._graph.reactivate_memory_edges_within_distance(
            explicit_nodes=explicit_nodes,
            max_distance=self._max_distance,
            current_sentence=current_sentence,
        )
