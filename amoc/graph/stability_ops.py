"""
Stability operations for Graph.

Contains methods for enforcing cumulative stability and carryover connectivity.
These methods mutate state and are separate from pure topology checks.
Moved from Graph class to separate topology from mutation-based logic.
"""

from typing import TYPE_CHECKING, Set
import networkx as nx

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node


class StabilityOps:
    """Stability operations that can be applied to a Graph."""

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def enforce_cumulative_stability(
        self,
        explicit_nodes: set,
    ) -> None:
        """
        SAFETY INVARIANT:

        If cumulative graph collapses structurally, replace it
        with the active subgraph.

        Collapse conditions:
            1) No explicit nodes active
            2) All nodes inactive
            3) Active projection empty
        """
        active_nodes, active_edges = self._graph.get_active_subgraph()

        active_empty = len(active_nodes) == 0
        explicit_active = any(node.active for node in explicit_nodes)
        all_inactive = all(not node.active for node in self._graph.nodes)

        if explicit_active and not all_inactive and not active_empty:
            return

        if active_empty:
            visible_edges = {e for e in self._graph.edges if e.visibility_score > 0}

            visible_nodes = {e.source_node for e in visible_edges} | {
                e.dest_node for e in visible_edges
            }

            new_nodes = set(visible_nodes)
            new_edges = set(visible_edges)

        else:
            new_nodes = set(active_nodes)
            new_edges = set(active_edges)

        for node in new_nodes:
            node.edges = []

        for edge in new_edges:
            edge.source_node.edges.append(edge)
            edge.dest_node.edges.append(edge)

        self._graph.nodes = new_nodes
        self._graph.edges = new_edges

    def enforce_carryover_connectivity(self, carryover_nodes: set) -> None:
        if not carryover_nodes:
            return

        active_edges = [e for e in self._graph.edges if e.active and e.visibility_score > 0]

        degree_map = {}

        for e in active_edges:
            degree_map[e.source_node] = degree_map.get(e.source_node, 0) + 1
            degree_map[e.dest_node] = degree_map.get(e.dest_node, 0) + 1

        for node in list(carryover_nodes):
            if degree_map.get(node, 0) == 0:
                node.active = False

        def build_active_graph():
            G = nx.Graph()
            for e in self._graph.edges:
                if e.active and e.visibility_score > 0:
                    G.add_edge(e.source_node, e.dest_node)
            return G

        G = build_active_graph()
        sub = G.subgraph(carryover_nodes)
        components = list(nx.connected_components(sub))

        if len(components) <= 1:
            return

        for comp_a in components:
            for comp_b in components:
                if comp_a is comp_b:
                    continue

                for e in self._graph.edges:
                    if not (
                        (e.source_node in comp_a and e.dest_node in comp_b)
                        or (e.source_node in comp_b and e.dest_node in comp_a)
                    ):
                        continue

                    e.active = True
                    e.visibility_score = max(e.visibility_score, 1)
                    e.source_node.active = True
                    e.dest_node.active = True

                    G = build_active_graph()
                    sub = G.subgraph(carryover_nodes)
                    components = list(nx.connected_components(sub))

                    if len(components) <= 1:
                        return
