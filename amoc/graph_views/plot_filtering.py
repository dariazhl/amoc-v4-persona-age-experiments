from typing import TYPE_CHECKING, List, Set, Dict, Optional, Tuple
from amoc.config.constants import MAX_EDGES_PER_NODE

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge


class PlotFiltering:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def get_edges_for_plotting(
        self,
        *,
        exclude_connective: bool = True,
        exclude_inferred: bool = True,
        exclude_persona_influenced: bool = True,
        active_only: bool = True,
    ) -> List["Edge"]:
        plot_edges = []
        for edge in self._graph.edges:
            if active_only and not edge.active:
                continue
            if exclude_connective and edge.forced_connection:
                continue
            if exclude_inferred and edge.inferred:
                continue
            if exclude_persona_influenced and edge.persona_influenced:
                continue
            plot_edges.append(edge)
        return plot_edges

    def get_edges_with_degree_cap(
        self,
        edges: List["Edge"],
        max_edges_per_node: Optional[int] = None,
    ) -> List["Edge"]:
        if max_edges_per_node is None:
            max_edges_per_node = self.MAX_EDGES_PER_NODE

        incident_edges: Dict["Node", List["Edge"]] = {}
        for edge in edges:
            incident_edges.setdefault(edge.source_node, []).append(edge)
            incident_edges.setdefault(edge.dest_node, []).append(edge)

        def edge_priority(edge: "Edge") -> Tuple[int, int]:
            forced_score = 1 if edge.forced_connection else 0
            inferred_score = 1 if edge.inferred else 0
            return (forced_score, inferred_score)

        edges_to_keep = set()

        for node, node_edges in incident_edges.items():
            if len(node_edges) <= max_edges_per_node:
                edges_to_keep.update(node_edges)
            else:
                sorted_edges = sorted(node_edges, key=edge_priority)
                edges_to_keep.update(sorted_edges[:max_edges_per_node])

        return [e for e in edges if e in edges_to_keep]

    def get_plot_ready_edges(
        self,
        *,
        active_only: bool = True,
        apply_degree_cap: bool = True,
        max_edges_per_node: Optional[int] = None,
    ) -> List["Edge"]:
        if getattr(self._graph, "_debug_no_filter", False):
            return list(self._graph.edges)

        filtered = self.get_edges_for_plotting(
            exclude_connective=True,
            exclude_inferred=True,
            exclude_persona_influenced=True,
            active_only=active_only,
        )

        if apply_degree_cap:
            filtered = self.get_edges_with_degree_cap(
                filtered, max_edges_per_node=max_edges_per_node
            )

        return filtered
