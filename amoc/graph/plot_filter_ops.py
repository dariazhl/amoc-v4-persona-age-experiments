from typing import TYPE_CHECKING, List, Set, Dict, Optional, Tuple

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class PlotFilterOps:
    MAX_EDGES_PER_NODE: int = 5

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

        node_edge_count: Dict["Node", List["Edge"]] = {}
        for edge in edges:
            if edge.source_node not in node_edge_count:
                node_edge_count[edge.source_node] = []
            if edge.dest_node not in node_edge_count:
                node_edge_count[edge.dest_node] = []
            node_edge_count[edge.source_node].append(edge)
            node_edge_count[edge.dest_node].append(edge)

        def edge_priority(edge: "Edge") -> Tuple[int, int]:
            forced_score = 1 if edge.forced_connection else 0
            inferred_score = 1 if edge.inferred else 0
            return (forced_score, inferred_score)

        edges_to_keep: Set["Edge"] = set()

        for node, node_edges in node_edge_count.items():
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
                filtered,
                max_edges_per_node=max_edges_per_node,
            )

        return filtered
