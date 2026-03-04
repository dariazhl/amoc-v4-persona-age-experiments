from __future__ import annotations

from typing import TYPE_CHECKING

from amoc.graph_views.active_graph import ActiveGraph

if TYPE_CHECKING:
    from amoc.graph.edge import Edge
    from amoc.graph.graph import Graph


class ActiveGraphBuilder:
    def __init__(self, active_graph: ActiveGraph):
        self._active_graph = active_graph

    def _edge_key(self, label: str, introduced_at: int) -> str:
        return f"{label}__introduced_{introduced_at}"

    def reset(self) -> None:
        self._active_graph.reset()

    def sync_edge(self, edge: "Edge", introduced_at: int) -> None:
        u = edge.source_node.get_text_representer()
        v = edge.dest_node.get_text_representer()
        key = self._edge_key(edge.label, introduced_at)
        if edge.active:
            if not self._active_graph.has_edge(u, v, key=key):
                self._active_graph.add_edge(u, v, key=key, label=edge.label)
        else:
            self._active_graph.remove_edge(u, v, key=key)

    def rebuild_from_graph(self, graph: "Graph", triplet_intro: dict) -> None:
        self.reset()
        for edge in graph.edges:
            label = edge.label or ""
            if not label.strip():
                continue
            u = edge.source_node.get_text_representer()
            v = edge.dest_node.get_text_representer()
            introduced = triplet_intro.get((u, label, v))
            if introduced is None:
                introduced = (
                    edge.created_at_sentence if edge.created_at_sentence is not None else -1
                )
            self.sync_edge(edge=edge, introduced_at=int(introduced))
