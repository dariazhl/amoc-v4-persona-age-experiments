from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from amoc.core.edge import Edge
    from amoc.core.graph import Graph


class ActiveGraph:
    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def reset(self) -> None:
        self._graph = nx.MultiDiGraph()

    def has_edge(self, u, v, key=None) -> bool:
        return self._graph.has_edge(u, v, key=key)

    def add_edge(self, u, v, key=None, **attrs):
        return self._graph.add_edge(u, v, key=key, **attrs)

    def remove_edge(self, u, v, key=None) -> None:
        if self._graph.has_edge(u, v, key=key):
            self._graph.remove_edge(u, v, key=key)

    def to_networkx(self) -> nx.MultiDiGraph:
        return self._graph

    def __getattr__(self, item):
        return getattr(self._graph, item)


# keeps the active graph synchronized with the core graph
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
                    edge.created_at_sentence
                    if edge.created_at_sentence is not None
                    else -1
                )
            self.sync_edge(edge=edge, introduced_at=int(introduced))
