from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from amoc.core.edge import Edge
    from amoc.core.graph import Graph


class CumulativeGraph:
    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    def has_edge(self, u, v, key=None) -> bool:
        return self._graph.has_edge(u, v, key=key)

    def add_edge(self, u, v, key=None, **attrs):
        return self._graph.add_edge(u, v, key=key, **attrs)

    def to_networkx(self) -> nx.MultiDiGraph:
        return self._graph

    def __getattr__(self, item):
        return getattr(self._graph, item)


class CumulativeGraphBuilder:
    def __init__(self, cumulative_graph: CumulativeGraph):
        self._cumulative_graph = cumulative_graph

    def _edge_key(self, label: str, introduced_at: int) -> str:
        return f"{label}__introduced_{introduced_at}"

    def sync_edge(self, edge: "Edge", introduced_at: int) -> None:
        u = edge.source_node.get_text_representer()
        v = edge.dest_node.get_text_representer()
        key = self._edge_key(edge.label, introduced_at)
        if not self._cumulative_graph.has_edge(u, v, key=key):
            self._cumulative_graph.add_edge(u, v, key=key, label=edge.label)

    def rebuild_from_graph(self, graph: "Graph", triplet_intro: dict) -> None:
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
