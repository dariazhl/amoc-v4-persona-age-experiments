from __future__ import annotations

import networkx as nx


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
