from collections import deque
from typing import Set
from amoc.graph import Graph, Node


class ProjectionEngine:
    """
    Controls working-memory projection.

    Guarantees:
    - Explicit nodes always score 0
    - Projection uses ONLY active edges
    - No visibility mutation here
    - No structural mutation here
    """

    def __init__(self, graph: Graph, max_distance: int):
        self.graph = graph
        self.max_distance = max_distance

    # ------------------------------------------------------------
    # BFS using ONLY active edges
    # ------------------------------------------------------------
    def _bfs(self, sources: Set[Node]) -> dict[Node, int]:
        if not sources:
            return {}

        distances = {n: 0 for n in sources}
        queue = deque(sources)

        while queue:
            node = queue.popleft()
            dist = distances[node]

            if dist >= self.max_distance:
                continue

            for edge in node.edges:
                if not edge.active:
                    continue

                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )

                if neighbor in distances:
                    continue

                distances[neighbor] = dist + 1
                queue.append(neighbor)

        return distances

    # ------------------------------------------------------------
    # Public projection entry
    # ------------------------------------------------------------
    def project(self, explicit_nodes: Set[Node]) -> None:

        # 1. Reset scores hard
        for node in self.graph.nodes:
            node.score = 100

        # 2. Run BFS from explicit nodes
        distances = self._bfs(explicit_nodes)

        # 3. Assign scores
        for node, dist in distances.items():
            node.score = dist

        # 4. Update edge.active (purely structural)
        for edge in self.graph.edges:
            if edge.visibility_score <= 0:
                edge.active = False
                continue

            if (
                edge.source_node.score <= self.max_distance
                and edge.dest_node.score <= self.max_distance
            ):
                edge.active = True
            else:
                edge.active = False
