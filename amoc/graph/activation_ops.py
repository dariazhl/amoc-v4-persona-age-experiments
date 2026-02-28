from typing import TYPE_CHECKING, Set, Dict, List, Tuple, Optional
from collections import deque

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class ActivationOps:
    """
    Activation operations for Graph.

    Responsibilities:
    - Active subgraph extraction
    - Edge reactivation and decay
    - Edge reinforcement (similar edge detection)
    - Node scoring based on distance
    """

    MAX_REACTIVATION_COUNT: int = 6

    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    # ==========================================================================
    # EDGE REINFORCEMENT (moved from Graph)
    # ==========================================================================

    def check_and_reinforce_similar_edge(
        self,
        edge: "Edge",
        edge_visibility: int,
    ) -> bool:
        """
        Check if a similar edge exists and reinforce it.

        If a similar edge exists (same source, dest, label), reinforce it by:
        - Incrementing visibility_score (capped at edge_visibility)
        - Setting active = True
        - Marking as asserted

        Args:
            edge: The new edge to check against existing edges
            edge_visibility: Maximum visibility score cap

        Returns:
            True if similar edge was found and reinforced, False otherwise
        """
        for other_edge in self._graph.edges:
            same_nodes = (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
            )
            if not same_nodes:
                continue

            if (
                edge.label.strip().lower() == other_edge.label.strip().lower()
                and not edge.inferred
            ):
                other_edge.visibility_score = min(
                    edge_visibility, other_edge.visibility_score + 1
                )
                other_edge.active = True
                other_edge.mark_as_asserted(reset_score=False)
                return True

        return False

    def get_similar_edge(self, edge: "Edge") -> Optional["Edge"]:
        """
        Get a similar existing edge (same source, dest, label).

        Args:
            edge: The edge to find a similar match for

        Returns:
            The similar edge if found, None otherwise
        """
        for other_edge in self._graph.edges:
            if (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
                and edge.label.strip().lower() == other_edge.label.strip().lower()
            ):
                return other_edge
        return None

    # ==========================================================================
    # ACTIVE SUBGRAPH
    # ==========================================================================

    def get_active_subgraph(self) -> Tuple[Set["Node"], Set["Edge"]]:
        active_edges: Set["Edge"] = {
            e for e in self._graph.edges if e.active and e.visibility_score > 0
        }
        active_nodes: Set["Node"] = {e.source_node for e in active_edges} | {
            e.dest_node for e in active_edges
        }
        return active_nodes, active_edges

    def deactivate_all_edges(self) -> None:
        for edge in self._graph.edges:
            edge.reset_for_sentence_start()

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set["Node"],
        max_distance: int,
        current_sentence: int,
    ) -> Set["Edge"]:
        """
        Reactivate edges within BFS distance of explicit nodes.

        This is a SEMANTIC operation, not a connectivity operation.
        It reactivates edges based on proximity to current sentence concepts,
        NOT to ensure graph connectivity.

        IMPORTANT: This method may leave the graph disconnected.
        StabilityOps.enforce_connectivity() must be called afterward
        to ensure connectivity invariants hold.

        Args:
            explicit_nodes: Nodes mentioned in current sentence
            max_distance: Maximum BFS distance to consider
            current_sentence: Current sentence index (unused, for signature compat)

        Returns:
            Set of reactivated edges
        """
        from amoc.graph.node import NodeType

        if not explicit_nodes or max_distance < 1:
            return set()

        concept_seeds = {
            n
            for n in explicit_nodes
            if n.node_type != NodeType.PROPERTY and not n.is_setting()
        }

        if not concept_seeds:
            return set()

        reachable_nodes: Dict["Node", int] = {n: 0 for n in concept_seeds}
        queue: deque = deque(concept_seeds)
        visited_edges: Set["Edge"] = set()
        candidate_edges: list[tuple[int, "Edge"]] = []

        while queue:
            node = queue.popleft()
            dist = reachable_nodes[node]

            if dist >= max_distance:
                continue

            for edge in node.edges:
                if edge in visited_edges:
                    continue
                if edge.forced_connection:
                    continue
                visited_edges.add(edge)

                if edge.visibility_score <= 0:
                    continue

                if edge.active:
                    continue

                candidate_edges.append((dist, edge))

                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor not in reachable_nodes:
                    reachable_nodes[neighbor] = dist + 1
                    queue.append(neighbor)

        candidate_edges.sort(key=lambda x: x[0])
        reactivated: Set["Edge"] = set()

        for dist, edge in candidate_edges[: self.MAX_REACTIVATION_COUNT]:
            edge.mark_as_reactivated(reset_score=False)
            reactivated.add(edge)

        return reactivated

    def decay_inactive_edges(self) -> None:
        for edge in list(self._graph.edges):
            if not edge.active:
                edge.reduce_visibility()

    def bfs_from_activated_nodes(
        self,
        activated_nodes: List["Node"],
        direction: str = "both",
    ) -> Dict["Node", int]:
        from amoc.graph.node import NodeType

        distances = {}
        concept_seeds = [
            node for node in activated_nodes if node.node_type != NodeType.PROPERTY
        ]

        queue = deque([(node, 0) for node in concept_seeds])
        while queue:
            curr_node, curr_distance = queue.popleft()
            if curr_node not in distances:
                distances[curr_node] = curr_distance
                for edge in curr_node.edges:
                    if not edge.active:
                        continue

                    next_node = None
                    if direction == "both":
                        next_node = (
                            edge.dest_node
                            if edge.source_node == curr_node
                            else edge.source_node
                        )
                    elif direction == "outgoing":
                        if edge.source_node == curr_node:
                            next_node = edge.dest_node
                    elif direction == "incoming":
                        if edge.dest_node == curr_node:
                            next_node = edge.source_node

                    if next_node is not None:
                        queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List["Node"]
    ) -> None:
        from amoc.graph.node import NodeType

        distances_to_activated_nodes = self.bfs_from_activated_nodes(activated_nodes)

        for node in self._graph.nodes:
            if node.node_type == NodeType.PROPERTY:
                continue

            if node in distances_to_activated_nodes:
                node.score = distances_to_activated_nodes[node]
            else:
                node.score = min(node.score + 1, 100)

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List["Node"]:
        from amoc.graph.node import NodeSource

        return [
            node
            for node in self._graph.nodes
            if node.score <= score_threshold
            and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)
        ]

    def get_top_k_nodes(self, nodes: List["Node"], k: int) -> List["Node"]:
        return sorted(nodes, key=lambda node: node.score)[:k]

    def get_top_concepts_nodes(self, k: int) -> List["Node"]:
        from amoc.graph.node import NodeType

        nodes = [node for node in self._graph.nodes if node.node_type == NodeType.CONCEPT]
        return self.get_top_k_nodes(nodes, k)

    def get_top_text_based_concepts(self, k: int) -> List["Node"]:
        from amoc.graph.node import NodeType, NodeSource

        nodes = [
            node
            for node in self._graph.nodes
            if node.node_type == NodeType.CONCEPT
            and node.node_source == NodeSource.TEXT_BASED
        ]
        return self.get_top_k_nodes(nodes, k)
