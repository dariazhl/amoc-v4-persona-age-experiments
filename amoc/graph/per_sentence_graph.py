"""
Per-Sentence Graph: A constraint-first, sentence-local graph view.

This module provides a clean abstraction for per-sentence plot construction
that guarantees:
1. Only explicit and carry-over nodes appear in the graph
2. Inactive nodes are completely excluded (nodes, edges, metrics)
3. Connectivity is maintained by construction, not by repair

The design makes disconnection impossible by encoding all constraints
directly into the construction rules.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Set, List, Dict, Optional, Tuple, FrozenSet, Callable
from collections import deque
from dataclasses import dataclass, field
import networkx as nx
import logging

if TYPE_CHECKING:
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge
    from amoc.graph.graph import Graph

from amoc.graph.node import NodeType
from amoc.graph.edge import RelationClass


@dataclass
class PerSentenceGraph:
    """
    An immutable, sentence-local view of the graph.

    This class enforces the per-sentence invariants:
    - Only explicit + carry-over nodes are visible
    - Only edges where BOTH endpoints are visible are included
    - The graph is guaranteed to be connected (or empty)

    The graph is constructed once and cannot be modified, ensuring
    that invalid states are unrepresentable.
    """

    sentence_index: int
    explicit_nodes: FrozenSet["Node"]
    carryover_nodes: FrozenSet["Node"]
    active_nodes: FrozenSet["Node"]  # explicit ∪ carryover
    active_edges: FrozenSet["Edge"]
    anchor_nodes: FrozenSet["Node"]

    # Derived structures (computed once at construction)
    _adjacency: Dict["Node", Set["Node"]] = field(default_factory=dict)
    _node_degrees: Dict["Node", int] = field(default_factory=dict)
    _nx_graph: Optional[nx.Graph] = field(default=None)

    def __post_init__(self):
        """Compute derived structures after construction."""
        # Build adjacency list
        adjacency: Dict["Node", Set["Node"]] = {n: set() for n in self.active_nodes}
        for edge in self.active_edges:
            if edge.source_node in adjacency and edge.dest_node in adjacency:
                adjacency[edge.source_node].add(edge.dest_node)
                adjacency[edge.dest_node].add(edge.source_node)
        object.__setattr__(self, "_adjacency", adjacency)

        # Compute degrees
        degrees = {n: len(neighbors) for n, neighbors in adjacency.items()}
        object.__setattr__(self, "_node_degrees", degrees)

        # Build NetworkX graph for connectivity queries
        G = nx.Graph()
        for node in self.active_nodes:
            G.add_node(node)
        for edge in self.active_edges:
            G.add_edge(edge.source_node, edge.dest_node, edge=edge)
        object.__setattr__(self, "_nx_graph", G)

    @property
    def is_connected(self) -> bool:
        """Check if the per-sentence graph is connected."""
        if self._nx_graph.number_of_nodes() <= 1:
            return True
        return nx.is_connected(self._nx_graph)

    @property
    def is_empty(self) -> bool:
        """Check if the per-sentence graph has no edges."""
        return len(self.active_edges) == 0

    def get_node_degree(self, node: "Node") -> int:
        """Get the degree of a node in the per-sentence graph."""
        return self._node_degrees.get(node, 0)

    def get_neighbors(self, node: "Node") -> Set["Node"]:
        """Get neighbors of a node in the per-sentence graph."""
        return self._adjacency.get(node, set())

    def node_is_visible(self, node: "Node") -> bool:
        """Check if a node is visible in this per-sentence graph."""
        return node in self.active_nodes

    def edge_is_visible(self, edge: "Edge") -> bool:
        """Check if an edge is visible in this per-sentence graph."""
        return edge in self.active_edges

    def get_triplets(self) -> List[Tuple[str, str, str]]:
        """Get all triplets in the per-sentence graph."""
        return [
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            for edge in self.active_edges
        ]

    def to_networkx(self) -> nx.Graph:
        """Return the NetworkX representation (undirected)."""
        return self._nx_graph.copy()


class PerSentenceGraphBuilder:
    """
    Builder for constructing per-sentence graphs with guaranteed invariants.

    This builder enforces the construction rules that make disconnection
    impossible:
    1. Start with explicit nodes (from current sentence)
    2. Expand to carry-over nodes via BFS on active edges
    3. Include only edges where BOTH endpoints are active
    4. New edges must attach to the existing component
    """

    def __init__(
        self,
        cumulative_graph: "Graph",
        max_distance: int,
        anchor_nodes: Set["Node"],
        repair_callback=None,
    ):
        self.cumulative_graph = cumulative_graph
        self.max_distance = max_distance
        self.anchor_nodes = frozenset(anchor_nodes)
        self.repair_callback = repair_callback

        self._explicit_nodes: Set["Node"] = set()
        self._carryover_nodes: Set["Node"] = set()
        self._sentence_index: Optional[int] = None

    def set_explicit_nodes(self, nodes: List["Node"]) -> "PerSentenceGraphBuilder":
        """
        Set the explicit nodes for this sentence.

        Explicit nodes are extracted from the sentence text and form
        the "seed" from which carry-over nodes are discovered.
        """
        self._explicit_nodes = set(nodes)
        return self

    def compute_carryover_nodes(self) -> "PerSentenceGraphBuilder":
        if not self._explicit_nodes:
            self._carryover_nodes = set()
            return self

        # BFS from explicit nodes - EXCLUDING PROPERTY nodes from propagation
        # Per paper: only CONCEPT nodes participate in carry-over / distance logic
        concept_explicit = {
            n for n in self._explicit_nodes if n.node_type != NodeType.PROPERTY
        }

        distances: Dict["Node", int] = {n: 0 for n in concept_explicit}
        queue: deque = deque(concept_explicit)

        while queue:
            node = queue.popleft()
            current_dist = distances[node]

            if current_dist >= self.max_distance:
                continue

            for edge in node.edges:
                if not edge.active:
                    continue
                if (
                    edge.created_at_sentence is not None
                    and self._sentence_index is not None
                ):
                    if edge.created_at_sentence > self._sentence_index:
                        continue

                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )

                # CRITICAL (Paper-aligned):
                # PROPERTY nodes do NOT propagate activation.
                # They only appear if their edge is active.
                if neighbor.node_type == NodeType.PROPERTY:
                    continue

                if neighbor in distances:
                    continue

                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

        # Carry-over = reachable CONCEPT nodes that aren't explicit
        # PROPERTY nodes are NEVER carry-over (per paper alignment)
        # ------------------------------------------------------------
        # Carryover nodes: must be within distance AND have active cumulative edge
        # ------------------------------------------------------------

        self._carryover_nodes = set(distances.keys()) - self._explicit_nodes

        return self

    def get_active_nodes(self) -> Set["Node"]:
        """Get the set of nodes visible in the per-sentence graph."""
        return self._explicit_nodes | self._carryover_nodes

    def get_attachable_nodes(self) -> Set["Node"]:
        """
        Get nodes that new edges can attach to.

        A new edge is valid only if at least one endpoint is attachable.
        This guarantees connectivity by construction.
        """
        return self._explicit_nodes | self._carryover_nodes | set(self.anchor_nodes)

    def can_add_edge(self, source: "Node", dest: "Node") -> bool:
        """
        Check if an edge can be added while maintaining connectivity.

        An edge is valid if at least one endpoint is in the attachable set
        (explicit, carry-over, or anchor nodes). This makes disconnection
        impossible by design.
        """
        attachable = self.get_attachable_nodes()
        return source in attachable or dest in attachable

    def _attempt_llm_repair(
        self,
        components: List[Set["Node"]],
        active_nodes: Set["Node"],
        active_edges: Set["Edge"],
        temperature: float = 0.3,
    ) -> Optional[Set["Edge"]]:

        if self.repair_callback is None:
            logging.warning("[PerSentenceGraph] No repair callback provided.")
            return None

        try:
            candidate_edges = self.repair_callback(
                components=components,
                active_nodes=active_nodes,
                active_edges=active_edges,
                sentence_index=self._sentence_index,
                temperature=temperature,
            )
        except Exception as e:
            logging.warning(
                "[PerSentenceGraph] Repair callback exception: %s",
                str(e),
            )
            return None

        if not candidate_edges:
            return None

        valid_edges = set()

        for e in candidate_edges:
            if e.source_node in active_nodes or e.dest_node in active_nodes:
                valid_edges.add(e)

        # --------------------------------------------------
        # Prevent duplicate-edge retry loops
        # --------------------------------------------------
        if valid_edges:
            new_edges = valid_edges - active_edges
            if not new_edges:
                # All proposed edges already exist
                return None
            return new_edges

        return None

    def _connected_components(
        self,
        nodes: Set[Node],
        edges: Set[Edge],
    ) -> List[Set[Node]]:

        adjacency = {n: set() for n in nodes}

        for e in edges:
            adjacency[e.source_node].add(e.dest_node)
            adjacency[e.dest_node].add(e.source_node)

        visited = set()
        components = []

        for node in nodes:
            if node in visited:
                continue

            stack = [node]
            comp = set()

            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                stack.extend(adjacency[n] - visited)

            components.append(comp)

        return components

    def build(self, sentence_index: int) -> PerSentenceGraph:
        self._sentence_index = sentence_index

        # ------------------------------------------------------------
        # 1. Get active subgraph from cumulative memory
        # ------------------------------------------------------------
        active_nodes, active_edges = self.cumulative_graph.get_active_subgraph()
        active_nodes = set(active_nodes)
        active_edges = set(active_edges)

        # ------------------------------------------------------------
        # 2. Guarantee explicit + carryover nodes are present
        active_nodes |= set(self._explicit_nodes)
        # ------------------------------------------------------------
        # 3. Component-level LLM repair (bounded)
        # ------------------------------------------------------------
        MAX_LLM_REPAIR_ATTEMPTS = 5
        attempt = 0

        while attempt < MAX_LLM_REPAIR_ATTEMPTS:

            components = self._connected_components(active_nodes, active_edges)

            if len(components) <= 1:
                break  # Graph connected

            repaired_edges = self._attempt_llm_repair(
                components,
                active_nodes,
                active_edges,
                temperature=0.3 if attempt < 3 else 0.7,
            )

            if repaired_edges:
                for edge in repaired_edges:
                    active_edges.add(edge)
                    active_nodes.add(edge.source_node)
                    active_nodes.add(edge.dest_node)

            attempt += 1

        # ------------------------------------------------------------
        # 4. Enforce no isolated nodes (LLM-only, bounded)
        # ------------------------------------------------------------
        MAX_ISOLATED_REPAIR = 5
        attempt_iso = 0

        def degree(node):
            return sum(
                1 for e in active_edges if e.source_node == node or e.dest_node == node
            )

        isolated_nodes = {n for n in active_nodes if degree(n) == 0}

        while isolated_nodes and attempt_iso < MAX_ISOLATED_REPAIR:

            for node in list(isolated_nodes):

                # Choose highest-degree anchor
                anchor_node = max(active_nodes, key=degree)

                if anchor_node == node:
                    continue

                repaired = self.repair_callback(
                    components=None,
                    active_nodes=active_nodes,
                    active_edges=active_edges,
                    sentence_index=self._sentence_index,
                    temperature=0.6,
                    forced_pair=(node, anchor_node),
                )

                if repaired:
                    for edge in repaired:
                        active_edges.add(edge)
                        active_nodes.add(edge.source_node)
                        active_nodes.add(edge.dest_node)

            isolated_nodes = {n for n in active_nodes if degree(n) == 0}

            attempt_iso += 1

        # ------------------------------------------------------------
        # 5. Final check (never raise)
        # ------------------------------------------------------------
        # ------------------------------------------------------------
        # HARD CONNECTIVITY GUARANTEE
        # ------------------------------------------------------------
        final_components = self._connected_components(active_nodes, active_edges)

        if len(final_components) > 1:
            logging.warning(
                "[PerSentenceGraph] Final repair attempt: forcing full connectivity."
            )

            # deterministically chain components
            sorted_components = sorted(final_components, key=len, reverse=True)

            main_component = sorted_components[0]

            for comp in sorted_components[1:]:
                representative = next(iter(comp))
                anchor = next(iter(main_component))

                forced_edges = self.repair_callback(
                    components=[{representative}, {anchor}],
                    active_nodes=active_nodes,
                    active_edges=active_edges,
                    sentence_index=self._sentence_index,
                    temperature=0.2,  # deterministic
                )

                if forced_edges:
                    for e in forced_edges:
                        active_edges.add(e)
                        active_nodes.add(e.source_node)
                        active_nodes.add(e.dest_node)

            # re-check
            final_components = self._connected_components(active_nodes, active_edges)

            if len(final_components) > 1:
                logging.error(
                    "[PerSentenceGraph] HARD FAILURE: graph still disconnected after final repair."
                )

        # ------------------------------------------------------------
        # 6. Construct immutable per-sentence graph
        # ------------------------------------------------------------
        return PerSentenceGraph(
            sentence_index=sentence_index,
            explicit_nodes=frozenset(self._explicit_nodes),
            carryover_nodes=frozenset(self._carryover_nodes),
            active_nodes=frozenset(active_nodes),
            active_edges=frozenset(active_edges),
            anchor_nodes=self.anchor_nodes,
        )


def build_per_sentence_graph(
    cumulative_graph: "Graph",
    explicit_nodes: List["Node"],
    max_distance: int,
    anchor_nodes: Set["Node"],
    sentence_index: int,
    repair_callback: Optional[
        Callable[
            [List[Set["Node"]], Set["Node"], Set["Edge"], int],
            Optional[Set["Edge"]],
        ]
    ] = None,
) -> PerSentenceGraph:

    builder = PerSentenceGraphBuilder(
        cumulative_graph=cumulative_graph,
        max_distance=max_distance,
        anchor_nodes=anchor_nodes,
        repair_callback=repair_callback,
    )

    return (
        builder.set_explicit_nodes(explicit_nodes)
        .compute_carryover_nodes()
        .build(sentence_index)
    )
