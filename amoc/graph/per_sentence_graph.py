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
from typing import TYPE_CHECKING, Set, List, Dict, Optional, Tuple, FrozenSet
from collections import deque
from dataclasses import dataclass, field
import networkx as nx

if TYPE_CHECKING:
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge
    from amoc.graph.graph import Graph

from amoc.graph.node import NodeType


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
    ):
        self.cumulative_graph = cumulative_graph
        self.max_distance = max_distance
        self.anchor_nodes = frozenset(anchor_nodes)

        # State accumulated during construction
        self._explicit_nodes: Set["Node"] = set()
        self._carryover_nodes: Set["Node"] = set()
        self._pending_edges: List["Edge"] = []
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
        """
        Compute carry-over nodes via BFS from explicit nodes.

        Carry-over nodes are those reachable from explicit nodes
        within max_distance hops via active edges. This is the
        "working memory" that persists across sentences.

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are EXCLUDED from carry-over logic.
        Per AMoC v4 paper: PROPERTY nodes have no independent activation.
        They only appear when their property edge is active in the current sentence.
        BFS should not traverse through or include PROPERTY nodes as carry-over.
        """
        if not self._explicit_nodes:
            self._carryover_nodes = set()
            return self

        # BFS from explicit nodes - EXCLUDING PROPERTY nodes from propagation
        # Per paper: only CONCEPT nodes participate in carry-over / distance logic
        concept_explicit = {
            n for n in self._explicit_nodes
            if n.node_type != NodeType.PROPERTY
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

                # CRITICAL: Skip PROPERTY nodes in BFS traversal
                # Per paper: PROPERTY nodes don't propagate activation
                if neighbor.node_type == NodeType.PROPERTY:
                    continue

                if neighbor in distances:
                    continue

                distances[neighbor] = current_dist + 1
                queue.append(neighbor)

        # Carry-over = reachable CONCEPT nodes that aren't explicit
        # PROPERTY nodes are NEVER carry-over (per paper alignment)
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

    def build(self, sentence_index: int) -> PerSentenceGraph:
        """
        Build the immutable per-sentence graph.

        This constructs the final view with all invariants enforced:
        - Only active nodes (explicit + carry-over) are included
        - Only edges with BOTH endpoints active are included
        - Property edges only included in their origin sentence (per AMoC paper)
        - PROPERTY nodes only included if they have an active property edge
        - The result is guaranteed connected (or empty)
        """
        self._sentence_index = sentence_index

        # CRITICAL (Paper-Aligned):
        # PROPERTY nodes are NOT included in active_nodes via carry-over.
        # They are only included if they have an active property edge in this sentence.
        # First, collect CONCEPT nodes (explicit + carry-over)
        concept_nodes = {
            n for n in (self._explicit_nodes | self._carryover_nodes)
            if n.node_type != NodeType.PROPERTY
        }

        # Filter edges: only those where BOTH endpoints are active
        # CRITICAL: Also enforce property edge sentence constraints
        active_edges: Set["Edge"] = set()
        property_nodes_with_active_edges: Set["Node"] = set()

        for edge in self.cumulative_graph.edges:
            if not edge.active:
                continue

            # Property edges must only be active in their origin sentence
            # Per AMoC paper: properties attach via "is" edges in their sentence only
            if edge.is_property_edge():
                if edge.violates_property_sentence_constraint(sentence_index):
                    continue
                # This property edge is valid - check if it connects to an active concept
                src_is_property = edge.source_node.node_type == NodeType.PROPERTY
                dst_is_property = edge.dest_node.node_type == NodeType.PROPERTY
                concept_end = edge.dest_node if src_is_property else edge.source_node
                property_end = edge.source_node if src_is_property else edge.dest_node

                # Only include if the CONCEPT end is in active nodes
                if concept_end in concept_nodes:
                    active_edges.add(edge)
                    property_nodes_with_active_edges.add(property_end)
            else:
                # Non-property edge: both endpoints must be active CONCEPT nodes
                if edge.source_node in concept_nodes and edge.dest_node in concept_nodes:
                    active_edges.add(edge)

        # CRITICAL: PROPERTY nodes are only visible if they have an active property edge
        # This implements: "PROPERTY nodes are visible in the ACTIVE graph if and only if
        # at least one PROPERTY edge involving that node is active in the current sentence"
        active_nodes = concept_nodes | property_nodes_with_active_edges

        # Also filter explicit_nodes and carryover_nodes for consistency
        # (PROPERTY nodes should never be in carryover, but filter explicit too)
        explicit_concepts = {
            n for n in self._explicit_nodes
            if n.node_type != NodeType.PROPERTY or n in property_nodes_with_active_edges
        }
        carryover_concepts = {
            n for n in self._carryover_nodes
            if n.node_type != NodeType.PROPERTY
        }

        return PerSentenceGraph(
            sentence_index=sentence_index,
            explicit_nodes=frozenset(explicit_concepts),
            carryover_nodes=frozenset(carryover_concepts),
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
) -> PerSentenceGraph:
    """
    Convenience function to build a per-sentence graph in one call.

    Args:
        cumulative_graph: The full cumulative graph with all nodes/edges
        explicit_nodes: Nodes extracted from the current sentence text
        max_distance: Maximum BFS distance for carry-over nodes
        anchor_nodes: Nodes that are always attachable (connectivity anchors)
        sentence_index: The 1-based sentence index

    Returns:
        A PerSentenceGraph with all invariants enforced
    """
    builder = PerSentenceGraphBuilder(
        cumulative_graph=cumulative_graph,
        max_distance=max_distance,
        anchor_nodes=anchor_nodes,
    )

    return (
        builder.set_explicit_nodes(explicit_nodes)
        .compute_carryover_nodes()
        .build(sentence_index)
    )
