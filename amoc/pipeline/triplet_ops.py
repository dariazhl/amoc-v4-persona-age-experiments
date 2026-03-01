# amoc/pipeline/triplet_ops.py
"""
Triplet extraction and graph export operations extracted from core.py.
Internal helper class - not a public API.
"""
from typing import TYPE_CHECKING, List, Tuple, Set, Optional, Dict
import networkx as nx

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class TripletOps:
    """
    Encapsulates all triplet extraction and graph export logic.
    Injected with references to parent state - does not own state.
    """

    def __init__(
        self,
        graph_ref: "Graph",
        cumulative_graph_ref: nx.MultiDiGraph,
        active_graph_ref: nx.MultiDiGraph,
        triplet_intro_ref: Dict[Tuple[str, str, str], int],
    ):
        self._graph = graph_ref
        self._cumulative_graph = cumulative_graph_ref
        self._active_graph = active_graph_ref
        self._triplet_intro = triplet_intro_ref

    # =========================================================
    # TRIPLET EXTRACTION
    # =========================================================

    def graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
        """
        Convert graph edges to triplet list.
        AMoCv4 surface-relation format: edges ARE the triplets.
        """
        triplets = []
        for edge in self._graph.edges:
            if only_active and not edge.active:
                continue
            # Skip empty labels
            if not edge.label or not str(edge.label).strip():
                continue
            # Skip self-loops
            if edge.source_node == edge.dest_node:
                continue

            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )
        return triplets

    def graph_to_triplets(self, graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
        """Convert NetworkX MultiDiGraph to triplet list."""
        triplets = []
        for u, v, data in graph.edges(data=True):
            label = data.get("label", "")
            triplets.append((u, label, v))
        return triplets

    def cumulative_triplets_upto(self, sentence_idx: int) -> List[Tuple[str, str, str]]:
        """
        Get cumulative triplets up to a given sentence index.
        """
        triplets = []
        for edge in self._graph.edges:
            if edge.created_at_sentence is not None:
                if edge.created_at_sentence <= sentence_idx:
                    triplets.append(
                        (
                            edge.source_node.get_text_representer(),
                            edge.label,
                            edge.dest_node.get_text_representer(),
                        )
                    )
            else:
                # No sentence info - include
                triplets.append(
                    (
                        edge.source_node.get_text_representer(),
                        edge.label,
                        edge.dest_node.get_text_representer(),
                    )
                )
        return triplets

    # =========================================================
    # EDGE KEY UTILITIES
    # =========================================================

    def edge_key(self, edge: "Edge") -> Tuple[str, str, str]:
        """Get canonical key for an edge."""
        return (
            edge.source_node.get_text_representer(),
            edge.label.lower().strip(),
            edge.dest_node.get_text_representer(),
        )

    def get_edge_activation_scores(self) -> Dict[Tuple[str, str, str], int]:
        """Get activation scores for all edges, keyed by (source, dest, label)."""
        scores = {}
        for edge in self._graph.edges:
            key = (
                edge.source_node.get_text_representer(),
                edge.dest_node.get_text_representer(),
                edge.label,
            )
            scores[key] = edge.activation_score
            # Also add 2-tuple key for compatibility
            scores[(key[0], key[1])] = edge.activation_score
        return scores

    # =========================================================
    # GRAPH RECORDING
    # =========================================================

    def record_edge_in_graphs(
        self,
        edge: "Edge",
        sentence_idx: Optional[int],
    ) -> None:
        """Record edge in cumulative and active tracking graphs."""
        src = edge.source_node.get_text_representer()
        dst = edge.dest_node.get_text_representer()
        label = edge.label

        # Add to cumulative graph
        self._cumulative_graph.add_edge(src, dst, label=label)

        # Track introduction sentence
        key = (src, label, dst)
        if key not in self._triplet_intro and sentence_idx is not None:
            self._triplet_intro[key] = sentence_idx

        # Add to active graph if edge is active
        if edge.active:
            self._active_graph.add_edge(src, dst, label=label)

    # =========================================================
    # SEMANTIC TRIPLET RECONSTRUCTION
    # =========================================================

    def reconstruct_semantic_triplets(
        self,
        only_active: bool = False,
        restrict_nodes: Optional[Set["Node"]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        AMoCv4 surface-relation format: edges ARE the semantic triplets.
        No reconstruction needed - just filter and return.
        """
        triplets = []

        for edge in self._graph.edges:
            if only_active and not edge.active:
                continue
            # Skip empty labels
            if not edge.label or not edge.label.strip():
                continue
            # Skip self-loops
            if edge.source_node == edge.dest_node:
                continue
            # Apply node restriction if provided
            if restrict_nodes is not None:
                if (
                    edge.source_node not in restrict_nodes
                    or edge.dest_node not in restrict_nodes
                ):
                    continue

            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
            )

        return triplets

    # =========================================================
    # FILTERED TRIPLETS FOR PLOTTING
    # =========================================================

    def get_filtered_triplets_for_plot(
        self, active_only: bool = True
    ) -> List[Tuple[str, str, str]]:
        """Get triplets filtered for visualization."""
        active_nodes, _ = self._graph.get_active_subgraph()
        active_names = {n.get_text_representer() for n in active_nodes}

        triplets = self.graph_edges_to_triplets(only_active=active_only)

        return [
            (s, r, o)
            for (s, r, o) in triplets
            if s in active_names and o in active_names
        ]

    # =========================================================
    # SENTENCE TRIPLET CAPTURE
    # =========================================================

    def capture_sentence_triplets(
        self,
        original_text: str,
        current_sentence_index: int,
        explicit_nodes: Set["Node"],
        nodes_with_active_edges: Set["Node"],
        sentence_triplets: List,
        anchor_drop_log: Optional[List] = None,
    ) -> None:
        current_nodes = explicit_nodes | nodes_with_active_edges
        for subj, rel, obj in self.reconstruct_semantic_triplets(
            only_active=False, restrict_nodes=current_nodes
        ):
            sentence_triplets.append(
                (
                    current_sentence_index,
                    original_text,
                    subj,
                    rel,
                    obj,
                    True,
                    True,
                    self._triplet_intro.get((subj, rel, obj), -1),
                )
            )

        if anchor_drop_log:
            for sent_idx, sent_text, subj, rel, obj in anchor_drop_log:
                sentence_triplets.append(
                    (
                        sent_idx,
                        sent_text,
                        subj,
                        rel,
                        obj,
                        False,
                        False,
                        -1,
                    )
                )
