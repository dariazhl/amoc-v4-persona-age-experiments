import logging
from typing import TYPE_CHECKING, List, Tuple, Set, Optional, Dict, Any

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge


class TripletRecorder:
    def __init__(
        self,
        graph_ref: "Graph",
        cumulative_graph_ref: Any,
        active_graph_ref: Any,
        triplet_intro_ref: Dict[Tuple[str, str, str], int],
    ):
        self._graph = graph_ref
        self._cumulative_graph = cumulative_graph_ref
        self._active_graph = active_graph_ref
        self._triplet_intro = triplet_intro_ref

    def graph_edges_to_triplets(
        self, only_active: bool = False
    ) -> List[Tuple[str, str, str]]:
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

    def cumulative_triplets_upto(self, sentence_idx: int) -> List[Tuple[str, str, str]]:
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

    def get_edge_activation_scores(self) -> Dict[Tuple[str, str, str], int]:
        scores = {}
        for edge in self._graph.edges:
            key = (
                edge.source_node.get_text_representer(),
                edge.dest_node.get_text_representer(),
                edge.label,
            )
            scores[key] = edge.visibility_score
            # Also add 2-tuple key for compatibility
            scores[(key[0], key[1])] = edge.visibility_score
        return scores

    # returns a list of all triplets currently in the graph
    def reconstruct_semantic_triplets(
        self,
        only_active: bool = False,
        restrict_nodes: Optional[Set["Node"]] = None,
    ) -> List[Tuple[str, str, str]]:
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

    def get_filtered_triplets_for_plot(
        self, active_only: bool = True
    ) -> List[Tuple[str, str, str]]:
        active_nodes, _ = self._graph.get_active_subgraph_wrapper()
        active_names = {n.get_text_representer() for n in active_nodes}

        triplets = self.graph_edges_to_triplets(only_active=active_only)

        return [
            (s, r, o)
            for (s, r, o) in triplets
            if s in active_names and o in active_names
        ]

    # list of (sentence_idx, sentence_text, subj, rel, obj, subj_active, obj_active, intro_sentence_idx) for all triplets in current sent
    def capture_sentence_triplets(
        self,
        original_text: str,
        current_sentence_index: int,
        explicit_nodes: Set["Node"],
        nodes_with_active_edges: Set["Node"],
        sentence_triplets: List,
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
