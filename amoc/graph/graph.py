from amoc.graph.node import Node
from amoc.graph.node import NodeType, NodeSource
from amoc.graph.edge import Edge
from collections import deque
from typing import List, Set, Dict, Optional, Tuple
import re


class Graph:
    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

    def add_or_get_node(
        self,
        lemmas: List[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
    ) -> Node:
        lemmas = [lemma.lower() for lemma in lemmas]
        actual_text_l = (actual_text or "").lower()
        node = self.get_node(lemmas)
        if node is None:
            node = Node(lemmas, actual_text_l, node_type, node_source, 0)
            self.nodes.add(node)
        else:
            node.add_actual_text(actual_text_l)
        return node

    def get_node(self, lemmas: List[str]) -> Optional[Node]:
        for node in self.nodes:
            if node.lemmas == lemmas:
                return node
        return None

    def get_edge_by_nodes_and_label(
        self, source_node: Node, dest_node: Node, label: str
    ) -> Optional[Edge]:
        for edge in self.edges:
            if (
                edge.source_node == source_node
                and edge.dest_node == dest_node
                and edge.label == label
            ):
                return edge
        return None

    def get_edge(self, edge: Edge) -> Optional[Edge]:
        for other_edge in self.edges:
            if edge == other_edge:
                return other_edge
        return None

    def add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
    ) -> Optional[Edge]:
        if source_node == dest_node:
            return None
        # Safety net: reject edges with empty/whitespace-only labels
        if not label or not isinstance(label, str) or not label.strip():
            return None
        edge = Edge(
            source_node,
            dest_node,
            label,
            edge_forget,
            active=True,
            created_at_sentence=created_at_sentence,
        )
        if self.check_if_similar_edge_exists(edge, edge_forget):
            return None
        self.edges.add(edge)
        if edge not in source_node.edges:
            source_node.edges.append(edge)
        if edge not in dest_node.edges:
            dest_node.edges.append(edge)

        return edge

    def check_if_similar_edge_exists(self, edge: Edge, edge_forget: int) -> bool:
        if edge in self.edges:
            self.get_edge(edge).forget_score = edge_forget
            self.get_edge(edge).active = True
            return True
        for other_edge in self.edges:
            same_nodes = (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
            )
            if same_nodes:
                # Merge concept↔property or label-similar edges between same nodes
                is_concept_property_pair = (
                    edge.source_node.node_type == NodeType.CONCEPT
                    and edge.dest_node.node_type == NodeType.PROPERTY
                ) or (
                    edge.dest_node.node_type == NodeType.CONCEPT
                    and edge.source_node.node_type == NodeType.PROPERTY
                )
                if is_concept_property_pair or edge.is_similar(other_edge):
                    other_edge.forget_score = edge_forget
                    other_edge.active = True
                    return True
        return False

    def deactivate_all_edges(self) -> None:
        """
        Deactivate all edges at the start of a new sentence.
        This implements the "cumulative memory, sentence-local activation" rule.
        Edges remain in memory but are marked inactive until reactivated.
        """
        for edge in self.edges:
            edge.deactivate()

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set[Node],
        max_distance: int,
        current_sentence: int,
    ) -> Set[Edge]:
        """
        Reactivate memory edges that are within max_distance of explicit sentence nodes.

        Property/attribute edges are only reactivated if current_sentence matches
        their origin_sentence (attributes attach only in their origin sentence).

        Returns the set of reactivated edges.
        """
        if not explicit_nodes or max_distance < 1:
            return set()

        # BFS to find nodes within distance
        reachable_nodes: Dict[Node, int] = {n: 0 for n in explicit_nodes}
        queue: deque = deque(explicit_nodes)
        visited_edges: Set[Edge] = set()
        reactivated: Set[Edge] = set()

        while queue:
            node = queue.popleft()
            dist = reachable_nodes[node]

            if dist >= max_distance:
                continue

            for edge in node.edges:
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                # Check if property edge should be reactivated
                # Property edges only attach in their origin sentence
                if edge.is_property_edge():
                    if edge.origin_sentence != current_sentence:
                        continue  # Skip - property edges don't reattach

                # Reactivate the edge
                edge.activate(reset_score=True)
                reactivated.add(edge)

                # Continue BFS to neighbors
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor not in reachable_nodes:
                    reachable_nodes[neighbor] = dist + 1
                    queue.append(neighbor)

        return reactivated

    def decay_inactive_edges(self) -> None:
        """
        Decay activation_score by 1 for all inactive edges.
        Called at the end of each sentence processing.
        """
        for edge in self.edges:
            edge.decay_activation()

    def get_active_subgraph(
        self, activation_threshold: int = 0
    ) -> Tuple[Set[Node], Set[Edge]]:
        """
        Get the active subgraph for layout computation.

        Only active structure shapes layout - inactive nodes/edges
        must not exert layout forces.

        Args:
            activation_threshold: Minimum activation_score for edge inclusion.
                                  0 means only currently active edges.

        Returns:
            Tuple of (active_nodes, active_edges) where:
            - active_edges: Edges that are active AND have activation_score > threshold
            - active_nodes: Nodes connected by active_edges
        """
        active_edges: Set[Edge] = set()
        active_nodes: Set[Node] = set()

        for edge in self.edges:
            if edge.active and edge.activation_score > activation_threshold:
                active_edges.add(edge)
                active_nodes.add(edge.source_node)
                active_nodes.add(edge.dest_node)

        return active_nodes, active_edges

    def get_active_triplets_with_scores(
        self,
    ) -> List[Tuple[str, str, str, bool, int]]:
        """
        Get triplets from active edges with their activation scores.

        Returns list of (source_text, label, dest_text, is_active, activation_score)
        for use in plotting with variable thickness/alpha.
        """
        triplets = []
        for edge in self.edges:
            triplets.append((
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
                edge.active,
                edge.activation_score,
            ))
        return triplets

    @staticmethod
    def canonicalize_relation_label(label: str) -> str:
        """
        Canonicalize relation labels before edge creation.
        Removes parser prefixes/artifacts and normalizes format.
        """
        if not label or not isinstance(label, str):
            return ""

        # Strip whitespace
        label = label.strip()

        # Remove common parser prefixes/artifacts
        prefixes_to_remove = [
            "nsubj:", "dobj:", "pobj:", "prep:", "amod:", "advmod:",
            "ROOT:", "VERB:", "NOUN:", "ADJ:", "dep:", "compound:",
            "agent:", "xcomp:", "ccomp:", "aux:", "auxpass:",
        ]
        for prefix in prefixes_to_remove:
            if label.lower().startswith(prefix.lower()):
                label = label[len(prefix):]

        # Remove trailing punctuation and artifacts
        label = re.sub(r'[^\w\s]$', '', label)
        label = label.strip()

        # Normalize whitespace
        label = re.sub(r'\s+', ' ', label)

        return label.lower()

    def bfs_from_activated_nodes(self, activated_nodes: List[Node]) -> Dict[Node, int]:
        distances = {}
        queue = deque([(node, 0) for node in activated_nodes])
        while queue:
            curr_node, curr_distance = queue.popleft()
            if curr_node not in distances:
                distances[curr_node] = curr_distance
                for edge in curr_node.edges:
                    if edge.active:
                        next_node = (
                            edge.dest_node
                            if edge.source_node == curr_node
                            else edge.source_node
                        )
                        queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List[Node]
    ) -> None:
        distances_to_activated_nodes = self.bfs_from_activated_nodes(activated_nodes)
        for node in self.nodes:
            node.score = distances_to_activated_nodes.get(node, 100)

    def get_word_lemma_score(self, word_lemmas: List[str]) -> Optional[float]:
        for node in self.nodes:
            if node.lemmas == word_lemmas:
                return node.score
        return None

    def get_top_k_nodes(self, nodes: List[Node], k: int) -> List[Node]:
        return sorted(nodes, key=lambda node: node.score)[:k]

    def get_top_concepts_nodes(self, k: int) -> List[Node]:
        nodes = [node for node in self.nodes if node.node_type == NodeType.CONCEPT]
        return self.get_top_k_nodes(nodes, k)

    def get_top_text_based_concepts(self, k: int) -> List[Node]:
        nodes = [
            node
            for node in self.nodes
            if node.node_type == NodeType.CONCEPT
            and node.node_source == NodeSource.TEXT_BASED
        ]
        return self.get_top_k_nodes(nodes, k)

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List[Node]:
        return [
            node
            for node in self.nodes
            if node.score <= score_threshold
            and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)
        ]

    def get_nodes_str(self, nodes: List[Node]) -> str:
        nodes_str = ""
        for node in sorted(nodes, key=lambda node: node.score):
            nodes_str += (
                "- "
                + f"{node.get_text_representer()} (type: {node.node_type}) (score: {node.score})"
                + "\n"
            )
        return nodes_str

    def get_edges_str(
        self, nodes: List[Node], only_text_based: bool = False, only_active: bool = True
    ) -> Tuple[str, List[Edge]]:
        used_edges = set()
        edges_str = ""
        count = 1
        for node in sorted(nodes, key=lambda node: node.score):
            for edge in node.edges:
                if only_active and edge.active == False:
                    continue
                if edge not in used_edges:
                    if not only_text_based:
                        edges_str += (
                            f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                            + "\n"
                        )
                        used_edges.add(edge)
                        count += 1
                    else:
                        if (
                            edge.source_node.node_source == NodeSource.TEXT_BASED
                            and edge.dest_node.node_source == NodeSource.TEXT_BASED
                        ):
                            edges_str += (
                                f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                                + "\n"
                            )
                            used_edges.add(edge)
                            count += 1
        return edges_str, list(used_edges)

    def get_active_graph_repr(self) -> str:
        edges = [edge for edge in self.edges if edge.active]
        nodes = set()
        for edge in edges:
            nodes.add(edge.source_node)
            nodes.add(edge.dest_node)
        s = "nodes:\n"
        for node in nodes:
            s += str(node) + "\n"
        s += "\nedges:\n"
        for edge in edges:
            s += str(edge) + "\n"
        return s

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()
