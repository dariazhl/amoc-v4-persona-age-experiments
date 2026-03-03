from amoc.graph.node import Node
from amoc.graph.node import NodeType, NodeSource, NodeProvenance
from amoc.graph.edge import Edge
from amoc.graph.node_activation_engine import NodeActivationEngine
from amoc.graph.stability_ops import StabilityOps
from amoc.graph.provenance_ops import ProvenanceOps
from typing import List, Set, Dict, Optional, Tuple, Callable
import logging
import networkx as nx


class Graph:
    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

        self._story_lemmas: Optional[Set[str]] = None
        self._persona_only_lemmas: Optional[Set[str]] = None
        self._current_sentence_idx: int = 0
        self._current_sentence_lemmas: Optional[Set[str]] = None

        self._activation_ops = NodeActivationEngine(self)
        self._stability_ops = StabilityOps(self)
        self._provenance_ops = ProvenanceOps(self)

    def set_current_sentence_lemmas(self, lemmas: Set[str]) -> None:
        self._current_sentence_lemmas = {l.lower() for l in lemmas}

    def set_current_sentence(self, sentence_idx: int) -> None:
        self._current_sentence_idx = sentence_idx

    def add_or_get_node(
        self,
        lemmas: list[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
        *,
        admit: Optional[Callable] = None,
        admit_kwargs: dict | None = None,
        origin_sentence: Optional[int] = None,
        provenance: Optional[NodeProvenance] = None,
        mark_explicit: bool = True,
    ):
        lemmas = [lemma.lower() for lemma in lemmas]
        primary_lemma = lemmas[0].lower() if lemmas else ""

        existing_node = self.get_node(lemmas)
        is_new_node = existing_node is None

        valid, reason = self._provenance_ops.validate_node_creation(
            lemmas=lemmas,
            node_type=node_type,
            provenance=provenance,
            is_new_node=is_new_node,
        )
        if not valid:
            return None

        if admit is not None:
            admit_kwargs = admit_kwargs or {}
            if not admit(lemma=lemmas[0], node_type=node_type, **admit_kwargs):
                return None

        actual_text_l = (actual_text or "").lower()

        node = existing_node
        if node is None:
            node = Node(
                lemmas,
                actual_text_l,
                node_type,
                node_source,
                0,
                origin_sentence=origin_sentence,
                provenance=provenance or NodeProvenance.STORY_TEXT,
            )
            self.nodes.add(node)
            if mark_explicit and origin_sentence is not None:
                node.mark_explicit_in_sentence(origin_sentence)
        else:
            node.add_actual_text(actual_text_l)
            if node.node_type != node_type:
                if node_type == NodeType.PROPERTY:
                    node.node_type = NodeType.PROPERTY

            if mark_explicit and origin_sentence is not None:
                if not self._provenance_ops.validate_explicit_marking(primary_lemma):
                    return None
                node.mark_explicit_in_sentence(origin_sentence)

        return node

    def get_node(self, lemmas: List[str]) -> Optional[Node]:
        for node in self.nodes:
            if node.lemmas == lemmas:
                return node
        return None

    def get_explicit_nodes_for_sentence(self, sentence_id: int):
        return [
            node for node in self.nodes if node.is_explicit_in_sentence(sentence_id)
        ]

    def add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_visibility: int,
        created_at_sentence: Optional[int] = None,
        *,
        relation_class=None,
        justification=None,
        persona_influenced: bool = False,
        inferred: bool = False,
    ) -> Optional[Edge]:
        if source_node == dest_node:
            return None
        if not label or not isinstance(label, str) or not label.strip():
            return None

        if inferred:
            # Prevent inferred edges from attaching to decayed nodes
            source_visibility_score = getattr(source_node, "visibility_score", None)
            dest_visibility_score = getattr(dest_node, "visibility_score", None)

            if (
                source_visibility_score is not None and source_visibility_score <= 0
            ) or (dest_visibility_score is not None and dest_visibility_score <= 0):
                return None

        edge = Edge(
            source_node,
            dest_node,
            label,
            edge_visibility,
            relation_class=relation_class,
            justification=justification,
            persona_influenced=persona_influenced,
            inferred=inferred,
            active=True,
            created_at_sentence=created_at_sentence,
        )

        existing = self._activation_ops.find_and_reinforce_similar_edge(
            edge, edge_visibility
        )
        if existing is not None:
            return existing

        self.edges.add(edge)
        if edge not in source_node.edges:
            source_node.edges.append(edge)
        if edge not in dest_node.edges:
            dest_node.edges.append(edge)

        return edge

    def get_edge(self, edge: Edge) -> Optional[Edge]:
        for other_edge in self.edges:
            if edge == other_edge:
                return other_edge
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

    def remove_edge(self, edge: Edge) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
        if edge.source_node and edge in edge.source_node.edges:
            edge.source_node.edges.remove(edge)
        if edge.dest_node and edge in edge.dest_node.edges:
            edge.dest_node.edges.remove(edge)

    def get_active_subgraph(self) -> Tuple[Set[Node], Set[Edge]]:
        return self._activation_ops.get_active_subgraph()

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source_node, edge.dest_node, edge=edge)
        return G

    def get_word_lemma_score(self, word_lemmas: List[str]) -> Optional[float]:
        for node in self.nodes:
            if node.lemmas == word_lemmas:
                return node.score
        return None

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

    def deactivate_all_edges(self) -> None:
        self._activation_ops.deactivate_all_edges()

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set[Node],
        max_distance: int,
        current_sentence: int,
    ) -> Set[Edge]:
        return self._activation_ops.reactivate_memory_edges_within_distance(
            explicit_nodes, max_distance, current_sentence
        )

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List[Node]:
        return self._activation_ops.get_active_nodes(score_threshold, only_text_based)

    def decay_inactive_edges(self) -> None:
        self._activation_ops.decay_inactive_edges()

    def enforce_cumulative_stability(self, explicit_nodes: set) -> None:
        self._stability_ops.enforce_cumulative_stability(explicit_nodes)

    def enforce_carryover_connectivity(self, carryover_nodes: set) -> None:
        self._stability_ops.enforce_carryover_connectivity(carryover_nodes)

    def is_active_connected(self, required_nodes: Optional[Set[Node]] = None) -> bool:
        return self._stability_ops.compute_active_connectivity(required_nodes)

    def is_cumulative_connected(self) -> bool:
        return self._stability_ops.compute_cumulative_connectivity()

    def get_disconnected_components(
        self, focus_nodes: Set[Node]
    ) -> Tuple[List[Set[Node]], int]:
        return self._stability_ops.get_disconnected_components(focus_nodes)

    def can_connect_via_cumulative(self, required_nodes: Set[Node]) -> bool:
        return self._stability_ops.can_connect_via_cumulative(required_nodes)

    def reconnect_via_cumulative(self, required_nodes: Set[Node]) -> Set[Edge]:
        return self._stability_ops.reconnect_via_cumulative(required_nodes)

    def enforce_connectivity(
        self,
        required_nodes: Set[Node],
        allow_reactivation: bool = True,
        enforce_cumulative: bool = False,
    ) -> bool:
        return self._stability_ops.reactivate_to_restore_connectivity(
            required_nodes, allow_reactivation, enforce_cumulative
        )

    def set_provenance_gate(
        self,
        story_lemmas: Set[str],
        persona_only_lemmas: Optional[Set[str]] = None,
    ) -> None:
        self._provenance_ops.set_provenance_gate(story_lemmas, persona_only_lemmas)

    def sanity_check_provenance(
        self,
        story_lemmas: set,
        persona_only_lemmas: set,
    ) -> list:
        return self._provenance_ops.sanity_check_provenance(
            story_lemmas, persona_only_lemmas
        )

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()
