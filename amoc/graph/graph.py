"""
Graph - Pure Topology Module

This module contains ONLY topology logic:
- Node/edge storage and retrieval
- Graph structure operations
- Connectivity checks (pure, non-mutating)

All other logic has been moved to:
- provenance_ops.py: Provenance gate and sanity checks
- validation_ops.py: AMoCv4 constraints and ontology validation
- activation_ops.py: Edge reactivation, decay, scoring
- plot_filter_ops.py: Edge filtering for visualization
- stability_ops.py: Cumulative stability enforcement
"""

from amoc.graph.node import Node
from amoc.graph.node import NodeType, NodeSource, NodeProvenance, NodeRole
from amoc.graph.edge import Edge
from typing import List, Set, Dict, Optional, Tuple, Callable
import re
import logging
import networkx as nx


class Graph:
    """
    Pure topology graph structure.

    Contains only:
    - Node/edge storage
    - Add/remove operations
    - Connectivity checks (pure, non-mutating)
    """

    # ==========================================================================
    # FORBIDDEN NODE LEMMAS (Hard Block)
    # ==========================================================================
    FORBIDDEN_NODE_LEMMAS: set[str] = {
        "student", "persona", "relation", "context", "object", "place",
        "story", "narrative", "sentence", "edge", "node", "property",
        "label", "target", "source", "pronoun", "noun", "user",
    }

    NARRATION_ARTIFACT_LEMMAS: set[str] = {
        "text", "sentence", "paragraph", "mention", "mentions", "narration", "story",
    }

    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

        # Provenance gate state (used by add_or_get_node)
        self._story_lemmas: Optional[Set[str]] = None
        self._persona_only_lemmas: Optional[Set[str]] = None
        self._current_sentence_idx: int = 0
        self._current_sentence_lemmas: Optional[Set[str]] = None

    # ==========================================================================
    # NODE OPERATIONS
    # ==========================================================================

    def set_current_sentence_lemmas(self, lemmas: Set[str]) -> None:
        """Set the lemma set for the current sentence."""
        self._current_sentence_lemmas = {l.lower() for l in lemmas}

    def set_current_sentence(self, sentence_idx: int) -> None:
        """Set the current sentence index."""
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
        node_role: Optional[NodeRole] = None,
        mark_explicit: bool = True,
    ):
        """Add a new node or get existing node with matching lemmas."""
        lemmas = [lemma.lower() for lemma in lemmas]
        if not lemmas or not lemmas[0]:
            return None
        primary_lemma = lemmas[0].lower()

        if len(primary_lemma) <= 1:
            return None

        GARBAGE_LEMMAS = {
            "edge", "node", "relation", "property", "label", "target",
            "source", "t", "type", "person", "approach", "kind", "thing",
        }

        if primary_lemma in GARBAGE_LEMMAS:
            return None

        if node_type != NodeType.EVENT and not re.match(r"^[a-zA-Z]+$", lemmas[0]):
            return None

        if any(lemma in self.FORBIDDEN_NODE_LEMMAS for lemma in lemmas):
            return None

        if any(lemma in self.NARRATION_ARTIFACT_LEMMAS for lemma in lemmas):
            return None

        if self._persona_only_lemmas and primary_lemma in self._persona_only_lemmas:
            return None

        existing_node = self.get_node(lemmas)
        if existing_node is None:
            if self._story_lemmas is not None:
                ed_stem = (
                    primary_lemma[:-2]
                    if primary_lemma.endswith("ed") and len(primary_lemma) > 2
                    else None
                )
                ing_stem = (
                    primary_lemma[:-3]
                    if primary_lemma.endswith("ing") and len(primary_lemma) > 3
                    else None
                )
                is_story_grounded = (
                    primary_lemma in self._story_lemmas
                    or (ed_stem and ed_stem in self._story_lemmas)
                    or (ing_stem and ing_stem in self._story_lemmas)
                )
                is_inferred = provenance == NodeProvenance.INFERRED_FROM_STORY

                if (
                    node_type != NodeType.PROPERTY
                    and not is_story_grounded
                    and not is_inferred
                ):
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
                node_role=node_role,
            )
            self.nodes.add(node)
            if mark_explicit and origin_sentence is not None:
                node.mark_explicit_in_sentence(origin_sentence)
        else:
            node.add_actual_text(actual_text_l)
            if node.node_type != node_type:
                if node_type == NodeType.PROPERTY:
                    node.node_type = NodeType.PROPERTY
            if node.node_role is None and node_role is not None:
                node.node_role = node_role

            if mark_explicit and origin_sentence is not None:
                if (
                    self._current_sentence_lemmas is not None
                    and primary_lemma not in self._current_sentence_lemmas
                ):
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

    # ==========================================================================
    # EDGE OPERATIONS
    # ==========================================================================

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
        """Add an edge to the graph."""
        if source_node == dest_node:
            return None
        if not label or not isinstance(label, str) or not label.strip():
            return None

        if inferred:
            if (
                hasattr(source_node, "visibility_score")
                and source_node.visibility_score <= 0
            ) or (
                hasattr(dest_node, "visibility_score")
                and dest_node.visibility_score <= 0
            ):
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

        if self._check_similar_edge_exists(edge, edge_visibility):
            return self._get_similar_edge(edge)

        self.edges.add(edge)
        if edge not in source_node.edges:
            source_node.edges.append(edge)
        if edge not in dest_node.edges:
            dest_node.edges.append(edge)

        return edge

    def _check_similar_edge_exists(self, edge: Edge, edge_visibility: int) -> bool:
        """Check if a similar edge exists and reinforce it."""
        for other_edge in self.edges:
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

    def _get_similar_edge(self, edge: Edge) -> Optional[Edge]:
        """Get a similar existing edge."""
        for other_edge in self.edges:
            if (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
                and edge.label.strip().lower() == other_edge.label.strip().lower()
            ):
                return other_edge
        return None

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

    # ==========================================================================
    # SUBGRAPH OPERATIONS
    # ==========================================================================

    def get_active_subgraph(self) -> Tuple[Set[Node], Set[Edge]]:
        """Get active nodes and edges."""
        active_edges: Set[Edge] = {
            e for e in self.edges if e.active and e.visibility_score > 0
        }
        active_nodes: Set[Node] = {e.source_node for e in active_edges} | {
            e.dest_node for e in active_edges
        }
        return active_nodes, active_edges

    # ==========================================================================
    # GRAPH VIEW BUILDERS (Pure, Non-Mutating)
    # ==========================================================================

    def cumulative_graph(self) -> nx.Graph:
        """
        Build a read-only NetworkX graph from ALL edges in cumulative memory.

        This is a snapshot view - modifications to the returned graph
        do NOT affect the underlying Graph structure.
        """
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source_node, edge.dest_node, edge=edge)
        return G

    def active_graph(self, required_nodes: Optional[Set[Node]] = None) -> nx.Graph:
        """
        Build a read-only NetworkX graph from active edges only.
        """
        G = nx.Graph()

        active_nodes = {n for n in self.nodes if n.active}
        for edge in self.edges:
            if edge.active:
                active_nodes.add(edge.source_node)
                active_nodes.add(edge.dest_node)
        if required_nodes:
            active_nodes |= required_nodes

        for node in active_nodes:
            G.add_node(node)

        for edge in self.edges:
            if edge.active:
                G.add_edge(edge.source_node, edge.dest_node, edge=edge)

        return G

    # ==========================================================================
    # CONNECTIVITY CHECKS (Pure, Non-Mutating)
    # ==========================================================================

    def check_active_connectivity(self) -> bool:
        """
        Check if the active graph is connected.
        Returns True if connected (or empty/single node), False if disconnected.
        """
        active_nodes = {n for n in self.nodes if n.active}
        active_edges = [e for e in self.edges if e.active]

        # Also include nodes from active edges
        for edge in active_edges:
            active_nodes.add(edge.source_node)
            active_nodes.add(edge.dest_node)

        if len(active_nodes) <= 1:
            return True

        G_active = nx.Graph()
        for node in active_nodes:
            G_active.add_node(node)
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node)

        return nx.is_connected(G_active)

    def check_cumulative_connectivity(self) -> bool:
        """Check if cumulative graph is connected."""
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source_node, edge.dest_node)
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)

    def get_disconnected_components(
        self,
        focus_nodes: Set[Node],
    ) -> Tuple[List[Set[Node]], int]:
        """
        Get disconnected components and identify the focus component.
        """
        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return ([], -1)

        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        if G_active.number_of_nodes() <= 1:
            return (
                ([set(G_active.nodes())], 0)
                if G_active.number_of_nodes() == 1
                else ([], -1)
            )

        components = [set(c) for c in nx.connected_components(G_active)]

        if len(components) <= 1:
            return (components, 0)

        focus_component_idx = 0
        for idx, comp in enumerate(components):
            if any(n in comp for n in focus_nodes):
                focus_component_idx = idx
                break

        return (components, focus_component_idx)

    def get_nodes_needing_connection(
        self,
        focus_nodes: Set[Node],
    ) -> List[Tuple[Node, Node]]:
        """
        Get pairs of nodes that need edges to restore connectivity.
        """
        components, focus_idx = self.get_disconnected_components(focus_nodes)

        if len(components) <= 1:
            return []

        focus_comp = components[focus_idx]
        pairs_needing_connection = []

        for idx, comp in enumerate(components):
            if idx == focus_idx:
                continue

            isolated_node = None
            for n in comp:
                if n.node_type == NodeType.CONCEPT:
                    isolated_node = n
                    break
            if isolated_node is None:
                isolated_node = next(iter(comp))

            focus_node = None
            for n in focus_comp:
                if n in focus_nodes and n.node_type == NodeType.CONCEPT:
                    focus_node = n
                    break
            if focus_node is None:
                for n in focus_comp:
                    if n.node_type == NodeType.CONCEPT:
                        focus_node = n
                        break
            if focus_node is None:
                focus_node = next(iter(focus_comp))

            pairs_needing_connection.append((isolated_node, focus_node))

        return pairs_needing_connection

    def ensure_active_connectivity(
        self,
        required_nodes: Set[Node],
    ) -> bool:
        """
        Check if the active graph can be connected using existing edges.

        This is a PURE CHECK that does NOT modify any edges. It determines
        whether connectivity is achievable by examining paths in the
        cumulative graph, but does not activate or promote any edges.

        Returns:
            True if graph is connected (or could be connected via existing
            cumulative edges), False otherwise.
        """
        active_edges = [e for e in self.edges if e.active]

        # Build active_nodes from:
        # 1. Nodes with n.active == True
        # 2. Nodes connected by active edges
        # 3. Required nodes (explicit + carryover)
        active_nodes = {n for n in self.nodes if n.active}
        for edge in active_edges:
            active_nodes.add(edge.source_node)
            active_nodes.add(edge.dest_node)
        active_nodes |= required_nodes

        # Build active graph using only edge.active == True
        G_active = nx.Graph()

        for node in active_nodes:
            G_active.add_node(node)

        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node)

        # If connected → return True
        if G_active.number_of_nodes() <= 1:
            return True
        if nx.is_connected(G_active):
            return True

        # Graph is disconnected - find components
        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return True

        # Identify focus component (largest component containing required nodes)
        focus_component_idx = 0
        max_required_count = 0
        for idx, comp in enumerate(components):
            required_in_comp = len(comp & required_nodes)
            if required_in_comp > max_required_count:
                max_required_count = required_in_comp
                focus_component_idx = idx

        # Build cumulative graph from ALL edges
        G_cumulative = nx.Graph()
        for edge in self.edges:
            G_cumulative.add_edge(edge.source_node, edge.dest_node, edge=edge)

        # Local connectors list (edges that WOULD connect components)
        connectors: Set[Edge] = set()

        # For each disconnected component, find shortest path to focus component
        focus_comp_nodes = components[focus_component_idx]
        for idx, comp in enumerate(components):
            if idx == focus_component_idx:
                continue

            best_path = None
            best_path_len = float("inf")

            for src in comp:
                if src not in G_cumulative:
                    continue
                for tgt in focus_comp_nodes:
                    if src == tgt:
                        continue
                    if tgt not in G_cumulative:
                        continue

                    try:
                        path = nx.shortest_path(G_cumulative, src, tgt)
                        if len(path) < best_path_len:
                            best_path = path
                            best_path_len = len(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            if best_path is None:
                # No cumulative path exists → return False
                return False

            # Collect edges along path into local connectors (DO NOT modify)
            for i in range(len(best_path) - 1):
                node_a = best_path[i]
                node_b = best_path[i + 1]

                edge_data = G_cumulative.get_edge_data(node_a, node_b)
                if edge_data is None:
                    continue

                edge = edge_data.get("edge")
                if edge is None:
                    continue

                if not edge.active:
                    connectors.add(edge)

        # Build final temporary graph using active edges + connectors
        G_final = nx.Graph()

        for node in active_nodes:
            G_final.add_node(node)

        for edge in active_edges:
            G_final.add_edge(edge.source_node, edge.dest_node)

        for edge in connectors:
            G_final.add_edge(edge.source_node, edge.dest_node)

        if G_final.number_of_nodes() <= 1:
            return True

        return nx.is_connected(G_final)

    # ==========================================================================
    # UTILITY METHODS
    # ==========================================================================

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

    def get_active_triplets_with_scores(
        self,
    ) -> List[Tuple[str, str, str, bool, int]]:
        """Get triplets from active edges with their activation scores."""
        triplets = []
        for edge in self.edges:
            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                    edge.active,
                    edge.activation_score,
                )
            )
        return triplets

    @staticmethod
    def canonicalize_relation_label(label: str) -> str:
        """Canonicalize relation labels before edge creation."""
        if not label or not isinstance(label, str):
            return ""

        label = label.strip()
        if not label:
            return ""

        prefixes_to_remove = [
            "nsubj:", "dobj:", "pobj:", "prep:", "amod:", "advmod:",
            "ROOT:", "VERB:", "NOUN:", "ADJ:", "dep:", "compound:",
            "agent:", "xcomp:", "ccomp:", "aux:", "auxpass:",
        ]
        for prefix in prefixes_to_remove:
            if label.lower().startswith(prefix.lower()):
                label = label[len(prefix):]

        label = re.sub(r"[^\w\s]+$", "", label)
        label = label.strip()
        label = re.sub(r"\s+", " ", label)

        if len(label) > 0:
            if re.search(r"(.)\1{2,}", label):
                label = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+$", r"\1", label)

            words = label.split()
            cleaned_words = []
            for word in words:
                if len(word) <= 2:
                    cleaned_words.append(word.lower())
                    continue
                if not re.search(r"[aeiou]", word.lower()):
                    continue
                cleaned_words.append(word.lower())

            if not cleaned_words:
                return ""
            label = " ".join(cleaned_words)

        label = label.lower().strip()

        if len(label) < 2:
            return ""

        return label

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()
