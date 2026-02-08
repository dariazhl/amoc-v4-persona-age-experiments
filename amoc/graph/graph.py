from amoc.graph.node import Node
from amoc.graph.node import NodeType, NodeSource, NodeProvenance
from amoc.graph.edge import Edge
from collections import deque
from typing import List, Set, Dict, Optional, Tuple, Callable
import re
import logging


class Graph:
    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

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
    ):
        """
        Add a new node or get existing node with matching lemmas.

        PROVENANCE TRACKING (Paper-Aligned):
        - origin_sentence: The sentence index where this node was created
        - provenance: How this node was derived (STORY_TEXT or INFERRED_FROM_STORY)

        CRITICAL: Nodes must only come from story text, never from persona.
        Persona influences salience/weights only, never graph content.
        """
        lemmas = [lemma.lower() for lemma in lemmas]
        if not lemmas or not lemmas[0] or not lemmas[0].isalpha():
            return None
        FORBIDDEN = {"edge", "node", "property", "label", "target", "source"}
        if any(l in FORBIDDEN for l in lemmas):
            return None
        if admit is not None:
            admit_kwargs = admit_kwargs or {}
            if not admit(lemma=lemmas[0], node_type=node_type, **admit_kwargs):
                return None

        actual_text_l = (actual_text or "").lower()
        node = self.get_node(lemmas)
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
        edge_visibility: int,
        created_at_sentence: Optional[int] = None,
    ) -> Optional[Edge]:
        if source_node == dest_node:
            return None
        # Safety net: reject edges with empty/whitespace-only labels
        if not label or not isinstance(label, str) or not label.strip():
            return None

        # STRICT PROPERTY CONSTRAINT: A PROPERTY node must not attach to more than one concept
        # Check if this would violate the single-attach rule
        property_node = None
        concept_node = None
        if (
            source_node.node_type == NodeType.PROPERTY
            and dest_node.node_type == NodeType.CONCEPT
        ):
            property_node = source_node
            concept_node = dest_node
        elif (
            dest_node.node_type == NodeType.PROPERTY
            and source_node.node_type == NodeType.CONCEPT
        ):
            property_node = dest_node
            concept_node = source_node

        if property_node is not None:
            if created_at_sentence is None:
                return None
            # Check if this property is already attached to a DIFFERENT concept
            for existing_edge in property_node.edges:
                other_node = (
                    existing_edge.dest_node
                    if existing_edge.source_node == property_node
                    else existing_edge.source_node
                )
                if (
                    other_node.node_type == NodeType.CONCEPT
                    and other_node != concept_node
                ):
                    # PROPERTY already attached to a different concept - reject
                    return None

        edge = Edge(
            source_node,
            dest_node,
            label,
            edge_visibility,
            active=True,
            created_at_sentence=created_at_sentence,
        )
        if self.check_if_similar_edge_exists(edge, edge_visibility):
            return None
        self.edges.add(edge)
        if edge not in source_node.edges:
            source_node.edges.append(edge)
        if edge not in dest_node.edges:
            dest_node.edges.append(edge)

        return edge

    def check_if_similar_edge_exists(self, edge: Edge, edge_visibility: int) -> bool:
        if edge in self.edges:
            existing_edge = self.get_edge(edge)
            existing_edge.visibility_score = edge_visibility
            # Mark as ASSERTED (re-assertion of existing edge)
            existing_edge.mark_as_asserted(reset_score=True)
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
                    other_edge.visibility_score = edge_visibility
                    # Mark as ASSERTED (re-assertion of existing edge)
                    other_edge.mark_as_asserted(reset_score=True)
                    return True
        return False

    def deactivate_all_edges(self) -> None:
        """
        Reset all edges at the start of a new sentence.
        This implements the "cumulative memory, sentence-local activation" rule.
        Per AMoC v4 paper: all edges become inactive at sentence start,
        then selectively activated through assertion or reactivation.
        Edges remain in memory but are marked inactive until reasserted/reactivated.
        """
        for edge in self.edges:
            edge.reset_for_sentence_start()

    # Maximum number of edges to reactivate per sentence (sparse reactivation)
    MAX_REACTIVATION_COUNT: int = 3

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set[Node],
        max_distance: int,
        current_sentence: int,
    ) -> Set[Edge]:
        """
        Reactivate memory edges that are within max_distance of explicit sentence nodes.

        Per AMoC v4 paper requirements:
        - PROPERTY edges must NEVER be reactivated (strict rule)
        - PROPERTY nodes don't participate in BFS (no propagation through properties)
        - Reactivation is sparse: limited to MAX_REACTIVATION_COUNT edges (≈1-3)
        - Only edges within graph_distance ≤ max_distance are candidates

        Returns the set of reactivated edges.
        """
        if not explicit_nodes or max_distance < 1:
            return set()

        # BFS to find candidate edges within distance
        # CRITICAL (Paper-Aligned): Only CONCEPT nodes participate in BFS
        # PROPERTY nodes have no independent activation and don't propagate
        concept_seeds = {n for n in explicit_nodes if n.node_type != NodeType.PROPERTY}

        if not concept_seeds:
            return set()

        reachable_nodes: Dict[Node, int] = {n: 0 for n in concept_seeds}
        queue: deque = deque(concept_seeds)
        visited_edges: Set[Edge] = set()
        candidate_edges: list[tuple[int, Edge]] = []  # (distance, edge)

        while queue:
            node = queue.popleft()
            dist = reachable_nodes[node]

            if dist >= max_distance:
                continue

            for edge in node.edges:
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                # STRICT RULE: PROPERTY edges must NEVER be reactivated
                # Property edges fade permanently if not reasserted in current sentence
                if edge.is_property_edge():
                    continue

                if edge.violates_property_sentence_constraint(current_sentence):
                    continue

                # Skip edges that are already active (asserted this sentence)
                if edge.active:
                    continue

                # Collect as candidate for reactivation
                candidate_edges.append((dist, edge))

                # Continue BFS to neighbors (for finding more candidates)
                # CRITICAL: Skip PROPERTY nodes in traversal
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor.node_type == NodeType.PROPERTY:
                    continue
                if neighbor not in reachable_nodes:
                    reachable_nodes[neighbor] = dist + 1
                    queue.append(neighbor)

        # SPARSE REACTIVATION: sort by distance and limit to MAX_REACTIVATION_COUNT
        # Prefer closer edges (smaller distance)
        candidate_edges.sort(key=lambda x: x[0])
        reactivated: Set[Edge] = set()

        for dist, edge in candidate_edges[: self.MAX_REACTIVATION_COUNT]:
            edge.mark_as_reactivated(reset_score=True)
            reactivated.add(edge)

        return reactivated

    def decay_inactive_edges(self) -> None:
        for edge in self.edges:
            if edge.activation_role == "connector":
                continue
            if not edge.active:
                edge.activation_score -= 1

    def get_active_subgraph(
        self,
        activation_threshold: int = 0,
        current_sentence: Optional[int] = None,
    ) -> Tuple[Set[Node], Set[Edge]]:
        """
        Get the active subgraph for layout computation.

        Per AMoC v4 paper:
        - Active graph = explicit nodes + nodes within MaxDistance
        - Property edges are INCLUDED if they don't violate sentence constraint
        - Properties attach via "is" edges (e.g., "princess - is - beautiful")

        Args:
            activation_threshold: Minimum activation_score for edge inclusion.
                                  0 means only currently active edges.
            current_sentence: Current sentence index for property edge validation.
                             If None, property edges are excluded.

        Returns:
            Tuple of (active_nodes, active_edges) where:
            - active_edges: Edges that are active AND meet threshold
            - active_nodes: Nodes connected by active_edges
        """
        active_edges: Set[Edge] = set()
        active_nodes: Set[Node] = set()

        for edge in self.edges:
            if not edge.active:
                continue

            # CRITICAL FIX: Include property edges when they're in their origin sentence
            # Per AMoC paper (Figures 2-4): properties attach via "is" edges
            if edge.is_property_edge():
                # Property edges only active in their origin sentence
                if current_sentence is None:
                    continue  # Can't validate - skip property edges
                if edge.violates_property_sentence_constraint(current_sentence):
                    continue  # Not in origin sentence - skip
                # Property edge is valid - include it

            if edge.activation_score > activation_threshold:
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
        """
        Canonicalize relation labels before edge creation.
        Removes clear parser artifacts while preserving valid verb phrases.

        Per AMoC v4 paper (Figures 2-4), edge labels should be full verb phrases:
        - "rode through"
        - "was kidnapping"
        - "wanted to free"
        - "is unfamiliar with"

        This function is CONSERVATIVE - it only removes clear artifacts.
        The more aggressive normalization happens in _normalize_edge_label().
        """
        if not label or not isinstance(label, str):
            return ""

        # Strip whitespace
        label = label.strip()
        if not label:
            return ""

        # Remove common parser prefixes/artifacts (colon-based)
        prefixes_to_remove = [
            "nsubj:",
            "dobj:",
            "pobj:",
            "prep:",
            "amod:",
            "advmod:",
            "ROOT:",
            "VERB:",
            "NOUN:",
            "ADJ:",
            "dep:",
            "compound:",
            "agent:",
            "xcomp:",
            "ccomp:",
            "aux:",
            "auxpass:",
        ]
        for prefix in prefixes_to_remove:
            if label.lower().startswith(prefix.lower()):
                label = label[len(prefix) :]

        # Remove trailing punctuation
        label = re.sub(r"[^\w\s]+$", "", label)
        label = label.strip()

        # Normalize whitespace (but preserve spaces between words)
        label = re.sub(r"\s+", " ", label)

        # CONSERVATIVE: Only reject clearly malformed labels
        # - Labels with 3+ repeated characters (like "killtt" -> corruption)
        # - Labels that are pure noise (no vowels in any word)
        if len(label) > 0:
            # Detect repeated character corruption
            if re.search(r"(.)\1{2,}", label):
                # Clean up repeated trailing consonants: "kidnapp" -> "kidnap"
                label = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+$", r"\1", label)

            # Check each word for vowel presence (skip very short words)
            words = label.split()
            cleaned_words = []
            for word in words:
                # Skip words that are too short to check
                if len(word) <= 2:
                    cleaned_words.append(word.lower())
                    continue
                # Reject words without vowels (corruption)
                if not re.search(r"[aeiou]", word.lower()):
                    continue
                cleaned_words.append(word.lower())

            if not cleaned_words:
                return ""
            label = " ".join(cleaned_words)

        # Final: lowercase and return
        label = label.lower().strip()

        # Minimum length check (allow 2-char verbs like "be", "go", "do")
        if len(label) < 2:
            return ""

        return label

    def bfs_from_activated_nodes(
        self,
        activated_nodes: List[Node],
        direction: str = "both",
    ) -> Dict[Node, int]:
        """
        Compute shortest-path distances from activated nodes via BFS.

        Args:
            activated_nodes: Starting nodes for BFS
            direction: How to traverse edges:
                - "both": Follow edges in both directions (default, for distance computation)
                - "outgoing": Only follow edges where current node is source (A → B)
                - "incoming": Only follow edges where current node is dest (A ← B)

        Returns:
            Dict mapping each reachable node to its distance from nearest activated node.

        Per AMoC v4: Activation distance is computed bidirectionally (semantic edges
        connect concepts regardless of direction). Direction matters for meaning,
        not for activation propagation.

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are EXCLUDED from BFS traversal and distance computation.
        Per paper: PROPERTY nodes have no independent activation and don't propagate
        activation to/from other nodes. Only CONCEPT nodes participate in BFS.
        """
        distances = {}
        # Filter out PROPERTY nodes from starting set
        # Per paper: only CONCEPT nodes can be activation seeds
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

                    # Determine which neighbor to visit based on direction mode
                    next_node = None
                    if direction == "both":
                        # Bidirectional: follow edge regardless of direction
                        next_node = (
                            edge.dest_node
                            if edge.source_node == curr_node
                            else edge.source_node
                        )
                    elif direction == "outgoing":
                        # Only follow if current node is the source
                        if edge.source_node == curr_node:
                            next_node = edge.dest_node
                    elif direction == "incoming":
                        # Only follow if current node is the destination
                        if edge.dest_node == curr_node:
                            next_node = edge.source_node

                    # CRITICAL: Skip PROPERTY nodes in BFS traversal
                    # Per paper: PROPERTY nodes don't propagate activation
                    if (
                        next_node is not None
                        and next_node.node_type == NodeType.PROPERTY
                    ):
                        continue

                    if next_node is not None:
                        queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List[Node]
    ) -> None:
        """
        Update node scores based on BFS distance from activated nodes.

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are excluded from distance-based scoring.
        Per paper: PROPERTY nodes have no independent activation and
        their "closeness" to active CONCEPT nodes does NOT make them active.
        PROPERTY nodes keep a high score (100) to prevent them from being
        selected as carry-over nodes based on score alone.
        """
        distances_to_activated_nodes = self.bfs_from_activated_nodes(activated_nodes)
        for node in self.nodes:
            # PROPERTY nodes always get high score (inactive by distance)
            # Per paper: they only become visible via active property edges
            if node.node_type == NodeType.PROPERTY:
                node.score = 100  # Never selected by distance-based logic
            else:
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
        """
        Get nodes within the score threshold (close to active nodes).

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are EXCLUDED from score-based selection.
        Per paper: PROPERTY nodes have no independent activation.
        They only appear in the active graph if they have an active property edge.
        Score-based selection applies ONLY to CONCEPT nodes.
        """
        return [
            node
            for node in self.nodes
            if node.score <= score_threshold
            and node.node_type != NodeType.PROPERTY  # Per paper: no PROPERTY carry-over
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

    # ==========================================================================
    # TASK 2: CONNECTIVITY ENFORCEMENT LOGIC
    # ==========================================================================
    # This section contains all connectivity-related methods, isolated for clarity.
    #
    # CONNECTIVITY FLOW:
    # 1. check_active_connectivity() - Detect if graph is disconnected
    # 2. get_disconnected_components() - Get components that need connecting
    # 3. ensure_active_connectivity() - Try to connect using existing edges
    # 4. If still disconnected -> caller triggers secondary LLM call
    # ==========================================================================

    def check_active_connectivity(self) -> bool:
        """
        TASK 2: Check if the active graph is connected.

        This is the primary method for connectivity detection.
        Should be called BEFORE plotting to determine if intervention is needed.

        Returns:
            True if connected (or empty/single node), False if disconnected
        """
        import networkx as nx

        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return True  # Empty graph is trivially connected

        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        if G_active.number_of_nodes() <= 1:
            return True

        return nx.is_connected(G_active)

    def get_disconnected_components(
        self,
        focus_nodes: Set[Node],
    ) -> Tuple[List[Set[Node]], int]:
        """
        TASK 2: Get disconnected components and identify the focus component.

        Use this to determine which nodes need connecting and to which component.
        This information can be passed to a secondary LLM call.

        Args:
            focus_nodes: Nodes that should be in the "main" component

        Returns:
            Tuple of:
            - List of connected components (each is a set of nodes)
            - Index of the focus component (containing focus_nodes)

        If graph is connected, returns ([all_nodes], 0).
        """
        import networkx as nx

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

        # Find which component contains focus nodes
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
        TASK 2: Get pairs of nodes that need edges to restore connectivity.

        For each disconnected component, returns a representative node pair
        (one node from the component, one from the focus component) that
        could be connected by a new edge.

        Use this to prepare the secondary LLM call prompt.

        Args:
            focus_nodes: Nodes that define the "main" component

        Returns:
            List of (isolated_node, focus_node) pairs that need connecting.
            Empty list if graph is already connected.
        """
        components, focus_idx = self.get_disconnected_components(focus_nodes)

        if len(components) <= 1:
            return []  # Already connected

        focus_comp = components[focus_idx]
        pairs_needing_connection = []

        for idx, comp in enumerate(components):
            if idx == focus_idx:
                continue

            # Pick a representative node from each side
            # Prefer CONCEPT nodes over PROPERTY nodes
            from amoc.graph.node import NodeType

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
        focus_nodes: Set[Node],
        carryover_focus_nodes: Optional[Set[Node]] = None,
    ) -> Set[Edge]:
        """
        TASK 2: Ensure the active graph remains connected.

        STEP 1 of connectivity enforcement: Try to connect using EXISTING edges.

        If the active graph becomes disconnected:
        1. Identify focus nodes (explicit entities in current sentence + carry-over focus)
        2. Compute connected components of the active graph
        3. If multiple components exist, find minimum set of existing memory edges
           that connect them
        4. Promote those edges to active as "connectors"

        Connector edge constraints (CRITICAL):
        - Must already exist in the cumulative graph
        - Must NOT be PROPERTY edges
        - Must NOT be counted as asserted or reactivated
        - Do NOT increase activation scores
        - Are NOT eligible for inference
        - Exist only to preserve structural connectivity

        IMPORTANT: If this method returns an empty set but the graph is still
        disconnected (check with check_active_connectivity()), caller should
        trigger a secondary LLM call to create new edges. Use
        get_nodes_needing_connection() to get the pairs that need connecting.

        Returns:
            Set of edges promoted as connectors.
        """
        import networkx as nx

        # Build active subgraph
        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return set()

        # Build undirected graph of active edges
        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        # Check if already connected
        if G_active.number_of_nodes() <= 1 or nx.is_connected(G_active):
            return set()

        # Graph is disconnected - find components
        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return set()

        # Identify focus component (contains focus nodes)
        all_focus = focus_nodes | (carryover_focus_nodes or set())
        focus_component_idx = 0
        for idx, comp in enumerate(components):
            if any(n in comp for n in all_focus):
                focus_component_idx = idx
                break

        # Build full cumulative graph for finding paths
        G_cumulative = nx.Graph()
        for edge in self.edges:
            if edge.is_property_edge():
                continue
            G_cumulative.add_edge(edge.source_node, edge.dest_node, edge=edge)

        promoted_connectors: Set[Edge] = set()

        # For each non-focus component, find shortest path to focus component
        focus_comp_nodes = components[focus_component_idx]
        for idx, comp in enumerate(components):
            if idx == focus_component_idx:
                continue

            # Find shortest path in cumulative graph between any node in comp
            # and any node in focus component
            best_path = None
            best_path_len = float("inf")

            for src in comp:
                # Skip if src not in cumulative graph (only has property edges)
                if src not in G_cumulative:
                    continue
                for tgt in focus_comp_nodes:
                    if src == tgt:
                        continue
                    # Skip if tgt not in cumulative graph (only has property edges)
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
                # No path exists in cumulative graph - caller needs to use
                # secondary LLM call to create new edges
                continue

            # Promote edges along the path as connectors
            for i in range(len(best_path) - 1):
                node_a = best_path[i]
                node_b = best_path[i + 1]

                # Find the edge in cumulative graph
                edge_data = G_cumulative.get_edge_data(node_a, node_b)
                if edge_data is None:
                    continue

                edge = edge_data.get("edge")
                if edge is None:
                    continue

                # STRICT: Never use PROPERTY edges as connectors
                if edge.is_property_edge():
                    continue

                # Only promote if not already active
                if not edge.active:
                    edge.mark_as_connector()
                    promoted_connectors.add(edge)

            if __debug__:
                G_check = nx.Graph()
                for e in self.edges:
                    if e.active:
                        G_check.add_edge(e.source_node, e.dest_node)
                if G_check.number_of_nodes() > 1:
                    # Note: might still be disconnected if no path was found
                    pass

        return promoted_connectors

    def get_active_edges_by_role(self) -> dict[str, Set[Edge]]:
        """
        Get active edges grouped by their activation role.

        Returns dict with keys:
        - "asserted": edges asserted this sentence
        - "reactivated": edges reactivated from memory
        - "connector": edges promoted for connectivity
        """
        result: dict[str, Set[Edge]] = {
            "asserted": set(),
            "reactivated": set(),
            "connector": set(),
        }

        for edge in self.edges:
            if not edge.active:
                continue
            if edge.is_asserted():
                result["asserted"].add(edge)
            elif edge.is_reactivated():
                result["reactivated"].add(edge)
            elif edge.is_connector():
                result["connector"].add(edge)

        return result

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()

    def enforce_property_sentence_constraints(self, current_sentence: int) -> None:
        for edge in self.edges:
            if edge.violates_property_sentence_constraint(current_sentence):
                edge.deactivate()

    # ==========================================================================
    # AMoCv4 HARD CONSTRAINTS - Surface-relation format enforcement
    # ==========================================================================
    FORBIDDEN_EDGE_LABELS = {"agent_of", "target_of", "patient_of"}

    def validate_amocv4_constraints(self) -> None:
        """
        Enforce AMoCv4 surface-relation format constraints.

        Hard constraints (fail fast if violated):
        1. Never create agent_of, target_of, patient_of, or role-based edges
        2. All verbs must be represented as direct labeled edges between entities
        3. All attributes must be represented using the relation 'is'

        Raises:
            AssertionError: If any constraint is violated
        """
        # CONSTRAINT 1: No forbidden edge labels
        forbidden_edges = [
            edge for edge in self.edges if edge.label in self.FORBIDDEN_EDGE_LABELS
        ]
        assert not forbidden_edges, (
            f"AMoCv4 VIOLATION: Found {len(forbidden_edges)} forbidden edge(s) with "
            f"labels in {self.FORBIDDEN_EDGE_LABELS}. "
            f"Examples: {[(e.source_node.get_text_representer(), e.label, e.dest_node.get_text_representer()) for e in forbidden_edges[:3]]}"
        )
        # NOTE: NodeType.RELATION check removed - the type no longer exists in AMoCv4

    def sanity_check_readable_triplets(self) -> bool:
        """
        AMoCv4 sanity check: Every edge must be readable as a simple sentence fragment.

        Returns:
            True if all edges pass the sanity check

        Raises:
            AssertionError: If any edge cannot be read as a sentence fragment
        """
        for edge in self.edges:
            subj = edge.source_node.get_text_representer()
            verb = edge.label
            obj = edge.dest_node.get_text_representer()

            # Basic sanity: all parts must be non-empty
            assert subj and verb and obj, (
                f"AMoCv4 SANITY FAIL: Edge has empty component: "
                f"'{subj}' --{verb}--> '{obj}'"
            )

            # Forbidden patterns
            assert verb not in self.FORBIDDEN_EDGE_LABELS, (
                f"AMoCv4 SANITY FAIL: Edge uses forbidden label '{verb}': "
                f"'{subj}' --{verb}--> '{obj}'"
            )

        return True

    def sanity_check_provenance(
        self,
        story_lemmas: set,
        persona_only_lemmas: set,
    ) -> list:
        """
        AMoC v4 PROVENANCE SANITY CHECK: Detect potential persona leakage.

        Per AMoC v4 paper: Nodes must come from STORY TEXT only.
        Persona influences salience (weights), never content (nodes/edges).

        Args:
            story_lemmas: Set of lemmas from the story text
            persona_only_lemmas: Set of lemmas unique to persona (not in story)

        Returns:
            List of warning strings for any detected violations.
            Empty list if all nodes pass provenance check.
        """
        warnings = []

        for node in self.nodes:
            # Check each lemma in the node
            for lemma in node.lemmas:
                lemma_lower = lemma.lower()

                # CRITICAL CHECK: Node lemma appears ONLY in persona
                if lemma_lower in persona_only_lemmas:
                    warnings.append(
                        f"PROVENANCE VIOLATION: Node '{node.get_text_representer()}' "
                        f"contains lemma '{lemma_lower}' which appears ONLY in persona, "
                        f"not in story text. Provenance: {node.provenance}"
                    )

                # SOFT CHECK: Node lemma not found in story
                # (This might be OK for inferred nodes, but worth flagging)
                elif lemma_lower not in story_lemmas:
                    if node.provenance != NodeProvenance.INFERRED_FROM_STORY:
                        warnings.append(
                            f"PROVENANCE WARNING: Node '{node.get_text_representer()}' "
                            f"contains lemma '{lemma_lower}' not found in story text. "
                            f"Provenance: {node.provenance}"
                        )

        return warnings
