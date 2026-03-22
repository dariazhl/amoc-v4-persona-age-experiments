import logging
from typing import TYPE_CHECKING, Optional, List, Set, Dict, Tuple
from collections import deque
import networkx as nx
from amoc.core.node import NodeSource
from amoc.config.constants import MAX_CARRYOVER, MAX_TRIPLETS
from dataclasses import dataclass


@dataclass
class DecayDecision:
    triplet: Tuple[str, str, str]
    score: int
    action: str
    was_connectivity_critical: bool
    reasoning: str = ""

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge


class Decay:
    def __init__(
        self,
        graph_ref: "Graph",
        llm_extractor,
        get_explicit_nodes: callable,
        get_story_context: callable,  # Added for semantic decay
        max_distance: int,
        edge_visibility: int,
        nr_relevant_edges: int,
        strict_reactivate: bool = True,  # uses LLM to select which edges to reactivate
    ):
        self._graph = graph_ref
        self._llm = llm_extractor
        self._get_explicit_nodes = get_explicit_nodes
        self._get_story_context = get_story_context  # Store for semantic decay
        self._max_distance = max_distance
        self._edge_visibility = edge_visibility
        self._nr_relevant_edges = nr_relevant_edges
        self._strict_reactivate = strict_reactivate
        self._current_sentence_index = None
        self._current_sentence_text = None
        self._persona = None
        self._record_edge_fn = None
        self._last_decay_decisions: List[DecayDecision] = []

    def set_decay_state_refs(
        self,
        anchor_nodes: Set["Node"],
        record_edge_fn: callable = None,
        persona: str = None,  # Added persona parameter
    ):
        self._record_edge_fn = record_edge_fn
        self._persona = persona  # Store persona for LLM calls

    def set_decay_sentence_context(
        self, idx: int, text: str = None
    ):  # Added text parameter
        self._current_sentence_index = idx
        if text:
            self._current_sentence_text = text

    # global decay = fade edges that are never reinforced in cumulative graph
    def apply_global_edge_decay(self) -> None:
        for edge in self._graph.edges:
            if edge.created_at_sentence == self._current_sentence_index:
                continue
            if not edge.asserted_this_sentence and not edge.reactivated_this_sentence:
                edge.reduce_visibility()
                if edge.visibility_score <= 0:
                    edge.visibility_score = 0
                    edge.active = False

    def apply_semantic_edge_decay(self) -> List[DecayDecision]:
        # Step 1: Collect candidates
        decay_candidates, edge_to_triplet, candidate_strings = (
            self.collect_decay_candidates()
        )
        if not decay_candidates:
            return []

        # Step 2: Get LLM scores
        scores, reasoning = self.get_decay_scores(candidate_strings)
        if scores is None:
            self.apply_fallback_decay(decay_candidates)
            return []

        reasoning_text = reasoning if reasoning else ""
        if reasoning_text:
            logging.info(f"llm reasoning for decay: {reasoning_text}")

        # Step 3: Build connectivity map
        connectivity_map = self.build_connectivity_map()

        # Step 4: Process each edge
        stats = {
            "reinforce": 0,
            "maintain": 0,
            "decay": 0,
            "decay_immediate": 0,
            "protected": 0,
        }
        decisions: List[DecayDecision] = []

        for edge in decay_candidates:
            triplet_str = edge_to_triplet[edge]
            score = self.normalize_score(scores.get(triplet_str, 2))
            is_critical = not self.can_remove_edge(edge, connectivity_map)
            triplet = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )

            if score == 0:
                if is_critical:
                    edge.reduce_visibility()
                    stats["protected"] += 1
                    decisions.append(DecayDecision(
                        triplet=triplet, score=score,
                        action="protected", was_connectivity_critical=True,
                        reasoning=reasoning_text,
                    ))
                else:
                    edge.visibility_score = 0
                    edge.active = False
                    stats["decay_immediate"] += 1
                    decisions.append(DecayDecision(
                        triplet=triplet, score=score,
                        action="removed", was_connectivity_critical=False,
                        reasoning=reasoning_text,
                    ))

            elif score == 1:
                if is_critical:
                    # decay twice
                    edge.reduce_visibility()
                    edge.reduce_visibility()
                    stats["protected"] += 1
                    decisions.append(DecayDecision(
                        triplet=triplet, score=score,
                        action="protected", was_connectivity_critical=True,
                        reasoning=reasoning_text,
                    ))
                else:
                    edge.reduce_visibility()
                    if edge.visibility_score <= 0:
                        edge.visibility_score = 0
                        edge.active = False
                    stats["decay"] += 1
                    logging.info(
                        f"immediately deactivated edge {triplet_str} (score=1)"
                    )
                    decisions.append(DecayDecision(
                        triplet=triplet, score=score,
                        action="decayed", was_connectivity_critical=False,
                        reasoning=reasoning_text,
                    ))

            elif score == 2:
                edge.visibility_score = max(0, edge.visibility_score - 1)
                if edge.visibility_score <= 0:
                    edge.active = False
                stats["maintain"] += 1
                logging.info(
                    f"fast decay for edge {triplet_str} (score=2, new score={edge.visibility_score})"
                )
                decisions.append(DecayDecision(
                    triplet=triplet, score=score,
                    action="maintained", was_connectivity_critical=is_critical,
                    reasoning=reasoning_text,
                ))

            elif score == 3:
                edge.visibility_score = min(
                    edge.visibility_score + 1, self._edge_visibility
                )
                edge.mark_as_reactivated(reset_score=False)
                stats["reinforce"] += 1
                decisions.append(DecayDecision(
                    triplet=triplet, score=score,
                    action="reinforced", was_connectivity_critical=is_critical,
                    reasoning=reasoning_text,
                ))

            else:
                edge.reduce_visibility()
                logging.info(
                    f"fallback decay for edge {triplet_str} (unexpected score={score})"
                )
                decisions.append(DecayDecision(
                    triplet=triplet, score=score,
                    action="decayed", was_connectivity_critical=is_critical,
                    reasoning=reasoning_text,
                ))

        # Step 5: Log stats
        logging.info(
            f"decay stats - reinforce: {stats['reinforce']}, "
            f"maintain: {stats['maintain']}, gradual decay: {stats['decay']}, "
            f"immediate decay: {stats['decay_immediate']}, protected: {stats['protected']}"
        )

        self.reinforce_multi_hop_chains()
        return decisions

    def collect_decay_candidates(self):
        decay_candidates = []
        candidate_strings = []
        edge_to_triplet = {}

        for edge in self._graph.edges:
            if edge.created_at_sentence == self._current_sentence_index:
                continue
            if edge.asserted_this_sentence or edge.reactivated_this_sentence:
                continue

            triplet = f"({edge.source_node.get_text_representer()}, {edge.label}, {edge.dest_node.get_text_representer()})"
            candidate_strings.append(triplet)
            decay_candidates.append(edge)
            edge_to_triplet[edge] = triplet

        return decay_candidates, edge_to_triplet, candidate_strings

    def get_decay_scores(self, candidate_strings):
        story_context = self._get_story_context() if self._get_story_context else ""
        current_sentence = self._current_sentence_text

        if not current_sentence:
            logging.warning("SEMANTIC_DECAY: No current sentence text, skipping")
            return None, None

        try:
            result = self._llm.check_narrative_relevance(
                story_context=story_context,
                current_sentence=current_sentence,
                active_triplets="\n".join(candidate_strings),
                persona=self._persona,
            )
        except Exception as e:
            logging.error(f"SEMANTIC_DECAY: LLM call failed: {e}")
            return None, None

        if not result or "scores" not in result:
            return None, None

        return result.get("scores", {}), result.get("reasoning", "")

    def normalize_score(self, score):
        try:
            score = int(score)
            if score < 0:
                return 0
            if score > 3:
                return 3
            return score
        except:
            return 2

    # prune carryover nodes and inferred nodes
    def apply_pruning(self, prev_sentences, threshold_for_pruning=5, aggressive=False):
        # Get all active edges
        all_active_triplets = []
        edge_to_obj = {}

        # Track which nodes are explicit in current sentence
        explicit_nodes = (
            self._get_explicit_nodes()
            if hasattr(self, "_get_explicit_nodes")
            else set()
        )
        explicit_node_names = {node.get_text_representer() for node in explicit_nodes}

        for edge in self._graph.edges:
            if edge.active:
                source_name = edge.source_node.get_text_representer()
                dest_name = edge.dest_node.get_text_representer()
                triplet = (source_name, edge.label, dest_name)
                triplet_str = f"({source_name}, {edge.label}, {dest_name})"
                all_active_triplets.append(triplet)
                edge_to_obj[triplet_str] = edge

        current_count = len(all_active_triplets)
        if current_count <= threshold_for_pruning:
            return

        logging.info(
            f"current size {len(all_active_triplets)}, target {threshold_for_pruning}"
        )

        # Get story context
        story_context = self._get_story_context() if self._get_story_context else ""
        current_sentence = self._current_sentence_text

        # Format for LLM
        triplet_strings = [f"({s}, {r}, {o})" for s, r, o in all_active_triplets]

        # Call LLM
        result = self._llm.prune_irrelevant_triplets_by_narrative(
            story_context=story_context,
            current_sentence=current_sentence,
            active_triplets="\n".join(triplet_strings),
            persona=self._persona,
            aggressive=aggressive,
        )

        if not result or "to_keep" not in result:
            logging.warning("pruning failed, keep current graph")
            return

        # Get kept triplets
        keep_set = set(result["to_keep"])

        if not aggressive and len(keep_set) > MAX_TRIPLETS:
            logging.info(
                f"First pass kept {len(keep_set)} edges – still too many. Running second pass."
            )
            self.apply_pruning(prev_sentences, threshold_for_pruning=0, aggressive=True)
            return

        # Build connectivity map for protection
        connectivity_map = self.build_connectivity_map()

        # Deactivate pruned edges with connectivity protection
        deactivated = 0
        protected = 0
        explicit_protected = 0

        for triplet_str, edge in edge_to_obj.items():
            if triplet_str not in keep_set:
                source_name = edge.source_node.get_text_representer()
                dest_name = edge.dest_node.get_text_representer()

                # Never prune edges involving explicit nodes
                if (
                    source_name in explicit_node_names
                    or dest_name in explicit_node_names
                ):
                    explicit_protected += 1
                    # Still decay it a bit
                    edge.reduce_visibility()
                    continue

                # Check if removing would break connectivity
                if self.can_remove_edge(edge, connectivity_map):
                    edge.active = False
                    edge.visibility_score = 0
                    deactivated += 1
                else:
                    protected += 1
                    # Still decay it a bit
                    edge.reduce_visibility()

        logging.info(
            f"pruning deactivated {deactivated} edges, "
            f"protected {protected} critical edges, "
            f"protected {explicit_protected} explicit edges"
        )

    # inactivates zombie nodes after pruning and decay
    def prune_inactive_edgeless_nodes(self) -> List["Node"]:
        # First pass: deactivate any ghost edges (active=True, visibility<=0)
        ghost_count = 0
        for edge in self._graph.edges:
            if edge.active and edge.visibility_score <= 0:
                edge.active = False
                ghost_count += 1
        if ghost_count:
            logging.info(
                f"deactivated {ghost_count} ghost edges (active but visibility=0)"
            )

        dangling_nodes = []

        for node in self._graph.nodes:
            active_edges = [e for e in node.edges if e.active]

            # Node has no active edges — it's dangling
            if not active_edges:
                if node.active:
                    dangling_nodes.append(node)
                continue

            # Node has active edges but all have visibility <= 0 (shouldn't happen after ghost pass, but safety net)
            if all(e.visibility_score <= 0 for e in active_edges):
                for e in active_edges:
                    e.active = False
                dangling_nodes.append(node)

        if dangling_nodes:
            logging.info(
                f"cleaned up {len(dangling_nodes)} dangling nodes: "
                f"{[n.get_text_representer() for n in dangling_nodes]}"
            )

        return dangling_nodes

    def build_connectivity_map(self) -> Dict["Node", Set["Node"]]:
        connectivity = {}
        for edge in self._graph.edges:
            if edge.active:
                connectivity.setdefault(edge.source_node, set()).add(edge.dest_node)
                connectivity.setdefault(edge.dest_node, set()).add(edge.source_node)
        return connectivity

    # Check if edge can be removed without disconnecting the graph using BFS
    def can_remove_edge(self, edge, connectivity_map) -> bool:
        source = edge.source_node
        dest = edge.dest_node

        # Fast path: if either node would become isolated, definitely can't remove
        source_neighbors = connectivity_map.get(source, set())
        dest_neighbors = connectivity_map.get(dest, set())

        if len(source_neighbors) == 0 or len(dest_neighbors) == 0:
            return False  # Should never happen with active edges

        # If both nodes have multiple connections, check if there's an alternative path
        if len(source_neighbors) > 1 and len(dest_neighbors) > 1:
            # Do BFS to see if source can reach dest without this edge
            return self.has_alternative_path(source, dest, edge, connectivity_map)

        # If one node has only this connection, check if removing would isolate it
        if len(source_neighbors) == 1 and dest not in source_neighbors:
            return False  # This edge IS the only connection for source
        if len(dest_neighbors) == 1 and source not in dest_neighbors:
            return False  # This edge IS the only connection for dest

        # For other cases, do BFS to be sure
        return self.has_alternative_path(source, dest, edge, connectivity_map)

    def has_alternative_path(
        self, source, dest, edge_to_remove, connectivity_map
    ) -> bool:
        # Build a temporary connectivity map without this edge
        temp_map = {}
        for node, neighbors in connectivity_map.items():
            if node == source:
                temp_map[node] = {n for n in neighbors if n != dest}
            elif node == dest:
                temp_map[node] = {n for n in neighbors if n != source}
            else:
                temp_map[node] = set(neighbors)

        # BFS from source to dest
        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()
            if current == dest:
                return True  # Found alternative path - safe to remove

            for neighbor in temp_map.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False  # No alternative path - edge is critical

    def apply_fallback_decay(self, edges):
        for edge in edges:
            edge.reduce_visibility()
            if edge.visibility_score <= 0:
                edge.visibility_score = 0
                edge.active = False

    def reinforce_multi_hop_chains(self) -> None:
        text_based_nodes = {
            n for n in self._graph.nodes if n.node_source == NodeSource.TEXT_BASED
        }
        inferred_nodes = {
            n for n in self._graph.nodes if n.node_source == NodeSource.INFERENCE_BASED
        }

        if not inferred_nodes:
            return

        reinforced_count = 0
        chain_edges = set()

        for inf_node in inferred_nodes:
            if not inf_node.active:
                continue

            visited = {inf_node}
            queue = deque([(inf_node, 0)])
            found_path = False

            while queue and not found_path:
                current, dist = queue.popleft()

                for edge in current.edges:
                    if not edge.active:
                        continue

                    neighbor = (
                        edge.dest_node
                        if edge.source_node == current
                        else edge.source_node
                    )
                    if neighbor in visited:
                        continue

                    if neighbor in text_based_nodes:
                        chain_edges.add(edge)
                        found_path = True
                        break

                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        for edge in chain_edges:
            edge.visibility_score = min(edge.visibility_score + 2, 5)
            if not edge.is_property_edge():
                edge.mark_as_reactivated(reset_score=False)
            reinforced_count += 1

        if reinforced_count > 0:
            logging.info(f"reinforced {reinforced_count} edges in inference chains")

    def enforce_node_limit(self, max_nodes: int = 20) -> None:
        # Only count ACTIVE nodes toward the limit — inactive nodes are
        # retained in memory and should not trigger further deactivation
        # active_count = sum(1 for n in self._graph.nodes if n.active)
        # if active_count <= max_nodes:
        #     return

        # # Protect explicit and carryover nodes from deactivation
        # protected_nodes = set()
        # if self._get_explicit_nodes:
        #     protected_nodes.update(self._get_explicit_nodes())

        # G_active, active_nodes, critical_nodes = self.identify_critical_nodes()
        # # Add protected nodes to critical so they are never candidates
        # critical_nodes = critical_nodes | protected_nodes
        # current_sentence = getattr(self, "_current_sentence_idx", 0)
        # node_scores = self.score_nodes(
        #     G_active, active_nodes, current_sentence, critical_nodes
        # )
        # candidates, excess = self.select_removal_candidates(
        #     node_scores, max_nodes, critical_nodes, active_only=True
        # )

        # if not candidates:
        #     return

        # safe_to_remove, would_fragment = self.simulate_removals(G_active, candidates)
        # removed = self.deactivate_nodes(safe_to_remove, would_fragment, excess)
        # self.log_removal_results(removed, excess, candidates)
        pass

    def identify_critical_nodes(self):
        G = nx.Graph()
        active_nodes = set()

        for node in self._graph.nodes:
            if node.active:
                active_nodes.add(node)
                G.add_node(node)

        for edge in self._graph.edges:
            if edge.active:
                G.add_edge(edge.source_node, edge.dest_node)

        critical_nodes = set()
        if len(active_nodes) > 1:
            critical_nodes = set(nx.articulation_points(G))

        return G, active_nodes, critical_nodes

    def score_nodes(self, G, active_nodes, current_sentence, critical_nodes):
        node_scores = {}

        for node in self._graph.nodes:
            if node in critical_nodes:
                node_scores[node] = float("inf")
                continue

            score = 0
            active_edge_count = sum(1 for e in node.edges if e.active)
            score += active_edge_count * 3

            # Boost inference-based nodes significantly — they represent
            # LLM-inferred knowledge that bridges concepts
            if node.node_source == NodeSource.TEXT_BASED:
                score += 5
            elif node.node_source == NodeSource.INFERENCE_BASED:
                score += 20

            # Extra boost for inference nodes that bridge to text-based nodes
            if node.node_source == NodeSource.INFERENCE_BASED:
                connects_to_text = any(
                    e.active
                    and (
                        (
                            e.source_node == node
                            and e.dest_node.node_source == NodeSource.TEXT_BASED
                        )
                        or (
                            e.dest_node == node
                            and e.source_node.node_source == NodeSource.TEXT_BASED
                        )
                    )
                    for e in node.edges
                )
                if connects_to_text:
                    score += 15

            if node.active:
                score += 8

            if (
                hasattr(node, "created_at_sentence")
                and node.created_at_sentence is not None
            ):
                age = current_sentence - node.created_at_sentence
                recency_score = max(0, 10 - min(age, 10))
                score += recency_score * 2

            if (
                hasattr(node, "last_active_sentence")
                and node.last_active_sentence is not None
            ):
                inactivity = current_sentence - node.last_active_sentence
                if inactivity <= 2:
                    score += 5
                elif inactivity <= 5:
                    score += 2

            if node in G and len(G) > 1:
                degree = G.degree(node)
                centrality = degree / (len(G) - 1)
                score += centrality * 5

            node_scores[node] = score

        return node_scores

    def select_removal_candidates(
        self, node_scores, max_nodes, critical_nodes, active_only=False
    ):
        # Separate candidates by source — prune text-based first
        text_nodes = []
        inference_nodes = []

        for n in node_scores:
            if n in critical_nodes:
                continue
            if active_only and not n.active:
                continue
            if n.node_source == NodeSource.TEXT_BASED:
                text_nodes.append(n)
            else:
                inference_nodes.append(n)

        sorted_text = sorted(text_nodes, key=lambda n: node_scores[n])
        sorted_inference = sorted(inference_nodes, key=lambda n: node_scores[n])

        if active_only:
            excess = sum(1 for n in self._graph.nodes if n.active) - max_nodes
        else:
            excess = len(self._graph.nodes) - max_nodes

        if excess <= 0:
            return [], 0

        # Take from text-based first, then inference-based only if needed
        if len(sorted_text) >= excess:
            candidates = sorted_text[:excess]
            logging.info(f"selecting {excess} text-based nodes for potential removal")
        else:
            candidates = sorted_text.copy()
            remaining = excess - len(sorted_text)
            candidates.extend(sorted_inference[:remaining])
            logging.info(
                f"taking all {len(sorted_text)} text-based + "
                f"{remaining} inference-based nodes for potential removal"
            )

        return candidates, excess

    def simulate_removals(self, G, candidates):
        safe_to_remove = []
        would_fragment = []
        G_copy = G.copy()

        for node in candidates:
            if node not in G_copy:
                safe_to_remove.append(node)
                continue

            neighbors = list(G_copy.neighbors(node))
            G_copy.remove_node(node)

            fragments = False
            for neighbor in neighbors:
                if neighbor in G_copy and G_copy.degree(neighbor) == 0:
                    fragments = True
                    break

            if not fragments and nx.is_connected(G_copy):
                safe_to_remove.append(node)
            else:
                would_fragment.append(node)
                G_copy.add_node(node)
                for neighbor in neighbors:
                    if neighbor in G_copy.nodes():
                        G_copy.add_edge(node, neighbor)

        return safe_to_remove, would_fragment

    def deactivate_nodes(self, safe_to_remove, would_fragment, excess):
        removed = 0

        for node in safe_to_remove[:excess]:
            self.deactivate_single_node(node)
            removed += 1
            if removed >= excess:
                return removed

        if removed < excess:
            additional_needed = excess - removed
            for node in would_fragment[:additional_needed]:
                logging.warning(
                    f"NODE_LIMIT: Deactivating node '{getattr(node, 'actual_texts', 'unknown')}' (may fragment graph)"
                )
                self.deactivate_single_node(node)
                removed += 1

        return removed

    def deactivate_single_node(self, node):
        for edge in list(node.edges):
            edge.active = False

        if hasattr(self._graph, "_inactive_nodes"):
            self._graph._inactive_nodes.add(node)

    def log_removal_results(self, removed, excess, candidates):
        if removed == 0:
            return

        final_G = nx.Graph()
        for node in self._graph.nodes:
            if node.active:
                final_G.add_node(node)
        for edge in self._graph.edges:
            if edge.active:
                final_G.add_edge(edge.source_node, edge.dest_node)

        components = list(nx.connected_components(final_G))

        text_removed = sum(
            1 for n in candidates[:removed] if n.node_source == NodeSource.TEXT_BASED
        )
        inference_removed = sum(
            1
            for n in candidates[:removed]
            if n.node_source == NodeSource.INFERENCE_BASED
        )

        logging.info(
            f"deactivated {removed}/{excess} nodes. "
            f"graph now has {len(final_G)} active nodes in {len(components)} components "
            f"(text-based: {text_removed}, inference-based: {inference_removed})"
        )

        removed_names = [
            getattr(n, "actual_texts", "unknown")
            for n in candidates[:removed]
            if hasattr(n, "actual_texts")
        ]
        logging.info(f"deactivated nodes: {removed_names}")

    def reactivate_relevant_edges(
        self,
        active_nodes: List["Node"],
        prev_sentences_text: str,
        newly_added_edges: List["Edge"],
    ) -> None:
        edges_text, edges = self._graph.get_edges_str(
            self._graph.nodes, only_active=False
        )

        if not self._strict_reactivate:
            for edge in edges:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)
            return

        raw_indices = self._llm.get_relevant_edges(
            edges_text, prev_sentences_text, None
        )

        valid_indices = []
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(edges):
                valid_indices.append(i)

        valid_indices = valid_indices[: self._nr_relevant_edges]

        active_node_set = set(active_nodes)

        if valid_indices:
            selected = set(valid_indices)
            for i in selected:
                edge = edges[i - 1]
                edge.visibility_score = self._edge_visibility
                if not edge.is_property_edge():
                    continue
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)
        else:
            selected = set()  # Edge not selected by LLM
            logging.info("llm didn't find any edges to reactivate")

        for idx, edge in enumerate(edges, start=1):
            if idx in selected or edge in newly_added_edges:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)

    def get_fallback_edges(
        self,
        edges: List["Edge"],
        newly_added_edges: List["Edge"],
        active_node_set: Set["Node"],
    ) -> Set["Edge"]:
        fallback = set()
        for idx, edge in enumerate(edges, start=1):
            if (
                edge in newly_added_edges
                or edge.source_node in active_node_set
                or edge.dest_node in active_node_set
            ):
                fallback.add(edge)
        return fallback

    def select_edges_for_reactivation(
        self,
        edges: List["Edge"],
        selected_indices: List[int],
        newly_added_edges: List["Edge"],
        active_node_set: Set["Node"],
    ) -> Set["Edge"]:
        edges_to_reactivate = set()
        for i in selected_indices:
            edge = edges[i - 1]
            edge.visibility_score = self._edge_visibility
            if edge.is_property_edge():
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)
            else:
                edges_to_reactivate.add(edge)

        for edge in newly_added_edges:
            if not edge.is_property_edge():
                edges_to_reactivate.add(edge)

        for idx, edge in enumerate(edges, start=1):
            if idx not in selected_indices:
                if (
                    edge.source_node in active_node_set
                    or edge.dest_node in active_node_set
                ) and not edge.is_property_edge():
                    edges_to_reactivate.add(edge)

        return edges_to_reactivate

    def process_edges(
        self, edges: List["Edge"], edges_to_reactivate: Set["Edge"]
    ) -> None:
        for edge in edges:
            if edge in edges_to_reactivate:
                if edge.is_property_edge():
                    continue
                edge.mark_as_reactivated(
                    reset_score=False, new_visibility=self._edge_visibility
                )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)
            # No decay here — apply_global_edge_decay handles it

    def propagate_activation_from_edges(self) -> None:
        pass

    def convert_to_landscape_score(self, raw_score: float) -> float:
        val = 5.0 - float(raw_score)
        if val < 0.0:
            return 0.0
        if val > 5.0:
            return 5.0
        return val

    # records the activation scores of all relevant nodes for the current sent
    # landscape model
    def record_sentence_activation_matrix(
        self,
        sentence_id: int,
        explicit_nodes: List["Node"],
        newly_inferred_nodes: Set["Node"],
        max_distance: int,
        node_token_fn: callable,
        append_record_fn: callable,
    ) -> None:
        explicit_set = set(explicit_nodes)
        distances = self.compute_distances_from_sources(
            explicit_set, max_distance=max_distance
        )

        token_to_raw_score = {}
        node_raw_score = {}

        for node in explicit_set:
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 0
                node_raw_score[node] = 0

        for node in newly_inferred_nodes:
            if node in explicit_set:
                continue
            token = node_token_fn(node)
            if token:
                token_to_raw_score[token] = 1
                node_raw_score[node] = 1

        for node, dist in distances.items():
            if node in explicit_set or dist <= 0:
                continue
            token = node_token_fn(node)
            if token and token not in token_to_raw_score:
                token_to_raw_score[token] = dist
                node_raw_score[node] = dist

        for token, raw_score in token_to_raw_score.items():
            append_record_fn(
                {
                    "sentence": sentence_id,
                    "token": token,
                    "score": self.convert_to_landscape_score(raw_score),
                }
            )

        # Add verb activations: take the max activation of connected nodes minus 0.5.
        # Paper: "Second, in the Landscape Model, verbs are treated as nodes. However, in AMoC v4.0, verbs are primarily found in the edges... Thus, verbs were assigned an activation score with the following procedure: the highest activation score of the nodes linked by the edge that the verb is part of, decayed by 0.5"
        verb_scores: Dict[str, float] = {}
        for edge in self._graph.edges:
            if not edge.active:
                continue
            label = (edge.label or "").strip().lower()
            if not label:
                continue
            src_tok = node_token_fn(edge.source_node)
            dst_tok = node_token_fn(edge.dest_node)
            if not src_tok or not dst_tok:
                continue
            src_raw = node_raw_score.get(edge.source_node, max_distance + 1)
            dst_raw = node_raw_score.get(edge.dest_node, max_distance + 1)
            src_act = self.convert_to_landscape_score(src_raw)
            dst_act = self.convert_to_landscape_score(dst_raw)
            verb_act = max(src_act, dst_act) - 0.5
            if verb_act < 0.0:
                verb_act = 0.0
            prev = verb_scores.get(label)
            if prev is None or verb_act > prev:
                verb_scores[label] = verb_act

        for token, score in verb_scores.items():
            append_record_fn({"sentence": sentence_id, "token": token, "score": score})

    def compute_distances_from_sources(
        self, sources: Set["Node"], max_distance: int
    ) -> Dict["Node", int]:
        if not sources:
            return {}
        distances = {s: 0 for s in sources}
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            dist = distances[node]
            if dist >= max_distance:
                continue
            for edge in node.edges:
                if not edge.active or edge.visibility_score <= 1:
                    continue
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if neighbor in distances:
                    continue
                distances[neighbor] = dist + 1
                queue.append(neighbor)
        return distances

    def restrict_active_nodes(self, explicit_nodes: List["Node"]) -> None:
        pass

    def has_active_attachment(self, lemma: str) -> bool:
        active_nodes = {n for n in self._graph.nodes if n.active}
        active_nodes |= self._get_explicit_nodes()
        return any(lemma in n.lemmas for n in active_nodes)

    def get_last_decay_decisions(self) -> List[DecayDecision]:
        return self._last_decay_decisions

    def get_decay_decisions_with_triplets(self) -> List[Tuple[Tuple[str, str, str], str]]:
        result = []
        for decision in self._last_decay_decisions:
            if decision.action in ("removed", "decayed", "protected", "maintained"):
                result.append((decision.triplet, decision.reasoning))
        return result

    def post_sentence_cleanup(self, prev_sentences):
        # First run semantic decay
        self._last_decay_decisions = self.apply_semantic_edge_decay()

        # Then pruning
        self.apply_pruning(prev_sentences)

        # Final node cleanup - edges get deactivated, but nodes do not
        for edge in self._graph.edges:
            if edge.visibility_score <= 0 and edge.active:
                edge.active = False

        self.prune_inactive_edgeless_nodes()

        # Final node cleanup - edges get deactivated, but nodes do not
        for edge in self._graph.edges:
            if edge.visibility_score <= 0 and edge.active:
                edge.active = False

        # # # node cap
        # explicit_nodes = (
        #     self._get_explicit_nodes()
        #     if hasattr(self, "_get_explicit_nodes")
        #     else set()
        # )
        # explicit_names = {n.get_text_representer() for n in explicit_nodes}

        # # Get all active carryover nodes
        # carryover_nodes = []
        # for n in self._graph.nodes:
        #     if not n.active or n.get_text_representer() in explicit_names:
        #         continue
        #     # Check if node has any active edge with score > 0
        #     has_active_edge = any(e.active and e.visibility_score > 0 for e in n.edges)
        #     if has_active_edge:
        #         carryover_nodes.append(n)

        # # If too many, keep only those with most edges
        # if len(carryover_nodes) > MAX_CARRYOVER:
        #     # Sort by number of active edges (descending)
        #     carryover_nodes.sort(
        #         key=lambda n: sum(1 for e in n.edges if e.active), reverse=True
        #     )

        #     # Keep top MAX_CARRYOVER, deactivate rest
        #     to_keep = carryover_nodes[:MAX_CARRYOVER]
        #     to_prune = carryover_nodes[MAX_CARRYOVER:]

        #     for node in to_prune:
        #         for edge in list(node.edges):
        #             if edge.active:
        #                 edge.active = False
        #                 edge.visibility_score = 0
        #         logging.info(f"Capped carryover node: {node.get_text_representer()}")
