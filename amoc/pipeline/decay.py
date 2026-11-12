import logging
import csv
import os
from typing import TYPE_CHECKING, Optional, List, Set, Dict, Tuple
from collections import deque
import networkx as nx
from amoc.core.node import NodeSource
from amoc.config.constants import (
    MAX_CARRYOVER, PRUNING_SIZE_THRESHOLD, REACTIVATION_VISIBILITY,
    CARRYOVER_SENIORITY_WEIGHT, DECAY_STEP,
    MATRIX_MIN_PROPAGATION_VISIBILITY,
)
from dataclasses import dataclass


MATRIX_SCORE_MIN = 0.0
MATRIX_SCORE_MAX = 5.0


def clamp_matrix_score(value: float) -> float:
    return max(MATRIX_SCORE_MIN, min(float(value), MATRIX_SCORE_MAX))

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
        get_story_context: callable,
        max_distance: int,
        edge_visibility: int,
        nr_relevant_edges: int,
        strict_reactivate: bool = True,
    ):
        self._graph = graph_ref
        self._llm = llm_extractor
        self._get_explicit_nodes = get_explicit_nodes
        self._get_story_context = get_story_context
        self._max_distance = max_distance
        self._edge_visibility = edge_visibility
        self._nr_relevant_edges = nr_relevant_edges
        self._strict_reactivate = strict_reactivate
        self._current_sentence_index = None
        self._current_sentence_text = None
        self._persona = None
        self._record_edge_fn = None
        self._last_decay_decisions: List[DecayDecision] = []
        self._full_activation_matrix: Dict[str, List[float]] = {}
        self._max_sentence_index: int = 0
        self._last_context_shift: bool = False

    def set_decay_state_refs(
        self,
        anchor_nodes: Set["Node"],
        record_edge_fn: callable = None,
        persona: str = None,
    ):
        self._record_edge_fn = record_edge_fn
        self._persona = persona

    def set_decay_sentence_context(self, idx: int, text: str = None):
        self._current_sentence_index = idx
        if text:
            self._current_sentence_text = text

    def reset_sentence_flags(self) -> None:
        asserted_count = sum(1 for e in self._graph.edges if e.asserted_this_sentence)
        logging.info(
            f"RESET FLAGS: {len(self._graph.edges)} edges reset "
            f"(clearing {asserted_count} asserted flags)"
        )
        for edge in self._graph.edges:
            edge.reset_for_sentence_start()

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
            self._last_context_shift = False
            return []

        # Step 2: Get LLM scores
        scores, reasoning, context_shift = self.get_decay_scores(candidate_strings)
        if scores is None or not isinstance(scores, dict):
            logging.warning(f"Decay scores invalid type: {type(scores)}. Using fallback decay.")
            self.apply_fallback_decay(decay_candidates)
            self._last_context_shift = False
            return []

        self.log_score_matches(decay_candidates, edge_to_triplet, scores)

        self._last_context_shift = context_shift
        if context_shift:
            logging.info(
                f"CONTEXT SHIFT detected at sentence {self._current_sentence_index}"
            )

        reasoning_text = reasoning if reasoning else ""
        if reasoning_text:
            logging.info(f"llm reasoning for decay: {reasoning_text}")

        # Step 3: Build connectivity map
        connectivity_map = self.build_connectivity_map()

        # Get current sentence tokens for endpoint checks
        current_sentence = self._current_sentence_text.lower() if self._current_sentence_text else ""
        current_sentence_tokens = set(current_sentence.split()) if current_sentence else set()

        # Step 4: Process each edge
        stats = {
            "maintain": 0,
            "decay": 0,
            "removed": 0,
            "reactivated": 0,
            "protected": 0,
        }
        decisions: List[DecayDecision] = []

        for edge in decay_candidates:
            triplet_str = edge_to_triplet[edge]
            score = self.normalize_score(scores.get(triplet_str, 2))
            is_critical = not self.can_remove_edge(edge, connectivity_map)
            old_vis = edge.visibility_score
            triplet = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )

            source_token = edge.source_node.get_text_representer().lower()
            dest_token = edge.dest_node.get_text_representer().lower()
            source_in_sentence = source_token in current_sentence_tokens
            dest_in_sentence = dest_token in current_sentence_tokens

            # SCORE 0: LOWEST RELEVANCE – immediate removal
            if score == 0:
                # Immediate removal regardless of critical status
                edge.visibility_score = 0
                edge.active = False
                stats["removed"] += 1
                action = "removed_immediate"
                
                decisions.append(
                    DecayDecision(
                        triplet=triplet,
                        score=score,
                        action=action,
                        was_connectivity_critical=is_critical,
                        reasoning=reasoning_text,
                    )
                )

            # SCORE 1: LOW RELEVANCE – per the LLM's own rubric, score 1 means
            # only the subject appears,  so the object is never expected to be in the sentence
            # here. Treat the same as score 0: immediate removal.
            elif score == 1:
                edge.visibility_score = 0
                edge.active = False
                stats["removed"] += 1
                action = "removed_immediate"

                decisions.append(
                    DecayDecision(
                        triplet=triplet,
                        score=score,
                        action=action,
                        was_connectivity_critical=is_critical,
                        reasoning=reasoning_text,
                    )
                )

            # SCORE 2: REACTIVATE (from inactive) or GRADUAL DECAY (if active)
            elif score == 2:
                if edge.visibility_score <= 0:
                    # Only reactivate if object appears in sentence
                    if dest_in_sentence:
                        edge.visibility_score = REACTIVATION_VISIBILITY
                        edge.active = True
                        edge.mark_as_reactivated(reset_score=False)
                        stats["reactivated"] = stats.get("reactivated", 0) + 1
                        action = "reactivated"
                    else:
                        # Object not mentioned – keep inactive
                        action = "not_reactivated"
                        logging.debug(f"Skipped reactivation of {triplet_str} – object not in sentence")
                else:
                    # Active edge – gradual decay (configurable)
                    edge.visibility_score -= DECAY_STEP
                    if edge.visibility_score <= 0:
                        edge.visibility_score = 0
                        edge.active = False
                        action = "removed"
                    else:
                        stats["decay"] += 1
                        action = "decayed"

                decisions.append(
                    DecayDecision(
                        triplet=triplet,
                        score=score,
                        action=action,
                        was_connectivity_critical=is_critical,
                        reasoning=reasoning_text,
                    )
                )

            # Fallback (should not happen)
            else:
                edge.visibility_score -= DECAY_STEP
                if edge.visibility_score <= 0:
                    edge.visibility_score = 0
                    edge.active = False
                logging.warning(
                    f"fallback decay for edge {triplet_str} (unexpected score={score})"
                )
                decisions.append(
                    DecayDecision(
                        triplet=triplet,
                        score=score,
                        action="decayed",
                        was_connectivity_critical=is_critical,
                        reasoning=reasoning_text,
                    )
                )

            logging.info(
                f"DECAY S{self._current_sentence_index}: {triplet_str} | "
                f"score={score} | vis {old_vis}→{edge.visibility_score} | "
                f"critical={is_critical}"
            )

        # Step 5: Log stats
        logging.info(
            f"decay stats - maintain: {stats['maintain']}, "
            f"decay: {stats['decay']}, removed: {stats['removed']}, "
            f"reactivated: {stats['reactivated']}, "
            f"protected: {stats['protected']}"
        )
        return decisions

    def collect_decay_candidates(self):
        decay_candidates = []
        candidate_strings = []
        edge_to_triplet = {}

        for edge in self._graph.edges:
            if edge.created_at_sentence == self._current_sentence_index:
                continue

            if edge.asserted_this_sentence:
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
            return None, None, False

        try:
            result = self._llm.check_narrative_relevance(
                story_context=story_context,
                current_sentence=current_sentence,
                active_triplets="\n".join(candidate_strings),
                persona=self._persona,
            )
        except Exception as e:
            logging.error(f"SEMANTIC_DECAY: LLM call failed: {e}")
            return None, None, False

        if not result or "scores" not in result:
            return None, None, False

        return (
            result.get("scores", {}),
            result.get("reasoning", ""),
            bool(result.get("context_shift", False)),
        )

    def log_score_matches(self, decay_candidates, edge_to_triplet, scores):
        sent = {edge_to_triplet[e] for e in decay_candidates}
        matched = [k for k in scores if k in sent]
        unmatched = [k for k in scores if k not in sent]
        counts = {0: 0, 1: 0, 2: 0, "other": 0}
        for k in matched:
            score = self.normalize_score(scores[k])
            counts[score if score in (0, 1, 2) else "other"] += 1
        logging.info(
            f"DECAY_SCORES S{self._current_sentence_index}: candidates={len(sent)} "
            f"returned={len(scores)} matched={len(matched)} "
            f"missing_default_2={len(sent) - len(matched)} "
            f"score0={counts[0]} score1={counts[1]} score2={counts[2]} other={counts['other']} "
            f"unmatched_keys={unmatched[:5]}"
        )
        return len(matched), counts

    def normalize_score(self, score):
        try:
            score = int(score)
            if score < 0:
                return 0
            if score > 2:
                return 2
            return score
        except:
            return 2

    def apply_pruning(
        self, prev_sentences, threshold_for_pruning=PRUNING_SIZE_THRESHOLD, aggressive=True
    ):
        all_active_triplets = []
        edge_to_obj = {}

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
            f"pruning: current size {len(all_active_triplets)}, target {threshold_for_pruning}"
        )

        story_context = self._get_story_context() if self._get_story_context else ""
        current_sentence = self._current_sentence_text

        triplet_strings = [f"({s}, {r}, {o})" for s, r, o in all_active_triplets]

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

        keep_set = set(result["to_keep"])

        connectivity_map = self.build_connectivity_map()

        removed = 0
        protected = 0
        explicit_protected = 0

        for triplet_str, edge in edge_to_obj.items():
            if triplet_str not in keep_set:
                if edge.asserted_this_sentence or edge.reactivated_this_sentence:
                    continue

                source_name = edge.source_node.get_text_representer()
                dest_name = edge.dest_node.get_text_representer()

                if source_name in explicit_node_names or dest_name in explicit_node_names:
                    explicit_protected += 1

                if self.can_remove_edge(edge, connectivity_map):
                    old_vis = edge.visibility_score
                    edge.visibility_score = 0
                    edge.active = False
                    removed += 1
                    logging.info(
                        f"PRUNING S{self._current_sentence_index}: {triplet_str} | "
                        f"vis {old_vis}→0"
                    )
                else:
                    protected += 1
                    edge.reduce_visibility() 
                    logging.debug(f"decayed critical edge: {triplet_str}")

        logging.info(
            f"pruning: removed {removed} edges, "
            f"decayed {protected} critical edges, "
            f"explicit edges affected: {explicit_protected}"
        )

    def prune_inactive_edgeless_nodes(self) -> List["Node"]:
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

            if not active_edges:
                if node.active:
                    dangling_nodes.append(node)
                continue

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

    def can_remove_edge(self, edge, connectivity_map) -> bool:
        source = edge.source_node
        dest = edge.dest_node

        source_neighbors = connectivity_map.get(source, set())
        dest_neighbors = connectivity_map.get(dest, set())

        if len(source_neighbors) == 0 or len(dest_neighbors) == 0:
            return False

        if len(source_neighbors) > 1 and len(dest_neighbors) > 1:
            return self.has_alternative_path(source, dest, edge, connectivity_map)

        if len(source_neighbors) == 1 and dest not in source_neighbors:
            return False
        if len(dest_neighbors) == 1 and source not in dest_neighbors:
            return False

        return self.has_alternative_path(source, dest, edge, connectivity_map)

    def has_alternative_path(
        self, source, dest, edge_to_remove, connectivity_map
    ) -> bool:
        temp_map = {}
        for node, neighbors in connectivity_map.items():
            if node == source:
                temp_map[node] = {n for n in neighbors if n != dest}
            elif node == dest:
                temp_map[node] = {n for n in neighbors if n != source}
            else:
                temp_map[node] = set(neighbors)

        visited = set()
        queue = deque([source])
        visited.add(source)

        while queue:
            current = queue.popleft()
            if current == dest:
                return True

            for neighbor in temp_map.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

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
            if edge.visibility_score < REACTIVATION_VISIBILITY:
                edge.visibility_score = REACTIVATION_VISIBILITY
                edge.active = True
                if not edge.reactivated_this_sentence:
                    edge.mark_as_reactivated(reset_score=True)
                reinforced_count += 1

        if reinforced_count > 0:
            logging.info(f"reinforced {reinforced_count} edges in inference chains")

    def enforce_node_limit(self, max_nodes: int = 20) -> None:
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

            if node.node_source == NodeSource.TEXT_BASED:
                score += 5
            elif node.node_source == NodeSource.INFERENCE_BASED:
                score += 20

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
                if edge.visibility_score <= 0:
                    edge.mark_as_reactivated(
                        reset_score=False, new_visibility=REACTIVATION_VISIBILITY
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
                if edge.visibility_score <= 0:
                    edge.mark_as_reactivated(reset_score=False, new_visibility=REACTIVATION_VISIBILITY)
                    logging.info(
                        f"REACTIVATE: ({edge.source_node.get_text_representer()}, "
                        f"{edge.label}, {edge.dest_node.get_text_representer()}) "
                        f"vis 0→{REACTIVATION_VISIBILITY}"
                    )
                if edge.is_property_edge():
                    continue
                if self._record_edge_fn:
                    self._record_edge_fn(edge, self._current_sentence_index)
        else:
            selected = set()
            logging.info("llm didn't find any edges to reactivate")

        for idx, edge in enumerate(edges, start=1):
            if idx in selected or edge in newly_added_edges:
                if edge.is_property_edge():
                    continue
                if edge.visibility_score <= 0:
                    edge.mark_as_reactivated(
                        reset_score=False, new_visibility=REACTIVATION_VISIBILITY
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
            if edge.visibility_score <= 0:
                edge.visibility_score = REACTIVATION_VISIBILITY
                edge.active = True
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
                if edge.visibility_score <= 0:
                    edge.mark_as_reactivated(
                        reset_score=False, new_visibility=REACTIVATION_VISIBILITY
                    )
                if self._record_edge_fn and (
                    edge.is_asserted() or edge.is_reactivated()
                ):
                    self._record_edge_fn(edge, self._current_sentence_index)

    def propagate_activation_from_edges(self) -> None:
        pass

    def convert_to_landscape_score(self, raw_score: float) -> float:
        val = 5.0 - float(raw_score)
        if val < 0.0:
            return 0.0
        if val > 5.0:
            return 5.0
        return val

    def is_activation_onset(self, token: str, sentence_id: int) -> bool:
        if sentence_id <= 1:
            return True
        history = self._full_activation_matrix.get(token)
        if not history or len(history) < sentence_id - 1:
            return True
        return history[sentence_id - 2] == 0.0

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
        # Compute distances from explicit nodes across active edges
        distances = self.compute_distances_from_sources(
            explicit_set, max_distance=max_distance
        )

        for node in self._graph.nodes:
            token = node_token_fn(node)
            if not token:
                continue

            dist = distances.get(node, max_distance + 1)
            # Nodes outside the reachable graph neighbourhood get score 0.
            # Previously convert_to_landscape_score(max_distance+1) = 5-(max_distance+1)
            # gave them a non-zero phantom base score (e.g. 2.0 when max_distance=2),
            # causing stale battle tokens to linger 4+ sentences in the matrix.
            if dist > max_distance:
                score = 0.0
            else:
                base_score = self.convert_to_landscape_score(dist)

                seniority = 0
                if node.first_seen_sentence is not None:
                    if node in explicit_set:
                        # Fix C: node explicitly re-mentioned → reset age penalty
                        seniority = 0
                    else:
                        seniority = max(0, sentence_id - node.first_seen_sentence)

                penalty = CARRYOVER_SENIORITY_WEIGHT * seniority
                score = max(0.0, base_score - penalty)

            if dist > max_distance:
                logging.debug(f"S{sentence_id} | {token} | dist={dist} (unreachable) score=0.0")
            else:
                logging.debug(
                    f"S{sentence_id} | {token} | dist={dist} base={base_score:.1f} "
                    f"seniority={seniority} penalty={penalty:.1f} score={score:.1f}"
                )

            if score > 0.0 and self.is_activation_onset(token, sentence_id):
                score = 5.0

            append_record_fn({
                "sentence": sentence_id,
                "token": token,
                "score": clamp_matrix_score(score),
            })

        node_raw_score = {}
        for node in self._graph.nodes:
            dist = distances.get(node, max_distance + 1)
            if dist > max_distance:
                node_raw_score[node] = 5.0  # unreachable → raw distance = max, verb score = 0
            else:
                base_score = self.convert_to_landscape_score(dist)
                seniority = 0
                if node.first_seen_sentence is not None:
                    if node in explicit_set:
                        # Fix C: re-mentioned node → no age penalty on verb scores either
                        seniority = 0
                    else:
                        seniority = max(0, sentence_id - node.first_seen_sentence)
                penalized_score = max(0.0, base_score - CARRYOVER_SENIORITY_WEIGHT * seniority)
                node_raw_score[node] = 5.0 - penalized_score

        linking_verbs = {
            'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'has', 'have', 'had', 'having', 'does', 'do', 'did',
            'involves', 'relates', 'includes', 'describes', 'becomes',
            'remains', 'seems', 'appears', 'constitutes', 'represents',
            'not related', 'not applicable', 'part of', 'marks',
            'not available', 'returns to', 'precedes'
        }

        verb_scores: Dict[str, float] = {}
        verb_onset_scores: Dict[str, float] = {}

        for edge in self._graph.edges:
            if not edge.active:
                continue

            label = (edge.label or "").strip()
            if not label:
                continue

            token = label.replace("_", " ").strip().lower()

            # Skip linking verbs and very short labels
            if token in linking_verbs or len(token) < 2:
                continue

            src_tok = node_token_fn(edge.source_node)
            dst_tok = node_token_fn(edge.dest_node)
            if not src_tok or not dst_tok:
                continue

            src_raw = node_raw_score.get(edge.source_node, max_distance + 1)
            dst_raw = node_raw_score.get(edge.dest_node, max_distance + 1)
            src_act = self.convert_to_landscape_score(src_raw)
            dst_act = self.convert_to_landscape_score(dst_raw)
            endpoint_act = max(src_act, dst_act)
            verb_act = endpoint_act - 0.5
            if verb_act < 0.0:
                verb_act = 0.0

            prev = verb_scores.get(token)
            if prev is None or verb_act > prev:
                verb_scores[token] = verb_act
                verb_onset_scores[token] = endpoint_act
                logging.debug(
                    f"Recording verb '{token}' from active edge: "
                    f"{edge.source_node.get_text_representer()} -{label}-> "
                    f"{edge.dest_node.get_text_representer()} (score={verb_act})"
                )

        for token in verb_scores:
            if verb_onset_scores.get(token, 0.0) > 0.0 and self.is_activation_onset(token, sentence_id):
                verb_scores[token] = 5.0

        for token, score in verb_scores.items():
            append_record_fn({"sentence": sentence_id, "token": token, "score": clamp_matrix_score(score)})
            
        self._max_sentence_index = max(self._max_sentence_index, sentence_id)
        
        # Collect all scores to update _full_activation_matrix
        all_scores: Dict[str, float] = {}
        for node in self._graph.nodes:
            token = node_token_fn(node)
            if token:
                dist = distances.get(node, max_distance + 1)
                if dist > max_distance:
                    score = 0.0
                else:
                    base_score = self.convert_to_landscape_score(dist)
                    seniority = 0
                    if node.first_seen_sentence is not None:
                        if node in explicit_set:
                            # Fix C: re-mentioned node → no age penalty in matrix record
                            seniority = 0
                        else:
                            seniority = max(0, sentence_id - node.first_seen_sentence)
                    score = max(0.0, base_score - CARRYOVER_SENIORITY_WEIGHT * seniority)
                if score > 0.0 and self.is_activation_onset(token, sentence_id):
                    score = 5.0
                all_scores[token] = clamp_matrix_score(score)

        for token, score in verb_scores.items():
            all_scores[token] = clamp_matrix_score(score)
            
        for token, score in all_scores.items():
            if token not in self._full_activation_matrix:
                self._full_activation_matrix[token] = [0.0] * (self._max_sentence_index - 1)
            while len(self._full_activation_matrix[token]) < self._max_sentence_index:
                self._full_activation_matrix[token].append(0.0)
            self._full_activation_matrix[token][sentence_id - 1] = score

    def export_activation_matrix_csv(self, output_path: str) -> None:
        if not self._full_activation_matrix:
            logging.warning("No activation matrix data to export")
            return

        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        max_len = max(len(scores) for scores in self._full_activation_matrix.values())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["token"] + [str(i + 1) for i in range(max_len)])
            for token, scores in sorted(self._full_activation_matrix.items()):
                padded = scores + [0.0] * (max_len - len(scores))
                writer.writerow([token] + padded)

        logging.info(f"Full activation matrix exported to {output_path}")
    
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
                if not edge.active or edge.visibility_score < MATRIX_MIN_PROPAGATION_VISIBILITY:
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

    def get_last_context_shift(self) -> bool:
        return self._last_context_shift

    def get_decay_decisions_with_triplets(
        self,
    ) -> List[Tuple[Tuple[str, str, str], str]]:
        result = []
        for decision in self._last_decay_decisions:
            if decision.action in ("removed", "decayed", "protected", "maintained"):
                result.append((decision.triplet, decision.reasoning))
        return result

    def post_sentence_cleanup(self, prev_sentences):
        # First run semantic decay
        self._last_decay_decisions = self.apply_semantic_edge_decay()

        # Then pruning
        self.apply_pruning(prev_sentences, aggressive=True)

        # Enforce active/visibility invariant
        for edge in self._graph.edges:
            edge.active = edge.visibility_score > 0

        self.prune_inactive_edgeless_nodes()

        # Enforce active/visibility invariant (post-pruning safety net)
        for edge in self._graph.edges:
            edge.active = edge.visibility_score > 0