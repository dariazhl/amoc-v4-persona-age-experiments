from typing import TYPE_CHECKING, Set, Optional, List
import networkx as nx
import logging
import re
import json

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge
from amoc.prompts.amoc_prompts import FORCED_CONNECTIVITY_EDGE_PROMPT


# Old code: no connectivity enforced
# New code: same principle applied on new code creates fragmentation
# Design: enforce connectivity at sentence level, with repair + fallback
class ConnectivityStabilizer:
    def __init__(
        self,
        graph_ref: "Graph",
        get_explicit_nodes: callable,
        get_carryover_nodes: callable,
        edge_visibility: int,
        llm_extractor=None,
    ):
        self._graph = graph_ref
        self._get_explicit_nodes = get_explicit_nodes
        self._get_carryover_nodes = get_carryover_nodes
        self._edge_visibility = edge_visibility
        self._llm = llm_extractor
        self._story_text: str = ""
        self._current_sentence_text: str = ""
        self._anchor_nodes: set = set()

    def set_context(self, story_text: str, current_sentence_text: str):
        self._story_text = story_text
        self._current_sentence_text = current_sentence_text

    def set_anchor_nodes(self, anchor_nodes: set) -> None:
        self._anchor_nodes = anchor_nodes

    def is_active_connected_wrapper(self) -> bool:
        required_nodes = self._get_explicit_nodes() | self._get_carryover_nodes()
        return self._graph.is_active_connected(required_nodes)

    def is_cumulative_connected_wrapper(self) -> bool:
        return self._graph.is_cumulative_connected()

    # Main pipeline to enforce connectivity with repair and fallback
    # Step 1. Deterministic repair
    # Step 2. LLM repair - try twice
    # Step 3. Final fallback with "relates_to" edges
    def run_connectivity_pipeline(
        self,
        prev_sentences: list,
        current_sentence_text: str,
        create_forced_edges_fn: callable,
    ) -> bool:
        explicit_nodes = self._get_explicit_nodes()
        carryover_nodes = self._get_carryover_nodes()
        required_nodes = explicit_nodes | carryover_nodes

        # Deterministic repair via ConnectivityRepair (reactivate cumulative edges)
        if self._graph.enforce_connectivity(required_nodes, allow_reactivation=True):
            if (
                self.is_active_connected_wrapper()
                and self.is_cumulative_connected_wrapper()
            ):
                return False

        # LLM repair - try 2x
        for attempt in range(2):
            create_forced_edges_fn(
                story_context=(
                    " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                ),
                current_sentence=current_sentence_text,
                mode="active",
            )
            if self.is_active_connected_wrapper():
                break

        if (
            self.is_active_connected_wrapper()
            and self.is_cumulative_connected_wrapper()
        ):
            return False

        # Final fallback "relates_to" edges
        self.apply_relates_to_fallback(required_nodes)

        if not self.is_active_connected_wrapper():
            logging.error("Active graph disconnected after all repairs")
            return True

        if not self.is_cumulative_connected_wrapper():
            self.connect_cumulative_components()
            if not self.is_cumulative_connected_wrapper():
                logging.error("Cumulative graph disconnected after all repairs")
                return True

        return False

    def node_sort_key(self, node, edge_pool):
        degree = sum(
            1 for e in edge_pool if e.source_node == node or e.dest_node == node
        )
        return (-degree, node.get_text_representer())

    # fallback = if deterministic + LLM repair fair, add generic "relates_to" edges
    # this is a last resort before rolling back to previous sentence state
    def apply_relates_to_fallback(self, required_nodes: set) -> None:
        # find disconnected components
        components, _ = self._graph.get_disconnected_components_wrapper(required_nodes)

        if len(components) <= 1:
            return

        # sort components by size
        components = sorted(components, key=len, reverse=True)
        largest_component = set(components[0])

        active_edge_pool = [e for e in self._graph.edges if e.active]

        active_nodes = self.get_nodes_with_active_edges()
        # find anchor candidates
        anchor_candidates = [
            n
            for n in largest_component
            if n in self._anchor_nodes and any(e.active for e in n.edges)
        ]

        if anchor_candidates:

            def anchor_score(n):
                active_degree = sum(1 for e in n.edges if e.active)
                return (
                    n.activation_score,
                    active_degree,
                    len(n.edges),
                    n.get_text_representer(),  # deterministic
                )

            backbone_node = max(anchor_candidates, key=anchor_score)
        else:
            backbone_node = max(
                largest_component,
                key=lambda n: (
                    n.activation_score,
                    sum(1 for e in n.edges if e.active),
                    len(n.edges),
                    n.get_text_representer(),
                ),
            )

        for comp in sorted(components[1:], key=len):
            comp_set = set(comp)
            if not (comp_set & required_nodes):
                continue

            explicit_nodes = self._get_explicit_nodes()
            explicit_in_comp = sorted(
                [n for n in comp_set if n in explicit_nodes],
                key=lambda n: n.get_text_representer(),
            )
            if explicit_in_comp:
                # if explicit nodes exist, pick the first one
                node_small = explicit_in_comp[0]
            else:
                # else pick the node with the smallest degree
                node_small = min(
                    comp_set, key=lambda n: self.node_sort_key(n, active_edge_pool)
                )
            # create generic edge
            edge = self._graph.add_edge(
                node_small,
                backbone_node,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )

            if edge:
                edge.asserted_this_sentence = False
                edge.reactivated_this_sentence = False
                largest_component.update(comp_set)

    # ensure cumulative graph is connected
    def connect_cumulative_components(self) -> None:
        G_full = self._graph.to_networkx()

        if G_full.number_of_nodes() <= 1:
            return

        components = list(nx.connected_components(G_full))
        if len(components) <= 1:
            return

        # Sort by size: largest first
        components = sorted(components, key=len, reverse=True)
        largest = set(components[0])

        # Deterministic representative node
        node_large = min(largest, key=lambda n: n.get_text_representer())

        # Connect smallest - largest
        for comp in sorted(components[1:], key=len):
            node_small = min(comp, key=lambda n: n.get_text_representer())
            edge = self._graph.add_edge(
                node_small,
                node_large,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )
            if edge:
                largest.update(comp)

    def get_nodes_with_active_edges(self) -> set:
        nodes = set()
        for edge in self._graph.edges:
            if edge.active:
                nodes.add(edge.source_node)
                nodes.add(edge.dest_node)
        return nodes

    def validate_sentence_state(self) -> bool:
        if not self.is_active_connected_wrapper():
            return False

        explicit_nodes = self._get_explicit_nodes()
        for node in explicit_nodes:
            if node not in self._graph.nodes:
                return False
            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self._graph.edges
            )
            if not has_active_edge and len(explicit_nodes) > 1:
                return False
        return True

    # issue: some explicit and carryover nodes in the graph are isolated
    # purpose: handle nodes that appear in the per‑sentence view but have no edges at all in the current active graph
    def repair_dangling_nodes(
        self,
        per_sentence_view,
        prev_sentences: list,
        normalize_edge_label_fn: callable,
        persona: str = "",
    ) -> bool:
        if per_sentence_view is None:
            return False

        active_nodes = set(per_sentence_view.explicit_nodes) | set(
            per_sentence_view.carryover_nodes
        )
        # find dangling nodes with no edges in the active graph
        dangling_nodes = []
        for node in active_nodes:
            has_edge = any(
                e.source_node == node or e.dest_node == node
                for e in per_sentence_view.active_edges
            )
            if not has_edge:
                dangling_nodes.append(node)

        if not dangling_nodes:
            return False

        any_repair_failed = False
        # find anchor candidates from active nodes with edges, sorted by degree
        for node in dangling_nodes:
            repair_success = False

            degree_sorted = sorted(
                per_sentence_view.active_nodes,
                key=lambda n: sum(
                    1
                    for e in per_sentence_view.active_edges
                    if e.source_node == n or e.dest_node == n
                ),
                reverse=True,
            )

            anchor = None
            for candidate in degree_sorted:
                if candidate != node:
                    anchor = candidate
                    break

            if anchor is None:
                any_repair_failed = True
                continue

            # Try LLM repair - 2 attemptS
            for _ in range(2):
                result = self._llm.get_forced_connectivity_edge_label(
                    node_a=node.get_text_representer(),
                    node_b=anchor.get_text_representer(),
                    story_context=(
                        " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                    ),
                    current_sentence=self._current_sentence_text,
                    persona=persona,
                )

                relation = result.get("label") if isinstance(result, dict) else result
                if not relation:
                    continue

                relation = normalize_edge_label_fn(relation)
                if not relation:
                    continue
                # if valid relation returned, add edge to graph with inferred=True
                edge = self._graph.add_edge(
                    node,
                    anchor,
                    relation,
                    self._edge_visibility,
                    inferred=True,
                )

                if edge:
                    repair_success = True
                    break

            if not repair_success:
                any_repair_failed = True

        # return True if any repair failed - caller should invoke enforce_connectivity
        return any_repair_failed

    def repair_connectivity_callback(
        self,
        components,
        active_nodes,
        active_edges,
        sentence_index,
        temperature: float = 0.3,
        forced_pair=None,
    ):
        if forced_pair is not None:
            representative, anchor_node = forced_pair
            components = [{representative}, {anchor_node}]

        if not components or len(components) <= 1:
            return None

        sorted_components = sorted(components, key=len, reverse=True)
        main_component = sorted_components[0]

        # DETERMINISTIC: Select anchor from main component
        anchor_node = min(
            [n for n in main_component if n in active_nodes],
            key=lambda n: self.node_sort_key(n, active_edges),
            default=None,
        )
        if anchor_node is None:
            return None

        edges_created = set()

        for comp in sorted_components[1:]:
            # DETERMINISTIC: Select representative from small component
            candidates = [n for n in comp if n in active_nodes]
            if not candidates:
                continue
            representative = min(
                candidates, key=lambda n: self.node_sort_key(n, active_edges)
            )

            prompt_text = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
                node_a=representative.get_text_representer(),
                node_b=anchor_node.get_text_representer(),
                story_context=self._story_text[:1500],
                current_sentence=self._current_sentence_text,
            )

            try:
                # no persona injection for connectivity repair
                response = self._llm.generate_raw(
                    prompt_text=prompt_text,
                    temperature=temperature,
                )

                if not response:
                    continue

                response = response.strip()

                # Expect direct JSON
                data = json.loads(response)
                label = data.get("label")

            except (json.JSONDecodeError, Exception) as e:
                logging.warning("Connectivity repair failed: %s", str(e))
                continue

            if not label:
                continue

            edge = self._graph.add_edge(
                source_node=representative,
                dest_node=anchor_node,
                label=label.strip().lower(),
                edge_visibility=self._edge_visibility,
                created_at_sentence=sentence_index,
                inferred=True,
            )

            if edge:
                edges_created.add(edge)

        return edges_created if edges_created else None

    def warn_if_cumulative_disconnected(self) -> None:
        if not self.is_cumulative_connected_wrapper():
            logging.warning("Cumulative graph disconnected - plots may show fragments")
