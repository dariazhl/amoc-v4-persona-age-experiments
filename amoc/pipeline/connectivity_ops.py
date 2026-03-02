from typing import TYPE_CHECKING, Set, Optional, List
import networkx as nx
import logging
import re
import json

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge
from amoc.prompts.amoc_prompts import FORCED_CONNECTIVITY_EDGE_PROMPT


class ConnectivityOps:
    def __init__(
        self,
        graph_ref: "Graph",
        get_explicit_nodes: callable,
        get_carryover_nodes: callable,
        edge_visibility: int,
        client_ref=None,
    ):
        self._graph = graph_ref
        self._get_explicit_nodes = get_explicit_nodes
        self._get_carryover_nodes = get_carryover_nodes
        self._edge_visibility = edge_visibility
        self._client = client_ref
        self._story_text: str = ""
        self._current_sentence_text: str = ""

    def set_context(self, story_text: str, current_sentence_text: str):
        self._story_text = story_text
        self._current_sentence_text = current_sentence_text

    def set_anchor_nodes(self, anchor_nodes: set) -> None:
        self._anchor_nodes = anchor_nodes

    def is_active_connected(self) -> bool:
        explicit_nodes = self._get_explicit_nodes()
        return self._graph.is_active_connected(explicit_nodes)

    def is_cumulative_connected(self) -> bool:
        return self._graph.is_cumulative_connected()

    def enforce_connectivity(
        self,
        prev_sentences: list,
        current_sentence_text: str,
        create_forced_edges_fn: callable,
    ) -> bool:
        explicit_nodes = self._get_explicit_nodes()
        carryover_nodes = self._get_carryover_nodes()
        required_nodes = explicit_nodes | carryover_nodes

        # PHASE 1: Deterministic repair via StabilityOps (reactivate cumulative edges)
        if self._graph.enforce_connectivity(required_nodes, allow_reactivation=True):
            if self.is_active_connected() and self.is_cumulative_connected():
                return False

        logging.debug("Deterministic repair insufficient, trying LLM repair")

        # PHASE 2: LLM repair - try twice
        for attempt in range(2):
            create_forced_edges_fn(
                story_context=(
                    " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                ),
                current_sentence=current_sentence_text,
                mode="active",
            )
            if self.is_active_connected():
                logging.debug("LLM repair succeeded on attempt %d", attempt + 1)
                break

        # Check if LLM repair was sufficient
        if self.is_active_connected() and self.is_cumulative_connected():
            return False

        logging.debug("LLM repair insufficient, applying fallback edges")

        # PHASE 3: Final fallback "relates_to" edges
        self._apply_relates_to_fallback(required_nodes)

        if not self.is_active_connected():
            logging.error("Active graph still disconnected after all repairs")
            return True  # rollback required

        # Verify cumulative connectivity
        if not self.is_cumulative_connected():
            self._apply_cumulative_fallback()

            if not self.is_cumulative_connected():
                logging.error("Cumulative graph still disconnected after all repairs")
                return True  # rollback required

        return False

    def _apply_relates_to_fallback(self, required_nodes: set) -> None:
        components, _ = self._graph.get_disconnected_components(required_nodes)

        if len(components) <= 1:
            return

        # Sort components by size
        components = sorted(components, key=len, reverse=True)
        largest_component = set(components[0])

        # Helper: deterministic node sorting key (degree DESC, then text ASC)
        def node_sort_key(n):
            degree = sum(
                1
                for e in self._graph.edges
                if e.active and (e.source_node == n or e.dest_node == n)
            )
            return (-degree, n.get_text_representer())

        anchor_nodes = getattr(self, "_anchor_nodes", set())

        anchor_candidates = sorted(
            [n for n in largest_component if n in anchor_nodes],
            key=lambda n: n.get_text_representer(),
        )
        if anchor_candidates:
            backbone_node = anchor_candidates[0]
        else:

            backbone_node = min(largest_component, key=node_sort_key)

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
                node_small = explicit_in_comp[0]
            else:
                node_small = min(comp_set, key=node_sort_key)

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
                logging.debug(
                    "Created fallback edge: %s -[relates_to]-> %s",
                    node_small.get_text_representer(),
                    backbone_node.get_text_representer(),
                )

    def _apply_cumulative_fallback(self) -> None:
        G_full = self._graph.to_networkx()

        if G_full.number_of_nodes() <= 1:
            return

        components = list(nx.connected_components(G_full))

        if len(components) <= 1:
            return

        # Map string names back to nodes
        name_to_node = {n.get_text_representer(): n for n in self._graph.nodes}

        # Sort by size: largest first
        components = sorted(components, key=len, reverse=True)
        largest = set(components[0])

        # DETERMINISTIC
        node_large_name = min(largest)

        # Connect smallest → largest
        for comp in sorted(components[1:], key=len):
            node_small_name = min(comp)

            node_small = name_to_node.get(node_small_name)
            node_large = name_to_node.get(node_large_name)

            if node_small is None or node_large is None:
                continue

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
                logging.debug(
                    "Created cumulative fallback edge: %s -[relates_to]-> %s",
                    node_small_name,
                    node_large_name,
                )

    def _get_nodes_with_active_edges(self) -> set:
        nodes = set()
        for edge in self._graph.edges:
            if edge.active:
                nodes.add(edge.source_node)
                nodes.add(edge.dest_node)
        return nodes

    def validate_sentence_state(self) -> bool:
        if not self.is_active_connected():
            return False

        explicit_nodes = self._get_explicit_nodes()

        for node in explicit_nodes:
            if node not in self._graph.nodes:
                return False

            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self._graph.edges
            )

            if not has_active_edge:
                if len(explicit_nodes) == 1:
                    continue
                return False

        return True

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

        logging.debug(
            "Found %d dangling nodes, attempting LLM repair",
            len(dangling_nodes),
        )

        any_repair_failed = False

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
                result = self._client.get_forced_connectivity_edge_label(
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

                edge = self._graph.add_edge(
                    node,
                    anchor,
                    relation,
                    self._edge_visibility,
                    inferred=True,
                )

                if edge:
                    repair_success = True
                    logging.debug(
                        "LLM repair for dangling node: %s -[%s]-> %s",
                        node.get_text_representer(),
                        relation,
                        anchor.get_text_representer(),
                    )
                    break

            if not repair_success:
                any_repair_failed = True
                logging.debug(
                    "LLM repair failed for dangling node: %s",
                    node.get_text_representer(),
                )

        # Return True if any repair failed - caller should invoke enforce_connectivity
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

        # Helper for deterministic node selection (degree DESC, text ASC)
        def node_sort_key(n):
            degree = sum(
                1 for e in active_edges if e.source_node == n or e.dest_node == n
            )
            return (-degree, n.get_text_representer())

        # DETERMINISTIC: Select anchor from main component
        anchor_node = min(
            [n for n in main_component if n in active_nodes],
            key=node_sort_key,
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
            representative = min(candidates, key=node_sort_key)

            prompt_text = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
                node_a=representative.get_text_representer(),
                node_b=anchor_node.get_text_representer(),
                story_context=self._story_text[:1500],
                current_sentence=self._current_sentence_text,
            )

            try:
                # no persona injection for connectivity repair
                response = self._client.generate_raw(
                    prompt_text=prompt_text,
                    temperature=temperature,
                )

                match = re.search(r"\{.*?\}", response, re.DOTALL)
                if not match:
                    continue

                match = re.search(r"\{.*\}", response, re.DOTALL)
                if not match:
                    continue
                data = json.loads(match.group())
                label = data.get("label")

            except Exception as e:
                logging.warning(
                    "JSON extraction failed: %s",
                    str(e),
                )
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
        if not self.is_cumulative_connected():
            logging.warning("Cumulative graph disconnected - plots may show fragments")
