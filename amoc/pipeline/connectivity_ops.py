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
        # Design:
        # 1. Deterministic repair via StabilityOps (reactivate cumulative edges)
        # 2. LLM repair (create_forced_edges_fn)
        # 3. Fallback "relates_to" edges as last resort
        rollback_needed = False
        explicit_nodes = self._get_explicit_nodes()
        carryover_nodes = self._get_carryover_nodes()
        required_nodes = explicit_nodes | carryover_nodes
        # Deterministic repair
        if self._graph.enforce_connectivity(required_nodes, allow_reactivation=True):
            return False

        logging.debug("Deterministic repair failed, trying LLM repair")

        # LLM repair - try twice
        for _ in range(2):
            create_forced_edges_fn(
                story_context=(
                    " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                ),
                current_sentence=current_sentence_text,
                mode="active",
            )
            if self.is_active_connected():
                return False

        logging.debug("LLM repair failed, trying fallback edges")

        # Fallback
        components, _ = self._graph.get_disconnected_components(required_nodes)

        if len(components) > 1:
            components = sorted(components, key=len, reverse=True)
            backbone = set(components[0])

            for comp in components[1:]:
                # Only connect components with protected nodes
                if not (set(comp) & required_nodes):
                    continue

                node_a = next(iter(backbone))
                node_b = next(iter(comp))

                edge = self._graph.add_edge(
                    node_a,
                    node_b,
                    "relates_to",
                    self._edge_visibility,
                    persona_influenced=False,
                    inferred=False,
                )

                if edge:
                    edge.asserted_this_sentence = False
                    edge.reactivated_this_sentence = False
                    backbone.update(comp)

        # Ensure explicit nodes are connected
        self._ensure_explicit_nodes_connected(explicit_nodes)

        if not self.is_active_connected():
            rollback_needed = True

        # Check cumulative connectivity
        if not rollback_needed and not self.is_cumulative_connected():
            self._repair_cumulative_connectivity()

            if not self.is_cumulative_connected():
                rollback_needed = True

        return rollback_needed

    def _ensure_explicit_nodes_connected(self, explicit_nodes: set) -> None:
        # deterministic repair
        if self._graph.enforce_connectivity(explicit_nodes, allow_reactivation=True):
            return

        # Fallback
        components, _ = self._graph.get_disconnected_components(explicit_nodes)

        if len(components) <= 1:
            return

        components = sorted(components, key=len, reverse=True)
        backbone = set(components[0])

        for comp in components[1:]:
            has_explicit = any(n in explicit_nodes for n in comp)
            if not has_explicit:
                continue

            node_a = next(iter(backbone))
            node_b = next(n for n in comp if n in explicit_nodes)

            edge = self._graph.add_edge(
                node_a,
                node_b,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )

            if edge:
                edge.asserted_this_sentence = False
                edge.reactivated_this_sentence = False
                backbone.update(comp)

    def _repair_cumulative_connectivity(self) -> None:
        if self.is_cumulative_connected():
            return

        # Build cumulative graph to find components
        G_full = self._graph.to_networkx()

        if G_full.number_of_nodes() <= 1:
            return

        components = list(nx.connected_components(G_full))

        if len(components) <= 1:
            return

        logging.warning(
            "Cumulative graph fragmented into %d components",
            len(components),
        )

        components = sorted(components, key=len, reverse=True)
        backbone = set(components[0])

        for comp in components[1:]:
            node_a = next(iter(backbone))
            node_b = next(iter(comp))

            edge = self._graph.add_edge(
                node_a,
                node_b,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )

            if edge:
                backbone.update(comp)

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

    # Edge case: dangling nodes - illegal state - must be repaired
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

        rollback_needed = False

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
                rollback_needed = True
                break

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

                relation = normalize_edge_label_fn(relation) or "relates_to"

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
                edge = self._graph.add_edge(
                    node,
                    anchor,
                    "relates_to",
                    self._edge_visibility,
                    persona_influenced=False,
                    inferred=False,
                )

                if edge:
                    repair_success = True

            if not repair_success:
                rollback_needed = True
                break

        return rollback_needed

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

        edges_created = set()

        for comp in sorted_components[1:]:

            representative = next(iter(comp))
            anchor_node = next(iter(main_component))

            if representative not in active_nodes:
                continue
            if anchor_node not in active_nodes:
                continue

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
