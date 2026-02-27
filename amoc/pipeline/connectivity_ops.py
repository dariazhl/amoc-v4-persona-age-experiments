from typing import TYPE_CHECKING, Set, Optional, List
import networkx as nx
import logging
import re
import json

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge

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
        active_edges = [e for e in self._graph.edges if e.active]

        G = nx.Graph()

        active_nodes = self._get_nodes_with_active_edges()

        for node in active_nodes:
            G.add_node(node)

        for node in self._get_explicit_nodes():
            G.add_node(node)

        for e in active_edges:
            G.add_edge(e.source_node, e.dest_node)

        if G.number_of_nodes() <= 1:
            return True

        return nx.is_connected(G)

    def is_cumulative_connected(self) -> bool:
        if not self._graph.nodes:
            return True

        G = nx.Graph()

        for node in self._graph.nodes:
            G.add_node(node)

        for e in self._graph.edges:
            G.add_edge(e.source_node, e.dest_node)

        if G.number_of_nodes() <= 1:
            return True

        return nx.is_connected(G)

    def _get_nodes_with_active_edges(self) -> set:
        nodes = set()
        for edge in self._graph.edges:
            if edge.active:
                nodes.add(edge.source_node)
                nodes.add(edge.dest_node)
        return nodes

    def enforce_connectivity(
        self,
        prev_sentences: list,
        current_sentence_text: str,
        create_forced_edges_fn: callable,
    ) -> bool:
        rollback_needed = False
        explicit_nodes = self._get_explicit_nodes()
        protected_nodes = explicit_nodes | self._get_carryover_nodes()

        if not self.is_active_connected():
            repair_success = False

            for _ in range(2):
                create_forced_edges_fn(
                    story_context=(
                        " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                    ),
                    current_sentence=current_sentence_text,
                    mode="active",
                )
                if self.is_active_connected():
                    repair_success = True
                    break

            if not repair_success:
                G = nx.Graph()

                for node in explicit_nodes:
                    G.add_node(node)

                for node in self._get_nodes_with_active_edges():
                    G.add_node(node)

                for e in self._graph.edges:
                    if e.active:
                        G.add_edge(e.source_node, e.dest_node)

                components = list(nx.connected_components(G))

                if len(components) > 1:
                    components = sorted(components, key=len, reverse=True)
                    backbone = set(components[0])

                    for comp in components[1:]:
                        if not (set(comp) & protected_nodes):
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
                            edge.active = True
                            edge.asserted_this_sentence = False
                            edge.reactivated_this_sentence = False
                            backbone.update(comp)

                repair_success = self.is_active_connected()

            self._ensure_explicit_nodes_connected(explicit_nodes)

            if not self.is_active_connected():
                rollback_needed = True

        if not rollback_needed and not self.is_cumulative_connected():
            self._repair_cumulative_connectivity()

            if not self.is_cumulative_connected():
                rollback_needed = True

        return rollback_needed

    def _ensure_explicit_nodes_connected(self, explicit_nodes: set) -> None:
        active_edges = [e for e in self._graph.edges if e.active]

        G_active = nx.Graph()

        for node in explicit_nodes:
            G_active.add_node(node)

        for e in active_edges:
            G_active.add_edge(e.source_node, e.dest_node)

        if G_active.number_of_nodes() > 1:
            components = list(nx.connected_components(G_active))
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
                    edge.active = True
                    edge.asserted_this_sentence = False
                    edge.reactivated_this_sentence = False
                    backbone.update(comp)

    def _repair_cumulative_connectivity(self) -> None:
        G_full = nx.Graph()

        for node in self._graph.nodes:
            G_full.add_node(node)

        for e in self._graph.edges:
            G_full.add_edge(e.source_node, e.dest_node)

        components = list(nx.connected_components(G_full))

        if len(components) > 1:
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

    def enforce_cumulative_connectivity_simple(self) -> None:
        G = nx.Graph()

        for edge in self._graph.edges:
            G.add_edge(edge.source_node, edge.dest_node)

        if len(G.nodes) == 0:
            return

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

        active_nodes = set(per_sentence_view.explicit_nodes) | set(per_sentence_view.carryover_nodes)

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
                        " ".join(prev_sentences[:-1])
                        if len(prev_sentences) > 1
                        else ""
                    ),
                    current_sentence=self._current_sentence_text,
                    persona=persona,
                )

                relation = (
                    result.get("label") if isinstance(result, dict) else result
                )
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
                    edge.active = True
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
                    edge.active = True
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
        from amoc.prompts.amoc_prompts import FORCED_CONNECTIVITY_EDGE_PROMPT

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

            messages = [{"role": "user", "content": prompt_text}]

            try:
                response = self._client.generate(
                    messages,
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
                    "[ConnectivityRepair] JSON extraction failed: %s",
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
