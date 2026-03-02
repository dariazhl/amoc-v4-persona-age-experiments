from typing import TYPE_CHECKING, Set, List, Tuple, Optional
import logging
import networkx as nx

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node
    from amoc.graph.edge import Edge


class StabilityOps:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def _build_active_graph(
        self, include_nodes: Optional[Set["Node"]] = None
    ) -> nx.Graph:
        G = nx.Graph()
        for e in self._graph.edges:
            if e.active and e.visibility_score > 0:
                G.add_edge(e.source_node, e.dest_node, edge=e)
        if include_nodes:
            for node in include_nodes:
                G.add_node(node)
        return G

    def _build_cumulative_graph(self) -> nx.Graph:
        return self._graph.to_networkx()

    def is_active_connected(self, required_nodes: Optional[Set["Node"]] = None) -> bool:
        G = self._build_active_graph(include_nodes=required_nodes)
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)

    def is_cumulative_connected(self) -> bool:
        if not self._graph.edges:
            return True
        G = self._build_cumulative_graph()
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)

    def get_disconnected_components(
        self, focus_nodes: Set["Node"]
    ) -> Tuple[List[Set["Node"]], int]:
        G = self._build_active_graph(include_nodes=focus_nodes)
        if G.number_of_nodes() <= 1:
            return ([set(G.nodes())] if G.number_of_nodes() == 1 else [], 0)

        components = [set(c) for c in nx.connected_components(G)]
        if len(components) <= 1:
            return (components, 0)

        focus_idx = 0
        max_focus_count = 0
        for idx, comp in enumerate(components):
            focus_in_comp = len(comp & focus_nodes)
            if focus_in_comp > max_focus_count:
                max_focus_count = focus_in_comp
                focus_idx = idx

        return (components, focus_idx)

    def can_connect_via_cumulative(self, required_nodes: Set["Node"]) -> bool:
        if self.is_active_connected(required_nodes):
            return True

        G_cumulative = self._build_cumulative_graph()
        G_active = self._build_active_graph(include_nodes=required_nodes)

        if G_active.number_of_nodes() <= 1:
            return True

        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return True

        for i, comp_a in enumerate(components):
            for comp_b in components[i + 1 :]:
                can_connect = False
                for node_a in comp_a:
                    if node_a not in G_cumulative:
                        continue
                    for node_b in comp_b:
                        if node_b not in G_cumulative:
                            continue
                        if nx.has_path(G_cumulative, node_a, node_b):
                            can_connect = True
                            break
                    if can_connect:
                        break
                if not can_connect:
                    return False
        return True

    def reconnect_via_cumulative(self, required_nodes: Set["Node"]) -> Set["Edge"]:
        if self.is_active_connected(required_nodes):
            return set()

        G_cumulative = self._build_cumulative_graph()
        G_active = self._build_active_graph(include_nodes=required_nodes)

        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return set()

        components = sorted(components, key=len, reverse=True)
        focus_comp = set(components[0])
        reactivated: Set["Edge"] = set()

        for comp in components[1:]:
            best_path = None
            best_len = float("inf")

            for src in comp:
                if src not in G_cumulative:
                    continue
                for tgt in focus_comp:
                    if tgt not in G_cumulative:
                        continue
                    try:
                        path = nx.shortest_path(G_cumulative, src, tgt)
                        if len(path) < best_len:
                            best_path = path
                            best_len = len(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            if best_path:
                for i in range(len(best_path) - 1):
                    edge_data = G_cumulative.get_edge_data(
                        best_path[i], best_path[i + 1]
                    )
                    if edge_data:
                        edge = edge_data.get("edge")
                        if edge and not edge.active:
                            edge.active = True
                            edge.visibility_score = max(edge.visibility_score, 1)
                            edge.source_node.active = True
                            edge.dest_node.active = True
                            reactivated.add(edge)
                focus_comp.update(comp)

        return reactivated

    def enforce_connectivity(
        self,
        required_nodes: Set["Node"],
        allow_reactivation: bool = True,
        enforce_cumulative: bool = False,
    ) -> bool:
        #  already connected
        if self.is_active_connected(required_nodes):
            if enforce_cumulative and not self.is_cumulative_connected():
                # This should not happen in normal operation
                logging.warning("Cumulative graph fragmented")
            return True

        # deterministic repair via cumulative edge reactivation
        if allow_reactivation:
            reactivated = self.reconnect_via_cumulative(required_nodes)

        active_connected = self.is_active_connected(required_nodes)

        if active_connected and enforce_cumulative:
            cumulative_connected = self.is_cumulative_connected()
            if not cumulative_connected:
                logging.warning("Active connected but cumulative fragmented")

        return active_connected

    def enforce_cumulative_stability(
        self,
        explicit_nodes: set,
    ) -> None:
        # connectivity for cumulative graph
        active_nodes, active_edges = self._graph.get_active_subgraph()

        active_empty = len(active_nodes) == 0
        explicit_active = any(node.active for node in explicit_nodes)
        all_inactive = all(not node.active for node in self._graph.nodes)

        if explicit_active and not all_inactive and not active_empty:
            return

        if active_empty:
            visible_edges = {e for e in self._graph.edges if e.visibility_score > 0}

            visible_nodes = {e.source_node for e in visible_edges} | {
                e.dest_node for e in visible_edges
            }

            new_nodes = set(visible_nodes)
            new_edges = set(visible_edges)

        else:
            new_nodes = set(active_nodes)
            new_edges = set(active_edges)

        for node in new_nodes:
            node.edges = []

        for edge in new_edges:
            edge.source_node.edges.append(edge)
            edge.dest_node.edges.append(edge)

        self._graph.nodes = new_nodes
        self._graph.edges = new_edges

    def enforce_carryover_connectivity(self, carryover_nodes: set) -> None:
        # carryover nodes get disconnected sometimes - added guard
        if not carryover_nodes:
            return

        G = self._build_active_graph()

        degree_map = {}
        for e in self._graph.edges:
            if e.active and e.visibility_score > 0:
                degree_map[e.source_node] = degree_map.get(e.source_node, 0) + 1
                degree_map[e.dest_node] = degree_map.get(e.dest_node, 0) + 1

        for node in list(carryover_nodes):
            if degree_map.get(node, 0) == 0:
                node.active = False

        G = self._build_active_graph()
        sub = G.subgraph(carryover_nodes)
        components = list(nx.connected_components(sub))

        if len(components) <= 1:
            return

        for comp_a in components:
            for comp_b in components:
                if comp_a is comp_b:
                    continue

                for e in self._graph.edges:
                    if not (
                        (e.source_node in comp_a and e.dest_node in comp_b)
                        or (e.source_node in comp_b and e.dest_node in comp_a)
                    ):
                        continue

                    e.active = True
                    e.visibility_score = max(e.visibility_score, 1)
                    e.source_node.active = True
                    e.dest_node.active = True

                    G = self._build_active_graph()
                    sub = G.subgraph(carryover_nodes)
                    components = list(nx.connected_components(sub))

                    if len(components) <= 1:
                        return
