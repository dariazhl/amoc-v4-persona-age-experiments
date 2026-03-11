from typing import TYPE_CHECKING, Set, List, Tuple, Optional
import logging
import networkx as nx

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge


class ConnectivityRepair:
    def __init__(self, graph_ref: "Graph"):
        self._graph = graph_ref

    def build_active_graph(
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

    def build_cumulative_graph(self) -> nx.Graph:
        return self._graph.to_networkx()

    # check if the active subgraph is connected
    def is_active_subgraph_connected(
        self, required_nodes: Optional[Set["Node"]] = None
    ) -> bool:
        G = self.build_active_graph(include_nodes=required_nodes)
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)

    def is_cumulative_graph_connected(self) -> bool:
        if not self._graph.edges:
            return True
        G = self.build_cumulative_graph()
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)

    def get_disconnected_components(
        self, focus_nodes: Set["Node"]
    ) -> Tuple[List[Set["Node"]], int]:
        G = self.build_active_graph(include_nodes=focus_nodes)
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
        if self.is_active_subgraph_connected(required_nodes):
            return True

        G_cumulative = self.build_cumulative_graph()
        G_active = self.build_active_graph(include_nodes=required_nodes)

        if G_active.number_of_nodes() <= 1:
            return True

        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return True

        for i, comp_a in enumerate(components):
            for comp_b in components[i + 1 :]:
                if not any(
                    node_a in G_cumulative
                    and node_b in G_cumulative
                    and nx.has_path(G_cumulative, node_a, node_b)
                    for node_a in comp_a
                    for node_b in comp_b
                ):
                    return False
        return True

    # issue: cumulative graph can be fragmented
    # purpose: reconnect disconnected cumulative graph
    def reconnect_via_cumulative(self, required_nodes: Set["Node"]) -> Set["Edge"]:
        if self.is_active_subgraph_connected(required_nodes):
            return set()

        G_cumulative = self.build_cumulative_graph()
        G_active = self.build_active_graph(include_nodes=required_nodes)

        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return set()

        components = sorted(components, key=len, reverse=True)
        focus_comp = set(components[0])
        reactivated = set()

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
                            reactivated.add(edge)
                focus_comp.update(comp)

        return reactivated

    def restore_connectivity(
        self,
        required_nodes: Set["Node"],
        allow_reactivation: bool = True,
        enforce_cumulative: bool = False,
    ) -> bool:
        #  already connected
        if self.is_active_subgraph_connected(required_nodes):
            if enforce_cumulative and not self.is_cumulative_graph_connected():
                # This should not happen in normal operation
                logging.warning("Cumulative graph fragmented")
            return True

        # deterministic repair via cumulative edge reactivation
        if allow_reactivation:
            self.reconnect_via_cumulative(required_nodes)

        active_connected = self.is_active_subgraph_connected(required_nodes)

        if active_connected and enforce_cumulative:
            if not self.is_cumulative_graph_connected():
                logging.warning("Active connected but cumulative fragmented")

        return active_connected

    def stabilize_cumulative_graph(
        self,
        explicit_nodes: set,
    ) -> None:
        active_nodes, active_edges = self._graph.get_active_subgraph_wrapper()

        active_empty = len(active_nodes) == 0
        explicit_active = any(node.active for node in explicit_nodes)
        all_inactive = all(not node.active for node in self._graph.nodes)

        if explicit_active and not all_inactive and not active_empty:
            return

        # Reactivate visible edges when active subgraph is empty
        if active_empty:
            for edge in self._graph.edges:
                if edge.visibility_score > 0 and not edge.active:
                    edge.active = True
                    logging.info(
                        f"reactivated visible edge "
                        f"'{edge.source_node.get_text_representer()}' -> "
                        f"'{edge.dest_node.get_text_representer()}'"
                    )

    def ensure_carryover_connected(self, carryover_nodes: set) -> None:
        # carryover nodes get disconnected sometimes - added guard
        if not carryover_nodes:
            return

        G = self.build_active_graph()

        degree_map = {}
        for e in self._graph.edges:
            if e.active and e.visibility_score > 0:
                degree_map[e.source_node] = degree_map.get(e.source_node, 0) + 1
                degree_map[e.dest_node] = degree_map.get(e.dest_node, 0) + 1

        # Nodes with no active edges are automatically inactive via property

        G = self.build_active_graph()
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

                    G = self.build_active_graph()
                    sub = G.subgraph(carryover_nodes)
                    components = list(nx.connected_components(sub))

                    if len(components) <= 1:
                        return
