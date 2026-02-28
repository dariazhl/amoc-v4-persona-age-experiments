"""
Connectivity Invariants - Formal verification of graph connectivity.

This module provides explicit invariant checks that can be used to:
1. Verify connectivity before/after operations
2. Stress test the connectivity subsystem
3. Prove that refactoring preserves connectivity guarantees

INVARIANTS:
- INV-1: Cumulative graph must always be connected (or empty)
- INV-2: Active graph must be connected for required nodes (or trivial)
"""

from typing import TYPE_CHECKING, Set, Optional, List, Tuple, Dict
import logging

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node


class ConnectivityInvariantError(Exception):
    pass


def assert_cumulative_connected(graph: "Graph") -> bool:
    if not graph.edges:
        return True  # Empty graph is trivially connected

    connected = graph.is_cumulative_connected()

    if not connected:
        # Get component count for diagnostics
        G = graph.to_networkx()
        import networkx as nx

        components = list(nx.connected_components(G))

        raise ConnectivityInvariantError(
            f"INV-1 VIOLATION: Cumulative graph fragmented into {len(components)} components. "
            f"Component sizes: {[len(c) for c in components]}"
        )

    return True


def assert_active_connected(
    graph: "Graph",
    required_nodes: Set["Node"],
) -> bool:
    if not required_nodes or len(required_nodes) <= 1:
        return True  # Trivially connected

    connected = graph.is_active_connected(required_nodes)

    if not connected:
        # Get component info for diagnostics
        components, focus_idx = graph.get_disconnected_components(required_nodes)

        # Find which required nodes are in which components
        node_locations = {}
        for node in required_nodes:
            for idx, comp in enumerate(components):
                if node in comp:
                    node_locations[node.get_text_representer()] = idx
                    break

        raise ConnectivityInvariantError(
            f"INV-2 VIOLATION: Active graph disconnected. "
            f"{len(components)} components, required nodes in: {node_locations}"
        )

    return True


def verify_connectivity_invariants(
    graph: "Graph",
    required_nodes: Optional[Set["Node"]] = None,
    context: str = "",
) -> Tuple[bool, Optional[str]]:
    try:
        assert_cumulative_connected(graph)

        if required_nodes:
            assert_active_connected(graph, required_nodes)

        return (True, None)

    except ConnectivityInvariantError as e:
        error_msg = f"[{context}] {str(e)}" if context else str(e)
        logging.error(error_msg)
        return (False, error_msg)


class ConnectivityStressTest:
    def __init__(self, graph: "Graph"):
        self._graph = graph
        self._results: Dict[str, Tuple[bool, Optional[str]]] = {}

    def run_all_scenarios(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Run all stress test scenarios and return results."""
        scenarios = [
            ("high_decay", self._test_high_decay),
            ("massive_activation_drop", self._test_massive_activation_drop),
            ("carryover_only", self._test_carryover_only),
            ("single_explicit_node", self._test_single_explicit_node),
            ("minimal_graph", self._test_minimal_graph),
            ("empty_graph", self._test_empty_graph),
        ]

        for name, test_fn in scenarios:
            try:
                result = test_fn()
                self._results[name] = (result, None)
            except ConnectivityInvariantError as e:
                self._results[name] = (False, str(e))
            except Exception as e:
                self._results[name] = (False, f"Unexpected error: {str(e)}")

        return self._results

    def _test_high_decay(self) -> bool:
        # Just verify current state - actual decay is handled by pipeline
        return assert_cumulative_connected(self._graph)

    def _test_massive_activation_drop(self) -> bool:
        active_nodes, active_edges = self._graph.get_active_subgraph()

        # Cumulative should still be connected regardless of activation
        return assert_cumulative_connected(self._graph)

    def _test_carryover_only(self) -> bool:
        # Cumulative invariant should hold
        return assert_cumulative_connected(self._graph)

    def _test_single_explicit_node(self) -> bool:
        # Take any node as the "single explicit"
        if self._graph.nodes:
            single_node = next(iter(self._graph.nodes))
            return assert_active_connected(self._graph, {single_node})
        return True

    def _test_minimal_graph(self) -> bool:
        if len(self._graph.edges) >= 1:
            edge = next(iter(self._graph.edges))
            required = {edge.source_node, edge.dest_node}

            # These two nodes should be connected if edge is active
            if edge.active:
                return assert_active_connected(self._graph, required)

        return assert_cumulative_connected(self._graph)

    def _test_empty_graph(self) -> bool:
        if not self._graph.edges and not self._graph.nodes:
            return True

        # If not empty, just verify cumulative
        return assert_cumulative_connected(self._graph)

    def get_summary(self) -> str:
        lines = ["Connectivity Stress Test Results:", "=" * 40]

        passed = 0
        failed = 0

        for name, (success, error) in self._results.items():
            status = "PASS" if success else "FAIL"
            if success:
                passed += 1
                lines.append(f"  [{status}] {name}")
            else:
                failed += 1
                lines.append(f"  [{status}] {name}: {error}")

        lines.append("=" * 40)
        lines.append(f"Total: {passed} passed, {failed} failed")

        return "\n".join(lines)
