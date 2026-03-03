import pytest
import sys
import importlib.util
from typing import Set, List, Optional
import networkx as nx


def load_module_directly(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Create stub for 'amoc' and 'amoc.graph' packages to prevent __init__.py loading
import types

if "amoc" not in sys.modules:
    sys.modules["amoc"] = types.ModuleType("amoc")
if "amoc.graph" not in sys.modules:
    amoc_graph = types.ModuleType("amoc.graph")
    sys.modules["amoc.graph"] = amoc_graph
    sys.modules["amoc"].graph = amoc_graph

# Load modules in dependency order (bottom-up)
base_path = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/amoc/graph"

# 1. Load leaf modules first (no internal amoc dependencies)
node_module = load_module_directly("amoc.graph.node", f"{base_path}/node.py")
edge_module = load_module_directly("amoc.graph.edge", f"{base_path}/edge.py")

# 2. Load ops modules (depend on node/edge via TYPE_CHECKING only)
activation_ops_module = load_module_directly(
    "amoc.graph.activation_ops", f"{base_path}/activation_ops.py"
)
stability_ops_module = load_module_directly(
    "amoc.graph.stability_ops", f"{base_path}/stability_ops.py"
)
provenance_module = load_module_directly(
    "amoc.graph.provenance_ops", f"{base_path}/provenance_ops.py"
)

# 3. Finally load graph module (imports all the above)
graph_module = load_module_directly("amoc.graph.graph", f"{base_path}/graph.py")

Node = node_module.Node
NodeType = node_module.NodeType
NodeSource = node_module.NodeSource
NodeProvenance = node_module.NodeProvenance
Edge = edge_module.Edge
Graph = graph_module.Graph


def cumulative_is_connected(graph: Graph) -> bool:
    G = nx.Graph()

    # Add all edges (regardless of active state)
    for edge in graph.edges:
        G.add_edge(edge.source_node, edge.dest_node)

    # Handle empty graph case
    if G.number_of_nodes() == 0:
        return True

    # Single node is trivially connected
    if G.number_of_nodes() == 1:
        return True

    return nx.is_connected(G)


def active_required_is_connected(graph: Graph, required_nodes: Set[Node]) -> bool:
    G = nx.Graph()

    # Add only active edges with positive visibility
    for edge in graph.edges:
        if edge.active and edge.visibility_score > 0:
            G.add_edge(edge.source_node, edge.dest_node)

    # Ensure all required nodes are in the graph
    for node in required_nodes:
        G.add_node(node)

    # Handle empty/trivial cases
    if len(required_nodes) == 0:
        return True

    if len(required_nodes) == 1:
        return True

    # Check if all required nodes are in the same connected component
    if G.number_of_nodes() == 0:
        return False

    # Get the connected component containing the first required node
    first_required = next(iter(required_nodes))
    if first_required not in G:
        return False

    connected_component = nx.node_connected_component(G, first_required)

    # All required nodes must be in this component
    for node in required_nodes:
        if node not in connected_component:
            return False

    return True


def get_active_nodes_from_edges(graph: Graph) -> Set[Node]:
    nodes = set()
    for edge in graph.edges:
        if edge.active and edge.visibility_score > 0:
            nodes.add(edge.source_node)
            nodes.add(edge.dest_node)
    return nodes


def count_connected_components_active(graph: Graph) -> int:
    G = nx.Graph()
    for edge in graph.edges:
        if edge.active and edge.visibility_score > 0:
            G.add_edge(edge.source_node, edge.dest_node)

    if G.number_of_nodes() == 0:
        return 0

    return nx.number_connected_components(G)


def count_connected_components_cumulative(graph: Graph) -> int:
    G = nx.Graph()
    for edge in graph.edges:
        G.add_edge(edge.source_node, edge.dest_node)

    if G.number_of_nodes() == 0:
        return 0

    return nx.number_connected_components(G)


# helper
def create_node(
    lemma: str,
    node_type: NodeType = NodeType.CONCEPT,
    node_source: NodeSource = NodeSource.TEXT_BASED,
    active: bool = True,
) -> Node:
    node = Node(
        lemmas=[lemma],
        actual_text=lemma,
        node_type=node_type,
        node_source=node_source,
        score=0,
        origin_sentence=1,
        provenance=NodeProvenance.STORY_TEXT,
    )
    node.active = active
    return node


def create_edge(
    source: Node,
    dest: Node,
    label: str = "relates_to",
    visibility_score: int = 3,
    active: bool = True,
) -> Edge:
    edge = Edge(
        source_node=source,
        dest_node=dest,
        label=label,
        visibility_score=visibility_score,
        active=active,
        created_at_sentence=1,
    )
    return edge


def add_edge_to_graph(graph: Graph, edge: Edge) -> None:
    # Ensure nodes are in graph
    graph.nodes.add(edge.source_node)
    graph.nodes.add(edge.dest_node)

    # Add edge
    graph.edges.add(edge)

    # Update node edge lists
    if edge not in edge.source_node.edges:
        edge.source_node.edges.append(edge)
    if edge not in edge.dest_node.edges:
        edge.dest_node.edges.append(edge)


class TestMinimalGraph:

    def test_minimal_graph_cumulative_connected(self):
        graph = Graph()

        # Create 2 nodes
        node_a = create_node("knight")
        node_b = create_node("horse")

        # Create 1 edge
        edge = create_edge(node_a, node_b, "rides", active=True)
        add_edge_to_graph(graph, edge)

        # Verify cumulative connectivity
        assert (
            cumulative_is_connected(graph) == True
        ), "Minimal graph (2 nodes, 1 edge) must be cumulatively connected"

    def test_minimal_graph_active_connected(self):
        """Minimal graph: active subgraph should connect required nodes."""
        graph = Graph()

        node_a = create_node("knight")
        node_b = create_node("horse")
        edge = create_edge(node_a, node_b, "rides", active=True)
        add_edge_to_graph(graph, edge)

        required = {node_a, node_b}
        assert (
            active_required_is_connected(graph, required) == True
        ), "Minimal graph with active edge must connect required nodes"

    def test_minimal_graph_inactive_edge(self):
        """Minimal graph with inactive edge: active check should fail."""
        graph = Graph()

        node_a = create_node("knight")
        node_b = create_node("horse")
        edge = create_edge(node_a, node_b, "rides", active=False)
        add_edge_to_graph(graph, edge)

        required = {node_a, node_b}

        # Cumulative should still be connected
        assert cumulative_is_connected(graph) == True

        # Active check should FAIL since edge is inactive
        assert (
            active_required_is_connected(graph, required) == False
        ), "Inactive edge should not connect required nodes in active check"


class TestMultiHopChain:
    # multi-hop chains
    def setup_method(self):
        self.graph = Graph()

        self.node_a = create_node("knight")
        self.node_b = create_node("castle")
        self.node_c = create_node("dragon")
        self.node_d = create_node("princess")

        self.edge_ab = create_edge(self.node_a, self.node_b, "enters")
        self.edge_bc = create_edge(self.node_b, self.node_c, "contains")
        self.edge_cd = create_edge(self.node_c, self.node_d, "guards")

        add_edge_to_graph(self.graph, self.edge_ab)
        add_edge_to_graph(self.graph, self.edge_bc)
        add_edge_to_graph(self.graph, self.edge_cd)

    def test_chain_cumulative_connected(self):
        assert cumulative_is_connected(self.graph) == True

    def test_chain_active_all_required(self):
        required = {self.node_a, self.node_b, self.node_c, self.node_d}
        assert active_required_is_connected(self.graph, required) == True

    def test_chain_endpoints_only(self):
        required = {self.node_a, self.node_d}
        assert active_required_is_connected(self.graph, required) == True

    def test_chain_middle_edge_removed(self):
        # Deactivate middle edge
        self.edge_bc.visibility_score = 0

        required = {self.node_a, self.node_d}

        # Cumulative should still be connected
        assert cumulative_is_connected(self.graph) == True

        # Active check should FAIL (chain is broken)
        assert active_required_is_connected(self.graph, required) == False


class TestDecayScenario:
    def setup_method(self):
        self.graph = Graph()

        # Cluster 1: knight, horse
        self.knight = create_node("knight")
        self.horse = create_node("horse")
        self.edge_kh = create_edge(self.knight, self.horse, "rides")

        # Cluster 2: dragon, treasure
        self.dragon = create_node("dragon")
        self.treasure = create_node("treasure")
        self.edge_dt = create_edge(self.dragon, self.treasure, "guards")

        # Bridge: knight -> dragon
        self.bridge = create_edge(self.knight, self.dragon, "fights")

        add_edge_to_graph(self.graph, self.edge_kh)
        add_edge_to_graph(self.graph, self.edge_dt)
        add_edge_to_graph(self.graph, self.bridge)

    def test_bridge_active_connected(self):
        required = {self.knight, self.horse, self.dragon, self.treasure}

        assert cumulative_is_connected(self.graph) == True
        assert active_required_is_connected(self.graph, required) == True

    def test_bridge_deactivated_cumulative_still_connected(self):
        self.bridge.visibility_score = 0

        assert (
            cumulative_is_connected(self.graph) == True
        ), "Cumulative graph should remain connected even with inactive edges"

    def test_bridge_deactivated_active_disconnected(self):
        self.bridge.visibility_score = 0

        # Full set should NOT be connected
        required = {self.knight, self.horse, self.dragon, self.treasure}
        assert active_required_is_connected(self.graph, required) == False

        # But each cluster should be connected internally
        cluster1 = {self.knight, self.horse}
        cluster2 = {self.dragon, self.treasure}

        assert active_required_is_connected(self.graph, cluster1) == True
        assert active_required_is_connected(self.graph, cluster2) == True

    def test_component_count_after_decay(self):
        # Initially 1 component
        assert count_connected_components_active(self.graph) == 1

        # After bridge decay, 2 components
        self.bridge.visibility_score = 0
        assert count_connected_components_active(self.graph) == 2

        # Cumulative should always be 1
        assert count_connected_components_cumulative(self.graph) == 1


class TestCarryoverOnlyScenario:

    def setup_method(self):
        self.graph = Graph()

        # Create nodes from sentence 1
        self.node_a = create_node("knight")
        self.node_b = create_node("castle")
        self.node_c = create_node("princess")

        # Set origin sentence to 1
        self.node_a.origin_sentence = 1
        self.node_b.origin_sentence = 1
        self.node_c.origin_sentence = 1

        # Mark explicit in sentence 1
        self.node_a.mark_explicit_in_sentence(1)
        self.node_b.mark_explicit_in_sentence(1)
        self.node_c.mark_explicit_in_sentence(1)

        # Create edges
        self.edge_ab = create_edge(self.node_a, self.node_b, "enters")
        self.edge_bc = create_edge(self.node_b, self.node_c, "contains")

        add_edge_to_graph(self.graph, self.edge_ab)
        add_edge_to_graph(self.graph, self.edge_bc)

    def test_carryover_nodes_connected_sentence_1(self):
        # All nodes are explicit in sentence 1
        assert self.node_a.is_explicit_in_sentence(1) == True
        assert self.node_b.is_explicit_in_sentence(1) == True
        assert self.node_c.is_explicit_in_sentence(1) == True

        required = {self.node_a, self.node_b, self.node_c}
        assert active_required_is_connected(self.graph, required) == True

    def test_carryover_nodes_in_sentence_2(self):
        # Check carryover status in sentence 2
        assert self.node_a.is_carryover_in_sentence(2) == True
        assert self.node_b.is_carryover_in_sentence(2) == True
        assert self.node_c.is_carryover_in_sentence(2) == True

        # Connectivity should still hold via active edges
        required = {self.node_a, self.node_b, self.node_c}
        assert active_required_is_connected(self.graph, required) == True


class TestLLMDisabledScenario:
    def test_deterministic_only_linear_chain(self):
        graph = Graph()

        # Create deterministic chain
        nodes = [create_node(f"node_{i}") for i in range(5)]

        for i in range(len(nodes) - 1):
            edge = create_edge(nodes[i], nodes[i + 1], f"relation_{i}")
            edge.inferred = False  # Not LLM-inferred
            add_edge_to_graph(graph, edge)

        # Should be fully connected
        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, set(nodes)) == True

    def test_deterministic_only_star_topology(self):
        graph = Graph()

        hub = create_node("hub")
        spokes = [create_node(f"spoke_{i}") for i in range(4)]

        for spoke in spokes:
            edge = create_edge(hub, spoke, "connects_to")
            edge.inferred = False
            add_edge_to_graph(graph, edge)

        all_nodes = {hub} | set(spokes)

        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, all_nodes) == True


class TestHeavyPruningScenario:
    def test_most_edges_pruned_but_spanning_tree_remains(self):
        graph = Graph()

        # Create 5 nodes with many edges
        nodes = [create_node(f"node_{i}") for i in range(5)]

        # Create full mesh (10 edges for 5 nodes)
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edge = create_edge(nodes[i], nodes[j], f"rel_{i}_{j}")
                add_edge_to_graph(graph, edge)
                edges.append(edge)

        # Verify fully connected
        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, set(nodes)) == True

        # Prune all but 4 edges (minimum spanning tree)
        # Keep: 0-1, 1-2, 2-3, 3-4
        for edge in edges:
            src_idx = int(edge.source_node.lemmas[0].split("_")[1])
            dst_idx = int(edge.dest_node.lemmas[0].split("_")[1])
            if abs(src_idx - dst_idx) != 1:
                edge.visibility_score = 0

        # Cumulative should still be connected
        assert cumulative_is_connected(graph) == True

        # Active should still be connected (spanning tree remains)
        assert active_required_is_connected(graph, set(nodes)) == True

    def test_pruning_creates_islands(self):
        graph = Graph()

        # Create two clusters
        cluster1 = [create_node(f"c1_node_{i}") for i in range(3)]
        cluster2 = [create_node(f"c2_node_{i}") for i in range(3)]

        # Connect within clusters
        for i in range(2):
            e1 = create_edge(cluster1[i], cluster1[i + 1], f"c1_rel_{i}")
            e2 = create_edge(cluster2[i], cluster2[i + 1], f"c2_rel_{i}")
            add_edge_to_graph(graph, e1)
            add_edge_to_graph(graph, e2)

        # Bridge between clusters
        bridge = create_edge(cluster1[0], cluster2[0], "bridge")
        add_edge_to_graph(graph, bridge)

        # Initially connected
        all_nodes = set(cluster1 + cluster2)
        assert active_required_is_connected(graph, all_nodes) == True

        # Prune bridge
        bridge.visibility_score = 0

        # Now should be disconnected
        assert active_required_is_connected(graph, all_nodes) == False

        # But each cluster should still be internally connected
        assert active_required_is_connected(graph, set(cluster1)) == True
        assert active_required_is_connected(graph, set(cluster2)) == True


class TestSentenceTransitionScenario:

    def test_new_explicit_connected_to_carryover(self):
        graph = Graph()

        # Sentence 1: Create knight and horse
        knight = create_node("knight")
        horse = create_node("horse")
        knight.mark_explicit_in_sentence(1)
        horse.mark_explicit_in_sentence(1)

        edge1 = create_edge(knight, horse, "rides")
        add_edge_to_graph(graph, edge1)

        # Sentence 2: knight is carryover, dragon is new explicit
        dragon = create_node("dragon")
        dragon.mark_explicit_in_sentence(2)

        edge2 = create_edge(knight, dragon, "fights")
        add_edge_to_graph(graph, edge2)

        # All three should be connected
        all_nodes = {knight, horse, dragon}
        assert active_required_is_connected(graph, all_nodes) == True

    def test_multiple_sentence_buildup(self):
        graph = Graph()

        # Sentence 1
        knight = create_node("knight")
        knight.mark_explicit_in_sentence(1)

        horse = create_node("horse")
        horse.mark_explicit_in_sentence(1)

        e1 = create_edge(knight, horse, "rides")
        add_edge_to_graph(graph, e1)

        # Verify after sentence 1
        assert active_required_is_connected(graph, {knight, horse}) == True

        # Sentence 2
        castle = create_node("castle")
        castle.mark_explicit_in_sentence(2)

        e2 = create_edge(knight, castle, "enters")
        add_edge_to_graph(graph, e2)

        # Verify after sentence 2
        assert active_required_is_connected(graph, {knight, horse, castle}) == True

        # Sentence 3
        dragon = create_node("dragon")
        dragon.mark_explicit_in_sentence(3)

        e3 = create_edge(castle, dragon, "contains")
        add_edge_to_graph(graph, e3)

        # Verify after sentence 3
        all_nodes = {knight, horse, castle, dragon}
        assert active_required_is_connected(graph, all_nodes) == True
        assert cumulative_is_connected(graph) == True


class TestLargeGraphScenario:
    # large chain
    def test_large_linear_chain(self):
        graph = Graph()

        nodes = [create_node(f"node_{i}") for i in range(50)]

        for i in range(len(nodes) - 1):
            edge = create_edge(nodes[i], nodes[i + 1], f"rel_{i}")
            add_edge_to_graph(graph, edge)

        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, set(nodes)) == True

        # Verify endpoints connected
        assert active_required_is_connected(graph, {nodes[0], nodes[-1]}) == True

    def test_large_star_with_secondary_hubs(self):
        graph = Graph()

        # Primary hub
        primary_hub = create_node("primary_hub")

        # Secondary hubs (5)
        secondary_hubs = [create_node(f"secondary_{i}") for i in range(5)]

        # Leaves for each secondary hub (10 each)
        all_leaves = []
        for i, sec_hub in enumerate(secondary_hubs):
            # Connect secondary to primary
            e1 = create_edge(primary_hub, sec_hub, f"connects_{i}")
            add_edge_to_graph(graph, e1)

            # Connect leaves to secondary
            for j in range(10):
                leaf = create_node(f"leaf_{i}_{j}")
                e2 = create_edge(sec_hub, leaf, f"has_{j}")
                add_edge_to_graph(graph, e2)
                all_leaves.append(leaf)

        # Total: 1 + 5 + 50 = 56 nodes
        all_nodes = {primary_hub} | set(secondary_hubs) | set(all_leaves)

        assert len(graph.nodes) == 56
        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, all_nodes) == True

    def test_large_graph_with_bottleneck(self):
        graph = Graph()

        # Left cluster (20 nodes, densely connected)
        left = [create_node(f"left_{i}") for i in range(20)]
        for i in range(len(left)):
            for j in range(i + 1, min(i + 3, len(left))):  # Connect to next 2
                e = create_edge(left[i], left[j], f"left_rel_{i}_{j}")
                add_edge_to_graph(graph, e)

        # Right cluster (20 nodes, densely connected)
        right = [create_node(f"right_{i}") for i in range(20)]
        for i in range(len(right)):
            for j in range(i + 1, min(i + 3, len(right))):
                e = create_edge(right[i], right[j], f"right_rel_{i}_{j}")
                add_edge_to_graph(graph, e)

        # Single bottleneck edge
        bottleneck = create_edge(left[0], right[0], "bottleneck")
        add_edge_to_graph(graph, bottleneck)

        all_nodes = set(left + right)

        # All connected
        assert active_required_is_connected(graph, all_nodes) == True

        # Deactivate bottleneck
        bottleneck.visibility_score = 0

        # Now disconnected
        assert active_required_is_connected(graph, all_nodes) == False

        # Cumulative still connected
        assert cumulative_is_connected(graph) == True


class TestEdgeCases:
    def test_empty_graph(self):
        graph = Graph()

        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, set()) == True

    def test_single_node_no_edges(self):
        graph = Graph()

        node = create_node("lonely")
        graph.nodes.add(node)

        # Cumulative is connected (single node)
        assert cumulative_is_connected(graph) == True

        # Single required node is trivially connected
        assert active_required_is_connected(graph, {node}) == True

    def test_self_loop(self):
        graph = Graph()

        node = create_node("self")

        # Graph.add_edge blocks self-loops, but test our utilities
        edge = Edge(
            source_node=node,
            dest_node=node,  # Self-loop
            label="self_ref",
            visibility_score=3,
            active=True,
        )

        graph.nodes.add(node)
        graph.edges.add(edge)
        node.edges.append(edge)

        # Single node is connected
        assert cumulative_is_connected(graph) == True
        assert active_required_is_connected(graph, {node}) == True

    def test_zero_visibility_edge(self):
        graph = Graph()

        node_a = create_node("a")
        node_b = create_node("b")

        edge = create_edge(node_a, node_b, "rel")
        edge.visibility_score = 0  # Faded edge

        add_edge_to_graph(graph, edge)

        # Cumulative should be connected (edge exists)
        assert cumulative_is_connected(graph) == True

        # Active should NOT connect (visibility = 0)
        assert active_required_is_connected(graph, {node_a, node_b}) == False

    def test_required_node_not_in_graph(self):
        graph = Graph()

        in_graph = create_node("in_graph")
        not_in_graph = create_node("not_in_graph")

        graph.nodes.add(in_graph)

        # Required node not in graph should still pass (added to nx graph)
        # But not connected since no edges
        assert active_required_is_connected(graph, {in_graph, not_in_graph}) == False


def assert_connectivity_invariants(
    graph: Graph,
    required_nodes: Set[Node],
    context: str = "",
) -> None:
    ctx = f" ({context})" if context else ""

    cum_connected = cumulative_is_connected(graph)
    assert cum_connected, (
        f"INVARIANT VIOLATION{ctx}: cumulative graph is disconnected "
        f"(components: {count_connected_components_cumulative(graph)})"
    )

    active_connected = active_required_is_connected(graph, required_nodes)
    assert active_connected, (
        f"INVARIANT VIOLATION{ctx}: required nodes not connected via active edges "
        f"(active components: {count_connected_components_active(graph)}, "
        f"required nodes: {len(required_nodes)})"
    )


class TestInvariantUtility:
    def test_passing_invariants(self):
        graph = Graph()

        a = create_node("a")
        b = create_node("b")
        e = create_edge(a, b, "rel")
        add_edge_to_graph(graph, e)

        # Should not raise
        assert_connectivity_invariants(graph, {a, b}, "test")

    def test_failing_cumulative(self):
        graph = Graph()

        # Create two disconnected edge components
        a = create_node("a")
        b = create_node("b")
        c = create_node("c")
        d = create_node("d")

        # Component 1: a-b
        e1 = create_edge(a, b, "rel1")
        add_edge_to_graph(graph, e1)

        # Component 2: c-d (disconnected from a-b)
        e2 = create_edge(c, d, "rel2")
        add_edge_to_graph(graph, e2)

        # Cumulative graph has 2 components
        with pytest.raises(AssertionError, match="cumulative graph is disconnected"):
            assert_connectivity_invariants(graph, {a, b, c, d}, "test")

    def test_failing_active(self):
        graph = Graph()

        a = create_node("a")
        b = create_node("b")
        e = create_edge(a, b, "rel", active=False)
        add_edge_to_graph(graph, e)

        # Cumulative passes, active fails
        with pytest.raises(
            AssertionError, match="required nodes not connected via active edges"
        ):
            assert_connectivity_invariants(graph, {a, b}, "test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
