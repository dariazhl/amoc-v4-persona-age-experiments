from typing import List, Tuple, Dict


def compute_graph_metrics(edges: List[Tuple[str, str]]) -> Dict[str, float]:
    nodes = set()
    adj = {}

    valid_edges = []

    for u, v in edges:
        if not u or not v:
            continue
        u, v = str(u), str(v)
        nodes.add(u)
        nodes.add(v)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
        valid_edges.append((u, v))

    n = len(nodes)
    m = len(valid_edges)

    if n == 0:
        return {
            "graph_num_nodes": 0,
            "graph_num_edges": 0,
            "graph_avg_degree": 0.0,
            "graph_density": 0.0,
            "graph_num_components": 0,
            "graph_largest_component_size": 0,
            "graph_largest_component_ratio": 0.0,
        }

    degrees = [len(adj[node]) for node in nodes]
    graph_avg_degree = sum(degrees) / n

    graph_density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    visited = set()
    graph_num_components = 0
    largest_component_size = 0

    for node in nodes:
        if node in visited:
            continue
        graph_num_components += 1
        stack = [node]
        comp_size = 0
        while stack:
            x = stack.pop()
            if x in visited:
                continue
            visited.add(x)
            comp_size += 1
            stack.extend(adj.get(x, []))
        largest_component_size = max(largest_component_size, comp_size)

    return {
        "graph_num_nodes": n,
        "graph_num_edges": m,
        "graph_avg_degree": graph_avg_degree,
        "graph_density": graph_density,
        "graph_num_components": graph_num_components,
        "graph_largest_component_size": largest_component_size,
        "graph_largest_component_ratio": largest_component_size / n,
    }
