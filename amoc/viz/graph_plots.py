import os
import re
import math
from typing import Any, List, Tuple, Iterable, Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
from amoc.config.paths import OUTPUT_ANALYSIS_DIR
from collections import defaultdict
import textwrap
from matplotlib.patches import FancyBboxPatch
from collections import deque
from collections import defaultdict
from matplotlib.gridspec import GridSpec


DEFAULT_BLUE_NODES: Iterable[str] = ()

# Sentinel key stored inside the positions dict to remember the original hub
_HUB_CENTER_KEY = "__hub_center__"


def pretty_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def compute_bfs_levels(G, hub):

    levels = {hub: 0}
    queue = deque([hub])

    while queue:
        current = queue.popleft()
        for neighbor in G.neighbors(current):
            if neighbor not in levels:
                levels[neighbor] = levels[current] + 1
                queue.append(neighbor)

    return levels


# Radial layout constants (shared between initial layout and incremental placement)
_RADIAL_BASE_RADIUS = 4.0
_RADIAL_RING_GAP = 4.5
_RADIAL_NODE_DIAMETER = 1.8
_RADIAL_MIN_DISTANCE = 3.0


# Golden angle (~137.5°) for optimal angular separation across rings
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def compute_radial_positions(G, hub):
    # BFS COMPUTATION
    levels = compute_bfs_levels(G, hub)

    # GROUP NODES BY LEVEL
    rings = {}
    for node, level in levels.items():
        rings.setdefault(level, []).append(node)

    pos = {}

    # Hub at center
    if hub in levels:
        pos[hub] = (0.0, 0.0)

    # Per-ring angular offset: each ring starts at a different angle
    # using the golden angle so nodes on adjacent rings never align
    # through the center.
    ring_offset_base = 0.0

    # PLACE EACH RING
    for level in sorted(rings.keys()):
        nodes = rings[level]
        if level == 0:
            # Hub already placed at (0,0)
            continue

        n = len(nodes)
        ring_offset = ring_offset_base + level * _GOLDEN_ANGLE

        # ensure no overlap using circumference math
        circumference_needed = n * _RADIAL_NODE_DIAMETER * 1.5
        min_radius = circumference_needed / (2 * math.pi)
        radius = max(_RADIAL_BASE_RADIUS + level * _RADIAL_RING_GAP, min_radius)

        if n == 1:
            pos[nodes[0]] = (
                radius * math.cos(ring_offset),
                radius * math.sin(ring_offset),
            )
            continue

        angle_step = 2 * math.pi / n

        for i, node in enumerate(sorted(nodes)):
            angle = ring_offset + i * angle_step
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # POST-PLACEMENT: break collinear triplets (three nodes roughly on a line
    # through the center).  Nudge the middle node perpendicular to the line.
    break_collinear_nodes(pos)

    return pos


def break_collinear_nodes(
    pos: Dict[str, Tuple[float, float]],
    threshold: float = 0.97,
    nudge_fraction: float = 0.15,
) -> None:
    node_list = [n for n in pos if pos[n] != (0.0, 0.0)]
    already_nudged: set = set()

    for i, a in enumerate(node_list):
        ax, ay = pos[a]
        a_len = math.hypot(ax, ay)
        if a_len < 1e-6:
            continue
        uax, uay = ax / a_len, ay / a_len

        for b in node_list[i + 1 :]:
            bx, by = pos[b]
            b_len = math.hypot(bx, by)
            if b_len < 1e-6:
                continue
            ubx, uby = bx / b_len, by / b_len

            dot = uax * ubx + uay * uby
            if abs(dot) < threshold:
                continue  # not collinear

            # Pick the node farther from center to nudge
            target = b if b_len >= a_len else a
            if target in already_nudged:
                continue

            tx, ty = pos[target]
            t_len = math.hypot(tx, ty)
            # Perpendicular direction (rotate 90°)
            px, py = -ty / t_len, tx / t_len
            offset = t_len * nudge_fraction
            pos[target] = (tx + px * offset, ty + py * offset)
            already_nudged.add(target)


def compute_edge_curvatures(
    edges_with_keys: List[Tuple[str, str, str]],
) -> Tuple[Dict[Tuple[str, str, str], float], Dict[Tuple[str, str, str], float]]:
    # Group edges by their node pair (undirected for overlap detection)
    pair_edges: Dict[tuple, List[Tuple[str, str, str]]] = defaultdict(list)
    for u, v, k in edges_with_keys:
        # Use sorted tuple for undirected grouping (to catch A->B and B->A)
        pair_key = tuple(sorted([u, v]))
        pair_edges[pair_key].append((u, v, k))

    curvatures: Dict[Tuple[str, str, str], float] = {}
    label_t_params: Dict[Tuple[str, str, str], float] = {}

    for pair_key, edges in pair_edges.items():
        n_edges = len(edges)
        if n_edges == 1:
            # Single edge - no curvature needed, label at midpoint
            curvatures[edges[0]] = 0.0
            label_t_params[edges[0]] = 0.5
        else:
            # Multiple edges between same node pair - assign varying curvatures
            # Spread curvatures symmetrically around 0
            # E.g., for 2 edges: [-0.15, 0.15], for 3: [-0.2, 0, 0.2]
            base_rad = 0.15  # Base curvature radius
            max_rad = 0.35  # Maximum curvature to prevent extreme curves

            # MULTI-EDGE LABEL PLACEMENT: Stagger t-parameters
            # For 2 edges: [0.4, 0.6], for 3: [0.35, 0.5, 0.65], etc.
            t_spread = 0.15  # How far from 0.5 to spread labels
            t_min = max(0.3, 0.5 - t_spread * (n_edges - 1) / 2)
            t_max = min(0.7, 0.5 + t_spread * (n_edges - 1) / 2)

            for idx, (u, v, k) in enumerate(edges):
                if n_edges == 2:
                    # Two edges: one curves up, one curves down
                    rad = base_rad if idx == 0 else -base_rad
                    t_val = 0.4 if idx == 0 else 0.6
                else:
                    # More edges: spread evenly
                    spread = min(max_rad, base_rad * (n_edges - 1) / 2)
                    rad = -spread + (2 * spread * idx / (n_edges - 1))
                    t_val = t_min + (t_max - t_min) * idx / (n_edges - 1)

                # For reciprocal edges (A->B vs B->A), ensure they curve opposite
                # by checking if this is the "reverse" direction
                if (u, v) != pair_key:
                    rad = -rad  # Flip for reverse direction

                curvatures[(u, v, k)] = rad
                label_t_params[(u, v, k)] = t_val

    return curvatures, label_t_params


def compute_label_position_curved(
    pos: Dict[str, Tuple[float, float]],
    u: str,
    v: str,
    curvature: float,
    t: float = 0.5,
) -> Tuple[float, float]:
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # Edge vector and perpendicular
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)

    if length < 1e-6:
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2

    # Perpendicular unit vector (rotated 90 degrees counterclockwise)
    px, py = -dy / length, dx / length

    # Control point offset: matplotlib arc3 uses offset = rad * length
    # where rad is the curvature parameter
    offset = curvature * length
    cx, cy = mx + px * offset, my + py * offset

    # Quadratic Bezier: B(t) = (1-t)²P0 + 2(1-t)tP1 + t²P2
    # where P0=(x1,y1), P1=(cx,cy), P2=(x2,y2)
    b_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
    b_y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2

    return b_x, b_y


def compute_label_angle_along_edge(
    pos: Dict[str, Tuple[float, float]],
    u: str,
    v: str,
    curvature: float,
    t: float = 0.5,
) -> float:
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # Edge vector
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)

    if length < 1e-6:
        return 0.0  # No rotation for zero-length edges

    # For quadratic Bezier, tangent at t is: B'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
    # We need to compute the control point matching matplotlib's arc3 style
    if abs(curvature) < 1e-6:
        # No curvature: tangent is chord direction
        tangent_x, tangent_y = dx, dy
    else:
        # Compute control point (matching matplotlib arc3: offset = rad * length)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        px, py = -dy / length, dx / length
        offset = curvature * length
        cx, cy = mx + px * offset, my + py * offset

        # Bezier derivative: B'(t) = 2[(1-t)(P1-P0) + t(P2-P1)]
        tangent_x = 2 * ((1 - t) * (cx - x1) + t * (x2 - cx))
        tangent_y = 2 * ((1 - t) * (cy - y1) + t * (y2 - cy))

    # Compute angle in degrees
    angle_rad = math.atan2(tangent_y, tangent_x)
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to keep text readable (not upside-down)
    # Text is readable when angle is in range [-90, 90]
    # If outside this range, flip by 180 degrees
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg


# box with active triplets
def draw_triplet_panel(
    ax,
    triplets: List[Tuple[str, str, str]],
    active_nodes: Optional[set] = None,
    max_triplets: int = 43,
    font_size: int = 9,
) -> None:
    ax.axis("off")

    # Filter triplets to only those involving active nodes
    if active_nodes is not None:
        filtered_triplets = [
            (s, r, o) for s, r, o in triplets if s in active_nodes and o in active_nodes
        ]
    else:
        filtered_triplets = triplets

    # Panel title
    ax.text(
        0.5,
        0.98,
        "Active Triplets",
        transform=ax.transAxes,
        fontsize=font_size + 3,
        fontweight="bold",
        ha="center",
        va="top",
        color="#333333",
    )

    if not filtered_triplets:
        ax.text(
            0.5,
            0.5,
            "No active triplets",
            transform=ax.transAxes,
            fontsize=font_size + 1,
            ha="center",
            va="center",
            color="#888888",
            style="italic",
        )
        return

    # Separator line below title
    ax.plot(
        [0.05, 0.95],
        [0.955, 0.955],
        transform=ax.transAxes,
        color="#cccccc",
        linewidth=1,
        clip_on=False,
    )

    display_triplets = filtered_triplets[:max_triplets]
    total = len(filtered_triplets)
    truncated = total > max_triplets

    # Compute vertical spacing
    y_start = 0.94
    line_height = min(0.020, 0.9 / max(len(display_triplets), 1))

    for idx, (s, r, o) in enumerate(display_triplets):
        y = y_start - idx * line_height

        s_clean = pretty_text(s)
        r_clean = pretty_text(r)
        o_clean = pretty_text(o)

        # Alternating row background
        if idx % 2 == 0:

            bg = FancyBboxPatch(
                (0.02, y - line_height * 0.55),
                0.96,
                line_height * 0.95,
                transform=ax.transAxes,
                facecolor="#f0f4f8",
                edgecolor="none",
                boxstyle="round,pad=0.002",
                clip_on=False,
            )
            ax.add_patch(bg)

        # Number
        ax.text(
            0.04,
            y,
            f"{idx + 1}.",
            transform=ax.transAxes,
            fontsize=font_size - 1,
            fontfamily="monospace",
            ha="right",
            va="center",
            color="#999999",
        )
        # Triplet text
        ax.text(
            0.07,
            y,
            f"({s_clean}, {r_clean}, {o_clean})",
            transform=ax.transAxes,
            fontsize=font_size,
            fontfamily="monospace",
            ha="left",
            va="center",
            color="#333333",
        )

    # Footer
    footer_y = y_start - len(display_triplets) * line_height - 0.02
    ax.plot(
        [0.05, 0.95],
        [footer_y + 0.01, footer_y + 0.01],
        transform=ax.transAxes,
        color="#cccccc",
        linewidth=1,
        clip_on=False,
    )
    footer_text = f"Total: {total} active triplet{'s' if total != 1 else ''}"
    if truncated:
        footer_text += f"  (showing {max_triplets})"
    ax.text(
        0.5,
        footer_y - 0.005,
        footer_text,
        transform=ax.transAxes,
        fontsize=font_size - 1,
        ha="center",
        va="top",
        color="#666666",
    )


def plot_amoc_triplets(
    triplets: List[Tuple[str, str, str]],
    persona: str,
    model_name: str,
    age: int,
    blue_nodes: Optional[Iterable[str]] = None,
    output_dir: str | None = None,
    step_tag: Optional[str] = None,
    largest_component_only: bool = False,
    sentence_text: str = "",
    deactivated_concepts: Optional[List[str]] = None,
    new_nodes: Optional[List[str]] = None,
    explicit_nodes: Optional[Iterable[str]] = None,
    ever_explicit_nodes: Optional[Iterable[str]] = None,
    inferred_nodes: Optional[
        Iterable[str]
    ] = None,  # Paper-aligned: Yellow = LLM-inferred
    salient_nodes: Optional[Iterable[str]] = None,
    inactive_nodes: Optional[Iterable[str]] = None,
    inactive_nodes_for_title: Optional[Iterable[str]] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    avoid_edge_overlap: bool = True,
    active_edges: Optional[set[Tuple[str, str, str]]] = None,
    hub_edge_explanations: Optional[List[str]] = None,
    show_all_edges: bool = False,
    edge_activation_scores: Optional[Dict[Tuple[str, str, str], int]] = None,
    layout_from_active_only: bool = True,
    active_triplets_for_overlay: Optional[List[Tuple[str, str, str]]] = None,
    show_triplet_overlay: bool = True,  # TASK 2: Control overlay visibility
    layout_depth: int = 3,
    graph: Optional[Any] = None,
) -> str:

    def expand_by_anchor(
        G_full: nx.Graph,
        G_snapshot: nx.Graph,
        anchors: set[str],
        hops: int = 2,
    ) -> nx.Graph:
        # Only expand within connected components that contain anchors
        # to prevent pulling in disconnected parts of the graph
        nodes_to_keep = set(G_snapshot.nodes())

        # Find all connected components in G_full
        if G_full.number_of_nodes() == 0:
            return G_full.subgraph(nodes_to_keep).copy()

        # NOTE: Main connectivity check is done after G is built (lines ~549-560)
        # This helper only operates on G_full which is always the full graph

        # Get the connected component(s) containing at least one anchor
        anchor_components: set[frozenset[str]] = set()
        for component in nx.connected_components(G_full):
            component_set = frozenset(component)
            if any(anchor in component_set for anchor in anchors):
                anchor_components.add(component_set)

        # Only consider nodes from components that contain anchors
        valid_nodes = set()
        for comp in anchor_components:
            valid_nodes |= set(comp)

        # Now expand from anchors, but only within valid components
        for anchor in anchors:
            if anchor not in G_full:
                continue
            if anchor not in valid_nodes:
                continue

            reachable = nx.single_source_shortest_path_length(
                G_full, anchor, cutoff=hops
            )
            # Only keep reachable nodes that are in valid components
            nodes_to_keep |= set(reachable.keys()) & valid_nodes

        return G_full.subgraph(nodes_to_keep).copy()

    blue_nodes = set(blue_nodes) if blue_nodes is not None else set(DEFAULT_BLUE_NODES)
    out_dir = output_dir or os.path.join(OUTPUT_ANALYSIS_DIR, "graphs")
    os.makedirs(out_dir, exist_ok=True)

    def _sanitize(component: str, max_len: int = 50) -> str:
        component = (component or "").replace("\n", " ").strip()
        component = component[:max_len]
        # Replace path separators and common bad filename chars
        return re.sub(r"[\\/:*?\"<>|]", "_", component)

    safe_model = _sanitize(model_name, max_len=60)
    safe_persona = _sanitize(persona, max_len=40)
    suffix = f"_{step_tag}" if step_tag else ""
    filename = f"amoc_graph_{safe_model}_{safe_persona}_{age}{suffix}.png"
    save_path = os.path.join(out_dir, filename)

    G = nx.MultiDiGraph()
    G_active = nx.MultiDiGraph()  # Active subgraph for layout computation

    edge_labels: Dict[Tuple[str, str, str], str] = {}
    edge_status: Dict[Tuple[str, str, str], str] = {}
    edge_scores: Dict[Tuple[str, str, str], int] = edge_activation_scores or {}
    # Build a lookup set of active triplets for the smart dedup and overlay.
    # Use active_triplets_for_overlay (authoritative source with correct labels
    # and directions) when available; fall back to active_edges for compat.
    _active_triplet_set: set = set()
    if active_triplets_for_overlay:
        _active_triplet_set = {
            (str(s).strip(), str(r).strip(), str(o).strip())
            for s, r, o in active_triplets_for_overlay
            if s and o
        }

    active_edge_set = active_edges or set()

    # Build sets for node categorization (needed for layout decision)
    inactive_node_set = set(inactive_nodes) if inactive_nodes else set()

    # dedup: keep at most one edge per undirected node pair,
    _best_edge_per_pair: Dict[Tuple[str, str], Tuple[str, str, str, bool]] = {}

    for src, rel, dst in triplets:
        src = str(src).strip()
        dst = str(dst).strip()
        rel = str(rel).strip()

        if not src or not dst or not rel:
            continue

        is_structural = rel.startswith("structural::")
        clean_rel = rel.replace("structural::", "").strip()

        # Check if this triplet is active — first via the authoritative
        # active_triplet_set, then via the legacy active_edge_set.
        is_active = (src, rel, dst) in _active_triplet_set
        if not is_active and not _active_triplet_set:
            is_active = (src, dst) in active_edge_set or (
                src,
                dst,
                clean_rel,
            ) in active_edge_set

        pair_key = tuple(sorted((src, dst)))

        if pair_key not in _best_edge_per_pair:
            _best_edge_per_pair[pair_key] = (src, clean_rel, dst, is_active)
        else:
            _, _, _, current_is_active = _best_edge_per_pair[pair_key]
            # Replace if the stored edge is inactive and this one is active
            if not current_is_active and is_active:
                _best_edge_per_pair[pair_key] = (src, clean_rel, dst, is_active)

    # Second pass: build G from the winning edges
    for src, clean_rel, dst, is_active in _best_edge_per_pair.values():
        edge_key = f"{clean_rel}"

        G.add_edge(src, dst, key=edge_key)
        edge_labels[(src, dst, edge_key)] = clean_rel
        edge_status[(src, dst, edge_key)] = "normal"

        # Add to active subgraph for layout computation
        involves_inactive = src in inactive_node_set or dst in inactive_node_set
        if layout_from_active_only and not involves_inactive and is_active:
            G_active.add_edge(src, dst, key=edge_key)

    # Ensure explicit nodes always exist in graph (even without edges)
    if explicit_nodes:
        for node in explicit_nodes:
            if node not in G:
                G.add_node(node)

    # Add inactive working-memory nodes to the graph so they render as gray nodes
    # (the caller now ensures these are only nodes that were previously in working memory)
    if inactive_nodes:
        for node in inactive_nodes:
            if node not in G:
                G.add_node(node)

        if graph is not None:
            inactive_node_names = set(inactive_nodes)
            known_nodes = set(G.nodes())

            for edge in graph.edges:
                source_name = edge.source_node.get_text_representer()
                dest_name = edge.dest_node.get_text_representer()

                if (
                    source_name in inactive_node_names
                    or dest_name in inactive_node_names
                ):
                    # only add edges between nodes already known to the plot
                    # (from triplets, explicit_nodes, or inactive_nodes)
                    # to avoid pulling in nodes that were never in working memory
                    if source_name not in known_nodes or dest_name not in known_nodes:
                        continue

                    G.add_edge(source_name, dest_name, key=edge.label)

    # If after injection the graph is still empty, return
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots(figsize=(34, 26))
        ax.axis("off")
        fig.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plotted_nodes = set(G.nodes())

    fig = plt.figure(figsize=(40, 26))
    gs = GridSpec(1, 2, width_ratios=[0.85, 0.15], figure=fig, wspace=0.02)
    ax = fig.add_subplot(gs[0])
    ax_triplets = fig.add_subplot(gs[1])

    # Build sets for node categorization
    inactive_node_set = set(inactive_nodes) if inactive_nodes else set()
    ever_explicit_node_set = set(ever_explicit_nodes) if ever_explicit_nodes else set()
    salient_node_set = set(salient_nodes) if salient_nodes else set()
    inferred_node_set = set(inferred_nodes) if inferred_nodes else set()

    # Frozen positions: reuse existing positions, only compute for new nodes
    frozen = {n: positions[n] for n in G.nodes() if positions and n in positions}
    new_nodes = [n for n in G.nodes() if n not in frozen]

    if not new_nodes:
        pos = dict(frozen)
    elif not frozen:
        # First call — compute full radial layout
        hub = max(G.degree, key=lambda x: x[1])[0]
        pos = compute_radial_positions(G, hub)
        if not pos:
            pos = nx.spring_layout(G, seed=42)
        # Handle any nodes _compute_radial_positions missed (disconnected)
        new_nodes = [n for n in G.nodes() if n not in pos]
        if new_nodes:
            count = len(new_nodes)
            radius = max(10.0, count * 1.5)
            for idx, node in enumerate(sorted(new_nodes)):
                angle = idx * _GOLDEN_ANGLE  # golden angle avoids collinearity
                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        # Cache the hub center for future calls
        if positions is not None and hub in pos:
            positions[_HUB_CENTER_KEY] = pos[hub]

            levels = compute_bfs_levels(G, hub)
            for node, level in levels.items():
                positions[f"{node}_level"] = level
    else:
        # Have frozen positions + new nodes — place on proper radial rings
        pos = dict(frozen)

        # Get hub center
        if positions is not None and _HUB_CENTER_KEY in positions:
            center_x, center_y = positions[_HUB_CENTER_KEY]
        else:
            center_x = sum(x for x, y in pos.values()) / len(pos)
            center_y = sum(y for x, y in pos.values()) / len(pos)

        # Find the hub node (closest to center)
        hub_node = None
        best_hub_dist = float("inf")
        for node, (x, y) in pos.items():
            d = math.hypot(x - center_x, y - center_y)
            if d < best_hub_dist:
                best_hub_dist = d
                hub_node = node

        if hub_node is None:
            hub_node = max(G.degree, key=lambda x: x[1])[0]

        # Compute BFS levels for ALL nodes (including new ones)
        levels = compute_bfs_levels(G, hub_node)

        # Group new nodes by their BFS level
        new_nodes_by_level = {}
        for node in new_nodes:
            level = levels.get(node, 1)  # Default to ring 1 if disconnected
            new_nodes_by_level.setdefault(level, []).append(node)

        # Place nodes level by level on their radial ring
        for level, level_nodes in new_nodes_by_level.items():
            if not level_nodes:
                continue

            # Base radius for this level
            radius = _RADIAL_BASE_RADIUS + level * _RADIAL_RING_GAP

            # Count existing nodes at approximately this radius
            existing_at_radius = []
            for node, (x, y) in pos.items():
                dist = math.hypot(x - center_x, y - center_y)
                if abs(dist - radius) < _RADIAL_RING_GAP / 2:
                    existing_at_radius.append(math.atan2(y - center_y, x - center_x))

            total_nodes_at_level = len(existing_at_radius) + len(level_nodes)

            # Expand ring radius if overcrowded
            circumference_needed = total_nodes_at_level * _RADIAL_NODE_DIAMETER * 1.5
            min_radius = circumference_needed / (2 * math.pi)
            radius = max(radius, min_radius)

            # Place each new node on the ring
            for i, node in enumerate(sorted(level_nodes)):
                target_angle = (2 * math.pi * i) / max(1, total_nodes_at_level)

                # Try angles to avoid collisions with existing nodes
                best_angle = target_angle
                best_min_dist = 0

                for attempt in range(36):
                    test_angle = target_angle + (attempt * math.pi / 18)
                    test_x = center_x + radius * math.cos(test_angle)
                    test_y = center_y + radius * math.sin(test_angle)

                    min_dist = min(
                        (
                            math.hypot(test_x - ex, test_y - ey)
                            for ex, ey in pos.values()
                        ),
                        default=float("inf"),
                    )

                    if min_dist >= _RADIAL_MIN_DISTANCE:
                        best_angle = test_angle
                        break
                    elif min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_angle = test_angle

                pos[node] = (
                    center_x + radius * math.cos(best_angle),
                    center_y + radius * math.sin(best_angle),
                )

    inactive_in_graph = [n for n in G.nodes() if n in inactive_node_set]
    active_in_graph = [n for n in G.nodes() if n not in inactive_node_set]

    # Draw inactive nodes with faded colors and reduced opacity
    if inactive_in_graph:
        inactive_colors = ["#d0d0d0" for _ in inactive_in_graph]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=inactive_in_graph,
            node_size=3800,
            node_color=inactive_colors,
            linewidths=1.5,
            edgecolors="#999999",
            alpha=0.4,
            ax=ax,
        )

    # Draw active nodes on top with full opacity
    if active_in_graph:
        # Paper-aligned: Yellow = LLM-inferred (INFERENCE_BASED), Blue = text-derived
        active_colors = [
            "#ffe8a0" if node in inferred_node_set else "#a0cbe2"
            for node in active_in_graph
        ]

        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=active_in_graph,
            node_size=3800,
            node_color=active_colors,
            linewidths=2.0,
            edgecolors="black",
            ax=ax,
        )

    edge_pairs = {(u, v) for u, v, _ in G.edges(keys=True)}
    reciprocals = {(u, v) for (u, v) in edge_pairs if (v, u) in edge_pairs}

    all_edges_with_keys = list(G.edges(keys=True))
    edge_curvatures, edge_label_t_params = compute_edge_curvatures(all_edges_with_keys)

    normal_edges = []
    implicit_edges = []
    structural_edges = []

    normal_edge_colors = []
    normal_edge_widths = []

    implicit_edge_colors = []
    implicit_edge_widths = []

    structural_edge_colors = []
    structural_edge_widths = []

    # Track edges involving inactive nodes for faded rendering
    inactive_edges = []
    inactive_edge_colors = []
    inactive_edge_widths = []

    # Uniform edge thickness constants
    _EDGE_WIDTH = 2.0  # all active edges
    _EDGE_WIDTH_INACTIVE = 1.2  # edges touching inactive nodes
    _EDGE_WIDTH_STRUCTURAL = 2.0  # structural (dashed) edges

    def check_edge_active(u, v, k):
        if _active_triplet_set:
            return (u, k, v) in _active_triplet_set
        return (u, v, k) in active_edge_set or (u, v) in active_edge_set

    for u, v, k in G.edges(keys=True):
        status = edge_status.get((u, v, k), "normal")
        is_active = check_edge_active(u, v, k)
        involves_inactive = u in inactive_node_set or v in inactive_node_set

        if status == "structural":
            structural_edges.append((u, v, k))
            structural_edge_colors.append(
                "green" if not involves_inactive else "#90c090"
            )
            structural_edge_widths.append(_EDGE_WIDTH_STRUCTURAL)

        elif status == "implicit":
            implicit_edges.append((u, v, k))
            implicit_edge_colors.append("#999999" if not is_active else "#666699")
            implicit_edge_widths.append(_EDGE_WIDTH)

        elif involves_inactive:
            inactive_edges.append((u, v, k))
            inactive_edge_colors.append("#cccccc")
            inactive_edge_widths.append(_EDGE_WIDTH_INACTIVE)

        else:
            normal_edges.append((u, v, k))
            if is_active:
                normal_edge_colors.append("black")
            else:
                normal_edge_colors.append("#cccccc")
            normal_edge_widths.append(_EDGE_WIDTH)

    # Uniform alpha: active = fully opaque, inactive = faded
    normal_edge_alphas = []
    for u, v, k in normal_edges:
        is_active = check_edge_active(u, v, k)
        normal_edge_alphas.append(1.0 if is_active else 0.4)

    # Edges are drawn with arrows pointing from u to v
    if normal_edges:
        # Group edges by (alpha, curvature) for efficient drawing
        # Each unique (alpha, curvature) combination gets its own draw call
        edge_draw_groups = defaultdict(list)
        for idx, (u, v, k) in enumerate(normal_edges):
            alpha = round(normal_edge_alphas[idx], 1)
            curvature = round(edge_curvatures.get((u, v, k), 0.0), 2)
            edge_draw_groups[(alpha, curvature)].append((idx, u, v, k))

        for (alpha_val, curvature), edge_group in edge_draw_groups.items():
            # INVARIANT: (u, v) preserves triplet direction - arrow points u → v
            group_edges = [(u, v) for idx, u, v, k in edge_group]
            group_colors = [normal_edge_colors[idx] for idx, u, v, k in edge_group]
            group_widths = [normal_edge_widths[idx] for idx, u, v, k in edge_group]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=group_edges,
                edge_color=group_colors,
                arrows=True,
                arrowsize=35,  # Large arrowheads for visibility
                arrowstyle="-|>",  # Filled arrow style
                node_size=3800,  # Match node size so arrows stop at node boundary
                width=group_widths,
                alpha=alpha_val,
                connectionstyle="arc3,rad=0.0",
                min_target_margin=15,  # Extra margin so arrowhead is fully visible
                ax=ax,
            )

    # Draw implicit edges as dashed lines with per-edge curvature
    if implicit_edges:
        implicit_curvature_groups = defaultdict(list)
        for idx, (u, v, k) in enumerate(implicit_edges):
            curvature = round(edge_curvatures.get((u, v, k), 0.0), 2)
            implicit_curvature_groups[curvature].append((idx, u, v, k))

        for curvature, edge_group in implicit_curvature_groups.items():
            group_edges = [(u, v) for idx, u, v, k in edge_group]
            group_colors = [implicit_edge_colors[idx] for idx, u, v, k in edge_group]
            group_widths = [implicit_edge_widths[idx] for idx, u, v, k in edge_group]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=group_edges,
                edge_color=group_colors,
                arrows=True,
                arrowsize=30,  # Large arrowheads for visibility
                arrowstyle="-|>",
                node_size=3800,  # Match node size so arrows stop at node boundary
                width=group_widths,
                style="dashed",
                connectionstyle="arc3,rad=0.0",
                min_target_margin=15,
                ax=ax,
            )

    # Draw structural edges with per-edge curvature
    if structural_edges:
        structural_curvature_groups = defaultdict(list)
        for idx, (u, v, k) in enumerate(structural_edges):
            curvature = round(edge_curvatures.get((u, v, k), 0.0), 2)
            structural_curvature_groups[curvature].append((idx, u, v, k))

        for curvature, edge_group in structural_curvature_groups.items():
            group_edges = [(u, v) for idx, u, v, k in edge_group]
            group_colors = [structural_edge_colors[idx] for idx, u, v, k in edge_group]
            group_widths = [structural_edge_widths[idx] for idx, u, v, k in edge_group]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=group_edges,
                edge_color=group_colors,
                width=group_widths,
                style="dashed",
                arrows=True,
                arrowsize=35,  # Large arrowheads for visibility
                arrowstyle="-|>",
                node_size=3800,  # Match node size so arrows stop at node boundary
                connectionstyle="arc3,rad=0.0",
                min_target_margin=15,
                ax=ax,
            )

    # Draw edges involving inactive nodes with per-edge curvature
    if inactive_edges:
        inactive_curvature_groups = defaultdict(list)
        for idx, (u, v, k) in enumerate(inactive_edges):
            curvature = round(edge_curvatures.get((u, v, k), 0.0), 2)
            inactive_curvature_groups[curvature].append((idx, u, v, k))

        for curvature, edge_group in inactive_curvature_groups.items():
            group_edges = [(u, v) for idx, u, v, k in edge_group]
            group_colors = [inactive_edge_colors[idx] for idx, u, v, k in edge_group]
            group_widths = [inactive_edge_widths[idx] for idx, u, v, k in edge_group]
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=group_edges,
                edge_color=group_colors,
                width=group_widths,
                arrows=True,
                arrowsize=25,  # Smaller but still visible for inactive edges
                arrowstyle="-|>",
                node_size=3800,  # Match node size so arrows stop at node boundary
                alpha=0.4,
                connectionstyle="arc3,rad=0.0",
                min_target_margin=15,
                ax=ax,
            )

    # FIX: Draw edge labels ALONG the edge direction, not perpendicular
    for u, v, k in G.edges(keys=True):
        if (u, v, k) not in edge_labels:
            continue

        is_active = check_edge_active(u, v, k)
        is_structural = str(k).startswith("structural::")
        involves_inactive = u in inactive_node_set or v in inactive_node_set

        curvature = edge_curvatures.get((u, v, k), 0.0)
        label_t = edge_label_t_params.get((u, v, k), 0.5)
        lx, ly = compute_label_position_curved(pos, u, v, curvature, t=label_t)

        # Compute rotation angle to align label along edge direction
        label_angle = compute_label_angle_along_edge(pos, u, v, curvature, t=label_t)

        label_text = edge_labels[(u, v, k)]

        # colors
        if is_structural:
            label_color = "green"
        elif not is_active:
            label_color = "#999999"
        elif involves_inactive:
            label_color = "#999999"
        else:
            label_color = "darkred"

        # Common bbox style for ALL labels to ensure readability
        base_bbox = dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
            pad=0.3,
            boxstyle="round,pad=0.15",
        )

        if is_structural:
            ax.text(
                lx,
                ly,
                pretty_text(label_text),
                fontsize=12,
                fontweight="bold",
                color=label_color,
                ha="center",
                va="center",
                rotation=label_angle,
                rotation_mode="anchor",
                bbox=dict(
                    facecolor="white",
                    edgecolor="green",
                    alpha=0.95,
                    pad=0.3,
                    boxstyle="round,pad=0.15",
                ),
                zorder=10,
            )
        elif not is_active or involves_inactive:
            ax.text(
                lx,
                ly,
                pretty_text(label_text),
                fontsize=10,
                color=label_color,
                ha="center",
                va="center",
                rotation=label_angle,
                rotation_mode="anchor",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                    pad=0.25,
                    boxstyle="round,pad=0.1",
                ),
                zorder=9,
            )
        else:
            ax.text(
                lx,
                ly,
                pretty_text(label_text),
                fontsize=12,
                color=label_color,
                ha="center",
                va="center",
                rotation=label_angle,
                rotation_mode="anchor",
                bbox=base_bbox,
                zorder=10,
            )

    # Draw labels separately for active and inactive nodes
    active_labels = {n: pretty_text(n) for n in G.nodes() if n not in inactive_node_set}
    inactive_labels = {n: pretty_text(n) for n in G.nodes() if n in inactive_node_set}

    # Draw active node labels (bold, black)
    if active_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            labels=active_labels,
            font_size=11,
            font_weight="bold",
            font_color="black",
            ax=ax,
        )

    # Draw inactive node labels (lighter, gray)
    if inactive_labels:
        nx.draw_networkx_labels(
            G,
            pos,
            labels=inactive_labels,
            font_size=10,
            font_weight="normal",
            font_color="#777777",
            alpha=0.6,
            ax=ax,
        )

    def _normalize_title_line(text: str, max_len: int = 220) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        return cleaned

    def _filter_to_plotted(nodes: Optional[Iterable[str]], plotted_nodes: set[str]):
        if not nodes:
            return None
        return [n for n in nodes if n in plotted_nodes]

    def _format_nodes_line(
        label: str,
        nodes: Optional[Iterable[str]],
        width: int = 140,
    ) -> Optional[str]:
        if nodes is None:
            return None

        cleaned = [pretty_text(n) for n in nodes if n]
        if not cleaned:
            return f"{label}: none"

        joined = ", ".join(cleaned)

        wrapped = textwrap.fill(
            joined,
            width=width,
            initial_indent=f"{label}: ",
            subsequent_indent=" " * (len(label) + 2),
            break_long_words=False,
            break_on_hyphens=False,
        )

        return wrapped

    title_persona = (persona[:150] + "...") if len(persona) > 150 else persona
    ax.set_title(f"AMoC Knowledge Graph: {model_name}", size=20, pad=20)
    sup_lines = []
    persona_line = _normalize_title_line(title_persona, max_len=180)
    sentence_idx = None
    if persona_line:
        sup_lines.append(f"Persona: {persona_line}")
    if sentence_text:
        sentence_line = _normalize_title_line(sentence_text, max_len=220)
        if step_tag and "sent" in step_tag:
            # Extract sentence index from step_tag if present (e.g., sent3_active)
            import re as _re

            m = _re.search(r"sent(\d+)", step_tag)
            if m:
                sentence_idx = int(m.group(1))
                sentence_line = f"{sentence_idx}: {sentence_line}"
        sup_lines.append(f"Sentence {sentence_line}")
    # Use inactive_nodes_for_title for title display (separate from rendering)
    # This allows inactive nodes to appear in title but NOT be rendered in graph
    inactive_for_title = (
        inactive_nodes_for_title
        if inactive_nodes_for_title is not None
        else (inactive_nodes if inactive_nodes is not None else deactivated_concepts)
    )

    explicit_nodes_filtered = (
        list(explicit_nodes) if explicit_nodes is not None else None
    )
    salient_nodes_filtered = _filter_to_plotted(salient_nodes, plotted_nodes)
    # Show all inactive nodes in title (not filtered to plotted) since it's informational
    inactive_nodes_filtered = list(inactive_for_title) if inactive_for_title else None

    if inactive_for_title is not None:
        sup_lines.append("\n")
    for label, items in [
        ("Explicit this sentence", explicit_nodes_filtered),
        (
            "Carry-over from previous sentences",
            (
                salient_nodes_filtered
                if (sentence_idx is not None and sentence_idx > 1)
                else None
            ),
        ),
        ("Inactive nodes (retained in memory)", inactive_nodes_filtered),
    ]:
        line = _format_nodes_line(label, items)
        if line:
            sup_lines.append(line)
    if explicit_nodes is None and salient_nodes is None:
        fallback = _format_nodes_line("Active this sentence", new_nodes)
        if fallback:
            sup_lines.append(fallback)
    # Add hub edge explanations if provided (when explicit nodes needed hub-anchoring)
    if hub_edge_explanations:
        sup_lines.append("\n")
        sup_lines.append("Hub-anchored connections (LLM explanation):")
        for explanation in hub_edge_explanations[:3]:  # Limit to 3 to avoid clutter
            truncated = (
                (explanation[:120] + "...") if len(explanation) > 120 else explanation
            )
            sup_lines.append(f"  • {truncated}")
    plt.suptitle(
        "\n".join(sup_lines),
        y=0.98,
        fontsize=12,
        style="italic",
        color="darkblue",
    )

    # Draw active triplets in dedicated right panel
    if show_triplet_overlay:
        active_nodes_for_filter = plotted_nodes - inactive_node_set

        overlay_triplets = []
        if active_triplets_for_overlay:
            for s, r, o in active_triplets_for_overlay:
                if s in active_nodes_for_filter and o in active_nodes_for_filter:
                    overlay_triplets.append((s, r, o))
        else:
            # Fallback for callers that don't pass active_triplets_for_overlay
            for u, v, k in G.edges(keys=True):
                if (u, v, k) in edge_labels:
                    if u in active_nodes_for_filter and v in active_nodes_for_filter:
                        if check_edge_active(u, v, k):
                            overlay_triplets.append((u, edge_labels[(u, v, k)], v))

        draw_triplet_panel(
            ax_triplets,
            overlay_triplets,
            active_nodes=active_nodes_for_filter,
            max_triplets=45,
            font_size=9,
        )
    else:
        ax_triplets.axis("off")

    ax.axis("off")
    fig.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if positions is not None:
        for node, coords in pos.items():
            if node not in positions:
                positions[node] = coords

    return save_path
