import os
import re
import math
from typing import List, Tuple, Iterable, Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
from amoc.config.paths import OUTPUT_ANALYSIS_DIR  # reuse existing output base
from collections import defaultdict

DEFAULT_BLUE_NODES: Iterable[str] = ()

# LAYOUT POLICY: Option B + D — Sub-Rings as primary, Collision as fallback
# When nodes at the same hop distance exceed threshold, split into sub-rings
MAX_NODES_PER_RING = 6  # Split ring if more than this many nodes
SUB_RING_RADIUS_OFFSET = (
    0.15  # Small radial offset between sub-rings (fraction of ring_step)
)
MIN_SPACING_PADDING = 0.25  # 25% padding on node diameter for collision detection
RADIUS_GROWTH_MIN = 1.8  # Minimum allowed radius growth factor
RADIUS_GROWTH_MAX = 2.2  # Maximum allowed radius growth factor

TRIVIAL_NODE_TEXTS = {
    "and",
    "or",
    "but",
    "nor",
    "so",
    "yet",
    "through",
    "with",
    "without",
    "to",
    "of",
    "in",
    "on",
    "at",
    "from",
    "into",
    "onto",
    "by",
    "for",
    "about",
    "over",
    "under",
    "after",
    "before",
    "during",
    "while",
    "as",
}


def _pretty_text(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _is_trivial_node_text(text: str) -> bool:
    return _pretty_text(text).lower() in TRIVIAL_NODE_TEXTS


def _wrap_angle(angle: float) -> float:
    two_pi = 2.0 * math.pi
    return angle % two_pi


def _circular_distance(a: float, b: float) -> float:
    two_pi = 2.0 * math.pi
    diff = abs(a - b) % two_pi
    return min(diff, two_pi - diff)


def _min_pairwise_distance(pos: Dict[str, Tuple[float, float]]) -> float:
    coords = list(pos.values())
    if len(coords) < 2:
        return float("inf")
    min_dist = float("inf")
    for i in range(len(coords)):
        x1, y1 = coords[i]
        for j in range(i + 1, len(coords)):
            x2, y2 = coords[j]
            dist = math.hypot(x1 - x2, y1 - y2)
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _set_axes_limits(ax, pos: Dict[str, Tuple[float, float]]) -> None:
    xs = [x for x, _ in pos.values()]
    ys = [y for _, y in pos.values()]
    if not xs or not ys:
        return
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(1.0, max_x - min_x)
    dy = max(1.0, max_y - min_y)
    pad_x = dx * 0.18
    pad_y = dy * 0.18
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)


def _node_required_center_distance_data(
    fig,
    ax,
    *,
    node_size: float,
    pad_px: float = 6.0,
    use_percentage_padding: bool = True,
) -> float:
    # node_size is matplotlib scatter "s" in points^2 (area).
    # LAYOUT POLICY: Use 25% padding (MIN_SPACING_PADDING) when use_percentage_padding=True
    fig.canvas.draw()
    dpi = fig.dpi
    radius_pts = math.sqrt(float(node_size) / math.pi)
    radius_px = radius_pts * dpi / 72.0

    # Calculate padding: either use percentage of diameter or fixed pixels
    if use_percentage_padding:
        # 25% of node diameter = 50% of radius
        diameter_px = 2.0 * radius_px
        effective_pad_px = diameter_px * MIN_SPACING_PADDING
    else:
        effective_pad_px = pad_px

    # Convert px to data units via the current data transform.
    x0, y0 = ax.transData.transform((0.0, 0.0))
    x1, _ = ax.transData.transform((1.0, 0.0))
    _, y1 = ax.transData.transform((0.0, 1.0))
    px_per_x = max(1e-6, abs(x1 - x0))
    px_per_y = max(1e-6, abs(y1 - y0))
    radius_data = max(
        (radius_px + effective_pad_px) / px_per_x,
        (radius_px + effective_pad_px) / px_per_y,
    )
    return 2.0 * radius_data


def _ring_required_radius(n_ring: int, target_min_dist: float) -> float:
    if n_ring <= 1:
        return target_min_dist * 1.6
    # For two-node rings, keep edges off a straight line by using a 120° spread.
    angular_sep = (2.0 * math.pi / 3.0) if n_ring == 2 else (2.0 * math.pi / n_ring)
    chord_factor = 2.0 * math.sin(angular_sep / 2.0)
    if chord_factor <= 0.0:
        chord_factor = 1e-6
    return (target_min_dist / chord_factor) * 1.05


def _enforce_minimum_spacing(
    fig,
    ax,
    pos: Dict[str, Tuple[float, float]],
    hub: Optional[str],
    *,
    node_size: float = 3800,
    pad_px: float = 8.0,
    scale_step: float = 1.06,
    max_iter: int = 140,
    freeze_nodes: Optional[set[str]] = None,
) -> None:
    if len(pos) < 2:
        return
    for _ in range(max_iter):
        _set_axes_limits(ax, pos)
        required = _node_required_center_distance_data(
            fig, ax, node_size=node_size, pad_px=pad_px
        )
        if _min_pairwise_distance(pos) >= required:
            return
        # Radial scale (keep angles) until node circles no longer overlap.
        for node, (x, y) in list(pos.items()):
            if freeze_nodes and node in freeze_nodes:
                continue
            if hub is not None and node == hub:
                continue
            pos[node] = (x * scale_step, y * scale_step)


def _level_angle_seed(level: int) -> float:
    # Golden-angle based deterministic seed per ring to avoid collinear
    # single-node rings while keeping layouts stable across runs.
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    return _wrap_angle(level * golden_angle)


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> Tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay), ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y), proj_x, proj_y


def _push_nodes_off_edges(
    fig,
    ax,
    pos: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    *,
    node_size: float = 3800,
    pad_px: float = 12.0,
    corridor_scale: float = 1.25,
    max_iter: int = 140,
) -> None:
    if len(pos) < 3 or not edges:
        return
    edges_list = [(u, v) for u, v in edges if u in pos and v in pos]
    if not edges_list:
        return
    for _ in range(max_iter):
        _set_axes_limits(ax, pos)
        required = _node_required_center_distance_data(
            fig, ax, node_size=node_size, pad_px=pad_px
        )
        corridor = required * corridor_scale
        moved = False
        for node, (x, y) in list(pos.items()):
            for u, v in edges_list:
                if node == u or node == v:
                    continue
                ux, uy = pos[u]
                vx, vy = pos[v]
                dist, proj_x, proj_y = _point_segment_distance(x, y, ux, uy, vx, vy)
                if dist >= corridor:
                    continue
                dx = x - proj_x
                dy = y - proj_y
                norm = math.hypot(dx, dy)
                if norm < 1e-6:
                    # Nudge sideways if perfectly aligned
                    dx, dy = -(vy - uy), vx - ux
                    norm = math.hypot(dx, dy)
                dx /= norm
                dy /= norm
                gap = corridor - dist
                step = max(gap * 1.25, required * 0.12)
                x += dx * step
                y += dy * step
                moved = True
            pos[node] = (x, y)
        if not moved:
            break


def _enforce_min_edge_length(
    pos: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str]],
    *,
    min_len: float,
    hub: Optional[str] = None,
    step_factor: float = 1.04,
    max_iter: int = 80,
) -> None:
    if min_len <= 0 or not edges:
        return
    for _ in range(max_iter):
        moved = False
        for u, v in edges:
            if u not in pos or v not in pos:
                continue
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist >= min_len:
                continue
            if dist < 1e-6:
                dx, dy, dist = 1.0, 0.0, 1.0
            deficit = (min_len - dist) * step_factor
            if hub is not None and u == hub and v != hub:
                # Keep hub fixed; push v away from hub.
                pos[v] = (x2 + (dx / dist) * deficit, y2 + (dy / dist) * deficit)
            elif hub is not None and v == hub and u != hub:
                pos[u] = (x1 - (dx / dist) * deficit, y1 - (dy / dist) * deficit)
            else:
                shift = deficit * 0.5
                pos[u] = (x1 - (dx / dist) * shift, y1 - (dy / dist) * shift)
                pos[v] = (x2 + (dx / dist) * shift, y2 + (dy / dist) * shift)
            moved = True
        if not moved:
            break


def _split_into_subrings(
    nodes: List[str],
    base_radius: float,
    ring_step: float,
    level_seed: float,
) -> Dict[str, Tuple[float, float]]:
    """
    LAYOUT POLICY B: Sub-Ring Expansion
    When nodes exceed MAX_NODES_PER_RING, split them across multiple sub-rings
    at slightly different radii to prevent overcrowding.

    Returns a dict of node -> (x, y) positions.
    """
    if not nodes:
        return {}

    positions: Dict[str, Tuple[float, float]] = {}
    nodes = list(nodes)
    n_nodes = len(nodes)

    if n_nodes <= MAX_NODES_PER_RING:
        # No split needed - place all on single ring
        if n_nodes == 1:
            angle = level_seed
            positions[nodes[0]] = (
                base_radius * math.cos(angle),
                base_radius * math.sin(angle),
            )
        elif n_nodes == 2:
            spread = 2.0 * math.pi / 3.0  # 120° to avoid straight line
            angles = [
                _wrap_angle(level_seed - spread / 2.0),
                _wrap_angle(level_seed + spread / 2.0),
            ]
            for i, node in enumerate(nodes):
                positions[node] = (
                    base_radius * math.cos(angles[i]),
                    base_radius * math.sin(angles[i]),
                )
        else:
            for idx, node in enumerate(nodes):
                angle = 2.0 * math.pi * idx / n_nodes + level_seed
                positions[node] = (
                    base_radius * math.cos(angle),
                    base_radius * math.sin(angle),
                )
        return positions

    # Split into sub-rings
    num_subrings = (n_nodes + MAX_NODES_PER_RING - 1) // MAX_NODES_PER_RING
    sub_ring_offset = ring_step * SUB_RING_RADIUS_OFFSET

    for subring_idx in range(num_subrings):
        start_idx = subring_idx * MAX_NODES_PER_RING
        end_idx = min(start_idx + MAX_NODES_PER_RING, n_nodes)
        subring_nodes = nodes[start_idx:end_idx]
        n_subring = len(subring_nodes)

        # Offset radius for each sub-ring
        subring_radius = base_radius + (subring_idx * sub_ring_offset)

        # Offset the starting angle for each sub-ring to stagger nodes
        subring_seed = level_seed + (subring_idx * math.pi / (num_subrings * 2))

        for idx, node in enumerate(subring_nodes):
            angle = 2.0 * math.pi * idx / n_subring + subring_seed
            positions[node] = (
                subring_radius * math.cos(angle),
                subring_radius * math.sin(angle),
            )

    return positions


def _choose_angles_in_gaps(
    *,
    existing_angles: Dict[str, float],
    missing_nodes: List[str],
    candidate_count: int,
    penalty_target: Dict[str, float] | None = None,
    penalty_weight: float = 0.0,
) -> Dict[str, float]:
    if not missing_nodes:
        return {}
    missing_nodes = list(missing_nodes)
    candidates = [2.0 * math.pi * i / candidate_count for i in range(candidate_count)]
    occupied: List[float] = list(existing_angles.values())
    chosen: Dict[str, float] = {}
    for node in missing_nodes:
        best_angle: float | None = None
        best_score = -1e9
        target = penalty_target.get(node) if penalty_target else None
        for cand in candidates:
            min_sep = min(
                (_circular_distance(cand, occ) for occ in occupied), default=math.pi
            )
            score = min_sep
            if target is not None and penalty_weight:
                score -= penalty_weight * _circular_distance(cand, target)
            if score > best_score:
                best_score = score
                best_angle = cand
        if best_angle is None:
            best_angle = 0.0
        chosen[node] = best_angle
        occupied.append(best_angle)
        try:
            candidates.remove(best_angle)
        except ValueError:
            pass
    return chosen


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
    salient_nodes: Optional[Iterable[str]] = None,
    inactive_nodes: Optional[Iterable[str]] = None,
    inactive_nodes_for_title: Optional[Iterable[str]] = None,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    avoid_edge_overlap: bool = True,
    active_edges: Optional[set[Tuple[str, str, str]]] = None,
    hub_edge_explanations: Optional[List[str]] = None,
    show_all_edges: bool = False,
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

    # =========================================================================
    # INVARIANT ENFORCEMENT DOCUMENTATION
    # =========================================================================
    #
    # This plotting function enforces the following strict invariants:
    #
    # INVARIANT 1 — No Dangling Nodes:
    #   Nodes must be derived EXCLUSIVELY from triplets. If a node has no edge,
    #   it must NOT appear. Connectivity is enforced UPSTREAM in core.py via
    #   _ensure_explicit_backbone_connected() and _ensure_displayed_nodes_connected().
    #
    # INVARIANT 2 — Graph Must Always Be Connected:
    #   The graph passed here MUST already be connected. Disconnected graphs
    #   are a hard error, not a layout problem. No visual compensation allowed.
    #
    # INVARIANT 3 — Plot Must Match Active Triplet CSV 1:1:
    #   Every plotted edge must exist in the CSV. No edge invention, no
    #   "__implicit__" synthesis. Empty relations are SKIPPED, not downgraded.
    #
    # DIVISION OF RESPONSIBILITY:
    #   - core.py: Enforces connectivity, creates edges, handles LLM fallback
    #   - graph_plots.py: Purely representational, assumes valid input
    # =========================================================================

    G = nx.MultiDiGraph()

    edge_labels: Dict[Tuple[str, str, str], str] = {}
    edge_status: Dict[Tuple[str, str, str], str] = {}
    active_edge_set = active_edges or set()

    for src, rel, dst in triplets:
        src = str(src).strip()
        dst = str(dst).strip()
        rel = str(rel).strip()

        # INVARIANT: semantic triplets must be complete
        if not src or not dst or not rel:
            continue

        is_structural = rel.startswith("structural::")
        clean_rel = rel.replace("structural::", "").strip()

        # Unique key per semantic relation
        edge_key = f"{clean_rel}"

        G.add_edge(src, dst, key=edge_key)

        edge_labels[(src, dst, edge_key)] = clean_rel

        if is_structural:
            edge_status[(src, dst, edge_key)] = "structural"
        else:
            edge_status[(src, dst, edge_key)] = "normal"

    # INVARIANT 1: Nodes derived ONLY from triplets - no injection
    if G.number_of_nodes() == 0:
        return save_path

    # INVARIANT 2: Hard assertion - graph MUST be connected
    # Disconnected graphs are a hard error, not a layout problem
    if G.number_of_nodes() > 1:
        undirected_G = G.to_undirected()
        if not nx.is_connected(undirected_G):
            components = list(nx.connected_components(undirected_G))
            component_sizes = [len(c) for c in components]
            raise RuntimeError(
                f"INVARIANT 2 VIOLATION: Disconnected graph passed to plot_amoc_triplets. "
                f"Found {len(components)} components with sizes {component_sizes}. "
                f"Connectivity must be enforced upstream in core.py, not compensated visually."
            )

    plotted_nodes = set(G.nodes())

    # largest_component_only is now only for backward compatibility
    # With strict invariants, the graph should always be connected

    fig, ax = plt.subplots(figsize=(22, 18))

    # Build sets for node categorization
    inactive_node_set = set(inactive_nodes) if inactive_nodes else set()
    explicit_node_set = set(explicit_nodes) if explicit_nodes else set()
    salient_node_set = set(salient_nodes) if salient_nodes else set()

    # Node colors: prioritize explicit/salient > inactive > default
    # Explicit/Salient (blue_nodes): bright blue #a0cbe2
    # Active (not in blue_nodes): bright yellow #ffe8a0
    # Inactive: faded gray #d0d0d0
    node_colors = []
    node_alphas = []
    for node in G.nodes():
        if node in inactive_node_set:
            # Inactive nodes: gray and faded
            node_colors.append("#d0d0d0")
            node_alphas.append(0.5)
        elif node in blue_nodes:
            # Explicit/salient nodes: bright blue
            node_colors.append("#a0cbe2")
            node_alphas.append(1.0)
        else:
            # Other active nodes: bright yellow
            node_colors.append("#ffe8a0")
            node_alphas.append(1.0)

    pos: Dict[str, Tuple[float, float]] = {}
    nodes = list(G.nodes())
    position_cache = positions or {}
    fixed_pos: Dict[str, Tuple[float, float]] = {
        node: position_cache[node] for node in nodes if node in position_cache
    }
    fixed_nodes = set(fixed_pos.keys())
    hub: Optional[str] = None

    max_label_len = max((len(_pretty_text(n)) for n in nodes), default=0)
    target_min_dist = 7.0 + max(0, max_label_len - 10) * 0.12

    if len(nodes) == 1:
        pos[nodes[0]] = (0.0, 0.0)
        hub = nodes[0]
    else:
        UG = G

        # Single-hub radial layout (hub centered) to keep the nodes at fixed positions
        if (
            positions is not None
            and "__HUB__" in positions
            and positions["__HUB__"] in nodes
        ):
            hub = positions["__HUB__"]
        else:
            hub_candidates = [n for n in nodes if n in blue_nodes] or nodes
            hub = max(hub_candidates, key=lambda n: (UG.degree(n), str(n)))
            if positions is not None:
                positions["__HUB__"] = hub

        pos[hub] = (0.0, 0.0)
        freeze_nodes = set(fixed_nodes)
        if hub is not None:
            freeze_nodes.add(hub)

        levels = nx.single_source_shortest_path_length(UG, hub)
        max_level = max(levels.values(), default=0)

        ring_step = max(12.0, target_min_dist * 1.55)
        radii: Dict[int, float] = {}

        # If we have cached coordinates, keep existing nodes fixed and only
        # place newly appearing nodes.
        # LAYOUT POLICY: Only new nodes may be adjusted (frozen nodes stay fixed)
        if fixed_pos:
            pos = dict(fixed_pos)
            if hub not in pos:
                pos[hub] = (0.0, 0.0)
            movable_nodes = [n for n in nodes if n not in pos]

            # Calculate initial radius for bounded growth enforcement
            initial_radius = max(
                (math.hypot(x, y) for x, y in pos.values() if (x, y) != (0.0, 0.0)),
                default=ring_step,
            )
            max_allowed_radius = initial_radius * RADIUS_GROWTH_MAX

            # Precompute ring radii based on both fixed + new nodes per level,
            # but never pull a new node inward past any already-placed node.
            max_r_by_level: Dict[int, float] = {}
            for node, (x, y) in pos.items():
                if node == hub:
                    continue
                lvl = levels.get(node)
                if lvl is None:
                    continue
                max_r_by_level[lvl] = max(
                    max_r_by_level.get(lvl, 0.0), math.hypot(x, y)
                )
            for level in range(1, max_level + 1):
                ring_nodes = [n for n in nodes if levels.get(n) == level]
                n_ring = len(ring_nodes)
                # LAYOUT POLICY B: Use effective count for sub-ring calculation
                effective_n = min(n_ring, MAX_NODES_PER_RING)
                required_r = _ring_required_radius(effective_n, target_min_dist)
                prev_r = radii.get(level - 1, 0.0)
                radii[level] = max(
                    required_r, prev_r + ring_step, max_r_by_level.get(level, 0.0)
                )

            # LAYOUT POLICY B: Place new nodes using sub-rings when overcrowded
            # Fill angular gaps around already placed nodes (if any).
            for level in range(1, max_level + 1):
                ring_new = sorted([n for n in movable_nodes if levels.get(n) == level])
                if not ring_new:
                    continue
                r = radii[level]

                # Check if we need sub-rings for new nodes at this level
                existing_at_level = [
                    n for n in pos if n != hub and levels.get(n) == level
                ]
                total_at_level = len(existing_at_level) + len(ring_new)

                if total_at_level > MAX_NODES_PER_RING and len(ring_new) > 1:
                    # Use sub-ring placement for new nodes only
                    level_seed = _level_angle_seed(level)
                    # Offset sub-ring seed to avoid existing nodes
                    existing_angles = [
                        _wrap_angle(math.atan2(y, x))
                        for n, (x, y) in pos.items()
                        if n in existing_at_level and (x != 0.0 or y != 0.0)
                    ]
                    if existing_angles:
                        # Find largest gap and place sub-rings there
                        avg_existing = sum(existing_angles) / len(existing_angles)
                        level_seed = _wrap_angle(
                            avg_existing + math.pi
                        )  # Opposite side

                    subring_positions = _split_into_subrings(
                        ring_new, r, ring_step, level_seed
                    )
                    pos.update(subring_positions)
                else:
                    # Standard gap-filling for small number of new nodes
                    default_angle = _level_angle_seed(level)
                    existing_angles = {
                        n: _wrap_angle(math.atan2(y, x))
                        for n, (x, y) in pos.items()
                        if n != hub
                        and levels.get(n) == level
                        and (x != 0.0 or y != 0.0)
                    }
                    chosen = _choose_angles_in_gaps(
                        existing_angles=existing_angles,
                        missing_nodes=ring_new,
                        candidate_count=max(
                            12, len(ring_new) + len(existing_angles) + 6
                        ),
                    )
                    for node in ring_new:
                        angle = chosen.get(node, default_angle)
                        pos[node] = (r * math.cos(angle), r * math.sin(angle))

            # Any node still not placed (shouldn't happen) goes to an outer ring.
            max_existing_r = max(
                (math.hypot(x, y) for x, y in pos.values()), default=0.0
            )
            for node in movable_nodes:
                if node in pos:
                    continue
                r = min(max_existing_r + ring_step, max_allowed_radius)
                pos[node] = (r, 0.0)

            # LAYOUT POLICY D: Collision-avoidance pass as fallback
            # Only moves newly placed nodes (keeps cached nodes in place).
            # BOUNDED RADIUS GROWTH: Respect max_allowed_radius
            movable_sorted = sorted(movable_nodes)
            for _ in range(180):
                _set_axes_limits(ax, pos)
                required = _node_required_center_distance_data(
                    fig, ax, node_size=3800, use_percentage_padding=True
                )
                moved = False
                for node in movable_nodes:
                    if node not in pos:
                        continue
                    x, y = pos[node]
                    if x == 0.0 and y == 0.0:
                        x = required
                        y = 0.0
                    # Push outward until no overlap with any other node.
                    for _inner in range(12):
                        min_dist = float("inf")
                        for other, (ox, oy) in pos.items():
                            if other == node:
                                continue
                            d = math.hypot(x - ox, y - oy)
                            if d < min_dist:
                                min_dist = d
                        if min_dist >= required:
                            break
                        # Check bounded growth before scaling
                        new_r = math.hypot(x * 1.10, y * 1.10)
                        if new_r > max_allowed_radius:
                            break  # Don't exceed bounded growth
                        x *= 1.10
                        y *= 1.10
                        moved = True
                    pos[node] = (x, y)
                # Also ensure movable nodes don't overlap each other after scaling.
                if not moved:
                    # Quick check
                    any_overlap = False
                    for i1, n1 in enumerate(movable_sorted):
                        if n1 not in pos:
                            continue
                        x1, y1 = pos[n1]
                        for n2 in movable_sorted[i1 + 1 :]:
                            if n2 not in pos:
                                continue
                            x2, y2 = pos[n2]
                            if math.hypot(x1 - x2, y1 - y2) < required:
                                any_overlap = True
                                break
                        if any_overlap:
                            break
                    if not any_overlap:
                        break
        else:
            # No cache: compute a full radial layout from scratch.
            # LAYOUT POLICY B: Use sub-rings when nodes exceed MAX_NODES_PER_RING
            # Precompute counts per ring to choose radii that avoid overlap on the ring.
            for level in range(1, max_level + 1):
                ring_nodes = [n for n in nodes if levels.get(n) == level]
                n_ring = len(ring_nodes)
                # For sub-ring calculation, use effective count per sub-ring
                effective_n = min(n_ring, MAX_NODES_PER_RING)
                required_r = _ring_required_radius(effective_n, target_min_dist)
                prev_r = radii.get(level - 1, 0.0)
                radii[level] = max(required_r, prev_r + ring_step)

            pos = {hub: (0.0, 0.0)}
            initial_radius = radii.get(1, ring_step)  # Track for bounded growth

            for level in range(1, max_level + 1):
                ring = sorted([n for n in nodes if levels.get(n) == level])
                if not ring:
                    continue
                r = radii[level]

                # LAYOUT POLICY B: Use sub-ring placement for overcrowded rings
                level_seed = _level_angle_seed(level)
                subring_positions = _split_into_subrings(ring, r, ring_step, level_seed)
                pos.update(subring_positions)

            # LAYOUT POLICY D: Collision-driven adjustment as fallback
            # Scale outward until node circles cannot overlap in screen space.
            # BOUNDED RADIUS GROWTH: Limit scaling to RADIUS_GROWTH_MAX
            max_allowed_radius = initial_radius * RADIUS_GROWTH_MAX
            for _ in range(120):
                _set_axes_limits(ax, pos)
                required = _node_required_center_distance_data(
                    fig, ax, node_size=3800, use_percentage_padding=True
                )
                if _min_pairwise_distance(pos) >= required:
                    break
                # Check if we've hit the radius bound
                current_max_r = max(
                    (math.hypot(x, y) for x, y in pos.values() if (x, y) != (0.0, 0.0)),
                    default=0.0,
                )
                if current_max_r >= max_allowed_radius:
                    break  # Don't exceed bounded growth
                for node, (x, y) in list(pos.items()):
                    if node == hub:
                        continue
                    new_x, new_y = x * 1.08, y * 1.08
                    # Respect bounded growth
                    if math.hypot(new_x, new_y) <= max_allowed_radius:
                        pos[node] = (new_x, new_y)
            _set_axes_limits(ax, pos)
            required = _node_required_center_distance_data(
                fig, ax, node_size=3800, use_percentage_padding=True
            )
            min_dist = _min_pairwise_distance(pos)
            if min_dist > 0 and min_dist < required:
                factor = min(
                    (required / min_dist) * 1.02,
                    RADIUS_GROWTH_MAX / RADIUS_GROWTH_MIN,
                )
                for node, (x, y) in list(pos.items()):
                    if node == hub:
                        continue
                    new_x, new_y = x * factor, y * factor
                    # Respect bounded growth
                    if math.hypot(new_x, new_y) <= max_allowed_radius:
                        pos[node] = (new_x, new_y)

    if pos and hub is None:
        hub = next(iter(pos.keys()))
    min_edge_len = max(6.5, target_min_dist * 0.9)
    _enforce_min_edge_length(
        pos, [(u, v) for u, v, _ in G.edges(keys=True)], min_len=min_edge_len, hub=hub
    )
    if avoid_edge_overlap:
        _push_nodes_off_edges(fig, ax, pos, list(G.edges()))
    _enforce_minimum_spacing(
        fig,
        ax,
        pos,
        hub,
        node_size=3800,
        pad_px=8.0,
        scale_step=1.06,
        max_iter=140,
        freeze_nodes=fixed_nodes,
    )
    if avoid_edge_overlap:
        _push_nodes_off_edges(fig, ax, pos, list(G.edges()))

    # LAYOUT POLICY D: Final collision-driven adjustment as safety net
    # Ensure no overlapping nodes by spreading them out (only new nodes move)
    _set_axes_limits(ax, pos)
    required_dist = _node_required_center_distance_data(
        fig, ax, node_size=3800, use_percentage_padding=True  # 25% padding
    )
    for _ in range(50):  # Limited iterations for final pass
        collision_found = False
        nodes_list = list(pos.keys())
        for i, n1 in enumerate(nodes_list):
            x1, y1 = pos[n1]
            for n2 in nodes_list[i + 1 :]:
                x2, y2 = pos[n2]
                dist = math.hypot(x2 - x1, y2 - y1)
                if dist < required_dist and dist > 1e-6:
                    # Push apart along the line connecting them
                    collision_found = True
                    dx, dy = x2 - x1, y2 - y1
                    norm = math.hypot(dx, dy)
                    dx, dy = dx / norm, dy / norm
                    push = (required_dist - dist) * 0.55
                    # LAYOUT POLICY: Only move non-frozen, non-hub nodes (new nodes only)
                    if n1 != hub and n1 not in fixed_nodes:
                        pos[n1] = (x1 - dx * push, y1 - dy * push)
                    if n2 != hub and n2 not in fixed_nodes:
                        pos[n2] = (x2 + dx * push, y2 + dy * push)
        if not collision_found:
            break

    # Draw inactive nodes first (underneath) with faded appearance
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
        active_colors = [
            "#a0cbe2" if node in blue_nodes else "#ffe8a0" for node in active_in_graph
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

    for u, v, k in G.edges(keys=True):
        status = edge_status.get((u, v, k), "normal")
        is_active = (u, v, k) in active_edge_set
        involves_inactive = u in inactive_node_set or v in inactive_node_set

        if status == "structural":
            structural_edges.append((u, v, k))
            structural_edge_colors.append(
                "green" if not involves_inactive else "#90c090"
            )
            structural_edge_widths.append(2.5 if not involves_inactive else 1.5)

        elif status == "implicit":
            implicit_edges.append((u, v, k))
            implicit_edge_colors.append("#999999" if not is_active else "#666699")
            implicit_edge_widths.append(1.0)

        elif involves_inactive:
            # Edges involving inactive nodes: faded gray
            inactive_edges.append((u, v, k))
            inactive_edge_colors.append("#cccccc")
            inactive_edge_widths.append(0.8)

        else:
            normal_edges.append((u, v, k))
            if is_active:
                normal_edge_colors.append("black")
                normal_edge_widths.append(1.3)
            else:
                normal_edge_colors.append("#cccccc")
                normal_edge_widths.append(1.2)

    # Draw normal edges as solid lines
    if normal_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v) for u, v, k in normal_edges],
            edge_color=normal_edge_colors,
            arrows=True,
            arrowsize=16,
            width=normal_edge_widths,
            connectionstyle="arc3,rad=0.2",
            ax=ax,
        )

    # POLICY A: Draw implicit edges as dashed lines (style rather than remove)
    if implicit_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=implicit_edges,
            edge_color=implicit_edge_colors,
            arrows=True,
            arrowsize=14,
            width=implicit_edge_widths,
            style="dashed",
            connectionstyle="arc3,rad=0.2",
            ax=ax,
        )

    if structural_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=structural_edges,
            edge_color=structural_edge_colors,
            width=structural_edge_widths,
            style="dashed",
            arrows=True,
            arrowsize=18,
            connectionstyle="arc3,rad=0.2",
            ax=ax,
        )

    # Draw edges involving inactive nodes with faded appearance
    if inactive_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=inactive_edges,
            edge_color=inactive_edge_colors,
            width=inactive_edge_widths,
            arrows=True,
            arrowsize=12,
            alpha=0.4,
            connectionstyle="arc3,rad=0.0",
            ax=ax,
        )

    def _label_offset(u: str, v: str) -> Tuple[float, float]:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return (x1 + x2) * 0.5, (y1 + y2) * 0.5
        t = 0.5
        return x1 * (1.0 - t) + x2 * t, y1 * (1.0 - t) + y2 * t

    for u, v, k in G.edges(keys=True):
        if (u, v, k) not in edge_labels:
            continue

        lx, ly = _label_offset(u, v)
        label_text = edge_labels[(u, v, k)]
        status = edge_status.get((u, v, k), "normal")
        involves_inactive = u in inactive_node_set or v in inactive_node_set

        if status == "structural":
            ax.text(
                lx,
                ly,
                _pretty_text(label_text),
                fontsize=12,
                fontweight="bold",
                color="green",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="green", pad=0.25),
            )
        elif status == "implicit":
            ax.text(
                lx,
                ly,
                "(implicit)",
                fontsize=11,
                fontstyle="italic",
                color="#666699",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )
        elif involves_inactive:
            ax.text(
                lx,
                ly,
                _pretty_text(label_text),
                fontsize=10,
                color="#999999",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )
        else:
            ax.text(
                lx,
                ly,
                _pretty_text(label_text),
                fontsize=12,
                color="darkred",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )

    # Draw labels separately for active and inactive nodes
    active_labels = {
        n: _pretty_text(n) for n in G.nodes() if n not in inactive_node_set
    }
    inactive_labels = {n: _pretty_text(n) for n in G.nodes() if n in inactive_node_set}

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
        # Avoid extremely long headers (e.g., if a full prompt leaks in) that
        # blow up the canvas and distort node placement by compacting and
        # truncating the line.
        cleaned = re.sub(r"<[^>]+>", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if len(cleaned) > max_len:
            cleaned = cleaned[: max_len - 3].rstrip() + "..."
        return cleaned

    def _filter_to_plotted(nodes: Optional[Iterable[str]], plotted_nodes: set[str]):
        if not nodes:
            return None
        return [n for n in nodes if n in plotted_nodes]

    def _format_nodes_line(label: str, nodes: Optional[Iterable[str]]) -> Optional[str]:
        if nodes is None:
            return None
        cleaned = [_pretty_text(n) for n in nodes if n]
        if not cleaned:
            return f"{label}: none"
        return f"{label}: " + ", ".join(cleaned)

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

    explicit_nodes_filtered = _filter_to_plotted(explicit_nodes, plotted_nodes)
    salient_nodes_filtered = _filter_to_plotted(salient_nodes, plotted_nodes)
    inactive_nodes_filtered = _filter_to_plotted(inactive_for_title, plotted_nodes)

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
    ax.axis("off")
    fig.savefig(save_path, format="PNG", dpi=300, bbox_inches="tight")
    plt.close(fig)

    if positions is not None:
        # Keep existing coordinates for nodes not present in this snapshot so
        # layout stays stable as nodes disappear/reappear across sentences.
        positions.update(pos)

    return save_path
