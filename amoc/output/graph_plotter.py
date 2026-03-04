from typing import TYPE_CHECKING, Optional, List, Tuple, Set, Dict, Iterable, Callable
import os
import logging
import math
import networkx as nx
import re

from amoc.graph.node import NodeType, NodeSource
from amoc.viz.graph_plots import plot_amoc_triplets

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node


class GraphPlotter:
    # Prompt contamination patterns to detect
    PROMPT_CONTAMINATION_PATTERNS = [
        "the text is:",
        "here is the text:",
        "the sentence is:",
        "replace the pronouns",
    ]

    def __init__(
        self,
        graph_ref: "Graph",
        output_dir: str,
        model_name: str,
        persona: str,
        persona_age: Optional[int] = None,
        layout_depth: int = 3,
        allow_multi_edges: bool = True,
    ):
        self._graph = graph_ref
        self._output_dir = output_dir
        self._model_name = model_name
        self._persona = persona
        self._persona_age = persona_age
        self._layout_depth = layout_depth
        self._allow_multi_edges = allow_multi_edges
        self._viz_positions: Dict[str, Tuple[float, float]] = {}
        self._prev_active_nodes: Set["Node"] = set()
        self._cumulative_deactivated_nodes: Set["Node"] = set()
        self._get_explicit_nodes_fn: Optional[Callable] = None
        self._get_edge_activation_scores_fn: Optional[Callable] = None
        self._graph_edges_to_triplets_fn: Optional[Callable] = None
        self._enforce_cumulative_connectivity_fn: Optional[Callable] = None
        self._story_lemmas: Set[str] = set()
        self._persona_only_lemmas: Set[str] = set()
        self._previous_active_triplets: List[Tuple[str, str, str]] = []

    def set_callbacks(
        self,
        get_explicit_nodes_fn: Callable,
        get_edge_activation_scores_fn: Callable,
        graph_edges_to_triplets_fn: Callable,
        enforce_cumulative_connectivity_fn: Callable,
    ):
        self._get_explicit_nodes_fn = get_explicit_nodes_fn
        self._get_edge_activation_scores_fn = get_edge_activation_scores_fn
        self._graph_edges_to_triplets_fn = graph_edges_to_triplets_fn
        self._enforce_cumulative_connectivity_fn = enforce_cumulative_connectivity_fn

    def set_lemmas(self, story_lemmas: Set[str], persona_only_lemmas: Set[str]):
        self._story_lemmas = story_lemmas
        self._persona_only_lemmas = persona_only_lemmas

    def get_viz_positions(self) -> Dict[str, Tuple[float, float]]:
        return self._viz_positions

    def set_viz_positions(self, value: Dict[str, Tuple[float, float]]) -> None:
        self._viz_positions = value

    def set_layout_depth(self, depth: int):
        self._layout_depth = depth

    def get_filtered_triplets_for_plot(
        self,
        triplets: List[Tuple[str, str, str]],
        active_node_names: Set[str],
    ) -> List[Tuple[str, str, str]]:
        return [
            (s, r, o)
            for (s, r, o) in triplets
            if s in active_node_names and o in active_node_names
        ]

    def plot_graph_snapshot(
        self,
        triplets: List[Tuple[str, str, str]],
        sentence_idx: int,
        sentence_text: str,
        explicit_nodes: Set["Node"],
        anchor_nodes: Set["Node"],
        output_path: Optional[str] = None,
        plot_fn: callable = None,
    ) -> Optional[str]:
        if plot_fn is None:
            return None

        # Filter triplets to active nodes only
        active_nodes, _ = self._graph.get_active_subgraph_wrapper()
        active_node_names = {n.get_text_representer() for n in active_nodes}
        filtered_triplets = self.get_filtered_triplets_for_plot(
            triplets, active_node_names
        )

        if not filtered_triplets:
            logging.warning(f"No active triplets for sentence {sentence_idx}")
            return None

        # Determine output path
        if output_path is None:
            safe_persona = self._sanitize_filename(self._persona[:50])
            output_path = os.path.join(
                self._output_dir,
                self._model_name,
                safe_persona,
                f"sentence_{sentence_idx}.png",
            )

        # Create directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get node categories
        explicit_names = {n.get_text_representer() for n in explicit_nodes}
        anchor_names = {n.get_text_representer() for n in anchor_nodes}

        # Compute newly activated nodes
        current_active = set(active_node_names)
        newly_activated = current_active - {
            n.get_text_representer() for n in self._prev_active_nodes
        }

        # Call plot function
        self._viz_positions = plot_fn(
            filtered_triplets,
            title=f"Sentence {sentence_idx}: {sentence_text[:60]}...",
            output_path=output_path,
            explicit_nodes=explicit_names,
            anchor_nodes=anchor_names,
            newly_activated=newly_activated,
            prev_positions=self._viz_positions,
            layout_depth=self._layout_depth,
        )

        # Update state
        self._prev_active_nodes = active_nodes
        self._cumulative_deactivated_nodes.update(
            self._prev_active_nodes - active_nodes
        )

        return output_path

    def _sanitize_filename(self, name: str, max_len: int = 80) -> str:
        name = (name or "").replace("\n", " ").strip()
        name = name[:max_len]
        name = re.sub(r"[\\/:*?\"<>|]", "_", name)
        name = re.sub(r"\s+", "_", name)
        return name or "unknown"

    def plot_sentence(
        self,
        sentence_idx: int,
        sentence_text: str,
        triplets: List[Tuple[str, str, str]],
        explicit_nodes: Set["Node"],
        anchor_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
        per_sentence_view,
        plot_fn: callable = None,
        matrix_suffix: Optional[str] = None,
    ) -> Optional[str]:
        if plot_fn is None:
            return None

        # Get active subgraph
        active_nodes, active_edges = self._graph.get_active_subgraph_wrapper()
        active_node_names = {n.get_text_representer() for n in active_nodes}

        # Filter triplets
        filtered_triplets = self.get_filtered_triplets_for_plot(
            triplets, active_node_names
        )

        if not filtered_triplets:
            return None

        # Build output path
        safe_persona = self._sanitize_filename(self._persona[:50])
        suffix_part = f"_{matrix_suffix}" if matrix_suffix else ""
        output_path = os.path.join(
            self._output_dir,
            self._model_name,
            safe_persona,
            f"sentence_{sentence_idx}{suffix_part}.png",
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get node categories
        explicit_names = {n.get_text_representer() for n in explicit_nodes}
        anchor_names = {n.get_text_representer() for n in anchor_nodes}
        carryover_names = {n.get_text_representer() for n in carryover_nodes}

        # Newly activated
        newly_activated = active_node_names - {
            n.get_text_representer() for n in self._prev_active_nodes
        }

        # Recently deactivated
        deactivated_names = {
            n.get_text_representer() for n in self._cumulative_deactivated_nodes
        }

        self._viz_positions = plot_fn(
            filtered_triplets,
            title=f"S{sentence_idx}: {sentence_text[:50]}...",
            output_path=output_path,
            explicit_nodes=explicit_names,
            anchor_nodes=anchor_names,
            carryover_nodes=carryover_names,
            newly_activated=newly_activated,
            deactivated=deactivated_names,
            prev_positions=self._viz_positions,
            layout_depth=self._layout_depth,
        )

        self._prev_active_nodes = active_nodes

        return output_path

    def plot_graph_snapshot_full(
        self,
        sentence_index: int,
        sentence_text: str,
        output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        only_active: bool = False,
        largest_component_only: bool = False,
        mode: str = "sentence_active",
        triplets_override: Optional[List[Tuple[str, str, str]]] = None,
        active_edges: Optional[set] = None,
        explicit_nodes: Optional[List[str]] = None,
        salient_nodes: Optional[List[str]] = None,
        inactive_nodes: Optional[List[str]] = None,
        active_triplets_for_overlay: Optional[List[Tuple[str, str, str]]] = None,
        property_nodes: Optional[List[str]] = None,
    ) -> None:

        sentence_text_lower = (sentence_text or "").lower().strip()
        for pattern in self.PROMPT_CONTAMINATION_PATTERNS:
            if sentence_text_lower.startswith(pattern):
                logging.error(
                    "Prompt leaked into sentence_text: %s",
                    sentence_text[:100],
                )
                sentence_text = sentence_text[len(pattern) :].strip()
                break

        # Route per-sentence plots into mode-specific subfolders
        plot_dir = output_dir
        if output_dir and mode in {"sentence_active", "sentence_cumulative"}:
            subdir = "active" if mode == "sentence_active" else "cumulative"
            plot_dir = os.path.join(output_dir, subdir)

        # Get triplets
        triplets = (
            triplets_override
            if triplets_override is not None
            else (
                self._graph_edges_to_triplets_fn(only_active=only_active)
                if self._graph_edges_to_triplets_fn
                else []
            )
        )

        # Build ACTIVE snapshot for safe plotting
        active_nodes_for_filter, _ = self._graph.get_active_subgraph_wrapper()
        active_node_names = {n.get_text_representer() for n in active_nodes_for_filter}

        # HARD EXPLICIT NODE VISIBILITY GUARANTEE
        if not triplets and explicit_nodes:
            triplets = []

        age_for_filename = self._persona_age if self._persona_age is not None else -1

        try:
            # Ensure every node that may be plotted has a concrete position
            nodes_to_plot = set()

            for u, _, v in triplets:
                if u:
                    nodes_to_plot.add(u)
                if v:
                    nodes_to_plot.add(v)

            # Explicit nodes must ALWAYS be plotted
            if explicit_nodes:
                for node_text in explicit_nodes:
                    if node_text:
                        nodes_to_plot.add(node_text)

            for lst in (inactive_nodes, explicit_nodes, salient_nodes, highlight_nodes):
                if lst:
                    nodes_to_plot.update(lst)

            # Assign positions for any missing nodes
            if nodes_to_plot:
                for node_text in nodes_to_plot:
                    if node_text not in self._viz_positions:
                        idx = len(self._viz_positions)
                        angle = 2 * math.pi * idx / max(1, len(nodes_to_plot))
                        radius = 1 + 0.5 * idx
                        self._viz_positions[node_text] = (
                            radius * math.cos(angle),
                            radius * math.sin(angle),
                        )

                # Keep historical nodes for cumulative plots
                for node_text in list(self._viz_positions.keys()):
                    if node_text not in nodes_to_plot:
                        nodes_to_plot.add(node_text)

            # PROVENANCE SANITY CHECK
            provenance_warnings = self._graph.sanity_check_provenance_wrapper(
                story_lemmas=self._story_lemmas,
                persona_only_lemmas=self._persona_only_lemmas,
            )
            for warning in provenance_warnings:
                logging.warning(warning)

            blue_nodes_combined = set()
            if highlight_nodes:
                blue_nodes_combined.update(highlight_nodes)

            # Get explicit nodes for plot
            explicit_nodes_current = (
                self._get_explicit_nodes_fn() if self._get_explicit_nodes_fn else set()
            )
            explicit_nodes_for_plot = sorted(
                {
                    n.get_text_representer()
                    for n in explicit_nodes_current
                    if n.get_text_representer()
                }
            )

            # ever_explicit
            ever_explicit_nodes_for_plot = sorted(
                {
                    node.get_text_representer()
                    for node in self._graph.nodes
                    if node.ever_explicit and node.get_text_representer()
                }
            )

            # Paper-aligned: Yellow = INFERENCE_BASED nodes
            inferred_nodes_for_plot = sorted(
                {
                    n.get_text_representer()
                    for n in active_nodes_for_filter
                    if n.node_source == NodeSource.INFERENCE_BASED
                }
            )

            # Enforce cumulative connectivity before plotting
            if self._enforce_cumulative_connectivity_fn:
                self._enforce_cumulative_connectivity_fn()

            # SAFE FILTERING: Only plot triplets where both endpoints are active
            triplets_for_plot = [
                (s, r, o)
                for (s, r, o) in triplets
                if s in active_node_names and o in active_node_names
            ]

            # Get edge activation scores
            edge_activation_scores = (
                self._get_edge_activation_scores_fn()
                if self._get_edge_activation_scores_fn
                else {}
            )

            saved_path = plot_amoc_triplets(
                triplets=triplets_for_plot,
                persona=self._persona,
                model_name=self._model_name,
                age=age_for_filename,
                blue_nodes=list(blue_nodes_combined) if blue_nodes_combined else None,
                output_dir=plot_dir,
                step_tag=(
                    f"sent{sentence_index+1}_{mode}"
                    if mode
                    else f"sent{sentence_index+1}"
                ),
                sentence_text=sentence_text,
                inactive_nodes=inactive_nodes,
                explicit_nodes=explicit_nodes_for_plot,
                ever_explicit_nodes=ever_explicit_nodes_for_plot,
                salient_nodes=salient_nodes,
                largest_component_only=largest_component_only,
                positions=self._viz_positions,
                active_edges=active_edges,
                edge_activation_scores=edge_activation_scores,
                layout_from_active_only=True,
                allow_multi_edges=self._allow_multi_edges,
                active_triplets_for_overlay=active_triplets_for_overlay,
                show_triplet_overlay=True,
                layout_depth=self._layout_depth,
                inferred_nodes=inferred_nodes_for_plot,
            )
            if triplets:
                logging.info(
                    "[Plot] Saved sentence %d graph to %s", sentence_index, saved_path
                )
            elif explicit_nodes:
                logging.info(
                    "[Plot] Saved sentence %d graph (explicit nodes only) to %s",
                    sentence_index,
                    saved_path,
                )
            else:
                logging.info(
                    "[Plot] Sentence %d graph empty (no explicit nodes, no edges)",
                    sentence_index,
                )
        except Exception:
            logging.error("Failed to plot graph snapshot", exc_info=True)

    def plot_sentence_views(
        self,
        sentence_idx: int,
        original_text: str,
        graphs_output_dir: Optional[str],
        highlight_nodes: Optional[Iterable[str]],
        inactive_nodes_for_plot: List[str],
        salient_nodes_for_plot: List[str],
        largest_component_only: bool,
        per_sentence_view,
        explicit_nodes_current_sentence: Set["Node"],
        reconstruct_semantic_triplets_fn: callable,
    ) -> None:
        # Active view - use per-sentence view for clean isolation
        if per_sentence_view is not None:
            active_nodes = set(per_sentence_view.explicit_nodes) | set(
                per_sentence_view.carryover_nodes
            )
        else:
            active_nodes = None

        explicit_nodes_for_plot = [
            node.get_text_representer()
            for node in explicit_nodes_current_sentence
            if node.get_text_representer()
        ]

        if per_sentence_view is not None:
            active_triplets = [
                (
                    e.source_node.get_text_representer(),
                    e.label,
                    e.dest_node.get_text_representer(),
                )
                for e in per_sentence_view.active_edges
            ]
        else:
            active_triplets = reconstruct_semantic_triplets_fn(only_active=True)

        # PROJECTION CONTINUITY + EXPLICIT FALLBACK
        if not active_triplets and explicit_nodes_for_plot:
            active_triplets = []

        # Store current projection snapshot for next iteration
        self._previous_active_triplets = list(active_triplets)

        property_nodes_for_plot = sorted(
            filter(
                None,
                {
                    node.get_text_representer()
                    for node in explicit_nodes_current_sentence
                    if node.node_type == NodeType.PROPERTY
                },
            )
        )

        # edges ARE the semantic triplets
        if per_sentence_view is not None:
            active_edge_pairs = {
                (
                    e.source_node.get_text_representer(),
                    e.dest_node.get_text_representer(),
                )
                for e in per_sentence_view.active_edges
            }
        else:
            snapshot_edges = [e for e in self._graph.edges if e.active]
            active_edge_pairs = {
                (
                    edge.source_node.get_text_representer(),
                    edge.dest_node.get_text_representer(),
                )
                for edge in snapshot_edges
            }

        # Plot active view
        self.plot_graph_snapshot_full(
            sentence_index=sentence_idx,
            sentence_text=original_text,
            output_dir=graphs_output_dir,
            highlight_nodes=highlight_nodes,
            inactive_nodes=inactive_nodes_for_plot,
            explicit_nodes=explicit_nodes_for_plot,
            salient_nodes=salient_nodes_for_plot,
            only_active=True,
            largest_component_only=largest_component_only,
            mode="sentence_active",
            triplets_override=active_triplets,
            active_edges=active_edge_pairs,
            active_triplets_for_overlay=active_triplets,
            property_nodes=property_nodes_for_plot,
        )

        # Cumulative memory view
        snapshot_edges = [e for e in self._graph.edges if e.active]
        cumulative_active_pairs = {
            (
                edge.source_node.get_text_representer(),
                edge.dest_node.get_text_representer(),
            )
            for edge in snapshot_edges
        }

        self.plot_graph_snapshot_full(
            sentence_index=sentence_idx,
            sentence_text=original_text,
            output_dir=graphs_output_dir,
            highlight_nodes=highlight_nodes,
            inactive_nodes=inactive_nodes_for_plot,
            explicit_nodes=explicit_nodes_for_plot,
            salient_nodes=salient_nodes_for_plot,
            only_active=False,
            largest_component_only=largest_component_only,
            mode="sentence_cumulative",
            triplets_override=reconstruct_semantic_triplets_fn(only_active=False),
            active_edges=cumulative_active_pairs,
            active_triplets_for_overlay=active_triplets,
            property_nodes=property_nodes_for_plot,
        )
