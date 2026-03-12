import os
import logging
from typing import List, Tuple, Dict, Optional, Any
from amoc.viz.graph_plots import plot_amoc_triplets


class ReverseGraphPlotter:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, "reverse_plots")
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_reverse_sequence(
        self,
        graph_states: List[Dict[str, Any]],
        base_kwargs: Dict[str, Any],
        positions: Dict[str, Tuple[float, float]],
        mode: str = "cumulative",
    ) -> List[str]:

        if len(graph_states) < 2:
            logging.warning("Need at least 2 states for reverse plotting")
            return []

        png_paths = []
        total_states = len(graph_states)

        # Create mode-specific subdirectory
        folder_names = {
            "cumulative": "cumulative",
            "paper": "paper",
        }
        mode_dir = os.path.join(self.output_dir, folder_names.get(mode, mode))
        os.makedirs(mode_dir, exist_ok=True)

        # Plot in reverse order (from last to first)
        for idx, state in enumerate(reversed(graph_states)):
            frame_num = total_states - idx
            logging.info(
                f"making reverse plot {idx+1}/{total_states} (original step {frame_num})"
            )

            # Merge base kwargs with state-specific kwargs
            frame_kwargs = base_kwargs.copy()
            frame_kwargs.update(state)

            # Get all nodes that appear in this state's triplets
            state_triplets = state.get("triplets", [])
            state_nodes = set()
            for s, r, o in state_triplets:
                state_nodes.add(s)
                state_nodes.add(o)

            # Also include explicit, salient, inactive
            if "explicit_nodes" in state:
                state_nodes.update(state["explicit_nodes"])
            if "salient_nodes" in state:
                state_nodes.update(state["salient_nodes"])
            if "inactive_nodes" in state:
                # Filter inactive nodes to only those that were ever in working memory
                # to prevent nodes from inactive edges (never in WM) appearing as danglers
                ever_in_wm = set(state.get("ever_in_wm", []))
                if ever_in_wm:
                    filtered_inactive = [
                        n for n in state["inactive_nodes"] if n in ever_in_wm
                    ]
                    state_nodes.update(filtered_inactive)
                else:
                    state_nodes.update(state["inactive_nodes"])

            # Filter positions to only include nodes that exist in this state
            filtered_positions = {
                node: pos for node, pos in positions.items() if node in state_nodes
            }

            frame_kwargs["positions"] = filtered_positions

            # Remove internal keys that plot_amoc_triplets doesn't accept
            frame_kwargs.pop("ever_in_wm", None)

            # Override output directory to our reverse plots folder
            original_step_tag = state.get("step_tag", f"step_{frame_num}")
            frame_kwargs["step_tag"] = f"reverse_{mode}_{original_step_tag}"
            frame_kwargs["output_dir"] = mode_dir

            # Generate the PNG
            try:
                png_path = plot_amoc_triplets(**frame_kwargs)
                png_paths.append(png_path)
                logging.info(f"saved reverse plot: {png_path}")
            except Exception as e:
                logging.error(
                    f"Failed to generate reverse plot for step {original_step_tag}: {e}"
                )
                continue

        return png_paths
