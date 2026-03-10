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
        mode_dir = os.path.join(self.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # Plot in reverse order (from last to first)
        for idx, state in enumerate(reversed(graph_states)):
            frame_num = total_states - idx
            logging.info(
                f"Generating reverse plot {idx+1}/{total_states} (Original Step {frame_num})"
            )

            # Merge base kwargs with state-specific kwargs
            frame_kwargs = base_kwargs.copy()
            frame_kwargs.update(state)

            # Use frozen positions
            frame_kwargs["positions"] = positions.copy()

            # Override output directory to our reverse plots folder
            original_step_tag = state.get("step_tag", f"step_{frame_num}")
            frame_kwargs["step_tag"] = f"reverse_{mode}_{original_step_tag}"
            frame_kwargs["output_dir"] = mode_dir

            # Generate the PNG
            try:
                png_path = plot_amoc_triplets(**frame_kwargs)
                png_paths.append(png_path)
                logging.info(f"Generated reverse plot: {png_path}")
            except Exception as e:
                logging.error(
                    f"Failed to generate reverse plot for step {original_step_tag}: {e}"
                )
                continue

        return png_paths
