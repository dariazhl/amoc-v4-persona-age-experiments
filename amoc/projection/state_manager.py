import logging
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from amoc.core.node import Node, NodeSource
from amoc.graph_views.per_sentence import PerSentenceGraph


class ProjectionStateManager:

    def __init__(
        self,
        graph_ref,
        max_distance: int,
        debug: bool = False,
    ):
        self.graph = graph_ref
        self.max_distance = max_distance
        self.debug = debug

        self.record_activation_matrix_wrapper_fn = None

        self._prev_active_nodes_for_plot: Set[Node] = set()
        self._cumulative_deactivated_nodes_for_plot: Set[Node] = set()
        self._recently_deactivated_nodes_for_inference: Set[Node] = set()

    def set_callbacks(
        self,
        record_sentence_activation_fn,
    ):
        self.record_activation_matrix_wrapper_fn = record_sentence_activation_fn

    def reset_state(self):
        self._prev_active_nodes_for_plot = set()
        self._cumulative_deactivated_nodes_for_plot = set()
        self._recently_deactivated_nodes_for_inference = set()

    def get_recently_deactivated_nodes(self) -> Set[Node]:
        return self._recently_deactivated_nodes_for_inference

    def compute_newly_inferred_nodes(
        self, nodes_before_sentence: Set[Node]
    ) -> Set[Node]:
        return {
            n
            for n in (set(self.graph.nodes) - nodes_before_sentence)
            if n.node_source == NodeSource.INFERENCE_BASED
        }

    # update internal state to track which nodes have become faded in this sentence compared to the previous one
    def update_projection_state(
        self,
        sentence_id: int,
        sentence_index: int,
        newly_inferred_nodes: Set[Node],
        per_sentence_view,
        explicit_nodes_current_sentence: Set[Node],
        persona: str,
    ) -> Tuple[Set[Node], List[str], List[str], List[str]]:
        # record activation
        if self.record_activation_matrix_wrapper_fn is not None:
            self.record_activation_matrix_wrapper_fn(
                sentence_id=sentence_id,
                explicit_nodes=list(explicit_nodes_current_sentence),
                newly_inferred_nodes=newly_inferred_nodes,
            )
        # find active nodes
        current_active_nodes = set(per_sentence_view.explicit_nodes) | set(
            per_sentence_view.carryover_nodes
        )
        # check for disconnection - it should not happen at this point, just checking
        if (
            per_sentence_view is not None
            and not per_sentence_view.is_empty()
            and not per_sentence_view.is_connected()
        ):
            logging.error(
                "Per-sentence graph disconnected at sentence %s for persona '%s'",
                sentence_id,
                persona,
            )
        # find recently deactivated nodes
        if sentence_index == 0:
            recently_deactivated_nodes: Set[Node] = set()
        else:
            appeared = current_active_nodes - self._prev_active_nodes_for_plot
            gone = self._prev_active_nodes_for_plot - current_active_nodes

            self._cumulative_deactivated_nodes_for_plot.update(gone)
            self._cumulative_deactivated_nodes_for_plot.difference_update(
                current_active_nodes
            )

            recently_deactivated_nodes = set(gone)
        # prepare plotting lists for explicit, carryover, inactive nodes
        if per_sentence_view is not None:
            explicit_nodes_for_plot = sorted(
                filter(
                    None,
                    {
                        n.get_text_representer()
                        for n in per_sentence_view.explicit_nodes
                    },
                )
            )

            salient_nodes_for_plot = sorted(
                filter(
                    None,
                    {
                        n.get_text_representer()
                        for n in per_sentence_view.carryover_nodes
                    },
                )
            )

            all_nodes = {
                n.get_text_representer()
                for n in self.graph.nodes
                if n.get_text_representer()
            }

            active_nodes = set(explicit_nodes_for_plot) | set(salient_nodes_for_plot)
            inactive_nodes_for_plot = sorted(all_nodes - active_nodes)

        else:
            explicit_nodes_for_plot = []
            salient_nodes_for_plot = []
            inactive_nodes_for_plot = []

        self._recently_deactivated_nodes_for_inference = recently_deactivated_nodes

        self._prev_active_nodes_for_plot = current_active_nodes

        return (
            recently_deactivated_nodes,
            explicit_nodes_for_plot,
            salient_nodes_for_plot,
            inactive_nodes_for_plot,
        )

    def build_projection(
        self,
        sentence_id: int,
        per_sentence_view: Optional["PerSentenceGraph"],
        explicit_nodes_current_sentence: Set[Node],
        previous_active_triplets: List,
    ) -> Optional["PerSentenceGraph"]:

        explicit_nodes_strict = list(explicit_nodes_current_sentence)

        if (
            per_sentence_view is not None
            and per_sentence_view.is_empty()
            and previous_active_triplets
        ):
            logging.debug("Per-sentence view empty — preserving previous projection.")

        if self.debug and per_sentence_view is not None:
            logging.info(
                "Per-sentence view for sentence %d: "
                "%d explicit, %d carry-over, %d active edges, connected=%s",
                sentence_id,
                len(per_sentence_view.explicit_nodes),
                len(per_sentence_view.carryover_nodes),
                len(per_sentence_view.active_edges),
                per_sentence_view.is_connected(),
            )

        return per_sentence_view
