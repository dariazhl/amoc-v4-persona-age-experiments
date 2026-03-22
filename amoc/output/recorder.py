import logging
from typing import TYPE_CHECKING, List, Tuple, Set, Optional, Dict
from amoc.output.models import DecayDecision, SentenceTripletRecord

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge

# Re-export for callers that import from this module
__all__ = [
    "DecayDecision",
    "SentenceTripletRecord",
    "TripletRecorder",
    "graph_edges_to_triplets",
    "get_edge_activation_scores",
]


class TripletRecorder:
    def __init__(
        self,
        graph_ref: "Graph",
        triplet_intro_ref: Dict[Tuple[str, str, str], int],
    ):
        self._graph = graph_ref
        self._triplet_intro = triplet_intro_ref
        self._sentence_records: List[SentenceTripletRecord] = []

    def reset(self) -> None:
        self._sentence_records.clear()

    def get_sentence_records(self) -> List[SentenceTripletRecord]:
        return self._sentence_records

    def capture_sentence_triplets(
        self,
        sentence_index: int,
        sentence_text: str,
        explicit_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
    ) -> None:
        from amoc.core.node import NodeSource

        explicit_text = []
        explicit_inferred = []
        carryover_text = []
        carryover_inferred = []
        inactive_text = []
        inactive_inferred = []

        for edge in self._graph.edges:
            # Skip empty labels
            if not edge.label or not str(edge.label).strip():
                continue
            # Skip self-loops
            if edge.source_node == edge.dest_node:
                continue

            src = edge.source_node
            dst = edge.dest_node
            triplet = (
                src.get_text_representer(),
                edge.label,
                dst.get_text_representer(),
            )

            src_is_explicit = src in explicit_nodes
            dst_is_explicit = dst in explicit_nodes
            src_is_carryover = src in carryover_nodes
            dst_is_carryover = dst in carryover_nodes

            is_inferred = (
                src.node_source == NodeSource.INFERENCE_BASED
                and dst.node_source == NodeSource.INFERENCE_BASED
            )

            # Determine state
            if not edge.active:
                if not (
                    src_is_explicit
                    or dst_is_explicit
                    or src_is_carryover
                    or dst_is_carryover
                ):
                    continue
                if is_inferred:
                    inactive_inferred.append(triplet)
                else:
                    inactive_text.append(triplet)
            elif src_is_explicit or dst_is_explicit:
                if is_inferred:
                    explicit_inferred.append(triplet)
                else:
                    explicit_text.append(triplet)
            elif src_is_carryover and dst_is_carryover:
                if is_inferred:
                    carryover_inferred.append(triplet)
                else:
                    carryover_text.append(triplet)
            elif src_is_carryover or dst_is_carryover:
                if is_inferred:
                    carryover_inferred.append(triplet)
                else:
                    carryover_text.append(triplet)

        record = SentenceTripletRecord(
            sentence_index=sentence_index,
            sentence_text=sentence_text,
            explicit_text_triplets=explicit_text,
            explicit_inferred_triplets=explicit_inferred,
            carryover_text_triplets=carryover_text,
            carryover_inferred_triplets=carryover_inferred,
            inactive_text_triplets=inactive_text,
            inactive_inferred_triplets=inactive_inferred,
        )
        self._sentence_records.append(record)

    def record_decay_decisions(
        self, sentence_index: int, decisions: List[DecayDecision]
    ) -> None:
        for record in self._sentence_records:
            if record.sentence_index == sentence_index:
                record.decay_decisions = decisions
                return
        logging.warning(
            f"record_decay_decisions: no record for sentence {sentence_index}"
        )


# ── Standalone utilities (used by GraphPlotter via callbacks) ────────────────


def graph_edges_to_triplets(
    graph: "Graph", only_active: bool = False
) -> List[Tuple[str, str, str]]:
    triplets = []
    for edge in graph.edges:
        if only_active and not edge.active:
            continue
        if not edge.label or not str(edge.label).strip():
            continue
        if edge.source_node == edge.dest_node:
            continue
        triplets.append(
            (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
        )
    return triplets


def get_edge_activation_scores(
    graph: "Graph",
) -> Dict[Tuple, int]:
    scores = {}
    for edge in graph.edges:
        key = (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )
        scores[key] = edge.visibility_score
        scores[(key[0], key[1])] = edge.visibility_score
    return scores
