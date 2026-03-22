import logging
from typing import TYPE_CHECKING, List, Tuple, Set, Optional, Dict
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge

__all__ = [
    "EdgeRecord",
    "TripletRecorderV2",
    "graph_edges_to_triplets",
    "get_edge_activation_scores",
]


@dataclass
class EdgeRecord:
    original_index: int = -1
    age_refined: int = -1
    regime: str = ""
    persona_text: str = ""
    model_name: str = ""
    sentence_idx: int = -1
    sentence_text: str = ""
    subject: str = ""
    relation: str = ""
    object: str = ""
    subject_source: str = ""
    object_source: str = ""
    triplet_source: str = ""
    edge_visibility: int = 0
    explicit_carryover: str = ""
    decay_explanation: str = ""


class TripletRecorderV2:
    def __init__(
        self,
        graph_ref: "Graph",
        triplet_intro_ref: Dict[Tuple[str, str, str], int],
    ):
        self._graph = graph_ref
        self._triplet_intro = triplet_intro_ref
        self._sentence_records: List[List[EdgeRecord]] = []
        self._current_decay_decisions: Dict[Tuple[str, str, str], str] = {}

        self._original_index: int = -1
        self._age_refined: int = -1
        self._regime: str = ""
        self._persona_text: str = ""
        self._model_name: str = ""

    def set_metadata(
        self,
        original_index: int,
        age_refined: int,
        regime: str,
        persona_text: str,
        model_name: str,
    ) -> None:
        self._original_index = original_index
        self._age_refined = age_refined
        self._regime = regime
        self._persona_text = persona_text
        self._model_name = model_name

    def reset(self) -> None:
        self._sentence_records.clear()
        self._current_decay_decisions.clear()

    def set_decay_decisions(self, decisions: List[Tuple[Tuple[str, str, str], str]]) -> None:
        self._current_decay_decisions.clear()
        for triplet, reasoning in decisions:
            self._current_decay_decisions[triplet] = reasoning

    def capture_sentence_edges(
        self,
        sentence_index: int,
        sentence_text: str,
        explicit_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
    ) -> None:
        records = []

        for edge in self._graph.edges:
            if not edge.label or not str(edge.label).strip():
                continue
            if edge.source_node == edge.dest_node:
                continue

            src = edge.source_node
            dst = edge.dest_node

            subject = src.get_text_representer()
            obj = dst.get_text_representer()
            relation = edge.label

            triplet = (subject, relation, obj)

            src_in_explicit = src in explicit_nodes
            dst_in_explicit = dst in explicit_nodes

            if not edge.active:
                explicit_carryover = "inactive"
            elif src_in_explicit or dst_in_explicit:
                explicit_carryover = "explicit"
            else:
                explicit_carryover = "carryover"

            decay_reasoning = ""
            if explicit_carryover in ("carryover", "inactive"):
                decay_reasoning = self._current_decay_decisions.get(triplet, "")

            record = EdgeRecord(
                original_index=self._original_index,
                age_refined=self._age_refined,
                regime=self._regime,
                persona_text=self._persona_text,
                model_name=self._model_name,
                sentence_idx=sentence_index,
                sentence_text=sentence_text,
                subject=subject,
                relation=relation,
                object=obj,
                subject_source=src.node_source.name,
                object_source=dst.node_source.name,
                triplet_source="INFERRED" if (src.node_source.name == "INFERENCE_BASED" and dst.node_source.name == "INFERENCE_BASED") else "TEXT_BASED",
                edge_visibility=edge.visibility_score,
                explicit_carryover=explicit_carryover,
                decay_explanation=decay_reasoning,
            )
            records.append(record)

        self._sentence_records.append(records)

    def get_all_records(self) -> List[EdgeRecord]:
        all_records = []
        for sentence_records in self._sentence_records:
            all_records.extend(sentence_records)
        return all_records

    def get_sentence_records(self) -> List[List[EdgeRecord]]:
        return self._sentence_records


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
