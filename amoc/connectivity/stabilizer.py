from typing import TYPE_CHECKING, Set, Optional, List
import networkx as nx
import logging
import re
import json

if TYPE_CHECKING:
    from amoc.core.graph import Graph
    from amoc.core.node import Node
    from amoc.core.edge import Edge
from amoc.prompts.amoc_prompts import FORCED_CONNECTIVITY_EDGE_PROMPT
from amoc.admission.triplet_validator import TripletValidator


class ConnectivityStabilizer:
    def __init__(
        self,
        graph_ref: "Graph",
        get_explicit_nodes: callable,
        get_carryover_nodes: callable,
        edge_visibility: int,
        llm_extractor=None,
        spacy_nlp=None,
    ):
        self._graph = graph_ref
        self._get_explicit_nodes = get_explicit_nodes
        self._get_carryover_nodes = get_carryover_nodes
        self._edge_visibility = edge_visibility
        self._llm = llm_extractor
        self._story_text: str = ""
        self._current_sentence_text: str = ""
        self._persona = ""
        self._validator = TripletValidator(
            linguistic_ops=None,
            extract_deterministic_fn=None,
            text_normalizer=None,
            client=llm_extractor,
            spacy_nlp=spacy_nlp,
        )

    def _validate_forced_relation(
        self, subj_text: str, relation: str, obj_text: str
    ) -> Optional[str]:
        if not self._validator.is_valid_relation_label(relation):
            logging.info(
                f"FORCED_EDGE rejected (invalid label): "
                f"({subj_text}, {relation}, {obj_text})"
            )
            return None

        result = self._validator.validate_triplet_relation(
            (subj_text, relation, obj_text)
        )

        if result["action"] in ("swap", "add_copula") and result.get(
            "corrected_triple"
        ):
            corrected_relation = result["corrected_triple"][1]
            logging.info(
                f"FORCED_EDGE corrected: ({subj_text}, {relation}, {obj_text}) "
                f"-> relation '{corrected_relation}'"
            )
            return corrected_relation

        if not result.get("valid", True):
            logging.info(
                f"FORCED_EDGE rejected ({result.get('reason')}): "
                f"({subj_text}, {relation}, {obj_text})"
            )
            return None

        return relation

    def set_context(self, story_text: str, current_sentence_text: str):
        self._story_text = story_text
        self._current_sentence_text = current_sentence_text

    def is_active_connected_wrapper(self) -> bool:
        required_nodes = self._get_explicit_nodes() | self._get_carryover_nodes()
        return self._graph.is_active_connected(required_nodes)

    def is_cumulative_connected_wrapper(self) -> bool:
        return self._graph.is_cumulative_connected()

    def run_connectivity_pipeline(
        self,
        prev_sentences: list,
        current_sentence_text: str,
        create_forced_edges_fn: callable,
    ) -> bool:
        explicit_nodes = self._get_explicit_nodes()
        carryover_nodes = self._get_carryover_nodes()
        required_nodes = explicit_nodes | carryover_nodes

        if self._graph.enforce_connectivity(required_nodes, allow_reactivation=True):
            if (
                self.is_active_connected_wrapper()
                and self.is_cumulative_connected_wrapper()
            ):
                return False

        for attempt in range(2):
            create_forced_edges_fn(
                story_context=(
                    " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                ),
                current_sentence=current_sentence_text,
                mode="active",
            )
            if self.is_active_connected_wrapper():
                break

        if (
            self.is_active_connected_wrapper()
            and self.is_cumulative_connected_wrapper()
        ):
            return False

        self.apply_relates_to_fallback(required_nodes)

        if not self.is_active_connected_wrapper():
            logging.error("Active graph disconnected after all repairs")
            return True

        if not self.is_cumulative_connected_wrapper():
            self.connect_cumulative_components()
            if not self.is_cumulative_connected_wrapper():
                logging.error("Cumulative graph disconnected after all repairs")
                return True

        return False

    def node_sort_key(self, node, edge_pool):
        degree = sum(
            1 for e in edge_pool if e.source_node == node or e.dest_node == node
        )
        return (-degree, node.get_text_representer())

    def apply_relates_to_fallback(self, required_nodes: set) -> None:
        components, _ = self._graph.get_disconnected_components_wrapper(required_nodes)

        all_required_nodes = set(required_nodes)
        covered_nodes = set().union(*components) if components else set()
        for node in all_required_nodes - covered_nodes:
            components.append({node})

        if len(components) <= 1:
            return

        components = sorted(components, key=len, reverse=True)
        largest_component = set(components[0])

        active_edge_pool = [e for e in self._graph.edges if e.active]
        backbone_node = max(
            largest_component,
            key=lambda n: (
                sum(1 for e in n.edges if e.active),
                len(n.edges),
                n.get_text_representer(),
            ),
        )

        for comp in sorted(components[1:], key=len):
            comp_set = set(comp)
            if not (comp_set & required_nodes):
                continue

            explicit_nodes = self._get_explicit_nodes()
            explicit_in_comp = sorted(
                [n for n in comp_set if n in explicit_nodes],
                key=lambda n: n.get_text_representer(),
            )
            if explicit_in_comp:
                node_small = explicit_in_comp[0]
            else:
                node_small = min(
                    comp_set, key=lambda n: self.node_sort_key(n, active_edge_pool)
                )
            edge = self._graph.add_edge(
                node_small,
                backbone_node,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )

            if edge:
                edge.mark_as_current_sentence(reset_score=True)
                largest_component.update(comp_set)

    def connect_cumulative_components(self) -> None:
        G_full = self._graph.to_networkx()

        if G_full.number_of_nodes() <= 1:
            return

        components = list(nx.connected_components(G_full))
        if len(components) <= 1:
            return

        components = sorted(components, key=len, reverse=True)
        largest = set(components[0])

        node_large = min(largest, key=lambda n: n.get_text_representer())

        for comp in sorted(components[1:], key=len):
            node_small = min(comp, key=lambda n: n.get_text_representer())
            edge = self._graph.add_edge(
                node_small,
                node_large,
                "relates_to",
                self._edge_visibility,
                persona_influenced=False,
                inferred=False,
            )
            if edge:
                edge.mark_as_current_sentence(reset_score=True)
                largest.update(comp)

    def get_nodes_with_active_edges(self) -> set:
        nodes = set()
        for edge in self._graph.edges:
            if edge.active:
                nodes.add(edge.source_node)
                nodes.add(edge.dest_node)
        return nodes

    def validate_active_connectivity(self) -> bool:
        if not self.is_active_connected_wrapper():
            return False

        explicit_nodes = self._get_explicit_nodes()
        for node in explicit_nodes:
            if node not in self._graph.nodes:
                return False
            has_active_edge = any(
                e.active and (e.source_node == node or e.dest_node == node)
                for e in self._graph.edges
            )
            if not has_active_edge and len(explicit_nodes) > 1:
                return False
        return True

    def repair_dangling_nodes(
        self,
        per_sentence_view,
        prev_sentences: list,
        normalize_edge_label_fn: callable,
        persona: str = "",
    ) -> bool:
        if per_sentence_view is None:
            return False

        active_nodes = set(per_sentence_view.explicit_nodes) | set(
            per_sentence_view.carryover_nodes
        )
        dangling_nodes = []
        for node in active_nodes:
            has_edge = any(
                e.source_node == node or e.dest_node == node
                for e in per_sentence_view.active_edges
            )
            if not has_edge:
                dangling_nodes.append(node)

        if not dangling_nodes:
            return False

        any_repair_failed = False
        for node in dangling_nodes:
            repair_success = False

            degree_sorted = sorted(
                per_sentence_view.active_nodes,
                key=lambda n: sum(
                    1
                    for e in per_sentence_view.active_edges
                    if e.source_node == n or e.dest_node == n
                ),
                reverse=True,
            )

            anchor = None
            for candidate in degree_sorted:
                if candidate != node:
                    anchor = candidate
                    break

            if anchor is None:
                any_repair_failed = True
                continue

            for _ in range(2):
                result = self._llm.get_forced_connectivity_edge_label(
                    node_a=node.get_text_representer(),
                    node_b=anchor.get_text_representer(),
                    story_context=(
                        " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                    ),
                    current_sentence=self._current_sentence_text,
                    persona=persona,
                )

                relation = result.get("label") if isinstance(result, dict) else result
                if not relation:
                    continue

                relation = normalize_edge_label_fn(relation)
                if not relation:
                    continue
                relation = self._validate_forced_relation(
                    node.get_text_representer(), relation, anchor.get_text_representer()
                )
                if not relation:
                    continue
                edge = self._graph.add_edge(
                    node,
                    anchor,
                    relation,
                    self._edge_visibility,
                    inferred=True,
                )

                if edge:
                    edge.mark_as_current_sentence(reset_score=True)
                    repair_success = True
                    break

            if not repair_success:
                any_repair_failed = True

        return any_repair_failed

    def repair_connectivity_callback(
        self,
        components,
        active_nodes,
        active_edges,
        sentence_index,
        temperature: float = 0.3,
        forced_pair=None,
    ):
        if forced_pair is not None:
            representative, anchor_node = forced_pair
            components = [{representative}, {anchor_node}]

        if not components or len(components) <= 1:
            return None

        sorted_components = sorted(components, key=len, reverse=True)
        main_component = sorted_components[0]

        anchor_node = min(
            [n for n in main_component if n in active_nodes],
            key=lambda n: self.node_sort_key(n, active_edges),
            default=None,
        )
        if anchor_node is None:
            return None

        edges_created = set()

        for comp in sorted_components[1:]:
            candidates = [n for n in comp if n in active_nodes]
            if not candidates:
                continue
            representative = min(
                candidates, key=lambda n: self.node_sort_key(n, active_edges)
            )

            prompt_text = FORCED_CONNECTIVITY_EDGE_PROMPT.format(
                node_a=representative.get_text_representer(),
                node_b=anchor_node.get_text_representer(),
                story_context=self._story_text[:1500],
                current_sentence=self._current_sentence_text,
            )

            try:
                response = self._llm.generate_raw(
                    prompt_text=prompt_text,
                    temperature=temperature,
                )

                if not response:
                    continue

                response = response.strip()

                data = json.loads(response)
                label = data.get("label")

            except (json.JSONDecodeError, Exception) as e:
                logging.warning("Connectivity repair failed: %s", str(e))
                continue

            if not label:
                continue

            edge = self._graph.add_edge(
                source_node=representative,
                dest_node=anchor_node,
                label=label.strip().lower(),
                edge_visibility=self._edge_visibility,
                created_at_sentence=sentence_index,
                inferred=True,
            )

            if edge:
                edges_created.add(edge)

        return edges_created if edges_created else None

    def warn_if_cumulative_disconnected(self) -> None:
        if not self.is_cumulative_connected_wrapper():
            logging.warning("Cumulative graph disconnected - plots may show fragments")

    def run_repair_pipeline(
        self,
        per_sentence_view,
        prev_sentences: list,
        current_sentence_text: str,
        normalize_edge_label_fn: callable,
        create_forced_edges_fn: callable,
        persona: str = "",
    ) -> None:
        self._persona = persona

        self.connect_isolated_explicit_node(
            per_sentence_view,
            prev_sentences,
            current_sentence_text,
            normalize_edge_label_fn,
        )

        self.repair_isolated_explicit_nodes(
            per_sentence_view, current_sentence_text, normalize_edge_label_fn
        )

        required_nodes = set(per_sentence_view.explicit_nodes) | set(
            per_sentence_view.carryover_nodes
        )
        self._graph.enforce_connectivity(required_nodes, allow_reactivation=True)

        self.create_forced_edges_via_llm(
            prev_sentences, current_sentence_text, create_forced_edges_fn
        )

        self.apply_relates_to_fallback(required_nodes)

        self.connect_cumulative_components()

        self.repair_dangling_nodes(
            per_sentence_view, prev_sentences, normalize_edge_label_fn, persona
        )

    def connect_isolated_explicit_node(
        self,
        per_sentence_view,
        prev_sentences: list,
        current_sentence_text: str,
        normalize_edge_label_fn: callable,
    ) -> None:
        explicit_nodes = per_sentence_view.explicit_nodes
        if len(explicit_nodes) != 1:
            return
        node = next(iter(explicit_nodes))
        active_nodes = self.get_nodes_with_active_edges()
        if node in active_nodes or not active_nodes:
            return

        anchor = max(
            active_nodes,
            key=lambda n: sum(
                1
                for e in self._graph.edges
                if e.active and (e.source_node == n or e.dest_node == n)
            ),
        )
        if anchor == node:
            return

        story_context = " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""

        result = self._llm.get_forced_connectivity_edge_label(
            node_a=node.get_text_representer(),
            node_b=anchor.get_text_representer(),
            story_context=story_context,
            current_sentence=current_sentence_text,
            persona=self._persona,
        )
        relation = result.get("label") if isinstance(result, dict) else result
        if not relation:
            relation = "relates_to"

        relation = normalize_edge_label_fn(relation)
        if not relation:
            return
        relation = self._validate_forced_relation(
            node.get_text_representer(), relation, anchor.get_text_representer()
        )
        if not relation:
            return

        edge = self._graph.add_edge(
            node,
            anchor,
            relation,
            self._edge_visibility,
            inferred=True,
        )
        if edge:
            edge.mark_as_current_sentence(reset_score=True)

    def repair_isolated_explicit_nodes(
        self,
        per_sentence_view,
        current_sentence_text: str,
        normalize_edge_label_fn: callable,
    ) -> None:
        active_nodes = self.get_nodes_with_active_edges()
        if not active_nodes:
            return

        for node in per_sentence_view.explicit_nodes:
            if any(
                e.source_node == node or e.dest_node == node
                for e in per_sentence_view.active_edges
            ):
                continue

            anchor = max(
                active_nodes,
                key=lambda n: sum(
                    1
                    for e in self._graph.edges
                    if e.active and (e.source_node == n or e.dest_node == n)
                ),
            )
            if anchor == node:
                continue

            story_context = ""

            result = self._llm.get_forced_connectivity_edge_label(
                node_a=node.get_text_representer(),
                node_b=anchor.get_text_representer(),
                story_context=story_context,
                current_sentence=current_sentence_text,
                persona=self._persona,
            )
            relation = result.get("label") if isinstance(result, dict) else result
            if not relation:
                continue

            relation = normalize_edge_label_fn(relation)
            if not relation:
                continue
            relation = self._validate_forced_relation(
                node.get_text_representer(), relation, anchor.get_text_representer()
            )
            if not relation:
                continue

            edge = self._graph.add_edge(
                node,
                anchor,
                relation,
                self._edge_visibility,
                inferred=True,
            )
            if edge:
                edge.mark_as_current_sentence(reset_score=True)

    def create_forced_edges_via_llm(
        self,
        prev_sentences: list,
        current_sentence_text: str,
        create_forced_edges_fn: callable,
    ) -> None:
        for attempt in range(2):
            create_forced_edges_fn(
                story_context=(
                    " ".join(prev_sentences[:-1]) if len(prev_sentences) > 1 else ""
                ),
                current_sentence=current_sentence_text,
                mode="active",
            )
            if self.is_active_connected_wrapper():
                break
