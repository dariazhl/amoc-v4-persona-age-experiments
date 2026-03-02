from typing import TYPE_CHECKING, Optional, List, Tuple
import logging
import networkx as nx

from amoc.graph.graph import Graph
from amoc.graph.node import Node, NodeType, NodeSource
from amoc.graph.edge import Edge
from amoc.nlp.spacy_utils import are_semantically_equivalent, get_semantic_class
from amoc.pipeline.text_filter_ops import TextFilterOps

if TYPE_CHECKING:
    from amoc.pipeline.core import AMoCv4


class EdgeOps:

    def __init__(
        self,
        graph_ref: "Graph",
        client_ref,
        spacy_nlp,
        get_explicit_nodes: callable,
        get_carryover_nodes: callable,
        get_attachable_nodes: callable,
        edge_visibility: int,
        allow_multi_edges: bool,
        debug: bool = False,
    ):
        self._graph = graph_ref
        self._client = client_ref
        self._spacy_nlp = spacy_nlp
        self._get_explicit_nodes = get_explicit_nodes
        self._get_carryover_nodes = get_carryover_nodes
        self._get_attachable_nodes = get_attachable_nodes
        self._edge_visibility = edge_visibility
        self._allow_multi_edges = allow_multi_edges
        self._debug = debug
        self._triplet_intro = {}
        self._persistent_is_edges = set()
        self._current_sentence_index = None
        self._normalize_endpoint_text_fn: Optional[callable] = None
        self._normalize_edge_label_fn: Optional[callable] = None
        self._is_valid_relation_label_fn: Optional[callable] = None
        self._find_node_by_text_fn: Optional[callable] = None
        self._add_edge_fn: Optional[callable] = None
        self._classify_relation_fn: Optional[callable] = None
        self._persona: str = ""

    def set_inference_callbacks(
        self,
        normalize_endpoint_text_fn: callable,
        normalize_edge_label_fn: callable,
        is_valid_relation_label_fn: callable,
        find_node_by_text_fn: callable,
        add_edge_fn: callable,
        classify_relation_fn: callable,
        persona: str,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self._find_node_by_text_fn = find_node_by_text_fn
        self._add_edge_fn = add_edge_fn
        self._classify_relation_fn = classify_relation_fn
        self._persona = persona

    def set_state_refs(
        self,
        triplet_intro: dict,
        persistent_is_edges: set,
    ):
        self._triplet_intro = triplet_intro
        self._persistent_is_edges = persistent_is_edges

    def configure_with_core(self, core: "AMoCv4") -> None:
        self.set_inference_callbacks(
            normalize_endpoint_text_fn=core._normalize_endpoint_text,
            normalize_edge_label_fn=core._normalize_edge_label,
            is_valid_relation_label_fn=core._is_valid_relation_label,
            find_node_by_text_fn=lambda t, c: core._node_ops.find_node_by_text(t, c),
            add_edge_fn=core._add_edge,
            classify_relation_fn=core._classify_relation,
            persona=core.persona,
        )
        self.set_state_refs(
            triplet_intro=core._triplet_intro,
            persistent_is_edges=core._persistent_is_edges,
        )

    def set_current_sentence(self, idx: int):
        self._current_sentence_index = idx

    def edge_key(self, edge: "Edge") -> Tuple[str, str, str]:
        return (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )

    def get_existing_edge_between_nodes(
        self, source_node: "Node", dest_node: "Node"
    ) -> Optional["Edge"]:
        for edge in self._graph.edges:
            if edge.source_node == source_node and edge.dest_node == dest_node:
                return edge
        return None

    def has_edge_between(
        self, a: "Node", b: "Node", relation_lemma: Optional[str] = None
    ) -> bool:
        for edge in a.edges:
            other = edge.dest_node if edge.source_node == a else edge.source_node
            if other == b:
                if relation_lemma is None:
                    return True
                if relation_lemma in edge.label.lower():
                    return True
        return False

    def create_edge_with_event_mediation(
        self,
        source_node: "Node",
        dest_node: "Node",
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
        relation_class=None,
        justification=None,
    ) -> Optional["Edge"]:
        sentence_idx = (
            created_at_sentence
            if created_at_sentence is not None
            else self._current_sentence_index
        )

        label = TextFilterOps.canonicalize_relation_label(label)
        if not label:
            return None

        return self._graph.add_edge(
            source_node,
            dest_node,
            label,
            edge_forget,
            created_at_sentence=sentence_idx,
            relation_class=relation_class,
            justification=justification,
        )

    def add_edge(
        self,
        source_node: "Node",
        dest_node: "Node",
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
        bypass_attachment_constraint: bool = False,
        relation_class=None,
        justification=None,
        persona_influenced: bool = False,
    ) -> Optional["Edge"]:
        # Reject self-loops - S-V-O triplets require distinct subject and object
        if source_node == dest_node:
            return None

        attachable = (
            self._get_attachable_nodes() if self._get_attachable_nodes else set()
        )

        if not bypass_attachment_constraint:
            if source_node not in attachable and dest_node not in attachable:
                return None

        dest_text = dest_node.get_text_representer()
        if dest_text and dest_text.strip().lower() == label.strip().lower():
            logging.warning(
                "Rejected duplicate verb-object edge: %s --%s--> %s",
                source_node.get_text_representer(),
                label,
                dest_text,
            )
            return None

        label = TextFilterOps.canonicalize_relation_label(label)
        if not label:
            return None

        use_sentence = (
            created_at_sentence
            if created_at_sentence is not None
            else self._current_sentence_index
        )

        if not self._allow_multi_edges:
            existing_edge = self.get_existing_edge_between_nodes(source_node, dest_node)
            if existing_edge is not None:
                old_label = existing_edge.label

                if are_semantically_equivalent(old_label, label):
                    existing_edge.label = label
                    existing_edge.visibility_score = edge_forget
                    existing_edge.created_at_sentence = use_sentence
                    existing_edge.mark_as_asserted(reset_score=True)

                    if self._debug:
                        logging.debug(
                            "REPLACED equivalent edge: %s --%s--> %s (was: %s, class: %s)",
                            source_node.get_text_representer(),
                            label,
                            dest_node.get_text_representer(),
                            old_label,
                            get_semantic_class(
                                label.split("_")[0] if "_" in label else label
                            ),
                        )

                    trip_id = (
                        existing_edge.source_node.get_text_representer(),
                        existing_edge.label,
                        existing_edge.dest_node.get_text_representer(),
                    )
                    if trip_id not in self._triplet_intro:
                        self._triplet_intro[trip_id] = (
                            use_sentence if use_sentence is not None else -1
                        )

                    if label.strip().lower() == "is":
                        self._persistent_is_edges.add(trip_id)

                    return existing_edge
                else:
                    self._graph.remove_edge(existing_edge)

        if (
            label == source_node.get_text_representer()
            or label == dest_node.get_text_representer()
        ):
            return None

        edge = self.create_edge_with_event_mediation(
            source_node,
            dest_node,
            label,
            edge_forget,
            created_at_sentence=use_sentence,
            relation_class=relation_class,
            justification=justification,
        )

        if edge:
            if use_sentence == self._current_sentence_index:
                edge.mark_as_asserted(reset_score=True)

            trip_id = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            if trip_id not in self._triplet_intro:
                self._triplet_intro[trip_id] = (
                    use_sentence if use_sentence is not None else -1
                )

            if label.strip().lower() == "is":
                trip_id = (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                )
                self._persistent_is_edges.add(trip_id)

        return edge

    def llm_generate_relation(
        self,
        node_a: "Node",
        node_b: "Node",
        story_context: str = "",
        current_sentence: str = "",
        persona: str = "",
    ) -> Optional[str]:
        try:
            result = self._client.get_forced_connectivity_edge_label(
                node_a=node_a.get_text_representer(),
                node_b=node_b.get_text_representer(),
                story_context=story_context,
                current_sentence=current_sentence,
                persona=persona,
            )

            if isinstance(result, dict):
                label = result.get("label")
            else:
                label = result

            return label if isinstance(label, str) and label.strip() else None

        except Exception:
            return None

    def create_forced_connectivity_edges(
        self,
        story_context: Optional[str] = None,
        current_sentence: Optional[str] = None,
        mode: str = "active",
        persona: str = "",
        normalize_edge_label_fn: callable = None,
    ) -> List["Edge"]:
        if mode == "active":
            protected_nodes = self._get_explicit_nodes() | self._get_carryover_nodes()
        else:
            protected_nodes = set(self._graph.nodes)

        components, _ = self._graph.get_disconnected_components(protected_nodes)

        if len(components) <= 1:
            return []

        if mode == "active":
            components = sorted(
                components,
                key=lambda c: len(set(c) & protected_nodes),
                reverse=True,
            )
        else:
            components = sorted(components, key=len, reverse=True)

        backbone = set(components[0])
        forced_edges = []

        for comp in components[1:]:
            node_a = next(iter(backbone))
            node_b = next(iter(comp))

            # Try LLM repair (2 attempts)
            for _ in range(2):
                result = self._client.get_forced_connectivity_edge_label(
                    node_a=node_a.get_text_representer(),
                    node_b=node_b.get_text_representer(),
                    story_context=story_context,
                    current_sentence=current_sentence,
                    persona=persona,
                )

                relation = result.get("label") if isinstance(result, dict) else result
                if not relation:
                    continue

                if normalize_edge_label_fn:
                    relation = normalize_edge_label_fn(relation)
                    if not relation:
                        continue

                edge = self._graph.add_edge(
                    node_a,
                    node_b,
                    relation,
                    self._edge_visibility,
                    inferred=True,
                )

                if edge:
                    forced_edges.append(edge)
                    backbone.update(comp)
                    break

        return forced_edges

    def record_edge_in_graphs(
        self,
        edge: "Edge",
        sentence_idx: Optional[int],
        cumulative_graph: nx.MultiDiGraph,
        active_graph: nx.MultiDiGraph,
        cumulative_triplet_records: list,
    ) -> None:
        u, v, lbl = self.edge_key(edge)

        if not lbl or not lbl.strip():
            return

        introduced = self._triplet_intro.get((u, lbl, v))
        if introduced is None:
            introduced = (
                edge.created_at_sentence if edge.created_at_sentence is not None else -1
            )
        self._triplet_intro[(u, lbl, v)] = int(introduced)
        edge_key = f"{lbl}__introduced_{introduced}"

        if sentence_idx is not None:
            cumulative_triplet_records.append(
                {
                    "subject": u,
                    "relation": lbl,
                    "object": v,
                    "sentence_idx": sentence_idx,
                    "introduced_at": introduced,
                    "active": edge.active,
                    "visibility_score": edge.visibility_score,
                }
            )

        if not cumulative_graph.has_edge(u, v, key=edge_key):
            cumulative_graph.add_edge(u, v, key=edge_key, label=lbl)

        if edge.active:
            if not active_graph.has_edge(u, v, key=edge_key):
                active_graph.add_edge(u, v, key=edge_key, label=lbl)
        else:
            if active_graph.has_edge(u, v, key=edge_key):
                active_graph.remove_edge(u, v, key=edge_key)

    def infer_edges_to_recently_deactivated(
        self,
        current_sentence_nodes: List["Node"],
        current_sentence_words: List[str],
        current_text: str,
        recently_deactivated_nodes: List["Node"],
        enforce_attachment: bool,
    ) -> List["Edge"]:
        if not enforce_attachment:
            return []

        recent = [n for n in recently_deactivated_nodes if n in self._graph.nodes]
        if not recent or not current_sentence_nodes:
            return []

        candidate_pairs: set[frozenset["Node"]] = set()
        for node in current_sentence_nodes:
            for other in recent:
                if node == other:
                    continue
                candidate_pairs.add(frozenset((node, other)))
        if not candidate_pairs:
            return []

        nodes_for_prompt = {n for pair in candidate_pairs for n in pair}

        def _node_line(node: "Node") -> str:
            return f" - ({node.get_text_representer()}, {node.node_type})\n"

        nodes_from_text = "".join(
            _node_line(n)
            for n in sorted(nodes_for_prompt, key=lambda x: x.get_text_representer())
        )
        graph_nodes_repr = self._graph.get_nodes_str(list(nodes_for_prompt))
        graph_edges_repr, _ = self._graph.get_edges_str(
            list(nodes_for_prompt), only_text_based=False
        )

        try:
            new_relationships = self._client.get_new_relationships(
                nodes_from_text,
                graph_nodes_repr,
                graph_edges_repr,
                current_text,
                self._persona,
            )
        except Exception:
            logging.error("LLM edge inference failed", exc_info=True)
            return []

        added: List["Edge"] = []
        for idx, relationship in enumerate(new_relationships):
            if relationship is None or isinstance(relationship, (int, float, bool)):
                continue
            if isinstance(relationship, dict):
                subj = relationship.get("subject") or relationship.get("head")
                rel = relationship.get("relation") or relationship.get("predicate")
                obj = relationship.get("object") or relationship.get("tail")
                if not (subj and rel and obj):
                    continue
                relationship = (str(subj), str(rel), str(obj))
            if not isinstance(relationship, (list, tuple)) or len(relationship) != 3:
                continue

            subj, rel, obj = relationship
            subj = self._normalize_endpoint_text_fn(subj, is_subject=True) or None
            obj = self._normalize_endpoint_text_fn(obj, is_subject=False) or None
            if subj is None or obj is None:
                continue
            if not subj or not obj:
                continue
            if not isinstance(subj, str) or not isinstance(obj, str):
                continue
            edge_label = rel.replace("(edge)", "").strip()
            edge_label = self._normalize_edge_label_fn(edge_label)
            if not self._is_valid_relation_label_fn(edge_label):
                continue

            subj_node = self._find_node_by_text_fn(subj, nodes_for_prompt)
            obj_node = self._find_node_by_text_fn(obj, nodes_for_prompt)
            if subj_node is None or obj_node is None:
                continue
            pair_key = frozenset((subj_node, obj_node))
            if pair_key not in candidate_pairs:
                continue

            edge = self._add_edge_fn(
                subj_node,
                obj_node,
                edge_label,
                self._edge_visibility,
            )
            if edge:
                added.append(edge)
        return added
