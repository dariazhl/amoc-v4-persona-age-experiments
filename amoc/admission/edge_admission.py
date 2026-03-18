from typing import TYPE_CHECKING, Optional, List, Tuple
import logging

from amoc.core.graph import Graph
from amoc.core.node import Node, NodeType, NodeSource
from amoc.core.edge import Edge
from amoc.admission.text_normalizer import TextNormalizer
from amoc.utils.spacy_utils import clean_label

if TYPE_CHECKING:
    from amoc.pipeline.orchestrator import AMoCv4


class EdgeAdmission:

    def __init__(
        self,
        graph_ref: "Graph",
        llm_extractor,
        spacy_nlp,
        get_explicit_nodes: callable,
        get_carryover_nodes: callable,
        get_attachable_nodes: callable,
        edge_visibility: int,
        debug: bool = False,
    ):
        self._graph = graph_ref
        self._llm = llm_extractor
        self._spacy_nlp = spacy_nlp
        self._get_explicit_nodes = get_explicit_nodes
        self._get_carryover_nodes = get_carryover_nodes
        self._get_attachable_nodes = get_attachable_nodes
        self._edge_visibility = edge_visibility
        self._debug = debug
        self._triplet_intro = {}
        self._current_sentence_index = None
        self._normalize_endpoint_text_fn: Optional[callable] = None
        self._normalize_edge_label_fn: Optional[callable] = None
        self._is_valid_relation_label_fn: Optional[callable] = None
        self._find_node_by_text_fn: Optional[callable] = None
        self.add_edge_wrapper_fn: Optional[callable] = None
        self._persona: str = ""

    def configure_edge_inference_callbacks(
        self,
        normalize_endpoint_text_fn: callable,
        normalize_edge_label_fn: callable,
        is_valid_relation_label_fn: callable,
        find_node_by_text_fn: callable,
        add_edge_fn: callable,
        persona: str,
    ):
        self._normalize_endpoint_text_fn = normalize_endpoint_text_fn
        self._normalize_edge_label_fn = normalize_edge_label_fn
        self._is_valid_relation_label_fn = is_valid_relation_label_fn
        self._find_node_by_text_fn = find_node_by_text_fn
        self.add_edge_wrapper_fn = add_edge_fn
        self._persona = persona

    def configure_edge_state_refs(
        self,
        triplet_intro: dict,
    ):
        self._triplet_intro = triplet_intro

    def configure_edge_admission_with_core(self, core: "AMoCv4") -> None:
        self.configure_edge_inference_callbacks(
            normalize_endpoint_text_fn=core._normalize_endpoint_text,
            normalize_edge_label_fn=core._normalize_edge_label,
            is_valid_relation_label_fn=core._is_valid_relation_label,
            find_node_by_text_fn=lambda t, c: core._node_ops.find_node_by_text(t, c),
            add_edge_fn=core.add_edge_wrapper,
            persona=core.persona,
        )
        self.configure_edge_state_refs(
            triplet_intro=core._triplet_intro,
        )

    # TODO: remove these lists from here
    # Negation phrases that should never appear as edge labels
    _NEGATION_PHRASES = frozenset(
        {
            "not related",
            "no connection",
            "not available",
            "not applicable",
            "not connected",
            "no relation",
            "no link",
            "not involved",
            "not associated",
            "without connection",
            "without relation",
            "not_related",
            "no_connection",
            "not_available",
            "not_applicable",
            "not_connected",
            "no_relation",
            "no_link",
            "not_involved",
            "not_associated",
            "without_connection",
            "without_relation",
            "unconnected",
            "unrelated",
            "disconnected",
            "disassociated",
            "nonapplicable",
            "nonexistent",
            "unavailable",
            "uninvolved",
        }
    )

    _NEGATION_WORDS = frozenset({"not", "no", "never", "neither", "nor", "without"})

    def is_negation_label(self, label: str) -> bool:
        if not label:
            return False
        normalized = label.lower().replace("_", " ").strip()
        if any(phrase in normalized for phrase in self._NEGATION_PHRASES):
            return True
        words = normalized.split()
        return any(w in self._NEGATION_WORDS for w in words)

    def set_edge_sentence_context(self, idx: int):
        self._current_sentence_index = idx

    # helper method
    def build_edge_triplet_key(self, edge: "Edge") -> Tuple[str, str, str]:
        return (
            edge.source_node.get_text_representer(),
            edge.dest_node.get_text_representer(),
            edge.label,
        )

    def find_existing_directed_edge(
        self, source_node: "Node", dest_node: "Node"
    ) -> Optional["Edge"]:
        for edge in self._graph.edges:
            if edge.source_node == source_node and edge.dest_node == dest_node:
                return edge
        return None

    def insert_normalized_edge(
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

        label = clean_label(label)
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
        # Reject self-loops bc S-V-O triplets require distinct subject and object
        if source_node == dest_node:
            return None

        # Safety net: reject negation relations that slipped past upstream validation
        if self.is_negation_label(label):
            logging.info(f"EDGE_ADMISSION: rejected negation edge label: '{label}'")
            return None

        # If both nodes are explicit in the current sentence, always allow
        explicit_nodes = self._get_explicit_nodes()
        both_explicit = source_node in explicit_nodes and dest_node in explicit_nodes

        # Also allow if this is the first sentence and we have explicit nodes
        is_first_sentence = self._current_sentence_index == 0
        has_explicit_nodes = len(explicit_nodes) > 0
        if both_explicit or (is_first_sentence and has_explicit_nodes):
            # Skip attachment constraint entirely
            pass
        else:
            # check attachment constraint
            attachable = (
                self._get_attachable_nodes() if self._get_attachable_nodes else set()
            )
            if not bypass_attachment_constraint:
                if source_node not in attachable and dest_node not in attachable:
                    logging.warning(
                        f"EDGE_ADMISSION: attachability failed: {source_node.get_text_representer()} not in attachable, "
                        f"{dest_node.get_text_representer()} not in attachable. Attachable: {[n.get_text_representer() for n in attachable]}"
                    )
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
        # clean label
        label = clean_label(label)
        if not label:
            return None

        use_sentence = (
            created_at_sentence
            if created_at_sentence is not None
            else self._current_sentence_index
        )
        existing_edge = self.find_existing_directed_edge(source_node, dest_node)
        if existing_edge is not None:
            old_label = existing_edge.label

            if old_label.strip().lower() == label.strip().lower():
                existing_edge.label = label
                existing_edge.visibility_score = edge_forget
                existing_edge.created_at_sentence = use_sentence
                existing_edge.mark_as_current_sentence(reset_score=True)

                if self._debug:
                    logging.info(
                        "replaced equivalent edge: %s --%s--> %s (was: %s)",
                        source_node.get_text_representer(),
                        label,
                        dest_node.get_text_representer(),
                        old_label,
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

                return existing_edge
            self._graph.remove_edge(existing_edge)
        # edge duplication ie. knight - forest - forest
        if (
            label == source_node.get_text_representer()
            or label == dest_node.get_text_representer()
        ):
            return None
        # add edge
        edge = self.insert_normalized_edge(
            source_node,
            dest_node,
            label,
            edge_forget,
            created_at_sentence=use_sentence,
            relation_class=relation_class,
            justification=justification,
        )
        # mark edge as part of current sentence with full activation
        if edge:
            if use_sentence == self._current_sentence_index:
                edge.mark_as_current_sentence(reset_score=True)

            trip_id = (
                edge.source_node.get_text_representer(),
                edge.label,
                edge.dest_node.get_text_representer(),
            )
            if trip_id not in self._triplet_intro:
                self._triplet_intro[trip_id] = (
                    use_sentence if use_sentence is not None else -1
                )

        return edge

    # used in stabilizer.py as fallback if edge is rejected by the LLM to prevent disconnection (last resort)
    def create_forced_connectivity_edges(
        self,
        story_context: Optional[str] = None,
        current_sentence: Optional[str] = None,
        mode: str = "active",
        persona: str = "",
        normalize_edge_label_fn: callable = None,
    ) -> List["Edge"]:
        # distinguish between active and cumulative graph
        if mode == "active":
            protected_nodes = self._get_explicit_nodes() | self._get_carryover_nodes()
        else:
            protected_nodes = set(self._graph.nodes)
        # find disconnected components
        components, _ = self._graph.get_disconnected_components_wrapper(protected_nodes)

        if len(components) <= 1:
            return []
        # sort fragments by size
        if mode == "active":
            components = sorted(
                components,
                key=lambda c: len(set(c) & protected_nodes),
                reverse=True,
            )
        else:
            components = sorted(components, key=len, reverse=True)
        # find the largest component
        backbone = set(components[0])
        forced_edges = []

        for comp in components[1:]:
            node_a = next(iter(backbone))
            node_b = next(iter(comp))

            # try LLM repair (2 attempts) to repair edge
            for _ in range(2):
                result = self._llm.get_forced_connectivity_edge_label(
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
                # add edge to backbone component
                if edge:
                    edge.mark_as_current_sentence(reset_score=True)
                    forced_edges.append(edge)
                    backbone.update(comp)
                    break

        return forced_edges

    def record_edge_in_graphs(
        self,
        edge: "Edge",
        sentence_idx: Optional[int],
    ) -> None:
        """Track when each triplet was first introduced."""
        u, v, lbl = self.build_edge_triplet_key(edge)

        if not lbl or not lbl.strip():
            return

        introduced = self._triplet_intro.get((u, lbl, v))
        if introduced is None:
            introduced = (
                edge.created_at_sentence if edge.created_at_sentence is not None else -1
            )
        self._triplet_intro[(u, lbl, v)] = int(introduced)

    def llm_attach_explicit_to_carryover(
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

        def node_line(node: "Node") -> str:
            return f" - ({node.get_text_representer()}, {node.node_type})\n"

        nodes_from_text = "".join(
            node_line(n)
            for n in sorted(nodes_for_prompt, key=lambda x: x.get_text_representer())
        )
        graph_nodes_repr = self._graph.get_nodes_str(list(nodes_for_prompt))
        graph_edges_repr, _ = self._graph.get_edges_str(
            list(nodes_for_prompt), only_text_based=False
        )

        try:
            new_relationships = self._llm.get_new_relationships(
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

            edge = self.add_edge_wrapper_fn(
                subj_node,
                obj_node,
                edge_label,
                self._edge_visibility,
            )
            if edge:
                added.append(edge)
        return added
