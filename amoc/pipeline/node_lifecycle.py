from typing import Optional
from amoc.graph import Node, Graph
from amoc.graph.node import NodeType, NodeSource, NodeProvenance, NodeRole


class NodeLifecycleManager:
    """
    Authoritative lifecycle controller for node creation and explicit marking.

    This class guarantees:
    - Single creation pathway
    - Single explicit marking pathway
    - Deterministic origin_sentence assignment
    """

    def __init__(self, graph: Graph):
        self.graph = graph

    def get_or_create(
        self,
        *,
        lemma: str,
        node_type: NodeType,
        node_source: NodeSource,
        provenance: NodeProvenance,
        sentence_index: int,
        node_role: Optional[NodeRole] = None,
        mark_explicit: bool = False,
    ) -> Optional[Node]:

        node = self.graph.get_node([lemma])

        if node is None:
            node = self.graph.add_or_get_node(
                [lemma],
                lemma,
                node_type,
                node_source,
                provenance=provenance,
                origin_sentence=sentence_index,
                node_role=node_role,
                mark_explicit=False,  # explicit handled below
            )

        if node is None:
            return None

        if mark_explicit:
            node.mark_explicit_in_sentence(sentence_index)

        return node
