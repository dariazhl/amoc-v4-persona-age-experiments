from .node import Node, NodeType, NodeSource
from .edge import (
    Edge,
    RelationClass,
    Justification,
    EventiveRole,
    InvalidEdgeError,
    enforce_ontology_invariants,
    assert_persona_did_not_modify_ontology,
    collect_edge_violations,
)
from .graph import Graph
from .per_sentence_graph import (
    PerSentenceGraph,
    PerSentenceGraphBuilder,
    build_per_sentence_graph,
)

__all__ = [
    "Node",
    "NodeType",
    "NodeSource",
    "Edge",
    "Graph",
    "PerSentenceGraph",
    "PerSentenceGraphBuilder",
    "build_per_sentence_graph",
    # Ontology types (non-blocking diagnostics)
    "RelationClass",
    "Justification",
    "EventiveRole",
    "InvalidEdgeError",  # DEPRECATED: kept for backwards compatibility
    "enforce_ontology_invariants",
    "assert_persona_did_not_modify_ontology",
    "collect_edge_violations",
]
