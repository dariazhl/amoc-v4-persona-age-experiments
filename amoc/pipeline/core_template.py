"""
AMoC v4 Core Pipeline - Template for Implementation

This template provides the structure for the AMoCGraph class with:
- All method signatures
- Docstrings explaining each method's purpose
- Comments guiding implementation
- Minimal stub bodies for you to fill in

Key Concepts:
- Explicit nodes: Extracted directly from sentence text (nouns, adjectives)
- Carry-over nodes: Reachable from explicit via active edges within MaxDistance
- Inactive nodes: Not reachable, edges have decayed
- Two-step construction: Step 1 (backbone) then Step 2 (inference)
"""

import logging
import os
import re
from typing import List, Tuple, Optional, Iterable
import pandas as pd
from spacy.tokens import Span, Token
import networkx as nx

from amoc.graph import Graph, Node, Edge, NodeType, NodeSource
from amoc.viz.graph_plots import plot_amoc_triplets
from amoc.llm.vllm_client import VLLMClient
from amoc.nlp.spacy_utils import (
    get_concept_lemmas,
    canonicalize_node_text,
    get_content_words_from_sent,
)
from collections import deque


# =============================================================================
# MODULE-LEVEL HELPER
# =============================================================================


def _sanitize_filename_component(component: str, max_len: int = 80) -> str:
    """
    Sanitize a string for use in filenames.

    Args:
        component: Raw string to sanitize
        max_len: Maximum length of output

    Returns:
        Sanitized string safe for filenames
    """
    # TODO: Replace unsafe characters, truncate to max_len
    pass


# =============================================================================
# MAIN CLASS
# =============================================================================


class AMoCGraph:
    """
    Associative Memory of Concepts (AMoC) Graph Builder.

    Processes text sentence-by-sentence to build a knowledge graph that
    represents concepts and their relationships, with memory decay and
    activation mechanisms.

    Key attributes:
        graph: The underlying Graph object storing nodes and edges
        cumulative_graph: NetworkX graph tracking all edges ever added
        active_graph: NetworkX graph tracking currently active edges
        strict_attachament_constraint: When True, enforces two-step construction
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        client: VLLMClient,
        spacy_nlp,
        context_length: int = 3,
        edge_forget: int = 4,
        max_distance_from_active_nodes: int = 2,
        strict_attachament_constraint: bool = True,
        persona: str = "barebones",
        debug: bool = False,
    ) -> None:
        """
        Initialize the AMoC graph builder.

        Args:
            client: LLM client for relationship extraction
            spacy_nlp: SpaCy language model for NLP processing
            context_length: Number of previous sentences to keep in context
            edge_forget: Number of sentences before an edge becomes inactive
            max_distance_from_active_nodes: BFS distance for carry-over nodes
            strict_attachament_constraint: Enable two-step construction guarantees
            persona: Persona type affecting LLM prompts
            debug: Enable verbose logging
        """
        # TODO: Store all parameters as instance attributes
        # TODO: Initialize self.graph = Graph()
        # TODO: Initialize NetworkX graphs: cumulative_graph, active_graph
        # TODO: Initialize tracking structures:
        #   - _triplet_intro: dict for (subj, rel, obj) -> sentence_index
        #   - _cumulative_triplet_records: list for activation records
        #   - _anchor_nodes: set of nodes that anchor connectivity
        #   - _fixed_hub: Optional hub node for backbone connectivity
        #   - _explicit_nodes_current_sentence: set of current explicit nodes
        pass

    # =========================================================================
    # GRAPH STATE & DISTANCE COMPUTATION
    # =========================================================================

    def _node_token_for_matrix(self, node: Node) -> str:
        """
        Get lowercase text representation of a node for matrix operations.

        Args:
            node: Node to get text for

        Returns:
            Lowercase stripped text representation
        """
        pass

    def _distances_from_sources_active_edges(
        self, sources: set[Node], max_distance: int
    ) -> dict[Node, int]:
        """
        BFS from source nodes using only ACTIVE edges.

        This is the core mechanism for determining carry-over nodes.
        Only nodes reachable within max_distance via active edges are included.

        Args:
            sources: Set of explicit nodes to start BFS from
            max_distance: Maximum BFS depth

        Returns:
            Dict mapping reachable nodes to their distance from sources

        Implementation:
            1. Initialize distances dict with sources at distance 0
            2. Use deque for BFS
            3. For each node, check ACTIVE edges only
            4. Add neighbors if within max_distance and not visited
        """
        pass

    def _get_nodes_with_active_edges(self) -> set[Node]:
        """
        Get all nodes that have at least one active edge.

        Returns:
            Set of nodes connected by active edges
        """
        pass

    def _get_active_node_degree(self, node: Node) -> int:
        """
        Get the degree of a node counting only ACTIVE edges.

        CRITICAL: This is the correct method for connectivity checks.
        Inactive edges don't appear in plots, so they don't count.

        Args:
            node: Node to check

        Returns:
            Number of ACTIVE edges connected to this node
        """
        pass

    def _has_active_edge_to_hub(self, node: Node, hub: Node) -> bool:
        """
        Check if node has an ACTIVE edge to the hub.

        CRITICAL: Only checks ACTIVE edges. Inactive edges don't count
        because they don't appear in the visualization.

        Args:
            node: Node to check
            hub: Hub node

        Returns:
            True if node has at least one ACTIVE edge to hub
        """
        pass

    def _restrict_active_to_current_explicit(self, explicit_nodes: List[Node]) -> None:
        """
        Reset activation scores: only explicit nodes start as fully active.

        This implements "Step 5" from Section 3.1.2 - text-based nodes
        get their scores reset to maximum.

        Args:
            explicit_nodes: Nodes extracted from current sentence text
        """
        pass

    # =========================================================================
    # SENTENCE ACTIVATION RECORDING (for analysis/visualization)
    # =========================================================================

    def _record_sentence_activation(
        self,
        sentence_id: int,
        explicit_nodes: List[Node],
        newly_inferred_nodes: set[Node],
    ) -> None:
        """
        Record activation state after processing a sentence.

        Updates cumulative_graph and active_graph NetworkX structures.
        Tracks which triplets are active at each sentence.

        Args:
            sentence_id: 1-indexed sentence number
            explicit_nodes: Nodes from current sentence text
            newly_inferred_nodes: Nodes added via inference this sentence

        Implementation:
            1. Update active_graph with currently active edges
            2. Record triplet activation states
            3. Update cumulative records for analysis
        """
        pass

    # =========================================================================
    # ATTACHMENT CONSTRAINT - TWO-STEP CONSTRUCTION
    # Aligned with AMoC v4 Paper Section 3.1.2
    # =========================================================================

    def _passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List[Node],
        graph_active_nodes: List[Node],
        graph_active_edge_nodes: Optional[set[Node]] = None,
    ) -> bool:
        """
        Check if a proposed edge passes the attachment constraint.

        In strict mode: edges must connect to active nodes or existing memory.
        In legacy mode: edges only need to "touch memory" (any existing node).

        Args:
            subject: Subject text of proposed edge
            obj: Object text of proposed edge
            current_sentence_words: Words from current sentence
            current_sentence_nodes: Nodes from current sentence
            graph_active_nodes: Currently active nodes
            graph_active_edge_nodes: Nodes with active edges

        Returns:
            True if edge should be allowed, False otherwise
        """
        pass

    def _is_explicit_backbone_connected(self, explicit_nodes: List[Node]) -> bool:
        """
        Check if explicit nodes form a connected subgraph.

        Step 1 requires that all explicit nodes be connected before
        proceeding to Step 2 (inference).

        Args:
            explicit_nodes: Nodes extracted from sentence text

        Returns:
            True if connected (or <= 1 node), False otherwise

        Implementation:
            1. Build NetworkX undirected graph from explicit nodes
            2. Add edges where BOTH endpoints are explicit
            3. Return nx.is_connected(G)
        """
        pass

    def _select_hub_node(self, explicit_nodes: List[Node]) -> Optional[Node]:
        """
        Select the best hub node for backbone connectivity.

        Prefers CONCEPT nodes over PROPERTY nodes, then by degree.
        Once selected, the hub is fixed for the story (_fixed_hub).

        Args:
            explicit_nodes: Candidate nodes for hub selection

        Returns:
            Selected hub node, or None if no candidates
        """
        pass

    def _ensure_explicit_backbone_connected(
        self,
        explicit_nodes: List[Node],
        sentence_text: str,
    ) -> List[Edge]:
        """
        Enforce hub-anchored topology for Step 1 backbone connectivity.

        If explicit nodes are disconnected, request edges from LLM
        to connect isolated nodes to the hub.

        Args:
            explicit_nodes: Nodes that must form connected backbone
            sentence_text: Current sentence for LLM context

        Returns:
            List of edges added to ensure connectivity

        Implementation:
            1. Select hub node
            2. For each explicit node not connected to hub:
               a. Request edge label from LLM (node -> hub)
               b. If valid label, add edge
            3. Return added edges
        """
        pass

    def _filter_inferred_for_attachment(
        self,
        relationship: Tuple[str, str, str],
        explicit_nodes: List[Node],
        explicit_words: List[str],
    ) -> bool:
        """
        Step 2 filter: inferences MUST attach to explicit backbone.

        At least one endpoint of an inferred relationship must be
        an explicit node from the current sentence.

        Args:
            relationship: (subject, relation, object) tuple
            explicit_nodes: Nodes from current sentence
            explicit_words: Words from current sentence

        Returns:
            True if at least one endpoint is explicit, False otherwise
        """
        pass

    def _build_per_sentence_view(
        self,
        explicit_nodes: List[Node],
        max_distance: int,
    ) -> Tuple[set[Node], set[Edge]]:
        """
        Build the per-sentence active view of the graph.

        Contains only:
        - Explicit nodes from current sentence
        - Carry-over nodes (BFS reachable via active edges)
        - Edges where BOTH endpoints are in the above set

        Args:
            explicit_nodes: Nodes from current sentence
            max_distance: BFS depth limit

        Returns:
            Tuple of (active_nodes, active_edges)
        """
        pass

    def _validate_per_sentence_connectivity(
        self,
        active_nodes: set[Node],
        active_edges: set[Edge],
        sentence_id: int,
    ) -> bool:
        """
        Validate that per-sentence graph is connected.

        Logs error if disconnected (no silent repair in strict mode).

        Args:
            active_nodes: Nodes in per-sentence view
            active_edges: Edges in per-sentence view
            sentence_id: For error logging

        Returns:
            True if connected, False otherwise
        """
        pass

    def _validate_connectivity_invariants(
        self,
        explicit_nodes: List[Node],
        sentence_id: int,
    ) -> None:
        """
        Validate connectivity invariants before plotting.

        CRITICAL: Uses ACTIVE edges only for all checks.
        Inactive edges don't appear in plots, so they don't count.

        Invariants checked:
        1. Every node has at least one ACTIVE edge (degree > 0)
        2. Every node has an ACTIVE edge to hub (direct or via path)
        3. The active graph is connected

        Args:
            explicit_nodes: Explicit nodes from current sentence
            sentence_id: For error logging

        Implementation:
            1. get_active_node_degree() - counts only ACTIVE edges
            2. has_active_edge_to_hub() - checks only ACTIVE edges
            3. Log warnings for any violations
        """
        pass

    def _ensure_displayed_nodes_connected(
        self,
        displayed_node_names: List[str],
        sentence_text: str,
    ) -> List[Tuple[str, str, str]]:
        """
        CRITICAL: Ensure ALL displayed nodes are connected through edges.

        This is the FINAL safety net before plotting. It guarantees:
        1. Every displayed node has at least one edge in the graph
        2. All displayed nodes are connected to the hub
        3. Returns filtered triplets containing ONLY edges between displayed nodes

        Args:
            displayed_node_names: List of node names that WILL be displayed
                                  (explicit + salient, NOT inactive)
            sentence_text: Current sentence for LLM context

        Returns:
            List of triplets where BOTH endpoints are in displayed_node_names.
            This ensures NO edges to/from inactive nodes appear in the plot.

        Implementation:
            1. Find displayed nodes with NO edges to other displayed nodes
            2. For each disconnected node, force-add edge to hub
            3. Return FILTERED triplets - only edges between displayed nodes
        """
        pass

    # =========================================================================
    # EDGE MANAGEMENT
    # =========================================================================

    def _add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_forget: int,
        created_at_sentence: Optional[int] = None,
    ) -> Optional[Edge]:
        """
        Add an edge to the graph with connectivity checks.

        In strict mode, edges must connect to the main component
        or current explicit nodes.

        Args:
            source_node: Edge source
            dest_node: Edge destination
            label: Relationship label
            edge_forget: Decay parameter
            created_at_sentence: Optional sentence index

        Returns:
            The created Edge, or None if rejected
        """
        pass

    def _has_edge_between(self, a: Node, b: Node) -> bool:
        """Check if any edge exists between two nodes (either direction)."""
        pass

    def _edge_key(self, edge: Edge) -> tuple[str, str, str]:
        """Get canonical (subj, rel, obj) tuple for an edge."""
        pass

    def _record_edge_in_graphs(self, edge: Edge, sentence_idx: Optional[int]) -> None:
        """Record edge in cumulative_graph and update _triplet_intro."""
        pass

    # =========================================================================
    # GRAPH CONNECTIVITY ENFORCEMENT
    # =========================================================================

    def reset_graph(self) -> None:
        """Reset the graph to empty state."""
        pass

    def _enforce_graph_connectivity(self) -> None:
        """
        Post-hoc connectivity enforcement.

        CRITICAL REQUIREMENTS:
        1. Every displayed node must have at least one ACTIVE edge
        2. Every displayed node must be connected to the hub
        3. Only count ACTIVE edges for connectivity checks
           (inactive edges don't appear in plots)

        Implementation:
        1. Find nodes with ACTIVE degree 0 (dangling in active view)
        2. For each dangling node, add edge to hub via LLM
        3. If LLM fails, use aggressive direct edge creation fallback
        4. Validate all nodes are connected before plotting
        """
        pass

    def _can_attach(self, node: Node) -> bool:
        """Check if a node can attach to the current graph structure."""
        pass

    # =========================================================================
    # EDGE REACTIVATION
    # =========================================================================

    def reactivate_relevant_edges(
        self,
        active_nodes: list[Node],
        prev_sentences_text: str,
        newly_added_edges: list[Edge],
    ) -> None:

        edges_text, graph_edges = self.graph.get_edges_str(
            self.graph.nodes, only_active=False
        )

        # Always activate structural edges
        for edge in graph_edges:
            if edge.metadata.get("structural"):
                edge.active = True
                edge.forget_score = self.edge_forget
                self._record_edge_in_graphs(edge, self._current_sentence_index)

        if not self.strict_reactivate_function:
            for edge in graph_edges:
                edge.active = True
                edge.forget_score = self.edge_forget
                self._record_edge_in_graphs(edge, self._current_sentence_index)
            return

        raw_indices = self.client.get_relevant_edges(
            edges_text, prev_sentences_text, self.persona
        )

        selected: set[int] = set()
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= len(graph_edges):
                selected.add(i)

        if not selected:
            for i, edge in enumerate(graph_edges, start=1):
                if edge in newly_added_edges:
                    selected.add(i)

        # Activate selected edges
        for i in selected:
            edge = graph_edges[i - 1]
            edge.active = True
            edge.forget_score = self.edge_forget
            self._record_edge_in_graphs(edge, self._current_sentence_index)

    # Deactivate others but preserve connectivity
    def active_connected() -> bool:
        G = nx.Graph()
        for e in graph_edges:
            if e.active:
                G.add_edge(e.source_node, e.dest_node)
        return G.number_of_nodes() <= 1 or nx.is_connected(G)

    for edge in graph_edges:
        if edge.metadata.get("structural"):
            continue
        if not edge.active:
            continue
        edge.active = False
        if not active_connected():
            edge.active = True
            edge.forget_score = 0
        self._record_edge_in_graphs(edge, self._current_sentence_index)

    def _infer_edges_to_recently_deactivated(
        self,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        current_all_text: str,
    ) -> List[Edge]:
        """
        Attempt to infer edges to recently deactivated nodes.

        Prevents important nodes from being lost by trying to
        reconnect them to current explicit nodes.

        Args:
            current_sentence_text_based_nodes: Current explicit nodes
            current_sentence_text_based_words: Current explicit words
            current_all_text: Current sentence text

        Returns:
            List of edges added to reconnect deactivated nodes
        """
        pass

    # =========================================================================
    # TEXT PROCESSING & NORMALIZATION
    # =========================================================================

    def resolve_pronouns(self, text: str) -> str:
        """
        Resolve pronouns in text using LLM.

        Args:
            text: Raw sentence text

        Returns:
            Text with pronouns replaced by referents
        """
        pass

    def _normalize_label(self, label: str) -> str:
        """Normalize edge label (lowercase, strip)."""
        pass

    def _normalize_endpoint_text(self, text: str, is_subject: bool) -> Optional[str]:
        """
        Normalize and validate endpoint text.

        Filters out invalid patterns like pronouns, articles, etc.

        Args:
            text: Raw endpoint text
            is_subject: True if subject position, False if object

        Returns:
            Normalized text, or None if invalid
        """
        pass

    def _canonicalize_and_classify_node_text(
        self, text: str
    ) -> Tuple[str, Optional[NodeType]]:
        """
        Canonicalize text and determine node type.

        Args:
            text: Raw node text

        Returns:
            Tuple of (canonical_text, NodeType or None if invalid)
        """
        pass

    def _classify_canonical_node_text(self, canon: str) -> Optional[NodeType]:
        """Classify canonical text as CONCEPT or PROPERTY."""
        pass

    def _find_node_by_text(
        self, text: str, node_pool: Iterable[Node]
    ) -> Optional[Node]:
        """Find node in pool matching text (by lemmas)."""
        pass

    def _appears_in_story(self, text: str) -> bool:
        """Check if text appears in any recorded story text."""
        pass

    # =========================================================================
    # RELATION LABEL VALIDATION
    # =========================================================================

    def _is_generic_relation(self, label: str) -> bool:
        """Check if label is too generic (e.g., 'relates to')."""
        pass

    def _is_blacklisted_relation(self, label: str) -> bool:
        """Check if label is in blacklist."""
        pass

    def _is_verb_relation(self, label: str) -> bool:
        """Check if label is a valid verb-based relation."""
        pass

    def _is_valid_relation_label(self, label: str) -> bool:
        """
        Comprehensive validation of relation label.

        Returns:
            True if label is valid for use as edge label
        """
        pass

    # =========================================================================
    # GRAPH CONVERSION & TRIPLETS
    # =========================================================================

    def _graph_edges_to_triplets(
        self, edges: Iterable[Edge]
    ) -> List[Tuple[str, str, str]]:
        """Convert edges to (subj, rel, obj) triplets."""
        pass

    def _graph_to_triplets(self, graph: nx.MultiDiGraph) -> List[Tuple[str, str, str]]:
        """Convert NetworkX graph to triplet list."""
        pass

    def _cumulative_triplets_upto(
        self, sentence_id: int, include_inactive: bool = False
    ) -> List[Tuple[str, str, str]]:
        """Get triplets cumulative up to given sentence."""
        pass

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def _plot_graph_snapshot(
        self,
        sentence_index: int,
        sentence_text: str,
        output_dir: str,
        highlight_nodes: List[str],
        inactive_nodes: List[str],
        inactive_nodes_for_title: Optional[List[str]],
        explicit_nodes: List[str],
        salient_nodes: List[str],
        only_active: bool,
        largest_component_only: bool,
        mode: str,
        triplets_override: Optional[List[Tuple[str, str, str]]] = None,
        active_edges: Optional[set] = None,
        hub_edge_explanations: Optional[List[str]] = None,
        show_all_edges: bool = True,
    ) -> None:
        """
        Plot and save a graph snapshot.

        IMPORTANT REQUIREMENTS:
        1. Explicit nodes and carryover nodes ARE displayed in the graph
        2. All displayed nodes must be FULLY CONNECTED through edges
        3. Inactive nodes are listed in the TITLE only, NOT displayed in graph
           (pass inactive_nodes=[] to hide them from the graph visualization)

        Args:
            sentence_index: Current sentence index (0-based)
            sentence_text: Current sentence text
            output_dir: Directory to save plots
            highlight_nodes: Nodes to highlight (blue nodes)
            inactive_nodes: Pass [] to NOT display inactive nodes in graph.
            inactive_nodes_for_title: Actual inactive nodes to list in title.
                           Pass the real inactive nodes here so they appear in title.
            explicit_nodes: Nodes from current sentence (blue)
            salient_nodes: Active but not explicit nodes (carry-over, yellow)
            only_active: If True, only show active subgraph
            largest_component_only: If True, only show largest component
            mode: 'sentence_active' or 'sentence_cumulative'
            triplets_override: Optional triplets to use instead of graph
            active_edges: Set of active edge tuples
            hub_edge_explanations: Optional explanations for hub edges
            show_all_edges: If True, show all edges (Policy A)
        """
        pass

    # =========================================================================
    # MAIN ANALYSIS PIPELINE
    # =========================================================================

    def analyze(
        self,
        text: str,
        highlight_nodes: List[str] = None,
        plot_after_each_sentence: bool = False,
        graphs_output_dir: str = "./graphs",
        largest_component_only: bool = True,
    ) -> Tuple[
        List[Node],
        List[Edge],
        pd.DataFrame,
        List[Tuple[int, str, str, str, str]],
    ]:
        """
        Main entry point: analyze text and build graph.

        Processes text sentence-by-sentence following Section 3.1.2:

        For each sentence:
            1. Resolve pronouns
            2. Extract explicit nodes (nouns, adjectives)
            3. STEP 1: Build explicit backbone (request edges from LLM)
            4. Ensure backbone connectivity (hub-anchored if needed)
            5. STEP 2: Inference enrichment (must attach to backbone)
            6. Update activation scores
            7. Reactivate relevant edges
            8. Record activation state
            9. Plot if requested

        Args:
            text: Full text to analyze
            highlight_nodes: Nodes to highlight in plots
            plot_after_each_sentence: Whether to generate plots
            graphs_output_dir: Output directory for plots
            largest_component_only: Only plot largest component

        Returns:
            Tuple of:
                - List of all nodes
                - List of all edges
                - DataFrame of cumulative triplet records
                - List of (sent_idx, sent_text, subj, rel, obj) tuples
        """
        # TODO: Split text into sentences using spacy_nlp
        # TODO: Initialize tracking variables
        # TODO: For each sentence:
        #   - Resolve pronouns
        #   - If i == 0: First sentence processing (init_graph)
        #   - Else: Subsequent sentence processing (get_new_relationships)
        #   - Apply two-step construction
        #   - Record activation
        #   - Plot if requested with these CRITICAL requirements:
        #
        # PLOTTING REQUIREMENTS:
        # 1. Explicit nodes and carryover nodes ARE displayed
        # 2. All displayed nodes must be FULLY CONNECTED through edges
        # 3. Inactive nodes are listed in TITLE only, NOT displayed
        #
        # Before plotting:
        # - Filter node lists to only include nodes in active_graph
        # - Enforce connectivity: all displayed nodes must have ACTIVE edges
        # - Use final safety net to force-add edges for disconnected nodes
        # - Pass inactive_nodes=[] to _plot_graph_snapshot for both views:
        #     * sentence_active view: inactive_nodes=[]
        #     * sentence_cumulative view: inactive_nodes=[]
        pass

    # =========================================================================
    # GRAPH INITIALIZATION (First Sentence)
    # =========================================================================

    def init_graph(self, sent: Span) -> None:
        """
        Initialize graph with first sentence.

        Extracts initial relationships using LLM and builds
        the starting graph structure.

        Args:
            sent: SpaCy Span for first sentence
        """
        pass

    # =========================================================================
    # RELATIONSHIP INFERENCE
    # =========================================================================

    def infer_new_relationships_step_0(
        self, sent: Span
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Infer new relationships for first sentence (Step 2).

        Args:
            sent: SpaCy Span for sentence

        Returns:
            Tuple of (concept_relationships, property_relationships)
        """
        pass

    def infer_new_relationships(
        self,
        current_all_text: str,
        current_sentence_text_based_nodes: List[Node],
        current_sentence_text_based_words: List[str],
        active_nodes_text: str,
        active_nodes_edges_text: str,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
        """
        Infer new relationships for subsequent sentences (Step 2).

        Args:
            current_all_text: Current sentence text
            current_sentence_text_based_nodes: Explicit nodes
            current_sentence_text_based_words: Explicit words
            active_nodes_text: String representation of active nodes
            active_nodes_edges_text: String representation of active edges

        Returns:
            Tuple of (concept_relationships, property_relationships)
        """
        pass

    # =========================================================================
    # ADDING INFERRED RELATIONSHIPS TO GRAPH
    # =========================================================================

    def add_inferred_relationships_to_graph_step_0(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        sent: Span,
        explicit_nodes: Optional[List[Node]] = None,
        explicit_words: Optional[List[str]] = None,
    ) -> None:
        """
        Add inferred relationships to graph (first sentence).

        Applies Step 2 filter: inferences must attach to explicit backbone.

        Args:
            inferred_relationships: List of (subj, rel, obj) tuples
            node_type: CONCEPT or PROPERTY
            sent: SpaCy Span for sentence
            explicit_nodes: Explicit nodes for filtering
            explicit_words: Explicit words for filtering
        """
        # TODO: For each relationship:
        #   1. Validate format
        #   2. Apply _filter_inferred_for_attachment
        #   3. Normalize endpoints
        #   4. Check _appears_in_story
        #   5. Check _passes_attachment_constraint
        #   6. Get or create nodes
        #   7. Add edge
        pass

    def add_inferred_relationships_to_graph(
        self,
        inferred_relationships: List[Tuple[str, str, str]],
        node_type: NodeType,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        active_graph_nodes: List[Node],
        added_edges: List[Edge],
    ) -> None:
        """
        Add inferred relationships to graph (subsequent sentences).

        Applies Step 2 filter: inferences must attach to explicit backbone.

        Args:
            inferred_relationships: List of (subj, rel, obj) tuples
            node_type: CONCEPT or PROPERTY
            curr_sentences_nodes: Current sentence explicit nodes
            curr_sentences_words: Current sentence explicit words
            active_graph_nodes: Currently active nodes
            added_edges: Edges added this sentence (for tracking)
        """
        pass

    # =========================================================================
    # NODE EXTRACTION & LOOKUP
    # =========================================================================

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        """
        Get node from text, optionally creating it.

        Args:
            text: Node text to find/create
            curr_sentences_nodes: Current sentence nodes
            curr_sentences_words: Current sentence words
            node_source: TEXT_BASED or INFERENCE_BASED
            create_node: Whether to create if not found

        Returns:
            Found or created Node, or None
        """
        pass

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List[Node],
        curr_sentences_nodes: List[Node],
        curr_sentences_words: List[str],
        node_source: NodeSource,
        create_node: bool,
    ) -> Optional[Node]:
        """
        Get node from relationship endpoint text.

        Searches current sentence nodes first, then active graph nodes.

        Args:
            text: Endpoint text
            graph_active_nodes: Currently active nodes
            curr_sentences_nodes: Current sentence nodes
            curr_sentences_words: Current sentence words
            node_source: TEXT_BASED or INFERENCE_BASED
            create_node: Whether to create if not found

        Returns:
            Found or created Node, or None
        """
        pass

    def get_senteces_text_based_nodes(
        self, sents: List[Span], create_unexistent_nodes: bool
    ) -> Tuple[List[Node], List[str]]:
        """
        Extract text-based nodes from sentences.

        Uses SpaCy POS tagging to identify:
        - NOUN, PROPN -> NodeType.CONCEPT
        - ADJ -> NodeType.PROPERTY

        Args:
            sents: List of SpaCy Spans
            create_unexistent_nodes: Whether to create new nodes

        Returns:
            Tuple of (nodes_list, words_list)
        """
        pass

    def is_content_word_and_non_stopword(self, word: Token) -> bool:
        """
        Check if word is a content word (noun/adj) and not a stopword.

        Args:
            word: SpaCy Token

        Returns:
            True if valid content word
        """
        pass
