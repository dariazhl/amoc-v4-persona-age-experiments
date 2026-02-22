from amoc.graph.node import Node
from amoc.graph.node import NodeType, NodeSource, NodeProvenance, NodeRole
from amoc.graph.edge import (
    Edge,
    RelationClass,
    Justification,
    enforce_ontology_invariants,
    assert_persona_did_not_modify_ontology,
)
from collections import deque
from typing import List, Set, Dict, Optional, Tuple, Callable
import re
import logging
import networkx as nx


class Graph:
    # ==========================================================================
    # LOW-INFORMATION EDGE LABELS (Issue B - Node Coagulation Prevention)
    # ==========================================================================
    # These labels create high-degree hub clusters and must be throttled.
    LOW_INFO_LABELS: set[str] = {"has", "relates_to", "is", "involves", "concerns"}

    # ==========================================================================
    # FORBIDDEN NODE LEMMAS (Phase 1 - Hard Block Persona & Meta Nodes)
    # ==========================================================================
    # Meta-ontological nouns and persona-derived terms that must NEVER become nodes.
    # These are silently rejected at node admission time.
    FORBIDDEN_NODE_LEMMAS: set[str] = {
        # Persona-derived terms
        "student",
        "persona",
        # Meta-ontological nouns (from LLM explanations/summaries)
        "relation",
        "context",
        "object",
        "place",
        "story",
        "narrative",
        "sentence",
        # Technical graph terms (legacy, kept for compatibility)
        "edge",
        "node",
        "property",
        "label",
        "target",
        "source",
        "pronoun",
        "noun",
        "user",
    }

    # Nodes that are NEVER allowed, even if explicit
    NARRATION_ARTIFACT_LEMMAS: set[str] = {
        "text",
        "sentence",
        "paragraph",
        "mention",
        "mentions",
        "narration",
        "story",
    }

    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()

        # ==========================================================================
        # PROVENANCE GATE (Issue A - Persona Leakage Prevention)
        # ==========================================================================
        # story_lemmas: Set of lemmas from story text (valid for node creation)
        # persona_only_lemmas: Set of lemmas ONLY in persona (must be blocked)
        self._story_lemmas: Optional[Set[str]] = None
        self._persona_only_lemmas: Optional[Set[str]] = None

        # ==========================================================================
        # EDGE BUDGET (Issue B - Node Coagulation Prevention)
        # ==========================================================================
        # Per-sentence, per-node, per-label edge counts to prevent hub formation
        # Structure: {sentence_idx: {node_lemma_tuple: {label: count}}}
        self._edge_budget: Dict[int, Dict[tuple, Dict[str, int]]] = {}
        self._current_sentence_idx: int = 0
        # Sentence-scoped lemma gate (for explicit node enforcement)
        self._current_sentence_lemmas: Optional[Set[str]] = None

    def set_current_sentence_lemmas(self, lemmas: Set[str]) -> None:
        """
        Set the lemma set for the current sentence.
        Used to enforce that explicit nodes must originate from this sentence.
        """
        self._current_sentence_lemmas = {l.lower() for l in lemmas}

    def set_provenance_gate(
        self,
        story_lemmas: Set[str],
        persona_only_lemmas: Optional[Set[str]] = None,
    ) -> None:
        """
        Configure the provenance gate for node creation.

        Args:
            story_lemmas: Set of lemmas from story text (valid for node creation)
            persona_only_lemmas: Set of lemmas ONLY in persona (must be blocked)
        """
        self._story_lemmas = {s.lower() for s in story_lemmas}
        self._persona_only_lemmas = (
            {s.lower() for s in persona_only_lemmas} if persona_only_lemmas else set()
        )

    def set_current_sentence(self, sentence_idx: int) -> None:
        """Set the current sentence index for edge budget tracking."""
        self._current_sentence_idx = sentence_idx
        # Initialize budget for this sentence if not exists
        if sentence_idx not in self._edge_budget:
            self._edge_budget[sentence_idx] = {}

    def add_or_get_node(
        self,
        lemmas: list[str],
        actual_text: str,
        node_type: NodeType,
        node_source: NodeSource,
        *,
        admit: Optional[Callable] = None,
        admit_kwargs: dict | None = None,
        origin_sentence: Optional[int] = None,
        provenance: Optional[NodeProvenance] = None,
        node_role: Optional[NodeRole] = None,
        mark_explicit: bool = True,
    ):
        """
        Add a new node or get existing node with matching lemmas.

        PROVENANCE TRACKING (Paper-Aligned):
        - origin_sentence: The sentence index where this node was created
        - provenance: How this node was derived (STORY_TEXT or INFERRED_FROM_STORY)

        NODE ROLE:
        - node_role: Semantic role (ACTOR, OBJECT, PROPERTY, SETTING)
        - SETTING nodes are locations/environments from prepositional phrases

        PHASE 2 - SENTENCE-SCOPED NODE PROVENANCE:
        - mark_explicit: If True, marks the node as explicit in origin_sentence
        - Only set mark_explicit=True if the node appears in the sentence's dependency parse
        - Carry-over nodes should use mark_explicit=False

        CRITICAL: Nodes must only come from story text, never from persona.
        Persona influences salience/weights only, never graph content.
        """
        lemmas = [lemma.lower() for lemma in lemmas]
        if not lemmas or not lemmas[0]:
            return None
        primary_lemma = lemmas[0].lower()

        # BLOCK single-character junk
        if len(primary_lemma) <= 1:
            return None

        # BLOCK known garbage tokens (includes vague/generic terms)
        GARBAGE_LEMMAS = {
            "edge",
            "node",
            "relation",
            "property",
            "label",
            "target",
            "source",
            "t",
            # Additional garbage per replication mode
            "type",
            "person",
            "approach",
            "kind",
            "thing",
        }

        if primary_lemma in GARBAGE_LEMMAS:
            return None
        # EVENT nodes can have non-alphabetic names like "killing_knight_dragon_s0"
        # Only apply isalpha check to CONCEPT and PROPERTY nodes
        import re

        if node_type != NodeType.EVENT and not re.match(r"^[a-zA-Z]+$", lemmas[0]):
            return None

        # ==========================================================================
        # PHASE 1: FORBIDDEN LEMMA BLOCKLIST (Hard Block Persona & Meta Nodes)
        # ==========================================================================
        if any(lemma in self.FORBIDDEN_NODE_LEMMAS for lemma in lemmas):
            return None

        if any(lemma in self.NARRATION_ARTIFACT_LEMMAS for lemma in lemmas):
            logging.debug(f"NARRATION ARTIFACT BLOCKED: {lemmas}")
            return None

        # ==========================================================================
        # HARD PROVENANCE GATE (Issue A - Persona Leakage Prevention)
        # ==========================================================================
        # CRITICAL: This gate MUST block persona-only lemmas BEFORE node creation.
        # Per AMoC v4 paper: Nodes come from STORY TEXT only, never persona.
        # GATE 1: Block lemmas that appear ONLY in persona (hard reject)
        if self._persona_only_lemmas and primary_lemma in self._persona_only_lemmas:
            logging.debug(
                f"PROVENANCE GATE: Blocked persona-only lemma '{primary_lemma}'"
            )
            return None

        # GATE 2: For new nodes, require story grounding (unless INFERRED_FROM_STORY)
        # Allow existing nodes to be retrieved (they were already validated)
        existing_node = self.get_node(lemmas)
        if existing_node is None:
            if self._story_lemmas is not None:
                primary_lemma = lemmas[0]
                ed_stem = (
                    primary_lemma[:-2]
                    if primary_lemma.endswith("ed") and len(primary_lemma) > 2
                    else None
                )

                ing_stem = (
                    primary_lemma[:-3]
                    if primary_lemma.endswith("ing") and len(primary_lemma) > 3
                    else None
                )

                is_story_grounded = (
                    primary_lemma in self._story_lemmas
                    or (ed_stem and ed_stem in self._story_lemmas)
                    or (ing_stem and ing_stem in self._story_lemmas)
                )

                is_inferred = provenance == NodeProvenance.INFERRED_FROM_STORY

                if (
                    node_type != NodeType.PROPERTY
                    and not is_story_grounded
                    and not is_inferred
                ):
                    return None

        # Legacy admit callback (additional filtering if provided)
        if admit is not None:
            admit_kwargs = admit_kwargs or {}
            if not admit(lemma=lemmas[0], node_type=node_type, **admit_kwargs):
                return None

        actual_text_l = (actual_text or "").lower()

        node = existing_node
        if node is None:
            node = Node(
                lemmas,
                actual_text_l,
                node_type,
                node_source,
                0,
                origin_sentence=origin_sentence,
                provenance=provenance or NodeProvenance.STORY_TEXT,
                node_role=node_role,
            )
            self.nodes.add(node)
            if mark_explicit and origin_sentence is not None:
                node.mark_explicit_in_sentence(origin_sentence)
        else:
            node.add_actual_text(actual_text_l)
            # Upgrade CONCEPT to PROPERTY if necessary
            if node.node_type != node_type:
                if node_type == NodeType.PROPERTY:
                    node.node_type = NodeType.PROPERTY
            # Update role if node exists but had no role and we're providing one
            if node.node_role is None and node_role is not None:
                node.node_role = node_role

            # ==========================================================================
            # PHASE 2: Mark existing node as explicit in current sentence
            # ==========================================================================
            # Only mark as explicit if:
            # 1. mark_explicit is True (node appears in sentence's dependency parse)
            # 2. origin_sentence is provided (we know which sentence we're in)
            #
            # INVARIANT: Carry-over nodes remain carry-over unless re-mentioned.
            if mark_explicit and origin_sentence is not None:
                # Enforce sentence-scoped explicit-node invariant
                if (
                    self._current_sentence_lemmas is not None
                    and primary_lemma not in self._current_sentence_lemmas
                ):
                    return None  # Reject explicit node not present in current sentence

                node.mark_explicit_in_sentence(origin_sentence)

        return node

    def get_node(self, lemmas: List[str]) -> Optional[Node]:
        for node in self.nodes:
            if node.lemmas == lemmas:
                return node
        return None

    def get_edge_by_nodes_and_label(
        self, source_node: Node, dest_node: Node, label: str
    ) -> Optional[Edge]:
        for edge in self.edges:
            if (
                edge.source_node == source_node
                and edge.dest_node == dest_node
                and edge.label == label
            ):
                return edge
        return None

    def get_edge(self, edge: Edge) -> Optional[Edge]:
        for other_edge in self.edges:
            if edge == other_edge:
                return other_edge
        return None

    def add_edge(
        self,
        source_node: Node,
        dest_node: Node,
        label: str,
        edge_visibility: int,
        created_at_sentence: Optional[int] = None,
        *,
        relation_class: Optional[RelationClass] = None,
        justification: Optional[Justification] = None,
        persona_influenced: bool = False,
        inferred: bool = False,
    ) -> Optional[Edge]:
        # Basic sanity checks (these don't block - just return None)
        if source_node == dest_node:
            return None
        # Safety net: reject edges with empty/whitespace-only labels
        if not label or not isinstance(label, str) or not label.strip():
            return None
        # ------------------------------------------------------------
        # Prevent inference from resurrecting decaying nodes
        # ------------------------------------------------------------
        if inferred:
            if (
                hasattr(source_node, "visibility_score")
                and source_node.visibility_score <= 0
            ) or (
                hasattr(dest_node, "visibility_score")
                and dest_node.visibility_score <= 0
            ):
                return None

            if created_at_sentence is not None:
                label_lower = label.lower().strip()
                source_key = tuple(source_node.lemmas)

                sentence_budget = self._edge_budget.setdefault(created_at_sentence, {})
                node_budget = sentence_budget.setdefault(source_key, {})

                current_count = node_budget.get(label_lower, 0)

                MAX_PER_LABEL = 3

                # ---------------------------------
                # SAFE grounding detection
                # ---------------------------------
                def _node_degree(node):
                    return sum(
                        1
                        for e in self.edges
                        if (e.source_node == node or e.dest_node == node)
                        and e.visibility_score > 0
                    )

                source_degree = _node_degree(source_node)
                dest_degree = _node_degree(dest_node)

                grounding_edge = source_degree == 0 or dest_degree == 0

                if not grounding_edge and current_count >= MAX_PER_LABEL:
                    return None

                node_budget[label_lower] = current_count + 1

        # PROPERTY constraint check (soft - records violation but allows edge)
        property_node = None
        concept_node = None
        if (
            source_node.node_type == NodeType.PROPERTY
            and dest_node.node_type == NodeType.CONCEPT
        ):
            property_node = source_node
            concept_node = dest_node
        elif (
            dest_node.node_type == NodeType.PROPERTY
            and source_node.node_type == NodeType.CONCEPT
        ):
            property_node = dest_node
            concept_node = source_node

        property_violation = None
        if property_node is not None:
            # Check if this property is already attached to a DIFFERENT concept
            for existing_edge in property_node.edges:
                other_node = (
                    existing_edge.dest_node
                    if existing_edge.source_node == property_node
                    else existing_edge.source_node
                )
                if (
                    other_node.node_type == NodeType.CONCEPT
                    and other_node != concept_node
                ):
                    property_violation = "PROPERTY node attached to multiple concepts"
                    break

        # ==========================================================================
        # NON-BLOCKING EDGE CREATION
        # ==========================================================================
        # Create edge regardless of ontology violations.
        # Violations are recorded as metadata, NOT used to block creation.
        if relation_class == RelationClass.CONNECTIVE:
            inferred = True
            persona_influenced = False
            skip_budget = True

        edge = Edge(
            source_node,
            dest_node,
            label,
            edge_visibility,
            relation_class=relation_class,
            justification=justification,
            persona_influenced=persona_influenced,
            inferred=inferred,
            active=True,
            created_at_sentence=created_at_sentence,
        )

        # CENTRALIZED ONTOLOGY DIAGNOSTICS (non-blocking)
        enforce_ontology_invariants(edge)

        # Record property violation if detected (non-blocking)
        if property_violation:
            edge.metadata.setdefault("ontology_violations", []).append(
                property_violation
            )

        if self.check_if_similar_edge_exists(edge, edge_visibility):
            return self.get_edge(edge)

        # ==========================================================================
        # CONNECTIVITY CHECK (non-blocking - records violation but allows edge)
        # ==========================================================================

        self.edges.add(edge)
        self._apply_structural_event_supersession(edge)
        if edge not in source_node.edges:
            source_node.edges.append(edge)
        if edge not in dest_node.edges:
            dest_node.edges.append(edge)

        return edge

    def _apply_structural_event_supersession(self, new_edge):
        """
        Abstract structural supersession:
        If a new edge causes a state change between a subject-object pair,
        remove previous state-preserving edges for that same pair.
        No lexical assumptions.
        """

        subject = new_edge.source_node
        object_ = new_edge.dest_node

        for edge in list(self.edges):
            if edge is new_edge:
                continue

            if edge.source_node == subject and edge.dest_node == object_:

                # Structural logic only
                if getattr(edge, "preserves_state", False) and getattr(
                    new_edge, "causes_state", False
                ):
                    self.remove_edge(edge)

        # Narrative collapse: kill supersedes fight relations
        # if "kill" in new_edge.label:
        # for edge in list(self.edges):
        #     if (
        #         edge.source_node == new_edge.source_node
        #         and edge.dest_node == new_edge.dest_node
        #         and "fight" in edge.label
        #     ):
        #         edge.visibility_score = 0
        #         edge.active = False

    def _would_maintain_connectivity(self, new_edge: Edge) -> bool:
        """
        Check if adding this edge would maintain graph connectivity.

        Uses CUMULATIVE structure, not sentence-local activity.
        For dependent relations, at least one endpoint must already exist
        in the cumulative memory graph.

        Returns:
            True if edge can be added without creating isolated component
            False if edge would create a disconnected component
        """
        # If graph is empty, first edge is always allowed
        if not self.edges:
            return True

        # Build CUMULATIVE graph structure
        # - Use ALL edges, not just active ones
        # - Exclude ATTRIBUTIVE edges (they don't maintain connectivity)
        # - Activity status is IGNORED - this is a memory check
        G = nx.Graph()
        for edge in self.edges:
            # Exclude ATTRIBUTIVE edges from connectivity graph
            if edge.relation_class == RelationClass.ATTRIBUTIVE:
                continue
            G.add_edge(edge.source_node, edge.dest_node)

        # Check if at least one endpoint already exists in cumulative memory
        source_connected = new_edge.source_node in G.nodes()
        dest_connected = new_edge.dest_node in G.nodes()

        # Dependent relations require at least one endpoint to be grounded
        # Allow inferred edges to introduce at most one new node
        # If graph is empty, first edge is allowed
        if G.number_of_nodes() == 0:
            return True

        # If both endpoints are new and graph already exists,
        # block creation of isolated component
        if not source_connected and not dest_connected:
            return False

        if new_edge.inferred:
            return True

        if getattr(new_edge, "inferred", False):
            return True

        # simulate addition and test connectivity
        G.add_edge(new_edge.source_node, new_edge.dest_node)
        return nx.is_connected(G)

    def check_if_similar_edge_exists(self, edge: Edge, edge_visibility: int) -> bool:
        """
        If an equivalent edge exists, softly reinforce it.
        DO NOT hard reset visibility (prevents immortal edges).
        """

        for other_edge in self.edges:

            same_nodes = (
                edge.source_node == other_edge.source_node
                and edge.dest_node == other_edge.dest_node
            )

            if not same_nodes:
                continue

            is_concept_property_pair = (
                edge.source_node.node_type == NodeType.CONCEPT
                and edge.dest_node.node_type == NodeType.PROPERTY
            ) or (
                edge.dest_node.node_type == NodeType.CONCEPT
                and edge.source_node.node_type == NodeType.PROPERTY
            )

            if (
                edge.label.strip().lower() == other_edge.label.strip().lower()
                and not edge.inferred
            ):
                # SOFT REINFORCEMENT (works for both inferred and non-inferred)
                other_edge.visibility_score = min(
                    edge_visibility, other_edge.visibility_score + 1
                )

                other_edge.active = True
                other_edge.mark_as_asserted(reset_score=False)

                return True

        return False

    def deactivate_all_edges(self) -> None:
        """
        Reset all edges at the start of a new sentence.
        This implements the "cumulative memory, sentence-local activation" rule.
        Per AMoC v4 paper: all edges become inactive at sentence start,
        then selectively activated through assertion or reactivation.
        Edges remain in memory but are marked inactive until reasserted/reactivated.
        """
        for edge in self.edges:
            edge.reset_for_sentence_start()

    # Maximum number of edges to reactivate per sentence (sparse reactivation)
    MAX_REACTIVATION_COUNT: int = 6

    def reactivate_memory_edges_within_distance(
        self,
        explicit_nodes: Set[Node],
        max_distance: int,
        current_sentence: int,
    ) -> Set[Edge]:

        if not explicit_nodes or max_distance < 1:
            return set()

        # BFS to find candidate edges within distance
        # CRITICAL (Paper-Aligned): Only CONCEPT nodes participate in BFS
        # PROPERTY nodes have no independent activation and don't propagate
        # SETTING nodes (locations/environments) are low-priority context nodes
        # that should NOT trigger reactivation of other edges
        # Use nodes connected by visible edges, not active edges
        memory_nodes = {e.source_node for e in self.edges} | {
            e.dest_node for e in self.edges
        }

        # NEW: allow inferred nodes to propagate activation
        inferred_nodes = {
            n for n in self.nodes if n.node_source == NodeSource.INFERENCE_BASED
        }

        concept_seeds = {
            n
            for n in explicit_nodes
            if n.node_type != NodeType.PROPERTY and not n.is_setting()
        }

        if not concept_seeds:
            return set()

        reachable_nodes: Dict[Node, int] = {n: 0 for n in concept_seeds}
        queue: deque = deque(concept_seeds)
        visited_edges: Set[Edge] = set()
        candidate_edges: list[tuple[int, Edge]] = []  # (distance, edge)

        while queue:
            node = queue.popleft()
            dist = reachable_nodes[node]

            if dist >= max_distance:
                continue

            for edge in node.edges:
                if edge in visited_edges:
                    continue
                visited_edges.add(edge)

                # Do not reactivate edges that have fully faded from memory
                # Temporal guard: prevent revival of very old edges
                # if edge.created_at_sentence is not None:
                #     if current_sentence - edge.created_at_sentence > 5:
                #         continue
                # Do not reactivate edges that have fully faded
                if edge.visibility_score <= 0:
                    continue

                # --------------------------------------------------
                # 1. PROPERTY EDGES (ATTRIBUTIVE)
                # --------------------------------------------------
                # Paper-aligned: PROPERTY edges do NOT reactivate
                if edge.relation_class == RelationClass.ATTRIBUTIVE:
                    continue

                # Skip edges that are already active (asserted this sentence)
                if edge.active:
                    continue

                # Collect as candidate for reactivation
                candidate_edges.append((dist, edge))

                # Continue BFS to neighbors (for finding more candidates)
                # CRITICAL: Skip PROPERTY and SETTING nodes in traversal
                # PROPERTY nodes have no independent activation
                # SETTING nodes are low-priority context that don't propagate activation
                neighbor = (
                    edge.dest_node if edge.source_node == node else edge.source_node
                )
                if (
                    neighbor.node_type == NodeType.PROPERTY
                    or neighbor.is_setting()
                    or (
                        neighbor.node_role is not None
                        and neighbor.node_role not in {NodeRole.ACTOR, NodeRole.OBJECT}
                    )
                ):
                    continue
                if neighbor not in reachable_nodes:
                    reachable_nodes[neighbor] = dist + 1
                    queue.append(neighbor)

        # SPARSE REACTIVATION: sort by distance and limit to MAX_REACTIVATION_COUNT
        # Prefer closer edges (smaller distance)
        candidate_edges.sort(key=lambda x: x[0])
        reactivated: Set[Edge] = set()

        if not candidate_edges:
            memory_edges = sorted(
                self.edges,
                key=lambda e: e.created_at_sentence or 0,
                reverse=True,
            )
            for edge in memory_edges:
                if edge.visibility_score > 0:
                    edge.mark_as_reactivated(reset_score=False)
                    reactivated.add(edge)
                    break

        for dist, edge in candidate_edges[: self.MAX_REACTIVATION_COUNT]:
            edge.mark_as_reactivated(reset_score=True)
            reactivated.add(edge)

        return reactivated

    def decay_inactive_edges(self) -> None:
        for edge in list(self.edges):
            if not edge.active:
                edge.reduce_visibility()

    def get_active_subgraph(
        self,
    ) -> Tuple[Set[Node], Set[Edge]]:

        active_edges: Set[Edge] = {
            e for e in self.edges if e.active and e.visibility_score > 0
        }

        active_nodes: Set[Node] = {e.source_node for e in active_edges} | {
            e.dest_node for e in active_edges
        }

        return active_nodes, active_edges

    def get_active_triplets_with_scores(
        self,
    ) -> List[Tuple[str, str, str, bool, int]]:
        """
        Get triplets from active edges with their activation scores.

        Returns list of (source_text, label, dest_text, is_active, activation_score)
        for use in plotting with variable thickness/alpha.
        """
        triplets = []
        for edge in self.edges:
            triplets.append(
                (
                    edge.source_node.get_text_representer(),
                    edge.label,
                    edge.dest_node.get_text_representer(),
                    edge.active,
                    edge.activation_score,
                )
            )
        return triplets

    @staticmethod
    def canonicalize_relation_label(label: str) -> str:
        """
        Canonicalize relation labels before edge creation.
        Removes clear parser artifacts while preserving valid verb phrases.

        Per AMoC v4 paper (Figures 2-4), edge labels should be full verb phrases:
        - "rode through"
        - "was kidnapping"
        - "wanted to free"
        - "is unfamiliar with"

        This function is CONSERVATIVE - it only removes clear artifacts.
        The more aggressive normalization happens in _normalize_edge_label().
        """
        if not label or not isinstance(label, str):
            return ""

        # Strip whitespace
        label = label.strip()
        if not label:
            return ""

        # Remove common parser prefixes/artifacts (colon-based)
        prefixes_to_remove = [
            "nsubj:",
            "dobj:",
            "pobj:",
            "prep:",
            "amod:",
            "advmod:",
            "ROOT:",
            "VERB:",
            "NOUN:",
            "ADJ:",
            "dep:",
            "compound:",
            "agent:",
            "xcomp:",
            "ccomp:",
            "aux:",
            "auxpass:",
        ]
        for prefix in prefixes_to_remove:
            if label.lower().startswith(prefix.lower()):
                label = label[len(prefix) :]

        # Remove trailing punctuation
        label = re.sub(r"[^\w\s]+$", "", label)
        label = label.strip()

        # Normalize whitespace (but preserve spaces between words)
        label = re.sub(r"\s+", " ", label)

        # CONSERVATIVE: Only reject clearly malformed labels
        # - Labels with 3+ repeated characters (like "killtt" -> corruption)
        # - Labels that are pure noise (no vowels in any word)
        if len(label) > 0:
            # Detect repeated character corruption
            if re.search(r"(.)\1{2,}", label):
                # Clean up repeated trailing consonants: "kidnapp" -> "kidnap"
                label = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1+$", r"\1", label)

            # Check each word for vowel presence (skip very short words)
            words = label.split()
            cleaned_words = []
            for word in words:
                # Skip words that are too short to check
                if len(word) <= 2:
                    cleaned_words.append(word.lower())
                    continue
                # Reject words without vowels (corruption)
                if not re.search(r"[aeiou]", word.lower()):
                    continue
                cleaned_words.append(word.lower())

            if not cleaned_words:
                return ""
            label = " ".join(cleaned_words)

        # Final: lowercase and return
        label = label.lower().strip()

        # Minimum length check (allow 2-char verbs like "be", "go", "do")
        if len(label) < 2:
            return ""

        return label

    def bfs_from_activated_nodes(
        self,
        activated_nodes: List[Node],
        direction: str = "both",
    ) -> Dict[Node, int]:
        """
        Compute shortest-path distances from activated nodes via BFS.

        Args:
            activated_nodes: Starting nodes for BFS
            direction: How to traverse edges:
                - "both": Follow edges in both directions (default, for distance computation)
                - "outgoing": Only follow edges where current node is source (A → B)
                - "incoming": Only follow edges where current node is dest (A ← B)

        Returns:
            Dict mapping each reachable node to its distance from nearest activated node.

        Per AMoC v4: Activation distance is computed bidirectionally (semantic edges
        connect concepts regardless of direction). Direction matters for meaning,
        not for activation propagation.

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are EXCLUDED from BFS traversal and distance computation.
        Per paper: PROPERTY nodes have no independent activation and don't propagate
        activation to/from other nodes. Only CONCEPT nodes participate in BFS.
        """
        distances = {}
        # Filter out PROPERTY nodes from starting set
        # Per paper: only CONCEPT nodes can be activation seeds
        concept_seeds = [
            node for node in activated_nodes if node.node_type != NodeType.PROPERTY
        ]

        queue = deque([(node, 0) for node in concept_seeds])
        while queue:
            curr_node, curr_distance = queue.popleft()
            if curr_node not in distances:
                distances[curr_node] = curr_distance
                for edge in curr_node.edges:
                    if not edge.active:
                        continue

                    # Determine which neighbor to visit based on direction mode
                    next_node = None
                    if direction == "both":
                        # Bidirectional: follow edge regardless of direction
                        next_node = (
                            edge.dest_node
                            if edge.source_node == curr_node
                            else edge.source_node
                        )
                    elif direction == "outgoing":
                        # Only follow if current node is the source
                        if edge.source_node == curr_node:
                            next_node = edge.dest_node
                    elif direction == "incoming":
                        # Only follow if current node is the destination
                        if edge.dest_node == curr_node:
                            next_node = edge.source_node

                    # CRITICAL: Skip PROPERTY nodes in BFS traversal
                    # Per paper: PROPERTY nodes don't propagate activation
                    if (
                        next_node is not None
                        and next_node.node_type == NodeType.PROPERTY
                    ):
                        continue

                    if next_node is not None:
                        queue.append((next_node, curr_distance + 1))
        return distances

    def set_nodes_score_based_on_distance_from_active_nodes(
        self, activated_nodes: List[Node]
    ) -> None:
        distances_to_activated_nodes = self.bfs_from_activated_nodes(activated_nodes)

        for node in self.nodes:
            # PROPERTY nodes are distance-agnostic (paper behavior)
            if node.node_type == NodeType.PROPERTY:
                continue  # do NOT overwrite score

            if node in distances_to_activated_nodes:
                node.score = distances_to_activated_nodes[node]
            else:
                # Instead of killing it (score=100),
                # gradually increase distance score
                node.score = min(node.score + 1, 100)

    def get_word_lemma_score(self, word_lemmas: List[str]) -> Optional[float]:
        for node in self.nodes:
            if node.lemmas == word_lemmas:
                return node.score
        return None

    def get_top_k_nodes(self, nodes: List[Node], k: int) -> List[Node]:
        return sorted(nodes, key=lambda node: node.score)[:k]

    def get_top_concepts_nodes(self, k: int) -> List[Node]:
        nodes = [node for node in self.nodes if node.node_type == NodeType.CONCEPT]
        return self.get_top_k_nodes(nodes, k)

    def get_top_text_based_concepts(self, k: int) -> List[Node]:
        nodes = [
            node
            for node in self.nodes
            if node.node_type == NodeType.CONCEPT
            and node.node_source == NodeSource.TEXT_BASED
        ]
        return self.get_top_k_nodes(nodes, k)

    def get_active_nodes(
        self, score_threshold: int, only_text_based: bool = False
    ) -> List[Node]:
        """
        Get nodes within the score threshold (close to active nodes).

        CRITICAL (Paper-Aligned):
        PROPERTY nodes are EXCLUDED from score-based selection.
        Per paper: PROPERTY nodes have no independent activation.
        They only appear in the active graph if they have an active property edge.
        Score-based selection applies ONLY to CONCEPT nodes.
        """
        return [
            node
            for node in self.nodes
            if node.score <= score_threshold
            and node.node_type != NodeType.PROPERTY  # Per paper: no PROPERTY carry-over
            and (not only_text_based or node.node_source == NodeSource.TEXT_BASED)
        ]

    def get_explicit_nodes_for_sentence(self, sentence_id: int):
        return [
            node for node in self.nodes if node.is_explicit_in_sentence(sentence_id)
        ]

    def get_nodes_str(self, nodes: List[Node]) -> str:
        nodes_str = ""
        for node in sorted(nodes, key=lambda node: node.score):
            nodes_str += (
                "- "
                + f"{node.get_text_representer()} (type: {node.node_type}) (score: {node.score})"
                + "\n"
            )
        return nodes_str

    def get_edges_str(
        self, nodes: List[Node], only_text_based: bool = False, only_active: bool = True
    ) -> Tuple[str, List[Edge]]:
        used_edges = set()
        edges_str = ""
        count = 1
        for node in sorted(nodes, key=lambda node: node.score):
            for edge in node.edges:
                if only_active and edge.active == False:
                    continue
                if edge not in used_edges:
                    if not only_text_based:
                        edges_str += (
                            f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                            + "\n"
                        )
                        used_edges.add(edge)
                        count += 1
                    else:
                        if (
                            edge.source_node.node_source == NodeSource.TEXT_BASED
                            and edge.dest_node.node_source == NodeSource.TEXT_BASED
                        ):
                            edges_str += (
                                f"{count}. {edge.source_node.get_text_representer()} - {edge.label} (edge) - {edge.dest_node.get_text_representer()}"
                                + "\n"
                            )
                            used_edges.add(edge)
                            count += 1
        return edges_str, list(used_edges)

    def get_active_graph_repr(self) -> str:
        edges = [edge for edge in self.edges if edge.active]
        nodes = set()
        for edge in edges:
            nodes.add(edge.source_node)
            nodes.add(edge.dest_node)
        s = "nodes:\n"
        for node in nodes:
            s += str(node) + "\n"
        s += "\nedges:\n"
        for edge in edges:
            s += str(edge) + "\n"
        return s

    # ==========================================================================
    # CONNECTIVITY ENFORCEMENT LOGIC
    # ==========================================================================
    # This section contains all connectivity-related methods, isolated for clarity.
    #
    # CONNECTIVITY FLOW:
    # 1. check_active_connectivity() - Detect if graph is disconnected
    # 2. get_disconnected_components() - Get components that need connecting
    # 3. ensure_active_connectivity() - Try to connect using existing edges
    # 4. If still disconnected -> caller triggers secondary LLM call
    # ==========================================================================

    def check_active_connectivity(self) -> bool:
        """
        Check if the active graph is connected.

        This is the primary method for connectivity detection.
        Should be called BEFORE plotting to determine if intervention is needed.

        Returns:
            True if connected (or empty/single node), False if disconnected
        """
        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return True  # Empty graph is trivially connected

        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        if G_active.number_of_nodes() <= 1:
            return True

        return nx.is_connected(G_active)

    def get_disconnected_components(
        self,
        focus_nodes: Set[Node],
    ) -> Tuple[List[Set[Node]], int]:
        """
        Get disconnected components and identify the focus component.

        Use this to determine which nodes need connecting and to which component.
        This information can be passed to a secondary LLM call.

        Args:
            focus_nodes: Nodes that should be in the "main" component

        Returns:
            Tuple of:
            - List of connected components (each is a set of nodes)
            - Index of the focus component (containing focus_nodes)

        If graph is connected, returns ([all_nodes], 0).
        """
        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return ([], -1)

        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        if G_active.number_of_nodes() <= 1:
            return (
                ([set(G_active.nodes())], 0)
                if G_active.number_of_nodes() == 1
                else ([], -1)
            )

        components = [set(c) for c in nx.connected_components(G_active)]

        if len(components) <= 1:
            return (components, 0)

        # Find which component contains focus nodes
        focus_component_idx = 0
        for idx, comp in enumerate(components):
            if any(n in comp for n in focus_nodes):
                focus_component_idx = idx
                break

        return (components, focus_component_idx)

    def get_nodes_needing_connection(
        self,
        focus_nodes: Set[Node],
    ) -> List[Tuple[Node, Node]]:
        """
        Get pairs of nodes that need edges to restore connectivity.

        For each disconnected component, returns a representative node pair
        (one node from the component, one from the focus component) that
        could be connected by a new edge.

        Use this to prepare the secondary LLM call prompt.

        Args:
            focus_nodes: Nodes that define the "main" component

        Returns:
            List of (isolated_node, focus_node) pairs that need connecting.
            Empty list if graph is already connected.
        """
        components, focus_idx = self.get_disconnected_components(focus_nodes)

        if len(components) <= 1:
            return []  # Already connected

        focus_comp = components[focus_idx]
        pairs_needing_connection = []

        for idx, comp in enumerate(components):
            if idx == focus_idx:
                continue

            # Pick a representative node from each side
            # Prefer CONCEPT nodes over PROPERTY nodes
            isolated_node = None
            for n in comp:
                if n.node_type == NodeType.CONCEPT:
                    isolated_node = n
                    break
            if isolated_node is None:
                isolated_node = next(iter(comp))

            focus_node = None
            for n in focus_comp:
                if n in focus_nodes and n.node_type == NodeType.CONCEPT:
                    focus_node = n
                    break
            if focus_node is None:
                for n in focus_comp:
                    if n.node_type == NodeType.CONCEPT:
                        focus_node = n
                        break
            if focus_node is None:
                focus_node = next(iter(focus_comp))

            pairs_needing_connection.append((isolated_node, focus_node))

        return pairs_needing_connection

    def ensure_active_connectivity(
        self,
        focus_nodes: Set[Node],
        carryover_focus_nodes: Optional[Set[Node]] = None,
    ) -> Set[Edge]:
        """
        Ensure the active graph remains connected.

        STEP 1 of connectivity enforcement: Try to connect using EXISTING edges.

        If the active graph becomes disconnected:
        1. Identify focus nodes (explicit entities in current sentence + carry-over focus)
        2. Compute connected components of the active graph
        3. If multiple components exist, find minimum set of existing memory edges
           that connect them
        4. Promote those edges to active as "connectors"

        Connector edge constraints (CRITICAL):
        - Must already exist in the cumulative graph
        - Must NOT be ATTRIBUTIVE edges
        - Must NOT be counted as asserted or reactivated
        - Do NOT increase activation scores
        - Are NOT eligible for inference
        - Exist only to preserve structural connectivity

        IMPORTANT: If this method returns an empty set but the graph is still
        disconnected (check with check_active_connectivity()), caller should
        trigger a secondary LLM call to create new edges. Use
        get_nodes_needing_connection() to get the pairs that need connecting.

        Returns:
            Set of edges promoted as connectors.
        """
        # Build active subgraph
        active_edges = [e for e in self.edges if e.active]
        if not active_edges:
            return set()

        # Build undirected graph of active edges
        G_active = nx.Graph()
        for edge in active_edges:
            G_active.add_edge(edge.source_node, edge.dest_node, edge=edge)

        # Check if already connected
        if G_active.number_of_nodes() <= 1 or nx.is_connected(G_active):
            return set()

        # Graph is disconnected - find components
        components = list(nx.connected_components(G_active))
        if len(components) <= 1:
            return set()

        # Identify focus component (contains focus nodes)
        all_focus = focus_nodes | (carryover_focus_nodes or set())
        focus_component_idx = 0
        for idx, comp in enumerate(components):
            if any(n in comp for n in all_focus):
                focus_component_idx = idx
                break

        # Build full cumulative graph for finding paths
        G_cumulative = nx.Graph()
        for edge in self.edges:
            if edge.relation_class == RelationClass.ATTRIBUTIVE:
                continue
            G_cumulative.add_edge(edge.source_node, edge.dest_node, edge=edge)

        promoted_connectors: Set[Edge] = set()

        # For each non-focus component, find shortest path to focus component
        focus_comp_nodes = components[focus_component_idx]
        for idx, comp in enumerate(components):
            if idx == focus_component_idx:
                continue

            # Find shortest path in cumulative graph between any node in comp
            # and any node in focus component
            best_path = None
            best_path_len = float("inf")

            for src in comp:
                # Skip if src not in cumulative graph (only has property edges)
                if src not in G_cumulative:
                    continue
                for tgt in focus_comp_nodes:
                    if src == tgt:
                        continue
                    # Skip if tgt not in cumulative graph (only has property edges)
                    if tgt not in G_cumulative:
                        continue
                    try:
                        path = nx.shortest_path(G_cumulative, src, tgt)
                        if len(path) < best_path_len:
                            best_path = path
                            best_path_len = len(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

            if best_path is None:
                # No path exists in cumulative graph - caller needs to use
                # secondary LLM call to create new edges
                continue

            # Promote edges along the path as connectors
            for i in range(len(best_path) - 1):
                node_a = best_path[i]
                node_b = best_path[i + 1]

                # Find the edge in cumulative graph
                edge_data = G_cumulative.get_edge_data(node_a, node_b)
                if edge_data is None:
                    continue

                edge = edge_data.get("edge")
                if edge is None:
                    continue

                # STRICT: Never use ATTRIBUTIVE edges as connectors
                if edge.relation_class == RelationClass.ATTRIBUTIVE:
                    continue

                # Only promote if not already active
                if not edge.active:
                    edge.structural = True
                    edge.active = True
                    # edge.mark_as_reactivated(reset_score=True)
                    promoted_connectors.add(edge)

            if __debug__:
                G_check = nx.Graph()
                for e in self.edges:
                    if e.active:
                        G_check.add_edge(e.source_node, e.dest_node)
                if G_check.number_of_nodes() > 1:
                    # Note: might still be disconnected if no path was found
                    pass

        return promoted_connectors

    def get_active_edges_by_role(self) -> dict[str, Set[Edge]]:
        """
        Get active edges grouped by their activation role.

        Returns dict with keys:
        - "asserted": edges asserted this sentence
        - "reactivated": edges reactivated from memory
        """
        result: dict[str, Set[Edge]] = {
            "asserted": set(),
            "reactivated": set(),
        }

        for edge in self.edges:
            if not edge.active:
                continue
            if edge.is_asserted():
                result["asserted"].add(edge)
            elif edge.is_reactivated():
                result["reactivated"].add(edge)

        return result

    def __str__(self) -> str:
        return "nodes: {}\n\nedges: {}".format(
            "\n".join([str(x) for x in self.nodes]),
            "\n".join([str(x) for x in self.edges]),
        )

    def __repr__(self) -> str:
        return self.__str__()

    # ==========================================================================
    # AMoCv4 HARD CONSTRAINTS - Surface-relation format enforcement
    # ==========================================================================
    FORBIDDEN_EDGE_LABELS = {"agent_of", "target_of", "patient_of", "relation"}

    def validate_amocv4_constraints(self) -> list[str]:
        """
        Check AMoCv4 surface-relation format constraints.

        Constraints checked:
        1. No agent_of, target_of, patient_of, or role-based edges
        2. All verbs must be represented as direct labeled edges between entities
        3. All attributes must be represented using the relation 'is'

        Returns:
            List of violation strings, empty if no violations.
            NEVER raises or blocks execution.
        """
        violations = []
        for edge in self.edges:
            if edge.label in self.FORBIDDEN_EDGE_LABELS:
                violations.append(
                    f"AMoCv4: Forbidden edge label '{edge.label}': "
                    f"{edge.source_node.get_text_representer()} -> {edge.dest_node.get_text_representer()}"
                )
        return violations

    def sanity_check_readable_triplets(self) -> list[str]:
        """
        AMoCv4 sanity check: Every edge must be readable as a simple sentence fragment.

        Returns:
            List of violation strings, empty if no violations.
            NEVER raises or blocks execution.
        """
        violations = []
        for edge in self.edges:
            subj = edge.source_node.get_text_representer()
            verb = edge.label
            obj = edge.dest_node.get_text_representer()

            # Basic sanity: all parts must be non-empty
            if not subj or not verb or not obj:
                violations.append(
                    f"AMoCv4: Edge has empty component: '{subj}' --{verb}--> '{obj}'"
                )

            # Forbidden patterns
            if verb in self.FORBIDDEN_EDGE_LABELS:
                violations.append(
                    f"AMoCv4: Edge uses forbidden label '{verb}': '{subj}' --{verb}--> '{obj}'"
                )

        return violations

    def sanity_check_provenance(
        self,
        story_lemmas: set,
        persona_only_lemmas: set,
    ) -> list:
        """
        AMoC v4 PROVENANCE SANITY CHECK: Detect potential persona leakage.

        Per AMoC v4 paper: Nodes must come from STORY TEXT only.
        Persona influences salience (weights), never content (nodes/edges).

        Args:
            story_lemmas: Set of lemmas from the story text
            persona_only_lemmas: Set of lemmas unique to persona (not in story)

        Returns:
            List of warning strings for any detected violations.
            Empty list if all nodes pass provenance check.
        """
        warnings = []

        for node in self.nodes:
            # Check each lemma in the node
            for lemma in node.lemmas:
                lemma_lower = lemma.lower()

                # CRITICAL CHECK: Node lemma appears ONLY in persona
                if lemma_lower in persona_only_lemmas:
                    warnings.append(
                        f"PROVENANCE VIOLATION: Node '{node.get_text_representer()}' "
                        f"contains lemma '{lemma_lower}' which appears ONLY in persona, "
                        f"not in story text. Provenance: {node.provenance}"
                    )

                # SOFT CHECK: Node lemma not found in story
                # (This might be OK for inferred nodes, but worth flagging)
                elif lemma_lower not in story_lemmas:
                    if node.provenance != NodeProvenance.INFERRED_FROM_STORY:
                        warnings.append(
                            f"PROVENANCE WARNING: Node '{node.get_text_representer()}' "
                            f"contains lemma '{lemma_lower}' not found in story text. "
                            f"Provenance: {node.provenance}"
                        )

        return warnings

    def remove_edge(self, edge: Edge) -> None:
        if edge in self.edges:
            self.edges.remove(edge)
        if edge.source_node and edge in edge.source_node.edges:
            edge.source_node.edges.remove(edge)
        if edge.dest_node and edge in edge.dest_node.edges:
            edge.dest_node.edges.remove(edge)

    def validate_persona_ontology_invariant(self) -> list[str]:
        """
        Check that persona did not modify structural ontology.

        Per Recommendation 1: Persona may affect expression (labels),
        NOT ontology (relation_class, justification).

        Returns:
            List of violation strings, empty if no violations.
            NEVER raises or blocks execution.
        """
        violations = []
        for edge in self.edges:
            # Call the non-blocking version which records to metadata
            assert_persona_did_not_modify_ontology(edge)
            # Collect any violations recorded on this edge
            edge_violations = edge.metadata.get("ontology_violations", [])
            for v in edge_violations:
                if "Persona" in v or "REC1" in v:
                    violations.append(
                        f"{edge.source_node.get_text_representer()} --{edge.label}--> "
                        f"{edge.dest_node.get_text_representer()}: {v}"
                    )
        return violations

    def validate_event_mediation_invariant(self) -> list[str]:
        """
        Check that all EVENTIVE edges are properly mediated by EVENT nodes.

        Per Recommendation 2: EVENTIVE relations must be mediated by EVENT nodes.
        Direct EVENTIVE edges between non-EVENT nodes are violations.

        Pattern: actor --participates_in--> EVENT --affects--> object

        Returns:
            List of violation strings, empty if no violations.
            NEVER raises or blocks execution.
        """
        violations = []
        for edge in self.edges:
            if edge.relation_class == RelationClass.EVENTIVE:
                if (
                    edge.source_node.node_type != NodeType.EVENT
                    and edge.dest_node.node_type != NodeType.EVENT
                ):
                    violations.append(
                        f"REC2: EVENTIVE edge not mediated: "
                        f"{edge.source_node.get_text_representer()} --{edge.label}--> "
                        f"{edge.dest_node.get_text_representer()} "
                        f"({edge.source_node.node_type} -> {edge.dest_node.node_type})"
                    )

            # Also check EVENT node attachment constraint
            is_event_involved = (
                edge.source_node.node_type == NodeType.EVENT
                or edge.dest_node.node_type == NodeType.EVENT
            )
            if is_event_involved:
                if edge.relation_class not in {
                    RelationClass.EVENTIVE,
                    RelationClass.CONNECTIVE,
                }:
                    violations.append(
                        f"REC2: EVENT node has invalid relation_class {edge.relation_class}: "
                        f"{edge.source_node.get_text_representer()} --{edge.label}--> "
                        f"{edge.dest_node.get_text_representer()}"
                    )

        return violations

    def collect_all_violations(self) -> list[str]:
        """
        Collect all ontology violations from all edges.

        Returns:
            List of all violation strings across all edges.
            NEVER raises or blocks execution.
        """
        violations = []
        for edge in self.edges:
            edge_violations = edge.metadata.get("ontology_violations", [])
            for v in edge_violations:
                violations.append(
                    f"{edge.source_node.get_text_representer()} --{edge.label}--> "
                    f"{edge.dest_node.get_text_representer()}: {v}"
                )
        return violations

    # ==========================================================================
    # PHASE 3: STRUCTURAL DE-COAGULATION (Edge Filtering for Plotting)
    # ==========================================================================
    # Not all edges deserve to be plotted. Filter out:
    # - CONNECTIVE edges (structural, not semantic)
    # - INFERRED edges (not explicit in text)
    # - PERSONA_INFLUENCED edges (persona-driven, not story-core)
    MAX_EDGES_PER_NODE: int = 5  # Soft cap on edges per node for visualization

    def get_edges_for_plotting(
        self,
        *,
        exclude_connective: bool = True,
        exclude_inferred: bool = True,
        exclude_persona_influenced: bool = True,
        active_only: bool = True,
    ) -> List[Edge]:
        """
        Get edges filtered for plotting (Phase 3 - De-Coagulation).

        This method filters edges to prevent visual clutter. It does NOT
        modify the graph structure - only returns a filtered view.

        Args:
            exclude_connective: Exclude CONNECTIVE relation_class edges
            exclude_inferred: Exclude inferred edges (edge.inferred=True)
            exclude_persona_influenced: Exclude persona-influenced edges
            active_only: Only include active edges

        Returns:
            List of edges suitable for plotting.
            Graph structure is UNCHANGED.
        """
        plot_edges = []
        for edge in self.edges:
            # Filter by active status
            if active_only and not edge.active:
                continue

            # PHASE 3: Exclude CONNECTIVE edges (structural, not semantic)
            if exclude_connective and edge.relation_class == RelationClass.CONNECTIVE:
                continue

            # PHASE 3: Exclude INFERRED edges (not explicit in text)
            if exclude_inferred and edge.inferred:
                continue

            # PHASE 3: Exclude PERSONA_INFLUENCED edges (persona-driven)
            if exclude_persona_influenced and edge.persona_influenced:
                continue

            plot_edges.append(edge)

        return plot_edges

    def get_edges_with_degree_cap(
        self,
        edges: List[Edge],
        max_edges_per_node: Optional[int] = None,
    ) -> List[Edge]:
        """
        Apply degree cap per node for visualization (Phase 3 - De-Coagulation).

        When a node has too many edges, keep only the most important ones:
        1. Prefer EVENTIVE over ATTRIBUTIVE
        2. Prefer TEXTUAL justification over IMPLIED

        This is a VISUALIZATION filter only - graph structure is UNCHANGED.

        Args:
            edges: List of edges to filter
            max_edges_per_node: Maximum edges per node (default: MAX_EDGES_PER_NODE)

        Returns:
            Filtered list of edges respecting degree cap.
        """
        if max_edges_per_node is None:
            max_edges_per_node = self.MAX_EDGES_PER_NODE

        # Count edges per node
        node_edge_count: Dict[Node, List[Edge]] = {}
        for edge in edges:
            if edge.source_node not in node_edge_count:
                node_edge_count[edge.source_node] = []
            if edge.dest_node not in node_edge_count:
                node_edge_count[edge.dest_node] = []
            node_edge_count[edge.source_node].append(edge)
            node_edge_count[edge.dest_node].append(edge)

        # Sort edges by priority for each node
        def edge_priority(edge: Edge) -> Tuple[int, int]:
            """Higher priority = lower number (sorted first)."""
            # Priority 1: EVENTIVE > STATIVE > ATTRIBUTIVE
            relation_priority = {
                RelationClass.EVENTIVE: 0,
                RelationClass.STATIVE: 1,
                RelationClass.ATTRIBUTIVE: 2,
                RelationClass.CONNECTIVE: 3,
            }
            rel_score = relation_priority.get(edge.relation_class, 2)

            # Priority 2: TEXTUAL > IMPLIED > CONNECTIVE
            justification_priority = {
                Justification.TEXTUAL: 0,
                Justification.IMPLIED: 1,
                Justification.CONNECTIVE: 2,
            }
            just_score = justification_priority.get(edge.justification, 1)

            return (rel_score, just_score)

        # Keep track of which edges to include
        edges_to_keep: Set[Edge] = set()

        for node, node_edges in node_edge_count.items():
            if len(node_edges) <= max_edges_per_node:
                # Under cap - keep all
                edges_to_keep.update(node_edges)
            else:
                # Over cap - keep best ones
                sorted_edges = sorted(node_edges, key=edge_priority)
                edges_to_keep.update(sorted_edges[:max_edges_per_node])

        return [e for e in edges if e in edges_to_keep]

    def get_plot_ready_edges(
        self,
        *,
        active_only: bool = True,
        apply_degree_cap: bool = True,
        max_edges_per_node: Optional[int] = None,
    ) -> List[Edge]:
        """
        Get edges ready for plotting with all Phase 3 filters applied.

        Convenience method combining:
        1. get_edges_for_plotting() - excludes CONNECTIVE, INFERRED, PERSONA_INFLUENCED
        2. get_edges_with_degree_cap() - applies soft degree cap

        Args:
            active_only: Only include active edges
            apply_degree_cap: Whether to apply degree cap
            max_edges_per_node: Maximum edges per node

        Returns:
            List of edges ready for plotting.
            Graph structure is UNCHANGED.
        """
        # --- PATCH 4: Debug mode override ---
        if getattr(self, "_debug_no_filter", False):
            return list(self.edges)

        # Step 1: Filter out non-semantic edges
        filtered = self.get_edges_for_plotting(
            exclude_connective=True,
            exclude_inferred=True,
            exclude_persona_influenced=True,
            active_only=active_only,
        )

        # Step 2: Apply degree cap if requested
        if apply_degree_cap:
            filtered = self.get_edges_with_degree_cap(
                filtered,
                max_edges_per_node=max_edges_per_node,
            )

        return filtered

    def check_cumulative_connectivity(self):
        G = nx.Graph()
        for edge in self.edges:
            G.add_edge(edge.source_node, edge.dest_node)
        if G.number_of_nodes() <= 1:
            return True
        return nx.is_connected(G)
