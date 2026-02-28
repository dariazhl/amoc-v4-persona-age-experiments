from typing import TYPE_CHECKING, Optional, List, Set
import logging

if TYPE_CHECKING:
    from amoc.graph.graph import Graph
    from amoc.graph.node import Node, NodeType, NodeSource, NodeProvenance
    from spacy.tokens import Span


class NodeOps:
    def __init__(
        self,
        graph_ref: "Graph",
        spacy_nlp,
        story_lemmas: Set[str],
        persona_only_lemmas: Set[str],
        max_distance_from_active_nodes: int,
        debug: bool = False,
    ):
        self._graph = graph_ref
        self._spacy_nlp = spacy_nlp
        self._story_lemmas = story_lemmas
        self._persona_only_lemmas = persona_only_lemmas
        self._max_distance = max_distance_from_active_nodes
        self._debug = debug
        self._ever_admitted_nodes: Set[str] = set()
        self._layout_depth = 3
        self._has_active_attachment_fn = None
        self._canonicalize_and_classify_fn = None

    def set_callbacks(
        self,
        has_active_attachment_fn: callable,
        canonicalize_and_classify_fn: callable,
    ):
        self._has_active_attachment_fn = has_active_attachment_fn
        self._canonicalize_and_classify_fn = canonicalize_and_classify_fn

    def admit_node(
        self,
        lemma: str,
        node_type: "NodeType",
        provenance: str,
        sent: Optional["Span"] = None,
    ) -> bool:
        from amoc.graph.node import NodeType

        lemma = (lemma or "").lower().strip()

        if not lemma:
            return False

        if len(lemma) < 4 and lemma not in self._story_lemmas:
            return False

        # Hard provenance guard
        if provenance in {
            "LLM_PROMPT",
            "GRAPH_SERIALIZATION",
            "CSV",
            "PLOTTING",
            "META",
        }:
            return False

        # Explicit nodes cannot be verbs
        if provenance == "STORY_EXPLICIT":
            if sent is None:
                return False

            token_matches = [tok for tok in sent if tok.lemma_.lower() == lemma]

            if not token_matches:
                return False

            for tok in token_matches:
                if tok.pos_ in {"VERB", "AUX"}:
                    return False

        # STORY_EXPLICIT admission
        if provenance == "STORY_EXPLICIT":
            if lemma not in self._story_lemmas:
                if lemma.endswith("s") and lemma[:-1] in self._story_lemmas:
                    pass
                else:
                    return False

        # Property grounding check
        if node_type == NodeType.PROPERTY:
            if sent is None:
                return False

            grounded = any(
                tok.lemma_.lower() == lemma
                and tok.pos_ == "ADJ"
                and tok.dep_ in {"amod", "acomp", "attr"}
                for tok in sent
            )

            if not grounded:
                return False

        # Inference admission
        is_story_grounded = lemma in self._story_lemmas

        is_allowed_inference = (
            provenance in {"INFERRED_RELATION", "INFERENCE_BASED"}
            and is_story_grounded
            and self._has_active_attachment_fn
            and self._has_active_attachment_fn(lemma)
        )

        if provenance != "STORY_EXPLICIT":
            if not is_allowed_inference:
                return False

        # Track new node admission
        is_new = lemma not in self._ever_admitted_nodes
        self._ever_admitted_nodes.add(lemma)

        if is_new:
            total_nodes = len(self._ever_admitted_nodes)

            if total_nodes > 40:
                self._layout_depth = max(self._layout_depth, 6)
            elif total_nodes > 25:
                self._layout_depth = max(self._layout_depth, 5)
            elif total_nodes > 12:
                self._layout_depth = max(self._layout_depth, 4)

        return True

    # =========================================================
    # PROVENANCE VALIDATION
    # =========================================================

    def validate_node_provenance(
        self,
        lemma: str,
        current_sentence_text: Optional[str] = None,
        *,
        allow_bootstrap: bool = False,
    ) -> bool:
        """
        Validate that a node has proper provenance.
        Per AMoC v4 paper: Nodes must come from STORY TEXT only.
        """
        from amoc.nlp.spacy_utils import get_semantic_class

        lemma_lower = lemma.lower()

        # HARD GATE 1: Reject persona-only lemmas
        if lemma_lower in self._persona_only_lemmas:
            if self._debug:
                logging.debug(
                    f"PROVENANCE GATE: Rejected persona-only lemma '{lemma_lower}'"
                )
            return False

        # HARD GATE 2: Must appear in story text
        if lemma_lower in self._story_lemmas:
            return True

        # Check current sentence if provided
        if current_sentence_text:
            sent_doc = self._spacy_nlp(current_sentence_text)
            sent_lemmas = {tok.lemma_.lower() for tok in sent_doc if tok.is_alpha}
            if lemma_lower in sent_lemmas:
                return True

        # Graph grounding: Allow concepts that already exist
        existing_node = self._graph.get_node([lemma_lower])
        if existing_node is not None:
            if self._debug:
                logging.debug(f"PROVENANCE GATE: Graph grounding for '{lemma_lower}'")
            return True

        candidate_class = get_semantic_class(lemma_lower)
        if candidate_class is not None:
            for node in self._graph.get_active_nodes(
                self._max_distance,
                only_text_based=False,
            ):
                node_class = get_semantic_class(node.get_text_representer())
                if node_class == candidate_class:
                    if self._debug:
                        logging.debug(
                            f"PROVENANCE GATE: Semantic-class grounding "
                            f"'{lemma_lower}' via class '{candidate_class}'"
                        )
                    return True

        # Bootstrap path
        if allow_bootstrap:
            return True

        if self._debug:
            logging.debug(
                f"PROVENANCE GATE: Rejected lemma '{lemma_lower}' - not grounded"
            )
        return False

    def validate_node_provenance_strict(
        self,
        lemma: str,
    ) -> bool:
        """Strict validation - lemma must be in story text."""
        lemma_lower = lemma.lower()

        if lemma_lower in self._persona_only_lemmas:
            return False

        return lemma_lower in self._story_lemmas

    # =========================================================
    # NODE LOOKUP AND CREATION
    # =========================================================

    def get_node_from_text(
        self,
        text: str,
        curr_sentences_nodes: List["Node"],
        curr_sentences_words: List[str],
        node_source: "NodeSource",
        create_node: bool,
    ) -> Optional["Node"]:
        """Get or create node from text."""
        from amoc.nlp.spacy_utils import get_concept_lemmas

        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]

        if create_node and self._canonicalize_and_classify_fn:
            canon, inferred_type = self._canonicalize_and_classify_fn(text)
            if inferred_type is None:
                return None

            lemmas = get_concept_lemmas(self._spacy_nlp, canon)
            if not self.admit_node(
                lemma=canon,
                node_type=inferred_type,
                provenance="TEXT_FALLBACK",
            ):
                return None

            return self._graph.add_or_get_node(
                lemmas, canon, inferred_type, node_source
            )

        return None

    def get_node_from_new_relationship(
        self,
        text: str,
        graph_active_nodes: List["Node"],
        curr_sentences_nodes: List["Node"],
        curr_sentences_words: List[str],
        node_source: "NodeSource",
        create_node: bool,
    ) -> Optional["Node"]:
        """Get or create node from new relationship extraction."""
        from amoc.nlp.spacy_utils import get_concept_lemmas
        from amoc.graph.node import NodeProvenance

        # 1. Exact sentence match
        if text in curr_sentences_words:
            return curr_sentences_nodes[curr_sentences_words.index(text)]

        # 2. Canonicalize
        if not self._canonicalize_and_classify_fn:
            return None

        canon, inferred_type = self._canonicalize_and_classify_fn(text)
        if inferred_type is None:
            return None

        lemmas = get_concept_lemmas(self._spacy_nlp, canon)
        if not lemmas:
            return None

        primary_lemma = lemmas[0]

        # 3. Try match active graph
        for node in graph_active_nodes:
            if lemmas == node.lemmas:
                return node

        # 4. Create node if allowed
        if create_node:
            if canon in {"subject", "object", "relation", "properties"}:
                return None

            if not self.admit_node(
                lemma=primary_lemma,
                node_type=inferred_type,
                provenance=NodeProvenance.STORY_TEXT,
            ):
                return None

            return self._graph.add_or_get_node(
                lemmas,
                canon,
                inferred_type,
                node_source,
            )

        return None

    def find_node_by_text(
        self,
        text: str,
        candidates,
    ) -> Optional["Node"]:
        """Find node by text among candidates."""
        from amoc.nlp.spacy_utils import canonicalize_node_text, get_concept_lemmas

        canon = canonicalize_node_text(self._spacy_nlp, text)
        lemmas = tuple(get_concept_lemmas(self._spacy_nlp, canon))
        for node in candidates:
            if lemmas == tuple(node.lemmas):
                return node
        return None

    def is_node_grounded(self, node: "Node") -> bool:
        """Check if node is grounded in story text."""
        for lemma in node.lemmas:
            if lemma.lower() in self._story_lemmas:
                return True
        return False

    def node_token_for_matrix(self, node: "Node") -> str:
        """Get token representation for matrix."""
        return (node.get_text_representer() or "").strip().lower()

    # =========================================================
    # ATTACHMENT CONSTRAINT
    # =========================================================

    def passes_attachment_constraint(
        self,
        subject: str,
        obj: str,
        current_sentence_words: List[str],
        current_sentence_nodes: List["Node"],
        graph_active_nodes: List["Node"],
        explicit_nodes: Set["Node"],
        carryover_nodes: Set["Node"],
        get_nodes_with_active_edges_fn: callable,
        graph_active_edge_nodes: Optional[Set["Node"]] = None,
        allow_inference_bridge: bool = False,
    ) -> bool:
        """
        Check if an edge passes the attachment constraint.

        The attachment constraint ensures new edges connect to existing
        active/explicit/carryover nodes, preventing disconnected islands.
        """
        from amoc.nlp.spacy_utils import canonicalize_node_text, get_concept_lemmas

        # Canonicalize
        subject = canonicalize_node_text(self._spacy_nlp, subject)
        obj = canonicalize_node_text(self._spacy_nlp, obj)

        subj_key = tuple(get_concept_lemmas(self._spacy_nlp, subject))
        obj_key = tuple(get_concept_lemmas(self._spacy_nlp, obj))

        # Bootstrap: first edge in graph
        if not self._graph.nodes:
            return True

        # --- Build frontier (connected component surface) ---
        active_nodes = set(get_nodes_with_active_edges_fn())
        frontier_nodes = active_nodes | explicit_nodes | carryover_nodes
        frontier_keys = {tuple(n.lemmas) for n in frontier_nodes}

        # Preserve already-connected relationships
        if subj_key in frontier_keys and obj_key in frontier_keys:
            return True

        # SAFE MULTI-HOP RULE
        # Allow if at least one endpoint touches frontier
        if subj_key in frontier_keys or obj_key in frontier_keys:
            return True

        # Otherwise reject (prevents island creation)
        return False

    # =========================================================
    # TEXT-BASED NODE EXTRACTION
    # =========================================================

    def get_sentences_text_based_nodes(
        self,
        previous_sentences: List["Span"],
        current_sentence_index: int,
        create_unexistent_nodes: bool = True,
    ) -> tuple:
        """
        Extract text-based nodes from sentences.

        Creates CONCEPT nodes for nouns and PROPERTY nodes for adjectives.
        Returns (nodes, words) tuple.
        """
        from amoc.graph.node import NodeType, NodeSource, NodeProvenance, NodeRole
        from amoc.nlp.spacy_utils import get_content_words_from_sent

        META_LEMMAS = {"subject", "object", "entity", "concept", "property"}

        text_based_nodes: List["Node"] = []
        text_based_words: List[str] = []

        for sent in previous_sentences:
            content_words = get_content_words_from_sent(self._spacy_nlp, sent)

            for word in content_words:
                lemma = word.lemma_.lower().strip()

                if not lemma:
                    continue

                # ------------------------------------------
                # CONCEPT NODES (nouns / proper nouns)
                # ------------------------------------------
                if word.pos_ in {"NOUN", "PROPN"}:
                    if lemma in META_LEMMAS:
                        continue

                    node_role = None
                    if word.dep_ in {"nsubj", "nsubjpass"}:
                        node_role = NodeRole.ACTOR
                    elif word.dep_ in {"pobj", "obl"}:
                        node_role = NodeRole.SETTING
                    else:
                        node_role = NodeRole.OBJECT

                    node = self._graph.get_node([lemma])

                    if node is None and create_unexistent_nodes:
                        node = self._graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.CONCEPT,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            node_role=node_role,
                            origin_sentence=current_sentence_index,
                            mark_explicit=False,
                        )

                    if node is None:
                        continue

                    node.mark_explicit_in_sentence(current_sentence_index)

                    text_based_nodes.append(node)
                    text_based_words.append(lemma)

                # ------------------------------------------
                # PROPERTY NODES (adjectives)
                # ------------------------------------------
                elif word.pos_ == "ADJ" or (
                    word.pos_ == "VERB"
                    and word.tag_ == "VBN"
                    and word.dep_ in {"acomp", "attr", "amod", "ROOT"}
                ):

                    node = self._graph.get_node([lemma])

                    if node is None and create_unexistent_nodes:
                        node = self._graph.add_or_get_node(
                            [lemma],
                            lemma,
                            NodeType.PROPERTY,
                            NodeSource.TEXT_BASED,
                            provenance=NodeProvenance.STORY_TEXT,
                            origin_sentence=current_sentence_index,
                            mark_explicit=False,
                        )

                    if node is None:
                        continue

                    node.mark_explicit_in_sentence(current_sentence_index)

                    text_based_nodes.append(node)
                    text_based_words.append(lemma)

        # Return unique
        seen = set()
        unique_nodes = []
        unique_words = []

        for node, word in zip(text_based_nodes, text_based_words):
            if node not in seen:
                seen.add(node)
                unique_nodes.append(node)
                unique_words.append(word)

        return unique_nodes, unique_words

    # =========================================================
    # PHRASE-LEVEL CONCEPT EXTRACTION
    # =========================================================

    def get_phrase_level_concepts(
        self,
        sent: "Span",
        admit_node_fn: callable,
    ) -> List["Node"]:
        """
        Extract phrase-level concepts from a sentence per AMoC v4 paper.

        Per paper Figures 2-4:
        - Node labels are single lowercase lemmas (e.g., "country" not "the country")
        - Determiners are NEVER included in node labels
        - Each noun becomes a CONCEPT node, each adjective a PROPERTY node
        """
        from amoc.graph.node import NodeType, NodeSource, NodeProvenance

        phrase_nodes = []

        # spaCy noun chunks = adjective + noun phrases
        for chunk in sent.noun_chunks:
            # Extract the head noun from the chunk (ignore determiners completely)
            head_noun = None
            for tok in chunk:
                if tok.pos_ in {"NOUN", "PROPN"}:
                    head_noun = tok
                    break

            if head_noun is None:
                continue

            # CRITICAL FIX: Use single lemma as node key, not full phrase
            # Per AMoC paper: nodes are "country" not "the country"
            lemma = head_noun.lemma_.lower()

            # actual_text should also be the clean lemma (no determiners)
            # This ensures get_text_representer() returns the clean label
            if not admit_node_fn(
                lemma=lemma,
                node_type=NodeType.CONCEPT,
                provenance="STORY_TEXT",
            ):
                continue

            node = self._graph.add_or_get_node(
                lemmas=[lemma],
                actual_text=lemma,
                node_type=NodeType.CONCEPT,
                node_source=NodeSource.TEXT_BASED,
                provenance=NodeProvenance.STORY_TEXT,
            )

            if node is not None:
                phrase_nodes.append(node)

        return phrase_nodes
