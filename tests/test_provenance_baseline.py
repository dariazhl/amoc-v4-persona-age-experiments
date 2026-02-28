"""
BASELINE TESTS: Provenance Validation Blocklists

These tests capture the CURRENT behavior of blocklist-based validation.
They serve as a baseline for Phase 3 parity testing during blocklist removal.

TEST CATEGORIES:
1. FORBIDDEN_NODE_LEMMAS - meta-linguistic/system terms
2. NARRATION_ARTIFACT_LEMMAS - narration style descriptors
3. GARBAGE_LEMMAS - overly generic/technical terms
4. VALID lemmas that should pass all checks
"""

import pytest
import sys
import importlib.util

# Direct module loading to avoid triggering amoc/__init__.py which imports spacy
def load_module_directly(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
base_path = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/amoc/graph"
node_module = load_module_directly("amoc.graph.node", f"{base_path}/node.py")
edge_module = load_module_directly("amoc.graph.edge", f"{base_path}/edge.py")
provenance_module = load_module_directly("amoc.graph.provenance_ops", f"{base_path}/provenance_ops.py")

Node = node_module.Node
NodeType = node_module.NodeType
NodeSource = node_module.NodeSource
NodeProvenance = node_module.NodeProvenance
Edge = edge_module.Edge
ProvenanceOps = provenance_module.ProvenanceOps


class MinimalGraph:
    """Minimal graph stub for testing ProvenanceOps without heavy dependencies."""
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self._story_lemmas = None
        self._persona_only_lemmas = None
        self._current_sentence_lemmas = None
        self._provenance_ops = ProvenanceOps(self)

    def add_or_get_node(
        self,
        lemmas,
        actual_text,
        node_type,
        node_source,
        **kwargs,
    ):
        """Simplified add_or_get_node that uses ProvenanceOps validation."""
        lemmas = [lemma.lower() for lemma in lemmas]
        primary_lemma = lemmas[0].lower() if lemmas else ""

        # Check if node already exists
        existing = None
        for n in self.nodes:
            if n.lemmas == lemmas:
                existing = n
                break

        valid, reason = self._provenance_ops.validate_node_creation(
            lemmas=lemmas,
            node_type=node_type,
            provenance=kwargs.get("provenance"),
            is_new_node=existing is None,
        )

        if not valid:
            return None

        if existing:
            return existing

        node = Node(
            lemmas,
            actual_text.lower() if actual_text else "",
            node_type,
            node_source,
            0,
        )
        self.nodes.add(node)
        return node


# Use MinimalGraph instead of full Graph
Graph = MinimalGraph


class TestBlocklistBaseline:
    """Baseline tests for blocklist behavior."""

    def setup_method(self):
        """Create fresh graph for each test."""
        self.graph = Graph()

    # =========================================================================
    # FORBIDDEN_NODE_LEMMAS TESTS
    # =========================================================================

    @pytest.mark.parametrize("lemma", [
        "student", "persona", "relation", "context", "object", "place",
        "story", "narrative", "sentence", "edge", "node", "property",
        "label", "target", "source", "pronoun", "noun", "user",
    ])
    def test_forbidden_lemmas_rejected(self, lemma):
        """FORBIDDEN_NODE_LEMMAS: Each lemma should be rejected."""
        node = self.graph.add_or_get_node(
            lemmas=[lemma],
            actual_text=lemma,
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, f"Forbidden lemma '{lemma}' was not rejected"

    @pytest.mark.parametrize("lemma", [
        "student", "persona", "relation", "context", "object",
    ])
    def test_forbidden_lemmas_rejected_as_secondary(self, lemma):
        """FORBIDDEN_NODE_LEMMAS: Should be rejected even as secondary lemma."""
        node = self.graph.add_or_get_node(
            lemmas=["validword", lemma],
            actual_text="valid word",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, f"Forbidden secondary lemma '{lemma}' was not rejected"

    # =========================================================================
    # NARRATION_ARTIFACT_LEMMAS TESTS
    # =========================================================================

    @pytest.mark.parametrize("lemma", [
        "text", "sentence", "paragraph", "mention", "mentions", "narration", "story",
    ])
    def test_narration_artifact_lemmas_rejected(self, lemma):
        """NARRATION_ARTIFACT_LEMMAS: Each lemma should be rejected."""
        node = self.graph.add_or_get_node(
            lemmas=[lemma],
            actual_text=lemma,
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, f"Narration artifact lemma '{lemma}' was not rejected"

    # =========================================================================
    # GARBAGE_LEMMAS TESTS
    # =========================================================================

    @pytest.mark.parametrize("lemma", [
        "edge", "node", "relation", "property", "label", "target",
        "source", "t", "type", "person", "approach", "kind", "thing",
    ])
    def test_garbage_lemmas_rejected_as_primary(self, lemma):
        """GARBAGE_LEMMAS: Should be rejected when primary lemma."""
        node = self.graph.add_or_get_node(
            lemmas=[lemma],
            actual_text=lemma,
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, f"Garbage lemma '{lemma}' was not rejected"

    def test_garbage_lemmas_allowed_as_secondary(self):
        """GARBAGE_LEMMAS: Should be allowed as secondary lemma (unlike FORBIDDEN)."""
        # This tests the difference: GARBAGE only checks primary lemma
        # while FORBIDDEN checks all lemmas
        # Note: "type" is in GARBAGE but not FORBIDDEN, so it should pass as secondary
        node = self.graph.add_or_get_node(
            lemmas=["validword", "type"],  # "type" is only in GARBAGE
            actual_text="valid word",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        # Since "type" is in GARBAGE but not FORBIDDEN, it should be allowed as secondary
        assert node is not None, "Garbage lemma as secondary should be allowed"

    # =========================================================================
    # VALID LEMMAS TESTS
    # =========================================================================

    @pytest.mark.parametrize("lemma", [
        "dog", "cat", "house", "mountain", "river", "john", "mary",
        "running", "walked", "beautiful", "quickly",
    ])
    def test_valid_lemmas_accepted(self, lemma):
        """Valid story lemmas should be accepted."""
        node = self.graph.add_or_get_node(
            lemmas=[lemma],
            actual_text=lemma,
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is not None, f"Valid lemma '{lemma}' was incorrectly rejected"

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_single_character_lemma_rejected(self):
        """Single character lemmas should be rejected (too short)."""
        node = self.graph.add_or_get_node(
            lemmas=["a"],
            actual_text="a",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, "Single character lemma should be rejected"

    def test_empty_lemma_rejected(self):
        """Empty lemma should be rejected."""
        node = self.graph.add_or_get_node(
            lemmas=[""],
            actual_text="",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, "Empty lemma should be rejected"

    def test_case_insensitivity(self):
        """Blocklist check should be case-insensitive."""
        # FORBIDDEN lemma in different cases
        node = self.graph.add_or_get_node(
            lemmas=["STUDENT"],
            actual_text="STUDENT",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, "Uppercase forbidden lemma should be rejected"

        node = self.graph.add_or_get_node(
            lemmas=["Student"],
            actual_text="Student",
            node_type=NodeType.CONCEPT,
            node_source=NodeSource.TEXT_BASED,
        )
        assert node is None, "Mixed case forbidden lemma should be rejected"

    # =========================================================================
    # OVERLAP DETECTION TESTS
    # =========================================================================

    def test_overlapping_blocklist_terms(self):
        """Test terms that appear in multiple blocklists."""
        # These terms appear in multiple lists - document the overlap
        overlap_terms = ["edge", "node", "relation", "property", "label", "target", "source", "story", "sentence"]

        for term in overlap_terms:
            node = self.graph.add_or_get_node(
                lemmas=[term],
                actual_text=term,
                node_type=NodeType.CONCEPT,
                node_source=NodeSource.TEXT_BASED,
            )
            assert node is None, f"Overlapping term '{term}' should be rejected"


class TestIsForbidenLemmaBaseline:
    """Baseline tests for is_forbidden_lemma() helper."""

    def setup_method(self):
        self.graph = Graph()

    def test_is_forbidden_lemma_true_for_all_blocklists(self):
        """is_forbidden_lemma should return True for all blocklist terms."""
        all_blocked = [
            # FORBIDDEN
            "student", "persona", "user",
            # NARRATION
            "text", "paragraph", "mention",
            # GARBAGE
            "type", "kind", "thing",
        ]
        for lemma in all_blocked:
            assert self.graph._provenance_ops.is_forbidden_lemma(lemma), \
                f"'{lemma}' should be recognized as forbidden"

    def test_is_forbidden_lemma_false_for_valid(self):
        """is_forbidden_lemma should return False for valid terms."""
        valid_terms = ["dog", "cat", "house", "running", "beautiful"]
        for lemma in valid_terms:
            assert not self.graph._provenance_ops.is_forbidden_lemma(lemma), \
                f"'{lemma}' should not be recognized as forbidden"


class TestBlocklistCategories:
    """Document what each semantic category is trying to filter."""

    def test_document_forbidden_purpose(self):
        """FORBIDDEN_LEMMAS_EQUIVALENT: Meta-linguistic and system terminology.

        These terms should never appear as story concepts because they:
        1. Are part of the system's vocabulary (node, edge, property)
        2. Are meta-references to the reading context (student, persona, user)
        3. Are abstract linguistic categories (pronoun, noun)
        4. Are meta-references to the narrative (story, narrative, sentence)
        """
        from amoc.graph.provenance_ops import SemanticCategory

        expected = {
            "student", "persona", "relation", "context", "object", "place",
            "story", "narrative", "sentence", "edge", "node", "property",
            "label", "target", "source", "pronoun", "noun", "user",
        }
        assert SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT == expected

    def test_document_narration_purpose(self):
        """NARRATION_ARTIFACT_EQUIVALENT: Narration-style descriptors.

        These terms describe the narrative structure rather than story content:
        1. "text", "sentence", "paragraph" - structural descriptions
        2. "mention", "mentions" - references to textual occurrence
        3. "narration", "story" - meta-references to the narrative itself
        """
        from amoc.graph.provenance_ops import SemanticCategory

        expected = {
            "text", "sentence", "paragraph", "mention", "mentions", "narration", "story",
        }
        assert SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT == expected

    def test_document_garbage_purpose(self):
        """GARBAGE_LEMMAS_EQUIVALENT: Overly generic or technical terms.

        These terms are too generic to carry useful semantic information:
        1. Graph technical terms (edge, node, relation, property, label, target, source)
        2. Single character or too short (t)
        3. Generic type references (type, kind, thing)
        4. Overly broad categories (person, approach)
        """
        from amoc.graph.provenance_ops import SemanticCategory

        expected = {
            "edge", "node", "relation", "property", "label", "target",
            "source", "t", "type", "person", "approach", "kind", "thing",
        }
        assert SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
