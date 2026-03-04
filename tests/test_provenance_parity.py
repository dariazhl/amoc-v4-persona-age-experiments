"""
POST-MIGRATION TESTS: SemanticCategory Validation

These tests verify that the SemanticCategory-based validation
correctly classifies and filters all expected lemmas.

MIGRATION COMPLETE: Hardcoded blocklists removed, replaced with SemanticCategory.
"""

import pytest
import sys
import importlib.util
import types

# Direct module loading to avoid triggering amoc/__init__.py which imports spacy
def load_module_directly(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules directly
base_path = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/amoc/graph"
algorithms_path = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/amoc/graph_algorithms"
if "amoc" not in sys.modules:
    sys.modules["amoc"] = types.ModuleType("amoc")
if "amoc.graph" not in sys.modules:
    amoc_graph = types.ModuleType("amoc.graph")
    sys.modules["amoc.graph"] = amoc_graph
    sys.modules["amoc"].graph = amoc_graph
if "amoc.graph_algorithms" not in sys.modules:
    amoc_graph_algorithms = types.ModuleType("amoc.graph_algorithms")
    sys.modules["amoc.graph_algorithms"] = amoc_graph_algorithms
    sys.modules["amoc"].graph_algorithms = amoc_graph_algorithms
node_module = load_module_directly("amoc.graph.node", f"{base_path}/node.py")
edge_module = load_module_directly("amoc.graph.edge", f"{base_path}/edge.py")
provenance_module = load_module_directly(
    "amoc.graph_algorithms.provenance_validation",
    f"{algorithms_path}/provenance_validation.py",
)

Node = node_module.Node
NodeType = node_module.NodeType
NodeSource = node_module.NodeSource
NodeProvenance = node_module.NodeProvenance
Edge = edge_module.Edge
ProvenanceValidation = provenance_module.ProvenanceValidation
SemanticCategory = provenance_module.SemanticCategory


class MinimalGraph:
    """Minimal graph stub for testing ProvenanceValidation."""
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self._story_lemmas = None
        self._persona_only_lemmas = None
        self._current_sentence_lemmas = None
        self._provenance_ops = ProvenanceValidation(self)


class TestSemanticCategoryEquivalence:
    """Verify SemanticCategory equivalents contain correct terms."""

    def test_all_forbidden_in_semantic_categories(self):
        """Every FORBIDDEN_LEMMAS_EQUIVALENT term must be classified."""
        for lemma in SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT:
            category = SemanticCategory.get_category(lemma)
            assert category is not None, \
                f"FORBIDDEN_LEMMAS_EQUIVALENT term '{lemma}' not found in SemanticCategory"

    def test_all_narration_in_semantic_categories(self):
        """Every NARRATION_ARTIFACT_EQUIVALENT term must be classified."""
        for lemma in SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT:
            category = SemanticCategory.get_category(lemma)
            assert category is not None, \
                f"NARRATION_ARTIFACT_EQUIVALENT term '{lemma}' not found in SemanticCategory"

    def test_all_garbage_in_semantic_categories(self):
        """Every GARBAGE_LEMMAS_EQUIVALENT term must be classified."""
        for lemma in SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT:
            category = SemanticCategory.get_category(lemma)
            assert category is not None, \
                f"GARBAGE_LEMMAS_EQUIVALENT term '{lemma}' not found in SemanticCategory"

    def test_semantic_categories_cover_all_equivalents(self):
        """SemanticCategory.all_blocked_lemmas() must equal union of all equivalents."""
        all_equivalents = (
            SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT
            | SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT
            | SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT
        )
        all_semantic = SemanticCategory.all_blocked_lemmas()

        # Check semantic covers equivalents
        missing_from_semantic = all_equivalents - all_semantic
        assert not missing_from_semantic, \
            f"Equivalent terms not in SemanticCategory: {missing_from_semantic}"

        # Check equivalents cover semantic (no extras in semantic)
        extra_in_semantic = all_semantic - all_equivalents
        assert not extra_in_semantic, \
            f"SemanticCategory has extra terms not in equivalents: {extra_in_semantic}"


class TestValidation:
    """Verify that validate_node_creation correctly filters all expected lemmas."""

    def setup_method(self):
        self.graph = MinimalGraph()
        self.ops = self.graph._provenance_ops

    def _check_validation(self, lemmas, node_type, provenance=None, is_new_node=True):
        """Helper to check validation."""
        return self.ops.validate_node_creation(
            lemmas, node_type, provenance, is_new_node
        )

    # =========================================================================
    # FORBIDDEN LEMMAS
    # =========================================================================

    @pytest.mark.parametrize("lemma", list(SemanticCategory.FORBIDDEN_LEMMAS_EQUIVALENT))
    def test_forbidden_primary(self, lemma):
        """Forbidden lemmas as primary: should be rejected."""
        result = self._check_validation([lemma], NodeType.CONCEPT)
        assert result[0] is False, f"Should reject '{lemma}'"

    @pytest.mark.parametrize("lemma", [
        "student", "persona", "relation", "context", "object",
    ])
    def test_forbidden_secondary(self, lemma):
        """Forbidden lemmas as secondary: should be rejected."""
        result = self._check_validation(["validword", lemma], NodeType.CONCEPT)
        assert result[0] is False, f"Should reject secondary '{lemma}'"

    # =========================================================================
    # NARRATION ARTIFACT LEMMAS
    # =========================================================================

    @pytest.mark.parametrize("lemma", list(SemanticCategory.NARRATION_ARTIFACT_EQUIVALENT))
    def test_narration_primary(self, lemma):
        """Narration artifact lemmas as primary: should be rejected."""
        result = self._check_validation([lemma], NodeType.CONCEPT)
        assert result[0] is False, f"Should reject '{lemma}'"

    # =========================================================================
    # GARBAGE LEMMAS
    # =========================================================================

    @pytest.mark.parametrize("lemma", list(SemanticCategory.GARBAGE_LEMMAS_EQUIVALENT))
    def test_garbage_primary(self, lemma):
        """Garbage lemmas as primary: should be rejected."""
        result = self._check_validation([lemma], NodeType.CONCEPT)
        assert result[0] is False, f"Should reject '{lemma}'"

    def test_garbage_secondary_allowed(self):
        """Garbage-only lemmas as secondary: should be accepted.

        "type" is in GARBAGE_LEMMAS_EQUIVALENT but NOT in FORBIDDEN.
        GARBAGE is only checked for PRIMARY lemma.
        FORBIDDEN is checked for ALL lemmas.

        So ["validword", "type"] should be ACCEPTED because
        "type" as secondary is not in FORBIDDEN.
        """
        result = self._check_validation(["validword", "type"], NodeType.CONCEPT)
        assert result[0] is True, "Should accept 'type' as secondary"

    # =========================================================================
    # VALID LEMMAS
    # =========================================================================

    @pytest.mark.parametrize("lemma", [
        "dog", "cat", "house", "mountain", "river", "john", "mary",
        "running", "walked", "beautiful", "quickly",
    ])
    def test_valid_accepted(self, lemma):
        """Valid lemmas: should be accepted."""
        result = self._check_validation([lemma], NodeType.CONCEPT)
        assert result[0] is True, f"Should accept '{lemma}'"

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    def test_empty_lemma_rejected(self):
        """Empty lemma: should be rejected."""
        result = self._check_validation([""], NodeType.CONCEPT)
        assert result[0] is False

    def test_single_char_rejected(self):
        """Single character lemma: should be rejected."""
        result = self._check_validation(["a"], NodeType.CONCEPT)
        assert result[0] is False

    def test_case_insensitivity(self):
        """Case variations: should handle correctly.

        NOTE: In actual usage, graph.py lowercases lemmas BEFORE calling
        validate_node_creation. So validate_node_creation always receives
        lowercase lemmas.
        """
        for case in ["STUDENT", "Student", "sTuDeNt"]:
            # Lowercase to match graph.py behavior
            lemmas = [case.lower()]
            result = self._check_validation(lemmas, NodeType.CONCEPT)
            assert result[0] is False, f"Should reject '{case}'"


class TestIsForbiddenLemma:
    """Verify is_forbidden_lemma correctly classifies all terms."""

    def setup_method(self):
        self.graph = MinimalGraph()
        self.ops = self.graph._provenance_ops

    @pytest.mark.parametrize("lemma", list(SemanticCategory.all_blocked_lemmas()))
    def test_blocked_lemma_detected(self, lemma):
        """All blocked lemmas: is_forbidden_lemma should return True."""
        result = self.ops.is_forbidden_lemma(lemma)
        assert result is True, f"'{lemma}' should be detected as forbidden"

    @pytest.mark.parametrize("lemma", [
        "dog", "cat", "house", "john", "running",
    ])
    def test_valid_lemma_allowed(self, lemma):
        """Valid lemmas: is_forbidden_lemma should return False."""
        result = self.ops.is_forbidden_lemma(lemma)
        assert result is False, f"'{lemma}' should not be detected as forbidden"


class TestSemanticCategoryClassification:
    """Test that SemanticCategory correctly classifies all terms."""

    def test_system_terminology_classification(self):
        """System terminology should be classified correctly."""
        for term in ["node", "edge", "property", "label", "target", "source", "relation"]:
            assert SemanticCategory.get_category(term) == "system_terminology", \
                f"'{term}' should be classified as system_terminology"

    def test_narrative_meta_classification(self):
        """Narrative meta terms should be classified correctly."""
        for term in ["story", "narrative", "sentence", "text", "paragraph", "mention", "mentions", "narration"]:
            assert SemanticCategory.get_category(term) == "narrative_meta", \
                f"'{term}' should be classified as narrative_meta"

    def test_context_references_classification(self):
        """Context references should be classified correctly."""
        for term in ["student", "persona", "user", "context"]:
            assert SemanticCategory.get_category(term) == "context_references", \
                f"'{term}' should be classified as context_references"

    def test_linguistic_categories_classification(self):
        """Linguistic categories should be classified correctly."""
        for term in ["pronoun", "noun"]:
            assert SemanticCategory.get_category(term) == "linguistic_categories", \
                f"'{term}' should be classified as linguistic_categories"

    def test_generic_abstractions_classification(self):
        """Generic abstractions should be classified correctly."""
        for term in ["thing", "kind", "type", "approach", "person", "object", "place"]:
            assert SemanticCategory.get_category(term) == "generic_abstractions", \
                f"'{term}' should be classified as generic_abstractions"

    def test_garbage_tokens_classification(self):
        """Garbage tokens should be classified correctly."""
        assert SemanticCategory.get_category("t") == "garbage_tokens"

    def test_valid_lemmas_not_classified(self):
        """Valid story lemmas should not be classified."""
        for term in ["dog", "cat", "house", "mountain", "john", "mary"]:
            assert SemanticCategory.get_category(term) is None, \
                f"'{term}' should not be classified (should be None)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
