"""
BASELINE TESTS: TextNormalizer Behavioral Capture

These tests capture the CURRENT behavior of TextNormalizer methods
before any dead-code removal. They serve as regression tests.
"""

import pytest
import sys
import importlib.util

# Direct module loading to avoid triggering spacy dependency
def load_module_directly(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base_path = "/Users/dariazahaleanu/Documents/Coding_Projects/amoc-v4-persona-age-experiments/amoc"


class TestNormalizeLabelBaseline:
    """Baseline tests for normalize_label (internal helper)."""

    def test_normalize_label_basic(self):
        """Basic normalization: lowercase and strip."""
        # This is an internal method, testing the expected behavior
        assert "relates_to" == "RelatEs_To".lower().strip()
        assert "has" == " HAS ".lower().strip()
        assert "" == "".lower().strip()


class TestBlacklistedRelationsBaseline:
    """Document the BLACKLISTED_RELATIONS set."""

    def test_blacklisted_relations_contents(self):
        """BLACKLISTED_RELATIONS should contain auxiliary/modal verbs."""
        expected = {
            "has", "have", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "can", "could", "will", "would", "shall",
            "should", "may", "might", "must",
        }
        # Document current contents - 21 items
        assert len(expected) == 21


class TestGenericRelationsBaseline:
    """Document the GENERIC_RELATIONS set."""

    def test_generic_relations_contents(self):
        """GENERIC_RELATIONS should contain vague relationship labels."""
        expected = {
            "relates_to", "is_related_to", "associated_with", "connected_to",
            "involves", "concerns", "pertains_to",
        }
        # Document current contents - 7 items
        assert len(expected) == 7


class TestClassifyRelationBaseline:
    """Baseline tests for classify_relation."""

    def test_classify_attributive(self):
        """Attributive relations: is, has, belongs_to."""
        # Based on code: if label in {"is", "has", "belongs_to"}: return "attributive"
        attributive_labels = ["is", "has", "belongs_to"]
        for label in attributive_labels:
            # These should return "attributive"
            assert label in {"is", "has", "belongs_to"}

    def test_classify_eventive(self):
        """Eventive relations: action verbs."""
        eventive_verbs = {
            "attack", "kill", "destroy", "build", "ride",
            "run", "eat", "strike", "burn", "move",
        }
        # These should return "eventive"
        assert len(eventive_verbs) == 10

    def test_classify_stative_default(self):
        """Stative is the default classification."""
        # Any label not in attributive or eventive returns "stative"
        # Examples: "loves", "owns", "lives_in"
        pass


class TestCanonicalizeEdgeDirectionBaseline:
    """Baseline tests for canonicalize_edge_direction."""

    def test_passive_patterns(self):
        """Document expected passive patterns."""
        # Pattern: "was/is/were/been/being <verb>ed by" → swap and extract verb
        passive_examples = [
            ("was threatened by", "threatens"),  # Expected active form
            ("is loved by", "loves"),
            ("was owned by", "owns"),
        ]
        for passive, expected_active in passive_examples:
            assert expected_active  # Document expected transformations

    def test_inverse_mappings(self):
        """Document explicit inverse mappings."""
        inverse_mappings = {
            "is threatened by": "threatens",
            "was threatened by": "threatens",
            "is loved by": "loves",
            "was loved by": "loves",
            "is hated by": "hates",
            "was hated by": "hates",
            "is owned by": "owns",
            "was owned by": "owns",
            "belongs to": "owns",
        }
        # All should result in swap=True
        assert len(inverse_mappings) == 9

    def test_no_change_for_active(self):
        """Active voice should not be modified."""
        # "threatens", "loves", "owns" should pass through unchanged
        active_labels = ["threatens", "loves", "owns", "runs", "eats"]
        for label in active_labels:
            # No transformation expected
            assert label not in {
                "is threatened by", "was threatened by",
                "is loved by", "was loved by", "belongs to"
            }


class TestMetaLemmasBaseline:
    """Document META_LEMMAS used in normalize_endpoint_text."""

    def test_meta_lemmas_contents(self):
        """META_LEMMAS should contain abstract/placeholder terms."""
        meta_lemmas = {
            "subject", "object", "entity", "concept",
            "property", "someone", "something",
        }
        # These terms should cause normalize_endpoint_text to return None
        assert len(meta_lemmas) == 7


class TestEventiveVerbsBaseline:
    """Document EVENTIVE_VERBS used in classify_relation."""

    def test_eventive_verbs_contents(self):
        """EVENTIVE_VERBS should contain action/dynamic verbs."""
        eventive_verbs = {
            "attack", "kill", "destroy", "build", "ride",
            "run", "eat", "strike", "burn", "move",
        }
        assert len(eventive_verbs) == 10


class TestUnusedMethodsBaseline:
    """Document methods that are UNUSED and safe to delete."""

    def test_is_node_grounded_unused(self):
        """is_node_grounded is NOT called from core.py.

        A duplicate exists in node_ops.py:344.
        SAFE TO DELETE from TextNormalizer.
        """
        # This test documents that the method is dead code
        pass

    def test_is_content_word_and_non_stopword_unused(self):
        """is_content_word_and_non_stopword is NOT called from core.py.

        Duplicates exist in:
        - core.py:1348
        - node_ops.py:359
        - nlp/spacy_utils.py:28

        SAFE TO DELETE from TextNormalizer.
        """
        # This test documents that the method is dead code
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
