import pytest
from carnot.training.semantic_pruning import SemanticPruner


def test_semantic_pruner_initialization():
    """Test initialization of SemanticPruner."""
    pruner = SemanticPruner(similarity_threshold=0.9)
    assert pruner.similarity_threshold == 0.9


def test_calculate_similarity():
    """Test Jaccard similarity calculation.

    Spec: REQ-FR11-041-1
    """
    pruner = SemanticPruner()

    # Exact match
    assert pruner._calculate_similarity("rule A B", "rule A B") == 1.0

    # Empty strings
    assert pruner._calculate_similarity("", "") == 1.0
    assert pruner._calculate_similarity("rule", "") == 0.0

    # Partial match
    # set1 = {rule, a, b}, set2 = {rule, a, c} -> intersection {rule, a}, union {rule, a, b, c} -> 2/4 = 0.5
    assert pruner._calculate_similarity("rule a b", "rule a c") == 0.5


def test_prune_redundant_rules():
    """Test pruning of redundant rules.

    Spec: REQ-FR11-041-2
    SCENARIO-FR11-041: Semantic Pruning of Redundant Rules
    """
    pruner = SemanticPruner(similarity_threshold=0.8)

    rules = [
        "The variable x must be greater than 0",
        "Variable x MUST be greater than 0",  # Highly similar to the first
        "The variable y must be less than 10",  # Different
        "x > 0 for all cases",  # Different enough based on words
    ]

    pruned = pruner.prune_redundant_rules(rules)

    assert len(pruned) == 3
    assert "The variable x must be greater than 0" in pruned
    assert "Variable x MUST be greater than 0" not in pruned
    assert "The variable y must be less than 10" in pruned
    assert "x > 0 for all cases" in pruned


def test_prune_empty_rules():
    """Test pruning with empty list."""
    pruner = SemanticPruner()
    assert pruner.prune_redundant_rules([]) == []
