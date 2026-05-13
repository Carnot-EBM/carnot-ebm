"""Tests for online training promotion ledger.

Spec traces: REQ-LEARN-1986, SCENARIO-LEARN-1986
"""

from carnot.training.online import ValidatorTreeLedger

def test_validator_tree_ledger_pass() -> None:
    """Test successful promotion with positive utility and no forgetting."""
    ledger = ValidatorTreeLedger()
    result = ledger.evaluate_promotion(
        utility_delta=0.5,
        soundness_mistakes=1,
        completeness_mistakes=2,
        previous_performance=0.8,
        new_performance=0.85
    )
    assert result["promotion_gate_passed"] is True
    assert result["utility_delta"] == 0.5
    assert result["soundness_mistakes"] == 1
    assert result["completeness_mistakes"] == 2
    assert result["non_forgetting_holds"] is True

def test_validator_tree_ledger_fail_utility() -> None:
    """Test failed promotion due to zero or negative utility."""
    ledger = ValidatorTreeLedger()
    result = ledger.evaluate_promotion(
        utility_delta=-0.1,
        soundness_mistakes=0,
        completeness_mistakes=0,
        previous_performance=0.8,
        new_performance=0.85
    )
    assert result["promotion_gate_passed"] is False

def test_validator_tree_ledger_fail_forgetting() -> None:
    """Test failed promotion due to catastrophic forgetting."""
    ledger = ValidatorTreeLedger()
    result = ledger.evaluate_promotion(
        utility_delta=0.5,
        soundness_mistakes=0,
        completeness_mistakes=0,
        previous_performance=0.8,
        new_performance=0.7
    )
    assert result["promotion_gate_passed"] is False

def test_validator_tree_ledger_accumulation() -> None:
    """Test accumulation of mistakes across evaluations."""
    ledger = ValidatorTreeLedger()
    ledger.evaluate_promotion(0.5, 1, 2, 0.8, 0.85)
    result = ledger.evaluate_promotion(0.1, 3, 1, 0.85, 0.86)
    assert result["soundness_mistakes"] == 4
    assert result["completeness_mistakes"] == 3
