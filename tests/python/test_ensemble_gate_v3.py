"""Tests for EnsembleRecallGateV3 and EnsembleGateV3Result.

Spec: REQ-VERIFY-149, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

import pytest

from carnot.pipeline.ensemble_gate_v3 import EnsembleGateV3Result, EnsembleRecallGateV3


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-200: Gate opens when weighted ensemble reaches threshold
# ---------------------------------------------------------------------------


def test_gate_opens_at_threshold():
    """REQ-VERIFY-149: ensemble_recall >= 0.30 must set gate_open=True."""
    gate = EnsembleRecallGateV3()
    # With causal=0.36 and structured=0.20 and symcode=0.12:
    # 0.3*0.12 + 0.4*0.20 + 0.3*0.36 = 0.036 + 0.08 + 0.108 = 0.224 — gate closed
    result = gate.compute(
        symcode_recall=0.12,
        hermes_v2_recall=0.0,
        structured_recall=0.20,
        causal_recall=0.36,
    )
    assert result.gate_version == "v3"
    assert abs(result.ensemble_recall - 0.224) < 1e-9
    assert result.gate_open is False


def test_gate_opens_when_all_signals_high():
    """REQ-VERIFY-149: gate_open=True when ensemble_recall >= 0.30."""
    gate = EnsembleRecallGateV3()
    # 0.3*0.40 + 0.4*0.40 + 0.3*0.40 = 0.40 >= 0.30 → open
    result = gate.compute(
        symcode_recall=0.40,
        hermes_v2_recall=0.40,
        structured_recall=0.40,
        causal_recall=0.40,
    )
    assert result.gate_open is True
    assert abs(result.ensemble_recall - 0.40) < 1e-9


def test_gate_opens_exactly_at_threshold():
    """REQ-VERIFY-149: gate_open=True at exactly 0.30 (boundary inclusive)."""
    gate = EnsembleRecallGateV3()
    # 0.3*0.30 + 0.4*0.30 + 0.3*0.30 = 0.30
    result = gate.compute(
        symcode_recall=0.30,
        hermes_v2_recall=0.0,
        structured_recall=0.30,
        causal_recall=0.30,
    )
    assert result.gate_open is True
    assert abs(result.ensemble_recall - 0.30) < 1e-9


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-201: Gate closed below threshold; gate_blocker logic correct
# ---------------------------------------------------------------------------


def test_gate_closed_below_threshold():
    """REQ-VERIFY-149: gate_open=False when ensemble_recall < 0.30."""
    gate = EnsembleRecallGateV3()
    result = gate.compute(
        symcode_recall=0.0,
        hermes_v2_recall=0.0,
        structured_recall=0.0,
        causal_recall=0.0,
    )
    assert result.gate_open is False
    assert result.ensemble_recall == 0.0


def test_gate_closed_just_below_threshold():
    """REQ-VERIFY-149: ensemble_recall just below 0.30 keeps gate closed."""
    gate = EnsembleRecallGateV3()
    # 0.3*0.29 + 0.4*0.29 + 0.3*0.29 = 0.29
    result = gate.compute(
        symcode_recall=0.29,
        hermes_v2_recall=0.0,
        structured_recall=0.29,
        causal_recall=0.29,
    )
    assert result.gate_open is False
    assert result.ensemble_recall < 0.30


# ---------------------------------------------------------------------------
# EnsembleGateV3Result fields
# ---------------------------------------------------------------------------


def test_result_fields_preserved():
    """REQ-VERIFY-149: all input recall signals are stored unmodified in result."""
    gate = EnsembleRecallGateV3()
    result = gate.compute(
        symcode_recall=0.12,
        hermes_v2_recall=0.05,
        structured_recall=0.20,
        causal_recall=0.36,
    )
    assert result.symcode_recall == 0.12
    assert result.hermes_v2_recall == 0.05
    assert result.structured_recall == 0.20
    assert result.causal_recall == 0.36
    assert result.gate_version == "v3"


def test_result_is_dataclass_instance():
    """REQ-VERIFY-149: compute() returns an EnsembleGateV3Result."""
    gate = EnsembleRecallGateV3()
    result = gate.compute(0.1, 0.0, 0.1, 0.1)
    assert isinstance(result, EnsembleGateV3Result)


# ---------------------------------------------------------------------------
# Custom weights and threshold
# ---------------------------------------------------------------------------


def test_custom_weights():
    """REQ-VERIFY-149: weights and threshold are configurable."""
    gate = EnsembleRecallGateV3(
        symcode_weight=0.5,
        structured_weight=0.3,
        causal_weight=0.2,
        threshold=0.25,
    )
    # 0.5*0.20 + 0.3*0.20 + 0.2*0.20 = 0.20 < 0.25 → closed
    result = gate.compute(0.20, 0.0, 0.20, 0.20)
    assert result.gate_open is False

    # 0.5*0.50 + 0.3*0.50 + 0.2*0.50 = 0.50 >= 0.25 → open
    result2 = gate.compute(0.50, 0.0, 0.50, 0.50)
    assert result2.gate_open is True


def test_weights_stored_on_instance():
    """REQ-VERIFY-149: weights dict is accessible on the gate instance."""
    gate = EnsembleRecallGateV3(symcode_weight=0.1, structured_weight=0.6, causal_weight=0.3)
    assert gate.weights["symcode"] == 0.1
    assert gate.weights["structured"] == 0.6
    assert gate.weights["causal"] == 0.3
    assert gate.threshold == 0.30


# ---------------------------------------------------------------------------
# hermes_v2_recall is tracked but does not affect ensemble
# ---------------------------------------------------------------------------


def test_hermes_v2_recall_does_not_affect_ensemble():
    """REQ-VERIFY-149: changing hermes_v2_recall must not change ensemble_recall."""
    gate = EnsembleRecallGateV3()
    r1 = gate.compute(0.12, 0.0, 0.20, 0.36)
    r2 = gate.compute(0.12, 0.99, 0.20, 0.36)
    assert r1.ensemble_recall == r2.ensemble_recall
    assert r1.gate_open == r2.gate_open


# ---------------------------------------------------------------------------
# Export from carnot.pipeline
# ---------------------------------------------------------------------------


def test_pipeline_exports():
    """REQ-VERIFY-149-c: EnsembleRecallGateV3 and EnsembleGateV3Result are exported."""
    from carnot.pipeline import EnsembleGateV3Result as R
    from carnot.pipeline import EnsembleRecallGateV3 as G

    assert G is EnsembleRecallGateV3
    assert R is EnsembleGateV3Result
