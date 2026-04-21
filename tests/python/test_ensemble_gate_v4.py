"""Tests for EnsembleGateV4.

Spec: REQ-VERIFY-147, REQ-VERIFY-148
"""

from __future__ import annotations

import pytest

from carnot.pipeline.ensemble_gate_v4 import EnsembleGateV4, EnsembleGateV4Result


# ---------------------------------------------------------------------------
# REQ-VERIFY-147: structured-first OR logic
# ---------------------------------------------------------------------------


def test_gate_opens_on_structured_recall_alone():
    """REQ-VERIFY-147: gate opens when structured_recall >= 0.20, others below threshold."""
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.10,
        hermes_v2_recall=0.0,
        structured_recall=0.20,
        causal_recall=0.10,
    )
    assert result.gate_open is True
    assert result.authorizes_vr is True
    assert result.honest_verdict == "gate_open_vr_authorized"


def test_gate_opens_on_causal_recall_even_if_structured_below_threshold():
    """REQ-VERIFY-147: gate opens when causal_recall >= 0.30, even if structured_recall < 0.20."""
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.12,
        hermes_v2_recall=0.0,
        structured_recall=0.10,  # below structured_threshold
        causal_recall=0.36,       # above max_component_threshold
    )
    assert result.gate_open is True
    assert result.authorizes_vr is True


def test_gate_opens_on_symcode_recall_meeting_max_threshold():
    """REQ-VERIFY-147: gate opens when symcode_recall >= 0.30, structured_recall < 0.20."""
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.35,
        hermes_v2_recall=0.0,
        structured_recall=0.05,
        causal_recall=0.10,
    )
    assert result.gate_open is True


def test_gate_closes_when_both_conditions_fail():
    """REQ-VERIFY-147: gate stays closed when neither structured nor max-component condition is met."""
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.10,
        hermes_v2_recall=0.90,  # high HermesV2 must NOT rescue the gate (REQ-VERIFY-148)
        structured_recall=0.10,  # below 0.20
        causal_recall=0.10,      # below 0.30
    )
    assert result.gate_open is False
    assert result.authorizes_vr is False
    assert result.honest_verdict == "gate_closed_vr_blocked"


# ---------------------------------------------------------------------------
# REQ-VERIFY-148: HermesV2 excluded from gate formula
# ---------------------------------------------------------------------------


def test_hermes_v2_does_not_rescue_closed_gate():
    """REQ-VERIFY-148: hermes_v2_recall=1.0 must not change gate_open when structured/causal/symcode all fail."""
    gate = EnsembleGateV4()
    with_high_hermes = gate.compute(0.05, 1.0, 0.05, 0.05)
    without_hermes = gate.compute(0.05, 0.0, 0.05, 0.05)
    # Gate must be closed in both cases — HermesV2 has no influence.
    assert with_high_hermes.gate_open is False
    assert without_hermes.gate_open is False


def test_ensemble_recall_excludes_hermes_v2():
    """REQ-VERIFY-148: ensemble_recall is the 3-component average, not including HermesV2."""
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.30,
        hermes_v2_recall=0.99,   # should NOT affect ensemble_recall
        structured_recall=0.30,
        causal_recall=0.30,
    )
    expected = (0.30 + 0.30 + 0.30) / 3
    assert abs(result.ensemble_recall - expected) < 1e-9


# ---------------------------------------------------------------------------
# Regression test: Exp 655 values must produce gate_open=True under v4
# ---------------------------------------------------------------------------


def test_gate_opens_on_exp655_recall_values():
    """Regression: Exp 655 recall values (gate_open=False under v3) must open under v4.

    Spec: REQ-VERIFY-147, REQ-VERIFY-148
    v3 values: symcode=0.12, hermes_v2=0.0, structured=0.20, causal=0.36, threshold=0.30
    v3 result: gate_open=False (ensemble 0.224 dragged below 0.30 by HermesV2=0.0)
    v4 result: gate_open=True  (causal_recall=0.36 >= max_component_threshold=0.30)
    """
    gate = EnsembleGateV4()
    result = gate.compute(
        symcode_recall=0.12,
        hermes_v2_recall=0.0,
        structured_recall=0.20,
        causal_recall=0.36,
    )
    assert result.gate_open is True, (
        f"Gate should open for Exp 655 recall values; got gate_open={result.gate_open}"
    )
    assert result.gate_version == "v4"
    assert result.authorizes_vr is True


# ---------------------------------------------------------------------------
# Dataclass integrity
# ---------------------------------------------------------------------------


def test_result_is_dataclass_instance():
    """EnsembleGateV4.compute() returns an EnsembleGateV4Result instance."""
    gate = EnsembleGateV4()
    result = gate.compute(0.5, 0.5, 0.5, 0.5)
    assert isinstance(result, EnsembleGateV4Result)


def test_custom_thresholds_respected():
    """EnsembleGateV4 constructor thresholds propagate to gate logic."""
    gate = EnsembleGateV4(structured_threshold=0.50, max_component_threshold=0.80)
    # With high thresholds the same Exp 655 values should close the gate.
    result = gate.compute(0.12, 0.0, 0.20, 0.36)
    assert result.gate_open is False
    assert result.gate_threshold == 0.50
