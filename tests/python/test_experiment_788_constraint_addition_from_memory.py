"""Tests for Exp 788: Constraint Addition from Memory.

Verifies that IsingConstraintGenerator correctly synthesises coupling rows
from session memory error patterns and injects them additively into IsingEBM.

Spec: REQ-LEARN-056, REQ-LEARN-057, SCENARIO-LEARN-100, SCENARIO-LEARN-101
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.models.ising import IsingConfig, IsingModel  # noqa: E402
from carnot.pipeline.constraint_generator import (  # noqa: E402
    CouplingRow,
    ErrorPattern,
    IsingConstraintGenerator,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_ising(input_dim: int = 32) -> IsingModel:
    """Return a fresh zero-init IsingModel for deterministic tests."""
    return IsingModel(IsingConfig(input_dim=input_dim, coupling_init="zeros"))


# ---------------------------------------------------------------------------
# REQ-LEARN-056 / SCENARIO-LEARN-100: synthesize_from_memory with carry_error
# ---------------------------------------------------------------------------


def test_synthesize_returns_coupling_for_carry_error_at_threshold() -> None:
    """synthesize_from_memory MUST return one CouplingRow when carry_error count >= threshold.

    Spec: REQ-LEARN-056, SCENARIO-LEARN-100
    """
    model = _make_ising()
    gen = IsingConstraintGenerator(model, threshold=3)
    patterns = [ErrorPattern(pattern_type="carry_error", count=3, example_step="37 + 45 = 72")]

    rows = gen.synthesize_from_memory(patterns)

    assert len(rows) == 1, "expected exactly one CouplingRow for carry_error at threshold"
    row = rows[0]
    assert row.J_value == -1.0, "carry_error coupling must be anti-ferromagnetic (J=-1.0)"
    assert 0 <= row.var1 < model.config.input_dim
    assert 0 <= row.var2 < model.config.input_dim
    assert row.var1 != row.var2, "var1 and var2 must be distinct"


def test_synthesize_returns_coupling_for_all_pattern_types() -> None:
    """All five canonical error types above threshold produce a CouplingRow.

    Spec: REQ-LEARN-056
    """
    model = _make_ising(input_dim=64)
    gen = IsingConstraintGenerator(model, threshold=3)
    types = ["carry_error", "sign_error", "unit_error", "comparison_error", "overflow_error"]
    patterns = [
        ErrorPattern(pattern_type=t, count=5, example_step=f"example of {t}")
        for t in types
    ]

    rows = gen.synthesize_from_memory(patterns)

    assert len(rows) == 5, f"expected 5 CouplingRows, got {len(rows)}"
    row_types_j = {r.J_value for r in rows}
    assert -1.0 in row_types_j
    assert 1.0 in row_types_j


# ---------------------------------------------------------------------------
# REQ-LEARN-056: synthesize_from_memory returns [] when all counts < threshold
# ---------------------------------------------------------------------------


def test_synthesize_returns_empty_when_all_below_threshold() -> None:
    """synthesize_from_memory MUST return [] when all pattern counts < threshold.

    Spec: REQ-LEARN-056, SCENARIO-LEARN-101
    """
    model = _make_ising()
    gen = IsingConstraintGenerator(model, threshold=3)
    patterns = [
        ErrorPattern(pattern_type="carry_error", count=2, example_step="only 2 occurrences"),
        ErrorPattern(pattern_type="sign_error", count=1, example_step="just once"),
    ]

    rows = gen.synthesize_from_memory(patterns)

    assert rows == [], f"expected empty list when all counts below threshold, got {rows}"


def test_synthesize_threshold_boundary_exactly_three() -> None:
    """count=3 meets threshold; count=2 does not.

    Spec: REQ-LEARN-056
    """
    model = _make_ising()
    gen = IsingConstraintGenerator(model, threshold=3)
    patterns = [
        ErrorPattern(pattern_type="carry_error", count=3, example_step="exactly 3"),
        ErrorPattern(pattern_type="sign_error", count=2, example_step="only 2"),
    ]

    rows = gen.synthesize_from_memory(patterns)

    assert len(rows) == 1
    assert rows[0].J_value == -1.0


def test_synthesize_unknown_pattern_type_skipped() -> None:
    """Unknown pattern_type must be silently skipped, not raise.

    Spec: REQ-LEARN-056
    """
    model = _make_ising()
    gen = IsingConstraintGenerator(model, threshold=3)
    patterns = [
        ErrorPattern(pattern_type="alien_error", count=99, example_step="unknown"),
    ]

    rows = gen.synthesize_from_memory(patterns)

    assert rows == []


# ---------------------------------------------------------------------------
# REQ-LEARN-056 / SCENARIO-LEARN-101: inject_couplings extends J without replacement
# ---------------------------------------------------------------------------


def test_inject_couplings_adds_to_existing_coupling() -> None:
    """inject_couplings MUST add J_value to existing coupling[var1, var2], not replace.

    Spec: REQ-LEARN-056, SCENARIO-LEARN-101
    """
    model = _make_ising(input_dim=10)
    gen = IsingConstraintGenerator(model, threshold=3)

    # Record all original coupling values
    original_coupling = jnp.array(model.coupling)

    row = CouplingRow(var1=0, var2=1, J_value=-1.0)
    gen.inject_couplings([row])

    # The target cells should have shifted by J_value
    assert float(model.coupling[0, 1]) == pytest.approx(float(original_coupling[0, 1]) + (-1.0))
    assert float(model.coupling[1, 0]) == pytest.approx(float(original_coupling[1, 0]) + (-1.0))

    # All other cells must be unchanged
    for i in range(10):
        for j in range(10):
            if (i, j) in {(0, 1), (1, 0)}:
                continue
            assert float(model.coupling[i, j]) == pytest.approx(float(original_coupling[i, j])), (
                f"coupling[{i},{j}] changed unexpectedly"
            )


def test_inject_couplings_empty_list_is_noop() -> None:
    """inject_couplings with an empty list must not modify the coupling matrix.

    Spec: REQ-LEARN-056
    """
    model = _make_ising(input_dim=8)
    gen = IsingConstraintGenerator(model, threshold=3)
    original = jnp.array(model.coupling)

    gen.inject_couplings([])

    assert jnp.allclose(model.coupling, original), "empty inject_couplings must not modify J"


def test_inject_couplings_cumulative_injections() -> None:
    """Two inject calls on the same cell accumulate (not overwrite).

    Spec: REQ-LEARN-056, SCENARIO-LEARN-101
    """
    model = _make_ising(input_dim=10)
    gen = IsingConstraintGenerator(model, threshold=3)

    original_val = float(model.coupling[2, 3])
    row = CouplingRow(var1=2, var2=3, J_value=-0.5)

    gen.inject_couplings([row])
    gen.inject_couplings([row])

    expected = original_val + 2 * (-0.5)
    assert float(model.coupling[2, 3]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# REQ-LEARN-057: compute_honest_verdict helper
# ---------------------------------------------------------------------------


def test_honest_verdict_positive() -> None:
    """Positive delta with patterns above threshold yields constraint_addition_positive.

    Spec: REQ-LEARN-057
    """
    from scripts.experiment_788_constraint_addition_from_memory import compute_honest_verdict

    assert compute_honest_verdict(0.05, 3) == "constraint_addition_positive"


def test_honest_verdict_zero() -> None:
    """Zero delta with patterns yields constraint_addition_zero.

    Spec: REQ-LEARN-057
    """
    from scripts.experiment_788_constraint_addition_from_memory import compute_honest_verdict

    assert compute_honest_verdict(0.0, 2) == "constraint_addition_zero"


def test_honest_verdict_negative() -> None:
    """Negative delta with patterns yields constraint_addition_negative.

    Spec: REQ-LEARN-057
    """
    from scripts.experiment_788_constraint_addition_from_memory import compute_honest_verdict

    assert compute_honest_verdict(-0.02, 1) == "constraint_addition_negative"


def test_honest_verdict_insufficient_patterns() -> None:
    """No patterns above threshold yields insufficient_patterns regardless of delta.

    Spec: REQ-LEARN-057
    """
    from scripts.experiment_788_constraint_addition_from_memory import compute_honest_verdict

    assert compute_honest_verdict(0.99, 0) == "insufficient_patterns"
    assert compute_honest_verdict(-0.99, 0) == "insufficient_patterns"


# ---------------------------------------------------------------------------
# REQ-LEARN-057: end-to-end artifact schema validation
# ---------------------------------------------------------------------------


def test_experiment_788_produces_valid_artifact(tmp_path: Path) -> None:
    """Running main() produces a JSON artifact with all required schema fields.

    Spec: REQ-LEARN-056, REQ-LEARN-057
    """
    import scripts.experiment_788_constraint_addition_from_memory as exp788

    deliverable = tmp_path / "experiment_788_constraint_addition_from_memory.json"
    original = exp788.DELIVERABLE
    exp788.DELIVERABLE = str(deliverable)
    # Also patch template deliverable
    original_tmpl_deliverable = None
    try:
        exp788.main()
        assert deliverable.exists(), "artifact JSON must be written by main()"
        artifact = json.loads(deliverable.read_text())

        required_fields = [
            "n_constraints_added",
            "net_improvement_dynamic",
            "net_improvement_static",
            "constraint_addition_delta",
            "n_patterns_above_threshold",
            "honest_verdict",
        ]
        for field in required_fields:
            assert field in artifact, f"artifact missing required field: {field}"

        valid_verdicts = {
            "constraint_addition_positive",
            "constraint_addition_zero",
            "constraint_addition_negative",
            "insufficient_patterns",
        }
        assert artifact["honest_verdict"] in valid_verdicts, (
            f"invalid honest_verdict: {artifact['honest_verdict']}"
        )
    finally:
        exp788.DELIVERABLE = original
