"""Tests for Exp 5108 KAN-PWA/MILP scale stress test.

Spec refs: REQ-KAN-5108, SCENARIO-KAN-5108.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5108_kan_pwa_milp_scale_stress_test as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
ARTIFACT_PATH = REPO / mod.RESULT_RELATIVE_PATH

# Small, fast unit counts for CI -- the real N=5..100 production sweep is run manually
# (it took ~7 minutes and legitimately found the scale wall; re-running it on every
# `pytest` invocation would make the suite unusably slow).
FAST_UNIT_COUNTS = (2, 3)
FAST_TIMEOUT_MS = 30_000


def test_req_kan_5108_spec_declares_scale_stress_contract() -> None:
    """REQ-KAN-5108: OpenSpec anchors the scale-stress-test contract before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-5108" in spec
    assert "SCENARIO-KAN-5108" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "100" in spec  # the production reference unit count


def test_scenario_kan_5108_binary_variable_count_matches_exp5091_exp5098_formula() -> None:
    """SCENARIO-KAN-5108: the N-unit encoding reproduces the exact prior binary-variable counts."""

    if importlib_z3_missing():
        pytest.skip("Z3 is not available in this environment")

    abstraction_2 = mod.build_n_unit_abstraction(2, seed=mod.RANDOM_SEED)
    abstraction_3 = mod.build_n_unit_abstraction(3, seed=mod.RANDOM_SEED + 1)

    # exp5091 (2-unit) had binary_variable_count=6; exp5098 (3-unit) had 9. Same N_KNOTS=4 ->
    # 3 segments/unit, so this generalized builder must reproduce those exact counts.
    assert abstraction_2.binary_variable_count == 6
    assert abstraction_3.binary_variable_count == 9


def test_req_kan_5108_solver_preserves_adversarial_rigor_at_small_n() -> None:
    """REQ-KAN-5108: the false-property control and margin abstention both hold at small N."""

    if importlib_z3_missing():
        pytest.skip("Z3 is not available in this environment")

    result = mod.solve_one_n(3, seed=mod.RANDOM_SEED, timeout_ms=FAST_TIMEOUT_MS)

    assert result.solver_status == "optimal"
    assert result.timed_out is False
    assert result.binary_variable_count == 9
    assert result.true_property_status == "verified"
    assert result.false_control_counterexampled is True
    assert result.false_control_counterexample is not None
    assert result.margin_property_status == "unproved_approximation_budget"


def test_req_kan_5108_timeout_path_is_honestly_reported() -> None:
    """REQ-KAN-5108: a solve that cannot finish within the budget reports unknown_timeout, not a fabricated result."""

    if importlib_z3_missing():
        pytest.skip("Z3 is not available in this environment")

    # A near-zero timeout on a real (small but non-trivial) instance deterministically times out,
    # exercising the honest-timeout path without waiting for a genuinely large N to blow up.
    result = mod.solve_one_n(3, seed=mod.RANDOM_SEED, timeout_ms=1)

    assert result.timed_out is True
    assert result.solver_status == "unknown_timeout"
    assert result.certified_upper_bound is None
    assert result.false_control_counterexampled is None
    assert result.margin_property_status is None


def test_req_kan_5108_sweep_stops_at_first_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-5108: run_sweep must not attempt a larger N after a timeout.

    Uses a stubbed solve_one_n (deterministic: N=2/3 solve, N=5 times out) rather than racing
    real Z3 solve-time thresholds -- real solve times are fast and machine-sensitive at this
    tiny scale, so a razor-thin timeout window would be a flaky test, not a real stress test."""

    def _fake_solve_one_n(n_units: int, seed: int, timeout_ms: int) -> mod.NResult:
        timed_out = n_units >= 5
        return mod.NResult(
            n_units=n_units,
            binary_variable_count=3 * n_units,
            constraint_count=21 * n_units + 1,
            pwa_piece_count=3 * n_units,
            solver_status="unknown_timeout" if timed_out else "optimal",
            timed_out=timed_out,
            solve_time_s=0.01 if not timed_out else float(timeout_ms) / 1000.0,
            certified_upper_bound=None if timed_out else 1.0,
            witness_inputs=None,
            true_property_status=None if timed_out else "verified",
            false_control_counterexampled=None if timed_out else True,
            false_control_counterexample=None,
            margin_property_status=None if timed_out else "unproved_approximation_budget",
        )

    monkeypatch.setattr(mod, "solve_one_n", _fake_solve_one_n)
    results = mod.run_sweep(unit_counts=(2, 3, 5, 10), timeout_ms=1000)

    n_units_attempted = [r.n_units for r in results]
    assert n_units_attempted == [2, 3, 5], (
        "sweep must stop at the first timeout (5), never attempting 10"
    )
    assert results[-1].timed_out is True


def test_req_kan_5108_artifact_fields_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-5108: artifact emits required schema fields and passes validate_artifact."""

    if importlib_z3_missing():
        pytest.skip("Z3 is not available in this environment")

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path, unit_counts=FAST_UNIT_COUNTS, timeout_ms=FAST_TIMEOUT_MS
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(payload["field_principles"])
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert "live_llm" not in artifact["inference_substrate"]
    assert artifact["realistic_kan_unit_count_reference"] == 100
    assert artifact["unit_counts_tested"] == [2, 3]
    assert artifact["adversarial_rigor_preserved_at_scale"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["reached_production_reference"] is False  # FAST_UNIT_COUNTS never reaches 100
    assert len(artifact["reproducibility_checksum"]) == 64
    mod.validate_artifact(artifact)


def test_deliverable_file_validates_and_reports_the_real_scale_wall() -> None:
    """SCENARIO-KAN-5108: the committed real-sweep deliverable is internally consistent and
    honestly reports the actual scale wall found (does NOT assert a fixed success verdict --
    finding a wall well below the production reference is a valid, expected outcome here,
    not a bug to force-pass)."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["adversarial_rigor_preserved_at_scale"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["largest_n_reached"] >= 5, "the smallest configured N must have solved"
    assert artifact["solver_timeout_hit"] is True, "the real sweep found the scale wall"
    assert artifact["largest_n_reached"] < artifact["realistic_kan_unit_count_reference"], (
        "the real result did not reach the production reference -- an honest negative"
    )


def importlib_z3_missing() -> bool:
    import importlib.util

    return importlib.util.find_spec("z3") is None
