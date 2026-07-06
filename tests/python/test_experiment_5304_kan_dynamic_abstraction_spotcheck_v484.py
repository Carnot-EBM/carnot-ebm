"""Tests for Exp 5304 dynamic KAN abstraction and spot-check diagnostics.

Spec refs: REQ-KAN-5304, SCENARIO-KAN-5304.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5304_kan_dynamic_abstraction_spotcheck_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _method(comparison: dict[str, object], method_id: str) -> dict[str, object]:
    methods = comparison["methods"]
    assert isinstance(methods, list)
    return next(row for row in methods if row["method_id"] == method_id)


def test_req_kan_5304_spec_declares_dynamic_diagnostic_contract() -> None:
    """REQ-KAN-5304: OpenSpec anchors the dynamic abstraction diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5304") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-KAN-5304",
        "SCENARIO-KAN-5304",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "static abstraction",
        "low-order ordering",
        "dynamic spot-check/refinement",
        "symbolic checker",
        "bounded region",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_kan_5304_accepts_true_property_and_rejects_false_property() -> None:
    """REQ-KAN-5304: all compared methods preserve true accept and false reject."""

    comparison = mod.run_method_comparison()

    for method_id in ("static_abstraction", "low_order_exp5291", "dynamic_spotcheck_refinement"):
        method = _method(comparison, method_id)
        assert method["certificate_success"] is True
        assert method["true_property_accepted"] is True
        assert method["false_property_rejected"] is True
        assert method["false_property_slack"] < 0.0
        assert method["bounded_scope_only"] is True

    success = comparison["certificate_success_by_method"]
    assert success["static_abstraction"] is True
    assert success["low_order_exp5291"] is True
    assert success["dynamic_spotcheck_refinement"] is True
    assert comparison["false_property_rejected"] is True


def test_req_kan_5304_dynamic_refinement_trigger_uses_slack_error_and_boundary() -> None:
    """REQ-KAN-5304: refinement is triggered by measured certificate signals."""

    comparison = mod.run_method_comparison()
    static = _method(comparison, "static_abstraction")
    dynamic = _method(comparison, "dynamic_spotcheck_refinement")
    help_metrics = comparison["dynamic_abstraction_helped"]

    assert dynamic["piece_count"] > static["piece_count"]
    assert dynamic["global_error_bound"] < static["global_error_bound"]
    assert dynamic["max_observed_envelope_gap"] < static["max_observed_envelope_gap"]
    assert dynamic["spotcheck_hit_rate"] > static["spotcheck_hit_rate"]
    assert help_metrics["helped"] is True
    assert help_metrics["success_improvement"] == pytest.approx(0.0)
    assert help_metrics["spotcheck_hit_rate_delta"] > 0.0
    assert set(dynamic["refinement_trigger"]["signals_seen"]) >= {
        "near_false_property_slack",
        "local_error",
        "boundary_proximity",
    }


def test_scenario_kan_5304_preserves_constraint_groups_and_symbolic_checker() -> None:
    """SCENARIO-KAN-5304: declarative groups survive symbolic validation."""

    comparison = mod.run_method_comparison()
    groups = mod.declarative_constraint_groups()
    symbolic = mod.symbolic_validate_comparison(comparison, groups)

    assert {group["group_id"] for group in groups} >= {
        "kan_component_envelopes",
        "pwa_piece_selectors",
        "factor_graph_boundary",
    }
    assert symbolic["valid"] is True
    assert symbolic["property_checks_valid"] is True
    assert symbolic["factor_boundary_valid"] is True
    assert symbolic["constraint_group_ids_seen"] == [group["group_id"] for group in groups]


def test_req_kan_5304_no_overclaim_outside_bounded_region() -> None:
    """REQ-KAN-5304: out-of-box points fail closed instead of widening the claim."""

    abstraction = mod.build_static_abstraction()

    assert mod.check_bounded_scope((0.6, 0.6, 0.6), abstraction) is True
    with pytest.raises(ValueError, match="outside certified bounded region"):
        mod.check_bounded_scope((0.6, 0.6), abstraction)
    with pytest.raises(ValueError, match="outside certified bounded region"):
        mod.check_bounded_scope((0.7, 0.6, 0.6), abstraction)

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5304", "outcome": "passed"}],
    )
    assert _value(artifact, "bounded_scope_only") is True
    assert any("no global KAN robustness claim" in limit for limit in artifact["claim_limits"])


def test_req_kan_5304_artifact_fields_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-5304: artifact emits required fields and validates fail-closed."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5304", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=result_path,
        duration_s=0.25,
        tests_run=tests_run,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "dynamic abstraction helped" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "dynamic_abstraction_helped")["helped"] is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "bounded_scope_only") is True
    assert artifact["tests_run"] == tests_run
    assert "REQ-KAN-5304" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64

    broken = copy.deepcopy(artifact)
    broken["false_property_rejected"] = mod.wrap_field("false_property_rejected", False)
    with pytest.raises(AssertionError, match="false property"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["bounded_scope_only"] = mod.wrap_field("bounded_scope_only", False)
    with pytest.raises(AssertionError, match="bounded scope"):
        mod.validate_artifact(broken)


def test_req_kan_5304_exact_fallback_and_partial_refinement_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KAN-5304: solver fallback and partial refinement stay honest."""

    abstraction = mod.build_static_abstraction()
    fallback = mod.solve_certificate(abstraction, solver_name="")
    partial_trigger = {
        "selected_component_indices": [0],
        "signals_seen": ["local_error"],
    }
    partial_refined = mod.refine_abstraction(abstraction, partial_trigger)
    partial_spot = mod.dynamic_spotcheck(partial_refined, partial_trigger)

    assert fallback.fallback_used is True
    assert fallback.solver_backend == "exact_vertex_enumeration_fallback"
    assert fallback.certificate_success is True
    assert fallback.false_property_rejected is True
    assert partial_refined.piece_count == abstraction.piece_count + 2
    assert partial_spot["passed"] is True
    assert mod._axis_samples((0.0, 1.0), 1) == (0.5,)

    monkeypatch.setattr(mod, "detect_solver", lambda: "")
    low_order = mod._low_order_summary(mod.uniform_spotcheck(abstraction))
    assert low_order["fallback_used"] is True
    assert low_order["certificate_success"] is True

    class BadEnvelope:
        component_count = 1
        input_box = ((0.0, 1.0),)

        def evaluate_actual(self, _point: tuple[float, ...]) -> float:
            return 1.0

        def evaluate_upper_envelope(self, _point: tuple[float, ...]) -> float:
            return 0.0

    bad_spot = mod._spotcheck_points(BadEnvelope(), [(0.5,)], profile="bad_envelope")
    assert bad_spot["envelope_violation_count"] == 1
    assert bad_spot["passed"] is False


def test_req_kan_5304_honest_verdict_blocked_and_null_edges() -> None:
    """REQ-KAN-5304: terminal verdict prefixes cover blocked and null paths."""

    comparison = mod.run_method_comparison()

    blocked = copy.deepcopy(comparison)
    blocked["symbolic_checker"]["valid"] = False
    assert mod._honest_verdict(blocked).startswith("blocked_")

    null = copy.deepcopy(comparison)
    null["dynamic_abstraction_helped"]["helped"] = False
    assert mod._honest_verdict(null).startswith("null:")


def test_deliverable_file_validates_for_scenario_kan_5304() -> None:
    """SCENARIO-KAN-5304: committed deliverable satisfies the V484 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "bounded_scope_only") is True
    assert _value(artifact, "false_property_rejected") is True
    assert _value(artifact, "dynamic_abstraction_helped")["helped"] is True
