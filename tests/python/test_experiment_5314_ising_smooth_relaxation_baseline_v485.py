"""Tests for Exp 5314 CPU smooth Ising relaxation baseline.

Spec refs: REQ-VERIFY-5314, SCENARIO-VERIFY-5314.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5314_ising_smooth_relaxation_baseline_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _row_by_class(rows: list[dict[str, object]], instance_class: str) -> dict[str, object]:
    return next(row for row in rows if row["instance_class"] == instance_class)


def test_req_verify_5314_spec_declares_smooth_relaxation_contract() -> None:
    """REQ-VERIFY-5314: OpenSpec anchors the CPU smooth-relaxation artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5314") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5314",
        "SCENARIO-VERIFY-5314",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "one-flip local-minimum",
        "`smooth_relaxation_ready`",
        "`cdcl_fallback_authoritative=true`",
        "`no_hardware_speedup_claim=true`",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5314_reuses_5299_and_5300_fixture_instances() -> None:
    """REQ-VERIFY-5314: fixture rows come from bounded upstream classes."""

    instances = mod.build_relaxation_instances()
    by_class = {instance.instance_class: instance for instance in instances}

    assert {instance.source_experiment for instance in instances} == {"exp5292", "exp5299"}
    assert set(by_class) == {
        "aligned_factor_sat",
        "misleading_factor_sat",
        "neutral_factor_sat",
        "aligned_repair",
        "misleading_repair",
        "neutral_noop_repair",
        "malformed_control",
        "semantic_wrong_control",
    }
    assert all(instance.source_fixture_id == "small_pair_sum" for instance in instances)
    assert all(instance.hardware_execution is False for instance in instances)
    assert by_class["aligned_factor_sat"].seed_literals == (3, 8)
    assert by_class["misleading_factor_sat"].seed_literals == (4, 7)
    assert by_class["semantic_wrong_control"].seed_literals == (2, 8)


def test_scenario_verify_5314_one_flip_local_minima_are_diagnosed() -> None:
    """SCENARIO-VERIFY-5314: every hint passes the one-flip local-minimum check."""

    rows = mod.run_benchmark()["per_instance_results"]

    assert all(row["smooth_relaxation"]["one_flip_local_minimum"] for row in rows)
    aligned = _row_by_class(rows, "aligned_factor_sat")
    misleading = _row_by_class(rows, "misleading_factor_sat")
    neutral = _row_by_class(rows, "neutral_factor_sat")

    assert aligned["smooth_relaxation"]["final_energy"] == 0
    assert aligned["smooth_relaxation"]["route"] == "use_hint"
    assert aligned["smooth_relaxation"]["metrics"]["conflicts"] < aligned["solver_only"]["metrics"]["conflicts"]

    assert misleading["smooth_relaxation"]["final_energy"] > 0
    assert misleading["smooth_relaxation"]["route"] == "fallback_solver_only"
    assert misleading["smooth_relaxation"]["fallback_used"] is True

    assert neutral["smooth_relaxation"]["final_energy"] > 0
    assert neutral["smooth_relaxation"]["route"] == "fallback_solver_only"


def test_scenario_verify_5314_misleading_classes_fallback_without_harm() -> None:
    """SCENARIO-VERIFY-5314: misleading local minima cannot override CDCL."""

    benchmark = mod.run_benchmark()
    rows = benchmark["per_instance_results"]

    for instance_class in mod.MISLEADING_CLASSES:
        row = _row_by_class(rows, instance_class)
        smooth = row["smooth_relaxation"]

        assert smooth["route"] == "fallback_solver_only"
        assert smooth["fallback_used"] is True
        assert smooth["final_status"] == row["solver_only"]["status"]
        assert smooth["final_model"] == row["solver_only"]["model"]
        assert smooth["metrics"] == row["solver_only"]["metrics"]
        assert row["conflict_delta_vs_solver_only"] == 0
        assert row["final_assignment_symbolically_valid"] is True

    behavior = benchmark["misleading_class_behavior"]
    assert behavior["misleading_class_harm"] == 0
    assert behavior["blocked_misleading_classes"] == list(mod.MISLEADING_CLASSES)
    assert behavior["ungated_hint_added_conflicts"] > 0


def test_req_verify_5314_compares_smooth_pbit_and_solver_baselines() -> None:
    """REQ-VERIFY-5314: benchmark records conflict deltas and fallback rate."""

    benchmark = mod.run_benchmark()
    comparison = benchmark["pbit_cdcl_comparison"]

    assert benchmark["smooth_relaxation_ready"] is True
    assert benchmark["one_flip_checks_passed"] is True
    assert benchmark["cdcl_fallback_authoritative"] is True
    assert benchmark["conflict_delta_vs_solver_only"] > 0
    assert 0.0 < benchmark["fallback_rate"] < 1.0
    assert comparison["solver_only_conflicts"] > comparison["smooth_conflicts"]
    assert comparison["pbit_ungated_conflicts"] >= comparison["smooth_conflicts"]
    assert comparison["smooth_vs_pbit_gated_conflict_delta"] == 0
    assert set(benchmark["conflict_delta_by_class"]) == {
        row["instance_class"] for row in benchmark["per_instance_results"]
    }


def test_scenario_verify_5314_symbolic_guard_rejects_label_or_model_drift() -> None:
    """SCENARIO-VERIFY-5314: final SAT labels and models must stay solver-valid."""

    instance = mod.build_relaxation_instances()[0]
    solver_only = mod.cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    invalid_model = list(range(1, instance.n_vars + 1))

    assert mod.final_assignment_symbolically_valid(instance, solver_only.status, solver_only.model)
    assert not mod.final_assignment_symbolically_valid(instance, "unsat", solver_only.model)
    assert not mod.final_assignment_symbolically_valid(instance, solver_only.status, invalid_model)


def test_req_verify_5314_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5314: artifact exposes principle fields and bare gates."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5314", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.25,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["smooth_relaxation_ready"] is True
    assert artifact["one_flip_checks_passed"] is True
    assert artifact["cdcl_fallback_authoritative"] is True
    assert artifact["conflict_delta_vs_solver_only"] > 0
    assert artifact["misleading_class_harm"] == 0
    assert artifact["no_hardware_speedup_claim"] is True
    assert _value(artifact, "tests_run") == tests_run
    assert "REQ-VERIFY-5314" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5314_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5314: invalid authority, speedup, or readiness claims fail."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5314", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["no_hardware_speedup_claim"] = False
    with pytest.raises(AssertionError, match="hardware"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["cdcl_fallback_authoritative"] = False
    with pytest.raises(AssertionError, match="CDCL"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["smooth_relaxation_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["misleading_class_harm"] = 1
    with pytest.raises(AssertionError, match="misleading"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5314() -> None:
    """SCENARIO-VERIFY-5314: deliverable JSON satisfies the V485 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["smooth_relaxation_ready"] is True
    assert artifact["one_flip_checks_passed"] is True
    assert artifact["cdcl_fallback_authoritative"] is True
    assert artifact["no_hardware_speedup_claim"] is True
