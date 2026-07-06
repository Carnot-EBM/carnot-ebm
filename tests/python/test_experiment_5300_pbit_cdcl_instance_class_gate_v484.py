"""Tests for Exp 5300 p-bit/CDCL instance-class gate.

Spec refs: REQ-VERIFY-5300, SCENARIO-VERIFY-5300.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5300_pbit_cdcl_instance_class_gate_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _row_by_class(rows: list[dict[str, object]], instance_class: str) -> dict[str, object]:
    return next(row for row in rows if row["instance_class"] == instance_class)


def test_req_verify_5300_spec_declares_instance_class_gate_contract() -> None:
    """REQ-VERIFY-5300: OpenSpec anchors the p-bit/CDCL gate artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5300") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5300",
        "SCENARIO-VERIFY-5300",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "assumption conflict prefix",
        "factor alignment score",
        "LNS repair agreement",
        "hardware_speedup_claimed.value",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5300_builds_replay_instances_from_5292_and_5299() -> None:
    """REQ-VERIFY-5300: replay instances reuse Exp 5292 and Exp 5299 fixtures."""

    instances = mod.build_gate_instances()
    by_class = {instance.instance_class: instance for instance in instances}

    assert mod.CLASSIFIER_KIND == "deterministic_threshold_rules_no_training"
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
    assert by_class["aligned_factor_sat"].assumption_literals == (3, 8)
    assert by_class["misleading_factor_sat"].assumption_literals == (4, 7)
    assert by_class["neutral_factor_sat"].assumption_literals == ()
    assert by_class["semantic_wrong_control"].assumption_literals == (2, 8)
    assert all(instance.hardware_execution is False for instance in instances)


def test_scenario_verify_5300_blocks_misleading_assumption_classes() -> None:
    """SCENARIO-VERIFY-5300: contradictions and LNS rejections block guidance."""

    benchmark = mod.run_benchmark()
    rows = benchmark["per_instance_results"]

    for instance_class in (
        "misleading_factor_sat",
        "misleading_repair",
        "semantic_wrong_control",
    ):
        row = _row_by_class(rows, instance_class)
        features = row["gate_features"]

        assert row["gate_decision"]["route"] == "block"
        assert row["gated"]["used_solver_only_fallback"] is True
        assert row["gated"]["metrics"] == row["solver_only"]["metrics"]
        assert features["assumption_conflict_prefix"] == "unsat_under_assumptions"
        assert features["contradiction_count"] > 0
        assert features["solver_overwrite_count"] > 0

    blocked = benchmark["misleading_class_blocked"]
    assert blocked["all_misleading_blocked"] is True
    assert blocked["blocked_classes"] == [
        "misleading_factor_sat",
        "misleading_repair",
        "semantic_wrong_control",
    ]


def test_req_verify_5300_routes_helpful_and_neutral_classes() -> None:
    """REQ-VERIFY-5300: helpful or neutral classes retain safe guidance."""

    rows = mod.run_benchmark()["per_instance_results"]
    aligned = _row_by_class(rows, "aligned_factor_sat")
    neutral = _row_by_class(rows, "neutral_factor_sat")
    lns_aligned = _row_by_class(rows, "aligned_repair")
    malformed = _row_by_class(rows, "malformed_control")

    for row in (aligned, lns_aligned):
        assert row["gate_decision"]["route"] == "allow"
        assert row["gate_features"]["factor_alignment_score"] == 1.0
        assert row["gated"]["metrics"]["conflicts"] < row["solver_only"]["metrics"]["conflicts"]
        assert row["correctness_preserved"] is True

    assert neutral["gate_decision"]["route"] == "allow"
    assert neutral["gate_decision"]["reason"] == "no_assumptions_neutral"
    assert neutral["gated"]["metrics"] == neutral["solver_only"]["metrics"]

    assert malformed["gate_decision"]["route"] == "block"
    assert malformed["gate_decision"]["reason"] == "malformed_or_rejected_lns_fixture"
    assert malformed["gated"]["metrics"] == malformed["solver_only"]["metrics"]


def test_req_verify_5300_benchmark_compares_three_arms_and_preserves_correctness() -> None:
    """REQ-VERIFY-5300: gated guidance beats ungated harm while preserving labels."""

    benchmark = mod.run_benchmark()
    aggregate = benchmark["aggregate_metrics"]

    assert benchmark["correctness_preserved"] is True
    assert benchmark["pbit_gate_ready"] is True
    assert aggregate["gated"]["conflicts"] < aggregate["ungated"]["conflicts"]
    assert aggregate["gated"]["conflicts"] < aggregate["solver_only"]["conflicts"]
    assert aggregate["ungated_vs_gated_delta"]["conflicts_saved_by_gate"] > 0
    assert aggregate["solver_only_vs_gated_delta"]["conflicts_saved"] > 0
    assert set(benchmark["conflicts_saved_by_class"]) == {
        "aligned_factor_sat",
        "misleading_factor_sat",
        "neutral_factor_sat",
        "aligned_repair",
        "misleading_repair",
        "neutral_noop_repair",
        "malformed_control",
        "semantic_wrong_control",
    }
    for row in benchmark["per_instance_results"]:
        assert set(row["solver_only"]["metrics"]) == {
            "conflicts",
            "decisions",
            "propagations",
            "restarts",
            "wall_clock_s",
        }
        assert set(row["ungated"]["metrics"]) == set(row["solver_only"]["metrics"])
        assert set(row["gated"]["metrics"]) == set(row["solver_only"]["metrics"])


def test_scenario_verify_5300_correctness_guard_rejects_label_and_model_drift() -> None:
    """SCENARIO-VERIFY-5300: label or model drift fails the correctness guard."""

    instance = mod.build_gate_instances()[0]
    solver_only = mod.cdcl.run_cdcl(instance.clauses, n_vars=instance.n_vars)
    valid_ungated = {
        "final_status": solver_only.status,
        "final_model": list(solver_only.model),
    }
    valid_gated = {
        "final_status": solver_only.status,
        "final_model": list(solver_only.model),
    }
    invalid_model = list(range(1, instance.n_vars + 1))
    fake_sat = mod.cdcl.CdclRun(
        status="sat",
        model=tuple(invalid_model),
        metrics=solver_only.metrics,
    )

    assert mod._row_correctness_preserved(instance, fake_sat, valid_ungated, valid_gated) is False
    assert mod._row_correctness_preserved(
        instance,
        solver_only,
        {"final_status": "unsat", "final_model": []},
        valid_gated,
    ) is False
    assert mod._row_correctness_preserved(
        instance,
        solver_only,
        valid_ungated,
        {"final_status": "unsat", "final_model": []},
    ) is False
    assert mod._row_correctness_preserved(
        instance,
        solver_only,
        {"final_status": solver_only.status, "final_model": invalid_model},
        valid_gated,
    ) is False


def test_req_verify_5300_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5300: artifact exposes principle-wrapped gate fields."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5300", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.33,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "p-bit/CDCL gate helped" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "pbit_gate_ready") is True
    assert _value(artifact, "misleading_class_blocked")["all_misleading_blocked"] is True
    assert _value(artifact, "correctness_preserved") is True
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert artifact["tests_run"] == tests_run
    assert "REQ-VERIFY-5300" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5300_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5300: invalid gate, correctness, or hardware claims are rejected."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5300", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["hardware_speedup_claimed"] = mod.wrap_field("hardware_speedup_claimed", True)
    with pytest.raises(AssertionError, match="hardware"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["correctness_preserved"] = mod.wrap_field("correctness_preserved", False)
    with pytest.raises(AssertionError, match="correctness"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["misleading_class_blocked"] = mod.wrap_field(
        "misleading_class_blocked",
        {"all_misleading_blocked": False, "blocked_classes": []},
    )
    with pytest.raises(AssertionError, match="misleading"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5300() -> None:
    """SCENARIO-VERIFY-5300: checked-in deliverable satisfies the V484 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "pbit_gate_ready") is True
    assert _value(artifact, "correctness_preserved") is True
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "ungated_vs_gated_delta")["conflicts_saved_by_gate"] > 0
