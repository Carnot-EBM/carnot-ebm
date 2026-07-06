"""Tests for Exp 5299 deterministic constraint-LNS repair fixture.

Spec refs: REQ-VERIFY-5299, SCENARIO-VERIFY-5299.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5299_constraint_lns_solver_repair_fixture_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _row_by_class(rows: list[dict[str, object]], instance_class: str) -> dict[str, object]:
    return next(row for row in rows if row["instance_class"] == instance_class)


def test_req_verify_5299_spec_declares_constraint_lns_contract() -> None:
    """REQ-VERIFY-5299: OpenSpec anchors the deterministic LNS fixture."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5299") : spec.index("### REQ-VERIFY-5272")
    ]

    for marker in (
        "REQ-VERIFY-5299",
        "SCENARIO-VERIFY-5299",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "destroyed variables",
        "declarative constraint-group metadata",
        "solver-only fallback baseline",
        "format-valid",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5299_instances_cover_required_lns_classes() -> None:
    """REQ-VERIFY-5299: instance classes include aligned, misleading, and no-op repairs."""

    instances = mod.build_lns_instances()
    counts = mod.instance_class_counts(instances)

    assert counts == {
        "aligned_repair": 1,
        "misleading_repair": 1,
        "neutral_noop_repair": 1,
        "malformed_control": 1,
        "semantic_wrong_control": 1,
    }
    assert {instance.source_fixture_id for instance in instances} == {"small_pair_sum"}
    assert all(instance.constraint_groups for instance in instances)
    assert all(instance.source_artifact == str(mod.v5278.RESULT_RELATIVE_PATH) for instance in instances)

    metadata = mod.constraint_group_metadata()
    assert {group["group_id"] for group in metadata} == {
        "a_one_hot",
        "b_one_hot",
        "sum_and_order_relation",
    }
    assert all(group["clause_indices"] for group in metadata)
    assert all(group["authority"] == "Minisat22 CDCL over original CNF" for group in metadata)


def test_scenario_verify_5299_aligned_and_noop_repairs_are_solver_accepted() -> None:
    """SCENARIO-VERIFY-5299: solver-accepted repairs still validate against the CNF."""

    rows = [mod.run_lns_instance(instance) for instance in mod.build_lns_instances()]
    aligned = _row_by_class(rows, "aligned_repair")
    neutral = _row_by_class(rows, "neutral_noop_repair")

    assert aligned["repair"]["solver_decision"] == "accepted"
    assert aligned["repair"]["primary_status"] == "sat"
    assert aligned["repair"]["candidate_format_valid"] is True
    assert aligned["fallback"]["used"] is False
    assert aligned["telemetry"]["destroy_operator"] == "destroy_full_assignment_neighborhood"
    assert aligned["telemetry"]["destroyed_variables"]
    assert aligned["telemetry"]["destroyed_clauses"]
    assert mod.verify_model(mod.base_clauses(), aligned["final"]["model"])

    assert neutral["repair"]["solver_decision"] == "accepted"
    assert neutral["telemetry"]["destroy_operator"] == "destroy_none_noop"
    assert neutral["telemetry"]["destroyed_variables"] == []
    assert neutral["telemetry"]["destroyed_clauses"] == []
    assert neutral["fallback"]["used"] is False
    assert mod.verify_model(mod.base_clauses(), neutral["final"]["model"])

    for row in (aligned, neutral):
        assert set(row["telemetry"]["solver_counters"]) == {
            "conflicts",
            "decisions",
            "propagations",
            "restarts",
        }


def test_scenario_verify_5299_rejects_malformed_structured_repair() -> None:
    """SCENARIO-VERIFY-5299: malformed repair output is rejected before solver scoring."""

    instance = next(
        row for row in mod.build_lns_instances() if row.instance_class == "malformed_control"
    )
    result = mod.run_lns_instance(instance)

    assert result["repair"]["candidate_format_valid"] is False
    assert result["repair"]["solver_decision"] == "rejected"
    assert result["repair"]["primary_status"] == "not_run"
    assert "assignments" in result["repair"]["rejection_reason"]
    assert result["fallback"]["used"] is True
    assert result["fallback"]["overwrite_count"] == 0
    assert result["final"]["status"] == "sat"
    assert mod.verify_model(instance.clauses, result["final"]["model"])
    assert mod.count_unsafe_false_accepts([result]) == 0


def test_scenario_verify_5299_rejects_semantic_errors_and_misleading_repairs() -> None:
    """SCENARIO-VERIFY-5299: format-valid semantic errors cannot become repair accepts."""

    rows = [mod.run_lns_instance(instance) for instance in mod.build_lns_instances()]
    semantic_wrong = _row_by_class(rows, "semantic_wrong_control")
    misleading = _row_by_class(rows, "misleading_repair")

    for row in (semantic_wrong, misleading):
        assert row["repair"]["candidate_format_valid"] is True
        assert row["repair"]["solver_decision"] == "rejected"
        assert row["repair"]["primary_status"] == "unsat"
        assert row["repair"]["candidate_solver_accepted"] is False
        assert row["fallback"]["used"] is True
        assert row["fallback"]["overwrite_count"] > 0
        assert row["final"]["status"] == row["baseline"]["status"] == "sat"
        assert row["solver_correctness_preserved"] is True
        assert mod.verify_model(mod.base_clauses(), row["final"]["model"])

    assert mod.count_unsafe_false_accepts([semantic_wrong, misleading]) == 0


def test_req_verify_5299_benchmark_reports_baseline_and_ready_gate() -> None:
    """REQ-VERIFY-5299: benchmark compares LNS repair with solver-only baseline."""

    benchmark = mod.run_benchmark()
    rows = benchmark["per_instance_results"]

    assert benchmark["constraint_lns_fixture_ready"] is True
    assert benchmark["solver_correctness_preserved"] is True
    assert benchmark["unsafe_false_accepts"] == 0
    assert benchmark["instance_class_counts"]["aligned_repair"] == 1
    assert benchmark["instance_class_counts"]["semantic_wrong_control"] == 1
    assert benchmark["classical_baseline_results"]["baseline_name"] == "solver_only_fallback"
    assert benchmark["classical_baseline_results"]["all_baseline_models_valid"] is True

    for row in rows:
        assert row["baseline"]["status"] == row["final"]["status"]
        assert row["baseline"]["model_valid"] is True
        assert row["telemetry"]["repair_operator"]
        assert "repair_candidate" in row["telemetry"]


def test_req_verify_5299_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5299: artifact exposes required fields and the bare ready gate."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "unit exp5299", "outcome": "passed"}]
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.25,
        tests_run=tests_run,
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "usable" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["constraint_lns_fixture_ready"] is True
    assert isinstance(artifact["constraint_lns_fixture_ready"], bool)
    assert _value(artifact, "instance_class_counts")["misleading_repair"] == 1
    assert _value(artifact, "solver_correctness_preserved") is True
    assert _value(artifact, "unsafe_false_accepts") == 0
    assert artifact["tests_run"] == tests_run
    assert "REQ-VERIFY-5299" in artifact["spec_refs"]
    assert len(artifact["reproducibility_checksum"]) == 64


def test_req_verify_5299_validation_fails_closed_on_schema_drift() -> None:
    """REQ-VERIFY-5299: invalid readiness, substrate, or safety claims are rejected."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5299", "outcome": "passed"}],
    )

    missing = copy.deepcopy(artifact)
    missing.pop("destroy_repair_telemetry")
    with pytest.raises(AssertionError, match="missing required field"):
        mod.validate_artifact(missing)

    broken = copy.deepcopy(artifact)
    broken["constraint_lns_fixture_ready"] = {"value": True}
    with pytest.raises(AssertionError, match="bare bool"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm")
    with pytest.raises(AssertionError, match=mod.INFERENCE_SUBSTRATE):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["unsafe_false_accepts"] = mod.wrap_field("unsafe_false_accepts", 1)
    with pytest.raises(AssertionError, match="unsafe"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_verify_5299() -> None:
    """SCENARIO-VERIFY-5299: checked-in deliverable satisfies the V484 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["constraint_lns_fixture_ready"] is True
    assert _value(artifact, "solver_correctness_preserved") is True
    assert _value(artifact, "unsafe_false_accepts") == 0
    assert _value(artifact, "classical_baseline_results")["baseline_name"] == (
        "solver_only_fallback"
    )
