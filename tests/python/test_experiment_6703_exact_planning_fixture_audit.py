"""Tests for the cold exact-planning fixture audit.

Spec: REQ-CONSTRAINT-6703, REQ-VERIFY-6703, REQ-SAFE-6703,
REQ-PIPELINE-6703, REQ-REPORT-6703, and their related scenarios.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6703_exact_planning_fixture_audit as exp


EXPECTED_SAMPLE_IDS = {
    "battery_dispatch-development-01",
    "battery_dispatch-headline-06",
    "battery_dispatch-headline-07",
    "inventory-development-01",
    "inventory-headline-02",
    "inventory-headline-03",
    "job_slot-development-00",
    "job_slot-headline-02",
    "job_slot-headline-07",
    "reservoir_control-development-01",
    "reservoir_control-headline-02",
    "reservoir_control-headline-04",
}


def passing_test_rows() -> list[dict[str, object]]:
    """Return complete command receipts for reducer and artifact tests."""

    rows: list[dict[str, object]] = []
    for check_id in exp.REQUIRED_TEST_CHECKS:
        rows.append(
            {
                "check_id": check_id,
                "command": f"check {check_id}",
                "exit_code": 0,
                "passed": True,
                "coverage_percent": 100.0 if check_id == "scoped_coverage" else None,
                "summary": "passed",
                "duration_s": 0.1,
            }
        )
    return rows


def tiny_spec(family: str) -> dict[str, object]:
    """Build a small typed specification with hand-checkable path totals."""

    common: dict[str, object] = {
        "family": family,
        "instance_id": f"tiny-{family}",
        "schema": "carnot.typed_finite_horizon_plan.v1",
        "stage_cost_shift": 0,
    }
    if family == "inventory":
        return {
            **common,
            "horizon": 1,
            "initial_state": {"stock": 0},
            "action_domain": [0, 1, 2],
            "parameters": {
                "actions": [0, 1, 2],
                "capacity": 2,
                "demand": [1],
                "holding_cost": 1,
                "initial": 0,
                "price": [2],
                "shortage_penalty": 5,
                "stage_cost_shift": 0,
                "terminal_holding_cost": 1,
            },
        }
    if family == "battery_dispatch":
        return {
            **common,
            "horizon": 1,
            "initial_state": {"charge": 1},
            "action_domain": [-1, 0, 1, 2],
            "parameters": {
                "actions": [-1, 0, 1, 2],
                "capacity": 2,
                "cycling_cost": 1,
                "initial": 1,
                "load": [1],
                "price": [3],
                "stage_cost_shift": 0,
                "terminal_penalty": 2,
                "terminal_target": 1,
            },
        }
    if family == "job_slot":
        return {
            **common,
            "horizon": 2,
            "initial_state": {"completed_mask": 0},
            "action_domain": [0, 1, 2],
            "parameters": {
                "actions": [0, 1, 2],
                "deadlines": [1, 1],
                "idle_cost": 2,
                "initial": 0,
                "missing_penalties": [5, 5],
                "prerequisites": [0, 1],
                "schedule_costs": [1, 1],
                "stage_cost_shift": 0,
            },
        }
    if family == "reservoir_control":
        return {
            **common,
            "horizon": 1,
            "initial_state": {"volume": 1},
            "action_domain": [0, 1, 2],
            "parameters": {
                "actions": [0, 1, 2],
                "capacity": 2,
                "deviation_penalty": 2,
                "flood_penalty": 1,
                "flood_threshold": 1,
                "inflow": [0],
                "initial": 1,
                "spill_penalty": 3,
                "stage_cost_shift": 0,
                "target_release": [1],
                "terminal_penalty": 3,
                "terminal_target": 0,
            },
        }
    raise AssertionError(f"unknown family: {family}")


def test_blinded_manifest_freezes_expected_identity_map() -> None:
    """REQ-CONSTRAINT-6703; SCENARIO-CONSTRAINT-6703-BLINDING."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    public_rows = exp.public_instance_rows(upstream)
    assert all("optimum" not in row for row in public_rows)
    manifest = exp.freeze_blinded_sample(public_rows)
    assert manifest["frozen_before_reported_label_read"] is True
    assert manifest["expected_instance_count"] == 12
    assert {row["instance"] for row in manifest["expected_identities"]} == EXPECTED_SAMPLE_IDS
    assert len(manifest["reveal_order"]) == 12
    assert manifest["manifest_hash"] == exp.manifest_checksum(manifest)

    duplicate = public_rows + [deepcopy(public_rows[0])]
    with pytest.raises(ValueError, match="duplicate public instance"):
        exp.freeze_blinded_sample(duplicate)
    contaminated = deepcopy(public_rows)
    contaminated[0]["optimum"] = {"total": 1}
    with pytest.raises(ValueError, match="reported labels"):
        exp.freeze_blinded_sample(contaminated)


def test_family_transitions_and_exhaustive_path_totals() -> None:
    """REQ-CONSTRAINT-6703; SCENARIO-CONSTRAINT-6703-COLD-RECOMPUTATION."""

    inventory = exp.exhaustive_solve(tiny_spec("inventory"))
    assert inventory["enumeration_count"] == 3
    assert inventory["optimum"] == 2
    assert inventory["optimum_plans"] == [[1]]

    battery = exp.exhaustive_solve(tiny_spec("battery_dispatch"))
    assert battery["optimum"] == 3
    assert battery["tie_set"] == [0, 1]
    illegal = exp.independent_transition(tiny_spec("battery_dispatch"), 0, 2, 2)
    assert illegal["legal"] is False
    assert illegal["reason"] == "grid_export_forbidden"

    jobs = exp.exhaustive_solve(tiny_spec("job_slot"))
    assert jobs["optimum"] == 2
    assert jobs["optimum_plans"] == [[1, 2]]
    repeated = exp.independent_transition(tiny_spec("job_slot"), 1, 1, 1)
    assert repeated["reason"] == "job_already_completed"
    missing_prerequisite = exp.independent_transition(tiny_spec("job_slot"), 0, 0, 2)
    assert missing_prerequisite["reason"] == "prerequisite_missing"

    reservoir = exp.exhaustive_solve(tiny_spec("reservoir_control"))
    assert reservoir["optimum"] == 0
    assert reservoir["optimum_plans"] == [[1]]
    assert (
        exp.independent_transition(tiny_spec("reservoir_control"), 0, 1, 2)["reason"]
        == "release_exceeds_available_water"
    )

    with pytest.raises(ValueError, match="unknown family"):
        exp.independent_transition({"family": "unknown", "parameters": {}}, 0, 0, 0)
    assert exp.independent_transition(tiny_spec("inventory"), 0, 0, 99)["reason"] == (
        "action_outside_domain"
    )
    with pytest.raises(ValueError, match="unknown family"):
        exp._terminal_cost({"family": "unknown", "parameters": {}}, 0)

    impossible = tiny_spec("battery_dispatch")
    impossible["action_domain"] = [1]
    impossible["parameters"]["actions"] = [1]
    impossible["initial_state"] = {"charge": 0}
    no_paths = exp.exhaustive_solve(impossible)
    assert no_paths["feasible"] is False
    assert no_paths["optimum"] is None
    assert no_paths["optimum_plans"] == []


def test_actual_selected_units_match_every_reported_exact_field() -> None:
    """REQ-VERIFY-6703; SCENARIO-VERIFY-6703-FIELD-PARITY."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    manifest = exp.freeze_blinded_sample(exp.public_instance_rows(upstream))
    solver_rows, comparison_rows = exp.recompute_selected_units(upstream, manifest)
    assert len(solver_rows) == 12
    assert all(row["receipt"].startswith("sha256:") for row in solver_rows)
    assert all(row["enumeration_count"] > 0 for row in solver_rows)
    assert comparison_rows
    assert all(row["disposition"] == "match" for row in comparison_rows)


def test_leakage_split_and_seal_scans_detect_shortcuts() -> None:
    """REQ-SAFE-6703; SCENARIO-SAFE-6703-LEAKAGE; SCENARIO-SAFE-6703-SEAL-TIMING."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    rows = exp.audit_leakage(upstream)
    assert rows
    assert all(row["pass_state"] for row in rows)

    leaked = deepcopy(upstream)
    leaked["instance_rows"][0]["prompt"] += " Exact optimum: 7."
    leaked_rows = exp.audit_leakage(leaked)
    assert any(
        row["check"] == "prompt_direct_label" and not row["pass_state"] for row in leaked_rows
    )

    duplicate = deepcopy(upstream)
    duplicate["instance_rows"][-1]["spec_hash"] = duplicate["instance_rows"][0]["spec_hash"]
    duplicate["instance_rows"][-1]["split"] = "development"
    duplicate_rows = exp.audit_leakage(duplicate)
    assert any(
        row["check"] == "development_headline_duplication" and not row["pass_state"]
        for row in duplicate_rows
    )

    stale = deepcopy(upstream)
    stale["label_seal_rows"][0]["prompt_hash"] = "sha256:stale"
    stale_rows = exp.audit_leakage(stale)
    assert any(row["check"] == "seal_integrity" and not row["pass_state"] for row in stale_rows)

    metadata = deepcopy(upstream)
    metadata["instance_rows"][0]["typed_spec"]["future_value"] = 7
    metadata_rows = exp.audit_leakage(metadata)
    assert any(
        row["check"] == "metadata_objective_encoding" and not row["pass_state"]
        for row in metadata_rows
    )


def test_metamorphic_and_mutation_cases_replay_from_raw_rows() -> None:
    """REQ-SAFE-6703; SCENARIO-SAFE-6703-MUTATIONS."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    rows = exp.audit_metamorphic_and_mutation_cases(upstream)
    assert {row["case"] for row in rows if row["kind"] == "metamorphic"} == {
        f"{family}:{transform}"
        for family in exp.FAMILIES
        for transform in exp.METAMORPHIC_TRANSFORMS
    }
    assert {row["case"] for row in rows if row["kind"] == "mutation"} == set(exp.REQUIRED_MUTATIONS)
    assert all(row["pass_state"] for row in rows)


def test_reducer_is_row_owned_and_fail_closed() -> None:
    """REQ-PIPELINE-6703; SCENARIO-PIPELINE-6703-ROW-REDUCTION."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    manifest = exp.freeze_blinded_sample(exp.public_instance_rows(upstream))
    solver_rows, comparisons = exp.recompute_selected_units(upstream, manifest)
    leakage = exp.audit_leakage(upstream)
    attacks = exp.audit_metamorphic_and_mutation_cases(upstream)
    coverage = exp.build_coverage_rows(
        upstream, manifest, solver_rows, comparisons, leakage, attacks
    )
    aggregate = exp.recompute_aggregate(
        coverage,
        solver_rows,
        comparisons,
        leakage,
        attacks,
        passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        blinding_clean=True,
    )
    assert aggregate["planning_fixture_audit_passed"] is True
    assert aggregate["failed_checks"] == []

    changed = deepcopy(comparisons)
    changed[0]["disposition"] = "mismatch"
    failed = exp.recompute_aggregate(
        coverage,
        solver_rows,
        changed,
        leakage,
        attacks,
        passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        blinding_clean=False,
    )
    assert failed["planning_fixture_audit_passed"] is False
    assert {"reported_parity", "blinding_chronology"} <= set(failed["failed_checks"])
    summary = exp._gate_summary({"failed_checks": ["coverage"]}, exp.BLINDING_PROTOCOL_INCIDENT)
    assert [row["check"] for row in summary] == ["blinding_chronology", "coverage"]


def test_missing_and_corrected_comparison_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CONSTRAINT-6703; SCENARIO-CONSTRAINT-6703-COVERAGE."""

    assert exp._comparison("unit", "field", exp._MISSING, 1)["disposition"] == "missing_reported"
    assert exp._comparison("unit", "field", 1, exp._MISSING)["disposition"] == "missing_recomputed"

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    manifest = exp.freeze_blinded_sample(exp.public_instance_rows(upstream))
    manifest["reveal_order"] = ["missing-instance"]
    solvers, comparisons = exp.recompute_selected_units(upstream, manifest)
    assert solvers == []
    assert comparisons[0]["disposition"] == "missing_reported"

    original = exp.recompute_selected_units

    def changed_rows(source: dict, frozen: dict) -> tuple[list[dict], list[dict]]:
        rows, compared = original(source, frozen)
        compared[0]["disposition"] = "mismatch"
        return rows, compared

    monkeypatch.setattr(exp, "recompute_selected_units", changed_rows)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )
    assert artifact["status"] == "corrected_fixture_mismatch_and_disqualified_blinding"
    assert artifact["honest_verdict"].startswith("corrected:")
    assert artifact["verdict_class"] == "partial"


def test_artifact_is_disqualified_by_recorded_cold_protocol_incident(tmp_path: Path) -> None:
    """REQ-REPORT-6703; SCENARIO-REPORT-6703-FAIL-CLOSED."""

    before = exp.protected_hashes(exp.REPO_ROOT)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=before,
    )
    assert artifact["status"] == "disqualified_blinding_chronology"
    assert artifact["honest_verdict"].startswith("disqualified:")
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["planning_fixture_audit_passed"] is False
    assert artifact["gate_check_summary"][0]["check"] == "blinding_chronology"
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert exp.validate_artifact(artifact) == []
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert len(artifact["per_unit_rows"]) == sum(
        len(artifact[field])
        for field in (
            "coverage_rows",
            "independent_solver_rows",
            "reported_vs_recomputed_rows",
            "leakage_rows",
            "metamorphic_mutation_rows",
        )
    )

    output = tmp_path / "nested" / "audit.json"
    receipt = exp.write_json_atomic(output, artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text()) == artifact

    changed = deepcopy(artifact)
    changed["planning_fixture_audit_passed"] = True
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "readiness_mismatch" in exp.validate_artifact(changed)
    assert exp.validate_artifact({}) == ["missing_required_fields"]

    def errors_for(field: str, value: object) -> list[str]:
        invalid = deepcopy(artifact)
        invalid[field] = value
        invalid["reproducibility_checksum"] = exp.artifact_checksum(invalid)
        return exp.validate_artifact(invalid)

    assert "inference_substrate_mismatch" in errors_for("inference_substrate", "wrong")
    assert "verifier_is_oracle_mismatch" in errors_for("verifier_is_oracle", True)
    assert "verdict_class_invalid" in errors_for("verdict_class", "wrong")
    assert "duration_invalid" in errors_for("duration_s", -1)
    assert "per_unit_rows_mismatch" in errors_for("per_unit_rows", [])
    assert "field_provenance_invalid" in errors_for("field_provenance", {})
    assert "manifest_hash_mismatch" in errors_for("blinded_sample_manifest", {})
    assert "aggregate_row_recomputation_mismatch" in errors_for("aggregate_row_recomputation", {})
    assert "failed_gate_summary_missing" in errors_for("gate_check_summary", [])


def test_blocked_upstream_and_command_receipts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-6703; SCENARIO-REPORT-6703-FAIL-CLOSED."""

    rows = exp.collect_preconditions(tmp_path)
    assert any(not row["passed"] for row in rows)
    blocked = exp.build_blocked_artifact("20260828", tmp_path, rows, 0.2)
    assert blocked["status"] == "blocked_precondition"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]
    assert exp.validate_artifact(blocked) == []
    broken_blocked = deepcopy(blocked)
    broken_blocked["gate_check_summary"] = []
    broken_blocked["reproducibility_checksum"] = exp.artifact_checksum(broken_blocked)
    assert "blocked_state_mismatch" in exp.validate_artifact(broken_blocked)
    blocked_output = tmp_path / "blocked-run.json"
    assert exp.run(date="20260828", root=tmp_path, output_path=blocked_output)["status"] == (
        "blocked_precondition"
    )

    receipt = exp.default_command_runner("printf ok", exp.REPO_ROOT)
    assert receipt["exit_code"] == 0
    assert receipt["stdout"] == "ok"

    def fake_runner(command: str, root: Path) -> dict[str, object]:
        assert root == exp.REPO_ROOT
        return {
            "command": command,
            "exit_code": 0,
            "stdout": "TOTAL 100 0 100%\n",
            "stderr": "",
            "duration_s": 0.1,
        }

    command_rows = exp.run_verification_commands(exp.REPO_ROOT, fake_runner)
    assert [row["check_id"] for row in command_rows] == list(exp.REQUIRED_TEST_CHECKS)
    assert all(row["passed"] for row in command_rows)
    assert (
        next(row for row in command_rows if row["check_id"] == "scoped_coverage")[
            "coverage_percent"
        ]
        == 100.0
    )

    monkeypatch.setattr(exp, "run_verification_commands", lambda root: passing_test_rows())
    output = tmp_path / "generated.json"
    assert exp.main(["--date", "20260828", "--output", str(output)]) == 0
    assert output.is_file()
    assert exp.main(["--validate", "--output", str(output)]) == 0
    output.write_text("{}")
    assert exp.main(["--validate", "--output", str(output)]) == 1
    output.write_text("[]")
    with pytest.raises(TypeError, match="JSON object"):
        exp.load_json(output)
    assert exp.main(["--validate", "--output", str(output)]) == 1
    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1

    monkeypatch.setattr(exp, "validate_artifact", lambda artifact: ["injected failure"])
    with pytest.raises(ValueError, match="injected failure"):
        exp.run(date="20260828", root=tmp_path, output_path=tmp_path / "invalid.json")
