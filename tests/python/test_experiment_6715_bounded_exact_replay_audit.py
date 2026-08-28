"""Tests for the bounded independent exact-replay audit.

Spec: REQ-CONSTRAINT-6715, REQ-VERIFY-6715, REQ-PIPELINE-6715,
REQ-REPORT-6715, and their related scenarios.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6715_bounded_exact_replay_audit as exp


EXPECTED_SAMPLE_IDS = {
    "inventory-headline-01",
    "inventory-headline-06",
    "battery_dispatch-headline-01",
    "battery_dispatch-headline-05",
    "job_slot-headline-01",
    "job_slot-headline-02",
    "reservoir_control-headline-01",
    "reservoir_control-headline-03",
}


def passing_test_rows() -> list[dict[str, object]]:
    """Build complete successful check receipts for reducer tests."""

    return [
        {
            "check_id": check_id,
            "command": f"check {check_id}",
            "exit_code": 0,
            "passed": True,
            "coverage_percent": 100.0 if check_id == "scoped_coverage" else None,
            "summary": "passed",
            "duration_s": 0.1,
        }
        for check_id in exp.REQUIRED_TEST_CHECKS
    ]


def tiny_spec(family: str) -> dict[str, object]:
    """Return a hand-checkable typed specification for one family."""

    common: dict[str, object] = {
        "family": family,
        "instance_id": f"tiny-{family}",
        "schema": "carnot.typed_finite_horizon_plan.v1",
        "state_encoding": "bounded_nonnegative_integer",
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


def frozen_actual() -> tuple[dict[str, object], dict[str, object]]:
    """Load Exp6702 and freeze only its public projection."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    stores = exp.embedded_store_receipts(upstream, exp.REPO_ROOT / exp.UPSTREAM_PATH)
    manifest = exp.freeze_sample(exp.public_instance_rows(upstream), stores)
    return upstream, manifest


def test_frozen_sample_has_exact_strata_hashes_caps_and_order() -> None:
    """REQ-CONSTRAINT-6715; SCENARIO-CONSTRAINT-6715-FROZEN-SAMPLE."""

    upstream = exp.load_json(exp.REPO_ROOT / exp.UPSTREAM_PATH)
    public = exp.public_instance_rows(upstream)
    assert all(not exp.REPORTED_LABEL_FIELDS.intersection(row) for row in public)
    stores = exp.embedded_store_receipts(upstream, exp.REPO_ROOT / exp.UPSTREAM_PATH)
    manifest = exp.freeze_sample(public, stores)

    assert manifest["expected_instance_count"] == 8
    assert {row["instance"] for row in manifest["instances"]} == EXPECTED_SAMPLE_IDS
    assert len(manifest["reveal_order"]) == 8
    assert manifest["caps"] == exp.PREREGISTERED_CAPS
    assert manifest["manifest_hash"] == exp.manifest_checksum(manifest)
    assert manifest["frozen_before_reported_label_read"] is True
    assert set(manifest["typed_specs"]) == EXPECTED_SAMPLE_IDS
    for family in exp.FAMILIES:
        family_rows = [row for row in manifest["instances"] if row["family"] == family]
        assert len(family_rows) == 2
        assert {row["selection_role"] for row in family_rows} == {"edge_probe", "contrast"}
    for row in manifest["instances"]:
        assert exp.sha256_json(manifest["typed_specs"][row["instance"]]) == row["spec_hash"]

    duplicate = public + [deepcopy(public[0])]
    with pytest.raises(ValueError, match="duplicate public instance"):
        exp.freeze_sample(duplicate, stores)
    contaminated = deepcopy(public)
    contaminated[0]["ties"] = True
    with pytest.raises(ValueError, match="reported labels"):
        exp.freeze_sample(contaminated, stores)
    corrupt_spec = deepcopy(public)
    selected_public = next(
        row for row in corrupt_spec if row["instance"] == "inventory-headline-01"
    )
    selected_public["typed_spec"]["parameters"]["stage_cost_shift"] = 1
    with pytest.raises(ValueError, match="specification hash"):
        exp.freeze_sample(corrupt_spec, stores)
    no_anchor = [row for row in public if row["instance"] != "inventory-headline-01"]
    with pytest.raises(ValueError, match="edge-probe"):
        exp.freeze_sample(no_anchor, stores)
    no_contrast = [
        row
        for row in public
        if row["family"] != "inventory" or row["instance"] == "inventory-headline-01"
    ]
    with pytest.raises(ValueError, match="contrast candidates"):
        exp.freeze_sample(no_contrast, stores)


def test_family_transitions_and_complete_path_values() -> None:
    """REQ-CONSTRAINT-6715; SCENARIO-CONSTRAINT-6715-EXHAUSTIVE."""

    inventory = exp.exhaustive_solve(tiny_spec("inventory"), exp.PREREGISTERED_CAPS)
    assert inventory["enumeration_count"] == 3
    assert inventory["feasible_plan_count"] == 3
    assert inventory["optimum"] == 2
    assert inventory["optimum_plans"] == [[1]]

    battery = exp.exhaustive_solve(tiny_spec("battery_dispatch"), exp.PREREGISTERED_CAPS)
    assert battery["optimum"] == 3
    assert battery["tie_set"] == [0, 1]
    assert exp.independent_transition(tiny_spec("battery_dispatch"), 0, 2, 2)["reason"] == (
        "grid_export_forbidden"
    )

    jobs = exp.exhaustive_solve(tiny_spec("job_slot"), exp.PREREGISTERED_CAPS)
    assert jobs["optimum"] == 2
    assert jobs["optimum_plans"] == [[1, 2]]
    assert exp.independent_transition(tiny_spec("job_slot"), 1, 1, 1)["reason"] == (
        "job_already_completed"
    )
    assert exp.independent_transition(tiny_spec("job_slot"), 0, 0, 2)["reason"] == (
        "prerequisite_missing"
    )

    reservoir = exp.exhaustive_solve(tiny_spec("reservoir_control"), exp.PREREGISTERED_CAPS)
    assert reservoir["optimum"] == 0
    assert reservoir["optimum_plans"] == [[1]]
    assert exp.independent_transition(tiny_spec("reservoir_control"), 0, 1, 2)["reason"] == (
        "release_exceeds_available_water"
    )

    assert exp.independent_transition(tiny_spec("inventory"), 0, 0, 99)["reason"] == (
        "action_outside_domain"
    )
    with pytest.raises(ValueError, match="unknown family"):
        exp.independent_transition({"family": "unknown", "parameters": {}}, 0, 0, 0)
    with pytest.raises(ValueError, match="unknown family"):
        exp.terminal_cost({"family": "unknown", "parameters": {}}, 0)


def test_caps_abort_before_enumeration_or_sample_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CONSTRAINT-6715; SCENARIO-CONSTRAINT-6715-CAPS."""

    too_long = tiny_spec("inventory")
    too_long["horizon"] = 7
    with pytest.raises(exp.CapExceeded) as caught:
        exp.exhaustive_solve(too_long, exp.PREREGISTERED_CAPS)
    assert caught.value.row["cap"] == "max_horizon"
    assert caught.value.row["passed"] is False

    too_many_actions = tiny_spec("inventory")
    too_many_actions["action_domain"] = list(range(6))
    with pytest.raises(exp.CapExceeded, match="max_action_count"):
        exp.exhaustive_solve(too_many_actions, exp.PREREGISTERED_CAPS)

    low_enumeration_caps = {
        **exp.PREREGISTERED_CAPS,
        "max_enumeration_per_instance": 2,
    }
    with pytest.raises(exp.CapExceeded, match="max_enumeration_per_instance"):
        exp.exhaustive_solve(tiny_spec("inventory"), low_enumeration_caps)

    low_state_caps = {**exp.PREREGISTERED_CAPS, "max_state_count": 1}
    with pytest.raises(exp.CapExceeded, match="max_state_count"):
        exp.exhaustive_solve(tiny_spec("inventory"), low_state_caps)

    budget = exp.EnumerationBudget(maximum=2)
    with pytest.raises(exp.CapExceeded, match="max_total_enumeration_count"):
        exp.exhaustive_solve(tiny_spec("inventory"), exp.PREREGISTERED_CAPS, budget=budget)
    assert budget.used == 0

    expired_caps = {**exp.PREREGISTERED_CAPS, "max_audit_wall_time_s": -1}
    with pytest.raises(exp.CapExceeded, match="max_audit_wall_time_s"):
        exp.exhaustive_solve(tiny_spec("inventory"), expired_caps)

    times = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(exp.time, "perf_counter", lambda: next(times))
    wall_caps = {**exp.PREREGISTERED_CAPS, "max_audit_wall_time_s": 1}
    with pytest.raises(exp.CapExceeded, match="max_audit_wall_time_s"):
        exp.exhaustive_solve(tiny_spec("inventory"), wall_caps)


def test_infeasible_spec_and_missing_frozen_units_fail_explicitly() -> None:
    """REQ-CONSTRAINT-6715; SCENARIO-CONSTRAINT-6715-EXHAUSTIVE."""

    impossible = tiny_spec("battery_dispatch")
    impossible["action_domain"] = [1]
    impossible["parameters"]["actions"] = [1]
    impossible["initial_state"] = {"charge": 0}
    solved = exp.exhaustive_solve(impossible, exp.PREREGISTERED_CAPS)
    assert solved["feasible"] is False
    assert solved["optimum"] is None
    assert solved["optimum_plans"] == []

    upstream, manifest = frozen_actual()
    bad_manifest = deepcopy(manifest)
    bad_manifest["selection_rule"] = "changed"
    with pytest.raises(ValueError, match="manifest hash"):
        exp.recompute_frozen_sample(upstream, bad_manifest)
    bad_spec_manifest = deepcopy(manifest)
    bad_spec_manifest["typed_specs"][manifest["reveal_order"][0]]["horizon"] = 99
    bad_spec_manifest["manifest_hash"] = exp.manifest_checksum(bad_spec_manifest)
    with pytest.raises(ValueError, match="frozen typed specification"):
        exp.recompute_frozen_sample(upstream, bad_spec_manifest)

    missing = deepcopy(upstream)
    missing_id = manifest["reveal_order"][0]
    missing["instance_rows"] = [
        row for row in missing["instance_rows"] if row["instance"] != missing_id
    ]
    result = exp.recompute_frozen_sample(missing, manifest)
    row = next(
        row
        for row in result["reported_vs_recomputed_rows"]
        if row["unit"] == missing_id and row["field"] == "instance"
    )
    assert row["disposition"] == "missing_reported"


def test_e2e_bounded_actual_replay_matches_every_exact_field() -> None:
    """REQ-VERIFY-6715; SCENARIO-VERIFY-6715-EXACT-PARITY."""

    upstream, manifest = frozen_actual()
    result = exp.recompute_frozen_sample(upstream, manifest)

    assert len(result["enumeration_rows"]) == 8
    assert {row["instance"] for row in result["enumeration_rows"]} == EXPECTED_SAMPLE_IDS
    assert sum(row["enumeration_count"] for row in result["enumeration_rows"]) == 42_530
    assert all(row["receipt"].startswith("sha256:") for row in result["enumeration_rows"])
    assert all(
        row["cap_state"] == "within_preregistered_caps" for row in result["enumeration_rows"]
    )
    assert result["reported_vs_recomputed_rows"]
    assert all(row["disposition"] == "match" for row in result["reported_vs_recomputed_rows"])
    assert len(result["edge_rows"]) == 4
    assert all(row["passed"] for row in result["edge_rows"])
    assert result["total_enumeration_count"] == 42_530


def test_solver_uses_frozen_specs_after_reported_labels_open() -> None:
    """REQ-VERIFY-6715; SCENARIO-CONSTRAINT-6715-FROZEN-SAMPLE."""

    upstream, manifest = frozen_actual()
    changed_report = deepcopy(upstream)
    selected = set(manifest["reveal_order"])
    for row in changed_report["instance_rows"]:
        if row["instance"] in selected:
            row["typed_spec"]["parameters"]["stage_cost_shift"] = 10_000

    result = exp.recompute_frozen_sample(changed_report, manifest)

    assert len(result["enumeration_rows"]) == 8
    assert all(row["disposition"] == "match" for row in result["reported_vs_recomputed_rows"])


def test_missing_and_mismatched_values_remain_explicit_rows() -> None:
    """REQ-VERIFY-6715; SCENARIO-VERIFY-6715-EXACT-PARITY."""

    missing_reported = exp.comparison_row("unit", "field", exp.MISSING, None)
    assert missing_reported["reported_value"] == {"state": "missing"}
    assert missing_reported["recomputed_value"] is None
    assert missing_reported["disposition"] == "missing_reported"

    missing_recomputed = exp.comparison_row("unit", "field", 0, exp.MISSING)
    assert missing_recomputed["reported_value"] == 0
    assert missing_recomputed["recomputed_value"] == {"state": "missing"}
    assert missing_recomputed["disposition"] == "missing_recomputed"

    mismatch = exp.comparison_row("unit", "field", 0, 1)
    assert mismatch["disposition"] == "mismatch"
    assert exp.comparison_row("unit", "field", None, None)["disposition"] == "match"

    nested = {"outer": {"value": None}}
    assert exp.nested_value(nested, "outer", "value") is None
    assert exp.nested_value(nested, "outer", "absent") is exp.MISSING


def test_row_reducer_is_complete_and_fail_closed() -> None:
    """REQ-PIPELINE-6715; SCENARIO-PIPELINE-6715-ROW-REDUCTION; SCENARIO-PIPELINE-6715-PER-UNIT."""

    upstream, manifest = frozen_actual()
    result = exp.recompute_frozen_sample(upstream, manifest)
    method = exp.method_fidelity_contract(exp.REPO_ROOT)
    aggregate = exp.recompute_aggregate(
        manifest=manifest,
        enumeration_rows=result["enumeration_rows"],
        state_action_rows=result["state_action_rows"],
        comparison_rows=result["reported_vs_recomputed_rows"],
        edge_rows=result["edge_rows"],
        cap_rows=result["cap_rows"],
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        method_contract=method,
    )
    assert aggregate["exact_replay_audit_passed"] is True
    assert aggregate["failed_checks"] == []

    rows = exp.build_per_unit_rows(
        result["enumeration_rows"],
        result["state_action_rows"],
        result["reported_vs_recomputed_rows"],
        result["edge_rows"],
        result["cap_rows"],
        aggregate["check_rows"],
    )
    assert {row["unit_type"] for row in rows} == {
        "enumeration",
        "state_action",
        "comparison",
        "edge_check",
        "cap_check",
        "audit_check",
    }

    changed = deepcopy(result["reported_vs_recomputed_rows"])
    changed[0]["disposition"] = "mismatch"
    failed = exp.recompute_aggregate(
        manifest=manifest,
        enumeration_rows=result["enumeration_rows"],
        state_action_rows=result["state_action_rows"],
        comparison_rows=changed,
        edge_rows=result["edge_rows"],
        cap_rows=result["cap_rows"],
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
        method_contract=method,
    )
    assert failed["exact_replay_audit_passed"] is False
    assert "exact_comparisons" in failed["failed_checks"]


def test_complete_and_blocked_artifacts_validate_with_provenance() -> None:
    """REQ-REPORT-6715; SCENARIO-REPORT-6715-ATOMIC; SCENARIO-REPORT-6715-BLOCKED."""

    before = exp.protected_hashes(exp.REPO_ROOT)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.25,
        protected_before=before,
    )
    assert artifact["status"] == "complete_reproduced"
    assert artifact["honest_verdict"].startswith("complete: reproduced")
    assert artifact["verdict_class"] == "positive"
    assert artifact["exact_replay_audit_passed"] is True
    assert artifact["gate_check_summary"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    for row in artifact["field_provenance"].values():
        assert set(exp.PROVENANCE_KEYS) <= set(row)

    changed = deepcopy(artifact)
    changed["status"] = "changed"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    missing = deepcopy(artifact)
    del missing["status"]
    assert exp.validate_artifact(missing) == ["missing_required_fields"]

    preconditions = exp.collect_preconditions(exp.REPO_ROOT)
    failed_preconditions = deepcopy(preconditions)
    failed_preconditions[0]["passed"] = False
    failed_preconditions[0]["observed"] = "missing"
    blocked = exp.build_blocked_artifact(
        "20260828", exp.REPO_ROOT, failed_preconditions, duration_s=0.1
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["exact_replay_audit_passed"] is False
    assert blocked["gate_check_summary"][0]["observed"] == "missing"
    assert exp.validate_artifact(blocked) == []


def test_artifact_validator_localizes_each_contract_failure() -> None:
    """REQ-REPORT-6715; SCENARIO-REPORT-6715-ATOMIC."""

    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )

    def resigned(**changes: object) -> dict[str, object]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return changed

    assert "inference_substrate_mismatch" in exp.validate_artifact(
        resigned(inference_substrate="wrong")
    )
    assert "verifier_is_oracle_mismatch" in exp.validate_artifact(resigned(verifier_is_oracle=True))
    assert "verdict_class_invalid" in exp.validate_artifact(resigned(verdict_class="wrong"))
    assert "duration_invalid" in exp.validate_artifact(resigned(duration_s=-1))
    assert "field_provenance_invalid" in exp.validate_artifact(resigned(field_provenance={}))

    units = deepcopy(artifact["per_unit_rows"])
    units.pop()
    assert "per_unit_rows_mismatch" in exp.validate_artifact(resigned(per_unit_rows=units))

    bad_manifest = deepcopy(artifact["frozen_sample_manifest"])
    bad_manifest["manifest_hash"] = "sha256:wrong"
    assert "manifest_hash_mismatch" in exp.validate_artifact(
        resigned(frozen_sample_manifest=bad_manifest)
    )

    bad_aggregate = deepcopy(artifact["aggregate_row_recomputation"])
    bad_aggregate["comparison_row_count"] += 1
    assert "aggregate_row_recomputation_mismatch" in exp.validate_artifact(
        resigned(aggregate_row_recomputation=bad_aggregate)
    )

    assert "passed_gate_summary_mismatch" in exp.validate_artifact(
        resigned(gate_check_summary=[{"passed": False}])
    )
    false_gate = resigned(exact_replay_audit_passed=False, gate_check_summary=[])
    false_errors = exp.validate_artifact(false_gate)
    assert "audit_gate_mismatch" in false_errors
    assert "failed_gate_summary_missing" in false_errors

    blocked_preconditions = deepcopy(artifact["preconditions_checked"])
    blocked_preconditions[0]["passed"] = False
    blocked = exp.build_blocked_artifact(
        "20260828", exp.REPO_ROOT, blocked_preconditions, duration_s=0.1
    )
    blocked["exact_replay_audit_passed"] = True
    blocked["gate_check_summary"] = []
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert "blocked_state_mismatch" in exp.validate_artifact(blocked)


def test_classification_and_cap_failure_are_not_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PIPELINE-6715; SCENARIO-PIPELINE-6715-ROW-REDUCTION."""

    method = {"satisfied": True}
    assert (
        exp._classification(
            {"exact_replay_audit_passed": False, "failed_checks": ["exact_comparisons"]},
            method,
        )[0]
        == "complete_corrected"
    )
    assert (
        exp._classification(
            {"exact_replay_audit_passed": False, "failed_checks": []},
            {"satisfied": False},
        )[0]
        == "disqualified_method_or_cap"
    )

    def capped(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise exp.CapExceeded(exp.cap_row("max_state_count", 128, 129, False, "capped-instance"))

    monkeypatch.setattr(exp, "recompute_frozen_sample", capped)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )
    assert artifact["status"] == "disqualified_method_or_cap"
    assert artifact["exact_replay_audit_passed"] is False
    assert artifact["cap_rows"][0]["observed"] == 129
    assert exp.validate_artifact(artifact) == []


def test_atomic_writer_command_receipts_and_run_orchestration(tmp_path: Path) -> None:
    """REQ-REPORT-6715; SCENARIO-REPORT-6715-ATOMIC."""

    target = tmp_path / "artifact.json"
    receipt = exp.write_json_atomic(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert receipt["atomic_replace"] is True
    assert not target.with_suffix(".json.tmp").exists()

    command = exp.default_command_runner(
        ".venv/bin/python -c 'print(\"receipt-ok\")'", exp.REPO_ROOT
    )
    assert command["exit_code"] == 0
    assert "receipt-ok" in command["stdout"]

    def fake_runner(command_text: str, root: Path) -> dict[str, object]:
        assert root == exp.REPO_ROOT
        if "coverage report" in command_text:
            stdout = "TOTAL 400 0 100%\n"
        elif "adversarial_verify.py" in command_text:
            stdout = json.dumps({"reports": [{"max_severity": 0}]})
        else:
            stdout = "passed"
        return {
            "command": command_text,
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "duration_s": 0.01,
        }

    checks = exp.run_verification_commands(exp.REPO_ROOT, runner=fake_runner)
    assert all(row["passed"] for row in checks)
    candidate = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )
    exp.write_json_atomic(target, candidate)
    operational = exp.run_artifact_checks(exp.REPO_ROOT, target, runner=fake_runner)
    assert all(row["passed"] for row in operational)
    assert all(
        row["critical_free"] is True
        for row in operational
        if row["check_id"] == "adversarial_verification"
    )

    def invalid_json_runner(command_text: str, root: Path) -> dict[str, object]:
        receipt = fake_runner(command_text, root)
        if "adversarial_verify.py" in command_text:
            receipt["stdout"] = "not-json"
        return receipt

    fallback = exp.run_artifact_checks(exp.REPO_ROOT, target, runner=invalid_json_runner)
    adversarial = next(row for row in fallback if row["check_id"] == "adversarial_verification")
    assert adversarial["critical_free"] is True

    output = tmp_path / "run.json"
    run_artifact = exp.run(
        date="20260828",
        root=exp.REPO_ROOT,
        output_path=output,
        runner=fake_runner,
    )
    assert output.is_file()
    assert run_artifact["exact_replay_audit_passed"] is True
    assert exp.validate_artifact(run_artifact) == []

    blocked_output = tmp_path / "blocked.json"
    blocked_run = exp.run(
        date="20260828",
        root=tmp_path,
        output_path=blocked_output,
        runner=fake_runner,
    )
    assert blocked_run["status"] == "blocked_precondition"
    assert blocked_output.is_file()


def test_run_raises_if_candidate_or_final_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6715; SCENARIO-REPORT-6715-ATOMIC."""

    def fake_runner(command_text: str, root: Path) -> dict[str, object]:
        del root
        stdout = "TOTAL 1 0 100%" if "coverage report" in command_text else "passed"
        if "adversarial_verify.py" in command_text:
            stdout = json.dumps({"reports": []})
        return {
            "command": command_text,
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
            "duration_s": 0.01,
        }

    original_validate = exp.validate_artifact
    monkeypatch.setattr(exp, "validate_artifact", lambda payload: ["candidate-invalid"])
    with pytest.raises(ValueError, match="candidate"):
        exp.run(
            date="20260828",
            root=exp.REPO_ROOT,
            output_path=tmp_path / "candidate.json",
            runner=fake_runner,
        )

    calls = iter([[], ["final-invalid"]])
    monkeypatch.setattr(exp, "validate_artifact", lambda payload: next(calls))
    with pytest.raises(ValueError, match="final-invalid"):
        exp.run(
            date="20260828",
            root=exp.REPO_ROOT,
            output_path=tmp_path / "final.json",
            runner=fake_runner,
        )
    monkeypatch.setattr(exp, "validate_artifact", original_validate)


def test_cli_validation_and_precondition_localization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6715; SCENARIO-REPORT-6715-BLOCKED."""

    checks = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert {
        "upstream_artifact",
        "planning_fixture_ready",
        "raw_stores_and_seals",
        "cpu",
        "ram_bytes",
        "disk_free_bytes",
        "artifact_schema",
        "audit_tools",
        "roadmap",
        "conductor",
    } <= {row["name"] for row in checks}

    missing_checks = exp.collect_preconditions(tmp_path)
    assert any(not row["passed"] for row in missing_checks)

    malformed_root = tmp_path / "malformed-root"
    (malformed_root / "results").mkdir(parents=True)
    (malformed_root / exp.UPSTREAM_PATH).write_text("[]", encoding="utf-8")
    malformed_checks = exp.collect_preconditions(malformed_root)
    upstream_check = next(row for row in malformed_checks if row["name"] == "upstream_artifact")
    assert upstream_check["passed"] is False
    assert "JSON object required" in upstream_check["observed"]["error"]

    assert exp._memory_bytes(tmp_path / "missing-meminfo") == 0
    empty_meminfo = tmp_path / "meminfo"
    empty_meminfo.write_text("no total here", encoding="utf-8")
    assert exp._memory_bytes(empty_meminfo) == 0

    def missing_package(name: str) -> str:
        assert name == "jsonschema"
        raise exp.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(exp.metadata, "version", missing_package)
    no_schema = exp.collect_preconditions(exp.REPO_ROOT)
    schema_row = next(row for row in no_schema if row["name"] == "artifact_schema")
    assert schema_row["passed"] is False

    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )
    output = tmp_path / "valid.json"
    exp.write_json_atomic(output, artifact)
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert exp.main(["--validate", "--output", str(tmp_path / "absent.json")]) == 1
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(malformed)]) == 1

    called: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        called.update(kwargs)
        return {}

    monkeypatch.setattr(exp, "run", fake_run)
    cli_output = tmp_path / "cli.json"
    assert exp.main(["--date", "20260828", "--output", str(cli_output)]) == 0
    assert called["date"] == "20260828"
    assert called["output_path"] == cli_output
