"""Tests for the exact finite-horizon planning fixture.

Spec: REQ-CONSTRAINT-6702, REQ-REPORT-6702, and all related scenarios.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6702_exact_planning_fixture_recovery as exp


def passing_test_rows() -> list[dict[str, object]]:
    """Return the complete owned verification roster used by reducer tests."""

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
            }
        )
    return rows


def solved_fixture() -> tuple[list[dict], list[dict], list[dict], dict[str, dict]]:
    """Build the deterministic rows once for concise test assertions."""

    instances = exp.generate_instances()
    instance_rows: list[dict] = []
    action_rows: list[dict] = []
    solver_rows: list[dict] = []
    labels: dict[str, dict] = {}
    for instance in instances:
        solved = exp.solve_instance(instance)
        labels[instance["instance_id"]] = solved["label"]
        instance_rows.append(exp.build_instance_row(instance, solved))
        action_rows.extend(solved["state_action_rows"])
        solver_rows.append(solved["solver_row"])
    return instance_rows, action_rows, solver_rows, labels


def test_generation_and_exact_dynamic_programming() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-EXACT-ROWS."""

    first = exp.generate_instances()
    second = exp.generate_instances()
    assert first == second
    assert len(first) == 40
    assert {row["family"] for row in first} == set(exp.FAMILIES)
    assert sum(row["split"] == "headline" for row in first) == 32
    assert sum(row["split"] == "development" for row in first) == 8
    assert all(row["horizon"] <= 8 for row in first)
    assert all(len(row["action_set"]) <= 5 for row in first)
    assert len({row["instance_id"] for row in first}) == len(first)
    assert all("Exact optimum" not in row["prompt"] for row in first)

    for family in exp.FAMILIES:
        instance = next(row for row in first if row["family"] == family)
        solved = exp.solve_instance(instance)
        assert solved["label"]["feasible"] is True
        assert solved["label"]["optimum_plan"]
        assert solved["label"]["total_optimum"] >= 0
        assert solved["state_action_rows"]
        assert exp.validate_solution(instance, solved["label"], solved["state_action_rows"]) == []
        assert all(
            row["future_value"] is not None and row["total_value"] is not None
            for row in solved["state_action_rows"]
            if row["legality"]
        )
        assert all(
            row["future_value"] is None and row["total_value"] is None
            for row in solved["state_action_rows"]
            if not row["legality"]
        )


def test_independent_enumeration_matches_every_owned_value() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-EXACT-ROWS."""

    instances = exp.generate_instances()
    subset = exp.independent_subset(instances)
    assert len(subset) == len(exp.FAMILIES)
    for instance in subset:
        solved = exp.solve_instance(instance)
        replay = exp.independent_enumerate(instance, solved["state_action_rows"])
        assert replay["passed"] is True
        assert replay["optimum"] == solved["label"]["total_optimum"]
        assert replay["action_value_mismatches"] == []
        assert replay["enumeration_count"] > 0


def test_label_seals_require_a_prompt_bound_commit() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-SEALED-LABELS."""

    instance = exp.generate_instances()[0]
    solved = exp.solve_instance(instance)
    store = exp.LabelSealStore(
        {instance["instance_id"]: (instance["prompt_hash"], solved["label"])}
    )
    seal = store.seal_row(instance["instance_id"])
    assert exp.LabelSealStore.verify_seal_row(seal)
    with pytest.raises(exp.LabelAccessError, match="commit receipt required"):
        store.read(instance["instance_id"])
    with pytest.raises(exp.LabelAccessError, match="invalid commit receipt"):
        store.read(instance["instance_id"], {"receipt_hash": "sha256:stale"})

    receipt = store.commit(instance["instance_id"], instance["prompt_hash"], [0, 1])
    assert store.read(instance["instance_id"], receipt) == solved["label"]
    with pytest.raises(exp.LabelAccessError, match="prompt hash mismatch"):
        store.commit(instance["instance_id"], "sha256:wrong", [0])
    with pytest.raises(exp.LabelAccessError, match="unknown instance"):
        store.read("unknown")

    other = exp.generate_instances()[1]
    other_solved = exp.solve_instance(other)
    multi_store = exp.LabelSealStore(
        {
            instance["instance_id"]: (instance["prompt_hash"], solved["label"]),
            other["instance_id"]: (other["prompt_hash"], other_solved["label"]),
        }
    )
    other_receipt = multi_store.commit(other["instance_id"], other["prompt_hash"], [0])
    with pytest.raises(exp.LabelAccessError, match="invalid commit receipt"):
        multi_store.read(instance["instance_id"], other_receipt)


def test_transition_and_independent_replay_fail_closed_branches() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-EXACT-ROWS."""

    instances = exp.generate_instances()
    inventory = next(row for row in instances if row["family"] == "inventory")
    assert exp.execute_transition(inventory, 0, inventory["initial_state"], 99) == (
        False,
        None,
        None,
        "action_outside_domain",
    )
    assert exp._independent_transition(inventory, 0, inventory["initial_state"], 99) == (
        False,
        None,
        None,
    )

    battery = deepcopy(next(row for row in instances if row["family"] == "battery_dispatch"))
    battery["parameters"]["load"][0] = 0
    assert exp.execute_transition(battery, 0, 2, 1) == (
        False,
        None,
        None,
        "grid_export_forbidden",
    )

    replay_instance = next(
        row
        for row in instances
        if any(not action["legality"] for action in exp.solve_instance(row)["state_action_rows"])
    )
    clean_rows = exp.solve_instance(replay_instance)["state_action_rows"]
    legal_index = next(index for index, row in enumerate(clean_rows) if row["legality"])
    illegal_index = next(index for index, row in enumerate(clean_rows) if not row["legality"])

    wrong_legality = deepcopy(clean_rows)
    wrong_legality[legal_index]["legality"] = False
    assert exp.independent_enumerate(replay_instance, wrong_legality)["action_value_mismatches"]

    wrong_total = deepcopy(clean_rows)
    wrong_total[legal_index]["total_value"] += 1
    assert exp.independent_enumerate(replay_instance, wrong_total)["action_value_mismatches"]

    illegal_total = deepcopy(clean_rows)
    illegal_total[illegal_index]["total_value"] = 0
    assert exp.independent_enumerate(replay_instance, illegal_total)["action_value_mismatches"]


def test_metamorphic_invariants_and_mutations() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-ATTACKS."""

    instances = exp.generate_instances()
    instance_rows, action_rows, _, labels = solved_fixture()
    metamorphic = exp.build_metamorphic_rows(instances)
    mutations = exp.build_mutation_rows(instances, instance_rows, action_rows, labels)
    assert len(metamorphic) == len(exp.FAMILIES) * len(exp.METAMORPHIC_TRANSFORMS)
    assert {row["transform"] for row in metamorphic} == set(exp.METAMORPHIC_TRANSFORMS)
    assert all(row["pass_state"] for row in metamorphic)
    assert {row["mutation"] for row in mutations} == set(exp.REQUIRED_MUTATIONS)
    assert all(row["observed_detection"] and row["pass_state"] for row in mutations)


def test_row_reducer_is_fail_closed() -> None:
    """REQ-CONSTRAINT-6702; SCENARIO-CONSTRAINT-6702-ROW-REDUCTION."""

    assert exp.FULL_SUITE_COMMAND not in {command for _, command in exp.VERIFICATION_COMMANDS}
    instances = exp.generate_instances()
    instance_rows, action_rows, solver_rows, labels = solved_fixture()
    seal_rows = exp.build_label_seal_rows(instances, labels)
    metamorphic_rows = exp.build_metamorphic_rows(instances)
    mutation_rows = exp.build_mutation_rows(instances, instance_rows, action_rows, labels)
    independent_rows = exp.build_independent_solver_rows(instances)
    aggregate = exp.recompute_aggregate(
        instance_rows=instance_rows,
        state_action_rows=action_rows,
        exact_solver_rows=solver_rows + independent_rows,
        label_seal_rows=seal_rows,
        metamorphic_rows=metamorphic_rows,
        mutation_rows=mutation_rows,
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
    )
    assert aggregate["planning_fixture_ready"] is True
    assert aggregate["headline_instance_count"] == 32
    assert aggregate["development_instance_count"] == 8
    assert aggregate["state_action_row_count"] == len(action_rows)
    assert aggregate["failed_checks"] == []

    missing = exp.recompute_aggregate(
        instance_rows=instance_rows[:-1],
        state_action_rows=action_rows,
        exact_solver_rows=solver_rows + independent_rows,
        label_seal_rows=seal_rows,
        metamorphic_rows=metamorphic_rows,
        mutation_rows=mutation_rows,
        tests_run=passing_test_rows(),
        preconditions_passed=True,
        protected_files_unchanged=True,
    )
    assert missing["planning_fixture_ready"] is False
    assert "instance_coverage" in missing["failed_checks"]

    failed_tests = passing_test_rows()
    failed_tests[0]["passed"] = False
    failed = exp.recompute_aggregate(
        instance_rows=instance_rows,
        state_action_rows=action_rows,
        exact_solver_rows=solver_rows + independent_rows,
        label_seal_rows=seal_rows,
        metamorphic_rows=metamorphic_rows,
        mutation_rows=mutation_rows,
        tests_run=failed_tests,
        preconditions_passed=False,
        protected_files_unchanged=False,
    )
    assert {"focused_tests", "preconditions", "protected_files"} <= set(failed["failed_checks"])


def test_artifact_contract_validation_and_checksum(tmp_path: Path) -> None:
    """REQ-REPORT-6702; SCENARIO-REPORT-6702-ATOMIC-PROVENANCE."""

    protected = exp.protected_hashes(exp.REPO_ROOT)
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.25,
        protected_before=protected,
    )
    assert artifact["planning_fixture_ready"] is True
    assert artifact["status"] == "complete_ready"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gate_check_summary"] == []
    assert exp.validate_artifact(artifact) == []
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"])
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert len(artifact["per_unit_rows"]) == sum(
        len(artifact[key])
        for key in (
            "instance_rows",
            "state_action_rows",
            "exact_solver_rows",
            "label_seal_rows",
            "metamorphic_rows",
            "mutation_rows",
        )
    )

    output = tmp_path / "nested" / "artifact.json"
    receipt = exp.write_json_atomic(output, artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text()) == artifact
    assert not output.with_suffix(".json.tmp").exists()

    changed = deepcopy(artifact)
    changed["planning_fixture_ready"] = False
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "readiness_mismatch" in exp.validate_artifact(changed)


def test_blocked_precondition_artifact_is_terminal(tmp_path: Path) -> None:
    """REQ-REPORT-6702; SCENARIO-REPORT-6702-BLOCKED."""

    rows = exp.collect_preconditions(tmp_path)
    assert any(not row["passed"] for row in rows)
    artifact = exp.build_blocked_artifact(
        date="20260828", root=tmp_path, preconditions=rows, duration_s=0.5
    )
    assert artifact["status"] == "blocked_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["planning_fixture_ready"] is False
    assert artifact["gate_check_summary"]
    assert exp.validate_artifact(artifact) == []

    built = exp.build_artifact(
        date="20260828",
        root=tmp_path,
        tests_run=[],
        duration_s=0.25,
        protected_before=exp.protected_hashes(tmp_path),
    )
    assert built["status"] == "blocked_precondition"

    output = tmp_path / "blocked.json"
    run_artifact = exp.run(date="20260828", root=tmp_path, output_path=output)
    assert run_artifact["status"] == "blocked_precondition"
    assert output.is_file()


def test_artifact_validation_localizes_each_contract_failure() -> None:
    """REQ-REPORT-6702; SCENARIO-REPORT-6702-ATOMIC-PROVENANCE."""

    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )

    def errors_for(field: str, value: object) -> list[str]:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed)

    assert "artifact_schema_mismatch" in errors_for("status", 1)
    assert "inference_substrate_mismatch" in errors_for("inference_substrate", "wrong")
    assert "verifier_is_oracle_mismatch" in errors_for("verifier_is_oracle", True)
    assert "per_unit_rows_mismatch" in errors_for("per_unit_rows", [])
    assert "aggregate_row_recomputation_mismatch" in errors_for("aggregate_row_recomputation", {})

    manifest = deepcopy(artifact["frozen_fixture_manifest"])
    manifest["manifest_hash"] = "sha256:wrong"
    assert "manifest_hash_mismatch" in errors_for("frozen_fixture_manifest", manifest)
    assert "ready_terminal_state_mismatch" in errors_for("status", "complete_wrong")
    assert "honest_verdict_mismatch" in errors_for("honest_verdict", "wrong")
    assert "ready_gate_summary_mismatch" in errors_for("gate_check_summary", [{"check": "wrong"}])
    assert "field_provenance_invalid" in errors_for("field_provenance", {})
    assert "duration_invalid" in errors_for("duration_s", -1)

    assert exp.validate_artifact({}) == ["missing_required_fields"]


def test_e2e_atomic_artifact_run_with_injected_receipts(tmp_path: Path) -> None:
    """REQ-REPORT-6702; SCENARIO-REPORT-6702-ATOMIC-PROVENANCE."""

    output = tmp_path / "experiment_6702.json"
    artifact = exp.run(
        date="20260828",
        root=exp.REPO_ROOT,
        output_path=output,
        tests_run=passing_test_rows(),
    )
    assert output.is_file()
    assert exp.load_json(output) == artifact
    assert exp.validate_artifact(artifact) == []


def test_command_receipts_and_validate_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-REPORT-6702; SCENARIO-REPORT-6702-ATOMIC-PROVENANCE."""

    receipt = exp.default_command_runner("printf ok", exp.REPO_ROOT)
    assert receipt["exit_code"] == 0
    assert receipt["stdout"] == "ok"

    def fake_run(command: str, root: Path) -> dict[str, object]:
        assert root == exp.REPO_ROOT
        return {
            "command": command,
            "exit_code": 0,
            "stdout": "TOTAL 100 0 100%\n",
            "stderr": "",
            "duration_s": 0.1,
        }

    rows = exp.run_verification_commands(exp.REPO_ROOT, runner=fake_run)
    assert [row["check_id"] for row in rows] == list(exp.REQUIRED_TEST_CHECKS)
    assert all(row["passed"] for row in rows)
    assert (
        next(row for row in rows if row["check_id"] == "scoped_coverage")["coverage_percent"]
        == 100.0
    )

    output = tmp_path / "valid.json"
    artifact = exp.build_artifact(
        date="20260828",
        root=exp.REPO_ROOT,
        tests_run=passing_test_rows(),
        duration_s=1.0,
        protected_before=exp.protected_hashes(exp.REPO_ROOT),
    )
    exp.write_json_atomic(output, artifact)
    assert exp.main(["--validate", "--output", str(output)]) == 0
    output.write_text("{}")
    assert exp.main(["--validate", "--output", str(output)]) == 1
    output.write_text("[]")
    with pytest.raises(TypeError, match="JSON object"):
        exp.load_json(output)
    assert exp.main(["--validate", "--output", str(output)]) == 1
    assert exp.main(["--validate", "--output", str(tmp_path / "missing.json")]) == 1

    monkeypatch.setattr(exp, "run_verification_commands", lambda root: passing_test_rows())
    generated = tmp_path / "generated.json"
    assert exp.main(["--date", "20260828", "--output", str(generated)]) == 0
    assert generated.is_file()
