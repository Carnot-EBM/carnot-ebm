"""Receipt-scoped admission tests for Exp6647.

Spec refs: REQ-INFRA-6647, SCENARIO-INFRA-6647-PREREGISTERED-OWNERSHIP,
SCENARIO-INFRA-6647-EXACT-FIELD-OWNERSHIP,
SCENARIO-INFRA-6647-MISSING-RECEIPT, REQ-REPORT-6647,
SCENARIO-REPORT-6647-READY, SCENARIO-REPORT-6647-GLOBAL-DIAGNOSTIC,
SCENARIO-REPORT-6647-BLOCKED-RECEIPT, and
SCENARIO-REPORT-6647-ATOMIC-PROVENANCE.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6647_receipt_scoped_admission_boundary as exp
from carnot.terminal_artifacts import classify_artifact_payload


REPO = Path(__file__).resolve().parents[2]


def _passing_rows() -> list[dict]:
    return [
        exp.make_check_row(
            definition,
            observed_value=True,
            exit_code=0,
            receipt={"check_id": definition["check_id"], "passed": True},
        )
        for definition in exp.PREREGISTERED_TASK_OWNED_CHECKS
    ]


def _ready_artifact(tmp_path: Path) -> dict:
    protected = exp.protected_hashes(REPO)
    return exp.build_artifact(
        date="20260826",
        root=REPO,
        duration_s=1.25,
        check_rows=_passing_rows(),
        global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
        protected_before=protected,
        preconditions=exp.collect_preconditions(REPO, tmp_path, protected),
        tests_run=exp.DEFAULT_TESTS_RUN,
    )


def test_req_6647_specs_and_frozen_owned_inventory() -> None:
    """REQ-INFRA-6647 and REQ-REPORT-6647 freeze the exact owned gate set."""

    infra = exp.INFRA_SPEC_PATH.read_text(encoding="utf-8")
    report = exp.SPEC_PATH.read_text(encoding="utf-8")
    for anchor in (
        "REQ-INFRA-6647",
        "SCENARIO-INFRA-6647-PREREGISTERED-OWNERSHIP",
        "SCENARIO-INFRA-6647-EXACT-FIELD-OWNERSHIP",
        "SCENARIO-INFRA-6647-MISSING-RECEIPT",
    ):
        assert anchor in infra
    for anchor in (
        "REQ-REPORT-6647",
        "SCENARIO-REPORT-6647-READY",
        "SCENARIO-REPORT-6647-GLOBAL-DIAGNOSTIC",
        "SCENARIO-REPORT-6647-BLOCKED-RECEIPT",
        "SCENARIO-REPORT-6647-ATOMIC-PROVENANCE",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert anchor in report
    assert tuple(row["check_id"] for row in exp.PREREGISTERED_TASK_OWNED_CHECKS) == (
        "acquisition",
        "same_device_exclusion",
        "independent_device_allowance",
        "token_pid_start_device_binding",
        "heartbeat",
        "phase_transitions",
        "unload_release",
        "crash_recovery",
        "tamper_detection",
        "atomic_artifact_write",
        "focused_tests",
        "spec_coverage",
        "applicable_e2e_checks",
    )
    assert len({row["check_id"] for row in exp.PREREGISTERED_TASK_OWNED_CHECKS}) == 13
    for definition in exp.PREREGISTERED_TASK_OWNED_CHECKS:
        assert set(definition) == {
            "ordinal",
            "check_id",
            "expected_value",
            "source",
            "command",
            "receipt_schema",
        }


def test_scenario_report_6647_ready_ignores_global_suite_exit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6647-READY keeps the global failure diagnostic-only."""

    artifact = _ready_artifact(tmp_path)
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete_ready"
    assert artifact["verdict_class"] is None
    assert artifact["task_owned_admission_ready_score"] == 1.0
    assert artifact["gate_check_summary"] == []
    assert artifact["global_suite_diagnostic"]["exit_code"] == 3
    assert artifact["global_suite_diagnostic"]["gating"] is False
    assert "xdist" in artifact["global_suite_diagnostic"]["summary"]
    assert artifact["aggregate_row_recomputation"]["excluded_diagnostic_count"] == 1
    assert artifact["per_unit_rows"][-1]["row_kind"] == "global_suite_diagnostic"
    assert artifact["prior_failure_receipt"]["exact_failed_reduction_field"] == "focused_tests"
    assert artifact["prior_failure_receipt"]["artifact_sha256"].startswith("sha256:")
    assert set(artifact["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == exp.payload_checksum(artifact)

    classification = classify_artifact_payload(artifact, path=exp.RESULT_PATH)
    assert classification.terminal is True
    assert classification.classification == "ready"

    adversarial = next(
        row for row in artifact["tests_run"] if row["command"] == exp.ADVERSARIAL_COMMAND
    )
    assert adversarial["exit_code"] == 1
    assert "one non-critical substrate review warning" in adversarial["summary"]


def test_scenario_infra_6647_missing_null_duplicate_and_reordered_fail_closed(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6647-MISSING-RECEIPT never invents an observed zero."""

    protected = exp.protected_hashes(REPO)

    def build(rows: list[dict]) -> dict:
        return exp.build_artifact(
            date="20260826",
            root=REPO,
            duration_s=1.0,
            check_rows=rows,
            global_suite_diagnostic=exp.load_global_suite_diagnostic(REPO),
            protected_before=protected,
            preconditions=exp.collect_preconditions(REPO, tmp_path, protected),
            tests_run=exp.DEFAULT_TESTS_RUN,
        )

    missing = build(_passing_rows()[:-1])
    assert missing["task_owned_admission_ready_score"] == 0.0
    assert missing["status"].startswith("blocked_")
    assert missing["honest_verdict"].startswith("blocked_")
    assert missing["gate_check_summary"] == [
        {
            "check": "applicable_e2e_checks",
            "expected_value": True,
            "observed_value": None,
            "reason": "missing_receipt",
        }
    ]

    null_rows = _passing_rows()
    null_rows[0] = exp.make_check_row(
        exp.PREREGISTERED_TASK_OWNED_CHECKS[0],
        observed_value=None,
        exit_code=None,
        receipt={"observed": None},
    )
    null = build(null_rows)
    assert null["gate_check_summary"][0]["observed_value"] is None
    assert null["gate_check_summary"][0]["reason"] == "null_observed_value"

    duplicate = build(_passing_rows() + [_passing_rows()[0]])
    assert duplicate["gate_check_summary"][0]["reason"] == "duplicate_receipt"

    reordered_rows = _passing_rows()
    reordered_rows[0], reordered_rows[1] = reordered_rows[1], reordered_rows[0]
    reordered = build(reordered_rows)
    assert reordered["gate_check_summary"][0]["reason"] == "receipt_order_mismatch"

    changed_definition_rows = _passing_rows()
    changed_definition_rows[0]["source"] = "wrong-source"
    changed_definition_rows[0]["receipt_hash"] = exp.receipt_hash(
        changed_definition_rows[0], excluded=("receipt_hash",)
    )
    changed_definition = build(changed_definition_rows)
    assert changed_definition["gate_check_summary"][0]["reason"] == "definition_mismatch"


def test_scenario_infra_6647_replays_all_owned_fixtures_in_fresh_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6647-PREREGISTERED-OWNERSHIP replays every owned check."""

    rows = exp.replay_task_owned_checks(tmp_path, exp.DEFAULT_TESTS_RUN)
    assert [row["check_id"] for row in rows] == [
        row["check_id"] for row in exp.PREREGISTERED_TASK_OWNED_CHECKS
    ]
    assert all(row["observed_value"] is True for row in rows)
    assert all(row["exit_code"] == 0 for row in rows)
    assert all(row["receipt_hash"].startswith("sha256:") for row in rows)
    assert len({row["fixture_path"] for row in rows[:10]}) == 10
    assert all(Path(row["fixture_path"]).is_relative_to(tmp_path) for row in rows[:10])
    assert all(Path(row["fixture_path"]).is_dir() for row in rows[:10])
    assert set(rows[3]["receipt"]["attack_ids"]) == {
        "wrong_token",
        "wrong_device",
        "pid_reuse",
    }


def test_scenario_report_6647_row_hash_and_exact_ownership_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6647-EXACT-FIELD-OWNERSHIP rejects changed owned rows."""

    ready = _ready_artifact(tmp_path)
    bad = deepcopy(ready)
    bad["task_owned_check_rows"][0]["observed_value"] = False
    bad["reproducibility_checksum"] = exp.payload_checksum(bad)
    errors = exp.validate_artifact(bad)
    assert "row_receipt_hash_mismatch:acquisition" in errors
    assert "aggregate_row_recomputation_mismatch" in errors

    diagnostic_changed = deepcopy(ready)
    diagnostic_changed["global_suite_diagnostic"]["exit_code"] = 99
    diagnostic_changed["global_suite_diagnostic"]["receipt_hash"] = exp.receipt_hash(
        diagnostic_changed["global_suite_diagnostic"], excluded=("receipt_hash",)
    )
    diagnostic_changed["per_unit_rows"][-1] = {
        "row_kind": "global_suite_diagnostic",
        **diagnostic_changed["global_suite_diagnostic"],
    }
    diagnostic_changed["field_provenance"] = exp.build_field_provenance(diagnostic_changed)
    diagnostic_changed["reproducibility_checksum"] = exp.payload_checksum(diagnostic_changed)
    assert exp.validate_artifact(diagnostic_changed) == []
    assert diagnostic_changed["task_owned_admission_ready_score"] == 1.0


def test_scenario_report_6647_validator_rejects_schema_provenance_and_checksum(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6647-ATOMIC-PROVENANCE validates all durable boundaries."""

    ready = _ready_artifact(tmp_path)

    def errors_for(**changes: object) -> list[str]:
        bad = deepcopy(ready)
        bad.update(changes)
        bad["reproducibility_checksum"] = exp.payload_checksum(bad)
        return exp.validate_artifact(bad)

    bad = deepcopy(ready)
    bad.pop("duration_s")
    bad["reproducibility_checksum"] = exp.payload_checksum(bad)
    assert "required_fields_mismatch" in exp.validate_artifact(bad)
    assert "inference_substrate_mismatch" in errors_for(inference_substrate="wrong")
    assert "verifier_is_oracle_mismatch" in errors_for(verifier_is_oracle=False)
    assert "ready_status_mismatch" in errors_for(status="blocked_wrong")
    assert "ready_verdict_class_mismatch" in errors_for(verdict_class="blocked")
    assert "ready_gate_summary_mismatch" in errors_for(gate_check_summary=[{"bad": True}])
    assert "preregistered_checks_mismatch" in errors_for(preregistered_task_owned_checks=[])
    assert "per_unit_rows_mismatch" in errors_for(per_unit_rows=[])
    assert "global_suite_diagnostic_missing" in errors_for(global_suite_diagnostic=None)
    assert "global_suite_diagnostic_gating" in errors_for(
        global_suite_diagnostic={
            **ready["global_suite_diagnostic"],
            "gating": True,
        }
    )
    assert "global_suite_diagnostic_hash_mismatch" in errors_for(
        global_suite_diagnostic={
            **ready["global_suite_diagnostic"],
            "receipt_hash": "sha256:bad",
        }
    )
    assert "reducer_contract_mismatch" in errors_for(reducer_contract={})
    assert "random_seed_mismatch" in errors_for(random_seed=1)
    assert "protected_files_changed" in errors_for(
        protected_files_unchanged={"all_unchanged": False}
    )

    invalid_provenance = deepcopy(ready["field_provenance"])
    invalid_provenance["status"] = {}
    assert "field_provenance_invalid:status" in errors_for(field_provenance=invalid_provenance)
    assert "field_provenance_mismatch" in errors_for(field_provenance={})

    bad = deepcopy(ready)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(bad)

    blocked = deepcopy(ready)
    blocked["task_owned_check_rows"][0] = exp.make_check_row(
        exp.PREREGISTERED_TASK_OWNED_CHECKS[0],
        observed_value=False,
        exit_code=1,
        receipt={"passed": False},
    )
    blocked = exp.finalize_reduction(blocked)
    assert exp.validate_artifact(blocked) == []
    assert "blocked_status_mismatch" in _mutated_errors(blocked, exp, status="wrong")
    assert "blocked_verdict_mismatch" in _mutated_errors(blocked, exp, honest_verdict="wrong")
    assert "blocked_verdict_class_mismatch" in _mutated_errors(blocked, exp, verdict_class=None)
    assert "blocked_gate_summary_mismatch" in _mutated_errors(blocked, exp, gate_check_summary=[])


def _mutated_errors(artifact: dict, module: object, **changes: object) -> list[str]:
    bad = deepcopy(artifact)
    bad.update(changes)
    bad["reproducibility_checksum"] = module.payload_checksum(bad)  # type: ignore[attr-defined]
    return module.validate_artifact(bad)  # type: ignore[attr-defined,no-any-return]


def test_req_report_6647_preconditions_and_prior_global_receipts(tmp_path: Path) -> None:
    """REQ-REPORT-6647 records inputs, host resources, prior truth, and no LLM."""

    protected = exp.protected_hashes(REPO)
    preconditions = exp.collect_preconditions(REPO, tmp_path, protected)
    assert preconditions["inputs"]["prior_artifact_sha256"].startswith("sha256:")
    assert preconditions["inputs"]["research_roadmap_sha256"] == protected["research-roadmap.yaml"]
    assert (
        preconditions["inputs"]["research_conductor_sha256"]
        == protected["scripts/research_conductor.py"]
    )
    assert preconditions["task_owned_fixture_inventory"] == [
        row["check_id"] for row in exp.PREREGISTERED_TASK_OWNED_CHECKS
    ]
    assert preconditions["python"]["executable"]
    assert preconditions["host_resources"]["cpu_count"]
    assert preconditions["host_resources"]["ram_bytes"] > 0
    assert preconditions["host_resources"]["disk_free_bytes"] > 0
    assert preconditions["no_llm"] == {
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
        "model_load_attempt_count": 0,
        "generation_attempt_count": 0,
        "llm_import_required": False,
    }
    diagnostic = exp.load_global_suite_diagnostic(REPO)
    assert diagnostic["command"] == exp.FULL_TEST_COMMAND
    assert diagnostic["known_issue_link"] == "ops/known-issues.md:2620"
    assert diagnostic["gating"] is False
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._read_prior(tmp_path) == {}
    missing_diagnostic = exp.load_global_suite_diagnostic(tmp_path)
    assert missing_diagnostic["exit_code"] is None
    assert missing_diagnostic["summary"] == "receipt missing"


def test_scenario_infra_6647_fixture_exception_becomes_owned_red_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6647-MISSING-RECEIPT keeps fixture errors explicit."""

    def fail(_path: Path) -> tuple[bool, dict]:
        raise RuntimeError("owned fixture failed")

    def pass_fixture(path: Path) -> tuple[bool, dict]:
        return True, {"path": str(path)}

    monkeypatch.setattr(exp, "FIXTURE_FUNCTIONS", (fail,) + (pass_fixture,) * 9)
    rows = exp.replay_task_owned_checks(tmp_path, exp.DEFAULT_TESTS_RUN)
    assert rows[0]["observed_value"] is False
    assert rows[0]["exit_code"] == 1
    assert rows[0]["receipt"] == {"error": "RuntimeError: owned fixture failed"}


def test_req_report_6647_run_cli_validate_and_refuse_invalid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-6647 runs end to end and validates the atomic artifact."""

    output = tmp_path / "exp6647.json"
    artifact = exp.run(
        date="20260826",
        root=REPO,
        result_path=output,
        work_dir=tmp_path / "work",
        tests_run=exp.DEFAULT_TESTS_RUN,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []
    assert exp.main(["--validate", "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["valid"] is True

    missing = tmp_path / "missing.json"
    assert exp.main(["--validate", "--output", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["errors"] == ["artifact_missing"]
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(unreadable)]) == 1
    assert "artifact_unreadable:JSONDecodeError" in capsys.readouterr().out

    cli_output = tmp_path / "cli.json"
    assert (
        exp.main(
            [
                "--date",
                "20260826",
                "--output",
                str(cli_output),
                "--work-dir",
                str(tmp_path / "cli-work"),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["task_owned_admission_ready_score"] == 1.0

    monkeypatch.setattr(exp, "validate_artifact", lambda _artifact: ["forced_invalid"])
    refused = tmp_path / "refused.json"
    with pytest.raises(ValueError, match="forced_invalid"):
        exp.run(
            date="20260826",
            root=REPO,
            result_path=refused,
            work_dir=tmp_path / "invalid-work",
            tests_run=exp.DEFAULT_TESTS_RUN,
        )
    assert not refused.exists()
