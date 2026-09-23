"""Tests for REQ-REPORT-7546 and SCENARIO-REPORT-7546-*.

Private fixtures never replace the current worktree inputs or terminal evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7546_v660_contract_methods as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True, required: bool = True) -> dict[str, Any]:
    """Build one private command receipt without starting a child process."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
        "required": required,
    }


def _validation(passed: bool = True) -> dict[str, Any]:
    """Supply each fixed command category to the pure artifact reducer."""

    required = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.TERMINAL_CHECK_NAMES,
    )
    receipts = [_receipt(name) for name in required]
    receipts.extend(
        _receipt(name, required=False) for name in exp.REPOSITORY_DIAGNOSTIC_CHECK_NAMES
    )
    if not passed:
        receipts[0] = _receipt(required[0], passed=False)
    return {
        "validation_receipts": receipts,
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _authorities() -> tuple[str, dict[str, Any]]:
    """Read the independent V660 authorities from this worktree."""

    markdown = (ROOT / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(roadmap, dict)
    return markdown, roadmap


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build deterministic evidence from authenticated repository inputs."""

    markdown, roadmap = _authorities()
    path, _selected, candidates = exp.resolve_v660_roadmap(ROOT)
    return exp.build_artifact(
        ROOT,
        path,
        candidates,
        exp.compare_contract_authorities(markdown, roadmap),
        exp.run_contract_mutation_controls(markdown, roadmap),
        exp.collect_v659_dispositions(ROOT),
        validation or _validation(),
        started_at_utc="2026-09-23T12:00:00+00:00",
        ended_at_utc="2026-09-23T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=exp.zero_test_phase_spans(),
    )


def test_complete_authorities_and_private_mutations() -> None:
    """REQ-REPORT-7546; SCENARIO-REPORT-7546-CONTRACT."""

    markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities(markdown, roadmap)
    assert comparison["passed"] is True
    assert comparison["errors"] == []
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(exp.EXPECTED_TASK_IDS)
    assert all(row["passed"] for row in comparison["contract_rows"])
    assert comparison["contract_rows"][3]["yaml"]["gates"]

    controls = exp.run_contract_mutation_controls(markdown, roadmap)
    assert len(controls) == 2 * len(exp.mutation_names())
    assert all(row["baseline_valid"] and row["rejected"] and row["qualified"] for row in controls)
    assert {row["mutation"] for row in controls} == set(exp.mutation_names())


def test_resolution_accepts_staged_then_active(tmp_path: Path) -> None:
    """REQ-REPORT-7546 accepts V660 before or after roadmap activation."""

    _markdown, roadmap = _authorities()
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    staged = tmp_path / exp.NEXT_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v660_roadmap(tmp_path)[0] == active

    stale = deepcopy(roadmap)
    stale["milestone"] = "2026.09.659"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    assert exp.resolve_v660_roadmap(tmp_path)[0] == active
    staged.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v660_roadmap(tmp_path)[0] == staged
    active.unlink()
    staged.write_text("- malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V660 roadmap authority"):
        exp.resolve_v660_roadmap(tmp_path)


def test_v659_dispositions_preserve_absence_and_historical_defects() -> None:
    """REQ-REPORT-7546; SCENARIO-REPORT-7546-CUSTODY."""

    rows = exp.collect_v659_dispositions(ROOT)
    assert len(rows) == 8
    assert [row["task_id"] for row in rows] == list(exp.V659_TASK_IDS)
    assert all(row["authenticated"] for row in rows)
    assert rows[0]["historical_execution_venue"] == "host_cpu"
    assert rows[0]["historical_flagged_adversarial"] is True
    assert rows[3]["current_invocation_count"] == 0
    assert rows[4]["evidence_kind"] == rows[5]["evidence_kind"] == "pre_gate_diagnostic"
    assert rows[4]["failed_field"] == rows[5]["failed_field"] == "native_tool_ready_score"
    assert rows[6]["producer_present"] is rows[7]["producer_present"] is False
    assert rows[6]["missing_is_zero_metric"] is rows[7]["missing_is_zero_metric"] is False

    promises = exp.unissued_promises()
    assert [row["task_id"] for row in promises] == [f"exp{number}" for number in range(7540, 7546)]
    assert all(row["issued"] is False and row["archive_entry_created"] is False for row in promises)


def test_method_records_keep_primary_limits_and_access_failures(tmp_path: Path) -> None:
    """REQ-REPORT-7546; SCENARIO-REPORT-7546-METHODS."""

    methods = exp.method_rows()
    assert 3 <= len(methods) <= 5
    assert {row["method_family"] for row in methods} >= {
        "safe_source_ablation",
        "adversarial_world_model_sequences",
        "online_proper_loss_recalibration",
    }
    assert all(
        row["primary_url"]
        and row["source_section"]
        and row["adaptation"]
        and row["counterexample"]
        and row["task_mapping"]
        and row["external_claim_is_carnot_measurement"] is False
        for row in methods
    )
    access = exp.secondary_access_rows()
    assert {(row["channel"], row["observed"]) for row in access} == {
        ("Semantic Scholar", "HTTP 429"),
        ("OpenReview", "browser challenge"),
    }
    assert all(row["complete"] is False for row in access)

    study = tmp_path / exp.STUDY_PATH
    study.parent.mkdir(parents=True, exist_ok=True)
    study.write_text("# Study record\n", encoding="utf-8")
    exp.write_method_records(tmp_path)
    first = study.read_text(encoding="utf-8")
    exp.write_method_records(tmp_path)
    assert study.read_text(encoding="utf-8") == first
    note = (tmp_path / exp.NOTE_PATH).read_text(encoding="utf-8")
    assert "External results are not local Carnot results" in note
    assert "HTTP 429" in note and "browser challenge" in note


def test_validation_plan_is_scoped_and_policy_checks_are_unchanged(tmp_path: Path) -> None:
    """REQ-REPORT-7546; SCENARIO-REPORT-7546-VALIDATION."""

    selected, _roadmap, _candidates = exp.resolve_v660_roadmap(ROOT)
    plan = exp.build_validation_plan(ROOT, tmp_path / "validation")
    assert exp.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(validation_scope.REQUIRED_CHECK_NAMES)
    commands = {row.name: row for row in plan}
    assert "tests/python" not in commands["focused_pytest"].argv
    assert ("-n", "0", "-o", "addopts=", "--no-cov") == commands["focused_pytest"].argv[1:6]
    assert any(arg.startswith("--basetemp=") for arg in commands["focused_pytest"].argv)
    assert "--fail-under=100" in commands["changed_module_coverage_report"].argv

    repository = exp.build_repository_check_plan(ROOT, selected)
    assert [row.spec.name for row in repository] == [
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.REPOSITORY_DIAGNOSTIC_CHECK_NAMES,
    ]
    assert all(row.required for row in repository[: len(exp.REQUIRED_REPOSITORY_CHECK_NAMES)])
    assert all(
        row.required is False for row in repository[len(exp.REQUIRED_REPOSITORY_CHECK_NAMES) :]
    )
    assert repository[-1].spec.argv == (str(ROOT / ".venv/bin/pytest"), "tests/python", "-q")
    terminal = exp._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert all(row.required for row in terminal)

    assert exp.prior_full_suite_receipt(tmp_path) is None
    prior = tmp_path / exp.RESULT_PATH
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text("{}", encoding="utf-8")
    assert exp.prior_full_suite_receipt(tmp_path) is None
    prior.write_text(
        json.dumps({"validation_receipts": [_receipt("full_python_suite", passed=False)]}),
        encoding="utf-8",
    )
    assert exp.prior_full_suite_receipt(tmp_path)["exit_code"] == 1


def test_complete_artifact_is_null_and_independently_reducible() -> None:
    """REQ-REPORT-7546; SCENARIO-REPORT-7546-VALIDATION."""

    artifact = _artifact()
    reduction = exp.independent_reduce(artifact)
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["run_date"] == "20260923"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["contract_ready_score"] == 1
    assert artifact["method_ingestion_complete_score"] == 1
    assert artifact["positive_claim"] is False
    assert artifact["honest_verdict"] == "complete_null_v660_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["operator_queue"]["e0_status"] == "operator_blocked"
    assert artifact["operator_queue"]["e6_status"] == "resolved"
    assert artifact["publication_gates"]["paper_ready"] is True
    assert artifact["publication_gates"]["unmet_gates"] == []
    assert reduction["contract_ready_score"] == 1
    assert reduction["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_cold_validator_rejects_evidence_and_validation_drift() -> None:
    """REQ-REPORT-7546 fails closed when protected evidence changes."""

    artifact = _artifact()
    drifted = deepcopy(artifact)
    drifted["task_contract_rows"] = []
    drifted["contract_comparison"] = {}
    drifted["contract_mutation_rows"][0]["qualified"] = False
    drifted["prior_dispositions"][0]["historical_execution_venue"] = "host"
    drifted["unissued_promises"][0]["issued"] = True
    drifted["method_rows"][0]["access_status"] = "failed"
    drifted["secondary_access_rows"] = []
    drifted["operator_queue"]["e0_status"] = "resolved"
    drifted["contract_ready_score"] = 1
    drifted["validation_receipts"] = []
    drifted["source_artifact_hashes"] = {}
    drifted["positive_claim"] = True
    drifted["publication_gates"] = {}
    drifted["field_principles"].pop("schema")
    drifted["acceptance_gate_results"] = []
    drifted["gate_check_summary"] = {}
    assert {
        "task_contract_rows_mismatch",
        "contract_comparison_mismatch",
        "contract_mutations_invalid",
        "prior_dispositions_mismatch",
        "unissued_promises_mismatch",
        "method_rows_mismatch",
        "secondary_access_rows_mismatch",
        "operator_queue_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "repository_health_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "contract_score_mismatch",
        "positive_claim_invalid",
        "publication_gates_mismatch",
        "field_principles_invalid",
        "gate_principles_invalid",
        "gate_summary_mismatch",
        "reproducibility_checksum_mismatch",
    }.issubset(exp.validate_artifact(drifted, root=ROOT))

    failed = _artifact(_validation(passed=False))
    assert failed["contract_ready_score"] == 0
    assert failed["verdict_class"] == "disqualified"
    assert failed["gate_check_summary"]["failed_count"] >= 1


def test_blocked_artifact_names_exact_missing_upstream() -> None:
    """REQ-REPORT-7546 records external absence as blocked, never partial."""

    artifact = exp.build_blocked_artifact(
        Path("results/missing-upstream.json"),
        "artifact_field.ready_score",
        1,
        None,
        "2026-09-23T12:00:00+00:00",
        10,
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "aggregation"
    failed = artifact["gate_check_summary"]["failed_checks"][0]
    assert failed["upstream"] == "results/missing-upstream.json"
    assert failed["field_path"] == "artifact_field.ready_score"
    assert failed["expected"] == 1
    assert failed["observed"] is None


def test_defensive_readers_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-7546 keeps malformed sources and dates explicit."""

    assert exp.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    _markdown, roadmap = _authorities()
    assert exp.compare_contract_authorities("not a contract", roadmap)["passed"] is False
    assert exp.compare_contract_authorities("not a contract", ["bad"])["errors"][0].startswith(
        "parse_error:"
    )
    assert exp._producer_declarations({"tasks": [None]}) == {}
    assert exp.terminal_state(False, True)[2] == "blocked"
    assert exp.terminal_state(True, False)[2] == "disqualified"
    assert exp.terminal_state(True, True)[2] == "null"
    assert exp._hashes_match({"source_artifact_hashes": None}, ROOT) is False
    assert exp._hashes_match({"source_artifact_hashes": {}}, tmp_path) is False
    assert "source_reduction_failed:ValueError" in exp.validate_artifact({}, root=tmp_path)

    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260922"])
    assert exp.parse_args(["--date", "20260923"]).date == "20260923"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_artifact()), encoding="utf-8")
    args = exp.parse_args(["--date", "20260923", "--validate", str(candidate)])
    assert args.validate == candidate


def test_private_mutator_branches_and_baseline_failure() -> None:
    """REQ-REPORT-7546 rejects each named authority defect independently."""

    markdown, roadmap = _authorities()
    for mutation in exp.mutation_names():
        assert exp._mutate_yaml(roadmap, mutation) != roadmap
        assert exp._mutate_markdown(markdown, mutation) != markdown
    stale = deepcopy(roadmap)
    stale["milestone"] = "2026.09.659"
    assert "yaml_milestone" in exp.compare_contract_authorities(markdown, stale)["errors"]
    assert exp._producer_declarations({"tasks": [{"id": "x", "gated_on": [None]}]}) == {"x": []}

    incomplete = deepcopy(roadmap)
    incomplete["tasks"] = incomplete["tasks"][:-1]
    controls = exp.run_contract_mutation_controls(markdown, incomplete)
    assert all(row["baseline_valid"] is False for row in controls)
    assert all(row["qualified"] is False for row in controls)
