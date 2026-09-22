"""Tests for REQ-REPORT-7532 and SCENARIO-REPORT-7532-*.

Private fixtures use ``tmp_path``. Historical terminal evidence stays read-only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7532_v659_contract_methods as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True, required: bool = True) -> dict[str, Any]:
    """Build a complete receipt without starting a child process."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "required": required,
    }


def _validation(passed: bool = True) -> dict[str, Any]:
    """Supply every required command category to pure reducers."""

    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.TERMINAL_CHECK_NAMES,
    )
    receipts = [_receipt(name) for name in names]
    receipts.extend(_receipt(name, required=False) for name in exp.BASELINE_CHECK_NAMES)
    if not passed:
        receipts[0] = _receipt(names[0], passed=False)
    return {
        "validation_receipts": receipts,
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _authorities() -> tuple[str, dict[str, Any]]:
    """Read the independent V659 authorities from this worktree."""

    markdown = (ROOT / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(roadmap, dict)
    return markdown, roadmap


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build deterministic evidence from the current authenticated inputs."""

    markdown, roadmap = _authorities()
    path, _selected, candidates = exp.resolve_v659_roadmap(ROOT)
    return exp.build_artifact(
        ROOT,
        path,
        candidates,
        exp.compare_contract_authorities(markdown, roadmap),
        exp.run_contract_mutation_controls(markdown, roadmap),
        exp.collect_v658_dispositions(ROOT),
        exp.collect_b2_corrections(ROOT),
        validation or _validation(),
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=exp.zero_test_phase_spans(),
    )


def test_v659_authority_shortfall_fails_closed() -> None:
    """REQ-REPORT-7532; SCENARIO-REPORT-7532-CONTRACT."""

    markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities(markdown, roadmap)
    assert comparison["passed"] is False
    assert "markdown_task_table_missing" in comparison["errors"]
    assert "yaml_task_count" in comparison["errors"]
    assert [row["id"] for row in comparison["yaml_task_rows"]] == list(exp.EXPECTED_TASK_IDS[:8])
    assert len(exp.EXPECTED_TASK_IDS) == 14

    mutations = exp.mutation_names()
    assert set(mutations) == {
        "count",
        "order",
        "id",
        "title",
        "path",
        "gate_field",
        "milestone",
        "substrate",
    }
    assert (
        exp._mutate_yaml(roadmap, "substrate")["tasks"][0]["inference_substrate_class"]
        == "changed_substrate"
    )


def test_resolution_prefers_matching_staged_then_active(tmp_path: Path) -> None:
    """REQ-REPORT-7532 accepts V659 before or after activation."""

    _markdown, roadmap = _authorities()
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    staged = tmp_path / exp.NEXT_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v659_roadmap(tmp_path)[0] == active

    stale = deepcopy(roadmap)
    stale["milestone"] = "2026.09.658"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    assert exp.resolve_v659_roadmap(tmp_path)[0] == active
    staged.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v659_roadmap(tmp_path)[0] == staged

    active.unlink()
    staged.write_text("- malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V659 roadmap authority"):
        exp.resolve_v659_roadmap(tmp_path)


def test_v658_and_b2_custody_preserves_literal_limits() -> None:
    """REQ-REPORT-7532; SCENARIO-REPORT-7532-CUSTODY."""

    rows = exp.collect_v658_dispositions(ROOT)
    assert len(rows) == 14
    assert all(row["authenticated"] for row in rows)
    absent = [row for row in rows if row["producer_present"] is False]
    assert [row["task_id"] for row in absent] == [
        f"exp{number}-{slug}"
        for number, slug in (
            (7518, "source-pilot"),
            (7519, "source-fit-capture"),
            (7520, "source-eval-capture"),
            (7521, "consistency-energy"),
            (7522, "source-evaluation"),
            (7523, "count-memory"),
            (7524, "count-online"),
        )
    ]
    source = next(row for row in rows if row["task_id"] == "exp7517-source-protocol")
    assert source["candidate_official_training_groups"] == 2487
    assert source["exposed_candidate_groups"] == 2487
    assert source["fresh_eligible_groups"] == 0

    corrections = exp.collect_b2_corrections(ROOT)
    assert len(corrections) == 2
    assert corrections[0]["progress_attempt_count"] == 60
    assert corrections[0]["progress_proxy_saturated"] is True
    assert corrections[1]["attempts_at_harness_token_cap"] == 60
    assert corrections[1]["harness_max_new_tokens"] == 256
    assert corrections[1]["shipped_production_setting_claimed"] is False
    assert corrections[1]["measurement_scope"] == "feasibility_only"


def test_four_methods_write_idempotent_bounded_records(tmp_path: Path) -> None:
    """REQ-REPORT-7532; SCENARIO-REPORT-7532-METHODS."""

    methods = exp.method_rows()
    assert {row["method_family"] for row in methods} == {
        "tool_grounding",
        "continual_calibration",
        "online_recalibration",
        "consistency",
    }
    assert all(
        row["primary_url"]
        and row["source_section"]
        and row["adaptation"]
        and row["counterexample"]
        and row["task_mapping"]
        and row["access_status"] == "accessed_20260922"
        and row["external_claim_is_carnot_measurement"] is False
        for row in methods
    )
    study = tmp_path / exp.STUDY_PATH
    study.parent.mkdir(parents=True, exist_ok=True)
    study.write_text("# Study record\n", encoding="utf-8")
    exp.write_method_records(tmp_path)
    first = study.read_text(encoding="utf-8")
    exp.write_method_records(tmp_path)
    assert study.read_text(encoding="utf-8") == first
    note = (tmp_path / exp.NOTE_PATH).read_text(encoding="utf-8")
    assert "External results are not local Carnot results" in note
    assert note.count("| tool_grounding |") == 1
    assert note.count("| consistency |") == 1


def test_validation_plans_are_scoped_and_health_is_separate(tmp_path: Path) -> None:
    """REQ-REPORT-7532; SCENARIO-REPORT-7532-VALIDATION."""

    selected, _roadmap, _candidates = exp.resolve_v659_roadmap(ROOT)
    repository = exp.build_repository_check_plan(ROOT, selected)
    assert [row.spec.name for row in repository] == [
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.BASELINE_CHECK_NAMES,
    ]
    by_name = {row.spec.name: row for row in repository}
    assert all(by_name[name].required for name in exp.REQUIRED_REPOSITORY_CHECK_NAMES)
    assert all(not by_name[name].required for name in exp.BASELINE_CHECK_NAMES)
    assert str(selected) in by_name["roadmap_schema"].spec.argv

    plan = exp.build_validation_plan(ROOT, tmp_path / "validation")
    assert exp.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(validation_scope.REQUIRED_CHECK_NAMES)
    commands = {row.name: row for row in plan}
    assert "tests/python" not in commands["focused_pytest"].argv
    assert "--no-cov" in commands["focused_pytest"].argv
    assert any(arg.startswith("--basetemp=") for arg in commands["focused_pytest"].argv)
    coverage_env = dict(commands["changed_module_coverage_report"].command_environment)
    assert coverage_env["COVERAGE_FILE"].startswith("/tmp/")
    terminal = exp._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
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


def test_artifact_is_complete_null_and_independently_reducible() -> None:
    """REQ-REPORT-7532; SCENARIO-REPORT-7532-VALIDATION."""

    artifact = _artifact()
    reduction = exp.independent_reduce(artifact)
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["run_date"] == "20260922"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["contract_ready_score"] == 0
    assert artifact["method_ingestion_complete_score"] == 1
    assert artifact["honest_verdict"] == "complete_blocked_incomplete_v659_authorities"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["operator_queue"]["e0_status"] == "operator_blocked"
    assert artifact["operator_queue"]["e6_status"] == "resolved"
    assert reduction["contract_ready_score"] == 0
    assert reduction["verdict_class"] == "blocked"
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_cold_validator_rejects_custody_and_validation_drift() -> None:
    """REQ-REPORT-7532 fails closed on protected evidence and receipts."""

    artifact = _artifact()
    drifted = deepcopy(artifact)
    drifted["task_contract_rows"] = []
    drifted["contract_mutation_rows"][0]["qualified"] = True
    drifted["prior_dispositions"][1]["exposed_candidate_groups"] = 0
    drifted["b2_corrections"][1]["harness_max_new_tokens"] = 4096
    drifted["method_rows"][0]["access_status"] = "failed"
    drifted["operator_queue"]["e0_status"] = "resolved"
    drifted["contract_ready_score"] = 1
    drifted["field_principles"].pop("schema")
    drifted["acceptance_gate_results"] = []
    assert {
        "task_contract_rows_mismatch",
        "contract_mutations_invalid",
        "prior_dispositions_mismatch",
        "b2_corrections_mismatch",
        "method_rows_mismatch",
        "operator_queue_mismatch",
        "contract_score_mismatch",
        "field_principles_invalid",
        "gate_principles_invalid",
        "reproducibility_checksum_mismatch",
    }.issubset(exp.validate_artifact(drifted, root=ROOT))

    failed = _artifact(_validation(passed=False))
    assert failed["contract_ready_score"] == 0
    assert failed["verdict_class"] == "blocked"
    assert failed["gate_check_summary"]["failed_count"] >= 1


def test_blocked_artifact_names_exact_missing_upstream() -> None:
    """REQ-REPORT-7532 records external absence as blocked, never partial."""

    artifact = exp.build_blocked_artifact(
        Path("results/missing-upstream.json"),
        "artifact_field.ready_score",
        1,
        None,
        "2026-09-22T12:00:00+00:00",
        10,
    )
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    failed = artifact["gate_check_summary"]["failed_checks"][0]
    assert failed["upstream"] == "results/missing-upstream.json"
    assert failed["field_path"] == "artifact_field.ready_score"
    assert failed["expected"] == 1
    assert failed["observed"] is None


def test_defensive_readers_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-7532 keeps malformed input and date failures explicit."""

    assert exp.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    _markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities("not a contract", roadmap)
    assert comparison["passed"] is False
    assert comparison["errors"][0] == "markdown_task_table_missing"
    assert exp._producer_declarations({"tasks": [None]}) == {}
    assert exp.terminal_state(False, True, False)[2] == "blocked"
    assert exp._hashes_match({"source_artifact_hashes": None}, ROOT) is False
    assert exp._hashes_match({"source_artifact_hashes": {}}, tmp_path) is False
    assert "source_reduction_failed:ValueError" in exp.validate_artifact({}, root=tmp_path)

    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260921"])
    assert exp.parse_args(["--date", "20260922"]).date == "20260922"
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_artifact()), encoding="utf-8")
    args = exp.parse_args(["--date", "20260922", "--validate", str(candidate)])
    assert args.validate == candidate


def test_private_mutations_and_defensive_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7532 covers every fail-closed private control branch."""

    _markdown, roadmap = _authorities()
    for mutation in exp.mutation_names():
        assert exp._mutate_yaml(roadmap, mutation) != roadmap
    stale = deepcopy(roadmap)
    stale["milestone"] = "2026.09.658"
    assert "yaml_milestone" in exp.compare_contract_authorities("not a contract", stale)["errors"]
    assert exp.compare_contract_authorities("not a contract", ["bad"])["errors"][0].startswith(
        "parse_error:"
    )
    assert exp._producer_declarations({"tasks": [{"id": "x", "gated_on": [None]}]}) == {"x": []}

    artifact = _artifact()
    monkeypatch.setattr(exp, "load_json", lambda _path: {})
    assert exp.collect_v658_dispositions(tmp_path) == []
    monkeypatch.undo()
    assert exp.terminal_state(True, False, False)[2] == "disqualified"
    assert exp.terminal_state(True, True, True)[2] == "null"

    artifact["contract_comparison"] = {}
    artifact["validation_receipts"] = []
    artifact["source_artifact_hashes"] = {}
    errors = exp.validate_artifact(artifact, root=ROOT)
    assert {
        "contract_comparison_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "repository_health_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
    }.issubset(errors)
