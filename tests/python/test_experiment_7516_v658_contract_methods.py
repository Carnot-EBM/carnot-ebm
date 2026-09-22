"""Tests for REQ-REPORT-7516 and SCENARIO-REPORT-7516-*.

Private writers use ``tmp_path``. Published V657 evidence stays read-only.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7516_v658_contract_methods as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from scripts.verdict_row_consistency_lint import check_artifact


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True, required: bool = True) -> dict[str, Any]:
    """Build one complete command receipt for a pure reducer test."""

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
    """Supply each required receipt without starting subprocesses."""

    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.TERMINAL_CHECK_NAMES,
    )
    receipts = [_receipt(name) for name in names]
    receipts.extend(_receipt(name, required=False) for name in exp.BASELINE_CHECK_NAMES)
    if not passed:
        receipts[1] = _receipt(names[1], passed=False)
    return {
        "validation_receipts": receipts,
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _authorities() -> tuple[str, dict[str, Any]]:
    """Read the independent V658 authorities from the current worktree."""

    markdown = (ROOT / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(roadmap, dict)
    return markdown, roadmap


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build deterministic evidence from current immutable inputs."""

    markdown, roadmap = _authorities()
    path, _selected, candidates = exp.resolve_v658_roadmap(ROOT)
    return exp.build_artifact(
        ROOT,
        path,
        candidates,
        exp.compare_contract_authorities(markdown, roadmap),
        exp.run_contract_mutation_controls(markdown, roadmap),
        exp.collect_v657_dispositions(ROOT),
        validation or _validation(),
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=exp.zero_test_phase_spans(),
    )


def test_v658_authorities_match_and_private_mutations_fail() -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-CONTRACT."""

    markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities(markdown, roadmap)
    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(exp.EXPECTED_TASK_IDS)
    assert len(comparison["contract_rows"]) == 14
    assert all(row["passed"] and row["principle"] for row in comparison["contract_rows"])

    mutations = exp.run_contract_mutation_controls(markdown, roadmap)
    assert len(mutations) == 14
    assert {row["mutation"] for row in mutations} == {
        "count",
        "order",
        "id",
        "title",
        "path",
        "gate_field",
        "milestone",
    }
    assert all(row["rejected"] and row["other_authority_readable"] for row in mutations)


def test_resolution_prefers_matching_staged_then_active(tmp_path: Path) -> None:
    """REQ-REPORT-7516 accepts V658 before or after activation."""

    _markdown, roadmap = _authorities()
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    staged = tmp_path / exp.NEXT_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v658_roadmap(tmp_path)[0] == active

    stale = deepcopy(roadmap)
    stale["milestone"] = "2026.09.657"
    staged.write_text(yaml.safe_dump(stale), encoding="utf-8")
    assert exp.resolve_v658_roadmap(tmp_path)[0] == active

    staged.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v658_roadmap(tmp_path)[0] == staged

    active.unlink()
    staged.write_text("- malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V658 roadmap authority"):
        exp.resolve_v658_roadmap(tmp_path)


def test_v657_custody_preserves_disqualified_and_valid_null_states() -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-CUSTODY."""

    rows = exp.collect_v657_dispositions(ROOT)
    by_id = {row["task_id"]: row for row in rows}
    assert len(rows) == 13
    assert all(row["producer_present"] and row["authenticated"] for row in rows)

    static = by_id["exp7508-static-audit"]
    assert static["original_verdict_class"] == "disqualified"
    assert static["original_flagged_adversarial"] is True
    assert static["strict_reader_exit_code"] != 0
    assert static["strict_no_headroom_advisories"] == 8
    assert static["strict_blocking_findings"] == 0
    assert static["signed_cost_difference_is_headroom_metric"] is False

    capstone = by_id["exp7515-capstone"]
    assert capstone["original_verdict_class"] == "disqualified"
    assert capstone["original_flagged_adversarial"] is True
    for task_id in ("exp7509-causal-online", "exp7510-causal-audit"):
        assert by_id[task_id]["original_verdict_class"] == "null"
        assert by_id[task_id]["original_flagged_adversarial"] is False


def test_valid_null_fixture_uses_absolute_costs_and_unchanged_guard(tmp_path: Path) -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-NULL-SHAPE."""

    fixture = exp.valid_null_fixture()
    assert exp.validate_valid_null_fixture(fixture) == []
    assert all(row["arm_a_cost"] == row["arm_b_cost"] for row in fixture["rows"])
    assert all(row["no_headroom"] is True for row in fixture["rows"])
    assert all(row["positive_claim"] is False for row in fixture["rows"])
    assert all(row["headroom_explanation"] for row in fixture["rows"])

    path = tmp_path / "valid-null.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    assert check_artifact(path) == ("ok", [])

    invalid = deepcopy(fixture)
    invalid["rows"][0].pop("arm_a_cost")
    invalid["rows"][0]["signed_cost_difference"] = 0.0
    assert "row_absolute_costs_missing:0" in exp.validate_valid_null_fixture(invalid)
    assert exp.validate_valid_null_fixture({}) == ["fixture_rows_required"]
    malformed = {"rows": [None, {"arm_a_cost": 0.2, "arm_b_cost": 0.4}]}
    assert exp.validate_valid_null_fixture(malformed) == [
        "row_mapping_required:0",
        "row_no_headroom_invalid:1",
        "row_null_context_invalid:1",
    ]


def test_four_method_families_write_idempotent_records(tmp_path: Path) -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-METHODS."""

    methods = exp.method_rows()
    assert len(methods) == 4
    assert {row["method_family"] for row in methods} == {
        "consistency",
        "calibration",
        "online_proper_loss",
        "thermodynamic_co_design",
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

    study = tmp_path / exp.STUDY_PATH
    study.parent.mkdir(parents=True, exist_ok=True)
    study.write_text("# Study record\n", encoding="utf-8")
    exp.write_method_records(tmp_path)
    first = study.read_text(encoding="utf-8")
    exp.write_method_records(tmp_path)
    assert study.read_text(encoding="utf-8") == first
    note = (tmp_path / exp.NOTE_PATH).read_text(encoding="utf-8")
    assert "Paper results are not local Carnot results" in note
    assert note.count("| consistency |") == 1
    assert note.count("| thermodynamic_co_design |") == 1


def test_validation_plans_are_scoped_and_repository_health_is_separate(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-VALIDATION."""

    selected, _roadmap, _candidates = exp.resolve_v658_roadmap(ROOT)
    fixture_path = tmp_path / "valid-null.json"
    fixture_path.write_text(json.dumps(exp.valid_null_fixture()), encoding="utf-8")
    repository = exp.build_repository_check_plan(ROOT, selected, fixture_path)
    assert [row.spec.name for row in repository] == [
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.BASELINE_CHECK_NAMES,
    ]
    by_name = {row.spec.name: row for row in repository}
    assert all(by_name[name].required for name in exp.REQUIRED_REPOSITORY_CHECK_NAMES)
    assert all(not by_name[name].required for name in exp.BASELINE_CHECK_NAMES)
    assert str(selected) in by_name["roadmap_schema"].spec.argv
    assert str(fixture_path) in by_name["valid_null_guard"].spec.argv

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
    assert all(row.required for row in terminal)


def test_artifact_is_complete_null_with_separate_timing_evidence() -> None:
    """REQ-REPORT-7516; SCENARIO-REPORT-7516-VALIDATION."""

    artifact = _artifact()
    reduction = exp.independent_reduce(artifact)
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["run_date"] == "20260922"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["contract_ready_score"] == 1
    assert artifact["method_ingestion_complete_score"] == 1
    assert artifact["honest_verdict"] == "complete_null_v658_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["e6_timed_profile"]["fully_timed_complete_episodes"] == 36
    assert artifact["e6_timed_profile"]["induction_generation_tokens"] == 430188
    assert artifact["e6_timed_profile"]["separate_from_v657_coarse_timers"] is True
    assert reduction["contract_ready_score"] == 1
    assert reduction["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert all(artifact["field_principles"].values())
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_cold_validator_rejects_drift_and_failed_required_validation() -> None:
    """REQ-REPORT-7516 fails closed on protected evidence and receipts."""

    artifact = _artifact()
    drifted = deepcopy(artifact)
    drifted["method_rows"][0]["source_section"] = "changed"
    drifted["task_contract_rows"] = []
    drifted["prior_dispositions"][5]["strict_no_headroom_advisories"] = 0
    drifted["valid_null_fixture_qualification"] = {}
    drifted["priority_disposition"] = {}
    drifted["e6_timed_profile"] = {}
    drifted["field_principles"].pop("schema")
    drifted["acceptance_gate_results"] = []
    assert {
        "method_rows_mismatch",
        "task_contract_rows_mismatch",
        "prior_dispositions_mismatch",
        "valid_null_fixture_mismatch",
        "priority_disposition_mismatch",
        "e6_timed_profile_mismatch",
        "contract_score_mismatch",
        "field_principles_invalid",
        "gate_principles_invalid",
        "reproducibility_checksum_mismatch",
    }.issubset(exp.validate_artifact(drifted, root=ROOT))

    failed = _artifact(_validation(passed=False))
    assert failed["contract_ready_score"] == 0
    assert failed["verdict_class"] == "disqualified"
    assert failed["gate_check_summary"]["failed_count"] >= 1

    defensive = deepcopy(artifact)
    defensive.update(
        {
            "schema": "changed",
            "invocation_counts": {},
            "task_contract_rows": [],
            "contract_comparison": {},
            "contract_mutation_rows": [],
            "prior_dispositions": [],
            "method_rows": [],
            "validation_receipts": [],
            "source_artifact_hashes": None,
        }
    )
    assert {
        "identity_invalid",
        "model_contract_invalid",
        "task_contract_rows_mismatch",
        "contract_comparison_mismatch",
        "contract_mutations_invalid",
        "prior_dispositions_mismatch",
        "method_rows_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
    }.issubset(exp.validate_artifact(defensive, root=ROOT))


def test_defensive_readers_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-7516 keeps malformed inputs and date failures explicit."""

    assert exp.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]
    assert exp.validate_valid_null_fixture([]) == ["fixture_mapping_required"]
    _markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities("not a contract", roadmap)
    assert comparison["passed"] is False
    assert comparison["errors"][0].startswith("parse_error:")
    assert exp._producer_declarations({"tasks": [None]}) == {}
    assert exp._producer_declarations({"tasks": [{"id": "x", "gated_on": [None]}]}) == {"x": []}
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
