"""Tests for REQ-REPORT-7489 and SCENARIO-REPORT-7489-*.

The tests keep contract mutations and document writes in private paths. They
read historical artifacts but never rewrite the tracked research record.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7489_v656_contract_methods as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Return one complete private receipt for reducer boundary tests."""

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
        "required": True,
    }


def _validation(passed: bool = True) -> dict[str, Any]:
    """Build every named validation receipt without running a subprocess."""

    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REPOSITORY_CHECK_NAMES,
        *exp.TERMINAL_CHECK_NAMES,
    )
    receipts = [_receipt(name) for name in names]
    if not passed:
        receipts[1] = _receipt(names[1], passed=False)
    return {
        "validation_receipts": receipts,
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _real_contract() -> tuple[str, dict[str, Any]]:
    """Read the two current V656 authorities for pure reducer tests."""

    markdown = (ROOT / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(roadmap, dict)
    return markdown, roadmap


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic artifact shell from authenticated repository bytes."""

    markdown, roadmap = _real_contract()
    contract = exp.compare_contract_authorities(markdown, roadmap)
    mutations = exp.run_contract_mutation_controls(markdown, roadmap)
    dispositions = exp.collect_v655_dispositions(ROOT)
    roadmap_path, _selected, _candidates = exp.resolve_v656_roadmap(ROOT)
    return exp.build_artifact(
        ROOT,
        roadmap_path,
        contract,
        mutations,
        dispositions,
        validation or _validation(),
        started_at_utc="2026-09-21T12:00:00+00:00",
        ended_at_utc="2026-09-21T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=exp.zero_test_phase_spans(),
    )


def test_v656_authorities_match_and_all_private_mutations_fail() -> None:
    """REQ-REPORT-7489; SCENARIO-REPORT-7489-CONTRACT."""

    markdown, roadmap = _real_contract()
    comparison = exp.compare_contract_authorities(markdown, roadmap)

    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(
        exp.EXPECTED_TASK_IDS
    )
    assert len(comparison["contract_rows"]) == 14
    assert all(row["principle"] for row in comparison["contract_rows"])

    mutations = exp.run_contract_mutation_controls(markdown, roadmap)
    assert len(mutations) == 12
    assert {row["mutation"] for row in mutations} == {
        "count",
        "id",
        "order",
        "path",
        "gate_field",
        "milestone",
    }
    assert all(row["rejected"] and row["other_authority_readable"] for row in mutations)


def test_roadmap_resolution_prefers_matching_next_then_active(tmp_path: Path) -> None:
    """REQ-REPORT-7489 selects only the V656 authority."""

    _markdown, roadmap = _real_contract()
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    staged = tmp_path / exp.NEXT_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap), encoding="utf-8")

    path, selected, candidates = exp.resolve_v656_roadmap(tmp_path)
    assert path == active
    assert selected["milestone"] == exp.MILESTONE
    assert candidates[0]["exists"] is False

    staged_value = deepcopy(roadmap)
    staged_value["milestone"] = "2026.09.655"
    staged.write_text(yaml.safe_dump(staged_value), encoding="utf-8")
    assert exp.resolve_v656_roadmap(tmp_path)[0] == active

    staged.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v656_roadmap(tmp_path)[0] == staged

    active.unlink()
    staged.write_text("- malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V656 roadmap authority"):
        exp.resolve_v656_roadmap(tmp_path)


def test_v655_dispositions_preserve_failures_absence_and_capstone() -> None:
    """REQ-REPORT-7489; SCENARIO-REPORT-7489-EVIDENCE."""

    rows = exp.collect_v655_dispositions(ROOT)
    by_id = {row["task_id"]: row for row in rows}

    assert len(rows) == 14
    assert [row["task_id"] for row in rows] == list(exp.V655_TASK_IDS)
    assert all(row["authenticated"] is True for row in rows)
    assert by_id["exp7475-contract-methods"]["original_verdict_class"] == "disqualified"
    assert [
        row["name"]
        for row in by_id["exp7475-contract-methods"]["failed_validation_receipts"]
    ] == ["overdue_priority"]
    assert [
        row["name"] for row in by_id["exp7484-decision-audit"]["failed_validation_receipts"]
    ] == ["adversarial_verify"]
    assert by_id["exp7484-decision-audit"]["original_inference_substrate"] == (
        "aggregation_from_hash_bound_raw_rows_and_frozen_numeric_checkpoints"
    )
    assert by_id["exp7486-arc-cost-panel-b"] == {
        **by_id["exp7486-arc-cost-panel-b"],
        "source_kind": "absent_producer",
        "producer_present": False,
        "source_sha256": None,
        "original_verdict_class": "blocked",
        "authenticated": True,
    }
    assert by_id["exp7488-capstone"]["source_kind"] == "disqualified_capstone"
    assert by_id["exp7488-capstone"]["producer_present"] is True
    assert by_id["exp7488-capstone"]["source_sha256"].startswith("sha256:")
    assert by_id["exp7488-capstone"]["original_verdict_class"] == "disqualified"


def test_six_methods_write_idempotent_bounded_records(tmp_path: Path) -> None:
    """REQ-REPORT-7489; SCENARIO-REPORT-7489-METHODS."""

    methods = exp.method_rows()
    assert len(methods) == 6
    assert {row["method"] for row in methods} == {
        "response_granularity",
        "evidence_alignment",
        "corpus_leakage_controls",
        "budgeted_feedback",
        "kan_support_limits",
        "on_chip_locality",
    }
    assert all(
        row["source_revision"]
        and row["reusable_component"]
        and row["limitation"]
        and row["task_mapping"]
        and row["external_claim_is_carnot_measurement"] is False
        and row["principle"]
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
    assert "External paper results are not Carnot measurements" in note
    assert note.count("| response_granularity |") == 1


def test_priority_parser_association_and_callsite_evidence_stay_separate() -> None:
    """REQ-REPORT-7489; SCENARIO-REPORT-7489-PRIORITY."""

    disposition = exp.priority_disposition(ROOT)
    assert disposition["priority_title"].startswith("AUTORESEARCH CIRCUIT BREAKER")
    assert disposition["addressed_marker_present"] is True
    assert disposition["exp7435_ready_score"] == 1
    assert disposition["parser_slug"] == "retro_timing_fallback_wiring"
    assert disposition["slug_present_in_selected_roadmap"] is True
    assert disposition["slug_match_proves_wiring"] is False
    assert disposition["timing_fallback"]["callsite_present"] is True
    assert disposition["timing_fallback"]["known_issue_state"] == "resolved"
    assert disposition["timing_fallback"]["current_task_modified_conductor"] is False

    obligations = exp.unresolved_obligations(ROOT)
    assert obligations[0]["obligation_id"] == "overdue_priority_cross_heading_association"
    assert obligations[0]["state"] == "unresolved_parser_scope"
    assert obligations[1]["prohibited_path"] == "scripts/research_conductor.py"


def test_repository_and_affected_validation_plans_are_exact(tmp_path: Path) -> None:
    """REQ-REPORT-7489 keeps every check file-scoped and unchanged."""

    selected, _roadmap, _candidates = exp.resolve_v656_roadmap(ROOT)
    repository = exp.build_repository_check_plan(ROOT, selected)
    assert [row.name for row in repository] == list(exp.REPOSITORY_CHECK_NAMES)
    assert all(str(selected) in row.argv for row in repository)

    private = tmp_path / "validation"
    plan = exp.build_validation_plan(ROOT, private)
    assert exp.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(validation_scope.REQUIRED_CHECK_NAMES)
    commands = {row.name: row for row in plan}
    assert "tests/python" not in commands["focused_pytest"].argv
    assert "--no-cov" in commands["focused_pytest"].argv
    assert any(arg.startswith("--basetemp=") for arg in commands["focused_pytest"].argv)
    assert dict(getattr(commands["changed_module_coverage_report"], "command_environment"))[
        "COVERAGE_FILE"
    ].startswith("/tmp/")


def test_terminal_artifact_is_complete_null_advisory_evidence() -> None:
    """REQ-REPORT-7489; SCENARIO-REPORT-7489-ARTIFACT."""

    artifact = _artifact()
    reduction = exp.independent_reduce(artifact)

    assert artifact["schema"] == exp.SCHEMA
    assert artifact["run_date"] == "20260921"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["contract_ready_score"] == 1
    assert artifact["honest_verdict"] == "complete_null_v656_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_adversarial"] is False
    assert len(artifact["v655_dispositions"]) == 14
    assert len(artifact["method_rows"]) == 6
    assert reduction["contract_ready_score"] == 1
    assert reduction["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert all(artifact["field_principles"].values())
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_cold_validator_rejects_drift_and_failed_required_validation() -> None:
    """REQ-REPORT-7489 fails closed on rows, hashes, scores, and receipts."""

    artifact = _artifact()
    drifted = deepcopy(artifact)
    drifted["method_rows"][0]["source_revision"] = "changed"
    drifted["source_artifact_hashes"][exp.DESIGN_PATH.as_posix()]["sha256"] = (
        "sha256:" + "0" * 64
    )
    drifted["contract_ready_score"] = 0
    drifted["field_principles"].pop("schema")
    drifted["acceptance_gate_results"][0]["principle"] = ""
    assert {
        "method_rows_mismatch",
        "source_hash_mismatch",
        "contract_score_mismatch",
        "field_principles_invalid",
        "gate_principles_invalid",
        "reproducibility_checksum_mismatch",
    }.issubset(exp.validate_artifact(drifted, root=ROOT))

    failed = _artifact(_validation(passed=False))
    assert failed["contract_ready_score"] == 0
    assert failed["verdict_class"] == "disqualified"
    assert failed["gate_check_summary"]["failed_count"] >= 1


def test_defensive_loaders_receipts_and_terminal_classification(tmp_path: Path) -> None:
    """REQ-REPORT-7489 rejects malformed inputs and preserves terminal semantics."""

    mapping = tmp_path / "mapping.json"
    mapping.write_text('{"a": 1}\n', encoding="utf-8")
    assert exp.load_json(mapping) == {"a": 1}
    mapping.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON mapping"):
        exp.load_json(mapping)

    document = tmp_path / "mapping.yaml"
    document.write_text("a: 1\n", encoding="utf-8")
    assert exp.load_yaml(document) == {"a": 1}
    document.write_text("[\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        exp.load_yaml(document)

    good = [_receipt(name) for name in exp.REPOSITORY_CHECK_NAMES]
    assert exp.receipts_pass(good, exp.REPOSITORY_CHECK_NAMES) is True
    assert exp.receipts_recorded(good, exp.REPOSITORY_CHECK_NAMES) is True
    assert exp.receipts_pass(good + [deepcopy(good[0])], exp.REPOSITORY_CHECK_NAMES) is False
    assert exp.receipts_recorded({}, exp.REPOSITORY_CHECK_NAMES) is False

    assert exp.terminal_state(False, True, False)[2] == "blocked"
    assert exp.terminal_state(True, False, False)[2] == "disqualified"
    assert exp.terminal_state(True, True, True)[2] == "null"


def test_contract_parse_failure_and_cli_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-7489 keeps malformed authority and date failures explicit."""

    _markdown, roadmap = _real_contract()
    comparison = exp.compare_contract_authorities("not a contract", roadmap)
    assert comparison["passed"] is False
    assert comparison["errors"][0].startswith("parse_error:")

    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260920"])
    assert exp.parse_args(["--date", "20260921"]).date == "20260921"

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_artifact()), encoding="utf-8")
    args = exp.parse_args(["--date", "20260921", "--validate", str(candidate)])
    assert args.validate == candidate

