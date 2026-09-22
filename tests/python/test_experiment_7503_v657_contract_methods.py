"""Tests for REQ-REPORT-7503 and SCENARIO-REPORT-7503-*.

All writer tests use ``tmp_path``. Historical results are read-only inputs.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7503_v657_contract_methods as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build a complete private receipt for pure reducer tests."""

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
        "required": name != "overdue_priority",
    }


def _validation(passed: bool = True) -> dict[str, Any]:
    """Provide each required current receipt without running a child process."""

    names = (
        *validation_scope.REQUIRED_CHECK_NAMES,
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        *exp.TERMINAL_CHECK_NAMES,
    )
    receipts = [_receipt(name) for name in names]
    receipts.append(_receipt("overdue_priority", passed=False))
    if not passed:
        receipts[1] = _receipt(names[1], passed=False)
    return {
        "validation_receipts": receipts,
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _authorities() -> tuple[str, dict[str, Any]]:
    """Read the current independent V657 authorities."""

    markdown = (ROOT / exp.DESIGN_PATH).read_text(encoding="utf-8")
    roadmap = yaml.safe_load((ROOT / exp.ACTIVE_ROADMAP_PATH).read_text(encoding="utf-8"))
    assert isinstance(roadmap, dict)
    return markdown, roadmap


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a deterministic artifact from authenticated repository bytes."""

    markdown, roadmap = _authorities()
    path, _selected, candidates = exp.resolve_v657_roadmap(ROOT)
    return exp.build_artifact(
        ROOT,
        path,
        candidates,
        exp.compare_contract_authorities(markdown, roadmap),
        exp.run_contract_mutation_controls(markdown, roadmap),
        exp.collect_v656_dispositions(ROOT),
        validation or _validation(),
        started_at_utc="2026-09-22T12:00:00+00:00",
        ended_at_utc="2026-09-22T12:00:01+00:00",
        started_monotonic_ns=10,
        ended_monotonic_ns=1_000_000_010,
        phase_spans=exp.zero_test_phase_spans(),
    )


def test_v657_authorities_match_and_all_private_mutations_fail() -> None:
    """REQ-REPORT-7503; SCENARIO-REPORT-7503-CONTRACT."""

    markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities(markdown, roadmap)

    assert comparison["passed"] is True
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(exp.EXPECTED_TASK_IDS)
    assert len(comparison["contract_rows"]) == 13
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


def test_roadmap_resolution_uses_only_a_matching_v657_authority(tmp_path: Path) -> None:
    """REQ-REPORT-7503 selects staged V657 before active V657."""

    _markdown, roadmap = _authorities()
    active = tmp_path / exp.ACTIVE_ROADMAP_PATH
    staged = tmp_path / exp.NEXT_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap), encoding="utf-8")

    path, selected, candidates = exp.resolve_v657_roadmap(tmp_path)
    assert path == active
    assert selected["milestone"] == exp.MILESTONE
    assert candidates[0]["exists"] is False

    staged_value = deepcopy(roadmap)
    staged_value["milestone"] = "2026.09.656"
    staged.write_text(yaml.safe_dump(staged_value), encoding="utf-8")
    assert exp.resolve_v657_roadmap(tmp_path)[0] == active

    staged.write_text(yaml.safe_dump(roadmap), encoding="utf-8")
    assert exp.resolve_v657_roadmap(tmp_path)[0] == staged

    active.unlink()
    staged.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="V657 roadmap authority"):
        exp.resolve_v657_roadmap(tmp_path)


def test_v656_custody_preserves_terminal_conductor_and_raw_only_states() -> None:
    """REQ-REPORT-7503; SCENARIO-REPORT-7503-CUSTODY."""

    rows = exp.collect_v656_dispositions(ROOT)
    by_id = {row["task_id"]: row for row in rows}

    assert len(rows) == 14
    assert sum(row["producer_present"] is True for row in rows) == 9
    assert sum(row["producer_present"] is False for row in rows) == 5
    assert all(row["authenticated"] is True for row in rows)
    assert by_id["exp7495-window-calibration"]["conductor_status"] == "FAIL"
    assert by_id["exp7496-causal-update-fixture"]["conductor_status"] == "FAIL"
    assert by_id["exp7497-causal-online-learning"]["conductor_status"] == "GATE_BLOCK"
    assert by_id["exp7501-service-placement"]["conductor_status"] == "GATE_BLOCK"
    assert by_id["exp7495-window-calibration"]["original_honest_verdict"] is None
    assert by_id["exp7497-causal-online-learning"]["original_honest_verdict"] is None

    panel_b = by_id["exp7499-arc-panel-b"]
    assert panel_b["custody_state"] == "raw_only_unpromoted_candidate"
    assert panel_b["completed_raw_episodes"] == 18
    assert panel_b["scientific_result_established"] is False
    assert set(panel_b["raw_custody_hashes"]) == {"session", "checkpoint", "candidate"}
    assert len({row["sha256"] for row in panel_b["raw_custody_hashes"].values()}) == 3


def test_five_methods_write_idempotent_bounded_records(tmp_path: Path) -> None:
    """REQ-REPORT-7503; SCENARIO-REPORT-7503-METHODS."""

    methods = exp.method_rows()
    assert len(methods) == 5
    assert {row["method"] for row in methods} == {
        "binary_scoring_token_expectation",
        "equal_detector_access",
        "aligned_vs_shuffled_information",
        "kan_retention",
        "whole_service_hardware_accounting",
    }
    assert all(
        row["source_section"]
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
    assert note.count("| binary_scoring_token_expectation |") == 1


def test_validation_plans_are_scoped_and_keep_overdue_priority_separate(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7503; SCENARIO-REPORT-7503-VALIDATION."""

    selected, _roadmap, _candidates = exp.resolve_v657_roadmap(ROOT)
    repository = exp.build_repository_check_plan(ROOT, selected)
    assert [row.name for row in repository] == [
        *exp.REQUIRED_REPOSITORY_CHECK_NAMES,
        "overdue_priority",
    ]
    assert all(str(selected) in row.argv for row in repository)

    plan = exp.build_validation_plan(ROOT, tmp_path / "validation")
    assert exp.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(validation_scope.REQUIRED_CHECK_NAMES)
    commands = {row.name: row for row in plan}
    assert "tests/python" not in commands["focused_pytest"].argv
    assert "--no-cov" in commands["focused_pytest"].argv
    assert any(arg.startswith("--basetemp=") for arg in commands["focused_pytest"].argv)
    coverage_env = dict(commands["changed_module_coverage_report"].command_environment)
    assert coverage_env["COVERAGE_FILE"].startswith("/tmp/")


def test_artifact_is_complete_null_advisory_evidence() -> None:
    """REQ-REPORT-7503; SCENARIO-REPORT-7503-VALIDATION."""

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
    assert artifact["honest_verdict"] == "complete_null_v657_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["priority_disposition"]["unresolved_count"] == 3
    assert reduction["contract_ready_score"] == 1
    assert reduction["verdict_class"] == "null"
    assert set(artifact["field_principles"]) == set(artifact)
    assert all(artifact["field_principles"].values())
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert exp.validate_artifact(artifact, root=ROOT) == []


def test_cold_validator_rejects_drift_and_failed_required_validation() -> None:
    """REQ-REPORT-7503 fails closed on rows, hashes, scores, and receipts."""

    artifact = _artifact()
    drifted = deepcopy(artifact)
    drifted["method_rows"][0]["source_section"] = "changed"
    drifted["task_contract_rows"] = []
    drifted["contract_ready_score"] = 0
    drifted["field_principles"].pop("schema")
    drifted["acceptance_gate_results"][0]["principle"] = ""
    assert {
        "method_rows_mismatch",
        "task_contract_rows_mismatch",
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
            "priority_disposition": {},
            "rows": [],
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
        "priority_disposition_mismatch",
        "rows_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
    }.issubset(exp.validate_artifact(defensive, root=ROOT))


def test_defensive_loaders_receipts_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-7503 keeps malformed inputs and date failures explicit."""

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

    good = [_receipt(name) for name in exp.REQUIRED_REPOSITORY_CHECK_NAMES]
    assert exp.receipts_pass(good, exp.REQUIRED_REPOSITORY_CHECK_NAMES) is True
    assert exp.receipts_pass({}, exp.REQUIRED_REPOSITORY_CHECK_NAMES) is False
    assert exp.receipts_recorded(good, exp.REQUIRED_REPOSITORY_CHECK_NAMES) is True
    assert (
        exp.receipts_pass(good + [deepcopy(good[0])], exp.REQUIRED_REPOSITORY_CHECK_NAMES) is False
    )
    assert exp._producer_declarations({"tasks": [None]}) == {}
    assert exp.terminal_state(False, True, False)[2] == "blocked"
    assert exp._hashes_match({"source_artifact_hashes": None}, ROOT) is False
    assert exp._hashes_match({"source_artifact_hashes": {}}, tmp_path) is False
    assert exp.validate_artifact([], root=ROOT) == ["artifact_mapping_required"]

    _markdown, roadmap = _authorities()
    comparison = exp.compare_contract_authorities("not a contract", roadmap)
    assert comparison["passed"] is False
    assert comparison["errors"][0].startswith("parse_error:")

    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260921"])
    assert exp.parse_args(["--date", "20260922"]).date == "20260922"

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(_artifact()), encoding="utf-8")
    args = exp.parse_args(["--date", "20260922", "--validate", str(candidate)])
    assert args.validate == candidate
