"""Behavior tests for the V655 contract and method evidence audit.

Spec refs: REQ-REPORT-7475 and SCENARIO-REPORT-7475-*.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7475_v655_contract_methods as audit
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build the exact command shape consumed by the independent reducer."""

    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {},
        "scope": "test_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7475_v655_contract_methods": str(
                (ROOT / audit.MODULE_PATH).resolve()
            )
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return current affected, repository, and terminal check receipts."""

    names = (*REQUIRED_CHECK_NAMES, *audit.REPOSITORY_CHECK_NAMES, *audit.TERMINAL_CHECK_NAMES)
    return {
        "validation_receipts": [_receipt(name, passed) for name in names],
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _source_rows() -> list[dict[str, Any]]:
    """Return deterministic access receipts without network traffic."""

    return audit.check_selected_sources(
        lambda source: {
            "access_state": "failed" if source["source_id"] == "efficiency" else "ok",
            "http_status": 503 if source["source_id"] == "efficiency" else 200,
            "detail": "fixture",
        }
    )


def _artifact(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a valid artifact-shaped value from current repository evidence."""

    roadmap_path, roadmap, _candidates = audit.resolve_v655_roadmap(ROOT)
    markdown = (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8")
    contract = audit.compare_contract_authorities(markdown, roadmap)
    return audit.build_artifact(
        ROOT,
        roadmap_path,
        contract,
        audit.run_contract_mutation_controls(markdown, roadmap),
        audit.collect_v654_dispositions(ROOT),
        _source_rows(),
        validation or _validation(),
        started_at_utc="2026-09-21T12:00:00+00:00",
        ended_at_utc="2026-09-21T12:00:01+00:00",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        phase_spans=audit.zero_test_phase_spans(),
    )


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-CONTRACT
def test_active_v655_authorities_match_all_fourteen_complete_rows() -> None:
    """The selected active YAML matches every field in the Markdown table."""

    path, roadmap, candidates = audit.resolve_v655_roadmap(ROOT)
    comparison = audit.load_contract_audit(ROOT)
    assert path == ROOT / "research-roadmap.yaml"
    assert roadmap["milestone"] == audit.MILESTONE
    assert candidates[0]["path"] == "research-roadmap-next.yaml"
    assert candidates[0]["observed"] is None
    assert comparison["passed"] is True
    assert comparison["markdown_milestone"] == audit.MILESTONE
    assert comparison["yaml_milestone"] == audit.MILESTONE
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(audit.EXPECTED_TASK_IDS)
    assert len(comparison["contract_rows"]) == 14
    assert all(row["passed"] for row in comparison["contract_rows"])
    assert all(
        set(row["checks"])
        == {"order", "id", "title", "deliverable", "phase", "substrate", "gates", "producer_fields"}
        for row in comparison["contract_rows"]
    )


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-CONTRACT
def test_roadmap_resolution_accepts_only_matching_next_then_active(tmp_path: Path) -> None:
    """A stale staged roadmap cannot displace the matching active authority."""

    active = {"milestone": audit.MILESTONE, "tasks": []}
    stale = {"milestone": "2026.09.654", "tasks": []}
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    path, value, candidates = audit.resolve_v655_roadmap(tmp_path)
    assert path == tmp_path / "research-roadmap.yaml"
    assert value == active
    assert candidates[0]["matches_milestone"] is False

    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    assert audit.resolve_v655_roadmap(tmp_path)[0] == tmp_path / "research-roadmap-next.yaml"

    (tmp_path / "research-roadmap-next.yaml").unlink()
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="V655 roadmap authority"):
        audit.resolve_v655_roadmap(tmp_path)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-CONTRACT
def test_private_mutations_reject_each_authority_and_keep_peer_readable() -> None:
    """Count, order, identity, field, path, and milestone drift all fail."""

    _path, roadmap, _candidates = audit.resolve_v655_roadmap(ROOT)
    markdown = (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8")
    rows = audit.run_contract_mutation_controls(markdown, roadmap)
    assert len(rows) == 12
    assert {row["authority"] for row in rows} == {"markdown", "yaml"}
    assert {row["mutation"] for row in rows} == {
        "count",
        "id",
        "order",
        "path",
        "field",
        "milestone",
    }
    assert all(row["rejected"] is True for row in rows)
    assert all(row["other_authority_readable"] is True for row in rows)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-CONTRACT
def test_advisory_contract_never_gates_independent_science() -> None:
    """A broken contract audit leaves every later task independently runnable."""

    comparison = audit.compare_contract_authorities("# missing\n", {"tasks": []})
    rows = audit.independent_branch_schedule(comparison)
    assert comparison["passed"] is False
    assert [row["task_id"] for row in rows] == list(audit.EXPECTED_TASK_IDS[1:])
    assert all(row["scheduled_independently"] is True for row in rows)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-EVIDENCE
def test_v654_dispositions_preserve_pregate_absence_flags_and_retirement() -> None:
    """All historical outcomes reproduce without promoting missing evidence."""

    rows = audit.collect_v654_dispositions(ROOT)
    assert len(rows) == 14
    by_id = {row["task_id"]: row for row in rows}
    assert all(row["authenticated"] is True for row in rows)
    assert sum(row["producer_present"] for row in rows) == 10
    assert by_id["exp7462-option-protocol"]["original_verdict_class"] == "disqualified"
    assert by_id["exp7462-option-protocol"]["flagged_adversarial"] is True
    assert by_id["exp7465-source-option-capture"]["observed_path"] == (
        "results/experiment_7465_source_option_capture.json"
    )
    assert by_id["exp7465-source-option-capture"]["source_kind"] == "conductor_pre_gate"
    assert by_id["exp7466-typed-energy-calibration"]["producer_present"] is False
    assert by_id["exp7467-factual-span-canary"]["extraction_retired"] is True
    assert by_id["exp7468-residual-learner"]["original_verdict_class"] == "circular_positive"
    assert by_id["exp7473-board-continuity"]["validation_failures"] == ["validation_log_missing:12"]
    assert by_id["exp7474-capstone"]["original_verdict_class"] == "disqualified"


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-METHODS
def test_six_methods_map_revision_component_failure_and_tasks() -> None:
    """The bounded source map retains the KAN-CL component-study limit."""

    sources = _source_rows()
    methods = audit.method_rows(sources)
    assert len(sources) == len(methods) == 6
    assert [row["method"] for row in methods] == [
        "semif_native_readout",
        "kan_cl_importance_anchor",
        "support_overlap_limit",
        "budgeted_feedback",
        "efficiency_claim_controls",
        "hardware_locality",
    ]
    assert all(row["source_revision"] for row in methods)
    assert all(row["usable_component"] and row["failure_mode"] for row in methods)
    assert all(row["task_mapping"] for row in methods)
    kan = methods[1]
    assert kan["study_scope"] == "head_component_only"
    assert kan["full_cnn_backbone_reproduction"] is False
    assert (
        next(row for row in sources if row["source_id"] == "efficiency")["access_state"] == "failed"
    )


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-PRIORITIES
def test_repository_checks_priority_map_and_deferred_obligations_are_exact() -> None:
    """Five unchanged guards run while E6, calibration, and FR-11 stay mapped."""

    roadmap_path, _roadmap, _candidates = audit.resolve_v655_roadmap(ROOT)
    plan = audit.build_repository_check_plan(ROOT, roadmap_path)
    assert [row.name for row in plan] == list(audit.REPOSITORY_CHECK_NAMES)
    assert str(roadmap_path) in plan[0].argv[-1]
    assert "scripts/research_conductor.py" not in {path.as_posix() for path in audit.WRITTEN_PATHS}
    mappings = audit.priority_mappings()
    assert mappings == [
        {"priority": "live_e6_follow_up", "task_ids": ["exp7478", "exp7485", "exp7486"]},
        {"priority": "calibration", "task_ids": ["exp7481"]},
        {"priority": "fr_11", "task_ids": ["exp7482", "exp7483"]},
    ]
    obligations = {row["obligation_id"]: row for row in audit.unresolved_obligations(ROOT)}
    assert obligations["scored_blackwell_runtime"]["state"] == "deferred_inaccessible"
    assert obligations["conductor_change"]["state"] == "user_forbidden"
    assert obligations["conductor_change"]["prohibited_path"] == ("scripts/research_conductor.py")


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-ARTIFACT
def test_terminal_artifact_is_complete_null_advisory_accounting() -> None:
    """A valid contract is ready with zero current model or training work."""

    artifact = _artifact()
    assert artifact["schema"] == audit.SCHEMA
    assert artifact["status"] == "complete_advisory_v655_contract_and_methods"
    assert artifact["honest_verdict"] == "complete_null_v655_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["small_ebm_training"]["performed"] is False
    assert artifact["gate_check_summary"]["passed"] is True
    assert len(artifact["rows"]) == artifact["sample_size_budget"]["completed"]
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"])
    assert audit.validate_artifact(artifact, root=ROOT) == []

    failed_validation = _validation()
    overdue = next(
        row for row in failed_validation["validation_receipts"] if row["name"] == "overdue_priority"
    )
    overdue.update(exit_code=1, passed=False, output_tail="active priority missing")
    failed_validation["repository_checks_passed"] = False
    disqualified = _artifact(failed_validation)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["contract_ready_score"] == 0
    assert audit.validate_artifact(disqualified, root=ROOT) == []


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-ARTIFACT
def test_cold_validator_rejects_historical_method_score_and_hash_drift() -> None:
    """Conclusion-bearing evidence and result changes fail independent replay."""

    artifact = _artifact()
    changed = deepcopy(artifact)
    changed["contract_ready_score"] = 0
    changed["v654_dispositions"][1]["original_verdict_class"] = "positive"
    changed["method_rows"][1]["study_scope"] = "full_system"
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    errors = audit.validate_artifact(changed, root=ROOT)
    assert "contract_score_mismatch" in errors
    assert "v654_dispositions_mismatch" in errors
    assert "method_rows_mismatch" in errors

    broken = deepcopy(artifact)
    broken["schema"] = "wrong"
    broken["MODEL_SPECS"] = ["wrong"]
    broken["model_specs"] = ["wrong"]
    broken["task_contract_rows"] = []
    broken["contract_comparison"] = {}
    broken["contract_mutation_rows"] = []
    broken["source_access_rows"] = []
    broken["branch_schedule_rows"] = []
    broken["validation_receipts"] = []
    broken["source_artifact_hashes"] = {}
    broken["field_principles"] = {}
    broken["promotion_score"] = 1
    assert {
        "identity_invalid",
        "model_contract_invalid",
        "task_contract_rows_mismatch",
        "contract_comparison_mismatch",
        "contract_mutations_invalid",
        "source_access_rows_invalid",
        "branch_schedule_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "promotion_score_nonzero",
        "field_principles_invalid",
        "reproducibility_checksum_mismatch",
    }.issubset(audit.validate_artifact(broken, root=ROOT))


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-ARTIFACT
def test_validation_plan_is_scoped_and_date_is_frozen(tmp_path: Path) -> None:
    """The affected runner has exact files, private temp state, and a fixed date."""

    private = tmp_path / "private"
    private.mkdir()
    plan = audit.build_validation_plan(ROOT, private)
    assert audit.validate_validation_plan(ROOT, plan) == []
    assert [row.name for row in plan] == list(REQUIRED_CHECK_NAMES)
    assert audit.date_argument("20260921") == "20260921"
    with pytest.raises(argparse.ArgumentTypeError, match="run date"):
        audit.date_argument("20260920")

    widened = list(plan)
    widened[1] = type(plan[1])(
        name=plan[1].name,
        argv=tuple(
            "tests/python" if value == audit.TEST_PATH.as_posix() else value
            for value in plan[1].argv
        ),
        scope=plan[1].scope,
        timeout_s=plan[1].timeout_s,
        command_environment=plan[1].command_environment,
    )
    assert audit.validate_validation_plan(ROOT, widened)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-CONTRACT
def test_mapping_loaders_and_capstone_shape_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed authority and disposition shapes cannot enter reduction."""

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[not, a, mapping]", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        audit.load_yaml(bad_yaml)
    bad_yaml.write_text("key: [", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        audit.load_yaml(bad_yaml)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON mapping"):
        audit.load_json(bad_json)
    assert audit._producer_declarations({"tasks": [None]}) == {}

    monkeypatch.setattr(audit, "load_json", lambda _path: {"task_dispositions": []})
    with pytest.raises(ValueError, match="fourteen"):
        audit.collect_v654_dispositions(tmp_path)
    wrong = [{"task_id": f"exp{index}"} for index in range(14)]
    monkeypatch.setattr(audit, "load_json", lambda _path: {"task_dispositions": wrong})
    with pytest.raises(ValueError, match="order changed"):
        audit.collect_v654_dispositions(tmp_path)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-ARTIFACT
def test_defensive_reducers_and_cold_reader_errors_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad receipts, absent hashes, and malformed artifacts return named errors."""

    assert audit.receipts_pass({}, REQUIRED_CHECK_NAMES) is False
    assert audit.receipts_recorded({}, REQUIRED_CHECK_NAMES) is False
    assert audit.receipts_recorded(_validation()["validation_receipts"], REQUIRED_CHECK_NAMES)
    assert audit._terminal_state(False, True, True)[2] == "blocked"
    assert audit._terminal_state(True, False, True)[2] == "disqualified"
    assert audit._hashes_match({}, ROOT) is False
    assert audit._hashes_match({"source_artifact_hashes": {}}, Path("/missing")) is False
    assert audit.validate_artifact([]) == ["artifact_mapping_required"]

    artifact = _artifact()
    artifact["priority_mappings"] = []
    artifact["unresolved_obligations"] = []
    artifact["acceptance_gate_results"] = []
    artifact["reproducibility_checksum"] = audit.reproducibility_checksum(artifact)
    errors = audit.validate_artifact(artifact, root=ROOT)
    assert "priority_mappings_mismatch" in errors
    assert "unresolved_obligations_mismatch" in errors
    assert "gate_principles_invalid" in errors

    original_resolve = audit.resolve_v655_roadmap
    monkeypatch.setattr(
        audit,
        "resolve_v655_roadmap",
        lambda _root: (_ for _ in ()).throw(ValueError("fixture")),
    )
    assert "contract_comparison_mismatch" in audit.validate_artifact(artifact, root=ROOT)
    monkeypatch.setattr(audit, "resolve_v655_roadmap", original_resolve)


# REQ-REPORT-7475 / SCENARIO-REPORT-7475-ARTIFACT
def test_terminal_plan_docs_and_cli_are_exact(tmp_path: Path) -> None:
    """Terminal readers, durable method records, and CLI parsing keep exact scope."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({}), encoding="utf-8")
    plan = audit._terminal_commands(ROOT, candidate)
    assert [row.spec.name for row in plan] == list(audit.TERMINAL_CHECK_NAMES)
    assert all(row.required and row.category == "required_validation" for row in plan)

    (tmp_path / audit.STUDY_PATH).write_text("# Study ledger\n", encoding="utf-8")
    rows = _source_rows()
    audit.write_method_records(tmp_path, rows)
    first = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    audit.write_method_records(tmp_path, rows)
    assert (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8") == first
    note = (tmp_path / audit.NOTE_PATH).read_text(encoding="utf-8")
    assert "head-component study" in note
    assert "full CNN and backbone system" in note

    args = audit.parse_args(["--date", "20260921", "--validate", str(candidate)])
    assert args.date == "20260921"
    assert args.validate == candidate
