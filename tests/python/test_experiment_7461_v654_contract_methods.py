"""Behavior tests for the V654 contract and method audit.

Spec refs: REQ-REPORT-7461 and SCENARIO-REPORT-7461-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7461_v654_contract_methods as audit
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one command receipt with the fields used by the cold reducer."""

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
            "carnot.experiment_7461_v654_contract_methods": str(
                (ROOT / audit.MODULE_PATH).resolve()
            )
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return exact affected, repository, and terminal command receipts."""

    names = (
        *REQUIRED_CHECK_NAMES,
        *audit.REPOSITORY_CHECK_NAMES,
        *audit.TERMINAL_CHECK_NAMES,
    )
    return {
        "validation_receipts": [_receipt(name, passed) for name in names],
        "affected_checks_passed": passed,
        "repository_checks_passed": passed,
        "terminal_checks_passed": passed,
    }


def _source_rows() -> list[dict[str, Any]]:
    """Make deterministic access receipts without making network requests."""

    return audit.check_selected_sources(
        lambda source: {
            "access_state": "failed" if source["source_id"] == "recap" else "ok",
            "http_status": 503 if source["source_id"] == "recap" else 200,
            "detail": "fixture",
        }
    )


def _artifact() -> dict[str, Any]:
    """Build a valid measured-shaped artifact from current repository evidence."""

    roadmap_path, roadmap, _candidates = audit.resolve_v654_roadmap(ROOT)
    markdown = (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8")
    contract = audit.compare_contract_authorities(markdown, roadmap)
    return audit.build_artifact(
        ROOT,
        roadmap_path,
        contract,
        audit.run_contract_mutation_controls(markdown, roadmap),
        audit.collect_v653_dispositions(ROOT),
        _source_rows(),
        _validation(),
        started_at_utc="2026-09-20T12:00:00+00:00",
        ended_at_utc="2026-09-20T12:00:01+00:00",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        phase_spans=audit.zero_test_phase_spans(),
    )


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-CONTRACT
def test_active_v654_authorities_match_all_fourteen_rows() -> None:
    """The complete Markdown table matches the milestone-selected YAML."""

    path, roadmap, candidates = audit.resolve_v654_roadmap(ROOT)
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


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-CONTRACT
def test_roadmap_resolution_prefers_only_matching_next(tmp_path: Path) -> None:
    """A stale staged roadmap cannot displace the matching active authority."""

    active = {"milestone": audit.MILESTONE, "tasks": []}
    stale = {"milestone": "2026.09.653", "tasks": []}
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    path, value, candidates = audit.resolve_v654_roadmap(tmp_path)
    assert path == tmp_path / "research-roadmap.yaml"
    assert value == active
    assert candidates[0]["matches_milestone"] is False

    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    assert audit.resolve_v654_roadmap(tmp_path)[0] == tmp_path / "research-roadmap-next.yaml"

    (tmp_path / "research-roadmap-next.yaml").unlink()
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="V654 roadmap authority"):
        audit.resolve_v654_roadmap(tmp_path)


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-CONTRACT
def test_private_mutations_reject_count_id_order_path_field_and_milestone() -> None:
    """Each required drift fails in each authority without damaging its peer."""

    _path, roadmap, _candidates = audit.resolve_v654_roadmap(ROOT)
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


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-CONTRACT
def test_contract_failure_does_not_gate_independent_science() -> None:
    """A broken advisory authority leaves all thirteen later tasks independent."""

    comparison = audit.compare_contract_authorities("# missing\n", {"tasks": []})
    rows = audit.independent_branch_schedule(comparison)
    assert comparison["passed"] is False
    assert [row["task_id"] for row in rows] == list(audit.EXPECTED_TASK_IDS[1:])
    assert all(row["scheduled_independently"] is True for row in rows)


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-EVIDENCE
def test_v653_dispositions_keep_actual_pregate_and_validation_failures() -> None:
    """Exact bytes retain every measured null, pre-gate path, and audit defect."""

    rows = audit.collect_v653_dispositions(ROOT)
    assert len(rows) == 14
    by_id = {row["task_id"]: row for row in rows}
    assert all(row["authenticated"] is True for row in rows)
    assert by_id["exp7452-source-embeddings"]["original_verdict_class"] == "null"
    assert by_id["exp7453-energy-calibration"]["source_kind"] == "structured_pre_gate"
    assert by_id["exp7453-energy-calibration"]["observed_path"] == (
        "results/experiment_7453_energy_calibration.json"
    )
    assert by_id["exp7454-continuous-learning"]["mixture_construction_retired"] is True
    audit_row = by_id["exp7455-decision-audit"]
    assert audit_row["original_verdict_class"] == "disqualified"
    assert {row["name"] for row in audit_row["failed_validation_receipts"]} == {
        "changed_module_coverage_report",
        "adversarial_verify",
    }
    assert "711      2    99%" in audit_row["failed_validation_receipts"][0]["output_tail"]
    assert any(
        "METHODOLOGY_MISSING" in row["output_tail"]
        for row in audit_row["failed_validation_receipts"]
    )
    assert by_id["exp7460-capstone"]["flagged_adversarial"] is True

    archive = audit.completion_archive_state(ROOT)
    assert archive["planning_latest_milestone"] == "2026.09.652"
    assert archive["current_latest_milestone"] == "2026.09.653"
    assert archive["archive_lag_milestones"] == 0
    assert archive["history_rewritten"] is False


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-METHODS
def test_six_sources_map_to_v654_methods_with_correct_jev_attribution() -> None:
    """Each source has a revision, usable mechanism, and bounded claim."""

    sources = _source_rows()
    methods = audit.method_rows(sources)
    assert len(sources) == len(methods) == 6
    assert [row["method"] for row in methods] == [
        "native_option_readout",
        "support_overlap_diagnostic",
        "feedback_budget_accounting",
        "retained_domain_regression",
        "on_chip_update_locality",
        "categorical_energy_interface",
    ]
    assert all(row["source_revision"] for row in methods)
    assert all(row["usable_mechanism"] and row["claim_boundary"] for row in methods)
    semif = methods[0]
    assert semif["jevbench_reversal_owner"] == "open-alternative-jev"
    assert semif["jevbench_reversal_owner"] != "SemIf"
    assert semif["source_revision"] == "ca3ba65f142967030ecb453346e94d6f476a69df"
    assert next(row for row in sources if row["source_id"] == "recap")["access_state"] == ("failed")


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-AUDITS
def test_repository_checks_priorities_and_forbidden_work_are_exact() -> None:
    """Five shipped guards run read-only and E0/E6 map to their declared tasks."""

    roadmap_path, _roadmap, _candidates = audit.resolve_v654_roadmap(ROOT)
    plan = audit.build_repository_check_plan(ROOT, roadmap_path)
    assert [row.name for row in plan] == list(audit.REPOSITORY_CHECK_NAMES)
    assert str(roadmap_path) in plan[0].argv[-1]
    assert "scripts/research_conductor.py" not in {path.as_posix() for path in audit.WRITTEN_PATHS}
    assert audit.priority_mappings() == [
        {
            "priority": "2026-09-20 SEMIF E0",
            "task_id": "exp7463-semif-e0-logprob-parity",
            "mapped": True,
        },
        {
            "priority": "2026-09-20 SEMIF E6",
            "task_id": "exp7464-semif-e6-decision-cost-profile",
            "mapped": True,
        },
    ]
    size = next(
        row
        for row in audit.unresolved_obligations(ROOT)
        if row["obligation_id"] == "conductor_artifact_size"
    )
    assert size["state"] == "outside_user_authorized_edits"
    assert size["prohibited_path"] == "scripts/research_conductor.py"


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-ARTIFACT
def test_terminal_artifact_is_complete_advisory_accounting() -> None:
    """A valid contract is ready without model work or positive science."""

    artifact = _artifact()
    assert artifact["schema"] == audit.SCHEMA
    assert artifact["status"] == "complete_advisory_v654_contract_and_methods"
    assert artifact["honest_verdict"] == "complete_null_v654_contract_methods_ingested"
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
    assert audit.validate_artifact(artifact, root=ROOT) == []
    assert audit._terminal_state(False, True, True)[2] == "blocked"
    assert audit._terminal_state(True, False, True)[2] == "disqualified"
    assert audit._terminal_state(True, True, False)[2] == "disqualified"


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-ARTIFACT
def test_cold_validator_rejects_evidence_score_and_method_drift() -> None:
    """Conclusion-bearing historical, method, and score changes fail replay."""

    artifact = _artifact()
    changed = deepcopy(artifact)
    changed["contract_ready_score"] = 0
    changed["v653_dispositions"][0]["original_verdict_class"] = "positive"
    changed["method_rows"][0]["jevbench_reversal_owner"] = "SemIf"
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    errors = audit.validate_artifact(changed, root=ROOT)
    assert "contract_score_mismatch" in errors
    assert "v653_dispositions_mismatch" in errors
    assert "method_rows_mismatch" in errors

    broken = deepcopy(artifact)
    broken["schema"] = "wrong"
    broken["MODEL_SPECS"] = ["wrong"]
    broken["model_specs"] = ["wrong"]
    broken["task_contract_rows"] = []
    broken["contract_comparison"] = {}
    broken["contract_mutation_rows"] = []
    broken["source_access_rows"] = []
    broken["completion_archive_state"] = {}
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
        "archive_state_mismatch",
        "branch_schedule_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "promotion_score_nonzero",
        "field_principles_invalid",
    } <= set(audit.validate_artifact(broken, root=ROOT))
    assert audit.validate_artifact([]) == ["artifact_mapping_required"]


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-ARTIFACT
def test_scoped_plan_terminal_readers_and_cli_are_exact(tmp_path: Path) -> None:
    """Validation stays file-scoped and the frozen date cannot drift."""

    commands = audit.build_validation_plan(ROOT, tmp_path)
    assert audit.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    terminal = audit._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(audit.TERMINAL_CHECK_NAMES)
    assert audit.parse_args(["--date", audit.RUN_DATE]).date == audit.RUN_DATE
    with pytest.raises(SystemExit):
        audit.parse_args(["--date", "20260919"])


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-METHODS
def test_method_records_are_idempotent_and_correct_attribution(tmp_path: Path) -> None:
    """The note refreshes while the studying marker appears only once."""

    (tmp_path / audit.STUDY_PATH).write_text("# Studying\n", encoding="utf-8")
    sources = _source_rows()
    audit.write_method_records(tmp_path, sources)
    first = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    audit.write_method_records(tmp_path, sources)
    second = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    assert first == second
    assert first.count("EXP7461-V654-METHOD-INGESTION") == 1
    note = (tmp_path / audit.NOTE_PATH).read_text(encoding="utf-8")
    assert "open-alternative-jev" in note
    assert "SemIf owns the 72-to-21" not in note
    assert "ARM–EBM" in note


# REQ-REPORT-7461 / SCENARIO-REPORT-7461-ARTIFACT
def test_mapping_readers_and_small_reducers_fail_closed(tmp_path: Path) -> None:
    """Malformed mappings and incomplete receipts cannot become evidence."""

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        audit.load_yaml(sequence)
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("key: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        audit.load_yaml(malformed)
    sequence_json = tmp_path / "sequence.json"
    sequence_json.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        audit.load_json(sequence_json)
    assert audit.receipts_pass({}, ["missing"]) is False
    assert audit._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    missing_roadmap = tmp_path / "missing-roadmap.yaml"
    assert audit._source_hashes(tmp_path, missing_roadmap, [{"observed_path": ""}]) == {}
    _path, roadmap, _candidates = audit.resolve_v654_roadmap(ROOT)
    assert audit._mutate_yaml(roadmap, "unknown") == roadmap

    target = tmp_path / audit.CAPSTONE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"task_dispositions": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="task_dispositions"):
        audit.collect_v653_dispositions(tmp_path)
