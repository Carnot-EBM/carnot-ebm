"""Behavior tests for the V653 contract and method audit.

Spec refs: REQ-REPORT-7447 and SCENARIO-REPORT-7447-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_7447_v653_contract_methods as audit
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one exact-shaped receipt for a pure reducer test."""

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
            "carnot.experiment_7447_v653_contract_methods": str(
                (ROOT / audit.MODULE_PATH).resolve()
            )
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return the affected, repository-guard, and terminal receipts."""

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
    """Return deterministic access outcomes without making network requests."""

    return audit.check_selected_sources(
        lambda source: {
            "access_state": "failed" if source["source_id"] == "crane" else "ok",
            "http_status": 503 if source["source_id"] == "crane" else 200,
            "detail": "fixture",
        }
    )


def _artifact(tmp_path: Path) -> dict[str, Any]:
    """Build a valid measured-shaped artifact from repository evidence."""

    roadmap_path, roadmap, _candidates = audit.resolve_v653_roadmap(ROOT)
    markdown = (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8")
    contract = audit.compare_contract_authorities(markdown, roadmap)
    return audit.build_artifact(
        ROOT,
        roadmap_path,
        contract,
        audit.run_contract_mutation_controls(markdown, roadmap),
        audit.collect_v652_dispositions(ROOT),
        audit.build_v652_evidence_rows(ROOT),
        _source_rows(),
        _validation(),
        started_at_utc="2026-09-20T12:00:00+00:00",
        ended_at_utc="2026-09-20T12:00:01+00:00",
        started_monotonic_ns=1_000_000_000,
        ended_monotonic_ns=2_000_000_000,
        phase_spans=audit.zero_test_phase_spans(),
    )


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-CONTRACT
def test_active_v653_authorities_match_all_fourteen_rows() -> None:
    """The complete Markdown table matches the milestone-selected YAML."""

    path, roadmap, candidates = audit.resolve_v653_roadmap(ROOT)
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


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-CONTRACT
def test_roadmap_resolution_prefers_only_a_matching_next_authority(tmp_path: Path) -> None:
    """A stale next roadmap cannot displace the matching active authority."""

    active = {"milestone": audit.MILESTONE, "tasks": []}
    stale = {"milestone": "2026.09.652", "tasks": []}
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    path, value, candidates = audit.resolve_v653_roadmap(tmp_path)
    assert path == tmp_path / "research-roadmap.yaml"
    assert value == active
    assert candidates[0]["matches_milestone"] is False

    (tmp_path / "research-roadmap-next.yaml").write_text(yaml.safe_dump(active), encoding="utf-8")
    path, _value, _candidates = audit.resolve_v653_roadmap(tmp_path)
    assert path == tmp_path / "research-roadmap-next.yaml"

    (tmp_path / "research-roadmap-next.yaml").unlink()
    (tmp_path / "research-roadmap.yaml").write_text(yaml.safe_dump(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="V653 roadmap authority"):
        audit.resolve_v653_roadmap(tmp_path)


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-CONTRACT
def test_six_private_mutations_fail_in_each_authority() -> None:
    """Count, ID, order, path, field, and milestone drift fail independently."""

    _path, roadmap, _candidates = audit.resolve_v653_roadmap(ROOT)
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


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-CONTRACT
def test_parse_failure_does_not_block_independent_branches() -> None:
    """A broken advisory authority leaves all later schedules independent."""

    comparison = audit.compare_contract_authorities("# missing\n", {"tasks": []})
    assert comparison["passed"] is False
    assert comparison["contract_rows"] == []
    branches = audit.independent_branch_schedule(comparison)
    assert [row["task_id"] for row in branches] == list(audit.EXPECTED_TASK_IDS[1:])
    assert all(row["scheduled_independently"] is True for row in branches)


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-EVIDENCE
def test_v652_dispositions_preserve_classes_flags_and_archive_lag() -> None:
    """All producer rows retain their terminal V652 authority."""

    rows = audit.collect_v652_dispositions(ROOT)
    assert len(rows) == 13
    by_id = {row["task_id"]: row for row in rows}
    assert all(row["authenticated"] is True for row in rows)
    assert by_id["exp7439-certified-decisions"]["original_verdict_class"] == "null"
    assert by_id["exp7441-decision-audit"]["original_verdict_class"] == "disqualified"
    assert by_id["exp7442-span-capture"]["flagged_adversarial"] is True
    assert by_id["exp7443-span-audit"]["flagged_adversarial"] is True
    assert by_id["exp7446-capstone"]["source_kind"] == "current_work"

    archive = audit.completion_archive_state(ROOT)
    assert archive["planning_latest_milestone"] == "2026.09.651"
    assert archive["planning_lag_recorded"] is True
    assert archive["current_latest_milestone"] == "2026.09.652"
    assert archive["history_rewritten"] is False


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-EVIDENCE
def test_six_v652_causes_are_authenticated_and_not_collapsed() -> None:
    """Nulls, evidence defects, runtime faults, exposure, and bounds stay separate."""

    rows = audit.build_v652_evidence_rows(ROOT)
    assert [row["evidence_id"] for row in rows] == [
        "static_null",
        "missing_expert_predictions",
        "parse_status_exception",
        "lost_lease_continuity",
        "arc_62_action_exposure",
        "persistence_bound",
    ]
    assert all(row["authenticated"] is True for row in rows)
    by_id = {row["evidence_id"]: row for row in rows}
    assert by_id["static_null"]["observed"] == "complete_null_no_registered_decision_benefit"
    assert by_id["missing_expert_predictions"]["observed"] == [
        "missing_expert_predictions",
        "weight_update_replay_incomplete",
    ]
    assert by_id["parse_status_exception"]["observed"] == "KeyError:'parse_status'"
    assert by_id["lost_lease_continuity"]["observed"]["lease_id"] is None
    assert by_id["arc_62_action_exposure"]["observed"] == [62, 62]
    assert by_id["persistence_bound"]["observed"] == pytest.approx(0.37696053267914603)
    assert by_id["missing_expert_predictions"]["source_task"] == "exp7441-decision-audit"


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-METHODS
def test_six_sources_map_to_v653_methods_and_faithbench_challenge() -> None:
    """Access failures remain data and every selected method has exact task hooks."""

    sources = _source_rows()
    assert len(sources) == 6
    assert {row["review_status"] for row in sources} == {"new_finding", "rechecked"}
    assert next(row for row in sources if row["source_id"] == "crane")["access_state"] == ("failed")
    methods = audit.method_rows(sources)
    assert [row["method"] for row in methods] == [
        "hidden_representation_probing",
        "cross_block_conditioning",
        "faithbench_challenge_corpus",
        "delayed_expert_losses",
        "compact_output_semantics",
        "on_chip_locality",
    ]
    faithbench = methods[2]
    assert faithbench["task_hooks"] == [
        "exp7449-source-protocol",
        "exp7453-energy-calibration",
        "exp7455-decision-audit",
    ]
    assert faithbench["challenge_revision"] == "cf89797d82812c23b5d5e5c121f1d9b8983bbbce"
    assert faithbench["predictor_exclusions"] == ["detector_predictions", "annotations"]
    assert all(row["primary_url"] and row["evidence_boundary"] for row in methods)


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-AUDITS
def test_repository_guard_plan_uses_selected_authority_without_activation() -> None:
    """The four shipped checks keep their own CLI or implementation semantics."""

    roadmap_path, _roadmap, _candidates = audit.resolve_v653_roadmap(ROOT)
    plan = audit.build_repository_check_plan(ROOT, roadmap_path)
    assert [row.name for row in plan] == list(audit.REPOSITORY_CHECK_NAMES)
    assert all("research-roadmap-next.yaml" not in row.argv for row in plan[:3])
    assert str(roadmap_path) in plan[0].argv[-1]
    assert "scripts/research_conductor.py" not in {path.as_posix() for path in audit.WRITTEN_PATHS}

    obligations = audit.unresolved_obligations(ROOT)
    size = next(row for row in obligations if row["obligation_id"] == "conductor_size_gate")
    assert size["state"] == "current_task_forbidden"
    assert size["prohibited_path"] == "scripts/research_conductor.py"
    assert size["current_task_action"] == "none"


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-ARTIFACT
def test_terminal_artifact_is_complete_advisory_accounting(tmp_path: Path) -> None:
    """A valid contract can be ready without claiming scientific benefit."""

    artifact = _artifact(tmp_path)
    assert artifact["schema"] == audit.SCHEMA
    assert artifact["status"] == "complete_advisory_v653_contract_and_methods"
    assert artifact["honest_verdict"] == "complete_null_v653_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["small_ebm_training"]["performed"] is False
    assert artifact["gate_check_summary"]["passed"] is True
    assert len(artifact["rows"]) == artifact["sample_size_budget"]["completed"]
    assert audit.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-ARTIFACT
def test_blocked_and_disqualified_states_name_exact_causes(tmp_path: Path) -> None:
    """External absence blocks, while current required-check failure disqualifies."""

    artifact = _artifact(tmp_path)
    assert audit._terminal_state(False, True, True)[2] == "blocked"
    assert audit._terminal_state(True, False, True)[2] == "disqualified"
    assert audit._terminal_state(True, True, False)[2] == "disqualified"
    assert audit._terminal_state(True, True, True)[2] == "null"

    failed = deepcopy(artifact)
    failed["preconditions_checked"][0]["passed"] = False
    failed["preconditions_checked"][0]["observed"] = None
    reduction = audit.independent_reduce(failed)
    assert reduction["verdict_class"] == "blocked"
    summary = audit.failure_summary(failed["preconditions_checked"])
    assert summary["first_failure"]["path"]
    assert summary["first_failure"]["expected"] == "readable_nonempty_bytes"
    assert summary["first_failure"]["observed"] is None


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-ARTIFACT
def test_cold_validator_rejects_authority_evidence_and_score_drift(tmp_path: Path) -> None:
    """Conclusion-bearing authority, historical, and score changes fail cold replay."""

    artifact = _artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["contract_ready_score"] = 0
    changed["v652_evidence_rows"][0]["observed"] = "positive"
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    errors = audit.validate_artifact(changed, root=ROOT)
    assert "contract_score_mismatch" in errors
    assert "v652_evidence_mismatch" in errors
    assert "promotion_score_nonzero" in errors

    broken = deepcopy(artifact)
    broken["schema"] = "wrong"
    broken["MODEL_SPECS"] = ["wrong"]
    broken["task_contract_rows"] = []
    broken["contract_comparison"] = {}
    broken["contract_mutation_rows"] = []
    broken["v652_dispositions"] = []
    broken["source_access_rows"] = []
    broken["method_rows"] = []
    broken["completion_archive_state"] = {}
    broken["branch_schedule_rows"] = []
    broken["validation_receipts"] = []
    broken["source_artifact_hashes"] = {}
    broken["field_principles"] = {}
    assert {
        "identity_invalid",
        "model_contract_invalid",
        "task_contract_rows_mismatch",
        "contract_comparison_mismatch",
        "contract_mutations_invalid",
        "v652_dispositions_mismatch",
        "source_access_rows_invalid",
        "method_rows_mismatch",
        "archive_state_mismatch",
        "branch_schedule_mismatch",
        "affected_validation_invalid",
        "repository_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "field_principles_invalid",
    } <= set(audit.validate_artifact(broken, root=ROOT))
    assert audit.validate_artifact([]) == ["artifact_mapping_required"]


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-ARTIFACT
def test_scoped_plan_terminal_readers_and_cli_are_exact(tmp_path: Path) -> None:
    """Validation stays file-scoped and the frozen execution date cannot drift."""

    commands = audit.build_validation_plan(ROOT, tmp_path)
    assert audit.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    terminal = audit._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(audit.TERMINAL_CHECK_NAMES)
    assert audit.parse_args(["--date", audit.RUN_DATE]).date == audit.RUN_DATE
    with pytest.raises(SystemExit):
        audit.parse_args(["--date", "20260919"])


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-ARTIFACT
def test_mapping_readers_and_small_reducers_fail_closed(tmp_path: Path) -> None:
    """Malformed mappings and incomplete receipts do not become valid evidence."""

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

    capstone = {"task_dispositions": "wrong"}
    target = tmp_path / audit.CAPSTONE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(capstone), encoding="utf-8")
    with pytest.raises(ValueError, match="task_dispositions"):
        audit.collect_v652_dispositions(tmp_path)


# REQ-REPORT-7447 / SCENARIO-REPORT-7447-METHODS
def test_method_records_are_idempotent(tmp_path: Path) -> None:
    """The note refreshes while the studying marker appears only once."""

    (tmp_path / audit.STUDY_PATH).write_text("# Studying\n", encoding="utf-8")
    sources = _source_rows()
    audit.write_method_records(tmp_path, sources)
    first = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    audit.write_method_records(tmp_path, sources)
    second = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    assert first == second
    assert first.count("EXP7447-V653-METHOD-INGESTION") == 1
    note = (tmp_path / audit.NOTE_PATH).read_text(encoding="utf-8")
    assert "hidden_representation_probing" in note
    assert "FaithBench" in note
    assert "scripts/research_conductor.py" in note
