"""Behavior tests for the V652 contract and method audit.

Spec refs: REQ-REPORT-7434 and SCENARIO-REPORT-7434-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7434_v652_contract_methods as audit
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one scoped receipt without replacing the production runner."""

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
            "carnot.experiment_7434_v652_contract_methods": str(
                (ROOT / audit.MODULE_PATH).resolve()
            )
        }
    return row


def _validation(passed: bool = True) -> dict[str, Any]:
    """Return exact-shaped affected and terminal validation receipts."""

    return {
        "validation_receipts": [
            _receipt(name, passed) for name in (*REQUIRED_CHECK_NAMES, *audit.TERMINAL_CHECK_NAMES)
        ],
        "required_checks_passed": passed,
        "terminal_validation_passed": passed,
    }


def _source_rows() -> list[dict[str, Any]]:
    """Return bounded deterministic access outcomes for reducer tests."""

    return audit.check_selected_sources(
        lambda source: {
            "access_state": "failed" if source["source_id"] == "kan_forgetting" else "ok",
            "http_status": 503 if source["source_id"] == "kan_forgetting" else 200,
            "detail": "fixture",
        }
    )


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-CONTRACT
def test_real_v652_authorities_match_all_thirteen_rows() -> None:
    """Both live authorities agree on every field in conductor order."""

    comparison = audit.load_contract_audit(ROOT)
    assert comparison["passed"] is True
    assert comparison["markdown_milestone"] == audit.MILESTONE
    assert comparison["yaml_milestone"] == audit.MILESTONE
    assert [row["unit_id"] for row in comparison["contract_rows"]] == list(audit.EXPECTED_TASK_IDS)
    assert len(comparison["contract_rows"]) == 13
    assert all(row["passed"] for row in comparison["contract_rows"])


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-CONTRACT
def test_each_contract_authority_is_mutated_independently() -> None:
    """Every contract field fails closed in Markdown and YAML separately."""

    roadmap = audit.load_yaml(ROOT / audit.ROADMAP_PATH)
    markdown = (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8")
    rows = audit.run_contract_mutation_controls(markdown, roadmap)
    expected_mutations = {
        "milestone",
        "count",
        "order",
        "title",
        "deliverable",
        "phase",
        "substrate",
        "gates",
    }
    assert len(rows) == 16
    assert {row["authority"] for row in rows} == {"markdown", "yaml"}
    assert {row["mutation"] for row in rows} == expected_mutations
    assert all(row["rejected"] is True for row in rows)
    assert all(row["other_authority_readable"] is True for row in rows)


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-CONTRACT
def test_contract_parse_errors_and_branch_scheduling_stay_separate() -> None:
    """A broken audit authority does not cascade-block independent tasks."""

    result = audit.compare_contract_authorities("# no table\n", {"tasks": []})
    assert result["passed"] is False
    assert result["contract_rows"] == []
    rows = audit.independent_branch_schedule(result)
    assert [row["task_id"] for row in rows] == list(audit.EXPECTED_TASK_IDS[1:])
    assert all(row["scheduled_independently"] is True for row in rows)
    assert all(row["contract_failure_effect"] == "none" for row in rows)


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-GATES
def test_real_conductor_gate_reader_covers_all_failure_shapes(tmp_path: Path) -> None:
    """Reader outcomes and evidence admissibility remain separate."""

    rows = audit.run_conductor_gate_controls(tmp_path)
    assert [row["case"] for row in rows] == [
        "passing",
        "zero",
        "missing_file",
        "missing_field",
        "none",
        "wrong_type",
        "disqualified_class",
        "flagged_evidence",
    ]
    by_case = {row["case"]: row for row in rows}
    assert by_case["passing"]["reader_passed"] is True
    assert by_case["passing"]["admissible"] is True
    assert by_case["zero"]["observed"] == 0
    assert by_case["missing_file"]["artifact_path"] is None
    assert "NO field" in by_case["missing_field"]["reason"]
    assert "as null" in by_case["none"]["reason"]
    assert "not comparable" in by_case["wrong_type"]["reason"]
    assert by_case["disqualified_class"]["observed"] == "disqualified"
    assert by_case["flagged_evidence"]["reader_passed"] is True
    assert by_case["flagged_evidence"]["admissible"] is False
    assert sum(row["admissible"] for row in rows) == 1


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-DISPOSITIONS
def test_v651_dispositions_preserve_pregate_and_quarantine_boundaries() -> None:
    """The thirteen old outcomes retain their exact paths and flags."""

    rows = audit.collect_prior_dispositions(ROOT)
    assert len(rows) == 13
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7421-contract-ingestion"]["original_verdict_class"] == "disqualified"
    assert by_id["exp7425-spline-prototype"]["original_verdict_class"] == ("circular_positive")
    assert by_id["exp7429-anchored-capture"]["flagged_adversarial"] is True
    blocked = by_id["exp7430-extraction-audit"]
    assert blocked["declared_path"] == "results/experiment_7430_v651_extraction_audit.json"
    assert blocked["declared_path_exists"] is False
    assert blocked["observed_path"] == "results/experiment_7430_extraction_audit.json"
    assert blocked["observed_path_exists"] is True
    assert blocked["source_kind"] == "structured_pre_gate"
    assert blocked["original_verdict_class"] == "blocked"
    assert by_id["exp7433-capstone"]["flagged_adversarial"] is True

    archive = audit.completion_archive_state(ROOT)
    assert archive["planning_latest_milestone"] == "2026.09.650"
    assert archive["planning_lag_recorded"] is True
    assert archive["current_latest_milestone"] == "2026.09.651"
    assert archive["history_rewritten"] is False


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-METHODS
def test_six_sources_map_to_three_bounded_methods_and_deferrals() -> None:
    """Access failure stays visible while methods map to exact V652 tasks."""

    sources = _source_rows()
    assert len(sources) == 6
    assert (
        next(row for row in sources if row["source_id"] == "kan_forgetting")["access_state"]
        == "failed"
    )
    methods = audit.method_mapping_rows(sources)
    assert [row["method"] for row in methods] == [
        "joint_certification",
        "expert_aggregation",
        "output_representation",
    ]
    assert methods[0]["task_hooks"] == [
        "exp7436-selection-protocol",
        "exp7439-certified-decisions",
        "exp7441-decision-audit",
    ]
    assert methods[1]["task_hooks"] == [
        "exp7438-mixture-prototype",
        "exp7440-mixture-learning",
        "exp7441-decision-audit",
    ]
    assert methods[2]["task_hooks"] == [
        "exp7437-span-protocol",
        "exp7442-span-capture",
        "exp7443-span-audit",
    ]
    assert all(row["assumptions"] and row["controls"] and row["deferrals"] for row in methods)

    priorities = audit.mandatory_priority_rows(ROOT)
    assert [row["date"] for row in priorities] == ["2026-09-19", "2026-09-18"]
    assert priorities[0]["owner"] == "exp7435-round-breaker"
    assert priorities[1]["state"] == "deferred_by_explicit_file_prohibition"
    assert priorities[1]["prohibited_path"] == "scripts/research_conductor.py"


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-ARTIFACT
def test_terminal_artifact_is_complete_advisory_accounting(tmp_path: Path) -> None:
    """Exact authorities yield readiness without claiming scientific benefit."""

    artifact = audit.build_artifact(
        ROOT,
        audit.load_contract_audit(ROOT),
        audit.run_contract_mutation_controls(
            (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8"),
            audit.load_yaml(ROOT / audit.ROADMAP_PATH),
        ),
        audit.run_conductor_gate_controls(tmp_path / "gates"),
        audit.collect_prior_dispositions(ROOT),
        _source_rows(),
        _validation(),
        started_at_utc="2026-09-19T18:00:00+00:00",
        ended_at_utc="2026-09-19T18:00:01+00:00",
        duration_s=1.0,
        phase_spans=audit.zero_test_phase_spans(),
    )
    assert artifact["schema"] == audit.SCHEMA
    assert artifact["status"] == "complete_advisory_contract_and_methods"
    assert artifact["honest_verdict"] == "complete_null_exact_contract_methods_ingested"
    assert artifact["verdict_class"] == "null"
    assert artifact["contract_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_exact_declared_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["small_ebm_training"]["performed"] is False
    assert artifact["gate_check_summary"]["passed"] is True
    assert len(artifact["rows"]) == artifact["sample_size_budget"]["completed"]
    assert audit.validate_artifact(artifact, root=ROOT) == []


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-ARTIFACT
def test_cold_validator_rejects_authority_disposition_and_score_drift(tmp_path: Path) -> None:
    """Independent reduction rejects changes to conclusion-bearing fields."""

    artifact = audit.build_artifact(
        ROOT,
        audit.load_contract_audit(ROOT),
        audit.run_contract_mutation_controls(
            (ROOT / audit.DESIGN_PATH).read_text(encoding="utf-8"),
            audit.load_yaml(ROOT / audit.ROADMAP_PATH),
        ),
        audit.run_conductor_gate_controls(tmp_path / "gates"),
        audit.collect_prior_dispositions(ROOT),
        _source_rows(),
        _validation(),
        started_at_utc="2026-09-19T18:00:00+00:00",
        ended_at_utc="2026-09-19T18:00:01+00:00",
        duration_s=1.0,
        phase_spans=audit.zero_test_phase_spans(),
    )

    changed = deepcopy(artifact)
    changed["contract_ready_score"] = 0
    changed["prior_dispositions"][0]["original_verdict_class"] = "positive"
    changed["promotion_score"] = 1
    changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
    errors = audit.validate_artifact(changed, root=ROOT)
    assert "contract_score_mismatch" in errors
    assert "prior_dispositions_mismatch" in errors
    assert "promotion_score_nonzero" in errors

    broken = deepcopy(artifact)
    broken["schema"] = "wrong"
    broken["MODEL_SPECS"] = ["wrong"]
    broken["task_contract_rows"] = []
    broken["contract_comparison"] = {}
    broken["contract_mutation_rows"] = []
    broken["conductor_gate_control_rows"] = []
    broken["source_access_rows"] = []
    broken["method_mapping_rows"] = []
    broken["mandatory_priority_rows"] = []
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
        "gate_controls_invalid",
        "source_access_rows_invalid",
        "method_mapping_rows_mismatch",
        "priority_rows_mismatch",
        "completion_archive_mismatch",
        "branch_schedule_mismatch",
        "affected_validation_invalid",
        "terminal_validation_invalid",
        "source_hash_mismatch",
        "field_principles_invalid",
    } <= set(audit.validate_artifact(broken, root=ROOT))
    assert audit.validate_artifact([]) == ["artifact_mapping_required"]


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-ARTIFACT
def test_scoped_plan_and_cli_are_exact(tmp_path: Path) -> None:
    """Validation remains file-scoped and the execution date cannot drift."""

    commands = audit.build_validation_plan(ROOT, tmp_path)
    assert audit.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in command.argv for command in commands)
    terminal = audit._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(audit.TERMINAL_CHECK_NAMES)
    assert audit.parse_args(["--date", audit.RUN_DATE]).date == audit.RUN_DATE
    with pytest.raises(SystemExit):
        audit.parse_args(["--date", "20260918"])


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-ARTIFACT
def test_json_and_yaml_readers_require_top_level_mappings(tmp_path: Path) -> None:
    """Readable sequence-shaped files cannot become authenticated authorities."""

    yaml_path = tmp_path / "sequence.yaml"
    yaml_path.write_text("- one\n- two\n", encoding="utf-8")
    json_path = tmp_path / "sequence.json"
    json_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        audit.load_yaml(yaml_path)
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("key: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        audit.load_yaml(invalid_yaml)
    with pytest.raises(ValueError, match="mapping"):
        audit.load_json(json_path)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        audit.load_json(malformed)


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-ARTIFACT
def test_small_reducers_cover_closed_error_boundaries(tmp_path: Path) -> None:
    """Malformed rows and each terminal cause remain explicit and deterministic."""

    declarations = audit._producer_declarations({"tasks": ["not-a-mapping"]})
    assert declarations == {}
    assert (
        audit._check_one_source(
            audit.SELECTED_SOURCES[0],
            lambda _source: {"access_state": "ok", "http_status": 200, "detail": "x"},
        )["access_state"]
        == "ok"
    )
    assert audit._receipts_pass({}, ["x"]) is False
    assert audit._gate_controls_valid({}) is False
    assert audit._hashes_match({"source_artifact_hashes": []}, ROOT) is False
    assert audit._terminal_state(False, True, True)[2] == "blocked"
    assert audit._terminal_state(True, False, True)[2] == "disqualified"
    assert audit._terminal_state(True, True, False)[0] == (
        "complete_disqualified_contract_authority"
    )
    assert audit._terminal_state(True, True, True)[2] == "null"

    capstone = {"task_dispositions": ["not-a-mapping"]}
    (tmp_path / audit.CAPSTONE_PATH).parent.mkdir(parents=True)
    (tmp_path / audit.CAPSTONE_PATH).write_text(json.dumps(capstone), encoding="utf-8")
    assert audit.collect_prior_dispositions(tmp_path) == []
    capstone["task_dispositions"] = "wrong"
    (tmp_path / audit.CAPSTONE_PATH).write_text(json.dumps(capstone), encoding="utf-8")
    with pytest.raises(ValueError, match="task_dispositions"):
        audit.collect_prior_dispositions(tmp_path)


# REQ-REPORT-7434 / SCENARIO-REPORT-7434-METHODS
def test_method_records_are_idempotent(tmp_path: Path) -> None:
    """The note is refreshed while the studying marker is appended only once."""

    (tmp_path / audit.STUDY_PATH).write_text("# Studying\n", encoding="utf-8")
    sources = _source_rows()
    audit._write_method_records(tmp_path, sources)
    first = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    audit._write_method_records(tmp_path, sources)
    second = (tmp_path / audit.STUDY_PATH).read_text(encoding="utf-8")
    assert first == second
    assert first.count("EXP7434-V652-METHOD-INGESTION") == 1
    note = (tmp_path / audit.NOTE_PATH).read_text(encoding="utf-8")
    assert "joint_certification" in note
    assert "scripts/research_conductor.py" in note
