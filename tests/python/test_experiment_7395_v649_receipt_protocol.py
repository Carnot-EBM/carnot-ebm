"""Tests for the V649 current-work receipt and assignment protocol.

Spec refs: REQ-REPORT-7395 and SCENARIO-REPORT-7395-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot import experiment_7395_v649_receipt_protocol as mod
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "command": f"check {name}",
        "command_argv": ["check", name],
        "command_environment": {},
        "scope": "unit_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "1" * 64,
        "passed": passed,
        "timed_out": False,
        "output_tail": "clean" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7383_v648_canary_reducer": str(
                (ROOT / "python/carnot/experiment_7383_v648_canary_reducer.py").resolve()
            ),
            "carnot.experiment_7395_v649_receipt_protocol": str((ROOT / mod.MODULE_PATH).resolve()),
            "carnot.reporting.current_work_receipt": str(
                (ROOT / mod.RECEIPT_HELPER_PATH).resolve()
            ),
        }
    return row


def _passing_receipts() -> list[dict[str, Any]]:
    return [_receipt(name) for name in (*REQUIRED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES)]


def test_scenario_report_7395_reducer_replays_exact_historical_inputs() -> None:
    """SCENARIO-REPORT-7395-REDUCER preserves transport, syntax, and proof scope."""

    replay = mod.replay_assignment_and_proof(ROOT)
    assert replay["assignment"]["usable_proposal_count"] == 4
    assert replay["assignment"]["sat_extendible_count"] == 0
    assert replay["assignment"]["assignment_reducer_ready_score"] == 1
    assert replay["proof"]["proof_boundary_replay_ready_score"] == 1
    assert replay["source_literal_fidelity_count"] == 4
    assert replay["usable_gate"]["operator"] == ">="
    assert replay["usable_gate"]["passed"] is True


def test_scenario_report_7395_reducer_uses_published_integer_parsing() -> None:
    """REQ-REPORT-7395 keeps integer source literals distinct from JSON numbers."""

    calls, formulas = mod.exp7383.load_assignment_evidence(ROOT)
    changed = deepcopy(calls)
    changed[0]["raw_reply"] = '{"assignments":[1.0,-2]}'
    changed[0]["raw_reply_sha256"] = mod.exp7383.sha256_text(changed[0]["raw_reply"])
    changed[0]["raw_response"]["choices"][0]["message"]["content"] = changed[0]["raw_reply"]
    reduced = mod.exp7383.reduce_assignment_receipts(changed, formulas)
    assert "variable_reference_invalid:0" in reduced["errors"]
    assert reduced["rows"][0]["schema_valid"] is False
    assert reduced["rows"][0]["sat_extendible"] is False


def test_scenario_report_7395_validation_plan_is_frozen(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-VALIDATION freezes all eight affected checks."""

    commands = mod.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(ROOT, commands) == []
    coverage = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(coverage.command_environment)["COVERAGE_FILE"].endswith("/.coverage")

    bad = [*commands, mod.validation_scope.CommandSpec("full_python_suite", ("true",), "broad")]
    assert "full_python_suite_forbidden" in mod.validate_validation_plan(ROOT, bad)


def test_scenario_report_7395_contract_is_advisory_and_mutations_reject() -> None:
    """SCENARIO-REPORT-7395-CONTRACT retains missing authority as advisory."""

    roadmap = mod.load_yaml(ROOT / "research-roadmap.yaml")
    result = mod.compare_contract_authorities(
        (ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text(encoding="utf-8"),
        roadmap,
    )
    assert result["yaml_milestone"] == mod.MILESTONE
    assert result["yaml_task_count"] == 14
    assert result["passed"] is False
    assert "markdown_milestone_mismatch" in result["errors"]

    rows = mod.yaml_contract_rows(roadmap)
    mutations = mod.run_contract_mutations(rows)
    assert [row["mutation"] for row in mutations] == [
        "missing_row",
        "reordered_rows",
        "wrong_field",
    ]
    assert all(row["rejected"] for row in mutations)


def test_scenario_report_7395_contract_parsers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-CONTRACT reports malformed and wrong V649 authorities."""

    assert mod.load_yaml(tmp_path / "missing.yaml") == {}
    broken = mod.compare_contract_authorities("not a contract", {})
    assert any(error.startswith("markdown_parse_error:") for error in broken["errors"])
    assert any(error.startswith("yaml_parse_error:") for error in broken["errors"])
    assert "markdown_task_count_mismatch" in broken["errors"]
    assert "yaml_task_count_mismatch" in broken["errors"]

    wrong = mod.load_yaml(ROOT / "research-roadmap.yaml")
    wrong["milestone"] = "wrong"
    wrong["tasks"] = wrong["tasks"][:-1]
    result = mod.compare_contract_authorities(
        (ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text(encoding="utf-8"),
        wrong,
    )
    assert "yaml_milestone_mismatch" in result["errors"]
    assert "yaml_task_count_mismatch" in result["errors"]


def test_scenario_report_7395_mutation_controls_cover_six_defects(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-MUTATIONS records all six rejecting controls."""

    rows, clean = mod.run_receipt_controls(tmp_path)
    assert clean["errors"] == []
    assert clean["model_invoked"] is False
    assert [row["mutation"] for row in rows] == list(mod.RECEIPT_MUTATIONS)
    assert all(row["rejected"] for row in rows)


def test_scenario_report_7395_artifact_cold_reducer_detects_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7395-ARTIFACT recomputes scores and source hashes."""

    artifact = mod.build_artifact_for_test(ROOT, tmp_path, _passing_receipts())
    assert mod.validate_artifact(artifact, root=ROOT, sidecar_root=tmp_path) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["assignment_reducer_ready_score"] == 1
    assert artifact["receipt_protocol_ready_score"] == 1
    assert artifact["promotion_score"] == 0

    changed = deepcopy(artifact)
    changed["assignment_reducer_ready_score"] = 0
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "assignment_reducer_ready_score_mismatch" in mod.validate_artifact(
        changed, root=ROOT, sidecar_root=tmp_path
    )

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(
        changed, root=ROOT, sidecar_root=tmp_path
    )


def test_req_report_7395_blocked_preconditions_name_exact_failure(tmp_path: Path) -> None:
    """REQ-REPORT-7395 maps absent unchanged input to an exact blocked summary."""

    checks, hashes = mod.collect_preconditions(tmp_path)
    artifact = mod.build_blocked_artifact(
        checks,
        hashes,
        started_at="2026-09-18T00:00:00Z",
        duration_s=0.01,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["first_failure"]["observed"] is None
    assert artifact["assignment_reducer_ready_score"] == 0
    assert artifact["receipt_protocol_ready_score"] == 0


def test_req_report_7395_validator_rejects_structure_and_failed_checks(tmp_path: Path) -> None:
    """REQ-REPORT-7395 fails closed on structure and affected validation."""

    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")
    failed = mod.build_artifact_for_test(ROOT, tmp_path, _passing_receipts())
    failed["validation_receipts"][0]["passed"] = False
    failed["reproducibility_checksum"] = mod.artifact_checksum(failed)
    errors = mod.validate_artifact(failed, root=ROOT, sidecar_root=tmp_path)
    assert "assignment_reducer_ready_score_mismatch" in errors
    assert "receipt_protocol_ready_score_mismatch" in errors

    terminal = mod.terminal_command_specs(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)


def test_req_report_7395_plan_and_artifact_defects_are_explicit(tmp_path: Path) -> None:
    """REQ-REPORT-7395 covers plan drift and every cold structural check."""

    commands = mod.build_validation_plan(ROOT, tmp_path / "plan")
    no_coverage = [row for row in commands if row.name != "changed_module_coverage_report"]
    plan_errors = mod.validate_validation_plan(ROOT, no_coverage)
    assert "required_command_names_changed" in plan_errors
    assert "coverage_file_not_preserved" in plan_errors

    artifact = mod.build_artifact_for_test(ROOT, tmp_path / "sidecars", _passing_receipts())
    changed = deepcopy(artifact)
    changed.update(
        schema="wrong",
        MODEL_SPECS=["unexpected"],
        inference_substrate_class="no_model_load",
        source_artifact_hashes={"missing.json": "sha256:bad"},
        field_principles={},
        assignment_replay={},
        promotion_score=1,
    )
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    errors = mod.validate_artifact(changed, root=ROOT, sidecar_root=tmp_path / "sidecars")
    assert {
        "identity_invalid",
        "current_invocation_declaration_invalid",
        "current_substrate_declaration_invalid",
        "source_hash_mismatch:missing.json",
        "field_principles_incomplete",
        "assignment_replay_mismatch",
        "promotion_score_nonzero",
    }.issubset(errors)

    blocked = mod.build_blocked_artifact(
        [{"passed": False, "observed": None}],
        {},
        started_at="2026-09-18T00:00:00Z",
        duration_s=0.01,
    )
    blocked["assignment_reducer_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_scores_nonzero" in mod.validate_artifact(blocked, root=ROOT)


def test_req_report_7395_build_artifact_classifies_blocked_and_disqualified(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-7395 keeps external absence separate from failed owned checks."""

    checks, hashes = mod.collect_preconditions(ROOT)
    blocked_checks = deepcopy(checks)
    blocked_checks[0]["passed"] = False
    blocked = mod.build_artifact(
        root=ROOT,
        sidecar_dir=tmp_path / "blocked",
        preconditions=blocked_checks,
        source_hashes=hashes,
        receipts=_passing_receipts(),
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:02Z",
        duration_s=2.0,
        phase_spans=[{"phase": "test", "start_s": 0.0, "end_s": 1.0}],
    )
    assert blocked["verdict_class"] == "blocked"

    failed_receipts = _passing_receipts()
    failed_receipts[0]["passed"] = False
    failed = mod.build_artifact(
        root=ROOT,
        sidecar_dir=tmp_path / "failed",
        preconditions=checks,
        source_hashes=hashes,
        receipts=failed_receipts,
        started_at="2026-09-18T00:00:00Z",
        ended_at="2026-09-18T00:00:02Z",
        duration_s=2.0,
        phase_spans=[{"phase": "test", "start_s": 0.0, "end_s": 1.0}],
    )
    assert failed["verdict_class"] == "disqualified"
