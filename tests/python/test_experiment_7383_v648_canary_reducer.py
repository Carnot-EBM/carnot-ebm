"""Tests for the V648 assignment receipt reducer repair.

Spec refs: REQ-REPORT-7383 and SCENARIO-REPORT-7383-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from carnot import experiment_7383_v648_canary_reducer as mod
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build a command receipt with the fields used by the cold reducer."""

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
        "output_tail": "ok" if passed else "failed",
    }
    if name == "worktree_imports":
        row["resolved_imports"] = {
            "carnot.experiment_7383_v648_canary_reducer": str((ROOT / mod.MODULE_PATH).resolve())
        }
    return row


def _passing_receipts(*, terminal: bool = True) -> list[dict[str, Any]]:
    names = list(REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(mod.TERMINAL_CHECK_NAMES)
    return [_receipt(name) for name in names]


def _raw_evidence() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return mod.load_assignment_evidence(ROOT)


@pytest.mark.parametrize(
    ("usable_count", "expected"),
    [(2, False), (3, True), (4, True)],
)
def test_scenario_report_7383_operators_use_usable_floor(usable_count: int, expected: bool) -> None:
    """SCENARIO-REPORT-7383-OPERATORS uses >= rather than equality."""

    rows = mod.build_receipt_gate_rows(
        usable_count=usable_count,
        promotion_score=0,
        execution_venue="host",
        verifier_is_oracle=True,
        verdict_class="circular_positive",
    )
    by_check = {row["check"]: row for row in rows}
    assert by_check["usable_assignment_proposals"]["operator"] == ">="
    assert by_check["usable_assignment_proposals"]["passed"] is expected
    assert by_check["promotion_forbidden"]["operator"] == "=="
    assert by_check["promotion_forbidden"]["passed"] is True
    assert by_check["execution_venue_closed"]["passed"] is True
    assert by_check["oracle_verdict_class"]["passed"] is True


def test_scenario_report_7383_replay_separates_transport_and_sat() -> None:
    """SCENARIO-REPORT-7383-REPLAY replays all four exact historical calls."""

    calls, formulas = _raw_evidence()
    reduced = mod.reduce_assignment_receipts(calls, formulas)
    assert reduced["errors"] == []
    assert reduced["usable_proposal_count"] == 4
    assert reduced["response_fidelity_count"] == 4
    assert reduced["sat_extendible_count"] == 0
    assert reduced["historical_transport_ready_score"] == 1
    assert reduced["assignment_reducer_ready_score"] == 1
    assert [row["call_index"] for row in reduced["rows"]] == [0, 1, 2, 3]
    assert all(row["usable_proposal"] for row in reduced["rows"])
    assert not any(row["sat_extendible"] for row in reduced["rows"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("response_hash", "response_hash_mismatch:0"),
        ("decoding_seed", "decoding_seed_mismatch:0"),
        ("unsupported_variable", "variable_reference_invalid:0"),
        ("truncated", "truncated_output:0"),
        ("missing_runtime_identity", "runtime_identity_missing:0"),
    ],
)
def test_scenario_report_7383_mutations_fail_their_own_checks(
    mutation: str, expected_error: str
) -> None:
    """SCENARIO-REPORT-7383-MUTATIONS rejects each raw receipt defect."""

    calls, formulas = _raw_evidence()
    changed = deepcopy(calls)
    row = changed[0]
    if mutation == "response_hash":
        row["raw_reply_sha256"] = "sha256:" + "0" * 64
    elif mutation == "decoding_seed":
        row["raw_request"]["seed"] += 1
        row["raw_request_sha256"] = mod.hash_json(row["raw_request"])
    elif mutation == "unsupported_variable":
        row["raw_reply"] = json.dumps({"assignments": [1, 999]}, separators=(",", ":"))
        row["raw_reply_sha256"] = mod.sha256_text(row["raw_reply"])
        row["raw_response"]["choices"][0]["message"]["content"] = row["raw_reply"]
    elif mutation == "truncated":
        row["finish_reason"] = "length"
    elif mutation == "missing_runtime_identity":
        row.pop("runtime_identity_receipt")
    reduced = mod.reduce_assignment_receipts(changed, formulas)
    assert expected_error in reduced["errors"]
    assert reduced["assignment_reducer_ready_score"] == 0


def test_scenario_report_7383_replays_proof_boundary_and_manifest() -> None:
    """SCENARIO-REPORT-7383-REPLAY keeps proof and manifest bytes hash-bound."""

    replay = mod.replay_proof_boundary(ROOT)
    assert replay["errors"] == []
    assert replay["proof_memory_replay_ready_score"] == 1
    assert replay["proof_boundary_replay_ready_score"] == 1
    assert replay["manifest_sha256"] == mod.sha256_file(ROOT / mod.SEALED_MANIFEST_PATH)
    assert replay["proof_boundary_artifact_sha256"] == mod.sha256_file(
        ROOT / mod.PROOF_BOUNDARY_PATH
    )


def test_scenario_report_7383_history_preserves_disqualification() -> None:
    """SCENARIO-REPORT-7383-HISTORY never promotes the old canary."""

    sidecar = mod.historical_model_inputs(ROOT)
    assert sidecar["artifact_sha256"] == mod.sha256_file(ROOT / mod.CANARY_PATH)
    assert sidecar["historical_verdict_class"] == "disqualified"
    assert sidecar["historical_flagged_adversarial"] is True
    assert sidecar["historical_model_specs"] == ["unsloth/Qwen3.8-27B-GGUF"]
    assert sidecar["counted_as_current"] is False
    assert sidecar["authorizes_current_readiness"] is False


def test_scenario_report_7383_artifact_reducer_detects_false_readiness() -> None:
    """SCENARIO-REPORT-7383-ARTIFACT rejects a forged readiness score."""

    artifact = mod.build_artifact_for_test(ROOT, _passing_receipts())
    assert mod.validate_artifact(artifact, root=ROOT, require_terminal=True) == []
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["execution_venue"] == "host"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["assignment_reducer_ready_score"] == 1
    assert artifact["proof_boundary_replay_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"

    forged = deepcopy(artifact)
    forged["assignment_reducer_ready_score"] = 0
    forged["reproducibility_checksum"] = mod.artifact_checksum(forged)
    assert "assignment_reducer_ready_score_mismatch" in mod.validate_artifact(
        forged, root=ROOT, require_terminal=True
    )

    forged = deepcopy(artifact)
    forged["execution_venue"] = {"host": "device"}
    forged["reproducibility_checksum"] = mod.artifact_checksum(forged)
    assert "current_execution_declaration_invalid" in mod.validate_artifact(
        forged, root=ROOT, require_terminal=True
    )


def test_scenario_report_7383_artifact_blocks_missing_external_input(tmp_path: Path) -> None:
    """REQ-REPORT-7383 maps unchanged missing input to a blocked artifact."""

    checks, _hashes = mod.collect_preconditions(tmp_path)
    artifact = mod.build_blocked_artifact(
        checks,
        run_date=mod.RUN_DATE,
        started_at="2026-09-18T00:00:00Z",
        duration_s=0.01,
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["observed_value"] in {
        "missing",
        "missing_or_empty",
    }
    assert artifact["assignment_reducer_ready_score"] == 0
    assert artifact["proof_boundary_replay_ready_score"] == 0


def test_scenario_report_7383_validation_plan_is_exact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7383-ARTIFACT derives the exact Exp7358 plan."""

    commands = mod.build_validation_plan(ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(ROOT, commands) == []
    assert all("full_python_suite" not in command.name for command in commands)
    coverage = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert "COVERAGE_FILE" in dict(coverage.command_environment)


def test_scenario_report_7383_defensive_reducers_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7383 covers malformed rows, proofs, operators, and plans."""

    with pytest.raises(ValueError, match="unsupported_comparison_operator"):
        mod.compare("!=", 1, 1)
    assert mod._response_content({}) is None

    malformed = mod.reduce_assignment_receipts([{}], [])
    assert "raw_call_count" in malformed["errors"]
    assert "raw_request_missing:0" in malformed["errors"]
    assert "request_hash_mismatch:0" in malformed["errors"]
    assert "formula_identity_invalid:0" in malformed["errors"]
    assert "runtime_identity_missing:0" in malformed["errors"]

    calls, formulas = _raw_evidence()
    invalid_runtime = deepcopy(calls)
    invalid_runtime[0]["runtime_identity_receipt"]["owned_by_task"] = False
    reduced = mod.reduce_assignment_receipts(invalid_runtime, formulas)
    assert "runtime_identity_invalid:0" in reduced["errors"]

    missing_proof = mod.replay_proof_boundary(tmp_path)
    assert missing_proof["proof_boundary_replay_ready_score"] == 0
    assert {
        "proof_memory_artifact_missing",
        "proof_boundary_artifact_missing",
        "sealed_manifest_missing",
    }.issubset(missing_proof["errors"])

    for relative in (mod.PROOF_MEMORY_PATH, mod.PROOF_BOUNDARY_PATH, mod.SEALED_MANIFEST_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    manifest = json.loads((tmp_path / mod.SEALED_MANIFEST_PATH).read_text(encoding="utf-8"))
    manifest["tampered"] = True
    (tmp_path / mod.SEALED_MANIFEST_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    tampered_proof = mod.replay_proof_boundary(tmp_path)
    assert "frozen_protocol_hash_mismatch" in tampered_proof["errors"]
    assert "proof_memory_provenance_hash_mismatch" in tampered_proof["errors"]
    assert any(
        error.startswith("proof_sidecar_hash_mismatch:") for error in tampered_proof["errors"]
    )

    commands = mod.build_validation_plan(ROOT, tmp_path / "plan")
    broken_plan = [*commands[1:], CommandSpec("full_python_suite", ("true",), "broad")]
    plan_errors = mod.validate_validation_plan(ROOT, broken_plan)
    assert "required_command_names_changed" in plan_errors
    assert "full_python_suite_forbidden" in plan_errors


def test_scenario_report_7383_validator_rejects_structural_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7383-ARTIFACT rejects structure and checksum mutations."""

    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.validate_artifact({})[0].startswith("missing_required_field:")
    assert mod._artifact_state({}, require_terminal=True)[2] == "blocked"

    artifact = mod.build_artifact_for_test(ROOT, _passing_receipts())
    changed = deepcopy(artifact)
    changed.update(
        schema="wrong",
        verdict_class="unknown",
        field_principles={},
        rows=[],
        assignment_reduction={},
        proof_boundary_replay={},
        historical_model_inputs={},
        discrepancy_rows=[],
        acceptance_gate_results=[],
        gate_check_summary={},
        reproducibility_checksum="sha256:bad",
    )
    changed["source_artifact_hashes"] = {"missing.json": "sha256:bad"}
    errors = mod.validate_artifact(changed, root=ROOT)
    assert {
        "identity_invalid",
        "verdict_class_invalid",
        "field_principles_incomplete",
        "source_hash_mismatch:missing.json",
        "assignment_reduction_mismatch",
        "assignment_rows_mismatch",
        "proof_boundary_replay_mismatch",
        "historical_model_inputs_mismatch",
        "discrepancy_rows_mismatch",
        "acceptance_gate_results_mismatch",
        "terminal_state_mismatch",
        "gate_check_summary_mismatch",
        "reproducibility_checksum_mismatch",
    }.issubset(errors)

    blocked = mod.build_blocked_artifact(
        [{"passed": False, "check": "missing", "observed_value": "missing"}],
        run_date=mod.RUN_DATE,
        started_at="2026-09-18T00:00:00Z",
        duration_s=0.01,
    )
    blocked["assignment_reducer_ready_score"] = 1
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_scores_nonzero" in mod.validate_artifact(blocked, root=tmp_path)

    candidate_state = mod._artifact_state(artifact, require_terminal=False)
    assert candidate_state[2] == "partial"
    failed = deepcopy(artifact)
    failed["validation_receipts"][0]["passed"] = False
    assert mod._artifact_state(failed, require_terminal=True)[2] == "disqualified"

    terminal = mod._terminal_commands(ROOT, tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(mod.TERMINAL_CHECK_NAMES)


def test_req_report_7395_finishes_exp7383_numeric_and_hash_paths(tmp_path: Path) -> None:
    """REQ-REPORT-7395 covers Exp7383 integer parsing and source-hash failures."""

    assert mod.compare("is", None, None) is True
    assert mod._response_content({"raw_response": {"choices": []}}) is None

    calls, formulas = _raw_evidence()
    changed = deepcopy(calls)
    changed[0]["raw_reply"] = '{"assignments":[1.0,-2]}'
    changed[0]["raw_reply_sha256"] = mod.sha256_text(changed[0]["raw_reply"])
    changed[0]["raw_response"]["choices"][0]["message"]["content"] = changed[0]["raw_reply"]
    changed[0]["decoded_assignments"] = None
    reduced = mod.reduce_assignment_receipts(changed, formulas)
    assert "variable_reference_invalid:0" in reduced["errors"]
    assert reduced["rows"][0]["sat_extendible"] is False

    artifact = mod.build_artifact_for_test(ROOT, _passing_receipts())
    artifact["source_artifact_hashes"] = {"missing.json": "sha256:bad"}
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    assert "source_hash_mismatch:missing.json" in mod.validate_artifact(artifact, root=ROOT)
