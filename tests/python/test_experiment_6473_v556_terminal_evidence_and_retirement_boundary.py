"""Tests for Exp6473 V556 terminal evidence and retirement boundary.

Spec refs: REQ-REPORT-6473,
SCENARIO-REPORT-6473-TERMINAL-ROWS,
SCENARIO-REPORT-6473-CLAIM-RECOMPUTE,
SCENARIO-REPORT-6473-RETIREMENT-BOUNDARY,
SCENARIO-REPORT-6473-NO-QUEUE-GATE,
SCENARIO-REPORT-6473-SCHEMA.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6473_v556_terminal_evidence_and_retirement_boundary as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH
_ARTIFACT_CACHE: dict[str, Any] | None = None


def _artifact() -> dict[str, Any]:
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = mod.build_artifact(
            repo_root=REPO,
            date="20260821",
            result_path=Path("/tmp/experiment_6473_test_result.json"),
            write=False,
            duration_s=1.0,
            tests_run=[{"command": "focused", "exit_code": 0}],
        )
    return copy.deepcopy(_ARTIFACT_CACHE)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_req_report_6473_spec_declares_required_contract() -> None:
    """REQ-REPORT-6473: OpenSpec owns the Exp6473 report contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6473") :]

    for marker in (
        "SCENARIO-REPORT-6473-TERMINAL-ROWS",
        "SCENARIO-REPORT-6473-CLAIM-RECOMPUTE",
        "SCENARIO-REPORT-6473-RETIREMENT-BOUNDARY",
        "SCENARIO-REPORT-6473-NO-QUEUE-GATE",
        "SCENARIO-REPORT-6473-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert mod.FIELD_PRINCIPLES[field] in section


def test_scenario_report_6473_terminal_rows_freeze_v556_artifacts() -> None:
    """SCENARIO-REPORT-6473-TERMINAL-ROWS: each V556 task is visible."""

    artifact = _artifact()
    rows = {row["task_id"]: row for row in artifact["v556_terminal_rows"]}

    assert mod.validate_artifact(artifact) == []
    assert [row["task_id"] for row in artifact["v556_terminal_rows"]] == [
        task.task_id for task in mod.EXPECTED_V556_TASKS
    ]
    assert artifact["artifact_hash_manifest"]["expected_count"] == 13
    assert artifact["artifact_hash_manifest"]["present_count"] == 11
    assert artifact["artifact_hash_manifest"]["absent_count"] == 2
    assert artifact["artifact_hash_manifest"]["zero_byte_count"] == 0

    exp6460 = rows["exp6460-v556-terminal-handoff-and-queue-integrity"]
    assert exp6460["bytes"] > 0
    assert exp6460["sha256"].startswith("sha256:")
    assert exp6460["terminal_verdict"].startswith("complete_blocked_v556")

    for task_id in (
        "exp6465-representation-objective-causal-ab-v2",
        "exp6467-held-exact-constraint-energy-selection-v2",
    ):
        row = rows[task_id]
        assert row["artifact_state"] == "missing"
        assert row["execution_state"] == "not_executed"
        assert row["cannot_support_result"] is True
        assert row["eligibility"]["eligible"] is False
        assert row["eligibility"]["reason"] == "absent_artifact_no_result"

    blocked_rows = [
        row
        for row in artifact["v556_terminal_rows"]
        if "blocked" in str(row["terminal_verdict"]).lower()
    ]
    assert blocked_rows
    for row in blocked_rows:
        summary = row["gate_diagnostics"]
        assert summary["check"]
        assert "expected" in summary
        assert "observed" in summary
        assert summary["evidence_path"]


def test_scenario_report_6473_recomputes_claim_eligibility_from_rows() -> None:
    """SCENARIO-REPORT-6473-CLAIM-RECOMPUTE: booleans are not inherited."""

    artifact = _artifact()
    recomputed = artifact["capstone_eligibility_recomputation"]

    assert recomputed["independent"]["science_claim_eligible"]["eligible"] is False
    assert recomputed["independent"]["continuous_learning_claim_eligible"]["eligible"] is True
    assert recomputed["independent"]["arc_claim_eligible"]["eligible"] is True
    assert recomputed["independent"]["hardware_claim_eligible"]["eligible"] is False
    assert recomputed["all_fields_match_capstone"] is True
    assert set(recomputed["matches_capstone"]) == {
        "science_claim_eligible",
        "continuous_learning_claim_eligible",
        "arc_claim_eligible",
        "hardware_claim_eligible",
    }
    assert recomputed["rule_inputs"]["failed_gate_count"] == 2
    assert recomputed["rule_inputs"]["science_no_result_task_ids"] == [
        "exp6465-representation-objective-causal-ab-v2",
        "exp6467-held-exact-constraint-energy-selection-v2",
    ]
    assert artifact["aggregate_row_recomputation"]["absence_not_negative_science_finding"] is True


def test_scenario_report_6473_retirement_boundary_is_narrow() -> None:
    """SCENARIO-REPORT-6473-RETIREMENT-BOUNDARY: only repeated shapes retire."""

    artifact = _artifact()
    rows = {row["task_id"]: row for row in artifact["retirement_boundary_rows"]}

    assert set(rows) == {
        "exp6460-v556-terminal-handoff-and-queue-integrity",
        "exp6464-fixed-slot-grounding-exact-logic-ab",
        "exp6466-held-verifier-budget-allocation-v2",
    }
    for row in rows.values():
        assert row["mechanical_retirement"] is True
        assert row["boundary_class"] == "mechanically_retired_scope"
        assert row["artifact_blocked"] is True
        assert row["retired_because"] == "retire_if_same_verdict matched terminal shape"

    terminal = {row["task_id"]: row for row in artifact["v556_terminal_rows"]}
    assert terminal["exp6465-representation-objective-causal-ab-v2"]["retirement"][
        "mechanical_retirement"
    ] is False
    assert terminal["exp6467-held-exact-constraint-energy-selection-v2"]["retirement"][
        "mechanical_retirement"
    ] is False


def test_scenario_report_6473_no_queue_or_downstream_gate_scope() -> None:
    """SCENARIO-REPORT-6473-NO-QUEUE-GATE: no retired transition repeats."""

    artifact = _artifact()

    assert artifact["staged_queue_validation_performed"] is False
    assert artifact["downstream_gate_count"] == 0
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["gate_check_summary"]["acceptance_gates"] == [
        {
            "condition": "All 13 V556 task IDs have a row or an explicit missing-artifact row.",
            "passed": True,
        },
        {
            "condition": "staged_queue_validation_performed=false AND downstream_gate_count=0.",
            "passed": True,
        },
    ]
    assert artifact["preconditions_checked"]["research_roadmap_next_yaml_validated"] is False
    assert artifact["preconditions_checked"]["git_state"]["head_sha"]
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["rows"] == artifact["per_unit_rows"]


def test_scenario_report_6473_schema_write_and_validation_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6473-SCHEMA: validation catches schema drift."""

    artifact = _artifact()
    out = tmp_path / "artifact.json"
    written = mod.build_artifact(
        repo_root=REPO,
        result_path=out,
        write=True,
        duration_s=1.0,
        tests_run=[{"command": "focused", "exit_code": 0}],
    )

    assert out.is_file()
    assert mod.load_json(out)["reproducibility_checksum"] == written["reproducibility_checksum"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(artifact["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)

    validations = [
        ("delete", "status", "missing required fields"),
        ("set", ("staged_queue_validation_performed", True), "staged queue validation must be false"),
        ("set", ("downstream_gate_count", 1), "downstream_gate_count must be 0"),
        ("set", ("verifier_is_oracle", False), "verifier_is_oracle must be true"),
        ("set", ("inference_substrate", "live_llm_inference"), "inference_substrate mismatch"),
        ("set", ("honest_verdict", "ok"), "honest_verdict lacks terminal prefix"),
        ("set", ("reproducibility_checksum", "sha256:bad"), "reproducibility_checksum mismatch"),
    ]
    for mode, spec, expected in validations:
        bad = copy.deepcopy(artifact)
        if mode == "delete":
            del bad[spec]
        else:
            dotted, value = spec
            bad[dotted] = value
        if expected != "reproducibility_checksum mismatch":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert any(expected in error for error in mod.validate_artifact(bad))

    bad = copy.deepcopy(artifact)
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_principles must cover exactly required fields" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["field_provenance"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["v556_terminal_rows"][0]["gate_diagnostics"] = {}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "blocked row missing normalized gate diagnostic" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["v556_terminal_rows"] = bad["v556_terminal_rows"][:-1]
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "V556 terminal row count mismatch" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["gate_check_summary"] = []
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "gate_check_summary must be a mapping" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["gate_check_summary"]["acceptance_gates"][0]["passed"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "all 13 V556 task IDs must be accounted" in mod.validate_artifact(bad)

    bad = copy.deepcopy(artifact)
    bad["gate_check_summary"]["acceptance_gates"][1]["passed"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "queue and downstream gate boundary must pass" in mod.validate_artifact(bad)


def test_scenario_report_6473_fixture_states_and_bad_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6473-TERMINAL-ROWS: malformed fixtures stay explicit."""

    present = tmp_path / "present.json"
    _write_json(present, {"status": "success", "honest_verdict": "success: fixture"})
    zero = tmp_path / "zero.json"
    zero.touch()
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    partial = tmp_path / "partial.json"
    _write_json(partial, {"status": "complete_partial"})
    flagged = tmp_path / "flagged.json"
    _write_json(flagged, {"status": "flagged"})

    monkeypatch.setattr(
        mod,
        "EXPECTED_V556_TASKS",
        (
            mod.ExpectedTask("exp1-present", "exp1", Path("present.json"), "other"),
            mod.ExpectedTask("exp2-zero", "exp2", Path("zero.json"), "other"),
            mod.ExpectedTask("exp3-bad", "exp3", Path("malformed.json"), "other"),
            mod.ExpectedTask("exp4-missing", "exp4", Path("missing.json"), "other"),
            mod.ExpectedTask("exp5-partial", "exp5", Path("partial.json"), "other"),
            mod.ExpectedTask("exp6-flagged", "exp6", Path("flagged.json"), "other"),
        ),
    )
    rows, payloads = mod.v556_terminal_rows(tmp_path, retirements={})
    by_id = {row["task_id"]: row for row in rows}

    assert set(payloads) == {"exp1", "exp5", "exp6"}
    assert by_id["exp1-present"]["artifact_state"] == "complete"
    assert by_id["exp2-zero"]["artifact_state"] == "zero_byte"
    assert by_id["exp3-bad"]["artifact_state"] == "malformed"
    assert "JSONDecodeError" in by_id["exp3-bad"]["load_error"]
    assert by_id["exp4-missing"]["artifact_state"] == "missing"
    assert by_id["exp5-partial"]["artifact_state"] == "partial"
    assert by_id["exp6-flagged"]["artifact_state"] == "flagged"
    assert mod.artifact_hash_manifest(rows)["zero_byte_paths"] == ["zero.json"]

    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod.validate_artifact(tmp_path / "does-not-exist.json")[0].startswith(
        "unloadable artifact"
    )
    assert mod._status_text(None) == ""
    assert mod._same_verdict_shape("complete_null_result", {"terminal_verdict": "null"}) is True
    assert mod._same_verdict_shape("success: exact", {"terminal_verdict": "success: exact"}) is True
    assert mod._value_at({"outer": 1}, "outer.inner") is None
    assert len(mod.tests_run_receipts(None)) == len(mod.DEFAULT_TEST_COMMANDS)

    monkeypatch.setattr(mod, "validate_artifact", lambda _artifact: ["forced validation error"])
    with pytest.raises(ValueError, match="forced validation error"):
        mod.build_artifact(repo_root=tmp_path, write=False, duration_s=1.0)
