"""Tests for Exp6298 terminal-evidence preflight.

Spec refs: REQ-INFRA-6298, SCENARIO-INFRA-6298-1,
SCENARIO-INFRA-6298-2, SCENARIO-INFRA-6298-3,
SCENARIO-INFRA-6298-4, SCENARIO-INFRA-6298-5,
SCENARIO-INFRA-6298-6.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6298_terminal_evidence_preflight_linter as mod
from carnot import terminal_evidence_preflight as preflight
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV
from carnot.terminal_artifacts import path_sha256


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/research-harnesses/spec.md"


def test_spec_declares_req_6298_fields_and_scenarios() -> None:
    """REQ-INFRA-6298: OpenSpec anchors the preflight contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6298") :]

    for token in (
        "REQ-INFRA-6298",
        "SCENARIO-INFRA-6298-1",
        "SCENARIO-INFRA-6298-2",
        "SCENARIO-INFRA-6298-3",
        "SCENARIO-INFRA-6298-4",
        "SCENARIO-INFRA-6298-5",
        "SCENARIO-INFRA-6298-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert token in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_v542_failure_replay_rejects_exact_prior_artifacts() -> None:
    """SCENARIO-INFRA-6298-1: V542 failure shapes remain rejected."""

    rows = preflight.replay_v542_failure_fixtures(REPO)
    by_id = {row["fixture_id"]: row for row in rows}

    exp6288 = by_id["exp6288"]
    assert exp6288["accepted"] is False
    assert exp6288["path_sha256"] == path_sha256(REPO / exp6288["path"])
    assert "duration_floor_violation" in exp6288["failure_classes"]
    assert "methodology_missing" in exp6288["failure_classes"]

    exp6289 = by_id["exp6289"]
    assert exp6289["accepted"] is False
    assert "test_exit_code_missing" in exp6289["failure_classes"]

    exp6290 = by_id["exp6290"]
    assert exp6290["accepted"] is False
    assert "test_exit_code_nonzero" in exp6290["failure_classes"]

    assert all(row["expected_accept"] is False for row in rows)


def test_synthetic_fixture_manifest_zero_false_counts() -> None:
    """SCENARIO-INFRA-6298-2: clean passes and bad fixtures fail."""

    manifest = preflight.build_synthetic_fixture_manifest()
    summary = preflight.evaluate_fixture_manifest(manifest)
    by_id = {row["fixture_id"]: row for row in summary["fixture_results"]}

    assert by_id["clean"]["accepted"] is True
    assert "missing_required_field" in by_id["missing_field"]["failure_classes"]
    assert "missing_terminal_prefix" in by_id["bad_prefix"]["failure_classes"]
    assert "gate_field_type_mismatch" in by_id["bad_gate_type"]["failure_classes"]
    assert "determination_dropped" in by_id["determination_drop"]["failure_classes"]
    assert summary["clean_fixture_accept_count"] == 1
    assert summary["bad_fixture_reject_count"] == 4
    assert type(summary["false_accept_count"]) is int
    assert type(summary["false_reject_count"]) is int
    assert summary["false_accept_count"] == 0
    assert summary["false_reject_count"] == 0


def test_substrate_methodology_and_duration_fail_closed() -> None:
    """SCENARIO-INFRA-6298-3: unknown compute and short timing are rejected."""

    payload = preflight.clean_fixture_payload()
    payload["inference_substrate"] = "mystery_live_gpu_substrate"
    payload["duration_s"] = 0.01
    payload["model_specs"] = {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}
    del payload["random_seed"]

    result = preflight.preflight_payload(payload, gate_fields=preflight.DEFAULT_GATE_FIELDS)

    assert result["accepted"] is False
    assert "unknown_compute_bound_substrate" in result["failure_classes"]
    assert "duration_floor_violation" in result["failure_classes"]
    assert "methodology_missing" in result["failure_classes"]
    floor = result["substrate_duration_and_methodology_check"]["duration_floor"]
    assert floor["min_duration_s"] == 60.0


def test_test_command_receipts_are_compared_not_executed() -> None:
    """SCENARIO-INFRA-6298-4: artifact command strings are never executed."""

    dangerous = "/bin/sh -c 'exit 99'"
    payload = preflight.clean_fixture_payload(
        test_commands=[dangerous],
        test_exit_codes={dangerous: 0},
    )

    result = preflight.preflight_payload(payload, gate_fields=preflight.DEFAULT_GATE_FIELDS)
    assert result["accepted"] is True
    assert result["test_command_and_exit_code_check"]["executed_commands"] == []

    missing = copy.deepcopy(payload)
    missing["test_exit_codes"] = {}
    missing_result = preflight.preflight_payload(missing, gate_fields=preflight.DEFAULT_GATE_FIELDS)
    assert "test_exit_code_missing" in missing_result["failure_classes"]

    nonzero = copy.deepcopy(payload)
    nonzero["test_exit_codes"] = {dangerous: 2}
    nonzero_result = preflight.preflight_payload(nonzero, gate_fields=preflight.DEFAULT_GATE_FIELDS)
    assert "test_exit_code_nonzero" in nonzero_result["failure_classes"]


def test_gate_fields_must_be_exact_bare_terminal_typed() -> None:
    """SCENARIO-INFRA-6298-5: staged gate fields fail closed."""

    wrapped = preflight.clean_fixture_payload()
    wrapped["terminal_evidence_preflight_ready_score"] = {
        "value": 1.0,
        "principle": "wrapped fixture",
    }
    wrapped_result = preflight.preflight_payload(wrapped, gate_fields=preflight.DEFAULT_GATE_FIELDS)
    assert "gate_field_not_bare" in wrapped_result["failure_classes"]

    wrong_type = preflight.clean_fixture_payload()
    wrong_type["terminal_evidence_preflight_ready_score"] = "1.0"
    wrong_type_result = preflight.preflight_payload(
        wrong_type, gate_fields=preflight.DEFAULT_GATE_FIELDS
    )
    assert "gate_field_type_mismatch" in wrong_type_result["failure_classes"]

    nonterminal = preflight.clean_fixture_payload()
    nonterminal["status"] = "running"
    nonterminal["honest_verdict"] = "running"
    nonterminal_result = preflight.preflight_payload(
        nonterminal, gate_fields=preflight.DEFAULT_GATE_FIELDS
    )
    assert "gate_field_nonterminal_artifact" in nonterminal_result["failure_classes"]


def test_defensive_edge_shapes_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6298-3 to SCENARIO-INFRA-6298-5: edge shapes are covered."""

    assert preflight._unique(["a", "a", "b"]) == ["a", "b"]
    assert preflight._is_finite_number(True) is False
    assert preflight._is_finite_number("1") is False
    assert preflight._is_substantive({"value": "x"}) is True
    assert preflight._protected_determination_key("metric", 1) is False

    assert preflight._type_matches(True, "bool") is True
    assert preflight._type_matches(1, "int") is True
    assert preflight._type_matches(1.0, "float") is True
    assert preflight._type_matches("x", "string") is True
    assert preflight._type_matches([], "list") is True
    assert preflight._type_matches({}, "dict") is True
    assert preflight._type_matches("x", "unknown") is False

    malformed = preflight.preflight_payload(["not", "an", "object"])
    assert malformed["failure_classes"] == ["malformed_payload"]
    missing_path = preflight.preflight_artifact_path(tmp_path / "missing.json")
    assert missing_path["failure_classes"] == ["malformed_payload"]

    payload = preflight.clean_fixture_payload()
    payload["field_principles"] = []
    payload["field_provenance"] = []
    payload["test_commands"] = []
    payload["test_exit_codes"] = []
    payload.pop("duration_s")
    payload.pop("random_seed")
    payload.pop("reproducibility_checksum")
    payload["preconditions_checked"] = {}
    edge = preflight.preflight_payload(
        payload,
        gate_fields=[{"field": "missing_gate_value", "expected_type": "number"}],
        baseline_payload={"metric": 1},
    )

    assert "field_principles_not_mapping" in edge["failure_classes"]
    assert "field_provenance_not_mapping" in edge["failure_classes"]
    assert "duration_missing" in edge["failure_classes"]
    assert "reproducibility_missing" in edge["failure_classes"]
    assert "test_commands_missing" in edge["failure_classes"]
    assert "test_exit_codes_not_mapping" in edge["failure_classes"]
    assert "gate_field_missing" in edge["failure_classes"]

    command_edge = preflight.clean_fixture_payload()
    command_edge["test_commands"] = ["declared"]
    command_edge["test_exit_codes"] = {"declared": "0", "extra": 0}
    command_result = preflight.preflight_payload(command_edge)
    assert "test_exit_code_extra" in command_result["failure_classes"]
    assert "test_exit_code_not_int" in command_result["failure_classes"]

    compute_gap = preflight.clean_fixture_payload()
    compute_gap["model_specs"] = {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}
    compute_gap.pop("inference_substrate")
    compute_gap.pop("duration_s")
    compute_gap.pop("random_seed")
    compute_gap.pop("reproducibility_checksum")
    compute_result = preflight.preflight_payload(compute_gap)
    assert "missing_inference_substrate" in compute_result["failure_classes"]
    assert "duration_missing" in compute_result["failure_classes"]
    missing_methodology = compute_result["substrate_duration_and_methodology_check"][
        "missing_methodology"
    ]
    assert "reproducibility_checksum" in missing_methodology
    assert "duration_s" in missing_methodology

    manifest_edge = preflight.evaluate_fixture_manifest({"fixtures": [None], "gate_fields": "bad"})
    assert manifest_edge["fixture_results"] == []


def test_report_validation_manifest_and_atomic_write(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6298-6: report schema and CLI artifact contract hold."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    env = {ARTIFACT_ROOT_ENV: str(artifact_root)}
    manifest = preflight.build_synthetic_fixture_manifest()
    manifest_path = mod.write_synthetic_fixture_manifest(manifest, REPO, env=env)
    before = mod.protected_hashes(REPO)
    report = mod.build_report(
        REPO,
        date="20260811",
        command_receipts=[
            {"command": mod.FOCUSED_TEST_COMMAND, "exit_code": 0},
            {"command": ".venv/bin/pytest tests/python -q", "exit_code": 0},
        ],
        before_hashes=before,
        synthetic_manifest_file_sha256=path_sha256(manifest_path),
        started_at=0.0,
    )

    assert mod.validate_report(report) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(report["field_provenance"])
    assert report["false_accept_count"] == 0
    assert type(report["false_accept_count"]) is int
    assert report["false_reject_count"] == 0
    assert type(report["false_reject_count"]) is int
    assert report["terminal_evidence_preflight_ready_score"] == 1.0
    assert report["verifier_is_oracle"] is False
    assert report["honest_verdict"].startswith("complete:")
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    report_path = mod.write_report(report, REPO, env=env)
    assert report_path == artifact_root / mod.RESULT_RELATIVE_PATH.name
    assert json.loads(report_path.read_text(encoding="utf-8")) == report

    broken = dict(report)
    broken["false_accept_count"] = 0.0
    assert "false_accept_count must be bare integer 0" in mod.validate_report(broken)
    broken = dict(report)
    broken["honest_verdict"] = "done"
    assert "honest_verdict lacks accepted Exp6298 prefix" in mod.validate_report(broken)


def test_write_report_rejects_invalid_payload(tmp_path: Path) -> None:
    """REQ-INFRA-6298: invalid reports do not get written."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    with pytest.raises(ValueError, match="invalid Exp6298 report"):
        mod.write_report({"status": "complete"}, REPO, env={ARTIFACT_ROOT_ENV: str(artifact_root)})
