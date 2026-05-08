"""Tests for Exp 1520 runtime-contract E2E harness.

Spec: REQ-VERIFY-1520, SCENARIO-VERIFY-1520.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import runtime_contract_e2e_harness as exp


def test_req_verify_1520_normalizer_preserves_explicit_labels_only() -> None:
    """REQ-VERIFY-1520: normalized rows carry component results and explicit labels."""

    rows = exp.normalize_contract_cases(
        safe_dsl_rows=[
            {
                "row_type": "candidate",
                "candidate_name": "compiled_candidate_that_is_not_a_contract_case",
            },
            {
                "row_type": "selected_set_summary",
                "candidate_names": ["certificate_transcript_consistency"],
                "accepted_labeled_row_ids": ["certificate:case-ok:trigger_certificate:0"],
                "false_accept_row_ids": ["certificate:case-bad:trigger_certificate:1"],
                "verifier_false_accept_rate": 0.5,
            },
        ],
        grammar_certificate_rows=[
            {
                "case_id": "case-ok",
                "decoder_mode": "trigger_grammar",
                "model_output": "model prose plus certificate",
                "parser_result": {"parsed": True},
                "deterministic_validation_passed": True,
                "false_accept_status": False,
                "verifier_result": {"accepted": True, "base_valid": True, "false_accept": False},
            }
        ],
        monitor_event_rows=[
            {
                "event_id": "runtime-1509-000001",
                "case_id": "case-ok",
                "event_kind": "monitor_decision",
                "validation_status": "pass",
                "verifier_false_accept": False,
            }
        ],
        structural_contract_rows=[
            {
                "graph_id": "case-bad:injected_violation:1",
                "case_id": "case-bad",
                "contract_family": "acquisition_path",
                "expected_violation": True,
                "detected_violation": True,
                "classifier_outcome": "true_positive",
            }
        ],
        source_paths={
            "safe_dsl": Path("safe.jsonl"),
            "grammar_certificate": Path("cert.jsonl"),
            "monitor_event": Path("monitor.jsonl"),
            "structural_contract": Path("structural.jsonl"),
        },
    )

    assert [row["contract_case_id"] for row in rows] == [
        "safe_dsl:certificate:case-ok:trigger_certificate:0",
        "safe_dsl:certificate:case-bad:trigger_certificate:1",
        "grammar_certificate:case-ok:trigger_grammar:1",
        "monitor_event:runtime-1509-000001",
        "structural_contract:case-bad:injected_violation:1:acquisition_path:1",
    ]
    assert rows[0]["safe_dsl_verifier_result"]["linked"] is True
    assert rows[0]["expected_label"] is True
    assert rows[1]["expected_label"] is False
    assert rows[2]["certificate_parse_result"]["parsed"] is True
    assert rows[2]["final_deterministic_accept"] is True
    assert rows[3]["expected_label"] is None
    assert rows[3]["monitor_event_result"]["validation_status"] == "pass"
    assert rows[4]["structural_contract_result"]["detected_violation"] is True
    assert rows[4]["final_deterministic_accept"] is False
    assert set(exp.REQUIRED_CONTRACT_CASE_FIELDS) <= set(rows[0])


def test_req_verify_1520_false_accept_ledger_ignores_unlabeled_rows() -> None:
    """REQ-VERIFY-1520: false accepts/rejects are counted only on explicit labels."""

    ledger = exp.compute_false_accept_ledger(
        [
            {"expected_label": False, "final_deterministic_accept": True},
            {"expected_label": False, "final_deterministic_accept": False},
            {"expected_label": True, "final_deterministic_accept": False},
            {"expected_label": None, "final_deterministic_accept": True},
            {"final_deterministic_accept": True},
        ]
    )

    assert ledger["explicit_label_count"] == 3
    assert ledger["explicit_reject_count"] == 2
    assert ledger["false_accept_count"] == 1
    assert ledger["false_accept_rate"] == pytest.approx(0.5)
    assert ledger["false_reject_count"] == 1


def test_scenario_verify_1520_runner_writes_ready_manifest_and_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1520: runner links all .116 families into one manifest."""

    paths = _source_paths(tmp_path)
    _write_source_files(paths)

    artifact = exp.run_runtime_contract_e2e_harness(
        output_path=paths["output"],
        manifest_path=paths["manifest"],
        safe_dsl_artifact_path=paths["safe_artifact"],
        safe_dsl_manifest_path=paths["safe_manifest"],
        grammar_certificate_artifact_path=paths["certificate_artifact"],
        grammar_certificate_manifest_path=tmp_path / "legacy_certificate_alias_missing.jsonl",
        monitor_artifact_path=paths["monitor_artifact"],
        monitor_event_manifest_path=tmp_path / "legacy_monitor_alias_missing.jsonl",
        structural_contract_artifact_path=paths["structural_artifact"],
        structural_contract_manifest_path=paths["structural_manifest"],
        product_line_artifact_path=paths["product_line_artifact"],
        focused_tests_passed=True,
    )
    manifest_rows = _read_jsonl(paths["manifest"])

    assert json.loads(paths["output"].read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["runtime_contract_e2e_ready"] is True
    assert artifact["source_artifacts_loaded"] is True
    assert artifact["contract_cases_total"] == 4
    assert artifact["safe_dsl_cases_linked"] == 1
    assert artifact["grammar_certificate_cases_linked"] == 1
    assert artifact["monitor_events_linked"] == 1
    assert artifact["structural_contract_cases_linked"] == 1
    assert artifact["false_accept_count"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_reject_count"] == 0
    assert artifact["focused_tests_passed"] is True
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(manifest_rows) == artifact["contract_cases_total"] + 1
    assert manifest_rows[-1]["row_type"] == "summary"
    assert manifest_rows[-1]["false_accept_rate"] == pytest.approx(0.0)
    assert manifest_rows[-1]["source_artifacts_loaded"] is True


def test_req_verify_1520_missing_required_sources_write_terminal_blocker(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1520: unresolved source artifacts produce exact missing-path blockers."""

    output = tmp_path / "experiment_1520.json"
    manifest = tmp_path / "manifest.jsonl"
    missing_safe_artifact = tmp_path / "missing_exp1507.json"
    missing_safe_manifest = tmp_path / "missing_safe.jsonl"

    artifact = exp.run_runtime_contract_e2e_harness(
        output_path=output,
        manifest_path=manifest,
        safe_dsl_artifact_path=missing_safe_artifact,
        safe_dsl_manifest_path=missing_safe_manifest,
        grammar_certificate_artifact_path=tmp_path / "missing_exp1508.json",
        grammar_certificate_manifest_path=tmp_path / "missing_certificate.jsonl",
        monitor_artifact_path=tmp_path / "missing_exp1509.json",
        monitor_event_manifest_path=tmp_path / "missing_monitor.jsonl",
        structural_contract_artifact_path=tmp_path / "missing_exp1510.json",
        structural_contract_manifest_path=tmp_path / "missing_structural.jsonl",
        product_line_artifact_path=tmp_path / "missing_exp1511.json",
        focused_tests_passed=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["runtime_contract_e2e_ready"] is False
    assert artifact["source_artifacts_loaded"] is False
    assert f"missing_safe_dsl_artifact:{missing_safe_artifact}" in artifact["blockers"]
    assert f"missing_safe_dsl_manifest:{missing_safe_manifest}" in artifact["blockers"]
    assert manifest.read_text(encoding="utf-8") == ""
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"].startswith("complete:")


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "output": tmp_path / "experiment_1520.json",
        "manifest": tmp_path / "runtime_contract_manifest.jsonl",
        "safe_artifact": tmp_path / "experiment_1507.json",
        "safe_manifest": tmp_path / "safe_1507.jsonl",
        "certificate_artifact": tmp_path / "experiment_1508.json",
        "certificate_manifest": tmp_path / "certificates_1508.jsonl",
        "monitor_artifact": tmp_path / "experiment_1509.json",
        "monitor_manifest": tmp_path / "monitor_1509.jsonl",
        "structural_artifact": tmp_path / "experiment_1510.json",
        "structural_manifest": tmp_path / "structural_1510.jsonl",
        "product_line_artifact": tmp_path / "experiment_1511.json",
    }


def _write_source_files(paths: dict[str, Path]) -> None:
    _write_json(
        paths["safe_artifact"],
        {
            "status": "complete",
            "verifier_induction_ready": True,
            "induction_manifest_path": str(paths["safe_manifest"]),
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_json(
        paths["certificate_artifact"],
        {
            "status": "complete",
            "certificate_decoder_ready": True,
            "decoder_manifest_path": str(paths["certificate_manifest"]),
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_json(
        paths["monitor_artifact"],
        {
            "status": "complete",
            "monitor_runtime_ready": True,
            "monitor_event_manifest_path": str(paths["monitor_manifest"]),
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_json(
        paths["structural_artifact"],
        {
            "status": "complete",
            "structural_contract_gate_ready": True,
            "contract_manifest_path": str(paths["structural_manifest"]),
            "false_accept_rate": 0.0,
        },
    )
    _write_json(
        paths["product_line_artifact"],
        {
            "status": "complete",
            "product_line_benchmark_ready": False,
            "verifier_false_accept_rate": 0.0,
        },
    )
    _write_jsonl(
        paths["safe_manifest"],
        [
            {
                "row_type": "selected_set_summary",
                "candidate_names": ["safe"],
                "accepted_labeled_row_ids": ["certificate:case-ok:trigger_certificate:0"],
                "false_accept_row_ids": [],
                "verifier_false_accept_rate": 0.0,
            }
        ],
    )
    _write_jsonl(
        paths["certificate_manifest"],
        [
            {
                "case_id": "case-ok",
                "decoder_mode": "trigger_grammar",
                "model_output": "bounded certificate",
                "parser_result": {"parsed": True},
                "deterministic_validation_passed": True,
                "false_accept_status": False,
                "verifier_result": {"accepted": True, "base_valid": True, "false_accept": False},
            }
        ],
    )
    _write_jsonl(
        paths["monitor_manifest"],
        [
            {
                "event_id": "runtime-1509-000001",
                "case_id": "case-ok",
                "event_kind": "monitor_decision",
                "validation_status": "pass",
                "verifier_false_accept": False,
            }
        ],
    )
    _write_jsonl(
        paths["structural_manifest"],
        [
            {
                "graph_id": "case-ok:known_good:0",
                "case_id": "case-ok",
                "contract_family": "acquisition_path",
                "expected_violation": False,
                "detected_violation": False,
                "classifier_outcome": "true_negative",
            }
        ],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
