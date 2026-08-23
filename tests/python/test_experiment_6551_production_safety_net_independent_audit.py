"""Tests for Exp6551 production Safety-Net independent audit.

Spec refs: REQ-REPORT-6551, REQ-REPORT-6551-REPLAY,
REQ-REPORT-6551-MISSING, REQ-REPORT-6551-DISABLED,
REQ-REPORT-6551-PARITY, REQ-REPORT-6551-EXACT,
REQ-REPORT-6551-FALLBACK, REQ-REPORT-6551-ROWS,
REQ-REPORT-6551-ATOMIC, SCENARIO-REPORT-6551-CLEAN,
SCENARIO-REPORT-6551-BLOCKED.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6551_production_safety_net_independent_audit as mod


TESTS_RUN = [{"command": "focused-exp6551", "exit_code": 0}]


def test_req_report_6551_spec_declares_independent_audit_contract() -> None:
    """REQ-REPORT-6551: OpenSpec owns the independent audit contract."""

    text = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6551") :]
    for marker in (
        "REQ-REPORT-6551-REPLAY",
        "REQ-REPORT-6551-MISSING",
        "REQ-REPORT-6551-DISABLED",
        "REQ-REPORT-6551-PARITY",
        "REQ-REPORT-6551-EXACT",
        "REQ-REPORT-6551-FALLBACK",
        "REQ-REPORT-6551-ROWS",
        "REQ-REPORT-6551-ATOMIC",
        "SCENARIO-REPORT-6551-CLEAN",
        "SCENARIO-REPORT-6551-BLOCKED",
    ):
        assert marker in section


def test_scenario_report_6551_clean_artifact_recomputes_ready_score(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6551-CLEAN: independent rows derive readiness."""

    artifact = mod.build_artifact(
        result_path=tmp_path / "experiment_6551.json",
        write=False,
        duration_s=0.001,
        tests_run=TESTS_RUN,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_production_safety_net_independent_audit_null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "null"
    assert artifact["production_safety_net_audited_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["input_existence_and_hash_receipts"]["all_required_inputs_present"] is True
    assert artifact["independent_build_identity_receipt"]["native_code_ran"] is True
    assert artifact["independent_exact_equality_receipt"]["all_exact_outputs_equal"] is True
    assert artifact["fallback_exception_and_rollback_audit"]["fallback_reachable"] is True
    assert artifact["fallback_exception_and_rollback_audit"]["rollback_restores_disabled"] is True
    assert artifact["shortcut_attack_matrix"]["all_attacks_fail_closed"] is True
    assert artifact["aggregate_row_recomputation"]["ready_score_from_rows"] == 1.0
    assert artifact["aggregate_row_recomputation"]["verdict_class_from_rows"] == "null"
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_report_6551_rows_cover_disabled_fallback_and_parity() -> None:
    """REQ-REPORT-6551-REPLAY: rows cover identity, parity, and fail-closed cases."""

    artifact = mod.build_artifact(write=False, duration_s=0.001, tests_run=TESTS_RUN)
    identity_rows = artifact["independent_disabled_identity_rows"]
    rows = artifact["independent_enabled_and_parity_rows"]
    conditions = {row["condition"] for row in rows}

    assert identity_rows
    assert rows == artifact["per_unit_rows"]
    assert all(row["serialized_request_bytes_equal"] for row in identity_rows)
    assert all(row["candidate_order_equal"] for row in identity_rows)
    assert all(row["outputs_equal"] for row in identity_rows)
    assert all(row["persistence_equal"] for row in identity_rows)
    assert {
        "compact_route",
        "boundary_abstention",
        "exception_lookup",
        "forced_fallback",
        "malformed_duplicate",
        "timeout",
        "stale_configuration",
        "nan_invalid_json",
    } <= conditions
    assert all(row["python_rust_decision_equal"] for row in rows)
    assert all(row["python_rust_decision_bytes_equal"] for row in rows)
    assert any(row["fallback_reason"] == "abstention" for row in rows)
    assert any(row["exception_hit"] for row in rows)
    assert all(row["candidate_preserved"] for row in rows)
    assert all(row["exact_output_equal_to_native"] for row in rows)


def test_scenario_report_6551_missing_inputs_block_with_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6551-BLOCKED: absent upstream inputs are terminal blocked."""

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "blocked.json",
        write=False,
        duration_s=0.001,
        tests_run=TESTS_RUN,
    )

    assert artifact["status"] == "blocked_production_safety_net_independent_audit"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["production_safety_net_audited_ready_score"] == 0.0
    assert artifact["input_existence_and_hash_receipts"]["all_required_inputs_present"] is False
    assert artifact["missing_input_disposition"]["terminal_disposition"] == "blocked"
    assert artifact["missing_input_disposition"]["missing_paths"]
    assert artifact["per_unit_rows"] == []
    assert artifact["gate_check_summary"]["all_gates_passed"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_report_6551_validation_detects_tampering_and_missing_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-6551-ROWS: validation rejects row and build tampering."""

    artifact = mod.build_artifact(write=False, duration_s=0.001, tests_run=TESTS_RUN)

    changed_row = deepcopy(artifact)
    changed_row["per_unit_rows"][0]["candidate_preserved"] = False
    changed_row["independent_enabled_and_parity_rows"] = changed_row["per_unit_rows"]
    changed_row["reproducibility_checksum"] = mod.reproducibility_checksum(changed_row)
    assert "aggregate recomputation mismatch" in mod.validate_artifact(changed_row)

    bad_score = deepcopy(artifact)
    bad_score["production_safety_net_audited_ready_score"] = 1.0
    bad_score["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    assert "ready score mismatch" in mod.validate_artifact(bad_score)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad_checksum)

    wrong_ready_class = deepcopy(artifact)
    wrong_ready_class["verdict_class"] = "partial"
    wrong_ready_class["reproducibility_checksum"] = mod.reproducibility_checksum(wrong_ready_class)
    assert "ready audit must use verdict_class null" in mod.validate_artifact(wrong_ready_class)

    blocked_nonzero = deepcopy(artifact)
    blocked_nonzero["verdict_class"] = "blocked"
    blocked_nonzero["reproducibility_checksum"] = mod.reproducibility_checksum(blocked_nonzero)
    assert "blocked verdict requires zero ready score" in mod.validate_artifact(blocked_nonzero)

    bad_identity = deepcopy(artifact)
    bad_identity["verdict_class"] = "partial"
    bad_identity["production_safety_net_audited_ready_score"] = 0.0
    bad_identity["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_identity["aggregate_row_recomputation"]["disabled_identity_exact"] = False
    bad_identity["reproducibility_checksum"] = mod.reproducibility_checksum(bad_identity)
    assert "disabled identity failed" in mod.validate_artifact(bad_identity)

    bad_parity = deepcopy(artifact)
    bad_parity["verdict_class"] = "partial"
    bad_parity["production_safety_net_audited_ready_score"] = 0.0
    bad_parity["aggregate_row_recomputation"]["ready_score_from_rows"] = 0.0
    bad_parity["aggregate_row_recomputation"]["python_rust_parity"] = False
    bad_parity["reproducibility_checksum"] = mod.reproducibility_checksum(bad_parity)
    assert "Python/Rust parity failed" in mod.validate_artifact(bad_parity)

    disqualified = deepcopy(artifact)
    disqualified["independent_exact_equality_receipt"]["all_exact_outputs_equal"] = False
    aggregate = mod.aggregate_row_recomputation(disqualified)
    assert aggregate["verdict_class_from_rows"] == "disqualified"
    assert mod._status_and_verdict(aggregate)[2] == "disqualified"  # noqa: SLF001

    partial = deepcopy(artifact)
    partial["shortcut_attack_matrix"]["all_attacks_fail_closed"] = False
    aggregate = mod.aggregate_row_recomputation(partial)
    assert aggregate["verdict_class_from_rows"] == "partial"
    assert mod._status_and_verdict(aggregate)[2] == "partial"  # noqa: SLF001

    def _raise_import(_name: str) -> object:
        raise ImportError("forced")

    monkeypatch.setattr(mod.importlib, "import_module", _raise_import)
    blocked = mod.build_artifact(write=False, duration_s=0.001, tests_run=TESTS_RUN)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["missing_input_disposition"]["missing_tools"] == ["carnot._rust"]
    assert mod.validate_artifact(blocked) == []


def test_req_report_6551_defensive_helpers_and_cli(tmp_path: Path) -> None:
    """REQ-REPORT-6551-ATOMIC: helpers and CLI validation are explicit."""

    assert mod.sha256_file(tmp_path / "missing") == "missing"
    assert mod._read_json(tmp_path / "missing.json") == {}  # noqa: SLF001
    bad_json = tmp_path / "bad-json.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}  # noqa: SLF001
    missing_command = mod._command_version(  # noqa: SLF001
        tmp_path,
        ["definitely-missing-carnot-exp6551-binary"],
    )
    assert missing_command["available"] is False
    assert (
        mod._candidate_ids_from_case(  # noqa: SLF001
            {"payload": {"candidate_ids": None}}
        )
        == ()
    )
    assert mod.independent_enabled_and_parity_rows(None) == []

    original_adapter_config = mod._adapter_config_for_case  # noqa: SLF001
    try:
        mod._adapter_config_for_case = (  # type: ignore[assignment]  # noqa: SLF001
            lambda _case: mod.SafetyNetRouterConfig(enabled=False)
        )
        disabled_decision, _time_s, _before, _after = mod._adapter_decision_for_case(  # noqa: SLF001
            mod._case_inputs()[0]  # noqa: SLF001
        )
        assert disabled_decision["route"] == "disabled"
    finally:
        mod._adapter_config_for_case = original_adapter_config  # type: ignore[assignment]  # noqa: SLF001

    result_path = tmp_path / "cli-exp6551.json"
    assert mod.main(["--date", "20260823", "--result-path", str(result_path)]) == 0
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written["status"] == "complete_production_safety_net_independent_audit_null"
    assert mod.main(["--validate", "--result-path", str(result_path)]) == 0

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}\n", encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1

    original_build = mod.build_artifact
    try:
        mod.build_artifact = lambda **_kwargs: {"bad": "artifact"}  # type: ignore[assignment]
        assert (
            mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad-build.json")]) == 1
        )
    finally:
        mod.build_artifact = original_build  # type: ignore[assignment]
