"""Tests for the Exp6262 terminal-artifact readiness contract artifact.

Spec refs: REQ-INFRA-6262, SCENARIO-INFRA-6262-1,
SCENARIO-INFRA-6262-2, SCENARIO-INFRA-6262-3,
SCENARIO-INFRA-6262-4, SCENARIO-INFRA-6262-5,
SCENARIO-INFRA-6262-6.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6262_terminal_artifact_readiness_contract as mod
from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-harnesses/spec.md"


def _fake_runner(argv: tuple[str, ...], _root: Path) -> JsonDict:
    command = " ".join(argv)
    return {
        "command": command,
        "exit_code": 0,
        "classification": "passed",
        "stdout_tail": "",
        "stderr_tail": "",
    }


def _build() -> JsonDict:
    return mod.build_report(REPO, date="20260810", command_runner=_fake_runner, duration_s=1.5)


def test_req_infra_6262_spec_declares_readiness_contract() -> None:
    """REQ-INFRA-6262: OpenSpec names the evidence boundary and artifact."""

    text = SPEC_PATH.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6262") :]

    for marker in (
        "REQ-INFRA-6262",
        "SCENARIO-INFRA-6262-1",
        "SCENARIO-INFRA-6262-2",
        "SCENARIO-INFRA-6262-3",
        "SCENARIO-INFRA-6262-4",
        "SCENARIO-INFRA-6262-5",
        "SCENARIO-INFRA-6262-6",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_infra_6262_build_report_records_required_controls() -> None:
    """SCENARIO-INFRA-6262-6: report records Exp6228, controls, and hashes."""

    report = _build()

    assert mod.validate_report(report) == []
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(report)
    assert report["status"] == "complete_ready"
    assert report["terminal_artifact_contract_ready_score"] == 1
    assert type(report["terminal_artifact_contract_ready_score"]) is int
    assert report["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert report["verifier_is_oracle"] is False
    assert report["honest_verdict"].startswith("complete:")
    assert report["reproducibility_checksum"] == mod.payload_checksum(report)

    exp6228 = report["exp6228_path_hash_and_exact_classification"]
    assert exp6228["classification"]["terminal"] is False
    assert exp6228["classification"]["classification"] == "unknown"
    assert exp6228["sha256"].startswith("sha256:")

    regression = report["exp6228_regression_flag_code_and_severity"]
    assert regression["kind"] == mod.NONTERMINAL_FLAG_KIND
    assert regression["severity"] == "critical"
    assert regression["classification"] == "unknown"

    assert report["exact_path_over_receipt_precedence"]["receipt_override_attempted"] is True
    assert report["exact_path_over_receipt_precedence"]["receipt_overrode"] is False
    assert report["receipt_override_negative_control"]["eligible"] is False
    assert report["readiness_missing_negative_control"]["classification"] == "missing"
    assert report["readiness_missing_negative_control"]["severity"] == "critical"
    assert report["honest_blocked_control_result"]["classification"] == "blocked"
    assert report["honest_blocked_control_result"]["flag_count"] == 0
    assert report["gate_skip_control_result"]["classification"] == "skipped"
    assert report["gate_skip_control_result"]["flag_count"] == 0
    assert report["gate_field_eligibility_contract"]["terminal_exact_bare"]["eligible"] is True
    assert report["gate_field_eligibility_contract"]["terminal_wrapped"]["eligible"] is False
    assert report["gate_field_eligibility_contract"]["terminal_nested"]["eligible"] is False
    assert report["gate_field_eligibility_contract"]["nonterminal_exact_bare"]["eligible"] is False

    for row in report["false_positive_fixture_results"].values():
        assert row["flag_count"] == 0
        assert row["terminal"] is True

    assert report["classifier_source_hash_before_after"]["unchanged"] is True
    assert report["adversarial_verifier_source_hash_before_after"]["unchanged"] is True
    assert report["protected_files_unchanged"]["unchanged"] is True
    assert report["protected_files_unchanged"]["scripts_research_conductor_py_untouched"] is True
    assert set(report["field_principles"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(report["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)


def test_req_infra_6262_validate_report_rejects_bad_outputs() -> None:
    """REQ-INFRA-6262: validation keeps readiness score conjunctive."""

    report = _build()

    missing = deepcopy(report)
    missing.pop("status")
    assert "missing:status" in mod.validate_report(missing)

    bad_score = deepcopy(report)
    bad_score["terminal_artifact_contract_ready_score"] = "1"
    bad_score["reproducibility_checksum"] = mod.payload_checksum(bad_score)
    assert "terminal_artifact_contract_ready_score" in mod.validate_report(bad_score)

    bad_exp = deepcopy(report)
    bad_exp["exp6228_regression_flag_code_and_severity"]["severity"] = "warn"
    bad_exp["reproducibility_checksum"] = mod.payload_checksum(bad_exp)
    assert "exp6228_regression_flag_code_and_severity" in mod.validate_report(bad_exp)

    bad_terminal_control = deepcopy(report)
    bad_terminal_control["false_positive_fixture_results"]["complete"]["flag_count"] = 1
    bad_terminal_control["reproducibility_checksum"] = mod.payload_checksum(bad_terminal_control)
    assert "false_positive_fixture_results" in mod.validate_report(bad_terminal_control)

    bad_commands = deepcopy(report)
    bad_commands["focused_test_results"][0]["exit_code"] = 1
    bad_commands["test_exit_codes"][bad_commands["focused_test_results"][0]["command"]] = 1
    bad_commands["reproducibility_checksum"] = mod.payload_checksum(bad_commands)
    assert "focused_test_results" in mod.validate_report(bad_commands)

    bad_hash = deepcopy(report)
    bad_hash["classifier_source_hash_before_after"]["unchanged"] = False
    bad_hash["reproducibility_checksum"] = mod.payload_checksum(bad_hash)
    assert "classifier_source_hash_before_after" in mod.validate_report(bad_hash)

    bad_principles = deepcopy(report)
    bad_principles["field_principles"]["status"] = ""
    bad_principles["field_provenance"]["status"]["principle"] = ""
    bad_principles["reproducibility_checksum"] = mod.payload_checksum(bad_principles)
    assert "field_principles:status" in mod.validate_report(bad_principles)

    bad_verdict = deepcopy(report)
    bad_verdict["honest_verdict"] = "ready without prefix"
    bad_verdict["reproducibility_checksum"] = mod.payload_checksum(bad_verdict)
    assert "honest_verdict" in mod.validate_report(bad_verdict)

    blocked_bad_verdict = deepcopy(report)
    blocked_bad_verdict["terminal_artifact_contract_ready_score"] = 0
    blocked_bad_verdict["focused_test_results"][0]["exit_code"] = 124
    blocked_bad_verdict["test_exit_codes"][
        blocked_bad_verdict["focused_test_results"][0]["command"]
    ] = 124
    blocked_bad_verdict["status"] = "blocked"
    blocked_bad_verdict["honest_verdict"] = "timeout without terminal prefix"
    blocked_bad_verdict["reproducibility_checksum"] = mod.payload_checksum(blocked_bad_verdict)
    assert "honest_verdict" in mod.validate_report(blocked_bad_verdict)

    bad_checksum = deepcopy(report)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.validate_report(bad_checksum)


def test_scenario_infra_6262_write_contract_uses_artifact_root_override(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6262-6: writer atomically emits the required JSON."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    report = mod.write_contract(
        REPO,
        date="20260810",
        command_runner=_fake_runner,
        duration_s=1.5,
        env={ARTIFACT_ROOT_ENV: str(artifact_root)},
    )
    target = artifact_root / mod.RESULT_RELATIVE_PATH.name

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert mod.validate_report(report) == []


def test_req_infra_6262_writer_refuses_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-INFRA-6262: invalid reports are not written."""

    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    monkeypatch.setattr(mod, "validate_report", lambda _report: ["forced"])

    with pytest.raises(ValueError, match="invalid Exp6262 terminal readiness contract"):
        mod.write_contract(
            REPO,
            date="20260810",
            command_runner=_fake_runner,
            env={ARTIFACT_ROOT_ENV: str(artifact_root)},
        )
    assert not (artifact_root / mod.RESULT_RELATIVE_PATH.name).exists()
