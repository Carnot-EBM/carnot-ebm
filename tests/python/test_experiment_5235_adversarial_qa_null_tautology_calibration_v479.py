"""Tests for Exp 5235 artifact-QA null/tautology calibration.

Spec refs: REQ-REPORT-5235, SCENARIO-REPORT-5235,
SCENARIO-REPORT-5235-COMPUTE-BOUND-GUARD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.adversarial_verify as av
from carnot import experiment_5235_adversarial_qa_null_tautology_calibration_v479 as mod


JsonDict = dict[str, Any]
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_payload(tmp_path: Path, name: str, payload: JsonDict) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report(tmp_path: Path, name: str, payload: JsonDict) -> JsonDict:
    return av.verify_artifact(_write_payload(tmp_path, name, payload))


def _flag_kinds(report: JsonDict, *, severity: str | None = None) -> set[str]:
    return {
        str(flag["kind"])
        for flag in report["flags"]
        if severity is None or flag["severity"] == severity
    }


def test_req_report_5235_spec_declares_calibration_contract() -> None:
    """REQ-REPORT-5235: OpenSpec anchors the calibration artifact and fixtures."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5235") : spec.index("REQ-REPORT-5221")]

    for marker in (
        "REQ-REPORT-5235",
        "SCENARIO-REPORT-5235",
        "SCENARIO-REPORT-5235-COMPUTE-BOUND-GUARD",
        str(mod.RESULT_RELATIVE_PATH),
        "canonical_pool_n == regenerated_rows",
        "n_scored == ties",
        "DURATION_TOO_SHORT",
        "METHODOLOGY_MISSING",
        "artifact_qa_lint_tests",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5235_builder_equality_is_not_critical_when_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5235: documented builder counts are structural equality."""

    report = _report(tmp_path, "builder", mod.expected_builder_equality_fixture())

    assert "TAUTOLOGY" not in _flag_kinds(report, severity="critical")


def test_scenario_report_5235_all_ties_null_is_not_critical_when_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5235: all-ties GAP-4 null counts are not copied metrics."""

    report = _report(tmp_path, "all_ties", mod.expected_all_ties_null_fixture())

    assert "TAUTOLOGY" not in _flag_kinds(report, severity="critical")
    assert "IMPLAUSIBLE_PERFECT" not in _flag_kinds(report, severity="critical")


def test_scenario_report_5235_suspicious_duplicate_scalars_still_flag(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5235: unrelated duplicate scalar metrics still quarantine."""

    report = _report(tmp_path, "duplicate", mod.suspicious_duplicate_scalar_fixture())

    assert "TAUTOLOGY" in _flag_kinds(report, severity="critical")


def test_scenario_report_5235_compute_bound_methodology_checks_are_preserved(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5235-COMPUTE-BOUND-GUARD: GGUF receipts remain required."""

    report = _report(tmp_path, "compute_bound", mod.compute_bound_missing_receipts_fixture())
    all_kinds = _flag_kinds(report)

    assert "DURATION_TOO_SHORT" in _flag_kinds(report, severity="critical")
    assert "METHODOLOGY_MISSING" in all_kinds


def test_req_report_5235_evaluates_all_fixture_reports_and_builds_artifact(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5235: artifact records a clean calibration decision."""

    reports = mod.evaluate_fixture_reports(tmp_path / "fixtures")
    artifact = mod.build_artifact(
        fixture_reports=reports,
        validation_commands_run=["unit fixture evaluation: PASS"],
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    assert artifact["qa_calibration_passed"] is True
    assert artifact["structural_null_rules_documented"] is True
    assert artifact["duration_methodology_checks_preserved"] is True
    assert artifact["gap4_reclassification_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_added_or_updated"] == [str(mod.TEST_RELATIVE_PATH)]


def test_req_report_5235_write_outputs_writes_valid_json(tmp_path: Path) -> None:
    """REQ-REPORT-5235: write_outputs emits the terminal result artifact."""

    artifact = mod.write_outputs(
        root=tmp_path,
        fixture_dir=tmp_path / "qa_fixtures",
        validation_commands_run=["write_outputs fixture: PASS"],
        duration_s=0.5,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(written)


def test_req_report_5235_terminal_artifact_lint_substrate_is_not_compute_bound(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5235: embedded GGUF fixture reports do not make Exp 5235 compute-bound."""

    artifact = mod.write_outputs(
        root=tmp_path,
        fixture_dir=tmp_path / "qa_fixtures",
        validation_commands_run=["write_outputs fixture: PASS"],
        duration_s=0.5,
    )
    report = av.verify_artifact(tmp_path / mod.RESULT_RELATIVE_PATH)

    assert artifact["inference_substrate"] == "artifact_qa_lint_tests"
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_report_5235_validation_rejects_wrapped_gate_and_bad_substrate(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-5235: required gate fields stay bare and substrate-specific."""

    artifact = mod.build_artifact(
        fixture_reports=mod.evaluate_fixture_reports(tmp_path / "fixtures"),
        validation_commands_run=["unit fixture evaluation: PASS"],
        duration_s=0.5,
    )

    wrapped = dict(artifact)
    wrapped["qa_calibration_passed"] = {"value": True, "principle": "bad"}
    with pytest.raises(ValueError, match="qa_calibration_passed_bare_bool"):
        mod.validate_artifact(wrapped)

    bad_substrate = dict(artifact)
    bad_substrate["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    missing_guard = dict(artifact)
    missing_guard["duration_methodology_checks_preserved"] = False
    with pytest.raises(ValueError, match="duration_methodology_checks_preserved"):
        mod.validate_artifact(missing_guard)


def test_req_report_5235_validation_rejects_each_contract_break(tmp_path: Path) -> None:
    """REQ-REPORT-5235: validation fails closed on malformed calibration artifacts."""

    artifact = mod.build_artifact(
        fixture_reports=mod.evaluate_fixture_reports(tmp_path / "fixtures"),
        validation_commands_run=["unit fixture evaluation: PASS"],
        duration_s=0.5,
    )

    broken = dict(artifact)
    broken.pop("schema")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = "calibration passed"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["gap4_reclassification_ready"] = False
    with pytest.raises(ValueError, match="gap4_reclassification_ready"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["tests_added_or_updated"] = []
    with pytest.raises(ValueError, match="tests_added_or_updated"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["validation_commands_run"] = [object()]
    with pytest.raises(ValueError, match="validation_commands_run"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["calibration_checks"] = {"qa_calibration_passed": False}
    with pytest.raises(ValueError, match="calibration_checks"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(broken)


def test_req_report_5235_failed_calibration_branch_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-5235: failed-calibration verdict stays terminal and CLI writes JSON."""

    reports = mod.evaluate_fixture_reports(tmp_path / "fixtures")
    failed = mod.build_artifact(
        fixture_reports=reports,
        validation_commands_run=["unit fixture evaluation: PASS"],
        research_conductor_py_untouched_confirmed=False,
    )
    assert failed["qa_calibration_passed"] is False
    assert failed["honest_verdict"].startswith("complete:")

    assert (
        mod.main(
            [
                "--root",
                str(tmp_path),
                "--fixture-dir",
                str(tmp_path / "main-fixtures"),
                "--duration-s",
                "0.2",
                "--validation-command",
                "main cli: PASS",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["validation_commands_run"] == ["main cli: PASS"]
    mod.validate_artifact(printed)
