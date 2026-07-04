"""Tests for Exp 5236 GAP-4 clean status after QA calibration.

Spec refs: REQ-REPORT-5236, SCENARIO-REPORT-5236-CLEAN-NULL,
SCENARIO-REPORT-5236-BLOCKED-RECHECK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5236_gap4_clean_status_after_qa_calibration_v479 as mod


JsonDict = dict[str, Any]
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _qa_artifact(*, passed: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_5235_adversarial_qa_null_tautology_calibration_v479",
        "experiment_id": 5235,
        "result_path": str(mod.EXP5235_RELATIVE_PATH),
        "qa_calibration_passed": passed,
        "gap4_reclassification_ready": passed,
        "duration_methodology_checks_preserved": True,
        "honest_verdict": "complete: QA calibration passed",
    }


def _pool_artifact(*, n: int = 120, usable: bool = True) -> JsonDict:
    return {
        "experiment": "experiment_5224_gap4_canonical_pool_builder_v478",
        "experiment_id": 5224,
        "result_path": str(mod.EXP5224_RELATIVE_PATH),
        "gap4_canonical_pool_usable": usable,
        "canonical_pool_n": n,
        "candidate_rows": [{"candidate_id": f"row-{idx:03d}"} for idx in range(n)],
        "protocol_fields_complete": True,
        "adversarial_verify_passed": True,
        "honest_verdict": "success: canonical GAP-4 pool usable for validation with n=120",
    }


def _validation_artifact(
    *,
    wins: int = 0,
    losses: int = 0,
    ties: int = 120,
    positive: bool = False,
) -> JsonDict:
    return {
        "experiment": "experiment_5225_gap4_clean_scale_validation_gated_v478",
        "experiment_id": 5225,
        "result_path": str(mod.EXP5225_RELATIVE_PATH),
        "gap4_clean_validation_complete": True,
        "n_scored": wins + losses + ties,
        "canonical_pool_n": 120,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "exact_test_p_value": 0.03125 if positive else 1.0,
        "exact_test_passes_min6_rule": positive,
        "effect_direction": "positive" if positive else "null",
        "precondition_errors": [],
        "adversarial_verify_passed": True,
        "honest_verdict": "complete: clean GAP-4 validation null decision",
    }


def _schema_pass() -> list[JsonDict]:
    return [
        {"name": "exp5224_artifact_schema_errors", "path": str(mod.EXP5224_RELATIVE_PATH), "passed": True, "errors": []},
        {"name": "exp5225_artifact_schema_errors", "path": str(mod.EXP5225_RELATIVE_PATH), "passed": True, "errors": []},
    ]


def _adversarial_pass() -> list[JsonDict]:
    return [
        {
            "name": "adversarial_verify",
            "path": str(mod.EXP5224_RELATIVE_PATH),
            "passed": True,
            "loaded": True,
            "flag_count": 0,
            "flags": [],
        },
        {
            "name": "adversarial_verify",
            "path": str(mod.EXP5225_RELATIVE_PATH),
            "passed": True,
            "loaded": True,
            "flag_count": 0,
            "flags": [],
        },
    ]


def test_req_report_5236_spec_declares_reclassification_contract() -> None:
    """REQ-REPORT-5236: OpenSpec anchors the frozen-artifact reclassification."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("REQ-REPORT-5236") : spec.index("REQ-REPORT-5221")]

    for marker in (
        "REQ-REPORT-5236",
        "SCENARIO-REPORT-5236-CLEAN-NULL",
        "SCENARIO-REPORT-5236-BLOCKED-RECHECK",
        str(mod.RESULT_RELATIVE_PATH),
        "clean_null",
        "blocked_flagged",
        "frozen_gap4_artifact_reclassification",
        "pool_regenerated",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5236_clean_null_reclassifies_frozen_artifacts() -> None:
    """SCENARIO-REPORT-5236-CLEAN-NULL: clean all-ties validation becomes clean_null."""

    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(),
        schema_reports=_schema_pass(),
        adversarial_reports=_adversarial_pass(),
        qa_recheck_commands=["adversarial_verify exp5224 exp5225: PASS"],
        duration_s=0.5,
    )

    mod.validate_artifact(artifact)
    assert artifact["gap4_status_decision"] == "clean_null"
    assert artifact["gap4_headline_eligible"] is False
    assert artifact["canonical_pool_n"] == 120
    assert artifact["wins"] == 0
    assert artifact["losses"] == 0
    assert artifact["ties"] == 120
    assert artifact["pool_regenerated"] is False
    assert artifact["ops_docs_updated"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["remaining_blocker"] is None
    assert "clean-null" in artifact["honest_verdict"]


def test_req_report_5236_clean_positive_is_only_headline_eligible_clean_decision() -> None:
    """REQ-REPORT-5236: headline eligibility requires the unchanged min-six positive."""

    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(wins=6, losses=0, ties=114, positive=True),
        schema_reports=_schema_pass(),
        adversarial_reports=_adversarial_pass(),
        qa_recheck_commands=["adversarial_verify exp5224 exp5225: PASS"],
    )

    mod.validate_artifact(artifact)
    assert artifact["gap4_status_decision"] == "clean_positive"
    assert artifact["gap4_headline_eligible"] is True
    assert artifact["wins"] == 6
    assert "clean-positive" in artifact["honest_verdict"]


def test_scenario_report_5236_blocked_flagged_records_exact_flag() -> None:
    """SCENARIO-REPORT-5236-BLOCKED-RECHECK: adversarial flags keep GAP-4 blocked."""

    flagged_reports = _adversarial_pass()
    flagged_reports[1] = {
        "name": "adversarial_verify",
        "path": str(mod.EXP5225_RELATIVE_PATH),
        "passed": False,
        "loaded": True,
        "flag_count": 1,
        "flags": [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "copied metric"}],
    }
    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(),
        schema_reports=_schema_pass(),
        adversarial_reports=flagged_reports,
        qa_recheck_commands=["adversarial_verify exp5224 exp5225: FAIL"],
    )

    mod.validate_artifact(artifact)
    assert artifact["gap4_status_decision"] == "blocked_flagged"
    assert artifact["gap4_headline_eligible"] is False
    assert "TAUTOLOGY" in artifact["remaining_blocker"]
    assert "still blocked" in artifact["honest_verdict"]


def test_scenario_report_5236_blocked_missing_receipts_fails_closed() -> None:
    """SCENARIO-REPORT-5236-BLOCKED-RECHECK: missing calibration/schema receipts block."""

    schema_reports = _schema_pass()
    schema_reports[0] = {
        "name": "exp5224_artifact_schema_errors",
        "path": str(mod.EXP5224_RELATIVE_PATH),
        "passed": False,
        "errors": ["canonical_pool_n"],
    }
    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(passed=False),
        pool_artifact=_pool_artifact(n=119),
        validation_artifact=_validation_artifact(),
        schema_reports=schema_reports,
        adversarial_reports=_adversarial_pass(),
        qa_recheck_commands=["schema recheck: FAIL"],
    )

    mod.validate_artifact(artifact)
    assert artifact["gap4_status_decision"] == "blocked_missing_receipts"
    assert artifact["gap4_headline_eligible"] is False
    assert "qa_calibration_not_passed" in artifact["remaining_blocker"]
    assert "canonical_pool_n_not_120" in artifact["remaining_blocker"]


def test_req_report_5236_write_outputs_reads_sources_and_writes_json(tmp_path: Path) -> None:
    """REQ-REPORT-5236: write_outputs emits the terminal reclassification artifact."""

    _write_json(tmp_path / mod.EXP5235_RELATIVE_PATH, _qa_artifact())
    _write_json(tmp_path / mod.EXP5224_RELATIVE_PATH, _pool_artifact())
    _write_json(tmp_path / mod.EXP5225_RELATIVE_PATH, _validation_artifact())

    artifact = mod.write_outputs(
        root=tmp_path,
        schema_rechecker=lambda _root, _pool, _validation: _schema_pass(),
        adversarial_rechecker=lambda _root: _adversarial_pass(),
        qa_recheck_commands=["manual CLI adversarial recheck: PASS"],
        duration_s=0.25,
    )

    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["gap4_status_decision"] == "clean_null"
    assert artifact["qa_calibration_artifact_path"] == str(mod.EXP5235_RELATIVE_PATH)
    assert artifact["qa_recheck_commands"][-1] == "manual CLI adversarial recheck: PASS"
    assert any("exp5224_artifact_schema_errors: PASS" in item for item in artifact["qa_recheck_commands"])
    mod.validate_artifact(written)


def test_req_report_5236_default_rechecks_and_missing_source_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-5236: default recheck helpers fail closed on missing receipts."""

    assert mod.load_source_artifacts(tmp_path / "absent") == ({}, {}, {})
    assert mod._file_sha256(tmp_path / "missing.json") is None
    assert mod._as_int(True) is None

    missing_schema = mod.default_schema_recheck(tmp_path, {}, {})
    malformed_schema = mod.default_schema_recheck(
        tmp_path,
        {"candidate_rows": [{"candidate_id": "bad"}]},
        {"experiment": "bad"},
    )
    adversarial = mod.default_adversarial_recheck(tmp_path)

    assert missing_schema[0]["errors"] == ["candidate_rows_missing"]
    assert missing_schema[1]["errors"] == ["exp5225_artifact_missing"]
    assert malformed_schema[0]["passed"] is False
    assert malformed_schema[1]["passed"] is False
    assert [report["loaded"] for report in adversarial] == [False, False]
    assert all(report["passed"] is False for report in adversarial)


def test_req_report_5236_receipt_blockers_cover_malformed_metrics() -> None:
    """REQ-REPORT-5236: malformed source metrics produce explicit receipt blockers."""

    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(usable=False),
        validation_artifact={
            "canonical_pool_n": 121,
            "gap4_clean_validation_complete": False,
            "precondition_errors": ["blocked"],
            "n_scored": None,
            "wins": "0",
            "losses": "0",
            "ties": "120",
        },
        schema_reports=[
            {
                "name": "schema_without_list_errors",
                "path": "x",
                "passed": False,
                "errors": "bad",
            }
        ],
        adversarial_reports=[
            {
                "name": "adversarial_verify",
                "path": str(mod.EXP5225_RELATIVE_PATH),
                "passed": False,
                "loaded": False,
                "flag_count": 0,
                "flags": [],
            }
        ],
    )

    blocker = artifact["remaining_blocker"]
    assert artifact["gap4_status_decision"] == "blocked_missing_receipts"
    for marker in (
        "exp5224_pool_not_usable",
        "exp5225_clean_validation_not_complete",
        "exp5225_precondition_errors_present",
        "exp5224_exp5225_pool_n_mismatch",
        "n_scored_missing",
        "wins_missing",
        "losses_missing",
        "ties_missing",
        "schema_without_list_errors: failed",
        "adversarial_recheck_missing_or_failed",
    ):
        assert marker in blocker


def test_req_report_5236_validation_rejects_wrapped_fields_and_bad_checksum() -> None:
    """REQ-REPORT-5236: required fields stay bare and checksum-protected."""

    artifact = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(),
        schema_reports=_schema_pass(),
        adversarial_reports=_adversarial_pass(),
        qa_recheck_commands=["adversarial_verify exp5224 exp5225: PASS"],
    )

    wrapped = dict(artifact)
    wrapped["gap4_headline_eligible"] = {"value": False}
    with pytest.raises(ValueError, match="gap4_headline_eligible_bare_bool"):
        mod.validate_artifact(wrapped)

    bad_decision = dict(artifact)
    bad_decision["gap4_status_decision"] = "positiveish"
    with pytest.raises(ValueError, match="gap4_status_decision"):
        mod.validate_artifact(bad_decision)

    bad_pool = dict(artifact)
    bad_pool["pool_regenerated"] = True
    with pytest.raises(ValueError, match="pool_regenerated"):
        mod.validate_artifact(bad_pool)

    bad_commands = dict(artifact)
    bad_commands["qa_recheck_commands"] = ["ok", object()]
    with pytest.raises(ValueError, match="qa_recheck_commands"):
        mod.validate_artifact(bad_commands)

    bad_int = dict(artifact)
    bad_int["wins"] = "0"
    with pytest.raises(ValueError, match="wins_bare_int"):
        mod.validate_artifact(bad_int)

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_report_5236_validation_rejects_each_remaining_contract_break() -> None:
    """REQ-REPORT-5236: validation fails closed for each schema contract break."""

    clean = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(),
        schema_reports=_schema_pass(),
        adversarial_reports=_adversarial_pass(),
    )
    positive = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(wins=6, losses=0, ties=114, positive=True),
        schema_reports=_schema_pass(),
        adversarial_reports=_adversarial_pass(),
    )
    flagged_reports = _adversarial_pass()
    flagged_reports[0] = {
        "name": "adversarial_verify",
        "path": str(mod.EXP5224_RELATIVE_PATH),
        "passed": False,
        "loaded": True,
        "flag_count": 1,
        "flags": [{"kind": "METHODOLOGY_MISSING"}],
    }
    blocked = mod.build_artifact(
        qa_artifact=_qa_artifact(),
        pool_artifact=_pool_artifact(),
        validation_artifact=_validation_artifact(),
        schema_reports=_schema_pass(),
        adversarial_reports=flagged_reports,
    )

    missing = dict(clean)
    missing.pop("schema")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_principles = dict(clean)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_headline = dict(clean)
    bad_headline["gap4_headline_eligible"] = True
    with pytest.raises(ValueError, match="gap4_headline_eligible"):
        mod.validate_artifact(bad_headline)

    bad_substrate = dict(clean)
    bad_substrate["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    blocked_no_reason = dict(blocked)
    blocked_no_reason["remaining_blocker"] = ""
    with pytest.raises(ValueError, match="remaining_blocker"):
        mod.validate_artifact(blocked_no_reason)

    clean_with_reason = dict(clean)
    clean_with_reason["remaining_blocker"] = "should not be set"
    with pytest.raises(ValueError, match="remaining_blocker"):
        mod.validate_artifact(clean_with_reason)

    no_prefix = dict(clean)
    no_prefix["honest_verdict"] = "GAP-4 is clean-null"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(no_prefix)

    clean_bad_verdict = dict(clean)
    clean_bad_verdict["honest_verdict"] = "complete: GAP-4 is null"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(clean_bad_verdict)

    positive_bad_verdict = dict(positive)
    positive_bad_verdict["honest_verdict"] = "success: GAP-4 is positive"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(positive_bad_verdict)

    blocked_bad_verdict = dict(blocked)
    blocked_bad_verdict["honest_verdict"] = "complete: GAP-4 blocked"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(blocked_bad_verdict)
