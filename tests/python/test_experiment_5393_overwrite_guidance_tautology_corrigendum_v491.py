"""Tests for Exp5393 overwrite-guidance tautology corrigendum.

Spec refs: REQ-VERIFY-5393, SCENARIO-VERIFY-5393.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.adversarial_verify as adversarial_verify
from carnot import (
    experiment_5393_overwrite_guidance_tautology_corrigendum_v491 as mod,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5393_overwrite_guidance_tautology_corrigendum_v491.py "
    "-q --no-cov"
)


def test_req_verify_5393_spec_declares_row_level_corrigendum_contract() -> None:
    """REQ-VERIFY-5393: OpenSpec anchors the row-level corrigendum."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5393") : spec.index("### REQ-VERIFY-5380")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5393",
        "SCENARIO-VERIFY-5393",
        str(mod.RESULT_RELATIVE_PATH),
        str(mod.SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH),
        "benign hints",
        "incomplete hints",
        "harmful hints",
        "contradictory hints",
        "no-hint controls",
        "raw hint",
        "solver pre-state",
        "solver post-state",
        "validity proof",
        "negative controls",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5393_identifies_exp5383_tautology_source() -> None:
    """REQ-VERIFY-5393: source review names the quarantined aggregate pair."""

    source = mod.load_source_artifact(REPO)
    review = mod.identify_source_tautology(source)

    assert review["source_flagged_artifact"] == str(
        mod.SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH
    )
    assert review["tautological_fields"] == [
        {
            "left": "forced_hint_harm_rate",
            "right": "legacy_all_mode_projection_validity_rate",
            "left_value": 0.75,
            "right_value": 0.75,
            "why_suspect": (
                "forced-hint harm and all-mode projection validity are distinct "
                "aggregate concepts and must not be treated as independent "
                "evidence when they agree exactly."
            ),
        }
    ]
    assert review["reused_aggregate_fields"] == []


def test_scenario_verify_5393_rows_cover_all_hint_and_control_families() -> None:
    """SCENARIO-VERIFY-5393: every row carries solver-authoritative evidence."""

    rows = mod.build_corrigendum_rows(REPO)
    families = {row["fixture_family"] for row in rows}

    assert families == set(mod.REQUIRED_FIXTURE_FAMILIES)
    assert len(rows) == mod.EXPECTED_ROW_COUNT

    for row in rows:
        assert row["raw_hint"] is None or isinstance(row["raw_hint"], list)
        assert {"status", "solution", "conflicts", "convergence_steps"} <= set(
            row["solver_pre_state"]
        )
        assert {"status", "solution", "conflicts", "convergence_steps"} <= set(
            row["solver_post_state"]
        )
        assert row["hint_action"] in {"accepted", "overwritten", "ignored"}
        assert row["validity_proof"]["solver_authoritative"] is True
        assert row["validity_proof"]["accepted_as_valid"] is bool(
            row["accepted_as_valid"]
        )
        assert isinstance(row["conflict_delta_vs_no_hint"], int)
        assert isinstance(row["fallback_result"]["complete"], bool)
        assert row["unsafe_status"]["unsafe_false_accept"] is False

    no_hint_controls = [
        row for row in rows if row["fixture_family"] == mod.NO_HINT_FAMILY
    ]
    assert no_hint_controls
    assert all(row["raw_hint"] is None for row in no_hint_controls)
    assert all(row["hint_action"] == "ignored" for row in no_hint_controls)
    assert all(row["negative_control"]["passed"] for row in no_hint_controls)


def test_scenario_verify_5393_recomputes_required_rates_from_rows() -> None:
    """SCENARIO-VERIFY-5393: readiness metrics derive only from rows."""

    rows = mod.build_corrigendum_rows(REPO)
    summary = mod.summarize_corrigendum_rows(rows)
    harmful_controls = [
        row
        for row in rows
        if row["negative_control"]["control_kind"]
        in {"forced_harmful_no_improvement", "forced_contradictory_no_improvement"}
    ]
    overwrite_harmful = [
        row
        for row in rows
        if row["fixture_family"] in {"harmful", "contradictory"}
        and row["guidance_mode"] == "overwrite_capable"
    ]

    assert summary["row_count"] == mod.EXPECTED_ROW_COUNT
    assert summary["fixture_families"] == list(mod.REQUIRED_FIXTURE_FAMILIES)
    assert summary["row_metric_denominator"] == mod.EXPECTED_ROW_COUNT
    assert summary["overwrite_rate_denominator"] == 42
    assert summary["forced_hint_harm_denominator"] == len(harmful_controls)
    assert summary["post_projection_validity_denominator"] == 70
    assert summary["fallback_completeness_denominator"] == len(overwrite_harmful)
    assert summary["negative_control_denominator"] == 42
    assert summary["overwrite_rate_from_rows"] == pytest.approx(42 / 42)
    assert summary["forced_hint_harm_rate_from_rows"] == pytest.approx(28 / 28)
    assert summary["post_projection_validity_rate_from_rows"] == pytest.approx(70 / 70)
    assert summary["fallback_completeness_rate_from_rows"] == pytest.approx(28 / 28)
    assert summary["negative_control_pass_rate"] == pytest.approx(42 / 42)
    assert summary["unsafe_false_accept_count"] == 0
    assert summary["row_level_evidence_clean"] is True

    assert all(row["hint_action"] == "overwritten" for row in overwrite_harmful)
    assert all(row["fallback_result"]["used"] for row in overwrite_harmful)
    assert all(row["fallback_result"]["complete"] for row in overwrite_harmful)
    assert all(row["negative_control"]["passed"] for row in harmful_controls)


def test_req_verify_5393_artifact_schema_and_adversarial_cleanliness(tmp_path: Path) -> None:
    """REQ-VERIFY-5393: artifact fields validate and adversarial checks pass."""

    artifact = mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND])
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    written = mod.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])
    report = adversarial_verify.verify_artifact(result_path)
    critical = [flag for flag in report["flags"] if flag["severity"] == "critical"]

    assert json.loads(result_path.read_text(encoding="utf-8")) == written
    assert written == artifact
    mod.validate_artifact(artifact)
    assert critical == []
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.07.491"
    assert artifact["source_flagged_artifact"] == str(
        mod.SOURCE_FLAGGED_ARTIFACT_RELATIVE_PATH
    )
    assert artifact["fixture_families"] == list(mod.REQUIRED_FIXTURE_FAMILIES)
    assert artifact["tautology_checks_passed"] is True
    assert artifact["overwrite_guidance_corrigendum_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES


def test_req_verify_5393_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5393: checked-in JSON is stable under row recomputation."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=checked_in["tests_run"])

    assert checked_in == replay
    mod.validate_artifact(checked_in)
    assert checked_in["overwrite_guidance_corrigendum_clean"] is True


def test_scenario_verify_5393_fails_closed_for_missing_or_unsafe_rows() -> None:
    """SCENARIO-VERIFY-5393: unsafe accepts or bad controls block readiness."""

    clean = mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    unsafe = deepcopy(clean)
    unsafe["row_evidence"][0]["unsafe_status"]["unsafe_false_accept"] = True
    unsafe["row_evidence"][0]["validity_proof"]["accepted_as_valid"] = True
    unsafe["unsafe_false_accept_count"] = 1
    unsafe["overwrite_guidance_corrigendum_clean"] = False
    unsafe["status"] = "blocked"
    unsafe["honest_verdict"] = "blocked: unsafe row accepted"
    with pytest.raises(ValueError, match="unsafe_false_accept_count"):
        mod.validate_artifact(unsafe)

    bad_controls = mod.build_artifact(
        root=REPO,
        tests_run=[TEST_COMMAND],
        row_overrides=lambda rows: _break_first_negative_control(rows),
    )
    assert bad_controls["status"] == "blocked"
    assert bad_controls["overwrite_guidance_corrigendum_clean"] is False
    assert "negative_controls_failed" in bad_controls["readiness_blockers"]


def test_req_verify_5393_helper_branches_are_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-5393: flagged and fallback helper branches stay tested."""

    monkeypatch.setattr(mod, "_local_tautology_checks_passed", lambda _artifact: False)
    flagged = mod.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    assert flagged["status"] == "flagged"
    assert flagged["tautology_checks_passed"] is False
    assert flagged["overwrite_guidance_corrigendum_clean"] is False
    assert flagged["honest_verdict"].startswith("flagged:")
    assert "adversarial_tautology_failed" in flagged["readiness_blockers"]
    assert mod._fallback_status(True, False) == "fallback_incomplete"


def _break_first_negative_control(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    broken = deepcopy(rows)
    for row in broken:
        negative_control = row["negative_control"]
        if isinstance(negative_control, dict) and negative_control["control_kind"]:
            negative_control["passed"] = False
            break
    return broken
