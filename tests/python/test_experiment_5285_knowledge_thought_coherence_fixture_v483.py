"""Tests for Exp 5285 knowledge-thought coherence fixture.

Spec refs: REQ-VERIFY-5285, SCENARIO-VERIFY-5285.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5285_knowledge_thought_coherence_fixture_v483 as mod


SPEC_PATH = Path("openspec/capabilities/verification/spec.md")


def test_req_verify_5285_spec_declares_offline_coherence_fixture_contract() -> None:
    """REQ-VERIFY-5285: OpenSpec anchors the offline CheckRLM-style fixture."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5285") : spec.index("### REQ-VERIFY-5272")]

    for marker in (
        "REQ-VERIFY-5285",
        "SCENARIO-VERIFY-5285",
        str(mod.FIXTURE_RELATIVE_PATH),
        str(mod.RESULT_RELATIVE_PATH),
        "offline_deterministic_fixture_no_llm",
        "coherence_fixture_ready",
        "supported, unsupported, partially supported",
        "safety-negative",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_verify_5285_fixture_has_required_labels_and_separates_format() -> None:
    """REQ-VERIFY-5285: fixture families and format labels are deterministic."""

    cases = mod.load_fixture()
    counts = mod.fixture_case_counts(cases)

    assert mod.FIXTURE_RELATIVE_PATH.exists()
    assert counts == {
        "supported": 2,
        "unsupported": 1,
        "partial": 1,
        "stale": 1,
        "contradictory": 1,
        "safety-negative": 1,
    }
    for case in cases:
        assert case.semantic_label == case.case_type
        assert case.expected_claims
        assert case.label_source == "curated_offline_knowledge_thought_fixture"

    malformed_supported = mod.case_by_id(cases, "ktc-007-supported-format-invalid")
    assert malformed_supported.case_type == "supported"
    assert malformed_supported.semantic_label == "supported"
    assert malformed_supported.format_valid is False

    supported_valid = mod.case_by_id(cases, "ktc-001-supported-runtime")
    assert supported_valid.format_valid is True
    assert mod.extract_claims(supported_valid.thought) == supported_valid.expected_claims
    assert mod.extract_claims(malformed_supported.thought) == []


def test_scenario_verify_5285_accepts_supported_and_rejects_bad_claims() -> None:
    """SCENARIO-VERIFY-5285: labels catch unsupported, stale, and unsafe claims."""

    summary = mod.evaluate_fixture(mod.load_fixture())
    by_id = {row["case_id"]: row for row in summary["case_results"]}

    assert by_id["ktc-001-supported-runtime"]["decision"] == "accept"
    assert by_id["ktc-001-supported-runtime"]["semantic_correct"] is True
    assert by_id["ktc-002-unsupported-sensor"]["decision"] == "reject"
    assert by_id["ktc-003-partial-trial"]["decision"] == "reject"
    assert by_id["ktc-004-stale-route"]["decision"] == "reject"
    assert by_id["ktc-005-contradictory-lab"]["decision"] == "reject"
    assert by_id["ktc-006-safety-negative-dose"]["decision"] == "reject"
    assert by_id["ktc-007-supported-format-invalid"]["semantic_correct"] is True
    assert by_id["ktc-007-supported-format-invalid"]["format_valid"] is False
    assert by_id["ktc-007-supported-format-invalid"]["decision"] == "reject"
    assert summary["unsafe_false_accepts"] == 0


def test_req_verify_5285_correction_locality_and_baseline_metrics() -> None:
    """REQ-VERIFY-5285: corrections are local and lexical baseline is reported."""

    summary = mod.evaluate_fixture(mod.load_fixture())
    checks = summary["correction_locality_checks"]
    baseline = summary["baseline_metrics"]

    assert checks["checked_count"] == 5
    assert checks["failed"] == []
    assert checks["passed"] is True
    for row in checks["rows"]:
        assert row["locality_passed"] is True
        assert row["edit_distance"] <= row["max_token_edits"]

    contradictory = next(
        row for row in checks["rows"] if row["case_id"] == "ktc-005-contradictory-lab"
    )
    assert set(contradictory["preserved_terms"]) == {"cedar", "lab", "opened"}

    assert baseline["metric"] == "claim_token_overlap"
    assert baseline["sample_count"] == 7
    assert baseline["threshold"] == pytest.approx(0.55)
    assert baseline["false_accepts"] >= 2
    assert baseline["unsafe_false_accepts"] >= 1


def test_scenario_verify_5285_run_writes_required_artifact_without_llm_calls(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5285: artifact is complete and uses no live model substrate."""

    artifact = mod.run(
        result_path=tmp_path / "experiment_5285.json",
        tests_run=[{"command": "unit ready", "outcome": "passed"}],
    )

    assert json.loads((tmp_path / "experiment_5285.json").read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "usable" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["coherence_fixture_ready"] is True
    assert artifact["fixture_case_counts"]["supported"] == 2
    assert artifact["unsafe_false_accepts"]["value"] == 0
    assert artifact["correction_locality_checks"]["value"]["passed"] is True
    assert artifact["tests_run"] == [{"command": "unit ready", "outcome": "passed"}]


def test_scenario_verify_5285_blocks_when_required_family_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5285: missing safety negatives fail the downstream gate."""

    incomplete_cases = [
        case for case in mod.load_fixture() if case.case_id != "ktc-006-safety-negative-dose"
    ]
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        cases=incomplete_cases,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["coherence_fixture_ready"] is False
    assert "missing case families: safety-negative" in artifact["coherence_fixture_ready_principle"]
    assert artifact["fixture_case_counts"]["safety-negative"] == 0

    synthetic_blocker_text = mod._ready_principle(
        {
            "ready": False,
            "missing_families": [],
            "unsafe_false_accepts": 1,
            "non_supported_accepts": ["ktc-x"],
            "correction_locality_checks": {"passed": False, "failed": ["ktc-y"]},
        }
    )
    assert "unsafe_false_accepts=1" in synthetic_blocker_text
    assert "non_supported_accepts=ktc-x" in synthetic_blocker_text
    assert "correction_locality_failed=ktc-y" in synthetic_blocker_text
