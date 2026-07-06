"""Tests for Exp 5325 deterministic rewrite-state fixture.

Spec refs: REQ-VERIFY-5325, SCENARIO-VERIFY-5325.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5325_theoria_rewrite_state_fixture_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _rows_by_case(evaluation: mod.JsonDict) -> dict[str, mod.JsonDict]:
    return {row["case_id"]: row for row in evaluation["case_results"]}


def test_req_verify_5325_spec_declares_rewrite_state_contract() -> None:
    """REQ-VERIFY-5325: OpenSpec anchors the deterministic rewrite-state gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5325") : spec.index("### REQ-VERIFY-5324")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5325",
        "SCENARIO-VERIFY-5325",
        str(mod.FIXTURE_RELATIVE_PATH),
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "rewrite_case_count",
        "rewrite_acceptability_rate",
        "complete_change_coverage_rate",
        "unsafe_rewrite_rejection_rate",
        "false_accept_count",
        "rewrite_state_fixture_ready",
        "safe paraphrase",
        "contradiction introduction",
        "missing required change",
        "fabricated premise/citation",
        "invalid premise preserved",
        "overbroad rewrite",
        "Exp 5326",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5325_fixture_has_required_typed_cases() -> None:
    """REQ-VERIFY-5325: fixture contains the six typed rewrite transition cases."""

    cases = mod.load_fixture()
    counts = mod.rewrite_case_type_counts(cases)

    assert (REPO / mod.FIXTURE_RELATIVE_PATH).exists()
    assert counts == {
        "safe_paraphrase": 1,
        "contradiction_introduction": 1,
        "missing_required_change": 1,
        "fabricated_premise_citation": 1,
        "invalid_premise_preserved": 1,
        "overbroad_rewrite": 1,
    }
    assert len(cases) == 6
    assert mod.case_by_id(cases, "rsf-001-safe-paraphrase").expected_accept is True
    assert [
        case.case_type for case in cases if case.expected_accept is False
    ] == [
        "contradiction_introduction",
        "missing_required_change",
        "fabricated_premise_citation",
        "invalid_premise_preserved",
        "overbroad_rewrite",
    ]

    for case in cases:
        assert case.label_source == "curated_deterministic_rewrite_state_fixture_v486"
        assert case.change_obligations
        assert case.source.expected_label in mod.SEMANTIC_LABELS
        assert case.target.expected_label in mod.SEMANTIC_LABELS


def test_scenario_verify_5325_accepts_only_safe_paraphrase() -> None:
    """SCENARIO-VERIFY-5325: safe rewrite accepts and every unsafe case rejects."""

    evaluation = mod.evaluate_fixture(mod.load_fixture())
    rows = _rows_by_case(evaluation)

    assert evaluation["rewrite_acceptability_rate"] == pytest.approx(1.0)
    assert evaluation["complete_change_coverage_rate"] == pytest.approx(1.0)
    assert evaluation["unsafe_rewrite_rejection_rate"] == pytest.approx(1.0)
    assert evaluation["false_accept_count"] == 0
    assert evaluation["ready"] is True

    safe = rows["rsf-001-safe-paraphrase"]
    assert safe["accepted"] is True
    assert safe["expected_accept"] is True
    assert safe["complete_change_coverage"] is True
    assert safe["label_preserved"] is True
    assert safe["target_label"] == "supported"
    assert safe["rejection_reasons"] == []

    rejected = [row for row in rows.values() if row["expected_accept"] is False]
    assert len(rejected) == 5
    assert all(row["accepted"] is False for row in rejected)
    assert all(row["unsafe_rewrite_rejected"] is True for row in rejected)


def test_scenario_verify_5325_detects_each_unsafe_transition() -> None:
    """SCENARIO-VERIFY-5325: each negative rewrite class has a deterministic blocker."""

    rows = _rows_by_case(mod.evaluate_fixture(mod.load_fixture()))

    contradiction = rows["rsf-002-contradiction-introduction"]
    assert contradiction["target_label"] == "contradictory"
    assert contradiction["conflict_keys"] == ["duration_minutes"]
    assert "contradiction_introduced" in contradiction["rejection_reasons"]
    assert "label_preservation_failed" in contradiction["rejection_reasons"]

    missing = rows["rsf-003-missing-required-change"]
    assert missing["complete_change_coverage"] is False
    assert missing["missing_obligations"] == ["rsf-003-obligation-review-state"]
    assert "missing_required_change" in missing["rejection_reasons"]

    fabricated = rows["rsf-004-fabricated-premise-citation"]
    assert fabricated["target_label"] == "unsupported"
    assert fabricated["fabricated_fact_keys"] == ["external_report_claim"]
    assert fabricated["fabricated_citations"] == ["phantom-report-77"]
    assert "fabricated_fact" in fabricated["rejection_reasons"]
    assert "fabricated_citation" in fabricated["rejection_reasons"]

    invalid = rows["rsf-005-invalid-premise-preserved"]
    assert invalid["source_label"] == "premise-invalid"
    assert invalid["target_label"] == "premise-invalid"
    assert invalid["label_preserved"] is True
    assert "invalid_premise_preserved" in invalid["rejection_reasons"]
    assert "expected_label_change_missing" in invalid["rejection_reasons"]

    overbroad = rows["rsf-006-overbroad-rewrite"]
    assert overbroad["target_label"] == "contradictory"
    assert overbroad["overbroad_fact_keys"] == ["scope"]
    assert "overbroad_fact_change" in overbroad["rejection_reasons"]


def test_req_verify_5325_run_writes_required_artifact_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-5325: run() writes principle fields and bare downstream gates."""

    tests_run = [{"command": "unit rewrite-state fixture", "outcome": "passed"}]
    artifact = mod.run(result_path=tmp_path / "experiment_5325.json", tests_run=tests_run)

    assert json.loads((tmp_path / "experiment_5325.json").read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_NAME
    assert artifact["milestone"]["value"] == "2026.07.486"
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["rewrite_case_count"] == 6
    assert artifact["rewrite_acceptability_rate"] == pytest.approx(1.0)
    assert artifact["complete_change_coverage_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_rewrite_rejection_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_count"] == 0
    assert artifact["rewrite_state_fixture_ready"] is True
    assert artifact["consumer_contract"]["next_experiment"] == "Exp5326"
    assert artifact["tests_run"]["value"] == tests_run


def test_scenario_verify_5325_blocks_missing_case_type_and_false_accept(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5325: missing cases and false accepts close the fixture gate."""

    cases = mod.load_fixture()
    incomplete_cases = [
        case for case in cases if case.case_type != "fabricated_premise_citation"
    ]
    incomplete = mod.run(
        result_path=tmp_path / "missing-case.json",
        cases=incomplete_cases,
        tests_run=[{"command": "unit missing case", "outcome": "passed"}],
    )

    mod.validate_artifact(incomplete)
    assert incomplete["status"]["value"] == "blocked"
    assert incomplete["honest_verdict"]["value"].startswith("blocked_")
    assert incomplete["rewrite_state_fixture_ready"] is False
    assert "missing case types: fabricated_premise_citation" in incomplete["readiness_blockers"]

    missing_case = mod.case_by_id(cases, "rsf-003-missing-required-change")
    repaired_target = replace(
        missing_case.target,
        attributes={**missing_case.target.attributes, "review_state": "approved"},
    )
    false_accept_case = replace(missing_case, target=repaired_target)
    false_accept_cases = tuple(
        false_accept_case if case.case_id == missing_case.case_id else case for case in cases
    )
    false_accept = mod.build_artifact(
        false_accept_cases,
        tests_run=[{"command": "unit false accept", "outcome": "passed"}],
    )

    mod.validate_artifact(false_accept)
    assert false_accept["status"]["value"] == "blocked"
    assert false_accept["rewrite_state_fixture_ready"] is False
    assert false_accept["false_accept_count"] == 1
    assert false_accept["unsafe_rewrite_rejection_rate"] < 1.0
    assert "false accepts: rsf-003-missing-required-change" in false_accept[
        "readiness_blockers"
    ]

    synthetic_blockers = mod._readiness_blockers(
        {
            "missing_case_types": [],
            "false_accept_ids": [],
            "label_mismatch_ids": ["label-x"],
            "rewrite_acceptability_rate": 1.0,
            "complete_change_coverage_rate": 1.0,
            "unsafe_rewrite_rejection_rate": 1.0,
        }
    )
    assert "label mismatches: label-x" in synthetic_blockers


def test_req_verify_5325_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5325: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(mod.load_fixture(), tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["rewrite_state_fixture_ready"] is True
    assert result["inference_substrate"]["value"] == "deterministic_rewrite_state_fixture"
    assert result["false_accept_count"] == 0
    mod.validate_artifact(result)
