"""Tests for Exp 5310 deterministic paraphrase-consistency fixture.

Spec refs: REQ-VERIFY-5310, SCENARIO-VERIFY-5310.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5310_paraphrase_consistency_fixture_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_verify_5310_spec_declares_no_llm_paraphrase_fixture_contract() -> None:
    """REQ-VERIFY-5310: OpenSpec anchors the deterministic fixture contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5310") : spec.index("### REQ-VERIFY-5297")]

    for marker in (
        "REQ-VERIFY-5310",
        "SCENARIO-VERIFY-5310",
        str(mod.FIXTURE_RELATIVE_PATH),
        str(mod.RESULT_RELATIVE_PATH),
        "deterministic_claim_paraphrase_fixture_no_llm",
        "paraphrase_fixture_ready",
        "semantically equivalent",
        "contradiction-preserving",
        "premise-invalid",
        "surface-only rewrites",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    normalized_section = " ".join(section.split())
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5310_fixture_has_required_labeled_groups() -> None:
    """REQ-VERIFY-5310: fixture families and expected labels are explicit."""

    groups = mod.load_fixture()
    counts = mod.fixture_family_counts(groups)

    assert mod.FIXTURE_RELATIVE_PATH.exists()
    assert counts == {
        "equivalent": 1,
        "contradiction-preserving": 1,
        "premise-invalid": 1,
        "surface-only": 1,
    }
    assert mod.group_by_id(groups, "pcf-001-supported-equivalent").anchor.expected_label == (
        "supported"
    )
    assert mod.group_by_id(groups, "pcf-002-contradiction-preserving").anchor.expected_label == (
        "contradictory"
    )
    assert mod.group_by_id(groups, "pcf-003-premise-invalid").anchor.expected_label == (
        "premise-invalid"
    )

    for group in groups:
        assert group.label_source == "curated_deterministic_claim_paraphrase_fixture_v485"
        assert group.variants
        for claim in (group.anchor, *group.variants):
            assert claim.expected_label in mod.SEMANTIC_LABELS


def test_scenario_verify_5310_preserves_equivalent_and_contradictory_labels() -> None:
    """SCENARIO-VERIFY-5310: equivalent and contradiction-preserving labels hold."""

    evaluation = mod.evaluate_fixture(mod.load_fixture())
    rows = {(row["group_id"], row["claim_id"]): row for row in evaluation["claim_results"]}

    assert evaluation["label_preservation_pass_rate"] == pytest.approx(1.0)
    assert (
        rows[("pcf-001-supported-equivalent", "pcf-001-v1-word-order")]["computed_label"]
        == "supported"
    )
    assert (
        rows[("pcf-001-supported-equivalent", "pcf-001-v1-word-order")]["label_preserved"] is True
    )
    assert (
        rows[("pcf-002-contradiction-preserving", "pcf-002-v1-same-wrong-values")]["computed_label"]
        == "contradictory"
    )
    assert (
        rows[("pcf-002-contradiction-preserving", "pcf-002-v1-same-wrong-values")][
            "label_preserved"
        ]
        is True
    )


def test_scenario_verify_5310_catches_contradiction_erasure_violation() -> None:
    """SCENARIO-VERIFY-5310: contradiction-erasing paraphrases are violations."""

    evaluation = mod.evaluate_fixture(mod.load_fixture())
    violation = next(
        row
        for row in evaluation["claim_results"]
        if row["claim_id"] == "pcf-002-v2-corrected-facts"
    )

    assert evaluation["contradiction_violation_caught_rate"] == pytest.approx(1.0)
    assert violation["expected_violation_type"] == "contradiction_erased"
    assert violation["computed_label"] == "supported"
    assert violation["expected_label_preservation"] is False
    assert violation["caught_expected_violation"] is True


def test_scenario_verify_5310_handles_invalid_premises_and_surface_false_positives() -> None:
    """SCENARIO-VERIFY-5310: invalid premises reject and surface overlap is not enough."""

    groups = mod.load_fixture()
    evaluation = mod.evaluate_fixture(groups)
    rows = {(row["group_id"], row["claim_id"]): row for row in evaluation["claim_results"]}

    assert evaluation["invalid_premise_handled"] is True
    assert rows[("pcf-003-premise-invalid", "pcf-003-anchor")]["computed_label"] == (
        "premise-invalid"
    )
    assert (
        rows[("pcf-003-premise-invalid", "pcf-003-v1-restated-premise")]["label_preserved"] is True
    )

    surface_trap = rows[("pcf-004-surface-only", "pcf-004-v2-negation-overlap")]
    assert surface_trap["expected_violation_type"] == "surface_overlap_label_flip"
    assert surface_trap["token_overlap_with_anchor"] >= 0.8
    assert surface_trap["computed_label"] == "contradictory"
    assert surface_trap["label_preserved"] is False
    assert surface_trap["caught_expected_violation"] is True


def test_req_verify_5310_run_writes_required_artifact_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-5310: run() writes principle fields and bare downstream gates."""

    tests_run = [{"command": "unit paraphrase fixture", "outcome": "passed"}]
    artifact = mod.run(result_path=tmp_path / "experiment_5310.json", tests_run=tests_run)

    assert json.loads((tmp_path / "experiment_5310.json").read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == mod.EXPERIMENT_NAME
    assert artifact["milestone"]["value"] == "2026.07.485"
    assert artifact["status"]["value"] == "complete"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["paraphrase_fixture_ready"] is True
    assert artifact["fixture_path"]["value"] == str(mod.FIXTURE_RELATIVE_PATH)
    assert artifact["paraphrase_group_count"] == 4
    assert artifact["label_preservation_pass_rate"] == pytest.approx(1.0)
    assert artifact["contradiction_violation_caught_rate"] == pytest.approx(1.0)
    assert artifact["invalid_premise_handled"] is True
    assert artifact["tests_run"]["value"] == tests_run


def test_scenario_verify_5310_blocks_when_required_family_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5310: missing fixture families close the downstream gate."""

    groups = mod.load_fixture()
    incomplete_groups = [group for group in groups if group.family != "premise-invalid"]
    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        groups=incomplete_groups,
        tests_run=[{"command": "unit blocked", "outcome": "passed"}],
    )

    mod.validate_artifact(artifact)
    assert artifact["status"]["value"] == "blocked"
    assert artifact["honest_verdict"]["value"].startswith("blocked_")
    assert artifact["paraphrase_fixture_ready"] is False
    assert artifact["invalid_premise_handled"] is False
    assert "missing families: premise-invalid" in artifact["readiness_blockers"]

    surface_group = mod.group_by_id(groups, "pcf-004-surface-only")
    unsupported_claim = mod.ParaphraseClaim(
        claim_id="pcf-synthetic-unsupported",
        text="The Noma audit checksum was 8f12 and the auditor was Rin.",
        premise_valid=True,
        facts={**surface_group.evidence_facts, "auditor": "rin"},
        expected_label="unsupported",
        expected_label_preservation=False,
        expected_violation_type=None,
    )
    assert mod.score_claim(unsupported_claim, surface_group).label == "unsupported"

    synthetic_blockers = mod._readiness_blockers(
        {
            "missing_families": [],
            "label_mismatches": ["label-x"],
            "preservation_mismatches": ["preserve-y"],
            "uncaught_violations": ["violation-z"],
            "invalid_premise_handled": True,
            "surface_false_positive_resisted": False,
        }
    )
    assert "label mismatches: label-x" in synthetic_blockers
    assert "preservation mismatches: preserve-y" in synthetic_blockers
    assert "uncaught violations: violation-z" in synthetic_blockers
    assert "surface false positive not resisted" in synthetic_blockers


def test_req_verify_5310_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5310: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(mod.load_fixture(), tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["paraphrase_fixture_ready"] is True
    assert result["inference_substrate"]["value"] == (
        "deterministic_claim_paraphrase_fixture_no_llm"
    )
    mod.validate_artifact(result)
