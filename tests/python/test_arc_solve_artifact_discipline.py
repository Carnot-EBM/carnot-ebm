"""Tests for ARC solve/scoring artifact discipline.

Spec refs: REQ-VERIFY-4437, SCENARIO-VERIFY-4437.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic import arc_solve_artifact_discipline as discipline


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
ARTIFACT_PATH = REPO / "results" / "experiment_4437_arc_artifact_discipline_template.json"


def test_req_verify_4437_spec_declares_template_lint_and_floors() -> None:
    """REQ-VERIFY-4437: OpenSpec names the helper, lint, and substrate floors."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-4437",
        "SCENARIO-VERIFY-4437",
        "python/carnot/agentic/arc_solve_artifact_discipline.py",
        "scripts/arc_artifact_lint.py",
        "aggregation_from_upstream_artifacts",
        "verifier_ensemble_against_cached_candidates",
        "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "live_llm_inference",
        "experiment_4437_arc_artifact_discipline_template.json",
    ):
        assert marker in spec


def test_req_verify_4437_helper_builds_offline_template_with_canonical_substrate() -> None:
    """REQ-VERIFY-4437: sub-60s offline ARC artifacts use the aggregation substrate."""

    artifact = discipline.build_arc_solve_artifact(
        experiment="experiment_9999_config_rule_solve",
        honest_verdict="complete: offline_config_rule_reproduced",
        inference_substrate=discipline.AGGREGATION_SUBSTRATE,
        duration_s=0.01,
        artifact_kind="arc_solve",
        result_path="results/experiment_9999_config_rule_solve.json",
        extra_fields={"offline_reproduced": True, "new_levels_reproduced": 0},
    )

    assert artifact["inference_substrate"] == discipline.AGGREGATION_SUBSTRATE
    assert artifact["duration_s"] == 0.01
    assert artifact["offline_reproduced"] is True
    assert artifact["field_principles"]["honest_verdict"] == "terminal-prefixed"
    assert artifact["field_principles"]["inference_substrate"].startswith("canonical")
    assert len(artifact["reproducibility_checksum"]) == 64
    assert discipline.validate_arc_solve_artifact(artifact) == []
    assert discipline.duration_floor_s(123) is None


def test_req_verify_4437_helper_accepts_arc_live_agent_no_llm_substrate() -> None:
    """REQ-VERIFY-4437: live ARC no-LLM receipts use the canonical 0.01s floor."""

    artifact = discipline.build_arc_solve_artifact(
        experiment="experiment_9996_arc_live_patch_receipt",
        honest_verdict="complete: no_bank_live_agent_receipt",
        inference_substrate=discipline.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE,
        duration_s=0.02,
        artifact_kind="arc_solve",
        result_path="results/experiment_9996_arc_live_patch_receipt.json",
        extra_fields={"solve_provenance": "live_agent_self_discovery", "level_delta": 0},
    )

    assert artifact["inference_substrate"] == discipline.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE
    assert discipline.duration_floor_s(discipline.ARC_LIVE_AGENT_NO_LLM_SUBSTRATE) == 0.01
    assert discipline.validate_arc_solve_artifact(artifact) == []

    too_short = dict(artifact, duration_s=0.0)
    assert [issue.kind for issue in discipline.validate_arc_solve_artifact(too_short)] == [
        "DURATION_BELOW_SUBSTRATE_FLOOR"
    ]


def test_scenario_verify_4437_helper_rejects_missing_short_and_nonterminal_cases() -> None:
    """SCENARIO-VERIFY-4437: helper validation reports every required failure."""

    bad = {
        "honest_verdict": "partial: config_rule_unfinished",
        "duration_s": 0.0,
        "artifact_kind": "arc_solve",
    }

    issues = discipline.validate_arc_solve_artifact(bad)
    kinds = {issue.kind for issue in issues}

    assert "MISSING_INFERENCE_SUBSTRATE" in kinds
    assert "NON_TERMINAL_HONEST_VERDICT" in kinds

    short_verifier = {
        "honest_verdict": "complete: cached_verifier_scored",
        "duration_s": 0.5,
        "inference_substrate": discipline.VERIFIER_SCORING_SUBSTRATE,
    }
    short_issues = discipline.validate_arc_solve_artifact(short_verifier)
    assert [issue.kind for issue in short_issues] == ["DURATION_BELOW_SUBSTRATE_FLOOR"]

    live_without_allow = {
        "honest_verdict": "success: live_llm_induction_solved",
        "duration_s": 61.0,
        "inference_substrate": discipline.LIVE_LLM_SUBSTRATE,
    }
    live_issues = discipline.validate_arc_solve_artifact(live_without_allow)
    assert [issue.kind for issue in live_issues] == ["LIVE_LLM_NOT_ALLOWLISTED"]
    assert discipline.validate_arc_solve_artifact(live_without_allow, allow_live=True) == []

    missing_duration = {
        "honest_verdict": "complete: offline_replay",
        "duration_s": "bad",
        "inference_substrate": discipline.AGGREGATION_SUBSTRATE,
    }
    duration_issues = discipline.validate_arc_solve_artifact(missing_duration)
    assert duration_issues[0].to_dict() == {
        "kind": "DURATION_MISSING",
        "detail": "duration_s must be a finite number for substrate floor checks.",
    }
    bool_duration = dict(missing_duration, duration_s=True)
    assert [issue.kind for issue in discipline.validate_arc_solve_artifact(bool_duration)] == [
        "DURATION_MISSING"
    ]


def test_req_verify_4437_builder_raises_for_invalid_template_inputs() -> None:
    """REQ-VERIFY-4437: the template cannot be built with an invalid substrate."""

    merged = discipline.build_arc_solve_artifact(
        experiment="experiment_9997_arc_solve",
        honest_verdict="complete: offline_replay",
        inference_substrate=discipline.AGGREGATION_SUBSTRATE,
        duration_s=0.01,
        artifact_kind="arc_solve",
        extra_fields={"field_principles": {"custom": "custom principle"}},
    )
    assert merged["field_principles"]["custom"] == "custom principle"

    with pytest.raises(ValueError) as exc:
        discipline.build_arc_solve_artifact(
            experiment="experiment_9998_arc_solve",
            honest_verdict="success: live_solver_claimed",
            inference_substrate=discipline.LIVE_LLM_SUBSTRATE,
            duration_s=2.0,
            artifact_kind="arc_solve",
        )

    message = str(exc.value)
    assert "DURATION_BELOW_SUBSTRATE_FLOOR" in message
    assert "LIVE_LLM_NOT_ALLOWLISTED" in message


def test_scenario_verify_4437_delivered_artifact_has_required_bare_fields() -> None:
    """SCENARIO-VERIFY-4437: the terminal artifact records the shipped guardrail."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith(("complete:", "success:", "passed:", "shipped:"))
    assert artifact["template_shipped"] is True
    assert artifact["tests_pass"] is True
    assert artifact["field_principles"]["honest_verdict"] == "terminal-prefixed"
    assert artifact["field_principles"]["template_shipped"] == (
        "bare bool: the helper + lint + tests landed green"
    )
    assert artifact["field_principles"]["tests_pass"] == (
        "bare bool: the new unit tests run and assert (Tests-Must-Run-and-Assert)"
    )
    assert artifact["spec_refs"] == ["REQ-VERIFY-4437", "SCENARIO-VERIFY-4437"]
    assert discipline.validate_arc_solve_artifact(artifact) == []
