"""Tests for Exp 5240 ARC rubric-to-patch synthesis.

Spec refs: REQ-REPORT-5240, SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS,
SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5240_arc_rubric_to_patch_synthesis_v479 as mod
from carnot.agentic import arc_solve_learning


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5240_spec_declares_patch_synthesis_contract() -> None:
    """REQ-REPORT-5240: OpenSpec declares the required Exp 5240 gate schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5240") :]

    for marker in (
        "REQ-REPORT-5240",
        "SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS",
        "SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE",
        mod.RESULT_RELATIVE_PATH,
        "recommended_live_patch_available",
        "patch_test_ready",
        "arc_live_path_patch_synthesis",
        "provenance_routing",
    ):
        assert marker in section


def test_scenario_report_5240_synthesizes_provenance_guard_patch() -> None:
    """SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS: evidence gates one live patch."""

    artifact = mod.build_artifact(root=REPO, tests_run=[{"command": "fixture", "passed": True}])

    assert artifact["recommended_live_patch_available"] is True
    assert artifact["patch_test_ready"] is True
    assert artifact["patch_path"] == mod.PATCH_RELATIVE_PATH
    assert artifact["patch_failure_mode_targeted"] == "provenance_routing"
    assert artifact["registry_precheck_done"] is True
    assert artifact["duplicate_solve_target_avoided"] is True
    assert artifact["model_specs"] is None
    assert artifact["inference_substrate"] == "arc_live_path_patch_synthesis"
    assert artifact["honest_verdict"].startswith("success:")
    assert "level solve" not in artifact["honest_verdict"].lower()


@pytest.mark.memory_watchdog_skip
def test_req_report_5240_live_agent_reaches_typed_memory_guard() -> None:
    """REQ-REPORT-5240: the live E3 recommendation path exposes the patch hook."""

    from carnot.agentic import arc_competition_agent as live_agent

    recommendation = live_agent._recommend_live_approach("zz99_definitely_unseen")
    guard = recommendation["typed_memory_provenance_guard"]

    assert guard["enabled"] is True
    assert guard["failure_mode_targeted"] == "provenance_routing"
    assert guard["recommended_heads"] == ["provenance", "failures", "skills_rubrics"]
    assert "block_gap1_registry_promotion_until_frozen_subset_gate" in guard[
        "blocked_arc_consumer_actions"
    ]
    assert "quarantine_gap4_candidate_pool_until_positive_validation" in guard[
        "blocked_arc_consumer_actions"
    ]
    assert recommendation["strategy"]["game"] == "zz99_definitely_unseen"


def test_scenario_report_5240_no_patch_when_evidence_is_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE: missing evidence fails closed."""

    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc_solve_registry.yaml").write_text(
        "schema_version: 1\nreproducible_total_levels: 69\ngames: []\n",
        encoding="utf-8",
    )

    artifact = mod.build_artifact(root=tmp_path, tests_run=[])

    assert artifact["recommended_live_patch_available"] is False
    assert artifact["patch_test_ready"] is False
    assert artifact["patch_path"] is None
    assert artifact["patch_failure_mode_targeted"] == "none"
    assert artifact["registry_precheck_done"] is True
    assert artifact["live_agent_reachability_evidence"] is None
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_5240_run_writes_bare_boolean_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-5240: run writes bare booleans and the required synthesis fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    tests_run = [{"command": "fixture command", "passed": True}]
    artifact = mod.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert isinstance(artifact["recommended_live_patch_available"], bool)
    assert isinstance(artifact["patch_test_ready"], bool)
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["tests_run"] == tests_run


def test_req_report_5240_repository_artifact_is_stable_and_replayable() -> None:
    """REQ-REPORT-5240: checked-in artifact matches deterministic synthesis."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(root=REPO, tests_run=result["tests_run"])

    assert result == replay
    assert result["recommended_live_patch_available"] is True
    assert result["patch_test_ready"] is True


def test_req_report_5240_guard_fails_closed_without_controlled_reuse(tmp_path: Path) -> None:
    """REQ-REPORT-5240: the live guard is disabled without controlled typed-memory reuse."""

    guard = arc_solve_learning.typed_memory_provenance_guard(root=tmp_path)

    assert guard["enabled"] is False
    assert guard["failure_mode_targeted"] == "none"
    assert guard["recommended_heads"] == []
