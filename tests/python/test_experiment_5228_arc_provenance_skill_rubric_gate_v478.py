"""Tests for Exp 5228 ARC provenance and skill-rubric gate.

Spec refs: REQ-REPORT-5228, SCENARIO-REPORT-5228-PROCESS-RUBRIC,
SCENARIO-REPORT-5228-PROVENANCE-GATE, SCENARIO-REPORT-5228-PATCH-DECISION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5228_arc_provenance_skill_rubric_gate_v478 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _live_trace(**overrides: Any) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "artifact_path": "results/synthetic_live_trace.json",
        "experiment": "synthetic_live_trace",
        "solve_provenance": "live_agent_self_discovery",
        "registry_precheck_passed": True,
        "duplicate_solve_avoided": True,
        "target_game": "re86",
        "target_level": 3,
        "new_levels_banked": 0,
        "candidate_selection": {
            "candidate_audit": [
                {"game": "lp85", "status": "skip_recorded_dry_next_level"},
                {"game": "re86", "status": "candidate_selected"},
            ],
            "prior_live_path_artifacts_consulted": [
                "results/experiment_5054_arc_live_path_self_discovery.json"
            ],
        },
        "honest_verdict": "complete_re86_no_new_level_residual_duplicate_depth",
        "live_agent_attempts": [
            {
                "attempt_id": "synthetic",
                "policy": "E3AgentPolicy",
                "actions_taken": 12,
                "budget": 36,
                "runtime_self_discovery": True,
                "self_discovery_lever": "bounded_e3_policy_no_archive_injection",
                "offline_source_reading_used": False,
                "offline_ground_truth_bfs_used": False,
                "per_game_bfs_used": False,
                "hand_built_adapter_used": False,
                "live_path_diagnostics": {
                    "policy_observations": 12,
                    "injection_exercised": False,
                },
                "reproduction_gate": {
                    "game": "re86",
                    "claimed_level": 0,
                    "reached_level": 0,
                    "reproduced": False,
                },
            }
        ],
    }
    trace.update(overrides)
    return trace


def test_req_report_5228_spec_declares_required_gate_contract() -> None:
    """REQ-REPORT-5228: OpenSpec declares the required Exp 5228 gate schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5228") :]

    assert "REQ-REPORT-5228" in section
    assert "SCENARIO-REPORT-5228-PROCESS-RUBRIC" in section
    assert "SCENARIO-REPORT-5228-PROVENANCE-GATE" in section
    assert "SCENARIO-REPORT-5228-PATCH-DECISION" in section
    assert mod.RESULT_RELATIVE_PATH in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in section
    for field in mod.RUBRIC_FIELDS:
        assert field in section
    assert "live_trace_process_rubric" in section


def test_scenario_report_5228_provenance_gate_blocks_non_live_evidence() -> None:
    """SCENARIO-REPORT-5228-PROVENANCE-GATE: only clean E3 live self-discovery is valid."""

    valid = mod.score_trace(_live_trace())
    assert valid["scores"]["provenance_validity"] == 1.0
    assert valid["accepted_for_patch_evidence"] is True

    development_proxy = _live_trace(
        solve_provenance="development_proxy",
        live_agent_attempts=[
            {
                "policy": "OfflineSolver",
                "runtime_self_discovery": False,
                "hand_built_adapter_used": True,
                "offline_source_reading_used": False,
                "offline_ground_truth_bfs_used": False,
                "per_game_bfs_used": False,
            }
        ],
    )
    proxy_score = mod.score_trace(development_proxy)
    assert proxy_score["scores"]["provenance_validity"] == 0.0
    assert proxy_score["accepted_for_patch_evidence"] is False
    assert "development_proxy" in proxy_score["provenance_blockers"]

    outer_loop = _live_trace(
        solve_provenance="outer_loop_re",
        live_agent_attempts=[
            {
                "policy": "E3AgentPolicy",
                "runtime_self_discovery": True,
                "offline_source_reading_used": True,
                "offline_ground_truth_bfs_used": False,
                "per_game_bfs_used": False,
                "hand_built_adapter_used": False,
            }
        ],
    )
    outer_score = mod.score_trace(outer_loop)
    assert outer_score["scores"]["provenance_validity"] == 0.0
    assert outer_score["accepted_for_patch_evidence"] is False
    assert "outer_loop_re" in outer_score["provenance_blockers"]
    assert "offline_source_reading_used" in outer_score["provenance_blockers"]


def test_scenario_report_5228_process_rubric_scores_independent_of_reward() -> None:
    """SCENARIO-REPORT-5228-PROCESS-RUBRIC: no-bank traces still get process scores."""

    scored = mod.score_trace(_live_trace())

    assert scored["new_levels_banked"] == 0
    assert scored["scores"]["skill_selection"] > 0.6
    assert scored["scores"]["skill_following"] > 0.6
    assert scored["scores"]["reflection_retry_quality"] > 0.4
    assert set(mod.RUBRIC_FIELDS) <= set(scored["scores"])


def test_scenario_report_5228_patch_decision_requires_concrete_nonduplicate_patch() -> None:
    """SCENARIO-REPORT-5228-PATCH-DECISION: usable rubric does not imply patch readiness."""

    memory = {
        "consumer_ready": True,
        "known_arc_nulls": {
            "reproducible_total_levels_delta": 0,
            "new_levels_banked": [],
        },
        "provenance_requirements": {
            "accepted": ["live_agent_self_discovery"],
            "blocked": ["development_proxy", "outer_loop_re", "offline_bfs", "hand_game_adapter"],
        },
        "rubric_fields": list(mod.RUBRIC_FIELDS),
    }
    registry = {"present": True, "reproducible_total_levels": 69, "games": {"re86": 2}}
    rubric = mod.build_rubric(
        traces=[_live_trace()],
        memory_setup=memory,
        registry_summary=registry,
    )
    decision = mod.recommend_patch(rubric)

    assert rubric["arc_skill_rubric_usable"] is True
    assert decision["recommended_live_patch_available"] is False
    assert "no scored live trace supplied a concrete patch proposal" in decision[
        "recommended_patch_summary"
    ]

    proposed = _live_trace(
        proposed_live_patch="Enable a bounded active-probe retry only after E3 stalls.",
        duplicate_solve_avoided=True,
    )
    ready = mod.recommend_patch(
        mod.build_rubric(traces=[proposed], memory_setup=memory, registry_summary=registry)
    )
    assert ready["recommended_live_patch_available"] is True
    assert ready["recommended_patch_summary"] == (
        "Enable a bounded active-probe retry only after E3 stalls."
    )


def test_req_report_5228_run_writes_required_artifacts(tmp_path: Path) -> None:
    """REQ-REPORT-5228: run writes the required bare booleans and rubric path."""

    result_path = tmp_path / "experiment_5228.json"
    rubric_path = tmp_path / "rubric.json"
    artifact = mod.run(root=REPO, result_path=result_path, rubric_path=rubric_path)

    assert result_path.exists()
    assert rubric_path.exists()
    on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert isinstance(artifact["arc_skill_rubric_usable"], bool)
    assert artifact["arc_skill_rubric_usable"] is True
    assert isinstance(artifact["recommended_live_patch_available"], bool)
    assert artifact["recommended_live_patch_available"] is False
    assert artifact["registry_precheck_done"] is True
    assert artifact["duplicate_solve_target_avoided"] is True
    assert artifact["no_outer_loop_re_used"] is True
    assert artifact["inference_substrate"] == "live_trace_process_rubric"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["rubric_path"] == str(rubric_path)
    assert set(mod.PROVENANCE_FIELDS) <= set(artifact["provenance_fields"])
    assert artifact["recommended_patch_summary"]
