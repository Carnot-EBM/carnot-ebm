"""Tests for Exp 4750 structural-alignment detector hardening.

Spec refs: REQ-ARC-WMTE-4712,
SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
pytestmark = pytest.mark.memory_watchdog_skip


def _duplicate_goal_grid() -> np.ndarray:
    grid = np.full((20, 24), 4, dtype=np.int16)
    for x, y in ((4, 4), (7, 4), (4, 7), (7, 7)):
        grid[y, x] = 11
    grid[5:7, 5:7] = 11
    grid[5:7, 14:16] = 11
    return grid


def test_req_arc_wmte_4712_spec_records_one_to_one_detector_gate() -> None:
    """REQ-ARC-WMTE-4712: diagnostics expose one matched goal per marker piece."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4712" in spec
    assert "goal_count <= piece_count" in spec
    assert "aligned_piece_count == piece_count" in spec
    assert "multi-exemplar fallback" in spec


def test_scenario_arc_wmte_4712_detector_deduplicates_same_color_distractor_goals() -> None:
    """SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL: extra same-color sprites do not over-segment goals."""

    from carnot.agentic.arc_value_learner import detect_marker_pair_shape_alignment

    diagnostics = detect_marker_pair_shape_alignment(_duplicate_goal_grid())

    assert diagnostics["piece_count"] == 1
    assert diagnostics["goal_count"] == 1
    assert diagnostics["goal_count"] <= diagnostics["piece_count"]
    assert diagnostics["aligned_piece_count"] == diagnostics["piece_count"]
    assert diagnostics["complete"] is True
    assert diagnostics["pairs"][0]["goal"]["bbox"] == [5, 5, 6, 6]


def test_scenario_arc_wmte_4712_lp85_real_post_l1_detector_positive_control() -> None:
    """SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL: lp85 post-L1 frame stays one-to-one paired."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750

    control = exp4750.detector_positive_control_from_l1_trace(REPO)

    assert control["available"] is True
    assert control["detector_piece_count"] == 2
    assert control["detector_goal_count"] <= control["detector_piece_count"]
    assert control["detector_goal_count"] != 42
    assert control["diagnostics"]["detected"] is True


def test_scenario_arc_wmte_4712_multi_exemplar_fallback_fits_all_l1_completions() -> None:
    """SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL: fallback banks >=2 L1-completion exemplars."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750

    fallback = exp4750.multi_exemplar_fallback_from_l1_traces(REPO)

    assert fallback["available"] is True
    assert fallback["exemplar_count"] >= 2
    assert fallback["fit_all"] is True
    assert all(
        row["detector_goal_count"] <= row["detector_piece_count"] for row in fallback["exemplars"]
    )
    assert {row["source"] for row in fallback["exemplars"]} >= {
        "experiment_4664_l1_trace",
        "arc3_lp85_offline_resolve",
    }


def test_req_arc_wmte_4712_offline_resolve_loader_ignores_malformed_moves(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4712: fallback source loading keeps malformed rows explicit and bounded."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750

    path = tmp_path / "results" / "arc3_lp85_offline_resolve.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "solution": [
                    None,
                    {"action": 6, "x": 7, "y": 9},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert exp4750._labels_from_arc3_lp85_offline_resolve(tmp_path) == [
        '{"action":6,"data":{"x":7,"y":9}}'
    ]


def test_req_arc_wmte_4712_l1_replay_reports_unavailable_when_trace_never_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4712: fallback replay does not fabricate a completed exemplar."""

    from carnot import experiment_4712_perception_grounded_l2_goal_lp85 as exp4712
    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750
    from carnot.agentic import arc_competition_agent as agent

    class FakeEnv:
        def reset(self) -> dict[str, int]:
            return {"level": 0}

        def step(self, _label: str) -> dict[str, int]:
            return {"level": 0}

    class FakeArc:
        def open_scorecard(self) -> str:
            return "scorecard"

        def make(self, _game_id: str, *, scorecard_id: str) -> FakeEnv:
            assert scorecard_id == "scorecard"
            return FakeEnv()

    monkeypatch.setattr(exp4712, "_gid", lambda _arc, _target: "lp85-fixture")
    monkeypatch.setattr(exp4712, "_apply_action_label", lambda env, label, _frame: env.step(label))
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame["level"]))

    result = exp4750._replay_l1_exemplar(
        FakeArc(),
        source="fixture",
        labels=['{"action":6,"data":{"x":1,"y":2}}'],
    )

    assert result == {
        "source": "fixture",
        "available": False,
        "reason": "trace_did_not_reach_l1_completion",
        "fit": False,
    }


def test_req_arc_wmte_4712_fallback_reports_missing_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-WMTE-4712: fallback records missing source artifacts instead of fabricating fits."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750
    from carnot.agentic import arc_solver_kit as kit

    monkeypatch.setattr(kit, "offline_arcade", lambda: object())

    fallback = exp4750.multi_exemplar_fallback_from_l1_traces(tmp_path)

    assert fallback["available"] is False
    assert fallback["exemplar_count"] == 0
    assert fallback["fit_all"] is False
    assert {row["source"] for row in fallback["missing_sources"]} == {
        "experiment_4664_l1_trace",
        "arc3_lp85_offline_resolve",
    }


def test_experiment_4750_artifact_fields_use_detector_metrics() -> None:
    """REQ-ARC-WMTE-4712: Exp4750 artifact surfaces detector and planning gates separately."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750

    upstream = {
        "honest_verdict": "complete: l2_perception_goal_no_deepening_residual_no_reachable_plan",
        "inference_substrate": "live_llm_inference",
        "preconditions_checked": {
            "offline_arcade": True,
            "structural_goal_provider_importable": True,
        },
        "goal_predicate_satisfiable": True,
        "l2_plan_reaches_goal": False,
        "offline_reproduced": False,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "detector_positive_control": {
            "diagnostics": {
                "goal_count": 1,
                "piece_count": 2,
                "aligned_piece_count": 1,
            }
        },
    }

    artifact = exp4750.artifact_from_reinduction(
        upstream,
        duration_s=61.0,
        multi_exemplar_fallback={"available": True, "exemplar_count": 2, "fit_all": True},
    )

    assert artifact["honest_verdict"].startswith("complete_detector_fixed_multi_exemplar")
    assert artifact["detector_goal_count"] == 1
    assert artifact["detector_piece_count"] == 2
    assert artifact["detector_aligned_piece_count"] == 1
    assert artifact["multi_exemplar_fallback_used"] is True
    assert artifact["multi_exemplar_fallback"]["fit_all"] is True
    assert artifact["goal_predicate_satisfiable"] is True
    assert artifact["l2_plan_reaches_goal"] is False
    assert artifact["verifier_is_oracle"] is False
    exp4750.validate_artifact(artifact)


def test_experiment_4750_artifact_branches_for_blocked_success_and_residuals() -> None:
    """REQ-ARC-WMTE-4712: Exp4750 verdicts distinguish detector, planner, and reproduction gates."""

    from carnot import experiment_4750_structural_alignment_detector_fix as exp4750

    success = exp4750.artifact_from_reinduction(
        {
            "honest_verdict": "success: upstream",
            "goal_predicate_satisfiable": True,
            "l2_plan_reaches_goal": True,
            "offline_reproduced": True,
            "duration_s": 62.0,
            "per_game": {
                "lp85": {
                    "structural_goal_diagnostics": {
                        "goal_count": 1,
                        "piece_count": 1,
                        "aligned_piece_count": 1,
                        "raw_goal_count": 42,
                    }
                }
            },
        }
    )
    assert success["honest_verdict"] == "success_detector_fixed_l2_bank"
    assert success["detector_raw_goal_count"] == 42
    assert success["residual_cause_hypothesis"] == "none"
    exp4750.validate_artifact(success)

    blocked = exp4750.artifact_from_reinduction(
        {
            "honest_verdict": "blocked_model_not_cached_qwen",
            "goal_predicate_satisfiable": False,
            "l2_plan_reaches_goal": False,
            "offline_reproduced": False,
        }
    )
    assert blocked["honest_verdict"] == "blocked_model_not_cached_qwen"
    assert blocked["residual_cause_hypothesis"] == "detector_pairing_gate_failed"
    exp4750.validate_artifact(blocked)

    unsat = exp4750.artifact_from_reinduction(
        {
            "honest_verdict": "complete: upstream",
            "goal_predicate_satisfiable": False,
            "l2_plan_reaches_goal": False,
            "offline_reproduced": False,
            "residual_cause_hypothesis": "goal_false",
            "detector_positive_control": {
                "diagnostics": {
                    "goal_count": 1,
                    "piece_count": 1,
                    "aligned_piece_count": 0,
                }
            },
        }
    )
    assert unsat["residual_cause_hypothesis"] == "goal_false"

    unreproduced = exp4750.artifact_from_reinduction(
        {
            "honest_verdict": "complete: upstream",
            "goal_predicate_satisfiable": True,
            "l2_plan_reaches_goal": True,
            "offline_reproduced": False,
            "detector_positive_control": {
                "diagnostics": {
                    "goal_count": 1,
                    "piece_count": 1,
                    "aligned_piece_count": 1,
                }
            },
        }
    )
    assert unreproduced["residual_cause_hypothesis"] == "offline_reproduction_missing"
