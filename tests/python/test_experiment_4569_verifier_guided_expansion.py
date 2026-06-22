"""Tests for Exp 4569 verifier-guided frontier expansion.

Spec refs: REQ-CAPSTONE-4569, SCENARIO-CAPSTONE-4569,
SCENARIO-CAPSTONE-4569-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from carnot import experiment_4569_verifier_guided_expansion as mod
from carnot.agentic.arc_discriminative_router import (
    CrossGameDiscriminativeExpansionPriority,
    RandomExpansionPriority,
)
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "offline_arcade": True,
        "arc_value_learner_discriminative_import": True,
        "trained_verifier_loadable": True,
        "cross_game_corpus_loadable": True,
        "offline_env_public_games": list(games),
        "leaderboard_submission": False,
    }


def _runner_factory(
    solved_by_mode: Mapping[str, set[str]],
    expansions_by_mode: Mapping[str, Mapping[str, int]] | None = None,
):
    expansions_by_mode = expansions_by_mode or {}

    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            solved = signature in solved_by_mode.get(mode, set())
            reached = 1 if solved else 0
            expansions = int(expansions_by_mode.get(mode, {}).get(signature, 12))
            return {
                "game": game,
                "variant_signature": signature,
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "winner_generated": solved,
                "reached_level": reached,
                "actions": 5 if solved else 9,
                "expanded_states_to_goal": expansions if solved else None,
                "expansions_used": expansions,
                "max_expansions": 32,
                "solution_labels": ["ACTION1"] if solved else [],
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": reached,
                    "reproduced": solved,
                },
                "blocked_reason": "",
                "expansion_priority_mode": mode,
            }

        return run

    return _runner


def test_req_capstone_4569_spec_declares_expansion_contract() -> None:
    """REQ-CAPSTONE-4569: OpenSpec declares the verifier-guided expansion schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4569" in spec
    assert "SCENARIO-CAPSTONE-4569" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4569_discriminative_priority_scores_frontier_frames() -> None:
    """REQ-CAPSTONE-4569: the verifier priority scores nodes, not action order."""

    class FakeVerifier:
        def proba_features(self, features: list[float]) -> float:
            return 0.9 if features[0] == pytest.approx(3.0) else 0.2

    near = SimpleNamespace(frame=[[3]], levels_completed=0, available_actions=[1])
    far = SimpleNamespace(frame=[[1]], levels_completed=0, available_actions=[1])
    priority = CrossGameDiscriminativeExpansionPriority(
        FakeVerifier(),
        featurize=lambda frame: [float(frame.frame[0][0])],
    )

    assert priority.verifier_is_oracle is False
    assert priority(near) < priority(far)
    assert priority.proba(near) > priority.proba(far)
    assert RandomExpansionPriority(seed=4569)(near) == RandomExpansionPriority(seed=4569)(near)


class _ToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return self._frame()

    def _frame(self) -> Any:
        values = {"root": 0, "bad": 1, "near": 2, "bad2": 3, "win": 9}
        level = 1 if self.state == "win" else 0
        return SimpleNamespace(
            frame=[[values[self.state]]],
            levels_completed=level,
            available_actions=[] if self.state == "win" else [1, 2],
            state="",
        )

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            return int(action.value)
        text = str(action)
        if "ACTION" in text:
            return int(text.rsplit("ACTION", 1)[-1])
        return int(action)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        aid = self._action_id(action)
        transitions = {
            ("root", 1): "bad",
            ("root", 2): "near",
            ("bad", 1): "bad2",
            ("bad", 2): "bad2",
            ("near", 1): "win",
            ("near", 2): "win",
            ("bad2", 1): "bad2",
            ("bad2", 2): "bad2",
        }
        self.state = transitions[(self.state, aid)]
        return self._frame()


def test_req_capstone_4569_graph_explore_uses_expansion_priority_for_generation() -> None:
    """REQ-CAPSTONE-4569: the priority changes which generated frontier node expands."""

    baseline_stats: dict[str, Any] = {}
    baseline, baseline_level = graph_explore_solve_v2(
        _ToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        stats=baseline_stats,
    )
    priority_stats: dict[str, Any] = {}
    guided, guided_level = graph_explore_solve_v2(
        _ToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        expansion_priority=lambda frame: 0.0 if frame.frame == [[2]] else 10.0,
        stats=priority_stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert guided_level == 1
    assert guided == [{"action": 2, "data": None}, {"action": 1, "data": None}]
    assert priority_stats["expansions"] == 3
    assert priority_stats["max_expansions"] == 3


def test_scenario_capstone_4569_artifact_reports_success_delta_and_controls(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4569: verifier expansion beats baseline and random priority."""

    games = ("g1", "g2", "g3", "g4")
    baseline = {"g1~color01"}
    verifier = {"g1~color01", "g2~color01", "g3~color01", "g4~color01"}
    random = {"g1~color01"}
    expansions = {
        "baseline": {"g1~color01": 9},
        "verifier_expansion": {
            "g1~color01": 4,
            "g2~color01": 5,
            "g3~color01": 6,
            "g4~color01": 7,
        },
        "random_priority": {"g1~color01": 11},
    }

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"baseline": baseline, "verifier_expansion": verifier, "random_priority": random},
            expansions,
        ),
        n_bootstrap=80,
    )

    assert artifact["honest_verdict"] == (
        "success: verifier_guided_expansion_generic_transfer_1.000_above_0.04"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["generic_transfer_rate_baseline"] == pytest.approx(0.25)
    assert artifact["generic_transfer_rate_with_expansion"] == pytest.approx(1.0)
    assert artifact["transfer_delta"] == pytest.approx(0.75)
    assert artifact["solve_rate_preserved"] is True
    assert artifact["random_priority_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["winner_generated"]["with_expansion"] is True
    assert artifact["expanded_states_to_goal_with_vs_without"]["strictly_lower_than_without"] is True
    assert artifact["chosen_submitted_config"] == "enable_verifier_guided_expansion"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4569_honest_null_keeps_generation_gap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4569-FIELD-PRINCIPLES: zero delta is an explicit null."""

    games = ("g1", "g2")
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {
                "baseline": solved,
                "verifier_expansion": solved,
                "random_priority": set(),
            }
        ),
        n_bootstrap=40,
        dominant_weight_family="predicate_distance",
    )

    assert artifact["honest_verdict"] == (
        "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened"
    )
    assert artifact["transfer_delta"] == 0.0
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["random_priority_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["missing_verifier_gaps"] == [
        "verifier_guided_expansion_no_value_added; winner_not_generated_for=1; strongest_weight_family=predicate_distance"
    ]
    assert mod.validate_artifact(artifact) == []
