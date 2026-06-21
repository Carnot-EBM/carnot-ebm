"""Tests for Exp 4556 verifier-router generic-transfer wiring.

Spec refs: REQ-CAPSTONE-4556, SCENARIO-CAPSTONE-4556,
SCENARIO-CAPSTONE-4556-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from carnot import experiment_4556_verifier_router_generic_transfer as mod
from carnot.agentic.arc_discriminative_router import (
    CrossGameDiscriminativeCandidateRouter,
    RandomCandidateRouter,
    dominant_feature_family_from_weights,
)
from carnot.agentic.arc_graph_explore import rich_action_candidates
from carnot.agentic.arc_value_learner import cross_game_feature_slices_v3


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _frame() -> SimpleNamespace:
    return SimpleNamespace(
        frame=[
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 2],
            [0, 0, 0, 0, 2],
            [0, 0, 3, 0, 0],
        ],
        levels_completed=0,
        available_actions=[1, 2, 6],
    )


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


def _runner_factory(solved_by_mode: Mapping[str, set[str]]):
    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            solved = str(spec["variant_signature"]) in solved_by_mode.get(mode, set())
            reached = 1 if solved else 0
            return {
                "game": game,
                "variant_signature": spec["variant_signature"],
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "reached_level": reached,
                "actions": 5 if solved else 9,
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": reached,
                    "reproduced": solved,
                },
                "blocked_reason": "",
            }

        return run

    return _runner


def test_req_capstone_4556_spec_declares_verifier_router_contract() -> None:
    """REQ-CAPSTONE-4556: OpenSpec declares the live verifier-router artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4556" in spec
    assert "SCENARIO-CAPSTONE-4556" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4556_rich_candidates_are_ranked_by_learned_router() -> None:
    """REQ-CAPSTONE-4556: rich_action_candidates delegates final ordering to the router."""

    frame = _frame()

    class PreferActionTwo:
        verifier_is_oracle = False

        def rank(self, _frame: Any, candidates: list[Any], **_: Any) -> list[Any]:
            return sorted(candidates, key=lambda c: (int(c.action_id) != 2, c.key))

    bare = rich_action_candidates(frame, by_salience=False)
    routed = rich_action_candidates(frame, by_salience=False, candidate_router=PreferActionTwo())

    assert [action.action_id for action in bare] != [action.action_id for action in routed]
    assert routed[0].action_id == 2
    assert {action.key for action in bare} == {action.key for action in routed}


def test_req_capstone_4556_cross_game_router_uses_action_conditioned_features() -> None:
    """REQ-CAPSTONE-4556: the router scores candidates through v3 action-conditioned features."""

    class FakeVerifier:
        def proba_features(self, features: list[float]) -> float:
            action_slice = cross_game_feature_slices_v3()["action_conditioned"]
            action_features = features[action_slice[0] : action_slice[1]]
            return 0.9 if action_features[2] == pytest.approx(1.0) else 0.1

    frame = _frame()
    router = CrossGameDiscriminativeCandidateRouter(FakeVerifier())
    ranked = router.rank(frame, rich_action_candidates(frame, by_salience=False))

    assert ranked[0].action_id == 2
    assert router.verifier_is_oracle is False


def test_req_capstone_4556_random_router_is_deterministic_positive_control() -> None:
    """REQ-CAPSTONE-4556: random-router control is deterministic for the same seed."""

    frame = _frame()
    candidates = rich_action_candidates(frame, by_salience=False)
    left = RandomCandidateRouter(seed=4556).rank(frame, candidates)
    right = RandomCandidateRouter(seed=4556).rank(frame, candidates)

    assert [action.key for action in left] == [action.key for action in right]
    assert {action.key for action in left} == {action.key for action in candidates}


def test_scenario_capstone_4556_artifact_reports_success_delta_and_controls(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4556: with-router transfer beats baseline and random control."""

    games = ("g1", "g2", "g3", "g4")
    baseline = {"g1~color01"}
    verifier = {"g1~color01", "g2~color01", "g3~color01", "g4~color01"}
    random = {"g1~color01"}

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"baseline": baseline, "verifier": verifier, "random": random}
        ),
        n_bootstrap=80,
    )

    assert artifact["honest_verdict"] == (
        "success: verifier_router_generic_transfer_1.000_above_0.04"
    )
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["generic_transfer_rate_baseline"] == pytest.approx(0.25)
    assert artifact["generic_transfer_rate_with_verifier"] == pytest.approx(1.0)
    assert artifact["generic_transfer_delta"] == pytest.approx(0.75)
    assert artifact["solve_rate_preserved"] is True
    assert artifact["random_router_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["newly_solved_variants"] == ["g2~color01", "g3~color01", "g4~color01"]
    assert artifact["chosen_submitted_config"] == "enable_verifier_router"
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4556_honest_null_keeps_delta_note(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4556-FIELD-PRINCIPLES: null delta is explicit, not tautological."""

    games = ("g1", "g2")
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=12,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"baseline": solved, "verifier": solved, "random": set()}
        ),
        n_bootstrap=40,
        dominant_weight_family="frame_delta",
    )

    assert artifact["honest_verdict"] == (
        "complete: verifier_router_no_value_added_honest_null_gap_sharpened"
    )
    assert artifact["generic_transfer_delta"] == 0.0
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["random_router_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["missing_verifier_gaps"] == [
        "verifier_router_no_value_added; strongest_weight_family=frame_delta"
    ]
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4556_dominant_family_uses_v3_weight_slices() -> None:
    """REQ-CAPSTONE-4556: null artifacts name the feature family with largest weight mass."""

    slices = cross_game_feature_slices_v3()
    width = max(stop for _start, stop in slices.values())
    weights = [0.0] * (width + 1)
    start, stop = slices["predicate_distance"]
    weights[start:stop] = [2.0] * (stop - start)

    assert dominant_feature_family_from_weights(weights) == "predicate_distance"
