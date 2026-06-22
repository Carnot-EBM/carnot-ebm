"""Tests for Exp 4594 goal-energy generation proposal prior.

Spec refs: REQ-CAPSTONE-4594, SCENARIO-CAPSTONE-4594,
SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from carnot import experiment_4594_goal_energy_generation_prior as mod
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions(games: tuple[str, ...]) -> dict[str, Any]:
    return {
        "ok": True,
        "offline_arcade": True,
        "goal_distance_importable": True,
        "variant_generator_importable": True,
        "graph_explore_importable": True,
        "offline_env_public_games": list(games),
        "leaderboard_submission": False,
    }


def _runner_factory(
    solved_by_mode: Mapping[str, set[str]],
    actions_by_mode: Mapping[str, Mapping[str, int]] | None = None,
    mechanic_by_signature: Mapping[str, str] | None = None,
):
    actions_by_mode = actions_by_mode or {}
    mechanic_by_signature = mechanic_by_signature or {}

    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            solved = signature in solved_by_mode.get(mode, set())
            reached = 1 if solved else 0
            actions = int(actions_by_mode.get(mode, {}).get(signature, 7 if solved else 21))
            mechanic = mechanic_by_signature.get(signature, "keyboard_graph")
            energy_generated = mode == "with_energy" and solved
            return {
                "game": game,
                "variant_signature": signature,
                "variant": int(spec["variant"]),
                "kind": spec["kind"],
                "reflect": spec.get("reflect"),
                "attempted": True,
                "solved": solved,
                "winner_generated": solved,
                "winner_generated_by_energy_prior": energy_generated,
                "reached_level": reached,
                "actions": actions,
                "actions_to_first_levelup": actions if solved else None,
                "solution_labels": ["ACTION1"] if solved else [],
                "reproduction_gate": {
                    "game": game,
                    "claimed_level": reached,
                    "reached_level": reached,
                    "reproduced": solved,
                },
                "selected_feature_route": {
                    "mechanic_class": mechanic,
                    "approach": "diversity_graph_explore"
                    if mechanic in {"click_graph", "config_toggle"}
                    else "systematic_bfs",
                },
                "selected_approach": "diversity_graph_explore"
                if mechanic in {"click_graph", "config_toggle"}
                else "systematic_bfs",
                "executed_approach": "graph_explore_solve_v2",
                "goal_energy_mode": "structural_progress",
                "proposal_prior_enabled": mode == "with_energy",
            }

        return run

    return _runner


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


def test_req_capstone_4594_spec_declares_goal_energy_contract() -> None:
    """REQ-CAPSTONE-4594: OpenSpec declares the goal-energy artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4594" in spec
    assert "SCENARIO-CAPSTONE-4594" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_capstone_4594_graph_explore_uses_energy_as_proposal_prior_only() -> None:
    """REQ-CAPSTONE-4594: energy changes action order, not frontier expansion priority."""

    baseline, baseline_level = graph_explore_solve_v2(
        _ToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
    )
    guided_stats: dict[str, Any] = {}
    guided, guided_level = graph_explore_solve_v2(
        _ToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        structural_energy_scorer=lambda _frame, candidate: -1.0
        if candidate.action_id == 2
        else 1.0,
        stats=guided_stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert guided_level == 1
    assert guided == [{"action": 2, "data": None}, {"action": 2, "data": None}]
    assert guided_stats["proposal_prior_enabled"] is True
    assert guided_stats["expansion_priority_enabled"] is False


def test_req_capstone_4594_goal_energy_prior_scores_avatar_and_structural_actions() -> None:
    """REQ-CAPSTONE-4594: proposal energy supports avatar-goal and structural fallback."""

    frame = SimpleNamespace(
        frame=np.array([[[0, 1, 0], [0, 0, 2], [0, 0, 0]]], dtype=np.int16),
        available_actions=[3, 4, 6],
    )
    avatar_prior = mod.GoalEnergyProposalPrior(
        avatar_color=1,
        goals=[(1.0, 2.0)],
        cell=1,
        energy_mode="avatar_goal_distance",
    )
    right = avatar_prior.candidate_delta_energy(frame, ArcAction(4, None, "right"))
    left = avatar_prior.candidate_delta_energy(frame, ArcAction(3, None, "left"))

    assert right < left

    structural_prior = mod.GoalEnergyProposalPrior(
        target_points=[(1.0, 2.0)],
        action_energy_deltas={1: -2.0},
        energy_mode="structural_progress",
    )
    assert structural_prior.candidate_delta_energy(frame, ArcAction(1, None, "action")) == -2.0
    near_click = structural_prior.candidate_delta_energy(
        frame, ArcAction(6, {"x": 2, "y": 1}, "near")
    )
    far_click = structural_prior.candidate_delta_energy(
        frame, ArcAction(6, {"x": 0, "y": 2}, "far")
    )
    assert near_click < far_click


def test_scenario_capstone_4594_artifact_reports_success_delta_and_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4594: energy prior beats no-energy on targeted variants."""

    games = ("g1", "g2", "g3", "g4")
    no_energy = {"g1~color01"}
    with_energy = {"g1~color01", "g2~color01", "g3~color01", "g4~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"no_energy": no_energy, "with_energy": with_energy},
            mechanic_by_signature={
                "g2~color01": "keyboard_graph",
                "g3~color01": "click_graph",
                "g4~color01": "config_toggle",
            },
        ),
        n_bootstrap=80,
    )

    assert artifact["honest_verdict"] == "success: goal_energy_generation_prior_winner_generated_up_3"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["winner_generated_rate_no_energy"] == pytest.approx(0.25)
    assert artifact["winner_generated_rate_with_energy"] == pytest.approx(1.0)
    assert artifact["winner_generated_delta"] == pytest.approx(0.75)
    assert artifact["no_energy_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["solve_rate_preserved"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["chosen_submitted_config"] == "enable_goal_energy_generation_prior"
    assert artifact["targeted_classes"] == list(mod.TARGETED_CLASSES)
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4594_honest_null_logs_targeted_gaps(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES: zero winner delta is annotated."""

    games = ("g1", "g2")
    solved = {"g1~color01"}
    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=games,
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(games),
        variant_runner_factory=_runner_factory(
            {"no_energy": solved, "with_energy": solved},
            actions_by_mode={
                "no_energy": {"g1~color01": 9},
                "with_energy": {"g1~color01": 9},
            },
            mechanic_by_signature={"g2~color01": "keyboard_graph"},
        ),
        n_bootstrap=40,
    )

    assert artifact["honest_verdict"] == (
        "complete: goal_energy_prior_no_value_honest_null_gap_sharpened"
    )
    assert artifact["winner_generated_delta"] == 0.0
    assert artifact["actions_delta"] == 0.0
    assert artifact["no_energy_control_passed"] is True
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["missing_verifier_gaps"] == [
        "goal_energy_residual keyboard_graph:systematic_bfs:structural_progress winner_generated=0 count=1"
    ]
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4594_loads_exp4582_targeted_residuals(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4594: targeted variants are traceable to the Exp 4582 residual."""

    result = tmp_path / "results" / "experiment_4582_feature_router_transfer.json"
    result.parent.mkdir()
    result.write_text(
        """{
          "feature_router_measurement": {
            "variant_attempts": [
              {"game":"cd82","variant_signature":"cd82~color01","variant":1,"kind":"color",
               "reflect":null,"attempted":true,"solved":false,
               "approach_variant_wired":true,
               "selected_feature_route":{"mechanic_class":"keyboard_graph"},
               "selected_approach":"systematic_bfs"},
              {"game":"ar25","variant_signature":"ar25~color01","variant":1,"kind":"color",
               "reflect":null,"attempted":true,"solved":false,
               "approach_variant_wired":false,
               "selected_feature_route":{"mechanic_class":"avatar_navigation"},
               "selected_approach":"goal_distance_astar"}
            ]
          }
        }""",
        encoding="utf-8",
    )

    specs = mod.load_targeted_variant_specs(tmp_path)

    assert specs == [
        {
            "game": "cd82",
            "variant": 1,
            "kind": "color",
            "reflect": None,
            "variant_signature": "cd82~color01",
            "mechanic_class": "keyboard_graph",
            "selected_approach": "systematic_bfs",
        }
    ]


def test_req_capstone_4594_precondition_misses_and_blocked_artifact(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4594: missing required resources produce terminal blocked artifacts."""

    assert mod._first_precondition_miss({"offline_arcade": False}) == "offline_arcade"
    assert (
        mod._first_precondition_miss({"offline_arcade": True, "goal_distance_importable": False})
        == "goal_distance_import"
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        preconditions_checked={
            "offline_arcade": True,
            "goal_distance_importable": True,
            "variant_generator_importable": True,
            "graph_explore_importable": True,
            "leaderboard_submission": True,
        },
    )

    assert artifact["honest_verdict"] == "complete: blocked_leaderboard_submission"
    assert artifact["false_negative_risk_checked"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4594_validate_artifact_reports_schema_errors(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4594-FIELD-PRINCIPLES: validation rejects bad field shapes."""

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        budget=32,
        preconditions_checked=_preconditions(("g1",)),
        variant_runner_factory=_runner_factory({"no_energy": set(), "with_energy": set()}),
        n_bootstrap=0,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "not terminal"
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    bad["winner_generated_rate_with_energy"] = "1"
    bad["winner_generated_delta_ci"] = [0.0]
    bad["no_energy_control_passed"] = "yes"
    bad["targeted_classes"] = "none"
    bad["field_principles"] = {}

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "winner_generated_rate_with_energy must be a bare float" in errors
    assert "winner_generated_delta_ci must be [float, float]" in errors
    assert "no_energy_control_passed must be a bare bool" in errors
    assert "targeted_classes must be a list" in errors
    assert "missing field principle for honest_verdict" in errors


def test_req_capstone_4594_helper_branches_and_io_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4594: deterministic helpers and wrappers stay covered."""

    assert mod._rate(0, 0) == 0.0
    assert mod._as_int("bad", default=4) == 4
    assert mod._gate_reproduced({"claimed_level": 1, "reached_level": 0, "reproduced": True}) is False
    assert mod._winner_generated_by_energy({"winner_generated_by_energy_prior": True}) is True
    assert mod._winner_generated_by_energy({"solved": True, "attempted": True}) is False
    assert mod._paired_bootstrap_delta_ci([], [], random_seed=1, n_bootstrap=5) == [0.0, 0.0]

    artifact = mod.build_artifact(
        root=tmp_path,
        public_games=("g1",),
        variant_ids=(1,),
        budget=8,
        preconditions_checked=_preconditions(("g1",)),
        variant_runner_factory=_runner_factory({"no_energy": set(), "with_energy": set()}),
        n_bootstrap=0,
    )
    missing_note = dict(artifact, null_delta_methodology_note="")
    assert (
        "null_delta_methodology_note required for zero winner_generated_delta"
        in mod.validate_artifact(missing_note)
    )
    missing_principles = dict(artifact, field_principles=[])
    assert "field_principles missing" in mod.validate_artifact(missing_principles)

    path = mod.write_artifact(tmp_path, artifact=artifact)
    assert path.exists()
    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, artifact=dict(artifact, honest_verdict="bad"))

    monkeypatch.setattr(mod, "build_artifact", lambda _root: artifact)
    writes: list[Mapping[str, Any]] = []
    monkeypatch.setattr(mod, "write_artifact", lambda _root, *, artifact: writes.append(artifact))
    assert mod.run(tmp_path, write=False) == artifact
    assert writes == []
    assert mod.run(tmp_path, write=True) == artifact
    assert writes == [artifact]
