"""Tests for Exp 4712 perception-grounded lp85 L2 structural goal.

Spec refs: REQ-ARC-WMTE-4712,
SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL,
SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
pytestmark = pytest.mark.memory_watchdog_skip


def _alignment_grid(*, second_aligned: bool = False) -> np.ndarray:
    grid = np.full((32, 32), 3, dtype=np.int16)
    grid[4:28, 4:28] = 4
    grid[10:12, 10:12] = 11
    grid[20:22, 20:22] = 11

    def add_corner_piece(x0: int, y0: int) -> None:
        for x, y in ((x0, y0), (x0 + 3, y0), (x0, y0 + 3), (x0 + 3, y0 + 3)):
            grid[y, x] = 11

    add_corner_piece(9, 9)
    add_corner_piece(19 if second_aligned else 15, 19 if second_aligned else 15)
    return grid


def test_req_arc_wmte_4712_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4712: OpenSpec anchors the structural L2 goal artifact."""

    from carnot import experiment_4712_perception_grounded_l2_goal_lp85 as exp4712

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4712",
        "SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL",
        "SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING",
        exp4712.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in exp4712.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4712_detector_aligns_corner_pieces_to_goal_sprites() -> None:
    """SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL: pieces and goals are detected objects."""

    from carnot.agentic.arc_value_learner import (
        detect_marker_pair_shape_alignment,
        structural_piece_sprite_alignment_goal,
    )

    partial = detect_marker_pair_shape_alignment(_alignment_grid(second_aligned=False))

    assert partial["goal_expression"] == "structural_piece_sprite_alignment_over_detected_objects"
    assert partial["object_centric_slot_count"] > 0
    assert partial["piece_count"] == 2
    assert partial["goal_count"] == 2
    assert partial["aligned_piece_count"] == 1
    assert partial["complete"] is False
    assert structural_piece_sprite_alignment_goal(_alignment_grid(second_aligned=False)) is False

    complete = detect_marker_pair_shape_alignment(_alignment_grid(second_aligned=True))

    assert complete["piece_count"] == 2
    assert complete["goal_count"] == 2
    assert complete["aligned_piece_count"] == 2
    assert complete["complete"] is True
    assert structural_piece_sprite_alignment_goal(_alignment_grid(second_aligned=True)) is True


def test_scenario_arc_wmte_4712_reinduction_uses_structural_goal_for_satisfiability() -> None:
    """SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING: structural goal overrides flat false goal."""

    from carnot.agentic.arc_executable_world_model import Transition
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction
    from carnot.agentic.arc_value_learner import structural_alignment_goal_candidate

    start = _alignment_grid(second_aligned=False)
    done = _alignment_grid(second_aligned=True)

    class FakeProposer:
        model_specs = "Qwen3.5-9B-MTP GGUF (/models/Qwen3.5-9B-Q4_K_M.gguf)"

        def induce(
            self,
            _game: str,
            _transitions: list[Any],
            _cell: int,
            *,
            previous_level_complete_grid: np.ndarray | None = None,
        ) -> tuple[bool, str]:
            return True, "ok"

        def refactor(self, _game: str, _counterexample: Any) -> tuple[bool, str]:
            return True, "ok"

    def engine(grid: np.ndarray, action: int, _data: Any) -> np.ndarray:
        if action == 1:
            return done.copy()
        return np.asarray(grid).copy()

    def plan_in_model(
        model_engine: Any, goal: Any, root_grid: np.ndarray
    ) -> list[dict[str, Any]] | None:
        next_grid = model_engine(root_grid.copy(), 1, None)
        return [{"action": 1, "data": None}] if goal(next_grid) else None

    result = execute_bounded_llm_reinduction(
        game="lp85",
        transitions=[
            Transition(start, 1, None, done, 1, 1),
            Transition(done, 1, None, done, 1, 1),
        ],
        cell=1,
        root_grid=start,
        proposer=FakeProposer(),
        candidate_provider=lambda model_engine, model_goal: [
            ("loaded", model_engine, model_goal)
        ],
        load_engine=lambda _game: (engine, lambda _grid: False),
        plan_in_model=plan_in_model,
        max_rounds=1,
        min_heldout_accuracy=1.0,
        structural_goal_provider=structural_alignment_goal_candidate,
    )

    assert result.planned is True
    assert result.goal_predicate_satisfiable is True
    assert result.goal_expression == "structural_piece_sprite_alignment_over_detected_objects"
    assert result.structural_goal_diagnostics["piece_count"] == 2
    assert result.rounds[0]["goal_expression"] == result.goal_expression
    assert result.rounds[0]["goal_predicate_satisfiable"] is True


def test_scenario_arc_wmte_4712_e3_levelup_passes_structural_goal_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING: live policy passes detector candidate."""

    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic.arc_llm_reinduction import LlmReinductionResult

    captured: dict[str, Any] = {}

    def fake_reinduction(**kwargs: Any) -> LlmReinductionResult:
        captured.update(kwargs)
        candidate = kwargs["structural_goal_provider"](kwargs["root_grid"])
        assert candidate is not None
        return LlmReinductionResult(
            planned=False,
            model_specs="Qwen3.5-9B-MTP GGUF",
            goal_predicate_satisfiable=True,
            goal_expression=candidate["goal_expression"],
            structural_goal_diagnostics=candidate["diagnostics"],
            skipped="no_reachable_plan_after_refinement",
        )

    monkeypatch.setattr(agent, "execute_bounded_llm_reinduction", fake_reinduction)

    policy = agent.E3AgentPolicy(
        "lp85",
        proposer=SimpleNamespace(model_specs="Qwen3.5-9B-MTP GGUF"),
        target_levels=2,
        program_synthesis_filter=False,
    )
    policy._pending_induction_reason = "level_up_reinduction"
    policy._start_level = 0
    policy._current_goal_level = 2
    policy._previous_level_complete_grid = np.zeros((2, 2), dtype=np.int16)
    policy.root_grid = _alignment_grid(second_aligned=False)
    policy.transitions = [
        SimpleNamespace(
            grid=policy.root_grid,
            action=1,
            data=None,
            next_grid=policy.root_grid,
            level_before=1,
            level_after=1,
        )
    ]

    policy._induce_and_plan()

    attempt = policy.induction_attempts[-1]
    assert captured["structural_goal_provider"] is not None
    assert attempt["goal_expression"] == "structural_piece_sprite_alignment_over_detected_objects"
    assert attempt["structural_goal_diagnostics"]["piece_count"] == 2
    assert attempt["goal_predicate_satisfiable"] is True
