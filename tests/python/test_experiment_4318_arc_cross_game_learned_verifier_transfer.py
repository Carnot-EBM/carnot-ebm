"""Tests for Exp 4318 ARC cross-game learned value-head transfer.

Spec refs: REQ-LEARN-4318, SCENARIO-LEARN-4318.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.agentic.arc_solver_kit import OfflineSolver
from carnot import experiment_4318_arc_cross_game_learned_verifier_transfer as exp4318


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
WRAPPER_PATH = REPO / "results" / "experiment_4318_arc_cross_game_learned_verifier_transfer.py"


class _Frame:
    def __init__(self, level: int, value: int) -> None:
        self.levels_completed = level
        self.frame = np.array([[value]], dtype=np.int16)


class _Env:
    def __init__(self) -> None:
        self._game = object()
        self.value = 0

    def reset(self) -> _Frame:
        self.value = 0
        return _Frame(0, self.value)


def test_req_learn_4318_spec_declares_required_contract() -> None:
    """REQ-LEARN-4318: OpenSpec declares the transfer artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-LEARN-4318",
        "SCENARIO-LEARN-4318",
        "SCENARIO-LEARN-4318-BLOCKED",
        "experiment_4318_arc_cross_game_learned_verifier_transfer.json",
        "python/carnot/experiment_4318_arc_cross_game_learned_verifier_transfer.py",
        "baseline_solves_held_out",
        "cross_game_state_reduction_ci95",
        "blocked_insufficient_solve_traces",
        "game-invariant ARC value-representation",
    ):
        assert marker in spec
    for field in exp4318.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4318_offline_solver_accepts_frame_aware_hooks() -> None:
    """REQ-LEARN-4318: OfflineSolver can route search with frame-derived state."""

    paths_seen: list[tuple[str, ...]] = []
    verifier_levels: list[int] = []

    def action_labels(_env: Any, frame: _Frame, path: tuple[str, ...]) -> list[str]:
        paths_seen.append(path)
        return ["advance"] if frame.levels_completed == 0 else []

    def apply(env: _Env, _label: str, _frame: Any) -> _Frame:
        env.value += 1
        return _Frame(1, env.value)

    def state_key(_game: Any, frame: _Frame) -> tuple[int, int]:
        return (frame.levels_completed, int(frame.frame[0, 0]))

    def verifier(_game: Any, frame: _Frame) -> float:
        verifier_levels.append(frame.levels_completed)
        return 0.0

    solver = OfflineSolver(
        "fake",
        action_labels,
        apply,
        state_key,
        verifier=verifier,
    )

    path, states = solver.solve_level(_Env(), 0, [], depth_cap=2)

    assert path == ["advance"]
    assert states == 1
    assert paths_seen == [()]
    assert verifier_levels == [0]


def test_req_learn_4318_blocked_artifact_is_terminal_and_bare() -> None:
    """SCENARIO-LEARN-4318-BLOCKED: insufficient traces fail closed."""

    artifact = exp4318.build_blocked_artifact(
        usable_games=["r11l", "ls20"],
        missing_games=["wa30", "lp85"],
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_insufficient_solve_traces"
    assert artifact["cross_game_transfer_helps"] is False
    assert artifact["baseline_solves_held_out"] is False
    assert artifact["cross_game_state_reduction"] == 0.0
    assert artifact["cross_game_state_reduction_ci95"] == [0.0, 0.0]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["model_specs"]["blocked_reason"] == "insufficient_solve_traces"
    assert exp4318.artifact_schema_errors(artifact) == []


def test_req_learn_4318_bootstrap_summary_requires_ci_separation() -> None:
    """REQ-LEARN-4318: transfer helps only when reduction and CI lower exceed 1."""

    level_rows = [
        {
            "held_out_game": "r11l",
            "level_index": 1,
            "states_uniform": 30,
            "states_transferred": 10,
            "baseline_solved": True,
            "transferred_solved": True,
        },
        {
            "held_out_game": "ls20",
            "level_index": 1,
            "states_uniform": 20,
            "states_transferred": 10,
            "baseline_solved": True,
            "transferred_solved": True,
        },
        {
            "held_out_game": "lp85",
            "level_index": 1,
            "states_uniform": 40,
            "states_transferred": 20,
            "baseline_solved": True,
            "transferred_solved": True,
        },
    ]

    summary = exp4318.summarize_state_reduction(
        level_rows,
        random_seed=7,
        n_resamples=2000,
    )

    assert summary["baseline_solves_held_out"] is True
    assert summary["cross_game_state_reduction"] == pytest.approx(2.25)
    assert summary["cross_game_state_reduction_ci95"][0] > 1.0
    assert summary["cross_game_transfer_helps"] is True
    assert summary["per_held_out_game_reduction"]["r11l"]["state_reduction"] == pytest.approx(3.0)


def test_req_learn_4318_schema_rejects_non_bare_fields() -> None:
    """REQ-LEARN-4318-5: required gate fields stay bare and oracle-distinct."""

    artifact = exp4318.build_blocked_artifact(
        usable_games=[],
        missing_games=["r11l", "ls20", "wa30", "lp85"],
        duration_s=0.0,
    )
    bad = dict(artifact)
    bad["cross_game_transfer_helps"] = 1
    bad["baseline_solves_held_out"] = "false"
    bad["cross_game_state_reduction"] = "0.0"
    bad["cross_game_state_reduction_ci95"] = {"lo": 0.0, "hi": 0.0}
    bad["verifier_is_oracle"] = True
    bad["random_seed"] = "4318"

    errors = exp4318.artifact_schema_errors(bad)

    for field in (
        "cross_game_transfer_helps",
        "baseline_solves_held_out",
        "cross_game_state_reduction",
        "cross_game_state_reduction_ci95",
        "verifier_is_oracle",
        "random_seed",
    ):
        assert any(field in error for error in errors)


def test_req_learn_4318_runner_writes_result_and_gap_on_null(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4318-6: null transfer writes artifact and logs the verifier gap."""

    fake_artifact = exp4318.build_complete_artifact(
        level_rows=[
            {
                "held_out_game": "r11l",
                "level_index": 1,
                "states_uniform": 10,
                "states_transferred": 10,
                "baseline_solved": True,
                "transferred_solved": True,
            }
        ],
        split_specs={"r11l": {"train_games": ["ls20", "wa30"], "held_out_game": "r11l"}},
        model_weight_specs={"r11l": {"n_samples": 4, "weights": [0.0, 0.0]}},
        trace_checksums={"r11l": "sha256:a", "ls20": "sha256:b", "wa30": "sha256:c"},
        duration_s=0.5,
        n_resamples=2000,
    )
    monkeypatch.setattr(exp4318, "evaluate_leave_one_game_out", lambda _repo: fake_artifact)

    artifact = exp4318.run(repo=tmp_path, write=True)

    written = tmp_path / exp4318.OUTPUT_REL
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert artifact["cross_game_transfer_helps"] is False
    assert exp4318.GAP_ID in gaps
    assert "game-invariant ARC value representation" in gaps


def test_results_wrapper_imports_main() -> None:
    """SCENARIO-LEARN-4318: results wrapper exposes the stable CLI entrypoint."""

    namespace = runpy.run_path(str(WRAPPER_PATH), run_name="exp4318_wrapper_test")

    assert namespace["main"] is exp4318.main
