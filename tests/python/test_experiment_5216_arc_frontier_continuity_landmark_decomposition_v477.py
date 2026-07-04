"""Tests for Exp 5216 ARC frontier continuity plus landmark decomposition.

Spec refs: REQ-REPORT-5216, SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS,
SCENARIO-REPORT-5216-ARTIFACT-GATE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_5216_arc_frontier_continuity_landmark_decomposition_v477 as mod
from carnot.agentic import arc_frontier_continuity_landmarks as lm
from carnot.agentic.arc_frontier_continuity_landmarks import (
    build_frontier_continuity_landmark_bank,
)
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


class _LandmarkToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return self._frame()

    def _frame(self) -> Any:
        grids = {
            "root": np.array([[0, 0, 0]], dtype=np.int16),
            "bad": np.array([[9, 0, 0]], dtype=np.int16),
            "landmark": np.array([[0, 4, 0]], dtype=np.int16),
            "win": np.array([[0, 4, 8]], dtype=np.int16),
        }
        return SimpleNamespace(
            frame=grids[self.state],
            levels_completed=1 if self.state == "win" else 0,
            available_actions=[] if self.state == "win" else [1, 2],
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
            ("root", 2): "landmark",
            ("bad", 1): "bad",
            ("bad", 2): "bad",
            ("landmark", 1): "win",
            ("landmark", 2): "landmark",
        }
        self.state = transitions[(self.state, aid)]
        return self._frame()


def _grid_of(frame: Any) -> np.ndarray:
    return np.asarray(frame.frame)


def _energy(frame: Any) -> float:
    return float(np.count_nonzero(_grid_of(frame) != np.array([[0, 4, 8]], dtype=np.int16)))


def _step(action: int) -> dict[str, Any]:
    return {"action": int(action), "data": None}


def _runtime_log() -> list[dict[str, Any]]:
    root = SimpleNamespace(frame=np.array([[0, 0, 0]], dtype=np.int16), levels_completed=0)
    landmark = SimpleNamespace(frame=np.array([[0, 4, 0]], dtype=np.int16), levels_completed=0)
    win = SimpleNamespace(frame=np.array([[0, 4, 8]], dtype=np.int16), levels_completed=1)
    return [
        {
            "frame_before": root,
            "frame_after": landmark,
            "path_before": [],
            "path_after": [_step(2)],
            "action": _step(2),
            "level_before": 0,
            "level_after": 0,
        },
        {
            "frame_before": landmark,
            "frame_after": win,
            "path_before": [_step(2)],
            "path_after": [_step(2), _step(1)],
            "action": _step(1),
            "level_before": 0,
            "level_after": 1,
        },
    ]


def _arm(
    *,
    reached: int,
    states: int,
    reproduced: bool = False,
    provenance: str = "development_proxy",
) -> dict[str, Any]:
    return {
        "reached_level": int(reached),
        "states_expanded": int(states),
        "solve_provenance": provenance,
        "reproduction_gate": {
            "game": "bp35",
            "reached_level": int(reached),
            "claimed_level": int(reached),
            "reproduced": bool(reproduced),
        },
    }


def test_req_report_5216_spec_declares_required_schema() -> None:
    """REQ-REPORT-5216: OpenSpec declares the required Exp 5216 artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5216" in spec
    assert "SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS" in spec
    assert "SCENARIO-REPORT-5216-ARTIFACT-GATE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "outer_loop_re is not headline-eligible" in spec


def test_scenario_report_5216_landmark_subsearch_uses_same_budget() -> None:
    """SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS: split search reaches the induced landmark first."""

    baseline_stats: dict[str, Any] = {}
    baseline, baseline_level = graph_explore_solve_v2(
        _LandmarkToyEnv(),
        0,
        max_expansions=4,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
        stats=baseline_stats,
    )
    bank = build_frontier_continuity_landmark_bank(
        root_frame=_LandmarkToyEnv().reset(),
        transition_logs=_runtime_log(),
        grid_of=_grid_of,
        goal_energy=_energy,
        enable_frontier_continuity=False,
        enable_landmark_decomposition=True,
    )
    seeded_stats: dict[str, Any] = {}
    seeded, seeded_level = graph_explore_solve_v2(
        _LandmarkToyEnv(),
        0,
        max_expansions=4,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
        frontier_seed_bank=bank,
        stats=seeded_stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert baseline_stats["expansions"] == 4
    assert seeded == [_step(2), _step(1)]
    assert seeded_level == 1
    assert seeded_stats["expansions"] == 4
    assert seeded_stats["frontier_seed_sequences_injected"] == 2
    assert seeded_stats["frontier_seed_actions_injected"] == 2
    assert bank.diagnostics()["landmark_count"] == 1


def test_scenario_report_5216_frontier_continuity_seeds_compatible_root() -> None:
    """SCENARIO-REPORT-5216-CONTINUITY-LANDMARKS: compatible prior frontiers seed the new root."""

    bank = build_frontier_continuity_landmark_bank(
        root_frame=_LandmarkToyEnv().reset(),
        transition_logs=_runtime_log(),
        grid_of=_grid_of,
        goal_energy=_energy,
        enable_frontier_continuity=True,
        enable_landmark_decomposition=False,
    )
    stats: dict[str, Any] = {}
    seeded, seeded_level = graph_explore_solve_v2(
        _LandmarkToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
        frontier_seed_bank=bank,
        stats=stats,
    )

    assert seeded == [_step(2), _step(1)]
    assert seeded_level == 1
    assert stats["frontier_seed_sequences_injected"] == 1
    assert stats["frontier_seed_actions_injected"] == 2
    assert bank.diagnostics()["continuity_sequence_count"] == 1


def test_scenario_report_5216_artifact_gate_counts_only_new_reproduced_levels() -> None:
    """SCENARIO-REPORT-5216-ARTIFACT-GATE: duplicate registry depths are not banked."""

    rows = [
        {
            "game": "bp35",
            "registry_depth": 2,
            "arms": {
                "flat_control": _arm(reached=2, states=64),
                "frontier_continuity": _arm(reached=2, states=64),
                "landmark_decomposition": _arm(reached=3, states=48, reproduced=False),
            },
        },
        {
            "game": "dc22",
            "registry_depth": 3,
            "arms": {
                "flat_control": _arm(reached=3, states=64, reproduced=True),
                "frontier_continuity": _arm(reached=3, states=60, reproduced=True),
                "landmark_decomposition": _arm(reached=4, states=40, reproduced=True),
            },
        },
    ]

    artifact = mod.build_artifact(
        per_game=rows,
        registry_depths={"bp35": 2, "dc22": 3},
        orphan_lint={"passed": True, "stdout_tail": "OK: all solver-like ARC modules are reachable"},
        duration_s=1.25,
    )

    assert artifact["target_games"] == ["bp35", "dc22"]
    assert artifact["duplicate_registry_precheck_passed"] is True
    assert artifact["offline_ground_truth_bfs"] is False
    assert artifact["read_game_source"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["new_levels_banked"] == [
        {"game": "dc22", "level": 4, "solve_provenance": "development_proxy"}
    ]
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["frontier_continuity_lift"]["dc22"]["states_expanded_delta"] == 4
    assert artifact["landmark_decomposition_lift"]["dc22"]["reached_level_delta"] == 1
    assert artifact["orphan_lint_result"].startswith("pass:")
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_report_5216_defensive_normalization_and_zero_bank_paths() -> None:
    """REQ-REPORT-5216: defensive branches keep schema honest without solve fabrication."""

    frame3d = SimpleNamespace(frame=np.array([[[1, 2]]], dtype=np.int16), levels_completed=2)
    bad_frame = SimpleNamespace(frame="bad")
    assert lm._grid2d(frame3d, _grid_of).shape == (1, 2)
    assert lm._grid2d(bad_frame, lambda _frame: (_ for _ in ()).throw(ValueError("bad"))) is None
    assert lm._data_key({"b": [object()], "a": True})[0] == ("a", True)
    assert lm._data_key(3.5) == 3.5
    assert lm._clean_step(4) == {"action": 4, "data": None}
    assert lm._path_suffix([_step(1)], [_step(2)]) == [_step(1)]
    root_sig = lm._structural_signature(np.array([[0]], dtype=np.int16))
    other_shape = lm._structural_signature(np.array([[0, 0]], dtype=np.int16))
    assert lm._compatible_signature((), root_sig) is False
    assert lm._compatible_signature(root_sig, other_shape) is False
    assert lm._energy(None, frame3d) is None
    assert lm._energy(lambda _frame: (_ for _ in ()).throw(RuntimeError("bad")), frame3d) is None
    assert lm._level({"frame_after": frame3d}, "level_after", "frame_after") == 2
    assert lm._candidate_keys([{"action": 6, "data": {"x": 1, "y": 2}}]) == {
        (6, (("x", 1), ("y", 2)))
    }
    assert lm._first_action_allowed([], []) is False
    assert lm._unique_sequences([[_step(1)], [_step(1)], [_step(2)], [_step(3)]], 2) == [
        [_step(1)],
        [_step(2)],
    ]

    bank = build_frontier_continuity_landmark_bank(
        root_frame=_LandmarkToyEnv().reset(),
        transition_logs=[
            {"frame_before": bad_frame, "frame_after": bad_frame, "path_after": [_step(9)]},
            *_runtime_log(),
        ],
        grid_of=lambda frame: np.asarray(frame.frame),
        goal_energy=_energy,
        max_sequences=2,
    )
    assert bank._frame_facts(bad_frame) is None
    assert bank.frontier_seed_sequences(bad_frame, []) == []
    assert bank.as_dict()["landmarks"][0]["goal_len"] == 1

    mismatch_bank = build_frontier_continuity_landmark_bank(
        root_frame=_LandmarkToyEnv().reset(),
        transition_logs=[
            {
                "frame_before": SimpleNamespace(
                    frame=np.array([[0, 0, 0]], dtype=np.int16), levels_completed=0
                ),
                "frame_after": SimpleNamespace(
                    frame=np.array([[0, 0, 8]], dtype=np.int16), levels_completed=1
                ),
                "path_before": [],
                "path_after": [_step(1)],
                "level_before": 0,
                "level_after": 1,
            },
            _runtime_log()[0],
        ],
        grid_of=_grid_of,
        goal_energy=_energy,
        max_sequence_len=0,
    )
    assert mismatch_bank.landmarks == []

    prefix_mismatch_bank = build_frontier_continuity_landmark_bank(
        root_frame=_LandmarkToyEnv().reset(),
        transition_logs=[
            {
                "frame_before": SimpleNamespace(
                    frame=np.array([[0, 0, 0]], dtype=np.int16), levels_completed=0
                ),
                "frame_after": SimpleNamespace(
                    frame=np.array([[0, 0, 8]], dtype=np.int16), levels_completed=1
                ),
                "path_before": [],
                "path_after": [_step(1)],
                "level_before": 0,
                "level_after": 1,
            },
            _runtime_log()[0],
        ],
        grid_of=_grid_of,
        goal_energy=_energy,
    )
    assert prefix_mismatch_bank.landmarks == []

    artifact = mod.build_artifact(
        per_game=[
            {
                "game": "bp35",
                "registry_depth": 2,
                "arms": {
                    "flat_control": _arm(reached=2, states=10),
                    "frontier_continuity": _arm(
                        reached=3, states=9, reproduced=True, provenance="outer_loop_re"
                    ),
                    "landmark_decomposition": _arm(reached=2, states=8),
                },
            }
        ],
        registry_depths={"bp35": 2},
        orphan_lint={"passed": False, "stderr_tail": "lint failed"},
        duration_s=0.1,
    )
    assert artifact["new_levels_banked"] == []
    assert artifact["reproducible_total_levels_delta"] == 0
    assert artifact["orphan_lint_result"] == "fail: lint failed"
    assert artifact["honest_verdict"].startswith("complete:")
