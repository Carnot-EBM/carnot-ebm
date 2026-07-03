"""Tests for Exp 5198 GAP-4891 MAP landmark prestage.

Spec refs: REQ-REPORT-5198, SCENARIO-REPORT-5198-MAP-PRESTAGE,
SCENARIO-REPORT-5198-THREE-ARM-GATE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot import experiment_5198_map_landmark_prestage_prototype_v476 as mod
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2
from carnot.agentic.arc_map_landmark_prestage import build_landmark_map


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


class _MapToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return self._frame()

    def _frame(self) -> Any:
        grids = {
            "root": np.array([[0, 0, 0]], dtype=np.int16),
            "bad": np.array([[7, 0, 0]], dtype=np.int16),
            "near": np.array([[0, 0, 1]], dtype=np.int16),
            "win": np.array([[0, 0, 2]], dtype=np.int16),
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
            ("root", 2): "near",
            ("bad", 1): "bad",
            ("bad", 2): "bad",
            ("near", 1): "win",
            ("near", 2): "near",
        }
        self.state = transitions[(self.state, aid)]
        return self._frame()


def _grid_of(frame: Any) -> np.ndarray:
    return np.asarray(frame.frame)


def _target_energy(frame: Any) -> float:
    grid = _grid_of(frame)
    return float(abs(2 - int(grid[0, 2])))


def _arm(
    *,
    states: int,
    banked: int,
    overhead: int,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "states_expanded": states,
        "levels_banked": banked,
        "map_overhead_steps": overhead,
        "offline_reproduced": banked > 0,
        "gate_reached": 1 + banked,
        "reproduction_gate": gate,
    }


def test_req_report_5198_spec_declares_schema_and_bare_gate() -> None:
    """REQ-REPORT-5198: OpenSpec declares the MAP prototype artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5198" in spec
    assert "SCENARIO-REPORT-5198-MAP-PRESTAGE" in spec
    assert "SCENARIO-REPORT-5198-THREE-ARM-GATE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "bare top-level boolean" in spec


def test_scenario_report_5198_map_frontier_seed_enumerates_landmark_path() -> None:
    """SCENARIO-REPORT-5198-MAP-PRESTAGE: MAP seeds trajectories before flat search."""

    baseline, baseline_level = graph_explore_solve_v2(
        _MapToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
    )
    target_region = np.array([[False, False, True]], dtype=bool)
    cognitive_map = build_landmark_map(
        _MapToyEnv(),
        start_level=0,
        max_steps=6,
        grid_of=_grid_of,
        goal_energy=_target_energy,
        target_region=target_region,
    )
    stats: dict[str, Any] = {}
    mapped, mapped_level = graph_explore_solve_v2(
        _MapToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
        frontier_seed_bank=cognitive_map,
        stats=stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert mapped_level == 1
    assert mapped == [{"action": 2, "data": None}, {"action": 1, "data": None}]
    assert cognitive_map.map_overhead_steps <= 6
    assert cognitive_map.reachable_region_count >= 3
    assert cognitive_map.relational_landmarks
    assert cognitive_map.effect_deltas[(2, None)]["target_touches"] >= 1
    assert stats["frontier_seed_enabled"] is True
    assert stats["frontier_seed_sequences_injected"] >= 1
    assert stats["frontier_seed_actions_injected"] == 2


def test_scenario_report_5198_artifact_uses_bare_bool_gate_and_negative_control() -> None:
    """SCENARIO-REPORT-5198-THREE-ARM-GATE: MAP promotion is reproduction gated."""

    rows = [
        {
            "game": "cd82",
            "prefix_level": 1,
            "arms": {
                "pruner_only": _arm(states=4000, banked=0, overhead=0),
                "map_only": _arm(
                    states=1200,
                    banked=1,
                    overhead=750,
                    gate={"game": "cd82", "reproduced": True, "reached_level": 2},
                ),
                "map_plus_pruner": _arm(states=1300, banked=0, overhead=750),
            },
        },
        {
            "game": "sk48",
            "prefix_level": 1,
            "arms": {
                "pruner_only": _arm(states=4000, banked=0, overhead=0),
                "map_only": _arm(states=4000, banked=0, overhead=750),
                "map_plus_pruner": _arm(states=4000, banked=0, overhead=750),
            },
        },
        {
            "game": "sp80",
            "prefix_level": 1,
            "arms": {
                "pruner_only": _arm(states=4000, banked=0, overhead=0),
                "map_only": _arm(states=4000, banked=0, overhead=750),
                "map_plus_pruner": _arm(states=4000, banked=0, overhead=750),
            },
        },
        {
            "game": "cn04",
            "prefix_level": 1,
            "arms": {
                "pruner_only": _arm(states=4000, banked=0, overhead=0),
                "map_only": _arm(states=4000, banked=0, overhead=750),
                "map_plus_pruner": _arm(states=4000, banked=0, overhead=750),
            },
        },
    ]

    artifact = mod.build_artifact(
        per_game=rows,
        unit_tests_still_passing=True,
        orphan_lint={"passed": True, "stdout_tail": "OK: all solver-like ARC modules"},
        duration_s=3.5,
        random_seed=mod.RANDOM_SEED,
    )

    assert artifact["lever_validated"] is True
    assert isinstance(artifact["lever_validated"], bool)
    assert artifact["field_principles"]["lever_validated"].startswith("MECHANICAL CONSTRAINT")
    assert artifact["per_arm_results"]["cd82"]["map_only"] == {
        "states_expanded": 1200,
        "levels_banked": 1,
        "map_overhead_steps": 750,
    }
    assert artifact["cn04_negative_control_stayed_clean"] is True
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["orphan_lint_result"].startswith("pass:")
    assert artifact["reproduction_gate_results"] == [
        {"game": "cd82", "arm": "map_only", "gate": {"game": "cd82", "reproduced": True, "reached_level": 2}}
    ]
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_report_5198_artifact_reports_persistent_wall_when_map_does_not_bank() -> None:
    """REQ-REPORT-5198: zero-bank MAP results are complete, not promoted."""

    rows = []
    for game in mod.GAMES:
        rows.append(
            {
                "game": game,
                "prefix_level": 1,
                "arms": {
                    "pruner_only": _arm(states=4000, banked=0, overhead=0),
                    "map_only": _arm(states=4000, banked=0, overhead=640),
                    "map_plus_pruner": _arm(states=4000, banked=0, overhead=640),
                },
            }
        )

    artifact = mod.build_artifact(
        per_game=rows,
        unit_tests_still_passing=True,
        orphan_lint={"passed": True, "stdout_tail": "OK"},
        duration_s=1.0,
        random_seed=mod.RANDOM_SEED,
    )

    assert artifact["lever_validated"] is False
    assert artifact["levels_banked"] == []
    assert artifact["reproduction_gate_results"] == []
    assert artifact["cn04_negative_control_stayed_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "enumeration wall persists" in artifact["honest_verdict"]
