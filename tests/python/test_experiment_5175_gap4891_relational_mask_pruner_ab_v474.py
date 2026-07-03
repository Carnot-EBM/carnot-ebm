"""Tests for Exp 5175 GAP-4891 relational-mask pruner A/B.

Spec refs: REQ-REPORT-5175, SCENARIO-REPORT-5175-PRUNER-AB.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot import experiment_5175_gap4891_relational_mask_pruner_ab_v474 as mod
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


class _MaskToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return self._frame()

    def _frame(self) -> Any:
        values = {"root": 0, "bad": 1, "near": 2, "win": 9}
        return SimpleNamespace(
            frame=np.array([[values[self.state]]], dtype=np.int16),
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


class _ActionOnePruner:
    def __init__(self) -> None:
        self.pruned = 0
        self.observed: list[tuple[int, bool]] = []

    @staticmethod
    def _action(label: Any) -> int:
        if isinstance(label, dict):
            return int(label["action"])
        return int(label)

    def should_prune(self, frame: Any, label: Any) -> bool:
        at_root = int(np.asarray(frame.frame)[0, 0]) == 0
        if at_root and self._action(label) == 1:
            self.pruned += 1
            return True
        return False

    def observe(
        self, frame_before: Any, label: Any, frame_after: Any, leveled_up: bool = False
    ) -> None:
        self.observed.append((self._action(label), bool(leveled_up)))

    def stats(self) -> dict[str, Any]:
        return {"pruned": self.pruned, "observed": len(self.observed)}


def _row(
    game: str,
    *,
    pruned_states: int,
    unpruned_states: int,
    pruned_reproduced: bool = False,
    unpruned_reproduced: bool = False,
) -> dict[str, Any]:
    prefix_level = 1
    return {
        "game": game,
        "prefix_level": prefix_level,
        "induce_fired": game != "cn04",
        "target_region_known": game != "cn04",
        "unpruned": {
            "states_expanded": unpruned_states,
            "reached_level": 2 if unpruned_reproduced else 1,
            "gate_reached": 2 if unpruned_reproduced else 1,
            "offline_reproduced": unpruned_reproduced,
            "reproducibility_checksum": "sha256:unpruned",
        },
        "pruned": {
            "states_expanded": pruned_states,
            "reached_level": 2 if pruned_reproduced else 1,
            "gate_reached": 2 if pruned_reproduced else 1,
            "offline_reproduced": pruned_reproduced,
            "reproducibility_checksum": "sha256:pruned",
            "pruner_stats": {"pruned": 7, "region_known": game != "cn04"},
        },
    }


def test_req_report_5175_spec_declares_required_artifact_fields() -> None:
    """REQ-REPORT-5175: OpenSpec declares the Stage-3 A/B artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-5175" in spec
    assert "SCENARIO-REPORT-5175-PRUNER-AB" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_5175_graph_explore_move_pruner_reduces_enumeration() -> None:
    """SCENARIO-REPORT-5175-PRUNER-AB: graph-explore can prune before expansion."""

    baseline, baseline_level = graph_explore_solve_v2(
        _MaskToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
    )
    pruner = _ActionOnePruner()
    stats: dict[str, Any] = {}
    pruned, pruned_level = graph_explore_solve_v2(
        _MaskToyEnv(),
        0,
        max_expansions=2,
        max_depth=4,
        heuristic=lambda _frame: 0.0,
        move_pruner=pruner,
        stats=stats,
    )

    assert baseline is None
    assert baseline_level == 0
    assert pruned_level == 1
    assert pruned == [{"action": 2, "data": None}, {"action": 1, "data": None}]
    assert pruner.observed == [(2, False), (1, True)]
    assert stats["move_pruner_enabled"] is True
    assert stats["move_pruned"] == 1
    assert stats["move_pruner_stats"] == {"pruned": 1, "observed": 2}


def test_req_report_5175_artifact_builder_reports_null_with_next_lever() -> None:
    """REQ-REPORT-5175: null results name the next GAP-4891 lever precisely."""

    rows = [
        _row("cd82", pruned_states=3200, unpruned_states=4000),
        _row("sk48", pruned_states=4000, unpruned_states=4000),
        _row("sp80", pruned_states=3900, unpruned_states=4000),
        _row("cn04", pruned_states=4000, unpruned_states=4000),
    ]

    artifact = mod.build_artifact(
        per_game=rows,
        unit_tests_still_passing=True,
        live_path_reachable=True,
        arc_orphan_solver_lint={"passed": True},
        duration_s=1.25,
        random_seed=mod.RANDOM_SEED,
    )

    assert set(mod.GAMES) == set(artifact["games_tested"])
    assert artifact["unit_tests_still_passing"] is True
    assert artifact["states_expanded_pruned"]["cd82"] == 3200
    assert artifact["states_expanded_unpruned"]["cd82"] == 4000
    assert artifact["states_expanded_reduction_pct"]["cd82"] == pytest.approx(20.0)
    assert artifact["new_level_reached_pruned"] == {
        "cd82": False,
        "sk48": False,
        "sp80": False,
        "cn04": False,
    }
    assert artifact["levels_banked"] == []
    assert artifact["cn04_negative_control_clean"] is True
    assert artifact["gap4891_status_recommendation"] == "building_with_new_lever_named"
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["live_path_reachable"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete_")
    assert "MAP" in artifact["honest_verdict"]
    assert artifact["reproducibility_checksum"].startswith("sha256:")


def test_req_report_5175_artifact_builder_reports_pruner_bank_as_success() -> None:
    """REQ-REPORT-5175: a reproduced pruned level marks GAP-4891 filled."""

    rows = [
        _row("cd82", pruned_states=1200, unpruned_states=4000, pruned_reproduced=True),
        _row("sk48", pruned_states=4000, unpruned_states=4000),
        _row("sp80", pruned_states=4000, unpruned_states=4000),
        _row("cn04", pruned_states=4000, unpruned_states=4000),
    ]

    artifact = mod.build_artifact(
        per_game=rows,
        unit_tests_still_passing=True,
        live_path_reachable=True,
        arc_orphan_solver_lint={"passed": True},
        duration_s=1.25,
        random_seed=mod.RANDOM_SEED,
    )

    assert artifact["new_level_reached_pruned"]["cd82"] is True
    assert artifact["new_level_reached_unpruned"]["cd82"] is False
    assert artifact["levels_banked"] == [
        {
            "game": "cd82",
            "new_level": 2,
            "offline_reproduced": True,
            "reproducibility_checksum": "sha256:pruned",
            "arm": "pruned",
        }
    ]
    assert artifact["gap4891_status_recommendation"] == "filled"
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["honest_verdict"].startswith("success_")
