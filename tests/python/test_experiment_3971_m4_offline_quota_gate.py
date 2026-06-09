"""Tests for Exp 3971 ARC-AGI-3 offline quota-gate readiness.

Spec coverage: REQ-PHASE4-016, SCENARIO-PHASE4-016.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_PATH = REPO / "scripts" / "experiments" / "experiment_3971_m4_offline_quota_gate.py"
OFFLINE_EVAL_PATH = REPO / "scripts" / "experiments" / "arc3_offline_eval.py"
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(spec.name)] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class FakeGameAction(IntEnum):
    ACTION0 = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6


def _install_fake_arcengine(monkeypatch) -> None:
    enums = types.ModuleType("arcengine.enums")
    enums.GameAction = FakeGameAction
    arcengine = types.ModuleType("arcengine")
    arcengine.enums = enums
    monkeypatch.setitem(sys.modules, "arcengine", arcengine)
    monkeypatch.setitem(sys.modules, "arcengine.enums", enums)


def _seed_hybrid_trace_artifacts(root: Path) -> None:
    _write_json(
        root / "results" / "experiment_3964_r11l_incremental_l2.json",
        {
            "real_env_confirmed": True,
            "ACCURACY_levels_solved": 1,
            "solve_log": [
                {"level": 1, "piece": [2, 3], "placement": [4, 5]},
            ],
        },
    )
    _write_json(
        root / "results" / "experiment_3965_lp85_incremental_l2.json",
        {
            "real_env_confirmed": True,
            "ACCURACY_levels_solved": 1,
            "solve_log": [
                {"level": 1, "click": [7, 8]},
            ],
        },
    )
    _write_json(
        root / "results" / "experiment_3966_third_game_first_solve.json",
        {
            "real_env_confirmed": True,
            "ACCURACY_levels_solved": 1,
            "game_solved": "sc25-635fd71a",
            "solve_log": [
                {"level": 0, "action": "click", "x": 11, "y": 12},
                {"level": 0, "action": "left"},
            ],
        },
    )


def test_spec_declares_m4_quota_gate_contract() -> None:
    """REQ-PHASE4-016: OpenSpec declares Exp 3971 before implementation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-016" in spec
    assert "SCENARIO-PHASE4-016" in spec
    assert "experiment_3971_m4_offline_quota_gate.json" in spec
    assert "scored-run readiness package without submitting online" in spec


def test_hybrid_policy_registered_and_replays_banked_solve_logs(monkeypatch, tmp_path: Path) -> None:
    """REQ-PHASE4-016: the hybrid policy reuses banked real-env-confirmed mechanics."""
    _install_fake_arcengine(monkeypatch)
    _seed_hybrid_trace_artifacts(tmp_path)
    offline = _load_module(OFFLINE_EVAL_PATH, "arc3_offline_eval_for_3971_policy_test")
    monkeypatch.setattr(offline, "REPO", tmp_path)
    offline._HYBRID_TRACE_CACHE = None

    assert offline.POLICIES["hybrid"] is offline.hybrid_policy

    frame = SimpleNamespace(available_actions=[6], levels_completed=0, frame=[[0]])
    ctx = {"game_id": "r11l-495a7899", "grid_w": 64, "grid_h": 64}
    action, data = offline.hybrid_policy(frame, ctx, rng=object())
    assert action == FakeGameAction.ACTION6
    assert data == {"x": 3, "y": 2}
    action, data = offline.hybrid_policy(frame, ctx, rng=object())
    assert action == FakeGameAction.ACTION6
    assert data == {"x": 5, "y": 4}

    sc25_frame = SimpleNamespace(available_actions=[1, 2, 3, 4, 6], levels_completed=0, frame=[[0]])
    sc25_ctx = {"game_id": "sc25-635fd71a", "grid_w": 64, "grid_h": 64}
    action, data = offline.hybrid_policy(sc25_frame, sc25_ctx, rng=object())
    assert action == FakeGameAction.ACTION6
    assert data == {"x": 11, "y": 12}
    action, data = offline.hybrid_policy(sc25_frame, sc25_ctx, rng=object())
    assert action == FakeGameAction.ACTION3
    assert data is None


def test_hybrid_policy_falls_back_to_no_induction_object_click(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-016: unsolved games use the no-induction object-click fallback."""
    _install_fake_arcengine(monkeypatch)
    _seed_hybrid_trace_artifacts(tmp_path)
    offline = _load_module(OFFLINE_EVAL_PATH, "arc3_offline_eval_for_3971_fallback_test")
    monkeypatch.setattr(offline, "REPO", tmp_path)
    offline._HYBRID_TRACE_CACHE = None

    frame = SimpleNamespace(available_actions=[6], levels_completed=0, frame=[[0, 0, 0], [0, 9, 9]])
    ctx = {"game_id": "unknown-0000", "grid_w": 3, "grid_h": 2}
    action, data = offline.hybrid_policy(frame, ctx, rng=object())

    assert action == FakeGameAction.ACTION6
    assert data == {"x": 1, "y": 1}


def test_build_readiness_artifact_clears_gate_with_required_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-016: gate clears only when hybrid beats prior 0 and measured baselines."""
    exp = _load_module(EXPERIMENT_PATH, "experiment_3971_m4_success_test")
    hybrid = {
        "ACCURACY_total_levels_solved": 3,
        "EFFICIENCY_mean_action_ratio_on_solved": 0.316,
        "per_game": [{"game_id": "r11l-495a7899", "levels_solved": 1}],
    }
    random_baseline = {"ACCURACY_total_levels_solved": 0, "EFFICIENCY_mean_action_ratio_on_solved": None}
    object_baseline = {"ACCURACY_total_levels_solved": 0, "EFFICIENCY_mean_action_ratio_on_solved": None}

    artifact = exp.build_readiness_artifact(
        games=["r11l-495a7899", "lp85-305b61c3", "sc25-635fd71a"],
        seed=3971,
        budget_factor=3.0,
        budget_cap=3000,
        hybrid=hybrid,
        random_baseline=random_baseline,
        object_click_baseline=object_baseline,
        duration_s=1.25,
    )

    assert artifact["hybrid_accuracy_levels_solved"] == 3
    assert artifact["baseline_accuracy_levels_solved"] == 0
    assert artifact["hybrid_efficiency_ratio"] == 0.316
    assert artifact["quota_gate_cleared"] is True
    assert artifact["scored_run_ready_for_operator"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["documented_sota_context"]["ewm_rhae_pct"] == 58.12
    assert artifact["honest_verdict"].startswith("success:")
    output = exp.write_artifact(artifact, tmp_path / "result.json")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_readiness_artifact_reports_gap_when_baseline_not_beaten() -> None:
    """SCENARIO-PHASE4-016: not clearing the gate is a complete verdict with the gap."""
    exp = _load_module(EXPERIMENT_PATH, "experiment_3971_m4_gap_test")
    artifact = exp.build_readiness_artifact(
        games=["r11l-495a7899"],
        seed=3971,
        budget_factor=3.0,
        budget_cap=3000,
        hybrid={"ACCURACY_total_levels_solved": 1, "EFFICIENCY_mean_action_ratio_on_solved": 0.5},
        random_baseline={"ACCURACY_total_levels_solved": 2, "EFFICIENCY_mean_action_ratio_on_solved": None},
        object_click_baseline={"ACCURACY_total_levels_solved": 0, "EFFICIENCY_mean_action_ratio_on_solved": None},
        duration_s=1.0,
    )

    assert artifact["quota_gate_cleared"] is False
    assert artifact["scored_run_ready_for_operator"] is False
    assert artifact["baseline_accuracy_levels_solved"] == 2
    assert artifact["gap_to_clearing_levels"] == 2
    assert artifact["honest_verdict"] == "complete: quota_gate_not_cleared_baseline_gap2"


def test_run_uses_same_games_and_budget_for_hybrid_and_baselines(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-PHASE4-016: Exp 3971 compares all policies on the same start-here games."""
    exp = _load_module(EXPERIMENT_PATH, "experiment_3971_m4_run_test")
    monkeypatch.setattr(exp, "REPO", tmp_path)
    _write_json(
        tmp_path / "results" / "arc_agi3_game_characterization.json",
        {
            "start_here_top8": [
                {"game_id": "lp85-305b61c3"},
                {"game_id": "r11l-495a7899"},
            ]
        },
    )
    calls = []

    def fake_evaluator(**kwargs):
        calls.append(kwargs)
        policy = kwargs["policy_name"]
        levels = 3 if policy == "hybrid" else 0
        return {
            "policy": policy,
            "ACCURACY_total_levels_solved": levels,
            "EFFICIENCY_mean_action_ratio_on_solved": 0.25 if levels else None,
            "per_game": [],
        }

    artifact = exp.run(
        seed=123,
        budget_factor=4.0,
        budget_cap=111,
        write=True,
        output_path=tmp_path / "results" / "experiment_3971_m4_offline_quota_gate.json",
        evaluator=fake_evaluator,
        precondition_checker=lambda: True,
    )

    assert [call["policy_name"] for call in calls] == ["hybrid", "random", "object_click"]
    assert all(call["games"] == ["lp85-305b61c3", "r11l-495a7899", "sc25-635fd71a"] for call in calls)
    assert all(call["budget_factor"] == 4.0 and call["budget_cap"] == 111 for call in calls)
    assert all(call["seed"] == 123 and call["write"] is True for call in calls)
    assert artifact["quota_gate_cleared"] is True
    written = tmp_path / "results" / "experiment_3971_m4_offline_quota_gate.json"
    assert json.loads(written.read_text(encoding="utf-8"))["random_seed"] == 123


def test_run_blocks_when_offline_arc_env_unavailable(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-016: missing offline ARC env writes a blocked terminal artifact."""
    exp = _load_module(EXPERIMENT_PATH, "experiment_3971_m4_blocked_test")

    artifact = exp.run(
        seed=3971,
        write=True,
        output_path=tmp_path / "blocked.json",
        evaluator=lambda **kwargs: (_ for _ in ()).throw(AssertionError("evaluator should not run")),
        precondition_checker=lambda: False,
    )

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["hybrid_accuracy_levels_solved"] == 0
    assert artifact["baseline_accuracy_levels_solved"] == 0
    assert artifact["quota_gate_cleared"] is False
    assert artifact["scored_run_ready_for_operator"] is False
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == artifact
