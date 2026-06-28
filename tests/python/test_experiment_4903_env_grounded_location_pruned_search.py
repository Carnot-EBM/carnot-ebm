"""Tests for Exp 4903 env-grounded location-pruned first-win search.

Spec refs: REQ-ARC-WMTE-4903,
SCENARIO-ARC-WMTE-4903-LOCATION-PRIOR-NOT-VALUE,
SCENARIO-ARC-WMTE-4903-FORK-VERDICT,
SCENARIO-ARC-WMTE-4903-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4903_env_grounded_location_pruned_search as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _act(action: int, data: dict[str, int] | None = None) -> dict[str, Any]:
    return {"action": action, "data": data}


def _ground_truth() -> dict[str, list[dict[str, Any]]]:
    return {
        "cd82": [_act(1)],
        "cn04": [_act(2)],
        "ls20": [_act(3)],
        "m0r0": [_act(4)],
        "tu93": [_act(2)],
    }


def _generator_ok(backend: str = "gpu0_cuda") -> dict[str, Any]:
    return {
        "ok": True,
        "generator_backend": backend,
        "backend": backend,
        "server": f"/fake/{backend}/llama-server",
        "model": "Qwen3.5-9B-MTP",
        "igpu_required": False,
        "launch_env_cuda_visible_devices": "0" if backend == "gpu0_cuda" else None,
    }


def _a1_artifact() -> dict[str, Any]:
    rows = {
        game: {
            "cell_recall": 0.727273,
            "value_acc_code_baseline": 0.1,
            "value_acc_decision_need": 0.0,
            "value_delta": -0.1,
            "planned_bucket": "NEVER_ENUMERATED",
            "migrated": False,
            "heldout_transition_ids": ["heldout:0", "heldout:1"],
        }
        for game in ("cd82", "cn04", "ls20", "m0r0")
    }
    return {
        "experiment_id": 4892,
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "engine_cell_recall_median": 0.727273,
        "per_game_value_gap": rows,
        "positive_control_value_gap": {
            "game": "tu93",
            "cell_recall": 0.8,
            "heldout_transition_ids": ["heldout:0", "heldout:1"],
        },
    }


def _baseline_artifact() -> dict[str, Any]:
    return {
        "experiment_id": 4896,
        "first_win_baseline": 0.04,
        "heldout_first_win_rate": 0.052632,
    }


def _row(
    game: str,
    *,
    baseline: float = 0.0,
    env: float = 0.0,
    bucket: str = "NEVER_ENUMERATED",
    actions: int | None = None,
    states: int = 1,
    migrated: bool | None = None,
) -> dict[str, Any]:
    return {
        "game": game,
        "first_win_baseline": baseline,
        "first_win_env_grounded": env,
        "delta": round(env - baseline, 6),
        "actions_to_first_win": actions,
        "states_expanded": states,
        "bucket": bucket,
        "migrated": bool(env > 0 and bucket == "COVERED") if migrated is None else bool(migrated),
        "baseline_bucket": "NEVER_ENUMERATED",
        "prior_top_rank_score": 1.0 if env else 0.25,
        "location_ranker_non_degenerate": bool(env),
        "change_value_predictions_used": 0,
        "real_env_value_reads": max(int(states), 1),
        "live_path_methods_called": [
            "StepwiseExplorer.action_prior",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _control(non_degenerate: bool = True) -> dict[str, Any]:
    return {
        "game": "tu93",
        "location_ranker_non_degenerate": bool(non_degenerate),
        "prior_top_rank_score": 1.0 if non_degenerate else 0.0,
        "true_changing_action_rank": 1 if non_degenerate else None,
    }


def test_req_arc_wmte_4903_spec_declares_env_grounded_contract() -> None:
    """REQ-ARC-WMTE-4903: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4903",
        "SCENARIO-ARC-WMTE-4903-LOCATION-PRIOR-NOT-VALUE",
        "SCENARIO-ARC-WMTE-4903-FORK-VERDICT",
        "SCENARIO-ARC-WMTE-4903-PARTIAL-CHECKPOINT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4903_location_prior_reads_real_env_values() -> None:
    """SCENARIO-ARC-WMTE-4903-LOCATION-PRIOR-NOT-VALUE: engine ranks WHERE, env supplies VALUE."""

    start = np.array([[0, 0], [0, 0]], dtype=int)

    def wrong_value_location_engine(grid: np.ndarray, action: int, data: Any) -> np.ndarray:
        out = np.asarray(grid).copy()
        if action == 2:
            out[0, 0] = 3
        return out

    def real_transition(grid: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
        out = np.asarray(grid).copy()
        if int(candidate["action"]) == 2:
            out[0, 0] = 9
        return out

    result = mod.interleaved_env_grounded_search(
        start,
        engine=wrong_value_location_engine,
        legal_actions=lambda _grid: [_act(1), _act(2)],
        real_transition=real_transition,
        is_goal=lambda grid: bool(np.asarray(grid)[0, 0] == 9),
        progress_score=lambda grid: float(np.asarray(grid)[0, 0] == 9),
        action_budget=2,
        top_k=1,
        random_seed=7,
    )

    assert result["first_win_reached"] is True
    assert result["actions_to_first_win"] == 1
    assert result["best_path"] == [_act(2)]
    assert result["change_location_prior_used_not_value"] is True
    assert result["change_value_predictions_used"] == 0
    assert result["real_env_value_reads"] == 1


def test_scenario_arc_wmte_4903_fork_verdict_uses_delta_ci_migration_and_cost() -> None:
    """SCENARIO-ARC-WMTE-4903-FORK-VERDICT: delta CI, migration, and cost name the fork."""

    unlocks = mod.build_artifact(
        per_game_first_win={
            "cd82": _row("cd82", baseline=0.0, env=1.0, bucket="COVERED", actions=4),
            "cn04": _row("cn04", baseline=0.0, env=1.0, bucket="COVERED", actions=6),
            "ls20": _row("ls20", baseline=0.0, env=1.0, bucket="COVERED", actions=8),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
        bounded_action_cost=20,
    )
    budget_bound = mod.build_artifact(
        per_game_first_win={
            "cd82": _row("cd82", baseline=0.0, env=1.0, bucket="COVERED", actions=40),
            "cn04": _row("cn04", baseline=0.0, env=1.0, bucket="COVERED", actions=42),
            "ls20": _row("ls20", baseline=0.0, env=1.0, bucket="COVERED", actions=44),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
        bounded_action_cost=20,
    )
    null = mod.build_artifact(
        per_game_first_win={
            "cd82": _row("cd82", baseline=0.0, env=0.0),
            "cn04": _row("cn04", baseline=0.0, env=0.0),
            "ls20": _row("ls20", baseline=0.0, env=0.0),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert unlocks["fork_verdict"] == "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"
    assert unlocks["honest_verdict"] == "success_env_grounded_search_first_win_unlocked_1.000000"
    assert unlocks["value_grounded_first_win_delta_median"] == 1.0
    assert unlocks["value_grounded_first_win_delta_ci95"] == [1.0, 1.0]
    assert unlocks["coverage_migration_count"] == 3
    assert unlocks["median_actions_to_first_win"] == 6.0
    assert budget_bound["fork_verdict"] == "SEARCH_BUDGET_BOUND"
    assert null["fork_verdict"] == "WALL_DEEPER_THAN_VALUE_PREDICTION"
    assert null["retire_if_same_verdict"] is True
    assert mod.artifact_schema_errors(unlocks) == []
    assert mod.artifact_schema_errors(budget_bound) == []
    assert mod.artifact_schema_errors(null) == []


def test_req_arc_wmte_4903_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4903: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_first_win={
            "cd82": _row("cd82", baseline=0.0, env=1.0, bucket="COVERED", actions=4),
            "cn04": _row("cn04", baseline=0.0, env=1.0, bucket="COVERED", actions=6),
            "ls20": _row("ls20", baseline=0.0, env=1.0, bucket="COVERED", actions=8),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )
    malformed = dict(artifact)
    malformed.update(
        {
            "honest_verdict": "not_terminal",
            "fork_verdict": "MAYBE",
            "value_grounded_first_win_delta_median": {"value": 1.0},
            "per_game_first_win": {"cd82": {"bucket": "MAYBE"}},
            "change_location_prior_used_not_value": False,
            "positive_control_non_degenerate": False,
            "planner_blind_to_banked_answer": False,
            "verifier_is_oracle": True,
            "live_path_reachable": False,
            "generator_backend": "cpu",
            "solve_provenance": "live_agent_self_discovery",
            "checkpoint_emitted": "yes",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "model_specs": [],
            "reproducibility_checksum": "sha256:bad",
        }
    )

    errors = mod.artifact_schema_errors(malformed)

    assert "honest_verdict_terminal_prefix" in errors
    assert "fork_verdict" in errors
    assert "value_grounded_first_win_delta_median" in errors
    assert "per_game_first_win.cd82.bucket" in errors
    assert "change_location_prior_used_not_value" in errors
    assert "positive_control_non_degenerate" in errors
    assert "planner_blind_to_banked_answer" in errors
    assert "verifier_is_oracle" in errors
    assert "live_path_reachable" in errors
    assert "generator_backend" in errors
    assert "solve_provenance" in errors
    assert "checkpoint_emitted" in errors
    assert "inference_substrate" in errors
    assert "model_specs" in errors
    assert "reproducibility_checksum" in errors
    assert mod.artifact_schema_errors({})[0].startswith("missing_field:")


def test_req_arc_wmte_4903_run_blocks_and_checkpoints_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4903-PARTIAL-CHECKPOINT: blocked and partial runs stay schema-valid."""

    common = {
        "root": tmp_path,
        "a1_artifact_loader": lambda _root: _a1_artifact(),
        "baseline_loader": lambda _root: _baseline_artifact(),
        "ground_truth_loader": lambda _root: _ground_truth(),
        "environment_games_loader": lambda _arcade: set(_ground_truth()),
        "game_measurer": lambda game, **_kwargs: _row(
            game, baseline=0.0, env=1.0, bucket="COVERED", actions=4
        ),
        "positive_control_runner": lambda **_kwargs: _control(),
        "live_path_checker": lambda _root: True,
        "write": False,
    }
    blocked_arcade = mod.run(
        **common,
        offline_arcade_checker=lambda: False,
        generator_checker=_generator_ok,
        now=iter([1.0, 1.1]).__next__,
    )
    blocked_generator = mod.run(
        **{**common, "now": iter([2.0, 2.1]).__next__},
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": False, "detail": "missing_qwen"},
    )
    blocked_a1 = mod.run(
        **{**common, "now": iter([3.0, 3.1]).__next__, "a1_artifact_loader": lambda _root: None},
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        baseline_loader=lambda _root: _baseline_artifact(),
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(
            game, baseline=0.0, env=1.0, bucket="COVERED", actions=4
        ),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([4.0, 4.1, 5.0, 5.1]).__next__,
        write=True,
        soft_elapsed_budget_s=0.05,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_generator["honest_verdict"] == "blocked_generator_unavailable"
    assert blocked_a1["honest_verdict"] == "blocked_a1_baseline_missing"
    assert partial["partial"] is True
    assert partial["checkpoint_emitted"] is True
    assert (tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json").exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == partial
    assert mod.artifact_schema_errors(partial) == []


def test_req_arc_wmte_4903_run_full_resume_and_degenerate_control(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4903: run aggregates rows, resumes checkpoints, and retires bad controls."""

    rows = {
        "cd82": _row("cd82", baseline=0.0, env=1.0, bucket="COVERED", actions=4),
        "cn04": _row("cn04", baseline=0.0, env=1.0, bucket="COVERED", actions=6),
        "ls20": _row("ls20", baseline=0.0, env=1.0, bucket="COVERED", actions=8),
    }
    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        baseline_loader=lambda _root: _baseline_artifact(),
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: dict(rows[game]),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([10.0, 10.1, 10.2, 10.3, 75.0]).__next__,
        write=True,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )
    resumed = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        baseline_loader=lambda _root: _baseline_artifact(),
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda *_args, **_kwargs: pytest.fail("checkpoint should be reused"),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([80.0, 80.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )
    retired = mod.build_artifact(
        per_game_first_win=rows,
        positive_control_game="tu93",
        positive_control_row=_control(non_degenerate=False),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert artifact["fork_verdict"] == "ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN"
    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True
    assert retired["honest_verdict"] == "complete_env_grounded_positive_control_degenerate_retired"
    assert retired["fork_verdict"] is None


def test_req_arc_wmte_4903_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4903: final artifact is the requested env-grounded deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["positive_control_non_degenerate"] is True
    assert artifact["change_location_prior_used_not_value"] is True
    assert artifact["planner_blind_to_banked_answer"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["generator_backend"] in {"gpu0_cuda", "igpu_hip"}
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
