"""Tests for Exp 4892 decision-need target value-gap fork probe.

Spec refs: REQ-ARC-WMTE-4892,
SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE,
SCENARIO-ARC-WMTE-4892-SAME-SPLIT-DELTA,
SCENARIO-ARC-WMTE-4892-FORK-VERDICT,
SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4892_decision_need_targets_value_gap as mod
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _transition(
    grid: np.ndarray,
    next_grid: np.ndarray,
    *,
    action: int = 1,
    data: dict[str, int] | None = None,
) -> Transition:
    return Transition(
        grid=np.asarray(grid),
        action=action,
        data=data,
        next_grid=np.asarray(next_grid),
        level_before=0,
        level_after=0,
    )


def _act(action: int, data: dict[str, int] | None = None) -> dict[str, Any]:
    return {"action": action, "data": data}


def _ground_truth() -> dict[str, list[dict[str, Any]]]:
    return {
        "cd82": [_act(1)],
        "cn04": [_act(2)],
        "ls20": [_act(3)],
        "m0r0": [_act(4)],
        "tu93": [_act(1)],
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


def _row(
    game: str,
    *,
    baseline: float,
    decision_need: float,
    recall: float = 0.75,
    bucket: str = "NEVER_ENUMERATED",
    fit_ids: list[int] | None = None,
    heldout_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "game": game,
        "cell_recall": recall,
        "value_acc_code_baseline": baseline,
        "value_acc_decision_need": decision_need,
        "value_delta": round(decision_need - baseline, 6),
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED" and game != "tu93",
        "author_transition_ids": fit_ids if fit_ids is not None else [0, 1],
        "heldout_transition_ids": heldout_ids if heldout_ids is not None else [2, 3],
        "baseline_transition_ids": [2, 3],
        "target_table_row_count": 3,
        "author_transition_count": 2,
        "heldout_transition_count": 2,
        "cold_transition_count": 4,
        "plan_length": 1 if bucket != "NEVER_ENUMERATED" else 0,
        "decision_need_target_kinds": ["action_effect", "object_persistence"],
        "live_path_methods_called": [
            "DecisionNeedTargetTable",
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _control(recall: float = 0.8) -> dict[str, Any]:
    return _row("tu93", baseline=0.2, decision_need=0.4, recall=recall, bucket="COVERED")


def _a1_artifact() -> dict[str, Any]:
    rows = {
        game: {
            "cell_recall_baseline": 0.7,
            "value_acc_baseline": 0.1,
            "baseline_transition_ids": ["heldout:0", "heldout:1"],
            "remeasure_transition_ids": ["heldout:0", "heldout:1"],
            "planned_bucket": "NEVER_ENUMERATED",
        }
        for game in ("cd82", "cn04", "ls20", "m0r0")
    }
    return {
        "experiment_id": 4882,
        "fork_verdict": "INDUCER_CEILING_HARD",
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "engine_cell_recall_median": 0.727273,
        "per_game_value_gap": rows,
        "positive_control_value_gap": {
            "game": "tu93",
            "cell_recall_baseline": 0.8,
            "value_acc_baseline": 0.2,
            "baseline_transition_ids": ["heldout:0", "heldout:1"],
            "remeasure_transition_ids": ["heldout:0", "heldout:1"],
        },
    }


def test_req_arc_wmte_4892_spec_declares_decision_need_contract() -> None:
    """REQ-ARC-WMTE-4892: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4892",
        "SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE",
        "SCENARIO-ARC-WMTE-4892-SAME-SPLIT-DELTA",
        "SCENARIO-ARC-WMTE-4892-FORK-VERDICT",
        "SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4892_decision_need_table_is_non_code_and_predicts_values() -> None:
    """SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE: authored targets improve value prediction."""

    fit = [
        _transition(
            np.array([[0, 0], [0, 0]], dtype=int),
            np.array([[7, 0], [0, 0]], dtype=int),
            action=6,
            data={"x": 0, "y": 0},
        )
    ]
    heldout = [
        _transition(
            np.array([[0, 0], [0, 0]], dtype=int),
            np.array([[7, 0], [0, 0]], dtype=int),
            action=6,
            data={"x": 0, "y": 0},
        )
    ]

    table = mod.DecisionNeedTargetTable.author(fit, game="toy", llm_targets=["action-effect"])
    score = mod.score_decision_need_table(table, heldout)

    assert table.representation_type == "non_code_decision_need_target_table"
    assert table.target_kinds() == ["action_effect"]
    assert score["cell_recall"] == 1.0
    assert score["changed_cell_value_accuracy"] == 1.0


def test_scenario_arc_wmte_4892_fork_verdict_uses_value_delta_ci_and_migration() -> None:
    """SCENARIO-ARC-WMTE-4892-FORK-VERDICT: value-delta CI and migration name the fork."""

    unlocks = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, decision_need=0.5, bucket="COVERED"),
            "cn04": _row("cn04", baseline=0.0, decision_need=0.5),
            "ls20": _row("ls20", baseline=0.0, decision_need=0.5),
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
    planner_gap = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, decision_need=0.5),
            "cn04": _row("cn04", baseline=0.0, decision_need=0.5),
            "ls20": _row("ls20", baseline=0.0, decision_need=0.5),
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
    invariant = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.2, decision_need=0.2),
            "cn04": _row("cn04", baseline=0.2, decision_need=0.1),
            "ls20": _row("ls20", baseline=0.2, decision_need=0.3),
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

    assert unlocks["fork_verdict"] == "REPRESENTATION_UNLOCKS_VALUE"
    assert unlocks["honest_verdict"] == "success_decision_need_value_gap_closed_0.500000"
    assert unlocks["decision_need_value_accuracy_delta_median"] == 0.5
    assert unlocks["decision_need_value_accuracy_delta_ci95"] == [0.5, 0.5]
    assert planner_gap["fork_verdict"] == "PLANNER_GAP"
    assert invariant["fork_verdict"] == "VALUE_GAP_REPRESENTATION_INVARIANT"
    assert mod.artifact_schema_errors(unlocks) == []
    assert mod.artifact_schema_errors(planner_gap) == []
    assert mod.artifact_schema_errors(invariant) == []


def test_req_arc_wmte_4892_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4892: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, decision_need=0.5),
            "cn04": _row("cn04", baseline=0.0, decision_need=0.5),
            "ls20": _row("ls20", baseline=0.0, decision_need=0.5),
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
            "per_game_value_gap": {"cd82": {"planned_bucket": "MAYBE"}},
            "positive_control_non_degenerate": False,
            "delta_on_truly_heldout_split": True,
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
    assert "per_game_value_gap.cd82.planned_bucket" in errors
    assert "positive_control_non_degenerate" in errors
    assert "delta_on_truly_heldout_split" in errors
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


def test_req_arc_wmte_4892_run_blocks_and_checkpoints_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT: blocked and partial runs stay schema-valid."""

    common = {
        "root": tmp_path,
        "a1_baseline_loader": lambda _root: _a1_artifact(),
        "ground_truth_loader": lambda _root: _ground_truth(),
        "environment_games_loader": lambda _arcade: set(_ground_truth()),
        "game_measurer": lambda game, **_kwargs: _row(game, baseline=0.0, decision_need=0.5),
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
    blocked_baseline = mod.run(
        **{**common, "now": iter([3.0, 3.1]).__next__, "a1_baseline_loader": lambda _root: None},
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_baseline_loader=lambda _root: _a1_artifact(),
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(game, baseline=0.0, decision_need=0.5),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([4.0, 4.1, 5.0, 5.1]).__next__,
        write=True,
        soft_elapsed_budget_s=0.05,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_generator["honest_verdict"] == "blocked_generator_unavailable"
    assert blocked_baseline["honest_verdict"] == "blocked_a1_baseline_missing"
    assert partial["partial"] is True
    assert partial["checkpoint_emitted"] is True
    assert (tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json").exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == partial
    assert mod.artifact_schema_errors(partial) == []


def test_req_arc_wmte_4892_run_full_resume_and_degenerate_control(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4892: run aggregates rows, resumes checkpoints, and retires bad controls."""

    rows = {
        "cd82": _row("cd82", baseline=0.0, decision_need=0.5),
        "cn04": _row("cn04", baseline=0.0, decision_need=0.5),
        "ls20": _row("ls20", baseline=0.0, decision_need=0.5),
    }
    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_baseline_loader=lambda _root: _a1_artifact(),
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
        a1_baseline_loader=lambda _root: _a1_artifact(),
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
        per_game_value_gap=rows,
        positive_control_game="tu93",
        positive_control_row=_control(recall=0.0),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=60.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert artifact["fork_verdict"] == "PLANNER_GAP"
    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True
    assert retired["honest_verdict"] == "complete_decision_need_positive_control_degenerate_retired"
    assert retired["fork_verdict"] is None


def test_req_arc_wmte_4892_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4892: final artifact is the requested decision-need deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["positive_control_non_degenerate"] is True
    assert artifact["delta_on_truly_heldout_split"] is True
    assert artifact["planner_blind_to_banked_answer"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["generator_backend"] in {"gpu0_cuda", "igpu_hip"}
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
