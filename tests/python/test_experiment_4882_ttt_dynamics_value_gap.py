"""Tests for Exp 4882 TTT dynamics value-gap fork probe.

Spec refs: REQ-ARC-WMTE-4882,
SCENARIO-ARC-WMTE-4882-GRADED-METRIC,
SCENARIO-ARC-WMTE-4882-DISJOINT-TTA-DELTA,
SCENARIO-ARC-WMTE-4882-PARTIAL-CHECKPOINT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4882_ttt_dynamics_value_gap as mod
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
    adapted: float,
    recall: float = 0.75,
    bucket: str = "NEVER_ENUMERATED",
    fit_ids: list[int] | None = None,
    heldout_ids: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "game": game,
        "cell_recall_baseline": recall,
        "cell_recall_adapted": recall,
        "value_acc_baseline": baseline,
        "value_acc_adapted": adapted,
        "value_delta": round(adapted - baseline, 6),
        "planned_bucket": bucket,
        "migrated": bucket == "COVERED" and game != "tu93",
        "fit_transition_ids": fit_ids if fit_ids is not None else [0, 1],
        "remeasure_transition_ids": heldout_ids if heldout_ids is not None else [2, 3],
        "baseline_transition_ids": [4, 5],
        "adapter_fit_transition_count": 2,
        "heldout_transition_count": 2,
        "cold_transition_count": 4,
        "plan_length": 1 if bucket != "NEVER_ENUMERATED" else 0,
        "live_path_methods_called": [
            "arc_live_ttt.CNNDynamics",
            "DynamicsValueAdapter",
            "arc_executable_world_model.plan_in_model",
        ],
    }


def _control(recall: float = 0.8) -> dict[str, Any]:
    return _row("tu93", baseline=0.2, adapted=0.4, recall=recall, bucket="COVERED")


def test_req_arc_wmte_4882_spec_declares_value_gap_contract() -> None:
    """REQ-ARC-WMTE-4882: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4882",
        "SCENARIO-ARC-WMTE-4882-GRADED-METRIC",
        "SCENARIO-ARC-WMTE-4882-DISJOINT-TTA-DELTA",
        "SCENARIO-ARC-WMTE-4882-PARTIAL-CHECKPOINT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4882_graded_metric_splits_location_from_value() -> None:
    """SCENARIO-ARC-WMTE-4882-GRADED-METRIC: score location recall separately from value accuracy."""

    g0 = np.zeros((2, 2), dtype=int)
    n0 = np.array([[1, 2], [0, 0]], dtype=int)
    g1 = np.zeros((2, 2), dtype=int)
    n1 = np.array([[0, 0], [3, 0]], dtype=int)
    transitions = [_transition(g0, n0), _transition(g1, n1)]

    def engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
        out = np.asarray(grid).copy()
        if out[0, 0] == 0 and out[0, 1] == 0:
            out[0, 0] = 1
            out[0, 1] = 9
        return out

    score = mod.score_graded_engine(engine, transitions)

    assert score["cell_recall"] == pytest.approx(2.0 / 3.0)
    assert score["changed_cell_value_accuracy"] == pytest.approx(0.5)
    assert score["actual_changed_cells"] == 3
    assert score["overlap_changed_cells"] == 2
    assert score["correct_changed_values"] == 1


def test_scenario_arc_wmte_4882_value_adapter_improves_values_on_disjoint_rows() -> None:
    """SCENARIO-ARC-WMTE-4882-DISJOINT-TTA-DELTA: adapter fits own transitions, scores held-out rows."""

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

    def wrong_value_engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[0, 0] = 5
        return out

    adapter = mod.DynamicsValueAdapter.fit(fit)
    baseline = mod.score_graded_engine(wrong_value_engine, heldout)
    adapted = mod.score_graded_engine(adapter.wrap(wrong_value_engine), heldout)

    assert baseline["cell_recall"] == 1.0
    assert baseline["changed_cell_value_accuracy"] == 0.0
    assert adapted["changed_cell_value_accuracy"] == 1.0


def test_scenario_arc_wmte_4882_fork_verdict_uses_value_delta_ci_and_migration() -> None:
    """SCENARIO-ARC-WMTE-4882-DISJOINT-TTA-DELTA: value-delta CI and migration name the fork."""

    beatable = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, adapted=0.5, bucket="COVERED"),
            "cn04": _row("cn04", baseline=0.0, adapted=0.5),
            "ls20": _row("ls20", baseline=0.0, adapted=0.5),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )
    planner_gap = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, adapted=0.5),
            "cn04": _row("cn04", baseline=0.0, adapted=0.5),
            "ls20": _row("ls20", baseline=0.0, adapted=0.5),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )
    hard = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.2, adapted=0.2),
            "cn04": _row("cn04", baseline=0.2, adapted=0.1),
            "ls20": _row("ls20", baseline=0.2, adapted=0.3),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert beatable["fork_verdict"] == "INDUCER_CEILING_BEATABLE"
    assert beatable["honest_verdict"] == "success_ttt_dynamics_value_gap_closed_0.500000"
    assert beatable["tta_changed_cell_value_accuracy_delta_median"] == 0.5
    assert beatable["tta_value_accuracy_delta_ci95"] == [0.5, 0.5]
    assert planner_gap["fork_verdict"] == "PLANNER_GAP"
    assert hard["fork_verdict"] == "INDUCER_CEILING_HARD"
    assert mod.artifact_schema_errors(beatable) == []
    assert mod.artifact_schema_errors(planner_gap) == []
    assert mod.artifact_schema_errors(hard) == []


def test_req_arc_wmte_4882_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4882: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, adapted=0.5),
            "cn04": _row("cn04", baseline=0.0, adapted=0.5),
            "ls20": _row("ls20", baseline=0.0, adapted=0.5),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
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


def test_req_arc_wmte_4882_run_blocks_and_checkpoints_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4882-PARTIAL-CHECKPOINT: blocked and partial runs stay schema-valid."""

    common = {
        "root": tmp_path,
        "ground_truth_loader": lambda _root: _ground_truth(),
        "environment_games_loader": lambda _arcade: set(_ground_truth()),
        "game_measurer": lambda game, **_kwargs: _row(game, baseline=0.0, adapted=0.5),
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
    blocked_games = mod.run(
        **{
            **common,
            "now": iter([3.0, 3.1]).__next__,
            "environment_games_loader": lambda _arcade: {"cd82", "tu93"},
        },
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        ground_truth_loader=lambda _root: _ground_truth(),
        environment_games_loader=lambda _arcade: set(_ground_truth()),
        game_measurer=lambda game, **_kwargs: _row(game, baseline=0.0, adapted=0.5),
        positive_control_runner=lambda **_kwargs: _control(),
        live_path_checker=lambda _root: True,
        now=iter([4.0, 4.1, 5.0, 5.1]).__next__,
        write=True,
        soft_elapsed_budget_s=0.05,
        heldout_games=("cd82", "cn04", "ls20"),
    )

    assert blocked_arcade["honest_verdict"] == "blocked_offline_arcade_missing"
    assert blocked_generator["honest_verdict"] == "blocked_generator_unavailable"
    assert blocked_games["honest_verdict"] == "blocked_no_heldout_games"
    assert partial["partial"] is True
    assert partial["checkpoint_emitted"] is True
    assert (tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json").exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == partial
    assert mod.artifact_schema_errors(partial) == []


def test_req_arc_wmte_4882_run_full_resume_and_degenerate_control(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-ARC-WMTE-4882: run aggregates rows, resumes checkpoints, and retires bad controls."""

    proposer = object()
    monkeypatch.setattr(mod.a1, "make_live_qwen_proposer", lambda: proposer)
    monkeypatch.setattr(
        mod.a1,
        "generator_available",
        lambda *, proposer: {**_generator_ok(), "proposer_id": id(proposer)},
    )
    rows = {
        "cd82": _row("cd82", baseline=0.0, adapted=0.5),
        "cn04": _row("cn04", baseline=0.0, adapted=0.5),
        "ls20": _row("ls20", baseline=0.0, adapted=0.5),
    }
    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=None,
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
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert artifact["fork_verdict"] == "PLANNER_GAP"
    assert artifact["preconditions_checked"]["generator"]["proposer_id"] == id(proposer)
    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True
    assert retired["honest_verdict"] == "complete_ttt_dynamics_positive_control_degenerate_retired"
    assert retired["fork_verdict"] is None


def test_req_arc_wmte_4882_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4882: final artifact is the requested value-gap deliverable."""

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
