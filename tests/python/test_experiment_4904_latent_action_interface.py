"""Tests for Exp 4904 latent-action interface value-gap fork probe.

Spec refs: REQ-ARC-WMTE-4904,
SCENARIO-ARC-WMTE-4904-A1-LOW-FIRST-WIN-GATE,
SCENARIO-ARC-WMTE-4904-LATENT-ACTION-INTERFACE,
SCENARIO-ARC-WMTE-4904-GENUINELY-LIVE,
SCENARIO-ARC-WMTE-4904-FORK-VERDICT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4904_latent_action_interface as mod
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
    latent_action: float,
    recall: float = 0.75,
    fit_ids: list[Any] | None = None,
    heldout_ids: list[Any] | None = None,
) -> dict[str, Any]:
    delta = round(latent_action - baseline, 6)
    return {
        "game": game,
        "cell_recall": recall,
        "value_acc_code_baseline": baseline,
        "value_acc_latent_action": latent_action,
        "delta": delta,
        "ci95": [delta, delta],
        "fit_transition_ids": fit_ids if fit_ids is not None else ["fit:0", "fit:1"],
        "heldout_transition_ids": heldout_ids
        if heldout_ids is not None
        else ["heldout:0", "heldout:1"],
        "baseline_transition_ids": ["heldout:0", "heldout:1"],
        "latent_token_count": 2,
        "accepted_token_count": 2,
        "action_embedding_count": 2,
        "fit_transition_count": 2,
        "heldout_transition_count": 2,
        "cold_transition_count": 4,
        "latent_action_summary": {"representation_type": "latent_action_interface"},
        "live_path_methods_called": [
            "LatentActionInterface",
            "arc_executable_world_model.load_engine",
        ],
    }


def _control(recall: float = 0.8) -> dict[str, Any]:
    return _row("tu93", baseline=0.2, latent_action=0.4, recall=recall)


def _a1_artifact(delta: float = -0.04) -> dict[str, Any]:
    return {
        "experiment_id": 4903,
        "fork_verdict": "WALL_DEEPER_THAN_VALUE_PREDICTION",
        "value_grounded_first_win_delta_median": delta,
        "value_grounded_first_win_delta_ci95": [delta, delta],
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "preconditions_checked": {
            "a1_baseline": {
                "ok": True,
                "path": "results/experiment_4892_decision_need_targets_value_gap.json",
            }
        },
    }


def _baseline_artifact() -> dict[str, Any]:
    rows = {
        game: {
            "game": game,
            "cell_recall": 0.7,
            "value_acc_code_baseline": 0.1,
            "value_acc_decision_need": 0.1,
            "value_delta": 0.0,
            "baseline_transition_ids": ["heldout:0", "heldout:1"],
            "heldout_transition_ids": ["heldout:0", "heldout:1"],
            "author_transition_ids": ["author:0"],
            "planned_bucket": "NEVER_ENUMERATED",
        }
        for game in ("cd82", "cn04", "ls20", "m0r0")
    }
    return {
        "experiment_id": 4892,
        "fork_verdict": "VALUE_GAP_REPRESENTATION_INVARIANT",
        "positive_control_game": "tu93",
        "positive_control_non_degenerate": True,
        "decision_need_value_accuracy_delta_median": -0.05,
        "per_game_value_gap": rows,
        "positive_control_value_gap": {
            "game": "tu93",
            "cell_recall": 0.8,
            "value_acc_code_baseline": 0.2,
            "value_acc_decision_need": 0.2,
            "heldout_transition_ids": ["heldout:0", "heldout:1"],
            "baseline_transition_ids": ["heldout:0", "heldout:1"],
        },
    }


def test_req_arc_wmte_4904_spec_declares_latent_action_contract() -> None:
    """REQ-ARC-WMTE-4904: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4904",
        "SCENARIO-ARC-WMTE-4904-A1-LOW-FIRST-WIN-GATE",
        "SCENARIO-ARC-WMTE-4904-LATENT-ACTION-INTERFACE",
        "SCENARIO-ARC-WMTE-4904-GENUINELY-LIVE",
        "SCENARIO-ARC-WMTE-4904-FORK-VERDICT",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4904_latent_action_tokens_score_values() -> None:
    """SCENARIO-ARC-WMTE-4904-LATENT-ACTION-INTERFACE: token alignment predicts values."""

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

    interface = mod.LatentActionInterface.induce(
        fit,
        game="toy",
        llm_tokens=["paint-click"],
    )
    score = mod.score_latent_action_interface(interface, heldout)

    assert interface.representation_type == "latent_action_interface"
    assert interface.summary()["accepted_token_count"] == 1
    assert interface.summary()["action_embedding_count"] == 1
    assert score["cell_recall"] == 1.0
    assert score["changed_cell_value_accuracy"] == 1.0


def test_scenario_arc_wmte_4904_fork_verdict_uses_value_delta_ci() -> None:
    """SCENARIO-ARC-WMTE-4904-FORK-VERDICT: CI excludes zero only for a real lift."""

    matters = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, latent_action=0.5),
            "cn04": _row("cn04", baseline=0.0, latent_action=0.5),
            "ls20": _row("ls20", baseline=0.0, latent_action=0.5),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        live_llm_invocations=3,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )
    invariant = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.2, latent_action=0.2),
            "cn04": _row("cn04", baseline=0.2, latent_action=0.1),
            "ls20": _row("ls20", baseline=0.2, latent_action=0.3),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        live_llm_invocations=3,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )

    assert matters["fork_verdict"] == "REPRESENTATION_MATTERS"
    assert matters["honest_verdict"] == "success_latent_action_value_gap_closed_0.500000"
    assert matters["latent_action_value_accuracy_delta_median"] == 0.5
    assert matters["latent_action_value_accuracy_delta_ci95"] == [0.5, 0.5]
    assert invariant["fork_verdict"] == "VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES"
    assert invariant["retire_if_same_verdict"] is True
    assert mod.artifact_schema_errors(matters) == []
    assert mod.artifact_schema_errors(invariant) == []


def test_req_arc_wmte_4904_schema_errors_are_explicit() -> None:
    """REQ-ARC-WMTE-4904: malformed artifacts fail closed with named errors."""

    artifact = mod.build_artifact(
        per_game_value_gap={
            "cd82": _row("cd82", baseline=0.0, latent_action=0.5),
            "cn04": _row("cn04", baseline=0.0, latent_action=0.5),
            "ls20": _row("ls20", baseline=0.0, latent_action=0.5),
        },
        positive_control_game="tu93",
        positive_control_row=_control(),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=75.0,
        live_llm_invocations=3,
        partial=False,
        checkpoint_emitted=True,
        bootstrap_iterations=25,
    )
    malformed = dict(artifact)
    malformed.update(
        {
            "honest_verdict": "not_terminal",
            "fork_verdict": "MAYBE",
            "per_game_value_gap": {"cd82": {"ci95": []}},
            "ran_genuinely_live": False,
            "delta_on_truly_heldout_split": True,
            "verifier_is_oracle": True,
            "live_path_reachable": False,
            "generator_backend": "cpu",
            "solve_provenance": "live_agent_self_discovery",
            "checkpoint_emitted": "yes",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "model_specs": [],
            "preconditions_checked": [],
            "reproducibility_checksum": "sha256:bad",
        }
    )

    errors = mod.artifact_schema_errors(malformed)

    assert "honest_verdict_terminal_prefix" in errors
    assert "fork_verdict" in errors
    assert "per_game_value_gap.cd82.ci95" in errors
    assert "ran_genuinely_live" in errors
    assert "delta_on_truly_heldout_split" in errors
    assert "verifier_is_oracle" in errors
    assert "live_path_reachable" in errors
    assert "generator_backend" in errors
    assert "solve_provenance" in errors
    assert "checkpoint_emitted" in errors
    assert "inference_substrate" in errors
    assert "model_specs" in errors
    assert "preconditions_checked" in errors
    assert "reproducibility_checksum" in errors
    assert mod.artifact_schema_errors({})[0].startswith("missing_field:")


def test_req_arc_wmte_4904_run_blocks_and_checkpoints_partial(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4904-A1-LOW-FIRST-WIN-GATE: gated runs stay valid."""

    common = {
        "root": tmp_path,
        "a1_artifact_loader": lambda _root: _a1_artifact(),
        "baseline_loader": lambda _root: _baseline_artifact(),
        "ground_truth_loader": lambda _root: _ground_truth(),
        "environment_games_loader": lambda _arcade: set(_ground_truth()),
        "game_measurer": lambda game, **_kwargs: _row(game, baseline=0.0, latent_action=0.5),
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
    skipped_gate = mod.run(
        **{
            **common,
            "now": iter([3.2, 3.3]).__next__,
            "a1_artifact_loader": lambda _root: _a1_artifact(delta=0.2),
        },
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
    )
    blocked_baseline = mod.run(
        **{**common, "now": iter([3.4, 3.5]).__next__, "baseline_loader": lambda _root: None},
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
        game_measurer=lambda game, **_kwargs: _row(game, baseline=0.0, latent_action=0.5),
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
    assert skipped_gate["honest_verdict"] == "skipped_a1_first_win_unlocked"
    assert blocked_baseline["honest_verdict"] == "blocked_a1_baseline_missing"
    assert partial["partial"] is True
    assert partial["checkpoint_emitted"] is True
    assert (tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json").exists()
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == partial
    assert mod.artifact_schema_errors(partial) == []


def test_req_arc_wmte_4904_run_full_resume_and_live_floor(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4904-GENUINELY-LIVE: full runs aggregate and resume checkpoints."""

    rows = {
        "cd82": _row("cd82", baseline=0.0, latent_action=0.5),
        "cn04": _row("cn04", baseline=0.0, latent_action=0.5),
        "ls20": _row("ls20", baseline=0.0, latent_action=0.5),
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
        now=iter([80.0, 80.1, 145.0]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )

    assert artifact["fork_verdict"] == "REPRESENTATION_MATTERS"
    assert artifact["ran_genuinely_live"] is True
    assert artifact["duration_s"] > mod.LIVE_DURATION_FLOOR_S
    assert artifact["preconditions_checked"]["a1_artifact"]["low_first_win_delta"] is True
    assert artifact["preconditions_checked"]["baseline"]["ok"] is True
    assert resumed["n_games_measured"] == 3
    assert resumed["checkpoint_emitted"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert mod.artifact_schema_errors(resumed) == []


def test_req_arc_wmte_4904_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4904: final artifact is the requested latent-action deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["positive_control_game"] == "tu93"
    assert artifact["delta_on_truly_heldout_split"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["generator_backend"] in {"gpu0_cuda", "igpu_hip"}
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["live_path_reachable"] is True
    assert artifact["ran_genuinely_live"] is True
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
