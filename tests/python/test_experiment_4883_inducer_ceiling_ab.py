"""Tests for Exp 4883 inducer-ceiling reference/local A/B.

Spec refs: REQ-ARC-WMTE-4883,
SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE,
SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB,
SCENARIO-ARC-WMTE-4883-ATTRIBUTION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4883_inducer_ceiling_ab as mod
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _transition() -> Transition:
    grid = np.zeros((2, 2), dtype=int)
    next_grid = np.array([[1, 0], [0, 0]], dtype=int)
    return Transition(grid, 1, None, next_grid, 0, 0)


def _generator_ok() -> dict[str, Any]:
    return {
        "ok": True,
        "generator_backend": "gpu0_cuda",
        "backend": "gpu0_cuda",
        "model": "Qwen3.5-9B-MTP",
        "server": "/fake/llama-server",
        "port": 8931,
        "launch_env_cuda_visible_devices": "0",
    }


def _a1_row(game: str, *, baseline: float = 0.25) -> dict[str, Any]:
    return {
        "game": game,
        "value_acc_baseline": baseline,
        "cell_recall_baseline": 0.5,
        "value_acc_adapted": baseline,
        "value_delta": 0.0,
        "fit_transition_ids": ["fit:0"],
        "baseline_transition_ids": ["baseline:0"],
        "remeasure_transition_ids": ["heldout:0"],
        "adapter_fit_transition_count": 1,
        "heldout_transition_count": 1,
        "cold_transition_count": 1,
    }


def _a1_artifact(*, delta: float = -0.01) -> dict[str, Any]:
    return {
        "experiment_id": 4882,
        "tta_changed_cell_value_accuracy_delta_median": delta,
        "per_game_value_gap": {
            "cd82": _a1_row("cd82"),
            "cn04": _a1_row("cn04"),
            "ls20": _a1_row("ls20"),
        },
        "tta_config": {
            "heldout_transitions": 1,
            "cold_transitions": 1,
            "heldout_games": ["cd82", "cn04", "ls20"],
        },
        "random_seed": 20260627,
    }


def _lane_row(game: str, *, lane: str, baseline: float, value: float) -> dict[str, Any]:
    delta = round(value - baseline, 6)
    return {
        "game": game,
        "lane": lane,
        "value_acc": round(value, 6),
        "cell_recall": 0.5,
        "delta_vs_baseline": delta,
        "ci95": [delta, delta],
        "a1_baseline_value_acc": round(baseline, 6),
        "a1_heldout_transition_ids": ["heldout:0"],
        "heldout_transition_ids": ["heldout:0"],
        "fit_transition_ids": ["fit:0"],
        "heldout_transition_count": 1,
        "live_path_methods_called": ["arc_executable_world_model.load_engine"],
        "residual": "ok",
    }


def _per_lane_rows(
    *,
    reference_values: tuple[float, float, float],
    local_values: tuple[float, float, float],
    baseline: float = 0.25,
) -> dict[str, dict[str, dict[str, Any]]]:
    games = ("cd82", "cn04", "ls20")
    return {
        "reference": {
            game: _lane_row(game, lane="reference", baseline=baseline, value=value)
            for game, value in zip(games, reference_values, strict=True)
        },
        "local": {
            game: _lane_row(game, lane="local", baseline=baseline, value=value)
            for game, value in zip(games, local_values, strict=True)
        },
    }


def test_req_arc_wmte_4883_spec_declares_inducer_ab_contract() -> None:
    """REQ-ARC-WMTE-4883: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4883",
        "SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE",
        "SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB",
        "SCENARIO-ARC-WMTE-4883-ATTRIBUTION",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4883_a1_low_value_gate_requires_split_and_delta() -> None:
    """SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE: missing/high A1 inputs block."""

    assert mod._a1_precondition(None)["ok"] is False
    assert mod._a1_precondition(_a1_artifact(delta=0.1))["ok"] is False
    malformed = _a1_artifact()
    malformed["per_game_value_gap"]["cd82"].pop("remeasure_transition_ids")

    bad = mod._a1_precondition(malformed)
    ok = mod._a1_precondition(_a1_artifact())

    assert bad["detail"] == "missing_per_game_baselines_or_split"
    assert ok["ok"] is True
    assert ok["tta_changed_cell_value_accuracy_delta_median"] == -0.01
    assert ok["n_games"] == 3


def test_scenario_arc_wmte_4883_same_split_ab_scores_loaded_lane_engines() -> None:
    """SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB: both lanes call load_engine and score same ids."""

    active: dict[str, str] = {}
    load_calls: list[str] = []

    def reference_engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[0, 0] = 1
        return out

    def local_engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
        out = np.asarray(grid).copy()
        out[0, 0] = 9
        return out

    def _inducer(lane: str):
        def _run(**kwargs: Any) -> dict[str, Any]:
            active[str(kwargs["game"])] = lane
            return {"ok": True, "note": lane}

        return _run

    def _load_engine(game: str) -> tuple[Any, None]:
        load_calls.append(game)
        return (reference_engine if active[game] == "reference" else local_engine), None

    row = mod.measure_game_with_inducer_lanes(
        game="cd82",
        a1_row={**_a1_row("cd82"), "value_acc_baseline": 0.0},
        a1_config={"heldout_transitions": 1, "cold_transitions": 1},
        lane_inducers={
            "reference": _inducer("reference"),
            "local": _inducer("local"),
        },
        transition_collector=lambda **_kwargs: ([ _transition() ], 1),
        engine_loader=_load_engine,
        random_seed=1,
    )

    assert load_calls == ["cd82", "cd82"]
    assert set(row) == {"reference", "local"}
    assert row["reference"]["value_acc"] == 1.0
    assert row["reference"]["delta_vs_baseline"] == 1.0
    assert row["local"]["value_acc"] == 0.0
    assert row["reference"]["heldout_transition_ids"] == ["heldout:0"]
    assert row["local"]["heldout_transition_ids"] == ["heldout:0"]


def test_scenario_arc_wmte_4883_attribution_uses_lane_delta_ci() -> None:
    """SCENARIO-ARC-WMTE-4883-ATTRIBUTION: lane lifts name the ceiling cause."""

    local_ceiling = mod.build_artifact(
        per_lane_per_game=_per_lane_rows(
            reference_values=(0.5, 0.5, 0.5),
            local_values=(0.25, 0.25, 0.25),
        ),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        a1_artifact=_a1_artifact(),
        bootstrap_iterations=25,
    )
    method_ceiling = mod.build_artifact(
        per_lane_per_game=_per_lane_rows(
            reference_values=(0.25, 0.25, 0.25),
            local_values=(0.25, 0.25, 0.25),
        ),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        a1_artifact=_a1_artifact(),
        bootstrap_iterations=25,
    )
    local_sufficient = mod.build_artifact(
        per_lane_per_game=_per_lane_rows(
            reference_values=(0.5, 0.5, 0.5),
            local_values=(0.5, 0.5, 0.5),
        ),
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        a1_artifact=_a1_artifact(),
        bootstrap_iterations=25,
    )

    assert local_ceiling["inducer_ceiling_attribution"] == "LOCAL_MODEL_IS_CEILING"
    assert local_ceiling["honest_verdict"] == "success_inducer_ceiling_LOCAL_MODEL_IS_CEILING"
    assert method_ceiling["inducer_ceiling_attribution"] == "METHOD_IS_CEILING"
    assert method_ceiling["honest_verdict"] == (
        "complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling"
    )
    assert local_sufficient["inducer_ceiling_attribution"] == "LOCAL_ALREADY_SUFFICIENT"
    assert mod.artifact_schema_errors(local_ceiling) == []
    assert mod.artifact_schema_errors(method_ceiling) == []
    assert mod.artifact_schema_errors(local_sufficient) == []


def test_req_arc_wmte_4883_run_writes_lane_checkpoints_and_schema_valid_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-ARC-WMTE-4883: run aggregates rows, checkpoints, and writes the result JSON."""

    proposer = object()
    monkeypatch.setattr(mod.a1, "make_live_qwen_proposer", lambda: proposer)
    monkeypatch.setattr(
        mod.a1,
        "generator_available",
        lambda *, proposer: {**_generator_ok(), "proposer_id": id(proposer)},
    )

    def _measure(game: str, **_kwargs: Any) -> dict[str, dict[str, Any]]:
        return {
            "reference": _lane_row(game, lane="reference", baseline=0.25, value=0.5),
            "local": _lane_row(game, lane="local", baseline=0.25, value=0.25),
        }

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=None,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        lane_measurer=_measure,
        now=iter([10.0, 10.1, 10.2, 10.3, 75.0]).__next__,
        write=True,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82__reference.json"

    assert loaded == artifact
    assert checkpoint.exists()
    assert artifact["inducer_ceiling_attribution"] == "LOCAL_MODEL_IS_CEILING"
    assert artifact["preconditions_checked"]["generator"]["proposer_id"] == id(proposer)
    assert artifact["n_games_measured"] == 3
    assert artifact["live_path_reachable"] is True
    assert mod.artifact_schema_errors(artifact) == []

    resumed = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        lane_measurer=lambda *_args, **_kwargs: pytest.fail("checkpoint should be reused"),
        now=iter([80.0, 80.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )
    assert resumed["n_games_measured"] == 3


def test_req_arc_wmte_4883_run_blocked_and_partial_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4883: blocked and partial runs stay schema-valid."""

    offline = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: False,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        now=iter([1.0, 1.1]).__next__,
        write=True,
    )
    generator = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=lambda: {"ok": False, "detail": "missing_qwen"},
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        now=iter([2.0, 2.1]).__next__,
        write=False,
    )
    a1_blocked = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(delta=0.2),
        live_path_checker=lambda _root: True,
        now=iter([3.0, 3.1]).__next__,
        write=False,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        lane_measurer=lambda game, **_kwargs: {
            "reference": _lane_row(game, lane="reference", baseline=0.25, value=0.5),
            "local": _lane_row(game, lane="local", baseline=0.25, value=0.25),
        },
        now=iter([4.0, 4.1, 5.0, 5.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        soft_elapsed_budget_s=0.05,
    )

    assert offline["honest_verdict"] == "blocked_offline_arcade_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert generator["honest_verdict"] == "blocked_generator_unavailable"
    assert a1_blocked["honest_verdict"] == "blocked_a1_baseline_missing"
    assert partial["partial"] is True
    assert partial["honest_verdict"] == "complete_inducer_ceiling_partial_budget_stop"


def test_req_arc_wmte_4883_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4883: final artifact is the requested inducer-ceiling deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["delta_on_truly_heldout_split"] is True
    assert artifact["reference_lane_is_ceiling_only"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert "reference_lane" in artifact["model_specs"]
    assert "local_lane" in artifact["model_specs"]
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
