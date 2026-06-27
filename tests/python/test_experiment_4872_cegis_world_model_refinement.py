"""Tests for Exp 4872 CEGIS executable world-model refinement.

Spec refs: REQ-ARC-WMTE-4872,
SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE,
SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE,
SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4872_cegis_world_model_refinement as mod
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _transition(value: int = 0, next_value: int = 1, *, action: int = 1) -> Transition:
    grid = np.array([[value, 0], [0, 0]], dtype=int)
    next_grid = np.array([[next_value, 0], [0, 0]], dtype=int)
    return Transition(grid, action, None, next_grid, 0, 0)


def _identity_engine(grid: np.ndarray, _action: int, _data: Any = None) -> np.ndarray:
    return np.asarray(grid).copy()


def _correct_engine(grid: np.ndarray, action: int, _data: Any = None) -> np.ndarray:
    out = np.asarray(grid).copy()
    if int(action) == 1:
        out[0, 0] = 1
    return out


def _regressing_engine(grid: np.ndarray, action: int, _data: Any = None) -> np.ndarray:
    out = _correct_engine(grid, action, _data)
    if int(action) == 5:
        out[1, 1] = 9
    return out


def _a1_artifact(median: float = 0.0) -> dict[str, Any]:
    return {
        "experiment_id": 4871,
        "median_engine_heldout_accuracy": median,
        "per_game_fork": {
            "cd82": {"engine_heldout_accuracy": 0.1},
            "cn04": {"engine_heldout_accuracy": 0.2},
            "ls20": {"engine_heldout_accuracy": 0.3},
        },
        "model_specs": {"name": "Qwen3.5-9B-MTP", "backend": "gpu0_cuda"},
    }


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


def _row(game: str, *, baseline: float, refined: float, fixed: int = 1) -> dict[str, Any]:
    return {
        "game": game,
        "baseline": baseline,
        "refined": refined,
        "delta": round(refined - baseline, 6),
        "a1_artifact_baseline": baseline,
        "counterexamples_fixed": fixed,
        "cegis_rounds": 1,
        "accepted_repairs": 1 if fixed else 0,
        "repair_transition_ids": [0],
        "remeasure_transition_ids": [1, 2],
        "observed_prefix_accuracy_before": 1.0,
        "observed_prefix_accuracy_after": 1.0,
        "repair_counterexample_count": 1,
        "remeasure_transition_count": 2,
        "rounds": [{"round": 1, "accepted": bool(fixed), "fixed_count": fixed}],
        "residual": "accepted_repair" if fixed else "no_accepted_repair",
    }


def test_req_arc_wmte_4872_spec_declares_cegis_contract() -> None:
    """REQ-ARC-WMTE-4872: OpenSpec anchors fields, scenarios, and result path."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-ARC-WMTE-4872",
        "SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE",
        "SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE",
        "SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4872_a1_low_accuracy_gate_blocks_before_repair(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE: high/missing A1 skips CEGIS."""

    calls: list[str] = []
    common = {
        "root": tmp_path,
        "offline_arcade_checker": lambda: True,
        "generator_checker": _generator_ok,
        "live_path_checker": lambda _root: True,
        "game_refiner": lambda game, **_kwargs: calls.append(game) or _row(game, baseline=0.0, refined=1.0),
        "now": iter([1.0, 1.1]).__next__,
        "write": False,
    }

    missing = mod.run(**common, a1_artifact_loader=lambda _root: None)
    high = mod.run(
        **{**common, "now": iter([2.0, 2.1]).__next__},
        a1_artifact_loader=lambda _root: _a1_artifact(median=0.5),
    )

    assert calls == []
    assert missing["honest_verdict"] == "blocked_a1_baseline_missing"
    assert high["honest_verdict"] == "blocked_a1_not_inducer_ceiling"
    assert missing["preconditions_checked"]["a1_baseline"]["ok"] is False
    assert high["preconditions_checked"]["a1_baseline"]["median_engine_heldout_accuracy"] == 0.5


def test_scenario_arc_wmte_4872_repair_acceptance_requires_fix_without_regression() -> None:
    """SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE: repairs must fix CEs and preserve replay."""

    repair = [_transition()]
    observed = [_transition(next_value=0, action=5)]

    accepted = mod.evaluate_repair_acceptance(
        previous_engine=_identity_engine,
        repaired_engine=_correct_engine,
        repair_counterexamples=repair,
        observed_prefix=observed,
    )
    no_fix = mod.evaluate_repair_acceptance(
        previous_engine=_identity_engine,
        repaired_engine=_identity_engine,
        repair_counterexamples=repair,
        observed_prefix=observed,
    )
    regression = mod.evaluate_repair_acceptance(
        previous_engine=_identity_engine,
        repaired_engine=_regressing_engine,
        repair_counterexamples=repair,
        observed_prefix=observed,
    )

    assert accepted["accepted"] is True
    assert accepted["fixed_count"] == 1
    assert accepted["observed_regressed"] is False
    assert no_fix["accepted"] is False
    assert no_fix["fixed_count"] == 0
    assert regression["accepted"] is False
    assert regression["fixed_count"] == 1
    assert regression["observed_regressed"] is True


def test_req_arc_wmte_4872_helpers_load_split_and_validate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4872: helper branches preserve deterministic gates and splits."""

    assert mod.load_a1_artifact(tmp_path) is None
    a1_path = tmp_path / mod.A1_RESULT_RELATIVE_PATH
    a1_path.parent.mkdir(parents=True)
    a1_path.write_text("{", encoding="utf-8")
    assert mod.load_a1_artifact(tmp_path) is None
    a1_path.write_text(json.dumps(_a1_artifact()), encoding="utf-8")
    assert mod.load_a1_artifact(tmp_path)["experiment_id"] == 4871

    transitions = [_transition(), _transition(), _transition(next_value=0, action=5)]
    split = mod.select_repair_and_remeasure_splits(
        engine=_identity_engine,
        heldout_transitions=transitions,
        max_repair_counterexamples=1,
        seed=1,
    )
    assert len(split["repair_indices"]) == 1
    assert set(split["repair_indices"]).isdisjoint(split["remeasure_indices"])

    assert mod._delta_values({"bad": [], "missing": {}, "ok": {"delta": "0.25"}}) == [0.25]
    assert mod.bootstrap_ci95([], iterations=10, seed=1) == [None, None]
    varied = mod.bootstrap_ci95([0.0, 0.5, 1.0], iterations=10, seed=1)
    assert varied[0] <= varied[1]

    assert mod._a1_precondition({"median_engine_heldout_accuracy": "bad"})["detail"] == (
        "missing_median_engine_heldout_accuracy"
    )
    assert mod._a1_precondition(
        {"median_engine_heldout_accuracy": 0.0, "per_game_fork": {}}
    )["detail"] == "missing_per_game_baselines"

    with pytest.raises(mod.DiagnosticError):
        mod._validate_or_raise({"bad": True})

    assert mod._load_checkpoint("missing", root=tmp_path) is None
    bad_checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "bad.json"
    bad_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    bad_checkpoint.write_text("{", encoding="utf-8")
    assert mod._load_checkpoint("bad", root=tmp_path) is None
    mod._write_checkpoint("ok", {"delta": 0.0}, root=tmp_path)
    assert mod._load_checkpoint("ok", root=tmp_path)["delta"] == 0.0


def test_scenario_arc_wmte_4872_truly_heldout_delta_drives_verdict() -> None:
    """SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA: disjoint CI gates success/null."""

    rows = {
        "cd82": _row("cd82", baseline=0.0, refined=0.5),
        "cn04": _row("cn04", baseline=0.0, refined=0.5),
        "ls20": _row("ls20", baseline=0.0, refined=0.5),
    }
    artifact = mod.build_artifact(
        per_game_accuracy_delta=rows,
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=True,
        bootstrap_iterations=25,
    )
    null = mod.build_artifact(
        per_game_accuracy_delta={
            "cd82": _row("cd82", baseline=0.5, refined=0.5, fixed=0),
            "cn04": _row("cn04", baseline=0.5, refined=0.5, fixed=0),
            "ls20": _row("ls20", baseline=0.5, refined=0.5, fixed=0),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=65.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=False,
        bootstrap_iterations=25,
    )

    assert artifact["honest_verdict"] == "success_cegis_engine_accuracy_lift_0.500000"
    assert artifact["cegis_heldout_accuracy_delta_median"] == 0.5
    assert artifact["cegis_heldout_accuracy_delta_ci95"] == [0.5, 0.5]
    assert artifact["delta_on_truly_heldout_split"] is True
    assert artifact["verifier_is_oracle"] is False
    assert mod.artifact_schema_errors(artifact) == []

    assert null["honest_verdict"] == (
        "complete_cegis_no_heldout_accuracy_lift_residual_positive_control_failed"
    )
    assert mod.artifact_schema_errors(null) == []

    assert mod.build_artifact(
        per_game_accuracy_delta={"cd82": _row("cd82", baseline=0.0, refined=1.0)},
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=True,
    )["honest_verdict"].endswith("_too_few_games")
    bad_split = {
        "cd82": {**_row("cd82", baseline=0.0, refined=1.0), "remeasure_transition_ids": [0]},
        "cn04": _row("cn04", baseline=0.0, refined=1.0),
        "ls20": _row("ls20", baseline=0.0, refined=1.0),
    }
    assert mod.build_artifact(
        per_game_accuracy_delta=bad_split,
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=True,
    )["honest_verdict"].endswith("_split_not_disjoint")
    assert mod.build_artifact(
        per_game_accuracy_delta={
            "cd82": _row("cd82", baseline=0.5, refined=0.5),
            "cn04": _row("cn04", baseline=0.5, refined=0.5),
            "ls20": _row("ls20", baseline=0.5, refined=0.5),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=True,
    )["honest_verdict"].endswith("_nonpositive_delta")
    assert mod.build_artifact(
        per_game_accuracy_delta={
            "cd82": _row("cd82", baseline=0.0, refined=0.1),
            "cn04": _row("cn04", baseline=0.2, refined=0.1),
            "ls20": _row("ls20", baseline=0.0, refined=0.1),
        },
        preconditions_checked={"generator": _generator_ok()},
        live_path_reachable=True,
        duration_s=1.0,
        partial=False,
        checkpoint_emitted=True,
        positive_control_passed=True,
        bootstrap_iterations=50,
    )["honest_verdict"].endswith("_ci_includes_zero")


def test_req_arc_wmte_4872_run_writes_checkpoints_and_schema_valid_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """REQ-ARC-WMTE-4872: run aggregates rows, checkpoints, and writes the result JSON."""

    rows = {
        "cd82": _row("cd82", baseline=0.0, refined=0.5),
        "cn04": _row("cn04", baseline=0.0, refined=0.5),
        "ls20": _row("ls20", baseline=0.0, refined=0.5),
    }
    proposer = object()
    monkeypatch.setattr(mod.a1, "make_live_qwen_proposer", lambda: proposer)
    monkeypatch.setattr(
        mod.a1,
        "generator_available",
        lambda *, proposer: {**_generator_ok(), "proposer_id": id(proposer)},
    )

    def _refine_with_round_checkpoint(game: str, **kwargs: Any) -> dict[str, Any]:
        kwargs["round_checkpoint"](game, {"delta": 0.0, "rounds": []})
        return dict(rows[game])

    artifact = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=None,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_refiner=_refine_with_round_checkpoint,
        now=iter([10.0, 10.1, 10.2, 10.3, 75.0]).__next__,
        write=True,
        heldout_games=("zzzz", "cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )

    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    checkpoint = tmp_path / mod.CHECKPOINT_RELATIVE_DIR / "cd82.json"

    assert loaded == artifact
    assert checkpoint.exists()
    assert artifact["positive_control_passed"] is True
    assert artifact["n_games_measured"] == 3
    assert artifact["live_path_reachable"] is True
    assert mod.artifact_schema_errors(artifact) == []

    resumed = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_refiner=lambda *_args, **_kwargs: pytest.fail("checkpoint should be reused"),
        now=iter([80.0, 80.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        bootstrap_iterations=25,
    )
    assert resumed["n_games_measured"] == 3


def test_req_arc_wmte_4872_run_blocked_and_partial_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4872: blocked and partial runs stay schema-valid."""

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
    live_path = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: False,
        now=iter([3.0, 3.1]).__next__,
        write=False,
    )
    partial = mod.run(
        root=tmp_path,
        offline_arcade_checker=lambda: True,
        generator_checker=_generator_ok,
        a1_artifact_loader=lambda _root: _a1_artifact(),
        live_path_checker=lambda _root: True,
        game_refiner=lambda game, **_kwargs: _row(game, baseline=0.0, refined=0.5),
        now=iter([4.0, 4.1, 5.0, 5.1]).__next__,
        write=False,
        heldout_games=("cd82", "cn04", "ls20"),
        soft_elapsed_budget_s=0.05,
    )

    assert offline["honest_verdict"] == "blocked_offline_arcade_missing"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert generator["honest_verdict"] == "blocked_generator_unavailable"
    assert live_path["honest_verdict"] == "blocked_live_path_unreachable"
    assert partial["partial"] is True
    assert partial["honest_verdict"].endswith("_partial_budget_stop")


def test_req_arc_wmte_4872_delivered_result_json_is_valid() -> None:
    """REQ-ARC-WMTE-4872: final artifact is the requested CEGIS deliverable."""

    artifact_path = REPO / mod.RESULT_RELATIVE_PATH
    artifact: dict[str, Any] = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["delta_on_truly_heldout_split"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["model_specs"]["name"] == "Qwen3.5-9B-MTP"
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
