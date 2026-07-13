"""Tests for Exp 4617 spatial value-head live graduation.

Spec refs: REQ-ARC-WMTE-4617, SCENARIO-ARC-WMTE-4617-LIVE-PATH,
SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS, SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


class _Frame:
    def __init__(self, grid: Any):
        self.frame = grid


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "e3_policy_import": True,
        "value_net_import": True,
        "a1_artifact_present": True,
        "a1_binding_bridge_cause": "compute_cost",
        "a1_indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
        "leaderboard_submission": False,
        "live_llm_inference": False,
    }


def _attempt(
    mode: str,
    signature: str,
    *,
    solved: bool,
    actions: int | None,
    reached_level: int = 1,
) -> dict[str, Any]:
    return {
        "game": signature.split("~", 1)[0],
        "variant_signature": signature,
        "variant": 1,
        "kind": "color",
        "reflect": None,
        "attempted": True,
        "solved": bool(solved),
        "first_win": bool(solved),
        "reached_level": int(reached_level if solved else 0),
        "actions": actions if actions is not None else 200,
        "actions_to_first_levelup": actions if solved else None,
        "solution_labels": ["{}"] if solved else [],
        "reproduction_gate": {
            "game": signature.split("~", 1)[0],
            "claimed_level": int(reached_level if solved else 0),
            "reached_level": int(reached_level if solved else 0),
            "reproduced": bool(solved),
        },
        "blocked_reason": "",
        "policy_mode": mode,
    }


def _runner_factory(rows_by_mode: Mapping[str, Mapping[str, dict[str, Any]]]):
    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            row = dict(rows_by_mode[mode][signature])
            row.setdefault("game", game)
            row.setdefault("variant_signature", signature)
            row.setdefault("attempted", True)
            return row

        return run

    return _runner


def test_req_arc_wmte_4617_spec_declares_spatial_value_live_contract() -> None:
    """REQ-ARC-WMTE-4617: OpenSpec declares the live graduation artifact schema."""

    from carnot import experiment_4617_graduate_spatial_value_head_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4617" in spec
    assert "SCENARIO-ARC-WMTE-4617-LIVE-PATH" in spec
    assert "SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS" in spec
    assert "SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


@pytest.mark.memory_watchdog_skip
def test_req_arc_wmte_4617_spatial_value_net_preserves_4x4_position_and_loads(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4617: SpatialValueNet is a live importable 4x4-pool value head."""

    from carnot.agentic.arc_value_net import (
        GRID,
        SpatialValueNet,
        load_live_spatial_value_head,
    )

    net = SpatialValueNet(device="cpu")
    assert net.spatial_pool_size == 4
    assert net.trained is False
    assert net(_Frame(np.zeros((GRID, GRID), dtype=np.int64))) == 0.0

    grids = []
    for marker, value in ((1, 3.0), (2, 9.0)):
        grid = np.zeros((GRID, GRID), dtype=np.int64)
        grid[marker, GRID - marker - 1] = marker
        grids.append(grid)
    net.fit(grids, [3.0, 9.0], epochs=1, batch=2, seed=4617)
    assert net.trained is True
    assert isinstance(net.predict_grid(grids[0]), float)
    assert isinstance(net(_Frame(grids[0])), float)

    model_path = tmp_path / "models" / "arc_spatial_value_head_live.json"
    net.save(model_path, meta={"spec_ref": "REQ-ARC-WMTE-4617"})
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "carnot_arc_spatial_value_cnn_v1"
    assert payload["spatial_pool"] == 4
    assert payload["kind"] == "spatial_grid_cnn_value_head"

    loaded = SpatialValueNet.load(model_path, device="cpu")
    assert loaded.trained is True
    assert isinstance(loaded(_Frame(grids[1])), float)
    assert load_live_spatial_value_head(root=tmp_path, device="cpu") is not None
    assert load_live_spatial_value_head(root=tmp_path, game="missing", device="cpu") is not None

    bad_path = tmp_path / "models" / "arc_spatial_value_head_bad.json"
    payload["spatial_pool"] = 2
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        SpatialValueNet.load(bad_path, device="cpu")

    missing_root = tmp_path / "missing-root"
    assert load_live_spatial_value_head(root=missing_root, device="cpu") is None
    corrupt = tmp_path / "corrupt" / "models" / "arc_spatial_value_head_live.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{not-json", encoding="utf-8")
    assert load_live_spatial_value_head(root=tmp_path / "corrupt", device="cpu") is None


def test_scenario_arc_wmte_4617_live_path_reachability(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4617-LIVE-PATH: both live entrypoints reach SpatialValueNet."""

    from carnot.agentic import arc_competition_agent as comp

    arc_loop_src = (REPO / "scripts" / "arc_loop_solve.py").read_text(encoding="utf-8")
    comp_src = inspect.getsource(comp)

    assert "load_live_spatial_value_head" in arc_loop_src
    # NOTE (2026-07-12): commit df207c1da (PHASE A3, the self-play loop) reintroduced
    # LearnedVerifier.load into _live_verifier_for_adapter as a NARROW, GATED
    # last-resort fallback -- tried ONLY when load_live_spatial_value_head(...)
    # returns None for that specific game (no spatial checkpoint trained yet) AND
    # a legacy per-game checkpoint + adapter.featurize both exist. This is
    # development_proxy scope only (arc_loop_solve.py is the offline dev/registry
    # tool, never the live scored path -- see CLAUDE.md's ARC Live-Path
    # Reachability Discipline), so PHASE A2's "the linear LearnedVerifier warm-
    # start actively misled" finding (about the SCORED path) does not apply here;
    # the assertion below checks the intent that motivated PHASE A2 (spatial is
    # tried FIRST, LearnedVerifier is fallback-only) rather than banning the
    # identifier outright, which would incorrectly flag this legitimate,
    # self-play-bootstrapping-only fallback as a regression.
    spatial_call_index = arc_loop_src.index("load_live_spatial_value_head(")
    learned_verifier_index = arc_loop_src.index("LearnedVerifier.load(")
    assert spatial_call_index < learned_verifier_index
    assert "load_live_spatial_value_head" in comp_src
    assert comp.SUBMITTED_AGENT_CONFIG["value_weight"] == comp.SUBMITTED_VALUE_WEIGHT
    assert 0.0 < comp.SUBMITTED_AGENT_CONFIG["value_weight"] <= 1e-9

    sentinel = lambda _frame: 4.0
    monkeypatch.setattr(comp, "load_live_spatial_value_head", lambda *_, **__: sentinel)
    monkeypatch.setattr(comp, "_load_linear_cross_game_value_head", lambda *_, **__: None)
    assert comp.load_cross_game_value_head() is sentinel

    linear = lambda _frame: 7.0
    monkeypatch.setattr(comp, "load_live_spatial_value_head", lambda *_, **__: None)
    monkeypatch.setattr(comp, "_load_linear_cross_game_value_head", lambda *_, **__: linear)
    assert comp.load_cross_game_value_head() is linear

    monkeypatch.setattr(comp, "load_live_spatial_value_head", lambda *_, **__: sentinel)
    policy = comp.E3AgentPolicy("paritytest", proposer=None, candidate_router=None)

    assert policy.explorer.value_head is sentinel
    assert policy.explorer.value_weight == comp.SUBMITTED_VALUE_WEIGHT
    assert policy.explorer.lazy_value_top_k <= comp.SUBMITTED_LAZY_VALUE_TOP_K


def test_scenario_arc_wmte_4617_arc_loop_uses_spatial_warm_start(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4617-LIVE-PATH: arc_loop selects the spatial live verifier first."""

    spec = importlib.util.spec_from_file_location(
        "arc_loop_solve_4617", REPO / "scripts" / "arc_loop_solve.py"
    )
    assert spec is not None and spec.loader is not None
    arc_loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arc_loop)

    hand = lambda _game: 9.0
    adapter = SimpleNamespace(hand_verifier=hand)
    spatial = lambda _frame: 1.0
    monkeypatch.setattr(arc_loop, "load_live_spatial_value_head", lambda **_kwargs: spatial)

    verifier, source = arc_loop._live_verifier_for_adapter("ls20", adapter)
    assert verifier is spatial
    assert source == "spatial_value_head_live_checkpoint"

    monkeypatch.setattr(arc_loop, "load_live_spatial_value_head", lambda **_kwargs: None)
    verifier, source = arc_loop._live_verifier_for_adapter("ls20", adapter)
    assert verifier is hand
    assert source == "hand_verifier_cold_start_no_spatial_checkpoint"


def test_req_arc_wmte_4617_artifact_success_can_come_from_actions_delta() -> None:
    """REQ-ARC-WMTE-4617: lower actions with preserved solve-rate is a valid win."""

    from carnot import experiment_4617_graduate_spatial_value_head_live as mod

    graduated = mod.measurement_from_attempts(
        [
            _attempt("graduated", "aa00~color01", solved=True, actions=8),
            _attempt("graduated", "bb00~color01", solved=False, actions=None),
        ]
    )
    linear = mod.measurement_from_attempts(
        [
            _attempt("linear", "aa00~color01", solved=True, actions=13),
            _attempt("linear", "bb00~color01", solved=False, actions=None),
        ]
    )
    bare = mod.measurement_from_attempts(
        [
            _attempt("bare", "aa00~color01", solved=True, actions=20),
            _attempt("bare", "bb00~color01", solved=False, actions=None),
        ]
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        graduated_measurement=graduated,
        linear_measurement=linear,
        bare_measurement=bare,
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["first_win_delta"] == 0.0
    assert artifact["actions_delta"] == 5.0
    assert artifact["solve_rate_preserved"] is True
    assert artifact["bare_and_linear_controls_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["value_weight_used"] == mod.GRADUATED_VALUE_WEIGHT
    assert artifact["bridge_fix_applied"]["mode"] == "decision_point_cached_tiebreak"
    assert "null_delta_methodology_note" in artifact
    assert 0.0 < artifact["chosen_submitted_config"]["value_weight"] <= 1e-9
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4617_runner_writes_three_arm_control_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS: runner writes three matched arms."""

    from carnot import experiment_4617_graduate_spatial_value_head_live as mod

    rows_by_mode = {
        "graduated": {
            "aa00~color01": _attempt("graduated", "aa00~color01", solved=True, actions=7),
            "bb00~color01": _attempt("graduated", "bb00~color01", solved=True, actions=11),
        },
        "linear": {
            "aa00~color01": _attempt("linear", "aa00~color01", solved=False, actions=None),
            "bb00~color01": _attempt("linear", "bb00~color01", solved=False, actions=None),
        },
        "bare": {
            "aa00~color01": _attempt("bare", "aa00~color01", solved=False, actions=None),
            "bb00~color01": _attempt("bare", "bb00~color01", solved=False, actions=None),
        },
    }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        public_games=("aa00", "bb00"),
        variant_ids=(1,),
        budget=50,
        variant_runner_factory=_runner_factory(rows_by_mode),
        parity_check=lambda _root: {"passed": True, "command": "pytest parity"},
        orphan_lint=lambda _root: {"passed": True, "command": "arc orphan lint"},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "success: spatial_value_head_graduated_live_first_win_up_2"
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["value_head_live_path_reachable"] is True
    assert artifact["first_win_rate_graduated"] == 1.0
    assert artifact["first_win_rate_linear_baseline"] == 0.0
    assert artifact["first_win_rate_bare"] == 0.0
    assert artifact["first_win_delta"] == 1.0
    assert artifact["first_win_ci"]["point"] == 1.0
    assert artifact["median_actions_to_first_levelup_graduated"] == 9.0
    assert artifact["actions_delta"] == 0.0
    assert artifact["parity_test_green"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4617_null_and_blocked_artifacts_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION: nulls and blocks fail closed."""

    from carnot import experiment_4617_graduate_spatial_value_head_live as mod

    graduated = mod.measurement_from_attempts(
        [_attempt("graduated", "aa00~color01", solved=False, actions=None)]
    )
    linear = mod.measurement_from_attempts(
        [_attempt("linear", "aa00~color01", solved=False, actions=None)]
    )
    bare = mod.measurement_from_attempts(
        [_attempt("bare", "aa00~color01", solved=False, actions=None)]
    )
    null_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        graduated_measurement=graduated,
        linear_measurement=linear,
        bare_measurement=bare,
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        duration_s=1.0,
    )

    assert null_artifact["honest_verdict"] == (
        "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened"
    )
    assert null_artifact["bare_and_linear_controls_passed"] is True
    assert "null_delta_methodology_note" in null_artifact
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert mod.artifact_schema_errors(null_artifact) == []

    broken = dict(null_artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["verifier_is_oracle"] = True
    broken["value_weight_used"] = 5.0
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_delta_methodology_note")
    errors = mod.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "value_weight_bounded" in errors
    assert "reproducibility_checksum" in errors
    assert "null_delta_methodology_note" in errors

    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        parity_check=lambda _root: {"passed": False},
        orphan_lint=lambda _root: {"passed": False},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["chosen_submitted_config"] == "unchanged"
