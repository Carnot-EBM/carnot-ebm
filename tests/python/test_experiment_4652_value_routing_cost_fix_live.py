"""Tests for Exp 4652 value-routing cost-fix productionization.

Spec refs: REQ-LEARN-4652, SCENARIO-LEARN-4652-COMPONENTS,
SCENARIO-LEARN-4652-VALUE-ROUTE, SCENARIO-LEARN-4652-LIVE-ARTIFACT.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest


pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _frame(grid: Any, levels: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame=np.asarray(grid, dtype=np.int16).tolist(), levels_completed=levels)


def _block_scipy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _component_signature(rows: list[dict[str, float]]) -> list[tuple[float, ...]]:
    return [
        tuple(round(float(row[key]), 12) for key in ("cy", "cx", "area", "color", "y0", "y1", "x0", "x1"))
        for row in rows
    ]


def test_req_learn_4652_spec_declares_value_routing_cost_fix_contract() -> None:
    """REQ-LEARN-4652: OpenSpec anchors the live value-routing cost-fix artifact."""

    from carnot import experiment_4652_value_routing_cost_fix_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4652" in spec
    assert "SCENARIO-LEARN-4652-COMPONENTS" in spec
    assert "SCENARIO-LEARN-4652-VALUE-ROUTE" in spec
    assert "SCENARIO-LEARN-4652-LIVE-ARTIFACT" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_learn_4652_component_fast_path_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-4652-COMPONENTS: scipy and fallback labels are output-identical."""

    from carnot.agentic import arc_agi3_world_model as wm
    from carnot.agentic import arc_value_learner as vl

    rng = np.random.default_rng(4652)
    grids = [
        rng.integers(0, 6, size=(int(rng.integers(2, 14)), int(rng.integers(2, 14))), dtype=np.int16)
        for _ in range(40)
    ]

    fast_objects = [wm.objects(grid) for grid in grids]
    fast_stats = [_component_signature(vl._component_stats_from_grid(grid.astype(float))) for grid in grids]
    _block_scipy_import(monkeypatch)
    fallback_objects = [wm.objects(grid) for grid in grids]
    fallback_stats = [
        _component_signature(vl._component_stats_from_grid(grid.astype(float))) for grid in grids
    ]

    assert fallback_objects == fast_objects
    assert fallback_stats == fast_stats


def test_scenario_learn_4652_value_route_uses_only_v2_plus_frame_delta() -> None:
    """SCENARIO-LEARN-4652-VALUE-ROUTE: live routing drops action and predicate slices."""

    from carnot.agentic.arc_value_learner import (
        cross_game_feature_slices_v3,
        cross_game_features_v2,
        cross_game_features_v3_value_routing,
    )

    prev = _frame(
        [
            [0, 0, 0, 0],
            [0, 1, 0, 2],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
        ],
        levels=0,
    )
    cur = _frame(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 2],
            [0, 0, 0, 0],
            [0, 0, 3, 0],
        ],
        levels=1,
    )
    goal = _frame(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
        ],
        levels=1,
    )

    slices = cross_game_feature_slices_v3()
    values = cross_game_features_v3_value_routing(
        cur,
        previous_frame=prev,
        action_id=2,
        goal_frame=goal,
    )
    action_goal_changed = cross_game_features_v3_value_routing(
        cur,
        previous_frame=prev,
        action_id=6,
        goal_frame=prev,
    )
    frame_only = cross_game_features_v3_value_routing(cur)

    expected_len = (slices["v2"][1] - slices["v2"][0]) + (
        slices["frame_delta"][1] - slices["frame_delta"][0]
    )
    assert len(values) == expected_len
    assert len(values) == len(cross_game_features_v2(cur)) + 9
    assert values == action_goal_changed
    assert values != frame_only


def test_req_learn_4652_submitted_policy_uses_positive_cost_fixed_value_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-4652-2/3: submitted E3 routes by cheap frame-delta value features."""

    from carnot.agentic import arc_competition_agent as comp
    from carnot.agentic.arc_value_learner import (
        cross_game_feature_slices_v3,
        cross_game_features_v3_value_routing,
    )

    slices = comp._value_routing_feature_indices()
    full_width = max(stop for _name, (_start, stop) in cross_game_feature_slices_v3().items())
    weights = [0.0] * full_width
    weights[slices[0]] = 2.0
    weights[slices[-1]] = 3.0
    weights.append(5.0)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "arc_verifier_cross_game_v3.json").write_text(
        json.dumps(
            {
                "schema": "carnot_arc_learned_verifier_v1",
                "kind": "linear_value_head",
                "weights": weights,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(comp, "load_live_spatial_value_head", lambda *args, **kwargs: None)
    head = comp._load_linear_cross_game_value_head(root=tmp_path)
    prev = _frame([[0, 0], [0, 1]], levels=0)
    cur = _frame([[0, 1], [0, 1]], levels=1)
    features = cross_game_features_v3_value_routing(cur, previous_frame=prev)

    assert head is not None
    assert head(cur, previous_frame=prev) == pytest.approx(
        max(0.0, 2.0 * features[0] + 3.0 * features[-1] + 5.0)
    )
    assert comp.SUBMITTED_VALUE_WEIGHT > 0.0
    assert comp.SUBMITTED_AGENT_CONFIG["value_weight"] == comp.SUBMITTED_VALUE_WEIGHT
    assert comp.SUBMITTED_AGENT_CONFIG["value_head_feature_subset"] == "cross_game_features_v3:v2_plus_frame_delta"
    policy = comp.E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    assert policy.explorer.value_weight == comp.SUBMITTED_VALUE_WEIGHT


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "e3_policy_import": True,
        "world_model_import": True,
        "value_learner_import": True,
        "scipy_ndimage": True,
        "spec_has_req_4652": True,
        "leaderboard_submission": False,
        "live_llm_inference": False,
    }


def _attempt(
    mode: str,
    signature: str,
    *,
    first_win: bool,
    reached_level: int,
    actions: int | None = 8,
) -> dict[str, Any]:
    return {
        "game": signature.split("~", 1)[0],
        "variant_signature": signature,
        "variant": 1,
        "kind": "color",
        "reflect": None,
        "attempted": True,
        "first_win": bool(first_win),
        "solved": bool(first_win),
        "reached_level": int(reached_level),
        "actions": actions if actions is not None else 200,
        "actions_to_first_levelup": actions if first_win else None,
        "solution_labels": ["{}"] if first_win else [],
        "reproduction_gate": {"reproduced": bool(first_win)},
        "blocked_reason": "",
        "policy_mode": mode,
        "timed_out": False,
    }


def test_scenario_learn_4652_artifact_schema_accepts_lift_and_null(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4652-LIVE-ARTIFACT: matched controls decide lift or residual null."""

    from carnot import experiment_4652_value_routing_cost_fix_live as mod

    routed = mod.measurement_from_attempts(
        [
            _attempt("value_routed", "aa00~color01", first_win=True, reached_level=2, actions=7),
            _attempt("value_routed", "bb00~color01", first_win=True, reached_level=2, actions=9),
        ]
    )
    baseline = mod.measurement_from_attempts(
        [
            _attempt("baseline", "aa00~color01", first_win=False, reached_level=0),
            _attempt("baseline", "bb00~color01", first_win=False, reached_level=0),
        ]
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        value_routed_measurement=routed,
        baseline_measurement=baseline,
        feature_cost={"per_node_feature_cost_ms": 0.42, "feature_output_identical_verified": True},
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        sim_timed_out=False,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "success: value_routing_cost_fixed_live_firstwin_up_2"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["live_path_reachable"] is True
    assert artifact["feature_output_identical_verified"] is True
    assert artifact["per_node_feature_cost_ms"] == 0.42
    assert artifact["sim_timed_out"] is False
    assert artifact["value_weight_set"] > 0.0
    assert artifact["live_first_win_rate_value_routed"] == 1.0
    assert artifact["live_solve_rate_value_routed"] == 1.0
    assert artifact["live_baseline_value_weight_zero"]["first_win_rate"] == 0.0
    assert artifact["first_win_rate_delta"] == 1.0
    assert artifact["solve_rate_delta"] == 1.0
    assert artifact["live_lift_ci"]["ci95"] == [1.0, 1.0]
    assert artifact["bare_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["residual_cause_hypothesis"] == "none"
    assert artifact["chosen_submitted_config"]["value_weight"] == artifact["value_weight_set"]
    assert artifact["parity_test_green"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []

    null_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        value_routed_measurement=baseline,
        baseline_measurement=baseline,
        feature_cost={"per_node_feature_cost_ms": 0.42, "feature_output_identical_verified": True},
        parity_test={"passed": True},
        orphan_lint={"passed": True},
        sim_timed_out=False,
        duration_s=1.0,
    )
    assert null_artifact["honest_verdict"] == (
        "complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration."
    )
    assert null_artifact["residual_cause_hypothesis"] == "distribution_shift_or_calibration"
    assert "null_delta_methodology_note" in null_artifact
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert mod.artifact_schema_errors(null_artifact) == []
