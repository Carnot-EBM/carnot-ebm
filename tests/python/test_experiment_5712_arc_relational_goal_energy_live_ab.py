"""Tests for Exp5712 relational goal-energy matched-budget live A/B.

Spec refs: REQ-ARC-WMTE-5712,
SCENARIO-ARC-WMTE-5712-MATCHED-BUDGET-AB,
SCENARIO-ARC-WMTE-5712-ROUTE-CONTROLS.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5712_arc_relational_goal_energy_live_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_wmte_5712_spec_declares_matched_ab_contract() -> None:
    """REQ-ARC-WMTE-5712: OpenSpec declares the A/B schema and promotion gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5712") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]

    for marker in (
        "SCENARIO-ARC-WMTE-5712-MATCHED-BUDGET-AB",
        "SCENARIO-ARC-WMTE-5712-ROUTE-CONTROLS",
        "results/experiment_5712_arc_relational_goal_energy_live_ab.json",
        "relational_live_ab_ready_score",
        "control arm SHALL remain the submitted full stack",
    ):
        assert marker in section


def test_req_arc_wmte_5712_blocked_precondition_keeps_denominators_honest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5712: missing gates block before any A/B episode is run."""

    monkeypatch.setattr(
        mod,
        "preconditions",
        lambda root=mod.REPO_ROOT: {"offline_arcade_importable": False, "ok": False},
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_matched_pairs must not run after a failed precondition")

    monkeypatch.setattr(mod, "run_matched_pairs", _fail_if_called)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["honest_verdict"] == "blocked: offline_arcade_importable"
    assert artifact["successful_pair_count"] == 0
    assert artifact["failed_pair_reasons"] == [{"reason": "offline_arcade_importable"}]
    assert artifact["registry_updated"] is False
    assert artifact["new_levels_claimed"] == 0
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _ok_preconditions(root=mod.REPO_ROOT):
    return {
        "offline_arcade_importable": True,
        "e3_policy_importable": True,
        "exp5711_ready": True,
        "ok": True,
    }


def _row(
    game: str,
    *,
    arm: str,
    levels: int,
    actions: int,
    order_changes: int = 0,
    route_activations: int = 0,
    fallback_count: int = 0,
    goal_bias_calls: int = 20,
) -> dict:
    return {
        "game": game,
        "arm": arm,
        "seed": mod.RANDOM_SEEDS[0],
        "start_level": 0,
        "reached_level": levels,
        "levels": levels,
        "actions": actions,
        "actions_to_first_levelup": actions if levels else None,
        "frontier_expansions": actions // 2,
        "candidate_count": actions * 2,
        "candidate_order_change_count": order_changes,
        "score_variance": 0.5 if order_changes else 0.0,
        "route_activation_count": route_activations,
        "invalid_actions": 0,
        "noop_count": 0,
        "fallback_count": fallback_count,
        "goal_bias_call_count": goal_bias_calls,
        "failed_reason": None,
    }


def _fake_successful_pairs(saved_actions: int = 40) -> dict:
    pairs = []
    for game in ("sp80", "cd82", "lp85"):
        pairs.append(
            {
                "game": game,
                "seed": mod.RANDOM_SEEDS[0],
                "control": _row(game, arm="control", levels=1, actions=100),
                "treatment": _row(
                    game,
                    arm="treatment",
                    levels=1,
                    actions=100 - saved_actions,
                    order_changes=1,
                    route_activations=3,
                    fallback_count=2,
                ),
                "failed_reason": None,
            }
        )
    return {"pairs": pairs, "duration_s": 1.25}


def test_req_arc_wmte_5712_ready_score_requires_interval_gain_and_no_regression(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5712-MATCHED-BUDGET-AB: promotion is bounded by paired gates."""

    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_matched_pairs", lambda *_a, **_kw: _fake_successful_pairs())

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["successful_pair_count"] == 3
    assert artifact["failed_pair_reasons"] == []
    assert artifact["levels_reproduced_by_arm"] == {"control": 3, "treatment": 3}
    assert artifact["environment_actions_by_arm"]["treatment"] < artifact[
        "environment_actions_by_arm"
    ]["control"]
    assert artifact["paired_intervals"]["actions_saved_per_reproduced_level"]["ci95_low"] > 0
    assert artifact["level_regression_count"] == 0
    assert artifact["candidate_order_change_count"]["treatment"] == 3
    assert artifact["route_activation_count"]["treatment"] == 9
    assert artifact["unsafe_route_accept_count"] == 0
    assert artifact["relational_live_ab_ready_score"] == 1.0


def test_req_arc_wmte_5712_level_regression_blocks_ready_score(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5712: any level regression beyond margin blocks promotion."""

    data = _fake_successful_pairs()
    data["pairs"][0]["treatment"] = {
        **data["pairs"][0]["treatment"],
        "levels": 0,
        "reached_level": 0,
        "actions": 100,
        "actions_to_first_levelup": None,
    }
    monkeypatch.setattr(mod, "preconditions", _ok_preconditions)
    monkeypatch.setattr(mod, "run_matched_pairs", lambda *_a, **_kw: data)

    artifact = mod.build_artifact(root=tmp_path)

    assert artifact["level_regression_count"] == 1
    assert artifact["relational_live_ab_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_arc_wmte_5712_route_controls_exercise_and_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-5712-ROUTE-CONTROLS: controls are present and safe."""

    controls = mod.run_route_controls()
    by_name = {row["name"]: row for row in controls}

    assert set(by_name) == {
        "disabled_route",
        "shuffled_score",
        "corrupted_mask",
        "always_route",
        "zero_variance",
    }
    assert by_name["shuffled_score"]["intended_ordering_changed"] is True
    assert by_name["shuffled_score"]["intended_exercise"] is True
    assert by_name["disabled_route"]["safe_fallback"] is True
    assert by_name["corrupted_mask"]["safe_fallback"] is True
    assert by_name["always_route"]["safe_fallback"] is True
    assert by_name["zero_variance"]["safe_fallback"] is True
    assert all(row["unsafe_route_accept"] is False for row in controls)


def test_req_arc_wmte_5712_repository_artifact_is_schema_valid() -> None:
    """REQ-ARC-WMTE-5712: checked-in artifact is the stable A/B receipt."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["registry_updated"] is False
    assert artifact["new_levels_claimed"] == 0
    assert artifact["inference_substrate"] == "matched_arc_live_policy_no_llm"
    assert artifact["successful_pair_count"] + len(artifact["failed_pair_reasons"]) > 0
    assert artifact["unsafe_route_accept_count"] == 0
    assert artifact["honest_verdict"].startswith(("complete:", "blocked:"))
    assert len(artifact["reproducibility_checksum"].removeprefix("sha256:")) == 64
