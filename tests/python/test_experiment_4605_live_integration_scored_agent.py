"""Tests for Exp 4605 scored-agent live integration.

Spec refs: REQ-CAPSTONE-4605, SCENARIO-CAPSTONE-4605,
SCENARIO-CAPSTONE-4605-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "e3_policy_import": True,
        "discriminative_router_import": True,
        "spec_has_req_4605": True,
        "leaderboard_submission": False,
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


def test_req_capstone_4605_spec_declares_live_integration_contract() -> None:
    """REQ-CAPSTONE-4605: OpenSpec declares the live integration artifact schema."""

    from carnot import experiment_4605_live_integration_scored_agent as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4605" in spec
    assert "SCENARIO-CAPSTONE-4605" in spec
    assert "SCENARIO-CAPSTONE-4605-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4605_submitted_policy_wires_safe_live_stack(monkeypatch) -> None:
    """REQ-CAPSTONE-4605/REQ-LEARN-4652: submitted E3 uses bounded value routing."""

    from carnot.agentic import arc_competition_agent as comp
    from carnot.agentic.arc_discriminative_router import CrossGameDiscriminativeCandidateRouter

    class StubVerifier:
        def proba_features(self, _features):
            return 0.5

    stub_router = CrossGameDiscriminativeCandidateRouter(StubVerifier())
    # Patch load_online_click_target_router, not load_cross_game_discriminative_router directly:
    # since commit ca8e078ccb (2026-07-25, Exp5904/5927) _load_submitted_candidate_router's
    # default path wraps the discriminative router inside an OnlineClickTargetRouter, so a stub
    # swapped in at load_cross_game_discriminative_router only ever reaches .base of that
    # wrapper, never policy.explorer.candidate_router itself.
    monkeypatch.setattr(
        comp.arc_discriminative_router,
        "load_online_click_target_router",
        lambda **_kwargs: stub_router,
    )

    policy = comp.E3AgentPolicy("tn36", proposer=None, value_head=lambda _frame: 0.0)

    assert policy.explorer.value_weight == comp.SUBMITTED_VALUE_WEIGHT
    assert 0.0 < policy.explorer.value_weight <= 1e-9
    assert policy.explorer.target_levels > 1
    assert policy.explorer.navigation_cost_tiebreak is True
    assert policy.explorer.candidate_router is stub_router
    assert policy.approach_recommendation["strategy"]["game"] == "tn36"
    assert policy.strategy_route == policy.approach_recommendation["strategy"]
    assert comp.SUBMITTED_AGENT_CONFIG["value_weight"] == comp.SUBMITTED_VALUE_WEIGHT
    assert comp.SUBMITTED_AGENT_CONFIG["value_head_feature_subset"] == (
        "cross_game_features_v3:v2_plus_frame_delta"
    )
    assert comp.SUBMITTED_AGENT_CONFIG["target_levels"] > 1
    assert comp.SUBMITTED_AGENT_CONFIG["discriminative_candidate_router_enabled"] is True
    assert comp.SUBMITTED_AGENT_CONFIG["verifier_is_oracle"] is False


def test_req_capstone_4605_artifact_success_can_come_from_actions_delta() -> None:
    """REQ-CAPSTONE-4605: lower actions with preserved solve-rate is a valid win."""

    from carnot import experiment_4605_live_integration_scored_agent as mod

    integrated = mod.measurement_from_attempts(
        [
            _attempt("integrated", "aa00~color01", solved=True, actions=8),
            _attempt("integrated", "bb00~color01", solved=False, actions=None),
        ]
    )
    bare = mod.measurement_from_attempts(
        [
            _attempt("bare", "aa00~color01", solved=True, actions=13),
            _attempt("bare", "bb00~color01", solved=False, actions=None),
        ]
    )
    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        integrated_measurement=integrated,
        bare_measurement=bare,
        parity_test={"passed": True},
        duration_s=1.0,
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["first_win_delta"] == 0.0
    assert artifact["actions_delta"] == 5.0
    assert artifact["solve_rate_preserved"] is True
    assert artifact["bare_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert 0.0 < artifact["value_weight_used"] <= 1e-9
    assert "null_delta_methodology_note" in artifact
    assert artifact["chosen_submitted_config"]["target_levels"] > 1
    assert 0.0 < artifact["chosen_submitted_config"]["value_weight"] <= 1e-9
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4605_runner_writes_matched_control_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4605: runner writes integrated-vs-bare measurements."""

    from carnot import experiment_4605_live_integration_scored_agent as mod

    rows_by_mode = {
        "integrated": {
            "aa00~color01": _attempt("integrated", "aa00~color01", solved=True, actions=7),
            "bb00~color01": _attempt("integrated", "bb00~color01", solved=True, actions=11),
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
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "success: live_integration_scored_first_win_up_2"
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["first_win_rate_integrated"] == 1.0
    assert artifact["first_win_rate_bare"] == 0.0
    assert artifact["first_win_delta"] == 1.0
    assert artifact["first_win_ci"]["point"] == 1.0
    assert artifact["median_actions_to_first_levelup_integrated"] == 9.0
    assert artifact["actions_delta"] == 0.0
    assert artifact["parity_test_green"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4605_null_and_blocked_artifacts_are_auditable(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4605: null and missing-resource paths fail closed."""

    from carnot import experiment_4605_live_integration_scored_agent as mod

    solved_without_action_count = _attempt(
        "integrated",
        "zz00~color01",
        solved=True,
        actions=None,
    )
    solved_without_action_count["actions"] = 0
    assert mod._actions_to_first_levelup(solved_without_action_count) is None

    unreproduced_new_win = _attempt("integrated", "yy00~color01", solved=True, actions=9)
    unreproduced_new_win["reproduction_gate"] = {"reproduced": False}
    assert (
        mod._offline_reproduced(
            mod.measurement_from_attempts([unreproduced_new_win]),
            mod.measurement_from_attempts([]),
        )
        is False
    )

    integrated = mod.measurement_from_attempts(
        [_attempt("integrated", "aa00~color01", solved=False, actions=None)]
    )
    bare = mod.measurement_from_attempts(
        [_attempt("bare", "aa00~color01", solved=False, actions=None)]
    )
    null_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        integrated_measurement=integrated,
        bare_measurement=bare,
        parity_test={"passed": True},
        duration_s=1.0,
    )

    assert null_artifact["honest_verdict"] == (
        "complete: live_integration_no_value_honest_null_gap_sharpened"
    )
    assert null_artifact["bare_control_passed"] is True
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
    assert "value_weight_zero" in errors
    assert "reproducibility_checksum" in errors
    assert "null_delta_methodology_note" in errors

    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        parity_check=lambda _root: {"passed": False},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["chosen_submitted_config"] == "unchanged"
