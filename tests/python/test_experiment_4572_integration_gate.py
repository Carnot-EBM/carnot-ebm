"""Tests for Exp 4572 submitted .422 integration gate.

Spec refs: REQ-ARC-WMTE-4572, SCENARIO-ARC-WMTE-4572.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4572_integration_gate as exp4572


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": ok,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "a4_artifact_present": True,
        "spec_has_req_4572": True,
        "leaderboard_submission": False,
        "ok": ok,
    }


def _action_measurement(
    median_actions: float,
    *,
    solved: tuple[str, ...] = exp4572.CORE_GAMES,
    actions_by_game: dict[str, int] | None = None,
) -> dict[str, object]:
    actions = actions_by_game or {game: int(median_actions) for game in solved}
    rows = [
        {
            "game": game,
            "levels": 1 if game in solved else 0,
            "solved": game in solved,
            "actions_to_first_levelup": actions.get(game),
            "deepest_level_reached": 1 if game in solved else 0,
        }
        for game in exp4572.GATE_GAMES
    ]
    return {
        "policy": "fixture",
        "games": list(exp4572.GATE_GAMES),
        "per_game": rows,
        "solved_games": list(solved),
        "actions_to_first_levelup_by_game": actions,
        "median_actions_to_first_levelup": float(median_actions),
        "measurement_source": "fixture",
    }


def _transfer_measurement(rate: float = exp4572.GENERIC_TRANSFER_BASELINE) -> dict[str, object]:
    attempted = 25
    solved = int(round(rate * attempted))
    return {
        "variant_specs": [],
        "variant_attempts": [],
        "variant_attempts_count": attempted,
        "variant_solved_count": solved,
        "generic_transfer_rate_over_variants": rate,
    }


def _a1_artifact(
    *,
    improved: bool = False,
    flagged: bool = False,
    permitted_null: bool = False,
    lost_core: bool = False,
) -> dict[str, object]:
    delta = 1.0 if improved else 0.0
    artifact: dict[str, object] = {
        "honest_verdict": "success: clickability_predictor_actions_to_levelup_1_below_blind"
        if improved
        else "complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened",
        "flagged_adversarial": flagged,
        "median_actions_to_first_levelup_baseline": 2.0,
        "median_actions_to_first_levelup_with_predictor": 1.0 if improved else 2.0,
        "actions_delta": delta,
        "actions_delta_ci": [0.2, 1.4] if improved else [0.0, 0.0],
        "positive_control_passed": True,
        "solve_rate_preserved": not lost_core,
        "offline_reproduced": True,
        "deepest_level_reached_per_core_game": {
            "baseline": {game: 1 for game in exp4572.CORE_GAMES},
            "with_predictor": {
                game: 0 if lost_core and game == "m0r0" else 1
                for game in exp4572.CORE_GAMES
            },
        },
    }
    if permitted_null:
        artifact["corrigendum_pending"] = [{"kind": "TAUTOLOGY"}]
        artifact["null_delta_methodology_note"] = "explicit null-delta TAUTOLOGY control."
    return artifact


def _a2_artifact(
    *,
    improved: bool = False,
    control: bool = True,
    preserved: bool = True,
    lost_core: bool = False,
) -> dict[str, object]:
    rate = 0.08 if improved else 0.0
    return {
        "honest_verdict": "success: verifier_guided_expansion_generic_transfer_0.080_above_0.04"
        if improved
        else "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened",
        "generic_transfer_rate_baseline": exp4572.GENERIC_TRANSFER_BASELINE,
        "generic_transfer_rate_with_expansion": rate,
        "transfer_delta": round(rate - exp4572.GENERIC_TRANSFER_BASELINE, 4),
        "transfer_ci": [0.01, 0.07] if improved else [-0.12, 0.0],
        "random_priority_control_passed": control,
        "solve_rate_preserved": preserved and not lost_core,
        "offline_reproduced": True,
        "deepest_level_reached_per_core_game": {
            "baseline": {game: 1 for game in exp4572.CORE_GAMES},
            "with_expansion": {
                game: 0 if lost_core and game == "lp85" else 1
                for game in exp4572.CORE_GAMES
            },
        },
    }


def _a4_artifact(*, banked: bool = False, lost_core: bool = False) -> dict[str, object]:
    return {
        "honest_verdict": "success: hidden_field_state_ka59_L2_offline_reproduced"
        if banked
        else "complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null",
        "offline_reproduced": banked,
        "reproduced_levels": 1 if banked else 0,
        "registry_updated": banked,
        "core_solves_preserved": not lost_core,
        "deepest_level_reached_per_core_game": {
            "baseline": {game: 1 for game in exp4572.CORE_GAMES},
            "hidden_field_state": {
                game: 0 if lost_core and game == "sp80" else 1
                for game in exp4572.CORE_GAMES
            },
        },
    }


def test_req_arc_wmte_4572_spec_declares_integration_contract() -> None:
    """REQ-ARC-WMTE-4572: OpenSpec anchors the .422 integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4572" in spec
    assert "SCENARIO-ARC-WMTE-4572" in spec
    assert exp4572.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4572.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4572_rejects_current_null_upstreams() -> None:
    """REQ-ARC-WMTE-4572: current A1/A2/A4 nulls do not become submitted levers."""

    decision = exp4572.select_integrated_levers(
        _a1_artifact(),
        _a2_artifact(control=False, preserved=False),
        _a4_artifact(),
    )

    assert decision["accepted_levers"] == []
    assert decision["rejected_levers"]["A1_clickability_predictor:action_efficiency"][
        "reason"
    ] == "no_action_efficiency_gain"
    assert decision["rejected_levers"]["A2_verifier_guided_expansion:generic_transfer"][
        "reason"
    ] == "random_priority_control_failed"
    assert decision["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == (
        "no_new_offline_bank"
    )
    assert decision["isolated_deltas"]["A1_clickability_predictor"]["actions_delta"] == 0.0
    assert decision["isolated_deltas"]["A2_verifier_guided_expansion"][
        "generic_transfer_delta"
    ] == -0.04

    permitted = exp4572.select_integrated_levers(
        _a1_artifact(flagged=True, permitted_null=True),
        _a2_artifact(control=False),
        _a4_artifact(),
    )
    assert permitted["upstream_summaries"]["A1_clickability_predictor"]["flag_status"] == (
        "permitted_flagged_null"
    )


def test_req_arc_wmte_4572_selects_only_real_metric_winners() -> None:
    """REQ-ARC-WMTE-4572: accepted levers must raise action, transfer, or a bank."""

    accepted = exp4572.select_integrated_levers(
        _a1_artifact(improved=True),
        _a2_artifact(improved=True, control=True),
        _a4_artifact(banked=True),
    )
    assert accepted["accepted_levers"] == [
        "A1_clickability_predictor:action_efficiency",
        "A2_verifier_guided_expansion:generic_transfer",
        "A4_hidden_field_state_probe:new_bank",
    ]

    lost = exp4572.select_integrated_levers(
        _a1_artifact(improved=True, lost_core=True),
        _a2_artifact(improved=True, control=True, lost_core=True),
        _a4_artifact(banked=True, lost_core=True),
    )
    assert lost["accepted_levers"] == []
    assert lost["rejected_levers"]["A1_clickability_predictor:action_efficiency"][
        "reason"
    ] == "core_level_regression"
    assert lost["rejected_levers"]["A2_verifier_guided_expansion:generic_transfer"][
        "reason"
    ] == "core_level_regression"
    assert lost["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == (
        "core_level_regression"
    )


def test_scenario_arc_wmte_4572_honest_null_artifact_is_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4572: no winner writes an honest two-metric null."""

    decision = exp4572.select_integrated_levers(
        _a1_artifact(),
        _a2_artifact(control=False),
        _a4_artifact(),
    )
    artifact = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=decision,
        baseline_action_measurement=_action_measurement(10.0),
        integrated_action_measurement=_action_measurement(10.0),
        transfer_measurement=_transfer_measurement(),
        random_seed=4572,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_raises_a_metric_honest_null"
    assert artifact["median_actions_to_first_levelup_integrated"] == 10.0
    assert artifact["generic_transfer_rate_integrated"] == exp4572.GENERIC_TRANSFER_BASELINE
    assert artifact["levers_integrated"] == []
    assert artifact["core_solves_preserved"] is True
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["additivity_checked"]["integrated_actions_delta"] == 0.0
    assert exp4572.artifact_schema_errors(artifact) == []

    out = exp4572.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4572_success_requires_lever_and_integrated_metric_lift() -> None:
    """REQ-ARC-WMTE-4572: submit readiness needs a lever plus an integrated lift."""

    decision = exp4572.select_integrated_levers(
        _a1_artifact(improved=True),
        _a2_artifact(control=False),
        _a4_artifact(),
    )
    artifact = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=decision,
        baseline_action_measurement=_action_measurement(10.0),
        integrated_action_measurement=_action_measurement(5.0),
        transfer_measurement=_transfer_measurement(),
        random_seed=4572,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "success: integrated_actions_to_levelup_below_blind_or_generic_transfer_above_0.04"
    )
    assert artifact["ready_for_operator_submit"] is True
    assert exp4572.artifact_schema_errors(artifact) == []

    dropped_core = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=decision,
        baseline_action_measurement=_action_measurement(10.0),
        integrated_action_measurement=_action_measurement(5.0, solved=("lp85", "sp80", "vc33")),
        transfer_measurement=_transfer_measurement(),
        random_seed=4572,
        duration_s=1.0,
    )
    assert dropped_core["honest_verdict"] == "complete: no_lever_raises_a_metric_honest_null"
    assert dropped_core["core_solves_preserved"] is False
    assert dropped_core["ready_for_operator_submit"] is False


def test_scenario_arc_wmte_4572_run_writes_injected_measurements(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4572: run writes the action and transfer artifact."""

    artifact = exp4572.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        load_upstream_artifacts=lambda _root: {
            "a1_clickability_predictor": _a1_artifact(),
            "a2_verifier_guided_expansion": _a2_artifact(control=False),
            "a4_hidden_field_state_probe": _a4_artifact(),
        },
        action_runner=lambda *, policy, **_kwargs: _action_measurement(
            10.0 if policy == "explorer" else 10.0
        ),
        transfer_runner=lambda **_kwargs: _transfer_measurement(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["operator_submission_performed"] is False
    assert (
        json.loads((tmp_path / exp4572.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        == artifact
    )


def test_req_arc_wmte_4572_schema_blocked_and_errors(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4572: blocked resources and schema failures stay explicit."""

    blocked = exp4572.run(
        write=False,
        preconditions_checked=_preconditions(ok=False),
        action_runner=lambda **_kwargs: _action_measurement(1.0),
        transfer_runner=lambda **_kwargs: _transfer_measurement(rate=1.0),
        now=lambda: 10.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade_import"
    assert blocked["ready_for_operator_submit"] is False

    valid = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=exp4572.select_integrated_levers(
            _a1_artifact(),
            _a2_artifact(control=False),
            _a4_artifact(),
        ),
        baseline_action_measurement=_action_measurement(10.0),
        integrated_action_measurement=_action_measurement(10.0),
        transfer_measurement=_transfer_measurement(),
        random_seed=4572,
        duration_s=1.0,
    )
    bad = {
        **valid,
        "honest_verdict": "done",
        "inference_substrate": "bad",
        "field_principles": {},
        "median_actions_to_first_levelup_integrated": None,
        "generic_transfer_rate_integrated": None,
        "levers_integrated": "bad",
        "additivity_checked": None,
        "core_solves_preserved": "yes",
        "heldout_solve_rate": None,
        "ready_for_operator_submit": True,
        "false_negative_risk_checked": False,
        "operator_submission_performed": True,
        "preconditions_checked": "bad",
        "reproducibility_checksum": "bad",
    }
    del bad["random_seed"]

    errors = exp4572.artifact_schema_errors(bad)
    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match" in errors
    assert "field_principles must match REQ-ARC-WMTE-4572" in errors
    assert "median_actions_to_first_levelup_integrated must be numeric" in errors
    assert "generic_transfer_rate_integrated must be numeric in [0,1]" in errors
    assert "levers_integrated must be a list" in errors
    assert "additivity_checked must be a mapping" in errors
    assert "core_solves_preserved must be bool" in errors
    assert "heldout_solve_rate must be numeric" in errors
    assert "ready_for_operator_submit cannot be true without success" in errors
    assert "false_negative_risk_checked must be true for complete/success artifacts" in errors
    assert "operator_submission_performed must be false" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4572.write_artifact({**valid, "honest_verdict": "bad"}, root=tmp_path)


def test_req_arc_wmte_4572_edge_case_rejections_and_fallback_metrics() -> None:
    """REQ-ARC-WMTE-4572: guard edge cases are explicit and measured."""

    flagged = exp4572.select_integrated_levers(
        _a1_artifact(improved=True, flagged=True),
        _a2_artifact(improved=True, control=True),
        _a4_artifact(banked=True),
    )
    assert flagged["rejected_levers"]["A1_clickability_predictor:action_efficiency"][
        "reason"
    ] == "flagged_adversarial"
    assert flagged["isolated_deltas"]["A1_clickability_predictor"]["actions_delta"] == 0.0

    a1_edges = [
        ({**_a1_artifact(improved=True), "actions_delta_ci": None}, "actions_delta_ci_missing"),
        ({**_a1_artifact(improved=True), "positive_control_passed": False}, "positive_control_failed"),
        ({**_a1_artifact(improved=True), "solve_rate_preserved": False}, "solve_rate_not_preserved"),
    ]
    for artifact, reason in a1_edges:
        decision = exp4572.select_integrated_levers(artifact, _a2_artifact(control=False), _a4_artifact())
        assert decision["rejected_levers"]["A1_clickability_predictor:action_efficiency"][
            "reason"
        ] == reason

    a2_edges = [
        ({**_a2_artifact(improved=True, control=True), "flagged_adversarial": True}, "flagged_adversarial"),
        (_a2_artifact(improved=False, control=True), "no_generic_transfer_gain"),
        ({**_a2_artifact(improved=True, control=True), "transfer_ci": None}, "transfer_ci_missing"),
        ({**_a2_artifact(improved=True, control=True), "solve_rate_preserved": False}, "solve_rate_not_preserved"),
        ({**_a2_artifact(improved=True, control=True), "offline_reproduced": False}, "offline_reproduction_missing"),
    ]
    for artifact, reason in a2_edges:
        decision = exp4572.select_integrated_levers(_a1_artifact(), artifact, _a4_artifact())
        assert decision["rejected_levers"]["A2_verifier_guided_expansion:generic_transfer"][
            "reason"
        ] == reason

    a4_edges = [
        ({**_a4_artifact(banked=True), "flagged_adversarial": True}, "flagged_adversarial"),
        ({**_a4_artifact(banked=True), "core_solves_preserved": False}, "core_solves_not_preserved"),
        ({**_a4_artifact(banked=True), "registry_updated": False}, "registry_update_missing"),
    ]
    for artifact, reason in a4_edges:
        decision = exp4572.select_integrated_levers(_a1_artifact(), _a2_artifact(control=False), artifact)
        assert decision["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == reason

    assert exp4572._float_or_none("bad") is None
    assert exp4572._levels_from_mapping(None) == {}
    assert exp4572._levels_from_nested({}, outer_keys=("missing",), direct_keys=("missing",)) == {
        game: 0 for game in exp4572.CORE_GAMES
    }
    assert exp4572._levels_from_nested(
        {"direct": {"deepest_level_by_game": {"lp85": 2}}},
        outer_keys=("missing",),
        direct_keys=("direct",),
    )["lp85"] == 2
    assert exp4572._corrigendum_kinds({}) == []
    assert exp4572._permitted_flagged_null({"flagged_adversarial": False}) is False
    assert exp4572._median([]) is None
    assert exp4572._solved_games({"per_game": [{"game": "lp85", "levels": 1}]}) == {"lp85"}
    assert exp4572._per_game_deepest_level({"per_game": [None, {"game": "lp85", "reached": 2}]})[
        "lp85"
    ] == 2
    assert exp4572._transfer_rate({"variant_attempts_count": 4, "variant_solved_count": 1}) == 0.25

    fallback = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=exp4572.select_integrated_levers(
            _a1_artifact(),
            _a2_artifact(control=False),
            _a4_artifact(),
        ),
        baseline_action_measurement={"measurement_source": "fixture", "per_game": [], "solved_games": []},
        integrated_action_measurement={"measurement_source": "fixture", "per_game": [], "solved_games": []},
        transfer_measurement={"variant_attempts_count": 4, "variant_solved_count": 1},
        random_seed=4572,
        duration_s=1.0,
    )
    assert fallback["median_actions_to_first_levelup_integrated"] == float("inf")
    assert fallback["generic_transfer_rate_integrated"] == 0.25

    success = exp4572.build_artifact(
        preconditions_checked=_preconditions(),
        upstream_decision=exp4572.select_integrated_levers(
            _a1_artifact(improved=True),
            _a2_artifact(control=False),
            _a4_artifact(),
        ),
        baseline_action_measurement=_action_measurement(10.0),
        integrated_action_measurement=_action_measurement(5.0),
        transfer_measurement=_transfer_measurement(),
        random_seed=4572,
        duration_s=1.0,
    )
    success_mutations = [
        (
            lambda item: item.__setitem__("median_actions_to_first_levelup_integrated", 10.0),
            "success requires action or generic-transfer lift",
        ),
        (
            lambda item: item.__setitem__("core_solves_preserved", False),
            "success requires core_solves_preserved=true",
        ),
        (
            lambda item: item.__setitem__("levers_integrated", []),
            "success requires an integrated lever",
        ),
    ]
    for mutate, expected in success_mutations:
        changed = dict(success)
        mutate(changed)
        assert expected in exp4572.artifact_schema_errors(changed)

    with pytest.raises(ValueError, match="false_negative_risk_checked"):
        exp4572.run(
            write=False,
            preconditions_checked=_preconditions(),
            load_upstream_artifacts=lambda _root: {
                "a1_clickability_predictor": _a1_artifact(),
                "a2_verifier_guided_expansion": _a2_artifact(control=False),
                "a4_hidden_field_state_probe": _a4_artifact(),
            },
            action_runner=lambda *, policy, **_kwargs: {
                **_action_measurement(10.0),
                "measurement_source": policy,
            },
            transfer_runner=lambda **_kwargs: _transfer_measurement(),
            now=lambda: 10.0,
        )
