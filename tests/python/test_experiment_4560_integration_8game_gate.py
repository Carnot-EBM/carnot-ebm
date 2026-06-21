"""Tests for Exp 4560 submitted A1/A2/A4 integration.

Spec refs: REQ-ARC-WMTE-4560, SCENARIO-ARC-WMTE-4560.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4560_integration_8game_gate as exp4560


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
        "spec_has_req_4560": True,
        "ok": ok,
    }


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4560.GATE_GAMES),
        "solved_games": list(exp4560.CORE_GAMES),
        "core_efficiency": exp4560.CORE_EFFICIENCY_BASELINE,
        "generic_transfer_rate_over_variants": exp4560.GENERIC_TRANSFER_BASELINE,
    }


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _a1_artifact(
    *,
    flagged: bool = True,
    permitted_null: bool = False,
    improved: bool = False,
    lost_core: bool = False,
) -> dict[str, object]:
    rate = 0.08 if improved else exp4560.GENERIC_TRANSFER_BASELINE
    artifact: dict[str, object] = {
        "honest_verdict": "complete: verifier_router_no_value_added_honest_null_gap_sharpened",
        "flagged_adversarial": flagged,
        "generic_transfer_rate_baseline": exp4560.GENERIC_TRANSFER_BASELINE,
        "generic_transfer_rate_with_verifier": rate,
        "generic_transfer_delta": round(rate - exp4560.GENERIC_TRANSFER_BASELINE, 4),
        "generic_transfer_ci": [0.01, 0.07] if improved else [0.0, 0.0],
        "solve_rate_preserved": not lost_core,
        "offline_reproduced": True,
        "random_router_control_passed": True,
        "deepest_level_reached_per_core_game": {
            "baseline": _levels(),
            "verifier_router": _levels(m0r0=0 if lost_core else 1),
        },
    }
    if permitted_null:
        artifact["corrigendum_pending"] = [{"kind": "TAUTOLOGY"}]
        artifact["null_delta_methodology_note"] = (
            "baseline==best with explicit null-delta control tautology."
        )
    return artifact


def _a2_artifact(
    *,
    improved: bool = False,
    lost_core: bool = False,
    preserved: bool = True,
) -> dict[str, object]:
    best = 3.0 if improved else exp4560.CORE_EFFICIENCY_BASELINE
    return {
        "honest_verdict": "success: executable_proposer_lp85_reached_L2"
        if improved
        else "complete: executable_proposer_positive_control_failed_no_deeper_barrier_refined",
        "core_efficiency_baseline": exp4560.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": None
        if not improved
        else round(best - exp4560.CORE_EFFICIENCY_BASELINE, 4),
        "core_solves_preserved": preserved and not lost_core,
        "offline_reproduced": improved,
        "positive_control_passed": improved,
        "deepest_level_reached_per_core_game": {
            "offline_dsl_baseline": _levels(),
            "executable_proposer": _levels(lp85=2 if improved else 1, m0r0=0 if lost_core else 1),
        },
    }


def _a4_artifact(*, banked: bool = False, lost_core: bool = False) -> dict[str, object]:
    return {
        "honest_verdict": "success: hidden_field_state_bank_offline_reproduced"
        if banked
        else "complete: hidden_field_state_gap_sharpened_no_bank_honest_null",
        "offline_reproduced": banked,
        "reproduced_levels": 1 if banked else 0,
        "registry_updated": banked,
        "core_solves_preserved": not lost_core,
        "deepest_level_reached_per_core_game": {
            "baseline": _levels(),
            "hidden_field_state": _levels(sp80=2 if banked else 1, m0r0=0 if lost_core else 1),
        },
    }


def _gate_result(
    *, core_efficiency: float = 2.0074, solved: list[str] | None = None
) -> dict[str, object]:
    solved_games = solved or list(exp4560.CORE_GAMES)
    rows = [
        {
            "game": game,
            "solved": game in solved_games,
            "actions": 10 if game in solved_games else 20,
            "efficiency": (
                core_efficiency / len(exp4560.CORE_GAMES) if game in exp4560.CORE_GAMES else 0.0
            ),
            "timed_out": False,
            "levels": 1 if game in solved_games else 0,
        }
        for game in exp4560.GATE_GAMES
    ]
    return {
        "pass": core_efficiency >= exp4560.CORE_EFFICIENCY_BASELINE,
        "baseline_guard": {"ok": True},
        "current": {
            "policy": "e3",
            "games": list(exp4560.GATE_GAMES),
            "per_game": rows,
            "solved_games": solved_games,
            "core_efficiency": core_efficiency,
            "timed_out_count": 0,
        },
    }


def _transfer_measurement(rate: float = 0.04) -> dict[str, object]:
    attempted = 25
    solved = int(round(rate * attempted))
    return {
        "variant_specs": [],
        "variant_attempts": [],
        "variant_attempts_count": attempted,
        "variant_solved_count": solved,
        "generic_transfer_rate_over_variants": rate,
    }


def test_req_arc_wmte_4560_spec_declares_two_metric_artifact_contract() -> None:
    """REQ-ARC-WMTE-4560: OpenSpec anchors the A1/A2/A4 integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4560" in spec
    assert "SCENARIO-ARC-WMTE-4560" in spec
    assert exp4560.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4560.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4560_rejects_landed_null_upstreams() -> None:
    """REQ-ARC-WMTE-4560: current A1/A2/A4 nulls do not become submitted levers."""

    decision = exp4560.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True),
        a2_artifact=_a2_artifact(),
        a4_artifact=_a4_artifact(),
    )

    assert decision["accepted_levers"] == []
    assert decision["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "flagged_adversarial"
    )
    assert decision["rejected_levers"]["A2_executable_world_model_proposer:core_l2"]["reason"] == (
        "no_core_efficiency_gain"
    )
    assert decision["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == (
        "no_new_offline_bank"
    )
    assert decision["isolated_deltas"]["A1_verifier_router"]["generic_transfer_delta"] == 0.0

    permitted = exp4560.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True, permitted_null=True),
        a2_artifact=_a2_artifact(),
        a4_artifact=_a4_artifact(),
    )
    assert permitted["upstream_summaries"]["A1_verifier_router"]["flag_status"] == (
        "permitted_flagged_null"
    )
    assert permitted["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "no_generic_transfer_gain"
    )


def test_req_arc_wmte_4560_selects_only_real_metric_winners() -> None:
    """REQ-ARC-WMTE-4560: accepted levers must raise transfer, CORE L2, or a bank."""

    accepted = exp4560.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, improved=True),
        a2_artifact=_a2_artifact(improved=True),
        a4_artifact=_a4_artifact(banked=True),
    )
    assert accepted["accepted_levers"] == [
        "A1_verifier_router:generic_transfer",
        "A2_executable_world_model_proposer:core_l2",
        "A4_hidden_field_state_probe:new_bank",
    ]

    lost = exp4560.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, improved=True, lost_core=True),
        a2_artifact=_a2_artifact(improved=True, lost_core=True),
        a4_artifact=_a4_artifact(banked=True, lost_core=True),
    )
    assert lost["accepted_levers"] == []
    assert lost["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "core_level_regression"
    )
    assert lost["rejected_levers"]["A2_executable_world_model_proposer:core_l2"]["reason"] == (
        "core_level_regression"
    )
    assert lost["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == (
        "core_level_regression"
    )


def test_scenario_arc_wmte_4560_honest_null_artifact_is_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4560: no winner writes an honest two-metric null."""

    decision = exp4560.select_integrated_levers(
        _a1_artifact(flagged=True),
        _a2_artifact(),
        _a4_artifact(),
    )
    artifact = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        transfer_measurement=_transfer_measurement(),
        random_seed=4560,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_raises_a_metric_honest_null"
    assert artifact["core_efficiency_integrated"] == exp4560.CORE_EFFICIENCY_BASELINE
    assert artifact["generic_transfer_rate_integrated"] == exp4560.GENERIC_TRANSFER_BASELINE
    assert artifact["core_solves_preserved"] is True
    assert artifact["levers_integrated"] == []
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["additivity_checked"]["naive_generic_transfer_delta"] == 0.0
    assert exp4560.artifact_schema_errors(artifact) == []

    out = exp4560.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4560_success_requires_lever_and_integrated_metric_lift() -> None:
    """REQ-ARC-WMTE-4560: submit readiness needs a lever plus an integrated lift."""

    decision = exp4560.select_integrated_levers(
        _a1_artifact(flagged=False, improved=True),
        _a2_artifact(),
        _a4_artifact(),
    )
    artifact = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        transfer_measurement=_transfer_measurement(rate=0.08),
        random_seed=4560,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == (
        "success: integrated_generic_transfer_0.0800_above_0.0400_"
        "or_core_efficiency_2.0074_above_2.0074"
    )
    assert artifact["ready_for_operator_submit"] is True
    assert exp4560.artifact_schema_errors(artifact) == []

    dropped_core = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(solved=["lp85", "sp80", "vc33"]),
        transfer_measurement=_transfer_measurement(rate=0.08),
        random_seed=4560,
        duration_s=1.0,
    )
    assert dropped_core["honest_verdict"] == "complete: no_lever_raises_a_metric_honest_null"
    assert dropped_core["core_solves_preserved"] is False
    assert dropped_core["ready_for_operator_submit"] is False


def test_scenario_arc_wmte_4560_run_writes_injected_measurements(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4560: run writes the 8-game and transfer artifact."""

    artifact = exp4560.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        load_upstream_artifacts=lambda _root: {
            "a1_verifier_router_generic_transfer": _a1_artifact(flagged=True),
            "a2_executable_world_model_proposer": _a2_artifact(),
            "a4_hidden_field_state_probe": _a4_artifact(),
        },
        gate_runner=lambda **_kwargs: _gate_result(),
        transfer_runner=lambda **_kwargs: _transfer_measurement(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["operator_submission_performed"] is False
    assert (
        json.loads((tmp_path / exp4560.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        == artifact
    )


def test_req_arc_wmte_4560_schema_blocked_and_loader_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4560: schema failures and loader fallbacks stay explicit."""

    blocked = exp4560.run(
        write=False,
        preconditions_checked=_preconditions(ok=False),
        baseline=_baseline(),
        gate_runner=lambda **_kwargs: _gate_result(core_efficiency=99.0),
        transfer_runner=lambda **_kwargs: _transfer_measurement(rate=1.0),
        now=lambda: 10.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade_import"
    assert blocked["ready_for_operator_submit"] is False

    valid = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4560.select_integrated_levers(
            _a1_artifact(flagged=True),
            _a2_artifact(),
            _a4_artifact(),
        ),
        gate_result=_gate_result(),
        transfer_measurement=_transfer_measurement(),
        random_seed=4560,
        duration_s=1.0,
    )
    bad = {
        **valid,
        "honest_verdict": "done",
        "inference_substrate": "bad",
        "field_principles": {},
        "core_efficiency_integrated": None,
        "generic_transfer_rate_integrated": None,
        "core_solves_preserved": "yes",
        "levers_integrated": "bad",
        "additivity_checked": None,
        "heldout_solve_rate": None,
        "ready_for_operator_submit": True,
        "false_negative_risk_checked": False,
        "operator_submission_performed": True,
        "reproducibility_checksum": "bad",
        "preconditions_checked": "bad",
    }
    del bad["random_seed"]

    errors = exp4560.artifact_schema_errors(bad)
    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match" in errors
    assert "field_principles must match REQ-ARC-WMTE-4560" in errors
    assert "core_efficiency_integrated must be numeric" in errors
    assert "generic_transfer_rate_integrated must be numeric in [0,1]" in errors
    assert "core_solves_preserved must be bool" in errors
    assert "levers_integrated must be a list" in errors
    assert "additivity_checked must be a mapping" in errors
    assert "heldout_solve_rate must be numeric" in errors
    assert "ready_for_operator_submit cannot be true without success" in errors
    assert "false_negative_risk_checked must be true for complete/success artifacts" in errors
    assert "operator_submission_performed must be false" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "preconditions_checked must be a mapping" in errors

    with pytest.raises(ValueError, match="honest_verdict"):
        exp4560.write_artifact({**valid, "honest_verdict": "bad"}, root=tmp_path)

    assert exp4560._read_json(tmp_path / "missing.json") == {}
    assert exp4560.load_gate_baseline(tmp_path)["core_efficiency"] == (
        exp4560.CORE_EFFICIENCY_BASELINE
    )
    assert exp4560._levels_from_mapping(None) == {}
    assert (
        exp4560._levels_from_nested(
            {"direct": {"deepest_level_by_game": {"lp85": 2}}},
            outer_keys=("missing",),
            direct_keys=("direct",),
        )["lp85"]
        == 2
    )
    assert exp4560._a1_control_levels({}) == {game: 0 for game in exp4560.CORE_GAMES}
    assert exp4560._float_or_none("bad") is None
    assert exp4560._permitted_flagged_null({"flagged_adversarial": False}) is False
    assert exp4560._a1_delta({"generic_transfer_rate_over_variants": 0.08}) == 0.04
    assert exp4560._a2_delta({"core_efficiency_best": None}) == 0.0

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )
    for name, relative in exp4560.UPSTREAM_ARTIFACTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")

    assert exp4560.load_gate_baseline(tmp_path)["core_efficiency"] == (
        exp4560.CORE_EFFICIENCY_BASELINE
    )
    assert sorted(exp4560.load_upstream_artifacts(tmp_path)) == sorted(exp4560.UPSTREAM_ARTIFACTS)

    success = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4560.select_integrated_levers(
            _a1_artifact(flagged=False, improved=True),
            _a2_artifact(),
            _a4_artifact(),
        ),
        gate_result=_gate_result(),
        transfer_measurement=_transfer_measurement(rate=0.08),
        random_seed=4560,
        duration_s=1.0,
    )
    success_mutations = [
        (
            lambda item: item.__setitem__("generic_transfer_rate_integrated", 0.0),
            "success requires core or generic-transfer lift",
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
        assert expected in exp4560.artifact_schema_errors(changed)

    with pytest.raises(ValueError, match="false_negative_risk_checked"):
        exp4560.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            baseline={"core_efficiency": 0.0, "generic_transfer_rate_over_variants": 0.0},
            load_upstream_artifacts=lambda _root: {
                "a1_verifier_router_generic_transfer": _a1_artifact(flagged=True),
                "a2_executable_world_model_proposer": _a2_artifact(),
                "a4_hidden_field_state_probe": _a4_artifact(),
            },
            gate_runner=lambda **_kwargs: _gate_result(),
            transfer_runner=lambda **_kwargs: _transfer_measurement(),
            now=lambda: 10.0,
        )


def test_req_arc_wmte_4560_edge_case_rejections_and_fallback_metrics() -> None:
    """REQ-ARC-WMTE-4560: edge-case guards remain explicit and measured."""

    no_ci = exp4560.select_integrated_levers(
        {**_a1_artifact(flagged=False, improved=True), "generic_transfer_ci": None},
        _a2_artifact(),
        _a4_artifact(),
    )
    assert no_ci["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "generic_transfer_ci_missing"
    )

    not_preserved = exp4560.select_integrated_levers(
        {**_a1_artifact(flagged=False, improved=True), "solve_rate_preserved": False},
        _a2_artifact(),
        _a4_artifact(),
    )
    assert not_preserved["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "solve_rate_not_preserved"
    )

    no_reproduction = exp4560.select_integrated_levers(
        {**_a1_artifact(flagged=False, improved=True), "offline_reproduced": False},
        _a2_artifact(),
        _a4_artifact(),
    )
    assert no_reproduction["rejected_levers"]["A1_verifier_router:generic_transfer"]["reason"] == (
        "offline_reproduction_missing"
    )

    a2_edges = [
        (
            _a2_artifact(improved=True, preserved=False),
            "core_solves_not_preserved",
        ),
        (
            {
                **_a2_artifact(improved=True),
                "deepest_level_reached_per_core_game": {
                    "offline_dsl_baseline": _levels(),
                    "executable_proposer": _levels(),
                },
            },
            "no_core_l2_reached",
        ),
        (
            {**_a2_artifact(improved=True), "offline_reproduced": False},
            "offline_reproduction_missing",
        ),
        (
            {**_a2_artifact(improved=True), "positive_control_passed": False},
            "positive_control_failed",
        ),
    ]
    for a2_artifact, reason in a2_edges:
        decision = exp4560.select_integrated_levers(
            _a1_artifact(flagged=True),
            a2_artifact,
            _a4_artifact(),
        )
        assert (
            decision["rejected_levers"]["A2_executable_world_model_proposer:core_l2"]["reason"]
            == reason
        )

    a4_edges = [
        (
            {**_a4_artifact(banked=True), "core_solves_preserved": False},
            "core_solves_not_preserved",
        ),
        ({**_a4_artifact(banked=True), "registry_updated": False}, "registry_update_missing"),
    ]
    for a4_artifact, reason in a4_edges:
        decision = exp4560.select_integrated_levers(
            _a1_artifact(flagged=True),
            _a2_artifact(),
            a4_artifact,
        )
        assert (
            decision["rejected_levers"]["A4_hidden_field_state_probe:new_bank"]["reason"] == reason
        )

    fallback_gate = {
        "current": {
            "efficiency_by_game": {game: 0.5 for game in exp4560.CORE_GAMES},
            "per_game": [
                {"game": "lp85", "solved": True, "levels": None},
                {"game": None, "solved": False},
            ],
        },
        "baseline_guard": {"ok": True},
    }
    artifact = exp4560.build_artifact(
        preconditions_checked=_preconditions(),
        baseline={"core_efficiency": 2.0074, "generic_transfer_rate_over_variants": 0.04},
        upstream_decision=exp4560.select_integrated_levers(
            _a1_artifact(flagged=True),
            _a2_artifact(),
            _a4_artifact(),
        ),
        gate_result=fallback_gate,
        transfer_measurement={"variant_attempts_count": 25, "variant_solved_count": 1},
        random_seed=4560,
        duration_s=1.0,
    )
    assert artifact["core_efficiency_integrated"] == 2.0
    assert artifact["generic_transfer_rate_integrated"] == 0.04
    assert artifact["per_game_deepest_level_reached"]["lp85"] == 1
    assert exp4560._core_efficiency({}) == 0.0
