"""Tests for Exp 4548 submitted A1/A4 integration.

Spec refs: REQ-ARC-WMTE-4548, SCENARIO-ARC-WMTE-4548.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4548_integration_8game_gate as exp4548


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _preconditions(ok: bool = True) -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": ok,
        "baseline_file_present": True,
        "a1_artifact_present": True,
        "a4_artifact_present": True,
        "spec_has_req_4548": True,
        "ok": ok,
    }


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4548.GATE_GAMES),
        "solved_games": list(exp4548.CORE_GAMES),
        "efficiency_by_game": {
            "lp85": 2.0069,
            "m0r0": 0.0003,
            "sp80": 0.0001,
            "vc33": 0.0001,
        },
        "core_efficiency": exp4548.CORE_EFFICIENCY_BASELINE,
        "median_actions_on_solved": 7760.0,
    }


def _levels(lp85: int = 1, m0r0: int = 1, sp80: int = 1, vc33: int = 1) -> dict[str, int]:
    return {"lp85": lp85, "m0r0": m0r0, "sp80": sp80, "vc33": vc33}


def _a1_artifact(
    *,
    flagged: bool = True,
    permitted_null: bool = False,
    improved: bool = False,
    lost_core: bool = False,
    offline_reproduced: bool = False,
    positive_control_passed: bool = True,
) -> dict[str, object]:
    control = _levels()
    treatment = _levels(lp85=2 if improved else 1, m0r0=0 if lost_core else 1)
    best = 3.25 if improved else exp4548.CORE_EFFICIENCY_BASELINE
    artifact: dict[str, object] = {
        "honest_verdict": "complete: llm_proposer_no_deeper_level_honest_null",
        "flagged_adversarial": flagged,
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "severity": "critical"},
            {"kind": "IMPLAUSIBLE_PERFECT", "severity": "info"},
        ],
        "core_efficiency_baseline": exp4548.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": round(best - exp4548.CORE_EFFICIENCY_BASELINE, 4),
        "core_solves_preserved": not lost_core,
        "offline_reproduced": offline_reproduced,
        "positive_control_passed": positive_control_passed,
        "deepest_level_reached_per_core_game": {
            "offline_dsl_baseline": control,
            "llm_proposer": treatment,
        },
        "chosen_submitted_config": "llm_proposer_reinduction_target_levels_2"
        if improved
        else "unchanged",
    }
    if permitted_null:
        artifact["corrigendum_pending"] = [{"kind": "TAUTOLOGY", "severity": "critical"}]
        artifact["core_efficiency_best"] = exp4548.CORE_EFFICIENCY_BASELINE
        artifact["efficiency_delta"] = 0.0
        artifact["null_delta_methodology_note"] = (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level."
        )
        artifact["deepest_level_reached_per_core_game"] = {
            "offline_dsl_baseline": control,
            "llm_proposer": control,
        }
    return artifact


def _a4_artifact(
    *,
    improved: bool = False,
    lost_core: bool = False,
    core_evidence: bool = False,
    core_solves_preserved: bool = True,
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "honest_verdict": "complete: frame_change_cnn_no_action_reduction_honest_null",
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "median_actions_to_first_levelup_blind": 1.0,
        "median_actions_to_first_levelup_cnn": 1.0,
        "solve_rate_preserved": True,
        "ranking_metrics": {
            "action_reduction": False,
            "solve_rate_preserved": True,
        },
    }
    if core_evidence:
        control = _levels(lp85=0, m0r0=1, sp80=1, vc33=1)
        treatment = _levels(lp85=1 if improved else 0, m0r0=0 if lost_core else 1)
        efficiency = 2.5 if improved else exp4548.CORE_EFFICIENCY_BASELINE
        artifact.update(
            {
                "core_efficiency_baseline": exp4548.CORE_EFFICIENCY_BASELINE,
                "core_efficiency_cnn": efficiency,
                "core_solves_preserved": core_solves_preserved and not lost_core,
                "blind_bfs_control": {
                    "core_efficiency": exp4548.CORE_EFFICIENCY_BASELINE,
                    "deepest_level_by_game": control,
                },
                "frame_change_ranker_measurement": {
                    "core_efficiency": efficiency,
                    "deepest_level_by_game": treatment,
                },
                "deepest_level_reached_per_core_game": {
                    "blind_bfs": control,
                    "cnn_ranker": treatment,
                },
            }
        )
    return artifact


def _gate_result(*, core_efficiency: float = 2.0074, solved: list[str] | None = None) -> dict[str, object]:
    solved_games = solved or list(exp4548.CORE_GAMES)
    rows = [
        {
            "game": game,
            "solved": game in solved_games,
            "actions": 10 if game in solved_games else 20,
            "efficiency": (core_efficiency / len(exp4548.CORE_GAMES)) if game in exp4548.CORE_GAMES else 0.0,
            "timed_out": False,
            "levels": 1 if game in solved_games else 0,
        }
        for game in exp4548.GATE_GAMES
    ]
    return {
        "pass": core_efficiency >= exp4548.CORE_EFFICIENCY_BASELINE,
        "verdict": "PASS fixture",
        "baseline_guard": {"ok": True},
        "current": {
            "policy": "e3",
            "games": list(exp4548.GATE_GAMES),
            "per_game": rows,
            "solved_games": solved_games,
            "solved_count": len(solved_games),
            "efficiency_by_game": {
                game: row["efficiency"] for game, row in zip(exp4548.GATE_GAMES, rows)
            },
            "core_efficiency": core_efficiency,
            "median_actions_on_core": 10.0,
            "median_actions_on_solved": 10.0,
            "timed_out_count": 0,
        },
    }


def test_req_arc_wmte_4548_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4548: OpenSpec anchors the A1/A4 integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4548" in spec
    assert "SCENARIO-ARC-WMTE-4548" in spec
    assert exp4548.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4548.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_wmte_4548_rejects_multi_flagged_a1_and_a4_without_core_evidence() -> None:
    """REQ-ARC-WMTE-4548: flagged A1 artifacts cannot be aggregated as wins."""

    decision = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True),
        a4_artifact=_a4_artifact(),
    )

    assert decision["accepted_levers"] == []
    assert decision["upstream_summaries"]["A1_llm_proposer_reinduction"]["flag_status"] == (
        "rejected_flagged_adversarial"
    )
    assert decision["rejected_levers"]["A1_llm_proposer_reinduction"]["reason"] == (
        "flagged_adversarial"
    )
    assert decision["rejected_levers"]["A4_frame_change_cnn_ranker:cnn_ranker"]["reason"] == (
        "no_core_efficiency_evidence"
    )
    assert decision["isolated_deltas"] == {
        "A1_llm_proposer_reinduction": 0.0,
        "A4_frame_change_cnn_ranker": 0.0,
    }

    permitted = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=True, permitted_null=True),
        a4_artifact=_a4_artifact(core_evidence=True),
    )
    assert permitted["upstream_summaries"]["A1_llm_proposer_reinduction"]["flag_status"] == (
        "permitted_flagged_null"
    )
    assert permitted["rejected_levers"]["A1_llm_proposer_reinduction:llm_proposer"]["reason"] == (
        "no_core_efficiency_gain"
    )
    assert permitted["rejected_levers"]["A4_frame_change_cnn_ranker:cnn_ranker"]["reason"] == (
        "no_core_efficiency_gain"
    )


def test_req_arc_wmte_4548_selects_only_core_efficiency_winners() -> None:
    """REQ-ARC-WMTE-4548: accepted levers must raise CORE efficiency and preserve levels."""

    accepted = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            improved=True,
            offline_reproduced=True,
            positive_control_passed=True,
        ),
        a4_artifact=_a4_artifact(improved=True, core_evidence=True),
    )
    assert accepted["accepted_levers"] == [
        "A1_llm_proposer_reinduction:llm_proposer",
        "A4_frame_change_cnn_ranker:cnn_ranker",
    ]

    lost = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            improved=True,
            lost_core=True,
            offline_reproduced=True,
        ),
        a4_artifact=_a4_artifact(improved=True, lost_core=True, core_evidence=True),
    )
    assert lost["accepted_levers"] == []
    assert lost["rejected_levers"]["A1_llm_proposer_reinduction:llm_proposer"]["reason"] == (
        "core_level_regression"
    )
    assert lost["rejected_levers"]["A4_frame_change_cnn_ranker:cnn_ranker"]["reason"] == (
        "core_level_regression"
    )

    unreproduced = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, improved=True, offline_reproduced=False),
        a4_artifact=_a4_artifact(core_evidence=True),
    )
    assert unreproduced["rejected_levers"]["A1_llm_proposer_reinduction:llm_proposer"]["reason"] == (
        "offline_reproduction_missing"
    )

    positive_failed = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(
            flagged=False,
            improved=True,
            offline_reproduced=True,
            positive_control_passed=False,
        ),
        a4_artifact=_a4_artifact(core_evidence=True),
    )
    assert positive_failed["rejected_levers"]["A1_llm_proposer_reinduction:llm_proposer"]["reason"] == (
        "positive_control_failed"
    )

    a4_not_preserved = exp4548.select_integrated_levers(
        a1_artifact=_a1_artifact(flagged=False, permitted_null=True),
        a4_artifact=_a4_artifact(improved=True, core_evidence=True, core_solves_preserved=False),
    )
    assert a4_not_preserved["rejected_levers"]["A4_frame_change_cnn_ranker:cnn_ranker"]["reason"] == (
        "core_solves_not_preserved"
    )


def test_scenario_arc_wmte_4548_honest_null_artifact_is_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4548: no winner writes an honest null with measured efficiency."""

    decision = exp4548.select_integrated_levers(_a1_artifact(flagged=True), _a4_artifact())
    artifact = exp4548.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(),
        random_seed=4548,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert artifact["core_efficiency_integrated"] == exp4548.CORE_EFFICIENCY_BASELINE
    assert artifact["core_solves_preserved"] is True
    assert artifact["levers_integrated"] == []
    assert artifact["heldout_solve_rate"] == 0.0
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["additivity_checked"]["naive_sum_delta"] == 0.0
    assert exp4548.artifact_schema_errors(artifact) == []

    out = exp4548.write_artifact(artifact, root=tmp_path)
    assert json.loads(out.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4548_success_requires_integrated_gain_and_core_preservation() -> None:
    """REQ-ARC-WMTE-4548: submit readiness needs accepted levers and a gate lift."""

    decision = exp4548.select_integrated_levers(
        _a1_artifact(
            flagged=False,
            improved=True,
            offline_reproduced=True,
            positive_control_passed=True,
        ),
        _a4_artifact(),
    )
    artifact = exp4548.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4548,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "success: integrated_core_efficiency_3.2500_above_2.0074"
    assert artifact["ready_for_operator_submit"] is True
    assert exp4548.artifact_schema_errors(artifact) == []

    dropped = exp4548.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        gate_result=_gate_result(core_efficiency=3.25, solved=["lp85", "sp80", "vc33"]),
        random_seed=4548,
        duration_s=1.0,
    )
    assert dropped["honest_verdict"] == "complete: no_lever_raises_core_efficiency_honest_null"
    assert dropped["core_solves_preserved"] is False
    assert dropped["ready_for_operator_submit"] is False


def test_scenario_arc_wmte_4548_run_writes_injected_gate_measurement(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4548: run writes the 8-game per-level gate artifact."""

    artifact = exp4548.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        load_upstream_artifacts=lambda _root: {
            "a1_llm_proposer_reinduction": _a1_artifact(flagged=True),
            "a4_frame_change_cnn_ranker": _a4_artifact(),
        },
        gate_runner=lambda **_kwargs: _gate_result(),
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["result_path"] == exp4548.RESULT_RELATIVE_PATH
    assert json.loads((tmp_path / exp4548.RESULT_RELATIVE_PATH).read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_4548_schema_blocked_and_loader_paths(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4548: schema failures and loader fallbacks stay explicit."""

    blocked = exp4548.run(
        write=False,
        preconditions_checked=_preconditions(ok=False),
        baseline=_baseline(),
        gate_runner=lambda **_kwargs: _gate_result(core_efficiency=99.0),
        now=lambda: 10.0,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade_import"
    assert blocked["levers_integrated"] == []
    assert blocked["ready_for_operator_submit"] is False

    valid = exp4548.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4548.select_integrated_levers(_a1_artifact(flagged=True), _a4_artifact()),
        gate_result=_gate_result(),
        random_seed=4548,
        duration_s=1.0,
    )
    bad = {
        **valid,
        "honest_verdict": "done",
        "inference_substrate": "bad",
        "field_principles": {},
        "core_efficiency_baseline": 0.0,
        "core_efficiency_integrated": None,
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

    errors = exp4548.artifact_schema_errors(bad)
    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match" in errors
    assert "field_principles must match REQ-ARC-WMTE-4548" in errors
    assert "core_efficiency_baseline must equal 2.0074" in errors
    assert "core_efficiency_integrated must be numeric" in errors
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
        exp4548.write_artifact({**valid, "honest_verdict": "bad"}, root=tmp_path)

    success = exp4548.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=exp4548.select_integrated_levers(
            _a1_artifact(flagged=False, improved=True, offline_reproduced=True),
            _a4_artifact(),
        ),
        gate_result=_gate_result(core_efficiency=3.25),
        random_seed=4548,
        duration_s=1.0,
    )
    success_mutations = [
        (
            lambda item: item.__setitem__("core_efficiency_integrated", 1.0),
            "success requires core_efficiency_integrated above baseline",
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
        assert expected in exp4548.artifact_schema_errors(changed)

    with pytest.raises(ValueError, match="false_negative_risk_checked"):
        exp4548.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            baseline={"core_efficiency": 0.0},
            load_upstream_artifacts=lambda _root: {
                "a1_llm_proposer_reinduction": _a1_artifact(flagged=True),
                "a4_frame_change_cnn_ranker": _a4_artifact(),
            },
            gate_runner=lambda **_kwargs: _gate_result(),
            now=lambda: 10.0,
        )

    assert exp4548._read_json(tmp_path / "missing.json") == {}
    assert exp4548._levels_from_mapping(None) == {}
    assert exp4548._permitted_flagged_null({"flagged_adversarial": False}) is False
    assert exp4548._a1_delta({"core_efficiency_best": 2.5}) == 0.4926
    assert exp4548._a4_delta({"core_efficiency_cnn": 2.5}) == 0.4926
    assert exp4548._a1_control_levels({}) == {game: 0 for game in exp4548.CORE_GAMES}
    assert exp4548._a1_treatment_levels({}) == {game: 0 for game in exp4548.CORE_GAMES}
    assert exp4548._a4_control_levels({}) == {game: 0 for game in exp4548.CORE_GAMES}
    assert exp4548._a4_treatment_levels({}) == {game: 0 for game in exp4548.CORE_GAMES}
    assert exp4548.load_gate_baseline(tmp_path / "missing")["core_efficiency"] == (
        exp4548.CORE_EFFICIENCY_BASELINE
    )

    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )
    for name, relative in exp4548.UPSTREAM_ARTIFACTS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}), encoding="utf-8")

    assert exp4548.load_gate_baseline(tmp_path)["core_efficiency"] == exp4548.CORE_EFFICIENCY_BASELINE
    assert sorted(exp4548.load_upstream_artifacts(tmp_path)) == sorted(exp4548.UPSTREAM_ARTIFACTS)
