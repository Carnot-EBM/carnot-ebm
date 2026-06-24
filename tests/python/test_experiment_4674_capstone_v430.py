"""Tests for Exp 4674 .430 capstone scorecard.

Spec refs: REQ-CAPSTONE-4674, SCENARIO-CAPSTONE-4674,
SCENARIO-CAPSTONE-4674-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4674_capstone_v430 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_l2_goal(
    *,
    satisfiable: bool = True,
    reaches_goal: bool = False,
    offline: bool = False,
    fixed_harness: bool = True,
    deepest_level: int = 1,
    games: list[str] | None = None,
) -> dict[str, Any]:
    reached_games = games if games is not None else (["lp85"] if deepest_level >= 2 else [])
    return {
        "experiment": "experiment_4664_l2_goal_predicate_induction_live",
        "honest_verdict": (
            "success: l2_goal_induction_generic_agent_reached_l2_lp85"
            if deepest_level >= 2 and reaches_goal
            else "complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient"
        ),
        "verifier_is_oracle": False,
        "goal_predicate_satisfiable": satisfiable,
        "l2_plan_reaches_goal": reaches_goal,
        "offline_reproduced": offline,
        "generic_agent_deepest_level": deepest_level,
        "generic_agent_l2_games": reached_games,
        "metric_harness": {
            "fixed": fixed_harness,
            "break_at_first_win": False,
            "target_levels": [1, 2],
            "degenerate_0_by_construction": not fixed_harness,
        },
        "duration_s": 254.37307,
        "inference_substrate": "live_llm_inference",
    }


def _a2_value_routing(
    *,
    first_win_delta: float = 0.0,
    solve_delta: float = 0.0,
    ci95: tuple[float, float] = (0.0, 0.0),
    excludes_baseline: bool = False,
    shift_before: float = 0.699108,
    shift_after: float = 0.0,
) -> dict[str, Any]:
    selected = max(first_win_delta, solve_delta)
    return {
        "experiment": "experiment_4665_dagger_distribution_shift_value_routing",
        "honest_verdict": (
            "success: dagger_distribution_corrected_live_firstwin_up_0.08"
            if selected > 0.0
            else "complete: dagger_distribution_corrected_no_live_lift_residual_logged."
        ),
        "verifier_is_oracle": False,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "first_win_rate_delta": first_win_delta,
        "solve_rate_delta": solve_delta,
        "live_first_win_rate_corrected": 0.04 + first_win_delta,
        "live_solve_rate_corrected": solve_delta,
        "winning_path_baseline_429": {"first_win_rate": 0.04, "solve_rate": 0.0},
        "live_lift_ci": {"ci95": list(ci95), "point": selected, "baseline": ".429_winning_path"},
        "ci_excludes_429_winning_path_baseline": excludes_baseline,
        "distribution_shift_score_before": shift_before,
        "distribution_shift_score_after": shift_after,
        "shift_score_delta": shift_after - shift_before,
        "offline_reproduced": True,
        "duration_s": 431.675612,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4666_levelup_selfplay",
        "honest_verdict": "success: dc22_L2_offline_reproduced",
        "verifier_is_oracle": False,
        "levels_reproduced": 1,
        "offline_reproduced": True,
        "reproduction_gate": {"game": "dc22", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4667_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_59_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 59,
        "live_submittable_count_prev": 58,
        "count_delta": 1,
        "levels_folded_in": ["dc22"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4668_primitive_persist_transfer",
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "l2_goal_predicate_operator"},
        "transfer_value_per_game": {"sc25": {"value_added": False}},
    }


def _a6_integration(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4669_integration_gate",
        "honest_verdict": "complete: integration_unchanged_both_levers_null",
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged,
        "live_first_win_rate_integrated": 0.04,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_multi_level_solve_rate_pre_integration": 0.0,
        "live_multi_level_solve_rate_delta_vs_pre_integration": 0.0,
    }


def _b1_harness() -> dict[str, Any]:
    return {
        "experiment": "experiment_4670_multilevel_harness_cigate",
        "honest_verdict": "success: multilevel_harness_cigate_plus_port_hygiene_shipped_tests_green",
        "verifier_is_oracle": False,
        "multilevel_metric_harness_fixed": True,
        "break_at_first_win": False,
        "target_levels": [1, 2],
        "tests_added": {"passed": True},
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4671_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_l2_goal_and_multilevel_metric_guards_tests_green.",
        "l2_goal_satisfiability_guard_added": True,
        "multilevel_nondegenerate_metric_guard_added": True,
        "tests_added": {"passed": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1_satisfiable: bool = True,
    a1_reaches_goal: bool = False,
    a1_offline: bool = False,
    a1_fixed_harness: bool = True,
    a1_deepest_level: int = 1,
    a2_first_win_delta: float = 0.0,
    a2_solve_delta: float = 0.0,
    a2_ci95: tuple[float, float] = (0.0, 0.0),
    a2_excludes_baseline: bool = False,
    a2_shift_before: float = 0.699108,
    a2_shift_after: float = 0.0,
    a6_flagged: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_l2_goal(
            satisfiable=a1_satisfiable,
            reaches_goal=a1_reaches_goal,
            offline=a1_offline,
            fixed_harness=a1_fixed_harness,
            deepest_level=a1_deepest_level,
        ),
        "A2": _a2_value_routing(
            first_win_delta=a2_first_win_delta,
            solve_delta=a2_solve_delta,
            ci95=a2_ci95,
            excludes_baseline=a2_excludes_baseline,
            shift_before=a2_shift_before,
            shift_after=a2_shift_after,
        ),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(flagged=a6_flagged),
        "B1": _b1_harness(),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 59) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4674": True,
        "registry_yaml_loadable": True,
        "registry_reproducible_total_levels": total,
        "summarize_artifact_py_available": True,
        "summarize_artifact_py_used_for_every_upstream": True,
        "upstream_artifacts_present": {name: True for name in mod.UPSTREAM_SOURCES},
        "missing_upstream_artifacts": [],
        "leaderboard_submission": False,
        "operator_only": True,
        "research_conductor_modified": False,
    }


def _paper_gate(ready: bool = True) -> dict[str, Any]:
    return {
        "paper_ready": ready,
        "frozen_fover_auroc": 0.9131,
        "gates": {
            "G1": {"pass": ready, "detail": "FoVer dual-condition AUROC artifact present"},
            "G2": {"pass": ready, "detail": "independent reproducer confirmed"},
            "G3": {"pass": ready, "detail": "narrowing-clean"},
            "G4": {"pass": ready, "detail": "numbers trace to artifact"},
        },
        "unmet_gates": [] if ready else ["G2"],
    }


def test_req_capstone_4674_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4674: OpenSpec declares the .430 scorecard fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4674" in spec
    assert "SCENARIO-CAPSTONE-4674" in spec
    assert "SCENARIO-CAPSTONE-4674-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4674_default_null_bridge_counts_registry_growth() -> None:
    """SCENARIO-CAPSTONE-4674: default .430 excludes null/flagged levers from headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_58_to_59"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_generic_agent_reached_l2"]["headline_counted"] is False
    assert artifact["a1_generic_agent_reached_l2"]["fixed_metric_harness"] is True
    assert artifact["a2_value_routing_live_lift"]["distribution_shift_dropped"] is True
    assert artifact["a2_value_routing_live_lift"]["ci_excludes_429_winning_path_baseline"] is False
    assert artifact["a2_value_routing_live_lift"]["headline_counted"] is False
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["reproducible_total_levels"] == 59
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["live_submittable_level_count"] == 59
    assert artifact["paper_ready"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4674_a1_success_requires_controls_and_fixed_harness() -> None:
    """SCENARIO-CAPSTONE-4674: A1 L2 counts only with satisfiable goal and fixed harness."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a1_reaches_goal=True,
            a1_offline=True,
            a1_deepest_level=2,
            a2_excludes_baseline=True,
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=58),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_generic_L2_lp85"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a1_generic_agent_reached_l2"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A1_l2_goal_induction"

    blocked_by_harness = mod.build_artifact(
        artifacts=_artifacts(
            a1_reaches_goal=True,
            a1_offline=True,
            a1_fixed_harness=False,
            a1_deepest_level=2,
            a2_excludes_baseline=True,
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=58),
        duration_s=0.001,
    )

    assert blocked_by_harness["honest_verdict"] == "complete: multi_level_deepening_levers_characterized_no_live_L2"
    assert blocked_by_harness["a1_generic_agent_reached_l2"]["fixed_metric_harness"] is False
    assert blocked_by_harness["flagged_artifacts_handled"]["control_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4664_l2_goal_predicate_induction_live.json"}
    ]


def test_scenario_capstone_4674_a2_success_requires_ci_and_shift_drop() -> None:
    """SCENARIO-CAPSTONE-4674: A2 value routing counts only with CI and shift improvement."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a2_first_win_delta=0.08,
            a2_ci95=(0.03, 0.12),
            a2_excludes_baseline=True,
            a2_shift_before=0.7,
            a2_shift_after=0.2,
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=58),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_value_routing_firstwin_up_0.08"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a2_value_routing_live_lift"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_distribution_shift_value_routing"

    blocked_by_shift = mod.build_artifact(
        artifacts=_artifacts(
            a2_first_win_delta=0.08,
            a2_ci95=(0.03, 0.12),
            a2_excludes_baseline=True,
            a2_shift_before=0.2,
            a2_shift_after=0.7,
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=58),
        duration_s=0.001,
    )

    assert blocked_by_shift["bridge_crossed_for_solve"] is False
    assert blocked_by_shift["a2_value_routing_live_lift"]["reason"] == "distribution_shift_not_reduced"


def test_req_capstone_4674_exclusion_guards_cover_vacuous_control_and_live_flags() -> None:
    """REQ-CAPSTONE-4674: flagged, vacuous, control-failed, and FNR-open inputs are excluded."""

    artifacts = _artifacts(a1_satisfiable=False, a2_excludes_baseline=True, a6_flagged=False)
    artifacts["A4"]["false_negative_risk_checked"] = False
    artifacts["A5"]["acceptance_gate_transfer"] = False

    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={
            "A3": [
                {
                    "kind": "FALSE_NEGATIVE_RISK",
                    "severity": "warn",
                    "detail": "false_negative_risk_open: bank control absent",
                }
            ],
            "A6": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "live_first_win_rate_integrated=0.04 and pre=0.04",
                }
            ],
        },
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A1"]["reason"] == "vacuous_goal"
    assert artifact["cited_upstream_artifacts"]["A3"]["reason"] == "false_negative_risk_open"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "positive_control_failed"
    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".430-B2 L2-goal-satisfiability"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".430-B2 multi-level-metric"] is True


def test_req_capstone_4674_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4674: missing upstreams block without fabricated headline metrics."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 59\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4674\n", encoding="utf-8")
    scripts_path = tmp_path / "scripts" / "summarize_artifact.py"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text("# test\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, publication_gate=_paper_gate(), write=True, duration_s=0.001)

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4668_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert artifact["bridge_crossed_for_solve"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4674_validation_and_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4674: schema validation, checksum, and defensive helpers fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(a2_excludes_baseline=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(False),
        preconditions_checked={**_preconditions(), "ok": False, "blocked_resource": "registry_yaml"},
        duration_s=0.001,
    )
    bad = dict(blocked)
    bad["verifier_is_oracle"] = True

    def bad_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"honest_verdict": "complete: invalid"}

    assert blocked["honest_verdict"] == "blocked_registry_yaml"
    assert blocked["paper_ready"] is False
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    assert mod._as_float(True, 3.0) == pytest.approx(3.0)
    assert mod._as_int(False, 4) == 4
    assert mod._file_sha256(tmp_path / "missing.json") is None
    assert mod._positive_control_failed({"positive_control_passed": False}) is True
    assert mod._positive_control_failed({"bare_control_passed": False}) is True
    assert mod._ci_excludes_zero({"ci95": "bad"}) is False
    assert mod._fixed_multilevel_metric_harness({"fixed_metric_harness": True}) is True
    assert mod._fixed_multilevel_metric_harness({"metric_harness": {"fixed": False}}) is False
    assert mod._a1_generic_agent_reached_l2(
        _a1_l2_goal(satisfiable=False, reaches_goal=True, offline=True, deepest_level=2),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "goal_predicate_not_satisfiable"
    assert mod._a1_generic_agent_reached_l2(
        _a1_l2_goal(reaches_goal=False, offline=True, deepest_level=2),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "l2_plan_does_not_reach_goal"
    assert mod._a1_generic_agent_reached_l2(
        _a1_l2_goal(reaches_goal=True, offline=False, deepest_level=2),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "offline_reproduction_missing"
    assert mod._a1_generic_agent_reached_l2(
        _a1_l2_goal(reaches_goal=True, offline=True, fixed_harness=False, deepest_level=2),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "fixed_multilevel_metric_harness_missing"
    assert mod._a1_generic_agent_reached_l2(
        _a1_l2_goal(reaches_goal=True, offline=True, deepest_level=1),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "generic_agent_did_not_reach_l2"
    assert mod._a2_value_routing_live_lift(
        _a2_value_routing(first_win_delta=0.08, ci95=(0.01, 0.02), excludes_baseline=True),
        {"included_in_headline": False, "reason": "manual_excluded"},
    )["reason"] == "manual_excluded"
    assert mod._a2_value_routing_live_lift(
        _a2_value_routing(first_win_delta=0.0, ci95=(0.01, 0.02), excludes_baseline=True),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "no_positive_live_lift"
    monkeypatch.setattr(
        mod,
        "publication_gate_reader",
        type("PublicationGateFixture", (), {"evaluate": staticmethod(lambda: _paper_gate())}),
    )
    assert mod._load_publication_gate()["paper_ready"] is True
    monkeypatch.setattr(
        mod,
        "_summarize_and_live_flags",
        lambda _path: (
            0,
            [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "fixture"}],
        ),
    )
    status = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "results/existing.json", "fixture"),
        root=tmp_path,
        artifact={"honest_verdict": "complete: fixture"},
        exists=True,
        live_flags_by_name=None,
    )
    missing_status = mod._source_status(
        name="X",
        source=mod.SourceSpec("X", "results/missing.json", "fixture"),
        root=tmp_path,
        artifact={},
        exists=False,
        live_flags_by_name=None,
    )
    assert status["summary_exit_code"] == 0
    assert status["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert missing_status["reason"] == "missing"
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)
    malformed = dict(blocked)
    malformed["honest_verdict"] = "still_running"
    malformed["reproducibility_checksum"] = mod.payload_checksum(malformed)
    assert "honest_verdict must be terminal-prefixed" in mod.validate_artifact(malformed)
    monkeypatch.setattr(mod, "build_artifact", bad_build)
    with pytest.raises(ValueError, match="missing required field"):
        mod.run(tmp_path, write=False, duration_s=0.001)
