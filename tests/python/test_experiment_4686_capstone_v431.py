"""Tests for Exp 4686 .431 capstone scorecard.

Spec refs: REQ-CAPSTONE-4686, SCENARIO-CAPSTONE-4686,
SCENARIO-CAPSTONE-4686-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4686-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4686_capstone_v431 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_subgoal(
    *,
    reached_level: int = 1,
    offline: bool = False,
    no_subgoal_level: int = 1,
    random_subgoal_level: int = 1,
    decomposition: list[dict[str, Any]] | None = None,
    reachable: list[dict[str, Any]] | None = None,
    verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4676_hierarchical_subgoal_search_live",
        "honest_verdict": verdict
        or (
            f"success: hierarchical_subgoal_generic_agent_new_level_lp85_L{reached_level}"
            if offline and reached_level >= 2
            else "complete: hierarchical_subgoal_no_new_level_residual_value_head_still_not_separating"
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "target_game": "lp85",
        "subgoal_decomposition": decomposition
        if decomposition is not None
        else [{"name": "bridge", "reachable": True}, {"name": "final_goal", "reachable": True}],
        "per_subgoal_reachable": reachable
        if reachable is not None
        else [{"name": "bridge", "reachable": True}, {"name": "final_goal", "reachable": True}],
        "generic_agent_reached_level": reached_level,
        "offline_reproduced": offline,
        "reproduced_levels": reached_level if offline else 0,
        "no_subgoal_ablation_reached_level": no_subgoal_level,
        "random_subgoal_ablation_reached_level": random_subgoal_level,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "duration_s": 3442.045397,
        "inference_substrate": "live_llm_inference",
    }


def _a2_factored(
    *,
    factored_coverage: float = 0.0,
    flat_coverage: float | None = 0.0,
    first_win_delta: float = -0.04,
    solve_delta: float = 0.0,
    ci: Mapping[str, Any] | None = None,
    offline: bool = False,
) -> dict[str, Any]:
    baseline = {"first_win_rate": 0.04, "solve_rate": 0.0}
    payload: dict[str, Any] = {
        "experiment": "experiment_4677_poe_world_factored_subgoal_planner",
        "honest_verdict": (
            "success: poe_world_factored_planner_coverage_up_live_firstwin_lift_lp85"
            if factored_coverage > 0.0 and first_win_delta > 0.0 and offline
            else "complete: poe_world_factored_planner_no_coverage_gain_residual_logged"
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "target_games": ["lp85"],
        "candidate_generation_coverage_factored": factored_coverage,
        "coverage_delta": factored_coverage - (flat_coverage if flat_coverage is not None else 0.0),
        "live_first_win_rate_factored": baseline["first_win_rate"] + first_win_delta,
        "live_solve_rate_factored": baseline["solve_rate"] + solve_delta,
        "live_baseline_flat_search": baseline,
        "first_win_rate_delta": first_win_delta,
        "solve_rate_delta": solve_delta,
        "live_lift_ci": dict(ci or {"metric": "first_win_rate_delta", "low": 0.0, "high": 0.0}),
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "offline_reproduced": offline,
        "duration_s": 60.000087,
        "inference_substrate": "live_llm_inference",
    }
    if flat_coverage is not None:
        payload["candidate_generation_coverage_flat_baseline"] = flat_coverage
    return payload


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4678_levelup_selfplay",
        "honest_verdict": "success: sb26_L2_offline_reproduced",
        "verifier_is_oracle": False,
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "target_game": "sb26",
        "reproduction_gate": {"game": "sb26", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4679_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_60_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 60,
        "live_submittable_count_prev": 59,
        "count_delta": 1,
        "levels_folded_in": ["sb26"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4680_primitive_persist_transfer",
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "programmatic_expert_trust_weighting_operator"},
        "offline_reproduced_new_level": False,
    }


def _a6_integration(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4681_integration_gate",
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


def _b1_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4682_generation_coverage_cigate",
        "honest_verdict": "success: generation_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green",
        "verifier_is_oracle": False,
        "coverage_metric_added": True,
        "honest_firstwin_floor_added": True,
        "coverage_floor_cigate_added": True,
        "tests_added": {"passed": True},
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4683_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_subgoal_decomposition_and_coverage_baseline_guards_tests_green.",
        "verifier_is_oracle": False,
        "subgoal_decomposition_guard_added": True,
        "coverage_baseline_guard_added": True,
        "tests_added": {"passed": True},
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


def _artifacts(
    *,
    a1: Mapping[str, Any] | None = None,
    a2: Mapping[str, Any] | None = None,
    a6_flagged: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": dict(a1 or _a1_subgoal()),
        "A2": dict(a2 or _a2_factored()),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(flagged=a6_flagged),
        "B1": _b1_guard(),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 60) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4686": True,
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


def test_req_capstone_4686_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4686: OpenSpec declares the .431 scorecard fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4686" in spec
    assert "SCENARIO-CAPSTONE-4686" in spec
    assert "SCENARIO-CAPSTONE-4686-BLOCKED-PRECONDITION" in spec
    assert "SCENARIO-CAPSTONE-4686-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4686_default_null_bridge_counts_registry_growth() -> None:
    """SCENARIO-CAPSTONE-4686: default .431 excludes null/flagged levers from headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_59_to_60"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_hierarchical_subgoal_new_level"]["headline_counted"] is False
    assert artifact["a1_hierarchical_subgoal_new_level"]["offline_reproduced"] is False
    assert artifact["a2_factored_planner_coverage_and_lift"]["coverage_delta"] == 0.0
    assert artifact["a2_factored_planner_coverage_and_lift"]["headline_counted"] is False
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["reproducible_total_levels"] == 60
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["live_submittable_level_count"] == 60
    assert artifact["paper_ready"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4686_a1_success_requires_decomposition_and_ablations() -> None:
    """SCENARIO-CAPSTONE-4686: A1 subgoal search counts only with decomposition controls."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1_subgoal(reached_level=2, offline=True, no_subgoal_level=1, random_subgoal_level=1),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_generic_new_level_via_generation_lp85"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a1_hierarchical_subgoal_new_level"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A1_hierarchical_subgoal_search"

    collapsed_control = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1_subgoal(reached_level=2, offline=True, no_subgoal_level=2, random_subgoal_level=1),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert collapsed_control["bridge_crossed_for_solve"] is False
    assert collapsed_control["cited_upstream_artifacts"]["A1"]["reason"] == "control_failed"
    assert collapsed_control["flagged_artifacts_handled"]["control_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4676_hierarchical_subgoal_search_live.json"}
    ]

    missing_decomposition = mod.build_artifact(
        artifacts=_artifacts(
            a1=_a1_subgoal(
                reached_level=2,
                offline=True,
                no_subgoal_level=1,
                random_subgoal_level=1,
                decomposition=[],
            ),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert missing_decomposition["bridge_crossed_for_solve"] is False
    assert missing_decomposition["cited_upstream_artifacts"]["A1"]["reason"] == "decomposition_missing"
    assert missing_decomposition["flagged_artifacts_handled"]["decomposition_missing_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4676_hierarchical_subgoal_search_live.json"}
    ]


def test_scenario_capstone_4686_a2_success_requires_coverage_and_ci() -> None:
    """SCENARIO-CAPSTONE-4686: A2 factored planner counts only with coverage and CI."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_factored(
                factored_coverage=1.0,
                flat_coverage=0.0,
                first_win_delta=0.12,
                ci={"metric": "first_win_rate_delta", "low": 0.03, "high": 0.15},
                offline=True,
            ),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_generic_new_level_via_generation_lp85"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a2_factored_planner_coverage_and_lift"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_factored_planner_coverage_and_lift"

    ci_includes_baseline = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_factored(
                factored_coverage=1.0,
                flat_coverage=0.0,
                first_win_delta=0.12,
                ci={"metric": "first_win_rate_delta", "low": -0.01, "high": 0.15},
                offline=True,
            ),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert ci_includes_baseline["bridge_crossed_for_solve"] is False
    assert ci_includes_baseline["a2_factored_planner_coverage_and_lift"]["reason"] == (
        "live_lift_ci_includes_flat_baseline"
    )

    flat_already_had_winner = mod.build_artifact(
        artifacts=_artifacts(
            a2=_a2_factored(
                factored_coverage=1.0,
                flat_coverage=0.5,
                first_win_delta=0.12,
                ci={"metric": "first_win_rate_delta", "low": 0.03, "high": 0.15},
                offline=True,
            ),
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 59},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=59),
        duration_s=0.001,
    )

    assert flat_already_had_winner["bridge_crossed_for_solve"] is False
    assert flat_already_had_winner["a2_factored_planner_coverage_and_lift"]["reason"] == (
        "winner_already_in_flat_baseline"
    )


def test_req_capstone_4686_exclusion_guards_cover_controls_and_live_flags() -> None:
    """REQ-CAPSTONE-4686: flagged, control-failed, FNR-open, and gate-failed inputs are excluded."""

    artifacts = _artifacts(a2=_a2_factored(factored_coverage=1.0, flat_coverage=None, first_win_delta=0.12))
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
        registry={"reproducible_total_levels": 60},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A2"]["reason"] == "control_failed"
    assert artifact["cited_upstream_artifacts"]["A3"]["reason"] == "false_negative_risk_open"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "positive_control_failed"
    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".430-B2 L2-goal/multi-level-metric"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".431-B2 subgoal-decomposition"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".431-B2 coverage-baseline"] is True


def test_req_capstone_4686_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4686-BLOCKED-PRECONDITION: missing upstreams block clean headlines."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 60\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4686\n", encoding="utf-8")
    scripts_path = tmp_path / "scripts" / "summarize_artifact.py"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text("# test\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, publication_gate=_paper_gate(), write=True, duration_s=0.001)

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4680_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert artifact["bridge_crossed_for_solve"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4686_validation_and_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4686: schema validation, checksum, and defensive helpers fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 60},
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
    assert mod._nontrivial_subgoal_decomposition("start -> bridge -> goal") is True
    assert mod._nontrivial_subgoal_decomposition(["only"]) is False
    assert mod._a1_hierarchical_subgoal_new_level(
        _a1_subgoal(reached_level=2, offline=False, no_subgoal_level=1, random_subgoal_level=1),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "offline_reproduction_missing"
    assert mod._a1_hierarchical_subgoal_new_level(
        _a1_subgoal(
            reached_level=2,
            offline=True,
            no_subgoal_level=1,
            random_subgoal_level=1,
            reachable=[{"name": "bridge", "reachable": False}, {"name": "final", "reachable": True}],
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "subgoal_not_reachable"
    assert mod._a2_factored_planner_coverage_and_lift(
        _a2_factored(
            factored_coverage=1.0,
            flat_coverage=0.0,
            first_win_delta=0.0,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
            offline=True,
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "no_positive_live_lift"
    assert mod._a2_factored_planner_coverage_and_lift(
        _a2_factored(
            factored_coverage=0.0,
            flat_coverage=0.0,
            first_win_delta=0.12,
            ci={"metric": "first_win_rate_delta", "low": 0.01, "high": 0.02},
            offline=True,
        ),
        {"included_in_headline": True, "reason": "manual"},
    )["reason"] == "coverage_delta_not_positive"
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
