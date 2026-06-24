"""Tests for Exp 4662 .429 capstone scorecard.

Spec refs: REQ-CAPSTONE-4662, SCENARIO-CAPSTONE-4662,
SCENARIO-CAPSTONE-4662-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4662_capstone_v429 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_value_routing(
    *,
    first_win_delta: float = 0.0,
    solve_delta: float = 0.0,
    ci95: tuple[float, float] = (0.0, 0.0),
    cost: float | None = 0.397451,
    timed_out: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "experiment_4652_value_routing_cost_fix_live",
        "honest_verdict": (
            "success: value_routing_cost_fixed_live_firstwin_up_0.08"
            if first_win_delta > 0.0
            else "complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration."
        ),
        "verifier_is_oracle": False,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "first_win_rate_delta": first_win_delta,
        "solve_rate_delta": solve_delta,
        "live_first_win_rate_value_routed": 0.04 + first_win_delta,
        "live_solve_rate_value_routed": solve_delta,
        "live_baseline_value_weight_zero": {"first_win_rate": 0.04, "solve_rate": 0.0, "value_weight": 0.0},
        "live_lift_ci": {
            "ci95": list(ci95),
            "metric": "first_win_rate" if first_win_delta >= solve_delta else "solve_rate",
            "point": max(first_win_delta, solve_delta),
        },
        "sim_timed_out": timed_out,
        "value_weight_set": 1e-12,
        "duration_s": 1.0,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }
    if cost is not None:
        payload["per_node_feature_cost_ms"] = cost
    return payload


def _a2_qd(*, winner: bool = False, ablation: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4653_energy_fitness_qd_generation_live",
        "honest_verdict": (
            "success: energy_fitness_qd_winner_generated_1"
            if winner
            else "complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened"
        ),
        "verifier_is_oracle": False,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "winner_generated": winner,
        "winner_generated_count": 1 if winner else 0,
        "random_mutation_ablation_passed": ablation,
        "qd_lift_ci": {"ci95": [0.2, 0.4] if winner and ablation else [0.0, 0.0], "point": 1 if winner else 0},
        "live_solve_rate_qd": 0.0,
        "live_solve_rate_search_baseline": 0.0,
        "first_win_rate_delta": 0.0,
        "solve_rate_delta": 0.0,
        "offline_reproduced": winner,
        "duration_s": 1.0,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4654_levelup_selfplay",
        "honest_verdict": "success: vc33_L2_offline_reproduced",
        "verifier_is_oracle": False,
        "reproduced_levels": 1,
        "offline_reproduced": True,
        "reproduction_gate": {"game": "vc33", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4655_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_58_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 58,
        "live_submittable_count_prev": 57,
        "count_delta": 1,
        "levels_folded_in": ["vc33"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4656_primitive_persist_transfer",
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "cost_fixed_value_routing_operator"},
        "transfer_value_per_game": {"bp35": {"value_added": False}},
    }


def _a6_integration(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4657_integration_gate",
        "honest_verdict": "success: integrated_a1_value_routing_cost_fix_shipped_parity_green",
        "verifier_is_oracle": False,
        "flagged_adversarial": flagged,
        "live_first_win_rate_integrated": 0.04,
        "live_first_win_rate_pre_integration": 0.04,
        "live_first_win_rate_delta_vs_pre_integration": 0.0,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_multi_level_solve_rate_pre_integration": 0.0,
        "live_multi_level_solve_rate_delta_vs_pre_integration": 0.0,
    }


def _b1_diagnostic() -> dict[str, Any]:
    return {
        "experiment": "experiment_4658_value_routing_cigate_diagnostic",
        "honest_verdict": "success: value_routing_cigate_plus_diagnostic_shipped_tests_green.",
        "verifier_is_oracle": False,
        "ci_gate": {"passed": True, "sim_timed_out": False},
        "distribution_shift_score": 0.699108,
        "tests_added": {"passed": True},
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4659_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_qd_ablation_and_value_routing_cost_guards_tests_green.",
        "qd_random_mutation_ablation_guard_added": True,
        "value_routing_cost_control_guard_added": True,
        "tests_added": {"passed": True},
    }


def _artifacts(
    *,
    a1_first_win_delta: float = 0.0,
    a1_solve_delta: float = 0.0,
    a1_ci95: tuple[float, float] = (0.0, 0.0),
    a1_cost: float | None = 0.397451,
    a1_timed_out: bool = False,
    a2_winner: bool = False,
    a2_ablation: bool = False,
    a6_flagged: bool = True,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_value_routing(
            first_win_delta=a1_first_win_delta,
            solve_delta=a1_solve_delta,
            ci95=a1_ci95,
            cost=a1_cost,
            timed_out=a1_timed_out,
        ),
        "A2": _a2_qd(winner=a2_winner, ablation=a2_ablation),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(flagged=a6_flagged),
        "B1": _b1_diagnostic(),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 58) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4662": True,
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
        "gates": {
            "G1": {"pass": ready, "detail": "FoVer dual-condition AUROC artifact present"},
            "G2": {"pass": ready, "detail": "independent reproducer confirmed"},
            "G3": {"pass": ready, "detail": "narrowing-clean"},
            "G4": {"pass": ready, "detail": "numbers trace to artifact"},
        },
        "unmet_gates": [] if ready else ["G2"],
    }


def test_req_capstone_4662_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4662: OpenSpec declares the .429 scorecard fields and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4662" in spec
    assert "SCENARIO-CAPSTONE-4662" in spec
    assert "SCENARIO-CAPSTONE-4662-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4662_default_null_bridge_counts_registry_growth() -> None:
    """SCENARIO-CAPSTONE-4662: default .429 gates exclude null/failed levers from bridge headline."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_57_to_58"
    assert artifact["bridge_crossed_for_solve"] is False
    assert artifact["a1_value_routing_live_lift"]["cost_controlled"] is True
    assert artifact["a1_value_routing_live_lift"]["ci_excludes_value_weight_zero_baseline"] is False
    assert artifact["a1_value_routing_live_lift"]["headline_counted"] is False
    assert artifact["a2_winner_generated"]["winner_generated"] is False
    assert artifact["a2_winner_generated"]["random_mutation_ablation_passed"] is False
    assert artifact["cited_upstream_artifacts"]["A2"]["reason"] == "random_mutation_ablation_failed"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["flagged_artifacts_handled"]["ablation_failed_artifacts"] == [
        {"name": "A2", "artifact": "results/experiment_4653_energy_fitness_qd_generation_live.json"}
    ]
    assert artifact["flagged_artifacts_handled"]["flagged_adversarial_artifacts"] == [
        {"name": "A6", "artifact": "results/experiment_4657_integration_gate.json"}
    ]
    assert artifact["reproducible_total_levels"] == 58
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["live_submittable_level_count"] == 58
    assert artifact["paper_ready"] is True
    assert artifact["leaderboard_submission"] is False
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4662_a1_firstwin_success_requires_cost_and_clean_ci() -> None:
    """SCENARIO-CAPSTONE-4662: A1 value-routing bridge wins only with cost control and CI."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a1_first_win_delta=0.08, a1_ci95=(0.04, 0.12), a2_ablation=True, a6_flagged=False),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 57},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=57),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_firstwin_up_0.08"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a1_value_routing_live_lift"]["headline_counted"] is True
    assert artifact["a1_value_routing_live_lift"]["selected_metric"] == "first_win_rate_delta"

    blocked_by_cost = mod.build_artifact(
        artifacts=_artifacts(
            a1_first_win_delta=0.08,
            a1_ci95=(0.04, 0.12),
            a1_cost=None,
            a1_timed_out=True,
            a2_ablation=True,
            a6_flagged=False,
        ),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 57},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=57),
        duration_s=0.001,
    )

    assert blocked_by_cost["honest_verdict"] == "complete: generation_guidance_levers_characterized_no_live_solve_lift"
    assert blocked_by_cost["a1_value_routing_live_lift"]["cost_controlled"] is False
    assert blocked_by_cost["flagged_artifacts_handled"]["cost_control_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4652_value_routing_cost_fix_live.json"}
    ]


def test_scenario_capstone_4662_a2_winner_success_requires_random_mutation_ablation() -> None:
    """SCENARIO-CAPSTONE-4662: A2 QD winner counts only after random-mutation ablation passes."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a2_winner=True, a2_ablation=True, a6_flagged=False),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 57},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(total=57),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_winner_generated_up_1"
    assert artifact["bridge_crossed_for_solve"] is True
    assert artifact["a2_winner_generated"]["headline_counted"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_energy_fitness_qd"


def test_req_capstone_4662_exclusion_guards_cover_control_flags_and_gates() -> None:
    """REQ-CAPSTONE-4662: flagged, control-failed, false-negative, and gate-failed inputs are excluded."""

    artifacts = _artifacts(a2_ablation=True, a6_flagged=False)
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
        registry={"reproducible_total_levels": 58},
        publication_gate=_paper_gate(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A3"]["reason"] == "false_negative_risk_open"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "positive_control_failed"
    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["live_submittable_level_count"] == 0
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".429-B2 QD-ablation"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".429-B2 value-routing-cost"] is True


def test_req_capstone_4662_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4662: missing upstreams block without fabricated headline metrics."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 58\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4662\n", encoding="utf-8")
    scripts_path = tmp_path / "scripts" / "summarize_artifact.py"
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    scripts_path.write_text("# test\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, publication_gate=_paper_gate(), write=True, duration_s=0.001)

    assert artifact["honest_verdict"] == "blocked_upstream_artifacts"
    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4656_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert artifact["bridge_crossed_for_solve"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4662_validation_and_defensive_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4662: schema validation, checksum, and defensive helpers fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(a2_ablation=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 58},
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
    assert mod._a1_value_routing_live_lift(
        _a1_value_routing(first_win_delta=0.08, ci95=(0.04, 0.12)),
        {"included_in_headline": True, "value_routing_cost_control_failed": True, "reason": "manual"},
    )["reason"] == "value_routing_cost_control_failed"
    assert mod._a1_value_routing_live_lift(
        _a1_value_routing(first_win_delta=0.0, ci95=(0.01, 0.02)),
        {"included_in_headline": True, "value_routing_cost_control_failed": False, "reason": "manual"},
    )["reason"] == "no_positive_live_lift"
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
