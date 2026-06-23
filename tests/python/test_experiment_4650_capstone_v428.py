"""Tests for Exp 4650 .428 capstone scorecard.

Spec refs: REQ-CAPSTONE-4650, SCENARIO-CAPSTONE-4650,
SCENARIO-CAPSTONE-4650-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4650_capstone_v428 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_goal_energy(*, solve_delta: float = 0.0, first_win_delta: float = 0.0, ablation: bool = False) -> dict[str, Any]:
    return {
        "experiment": "experiment_4640_goal_energy_generation_live",
        "honest_verdict": (
            f"success: goal_energy_live_generation_solverate_up_{solve_delta}"
            if solve_delta > 0.0
            else "complete: goal_energy_no_live_lift_honest_null_gap_sharpened"
        ),
        "verifier_is_oracle": False,
        "live_solve_rate_goal_energy": 0.04 + solve_delta,
        "live_solve_rate_baseline": 0.04,
        "solve_rate_delta": solve_delta,
        "first_win_rate_delta": first_win_delta,
        "uniform_energy_ablation_passed": ablation,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "offline_reproduced": True,
        "duration_s": 1.0,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _a2_expansion(*, multilevel_delta: float = 0.0, first_win_rate: float = 0.590909) -> dict[str, Any]:
    expansion_rate = max(0.0, multilevel_delta)
    return {
        "experiment": "experiment_4641_action_effect_expansion_prior_live",
        "honest_verdict": (
            f"success: action_effect_expansion_prior_live_deeper_solve_{multilevel_delta}"
            if multilevel_delta > 0.0
            else "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened"
        ),
        "verifier_is_oracle": False,
        "live_solve_rate_expansion": expansion_rate,
        "live_solve_rate_ranker_baseline": 0.0,
        "solve_rate_delta": expansion_rate,
        "depth_of_live_solve_delta": 1.0 if multilevel_delta > 0.0 else 0.0,
        "first_win_rate_delta": first_win_rate - 0.590909,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "offline_reproduced": True,
        "expansion_measurement": {
            "first_win_rate": first_win_rate,
            "live_solve_rate": expansion_rate,
            "depth_of_live_solve": 2.0 if multilevel_delta > 0.0 else 1.0,
        },
        "ranker_measurement": {
            "first_win_rate": 0.590909,
            "live_solve_rate": 0.0,
            "depth_of_live_solve": 1.0,
        },
        "duration_s": 1.0,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4642_levelup_selfplay",
        "honest_verdict": "success: ft09_L3_offline_reproduced",
        "verifier_is_oracle": False,
        "reproduced_levels": 1,
        "reproducible_total_levels_before": 56,
        "reproducible_total_levels_after": 57,
        "offline_reproduced": True,
        "reproduction_gate": {"game": "ft09", "claimed_level": 3, "reached_level": 3, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4643_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_57_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 57,
        "live_submittable_count_prev": 56,
        "count_delta": 1,
        "levels_folded_in": ["ft09"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4644_primitive_persist_transfer",
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "graded_goal_energy_search_heuristic_operator"},
        "transfer_games": ["bp35", "cd82", "dc22"],
        "transfer_value_per_game": {"bp35": {"value_added": False}},
    }


def _a6_integration() -> dict[str, Any]:
    return {
        "experiment": "experiment_4645_integration_gate",
        "honest_verdict": "success: integrated_live_submittable_raised_config_shipped",
        "verifier_is_oracle": False,
        "live_solve_rate_integrated": 0.04,
        "live_solve_rate_delta_vs_bare": 0.0,
        "live_multi_level_solve_rate_integrated": 0.0,
        "live_multi_level_solve_rate_delta_vs_bare": 0.0,
        "live_submittable_level_count_integrated": 57,
        "submitted_config_raised_metric_clean": True,
    }


def _b1_multilevel(*, rate: float = 0.0, first_win_rate: float = 0.590909) -> dict[str, Any]:
    return {
        "experiment": "experiment_4646_live_multi_level_solve_rate_metric",
        "honest_verdict": "success: live_multi_level_solve_rate_metric_helper_shipped_tests_green",
        "live_multi_level_solve_rate": rate,
        "live_attempt_count": 50,
        "multi_level_attempt_count": int(rate * 50),
        "coheadline_block": {
            "live_multi_level_solve_rate": rate,
            "live_action_efficiency": 0.758102,
            "first_win_rate": first_win_rate,
            "offline_to_live_transfer_ratio": 0.0,
            "live_submittable_level_count": 57,
            "reproducible_total_levels": 57,
        },
        "tests_added": {"passed": True},
    }


def _b2_guard(*, active: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4647_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_goal_energy_ablation_guard_tests_green.",
        "goal_energy_ablation_guard_added": active,
        "honest_ablation_not_flagged": active,
        "diagnostic_not_flagged": active,
        "tests_added": {"passed": active},
    }


def _artifacts(
    *,
    a1_solve_delta: float = 0.0,
    a1_first_win_delta: float = 0.0,
    a1_ablation: bool = False,
    a2_multilevel_delta: float = 0.0,
    b1_multilevel_rate: float = 0.0,
    first_win_rate: float = 0.590909,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_goal_energy(
            solve_delta=a1_solve_delta,
            first_win_delta=a1_first_win_delta,
            ablation=a1_ablation,
        ),
        "A2": _a2_expansion(multilevel_delta=a2_multilevel_delta, first_win_rate=first_win_rate),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(),
        "B1": _b1_multilevel(rate=b1_multilevel_rate, first_win_rate=first_win_rate),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 57) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4650": True,
        "registry_yaml_loadable": True,
        "registry_reproducible_total_levels": total,
        "offline_arcade": True,
        "upstream_artifacts_present": {name: True for name in mod.UPSTREAM_SOURCES},
        "missing_upstream_artifacts": [],
        "summarize_artifact_py_used_for_live_flags": True,
        "leaderboard_submission": False,
        "operator_only": True,
        "research_conductor_modified": False,
    }


def test_req_capstone_4650_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4650: OpenSpec declares the .428 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4650" in spec
    assert "SCENARIO-CAPSTONE-4650" in spec
    assert "SCENARIO-CAPSTONE-4650-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4650_default_null_falls_back_to_capability_growth() -> None:
    """SCENARIO-CAPSTONE-4650: ablation-failed A1 and null A2 do not claim bridge extension."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 57},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_56_to_57"
    assert artifact["uniform_energy_ablation_passed"] is False
    assert artifact["live_solve_rate_delta"]["clean_value"] is None
    assert artifact["live_solve_rate_delta"]["quarantined_value"] == pytest.approx(0.0)
    assert artifact["cited_upstream_artifacts"]["A1"]["reason"] == "uniform_energy_ablation_failed"
    assert artifact["flagged_artifacts_handled"]["ablation_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4640_goal_energy_generation_live.json"}
    ]
    assert artifact["live_multi_level_solve_rate"]["clean_value"] == pytest.approx(0.0)
    assert artifact["scorecard"]["headline"]["bridge_extended_by_energy_driven_generation"] is False
    assert artifact["scorecard"]["headline"]["a3_bank_plus_one"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4650_a1_success_requires_uniform_ablation() -> None:
    """SCENARIO-CAPSTONE-4650: A1 solve-rate lift is headline-valid only after ablation passes."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a1_solve_delta=0.08, a1_first_win_delta=0.08, a1_ablation=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(total=56),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_extended_live_solverate_up_0.08"
    assert artifact["uniform_energy_ablation_passed"] is True
    assert artifact["live_solve_rate_delta"]["clean_value"] == pytest.approx(0.08)
    assert artifact["live_solve_rate_delta"]["uniform_energy_ablation_passed"] is True
    assert artifact["offline_to_live_transfer_ratio"]["clean_value"] == pytest.approx(0.08)
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A1_goal_energy_solverate"


def test_scenario_capstone_4650_a2_multilevel_success_beats_ranker() -> None:
    """SCENARIO-CAPSTONE-4650: A2/B1 multi-level lift can extend the bridge."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a2_multilevel_delta=0.2, b1_multilevel_rate=0.2, first_win_rate=0.65),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(total=56),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_extended_live_multilevel_up_0.2"
    assert artifact["live_multi_level_solve_rate"]["clean_value"] == pytest.approx(0.2)
    assert artifact["live_multi_level_solve_rate"]["delta_vs_ranker_baseline"] == pytest.approx(0.2)
    assert artifact["first_win_rate_scored"]["clean_value"] == pytest.approx(0.65)
    assert artifact["first_win_rate_scored"]["regressed_vs_427_baseline"] is False
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_live_multi_level"


def test_req_capstone_4650_exclusion_guards_cover_controls_flags_and_gates() -> None:
    """REQ-CAPSTONE-4650: positive-control, false-negative, flagged, and gate failures are excluded."""

    artifacts = _artifacts(a1_ablation=True, a2_multilevel_delta=0.2, b1_multilevel_rate=0.2)
    artifacts["A2"]["false_negative_risk_checked"] = False
    artifacts["A5"]["acceptance_gate_transfer"] = False
    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={
            "A4": [
                {
                    "kind": "FALSE_NEGATIVE_RISK",
                    "severity": "warn",
                    "detail": "false_negative_risk_open: package control absent",
                }
            ],
            "A6": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "live_solve_rate_bare=0.04 and live_solve_rate_integrated=0.04",
                }
            ],
        },
        registry={"reproducible_total_levels": 57},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A2"]["reason"] == "positive_control_failed"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "false_negative_risk_open"
    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["live_multi_level_solve_rate"]["clean_value"] is None
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".428-B2 goal-energy-ablation"] is True


def test_req_capstone_4650_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4650: missing upstreams are recorded without fabricated metrics."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 57\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4650\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, write=True, duration_s=0.001)

    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4644_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4650_validation_and_helper_edges(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4650: schema, checksum, blocked, and defensive helper paths fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(a1_ablation=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 57},
        preconditions_checked={**_preconditions(), "ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.001,
    )
    bad = dict(blocked)
    bad["verifier_is_oracle"] = True

    def bad_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"honest_verdict": "complete: invalid"}

    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert mod._as_float(True, 3.0) == pytest.approx(3.0)
    assert mod._as_int(False, 4) == 4
    assert mod._file_sha256(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad)
    monkeypatch.setattr(mod, "build_artifact", bad_build)
    with pytest.raises(ValueError, match="missing required field"):
        mod.run(tmp_path, write=False, duration_s=0.001)
