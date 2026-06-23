"""Tests for Exp 4638 .427 capstone scorecard.

Spec refs: REQ-CAPSTONE-4638, SCENARIO-CAPSTONE-4638,
SCENARIO-CAPSTONE-4638-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4638_capstone_v427 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_curiosity(*, solve_delta: float = 0.0, coverage_delta: int = 2) -> dict[str, Any]:
    return {
        "experiment": "experiment_4628_dense_curiosity_progress_loop",
        "honest_verdict": "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened",
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": True,
        "live_solve_rate_loop": 0.04 + solve_delta,
        "live_solve_rate_bare": 0.04,
        "solve_rate_delta": solve_delta,
        "state_coverage_delta": coverage_delta,
        "first_win_rate_delta": 0.0,
        "live_lift_ci": {"metric": "state_coverage_delta", "point": coverage_delta / 25.0, "ci95": [0.0, 0.2]},
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "chosen_submitted_config": "unchanged",
        "offline_reproduced": True,
    }


def _a2_predictor(*, actions_delta: float = 1.0, first_win_delta: float = 0.1836363636) -> dict[str, Any]:
    return {
        "experiment": "experiment_4629_graduate_action_effect_predictor_live",
        "honest_verdict": (
            f"success: action_effect_predictor_graduated_live_efficiency_up_{int(actions_delta)}"
            if actions_delta > 0
            else "complete: action_effect_predictor_graduated_no_live_efficiency_honest_null_gap_sharpened"
        ),
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": True,
        "median_actions_to_first_levelup_predictor": 1.0 if actions_delta else 2.0,
        "median_actions_to_first_levelup_bare": 2.0,
        "actions_delta": actions_delta,
        "efficiency_score_term": 1.0 if actions_delta else 0.25,
        "actions_delta_ci": [actions_delta, actions_delta],
        "first_win_rate_delta": first_win_delta if actions_delta else 0.0,
        "solve_rate_preserved": True,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "parity_test_green": True,
        "chosen_submitted_config": (
            "frame_change_predictor_enabled:persistent_aem_plus_optional_cnn"
            if actions_delta
            else "unchanged"
        ),
        "offline_reproduced": {"all_new_solves_reproduced": True, "newly_solved_variants": []},
        "live_measurement": {
            "first_win_rate_predictor": 0.5909090909 if actions_delta else 0.4072727273,
            "first_win_rate_bare": 0.4072727273,
            "first_win_rate_delta": first_win_delta if actions_delta else 0.0,
            "solve_rate_predictor": 1.0,
            "solve_rate_bare": 1.0,
            "median_actions_to_first_levelup_predictor": 1.0 if actions_delta else 2.0,
            "median_actions_to_first_levelup_bare": 2.0,
            "actions_delta": actions_delta,
        },
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4630_levelup_selfplay",
        "honest_verdict": "success: ls20_L2_offline_reproduced",
        "verifier_is_oracle": False,
        "reproduced_levels": 1,
        "reproducible_total_levels_before": 55,
        "reproducible_total_levels_after": 56,
        "offline_reproduced": True,
        "reproduction_gate": {"game": "ls20", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4631_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_56_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 56,
        "live_submittable_count_prev": 55,
        "count_delta": 1,
        "levels_folded_in": ["ls20"],
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
        "refreshed_package_path": "results/experiment_4631_submission_package_operator_resubmit.json",
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4632_primitive_persist_transfer",
        "honest_verdict": "success: primitive_persisted_transfer_sp80_value_added",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "action_effect_predictor_persistent_aem"},
        "transfer_games": ["sp80", "wa30"],
        "transfer_value_per_game": {
            "sp80": {"value_added": True},
            "wa30": {"value_added": False},
        },
        "offline_reproduced": False,
        "reproducible_total_levels": 56,
    }


def _a6_integration(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4633_integration_gate",
        "honest_verdict": "success: integrated_action_efficiency_raised_config_shipped",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "live_solve_rate_integrated": 0.04,
        "live_solve_rate_bare": 0.04,
        "live_solve_rate_delta_vs_bare": 0.0,
        "actions_delta_vs_bare": 1.0,
        "live_action_efficiency_integrated": 0.758102,
        "submitted_config_raised_metric_clean": True,
        "submitted_to_leaderboard": False,
    }


def _b1_efficiency(*, efficiency: float = 0.758102) -> dict[str, Any]:
    return {
        "experiment": "experiment_4634_live_action_efficiency_metric",
        "honest_verdict": "success: live_action_efficiency_metric_helper_shipped_tests_green",
        "live_action_efficiency": efficiency,
        "coheadline_block": {
            "live_action_efficiency": efficiency,
            "first_win_rate": 0.590909,
            "reproducible_total_levels": 56,
            "live_submittable_level_count": 56,
            "offline_to_live_transfer_ratio": 0.0,
            "solved_level_count": 24,
        },
        "tests_added": {"passed": True},
        "preconditions_checked": {"ok": True, "offline_arcade": True},
    }


def _b2_guard(*, active: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4635_adversarial_verify_hardening",
        "honest_verdict": (
            "success: adversarial_verify_hardened_intrinsic_reward_guard_plus_cnn_substrate_tests_green."
            if active
            else "complete: adversarial_verify_hardening_not_run"
        ),
        "intrinsic_reward_overclaim_guard_added": active,
        "cnn_substrate_floor_added": active,
        "honest_diagnostic_not_flagged": active,
        "no_methodology_fast_run_still_fires": active,
        "tests_added": {"passed": active},
        "preconditions_checked": {"ok": active, "research_conductor_modified": False},
    }


def _artifacts(
    *,
    actions_delta: float = 1.0,
    a6_flagged: bool = True,
    b2_active: bool = True,
    a1_solve_delta: float = 0.0,
    a1_coverage_delta: int = 2,
) -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_curiosity(solve_delta=a1_solve_delta, coverage_delta=a1_coverage_delta),
        "A2": _a2_predictor(actions_delta=actions_delta),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(flagged=a6_flagged),
        "B1": _b1_efficiency(),
        "B2": _b2_guard(active=b2_active),
    }


def _preconditions(total: int = 56) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4638": True,
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


def test_req_capstone_4638_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4638: OpenSpec declares the .427 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4638" in spec
    assert "SCENARIO-CAPSTONE-4638" in spec
    assert "SCENARIO-CAPSTONE-4638-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_capstone_4638_crosses_bridge_by_clean_live_efficiency() -> None:
    """SCENARIO-CAPSTONE-4638: clean A2/B1 efficiency lift can cross the bridge."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(actions_delta=1.0, a6_flagged=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(total=56),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_efficiency_up_1"
    assert artifact["live_action_efficiency"]["clean_value"] == pytest.approx(0.758102)
    assert artifact["live_action_efficiency"]["actions_delta_vs_bare"] == pytest.approx(1.0)
    assert artifact["live_action_efficiency"]["bridge_crossed_clean"] is True
    assert artifact["offline_to_live_transfer_ratio"]["clean_value"] == pytest.approx(0.5)
    assert artifact["first_win_rate_scored"]["clean_value"] == pytest.approx(0.5909090909)
    assert artifact["scorecard"]["headline"]["bridge_crossed_by_generation"] is True
    assert artifact["scorecard"]["headline"]["crossing_source"] == "A2_live_action_efficiency"
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4638_excludes_flagged_integration_from_headline() -> None:
    """REQ-CAPSTONE-4638: stamped/live-critical A6 cannot support headline claims."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a6_flagged=True),
        live_flags_by_name={
            "A6": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "live_solve_rate_bare=0.04 and live_solve_rate_integrated=0.04",
                }
            ]
        },
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["flagged_artifacts_handled"]["excluded_artifacts"] == [
        "results/experiment_4633_integration_gate.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A6"]["included_in_headline"] is False
    assert artifact["cited_upstream_artifacts"]["A6"]["reason"] == "flagged_adversarial_or_live_critical_excluded"
    assert artifact["scorecard"]["A6"]["submitted_config_raised_metric_clean"] is False
    assert artifact["scorecard"]["verifier_is_oracle_claim_audit"]["all_included_value_claims_false"] is True


def test_req_capstone_4638_reports_a1_coverage_without_intrinsic_reward_overclaim() -> None:
    """REQ-CAPSTONE-4638: A1 exploration claims need downstream metrics and .427-B2 guard."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a1_coverage_delta=2, b2_active=True),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["live_solve_rate_delta"]["clean_value"] == pytest.approx(0.0)
    assert artifact["live_solve_rate_delta"]["state_coverage_delta"] == 2
    assert artifact["live_solve_rate_delta"]["downstream_metric_present"] is True
    assert artifact["live_solve_rate_delta"]["intrinsic_bonus_only_claim"] is False
    assert artifact["scorecard"]["A1"]["coverage_up_clean"] is True
    assert artifact["flagged_artifacts_handled"]["guards_applied"][".427-B2 intrinsic-reward"] is True


def test_scenario_capstone_4638_no_live_lift_falls_back_to_capability_growth() -> None:
    """SCENARIO-CAPSTONE-4638: no clean live lift reports capability growth honestly."""

    artifacts = _artifacts(actions_delta=0.0, a1_coverage_delta=0, a6_flagged=True)
    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_55_to_56"
    assert artifact["live_action_efficiency"]["bridge_crossed_clean"] is False
    assert artifact["live_solve_rate_delta"]["bridge_crossed_clean"] is False
    assert artifact["reproducible_total_levels"] == 56
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["ready_for_operator_submit"] is True


def test_req_capstone_4638_positive_control_failed_artifact_is_excluded() -> None:
    """REQ-CAPSTONE-4638: positive-control-failed upstreams fail closed."""

    artifacts = _artifacts()
    artifacts["A2"]["false_negative_risk_checked"] = False

    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: capability_grew_55_to_56"
    assert artifact["live_action_efficiency"]["clean_value"] is None
    assert artifact["live_action_efficiency"]["quarantined_value"] == pytest.approx(0.758102)
    assert artifact["flagged_artifacts_handled"]["positive_control_failed_artifacts"] == [
        {"name": "A2", "artifact": "results/experiment_4629_graduate_action_effect_predictor_live.json"}
    ]


def test_scenario_capstone_4638_writes_stable_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4638: write path emits a checksum-stable artifact."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    out = tmp_path / "experiment_4638_capstone_v427.json"

    written = mod.write_artifact(path=out, artifact=artifact)
    loaded = json.loads(written.read_text(encoding="utf-8"))

    assert written == out
    assert loaded["reproducibility_checksum"] == mod.payload_checksum(loaded)


def test_req_capstone_4638_run_reads_injected_files_and_records_missing(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4638: missing upstreams are recorded without fabrication."""

    for name, payload in _artifacts().items():
        if name == "A5":
            continue
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 56\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4638\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, write=True, duration_s=0.001)

    assert artifact["preconditions_checked"]["missing_upstream_artifacts"] == [
        "results/experiment_4632_primitive_persist_transfer.json"
    ]
    assert artifact["cited_upstream_artifacts"]["A5"]["exists"] is False
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_capstone_4638_records_gate_and_false_negative_exclusions() -> None:
    """REQ-CAPSTONE-4638: failed gates and open false-negative risk are reported."""

    artifacts = _artifacts()
    artifacts["A5"]["acceptance_gate_transfer"] = False
    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name={
            "A4": [
                {
                    "kind": "FALSE_NEGATIVE_RISK",
                    "severity": "warn",
                    "detail": "false_negative_risk_open: bare control absent",
                }
            ]
        },
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["cited_upstream_artifacts"]["A5"]["reason"] == "failed_acceptance_gate"
    assert artifact["cited_upstream_artifacts"]["A4"]["reason"] == "false_negative_risk_open"
    assert artifact["flagged_artifacts_handled"]["failed_acceptance_gate_overrides"] == [
        {
            "name": "A5",
            "artifact": "results/experiment_4632_primitive_persist_transfer.json",
            "failed_gates": ["acceptance_gate_transfer"],
        }
    ]
    assert artifact["flagged_artifacts_handled"]["false_negative_risk_open_artifacts"] == [
        {"name": "A4", "artifact": "results/experiment_4631_refresh_submission_package.json"}
    ]


def test_scenario_capstone_4638_solverate_and_null_verdict_paths() -> None:
    """SCENARIO-CAPSTONE-4638: solverate success and honest null verdicts are explicit."""

    solverate = mod.build_artifact(
        artifacts=_artifacts(actions_delta=0.0, a1_solve_delta=0.04, a1_coverage_delta=0),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(total=55),
        duration_s=0.001,
    )
    null = mod.build_artifact(
        artifacts=_artifacts(actions_delta=0.0, a1_coverage_delta=0),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(total=55),
        duration_s=0.001,
    )

    assert solverate["honest_verdict"] == "success: bridge_crossed_live_solverate_up_0.04"
    assert null["honest_verdict"] == "complete: generation_levers_characterized_no_live_lift"


def test_req_capstone_4638_blocked_and_validation_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CAPSTONE-4638: blocked preconditions and invalid writes fail closed."""

    blocked = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked={**_preconditions(), "ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.001,
    )
    bad_artifact = dict(blocked)
    bad_artifact["verifier_is_oracle"] = True

    def bad_build(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"honest_verdict": "complete: invalid"}

    monkeypatch.setattr(mod, "build_artifact", bad_build)

    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.write_artifact(path=tmp_path / "bad.json", artifact=bad_artifact)
    with pytest.raises(ValueError, match="missing required field"):
        mod.run(tmp_path, write=False, duration_s=0.001)


def test_req_capstone_4638_small_helpers_and_run_duration_path(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4638: helper edge cases and auto-duration run stay stable."""

    for name, payload in _artifacts().items():
        _write_json(tmp_path / mod.UPSTREAM_SOURCES[name].relative_path, payload)
    registry_path = tmp_path / mod.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 56\n", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# test\n", encoding="utf-8")
    spec_path = tmp_path / mod.SPEC_RELATIVE_PATH
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4638\n", encoding="utf-8")

    artifact = mod.run(tmp_path, live_flags_by_name={}, write=False)

    assert mod._file_sha256(tmp_path / "missing.json") is None
    assert mod._as_float(True, 7.0) == pytest.approx(7.0)
    assert mod._as_float("not-a-float", 3.0) == pytest.approx(3.0)
    assert mod._as_int(False, 4) == 4
    assert artifact["duration_s"] > 0.0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
