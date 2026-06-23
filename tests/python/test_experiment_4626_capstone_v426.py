"""Tests for Exp 4626 .426 capstone scorecard.

Spec refs: REQ-CAPSTONE-4626, SCENARIO-CAPSTONE-4626,
SCENARIO-CAPSTONE-4626-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4626_capstone_v426 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_bridge() -> dict[str, Any]:
    return {
        "experiment": "experiment_4616_offline_live_bridge_disambiguation",
        "honest_verdict": "success: bridge_cause_isolated_compute_fix_identified",
        "verifier_is_oracle": False,
        "binding_bridge_cause": "compute_cost",
        "indicated_fix": "decision-point-only eval/cached features for live frontier nodes",
        "offline_win_confirmed": True,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
    }


def _a2_live(*, flagged: bool = True, first_delta: float = 0.0, actions_delta: float = 0.0) -> dict[str, Any]:
    graduated = 0.04 + first_delta
    graduated_actions = 20.0 - actions_delta if actions_delta else 20.0
    return {
        "experiment": "experiment_4617_graduate_spatial_value_head_live",
        "honest_verdict": (
            "success: spatial_value_head_graduated_live_first_win_up_0.08"
            if first_delta > 0
            else "complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened"
        ),
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "first_win_rate_graduated": graduated,
        "first_win_rate_linear_baseline": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": first_delta,
        "first_win_ci": [first_delta, first_delta],
        "median_actions_to_first_levelup_graduated": graduated_actions,
        "median_actions_to_first_levelup_linear_baseline": 20.0,
        "median_actions_to_first_levelup_bare": 20.0,
        "actions_delta": actions_delta,
        "solve_rate_graduated": 0.04,
        "solve_rate_linear_baseline": 0.04,
        "solve_rate_bare": 0.04,
        "bare_and_linear_controls_passed": True,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": True,
        "offline_reproduced": True,
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4618_levelup_selfplay",
        "honest_verdict": "complete: sk48_delta_identified_no_bank",
        "reproduced_levels": 0,
        "offline_reproduced": False,
        "reproduction_gate": {"game": "sk48", "claimed_level": 1, "reproduced": True},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4619_refresh_submission_package",
        "honest_verdict": "complete: package_refreshed_unchanged_depth.",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 55,
        "count_delta": 0,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
        "refreshed_package_path": "results/experiment_4619_submission_package_operator_resubmit.json",
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4620_primitive_persist_transfer",
        "honest_verdict": "success: primitive_persisted_transfer_bp35_value_added",
        "verifier_is_oracle": False,
        "primitive_persisted": {"operator": "value_head_bridge_fix_operator"},
        "transfer_games": ["bp35", "dc22", "g50t"],
        "transfer_value_per_game": {
            "bp35": {"value_added": True, "live_first_win_lift": True, "efficiency_lift": 1},
            "dc22": {"value_added": True, "live_first_win_lift": True, "efficiency_lift": 2},
            "g50t": {"value_added": False, "live_first_win_lift": False, "efficiency_lift": 0},
        },
        "new_levels_banked": 0,
        "offline_reproduced": {"new_levels_banked": 0},
    }


def _a6_integration() -> dict[str, Any]:
    return {
        "experiment": "experiment_4621_integration_gate",
        "honest_verdict": "complete: integration_no_clean_metric_bare_config_kept_honest_null",
        "verifier_is_oracle": False,
        "offline_to_live_transfer_ratio_integrated": 0.0,
        "offline_to_live_transfer_ratio_delta_vs_baseline": 0.0,
        "first_win_rate_graduated": 0.0,
        "first_win_rate_bare": 0.0,
        "first_win_rate_delta_vs_bare": 0.0,
        "actions_delta_vs_bare": 0.0,
        "submitted_config_raised_metric_clean": False,
        "parity_test_green": True,
        "submitted_to_leaderboard": False,
    }


def _b1_ratio(*, live_lift: float = 0.0) -> dict[str, Any]:
    return {
        "experiment": "experiment_4622_offline_to_live_transfer_ratio_metric",
        "honest_verdict": "success: offline_to_live_transfer_ratio_metric_helper_shipped_tests_green",
        "offline_to_live_transfer_ratio": round(live_lift / 0.674466, 6) if live_lift else 0.0,
        "offline_auroc_component": 0.674466,
        "live_lift_component": live_lift,
        "first_win_lift_component": live_lift,
        "action_efficiency_lift_component": 0.0,
        "bridge_crossed": live_lift > 0,
        "live_lift_breakdown": {
            "graduated_first_win_rate": 0.04 + live_lift,
            "baseline_first_win_rate": 0.04,
            "graduated_median_actions_to_first_levelup": 20.0,
            "baseline_median_actions_to_first_levelup": 20.0,
        },
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4623_adversarial_verify_hardening",
        "honest_verdict": (
            "success: adversarial_verify_hardened_offline_live_overclaim_guard_plus_"
            "cheap_value_substrate_tests_green."
        ),
        "offline_live_overclaim_guard_added": True,
        "cheap_value_substrate_floor_added": True,
        "honest_offline_result_not_flagged": True,
        "tests_added": {"passed": True},
    }


def _artifacts(*, a2_flagged: bool = True, live_lift: float = 0.0) -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_bridge(),
        "A2": _a2_live(flagged=a2_flagged, first_delta=live_lift),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(),
        "B1": _b1_ratio(live_lift=live_lift),
        "B2": _b2_guard(),
    }


def _preconditions(total: int = 55) -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4626": True,
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


def test_req_capstone_4626_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4626: OpenSpec declares the .426 scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4626" in spec
    assert "SCENARIO-CAPSTONE-4626" in spec
    assert "SCENARIO-CAPSTONE-4626-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4626_excludes_stamped_a2_from_live_headline() -> None:
    """REQ-CAPSTONE-4626: stamped A2 cannot support a clean LIVE-win claim."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a2_flagged=True, live_lift=0.0),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: bridge_characterized_cause_isolated_no_live_lift"
    assert artifact["binding_bridge_cause"] == "compute_cost"
    assert artifact["offline_to_live_transfer_ratio"]["offline_auroc_component"] == pytest.approx(0.674466)
    assert artifact["offline_to_live_transfer_ratio"]["live_lift_component"] == pytest.approx(0.0)
    assert artifact["offline_to_live_transfer_ratio"]["bridge_crossed_clean"] is False
    assert artifact["offline_to_live_transfer_ratio"]["a2_live_claim_admissible"] is False
    assert artifact["first_win_rate_scored"]["clean_value"] is None
    assert artifact["first_win_rate_scored"]["quarantined_value"] == pytest.approx(0.04)
    assert artifact["flagged_artifacts_handled"]["excluded_artifacts"] == [
        "results/experiment_4617_graduate_spatial_value_head_live.json"
    ]
    assert artifact["scorecard"]["headline"]["bridge_crossed_clean"] is False
    assert artifact["scorecard"]["B2"]["offline_vs_live_overclaim_guard_active"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4626_clean_live_lift_crosses_bridge() -> None:
    """SCENARIO-CAPSTONE-4626: clean A2 live first-win lift can cross the bridge."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(a2_flagged=False, live_lift=0.08),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 56},
        preconditions_checked=_preconditions(total=56),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: bridge_crossed_live_first_win_up_0.08_cause_compute_cost"
    assert artifact["offline_to_live_transfer_ratio"]["clean_value"] == pytest.approx(0.118612)
    assert artifact["offline_to_live_transfer_ratio"]["bridge_crossed_clean"] is True
    assert artifact["first_win_rate_scored"]["clean_value"] == pytest.approx(0.12)
    assert artifact["first_win_rate_scored"]["delta_vs_linear_baseline"] == pytest.approx(0.08)
    assert artifact["reproducible_total_levels"] == 56
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["scorecard"]["headline"]["a3_bank_plus_one"] is True


def test_scenario_capstone_4626_reports_supporting_scorecard_and_submit_status() -> None:
    """SCENARIO-CAPSTONE-4626: A3/A4/A5/A6 support metrics stay audit-visible."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["reproducible_total_levels"] == 55
    assert artifact["reproducible_total_levels_delta"] == 0
    assert artifact["scorecard"]["A3"]["banked_plus_one"] is False
    assert artifact["live_submittable_level_count"] == 55
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["A4"]["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["A5"]["value_added_games"] == ["bp35", "dc22"]
    assert artifact["scorecard"]["A6"]["submitted_config_raised_metric_clean"] is False
    assert artifact["scorecard"]["verifier_is_oracle_claim_audit"]["all_included_value_claims_false"] is True
    assert artifact["cited_upstream_artifacts"]["A2"]["included_in_headline"] is False
    assert artifact["cited_upstream_artifacts"]["REGISTRY"]["included_in_headline"] is True


def test_req_capstone_4626_guards_positive_controls_and_overclaims() -> None:
    """REQ-CAPSTONE-4626: controls and offline-vs-live guards fail closed."""

    artifacts = _artifacts(a2_flagged=False, live_lift=0.10)
    artifacts["A1"] = {**artifacts["A1"], "positive_control_passed": False}
    flags = {
        "B1": [
            {
                "kind": "OFFLINE_VS_LIVE_OVERCLAIM",
                "severity": "critical",
                "detail": "LIVE-win claim lacks LIVE metric",
            }
        ]
    }

    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name=flags,
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["binding_bridge_cause"] == "unknown"
    assert artifact["offline_to_live_transfer_ratio"]["clean_value"] is None
    assert artifact["offline_to_live_transfer_ratio"]["bridge_crossed_clean"] is False
    assert artifact["flagged_artifacts_handled"]["positive_control_failed_artifacts"] == [
        {"name": "A1", "artifact": "results/experiment_4616_offline_live_bridge_disambiguation.json"}
    ]
    assert artifact["flagged_artifacts_handled"]["excluded_artifacts"] == [
        "results/experiment_4616_offline_live_bridge_disambiguation.json",
        "results/experiment_4622_offline_to_live_transfer_ratio_metric.json",
    ]


def test_scenario_capstone_4626_run_writes_deliverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-CAPSTONE-4626: run validates and writes the requested JSON."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec_path = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("schema_version: 1\nreproducible_total_levels: 55\n", encoding="utf-8")
    for name, source in mod.UPSTREAM_SOURCES.items():
        _write_json(tmp_path / source.relative_path, _artifacts()[name])
    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda root, statuses=None: _preconditions(),
    )

    artifact = mod.run(tmp_path, live_flags_by_name={}, write=True)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["duration_s"] > 0
    assert written["leaderboard_submission"] is False
    assert written["submitted_to_leaderboard"] is False
    assert written["ready_for_operator_submit"] is True
    assert mod.validate_artifact(written) == []


def test_req_capstone_4626_validation_and_helpers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4626: malformed scorecards and helper edge cases fail closed."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name={},
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    broken = dict(artifact)
    broken.update(
        {
            "honest_verdict": "maybe",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "offline_to_live_transfer_ratio": 0.0,
            "first_win_rate_scored": {"clean_value": "0.04"},
            "binding_bridge_cause": "mystery",
            "reproducible_total_levels": "55",
            "reproducible_total_levels_delta": "0",
            "live_submittable_level_count": "55",
            "ready_for_operator_submit": "true",
            "field_principles": {},
            "cited_upstream_artifacts": [],
            "preconditions_checked": [],
            "leaderboard_submission": True,
            "submitted_to_leaderboard": True,
            "reproducibility_checksum": "bad",
        }
    )

    errors = mod.validate_artifact(broken)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "offline_to_live_transfer_ratio must be object" in errors
    assert "first_win_rate_scored.clean_value must be float or null" in errors
    assert "binding_bridge_cause invalid" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "live_submittable_level_count must be bare int" in errors
    assert "ready_for_operator_submit must be bare bool" in errors
    assert "missing field principle for honest_verdict" in errors
    assert "cited_upstream_artifacts must be object" in errors
    assert "preconditions_checked must be object" in errors
    assert "leaderboard_submission must be false" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    missing_clean = dict(artifact)
    missing_clean["offline_to_live_transfer_ratio"] = {"quarantined_value": 0.0}
    missing_clean["reproducibility_checksum"] = mod.payload_checksum(missing_clean)
    assert "offline_to_live_transfer_ratio.clean_value missing" in mod.validate_artifact(missing_clean)

    low_submit = dict(artifact)
    low_submit["live_submittable_level_count"] = 33
    low_submit["reproducibility_checksum"] = mod.payload_checksum(low_submit)
    assert "ready_for_operator_submit requires count above 33" in mod.validate_artifact(low_submit)

    no_principles = dict(artifact)
    no_principles["field_principles"] = None
    no_principles["reproducibility_checksum"] = mod.payload_checksum(no_principles)
    assert "field_principles missing" in mod.validate_artifact(no_principles)

    wrong_checksum = dict(artifact)
    wrong_checksum["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(wrong_checksum)

    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json) == {}
    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("1\n", encoding="utf-8")
    assert mod._read_yaml(scalar_yaml) == {}
    assert mod._file_sha256(tmp_path / "missing.bin") is None
    assert mod._as_float(True, 3.0) == pytest.approx(3.0)
    assert mod._as_float("x", 3.0) == pytest.approx(3.0)
    assert mod._as_int(False, 4) == 4
    assert mod._as_int("x", 4) == 4

    source = mod.SourceSpec("X", "results/x.json", "fixture")
    missing = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={},
        exists=False,
        live_flags_by_name={},
        b2_active=False,
    )
    assert missing["reason"] == "missing"
    gate_failed = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"acceptance_gate_main": False},
        exists=True,
        live_flags_by_name={},
        b2_active=False,
    )
    assert gate_failed["reason"] == "failed_acceptance_gate"
    false_negative = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"honest_verdict": "complete: null"},
        exists=True,
        live_flags_by_name={
            "X": [
                {
                    "kind": "FALSE_NEGATIVE_RISK",
                    "severity": "warn",
                    "detail": "false_negative_risk_open: positive control missing",
                }
            ]
        },
        b2_active=False,
    )
    assert false_negative["reason"] == "false_negative_risk_open"
    tautology = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"honest_verdict": "complete: declared null"},
        exists=True,
        live_flags_by_name={
            "X": [
                {
                    "kind": "TAUTOLOGY",
                    "severity": "critical",
                    "detail": "declared_null_delta control-vs-treatment null-delta",
                }
            ]
        },
        b2_active=True,
    )
    assert tautology["reason"] == "included_clean_with_tautology_carveout"
    handled = mod._flagged_artifacts_handled(
        {"F": false_negative, "G": gate_failed, "T": tautology},
        _b2_guard(),
    )
    assert handled["false_negative_risk_open_artifacts"] == [
        {"name": "F", "artifact": "results/x.json"}
    ]
    assert handled["failed_acceptance_gate_overrides"] == [
        {"name": "G", "artifact": "results/x.json", "failed_gates": ["acceptance_gate_main"]}
    ]
    assert handled["tautology_carveouts"][0]["name"] == "T"

    blocked = mod.build_artifact(
        artifacts={},
        live_flags_by_name={},
        registry={},
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.001,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(path=tmp_path / "not-written.json", artifact=broken)

    monkeypatch.setattr(mod, "build_artifact", lambda *args, **kwargs: broken)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.run(tmp_path, write=False)
