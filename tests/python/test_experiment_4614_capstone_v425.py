"""Tests for Exp 4614 .425 capstone scorecard.

Spec refs: REQ-CAPSTONE-4614, SCENARIO-CAPSTONE-4614,
SCENARIO-CAPSTONE-4614-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4614_capstone_v425 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_world_model(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4604_world_model_trust_energy",
        "honest_verdict": "success: world_model_trust_energy_pass_rate_up_6_first_win_up",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "world_model_trust_pass_rate_new": 1.0,
        "world_model_trust_pass_rate_binary": 0.0,
        "trust_pass_rate_delta": 1.0,
        "trust_pass_numerator": 6,
        "trust_pass_denominator": 6,
        "first_win_rate_new": 1.0,
        "first_win_rate_binary": 0.0,
        "first_win_delta": 1.0,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "offline_reproduced": True,
        "measurements": [
            {
                "game": "ar25",
                "new_trust_pass": True,
                "new_planner_used": True,
                "new_correct_changed_cells": 2,
            }
        ],
    }


def _a2_scored(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "experiment": "experiment_4605_live_integration_scored_agent",
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "first_win_rate_integrated": 0.04,
        "first_win_rate_bare": 0.04,
        "first_win_delta": 0.0,
        "first_win_ci": [0.0, 0.0],
        "median_actions_to_first_levelup_integrated": None,
        "actions_delta": 0.0,
        "positive_control_passed": True,
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": True,
        "offline_reproduced": True,
    }


def _a3_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4606_levelup_selfplay",
        "honest_verdict": "complete: dc22_delta_identified_no_bank",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "target_game": "dc22",
        "registry_updated": True,
        "reproduction_gate": {"reproduced": False, "claimed_level": 2, "reached_level": 1},
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4607_refresh_submission_package",
        "honest_verdict": "complete: package_refreshed_unchanged_depth.",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 55,
        "count_delta": 0,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
        "refreshed_package_path": "results/experiment_4607_submission_package_operator_resubmit.json",
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4608_primitive_persist_transfer",
        "honest_verdict": "success: primitive_persisted_transfer_bp35_value_added",
        "verifier_is_oracle": False,
        "transfer_games": ["bp35", "dc22", "g50t"],
        "transfer_value_per_game": {
            "bp35": {"value_added": True, "trust_pass_added": True},
            "dc22": {"value_added": True, "trust_pass_added": True},
            "g50t": {"value_added": True, "trust_pass_added": True},
        },
        "new_levels_banked": 0,
        "offline_reproduced": {"new_levels_banked": 0},
    }


def _a6_integration() -> dict[str, Any]:
    return {
        "experiment": "experiment_4609_integration_gate",
        "honest_verdict": "complete: integration_no_clean_metric_bare_config_kept_honest_null",
        "verifier_is_oracle": False,
        "world_model_trust_pass_rate_integrated": 0.0,
        "world_model_trust_pass_rate_baseline": 0.0,
        "world_model_trust_pass_rate_delta_vs_baseline": 0.0,
        "first_win_rate_integrated": 0.0,
        "first_win_rate_bare": 0.0,
        "first_win_rate_delta_vs_bare": 0.0,
        "actions_delta_vs_bare": 0.0,
        "live_submittable_level_count_integrated": 55,
        "live_submittable_level_count_baseline": 55,
        "parity_test_green": True,
        "submitted_config_raised_metric_clean": False,
        "submitted_to_leaderboard": False,
    }


def _b1_metric() -> dict[str, Any]:
    return {
        "experiment": "experiment_4610_world_model_trust_pass_rate_metric",
        "honest_verdict": "success: world_model_trust_pass_rate_metric_helper_shipped_tests_green",
        "world_model_trust_pass_rate": 1.0,
        "trust_pass_numerator": 6,
        "trust_pass_denominator": 6,
        "world_model_trust_pass_rate_baseline": 0.0,
        "world_model_trust_pass_rate_delta": 1.0,
        "coheadline_block": {
            "generic_transfer_rate_over_variants": 0.04,
            "generic_transfer_ci": [0.0, 0.1],
            "action_efficiency_score": 1.0,
            "action_efficiency_ci": [1.0, 1.0],
            "median_actions_to_first_levelup": 12.0,
        },
    }


def _b2_guard() -> dict[str, Any]:
    return {
        "experiment": "experiment_4611_adversarial_verify_hardening",
        "honest_verdict": "success: adversarial_verify_hardened_tautology_carveout_plus_wm_trust_guard_tests_green.",
        "tautology_carveout_added": True,
        "wm_trust_guard_added": True,
        "regression_424_artifacts_unflagged": True,
        "tests_added": {"passed": True},
    }


def _artifacts() -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_world_model(),
        "A2": _a2_scored(),
        "A3": _a3_bank(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_integration(),
        "B1": _b1_metric(),
        "B2": _b2_guard(),
    }


def _live_flags() -> dict[str, list[dict[str, Any]]]:
    return {
        "A1": [{"kind": "DURATION_TOO_SHORT", "severity": "critical", "detail": "too fast"}],
        "B1": [
            {
                "kind": "WORLD_MODEL_TRUST_DEGENERACY",
                "severity": "critical",
                "detail": "verifier_is_oracle=None",
            }
        ],
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4614": True,
        "registry_yaml_loadable": True,
        "registry_reproducible_total_levels": 55,
        "offline_arcade": True,
        "upstream_artifacts_present": {name: True for name in mod.UPSTREAM_SOURCES},
        "missing_upstream_artifacts": [],
        "summarize_artifact_py_used_for_live_flags": True,
        "leaderboard_submission": False,
        "operator_only": True,
        "research_conductor_modified": False,
    }


def test_req_capstone_4614_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4614: OpenSpec declares the capstone scorecard fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4614" in spec
    assert "SCENARIO-CAPSTONE-4614" in spec
    assert "SCENARIO-CAPSTONE-4614-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4614_excludes_flagged_and_live_critical_headlines() -> None:
    """REQ-CAPSTONE-4614: flagged/live-critical A1/B1 values are audit-only."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "complete: pivot_characterized_capability_grew_55_to_55"
    assert artifact["world_model_trust_pass_rate"]["clean_value"] is None
    assert artifact["world_model_trust_pass_rate"]["quarantined_value"] == pytest.approx(1.0)
    assert artifact["world_model_trust_pass_rate"]["trust_pass_numerator"] == 6
    assert artifact["world_model_trust_pass_rate"]["trust_pass_denominator"] == 6
    assert artifact["world_model_trust_pass_rate"]["binary_gate_failures"] == "0/6"
    assert artifact["first_win_rate_scored"]["clean_value"] is None
    assert artifact["first_win_rate_scored"]["quarantined_value"] == pytest.approx(0.04)
    assert artifact["scorecard"]["headline"]["pivot_cracked_0_08_wall_clean"] is False
    assert artifact["scorecard"]["headline"]["scored_first_win_rate_raised_clean"] is False
    assert artifact["flagged_artifacts_handled"]["excluded_artifacts"] == [
        "results/experiment_4604_world_model_trust_energy.json",
        "results/experiment_4605_live_integration_scored_agent.json",
        "results/experiment_4610_world_model_trust_pass_rate_metric.json",
    ]
    assert mod.validate_artifact(artifact) == []


def test_scenario_capstone_4614_reports_bank_package_transfer_and_submit_status() -> None:
    """SCENARIO-CAPSTONE-4614: clean support metrics remain in the scorecard."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["reproducible_total_levels"] == 55
    assert artifact["reproducible_total_levels_delta"] == 0
    assert artifact["scorecard"]["A3"]["reproduced_levels"] == 0
    assert artifact["live_submittable_level_count"] == 55
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["A4"]["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["A5"]["value_added_games"] == ["bp35", "dc22", "g50t"]
    assert artifact["scorecard"]["generic_transfer"]["clean_value"] == pytest.approx(0.04)
    assert artifact["scorecard"]["action_efficiency"]["clean_value"] == pytest.approx(1.0)
    assert artifact["scorecard"]["A6"]["submitted_config_raised_metric_clean"] is False
    assert artifact["scorecard"]["B2"]["tautology_small_sample_carveout_applied"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["scorecard"]["verifier_is_oracle_claim_audit"]["all_included_value_claims_false"] is True


def test_req_capstone_4614_tautology_carveout_allows_real_kn_win() -> None:
    """REQ-CAPSTONE-4614: B2 shared-denominator carve-out prevents re-quarantine."""

    artifacts = _artifacts()
    artifacts["A1"] = _a1_world_model(flagged=False)
    artifacts["A2"] = {**_a2_scored(flagged=False), "first_win_rate_integrated": 0.08, "first_win_delta": 0.04}
    flags = {
        "A1": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "small shared-denominator k/N rates over denominator 6",
            }
        ],
        "A2": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "small shared-denominator first-win k/N rates over denominator 25",
            }
        ],
    }

    artifact = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name=flags,
        registry={"reproducible_total_levels": 56},
        preconditions_checked={**_preconditions(), "registry_reproducible_total_levels": 56},
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == "success: pivot_cracked_0.08_wall_trust_pass_6_first_win_up"
    assert artifact["world_model_trust_pass_rate"]["clean_value"] == pytest.approx(1.0)
    assert artifact["world_model_trust_pass_rate"]["headline_numbers_aggregated"] is True
    assert artifact["first_win_rate_scored"]["clean_value"] == pytest.approx(0.08)
    assert artifact["reproducible_total_levels"] == 56
    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["flagged_artifacts_handled"]["tautology_carveouts"][0]["name"] == "A1"
    assert artifact["flagged_artifacts_handled"]["tautology_carveouts"][1]["name"] == "A2"


def test_scenario_capstone_4614_run_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4614: run validates and writes the requested JSON."""

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

    artifact = mod.run(
        tmp_path,
        live_flags_by_name=_live_flags(),
        write=True,
        duration_s=0.001,
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert written["leaderboard_submission"] is False
    assert written["ready_for_operator_submit"] is True
    assert mod.validate_artifact(written) == []


def test_req_capstone_4614_validation_fails_closed() -> None:
    """REQ-CAPSTONE-4614: malformed scorecards cannot pass validation."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
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
            "world_model_trust_pass_rate": 1.0,
            "first_win_rate_scored": {"clean_value": "0.04"},
            "reproducible_total_levels": "55",
            "reproducible_total_levels_delta": "0",
            "live_submittable_level_count": "55",
            "ready_for_operator_submit": "true",
            "field_principles": {},
            "cited_upstream_artifacts": [],
            "preconditions_checked": [],
            "leaderboard_submission": True,
            "reproducibility_checksum": "bad",
        }
    )

    errors = mod.validate_artifact(broken)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "world_model_trust_pass_rate must be object" in errors
    assert "first_win_rate_scored.clean_value must be float or null" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "reproducible_total_levels_delta must be bare int" in errors
    assert "live_submittable_level_count must be bare int" in errors
    assert "ready_for_operator_submit must be bare bool" in errors
    assert "missing field principle for honest_verdict" in errors
    assert "cited_upstream_artifacts must be object" in errors
    assert "preconditions_checked must be object" in errors
    assert "leaderboard_submission must be false" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    missing_clean = dict(artifact)
    missing_clean["world_model_trust_pass_rate"] = {"quarantined_value": 1.0}
    missing_clean["reproducibility_checksum"] = mod.payload_checksum(missing_clean)
    assert "world_model_trust_pass_rate.clean_value missing" in mod.validate_artifact(missing_clean)

    missing_principles = dict(artifact)
    missing_principles["field_principles"] = None
    missing_principles["reproducibility_checksum"] = mod.payload_checksum(missing_principles)
    assert "field_principles missing" in mod.validate_artifact(missing_principles)

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(path=Path("/tmp/not-written.json"), artifact=broken)


def test_req_capstone_4614_defensive_helpers_and_blocked_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4614: helper branches stay deterministic and explicit."""

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

    class BadReader:
        @staticmethod
        def _live_flags(path: Path) -> list[dict[str, Any]]:
            raise RuntimeError(path)

    monkeypatch.setattr(mod, "artifact_reader", None)
    assert mod._live_flags(tmp_path / "x.json") == []
    monkeypatch.setattr(mod, "artifact_reader", BadReader)
    assert mod._live_flags(tmp_path / "x.json") == []

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
    fnr = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"positive_control_passed": False, "false_negative_risk_checked": True},
        exists=True,
        live_flags_by_name={},
        b2_active=False,
    )
    assert fnr["reason"] == "positive_control_failed"
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
    handled = mod._flagged_artifacts_handled(
        {"P": fnr, "F": false_negative, "G": gate_failed},
        {"tautology_carveout_added": True, "wm_trust_guard_added": True},
    )
    assert handled["positive_control_failed_artifacts"] == [
        {"name": "P", "artifact": "results/x.json"}
    ]
    assert handled["false_negative_risk_open_artifacts"] == [
        {"name": "F", "artifact": "results/x.json"}
    ]
    assert handled["failed_acceptance_gate_overrides"] == [
        {"name": "G", "artifact": "results/x.json", "failed_gates": ["acceptance_gate_main"]}
    ]
    assert mod._trust_counts(
        {
            "measurements": [
                {"new_trust_pass": True, "new_planner_used": True, "new_correct_changed_cells": 2},
                {"new_trust_pass": True, "new_planner_used": False, "new_correct_changed_cells": 2},
            ]
        },
        {},
    ) == (1, 2)

    from carnot.agentic import arc_solver_kit

    monkeypatch.setattr(
        arc_solver_kit,
        "offline_arcade",
        lambda: (_ for _ in ()).throw(RuntimeError("offline unavailable")),
    )
    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec_path = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text("REQ-CAPSTONE-4614\n", encoding="utf-8")
    registry_path = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text("reproducible_total_levels: 55\n", encoding="utf-8")
    failed_checks = mod.check_preconditions(tmp_path, statuses={})
    assert failed_checks["offline_arcade"] is False
    assert "offline_arcade_error" in failed_checks

    blocked = mod.build_artifact(
        artifacts={},
        live_flags_by_name={},
        registry={},
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        duration_s=0.001,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"

    good = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        registry={"reproducible_total_levels": 55},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    bad = dict(good)
    bad["ready_for_operator_submit"] = True
    bad["live_submittable_level_count"] = 33
    assert "ready_for_operator_submit requires count above 33" in mod.validate_artifact(bad)

    monkeypatch.setattr(mod, "build_artifact", lambda *args, **kwargs: broken_artifact(good))
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.run(tmp_path, write=False)


def broken_artifact(good: Mapping[str, Any]) -> dict[str, Any]:
    """Return one invalid artifact for the run validation branch."""

    bad = dict(good)
    bad["verifier_is_oracle"] = True
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    return bad
