"""Tests for Exp 4602 .424 capstone aggregation.

Spec refs: REQ-CAPSTONE-4602, SCENARIO-CAPSTONE-4602,
SCENARIO-CAPSTONE-4602-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4602_capstone_v424 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_flagged_positive() -> dict[str, Any]:
    return {
        "experiment": "experiment_4592_generation_completeness_wiring",
        "honest_verdict": "success: generation_completeness_winner_generated_2of25_above_1of25",
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
        "winner_generated_rate_with_wiring": 0.08,
        "winner_generated_rate_baseline": 0.04,
        "winner_generated_delta": 0.04,
        "generic_transfer_rate_with_wiring": 0.08,
        "generic_transfer_rate_baseline": 0.04,
        "transfer_delta": 0.04,
        "transfer_ci": [0.0, 0.12],
        "no_wiring_control_passed": True,
        "positive_control_passed": None,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": True,
        "chosen_submitted_config": "enable_wired_generation_dispatch",
        "offline_reproduced": True,
    }


def _a2_bank() -> dict[str, Any]:
    return {
        "experiment": "experiment_4593_levelup_selfplay",
        "honest_verdict": "success: ft09_L2_offline_reproduced",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "target_game": "ft09",
        "target_level": 2,
        "registry_updated": True,
        "registry_update": {
            "updated": True,
            "target_game": "ft09",
            "banked_levels": 1,
            "prior_game_levels": 1,
            "new_game_levels": 2,
            "prior_total_declared": 54,
            "new_total_declared": 55,
            "reconciled_total_delta": 1,
        },
        "reproduction_gate": {
            "claimed_level": 2,
            "reached_level": 2,
            "reproduced": True,
        },
        "verifier_checkpoint_updated": True,
    }


def _a3_false_negative_null() -> dict[str, Any]:
    return {
        "experiment": "experiment_4594_goal_energy_generation_prior",
        "honest_verdict": "complete: goal_energy_prior_no_value_honest_null_gap_sharpened",
        "verifier_is_oracle": False,
        "winner_generated_rate_with_energy": 0.0,
        "winner_generated_rate_no_energy": 0.0,
        "winner_generated_delta": 0.0,
        "generic_transfer_rate_with_energy": 0.0,
        "generic_transfer_rate_no_energy": 0.0,
        "actions_delta": 0.0,
        "no_energy_control_passed": True,
        "positive_control_passed": None,
        "false_negative_risk_checked": True,
        "solve_rate_preserved": True,
        "targeted_classes": [
            "keyboard_graph:systematic_bfs:variant_wired=True",
            "click_graph:diversity_graph_explore:variant_wired=True",
            "config_toggle:diversity_graph_explore:variant_wired=True",
        ],
        "chosen_submitted_config": "unchanged",
        "offline_reproduced": True,
    }


def _a4_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4595_refresh_submission_package",
        "honest_verdict": "success: package_refreshed_live_submittable_55_above_33",
        "verifier_is_oracle": False,
        "live_submittable_level_count": 55,
        "live_submittable_count_prev": 53,
        "count_delta": 2,
        "levels_folded_in": ["ar25", "ft09"],
        "refreshed_package_path": "results/experiment_4595_submission_package_operator_resubmit.json",
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _a5_transfer() -> dict[str, Any]:
    return {
        "experiment": "experiment_4596_primitive_persist_transfer",
        "honest_verdict": "success: primitive_persisted_transfer_ar25_value_added",
        "primitive_persisted": {"operator": "approach_dispatcher_operator"},
        "transfer_games": ["ar25", "cn04", "dc22"],
        "transfer_value_per_game": {
            "ar25": {"value_added": True, "winner_generated": True},
            "cn04": {"value_added": True, "winner_generated": True},
            "dc22": {"value_added": True, "winner_generated": True},
        },
        "new_levels_banked": 0,
        "offline_reproduced": {"new_levels_banked": 0},
    }


def _a6_flagged_integration() -> dict[str, Any]:
    return {
        "experiment": "experiment_4597_integration_gate",
        "honest_verdict": "success: integrated_live_submittable_55_above_33",
        "flagged_adversarial": True,
        "verifier_is_oracle": False,
        "winner_generated_rate_integrated": 0.04,
        "winner_generated_rate_baseline": 0.04,
        "generic_transfer_rate_integrated": 0.04,
        "generic_transfer_rate_baseline": 0.04,
        "generic_transfer_ci_integrated": [0.0, 0.12],
        "held_out_solve_rate": 0.04,
        "live_submittable_level_count_integrated": 55,
        "live_submittable_level_count_baseline": 33,
        "levers_integrated": [
            "A2_ft09_L2_banked_package_refresh",
            "A4_refreshed_live_submit_package",
        ],
        "ready_for_operator_submit": True,
        "positive_control_passed": None,
        "false_negative_risk_checked": True,
        "parity_green": True,
        "core_solves_preserved": {"passed": True},
        "upstream_lever_audit": {
            "A1": {"integrated": False, "reason": "flagged_adversarial_not_allowed_for_aggregation"},
            "A2": {"integrated": True, "reason": "ft09_L2_new_offline_reproduced_bank"},
            "A3": {"integrated": False, "reason": "no_admissible_goal_energy_metric_gain"},
            "A4": {"integrated": True, "reason": "refreshed_package_live_submittable_above_33"},
        },
    }


def _b1_flagged_coheadline() -> dict[str, Any]:
    return {
        "experiment": "experiment_4598_winner_generated_rate_metric",
        "honest_verdict": "shipped: winner_generated_rate_coheadline_wired",
        "flagged_adversarial": True,
        "winner_generated_rate": 0.04,
        "winner_generated_count": 1,
        "winner_generated_attempted_count": 25,
        "winner_generated_not_selected_count": 0,
        "generic_transfer_rate_over_variants": 0.04,
        "generic_transfer_ci": [0.0, 0.1],
        "generic_transfer_solved_count": 1,
        "action_efficiency_score": 1.0,
        "action_efficiency_ci": [1.0, 1.0],
        "reproducible_total_levels": 55,
        "live_submittable_level_count": 54,
        "reproducible_vs_submittable_gap": 1,
        "generation_vs_ranking_gap": 0.0,
    }


def _baseline() -> dict[str, Any]:
    return {"experiment": "arc3_live_submit", "live_total_levels": 33, "leaderboard_submitted": True}


def _artifacts() -> dict[str, dict[str, Any]]:
    return {
        "A1": _a1_flagged_positive(),
        "A2": _a2_bank(),
        "A3": _a3_false_negative_null(),
        "A4": _a4_package(),
        "A5": _a5_transfer(),
        "A6": _a6_flagged_integration(),
        "B1": _b1_flagged_coheadline(),
        "LIVE_BASELINE": _baseline(),
    }


def _live_flags() -> dict[str, list[dict[str, Any]]]:
    return {
        "A1": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "generic_transfer_rate_with_wiring=0.08 and winner_generated_rate_with_wiring=0.08",
            }
        ],
        "A3": [
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: positive_control_passed=None",
            }
        ],
        "A6": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "generic_transfer_rate_integrated=0.04 and held_out_solve_rate=0.04",
            },
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: positive_control_passed=None",
            },
        ],
        "B1": [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "generic_transfer_rate_over_variants=0.04 and winner_generated_rate=0.04",
            }
        ],
    }


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_or_opencode_md_read": True,
        "spec_has_req_4602": True,
        "registry_yaml_loadable": True,
        "registry_reproducible_total_levels": 55,
        "leaderboard_submission": False,
        "operator_only": True,
        "network_required": False,
        "research_conductor_modified": False,
    }


def test_req_capstone_4602_spec_declares_required_contract() -> None:
    """REQ-CAPSTONE-4602: OpenSpec declares the capstone fields before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4602",
        "SCENARIO-CAPSTONE-4602",
        "SCENARIO-CAPSTONE-4602-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4602_quarantines_flagged_generation_and_b1_metrics() -> None:
    """REQ-CAPSTONE-4602: flagged A1/B1 numbers cannot become clean headline wins."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["honest_verdict"] == (
        "complete: generation_wall_persists_residual_logged_capability_grew"
    )
    assert artifact["winner_generated_moved"]["moved"] is False
    assert artifact["winner_generated_moved"]["raw_reported_strictly_above_baseline"] is True
    assert artifact["winner_generated_moved"]["headline_numbers_aggregated"] is False
    assert artifact["generic_transfer_moved"]["moved"] is False
    assert artifact["generic_transfer_moved"]["a1"]["headline_numbers_aggregated"] is False
    assert artifact["winner_generated_rate"]["clean_value"] is None
    assert artifact["winner_generated_rate"]["quarantined_value"] == pytest.approx(0.04)
    assert artifact["generic_transfer_rate_over_variants"]["clean_value"] is None
    assert artifact["action_efficiency_score"]["clean_value"] is None
    assert set(artifact["flagged_artifacts_handled"]["excluded_artifacts"]) == {
        "results/experiment_4592_generation_completeness_wiring.json",
        "results/experiment_4597_integration_gate.json",
        "results/experiment_4598_winner_generated_rate_metric.json",
    }


def test_scenario_capstone_4602_reports_capability_growth_and_submit_ready_package() -> None:
    """SCENARIO-CAPSTONE-4602: clean A2/A4 evidence still reports capability and package wins."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["reproducible_total_levels_delta"] == 1
    assert artifact["reproducible_total_levels"] == 55
    assert artifact["scorecard"]["A2"]["reproducible_total_before"] == 54
    assert artifact["scorecard"]["A2"]["reproducible_total_after"] == 55
    assert artifact["live_submittable_level_count"] == 55
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["operator_submission_basis"]["last_submitted_scorecard_levels"] == 33
    assert artifact["operator_submission_basis"]["basis"] == "clean_a4_package_above_33"
    assert artifact["scorecard"]["A5"]["value_added_games"] == ["ar25", "cn04", "dc22"]
    assert artifact["scorecard"]["A6"]["submitted_config_raised_metric_clean"] is False
    assert mod.validate_artifact(artifact) == []


def test_req_capstone_4602_false_negative_null_and_null_delta_carveout() -> None:
    """REQ-CAPSTONE-4602: broken nulls stay open; B2 carve-out is diagnosis-only."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )

    assert artifact["goal_energy_helped"]["helped"] is False
    assert artifact["goal_energy_helped"]["null_is_clean"] is False
    assert artifact["goal_energy_helped"]["reason"] == "a3_false_negative_risk_open"
    assert artifact["flagged_artifacts_handled"]["positive_control_failed_guard"] == ["A3", "A6"]

    artifacts = _artifacts()
    artifacts["A1"] = {
        "flagged_adversarial": True,
        "honest_verdict": "complete: explicit_control_best_null",
        "efficiency_delta": 0.0,
        "null_delta_methodology_note": "control==best null-delta is the expected diagnosis.",
    }
    flags = _live_flags()
    flags["A1"] = [
        {
            "kind": "TAUTOLOGY",
            "severity": "critical",
            "detail": "routing_control=1.0 and routing_best=1.0",
        }
    ]
    carveout = mod.build_artifact(
        artifacts=artifacts,
        live_flags_by_name=flags,
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    assert carveout["flagged_artifacts_handled"]["null_delta_carveouts"][0]["artifact"] == (
        "results/experiment_4592_generation_completeness_wiring.json"
    )
    assert carveout["scorecard"]["A1"]["reason"] == (
        "flagged_null_delta_corrigendum_diagnosis_only"
    )


def test_scenario_capstone_4602_run_writes_deliverable(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4602: run validates and writes the requested JSON."""

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


def test_req_capstone_4602_validation_fails_closed() -> None:
    """REQ-CAPSTONE-4602: malformed capstones cannot pass as final results."""

    artifact = mod.build_artifact(
        artifacts=_artifacts(),
        live_flags_by_name=_live_flags(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    broken = dict(artifact)
    broken.update(
        {
            "honest_verdict": "maybe",
            "inference_substrate": "wrong",
            "winner_generated_moved": {"moved": "yes"},
            "generic_transfer_moved": [],
            "goal_energy_helped": {"helped": "no"},
            "reproducible_total_levels_delta": "1",
            "live_submittable_level_count": "55",
            "winner_generated_rate": 0.04,
            "action_efficiency_score": {"clean_value": "1.0"},
            "generic_transfer_rate_over_variants": {"quarantined_value": 0.04},
            "ready_for_operator_submit": "true",
            "leaderboard_submission": True,
            "field_principles": {},
            "coheadline_metrics": {},
            "reproducibility_checksum": "bad",
        }
    )

    errors = mod.validate_artifact(broken)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "inference_substrate mismatch" in errors
    assert "winner_generated_moved.moved must be bare bool" in errors
    assert "generic_transfer_moved must be object" in errors
    assert "goal_energy_helped.helped must be bare bool" in errors
    assert "reproducible_total_levels_delta must be bare int" in errors
    assert "live_submittable_level_count must be bare int" in errors
    assert "winner_generated_rate must be object" in errors
    assert "action_efficiency_score.clean_value must be float or null" in errors
    assert "generic_transfer_rate_over_variants.clean_value missing" in errors
    assert "ready_for_operator_submit must be bare bool" in errors
    assert "leaderboard_submission must be false" in errors
    assert "missing field principle for honest_verdict" in errors
    assert "coheadline_metrics.winner_generated_rate must match top-level field" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

    with pytest.raises(ValueError):
        mod.write_artifact(artifact=broken)


def test_req_capstone_4602_defensive_helpers_and_alternate_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4602: helper branches stay deterministic and explicit."""

    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json) == {}

    assert mod._read_yaml(tmp_path / "missing.yaml") == {}
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("[", encoding="utf-8")
    assert mod._read_yaml(bad_yaml) == {}
    scalar_yaml = tmp_path / "scalar.yaml"
    scalar_yaml.write_text("1\n", encoding="utf-8")
    assert mod._read_yaml(scalar_yaml) == {}
    assert mod._file_sha256(tmp_path / "missing.bin") is None
    assert mod._as_int(True, 7) == 7
    assert mod._as_int("x", 7) == 7
    assert mod._as_float(True, 0.5) == pytest.approx(0.5)
    assert mod._as_float("x", 0.5) == pytest.approx(0.5)
    assert mod._float_or_none(True) is None
    assert mod._float_or_none("x") is None

    class GoodReader:
        @staticmethod
        def _live_flags(path: Path) -> list[object]:
            return [{"kind": "OK"}, "skip"]

        @staticmethod
        def classify_known_false_positive_null_delta(
            artifact: Mapping[str, Any],
            flags: list[dict[str, Any]],
        ) -> str:
            return "not-a-mapping"

    class BadReader:
        @staticmethod
        def _live_flags(path: Path) -> list[dict[str, Any]]:
            raise RuntimeError(path)

        @staticmethod
        def classify_known_false_positive_null_delta(
            artifact: Mapping[str, Any],
            flags: list[dict[str, Any]],
        ) -> None:
            raise RuntimeError((artifact, flags))

    monkeypatch.setattr(mod, "artifact_reader", None)
    assert mod._live_flags(tmp_path / "x.json") == []
    assert mod._null_delta_corrigendum({}, []) is None
    monkeypatch.setattr(mod, "artifact_reader", GoodReader)
    assert mod._live_flags(tmp_path / "x.json") == [{"kind": "OK"}]
    assert mod._null_delta_corrigendum({}, []) is None
    monkeypatch.setattr(mod, "artifact_reader", BadReader)
    assert mod._live_flags(tmp_path / "x.json") == []
    assert mod._null_delta_corrigendum({}, []) is None

    source = mod.SourceSpec("X", "results/x.json", "fixture")
    missing_status = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={},
        exists=False,
        live_flags_by_name={},
    )
    assert missing_status["reason"] == "missing"
    gate_status = mod._source_status(
        name="X",
        source=source,
        root=tmp_path,
        artifact={"gate_main": False},
        exists=True,
        live_flags_by_name={},
    )
    assert gate_status["reason"] == "failed_acceptance_gate"
    handled = mod._flagged_artifacts_handled({"X": gate_status, "LIVE_BASELINE": missing_status})
    assert handled["failed_acceptance_gate_overrides"] == [
        {"name": "X", "artifact": "results/x.json", "failed_gates": ["gate_main"]}
    ]

    assert mod._a2_repro_delta({}, False) == (0, 0, 0)
    assert mod._a2_repro_delta({}, True) == (0, 0, 0)
    assert mod._a2_repro_delta(
        {
            "offline_reproduced": True,
            "registry_update": {"prior_total_declared": 1, "new_total_declared": 3, "updated": True},
            "reproduction_gate": {"reproduced": True},
        },
        True,
    ) == (1, 3, 2)
    assert mod._a2_repro_delta(
        {
            "offline_reproduced": False,
            "registry_update": {"prior_total_declared": 1, "new_total_declared": 3, "updated": False},
            "reproduction_gate": {"reproduced": False},
        },
        True,
    ) == (1, 3, 0)

    clean = _artifacts()
    clean["A1"] = {
        **_a1_flagged_positive(),
        "flagged_adversarial": False,
        "winner_generated_delta_ci": [0.01, 0.07],
        "transfer_ci": [0.01, 0.07],
    }
    clean["B1"] = {**_b1_flagged_coheadline(), "flagged_adversarial": False}
    success = mod.build_artifact(
        artifacts=clean,
        live_flags_by_name={},
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    assert success["honest_verdict"].startswith("success:")
    assert success["winner_generated_moved"]["moved"] is True
    assert success["generic_transfer_moved"]["moved"] is True
    assert success["winner_generated_rate"]["clean_value"] == pytest.approx(0.04)

    no_growth = _artifacts()
    no_growth["A2"] = {
        **_a2_bank(),
        "offline_reproduced": False,
        "registry_update": {**_a2_bank()["registry_update"], "updated": False, "reconciled_total_delta": 0},
        "reproduction_gate": {"reproduced": False},
    }
    no_growth_artifact = mod.build_artifact(
        artifacts=no_growth,
        live_flags_by_name=_live_flags(),
        preconditions_checked=_preconditions(),
        duration_s=0.001,
    )
    assert no_growth_artifact["honest_verdict"] == (
        "complete: generation_wall_persists_residual_logged_no_capability_growth"
    )

    broken_metric = dict(success)
    broken_metric["action_efficiency_score"] = {
        "clean_value": None,
        "quarantined_value": "bad",
        "source": None,
        "included_in_clean_headline": "yes",
    }
    broken_metric["coheadline_metrics"] = {
        **success["coheadline_metrics"],
        "action_efficiency_score": broken_metric["action_efficiency_score"],
    }
    metric_errors = mod.validate_artifact(broken_metric)
    assert "action_efficiency_score.quarantined_value must be float or null" in metric_errors
    assert "action_efficiency_score.source must be string" in metric_errors
    assert "action_efficiency_score.included_in_clean_headline must be bare bool" in metric_errors

    malformed = dict(success)
    malformed["preconditions_checked"] = []
    malformed["field_principles"] = None
    malformed["reproducibility_checksum"] = "sha256:bad"
    malformed_errors = mod.validate_artifact(malformed)
    assert "preconditions_checked must be object" in malformed_errors
    assert "field_principles missing" in malformed_errors
    assert "reproducibility_checksum mismatch" in malformed_errors

    run_artifact = mod.run(tmp_path, live_flags_by_name={}, write=False)
    assert run_artifact["duration_s"] >= 0.0001
    bad_run = dict(success)
    bad_run["winner_generated_moved"] = {"moved": "yes"}
    monkeypatch.setattr(mod, "build_artifact", lambda *args, **kwargs: bad_run)
    with pytest.raises(ValueError, match="winner_generated_moved.moved"):
        mod.run(tmp_path, write=False)
