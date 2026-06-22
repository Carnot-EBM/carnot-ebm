"""Tests for Exp 4597 ARC integration gate.

Spec refs: REQ-CAPSTONE-4597, SCENARIO-CAPSTONE-4597,
SCENARIO-CAPSTONE-4597-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4597_integration_gate as mod
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _a1_flagged_positive() -> dict[str, Any]:
    return {
        "experiment": "experiment_4592_generation_completeness_wiring",
        "flagged_adversarial": True,
        "honest_verdict": "success: generation_completeness_winner_generated_2of25_above_1of25",
        "winner_generated_rate_with_wiring": 0.08,
        "winner_generated_rate_baseline": 0.04,
        "winner_generated_delta": 0.04,
        "generic_transfer_rate_with_wiring": 0.08,
        "generic_transfer_rate_baseline": 0.04,
        "transfer_delta": 0.04,
        "no_wiring_control_passed": True,
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
        "registry_update": {
            "updated": True,
            "target_game": "ft09",
            "banked_levels": 1,
            "prior_game_levels": 1,
            "new_game_levels": 2,
            "reconciled_total_delta": 1,
        },
        "reproduction_gate": {
            "claimed_level": 2,
            "reached_level": 2,
            "reproduced": True,
        },
    }


def _a3_null() -> dict[str, Any]:
    return {
        "experiment": "experiment_4594_goal_energy_generation_prior",
        "honest_verdict": "complete: goal_energy_prior_no_value_honest_null_gap_sharpened",
        "winner_generated_rate_with_energy": 0.0,
        "winner_generated_rate_no_energy": 0.0,
        "winner_generated_delta": 0.0,
        "generic_transfer_rate_with_energy": 0.0,
        "generic_transfer_rate_no_energy": 0.0,
        "no_energy_control_passed": True,
        "solve_rate_preserved": True,
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
        "refreshed_package_path": mod.INTEGRATED_PACKAGE_RELATIVE_PATH,
        "ready_for_operator_submit": True,
        "offline_reproduced": True,
    }


def _transfer_measurement() -> dict[str, Any]:
    attempts = [
        {
            "game": "lp85",
            "attempted": True,
            "solved": True,
            "winner_generated": True,
            "reached_level": 1,
        },
        {"game": "sp80", "attempted": True, "solved": False, "reached_level": 0},
    ]
    return {
        "generic_transfer_rate_over_variants": 0.5,
        "generic_transfer_ci": [0.1, 0.9],
        "variant_attempts": attempts,
        "variant_attempts_count": 2,
        "variant_solved_count": 1,
    }


def _live_metrics() -> dict[str, Any]:
    return {
        "live_submittable_level_count": 55,
        "reproducible_total_levels": 55,
        "reproducible_vs_submittable_gap": 0,
        "refreshed_package_path": mod.INTEGRATED_PACKAGE_RELATIVE_PATH,
        "live_submittable_subset_of_reproducible": True,
        "per_game_live_submittable": [
            {
                "game": "ft09",
                "submittable_level": 2,
                "included": True,
                "registry_reproduced_level": 2,
                "offline_reproduced_level": 2,
                "has_replayable_trajectory": True,
                "has_env_adaptive_resolver": True,
                "env_matchable": True,
            }
        ],
    }


def test_req_capstone_4597_spec_declares_integration_contract() -> None:
    """REQ-CAPSTONE-4597: OpenSpec declares the integration artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4597",
        "SCENARIO-CAPSTONE-4597",
        "SCENARIO-CAPSTONE-4597-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4597_rejects_flagged_positive_allows_only_null_tautology() -> None:
    """REQ-CAPSTONE-4597: flagged positive upstream metrics cannot be aggregated."""

    assert mod.artifact_admissible_for_aggregation(_a1_flagged_positive())[0] is False

    allowed_null = {
        "flagged_adversarial": True,
        "transfer_delta": 0.0,
        "null_delta_methodology_note": "control==best null-delta is an explicit null.",
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "detail": "control==best null-delta"}],
    }
    assert mod.artifact_admissible_for_aggregation(allowed_null)[0] is True

    missing_note = dict(allowed_null, null_delta_methodology_note="")
    assert mod.artifact_admissible_for_aggregation(missing_note)[0] is False


def test_req_capstone_4597_selects_only_admissible_metric_raisers() -> None:
    """REQ-CAPSTONE-4597: A2/A4 integrate while flagged A1 and null A3 stay out."""

    audit = mod.audit_upstream_levers(
        a1_artifact=_a1_flagged_positive(),
        a2_artifact=_a2_bank(),
        a3_artifact=_a3_null(),
        a4_artifact=_a4_package(),
    )

    assert audit["levers_integrated"] == [
        "A2_ft09_L2_banked_package_refresh",
        "A4_refreshed_live_submit_package",
    ]
    assert audit["isolated_deltas"]["live_submittable"] == {"A2": 1, "A4": 2}
    assert audit["isolated_deltas"]["winner_generated"] == {}
    assert audit["isolated_deltas"]["generic_transfer"] == {}
    assert audit["upstream_lever_audit"]["A1"]["integrated"] is False
    assert audit["upstream_lever_audit"]["A3"]["integrated"] is False
    assert audit["disallowed_adversarial_inputs"][0]["lever"] == "A1"


def test_scenario_capstone_4597_builds_success_artifact_with_required_metrics() -> None:
    """SCENARIO-CAPSTONE-4597: artifact reports all three integrated metrics and additivity."""

    audit = mod.audit_upstream_levers(
        a1_artifact=_a1_flagged_positive(),
        a2_artifact=_a2_bank(),
        a3_artifact=_a3_null(),
        a4_artifact=_a4_package(),
    )
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        audit=audit,
        transfer_measurement=_transfer_measurement(),
        live_metrics=_live_metrics(),
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        parity_green=True,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: integrated_live_submittable_55_above_33"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["winner_generated_rate_integrated"] == pytest.approx(0.5)
    assert artifact["generic_transfer_rate_integrated"] == pytest.approx(0.5)
    assert artifact["live_submittable_level_count_integrated"] == 55
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["parity_green"] is True
    assert artifact["additivity_checked"]["live_submittable"]["interaction_delta"] == 19
    assert artifact["core_solves_preserved"]["passed"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4597_pure_helper_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4597: pure helper branches classify edge cases without fabrication."""

    assert mod._read_json(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert mod._read_json(bad_json) == {}
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert mod._read_json(list_json) == {}
    out_json = tmp_path / "nested" / "out.json"
    mod._write_json(out_json, {"ok": True})
    assert mod._read_json(out_json) == {"ok": True}

    assert mod._as_int(True, 7) == 7
    assert mod._as_int("x", 4) == 4
    assert mod._as_float(False, 1.5) == 1.5
    assert mod._as_float("x", 2.5) == 2.5
    assert mod._public_games(tmp_path) == []
    (tmp_path / "environment_files" / "b").mkdir(parents=True)
    (tmp_path / "environment_files" / "a").mkdir()
    assert mod._public_games(tmp_path) == ["a", "b"]

    invalid_null = {
        "flagged_adversarial": True,
        "transfer_delta": 0.0,
        "null_delta_methodology_note": "null delta",
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "detail": "control matched best"}],
    }
    assert mod.artifact_admissible_for_aggregation(invalid_null)[0] is False
    assert mod.artifact_admissible_for_aggregation({"positive_control_failed": True}) == (
        False,
        "positive_control_failed",
    )
    assert mod._a1_integrates({}) == (False, "no_admissible_generation_metric_gain", {})
    assert mod._a2_integrates({"positive_control_failed": True}) == (
        False,
        "positive_control_failed",
        0,
    )
    assert mod._a2_integrates({}) == (False, "missing_registry_or_reproduction_gate", 0)
    assert mod._a3_integrates({"positive_control_failed": True}) == (
        False,
        "positive_control_failed",
        {},
    )
    assert mod._a4_integrates({"positive_control_failed": True}) == (
        False,
        "positive_control_failed",
        0,
    )

    a1_positive = {
        **_a1_flagged_positive(),
        "flagged_adversarial": False,
        "actions_delta": 1.0,
    }
    a3_positive = {
        **_a3_null(),
        "winner_generated_delta": 0.25,
        "generic_transfer_rate_with_energy": 0.25,
        "generic_transfer_rate_no_energy": 0.0,
        "chosen_submitted_config": "enable_goal_energy_generation_prior",
    }
    audit = mod.audit_upstream_levers(
        a1_artifact=a1_positive,
        a2_artifact={"registry_update": {}, "reproduction_gate": {}},
        a3_artifact=a3_positive,
        a4_artifact={"live_submittable_level_count": 1, "count_delta": 0},
    )
    assert audit["levers_integrated"] == [
        "A1_wired_generation_dispatch",
        "A3_goal_energy_generation_prior",
    ]
    assert audit["upstream_lever_audit"]["A2"]["reason"] == "no_new_offline_reproduced_bank"
    assert audit["upstream_lever_audit"]["A4"]["reason"] == "no_admissible_refreshed_package_gain"

    assert mod._winner_generated_rate([]) == 0.0
    assert mod._held_out_deepest_by_game(
        [
            {"variant_signature": "aa~color01", "reached_level": 2},
            {"variant_signature": "", "reached_level": 7},
        ]
    ) == {"aa": 2}
    assert mod._per_game_deepest_from_live_metrics({"per_game_live_submittable": "bad"}) == {}
    assert mod._per_game_deepest_from_live_metrics(
        {"per_game_live_submittable": ["bad", {"game": "g", "submittable_level": "3"}]}
    ) == {"g": 3}
    assert mod._package_levels({"package_manifest": "bad"}) == {}
    assert mod._package_levels({"package_manifest": ["bad", {"game": "g", "levels": "2"}]}) == {
        "g": 2
    }

    baseline = tmp_path / "baseline.json"
    integrated = tmp_path / "integrated.json"
    mod._write_json(baseline, {"package_manifest": [{"game": "g", "levels": 2}]})
    mod._write_json(integrated, {"package_manifest": [{"game": "g", "levels": 1}]})
    preservation = mod.package_core_preservation(
        tmp_path,
        baseline_package_path="baseline.json",
        integrated_package_path="integrated.json",
    )
    assert preservation["passed"] is False
    assert preservation["dropped_games"] == ["g"]

    assert (
        mod._verdict(winner_rate=0.0, transfer_rate=0.0, live_count=0, ready=False)
        == "complete: no_lever_raises_a_metric_honest_null"
    )
    assert mod._verdict(winner_rate=0.1, transfer_rate=0.0, live_count=0, ready=True).startswith(
        "success: integrated_winner_generated"
    )
    assert mod._verdict(winner_rate=0.0, transfer_rate=0.1, live_count=0, ready=True).startswith(
        "success: integrated_generic_transfer"
    )
    assert (
        mod._verdict(winner_rate=0.0, transfer_rate=0.0, live_count=0, ready=True)
        == "complete: no_lever_raises_a_metric_honest_null"
    )

    assert mod.first_precondition_miss({"AGENTS.md": False}) == "AGENTS.md"
    ok_preconditions = {
        "AGENTS.md": True,
        "CODEX.md": True,
        "offline_arcade": True,
        "a1_artifact_present": True,
        "a2_artifact_present": True,
        "a3_artifact_present": True,
        "a4_artifact_present": True,
        "integrated_package_present": True,
        "spec_has_req_4597": True,
        "leaderboard_submission": True,
    }
    assert mod.first_precondition_miss(ok_preconditions) == "leaderboard_submission"
    ok_preconditions["leaderboard_submission"] = False
    assert mod.first_precondition_miss(ok_preconditions) is None

    bad = {"honest_verdict": "working", "ready_for_operator_submit": True}
    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "ready_for_operator_submit requires parity_green" in errors
    assert "reproducibility_checksum must be sha256-prefixed hex" in errors
    with pytest.raises(ValueError):
        mod.write_artifact(tmp_path, bad)

    valid = mod.build_artifact(
        preconditions_checked={"ok": True, "offline_arcade": True},
        audit=mod.audit_upstream_levers(
            a1_artifact=_a1_flagged_positive(),
            a2_artifact=_a2_bank(),
            a3_artifact=_a3_null(),
            a4_artifact=_a4_package(),
        ),
        transfer_measurement=_transfer_measurement(),
        live_metrics=_live_metrics(),
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
        parity_green=True,
        duration_s=0.1,
    )
    mod.write_artifact(tmp_path, valid)
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
