"""Tests for Exp 4585 ARC integration gate.

Spec refs: REQ-CAPSTONE-4585, SCENARIO-CAPSTONE-4585,
SCENARIO-CAPSTONE-4585-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4585_integration_gate as mod
from carnot.agentic.arc_competition_agent import SUBMITTED_AGENT_CONFIG


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _a1_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "success: live_submittable_count_53_above_33",
        "live_submittable_count_baseline": 33,
        "live_submittable_level_count": 53,
        "count_delta": 20,
        "ready_for_operator_submit": True,
        "refreshed_package_path": mod.A1_PACKAGE_RELATIVE_PATH,
        "offline_reproduced": {"ar25": 1, "sc25": 52},
    }


def _a2_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "success: ar25_L2_offline_reproduced",
        "offline_reproduced": True,
        "registry_update": {
            "updated": True,
            "target_game": "ar25",
            "banked_levels": 1,
            "prior_game_levels": 1,
            "new_game_levels": 2,
            "reconciled_total_delta": 1,
        },
        "solution_labels": ["3", "2", "2"],
    }


def _flagged_a3_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: feature_router_no_value_honest_null_transfer_gap_sharpened",
        "flagged_adversarial": True,
        "transfer_delta": 0.0,
        "random_route_control_passed": False,
        "null_delta_methodology_note": "transfer_delta==0.0 is an honest no-value null.",
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "detail": "control==best null-delta"},
            {"kind": "FALSE_NEGATIVE_RISK", "detail": "control failed"},
        ],
    }


def _flagged_a4_artifact() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: diversity_floor_no_transfer_honest_null_gap_sharpened",
        "flagged_adversarial": True,
        "firstwin_delta": 0,
        "null_delta_methodology_note": "firstwin_delta==0 is an honest no-transfer null.",
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "detail": "control==best null-delta"},
            {"kind": "FALSE_NEGATIVE_RISK", "detail": "control failed"},
        ],
    }


def _source_package() -> dict[str, Any]:
    return {
        "experiment": "experiment_4580_submission_package_live_gap_close",
        "schema": "carnot.exp4580.submission_package.v1",
        "claimed_total_levels": 53,
        "operator_only": True,
        "submitted_to_leaderboard": False,
        "package_manifest": [
            {
                "game": "ar25",
                "levels": 1,
                "offline_reproduced_level": 1,
                "registry_reproduced_level": 1,
                "trajectory_path": "results/arc3_live_banked_trajectories/ar25.json",
                "action_count": 15,
                "source": "results/experiment_4339_e3_explore_verify_plan_ar25.json",
                "env_matched": True,
                "env_match_basis": "offline_fresh_replay_or_env_adaptive_proxy",
                "adaptive_solver": "",
                "adaptive_labels": [],
                "claim_capped": False,
            },
            {
                "game": "sc25",
                "levels": 52,
                "offline_reproduced_level": 52,
                "registry_reproduced_level": 52,
                "trajectory_path": "results/arc3_live_banked_trajectories/sc25.json",
                "action_count": 102,
                "source": "results/experiment_4468_bank_sc25_provisional_levels.json",
                "env_matched": True,
                "env_match_basis": "offline_fresh_replay_or_env_adaptive_proxy",
                "adaptive_solver": "sc25_dynamic_cast_grid_origin_step",
                "adaptive_labels": ["move1"],
                "claim_capped": False,
            },
        ],
    }


def test_req_capstone_4585_spec_declares_integration_gate_contract() -> None:
    """REQ-CAPSTONE-4585: OpenSpec declares the integration artifact schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4585",
        "SCENARIO-CAPSTONE-4585",
        "SCENARIO-CAPSTONE-4585-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4585_adversarial_gate_allows_only_null_tautology() -> None:
    """REQ-CAPSTONE-4585: flagged upstreams are excluded unless only a null tautology is present."""

    assert mod.artifact_admissible_for_aggregation(_a1_artifact())[0] is True
    assert mod.artifact_admissible_for_aggregation(_flagged_a3_artifact())[0] is False

    allowed_null = {
        "flagged_adversarial": True,
        "transfer_delta": 0.0,
        "null_delta_methodology_note": "transfer_delta==0.0 is an explicit null-delta note.",
        "corrigendum_pending": [{"kind": "TAUTOLOGY", "detail": "control==best null-delta"}],
    }
    assert mod.artifact_admissible_for_aggregation(allowed_null)[0] is True

    missing_note = dict(allowed_null, null_delta_methodology_note="")
    assert mod.artifact_admissible_for_aggregation(missing_note)[0] is False


def test_req_capstone_4585_selects_only_real_positive_winners() -> None:
    """REQ-CAPSTONE-4585: A1/A2 integrate while flagged A3/A4 nulls stay out."""

    audit = mod.audit_upstream_levers(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        a3_artifact=_flagged_a3_artifact(),
        a4_artifact=_flagged_a4_artifact(),
    )

    assert audit["levers_integrated"] == [
        "A1_refreshed_live_submit_package",
        "A2_ar25_L2_banked_package_refresh",
    ]
    assert audit["isolated_deltas"]["live_submittable"] == {"A1": 20, "A2": 1}
    assert audit["isolated_deltas"]["generic_transfer"] == {}
    assert {row["lever"] for row in audit["disallowed_adversarial_inputs"]} == {"A3", "A4"}
    assert audit["upstream_lever_audit"]["A3"]["integrated"] is False


def test_scenario_capstone_4585_refreshes_package_with_ar25_l2_bank(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4585: integrated package preserves CORE rows and lifts ar25 to L2."""

    _write_json(
        tmp_path / mod.AR25_L2_SOURCE_RELATIVE_PATH,
        {
            "game": "ar25",
            "offline_reproduced": True,
            "reached_level": 2,
            "reproduced_levels": 2,
            "solution": [{"action": 3}, {"action": 2}, {"action": 2}],
        },
    )

    summary = mod.refresh_integrated_package(
        tmp_path,
        source_package=_source_package(),
        levers_integrated=[
            "A1_refreshed_live_submit_package",
            "A2_ar25_L2_banked_package_refresh",
        ],
    )

    assert summary["package_path"] == mod.INTEGRATED_PACKAGE_RELATIVE_PATH
    assert summary["claimed_total_levels"] == 54
    assert summary["core_solves_preserved"]["passed"] is True
    assert summary["per_game_deepest_level_integrated"]["ar25"] == 2
    assert summary["per_game_deepest_level_integrated"]["sc25"] == 52

    package = json.loads(
        (tmp_path / mod.INTEGRATED_PACKAGE_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    ar25 = next(row for row in package["package_manifest"] if row["game"] == "ar25")
    assert ar25["levels"] == 2
    assert ar25["source"] == mod.AR25_L2_SOURCE_RELATIVE_PATH
    assert ar25["action_count"] == 3

    trajectory = json.loads((tmp_path / ar25["trajectory_path"]).read_text(encoding="utf-8"))
    assert trajectory["action_count"] == 3
    assert trajectory["source"] == mod.AR25_L2_SOURCE_RELATIVE_PATH


def test_scenario_capstone_4585_artifact_schema_success_and_null() -> None:
    """SCENARIO-CAPSTONE-4585: artifact validates success and honest-null outcomes."""

    audit = mod.audit_upstream_levers(
        a1_artifact=_a1_artifact(),
        a2_artifact=_a2_artifact(),
        a3_artifact=_flagged_a3_artifact(),
        a4_artifact=_flagged_a4_artifact(),
    )
    package_summary = {
        "package_path": mod.INTEGRATED_PACKAGE_RELATIVE_PATH,
        "claimed_total_levels": 54,
        "per_game_deepest_level_integrated": {"ar25": 2, "sc25": 52},
        "core_solves_preserved": {
            "passed": True,
            "baseline_core_games": ["ar25", "sc25"],
            "dropped_games": [],
        },
    }
    transfer = {
        "generic_transfer_rate_over_variants": 0.04,
        "generic_transfer_ci": [0.0, 0.12],
        "variant_attempts_count": 2,
        "variant_solved_count": 0,
        "variant_attempts": [
            {"game": "heldout1", "attempted": True, "solved": False, "reached_level": 0},
            {"game": "heldout2", "attempted": True, "solved": False, "reached_level": 0},
        ],
    }

    # NOTE (2026-07-12): parity_green tests exp4585's OWN computation logic
    # (does its package_path match submitted_agent_config's
    # live_submit_package_path) -- pin a SYNTHETIC config for that field
    # rather than importing today's real, live SUBMITTED_AGENT_CONFIG
    # directly, so this test verifies the LOGIC deterministically instead of
    # being coupled to whichever experiment's package happens to be the
    # live one at any given moment (a later experiment, exp4643, has since
    # superseded exp4585's own package as the operative one -- real,
    # legitimate history, not something this logic test should depend on).
    synthetic_config = {
        **SUBMITTED_AGENT_CONFIG,
        "live_submit_package_path": mod.INTEGRATED_PACKAGE_RELATIVE_PATH,
    }

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        audit=audit,
        package_summary=package_summary,
        transfer_measurement=transfer,
        submitted_agent_config=synthetic_config,
        duration_s=0.1,
    )

    assert artifact["honest_verdict"] == "success: integrated_live_submittable_54_above_33"
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["additivity_checked"]["live_submittable"]["interaction_delta"] == 0
    assert artifact["core_solves_preserved"]["passed"] is True
    assert artifact["parity_green"] is True
    assert mod.artifact_schema_errors(artifact) == []

    null_artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        audit={
            **audit,
            "levers_integrated": [],
            "isolated_deltas": {"live_submittable": {}, "generic_transfer": {}},
        },
        package_summary={**package_summary, "claimed_total_levels": mod.LIVE_SUBMITTABLE_BASELINE},
        transfer_measurement={
            **transfer,
            "generic_transfer_rate_over_variants": mod.GENERIC_TRANSFER_BASELINE,
        },
        submitted_agent_config=synthetic_config,
        duration_s=0.1,
    )
    assert null_artifact["honest_verdict"] == "complete: no_lever_raises_a_metric_honest_null"
    assert null_artifact["ready_for_operator_submit"] is False
    assert mod.artifact_schema_errors(null_artifact) == []


def test_scenario_capstone_4585_run_writes_requested_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4585: run writes the integration JSON and package."""

    (tmp_path / "AGENTS.md").write_text("# AGENTS\n", encoding="utf-8")
    (tmp_path / "CODEX.md").write_text("# CODEX\n", encoding="utf-8")
    spec = tmp_path / "openspec" / "capabilities" / "capstone" / "spec.md"
    spec.parent.mkdir(parents=True)
    spec.write_text(SPEC_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    _write_json(tmp_path / mod.A1_RELATIVE_PATH, _a1_artifact())
    _write_json(tmp_path / mod.A2_RELATIVE_PATH, _a2_artifact())
    _write_json(tmp_path / mod.A3_RELATIVE_PATH, _flagged_a3_artifact())
    _write_json(tmp_path / mod.A4_RELATIVE_PATH, _flagged_a4_artifact())
    _write_json(tmp_path / mod.A1_PACKAGE_RELATIVE_PATH, _source_package())
    _write_json(
        tmp_path / mod.AR25_L2_SOURCE_RELATIVE_PATH,
        {
            "offline_reproduced": True,
            "reached_level": 2,
            "solution": [{"action": 3}, {"action": 2}, {"action": 2}],
        },
    )

    def variant_runner(game: str, spec: Mapping[str, Any], budget: int) -> Mapping[str, Any]:
        del spec, budget
        return {
            "game": game,
            "attempted": True,
            "solved": game == "heldout1",
            "reached_level": int(game == "heldout1"),
        }

    artifact = mod.run(
        tmp_path,
        offline_arcade_checker=lambda: True,
        public_games=["heldout1", "heldout2"],
        variant_ids=[1],
        variant_runner=variant_runner,
        n_bootstrap=0,
        now=iter([10.0, 10.25]).__next__,
    )

    assert artifact["honest_verdict"] == "success: integrated_live_submittable_54_above_33"
    assert artifact["generic_transfer_rate_integrated"] == 0.5
    assert artifact["held_out_solve_rate"] == 0.5
    # NOTE (2026-07-12): run() always reads the LIVE SUBMITTED_AGENT_CONFIG (no
    # injectable override), and a later experiment (exp4643) has since
    # replaced exp4585's own package as the operative live submission
    # package -- a real, legitimate history, not drift. This asserts the
    # artifact self-consistently embeds whatever IS currently live, not a
    # frozen comparison against exp4585's own historical constant.
    assert (
        artifact["submitted_agent_config"]["live_submit_package_path"]
        == (SUBMITTED_AGENT_CONFIG["live_submit_package_path"])
    )
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.INTEGRATED_PACKAGE_RELATIVE_PATH).exists()

    with pytest.raises(ValueError, match="missing required field"):
        mod.write_artifact({}, root=tmp_path)
