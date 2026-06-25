"""Tests for Exp 4740 held-out first-win readiness.

Spec refs: REQ-CAPSTONE-4740, SCENARIO-CAPSTONE-4740,
SCENARIO-CAPSTONE-4740-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4740-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4740_held_out_first_win_readiness as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _proxy(
    *,
    first_win_rate: float = 0.04,
    ci_lower: float = 0.0,
    attempts: int = 100,
    multi_level_rate: float = 0.0,
) -> JsonDict:
    solved = int(round(first_win_rate * attempts))
    deepened = int(round(multi_level_rate * attempts))
    rows = [
        {
            "attempted": True,
            "first_win": index < solved,
            "depth_reached": 2 if index < deepened else 1,
        }
        for index in range(attempts)
    ]
    return {
        "experiment": "experiment_4605_live_integration_scored_agent",
        "honest_verdict": "complete: live_integration_no_value_honest_null_gap_sharpened",
        "first_win_rate_integrated": first_win_rate,
        "first_win_ci": {
            "method": "paired_percentile_bootstrap",
            "point": first_win_rate - mod.FIRST_WIN_BASELINE,
            "ci95": [ci_lower, max(ci_lower, first_win_rate - mod.FIRST_WIN_BASELINE)],
        },
        "multi_level_solve_rate": multi_level_rate,
        "integrated_measurement": {
            "variant_attempts_count": attempts,
            "variant_attempts": rows,
        },
    }


def _floor(*, live_count: int = 60, reproduced: bool = True) -> JsonDict:
    return {
        "package_path": "results/experiment_4679_submission_package_operator_resubmit.json",
        "package_exists": True,
        "replay_package_floor_reproduced": reproduced,
        "live_submittable_level_count": live_count,
        "offline_reproduced": reproduced,
        "ready_for_operator_submit": live_count > 33 and reproduced,
        "note": mod.REPLAY_FLOOR_NOTE,
    }


def _preconditions() -> JsonDict:
    return {
        "ok": True,
        "offline_arcade": True,
        "experiment_4605_importable": True,
    }


def _parity(passed: bool = True) -> JsonDict:
    return {
        "passed": passed,
        "command": "pytest tests/python/test_arc_submitted_agent_parity.py -q --no-cov",
    }


def _lever_inputs(*, chosen: Any = "unchanged") -> JsonDict:
    return {
        "a1": {
            "path": "results/experiment_4737_goal_energy_candidate_generation_valid_test.json",
            "chosen_submitted_config": chosen,
        },
        "a2": {
            "path": "results/experiment_4738_energy_fitness_qd_generation_valid_test.json",
            "chosen_submitted_config": "unchanged",
        },
    }


def test_req_capstone_4740_spec_declares_readiness_contract() -> None:
    """REQ-CAPSTONE-4740: OpenSpec declares the .436 held-out readiness gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-CAPSTONE-4740",
        "SCENARIO-CAPSTONE-4740",
        "SCENARIO-CAPSTONE-4740-BLOCKED-PRECONDITION",
        "SCENARIO-CAPSTONE-4740-FIELD-PRINCIPLES",
        mod.RESULT_RELATIVE_PATH,
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec
    assert "It must not include a redundant `min_held_out_variant_attempts` field." in spec


def test_scenario_capstone_4740_flat_first_win_emits_required_markers() -> None:
    """SCENARIO-CAPSTONE-4740: flat first-win is an explicit no-change."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v436_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_flat_no_leaderboard_change"
    assert artifact["first_win_rate_integrated"] == 0.04
    assert artifact["first_win_baseline"] == 0.04
    assert artifact["first_win_delta_vs_baseline"] == 0.0
    assert artifact["first_win_ci_lower"] == 0.0
    assert artifact["held_out_variant_attempts"] == 100
    assert "min_held_out_variant_attempts" not in artifact
    assert artifact["held_out_variant_attempt_floor"] == "B>=100"
    assert artifact["submitted_config_current"] is True
    assert artifact["null_delta_methodology_note"]
    assert artifact["positive_control_passed"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["replay_package_floor_reproduced"] is True
    assert artifact["replay_count_is_not_the_score"] is True
    assert artifact["replay_floor"]["live_submittable_level_count"] == 60
    assert artifact["v436_lever_inputs"]["a1"]["chosen_submitted_config"] == "unchanged"
    assert artifact["v436_lever_inputs"]["a2"]["chosen_submitted_config"] == "unchanged"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4740_success_requires_ci_parity_current_and_b100() -> None:
    """SCENARIO-CAPSTONE-4740: readiness needs CI, parity, current config, and B>=100."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(
            first_win_rate=0.12,
            ci_lower=0.02,
            attempts=125,
            multi_level_rate=0.04,
        ),
        replay_floor=_floor(live_count=0, reproduced=False),
        v436_lever_inputs=_lever_inputs(),
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "success: held_out_first_win_improved_0.08"
    assert artifact["first_win_delta_vs_baseline"] == 0.08
    assert artifact["first_win_ci_lower"] == 0.02
    assert artifact["multi_level_deepen_rate_integrated"] == 0.04
    assert artifact["submitted_config_current"] is True
    assert artifact["positive_control_passed"] is True
    assert artifact["null_delta_methodology_note"] == ""
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []

    undersized = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02, attempts=99),
        replay_floor=_floor(live_count=60, reproduced=True),
        v436_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert undersized["ready_for_operator_submit"] is False
    assert undersized["honest_verdict"] == "complete: held_out_first_win_measurement_below_b100"
    assert "held_out_variant_attempts_below_minimum" in mod.artifact_schema_errors(undersized)


def test_req_capstone_4740_replay_count_cannot_mask_regression() -> None:
    """REQ-CAPSTONE-4740: replay package floor never creates readiness."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.0, ci_lower=-0.04, attempts=100),
        replay_floor=_floor(live_count=99, reproduced=True),
        v436_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: held_out_first_win_below_baseline_no_leaderboard_change"
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["replay_floor"]["live_submittable_level_count"] == 99
    assert artifact["replay_count_is_not_the_score"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4740_mismatched_submitted_config_invalidates_readiness() -> None:
    """REQ-CAPSTONE-4740: measured config must be the current submitted config."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.12, ci_lower=0.02, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v436_lever_inputs=_lever_inputs(chosen={"goal_energy_alpha": 0.25}),
        duration_s=1.0,
        submitted_config_snapshot={"goal_energy_alpha": 0.9},
    )

    assert artifact["submitted_config_current"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["held_out_first_win_readiness"] is False
    assert artifact["honest_verdict"] == "complete: submitted_config_not_current_no_leaderboard_change"
    assert mod.artifact_schema_errors(artifact) == []


def test_req_capstone_4740_submitted_config_current_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CAPSTONE-4740: current-config detection handles nulls and changed configs."""

    assert mod._neutral_config_choice(None) is True
    assert mod._submitted_config_current(v436_lever_inputs=_lever_inputs(), parity_green=False) is False
    assert (
        mod._submitted_config_current(
            v436_lever_inputs={"skip": 7, "a1": {"chosen_submitted_config": ""}},
            parity_green=True,
        )
        is True
    )
    assert (
        mod._submitted_config_current(
            v436_lever_inputs={"a1": {"chosen_submitted_config": "enable_goal_energy"}},
            parity_green=True,
        )
        is False
    )

    monkeypatch.setattr(mod, "_submitted_config_snapshot", lambda: {"goal_energy_alpha": 0.9})
    assert (
        mod._submitted_config_current(
            v436_lever_inputs={"a1": {"chosen_submitted_config": {"goal_energy_alpha": 0.9}}},
            parity_green=True,
        )
        is True
    )


def test_req_capstone_4740_schema_rejects_marker_and_shape_errors() -> None:
    """SCENARIO-CAPSTONE-4740-FIELD-PRINCIPLES: schema protects the artifact shape."""

    artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        parity_test=_parity(True),
        proxy_artifact=_proxy(first_win_rate=0.04, ci_lower=0.0, attempts=100),
        replay_floor=_floor(live_count=60, reproduced=True),
        v436_lever_inputs=_lever_inputs(),
        duration_s=1.0,
    )

    missing_note = dict(artifact)
    missing_note["null_delta_methodology_note"] = ""
    assert "null_delta_methodology_note" in mod.artifact_schema_errors(missing_note)

    redundant_minimum = dict(artifact)
    redundant_minimum["min_held_out_variant_attempts"] = 100
    assert "redundant_min_held_out_variant_attempts" in mod.artifact_schema_errors(
        redundant_minimum
    )

    wrong_checksum = dict(artifact)
    wrong_checksum["reproducibility_checksum"] = "sha256:not-it"
    assert "reproducibility_checksum" in mod.artifact_schema_errors(wrong_checksum)

    missing_field = dict(artifact)
    missing_field.pop("submitted_config_current")
    assert "missing required field submitted_config_current" in mod.artifact_schema_errors(
        missing_field
    )

    bad_shape = dict(artifact)
    bad_shape.update(
        {
            "honest_verdict": "not-terminal",
            "field_principles": {},
            "verifier_is_oracle": True,
            "replay_count_is_not_the_score": False,
            "submitted_to_leaderboard": True,
            "positive_control_passed": False,
            "held_out_first_win_readiness": False,
            "ready_for_operator_submit": False,
        }
    )
    errors = mod.artifact_schema_errors(bad_shape)
    assert "honest_verdict_terminal_prefix" in errors
    assert "field_principles" in errors
    assert "verifier_is_oracle_false" in errors
    assert "replay_count_is_not_the_score_true" in errors
    assert "submitted_to_leaderboard_false" in errors
    assert "positive_control_passed" in errors

    bad_readiness = dict(artifact)
    bad_readiness["held_out_first_win_readiness"] = False
    bad_readiness["ready_for_operator_submit"] = False
    errors = mod.artifact_schema_errors(bad_readiness)
    assert "held_out_first_win_readiness_gate" in errors
    assert "ready_for_operator_submit_gate" in errors

    current_gate = dict(artifact)
    current_gate["submitted_config_current"] = False
    current_gate["ready_for_operator_submit"] = True
    assert "submitted_config_current_gate" in mod.artifact_schema_errors(current_gate)
