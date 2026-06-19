"""Tests for Exp 4435 generic first-contact verdict fix.

Spec refs: REQ-REPORT-4435, SCENARIO-REPORT-4435.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4435_generic_first_contact_fixed as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"


def _ok_preconditions() -> dict[str, Any]:
    return {
        "offline_env_files_present": True,
        "arc_solver_kit_import": True,
        "arc_solve_learning_import": True,
        "focused_exp4423_pytest_green": True,
        "verdict_contract_fixed": True,
        "llm_induction_needed": False,
        "live_generator_gguf_cached_if_needed": True,
        "no_3090_inference": True,
        "leaderboard_submission": False,
        "ok": True,
    }


def _success_exp4423() -> dict[str, Any]:
    return {
        "experiment": "experiment_4423_generic_first_contact_breadth",
        "target_game": "dc22",
        "honest_verdict": "success: generic_first_contact_dc22_L1_offline_reproduced",
        "offline_reproduced": True,
        "reproduced_levels": 1,
        "verifier_is_oracle": False,
        "recommendation": {"target_game": "dc22", "recommended": [{"game": "s5i5"}]},
        "routing_options": [{"id": "closest_solved_recipe"}],
        "standing_loop_result": {
            "game": "dc22",
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproduction_gate": {"reproduced": True, "reached_level": 1},
        },
        "missing_verifier_gaps": [],
        "reproducibility_checksum": "a" * 64,
    }


def _no_level_exp4423() -> dict[str, Any]:
    return {
        "experiment": "experiment_4423_generic_first_contact_breadth",
        "target_game": "bp35",
        "honest_verdict": "complete: generic_first_contact_bp35_routed_no_new_level_gap_logged",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "verifier_is_oracle": False,
        "recommendation": {"target_game": "bp35", "recommended": [{"game": "r11l"}]},
        "routing_options": [{"id": "closest_solved_recipe"}],
        "standing_loop_result": {
            "game": "bp35",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "status": "needs_per_game_RE",
        },
        "missing_verifier_gaps": [
            {
                "gap_id": "GAP-4423-BP35-UNSELECTABLE-FIRST-CONTACT",
                "failure_mode": "needs_per_game_RE",
                "candidate_design": "derive a selectable verifier",
            }
        ],
        "reproducibility_checksum": "b" * 64,
    }


def test_req_report_4435_spec_declares_terminal_fixed_contract() -> None:
    """REQ-REPORT-4435: OpenSpec names the fixed verdict and reproduction fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4435" in spec
    assert "SCENARIO-REPORT-4435" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "partial:" in spec
    assert "arc_solve_learning.recommend_approach(game)" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_report_4435_success_counts_only_reproduction_gated_level(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4435: a routed solve banks a level only from reproduction evidence."""

    calls: list[dict[str, Any]] = []

    def run_first_contact(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _success_exp4423()

    artifact = mod.run(
        root=tmp_path,
        target_game="dc22",
        preconditions_checked=_ok_preconditions(),
        first_contact_run_fn=run_first_contact,
        now=lambda: 9.0,
    )

    assert calls[0]["target_game"] == "dc22"
    assert artifact["honest_verdict"] == "success: generic_first_contact_fixed_dc22_L1_offline_reproduced"
    assert artifact["verdict_contract_fixed"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["underlying_exp4423_honest_verdict"].startswith("success:")
    assert artifact["routing_recommendation"]["target_game"] == "dc22"
    assert artifact["standing_loop_result"]["reproduction_gate"]["reproduced"] is True
    assert artifact["no_3090_inference"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))[
        "reproduced_levels"
    ] == 1


def test_scenario_report_4435_no_level_is_terminal_complete_with_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4435: no generic advance is complete with the residual gap logged."""

    artifact = mod.run(
        root=tmp_path,
        target_game="bp35",
        preconditions_checked=_ok_preconditions(),
        first_contact_run_fn=lambda **_kwargs: _no_level_exp4423(),
        now=lambda: 4.0,
    )

    assert artifact["honest_verdict"] == "complete: generic_first_contact_bp35_routed_no_new_level_gap_logged"
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"][0]["gap_id"] == "GAP-4423-BP35-UNSELECTABLE-FIRST-CONTACT"
    assert artifact["residual_mechanic_gap_logged"] is True
    assert "partial:" not in artifact["honest_verdict"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_report_4435_blocked_precondition_does_not_run_solver(tmp_path: Path) -> None:
    """REQ-REPORT-4435: precondition misses stop before routing or solve fabrication."""

    calls: list[str] = []

    artifact = mod.run(
        root=tmp_path,
        target_game="dc22",
        preconditions_checked={**_ok_preconditions(), "focused_exp4423_pytest_green": False},
        first_contact_run_fn=lambda **_kwargs: calls.append("called") or _success_exp4423(),
        now=lambda: 1.0,
    )

    assert calls == []
    assert artifact["honest_verdict"] == "complete: blocked_focused_exp4423_pytest"
    assert artifact["verdict_contract_fixed"] is False
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["missing_verifier_gaps"] == []
    assert mod.artifact_schema_errors(artifact) == []


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"offline_env_files_present": False}, "offline_env_files"),
        ({"arc_solver_kit_import": False}, "arc_solver_kit"),
        ({"arc_solve_learning_import": False}, "arc_solve_learning"),
        ({"verdict_contract_fixed": False}, "verdict_contract_fixed"),
        ({"llm_induction_needed": True, "live_generator_gguf_cached_if_needed": False}, "live_generator_gguf"),
        ({"no_3090_inference": False}, "no_3090_inference_policy"),
        ({"leaderboard_submission": True}, "leaderboard_submission_policy"),
    ],
)
def test_req_report_4435_precondition_miss_names_resource(
    override: dict[str, Any],
    expected: str,
) -> None:
    """REQ-REPORT-4435: each missing precondition maps to a terminal blocked resource."""

    assert mod.first_precondition_miss({**_ok_preconditions(), **override}) == expected


def test_req_report_4435_schema_rejects_partial_and_fabricated_no_level(tmp_path: Path) -> None:
    """REQ-REPORT-4435: schema rejects partial prefixes, type drift, and missing gaps."""

    artifact = mod.run(
        root=tmp_path,
        target_game="bp35",
        preconditions_checked=_ok_preconditions(),
        first_contact_run_fn=lambda **_kwargs: _no_level_exp4423(),
        now=lambda: 2.0,
    )
    bad = {
        **artifact,
        "honest_verdict": "partial: retry_me",
        "verdict_contract_fixed": "true",
        "reproduced_levels": "0",
        "offline_reproduced": "false",
        "random_seed": "4435",
        "reproducibility_checksum": "z" * 64,
        "verifier_is_oracle": False,
        "missing_verifier_gaps": [],
        "no_3090_inference": False,
        "submitted_to_leaderboard": True,
        "field_principles": {**mod.FIELD_PRINCIPLES, "honest_verdict": {"principle": "wrong"}},
    }

    errors = mod.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "honest_verdict must not use partial prefix" in errors
    assert "verdict_contract_fixed must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be hex" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "complete no-new-level verdict requires missing_verifier_gaps" in errors
    assert "no_3090_inference must be true" in errors
    assert "submitted_to_leaderboard must be false" in errors
    assert "field_principles.honest_verdict must match REQ-REPORT-4435" in errors

    with pytest.raises(ValueError, match="honest_verdict"):
        mod.write_artifact(tmp_path, {"honest_verdict": "partial: invalid"})


def test_req_report_4435_contract_probe_detects_fixed_exp4423_vocabulary() -> None:
    """REQ-REPORT-4435: the contract probe accepts complete and rejects partial."""

    assert mod.exp4423_verdict_contract_fixed() is True
