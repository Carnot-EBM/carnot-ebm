"""Tests for Exp 4003 ARC-AGI-3 verifier-validated frontier scaling.

Spec refs: REQ-PHASE4-023, SCENARIO-PHASE4-023.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.agentic.arc_scale_level_frontier import (
    BANKED_FRONTIER,
    REQUIRED_ARTIFACT_FIELDS,
    GameFrontierResult,
    artifact_schema_errors,
    build_frontier_artifact,
    count_validated_rules,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import experiment_4003_scale_level_frontier as exp  # noqa: E402


SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _result(
    short_game: str,
    levels_completed: int,
    *,
    validated: int = 0,
    saved: int = 0,
    real: bool = True,
) -> GameFrontierResult:
    return GameFrontierResult(
        short_game=short_game,
        game_id=f"{short_game}-fake",
        banked_level=BANKED_FRONTIER[short_game],
        levels_completed=levels_completed,
        first_fail_level=None if levels_completed > BANKED_FRONTIER[short_game] else BANKED_FRONTIER[short_game] + 1,
        per_level_actions=[4] * levels_completed,
        baseline_actions_ref=[5] * levels_completed,
        verifier_validated_count=validated,
        actions_saved_vs_openloop=saved,
        real_env_confirmed=real,
        stall_reason="validated frontier held" if levels_completed == BANKED_FRONTIER[short_game] else "",
        level_summaries=[],
        solve_log=[],
        candidate_validations=[],
    )


def test_req_phase4_023_spec_declares_frontier_scaling_contract() -> None:
    """REQ-PHASE4-023: OpenSpec declares Exp 4003 and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-023" in spec
    assert "SCENARIO-PHASE4-023" in spec
    assert "experiment_4003_scale_level_frontier.json" in spec
    assert "GAP-4 executed-consistency energy" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_phase4_023_artifact_schema_requires_bare_fields() -> None:
    """REQ-PHASE4-023: Exp 4003 artifacts keep the multi-game frontier fields auditable."""

    artifact = exp._base_artifact(seed=4003, started=0.0, verdict="complete: level_frontier_holds_test")
    artifact.update(
        {
            "ACCURACY_total_levels_solved": 5,
            "new_levels_this_task": 0,
            "per_game_max_level": {"r11l": 3, "lp85": 1, "sc25": 1},
            "verifier_validated_count": 2,
            "actions_saved_vs_openloop": 2,
            "per_level_actions": {"r11l": [4, 8, 12], "lp85": [5], "sc25": [17]},
            "baseline_actions_ref": {"r11l": [4, 10, 12], "lp85": [17], "sc25": [17]},
            "real_env_confirmed": True,
            "duration_s": 1.0,
        }
    )

    assert artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["ACCURACY_total_levels_solved"] = "5"
    bad["per_game_max_level"] = {"r11l": "3"}
    bad["per_level_actions"] = {"r11l": [4, "8"]}
    bad["real_env_confirmed"] = 1
    bad["inference_substrate"] = 99
    bad["duration_s"] = "slow"
    bad["honest_verdict"] = "done"

    errors = artifact_schema_errors(bad)

    assert any("ACCURACY_total_levels_solved" in err for err in errors)
    assert any("per_game_max_level" in err for err in errors)
    assert any("per_level_actions" in err for err in errors)
    assert any("real_env_confirmed" in err for err in errors)
    assert any("inference_substrate" in err for err in errors)
    assert any("duration_s" in err for err in errors)
    assert any("honest_verdict" in err for err in errors)

    missing = dict(artifact)
    del missing["baseline_actions_ref"]

    assert "missing required field baseline_actions_ref" in artifact_schema_errors(missing)


def test_req_phase4_023_aggregation_counts_new_levels_from_banked_frontier() -> None:
    """REQ-PHASE4-023: totals are measured beyond r11l L3, lp85 L1, and sc25 L1."""

    artifact = build_frontier_artifact(
        [_result("r11l", 4, validated=3, saved=4), _result("lp85", 1), _result("sc25", 1)],
        seed=4003,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["ACCURACY_total_levels_solved"] == 6
    assert artifact["new_levels_this_task"] == 1
    assert artifact["per_game_max_level"] == {"r11l": 4, "lp85": 1, "sc25": 1}
    assert artifact["verifier_validated_count"] == 3
    assert artifact["actions_saved_vs_openloop"] == 4
    assert artifact["honest_verdict"] == "success: scaled_level_frontier_r11l_to_L4_total6"
    assert artifact_schema_errors(artifact) == []


def test_scenario_phase4_023_no_advance_uses_complete_stall_verdict() -> None:
    """SCENARIO-PHASE4-023: no new level reports the first held frontier honestly."""

    artifact = build_frontier_artifact(
        [_result("r11l", 3, saved=2), _result("lp85", 1), _result("sc25", 1)],
        seed=4003,
        started=0.0,
        inference_substrate="test_substrate",
    )

    assert artifact["ACCURACY_total_levels_solved"] == 5
    assert artifact["new_levels_this_task"] == 0
    assert artifact["honest_verdict"].startswith("complete: level_frontier_holds_")
    assert "r11l_L4" in artifact["honest_verdict"]
    assert artifact_schema_errors(artifact) == []


def test_req_phase4_023_validated_rule_counter_uses_heldout_energy_gate() -> None:
    """REQ-PHASE4-023: only held-out verifier-passing candidates count as validated rules."""

    rows = [
        {"candidate_id": "safe", "heldout_energy": 0.0, "heldout_n": 2, "selected": True},
        {"candidate_id": "demo-only", "heldout_energy": None, "heldout_n": 0, "selected": True},
        {"candidate_id": "mismatch", "heldout_energy": 0.25, "heldout_n": 2, "selected": True},
        {"candidate_id": "not-committed", "heldout_energy": 0.0, "heldout_n": 1, "selected": False},
    ]

    assert count_validated_rules(rows) == 1


def test_scenario_phase4_023_blocks_when_offline_arcade_unavailable(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-023: unavailable offline Arcade writes a blocked artifact."""

    monkeypatch.setattr(exp, "REPO", tmp_path)

    def unavailable() -> object:
        raise RuntimeError("offline missing")

    monkeypatch.setattr(exp, "_load_offline_arcade", unavailable)

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "blocked_arc_offline_env_unavailable"
    assert artifact["real_env_confirmed"] is False
    assert artifact["ACCURACY_total_levels_solved"] == 0
    assert artifact_schema_errors(artifact) == []
    written = tmp_path / "results" / exp.RESULT_NAME
    assert json.loads(written.read_text(encoding="utf-8"))["honest_verdict"] == "blocked_arc_offline_env_unavailable"


def test_scenario_phase4_023_success_uses_mocked_real_env_results(monkeypatch, tmp_path) -> None:
    """SCENARIO-PHASE4-023: a validated multi-game advance writes the success verdict."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "_load_offline_arcade", lambda: object())
    monkeypatch.setattr(
        exp,
        "_run_all_frontiers",
        lambda arc, budget: [
            _result("r11l", 4, validated=1, saved=2),
            _result("lp85", 2, validated=1),
            _result("sc25", 1),
        ],
    )

    artifact = exp.run(write=True)

    assert artifact["honest_verdict"] == "success: scaled_level_frontier_r11l_to_L4_total7"
    assert artifact["ACCURACY_total_levels_solved"] == 7
    assert artifact["new_levels_this_task"] == 2
    assert artifact["per_game_max_level"]["lp85"] == 2
    assert artifact["verifier_validated_count"] == 2
    assert artifact_schema_errors(artifact) == []
