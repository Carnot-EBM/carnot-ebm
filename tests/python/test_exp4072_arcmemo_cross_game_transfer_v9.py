"""Tests for Exp 4072 ArcMemo v9 cross-game transfer.

Spec refs: REQ-LEARN-4072, SCENARIO-LEARN-4072.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_arcmemo_cross_game_transfer_v9 as v9
from carnot.agentic.arc_arcmemo_cross_game_transfer_v9 import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_cross_game_transfer_artifact,
    build_v9_library,
    collect_prior_fragments,
)
from test_exp4062_arcmemo_cross_game_transfer_v8 import _prior_solved_rows, _write_json


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4072_arcmemo_cross_game_transfer_v9 as exp  # noqa: E402


def _exp4070_payload(*, solved: bool = True, seeded_actions: int | None = None) -> dict[str, object]:
    actions = [
        {"action": 6, "role": "cycle_cell", "grid": [18, 18], "target_color": 8},
        {"action": 6, "role": "cycle_cell", "grid": [18, 22], "target_color": 8},
        {"action": 6, "role": "cycle_cell", "grid": [26, 22], "target_color": 8},
        {"action": 6, "role": "cycle_cell", "grid": [18, 26], "target_color": 8},
    ]
    payload: dict[str, object] = {
        "experiment": "experiment_4070_ninth_game_explore_first",
        "honest_verdict": (
            "success: ninth_game_solved_ft09-0d8bbf25_at_action_4"
            if solved
            else "complete: ninth_game_no_solve_ft09-0d8bbf25_attempt_depth_4"
        ),
        "game_solved": solved,
        "real_env_confirmed": solved,
        "target_game": "ft09-0d8bbf25",
        "candidate_baseline_actions": 43,
        "first_solve_at_action": 4 if solved else -1,
        "exploration_actions_used": 1,
        "selected_candidate_reason": "selected fallback: ft09 is unsolved, win_difficulty=hard, L0 baseline_actions=43",
        "induced_mechanic": (
            "Observed ft09 Hkx cells cycle colors on click; induced a local non-navigation "
            "constraint model where bsT zero pixels require neighboring cells to equal the "
            "constraint center color and non-zero pixels require inequality."
        ),
        "solve_trace": {
            "actions": actions,
            "exploration_actions": actions[:1],
            "commit_actions": actions[1:],
            "induction_calls": [
                {
                    "call": "induce_ft09_local_constraint_color_cycle",
                    "mechanic": "clicking a visible Hkx cell cycles its center color",
                }
            ],
        },
    }
    if seeded_actions is not None:
        payload["v9_seeded_action_plan"] = actions[:seeded_actions]
    return payload


def test_req_learn_4072_spec_declares_v9_transfer_contract() -> None:
    """REQ-LEARN-4072: OpenSpec declares v9 transfer and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4072" in spec
    assert "SCENARIO-LEARN-4072" in spec
    assert "experiment_4072_arcmemo_cross_game_transfer_v9.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4072_builds_richer_library_from_all_eight_prior_games() -> None:
    """REQ-LEARN-4072-1: v9 library is richer than v8 and keeps source evidence."""

    fragments = collect_prior_fragments(_prior_solved_rows())
    library = build_v9_library(fragments)

    assert len(fragments) == 8
    assert {fragment["source_game"] for fragment in fragments} == {
        "r11l",
        "lp85",
        "sc25",
        "su15",
        "tn36",
        "cd82",
        "dc22",
        "sb26",
    }
    assert len(library) > 2
    assert "local_constraint_click_state_update_v9" in {item["name"] for item in library}
    for abstraction in library:
        assert {"name", "signature", "documentation", "source_games", "source_fragments"} <= set(abstraction)
        assert "lambda" in abstraction["lambda_abstraction"]
        assert abstraction["match_tokens"]


def test_scenario_learn_4072_real_exp4070_solve_reuses_prior_abstractions_but_loses_to_within_game() -> None:
    """SCENARIO-LEARN-4072: Exp 4070 is measured honestly against both baselines."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=_exp4070_payload(solved=True),
        duration_s=0.1,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["actions_cold"] == 43
    assert artifact["actions_within_game"] == 3
    assert artifact["actions_cross_game_v9"] == 4
    assert artifact["induction_calls_cold"] == 1
    assert artifact["induction_calls_within_game"] == 0
    assert artifact["induction_calls_cross_game_v9"] == 0
    assert artifact["n_reused_abstractions"] == 3
    assert {item["name"] for item in artifact["reused_abstractions"]} == {
        "click_state_transform_then_goal_commit_v9",
        "local_constraint_click_state_update_v9",
        "color_state_update_then_constraint_commit_v9",
    }
    assert artifact["target_evidence"] == "confirmed_solve"
    assert artifact["target_future_fragments"][0]["source_game"] == "ft09"
    assert all("ft09" not in item["source_games"] for item in artifact["reused_abstractions"])
    assert artifact["honest_verdict"] == (
        "complete: arcmemo_v9_no_cross_game_transfer_v9_not_cheaper_than_within_game"
    )
    assert artifact["transfer_assessment"] == "helped_vs_cold_but_lost_to_within_game"
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4072_success_requires_v9_to_beat_cold_and_within_game() -> None:
    """REQ-LEARN-4072-4: a win is allowed only below both action baselines."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=_exp4070_payload(solved=True, seeded_actions=2),
        duration_s=0.0,
    )

    assert artifact["cross_game_transfer_win"] is True
    assert artifact["actions_cross_game_v9"] == 2
    assert artifact["actions_cross_game_v9"] < artifact["actions_cold"]
    assert artifact["actions_cross_game_v9"] < artifact["actions_within_game"]
    assert artifact["honest_verdict"] == "success: arcmemo_v9_cross_game_transfer_43to2_actions"


def test_scenario_learn_4072_no_win_reasons_cover_ties_neutral_and_cold_failures() -> None:
    """REQ-LEARN-4072-4: no-win reasons distinguish ties, no reuse, and cold failures."""

    tied = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=_exp4070_payload(solved=True, seeded_actions=3),
        duration_s=0.0,
    )
    neutral_payload = _exp4070_payload(solved=True)
    neutral_payload["candidate_baseline_actions"] = 4
    neutral = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=neutral_payload,
        duration_s=0.0,
    )
    cold_failure_payload = _exp4070_payload(solved=True, seeded_actions=2)
    cold_failure_payload["candidate_baseline_actions"] = 2
    cold_failure = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=cold_failure_payload,
        duration_s=0.0,
    )
    no_reuse = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4070=_exp4070_payload(solved=True),
        duration_s=0.0,
    )
    hurt_payload = _exp4070_payload(solved=True)
    hurt_payload["candidate_baseline_actions"] = 2
    hurt = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=hurt_payload,
        duration_s=0.0,
    )

    assert tied["transfer_assessment"] == "helped_vs_cold_but_tied_within_game"
    assert neutral["transfer_assessment"] == "neutral_vs_cold"
    assert cold_failure["honest_verdict"] == "complete: arcmemo_v9_no_cross_game_transfer_v9_not_cheaper_than_cold"
    assert no_reuse["honest_verdict"] == "complete: arcmemo_v9_no_cross_game_transfer_no_prior_abstraction_fired"
    assert hurt["transfer_assessment"] == "hurt_or_unmeasured"


def test_scenario_learn_4072_attempt_trace_is_measured_without_solve_claim() -> None:
    """REQ-LEARN-4072-5: unsolved Exp 4070 attempts stay complete, not success."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=_exp4070_payload(solved=False, seeded_actions=2),
        duration_s=0.0,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["target_evidence"] == "attempt_trace_only"
    assert artifact["actions_cross_game_v9"] == 2
    assert artifact["honest_verdict"] == "complete: arcmemo_v9_no_cross_game_transfer_attempt_only"
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4072_missing_exp4070_fails_closed_without_losing_library_evidence() -> None:
    """REQ-LEARN-4072-6: missing target writes a valid zero-count artifact."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=None,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "complete: arcmemo_v9_no_cross_game_transfer_no_usable_4070_trace"
    assert artifact["cross_game_transfer_win"] is False
    assert artifact["actions_cold"] == 0
    assert artifact["actions_within_game"] == 0
    assert artifact["actions_cross_game_v9"] == 0
    assert artifact["n_reused_abstractions"] == 0
    assert artifact["n_prior_fragments"] == 8
    assert artifact["target_evidence"] == "no_usable_trace"
    assert artifact_schema_errors(artifact) == []


def test_req_learn_4072_schema_rejects_non_bare_required_fields() -> None:
    """REQ-LEARN-4072-2: required artifact fields stay bare JSON scalars."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4070=_exp4070_payload(solved=True, seeded_actions=2),
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["cross_game_transfer_win"] = 1
    bad["actions_cold"] = "43"
    bad["actions_within_game"] = 3.0
    bad["actions_cross_game_v9"] = "2"
    bad["n_reused_abstractions"] = True
    bad["inference_substrate"] = None
    wrong_substrate = dict(artifact)
    wrong_substrate["inference_substrate"] = "wrong"
    impossible_win = dict(artifact)
    impossible_win["actions_cross_game_v9"] = impossible_win["actions_within_game"]
    impossible_cold_win = dict(artifact)
    impossible_cold_win["actions_cold"] = impossible_cold_win["actions_cross_game_v9"]

    errors = artifact_schema_errors(bad) + artifact_schema_errors(wrong_substrate)
    win_errors = artifact_schema_errors(impossible_win) + artifact_schema_errors(impossible_cold_win)
    missing = artifact_schema_errors({})

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert any(field in error for error in errors + missing)
    assert any("below cold" in error for error in win_errors)
    assert any("below within-game" in error for error in win_errors)


def test_runner_writes_exp4072_result_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-LEARN-4072: runner writes the stable Exp 4072 JSON deliverable."""

    rows = _prior_solved_rows()
    for row in rows:
        _write_json(tmp_path / str(row["source_artifact"]), row["payload"])  # type: ignore[arg-type]
    _write_json(tmp_path / "results" / "experiment_4070_ninth_game_explore_first.json", _exp4070_payload())
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = tmp_path / "results" / "experiment_4072_arcmemo_cross_game_transfer_v9.json"
    assert artifact["honest_verdict"] == (
        "complete: arcmemo_v9_no_cross_game_transfer_v9_not_cheaper_than_within_game"
    )
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact


def test_runner_raises_on_schema_errors(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4072-2: runner refuses to write malformed Exp 4072 artifacts."""

    rows = _prior_solved_rows()
    for row in rows:
        _write_json(tmp_path / str(row["source_artifact"]), row["payload"])  # type: ignore[arg-type]
    _write_json(tmp_path / "results" / "experiment_4070_ninth_game_explore_first.json", _exp4070_payload())
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["schema broke"])

    with pytest.raises(ValueError, match="schema broke"):
        exp.run(write=False)


def test_private_helpers_cover_v9_fallback_paths() -> None:
    """REQ-LEARN-4072-5: fallback action logs and non-dict targets are bounded."""

    no_commit = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4070={
            "honest_verdict": "complete: no_commit_attempt",
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "ft09",
            "exploration_actions_used": 1,
            "solve_trace": {"actions": [{"action": 6}, {"action": 5}]},
        },
        duration_s=0.0,
    )

    assert no_commit["actions_within_game"] == 1
    assert v9._within_game_actions({"first_solve_at_action": 5, "exploration_actions_used": 2}) == 3
    assert build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4070={"honest_verdict": "complete: no trace"},
        duration_s=0.0,
    )["target_evidence"] == "no_usable_trace"
