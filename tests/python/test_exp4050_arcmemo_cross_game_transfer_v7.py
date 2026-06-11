"""Tests for Exp 4050 ArcMemo v7 cross-game concept-library transfer.

Spec refs: REQ-LEARN-4050, SCENARIO-LEARN-4050.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import carnot.agentic.arc_arcmemo_cross_game_transfer_v7 as v7
from carnot.agentic.arc_arcmemo_cross_game_transfer_v7 import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_cross_game_transfer_artifact,
    build_v7_library,
    collect_prior_fragments,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4050_arcmemo_cross_game_transfer_v7 as exp  # noqa: E402


def _prior_row(source_game: str, source_artifact: str, payload: dict[str, object]) -> dict[str, object]:
    return {"source_game": source_game, "source_artifact": source_artifact, "payload": payload}


def _prior_solved_rows() -> list[dict[str, object]]:
    return [
        _prior_row(
            "r11l",
            "results/experiment_3946_r11l_first_solve.json",
            {
                "experiment": "experiment_3946_r11l_first_solve",
                "real_env_confirmed": True,
                "first_solve_at_action": 4,
                "induced_select_place_mechanic": "Click selects a piece, 2nd click places it.",
            },
        ),
        _prior_row(
            "lp85",
            "results/experiment_3954_second_game_solve.json",
            {
                "experiment": "experiment_3954_second_game_solve",
                "game_solved": "lp85-305b61c3",
                "real_env_confirmed": True,
                "first_solve_at_action": 5,
                "induced_mechanic": "Clicking buttons applies a deterministic permutation.",
            },
        ),
        _prior_row(
            "sc25",
            "results/experiment_3966_third_game_first_solve.json",
            {
                "experiment": "experiment_3966_third_game_first_solve",
                "game_solved": "sc25-635fd71a",
                "real_env_confirmed": True,
                "first_solve_at_action": 17,
                "induced_mechanic": "Clicked cells toggle a target pattern before navigation exits.",
            },
        ),
        _prior_row(
            "su15",
            "results/experiment_4004_fourth_game_explore_first.json",
            {
                "experiment": "experiment_4004_fourth_game_explore_first",
                "ACCURACY_levels_solved": 1,
                "game_solved": "su15-1944f8ab",
                "real_env_confirmed": True,
                "first_solve_at_action": 14,
                "exploration_actions_used": 4,
                "induced_mechanic": "Lattice-aligned clicks move a sprite toward the target zone.",
            },
        ),
        _prior_row(
            "tn36",
            "results/experiment_4015_fifth_game_explore_first.json",
            {
                "experiment": "experiment_4015_fifth_game_explore_first",
                "ACCURACY_levels_solved": 1,
                "game_solved": "tn36-ef4dde99",
                "real_env_confirmed": True,
                "first_solve_at_action": 7,
                "exploration_actions_used": 4,
                "induced_mechanic": "Click targets toggle two-bit program rows before execute.",
            },
        ),
        _prior_row(
            "cd82",
            "results/experiment_4024_fifth_game_explore_first.json",
            {
                "experiment": "experiment_4024_fifth_game_explore_first",
                "game_solved": True,
                "target_game": "cd82-fb555c5d",
                "real_env_confirmed": True,
                "first_solve_at_action": 5,
                "exploration_actions_used": 4,
                "induced_mechanic": "ACTION5 paints the active basket region with a selected color.",
            },
        ),
        _prior_row(
            "dc22",
            "results/experiment_4038_seventh_game_explore_first.json",
            {
                "experiment": "experiment_4038_seventh_game_explore_first",
                "game_solved": True,
                "target_game": "dc22-fdcac232",
                "real_env_confirmed": True,
                "first_solve_at_action": 20,
                "exploration_actions_used": 2,
                "induced_mechanic": "Visible clicks toggle blockers before goal navigation completes.",
            },
        ),
    ]


def _exp4049_payload(*, solved: bool = True) -> dict[str, object]:
    actions = [
        {"action": 6, "role": "select_item", "color": 9, "x": 36, "y": 59},
        {"action": 6, "role": "place_slot", "color": 9, "x": 23, "y": 30},
        {"action": 6, "role": "select_item", "color": 14, "x": 20, "y": 59},
        {"action": 6, "role": "place_slot", "color": 14, "x": 29, "y": 30},
        {"action": 6, "role": "select_item", "color": 11, "x": 44, "y": 59},
        {"action": 6, "role": "place_slot", "color": 11, "x": 35, "y": 30},
        {"action": 6, "role": "select_item", "color": 15, "x": 28, "y": 59},
        {"action": 6, "role": "place_slot", "color": 15, "x": 41, "y": 30},
        {"action": 5, "role": "validate"},
    ]
    return {
        "experiment": "experiment_4049_eighth_game_explore_first",
        "honest_verdict": (
            "success: eighth_game_solved_sb26-7fbdac44_at_action_9"
            if solved
            else "complete: eighth_game_no_solve_sb26_budget"
        ),
        "game_solved": solved,
        "real_env_confirmed": solved,
        "target_game": "sb26-7fbdac44",
        "candidate_baseline_actions": 18,
        "first_solve_at_action": 9 if solved else -1,
        "exploration_actions_used": 2,
        "induced_mechanic": (
            "Observed sb26 item selection and slot placement before validation; "
            "slot colors must match target colors left-to-right."
        ),
        "solve_trace": {
            "target_game": "sb26-7fbdac44",
            "selection_reason": "L0 baseline_actions=18",
            "actions": actions,
            "exploration_actions": actions[:2],
            "commit_actions": actions[2:],
            "induction_calls": [
                {
                    "call": "induce_sb26_color_sequence_slot_matching",
                    "mechanic": "click item select, click empty slot place, ACTION5 validate colors",
                }
            ],
            "verification_decisions": [{"retained": solved, "actions_checked": 7}],
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_req_learn_4050_spec_declares_cross_game_contract() -> None:
    """REQ-LEARN-4050: OpenSpec declares v7 transfer and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4050" in spec
    assert "SCENARIO-LEARN-4050" in spec
    assert "experiment_4050_arcmemo_cross_game_transfer_v7.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4050_builds_library_from_seven_prior_games_only() -> None:
    """REQ-LEARN-4050-1: seven prior solved fragments become documented abstractions."""

    fragments = collect_prior_fragments(_prior_solved_rows())
    library = build_v7_library(fragments)

    assert len(fragments) == 7
    assert {fragment["source_game"] for fragment in fragments} == {
        "r11l",
        "lp85",
        "sc25",
        "su15",
        "tn36",
        "cd82",
        "dc22",
    }
    assert len(library) == 1
    abstraction = library[0]
    assert abstraction["name"] == "click_state_transform_then_goal_commit_v7"
    assert "sb26" not in abstraction["source_games"]
    assert {"name", "signature", "documentation", "source_games", "source_fragments"} <= set(abstraction)
    assert "lambda" in abstraction["lambda_abstraction"]


def test_req_learn_4050_fragment_collection_covers_fallback_paths() -> None:
    """REQ-LEARN-4050-1: fragment extraction handles older artifact shapes."""

    rows = [
        None,
        {},
        {"payload": None},
        {
            "source_artifact": "results/unknown.json",
            "payload": {
                "real_env_confirmed": True,
                "game_solved": "xx99-abc",
                "induced_mechanic": "Unknown click fragment.",
            },
        },
        {
            "source_artifact": "results/attempt_detail.json",
            "payload": {
                "real_env_confirmed": True,
                "attempt_details": [
                    {
                        "game_id": "yy88-def",
                        "induced_mechanic": "Attempt detail mechanic text.",
                    }
                ],
                "game_solved": True,
            },
        },
        {
            "source_game": "sb26",
            "source_artifact": "results/target_leak.json",
            "payload": {
                "real_env_confirmed": True,
                "game_solved": "sb26-7fbdac44",
                "first_solve_at_action": 9,
                "induced_mechanic": "Target fragment must not enter the prior library.",
            },
        },
        {
            "source_game": "bad",
            "source_artifact": "results/unconfirmed.json",
            "payload": {"real_env_confirmed": False, "game_solved": "bad-1"},
        },
        {
            "source_artifact": "results/unknown_source.json",
            "payload": {
                "real_env_confirmed": True,
                "game_solved": True,
            },
        },
    ]

    fragments = collect_prior_fragments(rows)

    assert [fragment["source_game"] for fragment in fragments] == ["xx99", "yy88", "unknown"]
    assert fragments[0]["name"] == "xx99_induced_fragment"
    assert fragments[1]["induced_program_fragment"] == "Attempt detail mechanic text."
    assert fragments[2]["induced_program_fragment"] == ""


def test_req_learn_4050_generic_recurring_family_gets_abstraction() -> None:
    """REQ-LEARN-4050-1: recurring non-click families also compress."""

    library = build_v7_library(
        [
            {"name": "region_a", "family": "region_state_commit", "source_game": "aa", "source_artifact": "a"},
            {"name": "region_b", "family": "region_state_commit", "source_game": "bb", "source_artifact": "b"},
        ]
    )

    assert library[0]["name"] == "region_state_commit_v7_abstraction"
    assert library[0]["source_games"] == ["aa", "bb"]


def test_scenario_learn_4050_reuse_does_not_claim_win_unless_it_beats_v6() -> None:
    """SCENARIO-LEARN-4050: v7 must beat both cold and within-game-v6 costs."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=_exp4049_payload(),
        duration_s=0.25,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["actions_cold"] == 18
    assert artifact["actions_within_game_v6"] == 7
    assert artifact["actions_cross_game_v7"] == 9
    assert artifact["induction_calls_cold"] == 1
    assert artifact["induction_calls_within_game_v6"] == 0
    assert artifact["induction_calls_cross_game_v7"] == 0
    assert artifact["n_reused_abstractions"] == 1
    assert artifact["honest_verdict"] == (
        "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_within_game_v6"
    )
    assert artifact["transfer_assessment"] == "helped_vs_cold_but_lost_to_within_game_v6"
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4050_success_requires_cross_game_to_beat_both() -> None:
    """REQ-LEARN-4050-4: a strict action win gets the success verdict."""

    exp4049 = _exp4049_payload()
    trace = dict(exp4049["solve_trace"])  # type: ignore[index]
    trace["actions"] = trace["actions"][:5]
    trace["commit_actions"] = trace["actions"] + [{"action": 5, "role": "extra_validate"}] * 2
    exp4049["solve_trace"] = trace
    exp4049["first_solve_at_action"] = 5
    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=exp4049,
        duration_s=0.1,
    )

    assert artifact["cross_game_transfer_win"] is True
    assert artifact["honest_verdict"] == "success: arcmemo_v7_cross_game_transfer_18to5_actions"
    assert artifact["transfer_assessment"] == "helped_vs_cold_and_within_game_v6"


def test_scenario_learn_4050_tie_and_cold_failure_verdicts_are_distinct() -> None:
    """REQ-LEARN-4050-4: no-win reasons distinguish v6 ties from cold failures."""

    tied = _exp4049_payload()
    tied_trace = dict(tied["solve_trace"])  # type: ignore[index]
    tied_trace["commit_actions"] = tied_trace["actions"][:7]
    tied["solve_trace"] = tied_trace
    tied["first_solve_at_action"] = 7
    tied_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=tied,
        duration_s=0.1,
    )

    not_cheaper_than_cold = _exp4049_payload()
    cold_trace = dict(not_cheaper_than_cold["solve_trace"])  # type: ignore[index]
    cold_trace["commit_actions"] = cold_trace["actions"] + [{"action": 5, "role": "extra_validate"}] * 3
    not_cheaper_than_cold["solve_trace"] = cold_trace
    not_cheaper_than_cold["candidate_baseline_actions"] = 8
    cold_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=not_cheaper_than_cold,
        duration_s=0.1,
    )

    assert tied_artifact["transfer_assessment"] == "helped_vs_cold_but_tied_within_game_v6"
    assert cold_artifact["honest_verdict"] == (
        "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_cold"
    )
    assert cold_artifact["transfer_assessment"] == "hurt_or_unmeasured"


def test_scenario_learn_4050_unmatched_library_keeps_induction_call() -> None:
    """REQ-LEARN-4050-4: unmatched prior fragments cannot reduce new-game induction."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[_prior_solved_rows()[1]],
        exp4049=_exp4049_payload(),
        duration_s=0.1,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["n_reused_abstractions"] == 0
    assert artifact["actions_cross_game_v7"] == 9
    assert artifact["induction_calls_cross_game_v7"] == 1
    assert artifact["honest_verdict"] == "complete: arcmemo_v7_no_cross_game_transfer_no_prior_abstraction_fired"
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4050_no_trace_and_fallback_trace_paths() -> None:
    """REQ-LEARN-4050-5: missing traces fail closed; fallback action logs still measure depth."""

    no_trace = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049={"honest_verdict": "complete: eighth_game_no_trace"},
        duration_s=0.0,
    )
    fallback_trace = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4049={
            "honest_verdict": "complete: fallback_action_plan_attempt",
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "sb26-7fbdac44",
            "candidate_baseline_actions": 3,
            "exploration_actions_used": 1,
            "induced_mechanic": "fallback induction text",
            "action_plan": [{"action": 6}, {"action": 5}],
        },
        duration_s=0.0,
    )
    no_commit_trace = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4049={
            "honest_verdict": "complete: no_commit_attempt",
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "sb26-7fbdac44",
            "exploration_actions_used": 1,
            "solve_trace": {"actions": [{"action": 6}, {"action": 5}]},
        },
        duration_s=0.0,
    )

    assert no_trace["honest_verdict"] == "complete: arcmemo_v7_no_cross_game_transfer_no_usable_4049_trace"
    assert fallback_trace["target_evidence"] == "attempt_trace_only"
    assert fallback_trace["actions_cross_game_v7"] == 2
    assert fallback_trace["induction_calls_cold"] == 1
    assert fallback_trace["transfer_assessment"] == "attempt_only_no_solve_claim"
    assert no_commit_trace["actions_within_game_v6"] == 1
    assert v7._within_game_v6_actions({"first_solve_at_action": 5, "exploration_actions_used": 2}) == 3


def test_scenario_learn_4050_neutral_action_assessment_is_reported() -> None:
    """REQ-LEARN-4050-4: equal cold and v7 action counts are neutral, not a win."""

    exp4049 = _exp4049_payload()
    exp4049["candidate_baseline_actions"] = 9
    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[_prior_solved_rows()[1]],
        exp4049=exp4049,
        duration_s=0.0,
    )

    assert artifact["actions_cross_game_v7"] == artifact["actions_cold"]
    assert artifact["transfer_assessment"] == "neutral_vs_cold"


def test_scenario_learn_4050_attempt_trace_is_measured_without_solve_claim() -> None:
    """REQ-LEARN-4050-5: unsolved Exp 4049 attempts are measured honestly."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=_exp4049_payload(solved=False),
        duration_s=0.0,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["target_evidence"] == "attempt_trace_only"
    assert artifact["actions_cross_game_v7"] == 9
    assert artifact["honest_verdict"] == "complete: arcmemo_v7_no_cross_game_transfer_attempt_only"
    assert artifact_schema_errors(artifact) == []


def test_req_learn_4050_schema_rejects_non_bare_required_fields() -> None:
    """REQ-LEARN-4050-2: required artifact fields stay bare JSON scalars."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4049=_exp4049_payload(),
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["cross_game_transfer_win"] = 1
    bad["actions_cold"] = "18"
    bad["actions_within_game_v6"] = 7.0
    bad["actions_cross_game_v7"] = "9"
    bad["n_reused_abstractions"] = True
    bad["inference_substrate"] = None
    wrong_substrate = dict(artifact)
    wrong_substrate["inference_substrate"] = "wrong"
    impossible_win = dict(artifact)
    impossible_win["cross_game_transfer_win"] = True
    impossible_cold_win = dict(impossible_win)
    impossible_cold_win["actions_cold"] = 8

    errors = artifact_schema_errors(bad) + artifact_schema_errors(wrong_substrate)
    win_errors = artifact_schema_errors(impossible_win) + artifact_schema_errors(impossible_cold_win)
    missing = artifact_schema_errors({})

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert any(field in error for error in errors + missing)
    assert any("below cold" in error or "below within-game v6" in error for error in win_errors)


def test_runner_writes_exp4050_result_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-LEARN-4050: runner writes the stable Exp 4050 JSON deliverable."""

    rows = _prior_solved_rows()
    for row in rows:
        _write_json(tmp_path / str(row["source_artifact"]), row["payload"])  # type: ignore[arg-type]
    _write_json(
        tmp_path / "results" / "experiment_4049_eighth_game_explore_first.json",
        _exp4049_payload(),
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = tmp_path / "results" / "experiment_4050_arcmemo_cross_game_transfer_v7.json"
    assert artifact["honest_verdict"] == (
        "complete: arcmemo_v7_no_cross_game_transfer_v7_not_cheaper_than_within_game_v6"
    )
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
