"""Tests for Exp 4062 ArcMemo v8 richer cross-game transfer.

Spec refs: REQ-LEARN-4062, SCENARIO-LEARN-4062.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import carnot.agentic.arc_arcmemo_cross_game_transfer_v8 as v8
from carnot.agentic.arc_arcmemo_cross_game_transfer_v8 import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_cross_game_transfer_artifact,
    build_v8_library,
    collect_prior_fragments,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4062_arcmemo_cross_game_transfer_v8 as exp  # noqa: E402


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
        _prior_row(
            "sb26",
            "results/experiment_4049_eighth_game_explore_first.json",
            _exp4049_payload(),
        ),
    ]


def _exp4049_payload() -> dict[str, object]:
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
        "honest_verdict": "success: eighth_game_solved_sb26-7fbdac44_at_action_9",
        "game_solved": True,
        "real_env_confirmed": True,
        "target_game": "sb26-7fbdac44",
        "candidate_baseline_actions": 18,
        "first_solve_at_action": 9,
        "exploration_actions_used": 2,
        "induced_mechanic": (
            "Observed sb26 item selection and slot placement before validation; "
            "slot colors must match target colors left-to-right."
        ),
        "solve_trace": {
            "actions": actions,
            "exploration_actions": actions[:2],
            "commit_actions": actions[2:],
            "induction_calls": [{"call": "induce_sb26_color_sequence_slot_matching"}],
        },
    }


def _exp4060_payload(*, solved: bool = True, seeded: bool = True) -> dict[str, object]:
    actions = [
        {"action": 6, "role": "select_item", "color": 2},
        {"action": 6, "role": "place_slot", "color": 2},
        {"action": 6, "role": "select_item", "color": 4},
        {"action": 6, "role": "place_slot", "color": 4},
        {"action": 6, "role": "select_item", "color": 8},
        {"action": 6, "role": "place_slot", "color": 8},
        {"action": 5, "role": "validate"},
        {"action": 5, "role": "confirm"},
    ]
    payload: dict[str, object] = {
        "experiment": "experiment_4060_ninth_game_explore_first",
        "honest_verdict": (
            "success: ninth_game_solved_zz99-at_action_8" if solved else "complete: ninth_game_attempt_zz99"
        ),
        "game_solved": solved,
        "real_env_confirmed": solved,
        "target_game": "zz99-1234abcd",
        "candidate_baseline_actions": 16,
        "first_solve_at_action": 8 if solved else -1,
        "exploration_actions_used": 1,
        "induced_mechanic": "Select colored items, place them into target slots, then validate the color sequence.",
        "solve_trace": {
            "actions": actions,
            "exploration_actions": actions[:1],
            "commit_actions": actions[1:],
            "induction_calls": [{"call": "induce_zz99_slot_sequence"}],
        },
    }
    if seeded:
        payload["v8_seeded_action_plan"] = actions[:5]
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_req_learn_4062_spec_declares_v8_transfer_contract() -> None:
    """REQ-LEARN-4062: OpenSpec declares v8 transfer and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4062" in spec
    assert "SCENARIO-LEARN-4062" in spec
    assert "experiment_4062_arcmemo_cross_game_transfer_v8.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4062_builds_richer_library_from_all_eight_prior_games() -> None:
    """REQ-LEARN-4062-1: all eight prior solved fragments become documented abstractions."""

    fragments = collect_prior_fragments(_prior_solved_rows())
    library = build_v8_library(fragments)

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
    assert len(library) >= 2
    names = {abstraction["name"] for abstraction in library}
    assert "click_state_transform_then_goal_commit_v8" in names
    assert "select_place_color_sequence_v8" in names
    color_sequence = next(item for item in library if item["name"] == "select_place_color_sequence_v8")
    assert color_sequence["source_games"] == ["sb26"]
    assert {"name", "signature", "documentation", "source_games", "source_fragments"} <= set(color_sequence)
    assert "lambda" in color_sequence["lambda_abstraction"]


def test_req_learn_4062_fragment_collection_skips_invalid_or_unconfirmed_rows() -> None:
    """REQ-LEARN-4062-1: only confirmed prior solves enter the v8 library."""

    fragments = collect_prior_fragments(
        [
            None,
            {},
            {"payload": None},
            {
                "source_game": "bad",
                "source_artifact": "results/bad.json",
                "payload": {"real_env_confirmed": False, "game_solved": "bad-1"},
            },
            _prior_solved_rows()[0],
        ]
    )

    assert [fragment["source_game"] for fragment in fragments] == ["r11l"]


def test_scenario_learn_4062_absent_exp4060_fails_closed_without_losing_library_evidence() -> None:
    """REQ-LEARN-4062-6: missing ninth-game trace writes a valid no-transfer artifact."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=None,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "complete: arcmemo_v8_no_cross_game_transfer_no_usable_4060_trace"
    assert artifact["cross_game_transfer_win"] is False
    assert artifact["actions_cold"] == 0
    assert artifact["actions_within_game"] == 0
    assert artifact["actions_cross_game_v8"] == 0
    assert artifact["n_reused_abstractions"] == 0
    assert artifact["n_prior_fragments"] == 8
    assert artifact["n_named_abstractions"] >= 2
    assert artifact["target_evidence"] == "no_usable_trace"
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4062_no_matching_abstraction_keeps_induction_cost() -> None:
    """REQ-LEARN-4062-4: confirmed targets without a fired abstraction cannot claim transfer."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4060=_exp4060_payload(solved=True, seeded=True),
        duration_s=0.0,
    )

    assert artifact["n_reused_abstractions"] == 0
    assert artifact["induction_calls_cross_game_v8"] == artifact["induction_calls_cold"]
    assert artifact["honest_verdict"] == "complete: arcmemo_v8_no_cross_game_transfer_no_prior_abstraction_fired"


def test_scenario_learn_4062_success_requires_v8_to_beat_cold_and_within_game() -> None:
    """SCENARIO-LEARN-4062: v8 transfer wins only below both action baselines."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=_exp4060_payload(solved=True, seeded=True),
        duration_s=0.1,
    )

    assert artifact["cross_game_transfer_win"] is True
    assert artifact["actions_cold"] == 16
    assert artifact["actions_within_game"] == 7
    assert artifact["actions_cross_game_v8"] == 5
    assert artifact["induction_calls_cold"] == 1
    assert artifact["induction_calls_within_game"] == 0
    assert artifact["induction_calls_cross_game_v8"] == 0
    assert artifact["n_reused_abstractions"] >= 2
    assert artifact["honest_verdict"] == "success: arcmemo_v8_cross_game_transfer_16to5_actions"
    assert artifact["transfer_assessment"] == "helped_vs_cold_and_within_game"
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4062_tie_neutral_and_cold_failure_assessments() -> None:
    """REQ-LEARN-4062-4: no-win reasons distinguish ties, neutral cases, and cold failures."""

    tied = _exp4060_payload(solved=True, seeded=True)
    tied["v8_seeded_action_plan"] = tied["solve_trace"]["actions"][:7]  # type: ignore[index]
    tied_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=tied,
        duration_s=0.0,
    )

    neutral = _exp4060_payload(solved=True, seeded=False)
    neutral["candidate_baseline_actions"] = 8
    neutral_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=neutral,
        duration_s=0.0,
    )

    not_cheaper_than_cold = _exp4060_payload(solved=True, seeded=True)
    not_cheaper_than_cold["candidate_baseline_actions"] = 4
    not_cheaper_than_cold_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=not_cheaper_than_cold,
        duration_s=0.0,
    )

    hurt = _exp4060_payload(solved=True, seeded=True)
    hurt["v8_seeded_action_plan"] = hurt["solve_trace"]["actions"] + [{"action": 5}] * 3  # type: ignore[index]
    hurt["candidate_baseline_actions"] = 6
    hurt_artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=hurt,
        duration_s=0.0,
    )

    assert tied_artifact["transfer_assessment"] == "helped_vs_cold_but_tied_within_game"
    assert neutral_artifact["transfer_assessment"] == "neutral_vs_cold"
    assert not_cheaper_than_cold_artifact["honest_verdict"] == (
        "complete: arcmemo_v8_no_cross_game_transfer_v8_not_cheaper_than_cold"
    )
    assert hurt_artifact["transfer_assessment"] == "hurt_or_unmeasured"


def test_scenario_learn_4062_reuse_without_seeded_shortcut_does_not_claim_win() -> None:
    """REQ-LEARN-4062-4: matching abstractions are not enough unless actions drop."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=_exp4060_payload(solved=True, seeded=False),
        duration_s=0.1,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["actions_cross_game_v8"] == 8
    assert artifact["actions_within_game"] == 7
    assert artifact["n_reused_abstractions"] >= 2
    assert artifact["honest_verdict"] == (
        "complete: arcmemo_v8_no_cross_game_transfer_v8_not_cheaper_than_within_game"
    )


def test_scenario_learn_4062_attempt_trace_is_measured_without_solve_claim() -> None:
    """REQ-LEARN-4062-5: unsolved Exp 4060 attempts stay complete, not success."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=_exp4060_payload(solved=False, seeded=True),
        duration_s=0.0,
    )

    assert artifact["cross_game_transfer_win"] is False
    assert artifact["target_evidence"] == "attempt_trace_only"
    assert artifact["actions_cross_game_v8"] == 5
    assert artifact["honest_verdict"] == "complete: arcmemo_v8_no_cross_game_transfer_attempt_only"
    assert artifact_schema_errors(artifact) == []


def test_req_learn_4062_schema_rejects_non_bare_required_fields() -> None:
    """REQ-LEARN-4062-2: required artifact fields stay bare JSON scalars."""

    artifact = build_cross_game_transfer_artifact(
        prior_solved_artifacts=_prior_solved_rows(),
        exp4060=_exp4060_payload(solved=True, seeded=True),
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["cross_game_transfer_win"] = 1
    bad["actions_cold"] = "16"
    bad["actions_within_game"] = 7.0
    bad["actions_cross_game_v8"] = "5"
    bad["n_reused_abstractions"] = True
    bad["inference_substrate"] = None
    wrong_substrate = dict(artifact)
    wrong_substrate["inference_substrate"] = "wrong"
    impossible_win = dict(artifact)
    impossible_win["actions_cross_game_v8"] = impossible_win["actions_within_game"]
    impossible_cold_win = dict(artifact)
    impossible_cold_win["actions_cold"] = impossible_cold_win["actions_cross_game_v8"]

    errors = artifact_schema_errors(bad) + artifact_schema_errors(wrong_substrate)
    win_errors = artifact_schema_errors(impossible_win) + artifact_schema_errors(impossible_cold_win)
    missing = artifact_schema_errors({})

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert any(field in error for error in errors + missing)
    assert any("below cold" in error for error in win_errors)
    assert any("below within-game" in error for error in win_errors)


def test_runner_writes_exp4062_result_json_for_missing_exp4060(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-LEARN-4062: runner writes the stable Exp 4062 JSON deliverable."""

    rows = _prior_solved_rows()
    for row in rows:
        _write_json(tmp_path / str(row["source_artifact"]), row["payload"])  # type: ignore[arg-type]
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = tmp_path / "results" / "experiment_4062_arcmemo_cross_game_transfer_v8.json"
    assert artifact["honest_verdict"] == "complete: arcmemo_v8_no_cross_game_transfer_no_usable_4060_trace"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact


def test_runner_raises_on_schema_errors(monkeypatch, tmp_path: Path) -> None:
    """REQ-LEARN-4062-2: runner refuses to write malformed Exp 4062 artifacts."""

    rows = _prior_solved_rows()
    for row in rows:
        _write_json(tmp_path / str(row["source_artifact"]), row["payload"])  # type: ignore[arg-type]
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["schema broke"])

    with pytest.raises(ValueError, match="schema broke"):
        exp.run(write=False)


def test_private_helpers_cover_fallback_paths() -> None:
    """REQ-LEARN-4062-5: fallback action logs and non-dict targets are bounded."""

    no_commit = build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4060={
            "honest_verdict": "complete: no_commit_attempt",
            "game_solved": False,
            "real_env_confirmed": False,
            "target_game": "zz99",
            "exploration_actions_used": 1,
            "solve_trace": {"actions": [{"action": 6}, {"action": 5}]},
        },
        duration_s=0.0,
    )

    assert no_commit["actions_within_game"] == 1
    assert v8._within_game_actions({"first_solve_at_action": 5, "exploration_actions_used": 2}) == 3
    assert build_cross_game_transfer_artifact(
        prior_solved_artifacts=[],
        exp4060={"honest_verdict": "complete: no trace"},
        duration_s=0.0,
    )["target_evidence"] == "no_usable_trace"
