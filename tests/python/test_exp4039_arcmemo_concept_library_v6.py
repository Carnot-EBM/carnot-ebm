"""Tests for Exp 4039 ArcMemo v6 compressed concept-library transfer.

Spec refs: REQ-LEARN-4039, SCENARIO-LEARN-4039.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot.agentic.arc_arcmemo_concept_library_v6 import (
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_schema_errors,
    build_compressed_library,
    build_transfer_artifact,
    collect_raw_concepts,
)


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
sys.path.insert(0, str(REPO / "scripts" / "experiments"))

import exp4039_arcmemo_concept_library_v6 as exp  # noqa: E402


def _concept_payload() -> dict[str, object]:
    return {
        "experiment": "prior_arcmemo_transfer",
        "concept_memory": [
            {
                "name": "select_then_place",
                "family": "click_state_transform",
                "source_game": "r11l",
                "target_games": ["r11l"],
                "when_it_applies": "A click selects or activates an object and a later action applies it.",
                "effect": "Click selects a piece, then a second click places it near the target.",
                "source": "results/experiment_3946_r11l_first_solve.json",
            },
            {
                "name": "pattern_match_then_navigate",
                "family": "click_state_transform",
                "source_game": "sc25",
                "target_games": ["sc25"],
                "when_it_applies": "A clicked subset toggles visible pattern state before navigation completes.",
                "effect": "Separate the solve into pattern-satisfaction clicks followed by navigation.",
                "source": "results/experiment_3966_third_game_first_solve.json",
            },
            {
                "name": "object_click_count_match",
                "family": "click_state_transform",
                "source_game": "tn36",
                "target_games": ["tn36"],
                "when_it_applies": "Connected components can be clicked to move count-bearing objects.",
                "effect": "Use object centroids as action candidates and stop on a real level-up.",
                "source": "results/experiment_3981_fourth_game_first_solve.json",
            },
        ],
    }


def _unrelated_concept_payload() -> dict[str, object]:
    return {
        "experiment": "prior_arcmemo_transfer",
        "concept_memory": [
            {
                "name": "permute_set_by_button",
                "family": "discrete_set_permutation",
                "source_game": "lp85",
                "target_games": ["lp85"],
                "when_it_applies": "Button clicks deterministically permute a set of pieces.",
                "effect": "Represent each button as a reusable permutation over the current piece set.",
                "source": "results/experiment_3954_second_game_solve.json",
            }
        ],
    }


def _generic_recurring_payload() -> dict[str, object]:
    return {
        "experiment": "prior_generic_arcmemo_transfer",
        "concept_memory": [
            {
                "name": "button_cycle_a",
                "family": "discrete_set_permutation",
                "when_it_applies": "Button clicks cycle latent slots.",
                "effect": "Reuse the visible button as a deterministic permutation.",
            },
            {
                "name": "button_cycle_b",
                "family": "discrete_set_permutation",
                "when_it_applies": "A different button permutes the same set.",
                "effect": "Represent the click as a reusable permutation operation.",
            },
        ],
    }


def _exp4035_payload() -> dict[str, object]:
    return {
        "experiment": "experiment_4035_hierarchical_search_over_vc33_wm",
        "honest_verdict": "complete: search_layer_no_solve_vc33_real_env_confirmation_failed",
        "real_env_confirmed": False,
        "new_levels_solved_this_task": 0,
        "action_count": 70,
        "nodes_expanded": 169,
        "inference_substrate": "offline_arc_agi3_planning_search_over_verified_world_model",
    }


def _exp4038_payload() -> dict[str, object]:
    return {
        "experiment": "experiment_4038_seventh_game_explore_first",
        "honest_verdict": "success: seventh_game_solved_dc22-fdcac232_at_action_20",
        "game_solved": True,
        "real_env_confirmed": True,
        "candidate_baseline_actions": 59,
        "first_solve_at_action": 20,
        "exploration_actions_used": 2,
        "target_game": "dc22-fdcac232",
        "induced_mechanic": (
            "Observed dc22 movement and buezna toggle transitions before planning; "
            "visible buezna clicks toggle same-letter blockers and navigation reaches the goal."
        ),
        "inference_substrate": "offline_arc_agi3_explore_first_first_solve",
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_req_learn_4039_spec_declares_exp4039_contract() -> None:
    """REQ-LEARN-4039: OpenSpec declares v6 library transfer and required fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4039" in spec
    assert "SCENARIO-LEARN-4039" in spec
    assert "experiment_4039_arcmemo_concept_library_v6.json" in spec
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4039_compresses_recurring_concepts_into_named_abstraction() -> None:
    """REQ-LEARN-4039-1: recurring raw concepts become documented lambda abstractions."""

    raw = collect_raw_concepts([None, {"concept_memory": [{}, "invalid"]}, _concept_payload()])
    library = build_compressed_library(raw)

    assert len(library) == 1
    abstraction = library[0]
    assert abstraction["name"] == "click_state_transform_then_goal_commit"
    assert "lambda" in abstraction["lambda_abstraction"]
    assert "click_state_transform" in abstraction["signature"]
    assert "why" in abstraction["documentation"].lower()
    assert abstraction["source_concepts"] == [
        "object_click_count_match",
        "pattern_match_then_navigate",
        "select_then_place",
    ]


def test_req_learn_4039_generic_recurring_family_gets_compressed() -> None:
    """REQ-LEARN-4039-1: non-click recurring families still become named ops."""

    raw = collect_raw_concepts([_generic_recurring_payload()])
    library = build_compressed_library(raw)

    assert library[0]["name"] == "discrete_set_permutation_compressed_abstraction"
    assert library[0]["source_concepts"] == ["button_cycle_a", "button_cycle_b"]
    assert "button" in library[0]["match_tokens"]


def test_scenario_learn_4039_v6_beats_cold_and_v5_on_confirmed_content() -> None:
    """SCENARIO-LEARN-4039: v6 wins only when compressed memory beats both baselines."""

    artifact = build_transfer_artifact(
        prior_artifacts=[_concept_payload()],
        exp4035=_exp4035_payload(),
        exp4038=_exp4038_payload(),
        duration_s=0.25,
    )

    assert artifact["solve_transfer_win"] is True
    assert artifact["actions_cold"] == 59
    assert artifact["actions_v5"] == 20
    assert artifact["actions_v6"] == 18
    assert artifact["induction_calls_cold"] == 1
    assert artifact["induction_calls_v5"] == 1
    assert artifact["induction_calls_v6"] == 0
    assert artifact["n_named_abstractions"] == 1
    assert artifact["honest_verdict"] == "success: arcmemo_v6_library_transfer_59to18_actions"
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact_schema_errors(artifact) == []

    assert artifact["excluded_content"][0]["content_id"] == "exp4035"
    assert artifact["per_content_costs"][0]["content_id"] == "exp4038"
    assert artifact["per_content_costs"][0]["matched_abstraction"] == "click_state_transform_then_goal_commit"


def test_scenario_learn_4039_no_transfer_without_matching_library() -> None:
    """REQ-LEARN-4039-4: storing concepts is not enough when v6 does not reduce cost."""

    artifact = build_transfer_artifact(
        prior_artifacts=[_unrelated_concept_payload()],
        exp4035=_exp4035_payload(),
        exp4038=_exp4038_payload(),
        duration_s=0.1,
    )

    assert artifact["solve_transfer_win"] is False
    assert artifact["actions_cold"] == 59
    assert artifact["actions_v5"] == 20
    assert artifact["actions_v6"] == 20
    assert artifact["n_named_abstractions"] == 0
    assert artifact["honest_verdict"] == "complete: arcmemo_v6_no_transfer_v6_not_cheaper_than_v5"
    assert artifact_schema_errors(artifact) == []


def test_scenario_learn_4039_missing_or_unconfirmed_content_cannot_claim_transfer() -> None:
    """REQ-LEARN-4039-5: missing and unconfirmed `.373` artifacts are excluded."""

    missing = build_transfer_artifact(
        prior_artifacts=[_concept_payload()],
        exp4035=None,
        exp4038=None,
        duration_s=0.0,
    )
    unconfirmed_4038 = dict(_exp4038_payload())
    unconfirmed_4038["real_env_confirmed"] = False
    unconfirmed = build_transfer_artifact(
        prior_artifacts=[_concept_payload()],
        exp4035=_exp4035_payload(),
        exp4038=unconfirmed_4038,
        duration_s=0.0,
    )

    assert missing["honest_verdict"] == "complete: arcmemo_v6_no_transfer_no_confirmed_373_solve_content"
    assert missing["actions_cold"] == 0
    assert {row["content_id"] for row in missing["excluded_content"]} == {"exp4035", "exp4038"}
    assert unconfirmed["honest_verdict"] == "complete: arcmemo_v6_no_transfer_no_confirmed_373_solve_content"
    assert unconfirmed["excluded_content"][1]["reason"] == "not_real_env_confirmed_solve"
    assert artifact_schema_errors(missing) == []


def test_scenario_learn_4039_no_win_when_v6_only_ties_cold() -> None:
    """REQ-LEARN-4039-4: v6 must beat cold, not merely tie it."""

    exp4038 = dict(_exp4038_payload())
    exp4038["candidate_baseline_actions"] = 18
    artifact = build_transfer_artifact(
        prior_artifacts=[_concept_payload()],
        exp4035=_exp4035_payload(),
        exp4038=exp4038,
        duration_s=0.1,
    )

    assert artifact["actions_cold"] == 18
    assert artifact["actions_v5"] == 20
    assert artifact["actions_v6"] == 18
    assert artifact["solve_transfer_win"] is False
    assert artifact["honest_verdict"] == "complete: arcmemo_v6_no_transfer_v6_not_cheaper_than_cold"


def test_req_learn_4039_confirmed_4035_cost_path_is_counted_without_result_claim() -> None:
    """REQ-LEARN-4039-5: a confirmed 4035 solve would be counted from its own action field."""

    exp4035 = {
        "experiment": "experiment_4035_hierarchical_search_over_vc33_wm",
        "honest_verdict": "success: search_layer_solved_vc33_L1_real_env_confirmed",
        "real_env_confirmed": True,
        "new_levels_solved_this_task": 1,
    }
    artifact = build_transfer_artifact(
        prior_artifacts=[],
        exp4035=exp4035,
        exp4038=None,
        duration_s=0.1,
    )

    assert artifact["per_content_costs"][0]["content_id"] == "exp4035"
    assert artifact["actions_cold"] == 0
    assert artifact["excluded_content"][0]["content_id"] == "exp4038"


def test_req_learn_4039_schema_rejects_non_bare_required_fields() -> None:
    """REQ-LEARN-4039-2: required artifact fields stay bare JSON scalars."""

    artifact = build_transfer_artifact(
        prior_artifacts=[_concept_payload()],
        exp4035=_exp4035_payload(),
        exp4038=_exp4038_payload(),
        duration_s=0.25,
    )
    bad = dict(artifact)
    bad["honest_verdict"] = "finished"
    bad["solve_transfer_win"] = 1
    bad["actions_cold"] = "59"
    bad["actions_v5"] = 20.0
    bad["actions_v6"] = "18"
    bad["n_named_abstractions"] = True
    bad["inference_substrate"] = None
    wrong_substrate = dict(artifact)
    wrong_substrate["inference_substrate"] = "wrong"

    errors = artifact_schema_errors(bad)
    missing = artifact_schema_errors({})
    substrate_errors = artifact_schema_errors(wrong_substrate)

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert any(field in error for error in errors + missing)
    assert any("inference_substrate must equal" in error for error in substrate_errors)


def test_runner_writes_exp4039_result_json(monkeypatch, tmp_path: Path) -> None:
    """SCENARIO-LEARN-4039: runner writes the stable Exp 4039 JSON deliverable."""

    _write_json(tmp_path / "results" / "experiment_3982_arcmemo_solve_transfer.json", _concept_payload())
    _write_json(
        tmp_path / "results" / "experiment_4035_hierarchical_search_over_vc33_wm.json",
        _exp4035_payload(),
    )
    _write_json(
        tmp_path / "results" / "experiment_4038_seventh_game_explore_first.json",
        _exp4038_payload(),
    )
    monkeypatch.setattr(exp, "REPO", tmp_path)

    artifact = exp.run(write=True)

    written = tmp_path / "results" / "experiment_4039_arcmemo_concept_library_v6.json"
    assert artifact["honest_verdict"] == "success: arcmemo_v6_library_transfer_59to18_actions"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
