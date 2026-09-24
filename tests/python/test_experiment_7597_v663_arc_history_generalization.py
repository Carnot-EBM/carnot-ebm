"""Tests for REQ-ARC-WMTE-7597 live ARC history generalization."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7597_v663_arc_history_generalization as exp


def observation(value: int, *, level: int = 0) -> dict[str, Any]:
    """Build one exact public observation with no hidden game state."""

    return {
        "frame": [[[value, value + 1], [value + 2, value + 3]]],
        "level": level,
        "legal_actions": ["ACTION1", "ACTION2", "RESET"],
        "termination": "NOT_FINISHED",
    }


def step(
    index: int,
    before: int | None,
    action: int | str,
    after: int | None,
    *,
    level_before: int = 0,
    level_after: int = 0,
) -> dict[str, Any]:
    """Build one action opportunity in the raw qualified-observer schema."""

    return {
        "action_index": index,
        "observation": None if before is None else observation(before, level=level_before),
        "action": {"kind": action, "coordinates": None},
        "outcome": None if after is None else observation(after, level=level_after),
        "reset_boundary": action == "RESET",
        "level_boundary": level_after != level_before,
        "terminated": after is None,
    }


def episode(game: str, seed: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one raw episode that can be independently reduced."""

    return {
        "schema": exp.RAW_EPISODE_SCHEMA,
        "episode_id": f"{game}:{seed}",
        "game": game,
        "seed": seed,
        "policy": "E3AgentPolicy",
        "action_limit": 600,
        "induction_disabled": True,
        "adapter_withheld": True,
        "stored_solutions_withheld": True,
        "game_source_read": False,
        "hidden_state_read": False,
        "offline_ground_truth_bfs": False,
        "live_llm_invoked": False,
        "steps": steps,
        "trajectory_supervisor": {
            "enabled": True,
            "mode": "shadow",
            "arm_outcomes": {"frontier_reset": {"fired": 0, "helped": 0}},
            "redirects": [],
        },
        "qualified_observer": {"enabled": True, "error_count": 0},
        "termination": {"reason": "action_limit", "action_opportunities": len(steps)},
    }


def repeating_episode(game: str, seed: int, *, target_offset: int = 0) -> dict[str, Any]:
    """Make repeated short-history keys and unique longer-history contexts."""

    steps = [step(0, None, "RESET", 0)]
    debruijn_prefix = (0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0)
    for index, before in enumerate(debruijn_prefix, start=1):
        after = (before + 3 + target_offset) % 7
        steps.append(step(index, before, 1, after))
    return episode(game, seed, steps)


def test_reduce_episode_reports_exact_key_support_and_unknown_singletons() -> None:
    """SCENARIO-ARC-WMTE-7597-CAUSAL-REDUCTION preserves exact operands."""

    raw = episode(
        "su15",
        exp.SEEDS[0],
        [
            step(0, None, "RESET", 1),
            step(1, 1, 1, 2),
            step(2, 1, 1, 2),
            step(3, 1, 1, 9),
            step(4, 4, 2, 5),
        ],
    )

    reduced = exp.reduce_episode(raw)
    zero = reduced["per_history"]["0"]
    repeated = next(row for row in zero["key_rows"] if row["support_count"] == 3)

    assert repeated["status"] == "conflicting"
    assert repeated["target_distribution"] == [
        {"target_sha256": exp.frame_sha256(observation(2)["frame"]), "count": 2},
        {"target_sha256": exp.frame_sha256(observation(9)["frame"]), "count": 1},
    ]
    assert repeated["conditional_conflict_numerator"] == 1
    assert repeated["conditional_conflict_denominator"] == 3
    assert repeated["conditional_conflict_rate"] == pytest.approx(1 / 3)
    assert zero["singleton_unknown_key_count"] == 1
    assert zero["repeated_key_count"] == 1
    assert zero["conflict_numerator"] == 1
    assert zero["conflict_denominator"] == 3


def test_reset_and_level_boundaries_clear_history_before_new_keys() -> None:
    """REQ-ARC-WMTE-7597 blocks histories at every reset and level boundary."""

    raw = episode(
        "sp80",
        exp.SEEDS[0],
        [
            step(0, None, "RESET", 1),
            step(1, 1, 1, 2),
            step(2, 2, "RESET", 1),
            step(3, 1, 2, 3),
            step(4, 3, 3, 4, level_before=0, level_after=1),
            step(5, 4, 4, 5, level_before=1, level_after=1),
        ],
    )

    reduced = exp.reduce_episode(raw)
    by_index = {row["action_index"]: row for row in reduced["transition_rows"]}

    assert by_index[2]["history_items_used"] == {"0": 0, "1": 0, "2": 0, "4": 0}
    assert by_index[3]["history_items_used"] == {"0": 0, "1": 0, "2": 0, "4": 0}
    assert by_index[5]["history_items_used"] == {"0": 0, "1": 0, "2": 0, "4": 0}
    assert reduced["reset_count"] == 2
    assert reduced["level_boundary_count"] == 1
    assert reduced["eligible_transition_count"] == 5


def test_second_seed_checks_only_first_seed_repeated_candidate_keys() -> None:
    """SCENARIO-ARC-WMTE-7597-CROSS-SEED-SUPPORT censors unknown keys."""

    first = episode(
        "ft09",
        exp.SEEDS[0],
        [
            step(0, None, "RESET", 0),
            step(1, 1, 1, 2),
            step(2, 1, 1, 2),
            step(3, 7, 2, 8),
        ],
    )
    second = episode(
        "ft09",
        exp.SEEDS[1],
        [
            step(0, None, "RESET", 0),
            step(1, 1, 1, 2),
            step(2, 1, 1, 9),
            step(3, 6, 2, 8),
        ],
    )

    comparison = exp.compare_seed_reductions(exp.reduce_episode(first), exp.reduce_episode(second))
    zero = comparison["per_history"]["0"]

    assert zero["candidate_key_count"] == 2
    assert zero["repeated_candidate_key_count"] == 1
    assert zero["supported_predictive_denominator"] == 2
    assert zero["exact_prediction_numerator"] == 1
    assert zero["cross_seed_conflict_numerator"] == 1
    assert zero["cross_seed_conflict_rate"] == 0.5
    assert zero["unseen_key_censored_count"] == 1
    assert zero["singleton_candidate_censored_count"] == 0


def test_longer_unique_keys_cannot_win_without_matched_common_support() -> None:
    """SCENARIO-ARC-WMTE-7597-MATCHED-SUPPORT excludes singleton purity."""

    first = exp.reduce_episode(repeating_episode("sb26", exp.SEEDS[0]))
    second = exp.reduce_episode(repeating_episode("sb26", exp.SEEDS[1]))
    comparison = exp.compare_seed_reductions(first, second)

    assert comparison["per_history"]["0"]["repeated_candidate_key_count"] >= 2
    assert comparison["matched_common_support"]["transition_count"] == 0
    assert comparison["matched_common_support"]["support_floor_key_count"] == 0
    assert comparison["matched_common_support"]["eligible_for_cross_game"] is False
    assert comparison["matched_common_support"]["reason"] == "insufficient_matched_repeat_support"


def supported_pair(game: str, *, n_keys: int, conflict: bool = False) -> tuple[dict, dict]:
    """Build two seeds with repeated keys at every history length."""

    first_steps = [step(0, None, "RESET", 0)]
    second_steps = [step(0, None, "RESET", 0)]
    index = 1
    for key_index in range(n_keys):
        context = 100 + key_index * 10
        for repeat in range(2):
            # Each key gets a fresh reset so every history arm uses the same empty context.
            first_steps.append(step(index, context, 1, context + 1))
            second_target = context + 2 if conflict and repeat == 1 else context + 1
            second_steps.append(step(index, context, 1, second_target))
            index += 1
            first_steps.append(step(index, context + 1, "RESET", context))
            second_steps.append(step(index, second_target, "RESET", context))
            index += 1
    return (
        episode(game, exp.SEEDS[0], first_steps),
        episode(game, exp.SEEDS[1], second_steps),
    )


def test_cross_game_floor_needs_twenty_matched_keys_on_three_games() -> None:
    """REQ-ARC-WMTE-7597 suppresses comparison below the declared support floor."""

    pairs = []
    for game in exp.GAMES:
        first, second = supported_pair(game, n_keys=19)
        pairs.append(
            exp.compare_seed_reductions(exp.reduce_episode(first), exp.reduce_episode(second))
        )

    result = exp.cross_game_history_comparison(pairs, bootstrap_seed=exp.BOOTSTRAP_SEED)

    assert result["ready"] is False
    assert result["reason"] == "insufficient_support"
    assert result["qualifying_game_count"] == 0
    assert result["required_game_count"] == 3
    assert result["minimum_matched_repeated_keys_per_game"] == 20
    assert result["bootstrap_differences"] is None


def test_cross_game_bootstrap_uses_games_as_clusters() -> None:
    """REQ-ARC-WMTE-7597 bootstraps qualifying games, not seeds or windows."""

    pairs = []
    for game_index, game in enumerate(exp.GAMES[:3]):
        first, second = supported_pair(game, n_keys=20, conflict=game_index == 0)
        pairs.append(
            exp.compare_seed_reductions(exp.reduce_episode(first), exp.reduce_episode(second))
        )

    result = exp.cross_game_history_comparison(pairs, bootstrap_seed=123, n_bootstrap=100)

    assert result["ready"] is True
    assert result["qualifying_games"] == list(exp.GAMES[:3])
    assert result["independent_cluster_count"] == 3
    assert result["bootstrap_draw_count"] == 100
    assert set(result["bootstrap_differences"]) == {"1", "2", "4"}
    for row in result["bootstrap_differences"].values():
        assert row["comparison"] == "history_0_conflict_rate_minus_longer_history"
        assert row["cluster_unit"] == "game"


def test_frozen_episode_plan_has_twelve_real_e3_units() -> None:
    """SCENARIO-ARC-WMTE-7597-LIVE-EPISODES freezes games, seeds, and budget."""

    plan = exp.frozen_episode_plan()

    assert len(plan) == 12
    assert {(row["game"], row["seed"]) for row in plan} == {
        (game, seed) for game in exp.GAMES for seed in exp.SEEDS
    }
    assert all(row["policy"] == "E3AgentPolicy" for row in plan)
    assert all(row["action_limit"] == 600 for row in plan)
    assert all(row["adapter_withheld"] is True for row in plan)
    assert all(row["induction_disabled"] is True for row in plan)
    assert all(row["live_llm_invoked"] is False for row in plan)


def test_raw_episode_validator_rejects_forbidden_or_incomplete_evidence() -> None:
    """REQ-ARC-WMTE-7597 rejects hidden-state, adapter, and model contamination."""

    raw = repeating_episode("g50t", exp.SEEDS[0])
    assert exp.validate_raw_episode(raw) == []

    mutations = {
        "wrong_policy": ("policy", "GameAdapter"),
        "induction_on": ("induction_disabled", False),
        "adapter_used": ("adapter_withheld", False),
        "stored_solution": ("stored_solutions_withheld", False),
        "source_read": ("game_source_read", True),
        "hidden_state": ("hidden_state_read", True),
        "ground_truth": ("offline_ground_truth_bfs", True),
        "llm": ("live_llm_invoked", True),
        "observer_missing": ("qualified_observer", {"enabled": False}),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(raw)
        changed[field] = value
        assert expected in exp.validate_raw_episode(changed)

    changed = deepcopy(raw)
    changed["steps"][1]["observation"].pop("legal_actions")
    assert "legal_actions_missing:1" in exp.validate_raw_episode(changed)


def test_raw_episode_validator_covers_malformed_public_rows() -> None:
    """REQ-ARC-WMTE-7597 rejects malformed identities, rows, and public frames."""

    raw = repeating_episode("g50t", exp.SEEDS[0])
    mutations = []
    for field, value in (("game", "outside-panel"), ("seed", -1), ("steps", None)):
        changed = deepcopy(raw)
        changed[field] = value
        mutations.append(changed)
    changed = deepcopy(raw)
    changed["steps"] = [None]
    mutations.append(changed)
    changed = deepcopy(raw)
    changed["steps"][1]["observation"] = {"legal_actions": []}
    mutations.append(changed)
    changed = deepcopy(raw)
    changed["steps"][1]["outcome"] = []
    mutations.append(changed)
    changed = deepcopy(raw)
    changed["steps"][1]["action"] = None
    mutations.append(changed)

    errors = [exp.validate_raw_episode(changed) for changed in mutations]
    assert "game_not_frozen" in errors[0]
    assert "seed_not_frozen" in errors[1]
    assert "steps_invalid" in errors[2]
    assert "action_index_invalid:0" in errors[3]
    assert "observation_invalid:1" in errors[4]
    assert "outcome_invalid:1" in errors[5]
    assert "action_invalid:1" in errors[6]
    with pytest.raises(ValueError, match="raw_episode_invalid"):
        exp.reduce_episode(mutations[0])


def test_reducer_defensive_branches_and_row_signs(capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-ARC-WMTE-7597 retains explicit errors and signed descriptive rows."""

    exp.progress(0.0, "test", "boundary", unit=1)
    assert "phase=test event=boundary" in capsys.readouterr().out
    assert exp._modal_target({"target_distribution": []}) == ""

    first = exp.reduce_episode(repeating_episode("su15", exp.SEEDS[0]))
    second = exp.reduce_episode(repeating_episode("su15", exp.SEEDS[1]))
    wrong_game = deepcopy(second)
    wrong_game["game"] = "sp80"
    with pytest.raises(ValueError, match="seed_game_mismatch"):
        exp.compare_seed_reductions(first, wrong_game)
    wrong_seed = deepcopy(second)
    wrong_seed["seed"] = exp.SEEDS[0]
    with pytest.raises(ValueError, match="seed_order_mismatch"):
        exp.compare_seed_reductions(first, wrong_seed)

    reduced = exp.independent_reduce_rows(
        [
            {"unit": "a", "arm": "x", "numerator": 1, "denominator": 2, "absolute_metric": 0.5},
            {"unit": "b", "arm": "x", "numerator": 0, "denominator": 2, "absolute_metric": -0.5},
        ]
    )
    assert reduced["sign_counts"]["positive"] == 1
    assert reduced["sign_counts"]["negative"] == 1


def test_raw_receipt_reducer_reports_each_custody_failure(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 names missing, malformed, and identity-mismatched raw logs."""

    missing = {
        "game": "su15",
        "seed": exp.SEEDS[0],
        "path": str(tmp_path / "missing"),
        "sha256": "x",
    }

    invalid_raw = repeating_episode("su15", exp.SEEDS[0])
    invalid_raw["policy"] = "wrong"
    invalid_path = tmp_path / "invalid.json"
    exp.atomic_json(invalid_path, invalid_raw)

    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{", encoding="utf-8")

    other_raw = repeating_episode("sp80", exp.SEEDS[0])
    other_path = tmp_path / "other.json"
    exp.atomic_json(other_path, other_raw)

    receipts = [
        missing,
        {
            "game": "su15",
            "seed": exp.SEEDS[0],
            "path": str(invalid_path),
            "sha256": exp.sha256_file(invalid_path),
        },
        {
            "game": "su15",
            "seed": exp.SEEDS[0],
            "path": str(broken_path),
            "sha256": exp.sha256_file(broken_path),
        },
        {
            "game": "su15",
            "seed": exp.SEEDS[0],
            "path": str(other_path),
            "sha256": exp.sha256_file(other_path),
        },
    ]
    errors = exp.reduce_raw_episode_receipts(receipts)["errors"]
    assert any(error.startswith("raw_missing:su15") for error in errors)
    assert any(error.startswith("raw_invalid:su15") for error in errors)
    assert any(error.startswith("raw_reduce_error:su15") for error in errors)
    assert any(error.startswith("raw_identity_mismatch:su15") for error in errors)


def test_supervisor_reduction_uses_only_actual_fired_and_helped_counts() -> None:
    """REQ-ARC-WMTE-7597 reports no-firing as nothing to refine."""

    empty = exp.reduce_trajectory_supervisor(
        {"arm_outcomes": {"reset": {"fired": 0, "helped": 0}}, "redirects": []}
    )
    assert empty == {
        "fired": 0,
        "helped": 0,
        "supervisor_refinement_supported": False,
        "reason": "no_firings_nothing_to_refine",
    }

    observed = exp.reduce_trajectory_supervisor(
        {
            "arm_outcomes": {
                "reset": {"fired": 3, "helped": 1},
                "tier": {"fired": 2, "helped": 2},
            },
            "redirects": [{"arm": "reset"}] * 5,
        }
    )
    assert observed["fired"] == 5
    assert observed["helped"] == 3
    assert observed["supervisor_refinement_supported"] is False
    assert observed["reason"] == "measurement_only_no_arm_change_authorized"


def test_learning_lifecycle_persists_reloads_and_rejects_duplicate(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 exercises delayed learning without model weights."""

    receipt = exp.exercise_learning_lifecycle(tmp_path / "learning")

    assert receipt["operations"] == [
        "predict",
        "release",
        "update",
        "persist",
        "reload",
        "duplicate_rejection",
    ]
    assert receipt["prediction_label_available"] is False
    assert receipt["state_changed_after_update"] is True
    assert receipt["reload_equal"] is True
    assert receipt["duplicate_rejected"] is True
    assert receipt["state_unchanged_after_duplicate"] is True
    assert receipt["model_weights_changed"] is False
    assert receipt["passed"] is True


def test_preconditions_check_registry_requirement_and_private_ownership(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 checks custody without requiring future output paths."""

    root = Path(__file__).resolve().parents[2]
    private = tmp_path / "owned"
    checks, hashes = exp.collect_preconditions(root, private)

    assert checks and all(row["passed"] for row in checks)
    assert set(exp.GAMES) <= {
        game for row in checks if row["check"] == "registry_precheck" for game in row["observed"]
    }
    assert exp.REQUIREMENT_ID in (root / exp.SPEC_REL).read_text(encoding="utf-8")
    checked_paths = {str(row["path"]) for row in checks}
    assert exp.RESULT_REL.as_posix() not in checked_paths
    assert exp.RAW_REL.as_posix() not in checked_paths
    assert hashes["ops/arc_solve_registry.yaml"].startswith("sha256:")


def test_blocked_artifact_names_all_failed_gate_operands() -> None:
    """REQ-ARC-WMTE-7597 makes missing upstream work terminal blocked evidence."""

    artifact = exp.build_blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        check="missing_source",
        upstream="worktree",
        path="missing.json",
        field="is_file",
        op="==",
        expected=True,
        observed=False,
    )

    assert artifact["honest_verdict"] == "complete_blocked_missing_source"
    assert artifact["verdict_class"] == "blocked"
    assert set(artifact["gate_check_summary"]) == {
        "check",
        "upstream",
        "path",
        "field",
        "op",
        "expected",
        "observed",
    }
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["arc_history_measurement_ready_score"] == 0
    assert artifact["history_support_ready_score"] == 0
    assert exp.validate_artifact(artifact) == []


def write_raw_panel(tmp_path: Path) -> list[dict[str, Any]]:
    """Write a complete synthetic twelve-episode panel and byte receipts."""

    receipts = []
    for game in exp.GAMES:
        for seed in exp.SEEDS:
            raw = repeating_episode(game, seed)
            path = tmp_path / "raw" / f"{game}-{seed}.json"
            exp.atomic_json(path, raw)
            receipts.append(
                {
                    "episode_id": raw["episode_id"],
                    "game": game,
                    "seed": seed,
                    "path": str(path),
                    "sha256": exp.sha256_file(path),
                    "process_exit_code": 0,
                    "copied_after_process_exit": True,
                }
            )
    return receipts


def test_fresh_raw_log_reduction_authenticates_all_twelve_episodes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 independently reduces exact raw-log bytes."""

    receipts = write_raw_panel(tmp_path)
    panel = exp.reduce_raw_episode_receipts(receipts)

    assert panel["valid"] is True
    assert panel["errors"] == []
    assert panel["episode_count"] == 12
    assert len(panel["per_episode_reductions"]) == 12
    assert len(panel["per_game_comparisons"]) == 6
    assert len(panel["per_game_results"]) == 48
    assert {row["history_length"] for row in panel["per_game_results"]} == {0, 1, 2, 4}
    assert panel["cross_game_comparison"]["reason"] == "insufficient_support"
    assert panel["history_support_ready_score"] == 0

    Path(receipts[0]["path"]).write_text("{}\n", encoding="utf-8")
    changed = exp.reduce_raw_episode_receipts(receipts)
    assert changed["valid"] is False
    assert any(error.startswith("raw_hash_mismatch:") for error in changed["errors"])


def test_rows_retain_absolute_metrics_operands_and_provenance(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 emits one auditable row per episode and history arm."""

    panel = exp.reduce_raw_episode_receipts(write_raw_panel(tmp_path))
    required = {
        "unit",
        "arm",
        "absolute_metric",
        "numerator",
        "denominator",
        "seed",
        "direction",
        "missing",
        "censored",
        "provenance",
    }

    assert len(panel["rows"]) == 48
    assert all(required <= set(row) for row in panel["rows"])
    assert all(row["unit"].count(":") == 1 for row in panel["rows"])
    assert all(row["arm"].startswith("history_") for row in panel["rows"])
    assert panel["independent_reduction"] == exp.independent_reduce_rows(panel["rows"])


def test_terminal_artifact_is_complete_null_when_support_floor_is_not_met(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7597-TERMINAL separates measurement readiness from benefit."""

    artifact = exp.build_test_artifact(tmp_path)
    errors = exp.validate_artifact(artifact)

    assert errors == []
    assert artifact["honest_verdict"] == "complete_null_insufficient_history_support"
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["live_llm_invoked"] is False
    assert artifact["inference_substrate"] == (
        "offline_arcade_live_agent_runtime_self_discovery_no_llm"
    )
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert all(value == 0 for value in artifact["invocation_counts"].values())
    assert artifact["arc_history_measurement_ready_score"] == 1
    assert artifact["history_support_ready_score"] == 0
    assert artifact["acceptance_gate_results"]["validity"]["passed"] is True
    assert artifact["acceptance_gate_results"]["readiness"]["passed"] is True
    assert artifact["acceptance_gate_results"]["benefit"]["passed"] is False
    assert artifact["acceptance_gate_results"]["retention"]["passed"] is True
    assert artifact["acceptance_gate_results"]["freshness"]["passed"] is True
    assert artifact["sample_size_budget"]["intended_independent_units"] == 6
    assert artifact["sample_size_budget"]["observed_independent_units"] == 6
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["new_solve_claimed"] is False
    assert artifact["official_leaderboard_score_claimed"] is False
    assert artifact["production_defaults_changed"] is False
    assert artifact["acceptance_threshold_changed"] is False
    assert artifact["qwen_thinking_behavior_changed"] is False
    assert artifact["supervisor_arms_changed"] is False
    assert artifact["reproducibility_checksum"] == exp.reproducibility_checksum(artifact)
    assert set(exp.REQUIRED_FIELD_PRINCIPLES) <= set(artifact["field_principles"])
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"].values())


def test_artifact_validator_recomputes_scores_rows_and_raw_hashes(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 rejects readiness, row, or evidence mutations."""

    artifact = exp.build_test_artifact(tmp_path)

    changed = deepcopy(artifact)
    changed["history_support_ready_score"] = 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "history_support_ready_score" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["rows"][0]["numerator"] += 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "independent_reduction" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["live_llm_invoked"] = True
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "live_llm_invoked" in exp.validate_artifact(changed)


def test_artifact_validator_rejects_schema_and_gate_mutations(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 fail-closes every required terminal field."""

    artifact = exp.build_test_artifact(tmp_path)
    mutations = {
        "identity": ("schema", "wrong"),
        "honest_verdict": ("honest_verdict", "partial"),
        "verdict_class": ("verdict_class", "unknown"),
        "MODEL_SPECS": ("MODEL_SPECS", ["forbidden"]),
        "invocation_counts": ("invocation_counts", {"model_loads": 1}),
        "field_principles": ("field_principles", {}),
        "verifier_is_oracle": ("verifier_is_oracle", True),
        "new_solve_claimed": ("new_solve_claimed", True),
        "inference_substrate": ("inference_substrate", "wrong"),
        "inference_substrate_class": ("inference_substrate_class", "wrong"),
        "planned_inference_substrate_class": ("planned_inference_substrate_class", "wrong"),
        "rows": ("rows", []),
        "sample_size_budget": ("sample_size_budget", {}),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "bad"
    assert "reproducibility_checksum" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["independent_reduction"] = {}
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "independent_reduction" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    Path(changed["raw_episode_receipts"][0]["path"]).unlink()
    changed["arc_history_measurement_ready_score"] = 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "arc_history_measurement_ready_score" in exp.validate_artifact(changed)


def test_artifact_validator_rejects_gate_mutations(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 requires named gates with principles and exact polarity."""

    artifact = exp.build_test_artifact(tmp_path)
    changes = [
        ("acceptance_gate_results", {}),
        ("acceptance_gate_results.validity.principle", ""),
        ("acceptance_gate_results.readiness.passed", False),
        ("acceptance_gate_results.benefit.passed", True),
    ]
    expected = ["acceptance_gate_results", "gate_principles", "readiness_gate", "benefit_gate"]
    for path, value in changes:
        changed = deepcopy(artifact)
        if "." not in path:
            changed[path] = value
        else:
            _, gate, field = path.split(".")
            changed["acceptance_gate_results"][gate][field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected.pop(0) in exp.validate_artifact(changed)


def test_blocked_validator_rejects_inconsistent_blocked_fields() -> None:
    """REQ-ARC-WMTE-7597 keeps blocked artifacts fail-closed and fully named."""

    artifact = exp.build_blocked_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.1,
        check="x",
        upstream="u",
        path="p",
        field="f",
        op="==",
        expected=True,
        observed=False,
    )
    mutations = {
        "gate_check_summary": ("gate_check_summary", {}),
        "inference_substrate_class": ("inference_substrate_class", "wrong"),
        "arc_history_measurement_ready_score": ("arc_history_measurement_ready_score", 1),
        "history_support_ready_score": ("history_support_ready_score", 1),
    }
    for expected, (field, value) in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert expected in exp.validate_artifact(changed)


def test_artifact_builder_covers_disqualified_and_supported_verdicts(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 distinguishes failed execution from supported null evidence."""

    artifact = exp.build_test_artifact(tmp_path)
    panel = {
        "valid": True,
        "arc_history_measurement_ready_score": 1,
        "history_support_ready_score": 1,
        "rows": artifact["rows"],
        "per_game_results": artifact["rows"],
        "per_episode_reductions": artifact["per_episode_reductions"],
        "per_game_comparisons": artifact["per_game_comparisons"],
        "support_versus_conflict_curve": artifact["support_versus_conflict_curve"],
        "cross_game_comparison": {"ready": True},
        "trajectory_supervisor": artifact["trajectory_supervisor"],
    }
    common = dict(
        repo_root=tmp_path,
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        panel=panel,
        raw_episode_receipts=artifact["raw_episode_receipts"],
        lifecycle=artifact["learning_lifecycle"],
        source_hashes={"source": "hash"},
        e2e_receipts=artifact["e2e_receipts"],
        terminal_receipts=artifact["terminal_validation_receipts"],
    )
    supported = exp.build_artifact(
        validation_receipts=artifact["scoped_validation_receipts"], **common
    )
    failed = exp.build_artifact(validation_receipts=[], **common)
    assert supported["honest_verdict"].startswith("complete_null_history_support_measured")
    assert failed["verdict_class"] == "disqualified"


def test_validation_and_e2e_commands_are_scoped_and_private(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7597 freezes changed files and private subprocess paths."""

    root = Path(__file__).resolve().parents[2]
    validation = exp.build_validation_commands(root, tmp_path / "validation")
    e2e = exp.build_e2e_commands(root, tmp_path / "e2e")

    assert [row.name for row in validation] == list(exp.REQUIRED_SCOPED_CHECKS)
    assert [row.name for row in e2e] == [
        "e2e_009",
        "e2e_010",
        "e2e_011",
        "e2e_012",
        "e2e_013",
        "llm_off_environment_smoke",
    ]
    joined = " ".join(item for command in (*validation, *e2e) for item in command.argv)
    assert "tests/python/test_experiment_7597_v663_arc_history_generalization.py" in joined
    assert "--no-cov" in joined
    assert "-n 0" in joined
    assert str(tmp_path) in joined
    assert str(root / "results") not in joined

    candidate = tmp_path / "candidate.json"
    terminal = exp.build_terminal_commands(root, candidate)
    assert [row.name for row in terminal] == list(exp.TERMINAL_NAMES)
    assert all(str(candidate) in row.argv for row in terminal)

    hashes = exp._source_hashes(root, {"upstream": "sha256:known"})
    assert hashes["upstream"] == "sha256:known"
    assert exp.MODULE_REL.as_posix() in hashes


def test_cold_replay_runs_raw_reduction_without_policy_or_model(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7597-TERMINAL replays saved logs in a fresh process."""

    artifact = exp.build_test_artifact(tmp_path)
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)

    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []


def test_replay_readers_reject_missing_or_inconsistent_operands(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7597-TERMINAL fail-closes malformed replay inputs."""

    artifact = exp.build_test_artifact(tmp_path)

    changed = deepcopy(artifact)
    changed.pop("raw_episode_receipts")
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "cold-no-raw.json"
    exp.atomic_json(path, changed)
    assert "raw_episode_receipts" in exp.cold_replay(path)

    changed = deepcopy(artifact)
    changed["support_versus_conflict_curve"] = []
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "cold-mismatch.json"
    exp.atomic_json(path, changed)
    assert "cold_support_versus_conflict_curve_mismatch" in exp.cold_replay(path)

    changed = deepcopy(artifact)
    changed["rows"] = None
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "rows-missing.json"
    exp.atomic_json(path, changed)
    assert exp.independent_replay(path) == ["rows"]

    changed = deepcopy(artifact)
    changed["rows"][0]["numerator"] = -1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "rows-invalid.json"
    exp.atomic_json(path, changed)
    assert exp.independent_replay(path)[0].startswith("independent_reduction:")

    changed = deepcopy(artifact)
    changed["independent_reduction"] = {}
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "row-reduction-mismatch.json"
    exp.atomic_json(path, changed)
    assert "independent_reduction" in exp.independent_replay(path)

    changed = deepcopy(artifact)
    changed["raw_episode_receipts"] = None
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "independent-no-raw.json"
    exp.atomic_json(path, changed)
    assert "raw_episode_receipts" in exp.independent_replay(path)

    changed = deepcopy(artifact)
    changed["rows"] = []
    changed["independent_reduction"] = exp.independent_reduce_rows([])
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    path = tmp_path / "raw-row-mismatch.json"
    exp.atomic_json(path, changed)
    assert "raw_rows" in exp.independent_replay(path)
