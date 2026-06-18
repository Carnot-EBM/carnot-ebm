"""Tests for Exp 4373 ar25/ka59/ft09 named-gap E3 continuation.

Spec refs: REQ-PHASE4-4373, SCENARIO-PHASE4-4373.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _write_model(repo: Path, game: str) -> Path:
    path = repo / exp.WORLD_MODEL_PATHS[game]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    return path


def _active_summary(game: str, *, n: int = 4) -> dict:
    return {
        "game": game,
        "active_transitions_collected": n,
        "target_actions": list(exp.TARGET_GAP_ACTIONS[game]),
        "action_counts": {str(exp.TARGET_GAP_ACTIONS[game][0]): n},
        "target_action_counts": {str(exp.TARGET_GAP_ACTIONS[game][0]): n},
        "diverse_object_config_signatures": n,
        "dataset_path": exp.ACTIVE_DATASET_PATHS[game],
        "dataset_sha256": "a" * 64,
        "collection_error": "",
    }


def _row(
    repo: Path,
    game: str,
    *,
    reached: int = 1,
    accuracy: float = 0.5,
    advanced: bool = False,
    active_n: int = 4,
) -> dict:
    return exp.build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=[accuracy],
        active_dataset_summary=_active_summary(game, n=active_n),
        world_model_path=_write_model(repo, game),
        plan=[f"{game}-step"] if advanced else [],
        reproduce_result={
            "game": game,
            "reached_level": reached,
            "claimed_level": exp.PRIOR_BEST_LEVELS[game] + 1,
            "reproduced": advanced,
        },
        residual_gap_class="none" if advanced else exp.RESIDUAL_GAP_CLASSES[game],
        targeted_gap_lemmas=[{"verifier_gated": advanced}],
        plan_source="unit_test",
    )


def test_req_phase4_4373_spec_declares_contract() -> None:
    """REQ-PHASE4-4373: OpenSpec declares the three-game named-gap contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4373" in spec
    assert "SCENARIO-PHASE4-4373" in spec
    assert "experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09.json" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_ar25_ka59_ft09_<n>_reproduced" in spec
    assert "complete_e3_ar25_ka59_ft09_partial" in spec
    assert "ar25` ACTION7 undo-stack transitions" in spec
    assert "`ka59` hidden StepCounter HUD-register transitions" in spec
    assert "`ft09` balanced action/color ACTION6 click coverage" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4373_active_summary_counts_target_actions(tmp_path: Path) -> None:
    """REQ-PHASE4-4373: targeted active summaries expose named-gap coverage."""

    grid0 = np.zeros((4, 4), dtype=int)
    grid1 = grid0.copy()
    grid1[1, 1] = 5
    transitions = [
        exp.CompactTransition(grid0, 7, None, grid1, 0, 0),
        exp.CompactTransition(grid1, 3, None, grid0, 0, 0),
    ]
    path, dataset_sha = exp.write_targeted_active_dataset(
        tmp_path,
        "ar25",
        transitions,
        random_seed=4373,
        collection_method="unit",
    )

    summary = exp.summarize_targeted_transitions(
        game="ar25",
        target_actions=exp.TARGET_GAP_ACTIONS["ar25"],
        transitions=transitions,
        dataset_path=path.relative_to(tmp_path),
        dataset_sha256=dataset_sha,
    )

    assert summary["active_transitions_collected"] == 2
    assert summary["target_action_counts"] == {"7": 1}
    assert summary["action_counts"] == {"3": 1, "7": 1}
    assert len(summary["dataset_sha256"]) == 64
    assert (tmp_path / exp.ACTIVE_DATASET_PATHS["ar25"]).exists()


def test_req_phase4_4373_build_artifact_counts_only_new_l2_reproductions(tmp_path: Path) -> None:
    """REQ-PHASE4-4373: only levels beyond prior L1 count as progress."""

    rows = [
        _row(tmp_path, "ar25", reached=2, accuracy=1.0, advanced=True),
        _row(tmp_path, "ka59", accuracy=0.75),
        _row(tmp_path, "ft09", accuracy=0.4),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=34,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4373,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["verifier_is_oracle"] is True
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert artifact["per_game_scorecard"][0]["residual_gap_class"] == "none"
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_4373_partial_artifact_preserves_named_gap_rows(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-4373: all-partial runs keep three named residual gaps."""

    rows = [_row(tmp_path, game, accuracy=0.6) for game in exp.TARGET_ORDER]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=33,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4373,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_ft09_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_game_scorecard"]] == list(exp.TARGET_ORDER)
    assert [row["residual_gap_class"] for row in artifact["per_game_scorecard"]] == [
        exp.RESIDUAL_GAP_CLASSES["ar25"],
        exp.RESIDUAL_GAP_CLASSES["ka59"],
        exp.RESIDUAL_GAP_CLASSES["ft09"],
    ]
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4373_checksum_binds_active_data_models_plans_results_and_seed(tmp_path: Path) -> None:
    """REQ-PHASE4-4373: checksum binds active data, models, plans, results, and seed."""

    rows = [_row(tmp_path, game) for game in exp.TARGET_ORDER]
    paths = list(exp.WORLD_MODEL_PATHS.values())
    path_hashes = exp.path_hashes(tmp_path, paths)
    active_hashes = {row["game"]: row["active_dataset_sha256"] for row in rows}

    base = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4373,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4373,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes={**active_hashes, "ft09": "b" * 64},
        random_seed=4373,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_4373_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4373: malformed bare fields and scorecard rows fail validation."""

    artifact = {
        "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
        "per_game_scorecard": ["bad-row", {"game": "ka59", "offline_reproduced": "false"}],
        "new_levels_reproduced": "0",
        "reproducible_total_levels": {"value": 33},
        "world_model_paths": [123],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4373",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_game_scorecard[0] must be dict" in errors
    assert "per_game_scorecard[1] missing prior_best_level" in errors
    assert "per_game_scorecard[1] missing active_transitions_collected" in errors
    assert "per_game_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None, "per_game_scorecard": "bad"})
    assert "missing honest_verdict" in missing
    assert "per_game_scorecard must be list" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_4373_missing_env_continues_other_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4373: missing envs block per game without fabrication."""

    for game in ("ka59", "ft09"):
        env = tmp_path / "environment_files" / game
        env.mkdir(parents=True)
        (env / "fixture").write_text("present", encoding="utf-8")
        _write_model(tmp_path, game)

    calls: list[str] = []

    def fake_runner(repo: Path, game: str, random_seed: int, active_budget: int, round_budget: int) -> dict:
        calls.append(f"{game}:{random_seed}:{active_budget}:{round_budget}")
        return _row(repo, game, reached=2, accuracy=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ka59", fake_runner)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ft09", fake_runner)

    artifact = exp.run_experiment(random_seed=4373, active_budget=9, round_budget=1)

    assert calls == ["ka59:4373:9:1", "ft09:4373:9:1"]
    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_2_reproduced"
    assert artifact["new_levels_reproduced"] == 2
    assert [row["checkpoint_status"] for row in artifact["per_game_scorecard"]] == [
        "blocked_offline_env_missing_ar25",
        "new_level_reproduced",
        "new_level_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_4373_gap_writer_and_schema_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4373: partial rows write named gaps and schema failures abort."""

    env = tmp_path / "environment_files" / "ar25"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    _write_model(tmp_path, "ar25")

    def fake_runner(repo: Path, game: str, random_seed: int, active_budget: int, round_budget: int) -> dict:
        return _row(repo, game, accuracy=0.9, advanced=False)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ar25", fake_runner)

    artifact = exp.run_experiment(random_seed=4373, active_budget=9, round_budget=1)

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_ft09_partial"
    gap_text = (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert exp.RESIDUAL_GAP_CLASSES["ar25"] in gap_text

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="Exp4373 artifact schema errors"):
        exp.run_experiment(random_seed=4373, active_budget=9, round_budget=1)


def test_req_phase4_4373_registry_total_parsing() -> None:
    """REQ-PHASE4-4373: registry total parsing is monotonic and optional."""

    text = "reproducible_total_levels: 33   # old\n"

    assert exp._registry_total(text, "reproducible_total_levels") == 33
    assert exp._registry_total(text, "missing") is None
    assert "reproducible_total_levels: 34" in exp._replace_total(
        text,
        "reproducible_total_levels",
        34,
        "new",
    )


def test_req_phase4_4373_prior_plan_helpers(tmp_path: Path) -> None:
    """REQ-PHASE4-4373: prior L1 plans seed next-level gates but do not count alone."""

    assert exp.prior_plan(tmp_path, "ar25") == []

    prior = tmp_path / "results" / "experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(
        json.dumps(
            {
                "per_game_scorecard": [
                    {"game": "ar25", "plan": ["3"], "verifier_accuracy": 0.8},
                    {"game": "ka59", "plan": ["4"], "verifier_accuracy": 0.6},
                ]
            }
        ),
        encoding="utf-8",
    )
    ft09 = tmp_path / "results" / "experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json"
    ft09.write_text(
        json.dumps(
            {
                "per_game_scorecard": [
                    {"game": "ft09", "plan": ["click"], "verifier_accuracy": 1.0}
                ]
            }
        ),
        encoding="utf-8",
    )

    assert exp.prior_plan(tmp_path, "ar25") == ["3"]
    assert exp.prior_plan(tmp_path, "ka59") == ["4"]
    assert exp.prior_plan(tmp_path, "ft09") == ["click"]
    assert exp.prior_plan(tmp_path, "missing") == []

    broken = tmp_path / "results" / "experiment_4362_e3_blocked_mechanic_levels_ar25_ka59.json"
    broken.write_text("{", encoding="utf-8")
    assert exp.prior_plan(tmp_path, "ar25") == []


def test_req_phase4_4373_pure_helper_edge_branches(tmp_path: Path) -> None:
    """REQ-PHASE4-4373: deterministic helper edge branches remain covered."""

    outside = tmp_path.parent / "outside_4373.py"
    outside.write_text("# outside\n", encoding="utf-8")
    raw = type(
        "RawTransition",
        (),
        {
            "grid": np.zeros((2, 2), dtype=int),
            "action": 6,
            "data": {"x": 1, "y": 1},
            "next_grid": np.ones((2, 2), dtype=int),
            "level_before": 0,
            "level_after": 1,
        },
    )()
    blocked = [exp.blocked_game_row(tmp_path, game) for game in exp.TARGET_ORDER]
    gap_path = tmp_path / exp.GAP_RELATIVE_PATH

    assert exp._relative_or_absolute(tmp_path, outside) == str(outside)
    assert exp._state_signature(np.array([1, 2, 3])) == ()
    assert exp._changed_cells(np.zeros((1, 2)), np.zeros((2, 1))) == -1
    compact = exp._to_compact([raw])
    assert compact[0].data == {"x": 1, "y": 1}
    assert exp._combined_verdict(blocked) == "blocked_offline_env_missing_ar25_ka59_ft09"
    assert exp._residual_mismatch_class("ar25", []) == "none"
    assert exp._residual_mismatch_class("ar25", [{"error": "boom"}]) == "engine_runtime_error_gap"
    assert exp._residual_mismatch_class("ar25", [{"action": 7}]) == exp.RESIDUAL_GAP_CLASSES["ar25"]
    assert exp._residual_mismatch_class("ar25", [{"action": 4}]) == "missing_world_model_rule_gap_actions_4"
    assert exp._apply_for_game("ar25") is not None
    assert exp._apply_for_game("ka59") is not None
    assert exp._apply_for_game("ft09") is not None
    with pytest.raises(KeyError):
        exp._apply_for_game("missing")

    exp._write_gap(gap_path, row=_row(tmp_path, "ar25", accuracy=0.8), checksum="a" * 64)
    exp._write_gap(gap_path, row=_row(tmp_path, "ar25", accuracy=0.9), checksum="b" * 64)
    gap_text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.9000" in gap_text
    assert "`" + "b" * 64 + "`" in gap_text
    assert "`" + "a" * 64 + "`" not in gap_text

    assert "reproducible_total_games: 17" in exp._replace_total(
        "reproducible_total_levels: 33\n",
        "reproducible_total_games",
        17,
        "new",
    )
    assert exp._registry_total_from_repo(tmp_path) is None
    registry = tmp_path / exp.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("reproducible_total_levels: 33\n", encoding="utf-8")
    assert exp._registry_total_from_repo(tmp_path) == 33
