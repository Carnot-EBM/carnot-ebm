"""Tests for Exp 4363 active-data mechanic-limited E3 tails.

Spec refs: REQ-PHASE4-087, SCENARIO-PHASE4-087.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4363_e3_mechanic_limited_tails_tr87_ft09 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _write_model(repo: Path, game: str) -> Path:
    path = repo / exp.WORLD_MODEL_PATHS[game]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    return path


def _active_summary(game: str, *, n: int = 3) -> dict:
    return {
        "game": game,
        "active_transitions_collected": n,
        "target_actions": list(exp.TARGET_GAP_ACTIONS[game]),
        "action_counts": {str(exp.TARGET_GAP_ACTIONS[game][0]): n},
        "target_action_counts": {str(exp.TARGET_GAP_ACTIONS[game][0]): n},
        "diverse_object_config_signatures": n,
        "dataset_path": f"results/arc_e3/{game}/active_data_4363.json",
        "dataset_sha256": "a" * 64,
        "collection_error": "",
    }


def _row(
    repo: Path,
    game: str,
    *,
    accuracy: float = 0.5,
    reproduced: bool = False,
    reached: int = 0,
    active_n: int = 3,
) -> dict:
    return exp.build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=[accuracy],
        active_dataset_summary=_active_summary(game, n=active_n),
        world_model_path=_write_model(repo, game),
        plan=[f"{game}-step"] if reproduced else [],
        reproduce_result={
            "game": game,
            "reached_level": reached,
            "claimed_level": 1,
            "reproduced": reproduced,
        },
        residual_mismatch_class="none" if reproduced else exp.RESIDUAL_GAP_CLASSES[game],
        plan_source="unit_test",
    )


def test_req_phase4_087_spec_declares_exp4363_contract() -> None:
    """REQ-PHASE4-087: OpenSpec declares the active-data tr87/ft09 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-087" in spec
    assert "SCENARIO-PHASE4-087" in spec
    assert "experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json" in spec
    assert "active_collect import" in spec
    assert "tr87` actions 1/2/3/4" in spec
    assert "`ft09` action 6" in spec
    assert "success_e3_tr87_ft09_<n>_reproduced" in spec
    assert "complete_e3_tr87_ft09_partial" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_087_active_summary_counts_target_actions() -> None:
    """REQ-PHASE4-087: active-data summaries expose target-action coverage."""

    transitions = [
        (np.array([[0, 1], [0, 0]]), (1,), np.array([[0, 2], [0, 0]])),
        (np.array([[0, 2], [0, 0]]), (2,), np.array([[0, 3], [0, 0]])),
        (np.array([[0, 3], [0, 0]]), (6, 1, 0), np.array([[0, 4], [0, 0]])),
    ]

    summary = exp.summarize_active_transitions(
        game="tr87",
        target_actions=(1, 2, 3, 4),
        transitions=transitions,
        dataset_path=Path("results/arc_e3/tr87/active_data_4363.json"),
        dataset_sha256="b" * 64,
    )

    assert summary["active_transitions_collected"] == 3
    assert summary["target_action_counts"] == {"1": 1, "2": 1, "3": 0, "4": 0}
    assert summary["action_counts"]["6"] == 1
    assert summary["diverse_object_config_signatures"] >= 1

    record = exp._transition_record(0, (transitions[0][0], {"action": 6, "data": {"x": 1, "y": 2}}, transitions[0][2]))
    assert record["action"] == 6
    assert record["data"] == {"x": 1, "y": 2}
    assert len(record["grid_sha256"]) == 64


def test_req_phase4_087_build_artifact_counts_reproduction_gated_l1(tmp_path: Path) -> None:
    """REQ-PHASE4-087: only offline reproduction-gated L1 rows count."""

    rows = [
        _row(tmp_path, "tr87", accuracy=1.0, reproduced=True, reached=1, active_n=4),
        _row(tmp_path, "ft09", accuracy=0.4, reproduced=False, reached=0, active_n=5),
    ]

    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4363,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["per_game_scorecard"][0]["active_transitions_collected"] == 4
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_087_partial_artifact_preserves_active_rows(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-087: honest partials still preserve active-data evidence."""

    rows = [_row(tmp_path, "tr87", accuracy=0.0), _row(tmp_path, "ft09", accuracy=0.05)]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4363,
        duration_s=2.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_tr87_ft09_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_game_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["per_game_scorecard"][1]["residual_mismatch_class"] == exp.RESIDUAL_GAP_CLASSES["ft09"]
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_087_checksum_binds_active_dataset_hashes(tmp_path: Path) -> None:
    """REQ-PHASE4-087: checksum binds active data, models, plans, results, and seed."""

    rows = [_row(tmp_path, "tr87"), _row(tmp_path, "ft09")]
    paths = list(exp.WORLD_MODEL_PATHS.values())
    path_hashes = exp.path_hashes(tmp_path, paths)
    active_hashes = {row["game"]: row["active_dataset_sha256"] for row in rows}

    base = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4363,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4363,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=path_hashes,
        active_dataset_hashes={**active_hashes, "ft09": "c" * 64},
        random_seed=4363,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_scenario_phase4_087_missing_env_row_continues_other_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-087: a missing env blocks one game without stopping the other."""

    env = tmp_path / "environment_files" / "ft09"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    _write_model(tmp_path, "ft09")
    calls: list[str] = []

    def fake_runner(repo: Path, game: str, random_seed: int, active_budget: int, round_budget: int) -> dict:
        calls.append(f"{game}:{random_seed}:{active_budget}:{round_budget}")
        return _row(repo, game, accuracy=1.0, reproduced=True, reached=1)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ft09", fake_runner)

    artifact = exp.run_experiment(random_seed=4363, active_budget=9, round_budget=1)

    assert calls == ["ft09:4363:9:1"]
    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_game_scorecard"]] == [
        "blocked_offline_env_missing_tr87",
        "success_e3_ft09_L1_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_087_schema_errors_are_specific() -> None:
    """REQ-PHASE4-087: malformed bare fields and scorecard rows fail validation."""

    artifact = {
        "honest_verdict": "complete_e3_tr87_ft09_partial",
        "per_game_scorecard": ["bad-row", {"game": "ft09", "offline_reproduced": "false"}],
        "world_model_paths": [123],
        "new_levels_reproduced": {"value": 0},
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4363",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_game_scorecard[0] must be dict" in errors
    assert "per_game_scorecard[1] missing verifier_accuracy" in errors
    assert "per_game_scorecard[1] missing active_transitions_collected" in errors
    assert "per_game_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None, "per_game_scorecard": "bad"})
    assert "missing honest_verdict" in missing
    assert "per_game_scorecard must be list" in missing
    assert "field_principles missing" in missing


def test_req_phase4_087_plan_label_helpers() -> None:
    """REQ-PHASE4-087: replay labels are deterministic for tr87 and ft09 plans."""

    ft09 = exp._ft09_labels_from_action_plan(
        [
            {"action": 6, "x": 36, "y": 36},
            {"action": 6, "x": 52, "y": 44},
        ]
    )
    tr87 = exp._first_reproducing_prefix(["a", "b", "c"], lambda labels: len(labels) >= 2)

    assert ft09 == [
        '{"action": 6, "data": {"x": 36, "y": 36}}',
        '{"action": 6, "data": {"x": 52, "y": 44}}',
    ]
    assert tr87 == ["a", "b"]
    assert exp._first_reproducing_prefix(["a"], lambda _labels: False) == []


def test_req_phase4_087_pure_helper_edge_branches(tmp_path: Path) -> None:
    """REQ-PHASE4-087: pure helper edge branches remain deterministic."""

    outside = tmp_path.parent / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    blocked = [exp.blocked_game_row(tmp_path, "tr87"), exp.blocked_game_row(tmp_path, "ft09")]

    assert exp._relative_or_absolute(tmp_path, outside) == str(outside)
    assert exp._action_int(7) == 7
    assert exp._action_data((6, 3, 4)) == {"x": 3, "y": 4}
    assert exp._action_data({"action": 6}) is None
    assert exp._action_data((1,)) is None
    assert exp._state_signature(np.array([1, 2, 3])) == ()
    assert exp._residual_mismatch_class("ft09", []) == "none"
    assert exp._residual_mismatch_class("ft09", [{"error": "boom"}]) == "engine_runtime_error_gap"
    assert exp._residual_mismatch_class("ft09", [{"action": 6}]) == exp.RESIDUAL_GAP_CLASSES["ft09"]
    assert exp._residual_mismatch_class("ft09", [{"action": 4}]) == "missing_world_model_rule_gap_actions_4"
    assert exp._combined_verdict(blocked) == "blocked_offline_env_missing_tr87_ft09"

    totals = "reproducible_total_levels: 32   # old\n"
    assert exp._registry_total(totals, "reproducible_total_levels") == 32
    assert exp._registry_total(totals, "missing") is None
    assert "reproducible_total_levels: 33" in exp._replace_total(
        totals,
        "reproducible_total_levels",
        33,
        "new",
    )
    assert "reproducible_total_games: 17" in exp._replace_total(totals, "reproducible_total_games", 17, "new")
