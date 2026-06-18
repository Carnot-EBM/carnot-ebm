"""Tests for Exp 4384 ar25/ka59/ft09 E3 lookahead hidden-rule continuation.

Spec refs: REQ-PHASE4-4384, SCENARIO-PHASE4-4384.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09 as exp


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
    fidelity: float = 0.5,
    advanced: bool = False,
) -> dict:
    return exp.build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=[accuracy],
        lookahead_fidelity_per_round=[fidelity],
        active_dataset_summary=_active_summary(game),
        world_model_path=_write_model(repo, game),
        skill_file_path=f"results/arc_e3/{game}/skill_4384.json",
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
        mechanic_checks_passed=advanced,
    )


def test_req_phase4_4384_spec_declares_contract() -> None:
    """REQ-PHASE4-4384: OpenSpec declares active-data plus K-step lookahead gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4384" in spec
    assert "SCENARIO-PHASE4-4384" in spec
    assert "experiment_4384_e3_blocked_mechanic_tails_ar25_ka59_ft09.json" in spec
    assert "Mind-Studio K-step LOOKAHEAD-FIDELITY" in spec
    assert "skill_4384.json" in spec
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


def test_req_phase4_4384_skill_file_records_named_register_lookahead(tmp_path: Path) -> None:
    """REQ-PHASE4-4384: skill files preserve Mind-Studio preplanning inputs."""

    path = exp.write_skill_file(tmp_path, "ka59", random_seed=4384, lookahead_k=3)
    data = json.loads((tmp_path / path).read_text(encoding="utf-8"))

    assert path == "results/arc_e3/ka59/skill_4384.json"
    assert data["game"] == "ka59"
    assert data["spec_refs"] == ["REQ-PHASE4-4384", "SCENARIO-PHASE4-4384"]
    assert data["mind_studio_source"] == "arXiv:2606.16070"
    assert data["lookahead_k"] == 3
    assert data["random_seed"] == 4384
    assert data["prior_best_level"] == 1
    assert data["target_level"] == 2
    assert data["target_named_gap"] == exp.RESIDUAL_GAP_CLASSES["ka59"]
    assert data["lookahead_fidelity_target"] == "named_register_k_step_rollout_matches_env_before_planning"
    assert data["verifier_is_oracle"] is True


def test_req_phase4_4384_k_step_fidelity_scores_named_register_rollouts() -> None:
    """REQ-PHASE4-4384: K-step fidelity compares induced rollout to the named register."""

    g0 = np.zeros((3, 3), dtype=int)
    g1 = g0.copy()
    g1[-1, :] = 1
    g2 = g1.copy()
    g2[-1, :] = 2
    transitions = [
        exp.CompactTransition(g0, 1, None, g1, 0, 0),
        exp.CompactTransition(g1, 2, None, g2, 0, 0),
    ]

    def good_engine(grid: np.ndarray, _action: int, _data: dict | None) -> np.ndarray:
        out = grid.copy()
        out[-1, :] += 1
        return out

    def bad_engine(grid: np.ndarray, _action: int, _data: dict | None) -> np.ndarray:
        return grid.copy()

    assert exp.compute_k_step_lookahead_fidelity(
        game="ka59",
        transitions=transitions,
        engine=good_engine,
        k=2,
    ) == 1.0
    assert exp.compute_k_step_lookahead_fidelity(
        game="ka59",
        transitions=transitions,
        engine=bad_engine,
        k=2,
    ) == 0.0
    assert exp.compute_k_step_lookahead_fidelity(
        game="ka59",
        transitions=[],
        engine=good_engine,
        k=2,
    ) == 0.0


def test_req_phase4_4384_build_artifact_counts_only_new_l2_reproductions(tmp_path: Path) -> None:
    """REQ-PHASE4-4384: only levels beyond prior L1 count as progress."""

    rows = [
        _row(tmp_path, "ar25", reached=2, accuracy=1.0, fidelity=1.0, advanced=True),
        _row(tmp_path, "ka59", accuracy=0.75, fidelity=0.25),
        _row(tmp_path, "ft09", accuracy=0.4, fidelity=0.5),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=35,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        skill_file_paths=[row["mind_studio_skill_file"] for row in rows],
        random_seed=4384,
        game_wall_time_s=1.5,
        lookahead_k=3,
        duration_s=1.25,
    )

    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert artifact["verifier_is_oracle"] is True
    assert artifact["game_wall_time_s"] == 1.5
    assert artifact["lookahead_k"] == 3
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert artifact["per_game_scorecard"][0]["lookahead_fidelity"] == 1.0
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_4384_partial_artifact_preserves_named_gap_rows(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-4384: all-partial runs keep three lookahead-gated rows."""

    rows = [_row(tmp_path, game, accuracy=0.6, fidelity=0.2) for game in exp.TARGET_ORDER]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        reproducible_total_levels=34,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        skill_file_paths=[row["mind_studio_skill_file"] for row in rows],
        random_seed=4384,
        game_wall_time_s=None,
        lookahead_k=3,
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
    assert all(isinstance(row["lookahead_fidelity"], float) for row in artifact["per_game_scorecard"])
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4384_checksum_binds_active_data_models_skills_plans_results_and_seed(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-4384: checksum binds datasets, models, skills, plans, results, and seed."""

    rows = [_row(tmp_path, game) for game in exp.TARGET_ORDER]
    paths = list(exp.WORLD_MODEL_PATHS.values())
    skill_paths = [row["mind_studio_skill_file"] for row in rows]
    path_hashes = exp.path_hashes(tmp_path, paths + skill_paths)
    active_hashes = {row["game"]: row["active_dataset_sha256"] for row in rows}

    base = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        skill_file_paths=skill_paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4384,
        game_wall_time_s=1.5,
        lookahead_k=3,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        skill_file_paths=skill_paths,
        path_hashes=path_hashes,
        active_dataset_hashes=active_hashes,
        random_seed=4384,
        game_wall_time_s=1.5,
        lookahead_k=3,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        skill_file_paths=skill_paths,
        path_hashes=path_hashes,
        active_dataset_hashes={**active_hashes, "ft09": "b" * 64},
        random_seed=4384,
        game_wall_time_s=1.5,
        lookahead_k=3,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_4384_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4384: malformed bare fields and lookahead rows fail validation."""

    artifact = {
        "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
        "per_game_scorecard": ["bad-row", {"game": "ka59", "offline_reproduced": "false"}],
        "new_levels_reproduced": "0",
        "reproducible_total_levels": {"value": 34},
        "world_model_paths": [123],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": "4384",
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
        "game_wall_time_s": "1.0",
        "lookahead_k": "3",
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_game_scorecard[0] must be dict" in errors
    assert "per_game_scorecard[1] missing prior_best_level" in errors
    assert "per_game_scorecard[1] missing lookahead_fidelity" in errors
    assert "per_game_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "per_game_scorecard[1].lookahead_fidelity must be bare number" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "random_seed must be bare int" in errors
    assert "game_wall_time_s must be numeric" in errors
    assert "lookahead_k must be int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None, "per_game_scorecard": "bad"})
    assert "missing honest_verdict" in missing
    assert "per_game_scorecard must be list" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_4384_missing_env_continues_other_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4384: missing envs block per game without fabrication."""

    for game in ("ka59", "ft09"):
        env = tmp_path / "environment_files" / game
        env.mkdir(parents=True)
        (env / "fixture").write_text("present", encoding="utf-8")
        _write_model(tmp_path, game)

    calls: list[str] = []

    def fake_runner(
        repo: Path,
        game: str,
        random_seed: int,
        active_budget: int,
        round_budget: int,
        lookahead_k: int,
        skill_file_path: str,
    ) -> dict:
        calls.append(f"{game}:{random_seed}:{active_budget}:{round_budget}:{lookahead_k}:{skill_file_path}")
        return _row(repo, game, reached=2, accuracy=1.0, fidelity=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ka59", fake_runner)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ft09", fake_runner)

    artifact = exp.run_experiment(
        random_seed=4384,
        active_budget=9,
        round_budget=1,
        game_wall_time_s=None,
        lookahead_k=3,
    )

    assert calls == [
        "ka59:4384:9:1:3:results/arc_e3/ka59/skill_4384.json",
        "ft09:4384:9:1:3:results/arc_e3/ft09/skill_4384.json",
    ]
    assert artifact["honest_verdict"] == "success_e3_ar25_ka59_ft09_2_reproduced"
    assert artifact["new_levels_reproduced"] == 2
    assert [row["checkpoint_status"] for row in artifact["per_game_scorecard"]] == [
        "blocked_offline_env_missing_ar25",
        "new_level_reproduced",
        "new_level_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / "results" / "arc_e3" / "ka59" / "skill_4384.json").exists()


def test_req_phase4_4384_timeout_and_path_hash_branches(tmp_path: Path) -> None:
    """REQ-PHASE4-4384: timeout rows and path hashing preserve honest evidence."""

    timeout = exp.timeout_game_row("ar25", target_wall_time_s=0.01)
    present = tmp_path / "solver.py"
    present.write_text("# solver\n", encoding="utf-8")
    hashes = exp.path_hashes(tmp_path, ["solver.py", "missing.py"])

    assert timeout["game"] == "ar25"
    assert timeout["new_reproduced_level"] == exp.PRIOR_BEST_LEVELS["ar25"]
    assert timeout["offline_reproduced"] is False
    assert timeout["checkpoint_status"] == "honest_partial_wall_time_cap_exhausted"
    assert timeout["residual_gap_class"] == "wall_time_cap_exhausted"
    assert hashes["solver.py"] == hashlib.sha256(b"# solver\n").hexdigest()
    assert hashes["missing.py"] == ""


def test_req_phase4_4384_active_dataset_and_helper_edge_branches(tmp_path: Path) -> None:
    """REQ-PHASE4-4384: pure helper edge branches stay deterministic."""

    outside = tmp_path.parent / "outside_4384.py"
    outside.write_text("# outside\n", encoding="utf-8")
    grid0 = np.zeros((2, 2), dtype=int)
    grid1 = np.ones((2, 2), dtype=int)
    grid2 = np.full((2, 2), 2, dtype=int)
    transitions = [
        exp.CompactTransition(grid0, 7, None, grid1, 0, 0),
        exp.CompactTransition(grid2, 7, None, grid2, 0, 0),
    ]

    path, dataset_sha = exp.write_targeted_active_dataset(
        tmp_path,
        "ar25",
        transitions,
        random_seed=4384,
        collection_method="unit",
        collection_error="",
    )

    assert exp._relative_or_absolute(tmp_path, outside) == str(outside)
    assert (tmp_path / exp.ACTIVE_DATASET_PATHS["ar25"]).exists()
    assert path == tmp_path / exp.ACTIVE_DATASET_PATHS["ar25"]
    assert len(dataset_sha) == 64
    assert exp._named_register("ar25", grid1).tolist() == grid1.tolist()

    def identity_engine(grid: np.ndarray, _action: int, _data: dict | None) -> np.ndarray:
        return grid.copy()

    def raising_engine(_grid: np.ndarray, _action: int, _data: dict | None) -> np.ndarray:
        raise RuntimeError("boom")

    assert exp.compute_k_step_lookahead_fidelity(
        game="ar25",
        transitions=transitions,
        engine=identity_engine,
        k=2,
    ) == 0.5
    assert exp.compute_k_step_lookahead_fidelity(
        game="ar25",
        transitions=transitions,
        engine=raising_engine,
        k=2,
    ) == 0.0


def test_req_phase4_4384_import_conductor_registry_gap_and_exception_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4384: import, git, registry, gap, and exception branches fail closed."""

    def fake_import_module(name: str):
        if name.endswith("experiment_4373_e3_blocked_mechanic_levels_ar25_ka59_ft09"):
            raise RuntimeError("missing")
        return object()

    monkeypatch.setattr(exp.importlib, "import_module", fake_import_module)
    assert exp._imports_ok() == {
        "harness_import": True,
        "solver_kit_import": True,
        "active_data_module_import": False,
    }

    git_repo = tmp_path / "repo"
    (git_repo / ".git").mkdir(parents=True)

    class Proc:
        stdout = " M scripts/research_conductor.py\n"

    monkeypatch.setattr(exp.subprocess, "run", lambda *_args, **_kwargs: Proc())
    assert exp._research_conductor_modified(git_repo) is True

    def boom(*_args, **_kwargs):
        raise TimeoutError("git slow")

    monkeypatch.setattr(exp.subprocess, "run", boom)
    assert exp._research_conductor_modified(git_repo) is False
    assert exp._research_conductor_modified(tmp_path / "not-git") is False

    blocked = [exp.blocked_game_row(tmp_path, game) for game in exp.TARGET_ORDER]
    assert exp._combined_verdict(blocked) == "blocked_offline_env_missing_ar25_ka59_ft09"
    assert exp._registry_total("reproducible_total_levels: 34\n", "reproducible_total_levels") == 34
    assert exp._registry_total("no total\n", "reproducible_total_levels") is None
    assert exp._registry_total_from_repo(tmp_path) is None
    registry = tmp_path / exp.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("reproducible_total_levels: 34\n", encoding="utf-8")
    assert exp._registry_total_from_repo(tmp_path) == 34

    exception = exp.exception_game_row("ft09", "Traceback\nValueError: bad")
    assert exception["checkpoint_status"] == "honest_partial_target_exception"
    assert exception["reproduce_result"]["exception"] == "ValueError: bad"

    gap_path = tmp_path / exp.GAP_RELATIVE_PATH
    row = _row(tmp_path, "ar25", accuracy=0.8, fidelity=0.25)
    exp._write_gap(gap_path, row=row, checksum="a" * 64)
    exp._write_gap(gap_path, row={**row, "verifier_accuracy": 0.9}, checksum="b" * 64)
    gap_text = gap_path.read_text(encoding="utf-8")
    assert "Exp4384 ar25 Mind-Studio lookahead residual gap" in gap_text
    assert "Best verifier accuracy: 0.9000" in gap_text
    assert "`" + "b" * 64 + "`" in gap_text
    assert "`" + "a" * 64 + "`" not in gap_text


def test_scenario_phase4_4384_gap_writer_and_schema_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4384: partial rows write named gaps and schema failures abort."""

    env = tmp_path / "environment_files" / "ar25"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    _write_model(tmp_path, "ar25")

    def fake_runner(
        repo: Path,
        game: str,
        _random_seed: int,
        _active_budget: int,
        _round_budget: int,
        _lookahead_k: int,
        _skill_file_path: str,
    ) -> dict:
        return _row(repo, game, accuracy=0.9, fidelity=0.2, advanced=False)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "ar25", fake_runner)

    artifact = exp.run_experiment(
        random_seed=4384,
        active_budget=9,
        round_budget=1,
        game_wall_time_s=None,
        lookahead_k=3,
    )

    assert artifact["honest_verdict"] == "complete_e3_ar25_ka59_ft09_partial"
    gap_text = (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert exp.RESIDUAL_GAP_CLASSES["ar25"] in gap_text

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="Exp4384 artifact schema errors"):
        exp.run_experiment(
            random_seed=4384,
            active_budget=9,
            round_budget=1,
            game_wall_time_s=None,
            lookahead_k=3,
        )
