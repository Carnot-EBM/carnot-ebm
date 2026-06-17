"""Tests for Exp 4352 tr87/ft09 E3 explore-verify-plan continuation.

Spec refs: REQ-PHASE4-084, SCENARIO-PHASE4-084.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import carnot.experiment_4352_e3_explore_verify_plan_tr87_ft09 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _write_model(repo: Path, game: str) -> Path:
    model = repo / exp.WORLD_MODEL_PATHS[game]
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    return model


def _row(
    repo: Path,
    game: str,
    *,
    accuracy: float = 0.5,
    reproduced_levels: int = 0,
    reproduced: bool = False,
) -> dict:
    return exp.build_game_scorecard(
        repo=repo,
        game=game,
        verifier_accuracy_per_round=[accuracy],
        world_model_path=_write_model(repo, game),
        plan_result={"planned": reproduced, "executed": reproduced, "solution": ["1"] if reproduced else []},
        reproduce_result={
            "game": game,
            "reached_level": reproduced_levels,
            "claimed_level": 1,
            "reproduced": reproduced,
        },
        residual_mismatch_class="none" if reproduced else "missing_world_model_rule_gap_actions_6",
        explore_lemmas=[{"action": 6, "changed_cells": 3, "verifier_gated": True}],
        adaptive_test_results=[{"name": f"{game}_mechanic_probe", "passed": accuracy >= 0.95}],
    )


def test_req_phase4_084_spec_declares_exp4352_contract() -> None:
    """REQ-PHASE4-084: OpenSpec declares the two-game upgraded E3 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-084" in spec
    assert "SCENARIO-PHASE4-084" in spec
    assert "experiment_4352_e3_explore_verify_plan_tr87_ft09.json" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_tr87_ft09_<n>_reproduced" in spec
    assert "complete_e3_tr87_ft09_partial" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_084_build_artifact_success_counts_reproduced_levels(tmp_path: Path) -> None:
    """REQ-PHASE4-084: only offline reproduction-gated levels count as progress."""

    rows = [
        _row(tmp_path, "tr87", accuracy=0.98, reproduced_levels=1, reproduced=True),
        _row(tmp_path, "ft09", accuracy=0.4, reproduced_levels=0, reproduced=False),
    ]

    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        world_model_paths=[exp.WORLD_MODEL_PATHS["tr87"], exp.WORLD_MODEL_PATHS["ft09"]],
        random_seed=4352,
        duration_s=2.5,
    )

    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert [row["game"] for row in artifact["per_game_scorecard"]] == ["tr87", "ft09"]
    assert artifact["per_game_scorecard"][0]["offline_reproduced"] is True
    assert artifact["world_model_paths"] == [exp.WORLD_MODEL_PATHS["tr87"], exp.WORLD_MODEL_PATHS["ft09"]]
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_084_partial_artifact_is_complete_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-084: honest partials preserve both per-game rows."""

    rows = [_row(tmp_path, "tr87", accuracy=0.0), _row(tmp_path, "ft09", accuracy=0.1)]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=rows,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4352,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_tr87_ft09_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["offline_reproduced"] for row in artifact["per_game_scorecard"]] == [False, False]
    assert artifact["per_game_scorecard"][1]["verifier_accuracy"] == 0.1
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_084_checksum_binds_rows_paths_hashes_and_seed(tmp_path: Path) -> None:
    """REQ-PHASE4-084: checksum binds model hashes, plans, reproduce results, and seed."""

    rows = [_row(tmp_path, "tr87"), _row(tmp_path, "ft09")]
    paths = [exp.WORLD_MODEL_PATHS["tr87"], exp.WORLD_MODEL_PATHS["ft09"]]
    hashes = exp.path_hashes(tmp_path, paths)

    base = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=hashes,
        random_seed=4352,
    )
    same = exp.compute_reproducibility_checksum(
        per_game_scorecard=rows,
        world_model_paths=paths,
        path_hashes=hashes,
        random_seed=4352,
    )
    changed = exp.compute_reproducibility_checksum(
        per_game_scorecard=[{**rows[0], "verifier_accuracy": 0.75}, rows[1]],
        world_model_paths=paths,
        path_hashes=hashes,
        random_seed=4352,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_scenario_phase4_084_missing_env_row_continues_other_game(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-084: a missing env blocks one row without stopping the other."""

    env = tmp_path / "environment_files" / "ft09"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    _write_model(tmp_path, "ft09")
    calls: list[str] = []

    def fake_score_game(game: str, *, repo: Path, random_seed: int, n_transitions: int) -> dict:
        calls.append(f"{game}:{random_seed}:{n_transitions}")
        return _row(repo, game, accuracy=0.96, reproduced_levels=1, reproduced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "score_game", fake_score_game)

    artifact = exp.run_experiment(random_seed=4352, n_transitions=7)

    assert calls == ["ft09:4352:7"]
    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_1_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_game_scorecard"]] == [
        "blocked_offline_env_missing_tr87",
        "success_e3_ft09_L1_reproduced",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_084_schema_errors_are_specific() -> None:
    """REQ-PHASE4-084: malformed bare fields and scorecard rows fail validation."""

    artifact = {
        "honest_verdict": "complete_e3_tr87_ft09_partial",
        "per_game_scorecard": ["bad-row", {"game": "ft09", "offline_reproduced": "false"}],
        "world_model_paths": [123],
        "new_levels_reproduced": {"value": 0},
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4352,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_game_scorecard[0] must be dict" in errors
    assert "per_game_scorecard[1] missing verifier_accuracy" in errors
    assert "per_game_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None, "per_game_scorecard": "bad"})
    assert "missing honest_verdict" in missing
    assert "per_game_scorecard must be list" in missing
    assert "field_principles missing" in missing


def test_req_phase4_084_transition_lemmas_and_residual_classes() -> None:
    """REQ-PHASE4-084: verifier-gated lemmas and residual classes are deterministic."""

    class FakeTransition:
        action = 6
        data = {"x": 1, "y": 2}
        level_before = 0
        level_after = 1

        @property
        def grid(self):
            import numpy as np

            return np.array([[0, 1], [2, 3]])

        @property
        def next_grid(self):
            import numpy as np

            return np.array([[0, 1], [4, 3]])

    lemmas = exp.collect_explore_lemmas([FakeTransition()])

    assert lemmas == [
        {
            "action": 6,
            "has_data": True,
            "changed_cells": 1,
            "level_delta": 1,
            "verifier_gated": True,
        }
    ]
    assert exp.residual_mismatch_class([]) == "none"
    assert exp.residual_mismatch_class([{"error": "boom"}]) == "engine_runtime_error_gap"
    assert exp.residual_mismatch_class([{"your_prediction_was_wrong_at": "wrong shape"}]) == (
        "world_model_shape_rule_gap"
    )
    assert exp.residual_mismatch_class([{"action": 7}]) == (
        "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )
    assert exp.residual_mismatch_class([{"action": 4}, {"action": 2}]) == (
        "missing_world_model_rule_gap_actions_2_4"
    )
    assert exp.residual_mismatch_class([{"your_prediction_was_wrong_at": []}]) == (
        "model_predicted_identity_when_transition_changed_gap"
    )


def test_req_phase4_084_helper_branches_cover_paths_labels_and_blocked_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-084: pure helpers keep edge branches deterministic."""

    model = _write_model(tmp_path, "tr87")
    outside = tmp_path.parent / "outside_world_model.py"
    outside.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    assert exp._relative_or_absolute(tmp_path, outside) == str(outside)
    assert exp._labels_from_plan(None) == []
    assert exp._labels_from_plan({"solution": ["1", {"action": 2, "data": None}, 3]}) == [
        "1",
        '{"action": 2, "data": null}',
    ]
    assert exp._plan_executed(None) is False

    blocked_rows = [exp.build_missing_game_scorecard(tmp_path, "tr87"), exp.build_missing_game_scorecard(tmp_path, "ft09")]
    assert exp._combined_verdict(blocked_rows) == "blocked_offline_env_missing_tr87_ft09"

    scorecard = exp.build_game_scorecard(
        repo=tmp_path,
        game="tr87",
        verifier_accuracy_per_round=[],
        world_model_path=model,
        plan_result=None,
        reproduce_result={"game": "tr87", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="gap",
        explore_lemmas=[],
        adaptive_test_results=[],
    )
    assert scorecard["verifier_accuracy"] == 0.0
    assert scorecard["plan_executed"] is False

    monkeypatch.setattr(
        exp.e3,
        "plan_and_execute",
        lambda game, engine, is_level_complete: {"game": game, "planned": True, "executed": True},
    )
    assert exp._planned_result("tr87", object(), object(), False)["planned"] is False
    assert exp._planned_result("tr87", object(), object(), True)["executed"] is True


def test_req_phase4_084_collect_lemmas_limits_noops_and_breaks() -> None:
    """REQ-PHASE4-084: exploration lemma collection is bounded and salience-biased."""

    import numpy as np

    transitions = []
    for index in range(5):
        grid = np.array([[index]])
        next_grid = np.array([[index if index < 3 else index + 1]])
        transitions.append(
            SimpleNamespace(
                grid=grid,
                next_grid=next_grid,
                action=6,
                data=None,
                level_before=0,
                level_after=0,
            )
        )

    lemmas = exp.collect_explore_lemmas(transitions, limit=3)

    assert len(lemmas) == 3
    assert [row["changed_cells"] for row in lemmas] == [0, 0, 1]
    assert exp.adaptive_world_model_tests("tr87", 0.95)[0]["passed"] is True
    assert exp.adaptive_world_model_tests("tr87", 0.94)[0]["passed"] is False


def test_req_phase4_084_score_game_uses_verifier_plan_and_reproduction_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-084: score_game gates planning and reproduction through verifier evidence."""

    import numpy as np

    _write_model(tmp_path, "tr87")
    transition = SimpleNamespace(
        grid=np.array([[0]]),
        next_grid=np.array([[1]]),
        action=6,
        data={"x": 1, "y": 1},
        level_before=0,
        level_after=0,
    )
    accuracies = iter([0.5, 1.0, 1.0])
    plans = iter(
        [
            {"game": "tr87", "planned": True, "executed": True, "level_up": True, "solution": []},
            {
                "game": "tr87",
                "planned": True,
                "executed": True,
                "level_up": True,
                "solution": [{"action": 1, "data": None}],
            },
        ]
    )

    class FakeVerifier:
        def __init__(self, transitions):
            self.transitions = transitions

        def score(self, _engine):
            return SimpleNamespace(
                accuracy=next(accuracies),
                mismatches=[{"action": 6, "your_prediction_was_wrong_at": []}],
            )

    reproduce_calls: list[list[str]] = []

    def fake_reproduce(game, labels, apply, claimed_level):
        reproduce_calls.append(labels)
        return {"game": game, "reached_level": claimed_level, "claimed_level": claimed_level, "reproduced": True}

    monkeypatch.setattr(exp.e3, "collect_transitions", lambda game, n, seed: ([transition], 1))
    monkeypatch.setattr(exp.e3, "WorldModelVerifier", FakeVerifier)
    monkeypatch.setattr(exp.e3, "load_engine", lambda game: (lambda grid, action, data: grid, lambda grid: False))
    monkeypatch.setattr(exp.e3, "plan_and_execute", lambda game, engine, done: next(plans))
    monkeypatch.setattr(exp.arc_solver_kit, "reproduce", fake_reproduce)

    low_accuracy = exp.score_game("tr87", repo=tmp_path, random_seed=4352, n_transitions=1)
    level_up_without_labels = exp.score_game("tr87", repo=tmp_path, random_seed=4352, n_transitions=1)
    reproduced = exp.score_game("tr87", repo=tmp_path, random_seed=4352, n_transitions=1)

    assert low_accuracy["plan_result"]["planned"] is False
    assert level_up_without_labels["reproduce_result"]["mode"] == "level_up_without_replayable_solution_labels"
    assert reproduced["offline_reproduced"] is True
    assert reproduce_calls == [['{"action": 1, "data": null}']]


def test_scenario_phase4_084_run_experiment_records_partial_gaps_and_schema_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-084: partial rows write gaps and malformed artifacts fail closed."""

    for game in exp.GAMES:
        env = tmp_path / "environment_files" / game
        env.mkdir(parents=True)
        (env / "fixture").write_text("present", encoding="utf-8")
        _write_model(tmp_path, game)

    def partial_score(game: str, *, repo: Path, random_seed: int, n_transitions: int) -> dict:
        return _row(repo, game, accuracy=0.25, reproduced_levels=0, reproduced=False)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "score_game", partial_score)

    artifact = exp.run_experiment(random_seed=4352, n_transitions=3)

    assert artifact["honest_verdict"] == "complete_e3_tr87_ft09_partial"
    gap_text = (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "Exp4352 tr87 E3 residual gap" in gap_text
    assert "Exp4352 ft09 E3 residual gap" in gap_text

    exp._write_gap(tmp_path / exp.GAP_RELATIVE_PATH, row=artifact["per_game_scorecard"][0], checksum="a" * 64)
    rewritten = (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert rewritten.count("Exp4352 tr87 E3 residual gap") == 1

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="Exp4352 artifact schema errors"):
        exp.run_experiment(random_seed=4352, n_transitions=3)
