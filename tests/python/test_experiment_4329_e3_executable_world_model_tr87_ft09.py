"""Tests for Exp 4329 tr87/ft09 E3 executable-world-model attempt.

Spec refs: REQ-PHASE4-076, SCENARIO-PHASE4-076.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4329_e3_executable_world_model_tr87_ft09 as exp
from carnot.agentic.arc_executable_world_model import Transition, VerifyResult


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _transition() -> Transition:
    g0 = np.array([[0, 1], [2, 3]], dtype=int)
    g1 = np.array([[0, 1], [2, 4]], dtype=int)
    return Transition(g0, 5, None, g1, 0, 0)


def _write_model(repo: Path, game: str) -> Path:
    model_path = repo / exp.world_model_relative_path(game)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    return model_path


def _present_env(repo: Path, game: str) -> None:
    env = repo / "environment_files" / game
    env.mkdir(parents=True, exist_ok=True)
    (env / "fixture").write_text("present", encoding="utf-8")


def test_req_phase4_076_spec_declares_exp4329_contract() -> None:
    """REQ-PHASE4-076: OpenSpec declares the two-game E3 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-076" in spec
    assert "SCENARIO-PHASE4-076" in spec
    assert "experiment_4329_e3_executable_world_model_tr87_ft09.json" in spec
    assert "blocked_offline_env_missing_tr87_ft09" in spec
    assert "success_e3_tr87_ft09_<n>_L1_reproduced" in spec
    assert "complete_e3_tr87_ft09_partial" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_scenario_phase4_076_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-076: both missing offline envs block without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4329)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_tr87_ft09"
    assert artifact["reproduced_levels_total"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["offline_env_present"]["tr87"] is False
    assert artifact["preconditions_checked"]["offline_env_present"]["ft09"] is False
    assert set(artifact["per_game_scorecard"]) == {"tr87", "ft09"}
    assert artifact["per_game_scorecard"]["tr87"]["status"] == "blocked_offline_env_missing_tr87"
    assert artifact["per_game_scorecard"]["ft09"]["status"] == "blocked_offline_env_missing_ft09"
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_076_build_game_scorecard_preserves_breadth_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-076: each game row preserves the independent E3 gate evidence."""

    model_path = _write_model(tmp_path, "tr87")
    scorecard = exp.build_game_scorecard(
        repo=tmp_path,
        game="tr87",
        status="complete_e3_tr87_partial_model_0.50",
        verifier_accuracy_per_round=[0.25, 0.5],
        world_model_path=model_path,
        plan_result={"planned": True, "executed": False, "divergence_step": {"action": 1}},
        reproduce_result={"game": "tr87", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="translation_rule_gap",
    )

    assert scorecard["offline_reproduced"] is False
    assert scorecard["reproduced_levels"] == 0
    assert scorecard["best_verifier_accuracy"] == 0.5
    assert scorecard["world_model_path"] == exp.world_model_relative_path("tr87")
    assert scorecard["world_model_sha256"] == exp.sha256_file(model_path)
    assert scorecard["plan_executed"] is False
    assert scorecard["plan_executed_detail"]["divergence_step"] == {"action": 1}
    assert scorecard["residual_mismatch_class"] == "translation_rule_gap"


def test_scenario_phase4_076_success_artifact_counts_only_reproduced_levels(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-076: total progress is the sum of reproduced per-game levels."""

    tr87_model = _write_model(tmp_path, "tr87")
    ft09_model = _write_model(tmp_path, "ft09")
    tr87 = exp.build_game_scorecard(
        repo=tmp_path,
        game="tr87",
        status="success_e3_tr87_L1_reproduced",
        verifier_accuracy_per_round=[1.0],
        world_model_path=tr87_model,
        plan_result={"planned": True, "executed": True, "level_up": True},
        reproduce_result={"game": "tr87", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
    )
    ft09 = exp.build_game_scorecard(
        repo=tmp_path,
        game="ft09",
        status="complete_e3_ft09_partial_model_0.20",
        verifier_accuracy_per_round=[0.2],
        world_model_path=ft09_model,
        plan_result={"planned": False},
        reproduce_result={"game": "ft09", "reached_level": 0, "claimed_level": 1, "reproduced": False},
        residual_mismatch_class="missing_world_model_rule_gap_actions_5",
    )

    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard={"tr87": tr87, "ft09": ft09},
        random_seed=4329,
        duration_s=3.0,
    )

    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_1_L1_reproduced"
    assert artifact["reproduced_levels_total"] == 1
    assert artifact["per_game_scorecard"]["tr87"]["offline_reproduced"] is True
    assert artifact["per_game_scorecard"]["ft09"]["offline_reproduced"] is False
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_076_game_scorecard_uses_absolute_path_outside_repo(tmp_path: Path) -> None:
    """REQ-PHASE4-076: world-model hashes remain auditable for out-of-tree files."""

    model_path = _write_model(tmp_path, "ft09")
    scorecard = exp.build_game_scorecard(
        repo=tmp_path / "other-root",
        game="ft09",
        status="success_e3_ft09_L1_reproduced",
        verifier_accuracy_per_round=[1.0],
        world_model_path=model_path,
        plan_result={"planned": True, "executed": True},
        reproduce_result={"game": "ft09", "reached_level": 1, "claimed_level": 1, "reproduced": True},
        residual_mismatch_class="none",
    )

    assert scorecard["world_model_path"] == str(model_path)


def test_scenario_phase4_076_partial_artifact_records_both_game_partials(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-076: no reproduced L1 is still complete with per-game partials."""

    scorecards = {}
    for game, accuracy in (("tr87", 0.4), ("ft09", 0.6)):
        scorecards[game] = exp.build_game_scorecard(
            repo=tmp_path,
            game=game,
            status=f"complete_e3_{game}_partial_model_{accuracy:.2f}",
            verifier_accuracy_per_round=[accuracy],
            world_model_path=_write_model(tmp_path, game),
            plan_result=None,
            reproduce_result={"game": game, "reached_level": 0, "claimed_level": 1, "reproduced": False},
            residual_mismatch_class="no_goal_predicate_gap",
        )

    artifact = exp.build_artifact(
        repo=tmp_path,
        per_game_scorecard=scorecards,
        random_seed=4329,
        duration_s=0.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_tr87_ft09_partial"
    assert artifact["reproduced_levels_total"] == 0
    assert artifact["per_game_scorecard"]["tr87"]["best_verifier_accuracy"] == 0.4
    assert artifact["per_game_scorecard"]["ft09"]["best_verifier_accuracy"] == 0.6


def test_req_phase4_076_checksum_is_stable_and_sensitive() -> None:
    """REQ-PHASE4-076: checksum binds both game rows and the random seed."""

    scorecard = {
        "tr87": {"world_model_sha256": "a" * 64, "plan_result": {"actions": [1]}, "reproduce_result": {}},
        "ft09": {"world_model_sha256": "b" * 64, "plan_result": {"actions": [2]}, "reproduce_result": {}},
    }
    base = exp.compute_reproducibility_checksum(per_game_scorecard=scorecard, random_seed=4329)
    same = exp.compute_reproducibility_checksum(per_game_scorecard=scorecard, random_seed=4329)
    changed = exp.compute_reproducibility_checksum(per_game_scorecard=scorecard, random_seed=4330)

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_076_residual_mismatch_classes_cover_known_gaps() -> None:
    """REQ-PHASE4-076: residual mismatch class records the partial-model gap."""

    assert exp.residual_mismatch_class([]) == "none"
    assert exp.residual_mismatch_class([{"error": "boom"}]) == "engine_runtime_error_gap"
    assert (
        exp.residual_mismatch_class([{"your_prediction_was_wrong_at": []}])
        == "model_predicted_identity_when_transition_changed_gap"
    )
    assert (
        exp.residual_mismatch_class([{"your_prediction_was_wrong_at": "wrong shape (1, 1)"}])
        == "world_model_shape_rule_gap"
    )
    assert (
        exp.residual_mismatch_class([{"action": 4}, {"action": 2}])
        == "missing_world_model_rule_gap_actions_2_4"
    )
    assert (
        exp.residual_mismatch_class([{"action": 7}, {"action": 1}])
        == "missing_world_model_rule_gap_hidden_undo_stack_action7"
    )


def test_req_phase4_076_schema_errors_are_specific() -> None:
    """REQ-PHASE4-076: artifact schema validation catches non-bare or missing fields."""

    bad = {
        "honest_verdict": "complete_e3_tr87_ft09_partial",
        "per_game_scorecard": [],
        "reproduced_levels_total": "0",
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4329,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_game_scorecard must be dict" in errors
    assert "reproduced_levels_total must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors
    assert "missing tr87 scorecard" in errors
    assert "missing ft09 scorecard" in errors

    row_bad = {
        "honest_verdict": "complete_e3_tr87_ft09_partial",
        "per_game_scorecard": {
            "tr87": {"offline_reproduced": "false", "plan_executed": None, "reproduced_levels": "0"},
            "ft09": {"offline_reproduced": False, "plan_executed": False, "reproduced_levels": 0},
        },
        "reproduced_levels_total": 0,
        "verifier_is_oracle": True,
        "preconditions_checked": {},
        "random_seed": 4329,
        "reproducibility_checksum": "a" * 64,
        "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
    }
    row_errors = exp.artifact_schema_errors(row_bad)

    assert "tr87.offline_reproduced must be bare bool" in row_errors
    assert "tr87.plan_executed must be bare bool" in row_errors
    assert "tr87.reproduced_levels must be bare int" in row_errors
    assert "tr87.world_model_sha256 missing" in row_errors
    assert "tr87.best_verifier_accuracy missing" in row_errors
    assert "ft09.world_model_sha256 missing" in row_errors
    assert "ft09.best_verifier_accuracy missing" in row_errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_076_write_gap_replaces_game_entry(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-076: partial runs keep one residual-gap entry per game."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(gap_path, game="tr87", best_accuracy=0.1, mismatch_class="first", checksum="a" * 64)
    exp._write_gap(gap_path, game="tr87", best_accuracy=0.2, mismatch_class="second", checksum="b" * 64)

    text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.2000" in text
    assert "second" in text
    assert "first" not in text


def test_scenario_phase4_076_apply_noop_returns_frame() -> None:
    """SCENARIO-PHASE4-076: reproduction fallback apply is explicit and deterministic."""

    frame = object()

    assert exp._apply_noop(None, "noop", frame) is frame


def test_scenario_phase4_076_run_experiment_blocks_when_both_envs_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-076: runner writes blocked artifact when both envs are absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4329, n_transitions=1)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_tr87_ft09"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_076_run_experiment_proceeds_when_one_env_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-076: one missing env does not suppress the present game."""

    _present_env(tmp_path, "tr87")
    _write_model(tmp_path, "tr87")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1,
            n_correct=0,
            accuracy=0.0,
            mismatches=[{"action": 5, "true_change": [(1, 1, 3, 4)]}],
        ),
    )
    monkeypatch.setattr(exp.e3, "plan_and_execute", lambda *_args, **_kwargs: {"planned": False})
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {"game": "tr87", "reached_level": 0, "claimed_level": 1, "reproduced": False},
    )

    artifact = exp.run_experiment(random_seed=4329, n_transitions=1)

    assert artifact["honest_verdict"] == "complete_e3_tr87_ft09_partial"
    assert artifact["per_game_scorecard"]["tr87"]["status"] == "complete_e3_tr87_partial_model_0.00"
    assert artifact["per_game_scorecard"]["ft09"]["status"] == "blocked_offline_env_missing_ft09"
    assert "REQ-PHASE4-076" in (tmp_path / exp.GAP_RELATIVE_PATH).read_text(encoding="utf-8")
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_076_run_experiment_writes_success_with_stubs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-076: runner accepts success only after reproduction returns L1."""

    for game in exp.GAMES:
        _present_env(tmp_path, game)
        _write_model(tmp_path, game)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: True))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=1, accuracy=1.0, mismatches=[]),
    )
    monkeypatch.setattr(
        exp.e3,
        "plan_and_execute",
        lambda game, *_args, **_kwargs: {
            "game": game,
            "planned": True,
            "executed": True,
            "level_up": True,
            "solution": ["noop"],
        },
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda game, *_args, **_kwargs: {"game": game, "reached_level": 1, "claimed_level": 1, "reproduced": True},
    )

    artifact = exp.run_experiment(random_seed=4329, n_transitions=1)

    assert artifact["honest_verdict"] == "success_e3_tr87_ft09_2_L1_reproduced"
    assert artifact["reproduced_levels_total"] == 2
    assert artifact["per_game_scorecard"]["tr87"]["offline_reproduced"] is True
    assert artifact["per_game_scorecard"]["ft09"]["offline_reproduced"] is True
    assert not (tmp_path / exp.GAP_RELATIVE_PATH).exists()


def test_scenario_phase4_076_run_experiment_raises_on_schema_regression(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SCENARIO-PHASE4-076: runner fails loudly if the artifact contract regresses."""

    for game in exp.GAMES:
        _present_env(tmp_path, game)
        _write_model(tmp_path, game)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1))
    monkeypatch.setattr(exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False))
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(n=1, n_correct=0, accuracy=0.0, mismatches=[]),
    )
    monkeypatch.setattr(exp.e3, "plan_and_execute", lambda *_args, **_kwargs: {"planned": False})
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4329 artifact schema errors"):
        exp.run_experiment(random_seed=4329, n_transitions=1)
