"""Tests for Exp 4350 ka59 E3 explore-verify-plan continuation.

Spec refs: REQ-PHASE4-082, SCENARIO-PHASE4-082.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import carnot.experiment_4350_e3_explore_verify_plan_ka59 as exp
from carnot.agentic.arc_executable_world_model import Transition, VerifyResult


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"
KA59_MODEL = REPO / "results" / "arc_e3" / "ka59" / "world_model.py"


def _transition(action: int = 4) -> Transition:
    g0 = np.array([[14, 0, 14], [14, 14, 14]], dtype=int)
    g1 = np.array([[14, 4, 14], [14, 14, 14]], dtype=int)
    return Transition(g0, action, None, g1, 0, 0)


def _load_ka59_world_model():
    spec = importlib.util.spec_from_file_location("ka59_world_model_test", KA59_MODEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req_phase4_082_spec_declares_exp4350_contract() -> None:
    """REQ-PHASE4-082: OpenSpec declares the ka59 continuation contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-082" in spec
    assert "SCENARIO-PHASE4-082" in spec
    assert "experiment_4350_e3_explore_verify_plan_ka59.json" in spec
    assert "blocked_offline_env_missing_ka59" in spec
    assert "success_e3_ka59_L1_reproduced" in spec
    assert "complete_e3_ka59_partial_model_<acc>" in spec
    assert "full 5x5 target-border rectangle" in spec
    assert "compare to .401's 0.56" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_082_target_underlay_is_bounded_to_confirmed_target_rectangle() -> None:
    """REQ-PHASE4-082: ka59 erase underlay does not leak nearby target rows."""

    wm = _load_ka59_world_model()
    grid = np.ones((12, 12), dtype=int)
    grid[5, 0:5] = wm.TARGET
    grid[9, 0:5] = wm.TARGET
    grid[5:10, 0] = wm.TARGET
    grid[5:10, 4] = wm.TARGET

    outside_target = wm._target_underlay(grid, 3, 5)
    overlapping_target = wm._target_underlay(grid, 3, 2)

    assert np.array_equal(outside_target, np.full((3, 3), wm.BACKGROUND, dtype=int))
    assert overlapping_target[2].tolist() == [wm.TARGET, wm.TARGET, wm.TARGET]
    assert np.array_equal(overlapping_target[:2], np.full((2, 3), wm.BACKGROUND, dtype=int))
    assert wm._is_target_border(np.ones((4, 5), dtype=int)) is False


def test_scenario_phase4_082_blocked_artifact_is_terminal_and_schema_valid(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-082: missing ka59 offline env blocks without fabrication."""

    artifact = exp.blocked_artifact(tmp_path, random_seed=4350)

    assert artifact["experiment"] == "experiment_4350_e3_explore_verify_plan_ka59"
    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ka59"
    assert artifact["verifier_accuracy_per_round"] == []
    assert artifact["explore_lemmas_collected"] == 0
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["spec_refs"] == ["REQ-PHASE4-082", "SCENARIO-PHASE4-082"]
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_082_helper_branches_and_schema_errors_are_specific(tmp_path: Path) -> None:
    """REQ-PHASE4-082: helper branches stay deterministic and schema errors stay explicit."""

    outside = tmp_path / "outside" / "world_model.py"
    external = tmp_path.parent / f"{tmp_path.name}_external" / "world_model.py"
    assert exp._relative_or_absolute(tmp_path, outside) == str(outside)
    assert exp._relative_or_absolute(tmp_path, external) == str(external)
    assert exp._rounds_with_prior(exp.PRIOR_EXP4340_ACCURACY) == [exp.PRIOR_EXP4340_ACCURACY]
    assert exp._plan_executed(None) is False

    bad = {
        "honest_verdict": "complete_e3_ka59_partial_model_0.00",
        "verifier_accuracy_per_round": "not-list",
        "explore_lemmas_collected": "0",
        "world_model_path": exp.WORLD_MODEL_RELATIVE_PATH,
        "world_model_sha256": "",
        "offline_reproduced": {"value": False},
        "reproduced_levels": "0",
        "plan_executed": None,
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4350,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)
    assert "missing world_model_sha256" not in errors
    assert "verifier_accuracy_per_round must be list" in errors
    assert "explore_lemmas_collected must be bare int" in errors
    assert "offline_reproduced must be bare bool" in errors
    assert "plan_executed must be bare bool" in errors
    assert "reproduced_levels must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_req_phase4_082_build_artifact_preserves_bare_gate_fields(tmp_path: Path) -> None:
    """REQ-PHASE4-082: solve gates remain bare values while principles are preserved."""

    model_path = tmp_path / exp.WORLD_MODEL_RELATIVE_PATH
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    artifact = exp.build_artifact(
        repo=tmp_path,
        verifier_accuracy_per_round=[0.5625, 0.9375],
        explore_lemmas_collected=8,
        world_model_path=model_path,
        plan_result={"planned": True, "executed": False, "divergence_step": {"action": 4}},
        reproduce_result={
            "game": "ka59",
            "reached_level": 0,
            "claimed_level": 1,
            "reproduced": False,
        },
        residual_mismatch_class="visible_target_underlay_gap",
        adaptive_tests_generated=4,
        random_seed=4350,
        duration_s=1.25,
    )

    assert artifact["experiment"] == "experiment_4350_e3_explore_verify_plan_ka59"
    assert artifact["honest_verdict"] == "complete_e3_ka59_partial_model_0.94"
    assert artifact["world_model_path"] == exp.WORLD_MODEL_RELATIVE_PATH
    assert artifact["world_model_sha256"] == exp.sha256_file(model_path)
    assert artifact["offline_reproduced"] is False
    assert artifact["reproduced_levels"] == 0
    assert artifact["plan_executed"] is False
    assert artifact["plan_executed_detail"]["divergence_step"] == {"action": 4}
    assert artifact["field_principles"]["verifier_accuracy_per_round"].endswith(
        "compare to .401's 0.56)."
    )
    assert artifact["spec_refs"] == ["REQ-PHASE4-082", "SCENARIO-PHASE4-082"]
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_082_gap_and_registry_writes_are_stable(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-082: partial gaps and success registry entries are auditable."""

    gap_path = tmp_path / "ops" / "verifier_gaps.md"
    exp._write_gap(gap_path, best_accuracy=0.1, mismatch_class="first", checksum="a" * 64)
    exp._write_gap(gap_path, best_accuracy=0.2, mismatch_class="second", checksum="b" * 64)
    gap_text = gap_path.read_text(encoding="utf-8")
    assert "Best verifier accuracy: 0.2000" in gap_text
    assert "second" in gap_text
    assert "first" not in gap_text

    missing_registry = tmp_path / "ops" / "missing.yaml"
    exp._update_registry_for_success(
        missing_registry, world_model_sha256="c" * 64, checksum="d" * 64
    )
    assert not missing_registry.exists()

    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "games:\n"
        "  - game: ar25\n"
        "    reproducibility: reproduced\n"
        "  # ... 15 still-unsolved games\n"
        "reproducible_total_levels: 15\n"
        "reproducible_total_games: 12\n"
        "#   + tu93 1 + cn04 1 + m0r0 1 + sk48 1 + ar25 1 = 15 across 12 games\n",
        encoding="utf-8",
    )
    exp._update_registry_for_success(registry, world_model_sha256="c" * 64, checksum="d" * 64)
    registry_text = registry.read_text(encoding="utf-8")
    assert "game: ka59" in registry_text
    assert "experiment_4350_e3_explore_verify_plan_ka59.json" in registry_text
    assert "reproducible_total_levels: 16" in registry_text
    exp._update_registry_for_success(registry, world_model_sha256="e" * 64, checksum="f" * 64)
    assert registry.read_text(encoding="utf-8") == registry_text


def test_scenario_phase4_082_run_experiment_blocks_when_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-082: runner writes blocked artifact when ka59 env is absent."""

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)

    artifact = exp.run_experiment(random_seed=4350, n_transitions=1, round_budget=8)

    assert artifact["honest_verdict"] == "blocked_offline_env_missing_ka59"
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_scenario_phase4_082_run_experiment_writes_success_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-082: runner accepts success only after reproduction returns L1."""

    env_dir = tmp_path / "environment_files" / "ka59"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ka59" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(
        exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1)
    )
    monkeypatch.setattr(
        exp.e3, "load_engine", lambda _game: (lambda grid, _action, _data: grid, lambda _grid: True)
    )
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1, n_correct=1, accuracy=1.0, mismatches=[]
        ),
    )
    monkeypatch.setattr(
        exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [{"verifier_gated": True}]
    )
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [{"passed": True}])
    monkeypatch.setattr(
        exp,
        "execute_model_grounded_plan",
        lambda _engine: {
            "planned": True,
            "executed": True,
            "level_up": True,
            "solution": list(exp.L1_SOLUTION_LABELS),
        },
    )
    monkeypatch.setattr(
        exp.arc_solver_kit,
        "reproduce",
        lambda *_args, **_kwargs: {
            "game": "ka59",
            "reached_level": 1,
            "claimed_level": 1,
            "reproduced": True,
        },
    )

    artifact = exp.run_experiment(random_seed=4350, n_transitions=1, round_budget=8)

    assert artifact["honest_verdict"] == "success_e3_ka59_L1_reproduced"
    assert artifact["verifier_accuracy_per_round"] == [exp.PRIOR_EXP4340_ACCURACY, 1.0]
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 1
    assert artifact["plan_executed"] is True
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()
    assert not (tmp_path / exp.GAP_RELATIVE_PATH).exists()


def test_scenario_phase4_082_run_experiment_records_partial_and_schema_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-082: unsolved runs record gaps and malformed artifacts fail."""

    env_dir = tmp_path / "environment_files" / "ka59"
    env_dir.mkdir(parents=True)
    (env_dir / "fixture").write_text("present", encoding="utf-8")
    model_path = tmp_path / "results" / "arc_e3" / "ka59" / "world_model.py"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "WORLD_MODEL_PATH", model_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "GAP_PATH", tmp_path / exp.GAP_RELATIVE_PATH)
    monkeypatch.setattr(exp, "REGISTRY_PATH", tmp_path / exp.REGISTRY_RELATIVE_PATH)
    monkeypatch.setattr(
        exp.e3, "collect_transitions", lambda *_args, **_kwargs: ([_transition()], 1)
    )
    monkeypatch.setattr(
        exp.e3,
        "load_engine",
        lambda _game: (lambda grid, _action, _data: grid, lambda _grid: False),
    )
    monkeypatch.setattr(
        exp.e3.WorldModelVerifier,
        "score",
        lambda self, _engine, max_mismatch=8: VerifyResult(
            n=1, n_correct=0, accuracy=0.0, mismatches=[]
        ),
    )
    monkeypatch.setattr(exp, "collect_explore_lemmas", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(exp, "adaptive_world_model_tests", lambda _engine: [])
    monkeypatch.setattr(exp, "execute_model_grounded_plan", lambda _engine: {"planned": False})

    artifact = exp.run_experiment(random_seed=4350, n_transitions=1, round_budget=8)
    assert artifact["honest_verdict"] == "complete_e3_ka59_partial_model_0.56"
    assert (tmp_path / exp.GAP_RELATIVE_PATH).exists()

    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])
    with pytest.raises(ValueError, match="Exp4350 artifact schema errors"):
        exp.run_experiment(random_seed=4350, n_transitions=1, round_budget=8)
