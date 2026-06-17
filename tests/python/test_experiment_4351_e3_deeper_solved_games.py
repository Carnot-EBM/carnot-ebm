"""Tests for Exp 4351 E3 deeper solved-games checkpoint run.

Spec refs: REQ-PHASE4-083, SCENARIO-PHASE4-083.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import carnot.experiment_4351_e3_deeper_solved_games as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _row(
    game: str,
    *,
    prior: int = 1,
    reached: int = 1,
    accuracy: float = 0.5,
    advanced: bool = False,
) -> dict:
    return {
        "game": game,
        "prior_best_level": prior,
        "new_reproduced_level": reached,
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "offline_reproduced": advanced,
        "reproduce_result": {
            "game": game,
            "reached_level": reached,
            "claimed_level": reached,
            "reproduced": advanced,
        },
        "plan": ["mock"],
        "residual_win_mechanic_gap_class": "none" if advanced else "bounded_deepen_no_new_level",
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial",
        "world_model_path": exp.WORLD_MODEL_PATHS[game],
    }


def test_req_phase4_083_spec_declares_exp4351_contract() -> None:
    """REQ-PHASE4-083: OpenSpec declares the multi-target deepen contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-083" in spec
    assert "SCENARIO-PHASE4-083" in spec
    assert "experiment_4351_e3_deeper_solved_games.json" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_deeper_<targets>_reproduced" in spec
    assert "complete_e3_deeper_partial" in spec
    assert "timed spike-trap program-editor solver path through L7" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_083_checksum_binds_rows_paths_and_seed() -> None:
    """REQ-PHASE4-083: checksum binds scorecard, paths, reproduction, and seed."""

    rows = [_row("sc25"), _row("tn36", prior=6, reached=7, advanced=True), _row("ar25")]
    hashes = {"a.py": "a" * 64}
    base = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4351,
    )
    same = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4351,
    )
    changed = exp.compute_reproducibility_checksum(
        per_target_scorecard=[{**rows[1], "new_reproduced_level": 6}],
        world_model_paths=["a.py"],
        path_hashes=hashes,
        random_seed=4351,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_083_build_artifact_success_counts_only_new_levels(tmp_path: Path) -> None:
    """REQ-PHASE4-083: only levels beyond the prior best count as new progress."""

    model = tmp_path / "results" / "arc_e3" / "sc25" / "world_model.py"
    model.parent.mkdir(parents=True)
    model.write_text("def engine(grid, action, data):\n    return grid\n", encoding="utf-8")
    solver = tmp_path / "scripts" / "arc3_tn36_offline_solver.py"
    solver.parent.mkdir(parents=True)
    solver.write_text("# solver\n", encoding="utf-8")

    rows = [
        _row("sc25", prior=1, reached=1, accuracy=1.0, advanced=False),
        _row("tn36", prior=6, reached=7, accuracy=1.0, advanced=True),
        _row("ar25", prior=1, reached=1, accuracy=0.8875, advanced=False),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=23,
        world_model_paths=[str(model.relative_to(tmp_path)), str(solver.relative_to(tmp_path))],
        random_seed=4351,
        duration_s=2.5,
    )

    assert artifact["honest_verdict"] == "success_e3_deeper_tn36_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 23
    assert artifact["verifier_is_oracle"] is True
    assert artifact["per_target_scorecard"][1]["game"] == "tn36"
    assert artifact["per_target_scorecard"][1]["offline_reproduced"] is True
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_083_partial_artifact_preserves_all_targets(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-083: all-partial runs keep one row per target and bare gates."""

    rows = [
        _row("sc25", accuracy=0.75),
        _row("tn36", prior=6, reached=6, accuracy=0.4),
        _row("ar25", accuracy=0.8875),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=22,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4351,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["verifier_is_oracle"] is True
    assert isinstance(artifact["reproducible_total_levels"], int)
    assert isinstance(artifact["new_levels_reproduced"], int)
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_083_schema_errors_are_specific() -> None:
    """REQ-PHASE4-083: schema validation catches wrapped or malformed gate fields."""

    bad = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": "not-list",
        "reproducible_total_levels": {"value": 23},
        "new_levels_reproduced": "1",
        "world_model_paths": ["a.py"],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4351,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard must be list" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_scenario_phase4_083_run_experiment_records_missing_envs_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-083: missing target envs block per target without fabrication."""

    env = tmp_path / "environment_files" / "tn36"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    solver = tmp_path / "scripts" / "arc3_tn36_offline_solver.py"
    solver.parent.mkdir(parents=True)
    solver.write_text("# solver\n", encoding="utf-8")

    calls: list[str] = []

    def fake_tn36_runner(_repo: Path, _random_seed: int) -> dict:
        calls.append("tn36")
        return _row("tn36", prior=6, reached=7, accuracy=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "tn36", fake_tn36_runner)

    artifact = exp.run_experiment(random_seed=4351)

    assert calls == ["tn36"]
    assert artifact["honest_verdict"] == "success_e3_deeper_tn36_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_target_scorecard"]] == [
        "blocked_offline_env_missing_sc25",
        "new_level_reproduced",
        "blocked_offline_env_missing_ar25",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_083_prior_artifact_rows_cover_present_and_missing_inputs(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-083: existing L1 artifacts are partials unless they advance deeper."""

    sc25_result = tmp_path / "results" / "experiment_4341_e3_sc25_reproduction.json"
    sc25_result.parent.mkdir(parents=True)
    sc25_result.write_text(
        json.dumps(
            {
                "verifier_accuracy_per_round": [0.5, 1.0],
                "offline_reproduced": True,
                "reproduced_levels": 1,
                "accepted_plan": ["cell0,1"],
            }
        ),
        encoding="utf-8",
    )

    sc25 = exp._run_sc25_target(tmp_path, 4351)
    ar25 = exp._run_ar25_target(tmp_path, 4351)

    assert sc25["verifier_accuracy"] == 1.0
    assert sc25["offline_reproduced"] is False
    assert sc25["plan"] == ["cell0,1"]
    assert sc25["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert ar25["verifier_accuracy"] == 0.0
    assert ar25["checkpoint_status"] == "honest_partial_prior_artifact_missing"


def test_req_phase4_083_tn36_runner_uses_reproduction_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-083: tn36 L7 counts only after the reproduction gate returns true."""

    class FakeSolver:
        @staticmethod
        def solve(max_level: int, cap: int):
            assert (max_level, cap) == (7, 400)
            return ([{"action": 6, "data": {"x": 1, "y": 2}}], 7)

    class FakeEnv:
        def __init__(self) -> None:
            self.calls = []

        def step(self, action, data=None):
            self.calls.append((action, data))
            return {"frame": True}

    def fake_reproduce(game, labels, apply, claimed_level):
        frame = apply(FakeEnv(), labels[0], None)
        assert frame == {"frame": True}
        return {
            "game": game,
            "reached_level": claimed_level,
            "claimed_level": claimed_level,
            "reproduced": True,
        }

    monkeypatch.setattr(exp, "_load_tn36_solver", lambda _repo: FakeSolver)
    monkeypatch.setattr(exp.arc_solver_kit, "reproduce", fake_reproduce)

    row = exp._run_tn36_target(tmp_path, 4351)

    assert row["game"] == "tn36"
    assert row["prior_best_level"] == 6
    assert row["new_reproduced_level"] == 7
    assert row["offline_reproduced"] is True
    assert row["trajectory_action_count"] == 1


def test_req_phase4_083_tn36_loader_and_registry_total_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-083: solver loading and registry-total parsing stay deterministic."""

    solver_path = tmp_path / exp.WORLD_MODEL_PATHS["tn36"]
    solver_path.parent.mkdir(parents=True)
    solver_path.write_text("VALUE = 17\n", encoding="utf-8")

    module = exp._load_tn36_solver(tmp_path)
    assert module.VALUE == 17

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    with pytest.raises(ImportError, match="cannot load"):
        exp._load_tn36_solver(tmp_path)

    registry = tmp_path / exp.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_text("reproducible_total_levels: 23\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) == 23
    registry.write_text("no total here\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) is None
    registry.unlink()
    assert exp._registry_total(tmp_path) is None


def test_req_phase4_083_schema_validation_covers_row_shape_errors() -> None:
    """REQ-PHASE4-083: malformed scorecard rows produce specific schema errors."""

    artifact = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": ["bad-row", {"game": "tn36", "offline_reproduced": "yes"}],
        "reproducible_total_levels": 23,
        "new_levels_reproduced": 0,
        "world_model_paths": [123],
        "verifier_is_oracle": "true",
        "preconditions_checked": {},
        "random_seed": 4351,
        "reproducibility_checksum": "a" * 64,
        "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_target_scorecard[0] must be dict" in errors
    assert "per_target_scorecard[1] missing prior_best_level" in errors
    assert "per_target_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be bare bool" in errors


def test_scenario_phase4_083_run_experiment_raises_on_internal_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-083: invalid assembled artifacts fail closed before writing."""

    checks = {
        "targets": {
            game: {"offline_env_present": False, "offline_env_path": str(tmp_path / game)}
            for game in exp.TARGET_ORDER
        },
        "harness_import": True,
        "solver_kit_import": True,
        "executable_world_model_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "preconditions", lambda _repo: checks)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4351 artifact schema errors"):
        exp.run_experiment(random_seed=4351)
