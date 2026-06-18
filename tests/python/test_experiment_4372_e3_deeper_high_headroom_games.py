"""Tests for Exp 4372 E3 deeper high-headroom next-level sweep.

Spec refs: REQ-PHASE4-4372, SCENARIO-PHASE4-4372.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import carnot.experiment_4372_e3_deeper_high_headroom_games as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "phase4_active_inference" / "spec.md"


def _row(
    game: str,
    *,
    prior: int | None = None,
    reached: int | None = None,
    accuracy: float = 0.5,
    advanced: bool = False,
) -> dict:
    prior_level = exp.PRIOR_BEST_LEVELS[game] if prior is None else prior
    reached_level = prior_level if reached is None else reached
    return {
        "game": game,
        "prior_best_level": prior_level,
        "new_reproduced_level": reached_level,
        "target_level": exp.TARGET_LEVELS[game],
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "offline_reproduced": advanced,
        "reproduce_result": {
            "game": game,
            "reached_level": reached_level,
            "claimed_level": reached_level,
            "reproduced": advanced,
        },
        "plan": ["mock"],
        "residual_win_mechanic_gap_class": "none" if advanced else "bounded_deepen_no_new_level",
        "checkpoint_status": "new_level_reproduced" if advanced else "honest_partial",
        "world_model_path": exp.WORLD_MODEL_PATHS[game],
    }


def test_req_phase4_4372_spec_declares_contract() -> None:
    """REQ-PHASE4-4372: OpenSpec declares the five-target Exp 4372 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4372" in spec
    assert "SCENARIO-PHASE4-4372" in spec
    assert "experiment_4372_e3_deeper_high_headroom_games.json" in spec
    assert "tn36`, `tr87`, `lp85`, `tu93`, and `sc25`" in spec
    assert "prior `reproducible_total_levels=33`" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_deeper_<targets>_reproduced" in spec
    assert "complete_e3_deeper_partial" in spec
    assert "sc25_l2_spell_delta_gap" in spec
    assert "tn36` L8 program-editor object-control" in spec
    assert "tr87` L7" in spec
    assert "tu93` L5 frame-based fresh_env branch-mode" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4372_checksum_binds_rows_paths_seed_and_caps() -> None:
    """REQ-PHASE4-4372: checksum binds scorecard, paths, seed, and cap metadata."""

    rows = [
        _row("tn36"),
        _row("tr87"),
        _row("lp85"),
        _row("tu93", reached=5, accuracy=1.0, advanced=True),
        _row("sc25"),
    ]
    hashes = {"solver.py": "a" * 64}
    base = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4372,
        target_wall_time_s=1.5,
    )
    same = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4372,
        target_wall_time_s=1.5,
    )
    changed = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4372,
        target_wall_time_s=2.0,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_4372_build_artifact_counts_only_new_reproduced_levels(tmp_path: Path) -> None:
    """REQ-PHASE4-4372: only levels beyond prior best count as new progress."""

    solver = tmp_path / "python" / "carnot" / "agentic" / "arc_game_adapters.py"
    solver.parent.mkdir(parents=True)
    solver.write_text("# adapters\n", encoding="utf-8")

    rows = [
        _row("tn36", accuracy=0.875),
        _row("tr87", reached=7, accuracy=1.0, advanced=True),
        _row("lp85", accuracy=0.8),
        _row("tu93", accuracy=0.8),
        _row("sc25", accuracy=0.5),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=34,
        world_model_paths=[str(solver.relative_to(tmp_path))],
        random_seed=4372,
        target_wall_time_s=1.5,
        duration_s=2.5,
    )

    assert artifact["honest_verdict"] == "success_e3_deeper_tr87_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["verifier_is_oracle"] is True
    assert artifact["target_wall_time_s"] == 1.5
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_4372_partial_artifact_preserves_all_targets(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-4372: all-partial runs keep one row per target and bare gates."""

    rows = [
        _row("tn36", accuracy=0.875),
        _row("tr87", accuracy=0.857),
        _row("lp85", accuracy=0.8),
        _row("tu93", accuracy=0.8),
        _row("sc25", accuracy=0.5),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=33,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4372,
        target_wall_time_s=1.0,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["verifier_is_oracle"] is True
    assert isinstance(artifact["reproducible_total_levels"], int)
    assert isinstance(artifact["new_levels_reproduced"], int)
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4372_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4372: schema validation catches wrapped or malformed gate fields."""

    bad = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": "not-list",
        "reproducible_total_levels": {"value": 33},
        "new_levels_reproduced": "1",
        "world_model_paths": ["a.py"],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4372,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
        "target_wall_time_s": "1.0",
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard must be list" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "target_wall_time_s must be numeric" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_req_phase4_4372_schema_validation_covers_row_shape_errors() -> None:
    """REQ-PHASE4-4372: malformed scorecard rows produce specific schema errors."""

    artifact = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": ["bad-row", {"game": "tn36", "offline_reproduced": "yes"}],
        "reproducible_total_levels": 33,
        "new_levels_reproduced": 0,
        "world_model_paths": [123],
        "verifier_is_oracle": "true",
        "preconditions_checked": {},
        "random_seed": 4372,
        "target_wall_time_s": 1.0,
        "reproducibility_checksum": "a" * 64,
        "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_target_scorecard[0] must be dict" in errors
    assert "per_target_scorecard[1] missing prior_best_level" in errors
    assert "per_target_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be bare bool" in errors


def test_scenario_phase4_4372_run_experiment_records_missing_envs_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4372: missing target envs block per target without fabrication."""

    env = tmp_path / "environment_files" / "tu93"
    env.mkdir(parents=True)
    (env / "fixture").write_text("present", encoding="utf-8")
    adapter = tmp_path / exp.WORLD_MODEL_PATHS["tu93"]
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# adapters\n", encoding="utf-8")

    calls: list[str] = []

    def fake_tu93_runner(_repo: Path, _random_seed: int) -> dict:
        calls.append("tu93")
        return _row("tu93", reached=5, accuracy=1.0, advanced=True)

    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setitem(exp.TARGET_RUNNERS, "tu93", fake_tu93_runner)

    artifact = exp.run_experiment(random_seed=4372, target_wall_time_s=None)

    assert calls == ["tu93"]
    assert artifact["honest_verdict"] == "success_e3_deeper_tu93_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_target_scorecard"]] == [
        "blocked_offline_env_missing_tn36",
        "blocked_offline_env_missing_tr87",
        "blocked_offline_env_missing_lp85",
        "new_level_reproduced",
        "blocked_offline_env_missing_sc25",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()


def test_req_phase4_4372_timeout_row_is_honest_partial() -> None:
    """REQ-PHASE4-4372: per-target wall-time exhaustion preserves an honest row."""

    row = exp.timeout_target_row("lp85", target_wall_time_s=0.01)

    assert row["game"] == "lp85"
    assert row["new_reproduced_level"] == exp.PRIOR_BEST_LEVELS["lp85"]
    assert row["offline_reproduced"] is False
    assert row["checkpoint_status"] == "honest_partial_wall_time_cap_exhausted"
    assert row["residual_win_mechanic_gap_class"] == "wall_time_cap_exhausted"


def test_req_phase4_4372_exception_row_is_honest_partial() -> None:
    """REQ-PHASE4-4372: target exceptions are preserved without claiming progress."""

    row = exp.exception_target_row("tr87", "Traceback\nValueError: boom")

    assert row["game"] == "tr87"
    assert row["new_reproduced_level"] == exp.PRIOR_BEST_LEVELS["tr87"]
    assert row["offline_reproduced"] is False
    assert row["checkpoint_status"] == "honest_partial_target_exception"
    assert row["reproduce_result"]["exception"] == "ValueError: boom"


def test_req_phase4_4372_prior_artifact_row_covers_present_and_missing_inputs(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-4372: existing sc25 L1 artifacts are partials unless they advance deeper."""

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

    present = exp._run_sc25_target(tmp_path, 4372)
    missing = exp._prior_artifact_row(
        repo=tmp_path,
        game="lp85",
        result_relative_path="results/missing.json",
        residual_gap="lp85_l5_permutation_bfs_no_new_reproduction_gap",
    )

    assert present["verifier_accuracy"] == 1.0
    assert present["offline_reproduced"] is False
    assert present["plan"] == ["cell0,1"]
    assert present["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert present["residual_win_mechanic_gap_class"] == "sc25_l2_spell_delta_gap"
    assert missing["verifier_accuracy"] == 0.0
    assert missing["checkpoint_status"] == "honest_partial_prior_artifact_missing"


def test_req_phase4_4372_tn36_runner_counts_only_new_reproduced_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: tn36 L8 counts only after the reproduction gate returns true."""

    class FakeSolver:
        @staticmethod
        def solve(max_level: int, cap: int):
            assert (max_level, cap) == (8, 500)
            return ([{"action": 6, "data": {"x": 1, "y": 2}}], 8)

    class FakeEnv:
        def step(self, action, data=None):
            return {"action": action, "data": data}

    def fake_reproduce(game, labels, apply, claimed_level):
        frame = apply(FakeEnv(), labels[0], None)
        assert frame["data"] == {"x": 1, "y": 2}
        return {
            "game": game,
            "reached_level": claimed_level,
            "claimed_level": claimed_level,
            "reproduced": True,
        }

    monkeypatch.setattr(exp, "_load_tn36_solver", lambda _repo: FakeSolver)
    monkeypatch.setattr(exp.arc_solver_kit, "reproduce", fake_reproduce)

    row = exp._run_tn36_target(tmp_path, 4372)

    assert row["game"] == "tn36"
    assert row["prior_best_level"] == 7
    assert row["new_reproduced_level"] == 8
    assert row["offline_reproduced"] is True
    assert row["trajectory_action_count"] == 1


def test_req_phase4_4372_adaptered_runner_accepts_only_reproduction_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: adaptered search hits are partials if reset replay rejects them."""

    def fake_solve_adaptered(game: str, target_level: int) -> dict:
        assert (game, target_level) == ("tu93", 5)
        return {
            "game": "tu93",
            "target": 5,
            "reached_level": 5,
            "moves": 66,
            "states_expanded": 962,
            "offline_reproduced": False,
            "solution_labels": ["a0"],
            "verifier_src": "hand_verifier_cold_start",
            "reproduction_gate": {
                "game": "tu93",
                "reached_level": 4,
                "claimed_level": 5,
                "reproduced": False,
            },
        }

    monkeypatch.setattr(exp, "_solve_adaptered", fake_solve_adaptered)

    row = exp._run_tu93_target(Path("."), 4372)

    assert row["new_reproduced_level"] == 4
    assert row["searched_level"] == 5
    assert row["offline_reproduced"] is False
    assert row["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert row["residual_win_mechanic_gap_class"] == (
        "tu93_l5_fresh_env_branch_mode_no_new_reproduction_gap"
    )


def test_req_phase4_4372_adaptered_target_wrappers_use_target_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: tr87 and lp85 wrappers route to their requested next levels."""

    calls: list[tuple[str, int]] = []

    def fake_solve_adaptered(game: str, target_level: int) -> dict:
        calls.append((game, target_level))
        return {
            "game": game,
            "target": target_level,
            "reached_level": exp.PRIOR_BEST_LEVELS[game],
            "offline_reproduced": False,
            "solution_labels": [],
            "reproduction_gate": {
                "game": game,
                "reached_level": exp.PRIOR_BEST_LEVELS[game],
                "claimed_level": target_level,
                "reproduced": False,
            },
        }

    monkeypatch.setattr(exp, "_solve_adaptered", fake_solve_adaptered)

    tr87 = exp._run_tr87_target(Path("."), 4372)
    lp85 = exp._run_lp85_target(Path("."), 4372)

    assert calls == [("tr87", 7), ("lp85", 5)]
    assert tr87["residual_win_mechanic_gap_class"] == (
        "tr87_l7_no_offline_level_available_or_no_new_reproduction_gap"
    )
    assert lp85["residual_win_mechanic_gap_class"] == (
        "lp85_l5_permutation_bfs_no_new_reproduction_gap"
    )


def test_req_phase4_4372_solve_adaptered_wrapper_imports_loop_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: adaptered games route through the reusable loop solver."""

    module = types.ModuleType("scripts.arc_loop_solve")
    module.solve_adaptered = lambda game, target_level: {  # type: ignore[attr-defined]
        "game": game,
        "target": target_level,
    }
    monkeypatch.setitem(sys.modules, "scripts.arc_loop_solve", module)

    assert exp._solve_adaptered("tr87", 7) == {"game": "tr87", "target": 7}


def test_req_phase4_4372_loader_registry_and_internal_schema_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: loader, registry parsing, and fail-closed schema stay deterministic."""

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
    registry.write_text("reproducible_total_levels: 33\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) == 33
    registry.write_text("no total here\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) is None
    registry.unlink()
    assert exp._registry_total(tmp_path) is None

    checks = {
        "targets": {
            game: {"offline_env_present": False, "offline_env_path": str(tmp_path / game)}
            for game in exp.TARGET_ORDER
        },
        "harness_import": True,
        "solver_kit_import": True,
        "arc_loop_solve_import": True,
        "executable_world_model_import": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "preconditions", lambda _repo: checks)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4372 artifact schema errors"):
        exp.run_experiment(random_seed=4372, target_wall_time_s=None)


def test_req_phase4_4372_precondition_failure_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: import and conductor-modified checks fail closed."""

    def fake_import_module(name: str):
        if name == "scripts.arc_loop_solve":
            raise RuntimeError("missing")
        return object()

    monkeypatch.setattr(exp.importlib, "import_module", fake_import_module)
    imports = exp._imports_ok()

    assert imports == {
        "harness_import": True,
        "solver_kit_import": True,
        "arc_loop_solve_import": False,
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


def test_req_phase4_4372_target_worker_reports_success_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: worker payloads preserve either row or traceback."""

    class FakeQueue:
        def __init__(self) -> None:
            self.items: list[dict] = []

        def put(self, item: dict) -> None:
            self.items.append(item)

    ok_queue = FakeQueue()
    monkeypatch.setitem(exp.TARGET_RUNNERS, "sc25", lambda _repo, _seed: _row("sc25"))
    exp._target_worker("sc25", ".", 4372, ok_queue)  # type: ignore[arg-type]

    assert ok_queue.items[0]["ok"] is True
    assert ok_queue.items[0]["row"]["game"] == "sc25"

    bad_queue = FakeQueue()

    def bad_runner(_repo: Path, _seed: int) -> dict:
        raise ValueError("boom")

    monkeypatch.setitem(exp.TARGET_RUNNERS, "sc25", bad_runner)
    exp._target_worker("sc25", ".", 4372, bad_queue)  # type: ignore[arg-type]

    assert bad_queue.items[0]["ok"] is False
    assert "ValueError: boom" in bad_queue.items[0]["traceback"]


def test_req_phase4_4372_run_target_with_cap_covers_timeout_empty_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4372: subprocess timeout/empty/success/error paths become honest rows."""

    class FakeQueue:
        payload: dict | None = None
        empty = False

        def get_nowait(self) -> dict:
            if self.empty:
                raise exp.queue.Empty
            assert self.payload is not None
            return self.payload

    class FakeProcess:
        alive = False

        def __init__(self, *_args, **_kwargs) -> None:
            self.terminated = False

        def start(self) -> None:
            pass

        def join(self, *_args) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated = True

    monkeypatch.setattr(exp.mp, "Queue", lambda: FakeQueue())
    monkeypatch.setattr(exp.mp, "Process", FakeProcess)

    FakeProcess.alive = True
    row = exp._run_target_with_cap("lp85", Path("."), 4372, 0.01)
    assert row["checkpoint_status"] == "honest_partial_wall_time_cap_exhausted"

    FakeProcess.alive = False
    FakeQueue.empty = True
    row = exp._run_target_with_cap("lp85", Path("."), 4372, 0.01)
    assert row["checkpoint_status"] == "honest_partial_target_exception"
    assert row["reproduce_result"]["exception"] == "lp85 runner exited without result"

    FakeQueue.empty = False
    FakeQueue.payload = {"ok": True, "row": _row("lp85", reached=5, advanced=True)}
    row = exp._run_target_with_cap("lp85", Path("."), 4372, 0.01)
    assert row["offline_reproduced"] is True

    FakeQueue.payload = {"ok": False, "traceback": "RuntimeError: fail"}
    row = exp._run_target_with_cap("lp85", Path("."), 4372, 0.01)
    assert row["checkpoint_status"] == "honest_partial_target_exception"
    assert row["reproduce_result"]["exception"] == "RuntimeError: fail"

    def bad_runner(_repo: Path, _seed: int) -> dict:
        raise RuntimeError("direct fail")

    monkeypatch.setitem(exp.TARGET_RUNNERS, "lp85", bad_runner)
    row = exp._run_target_with_cap("lp85", Path("."), 4372, None)
    assert row["checkpoint_status"] == "honest_partial_target_exception"


def test_req_phase4_4372_path_hashes_cover_missing_and_present_files(tmp_path: Path) -> None:
    """REQ-PHASE4-4372: path hashes are deterministic for present files and empty for missing."""

    present = tmp_path / "solver.py"
    present.write_text("# solver\n", encoding="utf-8")

    hashes = exp._path_hashes(tmp_path, ["solver.py", "missing.py"])

    assert hashes["solver.py"] == hashlib.sha256(b"# solver\n").hexdigest()
    assert hashes["missing.py"] == ""
