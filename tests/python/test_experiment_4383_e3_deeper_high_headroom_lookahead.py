"""Tests for Exp 4383 Mind-Studio lookahead E3 deepen sweep.

Spec refs: REQ-PHASE4-4383, SCENARIO-PHASE4-4383.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import carnot.experiment_4383_e3_deeper_high_headroom_lookahead as exp


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
    target_level = exp.TARGET_LEVELS[game]
    return {
        "game": game,
        "prior_best_level": prior_level,
        "new_reproduced_level": reached_level,
        "searched_level": reached_level,
        "target_level": target_level,
        "verifier_accuracy": accuracy,
        "verifier_accuracy_per_round": [accuracy],
        "lookahead_fidelity": exp.compute_lookahead_fidelity(
            reached_level=reached_level,
            target_level=target_level,
            reproduced=advanced,
        ),
        "lookahead_fidelity_per_round": [
            exp.compute_lookahead_fidelity(
                reached_level=reached_level,
                target_level=target_level,
                reproduced=advanced,
            )
        ],
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
        "mind_studio_skill_file": f"results/arc_e3/{game}/skill_4383.json",
        "entropy_selected_traces": [f"{game}:entropy_rank0:L{prior_level}->L{target_level}"],
        "lookahead_k": exp.LOOKAHEAD_K,
        "lookahead_status": "accepted_for_planning" if advanced else "honest_partial_prefix_only",
    }


def test_req_phase4_4383_spec_declares_contract() -> None:
    """REQ-PHASE4-4383: OpenSpec declares the four-target lookahead contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-PHASE4-4383" in spec
    assert "SCENARIO-PHASE4-4383" in spec
    assert "experiment_4383_e3_deeper_high_headroom_lookahead.json" in spec
    assert "`lp85`, `tu93`, `tn36`, and `tr87`" in spec
    assert "prior `reproducible_total_levels=34`" in spec
    assert "K-step LOOKAHEAD-FIDELITY" in spec
    assert "skill_4383.json" in spec
    assert "sxhtkytekm" in spec
    assert "blocked_offline_env_missing_<game>" in spec
    assert "success_e3_deeper_<targets>_reproduced" in spec
    assert "complete_e3_deeper_partial" in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for principle in exp.REQUIRED_FIELD_PRINCIPLES.values():
        assert principle in spec


def test_req_phase4_4383_skill_file_records_mind_studio_preplanning_inputs(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-4383: each target emits a lightweight Mind-Studio skill file."""

    path = exp.write_skill_file(tmp_path, "tn36", random_seed=4383, lookahead_k=3)
    data = json.loads((tmp_path / path).read_text(encoding="utf-8"))

    assert path == "results/arc_e3/tn36/skill_4383.json"
    assert data["game"] == "tn36"
    assert data["spec_refs"] == ["REQ-PHASE4-4383", "SCENARIO-PHASE4-4383"]
    assert data["mind_studio_source"] == "arXiv:2606.16070"
    assert data["prior_best_level"] == 7
    assert data["target_level"] == 8
    assert data["lookahead_k"] == 3
    assert data["random_seed"] == 4383
    assert data["entropy_selected_traces"] == ["tn36:entropy_rank0:L7->L8"]
    assert "sxhtkytekm" in data["target_mechanic_gap"]
    assert data["failure_mode"] == "oracle_grounded_or_leaked_mechanic_skill_file"


def test_req_phase4_4383_lookahead_fidelity_is_bare_and_prefix_sensitive() -> None:
    """REQ-PHASE4-4383: lookahead fidelity reports reproduced-prefix depth honestly."""

    assert exp.compute_lookahead_fidelity(reached_level=6, target_level=6, reproduced=True) == 1.0
    assert exp.compute_lookahead_fidelity(reached_level=5, target_level=6, reproduced=False) == 0.833333
    assert exp.compute_lookahead_fidelity(reached_level=-1, target_level=6, reproduced=False) == 0.0
    assert exp.compute_lookahead_fidelity(reached_level=10, target_level=6, reproduced=True) == 1.0
    assert exp.compute_lookahead_fidelity(reached_level=1, target_level=0, reproduced=True) == 0.0


def test_req_phase4_4383_attach_lookahead_fields_uses_skill_file(tmp_path: Path) -> None:
    """REQ-PHASE4-4383: scorecard rows carry skill-file and lookahead fields."""

    path = exp.write_skill_file(tmp_path, "lp85", random_seed=4383, lookahead_k=3)
    row = {
        "game": "lp85",
        "prior_best_level": 5,
        "new_reproduced_level": 5,
        "target_level": 6,
        "offline_reproduced": False,
        "reproduce_result": {"reached_level": 5, "reproduced": True},
    }

    out = exp.attach_lookahead_fields(row, skill_file_path=path, lookahead_k=3)

    assert out is row
    assert out["lookahead_fidelity"] == 0.833333
    assert out["lookahead_fidelity_per_round"] == [0.833333]
    assert out["mind_studio_skill_file"] == path
    assert out["entropy_selected_traces"] == ["lp85:entropy_rank0:L5->L6"]
    assert out["lookahead_status"] == "honest_partial_prefix_only"


def test_req_phase4_4383_checksum_binds_rows_paths_skills_seed_and_caps() -> None:
    """REQ-PHASE4-4383: checksum binds scorecard, paths, skill files, seed, and cap metadata."""

    rows = [
        _row("lp85"),
        _row("tu93", reached=5, accuracy=1.0, advanced=True),
        _row("tn36"),
        _row("tr87"),
    ]
    hashes = {"solver.py": "a" * 64, "results/arc_e3/tu93/skill_4383.json": "b" * 64}
    base = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4383,
        target_wall_time_s=1.5,
        lookahead_k=3,
    )
    same = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4383,
        target_wall_time_s=1.5,
        lookahead_k=3,
    )
    changed = exp.compute_reproducibility_checksum(
        per_target_scorecard=rows,
        world_model_paths=["solver.py"],
        path_hashes=hashes,
        random_seed=4383,
        target_wall_time_s=1.5,
        lookahead_k=4,
    )

    assert base == same
    assert base != changed
    assert len(base) == 64


def test_req_phase4_4383_build_artifact_counts_only_new_reproduced_levels(
    tmp_path: Path,
) -> None:
    """REQ-PHASE4-4383: only levels beyond prior best count as new progress."""

    solver = tmp_path / "python" / "carnot" / "agentic" / "arc_game_adapters.py"
    solver.parent.mkdir(parents=True)
    solver.write_text("# adapters\n", encoding="utf-8")

    rows = [
        _row("lp85", reached=6, accuracy=1.0, advanced=True),
        _row("tu93", accuracy=0.8),
        _row("tn36", accuracy=0.875),
        _row("tr87", accuracy=0.857),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=35,
        world_model_paths=[str(solver.relative_to(tmp_path))],
        random_seed=4383,
        target_wall_time_s=1.5,
        lookahead_k=3,
        duration_s=2.5,
    )

    assert artifact["honest_verdict"] == "success_e3_deeper_lp85_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert artifact["reproducible_total_levels"] == 35
    assert artifact["verifier_is_oracle"] is True
    assert artifact["target_wall_time_s"] == 1.5
    assert artifact["lookahead_k"] == 3
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    assert not exp.artifact_schema_errors(artifact)


def test_scenario_phase4_4383_partial_artifact_preserves_all_targets(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-4383: all-partial runs keep one row per target and bare gates."""

    rows = [
        _row("lp85", accuracy=0.833333),
        _row("tu93", accuracy=0.8),
        _row("tn36", accuracy=0.875),
        _row("tr87", accuracy=0.857143),
    ]
    artifact = exp.build_artifact(
        repo=tmp_path,
        per_target_scorecard=rows,
        reproducible_total_levels=34,
        world_model_paths=list(exp.WORLD_MODEL_PATHS.values()),
        random_seed=4383,
        target_wall_time_s=1.0,
        lookahead_k=3,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete_e3_deeper_partial"
    assert artifact["new_levels_reproduced"] == 0
    assert [row["game"] for row in artifact["per_target_scorecard"]] == list(exp.TARGET_ORDER)
    assert artifact["verifier_is_oracle"] is True
    assert isinstance(artifact["reproducible_total_levels"], int)
    assert isinstance(artifact["new_levels_reproduced"], int)
    assert all(isinstance(row["lookahead_fidelity"], float) for row in artifact["per_target_scorecard"])
    assert not exp.artifact_schema_errors(artifact)


def test_req_phase4_4383_schema_errors_are_specific() -> None:
    """REQ-PHASE4-4383: schema validation catches wrapped or malformed gate fields."""

    bad = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": "not-list",
        "reproducible_total_levels": {"value": 34},
        "new_levels_reproduced": "1",
        "world_model_paths": ["a.py"],
        "verifier_is_oracle": False,
        "preconditions_checked": {},
        "random_seed": 4383,
        "reproducibility_checksum": "short",
        "field_principles": {"honest_verdict": "wrong"},
        "target_wall_time_s": "1.0",
        "lookahead_k": "3",
    }

    errors = exp.artifact_schema_errors(bad)

    assert "per_target_scorecard must be list" in errors
    assert "reproducible_total_levels must be bare int" in errors
    assert "new_levels_reproduced must be bare int" in errors
    assert "verifier_is_oracle must be true" in errors
    assert "target_wall_time_s must be numeric" in errors
    assert "lookahead_k must be int" in errors
    assert "reproducibility_checksum must be 64-char sha256 hex" in errors
    assert "principle mismatch for honest_verdict" in errors

    missing = exp.artifact_schema_errors({"field_principles": None})
    assert "missing honest_verdict" in missing
    assert "field_principles missing" in missing


def test_req_phase4_4383_schema_validation_covers_row_shape_errors() -> None:
    """REQ-PHASE4-4383: malformed scorecard rows produce specific schema errors."""

    artifact = {
        "honest_verdict": "complete_e3_deeper_partial",
        "per_target_scorecard": ["bad-row", {"game": "lp85", "offline_reproduced": "yes"}],
        "reproducible_total_levels": 34,
        "new_levels_reproduced": 0,
        "world_model_paths": [123],
        "verifier_is_oracle": "true",
        "preconditions_checked": {},
        "random_seed": 4383,
        "target_wall_time_s": 1.0,
        "lookahead_k": 3,
        "reproducibility_checksum": "a" * 64,
        "field_principles": exp.REQUIRED_FIELD_PRINCIPLES,
    }

    errors = exp.artifact_schema_errors(artifact)

    assert "per_target_scorecard[0] must be dict" in errors
    assert "per_target_scorecard[1] missing prior_best_level" in errors
    assert "per_target_scorecard[1] missing lookahead_fidelity" in errors
    assert "per_target_scorecard[1].offline_reproduced must be bare bool" in errors
    assert "per_target_scorecard[1].lookahead_fidelity must be bare number" in errors
    assert "world_model_paths must be list[str]" in errors
    assert "verifier_is_oracle must be bare bool" in errors


def test_scenario_phase4_4383_run_experiment_records_missing_envs_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-PHASE4-4383: missing target envs block per target without fabrication."""

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

    artifact = exp.run_experiment(random_seed=4383, target_wall_time_s=None, lookahead_k=3)

    assert calls == ["tu93"]
    assert artifact["honest_verdict"] == "success_e3_deeper_tu93_reproduced"
    assert artifact["new_levels_reproduced"] == 1
    assert [row["checkpoint_status"] for row in artifact["per_target_scorecard"]] == [
        "blocked_offline_env_missing_lp85",
        "new_level_reproduced",
        "blocked_offline_env_missing_tn36",
        "blocked_offline_env_missing_tr87",
    ]
    assert (tmp_path / exp.RESULT_RELATIVE_PATH).exists()
    assert (tmp_path / "results" / "arc_e3" / "tu93" / "skill_4383.json").exists()


def test_req_phase4_4383_timeout_and_exception_rows_are_honest_partials() -> None:
    """REQ-PHASE4-4383: timeout and exception rows preserve no-progress state."""

    timeout = exp.timeout_target_row("lp85", target_wall_time_s=0.01)
    exception = exp.exception_target_row("tr87", "Traceback\nValueError: boom")

    assert timeout["game"] == "lp85"
    assert timeout["new_reproduced_level"] == exp.PRIOR_BEST_LEVELS["lp85"]
    assert timeout["offline_reproduced"] is False
    assert timeout["checkpoint_status"] == "honest_partial_wall_time_cap_exhausted"
    assert timeout["residual_win_mechanic_gap_class"] == "wall_time_cap_exhausted"
    assert exception["game"] == "tr87"
    assert exception["new_reproduced_level"] == exp.PRIOR_BEST_LEVELS["tr87"]
    assert exception["offline_reproduced"] is False
    assert exception["checkpoint_status"] == "honest_partial_target_exception"
    assert exception["reproduce_result"]["exception"] == "ValueError: boom"


def test_req_phase4_4383_adaptered_runner_accepts_only_reproduction_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: adaptered search hits are partials if reset replay rejects them."""

    def fake_solve_adaptered(game: str, target_level: int) -> dict:
        assert (game, target_level) == ("lp85", 6)
        return {
            "game": "lp85",
            "target": 6,
            "reached_level": 6,
            "moves": 66,
            "states_expanded": 962,
            "offline_reproduced": False,
            "solution_labels": ["a0"],
            "verifier_src": "hand_verifier_cold_start",
            "reproduction_gate": {
                "game": "lp85",
                "reached_level": 5,
                "claimed_level": 6,
                "reproduced": False,
            },
        }

    monkeypatch.setattr(exp, "_solve_adaptered", fake_solve_adaptered)

    row = exp._run_lp85_target(Path("."), 4383)

    assert row["new_reproduced_level"] == 5
    assert row["searched_level"] == 6
    assert row["offline_reproduced"] is False
    assert row["checkpoint_status"] == "honest_partial_no_new_level_reproduced"
    assert row["residual_win_mechanic_gap_class"] == "lp85_l6_permutation_bfs_no_new_reproduction_gap"


def test_req_phase4_4383_adaptered_target_wrappers_use_target_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: tu93 and tr87 wrappers route to their requested next levels."""

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

    tu93 = exp._run_tu93_target(Path("."), 4383)
    tr87 = exp._run_tr87_target(Path("."), 4383)

    assert calls == [("tu93", 5), ("tr87", 7)]
    assert tu93["residual_win_mechanic_gap_class"] == (
        "tu93_l5_fresh_env_branch_mode_no_new_reproduction_gap"
    )
    assert tr87["residual_win_mechanic_gap_class"] == (
        "tr87_l7_no_offline_level_available_or_no_new_reproduction_gap"
    )


def test_req_phase4_4383_tn36_runner_counts_only_new_reproduced_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: tn36 L8 counts only after the reproduction gate returns true."""

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

    row = exp._run_tn36_target(tmp_path, 4383)

    assert row["game"] == "tn36"
    assert row["prior_best_level"] == 7
    assert row["new_reproduced_level"] == 8
    assert row["offline_reproduced"] is True
    assert row["trajectory_action_count"] == 1


def test_req_phase4_4383_solve_adaptered_wrapper_imports_loop_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: adaptered games route through the reusable loop solver."""

    module = types.ModuleType("scripts.arc_loop_solve")
    module.solve_adaptered = lambda game, target_level: {  # type: ignore[attr-defined]
        "game": game,
        "target": target_level,
    }
    monkeypatch.setitem(sys.modules, "scripts.arc_loop_solve", module)

    assert exp._solve_adaptered("tr87", 7) == {"game": "tr87", "target": 7}


def test_req_phase4_4383_loader_registry_and_internal_schema_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: loader, registry parsing, and fail-closed schema stay deterministic."""

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
    registry.write_text("reproducible_total_levels: 34\n", encoding="utf-8")
    assert exp._registry_total(tmp_path) == 34
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
        "lookahead_fidelity_enabled": True,
        "trm_training_stood_down": True,
        "leaderboard_submission": False,
        "research_conductor_modified": False,
    }
    monkeypatch.setattr(exp, "REPO", tmp_path)
    monkeypatch.setattr(exp, "RESULT_PATH", tmp_path / exp.RESULT_RELATIVE_PATH)
    monkeypatch.setattr(exp, "preconditions", lambda _repo: checks)
    monkeypatch.setattr(exp, "artifact_schema_errors", lambda _artifact: ["forced"])

    with pytest.raises(ValueError, match="Exp4383 artifact schema errors"):
        exp.run_experiment(random_seed=4383, target_wall_time_s=None, lookahead_k=3)


def test_req_phase4_4383_precondition_failure_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: import and conductor-modified checks fail closed."""

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


def test_req_phase4_4383_target_worker_reports_success_and_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: worker payloads preserve either row or traceback."""

    class FakeQueue:
        def __init__(self) -> None:
            self.items: list[dict] = []

        def put(self, item: dict) -> None:
            self.items.append(item)

    ok_queue = FakeQueue()
    monkeypatch.setitem(exp.TARGET_RUNNERS, "lp85", lambda _repo, _seed: _row("lp85"))
    exp._target_worker("lp85", ".", 4383, ok_queue)  # type: ignore[arg-type]

    assert ok_queue.items[0]["ok"] is True
    assert ok_queue.items[0]["row"]["game"] == "lp85"

    bad_queue = FakeQueue()

    def bad_runner(_repo: Path, _seed: int) -> dict:
        raise ValueError("boom")

    monkeypatch.setitem(exp.TARGET_RUNNERS, "lp85", bad_runner)
    exp._target_worker("lp85", ".", 4383, bad_queue)  # type: ignore[arg-type]

    assert bad_queue.items[0]["ok"] is False
    assert "ValueError: boom" in bad_queue.items[0]["traceback"]


def test_req_phase4_4383_run_target_with_cap_covers_timeout_empty_and_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-PHASE4-4383: subprocess timeout/empty/success/error paths become honest rows."""

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
    row = exp._run_target_with_cap("lp85", Path("."), 4383, 0.01)
    assert row["checkpoint_status"] == "honest_partial_wall_time_cap_exhausted"

    FakeProcess.alive = False
    FakeQueue.empty = True
    row = exp._run_target_with_cap("lp85", Path("."), 4383, 0.01)
    assert row["checkpoint_status"] == "honest_partial_target_exception"
    assert row["reproduce_result"]["exception"] == "lp85 runner exited without result"

    FakeQueue.empty = False
    FakeQueue.payload = {"ok": True, "row": _row("lp85", reached=6, advanced=True)}
    row = exp._run_target_with_cap("lp85", Path("."), 4383, 0.01)
    assert row["offline_reproduced"] is True

    FakeQueue.payload = {"ok": False, "traceback": "RuntimeError: fail"}
    row = exp._run_target_with_cap("lp85", Path("."), 4383, 0.01)
    assert row["checkpoint_status"] == "honest_partial_target_exception"
    assert row["reproduce_result"]["exception"] == "RuntimeError: fail"

    def bad_runner(_repo: Path, _seed: int) -> dict:
        raise RuntimeError("direct fail")

    monkeypatch.setitem(exp.TARGET_RUNNERS, "lp85", bad_runner)
    row = exp._run_target_with_cap("lp85", Path("."), 4383, None)
    assert row["checkpoint_status"] == "honest_partial_target_exception"


def test_req_phase4_4383_path_hashes_cover_missing_and_present_files(tmp_path: Path) -> None:
    """REQ-PHASE4-4383: path hashes are deterministic for present files and empty for missing."""

    present = tmp_path / "solver.py"
    present.write_text("# solver\n", encoding="utf-8")

    hashes = exp._path_hashes(tmp_path, ["solver.py", "missing.py"])

    assert hashes["solver.py"] == hashlib.sha256(b"# solver\n").hexdigest()
    assert hashes["missing.py"] == ""
