"""Tests for REQ-ARC-WMTE-7491 timed E6 live-loop profiling."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot import experiment_7471_v654_arc_seam_observation as exp7471
from carnot import experiment_7491_e6_timed_live_profile as exp7491


class _Explorer:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def _candidates(self, _frame: Any, path: Any = None) -> list[dict[str, Any]]:
        self.calls.append(f"candidates:{len(path or [])}")
        return [{"action": 1, "data": None}]


class _Verifier:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def score(self, _engine: Any) -> str:
        self.calls.append("verify")
        return "verified"


class _Supervisor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def observe(self, _snapshot: Any) -> str:
        self.calls.append("supervise")
        return "no_redirect"


class _Policy:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.explorer = _Explorer(self.calls)
        self._decision_telemetry = exp7491.NOOP_RECORDER
        self.provenance = ["fresh_runtime_only"]

    def _call_plan_in_model(
        self, function: Any, engine: Any, goal: Any, grid: Any, **kwargs: Any
    ) -> Any:
        self.calls.append("planner")
        return function(engine, goal, grid, **kwargs)

    def _induce_and_plan_timed(self) -> None:
        self.calls.append("induce")
        recorder = self._decision_telemetry
        assert (
            recorder.time_world_model_verification(
                _Verifier(self.calls), object(), candidate_source="fixture"
            )
            == "verified"
        )
        assert (
            recorder.time_supervisor_selection(_Supervisor(self.calls), object()) == "no_redirect"
        )
        self._call_plan_in_model(
            lambda _engine, _goal, _grid: [1], object(), lambda _grid: False, object()
        )

    def next_move(self, _frames: Any, _latest: Any) -> tuple[int, None]:
        self.explorer._candidates(object(), path=[])
        self._induce_and_plan_timed()
        return 1, None


class _Environment:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, _move: Any) -> int:
        self.calls += 1
        return self.calls


def _drive(observer: exp7491.E6TimedObserver, steps: int = 3) -> dict[str, Any]:
    policy = _Policy()
    environment = _Environment()
    observer.install(policy)
    actions: list[tuple[int, None]] = []
    with observer.episode("complete"):
        for _ in range(steps):
            with observer.span("action_decision"):
                move = policy.next_move([], None)
                actions.append(move)
                with observer.span("environment_step", operation="step"):
                    environment.step(move)
    observer.finish("complete")
    return {
        "actions": actions,
        "policy_calls": policy.calls,
        "environment_calls": environment.calls,
        "provenance": policy.provenance,
        "rng": random.getstate(),
    }


def test_observer_extends_7471_and_defaults_off(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7491 imports 7471 and stays off by default."""

    path = tmp_path / "disabled.jsonl"
    observer = exp7491.E6TimedObserver("fixture", path)

    assert isinstance(observer, exp7471.E3SeamObserver)
    assert observer.enabled is False
    _drive(observer, steps=1)
    assert not path.exists()


def test_timers_preserve_actions_calls_environment_provenance_and_rng(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7491-PARITY proves timing is observation only."""

    random.seed(7491)
    off = _drive(exp7491.E6TimedObserver("off", tmp_path / "off.jsonl"))
    random.seed(7491)
    on_observer = exp7491.E6TimedObserver("on", tmp_path / "on.jsonl", enabled=True)
    on = _drive(on_observer)

    assert on == off
    assert on_observer.error_count == 0
    assert not (tmp_path / "off.jsonl").exists()
    assert (tmp_path / "on.jsonl").is_file()
    seams = {row["seam"] for row in exp7491.read_span_rows(tmp_path / "on.jsonl")}
    assert {
        "candidate_selection",
        "induction_and_generation",
        "world_model_verification",
        "supervisor",
        "planner",
        "environment_step",
    } <= seams


def test_nested_spans_reconcile_to_parent_time(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7491-RECONCILIATION accounts for every parent tick."""

    ticks = iter((0, 10, 20, 30, 40, 60, 90, 100))
    observer = exp7491.E6TimedObserver(
        "nested", tmp_path / "nested.jsonl", enabled=True, clock_ns=lambda: next(ticks)
    )
    with observer.episode("complete"):
        with observer.span("action_decision"):
            with observer.span("candidate_selection"):
                pass
            with observer.span("planner"):
                pass
    observer.finish("complete")

    rows = exp7491.read_span_rows(tmp_path / "nested.jsonl")
    reconciliation = exp7491.reconcile_spans(rows)
    assert reconciliation["passed"] is True
    assert all(
        row["duration_ns"] == row["direct_child_union_ns"] + row["gap_ns"]
        and row["gap_ns"] == row["exclusive_ns"]
        for row in reconciliation["rows"]
    )
    assert sum(row["exclusive_ns"] for row in rows) == 100
    assert all(row["start_monotonic_ns"] <= row["end_monotonic_ns"] for row in rows)
    required = {
        "episode_id",
        "decision_id",
        "parent_decision_id",
        "start_monotonic_ns",
        "end_monotonic_ns",
        "exclusive_ns",
        "concurrent",
        "terminal_disposition",
    }
    assert all(required <= set(row) for row in rows)


def test_backend_usage_joins_generation_span_by_request_id(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7491 joins backend usage without position-only attribution."""

    observer = exp7491.E6TimedObserver("tokens", tmp_path / "tokens.jsonl", enabled=True)
    with observer.episode("complete"):
        with observer.span("induction_and_generation", request_id="request-a"):
            pass
    observer.record_backend_usage(
        request_id="request-a",
        prompt_tokens=31,
        completion_tokens=17,
        total_tokens=48,
        usage_source="fixture_response",
    )
    observer.finish("complete")

    generation = next(
        row
        for row in exp7491.read_span_rows(tmp_path / "tokens.jsonl")
        if row["seam"] == "induction_and_generation"
    )
    assert generation["request_id"] == "request-a"
    assert generation["prompt_tokens"] == 31
    assert generation["completion_tokens"] == 17
    assert generation["total_tokens"] == 48
    assert generation["usage_source"] == "fixture_response"


def test_recorder_failure_is_caught_and_counted(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7491 keeps a failed writer outside policy behavior."""

    blocked_path = tmp_path / "spans.jsonl"
    blocked_path.mkdir()
    observer = exp7491.E6TimedObserver("failure", blocked_path, enabled=True)
    with observer.episode("complete"):
        with observer.span("candidate_selection"):
            pass
    observer.finish("complete")
    assert observer.error_count >= 1


def test_recorder_setup_failure_does_not_block_wrapped_work(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7491 fails open if a timer cannot start."""

    def broken_clock() -> int:
        raise RuntimeError("fixture timer unavailable")

    observer = exp7491.E6TimedObserver(
        "failure", tmp_path / "failed-clock.jsonl", enabled=True, clock_ns=broken_clock
    )
    work: list[str] = []
    with observer.span("candidate_selection"):
        work.append("completed")
    observer.finish("complete")

    assert work == ["completed"]
    assert observer.error_count >= 1


def test_frozen_panel_and_seeds_match_the_hash_rule() -> None:
    """SCENARIO-ARC-WMTE-7491-PANEL recovers the sealed 36 units."""

    frozen_path = exp7491.REPO_ROOT / exp7491.FROZEN_PANEL_PATH
    frozen = json.loads(frozen_path.read_text())
    schedule = exp7491.build_frozen_schedule(frozen["public_survey_games"])

    assert [row["game"] for row in frozen["selected_games"]] == list(exp7491.PANEL_GAMES)
    assert frozen["episode_seeds"] == list(exp7491.EPISODE_SEEDS)
    assert len(schedule) == 36
    assert schedule == frozen["schedule"]
    assert not set(exp7491.E4_GAMES) & set(exp7491.PANEL_GAMES)


def test_gpu_one_precondition_rejects_busy_or_wrong_device() -> None:
    """SCENARIO-ARC-WMTE-7491-PRECONDITIONS pins idle physical GPU 1."""

    idle = exp7491.select_gpu_one(
        [
            {"index": 0, "total_memory_mb": 24_000, "free_memory_mb": 24_000},
            {"index": 1, "total_memory_mb": 24_000, "free_memory_mb": 23_800},
        ]
    )
    busy = exp7491.select_gpu_one(
        [{"index": 1, "total_memory_mb": 24_000, "free_memory_mb": 23_400}]
    )

    assert idle is not None and idle["index"] == 1
    assert busy is None


def test_gpu_one_post_preflight_excludes_only_the_own_pid() -> None:
    """SCENARIO-ARC-WMTE-7491-PRECONDITIONS regresses the attempt-1 incident."""

    own_pid = 147_005
    own_cuda_init = exp7491.select_gpu_one(
        [
            {
                "index": 1,
                "total_memory_mb": 24_576,
                "free_memory_mb": 23_912,
                "compute_apps": [{"pid": own_pid, "used_memory_mb": 256}],
            }
        ],
        own_pid=own_pid,
    )
    foreign_cuda_job = exp7491.select_gpu_one(
        [
            {
                "index": 1,
                "total_memory_mb": 24_576,
                "free_memory_mb": 23_912,
                "compute_apps": [{"pid": 999_999, "used_memory_mb": 600}],
            }
        ],
        own_pid=own_pid,
    )

    assert own_cuda_init is not None
    assert own_cuda_init["used_memory_mb"] == 664
    assert own_cuda_init["own_process_memory_mb"] == 256
    assert own_cuda_init["residual_used_memory_mb"] == 408
    assert foreign_cuda_job is None


def test_blocked_artifact_is_terminal_and_makes_no_efficacy_claim(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7491-PRECONDITIONS writes an honest blocker."""

    artifact = exp7491.build_blocked_artifact(
        failed_check={"check": "gpu_1_idle", "passed": False, "observed": 600},
        duration_s=0.25,
        cited_artifacts=[],
        schedule=[
            {
                "episode_id": "sb26:seed-7491001",
                "game": "sb26",
                "seed": 7_491_001,
                "execution_order": 0,
            }
        ],
    )
    output = tmp_path / "blocked.json"
    exp7491.write_json(output, artifact)

    loaded = json.loads(output.read_text())
    assert loaded["honest_verdict"].startswith("blocked_")
    assert loaded["inference_substrate"] == "no_model_load"
    assert loaded["hidden_game_efficacy_claim"] is False
    assert loaded["rows"][0]["disposition"] == "unstarted"
    assert loaded["sample_size_budget"]["unstarted_units"] == 1
    assert list(tmp_path.rglob("*.json")) == [output]
