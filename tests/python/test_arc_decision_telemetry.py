"""Tests for REQ-ARC-WMTE-7465 ARC decision shadow telemetry."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from carnot.agentic import arc_action_provenance as provenance
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_decision_telemetry as telemetry
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot, TrajectorySupervisor


class _Frame:
    def __init__(self, grid: np.ndarray, level: int = 0) -> None:
        self.frame = [grid.tolist()]
        self.levels_completed = level
        self.state = "NOT_FINISHED"
        self.score = 0
        self.available_actions = [1, 2, 3, 4, 5, 6]


class _ScriptedEnvironment:
    def __init__(self) -> None:
        self.calls = 0

    def step(self) -> _Frame:
        self.calls += 1
        rng = np.random.RandomState(self.calls)
        grid = rng.randint(0, 4, size=(8, 8)).astype(int)
        return _Frame(grid)


class _CountingProposer:
    def __init__(self) -> None:
        self.calls = 0
        self.call_log: list[str] = []

    def induce(
        self, game: str, transitions: list[Any], cell: int, **_kwargs: Any
    ) -> tuple[bool, str]:
        self.calls += 1
        self.call_log.append(f"induce:{game}:{len(transitions)}:{cell}")
        return False, "unused"


def _drive(policy: E3AgentPolicy, n_actions: int) -> tuple[list[str], int]:
    env = _ScriptedEnvironment()
    frames: list[_Frame] = []
    latest = None
    actions: list[str] = []
    for _ in range(n_actions):
        kind, data = policy.next_move(frames, latest)
        actions.append(json.dumps({"action": kind, "data": data}, sort_keys=True))
        if kind is None:
            break
        latest = env.step()
        frames.append(latest)
    return actions, env.calls


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_keys(child)}
    return set()


def _clear_output_env(monkeypatch) -> None:
    monkeypatch.delenv(telemetry.TELEMETRY_PATH_ENV, raising=False)
    for name in telemetry.DEFAULT_OUTPUT_DIR_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_disabled_recorder_is_noop_and_writes_nothing(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7465-PARITY: the default recorder is inert."""

    path = tmp_path / "disabled.jsonl"
    monkeypatch.delenv(telemetry.TELEMETRY_ENV_FLAG, raising=False)
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))

    recorder = telemetry.maybe_make_recorder("xx11")
    recorder.begin_step(level_before=0, phase="explore")
    recorder.record_event("candidate_action_selection", {"option_count_total": 1}, 0.1)
    recorder.finish_episode(level_end=0, actions_used=1)

    assert recorder.enabled is False
    assert recorder.error_count == 0
    assert not path.exists()


def test_enabled_recorder_writes_valid_jsonl_and_summary(monkeypatch, tmp_path) -> None:
    """REQ-ARC-WMTE-7465 writes schema-valid rows and a pure summary."""

    path = tmp_path / "telemetry.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    recorder = telemetry.maybe_make_recorder("xx11-live")

    recorder.begin_step(level_before=2, phase="explore")
    recorder.record_candidate_action(
        [
            {"action": 6, "data": {"x": 3, "y": 4}, "score": 0.75},
            {"action": 1, "data": None, "score": 0.25},
        ],
        ranking_changed=False,
        wall_time_s=0.125,
    )
    recorder.record_event("induction_timing", {"decision": "continue_exploring"}, 0.25)
    recorder.finish_episode(level_end=3, actions_used=1)

    rows = _jsonl(path)
    assert [row["record_type"] for row in rows] == [
        "episode_start",
        "decision",
        "decision",
        "episode_end",
    ]
    assert all(row["schema_version"] == 1 for row in rows)
    assert all(row["game_id"] == "xx11-live" for row in rows)
    assert all(isinstance(row["monotonic_timestamp_s"], float) for row in rows)
    assert rows[1]["chosen_option_id"] == "action:6:x:3:y:4"

    summary = telemetry.summarize_telemetry(path)
    episode = summary["episodes"]["xx11-live"]
    assert summary["records_per_seam"] == {
        "candidate_action_selection": 1,
        "induction_timing": 1,
    }
    assert summary["option_count_distribution"] == {"2": 1}
    assert episode["wall_time_s_by_seam"]["candidate_action_selection"] == 0.125
    assert episode["total_wall_time_s"] == 0.375


def test_option_and_episode_caps_write_one_truncation_row(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7465-BOUNDS: options and episodes stay bounded."""

    path = tmp_path / "bounded.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    recorder = telemetry.maybe_make_recorder("xx11", max_records=4)
    recorder.begin_step(level_before=0, phase="explore")
    options = [
        {"action": 6, "data": {"x": index, "y": index + 1}, "score": index} for index in range(20)
    ]
    recorder.record_candidate_action(options, ranking_changed=True, wall_time_s=0.01)
    recorder.record_event("induction_timing", {"state_summary": "x" * 5000}, 0.02)
    recorder.record_event("supervisor_arm_selection", {"chosen_arm": "no_redirect"}, 0.03)
    recorder.finish_episode(level_end=0, actions_used=1)

    rows = _jsonl(path)
    candidate = next(row for row in rows if row.get("seam") == "candidate_action_selection")
    induction = next(row for row in rows if row.get("seam") == "induction_timing")
    truncated = [row for row in rows if row["record_type"] == "telemetry_truncated"]
    assert len(candidate["options"]) == telemetry.MAX_OPTIONS_PER_RECORD
    assert candidate["options_omitted"] == 5
    assert len(induction["state_summary"]) == telemetry.MAX_STATE_TEXT_CHARS
    assert len(truncated) == 1
    assert truncated[0]["reason"] == "episode_record_cap"
    assert len(rows) == 4


def test_run_byte_cap_writes_one_truncation_row(monkeypatch, tmp_path) -> None:
    """REQ-ARC-WMTE-7465 reserves space for a byte-cap marker."""

    path = tmp_path / "byte-bounded.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    recorder = telemetry.maybe_make_recorder("xx11", max_bytes=2048)
    recorder.begin_step(level_before=0, phase="explore")
    recorder.record_event(
        "induction_timing",
        {"state_summary": "x" * telemetry.MAX_STATE_TEXT_CHARS},
        0.02,
    )
    recorder.record_event("supervisor_arm_selection", {"chosen_arm": "no_redirect"}, 0.03)
    recorder.finish_episode(level_end=0, actions_used=1)

    rows = _jsonl(path)
    truncated = [row for row in rows if row["record_type"] == "telemetry_truncated"]
    assert len(truncated) == 1
    assert truncated[0]["reason"] == "run_byte_cap"
    assert path.stat().st_size <= 2048


def test_write_errors_are_swallowed_and_counted(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7465-FAILURE: writer failure cannot escape."""

    recorder = telemetry.DecisionTelemetryRecorder(
        "xx11",
        path=Path("/proc/definitely/not/writable/telemetry.jsonl"),
    )
    recorder.begin_step(level_before=0, phase="explore")
    recorder.record_event("induction_timing", {"decision": "induce"}, 0.1)
    recorder.finish_episode(level_end=0, actions_used=0)

    assert recorder.error_count >= 1
    assert not (tmp_path / "telemetry.jsonl").exists()


def test_candidate_metadata_errors_are_swallowed_and_counted(tmp_path) -> None:
    """REQ-ARC-WMTE-7465 keeps telemetry-only metadata fail-open."""

    recorder = telemetry.DecisionTelemetryRecorder("xx11", path=tmp_path / "safe.jsonl")

    class _Explorer:
        _decision_telemetry = recorder
        belief_candidate_selector = None

        @telemetry.capture_candidate_decision
        def candidates(self, *, path: Any) -> list[dict[str, Any]]:
            return [{"action": 1, "data": None}]

    expected = [{"action": 1, "data": None}]
    actual = _Explorer().candidates(path=object())
    recorder.finish_episode(level_end=0, actions_used=0)

    assert actual == expected
    assert recorder.error_count == 1


def test_no_game_source_or_hidden_keys_are_serialized(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7465-FAILURE: forbidden input fields are removed."""

    path = tmp_path / "safe.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    recorder = telemetry.maybe_make_recorder("xx11")
    recorder.begin_step(level_before=0, phase="verify")
    recorder.record_event(
        "world_model_hypothesis_gate",
        {
            "candidate_ids": ["candidate-a"],
            "game_source": "forbidden",
            "nested": {
                "hidden_information": "forbidden",
                "future_frame": [[9]],
                "adapter_data": {"name": "forbidden"},
            },
        },
        0.2,
    )
    recorder.finish_episode(level_end=0, actions_used=1)

    keys = _all_keys(_jsonl(path))
    assert keys.isdisjoint(telemetry.FORBIDDEN_KEYS)
    assert "candidate_ids" in keys


def test_missing_default_output_refuses_enable_with_one_warning(monkeypatch, caplog) -> None:
    """REQ-ARC-WMTE-7465 refuses an unsafe implicit output path."""

    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    _clear_output_env(monkeypatch)
    telemetry.reset_warning_state_for_tests()

    first = telemetry.maybe_make_recorder("xx11")
    second = telemetry.maybe_make_recorder("yy22")

    messages = [
        record.message for record in caplog.records if "decision telemetry" in record.message
    ]
    assert first.enabled is False
    assert second.enabled is False
    assert len(messages) == 1


def test_world_gate_and_supervisor_rows_keep_decision_fields(monkeypatch, tmp_path) -> None:
    """REQ-ARC-WMTE-7465 records both non-action decision seams."""

    path = tmp_path / "decision-seams.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    recorder = telemetry.maybe_make_recorder("xx11")
    recorder.begin_step(level_before=1, phase="induce")
    candidate = SimpleNamespace(name="candidate-a")
    score = SimpleNamespace(
        candidate=candidate,
        change_gate={"legacy_accuracy": 0.75, "cell_recall": 0.8, "change_accuracy": 0.7},
        heldout_accuracy=0.75,
        trust_energy=0.2,
        baseline_clears=True,
        heldout_best=True,
        nondegenerate=True,
        trust_pass=True,
        binary_gate_pass=True,
    )
    selection = SimpleNamespace(rows=[score], selected=candidate, selected_score=score)
    selector_calls: list[str] = []

    def selector(_transitions, _candidates, **_kwargs):
        selector_calls.append("select")
        return selection

    observed = recorder.time_world_model_selection(
        selector,
        [],
        [candidate],
        acceptance_threshold=0.5,
    )
    supervisor = TrajectorySupervisor(window=1)
    redirect = recorder.time_supervisor_selection(
        supervisor,
        TrajectorySnapshot(
            level=1,
            goal_bias_installed=True,
            induced=False,
            induction_attempts=0,
            new_transitions_since_induction=0,
            diversity_active=False,
        ),
    )
    policy = SimpleNamespace(
        proposer=SimpleNamespace(last_generated_tokens=17, last_prompt_tokens=31)
    )
    recorder.complete_induction(policy, {"planned": True}, 0.5)
    recorder.finish_episode(level_end=1, actions_used=1)

    rows = _jsonl(path)
    gate = next(row for row in rows if row.get("seam") == "world_model_hypothesis_gate")
    arm = next(row for row in rows if row.get("seam") == "supervisor_arm_selection")
    induction = next(
        row
        for row in rows
        if row.get("seam") == "induction_timing" and row.get("decision") == "induction_call"
    )
    assert observed is selection
    assert selector_calls == ["select"]
    assert gate["candidate_ids"] == ["candidate-a"]
    assert gate["candidates"][0]["exact_accuracy"] == 0.75
    assert gate["outcome"] == "accept"
    assert gate["plan_found"] is True
    assert redirect.arm == "drop_goal_bias"
    assert arm["chosen_arm"] == "drop_goal_bias"
    assert arm["trajectory_snapshot"]["level"] == 1
    assert induction["generated_tokens"] == 17


def test_telemetry_on_off_preserves_actions_provenance_calls_and_rng(monkeypatch, tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7465-PARITY uses the real policy with a fake environment."""

    monkeypatch.delenv("CARNOT_ARC_DISABLE_INDUCTION", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv(provenance.PROVENANCE_ENV_FLAG, "1")
    monkeypatch.setenv(provenance.PROVENANCE_DIR_ENV, str(tmp_path / "provenance"))
    monkeypatch.delenv(telemetry.TELEMETRY_ENV_FLAG, raising=False)
    monkeypatch.delenv(telemetry.TELEMETRY_PATH_ENV, raising=False)

    random.seed(734)
    np.random.seed(734)
    proposer_off = _CountingProposer()
    policy_off = E3AgentPolicy("xx11", proposer=proposer_off, explore_budget=1)
    actions_off, env_calls_off = _drive(policy_off, 40)
    provenance_off = policy_off.action_provenance().to_dict()
    rng_off = (
        policy_off.explorer._fd_rng.getstate(),
        policy_off.explorer._cps_rng.getstate(),
        policy_off.explorer._div_rng.getstate(),
        random.random(),
        float(np.random.random()),
    )

    path = tmp_path / "enabled.jsonl"
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(path))
    random.seed(734)
    np.random.seed(734)
    proposer_on = _CountingProposer()
    policy_on = E3AgentPolicy("xx11", proposer=proposer_on, explore_budget=1)
    actions_on, env_calls_on = _drive(policy_on, 40)
    provenance_on = policy_on.action_provenance().to_dict()
    rng_on = (
        policy_on.explorer._fd_rng.getstate(),
        policy_on.explorer._cps_rng.getstate(),
        policy_on.explorer._div_rng.getstate(),
        random.random(),
        float(np.random.random()),
    )
    policy_on.finish_decision_telemetry(level_end=0, actions_used=len(actions_on))
    telemetry_rows = _jsonl(path)

    assert actions_on == actions_off
    assert provenance_on == provenance_off
    assert proposer_on.call_log == proposer_off.call_log
    assert proposer_on.calls == proposer_off.calls == len(proposer_on.call_log) > 0
    assert env_calls_on == env_calls_off
    assert rng_on == rng_off
    assert path.exists()
    assert telemetry_rows[0]["record_type"] == "episode_start"
    assert telemetry_rows[0]["level_start"] == 0
    assert telemetry_rows[-1]["record_type"] == "episode_end"
    assert telemetry_rows[-1]["actions_used"] == len(actions_on)
    candidate_row = next(
        row for row in telemetry_rows if row.get("seam") == "candidate_action_selection"
    )
    assert set(candidate_row["state"]) == {
        "best_level",
        "explored_out",
        "path_depth",
        "steps_since_progress",
    }
    assert len(candidate_row["state_summary"]) <= telemetry.MAX_STATE_TEXT_CHARS


def test_module_is_in_live_entrypoint_import_closure() -> None:
    """REQ-ARC-WMTE-7465 keeps the recorder reachable from the scored entrypoint."""

    assert agent.arc_decision_telemetry is telemetry
    assert telemetry.__name__ in {
        module.__name__ for module in vars(agent).values() if hasattr(module, "__name__")
    }


def _fire_induction(recorder: telemetry.DecisionTelemetryRecorder) -> SimpleNamespace:
    policy = SimpleNamespace(
        explorer=SimpleNamespace(explored_out=True),
        transitions=[],
        induction_attempts=[],
        induced=False,
        _current_goal_level=1,
        _induction_attempt_count=0,
        _transitions_at_last_induction_attempt=0,
        proposer=SimpleNamespace(last_generated_tokens=17, last_prompt_tokens=31),
    )
    recorder.record_induction_decision(
        policy,
        stalled=True,
        won=False,
        decision=(True, "stall"),
        wall_time_s=0.001,
    )
    return policy


def _selection(name: str, accuracy: float) -> SimpleNamespace:
    candidate = SimpleNamespace(name=name)
    score = SimpleNamespace(
        candidate=candidate,
        change_gate={"legacy_accuracy": accuracy},
        heldout_accuracy=accuracy,
        trust_energy=0.2,
        baseline_clears=True,
        heldout_best=True,
        nondegenerate=True,
        trust_pass=accuracy >= 0.5,
        binary_gate_pass=accuracy >= 0.5,
    )
    return SimpleNamespace(rows=[score], selected=candidate, selected_score=score)


def test_fired_attempt_joins_tokens_verifier_and_frame_progress(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7530-ATTEMPT-OUTCOME joins one complete attempt row."""

    path = tmp_path / "attempt.jsonl"
    recorder = telemetry.DecisionTelemetryRecorder("xx11", path=path)
    recorder.begin_step(level_before=0, phase="explore")
    policy = _fire_induction(recorder)

    outcomes = iter((_selection("first", 0.2), _selection("second", 0.8)))
    for _ in range(2):
        recorder.time_world_model_selection(
            lambda _transitions, _candidates: next(outcomes),
            [],
            [],
            acceptance_threshold=0.5,
        )
    recorder.complete_induction(
        policy,
        {"reason": "stall", "planned": True, "transition_count": 4},
        0.5,
    )
    policy.transitions.append(
        SimpleNamespace(
            grid=np.zeros((2, 2), dtype=int),
            next_grid=np.ones((2, 2), dtype=int),
            level_before=0,
            level_after=0,
        )
    )
    recorder.begin_policy_step(policy, SimpleNamespace(levels_completed=0))
    recorder.observe_induction_progress(policy, SimpleNamespace(levels_completed=0))
    recorder.finish_episode(level_end=0, actions_used=1)

    rows = _jsonl(path)
    opportunity = next(
        row
        for row in rows
        if row.get("seam") == "induction_timing" and row.get("gate_decision") == "induce_now"
    )
    outcome = next(row for row in rows if row.get("record_type") == "induction_attempt")
    gates = [row for row in rows if row.get("seam") == "world_model_hypothesis_gate"]

    assert opportunity["attempt_id"] == outcome["attempt_id"]
    assert {row["attempt_id"] for row in gates} == {outcome["attempt_id"]}
    assert outcome["prompt_tokens"] == 31
    assert outcome["completion_tokens"] == 17
    assert outcome["induction_wall_time_s"] == 0.5
    assert outcome["planned"] is True
    assert outcome["verifier_results"] == ["escalate", "accept"]
    assert outcome["verifier_result"] == "accept"
    assert outcome["frame_change_progress"] is True
    assert outcome["level_up_progress"] is False
    assert outcome["progress_within_window"] is True
    assert outcome["progress_window_actions"] == telemetry.INDUCTION_PROGRESS_WINDOW_ACTIONS
    assert outcome["progress_actions_observed"] == 1
    assert outcome["progress_window_censored"] is False


def test_attempt_without_progress_closes_at_fixed_action_window(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7530 fixes non-progress attribution at 32 actions."""

    path = tmp_path / "window.jsonl"
    recorder = telemetry.DecisionTelemetryRecorder("xx11", path=path)
    recorder.begin_step(level_before=0, phase="explore")
    policy = _fire_induction(recorder)
    recorder.complete_induction(policy, {"reason": "stall", "planned": False}, 0.25)

    latest = SimpleNamespace(levels_completed=0)
    for _ in range(telemetry.INDUCTION_PROGRESS_WINDOW_ACTIONS):
        recorder.begin_policy_step(policy, latest)
        recorder.observe_induction_progress(policy, latest)
    recorder.finish_episode(level_end=0)

    outcome = next(row for row in _jsonl(path) if row.get("record_type") == "induction_attempt")
    assert outcome["progress_actions_observed"] == 32
    assert outcome["frame_change_progress"] is False
    assert outcome["level_up_progress"] is False
    assert outcome["progress_within_window"] is False
    assert outcome["progress_window_censored"] is False
    assert outcome["verifier_result"] == "not_observed"


def test_episode_end_censors_an_unfinished_progress_window(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7530-ATTEMPT-OUTCOME keeps censored fired attempts."""

    path = tmp_path / "censored.jsonl"
    recorder = telemetry.DecisionTelemetryRecorder("xx11", path=path)
    recorder.begin_step(level_before=0, phase="explore")
    policy = _fire_induction(recorder)
    recorder.complete_induction(policy, {"reason": "stall", "planned": False}, 0.25)
    latest = SimpleNamespace(levels_completed=0)
    for _ in range(3):
        recorder.begin_policy_step(policy, latest)
        recorder.observe_induction_progress(policy, latest)
    recorder.finish_episode(level_end=0, actions_used=3)

    outcome = next(row for row in _jsonl(path) if row.get("record_type") == "induction_attempt")
    assert outcome["progress_actions_observed"] == 3
    assert outcome["progress_within_window"] is False
    assert outcome["progress_window_censored"] is True


def test_extended_noop_hook_is_byte_inert() -> None:
    """SCENARIO-ARC-WMTE-7530-PARITY keeps the disabled hook byte-inert."""

    policy = SimpleNamespace(
        transitions=[{"sentinel": "unchanged"}],
        marker={"nested": [1, 2, 3]},
    )
    before = json.dumps(policy.__dict__, sort_keys=True)
    telemetry.NOOP_RECORDER.observe_induction_progress(
        policy,
        SimpleNamespace(levels_completed=9),
    )
    after = json.dumps(policy.__dict__, sort_keys=True)

    assert before == after
