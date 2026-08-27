"""Tests for REQ-ARC-WMTE-6681 and its exact live-outcome scenarios."""

from __future__ import annotations

import copy
from enum import Enum
import importlib.metadata
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
from carnot.agentic.arc_trajectory_supervisor import TraceAutomatonSupervisor
from carnot.agentic.arc_e3_outcome_transport import (
    E3OutcomeTransport,
    OutcomeLineageError,
    join_outcome_events,
    normalize_action,
    normalize_observation,
    observation_hash,
    run_lineage_attacks,
)
from carnot.agentic import arc_e3_outcome_transport as transport_module
from carnot import experiment_6681_arc_post_redirect_outcomes as exp


class _Frame:
    """Small observation with the fields returned by the ARC SDK."""

    def __init__(self, value: int, *, level: int = 0, state: str = "NOT_FINISHED") -> None:
        self.game_id = "held-a-live"
        self.frame = [np.full((4, 4), value, dtype=np.int8).tolist()]
        self.state = state
        self.levels_completed = level
        self.win_levels = 1
        self.guid = "live-guid"
        self.full_reset = False
        self.available_actions = [1, 2, 3, 4, 5, 6]


def _fsm() -> dict:
    return {
        "schema": "carnot.arc.trace_fsm.v1",
        "states": ["bootstrap", "productive", "observing", "stagnant_repeat"],
        "initial_state": "bootstrap",
        "features": [
            "previous_frame_changed",
            "same_action_run",
            "actions_since_observed_change",
            "level_progress_since_previous_action",
            "action_role_is_overhead",
            "consecutive_navigation_or_replay",
        ],
        "thresholds": {
            "same_action_run": 2,
            "actions_since_observed_change": 1,
            "consecutive_navigation_or_replay": 99,
        },
        "transitions": [],
        "redirect_arms": ["reset_after_stagnant_repeat"],
        "tie_rules": ["single_eligible_arm"],
        "training_support_actions": 10,
        "training_family_count": 2,
        "frozen_before_held_evaluation": True,
    }


def _one_transport(*, redirect: bool, index: int = 0, tuple_return: bool = False):
    transport = E3OutcomeTransport(
        family="held-a",
        attempt=index,
        episode_seed=6681000 + index,
        episode_id=f"held-a-attempt-{index}",
    )
    before = _Frame(index % 5)
    after = _Frame((index + 1) % 5, level=1 if index % 7 == 0 else 0)
    proposed = {"kind": 1, "data": None}
    selected = {"kind": "RESET", "data": None} if redirect else proposed
    transport.record_proposal(
        proposed_action=proposed,
        policy_selected_action=selected,
        observation_before=before,
        supervisor_decision={
            "fired": redirect,
            "state": "stagnant_repeat" if redirect else "observing",
            "arm": "reset_after_stagnant_repeat" if redirect else None,
        },
    )
    transport.record_application(selected)
    step_id = transport.begin_environment_step(selected)
    returned = (after, 0.25, False) if tuple_return else after
    transport.record_environment_return(step_id, returned)
    return transport


def _event_bundle(*, redirects: int = 30, controls: int = 30) -> dict:
    bundles = [
        _one_transport(redirect=True, index=index).events() for index in range(redirects)
    ] + [_one_transport(redirect=False, index=1000 + index).events() for index in range(controls)]
    return exp.merge_episode_events(bundles)


def test_scenario_6681_lineage_joins_after_reordering():
    """SCENARIO-ARC-WMTE-6681-LINEAGE uses identities, not event position."""

    events = _one_transport(redirect=True).events()
    expected_rows, expected_audit = join_outcome_events(events)
    reordered = {key: list(reversed(value)) for key, value in events.items()}
    actual_rows, actual_audit = join_outcome_events(reordered)

    assert actual_rows == expected_rows
    assert actual_audit == expected_audit
    row = actual_rows[0]
    assert row["lineage"] == {
        "proposal_id": row["proposal_id"],
        "application_id": row["application_id"],
        "environment_step_id": row["environment_step_id"],
        "outcome_id": row["outcome_id"],
    }


def test_scenario_6681_exact_return_keeps_absent_and_present_rewards():
    """SCENARIO-ARC-WMTE-6681-EXACT-RETURN never invents an ARC reward."""

    absent_rows, absent_audit = join_outcome_events(_one_transport(redirect=True, index=1).events())
    present_rows, present_audit = join_outcome_events(
        _one_transport(redirect=True, index=2, tuple_return=True).events()
    )

    assert absent_audit["ready"] is True
    assert absent_rows[0]["reward"] == {
        "present": False,
        "value": None,
        "source": "arc_agi.FrameDataRaw.step_return_schema",
        "synthetic": False,
    }
    assert present_audit["ready"] is True
    assert present_rows[0]["reward"] == {
        "present": True,
        "value": 0.25,
        "source": "environment_step_return[1]",
        "synthetic": False,
    }
    assert present_rows[0]["termination"]["terminated"] is False
    assert absent_rows[0]["observation_after"]["levels_completed"] == 0


def test_scenario_6681_canonical_step_preserves_raw_return_before_conversion(monkeypatch):
    """SCENARIO-ARC-WMTE-6681-EXACT-RETURN records raw SDK-only fields."""

    class _RawFrame(_Frame):
        def __init__(self) -> None:
            super().__init__(7)
            self.action_input = {"id": 1, "data": {"x": 2, "y": 3}}

    class _Env:
        def step(self, action, data=None, reasoning=None):
            del action, data, reasoning
            return _RawFrame()

    class _Base:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.game_id = kwargs["game_id"]
            self.arc_env = kwargs["arc_env"]

        def _convert_raw_frame_data(self, raw):
            return _Frame(raw.frame[0][0][0], level=raw.levels_completed, state=raw.state)

        def do_action_request(self, action):
            data = action.action_data.model_dump()
            raw = self.arc_env.step(action, data=data, reasoning=None)
            return self._convert_raw_frame_data(raw)

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    agent = make_carnot_agent(_Base)(game_id="held-a-live", arc_env=_Env())
    transport = E3OutcomeTransport(
        family="held-a", attempt=8, episode_seed=6681008, episode_id="raw-return"
    )
    agent._policy.install_outcome_transport(transport)
    agent._policy.plan = [{"action": 1, "data": None}]
    agent._policy.phase = "execute"
    agent._policy.induced = True

    action = agent.choose_action([_Frame(0)], _Frame(0))
    converted = agent.do_action_request(action)
    rows, _audit = join_outcome_events(transport.events())

    assert not hasattr(converted, "action_input")
    assert rows[0]["observation_after"]["action_input"] == {
        "id": 1,
        "data": {"x": 2, "y": 3},
    }


def test_scenario_6681_missing_outcome_and_duplicate_ids_fail_closed():
    """SCENARIO-ARC-WMTE-6681-MISSING-OUTCOME rejects ambiguous children."""

    events = _one_transport(redirect=True).events()
    missing = copy.deepcopy(events)
    missing["outcomes"].clear()
    rows, audit = join_outcome_events(missing)
    assert rows == []
    assert audit["ready"] is False
    assert audit["issues"][0]["reason"] == "outcome_child_count"

    duplicated = copy.deepcopy(events)
    duplicated["outcomes"].append(copy.deepcopy(duplicated["outcomes"][0]))
    with pytest.raises(OutcomeLineageError, match="duplicate outcome_id"):
        join_outcome_events(duplicated)

    second_child = copy.deepcopy(events)
    extra_outcome = copy.deepcopy(second_child["outcomes"][0])
    extra_outcome["outcome_id"] = exp.sha256_json({"second_child": extra_outcome})
    second_child["outcomes"].append(extra_outcome)
    rows, audit = join_outcome_events(second_child)
    assert rows == []
    assert audit["ready"] is False
    assert audit["issues"][0]["observed"] == 2


@pytest.mark.parametrize(
    ("table", "parent_key"),
    [
        ("applications", "proposal_id"),
        ("environment_steps", "application_id"),
        ("outcomes", "environment_step_id"),
    ],
)
def test_scenario_6681_orphan_children_fail_closed(table, parent_key):
    """SCENARIO-ARC-WMTE-6681-MISSING-OUTCOME rejects every missing parent."""

    events = _one_transport(redirect=True).events()
    orphan = copy.deepcopy(events[table][0])
    orphan[parent_key] = "sha256:missing-parent"
    identity_key = {
        "applications": "application_id",
        "environment_steps": "environment_step_id",
        "outcomes": "outcome_id",
    }[table]
    orphan[identity_key] = exp.sha256_json({"orphan": table})
    events[table].append(orphan)

    _rows, audit = join_outcome_events(events)

    assert audit["ready"] is False
    assert any(issue["reason"] == f"orphan_{table.removesuffix('s')}" for issue in audit["issues"])


def test_scenario_6681_attacks_cover_required_ambiguities():
    """SCENARIO-ARC-WMTE-6681-ATTACKS exercises the full fail-closed matrix."""

    events = _one_transport(redirect=True).events()
    rows = run_lineage_attacks(events)

    assert {row["attack_id"] for row in rows} == set(exp.ATTACK_IDS)
    assert all(row["passed"] for row in rows)
    reordered = next(row for row in rows if row["attack_id"] == "reordered_events")
    assert reordered["observed"] == "joined_by_identity"
    for attack_id in set(exp.ATTACK_IDS) - {"reordered_events"}:
        row = next(item for item in rows if item["attack_id"] == attack_id)
        assert row["observed"] == "rejected"


def test_scenario_6681_canonical_agent_step_joins_redirect_and_control(monkeypatch):
    """SCENARIO-ARC-WMTE-6681-CONTROLS-AND-TAGS reaches the factory step seam."""

    GameAction = pytest.importorskip("arcengine").GameAction

    class _Env:
        def __init__(self) -> None:
            self.value = 0

        def step(self, action, data=None, reasoning=None):
            del action, data, reasoning
            return _Frame(self.value)

    class _Base:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.game_id = kwargs["game_id"]
            self.arc_env = kwargs["arc_env"]

        def do_action_request(self, action):
            data = action.action_data.model_dump()
            raw = self.arc_env.step(action, data=data, reasoning=None)
            return self._convert_raw_frame_data(raw)

        def _convert_raw_frame_data(self, raw):
            return raw

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    agent_cls = make_carnot_agent(_Base)
    agent = agent_cls(game_id="held-a-live", arc_env=_Env())
    transport = E3OutcomeTransport(
        family="held-a", attempt=3, episode_seed=6681003, episode_id="canonical-live"
    )
    agent._policy.install_trace_automaton_supervisor(TraceAutomatonSupervisor(_fsm()))
    agent._policy.install_outcome_transport(transport)
    agent._policy.plan = [{"action": 1, "data": None}, {"action": 1, "data": None}]
    agent._policy.phase = "execute"
    agent._policy.induced = True
    frame = _Frame(0)

    first = agent.choose_action([frame], frame)
    assert first is GameAction.ACTION1
    agent.do_action_request(first)
    second = agent.choose_action([frame], frame)
    assert second is GameAction.RESET
    agent.do_action_request(second)

    rows, audit = join_outcome_events(transport.events())
    assert audit["ready"] is True
    assert [row["redirect_applied"] for row in rows] == [False, True]
    assert all(row["family"] == "held-a" and row["attempt"] == 3 for row in rows)
    assert rows[1]["applied_action"]["kind"] == "RESET"
    assert rows[1]["redirect_reason"] == "reset_after_stagnant_repeat"


def test_scenario_6681_step_timeout_and_error_are_terminal_rows():
    """REQ-ARC-WMTE-6681 keeps failed live returns visible and non-eligible."""

    timeout = E3OutcomeTransport(family="held-a", attempt=1, episode_seed=1, episode_id="timeout")
    timeout.record_proposal(
        proposed_action={"kind": 1, "data": None},
        policy_selected_action={"kind": "RESET", "data": None},
        observation_before=_Frame(0),
        supervisor_decision={"fired": True, "arm": "reset_after_stagnant_repeat"},
    )
    timeout.record_application({"kind": "RESET", "data": None})
    step_id = timeout.begin_environment_step({"kind": "RESET", "data": None})
    timeout.record_environment_failure(step_id, status="timeout", error="ten seconds")
    rows, audit = join_outcome_events(timeout.events())
    assert rows[0]["outcome_status"] == "timeout"
    assert audit["ready"] is False

    error = copy.deepcopy(timeout.events())
    error["outcomes"][0]["status"] = "environment_error"
    error["outcomes"][0]["error"] = "gateway unavailable"
    error["outcomes"][0]["outcome_id"] = exp.sha256_json(error["outcomes"][0])
    rows, audit = join_outcome_events(error)
    assert rows[0]["outcome_status"] == "environment_error"
    assert audit["ready"] is False


@pytest.mark.parametrize(
    ("failure", "status"),
    [(TimeoutError("late"), "timeout"), (RuntimeError("broken"), "environment_error")],
)
def test_scenario_6681_canonical_step_records_failure_before_reraise(monkeypatch, failure, status):
    """SCENARIO-ARC-WMTE-6681-ATTACKS reaches canonical step exceptions."""

    class _Env:
        def step(self, action, data=None, reasoning=None):
            del action, data, reasoning
            raise failure

    class _Base:
        def __init__(self, *args, **kwargs) -> None:
            del args
            self.game_id = kwargs["game_id"]
            self.arc_env = kwargs["arc_env"]

        def do_action_request(self, action):
            data = action.action_data.model_dump()
            return self.arc_env.step(action, data=data, reasoning=None)

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    agent = make_carnot_agent(_Base)(game_id="held-a-live", arc_env=_Env())
    transport = E3OutcomeTransport(
        family="held-a", attempt=4, episode_seed=6681004, episode_id=f"failure-{status}"
    )
    agent._policy.install_outcome_transport(transport)
    agent._policy.plan = [{"action": 1, "data": None}]
    agent._policy.phase = "execute"
    agent._policy.induced = True
    frame = _Frame(0)
    action = agent.choose_action([frame], frame)

    with pytest.raises(type(failure), match=str(failure)):
        agent.do_action_request(action)

    rows, audit = join_outcome_events(transport.events())
    assert rows[0]["outcome_status"] == status
    assert audit["ready"] is False


def test_scenario_6681_artifact_recomputes_ready_rows_and_no_solve(tmp_path):
    """SCENARIO-ARC-WMTE-6681-ARTIFACT and NO-SOLVE bind all raw evidence."""

    artifact = exp.build_artifact(
        episode_events=_event_bundle(),
        write=False,
        duration_s=1.25,
        run_date="20260827",
    )

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["arc_outcome_transport_ready"] is True
    assert artifact["eligible_redirect_outcome_rows"] == 30
    assert len(artifact["redirect_outcome_rows"]) == 30
    assert len(artifact["non_redirect_control_rows"]) == 30
    assert artifact["solve_claim_scope"] == "none"
    assert artifact["random_seed"]["online_environment_seed_effective"] is False
    assert artifact["canonical_path_receipt"]["environment_step_seam"] == (
        "CarnotAgent.do_action_request->arc_env.step(raw)->Agent._convert_raw_frame_data"
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["aggregate_row_recomputation"] == exp.recompute_aggregate_rows(
        artifact["redirect_outcome_rows"],
        artifact["non_redirect_control_rows"],
        artifact["lineage_attack_rows"],
    )
    assert exp.validate_artifact(artifact) == []

    output = tmp_path / "exp6681.json"
    assert (
        exp.main(
            ["--date", "20260827", "--result-path", str(output)], episode_events=_event_bundle()
        )
        == 0
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert exp.validate_artifact(saved) == []
    assert exp.main(["--validate", "--result-path", str(output)]) == 0


def test_scenario_6681_artifact_blocks_missing_redirect_and_validator_mutations():
    """REQ-ARC-WMTE-6681 derives blockers from rows and validates every gate."""

    events = _event_bundle(redirects=1, controls=1)
    events["outcomes"].pop(0)
    artifact = exp.build_artifact(
        episode_events=events,
        write=False,
        duration_s=0.5,
    )
    assert artifact["arc_outcome_transport_ready"] is False
    assert artifact["eligible_redirect_outcome_rows"] == 0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["failed_check"] == "redirect_exact_outcome_join"

    ready = exp.build_artifact(episode_events=_event_bundle(), write=False, duration_s=0.5)
    mutations = [
        (lambda row: row.pop("status"), "required fields mismatch"),
        (lambda row: row.__setitem__("status", "running"), "status lacks terminal prefix"),
        (lambda row: row.__setitem__("verdict_class", "positive"), "verdict class invalid"),
        (lambda row: row.__setitem__("inference_substrate", "wrong"), "substrate mismatch"),
        (lambda row: row.__setitem__("verifier_is_oracle", False), "oracle flag mismatch"),
        (lambda row: row.__setitem__("solve_claim_scope", "level"), "solve scope mismatch"),
        (
            lambda row: row.__setitem__("eligible_redirect_outcome_rows", 29),
            "eligible redirect count mismatch",
        ),
        (
            lambda row: row["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
            "protected files changed",
        ),
    ]
    for mutate, expected in mutations:
        changed = copy.deepcopy(ready)
        mutate(changed)
        assert expected in exp.validate_artifact(changed)


def test_scenario_6681_helpers_handle_atomic_paths_and_live_runner_failure(tmp_path, monkeypatch):
    """REQ-ARC-WMTE-6681 covers atomic output and named live access blocks."""

    assert exp.sha256_file(tmp_path / "missing") == "missing"
    output = exp.write_artifact_json(tmp_path / "nested" / "result.json", {"ok": True})
    assert output.is_file()

    monkeypatch.setattr(
        exp,
        "run_live_held_family_episodes",
        lambda **kwargs: (
            {"proposals": [], "applications": [], "environment_steps": [], "outcomes": []},
            {"error": "network unavailable"},
        ),
    )
    artifact = exp.build_artifact(write=False, duration_s=0.1)
    assert artifact["arc_outcome_transport_ready"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "live_access"

    def raise_live_path_error(**kwargs):
        del kwargs
        raise RuntimeError("base agent unavailable")

    monkeypatch.setattr(exp, "run_live_held_family_episodes", raise_live_path_error)
    artifact = exp.build_artifact(write=False, duration_s=0.1)
    assert artifact["status"] == "blocked_live_path"
    assert artifact["gate_check_summary"]["failed_check"] == "live_path"
    assert "base agent unavailable" in artifact["gate_check_summary"]["observed"]

    output = tmp_path / "invalid.json"
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--result-path", str(output)]) == 1


def test_scenario_6681_framework_loader_skips_optional_templates(tmp_path, monkeypatch):
    """SCENARIO-ARC-WMTE-6681-ARTIFACT loads only the canonical base agent."""

    package = tmp_path / "agents"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('template import')\n")
    (package / "recorder.py").write_text("class Recorder: pass\n")
    (package / "tracing.py").write_text("def trace_agent_session(function):\n    return function\n")
    (package / "agent.py").write_text(
        "from .recorder import Recorder\n"
        "from .tracing import trace_agent_session\n"
        "class Agent:\n    pass\n"
    )
    monkeypatch.setattr(exp, "FRAMEWORK_ROOT", tmp_path)
    monkeypatch.setattr(exp, "FRAMEWORK_AGENT_PATH", package / "agent.py")
    for name in list(sys.modules):
        if name.startswith("_carnot_arc_runtime"):
            monkeypatch.delitem(sys.modules, name)

    loaded = exp._load_framework_agent()

    assert loaded.__name__ == "Agent"
    assert "agents" not in loaded.__module__


def test_scenario_6681_normalizers_and_transport_defenses_fail_closed():
    """EXACT-RETURN and ATTACKS cover public boundary variants and misuse."""

    class _State(Enum):
        WON = "WIN"

    class _Dump:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"value": 4}

    class _Opaque:
        def __str__(self):
            return "opaque"

    assert transport_module._json_value(_State.WON) == "WIN"
    assert transport_module._json_value(np.array([1, 2])) == [1, 2]
    assert transport_module._json_value(_Dump()) == {"value": 4}
    assert transport_module._json_value(_Opaque()) == "opaque"
    assert normalize_action({"action": None, "data": None})["kind"] == "RESET"
    assert normalize_action(("ACTION2", None))["kind"] == 2
    assert normalize_observation(None) is None
    assert normalize_observation({"state": "WIN"}) == {"state": "WIN"}
    assert observation_hash({"state": "WIN"}).startswith("sha256:")

    tuple5 = E3OutcomeTransport(family="held-a", attempt=40, episode_seed=1, episode_id="tuple5")
    tuple5.record_proposal(
        proposed_action=(1, None),
        policy_selected_action=(1, None),
        observation_before=_Frame(0),
        supervisor_decision={"fired": False},
    )
    tuple5.record_application((1, None))
    tuple5_step = tuple5.begin_environment_step((1, None))
    tuple5.record_environment_return(tuple5_step, (_Frame(1), 2.0, True, False, {}))
    rows, _ = join_outcome_events(tuple5.events())
    assert rows[0]["return_schema"] == "tuple5"
    assert rows[0]["termination"]["terminated"] is True

    empty = E3OutcomeTransport(family="held-a", attempt=0, episode_seed=1, episode_id="empty")
    with pytest.raises(OutcomeLineageError):
        empty.record_application((1, None))
    with pytest.raises(OutcomeLineageError):
        empty.begin_environment_step((1, None))
    with pytest.raises(OutcomeLineageError):
        empty.record_environment_return("unknown", _Frame(1))

    pending = E3OutcomeTransport(family="held-a", attempt=0, episode_seed=1, episode_id="pending")
    pending.record_proposal(
        proposed_action=(1, None),
        policy_selected_action=(1, None),
        observation_before=_Frame(0),
        supervisor_decision=None,
    )
    with pytest.raises(OutcomeLineageError):
        pending.record_proposal(
            proposed_action=(1, None),
            policy_selected_action=(1, None),
            observation_before=_Frame(0),
            supervisor_decision=None,
        )
    with pytest.raises(OutcomeLineageError):
        pending.record_application((2, None))
    pending.record_application((1, None))
    with pytest.raises(OutcomeLineageError):
        pending.begin_environment_step((2, None))
    pending_step = pending.begin_environment_step((1, None))
    with pytest.raises(OutcomeLineageError):
        pending.begin_environment_step((1, None))
    with pytest.raises(OutcomeLineageError):
        pending.record_environment_return("wrong", _Frame(1))
    with pytest.raises(ValueError):
        pending.record_environment_failure(pending_step, status="cancelled", error="x")

    missing_id = _one_transport(redirect=True, index=41).events()
    missing_id["proposals"][0].pop("proposal_id")
    with pytest.raises(OutcomeLineageError):
        join_outcome_events(missing_id)
    proposal_only = _one_transport(redirect=True, index=42).events()
    proposal_only["applications"].clear()
    rows, audit = join_outcome_events(proposal_only)
    assert rows == []
    assert audit["issues"][0]["reason"] == "application_child_count"


def test_scenario_6681_blocked_reductions_and_host_fallbacks(tmp_path, monkeypatch):
    """REQ-ARC-WMTE-6681 covers deterministic blockers and provenance fallbacks."""

    assert exp._load_json(tmp_path / "absent.json") == {}
    original_read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, **kwargs: "NoMem: 1 kB")
    assert exp._memory_total_bytes() == 0

    def raise_oserror(self, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(Path, "read_text", raise_oserror)
    assert exp._memory_total_bytes() == 0
    monkeypatch.setattr(Path, "read_text", original_read_text)
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)),
    )
    assert exp._preconditions(exp.REPO_ROOT, live_metadata={})["sdk"]["version"] == "missing"

    low = exp.build_artifact(episode_events=_event_bundle(redirects=1, controls=0), write=False)
    assert low["status"] == "complete_transport_ready_below_downstream_row_floor"
    assert low["non_redirect_control_rows"] == []

    duplicate = _event_bundle(redirects=1, controls=0)
    duplicate["outcomes"].append(copy.deepcopy(duplicate["outcomes"][0]))
    blocked = exp.build_artifact(episode_events=duplicate, write=False)
    assert blocked["status"] == "blocked_ambiguous_redirect_outcome_lineage"

    readiness_mutation = copy.deepcopy(low)
    readiness_mutation["arc_outcome_transport_ready"] = False
    readiness_mutation["verdict_class"] = "blocked"
    readiness_mutation["reproducibility_checksum"] = exp.reproducibility_checksum(
        readiness_mutation
    )
    assert "readiness mismatch" in exp.validate_artifact(readiness_mutation)

    writes = []
    monkeypatch.setattr(
        exp,
        "write_artifact_json",
        lambda path, payload, **kwargs: writes.append(path),
    )
    exp.build_artifact(
        episode_events=_event_bundle(redirects=1, controls=0),
        result_path=Path("results/scoped.json"),
    )
    assert writes == [exp.REPO_ROOT / "results/scoped.json"]

    calls = []
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: calls.append(kwargs) or {})
    assert exp.main(["--result-path", "results/relative.json"]) == 0
    assert calls[0]["result_path"] == exp.REPO_ROOT / "results/relative.json"
