"""Tests for the current-event ARC live sentinel.

Spec refs: REQ-ARC-WMTE-7431 and SCENARIO-ARC-WMTE-7431-*.
"""

from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest

from carnot import experiment_7431_v651_arc_live_sentinel as mod


pytestmark = pytest.mark.memory_watchdog_skip


def _event(call: str, operation: str, state: str, tick: int) -> dict[str, object]:
    """Build one raw event in the shipped invocation-boundary schema."""

    row = {
        "schema": "carnot.arc_inference_boundary_event.v1",
        "event_id": f"sha256:{call}-{state}",
        "call_id": call,
        "operation": operation,
        "state": state,
        "recorded_monotonic_ns": tick,
        "started_monotonic_ns": tick if state == "attempted" else tick - 1,
        "owner_pid": 41,
        "child_pid": 42 if operation == "model_load" else 43,
        "model_identity": {
            "model_repository": mod.MODEL_ID,
            "model_filename": mod.MODEL_FILENAME,
            "model_revision": "fixture-revision",
            "model_path": f"/cache/{mod.MODEL_FILENAME}",
        },
    }
    if state in {"completed", "failed"}:
        row["ended_monotonic_ns"] = tick
    return row


def _schedule() -> list[dict[str, object]]:
    """Return the two frozen public development units."""

    return mod.build_schedule(["bp35", "cn04"])


def _episode(
    episode_id: str,
    *,
    actions: int = 1,
    disposition: str = "complete",
    requests: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one durable episode row for reducer tests."""

    return {
        "episode_id": episode_id,
        "disposition": disposition,
        "policy_entry": {
            "factory": "make_carnot_agent",
            "policy_class": "E3AgentPolicy",
            "choose_action_path": True,
            "is_done_path": True,
        },
        "first_action": ({"action": "ACTION1", "elapsed_s": 0.2} if actions else None),
        "first_observation": (
            {"level": 0, "frame_sha256": "sha256:" + "a" * 64} if actions else None
        ),
        "actions": actions,
        "start_level": 0,
        "max_level": 0,
        "level_progress": 0,
        "tool_feedback_consumed": 0,
        "supervisor": {"fired": 0, "consumed": 0},
        "elapsed_s": 1.0,
        "request_budget_receipt": {
            "episode_id": episode_id,
            "limit": 2,
            "attempted": len(requests or []),
            "completed": sum(row["disposition"] == "completed" for row in requests or []),
            "failed": sum(row["disposition"] == "failed" for row in requests or []),
            "cancelled": sum(row["disposition"] == "cancelled" for row in requests or []),
            "in_flight": 0,
            "remaining": 2 - len(requests or []),
            "accounting_valid": True,
            "closed": True,
            "callback_rows": requests or [],
        },
    }


def _receipts() -> list[dict[str, object]]:
    """Return one passing receipt for every required scoped and terminal check."""

    return [
        {
            "name": name,
            "command_argv": [name],
            "command_environment": {},
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "b" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (
            *mod.validation_scope.REQUIRED_CHECK_NAMES,
            *mod.REQUIRED_E2E,
            *mod.REQUIRED_TERMINAL,
        )
    ]


def _valid_artifact() -> dict[str, object]:
    """Build the reusable valid terminal fixture."""

    schedule = _schedule()
    requests = [
        {
            "episode_id": row["episode_id"],
            "request_id": f"request-{index}",
            "branch": "generation",
            "reservation_index": 0,
            "reserved_monotonic": 1.0,
            "terminal_monotonic": 2.0,
            "elapsed_s": 1.0,
            "disposition": "completed",
            "request_dispatched": True,
            "response_observed": True,
            "error": None,
            "cancel_reason": None,
            "recovered_after_restart": False,
        }
        for index, row in enumerate(schedule)
    ]
    episodes = [
        _episode(str(row["episode_id"]), requests=[requests[index]])
        for index, row in enumerate(schedule)
    ]
    events = [
        _event("load", "model_load", "attempted", 1),
        _event("load", "model_load", "completed", 2),
        _event("g1", "generation", "attempted", 3),
        _event("g1", "generation", "completed", 4),
        _event("g2", "generation", "attempted", 5),
        _event("g2", "generation", "completed", 6),
    ]
    return mod.build_artifact_for_test(
        schedule=schedule,
        episode_rows=episodes,
        boundary_events=events,
        validation_receipts=_receipts(),
    )


def test_req_arc_wmte_7431_contract_and_fixed_bounds_exist() -> None:
    """REQ-ARC-WMTE-7431 fixes model, venue, unit, action, and request limits."""

    text = mod.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-ARC-WMTE-7431" in text
    assert "SCENARIO-ARC-WMTE-7431-CURRENT-EVENTS" in text
    assert mod.MODEL_SPECS == ["unsloth/Qwen3.8-27B-GGUF"]
    assert mod.INFERENCE_SUBSTRATE_CLASS == "model_bounded_generation"
    assert mod.EXECUTION_VENUE == "host"
    assert mod.ACTION_LIMIT == 64
    assert mod.REQUEST_LIMIT == 2
    assert mod.MAX_NEW_TOKENS == 256
    assert mod.EPISODE_LIMIT_S == 240.0
    assert mod.AGGREGATE_LIVE_LIMIT_S == 900.0


def test_scenario_arc_wmte_7431_current_events_reduce_attempts_and_terminals() -> None:
    """SCENARIO-ARC-WMTE-7431-CURRENT-EVENTS counts raw current events only."""

    events = [
        _event("load", "model_load", "attempted", 1),
        _event("load", "model_load", "child_started", 2),
        _event("load", "model_load", "completed", 3),
        _event("g1", "generation", "attempted", 4),
        _event("g1", "generation", "completed", 5),
        _event("g2", "generation", "attempted", 6),
    ]
    reduced = mod.reduce_current_invocations(events, child_terminal=True)
    assert reduced["errors"] == []
    assert reduced["invocation_counts"] == {
        "model_loads_attempted": 1,
        "model_loads_completed": 1,
        "model_loads_failed": 0,
        "model_loads_cancelled": 0,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 2,
        "generation_calls_completed": 1,
        "generation_calls_failed": 0,
        "generation_calls_cancelled": 1,
        "generation_calls_in_flight": 0,
    }
    assert reduced["model_invoked"] is True
    assert reduced["inference_substrate_class"] == "model_bounded_generation"

    load_only = mod.reduce_current_invocations(events[:3], child_terminal=True)
    assert load_only["inference_substrate_class"] == "model_load_no_generation"
    empty = mod.reduce_current_invocations([], child_terminal=True)
    assert empty["inference_substrate_class"] == "no_model_load"


def test_scenario_arc_wmte_7431_sentinel_stops_second_episode() -> None:
    """SCENARIO-ARC-WMTE-7431-SENTINEL leaves unit two unstarted after no action."""

    schedule = _schedule()
    first = _episode(
        str(schedule[0]["episode_id"]), actions=0, disposition="censored_no_first_action"
    )
    reduced = mod.reduce_episode_panel(schedule, [first])
    assert reduced["sentinel_reached_first_action"] is False
    assert [row["progress_score"] for row in reduced["per_game_results"]] == [0, 0]
    assert [row["disposition"] for row in reduced["per_game_results"]] == [
        "censored_no_first_action",
        "unstarted",
    ]
    assert reduced["sample_size_budget"] == {
        "planned_units": 2,
        "attempted_units": 1,
        "completed_units": 0,
        "failed_units": 0,
        "censored_units": 1,
        "unstarted_units": 1,
        "independent_groups": ["public_adapter_withheld_development_proxy"],
        "stopping_rule": "Stop after two terminal episodes, or after sentinel failure before first action.",
        "request_limit_per_episode": 2,
        "action_limit_per_episode": 64,
        "episode_limit_s": 240.0,
        "aggregate_live_limit_s": 900.0,
    }


def test_scenario_arc_wmte_7431_budget_chain_rejects_excess_dispatch() -> None:
    """SCENARIO-ARC-WMTE-7431-BUDGET requires permit-before-request chains."""

    rows = [
        {
            "episode_id": "bp35:seed-7431651",
            "request_id": "one",
            "branch": "generation",
            "reservation_index": 0,
            "reserved_monotonic": 1.0,
            "terminal_monotonic": 2.0,
            "elapsed_s": 1.0,
            "disposition": "completed",
            "request_dispatched": True,
            "response_observed": True,
            "error": None,
            "cancel_reason": None,
            "recovered_after_restart": False,
        },
        {
            "episode_id": "bp35:seed-7431651",
            "request_id": "two",
            "branch": "repair",
            "reservation_index": 1,
            "reserved_monotonic": 3.0,
            "terminal_monotonic": 4.0,
            "elapsed_s": 1.0,
            "disposition": "failed",
            "request_dispatched": True,
            "response_observed": False,
            "error": "fixture",
            "cancel_reason": None,
            "recovered_after_restart": False,
        },
    ]
    reduced = mod.reduce_request_budget_rows(rows)
    assert reduced["permit_to_terminal_chains"] == 2
    assert reduced["excess_dispatches"] == 0
    assert reduced["unterminated_permits"] == 0

    broken = deepcopy(rows)
    broken.append({**deepcopy(rows[0]), "request_id": "three", "reservation_index": 2})
    assert mod.reduce_request_budget_rows(broken)["excess_dispatches"] == 1


def test_scenario_arc_wmte_7431_terminal_artifact_is_null_not_efficacy() -> None:
    """SCENARIO-ARC-WMTE-7431-TERMINAL makes reachability complete without benefit."""

    schedule = _schedule()
    requests = [
        {
            "episode_id": row["episode_id"],
            "request_id": f"request-{index}",
            "branch": "generation",
            "reservation_index": 0,
            "reserved_monotonic": 1.0,
            "terminal_monotonic": 2.0,
            "elapsed_s": 1.0,
            "disposition": "completed",
            "request_dispatched": True,
            "response_observed": True,
            "error": None,
            "cancel_reason": None,
            "recovered_after_restart": False,
        }
        for index, row in enumerate(schedule)
    ]
    episodes = [
        _episode(str(row["episode_id"]), requests=[requests[index]])
        for index, row in enumerate(schedule)
    ]
    events = [
        _event("load", "model_load", "attempted", 1),
        _event("load", "model_load", "completed", 2),
        _event("g1", "generation", "attempted", 3),
        _event("g1", "generation", "completed", 4),
        _event("g2", "generation", "attempted", 5),
        _event("g2", "generation", "completed", 6),
    ]
    artifact = mod.build_artifact_for_test(
        schedule=schedule,
        episode_rows=episodes,
        boundary_events=events,
        validation_receipts=_receipts(),
    )
    assert mod.validate_artifact(artifact) == []
    assert artifact["arc_sentinel_capture_complete_score"] == 1
    assert artifact["live_efficacy_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["hidden_leaderboard_claimed"] is False

    changed = deepcopy(artifact)
    changed["invocation_counts"]["generation_calls_attempted"] = 99
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "invocation_reduction_mismatch" in mod.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["live_efficacy_score"] = 1
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "nonpromotion_scores_invalid" in mod.validate_artifact(changed)


def test_req_arc_wmte_7431_validation_manifest_is_scoped(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7431 freezes exact affected paths and excludes the full suite."""

    commands = mod.build_validation_plan(mod.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(mod.validation_scope.REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(mod.REPO_ROOT, commands) == []
    assert "full_python_suite" not in {row.name for row in commands}
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert set(mod.VALIDATION_MANIFEST.test_paths) <= set(focused.argv)
    coverage = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(coverage.command_environment)["COVERAGE_FILE"].endswith("/.coverage")


def test_req_arc_wmte_7431_atomic_write_and_cli_date(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7431 publishes complete JSON and fixes the execution date."""

    output = tmp_path / "artifact.json"
    mod.atomic_json(output, {"terminal": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"terminal": True}
    assert mod.parse_args(["--date", "20260919"]).date == "20260919"
    assert mod.parse_args(["--date", "20260919", "--finalize-existing"]).finalize_existing
    with pytest.raises(SystemExit):
        mod.parse_args(["--date", "20260918"])


def test_support_readers_gates_and_duration_buckets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7431 keeps input identities and phase costs independently readable."""

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps({"status": "complete", "verdict_class": "null", "flagged_adversarial": False}),
        encoding="utf-8",
    )
    assert mod.sha256_file(source).startswith("sha256:")
    assert mod.load_object(source)["verdict_class"] == "null"
    assert mod.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert mod.load_object(malformed) == {}
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert mod.load_object(array) == {}
    record = mod._source_record(source, role="fixture")
    assert record["original_flags"]["flagged_adversarial"] is False

    assert mod._compare("in", ["null"], "null")
    assert mod._compare(">=", 1, 2)
    assert not mod._compare(">=", 1, None)
    with pytest.raises(ValueError, match="unsupported operator"):
        mod._compare("!=", 1, 2)
    failed = mod.gate_row(
        "fixture",
        "test",
        True,
        False,
        upstream="test",
        artifact_field="fixture",
        principle="exercise failure summary",
    )
    assert mod.gate_summary([failed])["first_failure"]["check"] == "fixture"
    assert mod._field_principles(["schema", "custom"])["custom"].startswith("The custom")
    assert mod._duration_breakdown(
        [
            {"phase": "validation", "duration_s": 1},
            {"phase": "live_model", "duration_s": 2},
            {"phase": "terminal_cold", "duration_s": 3},
            {"phase": "selection", "duration_s": 4},
        ]
    ) == {"validation_s": 1.0, "model_s": 2.0, "cold_start_s": 3.0, "other_s": 4.0}
    mod.progress(0.0, "test", "boundary", completed_units=1)
    assert "completed_units=1" in capsys.readouterr().out
    assert "+00:00" in mod.utc_now()


def test_repo_preconditions_and_label_blind_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7431 authenticates the real inputs before freezing the panel."""

    monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")
    checks, hashes, registry = mod.collect_preconditions(mod.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert mod.EXP7422_PATH.as_posix() in hashes
    selection = mod.freeze_two_game_rotation(registry)
    assert selection["passed"] is True
    assert len(selection["games"]) == 2
    assert selection["selection_used_current_outcomes"] is False

    monkeypatch.setattr(
        mod.yaml, "safe_load", lambda _text: (_ for _ in ()).throw(mod.yaml.YAMLError())
    )
    malformed_checks, _, malformed_registry = mod.collect_preconditions(mod.REPO_ROOT)
    assert malformed_registry == {}
    assert (
        next(row for row in malformed_checks if row["check"] == "manifest_and_registry_parse")[
            "passed"
        ]
        is False
    )


def test_reducers_cover_unstarted_invalid_and_transport_edges() -> None:
    """SCENARIO-ARC-WMTE-7431-SENTINEL retains every censored accounting edge."""

    schedule = _schedule()
    empty = mod.reduce_episode_panel(schedule, [])
    assert empty["sentinel_started"] is False
    assert empty["arc_sentinel_capture_complete_score"] == 0
    failed = _episode(str(schedule[0]["episode_id"]), disposition="failed")
    panel = mod.reduce_episode_panel(schedule, [failed])
    assert panel["sample_size_budget"]["failed_units"] == 1
    assert panel["sample_size_budget"]["unstarted_units"] == 1

    request_rows = [
        {
            "episode_id": "episode",
            "request_id": "open",
            "disposition": "in_flight",
            "request_dispatched": True,
            "reserved_monotonic": 2.0,
            "request_started_monotonic": 1.0,
        }
    ]
    request_reduction = mod.reduce_request_budget_rows(request_rows)
    assert request_reduction["unterminated_permits"] == 1
    assert request_reduction["ordering_errors"] == 1

    episode = _episode("episode")
    episode["request_budget_receipt"] = {
        "callback_rows": [
            None,
            {"reservation_index": 0, "request_id": "r", "disposition": "failed"},
        ]
    }
    episode["server_request_rows"] = [
        None,
        {
            "call_index": 0,
            "request_dispatched": True,
            "response_observed": False,
            "request_started_monotonic": 3.0,
            "error": "transport",
        },
    ]
    assert (
        mod._request_rows([{"request_budget_receipt": None}, episode])[0]["transport_error"]
        == "transport"
    )
    assert not mod._receipts_pass([*_receipts(), deepcopy(_receipts()[-1])], mod.REQUIRED_TERMINAL)


def test_durable_budget_persists_complete_failure_and_cancellation(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7431-BUDGET persists each permit transition as it occurs."""

    events = tmp_path / "budget.jsonl"
    budget = mod.DurableEpisodeRequestBudget("episode", limit=2, deadline_s=60, event_path=events)
    budget.reserve(branch="generation", request_id="complete").complete()
    budget.reserve(branch="repair", request_id="failed").fail("fixture")
    rows = mod._read_jsonl(events)
    assert [row["event"] for row in rows].count("permit_acquired") == 2
    assert [row["disposition"] for row in rows if row["event"] == "permit_terminal"] == [
        "completed",
        "failed",
    ]

    cancel_events = tmp_path / "cancel.jsonl"
    cancelled = mod.DurableEpisodeRequestBudget(
        "cancelled", limit=2, deadline_s=60, event_path=cancel_events
    )
    cancelled.reserve(branch="generation", request_id="open")
    cancelled.cancel("episode_terminal")
    assert mod._read_jsonl(cancel_events)[-1]["disposition"] == "cancelled"
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"ok": 1}\nnot-json\n[]\n', encoding="utf-8")
    assert mod._read_jsonl(broken) == [{"ok": 1}]
    assert mod._read_jsonl(tmp_path / "absent.jsonl") == []


def test_transport_capture_records_response_error_and_token_guard(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7431-BUDGET joins real transport outcomes to request rows."""

    events = tmp_path / "transport.jsonl"
    capture = mod.DurableRequestCapture(tmp_path, events)
    capture.begin_episode("game:seed")
    capture.original = lambda *_args, **_kwargs: io.BytesIO(b'{"content":"ok"}')
    request = urllib.request.Request(
        "http://127.0.0.1:8000/completion",
        data=json.dumps({"n_predict": 256}).encode(),
    )
    assert capture._open(request).read() == b'{"content":"ok"}'
    assert mod._read_jsonl(events)[-1]["event"] == "server_response"

    def fail(*_args: object, **_kwargs: object) -> object:
        raise OSError("fixture transport")

    capture.original = fail
    with pytest.raises(OSError, match="fixture transport"):
        capture._open(request)
    assert mod._read_jsonl(events)[-1]["event"] == "server_error"
    too_large = urllib.request.Request(
        "http://127.0.0.1:8000/completion",
        data=json.dumps({"max_tokens": 257}).encode(),
    )
    with pytest.raises(RuntimeError, match="token budget exceeded"):
        capture._open(too_large)
    capture.original = lambda *_args, **_kwargs: io.BytesIO(b"{}")
    malformed = urllib.request.Request("http://127.0.0.1:8000/completion", data=b"{")
    assert capture._open(malformed).read() == b"{}"
    capture.original = lambda *_args, **_kwargs: "bypass"
    assert capture._open("not-a-request") == "bypass"

    original = urllib.request.urlopen
    capture.install()
    assert urllib.request.urlopen == capture._open
    capture.restore()
    assert urllib.request.urlopen is original


def test_runtime_helpers_preserve_withholding_and_event_reduction(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7431 preserves adapter withholding and event identities."""

    env = mod.session_environment(
        {"CARNOT_ARC_PLAYBOOK_RETRIEVAL": "1", "KEEP": "yes"},
        gpu_index=3,
        port=9123,
        raw_dir=tmp_path,
    )
    assert "CARNOT_ARC_PLAYBOOK_RETRIEVAL" not in env
    assert env["KEEP"] == "yes"
    assert env["CARNOT_ARC_INDUCE_MAX_TOKENS"] == "256"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    blob = tmp_path / "blob"
    blob.write_bytes(b"model")
    snapshot = tmp_path / "snapshots" / "revision" / mod.MODEL_FILENAME
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)
    assert mod._absolute_model_path(snapshot) == str(snapshot.absolute())
    assert mod._absolute_model_path(snapshot) != str(snapshot.resolve())

    frame = SimpleNamespace(frame=SimpleNamespace(levels=[1, 3], frame=[[1, 2]]))
    assert mod._level(frame) == 3
    assert mod._level(object()) == 0
    assert mod._frame_hash(frame).startswith("sha256:")
    events = [
        {"event": "unrelated", "episode_id": "e", "call_index": 0},
        {"event": "server_request", "episode_id": "other", "call_index": 0},
        {"event": "server_request", "episode_id": "e", "call_index": 1, "a": 1},
        {"event": "server_response", "episode_id": "e", "call_index": 1, "b": 2},
    ]
    assert mod._transport_rows(events, "e") == [
        {"event": "server_response", "episode_id": "e", "call_index": 1, "a": 1, "b": 2}
    ]
    spans: list[dict[str, object]] = []
    mod._phase(spans, "fixture", 1.0, 0.0, 1, "checkpoint")
    assert spans[0]["checkpoint"] == "checkpoint"

    class Lease:
        transitions: list[tuple[str, dict[str, object]]] = []

        def transition(self, phase: str, **details: object) -> None:
            self.transitions.append((phase, details))

    lease = Lease()
    assert mod._observe_loaded_lease(lease, {"model_loaded": False}, False) is False
    assert mod._observe_loaded_lease(lease, {"model_loaded": True}, False) is True
    assert lease.transitions == [("resident", {"vram_mb": 0}), ("inferencing", {})]
    assert mod._observe_loaded_lease(lease, {"model_loaded": True}, True) is True


def test_cold_reduction_blocked_record_and_terminal_command_plan(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7431-TERMINAL validates complete and externally blocked records."""

    artifact = _valid_artifact()
    candidate = tmp_path / "candidate.json"
    mod.atomic_json(candidate, artifact)
    assert mod.independent_reduce_file(candidate)["matches_declared"] is True
    assert mod.validate_artifact(candidate) == []
    assert [row.name for row in mod.terminal_command_specs(mod.REPO_ROOT, candidate)] == list(
        mod.REQUIRED_TERMINAL
    )
    assert [row.name for row in mod.e2e_command_specs(mod.REPO_ROOT, tmp_path / "e2e")] == list(
        mod.REQUIRED_E2E
    )

    failed_check = mod.gate_row(
        "resource",
        "precondition",
        "available",
        None,
        upstream="fixture",
        artifact_field="resource",
        principle="External absence blocks work.",
    )
    blocked = mod._blocked_artifact(
        started_at="2026-09-19T00:00:00+00:00",
        duration_s=0.5,
        spans=[],
        checks=[failed_check],
        hashes={},
        receipts=[],
        schedule=_schedule(),
        selection={"passed": True},
    )
    assert mod.validate_artifact(blocked) == []
    blocked_cases: list[tuple[str, str, object]] = [
        ("model_invoked", "blocked_model_work_invalid", True),
        ("promotion_score", "nonpromotion_scores_invalid", 1),
        ("execution_venue", "execution_venue_invalid", "container"),
        ("duration_s", "duration_invalid", 0),
        ("honest_verdict", "verdict_invalid", "complete_wrong"),
        ("solve_credit", "solve_claim_invalid", 1),
        ("gate_check_summary", "blocked_gate_summary_invalid", {}),
        ("field_principles", "field_principles_incomplete", {}),
    ]
    for field, expected, replacement in blocked_cases:
        changed = deepcopy(blocked)
        changed[field] = replacement
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed), field
    changed = deepcopy(blocked)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "checksum_mismatch" in mod.validate_artifact(changed)


def test_validator_rejects_each_declared_boundary_drift() -> None:
    """REQ-ARC-WMTE-7431 cold validation rejects provenance, budget, and claim drift."""

    artifact = _valid_artifact()
    cases: list[tuple[str, str, object]] = [
        ("schema", "identity_invalid", "wrong"),
        ("model_invoked", "model_invoked_mismatch", False),
        ("inference_substrate_class", "substrate_class_mismatch", "no_model_load"),
        ("MODEL_SPECS", "MODEL_SPECS_invalid", []),
        ("execution_venue", "execution_venue_invalid", "container"),
        ("duration_s", "duration_invalid", 0),
        ("duration_s", "duration_floor_invalid", 1),
        ("sample_size_budget", "sample_size_budget_mismatch", {}),
        ("arc_sentinel_capture_complete_score", "capture_score_mismatch", 0),
        ("request_budget_reduction", "request_budget_reduction_mismatch", {}),
        ("solve_credit", "solve_claim_invalid", 1),
        ("verdict_class", "verdict_invalid", "positive"),
        ("honest_verdict", "verdict_invalid", "not-complete"),
        ("validation_receipts", "validation_receipts_invalid", []),
        ("field_principles", "field_principles_incomplete", {}),
    ]
    for field, expected, replacement in cases:
        changed = deepcopy(artifact)
        changed[field] = replacement
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed), field

    changed = deepcopy(artifact)
    changed["current_invocation_events"][0]["schema"] = "invalid"
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "invocation_events_invalid" in mod.validate_artifact(changed)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "checksum_mismatch" in mod.validate_artifact(changed)

    load_only = mod.build_artifact_for_test(
        schedule=_schedule(),
        episode_rows=[_episode(str(row["episode_id"])) for row in _schedule()],
        boundary_events=[
            _event("load", "model_load", "attempted", 1),
            _event("load", "model_load", "completed", 2),
        ],
        validation_receipts=_receipts(),
    )
    assert (
        load_only["inference_substrate"] == "owned_native_cuda_llama_cpp_model_load_no_generation"
    )
    no_model = mod.build_artifact_for_test(
        schedule=_schedule(),
        episode_rows=[],
        boundary_events=[],
        validation_receipts=[],
    )
    assert no_model["inference_substrate"] == "no_model_load"
    assert no_model["verdict_class"] == "disqualified"
    assert no_model["solve_provenance"] == "uncredited_no_runtime"


def test_cli_requires_date_without_replay() -> None:
    """REQ-ARC-WMTE-7431 refuses an undated public invocation."""

    with pytest.raises(SystemExit):
        mod.parse_args([])
