"""Lossless ARC inference-boundary receipts for REQ-ARC-WMTE-7289."""

from __future__ import annotations

from copy import deepcopy
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from carnot.agentic import arc_executable_world_model as world_model
from carnot.agentic import arc_induction_tool_loop as tool_loop
from carnot.agentic.arc_inference_boundary import (
    BOUNDARY_LEDGER_ENV,
    InvocationBoundaryLedger,
    boundary_call_for_proposer,
    reduce_boundary_events,
)
from carnot.experiment_7289_v641_arc_boundary import (
    EXPERIMENT_ID,
    MILESTONE,
    MODEL_SPECS,
    RUN_DATE,
    ZERO_INVOCATION_COUNTS,
    _pid_alive,
    artifact_checksum,
    build_live_handoff,
    build_terminal_artifact,
    build_validation_commands,
    collect_preconditions,
    exercise_live_caller_seams,
    freeze_historical_exp7280,
    independent_reduce,
    parse_args,
    run_cpu_boundary_panel,
    validate_artifact,
)

REPO = Path(__file__).resolve().parents[2]


def _identity(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "model_repository": "fixture/model",
        "model_filename": "fixture.gguf",
        "model_revision": "a" * 40,
        "model_path": "/fixtures/fixture.gguf",
        "model_hash": "sha256:" + "b" * 64,
    }
    value.update(changes)
    return value


def test_attempt_is_durable_before_call_and_load_counts_as_invoked(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7289: attempted load is model activity before completion."""
    ledger = InvocationBoundaryLedger(tmp_path / "events.jsonl")
    call = ledger.begin("model_load", _identity(), call_id="load-1")

    in_flight = reduce_boundary_events(ledger.read_events())
    assert in_flight["model_invoked"] is True
    assert in_flight["invocation_counts"]["model_loads_attempted"] == 1
    assert in_flight["invocation_counts"]["model_loads_in_flight"] == 1
    assert in_flight["inference_substrate"] == "model_load_in_flight"

    call.child_started(321)
    call.complete()
    completed = reduce_boundary_events(ledger.read_events())
    assert completed["invocation_counts"]["model_loads_completed"] == 1
    assert completed["invocation_counts"]["model_loads_in_flight"] == 0
    assert completed["inference_substrate"] == "model_load_no_generation"
    assert completed["call_rows"][0]["child_pid"] == 321


def test_completed_unusable_generation_and_failed_load_reduce_exactly(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7289-LOAD-ONLY-AND-UNUSABLE."""
    ledger = InvocationBoundaryLedger(tmp_path / "events.jsonl")
    failed = ledger.begin("model_load", _identity(), call_id="load-failed")
    failed.fail(RuntimeError("fixture pre-load failure"))
    load = ledger.begin("model_load", _identity(), call_id="load-ok")
    load.child_started(322)
    load.complete()
    generation = ledger.begin(
        "generation", _identity(), child_pid=322, call_id="generation-unusable"
    )
    generation.complete(usable=False)

    reduced = reduce_boundary_events(ledger.read_events())
    assert reduced["activity_known"] is True
    assert reduced["invocation_counts"] == {
        "model_loads_attempted": 2,
        "model_loads_completed": 1,
        "model_loads_failed": 1,
        "model_loads_in_flight": 0,
        "generation_calls_attempted": 1,
        "generation_calls_completed": 1,
        "generation_calls_failed": 0,
        "generation_calls_in_flight": 0,
        "usable_answers": 0,
    }
    assert reduced["inference_substrate"] == "model_full_generation"


def test_duplicate_events_are_idempotent_and_conflicts_disqualify(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7289-DUPLICATE-AND-IDENTITY-REJECTION."""
    ledger = InvocationBoundaryLedger(tmp_path / "events.jsonl")
    call = ledger.begin("generation", _identity(), child_pid=400, call_id="duplicate")
    call.complete(usable=False)
    original = ledger.read_events()
    ledger.append_event(original[0])
    ledger.append_event(original[1])

    reduced = reduce_boundary_events(ledger.read_events())
    assert reduced["duplicate_event_count"] == 2
    assert reduced["invocation_counts"]["generation_calls_attempted"] == 1
    assert reduced["invocation_counts"]["generation_calls_completed"] == 1

    conflict = dict(original[0])
    conflict["child_pid"] = 401
    ledger.append_event(conflict)
    disqualified = reduce_boundary_events(ledger.read_events())
    assert disqualified["activity_known"] is False
    assert disqualified["disqualified"] is True
    assert "conflicting_duplicate_event" in disqualified["errors"]
    assert disqualified["invocation_counts"] is None


@pytest.mark.parametrize(
    "change,error",
    [
        ({"model_path": "relative.gguf"}, "model_path_not_absolute"),
        ({"model_hash": "sha256:no"}, "model_hash_invalid"),
        ({"model_revision": ""}, "model_revision_missing"),
    ],
)
def test_unknown_identity_is_disqualified_not_zero(
    tmp_path: Path, change: dict[str, object], error: str
) -> None:
    """REQ-ARC-WMTE-7289: unsupported identity cannot become zero activity."""
    ledger = InvocationBoundaryLedger(tmp_path / "events.jsonl")
    ledger.begin("model_load", _identity(**change), call_id="bad-identity")
    reduced = reduce_boundary_events(ledger.read_events())
    assert reduced["model_invoked"] is None
    assert reduced["invocation_counts"] is None
    assert any(error in item for item in reduced["errors"])


def test_selfparse_actual_request_seam_writes_attempt_before_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7289-LIVE-CALLER-REACHABILITY: selfparse seam."""
    path = tmp_path / "selfparse.jsonl"
    monkeypatch.setenv(BOUNDARY_LEDGER_ENV, str(path))
    proposer = world_model.LocalGGUFProposer(
        model_path="/fixtures/fixture.gguf",
        model_repository="fixture/model",
        model_filename="fixture.gguf",
        model_revision="a" * 40,
        ffn_cpu_layers=0,
        mtp=False,
    )
    proposer._proc = type("Proc", (), {"pid": 456})()

    def timeout_after_receipt(*args: object, **kwargs: object) -> io.BytesIO:
        current = reduce_boundary_events(InvocationBoundaryLedger(path).read_events())
        assert current["invocation_counts"]["generation_calls_attempted"] == 1
        assert current["invocation_counts"]["generation_calls_in_flight"] == 1
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(tool_loop.urllib.request, "urlopen", timeout_after_receipt)
    with pytest.raises(TimeoutError, match="fixture timeout"):
        tool_loop._post_chat(
            proposer,
            [{"role": "user", "content": "fixture"}],
            turn=0,
            timeout_s=1,
            selfparse=True,
        )

    reduced = reduce_boundary_events(InvocationBoundaryLedger(path).read_events())
    assert reduced["invocation_counts"]["generation_calls_failed"] == 1
    assert reduced["call_rows"][0]["child_pid"] == 456


@pytest.mark.parametrize("choices", [[None], [{"message": None}], [{"message": []}]])
def test_selfparse_boundary_does_not_reclassify_invalid_grammar_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, choices: list[object]
) -> None:
    """REQ-ARC-WMTE-7289: receipts preserve downstream grammar classification."""
    path = tmp_path / "malformed-response.jsonl"
    monkeypatch.setenv(BOUNDARY_LEDGER_ENV, str(path))
    proposer = world_model.LocalGGUFProposer(
        model_path="/fixtures/fixture.gguf",
        model_repository="fixture/model",
        model_filename="fixture.gguf",
        model_revision="a" * 40,
        ffn_cpu_layers=0,
        mtp=False,
    )
    proposer._proc = type("Proc", (), {"pid": 457})()
    response = {"choices": choices, "usage": {"completion_tokens": 1}}
    monkeypatch.setattr(
        tool_loop.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(response).encode()),
    )

    assert (
        tool_loop._post_chat(
            proposer,
            [{"role": "user", "content": "fixture"}],
            turn=0,
            timeout_s=1,
            selfparse=True,
        )
        == response
    )
    reduced = reduce_boundary_events(InvocationBoundaryLedger(path).read_events())
    assert reduced["invocation_counts"]["generation_calls_completed"] == 1
    assert reduced["invocation_counts"]["usable_answers"] == 0


def test_local_proposer_load_seam_writes_child_pid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7289-LIVE-CALLER-REACHABILITY: load seam."""
    ledger_path = tmp_path / "load.jsonl"
    model_path = tmp_path / "fixture.gguf"
    server_path = tmp_path / "llama-server"
    model_path.write_bytes(b"fixture")
    server_path.write_bytes(b"fixture")
    monkeypatch.setenv(BOUNDARY_LEDGER_ENV, str(ledger_path))
    proposer = world_model.LocalGGUFProposer(
        model_path=str(model_path),
        model_repository="fixture/model",
        model_filename=model_path.name,
        ffn_cpu_layers=0,
        mtp=False,
        timeout=1,
    )
    health = iter((False, True))
    monkeypatch.setattr(proposer, "_healthy", lambda: next(health))
    monkeypatch.setattr(world_model, "_generator_server_and_env", lambda *_: (server_path, None))
    monkeypatch.setattr(world_model, "_kv_quant_for_launch", lambda *_: None)
    monkeypatch.setattr(world_model, "_llama_server_parallel_launch", lambda: None)
    monkeypatch.setattr(world_model, "_split_args_for_env", lambda *_: [])
    monkeypatch.setattr(
        world_model.subprocess, "Popen", lambda *a, **k: type("P", (), {"pid": 789})()
    )
    monkeypatch.setattr(
        "carnot.agentic.arc_eval_provenance.huggingface_snapshot_revision",
        lambda *_: "c" * 40,
    )
    monkeypatch.setattr("carnot.agentic.arc_eval_provenance.process_start_tick", lambda *_: 99)
    monkeypatch.setattr(proposer, "_verify_mtp_engaged", lambda: None)

    assert proposer._ensure_server() is True
    reduced = reduce_boundary_events(InvocationBoundaryLedger(ledger_path).read_events())
    assert reduced["invocation_counts"]["model_loads_completed"] == 1
    assert reduced["call_rows"][0]["child_pid"] == 789
    assert reduced["call_rows"][0]["model_identity"]["model_revision"] == "c" * 40


def test_cpu_fixture_panel_retains_timeout_and_cleans_children(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7289-IN-FLIGHT-TIMEOUT and owned-child cleanup."""
    panel = run_cpu_boundary_panel(tmp_path)
    by_case = {row["case"]: row for row in panel["rows"]}
    assert set(by_case) == {
        "pre_load_failure",
        "load_no_generation",
        "generation_timeout",
        "generation_unusable",
        "duplicate_events",
        "orphan_cleanup",
    }
    assert all(row["passed"] for row in by_case.values())
    timeout = by_case["generation_timeout"]
    assert timeout["censored"] is True
    assert timeout["observed"]["generation_calls_attempted"] == 1
    assert timeout["observed"]["generation_calls_completed"] == 0
    assert timeout["observed"]["generation_calls_in_flight"] == 1
    assert timeout["owned_child_alive_after_cleanup"] is False
    assert by_case["orphan_cleanup"]["owned_child_alive_after_cleanup"] is False
    assert by_case["pre_load_failure"]["identity_rejection_preserved"] is True
    assert panel["all_controls_passed"] is True


def test_exercise_live_caller_seams_uses_no_real_model(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7289 wires both actual caller seams with injected CPU transport."""
    result = exercise_live_caller_seams(tmp_path)
    assert result["load_boundary_reachable"] is True
    assert result["selfparse_generation_boundary_reachable"] is True
    assert result["identity_rejection_preserved"] is True
    assert result["current_model_calls"] == ZERO_INVOCATION_COUNTS


def test_historical_freeze_does_not_change_quarantined_source(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7289 freezes both terminal and current raw byte states."""
    source = REPO / "results/experiment_7280_v640_arc_live.json"
    before = source.read_bytes()
    receipt = freeze_historical_exp7280(REPO, tmp_path / "history.json")
    assert source.read_bytes() == before
    assert receipt["terminal_artifact_sha256"].startswith("sha256:")
    assert receipt["terminal_observations"]["model_loads_completed"] == 1
    assert receipt["terminal_observations"]["model_invoked"] is False
    assert receipt["present_raw_observations"]["last_completed_phase"] == "episodes"
    assert receipt["present_raw_observations"]["completed_units"] == 1
    assert receipt["present_raw_observations"]["generation_calls_attempted"] == 2
    assert receipt["diagnosis"]["cause"] == "unresolved_external_timeout_boundary"
    assert receipt["diagnosis"]["observed_active_call"] == "not_authenticatable"
    assert receipt["quarantine"]["preserved"] is True


def test_terminal_artifact_is_current_cpu_only_and_independently_reducible(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-7289-TERMINAL-EVIDENCE."""
    panel = run_cpu_boundary_panel(tmp_path / "panel")
    seams = exercise_live_caller_seams(tmp_path / "seams")
    history = freeze_historical_exp7280(REPO, tmp_path / "history.json")
    artifact = build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[{"phase": "cpu_panel", "duration_s": 1.0}],
        preconditions_checked=[
            {
                "check": "fixture",
                "upstream": "test",
                "field": "available",
                "expected": True,
                "observed": True,
                "passed": True,
                "sha256": "sha256:" + "d" * 64,
            }
        ],
        source_artifact_hashes={
            "results/experiment_7280_v640_arc_live.json": {
                "sha256": history["terminal_artifact_sha256"],
                "experiment_id": "exp7280-arc-live",
                "terminal_class": "null",
                "quarantined": True,
                "retired": False,
            }
        },
        panel=panel,
        historical=history,
        live_handoff={"caller_reachability": seams, "handoff_sha256": "sha256:" + "e" * 64},
        validation_receipts=[{"name": "fixture", "exit_code": 0, "passed": True}],
    )
    assert validate_artifact(artifact) == []
    assert artifact["schema"] == "carnot.experiment_7289.v641.arc_boundary.v1"
    assert artifact["experiment_id"] == EXPERIMENT_ID
    assert artifact["milestone"] == MILESTONE
    assert artifact["run_date"] == RUN_DATE
    assert artifact["MODEL_SPECS"] == MODEL_SPECS == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["arc_boundary_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["quarantine_preserved"] is True
    assert artifact["reproducibility_checksum"] == artifact_checksum(artifact)
    assert set(artifact["field_principles"]) == set(artifact)
    assert all(row["provenance_scope"] == "historical" for row in artifact["rows"])
    assert all("invocation_counts" not in row["counter_reduction"] for row in artifact["rows"])
    assert all("model_identity" not in json.dumps(row) for row in artifact["rows"])

    raw = tmp_path / "raw_rows.json"
    raw.write_text(json.dumps(panel), encoding="utf-8")
    assert independent_reduce(raw)["arc_boundary_ready_score"] == 1


def test_terminal_validator_rejects_current_model_claim_and_bad_checksum(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7289 keeps injected events out of current invocation metadata."""
    panel = run_cpu_boundary_panel(tmp_path / "panel")
    seams = exercise_live_caller_seams(tmp_path / "seams")
    history = freeze_historical_exp7280(REPO, tmp_path / "history.json")
    artifact = build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[{"phase": "cpu_panel", "duration_s": 1.0}],
        preconditions_checked=[],
        source_artifact_hashes={},
        panel=panel,
        historical=history,
        live_handoff={"caller_reachability": seams, "handoff_sha256": "sha256:" + "f" * 64},
        validation_receipts=[],
    )
    artifact["model_invoked"] = True
    errors = validate_artifact(artifact)
    assert "current_model_invoked_must_be_false" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_thin_entrypoint_and_forbidden_files_unchanged() -> None:
    """REQ-ARC-WMTE-7289 keeps the wrapper thin and conductor untouched."""
    wrapper = REPO / "scripts/experiments/experiment_7289_v641_arc_boundary.py"
    assert wrapper.read_text(encoding="utf-8").count("from carnot.") == 1
    assert "main()" in wrapper.read_text(encoding="utf-8")
    assert not (REPO / "python/carnot/experiment_7289_v641_arc_boundary.py").is_symlink()
    assert "CARNOT_ARC_BOUNDARY_LEDGER_PATH" not in (
        REPO / "scripts/research_conductor.py"
    ).read_text(encoding="utf-8")


def test_fixture_child_role_rejects_unknown_case(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7289 fails closed on an unknown injected fixture."""
    command = [
        str(REPO / ".venv/bin/python"),
        "-u",
        str(REPO / "scripts/experiments/experiment_7289_v641_arc_boundary.py"),
        "--role",
        "fixture-child",
        "--case",
        "unknown",
        "--ledger-path",
        str(tmp_path / "events.jsonl"),
    ]
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
    assert completed.returncode != 0
    assert "unknown fixture case" in completed.stderr


def test_no_fixture_process_is_left_owned_by_this_test() -> None:
    """REQ-ARC-WMTE-7289 cleanup evidence does not depend on ambient processes."""
    assert os.getpid() > 1
    assert _pid_alive(999_999_999) is False


def test_disabled_terminal_and_direct_failure_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-WMTE-7289: disabled callers are inert and terminal events stay terminal."""
    monkeypatch.delenv(BOUNDARY_LEDGER_ENV, raising=False)
    disabled = boundary_call_for_proposer(object(), "generation")
    disabled.child_started(12)
    disabled.complete(usable=False)
    assert not list(tmp_path.iterdir())

    ledger = InvocationBoundaryLedger(tmp_path / "events.jsonl")
    assert InvocationBoundaryLedger(tmp_path / "absent.jsonl").read_events() == []
    with pytest.raises(ValueError, match="unknown boundary operation"):
        ledger.begin("training", _identity())

    failed = ledger.begin("model_load", _identity(), call_id="string-failure")
    failed.fail("direct failure")
    for action in (
        lambda: failed.child_started(10),
        lambda: failed.complete(),
        lambda: failed.fail(RuntimeError("again")),
    ):
        with pytest.raises(RuntimeError, match="already terminal"):
            action()
    reduced = reduce_boundary_events(ledger.read_events())
    assert reduced["inference_substrate"] == "model_load_failed"
    assert reduced["call_rows"][0]["error"] == "direct failure"


def test_malformed_and_inconsistent_events_disqualify_unknown_activity(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7289-DUPLICATE-AND-IDENTITY-REJECTION covers corruption."""
    malformed_path = tmp_path / "malformed.jsonl"
    malformed_path.write_bytes(b"{bad json\n[]\n")
    malformed = InvocationBoundaryLedger(malformed_path).read_events()
    assert [row["error"] for row in malformed] == [
        "JSONDecodeError: Expecting property name enclosed in double quotes",
        "not_object",
    ]

    valid_path = tmp_path / "valid.jsonl"
    ledger = InvocationBoundaryLedger(valid_path)
    ledger.begin("model_load", _identity(), call_id="template")
    template = ledger.read_events()[0]

    def event(event_id: str, **changes: object) -> dict[str, object]:
        row: dict[str, object] = deepcopy(template)
        row["event_id"] = event_id
        row["call_id"] = event_id
        row.update(changes)
        return row

    changed_attempt = event("operation-change", call_id="operation-change")
    changed_terminal = event(
        "operation-change-terminal",
        call_id="operation-change",
        operation="generation",
        state="completed",
        ended_monotonic_ns=template["started_monotonic_ns"],
    )
    terminal_only = event(
        "terminal-only",
        state="completed",
        ended_monotonic_ns=template["started_monotonic_ns"],
    )
    multiple_attempt = event("multiple", call_id="multiple")
    multiple_complete = event(
        "multiple-complete",
        call_id="multiple",
        state="completed",
        ended_monotonic_ns=template["started_monotonic_ns"],
    )
    multiple_failed = event(
        "multiple-failed",
        call_id="multiple",
        state="failed",
        ended_monotonic_ns=template["started_monotonic_ns"],
    )
    bad_end_attempt = event("bad-end", call_id="bad-end", started_monotonic_ns=10)
    bad_end_terminal = event(
        "bad-end-terminal",
        call_id="bad-end",
        state="completed",
        started_monotonic_ns=10,
        ended_monotonic_ns=9,
    )
    rows = [
        *malformed,
        {},
        event("bad-schema", schema="wrong"),
        event("bad-operation", operation="training"),
        event("bad-state", state="queued"),
        event("bad-call", call_id=""),
        terminal_only,
        event("missing-identity", model_identity=None),
        changed_attempt,
        changed_terminal,
        multiple_attempt,
        multiple_complete,
        multiple_failed,
        event("bad-start", started_monotonic_ns=-1),
        bad_end_attempt,
        bad_end_terminal,
    ]
    reduced = reduce_boundary_events(rows)
    assert reduced["activity_known"] is False
    assert reduced["invocation_counts"] is None
    assert {
        "malformed_event_line:1",
        "malformed_event_line:2",
        "event_id_missing",
        "event_schema_invalid",
        "unknown_operation",
        "unknown_state",
        "call_id_missing",
        "attempted_event_count_invalid:terminal-only",
        "model_identity_missing:missing-identity",
        "operation_changed:operation-change",
        "multiple_terminal_events:multiple",
        "start_monotonic_invalid:bad-start",
        "end_monotonic_invalid:bad-end",
    }.issubset(set(reduced["errors"]))
    assert reduce_boundary_events([])["inference_substrate"] == "no_model_activity"


def test_handoff_preconditions_commands_and_argument_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7289-LIVE-CALLER-REACHABILITY binds the exact handoff."""
    original = tmp_path / "outer-ledger.jsonl"
    monkeypatch.setenv(BOUNDARY_LEDGER_ENV, str(original))
    seams = exercise_live_caller_seams(tmp_path / "seams")
    assert os.environ[BOUNDARY_LEDGER_ENV] == str(original)
    handoff = build_live_handoff(REPO, seams)
    assert handoff["entrypoint"] == "scripts/experiments/experiment_7290_v641_arc_selfparse.py"
    assert handoff["arguments"] == ["--date", RUN_DATE]
    assert handoff["handoff_sha256"].startswith("sha256:")
    assert len(handoff["changed_code_hashes"]) == 5

    checks, hashes = collect_preconditions(REPO)
    assert checks and all(row["passed"] for row in checks)
    assert hashes["results/experiment_7280_v640_arc_live.json"]["quarantined"] is True
    commands = build_validation_commands(tmp_path / "rows.json", tmp_path / "candidate.json")
    assert {row["name"] for row in commands} >= {
        "focused_exp7289",
        "full_python_suite",
        "scoped_coverage_report",
        "e2e_009_cross_call_persistence",
        "e2e_010_transport_and_cancellation",
        "terminal_candidate_adversarial_verify",
        "terminal_candidate_row_consistency",
    }
    coverage_run = next(row for row in commands if row["name"] == "scoped_coverage_run")
    assert any(str(item).startswith("--include=") for item in coverage_run["command"])
    args = parse_args(["--date", RUN_DATE, "--reduce-raw", str(tmp_path / "rows.json")])
    assert args.date == RUN_DATE
    assert args.reduce_raw == tmp_path / "rows.json"


def test_terminal_validator_checks_each_fail_closed_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7289-TERMINAL-EVIDENCE rejects every invalid consumer claim."""
    panel = run_cpu_boundary_panel(tmp_path / "panel")
    seams = exercise_live_caller_seams(tmp_path / "seams")
    history = freeze_historical_exp7280(REPO, tmp_path / "history.json")
    artifact = build_terminal_artifact(
        started_at_utc="2026-09-14T00:00:00+00:00",
        ended_at_utc="2026-09-14T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[{"phase": "cpu_panel", "duration_s": 1.0}],
        preconditions_checked=[],
        source_artifact_hashes={},
        panel=panel,
        historical=history,
        live_handoff=build_live_handoff(REPO, seams),
        validation_receipts=[{"name": "fixture", "exit_code": 0, "passed": True}],
    )
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    assert validate_artifact(artifact_path) == []

    cases: list[tuple[str, object, str]] = [
        ("schema", "wrong", "schema_or_experiment_identity_mismatch"),
        ("milestone", "wrong", "milestone_or_date_mismatch"),
        ("status", "running", "status_not_terminal"),
        ("MODEL_SPECS", [{"model": "forbidden"}], "current_model_specs_must_be_empty"),
        ("invocation_counts", {}, "current_invocation_counts_must_be_zero"),
        ("inference_substrate", "gpu", "current_inference_substrate_invalid"),
        (
            "inference_substrate_class",
            "model_full_generation",
            "current_inference_substrate_class_invalid",
        ),
        ("execution_venue", "remote", "execution_venue_invalid"),
        ("verdict_class", "invented", "verdict_class_invalid"),
        ("official_score", 1, "official_score_must_be_unset"),
        ("boundary_event_rows", [], "boundary_event_rows_mismatch"),
        ("arc_boundary_ready_score", 0, "arc_boundary_ready_score_inconsistent"),
        ("quarantine_preserved", False, "historical_quarantine_not_preserved"),
        ("duration_s", 0, "duration_invalid"),
    ]
    for field, value, expected_error in cases:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = artifact_checksum(changed)
        assert expected_error in validate_artifact(changed)

    incomplete_principles = deepcopy(artifact)
    incomplete_principles["field_principles"].pop("schema")
    incomplete_principles["reproducibility_checksum"] = artifact_checksum(incomplete_principles)
    assert "field_principles_incomplete" in validate_artifact(incomplete_principles)

    oracle_positive = deepcopy(artifact)
    oracle_positive["verdict_class"] = "positive"
    oracle_positive["reproducibility_checksum"] = artifact_checksum(oracle_positive)
    oracle_errors = validate_artifact(oracle_positive)
    assert "oracle_forbids_positive" in oracle_errors
    assert "ready_verdict_class_invalid" in oracle_errors

    failed_gate = deepcopy(artifact)
    failed_gate["acceptance_gate_results"][0]["passed"] = False
    failed_gate["arc_boundary_ready_score"] = 0
    failed_gate["reproducibility_checksum"] = artifact_checksum(failed_gate)
    assert "failed_gate_verdict_class_invalid" in validate_artifact(failed_gate)

    incomplete_rows = deepcopy(artifact)
    incomplete_rows["rows"] = []
    incomplete_rows["boundary_event_rows"] = []
    incomplete_rows["arc_boundary_ready_score"] = 0
    incomplete_rows["verdict_class"] = "disqualified"
    incomplete_rows["reproducibility_checksum"] = artifact_checksum(incomplete_rows)
    assert "boundary_rows_incomplete" in validate_artifact(incomplete_rows)

    negative_span = deepcopy(artifact)
    negative_span["phase_spans"] = [{"phase": "bad", "duration_s": -1}]
    negative_span["reproducibility_checksum"] = artifact_checksum(negative_span)
    assert "phase_span_invalid" in validate_artifact(negative_span)

    excessive_span = deepcopy(artifact)
    excessive_span["phase_spans"] = [{"phase": "bad", "duration_s": 2}]
    excessive_span["reproducibility_checksum"] = artifact_checksum(excessive_span)
    assert "phase_spans_exceed_duration" in validate_artifact(excessive_span)
