"""CPU validation for bound selfparse result resumption.

Spec refs: REQ-ARC-WMTE-7336 and SCENARIO-ARC-WMTE-7336-*.
"""

from __future__ import annotations

from copy import deepcopy
from argparse import Namespace
import json
import os
from pathlib import Path
import runpy
import time

import pytest

from carnot import experiment_7336_v644_arc_resume as exp
from carnot.agentic.arc_selfparse_result_resume import ResultResumeGuard


REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.memory_watchdog_skip


def test_req_7336_spec_identity_and_historical_first_loss() -> None:
    """SCENARIO-ARC-WMTE-7336-FIRST-LOSS binds the first absent transition."""

    spec = (REPO / exp.SPEC_PATH).read_text(encoding="utf-8")
    assert "## REQ-ARC-WMTE-7336:" in spec
    assert "SCENARIO-ARC-WMTE-7336-BOUND-EXACTLY-ONCE-RESUME" in spec
    assert exp.RUN_DATE == "20260916"
    assert exp.MILESTONE == "2026.09.644"
    assert exp.MODEL_SPECS == []

    receipt = exp.trace_exp7319_first_loss(REPO)
    assert receipt["reproduced"] is True
    assert receipt["first_lost_link"] == "next_request_payload_delivery"
    assert receipt["source_request_index"] == 1
    assert receipt["source_request_sha256"] == (
        "sha256:0c04846deb5bdb39f08cb58268fe2eacf45f6857be6bef15af9e2ee650cde09e"
    )
    assert receipt["source_response_sha256"] == (
        "sha256:845ac466e24f50ec1ae049f101af43017cf15c02943f939abe8261ae3f022d5e"
    )
    assert receipt["tool_event_sha256"] == (
        "sha256:dc85e81001ee56f9a795b321679938d65b1e494ddbc79c82aced2de5ae96405b"
    )
    assert receipt["tool_name"] == "list_transitions"
    assert receipt["runtime_result_ok"] is True
    assert receipt["payload_created"] is True
    assert receipt["payload_delivered_in_later_request"] is False
    assert receipt["receipt_captured"] is False
    assert receipt["completion_limit"] == 2
    assert receipt["completed_calls"] == 2
    assert receipt["remaining_call_budget"] == 0
    assert receipt["generated_tokens"] == 2087
    assert receipt["generated_token_limit"] == 4096
    assert receipt["token_budget_exhausted"] is False
    assert receipt["exit_reason"] == "episode_generation_call_limit_reached"


def _guard(now: float | None = None) -> ResultResumeGuard:
    instant = time.monotonic() if now is None else now
    return ResultResumeGuard(
        episode_id="game:episode",
        attempt_id="attempt:0",
        completion_limit=2,
        authority_expires_monotonic=instant + 30.0,
    )


def _offer(guard: ResultResumeGuard) -> dict[str, object]:
    return guard.offer_result(
        source_request_id="request:0",
        next_request_id="request:1",
        tool_names=["diff_grids"],
        bounded_response='<tool_response>\n{"ok": true, "after": 1}\n</tool_response>',
        dispatch_results=[{"ok": True, "after": 1}],
    )


def test_scenario_7336_guard_reserves_and_delivers_exactly_once() -> None:
    """SCENARIO-ARC-WMTE-7336-BOUND-EXACTLY-ONCE-RESUME pins delivery state."""

    now = time.monotonic()
    guard = _guard(now)
    assert guard.normal_request_allowed(completed_calls=0) is True
    assert guard.normal_request_allowed(completed_calls=1) is False
    offered = _offer(guard)
    assert offered["accepted"] is True

    payload = guard.prepare_next_request(
        request_id="request:1",
        episode_id="game:episode",
        attempt_id="attempt:0",
        now_monotonic=now + 1.0,
    )
    assert payload is not None
    assert payload.count("<tool_response>") == 1
    assert (
        guard.prepare_next_request(
            request_id="request:1",
            episode_id="game:episode",
            attempt_id="attempt:0",
            now_monotonic=now + 1.1,
        )
        is None
    )
    assert (
        guard.complete_request(request_id="request:1", response_received=True, timed_out=False)
        is True
    )
    receipt = guard.receipt()
    assert receipt["reserved_completion_slots"] == 1
    assert receipt["result_rows"][0]["delivery_count"] == 1
    assert receipt["result_rows"][0]["receipt_captured"] is True
    assert receipt["result_rows"][0]["state"] == "receipt_captured"
    assert receipt["result_rows"][0]["can_authorize_plan"] is False
    assert receipt["rejections"][-1]["reason"] == "duplicate_result_delivery"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("stale_episode", "stale_episode"),
        ("mismatched_attempt", "mismatched_attempt"),
        ("expired_authority", "expired_authority"),
        ("duplicate_result", "duplicate_result"),
        ("timeout_after_dispatch", "timeout_after_dispatch"),
        ("absent_result", "absent_result"),
    ],
)
def test_scenario_7336_rejection_matrix(case: str, expected: str) -> None:
    """SCENARIO-ARC-WMTE-7336-REJECTION-MATRIX keeps distinct failure reasons."""

    rows = {row["case"]: row for row in exp.run_rejection_panel()}
    assert rows[case]["reason"] == expected
    assert rows[case]["accepted"] is False
    assert rows[case]["plan_authorized"] is False
    assert rows[case]["attempted_calls_retained"] >= 1


def test_scenario_7336_actual_e3_result_to_action_and_controls(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7336-CONTROLS uses E3AgentPolicy and local dispatch."""

    receipt = exp.run_resume_control_panel(tmp_path / "panel")
    rows = {row["arm"]: row for row in receipt["rows"]}
    changed = rows["tool_needed_changed_input"]
    assert changed["policy_class"] == "E3AgentPolicy"
    assert changed["completion_limit"] == 2
    assert changed["completion_calls"] == 2
    assert changed["generated_token_limit"] == 4096
    assert changed["result_delivery_count"] == 1
    assert changed["receipt_captured"] is True
    assert changed["later_request_result_occurrences"] == 1
    assert changed["input_changed"] is True
    assert changed["verified_engine_installed"] is True
    assert changed["plan_installed"] is True
    assert changed["later_policy_action"] is True
    assert changed["passed"] is True

    withheld = rows["result_withheld"]
    assert withheld["successful_runtime_results"] == 1
    assert withheld["result_delivery_count"] == 0
    assert withheld["receipt_captured"] is False
    assert withheld["plan_installed"] is False
    assert withheld["passed"] is True

    no_tool = rows["no_tool_needed"]
    assert no_tool["completion_calls"] == 1
    assert no_tool["successful_runtime_results"] == 0
    assert no_tool["result_delivery_count"] == 0
    assert no_tool["plan_installed"] is True
    assert no_tool["later_policy_action"] is True
    assert no_tool["passed"] is True

    assert receipt["counts_as_current_model_invocation"] is False
    sidecar = Path(receipt["fixture_sidecar_path"])
    assert sidecar.is_file()
    assert receipt["fixture_sidecar_sha256"] == exp.sha256_file(sidecar)
    fixture = json.loads(sidecar.read_text(encoding="utf-8"))
    assert fixture["counts_as_current_model_invocation"] is False
    assert fixture["events"]


def _passing_validation_receipts() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "command": f"fixture {name}",
            "scope": "fixture",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": f"sha256:{index:064x}",
            "passed": True,
            "timed_out": False,
        }
        for index, name in enumerate(exp.REQUIRED_VALIDATION_NAMES, start=1)
    ]


def _e2e_receipts() -> list[dict[str, object]]:
    return [
        {
            "name": name,
            "command": f"fixture {name}",
            "scope": "offline_cpu",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": f"sha256:{index + 20:064x}",
            "passed": True,
            "timed_out": False,
        }
        for index, name in enumerate(exp.REQUIRED_E2E_NAMES, start=1)
    ]


def test_scenario_7336_terminal_artifact_and_independent_reduction(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7336-TERMINAL fails closed and checksum-binds evidence."""

    first_loss = exp.trace_exp7319_first_loss(REPO)
    panel = exp.run_resume_control_panel(tmp_path / "panel")
    rejection_rows = exp.run_rejection_panel()
    raw_rows = [*panel["rows"], *rejection_rows]
    raw_path = tmp_path / "raw_rows.json"
    exp.atomic_write(raw_path, {"rows": raw_rows})
    artifact = exp.build_terminal_artifact(
        first_loss=first_loss,
        control_panel=panel,
        rejection_rows=rejection_rows,
        validation_receipts=_passing_validation_receipts(),
        e2e_receipts=_e2e_receipts(),
        terminal_lint_receipts=[
            {"name": "adversarial_verify", "passed": True, "exit_code": 0},
            {"name": "verdict_row_consistency_strict", "passed": True, "exit_code": 0},
        ],
        source_hashes={
            exp.EXP7319_RESULT_PATH.as_posix(): {
                "sha256": exp.sha256_file(REPO / exp.EXP7319_RESULT_PATH),
                "role": "historical_diagnostic_source",
                "authorizes_readiness": False,
            }
        },
        raw_rows_path=raw_path,
        started_at_utc="2026-09-16T12:00:00+00:00",
        ended_at_utc="2026-09-16T12:00:01+00:00",
        duration_s=1.0,
        phase_spans=[
            {
                "phase": "cpu_panel",
                "started_offset_s": 0.0,
                "ended_offset_s": 1.0,
                "duration_s": 1.0,
                "completed_units": 9,
                "checkpoint_position": "cpu_panel_complete",
                "pending_operations": [],
            }
        ],
    )
    assert exp.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["arc_resume_ready_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["solve_provenance"] == "no_game_solve_cpu_transport_fixture"
    assert artifact["production_default_changed"] is False
    assert set(artifact) <= set(artifact["field_principles"])
    assert exp.independent_reduce(raw_path)["arc_resume_ready_score"] == 1

    for field, value, expected_error in (
        ("model_invoked", True, "current model declaration mismatch"),
        ("arc_resume_ready_score", 0, "readiness reduction mismatch"),
        ("solve_provenance", "live_agent_self_discovery", "solve provenance mismatch"),
        ("reproducibility_checksum", "sha256:" + "0" * 64, "checksum mismatch"),
    ):
        bad = deepcopy(artifact)
        bad[field] = value
        assert expected_error in exp.validate_artifact(bad)

    blocked = deepcopy(artifact)
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "blocked_missing_input"
    assert any(error.startswith("unsafe readiness") for error in exp.validate_artifact(blocked))


def test_req_7336_null_when_first_loss_is_not_reproduced(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7336 skips continuation and returns a complete null."""

    missing_root = tmp_path / "missing"
    receipt = exp.trace_exp7319_first_loss(missing_root)
    assert receipt["reproduced"] is False
    assert receipt["failed_checks"]
    artifact = exp.build_null_artifact(
        first_loss=receipt,
        source_hashes={},
        started_at_utc="2026-09-16T00:00:00+00:00",
        ended_at_utc="2026-09-16T00:00:00+00:00",
        duration_s=0.001,
    )
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_resume_ready_score"] == 0
    assert artifact["resume_control_rows"] == []
    assert exp.validate_artifact(artifact) == []


def test_req_7336_thin_entrypoint_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7336 keeps the scripts entrypoint free of experiment logic."""

    called: list[list[str] | None] = []
    monkeypatch.setattr(exp, "main", lambda argv=None: called.append(argv) or 0)
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(
            str(REPO / exp.WRAPPER_PATH),
            run_name="__main__",
        )
    assert stopped.value.code == 0
    assert called == [None]


def test_req_7336_guard_remaining_fail_closed_branches() -> None:
    """REQ-ARC-WMTE-7336 covers malformed offers, stale requests, and lost replies."""

    guard = _guard(100.0)
    assert guard.reject("manual_rejection", marker=1)["marker"] == 1
    assert (
        guard.offer_result(
            source_request_id="request:0",
            next_request_id="request:1",
            tool_names=[],
            bounded_response="",
            dispatch_results=[],
        )["reason"]
        == "absent_result"
    )
    guard = _guard(100.0)
    _offer(guard)
    assert (
        guard.prepare_next_request(
            request_id="request:2",
            episode_id="game:episode",
            attempt_id="attempt:0",
            now_monotonic=101.0,
        )
        is None
    )
    assert guard.receipt()["rejections"][-1]["reason"] == "stale_request"
    assert (
        guard.complete_request(request_id="request:2", response_received=True, timed_out=False)
        is False
    )

    guard = _guard(100.0)
    _offer(guard)
    assert guard.prepare_next_request(
        request_id="request:1",
        episode_id="game:episode",
        attempt_id="attempt:0",
        now_monotonic=101.0,
    )
    assert (
        guard.complete_request(request_id="request:1", response_received=False, timed_out=False)
        is False
    )
    assert guard.receipt()["rejections"][-1]["reason"] == "response_not_received"


def test_req_7336_malformed_loaders_and_historical_later_request(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7336 treats malformed JSON as absent and detects a later payload."""

    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert exp._load_json(broken) is None
    broken.write_text("[]", encoding="utf-8")
    assert exp._load_json(broken) is None
    assert exp._load_jsonl(tmp_path / "missing.jsonl") == []

    root = tmp_path / "historical"
    raw = root / exp.EXP7319_RAW_PATH
    requests = raw / "re86__direct_selfparse/requests"
    requests.mkdir(parents=True)
    bounded = '<tool_response>\n{"ok": true}\n</tool_response>'
    (root / exp.EXP7319_RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / exp.EXP7319_RESULT_PATH).write_text(
        json.dumps(
            {
                "sample_size_budget": {"completion_limit": 2, "generated_token_limit": 4096},
                "tool_use_chain": {"results_in_later_requests": 0, "policy_consumed_results": 0},
            }
        ),
        encoding="utf-8",
    )
    (requests / "01_request.json").write_text(json.dumps({"messages": []}), encoding="utf-8")
    (requests / "01_response.json").write_text(
        json.dumps({"choices": [{"finish_reason": "stop"}]}), encoding="utf-8"
    )
    later = requests / "02_request.json"
    later.write_text(bounded, encoding="utf-8")
    (raw / "tool_events.jsonl").write_text(
        json.dumps(
            {
                "bounded_response": bounded,
                "parsed_tool": "diff_grids",
                "parsed_arguments": {"t": 0},
                "dispatch_result": {"ok": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (raw / "live_session.json").write_text(
        json.dumps(
            {
                "requests": [
                    {"request_path": str(requests / "00_request.json")},
                    {"request_path": str(requests / "01_request.json")},
                    {"request_path": str(later)},
                ],
                "episodes": [
                    {
                        "generated_tokens": 10,
                        "induction_rows": [
                            {"proposer_note": "other", "tool_gap": {"terminated_by": "turn_cap"}}
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt = exp.trace_exp7319_first_loss(root)
    assert receipt["reproduced"] is False
    assert receipt["payload_delivered_in_later_request"] is True
    assert receipt["exit_reason"] == "turn_cap"


def test_req_7336_environment_restores_existing_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7336 leaves process configuration unchanged after fixtures."""

    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "existing")
    monkeypatch.setenv("CARNOT_ARC_SELFPARSE_RESULT_RESUME", "existing")
    with exp._case_environment("tool_needed_changed_input"):
        assert os.environ["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] == "1"
        assert "CARNOT_ARC_DISABLE_INDUCTION" not in os.environ
    assert os.environ["CARNOT_ARC_DISABLE_INDUCTION"] == "existing"
    assert os.environ["CARNOT_ARC_SELFPARSE_RESULT_RESUME"] == "existing"


def test_req_7336_validator_rejects_identity_and_field_mutations(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-7336-TERMINAL checks each required declaration."""

    artifact = exp.build_null_artifact(
        first_loss={"reproduced": False, "failed_checks": []},
        source_hashes={},
        started_at_utc="2026-09-16T00:00:00+00:00",
        ended_at_utc="2026-09-16T00:00:00+00:00",
        duration_s=0.1,
    )
    mutations = [
        (lambda row: row.pop("schema"), "missing required field: schema"),
        (lambda row: row.__setitem__("schema", "bad"), "schema or experiment identity mismatch"),
        (lambda row: row.__setitem__("run_date", "bad"), "milestone or run date mismatch"),
        (
            lambda row: row.__setitem__("invocation_counts", {}),
            "current invocation counts mismatch",
        ),
        (lambda row: row.__setitem__("inference_substrate", "bad"), "inference substrate mismatch"),
        (lambda row: row.__setitem__("execution_venue", "board"), "execution venue mismatch"),
        (lambda row: row.__setitem__("field_principles", {}), "field principles mismatch"),
    ]
    for mutate, expected in mutations:
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in exp.validate_artifact(changed)


def test_req_7336_validation_command_construction_and_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ARC-WMTE-7336 uses explicit E2E, scoped, lint, and hash inputs."""

    for relative in (
        exp.SPEC_PATH,
        exp.EXP7319_RESULT_PATH,
        exp.MODULE_PATH,
        exp.RESUME_MODULE_PATH,
        exp.LOOP_MODULE_PATH,
        exp.WRAPPER_PATH,
        exp.TEST_PATH,
        Path("ops/e2e-test-plan.md"),
        Path("ops/exclusion_manifest.yaml"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    sidecar = tmp_path / "fixture.json"
    sidecar.write_text("{}", encoding="utf-8")
    hashes = exp._source_hashes(tmp_path, {"fixture_sidecar_path": str(sidecar)})
    assert hashes[str(sidecar)]["counts_as_current_model_invocation"] is False
    assert hashes[exp.EXP7319_RESULT_PATH.as_posix()]["role"] == "historical_diagnostic_source"

    captured: list[object] = []
    monkeypatch.setattr(
        exp.validation_scope,
        "run_commands",
        lambda root, commands, **kwargs: captured.extend(commands) or [{"name": "ok"}],
    )
    assert exp._run_e2e(tmp_path, tmp_path / "private") == [{"name": "ok"}]
    assert [row.name for row in captured] == list(exp.REQUIRED_E2E_NAMES)
    lint_specs = exp._terminal_lint_specs(tmp_path, tmp_path / "candidate.json")
    assert [row.name for row in lint_specs] == [
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    monkeypatch.setattr(
        exp.validation_scope,
        "run_scoped_validation",
        lambda *args, **kwargs: {"validation_receipts": [{"name": "focused_pytest"}]},
    )
    assert exp._run_scoped_validation(tmp_path, tmp_path / "private") == [
        {"name": "focused_pytest"}
    ]
    exp._progress(time.monotonic(), "test", "boundary", units=1)
    assert "phase=test event=boundary" in capsys.readouterr().out


def test_req_7336_run_experiment_success_null_and_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7336 publishes null early and validates successful candidates."""

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    writes: list[Path] = []
    original_atomic = exp.atomic_write

    def recording_write(path: Path, value: dict[str, object]) -> None:
        writes.append(path)
        original_atomic(path, value)

    monkeypatch.setattr(exp, "atomic_write", recording_write)
    monkeypatch.setattr(exp, "trace_exp7319_first_loss", lambda root: {"reproduced": False})
    null = exp.run_experiment(Namespace(date=exp.RUN_DATE))
    assert null["arc_resume_ready_score"] == 0
    assert tmp_path / exp.RESULT_PATH in writes

    control_rows = [
        {"unit": "u", "arm": arm, "passed": True, "failures": [], "abstentions": 0}
        for arm in ("tool_needed_changed_input", "result_withheld", "no_tool_needed")
    ]
    rejection_rows = [
        {"unit": "u", "arm": arm, "passed": True, "failures": [], "abstentions": 1}
        for arm in (
            "stale_episode",
            "mismatched_attempt",
            "expired_authority",
            "duplicate_result",
            "timeout_after_dispatch",
            "absent_result",
        )
    ]
    monkeypatch.setattr(exp, "trace_exp7319_first_loss", lambda root: {"reproduced": True})
    monkeypatch.setattr(
        exp,
        "run_resume_control_panel",
        lambda path: {
            "rows": control_rows,
            "fixture_sidecar_path": "",
            "fixture_sidecar_sha256": "",
        },
    )
    monkeypatch.setattr(exp, "run_rejection_panel", lambda: rejection_rows)
    monkeypatch.setattr(exp, "_run_scoped_validation", lambda root, private: [])
    monkeypatch.setattr(exp, "_run_e2e", lambda root, private: [])
    monkeypatch.setattr(exp, "_source_hashes", lambda root, panel: {})
    monkeypatch.setattr(
        exp.validation_scope,
        "run_commands",
        lambda *args, **kwargs: [
            {"name": "adversarial_verify", "passed": True, "exit_code": 0},
            {"name": "verdict_row_consistency_strict", "passed": True, "exit_code": 0},
        ],
    )
    candidate = {"status": "complete", "arc_resume_ready_score": 1}
    monkeypatch.setattr(exp, "build_terminal_artifact", lambda **kwargs: deepcopy(candidate))
    monkeypatch.setattr(exp, "validate_artifact", lambda value: [])
    completed = exp.run_experiment(Namespace(date=exp.RUN_DATE))
    assert completed == candidate
    assert exp.main(["--date", exp.RUN_DATE]) == 0
    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE

    monkeypatch.setattr(exp, "validate_artifact", lambda value: ["injected"])
    with pytest.raises(RuntimeError, match="terminal artifact validation failed"):
        exp.run_experiment(Namespace(date=exp.RUN_DATE))

    monkeypatch.setattr(exp, "run_experiment", lambda args: {"status": "blocked"})
    assert exp.main(["--date", exp.RUN_DATE]) == 1
