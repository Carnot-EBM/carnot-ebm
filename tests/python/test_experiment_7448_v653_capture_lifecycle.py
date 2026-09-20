"""Qualification tests for the V653 capture lifecycle repair.

Spec refs: REQ-REPORT-7448 and SCENARIO-REPORT-7448-CALLBACK/CLEANUP/
OWNERSHIP/RESTART/ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot import experiment_7448_v653_capture_lifecycle as lifecycle
from carnot import gpu_lease_phase_journal as lease_api


REPO = Path(__file__).resolve().parents[2]


def test_req_report_7448_spec_and_no_model_contract_exist() -> None:
    """REQ-REPORT-7448: the behavior contract precedes implementation."""

    text = (REPO / lifecycle.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-7448") :]
    for anchor in (
        "SCENARIO-REPORT-7448-CALLBACK",
        "SCENARIO-REPORT-7448-CLEANUP",
        "SCENARIO-REPORT-7448-OWNERSHIP",
        "SCENARIO-REPORT-7448-RESTART",
        "SCENARIO-REPORT-7448-ARTIFACT",
        "inference_substrate_class=no_model_load",
        "capture_lifecycle_ready_score",
    ):
        assert anchor in section
    assert lifecycle.MODEL_SPECS == []
    assert lifecycle.INFERENCE_SUBSTRATE_CLASS == "no_model_load"
    assert lifecycle.EXECUTION_VENUE == "host"


def test_scenario_report_7448_callback_reproduces_and_repairs_exact_shape(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7448-CALLBACK: exact raw bytes cross the typed boundary."""

    historical = REPO / lifecycle.HISTORICAL_RAW_PATH
    before = historical.read_bytes()
    fixture = lifecycle.prepare_historical_callback_fixture(REPO, tmp_path)
    assert fixture["raw_sha256"] == lifecycle.sha256_file(historical)
    assert fixture["legacy_error"] == "KeyError:'parse_status'"

    row = lifecycle.normalize_callback_row(fixture["schedule"], fixture["transport"])
    assert row["parse_status"] == "valid"
    assert row["parse_errors"] == []
    assert row["callback_disposition"] == "correct_empty"
    assert row["semantic_success"] is False
    assert row["development_usable"] is False
    assert row["raw_request_sha256"] == fixture["raw_request_sha256"]
    assert row["raw_response_sha256"] == fixture["raw_response_sha256"]
    assert row["raw_reply_sha256"] == fixture["raw_reply_sha256"]
    assert fixture["private_raw_path"].read_bytes() == before
    assert historical.read_bytes() == before
    lifecycle.assert_native_consumer_shape(row)

    malformed = deepcopy(fixture["transport"])
    malformed["raw_reply"] = "{"
    assert (
        lifecycle.normalize_callback_row(fixture["schedule"], malformed)["callback_disposition"]
        == "malformed"
    )


@pytest.mark.parametrize(
    ("case", "terminal_state"),
    [
        ("callback_exception", "callback_exception"),
        ("development_gate_closure", "development_gate_closed"),
        ("timeout", "timeout"),
        ("partial_response", "partial_response"),
    ],
)
def test_scenario_report_7448_cleanup_releases_owned_child_and_lease(
    tmp_path: Path, case: str, terminal_state: str
) -> None:
    """SCENARIO-REPORT-7448-CLEANUP: each failure exits through one finally."""

    row = lifecycle.run_owned_lifecycle_case(case, tmp_path / case)
    assert row["case"] == case
    assert row["terminal_state"] == terminal_state
    assert row["passed"] is True
    assert row["raw_persisted_before_callback"] is True
    assert row["cleanup_entered"] is True
    assert row["child_reaped"] is True
    assert row["lease_released"] is True
    assert row["foreign_signal_count"] == 0
    assert set(row["identity_binding"]) >= {
        "task_id",
        "lease_id",
        "owner_pid",
        "owner_pid_start_ticks",
        "server_pid",
        "server_pid_start_ticks",
        "boot_id",
        "clock_segments",
    }
    journal = lease_api.read_journal(Path(row["journal_path"]))
    assert journal["released"] is True
    assert journal["phase"] == "terminal_blocked"


def test_scenario_report_7448_ownership_mutations_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7448-OWNERSHIP: identity loss and reuse never recover."""

    binding = lifecycle.build_private_identity_binding(tmp_path / "binding")
    expected = deepcopy(binding)
    assert lifecycle.authenticate_recovery(binding, expected)["authenticated"] is True

    missing = deepcopy(binding)
    missing.pop("owner_pid")
    assert lifecycle.authenticate_recovery(missing, expected)["reason"] == "missing_owner_identity"

    reused = deepcopy(binding)
    reused["owner_pid_start_ticks"] += 1
    assert lifecycle.authenticate_recovery(reused, expected)["reason"] == "owner_pid_reused"

    rebooted = deepcopy(binding)
    rebooted["boot_id"] = "changed-boot-identity"
    assert lifecycle.authenticate_recovery(rebooted, expected)["reason"] == "boot_identity_changed"

    wrong_segment = deepcopy(binding)
    wrong_segment["clock_segments"][0]["end_ns"] = (
        wrong_segment["clock_segments"][0]["start_ns"] - 1
    )
    assert (
        lifecycle.authenticate_recovery(wrong_segment, expected)["reason"]
        == "clock_segment_invalid"
    )

    wrong_server = deepcopy(binding)
    wrong_server["server_pid_start_ticks"] += 1
    assert lifecycle.authenticate_recovery(wrong_server, expected)["reason"] == "server_pid_reused"


def test_scenario_report_7448_foreign_live_owner_is_not_signalled(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7448-OWNERSHIP: a real foreign live PID fails closed."""

    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        start_ticks = lease_api.proc_start_ticks(process.pid)
        assert start_ticks is not None
        expected = lifecycle.identity_evidence(
            task_id=lifecycle.TASK_ID,
            lease_id="lease:expected",
            owner_pid=os.getpid(),
            owner_pid_start_ticks=lease_api.proc_start_ticks(os.getpid()) or 0,
            server_pid=process.pid,
            server_pid_start_ticks=start_ticks,
            boot_id=lifecycle.boot_identity(),
            clock_segments=[{"segment": "fixture", "start_ns": 1, "end_ns": 2}],
        )
        foreign = deepcopy(expected)
        foreign.update(
            {
                "lease_id": "lease:foreign",
                "owner_pid": process.pid,
                "owner_pid_start_ticks": start_ticks,
            }
        )
        decision = lifecycle.authenticate_recovery(foreign, expected)
        assert decision == {
            "authenticated": False,
            "reason": "foreign_live_owner",
            "signals_sent": [],
        }
        assert process.poll() is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_scenario_report_7448_restart_conserves_request_and_ack(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7448-RESTART: two fresh processes conserve one request."""

    row = lifecycle.run_interruption_restart(REPO, tmp_path / "restart")
    assert row["passed"] is True
    assert row["first_exit_code"] == lifecycle.INTERRUPTED_EXIT_CODE
    assert row["resume_exit_code"] == 0
    assert row["raw_persisted_before_callback"] is True
    assert row["request_attempts"] == 1
    assert row["duplicate_completed_requests"] == 0
    assert row["acknowledgement_persisted"] is True
    assert row["lost_acknowledgements"] == 0
    assert row["foreign_signal_count"] == 0
    receipt = row["current_work_receipt"]
    assert receipt["model_invoked"] is False
    assert receipt["invocation_counts"] == lifecycle.ZERO_INVOCATION_COUNTS
    assert receipt["status"] == "terminal_complete"


def test_scenario_report_7448_artifact_reduces_and_rejects_mutations(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-7448-ARTIFACT: raw rows alone recompute readiness."""

    artifact = lifecycle.build_fixture_artifact(REPO, tmp_path / "artifact")
    assert lifecycle.independent_reduce_artifact(artifact, root=REPO) == []
    assert lifecycle.validate_artifact(artifact, root=REPO) == []
    assert artifact["capture_lifecycle_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"] == "complete_null_capture_lifecycle_qualified"
    assert artifact["flagged_adversarial"] is False
    assert artifact["promotion_score"] == 0
    assert artifact["model_invoked"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["computation_duration_s"] == 0.0
    assert artifact["duration_s"] > 0.0
    assert "lifecycle_duration_s" not in artifact

    mutations = {
        "score": ("capture_lifecycle_ready_score", 0, "ready_score_mismatch"),
        "model": ("model_invoked", True, "model_invoked_mismatch"),
        "history": (
            "historical_failure_hashes",
            {"experiment_artifact_sha256": "sha256:changed"},
            "historical_failure_hashes_mismatch",
        ),
        "verdict": ("verdict_class", "positive", "verdict_mismatch"),
    }
    for field, changed, expected_error in mutations.values():
        candidate = deepcopy(artifact)
        candidate[field] = changed
        assert expected_error in lifecycle.independent_reduce_artifact(candidate, root=REPO)

    changed_row = deepcopy(artifact)
    changed_row["lifecycle_rows"][0]["passed"] = False
    changed_row["rows"] = deepcopy(changed_row["lifecycle_rows"])
    assert "lifecycle_readiness_mismatch" in lifecycle.independent_reduce_artifact(
        changed_row, root=REPO
    )

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum_mismatch" in lifecycle.validate_artifact(
        bad_checksum, root=REPO
    )


def test_req_report_7448_command_plan_and_cli_guards(tmp_path: Path) -> None:
    """REQ-REPORT-7448: validation stays scoped and dates fail closed."""

    commands = lifecycle.build_validation_commands(REPO, tmp_path / "private")
    assert tuple(command.name for command in commands) == lifecycle.AFFECTED_CHECK_NAMES
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "tests/python" not in focused.argv
    assert "--no-cov" in focused.argv
    assert ("-n", "0", "-o", "addopts=") == focused.argv[1:5]
    assert lifecycle.date_argument(lifecycle.RUN_DATE) == lifecycle.RUN_DATE
    with pytest.raises(ValueError, match="run_date_mismatch"):
        lifecycle.date_argument("20260919")
    with pytest.raises(ValueError, match="unknown_lifecycle_case"):
        lifecycle.run_owned_lifecycle_case("unknown", tmp_path / "unknown")
    assert lifecycle.load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    assert lifecycle.load_object(malformed) == {}


def test_req_report_7448_callback_and_cleanup_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7448: callback and cleanup helpers fail closed on bad state."""

    fixture = lifecycle.prepare_historical_callback_fixture(REPO, tmp_path / "fixture")
    schedule = deepcopy(fixture["schedule"])
    paragraph = str(schedule["paragraph"])
    transport = deepcopy(fixture["transport"])
    transport["raw_reply"] = json.dumps({"claims": [[0, len(paragraph)]]})
    transport["finish_reason"] = "stop"
    parsed = lifecycle.normalize_callback_row(schedule, transport)
    assert parsed["callback_disposition"] == "parsed_nonempty"

    with pytest.raises(ValueError, match="historical_callback_fixture_unavailable"):
        lifecycle.prepare_historical_callback_fixture(tmp_path / "absent", tmp_path / "private")

    class DeadProcess:
        pid = 999_999_999

        @staticmethod
        def poll() -> int:
            return 1

    with pytest.raises(RuntimeError, match="owned_child_identity_unavailable"):
        lifecycle._wait_for_start_ticks(DeadProcess())  # type: ignore[arg-type]

    tick_reads = iter((None, 7))
    monkeypatch.setattr(lifecycle.lease_api, "proc_start_ticks", lambda _pid: next(tick_reads))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _seconds: None)

    class WaitingProcess:
        pid = 42

        @staticmethod
        def poll() -> None:
            return None

    assert lifecycle._wait_for_start_ticks(WaitingProcess()) == 7  # type: ignore[arg-type]
    monkeypatch.undo()

    class StubbornProcess:
        pid = 123

        def __init__(self) -> None:
            self.waits = 0
            self.killed = False

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def terminate() -> None:
            return None

        def wait(self, timeout: int) -> int:
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return 0

        def kill(self) -> None:
            self.killed = True

    stubborn = StubbornProcess()
    signals = lifecycle._stop_owned_child(stubborn)  # type: ignore[arg-type]
    assert stubborn.killed is True
    assert [row["signal"] for row in signals] == ["SIGTERM", "SIGKILL"]

    original_ticks = lease_api.proc_start_ticks
    owner_calls = 0

    def lose_owner_on_second_read(pid: int) -> int | None:
        nonlocal owner_calls
        if pid == os.getpid():
            owner_calls += 1
            if owner_calls == 2:
                return None
        return original_ticks(pid)

    monkeypatch.setattr(lifecycle.lease_api, "proc_start_ticks", lose_owner_on_second_read)
    with pytest.raises(RuntimeError, match="owner_identity_unavailable"):
        lifecycle.run_owned_lifecycle_case("timeout", tmp_path / "lost-owner")
    monkeypatch.undo()

    def unexpected_callback(*_args: Any, **_kwargs: Any) -> lifecycle.CallbackRow:
        raise RuntimeError("unexpected_callback_failure")

    monkeypatch.setattr(lifecycle, "normalize_callback_row", unexpected_callback)
    with pytest.raises(RuntimeError, match="unexpected_callback_failure"):
        lifecycle.run_owned_lifecycle_case("timeout", tmp_path / "unexpected")
    monkeypatch.undo()

    def failed_release(_self: lease_api.GpuLease, **_kwargs: Any) -> dict[str, Any]:
        raise lease_api.LeaseError("release_fixture_failure")

    monkeypatch.setattr(lease_api.GpuLease, "release", failed_release)
    with pytest.raises(lease_api.LeaseError, match="release_fixture_failure"):
        lifecycle.run_owned_lifecycle_case("timeout", tmp_path / "release-failure")
    monkeypatch.undo()

    monkeypatch.setattr(lifecycle.lease_api, "proc_start_ticks", lambda _pid: None)
    with pytest.raises(RuntimeError, match="owner_identity_unavailable"):
        lifecycle.build_private_identity_binding(tmp_path / "missing-identity")


def test_scenario_report_7448_restart_worker_fail_closed_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7448-RESTART: direct worker guards reject missing and changed raw."""

    private = tmp_path / "worker"
    assert lifecycle._fixture_worker(private, "unknown") == 2
    assert lifecycle._fixture_worker(private, "persist") == lifecycle.INTERRUPTED_EXIT_CODE
    raw = private / "raw_response.json"
    raw.write_bytes(raw.read_bytes() + b"\n")
    assert lifecycle._fixture_worker(private, "resume") == 3
    assert (
        lifecycle._fixture_worker(tmp_path / "clean", "persist") == lifecycle.INTERRUPTED_EXIT_CODE
    )
    assert lifecycle._fixture_worker(tmp_path / "clean", "resume") == 0


def test_scenario_report_7448_reducer_and_validation_fail_closed_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-7448-ARTIFACT: malformed reducers and receipts cannot pass."""

    artifact = lifecycle.build_fixture_artifact(REPO, tmp_path / "artifact")
    assert lifecycle.validate_artifact([]) == ["artifact_not_object"]

    expected = lifecycle.build_private_identity_binding(tmp_path / "identity")
    absent = deepcopy(expected)
    absent.update(
        {
            "lease_id": "lease:absent",
            "owner_pid": 999_999_999,
            "owner_pid_start_ticks": 1,
        }
    )
    assert (
        lifecycle.authenticate_recovery(absent, expected)["reason"] == "durable_identity_mismatch"
    )

    invalids: list[tuple[str, Any, str]] = [
        ("schema", "changed", "identity_mismatch:schema"),
        ("invocation_counts", {"changed": 1}, "invocation_counts_mismatch"),
        ("lifecycle_rows", None, "lifecycle_rows_invalid"),
        ("rows", [], "rows_mismatch"),
        ("acceptance_gate_results", None, "acceptance_gates_invalid"),
        ("field_principles", {}, "field_principles_mismatch"),
        ("source_artifact_hashes", None, "source_hashes_invalid"),
    ]
    for field, changed, wanted in invalids:
        candidate = deepcopy(artifact)
        candidate[field] = changed
        assert wanted in lifecycle.independent_reduce_artifact(candidate, root=REPO)

    missing_case = deepcopy(artifact)
    missing_case["lifecycle_rows"] = missing_case["lifecycle_rows"][1:]
    missing_case["rows"] = deepcopy(missing_case["lifecycle_rows"])
    assert "lifecycle_case_set_mismatch" in lifecycle.independent_reduce_artifact(
        missing_case, root=REPO
    )

    bad_summary = deepcopy(artifact)
    bad_summary["gate_check_summary"] = {}
    assert "gate_check_summary_mismatch" in lifecycle.independent_reduce_artifact(
        bad_summary, root=REPO
    )

    bad_receipt = deepcopy(artifact)
    bad_receipt["source_artifact_hashes"] = {"bad": []}
    assert "source_receipt_invalid:bad" in lifecycle.independent_reduce_artifact(
        bad_receipt, root=REPO
    )
    bad_receipt["source_artifact_hashes"] = {
        "bad": {"path": "AGENTS.md", "sha256": "sha256:changed"}
    }
    assert "source_hash_mismatch:bad" in lifecycle.independent_reduce_artifact(
        bad_receipt, root=REPO
    )

    missing_receipts = lifecycle.required_receipt_errors([], require_terminal=True)
    assert len(missing_receipts) == len(lifecycle.AFFECTED_CHECK_NAMES) + len(
        lifecycle.TERMINAL_CHECK_NAMES
    )
    failed_receipts = [
        {"name": name, "passed": False, "exit_code": 1, "timed_out": False}
        for name in lifecycle.AFFECTED_CHECK_NAMES
    ]
    assert all(
        error.startswith("receipt_failed:")
        for error in lifecycle.required_receipt_errors(failed_receipts, require_terminal=False)
    )

    declared_failure = deepcopy(artifact)
    declared_failure["validation_receipts"] = failed_receipts
    assert "receipt_failed:focused_pytest" in lifecycle.independent_reduce_artifact(
        declared_failure, root=REPO, require_terminal=False
    )

    invalid_validation_row = deepcopy(artifact)
    invalid_validation_row["validation_receipts"] = [None]
    assert "validation_receipt_invalid" in lifecycle.validate_artifact(
        invalid_validation_row, root=REPO
    )
    log = tmp_path / "validation.log"
    log.write_text("measured", encoding="utf-8")
    bad_log = deepcopy(artifact)
    bad_log["validation_receipts"] = [
        {
            "name": "extra",
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "log_path": str(log),
            "log_sha256": "sha256:changed",
        }
    ]
    assert "validation_log_hash_mismatch:extra" in lifecycle.validate_artifact(bad_log, root=REPO)

    commands = lifecycle._terminal_commands(REPO, REPO / "results/private-candidate.json")
    assert tuple(row.name for row in commands) == lifecycle.TERMINAL_CHECK_NAMES

    passing_receipts = [
        {
            "name": name,
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "duration_s": 0.1,
        }
        for name in (*lifecycle.AFFECTED_CHECK_NAMES, *lifecycle.TERMINAL_CHECK_NAMES)
    ]
    finalized = lifecycle._finalize_with_validation(
        artifact, passing_receipts, require_terminal=True
    )
    assert finalized["capture_lifecycle_ready_score"] == 1
    failed = lifecycle._finalize_with_validation(artifact, [], require_terminal=True)
    assert failed["capture_lifecycle_ready_score"] == 0
    assert failed["verdict_class"] == "disqualified"

    monkeypatch.setattr(lifecycle, "collect_preconditions", lambda _root: ([{"passed": False}], {}))
    with pytest.raises(ValueError, match="fixture_precondition_failed"):
        lifecycle.build_fixture_artifact(REPO, tmp_path / "blocked")
    monkeypatch.undo()

    monkeypatch.setattr(lifecycle, "validate_command_plan", lambda *_args: ["drift"])
    with pytest.raises(ValueError, match="invalid_validation_plan:drift"):
        lifecycle.build_validation_commands(REPO, tmp_path / "invalid-plan")
