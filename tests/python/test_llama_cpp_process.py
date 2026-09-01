"""REQ-INFERENCE-6850 owned llama.cpp process lifecycle tests."""

from __future__ import annotations

import signal
from typing import Any

from carnot.inference import llama_cpp_process as process


class FakeProcessOps:
    """Record signals so tests can prove cleanup never broadens its target."""

    def __init__(self, wait_statuses: list[str] | None = None) -> None:
        self.wait_statuses = list(wait_statuses or ["exited"])
        self.signals: list[tuple[int, signal.Signals, bool]] = []

    def send_signal(self, pid: int, sig: signal.Signals, *, process_group: bool) -> None:
        self.signals.append((pid, sig, process_group))

    def wait_for_exit(self, pid: int, timeout_s: float) -> str:
        assert pid > 1
        assert timeout_s > 0
        return self.wait_statuses.pop(0)


def _identity(*, start: int = 100, command_hash: str = "sha256:command") -> dict[str, Any]:
    return {
        "pid": 4100,
        "exists": True,
        "start_time_ticks": start,
        "uid": 1000,
        "command_hash": command_hash,
        "process_group_id": 4100,
        "parent_identity": {"pid": 4000, "start_time_ticks": 90},
    }


def _receipt(token: str = "owned-secret") -> dict[str, Any]:
    return {
        **_identity(),
        "owned_by_task": True,
        "owner_pid": 4000,
        "owner_start_time_ticks": 90,
        "ownership_token_digest": process.ownership_token_digest(token),
        "port": 9123,
    }


def test_scenario_6850_stale_pid_is_already_exited_without_signal() -> None:
    """SCENARIO-INFERENCE-6850-PROCESS-IDENTITY treats an absent PID as stale."""

    ops = FakeProcessOps()
    receipt = process.cleanup_owned_process(
        _receipt(),
        token="owned-secret",
        current_identity=lambda pid: {"pid": pid, "exists": False},
        process_ops=ops,
        port_probe=lambda port: True,
        contract=process.process_contract(),
    )

    assert receipt["action"] == "already_exited"
    assert receipt["process_exit_confirmed"] is True
    assert receipt["port_release_confirmed"] is True
    assert receipt["unrelated_process_kill_count_delta"] == 0
    assert ops.signals == []


def test_scenario_6850_pid_reuse_refuses_cleanup_without_signal() -> None:
    """SCENARIO-INFERENCE-6850-PROCESS-IDENTITY rejects changed start ticks."""

    ops = FakeProcessOps()
    receipt = process.cleanup_owned_process(
        _receipt(),
        token="owned-secret",
        current_identity=lambda pid: _identity(start=101),
        process_ops=ops,
        port_probe=lambda port: False,
        contract=process.process_contract(),
    )

    assert receipt["action"] == "refused"
    assert "start_time_ticks" in receipt["ownership_errors"]
    assert receipt["port_release_confirmed"] is False
    assert ops.signals == []


def test_scenario_6850_occupied_port_never_targets_unrelated_listener() -> None:
    """SCENARIO-INFERENCE-6850-PORT-AND-ORPHAN blocks an unowned listener."""

    ops = FakeProcessOps()
    readiness = process.prepare_owned_port(
        9123,
        orphan_receipt=None,
        token=None,
        current_identity=lambda pid: _identity(),
        process_ops=ops,
        port_probe=lambda port: False,
        contract=process.process_contract(),
    )

    assert readiness["ready"] is False
    assert readiness["reason"] == "occupied_port_unowned"
    assert readiness["signals_sent"] == []
    assert ops.signals == []


def test_scenario_6850_matching_owned_orphan_is_reclaimed() -> None:
    """SCENARIO-INFERENCE-6850-PORT-AND-ORPHAN reclaims only a full identity match."""

    ops = FakeProcessOps()
    probes = iter([False, True])
    readiness = process.prepare_owned_port(
        9123,
        orphan_receipt=_receipt(),
        token="owned-secret",
        current_identity=lambda pid: _identity(),
        process_ops=ops,
        port_probe=lambda port: next(probes),
        contract=process.process_contract(),
    )

    assert readiness["ready"] is True
    assert readiness["owned_orphan_recovered"] is True
    assert readiness["cleanup"]["ownership_verified"] is True
    assert readiness["cleanup"]["port_release_confirmed"] is True
    assert ops.signals == [(4100, signal.SIGTERM, True)]


def test_scenario_6850_wrong_ownership_token_refuses_teardown() -> None:
    """SCENARIO-INFERENCE-6850-TEARDOWN binds cleanup to the opaque token."""

    ops = FakeProcessOps()
    receipt = process.cleanup_owned_process(
        _receipt(),
        token="wrong-secret",
        current_identity=lambda pid: _identity(),
        process_ops=ops,
        port_probe=lambda port: False,
        contract=process.process_contract(),
    )

    assert receipt["action"] == "refused"
    assert receipt["ownership_errors"] == ["ownership_token_digest"]
    assert receipt["unrelated_process_kill_count_delta"] == 0
    assert ops.signals == []


def test_scenario_6850_owned_teardown_is_bounded_and_port_checked() -> None:
    """SCENARIO-INFERENCE-6850-TEARDOWN confirms exit and port release."""

    ops = FakeProcessOps(["timeout", "exited"])
    receipt = process.cleanup_owned_process(
        _receipt(),
        token="owned-secret",
        current_identity=lambda pid: _identity(),
        process_ops=ops,
        port_probe=lambda port: True,
        contract=process.process_contract(cleanup_grace_s=0.1, kill_timeout_s=0.1),
    )

    assert receipt["action"] == "force_killed"
    assert receipt["ownership_verified"] is True
    assert receipt["process_exit_confirmed"] is True
    assert receipt["port_release_confirmed"] is True
    assert receipt["leak_free"] is True
    assert [item[1] for item in ops.signals] == [signal.SIGTERM, signal.SIGKILL]


def test_req_inference_6850_identity_contract_names_all_authority_mismatches() -> None:
    """REQ-INFERENCE-6850 requires every cleanup authority field to match."""

    recorded = _receipt("secret")
    recorded["owned_by_task"] = False
    errors = process.ownership_errors(recorded, {"parent_identity": "malformed"}, token="secret")

    assert errors == [
        "owned_by_task",
        "pid",
        "start_time_ticks",
        "uid",
        "command_hash",
        "process_group_id",
        "owner_pid",
        "owner_start_time_ticks",
    ]


def test_req_inference_6850_free_port_needs_no_orphan_cleanup() -> None:
    """REQ-INFERENCE-6850 admits an already-free exact port without signaling."""

    ops = FakeProcessOps()
    readiness = process.prepare_owned_port(
        9123,
        orphan_receipt=None,
        token=None,
        current_identity=lambda pid: _identity(),
        process_ops=ops,
        port_probe=lambda port: True,
        contract=process.process_contract(),
    )

    assert readiness == {
        "ready": True,
        "port": 9123,
        "owned_orphan_recovered": False,
        "signals_sent": [],
    }
    assert ops.signals == []
