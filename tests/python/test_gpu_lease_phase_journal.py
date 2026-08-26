"""Focused tests for the reusable task-scoped GPU lease.

Spec refs: REQ-INFRA-6633, SCENARIO-INFRA-6633-ATOMIC-RACE,
SCENARIO-INFRA-6633-INDEPENDENT-DEVICES,
SCENARIO-INFRA-6633-OWNER-AND-PHASES,
SCENARIO-INFRA-6633-FAIL-CLOSED, and
SCENARIO-INFRA-6633-CRASH-RECOVERY.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import gpu_lease_phase_journal as lease_api


REPO = Path(__file__).resolve().parents[2]


def _acquire(tmp_path: Path, *, device: str = "GPU-test-a") -> lease_api.GpuLease:
    return lease_api.GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id="exp6633-fixture",
        device_uuid=device,
        expected_model="fixture/model-Q4_K_M.gguf",
        vram_before_mb=4,
        ttl_s=5.0,
    )


def _complete(lease: lease_api.GpuLease) -> None:
    lease.transition("admitted")
    lease.transition("loading")
    lease.transition("resident", vram_mb=1028)
    lease.transition("inferencing")
    lease.transition("unloading")
    lease.transition(
        "validating",
        vram_mb=4,
        exit_code=0,
        unload_observed=True,
    )
    lease.transition("terminal_complete")


def test_req_infra_6633_spec_anchors_every_lease_property() -> None:
    """REQ-INFRA-6633: the OpenSpec contract exists before implementation."""

    text = (REPO / "openspec/capabilities/research-harnesses/spec.md").read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6633") :]
    for anchor in (
        "SCENARIO-INFRA-6633-ATOMIC-RACE",
        "SCENARIO-INFRA-6633-INDEPENDENT-DEVICES",
        "SCENARIO-INFRA-6633-OWNER-AND-PHASES",
        "SCENARIO-INFRA-6633-FAIL-CLOSED",
        "SCENARIO-INFRA-6633-CRASH-RECOVERY",
        "opaque random token",
        "device UUID",
        "PID start time",
        "heartbeat",
        "VRAM before use",
        "atomic replacement",
        "directory sync",
    ):
        assert anchor in section


def test_scenario_infra_6633_owner_heartbeat_phases_and_release(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6633-OWNER-AND-PHASES: one owner completes the journal."""

    lease = _acquire(tmp_path)
    acquired = lease.owner_receipt()
    assert acquired["task_id"] == "exp6633-fixture"
    assert acquired["device_uuid"] == "GPU-test-a"
    assert acquired["token_opaque"] is True
    assert acquired["token_length"] >= 32
    assert lease._token not in json.dumps(lease.document)

    heartbeat = lease.heartbeat()
    assert heartbeat["owner_verified"] is True
    assert heartbeat["expires_monotonic_ns"] > heartbeat["heartbeat_monotonic_ns"]

    _complete(lease)
    release = lease.release()
    written = lease_api.read_journal(lease.journal_path)
    assert release["released"] is True
    assert written["phase"] == "terminal_complete"
    assert written["released"] is True
    assert written["vram_mb"] == {"before": 4, "resident": 1028, "after": 4}
    assert written["exit_evidence"]["exit_code"] == 0
    assert written["unload_evidence"]["observed"] is True
    assert lease_api.validate_journal_document(written, check_freshness=False) == []
    assert len(written["phase_history"]) == len(lease_api.COMPLETE_PHASE_SEQUENCE)


def test_scenario_infra_6633_atomic_race_and_independent_devices(tmp_path: Path) -> None:
    """Atomic same-device exclusion does not block an independent UUID."""

    first = _acquire(tmp_path, device="GPU-same")
    with pytest.raises(lease_api.LeaseBusy):
        _acquire(tmp_path, device="GPU-same")

    second = _acquire(tmp_path, device="GPU-other")
    assert first.lock_path != second.lock_path
    assert first.journal_path != second.journal_path
    first.transition("terminal_blocked")
    second.transition("terminal_blocked")
    assert first.release()["signals_sent"] == []
    assert second.release()["signals_sent"] == []


def test_scenario_infra_6633_rejects_phase_and_owner_attacks(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6633-FAIL-CLOSED rejects skips and non-owner changes."""

    lease = _acquire(tmp_path)
    with pytest.raises(lease_api.TransitionError, match="transition_not_allowed"):
        lease.transition("loading")
    with pytest.raises(lease_api.OwnershipError, match="wrong_token"):
        lease.heartbeat(token="not-the-owner-token")
    with pytest.raises(lease_api.OwnershipError, match="wrong_device"):
        lease.transition("admitted", device_uuid="GPU-wrong")
    with pytest.raises(lease_api.OwnershipError, match="wrong_model"):
        lease.transition("admitted", expected_model="wrong/model.gguf")
    with pytest.raises(lease_api.OwnershipError, match="pid_start_mismatch"):
        lease.transition("admitted", pid_start_ticks=lease.pid_start_ticks + 1)

    _complete(lease)
    with pytest.raises(lease_api.TransitionError, match="terminal_already_set"):
        lease.transition("terminal_blocked")
    with pytest.raises(lease_api.OwnershipError, match="wrong_token"):
        lease.release(token="wrong")
    assert lease.release()["released"] is True


def test_scenario_infra_6633_timeout_missing_unload_and_reversal(tmp_path: Path) -> None:
    """Expiry, missing unload, and phase reversal all fail closed."""

    expired = lease_api.GpuLease.acquire(
        runtime_dir=tmp_path / "expired",
        task_id="expired",
        device_uuid="GPU-expired",
        expected_model="fixture/model.gguf",
        vram_before_mb=4,
        ttl_s=1.0,
    )
    with pytest.raises(lease_api.LeaseExpired):
        expired.heartbeat(now_ns=expired.document["expires_monotonic_ns"] + 1)
    expired.close()

    lease = _acquire(tmp_path / "unload")
    lease.transition("admitted")
    lease.transition("loading")
    lease.transition("resident", vram_mb=900)
    with pytest.raises(lease_api.TransitionError, match="transition_not_allowed"):
        lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=False)
    lease.transition("unloading")
    with pytest.raises(lease_api.TransitionError, match="missing_unload_evidence"):
        lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=False)
    lease.transition("validating", vram_mb=4, exit_code=0, unload_observed=True)
    with pytest.raises(lease_api.TransitionError, match="transition_not_allowed"):
        lease.transition("unloading")
    lease.transition("terminal_blocked")
    lease.release()


def test_scenario_infra_6633_atomic_write_tamper_and_pid_reuse(tmp_path: Path) -> None:
    """A partial replacement and validly rehashed PID reuse both fail closed."""

    target = tmp_path / "atomic.json"
    lease_api.write_json_atomic(target, {"version": 1})
    original = target.read_bytes()
    with pytest.raises(OSError, match="replace failed"):
        lease_api.write_json_atomic(
            target,
            {"version": 2},
            replace=lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
        )
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))

    lease = _acquire(tmp_path / "tamper")
    document = deepcopy(lease.document)
    document["owner"]["pid_start_ticks"] += 1
    document["checksum"] = lease_api.journal_checksum(document)
    errors = lease_api.validate_journal_document(
        document,
        expected_pid=lease.pid,
        expected_pid_start_ticks=lease.pid_start_ticks,
        check_freshness=False,
    )
    assert "pid_start_mismatch" in errors

    tampered = deepcopy(lease.document)
    tampered["expected_model"] = "tampered/model.gguf"
    lease.journal_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(lease_api.JournalError, match="checksum_mismatch"):
        lease_api.read_journal(lease.journal_path)
    lease.close()


def test_scenario_infra_6633_rejects_malformed_journal_variants(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6633-FAIL-CLOSED covers every structural evidence check."""

    lease = _acquire(tmp_path / "base")
    base = deepcopy(lease.document)
    lease.close()

    assert lease_api.validate_journal_document({})[0].startswith("missing_field:")
    assert lease_api._history_errors(None) == ["phase_history_missing"]
    assert "phase_event_invalid" in lease_api._history_errors([None])

    first = lease_api._phase_event(
        phase="terminal_blocked",
        previous_phase="wrong",
        previous_event_checksum="wrong",
        monotonic_ns=1,
        token_digest="sha256:test",
    )
    second = lease_api._phase_event(
        phase="terminal_complete",
        previous_phase="wrong",
        previous_event_checksum=first["event_checksum"],
        monotonic_ns=2,
        token_digest="sha256:test",
    )
    history_errors = lease_api._history_errors([first, second])
    assert {
        "initial_phase_invalid",
        "event_chain_mismatch",
        "phase_history_transition_invalid",
        "second_terminal",
    } <= set(history_errors)
    first["details"]["changed"] = True
    assert "event_checksum_mismatch" in lease_api._history_errors([first])

    malformed = deepcopy(base)
    malformed.update(
        {
            "schema": "wrong",
            "phase": "wrong",
            "owner": {"pid": 0, "pid_start_ticks": -1, "token_digest": "wrong"},
            "heartbeat_monotonic_ns": "wrong",
        }
    )
    malformed["phase_history"][-1]["phase"] = "other"
    malformed["checksum"] = "sha256:wrong"
    errors = lease_api.validate_journal_document(
        malformed,
        expected_pid=os.getpid(),
        expected_pid_start_ticks=base["owner"]["pid_start_ticks"],
        expected_device_uuid="GPU-other",
        expected_model="other/model.gguf",
        check_freshness=False,
    )
    assert {
        "schema_mismatch",
        "checksum_mismatch",
        "phase_invalid",
        "current_phase_history_mismatch",
        "pid_invalid",
        "pid_start_invalid",
        "token_digest_invalid",
        "pid_mismatch",
        "pid_start_mismatch",
        "device_mismatch",
        "model_mismatch",
        "monotonic_time_invalid",
    } <= set(errors)

    bad_time = deepcopy(base)
    bad_time["expires_monotonic_ns"] = bad_time["heartbeat_monotonic_ns"]
    bad_time["checksum"] = lease_api.journal_checksum(bad_time)
    assert "monotonic_time_order_invalid" in lease_api.validate_journal_document(
        bad_time, check_freshness=False
    )
    assert {"lease_expired", "stale_heartbeat"} <= set(
        lease_api.validate_journal_document(base, now_ns=base["expires_monotonic_ns"] + 1)
    )
    assert lease_api.validate_journal_document(base, now_ns=base["heartbeat_monotonic_ns"]) == []

    evidence = _acquire(tmp_path / "evidence")
    _complete(evidence)
    terminal = deepcopy(evidence.document)
    evidence.close()
    terminal["unload_evidence"] = {
        "required": False,
        "observed": False,
        "observed_monotonic_ns": None,
    }
    terminal["exit_evidence"]["exit_code"] = None
    terminal["vram_mb"]["resident"] = None
    terminal["vram_mb"]["after"] = None
    terminal["checksum"] = lease_api.journal_checksum(terminal)
    assert {
        "unload_requirement_missing",
        "missing_unload_evidence",
        "exit_evidence_missing",
        "vram_evidence_missing",
    } <= set(lease_api.validate_journal_document(terminal, check_freshness=False))

    released_early = deepcopy(base)
    released_early["released"] = True
    released_early["checksum"] = lease_api.journal_checksum(released_early)
    assert "nonterminal_release" in lease_api.validate_journal_document(
        released_early, check_freshness=False
    )

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    with pytest.raises(lease_api.JournalError, match="journal_unreadable"):
        lease_api.read_journal(malformed_path)
    malformed_path.write_text("[]", encoding="utf-8")
    with pytest.raises(lease_api.JournalError, match="journal_not_object"):
        lease_api.read_journal(malformed_path)


def test_scenario_infra_6633_identity_recovery_and_operation_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-INFRA-6633-CRASH-RECOVERY proves identity and recovery branches."""

    assert lease_api.proc_start_ticks(999_999_999) is None
    monkeypatch.setattr(lease_api, "proc_start_ticks", lambda _pid: None)
    with pytest.raises(lease_api.JournalError, match="pid_start_unavailable"):
        lease_api.current_process_identity()
    monkeypatch.undo()
    monkeypatch.setattr(lease_api.os, "readlink", lambda _path: (_ for _ in ()).throw(OSError()))
    assert lease_api.current_process_identity()["executable"] == lease_api.sys.executable
    monkeypatch.undo()

    with pytest.raises(ValueError, match="task_device_and_model_required"):
        lease_api.GpuLease.acquire(
            runtime_dir=tmp_path,
            task_id="",
            device_uuid="GPU-x",
            expected_model="m",
            vram_before_mb=0,
        )
    with pytest.raises(ValueError, match="ttl_must_be_positive"):
        lease_api.GpuLease.acquire(
            runtime_dir=tmp_path,
            task_id="t",
            device_uuid="GPU-x",
            expected_model="m",
            vram_before_mb=0,
            ttl_s=0,
        )

    live = _acquire(tmp_path / "recover", device="GPU-recover")
    live.close()
    with pytest.raises(lease_api.RecoveryError, match="recorded_owner_still_live"):
        _acquire(tmp_path / "recover", device="GPU-recover")
    monkeypatch.setattr(lease_api, "process_start_matches", lambda _pid, _start: False)
    recovered = _acquire(tmp_path / "recover", device="GPU-recover")
    assert recovered.document["recovery"]["performed"] is True
    assert recovered.document["lease_generation"] == 2
    monkeypatch.undo()
    recovered.transition("terminal_blocked")
    recovered.release()
    replacement = _acquire(tmp_path / "recover", device="GPU-recover")
    assert replacement.document["recovery"]["performed"] is False
    assert replacement.document["lease_generation"] == 3
    replacement.close()

    changed = _acquire(tmp_path / "changed")
    external = deepcopy(changed.document)
    external["task_id"] = "external-writer"
    external["checksum"] = lease_api.journal_checksum(external)
    lease_api.write_json_atomic(changed.journal_path, external)
    with pytest.raises(lease_api.JournalError, match="journal_changed_by_other_writer"):
        changed.heartbeat()
    changed.close()

    wrong_pid = _acquire(tmp_path / "wrong-pid")
    monkeypatch.setattr(lease_api.os, "getpid", lambda: wrong_pid.pid + 1)
    with pytest.raises(lease_api.OwnershipError, match="wrong_pid"):
        wrong_pid.heartbeat()
    monkeypatch.undo()
    wrong_pid.close()

    terminal = _acquire(tmp_path / "terminal")
    terminal.transition("terminal_blocked")
    with pytest.raises(lease_api.TransitionError, match="terminal_already_set"):
        terminal.heartbeat()
    terminal.release()
    terminal.close()

    resident = _acquire(tmp_path / "resident")
    resident.transition("admitted")
    resident.transition("loading")
    with pytest.raises(lease_api.TransitionError, match="resident_vram_missing"):
        resident.transition("resident")
    resident.close()

    validating = _acquire(tmp_path / "validating")
    validating.transition("admitted")
    validating.transition("loading")
    validating.transition("resident", vram_mb=1)
    validating.transition("unloading")
    with pytest.raises(lease_api.TransitionError, match="validation_exit_or_vram_missing"):
        validating.transition("validating", unload_observed=True)
    validating.close()

    early = _acquire(tmp_path / "early")
    with pytest.raises(lease_api.TransitionError, match="release_requires_terminal_phase"):
        early.release()
    lease_api._complete_fixture(early)
    assert early.release()["released"] is True
