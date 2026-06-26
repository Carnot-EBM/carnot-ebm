"""Tests for Exp 4787 KV260 SSH-only continuity.

Spec refs: REQ-HW-4787, SCENARIO-HW-4787.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4787_kv260_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4787 runner with queued SSH-only board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        timeout_s: float = 60.0,
    ) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class TickClock:
    """Deterministic clock for REQ-HW-4787 duration assertions."""

    def __init__(self) -> None:
        self.value = 40.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.25
        return current


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _listapps_loaded_stdout() -> str:
    return (
        "                   Accelerator  Accel_type                    Base    Pid   "
        "Base_type  #slots(RPU+PL+AIE)    slot->handle\n"
        "               carnot_ising_v4    XRT_FLAT         carnot_ising_v4  id_ok    "
        "XRT_FLAT             (0+0+0)              -1\n"
        "           carnot_ising_v2_n64    XRT_FLAT     carnot_ising_v2_n64  id_ok    "
        "XRT_FLAT             (0+0+0)           0->0,\n"
    )


def _board_state_stdout() -> str:
    return (
        "kv260\n"
        "Linux kv260 6.8.0-1029-xilinx #30-Ubuntu SMP PREEMPT_DYNAMIC aarch64 GNU/Linux\n"
        " 07:33:12 up 18 days, 13:24,  1 user,  load average: 0.13, 0.04, 0.01\n"
        "5\n"
    )


def _reachable_runner(
    *,
    direct_listapps: mod.CommandProbe,
    sudo_listapps: mod.CommandProbe | None = None,
) -> RecordingRunner:
    probes = {
        mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
        mod.KV260_LISTAPPS_COMMAND: [direct_listapps],
        mod.KV260_BOARD_STATE_COMMAND: [
            _probe(mod.KV260_BOARD_STATE_COMMAND, stdout=_board_state_stdout(), duration_s=0.4)
        ],
    }
    if sudo_listapps is not None:
        probes[mod.KV260_LISTAPPS_SUDO_COMMAND] = [sudo_listapps]
    return RecordingRunner(probes)


def _assert_required_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (isinstance(payload[field], dict) and "principle" in payload[field])


def test_req_hw_4787_spec_anchor_declares_ssh_only_required_contract() -> None:
    """REQ-HW-4787: OpenSpec declares fields, commands, and principle text."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4787" in spec
    assert "SCENARIO-HW-4787" in spec
    assert "experiment_4787_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "host SD-card device nodes are permanently retired for KV260" in spec
    assert "random_seed=4787" in spec
    assert mod.REACHABLE_NEXT_FORWARD_STEP in spec
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4787_blocked_ssh_stops_without_board_state_commands() -> None:
    """SCENARIO-HW-4787: unreachable SSH exits before any board-state probe."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(
                    mod.KV260_SSH_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host kria port 22: timed out\n",
                    duration_s=5.0,
                )
            ]
        }
    )

    payload = mod.build_artifact(command_runner=runner, clock=TickClock())

    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == 4787
    assert payload["spec_refs"] == ["REQ-HW-4787", "SCENARIO-HW-4787"]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["kv260_ssh_reachable"] is False
    assert payload["loaded_overlay"] is None
    assert payload["next_forward_step"] == mod.BLOCKED_NEXT_FORWARD_STEP
    assert "operator/audit" in payload["next_forward_step"]
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    assert payload["command_probes"]["kv260_board_state"] is None
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4787_reachable_board_records_direct_overlay_and_state() -> None:
    """SCENARIO-HW-4787: reachable SSH records direct xmutil and board state."""
    runner = _reachable_runner(
        direct_listapps=_probe(
            mod.KV260_LISTAPPS_COMMAND,
            stdout=_listapps_loaded_stdout(),
            duration_s=0.3,
        )
    )

    payload = mod.build_artifact(command_runner=runner, clock=TickClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_BOARD_STATE_COMMAND,
    ]
    assert payload["honest_verdict"] == "success: kv260_continuity_recorded"
    assert payload["kv260_ssh_reachable"] is True
    assert payload["xmutil_requires_sudo"] is False
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["board_state"]["hostname"] == "kv260"
    assert payload["board_state"]["uio_device_count"] == 5
    assert payload["next_forward_step"] == mod.REACHABLE_NEXT_FORWARD_STEP
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 4787
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4787_sudo_xmutil_fallback_preserves_direct_failure() -> None:
    """SCENARIO-HW-4787: root-required xmutil uses read-only sudo fallback."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    runner = _reachable_runner(
        direct_listapps=_probe(
            mod.KV260_LISTAPPS_COMMAND,
            exit_code=1,
            stderr=root_error,
            duration_s=0.3,
        ),
        sudo_listapps=_probe(
            mod.KV260_LISTAPPS_SUDO_COMMAND,
            stdout=_listapps_loaded_stdout(),
            duration_s=0.35,
        ),
    )

    payload = mod.build_artifact(command_runner=runner, clock=TickClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        mod.KV260_BOARD_STATE_COMMAND,
    ]
    assert payload["xmutil_requires_sudo"] is True
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert "root privileges" in payload["command_probes"]["kv260_xmutil_listapps"]["stderr"]
    assert payload["command_probes"]["kv260_xmutil_listapps_sudo"]["exit_code"] == 0
    mod.validate_artifact(payload)


def test_req_hw_4787_run_experiment_writes_requested_artifact(tmp_path: Path) -> None:
    """REQ-HW-4787: run_experiment writes the requested results JSON."""
    runner = _reachable_runner(
        direct_listapps=_probe(mod.KV260_LISTAPPS_COMMAND, stdout=_listapps_loaded_stdout())
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, clock=TickClock())
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == 4787
    assert payload["spec_refs"] == ["REQ-HW-4787", "SCENARIO-HW-4787"]
    assert payload["duration_s"] == 0.5
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_req_hw_4787_validation_rejects_schema_drift_and_wrong_precondition() -> None:
    """REQ-HW-4787: validation rejects schema drift and wrong-mechanism markers."""
    payload = mod.build_artifact(
        command_runner=_reachable_runner(
            direct_listapps=_probe(mod.KV260_LISTAPPS_COMMAND, stdout=_listapps_loaded_stdout())
        ),
        clock=TickClock(),
    )

    bad_schema = dict(payload, schema="stale")
    bad_schema["reproducibility_checksum"] = mod.payload_checksum(bad_schema)
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(bad_schema)

    bad_principles = dict(payload, field_principles={})
    bad_principles["reproducibility_checksum"] = mod.payload_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    wrapped = dict(payload)
    wrapped["honest_verdict"] = {"value": "wrapped", "principle": "forbidden"}
    wrapped["reproducibility_checksum"] = mod.payload_checksum(wrapped)
    with pytest.raises(ValueError, match="bare value"):
        mod.validate_artifact(wrapped)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd)

    stale_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(stale_checksum)
