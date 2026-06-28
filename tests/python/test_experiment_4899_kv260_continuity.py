"""Tests for Exp 4899 KV260 SSH-only continuity.

Spec refs: REQ-HW-4899, SCENARIO-HW-4899.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4899_kv260_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4899 runner with queued SSH-only board transcripts."""

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


class FlatClock:
    """Deterministic clock for REQ-HW-4899 duration floor assertions."""

    def __call__(self) -> float:
        return 4899.0


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
    direct_listapps: mod.CommandProbe | None = None,
    sudo_listapps: mod.CommandProbe | None = None,
) -> RecordingRunner:
    probes = {
        mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
        mod.KV260_LISTAPPS_COMMAND: [
            direct_listapps
            or _probe(
                mod.KV260_LISTAPPS_COMMAND,
                stdout=_listapps_loaded_stdout(),
                duration_s=0.3,
            )
        ],
        mod.KV260_BOARD_STATE_COMMAND: [
            _probe(
                mod.KV260_BOARD_STATE_COMMAND,
                stdout=_board_state_stdout(),
                duration_s=0.4,
            )
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


def test_req_hw_4899_spec_anchor_declares_terminal_continuity_contract() -> None:
    """REQ-HW-4899: OpenSpec declares the SSH-only terminal continuity contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4899" in spec
    assert "SCENARIO-HW-4899" in spec
    assert "experiment_4899_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "Host SD-card device nodes are permanently retired for KV260" in spec
    assert "No file changes" in spec
    assert "3-fail-skip" in spec
    assert "random_seed=4899" in spec
    assert "duration_s >= 0.0001" in spec
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4899_blocked_ssh_run_experiment_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4899: unreachable SSH still writes the blocked deliverable."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(
                    mod.KV260_SSH_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host kria port 22: No route to host\n",
                    duration_s=5.0,
                )
            ]
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, clock=FlatClock())
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == 4899
    assert payload["spec_refs"] == ["REQ-HW-4899", "SCENARIO-HW-4899"]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert payload["duration_s"] == 0.0001
    assert payload["kv260_ssh_reachable"] is False
    assert payload["preconditions_checked"][0]["available"] is False
    assert payload["preconditions_checked"][0]["discipline"] == "ssh_only_no_host_sd_card"
    assert "No route to host" in payload["preconditions_checked"][0]["observed"]
    assert payload["loaded_overlay"] is None
    assert payload["board_state"] == {"captured": False, "reason": "kv260_ssh_unreachable"}
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    assert payload["command_probes"]["kv260_board_state"] is None
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4899_reachable_board_records_terminal_state() -> None:
    """SCENARIO-HW-4899: reachable SSH records board state and terminal next step."""
    runner = _reachable_runner()

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_BOARD_STATE_COMMAND,
    ]
    assert payload["honest_verdict"] == "success_kv260_continuity_ok"
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert payload["duration_s"] == 0.0001
    assert payload["kv260_ssh_reachable"] is True
    assert payload["xmutil_requires_sudo"] is False
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["board_state"]["hostname"] == "kv260"
    assert payload["board_state"]["kernel"].startswith("Linux kv260")
    assert payload["board_state"]["uptime"].startswith("07:33:12 up")
    assert payload["board_state"]["uio_device_count"] == 5
    assert payload["next_forward_step"] == mod.REACHABLE_NEXT_FORWARD_STEP
    assert "GRADUATED: KV260 terminal criteria met" in payload["next_forward_step"]
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 4899
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4899_sudo_xmutil_fallback_preserves_transcripts() -> None:
    """SCENARIO-HW-4899: root-required xmutil uses the SSH-only sudo fallback."""
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

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

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


def test_req_hw_4899_validation_rejects_schema_drift_and_retired_host_storage() -> None:
    """REQ-HW-4899: validation rejects schema drift and retired host-storage markers."""
    payload = mod.build_artifact(command_runner=_reachable_runner(), clock=FlatClock())

    bad_schema = dict(payload, schema="stale")
    bad_schema["reproducibility_checksum"] = mod.payload_checksum(bad_schema)
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(bad_schema)

    bad_substrate = dict(payload, inference_substrate="hardware_smoke")
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="substrate"):
        mod.validate_artifact(bad_substrate)

    bad_principles = dict(payload, field_principles={})
    bad_principles["reproducibility_checksum"] = mod.payload_checksum(bad_principles)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd)

    stale_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(stale_checksum)
