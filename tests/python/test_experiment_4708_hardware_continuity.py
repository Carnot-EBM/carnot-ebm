"""Tests for Exp 4708 hardware continuity.

Spec refs: REQ-HW-4708, SCENARIO-HW-4708.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4708_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4708 runner with queued reachability and state transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _all_reachable_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_PRECONDITION_COMMAND: [
                _probe(mod.KV260_PRECONDITION_COMMAND, duration_s=0.11),
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(mod.POLARFIRE_PRECONDITION_COMMAND, duration_s=0.22),
            ],
            mod.GATEMATE_PRECONDITION_COMMAND: [
                _probe(
                    mod.GATEMATE_PRECONDITION_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.33,
                ),
            ],
            mod.KV260_STATE_COMMAND: [
                _probe(
                    mod.KV260_STATE_COMMAND,
                    stdout="app: carnot_ising_v4\napp: benchmark_shell\n",
                    duration_s=0.44,
                ),
            ],
            mod.KV260_UIO_SMOKE_COMMAND: [
                _probe(
                    mod.KV260_UIO_SMOKE_COMMAND,
                    stdout="/dev/uio0\n/dev/uio1\n",
                    duration_s=0.05,
                ),
            ],
            mod.POLARFIRE_STATE_COMMAND: [
                _probe(
                    mod.POLARFIRE_STATE_COMMAND,
                    stdout=" 12:01:02 up 12 days,  4:03,  1 user,  load average: 0.00\n",
                    duration_s=0.55,
                ),
            ],
        }
    )


def test_req_hw_4708_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4708: OpenSpec anchors fields, commands, and principle text."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4708" in spec
    assert "SCENARIO-HW-4708" in spec
    assert "experiment_4708_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "GateMate GM1Ax IDCODE" in spec
    assert "random_seed=4708" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert "blocked_polarfire_ssh_timeout" in spec
    assert "blocked_gatemate_usb_undetected" in spec
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4708_reachable_boards_record_required_state(tmp_path: Path) -> None:
    """SCENARIO-HW-4708: reachable boards capture overlay, uptime, IDCODE, and UIO."""
    runner = _all_reachable_runner()

    artifact = mod.build_artifact(command_runner=runner)
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.GATEMATE_PRECONDITION_COMMAND,
        mod.KV260_STATE_COMMAND,
        mod.KV260_UIO_SMOKE_COMMAND,
        mod.POLARFIRE_STATE_COMMAND,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == 4708
    assert saved["spec_refs"] == ["REQ-HW-4708", "SCENARIO-HW-4708"]
    assert saved["random_seed"] == 4708
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["honest_verdict"] == "success: hardware_continuity_3_of_3_boards_reachable"
    assert saved["reachable_board_count"] == 3
    assert saved["boards_reachable"] == {
        "kv260": True,
        "polarfire": True,
        "gatemate": True,
    }
    assert saved["per_board_reachability"] == saved["boards_reachable"]
    assert saved["kv260_precondition"]["command"] == (
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true"
    )
    assert saved["kv260_precondition"]["discipline"] == "ssh_only_no_host_sd_card"
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "exit_code": 0,
            "duration_s": 0.11,
            "observed": "returncode=0",
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
            "exit_code": 0,
            "duration_s": 0.22,
            "observed": "returncode=0",
        },
        {
            "resource": "gatemate_usb_detect",
            "available": True,
            "command": "openFPGALoader -c dirtyJtag --detect",
            "exit_code": 0,
            "duration_s": 0.33,
            "observed": "IDCode : 0x20000001 colognechip GateMate GM1Ax",
        },
    ]
    assert saved["per_board_state"]["kv260"]["status"] == "kv260_reachable_loaded_overlay_recorded"
    assert saved["per_board_state"]["kv260"]["state"]["state_type"] == "loaded_overlay"
    assert saved["per_board_state"]["kv260"]["energy_eval_smoke"]["state_type"] == "uio_devices"
    assert saved["per_board_state"]["polarfire"]["status"] == "polarfire_reachable_uptime_recorded"
    assert saved["per_board_state"]["gatemate"]["state"]["idcode"] == "0x20000001"
    assert saved["bitstream_build_attempted"] is False
    assert saved["fabric_acceleration_claimed"] is False
    assert saved["speedup_claim_made"] is False
    assert "mmcblk" not in json.dumps(saved).lower()
    assert saved["field_principles"] == mod.FIELD_PRINCIPLES
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_4708_blocked_boards_remain_visible_without_state_commands() -> None:
    """REQ-HW-4708: blocked board rows remain visible and do not stop the audit."""
    runner = RecordingRunner(
        {
            mod.KV260_PRECONDITION_COMMAND: [
                _probe(mod.KV260_PRECONDITION_COMMAND, 255, stderr="timeout", duration_s=0.2),
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(mod.POLARFIRE_PRECONDITION_COMMAND, 255, stderr="timeout", duration_s=0.3),
            ],
            mod.GATEMATE_PRECONDITION_COMMAND: [
                _probe(
                    mod.GATEMATE_PRECONDITION_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.4,
                ),
            ],
        }
    )

    artifact = mod.build_artifact(command_runner=runner)

    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.GATEMATE_PRECONDITION_COMMAND,
    ]
    assert artifact["honest_verdict"] == "success: hardware_continuity_0_of_3_boards_reachable"
    assert artifact["reachable_board_count"] == 0
    assert artifact["boards_reachable"] == {
        "kv260": False,
        "polarfire": False,
        "gatemate": False,
    }
    assert artifact["per_board_state"]["kv260"]["status"] == "blocked_kv260_ssh_unreachable"
    assert artifact["per_board_state"]["polarfire"]["status"] == "blocked_polarfire_ssh_timeout"
    assert artifact["per_board_state"]["gatemate"]["status"] == "blocked_gatemate_usb_undetected"
    for board in mod.BOARD_NAMES:
        assert artifact["per_board_state"][board]["state"] == {
            "captured": False,
            "reason": artifact["per_board_state"][board]["status"],
        }
    mod.validate_artifact(artifact)


def test_scenario_hw_4708_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-4708: run_experiment writes the requested results artifact."""
    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=_all_reachable_runner())
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment"] == 4708
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_4708_validation_rejects_bad_metadata_principles_and_checksum() -> None:
    """REQ-HW-4708: validation rejects wrong metadata, principles, and checksums."""
    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["experiment"] = 4696
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="4708"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["field_principles"]["speedup_claim_made"] = "not the required principle"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["honest_verdict"] = {"value": "wrapped", "principle": "forbidden"}
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="honest_verdict must remain a bare value"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(artifact)
