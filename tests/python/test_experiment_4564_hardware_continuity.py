"""Tests for Exp 4564 hardware continuity.

Spec refs: REQ-HW-4564, SCENARIO-HW-4564.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4564_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4564 runner with queued precondition and state transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], mod.CommandProbe]) -> None:
        self.probes = dict(probes)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        self.commands.append(command)
        if command not in self.probes:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command]


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _reachable_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_REACHABILITY_COMMAND: _probe(
                mod.KV260_REACHABILITY_COMMAND,
                duration_s=0.11,
            ),
            mod.GATEMATE_REACHABILITY_COMMAND: _probe(
                mod.GATEMATE_REACHABILITY_COMMAND,
                stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                duration_s=0.22,
            ),
            mod.POLARFIRE_REACHABILITY_COMMAND: _probe(
                mod.POLARFIRE_REACHABILITY_COMMAND,
                duration_s=0.33,
            ),
            mod.KV260_STATE_COMMAND: _probe(
                mod.KV260_STATE_COMMAND,
                stdout="app: carnot_ising_v4\napp: benchmark_shell\n",
                duration_s=0.44,
            ),
            mod.POLARFIRE_STATE_COMMAND: _probe(
                mod.POLARFIRE_STATE_COMMAND,
                stdout=" 12:01:02 up 9 days,  4:03,  1 user,  load average: 0.00\n",
                duration_s=0.55,
            ),
        }
    )


def test_req_hw_4564_spec_entry_declares_stateful_audit_contract() -> None:
    """REQ-HW-4564: OpenSpec anchors the preconditions, states, and principles."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4564" in spec
    assert "SCENARIO-HW-4564" in spec
    assert "experiment_4564_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'" in spec
    assert "GM1Ax IDCODE" in spec
    assert "polarfire uptime" in spec.lower()
    assert "random_seed=4564" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert "blocked_gatemate_usb_undetected" in spec
    assert "blocked_polarfire_ssh_timeout" in spec
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4564_reachable_boards_record_state_and_board_count(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4564: reachable boards capture xmutil, IDCODE, and uptime state."""
    runner = _reachable_runner()

    artifact = mod.build_artifact(command_runner=runner)
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_REACHABILITY_COMMAND,
        mod.GATEMATE_REACHABILITY_COMMAND,
        mod.POLARFIRE_REACHABILITY_COMMAND,
        mod.KV260_STATE_COMMAND,
        mod.POLARFIRE_STATE_COMMAND,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["field_principles"] == mod.FIELD_PRINCIPLES
    assert saved["honest_verdict"] == "complete: hardware_continuity_audit_3_boards_reachable"
    assert saved["reachable_board_count"] == 3
    assert saved["bitstream_build_attempted"] is False
    assert saved["fabric_acceleration_claimed"] is False
    assert saved["speedup_claim_made"] is False
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": mod.command_to_string(mod.KV260_REACHABILITY_COMMAND),
            "exit_code": 0,
            "duration_s": 0.11,
            "observed": "returncode=0",
        },
        {
            "resource": "gatemate_usb_detect",
            "available": True,
            "command": mod.command_to_string(mod.GATEMATE_REACHABILITY_COMMAND),
            "exit_code": 0,
            "duration_s": 0.22,
            "observed": "IDCode : 0x20000001 colognechip GateMate GM1Ax",
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": mod.command_to_string(mod.POLARFIRE_REACHABILITY_COMMAND),
            "exit_code": 0,
            "duration_s": 0.33,
            "observed": "returncode=0",
        },
    ]
    assert saved["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_reachable_state_recorded"
    assert saved["per_board_status"]["kv260"]["state"]["state_type"] == "xmutil_listapps"
    assert saved["per_board_status"]["kv260"]["state"]["command"] == (
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'xmutil listapps'"
    )
    assert "carnot_ising_v4" in saved["per_board_status"]["kv260"]["state"]["observed"]
    assert saved["per_board_status"]["gatemate"]["status"] == "gatemate_reachable_idcode_recorded"
    assert saved["per_board_status"]["gatemate"]["state"]["idcode"] == "0x20000001"
    assert saved["per_board_status"]["polarfire"]["status"] == "polarfire_reachable_state_recorded"
    assert saved["per_board_status"]["polarfire"]["state"]["state_type"] == "uptime"
    assert "up 9 days" in saved["per_board_status"]["polarfire"]["state"]["observed"]
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    encoded = json.dumps(saved).lower()
    assert "mmcblk" not in encoded
    assert "sudo -n xmutil" not in encoded
    mod.validate_artifact(saved)


def test_req_hw_4564_blocked_boards_do_not_run_state_commands() -> None:
    """REQ-HW-4564: blocked board rows remain honest without stopping the audit."""
    runner = RecordingRunner(
        {
            mod.KV260_REACHABILITY_COMMAND: _probe(
                mod.KV260_REACHABILITY_COMMAND,
                255,
                stderr="timeout",
                duration_s=0.2,
            ),
            mod.GATEMATE_REACHABILITY_COMMAND: _probe(
                mod.GATEMATE_REACHABILITY_COMMAND,
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                duration_s=0.3,
            ),
            mod.POLARFIRE_REACHABILITY_COMMAND: _probe(
                mod.POLARFIRE_REACHABILITY_COMMAND,
                255,
                stderr="timeout",
                duration_s=0.4,
            ),
        }
    )

    artifact = mod.build_artifact(command_runner=runner)

    assert runner.commands == [
        mod.KV260_REACHABILITY_COMMAND,
        mod.GATEMATE_REACHABILITY_COMMAND,
        mod.POLARFIRE_REACHABILITY_COMMAND,
    ]
    assert artifact["honest_verdict"] == "complete: hardware_continuity_audit_0_boards_reachable"
    assert artifact["reachable_board_count"] == 0
    assert artifact["per_board_status"]["kv260"]["status"] == "blocked_kv260_ssh_unreachable"
    assert artifact["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_usb_undetected"
    assert artifact["per_board_status"]["polarfire"]["status"] == "blocked_polarfire_ssh_timeout"
    for board in mod.BOARD_NAMES:
        assert artifact["per_board_status"][board]["state"] == {
            "captured": False,
            "reason": artifact["per_board_status"][board]["status"],
        }
    mod.validate_artifact(artifact)


def test_scenario_hw_4564_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-4564: run_experiment writes the requested results artifact."""
    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=_reachable_runner())
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment"] == 4564
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_4564_validation_rejects_metadata_state_and_checksum_drift() -> None:
    """REQ-HW-4564: validation rejects wrong metadata, stale state, and checksum drift."""
    artifact = mod.build_artifact(command_runner=_reachable_runner())
    artifact["experiment"] = 4552
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="4564"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_reachable_runner())
    artifact["per_board_status"]["kv260"]["state"]["command"] = (
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'sudo -n xmutil listapps'"
    )
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="kv260 state command"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_reachable_runner())
    artifact["per_board_status"] = {
        "value": artifact["per_board_status"],
        "principle": mod.FIELD_PRINCIPLES["per_board_status"],
    }
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="bare value"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_reachable_runner())
    artifact["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(artifact)
