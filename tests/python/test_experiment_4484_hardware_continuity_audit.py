"""Tests for Exp 4484 hardware continuity audit.

Spec refs: REQ-HW-4484, SCENARIO-HW-4484.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4484_hardware_continuity_audit as mod


class RecordingRunner:
    """SCENARIO-HW-4484 command runner with queued reachability transcripts."""

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


def _all_reachable_runner() -> RecordingRunner:
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
        }
    )


def test_req_hw_4484_spec_entry_declares_audit_contract() -> None:
    """REQ-HW-4484: OpenSpec anchors the three-command audit contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4484" in spec
    assert "SCENARIO-HW-4484" in spec
    assert "experiment_4484_hardware_continuity_audit.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "blocked_<board>_unreachable" in spec
    assert "host storage probing is out of scope" in spec
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4484_reachable_boards_record_next_forward_steps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4484: reachable boards get reachability plus next-step rows."""
    runner = _all_reachable_runner()

    artifact = mod.build_artifact(command_runner=runner)
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_REACHABILITY_COMMAND,
        mod.GATEMATE_REACHABILITY_COMMAND,
        mod.POLARFIRE_REACHABILITY_COMMAND,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["offline_reproduced"] is False
    assert saved["reproduced_levels"] == 0
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
            "resource": "gatemate_dirtyjtag_detect",
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
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_reachable_ssh"
    assert "SSH-only" in saved["per_board_status"]["kv260"]["next_forward_step"]
    assert saved["per_board_status"]["gatemate"]["status"] == (
        "gatemate_reachable_dirtyjtag_detect"
    )
    assert "n=16 Ising tile" in saved["per_board_status"]["gatemate"]["next_forward_step"]
    assert saved["per_board_status"]["polarfire"]["status"] == "polarfire_reachable_ssh"
    assert "hash-match" in saved["per_board_status"]["polarfire"]["next_forward_step"]
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_audit_4484_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4484_unreachable_boards_are_blocked_without_stopping_others() -> None:
    """REQ-HW-4484: blocked board rows do not suppress the other board audits."""
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
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["per_board_status"]["kv260"]["status"] == "blocked_kv260_unreachable"
    assert artifact["per_board_status"]["kv260"]["next_forward_step"] == (
        "blocked_kv260_unreachable"
    )
    assert artifact["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_status"]["polarfire"]["status"] == "polarfire_reachable_ssh"
    assert artifact["per_board_reachability"] == {
        "kv260": False,
        "gatemate": False,
        "polarfire": True,
    }
    mod.validate_artifact(artifact)


def test_scenario_hw_4484_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-4484: run_experiment writes the requested results artifact."""
    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=_all_reachable_runner())
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment"] == 4484
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_4484_validation_rejects_fabrication_and_schema_drift() -> None:
    """REQ-HW-4484: validation rejects wrappers, bad prefixes, and stale checksums."""
    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["honest_verdict"] = "blocked_all_unreachable"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["preconditions_checked"] = {
        "value": artifact["preconditions_checked"],
        "principle": mod.FIELD_PRINCIPLES["preconditions_checked"],
    }
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="bare value"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["per_board_status"]["kv260"]["precondition_command"] = "checked mmcblk"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(artifact)
