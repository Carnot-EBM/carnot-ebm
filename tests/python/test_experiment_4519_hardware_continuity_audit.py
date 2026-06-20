"""Tests for Exp 4519 hardware-task continuity audit.

Spec refs: REQ-HW-4519, SCENARIO-HW-4519.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4519_hardware_continuity_audit as mod


class RecordingRunner:
    """SCENARIO-HW-4519 runner with queued reachability transcripts."""

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


def test_req_hw_4519_spec_entry_declares_required_audit_fields() -> None:
    """REQ-HW-4519: OpenSpec anchors the command and field contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4519" in spec
    assert "SCENARIO-HW-4519" in spec
    assert "experiment_4519_hardware_continuity_audit.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "blocked_<board>_unreachable" in spec
    assert "SD-card mechanism is retired" in spec
    assert mod.REQUIRED_OPERATOR_FIELDS == (
        "honest_verdict",
        "inference_substrate",
        "kv260_reachable",
        "gatemate_detected",
        "polarfire_reachable",
        "preconditions_checked",
    )
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4519_reachable_boards_record_next_forward_steps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4519: reachable boards get top-level booleans and next steps."""
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
    assert saved["field_principles"] == mod.FIELD_PRINCIPLES
    assert saved["kv260_reachable"] is True
    assert saved["gatemate_detected"] is True
    assert saved["polarfire_reachable"] is True
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
    assert "latency transcript" in saved["per_board_status"]["kv260"]["next_forward_step"]
    assert "bitstream" in saved["per_board_status"]["kv260"]["next_forward_step"]
    assert saved["per_board_status"]["gatemate"]["status"] == (
        "gatemate_reachable_dirtyjtag_detect"
    )
    assert "n=16 Ising tile" in saved["per_board_status"]["gatemate"]["next_forward_step"]
    assert saved["per_board_status"]["polarfire"]["status"] == "polarfire_reachable_ssh"
    assert "sampler smoke" in saved["per_board_status"]["polarfire"]["next_forward_step"]
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_audit_4519_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4519_unreachable_boards_are_blocked_without_stopping_others() -> None:
    """REQ-HW-4519: blocked board rows remain honest per-board states."""
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
    assert artifact["kv260_reachable"] is False
    assert artifact["gatemate_detected"] is False
    assert artifact["polarfire_reachable"] is True
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


def test_scenario_hw_4519_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-4519: run_experiment writes the requested results artifact."""
    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=_all_reachable_runner())
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment"] == 4519
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_4519_validation_rejects_fabrication_and_schema_drift() -> None:
    """REQ-HW-4519: validation rejects wrappers, bad reachability, and stale checksums."""
    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["honest_verdict"] = "blocked_all_unreachable"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["kv260_reachable"] = {
        "value": artifact["kv260_reachable"],
        "principle": mod.FIELD_PRINCIPLES["kv260_reachable"],
    }
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="bare value"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["kv260_reachable"] = False
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="kv260_reachable mismatch"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["per_board_status"]["kv260"]["precondition_command"] = "checked mmcblk"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="forbidden"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="hardware_smoke"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["speedup_claim_made"] = True
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    with pytest.raises(ValueError, match="speedup claim"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(command_runner=_all_reachable_runner())
    artifact["reproducibility_checksum"] = "stale"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(artifact)
