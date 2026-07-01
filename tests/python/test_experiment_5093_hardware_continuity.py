"""Tests for Exp 5093 hardware continuity v2.

Spec refs: REQ-HW-5093, SCENARIO-HW-5093.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5093_hardware_continuity as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5093 runner with queued non-destructive probe transcripts."""

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


class FlatClock:
    """Deterministic clock for REQ-HW-5093 duration-floor assertions."""

    def __call__(self) -> float:
        return 5093.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _safe_uio_transcript(root: Path) -> Path:
    path = root / mod.SAFE_KV260_UIO_TRANSCRIPT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "carnot.kv260.safe_uio_register_transcript.v1",
                "operation": "uio_register_read",
                "mode": "read_only",
                "safe_for_continuity_audit": True,
                "device": "/dev/uio0",
                "offset": "0x0000",
                "value": "0x00000020",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _all_ready_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
            mod.POLARFIRE_ARCH_COMMAND: [
                _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="riscv64\n", duration_s=0.2)
            ],
            mod.POLARFIRE_PYTHON_COMMAND: [
                _probe(
                    mod.POLARFIRE_PYTHON_COMMAND,
                    stdout="Python 3.12.12\n",
                    duration_s=0.2,
                )
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(
                    mod.POLARFIRE_UPTIME_COMMAND,
                    stdout=" 01:24:00 up 8 days,  7:00,  0 user,  load average: 0.00\n",
                    duration_s=0.2,
                )
            ],
            mod.POLARFIRE_KERNEL_COMMAND: [
                _probe(
                    mod.POLARFIRE_KERNEL_COMMAND,
                    stdout="6.18.17-linux4microchip-2026.04.1\n",
                    duration_s=0.2,
                )
            ],
        }
    )


def test_req_hw_5093_spec_declares_v467_contract() -> None:
    """REQ-HW-5093: OpenSpec anchors v467 fields, commands, and boundaries."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5093",
        "SCENARIO-HW-5093",
        "experiment_5093_hardware_continuity.py",
        "results/experiment_5093_hardware_continuity_v467.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
        "openFPGALoader -c dirtyJtag --detect",
        "hardware_precheck_and_transcript_audit",
        "polarfire_dispatch_precheck_ready",
        "success_hardware_continuity_v467_no_speedup_claim",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5093_all_ready_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5093: ready boards stay non-destructive and no-speedup."""

    _safe_uio_transcript(tmp_path)
    runner = _all_ready_runner()

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_ARCH_COMMAND,
        mod.POLARFIRE_PYTHON_COMMAND,
        mod.POLARFIRE_UPTIME_COMMAND,
        mod.POLARFIRE_KERNEL_COMMAND,
    ]
    assert saved["honest_verdict"] == "success_hardware_continuity_v467_no_speedup_claim"
    assert saved["duration_s"] == 0.0001
    assert saved["inference_substrate"] == "hardware_precheck_and_transcript_audit"
    assert saved["kv260_ssh_ready"] is True
    assert saved["kv260_uio_transcript_path"] == str(mod.SAFE_KV260_UIO_TRANSCRIPT_REL_PATH)
    assert saved["kv260_speedup_claim_allowed"] is False
    assert saved["gatemate_detected"] is True
    assert saved["gatemate_terminal_state"] == "gatemate_detected_idcode_no_flash_terminal"
    assert saved["polarfire_detected"] is True
    assert saved["polarfire_dispatch_precheck_ready"] is True
    assert saved["destructive_actions_allowed"] is False
    assert saved["destructive_actions_taken"] == []
    assert saved["flagged_adversarial"] is False
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "exit_code": 0,
            "duration_s": 0.2,
            "observed": "returncode=0",
            "discipline": "ssh_only_no_host_sd_card",
        },
        {
            "resource": "gatemate_detect_command",
            "available": True,
            "command": "sh -lc 'command -v openFPGALoader'",
            "exit_code": 0,
            "duration_s": 0.1,
            "observed": "/opt/oss-cad-suite/bin/openFPGALoader",
            "discipline": "command_availability_only_no_detect_side_effect",
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
            "exit_code": 0,
            "duration_s": 0.4,
            "observed": "returncode=0",
            "discipline": "ssh_reachability_before_hash_dispatch_precheck",
        },
        {
            "resource": "destructive_actions_allowed",
            "available": False,
            "command": "policy",
            "exit_code": 0,
            "duration_s": 0.0001,
            "observed": "false",
            "discipline": "explicit_no_destructive_actions",
        },
    ]
    assert saved["board_matrix"]["kv260"]["uio_transcript_status"]["verified"] is True
    assert saved["board_matrix"]["gatemate"]["evidence"]["detected_idcode"] == "0x20000001"
    assert saved["board_matrix"]["gatemate"]["cable_state_inference"] == (
        "dirtyjtag_cable_and_gatemate_idcode_detected"
    )
    assert saved["board_matrix"]["polarfire"]["dispatch_executed"] is False
    assert saved["board_matrix"]["polarfire"]["dispatch_precheck"]["arch"] == "riscv64"
    assert "mmcblk" not in json.dumps(saved).lower()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5093_blocked_preconditions_do_not_run_state_commands(tmp_path: Path) -> None:
    """REQ-HW-5093: unavailable boards remain visible without follow-on commands."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(
                    mod.KV260_SSH_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host kria port 22: timeout\n",
                    duration_s=5.0,
                )
            ],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    exit_code=127,
                    stderr="openFPGALoader not found\n",
                    duration_s=0.1,
                )
            ],
            mod.POLARFIRE_SSH_COMMAND: [
                _probe(
                    mod.POLARFIRE_SSH_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host polarfire port 22: timeout\n",
                    duration_s=5.0,
                )
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert artifact["honest_verdict"] == "complete_hardware_continuity_v467_partial_board_blockers"
    assert artifact["kv260_ssh_ready"] is False
    assert artifact["kv260_uio_transcript_path"] is None
    assert artifact["gatemate_detected"] is False
    assert artifact["gatemate_terminal_state"] == "blocked_gatemate_detect_command_unavailable"
    assert artifact["polarfire_detected"] is False
    assert artifact["polarfire_dispatch_precheck_ready"] is False
    assert artifact["command_probes"]["gatemate_dirtyjtag_detect"] is None
    assert artifact["command_probes"]["polarfire_arch"] is None
    assert artifact["board_matrix"]["kv260"]["terminal_state"] == (
        "blocked_kv260_ssh_unreachable_no_uio_register_transcript"
    )
    assert artifact["board_matrix"]["kv260"]["uio_transcript_status"]["blocker"] == (
        "no_existing_safe_kv260_uio_register_transcript_verified"
    )
    assert artifact["destructive_actions_taken"] == []
    mod.validate_artifact(artifact)


def test_scenario_hw_5093_gatemate_and_polarfire_partial_blockers(tmp_path: Path) -> None:
    """SCENARIO-HW-5093: detect logs and dispatch-precheck blockers are preserved."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/usr/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
            mod.POLARFIRE_ARCH_COMMAND: [
                _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="x86_64\n", duration_s=0.2)
            ],
            mod.POLARFIRE_PYTHON_COMMAND: [
                _probe(
                    mod.POLARFIRE_PYTHON_COMMAND,
                    stdout="Python 3.12.12\n",
                    duration_s=0.2,
                )
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(mod.POLARFIRE_UPTIME_COMMAND, stdout=" up 1 day\n", duration_s=0.2)
            ],
            mod.POLARFIRE_KERNEL_COMMAND: [
                _probe(mod.POLARFIRE_KERNEL_COMMAND, stdout="kernel\n", duration_s=0.2)
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())

    assert artifact["honest_verdict"] == "complete_hardware_continuity_v467_partial_board_blockers"
    assert artifact["kv260_ssh_ready"] is True
    assert artifact["gatemate_detected"] is False
    assert artifact["gatemate_terminal_state"] == (
        "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal"
    )
    assert artifact["board_matrix"]["gatemate"]["cable_state_inference"] == (
        "dirtyjtag_cable_seen_no_gatemate_idcode"
    )
    assert "Jtag frequency" in artifact["board_matrix"]["gatemate"]["evidence"]["detect_log"]
    assert artifact["polarfire_detected"] is True
    assert artifact["polarfire_dispatch_precheck_ready"] is False
    assert artifact["board_matrix"]["polarfire"]["terminal_state"] == (
        "blocked_polarfire_hash_dispatch_precheck_not_ready"
    )
    assert artifact["board_matrix"]["polarfire"]["dispatch_precheck"]["blockers"] == [
        "polarfire_arch_not_riscv64"
    ]
    mod.validate_artifact(artifact)


def test_scenario_hw_5093_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5093: run_experiment writes the requested v467 artifact."""

    _safe_uio_transcript(tmp_path)
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment_id"] == 5093
    assert artifact["spec_refs"] == ["REQ-HW-5093", "SCENARIO-HW-5093"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5093_validation_rejects_overclaim_and_drift(tmp_path: Path) -> None:
    """REQ-HW-5093: validation rejects speedup, destructive, storage, and hash drift."""

    _safe_uio_transcript(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
    )

    bad_speedup = dict(artifact, kv260_speedup_claim_allowed=True)
    bad_speedup["reproducibility_checksum"] = mod.payload_checksum(bad_speedup)
    with pytest.raises(ValueError, match="speedup"):
        mod.validate_artifact(bad_speedup)

    bad_allowed = dict(artifact, destructive_actions_allowed=True)
    bad_allowed["reproducibility_checksum"] = mod.payload_checksum(bad_allowed)
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(bad_allowed)

    bad_action = dict(artifact, destructive_actions_taken=["flash_gatemate"])
    bad_action["reproducibility_checksum"] = mod.payload_checksum(bad_action)
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(bad_action)

    bad_storage = dict(artifact)
    bad_storage["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_hw_5093_edge_helpers_and_schema_error_paths(tmp_path: Path) -> None:
    """REQ-HW-5093: defensive helpers report malformed probes and schema drift."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/usr/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="no target response\n")
            ],
            mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
            mod.POLARFIRE_ARCH_COMMAND: [
                _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="riscv64\n", duration_s=0.2)
            ],
            mod.POLARFIRE_PYTHON_COMMAND: [
                _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.9.18\n")
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(mod.POLARFIRE_UPTIME_COMMAND, stdout=" up 1 day\n")
            ],
            mod.POLARFIRE_KERNEL_COMMAND: [
                _probe(mod.POLARFIRE_KERNEL_COMMAND, stdout="kernel\n")
            ],
        }
    )
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())

    assert artifact["gatemate_terminal_state"] == "blocked_gatemate_detect_failed_no_idcode_terminal"
    assert artifact["board_matrix"]["gatemate"]["cable_state_inference"] == (
        "detect_failed_or_no_dirtyjtag_response"
    )
    assert artifact["board_matrix"]["polarfire"]["dispatch_precheck"]["blockers"] == [
        "polarfire_python_precheck_failed"
    ]
    assert mod.parse_python_version("python missing") is None
    assert mod._observed(None) == ""
    assert mod._round_duration("bad") == 0.0001
    assert mod._duration_number("bad") == 0.0

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))

    bad_probes = dict(artifact, command_probes=[])
    bad_probes["reproducibility_checksum"] = mod.payload_checksum(bad_probes)
    assert "command_probes must be a dict" in mod.artifact_schema_errors(bad_probes)

    bad_matrix_type = dict(artifact, board_matrix=[])
    bad_matrix_type["reproducibility_checksum"] = mod.payload_checksum(bad_matrix_type)
    assert "board_matrix must be a dict" in mod.artifact_schema_errors(bad_matrix_type)

    bad_matrix_keys = dict(artifact, board_matrix={"kv260": {}})
    bad_matrix_keys["reproducibility_checksum"] = mod.payload_checksum(bad_matrix_keys)
    assert "board_matrix keys mismatch" in mod.artifact_schema_errors(bad_matrix_keys)

    bad_row = dict(
        artifact,
        board_matrix={"kv260": [], "gatemate": {}, "polarfire": {}},
    )
    bad_row["reproducibility_checksum"] = mod.payload_checksum(bad_row)
    assert "kv260 row invalid" in mod.artifact_schema_errors(bad_row)
