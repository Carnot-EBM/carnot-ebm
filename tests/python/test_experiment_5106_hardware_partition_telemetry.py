"""Tests for Exp 5106 hardware partition telemetry.

Spec refs: REQ-HW-5106, SCENARIO-HW-5106.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5106_hardware_partition_telemetry as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5106 runner with queued safe command transcripts."""

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
    """Deterministic clock for REQ-HW-5106 duration-floor assertions."""

    def __call__(self) -> float:
        return 5106.0


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
            mod.KV260_UIO_LIST_COMMAND: [
                _probe(mod.KV260_UIO_LIST_COMMAND, stdout="/dev/uio0\n/dev/uio1\n", duration_s=0.2)
            ],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_USB_EVIDENCE_COMMAND: [
                _probe(
                    mod.GATEMATE_USB_EVIDENCE_COMMAND,
                    stdout=(
                        "Bus 001 Device 006: ID 1209:c0ca Generic DirtyJTAG\n"
                        "Bus 001 Device 007: ID 1514:2008 Microchip FlashPro5\n"
                    ),
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


def test_req_hw_5106_spec_declares_partition_telemetry_contract() -> None:
    """REQ-HW-5106: OpenSpec anchors v468 fields, commands, and no-speedup scope."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5106",
        "SCENARIO-HW-5106",
        "results/experiment_5106_hardware_partition_telemetry_v468.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
        "hardware_smoke_and_static_mapping",
        "kv260_uio_transcript_collected",
        "polarfire_dispatch_precheck",
        "partition_telemetry",
        "complete_hardware_partition_telemetry_no_speedup_claim",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5106_ready_prechecks_write_no_speedup_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5106: safe board checks plus static mapping produce the artifact."""

    _safe_uio_transcript(tmp_path)
    runner = _all_ready_runner()

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_UIO_LIST_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.GATEMATE_USB_EVIDENCE_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_ARCH_COMMAND,
        mod.POLARFIRE_PYTHON_COMMAND,
        mod.POLARFIRE_UPTIME_COMMAND,
        mod.POLARFIRE_KERNEL_COMMAND,
    ]
    assert saved["honest_verdict"] == "complete_hardware_partition_telemetry_no_speedup_claim"
    assert saved["duration_s"] == 0.0001
    assert saved["inference_substrate"] == "hardware_smoke_and_static_mapping"
    assert saved["kv260_ssh_ready"] is True
    assert saved["kv260_uio_transcript_collected"] is True
    assert saved["kv260_blocker"] is None
    assert saved["gatemate_detected"] is True
    assert saved["gatemate_terminal_state"] == "gatemate_detected_idcode_no_flash_terminal"
    assert saved["polarfire_ssh_ready"] is True
    assert saved["polarfire_dispatch_precheck"]["ready"] is True
    assert saved["destructive_actions_allowed"] is False
    assert saved["speedup_claimed"] is False
    assert saved["flagged_adversarial"] is False
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "exit_code": 0,
            "duration_s": 0.2,
            "observed": "returncode=0",
            "discipline": "ssh_only_no_host_block_devices",
            "safety_constraints": [
                "ssh_only",
                "safe_board_side_commands_only",
                "no_host_block_device_inspection",
                "no_destructive_actions",
            ],
        },
        {
            "resource": "gatemate_detect_command",
            "available": True,
            "command": "sh -lc 'command -v openFPGALoader'",
            "exit_code": 0,
            "duration_s": 0.1,
            "observed": "/opt/oss-cad-suite/bin/openFPGALoader",
            "discipline": "command_availability_before_dirtyjtag_detect",
            "safety_constraints": [
                "detect_only",
                "no_flash",
                "no_program",
                "no_latency_claim",
            ],
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
            "exit_code": 0,
            "duration_s": 0.4,
            "observed": "returncode=0",
            "discipline": "ssh_hash_dispatch_preconditions_only",
            "safety_constraints": [
                "ssh_only",
                "no_scp",
                "no_dispatch",
                "no_flash",
            ],
        },
        {
            "resource": "destructive_actions_allowed",
            "available": False,
            "command": "policy",
            "exit_code": 0,
            "duration_s": 0.0001,
            "observed": "false",
            "discipline": "explicit_no_destructive_actions",
            "safety_constraints": ["destructive_actions_allowed_false"],
        },
    ]
    assert saved["gatemate_triage"]["usb_evidence"]["dirtyjtag_seen"] is True
    assert saved["gatemate_triage"]["toolchain_evidence"]["openfpgaloader_available"] is True
    assert saved["gatemate_triage"]["detect_evidence"]["detected_idcode"] == "0x20000001"
    assert saved["polarfire_dispatch_precheck"]["known_safe_dispatch_path"] == (
        "carnot.hardware.polarfire_dispatch_smoke.check_preconditions"
    )
    assert saved["partition_telemetry"] == mod.build_partition_telemetry()
    assert {row["mapping_kind"] for row in saved["partition_telemetry"]} == {
        "p_spin_hubo",
        "csp_neuromorphic",
        "tsu_static_mapping",
    }
    for row in saved["partition_telemetry"]:
        assert row["principle"]
        assert 0.0 <= row["coupling_density"] <= 1.0
        assert row["partition_count"] >= 1
        assert row["boundary_exchange_estimate"]
        assert row["expected_bottleneck"]
    assert "mmcblk" not in json.dumps(saved).lower()
    assert "/dev/disk" not in json.dumps(saved).lower()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5106_blocked_prechecks_preserve_exact_blockers(tmp_path: Path) -> None:
    """REQ-HW-5106: unreachable boards remain visible without unsafe follow-on commands."""

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
            mod.GATEMATE_USB_EVIDENCE_COMMAND: [
                _probe(mod.GATEMATE_USB_EVIDENCE_COMMAND, stdout="", duration_s=0.1)
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
        mod.GATEMATE_USB_EVIDENCE_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert artifact["honest_verdict"] == "complete_hardware_partition_telemetry_no_speedup_claim"
    assert artifact["kv260_ssh_ready"] is False
    assert artifact["kv260_uio_transcript_collected"] is False
    assert artifact["kv260_blocker"] == "blocked_kv260_ssh_unreachable"
    assert artifact["gatemate_detected"] is False
    assert artifact["gatemate_terminal_state"] == "blocked_gatemate_detect_command_unavailable"
    assert artifact["polarfire_ssh_ready"] is False
    assert artifact["polarfire_dispatch_precheck"]["ready"] is False
    assert artifact["polarfire_dispatch_precheck"]["blockers"] == ["polarfire_ssh_unreachable"]
    assert artifact["command_probes"]["kv260_uio_devices"] is None
    assert artifact["command_probes"]["gatemate_dirtyjtag_detect"] is None
    assert artifact["command_probes"]["polarfire_arch"] is None
    assert artifact["destructive_actions_allowed"] is False
    assert artifact["speedup_claimed"] is False
    mod.validate_artifact(artifact)


def test_scenario_hw_5106_partial_detect_and_dispatch_blockers(tmp_path: Path) -> None:
    """SCENARIO-HW-5106: GateMate terminal state and PolarFire blockers are preserved."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_UIO_LIST_COMMAND: [
                _probe(mod.KV260_UIO_LIST_COMMAND, stdout="/dev/uio0\n", duration_s=0.2)
            ],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/usr/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_USB_EVIDENCE_COMMAND: [
                _probe(
                    mod.GATEMATE_USB_EVIDENCE_COMMAND,
                    stdout="Bus 001 Device 006: ID 1209:c0ca Generic DirtyJTAG\n",
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
                _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.9.18\n", duration_s=0.2)
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

    assert artifact["kv260_ssh_ready"] is True
    assert artifact["kv260_uio_transcript_collected"] is False
    assert artifact["kv260_blocker"] == "no_safe_kv260_uio_register_transcript_collected"
    assert artifact["gatemate_detected"] is False
    assert artifact["gatemate_terminal_state"] == (
        "blocked_gatemate_dirtyjtag_cable_seen_no_gatemate_idcode_terminal"
    )
    assert artifact["gatemate_triage"]["usb_evidence"]["dirtyjtag_seen"] is True
    assert artifact["polarfire_ssh_ready"] is True
    assert artifact["polarfire_dispatch_precheck"]["ready"] is False
    assert artifact["polarfire_dispatch_precheck"]["blockers"] == [
        "polarfire_arch_not_riscv64",
        "polarfire_python_precheck_failed",
    ]
    mod.validate_artifact(artifact)


def test_scenario_hw_5106_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5106: run_experiment writes the requested v468 artifact."""

    _safe_uio_transcript(tmp_path)
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["experiment_id"] == 5106
    assert artifact["spec_refs"] == ["REQ-HW-5106", "SCENARIO-HW-5106"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5106_validation_rejects_overclaim_and_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-5106: validation rejects speedup, destructive, storage, and bad telemetry."""

    _safe_uio_transcript(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
    )

    bad_speedup = dict(artifact, speedup_claimed=True)
    bad_speedup["reproducibility_checksum"] = mod.payload_checksum(bad_speedup)
    with pytest.raises(ValueError, match="speedup"):
        mod.validate_artifact(bad_speedup)

    bad_allowed = dict(artifact, destructive_actions_allowed=True)
    bad_allowed["reproducibility_checksum"] = mod.payload_checksum(bad_allowed)
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(bad_allowed)

    bad_storage = dict(artifact)
    bad_storage["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_partition = dict(artifact, partition_telemetry=[{"mapping_kind": "p_spin_hubo"}])
    bad_partition["reproducibility_checksum"] = mod.payload_checksum(bad_partition)
    with pytest.raises(ValueError, match="partition"):
        mod.validate_artifact(bad_partition)

    bad_partition_type = dict(artifact, partition_telemetry={})
    bad_partition_type["reproducibility_checksum"] = mod.payload_checksum(bad_partition_type)
    with pytest.raises(ValueError, match="partition_telemetry must be a list"):
        mod.validate_artifact(bad_partition_type)

    bad_partition_row_type = dict(artifact, partition_telemetry=["not-a-row"])
    bad_partition_row_type["reproducibility_checksum"] = mod.payload_checksum(
        bad_partition_row_type
    )
    with pytest.raises(ValueError, match="partition telemetry row invalid"):
        mod.validate_artifact(bad_partition_row_type)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_hw_5106_edge_helpers_and_schema_error_paths(tmp_path: Path) -> None:
    """REQ-HW-5106: defensive helpers handle malformed probes and missing fields."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_UIO_LIST_COMMAND: [
                _probe(mod.KV260_UIO_LIST_COMMAND, stdout="/dev/uio0\n", duration_s=0.2)
            ],
            mod.GATEMATE_COMMAND_AVAILABLE_COMMAND: [
                _probe(
                    mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
                    stdout="/usr/bin/openFPGALoader\n",
                    duration_s=0.1,
                )
            ],
            mod.GATEMATE_USB_EVIDENCE_COMMAND: [
                _probe(mod.GATEMATE_USB_EVIDENCE_COMMAND, stdout="no matching usb\n")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="no target response\n")
            ],
            mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, duration_s=0.4)],
            mod.POLARFIRE_ARCH_COMMAND: [
                _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="riscv64\n", duration_s=0.2)
            ],
            mod.POLARFIRE_PYTHON_COMMAND: [
                _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python missing\n")
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(mod.POLARFIRE_UPTIME_COMMAND, stdout=" up 1 day\n")
            ],
            mod.POLARFIRE_KERNEL_COMMAND: [_probe(mod.POLARFIRE_KERNEL_COMMAND, stdout="kernel\n")],
        }
    )
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner, clock=FlatClock())

    assert (
        artifact["gatemate_terminal_state"] == "blocked_gatemate_detect_failed_no_idcode_terminal"
    )
    assert artifact["gatemate_triage"]["usb_evidence"]["dirtyjtag_seen"] is False
    assert artifact["polarfire_dispatch_precheck"]["blockers"] == [
        "polarfire_python_precheck_failed"
    ]
    assert mod.parse_python_version("python missing") is None
    assert mod.observed(None) == ""
    assert mod.round_duration("bad") == 0.0001
    assert mod.duration_number("bad") == 0.0
    assert mod.idcode_from_text("IDCODE 0xabcdef01") == "0xabcdef01"
    assert mod.idcode_from_text("no id") is None

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))

    bad_probes = dict(artifact, command_probes=[])
    bad_probes["reproducibility_checksum"] = mod.payload_checksum(bad_probes)
    assert "command_probes must be a dict" in mod.artifact_schema_errors(bad_probes)

    bad_preconditions = dict(artifact, preconditions_checked=[])
    bad_preconditions["reproducibility_checksum"] = mod.payload_checksum(bad_preconditions)
    assert "preconditions_checked resources mismatch" in mod.artifact_schema_errors(
        bad_preconditions
    )

    bad_triage = dict(artifact, gatemate_triage=[])
    bad_triage["reproducibility_checksum"] = mod.payload_checksum(bad_triage)
    assert "gatemate_triage must be a dict" in mod.artifact_schema_errors(bad_triage)

    bad_polarfire = dict(artifact, polarfire_dispatch_precheck=[])
    bad_polarfire["reproducibility_checksum"] = mod.payload_checksum(bad_polarfire)
    assert "polarfire_dispatch_precheck must be a dict" in mod.artifact_schema_errors(bad_polarfire)
