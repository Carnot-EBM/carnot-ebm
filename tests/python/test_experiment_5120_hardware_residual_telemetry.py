"""Tests for Exp 5120 hardware residual telemetry.

Spec refs: REQ-HW-5120, SCENARIO-HW-5120.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5120_hardware_residual_telemetry as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5120 runner with queued authenticated command outputs."""

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
    """Deterministic clock for REQ-HW-5120 duration accounting."""

    def __call__(self) -> float:
        return 5120.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _all_ready_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_UIO_LIST_COMMAND: [
                _probe(
                    mod.KV260_UIO_LIST_COMMAND,
                    stdout="/dev/uio0\n/dev/uio1\n/dev/uio4\n",
                    duration_s=0.2,
                )
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
                _probe(mod.POLARFIRE_UPTIME_COMMAND, stdout=" up 8 days\n", duration_s=0.2)
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


def _write_safe_uio_transcript(root: Path) -> Path:
    path = root / mod.SAFE_KV260_UIO_TRANSCRIPT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "carnot.kv260.safe_uio_register_transcript.v1",
                "operation": "uio_register_read",
                "mode": "read_only",
                "safe_for_continuity_audit": True,
                "command": "ssh kria python3 read_uio_register.py",
                "workload": "exp5120_residual_probe",
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


def test_req_hw_5120_spec_declares_residual_telemetry_contract() -> None:
    """REQ-HW-5120: OpenSpec anchors residual telemetry and no-speedup fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5120",
        "SCENARIO-HW-5120",
        "results/experiment_5120_hardware_residual_telemetry_v469.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
        "hardware_smoke_and_residual_telemetry_or_cpu_fallback",
        "residual_energy_by_sweep",
        "decay_exponent",
        "hardware_residual_telemetry_ready",
        "no_speedup_claim",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5120_ready_prechecks_write_cpu_residual_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5120: board prechecks plus CPU residual samples produce artifact."""

    runner = _all_ready_runner()

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260701",
    )
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
    assert saved["experiment_id"] == "exp5120-hardware-residual-telemetry-v469"
    assert saved["milestone"] == "2026.07.469"
    assert saved["honest_verdict"] == "complete_hardware_residual_telemetry_cpu_reference_no_speedup_claim"
    assert saved["inference_substrate"] == "hardware_smoke_and_residual_telemetry_or_cpu_fallback"
    assert saved["duration_s"] == 0.0001
    assert saved["kv260_ssh_checked"] is True
    assert saved["kv260_ssh_ready"] is True
    assert saved["kv260_host_block_devices_touched"] is False
    assert saved["gatemate_checked"] is True
    assert saved["gatemate_detected"] is True
    assert saved["polarfire_checked"] is True
    assert saved["polarfire_ssh_ready"] is True
    assert saved["hardware_residual_telemetry_ready"] is True
    assert saved["no_speedup_claim"] is True
    assert saved["flagged_adversarial"] is False
    assert saved["residual_source"] == "cpu_reference_residual_sweep"
    assert saved["kv260_uio_register_status"]["safe_transcript_verified"] is False
    assert saved["kv260_uio_register_status"]["blocker"] == "no_safe_kv260_uio_register_transcript"
    assert saved["command_transcripts"]["kv260_ssh"]["exit_code"] == 0
    assert saved["command_transcripts"]["gatemate_dirtyjtag_detect"]["stdout"].startswith("IDCode")
    assert saved["command_transcripts"]["polarfire_arch"]["stdout"] == "riscv64\n"
    assert saved["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert saved["preconditions_checked"][1]["resource"] == "kv260_host_block_devices_touched"
    assert saved["preconditions_checked"][1]["available"] is False
    assert saved["preconditions_checked"][2]["resource"] == "gatemate_dirtyjtag"
    assert saved["preconditions_checked"][3]["resource"] == "polarfire_ssh"
    assert saved["residual_energy_by_sweep"] == mod.compute_cpu_residual_sweep()[0]
    assert saved["decay_exponent"] == pytest.approx(
        mod.fit_decay_exponent(saved["residual_energy_by_sweep"])
    )
    assert saved["decay_exponent"] > 1.5
    assert saved["residual_partition_telemetry"]["communication_update_ratio"] > 0.0
    assert set(saved["workload_hashes"]) == {
        "cpu_reference_residual_sweep",
        "cpu_residual_samples",
        "kv260_uio_register_transcript",
        "board_timing_workload",
    }
    assert saved["workload_hashes"]["kv260_uio_register_transcript"] is None
    assert saved["workload_hashes"]["board_timing_workload"] is None
    assert saved["tests_run"] == mod.DEFAULT_TESTS_RUN
    encoded = json.dumps(saved, sort_keys=True).lower()
    assert "mmcblk" not in encoded
    assert "/dev/disk" not in encoded
    assert "extropic" not in encoded
    assert "tsu" not in encoded
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5120_unreachable_boards_still_record_cpu_reference(tmp_path: Path) -> None:
    """REQ-HW-5120: board blockers do not suppress the CPU residual fallback."""

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
                )
            ],
            mod.GATEMATE_USB_EVIDENCE_COMMAND: [
                _probe(mod.GATEMATE_USB_EVIDENCE_COMMAND, stdout="")
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

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260701",
    )

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.GATEMATE_COMMAND_AVAILABLE_COMMAND,
        mod.GATEMATE_USB_EVIDENCE_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert artifact["kv260_ssh_ready"] is False
    assert artifact["command_transcripts"]["kv260_uio_devices"] is None
    assert artifact["gatemate_checked"] is True
    assert artifact["gatemate_detected"] is False
    assert artifact["command_transcripts"]["gatemate_dirtyjtag_detect"] is None
    assert artifact["polarfire_checked"] is True
    assert artifact["polarfire_ssh_ready"] is False
    assert artifact["command_transcripts"]["polarfire_arch"] is None
    assert artifact["hardware_residual_telemetry_ready"] is True
    assert artifact["residual_energy_by_sweep"]
    assert artifact["board_precheck_summary"]["authenticated_board_precheck_count"] == 0
    assert artifact["board_precheck_summary"]["cpu_reference_residual_sweep_recorded"] is True
    mod.validate_artifact(artifact)


def test_req_hw_5120_safe_uio_transcript_hash_and_decay_helpers(tmp_path: Path) -> None:
    """REQ-HW-5120: safe UIO evidence is hashed, while unsafe transcripts block."""

    safe_path = _write_safe_uio_transcript(tmp_path)
    runner = _all_ready_runner()

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260701",
    )

    text = safe_path.read_text(encoding="utf-8")
    assert artifact["kv260_uio_register_status"]["safe_transcript_verified"] is True
    assert artifact["kv260_uio_register_status"]["path"] == str(
        mod.SAFE_KV260_UIO_TRANSCRIPT_REL_PATH
    )
    assert artifact["workload_hashes"]["kv260_uio_register_transcript"] == mod.sha256_text(text)
    assert mod.safe_uio_transcript_text(text) is True
    assert mod.safe_uio_transcript_text(text + "write_u32") is False
    assert mod.parse_uio_devices("/dev/uio2\n/dev/uio1\n/dev/uio2\n") == [
        "/dev/uio1",
        "/dev/uio2",
    ]
    assert mod.fit_decay_exponent([{"sweep": 0, "residual_energy": 1.0}]) is None
    assert (
        mod.fit_decay_exponent(
            [{"sweep": 0, "residual_energy": 1.0}, {"sweep": 1, "residual_energy": 0.0}]
        )
        is None
    )
    assert (
        mod.fit_decay_exponent(
            [{"sweep": 0, "residual_energy": 1.0}, {"sweep": 0, "residual_energy": 0.5}]
        )
        is None
    )
    assert (
        mod.gatemate_terminal_state(
            tool_available=True,
            detected=False,
            dirtyjtag_seen=True,
        )
        == "blocked_gatemate_dirtyjtag_seen_no_idcode_terminal"
    )
    assert (
        mod.gatemate_terminal_state(
            tool_available=True,
            detected=False,
            dirtyjtag_seen=False,
        )
        == "blocked_gatemate_no_usb_or_idcode_terminal"
    )
    polarfire_runner = RecordingRunner(
        {
            mod.POLARFIRE_ARCH_COMMAND: [
                _probe(mod.POLARFIRE_ARCH_COMMAND, stdout="x86_64\n")
            ],
            mod.POLARFIRE_PYTHON_COMMAND: [
                _probe(mod.POLARFIRE_PYTHON_COMMAND, stdout="Python 3.9.18\n")
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [_probe(mod.POLARFIRE_UPTIME_COMMAND, stdout=" up\n")],
            mod.POLARFIRE_KERNEL_COMMAND: [_probe(mod.POLARFIRE_KERNEL_COMMAND, stdout="k\n")],
        }
    )
    polarfire_bundle = mod.run_polarfire_prechecks(
        polarfire_ssh_probe=_probe(mod.POLARFIRE_SSH_COMMAND),
        command_runner=polarfire_runner,
    )
    assert polarfire_bundle["blockers"] == [
        "polarfire_arch_not_riscv64",
        "polarfire_python_precheck_failed",
    ]
    assert mod.parse_python_version("python missing") is None
    assert mod.observed(None) == ""
    assert mod.round_duration("bad") == 0.0001
    assert mod.numeric("bad") == 0.0
    mod.validate_artifact(artifact)


def test_req_hw_5120_validation_rejects_overclaims_and_fake_telemetry(tmp_path: Path) -> None:
    """REQ-HW-5120: validation rejects speedup claims, fake residuals, and drift."""

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
        run_date="20260701",
    )

    bad_speedup = dict(artifact, no_speedup_claim=False)
    bad_speedup["reproducibility_checksum"] = mod.payload_checksum(bad_speedup)
    with pytest.raises(ValueError, match="speedup"):
        mod.validate_artifact(bad_speedup)

    bad_ready = dict(artifact, hardware_residual_telemetry_ready=True, residual_energy_by_sweep=[])
    bad_ready["reproducibility_checksum"] = mod.payload_checksum(bad_ready)
    with pytest.raises(ValueError, match="residual"):
        mod.validate_artifact(bad_ready)

    bad_decay = dict(artifact, decay_exponent=0.1)
    bad_decay["reproducibility_checksum"] = mod.payload_checksum(bad_decay)
    with pytest.raises(ValueError, match="decay"):
        mod.validate_artifact(bad_decay)

    bad_policy = dict(artifact, kv260_host_block_devices_touched=True)
    bad_policy["reproducibility_checksum"] = mod.payload_checksum(bad_policy)
    with pytest.raises(ValueError, match="host block"):
        mod.validate_artifact(bad_policy)

    bad_storage = dict(artifact)
    bad_storage["command_transcripts"] = {"bad": {"command": "unsafe host storage marker"}}
    bad_storage["forbidden"] = "/dev/" + "disk"
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_tests = dict(artifact, tests_run=[])
    bad_tests["reproducibility_checksum"] = mod.payload_checksum(bad_tests)
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(bad_tests)

    bad_preconditions = dict(artifact, preconditions_checked=[])
    bad_preconditions["reproducibility_checksum"] = mod.payload_checksum(bad_preconditions)
    assert "preconditions_checked resources mismatch" in mod.artifact_schema_errors(
        bad_preconditions
    )

    bad_precondition_row = dict(artifact)
    preconditions = list(artifact["preconditions_checked"])
    preconditions[2] = "bad-row"
    bad_precondition_row["preconditions_checked"] = preconditions
    bad_precondition_row["reproducibility_checksum"] = mod.payload_checksum(bad_precondition_row)
    assert "bad precondition row" in mod.artifact_schema_errors(bad_precondition_row)

    bad_transcripts = dict(artifact, command_transcripts=[])
    bad_transcripts["reproducibility_checksum"] = mod.payload_checksum(bad_transcripts)
    assert "command_transcripts must be a dict" in mod.artifact_schema_errors(bad_transcripts)

    bad_hashes = dict(artifact, workload_hashes=[])
    bad_hashes["reproducibility_checksum"] = mod.payload_checksum(bad_hashes)
    assert "workload_hashes must be a dict" in mod.artifact_schema_errors(bad_hashes)

    bad_residual_type = dict(artifact, residual_energy_by_sweep={})
    bad_residual_type["reproducibility_checksum"] = mod.payload_checksum(bad_residual_type)
    assert "residual_energy_by_sweep must be a list" in mod.artifact_schema_errors(
        bad_residual_type
    )

    bad_residual_row = dict(artifact, residual_energy_by_sweep=["bad-row"])
    bad_residual_row["reproducibility_checksum"] = mod.payload_checksum(bad_residual_row)
    assert "residual row invalid" in mod.artifact_schema_errors(bad_residual_row)

    bad_summary = dict(artifact, board_precheck_summary=[])
    bad_summary["reproducibility_checksum"] = mod.payload_checksum(bad_summary)
    assert "board_precheck_summary must be a dict" in mod.artifact_schema_errors(bad_summary)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))


def test_scenario_hw_5120_run_experiment_and_cli_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5120: run_experiment and script entrypoints write the artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_all_ready_runner(),
        clock=FlatClock(),
        run_date="20260701",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["spec_refs"] == ["REQ-HW-5120", "SCENARIO-HW-5120"]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)

    def fake_run_experiment(**kwargs: object) -> Path:
        called["kwargs"] = kwargs
        payload = dict(artifact)
        path = Path(kwargs["repo_root"]) / mod.OUTPUT_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return path

    called: dict[str, object] = {}
    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert mod.main(["--date", "20260701", "--repo-root", str(tmp_path)]) == 0
    assert called["kwargs"]["run_date"] == "20260701"
    assert "honest_verdict" in capsys.readouterr().out

    script = importlib.import_module("scripts.experiment_5120_hardware_residual_telemetry_v469")
    monkeypatch.setattr(script, "experiment_main", lambda argv: 17)
    assert script.main(["--date", "20260701"]) == 17
