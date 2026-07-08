"""Tests for Exp 5411 safe hardware repeatability restoration receipts.

Spec refs: REQ-HW-5411, SCENARIO-HW-5411.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5411_hardware_repeatability_restoration_v492 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5411 runner with exact safe command expectations."""

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


class StepClock:
    """Deterministic clock for REQ-HW-5411 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5411.0

    def __call__(self) -> float:
        self.value += 0.125
        return self.value


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _workload_stdout(
    *,
    input_sha256: str | None = None,
    output_sha256: str | None = None,
    wall_time_s: float = 0.1,
) -> str:
    return (
        json.dumps(
            {
                "hostname": "mpfs-disco-kit",
                "input_sha256": input_sha256 or mod.POLARFIRE_EXPECTED_INPUT_SHA256,
                "output_sha256": output_sha256 or mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
                "python_version": "3.12.12",
                "uname": "Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64",
                "wall_time_s": wall_time_s,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _base_probes(
    *, openfpga_present: bool = True, gatemate_usb_visible: bool = True
) -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    openfpga_path = "/opt/oss-cad-suite/bin/openFPGALoader" if openfpga_present else ""
    openfpga_version = "openFPGALoader v1.1.1" if openfpga_present else ""
    gatemate_usb_stdout = "1209:c0ca DirtyJTAG\n" if gatemate_usb_visible else ""
    gatemate_usb_exit = 0 if gatemate_usb_visible else 1
    return {
        mod.HOST_DATE_COMMAND: [
            _probe(
                mod.HOST_DATE_COMMAND,
                stdout=(
                    "host=carnot-host\n"
                    "date_utc=2026-07-08T14:00:00Z\n"
                    "date_local=2026-07-08T10:00:00-0400\n"
                ),
            )
        ],
        mod.HARDWARE_ENV_COMMAND: [
            _probe(
                mod.HARDWARE_ENV_COMMAND,
                stdout=(
                    "CARNOT_MODE=live\n"
                    "EXTROPIC_API_KEY=do-not-record\n"
                    "KONA_API_KEY=also-hidden\n"
                    "PRIVATE_TOKEN=hidden\n"
                ),
            )
        ],
        mod.TOOL_VERSION_COMMAND: [
            _probe(
                mod.TOOL_VERSION_COMMAND,
                stdout=(
                    "ssh_path=/usr/bin/ssh\nssh_version=OpenSSH_10.0p1\n"
                    f"openFPGALoader_path={openfpga_path}\n"
                    f"openFPGALoader_version={openfpga_version}\n"
                    "yosys_path=/opt/oss-cad-suite/bin/yosys\n"
                    "yosys_version=Yosys 0.64\n"
                    "nextpnr-himbaechel_path=/opt/oss-cad-suite/bin/nextpnr-himbaechel\n"
                    "nextpnr-himbaechel_version=nextpnr-himbaechel 0.8\n"
                    "gmpack_path=/opt/oss-cad-suite/bin/gmpack\n"
                    "gmpack_version=gmpack 2026.04\n"
                    "lsusb_path=/usr/bin/lsusb\nlsusb_version=lsusb (usbutils) 018\n"
                ),
            )
        ],
        mod.GATEMATE_USB_COMMAND: [
            _probe(
                mod.GATEMATE_USB_COMMAND,
                exit_code=gatemate_usb_exit,
                stdout=gatemate_usb_stdout,
                stderr="" if gatemate_usb_visible else "USB device not found\n",
            )
        ],
        mod.POLARFIRE_USB_COMMAND: [
            _probe(mod.POLARFIRE_USB_COMMAND, stdout="1514:2008 FlashPro5\n")
        ],
        mod.GPU_CONTEXT_COMMAND: [
            _probe(
                mod.GPU_CONTEXT_COMMAND,
                stdout=("NVIDIA GeForce RTX 3090, 24576 MiB\nNVIDIA GeForce RTX 3090, 24576 MiB\n"),
            )
        ],
    }


def _runner(
    *,
    kv260_exit: int = 255,
    kv260_stdout: str = "",
    kv260_stderr: str = "ssh: Could not resolve hostname kria: Name or service not known\n",
    polarfire_status_exit: int = 0,
    polarfire_status_stdout: str = (
        "hostname=mpfs-disco-kit\n"
        "uname=Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64\n"
        "python=Python 3.12.12\n"
    ),
    polarfire_status_stderr: str = "",
    polarfire_workload_stdout: list[str] | None = None,
    polarfire_workload_exit: int = 0,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "GateMate Series GM1Ax IDCODE 0x20000001\n",
    openfpga_present: bool = True,
    gatemate_usb_visible: bool = True,
) -> RecordingRunner:
    probes = _base_probes(
        openfpga_present=openfpga_present,
        gatemate_usb_visible=gatemate_usb_visible,
    )
    probes[mod.KV260_SSH_TRUE_COMMAND] = [
        _probe(
            mod.KV260_SSH_TRUE_COMMAND,
            exit_code=kv260_exit,
            stdout=kv260_stdout,
            stderr=kv260_stderr,
        )
    ]
    probes[mod.POLARFIRE_STATUS_COMMAND] = [
        _probe(
            mod.POLARFIRE_STATUS_COMMAND,
            exit_code=polarfire_status_exit,
            stdout=polarfire_status_stdout,
            stderr=polarfire_status_stderr,
        )
    ]
    if polarfire_status_exit == 0:
        outputs = polarfire_workload_stdout or [
            _workload_stdout(wall_time_s=0.10),
            _workload_stdout(wall_time_s=0.12),
            _workload_stdout(wall_time_s=0.11),
        ]
        probes[mod.POLARFIRE_WORKLOAD_COMMAND] = [
            _probe(
                mod.POLARFIRE_WORKLOAD_COMMAND,
                exit_code=polarfire_workload_exit,
                stdout=stdout,
                stderr="" if polarfire_workload_exit == 0 else "workload failed\n",
            )
            for stdout in outputs
        ]
    if openfpga_present and gatemate_usb_visible:
        probes[mod.GATEMATE_DETECT_COMMAND] = [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=gatemate_detect_exit,
                stdout=gatemate_detect_stdout,
                stderr="" if gatemate_detect_exit == 0 else "dirtyJtag open failed\n",
            )
        ]
    return RecordingRunner(probes)


def _tests_run() -> list[dict[str, object]]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5411_hardware_repeatability_restoration_v492.py -q",
            "outcome": "passed in test fixture",
        }
    ]


def test_req_hw_5411_spec_declares_restoration_contract() -> None:
    """REQ-HW-5411: OpenSpec anchors the v492 restoration receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5411") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5411",
        "SCENARIO-HW-5411",
        str(mod.RESULT_RELATIVE_PATH),
        mod.KV260_REQUIRED_COMMAND_FORM,
        "preconditions_checked",
        "kv260_ssh_reachable",
        "kv260_host_sd_probe_used",
        "polarfire_reachable",
        "polarfire_repeat_count",
        "polarfire_repeat_hashes",
        "gatemate_reachable",
        "gatemate_destructive_probe_used",
        "repeated_same_workload_ready",
        "hardware_speedup_claim",
        "inference_substrate",
        "honest_verdict",
    ):
        assert marker in section


def test_scenario_hw_5411_builds_required_raw_receipt_fields() -> None:
    """SCENARIO-HW-5411: reachable PolarFire repeats restore same-workload evidence."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert runner.commands == [
        mod.HOST_DATE_COMMAND,
        mod.HARDWARE_ENV_COMMAND,
        mod.TOOL_VERSION_COMMAND,
        mod.GATEMATE_USB_COMMAND,
        mod.POLARFIRE_USB_COMMAND,
        mod.GPU_CONTEXT_COMMAND,
        mod.KV260_SSH_TRUE_COMMAND,
        mod.POLARFIRE_STATUS_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.POLARFIRE_WORKLOAD_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert artifact["preconditions_checked"] is True
    assert artifact["kv260_ssh_reachable"] is False
    assert artifact["kv260_host_sd_probe_used"] is False
    assert artifact["polarfire_reachable"] is True
    assert artifact["polarfire_repeat_count"] == 3
    assert artifact["polarfire_repeat_hashes"] == [mod.POLARFIRE_EXPECTED_OUTPUT_SHA256] * 3
    assert artifact["gatemate_reachable"] is True
    assert artifact["gatemate_destructive_probe_used"] is False
    assert artifact["repeated_same_workload_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "hardware_speedup_claim=false" in artifact["honest_verdict"]
    assert artifact["board_details"]["PolarFire"]["repeat_timing_variance"] == pytest.approx(
        0.000066666667
    )
    assert artifact["board_details"]["GateMate"]["board_identity"] == (
        "GateMate Series GM1Ax IDCODE 0x20000001"
    )
    assert artifact["blocked_reason"]["KV260"]["command"] == mod.KV260_REQUIRED_COMMAND_FORM
    assert all("timestamp_utc" in command for command in artifact["commands_run"])
    assert all(len(command["command_sha256"]) == 64 for command in artifact["commands_run"])
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_polarfire_unreachable_does_not_synthesize_repeatability() -> None:
    """REQ-HW-5411: unreachable PolarFire records the precondition failure."""

    runner = _runner(
        polarfire_status_exit=255,
        polarfire_status_stdout="",
        polarfire_status_stderr="ssh: connect to host polarfire port 22: No route to host\n",
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert mod.POLARFIRE_WORKLOAD_COMMAND not in runner.commands
    assert artifact["polarfire_reachable"] is False
    assert artifact["polarfire_repeat_count"] == 0
    assert artifact["polarfire_repeat_hashes"] == []
    assert artifact["repeated_same_workload_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == "unreachable"
    assert artifact["honest_verdict"].startswith("blocked:")
    mod.validate_artifact(artifact)


def test_hash_drift_keeps_repeatability_blocked() -> None:
    """REQ-HW-5411: repeated timing without hash agreement is not ready evidence."""

    runner = _runner(
        polarfire_workload_stdout=[
            _workload_stdout(wall_time_s=0.10),
            _workload_stdout(output_sha256="0" * 64, wall_time_s=0.12),
            _workload_stdout(wall_time_s=0.11),
        ]
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert artifact["polarfire_repeat_count"] == 3
    assert artifact["polarfire_repeat_hashes"] == [
        mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
        "0" * 64,
        mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
    ]
    assert artifact["repeated_same_workload_ready"] is False
    assert artifact["board_details"]["PolarFire"]["repeatability_class"] == (
        "non_reproducible_output_hash_drift"
    )
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == "output_sha256 mismatch"
    assert artifact["hardware_speedup_claim"] is False
    mod.validate_artifact(artifact)


def test_gatemate_tool_or_usb_precondition_failure_skips_detect() -> None:
    """REQ-HW-5411: GateMate detect runs only when non-destructive preconditions exist."""

    missing_tool_runner = _runner(openfpga_present=False)
    missing_tool = mod.build_artifact(
        command_runner=missing_tool_runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    assert mod.GATEMATE_DETECT_COMMAND not in missing_tool_runner.commands
    assert missing_tool["gatemate_reachable"] is False
    assert missing_tool["blocked_reason"]["GateMate"]["reason"] == "openFPGALoader unavailable"
    mod.validate_artifact(missing_tool)

    missing_usb_runner = _runner(gatemate_usb_visible=False)
    missing_usb = mod.build_artifact(
        command_runner=missing_usb_runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    assert mod.GATEMATE_DETECT_COMMAND not in missing_usb_runner.commands
    assert missing_usb["gatemate_reachable"] is False
    assert missing_usb["blocked_reason"]["GateMate"]["reason"] == "dirtyjtag usb not visible"
    mod.validate_artifact(missing_usb)


def test_parsers_and_validator_fail_closed_on_unsafe_or_fabricated_evidence() -> None:
    """REQ-HW-5411: validation rejects speedup, storage, destructive, and wrong KV260 evidence."""

    receipt, error = mod.parse_polarfire_workload_stdout("\nnot json\n")
    assert receipt is None
    assert error == "workload stdout is not valid JSON"

    receipt, error = mod.parse_polarfire_workload_stdout(
        json.dumps(
            {
                "hostname": "",
                "input_sha256": "1" * 64,
                "output_sha256": mod.POLARFIRE_EXPECTED_OUTPUT_SHA256,
                "python_version": 312,
                "uname": "Linux mpfs-disco-kit riscv64",
                "wall_time_s": -1.0,
            },
            sort_keys=True,
        )
    )
    assert isinstance(receipt, dict)
    assert error is not None
    assert "hostname missing" in error
    assert "input_sha256 mismatch" in error
    assert "wall_time_s invalid" in error
    assert "python_version invalid" in error
    assert mod.receipt_timestamp("20260708", 61) == "2026-07-08T00:01:01Z"
    assert mod.timing_variance([0.1]) is None
    assert mod._mapping_value({"a": 1}, "a", "b") is None
    with pytest.raises(ValueError, match="run_date"):
        mod.receipt_timestamp("2026-07-08", 0)

    malformed_runner = _runner(polarfire_workload_stdout=["not json\n", "not json\n", "not json\n"])
    malformed = mod.build_artifact(
        command_runner=malformed_runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    assert malformed["board_details"]["PolarFire"]["repeatability_class"] == (
        "insufficient_valid_board_local_repeats"
    )
    assert malformed["repeated_same_workload_ready"] is False
    mod.validate_artifact(malformed)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["hardware_speedup_claim"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claim"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["kv260_host_sd_probe_used"] = True
    with pytest.raises(AssertionError, match="kv260_host_sd_probe_used"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["commands_run"][6]["command"] = "ssh kria 'ls /dev/mmcblk*'"
    with pytest.raises(AssertionError, match="KV260 command"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["commands_run"].append(
        {
            "kind": "bad",
            "board": "GateMate",
            "timestamp_utc": "2026-07-08T00:00:30Z",
            "command": "openFPGALoader --write flash.bit",
            "command_path": "openFPGALoader",
            "command_sha256": "0" * 64,
            "timeout_s": 1.0,
            "exit_code": 0,
            "duration_s": 0.001,
            "outcome": "bad",
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
        }
    )
    with pytest.raises(AssertionError, match="destructive"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5411: run_experiment writes the requested v492 JSON artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["inference_substrate"] == "hardware_smoke"
    mod.validate_artifact(artifact)


def test_default_tests_run_keeps_cli_artifacts_valid() -> None:
    """REQ-HW-5411: CLI-style artifacts still carry verification provenance."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
    )

    assert artifact["tests_run"] == [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]
    mod.validate_artifact(artifact)
