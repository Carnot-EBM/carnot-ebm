"""Tests for Exp 5374 hardware continuity receipts.

Spec refs: REQ-HW-5374, SCENARIO-HW-5374.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5374_hardware_continuity_receipts_v489 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5374 runner with exact non-destructive command receipts."""

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
    """Deterministic clock for REQ-HW-5374 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5374.0

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
    wall_time_s: float = 0.0025,
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


def _base_probes(*, openfpga_present: bool = True) -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    openfpga_path = "/opt/oss-cad-suite/bin/openFPGALoader" if openfpga_present else ""
    openfpga_version = "openFPGALoader v1.1.1" if openfpga_present else ""
    return {
        mod.HOST_DATE_COMMAND: [
            _probe(
                mod.HOST_DATE_COMMAND,
                stdout=(
                    "host=carnot-host\n"
                    "date_utc=2026-07-07T14:00:00Z\n"
                    "date_local=2026-07-07T10:00:00-0400\n"
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
            _probe(mod.GATEMATE_USB_COMMAND, stdout="1209:c0ca DirtyJTAG\n")
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
    kv260_stderr: str = "ssh: connect to host kria port 22: No route to host\n",
    polarfire_status_exit: int = 0,
    polarfire_status_stdout: str = (
        "hostname=mpfs-disco-kit\n"
        "uname=Linux mpfs-disco-kit 6.18.17-linux4microchip-2026.04.1 riscv64\n"
        "python=Python 3.12.12\n"
    ),
    polarfire_status_stderr: str = "",
    polarfire_workload_exit: int = 0,
    polarfire_workload_stdout: str | None = None,
    gatemate_path_available: bool = False,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6.00MHz -> real 6.00MHz\nIDCode : 0x20000001\n",
    openfpga_present: bool = True,
) -> RecordingRunner:
    probes = _base_probes(openfpga_present=openfpga_present)
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
        probes[mod.POLARFIRE_WORKLOAD_COMMAND] = [
            _probe(
                mod.POLARFIRE_WORKLOAD_COMMAND,
                exit_code=polarfire_workload_exit,
                stdout=polarfire_workload_stdout
                if polarfire_workload_stdout is not None
                else _workload_stdout(),
                stderr="" if polarfire_workload_exit == 0 else "workload failed\n",
            )
        ]
    if gatemate_path_available and openfpga_present:
        probes[mod.GATEMATE_DETECT_COMMAND] = [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                exit_code=gatemate_detect_exit,
                stdout=gatemate_detect_stdout,
                stderr="" if gatemate_detect_exit == 0 else "dirtyJtag open failed\n",
            )
        ]
    return RecordingRunner(probes)


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _tests_run() -> list[dict[str, object]]:
    return [
        {
            "command": ".venv/bin/pytest tests/python/test_experiment_5374_hardware_continuity_receipts_v489.py -q",
            "outcome": "passed in test fixture",
        }
    ]


def test_req_hw_5374_spec_declares_required_receipt_fields() -> None:
    """REQ-HW-5374: OpenSpec anchors the v489 continuity receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5374") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5374",
        "SCENARIO-HW-5374",
        str(mod.RESULT_RELATIVE_PATH),
        mod.KV260_REQUIRED_COMMAND_FORM,
        "hardware_speedup_claim",
        "kv260_checked_via_ssh",
        "kv260_status",
        "polarfire_status",
        "polarfire_workload_hash",
        "gatemate_status",
        "commands_run",
        "no_host_mmcblk_kv260_evidence",
        "no_destructive_flash",
        "repeatability_evidence_present",
        "honest_verdict",
        "reachable/workload_receipt",
        "blocked_physical_or_jtag",
    ):
        assert marker in section


def test_scenario_hw_5374_records_fresh_receipts_without_speedup() -> None:
    """SCENARIO-HW-5374: fresh board receipts remain continuity-only."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260707",
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
    ]
    assert mod.GATEMATE_DETECT_COMMAND not in runner.commands
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "kv260_checked_via_ssh") is True
    assert _value(artifact, "kv260_status") == "unreachable"
    assert _value(artifact, "polarfire_status") == "reachable/workload_receipt"
    assert _value(artifact, "polarfire_workload_hash") == mod.POLARFIRE_EXPECTED_OUTPUT_SHA256
    assert _value(artifact, "gatemate_status") == "blocked_physical_or_jtag"
    assert _value(artifact, "no_host_mmcblk_kv260_evidence") is True
    assert _value(artifact, "no_destructive_flash") is True
    assert _value(artifact, "repeatability_evidence_present") is False
    assert "hardware_speedup_claim=false" in _value(artifact, "honest_verdict")
    assert "kv260=unreachable" in _value(artifact, "honest_verdict")
    assert "polarfire=reachable/workload_receipt" in _value(artifact, "honest_verdict")
    assert "gatemate=blocked_physical_or_jtag" in _value(artifact, "honest_verdict")

    commands = _value(artifact, "commands_run")
    assert isinstance(commands, list)
    assert any(command["command"] == mod.KV260_REQUIRED_COMMAND_FORM for command in commands)
    assert commands[-1]["kind"] == "polarfire_board_local_workload_receipt"
    assert "stdout_sha256" in commands[-1]
    assert "stderr_sha256" in commands[-1]
    assert len(commands[-1]["stdout_excerpt"]) <= mod.MAX_OUTPUT_EXCERPT_CHARS
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_polarfire_unreachable_stops_workload_lane_honestly() -> None:
    """REQ-HW-5374: unreachable PolarFire SSH leaves the workload hash null."""

    runner = _runner(
        polarfire_status_exit=255,
        polarfire_status_stdout="",
        polarfire_status_stderr="ssh: connect to host polarfire port 22: No route to host\n",
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert mod.POLARFIRE_WORKLOAD_COMMAND not in runner.commands
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "polarfire_status") == "unreachable"
    assert _value(artifact, "polarfire_workload_hash") is None
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == "unreachable"
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact)


def test_invalid_polarfire_workload_hash_fails_closed() -> None:
    """REQ-HW-5374: mismatched workload hashes do not authenticate a receipt."""

    runner = _runner(polarfire_workload_stdout=_workload_stdout(output_sha256="0" * 64))
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert _value(artifact, "polarfire_status") == "skipped: workload receipt invalid"
    assert _value(artifact, "polarfire_workload_hash") is None
    assert "output_sha256 mismatch" in artifact["blocked_reason"]["PolarFire"]["reason"]
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact)


def test_gatemate_detect_when_physical_path_available() -> None:
    """SCENARIO-HW-5374: GateMate detect remains status-only when available."""

    runner = _runner(kv260_exit=0, kv260_stderr="", gatemate_path_available=True)
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )

    assert mod.GATEMATE_DETECT_COMMAND in runner.commands
    assert _value(artifact, "kv260_status") == "reachable"
    assert _value(artifact, "gatemate_status") == "detected"
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "no_destructive_flash") is True
    mod.validate_artifact(artifact)


def test_gatemate_detect_failure_and_missing_toolchain_are_honest() -> None:
    """REQ-HW-5374: GateMate unavailable paths report blockers, not speedups."""

    failed_runner = _runner(
        gatemate_path_available=True,
        gatemate_detect_exit=1,
        gatemate_detect_stdout="",
    )
    failed = mod.build_artifact(
        command_runner=failed_runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )
    assert _value(failed, "gatemate_status") == "unreachable"
    assert failed["blocked_reason"]["GateMate"]["reason"] == "detect_failed"
    mod.validate_artifact(failed)

    missing_tool_runner = _runner(
        gatemate_path_available=True,
        openfpga_present=False,
    )
    missing_tool = mod.build_artifact(
        command_runner=missing_tool_runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )
    assert mod.GATEMATE_DETECT_COMMAND not in missing_tool_runner.commands
    assert _value(missing_tool, "gatemate_status") == "skipped: openFPGALoader unavailable"
    mod.validate_artifact(missing_tool)


def test_parser_and_status_helpers_cover_non_json_and_skips() -> None:
    """REQ-HW-5374: helper paths preserve malformed output and skipped boards."""

    excerpt = mod.short_excerpt("x" * (mod.MAX_OUTPUT_EXCERPT_CHARS + 20))
    assert excerpt.endswith("chars>")
    assert len(excerpt) <= mod.MAX_OUTPUT_EXCERPT_CHARS

    receipt, error = mod.parse_polarfire_workload_stdout("not json\n")
    assert receipt is None
    assert error == "workload stdout is not valid JSON"

    receipt, error = mod.parse_polarfire_workload_stdout(
        "\n"
        + json.dumps(
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

    assert mod._status_is_allowed("skipped: operator disabled") is True
    assert mod._status_is_allowed("") is False
    assert mod.gatemate_path_available_from_context({"hardware_environment": []}, None) is False
    assert mod.openfpgaloader_present({"tool_versions": []}) is False


def test_validator_rejects_speedup_destructive_and_kv260_host_storage_drift() -> None:
    """REQ-HW-5374: validator fails closed on speedup and unsafe evidence drift."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["hardware_speedup_claim"]["value"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claim must be false"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["commands_run"]["value"].append(
        {
            "kind": "bad",
            "command": "openFPGALoader --write flash.bit",
            "timeout_s": 1.0,
            "exit_code": 0,
            "duration_s": 0.001,
            "outcome": "bad",
            "stdout_excerpt": "",
            "stderr_excerpt": "",
            "stdout_sha256": mod.sha256_text(""),
            "stderr_sha256": mod.sha256_text(""),
        }
    )
    with pytest.raises(AssertionError, match="destructive command"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["commands_run"]["value"][0]["stdout_excerpt"] = "host path /dev/mmcblk0"
    with pytest.raises(AssertionError, match="host KV260 block-device evidence"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5374: run_experiment writes the requested v489 JSON artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact)


def test_default_tests_run_keeps_cli_artifacts_valid() -> None:
    """REQ-HW-5374: CLI-style artifacts still carry verification provenance."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
    )

    tests_run = _value(artifact, "tests_run")
    assert tests_run == [
        {
            "command": "verification not yet attached at artifact generation",
            "outcome": "pending_external_test_run",
        }
    ]
    mod.validate_artifact(artifact)
