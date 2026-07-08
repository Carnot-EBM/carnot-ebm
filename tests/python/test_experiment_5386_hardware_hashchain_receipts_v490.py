"""Tests for Exp 5386 hardware hash-chained receipts.

Spec refs: REQ-HW-5386, SCENARIO-HW-5386.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5386_hardware_hashchain_receipts_v490 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5386 runner with exact non-destructive command receipts."""

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
    """Deterministic clock for REQ-HW-5386 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5386.0

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
            "command": ".venv/bin/pytest tests/python/test_experiment_5386_hardware_hashchain_receipts_v490.py -q",
            "outcome": "passed in test fixture",
        }
    ]


def _assert_chain_valid(chain: list[dict[str, object]]) -> None:
    previous = mod.GENESIS_CHAIN_HASH
    for index, record in enumerate(chain):
        assert record["index"] == index
        assert record["previous_hash"] == previous
        assert record["record_hash"] == mod.hash_chain_record(record)
        assert len(record["command_sha256"]) == 64
        assert len(record["input_sha256"]) == 64
        assert len(record["output_sha256"]) == 64
        previous = str(record["record_hash"])


def test_req_hw_5386_spec_declares_hashchain_contract() -> None:
    """REQ-HW-5386: OpenSpec anchors the v490 hash-chain receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5386") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5386",
        "SCENARIO-HW-5386",
        str(mod.RESULT_RELATIVE_PATH),
        mod.KV260_REQUIRED_COMMAND_FORM,
        "hardware_hash_chained_receipt_ready",
        "hardware_speedup_claim",
        "boards_checked",
        "kv260_status",
        "polar_fire_status",
        "gatemate_status",
        "workload_hash_chain",
        "commands_run",
        "no_host_mmcblk_kv260_evidence",
        "no_destructive_flash",
        "repeatability_evidence_present",
        "receipt_contract_version",
        "honest_verdict",
        "command, input, and output hashes",
    ):
        assert marker in section


def test_scenario_hw_5386_records_hashchain_workload_receipt_without_speedup() -> None:
    """SCENARIO-HW-5386: a reachable PolarFire workload makes a valid chain."""

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
    ]
    assert mod.GATEMATE_DETECT_COMMAND not in runner.commands
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "hardware_hash_chained_receipt_ready") is True
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "boards_checked") == ["KV260", "PolarFire", "GateMate"]
    assert _value(artifact, "no_host_mmcblk_kv260_evidence") is True
    assert _value(artifact, "no_destructive_flash") is True
    assert _value(artifact, "repeatability_evidence_present") is False
    assert _value(artifact, "receipt_contract_version") == mod.RECEIPT_CONTRACT_VERSION
    assert "speedup_claim=false" in _value(artifact, "honest_verdict")
    assert "receipt_ready=true" in _value(artifact, "honest_verdict")

    kv260 = _value(artifact, "kv260_status")
    polar_fire = _value(artifact, "polar_fire_status")
    gatemate = _value(artifact, "gatemate_status")
    chain = _value(artifact, "workload_hash_chain")
    commands = _value(artifact, "commands_run")
    assert isinstance(kv260, dict)
    assert isinstance(polar_fire, dict)
    assert isinstance(gatemate, dict)
    assert isinstance(chain, list)
    assert isinstance(commands, list)
    assert kv260["status"] == "unreachable"
    assert kv260["check_method"] == "ssh_batchmode_true_only"
    assert polar_fire["status"] == "reachable/workload_receipt"
    assert polar_fire["workload_receipt"]["input_sha256"] == mod.POLARFIRE_EXPECTED_INPUT_SHA256
    assert polar_fire["workload_receipt"]["output_sha256"] == mod.POLARFIRE_EXPECTED_OUTPUT_SHA256
    assert gatemate["status"] == "blocked_physical_or_jtag"
    assert any(command["command"] == mod.KV260_REQUIRED_COMMAND_FORM for command in commands)
    assert [record["board"] for record in chain] == ["KV260", "PolarFire", "PolarFire"]
    assert chain[-1]["workload_receipt_validated"] is True
    assert chain[-1]["input_sha256"] == mod.POLARFIRE_EXPECTED_INPUT_SHA256
    assert chain[-1]["output_sha256"] == mod.POLARFIRE_EXPECTED_OUTPUT_SHA256
    _assert_chain_valid(chain)
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_polarfire_unreachable_leaves_receipt_not_ready() -> None:
    """REQ-HW-5386: no valid board workload means the hash-chain is not ready."""

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
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "hardware_hash_chained_receipt_ready") is False
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "polar_fire_status")["status"] == "unreachable"
    assert artifact["blocked_reason"]["PolarFire"]["reason"] == "unreachable"
    chain = _value(artifact, "workload_hash_chain")
    assert isinstance(chain, list)
    assert [record["board"] for record in chain] == ["KV260", "PolarFire"]
    assert all(record["workload_receipt_validated"] is False for record in chain)
    _assert_chain_valid(chain)
    mod.validate_artifact(artifact)


def test_invalid_polarfire_workload_hash_is_chained_but_not_ready() -> None:
    """REQ-HW-5386: mismatched workload hashes cannot authenticate readiness."""

    runner = _runner(polarfire_workload_stdout=_workload_stdout(output_sha256="0" * 64))
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )

    assert _value(artifact, "hardware_hash_chained_receipt_ready") is False
    assert _value(artifact, "polar_fire_status")["status"] == "skipped: workload receipt invalid"
    assert "output_sha256 mismatch" in artifact["blocked_reason"]["PolarFire"]["reason"]
    chain = _value(artifact, "workload_hash_chain")
    assert isinstance(chain, list)
    assert chain[-1]["workload_receipt_validated"] is False
    assert chain[-1]["output_sha256"] != mod.POLARFIRE_EXPECTED_OUTPUT_SHA256
    _assert_chain_valid(chain)
    mod.validate_artifact(artifact)


def test_gatemate_detect_when_physical_path_available() -> None:
    """SCENARIO-HW-5386: GateMate detect is chained only when the path is present."""

    runner = _runner(kv260_exit=0, kv260_stderr="", gatemate_path_available=True)
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )

    assert mod.GATEMATE_DETECT_COMMAND in runner.commands
    assert _value(artifact, "kv260_status")["status"] == "reachable"
    assert _value(artifact, "gatemate_status")["status"] == "detected"
    chain = _value(artifact, "workload_hash_chain")
    assert isinstance(chain, list)
    assert chain[-1]["board"] == "GateMate"
    assert chain[-1]["action"] == "safe_detect"
    assert _value(artifact, "hardware_speedup_claim") is False
    assert _value(artifact, "no_destructive_flash") is True
    _assert_chain_valid(chain)
    mod.validate_artifact(artifact)


def test_gatemate_detect_failure_and_missing_toolchain_are_honest() -> None:
    """REQ-HW-5386: GateMate unavailable paths report blockers, not speedups."""

    failed_runner = _runner(
        gatemate_path_available=True,
        gatemate_detect_exit=1,
        gatemate_detect_stdout="",
    )
    failed = mod.build_artifact(
        command_runner=failed_runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )
    assert _value(failed, "gatemate_status")["status"] == "unreachable"
    assert failed["blocked_reason"]["GateMate"]["reason"] == "detect_failed"
    mod.validate_artifact(failed)

    missing_tool_runner = _runner(
        gatemate_path_available=True,
        openfpga_present=False,
    )
    missing_tool = mod.build_artifact(
        command_runner=missing_tool_runner,
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        gatemate_physical_path_available=True,
        tests_run=_tests_run(),
    )
    assert mod.GATEMATE_DETECT_COMMAND not in missing_tool_runner.commands
    assert (
        _value(missing_tool, "gatemate_status")["status"] == "skipped: openFPGALoader unavailable"
    )
    mod.validate_artifact(missing_tool)


def test_parser_chain_and_timestamp_helpers_cover_edge_cases() -> None:
    """REQ-HW-5386: helper paths preserve malformed output and chain rules."""

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

    assert mod.receipt_timestamp("20260708", 0) == "2026-07-08T00:00:00Z"
    assert mod.receipt_timestamp("20260708", 61) == "2026-07-08T00:01:01Z"
    assert mod.board_identity("GateMate", {"dirtyjtag_usb": {"visible": True}}) == (
        "DirtyJTAG 1209:c0ca visible"
    )
    assert mod.board_identity("GateMate", {}) == "GateMate physical/JTAG path unavailable"
    assert mod.board_identity("OtherBoard", {}) == "OtherBoard"
    with pytest.raises(ValueError, match="run_date"):
        mod.receipt_timestamp("2026-07-08", 0)


def test_validator_rejects_broken_chain_speedup_destructive_and_host_storage() -> None:
    """REQ-HW-5386: validator fails closed on chain, speedup, and unsafe drift."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["workload_hash_chain"]["value"][1]["previous_hash"] = "f" * 64
    with pytest.raises(AssertionError, match="previous_hash"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["hardware_speedup_claim"]["value"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claim must be false"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
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
        run_date="20260708",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["commands_run"]["value"][0]["stdout_excerpt"] = "host path /dev/mmcblk0"
    with pytest.raises(AssertionError, match="host KV260 block-device evidence"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5386: run_experiment writes the requested v490 JSON artifact."""

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
    assert _value(artifact, "receipt_contract_version") == mod.RECEIPT_CONTRACT_VERSION
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert _value(artifact, "hardware_speedup_claim") is False
    mod.validate_artifact(artifact)


def test_default_tests_run_keeps_cli_artifacts_valid() -> None:
    """REQ-HW-5386: CLI-style artifacts still carry verification provenance."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260708",
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
