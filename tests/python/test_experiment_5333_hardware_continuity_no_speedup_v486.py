"""Tests for Exp 5333 hardware continuity receipts without speedup claims.

Spec refs: REQ-HW-5333, SCENARIO-HW-5333.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5333_hardware_continuity_no_speedup_v486 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5333 runner with exact non-destructive probe receipts."""

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
    """Deterministic clock for REQ-HW-5333 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5333.0

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


def _base_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.HOST_DATE_COMMAND: [
            _probe(
                mod.HOST_DATE_COMMAND,
                stdout=(
                    "host=carnot-host\n"
                    "date_utc=2026-07-07T13:00:00Z\n"
                    "date_local=2026-07-07T09:00:00-0400\n"
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
                    "openFPGALoader_path=/opt/oss-cad-suite/bin/openFPGALoader\n"
                    "openFPGALoader_version=openFPGALoader v1.1.1\n"
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
    polarfire_exit: int = 0,
    polarfire_stdout: str = "hostname=mpfs-disco-kit\nuname=Linux mpfs-disco-kit riscv64\n",
    polarfire_stderr: str = "",
    gatemate_setup_changed: bool = False,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6.00MHz -> real 6.00MHz\nIDCode : 0x20000001\n",
) -> RecordingRunner:
    probes = _base_probes()
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
            exit_code=polarfire_exit,
            stdout=polarfire_stdout,
            stderr=polarfire_stderr,
        )
    ]
    if gatemate_setup_changed:
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
            "command": ".venv/bin/pytest tests/python/test_experiment_5333_hardware_continuity_no_speedup_v486.py -q",
            "outcome": "passed in test fixture",
        }
    ]


def test_req_hw_5333_spec_declares_v486_required_fields() -> None:
    """REQ-HW-5333: OpenSpec anchors the v486 no-speedup receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5333") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5333",
        "SCENARIO-HW-5333",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "experiment_id",
        "milestone",
        "status",
        "honest_verdict",
        "preconditions_checked",
        "kv260_status",
        "polarfire_status",
        "gatemate_status",
        "authenticated_workload_run",
        "public_refs_context_only",
        "speedup_claim",
        "no_host_block_device_evidence",
        "tests_run",
        "Extropic/TSU",
        "Logical/Kona",
    ):
        assert marker in section


def test_scenario_hw_5333_records_reachability_without_workload_or_speedup() -> None:
    """SCENARIO-HW-5333: reachability receipts do not become workload or speedup claims."""

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
    ]
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "blocked_kv260_ssh_no_authenticated_workload"
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "honest_verdict").startswith("blocked_")
    assert "authenticated_workload_run=false" in _value(artifact, "honest_verdict")
    assert "public_refs=context_only" in _value(artifact, "honest_verdict")
    assert "speedup_claim=false" in _value(artifact, "honest_verdict")
    assert artifact["authenticated_workload_run"] is False
    assert artifact["public_refs_context_only"] is True
    assert artifact["speedup_claim"] is False
    assert artifact["no_host_block_device_evidence"] is True

    kv260 = _value(artifact, "kv260_status")
    polarfire = _value(artifact, "polarfire_status")
    gatemate = _value(artifact, "gatemate_status")
    commands = _value(artifact, "commands_run")
    preconditions = _value(artifact, "preconditions_checked")
    assert isinstance(kv260, dict)
    assert isinstance(polarfire, dict)
    assert isinstance(gatemate, dict)
    assert isinstance(commands, list)
    assert isinstance(preconditions, dict)
    assert kv260["status"] == "blocked_kv260_ssh_unreachable"
    assert kv260["command_form"] == mod.KV260_REQUIRED_COMMAND_FORM
    assert polarfire["status"] == "reachable_ssh_status_only"
    assert gatemate["status"] == "blocked_gatemate_physical_jtag_setup_unchanged"
    assert preconditions["kv260_check_method"] == "ssh_batchmode_true_only"
    assert preconditions["operator_visible_hardware_assumptions"]["no_speedup_claim"] is True
    assert commands[-2]["command"] == mod.KV260_REQUIRED_COMMAND_FORM
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_reachable_statuses_and_changed_gatemate_still_do_not_authenticate_workload() -> None:
    """REQ-HW-5333: reachable board status and JTAG evidence preserve no-speedup."""

    runner = _runner(
        kv260_exit=0,
        kv260_stdout="",
        kv260_stderr="",
        polarfire_exit=0,
        polarfire_stdout="hostname=polarfire\nuname=Linux polarfire 6.18.17 riscv64\n",
        polarfire_stderr="",
        gatemate_setup_changed=True,
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        gatemate_setup_changed=True,
        tests_run=_tests_run(),
    )

    assert mod.GATEMATE_DETECT_COMMAND in runner.commands
    assert _value(artifact, "status") == "complete_status_receipts_no_authenticated_workload"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "kv260_status")["ssh_reachable"] is True
    assert _value(artifact, "polarfire_status")["ssh_reachable"] is True
    assert _value(artifact, "gatemate_status")["status"] == "reachable_dirtyjtag_idcode_status_only"
    assert artifact["authenticated_workload_run"] is False
    assert artifact["speedup_claim"] is False
    assert artifact["public_refs_context_only"] is True
    mod.validate_artifact(artifact)


def test_status_value_covers_polarfire_and_gatemate_blockers() -> None:
    """REQ-HW-5333: status summaries preserve non-KV260 blockers."""

    assert (
        mod.status_value(
            kv260_reachable=True,
            polarfire_reachable=False,
            gatemate_blocked_after_changed_setup=False,
        )
        == "blocked_polarfire_status_no_authenticated_workload"
    )
    assert (
        mod.status_value(
            kv260_reachable=True,
            polarfire_reachable=True,
            gatemate_blocked_after_changed_setup=True,
        )
        == "blocked_gatemate_jtag_no_authenticated_workload"
    )


def test_validator_rejects_boolean_and_host_block_device_drift() -> None:
    """REQ-HW-5333: validator fails closed on workload, speedup, and host-device drift."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )

    artifact["authenticated_workload_run"] = True
    with pytest.raises(AssertionError, match="authenticated_workload_run must be false"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["speedup_claim"] = True
    with pytest.raises(AssertionError, match="speedup_claim must be false"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    artifact["no_host_block_device_evidence"] = False
    with pytest.raises(AssertionError, match="no_host_block_device_evidence must be true"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260707",
        commit="abc123",
        tests_run=_tests_run(),
    )
    _value(artifact, "commands_run").append(
        {
            "command": "ls /dev/mmcblk0",
            "outcome": "bad",
            "exit_code": 0,
            "timeout_s": 1.0,
            "duration_s": 0.001,
        }
    )
    artifact["no_host_block_device_evidence"] = True
    with pytest.raises(AssertionError, match="host block-device marker present"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5333: run_experiment writes the requested v486 JSON artifact."""

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
    assert artifact["speedup_claim"] is False
    mod.validate_artifact(artifact)


def test_default_tests_run_keeps_cli_artifacts_valid() -> None:
    """REQ-HW-5333: CLI-style artifacts still carry a principle-wrapped tests field."""

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
