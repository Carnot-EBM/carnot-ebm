"""Tests for Exp 5319 hardware continuity no-speedup receipts.

Spec refs: REQ-HW-5319, SCENARIO-HW-5319.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5319_hardware_continuity_no_speedup_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5319 runner with exact non-destructive command receipts."""

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
    """Deterministic clock for REQ-HW-5319 duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5319.0

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
                    "date_utc=2026-07-06T13:00:00Z\n"
                    "date_local=2026-07-06T09:00:00-0400\n"
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
                stdout=(
                    "NVIDIA GeForce RTX 3090, 24576 MiB\n"
                    "NVIDIA GeForce RTX 3090, 24576 MiB\n"
                ),
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


def test_req_hw_5319_spec_declares_v485_required_fields() -> None:
    """REQ-HW-5319: OpenSpec anchors the v485 no-speedup receipt contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5319") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5319",
        "SCENARIO-HW-5319",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "experiment_id",
        "milestone",
        "status",
        "preconditions_checked",
        "kv260_ssh_reachable",
        "polarfire_status_reachable",
        "gatemate_physical_jtag_changed",
        "authenticated_workload_run",
        "no_speedup_claim",
        "public_hardware_references_used_as_context_only",
        "Dual RTX 3090",
        "Extropic/TSU",
        "Logical/Kona",
    ):
        assert marker in section


def test_scenario_hw_5319_records_current_blockers_and_context_only_boundaries() -> None:
    """SCENARIO-HW-5319: reachability receipts do not become speedup claims."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
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
    assert "no_speedup_claim" in _value(artifact, "honest_verdict")
    assert artifact["kv260_ssh_reachable"] is False
    assert artifact["polarfire_status_reachable"] is True
    assert artifact["gatemate_physical_jtag_changed"] is False
    assert artifact["authenticated_workload_run"] is False
    assert artifact["no_speedup_claim"] is True
    assert artifact["public_hardware_references_used_as_context_only"] is True

    preconditions = _value(artifact, "preconditions_checked")
    assert isinstance(preconditions, dict)
    assert preconditions["kv260_check_method"] == "ssh_batchmode_true_only"
    assert preconditions["operator_visible_hardware_assumptions"]["no_speedup_claim"] is True
    assert preconditions["gpu_runtime_context"]["rtx3090_count"] == 2
    assert preconditions["gpu_runtime_context"]["sampler_hardware_acceleration_claimed"] is False
    assert preconditions["public_reference_boundaries"]["extropic_tsu"]["context_only"] is True
    assert preconditions["public_reference_boundaries"]["logical_kona"]["context_only"] is True
    assert "do-not-record" not in json.dumps(artifact)
    assert "also-hidden" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    assert len(_value(artifact, "commands_run")) == len(runner.commands)
    mod.validate_artifact(artifact)


def test_reachable_statuses_and_changed_gatemate_still_do_not_authenticate_workload() -> None:
    """REQ-HW-5319: reachable board status and GPU context still preserve no-speedup."""

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
        run_date="20260706",
        commit="abc123",
        gatemate_setup_changed=True,
    )

    assert mod.GATEMATE_DETECT_COMMAND in runner.commands
    assert artifact["kv260_ssh_reachable"] is True
    assert artifact["polarfire_status_reachable"] is True
    assert artifact["gatemate_physical_jtag_changed"] is True
    assert artifact["authenticated_workload_run"] is False
    assert artifact["no_speedup_claim"] is True
    assert _value(artifact, "status") == "complete_status_receipts_no_authenticated_workload"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "gatemate=reachable_dirtyjtag_idcode_status_only" in _value(
        artifact, "honest_verdict"
    )
    mod.validate_artifact(artifact)


def test_req_hw_5319_status_value_covers_non_kv260_blockers() -> None:
    """REQ-HW-5319: status summaries preserve PolarFire and GateMate blockers."""

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


def test_validator_rejects_wrapped_id_boolean_drift_and_speedup_overclaim() -> None:
    """REQ-HW-5319: validator fails closed on schema, boolean, and speedup drift."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )

    artifact["experiment_id"] = mod.EXPERIMENT_ID
    with pytest.raises(AssertionError, match="experiment_id must be principle-wrapped"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )
    artifact["kv260_ssh_reachable"] = {"value": False}
    with pytest.raises(AssertionError, match="kv260_ssh_reachable must be a bare boolean"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )
    artifact["no_speedup_claim"] = False
    with pytest.raises(AssertionError, match="no_speedup_claim must stay true"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
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
    with pytest.raises(AssertionError, match="host storage marker present"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5319: run_experiment writes the requested v485 JSON artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert artifact["no_speedup_claim"] is True
    mod.validate_artifact(artifact)
