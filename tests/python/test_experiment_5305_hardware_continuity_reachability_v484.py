"""Tests for Exp 5305 hardware continuity reachability receipts.

Spec refs: REQ-HW-5305, SCENARIO-HW-5305.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5305_hardware_continuity_reachability_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5305 runner with exact non-destructive probe receipts."""

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
    """Deterministic clock for REQ-HW-5305 duration and checksum tests."""

    def __init__(self) -> None:
        self.value = 5305.0

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
                    "date_utc=2026-07-06T12:00:00Z\n"
                    "date_local=2026-07-06T08:00:00-0400\n"
                ),
            )
        ],
        mod.HARDWARE_ENV_COMMAND: [
            _probe(
                mod.HARDWARE_ENV_COMMAND,
                stdout="CARNOT_MODE=live\nEXTROPIC_API_KEY=do-not-record\nPRIVATE_TOKEN=hidden\n",
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
    }


def _runner(
    *,
    kv260_exit: int = 255,
    kv260_stdout: str = "",
    kv260_stderr: str = "ssh: Could not resolve hostname kv260.local: Name or service not known\n",
    polarfire_exit: int = 0,
    polarfire_stdout: str = "hostname=mpfs-disco-kit\nuname=Linux mpfs-disco-kit riscv64\n",
    polarfire_stderr: str = "",
    gatemate_setup_changed: bool = False,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6.00MHz -> real 6.00MHz\nidcode 0x20000001\n",
) -> RecordingRunner:
    probes = _base_probes()
    probes[mod.KV260_REACHABILITY_COMMAND] = [
        _probe(
            mod.KV260_REACHABILITY_COMMAND,
            exit_code=kv260_exit,
            stdout=kv260_stdout,
            stderr=kv260_stderr,
        )
    ]
    probes[mod.POLARFIRE_REACHABILITY_COMMAND] = [
        _probe(
            mod.POLARFIRE_REACHABILITY_COMMAND,
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


def test_req_hw_5305_spec_declares_v484_required_fields() -> None:
    """REQ-HW-5305: OpenSpec anchors the v484 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5305") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5305",
        "SCENARIO-HW-5305",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "kv260_status",
        "polarfire_status",
        "gatemate_status",
        "hardware_speedup_claimed",
        "blocked_reason",
        "commands_run",
        "no-speedup",
    ):
        assert marker in section


def test_scenario_hw_5305_records_step0_and_status_only_blockers() -> None:
    """SCENARIO-HW-5305: Step 0 plus board status receipts stay non-accelerating."""

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
        mod.KV260_REACHABILITY_COMMAND,
        mod.POLARFIRE_REACHABILITY_COMMAND,
    ]
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "hardware_evidence_level") == mod.HARDWARE_EVIDENCE_LEVEL
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "honest_verdict").startswith("blocked_board_reachability:")
    assert "no_speedup_claim" in _value(artifact, "honest_verdict")

    kv260 = _value(artifact, "kv260_status")
    polarfire = _value(artifact, "polarfire_status")
    gatemate = _value(artifact, "gatemate_status")
    blockers = _value(artifact, "blocked_reason")
    preconditions = _value(artifact, "preconditions_checked")
    assert isinstance(kv260, dict)
    assert isinstance(polarfire, dict)
    assert isinstance(gatemate, dict)
    assert isinstance(blockers, dict)
    assert isinstance(preconditions, dict)
    assert kv260["status"] == "blocked_kv260_ssh_unreachable"
    assert polarfire["status"] == "reachable_ssh_status_only"
    assert gatemate["status"] == "blocked_gatemate_physical_jtag_setup_unchanged"
    assert "kv260.local" in blockers["KV260"]["stderr"]
    assert blockers["PolarFire"] is None
    assert blockers["GateMate"]["reason"] == "operator_setup_unchanged_physical_jtag_block_carried_forward"
    assert blockers["GateMate"]["prior_evidence"] == str(mod.PRIOR_RESULT_RELATIVE_PATH)
    assert preconditions["host_date"]["host"] == "carnot-host"
    assert preconditions["ssh_targets"] == {"KV260": "kria", "PolarFire": "polarfire"}
    assert preconditions["operator_visible_hardware_assumptions"]["kv260_checked_by_ssh_only"] is True
    assert preconditions["operator_visible_hardware_assumptions"]["no_speedup_claim"] is True
    assert "do-not-record" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    assert len(artifact["commands_run"]) == len(runner.commands)
    mod.validate_artifact(artifact)


def test_reachable_boards_and_changed_gatemate_setup_are_status_only() -> None:
    """REQ-HW-5305: reachable status probes do not become workload success."""

    runner = _runner(
        kv260_exit=0,
        kv260_stdout="hostname=kria\nuname=Linux kria 6.1.0 xilinx\nxmutil=xmutil 2024.2\nuio=/dev/uio0\n",
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
    kv260 = _value(artifact, "kv260_status")
    polarfire = _value(artifact, "polarfire_status")
    gatemate = _value(artifact, "gatemate_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(kv260, dict)
    assert isinstance(polarfire, dict)
    assert isinstance(gatemate, dict)
    assert kv260["status"] == "reachable_ssh_status_only"
    assert "Linux kria" in kv260["remote_identifier"]
    assert polarfire["status"] == "reachable_ssh_status_only"
    assert "riscv64" in polarfire["remote_identifier"]
    assert gatemate["status"] == "reachable_dirtyjtag_idcode_status_only"
    assert gatemate["speedup_claimed"] is False
    assert blockers == {"KV260": None, "PolarFire": None, "GateMate": None}
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "hardware_speedup_claimed") is False
    mod.validate_artifact(artifact)


def test_gatemate_changed_setup_probe_failure_is_honest_status_blocker() -> None:
    """SCENARIO-HW-5305: failed GateMate status probes remain blocked evidence."""

    artifact = mod.build_artifact(
        command_runner=_runner(
            kv260_exit=0,
            kv260_stdout="hostname=kria\nuname=Linux kria 6.1.0 xilinx\n",
            kv260_stderr="",
            polarfire_exit=0,
            polarfire_stdout="hostname=polarfire\nuname=Linux polarfire 6.18.17 riscv64\n",
            polarfire_stderr="",
            gatemate_setup_changed=True,
            gatemate_detect_exit=1,
            gatemate_detect_stdout="no idcode\n",
        ),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
        gatemate_setup_changed=True,
    )

    gatemate = _value(artifact, "gatemate_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(gatemate, dict)
    assert isinstance(blockers, dict)
    assert gatemate["status"] == "blocked_gatemate_dirtyjtag_status_probe_failed"
    assert blockers["GateMate"]["reason"] == "blocked_gatemate_dirtyjtag_status_probe_failed"
    assert _value(artifact, "honest_verdict").startswith("blocked_board_status:")
    mod.validate_artifact(artifact)


def test_validator_rejects_wrong_claims_and_retired_storage_markers() -> None:
    """REQ-HW-5305: validator fails closed on speedup, substrate, and SD-card drift."""

    summary = mod.parse_host_date(_probe(mod.HOST_DATE_COMMAND, exit_code=127, stderr="date failed\n"))
    assert summary["host"] == "not_recorded"
    assert summary["exit_code"] == 127

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )

    artifact["hardware_speedup_claimed"]["value"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claimed must be false"):
        mod.validate_artifact(artifact)

    artifact["hardware_speedup_claimed"]["value"] = False
    artifact["inference_substrate"]["value"] = "hardware_smoke"
    with pytest.raises(AssertionError, match="inference_substrate mismatch"):
        mod.validate_artifact(artifact)

    artifact["inference_substrate"]["value"] = mod.INFERENCE_SUBSTRATE
    artifact["commands_run"].append({"command": "ls /dev/mmcblk0", "outcome": "bad"})
    with pytest.raises(AssertionError, match="host storage marker present"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5305: run_experiment writes the requested JSON artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260706",
        commit="abc123",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == list(mod.SPEC_REFS)
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert _value(artifact, "hardware_speedup_claimed") is False
    mod.validate_artifact(artifact)
