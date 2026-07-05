"""Tests for Exp 5279 hardware continuity reachability receipts.

Spec refs: REQ-HW-5279, SCENARIO-HW-5279.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5279_hardware_continuity_reachability_v482 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5279 command runner that preserves command ordering."""

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
    """Deterministic clock for stable durations and checksums."""

    def __init__(self) -> None:
        self.value = 5279.0

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
        mod.SSH_CONFIG_KV260_COMMAND: [
            _probe(
                mod.SSH_CONFIG_KV260_COMMAND,
                stdout="host kria\nhostname kv260.local\nuser ubuntu\nport 22\nidentityfile ~/.ssh/id_ed25519\n",
            )
        ],
        mod.SSH_CONFIG_POLARFIRE_COMMAND: [
            _probe(
                mod.SSH_CONFIG_POLARFIRE_COMMAND,
                stdout="host polarfire\nhostname mpfs-disco-kit.local\nuser root\nport 22\n",
            )
        ],
        mod.HOST_RESOLUTION_KV260_COMMAND: [
            _probe(
                mod.HOST_RESOLUTION_KV260_COMMAND,
                exit_code=2,
                stderr="Name or service not known\n",
            )
        ],
        mod.HOST_RESOLUTION_POLARFIRE_COMMAND: [
            _probe(
                mod.HOST_RESOLUTION_POLARFIRE_COMMAND,
                exit_code=2,
                stderr="Name or service not known\n",
            )
        ],
        mod.HARDWARE_ENV_COMMAND: [
            _probe(
                mod.HARDWARE_ENV_COMMAND,
                stdout="CARNOT_MODE=live\nEXTROPIC_API_KEY=do-not-record\nPRIVATE_TOKEN=hidden\n",
            )
        ],
        mod.TOOLCHAIN_PRESENCE_COMMAND: [
            _probe(
                mod.TOOLCHAIN_PRESENCE_COMMAND,
                stdout=(
                    "ssh=/usr/bin/ssh\n"
                    "scp=/usr/bin/scp\n"
                    "openFPGALoader=/opt/oss-cad-suite/bin/openFPGALoader\n"
                    "yosys=/opt/oss-cad-suite/bin/yosys\n"
                    "nextpnr-himbaechel=/opt/oss-cad-suite/bin/nextpnr-himbaechel\n"
                    "gmpack=/opt/oss-cad-suite/bin/gmpack\n"
                    "vivado=\n"
                    "lsusb=/usr/bin/lsusb\n"
                ),
            )
        ],
        mod.GATEMATE_USB_COMMAND: [
            _probe(mod.GATEMATE_USB_COMMAND, exit_code=1, stderr="not found\n")
        ],
        mod.POLARFIRE_USB_COMMAND: [
            _probe(mod.POLARFIRE_USB_COMMAND, exit_code=1, stderr="not found\n")
        ],
    }


def _runner(
    *,
    kv260_exit: int = 255,
    kv260_stderr: str = "ssh: Could not resolve hostname kv260.local: Name or service not known\n",
    polarfire_exit: int = 255,
    polarfire_stderr: str = "ssh: Could not resolve hostname mpfs-disco-kit.local: Name or service not known\n",
    polarfire_workload_exit: int = 0,
    polarfire_workload_stdout: str = "terminal_workload_exists=true\n",
    gatemate_setup_changed: bool = False,
    gatemate_detect_exit: int = 0,
    gatemate_detect_stdout: str = "Jtag frequency : requested 6.00MHz -> real 6.00MHz\nidcode 0x20000001\n",
) -> RecordingRunner:
    probes = _base_probes()
    probes[mod.KV260_SSH_COMMAND] = [
        _probe(mod.KV260_SSH_COMMAND, exit_code=kv260_exit, stderr=kv260_stderr)
    ]
    probes[mod.KV260_BOARD_SUMMARY_COMMAND] = [
        _probe(
            mod.KV260_BOARD_SUMMARY_COMMAND,
            stdout="kria\nLinux kria 6.1.0\n/dev/uio0\n",
        )
    ]
    probes[mod.POLARFIRE_SSH_COMMAND] = [
        _probe(
            mod.POLARFIRE_SSH_COMMAND,
            exit_code=polarfire_exit,
            stderr=polarfire_stderr,
        )
    ]
    probes[mod.POLARFIRE_TERMINAL_WORKLOAD_COMMAND] = [
        _probe(
            mod.POLARFIRE_TERMINAL_WORKLOAD_COMMAND,
            exit_code=polarfire_workload_exit,
            stdout=polarfire_workload_stdout,
            stderr="" if polarfire_workload_exit == 0 else "terminal workload probe failed\n",
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


def test_req_hw_5279_spec_declares_continuity_receipt_contract() -> None:
    """REQ-HW-5279: OpenSpec anchors the v482 receipt and no-speedup contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5279") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5279",
        "SCENARIO-HW-5279",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "hardware_evidence_level",
        "per_board_status",
        "host_sd_card_precondition_used",
        "hardware_speedup_claimed",
        "blocked_reason",
        "terminal workload",
        "no-speedup",
    ):
        assert marker in section


def test_scenario_hw_5279_unreachable_boards_record_exact_blockers() -> None:
    """SCENARIO-HW-5279: SSH failures become board-specific blocked reasons."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
    )

    assert runner.commands == [
        mod.SSH_CONFIG_KV260_COMMAND,
        mod.SSH_CONFIG_POLARFIRE_COMMAND,
        mod.HOST_RESOLUTION_KV260_COMMAND,
        mod.HOST_RESOLUTION_POLARFIRE_COMMAND,
        mod.HARDWARE_ENV_COMMAND,
        mod.TOOLCHAIN_PRESENCE_COMMAND,
        mod.GATEMATE_USB_COMMAND,
        mod.POLARFIRE_USB_COMMAND,
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "hardware_evidence_level") == "reachability_status_receipt_only"
    assert _value(artifact, "host_sd_card_precondition_used") is False
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert _value(artifact, "honest_verdict").startswith("blocked_board_reachability:")
    assert "no_speedup_claim" in _value(artifact, "honest_verdict")

    statuses = _value(artifact, "per_board_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(statuses, dict)
    assert isinstance(blockers, dict)
    assert statuses["KV260"]["status"] == "blocked_kv260_ssh_unreachable"
    assert statuses["PolarFire"]["status"] == "blocked_polarfire_ssh_unreachable"
    assert statuses["GateMate"]["status"] == "blocked_gatemate_physical_jtag_setup_unchanged"
    assert "kv260.local" in blockers["KV260"]["stderr"]
    assert "mpfs-disco-kit.local" in blockers["PolarFire"]["stderr"]
    assert blockers["GateMate"]["reason"] == "operator_setup_unchanged_physical_jtag_block_carried_forward"
    assert "do-not-record" not in json.dumps(artifact)
    assert "PRIVATE_TOKEN" not in json.dumps(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    assert len(artifact["commands_run"]) == len(runner.commands)
    mod.validate_artifact(artifact)


def test_reachable_ssh_boards_and_changed_gatemate_setup_are_status_only() -> None:
    """REQ-HW-5279: reachable probes stay continuity evidence, not acceleration."""

    runner = _runner(
        kv260_exit=0,
        kv260_stderr="",
        polarfire_exit=0,
        polarfire_stderr="",
        gatemate_setup_changed=True,
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        gatemate_setup_changed=True,
    )

    assert mod.KV260_BOARD_SUMMARY_COMMAND in runner.commands
    assert mod.POLARFIRE_TERMINAL_WORKLOAD_COMMAND in runner.commands
    assert mod.GATEMATE_DETECT_COMMAND in runner.commands

    statuses = _value(artifact, "per_board_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(statuses, dict)
    assert isinstance(blockers, dict)
    assert statuses["KV260"]["status"] == "reachable_ssh_board_level_checked"
    assert statuses["PolarFire"]["status"] == "reachable_terminal_workload_present"
    assert statuses["PolarFire"]["terminal_workload_exists"] is True
    assert statuses["GateMate"]["status"] == "reachable_dirtyjtag_idcode_status_only"
    assert blockers == {"KV260": None, "PolarFire": None, "GateMate": None}
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert _value(artifact, "hardware_speedup_claimed") is False
    assert statuses["GateMate"]["speedup_claimed"] is False
    mod.validate_artifact(artifact)


def test_polarfire_reachable_without_terminal_workload_records_blocker() -> None:
    """REQ-HW-5279: PolarFire reachability alone is not terminal workload evidence."""

    artifact = mod.build_artifact(
        command_runner=_runner(
            polarfire_exit=0,
            polarfire_stderr="",
            polarfire_workload_stdout="terminal_workload_exists=false\n",
        ),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
    )

    statuses = _value(artifact, "per_board_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(statuses, dict)
    assert isinstance(blockers, dict)
    assert statuses["PolarFire"]["status"] == "reachable_terminal_workload_missing"
    assert statuses["PolarFire"]["terminal_workload_exists"] is False
    assert blockers["PolarFire"]["reason"] == "blocked_polarfire_terminal_workload_missing"
    mod.validate_artifact(artifact)


def test_validate_artifact_rejects_host_sd_or_speedup_claims() -> None:
    """REQ-HW-5279: validator rejects wrong-mechanism and speedup fields."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
    )

    artifact["hardware_speedup_claimed"]["value"] = True
    with pytest.raises(AssertionError, match="hardware_speedup_claimed must be false"):
        mod.validate_artifact(artifact)

    artifact["hardware_speedup_claimed"]["value"] = False
    artifact["host_sd_card_precondition_used"]["value"] = True
    with pytest.raises(AssertionError, match="host_sd_card_precondition_used must be false"):
        mod.validate_artifact(artifact)

    artifact["host_sd_card_precondition_used"]["value"] = False
    artifact["commands_run"].append({"command": "ls /dev/mmcblk0", "outcome": "bad"})
    with pytest.raises(AssertionError, match="host storage marker present"):
        mod.validate_artifact(artifact)


def test_helper_fail_closed_branches_and_gatemate_only_blocker() -> None:
    """REQ-HW-5279: helper parsers and GateMate-only blockers fail closed."""

    env = mod.parse_hardware_environment(
        _probe(mod.HARDWARE_ENV_COMMAND, exit_code=127, stderr="env unavailable\n")
    )
    assert env["CARNOT_MODE"]["present"] is False

    ssh_config = mod.parse_ssh_config(
        _probe(mod.SSH_CONFIG_KV260_COMMAND, exit_code=255, stderr="bad ssh config\n")
    )
    assert ssh_config["available"] is False
    assert ssh_config["error"] == "bad ssh config"
    assert mod.sanitized_ssh_config_stdout({"selected": None}) == ""

    artifact = mod.build_artifact(
        command_runner=_runner(
            kv260_exit=0,
            kv260_stderr="",
            polarfire_exit=0,
            polarfire_stderr="",
            gatemate_setup_changed=True,
            gatemate_detect_exit=1,
            gatemate_detect_stdout="no idcode\n",
        ),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        gatemate_setup_changed=True,
    )

    statuses = _value(artifact, "per_board_status")
    blockers = _value(artifact, "blocked_reason")
    assert isinstance(statuses, dict)
    assert isinstance(blockers, dict)
    assert statuses["GateMate"]["status"] == "blocked_gatemate_dirtyjtag_status_probe_failed"
    assert blockers["GateMate"]["reason"] == "blocked_gatemate_dirtyjtag_status_probe_failed"
    assert _value(artifact, "honest_verdict").startswith("blocked_board_status:")
    mod.validate_artifact(artifact)


def test_run_experiment_writes_stable_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5279: run_experiment writes the requested JSON artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
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
