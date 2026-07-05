"""Tests for Exp 5266 hardware thermodynamic schedule boundary.

Spec refs: REQ-HW-5266, SCENARIO-HW-5266.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5266_hardware_thermodynamic_schedule_boundary_v481 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/fpga/spec.md"


class RecordingRunner:
    """SCENARIO-HW-5266 command runner with deterministic precondition receipts."""

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
    """Deterministic clock for stable duration and checksum assertions."""

    def __init__(self) -> None:
        self.value = 5266.0

    def __call__(self) -> float:
        self.value += 0.25
        return self.value


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def _probe(
    command: tuple[str, ...],
    *,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _runner(
    *,
    kv260_exit: int = 0,
    kv260_stderr: str = "",
    polarfire_exit: int = 0,
    polarfire_stderr: str = "",
    gpu_exit: int = 0,
    cpu_stdout: str = "Architecture: x86_64\nModel name: Unit Test CPU\n",
    env_stdout: str = "CARNOT_KV260_HOST=kria\nEXTROPIC_API_KEY=secret-token\n",
) -> RecordingRunner:
    probes: dict[tuple[str, ...], list[mod.CommandProbe]] = {
        mod.HOST_CPU_COMMAND: [
            _probe(
                mod.HOST_CPU_COMMAND,
                stdout=cpu_stdout,
                duration_s=0.02,
            )
        ],
        mod.HOST_GPU_COMMAND: [
            _probe(
                mod.HOST_GPU_COMMAND,
                exit_code=gpu_exit,
                stdout="GPU 0: Unit Test GPU\n" if gpu_exit == 0 else "",
                stderr="" if gpu_exit == 0 else "nvidia-smi not available\n",
                duration_s=0.03,
            )
        ],
        mod.HARDWARE_ENV_COMMAND: [
            _probe(mod.HARDWARE_ENV_COMMAND, stdout=env_stdout, duration_s=0.04)
        ],
        mod.KV260_SSH_COMMAND: [
            _probe(
                mod.KV260_SSH_COMMAND,
                exit_code=kv260_exit,
                stderr=kv260_stderr,
                duration_s=0.2,
            )
        ],
        mod.POLARFIRE_SSH_COMMAND: [
            _probe(
                mod.POLARFIRE_SSH_COMMAND,
                exit_code=polarfire_exit,
                stderr=polarfire_stderr,
                duration_s=0.3,
            )
        ],
    }
    return RecordingRunner(probes)


def test_req_hw_5266_spec_declares_required_artifact_contract() -> None:
    """REQ-HW-5266: OpenSpec anchors the v481 artifact and no-speedup contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5266") : spec.index("### SCENARIO-HW-4910")]

    for marker in (
        "REQ-HW-5266",
        "SCENARIO-HW-5266",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "preconditions_checked",
        "thermodynamic_boundary_updated",
        "speedup_claimed=false",
        "commands_run",
        "autocorrelation",
        "blocked_physical_jtag",
    ):
        assert marker in section
    for field in mod.REQUIRED_WRAPPED_FIELDS:
        assert f"`{field}`" in section


def test_scenario_hw_5266_success_path_records_preconditions_and_ssh_only_boards() -> None:
    """SCENARIO-HW-5266: reachable boards use preconditions then SSH reachability."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
    )

    assert runner.commands == [
        mod.HOST_CPU_COMMAND,
        mod.HOST_GPU_COMMAND,
        mod.HARDWARE_ENV_COMMAND,
        mod.KV260_SSH_COMMAND,
        mod.POLARFIRE_SSH_COMMAND,
    ]
    assert _value(artifact, "kv260_status") == "reachable"
    assert _value(artifact, "polarfire_status") == "reachable"
    assert _value(artifact, "gatemate_status") == "blocked_physical_jtag"
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "thermodynamic_boundary_updated") is True
    assert artifact["speedup_claimed"] is False
    assert artifact["speedup_claimed_principle"] == mod.SPEEDUP_CLAIMED_PRINCIPLE
    assert "no_speedup_claim" in _value(artifact, "honest_verdict")
    assert _value(artifact, "honest_verdict").startswith("complete:")
    preconditions = _value(artifact, "preconditions_checked")
    assert isinstance(preconditions, dict)
    assert preconditions["host_cpu"]["model"] == "Unit Test CPU"
    assert preconditions["host_gpu"]["available"] is True
    assert preconditions["hardware_environment"]["EXTROPIC_API_KEY"]["present"] is True
    assert "secret-token" not in json.dumps(preconditions)
    assert len(artifact["commands_run"]) == 5
    assert artifact["commands_run"][-2]["command"] == mod.command_to_string(
        mod.KV260_SSH_COMMAND
    )
    assert artifact["commands_run"][-2]["timeout_s"] == 10.0
    assert artifact["commands_run"][-1]["outcome"] == "reachable"
    assert "/dev/mmcblk" not in json.dumps(artifact).lower()
    mod.validate_artifact(artifact)


def test_precondition_fallbacks_record_gpu_error_and_custom_env_without_values() -> None:
    """REQ-HW-5266: preconditions are honest even when host details are absent."""

    artifact = mod.build_artifact(
        command_runner=_runner(
            gpu_exit=127,
            cpu_stdout="unexpected lscpu output\n",
            env_stdout="CARNOT_CUSTOM_PROBE=enabled\nPRIVATE_KEY=do-not-record\n",
        ),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
    )
    preconditions = _value(artifact, "preconditions_checked")
    assert isinstance(preconditions, dict)

    assert preconditions["host_cpu"]["model"] == "unknown"
    assert preconditions["host_cpu"]["architecture"] == "unknown"
    assert preconditions["host_gpu"]["available"] is False
    assert "nvidia-smi not available" in preconditions["host_gpu"]["error"]
    assert preconditions["hardware_environment"]["CARNOT_CUSTOM_PROBE"]["present"] is True
    assert "enabled" not in json.dumps(preconditions)
    assert "PRIVATE_KEY" not in json.dumps(preconditions)
    mod.validate_artifact(artifact)


def test_safe_probe_missing_blocks_boards_without_guessing_reachability() -> None:
    """REQ-HW-5266: missing safe probe scripts produce explicit board blockers."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
        safe_probe_scripts_present=False,
    )

    assert runner.commands == [
        mod.HOST_CPU_COMMAND,
        mod.HOST_GPU_COMMAND,
        mod.HARDWARE_ENV_COMMAND,
    ]
    assert _value(artifact, "kv260_status") == "blocked_safe_probe_missing"
    assert _value(artifact, "polarfire_status") == "blocked_safe_probe_missing"
    assert _value(artifact, "honest_verdict").startswith("blocked_safe_probe_missing:")
    assert "safe_probe_missing" in artifact["board_probe_notes"]["kv260"]
    assert artifact["speedup_claimed"] is False
    mod.validate_artifact(artifact)


def test_unreachable_ssh_records_command_timeout_and_error() -> None:
    """REQ-HW-5266: board blockers preserve exact SSH command receipts."""

    runner = _runner(
        kv260_exit=255,
        kv260_stderr="ssh: connect to host kria port 22: timeout\n",
        polarfire_exit=255,
        polarfire_stderr="ssh: connect to host polarfire port 22: No route to host\n",
    )
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
    )

    assert _value(artifact, "kv260_status") == "blocked_kv260_ssh_unreachable"
    assert _value(artifact, "polarfire_status") == "blocked_polarfire_ssh_unreachable"
    assert artifact["commands_run"][-2]["exit_code"] == 255
    assert artifact["commands_run"][-2]["outcome"] == "blocked_kv260_ssh_unreachable"
    assert "timeout" in artifact["commands_run"][-2]["stderr"]
    assert artifact["commands_run"][-1]["command"] == mod.command_to_string(
        mod.POLARFIRE_SSH_COMMAND
    )
    assert _value(artifact, "honest_verdict").startswith("blocked_board_reachability:")
    mod.validate_artifact(artifact)


def test_gatemate_setup_change_does_not_invent_software_workaround() -> None:
    """SCENARIO-HW-5266: physical setup changes are recorded without JTAG workarounds."""

    runner = _runner()
    artifact = mod.build_artifact(
        command_runner=runner,
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
        physical_setup_changed=True,
    )

    assert _value(artifact, "gatemate_status") == "not_checked_physical_setup_changed"
    assert artifact["gatemate_carry_forward"]["physical_setup_changed"] is True
    assert "software workaround" not in json.dumps(artifact["commands_run"]).lower()
    assert all("gatemate" not in item["command"].lower() for item in artifact["commands_run"])
    mod.validate_artifact(artifact)


def test_validate_artifact_rejects_speedup_claim_without_receipt() -> None:
    """REQ-HW-5266: schema validation rejects any accidental speedup claim."""

    artifact = mod.build_artifact(
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
        boundary_note_written=True,
    )
    artifact["speedup_claimed"] = True

    with pytest.raises(AssertionError, match="speedup_claimed must be false"):
        mod.validate_artifact(artifact)


def test_run_experiment_writes_boundary_note_and_result(tmp_path: Path) -> None:
    """SCENARIO-HW-5266: run_experiment writes the JSON and ops boundary note."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=_runner(),
        clock=StepClock(),
        run_date="20260705",
        commit="abc123",
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    note_path = tmp_path / str(_value(artifact, "thermodynamic_boundary_note_path"))
    note_text = note_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / mod.RESULT_RELATIVE_PATH
    assert "arXiv:2607.00170" in note_text
    assert "autocorrelation" in note_text
    assert "sampler-cost" in note_text
    assert "Extropic" in note_text
    assert "XTR-0" in note_text
    assert "future requirement" in note_text
    assert "no local SDK or device" in note_text
    assert "No speedup claim" in note_text
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)
