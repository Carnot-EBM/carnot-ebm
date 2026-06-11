"""Tests for Exp 4064 hardware continuity.

Spec refs: REQ-HW-4064, SCENARIO-HW-4064.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_3867_polarfire_soc_smoke_v4 as polarfire_helper
from carnot import experiment_4064_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4064 command runner with static and dynamic transcripts."""

    def __init__(
        self,
        probes: dict[tuple[str, ...], list[mod.CommandProbe]],
        *,
        allow_dynamic_steps: bool = False,
    ) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.allow_dynamic_steps = allow_dynamic_steps
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        del timeout_s
        command = tuple(command)
        self.commands.append(command)
        if command in self.probes and self.probes[command]:
            return self.probes[command].pop(0)
        if self.allow_dynamic_steps:
            dynamic = self._dynamic_step_probe(command)
            if dynamic is not None:
                return dynamic
        raise AssertionError(f"unexpected command: {command!r}")

    def _dynamic_step_probe(self, command: tuple[str, ...]) -> mod.CommandProbe | None:
        if command[0] == "openFPGALoader" and "-b" in command:
            return _probe(
                command,
                stdout=(
                    "Board default cable overridden with dirtyJtag\n"
                    "Load SRAM via JTAG: 100.00%\nDone\n"
                ),
                duration_s=1.25,
            )
        if command[0] == "scp":
            return _probe(command, duration_s=0.2)
        if command[0] == "ssh" and len(command) >= 3 and command[1] == "polarfire":
            shell = command[2]
            if shell == "python3 --version":
                return _probe(command, stdout="Python 3.12.12\n", duration_s=0.1)
            if shell.startswith("mkdir -p "):
                return _probe(command, duration_s=0.1)
            if "python3 runner.py workload.json" in shell:
                workload = polarfire_helper.build_ising_workload(mod.RANDOM_SEED)
                result = polarfire_helper.evaluate_ising_workload(workload)
                return _probe(
                    command,
                    stdout=json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
                    stderr="cycles=1\n",
                    duration_s=0.4,
                )
            if shell.startswith("rm -rf "):
                return _probe(command, duration_s=0.1)
            if "thermal_zone" in shell:
                return _probe(command, stdout="42000\n", duration_s=0.1)
        return None


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _seed_bitstream(repo_root: Path) -> Path:
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_3866_gatemate_ising_tile_flash_v2"
        / "gatemate_ising_n16.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"REQ-HW-4064 fake n16 bitstream\n")
    prior = repo_root / "results" / "experiment_3866_gatemate_ising_tile_flash_v2.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text(json.dumps({"bitstream_path": str(bitstream)}), encoding="utf-8")
    return bitstream


def _success_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.KV260_SSH_PRECONDITION: [
            _probe(mod.KV260_SSH_PRECONDITION, duration_s=0.31)
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout=(
                    "index 0:\n"
                    "\tidcode 0x20000001\n"
                    "\tmanufacturer colognechip\n"
                    "\tfamily GateMate Series\n"
                ),
                duration_s=0.12,
            ),
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="idcode 0x20000001 colognechip GateMate GM1Ax\n",
                duration_s=0.13,
            ),
        ],
        mod.POLARFIRE_SSH_PRECONDITION: [
            _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.44)
        ],
    }


def _assert_required_principles(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_4064_spec_entry_declares_continuity_contract() -> None:
    """REQ-HW-4064: OpenSpec anchors preconditions and forward steps."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4064" in spec
    assert "SCENARIO-HW-4064" in spec
    assert "experiment_4064_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "gatemate_step_taken" in spec
    assert "polarfire_step_taken" in spec
    assert "kv260_terminal_confirmed" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4064_success_writes_drive_to_terminal_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4064: reachable GateMate/PolarFire take concrete steps."""
    bitstream = _seed_bitstream(tmp_path)
    runner = RecordingRunner(_success_probes(), allow_dynamic_steps=True)

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands[:3] == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert runner.commands[3][0] == "openFPGALoader"
    assert runner.commands[4] == mod.GATEMATE_DETECT_COMMAND
    assert all(command[0:2] != ("ssh", "kria") for command in runner.commands[3:])
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert artifact["kv260_terminal_confirmed"] is True
    assert artifact["gatemate_step_taken"] == (
        "gatemate_existing_n16_bitstream_flash_detect_smoke_succeeded"
    )
    assert artifact["polarfire_step_taken"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_gatemate_"
        "gatemate_existing_n16_bitstream_flash_detect_smoke_succeeded_"
        "polarfire_polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert artifact["gatemate_step"]["bitstream_path"].endswith("gatemate_ising_n16.bit")
    assert artifact["gatemate_step"]["bitstream_sha256"] == hashlib.sha256(
        bitstream.read_bytes()
    ).hexdigest()
    assert artifact["polarfire_step"]["result_hash_match"] is True
    assert artifact["polarfire_step"]["board_result_sha256"] == artifact["polarfire_step"][
        "cpu_reference_sha256"
    ]
    assert [entry["resource"] for entry in artifact["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert all(isinstance(entry["available"], bool) for entry in artifact["preconditions_checked"])
    assert set(artifact["per_board_duration_s"]) == {"kv260", "gatemate", "polarfire"}
    assert all(float(value) > 0 for value in artifact["per_board_duration_s"].values())
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    _assert_required_principles(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4064_unreachable_boards_are_per_board_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4064: board misses are recorded and other boards continue."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    1,
                    stdout="no idcode",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.4)
            ],
        }
    )
    calls: list[str] = []

    def unexpected_gatemate(**_: Any) -> mod.StepOutcome:
        calls.append("gatemate")
        raise AssertionError("GateMate step should be skipped")

    def polar_step(**_: Any) -> mod.StepOutcome:
        calls.append("polarfire")
        return mod.StepOutcome(
            step_taken="polarfire_hash_verified_cpu_dispatch_succeeded",
            terminal_state="reachable_hash_verified_cpu_dispatch_recorded",
            success=True,
            duration_s=0.9,
            details={"result_hash_match": True},
        )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_step_runner=unexpected_gatemate,
        polarfire_step_runner=polar_step,
    )

    assert calls == ["polarfire"]
    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["kv260_terminal_confirmed"] is False
    assert artifact["per_board_reachability"] == {
        "kv260": False,
        "gatemate": False,
        "polarfire": True,
    }
    assert artifact["gatemate_step_taken"] == "blocked_gatemate_unreachable"
    assert artifact["polarfire_step_taken"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert artifact["per_board_terminal_state"] == {
        "kv260": "blocked_kv260_unreachable",
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "reachable_hash_verified_cpu_dispatch_recorded",
    }
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_scenario_hw_4064_all_boards_unreachable_stops_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4064: all misses produce the all-board blocked verdict."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stderr="no board")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["gatemate_step_taken"] == "blocked_gatemate_unreachable"
    assert artifact["polarfire_step_taken"] == "blocked_polarfire_unreachable"
    assert artifact["kv260_terminal_confirmed"] is False
    _assert_required_principles(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4064_reachable_gatemate_without_bitstream_records_blocker(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4064: a reachable GateMate still needs a concrete bitstream."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="idcode 0x20000001 colognechip GateMate\n",
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout")
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["gatemate_step_taken"] == "blocked_gatemate_no_existing_n16_bitstream"
    assert artifact["gatemate_step"]["candidate_paths_checked"]
    assert artifact["polarfire_step_taken"] == "blocked_polarfire_unreachable"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)
