"""Tests for Exp 4240 hardware continuity.

Spec refs: REQ-HW-4240, SCENARIO-HW-4240.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4240_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4240 command runner with queued board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        del timeout_s
        command = tuple(command)
        self.commands.append(command)
        if command in self.probes and self.probes[command]:
            return self.probes[command].pop(0)
        raise AssertionError(f"unexpected command: {command!r}")


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _make_bitstream(repo_root: Path) -> Path:
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_3866_gatemate_ising_tile_flash_v2"
        / "gatemate_ising_n16.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"REQ-HW-4240 fake n16 bitstream\n")
    return bitstream


def _write_previous_4228(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4228_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: hardware_continuity_4228_gatemate_blocked_"
                    "polarfire_hash_verified_kv260_terminal"
                ),
                "per_board_reachability": {
                    "kv260": True,
                    "gatemate": False,
                    "polarfire": True,
                },
                "gatemate_step_taken": "blocked_gatemate_unreachable",
                "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
                "gatemate_step": {
                    "next_concrete_step": (
                        "Recover GM1Ax IDCODE visibility before retrying the n=16 flash."
                    ),
                },
                "per_board_status": {
                    "gatemate": {"status": "blocked_gatemate_unreachable"},
                    "polarfire": {"status": "polarfire_hash_verified_cpu_dispatch_succeeded"},
                    "kv260": {"status": "kv260_terminal_confirmed_ssh_only"},
                },
            }
        ),
        encoding="utf-8",
    )


def _polar_step(**_: Any) -> mod.StepOutcome:
    return mod.StepOutcome(
        step_taken="polarfire_hash_verified_cpu_dispatch_succeeded",
        terminal_state="reachable_hash_verified_cpu_dispatch_recorded",
        success=True,
        duration_s=0.67,
        details={
            "step": "hash_verified_cpu_dispatch_smoke",
            "result_hash_match": True,
            "next_concrete_step": "Run end-to-end Carnot dispatch on PolarFire with hash-match.",
        },
    )


def test_req_hw_4240_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4240: OpenSpec anchors fields, principles, and preconditions."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4240" in spec
    assert "SCENARIO-HW-4240" in spec
    assert "experiment_4240_hardware_continuity.json" in spec
    assert "experiment_4228_hardware_continuity.json" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["per_board_status"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "0x20000001" in spec
    assert "GM1Ax" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4240_records_timers_preconditions_and_next_steps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4240: each board gets reachability, timer, status, and step."""
    _write_previous_4228(tmp_path)
    bitstream = _make_bitstream(tmp_path)
    flash_command = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        mod.GATEMATE_FLASH_BOARD,
        str(bitstream),
    )
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.21)],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.34,
                ),
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.08,
                ),
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.55)
            ],
            flash_command: [
                _probe(
                    flash_command,
                    1,
                    stderr="Board default cable overridden with dirtyJtag\nError: no device found\n",
                    duration_s=0.13,
                )
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_step_runner=_polar_step,
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
        flash_command,
        mod.GATEMATE_DETECT_COMMAND,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["source_context"]["previous_experiment"] == 4228
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["source_context"]["most_recent_hardware_continuity_artifact"].endswith(
        "experiment_4228_hardware_continuity.json"
    )
    timer_ids = [saved["per_board_status"][board]["timer_id"] for board in mod.BOARD_NAMES]
    assert len(set(timer_ids)) == 3
    assert all(timer_id.endswith("_precondition_plus_forward_step_wall_clock") for timer_id in timer_ids)
    assert saved["field_principles"]["honest_verdict"] == mod.FIELD_PRINCIPLES["honest_verdict"]
    assert saved["field_principles"]["per_board_status"] == mod.FIELD_PRINCIPLES["per_board_status"]
    assert saved["field_principles"]["preconditions_checked"] == (
        mod.FIELD_PRINCIPLES["preconditions_checked"]
    )
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_terminal_confirmed_ssh_only"
    assert saved["per_board_status"]["gatemate"]["status"] == (
        "gatemate_existing_n16_bitstream_flash_blocked_returncode_1"
    )
    assert "no device found" in saved["per_board_status"]["gatemate"]["next_concrete_step"]
    assert saved["per_board_status"]["polarfire"]["hash_match"] is True
    assert "hash-match" in saved["per_board_status"]["polarfire"]["next_concrete_step"]
    assert "BatchMode=yes" in saved["per_board_status"]["polarfire"]["precondition_command"]
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_4240_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_scenario_hw_4240_unreachable_board_statuses_do_not_block_others(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4240: blocked per-board verdicts still allow other steps."""
    _write_previous_4228(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.4)
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_step_runner=_polar_step,
    )

    assert artifact["per_board_status"]["kv260"]["status"] == "blocked_kv260_unreachable"
    assert artifact["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_status"]["gatemate"]["reachable"] is False
    assert "GM1Ax" in artifact["per_board_status"]["gatemate"]["next_concrete_step"]
    assert artifact["per_board_status"]["polarfire"]["status"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert artifact["per_board_status"]["polarfire"]["reachable"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_hw_4240_required_runner_lives_under_results() -> None:
    """REQ-HW-4240: the required `python3 results/...` runner is present."""
    runner = Path("results/experiment_4240_hardware_continuity.py")

    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "experiment_4240_hardware_continuity" in text
    assert "run_experiment" in text


def test_scenario_hw_4240_run_experiment_writes_blocked_artifact_without_prior(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4240: the runner writes honest blocked rows without prior context."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stderr="no dirtyjtag", duration_s=0.3)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.4)
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["source_context"]["previous_artifact_read"] is False
    assert artifact["per_board_status"]["kv260"]["status"] == "blocked_kv260_unreachable"
    assert artifact["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_status"]["polarfire"]["status"] == "blocked_polarfire_unreachable"
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)


def test_req_hw_4240_validate_artifact_rejects_bad_schema() -> None:
    """REQ-HW-4240: validation rejects artifacts that are not Exp 4240."""
    with pytest.raises(ValueError, match="schema must identify Exp 4240"):
        mod.validate_artifact({"schema": "wrong"})
