"""Tests for Exp 4104 hardware continuity.

Spec refs: REQ-HW-4104, SCENARIO-HW-4104.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4104_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4104 command runner with queued board transcripts."""

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
    bitstream.write_bytes(b"REQ-HW-4104 fake n16 bitstream\n")
    return bitstream


def _write_previous_4096(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4096_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: hardware_continuity_gatemate_blocked_gatemate_unreachable"
                ),
                "per_board_reachability": {
                    "kv260": True,
                    "gatemate": False,
                    "polarfire": True,
                },
                "gatemate_step_taken": "blocked_gatemate_unreachable",
                "gatemate_step": {
                    "previous_377_flash_exit_code": 1,
                    "previous_377_flash_error": "Error: no device found",
                    "next_concrete_step": "Recover GM1Ax IDCODE visibility before flash retry.",
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
            "next_concrete_step": "Run the full Carnot dispatch path on PolarFire with the same hash-match guard.",
        },
    )


def test_req_hw_4104_spec_entry_declares_per_board_status_contract() -> None:
    """REQ-HW-4104: OpenSpec anchors status, principles, and allowed preconditions."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4104" in spec
    assert "SCENARIO-HW-4104" in spec
    assert "experiment_4104_hardware_continuity.json" in spec
    assert "per_board_status" in spec
    assert mod.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert mod.FIELD_PRINCIPLES["per_board_status"] in spec
    assert mod.FIELD_PRINCIPLES["preconditions_checked"] in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "0x20000001" in spec
    assert "GM1Ax" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4104_reachable_boards_record_status_and_steps(tmp_path: Path) -> None:
    """SCENARIO-HW-4104: each board gets reachability, status, timer, and next step."""
    _write_previous_4096(tmp_path)
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
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, duration_s=0.21)
            ],
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
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["field_principles"]["honest_verdict"] == mod.FIELD_PRINCIPLES["honest_verdict"]
    assert saved["field_principles"]["per_board_status"] == mod.FIELD_PRINCIPLES["per_board_status"]
    assert saved["field_principles"]["preconditions_checked"] == mod.FIELD_PRINCIPLES["preconditions_checked"]
    assert saved["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert set(saved["per_board_status"]) == {"kv260", "gatemate", "polarfire"}
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_terminal_confirmed_ssh_only"
    assert saved["per_board_status"]["kv260"]["next_concrete_step"] == (
        "kv260_terminal_state_confirmed_via_ssh"
    )
    assert saved["per_board_status"]["gatemate"]["status"] == (
        "gatemate_existing_n16_bitstream_flash_blocked_returncode_1"
    )
    assert "no device found" in saved["per_board_status"]["gatemate"]["next_concrete_step"]
    assert saved["per_board_status"]["polarfire"]["status"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert saved["per_board_status"]["polarfire"]["hash_match"] is True
    assert [entry["resource"] for entry in saved["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert "BatchMode=yes" in saved["per_board_status"]["polarfire"]["precondition_command"]
    assert len(set(saved["per_board_duration_s"].values())) == 3
    assert saved["source_context"]["previous_experiment"] == 4096
    assert saved["source_context"]["most_recent_gatemate_artifact"].endswith(
        "experiment_4096_hardware_continuity.json"
    )
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_scenario_hw_4104_unreachable_board_statuses_do_not_block_others(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4104: blocked per-board statuses still allow other board steps."""
    _write_previous_4096(tmp_path)
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


def test_req_hw_4104_gate_detect_without_gm1ax_is_not_reachable(tmp_path: Path) -> None:
    """REQ-HW-4104: GateMate reachability requires the GM1Ax IDCODE."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="Jtag frequency only\n", duration_s=0.3)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.4)
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["gatemate_reachable"] is False
    assert artifact["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_status"]["gatemate"]["precondition_available"] is False
    assert "GM1Ax" in artifact["per_board_status"]["gatemate"]["next_concrete_step"]
    mod.validate_artifact(artifact)


def test_req_hw_4104_validation_rejects_required_field_principles_and_sd_markers(
    tmp_path: Path,
) -> None:
    """REQ-HW-4104: required fields, exact principles, and SSH-only evidence are enforced."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stderr="no board", duration_s=0.3)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.4)
            ],
        }
    )
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        bad = dict(artifact)
        bad.pop(field)
        with pytest.raises(ValueError, match="missing required"):
            mod.validate_artifact(bad)

    bad_principle = json.loads(json.dumps(artifact))
    bad_principle["field_principles"]["per_board_status"] = "too vague"
    with pytest.raises(ValueError, match="principle"):
        mod.validate_artifact(bad_principle)

    bad_sd = json.loads(json.dumps(artifact))
    bad_sd["preconditions_checked"][0]["command"] = "ls /dev/mmcblk0"
    bad_sd["reproducibility_checksum"] = mod.payload_checksum(bad_sd)
    with pytest.raises(ValueError, match="SD-card"):
        mod.validate_artifact(bad_sd)

    bad_prefix = json.loads(json.dumps(artifact))
    bad_prefix["honest_verdict"] = "hardware continuity done"
    bad_prefix["reproducibility_checksum"] = mod.payload_checksum(bad_prefix)
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(bad_prefix)


def test_req_hw_4104_run_experiment_writes_deliverable_json(tmp_path: Path) -> None:
    """REQ-HW-4104: run_experiment writes the requested results artifact."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stderr="no board", duration_s=0.3)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.4)
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["per_board_status"]["kv260"]["status"] == "blocked_kv260_unreachable"
    assert saved["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    assert saved["per_board_status"]["polarfire"]["status"] == "blocked_polarfire_unreachable"
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_scenario_hw_4104_gatemate_step_names_success_and_detect_blocker(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4104: GateMate flash step names success and post-detect blockers."""
    bitstream = _make_bitstream(tmp_path)
    flash_command = (
        "openFPGALoader",
        "-c",
        "dirtyJtag",
        "-b",
        mod.GATEMATE_FLASH_BOARD,
        str(bitstream),
    )
    success_runner = RecordingRunner(
        {
            flash_command: [_probe(flash_command, stdout="write ok\n", duration_s=0.11)],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.12,
                )
            ],
        }
    )

    success = mod.run_gatemate_forward_step(
        repo_root=tmp_path,
        command_runner=success_runner,
    )

    assert success.step_taken == "gatemate_existing_n16_bitstream_flash_detect_smoke_succeeded"
    assert success.success is True
    assert "succeeded" in success.details["next_concrete_step"]

    blocked_runner = RecordingRunner(
        {
            flash_command: [_probe(flash_command, stdout="write ok\n", duration_s=0.11)],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.12,
                )
            ],
        }
    )

    blocked = mod.run_gatemate_forward_step(
        repo_root=tmp_path,
        command_runner=blocked_runner,
    )

    assert blocked.step_taken == "gatemate_existing_n16_bitstream_post_flash_detect_blocked"
    assert blocked.success is False
    assert "rc=0" in blocked.details["next_concrete_step"]


def test_scenario_hw_4104_gatemate_step_handles_missing_bitstream(tmp_path: Path) -> None:
    """SCENARIO-HW-4104: reachable GateMate needs an n=16 bitstream to flash."""

    def unexpected_runner(command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        raise AssertionError(f"unexpected command {command!r} at timeout {timeout_s}")

    outcome = mod.run_gatemate_forward_step(
        repo_root=tmp_path,
        command_runner=unexpected_runner,
    )

    assert outcome.step_taken == "blocked_gatemate_no_existing_n16_bitstream"
    assert outcome.success is False
    assert "Rebuild the GateMate n=16 bitstream" in outcome.details["next_concrete_step"]


def test_req_hw_4104_helper_branches_preserve_next_steps_and_errors() -> None:
    """REQ-HW-4104: helper branches preserve concrete next steps and flash errors."""
    assert mod._extract_flash_error_from_transcripts("not a list") is None
    assert (
        mod._extract_flash_error_from_transcripts(
            [
                "not a dict",
                {"stage": "post_flash_detect_smoke", "output_excerpt": "ignored"},
                {
                    "stage": "flash_existing_bitstream",
                    "output_excerpt": "info\nError: no device found\n",
                },
            ]
        )
        == "Error: no device found"
    )
    assert mod._extract_flash_error("fails to open device 0") == "fails to open device 0"
    assert mod._extract_flash_error("warning only") == "warning only"
    assert mod._extract_flash_error("") == ""
    assert (
        mod._next_concrete_step(
            "gatemate",
            {
                "gatemate_step": {},
                "gatemate_step_taken": "gatemate_manual_retry_named",
                "per_board_reachability": {"gatemate": True},
            },
        )
        == "gatemate_manual_retry_named"
    )
    assert (
        mod._next_concrete_step(
            "polarfire",
            {
                "polarfire_step": {},
                "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
                "per_board_reachability": {"polarfire": True},
            },
        )
        == "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
