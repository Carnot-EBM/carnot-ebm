"""Tests for Exp 4096 hardware continuity.

Spec refs: REQ-HW-4096, SCENARIO-HW-4096.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4096_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4096 command runner with queued hardware transcripts."""

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
    bitstream.write_bytes(b"REQ-HW-4096 fake n16 bitstream\n")
    return bitstream


def _write_previous_4084(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4084_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: hardware_continuity_gatemate_flash_blocked"
                ),
                "per_board_reachability": {
                    "kv260": True,
                    "gatemate": True,
                    "polarfire": True,
                },
                "gatemate_step_taken": (
                    "gatemate_existing_n16_bitstream_flash_blocked_returncode_1"
                ),
                "gatemate_step": {
                    "flash_exit_code": 1,
                    "command_transcripts": [
                        {
                            "stage": "flash_existing_bitstream",
                            "output_excerpt": (
                                "Board default cable overridden with dirtyJtag\n"
                                "Error: no device found\n"
                            ),
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _assert_required_principles(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
    assert "records which board accesses were verified before the smoke" in (
        payload["field_principles"]["preconditions_checked"]
    )


def test_req_hw_4096_spec_entry_declares_board_contract() -> None:
    """REQ-HW-4096: OpenSpec anchors IDCODE, flash-error, and artifact fields."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4096" in spec
    assert "SCENARIO-HW-4096" in spec
    assert "experiment_4096_hardware_continuity.json" in spec
    assert "results/experiment_4084_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "0x20000001" in spec
    assert "GM1Ax" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "flash_error" in spec
    assert "next_concrete_step" in spec
    assert "per_board_reachability" in spec
    assert "preconditions_checked" in spec
    assert "inference_substrate" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4096_flash_blocker_records_error_and_next_step(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4096: reachable boards record the GateMate flash blocker."""
    _write_previous_4084(tmp_path)
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
                    stdout=(
                        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
                        "IDCode : 0x20000001 colognechip GateMate GM1Ax\n"
                    ),
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

    def polar_step(**_: Any) -> mod.StepOutcome:
        return mod.StepOutcome(
            step_taken="polarfire_hash_verified_cpu_dispatch_succeeded",
            terminal_state="reachable_hash_verified_cpu_dispatch_recorded",
            success=True,
            duration_s=0.67,
            details={"step": "hash_verified_cpu_dispatch_smoke", "result_hash_match": True},
        )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_step_runner=polar_step,
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
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert saved["gatemate_step_taken"] == (
        "gatemate_existing_n16_bitstream_flash_blocked_returncode_1"
    )
    assert saved["gatemate_step"]["flash_error"] == "Error: no device found"
    assert "GM1Ax" in saved["gatemate_step"]["next_concrete_step"]
    assert "openFPGALoader" in saved["gatemate_step"]["next_concrete_step"]
    assert saved["gatemate_step"]["previous_377_flash_error"] == "Error: no device found"
    assert saved["source_context"] == {
        "previous_experiment": 4084,
        "previous_artifact_read": True,
        "previous_honest_verdict": "complete: hardware_continuity_gatemate_flash_blocked",
        "previous_per_board_reachability": {
            "kv260": True,
            "gatemate": True,
            "polarfire": True,
        },
        "previous_gatemate_step_taken": (
            "gatemate_existing_n16_bitstream_flash_blocked_returncode_1"
        ),
        "previous_gatemate_flash_exit_code": 1,
        "previous_gatemate_flash_error": "Error: no device found",
    }
    assert saved["per_board_next_step"] == {
        "kv260": "kv260_terminal_opportunistic_confirm_only",
        "gatemate": "gatemate_existing_n16_bitstream_flash_blocked_returncode_1",
        "polarfire": "polarfire_hash_verified_cpu_dispatch_succeeded",
    }
    assert [entry["resource"] for entry in saved["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert all(isinstance(entry["available"], bool) for entry in saved["preconditions_checked"])
    assert len(set(saved["per_board_duration_s"].values())) == 3
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    _assert_required_principles(saved)
    mod.validate_artifact(saved)


def test_scenario_hw_4096_gate_detect_without_gm1ax_is_not_reachable(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4096: GateMate reachability requires the requested GM1Ax IDCODE."""
    _write_previous_4084(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.4)
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["per_board_reachability"] == {
        "kv260": True,
        "gatemate": False,
        "polarfire": False,
    }
    assert artifact["gatemate_step_taken"] == "blocked_gatemate_unreachable"
    assert artifact["gatemate_step"]["blocker"] == "blocked_gatemate_unreachable"
    assert artifact["gatemate_step"]["previous_377_flash_error"] == "Error: no device found"
    assert "GM1Ax" in artifact["gatemate_step"]["next_concrete_step"]
    assert artifact["source_context"]["previous_artifact_read"] is True
    mod.validate_artifact(artifact)


def test_req_hw_4096_validation_rejects_required_field_and_substrate(
    tmp_path: Path,
) -> None:
    """REQ-HW-4096: required artifact fields stay mandatory and bare."""
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

    for field in ("honest_verdict", "per_board_reachability", "preconditions_checked"):
        bad = dict(artifact)
        bad.pop(field)
        with pytest.raises(ValueError, match="required fields|missing"):
            mod.validate_artifact(bad)

    with pytest.raises(ValueError, match="hardware_smoke"):
        mod.validate_artifact(artifact | {"inference_substrate": "model_inference"})


def test_req_hw_4096_run_experiment_writes_blocked_artifact(tmp_path: Path) -> None:
    """REQ-HW-4096: run_experiment writes the requested deliverable JSON."""
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
    assert saved["honest_verdict"] == "blocked_all_boards_unreachable"
    assert saved["source_context"]["previous_experiment"] == 4084
    assert "GM1Ax" in saved["gatemate_step"]["next_concrete_step"]
    mod.validate_artifact(saved)


def test_scenario_hw_4096_gatemate_step_handles_no_bitstream(tmp_path: Path) -> None:
    """SCENARIO-HW-4096: reachable GateMate still needs a concrete n=16 bitstream."""

    def unexpected_runner(command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        raise AssertionError(f"unexpected command {command!r} at timeout {timeout_s}")

    outcome = mod.run_gatemate_forward_step(
        repo_root=tmp_path,
        command_runner=unexpected_runner,
    )

    assert outcome.step_taken == "blocked_gatemate_no_existing_n16_bitstream"
    assert outcome.success is False
    assert outcome.details["next_concrete_step"].startswith("Rebuild the GateMate n=16")
    assert outcome.details["previous_377_flash_error"] is None


def test_scenario_hw_4096_gatemate_step_success_and_post_detect_blocker(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4096: GateMate step names success and post-flash detect blockers."""
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


def test_req_hw_4096_flash_error_helpers_cover_loader_shapes() -> None:
    """REQ-HW-4096: flash-error extraction handles prior and live transcript shapes."""
    assert mod._extract_flash_error_from_transcripts("not a list") is None
    assert (
        mod._extract_flash_error_from_transcripts(
            [
                "not a dict",
                {"stage": "post_flash_detect_smoke", "output_excerpt": "ignored"},
                {
                    "stage": "flash_existing_bitstream",
                    "output_excerpt": "fails to open device",
                },
            ]
        )
        == "fails to open device"
    )
    assert mod._extract_flash_error("first\nlast line") == "last line"
    assert mod._extract_flash_error("") == ""
