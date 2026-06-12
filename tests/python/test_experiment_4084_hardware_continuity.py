"""Tests for Exp 4084 hardware continuity.

Spec refs: REQ-HW-4084, SCENARIO-HW-4084.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4084_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4084 command runner with queued board-access transcripts."""

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


def _success_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.KV260_SSH_PRECONDITION: [
            _probe(mod.KV260_SSH_PRECONDITION, duration_s=0.31)
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                duration_s=0.12,
            )
        ],
        mod.POLARFIRE_SSH_PRECONDITION: [
            _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.44)
        ],
    }


def _assert_required_principles(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
    assert "records which board accesses were verified before the smoke" in (
        payload["field_principles"]["preconditions_checked"]
    )


def test_req_hw_4084_spec_entry_declares_continuity_contract() -> None:
    """REQ-HW-4084: OpenSpec anchors required fields and board preconditions."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4084" in spec
    assert "SCENARIO-HW-4084" in spec
    assert "experiment_4084_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "post-2026-06-11" in spec
    assert "re-plug continuity check" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "per_board_reachability" in spec
    assert "per_board_next_step" in spec
    assert "inference_substrate" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4084_success_records_required_artifact_fields(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4084: reachable boards record reachability and next steps."""
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "experiment_4074_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: hardware_continuity_prior",
                "per_board_reachability": {
                    "kv260": True,
                    "gatemate": False,
                    "polarfire": True,
                },
            }
        ),
        encoding="utf-8",
    )
    runner = RecordingRunner(_success_probes())

    def gate_step(**_: Any) -> mod.StepOutcome:
        return mod.StepOutcome(
            step_taken="gatemate_dirtyjtag_replug_detect_confirmed_next_flash_n16",
            terminal_state="reachable_dirtyjtag_replug_detect_confirmed",
            success=True,
            duration_s=0.25,
            details={
                "step": "dirtyjtag_detect_replug_confirmed",
                "next_concrete_step": "flash_existing_n16_bitstream_plus_dirtyjtag_detect_smoke",
            },
        )

    def polar_step(**_: Any) -> mod.StepOutcome:
        return mod.StepOutcome(
            step_taken="polarfire_hash_verified_cpu_dispatch_succeeded",
            terminal_state="reachable_hash_verified_cpu_dispatch_recorded",
            success=True,
            duration_s=0.5,
            details={"step": "hash_verified_cpu_dispatch_smoke", "result_hash_match": True},
        )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        gatemate_step_runner=gate_step,
        polarfire_step_runner=polar_step,
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
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
    assert saved["per_board_next_step"] == {
        "kv260": "kv260_terminal_opportunistic_confirm_only",
        "gatemate": "gatemate_dirtyjtag_replug_detect_confirmed_next_flash_n16",
        "polarfire": "polarfire_hash_verified_cpu_dispatch_succeeded",
    }
    assert [entry["resource"] for entry in saved["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert all(isinstance(entry["available"], bool) for entry in saved["preconditions_checked"])
    assert all(float(value) > 0 for value in saved["per_board_duration_s"].values())
    assert saved["source_context"] == {
        "previous_experiment": 4074,
        "previous_artifact_read": True,
        "previous_honest_verdict": "complete: hardware_continuity_prior",
        "previous_per_board_reachability": {
            "kv260": True,
            "gatemate": False,
            "polarfire": True,
        },
    }
    assert saved["gatemate_replug_context"] == {
        "date": "2026-06-11",
        "reachability_command": "openFPGALoader -c dirtyJtag --detect",
    }
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    _assert_required_principles(saved)
    mod.validate_artifact(saved)


def test_scenario_hw_4084_unreachable_boards_stop_after_preconditions(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4084: all board misses produce a terminal blocked verdict."""
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

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["per_board_next_step"] == {
        "kv260": "blocked_kv260_unreachable",
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "blocked_polarfire_unreachable",
    }
    assert artifact["source_context"] == {
        "previous_experiment": 4074,
        "previous_artifact_read": False,
        "previous_honest_verdict": None,
        "previous_per_board_reachability": None,
    }
    assert artifact["gatemate_replug_context"]["date"] == "2026-06-11"
    _assert_required_principles(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_4084_validation_rejects_missing_required_fields(tmp_path: Path) -> None:
    """REQ-HW-4084: required artifact fields stay mandatory and bare."""
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

    for field in ("honest_verdict", "per_board_reachability", "preconditions_checked"):
        bad = dict(artifact)
        bad.pop(field)
        with pytest.raises(ValueError, match="required fields|missing"):
            mod.validate_artifact(bad)

    with pytest.raises(ValueError, match="hardware_smoke"):
        mod.validate_artifact(artifact | {"inference_substrate": "model_inference"})
