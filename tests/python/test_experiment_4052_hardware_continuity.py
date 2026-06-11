"""Tests for Exp 4052 hardware continuity.

Spec refs: REQ-HW-4052, SCENARIO-HW-4052.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4052_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4052 command runner with queued board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []
        self.stdin_by_command: dict[tuple[str, ...], str | None] = {}

    def __call__(
        self,
        command: tuple[str, ...],
        stdin: str | None = None,
        timeout_s: float = 60.0,
    ) -> mod.CommandProbe:
        del timeout_s
        self.commands.append(command)
        self.stdin_by_command[command] = stdin
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _ticks(*values: float) -> mod.Clock:
    iterator = iter(values)
    return lambda: next(iterator)


def _board_stdout(sample_count: int = mod.BOARD_SAMPLE_COUNT) -> str:
    samples = [0.12 + 0.01 * (idx % 4) for idx in range(sample_count)]
    payload = {
        "schema": "carnot.kv260.uio_register_latency_transcript.v1",
        "sample_count": sample_count,
        "per_sample_wall_ms": samples,
        "per_batch_wall_ms": round(sum(samples) + 0.33, 6),
        "fixed_compute_budget": {
            "spin_count": mod.BOARD_SPIN_COUNT,
            "sample_count": sample_count,
            "max_degree": mod.BOARD_MAX_DEGREE,
            "beta_final_q88": mod.BOARD_BETA_FINAL_Q88,
            "trigger_mode": "uio_register_read_once_per_sample",
        },
        "selected_uio": "/dev/uio4",
        "selected_uio_addr_hex": "0x00000000a0000000",
        "read_offset_hex": "0x0",
        "final_register_value_hex": "0x00000004",
    }
    return "BOARD_HARNESS_START exp4052\n" + json.dumps(payload, sort_keys=True) + "\n"


def _success_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
        mod.KV260_LISTAPPS_COMMAND: [
            _probe(
                mod.KV260_LISTAPPS_COMMAND,
                stdout="carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok -1\n",
            )
        ],
        mod.KV260_LATENCY_COMMAND: [
            _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout(), duration_s=0.2)
        ],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_DETECT_COMMAND,
                stdout="IDCode : 0x20000001 colognechip GateMate\n",
            )
        ],
        mod.POLARFIRE_SSH_PRECONDITION: [_probe(mod.POLARFIRE_SSH_PRECONDITION)],
        mod.POLARFIRE_CONTINUITY_COMMAND: [
            _probe(mod.POLARFIRE_CONTINUITY_COMMAND, stdout="Linux polarfire\nup 1 day\n")
        ],
    }


def _assert_required_principles(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_4052_spec_entry_declares_terminal_latency_contract() -> None:
    """REQ-HW-4052: OpenSpec anchors the terminal latency-transcript contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4052" in spec
    assert "SCENARIO-HW-4052" in spec
    assert "experiment_4052_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "id_ok" in spec
    assert "kv260_latency_step_taken" in spec
    assert "kv260_latency_transcript_landed_4052" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4052_success_stamps_terminal_latency_transcript(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4052: overlay plus UIO transcript lands terminal KV260 state."""
    runner = RecordingRunner(_success_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(10.0, 10.1, 11.1, 11.4, 11.9, 12.2, 13.0, 14.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LATENCY_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.POLARFIRE_CONTINUITY_COMMAND,
    ]
    assert runner.stdin_by_command[mod.KV260_LATENCY_COMMAND] == mod.BOARD_HARNESS_SOURCE
    assert "BOARD_HARNESS_START exp4052" in mod.BOARD_HARNESS_SOURCE
    assert "uio_register_read_once_per_sample" in mod.BOARD_HARNESS_SOURCE
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["kv260_overlay_loaded"] is True
    assert artifact["kv260_loaded_overlay_name"] == "carnot_ising_v2_n64"
    assert artifact["kv260_latency_step_taken"] is True
    assert len(artifact["kv260_latency_samples_ms"]) == mod.BOARD_SAMPLE_COUNT
    assert artifact["kv260_latency_median_ms"] == pytest.approx(0.135)
    assert artifact["kv260_state"] == "reachable_overlay_loaded_latency_transcript_recorded"
    assert artifact["per_board_terminal_state"]["kv260"] == artifact["kv260_state"]
    assert artifact["per_board_next_step"]["kv260"] == (
        "kv260_terminal_state_overlay_loaded_latency_transcript_landed"
    )
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_latency_transcript_landed_4052"
    )
    assert artifact["per_board_duration_s"] == {
        "kv260": 1.0,
        "gatemate": 0.5,
        "polarfire": 0.8,
    }
    assert len(set(artifact["per_board_duration_s"].values())) == 3
    assert artifact["duration_s"] == 4.0
    assert [entry["resource"] for entry in artifact["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    _assert_required_principles(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4052_latency_failure_is_not_terminal_claim(tmp_path: Path) -> None:
    """SCENARIO-HW-4052: a failed board transcript remains an honest blocker."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v4 active\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, 1, stdout="BOARD_HARNESS_START exp4052\n")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="Jtag frequency only\n")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(2.0, 2.1, 3.0, 3.2, 3.7, 3.9, 4.6, 5.0),
    )

    assert artifact["kv260_overlay_loaded"] is True
    assert artifact["kv260_latency_step_taken"] is False
    assert artifact["kv260_state"] == "reachable_overlay_loaded_latency_transcript_blocked"
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_latency_transcript_blocked_4052"
    )
    assert "latency_transcript_landed_4052" not in artifact["honest_verdict"]
    assert artifact["per_board_next_step"]["gatemate"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_next_step"]["polarfire"] == "blocked_polarfire_unreachable"
    mod.validate_artifact(artifact)


def test_scenario_hw_4052_unreachable_boards_remain_per_board_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4052: board misses are recorded and other checks continue."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="ssh timeout\n")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stdout="no idcode\n")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="ssh timeout\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(5.0, 5.1, 5.5, 5.8, 6.4, 6.7, 7.6, 8.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["kv260_overlay_loaded"] is False
    assert artifact["kv260_latency_step_taken"] is False
    assert artifact["per_board_reachability"] == {
        "kv260": False,
        "gatemate": False,
        "polarfire": False,
    }
    assert artifact["per_board_terminal_state"] == {
        "kv260": "blocked_kv260_unreachable",
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "blocked_polarfire_unreachable",
    }
    assert artifact["per_board_next_step"] == {
        "kv260": "blocked_kv260_unreachable",
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "blocked_polarfire_unreachable",
    }
    _assert_required_principles(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4052_reachable_kv260_without_overlay_skips_latency(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4052: no KV260 latency claim is made without overlay proof."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="No active app\n"),
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="No active app\n"),
            ],
            mod.KV260_LOADAPP_COMMAND: [
                _probe(mod.KV260_LOADAPP_COMMAND, 1, stderr="load failed\n")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, 1, stdout="no board\n")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(8.0, 8.1, 8.9, 9.2, 9.7, 9.9, 10.5, 11.0),
    )

    assert mod.KV260_LATENCY_COMMAND not in runner.commands
    assert artifact["kv260_reachable"] is True
    assert artifact["kv260_overlay_loaded"] is False
    assert artifact["kv260_latency_step_taken"] is False
    assert artifact["kv260_state"] == "reachable_overlay_absent_latency_skipped"
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_overlay_absent_latency_skipped_4052"
    )
    mod.validate_artifact(artifact)


def test_scenario_hw_4052_kv260_miss_does_not_hide_gatemate_reachability(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4052: a KV260 miss is per-board when GateMate is reachable."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="ssh timeout\n")
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate\n",
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(12.0, 12.1, 12.5, 12.7, 13.4, 13.6, 14.2, 15.0),
    )

    assert artifact["per_board_reachability"] == {
        "kv260": False,
        "gatemate": True,
        "polarfire": False,
    }
    assert artifact["per_board_next_step"]["kv260"] == "blocked_kv260_unreachable"
    assert artifact["per_board_next_step"]["gatemate"] != "blocked_gatemate_unreachable"
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_unreachable_4052"
    )
    mod.validate_artifact(artifact)


def test_req_hw_4052_run_experiment_writes_requested_json(tmp_path: Path) -> None:
    """REQ-HW-4052: run_experiment writes the requested deliverable JSON."""
    runner = RecordingRunner(_success_probes())

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(20.0, 20.1, 21.1, 21.4, 21.9, 22.2, 23.0, 24.0),
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == mod.EXPERIMENT_ID
    assert payload["spec_refs"] == mod.SPEC_REFS
    assert payload["kv260_overlay_loaded"] is True
    assert payload["kv260_latency_step_taken"] is True
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)
