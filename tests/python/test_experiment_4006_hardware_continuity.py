"""Tests for Exp 4006 ARC hardware continuity.

Spec refs: REQ-HW-4006, SCENARIO-HW-4006.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4006_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4006 command runner with deterministic board transcripts."""

    def __init__(self, results: dict[tuple[str, ...], mod.CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float) -> mod.CommandResult:
        del timeout_s
        self.commands.append(command)
        if command not in self.results:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.results[command]


def _result(
    command: tuple[str, ...],
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandResult:
    return mod.CommandResult(command, returncode, stdout, stderr, duration_s)


def _ticks(*values: float) -> mod.Clock:
    iterator = iter(values)
    return lambda: next(iterator)


def _assert_required_fields_are_bare(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def _successful_results() -> dict[tuple[str, ...], mod.CommandResult]:
    return {
        mod.KV260_SSH_PRECONDITION: _result(mod.KV260_SSH_PRECONDITION),
        mod.KV260_LISTAPPS_COMMAND: _result(
            mod.KV260_LISTAPPS_COMMAND,
            stdout="carnot_ising_v4 active\n",
            duration_s=0.20,
        ),
        mod.KV260_UIO_COMMAND: _result(
            mod.KV260_UIO_COMMAND,
            stdout="/dev/uio0\n/dev/uio1\n",
            duration_s=0.10,
        ),
        mod.GATEMATE_DETECT_COMMAND: _result(
            mod.GATEMATE_DETECT_COMMAND,
            stdout="Jtag frequency : requested 6.00MHz\nIDCode : 0x20000001 colognechip GateMate\n",
            duration_s=0.30,
        ),
        mod.POLARFIRE_SSH_PRECONDITION: _result(mod.POLARFIRE_SSH_PRECONDITION),
        mod.POLARFIRE_CONTINUITY_COMMAND: _result(
            mod.POLARFIRE_CONTINUITY_COMMAND,
            stdout="Linux polarfire 6.1\nup 12 minutes\n",
            duration_s=0.40,
        ),
    }


def test_req_hw_4006_spec_entry_declares_arc_continuity_contract() -> None:
    """REQ-HW-4006: OpenSpec anchors the Exp 4006 continuity artifact."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4006" in spec
    assert "SCENARIO-HW-4006" in spec
    assert "experiment_4006_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "per_board_next_step" in spec
    assert "per_board_duration_s" in spec
    assert "blocked_<board>_unreachable" in spec
    assert "/dev/mmcblk" in spec


def test_req_hw_4006_success_records_evidence_next_steps_and_distinct_timers(
    tmp_path: Path,
) -> None:
    """REQ-HW-4006: reachable boards keep bare reachability and board timers."""
    runner = RecordingRunner(_successful_results())
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(10.0, 10.1, 10.7, 11.0, 11.8, 12.0, 13.1, 14.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.POLARFIRE_CONTINUITY_COMMAND,
    ]
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["kv260_reachable"] is True
    assert artifact["gatemate_reachable"] is True
    assert artifact["polarfire_reachable"] is True
    assert artifact["per_board_duration_s"] == {
        "kv260": 0.6,
        "gatemate": 0.8,
        "polarfire": 1.1,
    }
    assert len(set(artifact["per_board_duration_s"].values())) == 3
    assert artifact["duration_s"] == 4.0
    assert artifact["per_board_next_step"] == {
        "kv260": "kv260_forward_step_run_terminal_overlay_latency_smoke",
        "gatemate": "gatemate_forward_step_run_minimal_ising_tile_smoke",
        "polarfire": "polarfire_forward_step_run_hash_verified_soft_cpu_dispatch",
    }
    assert artifact["kv260_loaded_overlay"] == "carnot_ising_v4"
    assert artifact["kv260_uio_devices"] == ["/dev/uio0", "/dev/uio1"]
    assert "colognechip GateMate" in artifact["gatemate_detect_output"]
    assert "up 12 minutes" in artifact["polarfire_continuity_output"]
    assert [entry["resource"] for entry in artifact["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert all(isinstance(entry["available"], bool) for entry in artifact["preconditions_checked"])
    assert artifact["honest_verdict"].startswith("complete: hardware_continuity_4006")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4006_unreachable_boards_record_blocked_next_steps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4006: unreachable boards are blocked per board and continue."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: _result(
                mod.KV260_SSH_PRECONDITION,
                returncode=255,
                stderr="ssh: connect to host kria timed out",
            ),
            mod.GATEMATE_DETECT_COMMAND: _result(
                mod.GATEMATE_DETECT_COMMAND,
                returncode=1,
                stdout="IDCode : 0x00000000 unknown",
            ),
            mod.POLARFIRE_SSH_PRECONDITION: _result(
                mod.POLARFIRE_SSH_PRECONDITION,
                returncode=255,
                stderr="ssh: connect to host polarfire timed out",
            ),
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(1.0, 1.2, 1.6, 1.7, 2.4, 2.6, 3.5, 4.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["kv260_reachable"] is False
    assert artifact["gatemate_reachable"] is False
    assert artifact["polarfire_reachable"] is False
    assert artifact["per_board_next_step"] == {
        "kv260": "blocked_kv260_unreachable",
        "gatemate": "blocked_gatemate_unreachable",
        "polarfire": "blocked_polarfire_unreachable",
    }
    assert artifact["per_board_duration_s"] == {
        "kv260": 0.4,
        "gatemate": 0.7,
        "polarfire": 0.9,
    }
    assert artifact["kv260_loaded_overlay"] is None
    assert artifact["gatemate_detect_output"] == "IDCode : 0x00000000 unknown"
    assert artifact["polarfire_continuity_output"] == "skipped: polarfire unreachable"
    mod.validate_artifact(artifact)


def test_req_hw_4006_run_experiment_writes_requested_json(tmp_path: Path) -> None:
    """REQ-HW-4006: run_experiment writes the requested deliverable JSON."""
    runner = RecordingRunner(_successful_results())
    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(2.0, 2.1, 2.7, 3.0, 3.8, 4.0, 5.1, 6.0),
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_fields_are_bare(payload)
    mod.validate_artifact(payload)


def test_req_hw_4006_validation_rejects_wrong_schema_principles_and_kv260_step(
    tmp_path: Path,
) -> None:
    """REQ-HW-4006: validation rejects stale schemas, wrappers, and bad timers."""
    good = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_successful_results()),
        clock=_ticks(5.0, 5.1, 5.7, 6.0, 6.8, 7.0, 8.1, 9.0),
    )

    mutations = [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(experiment=3995), "experiment"),
        (lambda item: item.update(spec_refs=["REQ-HW-3995"]), "spec_refs"),
        (lambda item: item.update(random_seed=3995), "random_seed"),
        (lambda item: item.pop("per_board_next_step"), "missing required fields"),
        (lambda item: item.update(field_principles=[]), "field_principles"),
        (lambda item: item["field_principles"].pop("per_board_next_step"), "missing"),
        (
            lambda item: item["field_principles"].update(
                per_board_next_step="stale principle"
            ),
            "principle",
        ),
        (
            lambda item: item.update(kv260_reachable={"value": True, "principle": "wrapped"}),
            "bare value",
        ),
        (lambda item: item.update(kv260_reachable="yes"), "must be bool"),
        (lambda item: item.update(inference_substrate="live_model"), "hardware_smoke"),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (lambda item: item.update(duration_s=0.0), "duration_s"),
        (
            lambda item: item.update(
                per_board_duration_s={"kv260": 1.0, "gatemate": 1.0, "polarfire": 2.0}
            ),
            "distinct",
        ),
        (
            lambda item: item.update(
                kv260_reachable=True,
                per_board_next_step={
                    "kv260": "kv260_forward_step_run_overlay_latency_smoke",
                    "gatemate": "gatemate_forward_step_run_minimal_ising_tile_smoke",
                    "polarfire": "polarfire_forward_step_run_hash_verified_soft_cpu_dispatch",
                },
            ),
            "terminal",
        ),
        (lambda item: item.update(preconditions_checked=[]), "preconditions_checked"),
        (lambda item: item.update(honest_verdict="pending"), "terminal prefix"),
        (lambda item: item.update(kv260_command_transcripts={"bad": "/dev/mmcblk0"}), "host"),
        (lambda item: item.update(reproducibility_checksum="0" * 64), "checksum"),
    ]
    for mutation, expected in mutations:
        bad = json.loads(json.dumps(good))
        mutation(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_hw_4006_kv260_next_step_points_toward_terminal() -> None:
    """REQ-HW-4006: KV260 next steps stay terminal-oriented."""
    assert (
        mod.kv260_next_step(True, "carnot_ising", ["/dev/uio0"])
        == "kv260_forward_step_run_terminal_overlay_latency_smoke"
    )
    assert (
        mod.kv260_next_step(True, "carnot_ising", [])
        == "kv260_forward_step_restore_uio_binding_then_terminal_latency_smoke"
    )
    assert (
        mod.kv260_next_step(True, None, [])
        == "kv260_forward_step_load_terminal_overlay_per_north_star_section_3"
    )
    assert mod.kv260_next_step(False, None, []) == "blocked_kv260_unreachable"
