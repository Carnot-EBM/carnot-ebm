"""Tests for Exp 4040 hardware continuity.

Spec refs: REQ-HW-4040, SCENARIO-HW-4040.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4040_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4040 command runner with queued board transcripts."""

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
    samples = [0.20 + 0.01 * (idx % 5) for idx in range(sample_count)]
    payload = {
        "schema": "carnot.kv260.remote_latency_harness.v1",
        "sample_count": sample_count,
        "per_sample_wall_ms": samples,
        "per_batch_wall_ms": round(sum(samples) + 0.75, 6),
        "fixed_compute_budget": {
            "spin_count": mod.BOARD_SPIN_COUNT,
            "sample_count": sample_count,
            "max_degree": mod.BOARD_MAX_DEGREE,
            "beta_final_q88": mod.BOARD_BETA_FINAL_Q88,
            "trigger_mode": "reset_trigger_poll_done_once_per_sample",
        },
        "selected_uio": "/dev/uio4",
        "selected_uio_addr_hex": "0x00000000a0000000",
    }
    return "BOARD_HARNESS_START exp4040\n" + json.dumps(payload, sort_keys=True) + "\n"


def _base_success_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
        mod.KV260_LISTAPPS_COMMAND: [
            _probe(mod.KV260_LISTAPPS_COMMAND, stdout="No accelerator active\n"),
            _probe(
                mod.KV260_LISTAPPS_COMMAND,
                stdout="carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 0->0,\n",
            ),
        ],
        mod.KV260_LOADAPP_COMMAND: [
            _probe(mod.KV260_LOADAPP_COMMAND, stdout="loaded carnot_ising_v2_n64\n")
        ],
        mod.KV260_LATENCY_COMMAND: [
            _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout(), duration_s=0.25)
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


def _assert_required_fields_are_bare(payload: dict[str, Any]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (
            isinstance(payload[field], dict) and set(payload[field]) == {"value", "principle"}
        )


def test_req_hw_4040_spec_entry_declares_kv260_terminal_drive_contract() -> None:
    """REQ-HW-4040: OpenSpec anchors the continuity and KV260 terminal gate."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4040" in spec
    assert "SCENARIO-HW-4040" in spec
    assert "experiment_4040_hardware_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "xmutil loadapp carnot_ising_v2_n64" in spec
    assert "kv260_overlay_loaded" in spec
    assert "board-latency transcript step" in spec
    assert "/dev/mmcblk" in spec


def test_scenario_hw_4040_absent_overlay_is_loaded_before_latency_step(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4040: absent KV260 overlay is loaded before latency capture."""
    runner = RecordingRunner(_base_success_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(10.0, 10.1, 11.1, 11.4, 11.9, 12.2, 13.0, 14.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LOADAPP_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LATENCY_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.POLARFIRE_CONTINUITY_COMMAND,
    ]
    assert runner.stdin_by_command[mod.KV260_LATENCY_COMMAND] == mod.BOARD_HARNESS_SOURCE
    assert artifact["schema"] == mod.SCHEMA
    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["spec_refs"] == mod.SPEC_REFS
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["per_board_reachability"] == {
        "kv260": True,
        "gatemate": True,
        "polarfire": True,
    }
    assert artifact["kv260_overlay_loaded"] is True
    assert artifact["kv260_loaded_overlay_name"] == "carnot_ising_v2_n64"
    assert artifact["kv260_latency_step_taken"] is True
    assert len(artifact["kv260_latency_samples_ms"]) == mod.BOARD_SAMPLE_COUNT
    assert artifact["kv260_latency_median_ms"] == pytest.approx(0.22)
    assert artifact["kv260_state"] == "reachable_overlay_loaded_latency_step_recorded"
    assert artifact["per_board_terminal_state"]["kv260"] == artifact["kv260_state"]
    assert artifact["per_board_next_step"]["kv260"] == (
        "kv260_terminal_state_overlay_loaded_latency_transcript_landed"
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
    assert all(isinstance(entry["available"], bool) for entry in artifact["preconditions_checked"])
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_overlay_loaded_latency_step_landed_4040"
    )
    assert artifact["fabric_acceleration_claimed"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert "mmcblk" not in json.dumps(artifact).lower()
    _assert_required_fields_are_bare(artifact)
    mod.validate_artifact(artifact)


def test_scenario_hw_4040_loaded_overlay_skips_loadapp(tmp_path: Path) -> None:
    """SCENARIO-HW-4040: existing Carnot overlay is enough for latency step."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v4 active\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="IDCode : 0x20000001 GM1A\n")
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [_probe(mod.POLARFIRE_SSH_PRECONDITION)],
            mod.POLARFIRE_CONTINUITY_COMMAND: [
                _probe(mod.POLARFIRE_CONTINUITY_COMMAND, stdout="Linux polarfire\n")
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(1.0, 1.1, 1.8, 2.0, 2.6, 2.9, 3.8, 4.5),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LATENCY_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.POLARFIRE_CONTINUITY_COMMAND,
    ]
    assert mod.KV260_LOADAPP_COMMAND not in runner.commands
    assert artifact["kv260_overlay_loaded"] is True
    assert artifact["kv260_loaded_overlay_name"] == "carnot_ising_v4"
    assert artifact["kv260_latency_step_taken"] is True
    mod.validate_artifact(artifact)


def test_scenario_hw_4040_xmutil_sudo_fallback_stays_on_ssh_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4040: xmutil sudo fallback still uses KV260 SSH only."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr=root_error),
                _probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr=root_error),
            ],
            mod.KV260_LISTAPPS_SUDO_COMMAND: [
                _probe(mod.KV260_LISTAPPS_SUDO_COMMAND, stdout="No active app\n"),
                _probe(
                    mod.KV260_LISTAPPS_SUDO_COMMAND,
                    stdout="carnot_ising_v2_n64 XRT_FLAT 0->0,\n",
                ),
            ],
            mod.KV260_LOADAPP_COMMAND: [
                _probe(mod.KV260_LOADAPP_COMMAND, 1, stderr=root_error)
            ],
            mod.KV260_LOADAPP_SUDO_COMMAND: [
                _probe(
                    mod.KV260_LOADAPP_SUDO_COMMAND,
                    stdout="loaded carnot_ising_v2_n64\n",
                )
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
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
        clock=_ticks(20.0, 20.1, 21.4, 21.6, 22.0, 22.3, 23.0, 23.5),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        mod.KV260_LOADAPP_COMMAND,
        mod.KV260_LOADAPP_SUDO_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        mod.KV260_LATENCY_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["kv260_overlay_loaded"] is True
    assert artifact["kv260_latency_step_taken"] is True
    assert artifact["per_board_reachability"] == {
        "kv260": True,
        "gatemate": False,
        "polarfire": False,
    }
    assert artifact["per_board_next_step"]["gatemate"] == "blocked_gatemate_unreachable"
    assert artifact["per_board_next_step"]["polarfire"] == "blocked_polarfire_unreachable"
    assert "mmcblk" not in json.dumps(runner.commands).lower()
    mod.validate_artifact(artifact)


def test_scenario_hw_4040_unreachable_boards_keep_blocked_next_steps(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4040: board misses are per-board blockers and continue."""
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
        clock=_ticks(2.0, 2.1, 2.5, 2.8, 3.4, 3.7, 4.6, 5.0),
    )

    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert artifact["honest_verdict"] == "blocked_all_boards_unreachable"
    assert artifact["kv260_overlay_loaded"] is False
    assert artifact["kv260_latency_step_taken"] is False
    assert artifact["kv260_latency_samples_ms"] == []
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
    mod.validate_artifact(artifact)


def test_scenario_hw_4040_overlay_load_failure_blocks_latency_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4040: no latency step is claimed without overlay confirmation."""
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
        clock=_ticks(1.0, 1.1, 1.9, 2.1, 2.5, 2.8, 3.5, 4.0),
    )

    assert mod.KV260_LATENCY_COMMAND not in runner.commands
    assert artifact["kv260_reachable"] is True
    assert artifact["kv260_overlay_loaded"] is False
    assert artifact["kv260_latency_step_taken"] is False
    assert artifact["kv260_state"] == "reachable_overlay_absent_latency_skipped"
    assert artifact["per_board_next_step"]["kv260"] == (
        "kv260_forward_step_load_terminal_overlay_per_north_star_section_3"
    )
    assert artifact["honest_verdict"].startswith(
        "complete: hardware_continuity_kv260_overlay_absent_latency_skipped_4040"
    )
    mod.validate_artifact(artifact)


def test_req_hw_4040_run_experiment_writes_requested_json(tmp_path: Path) -> None:
    """REQ-HW-4040: run_experiment writes the requested deliverable JSON."""
    runner = RecordingRunner(_base_success_probes())

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        clock=_ticks(5.0, 5.1, 6.1, 6.4, 6.9, 7.2, 8.0, 9.0),
    )
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["kv260_overlay_loaded"] is True
    assert payload["kv260_latency_step_taken"] is True
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_fields_are_bare(payload)
    mod.validate_artifact(payload)


def test_req_hw_4040_validation_rejects_invalid_required_contract(
    tmp_path: Path,
) -> None:
    """REQ-HW-4040: validation rejects stale schema, wrappers, and fake claims."""
    good = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_base_success_probes()),
        clock=_ticks(30.0, 30.1, 31.1, 31.4, 31.9, 32.2, 33.0, 34.0),
    )

    mutations = [
        (lambda item: item.update(schema="wrong"), "schema"),
        (lambda item: item.update(experiment=4027), "experiment"),
        (lambda item: item.update(spec_refs=["REQ-HW-4027"]), "spec_refs"),
        (lambda item: item.update(random_seed=4027), "random_seed"),
        (lambda item: item.pop("kv260_overlay_loaded"), "missing required fields"),
        (lambda item: item.update(field_principles=[]), "field_principles"),
        (lambda item: item["field_principles"].pop("kv260_overlay_loaded"), "missing"),
        (
            lambda item: item["field_principles"].update(kv260_overlay_loaded="stale"),
            "principle",
        ),
        (
            lambda item: item.update(kv260_overlay_loaded={"value": True, "principle": "bad"}),
            "bare value",
        ),
        (lambda item: item.update(kv260_overlay_loaded="yes"), "must be bool"),
        (lambda item: item.update(kv260_reachable="yes"), "must be bool"),
        (
            lambda item: item.update(
                per_board_reachability={
                    "kv260": False,
                    "gatemate": True,
                    "polarfire": True,
                }
            ),
            "match scalar",
        ),
        (lambda item: item.update(per_board_reachability={"kv260": True}), "keyed"),
        (
            lambda item: item.update(
                per_board_reachability={
                    "kv260": True,
                    "gatemate": "yes",
                    "polarfire": True,
                }
            ),
            "values must be bool",
        ),
        (lambda item: item.update(per_board_terminal_state={"kv260": "ok"}), "terminal state"),
        (
            lambda item: item["per_board_terminal_state"].update(kv260="other"),
            "observed terminal state",
        ),
        (
            lambda item: item.update(
                per_board_duration_s={"kv260": 1.0, "gatemate": 1.0, "polarfire": 2.0}
            ),
            "distinct",
        ),
        (lambda item: item.update(duration_s=0), "duration_s"),
        (lambda item: item.update(preconditions_checked=[]), "preconditions_checked"),
        (lambda item: item.update(inference_substrate="live_model"), "hardware_smoke"),
        (lambda item: item.update(fabric_acceleration_claimed=True), "must be false"),
        (
            lambda item: item.update(
                kv260_overlay_loaded=True,
                kv260_loaded_overlay_name=None,
            ),
            "overlay name",
        ),
        (
            lambda item: item.update(
                kv260_overlay_loaded=False,
                kv260_latency_step_taken=True,
            ),
            "latency step requires overlay",
        ),
        (
            lambda item: item.update(kv260_latency_step_taken="yes"),
            "latency step must be bool",
        ),
        (
            lambda item: item.update(kv260_latency_step_taken=True, kv260_latency_samples_ms=[]),
            "latency samples",
        ),
        (
            lambda item: item.update(
                kv260_latency_step_taken=True,
                kv260_latency_samples_ms=[0.1, -0.2],
            ),
            "positive",
        ),
        (
            lambda item: item.update(
                kv260_command_transcripts={"bad": "/dev/mmcblk0"},
            ),
            "host",
        ),
        (lambda item: item.update(honest_verdict="pending"), "terminal prefix"),
        (lambda item: item.update(reproducibility_checksum="0" * 64), "checksum"),
    ]
    for mutation, expected in mutations:
        bad = json.loads(json.dumps(good))
        mutation(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)
