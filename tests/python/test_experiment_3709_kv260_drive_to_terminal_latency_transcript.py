"""Tests for Exp 3709 KV260 terminal-candidate latency transcript.

Spec refs: REQ-HW-3709, SCENARIO-HW-3709.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot import experiment_3709_kv260_drive_to_terminal_latency_transcript as mod


class RecordingRunner:
    """Synthetic SSH runner that returns queued probes and rejects surprises."""

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
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _board_stdout(sample_count: int = mod.BOARD_SAMPLE_COUNT) -> str:
    samples = [0.40 + 0.01 * (idx % 7) for idx in range(sample_count)]
    payload = {
        "schema": "carnot.kv260.remote_latency_harness.v1",
        "sample_count": sample_count,
        "per_sample_wall_ms": samples,
        "per_batch_wall_ms": round(sum(samples) + 1.25, 6),
        "fixed_compute_budget": {
            "spin_count": mod.BOARD_SPIN_COUNT,
            "sample_count": sample_count,
            "max_degree": mod.BOARD_MAX_DEGREE,
            "beta_final_q88": mod.BOARD_BETA_FINAL_Q88,
            "trigger_mode": "reset_trigger_poll_done_once_per_sample",
        },
        "selected_uio": "/dev/uio0",
        "selected_uio_addr_hex": "0x00000000a0000000",
    }
    return "BOARD_HARNESS_START exp3709\n" + json.dumps(payload, sort_keys=True) + "\n"


def _assert_required_fields_and_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert "principle" not in str(payload[field]).lower()


def test_req_hw_3709_spec_anchor_declares_terminal_latency_contract() -> None:
    """REQ-HW-3709: OpenSpec declares SSH-only board latency transcript rules."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3709" in spec
    assert "SCENARIO-HW-3709" in spec
    assert "experiment_3709_kv260_drive_to_terminal_latency_transcript.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "xmutil loadapp carnot_ising_v2_n64" in spec
    assert "sudo xmutil" in spec
    assert "host SD-card device-node precondition is permanently retired" in spec
    assert "at least 30 positive on-board latency samples" in spec


@pytest.mark.parametrize(
    ("outcome", "probes", "expected"),
    [
        pytest.param(
            "blocked_ssh_unreachable",
            {
                mod.KV260_SSH_COMMAND: [
                    _probe(
                        mod.KV260_SSH_COMMAND,
                        255,
                        stderr="ssh: connect to host kria port 22: timed out\n",
                    )
                ]
            },
            {
                "verdict": mod.BLOCKED_VERDICT,
                "commands": [mod.KV260_SSH_COMMAND],
                "terminal": False,
                "ssh": False,
                "sample_count": 0,
            },
            id="blocked_ssh_unreachable",
        ),
        pytest.param(
            "latency_transcript_captured",
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="No apps loaded\n"),
                    _probe(
                        mod.KV260_LISTAPPS_COMMAND,
                        0,
                        stdout="carnot_ising_v2_n64  RUNNING\n",
                    ),
                ],
                mod.KV260_LOADAPP_COMMAND: [
                    _probe(
                        mod.KV260_LOADAPP_COMMAND,
                        0,
                        stdout="loaded carnot_ising_v2_n64\n",
                    )
                ],
                mod.KV260_LATENCY_COMMAND: [
                    _probe(mod.KV260_LATENCY_COMMAND, 0, stdout=_board_stdout())
                ],
            },
            {
                "verdict": mod.SUCCESS_VERDICT,
                "commands": [
                    mod.KV260_SSH_COMMAND,
                    mod.KV260_LISTAPPS_COMMAND,
                    mod.KV260_LOADAPP_COMMAND,
                    mod.KV260_LISTAPPS_COMMAND,
                    mod.KV260_LATENCY_COMMAND,
                ],
                "terminal": True,
                "ssh": True,
                "sample_count": mod.BOARD_SAMPLE_COUNT,
            },
            id="latency_transcript_captured",
        ),
    ],
)
def test_scenario_hw_3709_honest_outcomes_are_parametrized(
    outcome: str,
    probes: dict[tuple[str, ...], list[mod.CommandProbe]],
    expected: dict[str, object],
) -> None:
    """SCENARIO-HW-3709: synthetic fixtures cover captured and blocked outcomes."""
    runner = RecordingRunner(probes)

    payload = mod.build_artifact(command_runner=runner, duration_s=2.5)

    assert outcome in {"latency_transcript_captured", "blocked_ssh_unreachable"}
    assert runner.commands == expected["commands"]
    assert payload["honest_verdict"] == expected["verdict"]
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["kv260_ssh_reachable"] is expected["ssh"]
    assert payload["terminal_condition_met"] is expected["terminal"]
    assert len(payload["board_latency_samples"]) == expected["sample_count"]
    assert payload["speedup_claim_avoided_assert"] is True
    assert "gguf" not in json.dumps(payload).lower()
    assert "cuda" not in json.dumps(payload).lower()
    assert "/dev/mmcblk" not in json.dumps(runner.commands)
    assert "/dev/disk" not in json.dumps(runner.commands)
    _assert_required_fields_and_principles(payload)
    if expected["terminal"]:
        assert payload["kv260_overlay_loaded"] == "carnot_ising_v2_n64"
        assert payload["board_latency_median_ms"] == pytest.approx(0.43)
        assert runner.stdin_by_command[mod.KV260_LATENCY_COMMAND] == mod.BOARD_HARNESS_SOURCE
    else:
        assert payload["kv260_overlay_loaded"] is None
        assert payload["board_latency_median_ms"] is None


def test_scenario_hw_3709_loaded_overlay_skips_loadapp() -> None:
    """SCENARIO-HW-3709: listed Carnot overlay is enough before latency capture."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="carnot_ising_v4\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, 0, stdout=_board_stdout())
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=3.0)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LATENCY_COMMAND,
    ]
    assert mod.KV260_LOADAPP_COMMAND not in runner.commands
    assert payload["kv260_overlay_loaded"] == "carnot_ising_v4"
    assert payload["terminal_condition_met"] is True


def test_scenario_hw_3709_xmutil_root_privilege_fallback_stays_over_ssh() -> None:
    """SCENARIO-HW-3709: sudo xmutil fallback preserves the non-sudo probe."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr=root_error)
            ],
            mod.KV260_LISTAPPS_SUDO_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_SUDO_COMMAND,
                    0,
                    stdout="carnot_ising_v4\ncarnot_ising_v2_n64\n",
                )
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, 0, stdout=_board_stdout())
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=3.0)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        mod.KV260_LATENCY_COMMAND,
    ]
    assert payload["command_probes"]["kv260_xmutil_listapps_initial"]["exit_code"] == 1
    assert (
        payload["command_probes"]["kv260_xmutil_listapps_initial_sudo"]["command"]
        == mod.command_to_string(mod.KV260_LISTAPPS_SUDO_COMMAND)
    )
    assert payload["kv260_overlay_loaded"] == "carnot_ising_v2_n64"
    assert payload["terminal_condition_met"] is True


def test_req_hw_3709_field_principles_match_required_why_annotations() -> None:
    """REQ-HW-3709: required fields store values and principles store the why."""
    assert mod.FIELD_PRINCIPLES == {
        "honest_verdict": "Terminal prefix for reconciler classification.",
        "inference_substrate": (
            "SSH-attached board test; per-board duration floor."
        ),
        "preconditions_checked": (
            "Records the SSH-reachability check -- the correct KV260 precondition, "
            "not host SD card."
        ),
        "kv260_ssh_reachable": (
            "The honest board state; an unreachable board is a blocked_*, "
            "not a fabricated pass."
        ),
        "kv260_overlay_loaded": (
            "Confirms the carnot_ising overlay is the latest real-board-deployable "
            "bitstream."
        ),
        "board_latency_samples": (
            "The raw on-board per-sample latency distribution (>=30) -- the "
            "terminal-state transcript, not a single fabricated number."
        ),
        "board_latency_median_ms": (
            "Median on-board latency -- the POC functional anchor (NOT a speedup "
            "claim)."
        ),
        "terminal_condition_met": (
            "True iff a non-fabricated board-latency transcript + overlay "
            "confirmation satisfies the north-star sec-3 terminal condition."
        ),
        "speedup_claim_avoided_assert": (
            "Asserts NO thermalization/equilibrium/hardware-speedup claim is made "
            "(Paper-v6 Narrowing #2/#3)."
        ),
        "random_seed": "Determinism precondition.",
        "reproducibility_checksum": "Drift detection.",
        "duration_s": "Plausibility floor.",
    }


def test_req_hw_3709_extract_board_payload_rejects_missing_json() -> None:
    """REQ-HW-3709: board stdout must include a real JSON transcript."""
    with pytest.raises(ValueError, match="final JSON object"):
        mod.extract_board_payload("only logs\nno json here\n")


def test_req_hw_3709_validate_rejects_short_or_nonpositive_transcripts() -> None:
    """REQ-HW-3709: terminal transcript requires >=30 positive raw samples."""
    payload = mod.extract_board_payload(_board_stdout(sample_count=29))
    with pytest.raises(ValueError, match="at least 30"):
        mod.validate_board_payload(payload)

    payload = mod.extract_board_payload(_board_stdout(sample_count=30))
    payload["per_sample_wall_ms"][3] = 0.0
    with pytest.raises(ValueError, match="positive"):
        mod.validate_board_payload(payload)


def test_req_hw_3709_run_experiment_writes_checksum_and_schema(tmp_path: Path) -> None:
    """REQ-HW-3709: result JSON has required fields, principles, and checksum."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="carnot_ising_v2_n64\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, 0, stdout=_board_stdout())
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=4.0)

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == "exp3709"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 4.0
    assert payload["field_principles"] == mod.FIELD_PRINCIPLES
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3709_run_command_captures_output_and_os_errors() -> None:
    """REQ-HW-3709: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')"),
        timeout_s=10,
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    stdin_probe = mod.run_command(
        (sys.executable, "-"),
        stdin="import sys\nprint(sys.stdin.read().strip() or 'EMPTY')\n",
        timeout_s=10,
    )
    assert stdin_probe.stdout == "EMPTY\n"

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-3709",))
    assert missing_probe.exit_code == 127
    assert missing_probe.command == ("/definitely/missing/ssh-for-req-hw-3709",)


def test_scenario_hw_3709_script_wrapper_exists() -> None:
    """SCENARIO-HW-3709: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3709_kv260_drive_to_terminal_latency_transcript.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3709_kv260_drive_to_terminal_latency_transcript" in text
    assert "main" in text
