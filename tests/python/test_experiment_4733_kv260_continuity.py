"""Tests for Exp 4733 KV260 continuity.

Spec refs: REQ-HW-4733, SCENARIO-HW-4733.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot import experiment_4733_kv260_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4733 command runner with queued SSH transcripts."""

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
        assert timeout_s > 0.0
        command = tuple(command)
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
    return mod.CommandProbe(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _board_stdout(sample_count: int = mod.BOARD_SAMPLE_COUNT) -> str:
    samples = [3.0 + 0.1 * (idx % 5) for idx in range(sample_count)]
    payload = {
        "schema": "carnot.kv260.remote_latency_harness.v2",
        "sample_count": sample_count,
        "per_sample_wall_clock_us": samples,
        "per_batch_wall_clock_us": round(sum(samples) + 4.0, 6),
        "fixed_compute_budget": dict(mod.DEFAULT_FIXED_COMPUTE_BUDGET),
        "selected_uio": "/dev/uio4",
        "selected_uio_addr_hex": "0x00000000a0000000",
        "final_spin_words_hex": ["0xffffffff", "0x00000000"],
    }
    return "BOARD_HARNESS_START exp4733\n" + json.dumps(payload, sort_keys=True) + "\n"


def _assert_required_fields_and_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert field in payload["field_principles"]
        assert "principle" not in str(payload[field]).lower()


def test_req_hw_4733_spec_anchor_declares_required_contract() -> None:
    """REQ-HW-4733: OpenSpec declares the SSH-gated KV260 transcript contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4733" in spec
    assert "SCENARIO-HW-4733" in spec
    assert "experiment_4733_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "sudo xmutil loadapp carnot_ising_v4_n64" in spec
    assert "ssh kria 'sudo python3 -'" in spec
    assert "host SD-card device-node precondition" in spec
    assert "kv260_latency_numbers" in spec
    assert "kv260_synthesis_succeeded" in spec
    assert "verifier_is_oracle=false" in spec
    for field in mod.REQUIRED_OPERATOR_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4733_blocked_ssh_writes_honest_continuity_record() -> None:
    """SCENARIO-HW-4733: unreachable SSH records no fabricated latency values."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(
                    mod.KV260_SSH_COMMAND,
                    255,
                    stderr="ssh: connect to host kria port 22: timed out\n",
                    duration_s=5.0,
                )
            ]
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=5.2)

    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["honest_verdict"] == "complete:/blocked_kv260_ssh_unreachable"
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["kv260_ssh_reachable"] is False
    assert payload["kv260_latency_numbers"] is None
    assert payload["kv260_synthesis_succeeded"] is False
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 4733
    assert payload["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": False,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "exit_code": 255,
            "duration_s": 5.0,
            "observed": "ssh: connect to host kria port 22: timed out",
            "discipline": "ssh_only_no_host_sd_card",
        }
    ]
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_fields_and_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4733_sudo_xmutil_fallback_loads_and_captures_latency() -> None:
    """SCENARIO-HW-4733: sudo xmutil fallback captures the UIO latency transcript."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    load_command = mod.loadapp_command("carnot_ising_v2_n64")
    available_apps = (
        "carnot_ising_v4 XRT_FLAT carnot_ising_v4 id_ok XRT_FLAT (0+0+0) -1\n"
        "carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok XRT_FLAT (0+0+0) -1\n"
    )
    loaded_apps = (
        "carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok XRT_FLAT "
        "(0+0+0) 0->0,\n"
    )
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr=root_error, duration_s=0.3)
            ],
            mod.KV260_LISTAPPS_SUDO_COMMAND: [
                _probe(mod.KV260_LISTAPPS_SUDO_COMMAND, stdout=available_apps, duration_s=0.4),
                _probe(mod.KV260_LISTAPPS_SUDO_COMMAND, stdout=loaded_apps, duration_s=0.5),
            ],
            load_command: [
                _probe(load_command, stdout="Loaded with slot_handle 0\n", duration_s=2.0)
            ],
            mod.KV260_BITSTREAM_SHA_COMMAND: [
                _probe(
                    mod.KV260_BITSTREAM_SHA_COMMAND,
                    stdout=f"{'a' * 64}  /lib/firmware/carnot_ising_v2_n64.bit.bin\n",
                    duration_s=0.1,
                )
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout(), duration_s=1.5)
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=4.7)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        load_command,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
        mod.KV260_BITSTREAM_SHA_COMMAND,
        mod.KV260_LATENCY_COMMAND,
    ]
    assert runner.stdin_by_command[mod.KV260_LATENCY_COMMAND] == mod.BOARD_HARNESS_SOURCE
    assert payload["honest_verdict"] == "success: kv260_latency_transcript_captured"
    assert payload["kv260_ssh_reachable"] is True
    assert payload["kv260_synthesis_succeeded"] is True
    assert payload["overlay_loaded"] == "carnot_ising_v2_n64"
    assert payload["bitstream_sha256"] == "a" * 64
    numbers = payload["kv260_latency_numbers"]
    assert numbers["unit"] == "us"
    assert numbers["sample_count"] == mod.BOARD_SAMPLE_COUNT
    assert len(numbers["per_sample_wall_clock_us"]) == mod.BOARD_SAMPLE_COUNT
    assert numbers["median_us"] == pytest.approx(3.2)
    assert numbers["min_us"] == pytest.approx(3.0)
    assert numbers["max_us"] == pytest.approx(3.4)
    assert numbers["selected_uio"] == "/dev/uio4"
    assert payload["verifier_is_oracle"] is False
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_fields_and_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4733_loaded_overlay_skips_loadapp() -> None:
    """SCENARIO-HW-4733: an already loaded Carnot overlay avoids reload churn."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout="carnot_ising_v4 RUNNING slot_handle 0\n",
                )
            ],
            mod.KV260_BITSTREAM_SHA_COMMAND: [
                _probe(mod.KV260_BITSTREAM_SHA_COMMAND, stdout=f"{'b' * 64}  firmware.bit\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=2.0)

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_BITSTREAM_SHA_COMMAND,
        mod.KV260_LATENCY_COMMAND,
    ]
    assert payload["overlay_loaded"] == "carnot_ising_v4"
    assert payload["kv260_synthesis_succeeded"] is True
    mod.validate_artifact(payload)


def test_req_hw_4733_board_payload_validation_rejects_bad_transcripts() -> None:
    """REQ-HW-4733: terminal latency numbers require >=30 positive samples."""
    payload = mod.extract_board_payload(_board_stdout(sample_count=29))
    with pytest.raises(ValueError, match="at least 30"):
        mod.validate_board_payload(payload)

    payload = mod.extract_board_payload(_board_stdout(sample_count=30))
    payload["per_sample_wall_clock_us"][0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        mod.validate_board_payload(payload)

    payload = mod.extract_board_payload(_board_stdout(sample_count=30))
    payload["per_batch_wall_clock_us"] = 0.0
    with pytest.raises(ValueError, match="batch"):
        mod.validate_board_payload(payload)

    with pytest.raises(ValueError, match="final JSON"):
        mod.extract_board_payload("BOARD_HARNESS_START exp4733\nno json\n")


def test_req_hw_4733_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-4733: run_experiment writes the requested results JSON."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v4 RUNNING\n")
            ],
            mod.KV260_BITSTREAM_SHA_COMMAND: [
                _probe(mod.KV260_BITSTREAM_SHA_COMMAND, stdout=f"{'c' * 64}  firmware.bit\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=3.0)
    payload = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == 4733
    assert payload["spec_refs"] == ["REQ-HW-4733", "SCENARIO-HW-4733"]
    assert payload["duration_s"] == 3.0
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_req_hw_4733_validation_and_command_helpers() -> None:
    """REQ-HW-4733: validation rejects stale metadata and command probes work."""
    payload = mod.build_artifact(
        command_runner=RecordingRunner(
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v4 RUNNING\n")
                ],
                mod.KV260_BITSTREAM_SHA_COMMAND: [
                    _probe(mod.KV260_BITSTREAM_SHA_COMMAND, stdout=f"{'d' * 64}  firmware.bit\n")
                ],
                mod.KV260_LATENCY_COMMAND: [
                    _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
                ],
            }
        ),
        duration_s=1.0,
    )

    bad_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    wrapped = dict(payload)
    wrapped["honest_verdict"] = {"value": "success", "principle": "forbidden"}
    wrapped["reproducibility_checksum"] = mod.payload_checksum(wrapped)
    with pytest.raises(ValueError, match="bare value"):
        mod.validate_artifact(wrapped)

    bad_samples = dict(payload)
    bad_samples["kv260_latency_numbers"] = dict(payload["kv260_latency_numbers"])
    bad_samples["kv260_latency_numbers"]["per_sample_wall_clock_us"] = [1.0]
    bad_samples["reproducibility_checksum"] = mod.payload_checksum(bad_samples)
    with pytest.raises(ValueError, match="at least 30"):
        mod.validate_artifact(bad_samples)

    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"),
        stdin="REQ-HW-4733",
        timeout_s=10,
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "REQ-HW-4733\n"

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-4733",))
    assert missing_probe.exit_code == 127
    assert missing_probe.command == ("/definitely/missing/ssh-for-req-hw-4733",)

    assert mod._select_load_app("no Carnot rows listed") == "carnot_ising_v2_n64"
    assert mod._parse_bitstream_sha("sha256sum found no bitstream") is None


def test_scenario_hw_4733_load_output_can_confirm_overlay_when_listapps_is_stale() -> None:
    """SCENARIO-HW-4733: loadapp success can confirm the overlay if listapps lags."""
    load_command = mod.loadapp_command("carnot_ising_v2_n64")
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v2_n64 ... -1\n"),
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v2_n64 ... -1\n"),
            ],
            load_command: [
                _probe(load_command, stdout="Loaded with slot_handle 0\n")
            ],
            mod.KV260_BITSTREAM_SHA_COMMAND: [
                _probe(mod.KV260_BITSTREAM_SHA_COMMAND, stdout=f"{'e' * 64}  firmware.bit\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(mod.KV260_LATENCY_COMMAND, stdout=_board_stdout())
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=1.0)

    assert payload["overlay_loaded"] == "carnot_ising_v2_n64"
    assert payload["kv260_synthesis_succeeded"] is True
    mod.validate_artifact(payload)


def test_scenario_hw_4733_latency_harness_failure_is_not_fabricated() -> None:
    """SCENARIO-HW-4733: a failed on-board harness raises before writing success."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, stdout="carnot_ising_v4 RUNNING\n")
            ],
            mod.KV260_BITSTREAM_SHA_COMMAND: [
                _probe(mod.KV260_BITSTREAM_SHA_COMMAND, stdout=f"{'f' * 64}  firmware.bit\n")
            ],
            mod.KV260_LATENCY_COMMAND: [
                _probe(
                    mod.KV260_LATENCY_COMMAND,
                    exit_code=1,
                    stderr="sampler poll timed out",
                )
            ],
        }
    )

    with pytest.raises(RuntimeError, match="latency harness failed"):
        mod.build_artifact(command_runner=runner, duration_s=1.0)
