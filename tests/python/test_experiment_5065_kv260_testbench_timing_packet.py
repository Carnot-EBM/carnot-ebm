"""Tests for Exp 5065 KV260 transcript-backed testbench timing packet.

Spec refs: REQ-HW-5065, SCENARIO-HW-5065.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5065_kv260_testbench_timing_packet as mod


class RecordingRunner:
    """SCENARIO-HW-5065 runner with queued SSH-only board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], timeout_s: float = 60.0) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class FlatClock:
    """Deterministic clock for REQ-HW-5065 duration-floor assertions."""

    def __call__(self) -> float:
        return 5065.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _listapps_expected_overlay_stdout() -> str:
    return (
        "                   Accelerator  Accel_type                    Base    Pid   "
        "Base_type  #slots(RPU+PL+AIE)    slot->handle\n"
        "               carnot_ising_v4    XRT_FLAT         carnot_ising_v4  id_ok    "
        "XRT_FLAT             (0+0+0)              -1\n"
        "           carnot_ising_v2_n64    XRT_FLAT     carnot_ising_v2_n64  id_ok    "
        "XRT_FLAT             (0+0+0)           0->0,\n"
    )


def _listapps_wrong_overlay_stdout() -> str:
    return (
        "                   Accelerator  Accel_type                    Base    Pid   "
        "Base_type  #slots(RPU+PL+AIE)    slot->handle\n"
        "               carnot_ising_v4    XRT_FLAT         carnot_ising_v4  id_ok    "
        "XRT_FLAT             (0+0+0)           0->0,\n"
    )


def _board_payload() -> dict[str, object]:
    return mod.run_cpu_reference(clock=FlatClock())


def _success_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    exit_code=1,
                    stderr="xmutil should be called with root privileges. Please try again using 'sudo'.\n",
                    duration_s=0.3,
                )
            ],
            mod.KV260_LISTAPPS_SUDO_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_SUDO_COMMAND,
                    stdout=_listapps_expected_overlay_stdout(),
                    duration_s=0.4,
                )
            ],
            mod.KV260_UIO_COMMAND: [
                _probe(
                    mod.KV260_UIO_COMMAND,
                    stdout="/dev/uio0\n/dev/uio1\n/dev/uio4\n",
                    duration_s=0.5,
                )
            ],
            mod.KV260_TESTBENCH_COMMAND: [
                _probe(
                    mod.KV260_TESTBENCH_COMMAND,
                    stdout=json.dumps(_board_payload()) + "\n",
                    duration_s=0.75,
                )
            ],
        }
    )


def _success_packet() -> mod.BuiltPacket:
    return mod.build_packet(command_runner=_success_runner(), clock=FlatClock())


def test_req_hw_5065_spec_declares_transcript_backed_contract() -> None:
    """REQ-HW-5065: OpenSpec anchors transcript-backed KV260 evidence."""

    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5065",
        "SCENARIO-HW-5065",
        "experiment_5065_kv260_testbench_timing_packet.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
        "carnot_ising_v2_n64",
        "board_transcript_path",
        "transcript_sha256",
        "structured_testbench_evidence",
        "optional_board_prechecks",
        "Host SD-card device nodes MUST NOT be used",
    ):
        assert marker in spec


def test_scenario_hw_5065_blocked_ssh_writes_transcript_packet(tmp_path: Path) -> None:
    """SCENARIO-HW-5065: unreachable SSH writes blocked artifact and transcript."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [
                _probe(
                    mod.KV260_SSH_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host kria port 22: No route to host\n",
                    duration_s=5.0,
                )
            ]
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, clock=FlatClock())
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    transcript_path = tmp_path / payload["board_transcript_path"]
    transcript_text = transcript_path.read_text(encoding="utf-8")

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["kv260_ssh_reachable"] is False
    assert payload["overlay_loaded"] is False
    assert payload["loaded_overlay"] is None
    assert payload["cpu_reference_ok"] is True
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet_built"] is False
    assert payload["board_transcript_path"] == str(mod.TRANSCRIPT_REL_PATH)
    assert payload["transcript_sha256"] == hashlib.sha256(transcript_text.encode()).hexdigest()
    assert "No route to host" in transcript_text
    assert payload["structured_testbench_evidence"]["status"] == "blocked_kv260_ssh_unreachable"
    assert payload["optional_board_prechecks"]["gatemate"]["status"] == "not_run_scope_guard"
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload, transcript_text=transcript_text)


def test_scenario_hw_5065_expected_overlay_builds_structured_packet() -> None:
    """SCENARIO-HW-5065: confirmed overlay gates parity, timing, and transcript hash."""

    packet = _success_packet()
    payload = packet.artifact

    assert payload["honest_verdict"] == "success_kv260_testbench_timing_packet_built"
    assert payload["kv260_ssh_reachable"] is True
    assert payload["overlay_loaded"] is True
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["cpu_reference_ok"] is True
    assert payload["kv260_result_ok"] is True
    assert payload["timing_ratio_packet_built"] is True
    assert payload["xmutil_requires_sudo"] is True
    assert payload["uio_devices"] == ["/dev/uio0", "/dev/uio1", "/dev/uio4"]
    assert payload["transcript_sha256"] == hashlib.sha256(
        packet.transcript_text.encode()
    ).hexdigest()
    assert "root privileges" in packet.transcript_text
    assert "bounded_sparse_pbit_parity_n64" in packet.transcript_text

    evidence = payload["structured_testbench_evidence"]
    assert evidence["status"] == "packet_built"
    assert evidence["transcript"]["sha256"] == payload["transcript_sha256"]
    assert evidence["parity"]["mismatches"] == []
    assert evidence["timing_ratio_packet"]["parity_match"] is True
    assert evidence["board_result"]["final_state_checksum"] == payload["cpu_reference"][
        "final_state_checksum"
    ]
    assert payload["timing_ratio_packet"]["cpu_to_kv260_command_wall_ratio"] == pytest.approx(
        0.000133333333
    )
    assert "no_general_fpga_speedup_claim" in payload["local_claim_scope"]
    mod.validate_artifact(payload, transcript_text=packet.transcript_text)


def test_scenario_hw_5065_overlay_not_confirmed_blocks_workload() -> None:
    """SCENARIO-HW-5065: wrong overlay is recorded but does not run the testbench."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout=_listapps_wrong_overlay_stdout(),
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [
                _probe(mod.KV260_UIO_COMMAND, stdout="/dev/uio0\n", duration_s=0.4)
            ],
        }
    )

    packet = mod.build_packet(command_runner=runner, clock=FlatClock())
    payload = packet.artifact

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert payload["honest_verdict"] == "blocked_kv260_expected_overlay_not_confirmed"
    assert payload["overlay_loaded"] is False
    assert payload["loaded_overlay"] == "carnot_ising_v4"
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet_built"] is False
    assert payload["command_probes"]["kv260_testbench_workload"] is None
    assert payload["structured_testbench_evidence"]["status"] == (
        "blocked_kv260_expected_overlay_not_confirmed"
    )
    mod.validate_artifact(payload, transcript_text=packet.transcript_text)


def test_scenario_hw_5065_parity_failure_blocks_timing_packet() -> None:
    """SCENARIO-HW-5065: mismatched board output cannot build a timing packet."""

    board_payload = _board_payload()
    board_payload["energy"] = int(board_payload["energy"]) + 1
    runner = _success_runner()
    runner.probes[mod.KV260_TESTBENCH_COMMAND] = [
        _probe(
            mod.KV260_TESTBENCH_COMMAND,
            stdout=json.dumps(board_payload) + "\n",
            duration_s=0.75,
        )
    ]

    packet = mod.build_packet(command_runner=runner, clock=FlatClock())
    payload = packet.artifact

    assert payload["honest_verdict"] == "blocked_kv260_testbench_parity_failed"
    assert payload["overlay_loaded"] is True
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet_built"] is False
    assert payload["timing_ratio_packet"] is None
    assert "energy" in payload["structured_testbench_evidence"]["parity"]["mismatches"]
    mod.validate_artifact(payload, transcript_text=packet.transcript_text)

    failing_runner = _success_runner()
    failing_runner.probes[mod.KV260_TESTBENCH_COMMAND] = [
        _probe(
            mod.KV260_TESTBENCH_COMMAND,
            exit_code=2,
            stderr="runtime failed\n",
            duration_s=0.75,
        )
    ]
    failed_packet = mod.build_packet(command_runner=failing_runner, clock=FlatClock())
    failed_payload = failed_packet.artifact

    assert failed_payload["honest_verdict"] == "blocked_kv260_testbench_parity_failed"
    assert failed_payload["kv260_workload"] is None
    assert failed_payload["kv260_result_ok"] is False
    assert "runtime failed" in failed_packet.transcript_text
    mod.validate_artifact(failed_payload, transcript_text=failed_packet.transcript_text)


def test_req_hw_5065_helpers_and_validation_reject_drift() -> None:
    """REQ-HW-5065: helpers and schema guards reject transcript/schema drift."""

    reference = mod.run_cpu_reference(clock=FlatClock())
    assert reference["duration_s"] == pytest.approx(0.0001)
    assert mod.parse_testbench_stdout("noise\n" + json.dumps(reference) + "\n") == reference
    assert mod.parse_testbench_stdout("not-json") is None
    assert mod.parse_testbench_stdout("{not-json}") is None
    assert mod.parse_uio_devices("/dev/uio1\n/dev/uio1\n/dev/uio3\n") == [
        "/dev/uio1",
        "/dev/uio3",
    ]
    assert mod.confirmed_existing_overlay(_listapps_expected_overlay_stdout()) == (
        True,
        "carnot_ising_v2_n64",
    )
    assert mod.confirmed_existing_overlay("no accelerator rows\n") == (False, None)

    packet = _success_packet()
    payload = packet.artifact

    bad_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum, transcript_text=packet.transcript_text)

    bad_scope = dict(payload, local_claim_scope="general FPGA speedup claim")
    bad_scope["reproducibility_checksum"] = mod.payload_checksum(bad_scope)
    with pytest.raises(ValueError, match="local_claim_scope"):
        mod.validate_artifact(bad_scope, transcript_text=packet.transcript_text)

    bad_transcript = dict(payload, transcript_sha256="0" * 64)
    bad_transcript["structured_testbench_evidence"] = dict(
        payload["structured_testbench_evidence"],
        transcript=dict(payload["structured_testbench_evidence"]["transcript"], sha256="0" * 64),
    )
    bad_transcript["reproducibility_checksum"] = mod.payload_checksum(bad_transcript)
    with pytest.raises(ValueError, match="transcript"):
        mod.validate_artifact(bad_transcript, transcript_text=packet.transcript_text)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd, transcript_text=packet.transcript_text)
