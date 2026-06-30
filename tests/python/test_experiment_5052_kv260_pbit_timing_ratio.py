"""Tests for Exp 5052 KV260 p-bit timing-ratio parity packet.

Spec refs: REQ-HW-5052, SCENARIO-HW-5052.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5052_kv260_pbit_timing_ratio as mod


class RecordingRunner:
    """SCENARIO-HW-5052 runner with queued SSH-only board transcripts."""

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
    """Deterministic clock for REQ-HW-5052 duration floor assertions."""

    def __call__(self) -> float:
        return 5052.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _listapps_loaded_stdout() -> str:
    return (
        "                   Accelerator  Accel_type                    Base    Pid   "
        "Base_type  #slots(RPU+PL+AIE)    slot->handle\n"
        "               carnot_ising_v4    XRT_FLAT         carnot_ising_v4  id_ok    "
        "XRT_FLAT             (0+0+0)              -1\n"
        "           carnot_ising_v2_n64    XRT_FLAT     carnot_ising_v2_n64  id_ok    "
        "XRT_FLAT             (0+0+0)           0->0,\n"
    )


def _listapps_not_loaded_stdout() -> str:
    return (
        "                   Accelerator  Accel_type                    Base    Pid   "
        "Base_type  #slots(RPU+PL+AIE)    slot->handle\n"
        "              k24-starter-kits    XRT_FLAT        k24-starter-kits  id_ok    "
        "XRT_FLAT             (0+0+0)              -1\n"
    )


def _board_payload() -> dict[str, object]:
    return mod.run_pbit_reference(clock=FlatClock())


def _success_payload() -> dict[str, object]:
    runner = RecordingRunner(
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
                    stdout=_listapps_loaded_stdout(),
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
            mod.KV260_PBIT_WORKLOAD_COMMAND: [
                _probe(
                    mod.KV260_PBIT_WORKLOAD_COMMAND,
                    stdout=json.dumps(_board_payload()) + "\n",
                    duration_s=0.75,
                )
            ],
        }
    )
    return mod.build_artifact(command_runner=runner, clock=FlatClock())


def test_req_hw_5052_spec_declares_ssh_only_timing_packet_contract() -> None:
    """REQ-HW-5052: OpenSpec anchors the local timing-ratio packet schema."""

    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5052",
        "SCENARIO-HW-5052",
        "experiment_5052_kv260_pbit_timing_ratio.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
        "bounded_sparse_pbit_parity_n64",
        "timing_ratio_packet_built",
        "cpu_reference_ok",
        "kv260_result_ok",
        "local_claim_scope",
    ):
        assert marker in spec
    assert "Host SD-card device nodes MUST NOT be used" in spec


def test_scenario_hw_5052_blocked_ssh_still_writes_cpu_reference_packet(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-5052: unreachable SSH writes blocked artifact with CPU reference."""

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

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["kv260_ssh_reachable"] is False
    assert payload["overlay_loaded"] is False
    assert payload["workload_name"] == mod.WORKLOAD_NAME
    assert payload["n_variables"] == 64
    assert payload["timing_ratio_packet_built"] is False
    assert payload["cpu_reference_ok"] is True
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet"] is None
    assert payload["command_probes"]["kv260_pbit_workload"] is None
    assert payload["preconditions_checked"][0]["discipline"] == "ssh_only_no_host_sd_card"
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_5052_loaded_overlay_builds_parity_timing_ratio_packet() -> None:
    """SCENARIO-HW-5052: loaded overlay gates KV260 workload and parity packet."""

    payload = _success_payload()

    assert payload["honest_verdict"] == "success_kv260_pbit_timing_ratio_packet_built"
    assert payload["kv260_ssh_reachable"] is True
    assert payload["overlay_loaded"] is True
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["uio_devices"] == ["/dev/uio0", "/dev/uio1", "/dev/uio4"]
    assert payload["xmutil_requires_sudo"] is True
    assert payload["cpu_reference_ok"] is True
    assert payload["kv260_result_ok"] is True
    assert payload["timing_ratio_packet_built"] is True
    packet = payload["timing_ratio_packet"]
    assert packet["parity_match"] is True
    assert packet["iterations"] == 128
    assert packet["flips"] == payload["cpu_reference"]["flips"]
    assert packet["cpu_wall_clock_s"] == pytest.approx(0.0001)
    assert packet["kv260_command_wall_clock_s"] == pytest.approx(0.75)
    assert packet["kv260_board_reported_workload_s"] == pytest.approx(0.0001)
    assert packet["cpu_to_kv260_command_wall_ratio"] == pytest.approx(0.000133333333)
    assert "no_general_fpga_speedup_claim" in payload["local_claim_scope"]
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 5052
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_5052_reachable_without_overlay_skips_board_workload() -> None:
    """SCENARIO-HW-5052: missing overlay records probe state without ratio claim."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout=_listapps_not_loaded_stdout(),
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [
                _probe(mod.KV260_UIO_COMMAND, stdout="/dev/uio0\n", duration_s=0.4)
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert payload["honest_verdict"] == "success_kv260_reachable_overlay_not_loaded"
    assert payload["overlay_loaded"] is False
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet_built"] is False
    assert payload["kv260_workload"] is None
    assert payload["timing_ratio_packet"] is None
    mod.validate_artifact(payload)


def test_scenario_hw_5052_loaded_overlay_parity_failure_is_not_packet() -> None:
    """SCENARIO-HW-5052: failed board workload cannot build a timing packet."""

    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout="carnot_ising_v4 running\n",
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [
                _probe(mod.KV260_UIO_COMMAND, stdout="/dev/uio0\n", duration_s=0.4)
            ],
            mod.KV260_PBIT_WORKLOAD_COMMAND: [
                _probe(
                    mod.KV260_PBIT_WORKLOAD_COMMAND,
                    exit_code=2,
                    stderr="runtime failed\n",
                    duration_s=0.6,
                )
            ],
        }
    )

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert payload["honest_verdict"] == "success_kv260_reachable_overlay_loaded_parity_failed"
    assert payload["overlay_loaded"] is True
    assert payload["loaded_overlay"] == "carnot_ising_v4"
    assert payload["kv260_result_ok"] is False
    assert payload["timing_ratio_packet_built"] is False
    assert payload["kv260_workload"] is None
    assert payload["timing_ratio_packet"] is None
    mod.validate_artifact(payload)


def test_req_hw_5052_helpers_and_schema_reject_drift() -> None:
    """REQ-HW-5052: parser and schema guards reject parity packet drift."""

    reference = mod.run_pbit_reference(clock=FlatClock())
    assert reference["workload_name"] == mod.WORKLOAD_NAME
    assert reference["n_variables"] == 64
    assert reference["iterations"] == 128
    assert reference["duration_s"] == pytest.approx(0.0001)
    assert mod.parse_workload_stdout("noise\n" + json.dumps(reference) + "\n") == reference
    assert mod.parse_workload_stdout("not json") is None
    assert mod.parse_workload_stdout("{not-json}") is None

    payload = _success_payload()
    bad_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    bad_scope = dict(payload, local_claim_scope="global FPGA speedup claim")
    bad_scope["reproducibility_checksum"] = mod.payload_checksum(bad_scope)
    with pytest.raises(ValueError, match="local_claim_scope"):
        mod.validate_artifact(bad_scope)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd)
