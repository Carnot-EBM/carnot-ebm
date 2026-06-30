"""Tests for Exp 5037 KV260 SSH-only overlay/UIO energy continuity.

Spec refs: REQ-HW-5037, SCENARIO-HW-5037.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5037_kv260_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-5037 runner with queued SSH-only board transcripts."""

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
    """Deterministic clock for REQ-HW-5037 duration floor assertions."""

    def __call__(self) -> float:
        return 5037.0


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


def _energy_stdout(energy: int = -7) -> str:
    return (
        json.dumps(
            {
                "problem": "tiny_quadratic_ising_constraint_energy",
                "energy": energy,
                "expected_energy": -7,
                "duration_s": 0.000123,
            }
        )
        + "\n"
    )


def _reachable_runner(
    *,
    direct_listapps: mod.CommandProbe | None = None,
    energy_probe: mod.CommandProbe | None = None,
) -> RecordingRunner:
    probes = {
        mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, duration_s=0.2)],
        mod.KV260_LISTAPPS_COMMAND: [
            direct_listapps
            or _probe(
                mod.KV260_LISTAPPS_COMMAND,
                stdout=_listapps_loaded_stdout(),
                duration_s=0.3,
            )
        ],
        mod.KV260_UIO_COMMAND: [
            _probe(
                mod.KV260_UIO_COMMAND,
                stdout="/dev/uio0\n/dev/uio1\n/dev/uio4\n",
                duration_s=0.4,
            )
        ],
    }
    if energy_probe is not None:
        probes[mod.KV260_ENERGY_COMMAND] = [energy_probe]
    return RecordingRunner(probes)


def _success_payload() -> dict[str, object]:
    runner = _reachable_runner(
        energy_probe=_probe(
            mod.KV260_ENERGY_COMMAND,
            stdout=_energy_stdout(),
            duration_s=0.44,
        )
    )
    return mod.build_artifact(command_runner=runner, clock=FlatClock())


def test_req_hw_5037_spec_anchor_declares_overlay_uio_energy_contract() -> None:
    """REQ-HW-5037: OpenSpec declares the SSH-only overlay/UIO energy contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-5037" in spec
    assert "SCENARIO-HW-5037" in spec
    assert "experiment_5037_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "Host SD-card device nodes MUST NOT be used" in spec
    assert "on_board_energy_duration_s > 0" in spec
    assert "random_seed=5037" in spec
    assert "duration_s >= 0.0001" in spec
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_5037_blocked_ssh_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5037: unreachable SSH still writes the blocked deliverable."""
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
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment"] == 5037
    assert payload["spec_refs"] == ["REQ-HW-5037", "SCENARIO-HW-5037"]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["kv260_ssh_reachable"] is False
    assert payload["kv260_ssh_exit_code"] == 255
    assert payload["preconditions_checked"][0]["discipline"] == "ssh_only_no_host_sd_card"
    assert payload["overlay_state"]["loaded_overlay"] is None
    assert payload["overlay_state"]["uio_devices"] == []
    assert payload["on_board_energy_duration_s"] is None
    assert payload["energy_smoke"] is None
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    assert payload["command_probes"]["kv260_uio_devices"] is None
    assert payload["command_probes"]["kv260_energy_smoke"] is None
    assert payload["random_seed"] == 5037
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_5037_reachable_loaded_overlay_runs_energy_smoke() -> None:
    """SCENARIO-HW-5037: loaded Carnot overlay gates the energy smoke."""
    runner = _reachable_runner(
        energy_probe=_probe(
            mod.KV260_ENERGY_COMMAND,
            stdout=_energy_stdout(),
            duration_s=0.44,
        )
    )

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
        mod.KV260_ENERGY_COMMAND,
    ]
    assert payload["honest_verdict"] == "success_kv260_reachable_overlay_loaded_energy_ok"
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["duration_s"] == 0.0001
    assert payload["kv260_ssh_reachable"] is True
    assert payload["overlay_state"]["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["overlay_state"]["uio_devices"] == ["/dev/uio0", "/dev/uio1", "/dev/uio4"]
    assert payload["on_board_energy_duration_s"] == 0.44
    assert payload["energy_smoke"]["energy"] == -7
    assert payload["energy_smoke"]["success"] is True
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 5037
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_5037_reachable_without_overlay_skips_energy_smoke() -> None:
    """SCENARIO-HW-5037: no loaded Carnot overlay stays reachability-only."""
    runner = _reachable_runner(
        direct_listapps=_probe(
            mod.KV260_LISTAPPS_COMMAND,
            stdout=_listapps_not_loaded_stdout(),
            duration_s=0.3,
        )
    )

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert payload["honest_verdict"] == "success_kv260_reachable_overlay_not_loaded"
    assert payload["overlay_state"]["loaded_overlay"] is None
    assert payload["on_board_energy_duration_s"] is None
    assert payload["energy_smoke"] is None
    assert payload["command_probes"]["kv260_energy_smoke"] is None
    mod.validate_artifact(payload)


def test_req_hw_5037_validation_rejects_schema_drift_and_wrong_precondition() -> None:
    """REQ-HW-5037: validation rejects schema drift and retired host-storage markers."""
    assert mod.parse_uio_devices("/dev/uio1 /dev/uio1\n/dev/uio3\n") == [
        "/dev/uio1",
        "/dev/uio3",
    ]
    assert mod.loaded_overlay_from_xmutil("carnot_ising loaded\n") == "carnot_ising"
    assert mod.parse_energy_smoke_stdout(_energy_stdout())["energy"] == -7
    payload = _success_payload()

    bad_experiment = dict(payload, experiment=5009)
    bad_experiment["reproducibility_checksum"] = mod.payload_checksum(bad_experiment)
    with pytest.raises(ValueError, match="experiment"):
        mod.validate_artifact(bad_experiment)

    bad_spec_refs = dict(payload, spec_refs=["REQ-HW-5009", "SCENARIO-HW-5009"])
    bad_spec_refs["reproducibility_checksum"] = mod.payload_checksum(bad_spec_refs)
    with pytest.raises(ValueError, match="spec_refs"):
        mod.validate_artifact(bad_spec_refs)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd)

    stale_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(stale_checksum)
