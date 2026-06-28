"""Tests for Exp 4932 KV260 SSH-only overlay continuity.

Spec refs: REQ-HW-4932, SCENARIO-HW-4932.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4932_kv260_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4932 runner with queued SSH-only board transcripts."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        timeout_s: float = 60.0,
    ) -> mod.CommandProbe:
        assert timeout_s > 0.0
        command = tuple(command)
        self.commands.append(command)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


class FlatClock:
    """Deterministic clock for REQ-HW-4932 duration floor assertions."""

    def __call__(self) -> float:
        return 4932.0


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


def _reachable_runner(
    *,
    direct_listapps: mod.CommandProbe | None = None,
    sudo_listapps: mod.CommandProbe | None = None,
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
    }
    if sudo_listapps is not None:
        probes[mod.KV260_LISTAPPS_SUDO_COMMAND] = [sudo_listapps]
    return RecordingRunner(probes)


def _assert_required_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert payload["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
        assert not (isinstance(payload[field], dict) and "principle" in payload[field])


def test_req_hw_4932_spec_anchor_declares_ssh_only_overlay_contract() -> None:
    """REQ-HW-4932: OpenSpec declares the SSH-only overlay contract."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4932" in spec
    assert "SCENARIO-HW-4932" in spec
    assert "experiment_4932_kv260_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "Host SD-card device nodes MUST NOT be used" in spec
    assert "blocked_kv260_wrong_mechanism_sd_card_precondition" in spec
    assert "random_seed=4932" in spec
    assert "duration_s >= 0.0001" in spec
    for field in mod.REQUIRED_PRINCIPLE_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_hw_4932_blocked_ssh_run_experiment_writes_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4932: unreachable SSH still writes the blocked deliverable."""
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
    assert payload["experiment"] == 4932
    assert payload["spec_refs"] == ["REQ-HW-4932", "SCENARIO-HW-4932"]
    assert payload["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["duration_s"] == 0.0001
    assert payload["kv260_ssh_reachable"] is False
    assert payload["preconditions_checked"][0]["available"] is False
    assert payload["preconditions_checked"][0]["discipline"] == "ssh_only_no_host_sd_card"
    assert "No route to host" in payload["preconditions_checked"][0]["observed"]
    assert payload["loaded_overlay"] is None
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    assert "mmcblk" not in json.dumps(payload).lower()
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4932_reachable_board_records_overlay() -> None:
    """SCENARIO-HW-4932: reachable SSH records the current board overlay."""
    runner = _reachable_runner()

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
    ]
    assert payload["honest_verdict"] == "success_kv260_continuity_ok"
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["duration_s"] == 0.0001
    assert payload["kv260_ssh_reachable"] is True
    assert payload["xmutil_requires_sudo"] is False
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert payload["verifier_is_oracle"] is False
    assert payload["random_seed"] == 4932
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)
    _assert_required_principles(payload)
    mod.validate_artifact(payload)


def test_scenario_hw_4932_sudo_xmutil_fallback_preserves_transcripts() -> None:
    """SCENARIO-HW-4932: root-required xmutil uses the SSH-only sudo fallback."""
    root_error = "xmutil should be called with root privileges. Please try again using 'sudo'.\n"
    runner = _reachable_runner(
        direct_listapps=_probe(
            mod.KV260_LISTAPPS_COMMAND,
            exit_code=1,
            stderr=root_error,
            duration_s=0.3,
        ),
        sudo_listapps=_probe(
            mod.KV260_LISTAPPS_SUDO_COMMAND,
            stdout=_listapps_loaded_stdout(),
            duration_s=0.35,
        ),
    )

    payload = mod.build_artifact(command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_SSH_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_LISTAPPS_SUDO_COMMAND,
    ]
    assert payload["xmutil_requires_sudo"] is True
    assert payload["loaded_overlay"] == "carnot_ising_v2_n64"
    assert "root privileges" in payload["command_probes"]["kv260_xmutil_listapps"]["stderr"]
    assert payload["command_probes"]["kv260_xmutil_listapps_sudo"]["exit_code"] == 0
    mod.validate_artifact(payload)


def test_req_hw_4932_validation_rejects_schema_drift_and_wrong_precondition() -> None:
    """REQ-HW-4932: validation rejects schema drift and retired host-storage markers."""
    assert mod.WRONG_MECHANISM_VERDICT == "blocked_kv260_wrong_mechanism_sd_card_precondition"
    assert mod.loaded_overlay_from_xmutil("carnot_ising loaded\n") == "carnot_ising"
    assert mod.loaded_overlay_from_xmutil("no accelerator rows\n") is None
    payload = mod.build_artifact(command_runner=_reachable_runner(), clock=FlatClock())

    bad_schema = dict(payload, schema="stale")
    bad_schema["reproducibility_checksum"] = mod.payload_checksum(bad_schema)
    with pytest.raises(ValueError, match="schema"):
        mod.validate_artifact(bad_schema)

    bad_substrate = dict(payload, inference_substrate="aggregation_from_upstream_artifacts")
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="substrate"):
        mod.validate_artifact(bad_substrate)

    host_sd = dict(payload)
    host_sd["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    host_sd["reproducibility_checksum"] = mod.payload_checksum(host_sd)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(host_sd)

    stale_checksum = dict(payload, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(stale_checksum)
