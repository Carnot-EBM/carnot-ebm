"""Tests for Exp 4411 KV260 SSH-only hardware continuity.

Spec refs: REQ-HW-4411, SCENARIO-HW-4411.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4411_hardware_continuity_kv260 as mod


class RecordingRunner:
    """SCENARIO-HW-4411 command runner with queued KV260 transcripts."""

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


def _write_previous_4400(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4400_hardware_continuity_kv260.json").write_text(
        json.dumps(
            {
                "honest_verdict": "success_kv260_continuity_terminal_reached",
                "kv260_reachable": True,
                "loaded_overlay": {"carnot_ising_loaded": False},
            }
        ),
        encoding="utf-8",
    )


def _write_terminal_2742(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_2742_kv260_latency_transcript_terminal.json").write_text(
        json.dumps(
            {
                "kv260_synthesis_succeeded": True,
                "kv260_terminal": True,
                "n_cycles_measured": 100,
                "kv260_latency_mean_us": 3.183,
            }
        ),
        encoding="utf-8",
    )


def test_req_hw_4411_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4411: OpenSpec anchors fields, principles, and SSH-only commands."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4411" in spec
    assert "SCENARIO-HW-4411" in spec
    assert "experiment_4411_hardware_continuity_kv260.json" in spec
    assert "experiment_4400_hardware_continuity_kv260.json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert "Host SD-card presence is NEVER a valid precondition" in spec


def test_scenario_hw_4411_reachable_records_loaded_overlay_uio_and_terminal(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4411: reachable KV260 records overlay, UIO, and terminal evidence."""
    _write_previous_4400(tmp_path)
    _write_terminal_2742(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout=(
                        "Accelerator Accel_type Base Pid\n"
                        "carnot_ising_v4 XRT_FLAT carnot_ising_v4 id_ok\n"
                    ),
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [
                _probe(mod.KV260_UIO_COMMAND, stdout="/dev/uio0\n/dev/uio3\n", duration_s=0.4)
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.KV260_LISTAPPS_COMMAND,
        mod.KV260_UIO_COMMAND,
    ]
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["field_principles"]["random_seed"] == mod.FIELD_PRINCIPLES["random_seed"]
    assert saved["source_context"]["previous_experiment"] == 4400
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["source_context"]["previous_honest_verdict"] == (
        "success_kv260_continuity_terminal_reached"
    )
    assert saved["kv260_reachable"] is True
    assert saved["loaded_overlay"]["carnot_ising_loaded"] is True
    assert saved["loaded_overlay"]["overlay_names"] == ["carnot_ising_v4"]
    assert saved["uio_device_presence"]["uio_devices_present"] is True
    assert saved["uio_device_presence"]["devices"] == ["/dev/uio0", "/dev/uio3"]
    assert saved["kv260_terminal_state_reached"] is True
    assert saved["honest_verdict"] == "success_kv260_continuity_terminal_reached"
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4411_blocks_without_overlay_or_uio_when_kv260_ssh_fails(
    tmp_path: Path,
) -> None:
    """REQ-HW-4411: KV260 SSH failure is an honest blocked non-fabrication."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [
                _probe(mod.KV260_SSH_PRECONDITION, 255, stderr="ssh timeout", duration_s=0.6)
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert runner.commands == [mod.KV260_SSH_PRECONDITION]
    assert saved["honest_verdict"] == "blocked_kv260_ssh_unreachable"
    assert saved["kv260_reachable"] is False
    assert saved["loaded_overlay"]["status"] == "not_run_kv260_ssh_unreachable"
    assert saved["uio_device_presence"]["status"] == "not_run_kv260_ssh_unreachable"
    assert saved["kv260_terminal_state_reached"] is False
    assert saved["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert saved["preconditions_checked"][0]["available"] is False
    assert "xmutil listapps" not in json.dumps(saved["loaded_overlay"])
    assert "ls /dev/uio" not in json.dumps(saved["uio_device_presence"])
    mod.validate_artifact(saved)


def test_req_hw_4411_validation_rejects_stale_identity_and_bad_source_context(
    tmp_path: Path,
) -> None:
    """REQ-HW-4411: validation rejects stale identity and missing Exp 4400 context."""
    _write_previous_4400(tmp_path)
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_valid_runner())
    artifact["experiment"] = 4400

    with pytest.raises(ValueError, match="experiment must be 4411"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_valid_runner())
    artifact["source_context"]["previous_experiment"] = 4389
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="Exp 4400"):
        mod.validate_artifact(artifact)


def test_req_hw_4411_required_runner_lives_under_results() -> None:
    """REQ-HW-4411: the required `python3 results/...` runner is present."""
    runner = Path("results/experiment_4411_hardware_continuity_kv260.py")

    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "experiment_4411_hardware_continuity_kv260" in text
    assert "run_experiment" in text


def _valid_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout="carnot_ising_v4 id_ok\n",
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [_probe(mod.KV260_UIO_COMMAND, stdout="/dev/uio4\n")],
        }
    )
