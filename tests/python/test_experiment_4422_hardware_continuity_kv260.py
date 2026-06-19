"""Tests for Exp 4422 KV260 SSH-only continuity.

Spec refs: REQ-HW-4422, SCENARIO-HW-4422.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4422_hardware_continuity_kv260 as mod


class RecordingRunner:
    """SCENARIO-HW-4422 command runner with queued KV260 transcripts."""

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


def _write_previous_4411(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4411_hardware_continuity_kv260.json").write_text(
        json.dumps(
            {
                "honest_verdict": "success_kv260_continuity_terminal_reached",
                "kv260_reachable": True,
                "loaded_overlay": {"overlay_names": ["carnot_ising_v4"]},
            }
        ),
        encoding="utf-8",
    )


def test_req_hw_4422_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4422: OpenSpec anchors fields, principles, and SSH-only commands."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4422" in spec
    assert "SCENARIO-HW-4422" in spec
    assert "experiment_4422_hardware_continuity_kv260.json" in spec
    assert "experiment_4411_hardware_continuity_kv260.json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh kria 'ls /dev/uio*'" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert "Host SD-card presence is NEVER a valid precondition" in spec


def test_scenario_hw_4422_reachable_reports_bare_overlay_and_uio(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4422: reachable KV260 emits bare overlay and UIO fields."""
    _write_previous_4411(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout=(
                        "Accelerator Accel_type Base Pid\n"
                        "carnot_ising_v2_n64 XRT_FLAT carnot_ising_v2_n64 id_ok\n"
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
    assert saved["source_context"]["previous_experiment"] == 4411
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["kv260_reachable"] is True
    assert saved["loaded_overlay"] == "carnot_ising_v2_n64"
    assert saved["uio_present"] is True
    assert saved["overlay_probe"]["overlay_names"] == ["carnot_ising_v2_n64"]
    assert saved["uio_probe"]["devices"] == ["/dev/uio0", "/dev/uio3"]
    assert saved["honest_verdict"] == "success_kv260_reachable_overlay_carnot_ising_v2_n64"
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4422_blocks_cleanly_without_overlay_or_uio_when_ssh_fails(
    tmp_path: Path,
) -> None:
    """REQ-HW-4422: failed SSH precondition is an honest documented skip."""
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
    assert saved["loaded_overlay"] is None
    assert saved["uio_present"] is None
    assert saved["overlay_probe"]["status"] == "not_run_kv260_ssh_unreachable"
    assert saved["uio_probe"]["status"] == "not_run_kv260_ssh_unreachable"
    assert saved["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert saved["preconditions_checked"][0]["available"] is False
    assert "xmutil listapps" not in json.dumps(saved["overlay_probe"])
    assert "ls /dev/uio" not in json.dumps(saved["uio_probe"])
    mod.validate_artifact(saved)


def test_req_hw_4422_reachable_preserves_unknown_overlay_and_absent_uio(
    tmp_path: Path,
) -> None:
    """REQ-HW-4422: reachable probes may honestly report unknown overlay and no UIO."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    exit_code=1,
                    stderr="xmutil should be called with root privileges",
                    duration_s=0.3,
                )
            ],
            mod.KV260_UIO_COMMAND: [_probe(mod.KV260_UIO_COMMAND, stdout="", duration_s=0.4)],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["kv260_reachable"] is True
    assert artifact["loaded_overlay"] is None
    assert artifact["uio_present"] is False
    assert artifact["overlay_probe"]["status"] == "overlay_returncode_1"
    assert artifact["uio_probe"]["status"] == "uio_not_seen"
    assert artifact["honest_verdict"] == "success_kv260_reachable_overlay_unknown"
    mod.validate_artifact(artifact)


def test_req_hw_4422_validation_rejects_wrapped_required_fields(tmp_path: Path) -> None:
    """REQ-HW-4422: required fields stay bare, not value/principle wrappers."""
    _write_previous_4411(tmp_path)
    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_valid_runner())
    artifact["kv260_reachable"] = {"value": True, "principle": mod.FIELD_PRINCIPLES["kv260_reachable"]}
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="kv260_reachable must be a bare bool"):
        mod.validate_artifact(artifact)

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=_valid_runner())
    artifact["source_context"]["previous_experiment"] = 4400
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    with pytest.raises(ValueError, match="Exp 4411"):
        mod.validate_artifact(artifact)


def test_req_hw_4422_required_runner_lives_under_results() -> None:
    """REQ-HW-4422: the required `python3 results/...` runner is present."""
    runner = Path("results/experiment_4422_hardware_continuity_kv260.py")

    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "experiment_4422_hardware_continuity_kv260" in text
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
