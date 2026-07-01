"""Tests for Exp 5079 board continuity matrix.

Spec refs: REQ-HW-5079, SCENARIO-HW-5079.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5079_board_continuity_matrix as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"
SOURCE_5065_ARTIFACT = REPO / mod.PRIOR_KV260_ARTIFACT_REL_PATH
SOURCE_5065_TRANSCRIPT = REPO / mod.PRIOR_KV260_TRANSCRIPT_REL_PATH


class RecordingRunner:
    """SCENARIO-HW-5079 runner with queued non-destructive precheck transcripts."""

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
    """Deterministic clock for REQ-HW-5079 duration-floor assertions."""

    def __call__(self) -> float:
        return 5079.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    artifact_path = root / mod.PRIOR_KV260_ARTIFACT_REL_PATH
    transcript_path = root / mod.PRIOR_KV260_TRANSCRIPT_REL_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(SOURCE_5065_ARTIFACT.read_text(encoding="utf-8"), encoding="utf-8")
    transcript_path.write_text(
        SOURCE_5065_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8"
    )
    return root


def _all_reachable_runner() -> RecordingRunner:
    return RecordingRunner(
        {
            mod.KV260_PRECONDITION_COMMAND: [
                _probe(mod.KV260_PRECONDITION_COMMAND, duration_s=0.2)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.3,
                )
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(mod.POLARFIRE_PRECONDITION_COMMAND, duration_s=0.4)
            ],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(
                    mod.POLARFIRE_UPTIME_COMMAND,
                    stdout=" 01:24:00 up 8 days,  7:00,  1 user,  load average: 0.00\n",
                    duration_s=0.5,
                )
            ],
        }
    )


def test_req_hw_5079_spec_declares_board_matrix_contract() -> None:
    """REQ-HW-5079: OpenSpec anchors the board matrix fields and commands."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-HW-5079",
        "SCENARIO-HW-5079",
        "experiment_5079_board_continuity_matrix.py",
        "results/experiment_5079_board_continuity_matrix_v466.json",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'",
        "openFPGALoader -c dirtyJtag --detect",
        "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire 'true'",
        "hardware_precheck_and_upstream_artifact_audit",
        "success_board_continuity_matrix_written_no_speedup_claim",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5079_reachable_matrix_audits_prior_kv260_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-5079: all boards visible without flashing or speedup claims."""

    root = _fixture_root(tmp_path)
    runner = _all_reachable_runner()

    artifact = mod.build_artifact(repo_root=root, command_runner=runner, clock=FlatClock())
    out_path = mod.write_artifact(root, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == root / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.POLARFIRE_UPTIME_COMMAND,
    ]
    assert saved["honest_verdict"] == "success_board_continuity_matrix_written_no_speedup_claim"
    assert saved["duration_s"] == 0.0001
    assert saved["inference_substrate"] == "hardware_precheck_and_upstream_artifact_audit"
    assert saved["kv260_ssh_ready"] is True
    assert saved["kv260_prior_transcript_verified"] is True
    assert saved["kv260_speedup_claim_allowed"] is False
    assert saved["gatemate_detected"] is True
    assert saved["gatemate_terminal_state"] == (
        "gatemate_detected_toolchain_unblocked_no_carnot_tile_flashed_or_timed"
    )
    assert saved["polarfire_detected"] is True
    assert saved["polarfire_terminal_state"] == (
        "polarfire_ssh_attached_no_carnot_dispatch_executed"
    )
    assert saved["destructive_actions_taken"] == []
    assert saved["flagged_adversarial"] is False
    assert set(saved["field_principles"]) >= set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert saved["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes kria true",
            "exit_code": 0,
            "duration_s": 0.2,
            "observed": "returncode=0",
            "discipline": "ssh_only_no_host_sd_card",
        },
        {
            "resource": "gatemate_dirtyjtag_detect",
            "available": True,
            "command": "openFPGALoader -c dirtyJtag --detect",
            "exit_code": 0,
            "duration_s": 0.3,
            "observed": "IDCode : 0x20000001 colognechip GateMate GM1Ax",
            "discipline": "detect_only_no_flash_no_program",
        },
        {
            "resource": "polarfire_ssh",
            "available": True,
            "command": "ssh -o ConnectTimeout=5 -o BatchMode=yes polarfire true",
            "exit_code": 0,
            "duration_s": 0.4,
            "observed": "returncode=0",
            "discipline": "ssh_reachability_only_no_dispatch",
        },
    ]
    kv260 = saved["board_matrix"]["kv260"]
    assert kv260["terminal_state"] == "kv260_ssh_ready_prior_transcript_verified_no_speedup_claim"
    assert kv260["evidence"]["cpu_board_parity"] == "match"
    assert kv260["evidence"]["timing_packet_present"] is True
    assert kv260["evidence"]["transcript_sha256_verified"] is True
    assert "no_general_fpga_speedup_claim" in kv260["limitations"]
    assert saved["board_matrix"]["gatemate"]["detected"] is True
    assert saved["board_matrix"]["gatemate"]["destructive_actions_taken"] == []
    assert saved["board_matrix"]["polarfire"]["state_probe"]["captured"] is True
    assert "up 8 days" in saved["board_matrix"]["polarfire"]["state_probe"]["observed"]
    assert "mmcblk" not in json.dumps(saved).lower()
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    mod.validate_artifact(saved)


def test_req_hw_5079_blocked_boards_remain_visible_without_state_commands(
    tmp_path: Path,
) -> None:
    """REQ-HW-5079: blocked boards do not trigger destructive or state commands."""

    root = _fixture_root(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_PRECONDITION_COMMAND: [
                _probe(
                    mod.KV260_PRECONDITION_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host kria port 22: timeout\n",
                    duration_s=5.0,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                    duration_s=0.2,
                )
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(
                    mod.POLARFIRE_PRECONDITION_COMMAND,
                    exit_code=255,
                    stderr="ssh: connect to host polarfire port 22: timeout\n",
                    duration_s=5.0,
                )
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=root, command_runner=runner, clock=FlatClock())

    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
    ]
    assert artifact["kv260_ssh_ready"] is False
    assert artifact["kv260_prior_transcript_verified"] is True
    assert artifact["gatemate_detected"] is False
    assert artifact["polarfire_detected"] is False
    assert artifact["board_matrix"]["kv260"]["terminal_state"] == (
        "blocked_kv260_ssh_unreachable_prior_transcript_verified"
    )
    assert artifact["gatemate_terminal_state"] == "blocked_gatemate_usb_undetected"
    assert artifact["polarfire_terminal_state"] == "blocked_polarfire_ssh_unreachable"
    assert artifact["board_matrix"]["polarfire"]["state_probe"]["captured"] is False
    assert artifact["destructive_actions_taken"] == []
    mod.validate_artifact(artifact)


def test_scenario_hw_5079_bad_prior_transcript_blocks_kv260_prior_verification(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-5079: a stale Exp 5065 transcript is visible as a KV260 limit."""

    root = _fixture_root(tmp_path)
    (root / mod.PRIOR_KV260_TRANSCRIPT_REL_PATH).write_text("stale\n", encoding="utf-8")

    artifact = mod.build_artifact(
        repo_root=root,
        command_runner=_all_reachable_runner(),
        clock=FlatClock(),
    )

    assert artifact["kv260_ssh_ready"] is True
    assert artifact["kv260_prior_transcript_verified"] is False
    assert artifact["kv260_speedup_claim_allowed"] is False
    assert artifact["board_matrix"]["kv260"]["terminal_state"] == (
        "blocked_kv260_prior_transcript_unverified_no_speedup_claim"
    )
    assert artifact["kv260_prior_summary"]["verified"] is False
    assert artifact["kv260_prior_summary"]["errors"]
    mod.validate_artifact(artifact)


def test_scenario_hw_5079_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-HW-5079: run_experiment writes the requested result artifact."""

    root = _fixture_root(tmp_path)
    out_path = mod.run_experiment(
        repo_root=root,
        command_runner=_all_reachable_runner(),
        clock=FlatClock(),
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == root / mod.OUTPUT_REL_PATH
    assert artifact["experiment_id"] == 5079
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_hw_5079_validation_rejects_overclaim_and_destructive_drift(
    tmp_path: Path,
) -> None:
    """REQ-HW-5079: validation rejects speedup claims, flash actions, and stale hashes."""

    artifact = mod.build_artifact(
        repo_root=_fixture_root(tmp_path),
        command_runner=_all_reachable_runner(),
        clock=FlatClock(),
    )

    bad_speedup = dict(artifact, kv260_speedup_claim_allowed=True)
    bad_speedup["reproducibility_checksum"] = mod.payload_checksum(bad_speedup)
    with pytest.raises(ValueError, match="speedup"):
        mod.validate_artifact(bad_speedup)

    bad_action = dict(artifact, destructive_actions_taken=["flash_gatemate"])
    bad_action["reproducibility_checksum"] = mod.payload_checksum(bad_action)
    with pytest.raises(ValueError, match="destructive"):
        mod.validate_artifact(bad_action)

    bad_storage = dict(artifact)
    bad_storage["preconditions_checked"] = [{"resource": "/dev/mmcblk0"}]
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)
