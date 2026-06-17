"""Tests for Exp 4334 hardware continuity.

Spec refs: REQ-HW-4334, SCENARIO-HW-4334.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4334_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4334 command runner with queued board transcripts."""

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


def _write_previous_4322(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4322_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": (
                    "complete: hardware_continuity_4322_kv260_terminal_"
                    "polarfire_hash_verified_gatemate_blocked"
                ),
                "per_board_reachability": {
                    "kv260": True,
                    "polarfire": True,
                    "gatemate": False,
                },
                "per_board_status": {
                    "kv260": {"status": ("kv260_terminal_xmutil_listapps_blocked_returncode_1")},
                    "polarfire": {"status": "polarfire_hash_verified_cpu_dispatch_succeeded"},
                    "gatemate": {"status": "blocked_gatemate_unreachable"},
                },
                "kv260_step_taken": ("kv260_terminal_xmutil_listapps_blocked_returncode_1"),
                "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
                "gatemate_step_taken": "blocked_gatemate_unreachable",
            }
        ),
        encoding="utf-8",
    )


def _polar_step(**_: Any) -> mod.StepOutcome:
    return mod.StepOutcome(
        step_taken="polarfire_hash_verified_cpu_dispatch_succeeded",
        terminal_state="reachable_hash_verified_cpu_dispatch_recorded",
        success=True,
        duration_s=0.67,
        details={
            "step": "hash_verified_cpu_dispatch_smoke",
            "result_hash_match": True,
            "board_result_sha256": "abc123",
            "cpu_reference_sha256": "abc123",
            "next_concrete_step": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    )


def test_req_hw_4334_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4334: OpenSpec anchors fields, principles, and preconditions."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4334" in spec
    assert "SCENARIO-HW-4334" in spec
    assert "experiment_4334_hardware_continuity.json" in spec
    assert "experiment_4322_hardware_continuity.json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "0x20000001" in spec
    assert "GM1Ax" in spec
    assert "host SD-card block-device checks" in spec


def test_scenario_hw_4334_records_reachable_board_opportunistic_results(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4334: reachable boards get SSH/USB status plus smoke details."""
    _write_previous_4322(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.21)],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.55)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.34,
                )
            ],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout="app: carnot_ising_v4 status: loaded\n",
                    duration_s=0.11,
                )
            ],
        }
    )

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_step_runner=_polar_step,
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_SSH_PRECONDITION,
        mod.POLARFIRE_SSH_PRECONDITION,
        mod.GATEMATE_DETECT_COMMAND,
        mod.KV260_LISTAPPS_COMMAND,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["random_seed"] == mod.RANDOM_SEED
    assert saved["source_context"]["previous_experiment"] == 4322
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["source_context"]["previous_kv260_step_taken"] == (
        "kv260_terminal_xmutil_listapps_blocked_returncode_1"
    )
    assert saved["source_context"]["most_recent_hardware_continuity_artifact"].endswith(
        "experiment_4322_hardware_continuity.json"
    )
    assert saved["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert saved["preconditions_checked"][1]["resource"] == "polarfire_ssh"
    assert saved["preconditions_checked"][2]["resource"] == "gatemate_jtag_detect"
    timer_ids = [saved["per_board_status"][board]["timer_id"] for board in mod.BOARD_NAMES]
    assert len(set(timer_ids)) == 3
    assert all(
        timer_id.endswith("_precondition_plus_opportunistic_check_wall_clock")
        for timer_id in timer_ids
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert saved["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
    assert saved["per_board_status"]["kv260"]["status"] == "kv260_terminal_confirmed_ssh_only"
    assert saved["per_board_status"]["kv260"]["ssh_only_terminal_status"] == (
        "terminal_confirmed_via_xmutil_listapps_ssh_only"
    )
    assert saved["per_board_status"]["polarfire"]["status"] == (
        "polarfire_hash_verified_cpu_dispatch_succeeded"
    )
    assert saved["per_board_status"]["polarfire"]["hash_match"] is True
    assert saved["per_board_status"]["gatemate"]["status"] == "gatemate_idcode_detected"
    assert saved["per_board_status"]["gatemate"]["idcode"] == "0x20000001"
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_4334_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4334_run_experiment_writes_xmutil_blocked_status(tmp_path: Path) -> None:
    """REQ-HW-4334: runner writes honest SSH-only KV260 listapps failures."""
    _write_previous_4322(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(
                    mod.POLARFIRE_SSH_PRECONDITION,
                    255,
                    stderr="timeout",
                    duration_s=0.4,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="no idcode\n", duration_s=0.3)
            ],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    1,
                    stderr="xmutil unavailable",
                    duration_s=0.1,
                )
            ],
        }
    )

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=runner,
        polarfire_step_runner=_polar_step,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["per_board_status"]["kv260"]["status"] == (
        "kv260_terminal_xmutil_listapps_blocked_returncode_1"
    )
    assert saved["per_board_status"]["kv260"]["ssh_only_terminal_status"] == (
        "ssh_reachable_xmutil_listapps_blocked_ssh_only"
    )
    assert saved["per_board_status"]["polarfire"]["status"] == "blocked_polarfire_unreachable"
    assert saved["per_board_status"]["gatemate"]["status"] == "blocked_gatemate_unreachable"
    mod.validate_artifact(saved)


def test_req_hw_4334_required_runner_lives_under_results() -> None:
    """REQ-HW-4334: the required `python3 results/...` runner is present."""
    runner = Path("results/experiment_4334_hardware_continuity.py")

    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "experiment_4334_hardware_continuity" in text
    assert "run_experiment" in text
