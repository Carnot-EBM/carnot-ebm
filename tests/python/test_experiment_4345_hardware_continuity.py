"""Tests for Exp 4345 opportunistic hardware continuity.

Spec refs: REQ-HW-4345, SCENARIO-HW-4345.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4345_hardware_continuity as mod


class RecordingRunner:
    """SCENARIO-HW-4345 command runner with queued board transcripts."""

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


def _write_previous_4334(repo_root: Path) -> None:
    results = repo_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "experiment_4334_hardware_continuity.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: hardware_continuity_4334_prior",
                "kv260_reachable": True,
                "per_board_reachability": {
                    "kv260": True,
                    "polarfire": True,
                    "gatemate": False,
                },
            }
        ),
        encoding="utf-8",
    )


def test_req_hw_4345_spec_entry_declares_required_artifact_contract() -> None:
    """REQ-HW-4345: OpenSpec anchors fields, commands, and blocked fallback."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-4345" in spec
    assert "SCENARIO-HW-4345" in spec
    assert "experiment_4345_hardware_continuity.json" in spec
    assert "experiment_4334_hardware_continuity.json" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "openFPGALoader -c dirtyJtag --detect" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "blocked_kv260_ssh_unreachable" in spec
    assert "Host SD-card block-device checks" in spec


def test_scenario_hw_4345_records_reachable_kv260_and_opportunistic_boards(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-4345: reachable KV260 emits board list and transcript."""
    _write_previous_4334(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(
                    mod.KV260_LISTAPPS_COMMAND,
                    stdout="app: carnot_ising_v2_n64 status: loadable\n",
                    duration_s=0.3,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="IDCode : 0x20000001 colognechip GateMate GM1Ax\n",
                    duration_s=0.4,
                )
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, duration_s=0.5)
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
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_SSH_PRECONDITION,
    ]
    assert saved["schema"] == mod.SCHEMA
    assert saved["experiment"] == mod.EXPERIMENT_ID
    assert saved["spec_refs"] == mod.SPEC_REFS
    assert saved["source_context"]["previous_experiment"] == 4334
    assert saved["source_context"]["previous_artifact_read"] is True
    assert saved["kv260_reachable"] is True
    assert saved["boards_probed"] == [
        {"name": "kv260", "reachable": True, "state": "kv260_carnot_ising_listapps_seen"},
        {"name": "gatemate", "reachable": True, "state": "gatemate_idcode_0x20000001"},
        {"name": "polarfire", "reachable": True, "state": "polarfire_ssh_reachable"},
    ]
    assert [entry["resource"] for entry in saved["preconditions_checked"]] == [
        "kv260_ssh",
        "gatemate_jtag_detect",
        "polarfire_ssh",
    ]
    assert [entry["stage"] for entry in saved["board_state_transcript"]] == [
        "kv260_ssh_precondition",
        "kv260_xmutil_listapps",
        "gatemate_dirtyjtag_detect",
        "polarfire_ssh_precondition",
    ]
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert saved["field_principles"][field] == mod.FIELD_PRINCIPLES[field]
    assert saved["honest_verdict"].startswith("complete: hardware_continuity_4345_")
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "mmcblk" not in json.dumps(saved).lower()
    mod.validate_artifact(saved)


def test_req_hw_4345_blocks_without_secondary_probes_when_kv260_ssh_fails(
    tmp_path: Path,
) -> None:
    """REQ-HW-4345: KV260 SSH failure is an honest blocked non-fabrication."""
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
    assert saved["boards_probed"] == [
        {"name": "kv260", "reachable": False, "state": "blocked_kv260_ssh_unreachable"}
    ]
    assert saved["preconditions_checked"][0]["resource"] == "kv260_ssh"
    assert saved["preconditions_checked"][0]["available"] is False
    assert saved["source_context"]["previous_artifact_read"] is False
    assert "gatemate_jtag_detect" not in json.dumps(saved)
    assert "polarfire_ssh" not in json.dumps(saved)
    mod.validate_artifact(saved)


def test_req_hw_4345_records_reachable_kv260_with_opportunistic_blocks(
    tmp_path: Path,
) -> None:
    """REQ-HW-4345: reachable KV260 can coexist with blocked optional boards."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_PRECONDITION: [_probe(mod.KV260_SSH_PRECONDITION, duration_s=0.2)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 1, stderr="xmutil needs sudo", duration_s=0.3)
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout="no idcode\n", duration_s=0.4)
            ],
            mod.POLARFIRE_SSH_PRECONDITION: [
                _probe(mod.POLARFIRE_SSH_PRECONDITION, 255, stderr="timeout", duration_s=0.5)
            ],
        }
    )

    artifact = mod.build_artifact(repo_root=tmp_path, command_runner=runner)

    assert artifact["kv260_reachable"] is True
    assert artifact["boards_probed"] == [
        {
            "name": "kv260",
            "reachable": True,
            "state": "kv260_xmutil_listapps_blocked_returncode_1",
        },
        {"name": "gatemate", "reachable": False, "state": "blocked_gatemate_unreachable"},
        {"name": "polarfire", "reachable": False, "state": "blocked_polarfire_unreachable"},
    ]
    assert artifact["honest_verdict"].startswith("complete: hardware_continuity_4345_")
    mod.validate_artifact(artifact)


def test_req_hw_4345_required_runner_lives_under_results() -> None:
    """REQ-HW-4345: the required `python3 results/...` runner is present."""
    runner = Path("results/experiment_4345_hardware_continuity.py")

    assert runner.exists()
    text = runner.read_text(encoding="utf-8")
    assert "experiment_4345_hardware_continuity" in text
    assert "run_experiment" in text
