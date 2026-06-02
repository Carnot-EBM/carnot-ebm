"""Tests for Exp 3698 KV260 continuity v25.

Spec refs: REQ-HW-3698, SCENARIO-HW-3698.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot import experiment_3698_kv260_continuity_v25 as mod


class RecordingRunner:
    """Deterministic command runner that fails on any non-contract command."""

    def __init__(self, probes: dict[tuple[str, ...], mod.CommandProbe]) -> None:
        self.probes = probes
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> mod.CommandProbe:
        self.commands.append(command)
        if command not in self.probes:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command]


def _probe(
    command: tuple[str, ...],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
) -> mod.CommandProbe:
    return mod.CommandProbe(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr)


def _assert_required_fields_and_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert "principle" not in str(payload[field]).lower()


def test_req_hw_3698_spec_anchor_declares_ssh_only_eight_streak_contract() -> None:
    """REQ-HW-3698: OpenSpec declares v25 SSH-only continuity before code."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3698" in spec
    assert "SCENARIO-HW-3698" in spec
    assert "experiment_3698_kv260_continuity_v25.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "host SD-card device-node precondition is permanently retired" in spec
    assert "eight consecutive unreachable" in spec


def test_scenario_hw_3698_unreachable_records_blocked_and_eight_milestone_streak() -> None:
    """SCENARIO-HW-3698: unreachable SSH writes a blocked artifact, not a pass."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: _probe(
                mod.KV260_SSH_COMMAND,
                255,
                stderr="ssh: Could not resolve hostname kv260.local\n",
            )
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=0.75)

    assert runner.commands == [mod.KV260_SSH_COMMAND]
    assert payload["honest_verdict"] == mod.BLOCKED_VERDICT
    assert payload["inference_substrate"] == "hardware_smoke"
    assert "gguf" not in json.dumps(payload).lower()
    assert "cuda" not in json.dumps(payload).lower()
    assert payload["kv260_ssh_reachable"] is False
    assert payload["kv260_overlay_loaded"] is None
    assert payload["kv260_continuity_state"] == "blocked_ssh_unreachable"
    assert payload["consecutive_unreachable_milestones"] == 8
    assert payload["operator_action_required"] is True
    assert ".331" in payload["operator_action_item"]
    assert ".337" in payload["operator_action_item"]
    assert ".338" in payload["operator_action_item"]
    assert "8 consecutive milestones" in payload["operator_action_item"]
    assert payload["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "command": mod.command_to_string(mod.KV260_SSH_COMMAND),
            "available": False,
            "exit_code": 255,
        }
    ]
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    assert "/dev/mmcblk" not in json.dumps(payload)
    assert "/dev/disk" not in json.dumps(payload)
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3698_reachable_queries_xmutil_and_resets_current_streak() -> None:
    """SCENARIO-HW-3698: reachable SSH records xmutil overlay continuity state."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: _probe(mod.KV260_SSH_COMMAND, 0),
            mod.KV260_OVERLAY_COMMAND: _probe(
                mod.KV260_OVERLAY_COMMAND,
                0,
                stdout="kv260-dpu\n",
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=1.5)

    assert runner.commands == [mod.KV260_SSH_COMMAND, mod.KV260_OVERLAY_COMMAND]
    assert payload["honest_verdict"] == mod.REACHABLE_VERDICT
    assert payload["kv260_ssh_reachable"] is True
    assert payload["kv260_overlay_loaded"] == "kv260-dpu"
    assert payload["kv260_continuity_state"] == "reachable_overlay_state_recorded"
    assert payload["consecutive_unreachable_milestones"] == 0
    assert payload["operator_action_required"] is False
    assert payload["operator_action_item"] == "none"
    overlay_probe = payload["command_probes"]["kv260_xmutil_listapps"]
    assert overlay_probe["command"] == mod.command_to_string(mod.KV260_OVERLAY_COMMAND)
    assert overlay_probe["stdout"] == "kv260-dpu\n"
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3698_reachable_overlay_failure_still_records_reachable_board() -> None:
    """REQ-HW-3698: SSH reachability is the board precondition; xmutil is state."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: _probe(mod.KV260_SSH_COMMAND, 0),
            mod.KV260_OVERLAY_COMMAND: _probe(
                mod.KV260_OVERLAY_COMMAND,
                1,
                stderr="xmutil: listapps failed\n",
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=2.0)

    assert payload["honest_verdict"] == mod.REACHABLE_VERDICT
    assert payload["kv260_ssh_reachable"] is True
    assert payload["kv260_overlay_loaded"] is None
    assert payload["kv260_continuity_state"] == "reachable_overlay_query_failed"
    assert payload["command_probes"]["kv260_xmutil_listapps"]["exit_code"] == 1


def test_req_hw_3698_field_principles_match_required_why_annotations() -> None:
    """REQ-HW-3698: required fields store values and principles store the why."""
    assert mod.FIELD_PRINCIPLES == {
        "honest_verdict": "Terminal prefix for reconciler classification.",
        "inference_substrate": "SSH-attached board test; per-board duration floor.",
        "preconditions_checked": (
            "Records the SSH-reachability check -- the correct KV260 precondition, "
            "not host SD card."
        ),
        "kv260_ssh_reachable": (
            "The honest board state; an unreachable board is a blocked_*, "
            "not a fabricated pass."
        ),
        "kv260_overlay_loaded": "Continuity state if reachable.",
        "consecutive_unreachable_milestones": (
            "Tracks the .331->.337 unreachable streak so a persistent outage "
            "surfaces as an operator-action item."
        ),
        "random_seed": "Determinism precondition.",
        "reproducibility_checksum": "Drift detection.",
        "duration_s": "Plausibility floor.",
    }


def test_req_hw_3698_continuity_state_names_reachable_missing_overlay_probe() -> None:
    """REQ-HW-3698: continuity state names stay distinct from overlay values."""
    assert mod._continuity_state(True, None) == "reachable_overlay_not_checked"


def test_req_hw_3698_write_artifact_preserves_checksum_and_schema(tmp_path: Path) -> None:
    """REQ-HW-3698: result JSON has required fields, principles, and checksum."""
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: _probe(
                mod.KV260_SSH_COMMAND,
                255,
                stderr="ssh: connect to host kria port 22: timed out\n",
            )
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=3.0)

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == "exp3698"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 3.0
    assert payload["field_principles"] == mod.FIELD_PRINCIPLES
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3698_run_command_captures_output_and_os_errors() -> None:
    """REQ-HW-3698: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')")
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-3698",))
    assert missing_probe.exit_code == 127
    assert missing_probe.command == ("/definitely/missing/ssh-for-req-hw-3698",)


def test_scenario_hw_3698_script_wrapper_exists() -> None:
    """SCENARIO-HW-3698: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3698_kv260_continuity_v25.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3698_kv260_continuity_v25" in text
    assert "main" in text
