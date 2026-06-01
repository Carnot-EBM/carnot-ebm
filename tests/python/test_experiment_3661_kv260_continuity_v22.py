"""Tests for Exp 3661 KV260 continuity v22.

Spec refs: REQ-HW-3661, SCENARIO-HW-3661.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot import experiment_3661_kv260_continuity_v22 as mod


class RecordingRunner:
    """Deterministic command runner that exposes accidental non-SSH checks."""

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


def test_req_hw_3661_spec_anchor_declares_ssh_only_five_streak_contract() -> None:
    """REQ-HW-3661: OpenSpec declares v22 SSH-only continuity before code."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3661" in spec
    assert "SCENARIO-HW-3661" in spec
    assert "experiment_3661_kv260_continuity_v22.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "host SD-card device-node precondition is permanently retired" in spec
    assert "five consecutive unreachable milestones" in spec


def test_scenario_hw_3661_unreachable_records_blocked_and_five_milestone_streak() -> None:
    """SCENARIO-HW-3661: unreachable SSH writes a blocked artifact, not a pass."""
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
    assert payload["kv260_ssh_reachable"] is False
    assert payload["kv260_overlay_loaded"] is None
    assert payload["consecutive_unreachable_milestones"] == 5
    assert payload["operator_action_required"] is True
    assert ".331" in payload["operator_action_item"]
    assert ".335" in payload["operator_action_item"]
    assert payload["preconditions_checked"] == [
        {
            "resource": "kv260_ssh",
            "command": mod.command_to_string(mod.KV260_SSH_COMMAND),
            "available": False,
            "exit_code": 255,
        }
    ]
    assert payload["command_probes"]["kv260_xmutil_listapps"] is None
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3661_reachable_queries_xmutil_and_resets_current_streak() -> None:
    """SCENARIO-HW-3661: reachable SSH records xmutil overlay continuity state."""
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
    assert payload["consecutive_unreachable_milestones"] == 0
    assert payload["operator_action_required"] is False
    assert payload["operator_action_item"] == "none"
    overlay_probe = payload["command_probes"]["kv260_xmutil_listapps"]
    assert overlay_probe["command"] == mod.command_to_string(mod.KV260_OVERLAY_COMMAND)
    assert overlay_probe["stdout"] == "kv260-dpu\n"
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3661_reachable_overlay_failure_still_records_reachable_board() -> None:
    """REQ-HW-3661: SSH reachability is the board precondition; xmutil is state."""
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
    assert payload["command_probes"]["kv260_xmutil_listapps"]["exit_code"] == 1


def test_req_hw_3661_continuity_state_names_reachable_missing_overlay_probe() -> None:
    """REQ-HW-3661: continuity state names stay distinct from overlay values."""
    assert mod._continuity_state(True, None) == "reachable_overlay_not_checked"


def test_req_hw_3661_write_artifact_preserves_checksum_and_schema(tmp_path: Path) -> None:
    """REQ-HW-3661: result JSON has required fields, principles, and checksum."""
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
    assert payload["experiment_id"] == "exp3661"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 3.0
    assert payload["field_principles"]["preconditions_checked"].startswith(
        "Records the SSH-reachability check"
    )
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3661_run_command_captures_output_and_os_errors() -> None:
    """REQ-HW-3661: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')")
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-3661",))
    assert missing_probe.exit_code == 127
    assert "No such file" in missing_probe.stderr


def test_scenario_hw_3661_script_wrapper_exists() -> None:
    """SCENARIO-HW-3661: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3661_kv260_continuity_v22.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3661_kv260_continuity_v22" in text
    assert "main" in text
