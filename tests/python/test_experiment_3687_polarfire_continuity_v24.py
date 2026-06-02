"""Tests for Exp 3687 PolarFire continuity v24.

Spec refs: REQ-HW-3687, SCENARIO-HW-3687.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot import experiment_3687_polarfire_continuity_v24 as mod


class RecordingRunner:
    """Deterministic command runner so tests can describe board states without SSH."""

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
        assert "(principle:" not in str(payload[field])


def test_req_hw_3687_spec_anchor_declares_reachable_continuity_contract() -> None:
    """REQ-HW-3687: OpenSpec declares the PolarFire v24 continuity artifact."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3687" in spec
    assert "SCENARIO-HW-3687" in spec
    assert "experiment_3687_polarfire_continuity_v24.json" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "complete: polarfire_continuity_confirmed_reachable" in spec
    assert "complete: blocked_polarfire_ssh_timeout" in spec


def test_scenario_hw_3687_unreachable_stops_after_precondition() -> None:
    """SCENARIO-HW-3687: failed SSH precondition records blocked board state."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_COMMAND: _probe(
                mod.POLARFIRE_SSH_COMMAND,
                255,
                stderr="ssh: connect to host polarfire port 22: timed out\n",
            )
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=0.9)

    assert runner.commands == [mod.POLARFIRE_SSH_COMMAND]
    assert payload["honest_verdict"] == mod.BLOCKED_VERDICT
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["polarfire_ssh_reachable"] is False
    assert payload["polarfire_uptime"] is None
    assert payload["polarfire_carnot_dispatch_path"] is None
    assert payload["polarfire_continuity_state"] == "blocked_ssh_timeout"
    assert payload["preconditions_checked"] == [
        {
            "resource": "polarfire_ssh",
            "command": mod.command_to_string(mod.POLARFIRE_SSH_COMMAND),
            "available": False,
            "exit_code": 255,
        }
    ]
    assert payload["command_probes"]["polarfire_uptime"] is None
    assert payload["command_probes"]["polarfire_carnot_dispatch_path"] is None
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3687_reachable_records_deflagged_uptime_and_dispatch_path() -> None:
    """SCENARIO-HW-3687: reachable SSH records live continuity values."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_COMMAND: _probe(mod.POLARFIRE_SSH_COMMAND, 0),
            mod.POLARFIRE_UPTIME_COMMAND: _probe(
                mod.POLARFIRE_UPTIME_COMMAND,
                0,
                stdout="02:57:09 up 3 days, load average: 0.01, 0.00, 0.00\n",
            ),
            mod.POLARFIRE_DISPATCH_COMMAND: _probe(
                mod.POLARFIRE_DISPATCH_COMMAND,
                0,
                stdout="/usr/bin/carnot\n",
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=1.4)

    assert runner.commands == [
        mod.POLARFIRE_SSH_COMMAND,
        mod.POLARFIRE_UPTIME_COMMAND,
        mod.POLARFIRE_DISPATCH_COMMAND,
    ]
    assert payload["honest_verdict"] == mod.REACHABLE_VERDICT
    assert payload["polarfire_ssh_reachable"] is True
    assert payload["polarfire_uptime"] == "02:57:09 up 3 days, load average: 0.01, 0.00, 0.00"
    assert payload["polarfire_carnot_dispatch_path"] == "/usr/bin/carnot"
    assert payload["polarfire_continuity_state"] == (
        "reachable_uptime_and_dispatch_path_recorded_deflagged"
    )
    assert not any("flag" in key for key in payload if key.startswith("polarfire_"))
    dispatch_probe = payload["command_probes"]["polarfire_carnot_dispatch_path"]
    assert dispatch_probe["command"] == mod.command_to_string(mod.POLARFIRE_DISPATCH_COMMAND)
    assert dispatch_probe["stdout"] == "/usr/bin/carnot\n"
    _assert_required_fields_and_principles(payload)


def test_req_hw_3687_reachable_probe_failures_are_recorded_as_values() -> None:
    """REQ-HW-3687: reachable SSH still stores uptime/path values if probes fail."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_COMMAND: _probe(mod.POLARFIRE_SSH_COMMAND, 0),
            mod.POLARFIRE_UPTIME_COMMAND: _probe(
                mod.POLARFIRE_UPTIME_COMMAND,
                1,
                stderr="uptime: failed\n",
            ),
            mod.POLARFIRE_DISPATCH_COMMAND: _probe(
                mod.POLARFIRE_DISPATCH_COMMAND,
                1,
                stderr="which: no carnot in path\n",
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=2.0)

    assert payload["honest_verdict"] == mod.REACHABLE_VERDICT
    assert payload["polarfire_ssh_reachable"] is True
    assert payload["polarfire_uptime"] == "unknown"
    assert payload["polarfire_carnot_dispatch_path"] == "not_found"
    assert payload["polarfire_continuity_state"] == "reachable_probe_values_incomplete"
    assert payload["command_probes"]["polarfire_uptime"]["exit_code"] == 1
    assert payload["command_probes"]["polarfire_carnot_dispatch_path"]["exit_code"] == 1


def test_req_hw_3687_write_artifact_preserves_schema_and_checksum(tmp_path: Path) -> None:
    """REQ-HW-3687: result JSON has required fields, principles, and checksum."""
    runner = RecordingRunner(
        {
            mod.POLARFIRE_SSH_COMMAND: _probe(mod.POLARFIRE_SSH_COMMAND, 0),
            mod.POLARFIRE_UPTIME_COMMAND: _probe(
                mod.POLARFIRE_UPTIME_COMMAND,
                0,
                stdout="up 10 days\n",
            ),
            mod.POLARFIRE_DISPATCH_COMMAND: _probe(
                mod.POLARFIRE_DISPATCH_COMMAND,
                0,
                stdout="/opt/carnot/bin/carnot\n",
            ),
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=3.0)

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == "exp3687"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 3.0
    assert payload["field_principles"]["honest_verdict"] == (
        "Terminal prefix for reconciler classification."
    )
    assert payload["field_principles"]["inference_substrate"] == (
        "SSH-attached board test."
    )
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3687_run_command_captures_output_and_exec_errors() -> None:
    """REQ-HW-3687: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')")
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-3687",))
    assert missing_probe.exit_code == 127
    assert "No such file" in missing_probe.stderr


def test_scenario_hw_3687_script_wrapper_exists() -> None:
    """SCENARIO-HW-3687: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3687_polarfire_continuity_v24.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3687_polarfire_continuity_v24" in text
    assert "main" in text
