"""Tests for Exp 3676 GateMate continuity audit v23.

Spec refs: REQ-HW-3676, SCENARIO-HW-3676.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from carnot import experiment_3676_gatemate_continuity_audit_v23 as mod


class RecordingRunner:
    """Deterministic command runner so tests can audit JTAG reachability only."""

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
    assert (
        payload["field_principles"]["openfpgaloader_installed"]
        == "Records the root blocker (tool missing) honestly so the audit trail is current."
    )
    assert (
        payload["field_principles"]["gatemate_idcode_detected"]
        == "Honest detect result (bool or null) -- not a fabricated flash pass."
    )
    assert (
        payload["field_principles"]["known_blocker"]
        == "Names the missing-tool / flash-smoke host-IO hang so the audit trail stays current."
    )


def test_req_hw_3676_spec_anchor_declares_gatemate_audit_contract() -> None:
    """REQ-HW-3676: OpenSpec declares the GateMate v23 continuity audit."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3676" in spec
    assert "SCENARIO-HW-3676" in spec
    assert "experiment_3676_gatemate_continuity_audit_v23.json" in spec
    assert "command -v openFPGALoader" in spec
    assert "timeout --kill-after=2s 5s openFPGALoader -c dirtyJtag --detect" in spec
    assert mod.TERMINAL_VERDICT in spec


def test_scenario_hw_3676_missing_openfpgaloader_stops_before_detect() -> None:
    """SCENARIO-HW-3676: missing openFPGALoader records null detect and stops."""
    runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                1,
                stderr="not found\n",
            )
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=0.7)

    assert runner.commands == [mod.OPENFPGALOADER_WHICH_COMMAND]
    assert payload["honest_verdict"] == mod.TERMINAL_VERDICT
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["openfpgaloader_installed"] is False
    assert payload["gatemate_idcode_detected"] is None
    assert payload["known_blocker"] == "openFPGALoader not found on PATH; flash/smoke host-IO hang"
    assert payload["command_probes"]["gatemate_jtag_detect"] is None
    _assert_required_fields_and_principles(payload)


def test_scenario_hw_3676_detect_success_records_idcode_without_flash() -> None:
    """SCENARIO-HW-3676: successful JTAG detect is recorded without flash smoke."""
    detect_stdout = (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )
    runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                0,
                stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                0,
                stdout=detect_stdout,
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=1.2)

    assert runner.commands == [mod.OPENFPGALOADER_WHICH_COMMAND, mod.GATEMATE_DETECT_COMMAND]
    assert all("flash" not in " ".join(command).lower() for command in runner.commands)
    assert payload["openfpgaloader_installed"] is True
    assert payload["openfpgaloader_path"] == "/opt/oss-cad-suite/bin/openFPGALoader"
    assert payload["gatemate_idcode_detected"] is True
    assert payload["known_blocker"] == "flash/smoke host-IO hang"
    detect_probe = payload["command_probes"]["gatemate_jtag_detect"]
    assert detect_probe["stdout"] == detect_stdout
    _assert_required_fields_and_principles(payload)


def test_req_hw_3676_detect_failure_records_false_and_blocker() -> None:
    """REQ-HW-3676: failed JTAG detect is a blocked value, not a fabricated pass."""
    runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                0,
                stdout="/usr/bin/openFPGALoader\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                1,
                stderr="fails to open device\n",
            ),
        }
    )

    payload = mod.build_artifact(command_runner=runner, duration_s=1.5)

    assert payload["openfpgaloader_installed"] is True
    assert payload["gatemate_idcode_detected"] is False
    assert payload["known_blocker"] == (
        "openFPGALoader detect failed exit_code=1; flash/smoke host-IO hang"
    )
    assert payload["command_probes"]["gatemate_jtag_detect"]["stderr"] == "fails to open device\n"


def test_req_hw_3676_detect_timeout_and_no_idcode_are_honest_failures() -> None:
    """REQ-HW-3676: timeout and non-IDCODE output remain honest failed detects."""
    timeout_runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                0,
                stdout="/usr/bin/openFPGALoader\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                124,
                stderr="timed out\n",
            ),
        }
    )
    timeout_payload = mod.build_artifact(command_runner=timeout_runner, duration_s=1.6)
    assert timeout_payload["gatemate_idcode_detected"] is False
    assert timeout_payload["known_blocker"] == (
        "openFPGALoader detect timed out; flash/smoke host-IO hang"
    )

    odd_runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                0,
                stdout="/usr/bin/openFPGALoader\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                0,
                stdout="JTAG chain found but no model text\n",
            ),
        }
    )
    odd_payload = mod.build_artifact(command_runner=odd_runner, duration_s=1.7)
    assert odd_payload["gatemate_idcode_detected"] is False
    assert odd_payload["known_blocker"] == (
        "openFPGALoader detect did not report a GateMate IDCODE; flash/smoke host-IO hang"
    )


def test_req_hw_3676_write_artifact_preserves_schema_and_checksum(tmp_path: Path) -> None:
    """REQ-HW-3676: result JSON has required fields, principles, and checksum."""
    runner = RecordingRunner(
        {
            mod.OPENFPGALOADER_WHICH_COMMAND: _probe(
                mod.OPENFPGALOADER_WHICH_COMMAND,
                0,
                stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
            ),
            mod.GATEMATE_DETECT_COMMAND: _probe(
                mod.GATEMATE_DETECT_COMMAND,
                0,
                stdout="idcode 0x20000001\nfamily GateMate Series\n",
            ),
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=2.0)

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == "exp3676"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 2.0
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3676_run_command_captures_output_and_exec_errors() -> None:
    """REQ-HW-3676: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')")
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    missing_probe = mod.run_command(("/definitely/missing/openfpgaloader-for-req-hw-3676",))
    assert missing_probe.exit_code == 127
    assert "No such file" in missing_probe.stderr


def test_scenario_hw_3676_script_wrapper_exists() -> None:
    """SCENARIO-HW-3676: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3676_gatemate_continuity_audit_v23.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3676_gatemate_continuity_audit_v23" in text
    assert "main" in text
