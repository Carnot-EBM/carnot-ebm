"""Tests for Exp 3721 consolidated hardware continuity.

Spec refs: REQ-HW-3721, SCENARIO-HW-3721.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from carnot import experiment_3721_hardware_kv260_terminal_confirm_and_continuity as mod


class RecordingRunner:
    """Synthetic command runner so tests describe board states without network IO."""

    def __init__(self, probes: dict[tuple[str, ...], list[mod.CommandProbe]]) -> None:
        self.probes = {command: list(values) for command, values in probes.items()}
        self.commands: list[tuple[str, ...]] = []
        self.timeouts: list[float] = []

    def __call__(
        self,
        command: tuple[str, ...],
        timeout_s: float = 60.0,
    ) -> mod.CommandProbe:
        self.commands.append(command)
        self.timeouts.append(timeout_s)
        if command not in self.probes or not self.probes[command]:
            raise AssertionError(f"unexpected command: {command!r}")
        return self.probes[command].pop(0)


def _probe(
    command: tuple[str, ...],
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
    )


def _write_exp3709_transcript(
    repo_root: Path,
    *,
    sample_count: int = 32,
    include_harness_evidence: bool = True,
    positive_samples: bool = True,
    terminal_condition_met: bool = True,
    inference_substrate: str = "hardware_smoke",
    median_ms: float | None = None,
) -> Path:
    samples = [round(0.025 + 0.0001 * (idx % 5), 6) for idx in range(sample_count)]
    if samples and not positive_samples:
        samples[2] = 0.0
    harness_payload = {
        "schema": "carnot.kv260.remote_latency_harness.v1",
        "sample_count": sample_count,
        "per_sample_wall_ms": samples,
    }
    harness_stdout = "plain logs without timing JSON\n"
    if include_harness_evidence:
        harness_stdout = (
            "BOARD_HARNESS_START exp3709\n"
            + json.dumps(harness_payload, sort_keys=True)
            + "\n"
        )
    payload = {
        "schema": "carnot.kv260_terminal_latency_transcript.v1",
        "experiment_id": "exp3709",
        "inference_substrate": inference_substrate,
        "terminal_condition_met": terminal_condition_met,
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
        "board_latency_samples": samples,
        "board_latency_median_ms": (
            median_ms if median_ms is not None else samples[len(samples) // 2]
        )
        if samples
        else None,
        "command_probes": {
            "kv260_latency_harness": {
                "command": "ssh kria 'sudo python3 -'",
                "exit_code": 0,
                "stdout": harness_stdout,
                "stderr": "",
                "combined_output": harness_stdout,
            }
        },
    }
    out_path = repo_root / mod.EXP3709_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _assert_required_fields_and_principles(payload: dict[str, object]) -> None:
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]
        assert "principle" not in str(payload[field]).lower()


def test_req_hw_3721_spec_anchor_declares_consolidated_hardware_contract() -> None:
    """REQ-HW-3721: OpenSpec declares the consolidated hardware audit."""
    spec = Path("openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")
    assert "REQ-HW-3721" in spec
    assert "SCENARIO-HW-3721" in spec
    assert "experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json" in spec
    assert "ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'" in spec
    assert "ssh kria 'xmutil listapps'" in spec
    assert "experiment_3709_kv260_drive_to_terminal_latency_transcript.json" in spec
    assert "ssh -o ConnectTimeout=5 polarfire 'true'" in spec
    assert "command -v openFPGALoader" in spec
    assert "host SD-card device-node precondition is permanently retired" in spec


@pytest.mark.parametrize(
    ("outcome", "transcript_kwargs", "probes", "expected"),
    [
        pytest.param(
            "kv260_terminal_confirmed",
            {},
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(
                        mod.KV260_LISTAPPS_COMMAND,
                        1,
                        stderr=(
                            "xmutil should be called with root privileges. "
                            "Please try again using 'sudo'.\n"
                        ),
                    )
                ],
                mod.KV260_LISTAPPS_SUDO_COMMAND: [
                    _probe(
                        mod.KV260_LISTAPPS_SUDO_COMMAND,
                        0,
                        stdout="carnot_ising_v4\ncarnot_ising_v2_n64\n",
                    )
                ],
                mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, 0)],
                mod.POLARFIRE_UPTIME_COMMAND: [
                    _probe(mod.POLARFIRE_UPTIME_COMMAND, 0, stdout="up 3 days\n")
                ],
                mod.POLARFIRE_DISPATCH_COMMAND: [
                    _probe(mod.POLARFIRE_DISPATCH_COMMAND, 0, stdout="/usr/bin/carnot\n")
                ],
                mod.GATEMATE_OPENFPGALOADER_COMMAND: [
                    _probe(
                        mod.GATEMATE_OPENFPGALOADER_COMMAND,
                        0,
                        stdout="/opt/oss-cad-suite/bin/openFPGALoader\n",
                    )
                ],
            },
            {
                "verdict": mod.KV260_TERMINAL_VERDICT,
                "terminal": True,
                "kv260_ssh": True,
                "overlay": True,
                "transcript_present": True,
                "recommendation": mod.MANDATE_LIFT_RECOMMENDATION,
                "commands": [
                    mod.KV260_SSH_COMMAND,
                    mod.KV260_LISTAPPS_COMMAND,
                    mod.KV260_LISTAPPS_SUDO_COMMAND,
                    mod.POLARFIRE_SSH_COMMAND,
                    mod.POLARFIRE_UPTIME_COMMAND,
                    mod.POLARFIRE_DISPATCH_COMMAND,
                    mod.GATEMATE_OPENFPGALOADER_COMMAND,
                ],
            },
            id="kv260_terminal_confirmed",
        ),
        pytest.param(
            "kv260_unreachable",
            {},
            {
                mod.KV260_SSH_COMMAND: [
                    _probe(
                        mod.KV260_SSH_COMMAND,
                        255,
                        stderr="ssh: connect to host kria port 22: timed out\n",
                    )
                ],
                mod.POLARFIRE_SSH_COMMAND: [
                    _probe(
                        mod.POLARFIRE_SSH_COMMAND,
                        255,
                        stderr="ssh: connect to host polarfire port 22: timed out\n",
                    )
                ],
                mod.GATEMATE_OPENFPGALOADER_COMMAND: [
                    _probe(mod.GATEMATE_OPENFPGALOADER_COMMAND, 1, stderr="not found\n")
                ],
            },
            {
                "verdict": mod.KV260_UNREACHABLE_VERDICT,
                "terminal": False,
                "kv260_ssh": False,
                "overlay": False,
                "transcript_present": True,
                "recommendation": mod.NO_LIFT_KV260_UNREACHABLE_RECOMMENDATION,
                "commands": [
                    mod.KV260_SSH_COMMAND,
                    mod.POLARFIRE_SSH_COMMAND,
                    mod.GATEMATE_OPENFPGALOADER_COMMAND,
                ],
            },
            id="kv260_unreachable",
        ),
        pytest.param(
            "partial",
            {"sample_count": 29},
            {
                mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
                mod.KV260_LISTAPPS_COMMAND: [
                    _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="k26-starter-kits\n")
                ],
                mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, 0)],
                mod.POLARFIRE_UPTIME_COMMAND: [
                    _probe(mod.POLARFIRE_UPTIME_COMMAND, 1, stderr="uptime failed\n")
                ],
                mod.POLARFIRE_DISPATCH_COMMAND: [
                    _probe(mod.POLARFIRE_DISPATCH_COMMAND, 1, stderr="which failed\n")
                ],
                mod.GATEMATE_OPENFPGALOADER_COMMAND: [
                    _probe(
                        mod.GATEMATE_OPENFPGALOADER_COMMAND,
                        0,
                        stdout="/usr/bin/openFPGALoader\n",
                    )
                ],
            },
            {
                "verdict": mod.PARTIAL_VERDICT,
                "terminal": False,
                "kv260_ssh": True,
                "overlay": False,
                "transcript_present": False,
                "recommendation": mod.NO_LIFT_PARTIAL_RECOMMENDATION,
                "commands": [
                    mod.KV260_SSH_COMMAND,
                    mod.KV260_LISTAPPS_COMMAND,
                    mod.POLARFIRE_SSH_COMMAND,
                    mod.POLARFIRE_UPTIME_COMMAND,
                    mod.POLARFIRE_DISPATCH_COMMAND,
                    mod.GATEMATE_OPENFPGALOADER_COMMAND,
                ],
            },
            id="partial",
        ),
    ],
)
def test_scenario_hw_3721_honest_outcomes_are_parametrized(
    tmp_path: Path,
    outcome: str,
    transcript_kwargs: dict[str, object],
    probes: dict[tuple[str, ...], list[mod.CommandProbe]],
    expected: dict[str, object],
) -> None:
    """SCENARIO-HW-3721: terminal, unreachable, and partial outcomes are honest."""
    _write_exp3709_transcript(tmp_path, **transcript_kwargs)
    runner = RecordingRunner(probes)

    payload = mod.build_artifact(repo_root=tmp_path, command_runner=runner, duration_s=3.5)

    assert outcome in {"kv260_terminal_confirmed", "kv260_unreachable", "partial"}
    assert runner.commands == expected["commands"]
    assert payload["honest_verdict"] == expected["verdict"]
    assert payload["inference_substrate"] == "hardware_smoke"
    assert payload["kv260_ssh_reachable"] is expected["kv260_ssh"]
    assert payload["kv260_overlay_loaded"] is expected["overlay"]
    assert payload["kv260_terminal_transcript_present"] is expected["transcript_present"]
    assert payload["kv260_terminal_condition_confirmed"] is expected["terminal"]
    assert payload["kv260_mandate_lift_recommendation"] == expected["recommendation"]
    assert payload["speedup_claim_avoided_assert"] is True
    assert "gguf" not in json.dumps(payload).lower()
    assert "cuda" not in json.dumps(payload).lower()
    assert "/dev/mmcblk" not in json.dumps(runner.commands)
    assert "/dev/disk" not in json.dumps(runner.commands)
    assert all("--detect" not in " ".join(command) for command in runner.commands)
    assert all("flash" not in " ".join(command).lower() for command in runner.commands)
    preconditions = {entry["resource"]: entry for entry in payload["preconditions_checked"]}
    assert preconditions["kv260_ssh"]["command"] == mod.command_to_string(
        mod.KV260_SSH_COMMAND
    )
    assert preconditions["polarfire_ssh"]["command"] == mod.command_to_string(
        mod.POLARFIRE_SSH_COMMAND
    )
    assert preconditions["gatemate_openfpgaloader"]["command"] == (
        "command -v openFPGALoader"
    )
    _assert_required_fields_and_principles(payload)

    if expected["terminal"]:
        assert payload["kv260_overlay_name"] == "carnot_ising_v4"
        assert payload["kv260_terminal_transcript_sample_count"] == 32
        assert payload["polarfire_uptime"] == "up 3 days"
        assert payload["polarfire_carnot_dispatch_path"] == "/usr/bin/carnot"
        assert payload["gatemate_openfpgaloader_path"] == (
            "/opt/oss-cad-suite/bin/openFPGALoader"
        )
    if outcome == "kv260_unreachable":
        assert payload["command_probes"]["kv260_xmutil_listapps"] is None
        assert payload["polarfire_continuity_state"] == "blocked_ssh_timeout"
        assert payload["gatemate_openfpgaloader_installed"] is False
    if outcome == "partial":
        assert payload["polarfire_uptime"] == "unknown"
        assert payload["polarfire_carnot_dispatch_path"] == "not_found"
        assert payload["polarfire_continuity_state"] == "reachable_probe_values_incomplete"


def test_req_hw_3721_transcript_validation_rejects_missing_and_fabricated_inputs(
    tmp_path: Path,
) -> None:
    """REQ-HW-3721: Exp 3709 evidence needs >=30 positive real timing samples."""
    missing = mod.inspect_exp3709_terminal_transcript(tmp_path)
    assert missing["exists"] is False
    assert missing["non_fabricated"] is False
    assert "missing_exp3709_artifact" in missing["validation_reasons"]

    invalid_path = tmp_path / mod.EXP3709_REL_PATH
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_text("{not json", encoding="utf-8")
    invalid = mod.inspect_exp3709_terminal_transcript(tmp_path)
    assert invalid["non_fabricated"] is False
    assert "invalid_json" in invalid["validation_reasons"]

    _write_exp3709_transcript(tmp_path, include_harness_evidence=False)
    no_harness = mod.inspect_exp3709_terminal_transcript(tmp_path)
    assert no_harness["exists"] is True
    assert no_harness["non_fabricated"] is False
    assert "board_harness_timing_evidence_missing" in no_harness["validation_reasons"]

    _write_exp3709_transcript(tmp_path, positive_samples=False)
    nonpositive = mod.inspect_exp3709_terminal_transcript(tmp_path)
    assert nonpositive["non_fabricated"] is False
    assert "latency_samples_not_positive" in nonpositive["validation_reasons"]

    _write_exp3709_transcript(
        tmp_path,
        terminal_condition_met=False,
        inference_substrate="aggregation_from_upstream_artifacts",
        median_ms=0.0,
    )
    wrong_metadata = mod.inspect_exp3709_terminal_transcript(tmp_path)
    assert wrong_metadata["non_fabricated"] is False
    assert wrong_metadata["median_ms"] > 0.0
    assert "exp3709_terminal_condition_not_met" in wrong_metadata["validation_reasons"]
    assert (
        "exp3709_inference_substrate_not_hardware_smoke"
        in wrong_metadata["validation_reasons"]
    )


def test_req_hw_3721_run_experiment_writes_checksum_and_schema(tmp_path: Path) -> None:
    """REQ-HW-3721: result JSON has required fields, principles, and checksum."""
    _write_exp3709_transcript(tmp_path)
    runner = RecordingRunner(
        {
            mod.KV260_SSH_COMMAND: [_probe(mod.KV260_SSH_COMMAND, 0)],
            mod.KV260_LISTAPPS_COMMAND: [
                _probe(mod.KV260_LISTAPPS_COMMAND, 0, stdout="carnot_ising_v2_n64\n")
            ],
            mod.POLARFIRE_SSH_COMMAND: [_probe(mod.POLARFIRE_SSH_COMMAND, 0)],
            mod.POLARFIRE_UPTIME_COMMAND: [
                _probe(mod.POLARFIRE_UPTIME_COMMAND, 0, stdout="up 10 days\n")
            ],
            mod.POLARFIRE_DISPATCH_COMMAND: [
                _probe(mod.POLARFIRE_DISPATCH_COMMAND, 0, stdout="/opt/carnot/bin/carnot\n")
            ],
            mod.GATEMATE_OPENFPGALOADER_COMMAND: [
                _probe(mod.GATEMATE_OPENFPGALOADER_COMMAND, 1, stderr="not found\n")
            ],
        }
    )

    out_path = mod.run_experiment(repo_root=tmp_path, command_runner=runner, duration_s=4.0)

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == mod.SCHEMA
    assert payload["experiment_id"] == "exp3721"
    assert payload["random_seed"] == mod.RANDOM_SEED
    assert payload["duration_s"] == 4.0
    assert payload["field_principles"] == mod.FIELD_PRINCIPLES
    checksum_payload = dict(payload)
    checksum_payload.pop("reproducibility_checksum")
    assert payload["reproducibility_checksum"] == mod.sha256_payload(checksum_payload)
    _assert_required_fields_and_principles(payload)


def test_req_hw_3721_validate_artifact_rejects_missing_required_fields() -> None:
    """REQ-HW-3721: validation catches schema omissions before writing."""
    with pytest.raises(ValueError, match="artifact missing required fields"):
        mod.validate_artifact({"honest_verdict": mod.PARTIAL_VERDICT})


def test_req_hw_3721_run_command_captures_output_and_exec_errors() -> None:
    """REQ-HW-3721: command probes preserve command, output, and failed execs."""
    ok_probe = mod.run_command(
        (sys.executable, "-c", "import sys; print('OK'); sys.stderr.write('ERR\\n')"),
        timeout_s=10,
    )
    assert ok_probe.exit_code == 0
    assert ok_probe.stdout == "OK\n"
    assert ok_probe.stderr == "ERR\n"
    assert ok_probe.combined_output == "OK\nERR\n"
    assert sys.executable in ok_probe.as_dict()["command"]

    missing_probe = mod.run_command(("/definitely/missing/ssh-for-req-hw-3721",))
    assert missing_probe.exit_code == 127
    assert missing_probe.command == ("/definitely/missing/ssh-for-req-hw-3721",)


def test_scenario_hw_3721_script_wrapper_exists() -> None:
    """SCENARIO-HW-3721: conductor entrypoint delegates to the module main."""
    script = Path("scripts/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.py")
    text = script.read_text(encoding="utf-8")
    assert "experiment_3721_hardware_kv260_terminal_confirm_and_continuity" in text
    assert "main" in text
