"""Tests for Exp 5179 hardware continuity board timing diagnostics.

Spec refs: REQ-HW-5179, SCENARIO-HW-5179.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from carnot import experiment_5179_hardware_continuity_board_timing as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """SCENARIO-HW-5179 runner that returns exact command transcripts."""

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
    """Deterministic REQ-HW-5179 clock for checksum-stable tests."""

    def __call__(self) -> float:
        return 5179.0


def _probe(
    command: tuple[str, ...],
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_s: float = 0.01,
) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _write_wishlist_update(root: Path) -> None:
    path = root / "research-hardware-wishlist.md"
    path.write_text(
        "# Hardware Wishlist\n\n"
        "### Active hardware tracks (test fixture)\n\n"
        "| Track | Why active now | Boundary |\n"
        "|---|---|---|\n"
        + "\n".join(f"| {marker} | test status | no speedup claim |" for marker in mod.WISHLIST_MARKERS)
        + "\n",
        encoding="utf-8",
    )


def _board_stdout(workload_hash: str, executable_hash: str, expected_energy: int) -> str:
    return (
        json.dumps(
            {
                "correctness": {"energy_matches_expected": True},
                "duration_s": 0.000123,
                "energy": expected_energy,
                "executable_sha256": executable_hash,
                "inference_substrate": mod.INFERENCE_SUBSTRATE,
                "sample_quality": {"finite_energy": True, "sample_count": 8},
                "workload_sha256": workload_hash,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _reachable_gatemate_stdout() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
        "\tirlength 6\n"
    )


def _scan_usb_stdout() -> str:
    return (
        "Bus device vid:pid       probe_type manufacturer serial           product\n"
        "003 006    0x1209:0xc0ca dirtyJtag  Jean THOMAS  1861832311111616 DirtyJTAG\n"
    )


def _usb_enumeration_stdout() -> str:
    return (
        "/dev/bus/usb/003/006 660 root:uucp\n"
        "DEVNAME=/dev/ttyACM0\n"
        "ID_MODEL=DirtyJTAG\n"
        "ID_VENDOR_ID=1209\n"
        "ID_MODEL_ID=c0ca\n"
        "ID_SERIAL=Jean_THOMAS_DirtyJTAG_1861832311111616\n"
        "ID_USB_DRIVER=cdc_acm\n"
    )


def _diagnostic_success_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    return {
        mod.GATEMATE_SCAN_USB_COMMAND: [_probe(mod.GATEMATE_SCAN_USB_COMMAND, stdout=_scan_usb_stdout())],
        mod.GATEMATE_VERSION_COMMAND: [
            _probe(mod.GATEMATE_VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
        ],
        mod.GATEMATE_USB_ENUMERATION_COMMAND: [
            _probe(mod.GATEMATE_USB_ENUMERATION_COMMAND, stdout=_usb_enumeration_stdout())
        ],
        mod.GATEMATE_DMESG_COMMAND: [
            _probe(
                mod.GATEMATE_DMESG_COMMAND,
                stdout="[Thu Jul  2 21:30:37 2026] usb 3-2.3: reset full-speed USB device number 6 using xhci_hcd\n",
            )
        ],
        mod.GATEMATE_VERBOSE_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_VERBOSE_DETECT_COMMAND,
                stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\nfound 0 devices\n",
            )
        ],
        mod.GATEMATE_LOW_FREQ_DETECT_COMMAND: [
            _probe(
                mod.GATEMATE_LOW_FREQ_DETECT_COMMAND,
                stdout="Jtag frequency : requested 1000000 Hz -> real 1000000 Hz\n",
            )
        ],
    }


def _all_reachable_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    kv260_command = mod.kv260_workload_command()
    polarfire_command = mod.polarfire_workload_command()
    probes = _diagnostic_success_probes()
    probes.update(
        {
            mod.KV260_PRECONDITION_COMMAND: [
                _probe(mod.KV260_PRECONDITION_COMMAND, duration_s=0.2)
            ],
            kv260_command: [
                _probe(
                    kv260_command,
                    stdout=_board_stdout(
                        mod.KV260_WORKLOAD_HASH,
                        mod.INLINE_EXECUTABLE_HASH,
                        mod.KV260_EXPECTED_ENERGY,
                    ),
                    duration_s=0.31,
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout=_reachable_gatemate_stdout(), duration_s=0.4),
                _probe(mod.GATEMATE_DETECT_COMMAND, stdout=_reachable_gatemate_stdout(), duration_s=0.41),
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [
                _probe(mod.POLARFIRE_PRECONDITION_COMMAND, duration_s=0.5)
            ],
            polarfire_command: [
                _probe(
                    polarfire_command,
                    stdout=_board_stdout(
                        mod.POLARFIRE_WORKLOAD_HASH,
                        mod.INLINE_EXECUTABLE_HASH,
                        mod.POLARFIRE_EXPECTED_ENERGY,
                    ),
                    duration_s=0.61,
                )
            ],
        }
    )
    return probes


def _blocked_gatemate_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    kv260_command = mod.kv260_workload_command()
    polarfire_command = mod.polarfire_workload_command()
    probes = _diagnostic_success_probes()
    probes[mod.GATEMATE_SCAN_USB_COMMAND].append(
        _probe(mod.GATEMATE_SCAN_USB_COMMAND, stdout=_scan_usb_stdout())
    )
    probes.update(
        {
            mod.KV260_PRECONDITION_COMMAND: [_probe(mod.KV260_PRECONDITION_COMMAND)],
            kv260_command: [
                _probe(
                    kv260_command,
                    stdout=_board_stdout(
                        mod.KV260_WORKLOAD_HASH,
                        mod.INLINE_EXECUTABLE_HASH,
                        mod.KV260_EXPECTED_ENERGY,
                    ),
                )
            ],
            mod.GATEMATE_DETECT_COMMAND: [
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                ),
                _probe(
                    mod.GATEMATE_DETECT_COMMAND,
                    stdout="Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n",
                ),
            ],
            mod.GATEMATE_POWER_PORT_COMMAND: [
                _probe(
                    mod.GATEMATE_POWER_PORT_COMMAND,
                    exit_code=127,
                    stdout="uhubctl not installed; physical port power-cycle not available from this shell\n",
                )
            ],
            mod.GATEMATE_USB_RESET_COMMAND: [
                _probe(mod.GATEMATE_USB_RESET_COMMAND, stdout="Resetting DirtyJTAG ... ok\n")
            ],
            mod.POLARFIRE_PRECONDITION_COMMAND: [_probe(mod.POLARFIRE_PRECONDITION_COMMAND)],
            polarfire_command: [
                _probe(
                    polarfire_command,
                    stdout=_board_stdout(
                        mod.POLARFIRE_WORKLOAD_HASH,
                        mod.INLINE_EXECUTABLE_HASH,
                        mod.POLARFIRE_EXPECTED_ENERGY,
                    ),
                )
            ],
        }
    )
    return probes


def _valid_artifact(tmp_path: Path) -> dict[str, object]:
    _write_wishlist_update(tmp_path)
    return mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_all_reachable_probes()),
        clock=FlatClock(),
        run_date="20260702",
    )


def _with_checksum(artifact: dict[str, object], **updates: object) -> dict[str, object]:
    bad = copy.deepcopy(artifact)
    bad.update(updates)
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    return bad


def test_req_hw_5179_spec_declares_gatemate_diagnostic_contract() -> None:
    """REQ-HW-5179: OpenSpec anchors the v474 diagnostic artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-HW-5179") :]

    for marker in (
        "REQ-HW-5179",
        "SCENARIO-HW-5179",
        "experiment_5179_hardware_continuity_board_timing_v474.json",
        "gatemate_idcode_diagnostic_attempts",
        "openFPGALoader --scan-usb",
        "openFPGALoader -V",
        "blocked_gatemate_dirtyjtag_idcode",
        "inference_substrate=hardware_smoke",
        "hardware_wishlist_updated=true",
        "no_speedup_claim=true",
        "hardware_speedup_claimed=false",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_hw_5179_all_reachable_writes_hash_verified_transcripts(tmp_path: Path) -> None:
    """SCENARIO-HW-5179: reachable boards get hash-verified timing transcripts."""

    _write_wishlist_update(tmp_path)
    runner = RecordingRunner(_all_reachable_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )
    out_path = mod.write_artifact(tmp_path, artifact)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.kv260_workload_command(),
        mod.GATEMATE_SCAN_USB_COMMAND,
        mod.GATEMATE_VERSION_COMMAND,
        mod.GATEMATE_USB_ENUMERATION_COMMAND,
        mod.GATEMATE_DMESG_COMMAND,
        mod.GATEMATE_VERBOSE_DETECT_COMMAND,
        mod.GATEMATE_LOW_FREQ_DETECT_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.polarfire_workload_command(),
    ]
    assert saved["experiment_id"] == mod.EXPERIMENT_ID
    assert saved["milestone"] == "2026.07.474"
    assert saved["honest_verdict"].startswith("complete_")
    assert "kv260:reachable" in saved["honest_verdict"]
    assert "gatemate:reachable_idcode_resolved" in saved["honest_verdict"]
    assert "polarfire:reachable" in saved["honest_verdict"]
    assert saved["inference_substrate"] == "hardware_smoke"
    assert saved["boards_reachable_count"] == 3
    assert saved["hardware_wishlist_updated"] is True
    assert saved["kv260_result"]["reachable"] is True
    assert saved["kv260_result"]["workload_hash"] == mod.KV260_WORKLOAD_HASH
    assert saved["kv260_result"]["hash_verified"] is True
    assert saved["gatemate_result"]["reachable"] is True
    assert saved["gatemate_result"]["workload_hash"] == mod.GATEMATE_WORKLOAD_HASH
    assert saved["gatemate_result"]["hash_verified"] is True
    assert saved["polarfire_result"]["reachable"] is True
    assert saved["polarfire_result"]["workload_hash"] == mod.POLARFIRE_WORKLOAD_HASH
    assert saved["sample_quality_evidence"]["speedup_evidence_claimed"] is False
    assert saved["no_speedup_claim"] is True
    assert saved["hardware_speedup_claimed"] is False
    assert saved["kv260_host_block_devices_touched"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert "/dev/mmcblk" not in json.dumps(saved, sort_keys=True).lower()

    methods = {attempt["method"] for attempt in saved["gatemate_idcode_diagnostic_attempts"]}
    assert {"scan_usb", "tool_version_compare", "usb_enumeration", "verbose_detect"} <= methods
    assert any(attempt["method"] != "detect" for attempt in saved["gatemate_idcode_diagnostic_attempts"])
    mod.validate_artifact(saved)


def test_scenario_hw_5179_gatemate_idcode_block_runs_differential_diagnostics(tmp_path: Path) -> None:
    """SCENARIO-HW-5179: blocked GateMate includes real non-identical diagnostics."""

    _write_wishlist_update(tmp_path)
    runner = RecordingRunner(_blocked_gatemate_probes())

    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=runner,
        clock=FlatClock(),
        run_date="20260702",
    )

    assert runner.commands == [
        mod.KV260_PRECONDITION_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.POLARFIRE_PRECONDITION_COMMAND,
        mod.kv260_workload_command(),
        mod.GATEMATE_SCAN_USB_COMMAND,
        mod.GATEMATE_VERSION_COMMAND,
        mod.GATEMATE_USB_ENUMERATION_COMMAND,
        mod.GATEMATE_DMESG_COMMAND,
        mod.GATEMATE_VERBOSE_DETECT_COMMAND,
        mod.GATEMATE_LOW_FREQ_DETECT_COMMAND,
        mod.GATEMATE_POWER_PORT_COMMAND,
        mod.GATEMATE_USB_RESET_COMMAND,
        mod.GATEMATE_SCAN_USB_COMMAND,
        mod.GATEMATE_DETECT_COMMAND,
        mod.polarfire_workload_command(),
    ]
    assert artifact["honest_verdict"].startswith("complete_")
    assert "kv260:reachable" in artifact["honest_verdict"]
    assert "gatemate:blocked_gatemate_dirtyjtag_idcode_unresolved" in artifact["honest_verdict"]
    assert "polarfire:reachable" in artifact["honest_verdict"]
    assert artifact["boards_reachable_count"] == 2
    assert artifact["kv260_result"]["reachable"] is True
    assert artifact["gatemate_result"]["reachable"] is False
    assert artifact["gatemate_result"]["blocked_reason"] == "blocked_gatemate_dirtyjtag_idcode"
    assert artifact["gatemate_result"]["latency_transcript"] is None
    assert artifact["polarfire_result"]["reachable"] is True
    assert artifact["sample_quality_evidence"]["reachable_boards"] == ["kv260", "polarfire"]
    assert artifact["command_transcripts"]["gatemate_precondition"]["combined_output"] == (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
    )
    assert artifact["command_transcripts"]["gatemate_post_reset_detect"]["combined_output"] == (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
    )

    attempts = artifact["gatemate_idcode_diagnostic_attempts"]
    methods = [attempt["method"] for attempt in attempts]
    for required in (
        "scan_usb",
        "tool_version_compare",
        "usb_enumeration",
        "kernel_log",
        "verbose_detect",
        "low_frequency_detect",
        "power_or_port_cycle",
        "usb_reset",
        "post_reset_detect",
        "physical_reseat_or_port_move",
    ):
        assert required in methods
    assert any("DirtyJTAG enumerated" in attempt["outcome"] for attempt in attempts)
    assert any("matches known-good" in attempt["outcome"] for attempt in attempts)
    assert any("requires operator physical access" in attempt["outcome"] for attempt in attempts)
    mod.validate_artifact(artifact)


def test_req_hw_5179_validation_rejects_overclaims_and_schema_drift(tmp_path: Path) -> None:
    """REQ-HW-5179: validation rejects overclaims, bad diagnostics, and unsafe storage."""

    artifact = _valid_artifact(tmp_path)

    for field, value in (
        ("inference_substrate", "local_board_transcripts_or_blocked"),
        ("no_speedup_claim", False),
        ("hardware_speedup_claimed", True),
        ("kv260_host_block_devices_touched", True),
        ("hardware_wishlist_updated", False),
    ):
        bad = _with_checksum(artifact, **{field: value})
        with pytest.raises(ValueError, match=field):
            mod.validate_artifact(bad)

    bad_verdict = _with_checksum(artifact, honest_verdict="blocked_one_board_unreachable")
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_count = _with_checksum(artifact, boards_reachable_count=2)
    with pytest.raises(ValueError, match="boards_reachable_count"):
        mod.validate_artifact(bad_count)

    bad_diagnostics = _with_checksum(
        artifact,
        gatemate_idcode_diagnostic_attempts=[
            {"attempt": 1, "method": "detect", "outcome": "same detect only"}
        ],
    )
    with pytest.raises(ValueError, match="gatemate_idcode_diagnostic_attempts"):
        mod.validate_artifact(bad_diagnostics)

    for malformed in ([], ["not-a-dict"], [{"attempt": 1}], [{"attempt": "1", "method": "", "outcome": ""}]):
        bad = _with_checksum(artifact, gatemate_idcode_diagnostic_attempts=malformed)
        with pytest.raises(ValueError, match="gatemate_idcode_diagnostic_attempts"):
            mod.validate_artifact(bad)

    bad_result = copy.deepcopy(artifact)
    bad_result["kv260_result"]["reachable"] = "yes"
    bad_result["reproducibility_checksum"] = mod.payload_checksum(bad_result)
    with pytest.raises(ValueError, match="kv260_result"):
        mod.validate_artifact(bad_result)

    bad_storage = copy.deepcopy(artifact)
    bad_storage["command_transcripts"] = {"unsafe": {"command": "ls /dev/" + "mmcblk0"}}
    bad_storage["reproducibility_checksum"] = mod.payload_checksum(bad_storage)
    with pytest.raises(ValueError, match="host storage"):
        mod.validate_artifact(bad_storage)

    bad_checksum = dict(artifact, reproducibility_checksum="stale")
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)

    assert "missing required fields" in "; ".join(mod.artifact_schema_errors({}))
    assert "kv260_result must be a dict" in mod.artifact_schema_errors(
        _with_checksum(artifact, kv260_result=[])
    )


def test_req_hw_5179_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-HW-5179: helper parsing, hash checks, and blockers fail closed."""

    assert mod.ising_energy(mod.KV260_WORKLOAD) == mod.KV260_EXPECTED_ENERGY
    assert mod.parse_probe_json(None) == {}
    assert mod.parse_probe_json(_probe(mod.KV260_PRECONDITION_COMMAND, stdout="not-json\n")) == {}
    assert mod.parse_probe_json(_probe(mod.KV260_PRECONDITION_COMMAND, stdout="[1, 2]\n")) == {}
    assert mod.wishlist_has_update(tmp_path) is False
    assert mod.outcome_has_idcode(_probe(mod.GATEMATE_DETECT_COMMAND, stdout=_reachable_gatemate_stdout()))
    assert not mod.outcome_has_idcode(_probe(mod.GATEMATE_DETECT_COMMAND, stdout="frequency only\n"))
    assert mod.detect_outcome(None) == "not_run"
    assert "resolved" in mod.detect_outcome(
        _probe(mod.GATEMATE_DETECT_COMMAND, stdout=_reachable_gatemate_stdout())
    )
    assert mod.scan_usb_outcome(None) == "not_run"
    assert "not enumerated" in mod.scan_usb_outcome(
        _probe(mod.GATEMATE_SCAN_USB_COMMAND, stdout="no probes\n")
    )
    assert mod.version_outcome(None) == "not_run"
    assert "version drift" in mod.version_outcome(
        _probe(mod.GATEMATE_VERSION_COMMAND, stdout="openFPGALoader v9.9.9\n")
    )
    assert mod.usb_enumeration_outcome(None) == "not_run"
    assert "did not confirm" in mod.usb_enumeration_outcome(
        _probe(mod.GATEMATE_USB_ENUMERATION_COMMAND, stdout="ID_MODEL=Other\n")
    )
    assert mod.kernel_log_outcome(None) == "not_run"
    assert "unavailable or empty" in mod.kernel_log_outcome(
        _probe(mod.GATEMATE_DMESG_COMMAND, exit_code=1, stderr="dmesg denied\n")
    )
    assert mod.changed_detect_outcome(None) == "not_run"
    assert "resolved" in mod.changed_detect_outcome(
        _probe(mod.GATEMATE_VERBOSE_DETECT_COMMAND, stdout=_reachable_gatemate_stdout())
    )
    assert mod.power_port_outcome(None) == "not_run"
    assert "host power/port utility reported" in mod.power_port_outcome(
        _probe(mod.GATEMATE_POWER_PORT_COMMAND, stdout="hub power ok\n")
    )
    assert mod.usb_reset_outcome(None) == "not_run"
    assert "unavailable or failed" in mod.usb_reset_outcome(
        _probe(mod.GATEMATE_USB_RESET_COMMAND, exit_code=1, stderr="reset failed\n")
    )
    assert mod.dirtyjtag_seen(None) is False

    output = {
        "workload_sha256": "b" * 64,
        "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
        "sample_quality": {"finite_energy": True},
        "correctness": {"energy_matches_expected": True},
    }
    assert mod.ssh_output_hash_verified("a" * 64, output) is False
    assert mod.ssh_output_has_evidence({}) is False
    assert mod.ssh_workload_blocker("unit", None, "a" * 64, {}) == "blocked_unit_workload_missing"
    assert (
        mod.ssh_workload_blocker(
            "unit",
            _probe(mod.KV260_PRECONDITION_COMMAND, exit_code=1),
            "a" * 64,
            {},
        )
        == "blocked_unit_workload_command"
    )
    evidence_missing = {
        "workload_sha256": "a" * 64,
        "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }
    assert (
        mod.ssh_workload_blocker(
            "unit",
            _probe(mod.KV260_PRECONDITION_COMMAND),
            "a" * 64,
            evidence_missing,
        )
        == "blocked_unit_workload_evidence"
    )

    gate_command_failed = mod.finish_gatemate_board(
        precondition_probe=_probe(
            mod.GATEMATE_DETECT_COMMAND,
            stdout=_reachable_gatemate_stdout(),
        ),
        command_runner=RecordingRunner(
            {
                **_diagnostic_success_probes(),
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="detect failed\n")
                ],
            }
        ),
    )
    assert gate_command_failed["blocked_reason"] == "blocked_gatemate_workload_command"

    gate_idcode_failed = mod.finish_gatemate_board(
        precondition_probe=_probe(
            mod.GATEMATE_DETECT_COMMAND,
            stdout=_reachable_gatemate_stdout(),
        ),
        command_runner=RecordingRunner(
            {
                **_diagnostic_success_probes(),
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(mod.GATEMATE_DETECT_COMMAND, stdout="IDCode : 0x00000000\n")
                ],
            }
        ),
    )
    assert gate_idcode_failed["blocked_reason"] == "blocked_gatemate_workload_idcode"

    precondition_failed = mod.finish_gatemate_board(
        precondition_probe=_probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="fails to open\n"),
        command_runner=RecordingRunner(
            {
                **_diagnostic_success_probes(),
                mod.GATEMATE_POWER_PORT_COMMAND: [
                    _probe(mod.GATEMATE_POWER_PORT_COMMAND, stdout="power utility unavailable\n", exit_code=127)
                ],
                mod.GATEMATE_USB_RESET_COMMAND: [
                    _probe(mod.GATEMATE_USB_RESET_COMMAND, stderr="reset failed\n", exit_code=1)
                ],
            }
        ),
    )
    assert precondition_failed["blocked_reason"] == "blocked_gatemate_dirtyjtag"
    assert precondition_failed["workload_probe"] is None
    assert precondition_failed["diagnostic_probes"]["post_reset_detect"] is None

    _write_wishlist_update(tmp_path)
    artifact = mod.build_artifact(
        repo_root=tmp_path,
        command_runner=RecordingRunner(
            {
                mod.KV260_PRECONDITION_COMMAND: [
                    _probe(mod.KV260_PRECONDITION_COMMAND, exit_code=255, stderr="no route\n")
                ],
                mod.GATEMATE_DETECT_COMMAND: [
                    _probe(mod.GATEMATE_DETECT_COMMAND, exit_code=1, stderr="fails to open\n")
                ],
                **_diagnostic_success_probes(),
                mod.GATEMATE_POWER_PORT_COMMAND: [
                    _probe(mod.GATEMATE_POWER_PORT_COMMAND, stdout="power utility unavailable\n", exit_code=127)
                ],
                mod.GATEMATE_USB_RESET_COMMAND: [
                    _probe(mod.GATEMATE_USB_RESET_COMMAND, stderr="reset failed\n", exit_code=1)
                ],
                mod.POLARFIRE_PRECONDITION_COMMAND: [_probe(mod.POLARFIRE_PRECONDITION_COMMAND)],
                mod.polarfire_workload_command(): [
                    _probe(mod.polarfire_workload_command(), stdout=json.dumps(output) + "\n")
                ],
            }
        ),
        clock=FlatClock(),
        run_date="20260702",
    )
    assert artifact["kv260_result"]["blocked_reason"] == "blocked_kv260_ssh"
    assert artifact["gatemate_result"]["blocked_reason"] == "blocked_gatemate_dirtyjtag"
    assert artifact["polarfire_result"]["blocked_reason"] == "blocked_polarfire_workload_hash"
    assert artifact["boards_reachable_count"] == 1

    partial_root = tmp_path / "partial"
    partial_root.mkdir()
    partial_doc = partial_root / "research-hardware-wishlist.md"
    partial_doc.write_text("# Hardware Wishlist\n", encoding="utf-8")
    assert mod.ensure_hardware_wishlist_update(partial_root) == partial_doc
    assert mod.wishlist_has_update(partial_root) is True
    already = partial_doc.read_text(encoding="utf-8")
    assert mod.ensure_hardware_wishlist_update(partial_root) == partial_doc
    assert partial_doc.read_text(encoding="utf-8") == already


def test_scenario_hw_5179_run_experiment_and_cli_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-HW-5179: run_experiment and script entrypoints write the artifact."""

    out_path = mod.run_experiment(
        repo_root=tmp_path,
        command_runner=RecordingRunner(_all_reachable_probes()),
        clock=FlatClock(),
        run_date="20260702",
        update_wishlist=True,
    )
    artifact = json.loads(out_path.read_text(encoding="utf-8"))

    assert out_path == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["spec_refs"] == ["REQ-HW-5179", "SCENARIO-HW-5179"]
    assert artifact["hardware_wishlist_updated"] is True
    mod.validate_artifact(artifact)

    called: dict[str, object] = {}

    def fake_run_experiment(**kwargs: object) -> Path:
        called["kwargs"] = kwargs
        path = Path(kwargs["repo_root"]) / mod.OUTPUT_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        return path

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    assert mod.main(["--date", "20260702", "--repo-root", str(tmp_path)]) == 0
    assert called["kwargs"]["run_date"] == "20260702"
    assert "gatemate_idcode_diagnostic_attempts" in capsys.readouterr().out

    script = importlib.import_module(
        "scripts.experiment_5179_hardware_continuity_board_timing_v474"
    )
    monkeypatch.setattr(script, "experiment_main", lambda argv: 19)
    assert script.main(["--date", "20260702"]) == 19
