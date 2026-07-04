"""Tests for Exp 5217 hardware continuity + GateMate narrowing (v477).

Spec refs: REQ-HW-5217, SCENARIO-HW-5217.

These tests drive the pure, deterministic experiment module with an injected
command runner so the whole artifact — including the debug-level raw-TDO
narrowing to the physical (cable_or_port) layer — is reproduced without live
hardware.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5217_hardware_continuity_v477 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "fpga" / "spec.md"


class RecordingRunner:
    """REQ-HW-5217 runner returning canned transcripts keyed by command."""

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
    """Deterministic clock so the reproducibility checksum is stable."""

    def __call__(self) -> float:
        return 5217.0


def _probe(command, exit_code=0, stdout="", stderr="", duration_s=0.01) -> mod.CommandProbe:
    return mod.CommandProbe(command, exit_code, stdout, stderr, duration_s)


def _board_stdout(workload_hash: str) -> str:
    return (
        json.dumps(
            {
                "correctness": {"energy_matches_expected": True},
                "duration_s": 0.000123,
                "energy": 0,
                "executable_sha256": mod.INLINE_EXECUTABLE_HASH,
                "inference_substrate": mod.INFERENCE_SUBSTRATE,
                "sample_quality": {"finite_energy": True, "sample_count": 8},
                "workload_sha256": workload_hash,
            },
            sort_keys=True,
        )
        + "\n"
    )


def _scan_usb_stdout() -> str:
    return (
        "Bus device vid:pid       probe_type manufacturer serial           product\n"
        "003 006    0x1209:0xc0ca dirtyJtag  Jean THOMAS  1861832311111616 DirtyJTAG\n"
    )


def _usb_enum_stdout() -> str:
    return (
        "Bus device vid:pid       probe_type manufacturer serial           product\n"
        "003 006    0x1209:0xc0ca dirtyJtag  Jean THOMAS  1861832311111616 DirtyJTAG\n"
        "/dev/bus/usb/003/006 660 root:uucp\n"
        "ianblenke wheel uucp users\n"
    )


def _firmware_stdout() -> str:
    return "bcdDevice=0111 product=DirtyJTAG\n"


def _resolved_detect_stdout() -> str:
    return (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
    )


def _no_idcode_stdout(freq: str = "6000000") -> str:
    return f"Jtag frequency : requested {freq} Hz -> real {freq} Hz\nfound 0 devices\n"


def _debug_floating_stdout(freq: str = "6000000") -> str:
    return (
        f"Jtag frequency : requested {freq} Hz -> real {freq} Hz\n"
        "Raw IDCODE:\n- 0 -> 0xffffffff\n"
        "Fetched TDI, end-of-chain\n"
        "found 0 devices\n"
    )


KV_WORKLOAD_CMD = mod.ssh_workload_command("kria", mod.KV260_WORKLOAD, mod.KV260_WORKLOAD_HASH)
PF_WORKLOAD_CMD = mod.ssh_workload_command(
    "polarfire", mod.POLARFIRE_WORKLOAD, mod.POLARFIRE_WORKLOAD_HASH
)


def _blocked_probes() -> dict[tuple[str, ...], list[mod.CommandProbe]]:
    """The real current state: all SSH boards up, GateMate raw TDO floating."""
    return {
        mod.KV260_PRECONDITION_COMMAND: [_probe(mod.KV260_PRECONDITION_COMMAND)],
        mod.POLARFIRE_PRECONDITION_COMMAND: [_probe(mod.POLARFIRE_PRECONDITION_COMMAND)],
        mod.GATEMATE_DETECT_COMMAND: [
            _probe(mod.GATEMATE_DETECT_COMMAND, stdout=_no_idcode_stdout())
        ],
        KV_WORKLOAD_CMD: [_probe(KV_WORKLOAD_CMD, stdout=_board_stdout(mod.KV260_WORKLOAD_HASH))],
        PF_WORKLOAD_CMD: [
            _probe(PF_WORKLOAD_CMD, stdout=_board_stdout(mod.POLARFIRE_WORKLOAD_HASH))
        ],
        mod.GATEMATE_SCAN_USB_COMMAND: [
            _probe(mod.GATEMATE_SCAN_USB_COMMAND, stdout=_scan_usb_stdout())
        ],
        mod.GATEMATE_VERSION_COMMAND: [
            _probe(mod.GATEMATE_VERSION_COMMAND, stdout="openFPGALoader v1.1.1\n")
        ],
        mod.GATEMATE_USB_ENUMERATION_COMMAND: [
            _probe(mod.GATEMATE_USB_ENUMERATION_COMMAND, stdout=_usb_enum_stdout())
        ],
        mod.GATEMATE_LOW_FREQ_DETECT_COMMAND: [
            _probe(mod.GATEMATE_LOW_FREQ_DETECT_COMMAND, stdout=_no_idcode_stdout("100000"))
        ],
        mod.GATEMATE_DOC_FREQ_DETECT_COMMAND: [
            _probe(mod.GATEMATE_DOC_FREQ_DETECT_COMMAND, stdout=_no_idcode_stdout("15000000"))
        ],
        mod.GATEMATE_DEBUG_DETECT_COMMAND: [
            _probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, stdout=_debug_floating_stdout())
        ],
        mod.GATEMATE_PROBE_FIRMWARE_COMMAND: [
            _probe(mod.GATEMATE_PROBE_FIRMWARE_COMMAND, stdout=_firmware_stdout())
        ],
    }


def _build_blocked() -> dict:
    runner = RecordingRunner(_blocked_probes())
    return mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date="20260704")


# --------------------------------------------------------------------------- #
# Full-artifact scenarios
# --------------------------------------------------------------------------- #


def test_blocked_artifact_narrows_to_cable_or_port_from_floating_tdo():
    """SCENARIO-HW-5217: all-ones raw TDO -> physical cable_or_port narrowing."""
    artifact = _build_blocked()
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["spec_refs"] == ["REQ-HW-5217", "SCENARIO-HW-5217"]
    assert artifact["random_seed"] == 5217
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["kv260_precondition_command"] == mod.KV260_PRECONDITION_COMMAND_STR
    assert artifact["gatemate_raw_idcode"] == "0xffffffff"
    assert artifact["gatemate_diagnostic_narrowed_to"] == "cable_or_port"
    assert artifact["gatemate_leading_untested_hypothesis"] == "physical_board"
    assert (
        artifact["gatemate_status"]["status"]
        == "blocked_gatemate_dirtyjtag_idcode_unresolved_v477"
    )
    assert artifact["gatemate_status"]["reachable"] is False
    assert artifact["gatemate_status"]["raw_idcode"] == "0xffffffff"
    assert artifact["boards_reachable_count"] == 2
    assert artifact["honest_verdict"].startswith("complete_")
    assert "no_speedup_claim" in artifact["honest_verdict"]
    assert artifact["no_speedup_claim"] is True
    # Five candidate layers mechanically eliminated (incl. the new probe firmware).
    eliminated = {c["cause"] for c in artifact["gatemate_eliminated_causes"]}
    assert eliminated == {
        "usb_level",
        "permissions",
        "firmware_version",
        "probe_firmware_version",
        "clock_rate",
    }
    mod.validate_artifact(artifact)


def test_blocked_artifact_includes_new_live_angles():
    """REQ-HW-5217: the new angles include the debug raw-TDO + probe-firmware reads."""
    artifact = _build_blocked()
    angles = {a["angle"]: a for a in artifact["new_diagnostic_angles_tried_this_milestone"]}
    assert "debug_level_raw_idcode_capture" in angles
    assert "dirtyjtag_probe_firmware_version" in angles
    assert "cable_or_port_swap" in angles
    assert angles["debug_level_raw_idcode_capture"]["live"] is True
    assert "0xffffffff" in angles["debug_level_raw_idcode_capture"]["finding"]
    assert "idle" in angles["debug_level_raw_idcode_capture"]["finding"].lower()
    assert angles["dirtyjtag_probe_firmware_version"]["live"] is True
    assert "0111" in angles["dirtyjtag_probe_firmware_version"]["finding"]


def test_blocked_artifact_polarfire_not_terminal():
    """REQ-HW-5217: a reachable smoke does not close the PolarFire terminal bar."""
    artifact = _build_blocked()
    pf = artifact["polarfire_status"]
    assert pf["reachable"] is True
    assert pf["polarfire_workload_validated"] is False
    assert mod.is_sha256(pf["workload_hash"])
    assert (
        "end-to-end" in pf["terminal_bar_rationale"] or "dispatch" in pf["terminal_bar_rationale"]
    )
    kv = artifact["kv260_status"]
    assert kv.startswith("reachable")


def test_blocked_artifact_checksum_stable_and_no_host_storage():
    a1 = _build_blocked()
    a2 = _build_blocked()
    assert a1["reproducibility_checksum"] == a2["reproducibility_checksum"]
    assert mod.no_host_storage(a1)
    assert "mmcblk" not in json.dumps(a1).lower()


def test_specific_raw_idcode_stays_jtag_protocol_level():
    """A specific (non-idle) raw TDO word -> a present-but-wrong TAP, not physical."""
    probes = _blocked_probes()
    debug_specific = (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "Raw IDCODE:\n- 0 -> 0x12345678\n"
        "found 0 devices\n"
    )
    probes[mod.GATEMATE_DEBUG_DETECT_COMMAND] = [
        _probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, stdout=debug_specific)
    ]
    runner = RecordingRunner(probes)
    artifact = mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date="20260704")
    assert artifact["gatemate_raw_idcode"] == "0x12345678"
    assert artifact["gatemate_diagnostic_narrowed_to"] == "jtag_protocol_level"
    assert artifact["gatemate_leading_untested_hypothesis"] == "cable_or_port"
    angle = next(
        a
        for a in artifact["new_diagnostic_angles_tried_this_milestone"]
        if a["angle"] == "debug_level_raw_idcode_capture"
    )
    assert "protocol layer" in angle["finding"]


def test_debug_without_raw_line_stays_jtag_protocol_level():
    """A debug detect that never printed a raw word -> generic protocol layer."""
    probes = _blocked_probes()
    probes[mod.GATEMATE_DEBUG_DETECT_COMMAND] = [
        _probe(mod.GATEMATE_DEBUG_DETECT_COMMAND, stdout=_no_idcode_stdout())
    ]
    runner = RecordingRunner(probes)
    artifact = mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date="20260704")
    assert artifact["gatemate_raw_idcode"] is None
    assert artifact["gatemate_diagnostic_narrowed_to"] == "jtag_protocol_level"
    angle = next(
        a
        for a in artifact["new_diagnostic_angles_tried_this_milestone"]
        if a["angle"] == "debug_level_raw_idcode_capture"
    )
    assert "found 0 devices" in angle["finding"]


def test_resolved_scenario_marks_board_reachable():
    """When --detect reads the IDCODE, the board is reachable and narrowing=resolved."""
    probes = _blocked_probes()
    probes[mod.GATEMATE_DETECT_COMMAND] = [
        _probe(mod.GATEMATE_DETECT_COMMAND, stdout=_resolved_detect_stdout())
    ]
    runner = RecordingRunner(probes)
    artifact = mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date="20260704")
    assert artifact["gatemate_diagnostic_narrowed_to"] == "resolved"
    assert artifact["gatemate_status"]["reachable"] is True
    assert artifact["gatemate_status"]["status"] == "resolved"
    assert artifact["gatemate_leading_untested_hypothesis"] == "none_board_reachable"
    assert artifact["gatemate_raw_idcode"] is None
    assert artifact["boards_reachable_count"] == 3
    assert "reachable_idcode_resolved" in artifact["honest_verdict"]
    # Resolved path runs only the 3 base diagnostics (no freq sweep / debug / firmware).
    angles = {a["angle"] for a in artifact["new_diagnostic_angles_tried_this_milestone"]}
    assert "debug_level_raw_idcode_capture" in angles  # static angle always present
    debug_angle = next(
        a
        for a in artifact["new_diagnostic_angles_tried_this_milestone"]
        if a["angle"] == "debug_level_raw_idcode_capture"
    )
    assert debug_angle["finding"] == "not_run"
    fw_angle = next(
        a
        for a in artifact["new_diagnostic_angles_tried_this_milestone"]
        if a["angle"] == "dirtyjtag_probe_firmware_version"
    )
    assert fw_angle["finding"] == "not_run"
    # The 3 base probes still run on the resolved path (no freq sweep / firmware),
    # so USB/permissions/tool-version are eliminated but clock_rate/probe_firmware are not.
    assert {c["cause"] for c in artifact["gatemate_eliminated_causes"]} == {
        "usb_level",
        "permissions",
        "firmware_version",
    }


def test_all_ssh_unreachable_counts_zero():
    probes = _blocked_probes()
    probes[mod.KV260_PRECONDITION_COMMAND] = [_probe(mod.KV260_PRECONDITION_COMMAND, exit_code=255)]
    probes[mod.POLARFIRE_PRECONDITION_COMMAND] = [
        _probe(mod.POLARFIRE_PRECONDITION_COMMAND, exit_code=255)
    ]
    runner = RecordingRunner(probes)
    artifact = mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date="20260704")
    assert artifact["boards_reachable_count"] == 0
    assert artifact["kv260_status"].startswith("unreachable")
    assert artifact["polarfire_status"]["reachable"] is False
    assert "polarfire_workload_validated=false" in artifact["polarfire_status"]["summary"]
    assert "blocked_kv260_ssh" in artifact["honest_verdict"]


# --------------------------------------------------------------------------- #
# Pure-helper unit tests
# --------------------------------------------------------------------------- #


def test_parse_probe_json_variants():
    assert mod.parse_probe_json(None) == {}
    assert mod.parse_probe_json(_probe(("x",), stdout="   ")) == {}
    assert mod.parse_probe_json(_probe(("x",), stdout='{"a": 1}\n')) == {"a": 1}
    assert mod.parse_probe_json(_probe(("x",), stdout="not json\n")) == {}
    assert mod.parse_probe_json(_probe(("x",), stdout="[1, 2]\n")) == {}


def test_finish_ssh_board_branches():
    unreachable = mod.finish_ssh_board(
        board="kv260",
        precondition_probe=_probe(("ssh",), exit_code=255),
        workload_command=("ssh", "kria", "x"),
        workload_hash=mod.KV260_WORKLOAD_HASH,
        command_runner=RecordingRunner({}),
    )
    assert unreachable["blocked_reason"] == "blocked_kv260_ssh"
    assert unreachable["hash_verified"] is False

    cmd = ("ssh", "kria", "x")
    fail = mod.finish_ssh_board(
        board="kv260",
        precondition_probe=_probe(("ssh",)),
        workload_command=cmd,
        workload_hash=mod.KV260_WORKLOAD_HASH,
        command_runner=RecordingRunner({cmd: [_probe(cmd, exit_code=1)]}),
    )
    assert fail["blocked_reason"] == "blocked_kv260_workload_command"

    mismatch = mod.finish_ssh_board(
        board="kv260",
        precondition_probe=_probe(("ssh",)),
        workload_command=cmd,
        workload_hash=mod.KV260_WORKLOAD_HASH,
        command_runner=RecordingRunner({cmd: [_probe(cmd, stdout=_board_stdout("deadbeef"))]}),
    )
    assert mismatch["blocked_reason"] == "blocked_kv260_workload_hash"

    ok = mod.finish_ssh_board(
        board="kv260",
        precondition_probe=_probe(("ssh",)),
        workload_command=cmd,
        workload_hash=mod.KV260_WORKLOAD_HASH,
        command_runner=RecordingRunner(
            {cmd: [_probe(cmd, stdout=_board_stdout(mod.KV260_WORKLOAD_HASH))]}
        ),
    )
    assert ok["hash_verified"] is True
    assert ok["workload_hash"] == mod.KV260_WORKLOAD_HASH
    assert ok["correctness"] == {"energy_matches_expected": True}


def test_dirtyjtag_seen_variants():
    assert mod.dirtyjtag_seen(None) is False
    assert mod.dirtyjtag_seen(_probe(("x",), stdout="DirtyJTAG here")) is True
    assert mod.dirtyjtag_seen(_probe(("x",), stdout="Bus 003 Device 006: ID 1209:c0ca")) is True
    assert mod.dirtyjtag_seen(_probe(("x",), stdout="ID_VENDOR_ID=1209")) is True
    assert mod.dirtyjtag_seen(_probe(("x",), stdout="nothing")) is False


def test_version_matches_known_good():
    assert mod.version_matches_known_good(None) is False
    assert mod.version_matches_known_good(_probe(("x",), stdout="openFPGALoader v1.1.1\n")) is True
    assert mod.version_matches_known_good(_probe(("x",), stdout="openFPGALoader v1.0.0\n")) is False


def test_permissions_ok_variants():
    assert mod.permissions_ok(None) is False
    assert mod.permissions_ok(_probe(("x",), exit_code=1, stdout="root:uucp uucp")) is False
    assert (
        mod.permissions_ok(_probe(("x",), stdout="node root:uucp\nian wheel uucp users\n")) is True
    )
    assert mod.permissions_ok(_probe(("x",), stdout="node root:root\nian wheel users\n")) is False


def test_clock_sweep_and_freq_sweep_done():
    empty = {
        "debug_detect": _probe(("x",), stdout=_debug_floating_stdout()),
        "low_freq_detect": _probe(("x",), stdout=_no_idcode_stdout("100000")),
        "doc_freq_detect": _probe(("x",), stdout=_no_idcode_stdout("15000000")),
    }
    assert mod.clock_sweep_all_failed(empty) is True
    assert mod.freq_sweep_done(empty) is True
    assert mod.clock_sweep_all_failed({}) is False
    assert mod.freq_sweep_done({}) is False
    one_hit = dict(empty)
    one_hit["doc_freq_detect"] = _probe(("x",), stdout=_resolved_detect_stdout())
    assert mod.clock_sweep_all_failed(one_hit) is False


def test_raw_idcode_from_debug_variants():
    assert mod.raw_idcode_from_debug(None) is None
    assert mod.raw_idcode_from_debug(_probe(("x",), stdout=_debug_floating_stdout())) == "0xffffffff"
    assert (
        mod.raw_idcode_from_debug(_probe(("x",), stdout="Raw IDCODE:\n- 0 -> 0xABCD1234\n"))
        == "0xabcd1234"
    )
    assert mod.raw_idcode_from_debug(_probe(("x",), stdout="found 0 devices\n")) is None


def test_is_floating_tdo_variants():
    assert mod.is_floating_tdo(None) is False
    assert mod.is_floating_tdo("0xffffffff") is True
    assert mod.is_floating_tdo("0xFFFFFFFF") is True
    assert mod.is_floating_tdo("0x00000000") is True
    assert mod.is_floating_tdo("0x12345678") is False
    assert mod.is_floating_tdo("0x") is False


def test_probe_firmware_helpers():
    assert mod.probe_firmware_bcd(None) is None
    assert mod.probe_firmware_bcd(_probe(("x",), stdout=_firmware_stdout())) == "0111"
    assert mod.probe_firmware_bcd(_probe(("x",), stdout="no firmware here")) is None
    assert mod.probe_firmware_unchanged(_probe(("x",), stdout=_firmware_stdout())) is True
    assert mod.probe_firmware_unchanged(_probe(("x",), stdout="bcdDevice=0200\n")) is False
    assert mod.probe_firmware_unchanged(None) is False


def test_narrow_gatemate_failure_all_branches():
    base = dict(
        idcode_resolved=False,
        usb_enumerated=True,
        perms_ok=True,
        version_ok=True,
        sweep_done=True,
        scan_chain_empty=True,
        raw_idcode="0xffffffff",
    )
    assert mod.narrow_gatemate_failure(**{**base, "idcode_resolved": True}) == "resolved"
    assert mod.narrow_gatemate_failure(**{**base, "usb_enumerated": False}) == "usb_level"
    assert mod.narrow_gatemate_failure(**{**base, "perms_ok": False}) == "permissions"
    assert mod.narrow_gatemate_failure(**{**base, "version_ok": False}) == "firmware_version"
    assert mod.narrow_gatemate_failure(**{**base, "sweep_done": False}) == "clock_rate"
    assert mod.narrow_gatemate_failure(**base) == "cable_or_port"
    assert mod.narrow_gatemate_failure(**{**base, "raw_idcode": "0x12345678"}) == "jtag_protocol_level"
    assert mod.narrow_gatemate_failure(**{**base, "raw_idcode": None}) == "jtag_protocol_level"
    assert mod.narrow_gatemate_failure(**{**base, "scan_chain_empty": False}) == "unknown"


def test_gatemate_leading_hypothesis():
    assert mod.gatemate_leading_hypothesis("resolved") == "none_board_reachable"
    assert mod.gatemate_leading_hypothesis("cable_or_port") == "physical_board"
    assert mod.gatemate_leading_hypothesis("jtag_protocol_level") == "cable_or_port"
    assert mod.gatemate_leading_hypothesis("usb_level") == "usb_level"


def test_gatemate_eliminated_causes_all_and_none():
    full = mod.gatemate_eliminated_causes(
        usb_enumerated=True, perms_ok=True, version_ok=True, sweep_done=True, probe_fw_ok=True
    )
    assert {c["cause"] for c in full} == {
        "usb_level",
        "permissions",
        "firmware_version",
        "probe_firmware_version",
        "clock_rate",
    }
    none = mod.gatemate_eliminated_causes(
        usb_enumerated=False, perms_ok=False, version_ok=False, sweep_done=False, probe_fw_ok=False
    )
    assert none == []


def test_build_new_angles_firmware_and_debug_fallbacks():
    # Firmware probe present but with no bcdDevice line -> raw combined_output finding.
    probes = {
        "debug_detect": _probe(("x",), stdout=_debug_floating_stdout()),
        "probe_firmware": _probe(("x",), stdout="garbage line\n"),
    }
    angles = {a["angle"]: a for a in mod.build_new_angles(probes, "0xffffffff")}
    assert angles["dirtyjtag_probe_firmware_version"]["finding"] == "garbage line"
    # No probes at all -> both live angles report not_run.
    empties = {a["angle"]: a for a in mod.build_new_angles({}, None)}
    assert empties["debug_level_raw_idcode_capture"]["finding"] == "not_run"
    assert empties["dirtyjtag_probe_firmware_version"]["finding"] == "not_run"


def test_kv260_status_text_smoke_blocked_branch():
    result = {
        "reachable": True,
        "hash_verified": False,
        "workload_hash": None,
        "blocked_reason": "blocked_kv260_workload_hash",
    }
    assert (
        mod.kv260_status_text(result) == "reachable but smoke blocked (blocked_kv260_workload_hash)"
    )


def test_terminal_prefix_ok():
    assert mod.terminal_prefix_ok("complete_x") is True
    assert mod.terminal_prefix_ok("success: y") is True
    assert mod.terminal_prefix_ok("blocked_z") is False


# --------------------------------------------------------------------------- #
# Validator (malformed-artifact) tests
# --------------------------------------------------------------------------- #


def test_missing_field_reports_error():
    artifact = _build_blocked()
    del artifact["honest_verdict"]
    errors = mod.artifact_schema_errors(artifact)
    assert any("missing required fields" in e for e in errors)


def test_validate_artifact_raises_on_bad_narrowing():
    artifact = _build_blocked()
    artifact["gatemate_diagnostic_narrowed_to"] = "not_a_layer"
    with pytest.raises(ValueError):
        mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda a: a.update(schema="x"), "schema mismatch"),
        (lambda a: a.update(experiment="x"), "experiment mismatch"),
        (lambda a: a.update(experiment_id="x"), "experiment_id mismatch"),
        (lambda a: a.update(milestone="x"), "milestone mismatch"),
        (lambda a: a.update(spec_refs=["x"]), "spec_refs mismatch"),
        (lambda a: a.update(random_seed=1), "random_seed mismatch"),
        (lambda a: a.update(inference_substrate="x"), "inference_substrate mismatch"),
        (lambda a: a.update(kv260_precondition_command="x"), "kv260_precondition_command mismatch"),
        (lambda a: a.update(field_principles={}), "field_principles mismatch"),
        (lambda a: a.update(honest_verdict="blocked_x"), "honest_verdict prefix mismatch"),
        (lambda a: a.update(no_speedup_claim=False), "no_speedup_claim mismatch"),
        (lambda a: a.update(hardware_speedup_claimed=True), "hardware_speedup_claimed mismatch"),
        (
            lambda a: a.update(kv260_host_block_devices_touched=True),
            "kv260_host_block_devices_touched mismatch",
        ),
        (lambda a: a.update(conductor_modified=True), "conductor_modified mismatch"),
        (lambda a: a.update(kv260_status="reachable /dev/mmcblk0"), "host storage marker present"),
        (lambda a: a.update(boards_reachable_count=9), "boards_reachable_count mismatch"),
    ],
)
def test_top_level_expect_failures(mutate, needle):
    artifact = _build_blocked()
    mutate(artifact)
    errors = mod.artifact_schema_errors(artifact)
    assert any(needle in e for e in errors)


def test_checksum_mismatch_detected():
    artifact = _build_blocked()
    artifact["duration_s"] = artifact["duration_s"] + 1.0
    errors = mod.artifact_schema_errors(artifact)
    assert any("checksum mismatch" in e for e in errors)


def test_validate_polarfire_status_branches():
    artifact = _build_blocked()
    artifact["polarfire_status"] = "not a dict"
    assert any("polarfire_status must be a dict" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _build_blocked()
    artifact["polarfire_status"] = dict(
        artifact["polarfire_status"], polarfire_workload_validated=True
    )
    assert any(
        "polarfire_workload_validated must be false" in e
        for e in mod.artifact_schema_errors(artifact)
    )

    artifact = _build_blocked()
    artifact["polarfire_status"] = dict(artifact["polarfire_status"], workload_hash="nothex")
    assert any("polarfire workload_hash invalid" in e for e in mod.artifact_schema_errors(artifact))


def test_validate_gatemate_status_branches():
    artifact = _build_blocked()
    artifact["gatemate_status"] = "not a dict"
    assert any("gatemate_status must be a dict" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _build_blocked()
    artifact["gatemate_status"] = dict(artifact["gatemate_status"], reachable="no")
    assert any(
        "gatemate_status reachable must be bool" in e for e in mod.artifact_schema_errors(artifact)
    )

    artifact = _build_blocked()
    artifact["gatemate_status"] = dict(artifact["gatemate_status"], status="wrong_label")
    assert any(
        "gatemate blocked status label mismatch" in e for e in mod.artifact_schema_errors(artifact)
    )

    artifact = _build_blocked()
    artifact["gatemate_status"] = dict(
        artifact["gatemate_status"], blocked_reason="blocked_other_x"
    )
    assert any(
        "gatemate blocked_reason mismatch" in e for e in mod.artifact_schema_errors(artifact)
    )


def test_validate_new_angles_branches():
    artifact = _build_blocked()
    artifact["new_diagnostic_angles_tried_this_milestone"] = []
    assert any("non-empty list" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _build_blocked()
    artifact["new_diagnostic_angles_tried_this_milestone"] = [{"angle": "x"}]
    assert any("angle/method/finding" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _build_blocked()
    artifact["new_diagnostic_angles_tried_this_milestone"] = [
        {"angle": "dirtyjtag_probe_firmware_version", "method": "m", "finding": "f"},
        {"angle": "cable_or_port_swap", "method": "m", "finding": "f"},
    ]
    assert any(
        "missing debug_level_raw_idcode_capture" in e for e in mod.artifact_schema_errors(artifact)
    )

    artifact = _build_blocked()
    artifact["new_diagnostic_angles_tried_this_milestone"] = [
        {"angle": "debug_level_raw_idcode_capture", "method": "m", "finding": "f"},
        {"angle": "cable_or_port_swap", "method": "m", "finding": "f"},
    ]
    assert any(
        "missing dirtyjtag_probe_firmware_version" in e
        for e in mod.artifact_schema_errors(artifact)
    )

    artifact = _build_blocked()
    artifact["new_diagnostic_angles_tried_this_milestone"] = [
        {"angle": "debug_level_raw_idcode_capture", "method": "m", "finding": "f"},
        {"angle": "dirtyjtag_probe_firmware_version", "method": "m", "finding": "f"},
    ]
    assert any("missing cable_or_port_swap" in e for e in mod.artifact_schema_errors(artifact))


def test_validate_preconditions_branches():
    artifact = _build_blocked()
    artifact["preconditions_checked"] = artifact["preconditions_checked"][:2]
    assert any("must have 3 entries" in e for e in mod.artifact_schema_errors(artifact))

    artifact = _build_blocked()
    bad = copy.deepcopy(artifact["preconditions_checked"])
    bad[0] = {"board": "kv260"}
    artifact["preconditions_checked"] = bad
    assert any("board/resource/available" in e for e in mod.artifact_schema_errors(artifact))


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def test_run_experiment_writes_artifact(tmp_path):
    runner = RecordingRunner(_blocked_probes())
    out = mod.run_experiment(
        repo_root=tmp_path, command_runner=runner, clock=FlatClock(), run_date="20260704"
    )
    assert out == tmp_path / mod.OUTPUT_REL_PATH
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["gatemate_diagnostic_narrowed_to"] == "cable_or_port"
    mod.validate_artifact(written)


def test_main_prints_and_returns_zero(tmp_path, monkeypatch, capsys):
    def fake_run_experiment(*, repo_root, run_date):
        runner = RecordingRunner(_blocked_probes())
        artifact = mod.build_artifact(command_runner=runner, clock=FlatClock(), run_date=run_date)
        return mod.write_artifact(tmp_path, artifact)

    monkeypatch.setattr(mod, "run_experiment", fake_run_experiment)
    rc = mod.main(["--date", "20260704"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gatemate_diagnostic_narrowed_to: cable_or_port" in out
    assert "gatemate_raw_idcode: 0xffffffff" in out
    assert "debug_level_raw_idcode_capture" in out


def test_spec_defines_req_and_scenario():
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-HW-5217" in text
    assert "### SCENARIO-HW-5217" in text
