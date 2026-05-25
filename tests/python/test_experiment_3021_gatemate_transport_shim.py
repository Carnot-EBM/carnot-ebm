"""Tests for Exp 3021 GateMate RTL/CCF host-visible transport shim diagnosis.

Spec refs: REQ-HW-084, SCENARIO-HW-084.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3021_gatemate_rtl_ccf_host_visible_transport_shim import (
    ARTIFACT_FILENAME,
    CommandResult,
    _extract_usb_bus_device,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"
REQUIRED_FIELDS = (
    "gatemate_transport_rtl_ready",
    "host_visible_io_plan_ready",
    "preconditions_checked",
    "board_detected",
    "rtl_paths",
    "ccf_paths",
    "io_transport_path",
    "simulation_or_lint_passed",
    "pnr_or_synthesis_attempted",
    "transcript_paths",
    "sampler_claim_made",
    "speedup_claim_made",
    "honest_verdict",
)


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _tool_paths() -> dict[str, str]:
    return {
        "yosys": "/suite/bin/yosys",
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "gmpack": "/suite/bin/gmpack",
        "openFPGALoader": "/suite/bin/openFPGALoader",
        "lsusb": "/usr/bin/lsusb",
    }


def _write_package(
    repo_root: Path,
    *,
    ccf_text: str | None = None,
    rtl_text: str | None = None,
) -> None:
    hw_dir = repo_root / "hardware" / "gatemate"
    hw_dir.mkdir(parents=True, exist_ok=True)
    (hw_dir / "ising_n16_gatemate.ccf").write_text(
        ccf_text
        if ccf_text is not None
        else (
            "# GateMate build-only constraints\n"
            "# no physical Pin_in/Pin_out locations\n"
            "# allow-unconstrained\n"
        ),
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate.v").write_text(
        rtl_text
        if rtl_text is not None
        else (
            "module ising_n16_gatemate(input clk, output reg done, "
            "output reg [15:0] spin_out); endmodule\n"
        ),
        encoding="utf-8",
    )
    (hw_dir / "ising_n16_gatemate_test_vector.json").write_text(
        json.dumps({"schema": "carnot.gatemate.ising_n16_test_vector.v1"}),
        encoding="utf-8",
    )
    bitstream = (
        repo_root
        / "build"
        / "gatemate"
        / "experiment_2956_gatemate_n16"
        / "ising_n16_gatemate.bit"
    )
    bitstream.parent.mkdir(parents=True, exist_ok=True)
    bitstream.write_bytes(b"gatemate-test-bitstream")


def _runner(*, lint_returncode: int = 0):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple(args)
        calls.append(call)
        exe = Path(args[0]).name
        if exe == "yosys" and args[1:] == ["-V"]:
            return CommandResult(0, "Yosys 0.64+149\n", "")
        if exe == "nextpnr-himbaechel":
            return CommandResult(0, "nextpnr-himbaechel 0.10\n", "")
        if exe == "gmpack":
            return CommandResult(1, "", "Open Source Tools for GateMate FPGAs Version v1.13\n")
        if exe == "openFPGALoader" and args[1:] == ["-V"]:
            return CommandResult(0, "openFPGALoader v1.1.1\n", "")
        if exe == "lsusb":
            return CommandResult(0, "1209:c0ca (bus 3, device 14) path: 2.3\n", "")
        if exe == "openFPGALoader" and args[1:] == ["-c", "dirtyJtag", "--detect"]:
            return CommandResult(0, "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n", "")
        if exe == "yosys" and args[1] == "-p":
            return CommandResult(lint_returncode, "Checking module ising_n16_gatemate\n", "")
        raise AssertionError(f"unexpected command: {args}")

    return run, calls


def test_req_hw_084_spec_entry_present() -> None:
    """REQ-HW-084: the FPGA spec anchors the Exp 3021 artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-HW-084" in spec
    assert "SCENARIO-HW-084" in spec
    assert ARTIFACT_FILENAME in spec


def test_req_hw_084_missing_toolchain_or_board_stops_before_lint(tmp_path: Path) -> None:
    """REQ-HW-084: missing setup preconditions emit the required blocked verdict."""
    _write_package(tmp_path)
    runner, calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from({}),
        monotonic=_clock([1.0, 1.25]),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "blocked_gatemate_precondition_missing"
    assert artifact["preconditions_checked"] is True
    assert artifact["board_detected"] is False
    assert artifact["gatemate_transport_rtl_ready"] is False
    assert artifact["host_visible_io_plan_ready"] is False
    assert artifact["simulation_or_lint_passed"] is False
    assert artifact["pnr_or_synthesis_attempted"] is False
    assert artifact["sampler_claim_made"] is False
    assert artifact["speedup_claim_made"] is False
    assert calls == []


def test_scenario_hw_084_build_only_ccf_blocks_transport_but_runs_lint(
    tmp_path: Path,
) -> None:
    """SCENARIO-HW-084: internal done/spin_out ports are not host-visible IO."""
    _write_package(tmp_path)
    usb_node = tmp_path / "dev" / "bus" / "usb" / "003" / "014"
    usb_node.parent.mkdir(parents=True)
    usb_node.write_bytes(b"")
    runner, calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        monotonic=_clock([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5]),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert artifact["honest_verdict"] == "complete: blocked_gatemate_transport_pinout_missing"
    assert artifact["preconditions_checked"] is True
    assert artifact["board_detected"] is True
    assert artifact["gatemate_transport_rtl_ready"] is False
    assert artifact["host_visible_io_plan_ready"] is False
    assert artifact["io_transport_path"].startswith("blocked:")
    assert "Pin_out" in artifact["blockers"][0]
    assert artifact["simulation_or_lint_passed"] is True
    assert artifact["pnr_or_synthesis_attempted"] is False
    assert artifact["transcript_sha256"].keys() == set(artifact["transcript_paths"])
    assert all("-b" not in call for call in calls for call in call)


def test_req_hw_084_bound_status_without_reader_is_not_host_plan_ready(
    tmp_path: Path,
) -> None:
    """REQ-HW-084: physical status output still needs a concrete reader path."""
    _write_package(tmp_path, ccf_text="Pin_out done Loc = IO_EB_B7\n")
    usb_node = tmp_path / "dev" / "bus" / "usb" / "003" / "014"
    usb_node.parent.mkdir(parents=True)
    usb_node.write_bytes(b"")
    runner, _calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        monotonic=_clock([20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5]),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert artifact["gatemate_transport_rtl_ready"] is True
    assert artifact["host_visible_io_plan_ready"] is False
    assert artifact["io_transport_path"] == "blocked:gatemate_reader_missing_for_done"
    assert "reader" in artifact["blockers"][0]
    assert artifact["honest_verdict"] == "complete: blocked_gatemate_transport_reader_missing"


def test_req_hw_084_bound_uart_with_reader_marks_plan_ready(tmp_path: Path) -> None:
    """REQ-HW-084: a bound status output plus reader is the minimum ready plan."""
    _write_package(
        tmp_path,
        ccf_text="Pin_out uart_tx Loc = IO_EB_B7\n",
        rtl_text=(
            "module ising_n16_gatemate(input clk, output uart_tx, output done, "
            "output [15:0] spin_out); endmodule\n"
        ),
    )
    reader = tmp_path / "scripts" / "gatemate_uart_reader.py"
    reader.parent.mkdir(parents=True)
    reader.write_text("import serial\nserial.Serial('/dev/ttyUSB0')\n", encoding="utf-8")
    usb_node = tmp_path / "dev" / "bus" / "usb" / "003" / "014"
    usb_node.parent.mkdir(parents=True)
    usb_node.write_bytes(b"")
    runner, _calls = _runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=runner,
        which_func=_which_from(_tool_paths()),
        monotonic=_clock([30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5]),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    assert artifact["gatemate_transport_rtl_ready"] is True
    assert artifact["host_visible_io_plan_ready"] is True
    assert artifact["io_transport_path"].startswith("uart_tx:")
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"] == "complete: gatemate_host_visible_transport_plan_ready"


def test_req_hw_084_usb_parser_accepts_standard_and_compact_lsusb() -> None:
    """REQ-HW-084: board contact evidence can come from both lsusb formats."""
    compact = "1209:c0ca (bus 3, device 14) path: 2.3"
    standard = "Bus 003 Device 014: ID 1209:c0ca Generic DirtyJTAG"
    multiline = "1d6b:0003 (bus 8, device 1)\n1209:c0ca (bus 3, device 14) path: 2.3"

    assert _extract_usb_bus_device(compact) == ("003", "014")
    assert _extract_usb_bus_device(standard) == ("003", "014")
    assert _extract_usb_bus_device(multiline) == ("003", "014")
    assert _extract_usb_bus_device("no board") == ("", "")


def test_scenario_hw_084_run_experiment_writes_required_json(tmp_path: Path) -> None:
    """SCENARIO-HW-084: run_experiment writes the v1 terminal artifact."""
    _write_package(tmp_path)
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=lambda args, timeout_s: CommandResult(99, "", "unexpected"),
        which_func=_which_from({}),
        monotonic=_clock([40.0, 40.5]),
        usb_device_root=tmp_path / "dev" / "bus" / "usb",
    )

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_FIELDS if field not in loaded]
    assert missing == []
    assert loaded == artifact
    assert destination.name == ARTIFACT_FILENAME
