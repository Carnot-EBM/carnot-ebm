"""Tests for Exp 3866 GateMate Ising tile terminal flash v2.

Spec refs: REQ-HW-109, SCENARIO-HW-109.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.experiment_3866_gatemate_ising_tile_flash_v2 import (
    ARTIFACT_FILENAME,
    FIELD_PRINCIPLES,
    REQUIRED_ARTIFACT_FIELDS,
    CommandResult,
    _failure_reason,
    _flash_pending_reason,
    _parse_board_detect,
    _parse_fmax_mhz,
    _parse_utilization,
    _verdict,
    build_artifact,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

DETECT_OK = (
    "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
    "index 0:\n"
    "\tidcode 0x20000001\n"
    "\tmanufacturer colognechip\n"
    "\tfamily GateMate Series\n"
    "\tmodel  GM1Ax\n"
)


def _clock(values: list[float]):
    state = iter(values)

    def monotonic() -> float:
        return next(state)

    return monotonic


def _paths() -> dict[str, str]:
    return {
        "yosys": "/suite/bin/yosys",
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "gmpack": "/suite/bin/gmpack",
        "openFPGALoader": "/suite/bin/openFPGALoader",
    }


def _which_from(paths: dict[str, str]):
    def which(name: str) -> str | None:
        return paths.get(name)

    return which


def _write_rtl(repo_root: Path) -> Path:
    rtl = repo_root / "rtl" / "gatemate_ising_n16.v"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text(
        "module gatemate_ising_n16(input wire S_AXI_ACLK, output wire [15:0] out);\n"
        "reg [15:0] state;\n"
        "always @(posedge S_AXI_ACLK) state <= state + 16'h1;\n"
        "assign out = state;\n"
        "endmodule\n",
        encoding="utf-8",
    )
    return rtl


def _tool_runner(
    *,
    detect_text: str = DETECT_OK,
    detect_rc: int = 0,
    flash_rc: int = 0,
    synthesis_rc: int = 0,
):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call == ("yosys", "-V"):
            return CommandResult(0, "Yosys 0.64+149\n", "")
        if call == ("yosys", "-Q", "-p", "help synth_gatemate"):
            return CommandResult(0, "synth_gatemate command help\n", "")
        if call == ("nextpnr-himbaechel", "--help"):
            return CommandResult(0, "Usage: nextpnr-himbaechel --device CCGM1A1\n", "")
        if call == ("openFPGALoader", "-c", "dirtyJtag", "--detect"):
            return CommandResult(detect_rc, detect_text, "")
        if call[0] == "yosys" and "synth_gatemate" in " ".join(args):
            if synthesis_rc != 0:
                return CommandResult(synthesis_rc, "", "synthesis failed\n")
            json_out = Path(args[-1].split("-json ", 1)[1].split(";")[0])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text('{"modules":{}}\n', encoding="utf-8")
            return CommandResult(
                0,
                "Number of cells:              251\n"
                "     CC_DFF                  53\n"
                "     CC_LUT2                 19\n"
                "     CC_LUT3                 10\n"
                "     CC_LUT4                 20\n",
                "",
            )
        if call[0] == "nextpnr-himbaechel" and "--device" in call:
            out_vopt = next(item for item in call if item.startswith("out="))
            cfg_path = Path(out_vopt.removeprefix("out="))
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("cfg-bitstream-text\n", encoding="utf-8")
            pnr_json = Path(call[call.index("--write") + 1])
            pnr_json.write_text('{"pnr":true}\n', encoding="utf-8")
            return CommandResult(
                0,
                "Info: Max frequency for clock 'S_AXI_ACLK': 48.50 MHz\n"
                "Info:              CPE_LT:      76/  40960     0%\n"
                "Info:              CPE_FF:      53/  40960     0%\n",
                "",
            )
        if call[0] == "gmpack":
            Path(args[2]).write_bytes(b"gate-mate-bitstream")
            return CommandResult(0, "Packed bitstream\n", "")
        if call[:4] == ("openFPGALoader", "-c", "dirtyJtag", "-b"):
            return CommandResult(
                flash_rc, "write SRAM OK\n" if flash_rc == 0 else "", "flash failed\n"
            )
        raise AssertionError(f"unexpected command: {call}")

    return run, calls


def test_exp3866_spec_entry_present() -> None:
    """REQ-HW-109: the FPGA capability spec anchors the terminal flash artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-109" in spec
    assert "SCENARIO-HW-109" in spec
    assert ARTIFACT_FILENAME in spec


def test_parse_helpers_accept_real_gateMate_shapes() -> None:
    """SCENARIO-HW-109: parser helpers extract IDCODE, utilization, and Fmax."""
    assert _parse_board_detect(CommandResult(0, DETECT_OK, "")) is True
    utilization = _parse_utilization(
        "     CC_DFF                  53\n"
        "     CC_LUT2                 19\n"
        "     CC_LUT3                 10\n"
        "     CC_LUT4                 20\n",
        "Info: CPE_LT:      76/  40960     0%\nInfo: CPE_FF:      53/  40960     0%\n",
    )
    assert utilization["lut_count"] == 49
    assert utilization["dff_count"] == 53
    assert utilization["nextpnr_resources"]["CPE_LT"]["used"] == 76
    count_first = _parse_utilization("       5   CC_LUT4\n", "")
    assert count_first["lut_count"] == 5
    assert _parse_fmax_mhz("Info: Max frequency for clock 'clk': 48.50 MHz\n") == 48.5
    assert _parse_fmax_mhz("no clock reported\n") is None


def test_verdict_helpers_cover_blocked_and_failure_labels() -> None:
    """REQ-HW-109: verdict helpers keep blocked and failed states explicit."""
    assert (
        _failure_reason("pnr", CommandResult(0, "", "unsupported CC_LUT4"))
        == "pnr_unsupported_cell"
    )
    assert _failure_reason("pack", CommandResult(2, "", "boom")) == "pack_returncode_2"
    assert _failure_reason("pack", CommandResult(0, "", "")) == "pack_failed"
    assert (
        _flash_pending_reason(CommandResult(0, "", "")) == "openfpgaloader_no_acceptance_evidence"
    )
    assert (
        _verdict(
            blocker="blocked_gatemate_board_not_detected",
            build_failure_reason="",
            synth_pnr_pack_succeeded=False,
            flash_result=None,
            flashed=False,
            utilization=None,
            fmax_mhz=None,
            sample_timing_us=None,
        )
        == "blocked_gatemate_board_not_detected"
    )


def test_exp3866_success_artifact_has_terminal_fields(tmp_path: Path) -> None:
    """SCENARIO-HW-109: preconditions pass, build/pack/flash succeeds, timing is not fabricated."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([10.0, 13.25]),
    )

    assert [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact] == []
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_ising_tile_n16_flashed_terminal"
    )
    assert artifact["gatemate_bitstream_flashed"] is True
    assert artifact["synth_pnr_pack_succeeded"] is True
    assert artifact["lut_dff_utilization"]["lut_count"] == 49
    assert artifact["lut_dff_utilization"]["dff_count"] == 53
    assert artifact["fmax_mhz"] == 48.5
    assert artifact["sample_timing_us"] is None
    assert "not fabricating" in artifact["sample_timing_note"]
    assert artifact["run_duration_s"] == 3.25
    assert artifact["duration_s"] == 3.25
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert set(FIELD_PRINCIPLES) <= set(artifact["field_provenance"])
    assert all(
        artifact["field_provenance"][field]["principle"] for field in REQUIRED_ARTIFACT_FIELDS
    )
    assert any(
        "nextpnr-himbaechel --device CCGM1A1" in entry["command"]
        for entry in artifact["command_transcript"]
    )
    assert not any("nextpnr-gatemate" in " ".join(call) for call in calls)


def test_exp3866_missing_himbaechel_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-109: missing himbaechel precondition exits before fabrication."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner()
    paths = _paths()
    del paths["nextpnr-himbaechel"]

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(paths),
        monotonic=_clock([1.0, 1.5]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_himbaechel_missing"
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["synth_pnr_pack_succeeded"] is False
    assert artifact["lut_dff_utilization"] is None
    assert artifact["fmax_mhz"] is None
    assert artifact["preconditions_checked"][1]["resource"] == "nextpnr-himbaechel"
    assert artifact["preconditions_checked"][1]["available"] is False
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3866_missing_yosys_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-109: missing yosys emits the specific blocked verdict."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner()
    paths = _paths()
    del paths["yosys"]

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(paths),
        monotonic=_clock([1.0, 1.25]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_yosys_missing"
    assert artifact["preconditions_checked"][0]["available"] is False
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3866_board_not_detected_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-109: board detect failure writes blocked_board_not_detected and stops."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner(detect_text="no jtag chain found\n", detect_rc=1)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([2.0, 2.75]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_board_not_detected"
    assert artifact["preconditions_checked"][-1]["resource"] == "gatemate_board_detect"
    assert artifact["preconditions_checked"][-1]["available"] is False
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3866_missing_rtl_records_build_failure(tmp_path: Path) -> None:
    """REQ-HW-109: missing RTL fails as a build artifact, not fabricated hardware."""
    run_command, _calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([2.0, 3.0]),
    )

    assert artifact["honest_verdict"] == (
        "complete: gatemate_synth_pnr_pack_failed_rtl_missing_lutunknown_fmaxunknown"
    )
    assert artifact["synth_pnr_pack_succeeded"] is False
    assert artifact["gatemate_bitstream_flashed"] is False


def test_exp3866_synthesis_failure_is_complete_not_fabricated(tmp_path: Path) -> None:
    """SCENARIO-HW-109: a failed build stage records a complete failed-build verdict."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner(synthesis_rc=2)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([3.0, 4.0]),
    )

    assert artifact["honest_verdict"] == (
        "complete: gatemate_synth_pnr_pack_failed_synthesis_returncode_2_lutunknown_fmaxunknown"
    )
    assert artifact["synth_pnr_pack_succeeded"] is False
    assert artifact["command_transcript"][0]["stage"] == "synthesis"


def test_exp3866_pack_succeeds_flash_fails_is_partial(tmp_path: Path) -> None:
    """SCENARIO-HW-109: synth/P&R/pack success with flash failure is a partial verdict."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner(flash_rc=1)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([3.0, 8.0]),
    )

    assert artifact["synth_pnr_pack_succeeded"] is True
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["honest_verdict"].startswith(
        "complete: gatemate_synth_pnr_pack_succeeded_flash_pending_openfpgaloader_returncode_1"
    )
    assert artifact["lut_dff_utilization"]["lut_count"] == 49
    assert artifact["fmax_mhz"] == 48.5


def test_exp3866_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-109: run_experiment writes the requested JSON deliverable."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner()
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([4.0, 6.0]),
    )

    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert destination.name == ARTIFACT_FILENAME
