"""Tests for Exp 3889 GateMate continuity corrigendum.

Spec refs: REQ-HW-3889, SCENARIO-HW-3889.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_3889_gatemate_continuity_corrigendum as exp


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
BITSTREAM_BYTES = b"exp3889-gatemate-bitstream"


def _clock(step: float = 0.25):
    state = {"value": 0.0}

    def monotonic() -> float:
        current = state["value"]
        state["value"] += step
        return current

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


def _write_rtl(repo_root: Path) -> None:
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


def _tool_runner(
    *,
    detect_text: str = DETECT_OK,
    detect_rc: int = 0,
    synthesis_rc: int = 0,
    flash_rc: int = 0,
    readback_supported: bool = False,
    readback_matches: bool = True,
):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> exp.CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call == ("yosys", "-V"):
            return exp.CommandResult(0, "Yosys 0.64+149\n", "")
        if call == ("yosys", "-Q", "-p", "help synth_gatemate"):
            return exp.CommandResult(0, "synth_gatemate command help\n", "")
        if call == ("nextpnr-himbaechel", "--help"):
            return exp.CommandResult(0, "Usage: nextpnr-himbaechel --device CCGM1A1\n", "")
        if call == ("openFPGALoader", "-c", "dirtyJtag", "--detect"):
            return exp.CommandResult(detect_rc, detect_text, "")
        if call[0] == "yosys" and "synth_gatemate" in " ".join(args):
            if synthesis_rc:
                return exp.CommandResult(synthesis_rc, "", "synthesis failed\n")
            json_out = Path(args[-1].split("-json ", 1)[1].split(";")[0])
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text('{"modules":{}}\n', encoding="utf-8")
            return exp.CommandResult(
                0,
                "Number of cells:              251\n"
                "     CC_DFF                  53\n"
                "     CC_LUT2                 19\n",
                "",
            )
        if call[0] == "nextpnr-himbaechel" and "--device" in call:
            out_vopt = next(item for item in call if item.startswith("out="))
            cfg_path = Path(out_vopt.removeprefix("out="))
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("cfg-bitstream-text\n", encoding="utf-8")
            pnr_json = Path(call[call.index("--write") + 1])
            pnr_json.write_text('{"pnr":true}\n', encoding="utf-8")
            return exp.CommandResult(
                0,
                "Info: Max frequency for clock 'S_AXI_ACLK': 50.00 MHz\n"
                "Info:              CPE_LT:      19/  40960     0%\n"
                "Info:              CPE_FF:      53/  40960     0%\n",
                "",
            )
        if call[0] == "gmpack":
            Path(args[2]).write_bytes(BITSTREAM_BYTES)
            return exp.CommandResult(0, "Packed bitstream\n", "")
        if call[:4] == ("openFPGALoader", "-c", "dirtyJtag", "-b"):
            return exp.CommandResult(
                flash_rc,
                "Load SRAM via JTAG: 100.00%\nDone\n" if flash_rc == 0 else "",
                "flash failed\n" if flash_rc else "",
            )
        if call == ("openFPGALoader", "--help"):
            help_text = "--readback readback.bin\n" if readback_supported else "--dump-flash --verify\n"
            return exp.CommandResult(0, help_text, "")
        if call[:3] == ("openFPGALoader", "-c", "dirtyJtag") and "--readback" in call:
            readback_path = Path(call[-1])
            readback_path.parent.mkdir(parents=True, exist_ok=True)
            readback_path.write_bytes(BITSTREAM_BYTES if readback_matches else b"different")
            return exp.CommandResult(0, "readback complete\n", "")
        raise AssertionError(f"unexpected command: {call}")

    return run, calls


def test_exp3889_spec_entry_present() -> None:
    """REQ-HW-3889: the FPGA capability spec anchors the corrigendum artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-3889" in spec
    assert "SCENARIO-HW-3889" in spec
    assert exp.ARTIFACT_FILENAME in spec


def test_exp3889_readback_decision_helper() -> None:
    """SCENARIO-HW-3889: readback support is based on tool help, not assumption."""
    supported, reason = exp._readback_decision(exp.CommandResult(0, "--readback file\n", ""))
    assert supported is True
    assert "--readback" in reason
    unsupported, reason = exp._readback_decision(exp.CommandResult(0, "--dump-flash --verify\n", ""))
    assert unsupported is False
    assert "SPI-flash" in reason
    unknown, reason = exp._readback_decision(exp.CommandResult(0, "ordinary help\n", ""))
    assert unknown is False
    assert "does not advertise" in reason


def test_exp3889_verdict_helper_preserves_blocker_and_unknown_failure() -> None:
    """REQ-HW-3889: verdict helper keeps blocker and failed-flow prefixes explicit."""
    assert (
        exp._verdict(
            blocker="blocked_gatemate_board_unreachable",
            flashed=False,
            readback_supported=False,
            readback_verified=False,
            fmax_mhz=None,
        )
        == "blocked_gatemate_board_unreachable"
    )
    assert exp._verdict(
        blocker="",
        flashed=False,
        readback_supported=False,
        readback_verified=False,
        fmax_mhz=None,
    ) == "blocked_gatemate_flash_flow_failed_unknown"


def test_exp3889_clean_terminal_when_readback_unsupported(tmp_path: Path) -> None:
    """SCENARIO-HW-3889: unsupported readback is clean when flash succeeds and timers differ."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner(readback_supported=False)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    missing = [field for field in exp.REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_continuity_CLEAN_terminal_fmax50.00_readbackunsupported_no_tautology"
    )
    assert artifact["gatemate_bitstream_flashed"] is True
    assert artifact["readback_supported"] is False
    assert artifact["readback_verified"] is False
    assert artifact["duration_s"] != artifact["run_duration_s"]
    assert artifact["duration_s"] > artifact["run_duration_s"] > 0
    assert artifact["lut_used"] == 19
    assert artifact["dff_used"] == 53
    assert artifact["sample_timing_us"] == pytest.approx(0.02)
    assert artifact["sample_timing_basis"] == "fmax_one_clock_spin_update"
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not any("--readback" in call for call in calls)
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict)
        assert artifact["field_provenance"][field]["principle"]


def test_exp3889_clean_terminal_when_readback_hash_matches(tmp_path: Path) -> None:
    """REQ-HW-3889: supported readback verifies only when its hash matches the bitstream."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner(readback_supported=True, readback_matches=True)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert artifact["readback_supported"] is True
    assert artifact["readback_attempted"] is True
    assert artifact["readback_verified"] is True
    assert artifact["readback_sha256"] == artifact["bitstream_sha256"]
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_continuity_CLEAN_terminal_fmax50.00_readbacktrue_no_tautology"
    )
    assert any("--readback" in call for call in calls)


def test_exp3889_supported_readback_mismatch_is_inconclusive(tmp_path: Path) -> None:
    """SCENARIO-HW-3889: supported readback that does not verify remains caveated."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner(readback_supported=True, readback_matches=False)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert artifact["gatemate_bitstream_flashed"] is True
    assert artifact["readback_supported"] is True
    assert artifact["readback_verified"] is False
    assert artifact["readback_sha256"] != artifact["bitstream_sha256"]
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_continuity_flashed_readback_inconclusive_fmax50.00"
    )


def test_exp3889_missing_tool_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-3889: missing toolchain emits the required blocker and stops."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner()
    paths = _paths()
    del paths["nextpnr-himbaechel"]

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(paths),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["readback_supported"] is False
    assert artifact["run_duration_s"] == 0.0
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3889_board_unreachable_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-3889: failed DirtyJTAG detect emits the required board blocker."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner(detect_text="no jtag chain found\n", detect_rc=1)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_board_unreachable"
    assert artifact["preconditions_checked"][0]["resource"] == "gatemate_toolchain"
    assert artifact["preconditions_checked"][-1]["resource"] == "gatemate_board_detect"
    assert artifact["preconditions_checked"][-1]["available"] is False
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3889_synthesis_failure_records_failed_flow(tmp_path: Path) -> None:
    """SCENARIO-HW-3889: a build-stage failure stops before flash or readback."""
    _write_rtl(tmp_path)
    run_command, calls = _tool_runner(synthesis_rc=2)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == (
        "blocked_gatemate_flash_flow_failed_synthesis_returncode_2"
    )
    assert artifact["synth_pnr_pack_succeeded"] is False
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["readback_attempted"] is False
    assert not any(call[:4] == ("openFPGALoader", "-c", "dirtyJtag", "-b") for call in calls)


def test_exp3889_flash_failure_records_failed_flow(tmp_path: Path) -> None:
    """SCENARIO-HW-3889: synth/P&R/pack success with flash failure is not terminal."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner(flash_rc=3)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert artifact["honest_verdict"] == (
        "blocked_gatemate_flash_flow_failed_openfpgaloader_returncode_3"
    )
    assert artifact["synth_pnr_pack_succeeded"] is True
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["run_duration_s"] > 0
    assert artifact["readback_supported"] is False


def test_exp3889_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-3889: run_experiment writes the requested JSON deliverable."""
    _write_rtl(tmp_path)
    run_command, _calls = _tool_runner(readback_supported=False)
    destination = tmp_path / "results" / exp.ARTIFACT_FILENAME

    artifact = exp.run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert on_disk == artifact
    assert destination.name == exp.ARTIFACT_FILENAME
