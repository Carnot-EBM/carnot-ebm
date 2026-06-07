"""Tests for Exp 3900 GateMate terminal confirmation.

Spec refs: REQ-HW-3900, SCENARIO-HW-3900.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot import experiment_3900_gatemate_terminal_confirmation as exp


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
BITSTREAM_BYTES = b"exp3900-gatemate-bitstream"


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


def _write_prior(repo_root: Path, content: bytes = b"prior-exp3889") -> str:
    prior = repo_root / "results" / "experiment_3889_gatemate_continuity_corrigendum.json"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


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


def test_exp3900_spec_entry_present() -> None:
    """REQ-HW-3900: the FPGA capability spec anchors terminal confirmation."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-3900" in spec
    assert "SCENARIO-HW-3900" in spec
    assert exp.ARTIFACT_FILENAME in spec


def test_exp3900_terminal_when_readback_unsupported(tmp_path: Path) -> None:
    """SCENARIO-HW-3900: unsupported readback can graduate when flash and smoke pass."""
    _write_rtl(tmp_path)
    prior_sha = _write_prior(tmp_path)
    run_command, calls = _tool_runner(readback_supported=False)

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock(),
    )

    assert [field for field in exp.REQUIRED_ARTIFACT_FIELDS if field not in artifact] == []
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_TERMINAL_reached_fmax50.00_readbackunsupported"
    )
    assert artifact["honest_verdict"].endswith("_can_graduate_to_opportunistic")
    assert artifact["terminal_state_reached"] is True
    assert artifact["gatemate_bitstream_flashed"] is True
    assert artifact["smoke_ok"] is True
    assert artifact["no_tautology"] is True
    assert artifact["readback_supported"] is False
    assert artifact["readback_verified"] is False
    assert artifact["duration_s"] != artifact["run_duration_s"]
    assert artifact["duration_s"] > artifact["run_duration_s"] > 0
    assert artifact["lut_used"] == 19
    assert artifact["dff_used"] == 53
    assert artifact["prior_corrigendum_sha256"] == prior_sha
    assert artifact["inference_substrate"] == "hardware_smoke"
    assert len(artifact["reproducibility_checksum"]) == 64
    assert not any("--readback" in call for call in calls)
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert not isinstance(artifact[field], dict)
        assert artifact["field_provenance"][field]["principle"]


def test_exp3900_terminal_when_readback_hash_matches(tmp_path: Path) -> None:
    """REQ-HW-3900: supported readback graduates only when it verifies the bitstream."""
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
    assert artifact["terminal_state_reached"] is True
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_TERMINAL_reached_fmax50.00_readbacktrue"
    )
    assert any("--readback" in call for call in calls)


def test_exp3900_supported_readback_mismatch_stays_mandatory(tmp_path: Path) -> None:
    """SCENARIO-HW-3900: supported readback mismatch remains mandatory."""
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
    assert artifact["terminal_state_reached"] is False
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_flashed_readback_inconclusive_fmax50.00"
    )
    assert artifact["honest_verdict"].endswith("_stays_mandatory")


def test_exp3900_timer_tautology_blocks_terminal_graduation() -> None:
    """REQ-HW-3900: equal duration timers keep terminal_state_reached false."""
    base = {
        "honest_verdict": "success: prior",
        "duration_s": 1.0,
        "run_duration_s": 1.0,
        "readback_verified": False,
        "readback_supported": False,
        "gatemate_bitstream_flashed": True,
        "synth_pnr_pack_succeeded": True,
        "fmax_mhz": 50.0,
        "lut_used": 19,
        "dff_used": 53,
        "preconditions_checked": [],
        "reproducibility_checksum": "old",
        "inference_substrate": "hardware_smoke",
        "command_transcript": [{"stage": "flash", "returncode": 0}],
        "field_provenance": {},
    }

    artifact = exp.confirm_terminal_artifact(base, prior_corrigendum_sha256="")

    assert artifact["no_tautology"] is False
    assert artifact["terminal_state_reached"] is False
    assert artifact["honest_verdict"].startswith(
        "success: gatemate_flashed_readback_inconclusive_fmax50.00"
    )


def test_exp3900_helper_edges_are_explicit() -> None:
    """SCENARIO-HW-3900: helper edge cases keep labels and smoke fallback deterministic."""
    assert exp._metric_label(None) == "unknown"
    assert exp._metric_label(7) == "7"
    assert exp._readback_label({"readback_verified": False, "readback_supported": True}) == "false"
    assert exp._smoke_ok(
        {
            "gatemate_bitstream_flashed": True,
            "synth_pnr_pack_succeeded": True,
            "command_transcript": [],
        }
    )


def test_exp3900_missing_tool_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-3900: missing toolchain emits the required blocker and stops."""
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
    assert artifact["terminal_state_reached"] is False
    assert artifact["gatemate_bitstream_flashed"] is False
    assert artifact["run_duration_s"] == 0.0
    assert not any(call[0] == "yosys" and "-l" in call for call in calls)


def test_exp3900_failed_flow_uses_blocked_prefix() -> None:
    """SCENARIO-HW-3900: non-flashed non-blocked base output is relabeled blocked."""
    base = {
        "honest_verdict": "complete: prior_failed",
        "duration_s": 2.0,
        "run_duration_s": 0.0,
        "readback_verified": False,
        "readback_supported": False,
        "gatemate_bitstream_flashed": False,
        "synth_pnr_pack_succeeded": False,
        "fmax_mhz": None,
        "lut_used": 0,
        "dff_used": 0,
        "preconditions_checked": [],
        "reproducibility_checksum": "old",
        "inference_substrate": "hardware_smoke",
        "command_transcript": [],
        "field_provenance": {},
    }

    artifact = exp.confirm_terminal_artifact(base, prior_corrigendum_sha256="")

    assert artifact["honest_verdict"] == "blocked_gatemate_flash_flow_failed_unknown"
    assert artifact["terminal_state_reached"] is False


def test_exp3900_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-3900: run_experiment writes the requested JSON deliverable."""
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
