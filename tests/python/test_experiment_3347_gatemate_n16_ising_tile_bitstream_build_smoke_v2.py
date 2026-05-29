"""Tests for Exp 3347 GateMate n=16 Ising tile build + detect smoke (v2).

Spec refs: REQ-HW-102, SCENARIO-HW-102.

These tests inject deterministic command runners so the full build + detect
control flow is exercised without touching real hardware or the OSS CAD Suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from carnot.experiment_2956_gatemate_n16_bitstream_build import (
    CommandResult,
    EXP2955_FILENAME,
)
from carnot.experiment_3347_gatemate_n16_ising_tile_bitstream_build_smoke_v2 import (
    ARTIFACT_FILENAME,
    RANDOM_SEED,
    _parse_detect,
    _parse_synthesis_log_cells,
    _relative_files,
    _resource_summary,
    build_smoke_artifact,
    run_detect,
    run_experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "files_updated",
    "yosys_version",
    "nextpnr_himbaechel_version",
    "gmpack_version",
    "dirtyjtag_detected",
    "build_succeeded",
    "bitstream_path",
    "bitstream_checksum",
    "resource_summary",
    "flash_smoke_status",
    "command_transcript",
    "blocked_reasons",
)

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


def _write_exp2955(repo_root: Path, *, ready: bool = True) -> Path:
    rtl = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    vector = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text(
        "module ising_n16_gatemate(input wire clk, output wire [15:0] spin_out);\n"
        "assign spin_out = 16'hace1;\nendmodule\n",
        encoding="utf-8",
    )
    ccf.write_text("# allow-unconstrained\n", encoding="utf-8")
    vector.write_text('{"schema":"test"}\n', encoding="utf-8")
    payload = {
        "gatemate_constraints_ready": ready,
        "rtl": {"path": str(rtl), "top_module": "ising_n16_gatemate", "n16": True},
        "constraints_file_paths": [str(ccf)],
        "test_vector_paths": [str(vector)],
        "top_module": "ising_n16_gatemate",
    }
    path = repo_root / "results" / EXP2955_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _tool_runner(*, detect_text: str = DETECT_OK, detect_rc: int = 0):
    """Inject a runner covering version probes, build stages, and detect."""
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call == ("yosys", "-V"):
            return CommandResult(0, "Yosys 0.64+197\n", "")
        if call == ("nextpnr-himbaechel", "--version"):
            return CommandResult(0, '"nextpnr-himbaechel" Version nextpnr-0.10\n', "")
        if call == ("gmpack", "--version"):
            return CommandResult(1, "", "GateMate Tools Version v1.13\n")
        if call == ("openFPGALoader", "-V"):
            return CommandResult(0, "openFPGALoader v1.1.1\n", "")
        if call[:2] == ("openFPGALoader", "-c"):
            return CommandResult(detect_rc, detect_text, "")
        joined = " ".join(args)
        if call[0] == "yosys" and "synth_gatemate" in joined:
            json_match = re.search(r"-json\s+([^;]+)", joined)
            assert json_match is not None
            target = Path(json_match.group(1))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"modules":{}}\n', encoding="utf-8")
            return CommandResult(
                0, "Number of cells:               5\n     CC_LUT2                   3\n", ""
            )
        if call[0] == "nextpnr-himbaechel" and "--json" in call:
            out_vopt = next(item for item in call if item.startswith("out="))
            cfg_path = Path(out_vopt.removeprefix("out="))
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("cfg\n", encoding="utf-8")
            Path(call[call.index("--write") + 1]).write_text('{"pnr":true}\n', encoding="utf-8")
            return CommandResult(
                0,
                "Info: Max frequency for clock 'clk': 48.50 MHz\n"
                "Info:   CPE: 42/1024 4%\n",
                "",
            )
        if call[0] == "gmpack":
            Path(args[2]).write_bytes(b"gate-mate-bitstream")
            return CommandResult(0, "Writing bitstream\n", "")
        raise AssertionError(f"unexpected command: {call}")

    return run, calls


def test_exp3347_spec_entry_present() -> None:
    """REQ-HW-102: the FPGA capability spec anchors the build+smoke artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-102" in spec
    assert "SCENARIO-HW-102" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp3347_build_and_detect_complete(tmp_path: Path) -> None:
    """SCENARIO-HW-102: a ready package builds and a detected board yields smoke."""
    _write_exp2955(tmp_path)
    run_command, calls = _tool_runner()

    artifact = build_smoke_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
    )

    assert [f for f in REQUIRED_FIELDS if f not in artifact] == []
    assert artifact["honest_verdict"] == "complete: gatemate_n16_bitstream_built_board_detected"
    assert artifact["build_succeeded"] is True
    assert artifact["dirtyjtag_detected"] is True
    assert artifact["flash_smoke_status"] == "detect_only_idcode_readback"
    assert artifact["inference_substrate"] == "hardware_build"
    assert artifact["random_seed"] == RANDOM_SEED
    assert artifact["blocked_reasons"] == []
    assert artifact["yosys_version"] == "Yosys 0.64+197"
    assert artifact["nextpnr_himbaechel_version"].startswith('"nextpnr-himbaechel"')
    assert artifact["gmpack_version"] == "GateMate Tools Version v1.13"
    assert (
        artifact["bitstream_checksum"]
        == hashlib.sha256(b"gate-mate-bitstream").hexdigest()
    )
    assert artifact["resource_summary"]["timing_summary"]["max_frequency_mhz"] == 48.5
    stages = [entry["stage"] for entry in artifact["command_transcript"]]
    assert stages == ["synthesis", "pnr", "pack", "detect"]
    assert "nextpnr-himbaechel --device CCGM1A1" in artifact["command_transcript"][1]["command"]
    assert not any("nextpnr-gatemate" in " ".join(call) for call in calls)
    assert len(artifact["reproducibility_checksum"]) == 64


def test_exp3347_build_succeeds_board_undetected(tmp_path: Path) -> None:
    """REQ-HW-102: an undetected board skips the smoke but the build still passes."""
    _write_exp2955(tmp_path)
    run_command, _calls = _tool_runner(detect_text="no devices\n", detect_rc=1)

    artifact = build_smoke_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
    )

    assert artifact["build_succeeded"] is True
    assert artifact["dirtyjtag_detected"] is False
    assert artifact["flash_smoke_status"] == "skipped_board_not_detected"
    assert artifact["honest_verdict"] == "complete: gatemate_n16_bitstream_built_board_undetected"
    assert artifact["blocked_reasons"] == []


def test_exp3347_missing_build_tool_blocks(tmp_path: Path) -> None:
    """REQ-HW-102: a missing build tool blocks the build and skips the smoke."""
    _write_exp2955(tmp_path)
    run_command, calls = _tool_runner()
    paths = _paths()
    del paths["gmpack"]

    artifact = build_smoke_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(paths),
    )

    assert artifact["build_succeeded"] is False
    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["flash_smoke_status"] == "skipped_build_blocked"
    assert artifact["dirtyjtag_detected"] is False
    assert artifact["blocked_reasons"] == ["missing toolchain: gmpack"]
    # The board is never probed when the build is blocked.
    assert not any(call[:2] == ("openFPGALoader", "-c") for call in calls)


def test_exp3347_openfpgaloader_missing_skips_smoke(tmp_path: Path) -> None:
    """REQ-HW-102: a built bitstream with no loader records a skipped smoke."""
    _write_exp2955(tmp_path)
    run_command, _calls = _tool_runner()
    paths = _paths()
    # build needs only yosys/nextpnr/gmpack; drop the loader after the build.
    # exp2956 requires openFPGALoader as a build precondition, so to exercise
    # the loader-missing detect branch directly we call run_detect.
    detect = run_detect(
        run_command=run_command,
        which_func=_which_from({k: v for k, v in paths.items() if k != "openFPGALoader"}),
    )
    assert detect["dirtyjtag_detected"] is False
    assert detect["flash_smoke_status"] == "skipped_openfpgaloader_missing"
    assert detect["detect_command"] == ""


def test_exp3347_parse_detect_requires_idcode_and_gatemate() -> None:
    """REQ-HW-102: detection only claimed when the transcript proves a GateMate."""
    assert _parse_detect(CommandResult(0, DETECT_OK, "")) is True
    # A bare adapter with no FPGA in the chain is not a detection.
    assert _parse_detect(CommandResult(0, "idcode 0x0\n", "")) is False
    # Non-zero return code is never a detection.
    assert _parse_detect(CommandResult(1, DETECT_OK, "")) is False


def test_exp3347_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-102: run_experiment writes the v2 deliverable JSON to disk."""
    _write_exp2955(tmp_path)
    run_command, _calls = _tool_runner()
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=run_command,
        which_func=_which_from(_paths()),
    )

    assert destination.exists()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["build_succeeded"] is True
    # files_updated paths are recorded relative to the repo root.
    assert all(not Path(p).is_absolute() for p in loaded["files_updated"])


REAL_SYNTH_LOG = (
    "COMMAND: yosys -p ...\nRETURNCODE: 0\nSTDOUT:\n"
    "   === ising_n16_gatemate ===\n"
    "     2217   CC_DFF\n"
    "       28   CC_LUT1\n"
    "      336   CC_LUT2\n"
    "STDERR:\n"
)


def test_exp3347_parse_synthesis_log_cells_real_format(tmp_path: Path) -> None:
    """REQ-HW-102: count-then-name yosys tally lines parse from the build log."""
    log = tmp_path / "synthesis.log"
    log.write_text(REAL_SYNTH_LOG, encoding="utf-8")
    counts = _parse_synthesis_log_cells([str(log)])
    assert counts == {"CC_DFF": 2217, "CC_LUT1": 28, "CC_LUT2": 336}
    # No paths and missing files both yield an empty (honest) tally.
    assert _parse_synthesis_log_cells([]) == {}
    assert _parse_synthesis_log_cells([str(tmp_path / "nope.log")]) == {}


def test_exp3347_resource_summary_augments_empty_utilization(tmp_path: Path) -> None:
    """REQ-HW-102: empty upstream utilization is backfilled from the synth log."""
    log = tmp_path / "synthesis.log"
    log.write_text(REAL_SYNTH_LOG, encoding="utf-8")
    build_artifact = {
        "timing_summary": {"max_frequency_mhz": 15.69},
        "utilization_summary": {"yosys_cell_counts": {}},
        "build_log_paths": [str(log)],
    }
    summary = _resource_summary(build_artifact)
    util = summary["utilization_summary"]
    assert util["yosys_cell_counts"] == {"CC_DFF": 2217, "CC_LUT1": 28, "CC_LUT2": 336}
    assert util["yosys_cells_total"] == 2217 + 28 + 336
    assert util["cell_counts_source"] == "synthesis_log_reparse"
    assert summary["timing_summary"]["max_frequency_mhz"] == 15.69


def test_exp3347_resource_summary_keeps_real_upstream_counts() -> None:
    """REQ-HW-102: populated upstream utilization is never overwritten."""
    build_artifact = {
        "timing_summary": {},
        "utilization_summary": {"yosys_cell_counts": {"CC_LUT2": 3}, "yosys_cells_total": 3},
        "build_log_paths": [],
    }
    summary = _resource_summary(build_artifact)
    assert summary["utilization_summary"]["yosys_cell_counts"] == {"CC_LUT2": 3}
    assert "cell_counts_source" not in summary["utilization_summary"]


def test_exp3347_resource_summary_empty_log_leaves_blank(tmp_path: Path) -> None:
    """REQ-HW-102: a missing log leaves utilization honestly empty, not faked."""
    build_artifact = {
        "timing_summary": {},
        "utilization_summary": {},
        "build_log_paths": [str(tmp_path / "absent.log")],
    }
    summary = _resource_summary(build_artifact)
    assert summary["utilization_summary"] == {}


def test_exp3347_relative_files_handles_outside_paths(tmp_path: Path) -> None:
    """REQ-HW-102: paths outside the repo root are recorded verbatim, blanks dropped."""
    outside = "/opt/oss-cad-suite/bin/gmpack"
    inside = tmp_path / "build" / "x.bit"
    result = _relative_files(tmp_path, ["", str(inside), outside])
    assert result == ["build/x.bit", outside]


def test_exp3347_run_experiment_default_runner_blocks_without_subprocess(tmp_path: Path) -> None:
    """REQ-HW-102: default runner path blocks on missing exp2955 with no commands.

    With no tools resolvable and no exp2955 package present, the build blocks at
    its first precondition, so the default subprocess runner is selected but
    never actually invoked — exercising the lazy-import branch safely.
    """
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=None,
        which_func=lambda name: None,
    )

    assert destination.exists()
    assert artifact["build_succeeded"] is False
    assert artifact["honest_verdict"] == "blocked_exp2955_constraints_not_ready"
    assert artifact["blocked_reasons"]
