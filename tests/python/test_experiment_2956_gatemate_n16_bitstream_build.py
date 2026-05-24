"""Tests for Exp 2956 GateMate n=16 bitstream build.

Spec refs: REQ-HW-076, SCENARIO-HW-076.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from carnot.experiment_2956_gatemate_n16_bitstream_build import (
    ARTIFACT_FILENAME,
    EXP2955_FILENAME,
    CommandResult,
    _failure_excerpt,
    _parse_version_text,
    build_artifact,
    run_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "fpga" / "spec.md"

REQUIRED_FIELDS = (
    "honest_verdict",
    "preconditions_checked",
    "gatemate_bitstream_built",
    "synthesis_command",
    "pnr_command",
    "pack_command",
    "bitstream_path",
    "bitstream_sha256",
    "timing_summary",
    "utilization_summary",
    "build_log_paths",
    "failure_command",
    "failure_excerpt",
    "inference_substrate",
    "duration_s",
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


def _paths() -> dict[str, str]:
    return {
        "yosys": "/suite/bin/yosys",
        "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
        "gmpack": "/suite/bin/gmpack",
        "openFPGALoader": "/suite/bin/openFPGALoader",
    }


def _write_exp2955(repo_root: Path, *, ready: bool = True) -> Path:
    rtl = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.v"
    ccf = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate.ccf"
    vector = repo_root / "hardware" / "gatemate" / "ising_n16_gatemate_test_vector.json"
    rtl.parent.mkdir(parents=True, exist_ok=True)
    rtl.write_text(
        "\n".join(
            [
                "module ising_n16_gatemate(input wire clk, output wire [15:0] spin_out);",
                "localparam integer N_VARIABLES = 16;",
                "assign spin_out = 16'hace1;",
                "endmodule",
                "",
            ]
        ),
        encoding="utf-8",
    )
    ccf.write_text(
        "# Spec: REQ-HW-075, SCENARIO-HW-075.\n# allow-unconstrained\n",
        encoding="utf-8",
    )
    vector.write_text('{"schema":"test"}\n', encoding="utf-8")
    payload = {
        "honest_verdict": "complete: gatemate_constraints_materialized",
        "gatemate_constraints_ready": ready,
        "rtl": {"path": str(rtl), "top_module": "ising_n16_gatemate", "n16": True},
        "constraints_file_paths": [str(ccf)],
        "test_vector_paths": [str(vector)],
        "top_module": "ising_n16_gatemate",
        "clock_assumption": "12.0 MHz nextpnr target frequency",
    }
    path = repo_root / "results" / EXP2955_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _tool_runner(*, pnr_fails: bool = False, pack_writes_bitstream: bool = True):
    calls: list[tuple[str, ...]] = []

    def run(args: list[str], timeout_s: float) -> CommandResult:
        del timeout_s
        call = tuple([Path(args[0]).name, *args[1:]])
        calls.append(call)
        if call == ("yosys", "-V"):
            return CommandResult(0, "Yosys 0.64+149\n", "")
        if call == ("nextpnr-himbaechel", "--version"):
            return CommandResult(0, '"nextpnr-himbaechel" Version nextpnr-0.10-test\n', "")
        if call == ("gmpack", "--version"):
            return CommandResult(
                1,
                "",
                "Error: unrecognised option '--version'\n"
                "Open Source Tools for GateMate FPGAs Version v1.13-test\n",
            )
        if call == ("openFPGALoader", "-V"):
            return CommandResult(0, "openFPGALoader v1.1.1\n", "")
        joined = " ".join(args)
        if call[0] == "yosys" and "synth_gatemate" in joined:
            json_match = re.search(r"-json\s+([^;]+)", joined)
            assert json_match is not None
            Path(json_match.group(1)).parent.mkdir(parents=True, exist_ok=True)
            Path(json_match.group(1)).write_text('{"modules":{}}\n', encoding="utf-8")
            return CommandResult(
                0,
                "Number of cells:               5\n"
                "     CC_DFF                    2\n"
                "     CC_LUT2                   3\n",
                "",
            )
        if call[0] == "nextpnr-himbaechel" and "--json" in call:
            if pnr_fails:
                return CommandResult(1, "", "Error: CCF parser failed at line 1\n")
            out_vopt = next(item for item in call if item.startswith("out="))
            cfg_path = Path(out_vopt.removeprefix("out="))
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text("cfg\n", encoding="utf-8")
            write_path = Path(call[call.index("--write") + 1])
            write_path.write_text('{"pnr":true}\n', encoding="utf-8")
            return CommandResult(
                0,
                "Info: Max frequency for clock 'clk': 48.50 MHz\n"
                "Info: Device utilisation:\n"
                "Info:   CPE: 42/1024 4%\n",
                "",
            )
        if call[0] == "gmpack":
            if pack_writes_bitstream:
                Path(args[2]).write_bytes(b"gate-mate-bitstream")
            return CommandResult(0, "Writing bitstream\n", "")
        raise AssertionError(f"unexpected command: {call}")

    return run, calls


def test_exp2956_spec_entry_present() -> None:
    """REQ-HW-076: the FPGA capability spec anchors the build artifact."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-HW-076" in spec
    assert "SCENARIO-HW-076" in spec
    assert ARTIFACT_FILENAME in spec


def test_exp2956_builds_bitstream_and_records_schema(tmp_path: Path) -> None:
    """SCENARIO-HW-076: ready constraints produce a hashed bitstream artifact."""
    _write_exp2955(tmp_path)
    run_command, calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([10.0, 12.5]),
    )

    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    assert missing == []
    assert artifact["honest_verdict"] == "complete: gatemate_n16_bitstream_built"
    assert artifact["gatemate_bitstream_built"] is True
    assert artifact["inference_substrate"] == "hardware_build"
    assert artifact["duration_s"] == 2.5
    assert artifact["failure_command"] == ""
    assert artifact["failure_excerpt"] == ""
    assert Path(artifact["bitstream_path"]).exists()
    assert artifact["bitstream_sha256"] == hashlib.sha256(b"gate-mate-bitstream").hexdigest()
    assert len(artifact["build_log_paths"]) == 3
    assert all(Path(path).exists() for path in artifact["build_log_paths"])
    assert "synth_gatemate -top ising_n16_gatemate" in artifact["synthesis_command"]
    assert "-luttree" in artifact["synthesis_command"]
    assert "nextpnr-himbaechel --device CCGM1A1" in artifact["pnr_command"]
    assert "--freq 12.0" in artifact["pnr_command"]
    assert "--vopt allow-unconstrained" in artifact["pnr_command"]
    assert "ccf=" in artifact["pnr_command"]
    assert artifact["pack_command"].startswith("/suite/bin/gmpack ")
    assert artifact["timing_summary"]["requested_frequency_mhz"] == 12.0
    assert artifact["timing_summary"]["max_frequency_mhz"] == 48.5
    assert artifact["timing_summary"]["timing_met"] is True
    assert artifact["utilization_summary"]["yosys_cells_total"] == 5
    assert artifact["utilization_summary"]["yosys_cell_counts"]["CC_LUT2"] == 3
    assert artifact["utilization_summary"]["nextpnr_resource_lines"] == [
        "Info:   CPE: 42/1024 4%"
    ]
    assert not any("-b" in call or "olimex_gatemateevb" in call for call in calls)


def test_exp2956_blocks_when_exp2955_is_not_ready(tmp_path: Path) -> None:
    """REQ-HW-076: an unready constraints package blocks before tool execution."""
    _write_exp2955(tmp_path, ready=False)
    run_command, calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([20.0, 20.25]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2955_constraints_not_ready"
    assert artifact["gatemate_bitstream_built"] is False
    assert artifact["failure_excerpt"] == "exp2955 gatemate_constraints_ready is false"
    assert calls == []


def test_exp2956_blocks_when_exp2955_artifact_is_missing(tmp_path: Path) -> None:
    """REQ-HW-076: missing exp2955 evidence blocks before tool execution."""
    run_command, calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([22.0, 22.1]),
    )

    assert artifact["honest_verdict"] == "blocked_exp2955_constraints_not_ready"
    assert "missing exp2955 artifact" in artifact["failure_excerpt"]
    assert calls == []


def test_exp2956_blocks_on_invalid_exp2955_source_metadata(tmp_path: Path) -> None:
    """REQ-HW-076: ready=true still requires files, constraints, and top identity."""
    exp_path = _write_exp2955(tmp_path)

    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    Path(payload["constraints_file_paths"][0]).unlink()
    exp_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=_tool_runner()[0],
        which_func=_which_from(_paths()),
        monotonic=_clock([23.0, 23.1]),
    )
    assert artifact["failure_excerpt"].startswith("missing exp2955 source file:")

    exp_path = _write_exp2955(tmp_path)
    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    payload["constraints_file_paths"] = []
    exp_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=_tool_runner()[0],
        which_func=_which_from(_paths()),
        monotonic=_clock([24.0, 24.1]),
    )
    assert artifact["failure_excerpt"] == "exp2955 constraints_file_paths is empty"

    exp_path = _write_exp2955(tmp_path)
    payload = json.loads(exp_path.read_text(encoding="utf-8"))
    payload["top_module"] = "wrong_top"
    payload["rtl"]["top_module"] = "wrong_top"
    exp_path.write_text(json.dumps(payload), encoding="utf-8")
    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=_tool_runner()[0],
        which_func=_which_from(_paths()),
        monotonic=_clock([25.0, 25.1]),
    )
    assert artifact["failure_excerpt"] == "exp2955 top_module is not ising_n16_gatemate"


def test_exp2956_missing_tool_blocks_before_synthesis(tmp_path: Path) -> None:
    """REQ-HW-076: all current OSS CAD Suite tools are required preconditions."""
    _write_exp2955(tmp_path)
    run_command, calls = _tool_runner()

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(
            {
                "yosys": "/suite/bin/yosys",
                "nextpnr-himbaechel": "/suite/bin/nextpnr-himbaechel",
                "openFPGALoader": "/suite/bin/openFPGALoader",
            }
        ),
        monotonic=_clock([30.0, 30.5]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_toolchain_missing"
    assert artifact["gatemate_bitstream_built"] is False
    assert artifact["failure_excerpt"] == "missing toolchain: gmpack"
    assert not any("synth_gatemate" in " ".join(call) for call in calls)


def test_exp2956_pnr_failure_records_failing_command_and_excerpt(tmp_path: Path) -> None:
    """SCENARIO-HW-076: command failure artifacts preserve the actionable error."""
    _write_exp2955(tmp_path)
    run_command, calls = _tool_runner(pnr_fails=True)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([40.0, 43.0]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_pnr_failed"
    assert artifact["gatemate_bitstream_built"] is False
    assert artifact["failure_command"] == artifact["pnr_command"]
    assert artifact["failure_excerpt"] == "Error: CCF parser failed at line 1"
    assert len(artifact["build_log_paths"]) == 2
    assert "CCF parser failed" in Path(artifact["build_log_paths"][-1]).read_text(
        encoding="utf-8"
    )
    assert not any(call[0] == "gmpack" and call[1:] != ("--version",) for call in calls)


def test_exp2956_pack_without_bitstream_is_blocked(tmp_path: Path) -> None:
    """REQ-HW-076: built=true requires an actual bitstream file and hash."""
    _write_exp2955(tmp_path)
    run_command, _calls = _tool_runner(pack_writes_bitstream=False)

    artifact = build_artifact(
        repo_root=tmp_path,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([45.0, 46.0]),
    )

    assert artifact["honest_verdict"] == "blocked_gatemate_bitstream_missing"
    assert artifact["gatemate_bitstream_built"] is False
    assert artifact["failure_command"] == artifact["pack_command"]
    assert artifact["failure_excerpt"].endswith("did not create the bitstream")
    assert artifact["bitstream_sha256"] == ""


def test_exp2956_run_experiment_writes_artifact(tmp_path: Path) -> None:
    """REQ-HW-076: run_experiment writes the required v4 deliverable JSON."""
    _write_exp2955(tmp_path)
    run_command, _calls = _tool_runner()
    destination = tmp_path / "results" / ARTIFACT_FILENAME

    artifact = run_experiment(
        repo_root=tmp_path,
        artifact_path=destination,
        run_command=run_command,
        which_func=_which_from(_paths()),
        monotonic=_clock([50.0, 51.5]),
    )

    assert destination.exists()
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded == artifact
    assert loaded["gatemate_bitstream_built"] is True


def test_exp2956_parser_fallbacks_are_honest() -> None:
    """REQ-HW-076: empty versions and non-error failures are not fabricated."""
    assert _parse_version_text(CommandResult(1, "", "Error: no version\n")) == ""
    assert _failure_excerpt(CommandResult(7, "router stopped\n", "")) == "router stopped"
    assert _failure_excerpt(CommandResult(7, "", "")) == "command exited with return code 7"
