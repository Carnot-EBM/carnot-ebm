#!/usr/bin/env python3
"""Experiment 349: Open-source FPGA synthesis of KV260 Ising sampler.

**Researcher summary:**
    The KV260 FPGA has been physically present since Exp 288 but has never run
    real hardware due to a missing bitfile.  The previous approach (Exp 313)
    assumed a pre-built bitfile.  This experiment proves the open-source
    synthesis path is viable: yosys + nextpnr-xilinx can generate a netlist
    (and optionally a placed-and-routed bitfile) from the Ising sampler RTL
    written in Exp 291, without requiring a Vivado license.

    Key papers:
      - arXiv 2503.01177  — sparsified connectivity reduces LUT count
      - arXiv 2602.15985  — 77.5μs convergence target for ≤100 spins

**Detailed explanation for engineers:**
    The synthesis pipeline has two stages:

    Stage 1 — Synthesis (yosys):
        yosys -p "synth_xilinx -top ising_sampler_128 -flatten;
                  write_json /tmp/netlist_349.json"
              hardware/kv260/ising_sampler_v1.v

        yosys performs technology-independent synthesis then maps to Xilinx
        primitives (LUTs, FFs, BRAMs).  The stdout report contains lines like:

            Number of cells:              2048
            ...
            LUT6:                          512
            FDRE:                          128

        parse_synthesis_output() extracts these counts via regex.

    Stage 2 — Place-and-Route (nextpnr-xilinx, optional):
        If nextpnr-xilinx is installed, it runs P&R for the KV260 device
        (xczu5ev-sfvc784-2-e).  A successful P&R writes
        hardware/kv260/carnot_ising.bit.

    honest_verdict options:
        "synthesis_success"        — yosys + P&R completed, bitfile written
        "synthesis_partial"        — yosys netlist produced; no P&R (nextpnr
                                     absent or P&R failed)
        "blocked_missing_yosys"   — yosys not installed
        "blocked_missing_verilog" — RTL source not found on disk
        "synthesis_failed"         — yosys ran but exited non-zero

**Output artifact:** results/experiment_349_kv260_synthesis.json
    schema: "carnot.fpga_synthesis.v1"
    Fields: synthesis_result (SynthesisResult dict), lut_count, ff_count,
            bitfile_generated, honest_verdict, prereqs_checked

Usage::

    JAX_PLATFORMS=cpu python scripts/experiment_349_kv260_synthesis.py

Spec: REQ-HW-003, SCENARIO-HW-005, SCENARIO-HW-006
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 349
VERILOG_SOURCE = Path("hardware/kv260/ising_sampler_v1.v")
NETLIST_PATH = Path("/tmp/netlist_349.json")
BITFILE_PATH = Path("hardware/kv260/carnot_ising.bit")
OUTPUT_PATH = Path("results/experiment_349_kv260_synthesis.json")

# KV260 device string for nextpnr-xilinx.
# xczu5ev is the Zynq UltraScale+ on the KV260 carrier card.
KV260_DEVICE = "xczu5ev-sfvc784-2-e"

# Top-level module name in the Verilog RTL.
TOP_MODULE = "ising_sampler_128"


# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------


def check_yosys_available() -> bool:
    """Return True if yosys is installed and responds to --version.

    Why: yosys is the open-source synthesis tool.  We call it as a subprocess
    rather than importing a library, so availability means "executable on PATH".
    We capture the returncode; a non-zero return or FileNotFoundError means
    yosys is absent.  We never raise — callers get a clean boolean.
    """
    try:
        result = subprocess.run(
            ["yosys", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def check_nextpnr_available() -> bool:
    """Return True if nextpnr-xilinx is installed and responds to --version.

    Why: nextpnr-xilinx provides place-and-route for Xilinx devices in the
    open-source toolchain.  Same subprocess probe pattern as check_yosys_available.
    """
    try:
        result = subprocess.run(
            ["nextpnr-xilinx", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def check_verilog_source_exists(path: Path | str) -> bool:
    """Return True if the Verilog RTL source file exists at the given path.

    Why: synthesis cannot proceed if the source file is missing.  This is a
    simple existence check — we do not validate syntax here; yosys does that.
    """
    return os.path.exists(str(path))


# ---------------------------------------------------------------------------
# Synthesis output parser
# ---------------------------------------------------------------------------


def parse_synthesis_output(stdout: str) -> dict:
    """Extract LUT count, FF count, and a timing summary from yosys stdout.

    Why: yosys synthesis reports write resource-utilization lines to stdout in
    a human-readable table format.  We use regex to fish out the quantities
    we care about rather than parsing the full JSON netlist (which would require
    an additional dependency).

    The typical yosys synth_xilinx output contains lines like::

        Number of cells:               2048
        ...
          LUT1:                          32
          LUT2:                         128
          LUT3:                         256
          LUT4:                         512
          LUT5:                         128
          LUT6:                        1024
          FDRE:                         256
          FDSE:                          64

    We sum all LUT* entries as lut_count and all FD* entries as ff_count.

    Returns a dict with keys:
        lut_count   — total LUT cells (int or None if not found)
        ff_count    — total flip-flop cells (int or None if not found)
        raw_lines   — list of raw resource lines for debugging

    This function never raises: if parsing fails it returns Nones with an
    empty raw_lines list.
    """
    lut_count: Optional[int] = None
    ff_count: Optional[int] = None
    raw_lines: list[str] = []

    # Patterns for LUTn and FD* (flip-flop) cells.
    lut_pattern = re.compile(r"^\s+(LUT\d+)\s*:\s*(\d+)", re.MULTILINE)
    ff_pattern = re.compile(r"^\s+(FD\w+)\s*:\s*(\d+)", re.MULTILINE)
    # Also capture the aggregate "Number of cells" line for reference.
    cell_total_pattern = re.compile(r"Number of cells:\s*(\d+)", re.MULTILINE)

    for match in lut_pattern.finditer(stdout):
        raw_lines.append(match.group(0).strip())
        count = int(match.group(2))
        lut_count = (lut_count or 0) + count

    for match in ff_pattern.finditer(stdout):
        raw_lines.append(match.group(0).strip())
        count = int(match.group(2))
        ff_count = (ff_count or 0) + count

    cell_match = cell_total_pattern.search(stdout)
    if cell_match:
        raw_lines.append(cell_match.group(0).strip())

    return {
        "lut_count": lut_count,
        "ff_count": ff_count,
        "raw_lines": raw_lines,
    }


# ---------------------------------------------------------------------------
# SynthesisResult dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SynthesisResult:
    """Record of a single synthesis attempt, structured for JSON serialization.

    Why: a plain dataclass keeps the synthesis state machine output cleanly
    separated from the JSON artifact schema.  honest_verdict is the primary
    signal for the next engineer picking up this work.

    Fields:
        yosys_available      — yosys executable found on PATH
        nextpnr_available    — nextpnr-xilinx executable found on PATH
        verilog_found        — RTL source file exists on disk
        synthesis_attempted  — yosys was actually invoked
        synthesis_success    — yosys exited 0 AND produced usable output
        lut_count            — total LUT cells from synthesis report
        ff_count             — total FF cells from synthesis report
        honest_verdict       — one of the five approved verdict strings
    """

    yosys_available: bool
    nextpnr_available: bool
    verilog_found: bool
    synthesis_attempted: bool
    synthesis_success: bool
    lut_count: Optional[int]
    ff_count: Optional[int]
    honest_verdict: str

    # Approved vocabulary — callers must use one of these strings.
    APPROVED_VERDICTS = frozenset(
        [
            "synthesis_success",
            "synthesis_partial",
            "blocked_missing_yosys",
            "blocked_missing_verilog",
            "synthesis_failed",
        ]
    )

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON output."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Core synthesis runner
# ---------------------------------------------------------------------------


def run_synthesis(
    verilog_path: Path,
    netlist_path: Path = NETLIST_PATH,
    *,
    _subprocess_run=subprocess.run,
) -> tuple[int, str, str]:
    """Run yosys synthesis on the given Verilog file.

    Why: extracted as a separate function so tests can inject a fake
    _subprocess_run without touching os.environ or the filesystem.

    Returns (returncode, stdout, stderr).
    """
    cmd = [
        "yosys",
        "-p",
        (
            f"synth_xilinx -top {TOP_MODULE} -flatten; "
            f"write_json {netlist_path}"
        ),
        str(verilog_path),
    ]
    result = _subprocess_run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout, result.stderr


def run_nextpnr(
    netlist_path: Path,
    bitfile_path: Path,
    *,
    _subprocess_run=subprocess.run,
) -> tuple[int, str, str]:
    """Run nextpnr-xilinx place-and-route.

    Why: separated from run_synthesis so tests can stub each tool independently.
    P&R is optional; callers treat a non-zero returncode as "partial" rather
    than "failed".

    Returns (returncode, stdout, stderr).
    """
    cmd = [
        "nextpnr-xilinx",
        "--device",
        KV260_DEVICE,
        "--json",
        str(netlist_path),
        "--write",
        str(bitfile_path),
    ]
    result = _subprocess_run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(
    output_path: Path = OUTPUT_PATH,
    verilog_path: Path = VERILOG_SOURCE,
    netlist_path: Path = NETLIST_PATH,
    bitfile_path: Path = BITFILE_PATH,
    *,
    write_output: bool = True,
    _subprocess_run=subprocess.run,
) -> dict:
    """Execute the open-source synthesis experiment and return the artifact dict.

    Why: the function signature accepts injectable dependencies (output_path,
    verilog_path, _subprocess_run) so unit tests can exercise every branch
    without real tools or filesystem side-effects.

    Branches:
        1. yosys absent              → blocked_missing_yosys
        2. verilog source absent     → blocked_missing_verilog
        3. yosys exits non-zero      → synthesis_failed
        4. yosys OK, nextpnr absent  → synthesis_partial
        5. yosys OK, nextpnr OK, P&R succeeds → synthesis_success
        6. yosys OK, nextpnr OK, P&R fails    → synthesis_partial
    """
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # --- prerequisite checks ---
    yosys_ok = check_yosys_available()
    nextpnr_ok = check_nextpnr_available()
    verilog_ok = check_verilog_source_exists(verilog_path)

    prereqs_checked = {
        "yosys_available": yosys_ok,
        "nextpnr_available": nextpnr_ok,
        "verilog_found": verilog_ok,
        "verilog_path": str(verilog_path),
    }

    # Determine early-exit conditions.
    if not yosys_ok:
        result = SynthesisResult(
            yosys_available=False,
            nextpnr_available=nextpnr_ok,
            verilog_found=verilog_ok,
            synthesis_attempted=False,
            synthesis_success=False,
            lut_count=None,
            ff_count=None,
            honest_verdict="blocked_missing_yosys",
        )
        return _build_artifact(
            started_at, prereqs_checked, result, bitfile_path,
            output_path, write_output
        )

    if not verilog_ok:
        result = SynthesisResult(
            yosys_available=True,
            nextpnr_available=nextpnr_ok,
            verilog_found=False,
            synthesis_attempted=False,
            synthesis_success=False,
            lut_count=None,
            ff_count=None,
            honest_verdict="blocked_missing_verilog",
        )
        return _build_artifact(
            started_at, prereqs_checked, result, bitfile_path,
            output_path, write_output
        )

    # --- run yosys synthesis ---
    returncode, stdout, stderr = run_synthesis(
        verilog_path, netlist_path, _subprocess_run=_subprocess_run
    )

    if returncode != 0:
        result = SynthesisResult(
            yosys_available=True,
            nextpnr_available=nextpnr_ok,
            verilog_found=True,
            synthesis_attempted=True,
            synthesis_success=False,
            lut_count=None,
            ff_count=None,
            honest_verdict="synthesis_failed",
        )
        return _build_artifact(
            started_at, prereqs_checked, result, bitfile_path,
            output_path, write_output,
            yosys_stdout=stdout, yosys_stderr=stderr
        )

    # Parse resource utilization from synthesis report.
    parsed = parse_synthesis_output(stdout)
    lut_count = parsed["lut_count"]
    ff_count = parsed["ff_count"]

    # --- attempt place-and-route if nextpnr is available ---
    bitfile_generated = False
    if nextpnr_ok and netlist_path.exists():
        pnr_rc, pnr_stdout, pnr_stderr = run_nextpnr(
            netlist_path, bitfile_path, _subprocess_run=_subprocess_run
        )
        if pnr_rc == 0 and bitfile_path.exists():
            bitfile_generated = True
            verdict = "synthesis_success"
        else:
            verdict = "synthesis_partial"
    else:
        verdict = "synthesis_partial"

    result = SynthesisResult(
        yosys_available=True,
        nextpnr_available=nextpnr_ok,
        verilog_found=True,
        synthesis_attempted=True,
        synthesis_success=(verdict == "synthesis_success"),
        lut_count=lut_count,
        ff_count=ff_count,
        honest_verdict=verdict,
    )
    return _build_artifact(
        started_at, prereqs_checked, result, bitfile_path,
        output_path, write_output,
        yosys_stdout=stdout,
        bitfile_generated=bitfile_generated,
    )


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def _build_artifact(
    started_at: str,
    prereqs_checked: dict,
    synthesis_result: SynthesisResult,
    bitfile_path: Path,
    output_path: Path,
    write_output: bool,
    yosys_stdout: str = "",
    yosys_stderr: str = "",
    bitfile_generated: bool = False,
) -> dict:
    """Assemble and optionally write the standardised experiment artifact.

    Why: separated from run_experiment to keep that function's control-flow
    readable.  All artifact fields are set here in one place.
    """
    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": "carnot.fpga_synthesis.v1",
        "run_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "prereqs_checked": prereqs_checked,
        "synthesis_result": synthesis_result.to_dict(),
        "lut_count": synthesis_result.lut_count,
        "ff_count": synthesis_result.ff_count,
        "bitfile_generated": bitfile_generated,
        "bitfile_path": str(bitfile_path) if bitfile_generated else None,
        "honest_verdict": synthesis_result.honest_verdict,
        "spec_requirements": [
            "REQ-HW-003",
            "SCENARIO-HW-005",
            "SCENARIO-HW-006",
        ],
    }

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"Artifact written to {output_path}", file=sys.stderr)

    return artifact


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 349 and print a summary to stdout."""
    artifact = run_experiment()
    verdict = artifact["honest_verdict"]
    lut = artifact.get("lut_count")
    ff = artifact.get("ff_count")
    bitfile = artifact.get("bitfile_generated", False)

    print(f"Experiment 349: KV260 open-source synthesis")
    print(f"  honest_verdict      : {verdict}")
    print(f"  lut_count           : {lut}")
    print(f"  ff_count            : {ff}")
    print(f"  bitfile_generated   : {bitfile}")
    print(f"  artifact            : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
