#!/usr/bin/env python3
"""Experiment 851 — iCE40 HX8K N=16 Ising Sampler Bitstream.

MOTIVATION:
    Exp 839 synthesised the N=32 v3 Ising sampler at 3952 LUTs, which exceeds the
    iCE40 HX8K effective P&R budget (~3500-4000 LUTs).  Coupling registers scale as
    N^2, so dropping from N=32 to N=16 reduces coupling storage by 4x and is expected
    to produce a ~1000-1500 LUT design well within the 7680 LUT total budget.

    This experiment uses a *simplified deterministic* design (no LFSR, no EMA) to
    prove the full synthesis → P&R → bitstream toolchain works end-to-end on iCE40.
    Stochastic sampling is deferred to a follow-up experiment once we confirm P&R fits.

DELIVERABLE:
    results/experiment_851_ice40_n16_bitstream.json with honest_verdict field set to:
      "bitstream_generated"  — .bin produced with valid 0x7E magic header
      "pnr_failed_n16"       — nextpnr failed even at N=16 (unexpected; diagnose LUTs)
      "synthesis_failed"     — yosys failed to synthesise the N=16 design
      "tools_missing"        — OSS-CAD-Suite not found at expected path

Spec: REQ-FPGA-005, SCENARIO-FPGA-006
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to repo root so the script works from any CWD)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
OSS_CAD_BIN = Path(os.path.expanduser("~/tools/oss-cad-suite/bin"))
VERILOG_SRC = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_n16.v"
OUTPUT_DIR = REPO_ROOT / "output"
SYNTH_JSON = OUTPUT_DIR / "ising_n16_synth.json"
PNR_ASC = OUTPUT_DIR / "ising_n16.asc"
BITSTREAM = OUTPUT_DIR / "carnot_ising_n16.bin"
DELIVERABLE = REPO_ROOT / "results" / "experiment_851_ice40_n16_bitstream.json"


def check_tools() -> tuple[bool, str]:
    """Return (tools_found, oss_cad_bin_path).

    Checks that yosys, nextpnr-ice40, and icepack are all present at the
    expected OSS-CAD-Suite installation path.  These tools are part of the
    open-source iCE40 FPGA toolchain (Project IceStorm + Yosys + nextpnr).
    """
    required = ["yosys", "nextpnr-ice40", "icepack"]
    for tool in required:
        if not (OSS_CAD_BIN / tool).exists():
            return False, str(OSS_CAD_BIN)
    return True, str(OSS_CAD_BIN)


def run_tool(cmd: list[str], label: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr).

    WHY: subprocess.run with a timeout ensures a stuck synthesis tool doesn't
    hang the experiment indefinitely.  We capture both stdout and stderr so
    the log can be stored in the result artifact for post-mortem debugging.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"{label} timed out after {timeout}s"
    except FileNotFoundError as exc:
        return -1, "", f"{label} binary not found: {exc}"


def parse_lut_count(yosys_stdout: str) -> int:
    """Extract the synthesised LUT count from yosys stdout.

    Yosys prints a summary line like:
        Number of cells:              1234
    or for the SB_LUT4 primitive specifically:
        SB_LUT4              456
    We prefer the SB_LUT4 count because that is the actual iCE40 LUT primitive
    (other "cells" include DFFs and BRAMs which do not consume LUT rows).
    Falls back to the generic cell count if the SB_LUT4 line is absent.
    Returns 0 if no count is found (indicates synthesis produced no output).
    """
    # Try SB_LUT4 count first (most accurate for iCE40)
    m = re.search(r"SB_LUT4\s+(\d+)", yosys_stdout)
    if m:
        return int(m.group(1))
    # Fallback: total cell count
    m = re.search(r"Number of cells:\s+(\d+)", yosys_stdout)
    if m:
        return int(m.group(1))
    return 0


def parse_fmax_mhz(nextpnr_stderr: str) -> float:
    """Extract the maximum clock frequency from nextpnr timing summary.

    nextpnr-ice40 prints timing info to stderr, e.g.:
        Info: Max frequency for clock 'clk': 62.35 MHz (PASSED at 12.00 MHz)
    Returns 0.0 if the line is absent (non-constrained design has no fmax).
    """
    m = re.search(r"Max frequency for clock.*?:\s*([\d.]+)\s*MHz", nextpnr_stderr)
    if m:
        return float(m.group(1))
    return 0.0


def synthesise_n16() -> dict:
    """Run yosys synthesis targeting iCE40 on the N=16 Ising sampler.

    Returns a dict with keys: success, lut_count, log, synth_json_path.
    WHY two-step (synth then P&R): yosys converts RTL to a technology-mapped
    netlist (JSON); nextpnr then places that netlist onto actual iCE40 routing
    resources.  Separating the steps lets us inspect the LUT count before
    committing to the more expensive P&R.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yosys_bin = str(OSS_CAD_BIN / "yosys")

    # Build the synthesis script inline.  The synth_ice40 pass runs the full
    # iCE40 synthesis flow: flatten → techmap → abc → map to SB_LUT4/SB_DFF.
    synth_script = f"read_verilog {VERILOG_SRC}; synth_ice40 -top ising_sampler -json {SYNTH_JSON}"
    cmd = [yosys_bin, "-p", synth_script]
    rc, stdout, stderr = run_tool(cmd, "yosys", timeout=180)

    combined_log = stdout + stderr
    lut_count = parse_lut_count(combined_log)

    return {
        "success": rc == 0,
        "lut_count": lut_count,
        "log": combined_log[:8000],  # truncate for artifact size
        "synth_json_path": str(SYNTH_JSON) if rc == 0 else None,
    }


def place_and_route() -> dict:
    """Run nextpnr-ice40 place-and-route for the iCE40 HX8K ct256 package.

    Returns a dict with keys: success, fmax_mhz, log.
    WHY HX8K ct256: this is the onboard iCE40 on the Kria KV260's debug
    interface FPGA.  The HX8K has 7680 LUTs; ct256 is the 256-ball BGA package.
    """
    pnr_bin = str(OSS_CAD_BIN / "nextpnr-ice40")
    cmd = [
        pnr_bin,
        "--hx8k",
        "--package",
        "ct256",
        "--json",
        str(SYNTH_JSON),
        "--asc",
        str(PNR_ASC),
    ]
    rc, stdout, stderr = run_tool(cmd, "nextpnr-ice40", timeout=300)
    combined_log = stdout + stderr
    fmax_mhz = parse_fmax_mhz(combined_log)

    return {
        "success": rc == 0,
        "fmax_mhz": fmax_mhz,
        "log": combined_log[:8000],
    }


def pack_bitstream() -> dict:
    """Run icepack to convert the nextpnr ASCII (.asc) file to a binary bitstream.

    Returns a dict with keys: success, bitstream_path, size_bytes, valid_header.
    WHY 0x7E magic: the iCE40 bitstream format (Project IceStorm reverse-engineered)
    always starts with byte 0xFF 0x00 in the preamble, but the icepack output file
    begins with 0x7E (ASCII tilde).  This is the documented magic for a valid
    icepack-produced bitstream.
    """
    icepack_bin = str(OSS_CAD_BIN / "icepack")
    cmd = [icepack_bin, str(PNR_ASC), str(BITSTREAM)]
    rc, stdout, stderr = run_tool(cmd, "icepack", timeout=60)

    if rc != 0 or not BITSTREAM.exists():
        return {
            "success": False,
            "bitstream_path": None,
            "size_bytes": 0,
            "valid_header": False,
            "log": (stdout + stderr)[:2000],
        }

    size_bytes = BITSTREAM.stat().st_size
    with open(BITSTREAM, "rb") as f:
        first_byte = f.read(1)
    valid_header = len(first_byte) == 1 and first_byte[0] == 0x7E

    return {
        "success": True,
        "bitstream_path": str(BITSTREAM),
        "size_bytes": size_bytes,
        "valid_header": valid_header,
        "log": (stdout + stderr)[:2000],
    }


def main() -> None:
    """Run the full synthesis → P&R → bitstream pipeline and write the deliverable."""
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    # Step 1: Verify toolchain
    tools_found, oss_cad_bin = check_tools()
    if not tools_found:
        artifact = build_artifact(
            started_at=started_at,
            duration_s=time.time() - t0,
            honest_verdict="tools_missing",
            lut_count_n16=0,
            fmax_mhz=0.0,
            bitstream_generated=False,
            bitstream_size_bytes=0,
            bitstream_valid_header=False,
            output_path=None,
            synth_log="tools not found",
            pnr_log="",
            pack_log="",
            oss_cad_bin=oss_cad_bin,
        )
        write_deliverable(artifact)
        sys.exit(0)

    # Step 2: Synthesis
    print("[851] Running yosys synthesis...")
    synth = synthesise_n16()
    if not synth["success"]:
        artifact = build_artifact(
            started_at=started_at,
            duration_s=time.time() - t0,
            honest_verdict="synthesis_failed",
            lut_count_n16=synth["lut_count"],
            fmax_mhz=0.0,
            bitstream_generated=False,
            bitstream_size_bytes=0,
            bitstream_valid_header=False,
            output_path=None,
            synth_log=synth["log"],
            pnr_log="",
            pack_log="",
            oss_cad_bin=oss_cad_bin,
        )
        write_deliverable(artifact)
        print(f"[851] Synthesis FAILED. LUT count: {synth['lut_count']}")
        sys.exit(0)

    print(f"[851] Synthesis OK. LUT count: {synth['lut_count']}")

    # Step 3: Place-and-route
    print("[851] Running nextpnr-ice40 P&R...")
    pnr = place_and_route()
    if not pnr["success"]:
        artifact = build_artifact(
            started_at=started_at,
            duration_s=time.time() - t0,
            honest_verdict="pnr_failed_n16",
            lut_count_n16=synth["lut_count"],
            fmax_mhz=pnr["fmax_mhz"],
            bitstream_generated=False,
            bitstream_size_bytes=0,
            bitstream_valid_header=False,
            output_path=None,
            synth_log=synth["log"],
            pnr_log=pnr["log"],
            pack_log="",
            oss_cad_bin=oss_cad_bin,
        )
        write_deliverable(artifact)
        print(f"[851] P&R FAILED. fmax: {pnr['fmax_mhz']} MHz")
        sys.exit(0)

    print(f"[851] P&R OK. fmax: {pnr['fmax_mhz']} MHz")

    # Step 4: Pack bitstream
    print("[851] Running icepack...")
    pack = pack_bitstream()
    if not pack["success"]:
        artifact = build_artifact(
            started_at=started_at,
            duration_s=time.time() - t0,
            honest_verdict="pnr_failed_n16",
            lut_count_n16=synth["lut_count"],
            fmax_mhz=pnr["fmax_mhz"],
            bitstream_generated=False,
            bitstream_size_bytes=0,
            bitstream_valid_header=False,
            output_path=None,
            synth_log=synth["log"],
            pnr_log=pnr["log"],
            pack_log=pack.get("log", ""),
            oss_cad_bin=oss_cad_bin,
        )
        write_deliverable(artifact)
        sys.exit(0)

    print(f"[851] Bitstream packed: {pack['bitstream_path']} ({pack['size_bytes']} bytes)")
    print(f"[851] Valid header (0x7E): {pack['valid_header']}")

    # Step 5: Final verdict
    if pack["valid_header"]:
        honest_verdict = "bitstream_generated"
    else:
        honest_verdict = "pnr_failed_n16"  # pack produced something but header wrong

    artifact = build_artifact(
        started_at=started_at,
        duration_s=time.time() - t0,
        honest_verdict=honest_verdict,
        lut_count_n16=synth["lut_count"],
        fmax_mhz=pnr["fmax_mhz"],
        bitstream_generated=pack["valid_header"],
        bitstream_size_bytes=pack["size_bytes"],
        bitstream_valid_header=pack["valid_header"],
        output_path=pack["bitstream_path"],
        synth_log=synth["log"],
        pnr_log=pnr["log"],
        pack_log=pack.get("log", ""),
        oss_cad_bin=oss_cad_bin,
    )
    write_deliverable(artifact)
    print(f"[851] DONE. honest_verdict={honest_verdict}")


def build_artifact(
    *,
    started_at: str,
    duration_s: float,
    honest_verdict: str,
    lut_count_n16: int,
    fmax_mhz: float,
    bitstream_generated: bool,
    bitstream_size_bytes: int,
    bitstream_valid_header: bool,
    output_path: str | None,
    synth_log: str,
    pnr_log: str,
    pack_log: str,
    oss_cad_bin: str,
) -> dict:
    """Assemble the standardised result artifact for experiment 851.

    All required schema fields are included so the conductor can parse the
    artifact without field-missing errors.
    """
    return {
        "experiment": 851,
        "title": "iCE40 HX8K N=16 Ising Sampler Bitstream",
        "run_date": time.strftime("%Y%m%d", time.gmtime()),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(duration_s, 3),
        "status": "success" if bitstream_generated else "partial",
        "honest_verdict": honest_verdict,
        "gate_exp839_lut_count_n32": 3952,
        "lut_count_n16": lut_count_n16,
        "lut_reduction_factor": round(3952 / lut_count_n16, 2) if lut_count_n16 > 0 else None,
        "fmax_mhz": fmax_mhz,
        "bitstream_generated": bitstream_generated,
        "bitstream_size_bytes": bitstream_size_bytes,
        "bitstream_valid_header": bitstream_valid_header,
        "output_path": output_path,
        "oss_cad_bin": oss_cad_bin,
        "verilog_src": str(VERILOG_SRC),
        "synth_log": synth_log,
        "pnr_log": pnr_log,
        "pack_log": pack_log,
        "schema": sorted(
            [
                "experiment",
                "title",
                "run_date",
                "started_at",
                "finished_at",
                "duration_s",
                "status",
                "honest_verdict",
                "gate_exp839_lut_count_n32",
                "lut_count_n16",
                "lut_reduction_factor",
                "fmax_mhz",
                "bitstream_generated",
                "bitstream_size_bytes",
                "bitstream_valid_header",
                "output_path",
                "oss_cad_bin",
                "verilog_src",
                "synth_log",
                "pnr_log",
                "pack_log",
            ]
        ),
        "invariant_violations": [],
    }


def write_deliverable(artifact: dict) -> None:
    """Write the result artifact JSON to the deliverable path.

    Uses an atomic write pattern (write temp file then rename) to avoid
    leaving a partially-written JSON if the process is killed mid-write.
    WHY atomic: the conductor checks for the deliverable file to know if the
    experiment completed; a partial file would cause a silent parse failure.
    """
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DELIVERABLE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(artifact, f, indent=2)
    tmp.rename(DELIVERABLE)
    print(f"[851] Deliverable written to {DELIVERABLE}")


if __name__ == "__main__":
    main()
