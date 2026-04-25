#!/usr/bin/env python3
"""Exp 859 — iCE40 N=8 pure combinational energy oracle.

This experiment synthesises, places-and-routes, and generates a bitstream for
ising_energy_n8_comb.v — a pure combinational (no clock, no flip-flops) N=8
Ising energy oracle targeting the iCE40 HX8K FPGA.

**Why this exists:**
    Exp 851 showed that the N=16 sequential Verilog (with spin_state registers
    clocked by always @(posedge clk)) inferred 2077 DFFs + 9918 LUT4-only cells,
    totalling 12258 LCs — 159% of the iCE40 HX8K's 7680-LC budget.  nextpnr-ice40
    refused to place the design.

    The root cause is architectural: Gibbs sampling requires sequential state.
    The fix is to move that state out of the FPGA entirely.  Python drives the
    spin configuration; the FPGA evaluates E(s) combinationally.  No clocks.
    No flip-flops.  Input = spin_in[7:0].  Output = energy_out[15:0].

**Acceptance criteria (REQ-FPGA-030):**
    - bitstream_generated = True
    - lut_count < 500  (target ~134 based on synthesis preview)
    - sequential_logic_present = False  (no DFF in yosys output)

Spec: REQ-FPGA-030, SCENARIO-FPGA-040
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

# Allow import from project root when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VERILOG_SRC = _ROOT / "hardware" / "fpga" / "ising_energy_n8_comb.v"
PCF_FILE    = _ROOT / "hardware" / "fpga" / "hx8k.pcf"
JSON_OUT    = _ROOT / "hardware" / "fpga" / "ising_energy_n8_comb.json"
ASC_OUT     = _ROOT / "hardware" / "fpga" / "ising_energy_n8_comb.asc"
BIN_OUT     = _ROOT / "hardware" / "fpga" / "ising_energy_n8_comb.bin"
OSS_CAD_BIN = Path(os.environ.get("OSS_CAD_BIN", "/home/ianblenke/tools/oss-cad-suite/bin"))


def _oss_cmd(name: str) -> str:
    """Return the full path to an OSS-CAD-Suite binary.

    Why resolve against OSS_CAD_BIN rather than relying on PATH:
        The OSS-CAD-Suite environment script exports its own PATH, but tests
        mock subprocess.run and should not depend on PATH being configured.
        Constructing the path explicitly keeps the experiment testable in CI
        without having the tools installed.
    """
    candidate = OSS_CAD_BIN / name
    if candidate.exists():
        return str(candidate)
    # Fall back to name alone — relies on PATH (set by oss-cad-suite/environment).
    return name


def _run(cmd: list[str], *, label: str) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    Streams stderr to the terminal so synthesis/P&R progress is visible, while
    capturing both streams for artifact storage.  This mirrors the pattern used
    in Exp 851 (scripts/experiment_851_ice40_n16_bitstream.py).
    """
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(f"--- {label} returncode={result.returncode} ---")
    if result.stdout:
        print(result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout)
    if result.stderr:
        print(result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
              file=sys.stderr)
    return result.returncode, result.stdout, result.stderr


def _parse_synth_lut_count(synth_output: str) -> int:
    """Extract the SB_LUT4 count from yosys synth_ice40 stdout.

    Yosys prints a cell statistics table that looks like:
        === ising_energy_n8_comb ===
           132   SB_LUT4
    We want the integer on that line.

    Returns 0 if the count cannot be parsed (synthesis probably failed).
    """
    match = re.search(r"(\d+)\s+SB_LUT4", synth_output)
    if match:
        return int(match.group(1))
    return 0


def _parse_pnr_lut_count(pnr_output: str) -> int:
    """Extract the ICESTORM_LC utilisation count from nextpnr-ice40 output.

    nextpnr prints a utilisation table:
        Info: Device utilisation:
        Info: 	         ICESTORM_LC:     134/   7680     1%
    We want the numerator (134 in this example).

    Returns 0 if the count cannot be parsed (P&R probably failed).
    """
    match = re.search(r"ICESTORM_LC:\s+(\d+)/", pnr_output)
    if match:
        return int(match.group(1))
    return 0


def _has_sequential_logic(synth_output: str) -> bool:
    """Return True if yosys inferred any DFF (flip-flop) cells.

    A pure combinational design must produce zero DFF instances.  Yosys names
    flip-flop primitives with the prefix 'SB_DFF' in its cell statistics table.
    We search the section AFTER the 'SB_LUT4' line to avoid false positives on
    the cells_sim.v parsing header lines that also mention DFF cell names.
    """
    # Only check the final stats table, which appears after "Chip utilisation"
    # or the bottom section of the synthesis report.
    # The stats block looks like:
    #      132   SB_LUT4
    #        0   SB_DFF   ← would appear here if any were inferred
    # Since there are zero DFFs in a combinational design, this line simply
    # won't appear in the count table.
    stats_match = re.search(r"(\d+)\s+SB_DFF", synth_output)
    if stats_match:
        return int(stats_match.group(1)) > 0
    return False


def main() -> None:
    """Run Exp 859 end-to-end: synth → P&R → bitstream → artifact."""
    tmpl = ExperimentTemplate(
        exp_id=859,
        title="iCE40 N=8 combinational energy oracle",
        deliverable="results/experiment_859_ice40_n8_combinational.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1 — Synthesis (yosys synth_ice40)
    # ------------------------------------------------------------------
    synth_cmd = [
        _oss_cmd("yosys"),
        "-p",
        (
            f"synth_ice40 -top ising_energy_n8_comb "
            f"-json {JSON_OUT}"
        ),
        str(VERILOG_SRC),
    ]
    synth_rc, synth_stdout, synth_stderr = _run(synth_cmd, label="yosys synth_ice40")
    synth_log = synth_stdout + synth_stderr

    if synth_rc != 0:
        artifact = tmpl.build_result(
            {
                "synthesis_lut_count": 0,
                "pnr_lut_count": 0,
                "bitstream_generated": False,
                "sequential_logic_present": False,
                "lut_count": 0,
                "honest_verdict": "synthesis_failed",
                "synth_log": synth_log,
                "pnr_log": "",
                "verilog_src": str(VERILOG_SRC),
            },
            status="error",
        )
        Path("results/experiment_859_ice40_n8_combinational.json").write_text(
            __import__("json").dumps(artifact, indent=2)
        )
        tmpl.assert_deliverable_written()
        return

    synthesis_lut_count = _parse_synth_lut_count(synth_log)
    sequential_logic_present = _has_sequential_logic(synth_log)
    print(f"Synthesis: {synthesis_lut_count} SB_LUT4, "
          f"DFF present: {sequential_logic_present}")

    # ------------------------------------------------------------------
    # Step 2 — Place and Route (nextpnr-ice40)
    # ------------------------------------------------------------------
    pnr_cmd = [
        _oss_cmd("nextpnr-ice40"),
        "--hx8k",
        "--package", "ct256",
        "--json", str(JSON_OUT),
        "--pcf", str(PCF_FILE),
        "--asc", str(ASC_OUT),
        "--pcf-allow-unconstrained",
    ]
    pnr_rc, pnr_stdout, pnr_stderr = _run(pnr_cmd, label="nextpnr-ice40")
    pnr_log = pnr_stdout + pnr_stderr

    if pnr_rc != 0:
        artifact = tmpl.build_result(
            {
                "synthesis_lut_count": synthesis_lut_count,
                "pnr_lut_count": 0,
                "bitstream_generated": False,
                "sequential_logic_present": sequential_logic_present,
                "lut_count": synthesis_lut_count,
                "honest_verdict": "pnr_failed",
                "synth_log": synth_log,
                "pnr_log": pnr_log,
                "verilog_src": str(VERILOG_SRC),
            },
            status="error",
        )
        Path("results/experiment_859_ice40_n8_combinational.json").write_text(
            __import__("json").dumps(artifact, indent=2)
        )
        tmpl.assert_deliverable_written()
        return

    pnr_lut_count = _parse_pnr_lut_count(pnr_log)
    print(f"P&R: {pnr_lut_count} ICESTORM_LCs (of 7680)")

    # ------------------------------------------------------------------
    # Step 3 — Bitstream (icepack)
    # ------------------------------------------------------------------
    pack_cmd = [
        _oss_cmd("icepack"),
        str(ASC_OUT),
        str(BIN_OUT),
    ]
    pack_rc, pack_stdout, pack_stderr = _run(pack_cmd, label="icepack")
    pack_log = pack_stdout + pack_stderr

    bitstream_generated = (pack_rc == 0) and BIN_OUT.exists()
    bitstream_size_bytes = BIN_OUT.stat().st_size if bitstream_generated else 0
    print(f"Bitstream generated: {bitstream_generated} "
          f"({bitstream_size_bytes} bytes)")

    # ------------------------------------------------------------------
    # Step 4 — Determine honest verdict
    # ------------------------------------------------------------------
    lut_count = pnr_lut_count  # P&R count is authoritative
    if not bitstream_generated:
        honest_verdict = "pack_failed"
    elif lut_count >= 500:
        honest_verdict = "lut_over_budget"
    elif lut_count == 0:
        # Parsing failed; treat as unknown error rather than false success
        honest_verdict = "lut_parse_failed"
    elif sequential_logic_present:
        honest_verdict = "sequential_logic_detected"
    else:
        honest_verdict = "fpga_oracle_ready"

    status = "success" if honest_verdict == "fpga_oracle_ready" else "partial"

    # ------------------------------------------------------------------
    # Step 5 — Write artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "synthesis_lut_count": synthesis_lut_count,
            "pnr_lut_count": pnr_lut_count,
            "lut_count": lut_count,
            "bitstream_generated": bitstream_generated,
            "bitstream_size_bytes": bitstream_size_bytes,
            "sequential_logic_present": sequential_logic_present,
            "honest_verdict": honest_verdict,
            "prior_failure_addressed": {
                "experiment_id": "exp851-ice40-n16-bitstream",
                "verdict": "pnr_failed_n16",
                "root_cause": "sequential spin registers inferred 2077 DFFs "
                              "+ 9918 LUT4 cells = 12258 LCs (159% of 7680 budget)",
                "addressed_by": "removed all sequential logic; Gibbs sampling "
                                "moved to Python; FPGA is now pure combinational "
                                "energy oracle only",
            },
            "verilog_src": str(VERILOG_SRC),
            "bitstream_path": str(BIN_OUT) if bitstream_generated else None,
            "oss_cad_bin": str(OSS_CAD_BIN),
            "synth_log": synth_log,
            "pnr_log": pnr_log,
            "pack_log": pack_log,
        },
        status=status,
    )

    import json
    out_path = Path("results/experiment_859_ice40_n8_combinational.json")
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Artifact written to {out_path}")
    print(f"honest_verdict: {honest_verdict}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
