#!/usr/bin/env python3
"""Experiment 701 — KV260 Ising v3 Synthesis (Vivado or yosys proxy).

**Researcher summary (RETRO-072):**
    The KV260 FPGA board arrived on 2026-04-20 (RETRO-070 closed in milestone .52)
    and the Ising sampler v3 RTL (ising_sampler_v3.v) with EMA inertia dynamics was
    written during Exp 648.  RETRO-072 was opened because Vivado was not installed
    when the board arrived, so synthesis could not run.

    This experiment resolves RETRO-072 by:
    1. Checking whether Vivado (commercial) or yosys (open-source) is available.
    2. Running whichever tool is found on ising_sampler_v3.v targeting the KV260.
    3. Parsing the utilization and timing reports.
    4. Reporting a structured synthesis result with ``honest_verdict``.

    A bitstream is NOT generated here — that is deferred to RETRO-073 in .54.
    Synthesis report alone is sufficient for .53.

**Why synthesis first, bitstream second?**
    Synthesis tells us whether the RTL compiles cleanly and meets our
    50 MHz / 20% LUT targets (REQ-HW-037, REQ-HW-038) before investing in
    the longer implementation + place-and-route run.  A failing synthesis
    means the RTL needs revision — better to know now than after a 4-hour PnR.

**Tool fallback hierarchy:**
    1. Vivado 2024.2 (free WebPACK tier) — full timing-aware synthesis.
    2. yosys (open-source) — logic-only synthesis, no timing, LUT estimate only.
    3. No tool — records "synthesis_blocked_no_tool" and documents RETRO-072.

Spec: REQ-HW-037, REQ-HW-038,
      SCENARIO-HW-037, SCENARIO-HW-038, SCENARIO-HW-039
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root discovery (must precede local imports)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 701
TITLE = "KV260 Ising v3 Synthesis — Vivado or yosys proxy"
DELIVERABLE = "results/experiment_701_kv260_ising_v3_synthesis.json"
SCHEMA = "carnot.fpga.synthesis.v1"

# Target device: Kria KV260 Kria SOM, Zynq UltraScale+ CG
PART_NUMBER = "xck26-sfvc784-2LV-c"

# KV260 fabric has ~117,120 LUTs; REQ-HW-038 requires < 20 % = 23,424 LUTs.
KV260_TOTAL_LUTS = 117_120
LUT_BUDGET_PCT = 20
LUT_BUDGET = int(KV260_TOTAL_LUTS * LUT_BUDGET_PCT / 100)  # 23_424

# Timing target (REQ-HW-037): WNS >= 0 ns at 50 MHz means the design
# closes timing at the requested frequency.
TIMING_TARGET_MHZ = 50

RTL_PATH = _REPO_ROOT / "rtl" / "ising_sampler_v3.v"
TCL_DIR = _REPO_ROOT / "tcl"
RESULTS_DIR = _REPO_ROOT / "results"
KNOWN_ISSUES_PATH = _REPO_ROOT / "ops" / "known-issues.md"


# ---------------------------------------------------------------------------
# Tool availability checks
# ---------------------------------------------------------------------------


def check_vivado() -> bool:
    """Return True when Vivado is installed and responds to ``vivado -version``.

    Vivado exits with return-code 0 on success.  Any non-zero return or
    FileNotFoundError (tool not on PATH) means it is unavailable.
    """
    try:
        result = subprocess.run(
            ["vivado", "-version"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_yosys() -> bool:
    """Return True when yosys is installed and responds to ``yosys -V``.

    yosys exits with return-code 0 on success.
    """
    try:
        result = subprocess.run(
            ["yosys", "-V"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Vivado synthesis
# ---------------------------------------------------------------------------


def write_vivado_tcl(tcl_path: Path) -> None:
    """Write the Vivado non-project synthesis TCL script.

    Non-project mode is used because it avoids creating a Vivado project
    directory tree (.xpr, cache, logs) — the script runs, writes reports,
    and exits cleanly.  Batch mode (-mode batch) means no GUI starts.

    The ``synth_design`` command is Vivado's main synthesis entry-point.
    We pass ``-top`` (the top-level module name) and ``-part`` (the exact
    device part number for the KV260 Kria SOM).
    """
    tcl_path.parent.mkdir(parents=True, exist_ok=True)
    tcl_path.write_text(
        f"""\
# Vivado non-project synthesis for ising_sampler_v3 — auto-generated by Exp 701.
# Target: {PART_NUMBER}
# Spec: REQ-HW-037, REQ-HW-038
read_verilog {RTL_PATH}
synth_design -top ising_sampler_v3 -part {PART_NUMBER}
report_utilization -file {RESULTS_DIR / "ising_v3_utilization.rpt"}
report_timing_summary -file {RESULTS_DIR / "ising_v3_timing.rpt"}
"""
    )


def run_vivado_synthesis(tcl_path: Path) -> tuple[int, str, str]:
    """Run Vivado in batch mode with the given TCL script.

    Returns (returncode, stdout, stderr).  A returncode of 0 means Vivado
    completed without fatal errors; non-zero means synthesis failed.
    """
    result = subprocess.run(
        ["vivado", "-mode", "batch", "-source", str(tcl_path)],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    return result.returncode, result.stdout, result.stderr


def parse_utilization_report(report_path: Path) -> dict[str, int | None]:
    """Extract LUT, FF, and BRAM counts from a Vivado utilization report.

    Vivado's utilization report uses a table format like::

        | LUT as Logic             |  1234 |  ...

    We use regex to pull the integer from the second column.  Returns
    ``None`` for any field that cannot be parsed (rather than raising)
    so the caller can log a warning without crashing.

    Why regex over XML: Vivado's batch-mode text report is stable across
    versions and does not require the Vivado Tcl API at parse time.  The
    XML report requires vivado_report_utils which is only available inside
    a Vivado session.
    """
    if not report_path.exists():
        return {"LUT_count": None, "FF_count": None, "BRAM_count": None}

    text = report_path.read_text(errors="replace")

    def _extract(pattern: str) -> int | None:
        m = re.search(pattern, text)
        if m:
            try:
                return int(m.group(1).replace(",", "").strip())
            except ValueError:
                return None
        return None

    return {
        "LUT_count": _extract(r"LUT as Logic\s*\|\s*([\d,]+)"),
        "FF_count": _extract(r"Register as Flip Flop\s*\|\s*([\d,]+)"),
        "BRAM_count": _extract(r"Block RAM Tile\s*\|\s*([\d,]+)"),
    }


def parse_timing_report(report_path: Path) -> tuple[float | None, bool | None]:
    """Extract Worst Negative Slack (WNS) from a Vivado timing summary report.

    Returns ``(WNS_ns, timing_met)`` where:
    - ``WNS_ns`` is the slack in nanoseconds (float).
    - ``timing_met`` is True when WNS >= 0 (design closes timing at target).

    WNS < 0 means the design violates the timing constraint — the critical path
    is longer than one clock period at the target frequency (50 MHz = 20 ns).
    WNS >= 0 means the design has slack to spare.

    Timing not met does NOT necessarily mean the design fails at lower
    frequencies — it means the 50 MHz target is not achievable as-is.
    """
    if not report_path.exists():
        return None, None

    text = report_path.read_text(errors="replace")
    m = re.search(r"WNS\(ns\)\s+TNS\(ns\).*?\n\s*(-?[\d.]+)", text, re.DOTALL)
    if not m:
        m = re.search(r"Design Timing Summary.*?(-?[\d.]+)", text, re.DOTALL)
    if m:
        try:
            wns = float(m.group(1))
            return wns, wns >= 0.0
        except ValueError:
            pass
    return None, None


# ---------------------------------------------------------------------------
# Yosys proxy synthesis
# ---------------------------------------------------------------------------


def write_yosys_script(ys_path: Path) -> None:
    """Write the yosys synthesis script.

    ``synth -top <module>`` runs the standard synthesis flow (read, elaborate,
    technology-map to generic cells) and prints a ``stat`` summary with cell
    counts.  We capture stdout to parse the LUT estimate.

    yosys ``stat`` does not give FPGA-specific LUT counts — it gives generic
    cell counts after technology mapping to standard cells.  We treat the
    total cell count as a proxy for LUT usage (conservative over-estimate).
    """
    ys_path.parent.mkdir(parents=True, exist_ok=True)
    ys_path.write_text(
        f"""\
# yosys proxy synthesis for ising_sampler_v3 — auto-generated by Exp 701.
# Spec: REQ-HW-037 (proxy), REQ-HW-038 (proxy)
read_verilog {RTL_PATH}
synth -top ising_sampler_v3
stat
"""
    )


def run_yosys_synthesis(ys_path: Path) -> tuple[int, str, str]:
    """Run yosys with the given script.

    Returns (returncode, stdout, stderr).
    """
    result = subprocess.run(
        ["yosys", str(ys_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    return result.returncode, result.stdout, result.stderr


def parse_yosys_stat(stdout: str) -> int | None:
    """Extract estimated LUT count from yosys stat output.

    yosys stat prints lines like::

        Number of cells:                 1234

    or after technology mapping::

        $_DFF_P_                          56

    We look for ``Number of cells`` as the primary proxy.  If that is absent
    we fall back to summing all cell-type counts.

    Why this is a proxy: yosys maps to its own internal cell library, not the
    Xilinx LUT6 primitive.  The cell count is a rough upper bound; actual
    LUT usage after Vivado synthesis would typically be 40-60% lower due to
    logic sharing and LUT packing.
    """
    # Primary: "Number of cells: N"
    m = re.search(r"Number of cells:\s+([\d,]+)", stdout)
    if m:
        try:
            return int(m.group(1).replace(",", "").strip())
        except ValueError:
            pass

    # Fallback: sum all "$_<type>_: N" lines
    total = 0
    for m in re.finditer(r"\$\w+\s+([\d,]+)", stdout):
        try:
            total += int(m.group(1).replace(",", "").strip())
        except ValueError:
            pass
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# Known-issues update
# ---------------------------------------------------------------------------


def append_known_issue(note: str) -> None:
    """Append a RETRO-072 note to ops/known-issues.md (non-destructively).

    The CLAUDE.md documentation policy forbids removing existing content from
    ops/spec documents.  This function always appends — it never overwrites.
    """
    if not KNOWN_ISSUES_PATH.exists():
        return
    existing = KNOWN_ISSUES_PATH.read_text()
    if note not in existing:
        with KNOWN_ISSUES_PATH.open("a") as fh:
            fh.write(f"\n{note}\n")


# ---------------------------------------------------------------------------
# honest_verdict logic
# ---------------------------------------------------------------------------


def compute_honest_verdict(
    synthesis_tool: str,
    timing_met: bool | None,
    lut_count: int | None,
) -> str:
    """Map synthesis outcome to a canonical honest_verdict string.

    The honest_verdict follows the same pattern as other Carnot experiments:
    it records what was actually proven, not what was attempted.

    - ``synthesis_timing_met``: Vivado ran and WNS >= 0 (timing closed at target).
    - ``synthesis_timing_failed``: Vivado ran but WNS < 0 (timing violation).
    - ``synthesis_lut_estimate_only``: yosys ran and produced a cell count proxy.
    - ``synthesis_blocked_no_tool``: Neither Vivado nor yosys is available.

    Why separate "failed" from "estimate_only": timing failure is actionable
    (we need to refactor the RTL), whereas "estimate only" is informational
    (we have a proxy but need Vivado for a real verdict).
    """
    if synthesis_tool == "vivado":
        if timing_met is True:
            return "synthesis_timing_met"
        return "synthesis_timing_failed"
    if synthesis_tool == "yosys":
        if lut_count is not None:
            return "synthesis_lut_estimate_only"
        return "synthesis_blocked_yosys_parse_failed"
    return "synthesis_blocked_no_tool"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run KV260 Ising v3 synthesis and produce structured JSON deliverable."""

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=120,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    with watchdog:
        # ----------------------------------------------------------------
        # 1. Tool availability
        # ----------------------------------------------------------------
        vivado_available = check_vivado()
        yosys_available = check_yosys()

        print(f"[Exp {EXP_ID}] vivado_available={vivado_available} "
              f"yosys_available={yosys_available}")

        synthesis_tool: str
        lut_count: int | None = None
        ff_count: int | None = None
        bram_count: int | None = None
        timing_met: bool | None = None
        wns_ns: float | None = None

        # ----------------------------------------------------------------
        # 2. Run synthesis
        # ----------------------------------------------------------------
        if vivado_available:
            synthesis_tool = "vivado"
            tcl_path = TCL_DIR / "synth_ising_v3.tcl"
            write_vivado_tcl(tcl_path)

            print(f"[Exp {EXP_ID}] Running Vivado synthesis "
                  f"(part={PART_NUMBER}) ...")
            rc, _stdout, _stderr = run_vivado_synthesis(tcl_path)
            print(f"[Exp {EXP_ID}] Vivado returncode={rc}")

            util = parse_utilization_report(
                RESULTS_DIR / "ising_v3_utilization.rpt"
            )
            lut_count = util["LUT_count"]
            ff_count = util["FF_count"]
            bram_count = util["BRAM_count"]

            wns_ns, timing_met = parse_timing_report(
                RESULTS_DIR / "ising_v3_timing.rpt"
            )
            print(f"[Exp {EXP_ID}] LUT={lut_count} FF={ff_count} "
                  f"BRAM={bram_count} WNS={wns_ns} timing_met={timing_met}")

        elif yosys_available:
            synthesis_tool = "yosys"
            ys_path = _REPO_ROOT / "yosys_script.ys"
            write_yosys_script(ys_path)

            print(f"[Exp {EXP_ID}] Running yosys proxy synthesis ...")
            rc, stdout, _stderr = run_yosys_synthesis(ys_path)
            print(f"[Exp {EXP_ID}] yosys returncode={rc}")

            lut_count = parse_yosys_stat(stdout)
            print(f"[Exp {EXP_ID}] yosys proxy LUT estimate={lut_count}")

        else:
            synthesis_tool = "none_available"
            note = (
                "## RETRO-072 update (Exp 701, 20260422)\n"
                "Vivado not installed; yosys not found.  "
                "Synthesis blocked.  Install one of:\n"
                "  - AMD Vivado 2024.2 (free WebPACK from xilinx.com)\n"
                "  - yosys (`sudo pacman -S yosys` on CachyOS)\n"
                "RETRO-073 opened for milestone .54."
            )
            append_known_issue(note)
            print(f"[Exp {EXP_ID}] No synthesis tool found — "
                  "recording synthesis_blocked_no_tool.")

        # ----------------------------------------------------------------
        # 3. Determine honest_verdict and RETRO-072 status
        # ----------------------------------------------------------------
        honest_verdict = compute_honest_verdict(
            synthesis_tool, timing_met, lut_count
        )
        retro_072_resolved = honest_verdict in (
            "synthesis_timing_met",
            "synthesis_lut_estimate_only",
        )

        # ----------------------------------------------------------------
        # 4. Check REQ-HW-038 budget (when data is available)
        # ----------------------------------------------------------------
        lut_budget_met: bool | None = None
        if lut_count is not None:
            lut_budget_met = lut_count < LUT_BUDGET
            print(f"[Exp {EXP_ID}] LUT budget: {lut_count} / {LUT_BUDGET} "
                  f"(<{LUT_BUDGET_PCT}% KV260 fabric) => met={lut_budget_met}")

        # ----------------------------------------------------------------
        # 5. Write deliverable
        # ----------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": SCHEMA,
                "synthesis_tool": synthesis_tool,
                "part_number": PART_NUMBER,
                "rtl_path": str(RTL_PATH.relative_to(_REPO_ROOT)),
                "LUT_count": lut_count,
                "FF_count": ff_count,
                "BRAM_count": bram_count,
                "LUT_budget": LUT_BUDGET,
                "LUT_budget_pct": LUT_BUDGET_PCT,
                "lut_budget_met": lut_budget_met,
                "timing_target_mhz": TIMING_TARGET_MHZ,
                "timing_met": timing_met,
                "WNS_ns": wns_ns,
                "retro_072_resolved": retro_072_resolved,
                "honest_verdict": honest_verdict,
            },
            status="success" if retro_072_resolved else "blocked",
        )

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.write_text(json.dumps(artifact, indent=2))
        print(f"[Exp {EXP_ID}] Deliverable written: {output_path}")
        print(f"[Exp {EXP_ID}] honest_verdict={honest_verdict} "
              f"retro_072_resolved={retro_072_resolved}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
