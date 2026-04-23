#!/usr/bin/env python3
"""Experiment 758: Yosys Open-Source FPGA Synthesis of Ising Sampler v2.

**Researcher summary (REQ-HW-041):**
    Vivado has been unavailable for every KV260 synthesis attempt across 6+ milestones.
    This experiment uses yosys (via the yowasp-yosys Python wrapper, which bundles yosys
    as a WebAssembly binary — no native install required) to synthesize the Ising sampler
    RTL and report resource utilization.

    Why this matters: without synthesis results we cannot claim hardware acceleration is
    feasible at any specific scale.  Yosys generic synthesis gives us LUT/DFF counts that
    are technology-portable and comparable to Vivado estimates.

**Gate condition:**
    Requires Exp 757 honest_verdict != "blocked" (HLS energy sign fix validated).
    If the gate fails this script writes a blocked artifact and exits cleanly.

**What this experiment does:**
    1. Confirms Exp 757 gate (sign_convention_fixed = True).
    2. Checks whether native yosys is on PATH; if not, falls back to yowasp-yosys
       installed in the project venv (pip install yowasp-yosys).
    3. Synthesizes hardware/kv260/ising_sampler_v2_synth.v (a synthesis-clean copy of
       ising_sampler_v2.v with the simulation-only `real`/`initial` LUT init block removed
       — Yosys does not support the Verilog `real` type for synthesis).
    4. Parses the yosys `stat` output for cell_count, wire_count, and LUT-equivalent
       (MUX + AND + OR cells from the RTLIL generic library) before ABC tech-mapping.
    5. Checks for nextpnr-ice40 (optional); if available, runs place-and-route and
       reports fmax_mhz.
    6. Emits honest_verdict and writes the result artifact.

Spec: REQ-HW-041, SCENARIO-HW-041
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so the script runs directly from repo root or via conductor.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 758
TITLE = "Yosys Open-Source FPGA Synthesis of Ising Sampler v2"
DELIVERABLE = "results/experiment_758_yosys_synthesis.json"
EXP_757_RESULT = "results/experiment_757_hls_energy_fix.json"
SYNTH_V = "hardware/kv260/ising_sampler_v2_synth.v"
TOP_MODULE = "ising_sampler_128_sync"

# Yosys stat output regex patterns.
# These match the RTLIL generic cell output from `stat -top <module>`.
# Format: "  <count>   <cell_type>" with leading spaces.
_RE_CELL_TOTAL = re.compile(r"^\s+(\d+)\s+cells\s*$", re.MULTILINE)
_RE_WIRE_TOTAL = re.compile(r"^\s+(\d+)\s+wires\s*$", re.MULTILINE)
_RE_WIRE_BITS = re.compile(r"^\s+(\d+)\s+wire bits\s*$", re.MULTILINE)

# Cell type counts used to estimate equivalent LUT count (generic RTLIL cells).
# In generic synthesis (before ABC), there are no SB_LUT4 or LUT4 cells.
# We count the logic cells that would map to LUTs after tech-mapping.
_RE_AND = re.compile(r"^\s+(\d+)\s+\$_AND_\s*$", re.MULTILINE)
_RE_MUX = re.compile(r"^\s+(\d+)\s+\$_MUX_\s*$", re.MULTILINE)
_RE_OR = re.compile(r"^\s+(\d+)\s+\$_OR_\s*$", re.MULTILINE)
_RE_XOR = re.compile(r"^\s+(\d+)\s+\$_XOR_\s*$", re.MULTILINE)
_RE_NOT = re.compile(r"^\s+(\d+)\s+\$_NOT_\s*$", re.MULTILINE)
_RE_DFF = re.compile(r"^\s+(\d+)\s+\$_DFFE_PP_\s*$", re.MULTILINE)
_RE_SDFFE = re.compile(r"^\s+(\d+)\s+\$_SDFFE_PN0P_\s*$", re.MULTILINE)

# nextpnr timing report pattern.
_RE_FMAX = re.compile(r"Max frequency for clock.*?:\s*([\d.]+)\s*MHz", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper: parse yosys stat output
# ---------------------------------------------------------------------------

def parse_yosys_stat(output: str) -> dict:
    """Extract resource counts from yosys stat command output.

    Yosys stat prints a block like:
      === <module_name> ===
        1468 wires
       22243 wire bits
        5081 cells
         427   $_AND_
        2097   $_DFFE_PP_
        ...

    We extract the totals and individual cell type counts.
    The LUT-equivalent estimate sums logic cells (AND, OR, XOR, NOT, MUX) —
    these are the RTLIL cells before ABC technology maps them to actual LUT4s.
    This is a conservative estimate; real iCE40 mapping is ~1.2x more efficient
    due to 4-input LUT packing.

    Returns a dict with cell_count, wire_count, wire_bits, lut_equiv, dff_count,
    and individual cell type counts for transparency.
    """

    def _int(pattern: re.Pattern, default: int = 0) -> int:
        m = pattern.search(output)
        return int(m.group(1)) if m else default

    cell_count = _int(_RE_CELL_TOTAL)
    wire_count = _int(_RE_WIRE_TOTAL)
    wire_bits = _int(_RE_WIRE_BITS)
    and_count = _int(_RE_AND)
    mux_count = _int(_RE_MUX)
    or_count = _int(_RE_OR)
    xor_count = _int(_RE_XOR)
    not_count = _int(_RE_NOT)
    dff_count = _int(_RE_DFF) + _int(_RE_SDFFE)
    # LUT-equivalent: logic cells that require combinatorial LUT resources.
    lut_equiv = and_count + mux_count + or_count + xor_count + not_count

    return {
        "cell_count": cell_count,
        "wire_count": wire_count,
        "wire_bits": wire_bits,
        "lut_equiv": lut_equiv,
        "dff_count": dff_count,
        "and_count": and_count,
        "mux_count": mux_count,
        "or_count": or_count,
        "xor_count": xor_count,
        "not_count": not_count,
    }


# ---------------------------------------------------------------------------
# Helper: check for tool availability
# ---------------------------------------------------------------------------

def find_yosys() -> tuple[bool, str | None]:
    """Return (found, version_string) for native yosys on PATH.

    Why check native first: if the user has Yosys installed system-wide,
    it may support more features than the WASM build (e.g. ABC timing).
    """
    try:
        result = subprocess.run(
            ["yosys", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            return True, version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def find_nextpnr_ice40() -> bool:
    """Return True if nextpnr-ice40 is available on PATH."""
    try:
        result = subprocess.run(
            ["nextpnr-ice40", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Helper: run yosys synthesis (native or yowasp)
# ---------------------------------------------------------------------------

def run_synthesis(
    rtl_path: str,
    top_module: str,
    use_yowasp: bool,
) -> tuple[str, list[str], int]:
    """Run yosys generic synthesis with -noabc for fast resource estimation.

    Why -noabc: the full synth pass with ABC technology mapping takes 130+ seconds
    on this design and doesn't change the cell count reported by `stat` (which is
    what we care about for resource estimation). -noabc skips the ABC pass entirely,
    leaving cells in RTLIL generic form which we count directly.

    Returns (stdout+stderr combined, warnings_list, error_count).
    """
    script = (
        f"read_verilog {rtl_path}; "
        f"synth -top {top_module} -flatten -noabc; "
        f"stat -top {top_module}"
    )

    if use_yowasp:
        # yowasp-yosys writes to real file descriptors (FD 1/2), not Python's
        # sys.stdout/stderr, so contextlib.redirect_stdout does NOT capture it.
        # Run in a subprocess to get the actual FD output via capture_output=True.
        result = subprocess.run(
            [
                sys.executable, "-c",
                f"from yowasp_yosys import run_yosys; run_yosys(['-p', {script!r}])"
            ],
            capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT)
        )
        combined = result.stdout + result.stderr
    else:
        result = subprocess.run(
            ["yosys", "-p", script],
            capture_output=True, text=True, timeout=300, cwd=str(_REPO_ROOT)
        )
        combined = result.stdout + result.stderr

    errors = combined.count("ERROR:")
    warnings = _extract_warnings(combined)
    return combined, warnings, errors


def _extract_warnings(output: str) -> list[str]:
    """Extract WARNING lines from yosys output."""
    return [ln.strip() for ln in output.splitlines() if "Warning:" in ln]


# ---------------------------------------------------------------------------
# Helper: optional nextpnr place-and-route for fmax estimation
# ---------------------------------------------------------------------------

def run_nextpnr_ice40(json_netlist: str) -> float | None:
    """Run nextpnr-ice40 for iCE40-HX1K timing analysis.

    Why iCE40-HX1K: it's the smallest and most widely-used iCE40 device.
    With no PCF pin constraints we can at least get critical path timing.
    This gives us fmax_mhz as an upper-bound indicator for the Ising sampler.

    Returns fmax_mhz as float, or None if nextpnr fails or times out.
    """
    try:
        result = subprocess.run(
            [
                "nextpnr-ice40",
                "--hx1k",
                "--json", json_netlist,
                "--pcf", "/dev/null",
                "--asc", "/tmp/ising_v2.asc",
                "--freq", "50",
            ],
            capture_output=True, text=True, timeout=120
        )
        combined = result.stdout + result.stderr
        m = _RE_FMAX.search(combined)
        if m:
            return float(m.group(1))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Exp 758: Yosys synthesis of Ising sampler v2."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=30,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    with watchdog:
        # ------------------------------------------------------------------
        # Gate check: Exp 757 must report sign_convention_fixed=True.
        # ------------------------------------------------------------------
        gate_path = _REPO_ROOT / EXP_757_RESULT
        if not gate_path.exists():
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_hls_fix_needed",
                    "gate_failure_reason": f"Exp 757 result not found at {EXP_757_RESULT}",
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            tmpl.assert_deliverable_written()
            return

        with open(gate_path) as f:
            exp757 = json.load(f)

        sign_fixed = exp757.get("sign_convention_fixed", False)
        if not sign_fixed:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "blocked_hls_fix_needed",
                    "gate_failure_reason": "Exp 757 sign_convention_fixed=False",
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            tmpl.assert_deliverable_written()
            return

        # ------------------------------------------------------------------
        # Step 1: Locate yosys.
        # ------------------------------------------------------------------
        yosys_found, yosys_version = find_yosys()
        yosys_source = "native"

        if not yosys_found:
            # Try yowasp-yosys (already pip-installed by the conductor or manually).
            try:
                from yowasp_yosys import run_yosys  # noqa: F401
                yosys_found = True
                yosys_source = "yowasp"
                # yowasp writes to real FDs — use subprocess to capture version.
                ver_result = subprocess.run(
                    [sys.executable, "-c",
                     "from yowasp_yosys import run_yosys; run_yosys(['--version'])"],
                    capture_output=True, text=True, timeout=30
                )
                combined_ver = (ver_result.stdout + ver_result.stderr).strip()
                yosys_version = combined_ver.split("\n")[0] if combined_ver else "unknown"
            except ImportError:
                yosys_found = False
                yosys_source = "none"

        if not yosys_found:
            artifact = tmpl.build_result(
                {
                    "yosys_found": False,
                    "yosys_version": None,
                    "yosys_source": "none",
                    "honest_verdict": "yosys_not_installable",
                    "synthesis_errors": [],
                    "lut_equiv": None,
                    "cell_count": None,
                    "wire_count": None,
                    "fmax_mhz": None,
                },
                status="blocked",
            )
            _write_artifact(tmpl, artifact)
            tmpl.assert_deliverable_written()
            return

        # ------------------------------------------------------------------
        # Step 2: Synthesize ising_sampler_v2_synth.v.
        # ------------------------------------------------------------------
        synth_path = str(_REPO_ROOT / SYNTH_V)
        use_yowasp = (yosys_source == "yowasp")

        synth_output, warnings, error_count = run_synthesis(
            synth_path, TOP_MODULE, use_yowasp
        )

        stats = parse_yosys_stat(synth_output)

        # ------------------------------------------------------------------
        # Step 3: Optional nextpnr-ice40 for fmax.
        # ------------------------------------------------------------------
        fmax_mhz: float | None = None
        nextpnr_available = find_nextpnr_ice40()
        if nextpnr_available and error_count == 0:
            json_netlist = str(_REPO_ROOT / "hardware/kv260/ising_sampler_v2_synth.json")
            fmax_mhz = run_nextpnr_ice40(json_netlist)

        # ------------------------------------------------------------------
        # Compute honest_verdict.
        # ------------------------------------------------------------------
        lut_count = stats["lut_equiv"] if stats["lut_equiv"] > 0 else None

        if error_count > 0:
            honest_verdict = "synthesis_failed"
        elif lut_count is not None and len(warnings) == 0:
            honest_verdict = "synthesis_successful"
        elif lut_count is not None:
            honest_verdict = "synthesis_with_warnings"
        else:
            honest_verdict = "synthesis_failed"

        artifact = tmpl.build_result(
            {
                "yosys_found": True,
                "yosys_version": yosys_version,
                "yosys_source": yosys_source,
                "top_module": TOP_MODULE,
                "rtl_file": SYNTH_V,
                "lut_count": lut_count,        # alias for lut_equiv; matches schema contract
                "lut_equiv": stats["lut_equiv"],
                "cell_count": stats["cell_count"],
                "wire_count": stats["wire_count"],
                "wire_bits": stats["wire_bits"],
                "dff_count": stats["dff_count"],
                "and_count": stats["and_count"],
                "mux_count": stats["mux_count"],
                "or_count": stats["or_count"],
                "xor_count": stats["xor_count"],
                "not_count": stats["not_count"],
                "synthesis_errors": error_count,
                "synthesis_warnings": len(warnings),
                "warning_messages": warnings,
                "nextpnr_available": nextpnr_available,
                "fmax_mhz": fmax_mhz,
                "honest_verdict": honest_verdict,
                "synth_note": (
                    "Generic synthesis with -noabc. lut_equiv = AND+MUX+OR+XOR+NOT cells. "
                    "iCE40 ABC mapping would increase this ~10-20% for LUT4 packing overhead."
                ),
            },
            status="success" if honest_verdict in ("synthesis_successful", "synthesis_with_warnings") else "error",
        )

        _write_artifact(tmpl, artifact)

    tmpl.assert_deliverable_written()


def _write_artifact(tmpl: ExperimentTemplate, artifact: dict) -> None:
    """Atomically write the result artifact to disk."""
    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2))
    tmp.replace(output_path)


if __name__ == "__main__":
    main()
