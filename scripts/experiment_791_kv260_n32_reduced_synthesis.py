#!/usr/bin/env python3
"""Experiment 791 — KV260 N=32 Reduced-Spin iCE40 Synthesis and Place-and-Route.

**Research question:**
    Exp 776 failed iCE40 HX8K place-and-route timing with the N=64 Ising sampler
    (~2821 LUTs after synth_ice40).  Does halving to N=32 spins with MAX_DEGREE=8
    — which cuts the coupling matrix routing in half — allow nextpnr-ice40 to close
    timing and produce a real .bin bitstream?

**Why N=32 is the right fix:**
    The iCE40 HX8K has 7680 LUTs and simpler LUT4 primitives than Xilinx.  N=64
    consumed ~37% of the HX8K after synth_ice40, leaving limited routing headroom
    for the dense coupling matrix wiring.  N=32 with MAX_DEGREE=8 is estimated at
    600-800 LUTs (~8-10% utilization), giving nextpnr-ice40 90% routing slack and
    a much shorter critical path through the h_eff accumulator logic.

**Honest verdict mapping:**
    bitstream_generated_n32_ice40:  icepack produced a .bin file, PnR timing closed.
    pnr_successful_no_bitstream:    nextpnr succeeded but icepack was unavailable.
    pnr_success_lut_fit:            PnR succeeded and LUT utilization < 90%.
    pnr_failed_timing_n32:          nextpnr ran but could not close timing at N=32.
    tools_unavailable:              nextpnr-ice40 (native or yowasp) not found.

Spec: REQ-HW-043, SCENARIO-HW-043
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

DELIVERABLE = "results/experiment_791_kv260_n32_reduced_synthesis.json"
RTL_FILE = "hardware/kv260/ising_sampler_v2.v"
SYNTH_SCRIPT = "hardware/kv260/synth_yosys_ice40_n32.ys"
ICE40_JSON = "hardware/kv260/ising_sampler_n32_ice40.json"
ICE40_ASC = "hardware/kv260/ising_sampler_n32.asc"
ICE40_BIN = "hardware/kv260/ising_sampler_n32.bin"

# Clock frequency for nextpnr PnR — conservative vs Exp 776's 12 MHz to give
# more routing slack on the dense h_eff accumulator wiring.
CLOCK_MHZ = 8


def _run(cmd: list[str], timeout: int = 300) -> tuple[int, str, str]:
    """Run a subprocess command and return (returncode, stdout, stderr).

    Each tool invocation is capped at the given timeout to prevent a stuck
    synthesis tool from consuming the full 45-minute watchdog budget.
    FileNotFoundError is caught and treated as returncode=-1 so the caller
    can check rc != 0 without a separate exception path.
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
        return -1, "", f"Command timed out after {timeout}s: {cmd}"
    except FileNotFoundError:
        return -1, "", f"Executable not found: {cmd[0]}"


def _which(name: str) -> bool:
    """Return True if the executable is reachable on PATH."""
    rc, _, _ = _run(["which", name], timeout=5)
    return rc == 0


def _find_yosys() -> tuple[bool, str]:
    """Locate Yosys (native binary or yowasp Python wrapper).

    Priority order:
      1. Native 'yosys' on PATH — fastest, no Python overhead.
      2. yowasp_yosys Python module — works without native Yosys installed.

    Returns (found, command_to_use).  The command string is split()-able
    so callers can prepend it to argument lists uniformly.
    """
    if _which("yosys"):
        return True, "yosys"
    rc, _, _ = _run([sys.executable, "-m", "yowasp_yosys", "--version"], timeout=30)
    if rc == 0:
        return True, f"{sys.executable} -m yowasp_yosys"
    return False, ""


def _find_nextpnr_ice40() -> tuple[bool, str]:
    """Locate nextpnr-ice40 (native binary or yowasp Python wrapper).

    Priority order:
      1. Native 'nextpnr-ice40' on PATH.
      2. yowasp_nextpnr_ice40 Python module (WASM-compiled, no build needed).
      3. Attempt pip install yowasp-nextpnr-ice40 and retry.

    Returns (found, command_to_use).
    """
    if _which("nextpnr-ice40"):
        return True, "nextpnr-ice40"

    rc, out, _ = _run([sys.executable, "-m", "yowasp_nextpnr_ice40", "--version"], timeout=60)
    if rc == 0 or "nextpnr" in out.lower():
        return True, f"{sys.executable} -m yowasp_nextpnr_ice40"

    # Attempt installation — yowasp-nextpnr-ice40 is the easiest path on a machine
    # without native FPGA tools.
    rc2, _, _ = _run(
        [sys.executable, "-m", "pip", "install", "yowasp-nextpnr-ice40"],
        timeout=120,
    )
    if rc2 == 0:
        rc3, out3, _ = _run(
            [sys.executable, "-m", "yowasp_nextpnr_ice40", "--version"], timeout=60
        )
        if rc3 == 0 or "nextpnr" in out3.lower():
            return True, f"{sys.executable} -m yowasp_nextpnr_ice40"

    return False, ""


def _find_icepack() -> tuple[bool, str]:
    """Locate icepack (native binary or yowasp_icepack Python wrapper).

    icepack converts nextpnr's ASCII bitstream (.asc) to a binary blob (.bin)
    that can be flashed to an iCE40 device.  It is part of the icestorm suite.
    yowasp-icestorm provides a Python-wrapped version.

    Returns (found, command_to_use).
    """
    if _which("icepack"):
        return True, "icepack"

    rc, _, _ = _run([sys.executable, "-m", "yowasp_icepack", "--version"], timeout=30)
    if rc == 0:
        return True, f"{sys.executable} -m yowasp_icepack"

    rc2, _, _ = _run(
        [sys.executable, "-m", "pip", "install", "yowasp-icestorm"],
        timeout=120,
    )
    if rc2 == 0:
        rc3, _, _ = _run([sys.executable, "-m", "yowasp_icepack", "--version"], timeout=30)
        if rc3 == 0:
            return True, f"{sys.executable} -m yowasp_icepack"

    return False, ""


def _run_yosys_synthesis(
    repo_root: Path, yosys_cmd: str
) -> tuple[bool, int | None, str]:
    """Run Yosys synth_ice40 with N_SPINS=32, MAX_DEGREE=8 via the n32 script.

    The synth script uses chparam to override the module defaults at elaboration
    time — no RTL edits needed.  We run from the repo root so relative paths in
    the .ys file resolve correctly.

    Returns (synthesis_ok, lut_count_n32, combined_log).
    lut_count_n32 is None when the stat output could not be parsed.
    """
    script_path = repo_root / SYNTH_SCRIPT
    cmd = yosys_cmd.split() + [str(script_path)]
    rc, stdout, stderr = _run(cmd, timeout=180)
    combined = stdout + "\n" + stderr

    json_out = repo_root / ICE40_JSON
    synthesis_ok = rc == 0 and json_out.exists()

    # Parse LUT count from Yosys 'stat' output lines such as:
    #   "   SB_LUT4:    612"
    # or the generic:
    #   "   Number of cells:  612"
    lut_count_n32: int | None = None
    for line in combined.splitlines():
        m = re.search(r"SB_LUT4\s*:\s*(\d+)", line)
        if m:
            lut_count_n32 = int(m.group(1))
            break
        m2 = re.search(r"Number of cells\s*:\s*(\d+)", line)
        if m2 and lut_count_n32 is None:
            lut_count_n32 = int(m2.group(1))

    return synthesis_ok, lut_count_n32, combined


def _run_nextpnr(
    repo_root: Path, nextpnr_cmd: str
) -> tuple[bool, float | None, float | None, str]:
    """Run nextpnr-ice40 place-and-route targeting the HX8K at CLOCK_MHZ MHz.

    We use --pcf /dev/null because we have no pin-constraint file for a generic
    iCE40 validation run (we are checking that the RTL closes timing, not
    programming a physical board).

    Returns (pnr_success_ice40, critical_path_ns, lut_utilization_pct, log).
    critical_path_ns is derived from the fmax line; lut_utilization_pct from the
    logic cells / ICESTORM_LC line.
    """
    json_in = repo_root / ICE40_JSON
    asc_out = repo_root / ICE40_ASC

    cmd = nextpnr_cmd.split() + [
        "--hx8k",
        "--json", str(json_in),
        "--pcf", "/dev/null",
        "--asc", str(asc_out),
        "--freq", str(CLOCK_MHZ),
    ]
    rc, stdout, stderr = _run(cmd, timeout=300)
    combined = stdout + "\n" + stderr

    pnr_success_ice40 = rc == 0 and asc_out.exists()

    critical_path_ns: float | None = None
    lut_utilization_pct: float | None = None

    for line in combined.splitlines():
        ll = line.lower()
        if "max frequency" in ll and "mhz" in ll:
            m = re.search(r"([\d.]+)\s*MHz", line, re.IGNORECASE)
            if m:
                fmax_mhz = float(m.group(1))
                if fmax_mhz > 0:
                    critical_path_ns = round(1000.0 / fmax_mhz, 2)
        if "icestorm_lc" in ll or "logic cells" in ll:
            m2 = re.search(r"(\d+)/\s*(\d+)\s+(\d+)%", line)
            if m2:
                used = int(m2.group(1))
                total = int(m2.group(2))
                if total > 0:
                    lut_utilization_pct = round(100.0 * used / total, 1)

    return pnr_success_ice40, critical_path_ns, lut_utilization_pct, combined


def _run_icepack(repo_root: Path, icepack_cmd: str) -> tuple[bool, str | None]:
    """Convert the .asc ASCII bitstream to a binary .bin file.

    Returns (bitstream_generated, bitstream_path_str).
    bitstream_path_str is None if icepack fails or the .asc is absent.
    """
    asc_path = repo_root / ICE40_ASC
    bin_path = repo_root / ICE40_BIN

    if not asc_path.exists():
        return False, None

    cmd = icepack_cmd.split() + [str(asc_path), str(bin_path)]
    rc, _, _ = _run(cmd, timeout=120)
    success = rc == 0 and bin_path.exists()
    return success, str(bin_path) if success else None


def _check_synth_parameters_in_script(repo_root: Path) -> bool:
    """Verify the synthesis script references N_SPINS=32 and MAX_DEGREE=8.

    This is a defensive check — if the script was somehow overwritten without
    the chparam line, we would silently synthesize the N=64 default and get
    incorrect LUT count measurements.  Better to catch it early.
    """
    script = (repo_root / SYNTH_SCRIPT).read_text()
    return "N_SPINS 32" in script and "MAX_DEGREE 8" in script


def run_experiment(tmpl: ExperimentTemplate) -> dict:
    """Execute the N=32 iCE40 synthesis and place-and-route pipeline.

    Steps:
      1. Check tools (Yosys, nextpnr-ice40, icepack).
      2. Verify the synth script sets N_SPINS=32, MAX_DEGREE=8.
      3. Run Yosys synth_ice40 → netlist JSON.
      4. Run nextpnr-ice40 --hx8k --freq 8 → .asc.
      5. Run icepack → .bin.
      6. Determine honest_verdict from outcomes.

    Returns a flat dict of all result fields consumed by build_result().
    """
    repo_root = Path(tmpl._repo_root)

    # Step 1: Tool discovery
    yosys_found, yosys_cmd = _find_yosys()
    nextpnr_ice40_found, nextpnr_cmd = _find_nextpnr_ice40()
    icepack_found, icepack_cmd = _find_icepack()

    tools_available = {
        "yosys": yosys_found,
        "nextpnr_ice40": nextpnr_ice40_found,
        "icepack": icepack_found,
    }

    if not nextpnr_ice40_found:
        # Without nextpnr-ice40 we cannot produce a routed netlist or bitstream.
        # Record the tool availability map so the conductor knows what to install.
        return {
            "tools_available": tools_available,
            "synthesis_ok": False,
            "lut_count_n32": None,
            "pnr_success_ice40": False,
            "critical_path_ns": None,
            "lut_utilization_pct": None,
            "bitstream_generated": False,
            "bitstream_path": None,
            "honest_verdict": "tools_unavailable",
        }

    # Step 2: Guard — ensure synth script has correct parameters
    params_ok = _check_synth_parameters_in_script(repo_root)

    synthesis_ok = False
    lut_count_n32: int | None = None
    pnr_success_ice40 = False
    critical_path_ns: float | None = None
    lut_utilization_pct: float | None = None
    bitstream_generated = False
    bitstream_path: str | None = None

    # Step 3: Yosys synthesis
    if yosys_found and params_ok:
        synthesis_ok, lut_count_n32, _synth_log = _run_yosys_synthesis(repo_root, yosys_cmd)

    # Step 4: Place-and-route
    if synthesis_ok:
        pnr_success_ice40, critical_path_ns, lut_utilization_pct, _pnr_log = _run_nextpnr(
            repo_root, nextpnr_cmd
        )

    # Step 5: Bitstream generation
    if pnr_success_ice40 and icepack_found:
        bitstream_generated, bitstream_path = _run_icepack(repo_root, icepack_cmd)

    # Step 6: Verdict
    if bitstream_generated:
        honest_verdict = "bitstream_generated_n32_ice40"
    elif pnr_success_ice40 and lut_utilization_pct is not None and lut_utilization_pct < 90:
        honest_verdict = "pnr_success_lut_fit"
    elif pnr_success_ice40:
        honest_verdict = "pnr_successful_no_bitstream"
    elif synthesis_ok:
        honest_verdict = "pnr_failed_timing_n32"
    else:
        honest_verdict = "tools_unavailable"

    return {
        "tools_available": tools_available,
        "synthesis_ok": synthesis_ok,
        "lut_count_n32": lut_count_n32,
        "pnr_success_ice40": pnr_success_ice40,
        "critical_path_ns": critical_path_ns,
        "lut_utilization_pct": lut_utilization_pct,
        "bitstream_generated": bitstream_generated,
        "bitstream_path": bitstream_path,
        "honest_verdict": honest_verdict,
    }


def main() -> None:
    """Run Experiment 791: N=32 iCE40 synthesis and P&R for timing closure."""
    tmpl = ExperimentTemplate(
        exp_id=791,
        title="KV260 N=32 Reduced-Spin iCE40 Synthesis — Timing Fix for Exp 776",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )

    with ExperimentTimeoutWatchdog(791, timeout_minutes=45, result_path=DELIVERABLE):
        tmpl.setup()

        try:
            data = run_experiment(tmpl)
            artifact = tmpl.build_result(data, status="success")
        except Exception as exc:
            artifact = tmpl.build_result(
                {
                    "tools_available": {},
                    "synthesis_ok": False,
                    "lut_count_n32": None,
                    "pnr_success_ice40": False,
                    "critical_path_ns": None,
                    "lut_utilization_pct": None,
                    "bitstream_generated": False,
                    "bitstream_path": None,
                    "honest_verdict": "tools_unavailable",
                    "error": str(exc),
                },
                status="error",
            )

        output_path = Path(tmpl._repo_root) / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(
            f"[Exp 791] verdict={artifact.get('honest_verdict')}  "
            f"lut_count_n32={artifact.get('lut_count_n32')}  "
            f"bitstream_generated={artifact.get('bitstream_generated')}"
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
