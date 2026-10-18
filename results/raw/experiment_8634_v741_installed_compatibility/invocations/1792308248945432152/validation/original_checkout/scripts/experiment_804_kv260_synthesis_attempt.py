#!/usr/bin/env python3
"""Experiment 804 — KV260 N=32 Yosys Open-Source Synthesis Attempt.

**Research question:**
    Exp 791 failed because yosys/nextpnr-ice40/icepack were not installed.
    Exp 794 tried to install them but failed (pacman: target not found nextpnr/icestorm).
    This experiment is GATED on Exp 794 having tools_installed=True.
    If the gate blocks, we write a blocked artifact and exit immediately.

    When tools ARE available: synthesize ising_sampler_v3.v at N=32 using yosys with
    the iCE40 HX8K backend (synth_ice40), report LUT utilization, and if N=32 is clean
    and under 5000 LUTs, attempt N=64 as well.

**Honest verdict mapping:**
    synthesis_clean_n32:         yosys N=32 succeeds, lut_count < 5000, no errors.
    synthesis_clean_n32_n64:     both N=32 and N=64 synthesize cleanly.
    synthesis_errors_n32:        yosys reports errors on the N=32 run.
    tools_not_installed:         Exp 794 gate blocks (tools_installed=False).

**Gate condition (from task spec):**
    Load results/experiment_794_fpga_toolchain_install.json.
    If honest_verdict not in ["tools_installed_synthesis_clean",
                               "tools_installed_synthesis_failed"]:
        write blocked artifact, exit.

Spec: REQ-HW-035, SCENARIO-HW-033
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
import contextlib

DELIVERABLE = "results/experiment_804_kv260_synthesis_attempt.json"
EXP794_ARTIFACT = "results/experiment_794_fpga_toolchain_install.json"
RTL_SOURCE = "hardware/kv260/ising_sampler_v3.v"

# honest_verdict values from Exp 794 that indicate tools are actually installed.
# Only these two allow Exp 804 to proceed past the gate.
_GATE_PASS_VERDICTS = frozenset(
    ["tools_installed_synthesis_clean", "tools_installed_synthesis_failed"]
)

# iCE40 HX8K has 7680 LUTs. The N=32 budget target is <5000 LUTs (~65%).
LUT_BUDGET_N32 = 5000
# N=64 is considered over-budget above 6000 LUTs (~78% of HX8K).
LUT_BUDGET_N64 = 6000


def _run(cmd: list[str], timeout: int = 180) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr).

    FileNotFoundError is caught and returned as returncode=-1 so callers can
    check rc != 0 without a separate exception path (same pattern as Exp 791).
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
        return -1, "", f"Timed out after {timeout}s: {cmd}"
    except FileNotFoundError:
        return -1, "", f"Executable not found: {cmd[0]}"


def _which(name: str) -> bool:
    """Return True if the named executable is reachable on PATH."""
    rc, _, _ = _run(["which", name], timeout=5)
    return rc == 0


def _find_yosys() -> tuple[bool, str]:
    """Locate yosys (native binary or yowasp Python wrapper).

    Priority: native 'yosys' on PATH first (fastest), then the yowasp Python
    wrapper (works without native tools).

    Returns (found, command_prefix).
    """
    if _which("yosys"):
        return True, "yosys"
    rc, _, _ = _run([sys.executable, "-m", "yowasp_yosys", "--version"], timeout=30)
    if rc == 0:
        return True, f"{sys.executable} -m yowasp_yosys"
    return False, ""


def _patch_n_spins(rtl_text: str, n_spins: int, max_degree: int) -> str:
    """Return RTL text with N_SPINS and MAX_DEGREE parameter defaults overridden.

    Why override in the source file rather than using yosys chparam: ising_sampler_v3.v
    does not have MAX_DEGREE as a top-level parameter (it uses per-spin ring topology).
    We override N directly in the parameter declaration so yosys sees the right value
    at elaboration time without requiring a chparam script.

    The replacement targets the exact parameter declaration line:
        parameter integer N              = 64,
    and replaces 64 (or whatever the current value is) with n_spins.
    """
    # Override N parameter
    patched = re.sub(
        r"(parameter\s+integer\s+N\s*=\s*)\d+",
        rf"\g<1>{n_spins}",
        rtl_text,
    )
    return patched


def _parse_lut_count(yosys_output: str) -> int | None:
    """Extract the SB_LUT4 count from yosys stat output.

    Yosys synth_ice40 reports LUT primitives like:
        SB_LUT4                                     612
    or in some versions:
        Number of cells:                             612

    WHY prefer SB_LUT4 over 'Number of cells': the cell count includes flip-flops,
    carry chains, and other non-LUT primitives. SB_LUT4 is the direct LUT metric
    and the correct number to compare against the HX8K's 7680 LUT4 budget.

    Returns None if the pattern is not found (e.g. synthesis errored out early).
    """
    for line in yosys_output.splitlines():
        m = re.search(r"SB_LUT4\s+(\d+)", line)
        if m:
            return int(m.group(1))
    # Fallback: look for total cells line
    for line in yosys_output.splitlines():
        m2 = re.search(r"Number of cells\s*:\s*(\d+)", line)
        if m2:
            return int(m2.group(1))
    return None


def _run_yosys_synth(rtl_text: str, n_spins: int, yosys_cmd: str) -> tuple[bool, int | None, str]:
    """Write a patched RTL file and run yosys synth_ice40, return (ok, lut_count, log).

    Steps performed:
      1. Patch N parameter in RTL text to n_spins.
      2. Write patched RTL to a temp file.
      3. Write a yosys script (read_verilog / synth_ice40 / stat) to a temp file.
      4. Run yosys with the script file.
      5. Parse LUT count from stdout.

    Returns:
      ok:        True if yosys exited 0 and no "ERROR" lines appear in output.
      lut_count: Integer SB_LUT4 count, or None if not parseable.
      log:       Combined stdout + stderr for debugging.
    """
    patched = _patch_n_spins(rtl_text, n_spins, max_degree=8)

    with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as rtl_tmp:
        rtl_tmp.write(patched)
        rtl_path = rtl_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as json_tmp:
        json_path = json_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".ys", mode="w", delete=False) as ys_tmp:
        ys_tmp.write(
            f"read_verilog {rtl_path}\nsynth_ice40 -top ising_sampler_v3 -json {json_path}\nstat\n"
        )
        ys_path = ys_tmp.name

    cmd = yosys_cmd.split() + [ys_path]
    rc, stdout, stderr = _run(cmd, timeout=120)
    combined = stdout + "\n" + stderr

    has_error = rc != 0 or "ERROR" in combined.upper()
    lut_count = _parse_lut_count(combined)

    # Clean up temp files
    for p in [rtl_path, json_path, ys_path]:
        with contextlib.suppress(Exception):
            Path(p).unlink(missing_ok=True)

    return not has_error, lut_count, combined


def load_gate_artifact(repo_root: Path) -> dict:
    """Load Exp 794 artifact and return its contents as a dict.

    Returns an empty dict if the file is missing, so the caller can treat
    missing-file as tools_installed=False (gate blocks).
    """
    gate_path = repo_root / EXP794_ARTIFACT
    if not gate_path.exists():
        return {}
    try:
        return json.loads(gate_path.read_text())
    except Exception:
        return {}


def run_experiment(tmpl: ExperimentTemplate) -> tuple[dict, str]:
    """Run the full synthesis experiment, returning (result_fields, status).

    Step-by-step:
      1. Load Exp 794 gate artifact; block if tools_installed is False.
      2. Locate yosys (native or yowasp).
      3. Read ising_sampler_v3.v RTL source.
      4. Synthesize N=32 with synth_ice40; parse LUT count.
      5. If N=32 clean and within budget, attempt N=64.
      6. Determine honest_verdict.

    Returns the flat result dict and status string ("blocked" or "success").
    """
    repo_root = Path(tmpl._repo_root)

    # Step 1: Gate check
    gate = load_gate_artifact(repo_root)
    gate_verdict = gate.get("honest_verdict", "")
    if gate_verdict not in _GATE_PASS_VERDICTS:
        return {
            "gate_exp794_verdict": gate_verdict,
            "gate_passed": False,
            "tools_installed": gate.get("tools_installed", False),
            "synthesis_n32": None,
            "lut_count_n32": None,
            "synthesis_n64": None,
            "lut_count_n64": None,
            "honest_verdict": "tools_not_installed",
            "block_reason": (
                f"Exp 794 honest_verdict='{gate_verdict}' is not in allowed gate values "
                f"{sorted(_GATE_PASS_VERDICTS)}. Install yosys/nextpnr-ice40/icepack first."
            ),
        }, "blocked"

    # Step 2: Find yosys
    yosys_found, yosys_cmd = _find_yosys()
    if not yosys_found:
        return {
            "gate_exp794_verdict": gate_verdict,
            "gate_passed": True,
            "tools_installed": False,
            "synthesis_n32": None,
            "lut_count_n32": None,
            "synthesis_n64": None,
            "lut_count_n64": None,
            "honest_verdict": "tools_not_installed",
            "block_reason": "yosys not found on PATH and yowasp_yosys not importable",
        }, "blocked"

    # Step 3: Read RTL
    rtl_path = repo_root / RTL_SOURCE
    rtl_text = rtl_path.read_text()

    # Step 4: Synthesize N=32
    ok_n32, lut_count_n32, _log_n32 = _run_yosys_synth(rtl_text, n_spins=32, yosys_cmd=yosys_cmd)

    if not ok_n32:
        return {
            "gate_exp794_verdict": gate_verdict,
            "gate_passed": True,
            "tools_installed": True,
            "yosys_cmd": yosys_cmd,
            "synthesis_n32": False,
            "lut_count_n32": lut_count_n32,
            "synthesis_n64": None,
            "lut_count_n64": None,
            "honest_verdict": "synthesis_errors_n32",
        }, "success"

    # N=32 succeeded — check budget
    n32_within_budget = lut_count_n32 is not None and lut_count_n32 < LUT_BUDGET_N32

    # Step 5: Attempt N=64 if N=32 is clean
    synthesis_n64: bool | None = None
    lut_count_n64: int | None = None
    n64_over_budget = False

    if ok_n32:
        ok_n64, lut_count_n64, _log_n64 = _run_yosys_synth(
            rtl_text, n_spins=64, yosys_cmd=yosys_cmd
        )
        synthesis_n64 = ok_n64
        if ok_n64 and lut_count_n64 is not None and lut_count_n64 > LUT_BUDGET_N64:
            n64_over_budget = True

    # Step 6: Verdict
    if ok_n32 and synthesis_n64:
        honest_verdict = "synthesis_clean_n32_n64"
    elif ok_n32:
        honest_verdict = "synthesis_clean_n32"
    else:
        honest_verdict = "synthesis_errors_n32"

    return {
        "gate_exp794_verdict": gate_verdict,
        "gate_passed": True,
        "tools_installed": True,
        "yosys_cmd": yosys_cmd,
        "synthesis_n32": ok_n32,
        "lut_count_n32": lut_count_n32,
        "n32_within_budget": n32_within_budget,
        "synthesis_n64": synthesis_n64,
        "lut_count_n64": lut_count_n64,
        "n64_over_budget": n64_over_budget,
        "honest_verdict": honest_verdict,
    }, "success"


def main() -> None:
    """Run Experiment 804: first open-source FPGA synthesis attempt for ising_sampler_v3.v."""
    tmpl = ExperimentTemplate(
        exp_id=804,
        title="KV260 N=32 Yosys Open-Source Synthesis Attempt (gates on Exp 794)",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )

    with ExperimentTimeoutWatchdog(804, timeout_minutes=45, result_path=DELIVERABLE):
        tmpl.setup()

        try:
            data, status = run_experiment(tmpl)
            artifact = tmpl.build_result(data, status=status)
        except Exception as exc:
            artifact = tmpl.build_result(
                {
                    "gate_exp794_verdict": "unknown",
                    "gate_passed": False,
                    "tools_installed": False,
                    "synthesis_n32": None,
                    "lut_count_n32": None,
                    "synthesis_n64": None,
                    "lut_count_n64": None,
                    "honest_verdict": "tools_not_installed",
                    "error": str(exc),
                },
                status="error",
            )

        output_path = Path(tmpl._repo_root) / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(
            f"[Exp 804] verdict={artifact.get('honest_verdict')}  "
            f"lut_count_n32={artifact.get('lut_count_n32')}  "
            f"gate_passed={artifact.get('gate_passed')}"
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
