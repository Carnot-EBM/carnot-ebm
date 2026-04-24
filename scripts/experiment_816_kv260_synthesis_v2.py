#!/usr/bin/env python3
"""Experiment 816 — KV260 Ising Sampler v3 Synthesis via OSS-CAD-Suite Yosys.

**Research question:**
    Exps 791/794/804 all failed at tool-check because system yosys was absent or
    broken.  Exp 807 installed OSS-CAD-Suite to ~/tools/oss-cad-suite/.  This
    experiment is GATED on Exp 807 having honest_verdict in
    ['tools_installed_synthesis_clean', 'already_installed'].

    When tools ARE available: synthesize ising_sampler_v3.v at N=32 using the
    OSS-CAD-Suite yosys (NOT system yosys) with the iCE40 HX8K backend
    (synth_ice40).  Report SB_LUT4 utilization.  If N=32 is clean and under
    3000 LUTs, attempt N=64 as well.

**Why OSS-CAD-Suite yosys specifically:**
    System yosys (from pacman) was consistently absent or out of date on the
    build host across three prior attempts.  OSS-CAD-Suite ships a self-contained
    yosys binary at ~/tools/oss-cad-suite/bin/yosys with all back-ends compiled in,
    independent of the host package manager.  REQ-HW-038 mandates using this path
    to avoid PATH-order surprises.

**Honest verdict mapping:**
    synthesis_clean_n32_n64:    N=32 and N=64 both synthesize cleanly, n64 < 6000 LUT.
    synthesis_clean_n32:        N=32 clean, lut_count_n32 < 5000.
    synthesis_n32_over_budget:  N=32 yosys exits 0 but lut_count_n32 >= 5000.
    synthesis_errors_n32:       yosys reports ERROR or exits non-zero on N=32.
    blocked_tools_not_at_expected_path:  ~/tools/oss-cad-suite/bin/yosys absent.
    gated_tools_not_installed:  Exp 807 gate fails (tools not confirmed installed).

Spec: REQ-HW-038, SCENARIO-HW-035
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
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

DELIVERABLE = "results/experiment_816_kv260_synthesis_v2.json"
EXP807_ARTIFACT = "results/experiment_807_oss_cad_suite_install.json"
RTL_SOURCE = "hardware/kv260/ising_sampler_v3.v"

# OSS-CAD-Suite installs to ~/tools/oss-cad-suite/bin by convention (Exp 807).
OSS_CAD_BIN = Path.home() / "tools" / "oss-cad-suite" / "bin"

# Exp 807 honest_verdict values that confirm the toolchain is installed.
_GATE_PASS_VERDICTS = frozenset(["tools_installed_synthesis_clean", "already_installed"])

# iCE40 HX8K has 7680 LUTs.
# N=32 must be < 5000 to count as within budget (REQ-HW-038).
LUT_BUDGET_N32 = 5000
# Only attempt N=64 expansion when N=32 is comfortably small (< 3000 LUTs).
LUT_EXPAND_THRESHOLD = 3000
# N=64 is over budget above 6000 LUTs.
LUT_BUDGET_N64 = 6000


def _run(cmd: list[str], timeout: int = 180) -> tuple[int, str, str]:
    """Run a subprocess; return (returncode, stdout, stderr).

    FileNotFoundError and TimeoutExpired are caught and returned as rc=-1 so
    callers can branch on rc != 0 without a separate exception path.
    This mirrors the pattern established in Exp 791 and used in Exp 804.
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


def _parse_lut_count(yosys_output: str) -> int | None:
    """Extract the SB_LUT4 count from yosys stat output.

    Yosys synth_ice40 reports LUT primitives as:
        SB_LUT4                                     612

    WHY prefer SB_LUT4 over 'Number of cells': the cell count includes flip-flops,
    carry chains, and other non-LUT primitives.  SB_LUT4 is the direct iCE40 LUT
    metric and the correct number to compare against the HX8K's 7680 LUT4 budget.

    Falls back to 'Number of cells' if SB_LUT4 is absent (some yosys versions omit
    it when targeting generic backends or when elaboration is partial).

    Returns None when neither pattern is found (e.g. synthesis aborted early).

    Spec: REQ-HW-038
    """
    for line in yosys_output.splitlines():
        # Format A (older yosys): "     SB_LUT4                       612"
        m = re.search(r"SB_LUT4\s+(\d+)", line)
        if m:
            return int(m.group(1))
        # Format B (yosys 0.64+): "     3952   SB_LUT4"  (count before name)
        m2 = re.search(r"(\d+)\s+SB_LUT4", line)
        if m2:
            return int(m2.group(1))
    for line in yosys_output.splitlines():
        m3 = re.search(r"Number of cells\s*:\s*(\d+)", line)
        if m3:
            return int(m3.group(1))
    return None


def _patch_n_spins(rtl_text: str, n_spins: int) -> str:
    """Return RTL text with the N parameter default overridden to n_spins.

    We patch the source text rather than using yosys chparam because MAX_DEGREE is
    not a top-level parameter in v3 (ring topology is hardwired).  Patching the
    parameter declaration directly is the simplest approach that works across yosys
    versions without requiring an elaboration pass first.

    Targets the exact declaration pattern used in ising_sampler_v3.v:
        parameter integer N              = 64,

    Spec: REQ-HW-038
    """
    return re.sub(
        r"(parameter\s+integer\s+N\s*=\s*)\d+",
        rf"\g<1>{n_spins}",
        rtl_text,
    )


def _run_yosys_synth(
    rtl_text: str,
    n_spins: int,
    yosys_bin: Path,
    timeout: int = 180,
) -> tuple[bool, int | None, str]:
    """Write a patched RTL file and run OSS-CAD-Suite yosys synth_ice40.

    Returns:
        ok:        True if yosys exited 0 and no ERROR lines appear.
        lut_count: SB_LUT4 count integer, or None if not parseable.
        log:       Combined stdout + stderr for debugging / artifact storage.

    The JSON netlist output path is passed to yosys even though we do not use it
    downstream — omitting it causes some yosys versions to error on the stat pass.

    Spec: REQ-HW-038
    """
    patched = _patch_n_spins(rtl_text, n_spins)

    with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as rtl_tmp:
        rtl_tmp.write(patched)
        rtl_path = rtl_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_tmp:
        json_path = json_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".ys", mode="w", delete=False) as ys_tmp:
        ys_tmp.write(
            f"read_verilog {rtl_path}\n"
            f"synth_ice40 -top ising_sampler_v3 -json {json_path}\n"
            "stat\n"
        )
        ys_path = ys_tmp.name

    cmd = [str(yosys_bin), ys_path]
    rc, stdout, stderr = _run(cmd, timeout=timeout)
    combined = stdout + "\n" + stderr

    has_error = rc != 0 or "ERROR" in combined.upper()
    lut_count = _parse_lut_count(combined)

    for p in [rtl_path, json_path, ys_path]:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    return not has_error, lut_count, combined


def load_gate_artifact(repo_root: Path) -> dict:
    """Load Exp 807 artifact and return its contents as a dict.

    Returns an empty dict if the file is missing or corrupt, so the caller
    treats missing-file as tools_not_installed (gate blocks conservatively).
    """
    gate_path = repo_root / EXP807_ARTIFACT
    if not gate_path.exists():
        return {}
    try:
        return json.loads(gate_path.read_text())
    except Exception:
        return {}


def run_experiment(tmpl: ExperimentTemplate) -> tuple[dict, str]:
    """Run the full synthesis experiment.

    Returns (result_fields, status) where status is 'blocked' or 'success'.

    Steps:
      1. Load Exp 807 gate artifact; block if verdict not in allowed set.
      2. Verify OSS-CAD-Suite yosys exists at the expected path.
      3. Read ising_sampler_v3.v RTL source.
      4. Synthesize N=32 with synth_ice40; parse LUT count.
      5. If N=32 clean and lut_count < 3000, attempt N=64.
      6. Determine honest_verdict.

    Spec: REQ-HW-038, SCENARIO-HW-035
    """
    repo_root = Path(tmpl._repo_root)

    # Step 1: Gate check — require Exp 807 to confirm OSS-CAD-Suite is installed.
    gate = load_gate_artifact(repo_root)
    gate_verdict = gate.get("honest_verdict", "")
    if gate_verdict not in _GATE_PASS_VERDICTS:
        return {
            "gate_exp807_verdict": gate_verdict,
            "gate_passed": False,
            "honest_verdict": "gated_tools_not_installed",
            "block_reason": (
                f"Exp 807 honest_verdict='{gate_verdict}' not in allowed values "
                f"{sorted(_GATE_PASS_VERDICTS)}.  Run Exp 807 first."
            ),
            "lut_count_n32": None,
            "lut_count_n64": None,
            "synthesis_n32_ok": None,
            "synthesis_n64_ok": None,
            "yosys_log_n32": None,
            "yosys_log_n64": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    # Step 2: Verify OSS-CAD-Suite yosys at expected path.
    # WHY explicit path: REQ-HW-038 mandates ~/tools/oss-cad-suite/bin/yosys to
    # avoid accidentally using a broken system yosys from PATH (root cause of
    # Exps 791/794/804 failures).
    yosys_bin = OSS_CAD_BIN / "yosys"
    if not yosys_bin.exists():
        return {
            "gate_exp807_verdict": gate_verdict,
            "gate_passed": True,
            "honest_verdict": "blocked_tools_not_at_expected_path",
            "block_reason": f"OSS-CAD-Suite yosys not found at {yosys_bin}",
            "lut_count_n32": None,
            "lut_count_n64": None,
            "synthesis_n32_ok": None,
            "synthesis_n64_ok": None,
            "yosys_log_n32": None,
            "yosys_log_n64": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    # Step 3: Read RTL source.
    rtl_path = repo_root / RTL_SOURCE
    if not rtl_path.exists():
        return {
            "gate_exp807_verdict": gate_verdict,
            "gate_passed": True,
            "honest_verdict": "blocked_rtl_missing",
            "block_reason": f"RTL source not found: {rtl_path}",
            "lut_count_n32": None,
            "lut_count_n64": None,
            "synthesis_n32_ok": None,
            "synthesis_n64_ok": None,
            "yosys_log_n32": None,
            "yosys_log_n64": None,
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "blocked"

    rtl_text = rtl_path.read_text()

    # Step 4: Synthesize N=32.
    n32_ok, lut_n32, log_n32 = _run_yosys_synth(rtl_text, 32, yosys_bin, timeout=180)

    if not n32_ok or lut_n32 is None:
        return {
            "gate_exp807_verdict": gate_verdict,
            "gate_passed": True,
            "synthesis_n32_ok": n32_ok,
            "lut_count_n32": lut_n32,
            "synthesis_n64_ok": None,
            "lut_count_n64": None,
            "yosys_log_n32": log_n32[-3000:],
            "yosys_log_n64": None,
            "honest_verdict": "synthesis_errors_n32",
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "success"

    if lut_n32 >= LUT_BUDGET_N32:
        return {
            "gate_exp807_verdict": gate_verdict,
            "gate_passed": True,
            "synthesis_n32_ok": n32_ok,
            "lut_count_n32": lut_n32,
            "synthesis_n64_ok": None,
            "lut_count_n64": None,
            "yosys_log_n32": log_n32[-3000:],
            "yosys_log_n64": None,
            "honest_verdict": "synthesis_n32_over_budget",
            "oss_cad_bin": str(OSS_CAD_BIN),
        }, "success"

    # Step 5: Attempt N=64 if N=32 is comfortably small.
    n64_ok: bool | None = None
    lut_n64: int | None = None
    log_n64: str | None = None

    if lut_n32 < LUT_EXPAND_THRESHOLD:
        n64_ok, lut_n64, log_n64 = _run_yosys_synth(rtl_text, 64, yosys_bin, timeout=300)

    # Step 6: Determine honest_verdict.
    if n64_ok is True and lut_n64 is not None and lut_n64 < LUT_BUDGET_N64:
        honest_verdict = "synthesis_clean_n32_n64"
    else:
        honest_verdict = "synthesis_clean_n32"

    if n64_ok is True and lut_n64 is not None and lut_n64 >= LUT_BUDGET_N64:
        n64_note = "over_budget_n64"
    else:
        n64_note = None

    return {
        "gate_exp807_verdict": gate_verdict,
        "gate_passed": True,
        "synthesis_n32_ok": n32_ok,
        "lut_count_n32": lut_n32,
        "synthesis_n64_ok": n64_ok,
        "lut_count_n64": lut_n64,
        "n64_note": n64_note,
        "yosys_log_n32": log_n32[-3000:],
        "yosys_log_n64": (log_n64[-3000:] if log_n64 else None),
        "honest_verdict": honest_verdict,
        "oss_cad_bin": str(OSS_CAD_BIN),
    }, "success"


def main() -> None:
    """Entry point for Experiment 816."""
    # apply_env_autofix MUST be called before any JAX or CUDA import.
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        exp_id=816,
        title="KV260 Ising Sampler v3 Synthesis via OSS-CAD-Suite Yosys",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(816, timeout_minutes=60, result_path=str(tmpl._output_path)):
        data, status = run_experiment(tmpl)

    artifact = tmpl.build_result(data, status=status)
    out_path = tmpl._repo_root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 816] honest_verdict={artifact.get('honest_verdict')}  "
          f"lut_count_n32={artifact.get('lut_count_n32')}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
