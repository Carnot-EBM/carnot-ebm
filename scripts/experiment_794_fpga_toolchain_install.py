#!/usr/bin/env python3
"""Experiment 794: Install FPGA open-source toolchain and run minimal synthesis proof-of-concept.

Context:
    Exp 791 (KV260 N=32 reduced synthesis) failed because yosys, nextpnr-ice40,
    and icepack were all absent from PATH (retro tag: RETRO-KV260-TOOLS-UNAVAILABLE).
    This experiment installs those tools, verifies their presence, and if all three
    are available runs a minimal 2-spin Ising synthesis as proof-of-concept.
    A passing result here gates Exp 804 (full KV260 synthesis attempt).

Protocol:
    1. apply_env_autofix() FIRST — no JAX imports happen before this.
    2. ExperimentTimeoutWatchdog(794, timeout_minutes=30) hard cap.
    3. Check tool availability: yosys, nextpnr-ice40, icepack.
    4. If any tool is missing: attempt pacman install, re-check.
    5. If all tools present: synthesize 2-spin Ising test module with synth_ice40.
    6. Record honest_verdict and write deliverable JSON.

Spec: REQ-HW-032, REQ-HW-033, REQ-HW-034, SCENARIO-HW-032
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# apply_env_autofix MUST precede any JAX or CUDA import — it patches env vars
# that JAX reads at import time (JAX_PLATFORMS, XLA_FLAGS, etc.).
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DELIVERABLE = str(_REPO_ROOT / "results" / "experiment_794_fpga_toolchain_install.json")

# ---------------------------------------------------------------------------
# Minimal 2-spin Ising test Verilog — synthesized as proof-of-concept.
# This module is intentionally tiny: one register, one conditional, no memory.
# It maps cleanly to 1-2 LUTs on any FPGA target.
# ---------------------------------------------------------------------------

_ISING2_VERILOG = """\
module test_ising2(
    input clk,
    input signed [7:0] J,
    output reg [1:0] spin
);
    // Metropolis accept step: if coupling J is negative (bit 7 set), anti-align spins.
    // This encodes the simplest possible 2-spin Ising interaction.
    always @(posedge clk)
        spin <= (J[7]) ? 2'b10 : 2'b01;
endmodule
"""

# ---------------------------------------------------------------------------
# Tool-check helpers
# ---------------------------------------------------------------------------


def _check_tool(cmd: list[str], timeout: int = 10) -> tuple[bool, str]:
    """Run a version-check command and return (present, version_string).

    Why capture_output and not just check=False: we want the stdout/stderr
    text for the artifact even when the tool exits non-zero (icepack --help
    exits 1 but still signals the tool is present).
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        output = (r.stdout + r.stderr).strip()
        # Any output at all (even from exit-1) means the binary is present.
        present = bool(output) or r.returncode == 0
        return present, output[:200]
    except FileNotFoundError:
        return False, "not found"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def check_tools() -> dict[str, dict]:
    """Check availability of the three required open-source FPGA tools.

    Returns a dict keyed by tool name with keys: present (bool), version (str).
    The three tools must all be present for synthesis to be attempted (REQ-HW-032,
    REQ-HW-033, REQ-HW-034).
    """
    tools = {
        "yosys": _check_tool(["yosys", "--version"]),
        "nextpnr-ice40": _check_tool(["nextpnr-ice40", "--version"]),
        "icepack": _check_tool(["icepack", "--help"]),
    }
    return {
        name: {"present": present, "version": ver}
        for name, (present, ver) in tools.items()
    }


def attempt_pacman_install(timeout: int = 300) -> tuple[bool, str]:
    """Attempt to install yosys, nextpnr, and icestorm via pacman.

    Why pacman: this experiment targets CachyOS (Arch-based).  pacman is the
    canonical package manager; apt/dnf are absent.  The packages needed are:
      - yosys        (provides the yosys binary)
      - nextpnr      (provides nextpnr-ice40)
      - icestorm     (provides icepack and iceprog)

    Returns (success, output) where success is True only if pacman exits 0.
    """
    try:
        r = subprocess.run(
            ["sudo", "pacman", "-S", "--noconfirm", "yosys", "nextpnr", "icestorm"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode == 0, (r.stdout + r.stderr).strip()[:2000]
    except FileNotFoundError:
        return False, "sudo not found — cannot install"
    except subprocess.TimeoutExpired:
        return False, "pacman install timed out"


def run_synthesis(verilog_src: str) -> dict:
    """Write verilog to a temp file and synthesize with yosys synth_ice40.

    Why synth_ice40: the full synthesis target matches what Exp 804 will use for
    the real KV260 flow; validating it here ensures the tool flags (primitives,
    mapping) are correct before we attempt the 32-spin Ising circuit.

    Returns a dict with keys: success (bool), lut_count (int|None),
    stderr_snippet (str), stdout_snippet (str).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = os.path.join(tmpdir, "test_ising2.v")
        json_path = os.path.join(tmpdir, "test_ising2.json")
        with open(v_path, "w") as fh:
            fh.write(verilog_src)

        try:
            r = subprocess.run(
                [
                    "yosys",
                    "-p",
                    f"synth_ice40; write_json {json_path}",
                    v_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "lut_count": None,
                "stderr_snippet": "yosys timed out",
                "stdout_snippet": "",
            }

        stdout = r.stdout[:3000]
        stderr = r.stderr[:3000]
        success = r.returncode == 0 and "ERROR" not in stderr.upper()

        lut_count: int | None = None
        if success and os.path.exists(json_path):
            try:
                with open(json_path) as jf:
                    netlist = json.load(jf)
                # Count SB_LUT4 cells across all modules — the iCE40 LUT primitive.
                total = 0
                for mod in netlist.get("modules", {}).values():
                    cells = mod.get("cells", {})
                    total += sum(
                        1 for c in cells.values() if c.get("type") == "SB_LUT4"
                    )
                lut_count = total
            except Exception:
                pass  # lut_count stays None; success remains True if yosys exited 0

        return {
            "success": success,
            "lut_count": lut_count,
            "stderr_snippet": stderr[:500],
            "stdout_snippet": stdout[:500],
        }


def classify_verdict(
    tools: dict[str, dict],
    install_attempted: bool,
    install_success: bool,
    synth: dict | None,
) -> str:
    """Map observed tool/synthesis state to a canonical honest_verdict string.

    The five verdicts mirror REQ-HW-043-4 (the prior synthesis REQ) so that
    the conductor's reconciliation logic can use a consistent vocabulary:

      tools_installed_synthesis_clean     — all 3 tools present + synth no errors
      tools_installed_synthesis_failed    — all 3 tools present but synth errors
      tools_installed_synthesis_skipped   — all 3 tools present, synth not attempted
      tools_not_installed_install_attempted — install ran but tools still missing
      tools_not_installed_install_skipped   — sudo not available, install not tried
    """
    all_present = all(t["present"] for t in tools.values())
    if all_present:
        if synth is None:
            return "tools_installed_synthesis_skipped"
        if synth["success"]:
            return "tools_installed_synthesis_clean"
        return "tools_installed_synthesis_failed"
    if install_attempted:
        return "tools_not_installed_install_attempted"
    return "tools_not_installed_install_skipped"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        794,
        "FPGA Toolchain Install: yosys + nextpnr-ice40 + icepack (gates Exp 804)",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(794, timeout_minutes=30, result_path=_DELIVERABLE):
        t0 = time.time()

        # Step 1: Check initial tool availability.
        _log.info("Checking initial tool availability …")
        tools_before = check_tools()
        for name, info in tools_before.items():
            _log.info("  %s: present=%s  version=%s", name, info["present"], info["version"])

        all_present_before = all(t["present"] for t in tools_before.values())

        install_attempted = False
        install_success = False
        install_output = ""
        tools_after = tools_before

        if not all_present_before:
            # Step 2: Attempt installation via pacman.
            _log.info("One or more tools missing — attempting pacman install …")
            install_attempted = True
            install_success, install_output = attempt_pacman_install()
            _log.info("pacman install returned success=%s", install_success)

            # Re-check after install.
            tools_after = check_tools()
            for name, info in tools_after.items():
                _log.info(
                    "  (post-install) %s: present=%s  version=%s",
                    name,
                    info["present"],
                    info["version"],
                )

        all_present_after = all(t["present"] for t in tools_after.values())

        # Step 3: If all tools present, run minimal synthesis.
        synth_result: dict | None = None
        if all_present_after:
            _log.info("All tools present — running minimal 2-spin Ising synthesis …")
            synth_result = run_synthesis(_ISING2_VERILOG)
            _log.info(
                "Synthesis: success=%s  lut_count=%s",
                synth_result["success"],
                synth_result["lut_count"],
            )

        verdict = classify_verdict(tools_after, install_attempted, install_success, synth_result)
        _log.info("honest_verdict: %s", verdict)

        artifact_payload: dict = {
            "tools_installed": all_present_after,
            "tools_before_install": tools_before,
            "install_attempted": install_attempted,
            "install_success": install_success,
            "install_output_snippet": install_output[:1000],
            "tools_after_install": tools_after,
            "synthesis": synth_result,
            "honest_verdict": verdict,
            "gates": "Exp 804 (KV260 N=32 full synthesis)",
        }

        status = "success" if all_present_after else "blocked"
        artifact = tmpl.build_result(artifact_payload, status=status)
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))
        _log.info("Artifact written: %s", _DELIVERABLE)
        _log.info("Duration: %.1f s", time.time() - t0)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
