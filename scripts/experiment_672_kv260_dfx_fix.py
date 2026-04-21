#!/usr/bin/env python3
"""Experiment 672 — KV260 dfx-mgr Alternative Bitfile Loading.

**Goal:**
    Exp 661 was blocked with ``honest_verdict = blocked_on_dfx_mgr_load_failure``.
    The failure was a software protocol issue: ``dfx-mgr-client -load`` timed out
    because the remote dfx-mgr daemon at 192.168.51.98 was not responding.

    This experiment systematically tries four alternative bitfile loading methods
    in order, records which (if any) succeeds, and if all fail it diagnoses the
    root cause so a human operator can fix it.

**Methods attempted (in order):**
    1. dfx-mgr-client -load  — the standard Kria DFX manager (what Exp 661 tried)
    2. fpgautil -b           — alternative DFX utility shipped with some Xilinx kernels
    3. dd if=<bit> of=/dev/xdevcfg — direct FPGA configuration device (Zynq legacy)
    4. sysfs firmware loading — copy to /lib/firmware/ and trigger via sysfs node

**Environment variables (required to run against real hardware):**
    CARNOT_KV260_BITFILE : absolute path to the .bit or .bit.bin bitfile.
                           If unset or the file does not exist, the experiment
                           writes a ``blocked`` artifact with instructions and exits 0.

**Honest-verdict enum:**
    - dfx_method_found_<method>       : one method loaded the bitfile successfully
    - dfx_protocol_diagnosed_<diag>   : all methods failed; diag encodes root cause
    - blocked_bitfile_not_configured  : CARNOT_KV260_BITFILE not set or file missing

Spec: REQ-VERIFY-083, REQ-INFRA-007
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 672
TITLE = "KV260 dfx-mgr Alternative Bitfile Loading"
DELIVERABLE = "results/experiment_672_kv260_dfx_fix.json"
SUBPROCESS_TIMEOUT_S = 30


# ---------------------------------------------------------------------------
# CPU baseline helper
# ---------------------------------------------------------------------------


def _run_cpu_baseline(n_spins: int = 64, n_trials: int = 100) -> dict:
    """Run a lightweight Python Ising sampler CPU baseline.

    Why we do this even when targeting FPGA: the CPU baseline establishes the
    reference latency so that once hardware is running (Exp 673 follow-up) we
    can compute the FPGA speedup without re-running the CPU side.

    Returns a dict with mean/std latency in microseconds and trial count.
    """
    import random

    spins = [1 if random.random() > 0.5 else -1 for _ in range(n_spins)]
    latencies_us = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        for i in range(n_spins):
            # Single Gibbs step: compute local field and flip spin probabilistically.
            # Using uniform random couplings J=0 (no coupling) for the baseline timing
            # test — we only care about the per-spin iteration cost here, not sampling
            # correctness, which is covered by unit tests elsewhere.
            h_i = 0.0
            prob_up = 1.0 / (1.0 + (2.0 * h_i * spins[i]))
            spins[i] = 1 if random.random() < prob_up else -1
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    mean_lat = sum(latencies_us) / len(latencies_us)
    std_lat = (sum((x - mean_lat) ** 2 for x in latencies_us) / len(latencies_us)) ** 0.5
    return {
        "n_spins": n_spins,
        "n_trials": n_trials,
        "mean_latency_us": round(mean_lat, 3),
        "std_latency_us": round(std_lat, 3),
    }


# ---------------------------------------------------------------------------
# DFX loading methods
# ---------------------------------------------------------------------------


def _try_method(method_name: str, cmd: list[str], env: dict | None = None) -> dict:
    """Run one loading method and return a structured result dict.

    Captures up to 500 chars of stdout/stderr so the artifact stays compact
    while still containing enough context to diagnose failures without SSHing
    into the board.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_S,
            env=env,
        )
        return {
            "method_name": method_name,
            "exit_code": result.returncode,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500],
            "success": result.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "method_name": method_name,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Timed out after {SUBPROCESS_TIMEOUT_S}s",
            "success": False,
            "timed_out": True,
        }
    except FileNotFoundError:
        # The command itself doesn't exist on this machine (e.g. fpgautil not installed).
        return {
            "method_name": method_name,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Command not found: {cmd[0]}",
            "success": False,
            "timed_out": False,
        }


def _diagnose_failure(methods_tried: list[dict]) -> str:
    """Infer the root cause from the stderr of all failed methods.

    Why a single diagnosis string rather than per-method: the conductor needs
    one clear signal to act on.  Merging all stderr into a single label avoids
    the ambiguity of "method 1 says X, method 2 says Y".

    Possible returned labels:
    - firmware_not_found    : /lib/firmware path issue or missing .dtbo
    - permission_denied     : sudo not configured or /dev/xdevcfg permissions wrong
    - protocol_error        : dfx-mgr daemon unreachable (our Exp 661 failure)
    - command_not_found     : none of the four tools are installed
    - unknown               : stderr did not match any known pattern
    """
    all_stderr = " ".join(r.get("stderr", "") for r in methods_tried).lower()

    if "not found" in all_stderr and all(
        r.get("stderr", "").lower().startswith("command not found") or
        "not found" in r.get("stderr", "").lower()
        for r in methods_tried
        if "command not found" in r.get("stderr", "").lower() or
        r.get("exit_code") == -1
    ):
        # Check if ALL methods failed with "not found"
        all_cmd_missing = all(
            "command not found" in r.get("stderr", "").lower() or
            "not found: " in r.get("stderr", "").lower()
            for r in methods_tried
        )
        if all_cmd_missing:
            return "command_not_found"

    # Protocol errors (daemon unreachable) take priority — this is the Exp 661
    # failure mode and is more actionable than a permission error on a copy step.
    if "timeout" in all_stderr or "not responding" in all_stderr or "connection refused" in all_stderr:
        return "protocol_error"

    if "permission denied" in all_stderr or "operation not permitted" in all_stderr:
        return "permission_denied"

    if "no such file" in all_stderr or "firmware" in all_stderr:
        return "firmware_not_found"

    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Exp 672.

    Checks for CARNOT_KV260_BITFILE, tries four loading methods in order,
    records results, and writes the deliverable JSON.
    """
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
    )
    tmpl.setup()

    repo_root = tmpl._repo_root
    output_path = repo_root / DELIVERABLE

    # ------------------------------------------------------------------
    # Environment check: is the bitfile configured and present?
    # ------------------------------------------------------------------
    bitfile_env = os.environ.get("CARNOT_KV260_BITFILE", "")
    if not bitfile_env:
        artifact = tmpl.build_result(
            {
                "bitfile_path": None,
                "methods_tried": [],
                "method_that_succeeded": None,
                "diagnosis": "CARNOT_KV260_BITFILE not set",
                "honest_verdict": "blocked_bitfile_not_configured",
                "fix_instructions": (
                    "Set CARNOT_KV260_BITFILE=/path/to/carnot_ising_v2_n64.bit.bin "
                    "before running this experiment.  The bitfile was synthesized in "
                    "Exp 624 and lives in hardware/kv260/app/build/carnot_ising_v2_n64/."
                ),
            },
            status="blocked",
        )
        output_path.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    bitfile_path = Path(bitfile_env)
    if not bitfile_path.exists():
        artifact = tmpl.build_result(
            {
                "bitfile_path": str(bitfile_path),
                "methods_tried": [],
                "method_that_succeeded": None,
                "diagnosis": f"Bitfile does not exist: {bitfile_path}",
                "honest_verdict": "blocked_bitfile_not_configured",
                "fix_instructions": (
                    f"File {bitfile_path} does not exist.  Verify the path or "
                    "re-run Exp 624 to regenerate the bitfile."
                ),
            },
            status="blocked",
        )
        output_path.write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    # ------------------------------------------------------------------
    # Try each loading method in order
    # ------------------------------------------------------------------
    methods_tried: list[dict] = []
    method_that_succeeded: str | None = None

    # Method 1: dfx-mgr-client (standard Kria DFX manager)
    # This is the method Exp 661 attempted.  It failed because the dfx-mgr
    # daemon was not responding at 192.168.51.98.  We retry here in case the
    # daemon has since been restarted.
    r1 = _try_method(
        "dfx_mgr_client",
        ["dfx-mgr-client", "-load", "-name", str(bitfile_path)],
    )
    methods_tried.append(r1)
    if r1["success"]:
        method_that_succeeded = "dfx_mgr_client"

    # Method 2: fpgautil (alternative DFX utility from some Xilinx kernel packages)
    # Bypasses the dfx-mgr daemon entirely — loads the bitfile directly via
    # the kernel's fpga_manager subsystem.
    if method_that_succeeded is None:
        r2 = _try_method(
            "fpgautil",
            ["fpgautil", "-b", str(bitfile_path)],
        )
        methods_tried.append(r2)
        if r2["success"]:
            method_that_succeeded = "fpgautil"

    # Method 3: dd to /dev/xdevcfg (legacy Zynq FPGA configuration device)
    # Available on older Xilinx kernels.  On Ubuntu 24.04 + 6.8 this node
    # may not exist, but worth trying — it does not require any daemon.
    if method_that_succeeded is None:
        r3 = _try_method(
            "dd_xdevcfg",
            ["dd", f"if={bitfile_path}", "of=/dev/xdevcfg"],
        )
        methods_tried.append(r3)
        if r3["success"]:
            method_that_succeeded = "dd_xdevcfg"

    # Method 4: sysfs firmware trigger
    # Copy the bitfile to /lib/firmware/ and write the filename to the
    # fpga_manager's firmware sysfs node to trigger a load.  This is the
    # lowest-level method and works on any kernel with CONFIG_FPGA_MGR.
    if method_that_succeeded is None:
        firmware_dest = Path("/lib/firmware") / bitfile_path.name
        sysfs_node = "/sys/class/fpga_manager/fpga0/firmware"
        copy_result = _try_method(
            "sysfs_firmware_copy",
            ["cp", str(bitfile_path), str(firmware_dest)],
        )
        methods_tried.append(copy_result)
        if copy_result["success"]:
            # File is now in /lib/firmware/; trigger load via sysfs.
            trigger_result = _try_method(
                "sysfs_firmware_trigger",
                ["sh", "-c", f"echo '{bitfile_path.name}' > {sysfs_node}"],
            )
            methods_tried.append(trigger_result)
            if trigger_result["success"]:
                method_that_succeeded = "sysfs_firmware"

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    if method_that_succeeded is not None:
        cpu_baseline = _run_cpu_baseline(n_spins=64, n_trials=100)
        honest_verdict = f"dfx_method_found_{method_that_succeeded}"
        artifact = tmpl.build_result(
            {
                "bitfile_path": str(bitfile_path),
                "methods_tried": methods_tried,
                "method_that_succeeded": method_that_succeeded,
                "diagnosis": None,
                "cpu_baseline": cpu_baseline,
                "hardware_latency_placeholder": (
                    "Hardware latency measurement deferred to Exp 673 follow-up "
                    "once AXI register access is verified."
                ),
                "honest_verdict": honest_verdict,
            },
            status="partial",
        )
    else:
        diagnosis = _diagnose_failure(methods_tried)
        fix_map = {
            "protocol_error": (
                "The dfx-mgr daemon is not responding.  On the KV260 run: "
                "sudo systemctl restart dfx-mgr  "
                "and verify the daemon is listening with: "
                "dfx-mgr-client -listPackage"
            ),
            "permission_denied": (
                "sudo permissions are not configured for dfx-mgr-client or /dev/xdevcfg. "
                "Add the following to /etc/sudoers.d/carnot: "
                "<user> ALL=(ALL) NOPASSWD: /usr/bin/dfx-mgr-client, /usr/sbin/fpgautil"
            ),
            "firmware_not_found": (
                "The bitfile bundle is missing from /lib/firmware/xilinx/. "
                "Re-run the bundle install step from Exp 661: "
                "scp -r hardware/kv260/app/build/carnot_ising_v2_n64 kria:/lib/firmware/xilinx/ "
                "then retry: xmutil loadapp carnot_ising_v2_n64"
            ),
            "command_not_found": (
                "None of the four loading tools (dfx-mgr-client, fpgautil, dd, sh) "
                "are available.  This experiment must run on the KV260 itself or "
                "via SSH.  Set CARNOT_KV260_BITFILE to the remote path and run "
                "this script as: ssh kria python3 scripts/experiment_672_kv260_dfx_fix.py"
            ),
            "unknown": (
                "Unrecognised failure pattern.  Inspect the methods_tried[*].stderr "
                "fields in this artifact and consult the KV260 Ubuntu 24.04 bring-up "
                "guide at hardware/kv260/README.md."
            ),
        }
        honest_verdict = f"dfx_protocol_diagnosed_{diagnosis}"
        artifact = tmpl.build_result(
            {
                "bitfile_path": str(bitfile_path),
                "methods_tried": methods_tried,
                "method_that_succeeded": None,
                "diagnosis": diagnosis,
                "fix_instructions": fix_map.get(diagnosis, fix_map["unknown"]),
                "honest_verdict": honest_verdict,
            },
            status="blocked",
        )

    output_path.write_text(json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
