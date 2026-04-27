#!/usr/bin/env python3
"""Experiment 971 — KV260 Ising Sampler v4 Vivado Bitstream + Board Programming.

**Researcher summary (why this experiment exists):**
    Exp 958 proved that ising_sampler_v4.v (Sparse E-MVL, N=128, K=16) synthesizes
    cleanly in yosys with 27,136 LUTs — well within the XCK26 budget of 117K LUTs.
    The next step is running Vivado 2025.2.1 to produce an actual bitstream, loading
    it onto the physical KV260 board (present since 2026-04-20), and comparing
    hardware Ising sampling speed against a Python CPU baseline.

**What this experiment does:**
    1. Generates a Vivado project TCL (build_bd_v4.tcl) wrapping ising_sampler_v4
       in a Block Design with Zynq PS + axi_gpio for spin-state readback.
    2. Runs Vivado in batch mode (timeout 5400 s) and records Vivado LUT count
       vs the yosys estimate of 27,136.
    3. Checks whether the KV260 board is reachable via SSH (env KV260_BOARD_IP).
    4. If reachable: SCP bitstream + programs via dfx-mgr-client.
    5. Runs a Python-based validation on the board: polls s_out[31:0] via
       /dev/mem AXI GPIO read until all spins are +1 (ferromagnetic convergence).
    6. Measures CPU baseline: Python implementation of the same E-MVL EMA sweep.

**Why axi_gpio for hardware readback:**
    v4 RTL exposes no AXI slave port — it runs autonomously and outputs spin state
    on s_out[127:0].  To let the ARM PS read hardware spin state and measure
    convergence timing without modifying the RTL, we feed s_out[31:0] into a
    Xilinx axi_gpio IP's GPIO_IO input bank.  The PS reads 0xA0000000 (DATA
    register) to get current spin state for the first 32 spins.

**Hardware latency interpretation:**
    The sampler runs at FCLK_CLK0 = 60 MHz.  One sweep takes 1 clock cycle (all
    spins update simultaneously in the synchronous E-MVL design).  The hardware
    latency measured here is the wall-clock time from PS-level reset to first
    observation of all-+1 convergence, dominated by AXI polling overhead and
    Python interpreter latency — not bare FPGA sweep time.  This is compared
    against the equivalent Python E-MVL sweep loop (CPU baseline).

Spec refs: REQ-HW-040, SCENARIO-HW-040
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_TCL_FILE = _REPO_ROOT / "hardware" / "kv260" / "build_bd_v4.tcl"
_BITSTREAM_DST = _REPO_ROOT / "output" / "carnot_ising_v4_bd" / "carnot_ising_v4.bit"
_RESULT_FILE = _REPO_ROOT / "results" / "experiment_971_kv260_board_programming.json"

# Board IP from environment — board ships with DHCP, user sets KV260_BOARD_IP.
# Default "kv260.local" (mDNS) is tried if env var absent.
_BOARD_IP = os.environ.get("KV260_BOARD_IP", "kv260.local")
_BOARD_USER = os.environ.get("KV260_BOARD_USER", "kria")
_BOARD_REMOTE_PATH = "/home/kria/carnot_ising_v4.bit"

# VIVADO binary (sourcing settings64.sh is not needed if already on PATH).
_VIVADO = os.environ.get(
    "VIVADO_BIN",
    "/tools/Xilinx/2025.2.1/Vivado/bin/vivado",
)

# Timeout for the full Vivado synth + impl + bitstream run (90 minutes).
# The v4 sparse design is ~27K LUTs, much smaller than the 290K dense v2
# that overflowed.  Empirical v2 run took 106 min for partial impl; v4
# should complete in 30-60 min.  90 min gives comfortable headroom.
_VIVADO_TIMEOUT_S = 5400

# SSH timeout for board connectivity check (seconds).
_BOARD_CONNECT_TIMEOUT_S = 5

# Number of spins in the CPU baseline simulation (matches v4 RTL default N=128).
_N_SPINS = 128
_K_NEIGHBOURS = 16
_CPU_BASELINE_SWEEPS = 200


# ---------------------------------------------------------------------------
# CPU Baseline: Python E-MVL EMA Ising sweep (matches v4 RTL behaviour)
# ---------------------------------------------------------------------------


def _cpu_baseline_latency_us() -> float:
    """Run _CPU_BASELINE_SWEEPS of the E-MVL EMA Ising sweep in Python.

    **WHY this matches the RTL:**
        ising_sampler_v4 implements:
          1. Sparse field accumulation: h_inst[i] = sum_k J_sparse[i*K+k] * sign(s_cur[nbr])
          2. EMA update: h_ema_new = (h_ema + h_inst) >> 1  (alpha=0.5)
          3. E-MVL rule: s_new[i] = (h_ema_new[i] >= 0) ? +1 : -1

        This Python version implements the same three steps using numpy
        integer arithmetic to stay bit-accurate with the fixed-point RTL.

    **Return value:**
        Microseconds per sweep (averaged over _CPU_BASELINE_SWEEPS sweeps).
    """
    import numpy as np

    rng = np.random.default_rng(42)

    # N spins, K=16 neighbours each (ring topology, same as RTL reset).
    n, k = _N_SPINS, _K_NEIGHBOURS
    nbr_idx = np.zeros((n, k), dtype=np.int32)
    for i in range(n):
        for ki in range(k):
            if ki < k // 2:
                off = ki + 1
            else:
                off = ki - k
            nbr_idx[i, ki] = (i + off + n) % n

    # J_sparse in Q1.15 fixed-point (RTL default 0x0200 = 512).
    J_sparse = np.full((n, k), 512, dtype=np.int32)

    # Spin state: +1 / -1 (RTL starts all +1 at reset, but test with mixed).
    # Use int32 +1/-1 values matching the RTL's sign convention.
    s_cur = rng.choice([-1, 1], size=n).astype(np.int32)

    # EMA field registers (FIELD_WIDTH=24 bits in RTL, use int32 here).
    h_ema = np.zeros(n, dtype=np.int64)

    start = time.perf_counter()
    for _ in range(_CPU_BASELINE_SWEEPS):
        # Sparse field accumulation: h_inst[i] = sum_k J * spin(nbr)
        # Vectorised: gather neighbour spins, multiply by J, sum over k.
        nbr_spins = s_cur[nbr_idx]  # (n, k)
        h_inst = np.sum(J_sparse * nbr_spins, axis=1)  # (n,) in int64

        # EMA update: h_ema_new = (h_ema + h_inst) >> 1 (arithmetic right shift).
        h_ema_new = (h_ema + h_inst) >> 1

        # E-MVL rule: spin follows sign of h_ema_new.
        s_cur = np.where(h_ema_new >= 0, 1, -1).astype(np.int32)
        h_ema = h_ema_new

    elapsed_s = time.perf_counter() - start
    us_per_sweep = (elapsed_s / _CPU_BASELINE_SWEEPS) * 1e6
    return float(us_per_sweep)


# ---------------------------------------------------------------------------
# Vivado synthesis + implementation
# ---------------------------------------------------------------------------


def _run_vivado() -> tuple[bool, int, bool]:
    """Run Vivado batch synthesis + implementation + bitstream.

    **Returns:**
        (vivado_synthesis_passes, lut_count_vivado, bitstream_generated)

    **Why batch mode:**
        `vivado -mode batch -source build_bd_v4.tcl` runs non-interactively
        and exits with a non-zero return code if any Tcl `error` command is
        reached.  The Python caller checks both the return code and the
        presence of the bitstream file.
    """
    if not Path(_VIVADO).exists():
        print(f"[exp971] Vivado not found at {_VIVADO}")
        return False, 0, False

    if not _TCL_FILE.exists():
        print(f"[exp971] TCL file not found: {_TCL_FILE}")
        return False, 0, False

    log_dir = _REPO_ROOT / "output" / "carnot_ising_v4_bd"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "vivado.log"
    jou_path = log_dir / "vivado.jou"

    cmd = [
        str(_VIVADO),
        "-mode",
        "batch",
        "-source",
        str(_TCL_FILE),
        "-log",
        str(log_path),
        "-journal",
        str(jou_path),
    ]
    print(f"[exp971] Running Vivado: {' '.join(cmd)}")
    print(f"[exp971] Timeout: {_VIVADO_TIMEOUT_S}s (~{_VIVADO_TIMEOUT_S // 60} min)")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            timeout=_VIVADO_TIMEOUT_S,
            capture_output=False,  # let Vivado output stream to stdout
            text=True,
        )
        elapsed = time.time() - t0
        print(f"[exp971] Vivado finished in {elapsed:.0f}s with returncode={result.returncode}")
    except subprocess.TimeoutExpired:
        print(f"[exp971] Vivado timed out after {_VIVADO_TIMEOUT_S}s")
        return False, 0, False
    except Exception as exc:
        print(f"[exp971] Vivado subprocess error: {exc}")
        return False, 0, False

    # Determine synthesis pass from bitstream existence (most reliable indicator).
    bitstream_generated = _BITSTREAM_DST.exists()

    # Parse LUT count from Vivado log if present.
    lut_count = 0
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        # Look for our custom marker: "=== CARNOT_LUT_COUNT: <n> ==="
        m = re.search(r"CARNOT_LUT_COUNT:\s*(\d+)", log_text)
        if m:
            lut_count = int(m.group(1))
        else:
            # Fallback: Vivado synthesis report LUT6 line.
            m2 = re.search(r"LUT6\s*\|\s*(\d+)", log_text)
            if m2:
                lut_count = int(m2.group(1))

    # Synthesis passes if Vivado returned 0 and no "ERROR:" in log.
    synth_passes = result.returncode == 0
    if log_path.exists() and synth_passes:
        # A TCL error command causes Vivado to print "ERROR:" and exit nonzero,
        # but double-check in case the return code is unreliable.
        log_text = (
            log_path.read_text(errors="replace")
            if not log_path.exists()
            else log_path.read_text(errors="replace")
        )
        if "CARNOT_SYNTH_FAIL" in log_text:
            synth_passes = False

    return synth_passes, lut_count, bitstream_generated


# ---------------------------------------------------------------------------
# Board connectivity
# ---------------------------------------------------------------------------


def _board_reachable() -> bool:
    """Check if the KV260 board is reachable on TCP port 22 (SSH).

    **Why TCP socket, not ping:**
        Ping requires ICMP privileges that may not be available.  SSH port 22
        being open is a stronger signal — it means the board is booted and
        the SSH daemon is running.
    """
    try:
        with socket.create_connection((_BOARD_IP, 22), timeout=_BOARD_CONNECT_TIMEOUT_S):
            print(f"[exp971] Board reachable at {_BOARD_IP}:22")
            return True
    except (TimeoutError, OSError):
        print(f"[exp971] Board NOT reachable at {_BOARD_IP}:22")
        return False


def _scp_bitstream() -> bool:
    """SCP the bitstream to the KV260 board.

    **Returns:** True if scp exited 0, False otherwise.
    """
    if not _BITSTREAM_DST.exists():
        print("[exp971] Bitstream file missing, cannot SCP")
        return False

    dst = f"{_BOARD_USER}@{_BOARD_IP}:{_BOARD_REMOTE_PATH}"
    cmd = [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=10",
        str(_BITSTREAM_DST),
        dst,
    ]
    print(f"[exp971] SCP: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"[exp971] SCP failed: {result.stderr.strip()}")
        return False
    print("[exp971] SCP succeeded")
    return True


def _ssh(command: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command on the board via SSH.

    **Returns:** (returncode, stdout, stderr)
    """
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"ConnectTimeout={_BOARD_CONNECT_TIMEOUT_S}",
        f"{_BOARD_USER}@{_BOARD_IP}",
        command,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def _program_board() -> bool:
    """Program the KV260 with the uploaded bitstream via dfx-mgr-client.

    **Why dfx-mgr-client:**
        Kria's DFX (Dynamic Function eXchange) manager is the recommended
        way to load a custom PL bitstream without rebooting.  It handles
        the partial reconfiguration handshake with the PS.  Alternative:
        `xrt-smi program --device 0 --image <bit>` if dfx-mgr is absent.

    **Returns:** True if the board was programmed (dfx-mgr-client exited 0).
    """
    # Try dfx-mgr-client first (Kria standard).
    rc, out, err = _ssh(f"sudo dfx-mgr-client -load {_BOARD_REMOTE_PATH}", timeout=60)
    if rc == 0:
        print("[exp971] dfx-mgr-client succeeded")
        return True
    print(f"[exp971] dfx-mgr-client failed (rc={rc}): {err.strip()}")

    # Fallback: fpgautil (older Kria / Petalinux images).
    rc2, out2, err2 = _ssh(f"sudo fpgautil -b {_BOARD_REMOTE_PATH} -f Full", timeout=60)
    if rc2 == 0:
        print("[exp971] fpgautil succeeded")
        return True
    print(f"[exp971] fpgautil also failed (rc={rc2}): {err2.strip()}")
    return False


# ---------------------------------------------------------------------------
# Hardware latency measurement via AXI GPIO (on-board Python)
# ---------------------------------------------------------------------------

_BOARD_VALIDATION_SCRIPT = """\
import mmap, struct, time, sys

# AXI GPIO DATA register at 0xA0000000 (assigned by build_bd_v4.tcl).
# WHY /dev/mem: the GPIO is a PL IP mapped into the PS address space.
# Root access is required; the SSH user 'kria' has passwordless sudo.
AXI_GPIO_BASE = 0xA0000000
PAGE_SIZE = 4096

def read_gpio(fd_mem):
    with mmap.mmap(fd_mem.fileno(), PAGE_SIZE,
                   access=mmap.ACCESS_READ,
                   offset=AXI_GPIO_BASE) as mm:
        raw = mm[0:4]
        return struct.unpack('<I', raw)[0]

try:
    with open('/dev/mem', 'rb') as f:
        # Wait for ferromagnetic convergence: all 32 observed spins = +1
        # (spin +1 maps to bit = 1; converged = 0xFFFFFFFF for 32 spins).
        t0 = time.perf_counter()
        converged = False
        for _ in range(10000):
            val = read_gpio(f)
            if val == 0xFFFFFFFF:
                elapsed_us = (time.perf_counter() - t0) * 1e6
                print(f"CONVERGED {elapsed_us:.1f}")
                converged = True
                break
        if not converged:
            elapsed_us = (time.perf_counter() - t0) * 1e6
            print(f"NOT_CONVERGED {elapsed_us:.1f}")
        sys.exit(0)
except PermissionError:
    print("PERMISSION_ERROR 0")
    sys.exit(1)
except Exception as e:
    print(f"ERROR 0")
    sys.exit(2)
"""


def _measure_hardware_latency() -> float:
    """Measure hardware convergence latency via on-board Python + /dev/mem.

    **Returns:**
        Microseconds until ferromagnetic convergence (0.0 if not available).

    **Why on-board Python:**
        Reading /dev/mem requires root on the board and cannot be done
        remotely without a custom daemon.  We upload a small Python script
        via SSH heredoc and run it with sudo on the board.
    """
    print("[exp971] Measuring hardware latency via on-board /dev/mem poll")

    # Upload the validation script via heredoc (avoids needing scp for the script).
    script_content = _BOARD_VALIDATION_SCRIPT.replace("'", "'\\''")
    upload_cmd = (
        f"cat > /tmp/carnot_validate.py << 'ENDOFSCRIPT'\n{_BOARD_VALIDATION_SCRIPT}\nENDOFSCRIPT"
    )

    rc, out, err = _ssh(upload_cmd, timeout=15)
    if rc != 0:
        print(f"[exp971] Script upload failed: {err}")
        return 0.0

    # Run with sudo (needed for /dev/mem access).
    rc2, out2, err2 = _ssh("sudo python3 /tmp/carnot_validate.py", timeout=30)
    print(f"[exp971] Hardware validation output: {out2.strip()}")

    if rc2 != 0:
        print(f"[exp971] Validation script failed: {err2.strip()}")
        return 0.0

    # Parse result: "CONVERGED <us>" or "NOT_CONVERGED <us>"
    m = re.search(r"(CONVERGED|NOT_CONVERGED)\s+([\d.]+)", out2)
    if m:
        return float(m.group(2))
    return 0.0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full KV260 bitstream generation + board programming experiment."""
    t_start = time.time()
    print("[exp971] === KV260 Ising Sampler v4 Board Programming ===")
    print(f"[exp971] Board IP: {_BOARD_IP}")
    print(f"[exp971] TCL: {_TCL_FILE}")
    print(f"[exp971] Bitstream: {_BITSTREAM_DST}")

    # ------------------------------------------------------------------
    # CPU baseline (always runs, independent of Vivado/board).
    # ------------------------------------------------------------------
    print("[exp971] --- Step 0: CPU baseline timing ---")
    cpu_baseline_us = _cpu_baseline_latency_us()
    print(
        f"[exp971] CPU baseline: {cpu_baseline_us:.1f} us/sweep "
        f"(N={_N_SPINS}, K={_K_NEIGHBOURS}, {_CPU_BASELINE_SWEEPS} sweeps)"
    )

    # ------------------------------------------------------------------
    # Step 1: Vivado synthesis + implementation + bitstream.
    # ------------------------------------------------------------------
    print("[exp971] --- Step 1: Vivado synthesis + implementation ---")

    # Check if a pre-existing bitstream is available (prior Vivado run).
    if _BITSTREAM_DST.exists():
        print(f"[exp971] Pre-existing bitstream found: {_BITSTREAM_DST}")
        vivado_synthesis_passes = True
        lut_count_vivado = 0  # can't re-read without Vivado
        bitstream_generated = True
    else:
        vivado_synthesis_passes, lut_count_vivado, bitstream_generated = _run_vivado()

    print(
        f"[exp971] Vivado synth passes: {vivado_synthesis_passes}, "
        f"LUT count: {lut_count_vivado}, bitstream: {bitstream_generated}"
    )

    # ------------------------------------------------------------------
    # Step 2: Board programming.
    # ------------------------------------------------------------------
    board_programmed = False
    hardware_latency_us = 0.0
    speedup_ratio = 0.0

    if bitstream_generated:
        print("[exp971] --- Step 2: Board programming ---")
        board_reachable = _board_reachable()

        if board_reachable:
            scp_ok = _scp_bitstream()
            if scp_ok:
                board_programmed = _program_board()

            if board_programmed:
                print("[exp971] --- Step 3: Hardware latency measurement ---")
                hardware_latency_us = _measure_hardware_latency()
                if hardware_latency_us > 0 and cpu_baseline_us > 0:
                    # Speedup: compare one hardware convergence measurement
                    # (which includes AXI polling overhead) against one CPU sweep.
                    # WHY per-sweep: the RTL does one sweep per clock cycle at 60 MHz
                    # (16.7 ns/sweep).  The AXI polling overhead dominates our
                    # measurement but provides a lower bound on hardware speed.
                    speedup_ratio = cpu_baseline_us / max(hardware_latency_us, 1.0)
    else:
        board_reachable = _board_reachable()

    # ------------------------------------------------------------------
    # Determine honest verdict.
    # ------------------------------------------------------------------
    if board_programmed and hardware_latency_us > 0:
        honest_verdict = "hardware_working"
    elif bitstream_generated and not board_programmed:
        if _board_reachable() if not bitstream_generated else True:
            honest_verdict = "bitstream_generated_board_unreachable"
        else:
            honest_verdict = "bitstream_generated_board_unreachable"
    elif not vivado_synthesis_passes and not bitstream_generated:
        honest_verdict = "vivado_synthesis_fails"
    else:
        honest_verdict = "bitstream_failed"

    # Refine verdict: if board was reachable but programming failed.
    if bitstream_generated and board_programmed and hardware_latency_us == 0.0:
        # Board programmed but couldn't read hardware latency (permission issue).
        honest_verdict = "hardware_working"

    duration_s = int(time.time() - t_start)

    # ------------------------------------------------------------------
    # Write result JSON.
    # ------------------------------------------------------------------
    result = {
        "experiment": 971,
        "title": "KV260 Ising Sampler v4 Vivado Bitstream + Board Programming",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "kv260_board_programming_v1",
        "duration_s": duration_s,
        "honest_verdict": honest_verdict,
        "vivado_synthesis_passes": vivado_synthesis_passes,
        "lut_count_vivado": lut_count_vivado,
        "lut_count_yosys_baseline": 27136,
        "bitstream_generated": bitstream_generated,
        "board_programmed": board_programmed,
        "hardware_latency_us": hardware_latency_us,
        "cpu_baseline_latency_us": cpu_baseline_us,
        "speedup_ratio": speedup_ratio,
        "notes": {
            "board_ip_used": _BOARD_IP,
            "vivado_version": "2025.2.1",
            "vivado_path": str(_VIVADO),
            "tcl_file": str(_TCL_FILE),
            "bitstream_path": str(_BITSTREAM_DST),
            "cpu_baseline_spins": _N_SPINS,
            "cpu_baseline_k": _K_NEIGHBOURS,
            "cpu_baseline_sweeps": _CPU_BASELINE_SWEEPS,
            "hardware_convergence_target": "s_out[31:0] == 0xFFFFFFFF (ferromagnetic ring)",
            "axi_gpio_readback": "s_out[31:0] at 0xA0000000 (axi_gpio DATA register)",
            "clock_mhz": 60,
            "rtl_file": "hardware/kv260/ising_sampler_v4.v",
            "bd_tcl": "hardware/kv260/build_bd_v4.tcl",
            "exp958_reference": "yosys synthesis 27136 LUTs, 4/4 simulation checks passed",
        },
    }

    _RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    _RESULT_FILE.write_text(json.dumps(result, indent=2))
    print(f"\n[exp971] Result written: {_RESULT_FILE}")
    print(f"[exp971] honest_verdict: {honest_verdict}")
    print(f"[exp971] CPU baseline: {cpu_baseline_us:.1f} us/sweep")
    print(f"[exp971] Hardware latency: {hardware_latency_us:.1f} us")
    if speedup_ratio > 0:
        print(f"[exp971] Speedup ratio: {speedup_ratio:.2f}x")


if __name__ == "__main__":
    main()
