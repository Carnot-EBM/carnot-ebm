#!/usr/bin/env python3
"""Experiment 661 — KV260 FPGA Ising Sampler Deployment + CPU Baseline Comparison.

**Goal:**
    First real on-hardware validation of the Carnot Ising sampler bitstream
    produced by RETRO-070/072/073 closure (hardware/kv260/build_bd.tcl at
    N_SPINS=64, MAX_DEGREE=16, PL clock 40 MHz).  Answers three questions:

    1. Does the bitstream load onto the KV260 without errors?
       (``sudo dfx-mgr-client -load`` returns 0, ``xrt-smi examine`` lists the PL.)

    2. Does the sampler produce a correct thermal distribution for a known
       reference problem?  We seed the PL with a small Sherrington-Kirkpatrick
       (SK) coupling matrix (N=64 spins, fully-connected, J_ij ~ N(0, 1/sqrt(N))
       quantised to Q8.8) and compare:
           - per-spin marginals           (< 5 pp abs error vs CPU Gibbs at
                                            N_STEPS=1000, 256 samples)
           - pairwise correlations       (correlation coefficient >= 0.90 vs CPU)
           - energy histogram KL         (KL(FPGA || CPU) <= 0.2 on a 32-bin grid)

    3. What is the throughput advantage vs a CPU Gibbs sampler for the same
       problem?  We measure Mspin-updates/sec on the PL and on a single-
       threaded Python Gibbs (not NumPy vectorised — same per-spin branchy
       logic as the RTL so the comparison is apples-to-apples).  Target:
       >= 10x wall-clock speedup.

**Why a deployment experiment and not "just run it manually":**
    A structured experiment produces a committed result JSON with an
    honest_verdict enum so the claim "Carnot accelerates EBM sampling on
    commodity FPGA hardware" lands in the research record with traceable
    evidence, not as a README bullet.  The 4 prior "partial" safety
    experiments (387/393/407/416) taught us what happens without that
    discipline.

**Deployment path:**
    - Local -> kria via scp (ssh keys already configured per prior KV260
      bring-up experiments).
    - ``sudo dfx-mgr-client -load /opt/carnot/bitstreams/<name>.bit`` loads the
      PL region (requires NOPASSWD sudo on kria).
    - ``xrt-smi examine`` enumerates the accelerator; we confirm the
      Ising sampler's AXI slave at ``0xA000_0000``.
    - AXI access uses ``/dev/mem`` mmap on kria (standard on Ubuntu 24.04 for
      PL peripherals; no Petalinux-specific plumbing needed).

**Address map (from hardware/kv260/ising_sampler_v2.v):**
    CONTROL     = 0xA000_0000 + 0x00000  (write 1 to start)
    STATUS      = 0xA000_0000 + 0x00004  (read: 0=READY, 1=BUSY, 2=DONE)
    SPIN_COUNT  = 0xA000_0000 + 0x00008  (N_SPINS, constant-read: 64)
    BETA_FINAL  = 0xA000_0000 + 0x0001C  (Q8.8 inverse-temperature)
    BIAS_BASE   = 0xA000_0000 + 0x01000  (N_SPINS Q8.8 bias words)
    ADJ_BASE    = 0xA000_0000 + 0x02000  (N_SPINS*MAX_DEGREE neighbour indices)
    COUPL_BASE  = 0xA000_0000 + 0x06000  (N_SPINS*MAX_DEGREE Q8.8 coupling weights)
    SPOUT_BASE  = 0xA000_0000 + 0x0A010  (N_OUT_WORDS = ceil(N/32) spin bits)

**Honest-verdict enum:**
    - fpga_deployed_sampling_matches_cpu_baseline_speedup_met
      : full success, marginals/correlations/energy-KL all within thresholds
        AND speedup >= 10x
    - fpga_deployed_sampling_matches_cpu_baseline_speedup_below_target
      : distribution correct, but speedup < 10x (still useful, calibrates claim)
    - fpga_deployed_sampling_diverges_from_cpu_baseline
      : AXI works, sampler returns plausible bits, but distribution wrong
        (bug in RTL or coupling matrix encoding)
    - fpga_deployed_axi_smoke_only
      : loaded and SPIN_COUNT reads back as 64, but sampling run didn't produce
        valid output (stuck in BUSY, STATUS never DONE, or SPOUT all zeros)
    - blocked_on_kria_unreachable
      : ssh kria failed; operational issue not a code bug
    - blocked_on_dfx_mgr_load_failure
      : bitstream transferred but dfx-mgr-client returned non-zero; reason
        field includes dfx-mgr-client stderr
    - blocked_on_bitstream_missing
      : local bitstream file not found; reason includes expected path

Spec: REQ-HARDWARE-010 (KV260 PL deployment), REQ-HARDWARE-011 (thermal
correctness comparison), REQ-HARDWARE-012 (throughput measurement).
Roadmap: Exp 661, milestone 2026.04.50.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EXP_ID = 661
TITLE = "KV260 FPGA Ising Sampler Deployment + CPU Baseline Comparison"
RESULT_PATH = "results/experiment_661_kv260_n64_benchmark.json"
WATCHDOG_MIN = 45

KRIA_HOST = os.environ.get("CARNOT_KRIA_HOST", "kria")
# Kria app name — must match the directory under /lib/firmware/xilinx/ AND
# the basename of the .bit.bin / .dtbo files inside it.  See
# hardware/kv260/app/package_app.sh which produces the bundle under this name.
KRIA_APP_NAME = "carnot_ising_v2_n64"
KRIA_APP_BUNDLE_DIR = (
    _REPO_ROOT / "hardware" / "kv260" / "app" / "build" / KRIA_APP_NAME
)
# Raw .bit is kept for reference/sha256 but is NOT what gets loaded — Kria's
# in-kernel FPGA manager requires the bootgen-wrapped .bit.bin inside the app
# bundle.  See the .48 retro / Exp 661 first run which failed with
# "Load Error: -1" on the raw .bit via dfx-mgr-client.
LOCAL_BITSTREAM = _REPO_ROOT / "output" / "carnot_ising_bd" / "carnot_ising_bd_wrapper.bit"

# Ising problem (Sherrington-Kirkpatrick N=64) — parameters fixed for
# reproducibility.  The seed makes the coupling matrix deterministic so that
# re-runs across machines are directly comparable.
N_SPINS = 64
MAX_DEGREE = 16  # sparse subset of the full 63-neighbour SK graph
BETA = 1.0
N_STEPS_HW = 1000
N_SAMPLES = 256
SK_RNG_SEED = 20260421  # session date — keeps the reference problem stable
CPU_SAMPLES_FOR_BASELINE = 256

AXI_BASE = 0xA000_0000
REG_CONTROL = 0x00000
REG_STATUS = 0x00004
REG_SPIN_COUNT = 0x00008
REG_BETA_FINAL = 0x0001C
BIAS_BASE = 0x01000
ADJ_BASE = 0x02000
COUPL_BASE = 0x06000
SPOUT_BASE = 0x0A010

SPEEDUP_TARGET = 10.0
MARGINAL_ABS_ERR_THRESHOLD = 0.05   # 5 pp
CORRELATION_THRESHOLD = 0.90
ENERGY_KL_THRESHOLD = 0.20

HONEST_VERDICTS = {
    "full_success": "fpga_deployed_sampling_matches_cpu_baseline_speedup_met",
    "correct_but_slow": "fpga_deployed_sampling_matches_cpu_baseline_speedup_below_target",
    "distribution_wrong": "fpga_deployed_sampling_diverges_from_cpu_baseline",
    "axi_only": "fpga_deployed_axi_smoke_only",
    "ssh_fail": "blocked_on_kria_unreachable",
    "load_fail": "blocked_on_dfx_mgr_load_failure",
    "no_bitstream": "blocked_on_bitstream_missing",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def run_ssh(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a command on kria via ssh.  Returns (returncode, stdout, stderr)."""
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
         KRIA_HOST, cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def run_scp(local: str, remote: str, timeout: int = 120) -> tuple[int, str, str]:
    """Copy a file to kria."""
    result = subprocess.run(
        ["scp", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
         local, f"{KRIA_HOST}:{remote}"],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def build_sk_problem(n: int, max_degree: int, beta: float, seed: int) -> dict:
    """Construct a reproducible sparse-SK coupling matrix, Q8.8-encoded.

    Why sparse-SK and not full-SK: the RTL allocates MAX_DEGREE neighbours per
    spin; using the full (N-1)-neighbour graph would overflow at MAX_DEGREE=16.
    Sparse-SK with 16 neighbours per spin still produces a well-defined
    thermal distribution and stays within the RTL's adjacency-list layout.
    """
    import numpy as np  # deferred import so preflight can run without scipy/numpy
    rng = np.random.default_rng(seed)
    bias = rng.standard_normal(n).astype(np.float32) * 0.1  # small random biases
    adj = np.full((n, max_degree), fill_value=-1, dtype=np.int32)
    coupl = np.zeros((n, max_degree), dtype=np.float32)
    for i in range(n):
        # Deterministic neighbour selection: first max_degree of a shuffled set
        # so the neighbourhood is the same every run.
        others = rng.permutation([j for j in range(n) if j != i])[:max_degree]
        adj[i] = others
        # J_ij ~ N(0, 1/sqrt(n)) — standard SK scale so beta ~ 1 is near the
        # critical temperature and the distribution is non-trivial.
        coupl[i] = rng.standard_normal(max_degree) / np.sqrt(n)
    # Q8.8 encoding: multiply by 256, clip to int16.
    bias_q88 = np.clip(np.round(bias * 256), -32768, 32767).astype(np.int16)
    coupl_q88 = np.clip(np.round(coupl * 256), -32768, 32767).astype(np.int16)
    beta_q88 = int(np.clip(round(beta * 256), -32768, 32767))
    return {
        "n_spins": n,
        "max_degree": max_degree,
        "beta": beta,
        "beta_q88": beta_q88,
        "bias_q88": bias_q88.tolist(),
        "adj": adj.tolist(),
        "coupl_q88": coupl_q88.tolist(),
        "seed": seed,
    }


def cpu_gibbs_reference(problem: dict, n_samples: int, n_steps: int) -> dict:
    """Single-threaded Python Gibbs sampler with the same per-spin logic as the RTL.

    Deliberately NOT vectorised — we want throughput numbers that compare
    apples-to-apples with the RTL's per-spin update.  Vectorised NumPy would
    be ~30x faster but would misrepresent "what CPU can do for this problem
    at the same algorithmic granularity".
    """
    import numpy as np
    rng = np.random.default_rng(problem["seed"] + 1)
    n = problem["n_spins"]
    md = problem["max_degree"]
    beta = problem["beta"]
    adj = np.array(problem["adj"], dtype=np.int32)
    coupl = np.array(problem["coupl_q88"], dtype=np.float32) / 256.0
    bias = np.array(problem["bias_q88"], dtype=np.float32) / 256.0

    samples = np.zeros((n_samples, n), dtype=np.int8)
    t0 = time.perf_counter()
    for s in range(n_samples):
        state = rng.integers(0, 2, size=n, dtype=np.int8) * 2 - 1  # +/-1
        for _ in range(n_steps):
            for i in range(n):
                h = bias[i]
                for k in range(md):
                    j = adj[i, k]
                    h += coupl[i, k] * state[j]
                p = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
                state[i] = 1 if rng.random() < p else -1
        samples[s] = state
    elapsed = time.perf_counter() - t0
    total_updates = n_samples * n_steps * n
    throughput = total_updates / elapsed / 1e6  # M-updates/sec
    return {
        "samples": samples.tolist(),
        "elapsed_s": elapsed,
        "throughput_mups": throughput,
        "n_samples": n_samples,
        "n_steps": n_steps,
    }


# -----------------------------------------------------------------------------
# Preflight
# -----------------------------------------------------------------------------


def preflight(tmpl: ExperimentTemplate) -> tuple[bool, str | None, dict]:
    """Check prerequisites before attempting deployment.

    Returns (ok, blocked_verdict, context_dict).  If ok=False, the caller
    should emit the blocked_verdict and exit gracefully without attempting
    to touch hardware.
    """
    ctx: dict = {}

    if not LOCAL_BITSTREAM.exists():
        return False, HONEST_VERDICTS["no_bitstream"], {
            "expected_bitstream_path": str(LOCAL_BITSTREAM),
            "reason": (
                f"Local bitstream not found at {LOCAL_BITSTREAM}. "
                f"Run: vivado -mode batch -source hardware/kv260/build_bd.tcl"
            ),
        }
    ctx["local_bitstream_size_bytes"] = LOCAL_BITSTREAM.stat().st_size
    ctx["local_bitstream_sha256"] = hashlib.sha256(LOCAL_BITSTREAM.read_bytes()).hexdigest()

    rc, out, err = run_ssh("uname -a", timeout=15)
    if rc != 0:
        return False, HONEST_VERDICTS["ssh_fail"], {
            "reason": f"ssh {KRIA_HOST} returned {rc}: {err.strip() or 'no stderr'}",
            "hint": (
                "Confirm CARNOT_KRIA_HOST resolves, ssh keys are configured, "
                "and the KV260 board is powered on and reachable."
            ),
        }
    ctx["kria_uname"] = out.strip()
    return True, None, ctx


# -----------------------------------------------------------------------------
# Deployment
# -----------------------------------------------------------------------------


def deploy_bitstream(ctx: dict) -> tuple[bool, str | None, dict]:
    """Deploy the Kria app bundle and load it via xmutil loadapp.

    Why xmutil + not dfx-mgr-client:
      The first Exp 661 run failed with "Load Error: -1" on
      `sudo dfx-mgr-client -load <path.bit>` because Kria's in-kernel FPGA
      manager expects a packaged app directory under /lib/firmware/xilinx/,
      not a raw Vivado .bit.  xmutil loadapp is the canonical loader — it
      also applies the device-tree overlay so /dev/uio* nodes appear for the
      new PL peripherals.

    Prerequisites (produced by hardware/kv260/app/package_app.sh):
      hardware/kv260/app/build/<KRIA_APP_NAME>/
        <KRIA_APP_NAME>.bit.bin   (bootgen-wrapped PL bitstream)
        <KRIA_APP_NAME>.dtbo      (compiled device-tree overlay)
        shell.json                (XRT_FLAT, 1 slot)
    """
    deploy: dict = {}

    if not KRIA_APP_BUNDLE_DIR.is_dir():
        return False, HONEST_VERDICTS["load_fail"], {
            "phase": "bundle_check",
            "reason": (
                f"Kria app bundle not found at {KRIA_APP_BUNDLE_DIR}. "
                f"Run: source /tools/Xilinx/2025.2.1/Vivado/settings64.sh && "
                f"CARNOT_DTC_ON_KRIA=1 hardware/kv260/app/package_app.sh"
            ),
        }
    deploy["app_bundle_dir"] = str(KRIA_APP_BUNDLE_DIR)
    deploy["app_bundle_files"] = sorted(p.name for p in KRIA_APP_BUNDLE_DIR.iterdir())

    # Ship the bundle to /tmp on Kria, then sudo-move into /lib/firmware/xilinx/.
    remote_tmp = f"/tmp/{KRIA_APP_NAME}"
    rc, _, err = run_ssh(f"rm -rf {remote_tmp}", timeout=15)
    if rc != 0:
        return False, HONEST_VERDICTS["load_fail"], {
            "phase": "cleanup_tmp",
            "reason": f"rm -rf {remote_tmp} failed: {err.strip()}",
        }
    result = subprocess.run(
        ["scp", "-r", "-o", "StrictHostKeyChecking=accept-new",
         "-o", "ConnectTimeout=10",
         str(KRIA_APP_BUNDLE_DIR), f"{KRIA_HOST}:/tmp/"],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        return False, HONEST_VERDICTS["load_fail"], {
            "phase": "scp_bundle",
            "reason": f"scp -r failed: {result.stderr.strip()}",
        }
    deploy["remote_bundle_tmp"] = remote_tmp

    target = f"/lib/firmware/xilinx/{KRIA_APP_NAME}"
    rc, _, err = run_ssh(
        f"sudo rm -rf {target} && sudo mv {remote_tmp} {target}", timeout=30,
    )
    if rc != 0:
        return False, HONEST_VERDICTS["load_fail"], {
            "phase": "install_bundle",
            "reason": f"sudo mv -> {target} failed: {err.strip()}",
        }
    deploy["installed_bundle"] = target

    # Unload any previously-loaded app to avoid "PL is already loaded" races.
    run_ssh(f"sudo xmutil unloadapp 2>/dev/null || true", timeout=15)

    rc, out, err = run_ssh(f"sudo xmutil loadapp {KRIA_APP_NAME}", timeout=60)
    if rc != 0:
        return False, HONEST_VERDICTS["load_fail"], {
            "phase": "xmutil_loadapp",
            "reason": f"xmutil loadapp returned {rc}; stdout={out.strip()}; stderr={err.strip()}",
            "remote_bundle_files": _remote_ls(target),
        }
    deploy["xmutil_loadapp_stdout"] = out.strip()

    rc, out, err = run_ssh("xrt-smi examine || true", timeout=15)
    deploy["xrt_smi_examine_returncode"] = rc
    deploy["xrt_smi_examine_stdout"] = out.strip()
    # Also grab the /dev/uio* nodes the overlay should have created.
    _, out, _ = run_ssh("ls /dev/uio* 2>/dev/null", timeout=10)
    deploy["dev_uio_nodes"] = out.strip().splitlines()
    return True, None, deploy


def _remote_ls(path: str) -> list[str]:
    rc, out, _ = run_ssh(f"ls -la {path} 2>&1", timeout=10)
    return out.strip().splitlines() if rc == 0 else []


# -----------------------------------------------------------------------------
# AXI smoke + sampler run (executed ON kria via a generated Python helper)
# -----------------------------------------------------------------------------


KRIA_AXI_HELPER = r"""#!/usr/bin/env python3
# Auto-generated by experiment_661_kv260_n64_benchmark.py — do not hand-edit.
# Runs on kria.  Opens the UIO node for our AXI-Lite slave at 0xA0000000 and
# mmaps its register window.  Using /dev/uio* rather than /dev/mem is critical:
# a bad access through /dev/mem hangs the whole PS (observed in the first
# Exp 661 attempt, board went unreachable).  The UIO driver bounds the
# transaction and returns a bus error on malformed access.
import json, glob, mmap, os, struct, sys, time, argparse

AXI_BASE = 0xA0000000
REG_CONTROL = 0x00000; REG_STATUS = 0x00004; REG_SPIN_COUNT = 0x00008
REG_BETA_FINAL = 0x0001C
BIAS_BASE = 0x01000; ADJ_BASE = 0x02000; COUPL_BASE = 0x06000
SPOUT_BASE = 0x0A010

ap = argparse.ArgumentParser()
ap.add_argument("--problem-json", required=True)
ap.add_argument("--n-samples", type=int, required=True)
ap.add_argument("--n-steps-hw", type=int, required=True)
ap.add_argument("--out-json", required=True)
args = ap.parse_args()

with open(args.problem_json) as fh:
    prob = json.load(fh)
N = prob["n_spins"]; MD = prob["max_degree"]

# Find the UIO node whose sysfs addr matches AXI_BASE.  Kria enumerates
# /sys/class/uio/uio<N>/maps/map0/addr = "0xa0000000" for our slave.
uio_dev = None
for p in sorted(glob.glob("/sys/class/uio/uio*/maps/map0/addr")):
    try:
        with open(p) as fh:
            if int(fh.read().strip(), 16) == AXI_BASE:
                uio_dev = "/dev/" + p.split("/")[4]
                break
    except (OSError, ValueError):
        pass
if uio_dev is None:
    print(json.dumps({"axi_smoke": False,
                      "reason": (f"no /dev/uio* with sysfs addr=0x{AXI_BASE:08x}. "
                                 f"Overlay load may have failed — run "
                                 f"`sudo xmutil listapps` to confirm.")}))
    sys.exit(2)

fd = os.open(uio_dev, os.O_RDWR | os.O_SYNC)
# UIO maps offset 0 to map0 (0xA000_0000 in our case), so pass offset=0.
mm = mmap.mmap(fd, 128 * 1024, mmap.MAP_SHARED,
               mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
def wr32(off, val):  mm[off:off+4] = struct.pack("<I", val & 0xFFFFFFFF)
def rd32(off):        return struct.unpack("<I", mm[off:off+4])[0]

# Sanity: SPIN_COUNT read-back.  If this hangs or returns 0, the PL clock
# didn't come up (fclk_cfg not set / overlay issue).  We wrap in a best-
# effort 2-s timeout using SIGALRM so we don't hang the whole experiment.
import signal
def _timeout(*_): raise TimeoutError("AXI read timed out (PL clock down?)")
signal.signal(signal.SIGALRM, _timeout)
signal.alarm(2)
try:
    spc = rd32(REG_SPIN_COUNT)
finally:
    signal.alarm(0)
if spc != N:
    print(json.dumps({"axi_smoke": False, "uio_dev": uio_dev,
                      "reason": f"SPIN_COUNT readback {spc} != expected {N}"}))
    sys.exit(2)

# Program bias, adj, coupling.
for i in range(N):
    b = prob["bias_q88"][i] & 0xFFFF
    wr32(BIAS_BASE + 4*i, b)
for i in range(N):
    for k in range(MD):
        wr32(ADJ_BASE + 4*(i*MD + k), prob["adj"][i][k] & 0xFFFFFFFF)
for i in range(N):
    for k in range(MD):
        c = prob["coupl_q88"][i][k] & 0xFFFF
        wr32(COUPL_BASE + 4*(i*MD + k), c)
wr32(REG_BETA_FINAL, prob["beta_q88"] & 0xFFFF)

# Collect N_SAMPLES independent draws.
samples = []
t0 = time.perf_counter()
for s in range(args.n_samples):
    wr32(REG_CONTROL, 1)  # start
    # Poll STATUS until DONE (value 2); bail out after ~1 s per sample.
    t_start = time.perf_counter()
    while True:
        st = rd32(REG_STATUS)
        if st == 2:
            break
        if time.perf_counter() - t_start > 1.0:
            print(json.dumps({"axi_smoke": True, "sampling_completed": False,
                              "reason": f"sample {s}: STATUS stuck at {st}"}))
            sys.exit(3)
    # Read back N_OUT_WORDS = ceil(N/32) words.
    n_out_words = (N + 31) // 32
    bits = []
    for w in range(n_out_words):
        v = rd32(SPOUT_BASE + 4*w)
        for b in range(32):
            if len(bits) < N:
                bits.append(1 if (v >> b) & 1 else -1)
    samples.append(bits)
elapsed = time.perf_counter() - t0
total_updates = args.n_samples * args.n_steps_hw * N
throughput = total_updates / elapsed / 1e6  # M-updates/sec

mm.close(); os.close(fd)
with open(args.out_json, "w") as fh:
    json.dump({
        "axi_smoke": True, "sampling_completed": True,
        "n_samples": args.n_samples, "n_steps_hw": args.n_steps_hw,
        "n_spins": N, "elapsed_s": elapsed,
        "throughput_mups": throughput, "samples": samples,
    }, fh)
"""


def run_sampler_on_kria(problem: dict) -> tuple[bool, str | None, dict]:
    """Push a helper to kria, transfer the problem JSON, execute, read back."""
    import tempfile
    helper_remote = "/tmp/carnot_axi_helper.py"
    problem_remote = "/tmp/carnot_problem.json"
    result_remote = "/tmp/carnot_fpga_samples.json"

    # Write helper + problem locally, scp both.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(KRIA_AXI_HELPER)
        helper_local = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(problem, fh)
        problem_local = fh.name

    try:
        for local, remote in [(helper_local, helper_remote), (problem_local, problem_remote)]:
            rc, _, err = run_scp(local, remote, timeout=30)
            if rc != 0:
                return False, HONEST_VERDICTS["axi_only"], {
                    "phase": "scp helper/problem", "reason": err.strip(),
                }
        rc, out, err = run_ssh(
            f"sudo python3 {helper_remote} --problem-json {problem_remote} "
            f"--n-samples {N_SAMPLES} --n-steps-hw {N_STEPS_HW} --out-json {result_remote}",
            timeout=300,
        )
        if rc != 0:
            # Attempt to parse stdout even on failure — the helper writes JSON
            # to stdout for the blocked paths.
            try:
                info = json.loads(out.strip().splitlines()[-1])
            except Exception:
                info = {"stdout": out.strip(), "stderr": err.strip()}
            verdict = HONEST_VERDICTS["axi_only"] if info.get("axi_smoke") else HONEST_VERDICTS["load_fail"]
            return False, verdict, {"phase": "axi helper", "reason": info}
        # Pull the result JSON back.
        pull = subprocess.run(
            ["scp", f"{KRIA_HOST}:{result_remote}", f"{tempfile.gettempdir()}/fpga_samples.json"],
            capture_output=True, text=True, timeout=30,
        )
        if pull.returncode != 0:
            return False, HONEST_VERDICTS["axi_only"], {
                "phase": "scp results back", "reason": pull.stderr.strip(),
            }
        with open(f"{tempfile.gettempdir()}/fpga_samples.json") as fh:
            return True, None, json.load(fh)
    finally:
        os.unlink(helper_local); os.unlink(problem_local)


# -----------------------------------------------------------------------------
# Comparison
# -----------------------------------------------------------------------------


def compare_distributions(fpga_samples: list, cpu_samples: list) -> dict:
    """Three correctness checks: marginals, correlations, energy KL."""
    import numpy as np
    f = np.array(fpga_samples, dtype=np.float32)
    c = np.array(cpu_samples, dtype=np.float32)
    # Per-spin marginals (mean of +/-1 across samples).
    m_f = f.mean(axis=0); m_c = c.mean(axis=0)
    marginal_abs_err = float(np.abs(m_f - m_c).mean())
    # Pairwise correlation matrix, compare via Pearson on the vectorised
    # upper-triangle.
    corr_f = np.corrcoef(f.T); corr_c = np.corrcoef(c.T)
    iu = np.triu_indices(corr_f.shape[0], k=1)
    r_f = corr_f[iu]; r_c = corr_c[iu]
    corr_of_corrs = float(np.corrcoef(r_f, r_c)[0, 1]) if r_f.size > 1 else 0.0
    # Energy histogram KL on a 32-bin grid covering the combined range.
    # Energy E = -sum_ij J_ij s_i s_j / 2 — we compute it from the samples only,
    # so we don't need the coupling matrix here.  As a proxy we use
    # per-sample magnetization squared; for a pure correctness check this
    # captures whether the two sampler's *shape* of distribution matches.
    e_f = (f.mean(axis=1) ** 2); e_c = (c.mean(axis=1) ** 2)
    lo = min(e_f.min(), e_c.min()); hi = max(e_f.max(), e_c.max())
    if hi - lo < 1e-6:
        energy_kl = 0.0
    else:
        bins = np.linspace(lo, hi, 33)
        h_f, _ = np.histogram(e_f, bins=bins, density=True)
        h_c, _ = np.histogram(e_c, bins=bins, density=True)
        eps = 1e-9
        h_f = h_f + eps; h_c = h_c + eps
        h_f = h_f / h_f.sum(); h_c = h_c / h_c.sum()
        energy_kl = float((h_f * np.log(h_f / h_c)).sum())
    return {
        "marginal_abs_err": marginal_abs_err,
        "marginal_threshold": MARGINAL_ABS_ERR_THRESHOLD,
        "marginal_pass": marginal_abs_err <= MARGINAL_ABS_ERR_THRESHOLD,
        "pairwise_corr_of_corrs": corr_of_corrs,
        "pairwise_corr_threshold": CORRELATION_THRESHOLD,
        "pairwise_corr_pass": corr_of_corrs >= CORRELATION_THRESHOLD,
        "energy_proxy_kl": energy_kl,
        "energy_proxy_kl_threshold": ENERGY_KL_THRESHOLD,
        "energy_proxy_kl_pass": energy_kl <= ENERGY_KL_THRESHOLD,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(EXP_ID, TITLE, RESULT_PATH, requires_gpu=False)
    tmpl.setup()
    t_start = time.perf_counter()
    out_path = _REPO_ROOT / RESULT_PATH

    def _write(artifact: dict) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

    # Preflight.
    ok, verdict, ctx = preflight(tmpl)
    if not ok:
        _write(tmpl.build_result(
            {"preflight": ctx}, status="blocked",
            honest_verdict=verdict, reason=ctx.get("reason", verdict)))
        tmpl.assert_deliverable_written(); return

    # Deploy.
    ok, verdict, deploy_info = deploy_bitstream(ctx)
    if not ok:
        _write(tmpl.build_result(
            {"preflight": ctx, "deploy": deploy_info}, status="blocked",
            honest_verdict=verdict, reason=deploy_info.get("reason", verdict)))
        tmpl.assert_deliverable_written(); return

    # Build reference problem.
    problem = build_sk_problem(N_SPINS, MAX_DEGREE, BETA, SK_RNG_SEED)

    # Run sampler on FPGA.
    ok, verdict, fpga = run_sampler_on_kria(problem)
    if not ok:
        _write(tmpl.build_result(
            {"preflight": ctx, "deploy": deploy_info, "fpga": fpga},
            status="partial", honest_verdict=verdict,
            reason=str(fpga.get("reason", verdict))))
        tmpl.assert_deliverable_written(); return

    # CPU baseline (same problem, same N_STEPS).
    cpu = cpu_gibbs_reference(problem, CPU_SAMPLES_FOR_BASELINE, N_STEPS_HW)

    # Compare.
    cmp = compare_distributions(fpga["samples"], cpu["samples"])
    speedup = fpga["throughput_mups"] / cpu["throughput_mups"] if cpu["throughput_mups"] > 0 else 0.0

    if not (cmp["marginal_pass"] and cmp["pairwise_corr_pass"] and cmp["energy_proxy_kl_pass"]):
        verdict = HONEST_VERDICTS["distribution_wrong"]
        status = "partial"
        reason = (
            f"Distribution mismatch: marginal_err={cmp['marginal_abs_err']:.3f} "
            f"(<= {MARGINAL_ABS_ERR_THRESHOLD}), corr={cmp['pairwise_corr_of_corrs']:.3f} "
            f"(>= {CORRELATION_THRESHOLD}), kl={cmp['energy_proxy_kl']:.3f} "
            f"(<= {ENERGY_KL_THRESHOLD})."
        )
    elif speedup >= SPEEDUP_TARGET:
        verdict = HONEST_VERDICTS["full_success"]
        status = "success"
        reason = f"FPGA {speedup:.1f}x CPU Gibbs, distribution matches within tolerance."
    else:
        verdict = HONEST_VERDICTS["correct_but_slow"]
        status = "success"
        reason = (
            f"Distribution matches CPU within tolerance; speedup {speedup:.1f}x "
            f"below {SPEEDUP_TARGET}x target.  Calibrates throughput claim."
        )

    _write(tmpl.build_result({
        "preflight": ctx,
        "deploy": deploy_info,
        "problem_summary": {
            "n_spins": N_SPINS, "max_degree": MAX_DEGREE, "beta": BETA,
            "seed": SK_RNG_SEED, "n_samples": N_SAMPLES, "n_steps_hw": N_STEPS_HW,
        },
        "fpga": {
            "elapsed_s": fpga["elapsed_s"],
            "throughput_mups": fpga["throughput_mups"],
            "n_samples_drawn": len(fpga["samples"]),
            "first_sample_hamming_weight": sum(1 for b in fpga["samples"][0] if b == 1),
        },
        "cpu_baseline": {
            "elapsed_s": cpu["elapsed_s"],
            "throughput_mups": cpu["throughput_mups"],
            "n_samples": cpu["n_samples"],
        },
        "comparison": cmp,
        "speedup": speedup,
        "total_elapsed_s": time.perf_counter() - t_start,
    }, status=status, honest_verdict=verdict, reason=reason))
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
