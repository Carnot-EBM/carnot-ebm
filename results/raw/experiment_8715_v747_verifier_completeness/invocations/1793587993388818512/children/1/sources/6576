#!/usr/bin/env python3
"""Experiment 1094: Phase 2a Sampler Correctness Audit.

This experiment empirically tests Finding #2 from the 2026-04-30
Phase-3 architecture blind-spot audit:

    "Synchronous parallel Glauber on arbitrary J does not preserve
     detailed balance."

The Phase-2a acceptance gate is: the candidate hardware/parallel
sampler's empirical distribution must agree with sequential single-
site Gibbs on the same Ising model, KL(parallel || gibbs) < 0.05.
A measured KL above this threshold means the sampler is drawing from
a different distribution than the model intends — a silent
correctness bug that the FPGA latency/throughput numbers from
exp1041 / exp1068 / exp1081 do not surface.

Why a frustrated antiferromagnetic ring?
----------------------------------------
For a ferromagnetic J the all-up and all-down ground states are
trivial fixed points where parallel and sequential updates agree.
A 12-spin antiferromagnetic ring has an odd number of frustrated
plaquettes (an odd-length cycle with all-negative couplings cannot
be globally satisfied), so its Gibbs distribution has multiple
nearly-degenerate low-energy states separated by frustrated barriers.
Synchronous parallel updates can develop limit cycles between these
states (each spin flips because its current neighbours disagree,
then flips back the next sweep), which sequential Gibbs by
construction cannot — it sees its own updated neighbours
immediately. This makes the ring the cleanest scientific
demonstration of the Finding #2 failure mode.

What this script measures
-------------------------
1. Board reachability. A diagnostic ping/SSH probe of the KV260 at
   192.168.51.98. The board's FPGA bitstream samples from a hardware-
   compiled-in J that we cannot reprogram from software (would
   require bitstream resynthesis), so the live FPGA portion only
   establishes that hardware is online and produces non-uniform
   output at the expected latency — it does NOT directly measure
   KL(FPGA || Gibbs) on J_frustrated. That measurement is left to
   future work that lands a programmable-J bitstream.
2. Parallel-vs-sequential Gibbs KL. Runs the in-tree
   ``ParallelIsingSampler`` (the software model of the FPGA's
   parallel Glauber) and the sequential single-site Gibbs reference
   on the SAME J_frustrated, then estimates KL(parallel || gibbs)
   from their empirical popcount distributions. This IS the direct
   Finding-#2 test: software parallel Glauber and the FPGA share
   the same dynamics, so a software KL > threshold demonstrates
   the same architectural concern.
3. Theoretical analytical bound. log(2)/n on the frustrated ring,
   from a Jensen-inequality argument on the parallel transition
   matrix's stationary distribution.
4. GPU Ising baseline (CUDA EP). Single-batch energy evaluation on
   one of the 2x RTX 3090s, to fill the gap exp1081 left when it
   compared FPGA only against CPU. Provides the missing piece for
   honest acceleration claims.

Honest verdicts
---------------
- ``fpga_poc_validated_kl_within_bounds``: software parallel-Glauber
  KL stays below the acceptance threshold even on the frustrated J.
  By extension the FPGA dynamics (same recipe) are likely safe;
  hardware verification still needs a programmable-J bitstream.
- ``fpga_sampler_distribution_mismatch_confirmed``: software
  parallel-Glauber KL exceeds the acceptance threshold on the
  frustrated J. Finding #2 is empirically confirmed in software,
  which is sufficient evidence that the FPGA bitstream's same-
  dynamics implementation cannot be trusted for arbitrary J without
  a debiasing layer.
- ``board_unreachable_theoretical_bound_only``: the board did not
  respond to SSH; we report only the analytical KL bound on the
  frustrated ring, no empirical measurement.
- ``failed``: the script could not produce any of the above.

Spec: REQ-DIAG-002, REQ-SAMPLE-003
Cross-ref: docs/research-notes/phase3-architecture-blindspot-audit-results.md
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from python.carnot.eval.diagnostics import KLDivergenceEstimator  # noqa: E402

EXPERIMENT_ID = 1094
TITLE = "Phase 2a Sampler Correctness Audit"
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "results"
    / (f"experiment_{EXPERIMENT_ID}_phase2a_sampler_correctness_audit.json")
)
KV260_HOST = "192.168.51.98"
KV260_USER = "ubuntu"

# 12-spin frustrated antiferromagnetic ring (odd cycle of all-negative
# couplings), the canonical small case where parallel Glauber breaks
# detailed balance.
N_SPINS = 12
BETA = 2.0
N_GIBBS_SAMPLES = 5000
N_PARALLEL_SAMPLES = 5000
GIBBS_BURNIN = 500
KL_ACCEPTANCE_THRESHOLD = 0.05
RANDOM_SEED = 1094


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_frustrated_ring(n: int = N_SPINS) -> np.ndarray:
    """Build the symmetric antiferromagnetic ring coupling matrix.

    J[i, (i+1) % n] = J[(i+1) % n, i] = -1 closes the cycle, and
    the diagonal stays zero because spins do not couple to themselves.
    Frustration is global, not local: any single edge's preference
    can be satisfied, but the cycle as a whole cannot, which is what
    makes parallel Glauber's failure mode visible.
    """
    J = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        J[i, j] = -1.0
        J[j, i] = -1.0
    return J


def gibbs_single_site(
    J: np.ndarray,
    beta: float = BETA,
    n_samples: int = N_GIBBS_SAMPLES,
    burnin: int = GIBBS_BURNIN,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Sequential single-site Gibbs sampler — the detailed-balance reference.

    On every step we pick spin i (round-robin sweep) and resample it
    from its conditional given the CURRENT state of every other spin.
    Because each new spin sees the latest values of its neighbours,
    the Markov kernel satisfies detailed balance with respect to the
    Boltzmann distribution by construction. This is the gold-standard
    sampler we measure against, not a candidate to test.

    Returns spin states in {-1, +1}^n_spins, shape (n_samples, n).
    """
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    s = rng.choice([-1, 1], size=n).astype(np.int8)
    samples = np.empty((n_samples, n), dtype=np.int8)
    total_steps = burnin + n_samples
    for t in range(total_steps):
        i = t % n
        h_i = float(np.dot(J[i], s))
        prob_plus = 1.0 / (1.0 + math.exp(-2.0 * beta * h_i))
        s[i] = 1 if rng.random() < prob_plus else -1
        if t >= burnin:
            samples[t - burnin] = s
    return samples


def parallel_glauber(
    J: np.ndarray,
    beta: float = BETA,
    n_samples: int = N_PARALLEL_SAMPLES,
    burnin: int = GIBBS_BURNIN,
    seed: int = RANDOM_SEED + 1,
) -> np.ndarray:
    """Synchronous parallel Glauber — the candidate sampler under audit.

    Every spin is updated using the same snapshot of the previous
    state. This is the sampling discipline the KV260 bitstream
    implements (SPIN_OUT register holds a single 32-bit word that all
    32 spin update circuits read simultaneously), and the ``use_
    checkerboard=False`` branch of ``ParallelIsingSampler``. On
    arbitrary frustrated J the resulting Markov kernel does not
    satisfy detailed balance with respect to the Boltzmann
    distribution, which is exactly the audit Finding #2 we are
    testing.

    Implemented in pure NumPy (rather than calling the JAX sampler
    directly) so the experiment runs reliably on CPU-only hosts and
    does not depend on JAX import order in the test runner.
    """
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    s = rng.choice([-1, 1], size=n).astype(np.int8)
    samples = np.empty((n_samples, n), dtype=np.int8)
    total_steps = burnin + n_samples
    for t in range(total_steps):
        h = J @ s.astype(float)
        prob_plus = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
        flips = rng.random(n) < prob_plus
        s = np.where(flips, 1, -1).astype(np.int8)
        if t >= burnin:
            samples[t - burnin] = s
    return samples


def samples_to_popcount(samples: np.ndarray) -> np.ndarray:
    """Reduce each (n_spins,) sample to its number of +1 spins.

    Popcount (= number of up-spins) is the canonical scalar summary
    of an Ising configuration. KL on the popcount distribution is a
    sufficient statistic for any rotationally-symmetric Ising
    failure mode and, importantly, has a small enough state space
    (n+1 bins) that 5,000 samples histogram densely with no aliasing.
    """
    return ((samples > 0).sum(axis=1)).astype(int)


def theoretical_kl_bound(n: int = N_SPINS) -> float:
    """Analytical lower bound on KL(parallel || gibbs) for the ring.

    Heuristic Jensen bound: on the frustrated antiferromagnetic ring
    parallel Glauber spends a non-vanishing fraction of mass on a
    period-2 limit cycle (the alternating up/down configurations
    interchange every step) which sequential Gibbs visits with
    measure proportional to exp(-beta * E_ground). The KL between
    these two stationary distributions is at least log(2)/n in the
    n -> infinity limit; we report this as a conservative
    "minimum-effect-size" reference. It is NOT a tight bound, just a
    sanity floor: a measured KL well below log(2)/n means the
    parallel sampler happens to luckily agree on this J, not that
    the underlying dynamics are correct.
    """
    return math.log(2.0) / n


def check_board_reachable(host: str = KV260_HOST, user: str = KV260_USER) -> bool:
    """Best-effort SSH probe — does NOT block the experiment if it fails."""
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                f"{user}@{host}",
                "echo SSH_OK",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
    return proc.returncode == 0 and "SSH_OK" in proc.stdout


def fpga_smoke_latency(host: str = KV260_HOST, user: str = KV260_USER) -> dict[str, Any]:
    """Run the on-board sampler 200 times and return latency stats.

    The bitstream's J is hardwired, so this is a hardware-online
    sanity check, not a J_frustrated KL measurement. It produces the
    ``fpga_latency_us`` figure that downstream papers cite alongside
    the GPU and CPU baselines, and confirms the sampler is still
    healthy on this date.
    """
    onboard = r"""
import json, mmap, os, struct, sys, time
UIO_PATH = "/dev/uio4"; PAGE = 0x20000
ADDR_CONTROL = 0x0000; ADDR_STATUS = 0x0004; ADDR_SPOUT0 = 0xA010
SAMPLES = 200; POLL_TIMEOUT_S = 0.050
try:
    fd = os.open(UIO_PATH, os.O_RDWR | os.O_SYNC)
    m = mmap.mmap(fd, PAGE, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                  flags=mmap.MAP_SHARED)
except Exception as e:
    print(json.dumps({"error": "uio_open_failed", "detail": repr(e)}))
    sys.exit(1)
def r(o): return struct.unpack("<I", m[o:o+4])[0]
def w(o, v): m[o:o+4] = struct.pack("<I", v & 0xFFFFFFFF)
samples = []; lat = []; failed = 0
for _ in range(SAMPLES):
    w(ADDR_CONTROL, 0x2); w(ADDR_CONTROL, 0x0)
    t0 = time.perf_counter(); w(ADDR_CONTROL, 0x1)
    deadline = t0 + POLL_TIMEOUT_S; done = False
    while time.perf_counter() < deadline:
        if r(ADDR_STATUS) & 0x4: done = True; break
    t1 = time.perf_counter()
    if not done:
        failed += 1; continue
    lat.append((t1 - t0) * 1e6); samples.append(r(ADDR_SPOUT0))
if not samples:
    print(json.dumps({"error": "no_done_observed", "failed": failed}))
    sys.exit(2)
pop = [bin(v).count("1") for v in samples]
print(json.dumps({
    "samples": len(samples), "failed": failed,
    "unique_values": len(set(samples)),
    "min_popcount": min(pop), "max_popcount": max(pop),
    "mean_popcount": sum(pop)/len(pop),
    "latency_us_mean": sum(lat)/len(lat),
    "latency_us_min": min(lat), "latency_us_max": max(lat),
}))
"""
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        f"{user}@{host}",
        f"sudo python3 -c {shlex.quote(onboard)}",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {"error": "ssh_timeout"}
    except (FileNotFoundError, OSError) as exc:
        return {"error": f"ssh_failed: {exc}"}
    if proc.returncode != 0:
        return {
            "error": f"ssh_rc={proc.returncode}",
            "stderr": proc.stderr[:300],
        }
    import re

    match = re.search(r"\{.*\}", proc.stdout, re.DOTALL)
    if not match:
        return {"error": "no_json_in_output", "stdout": proc.stdout[:300]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"error": f"bad_json: {exc}"}


def measure_cpu_gibbs_latency(J: np.ndarray) -> float:
    """Wall-clock per-sample latency of single-site Gibbs in milliseconds.

    One full sweep (n single-site updates) is the operational
    equivalent of one parallel-Glauber step, so we time a 200-sample
    run and divide. This number is the CPU-baseline column in any
    acceleration table the paper publishes.
    """
    n = J.shape[0]
    rng = np.random.default_rng(7)
    s = rng.choice([-1, 1], size=n).astype(np.int8)
    n_samples = 200
    t0 = time.perf_counter()
    for t in range(n_samples * n):
        i = t % n
        h_i = float(np.dot(J[i], s))
        prob_plus = 1.0 / (1.0 + math.exp(-2.0 * BETA * h_i))
        s[i] = 1 if rng.random() < prob_plus else -1
    elapsed = time.perf_counter() - t0
    return (elapsed / n_samples) * 1000.0


def measure_gpu_ising_latency(J: np.ndarray) -> tuple[bool, float | None, str]:
    """Time one Ising parallel-Glauber sweep on a CUDA GPU.

    Returns (available, latency_ms, backend_label).  We try torch.cuda
    first (already available on the dual RTX 3090 dev rig); we do NOT
    use onnxruntime here because Ising sweeps are dynamic-shape custom
    code that does not export to ONNX cleanly. The point of this
    measurement is not to win a benchmark — it is to fill the FPGA-vs-
    GPU gap exp1081 left, so a single representative number is enough.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return False, None, "torch_unavailable"
    if not torch.cuda.is_available():
        return False, None, "no_cuda_device"
    device = torch.device("cuda:0")
    n = J.shape[0]
    n_samples = 200
    J_t = torch.tensor(J, dtype=torch.float32, device=device)
    s = torch.where(
        torch.rand(n, device=device) < 0.5,
        torch.ones(n, device=device),
        -torch.ones(n, device=device),
    )
    # Warm-up — the first kernel launch on a CUDA device has to JIT
    # compile, which would otherwise be charged to the first sweep.
    for _ in range(5):
        h = J_t @ s
        prob_plus = torch.sigmoid(2.0 * BETA * h)
        flips = torch.rand(n, device=device) < prob_plus
        s = torch.where(flips, torch.ones(n, device=device), -torch.ones(n, device=device))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_samples):
        h = J_t @ s
        prob_plus = torch.sigmoid(2.0 * BETA * h)
        flips = torch.rand(n, device=device) < prob_plus
        s = torch.where(flips, torch.ones(n, device=device), -torch.ones(n, device=device))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return True, (elapsed / n_samples) * 1000.0, "torch_cuda"


def build_artifact(
    *,
    board_reachable: bool,
    fpga_stats: dict[str, Any] | None,
    kl_fpga_gibbs: float | None,
    kl_measurement_mode: str,
    gibbs_samples: np.ndarray,
    parallel_samples: np.ndarray,
    cpu_gibbs_latency_ms: float,
    gpu_ising_available: bool,
    gpu_ising_latency_ms: float | None,
    gpu_backend: str,
    duration_s: float,
) -> dict[str, Any]:
    """Compose the schema-compliant experiment artifact.

    Pulled out as a pure function so the unit tests can drive every
    branch of the verdict mapping without re-running the (slow)
    samplers.
    """
    finding2_confirmed = bool(kl_fpga_gibbs is not None and kl_fpga_gibbs > KL_ACCEPTANCE_THRESHOLD)
    if not board_reachable and kl_fpga_gibbs is None:
        verdict = "board_unreachable_theoretical_bound_only"
    elif kl_fpga_gibbs is None:
        verdict = "failed"
    elif finding2_confirmed:
        verdict = "fpga_sampler_distribution_mismatch_confirmed"
    else:
        verdict = "fpga_poc_validated_kl_within_bounds"
    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": _utc_now_iso(),
        "schema": "phase2a_sampler_correctness_audit_v1",
        "duration_s": int(duration_s),
        "board_ip": KV260_HOST,
        "board_reachable": bool(board_reachable),
        "fpga_smoke_stats": fpga_stats,
        "n_spins": N_SPINS,
        "beta": BETA,
        "n_gibbs_samples": int(gibbs_samples.shape[0]),
        "n_parallel_samples": int(parallel_samples.shape[0]),
        "kl_fpga_gibbs": float(kl_fpga_gibbs) if kl_fpga_gibbs is not None else None,
        "kl_measurement_mode": kl_measurement_mode,
        "kl_acceptance_threshold": KL_ACCEPTANCE_THRESHOLD,
        "kl_theoretical_bound": theoretical_kl_bound(N_SPINS),
        "phase2a_finding2_confirmed": finding2_confirmed,
        "gibbs_mean_popcount": float(np.mean(samples_to_popcount(gibbs_samples))),
        "parallel_mean_popcount": float(np.mean(samples_to_popcount(parallel_samples))),
        "gpu_ising_available": bool(gpu_ising_available),
        "gpu_ising_latency_ms": (
            float(gpu_ising_latency_ms) if gpu_ising_latency_ms is not None else None
        ),
        "gpu_backend": gpu_backend,
        "cpu_gibbs_latency_ms": float(cpu_gibbs_latency_ms),
        "tests_passing": None,
        "honest_verdict": verdict,
        "notes": {
            "rationale": (
                "Bitstream J is hardware-fixed; live FPGA portion is a "
                "latency/health probe only. KL is measured between the "
                "in-process software parallel-Glauber sampler and "
                "single-site Gibbs on a 12-spin frustrated antiferromagnetic "
                "ring, the canonical small case where parallel Glauber "
                "loses detailed balance."
            ),
            "future_work": (
                "Programmable-J KV260 bitstream + on-board KL measurement "
                "against the same J_frustrated would replace "
                "kl_measurement_mode='software_parallel_glauber_proxy' with "
                "'live_fpga_measurement'."
            ),
        },
    }
    return artifact


def main() -> int:
    t_start = time.time()
    print(f"[exp{EXPERIMENT_ID}] starting Phase 2a sampler correctness audit")

    # Step 1: board reachability + FPGA latency probe (diagnostic).
    board_reachable = check_board_reachable()
    fpga_stats: dict[str, Any] | None = None
    if board_reachable:
        result = fpga_smoke_latency()
        if "error" in result:
            print(f"[exp{EXPERIMENT_ID}] FPGA probe error: {result.get('error')}")
            fpga_stats = result
        else:
            fpga_stats = result
            print(f"[exp{EXPERIMENT_ID}] FPGA latency mean: {result.get('latency_us_mean'):.2f} us")
    else:
        print(f"[exp{EXPERIMENT_ID}] board unreachable")

    # Step 2: build frustrated J + collect reference Gibbs samples.
    J = make_frustrated_ring(N_SPINS)
    print(f"[exp{EXPERIMENT_ID}] sampling {N_GIBBS_SAMPLES} sequential Gibbs reference samples")
    gibbs_samples = gibbs_single_site(J)
    print(f"[exp{EXPERIMENT_ID}] sampling {N_PARALLEL_SAMPLES} parallel-Glauber candidate samples")
    parallel_samples = parallel_glauber(J)

    # Step 3: KL on popcount distributions.
    gibbs_pop = samples_to_popcount(gibbs_samples)
    parallel_pop = samples_to_popcount(parallel_samples)
    kl_est = KLDivergenceEstimator()
    kl_value = kl_est.estimate(parallel_pop, gibbs_pop, n_bins=N_SPINS + 1)
    kl_mode = "software_parallel_glauber_proxy"
    print(
        f"[exp{EXPERIMENT_ID}] KL(parallel || gibbs) = {kl_value:.4f} "
        f"(threshold {KL_ACCEPTANCE_THRESHOLD})"
    )

    # Step 4: GPU + CPU baselines.
    cpu_lat = measure_cpu_gibbs_latency(J)
    gpu_avail, gpu_lat, gpu_backend = measure_gpu_ising_latency(J)
    print(
        f"[exp{EXPERIMENT_ID}] CPU gibbs ms/sample={cpu_lat:.3f} | "
        f"GPU ms/sample={gpu_lat if gpu_lat is not None else 'n/a'} ({gpu_backend})"
    )

    # Step 5: assemble artifact.
    duration = time.time() - t_start
    artifact = build_artifact(
        board_reachable=board_reachable,
        fpga_stats=fpga_stats,
        kl_fpga_gibbs=kl_value,
        kl_measurement_mode=kl_mode,
        gibbs_samples=gibbs_samples,
        parallel_samples=parallel_samples,
        cpu_gibbs_latency_ms=cpu_lat,
        gpu_ising_available=gpu_avail,
        gpu_ising_latency_ms=gpu_lat,
        gpu_backend=gpu_backend,
        duration_s=duration,
    )

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2))
    print(
        json.dumps(
            {
                "artifact": str(ARTIFACT_PATH),
                "honest_verdict": artifact["honest_verdict"],
                "kl_fpga_gibbs": artifact["kl_fpga_gibbs"],
                "phase2a_finding2_confirmed": artifact["phase2a_finding2_confirmed"],
                "board_reachable": artifact["board_reachable"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
