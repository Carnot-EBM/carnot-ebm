#!/usr/bin/env python3
"""Experiment 1122: KV260 Ising sampler v4 (sparse + inertia) Python simulation.

**Why this experiment exists (researcher summary):**
    exp1094 measured KL=3.07 for the v1 fully-parallel synchronous Glauber
    on a frustrated antiferromagnetic ring — about 60x the Phase-2a
    acceptance threshold (KL < 0.05). exp1109 fixed correctness by
    falling back to *sequential* single-site updates (KL ~ 0.025) but
    paid a roughly 23x slowdown vs the parallel design.

    v4 (``hardware/kv260/ising_sampler_v4.v``,
    ``hardware/kv260/ising_sampler_v4_spec.md``) is the candidate
    architecture that tries to recover the parallel-speed regime
    *without* breaking detailed balance, by adding two ingredients
    drawn from the literature:

        * **Sparse coupling** (each spin reads K neighbours, not N-1) —
          cuts the LUT budget at N=128 from ~290K (over) to ~36K (well
          within the XCK26 117K budget).
        * **Per-spin EMA inertia** (arXiv 2604.17109 "p-bit inertia") —
          smooths the local-field signal across cycles, suppressing the
          period-2 oscillation that fully-parallel synchronous Glauber
          drops into on frustrated graphs.

    This experiment validates the v4 dynamics in float64 Python *before*
    we attempt RTL synthesis and bitstream burn. The headline number is:

        kl_v4_best := min over alpha_ema in {0.1, 0.3, 0.5, 0.7} of
                       KL(v4_parallel_inertia_sim || true_Gibbs)

    on the same N=8 antiferromagnetic ring exp1094 used. Acceptance
    gate: kl_v4_best < 0.05. If the gate passes, v4 is the target
    architecture (parallel-fast AND correct). If it fails, the
    parallel + inertia hypothesis is rejected at this beta and the
    next milestone must either (a) try larger K, (b) try different
    EMA discretisation, or (c) keep the v3 sequential path for
    correctness.

**Honest verdicts:**
    - ``v4_kl_below_threshold_parallel_correct``: best alpha_ema gives
      KL < 0.05 — v4 hypothesis confirmed in float64.
    - ``v4_kl_above_threshold``: even the best alpha_ema fails the gate;
      parallel + inertia is insufficient on this J at this beta.
    - ``simulation_only_hardware_pending``: simulation succeeded but
      hardware bitstream synth+deploy was not attempted (Vivado absent
      or openXC7 cannot target Zynq UltraScale+).
    - ``partial``: simulation completed but produced a result the
      caller should treat with caution (e.g., a stochastic outlier).
    - ``failed``: the script could not produce a clean measurement.

Spec: REQ-HARDWARE-016, SCENARIO-HARDWARE-016, REQ-SAMPLE-012.
Cross-ref: results/experiment_1094_phase2a_sampler_correctness_audit.json,
           results/experiment_1109_kv260_ising_sampler_v3_sequential.json.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE = REPO_ROOT / "results" / "experiment_1122_kv260_v4_python_sim.json"
V4_SPEC_PATH = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v4_spec.md"
V4_RTL_PATH = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v4.v"
SAMPLER_SIM_PATH = REPO_ROOT / "python" / "carnot" / "hardware" / "sampler_sim.py"

EXPERIMENT_ID = 1122
TITLE = "KV260 Ising Sampler v4 — Sparse + Inertia Python Simulation"

KV260_HOST = "192.168.51.98"
KV260_USER = "ubuntu"

# Acceptance threshold mirrors exp1094 / exp1109 — Phase-2a gate.
KL_THRESHOLD = 0.05

# Same numerical setup as exp1094 / exp1109 so all three runs sit on
# the same operating point and the KL numbers are directly comparable.
N_SPINS = 8
BETA = 2.0
N_RECORD = 60_000
BURN_IN_SWEEPS = 500
RANDOM_SEED = 1122

# K=2 picks out the immediate ring neighbours, exactly recreating the
# antiferromagnetic-ring topology exp1094 used. v4 runs at K=16 in
# hardware on N=128, but the small-N validation only needs the ring
# to expose the parallel-update failure mode.
K_NEIGHBORS = 2

# alpha_ema sweep: 0.1 ≈ "almost no inertia" through 0.7 ≈ "heavy
# smoothing". The hardware spec hard-codes alpha = 0.5 (right shift),
# which is the middle of the sweep — we keep it in the list to
# verify the spec's chosen knob is in fact a good knob.
ALPHA_EMA_SWEEP = [0.1, 0.3, 0.5, 0.7]

# Baselines copied from prior experiments for the artifact (the
# experiment script does not re-run them).
KL_V3_SEQUENTIAL_BASELINE = 0.025  # exp1109
KL_V1_PARALLEL_BASELINE = 3.07  # exp1094


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_sampler_sim():
    """Import sampler_sim by file path so JAX import order does not block us.

    The sibling experiments (exp1094, exp1109) use the same trick. The
    package's ``__init__`` pulls JAX, which on this rig sometimes hits
    the ROCm-on-thrml crash — so we side-step the package and load the
    pure-NumPy module directly.
    """
    spec = importlib.util.spec_from_file_location("sampler_sim", SAMPLER_SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sampler_sim", mod)
    spec.loader.exec_module(mod)
    return mod


def _check_v4_spec() -> dict[str, Any]:
    """Sanity-check that the v4 RTL + spec exist and contain the right ingredients.

    The structural invariant we look for: the spec mentions sparse
    coupling, inertia, and synchronous parallel updates; the Verilog
    declares h_ema, sparse nbr_idx, and the sign-bit MSB rule. If any
    of those are missing the simulation is not actually modelling v4
    and the experiment must be classified as ``failed``.
    """
    info: dict[str, Any] = {
        "v4_spec_path": str(V4_SPEC_PATH.relative_to(REPO_ROOT)),
        "v4_rtl_path": str(V4_RTL_PATH.relative_to(REPO_ROOT)),
        "v4_spec_read": False,
        "v4_rtl_read": False,
    }
    if V4_SPEC_PATH.exists():
        spec_text = V4_SPEC_PATH.read_text()
        info["v4_spec_read"] = True
        info["v4_spec_mentions_sparse"] = "Sparse" in spec_text or "K=16" in spec_text
        info["v4_spec_mentions_inertia"] = "EMA" in spec_text or "inertia" in spec_text.lower()
        info["v4_spec_mentions_synchronous"] = "ynchronous" in spec_text
    if V4_RTL_PATH.exists():
        rtl_text = V4_RTL_PATH.read_text()
        info["v4_rtl_read"] = True
        info["v4_rtl_has_h_ema"] = "h_ema" in rtl_text
        info["v4_rtl_has_nbr_idx"] = "nbr_idx" in rtl_text
        info["v4_rtl_has_sign_bit_rule"] = "FIELD_WIDTH-1]" in rtl_text
    return info


def _ssh_check_board(host: str = KV260_HOST, user: str = KV260_USER) -> dict[str, Any]:
    """Probe the KV260 over SSH and report what device nodes / firmware are present.

    We deliberately keep this best-effort: any failure (no SSH key, no
    network, no DNS) collapses to ``reachable=False`` rather than
    raising, so the simulation portion of the experiment can still
    write a valid artifact in airgapped environments.
    """
    cmd = (
        "ls /dev/uio* 2>/dev/null; "
        "ls /dev/xdma* 2>/dev/null; "
        "ls /lib/firmware/xilinx 2>/dev/null; "
        "echo HOST_OK"
    )
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
                cmd,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    out = proc.stdout
    return {
        "reachable": proc.returncode == 0 and "HOST_OK" in out,
        "uio_nodes": [line for line in out.splitlines() if line.startswith("/dev/uio")],
        "xdma_nodes": [line for line in out.splitlines() if line.startswith("/dev/xdma")],
        "firmware_dirs": [
            line
            for line in out.splitlines()
            if line and not line.startswith("/") and line != "HOST_OK"
        ],
        "rc": proc.returncode,
    }


def _vivado_available() -> bool:
    """Return True iff the Xilinx Vivado synthesiser is on PATH.

    A fully-wired hardware deploy of v4 would call ``vivado -mode batch``
    on ``hardware/kv260/synth_ising.tcl``. Without Vivado the deploy
    leg is impossible; we just record this and continue with the
    Python-simulation deliverable.
    """
    return shutil.which("vivado") is not None


def _check_openxc7() -> dict[str, Any]:
    """Document whether openXC7 (open-source Xilinx synth) covers KV260.

    openXC7 (https://github.com/openxc7/openXC7) is the open-source
    Vivado alternative built on yosys + nextpnr-xilinx. As of the
    project's published scope it supports Artix-7 / Kintex-7 /
    Spartan-7 only — i.e., the 7-series families. The KV260 carries
    a Zynq UltraScale+ MPSoC (XCZU5EV), which is a *UltraScale+* part,
    not 7-series, so openXC7 cannot target it. We record this rather
    than try to install the package, because:

        * pip is not available on the research host (verified).
        * Even with pip, an install would not gain us a synthesis
          path — the device family is fundamentally outside the
          tool's scope.

    Future re-evaluation point: if openXC7 ever extends to UltraScale+
    we can revisit and synthesise v4 without Vivado.
    """
    return {
        "openxc7_zynq_supported": False,
        "openxc7_support_note": (
            "openXC7 supports Artix-7 / Kintex-7 / Spartan-7 only "
            "(7-series). KV260 = Zynq UltraScale+ MPSoC (XCZU5EV) — "
            "outside openXC7 scope. Vivado required for v4 bitstream."
        ),
        "openxc7_install_attempted": False,
        "openxc7_install_skip_reason": (
            "pip unavailable in research host environment AND device family unsupported by tool"
        ),
    }


def _attempt_load_v3_bitstream(board_state: dict[str, Any]) -> dict[str, Any]:
    """Best-effort reload of the v3-era bitstream already on the KV260.

    We do NOT push a new bitstream — that would require Vivado-built
    artifacts we cannot produce here. Instead we just observe what the
    board is currently running. The earlier ``ls /lib/firmware/xilinx``
    probe shows whether ``carnot_ising_v4`` or an older firmware is
    installed; we record that as the ``current_firmware_dirs`` field.

    Returns a dict suitable for direct merge into the artifact under
    the ``hardware_*`` namespace.
    """
    if not board_state.get("reachable", False):
        return {
            "hardware_synthesis_attempted": False,
            "hardware_load_attempt": "skipped_board_unreachable",
            "hardware_kl_measured": False,
        }
    return {
        "hardware_synthesis_attempted": False,
        "hardware_load_attempt": (
            "skipped_no_vivado_and_existing_firmware_observed"
            if board_state.get("firmware_dirs")
            else "skipped_no_vivado"
        ),
        "current_firmware_dirs": board_state.get("firmware_dirs", []),
        "hardware_kl_measured": False,
    }


def _run_alpha_sweep(sim_mod) -> dict[str, Any]:
    """Run the v4 sampler over each alpha_ema and return per-alpha KL.

    Procedure (matches exp1094 / exp1109 setup):
        1. Build the N=8 antiferromagnetic ring problem (J = -1 on each
           edge, beta = 2.0). This is the same J / beta pair on which
           v1 parallel posted KL=3.07 and v3 sequential posted KL=0.025
           — so any v4 number sits directly on that comparison line.
        2. Build the sparse K=2 ring topology table. K=2 reduces the
           sparse representation to "left and right immediate
           neighbour", which IS the underlying ring exp1094 used.
        3. For each alpha in ALPHA_EMA_SWEEP: run BURN_IN_SWEEPS warm-
           up sweeps + N_RECORD recorded sweeps in stochastic mode.
           Compute KL against the closed-form Boltzmann distribution.

    Stochastic mode is the right comparison surface because true Gibbs
    is itself a Boltzmann sampler at finite beta. Pure deterministic
    sign(h_ema) (E-MVL) does not converge to any finite-temperature
    Boltzmann; we report it separately at the spec-mandated alpha=0.5
    so the artifact captures both data points.
    """
    problem = sim_mod.antiferromagnetic_ring(n_spins=N_SPINS, beta=BETA)
    nbr_idx, j_sparse = sim_mod.SparseInertiaIsingSamplerV4.build_ring_topology(
        n_spins=N_SPINS, k=K_NEIGHBORS, j_value=-1.0
    )

    per_alpha: list[dict[str, Any]] = []
    t_total = time.time()
    for idx, alpha in enumerate(ALPHA_EMA_SWEEP):
        t0 = time.time()
        sampler = sim_mod.SparseInertiaIsingSamplerV4(
            n_spins=N_SPINS,
            k_neighbors=K_NEIGHBORS,
            alpha_ema=alpha,
            beta_temperature=BETA,
            seed=RANDOM_SEED + idx,
            mode="stochastic",
        )
        samples = sampler.sample(
            nbr_idx=nbr_idx,
            j_sparse=j_sparse,
            n_steps=N_RECORD,
            burn_in_sweeps=BURN_IN_SWEEPS,
        )
        kl = sim_mod.kl_against_true_gibbs(samples, problem)
        runtime = time.time() - t0
        per_alpha.append(
            {
                "alpha_ema": alpha,
                "kl_v4_vs_gibbs": float(kl),
                "runtime_s": round(runtime, 2),
                "below_threshold": bool(kl < KL_THRESHOLD),
            }
        )

    # Pure deterministic E-MVL pass (alpha=0.5) — recorded for spec
    # documentation, NOT used in the alpha-sweep best-of decision.
    det_t0 = time.time()
    det_sampler = sim_mod.SparseInertiaIsingSamplerV4(
        n_spins=N_SPINS,
        k_neighbors=K_NEIGHBORS,
        alpha_ema=0.5,
        beta_temperature=BETA,
        seed=RANDOM_SEED + 100,
        mode="deterministic",
    )
    # Deterministic dynamics converge fast; a few hundred sweeps is
    # ample to characterise the fixed-point distribution.
    det_samples = det_sampler.sample(
        nbr_idx=nbr_idx, j_sparse=j_sparse, n_steps=2000, burn_in_sweeps=200
    )
    det_kl = sim_mod.kl_against_true_gibbs(det_samples, problem)
    det_runtime = time.time() - det_t0

    best = min(per_alpha, key=lambda r: r["kl_v4_vs_gibbs"])
    return {
        "per_alpha": per_alpha,
        "kl_v4_best": best["kl_v4_vs_gibbs"],
        "best_alpha_ema": best["alpha_ema"],
        "deterministic_emvl": {
            "alpha_ema": 0.5,
            "kl_vs_gibbs": float(det_kl),
            "n_steps": 2000,
            "runtime_s": round(det_runtime, 2),
            "note": (
                "Pure E-MVL sign(h_ema) is not a finite-temperature "
                "Boltzmann sampler. KL is reported for completeness "
                "and is expected to be large; do not gate on it."
            ),
        },
        "alpha_sweep_runtime_s": round(time.time() - t_total, 2),
        "sim_config": {
            "n_spins": N_SPINS,
            "beta": BETA,
            "n_record": N_RECORD,
            "burn_in_sweeps": BURN_IN_SWEEPS,
            "k_neighbors": K_NEIGHBORS,
            "topology": "antiferromagnetic_ring_periodic_J=-1",
            "record_cadence": "per_sweep",
            "mode": "stochastic_pbit_glauber_on_ema_field",
        },
    }


def _classify_verdict(kl_v4_best: float | None, hardware_kl: float | None) -> str:
    """Map the measurement booleans to one of the allowed honest verdicts."""
    if kl_v4_best is None:
        return "failed"
    if kl_v4_best >= KL_THRESHOLD:
        return "v4_kl_above_threshold"
    if hardware_kl is None:
        return "simulation_only_hardware_pending"
    if hardware_kl < KL_THRESHOLD:
        return "v4_kl_below_threshold_parallel_correct"
    return "v4_kl_above_threshold"


def main() -> int:
    """Run the experiment end-to-end and write the deliverable JSON."""
    t_start = time.time()
    print(f"[exp{EXPERIMENT_ID}] starting v4 sparse-inertia Python simulation")

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": _utc_now_iso(),
        "schema": "kv260_v4_python_sim_v1",
        "kl_v4_threshold": KL_THRESHOLD,
        "kl_v3_sequential_baseline": KL_V3_SEQUENTIAL_BASELINE,
        "kl_v1_parallel_baseline": KL_V1_PARALLEL_BASELINE,
        "board_ip": KV260_HOST,
    }

    # 1. Spec / RTL presence and structural sanity.
    artifact.update(_check_v4_spec())

    # 2. Python simulation (the load-bearing deliverable).
    kl_best: float | None = None
    best_alpha: float | None = None
    try:
        sim_mod = _load_sampler_sim()
        sweep = _run_alpha_sweep(sim_mod)
        artifact.update(sweep)
        kl_best = sweep["kl_v4_best"]
        best_alpha = sweep["best_alpha_ema"]
        artifact["kv260_v4_kl_measured"] = True
        artifact["kl_v4_below_threshold"] = bool(kl_best is not None and kl_best < KL_THRESHOLD)
        print(
            f"[exp{EXPERIMENT_ID}] best alpha_ema={best_alpha} "
            f"KL(v4 || gibbs)={kl_best:.4f} (threshold {KL_THRESHOLD})"
        )
    except Exception as e:  # pragma: no cover - defensive path, exercised in tests
        artifact.update(
            {
                "sim_error": f"{type(e).__name__}: {e}",
                "kv260_v4_kl_measured": False,
                "kl_v4_best": None,
                "best_alpha_ema": None,
                "kl_v4_below_threshold": False,
            }
        )

    # 3. Hardware reachability + bitstream load (best-effort).
    board_state = _ssh_check_board()
    artifact["hardware_board_reachable"] = bool(board_state.get("reachable", False))
    artifact["hardware_board_state"] = board_state
    artifact["vivado_available"] = _vivado_available()
    artifact.update(_check_openxc7())
    artifact.update(_attempt_load_v3_bitstream(board_state))

    # 4. Verdict.
    artifact["honest_verdict"] = _classify_verdict(
        artifact.get("kl_v4_best"),
        None,  # no hardware KL available — Vivado not on PATH and openXC7 cannot target the part
    )

    artifact["duration_s"] = round(time.time() - t_start, 2)
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DELIVERABLE}")
    print(f"verdict: {artifact['honest_verdict']}")
    print(f"kl_v4_best: {artifact.get('kl_v4_best')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
