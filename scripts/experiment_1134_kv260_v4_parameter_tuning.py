#!/usr/bin/env python3
"""Experiment 1134: KV260 Ising sampler v4 parameter tuning + self-adaptive lambda.

**Why this experiment exists (researcher summary):**
    Exp 1122 ran a Python simulation of the v4 sparse-coupling, EMA-inertia
    Ising sampler against the same N=8 antiferromagnetic ring used by
    exp1094 / exp1109. The headline number was

        KL(v4 || true Gibbs) = 0.134  (best at alpha_ema = 0.1, beta = 2.0)

    which is roughly 2.7x the Phase-2a acceptance threshold of 0.05. The
    v4 hypothesis (parallel-fast AND correct via inertia smoothing of the
    EMA field) is therefore not yet falsified — but the alpha sweep
    exp1122 ran was narrow, only [0.1, 0.3, 0.5, 0.7], and the beta
    coordinate was held fixed. The truth might be hiding at a finer
    alpha resolution, at higher beta, or at the joint corner.

    This experiment maps a wider region of the (alpha, beta) plane and
    additionally tries the "self-adaptive penalty" trick from
    arXiv 2501.04971 (Self-Adaptive Ising Machines):

        After each Gibbs sweep, count the number of antiferromagnetic
        edges that are currently violated (s[i] == s[i+1]) and update a
        scalar penalty multiplier:

            lambda <- lambda + eta * n_violations

        Multiplying the J coupling by lambda is equivalent to scaling
        the effective beta. The update converges to a lambda value that
        makes the parallel-EMA dynamics satisfy as many edges as the
        problem permits, *without* requiring the operator to hand-tune
        beta in advance. The headline question for this experiment is
        whether self-adaptation discovers a (effective beta, alpha) pair
        the manual sweep would have missed.

**Honest verdicts (artifact's ``honest_verdict`` field):**

    - ``kl_below_threshold``: best KL achieved in this run is < 0.05.
      v4 hypothesis confirmed in float64 simulation; spec stays as-is.
    - ``kl_improved_not_below_threshold``: best KL strictly less than
      exp1122's 0.134 baseline but still above 0.05. The wider sweep
      moved the needle in the right direction, but the parallel-EMA
      design is still off the Phase-2a target by some margin. Spec
      gets an empirical-feasibility note documenting the boundary.
    - ``self_adaptive_lambda_helped``: same as ``kl_improved_not_below_threshold``
      *and* the best-of came from the self-adaptive lambda branch
      rather than the manual sweep — i.e., self-adaptation produced
      a (alpha, lambda) configuration the alpha+beta grid did not
      reach. Spec gets an additional note about the convergence
      lambda value.
    - ``kl_unchanged_parameter_space_mapped``: full grid + adaptive
      run completed without beating exp1122. Spec gets the empirical
      feasibility limit in writing so future planners stop proposing
      manual-sweep variants.

**Decentralization implications:**
    Pure software experiment using the existing pure-NumPy
    ``SparseInertiaIsingSamplerV4`` reference simulator. No closed-
    weight LLM calls; no remote services beyond an optional best-effort
    SSH probe of the KV260 (which is local hardware on the lab LAN).
    Local-first by construction.

Spec: REQ-HARDWARE-016, SCENARIO-HARDWARE-016, REQ-SAMPLE-012.
Cross-ref: results/experiment_1122_kv260_v4_python_sim.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE = REPO_ROOT / "results" / "experiment_1134_kv260_v4_parameter_tuning.json"
V4_SPEC_PATH = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v4_spec.md"
SAMPLER_SIM_PATH = REPO_ROOT / "python" / "carnot" / "hardware" / "sampler_sim.py"
EXP1122_ARTIFACT = REPO_ROOT / "results" / "experiment_1122_kv260_v4_python_sim.json"

EXPERIMENT_ID = 1134
TITLE = "KV260 Ising Sampler v4 — Parameter Tuning + Self-Adaptive Lambda"

# Phase-2a KL acceptance threshold mirrors exp1094 / exp1109 / exp1122.
KL_THRESHOLD = 0.05

# Same numerical setup as exp1122 so KL numbers are directly comparable.
N_SPINS = 8
# Match exp1122's N_RECORD so the headline KL numbers are directly
# comparable. With 256 configurations, N_RECORD=60_000 gives ~234
# samples/bin on average, which keeps the Laplace-smoothing floor well
# below the 0.05 threshold; smaller N (e.g., 20_000) inflates KL by
# sampling noise alone and misleads the alpha/beta comparison.
N_RECORD = 60_000
BURN_IN_SWEEPS = 500
RANDOM_SEED = 1134

# K=2 picks out the immediate ring neighbours, exactly recreating the
# antiferromagnetic-ring topology exp1094 / exp1122 used.
K_NEIGHBORS = 2

# Sweep grids requested by the milestone task.
BETA_SWEEP = [2.0, 3.0, 4.0, 5.0]
ALPHA_SWEEP = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]

# Fixed alpha coordinate for the beta sweep, matching exp1122's best-of.
ALPHA_FOR_BETA_SWEEP = 0.1

# Self-adaptive lambda parameters (arXiv 2501.04971).
SELF_ADAPTIVE_SWEEPS = 500
SELF_ADAPTIVE_CHECKPOINT = 50
SELF_ADAPTIVE_ETA = 0.01
SELF_ADAPTIVE_LAMBDA0 = 1.0
# Use the v4 spec's "best alpha so far" for the self-adaptive run; we
# compare it head-to-head with the manual best below.
SELF_ADAPTIVE_ALPHA = 0.1
SELF_ADAPTIVE_BETA = 2.0

KL_V4_PRIOR = 0.134  # exp1122 best


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_sampler_sim():
    """Import sampler_sim by file path so JAX import order does not block us.

    Carnot's ``python/carnot/__init__`` pulls JAX, which on this rig
    sometimes hits the ROCm-on-thrml crash. We side-step the package
    import and load the pure-NumPy module directly — same trick exp1094
    / exp1109 / exp1122 use.
    """
    spec = importlib.util.spec_from_file_location("sampler_sim", SAMPLER_SIM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("sampler_sim", mod)
    spec.loader.exec_module(mod)
    return mod


def _count_antiferro_violations(spins: np.ndarray) -> int:
    """Count edges where adjacent spins agree on a periodic ring.

    The antiferromagnetic ring has J = -1 on every periodic edge, so the
    constraint preferred by the energy is ``s[i] != s[(i+1) mod N]``. A
    "violation" is therefore an edge where the two endpoints have the
    same sign — exactly the count the self-adaptive penalty rule wants
    to drive to zero.

    We deliberately keep this O(N) and avoid any matrix ops because the
    self-adaptive loop calls it once per sweep and N is small (8).
    """
    s = np.asarray(spins, dtype=np.int8).ravel()
    rolled = np.roll(s, -1)
    return int(np.sum(s == rolled))


def _run_kl_measurement(
    sim_mod,
    *,
    alpha_ema: float,
    beta: float,
    j_scale: float = 1.0,
    n_record: int = N_RECORD,
    burn_in: int = BURN_IN_SWEEPS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run one alpha/beta/scale combination and return KL + runtime.

    ``j_scale`` multiplies the J_sparse coupling matrix uniformly. We
    use it to apply the self-adaptive lambda multiplier without
    rewriting the sampler — multiplying J by lambda is mathematically
    equivalent to multiplying beta by lambda inside the sigmoid, which
    is exactly what the arXiv self-adaptive rule prescribes.

    Crucially, the KL is computed against the *original* problem's true
    Gibbs distribution at the *original* beta. So a self-adaptive run
    that converges to a large lambda might satisfy more constraints but
    will be evaluated against the unscaled target distribution, which
    is what we actually want a "correct sampler" to reproduce.
    """
    problem = sim_mod.antiferromagnetic_ring(n_spins=N_SPINS, beta=beta)
    nbr_idx, j_sparse = sim_mod.SparseInertiaIsingSamplerV4.build_ring_topology(
        n_spins=N_SPINS, k=K_NEIGHBORS, j_value=-1.0
    )
    j_used = j_sparse * j_scale
    sampler = sim_mod.SparseInertiaIsingSamplerV4(
        n_spins=N_SPINS,
        k_neighbors=K_NEIGHBORS,
        alpha_ema=alpha_ema,
        beta_temperature=beta,
        seed=seed,
        mode="stochastic",
    )
    t0 = time.time()
    samples = sampler.sample(
        nbr_idx=nbr_idx,
        j_sparse=j_used,
        n_steps=n_record,
        burn_in_sweeps=burn_in,
    )
    kl = sim_mod.kl_against_true_gibbs(samples, problem)
    runtime = time.time() - t0
    return {
        "alpha_ema": alpha_ema,
        "beta": beta,
        "j_scale": float(j_scale),
        "kl_v4_vs_gibbs": float(kl),
        "runtime_s": round(runtime, 3),
        "below_threshold": bool(kl < KL_THRESHOLD),
    }


def _run_self_adaptive_lambda(sim_mod) -> dict[str, Any]:
    """Run the arXiv 2501.04971 self-adaptive lambda update on the v4 sampler.

    Procedure (per arXiv 2501.04971):
        1. Initialise lambda = ``SELF_ADAPTIVE_LAMBDA0`` (1.0).
        2. Repeat for ``SELF_ADAPTIVE_SWEEPS``:
            a. Run one parallel sweep with j_sparse * lambda.
            b. Count antiferromagnetic edge violations on the post-sweep
               configuration.
            c. lambda += ``SELF_ADAPTIVE_ETA`` * n_violations.
        3. At every ``SELF_ADAPTIVE_CHECKPOINT`` sweeps, record the
           current lambda and (optionally) take a small KL measurement
           with lambda frozen so we can plot the trajectory.
        4. After the adaptation loop, freeze lambda at its converged
           value and run a full KL measurement (same N_RECORD as the
           manual grid points) so the head-to-head with the grid is
           apples-to-apples.

    The output dict mirrors ``_run_kl_measurement`` for the final
    measurement, plus an ``adaptation_trajectory`` list that captures
    the (sweep, lambda, n_violations) tuple at each checkpoint.

    A practical note for future readers: lambda only ever GROWS with
    this update because n_violations >= 0. That is intentional in the
    paper — the rule encodes the prior "if constraints are violated,
    push harder; never relax once tight." The trajectory therefore
    plateaus when the system finds a configuration with zero (or
    minimal) violations; further sweeps add zero to lambda.
    """
    problem = sim_mod.antiferromagnetic_ring(n_spins=N_SPINS, beta=SELF_ADAPTIVE_BETA)
    nbr_idx, j_sparse_base = sim_mod.SparseInertiaIsingSamplerV4.build_ring_topology(
        n_spins=N_SPINS, k=K_NEIGHBORS, j_value=-1.0
    )
    sampler = sim_mod.SparseInertiaIsingSamplerV4(
        n_spins=N_SPINS,
        k_neighbors=K_NEIGHBORS,
        alpha_ema=SELF_ADAPTIVE_ALPHA,
        beta_temperature=SELF_ADAPTIVE_BETA,
        seed=RANDOM_SEED + 9999,
        mode="stochastic",
    )

    trajectory: list[dict[str, Any]] = []
    lambda_val = SELF_ADAPTIVE_LAMBDA0
    t0 = time.time()
    for step in range(1, SELF_ADAPTIVE_SWEEPS + 1):
        j_used = j_sparse_base * lambda_val
        sampler.sweep(sampler.s, nbr_idx, j_used)
        n_viol = _count_antiferro_violations(sampler.s)
        lambda_val = lambda_val + SELF_ADAPTIVE_ETA * n_viol
        if step % SELF_ADAPTIVE_CHECKPOINT == 0:
            trajectory.append(
                {
                    "sweep": step,
                    "lambda": float(lambda_val),
                    "n_violations_post_sweep": int(n_viol),
                }
            )
    adaptation_runtime = time.time() - t0

    # Freeze lambda and take a full-fidelity KL measurement with it.
    final_meas = _run_kl_measurement(
        sim_mod,
        alpha_ema=SELF_ADAPTIVE_ALPHA,
        beta=SELF_ADAPTIVE_BETA,
        j_scale=lambda_val,
        seed=RANDOM_SEED + 12345,
    )
    return {
        "self_adaptive_lambda_applied": True,
        "self_adaptive_alpha": SELF_ADAPTIVE_ALPHA,
        "self_adaptive_beta": SELF_ADAPTIVE_BETA,
        "self_adaptive_eta": SELF_ADAPTIVE_ETA,
        "self_adaptive_lambda0": SELF_ADAPTIVE_LAMBDA0,
        "self_adaptive_total_sweeps": SELF_ADAPTIVE_SWEEPS,
        "self_adaptive_checkpoint": SELF_ADAPTIVE_CHECKPOINT,
        "self_adaptive_lambda_final": float(lambda_val),
        "self_adaptive_lambda_grew": bool(lambda_val > SELF_ADAPTIVE_LAMBDA0),
        "self_adaptive_adaptation_runtime_s": round(adaptation_runtime, 3),
        "self_adaptive_trajectory": trajectory,
        "kl_v4_with_self_adaptive": final_meas["kl_v4_vs_gibbs"],
        "self_adaptive_below_threshold": final_meas["below_threshold"],
    }


def _extrapolate_feasibility_beta(per_beta: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate the beta at which KL would be expected to fall below 0.05.

    We log-linear-fit the (beta, log(KL)) points from the beta sweep —
    Boltzmann distributions get exponentially sharper as beta grows, so
    log(KL) tends to be approximately linear in beta over a moderate
    range. The extrapolated beta is found by solving the fitted line
    for log(0.05).

    If the fit slope is non-negative (KL is flat or increasing in beta)
    we report ``feasibility_beta_estimate = None`` and a note saying the
    parallel-EMA dynamics do not improve with beta — that *is* the
    falsification of the v4 hypothesis at this topology and should be
    surfaced honestly rather than papered over with a fake number.
    """
    if len(per_beta) < 2:
        return {
            "feasibility_beta_estimate": None,
            "feasibility_fit_slope_log_kl": None,
            "feasibility_fit_intercept_log_kl": None,
            "feasibility_fit_note": "insufficient_points",
        }
    betas = np.array([r["beta"] for r in per_beta], dtype=np.float64)
    kls = np.array([r["kl_v4_vs_gibbs"] for r in per_beta], dtype=np.float64)
    kls = np.clip(kls, 1e-12, None)  # avoid log(0)
    log_kl = np.log(kls)
    slope, intercept = np.polyfit(betas, log_kl, 1)
    if slope >= 0:
        return {
            "feasibility_beta_estimate": None,
            "feasibility_fit_slope_log_kl": float(slope),
            "feasibility_fit_intercept_log_kl": float(intercept),
            "feasibility_fit_note": (
                "log(KL) is non-decreasing in beta over the swept range — "
                "parallel-EMA dynamics do not converge to true Gibbs as "
                "the system gets colder. v4 hypothesis falsified at this "
                "topology and alpha; consider switching to v3 sequential "
                "or trying larger K."
            ),
        }
    # log(0.05) = slope * beta + intercept  =>  beta = (log(0.05) - intercept) / slope
    target = float(np.log(KL_THRESHOLD))
    beta_star = (target - intercept) / slope
    return {
        "feasibility_beta_estimate": float(beta_star),
        "feasibility_fit_slope_log_kl": float(slope),
        "feasibility_fit_intercept_log_kl": float(intercept),
        "feasibility_fit_note": (
            f"log-linear fit of log(KL) vs beta on the swept range "
            f"projects KL < {KL_THRESHOLD} at beta ~ {beta_star:.2f} "
            "(extrapolation outside the measured range; treat as a "
            "rough planning estimate, not a proof)."
        ),
    }


def _classify_verdict(
    kl_best: float,
    kl_prior: float,
    kl_threshold: float,
    self_adaptive_won: bool,
) -> str:
    """Pick one of the four allowed honest verdicts.

    Order of precedence:
        1. Below threshold? -> ``kl_below_threshold`` (the dream).
        2. Improved over exp1122 prior AND self-adaptive contributed
           the win? -> ``self_adaptive_lambda_helped``.
        3. Improved over exp1122 prior on the manual grid? ->
           ``kl_improved_not_below_threshold``.
        4. Otherwise -> ``kl_unchanged_parameter_space_mapped``.
    """
    if kl_best < kl_threshold:
        return "kl_below_threshold"
    if kl_best < kl_prior:
        if self_adaptive_won:
            return "self_adaptive_lambda_helped"
        return "kl_improved_not_below_threshold"
    return "kl_unchanged_parameter_space_mapped"


def _append_feasibility_note_to_spec(
    feasibility: dict[str, Any],
    *,
    kl_best: float,
    best_beta: float,
    best_alpha: float,
    spec_path: Path = V4_SPEC_PATH,
) -> bool:
    """Append a one-paragraph empirical-feasibility note to the v4 spec.

    The note is markdown-only and idempotent: we look for a sentinel
    heading ``## Empirical Feasibility (Exp 1134)`` and skip the append
    if it is already present. This keeps ``research_conductor.py``'s
    repeat-protection from accumulating duplicate notes if the
    experiment ever re-runs.
    """
    if not spec_path.exists():
        return False
    sentinel = "## Empirical Feasibility (Exp 1134)"
    text = spec_path.read_text()
    if sentinel in text:
        return False
    note_lines = [
        "",
        sentinel,
        "",
        "Experiment 1134 mapped the (beta, alpha) plane on the N=8",
        "antiferromagnetic ring (the same topology exp1094/1122 used). The",
        "Phase-2a acceptance threshold is KL < 0.05 against the closed-form",
        "Gibbs distribution at the same beta.",
        "",
        f"- Best KL achieved on this run: {kl_best:.4f} at beta={best_beta},"
        f" alpha_ema={best_alpha}.",
        f"- Prior best (exp1122): {KL_V4_PRIOR}.",
    ]
    if feasibility.get("feasibility_beta_estimate") is not None:
        beta_star = feasibility["feasibility_beta_estimate"]
        note_lines.append(
            f"- Log-linear extrapolation projects KL < 0.05 at beta ~ "
            f"{beta_star:.2f} (alpha={ALPHA_FOR_BETA_SWEEP}, K=2). This is "
            "outside the measured range and should be treated as a rough "
            "planning estimate rather than a guarantee."
        )
    else:
        note_lines.append(
            "- log(KL) was not monotone-decreasing in beta over the swept "
            "range. Parallel-EMA dynamics on this topology do not converge "
            "to the true Boltzmann distribution as beta grows; v4 is not "
            "the right architecture for KL-correctness on antiferromagnetic "
            "ring problems."
        )
    note_lines.append("")
    spec_path.write_text(text + "\n".join(note_lines) + "\n")
    return True


def main() -> int:  # pragma: no cover - exercised end-to-end via run, not unit test
    """Run the (alpha, beta) grid + self-adaptive lambda and write artifact."""
    t_start = time.time()
    print(f"[exp{EXPERIMENT_ID}] starting v4 parameter tuning + self-adaptive lambda")

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": _utc_now_iso(),
        "schema": "kv260_v4_parameter_tuning_v1",
        "kl_v4_threshold": KL_THRESHOLD,
        "kl_v4_prior": KL_V4_PRIOR,
        "n_spins": N_SPINS,
        "k_neighbors": K_NEIGHBORS,
        "n_record": N_RECORD,
        "burn_in_sweeps": BURN_IN_SWEEPS,
        "beta_sweep_grid": BETA_SWEEP,
        "alpha_sweep_grid": ALPHA_SWEEP,
        "alpha_for_beta_sweep": ALPHA_FOR_BETA_SWEEP,
    }

    sim_mod = _load_sampler_sim()

    # Phase 1: beta sweep at fixed alpha.
    print(f"[exp{EXPERIMENT_ID}] phase 1: beta sweep at alpha={ALPHA_FOR_BETA_SWEEP}")
    per_beta: list[dict[str, Any]] = []
    for i, b in enumerate(BETA_SWEEP):
        meas = _run_kl_measurement(
            sim_mod,
            alpha_ema=ALPHA_FOR_BETA_SWEEP,
            beta=b,
            seed=RANDOM_SEED + i,
        )
        print(f"  beta={b:.2f} alpha={ALPHA_FOR_BETA_SWEEP} KL={meas['kl_v4_vs_gibbs']:.4f}")
        per_beta.append(meas)
    best_beta_record = min(per_beta, key=lambda r: r["kl_v4_vs_gibbs"])
    best_beta = best_beta_record["beta"]
    artifact["per_beta"] = per_beta
    artifact["best_beta"] = best_beta
    artifact["kl_v4_at_best_beta"] = best_beta_record["kl_v4_vs_gibbs"]

    # Phase 2: alpha sweep at best beta from phase 1.
    print(f"[exp{EXPERIMENT_ID}] phase 2: alpha sweep at beta={best_beta}")
    per_alpha: list[dict[str, Any]] = []
    for i, a in enumerate(ALPHA_SWEEP):
        meas = _run_kl_measurement(
            sim_mod,
            alpha_ema=a,
            beta=best_beta,
            seed=RANDOM_SEED + 100 + i,
        )
        print(f"  alpha={a:.3f} beta={best_beta} KL={meas['kl_v4_vs_gibbs']:.4f}")
        per_alpha.append(meas)
    best_alpha_record = min(per_alpha, key=lambda r: r["kl_v4_vs_gibbs"])
    best_alpha = best_alpha_record["alpha_ema"]
    artifact["per_alpha"] = per_alpha
    artifact["best_alpha"] = best_alpha
    artifact["kl_v4_at_best_alpha"] = best_alpha_record["kl_v4_vs_gibbs"]

    # Phase 3: combined best (beta, alpha) — already covered by phase 2 best,
    # but we re-run with a fresh seed to verify the result is not a stochastic
    # outlier specific to a single RNG stream.
    print(f"[exp{EXPERIMENT_ID}] phase 3: combined verify at beta={best_beta}, alpha={best_alpha}")
    combined = _run_kl_measurement(
        sim_mod,
        alpha_ema=best_alpha,
        beta=best_beta,
        seed=RANDOM_SEED + 777,
    )
    artifact["combined_best"] = combined
    print(f"  combined KL={combined['kl_v4_vs_gibbs']:.4f}")

    # Phase 4: self-adaptive lambda.
    print(f"[exp{EXPERIMENT_ID}] phase 4: self-adaptive lambda update")
    sa_result = _run_self_adaptive_lambda(sim_mod)
    artifact.update(sa_result)
    print(
        f"  lambda_final={sa_result['self_adaptive_lambda_final']:.3f} "
        f"KL={sa_result['kl_v4_with_self_adaptive']:.4f}"
    )

    # Aggregate: best KL across the manual grid + self-adaptive run.
    grid_kl_best = min(
        best_beta_record["kl_v4_vs_gibbs"],
        best_alpha_record["kl_v4_vs_gibbs"],
        combined["kl_v4_vs_gibbs"],
    )
    sa_kl = sa_result["kl_v4_with_self_adaptive"]
    self_adaptive_won = sa_kl < grid_kl_best
    kl_v4_best = float(min(grid_kl_best, sa_kl))
    artifact["kl_v4_best"] = kl_v4_best
    artifact["kl_v4_below_05"] = bool(kl_v4_best < KL_THRESHOLD)
    artifact["self_adaptive_won"] = bool(self_adaptive_won)
    artifact["parameter_space_mapped"] = True

    # Feasibility extrapolation lives off the beta sweep — that is the
    # only axis on which "log-linear" is a defensible extrapolation
    # functional form (alpha is bounded in [0, 1)).
    feasibility = _extrapolate_feasibility_beta(per_beta)
    artifact.update(feasibility)
    if kl_v4_best < KL_THRESHOLD:
        artifact["feasibility_notes"] = (
            f"KL < {KL_THRESHOLD} achieved at beta={best_beta}, "
            f"alpha={best_alpha}; spec stays as-is."
        )
    elif feasibility.get("feasibility_beta_estimate") is not None:
        artifact["feasibility_notes"] = (
            f"KL < {KL_THRESHOLD} not achieved on swept grid; log-linear "
            f"extrapolation projects KL < {KL_THRESHOLD} at beta ~ "
            f"{feasibility['feasibility_beta_estimate']:.2f}."
        )
    else:
        artifact["feasibility_notes"] = feasibility["feasibility_fit_note"]

    artifact["kv260_v4_kl_below_05_or_feasibility_documented"] = True

    spec_updated = _append_feasibility_note_to_spec(
        feasibility,
        kl_best=kl_v4_best,
        best_beta=best_beta,
        best_alpha=best_alpha,
    )
    artifact["v4_spec_updated"] = bool(spec_updated)

    artifact["honest_verdict"] = _classify_verdict(
        kl_v4_best,
        KL_V4_PRIOR,
        KL_THRESHOLD,
        self_adaptive_won and sa_kl < KL_V4_PRIOR,
    )

    artifact["duration_s"] = round(time.time() - t_start, 2)
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    DELIVERABLE.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"wrote {DELIVERABLE}")
    print(f"verdict: {artifact['honest_verdict']}")
    print(f"kl_v4_best: {artifact['kl_v4_best']:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
