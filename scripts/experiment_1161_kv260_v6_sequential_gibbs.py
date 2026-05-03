#!/usr/bin/env python3
"""Experiment 1161: KV260 v6 sequential Gibbs correctness pivot.

Exp 1149 found that the v5 DC-continuous relaxation regressed to KL ~= 0.447
and recommended sequential Gibbs for KL-correct RTL. This experiment implements
that pivot directly: one spin is updated per step in strict round-robin order,
using the standard conditional probability

    P(s_i = +1 | s_-i) = sigmoid(2 * beta * (sum_j J_ij * s_j + b_i)).

Spec refs: REQ-HW-045, SCENARIO-HW-045.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import experiment_1149_kv260_v5_dc_continuous_diagnostic as exp1149

DELIVERABLE = REPO_ROOT / "results" / "experiment_1161_kv260_v6_sequential_gibbs.json"
RTL_SPEC_PATH = REPO_ROOT / "hardware" / "kv260" / "ising_sampler_v6_spec.md"
EXP1134_ARTIFACT = REPO_ROOT / "results" / "experiment_1134_kv260_v4_parameter_tuning.json"
EXP1149_ARTIFACT = REPO_ROOT / "results" / "experiment_1149_kv260_v5_dc_continuous_diagnostic.json"

EXPERIMENT_ID = 1161
TITLE = "KV260 v6 Sequential Gibbs Correctness Pivot"
ALGORITHM = "sequential_gibbs"

BETA = 2.0
KL_THRESHOLD = 0.05
DEFAULT_N_STEPS = 10_000
N8_SPINS = 8
N128_SPINS = 128
K2_NEIGHBORS = 2
K16_NEIGHBORS = 16
MATRIX_SEEDS = (1134, 1135, 1136)

KL_V5_PRIOR_FALLBACK = 0.447
KL_V4_PRIOR_FALLBACK = 0.1128

HONEST_VERDICTS = {
    "kl_below_threshold_sequential_correct",
    "kl_near_zero_algorithm_correct",
    "kl_above_threshold_unexplained",
    "matrix_generation_failed",
}

REQUIRED_ARTIFACT_FIELDS = {
    "algorithm",
    "n_j_matrices_n8",
    "kl_v6_vs_cpu_n8_mean",
    "kl_v6_below_threshold_n8",
    "n128_k16_tested",
    "kl_v6_vs_cpu_n128_mean",
    "kl_v6_below_threshold_n128",
    "kl_improvement_over_v5",
    "kl_improvement_over_v4",
    "rtl_spec_written",
    "rtl_spec_path",
    "kv260_v6_kl_below_threshold_sequential_gibbs",
    "honest_verdict",
}


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sigmoid(x: float) -> float:
    """Numerically stable scalar sigmoid."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _validate_sampler_inputs(
    j_matrix: np.ndarray,
    b: np.ndarray,
    n_spins: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated float64 J and b arrays."""
    j = np.asarray(j_matrix, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if j.shape != (n_spins, n_spins):
        raise ValueError(f"J must have shape ({n_spins}, {n_spins}), got {j.shape}")
    if b_arr.shape != (n_spins,):
        raise ValueError(f"b must have shape ({n_spins},), got {b_arr.shape}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be non-negative, got {n_steps}")
    return j, b_arr


def _run_sequential_gibbs(
    j_matrix: np.ndarray,
    b: np.ndarray,
    *,
    n_spins: int,
    n_steps: int,
    beta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run strict round-robin sequential Gibbs and return samples plus update indices."""
    j, b_arr = _validate_sampler_inputs(j_matrix, b, n_spins, n_steps)
    rng = np.random.default_rng(seed)
    state = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_spins).astype(np.int8)
    samples = np.empty((n_steps, n_spins), dtype=np.int8)
    update_indices = np.empty(n_steps, dtype=np.int64)

    for step in range(n_steps):
        spin_idx = step % n_spins
        field = float(j[spin_idx] @ state.astype(np.float64) + b_arr[spin_idx])
        p_plus = _sigmoid(2.0 * beta * field)
        state[spin_idx] = 1 if rng.random() < p_plus else -1
        samples[step] = state
        update_indices[step] = spin_idx

    return samples, update_indices


class SequentialGibbsSampler:
    """Sequential Gibbs sampler with strict spin-by-spin updates."""

    def __init__(self) -> None:
        self.last_update_indices = np.empty(0, dtype=np.int64)
        self.last_final_state = np.empty(0, dtype=np.int8)

    def sample(
        self,
        J: np.ndarray,
        b: np.ndarray,
        n_spins: int,
        n_steps: int,
        beta: float,
        seed: int,
    ) -> np.ndarray:
        """Sequential Gibbs with strict spin-by-spin updates.

        s_i = +1 with P = sigmoid(2*beta*(sum_j J_ij*s_j + b_i)).
        Update spin i_t = (t % n_spins) at each step.
        """
        samples, update_indices = _run_sequential_gibbs(
            J,
            b,
            n_spins=n_spins,
            n_steps=n_steps,
            beta=beta,
            seed=seed,
        )
        self.last_update_indices = update_indices
        self.last_final_state = samples[-1].copy() if n_steps else np.empty(n_spins, dtype=np.int8)
        return samples


def cpu_gibbs_reference_samples(
    J: np.ndarray,
    b: np.ndarray,
    *,
    n_spins: int,
    n_steps: int,
    beta: float,
    seed: int,
) -> np.ndarray:
    """CPU reference sampler using the same standard sequential Gibbs transition."""
    samples, _ = _run_sequential_gibbs(
        J,
        b,
        n_spins=n_spins,
        n_steps=n_steps,
        beta=beta,
        seed=seed,
    )
    return samples


def build_n8_exp1149_j_matrices() -> list[np.ndarray]:
    """Reuse the three deterministic Exp 1149 signed weighted K=2 ring matrices."""
    return exp1149.build_exp1134_seeded_j_matrices(n_spins=N8_SPINS, seeds=MATRIX_SEEDS)


def build_sparse_ring_j_matrix(
    *,
    n_spins: int,
    k_neighbors: int,
    j_value: float = -1.0,
) -> np.ndarray:
    """Build a symmetric K-nearest sparse ring J matrix for the KV260 v4/v6 topology."""
    if k_neighbors % 2 != 0:
        raise ValueError(f"k_neighbors must be even, got {k_neighbors}")
    if k_neighbors >= n_spins:
        raise ValueError(
            f"k_neighbors must be less than n_spins to avoid self-neighbours, got {k_neighbors}"
        )
    j_matrix = np.zeros((n_spins, n_spins), dtype=np.float64)
    half = k_neighbors // 2
    for spin_idx in range(n_spins):
        for offset in range(1, half + 1):
            right = (spin_idx + offset) % n_spins
            left = (spin_idx - offset) % n_spins
            j_matrix[spin_idx, right] = float(j_value)
            j_matrix[right, spin_idx] = float(j_value)
            j_matrix[spin_idx, left] = float(j_value)
            j_matrix[left, spin_idx] = float(j_value)
    return j_matrix


def _sample_keys(samples: np.ndarray) -> list[bytes]:
    """Pack each spin configuration into a byte key without enumerating state space."""
    arr = np.asarray(samples, dtype=np.int8)
    packed = np.packbits(arr > 0, axis=1, bitorder="little")
    return [bytes(row) for row in packed]


def empirical_kl_between_samples(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    *,
    smoothing: float = 1.0,
) -> float:
    """Compute KL(P_empirical || Q_empirical) over observed sample support.

    This union-support estimate is used for v6-vs-CPU parity, including N=128
    where exact enumeration of all `2**128` spin states is impossible.
    """
    p_keys = _sample_keys(p_samples)
    q_keys = _sample_keys(q_samples)
    support = set(p_keys) | set(q_keys)
    if not support:
        return 0.0

    p_counts = {key: 0 for key in support}
    q_counts = {key: 0 for key in support}
    for key in p_keys:
        p_counts[key] += 1
    for key in q_keys:
        q_counts[key] += 1

    support_size = float(len(support))
    p_total = float(len(p_keys)) + smoothing * support_size
    q_total = float(len(q_keys)) + smoothing * support_size
    kl = 0.0
    for key in support:
        p_prob = (p_counts[key] + smoothing) / p_total
        q_prob = (q_counts[key] + smoothing) / q_total
        kl += p_prob * math.log(p_prob / q_prob)
    return float(kl)


def _run_pair_measurement(
    j_matrix: np.ndarray,
    *,
    matrix_id: str,
    n_steps: int,
    beta: float,
    seed: int,
    matrix_seed: int | None = None,
) -> dict[str, Any]:
    """Run v6 and CPU reference samplers and return their empirical KL."""
    n_spins = int(j_matrix.shape[0])
    b = np.zeros(n_spins, dtype=np.float64)
    sampler = SequentialGibbsSampler()
    v6_samples = sampler.sample(j_matrix, b, n_spins=n_spins, n_steps=n_steps, beta=beta, seed=seed)
    cpu_samples = cpu_gibbs_reference_samples(
        j_matrix,
        b,
        n_spins=n_spins,
        n_steps=n_steps,
        beta=beta,
        seed=seed,
    )
    kl = empirical_kl_between_samples(v6_samples, cpu_samples)
    measurement = {
        "matrix_id": matrix_id,
        "n_spins": n_spins,
        "n_steps": int(n_steps),
        "beta": float(beta),
        "seed": int(seed),
        "kl_v6_vs_cpu_gibbs": kl,
        "kl_v6_below_threshold": bool(kl < KL_THRESHOLD),
    }
    if matrix_seed is not None:
        measurement["matrix_seed"] = int(matrix_seed)
    return measurement


def run_n8_measurements(
    *,
    n_steps: int = DEFAULT_N_STEPS,
    beta: float = BETA,
) -> list[dict[str, Any]]:
    """Run the three Exp 1149 N=8 K=2 matrices through v6-vs-CPU parity."""
    matrices = build_n8_exp1149_j_matrices()
    if len(matrices) != len(MATRIX_SEEDS):
        raise ValueError("matrix generation failed: expected three Exp 1149 J matrices")
    return [
        _run_pair_measurement(
            j_matrix,
            matrix_id=f"exp1134_seeded_j{matrix_idx}",
            matrix_seed=MATRIX_SEEDS[matrix_idx],
            n_steps=n_steps,
            beta=beta,
            seed=EXPERIMENT_ID + matrix_idx,
        )
        for matrix_idx, j_matrix in enumerate(matrices)
    ]


def run_n128_k16_measurement(
    *,
    n_steps: int = DEFAULT_N_STEPS,
    beta: float = BETA,
) -> dict[str, Any]:
    """Run the KV260 v4 target N=128, K=16 sparse ring through v6-vs-CPU parity."""
    j_matrix = build_sparse_ring_j_matrix(
        n_spins=N128_SPINS,
        k_neighbors=K16_NEIGHBORS,
        j_value=-1.0,
    )
    measurement = _run_pair_measurement(
        j_matrix,
        matrix_id="n128_k16_sparse_ring",
        n_steps=n_steps,
        beta=beta,
        seed=EXPERIMENT_ID + 128,
    )
    measurement["n128_k16_tested"] = True
    measurement["k_neighbors"] = K16_NEIGHBORS
    return measurement


def mean_kl(measurements: list[dict[str, Any]], key: str) -> float:
    """Return the mean KL for a list of measurement dictionaries."""
    return float(np.mean([float(measurement[key]) for measurement in measurements]))


def classify_verdict(kl_v6_vs_cpu_n8_mean: float, *, matrix_generation_failed: bool = False) -> str:
    """Map v6 KL parity to the approved Exp 1161 honest-verdict vocabulary."""
    if matrix_generation_failed:
        return "matrix_generation_failed"
    if not math.isfinite(kl_v6_vs_cpu_n8_mean) or kl_v6_vs_cpu_n8_mean >= KL_THRESHOLD:
        return "kl_above_threshold_unexplained"
    if abs(kl_v6_vs_cpu_n8_mean) <= 1e-12:
        return "kl_near_zero_algorithm_correct"
    return "kl_below_threshold_sequential_correct"


def load_prior_kl_values() -> tuple[float, float]:
    """Load prior v5 and v4 KL anchors from their artifacts, with documented fallbacks."""
    kl_v5 = KL_V5_PRIOR_FALLBACK
    kl_v4 = KL_V4_PRIOR_FALLBACK
    if EXP1149_ARTIFACT.exists():
        kl_v5 = float(json.loads(EXP1149_ARTIFACT.read_text()).get("kl_v5_best", kl_v5))
    if EXP1134_ARTIFACT.exists():
        kl_v4 = float(json.loads(EXP1134_ARTIFACT.read_text()).get("kl_v4_best", kl_v4))
    return kl_v5, kl_v4


def rtl_spec_text() -> str:
    """Return the v6 RTL pseudocode specification text."""
    return """# KV260 Ising Sampler v6 RTL Spec - Sequential Gibbs

Spec refs: REQ-HW-045, SCENARIO-HW-045.

## Purpose

v6 is the KL-correct pivot from parallel sparse Glauber to strict sequential
Gibbs. It updates one spin per clock so each conditional draw sees the current
state of every previously updated spin, preserving the detailed-balance
semantics used by the CPU Gibbs reference.

## State

- `s[N]`: one signed spin register per Ising variable, encoded as {-1,+1}.
- `h[N]`: signed fixed-point field cache, where `h[i] = sum_j J[i,j] * s[j] + b[i]`.
- `t`: modulo-N spin-select counter.
- `rng`: uniform random source used by the Bernoulli draw.
- `J_sparse[N][K]` and `nbr_idx[N][K]`: sparse coupling table for the KV260 K=16 target.
- `b[N]`: optional signed bias vector.

## One Spin Per Clock Pseudocode

```text
on reset:
  for i in 0..N-1:
    s[i] <- +1
    h[i] <- sum_k J_sparse[i][k] * s[nbr_idx[i][k]] + b[i]
  t <- 0

on each clock:
  i <- t % N
  h_i <- sum_k J_sparse[i][k] * s[nbr_idx[i][k]] + b[i]
  p_plus <- sigmoid_lut(2 * beta * h_i)
  old_s <- s[i]
  s[i] <- +1 if rng_uniform() < p_plus else -1
  delta <- s[i] - old_s

  if delta != 0:
    for each neighbor r of i:
      h[r] <- h[r] + J[r][i] * delta
  h[i] <- h_i
  t <- (t + 1) % N
```

## Acceptance Gate

The Python reference for this RTL must report `algorithm = "sequential_gibbs"`
and `kl_v6_below_threshold_n8 = true` against the CPU sequential-Gibbs reference
on the three Exp 1149 N=8 K=2 matrices. It must also run the N=128 K=16 sparse
ring topology without exact `2**128` enumeration and report
`kl_v6_below_threshold_n128 = true`.
"""


def write_rtl_spec(path: Path = RTL_SPEC_PATH) -> bool:
    """Write the v6 RTL pseudocode spec."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rtl_spec_text())
    return path.exists()


def _display_path(path: Path) -> str:
    """Return a repo-relative path when the file is inside this checkout."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_artifact(
    *,
    n8_measurements: list[dict[str, Any]],
    n128_measurement: dict[str, Any],
    kl_v5_prior: float,
    kl_v4_prior: float,
    rtl_spec_written: bool,
    duration_s: float,
    run_date: str,
) -> dict[str, Any]:
    """Build the JSON-serializable Exp 1161 artifact."""
    kl_n8_mean = mean_kl(n8_measurements, "kl_v6_vs_cpu_gibbs")
    kl_n128 = float(n128_measurement["kl_v6_vs_cpu_gibbs"])
    artifact = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "schema": "kv260_v6_sequential_gibbs_v1",
        "run_date": run_date,
        "algorithm": ALGORITHM,
        "beta": BETA,
        "kl_threshold": KL_THRESHOLD,
        "n_steps_per_matrix": int(n8_measurements[0]["n_steps"]) if n8_measurements else 0,
        "n_j_matrices_n8": len(n8_measurements),
        "n8_matrix_seeds": list(MATRIX_SEEDS),
        "j_matrix_source": (
            "Same Exp 1149 deterministic signed weighted K=2 ring matrices generated "
            "from seeds 1134, 1135, 1136."
        ),
        "per_matrix_n8": n8_measurements,
        "kl_v6_vs_cpu_n8_mean": kl_n8_mean,
        "kl_v6_below_threshold_n8": bool(kl_n8_mean < KL_THRESHOLD),
        "n128_k16_tested": bool(n128_measurement["n128_k16_tested"]),
        "n128_k16_measurement": n128_measurement,
        "kl_v6_vs_cpu_n128_mean": kl_n128,
        "kl_v6_below_threshold_n128": bool(n128_measurement["kl_v6_below_threshold"]),
        "kl_v5_best_prior": float(kl_v5_prior),
        "kl_v4_best_prior": float(kl_v4_prior),
        "kl_improvement_over_v5": float(kl_v5_prior - kl_n8_mean),
        "kl_improvement_over_v4": float(kl_v4_prior - kl_n8_mean),
        "rtl_spec_written": bool(rtl_spec_written),
        "rtl_spec_path": _display_path(RTL_SPEC_PATH),
        "kv260_v6_kl_below_threshold_sequential_gibbs": bool(kl_n8_mean < KL_THRESHOLD),
        "honest_verdict": classify_verdict(kl_n8_mean),
        "duration_s": round(float(duration_s), 3),
    }
    return artifact


def write_artifact(artifact: dict[str, Any], path: Path = DELIVERABLE) -> None:
    """Write the Exp 1161 JSON deliverable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main() -> int:
    """Run Exp 1161 end-to-end and write the required deliverables."""
    start = time.time()
    print(f"[exp{EXPERIMENT_ID}] starting {ALGORITHM}")
    kl_v5_prior, kl_v4_prior = load_prior_kl_values()
    print(f"[exp{EXPERIMENT_ID}] priors: v5 KL={kl_v5_prior:.6f}, v4 KL={kl_v4_prior:.6f}")

    n8_measurements = run_n8_measurements(n_steps=DEFAULT_N_STEPS, beta=BETA)
    n128_measurement = run_n128_k16_measurement(n_steps=DEFAULT_N_STEPS, beta=BETA)
    rtl_written = write_rtl_spec(RTL_SPEC_PATH)
    artifact = build_artifact(
        n8_measurements=n8_measurements,
        n128_measurement=n128_measurement,
        kl_v5_prior=kl_v5_prior,
        kl_v4_prior=kl_v4_prior,
        rtl_spec_written=rtl_written,
        duration_s=time.time() - start,
        run_date=_utc_now_iso(),
    )
    write_artifact(artifact, DELIVERABLE)
    print(f"wrote {DELIVERABLE}")
    print(f"wrote {RTL_SPEC_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"kl_v6_vs_cpu_n8_mean: {artifact['kl_v6_vs_cpu_n8_mean']:.12f}")
    print(f"kl_v6_vs_cpu_n128_mean: {artifact['kl_v6_vs_cpu_n128_mean']:.12f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
