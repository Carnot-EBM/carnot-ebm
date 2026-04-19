#!/usr/bin/env python3
"""Exp 521 — Hallucination Basin Detector: basin depth from hidden-state trajectories.

**What this experiment measures:**
    Does basin depth (arXiv 2604.04743) distinguish correct-reasoning hidden states
    (deep-basin attractors) from hallucinated hidden states (shallow basins / saddle points)?

    200 synthetic hidden-state trajectories are generated:
    - 100 'correct': states sampled near a global energy minimum (deep basin)
    - 100 'hallucinated': states sampled near a saddle point (shallow basin)

    HallucinationBasinDetector AUROC is compared to a random baseline.
    AUROC > baseline → viable Tier 0d candidate for the verification cascade.

Spec: REQ-VERIFY-107, REQ-VERIFY-108
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Step a: apply_env_autofix FIRST (REQ-INFRA-060)
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step b: ExperimentTimeoutWatchdog
# ---------------------------------------------------------------------------

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(521, timeout_minutes=25)
_watchdog.start()

# ---------------------------------------------------------------------------
# Step c: ExperimentTemplate
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    521,
    "Hallucination Basin Detector",
    "results/experiment_521_hallucination_basin_detector.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Step d: DeliverableGuard
# ---------------------------------------------------------------------------

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

_guard = DeliverableGuard(str(_REPO / "results" / "experiment_521_hallucination_basin_detector.json"))

# ---------------------------------------------------------------------------
# Imports for experiment body
# ---------------------------------------------------------------------------

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from carnot.pipeline.hallucination_basin import HallucinationBasinDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Step e: Generate 200 synthetic hidden-state trajectories
# ---------------------------------------------------------------------------

print("[521] Generating 200 synthetic hidden-state trajectories...")

HIDDEN_DIM = 32   # hidden-state dimensionality (small for speed)
SEQ_LEN = 10      # timesteps per trajectory
N_CORRECT = 100
N_HALLUCINATED = 100

rng = np.random.RandomState(42)


def quadratic_energy(x: jnp.ndarray) -> float:
    """Global minimum at origin: energy = sum(x**2).

    Used as the energy proxy.  In production this would be a trained IsingEBM.
    """
    return float(jnp.sum(x**2))


# 'Correct' trajectories: states sampled tightly around the global minimum (origin).
# Small Gaussian noise around 0 → all states sit in the deep quadratic basin.
correct_trajectories: list[tuple[jnp.ndarray, int]] = []
for _ in range(N_CORRECT):
    # Small perturbations around the minimum → deep basin
    states = rng.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) * 0.05
    correct_trajectories.append((jnp.asarray(states), 0))

# 'Hallucinated' trajectories: states sampled far from any minimum.
# Large norm + random direction → near saddle points or slopes of the quadratic.
# We set the mean to a large value so the states are far from the basin bottom.
hallucinated_trajectories: list[tuple[jnp.ndarray, int]] = []
for _ in range(N_HALLUCINATED):
    # States far from origin: energy is large but basin depth is near zero
    # because the quadratic gradient is uniform — perturbations in any direction
    # from a non-minimum point will sometimes find lower energy nearby.
    # To ensure shallowness: sample near a saddle of a modified energy, or
    # simply use the linear trick: perturb along the gradient direction.
    # Here: sample along a random unit vector at distance 2.0 from origin.
    # At a point far from the origin on the quadratic, the depth is non-trivially
    # shallow because perturbations toward the origin lower energy.
    direction = rng.randn(HIDDEN_DIM).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    # Add small noise around the off-minimum point
    noise = rng.randn(SEQ_LEN, HIDDEN_DIM).astype(np.float32) * 0.1
    states = noise + direction[None, :] * 3.0  # 3.0 units from origin = far from basin
    hallucinated_trajectories.append((jnp.asarray(states), 1))

all_trajectories = correct_trajectories + hallucinated_trajectories
all_labels = [0] * N_CORRECT + [1] * N_HALLUCINATED

print(f"[521] Generated {len(all_trajectories)} trajectories "
      f"({N_CORRECT} correct, {N_HALLUCINATED} hallucinated).")

# ---------------------------------------------------------------------------
# Step f: SpilledEnergy baseline (random baseline if unavailable)
# ---------------------------------------------------------------------------

print("[521] Computing baseline AUROC (random baseline)...")

# The SpilledEnergyDetector operates on logits, not hidden states, so it is not
# directly applicable here.  We use a random classifier as the baseline to measure
# whether the basin detector beats chance.
np.random.seed(0)
random_scores = np.random.uniform(0, 1, len(all_labels))

try:
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    baseline_auroc = float(roc_auc_score(all_labels, random_scores))
except Exception:
    baseline_auroc = 0.5

print(f"[521] Baseline AUROC (random): {baseline_auroc:.4f}")

# ---------------------------------------------------------------------------
# Step g: HallucinationBasinDetector — compute AUROC
# ---------------------------------------------------------------------------

print("[521] Running HallucinationBasinDetector on all trajectories...")

detector = HallucinationBasinDetector(
    energy_fn=quadratic_energy,
    n_perturbations=8,
    perturbation_scale=0.1,
)

basin_result = detector.benchmark(all_trajectories)
basin_detector_auroc = basin_result["auroc"]

print(f"[521] HallucinationBasinDetector AUROC: {basin_detector_auroc:.4f}")

# ---------------------------------------------------------------------------
# Step h: Build artifact
# ---------------------------------------------------------------------------

basin_detector_viable = bool(basin_detector_auroc > baseline_auroc)
honest_verdict = "viable_tier0d" if basin_detector_viable else "no_improvement"

print(f"[521] basin_detector_viable={basin_detector_viable}, honest_verdict={honest_verdict}")

artifact = tmpl.build_result(
    {
        "n_trajectories": len(all_trajectories),
        "n_correct": N_CORRECT,
        "n_hallucinated": N_HALLUCINATED,
        "hidden_dim": HIDDEN_DIM,
        "seq_len": SEQ_LEN,
        "n_perturbations": 8,
        "perturbation_scale": 0.1,
        "basin_detector_auroc": basin_detector_auroc,
        "baseline_auroc": baseline_auroc,
        "basin_detector_viable": basin_detector_viable,
        "honest_verdict": honest_verdict,
        "energy_proxy": "quadratic_sum_x2",
        "arxiv_ref": "2604.04743",
    },
    status="success",
    schema="carnot.basin_detector.v1",
)

# Write the deliverable
out_path = _REPO / "results" / "experiment_521_hallucination_basin_detector.json"
out_path.write_text(json.dumps(artifact, indent=2))
print(f"[521] Wrote deliverable to {out_path}")

# ---------------------------------------------------------------------------
# Step i: assert deliverable written — FINAL LINE
# ---------------------------------------------------------------------------

tmpl.assert_deliverable_written()
