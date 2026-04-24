#!/usr/bin/env python3
"""Exp 812: Ising Constraint Injection — wire EmbeddingConstraintStore into IsingEBM coupling matrix.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA fix):**
    Exps 788 and 801 both showed constraint_addition_delta=0.0 because retrieved
    constraint embeddings were stored as metadata ConstraintResult entries only —
    they never entered the Ising energy computation.  The fix (IsingConstraintInjector)
    projects each retrieved embedding to a (n_spins,) bias via a linear map W and adds
    it to the diagonal of J before computing energy.  This makes the Ising energy
    sensitive to which constraint fired, closing RETRO-CONSTRAINT-ZERO-DELTA.

**honest_verdict logic:**
    - "injection_works"          if mean energy_delta_pct > 0 for error responses
    - "injection_no_delta"       if mean delta_pct <= 0 (constraint had no effect)
    - "injection_negative_delta" if mean delta_pct < 0 (sign error in bias direction)

Spec: REQ-VERIFY-095, REQ-VERIFY-096, SCENARIO-VERIFY-129
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Force CPU JAX — all operations here are EBM, no LLM inference needed.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    EmbeddingConstraintStore,
)
from carnot.pipeline.ising_constraint_injector import (  # noqa: E402
    ConstraintInjectionResult,
    IsingConstraintInjector,
)
from carnot.models.ising import IsingConfig, IsingModel  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

import jax
import jax.numpy as jnp
import numpy as np

EXP_ID = 812
TITLE = "Ising Constraint Injection — wire EmbeddingConstraintStore into IsingEBM coupling matrix"
DELIVERABLE = "results/experiment_812_ising_constraint_injection.json"
TIMEOUT_MINUTES = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# 10 test response texts: 5 with arithmetic carry errors, 5 without.
ERROR_RESPONSES = [
    "The sum 255 + 1 = 255 because carry does not propagate beyond 8 bits.",
    "Adding 127 + 1 gives 127 since there is no carry out from the high bit.",
    "Result: 0xFF + 0x01 = 0xFF, carry is lost so the answer stays at 255.",
    "Binary addition: 1111 + 0001 = 1111 because the carry is dropped silently.",
    "Overflow: 11111111 + 00000001 = 11111111, carry propagation ignored.",
]

CLEAN_RESPONSES = [
    "The sum 255 + 1 = 256, with carry propagating to the next higher bit.",
    "Adding 127 + 1 gives 128 because the carry flips bit 7 to 1 and clears lower bits.",
    "Binary: 1111 + 0001 = 10000, carry propagates correctly to produce 16.",
    "0xFF + 0x01 = 0x100 (256 in decimal), carry out from bit 7.",
    "Correct arithmetic: 11111111 + 00000001 = 100000000 (carry fully propagated).",
]


def compute_energy_no_injection(ising: IsingModel, spins: np.ndarray) -> float:
    """Standard Ising energy without any constraint bias.

    E = -0.5 * spins^T J spins  (ignoring bias term for comparable measurement).
    """
    J = np.array(ising.coupling, dtype=np.float64)
    return float(-0.5 * spins @ J @ spins)


def run_injection_trial(
    store: EmbeddingConstraintStore,
    injector: IsingConstraintInjector,
    ising: IsingModel,
    spins: np.ndarray,
    response_text: str,
) -> ConstraintInjectionResult:
    """Run one injection trial and return a ConstraintInjectionResult.

    Steps:
        1. Retrieve top-3 constraints from store for the response text.
        2. Compute energy WITHOUT injection (original J).
        3. Compute energy WITH injection (J + diag(projected bias)).
        4. Compute energy_delta_pct and honest_verdict.
    """
    retrieved = store.retrieve(response_text, top_k=3)
    embeddings = [c.embedding for c in retrieved if c.embedding]

    energy_no = compute_energy_no_injection(ising, spins)
    projection_applied = len(embeddings) > 0

    if projection_applied:
        energy_yes = injector.compute_energy_with_injection(ising, spins, embeddings)
    else:
        energy_yes = energy_no

    if abs(energy_no) > 1e-12:
        delta_pct = 100.0 * (energy_yes - energy_no) / abs(energy_no)
    else:
        delta_pct = 0.0

    if delta_pct > 0:
        verdict = "injection_works"
    elif delta_pct < 0:
        verdict = "injection_negative_delta"
    else:
        verdict = "injection_no_delta"

    return ConstraintInjectionResult(
        n_constraints_retrieved=len(retrieved),
        embedding_dim=injector.embedding_dim,
        n_spins=injector.n_spins,
        projection_applied=projection_applied,
        energy_without_injection=energy_no,
        energy_with_injection=energy_yes,
        energy_delta_pct=delta_pct,
        honest_verdict=verdict,
    )


def main() -> None:
    # MANDATORY: apply env autofix before any JAX or model code runs.
    apply_env_autofix()

    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)
    watchdog.start()

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # --- Setup constraint store ---
    store = EmbeddingConstraintStore()
    # Bootstrap from legacy CaseMemory patterns covering carry, sign, unit, comparison, causal.
    store.from_casememory_patterns(
        {"carry": 4, "sign": 2, "unit": 1, "comparison": 1, "causal": 1}
    )
    n_stored = len(store._store)

    # --- Setup injector ---
    N_SPINS = 64
    injector = IsingConstraintInjector(embedding_dim=384, n_spins=N_SPINS)

    # --- Setup Ising model ---
    ising_config = IsingConfig(input_dim=N_SPINS, coupling_init="xavier_uniform")
    ising = IsingModel(ising_config, key=jax.random.PRNGKey(42))

    # Use fixed random spins for reproducibility across error and clean responses.
    rng = np.random.default_rng(7)
    spins = rng.choice([-1.0, 1.0], size=N_SPINS)

    # --- Run trials ---
    error_results: list[ConstraintInjectionResult] = []
    clean_results: list[ConstraintInjectionResult] = []

    for resp in ERROR_RESPONSES:
        r = run_injection_trial(store, injector, ising, spins, resp)
        error_results.append(r)

    for resp in CLEAN_RESPONSES:
        r = run_injection_trial(store, injector, ising, spins, resp)
        clean_results.append(r)

    # --- Aggregate metrics ---
    mean_error_delta = float(np.mean([r.energy_delta_pct for r in error_results]))
    mean_clean_delta = float(np.mean([r.energy_delta_pct for r in clean_results]))

    # Key test: error responses should have higher energy with injection.
    if mean_error_delta > 0:
        honest_verdict = "injection_works"
    elif mean_error_delta < 0:
        honest_verdict = "injection_negative_delta"
    else:
        honest_verdict = "injection_no_delta"

    per_trial_error = [
        {
            "response_snippet": r[:60] + "..." if len(r) > 60 else r,
            "n_constraints_retrieved": res.n_constraints_retrieved,
            "energy_without_injection": round(res.energy_without_injection, 6),
            "energy_with_injection": round(res.energy_with_injection, 6),
            "energy_delta_pct": round(res.energy_delta_pct, 4),
            "honest_verdict": res.honest_verdict,
        }
        for r, res in zip(ERROR_RESPONSES, error_results)
    ]
    per_trial_clean = [
        {
            "response_snippet": r[:60] + "..." if len(r) > 60 else r,
            "n_constraints_retrieved": res.n_constraints_retrieved,
            "energy_without_injection": round(res.energy_without_injection, 6),
            "energy_with_injection": round(res.energy_with_injection, 6),
            "energy_delta_pct": round(res.energy_delta_pct, 4),
            "honest_verdict": res.honest_verdict,
        }
        for r, res in zip(CLEAN_RESPONSES, clean_results)
    ]

    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "embedding_dim": injector.embedding_dim,
            "n_spins": N_SPINS,
            "n_stored_constraints": n_stored,
            "n_error_responses": len(ERROR_RESPONSES),
            "n_clean_responses": len(CLEAN_RESPONSES),
            "mean_energy_delta_pct_errors": round(mean_error_delta, 4),
            "mean_energy_delta_pct_clean": round(mean_clean_delta, 4),
            "per_trial_error": per_trial_error,
            "per_trial_clean": per_trial_clean,
            "retro_constraint_zero_delta_closed": honest_verdict == "injection_works",
        },
        status="success",
    )

    out_path = Path(DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp {EXP_ID}] Deliverable written: {DELIVERABLE}")
    print(f"[Exp {EXP_ID}] honest_verdict={honest_verdict}")
    print(f"[Exp {EXP_ID}] mean_energy_delta_pct_errors={mean_error_delta:.4f}")

    watchdog.stop()
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
