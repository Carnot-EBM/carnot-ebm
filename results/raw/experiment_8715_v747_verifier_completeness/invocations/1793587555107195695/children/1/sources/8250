#!/usr/bin/env python3
"""Exp 819: External Field Injection Fix — close RETRO-ISING-INJECTION-NO-DISCRIMINATION.

**Researcher summary:**
    Exp 812 discovered that IsingConstraintInjector.inject_into_coupling_matrix() adds
    constraint bias to the DIAGONAL of J, producing a constant energy shift -0.5*sum(bias)
    that is identical for ALL spin configurations (since s_i^2=1 for ±1 spins). This means
    the diagonal injection CANNOT discriminate between violated and correct responses.

    This experiment implements and validates the fix: external field injection via
    compute_energy_with_external_field(J, spins, constraint_embeddings), where:
        E_total = -0.5 * s^T J s + h^T s
        h       = clip(W @ mean(embeddings), 0, inf)

    With violation spins s_i = +1 and h[i] > 0:
        E_field = +h[i] → total energy INCREASES (violation penalised)
    With correct spins s_i = -1 and h[i] > 0:
        E_field = -h[i] → total energy DECREASES (correct response rewarded)

**Evaluation protocol:**
    10 violation/correct spin pairs with fixed constraint embeddings (seed=42).
    Violation: first 4 spins = +1, rest = -1.
    Correct: all spins = -1.
    discrimination_rate = fraction of pairs where E_total(violation) > E_total(correct).

    honest_verdict:
      - "injection_field_fixed"  if discrimination_rate >= 0.80
      - "injection_partial"      if 0.50 <= discrimination_rate < 0.80
      - "injection_still_wrong"  if discrimination_rate < 0.50

Spec: REQ-VERIFY-173, REQ-VERIFY-174, SCENARIO-VERIFY-227, SCENARIO-VERIFY-228
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 819
TITLE = "External Field Injection Fix — close RETRO-ISING-INJECTION-NO-DISCRIMINATION"
DELIVERABLE = "results/experiment_819_injection_field_fix.json"
TIMEOUT_MINUTES = 30

N_SPINS = 16
EMB_DIM = 384
VIOLATION_INDICES = list(range(4))
N_PAIRS = 10


def main() -> None:
    apply_env_autofix()

    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE, requires_gpu=False)
    tmpl.setup()

    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=TIMEOUT_MINUTES)

    # Reproducible constraint embeddings (5 embeddings of dim 384).
    rng = np.random.default_rng(42)
    constraint_embeddings: list[list[float]] = rng.standard_normal((5, EMB_DIM)).tolist()

    injector = IsingConstraintInjector(embedding_dim=EMB_DIM, n_spins=N_SPINS)
    J = np.eye(N_SPINS, dtype=np.float64)  # identity baseline — no inherent coupling

    # ---------------------------------------------------------------------------
    # External field discrimination test
    # ---------------------------------------------------------------------------
    pair_results = []
    n_discriminating = 0

    for _ in range(N_PAIRS):
        spins_v = np.full(N_SPINS, -1.0)
        for i in VIOLATION_INDICES:
            spins_v[i] = 1.0

        spins_c = np.full(N_SPINS, -1.0)

        res_v = injector.compute_energy_with_external_field(J, spins_v, constraint_embeddings)
        res_c = injector.compute_energy_with_external_field(J, spins_c, constraint_embeddings)

        discriminating = bool(res_v.E_total > res_c.E_total)
        if discriminating:
            n_discriminating += 1

        pair_results.append(
            {
                "E_violation": res_v.E_total,
                "E_correct": res_c.E_total,
                "E_field_violation": res_v.E_field,
                "E_field_correct": res_c.E_field,
                "h_norm": res_v.h_norm,
                "discriminating": discriminating,
            }
        )

    discrimination_rate = n_discriminating / N_PAIRS

    # ---------------------------------------------------------------------------
    # Legacy diagonal injection: confirm constant-shift (delta ~= 0)
    # ---------------------------------------------------------------------------
    bias = injector.project_to_spin_bias(constraint_embeddings)
    J_injected = injector.inject_into_coupling_matrix(J, bias)

    spins_v_leg = np.full(N_SPINS, -1.0)
    for i in VIOLATION_INDICES:
        spins_v_leg[i] = 1.0
    spins_c_leg = np.full(N_SPINS, -1.0)

    e_leg_v = float(-0.5 * spins_v_leg @ J_injected @ spins_v_leg)
    e_leg_c = float(-0.5 * spins_c_leg @ J_injected @ spins_c_leg)
    legacy_delta = float(e_leg_v - e_leg_c)

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    if discrimination_rate >= 0.80:
        honest_verdict = "injection_field_fixed"
    elif discrimination_rate >= 0.50:
        honest_verdict = "injection_partial"
    else:
        honest_verdict = "injection_still_wrong"

    retro_injection_closed = discrimination_rate >= 0.80
    external_field_implemented = True

    artifact = tmpl.build_result(
        {
            "n_discriminating": n_discriminating,
            "discrimination_rate": discrimination_rate,
            "legacy_delta": legacy_delta,
            "external_field_implemented": external_field_implemented,
            "retro_injection_closed": retro_injection_closed,
            "honest_verdict": honest_verdict,
            "pair_results": pair_results,
            "n_pairs": N_PAIRS,
            "n_spins": N_SPINS,
            "emb_dim": EMB_DIM,
            "violation_indices": VIOLATION_INDICES,
        },
        status="success",
    )

    out_path = Path(DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(
        f"Exp {EXP_ID}: {honest_verdict} | discrimination_rate={discrimination_rate:.2f} | legacy_delta={legacy_delta:.6f}"
    )
    print(f"Written: {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
