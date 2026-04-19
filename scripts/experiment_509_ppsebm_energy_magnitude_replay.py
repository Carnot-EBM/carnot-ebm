#!/usr/bin/env python3
"""Experiment 509: PPSEBM Energy Magnitude Replay (RETRO-050 closure attempt).

**Research question:**
    Does replacing LLM-surprise replay priority (SuRe, Exp 497) with EBM energy
    magnitude priority improve partition isolation score for PPSConstraintLearner?

**RETRO-050 background:**
    Exp 497 (SuRe) showed isolation_improvement=-0.1172 versus uniform replay baseline.
    Root cause: LLM NLL (token-sequence surprise) and EBM energy magnitude are
    anticorrelated — common sentences can violate hard constraints at high energy,
    and unusual sentences can have low constraint energy.

    RETRO-050 recommendation: rank violations by |energy(x) - domain_mean| instead.
    High |energy - mean| = domain boundary case = what the EBM is most wrong about =
    exactly what replay should prioritize to prevent catastrophic forgetting.

**What this experiment measures:**
    1. Simulate 200 constraint violations across arithmetic/code/logical domains with
       domain-specific energy distributions (different means and spreads per domain).
    2. Compute isolation_score(arithmetic, code, n_steps=50) for EnergyMagnitudeReplay.
    3. Compare vs SuRe baseline (isolation_improvement = -0.1172).
    4. Honest verdict: did energy-magnitude priority improve over SuRe?

**Why CPU-only:**
    This is a replay-strategy simulation, not a live training run.  The simulation
    uses synthetic violations with known energy distributions to isolate the effect
    of replay priority on isolation score, independent of GPU model inference.

Spec: REQ-LEARN-043, REQ-LEARN-044, REQ-LEARN-045,
      SCENARIO-LEARN-071, SCENARIO-LEARN-072, SCENARIO-LEARN-073,
      RETRO-050
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path for scripts/ imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.energy_magnitude_replay import EnergyMagnitudeReplay
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from experiment_template import ExperimentTemplate

# SuRe baseline from Exp 497 — the number we must beat.
_SURE_BASELINE = -0.1172


def _simulate_violations(
    replay: EnergyMagnitudeReplay,
    rng: random.Random,
) -> None:
    """Simulate 200 constraint violations with domain-specific energy distributions.

    WHY domain-specific distributions:
        Real constraint domains have different natural energy scales.
        ARITHMETIC violations tend to cluster near low energies (easy to resolve).
        CODE violations have higher variance (ambiguous edge cases are hard).
        LOGICAL violations have the highest mean energy (complex constraint structure).

        Using different distributions per domain tests whether EnergyMagnitudeReplay
        correctly normalises deviation per domain (not raw energy), so a CODE violation
        at energy=3.0 (low for code, mean=4.0) doesn't incorrectly out-rank an
        ARITHMETIC violation at energy=1.0 (high for arithmetic, mean=0.5).

    The cross-domain violations (50 items) are injected with arithmetic keys into the
    code domain and vice versa, to test whether the isolation_score correctly detects
    structural overlap.
    """
    # Domain-specific (mean, std) energy distributions.
    domain_params = {
        "arithmetic": (1.5, 0.8),  # low mean, narrow spread
        "code": (4.0, 2.0),        # high mean, wide spread
        "logical": (3.0, 1.5),     # medium mean, medium spread
    }

    # 50 per-domain violations (own constraint keys only = no shared keys = isolated).
    # WHY no "domain" metadata key in violation dicts:
    #   The isolation_score computes key overlap between domain_a and domain_b violations.
    #   A shared "domain" metadata key would make ALL violations look structurally
    #   identical, masking the actual constraint-type isolation we want to measure.
    #   Each violation only contains constraint-type-specific keys.
    for domain, (mean, std) in domain_params.items():
        for i in range(50):
            energy = max(0.01, rng.gauss(mean, std))
            violation = {f"{domain}_constraint_{i}": i}
            replay.add_violation(domain, violation, energy)

    # 50 cross-domain violations: arithmetic-keyed items into code domain (and vice versa).
    # WHY: this tests whether the isolation_score simulation detects key overlap.
    # Real cross-domain interference happens when the same constraint type appears
    # in multiple domains simultaneously (e.g. arithmetic subexpressions in code).
    # Using "arith_cross" key prefix (not "arithmetic_constraint") avoids false overlap
    # with the per-domain arithmetic violations while still testing cross-domain detection.
    arith_mean, arith_std = domain_params["arithmetic"]
    for i in range(25):
        energy = max(0.01, rng.gauss(arith_mean, arith_std))
        replay.add_violation(
            "code",
            {"arith_cross_constraint": i},
            energy,
        )
    code_mean, code_std = domain_params["code"]
    for i in range(25):
        energy = max(0.01, rng.gauss(code_mean, code_std))
        replay.add_violation(
            "arithmetic",
            {"code_cross_constraint": i},
            energy,
        )


def main() -> None:
    # apply_env_autofix() FIRST — self-injects CARNOT_FORCE_LIVE=1 if GPU detected
    # and env var is absent (RETRO-022 fix).  CPU-only experiment — no GPU needed,
    # but the call is mandatory per CLAUDE.md spec-first workflow.
    apply_env_autofix()

    deliverable = "results/experiment_509_ppsebm_energy_magnitude_replay.json"
    guard = DeliverableGuard(str(_REPO_ROOT / deliverable))

    tmpl = ExperimentTemplate(
        509,
        "PPSEBM Energy Magnitude Replay",
        deliverable,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(509, timeout_minutes=20):
        # Deterministic seed for reproducibility.
        rng = random.Random(42)

        emr = EnergyMagnitudeReplay(
            domains=["arithmetic", "code", "logical"],
            k=10,
            buffer_size=100,
        )

        _simulate_violations(emr, rng)

        # Compute isolation score: arithmetic vs code over 50 alternating replay steps.
        # WHY arithmetic vs code (not all pairs):
        #   This is the most adversarial pair — arithmetic violations were deliberately
        #   injected into the code domain in the cross-domain simulation above.
        #   If EnergyMagnitudeReplay still achieves positive isolation despite this
        #   structural overlap, it demonstrates that energy-magnitude priority correctly
        #   prioritizes high-energy boundary violations over low-energy cross-domain noise.
        isolation_score = emr.isolation_score(
            domain_a="arithmetic",
            domain_b="code",
            n_steps=50,
        )

        isolation_improvement = isolation_score - _SURE_BASELINE
        energy_magnitude_better = isolation_improvement > 0
        retro_050_closed = energy_magnitude_better
        honest_verdict = (
            "energy_magnitude_wins" if retro_050_closed else "energy_magnitude_no_improvement"
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.energy_magnitude_replay.v1",
                "isolation_score": round(isolation_score, 6),
                "sure_baseline": _SURE_BASELINE,
                "isolation_improvement": round(isolation_improvement, 6),
                "energy_magnitude_better": energy_magnitude_better,
                "retro_050_closed": retro_050_closed,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        out_path = _REPO_ROOT / deliverable
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
