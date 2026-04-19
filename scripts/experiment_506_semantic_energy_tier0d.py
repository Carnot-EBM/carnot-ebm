#!/usr/bin/env python3
"""Experiment 506: BoltzmannSemanticEnergy Tier 0d — Semantic Cluster Hallucination Detection.

**Researcher summary (arXiv 2508.14496, August 2025):**
    "Semantic Energy: Detecting LLM Hallucinations via Boltzmann-Weighted Cluster Energy"
    introduces a hallucination detection method that combines two complementary signals:

    1. SEMANTIC CLUSTERING: group tokens by meaning (cosine-similarity k-means) to
       identify HOW MANY distinct semantic alternatives the model is entertaining.

    2. BOLTZMANN ENERGY: for each cluster, compute exp(-energy/T) where energy = -mean_logit/T.
       This measures HOW CONFIDENT the model is about each semantic alternative.

    The combined signal overcomes Semantic Entropy's weakness in high-confidence failures:
    SE measures IF the model is uncertain (spread of meanings), but Boltzmann energy
    captures WHETHER the model is confident about each meaning (even when overconfident).

    arXiv 2508.14496 result: 13% average AUROC improvement over SE in cases where
    SE is overconfident (model gives same wrong answer repeatedly — SE=low, energy=high).

**Cascade position:**
    Tier 0a: ThinkProbe        — generative CoT verdict
    Tier 0b: SpilledEnergy     — per-token NLL discrepancy
    Tier 0c: NUPProbe          — continuation entropy (AUC=0.600)
    Tier 0d: BoltzmannSemantic — semantic cluster energy (THIS EXPERIMENT)
    Tier 1:  SinkProbe         — attention sink concentration
    Tier 3:  Ising             — constraint-based energy minimisation

**Synthetic benchmark design:**
    50 CORRECT responses: low-variance logits (model confident, low energy)
        token_logits = {token_i: base_logit + small_noise}
        base_logit = +2.0, noise ∈ [-0.2, +0.2]

    50 HALLUCINATED responses: high-variance logits (model uncertain, high energy)
        token_logits = {token_i: sampled from U(-5, +5)}

    Oracle baseline (SpilledEnergy): loaded from Exp 433 result JSON if available,
    else uses 0.600 (the NUP-Probe AUC from the cascade literature).

**honest_verdict semantics:**
    'semantic_energy_viable'         — auroc > spilled_energy_baseline
    'semantic_energy_no_improvement' — auroc <= spilled_energy_baseline

Spec: REQ-VERIFY-101, REQ-VERIFY-102, REQ-VERIFY-103
SCENARIO-VERIFY-134, SCENARIO-VERIFY-135, SCENARIO-VERIFY-136
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root setup — must precede carnot imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# STEP 1: apply_env_autofix() FIRST, before any CUDA import
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_env_fix = apply_env_autofix()

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.semantic_energy_boltzmann import BoltzmannSemanticEnergy  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_506_semantic_energy_tier0d.json"
_SPILLED_ENERGY_BASELINE_DEFAULT = 0.600  # NUPProbe AUC fallback

# Seed for reproducible synthetic data generation
_RNG_SEED = 506


def _load_spilled_energy_baseline() -> float:
    """Load SpilledEnergy AUROC from Exp 433 result JSON, or return default.

    Tries to read results/experiment_433_spilled_energy_detector.json.
    If absent or malformed, returns _SPILLED_ENERGY_BASELINE_DEFAULT (0.600).
    This avoids hard-coding a number that may have been updated by a real run.
    """
    candidate_paths = [
        _REPO_ROOT / "results" / "experiment_433_spilled_energy_detector.json",
        _REPO_ROOT / "results" / "experiment_433_spilled_energy.json",
    ]
    for path in candidate_paths:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                auroc = data.get("auroc") or data.get("spilled_energy_auroc")
                if auroc is not None:
                    return float(auroc)
            except Exception:
                pass
    return _SPILLED_ENERGY_BASELINE_DEFAULT


def _generate_synthetic_data(
    n_correct: int = 50,
    n_hallucinated: int = 50,
    n_tokens: int = 20,
    rng: random.Random | None = None,
) -> tuple[list[tuple[str, dict[str, float]]], list[bool]]:
    """Generate synthetic (response, token_logits, is_correct) triples.

    CORRECT responses: tokens with logits drawn from N(2.0, 0.2^2).
        A confident model has most probability mass on a few tokens, so logits
        cluster around a high value with low variance — low Boltzmann energy.

    HALLUCINATED responses: tokens with logits drawn from U(-5, +5).
        An uncertain model spreads probability mass widely — logits span the full
        range, creating high-variance clusters and high Boltzmann energy.

    Returns:
        responses: list of (response_text, token_logits) tuples
        ground_truth: parallel list of bool (True=correct, False=hallucinated)
    """
    if rng is None:
        rng = random.Random(_RNG_SEED)

    responses: list[tuple[str, dict[str, float]]] = []
    ground_truth: list[bool] = []

    # Vocabulary: reuse a fixed set of tokens for reproducibility
    vocab = [f"token_{i:03d}" for i in range(n_tokens)]

    for idx in range(n_correct):
        # Correct: low-variance logits around +2.0
        logits = {t: rng.gauss(2.0, 0.2) for t in vocab}
        responses.append((f"correct response {idx}: The answer is 42.", logits))
        ground_truth.append(True)

    for idx in range(n_hallucinated):
        # Hallucinated: high-variance logits sampled uniformly from [-5, +5]
        logits = {t: rng.uniform(-5.0, 5.0) for t in vocab}
        responses.append((f"hallucinated response {idx}: The answer is uncertain.", logits))
        ground_truth.append(False)

    # Shuffle so correct and hallucinated are interleaved
    combined = list(zip(responses, ground_truth))
    rng.shuffle(combined)
    responses_out, gt_out = zip(*combined) if combined else ([], [])
    return list(responses_out), list(gt_out)


def main() -> None:
    """Run Experiment 506: BoltzmannSemanticEnergy Tier 0d benchmark."""

    # ------------------------------------------------------------------
    # Gate chain: Watchdog → Template → DeliverableGuard
    # ------------------------------------------------------------------
    with ExperimentTimeoutWatchdog(506, timeout_minutes=20):
        tmpl = ExperimentTemplate(
            506,
            "Semantic Energy Tier 0d",
            _DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()
        guard = DeliverableGuard(str(_REPO_ROOT / _DELIVERABLE))

        # ------------------------------------------------------------------
        # Load SpilledEnergy baseline AUROC
        # ------------------------------------------------------------------
        spilled_energy_baseline = _load_spilled_energy_baseline()

        # ------------------------------------------------------------------
        # Generate 100 synthetic responses (50 correct, 50 hallucinated)
        # ------------------------------------------------------------------
        rng = random.Random(_RNG_SEED)
        responses, ground_truth = _generate_synthetic_data(
            n_correct=50,
            n_hallucinated=50,
            n_tokens=20,
            rng=rng,
        )

        # ------------------------------------------------------------------
        # Run BoltzmannSemanticEnergy benchmark
        # ------------------------------------------------------------------
        bse = BoltzmannSemanticEnergy(n_clusters=10, temperature=1.0)
        bench = bse.benchmark(responses, ground_truth)

        auroc: float = bench["auroc"]
        skip_rate: float = bench["skip_rate"]
        semantic_energy_better: bool = auroc > spilled_energy_baseline

        honest_verdict = (
            "semantic_energy_viable"
            if semantic_energy_better
            else "semantic_energy_no_improvement"
        )

        # ------------------------------------------------------------------
        # Build artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "result_schema": "carnot.semantic_energy.v1",
                "auroc": auroc,
                "skip_rate": skip_rate,
                "spilled_energy_baseline": spilled_energy_baseline,
                "semantic_energy_better": semantic_energy_better,
                "honest_verdict": honest_verdict,
                "n_correct": bench["n_correct"],
                "n_hallucinated": bench["n_hallucinated"],
                "n_total": bench["n_total"],
                "n_clusters": bse.n_clusters,
                "temperature": bse.temperature,
                "env_autofix_gpu_detected": _env_fix.gpu_detected,
            },
            status="success",
        )

        # ------------------------------------------------------------------
        # Write deliverable
        # ------------------------------------------------------------------
        out_path = _REPO_ROOT / _DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

        print(f"Exp 506 complete: auroc={auroc:.4f}, baseline={spilled_energy_baseline:.4f}, "
              f"verdict={honest_verdict}")

        # ------------------------------------------------------------------
        # Update _bmad/architecture.md Tier 0d entry
        # ------------------------------------------------------------------
        _update_architecture_md()

        # FINAL LINE: assert deliverable was written
        tmpl.assert_deliverable_written()


def _update_architecture_md() -> None:
    """Add Tier 0d: BoltzmannSemanticEnergy to the Verification Pipeline Tiers table.

    Reads _bmad/architecture.md, finds the tiers table, and inserts the Tier 0d row
    if it is not already present.  No-op if the entry already exists.
    """
    arch_path = _REPO_ROOT / "_bmad" / "architecture.md"
    if not arch_path.exists():
        return

    content = arch_path.read_text()

    tier_0d_marker = "Tier 0d"
    if tier_0d_marker in content:
        return  # Already present

    # Insert after the Tier 0c row (NUPProbe)
    tier_0c_marker = "Tier 0c"
    if tier_0c_marker not in content:
        return  # Table structure unexpected — skip safely

    tier_0d_row = (
        "| Tier 0d | BoltzmannSemanticEnergy | Semantic cluster Boltzmann-weighted energy "
        "(arXiv 2508.14496) | CPU | Exp 506 |\n"
    )

    # Find the line after the Tier 0c row and insert
    lines = content.split("\n")
    new_lines = []
    inserted = False
    for line in lines:
        new_lines.append(line)
        if not inserted and tier_0c_marker in line and "|" in line:
            new_lines.append(tier_0d_row.rstrip("\n"))
            inserted = True

    if inserted:
        arch_path.write_text("\n".join(new_lines))


if __name__ == "__main__":
    main()
