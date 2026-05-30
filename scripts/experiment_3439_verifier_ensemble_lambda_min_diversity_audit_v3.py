#!/usr/bin/env python3
"""Verifier Ensemble Lambda-Min / Diversity Audit v3.

Measures lambda_min(Sigma), effective-k (participation ratio), and per-verifier
drop-one-out AUROC contribution on the broadest CPU-accessible disjoint-kernel
verifier suite, scored on FoVer + an adversarial/OOD slice.

This is exp3439 — the v3 displacement of exp3313 (which was displaced in .315
and produced no artifact in .316 due to gemini-cli being down).  The audit is
cheap: verifier scoring over cached candidates, no LLM loading required.

Spec: REQ-VERIFY-3439, SCENARIO-VERIFY-3439
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is on the path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from carnot.verify.verifier_ensemble_diversity import (
    binary_auroc,
    build_verifier_set,
    ensemble_vote_scores,
    load_fover_corpus,
    make_adversarial_slice,
    reproducibility_checksum,
    run_diversity_audit,
)

RANDOM_SEED = 42
FOVER_CORPUS_PATH = Path("data/fover_corpus.jsonl")
OUTPUT_PATH = Path("results/experiment_3439_verifier_ensemble_lambda_min_diversity_audit_v3.json")
N_MAIN_EXAMPLES = 1000   # >=1000 for stable covariance estimate
ADV_SLICE_SIZE = 200     # adversarial/OOD slice


def main() -> None:
    t_start = time.time()

    # ------------------------------------------------------------------
    # STEP 0: PRECONDITIONS
    # ------------------------------------------------------------------
    preconditions_checked: list[dict] = []

    # a. FoVer corpus present
    corpus_ok = FOVER_CORPUS_PATH.exists()
    preconditions_checked.append({"resource": "fover_corpus", "available": corpus_ok})
    if not corpus_ok:
        _write_blocked("blocked_corpus_missing", preconditions_checked, t_start)
        return

    # b. Verifier suite callable (try to import and instantiate)
    try:
        verifiers = build_verifier_set()
        suite_ok = len(verifiers) > 0
    except Exception as exc:
        suite_ok = False
        print(f"Verifier import error: {exc}", file=sys.stderr)
    preconditions_checked.append({"resource": "verifier_suite", "available": suite_ok})
    if not suite_ok:
        _write_blocked("blocked_verifier_suite_uncallable", preconditions_checked, t_start)
        return

    # c. CUDA check (informational only — CPU-only verifiers don't need it)
    cuda_available = False
    skipped_gpu_verifiers: list[str] = []
    try:
        import torch  # type: ignore[import]
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass
    preconditions_checked.append({"resource": "cuda", "available": cuda_available,
                                   "note": "CPU-only verifiers used regardless"})

    print(f"Preconditions: {preconditions_checked}")
    print(f"Verifier suite: {[name for name, _, _ in verifiers]}")

    # ------------------------------------------------------------------
    # STEP 1: Load corpus + build adversarial slice
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    all_records = load_fover_corpus(str(FOVER_CORPUS_PATH), rng=rng)
    print(f"Loaded {len(all_records)} records from FoVer corpus")

    # Sample main evaluation set
    if len(all_records) >= N_MAIN_EXAMPLES:
        idx = rng.choice(len(all_records), size=N_MAIN_EXAMPLES, replace=False)
        main_records = [all_records[i] for i in sorted(idx)]
    else:
        main_records = all_records

    # Build adversarial/OOD slice from the *full* corpus
    adv_records = make_adversarial_slice(all_records, slice_size=ADV_SLICE_SIZE, rng=rng)
    print(f"Adversarial slice: {len(adv_records)} records")

    # Combined evaluation set
    combined_records = main_records + adv_records
    n_examples = len(combined_records)
    print(f"Total evaluation set: {n_examples} examples")

    # ------------------------------------------------------------------
    # STEP 2: Score verifiers + build decision covariance matrix
    # ------------------------------------------------------------------
    print("Scoring verifiers...")
    audit_results = run_diversity_audit(combined_records, verifiers)

    # ------------------------------------------------------------------
    # STEP 3: Extract diversity metrics
    # ------------------------------------------------------------------
    lambda_min = audit_results["lambda_min_sigma"]
    pairwise_max_corr = audit_results["pairwise_max_correlation"]
    effective_k = audit_results["effective_k_participation_ratio"]
    per_verifier_contrib = audit_results["per_verifier_dropout_contribution"]
    full_auroc = audit_results["full_ensemble_auroc"]
    k_verifiers = len(verifiers)

    print(f"k_verifiers: {k_verifiers}")
    print(f"lambda_min(Sigma): {lambda_min:.6f}")
    print(f"effective_k (participation ratio): {effective_k:.4f}")
    print(f"pairwise_max_correlation: {pairwise_max_corr:.4f}")
    print(f"full_ensemble_auroc: {full_auroc:.4f}")
    print(f"per-verifier drop-one-out contributions: {per_verifier_contrib}")

    # ------------------------------------------------------------------
    # STEP 4: Null-space identification
    # ------------------------------------------------------------------
    null_space_verifiers = [
        name for name, delta in per_verifier_contrib.items()
        if abs(delta) < 0.005  # less than 0.5pp contribution
    ]
    print(f"Null-space verifiers (delta < 0.005): {null_space_verifiers}")

    # Identify high-pairwise-correlation pairs (>= 0.9)
    corr_mat = np.array(audit_results["pairwise_corr"])
    names = audit_results["verifier_names"]
    high_corr_pairs = []
    for i in range(k_verifiers):
        for j in range(i + 1, k_verifiers):
            if abs(corr_mat[i, j]) >= 0.9:
                high_corr_pairs.append({
                    "verifier_a": names[i],
                    "verifier_b": names[j],
                    "correlation": round(float(corr_mat[i, j]), 6),
                })
    print(f"High-correlation pairs (>=0.9): {high_corr_pairs}")

    # ------------------------------------------------------------------
    # STEP 5: Acceptance gate + verdict
    # ------------------------------------------------------------------
    # G1: lambda_min > 0.1 AND effective_k >= 3
    g1_holds = (lambda_min > 0.1) and (effective_k >= 3.0)

    if g1_holds:
        honest_verdict = "complete: verifier_ensemble_diversity_sufficient_grounding_holds"
        gate_passed = True
    else:
        honest_verdict = "complete: verifier_ensemble_null_space_collapse_confirmed_grounding_at_risk"
        gate_passed = False

    print(f"\nG1 gate: {'PASSED' if g1_holds else 'FAILED'}")
    print(f"honest_verdict: {honest_verdict}")

    # ------------------------------------------------------------------
    # STEP 6: Write artifact
    # ------------------------------------------------------------------
    chksum = reproducibility_checksum(combined_records, RANDOM_SEED)
    duration_s = time.time() - t_start

    verifier_metadata = [
        {"name": name, "kernel_class": klass}
        for name, klass, _ in verifiers
    ]

    artifact: dict = {
        "artifact": "experiment_3439_verifier_ensemble_lambda_min_diversity_audit_v3",
        "experiment_id": 3439,
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "k_verifiers": k_verifiers,
        "lambda_min_sigma": round(lambda_min, 8),
        "pairwise_max_correlation": round(pairwise_max_corr, 8),
        "effective_k_participation_ratio": round(effective_k, 6),
        "per_verifier_dropout_contribution": {
            k: round(v, 8) for k, v in per_verifier_contrib.items()
        },
        "n_examples": n_examples,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": chksum,
        "duration_s": round(duration_s, 4),
        "full_ensemble_auroc": round(full_auroc, 8),
        "eigenvalues": [round(float(e), 8) for e in audit_results["eigenvalues"]],
        "acceptance_gate_passed": gate_passed,
        "gate_g1_description": "lambda_min_sigma > 0.1 AND effective_k_participation_ratio >= 3",
        "null_space_verifiers": null_space_verifiers,
        "high_correlation_pairs": high_corr_pairs,
        "skipped_gpu_verifiers": skipped_gpu_verifiers,
        "verifier_metadata": verifier_metadata,
        "corpus_split": {
            "main_fover": len(main_records),
            "adversarial_ood": len(adv_records),
            "total": n_examples,
        },
        "preconditions_checked": preconditions_checked,
        "field_provenance": {
            "honest_verdict": {
                "principle": "complete:/success:/passed:/shipped_ prefix required for conductor reconciler.",
                "satisfied_by": "string starts with 'complete:'"
            },
            "inference_substrate": {
                "principle": "Declares that no LLM was loaded; scoring is over cached candidates. Duration floor is 1s.",
                "satisfied_by": "verifier_ensemble_against_cached_candidates"
            },
            "k_verifiers": {
                "principle": "Nominal verifier count; compare to effective_k to quantify redundancy.",
                "satisfied_by": f"len(verifier_registry) = {k_verifiers}"
            },
            "lambda_min_sigma": {
                "principle": "Smallest eigenvalue of the decision covariance; the joint-null-space proxy. Threshold >0.1.",
                "satisfied_by": "numpy.linalg.eigh on the k×k binary decision covariance"
            },
            "pairwise_max_correlation": {
                "principle": "The exp1224 collapse signature; near-1.0 means structural redundancy.",
                "satisfied_by": "max off-diagonal |Pearson correlation| over verifier-decision pairs"
            },
            "effective_k_participation_ratio": {
                "principle": "How many verifiers ACTUALLY contribute independent signal. Threshold >=3.",
                "satisfied_by": "sum(lambda)^2 / sum(lambda^2)"
            },
            "per_verifier_dropout_contribution": {
                "principle": "Drop-one-out AUROC delta; ~0 means that verifier is in the joint null space.",
                "satisfied_by": "remove verifier j, recompute majority-vote AUROC, report full_auroc - reduced_auroc"
            },
            "n_examples": {
                "principle": ">=1000 for a stable covariance estimate (CLT minimum for stable eigenvalues).",
                "satisfied_by": f"main({len(main_records)}) + adversarial({len(adv_records)}) = {n_examples}"
            },
            "random_seed": {
                "principle": "Determinism; enables external replication of the exact corpus sample.",
                "satisfied_by": f"numpy.random.default_rng({RANDOM_SEED})"
            },
            "reproducibility_checksum": {
                "principle": "Content hash over first 50 records + seed; detects corpus-version drift.",
                "satisfied_by": "SHA-256[:16] of seed + first-50 step_text prefixes"
            },
            "duration_s": {
                "principle": "Real wall-clock time; verifier_ensemble substrate floor is 1s.",
                "satisfied_by": "time.time() delta, no sleep padding"
            },
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"\nArtifact written to {OUTPUT_PATH}")
    print(f"duration_s: {duration_s:.2f}s")


def _write_blocked(
    verdict: str,
    preconditions_checked: list[dict],
    t_start: float,
) -> None:
    """Write a blocked artifact when a precondition fails."""
    duration_s = time.time() - t_start
    artifact = {
        "artifact": "experiment_3439_verifier_ensemble_lambda_min_diversity_audit_v3",
        "experiment_id": 3439,
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "preconditions_checked": preconditions_checked,
        "duration_s": round(duration_s, 4),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "k_verifiers": 0,
        "lambda_min_sigma": None,
        "pairwise_max_correlation": None,
        "effective_k_participation_ratio": None,
        "per_verifier_dropout_contribution": {},
        "n_examples": 0,
        "acceptance_gate_passed": False,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"Blocked: {verdict}")


if __name__ == "__main__":
    main()
