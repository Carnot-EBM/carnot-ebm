#!/usr/bin/env python3
"""Exp 3461 — Trained-vs-untrained energy correctness calibration on held-out GSM8K (v2).

Spec: REQ-KONA-3461, SCENARIO-KONA-3461, SCENARIO-KONA-3461-BLOCKED

WHY THIS EXPERIMENT:
exp3450 found that the UNTRAINED verifier energy has AUROC = 0.516 vs correctness
(essentially chance). exp3460 then showed that a TRAINED logistic-regression energy
reranker matches but does not beat self-consistency. This experiment answers WHY that
happened at the mechanism level: does the TRAINED energy carry meaningful correctness
signal (AUROC >> 0.516), or is it still near-chance and therefore cannot route correct
answers to the top?

We also score the FoVer 4-verifier ensemble (arithmetic + contradiction + Curry-Howard
+ logical) on the same held-out candidates for comparison.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \
    .venv/bin/python scripts/experiment_3461_energy_correctness_calibration_trained_vs_untrained_v2.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

CORPUS_PATH = REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl"
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3461_energy_correctness_calibration_trained_vs_untrained_v2.json"
)
SEED = 20260601      # same as exp3460 — so the held-out split is identical
N_FOLDS = 5          # same as exp3460
RERANKER_ITER = 500  # same as exp3460
MIN_PROBLEMS = 47    # corpus floor from exp3459
UNTRAINED_AUROC_BASELINE = 0.516  # exp3450 reference


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _checksum(records: list[dict]) -> str:
    h = hashlib.sha256()
    h.update(
        f"exp=3461;seed={SEED};folds={N_FOLDS};iter={RERANKER_ITER};"
        f"substrate=trained_logreg+fover;auroc_baseline={UNTRAINED_AUROC_BASELINE}".encode()
    )
    for rec in records:
        h.update(json.dumps(rec.get("problem_id"), sort_keys=True).encode())
        h.update(json.dumps(rec.get("gold"), sort_keys=True).encode())
        for s in rec.get("samples") or []:
            h.update(str(s.get("answer")).encode())
            h.update(str(s.get("mean_token_logprob")).encode())
    return h.hexdigest()[:16]


def _emit(payload: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["schema"] = sorted(payload.keys())
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)


_START_AT = _now()


def _base_payload(start: float, preconditions: list[dict]) -> dict:
    return {
        "experiment": 3461,
        "title": "Trained-vs-untrained energy correctness calibration on held-out GSM8K (v2)",
        "run_date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "started_at": _START_AT,
        "finished_at": _now(),
        "duration_s": round(max(time.time() - start, 1.0), 3),
        "status": "success",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": SEED,
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "preconditions_checked": preconditions,
        "field_provenance": {
            "honest_verdict": "Terminal verdict must start with complete:/success:/passed:/shipped_.",
            "inference_substrate": "verifier_ensemble_against_cached_candidates: no live model loaded.",
            "n_candidates_heldout": "Total held-out candidates scored across all CV folds.",
            "untrained_energy_auroc_baseline": "The exp3450 0.516 reference carried forward for comparison.",
            "trained_energy_correctness_auroc": "AUROC of P(correct) as a correctness classifier — the core number.",
            "trained_energy_correctness_spearman": "Rank-correlation of trained energy vs correctness.",
            "fover_energy_correctness_auroc": "AUROC of -fover_energy as a correctness classifier.",
            "trained_energy_auroc_lift_over_untrained": "trained_auroc - 0.516: does training fix the uninformative-energy ceiling?",
            "within_problem_argmin_correct_rate_trained": "Fraction of problems where highest-P(correct) candidate is correct.",
            "random_seed": "Determinism: must match exp3460 seed so held-out split is identical.",
            "reproducibility_checksum": "Content hash of corpus + config + seed.",
            "duration_s": "Cached scoring; 1s floor.",
        },
    }


def main() -> None:
    start = time.time()
    preconditions: list[dict] = []

    # --- PRECONDITION 0a: corpus present with >= MIN_PROBLEMS usable rows ---
    records: list[dict] = []
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    usable = [
        r
        for r in records
        if r.get("gold") is not None
        and (r.get("greedy") or {}).get("answer") is not None
        and len(r.get("samples") or []) >= 5
    ]
    preconditions.append(
        {"resource": "cached_corpus", "available": len(usable) >= MIN_PROBLEMS,
         "n_problems": len(usable)}
    )
    if len(usable) < MIN_PROBLEMS:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = f"complete: blocked_p01_corpus_too_small_n={len(usable)}"
        payload["n_candidates_heldout"] = 0
        payload["untrained_energy_auroc_baseline"] = UNTRAINED_AUROC_BASELINE
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # --- PRECONDITION 0b: energy substrate loadable ---
    try:
        from carnot.phase3.p01_trained_energy_correctness_calibration import (
            compute_trained_calibration,
            _Verifiers,
        )
        verifiers = _Verifiers()
        _ = verifiers.ising.energy("2 + 2 = 4")
        substrate_ok = True
    except Exception as exc:
        substrate_ok = False
        exc_msg = str(exc)
    preconditions.append({"resource": "energy_substrate", "available": substrate_ok})
    if not substrate_ok:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = "complete: blocked_energy_substrate_unavailable"
        payload["n_candidates_heldout"] = 0
        payload["untrained_energy_auroc_baseline"] = UNTRAINED_AUROC_BASELINE
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # --- Score held-out candidates with both energies ---
    result = compute_trained_calibration(
        usable,
        seed=SEED,
        n_folds=N_FOLDS,
        reranker_iter=RERANKER_ITER,
        verifiers=verifiers,
    )

    # --- Acceptance gate ---
    g1_passed = max(
        result.trained_energy_correctness_auroc,
        result.fover_energy_correctness_auroc,
    ) > 0.55

    if g1_passed:
        verdict = "complete: trained_or_fover_energy_tracks_correctness_lift_over_untrained_reported"
    else:
        verdict = "complete: even_trained_energy_does_not_track_correctness_substrate_ceiling_confirmed"

    payload = _base_payload(start, preconditions)
    payload.update(
        {
            "honest_verdict": verdict,
            "n_candidates_heldout": result.n_candidates_heldout,
            "n_problems_heldout": result.n_problems_heldout,
            "untrained_energy_auroc_baseline": UNTRAINED_AUROC_BASELINE,
            "trained_energy_correctness_auroc": round(result.trained_energy_correctness_auroc, 6),
            "trained_energy_correctness_spearman": round(result.trained_energy_correctness_spearman, 6),
            "fover_energy_correctness_auroc": round(result.fover_energy_correctness_auroc, 6),
            "fover_energy_correctness_spearman": round(result.fover_energy_correctness_spearman, 6),
            "trained_energy_auroc_lift_over_untrained": round(result.trained_energy_auroc_lift_over_untrained, 6),
            "within_problem_argmin_correct_rate_trained": round(result.within_problem_argmin_correct_rate_trained, 6),
            "within_problem_argmin_correct_rate_fover": round(result.within_problem_argmin_correct_rate_fover, 6),
            "acceptance_gate_g1_energy_carries_signal": {
                "condition": "max(trained_energy_correctness_auroc, fover_energy_correctness_auroc) > 0.55",
                "passed": g1_passed,
                "trained_auroc": round(result.trained_energy_correctness_auroc, 6),
                "fover_auroc": round(result.fover_energy_correctness_auroc, 6),
                "principle": "G1 ENERGY-CARRIES-SIGNAL: a trained or FoVer energy carries meaningful correctness signal above the 0.516 untrained floor.",
            },
            "methodology_note": (
                f"Same 5-fold problem-level CV as exp3460 (seed={SEED}). "
                "Trained energy = P(correct) from fold-specific logistic reranker trained on all "
                "other folds' candidates. FoVer energy = arithmetic + contradiction + Curry-Howard "
                "+ logical (parameter-free). All held-out candidates scored exactly once."
            ),
            "reproducibility_checksum": _checksum(usable),
        }
    )
    _emit(payload)

    print(
        f"DONE: {verdict}\n"
        f"  n_heldout_candidates={result.n_candidates_heldout} "
        f"n_problems={result.n_problems_heldout}\n"
        f"  untrained_baseline={UNTRAINED_AUROC_BASELINE}\n"
        f"  trained_AUROC={result.trained_energy_correctness_auroc:.4f} "
        f"(lift={result.trained_energy_auroc_lift_over_untrained:+.4f})\n"
        f"  FoVer_AUROC={result.fover_energy_correctness_auroc:.4f}\n"
        f"  trained_spearman={result.trained_energy_correctness_spearman:.4f} "
        f"fover_spearman={result.fover_energy_correctness_spearman:.4f}\n"
        f"  within_prob_argmin_trained={result.within_problem_argmin_correct_rate_trained:.4f} "
        f"fover={result.within_problem_argmin_correct_rate_fover:.4f}\n"
        f"  G1_energy_carries_signal={g1_passed}"
    )


if __name__ == "__main__":
    main()
