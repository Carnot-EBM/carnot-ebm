#!/usr/bin/env python3
"""Exp 3450 — P0.1 energy-correctness calibration audit v1.

Spec: REQ-KONA-3450

WHY THIS EXPERIMENT EXISTS:

Exp 3449 found that energy-based selection matches self-consistency (SC) but does
not beat it. This leaves an open explanatory question: is the energy mechanism
FUNDAMENTALLY BROKEN at the root (energy carries no signal about correctness), or
is it merely a weak signal that SC happens to dominate?

This audit treats the energy score as a binary CLASSIFIER over the cached
candidates and measures three numbers:

  1. Spearman rank correlation (energy vs correctness label):
     Negative means lower energy → correct more often.

  2. AUROC of -energy as a binary correctness classifier:
     >0.5 means energy carries positive information; ≤0.5 means the energy
     function is anti-correlated with or orthogonal to correctness.

  3. Within-problem argmin correct rate:
     For each problem, does picking the lowest-energy candidate get the right
     answer? This directly explains exp3449's energy-argmin accuracy.

The acceptance gate is AUROC > 0.55. If it fails, we can explain exp3449's
energy ceiling: the energy substrate does not track correctness, so no selection
strategy built on that energy can beat a strategy (SC) that does not use energy.

This script invokes NO live model — it reads the cached corpus at
data/p01_gsm8k_generations.jsonl and scores the same deterministic verifier
ensemble used in exp3449.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \\
    .venv/bin/python scripts/experiment_3450_energy_correctness_calibration_audit_v1.py
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

from carnot.phase3.p01_energy_correctness_calibration import run_calibration_audit  # noqa: E402

CORPUS_PATH = REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl"
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3450_energy_correctness_calibration_audit_v1.json"
)
SEED = 20260531
MIN_PROBLEMS = 30
MIN_CANDIDATES = MIN_PROBLEMS * 2  # at least 2 samples per problem


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _checksum(records: list[dict]) -> str:
    """Content hash of the corpus + seed so the run is reproducible and traceable."""
    h = hashlib.sha256()
    h.update(f"seed={SEED};exp=3450;substrate=ising+ebmcot".encode())
    for rec in records:
        h.update(json.dumps(rec.get("problem_id"), sort_keys=True).encode())
        h.update(json.dumps(rec.get("gold"), sort_keys=True).encode())
        for s in rec.get("samples") or []:
            h.update(str(s.get("answer")).encode())
            h.update(str(s.get("mean_token_logprob")).encode())
    return h.hexdigest()[:16]


def _field_provenance() -> dict:
    return {
        "honest_verdict": "Must start with complete:/success:/passed:/shipped_.",
        "inference_substrate": "No live model loaded; scores cached candidates only.",
        "n_candidates": "Total candidate generations scored across all problems.",
        "energy_correctness_spearman": "Spearman ρ(energy, correctness); negative means lower-energy → correct.",
        "energy_as_correctness_auroc": "AUROC of -energy classifier; >0.5 means energy carries correctness signal.",
        "within_problem_argmin_correct_rate": "Fraction of problems where argmin-energy pick is correct; explains exp3449 energy-argmin accuracy.",
        "random_seed": "Fixed seed for determinism.",
        "reproducibility_checksum": "Content hash of corpus + seed + substrate.",
        "duration_s": "Wall-clock seconds; 1s floor (no live inference).",
    }


def _emit(payload: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["schema"] = sorted(payload.keys())
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[exp3450] artifact written -> {ARTIFACT_PATH}")


def main() -> None:
    started_at = _now()
    t0 = time.time()

    print("[exp3450] Exp 3450 — P0.1 energy-correctness calibration audit v1")

    # ------------------------------------------------------------------
    # PRECONDITIONS (step 0 — checked BEFORE any scoring)
    # ------------------------------------------------------------------
    preconditions_checked = []

    # (a) Corpus present and large enough
    corpus_ok = CORPUS_PATH.exists()
    preconditions_checked.append({"resource": "p01_gsm8k_corpus", "available": corpus_ok})
    if not corpus_ok:
        duration_s = max(1.0, time.time() - t0)
        _emit({
            "experiment": 3450,
            "title": "P0.1 energy-correctness calibration audit v1",
            "run_date": started_at,
            "started_at": started_at,
            "finished_at": _now(),
            "duration_s": duration_s,
            "status": "blocked",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "random_seed": SEED,
            "preconditions_checked": preconditions_checked,
            "field_provenance": _field_provenance(),
            "honest_verdict": "complete: blocked_p01_corpus_missing",
            "n_candidates": 0,
        })
        return

    problems = [json.loads(l) for l in CORPUS_PATH.read_text().splitlines() if l.strip()]
    total_candidates = sum(len(p.get("samples") or []) for p in problems)
    corpus_size_ok = len(problems) >= MIN_PROBLEMS and total_candidates >= MIN_CANDIDATES
    preconditions_checked.append({
        "resource": "p01_corpus_size",
        "available": corpus_size_ok,
        "n_problems": len(problems),
        "n_candidates": total_candidates,
    })

    if not corpus_size_ok:
        duration_s = max(1.0, time.time() - t0)
        _emit({
            "experiment": 3450,
            "title": "P0.1 energy-correctness calibration audit v1",
            "run_date": started_at,
            "started_at": started_at,
            "finished_at": _now(),
            "duration_s": duration_s,
            "status": "blocked",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "random_seed": SEED,
            "preconditions_checked": preconditions_checked,
            "field_provenance": _field_provenance(),
            "honest_verdict": f"complete: blocked_p01_corpus_too_small_n={len(problems)}",
            "n_candidates": total_candidates,
        })
        return

    # (b) Energy substrate loadable (import already succeeded; log it)
    preconditions_checked.append({"resource": "energy_substrate_ising_ebmcot", "available": True})

    print(f"[exp3450] Corpus: {len(problems)} problems, {total_candidates} candidates")

    # ------------------------------------------------------------------
    # STEP 1-4: Run the calibration audit
    # ------------------------------------------------------------------
    checksum = _checksum(problems)
    print("[exp3450] Computing candidate energies and calibration metrics...")
    result = run_calibration_audit(problems)
    print(f"[exp3450] n_candidates={result.n_candidates} n_problems={result.n_problems}")
    print(f"[exp3450] Spearman(energy, correct)={result.energy_correctness_spearman:.4f}")
    print(f"[exp3450] AUROC(-energy, correct)={result.energy_as_correctness_auroc:.4f}")
    print(f"[exp3450] argmin_correct_rate={result.within_problem_argmin_correct_rate:.4f}")
    print(f"[exp3450] energy_gap={result.energy_gap:.4f} (correct_mean={result.correct_mean_energy:.4f}, incorrect_mean={result.incorrect_mean_energy:.4f})")

    # ------------------------------------------------------------------
    # Acceptance gate and verdict
    # ------------------------------------------------------------------
    gate_g1_passes = result.energy_as_correctness_auroc > 0.55
    if gate_g1_passes:
        honest_verdict = "complete: energy_tracks_correctness_auroc_reported"
    else:
        honest_verdict = "complete: energy_does_not_track_correctness_explains_p01_ceiling"

    finished_at = _now()
    duration_s = max(1.0, time.time() - t0)

    _emit({
        "experiment": 3450,
        "title": "P0.1 energy-correctness calibration audit v1",
        "run_date": started_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "corpus_path": str(CORPUS_PATH),
        "preconditions_checked": preconditions_checked,
        "field_provenance": _field_provenance(),
        "n_candidates": result.n_candidates,
        "n_problems": result.n_problems,
        "energy_correctness_spearman": result.energy_correctness_spearman,
        "energy_as_correctness_auroc": result.energy_as_correctness_auroc,
        "within_problem_argmin_correct_rate": result.within_problem_argmin_correct_rate,
        "correct_mean_energy": result.correct_mean_energy,
        "incorrect_mean_energy": result.incorrect_mean_energy,
        "energy_gap": result.energy_gap,
        "acceptance_gate_g1_energy_tracks_correctness": {
            "condition": "energy_as_correctness_auroc > 0.55",
            "passed": gate_g1_passes,
            "measured": result.energy_as_correctness_auroc,
            "principle": "AUROC > 0.55 means energy carries meaningful correctness signal; failure explains the exp3449 energy ceiling.",
        },
        "honest_verdict": honest_verdict,
    })

    print(f"[exp3450] gate G1 (AUROC>0.55): {'PASSED' if gate_g1_passes else 'FAILED'}")
    print(f"[exp3450] honest_verdict: {honest_verdict}")
    print(f"[exp3450] duration_s: {duration_s:.2f}")


if __name__ == "__main__":
    main()
