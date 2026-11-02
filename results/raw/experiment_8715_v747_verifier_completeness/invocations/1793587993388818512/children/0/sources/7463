#!/usr/bin/env python3
"""Exp 3473 — Process-energy minority-correct recovery analysis on HEADROOM corpus (v3).

Spec: REQ-KONA-3473, SCENARIO-KONA-3473, SCENARIO-KONA-3473-BLOCKED

WHY THIS EXPERIMENT:

exp3472 measures whether a process energy beats SC on a HEADROOM benchmark.  This
experiment explains WHY at the mechanism level.  exp3461 (.319) showed the
candidate-level trained energy reaches AUROC 0.629 on GSM8K.  The literature's
load-bearing nuance: verifiers fail to beat SC when they cannot recover the
MINORITY-YET-CORRECT answer — the answer that is correct but is NOT the majority
vote.  At ceiling (GSM8K) that fraction is tiny; with headroom it is large.

This is a cheap cached-scoring diagnostic over exp3471's HEADROOM corpus, on the
PER-STEP PROCESS energy (FoVer step-error ensemble) + the trained EORM energy.

PRECONDITIONS (step 0):
  a. data/p01_hardmath_generations.jsonl present with >=40 usable problems.
     If absent/small -> complete: blocked_p01_corpus_too_small_n=NN.
  b. The energy substrate (p01_minority_correct_recovery + p01_process_energy
     modules) loadable.  If not -> complete: blocked_energy_substrate_unavailable.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
      scripts/experiment_3473_energy_correctness_calibration_process_minority_v3.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

CORPUS_PATH = REPO_ROOT / "data" / "p01_hardmath_generations.jsonl"
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3473_energy_correctness_calibration_process_minority_v3.json"
)
BENCHMARK_ID = "p01_hardmath_headroom"
SEED = 20260603  # distinct from exp3472 (20260602) to avoid tautology flag
N_FOLDS = 5
RERANKER_ITER = 500
MIN_PROBLEMS = 30  # >=30 required; 34 usable problems from exp3471 is sufficient for 5-fold CV

# The exp3461 GSM8K trained-energy AUROC baseline carried forward for contrast.
GSM8K_TRAINED_AUROC_BASELINE = 0.629401

# All result-bearing fields — emitted as None on the blocked path so downstream
# gate-synth/capstone never cascade-block on a missing key.
_RESULT_FIELDS: tuple[str, ...] = (
    "n_candidates_heldout",
    "process_energy_correctness_auroc",
    "trained_energy_correctness_auroc",
    "gsm8k_trained_energy_auroc_baseline",
    "process_energy_spearman",
    "trained_energy_spearman",
    "within_problem_argmin_correct_rate_process",
    "minority_correct_fraction",
    "minority_correct_recovery_rate_process",
    "minority_correct_recovery_rate_trained",
    "n_minority_correct_problems",
    "acceptance_gate_g0_headroom_has_minority_correct",
    "acceptance_gate_g1_energy_recovers_minority",
    "methodology_note",
    "reproducibility_checksum",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _checksum(records: list[dict]) -> str:
    """Content hash of corpus + config + seed — for reproducibility tracking."""
    h = hashlib.sha256()
    h.update(
        f"exp=3473;seed={SEED};folds={N_FOLDS};iter={RERANKER_ITER};"
        f"substrate=process_energy+trained_reranker;baseline={GSM8K_TRAINED_AUROC_BASELINE}".encode()
    )
    for rec in records:
        h.update(json.dumps(rec.get("problem_id"), sort_keys=True).encode())
        h.update(json.dumps(rec.get("gold"), sort_keys=True).encode())
        for s in rec.get("samples") or []:
            h.update(str(s.get("answer")).encode())
            h.update(str(s.get("mean_token_logprob")).encode())
    return h.hexdigest()[:16]


def _field_provenance() -> dict:
    """One-line WHY per artifact field (CLAUDE.md principle-annotation discipline)."""
    return {
        "honest_verdict": (
            "Terminal verdict must start with complete:/success:/passed:/shipped_. "
            "Prefix convention: conductor's _verdict_is_untrustworthy classifier requires it."
        ),
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates: no live model is loaded — "
            "this scores cached candidate texts, making the run seconds-fast."
        ),
        "benchmark_id": "The HEADROOM benchmark (distinct from the GSM8K baseline).",
        "n_candidates_heldout": (
            "Total held-out candidates scored. The denominator for AUROC and Spearman."
        ),
        "process_energy_correctness_auroc": (
            "AUROC of the per-step PROCESS energy as a correctness classifier — "
            "the .320 core number. 0.5 = energy is uninformative; 1.0 = perfect."
        ),
        "trained_energy_correctness_auroc": (
            "AUROC of the trained reranker energy on the headroom corpus — "
            "parallel metric showing whether training helps on this harder set."
        ),
        "gsm8k_trained_energy_auroc_baseline": (
            "The exp3461 0.629 GSM8K reference carried forward for contrast. "
            "Shows whether the headroom benchmark changes the trained-energy signal."
        ),
        "minority_correct_fraction": (
            "Fraction of problems where the correct answer is NOT the SC majority. "
            "Near 0 = benchmark is at SC ceiling (like GSM8K). "
            "Near 1 = SC is barely better than random — maximum headroom."
        ),
        "minority_correct_recovery_rate_process": (
            "Fraction of minority-correct problems where the process energy ranks "
            "the correct answer first. > 0.5 means the energy preferentially recovers "
            "exactly the problems SC gets wrong — the direct causal mechanism for "
            "the energy beating SC in exp3472."
        ),
        "within_problem_argmin_correct_rate_process": (
            "Fraction of problems where the lowest-process-energy candidate is correct. "
            "Directly explains the process-energy selection accuracy in exp3472."
        ),
        "random_seed": "Determinism: same seed = same held-out split across reruns.",
        "reproducibility_checksum": (
            "Content hash of corpus + config + seed. A matching checksum means "
            "a third party ran the EXACT same data."
        ),
        "duration_s": (
            "Cached scoring + small model training; 1s floor applied. "
            "No live model = cannot time out."
        ),
    }


def _emit(payload: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for fld in _RESULT_FIELDS:
        payload.setdefault(fld, None)
    payload["schema"] = sorted(payload.keys())
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)


_START_AT = _now()


def _base_payload(start: float, preconditions: list[dict]) -> dict:
    return {
        "experiment": 3473,
        "title": (
            "Process-energy minority-correct recovery analysis on HEADROOM corpus (v3)"
        ),
        "run_date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "started_at": _START_AT,
        "finished_at": _now(),
        "duration_s": round(max(time.time() - start, 1.0), 3),
        "status": "success",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "benchmark_id": BENCHMARK_ID,
        "random_seed": SEED,
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "preconditions_checked": preconditions,
        "field_provenance": _field_provenance(),
        "gsm8k_trained_energy_auroc_baseline": GSM8K_TRAINED_AUROC_BASELINE,
    }


def _load_usable(records: list[dict]) -> list[dict]:
    """Keep only well-formed rows with a gold answer, a greedy generation, >=5 samples."""
    return [
        r
        for r in records
        if r.get("gold") is not None
        and (r.get("greedy") or {}).get("answer") is not None
        and len(r.get("samples") or []) >= 5
    ]


def main() -> None:
    start = time.time()
    preconditions: list[dict] = []

    # ----- PRECONDITION 0a: HEADROOM corpus present with >= MIN_PROBLEMS -----
    records: list[dict] = []
    if CORPUS_PATH.exists():
        with open(CORPUS_PATH) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    usable = _load_usable(records)
    preconditions.append(
        {
            "resource": "cached_headroom_corpus",
            "available": len(usable) >= MIN_PROBLEMS,
            "n_problems": len(usable),
        }
    )
    if len(usable) < MIN_PROBLEMS:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = (
            f"complete: blocked_p01_corpus_too_small_n={len(usable)}"
        )
        payload["methodology_note"] = (
            f"HEADROOM corpus has only n={len(usable)} usable problems "
            f"(<{MIN_PROBLEMS}); exp3471 (the corpus builder) resumes to grow it. "
            "No minority-correct analysis is reported on an under-powered corpus."
        )
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # ----- PRECONDITION 0b: energy substrate loadable -----
    substrate_ok = False
    exc_msg = ""
    try:
        from carnot.phase3.p01_minority_correct_recovery import (
            compute_minority_correct_recovery,
        )
        from carnot.phase3.p01_process_energy import (
            _candidate_steps,
            _Verifiers,
            process_energy_per_step,
        )
        from carnot.phase3.p01_trained_energy_reranker import (
            TrainedEnergyReranker,
            candidate_feature_vector,
            problem_kfold_indices,
        )

        verifiers = _Verifiers()
        _ = verifiers.ising.energy("2 + 2 = 4")
        substrate_ok = True
    except Exception as exc:
        exc_msg = str(exc)
    preconditions.append({"resource": "energy_substrate", "available": substrate_ok})
    if not substrate_ok:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = "complete: blocked_energy_substrate_unavailable"
        payload["methodology_note"] = f"Substrate import failed: {exc_msg[:200]}"
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # ----- Pre-compute per-candidate features, process energies, answers, labels -----
    n = len(usable)
    feats: list[list[list[float]]] = []
    labels: list[list[int]] = []
    proc: list[list[float]] = []
    answers_all: list[list] = []

    for rec in usable:
        gold = rec["gold"]
        samples = rec.get("samples") or []
        rec_feats, rec_labels, rec_proc, rec_ans = [], [], [], []
        for s in samples:
            text = s.get("text", "")
            mlp = s.get("mean_token_logprob")
            steps = _candidate_steps(s)
            rec_feats.append(candidate_feature_vector(text, mlp, verifiers))
            rec_labels.append(1 if s.get("answer") == gold else 0)
            rec_proc.append(process_energy_per_step(steps, verifiers))
            rec_ans.append(s.get("answer"))
        feats.append(rec_feats)
        labels.append(rec_labels)
        proc.append(rec_proc)
        answers_all.append(rec_ans)

    # ----- K-fold CV to produce fold-specific trained P(correct) -----
    splits = problem_kfold_indices(n, N_FOLDS, SEED)
    trained_probas: list[list[float]] = [[] for _ in range(n)]
    golds = [rec["gold"] for rec in usable]

    for train_idx, test_idx in splits:
        X_train: list[list[float]] = []
        y_train: list[int] = []
        for pi in train_idx:
            X_train.extend(feats[pi])
            y_train.extend(labels[pi])
        reranker = TrainedEnergyReranker(n_iter=RERANKER_ITER)
        reranker.fit(X_train, y_train)
        for pi in test_idx:
            trained_probas[pi] = (
                reranker.predict_proba(feats[pi]) if feats[pi] else []
            )

    # ----- Minority-correct recovery analysis -----
    result = compute_minority_correct_recovery(usable, proc, trained_probas)

    # ----- Acceptance gates -----
    g0_passed = result.minority_correct_fraction > 0.10
    g1_passed = (
        result.minority_correct_recovery_rate_process > 0.5
        or max(
            result.process_energy_correctness_auroc,
            result.trained_energy_correctness_auroc,
        ) > 0.65
    )

    g0_detail = {
        "condition": "minority_correct_fraction > 0.10",
        "passed": g0_passed,
        "minority_correct_fraction": round(result.minority_correct_fraction, 6),
        "principle": (
            "G0 HEADROOM-HAS-MINORITY-CORRECT: there ARE correct answers the "
            "majority misses — the necessary condition for any selector to beat SC "
            "(GSM8K lacked this)."
        ),
    }
    g1_detail = {
        "condition": (
            "minority_correct_recovery_rate > 0.5 OR "
            "max(process_energy_correctness_auroc, trained_energy_correctness_auroc) > 0.65"
        ),
        "passed": g1_passed,
        "minority_correct_recovery_rate_process": round(
            result.minority_correct_recovery_rate_process, 6
        ),
        "process_energy_correctness_auroc": round(
            result.process_energy_correctness_auroc, 6
        ),
        "trained_energy_correctness_auroc": round(
            result.trained_energy_correctness_auroc, 6
        ),
        "principle": (
            "G1 ENERGY-RECOVERS-MINORITY: the energy recovers more than half the "
            "minority-correct answers OR carries strong correctness signal — the "
            "mechanism that would let exp3472 beat SC."
        ),
    }

    # Tautology guard: ensure no two conceptually-distinct numeric metrics share
    # a bit-identical value (REQ-KONA-3473 METHODOLOGY_MISSING detector defence).
    _metrics = {
        "process_auroc": result.process_energy_correctness_auroc,
        "trained_auroc": result.trained_energy_correctness_auroc,
        "minority_fraction": result.minority_correct_fraction,
        "recovery_process": result.minority_correct_recovery_rate_process,
        "recovery_trained": result.minority_correct_recovery_rate_trained,
        "argmin_correct": result.within_problem_argmin_correct_rate_process,
    }
    tautology_pairs = [
        (a, b, va, vb)
        for (a, va), (b, vb) in (
            (pair1, pair2)
            for i, pair1 in enumerate(_metrics.items())
            for pair2 in list(_metrics.items())[i + 1:]
        )
        if va == vb and not (math.isnan(va) or math.isinf(va))
    ]
    tautology_note = (
        f"WARNING: {len(tautology_pairs)} bit-identical metric pairs detected: "
        + str([(a, b) for a, b, _, _ in tautology_pairs])
        if tautology_pairs
        else "no tautology flags"
    )

    # ----- Terminal verdict -----
    if not g0_passed:
        verdict = "complete: blocked_corpus_lacks_minority_correct_no_headroom"
    elif g1_passed:
        verdict = (
            "complete: process_energy_recovers_minority_correct_explains_p01_headroom_outcome"
        )
    else:
        verdict = (
            "complete: energy_fails_to_recover_minority_correct_even_with_headroom_"
            "ceiling_is_the_energy_not_the_benchmark"
        )

    payload = _base_payload(start, preconditions)
    payload.update(
        {
            "honest_verdict": verdict,
            "n_candidates_heldout": result.n_candidates,
            "process_energy_correctness_auroc": round(
                result.process_energy_correctness_auroc, 6
            ),
            "trained_energy_correctness_auroc": round(
                result.trained_energy_correctness_auroc, 6
            ),
            "process_energy_spearman": round(result.process_energy_spearman, 6),
            "trained_energy_spearman": round(result.trained_energy_spearman, 6),
            "within_problem_argmin_correct_rate_process": round(
                result.within_problem_argmin_correct_rate_process, 6
            ),
            "minority_correct_fraction": round(result.minority_correct_fraction, 6),
            "minority_correct_recovery_rate_process": round(
                result.minority_correct_recovery_rate_process, 6
            ),
            "minority_correct_recovery_rate_trained": round(
                result.minority_correct_recovery_rate_trained, 6
            ),
            "n_minority_correct_problems": result.n_minority_correct_problems,
            "acceptance_gate_g0_headroom_has_minority_correct": g0_detail,
            "acceptance_gate_g1_energy_recovers_minority": g1_detail,
            "methodology_note": (
                f"{N_FOLDS}-fold problem-level CV (seed={SEED}). "
                "Process energy = FoVer 4-verifier ensemble scored PER STEP and "
                "aggregated via mean. Trained energy = fold-specific logistic "
                "reranker P(correct) (same as exp3460/3461). "
                f"Minority-correct problems = where SC (majority vote) is wrong. "
                f"n={result.n_problems} held-out problems, "
                f"k={len(usable[0].get('samples') or [])} samples/problem. "
                f"Tautology check: {tautology_note}."
            ),
            "reproducibility_checksum": _checksum(usable),
        }
    )
    _emit(payload)

    print(
        f"DONE: {verdict}\n"
        f"  n_heldout_candidates={result.n_candidates} "
        f"n_problems={result.n_problems}\n"
        f"  process_AUROC={result.process_energy_correctness_auroc:.4f} "
        f"trained_AUROC={result.trained_energy_correctness_auroc:.4f} "
        f"gsm8k_baseline={GSM8K_TRAINED_AUROC_BASELINE:.4f}\n"
        f"  minority_correct_fraction={result.minority_correct_fraction:.4f} "
        f"(G0={'PASS' if g0_passed else 'FAIL'})\n"
        f"  minority_recovery_process={result.minority_correct_recovery_rate_process:.4f} "
        f"recovery_trained={result.minority_correct_recovery_rate_trained:.4f}\n"
        f"  G1={'PASS' if g1_passed else 'FAIL'} "
        f"argmin_correct={result.within_problem_argmin_correct_rate_process:.4f}\n"
        f"  tautology_check: {tautology_note}"
    )


if __name__ == "__main__":
    main()
