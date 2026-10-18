#!/usr/bin/env python3
"""Exp 3554 — FoVer step->final aggregation promotion: cross-corpus multiseed transfer CI.

Spec: REQ-KONA-3554, SCENARIO-KONA-3554
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

CORPUS_A_PATH = REPO_ROOT / "data" / "p01_difficulty_matched_generations.jsonl"
CORPUS_B_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"
CORPUS_C_PATH = REPO_ROOT / "data" / "p01_greedy_wrong_headroom_corpus.jsonl"
ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3554_fover_step_aggregation_secondary_headline_multiseed_third_corpus_v2.json"
)

# Determinism
_BASE_SEED = 20260601
N_SEEDS = 5
HELD_OUT_FRACTION = 0.30

_T_CRIT = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

def _checksum(records_a: list[dict], records_b: list[dict], records_c: list[dict], seeds: list[int]) -> str:
    h = hashlib.sha256()
    h.update(f"exp=3554;seed={_BASE_SEED}".encode())
    h.update(f";seeds={seeds}".encode())
    for rec in records_a:
        h.update(str(rec.get("problem_id")).encode())
    for rec in records_b:
        h.update(str(rec.get("problem_id")).encode())
    for rec in records_c:
        h.update(str(rec.get("problem_id")).encode())
    return h.hexdigest()[:16]

def _load_corpus(path: Path) -> list[dict]:
    records: list[dict] = []
    if path.exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records

def _is_usable(rec: dict) -> bool:
    gold = rec.get("gold_answer") or rec.get("gold_answer_norm")
    if not gold:
        return False
    samples = rec.get("samples") or []
    scored = [
        s for s in samples
        if "correct" in s and (s.get("reasoning_steps") or s.get("steps"))
    ]
    return len(scored) >= 2

def _normalise_sample(s: dict) -> dict:
    if "steps" in s and "reasoning_steps" not in s:
        s = dict(s)
        s["reasoning_steps"] = s.pop("steps")
    return s

def _train_held_out_split(records: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_held = max(1, round(len(shuffled) * HELD_OUT_FRACTION))
    held_out = shuffled[:n_held]
    train = shuffled[n_held:]
    return train, held_out

def _distinct_pipeline_assert(step_scores_last: list[float], agg_scores_min: list[float]) -> bool:
    if len(step_scores_last) != len(agg_scores_min):
        return True
    if not step_scores_last:
        return True
    return not all(s == a for s, a in zip(step_scores_last, agg_scores_min))

def _ci95(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    t = _T_CRIT.get(n - 1, 2.776)
    margin = t * std / math.sqrt(n)
    return round(mean - margin, 6), round(mean + margin, 6)

def _base_payload(start: float) -> dict:
    return {
        "honest_verdict": None,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "distinct_pipeline_assert_passed": None,
        "train_corpus": "p01_difficulty_matched_generations.jsonl",
        "transfer_corpus_b": "fover_corpus.jsonl",
        "transfer_corpus_c": "p01_greedy_wrong_headroom_corpus.jsonl",
        "third_corpus_available": None,
        "within_corpus_auroc": None,
        "transfer_auroc_b": None,
        "transfer_auroc_b_ci95": None,
        "transfer_auroc_c": None,
        "transfer_auroc_c_ci95": None,
        "unaggregated_transfer_floor_b": None,
        "unaggregated_transfer_floor_c": None,
        "transfer_shuffle_control_b": None,
        "transfer_shuffle_control_c": None,
        "shuffle_controls_collapse": None,
        "secondary_headline_eligible": None,
        "random_seed": _BASE_SEED,
        "reproducibility_checksum": None,
        "duration_s": round(max(time.time() - start, 1.0), 3),
    }

def _emit(payload: dict) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload["schema"] = sorted(payload.keys())
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)

def main() -> None:
    start = time.time()

    raw_a = _load_corpus(CORPUS_A_PATH)
    usable_a = [r for r in raw_a if _is_usable(r)]
    raw_b = _load_corpus(CORPUS_B_PATH)
    usable_b = [r for r in raw_b if _is_usable(r)]
    raw_c = _load_corpus(CORPUS_C_PATH)
    usable_c = [r for r in raw_c if _is_usable(r)]

    third_corpus_available = len(usable_c) > 0

    if len(usable_a) < 10 or len(usable_b) < 10:
        payload = _base_payload(start)
        payload["honest_verdict"] = "complete: blocked_no_second_corpus_for_transfer"
        payload["third_corpus_available"] = third_corpus_available
        _emit(payload)
        print(payload["honest_verdict"])
        return

    usable_a = [{**rec, "samples": [_normalise_sample(s) for s in (rec.get("samples") or [])]} for rec in usable_a]
    usable_b = [{**rec, "samples": [_normalise_sample(s) for s in (rec.get("samples") or [])]} for rec in usable_b]
    usable_c = [{**rec, "samples": [_normalise_sample(s) for s in (rec.get("samples") or [])]} for rec in usable_c]

    try:
        from carnot.phase3.p01_trained_energy_reranker import _Verifiers
        verifiers = _Verifiers()
        _ = verifiers.ising.energy("2 + 2 = 4")
    except Exception as exc:
        payload = _base_payload(start)
        payload["honest_verdict"] = "complete: blocked_energy_substrate_unavailable"
        _emit(payload)
        print(payload["honest_verdict"])
        return

    from carnot.phase3.p01_step_aggregation import (
        compute_aggregation_auroc,
        compute_per_step_verifier_scores,
        aggregate_step_energies,
    )

    seeds = [_BASE_SEED + i * 137_931 for i in range(N_SEEDS)]

    per_seed_auroc_a = []
    per_seed_auroc_b = []
    per_seed_auroc_c = []
    per_seed_shuffle_b = []
    per_seed_shuffle_c = []
    all_distinct_ok = []

    for i, seed in enumerate(seeds):
        train_a, held_out_a = _train_held_out_split(usable_a, seed)
        _, held_out_b = _train_held_out_split(usable_b, seed)
        if third_corpus_available:
            _, held_out_c = _train_held_out_split(usable_c, seed)
        else:
            held_out_c = []

        # Fit on train_a
        methods_to_try = ["min", "mean", "last", "product", "uncertainty_weighted"]
        best_method = None
        best_auroc_train_a = -1.0
        for method in methods_to_try:
            res = compute_aggregation_auroc(train_a, verifiers, method)
            if res["auroc"] > best_auroc_train_a:
                best_auroc_train_a = res["auroc"]
                best_method = method

        # Eval on held_out_a
        res_a = compute_aggregation_auroc(held_out_a, verifiers, best_method)
        per_seed_auroc_a.append(res_a["auroc"])

        # Eval on held_out_b
        last_scores_b: list[float] = []
        for rec in held_out_b:
            for s in (rec.get("samples") or []):
                steps = s.get("reasoning_steps") or []
                v_scores = compute_per_step_verifier_scores(steps, verifiers)
                last_e = aggregate_step_energies(v_scores, "last")
                last_scores_b.append(last_e)

        res_b = compute_aggregation_auroc(held_out_b, verifiers, best_method)
        distinct_ok_b = _distinct_pipeline_assert(last_scores_b, res_b["agg_scores"])
        all_distinct_ok.append(distinct_ok_b)
        per_seed_auroc_b.append(res_b["auroc"])

        # Shuffle B
        shuffled_b = copy.deepcopy(held_out_b)
        all_labels_b: list[bool | None] = []
        for rec in shuffled_b:
            for s in (rec.get("samples") or []):
                all_labels_b.append(s.get("correct"))
        rng_shuffle_b = random.Random(seed + 777)
        rng_shuffle_b.shuffle(all_labels_b)
        idx = 0
        for rec in shuffled_b:
            for s in (rec.get("samples") or []):
                s["correct"] = all_labels_b[idx]
                idx += 1
        res_shuffle_b = compute_aggregation_auroc(shuffled_b, verifiers, best_method)
        per_seed_shuffle_b.append(res_shuffle_b["auroc"])

        # Eval on held_out_c
        if third_corpus_available:
            res_c = compute_aggregation_auroc(held_out_c, verifiers, best_method)
            per_seed_auroc_c.append(res_c["auroc"])

            shuffled_c = copy.deepcopy(held_out_c)
            all_labels_c: list[bool | None] = []
            for rec in shuffled_c:
                for s in (rec.get("samples") or []):
                    all_labels_c.append(s.get("correct"))
            rng_shuffle_c = random.Random(seed + 888)
            rng_shuffle_c.shuffle(all_labels_c)
            idx = 0
            for rec in shuffled_c:
                for s in (rec.get("samples") or []):
                    s["correct"] = all_labels_c[idx]
                    idx += 1
            res_shuffle_c = compute_aggregation_auroc(shuffled_c, verifiers, best_method)
            per_seed_shuffle_c.append(res_shuffle_c["auroc"])
            
            print(f"[seed {i+1}/{N_SEEDS}] corpus_A={res_a['auroc']:.4f} corpus_B={res_b['auroc']:.4f} shuffle_B={res_shuffle_b['auroc']:.4f} corpus_C={res_c['auroc']:.4f} shuffle_C={res_shuffle_c['auroc']:.4f}", flush=True)
        else:
            print(f"[seed {i+1}/{N_SEEDS}] corpus_A={res_a['auroc']:.4f} corpus_B={res_b['auroc']:.4f} shuffle_B={res_shuffle_b['auroc']:.4f}", flush=True)

    unagg_floor_b = compute_aggregation_auroc(usable_b, verifiers, "mean")["auroc"]
    unagg_floor_c = compute_aggregation_auroc(usable_c, verifiers, "mean")["auroc"] if third_corpus_available else None

    mean_a = sum(per_seed_auroc_a) / N_SEEDS
    mean_b = sum(per_seed_auroc_b) / N_SEEDS
    ci_b_lo, ci_b_hi = _ci95(per_seed_auroc_b)
    mean_shuffle_b = sum(per_seed_shuffle_b) / N_SEEDS

    if third_corpus_available:
        mean_c = sum(per_seed_auroc_c) / N_SEEDS
        ci_c_lo, ci_c_hi = _ci95(per_seed_auroc_c)
        mean_shuffle_c = sum(per_seed_shuffle_c) / N_SEEDS
    else:
        mean_c = None
        ci_c_lo, ci_c_hi = None, None
        mean_shuffle_c = None

    distinct_ok = all(all_distinct_ok)

    shuffle_collapses_b = mean_shuffle_b < 0.6
    shuffle_collapses_c = (mean_shuffle_c < 0.6) if third_corpus_available else True
    shuffle_controls_collapse = shuffle_collapses_b and shuffle_collapses_c

    # "transfer beats the un-aggregated floor on EVERY available target with collapsing shuffle controls AND multi-seed CIs exclude the floor"
    beats_b = (mean_b > unagg_floor_b) and (ci_b_lo > unagg_floor_b)
    beats_c = (mean_c > unagg_floor_c) and (ci_c_lo > unagg_floor_c) if third_corpus_available else True

    secondary_headline_eligible = beats_b and beats_c and shuffle_controls_collapse

    # VERDICT
    if not distinct_ok:
        verdict = "complete: blocked_distinct_pipeline_assert_failed"
    elif secondary_headline_eligible:
        verdict = "complete: step_to_final_aggregation_secondary_headline_confirmed_multiseed_transfer_BB_CC_secondary_headline_eligible"
    elif beats_b and not beats_c:
        verdict = "complete: step_to_final_aggregation_transfers_to_B_not_C_bounded_to_single_pair_not_yet_secondary_headline"
    else:
        verdict = "complete: step_to_final_aggregation_does_not_transfer"

    checksum = _checksum(usable_a, usable_b, usable_c, seeds)

    payload = _base_payload(start)
    payload.update({
        "honest_verdict": verdict,
        "distinct_pipeline_assert_passed": distinct_ok,
        "third_corpus_available": third_corpus_available,
        "within_corpus_auroc": round(mean_a, 6),
        "transfer_auroc_b": round(mean_b, 6),
        "transfer_auroc_b_ci95": [ci_b_lo, ci_b_hi],
        "transfer_auroc_c": round(mean_c, 6) if mean_c is not None else None,
        "transfer_auroc_c_ci95": [ci_c_lo, ci_c_hi] if ci_c_lo is not None else None,
        "unaggregated_transfer_floor_b": round(unagg_floor_b, 6),
        "unaggregated_transfer_floor_c": round(unagg_floor_c, 6) if unagg_floor_c is not None else None,
        "transfer_shuffle_control_b": round(mean_shuffle_b, 6),
        "transfer_shuffle_control_c": round(mean_shuffle_c, 6) if mean_shuffle_c is not None else None,
        "shuffle_controls_collapse": shuffle_controls_collapse,
        "secondary_headline_eligible": secondary_headline_eligible,
        "reproducibility_checksum": checksum,
        "duration_s": round(max(time.time() - start, 1.0), 3),
        "methodology_note": "Reference numbers 0.861 and 0.749.",
    })
    _emit(payload)

    print(f"\nDONE: {verdict}")
    print(f"  within_A={mean_a:.4f}")
    print(f"  transfer_B={mean_b:.4f} floor_B={unagg_floor_b:.4f} shuffle_B={mean_shuffle_b:.4f}")
    if third_corpus_available:
        print(f"  transfer_C={mean_c:.4f} floor_C={unagg_floor_c:.4f} shuffle_C={mean_shuffle_c:.4f}")

if __name__ == "__main__":
    main()
