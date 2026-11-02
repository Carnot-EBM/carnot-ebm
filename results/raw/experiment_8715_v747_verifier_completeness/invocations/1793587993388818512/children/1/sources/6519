#!/usr/bin/env python3
"""Energy Inversion Root Cause Diagnostics — Deep Think 2026-05-01 prescription.

Runs the four hypothesis tests Deep Think specified for distinguishing the
root cause of the persistent energy inversion observed across exp1099 / 1100 /
1110 / 1118 (correct LLM outputs scoring HIGHER energy than incorrect ones,
when the EBM was trained for the opposite).

Sign convention (Deep Think): healthy state has
    delta_E = mean(E_incorrect) - mean(E_correct) > 0
inverted state has delta_E < 0.

Hypotheses
----------
A. Corpus narrowness — training data too narrow vs OOD eval distribution.
B. Loss-function geometry — loss formula structurally rewards inversion OOD.
C. Lipschitz over-regularization — energy collapses onto low-frequency proxy.
D. Verifier null-space contamination — inversion strictly localized to
   ensemble blind spots.

Input data
----------
Each test takes a JSONL file with per-pair records. Schema:

    {
      "energy": float,       # EBM energy score
      "is_correct": bool,    # ground-truth correctness label
      "text": str,           # candidate text (for length proxy in Test C)
      "verifier_scores": [float, ...],  # per-verifier scores (for Test D)
      "question_id": str,    # for matching correct/incorrect pairs
      ...
    }

If a field is missing, the corresponding test is skipped with a clear
diagnostic message rather than silently producing garbage.

Why a parameterized input file
------------------------------
The exp1099 / 1100 / 1110 / 1118 / 1120 result JSONs only emit *summary
statistics* (mean_correct_energy, mean_incorrect_energy). Deep Think's tests
require per-pair data that was never logged. The right structural fix is to
amend those experiment scripts to also emit a per-pair JSONL alongside the
summary JSON. This script consumes that JSONL once it exists.

Usage
-----
    python scripts/energy_inversion_diagnostics.py \\
        --id-jsonl data/exp1110_per_pair.jsonl \\
        --ood-jsonl data/exp1100_per_pair.jsonl \\
        [--ood1118-jsonl data/exp1118_per_pair.jsonl] \\
        [--summary-only]   # use mean_correct/incorrect_energy stats from
                           # existing artifacts when per-pair data unavailable

Spec: REQ-DIAG-001 — Deep Think 2026-05-01 four-hypothesis diagnostic.
"""

# Batching-audit note: per-pair iteration over candidate JSONL records is
# pure analysis (compute statistics on already-generated energies); no LLM
# inference is invoked. BatchedInferenceRunner does not apply.

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Project root so we can locate result JSONs by relative path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------


@dataclass
class PairRecord:
    """One per-pair record from the per-pair JSONL (schema documented in
    module docstring). Fields are typed permissively so downstream tests
    can probe what's available and skip cleanly when not.
    """

    energy: float
    is_correct: bool
    text: str | None
    verifier_scores: list[float] | None
    question_id: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PairRecord | None:
        e = d.get("energy")
        if e is None:
            e = d.get("energy_score")
        ic = d.get("is_correct")
        if e is None or ic is None:
            return None
        # Length probe takes either a "text" or "completion" or "step_text".
        text = d.get("text") or d.get("completion") or d.get("step_text")
        # Verifier scores: prefer continuous logits over binary verdicts.
        vs = d.get("verifier_scores")
        if vs is None:
            # Fall back to a single binary verdict if that's all we have;
            # Test D will downgrade trustworthiness in this case.
            vv = d.get("verifier_verdict")
            if vv is not None:
                vs = [1.0 if vv == "correct" else 0.0]
        return cls(
            energy=float(e),
            is_correct=bool(ic),
            text=text,
            verifier_scores=list(vs) if vs is not None else None,
            question_id=d.get("question_id"),
        )


# ----------------------------------------------------------------------
# Statistics helpers (no SciPy / NumPy dependency — keep this script
# trivially portable so it runs anywhere the conductor runs).
# ----------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation. NaN if degenerate (zero variance)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _r_squared(xs: list[float], ys: list[float]) -> float:
    """R^2 of simple linear regression y = a*x + b."""
    r = _pearson(xs, ys)
    return float("nan") if math.isnan(r) else r * r


def _partial_correlation(xs: list[float], ys: list[float], zs: list[float]) -> float:
    """Partial correlation r(x, y | z) via residualization."""
    if not (len(xs) == len(ys) == len(zs) >= 3):
        return float("nan")
    rxy = _pearson(xs, ys)
    rxz = _pearson(xs, zs)
    ryz = _pearson(ys, zs)
    if any(math.isnan(r) for r in (rxy, rxz, ryz)):
        return float("nan")
    denom = math.sqrt((1 - rxz**2) * (1 - ryz**2))
    if denom == 0:
        return float("nan")
    return (rxy - rxz * ryz) / denom


def _euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


def load_jsonl(path: Path) -> list[PairRecord]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = PairRecord.from_dict(d)
            if r is not None:
                records.append(r)
    return records


def load_summary_stats(path: Path) -> dict[str, float] | None:
    """Extract mean_correct_energy / mean_incorrect_energy from a result JSON
    when per-pair data isn't available. Returns None if neither field present.
    """
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    mc = d.get("mean_correct_energy") or d.get("mean_correct_energy_after")
    mi = d.get("mean_incorrect_energy") or d.get("mean_incorrect_energy_after")
    if mc is None or mi is None:
        return None
    return {
        "mean_correct_energy": float(mc),
        "mean_incorrect_energy": float(mi),
        "delta_E": float(mi) - float(mc),
        "n": int(d.get("n_outputs_run") or d.get("n_sota_holdout_correct") or 0)
        + int(d.get("n_sota_holdout_incorrect") or 0),
    }


# ----------------------------------------------------------------------
# Test A — Corpus narrowness
# ----------------------------------------------------------------------


def test_a_corpus(id_records: list[PairRecord], ood_records: list[PairRecord]) -> dict:
    """Threshold (supports A): delta_E_ID >> 0 AND delta_E_OOD < 0.
    Threshold (rejects A): delta_E_ID <= 0 (model inverted on its own training).
    """
    if not id_records or not ood_records:
        return {
            "test": "A",
            "status": "data_unavailable",
            "reason": "needs both ID and OOD per-pair JSONL with energy + is_correct",
        }
    e_id_c = [r.energy for r in id_records if r.is_correct]
    e_id_i = [r.energy for r in id_records if not r.is_correct]
    e_ood_c = [r.energy for r in ood_records if r.is_correct]
    e_ood_i = [r.energy for r in ood_records if not r.is_correct]
    if not (e_id_c and e_id_i and e_ood_c and e_ood_i):
        return {
            "test": "A",
            "status": "data_partial",
            "reason": (
                f"need both classes in both sets; got ID(c={len(e_id_c)},"
                f" i={len(e_id_i)}), OOD(c={len(e_ood_c)}, i={len(e_ood_i)})"
            ),
        }
    delta_id = _mean(e_id_i) - _mean(e_id_c)
    delta_ood = _mean(e_ood_i) - _mean(e_ood_c)
    supports = delta_id > 0 and delta_ood < 0
    rejects = delta_id <= 0
    if supports:
        verdict = "supports"
    elif rejects:
        verdict = "rejects"
    else:
        verdict = "inconclusive"
    return {
        "test": "A",
        "status": "computed",
        "verdict": verdict,
        "delta_E_ID": round(delta_id, 6),
        "delta_E_OOD": round(delta_ood, 6),
        "n_id": len(id_records),
        "n_ood": len(ood_records),
        "interpretation": (
            "supports A (corpus shift causes inversion)"
            if supports
            else (
                "rejects A (model inverted on training distribution too)"
                if rejects
                else "inconclusive"
            )
        ),
    }


# ----------------------------------------------------------------------
# Test B — Loss-function geometry
# ----------------------------------------------------------------------


def test_b_loss_geometry(ood_records: list[PairRecord], loss_fn: callable | None = None) -> dict:
    """Pair correct/incorrect candidates per question. Compute theoretical
    loss using the (frozen) energies via the original training loss formula.
    Pearson r(delta_E_pair, L_pair) > 0 supports B (loss rewards inversion).

    The training loss formula is task-specific. This function is parameterized
    on a callable that takes (E_correct, E_incorrect) and returns a scalar L.
    Without that callable, the test cannot run.
    """
    if loss_fn is None:
        return {
            "test": "B",
            "status": "loss_formula_required",
            "reason": (
                "needs the exact training loss formula reconstructed as a"
                " callable. Read scripts/experiment_1099_*.py and"
                " scripts/experiment_1110_*.py for the contrastive loss"
                " definition; pass it via the --loss-fn argument."
            ),
        }
    if not ood_records:
        return {"test": "B", "status": "data_unavailable"}
    # Pair by question_id
    by_q: dict[str, dict[str, PairRecord]] = {}
    for r in ood_records:
        if not r.question_id:
            continue
        slot = "correct" if r.is_correct else "incorrect"
        by_q.setdefault(r.question_id, {})[slot] = r
    pairs = [
        (v["correct"].energy, v["incorrect"].energy)
        for v in by_q.values()
        if "correct" in v and "incorrect" in v
    ]
    if len(pairs) < 3:
        return {
            "test": "B",
            "status": "insufficient_paired_data",
            "n_pairs_found": len(pairs),
        }
    deltas = [ei - ec for ec, ei in pairs]
    losses = [loss_fn(ec, ei) for ec, ei in pairs]
    r = _pearson(deltas, losses)
    supports = not math.isnan(r) and r > 0
    return {
        "test": "B",
        "status": "computed",
        "n_pairs": len(pairs),
        "pearson_r_delta_loss": round(r, 6) if not math.isnan(r) else None,
        "verdict": "supports" if supports else "rejects",
        "interpretation": (
            "supports B (loss formula rewards inversion OOD)"
            if supports
            else "rejects B (geometry fights inversion as intended)"
        ),
    }


# ----------------------------------------------------------------------
# Test C — Lipschitz over-regularization
# ----------------------------------------------------------------------


def test_c_lipschitz(ood_records: list[PairRecord]) -> dict:
    """R^2 of energy ~ length, plus partial correlation
    r(energy, is_correct | length).

    Supports C: R^2_length > 0.5 AND partial r approx 0.
    Rejects C: R^2_length < 0.1 AND inversion persists independent of length.
    """
    has_text = [r for r in ood_records if r.text is not None]
    if len(has_text) < 5:
        return {
            "test": "C",
            "status": "data_unavailable",
            "reason": f"needs records with text/completion field; got {len(has_text)}",
        }
    energies = [r.energy for r in has_text]
    lengths = [float(len(r.text)) for r in has_text]
    correctness = [1.0 if r.is_correct else 0.0 for r in has_text]
    r2 = _r_squared(lengths, energies)
    partial_r = _partial_correlation(energies, correctness, lengths)
    supports = (
        not math.isnan(r2) and r2 > 0.5 and not math.isnan(partial_r) and abs(partial_r) < 0.1
    )
    rejects = not math.isnan(r2) and r2 < 0.1
    if supports:
        verdict = "supports"
    elif rejects:
        verdict = "rejects"
    else:
        verdict = "inconclusive"
    return {
        "test": "C",
        "status": "computed",
        "n": len(has_text),
        "R_squared_energy_vs_length": round(r2, 6) if not math.isnan(r2) else None,
        "partial_r_energy_correctness_given_length": (
            round(partial_r, 6) if not math.isnan(partial_r) else None
        ),
        "verdict": verdict,
        "interpretation": (
            "supports C (energy collapsed onto length proxy)"
            if supports
            else (
                "rejects C (length not the dominant proxy)"
                if rejects
                else "inconclusive — try additional proxies (token count, step count)"
            )
        ),
        "caveat": "length is one of several low-frequency proxies; if test inconclusive, retest with token_count, step_count, code_block_count",
    }


# ----------------------------------------------------------------------
# Test D — Verifier null-space contamination
# ----------------------------------------------------------------------


def test_d_null_space(records: list[PairRecord]) -> dict:
    """Pair correct/incorrect per question. Compute Euclidean ΔV in 6-D
    verifier-score space. Bottom 20% (low ΔV = ensemble blind spot) vs top
    20%. ΔE in each subset.

    Supports D: ΔE << 0 in low-ΔV subset AND ΔE > 0 in high-ΔV subset.
    """
    has_v = [r for r in records if r.verifier_scores is not None and len(r.verifier_scores) > 0]
    if not has_v:
        return {"test": "D", "status": "data_unavailable", "reason": "no verifier_scores"}
    by_q: dict[str, dict[str, PairRecord]] = {}
    for r in has_v:
        if not r.question_id:
            continue
        slot = "correct" if r.is_correct else "incorrect"
        by_q.setdefault(r.question_id, {})[slot] = r
    pairs = [
        (v["correct"], v["incorrect"]) for v in by_q.values() if "correct" in v and "incorrect" in v
    ]
    if len(pairs) < 10:
        return {
            "test": "D",
            "status": "insufficient_paired_data",
            "n_pairs_found": len(pairs),
        }
    delta_v = [
        _euclidean(c.verifier_scores, i.verifier_scores)
        if len(c.verifier_scores) == len(i.verifier_scores)
        else float("nan")
        for c, i in pairs
    ]
    delta_e = [i.energy - c.energy for c, i in pairs]
    indexed = sorted(range(len(pairs)), key=lambda k: delta_v[k])
    n = len(indexed)
    p20 = max(1, n // 5)
    low_idx = indexed[:p20]
    high_idx = indexed[-p20:]
    de_low = _mean([delta_e[k] for k in low_idx])
    de_high = _mean([delta_e[k] for k in high_idx])
    # Continuous-vs-binary heuristic: if any record has a single-element
    # verifier_scores that's exactly 0 or 1, downgrade trustworthiness.
    binary_only = all(
        len(r.verifier_scores) == 1 and r.verifier_scores[0] in (0.0, 1.0) for r in has_v
    )
    supports = de_low < 0 and de_high > 0
    return {
        "test": "D",
        "status": "computed",
        "n_pairs": len(pairs),
        "delta_E_low_dV_quintile": round(de_low, 6),
        "delta_E_high_dV_quintile": round(de_high, 6),
        "verdict": "supports" if supports else "rejects",
        "interpretation": (
            "supports D (inversion localized to ensemble blind spots)"
            if supports
            else "rejects D (inversion uniform across verifier-distance deciles)"
        ),
        "trustworthiness": (
            "low — verifier_scores look binary (0/1); test needs continuous logits"
            if binary_only
            else "ok"
        ),
    }


# ----------------------------------------------------------------------
# Decision tree
# ----------------------------------------------------------------------


def decide(results: dict[str, dict]) -> dict:
    a = results.get("A", {}).get("verdict")
    b = results.get("B", {}).get("verdict")
    c = results.get("C", {}).get("verdict")
    d = results.get("D", {}).get("verdict")

    if b == "supports":
        return {
            "primary_cause": "loss_geometry",
            "exp1120_will_resolve": False,
            "recommended_87_action": "cancel_exp1121_production_wiring",
            "recommended_88_action": "EBM_head_loss_redesign",
        }
    if (
        a == "supports"
        and b in ("rejects", None)
        and c in ("rejects", None)
        and d in ("rejects", None)
    ):
        return {
            "primary_cause": "corpus",
            "exp1120_will_resolve": True,
            "recommended_87_action": "proceed_exp1121_production_wiring",
            "recommended_88_action": "none",
        }
    if c == "supports":
        return {
            "primary_cause": "lipschitz_overregularization",
            "exp1120_will_resolve": False,
            "recommended_87_action": "monitor_exp1120_outcome_first",
            "recommended_88_action": "spectral_norm_ablation",
        }
    if d == "supports":
        return {
            "primary_cause": "verifier_null_space",
            "exp1120_will_resolve": False,
            "recommended_87_action": "monitor_exp1120_outcome_first",
            "recommended_88_action": "null_space_measurement_on_k5_ensemble",
        }
    return {
        "primary_cause": "ambiguous",
        "exp1120_will_resolve": "unknown",
        "recommended_87_action": "wait_for_exp1120_then_rerun_diagnostics",
        "recommended_88_action": "tbd",
    }


# ----------------------------------------------------------------------
# Summary-only fallback (when per-pair JSONL not available)
# ----------------------------------------------------------------------


def summary_only_partial_diagnostic() -> dict:
    """When per-pair data is unavailable, compute a partial Test A from
    summary-statistics fields in existing result JSONs. Cannot run B/C/D.
    """
    artifacts = {
        "exp1100_OOD_initial": load_summary_stats(
            PROJECT_ROOT / "results/experiment_1100_cascade_validation_sota_outputs.json"
        ),
        "exp1099_RLVR_baseline": load_summary_stats(
            PROJECT_ROOT / "results/experiment_1099_rlvr_ssd_integration_v1.json"
        ),
        "exp1110_nondegenerate": load_summary_stats(
            PROJECT_ROOT / "results/experiment_1110_rlvr_ssd_v2_nondegenerate_live_gpu.json"
        ),
        "exp1118_grpo": load_summary_stats(
            PROJECT_ROOT / "results/experiment_1118_grpo_energy_prm_v1.json"
        ),
        "exp1120_post_retrain": load_summary_stats(
            PROJECT_ROOT / "results/experiment_1120_energy_verifier_retrain_sota.json"
        ),
    }
    return {
        "summary_stats_available": {k: v for k, v in artifacts.items() if v is not None},
        "summary_stats_missing": [k for k, v in artifacts.items() if v is None],
        "diagnostic_status": (
            "Tests B/C/D require per-pair JSONL that has never been emitted by"
            " any experiment. The right structural fix is to amend"
            " experiment_1100, _1110, _1118, _1120 to also emit"
            " results/experiment_NNNN_per_pair.jsonl alongside the summary"
            " JSON. ~10 lines per script."
        ),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--id-jsonl", type=Path, default=None, help="In-domain per-pair JSONL (e.g., from exp1110)"
    )
    parser.add_argument(
        "--ood-jsonl",
        type=Path,
        default=None,
        help="Out-of-domain per-pair JSONL (e.g., from exp1100/1120)",
    )
    parser.add_argument(
        "--ood1118-jsonl",
        type=Path,
        default=None,
        help="exp1118-class JSONL with verifier_scores for Test D",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Fall back to summary-stats analysis when per-pair unavailable",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/energy_inversion_diagnostics.json"
    )
    args = parser.parse_args()

    if args.summary_only or not (args.id_jsonl and args.ood_jsonl):
        # Honest path: we don't have per-pair data, so report what's possible.
        out = {"mode": "summary_only", **summary_only_partial_diagnostic()}
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print("\n=== Energy Inversion Diagnostics — SUMMARY-ONLY MODE ===\n")
        print(json.dumps(out, indent=2))
        # Compute partial Test A from summary stats
        avail = out["summary_stats_available"]
        if "exp1100_OOD_initial" in avail and "exp1120_post_retrain" in avail:
            before = avail["exp1100_OOD_initial"]["delta_E"]
            after = avail["exp1120_post_retrain"]["delta_E"]
            print(f"\nPartial Test A (summary):")
            print(f"  delta_E_OOD before retrain (exp1100) = {before:+.6f}")
            print(f"  delta_E_OOD after retrain (exp1120)  = {after:+.6f}")
            if before < 0 and after > 0:
                print("  → corpus retraining FLIPPED the inversion")
                print("  → supports Hypothesis A (corpus is primary cause)")
            elif before < 0 and after >= 0:
                print("  → corpus retraining IMPROVED but did not flip")
                print("  → suggests A + (C or D)")
            elif after <= before:
                print("  → corpus retraining did NOT improve")
                print("  → REJECTS Hypothesis A; B/C/D need per-pair tests")
        elif "exp1100_OOD_initial" in avail:
            d = avail["exp1100_OOD_initial"]["delta_E"]
            print(f"\nPartial baseline (exp1100 only):")
            print(f"  delta_E_OOD = {d:+.6f}")
            print(f"  (CONFIRMED: {'inverted' if d < 0 else 'healthy'} on SOTA outputs)")
        return 0

    id_records = load_jsonl(args.id_jsonl) if args.id_jsonl else []
    ood_records = load_jsonl(args.ood_jsonl) if args.ood_jsonl else []
    ood1118 = load_jsonl(args.ood1118_jsonl) if args.ood1118_jsonl else ood_records

    results = {
        "A": test_a_corpus(id_records, ood_records),
        "B": test_b_loss_geometry(ood_records, loss_fn=None),  # callable injection TBD
        "C": test_c_lipschitz(ood_records),
        "D": test_d_null_space(ood1118),
    }
    decision = decide(results)

    out = {
        "mode": "per_pair",
        "n_id_records": len(id_records),
        "n_ood_records": len(ood_records),
        "n_ood1118_records": len(ood1118),
        "tests": results,
        "decision": decision,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
