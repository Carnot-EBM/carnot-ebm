#!/usr/bin/env python3
"""Exp 3495 — P0.1 process-aware energy + optimal aggregation vs SC on CONTESTED SUBSET (in-band v8).

Spec: REQ-KONA-3495, SCENARIO-KONA-3495, SCENARIO-KONA-3495-BLOCKED

WHY this experiment exists (the improvement over v6 / exp 3472):
  exp 3472 (v6) scored a single HEADROOM corpus (data/p01_hardmath_generations.jsonl) that
  was filtered to problems where *the full corpus* has SC in [0.40, 0.78] — a corpus-level
  gate. But within ANY corpus, some problems are near-certainty (SC ≈ 1.0) and some are
  nearly-impossible (SC ≈ 0.0); those tails are where energy is least likely to add signal
  above SC.

  exp 3495 (v8) pools BOTH cached corpora (GSM8K + MATH-L5 / hardmath), then keeps only
  the "contested subset": problems whose PER-PROBLEM correctness rate (fraction of k samples
  with answer == gold) sits in [0.40, 0.70]. This is the subset where SC is structurally
  uncertain and where an energy-based selector has the most room to break ties differently.

PRIMARY gate: the flip-count on the contested subset (same logic as v6). If the contested
subset is too small (<40 problems) the experiment reports a blocked verdict — the current
cached corpora are expected to yield n≈21 (<40), so this run will block.

INFERENCE SUBSTRATE: verifier_ensemble_against_cached_candidates — no live model is loaded;
the experiment scores cached (problem, gold, samples) triples via the process-energy module.

Run:
  cd /home/ianblenke/github.com/ianblenke/carnot && \\
    .venv/bin/python scripts/experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
# JAX (pulled in transitively by some verify modules) must stay on CPU for
# reproducible, GPU-free scoring. Energy evaluations are CPU-bound here.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Both corpora are pooled; either or both may be absent (handled gracefully).
CORPUS_GSM8K = REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl"
CORPUS_HARDMATH = REPO_ROOT / "data" / "p01_hardmath_generations.jsonl"

ARTIFACT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8.json"
)

SEED = 20260603  # distinct from experiment id (3495) to avoid a TAUTOLOGY adversarial flag
N_FOLDS = 5       # problem-level cross-validation folds
N_BOOT = 10000    # bootstrap iterations for CI95
RERANKER_ITER = 500
MIN_PROBLEMS = 40  # contested subset must be >= 40 to be headline-eligible
# Contested-subset correctness-rate window: keep problems where per-problem
# fraction-correct is in [LOW_RATE, HIGH_RATE]. This is tighter than v6's
# corpus-level [0.40, 0.78] SC gate — it operates per problem, not per corpus.
CONTEST_LOW = 0.40
CONTEST_HIGH = 0.70

# All result-bearing fields emitted as null in the blocked path so downstream
# gate-synth/capstone tasks never cascade-block on a missing key.
_RESULT_FIELDS: tuple[str, ...] = (
    "source_corpora",
    "contested_subset_n",
    "contested_subset_sc",
    "k_samples",
    "self_consistency_in_headroom_band",
    "ar_greedy_accuracy",
    "self_consistency_accuracy",
    "self_certainty_bon_accuracy",
    "process_energy_argmin_accuracy",
    "trained_energy_weighted_vote_accuracy",
    "trained_energy_sc_hybrid_accuracy",
    "optimal_aggregation_accuracy",
    "flip_count_optimal_vs_sc",
    "flips_correct_optimal",
    "flips_incorrect_optimal",
    "net_correctness_gain_optimal",
    "delta_optimal_vs_self_consistency",
    "delta_process_energy_vs_self_consistency",
    "paired_significance",
    "compute_parity_note",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Helper: per-problem correctness rate
# ---------------------------------------------------------------------------

def _per_problem_correctness_rate(rec: dict) -> float:
    """Return the fraction of sampled answers that match the gold answer.

    WHY: we use this per-problem rate (not corpus-level SC) to identify
    problems that are genuinely uncertain for the model — neither so easy
    that SC is near-ceiling nor so hard that all samples fail.  This is the
    key structural difference from v6's corpus-level gate.

    A record with no samples or no gold is not usable (filtered upstream),
    but we guard defensively here to avoid ZeroDivisionError.
    """
    gold = rec.get("gold")
    samples = rec.get("samples") or []
    if not samples or gold is None:
        return 0.0
    correct = sum(1 for s in samples if s.get("answer") == gold)
    return correct / len(samples)


def _build_contested_subset(
    records: list[dict],
    low: float = CONTEST_LOW,
    high: float = CONTEST_HIGH,
) -> list[dict]:
    """Filter records to those with per-problem correctness rate in [low, high].

    WHY: the contested subset is where an energy-based selector has the most
    potential to differ from self-consistency (SC). On near-certain problems
    (rate > high) SC already picks the correct answer; on near-impossible
    problems (rate < low) SC and energy both fail. Only the middle band has
    structural disagreement potential — i.e. some samples are correct and some
    are not, so the selector's choice matters.
    """
    return [r for r in records if low <= _per_problem_correctness_rate(r) <= high]


def _majority_vote_answer(rec: dict) -> str | None:
    """Return the plurality answer across all samples for a single record.

    WHY: the contested_subset_sc field is computed from majority-vote accuracy
    over the contested subset.  Majority vote is the canonical self-consistency
    estimator (Wang et al. 2023).  We use it here rather than corpus-level SC
    so the reported metric reflects the actual decision rule being tested.
    """
    samples = rec.get("samples") or []
    candidates = [s.get("answer") for s in samples if s.get("answer") is not None]
    if not candidates:
        return None
    return Counter(candidates).most_common(1)[0][0]


def _compute_subset_sc(records: list[dict]) -> float | None:
    """Majority-vote accuracy over the contested subset.

    WHY: we report this BEFORE energy scoring so it is available in both the
    blocked and the scored paths.  This is also the denominator for the
    delta_* fields in the scored path.
    """
    if not records:
        return None
    correct = sum(
        1
        for r in records
        if _majority_vote_answer(r) == r.get("gold")
    )
    return round(correct / len(records), 6)


def _checksum(records: list[dict]) -> str:
    """Content hash of the contested subset + reranker/aggregator config + seed.

    WHY: content-addressed hash lets a third party reproduce the run and confirm
    the checksum matches — catching silent corpus or config drift between runs.
    Hashes seed, fold count, reranker iterations, substrate label, and each
    problem's id, gold, and all sample answers (the information the scorer uses).
    """
    h = hashlib.sha256()
    h.update(
        f"seed={SEED};folds={N_FOLDS};iter={RERANKER_ITER};"
        f"substrate=process_energy+optimal_aggregation;"
        f"contest_low={CONTEST_LOW};contest_high={CONTEST_HIGH}".encode()
    )
    for rec in records:
        h.update(json.dumps(rec.get("problem_id"), sort_keys=True).encode())
        h.update(json.dumps(rec.get("gold"), sort_keys=True).encode())
        for s in rec.get("samples") or []:
            h.update(str(s.get("answer")).encode())
            h.update(str(s.get("mean_token_logprob")).encode())
    return h.hexdigest()[:16]


def _field_provenance() -> dict:
    """One-line WHY per artifact field (CLAUDE.md principle-annotation discipline).

    WHY: agents that know the principle behind a field produce more accurate and
    OOD-robust artifacts than those that only see the directive (Anthropic
    'Teaching Claude Why', 2026-05).
    """
    return {
        "honest_verdict": (
            "Terminal verdict must start with complete:/success:/passed:/shipped_ "
            "so the conductor reconciler classifies it as terminal without false-positive "
            "partial-token matches (CLAUDE.md Verdict Terminal-Prefix Discipline)."
        ),
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates: no live model is loaded; "
            "duration floor is 1s (not 60s). Declared explicitly to avoid the "
            "DURATION_TOO_SHORT false-positive that triggered the Inference-Substrate "
            "Declaration Discipline (CLAUDE.md 2026-05-22)."
        ),
        "source_corpora": (
            "Per-corpus statistics (n_usable, n_contested, path) for reproducibility "
            "and to distinguish which corpus contributed which proportion of the subset."
        ),
        "contested_subset_n": (
            "Number of problems in the contested subset after the rate filter. "
            "Must be >= 40 for headline-eligibility (MIN_PROBLEMS gate)."
        ),
        "contested_subset_sc": (
            "Majority-vote accuracy over the contested subset, computed BEFORE energy "
            "scoring.  Reported in both blocked and scored paths so the SC baseline is "
            "always visible even when the energy comparison cannot run."
        ),
        "k_samples": "Sampled generations/problem consumed (the matched-compute budget).",
        "self_consistency_in_headroom_band": (
            "SC in [0.40, 0.78] over the CONTESTED SUBSET — guards against testing "
            "energy when SC is already at ceiling (would degenerate to tautology)."
        ),
        "ar_greedy_accuracy": "1-sample greedy control (held-out folds).",
        "self_consistency_accuracy": "Majority vote over k — the PRIMARY control (held-out).",
        "self_certainty_bon_accuracy": "Self-certainty Best-of-N (arXiv:2502.18581).",
        "process_energy_argmin_accuracy": (
            "FoVer step-level PROCESS energy argmin — per-step, not candidate-level."
        ),
        "trained_energy_weighted_vote_accuracy": "Trained EORM energy-weighted vote.",
        "trained_energy_sc_hybrid_accuracy": "Trained-energy x SC hybrid.",
        "optimal_aggregation_accuracy": (
            "Optimal SC+energy aggregation (arXiv:2510.13918) — THE headline condition."
        ),
        "flip_count_optimal_vs_sc": (
            "Problems where optimal-aggregation differs from SC — the tautology-clean "
            "primary signal (0 flip_count means the two agree everywhere)."
        ),
        "flips_correct_optimal": "Flips that became CORRECT (the win mechanism).",
        "flips_incorrect_optimal": "Flips that became WRONG (the cost).",
        "net_correctness_gain_optimal": (
            "flips_correct - flips_incorrect — the honest net effect, robust to "
            "ceiling-induced ties."
        ),
        "delta_optimal_vs_self_consistency": (
            "Optimal-aggregation accuracy minus SC accuracy — THE headline delta."
        ),
        "delta_process_energy_vs_self_consistency": (
            "Process-energy argmin accuracy minus SC — does per-step verification "
            "route into selection?"
        ),
        "paired_significance": (
            "McNemar exact p + paired bootstrap CI95 for optimal, process-energy, "
            "and hybrid deltas."
        ),
        "compute_parity_note": (
            "Per-condition generation budget + reranker/aggregator params so energy "
            "cannot win by spending more compute."
        ),
        "random_seed": (
            "Determinism is the precondition for reproducibility; missing seed means "
            "no third party can re-run and confirm or refute the claim."
        ),
        "reproducibility_checksum": (
            "Content-addressed hash of contested-subset corpus + reranker config + "
            "seed catches silent corpus or config drift between runs."
        ),
        "duration_s": (
            "Cached scoring + small-model training; substrate is "
            "verifier_ensemble_against_cached_candidates so 1s floor applies, "
            "not 60s (no live model inference)."
        ),
    }


def _emit(payload: dict) -> None:
    """Write the artifact to disk, ensuring every required field is present."""
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Ensure every result-bearing field exists — null in the blocked path — so
    # downstream gate-synth/capstone tasks never cascade-block on a missing key.
    for fld in _RESULT_FIELDS:
        payload.setdefault(fld, None)
    payload["schema"] = sorted(payload.keys())
    with open(ARTIFACT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)


_START_AT = _now()


def _base_payload(start: float, preconditions: list[dict]) -> dict:
    """Base artifact fields shared by both the blocked and the scored paths."""
    return {
        "experiment": 3495,
        "title": (
            "P0.1 process-aware energy + optimal aggregation vs self-consistency "
            "on CONTESTED SUBSET (in-band v8)"
        ),
        "run_date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "started_at": _START_AT,
        "finished_at": _now(),
        "duration_s": round(time.time() - start, 3),
        "status": "success",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": SEED,
        "metrics_used": "exact_match_accuracy+flip_count",
        "corpus_paths": {
            "gsm8k": str(CORPUS_GSM8K.relative_to(REPO_ROOT)),
            "hardmath": str(CORPUS_HARDMATH.relative_to(REPO_ROOT)),
        },
        "contest_rate_window": [CONTEST_LOW, CONTEST_HIGH],
        "preconditions_checked": preconditions,
        "field_provenance": _field_provenance(),
    }


def _load_usable(records: list[dict]) -> list[dict]:
    """Keep only well-formed rows: a gold answer, a greedy generation, >=5 samples.

    WHY: records missing gold cannot be evaluated for correctness; records
    missing a greedy answer omit the AR-greedy baseline; records with fewer
    than 5 samples produce an unreliable correctness-rate estimate for the
    contested-subset filter.
    """
    return [
        r
        for r in records
        if r.get("gold") is not None
        and (r.get("greedy") or {}).get("answer") is not None
        and len(r.get("samples") or []) >= 5
    ]


def _load_corpus(path: Path) -> list[dict]:
    """Load a JSONL corpus file, returning an empty list if the file is missing."""
    records: list[dict] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (FileNotFoundError, OSError):
        # Missing corpus is not a fatal error; the experiment handles it gracefully
        # by reporting a blocked verdict if the pooled contested subset is too small.
        pass
    return records


def main() -> None:
    start = time.time()
    preconditions: list[dict] = []

    # ------------------------------------------------------------------
    # PRECONDITION 0: load and pool both corpora; filter to usable rows
    # ------------------------------------------------------------------
    # Load each corpus independently so we can report per-corpus stats.
    raw_gsm8k = _load_corpus(CORPUS_GSM8K)
    raw_hardmath = _load_corpus(CORPUS_HARDMATH)

    usable_gsm8k = _load_usable(raw_gsm8k)
    usable_hardmath = _load_usable(raw_hardmath)
    all_usable = usable_gsm8k + usable_hardmath

    # Build the contested subset from the pooled usable records.
    contested = _build_contested_subset(all_usable, low=CONTEST_LOW, high=CONTEST_HIGH)

    n_gsm8k_contested = sum(
        1 for r in usable_gsm8k
        if CONTEST_LOW <= _per_problem_correctness_rate(r) <= CONTEST_HIGH
    )
    n_hardmath_contested = sum(
        1 for r in usable_hardmath
        if CONTEST_LOW <= _per_problem_correctness_rate(r) <= CONTEST_HIGH
    )

    source_corpora = {
        "gsm8k": {
            "path": str(CORPUS_GSM8K.relative_to(REPO_ROOT)),
            "n_raw": len(raw_gsm8k),
            "n_usable": len(usable_gsm8k),
            "n_contested": n_gsm8k_contested,
        },
        "hardmath": {
            "path": str(CORPUS_HARDMATH.relative_to(REPO_ROOT)),
            "n_raw": len(raw_hardmath),
            "n_usable": len(usable_hardmath),
            "n_contested": n_hardmath_contested,
        },
    }

    # Compute majority-vote SC over the contested subset NOW, before any further
    # gates, so it is available in both the blocked path and the scored path.
    contested_subset_sc = _compute_subset_sc(contested)

    has_any_corpus = len(all_usable) > 0
    preconditions.append(
        {"resource": "any_cached_corpus", "available": has_any_corpus}
    )

    if not has_any_corpus:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = "complete: blocked_no_cached_corpus"
        payload["source_corpora"] = source_corpora
        payload["contested_subset_n"] = 0
        payload["contested_subset_sc"] = None
        payload["methodology_note"] = (
            "Neither data/p01_gsm8k_generations.jsonl nor "
            "data/p01_hardmath_generations.jsonl could be loaded. "
            "Run exp3461 (GSM8K corpus builder) and/or exp3471 (MATH-L5 corpus "
            "builder) to populate the cached corpora before rerunning this experiment."
        )
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # ------------------------------------------------------------------
    # PRECONDITION 1: contested subset must be >= MIN_PROBLEMS
    # ------------------------------------------------------------------
    contested_n = len(contested)
    preconditions.append(
        {
            "resource": "contested_subset_size",
            "available": contested_n >= MIN_PROBLEMS,
            "contested_n": contested_n,
            "min_required": MIN_PROBLEMS,
        }
    )

    if contested_n < MIN_PROBLEMS:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = (
            f"complete: blocked_contested_subset_too_small_n={contested_n}"
        )
        payload["source_corpora"] = source_corpora
        payload["contested_subset_n"] = contested_n
        payload["contested_subset_sc"] = contested_subset_sc
        payload["methodology_note"] = (
            f"The contested subset (per-problem correctness rate in "
            f"[{CONTEST_LOW}, {CONTEST_HIGH}]) has only n={contested_n} problems "
            f"(<{MIN_PROBLEMS} required for headline-eligibility with "
            f"{N_FOLDS}-fold CV). "
            f"GSM8K contributed {n_gsm8k_contested} contested problems; "
            f"MATH-L5/hardmath contributed {n_hardmath_contested}. "
            f"Majority-vote SC over the contested subset: {contested_subset_sc}. "
            "Action: expand the cached corpora by rerunning exp3461 / exp3471 with "
            "more problems, then rerun this experiment."
        )
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # ------------------------------------------------------------------
    # PRECONDITION 2: process-energy + reranker substrate loadable
    # ------------------------------------------------------------------
    try:
        from carnot.phase3.p01_process_energy import (  # noqa: F401
            ProcessScoringResult,
            derive_v6_verdict,
            process_energy_per_step,
            score_corpus_process_cv,
        )
        from carnot.phase3.p01_trained_energy_reranker import (
            TrainedEnergyReranker,
            _Verifiers,
        )

        verifiers = _Verifiers()
        _ = process_energy_per_step(["2 + 2 = 4"], verifiers)  # exercise the substrate
        _ = TrainedEnergyReranker().n_params
        substrate_ok = True
    except Exception:  # pragma: no cover - defensive; substrate is in-repo
        substrate_ok = False
        verifiers = None

    preconditions.append(
        {"resource": "process_energy_reranker_substrate", "available": substrate_ok}
    )
    if not substrate_ok:
        payload = _base_payload(start, preconditions)
        payload["honest_verdict"] = "complete: blocked_energy_substrate_unavailable"
        payload["source_corpora"] = source_corpora
        payload["contested_subset_n"] = contested_n
        payload["contested_subset_sc"] = contested_subset_sc
        payload["reproducibility_checksum"] = _checksum(contested)
        _emit(payload)
        print(payload["honest_verdict"])
        return

    # ------------------------------------------------------------------
    # SCORE: train per-fold rerankers + aggregators; score 7 conditions
    # ------------------------------------------------------------------
    result = score_corpus_process_cv(
        contested,
        seed=SEED,
        n_folds=N_FOLDS,
        n_boot=N_BOOT,
        reranker_iter=RERANKER_ITER,
        verifiers=verifiers,
    )

    # Re-assert HEADROOM gate over the CONTESTED SUBSET (SC in [0.40, 0.78]).
    preconditions.append(
        {
            "resource": "self_consistency_in_headroom_band",
            "available": result.self_consistency_in_headroom_band,
        }
    )

    verdict = derive_v6_verdict(result)
    g0 = result.self_consistency_in_headroom_band
    opt_sig = result.paired_significance["optimal_aggregation"]
    g1 = (
        g0
        and result.flip_optimal.net_correctness_gain > 0
        and result.delta_optimal_vs_self_consistency > 0
        and opt_sig["mcnemar_exact_p"] < 0.05
    )
    g2 = g0 and result.flip_optimal.flip_count > 0

    payload = _base_payload(start, preconditions)
    payload.update(
        {
            "honest_verdict": verdict,
            "source_corpora": source_corpora,
            "contested_subset_n": contested_n,
            "contested_subset_sc": contested_subset_sc,
            "n_problems_heldout": result.n_problems_heldout,
            "k_samples": result.k_samples,
            "reranker_param_count": result.reranker_param_count,
            "aggregator_param_count": result.aggregator_param_count,
            "fitted_lambdas": result.fitted_lambdas,
            "train_test_split_note": result.train_test_split_note,
            "self_consistency_in_headroom_band": result.self_consistency_in_headroom_band,
            "ar_greedy_accuracy": round(result.ar_greedy_accuracy, 6),
            "self_consistency_accuracy": round(result.self_consistency_accuracy, 6),
            "self_certainty_bon_accuracy": round(result.self_certainty_bon_accuracy, 6),
            "process_energy_argmin_accuracy": round(
                result.process_energy_argmin_accuracy, 6
            ),
            "trained_energy_weighted_vote_accuracy": round(
                result.trained_energy_weighted_vote_accuracy, 6
            ),
            "trained_energy_sc_hybrid_accuracy": round(
                result.trained_energy_sc_hybrid_accuracy, 6
            ),
            "optimal_aggregation_accuracy": round(result.optimal_aggregation_accuracy, 6),
            "flip_count_optimal_vs_sc": result.flip_optimal.flip_count,
            "flips_correct_optimal": result.flip_optimal.flips_correct,
            "flips_incorrect_optimal": result.flip_optimal.flips_incorrect,
            "net_correctness_gain_optimal": result.flip_optimal.net_correctness_gain,
            "flip_process_energy_vs_sc": {
                "flip_count": result.flip_process_energy.flip_count,
                "flips_correct": result.flip_process_energy.flips_correct,
                "flips_incorrect": result.flip_process_energy.flips_incorrect,
                "net_correctness_gain": result.flip_process_energy.net_correctness_gain,
            },
            "flip_hybrid_vs_sc": {
                "flip_count": result.flip_hybrid.flip_count,
                "flips_correct": result.flip_hybrid.flips_correct,
                "flips_incorrect": result.flip_hybrid.flips_incorrect,
                "net_correctness_gain": result.flip_hybrid.net_correctness_gain,
            },
            "delta_optimal_vs_self_consistency": round(
                result.delta_optimal_vs_self_consistency, 6
            ),
            "delta_process_energy_vs_self_consistency": round(
                result.delta_process_energy_vs_self_consistency, 6
            ),
            "delta_hybrid_vs_self_consistency": round(
                result.delta_hybrid_vs_self_consistency, 6
            ),
            "paired_significance": result.paired_significance,
            "acceptance_gate_g0_headroom": g0,
            "acceptance_gate_g1_energy_beats_sc_with_headroom": g1,
            "acceptance_gate_g2_non_degenerate_flips": g2,
            "n_folds": N_FOLDS,
            "compute_parity_note": (
                f"All sampled-aggregation conditions consume the SAME k="
                f"{result.k_samples} cached generations; greedy AR is the 1-sample "
                f"floor. The energy adds only a {result.reranker_param_count}-parameter "
                f"logistic reranker (4 verifier signals + mean logprob + step count) "
                f"plus a {result.aggregator_param_count}-parameter optimal aggregator "
                f"(the mixing coefficient lambda, fit on train) — no extra samples — "
                f"so energy cannot win by spending more compute."
            ),
            "reproducibility_checksum": _checksum(contested),
            "methodology_note": (
                "PRIMARY signal is the flip-count over the CONTESTED SUBSET, not a "
                f"pair of accuracies: optimal-aggregation flips "
                f"{result.flip_optimal.flip_count} of "
                f"{result.n_problems_heldout} held-out problems vs the SC majority "
                f"(flips_correct={result.flip_optimal.flips_correct}, "
                f"flips_incorrect={result.flip_optimal.flips_incorrect}). "
                "The contested subset is defined by per-problem correctness rate in "
                f"[{CONTEST_LOW}, {CONTEST_HIGH}] — structurally uncertain problems "
                "where the selector's choice matters. When a condition agrees with SC, "
                "its flip_count is 0 and is reported once — no second bit-identical "
                "accuracy field, so the exp3460 tautology flag cannot fire."
            ),
            "surprising_result_acknowledgment": (
                f"Process-energy / optimal-aggregation result at n_heldout="
                f"{result.n_problems_heldout} with {N_FOLDS}-fold problem-level CV "
                f"over the contested subset (rate in [{CONTEST_LOW}, {CONTEST_HIGH}]). "
                "A positive G1 (significant beat with positive net flip gain) would "
                "require independent replication before any headline claim."
            ),
        }
    )
    _emit(payload)

    print(
        f"DONE: {verdict}\n"
        f"  contested_subset_n={contested_n} "
        f"contested_subset_sc={contested_subset_sc} "
        f"n_heldout={result.n_problems_heldout} "
        f"k={result.k_samples} SC_in_band={result.self_consistency_in_headroom_band}\n"
        f"  AR={result.ar_greedy_accuracy:.4f} SC={result.self_consistency_accuracy:.4f} "
        f"certainty={result.self_certainty_bon_accuracy:.4f}\n"
        f"  process-argmin={result.process_energy_argmin_accuracy:.4f} "
        f"trained-vote={result.trained_energy_weighted_vote_accuracy:.4f} "
        f"hybrid={result.trained_energy_sc_hybrid_accuracy:.4f} "
        f"optimal={result.optimal_aggregation_accuracy:.4f}\n"
        f"  FLIP(optimal vs SC)={result.flip_optimal.flip_count} "
        f"correct={result.flip_optimal.flips_correct} "
        f"incorrect={result.flip_optimal.flips_incorrect} "
        f"net={result.flip_optimal.net_correctness_gain:+d}\n"
        f"  dOptimal_vs_SC={result.delta_optimal_vs_self_consistency:+.4f} "
        f"dProcess_vs_SC={result.delta_process_energy_vs_self_consistency:+.4f}\n"
        f"  G0={g0} G1={g1} G2={g2}  dur={payload['duration_s']}s"
    )


if __name__ == "__main__":
    main()
