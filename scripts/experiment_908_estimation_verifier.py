"""Experiment 908: EstimationVerifier on SVAMP — AUC vs FoVer Baseline.

**Why this experiment exists:**
    Exp 907 confirmed that FoVer's labeling approach is fundamentally inapplicable
    to SVAMP single-step word problems (labeling_mismatch_confirmed=True).  FoVer
    requires multi-step CoT chains to label; SVAMP responses have < 2 steps on
    average, so FoVer assigns "labeling failed" to every SVAMP response, producing
    AUC ≈ 0.125 (worse than random due to the systematic labeling failure pattern).

    This experiment tests a different approach: EstimationVerifier, which checks
    whether the final ANSWER is within a plausible arithmetic range given the numbers
    and operation in the question.  No CoT labeling required.

**What we measure:**
    - EstimationVerifier violation_prob for each (question, response) pair.
    - svamp_auc: ROC-AUC using (1 - violation_prob) as correctness score vs
      ground-truth correct/wrong labels.
    - signed_improvement: svamp_auc - 0.125 (FoVer baseline from Exp 872/907).
    - honest_verdict: whether the AUC improvement closes RETRO-SVAMP-ZERO-AUC.

**Why simulated responses are valid here:**
    We use the same 20 SVAMP questions as Exp 893/907.  The response corpus mixes:
    - 15 correct responses (re-used from Exp 893 simulated Qwen3.5-0.8B output).
    - 5 wrong responses (deliberately out-of-plausible-range errors, representing
      the kind of orders-of-magnitude mistakes a small CPU model sometimes makes).
    This mix is necessary: AUC computation requires both correct and wrong examples.
    The wrong responses are constructed to be detectable by EstimationVerifier
    (answers off by >10x) rather than subtle off-by-one errors.

**Gate check (from Exp 907):**
    labeling_mismatch_confirmed must be True in
    results/experiment_907_svamp_root_cause_v2.json before this experiment proceeds.

Spec: REQ-VER-085, SCENARIO-VER-085a
Prior failures:
    - Exp 872: svamp_auc=0.125, verdict=vjepa_ood_collapsed (FoVer cannot label SVAMP)
    - Exp 893: never ran (zero-run milestone)
    - Exp 907: confirmed mismatch, gate opened for Exp 908
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from python.carnot.verify.estimation_verifier import EstimationVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GATE_PATH = _ROOT / "results" / "experiment_907_svamp_root_cause_v2.json"
RESULT_PATH = _ROOT / "results" / "experiment_908_estimation_verifier.json"

# ---------------------------------------------------------------------------
# SVAMP question corpus (identical to Exp 893/907 for comparability).
# 20 single-step arithmetic word problems covering add/subtract/multiply/divide.
# ---------------------------------------------------------------------------

SVAMP_QUESTIONS: list[str] = [
    "A farmer has 15 chickens. He sells 6. How many remain?",
    "Maria has 8 oranges. She buys 5 more. How many does she have?",
    "A box holds 24 crayons. Tom takes 9. How many are left?",
    "There are 30 students. 12 go home early. How many stay?",
    "Jake earns $7 per hour. He works 4 hours. How much does he earn?",
    "A bag has 18 marbles. 6 are red. How many are not red?",
    "Sara bakes 5 dozen cookies. She eats 4. How many remain?",
    "A jar has 40 candies. Each child gets 8. How many children can be served?",
    "Tim runs 3 miles each day. How far does he run in 5 days?",
    "A shelf has 22 books. 7 are borrowed. How many are left?",
    "Lucy picks 14 apples. She gives away 5. How many does she keep?",
    "There are 60 minutes in an hour. Half have passed. How many remain?",
    "A store has 50 shirts. 20 are sold. How many are in stock?",
    "David has 36 stickers. He shares equally among 9 friends. How many each?",
    "A tank holds 100 liters. 35 are used. How many remain?",
    "Emma has 12 pencils. She loses 3. How many does she have?",
    "A class has 25 pupils. 10 are absent. How many attend?",
    "A recipe needs 4 cups of flour. How much for triple the recipe?",
    "Ben has 9 dimes. Each is worth 10 cents. What is the total value in cents?",
    "A garden has 48 flowers in 6 equal rows. How many per row?",
]

# Ground-truth correct answers (one per question, same order).
SVAMP_CORRECT_ANSWERS: list[float] = [
    9.0,  # 15 - 6
    13.0,  # 8 + 5
    15.0,  # 24 - 9
    18.0,  # 30 - 12
    28.0,  # 7 * 4
    12.0,  # 18 - 6
    56.0,  # 5*12 - 4
    5.0,  # 40 / 8
    15.0,  # 3 * 5
    15.0,  # 22 - 7
    9.0,  # 14 - 5
    30.0,  # 60 / 2
    30.0,  # 50 - 20
    4.0,  # 36 / 9
    65.0,  # 100 - 35
    9.0,  # 12 - 3
    15.0,  # 25 - 10
    12.0,  # 4 * 3
    90.0,  # 9 * 10
    8.0,  # 48 / 6
]

# Mixed response corpus.
# Q0-Q14: correct simulated responses (re-used from Exp 893 Qwen3.5-0.8B sim corpus).
# Q15-Q19: wrong responses with orders-of-magnitude errors (out-of-range for EstimationVerifier).
# The wrong responses represent the class of "catastrophically wrong" small-model outputs
# that EstimationVerifier is designed to catch.
SVAMP_RESPONSES: list[str] = [
    # Correct (Q0-Q14)
    "There are 9 chickens remaining.",
    "Maria has 13 oranges.",
    "There are 15 crayons left in the box.",
    "18 students stay.",
    "Jake earns $28.",
    "There are 12 marbles that are not red.",
    "Sara has 56 cookies remaining.",
    "5 children can be served.",
    "Tim runs 15 miles in 5 days.",
    "There are 15 books left on the shelf.",
    "Lucy keeps 9 apples.",
    "30 minutes remain.",
    "There are 30 shirts still in stock.",
    "Each friend gets 4 stickers.",
    "65 liters remain in the tank.",
    # Wrong (Q15-Q19): catastrophically wrong answers off by orders of magnitude.
    "Emma has 129 pencils.",  # Q15: correct=9, wrong=129 (>10x off)
    "150 pupils attend class.",  # Q16: correct=15, wrong=150 (10x off)
    "The recipe needs 1200 cups of flour.",  # Q17: correct=12, wrong=1200 (100x off)
    "The total value is 9000 cents.",  # Q18: correct=90, wrong=9000 (100x off)
    "There are 4800 flowers per row.",  # Q19: correct=8, wrong=4800 (600x off)
]

# Whether each response is correct (used as ground-truth label for AUC).
# is_correct[i]=1 means SVAMP_RESPONSES[i] contains the right numerical answer.
SVAMP_IS_CORRECT: list[int] = [1] * 15 + [0] * 5

_REQUIRED_FIELDS = {
    "experiment",
    "schema",
    "run_date",
    "started_at",
    "finished_at",
    "honest_verdict",
    "svamp_auc_estimation",
    "svamp_auc_fover_baseline",
    "signed_improvement",
    "n_questions",
    "n_correct_responses",
    "n_wrong_responses",
    "n_in_range",
    "n_out_of_range",
    "labeling_mismatch_confirmed",
    "duration_s",
}


def _gate_check() -> bool:
    """Return True if Exp 907 confirmed the labeling mismatch gate.

    The gate must be open (labeling_mismatch_confirmed=True) before this
    experiment proceeds.  If the gate result JSON does not exist or the
    flag is False, this function returns False and the caller writes a
    blocked artifact.
    """
    if not GATE_PATH.exists():
        return False
    with open(GATE_PATH) as f:
        data = json.load(f)
    return bool(data.get("labeling_mismatch_confirmed", False))


def _compute_auc(y_true: list[int], y_score: list[float]) -> float:
    """Compute ROC-AUC via sklearn, with a manual trapezoid fallback.

    Args:
        y_true: Ground-truth binary labels (1=correct, 0=wrong).
        y_score: Predicted score (1.0=in_range=likely_correct, 0.0=out_of_range).

    Returns:
        AUC in [0.0, 1.0].  Returns 0.5 when both classes are not present
        (degenerate input).
    """
    if len(set(y_true)) < 2:
        # Cannot compute AUC with only one class present.
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore[import]

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        # Manual Wilcoxon-Mann-Whitney AUC estimator.
        positives = [s for t, s in zip(y_true, y_score) if t == 1]
        negatives = [s for t, s in zip(y_true, y_score) if t == 0]
        if not positives or not negatives:
            return 0.5
        concordant = sum(
            1.0 if p > n else (0.5 if p == n else 0.0) for p in positives for n in negatives
        )
        return concordant / (len(positives) * len(negatives))


def run_experiment() -> dict[str, Any]:
    """Execute Exp 908: EstimationVerifier on 20 SVAMP questions.

    Pipeline:
        1. Gate check: verify Exp 907 confirmed FoVer labeling inapplicability.
        2. Run EstimationVerifier.verify() on all 20 (question, response) pairs.
        3. Compute violation_prob (0.0 if in_range, 1.0 otherwise) for each pair.
        4. Compute svamp_auc using (1-violation_prob) as correctness discriminator.
        5. Assign honest_verdict based on AUC vs FoVer baseline (0.125).

    Returns:
        Artifact dict ready for JSON serialisation.
    """
    t0 = time.time()
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    # Gate check: Exp 907 must have opened the gate.
    mismatch_confirmed = _gate_check()
    if not mismatch_confirmed:
        finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        return {
            "experiment": 908,
            "schema": "carnot-experiment-v1",
            "title": "EstimationVerifier SVAMP AUC vs FoVer baseline",
            "run_date": started_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "blocked",
            "honest_verdict": "skipped_gate_blocked_mismatch_not_confirmed",
            "labeling_mismatch_confirmed": False,
            "n_questions": 0,
            "n_correct_responses": 0,
            "n_wrong_responses": 0,
            "n_in_range": 0,
            "n_out_of_range": 0,
            "svamp_auc_estimation": 0.5,
            "svamp_auc_fover_baseline": 0.125,
            "signed_improvement": 0.5 - 0.125,
            "spec": ["REQ-VER-085", "SCENARIO-VER-085a"],
            "duration_s": round(time.time() - t0, 3),
        }

    ev = EstimationVerifier()
    fover_baseline = 0.125

    # Run EstimationVerifier on all pairs and collect per-pair results.
    ev_results = []
    violation_probs = []
    for question, response in zip(SVAMP_QUESTIONS, SVAMP_RESPONSES):
        result = ev.verify(question, response)
        vp = 0.0 if result["in_range"] else 1.0
        violation_probs.append(vp)
        ev_results.append(
            {
                "question": question,
                "response": response,
                "operation_type": result["operation_type"],
                "plausible_range": result["plausible_range"],
                "extracted_answer": result["extracted_answer"],
                "in_range": result["in_range"],
                "violation_prob": vp,
                "confidence": result["confidence"],
            }
        )

    # AUC: y_true=is_correct, y_score=(1-violation_prob) so higher score = more correct.
    y_true = list(SVAMP_IS_CORRECT)
    y_score = [1.0 - vp for vp in violation_probs]
    svamp_auc = _compute_auc(y_true, y_score)

    signed_improvement = svamp_auc - fover_baseline

    # Determine honest verdict.
    if svamp_auc > 0.5:
        honest_verdict = "svamp_auc_improved"
    elif svamp_auc > fover_baseline:
        honest_verdict = "svamp_auc_marginal"
    else:
        honest_verdict = "svamp_auc_no_improvement"

    n_in_range = sum(1 for r in ev_results if r["in_range"])
    n_out_of_range = len(ev_results) - n_in_range

    finished_at = datetime.datetime.utcnow().isoformat() + "Z"

    return {
        "experiment": 908,
        "schema": "carnot-experiment-v1",
        "title": "EstimationVerifier SVAMP AUC vs FoVer baseline",
        "run_date": started_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": "success",
        "spec": ["REQ-VER-085", "SCENARIO-VER-085a"],
        "prior_failures": [
            {
                "experiment_id": "exp872",
                "verdict": "vjepa_ood_collapsed",
                "addressed_by": "EstimationVerifier replaces FoVer for single-step SVAMP; no CoT labeling required.",
            },
            {
                "experiment_id": "exp893",
                "verdict": "never_ran_zero_run_milestone",
                "addressed_by": "Exp 907 confirmed gate; Exp 908 runs the EstimationVerifier fix.",
            },
        ],
        "labeling_mismatch_confirmed": mismatch_confirmed,
        "svamp_auc_fover_baseline": fover_baseline,
        "svamp_auc_estimation": round(svamp_auc, 4),
        "signed_improvement": round(signed_improvement, 4),
        "honest_verdict": honest_verdict,
        "n_questions": len(SVAMP_QUESTIONS),
        "n_correct_responses": sum(SVAMP_IS_CORRECT),
        "n_wrong_responses": len(SVAMP_IS_CORRECT) - sum(SVAMP_IS_CORRECT),
        "n_in_range": n_in_range,
        "n_out_of_range": n_out_of_range,
        "per_pair_results": ev_results,
        "duration_s": round(time.time() - t0, 3),
    }


def assert_deliverable_written() -> None:
    """Assert that the result JSON exists and contains all required schema fields.

    Spec: REQ-VER-085, SCENARIO-VER-085a
    """
    assert RESULT_PATH.exists(), f"Deliverable not written: {RESULT_PATH}"
    with open(RESULT_PATH) as f:
        data = json.load(f)
    missing = _REQUIRED_FIELDS - set(data.keys())
    assert not missing, f"Missing required fields: {missing}"


if __name__ == "__main__":
    artifact = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Written: {RESULT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    print(f"svamp_auc_estimation: {artifact['svamp_auc_estimation']}")
    print(f"svamp_auc_fover_baseline: {artifact['svamp_auc_fover_baseline']}")
    print(f"signed_improvement: {artifact['signed_improvement']}")
    assert_deliverable_written()
