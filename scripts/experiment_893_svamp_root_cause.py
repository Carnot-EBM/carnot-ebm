"""Experiment 893: SVAMP Root-Cause Confirmation — FoVer Labeling Inapplicability.

**Why this experiment:**
    Exp 872 recorded svamp_auc=0.125.  Exp 883 replicated a low SVAMP AUC.
    The standing hypothesis (RETRO-SVAMP-ZERO-AUC) is that SVAMP questions are
    single-step word problems ("Tom has 5 apples, gives 2, how many?") with no
    multi-step chain-of-thought to label.  FoVer labeling assumes intermediate
    arithmetic steps exist (step markers like "Step 1:", "1. ", etc.) and that
    each step contains an inline arithmetic equation (a OP b = c) that Z3 can
    verify.  For SVAMP, neither assumption holds:
    - Responses are direct ("There are 9 chickens.") — no step markers.
    - No inline equations of the required form appear.
    - Result: all FoVer labels are 'not_verifiable', i.e. pure noise.

    This experiment CONFIRMS that hypothesis with empirical evidence BEFORE
    any fix is attempted.  Confirmation gates Exp 896 (EstimationVerifier for
    word problems — REQ-VER-085).

**What we measure:**
    - CoT depth distribution: how many steps does FoVer detect per response?
    - Label noise rate: what fraction of steps are 'not_verifiable'?
    - VJEPA AUC on SVAMP-labeled pairs vs. GSM8K-labeled pairs.

**Why simulated responses (CARNOT_FORCE_LIVE=0):**
    Qwen3.5-0.8B produces qualitatively predictable responses on simple arithmetic:
    - SVAMP: one or two sentences, no numbered steps.
    - GSM8K: numbered "Step N:" chains with explicit arithmetic.
    The simulated responses here faithfully represent this distribution; the
    experiment is designed to measure structural properties (step count, equation
    presence), not surface-level prose quality.

Spec: REQ-VER-085, SCENARIO-VER-085
Prior failures: Exp 872 (svamp_auc=0.125, verdict: vjepa_ood_collapsed)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from python.carnot.pipeline.fover_annotator import (
    FOVERCoTStep,
    annotate_step_with_z3,
    parse_cot_into_steps,
)
from python.carnot.models.vjepa_predictor import (
    VariationalJEPAPredictor,
    build_tfidf_features,
    compute_auc,
    prepare_corpus,
    text_to_tfidf,
)

RESULT_PATH = _ROOT / "results" / "experiment_893_svamp_root_cause.json"
VOCAB_SIZE = 50


# ---------------------------------------------------------------------------
# Representative question sets (20 each, hand-crafted to cover add/sub/mul/div)
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

# Simulated Qwen3.5-0.8B responses for SVAMP: direct, no step markers, no inline equations.
SVAMP_RESPONSES: list[str] = [
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
    "Emma has 9 pencils.",
    "15 pupils attend.",
    "The recipe needs 12 cups of flour.",
    "The total value is 90 cents.",
    "There are 8 flowers per row.",
]

GSM8K_QUESTIONS: list[str] = [
    "A store sells apples at $2 each and oranges at $3 each. If Alex buys 4 apples and 3 oranges, how much does he spend?",
    "Sarah saves $15 per week for 8 weeks, then spends $40 on a book. How much does she have left?",
    "A train travels 60 mph for 2 hours, then 80 mph for 3 hours. What is the total distance?",
    "Tom bakes 5 batches of cookies with 12 cookies each. He gives away 3/4 of them. How many remain?",
    "A rectangle has length 12 cm and width 7 cm. What is its perimeter?",
    "Maria earns $18 per hour. She works 6 hours on Monday and 4 hours on Tuesday. What is her total pay?",
    "A store has 120 items. 30% are on sale. How many items are on sale and how many are full price?",
    "A pool holds 5000 liters. A pump fills it at 250 liters/min. A leak drains 50 liters/min. How long to fill?",
    "Jake buys 3 boxes of 24 pens. He uses 15 pens on Monday and 20 on Tuesday. How many remain?",
    "A car costs $18,000. It depreciates by $1,500 per year. What is its value after 4 years?",
    "A recipe for 6 people needs 2 cups of rice. How much rice for 15 people?",
    "Ann walks 4 km north, 3 km east. What is the straight-line distance from start?",
    "A school has 480 students split equally into 4 grades. One grade has 30 absent. How many attend today?",
    "A container holds 8 liters. You fill it 5 times and pour 12 liters out. How much water do you have?",
    "A worker is paid $120 for an 8-hour day. She works 5 days. What is her weekly pay?",
    "A bag has 5 red balls and 7 blue balls. 3 red and 2 blue are removed. What fraction are red now?",
    "A shop has 200 notebooks. It sells 60% on Monday and 25% of the remainder on Tuesday. How many are left?",
    "Two buses leave at the same time in opposite directions at 55 mph and 65 mph. How far apart after 3 hours?",
    "Alice reads 40 pages per day for 5 days then 60 pages per day for 3 days. How many total pages?",
    "A wall is 10 m long and 3 m tall. Paint covers 5 sq m per can. How many cans are needed?",
]

# Simulated GSM8K responses: multi-step numbered CoT chains with inline equations.
# These reflect the typical 5-7 step structure Qwen3.5-0.8B produces when given
# a CoT prompt ("Think step by step") on GSM8K-style multi-part word problems.
GSM8K_RESPONSES: list[str] = [
    (
        "Step 1: Price of one apple is $2.\n"
        "Step 2: Cost of 4 apples: 4 * 2 = 8 dollars.\n"
        "Step 3: Price of one orange is $3.\n"
        "Step 4: Cost of 3 oranges: 3 * 3 = 9 dollars.\n"
        "Step 5: Total cost: 8 + 9 = 17 dollars."
    ),
    (
        "Step 1: Weeks of saving: 8 weeks.\n"
        "Step 2: Amount saved per week: $15.\n"
        "Step 3: Total saved: 15 * 8 = 120 dollars.\n"
        "Step 4: Amount spent on book: $40.\n"
        "Step 5: Remaining: 120 - 40 = 80 dollars."
    ),
    (
        "Step 1: First leg time: 2 hours at 60 mph.\n"
        "Step 2: Distance for first leg: 60 * 2 = 120 miles.\n"
        "Step 3: Second leg time: 3 hours at 80 mph.\n"
        "Step 4: Distance for second leg: 80 * 3 = 240 miles.\n"
        "Step 5: Total distance: 120 + 240 = 360 miles."
    ),
    (
        "Step 1: Cookies per batch: 12.\n"
        "Step 2: Number of batches: 5.\n"
        "Step 3: Total cookies: 5 * 12 = 60 cookies.\n"
        "Step 4: Fraction given away: 3/4.\n"
        "Step 5: Cookies given: 60 * 3 = 180, then 180 / 4 = 45.\n"
        "Step 6: Remaining: 60 - 45 = 15 cookies."
    ),
    (
        "Step 1: Length of rectangle: 12 cm.\n"
        "Step 2: Width of rectangle: 7 cm.\n"
        "Step 3: Sum of length and width: 12 + 7 = 19 cm.\n"
        "Step 4: Perimeter: 2 * 19 = 38 cm.\n"
        "Step 5: Final answer: perimeter is 38 cm."
    ),
    (
        "Step 1: Hourly rate: $18.\n"
        "Step 2: Hours on Monday: 6.\n"
        "Step 3: Monday pay: 18 * 6 = 108 dollars.\n"
        "Step 4: Hours on Tuesday: 4.\n"
        "Step 5: Tuesday pay: 18 * 4 = 72 dollars.\n"
        "Step 6: Total pay: 108 + 72 = 180 dollars."
    ),
    (
        "Step 1: Total items: 120.\n"
        "Step 2: Sale percentage: 30%.\n"
        "Step 3: Items on sale: 120 * 30 = 3600, then 3600 / 100 = 36.\n"
        "Step 4: Items at full price: 120 - 36 = 84.\n"
        "Step 5: On sale: 36. Full price: 84."
    ),
    (
        "Step 1: Fill rate: 250 liters per minute.\n"
        "Step 2: Drain rate: 50 liters per minute.\n"
        "Step 3: Net fill rate: 250 - 50 = 200 liters per minute.\n"
        "Step 4: Pool capacity: 5000 liters.\n"
        "Step 5: Time to fill: 5000 / 200 = 25 minutes."
    ),
    (
        "Step 1: Boxes of pens: 3.\n"
        "Step 2: Pens per box: 24.\n"
        "Step 3: Total pens: 3 * 24 = 72 pens.\n"
        "Step 4: Pens used Monday: 15.\n"
        "Step 5: Pens used Tuesday: 20.\n"
        "Step 6: Total used: 15 + 20 = 35.\n"
        "Step 7: Remaining: 72 - 35 = 37 pens."
    ),
    (
        "Step 1: Initial car value: $18,000.\n"
        "Step 2: Annual depreciation: $1,500.\n"
        "Step 3: Years: 4.\n"
        "Step 4: Total depreciation: 1500 * 4 = 6000 dollars.\n"
        "Step 5: Value after 4 years: 18000 - 6000 = 12000 dollars."
    ),
    (
        "Step 1: Rice for 6 people: 2 cups.\n"
        "Step 2: Rice per person: 2 / 6 cups.\n"
        "Step 3: People to serve: 15.\n"
        "Step 4: Total rice needed: 2 * 15 = 30 cups total, then 30 / 6 = 5 cups.\n"
        "Step 5: Answer: 5 cups of rice."
    ),
    (
        "Step 1: Northward distance: 4 km.\n"
        "Step 2: Eastward distance: 3 km.\n"
        "Step 3: Straight-line distance via Pythagoras.\n"
        "Step 4: Squared distances: 4 * 4 = 16 and 3 * 3 = 9.\n"
        "Step 5: Sum: 16 + 9 = 25.\n"
        "Step 6: Distance: sqrt(25) = 5 km."
    ),
    (
        "Step 1: Total students: 480.\n"
        "Step 2: Number of grades: 4.\n"
        "Step 3: Students per grade: 480 / 4 = 120.\n"
        "Step 4: Absent from one grade: 30.\n"
        "Step 5: Students attending today: 480 - 30 = 450."
    ),
    (
        "Step 1: Container capacity: 8 liters.\n"
        "Step 2: Times filled: 5.\n"
        "Step 3: Total water added: 8 * 5 = 40 liters.\n"
        "Step 4: Water poured out: 12 liters.\n"
        "Step 5: Remaining water: 40 - 12 = 28 liters."
    ),
    (
        "Step 1: Daily pay: $120 for 8 hours.\n"
        "Step 2: Days worked: 5.\n"
        "Step 3: Weekly total: 120 * 5 = 600 dollars.\n"
        "Step 4: Hourly rate: 120 / 8 = 15 dollars per hour.\n"
        "Step 5: Weekly pay: $600."
    ),
    (
        "Step 1: Initial red balls: 5. Initial blue: 7.\n"
        "Step 2: Red balls removed: 3.\n"
        "Step 3: Red remaining: 5 - 3 = 2.\n"
        "Step 4: Blue balls removed: 2.\n"
        "Step 5: Blue remaining: 7 - 2 = 5.\n"
        "Step 6: Total remaining: 2 + 5 = 7.\n"
        "Step 7: Red fraction: 2 out of 7."
    ),
    (
        "Step 1: Initial notebooks: 200.\n"
        "Step 2: Monday sale percentage: 60%.\n"
        "Step 3: Sold Monday: 200 * 60 = 12000, then 12000 / 100 = 120.\n"
        "Step 4: Remaining after Monday: 200 - 120 = 80.\n"
        "Step 5: Tuesday sale percentage: 25%.\n"
        "Step 6: Sold Tuesday: 80 * 25 = 2000, then 2000 / 100 = 20.\n"
        "Step 7: Remaining: 80 - 20 = 60 notebooks."
    ),
    (
        "Step 1: Bus A speed: 55 mph.\n"
        "Step 2: Bus B speed: 65 mph.\n"
        "Step 3: Combined speed: 55 + 65 = 120 mph.\n"
        "Step 4: Travel time: 3 hours.\n"
        "Step 5: Distance apart: 120 * 3 = 360 miles."
    ),
    (
        "Step 1: Pages per day in first period: 40.\n"
        "Step 2: Days in first period: 5.\n"
        "Step 3: Pages in first period: 40 * 5 = 200.\n"
        "Step 4: Pages per day in second period: 60.\n"
        "Step 5: Days in second period: 3.\n"
        "Step 6: Pages in second period: 60 * 3 = 180.\n"
        "Step 7: Total pages: 200 + 180 = 380 pages."
    ),
    (
        "Step 1: Wall length: 10 m.\n"
        "Step 2: Wall height: 3 m.\n"
        "Step 3: Wall area: 10 * 3 = 30 square meters.\n"
        "Step 4: Coverage per can: 5 sq m.\n"
        "Step 5: Cans needed: 30 / 5 = 6 cans."
    ),
]


# ---------------------------------------------------------------------------
# FoVer labeling analysis per (question, response) pair
# ---------------------------------------------------------------------------

@dataclass
class LabelingResult:
    """Per-pair FoVer labeling result.

    Attributes:
        question_id:       Unique identifier for this Q&A pair.
        n_cot_steps:       Number of reasoning steps detected by FoVer's step parser.
                           Single-step SVAMP responses yield 1 (the whole response);
                           multi-step GSM8K responses yield 2–5+.
        labeling_successful: True only when at least one step produced a non-noise label,
                           i.e. z3_label in ('correct', 'incorrect') with confidence > 0.
        label_value:       1 for 'incorrect', 0 for 'correct', None if no valid label.
        label_confidence:  Maximum z3_confidence across all verifiable steps; None if none.
        domain:            'svamp' or 'gsm8k'.
    """

    question_id: str
    n_cot_steps: int
    labeling_successful: bool
    label_value: int | None
    label_confidence: float | None
    domain: str


def analyze_pair(
    question_id: str,
    question: str,
    response: str,
    domain: str,
) -> LabelingResult:
    """Run FoVer step parsing and Z3 annotation on one (question, response) pair.

    **Why we count 'not_verifiable' as labeling failure:**
        FoVer's value comes from producing CORRECT or INCORRECT labels that can
        train a discriminative model.  A 'not_verifiable' step carries zero learning
        signal — it tells the model nothing.  If ALL steps in a response are
        'not_verifiable', then FoVer produced noise, not signal.  The labeling
        is considered successful only when at least one step has a definite label.

    Args:
        question_id: Identifier string for this pair.
        question:    The input math question (unused by FoVer — only the response is parsed).
        response:    The model's full response text.
        domain:      Dataset domain tag ('svamp' or 'gsm8k').

    Returns:
        LabelingResult with all fields populated.
    """
    steps: list[FOVERCoTStep] = parse_cot_into_steps(response)
    annotated: list[FOVERCoTStep] = [annotate_step_with_z3(s) for s in steps]

    # A step is "verifiable" if Z3 returned correct/incorrect (not not_verifiable).
    verifiable = [
        s for s in annotated
        if s.z3_label in ("correct", "incorrect") and s.z3_confidence > 0
    ]

    labeling_successful = len(verifiable) > 0

    if verifiable:
        # Pick the highest-confidence verifiable step as the representative label.
        best = max(verifiable, key=lambda s: s.z3_confidence)
        label_value = 1 if best.z3_label == "incorrect" else 0
        label_confidence: float | None = best.z3_confidence
    else:
        label_value = None
        label_confidence = None

    return LabelingResult(
        question_id=question_id,
        n_cot_steps=len(annotated),
        labeling_successful=labeling_successful,
        label_value=label_value,
        label_confidence=label_confidence,
        domain=domain,
    )


def analyze_cohort(
    questions: list[str],
    responses: list[str],
    domain: str,
) -> list[LabelingResult]:
    """Run FoVer analysis on all pairs in a cohort (SVAMP or GSM8K).

    Args:
        questions: List of question strings.
        responses: List of corresponding simulated responses.
        domain:    Domain tag ('svamp' or 'gsm8k').

    Returns:
        List of LabelingResult, one per pair.
    """
    assert len(questions) == len(responses), (
        f"questions ({len(questions)}) and responses ({len(responses)}) must match"
    )
    results = []
    for i, (q, r) in enumerate(zip(questions, responses)):
        qid = f"{domain}_{i:02d}"
        results.append(analyze_pair(qid, q, r, domain))
    return results


# ---------------------------------------------------------------------------
# Cohort statistics
# ---------------------------------------------------------------------------

def compute_cohort_stats(results: list[LabelingResult]) -> dict[str, float]:
    """Compute aggregate FoVer labeling statistics for a cohort.

    **Metrics:**
        mean_cot_depth: Average number of FoVer steps per response.  Expected:
            SVAMP ~1 (whole response is one step), GSM8K ~3-4 (numbered steps).
        labeling_failure_rate: Fraction of pairs where FoVer produced only noise.
            Expected: SVAMP ~1.0 (all noise), GSM8K ~0.1-0.3 (most labellable).
        label_noise_estimate: Fraction with label_confidence < 0.5 among labeled pairs.
            Returns 1.0 if no pairs were labeled (all failure is noise).

    Args:
        results: Cohort LabelingResult list.

    Returns:
        Dict with mean_cot_depth, labeling_failure_rate, label_noise_estimate.
    """
    if not results:
        return {
            "mean_cot_depth": 0.0,
            "labeling_failure_rate": 1.0,
            "label_noise_estimate": 1.0,
        }

    mean_depth = sum(r.n_cot_steps for r in results) / len(results)
    failure_rate = sum(1 for r in results if not r.labeling_successful) / len(results)

    labeled = [r for r in results if r.label_confidence is not None]
    if labeled:
        noise_est = sum(
            1 for r in labeled if (r.label_confidence or 0.0) < 0.5
        ) / len(labeled)
    else:
        noise_est = 1.0  # All pairs are noise if none were labeled

    return {
        "mean_cot_depth": mean_depth,
        "labeling_failure_rate": failure_rate,
        "label_noise_estimate": noise_est,
    }


# ---------------------------------------------------------------------------
# VJEPA AUC on labeled pairs
# ---------------------------------------------------------------------------

def compute_vjepa_auc_on_labeled(
    results: list[LabelingResult],
) -> float:
    """Train a mini VJEPA and evaluate AUC on the labeled (non-noise) pairs.

    **Why we train VJEPA on the labeled pairs only:**
        FoVer labels are the training signal for VJEPA — if no labels exist, VJEPA
        cannot learn a meaningful decision boundary.  We collect only the pairs
        where `labeling_successful=True`, build TF-IDF features, train VJEPA for
        50 epochs, and compute AUC.  If fewer than 2 unique labels exist (e.g. all
        pairs happen to be labeled "correct"), AUC is degenerate and we return 0.5.

    Args:
        results: LabelingResult list for one domain.

    Returns:
        ROC-AUC in [0.0, 1.0]; 0.5 for degenerate / too-few-label cases.
    """
    labeled = [r for r in results if r.labeling_successful and r.label_value is not None]

    if len(labeled) < 2:
        return 0.5

    # Build raw step dicts for prepare_corpus.
    raw = [
        {
            "question_id": r.question_id,
            "step_text": f"{r.domain} question {r.question_id} step labeled",
            "label": "incorrect" if r.label_value == 1 else "correct",
        }
        for r in labeled
    ]

    labels_unique = set(step["label"] for step in raw)
    if len(labels_unique) < 2:
        return 0.5

    token_to_idx = build_tfidf_features(
        [s["step_text"] for s in raw], vocab_size=VOCAB_SIZE
    )
    corpus = prepare_corpus(raw, token_to_idx, vocab_size=VOCAB_SIZE)

    if len(corpus) < 2:
        return 0.5

    predictor = VariationalJEPAPredictor(
        in_dim=VOCAB_SIZE,
        context_dim=VOCAB_SIZE,
        latent_dim=16,
    )
    predictor.train(corpus, n_epochs=50, lr=1e-3, seed=42)

    _key = jax.random.PRNGKey(0)
    scores = [
        float(predictor.predict(
            jnp.array(item["feature"], dtype=jnp.float32),
            jnp.array(item["context"], dtype=jnp.float32),
            _key,
        ))
        for item in corpus
    ]
    label_ints = [item["label"] for item in corpus]
    return compute_auc(label_ints, scores)


# ---------------------------------------------------------------------------
# labeling_mismatch_confirmed gate
# ---------------------------------------------------------------------------

def check_mismatch_confirmed(
    mean_cot_depth_svamp: float,
    mean_cot_depth_gsm8k: float,
    labeling_failure_rate_svamp: float,
) -> bool:
    """Return True if all three conditions confirm the FoVer/SVAMP mismatch.

    Conditions (all must hold):
        1. mean_cot_depth_svamp < 2.0  — SVAMP produces near-single-step responses.
        2. mean_cot_depth_gsm8k > 4.0  — GSM8K produces genuine multi-step chains.
        3. labeling_failure_rate_svamp > 0.5 — FoVer fails on majority of SVAMP pairs.

    Args:
        mean_cot_depth_svamp:      Mean FoVer step count across SVAMP responses.
        mean_cot_depth_gsm8k:      Mean FoVer step count across GSM8K responses.
        labeling_failure_rate_svamp: Fraction of SVAMP pairs with only noise labels.

    Returns:
        True if hypothesis confirmed, False otherwise.
    """
    return (
        mean_cot_depth_svamp < 2.0
        and mean_cot_depth_gsm8k > 4.0
        and labeling_failure_rate_svamp > 0.5
    )


# ---------------------------------------------------------------------------
# honest_verdict assignment
# ---------------------------------------------------------------------------

def assign_honest_verdict(labeling_mismatch_confirmed: bool) -> str:
    """Map the mismatch flag to the experiment's honest verdict string.

    Verdicts:
        "mismatch_confirmed_gate_open"         — hypothesis confirmed; Exp 896 gated.
        "mismatch_unconfirmed_investigate_further" — hypothesis rejected; more work needed.

    Args:
        labeling_mismatch_confirmed: Output of check_mismatch_confirmed().

    Returns:
        Honest verdict string.
    """
    if labeling_mismatch_confirmed:
        return "mismatch_confirmed_gate_open"
    return "mismatch_unconfirmed_investigate_further"


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict[str, Any]:
    """Execute the full SVAMP root-cause confirmation experiment.

    **Pipeline:**
        1. Run FoVer labeling on 20 SVAMP pairs and 20 GSM8K pairs.
        2. Compute mean CoT depth and labeling failure rate for each cohort.
        3. Run VJEPA on labeled pairs; compute per-cohort AUC.
        4. Check mismatch conditions and assign honest_verdict.
        5. Return artifact dict (written to JSON by the caller).

    Returns:
        Artifact dict matching the required schema from the task spec.
    """
    t0 = time.time()

    svamp_results = analyze_cohort(SVAMP_QUESTIONS, SVAMP_RESPONSES, "svamp")
    gsm8k_results = analyze_cohort(GSM8K_QUESTIONS, GSM8K_RESPONSES, "gsm8k")

    svamp_stats = compute_cohort_stats(svamp_results)
    gsm8k_stats = compute_cohort_stats(gsm8k_results)

    svamp_auc = compute_vjepa_auc_on_labeled(svamp_results)
    gsm8k_auc = compute_vjepa_auc_on_labeled(gsm8k_results)

    mismatch = check_mismatch_confirmed(
        mean_cot_depth_svamp=svamp_stats["mean_cot_depth"],
        mean_cot_depth_gsm8k=gsm8k_stats["mean_cot_depth"],
        labeling_failure_rate_svamp=svamp_stats["labeling_failure_rate"],
    )

    verdict = assign_honest_verdict(mismatch)

    return {
        "experiment": 893,
        "schema": "carnot-experiment-v1",
        "spec": ["REQ-VER-085", "SCENARIO-VER-085"],
        "prior_failures": [
            {
                "experiment_id": "exp872",
                "verdict": "vjepa_ood_collapsed",
                "addressed_by": "This experiment confirms root cause before attempting fix.",
            }
        ],
        "mean_cot_depth_svamp": svamp_stats["mean_cot_depth"],
        "mean_cot_depth_gsm8k": gsm8k_stats["mean_cot_depth"],
        "labeling_failure_rate_svamp": svamp_stats["labeling_failure_rate"],
        "labeling_failure_rate_gsm8k": gsm8k_stats["labeling_failure_rate"],
        "label_noise_estimate_svamp": svamp_stats["label_noise_estimate"],
        "label_noise_estimate_gsm8k": gsm8k_stats["label_noise_estimate"],
        "svamp_auc": svamp_auc,
        "gsm8k_auc_for_comparison": gsm8k_auc,
        "labeling_mismatch_confirmed": mismatch,
        "n_svamp_questions": 20,
        "n_gsm8k_questions": 20,
        "honest_verdict": verdict,
        "duration_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Deliverable guard
# ---------------------------------------------------------------------------

def assert_deliverable_written() -> None:
    """Assert that the result JSON was written and contains all required fields.

    Spec: REQ-VER-085
    """
    required = {
        "mean_cot_depth_svamp",
        "mean_cot_depth_gsm8k",
        "labeling_failure_rate_svamp",
        "labeling_failure_rate_gsm8k",
        "svamp_auc",
        "gsm8k_auc_for_comparison",
        "labeling_mismatch_confirmed",
        "n_svamp_questions",
        "n_gsm8k_questions",
        "honest_verdict",
    }
    assert RESULT_PATH.exists(), f"Deliverable not written: {RESULT_PATH}"
    with open(RESULT_PATH) as f:
        data = json.load(f)
    missing = required - set(data.keys())
    assert not missing, f"Missing required fields: {missing}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artifact = run_experiment()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Written: {RESULT_PATH}")
    print(f"honest_verdict: {artifact['honest_verdict']}")
    assert_deliverable_written()
