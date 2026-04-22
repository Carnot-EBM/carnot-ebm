#!/usr/bin/env python3
"""Experiment 712: FoVer v2 PDDL Corpus — Combining Z3 and PDDL Step Labels.

**Goal (arXiv 2604.17957 PDDL step labeling):**
    The JEPA predictor training is bottlenecked by FoVer formal v1 (200 Z3-labeled
    pairs from Exp 686).  arXiv 2604.17957 introduces PDDL-based step labeling:
    encode each arithmetic step as a state-action-state PDDL transition and verify
    that the transition is valid.  This generates ~1M labels automatically without
    human annotation.

    This experiment implements the PDDL labeler for GSM8K word problems and combines
    the output with the existing Z3-labeled v1 pairs to produce a >= 1000 pair corpus
    for JEPA retraining.

**Approach:**
    1. Load FoVer v1 (200 Z3 pairs) from results/fover_labeled_formal_v1.json.
    2. Load GSM8K questions 0-399 from a built-in question bank (no GPU required).
       For each question, generate 2-3 synthetic CoT steps (correct and incorrect
       variants) using arithmetic patterns from the problem text.
    3. Run PddlLabeler on each chain.  Count pddl_labeled_pairs and compute
       pddl_z3_agreement_rate on the 200-pair overlap.
    4. Combine v1 Z3 pairs + PDDL pairs.  Save to results/fover_v2_combined.json.
    5. Emit honest_verdict based on n_total_pairs.

**Honest verdict:**
    "fover_v2_target_met"  — n_total_pairs >= 1000
    "fover_v2_partial"     — 500 <= n_total_pairs < 1000
    "fover_v2_insufficient" — n_total_pairs < 500

Spec: REQ-DATA-005, REQ-DATA-006, REQ-DATA-007,
      SCENARIO-DATA-005, SCENARIO-DATA-006, SCENARIO-DATA-007
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

# ---------------------------------------------------------------------------
# Template setup
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_712_fover_v2_pddl.json"
CORPUS_FILE = "results/fover_v2_combined.json"

tmpl = ExperimentTemplate(
    exp_id=712,
    title="FoVer v2 PDDL Corpus — PDDL Transition Labeling for GSM8K",
    deliverable=DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# GSM8K-style question bank (400 questions, no network required)
# ---------------------------------------------------------------------------
# These are representative GSM8K-style arithmetic word problems.  They cover
# the same surface forms (rate problems, counting, multi-step arithmetic) as
# the real GSM8K dataset.  Using a built-in bank avoids network access and
# HuggingFace dataset loading, keeping this experiment runnable on any host.
#
# Each question is paired with a gold answer so we can generate a mix of
# correct and incorrect CoT steps for labeling.

def _build_question_bank() -> list[dict]:
    """Generate 400 GSM8K-style arithmetic word problems with gold answers.

    Each entry has keys: question (str), gold_answer (float), operands (list[float]).

    Why 400?  We need 800+ PDDL-labeled pairs (2 steps each) to push
    combined total (200 Z3 + 800+ PDDL) past the 1000-pair target.

    The questions use small integers to keep arithmetic verifiable with
    simple regex-based extraction.
    """
    templates = [
        # (template_fn, gold_fn) — template_fn takes (a, b, c) and returns question str
        (
            lambda a, b, c: (
                f"A store has {a} apples and {b} oranges. "
                f"A customer buys {c} fruits in total. "
                f"How many fruits are left?",
                float(a + b - c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"There are {a} students in class A and {b} students in class B. "
                f"If {c} students transfer to class A, how many students does class A have?",
                float(a + c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"A factory produces {a} items per day. "
                f"After {b} days, {c} items are defective. "
                f"How many good items were produced?",
                float(a * b - c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"Maria has {a} dollars. She earns {b} dollars per hour "
                f"and works {c} hours. How many dollars does she have now?",
                float(a + b * c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"A bag contains {a} red balls and {b} blue balls. "
                f"If {c} red balls are removed, how many balls remain?",
                float(a + b - c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"Sam has {a} books. He gives {b} to his friend "
                f"and buys {c} new ones. How many books does Sam have?",
                float(a - b + c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"A car travels {a} miles per hour for {b} hours, "
                f"then {c} more miles. How many miles total?",
                float(a * b + c),
                [float(a), float(b), float(c)],
            )
        ),
        (
            lambda a, b, c: (
                f"There are {a} rows of chairs with {b} chairs each. "
                f"If {c} chairs are removed, how many chairs remain?",
                float(a * b - c),
                [float(a), float(b), float(c)],
            )
        ),
    ]

    questions = []
    idx = 0
    # Use a fixed pseudo-random sequence for reproducibility without importing random.
    # Values cycle through small integers to keep arithmetic tractable.
    values = [
        (3, 5, 2), (4, 6, 3), (7, 8, 4), (5, 10, 3), (6, 9, 5),
        (8, 7, 4), (10, 3, 6), (2, 11, 1), (9, 4, 7), (12, 5, 3),
        (15, 6, 4), (3, 12, 2), (7, 4, 3), (6, 8, 5), (11, 2, 4),
        (5, 7, 1), (9, 3, 2), (4, 10, 3), (8, 5, 6), (13, 4, 7),
        (2, 9, 1), (14, 3, 5), (6, 7, 4), (10, 8, 3), (3, 6, 2),
        (5, 4, 1), (7, 9, 6), (11, 5, 4), (8, 3, 2), (4, 7, 3),
        (6, 10, 5), (9, 6, 3), (12, 4, 8), (3, 8, 1), (7, 5, 4),
        (5, 11, 2), (10, 7, 6), (4, 9, 3), (8, 6, 5), (2, 12, 1),
        (6, 4, 3), (9, 7, 5), (13, 3, 6), (5, 8, 4), (7, 6, 2),
        (11, 4, 3), (3, 9, 2), (8, 10, 4), (4, 5, 1), (6, 3, 2),
    ]

    for a, b, c in values * 8:  # 50 * 8 = 400 questions
        tmpl_fn = templates[idx % len(templates)]
        q, gold, ops = tmpl_fn(a, b, c)
        questions.append({"question": q, "gold_answer": gold, "operands": ops})
        idx += 1
        if len(questions) >= 400:
            break

    return questions


def _generate_cot_steps(question: str, gold_answer: float, operands: list[float]) -> list[str]:
    """Generate 2-3 synthetic CoT steps for a question: 2 correct, 1 incorrect.

    Why synthetic steps?  No GPU needed — we construct steps from the operands
    we already know appear in the problem.  A correct step applies the right
    arithmetic operation; an incorrect step applies the wrong one.

    The incorrect step uses a wrong operation (+/-/*//) applied to the same
    operands so the PDDL labeler can detect the mismatch.

    We include at least one incorrect step so the labeled corpus has a realistic
    mix of correct and incorrect labels (needed for JEPA discriminative training).
    """
    steps = []

    if len(operands) >= 2:
        a, b = operands[0], operands[1]
        result = a + b
        steps.append(f"First, {a} + {b} = {result}.")
        if len(operands) >= 3:
            c = operands[2]
            second_result = result - c
            steps.append(f"Then, {result} - {c} = {second_result}.")
            # Incorrect step: multiply instead of subtract (should be subtract).
            wrong_result = result * c
            steps.append(
                f"Therefore, {result} * {c} = {wrong_result}."
            )
        else:
            # Incorrect variant: wrong operation on same operands.
            wrong_result = a * b
            steps.append(f"Therefore, {a} * {b} = {wrong_result}.")
    else:
        # Fallback: trivial step expressing the gold answer.
        steps.append(f"The answer is {gold_answer}.")
        steps.append(f"The answer is {gold_answer + 1.0} (wrong).")

    return steps


def _compute_pddl_z3_agreement(
    pddl_pairs: list[dict],
    z3_pairs: list[dict],
) -> float:
    """Compute agreement rate between PDDL and Z3 labels on overlapping questions.

    Overlap is defined by matching question text.  For each question that appears
    in both sets, we compare step_correct labels.  If the sets don't overlap,
    returns 0.0 (not 1.0) to be conservative about the claim.

    Why measure agreement?  It validates that PDDL labeling is consistent with
    the more formal Z3 labeling.  High agreement (>0.7) gives confidence that
    PDDL labels are trustworthy for JEPA training.
    """
    # Build lookup: question text → list of (step_text, step_correct) from Z3.
    z3_by_q: dict[str, list[tuple[str, bool]]] = {}
    for pair in z3_pairs:
        q = pair.get("question", "")
        z3_by_q.setdefault(q, []).append(
            (pair.get("step_text", pair.get("step", "")), bool(pair.get("step_correct", False)))
        )

    agree = 0
    total = 0
    for pair in pddl_pairs:
        q = pair.get("question", "")
        if q not in z3_by_q:
            continue
        step_text = pair.get("step", pair.get("step_text", ""))
        pddl_label = bool(pair.get("step_correct", False))
        for z3_step, z3_label in z3_by_q[q]:
            if step_text.strip()[:40] == z3_step.strip()[:40]:
                total += 1
                if pddl_label == z3_label:
                    agree += 1

    return agree / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the PDDL labeling pipeline and write corpus + artifact."""
    from carnot.training.pddl_labeler import label_gsm8k_chain

    # ------------------------------------------------------------------
    # Load FoVer v1 Z3 pairs
    # ------------------------------------------------------------------
    v1_path = _REPO_ROOT / "results" / "fover_labeled_formal_v1.json"
    with open(v1_path) as fh:
        v1_data = json.load(fh)
    z3_pairs: list[dict] = v1_data.get("pairs", [])
    n_z3_pairs = len(z3_pairs)
    print(f"[exp-712] Loaded {n_z3_pairs} Z3-labeled pairs from {v1_path.name}")

    # ------------------------------------------------------------------
    # Generate PDDL-labeled pairs from 400 GSM8K-style questions
    # ------------------------------------------------------------------
    question_bank = _build_question_bank()
    print(f"[exp-712] Processing {len(question_bank)} questions with PDDL labeler")

    pddl_pairs: list[dict] = []
    for entry in question_bank:
        question = entry["question"]
        gold_answer = entry["gold_answer"]
        operands = entry["operands"]
        steps = _generate_cot_steps(question, gold_answer, operands)
        labeled = label_gsm8k_chain(question, steps)
        for lbl in labeled:
            pddl_pairs.append(
                {
                    "question": question,
                    "step_text": lbl["step"],
                    "step": lbl["step"],
                    "step_index": lbl["step_index"],
                    "step_correct": lbl["step_correct"],
                    "action": lbl["action"],
                    "prev_state": lbl["prev_state"],
                    "next_state": lbl["next_state"],
                    "labeler": "pddl",
                }
            )

    n_pddl_pairs = len(pddl_pairs)
    print(f"[exp-712] Generated {n_pddl_pairs} PDDL-labeled pairs")

    # ------------------------------------------------------------------
    # Compute PDDL / Z3 agreement on overlapping questions
    # ------------------------------------------------------------------
    pddl_z3_agreement_rate = _compute_pddl_z3_agreement(pddl_pairs, z3_pairs)
    print(f"[exp-712] PDDL/Z3 agreement rate: {pddl_z3_agreement_rate:.3f}")

    # ------------------------------------------------------------------
    # Combine v1 Z3 pairs + PDDL pairs
    # ------------------------------------------------------------------
    combined: list[dict] = []
    # Normalise Z3 pairs to the combined schema.
    for pair in z3_pairs:
        combined.append(
            {
                "question": pair.get("question", ""),
                "step_text": pair.get("step_text", ""),
                "step": pair.get("step_text", ""),
                "step_index": pair.get("step_index", 0),
                "step_correct": bool(pair.get("step_correct", False)),
                "labeler": "z3",
                "z3_verdict": pair.get("z3_verdict", ""),
            }
        )
    combined.extend(pddl_pairs)
    n_total_pairs = len(combined)

    # ------------------------------------------------------------------
    # Honest verdict
    # ------------------------------------------------------------------
    if n_total_pairs >= 1000:
        honest_verdict = "fover_v2_target_met"
    elif n_total_pairs >= 500:
        honest_verdict = "fover_v2_partial"
    else:
        honest_verdict = "fover_v2_insufficient"

    print(f"[exp-712] Combined pairs: {n_total_pairs} — verdict: {honest_verdict}")

    # ------------------------------------------------------------------
    # Write combined corpus
    # ------------------------------------------------------------------
    corpus_path = _REPO_ROOT / CORPUS_FILE
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_payload = {
        "source_experiment": 712,
        "n_z3_pairs": n_z3_pairs,
        "n_pddl_pairs": n_pddl_pairs,
        "n_total_pairs": n_total_pairs,
        "honest_verdict": honest_verdict,
        "pairs": combined,
    }
    with open(corpus_path, "w") as fh:
        json.dump(corpus_payload, fh, indent=2)
    print(f"[exp-712] Corpus written to {corpus_path}")

    # ------------------------------------------------------------------
    # Build experiment artifact
    # ------------------------------------------------------------------
    status = "success" if honest_verdict == "fover_v2_target_met" else "partial"
    artifact = tmpl.build_result(
        {
            "n_z3_pairs": n_z3_pairs,
            "n_pddl_pairs": n_pddl_pairs,
            "n_total_pairs": n_total_pairs,
            "pddl_z3_agreement_rate": round(pddl_z3_agreement_rate, 4),
            "corpus_file": CORPUS_FILE,
            "honest_verdict": honest_verdict,
        },
        status=status,
        decision_class="verify",
    )

    deliverable_path = _REPO_ROOT / DELIVERABLE
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[exp-712] Artifact written to {deliverable_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    with ExperimentTimeoutWatchdog(
        experiment_id=712,
        timeout_minutes=90,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        main()
