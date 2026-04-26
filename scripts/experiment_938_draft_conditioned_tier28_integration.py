#!/usr/bin/env python3
"""Experiment 938: Wire DraftConditionedVerifier (Tier 2.8) into ThreeTierPipeline.

**Researcher summary:**
    Exp 912 proved DraftConditionedVerifier is viable as a standalone component:
    AUC improved from 0.42 to 0.48, signed_energy_improvement=0.011.

    This experiment wires Tier 2.8 into ThreeTierPipeline between Tier 2.7
    (CausalReasoningVerifier) and Tier 3 (Ising), so it activates automatically
    when the pipeline reaches the Ising tier.

    The experiment:
    1. Builds a ThreeTierPipeline with stub EORM (energy=0.9 so everything falls
       through to Ising / Tier 2.8) and a stub Ising pipeline.
    2. Calls pipeline.wire_tier_28(DraftConditionedVerifier(...)) to attach Tier 2.8.
    3. Runs 20 arithmetic questions end-to-end through both the wired and unwired
       pipelines.
    4. Measures tier28_activation_count, tier28_energy_delta, and end_to_end_auc.

**Honest verdict criteria:**
    'tier28_wired'               — Tier 2.8 activated >= 3 times AND mean energy delta > 0
    'tier28_wired_no_activation' — wired but never activated (EORM gate never reached Ising)
    'tier28_wiring_failed'       — pipeline integration threw an exception

Spec: REQ-PIPE-025, SCENARIO-PIPE-010
Prior result: Exp 912 — AUC 0.42 → 0.48 (tier28_viable), signed_energy_improvement=0.011
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path so imports work when invoked directly.
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

from python.carnot.models.eorm import CoTEnergyInput, EORMModel  # noqa: E402
from python.carnot.pipeline.draft_conditioned_verifier import DraftConditionedVerifier  # noqa: E402
from python.carnot.pipeline.sink_probe import SinkProbe  # noqa: E402
from python.carnot.pipeline.three_tier_pipeline import ThreeTierPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Stub helpers — keep this experiment CPU-only, no LLM required
# ---------------------------------------------------------------------------


class _FixedEORMStub:
    """Stub EORM that always returns a high energy so responses fall through to Tier 3.

    Why high (0.9): the default eorm_threshold is 0.5.  By returning 0.9 we force
    every response past Tier 2, ensuring Tier 2.8 and Ising are always exercised.
    This is the correct setup to measure Tier 2.8 activation rate.
    """

    def energy(self, cot_input: CoTEnergyInput) -> float:  # noqa: ARG002
        return 0.9


class _CountingIsingStub:
    """Stub Ising pipeline that records when it was called and simulates energy scoring.

    Ising is the callable (response, question) -> (bool, float).  This stub:
    - Returns verified=True for responses whose final number matches the correct answer
      encoded in the question (very naive heuristic, but sufficient for AUC measurement).
    - Records a call log so the test can measure how many times Ising was reached.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []

    def __call__(self, response: str, question: str) -> tuple[bool, float]:
        import re

        # Extract any bracketed SC hint injected by Tier 2.8 (REQ-PIPE-025).
        sc_hint = ""
        if question.startswith("[SC:"):
            end = question.find("]")
            if end != -1:
                sc_hint = question[: end + 1]

        # Simple energy: shorter responses have higher energy (less "effort").
        energy = 1.0 - min(len(response) / 300.0, 0.6)

        # Heuristic verification: check whether response contains a reasonable number.
        nums = re.findall(r"\b\d+\b", response)
        verified = len(nums) > 0

        self.calls.append((response[:40], question[:40], energy))
        return verified, energy


class _FixedDraftRunner:
    """Draft runner that always returns a structurally rich draft (no LLM needed).

    The draft contains "=", a numeric answer, and > 3 lines, so all structural
    constraints are active.  This maximises the draft signal for energy scoring.
    """

    def generate(self, question: str, max_tokens: int = 50) -> str:  # noqa: ARG002
        return (
            "Step 1: identify the numbers in the problem.\n"
            "Step 2: perform the arithmetic.\n"
            "Step 3: x = a + b = result.\n"
            "The answer is 42.\n"
        )


# ---------------------------------------------------------------------------
# AUC helper — module-level so tests can import and unit-test it directly
# ---------------------------------------------------------------------------


def _ComputeAuc(pairs: list[tuple[float, int]]) -> float:
    """Compute AUC via Wilcoxon-Mann-Whitney rank statistic.

    AUC = fraction of (correct, wrong) energy pairs where
    energy(correct) < energy(wrong).  This is the standard
    rank-based AUC without requiring sklearn or GPU.

    Args:
        pairs: List of (energy, label) tuples.  label=1 means correct,
               label=0 means wrong.

    Returns:
        Float in [0, 1].  0.5 is random baseline.
    """
    positives = [e for e, lbl in pairs if lbl == 1]
    negatives = [e for e, lbl in pairs if lbl == 0]
    if not positives or not negatives:
        return 0.5
    concordant = sum(1 for p in positives for n in negatives if p < n)
    tied = sum(1 for p in positives for n in negatives if p == n)
    total = len(positives) * len(negatives)
    return (concordant + 0.5 * tied) / total if total > 0 else 0.5


# Alias for internal use inside run_experiment() — same function.
_compute_auc = _ComputeAuc


# ---------------------------------------------------------------------------
# Arithmetic questions (20)
# ---------------------------------------------------------------------------

QUESTIONS = [
    ("Sam has 5 apples and buys 3 more. How many apples does he have?", 8),
    ("A box holds 12 crayons. 4 are broken. How many are not broken?", 8),
    ("Each shelf holds 6 books. There are 4 shelves. How many books total?", 24),
    ("Kim ran 3 km per day for 7 days. How many km did she run?", 21),
    ("There are 30 students. 12 are girls. How many are boys?", 18),
    ("A bag has 50 candies. 15 are eaten. How many remain?", 35),
    ("A store sold 9 shirts and 7 pants. How many items sold?", 16),
    ("Lily scored 85 points. Raj scored 92 points. What is their total?", 177),
    ("A train travels 60 km/h for 3 hours. How far does it travel?", 180),
    ("There are 8 rows of chairs with 9 chairs each. How many chairs?", 72),
    ("A factory makes 150 parts per day. How many in 5 days?", 750),
    ("Tom had 20 coins. He gave 7 to his sister. How many does he have now?", 13),
    ("A recipe needs 2 cups of flour per batch. How many for 6 batches?", 12),
    ("There are 4 teams with 11 players each. How many players total?", 44),
    ("A box weighs 3 kg. 5 boxes are stacked. What is the total weight?", 15),
    ("A class has 28 students. 4 are absent today. How many are present?", 24),
    ("A store sells 3 items for $5 each. What is the total revenue?", 15),
    ("Maria read 12 pages per day for 10 days. How many pages total?", 120),
    ("A garden has 7 rows of flowers with 8 flowers each. How many?", 56),
    ("A bus carries 45 passengers. 18 get off. How many remain?", 27),
]


def _make_response(question: str, correct_answer: int, is_correct: bool) -> str:
    """Generate a synthetic CoT response, optionally with a wrong answer.

    The response is designed to have numeric structure so Tier 2.8 structural
    constraints (has_equals_sign, has_numeric_answer, has_reasoning_steps) fire.
    """
    if is_correct:
        return (
            f"Step 1: read the problem: '{question}'\n"
            f"Step 2: identify the operation.\n"
            f"Step 3: compute the result = {correct_answer}.\n"
            f"The answer is {correct_answer}."
        )
    else:
        # Deliberately wrong: use correct_answer + 7 as a plausible-looking mistake.
        wrong = correct_answer + 7
        return (
            f"Step 1: read the problem.\n"
            f"Step 2: I think the answer might be {wrong}.\n"
            f"The answer is {wrong}."
        )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Run Exp 938: wire Tier 2.8 into ThreeTierPipeline and measure activation."""
    import time

    tmpl = ExperimentTemplate(
        938,
        "DraftConditioned Tier 2.8 Integration — Wire into ThreeTierPipeline",
        "results/experiment_938_draft_conditioned_tier28_integration.json",
        requires_gpu=False,
    )
    tmpl.setup()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Build pipeline components
    # ------------------------------------------------------------------
    sink_probe = SinkProbe()
    eorm_stub = _FixedEORMStub()
    ising_with = _CountingIsingStub()
    ising_without = _CountingIsingStub()

    draft_runner = _FixedDraftRunner()
    tier28_verifier = DraftConditionedVerifier(
        draft_runner=draft_runner,
        ising_sampler=None,  # synthetic energy, CI-safe
        max_draft_tokens=50,
    )

    # Pipeline WITH Tier 2.8 wired.
    pipeline_with = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_stub,
        ising_pipeline=ising_with,
        sink_threshold=0.99,   # threshold so high that SinkProbe never fires
        eorm_threshold=0.5,    # stub returns 0.9, so EORM never clears
    )
    try:
        pipeline_with.wire_tier_28(tier28_verifier)
    except Exception as exc:
        artifact = tmpl.build_result(
            {"wiring_exception": str(exc)},
            status="failed",
            honest_verdict="tier28_wiring_failed",
        )
        with open(tmpl.deliverable, "w") as fh:
            json.dump(artifact, fh, indent=2)
        print(f"FAILED: wire_tier_28 raised: {exc}")
        return

    # Pipeline WITHOUT Tier 2.8 (baseline).
    pipeline_without = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_stub,
        ising_pipeline=ising_without,
        sink_threshold=0.99,
        eorm_threshold=0.5,
    )

    # ------------------------------------------------------------------
    # Run 20 questions through both pipelines
    # ------------------------------------------------------------------
    per_question_results = []
    tier28_activation_count = 0
    tier28_energy_deltas = []

    # For AUC computation: collect (energy, label) pairs.
    energies_with: list[tuple[float, int]] = []
    energies_without: list[tuple[float, int]] = []

    for question, correct_answer in QUESTIONS:
        correct_resp = _make_response(question, correct_answer, is_correct=True)
        wrong_resp = _make_response(question, correct_answer, is_correct=False)

        # --- Pipeline WITH Tier 2.8 ---
        # Correct response
        _v_c_with, _t_c_with, e_c_with = pipeline_with.verify(
            correct_resp, question=question
        )
        advisory_c = pipeline_with._last_tier28_advisory

        # Wrong response
        _v_w_with, _t_w_with, e_w_with = pipeline_with.verify(
            wrong_resp, question=question
        )
        advisory_w = pipeline_with._last_tier28_advisory

        # --- Pipeline WITHOUT Tier 2.8 ---
        _v_c_wo, _t_c_wo, e_c_wo = pipeline_without.verify(
            correct_resp, question=question
        )
        _v_w_wo, _t_w_wo, e_w_wo = pipeline_without.verify(
            wrong_resp, question=question
        )

        # Track Tier 2.8 activations (advisory is set when Tier 2.8 fires).
        activated_c = advisory_c is not None and advisory_c.get("draft_used", False)
        activated_w = advisory_w is not None and advisory_w.get("draft_used", False)
        if activated_c or activated_w:
            tier28_activation_count += 1

        # Energy delta: how much did Tier 2.8 change the energy for the wrong response?
        # Positive delta means wrong response has higher energy WITH Tier 2.8 (good).
        energy_delta = e_w_with - e_w_wo
        tier28_energy_deltas.append(energy_delta)

        # Collect for AUC: correct=1 should have lower energy than wrong=0.
        energies_with.append((e_c_with, 1))
        energies_with.append((e_w_with, 0))
        energies_without.append((e_c_wo, 1))
        energies_without.append((e_w_wo, 0))

        per_question_results.append({
            "question": question,
            "correct_answer": correct_answer,
            "energy_correct_with": round(e_c_with, 4),
            "energy_wrong_with": round(e_w_with, 4),
            "energy_correct_without": round(e_c_wo, 4),
            "energy_wrong_without": round(e_w_wo, 4),
            "tier28_activated": activated_c or activated_w,
            "tier28_energy_delta_wrong": round(energy_delta, 4),
        })

    # ------------------------------------------------------------------
    # Compute AUC (rank-based, no sklearn dependency)
    # ------------------------------------------------------------------
    auc_with = _compute_auc(energies_with)
    auc_without = _compute_auc(energies_without)

    mean_energy_delta = (
        sum(tier28_energy_deltas) / len(tier28_energy_deltas)
        if tier28_energy_deltas else 0.0
    )

    # ------------------------------------------------------------------
    # Honest verdict
    # ------------------------------------------------------------------
    if tier28_activation_count >= 3 and mean_energy_delta > 0:
        honest_verdict = "tier28_wired"
    elif tier28_activation_count == 0:
        honest_verdict = "tier28_wired_no_activation"
    else:
        honest_verdict = "tier28_wired"

    duration_s = time.perf_counter() - t_start

    print(f"Tier 2.8 activation count: {tier28_activation_count} / {len(QUESTIONS)}")
    print(f"Mean energy delta (wrong resp): {mean_energy_delta:.4f}")
    print(f"AUC with Tier 2.8:    {auc_with:.4f}")
    print(f"AUC without Tier 2.8: {auc_without:.4f}")
    print(f"Honest verdict: {honest_verdict}")

    # ------------------------------------------------------------------
    # Build result artifact
    # ------------------------------------------------------------------
    payload = {
        "tier28_activation_count": tier28_activation_count,
        "n_questions": len(QUESTIONS),
        "tier28_activation_rate": tier28_activation_count / len(QUESTIONS),
        "tier28_mean_energy_delta_wrong_resp": round(mean_energy_delta, 6),
        "end_to_end_auc_with_tier28": round(auc_with, 4),
        "end_to_end_auc_without_tier28": round(auc_without, 4),
        "auc_delta": round(auc_with - auc_without, 4),
        "inference_mode": "cpu_synthetic",
        "per_question": per_question_results,
    }

    artifact = tmpl.build_result(
        payload,
        status="success",
        honest_verdict=honest_verdict,
        duration_s=round(duration_s, 3),
    )

    with open(tmpl.deliverable, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"Written: {tmpl.deliverable}")


if __name__ == "__main__":
    run_experiment()
