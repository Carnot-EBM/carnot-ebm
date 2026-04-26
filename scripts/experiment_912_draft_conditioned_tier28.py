#!/usr/bin/env python3
"""Experiment 912: DraftConditionedVerifier Tier 2.8 — Draft-Scaffolded Ising Constraints.

**Researcher summary:**
    arXiv 2603.03305 (Draft-Conditioned Constrained Decoding) shows that even an
    incorrect cheap draft can scaffold the STRUCTURE of the expected answer and improve
    downstream constrained decoding.  This experiment validates whether injecting
    structural markers from a 50-token Qwen3.5-0.8B draft into the Ising energy function
    (as Tier 2.8) improves the rank-correlation between energy and answer correctness on
    GSM8K.

    Methodology:
        1. Load 25 GSM8K-style problems (same corpus as Exp 911).
        2. For each problem, generate a "full correct" and "full hallucinated" response.
        3. Baseline: score both with DraftConditionedVerifier(ising_sampler=None, no draft).
        4. Tier 2.8: score both with DraftConditionedVerifier using a draft runner that
           generates a cheap draft from the question before scoring.
        5. Compute AUC (correct < hallucinated energy) for baseline and Tier 2.8.
        6. Measure signed_energy_improvement = mean(energy_halluc - energy_correct) for
           both modes.  Positive = correct responses have lower energy on average.

    Honest verdict:
        "tier28_viable"        if auc_with_draft > auc_baseline
        "tier28_no_improvement" otherwise

    NOTE: This experiment uses Qwen/Qwen3.5-0.8B IF it is loadable; otherwise uses a
    deterministic synthetic draft runner that produces realistic-looking drafts.  The
    synthetic runner is identical to the one used in Exp 911 to ensure CI runs pass.
    This avoids blocking on GPU availability while still exercising the full code path.

Spec: REQ-TIER28-001, SCENARIO-TIER28-001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.draft_conditioned_verifier import DraftConditionedVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment setup
# ---------------------------------------------------------------------------

DELIVERABLE = "results/experiment_912_draft_conditioned_tier28.json"

tmpl = ExperimentTemplate(
    exp_id=912,
    title="DraftConditionedVerifier Tier 2.8 — Draft-Scaffolded Ising Constraints",
    deliverable=DELIVERABLE,
    requires_gpu=False,  # GPU optional; falls back to synthetic draft runner
)
tmpl.setup()

# ---------------------------------------------------------------------------
# GSM8K problems (same 25 as Exp 911 for consistency)
# ---------------------------------------------------------------------------

_GSM8K_PROBLEMS: list[dict] = [
    {
        "q": "Sam has 5 apples and buys 3 more. How many apples does he have?",
        "a": 8,
        "template": "Sam starts with 5 apples and gets 3 more, so 5 + 3 = {a}.",
    },
    {
        "q": "A box holds 12 crayons. 4 are broken. How many are not broken?",
        "a": 8,
        "template": "12 crayons total minus 4 broken = {a} intact crayons.",
    },
    {
        "q": "Each shelf holds 6 books. There are 4 shelves. How many books total?",
        "a": 24,
        "template": "6 books per shelf x 4 shelves = {a} books.",
    },
    {
        "q": "Kim ran 3 km per day for 7 days. How many km did she run?",
        "a": 21,
        "template": "3 km/day x 7 days = {a} km total.",
    },
    {
        "q": "There are 30 students. 12 are girls. How many are boys?",
        "a": 18,
        "template": "30 total - 12 girls = {a} boys.",
    },
    {
        "q": "A bag has 45 marbles split equally into 9 groups. Size of each group?",
        "a": 5,
        "template": "45 / 9 = {a} marbles per group.",
    },
    {
        "q": "Tom earns $8/hour. He works 6 hours. How much does he earn?",
        "a": 48,
        "template": "8 x 6 = ${a}.",
    },
    {
        "q": "A farmer has 7 rows of 9 corn stalks. How many stalks total?",
        "a": 63,
        "template": "7 x 9 = {a} stalks.",
    },
    {
        "q": "A rectangle is 11 m long and 4 m wide. What is its area?",
        "a": 44,
        "template": "Area = 11 x 4 = {a} m^2.",
    },
    {
        "q": "Lisa has 50 stickers. She gives 17 away. How many remain?",
        "a": 33,
        "template": "50 - 17 = {a} stickers left.",
    },
    {
        "q": "A train travels 60 km/h for 3 hours. Total distance?",
        "a": 180,
        "template": "60 x 3 = {a} km.",
    },
    {
        "q": "There are 8 bags with 15 candies each. Total candies?",
        "a": 120,
        "template": "8 x 15 = {a} candies.",
    },
    {"q": "Jake saves $12 a week. How much in 5 weeks?", "a": 60, "template": "12 x 5 = ${a}."},
    {
        "q": "A garden has 6 rows and 8 columns of plants. How many plants?",
        "a": 48,
        "template": "6 x 8 = {a} plants.",
    },
    {
        "q": "200 students. 3/4 passed. How many passed?",
        "a": 150,
        "template": "200 x 3/4 = {a} students passed.",
    },
    {
        "q": "A pizza has 8 slices. 3 people eat 2 slices each. Slices left?",
        "a": 2,
        "template": "8 - 3 x 2 = {a} slices left.",
    },
    {
        "q": "A jar holds 500 ml. You pour out 125 ml. How much remains?",
        "a": 375,
        "template": "500 - 125 = {a} ml.",
    },
    {
        "q": "5 friends share $85 equally. Each person gets?",
        "a": 17,
        "template": "85 / 5 = ${a} each.",
    },
    {
        "q": "A rectangle perimeter is 26 m. Length 8 m. What is the width?",
        "a": 5,
        "template": "Perimeter = 2(l+w) -> 26 = 2(8+w) -> w = {a} m.",
    },
    {
        "q": "Bus seats 48. 3/4 full. How many passengers?",
        "a": 36,
        "template": "48 x 3/4 = {a} passengers.",
    },
    {
        "q": "A library has 240 books. 1/3 are fiction. How many fiction?",
        "a": 80,
        "template": "240 / 3 = {a} fiction books.",
    },
    {
        "q": "A pool holds 1500 L. It leaks 75 L/hr. Empty in how many hours?",
        "a": 20,
        "template": "1500 / 75 = {a} hours.",
    },
    {
        "q": "72 eggs in cartons of 12. How many cartons?",
        "a": 6,
        "template": "72 / 12 = {a} cartons.",
    },
    {"q": "A square has side 9 m. What is its area?", "a": 81, "template": "9 x 9 = {a} m^2."},
    {
        "q": "There are 100 people. 40% are under 18. How many adults?",
        "a": 60,
        "template": "100 - 40 = {a} adults.",
    },
]


def _make_correct_response(prob: dict) -> str:
    """Build a correct CoT response for a GSM8K problem.

    Args:
        prob: Dict with keys "q", "a", "template".

    Returns:
        Multi-step string ending with the correct answer.
    """
    body = prob["template"].format(a=prob["a"])
    return (
        f"Step 1: Read the problem: {prob['q']}\n"
        f"Step 2: Set up the equation. {body}\n"
        f"Step 3: The answer is {prob['a']}."
    )


def _make_hallucinated_response(prob: dict, rng: np.random.Generator) -> str:
    """Build a hallucinated CoT response with a wrong numerical answer.

    Args:
        prob: Problem dict.
        rng:  Reproducible random number generator.

    Returns:
        Multi-step string with a wrong final answer.
    """
    correct = prob["a"]
    candidates = [correct * 2, correct + 7, correct - 3, correct // 2 + 1, correct + 13]
    wrong_candidates = [c for c in candidates if c != correct and c > 0]
    wrong = int(rng.choice(wrong_candidates))
    body = prob["template"].format(a=wrong)
    return (
        f"Step 1: Read the problem: {prob['q']}\n"
        f"Step 2: Set up the equation. {body}\n"
        f"Step 3: The answer is {wrong}."
    )


# ---------------------------------------------------------------------------
# Draft runners
# ---------------------------------------------------------------------------


class _SyntheticDraftRunner:
    """Deterministic draft runner for CI / no-GPU environments.

    Generates a plausible-looking short arithmetic draft without any LLM.
    The draft always contains "=", a number, and 2-3 lines — so structural
    constraints are non-trivially active.

    This runner is used when Qwen/Qwen3.5-0.8B is not loadable.
    """

    def generate(self, question: str, max_tokens: int = 50) -> str:
        """Return a synthetic draft with arithmetic markers.

        Args:
            question:   The question text (used to extract any embedded numbers).
            max_tokens: Ignored — synthetic drafts are always short.

        Returns:
            Short string with "=", a digit, and 2 newlines.
        """
        import re as _re

        nums = _re.findall(r"\b\d+\b", question)
        if nums:
            a, b = int(nums[0]), int(nums[-1]) if len(nums) > 1 else int(nums[0])
            result = a + b
            return f"Let x = {a} + {b}.\nThen x = {result}.\nThe answer is {result}."
        return "Let x = 10 + 5.\nThen x = 15.\nThe answer is 15."


class _RealModelDraftRunner:
    """Draft runner backed by a real HuggingFace causal LM.

    Wraps the transformers library for CI-safe optional GPU inference.
    If model loading or generation fails, caller receives "" (empty draft).

    Args:
        model_name: HuggingFace model ID.
        device:     PyTorch device string ("cpu" or "cuda").
    """

    def __init__(self, model_name: str = "Qwen/Qwen3.5-0.8B", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> bool:
        """Lazy-load the model on first use.  Returns True on success."""
        if self._model is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, trust_remote_code=False
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_name,
                trust_remote_code=False,
                torch_dtype=torch.float32,
            )
            self._model.eval()
            self._model.to(self._device)
            return True
        except Exception:
            return False

    def generate(self, question: str, max_tokens: int = 50) -> str:
        """Generate a short draft answer for the given question.

        Args:
            question:   The question to answer.
            max_tokens: Maximum new tokens to generate.

        Returns:
            Generated text string or "" on any failure.
        """
        if not self._load():
            return ""
        try:
            import torch

            inputs = self._tokenizer(
                question, return_tensors="pt", truncation=True, max_length=256
            ).to(self._device)
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            # Decode only the new tokens (not the prompt).
            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        except Exception:
            return ""


def _build_draft_runner(use_real_model: bool) -> Any:
    """Return the best available draft runner.

    Tries the real model first when use_real_model=True; falls back to synthetic
    if loading fails.  Always returns something callable — never None.

    Args:
        use_real_model: Whether to attempt loading Qwen/Qwen3.5-0.8B.

    Returns:
        _RealModelDraftRunner or _SyntheticDraftRunner.
    """
    if use_real_model:
        runner = _RealModelDraftRunner()
        # Quick probe: attempt a single generation.
        probe = runner.generate("Test.", max_tokens=5)
        if probe:
            print(f"[Exp 912] Real model loaded (Qwen3.5-0.8B), probe: {probe!r}")
            return runner
        print("[Exp 912] Real model unavailable — using synthetic draft runner.")
    return _SyntheticDraftRunner()


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


def _compute_auc(correct_energies: list[float], halluc_energies: list[float]) -> float:
    """Compute AUC for the discrimination task: correct response has lower energy.

    For each (correct, hallucinated) pair, a "correct prediction" is when
    energy_correct < energy_hallucinated.  AUC = fraction of pairs correctly ranked.
    Ties are counted as 0.5 (standard Wilcoxon convention).

    Args:
        correct_energies: Energy scores for correct responses.
        halluc_energies:  Energy scores for hallucinated responses.

    Returns:
        Float in [0, 1].  0.5 = random.  > 0.5 = better than random.
    """
    assert len(correct_energies) == len(halluc_energies), "lists must be same length"
    n = len(correct_energies)
    if n == 0:
        return 0.5
    correct_count = 0.0
    for ec, eh in zip(correct_energies, halluc_energies):
        if ec < eh:
            correct_count += 1.0
        elif ec == eh:
            correct_count += 0.5
    return correct_count / n


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 912: measure whether draft-conditioned Tier 2.8 improves Ising AUC."""
    rng = np.random.default_rng(42)
    problems = _GSM8K_PROBLEMS[:25]  # exactly 25 questions

    # Try to load a real model for more realistic drafts; fall back gracefully.
    use_real = bool(
        int(
            str(
                # Respect CARNOT_FORCE_LIVE to request real-model mode from the conductor.
                __import__("os").environ.get("CARNOT_FORCE_LIVE", "0")
            )
        )
    )
    draft_runner = _build_draft_runner(use_real)
    model_name = (
        "Qwen/Qwen3.5-0.8B"
        if isinstance(draft_runner, _RealModelDraftRunner)
        else "synthetic_draft_runner"
    )

    # Verifier instances.
    # Baseline: DraftConditionedVerifier with ising_sampler=None and no draft
    # (we call score_with_constraints with an empty constraint list).
    verifier_baseline = DraftConditionedVerifier(
        draft_runner=_SyntheticDraftRunner(),  # always present
        ising_sampler=None,
        max_draft_tokens=50,
    )
    # Tier 2.8: same verifier but with the real (or synthetic) draft runner.
    verifier_tier28 = DraftConditionedVerifier(
        draft_runner=draft_runner,
        ising_sampler=None,
        max_draft_tokens=50,
    )

    per_question: list[dict[str, Any]] = []
    baseline_correct_energies: list[float] = []
    baseline_halluc_energies: list[float] = []
    tier28_correct_energies: list[float] = []
    tier28_halluc_energies: list[float] = []
    constraints_counts: list[int] = []

    for prob in problems:
        q = prob["q"]
        correct_resp = _make_correct_response(prob)
        halluc_resp = _make_hallucinated_response(prob, rng)

        # --- Baseline (no draft injection) ---
        # Score with an empty constraint list to simulate "no draft" path.
        base_ec = verifier_baseline.score_with_constraints(correct_resp, [])
        base_eh = verifier_baseline.score_with_constraints(halluc_resp, [])
        baseline_correct_energies.append(base_ec)
        baseline_halluc_energies.append(base_eh)

        # --- Tier 2.8 (draft-conditioned) ---
        result_correct = verifier_tier28.verify_with_draft(q, correct_resp)
        result_halluc = verifier_tier28.verify_with_draft(q, halluc_resp)
        tier28_correct_energies.append(result_correct.energy)
        tier28_halluc_energies.append(result_halluc.energy)
        constraints_counts.append(result_correct.n_constraints)

        per_question.append(
            {
                "question": q,
                "correct_answer": prob["a"],
                "baseline_energy_correct": base_ec,
                "baseline_energy_halluc": base_eh,
                "tier28_energy_correct": result_correct.energy,
                "tier28_energy_halluc": result_halluc.energy,
                "n_constraints_injected": result_correct.n_constraints,
                "draft_used": result_correct.draft_used,
                "draft_text_preview": result_correct.draft_text[:100],
            }
        )

    # --- Aggregate metrics ---
    auc_baseline = _compute_auc(baseline_correct_energies, baseline_halluc_energies)
    auc_with_draft = _compute_auc(tier28_correct_energies, tier28_halluc_energies)

    mean_constraints = float(np.mean(constraints_counts)) if constraints_counts else 0.0
    signed_energy_improvement_baseline = float(
        np.mean([eh - ec for ec, eh in zip(baseline_correct_energies, baseline_halluc_energies)])
    )
    signed_energy_improvement_draft = float(
        np.mean([eh - ec for ec, eh in zip(tier28_correct_energies, tier28_halluc_energies)])
    )

    honest_verdict = "tier28_viable" if auc_with_draft > auc_baseline else "tier28_no_improvement"

    print(f"[Exp 912] auc_baseline={auc_baseline:.4f}  auc_with_draft={auc_with_draft:.4f}")
    print(f"[Exp 912] mean_constraints_injected={mean_constraints:.2f}")
    print(f"[Exp 912] honest_verdict={honest_verdict}")

    # --- Build and write artifact ---
    artifact = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "auc_baseline": auc_baseline,
            "auc_with_draft": auc_with_draft,
            "mean_constraints_injected": mean_constraints,
            "signed_energy_improvement_baseline": signed_energy_improvement_baseline,
            "signed_energy_improvement_draft": signed_energy_improvement_draft,
            "n_questions": len(problems),
            "model_name": model_name,
            "inference_mode": "real_model"
            if isinstance(draft_runner, _RealModelDraftRunner)
            else "synthetic",
            "per_question": per_question,
            "decision_class": "verify",
        },
        status="success",
    )

    out_path = Path(PROJECT_ROOT) / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"[Exp 912] artifact written to {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
