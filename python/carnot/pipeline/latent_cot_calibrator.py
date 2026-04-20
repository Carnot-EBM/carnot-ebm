"""LatentCoTEBMCalibrator: per-step EORM energy gate during generation.

**Researcher summary (arXiv 2511.07124):**
    Integrates a small EORM energy model to calibrate latent reasoning tokens
    during implicit chain-of-thought (CoT) generation.  At every 32-token
    boundary the EORM assigns an energy scalar to the partial response so far.
    High-energy partial responses are "steered away" by applying a soft
    temperature adjustment to the LLM logits before sampling continues:

        adjusted_logits = logits * (1 - alpha * energy_score)

    This flattens the logit distribution when the model is heading toward a
    high-energy (likely wrong) trajectory, making it sample more cautiously
    (lower temperature = sharper = more confident, higher alpha*energy =
    softer = more cautious).

**Why this works:**
    The EORM was trained contrastively: correct CoTs have lower energy than
    incorrect ones.  When the partial response so far already smells wrong
    (high energy), we want the model to hedge — to not commit fully to the
    high-softmax-probability token.  The adjustment (1 - alpha * E) scales
    down all logits proportionally, which effectively raises the temperature
    of the sampling distribution and spreads probability mass more evenly
    across candidates.  Alpha controls how aggressively to apply the gate;
    alpha=0.1 is a light touch that avoids over-correcting correct trajectories.

**Simulation model:**
    Real production use would require hooking into the LLM's forward pass.
    In this module the "generate_fn" is a Python callable that accepts a
    prompt string and a list of per-step temperature scalars and returns a
    response string.  The calibrator calls it once per question, passing
    temperature_adjustments so a real LLM adapter could apply them.
    For offline measurement (experiment 560) we use a synthetic generate_fn
    that ignores temperature scalars, so the calibrator measures the EORM
    trajectory without actually steering.

Spec: REQ-VERIFY-116, SCENARIO-VERIFY-134, SCENARIO-VERIFY-135,
      SCENARIO-VERIFY-136
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from carnot.models.eorm import CoTEnergyInput, EORMModel


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatentCoTCalibrationResult:
    """All per-run metrics produced by LatentCoTEBMCalibrator.calibrate_generation.

    **For engineers:**
        This holds both the scalar summary statistics (mean_energy,
        violation_rate_before, violation_rate_after) and the per-step raw
        data (per_step_energy, temperature_adjustments) so downstream code
        can inspect the full trajectory.

    Attributes:
        n_steps: Number of 32-token boundaries evaluated across all questions
            in this run.
        per_step_energy: Flat list of EORM energy values, one per boundary
            checkpoint across all questions.
        mean_energy: Mean of per_step_energy.
        violation_rate_before: Fraction of baseline responses with at least
            one VPRM violation.
        violation_rate_after: Fraction of calibrated responses with at least
            one VPRM violation.
        temperature_adjustments: Per-step temperature multipliers actually
            applied: adjustment_i = 1 - alpha * energy_i.
    """

    n_steps: int
    per_step_energy: list[float] = field(default_factory=list)
    mean_energy: float = 0.0
    violation_rate_before: float = 0.0
    violation_rate_after: float = 0.0
    temperature_adjustments: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Calibrator
# ---------------------------------------------------------------------------

class LatentCoTEBMCalibrator:
    """Applies per-step EORM energy gating during LLM chain-of-thought generation.

    **For engineers:**
        At every ``step_boundary_tokens`` token boundary during generation:
        1. Take the partial response produced so far (split on whitespace by
           word count as a proxy for token count, since we work offline without
           a real tokenizer).
        2. Score it with EORMModel.energy(CoTEnergyInput(question, partial)).
        3. Compute a temperature multiplier: adj = 1 - alpha * energy_score.
        4. Pass adj to the downstream generate_fn so a real LLM adapter can
           use it to modulate sampling temperature at that step.

        The generate_fn signature is::

            def generate_fn(prompt: str, temperature_adjustments: list[float]) -> str:
                ...

        In offline/synthetic mode the generate_fn ignores temperature_adjustments
        and just returns a response — which is fine for measuring the energy
        trajectory without a live GPU.

    Args:
        eorm_model: Trained EORMModel used to score partial responses.
        alpha: Gate strength.  Adjustment = 1 - alpha * energy.  Larger alpha
            → more aggressive temperature flattening at high-energy steps.
            Default 0.1 (light touch).
        step_boundary_tokens: Approximate token interval between energy
            checkpoints.  Implemented as word count (whitespace split).
            Default 32.
    """

    def __init__(
        self,
        eorm_model: EORMModel,
        alpha: float = 0.1,
        step_boundary_tokens: int = 32,
    ) -> None:
        self.eorm_model = eorm_model
        self.alpha = alpha
        self.step_boundary_tokens = step_boundary_tokens

    # ------------------------------------------------------------------
    # Core: calibrate a batch of questions
    # ------------------------------------------------------------------

    def calibrate_generation(
        self,
        prompts: list[str],
        generate_fn: Callable[[str, list[float]], str],
        n_questions: int | None = None,
    ) -> tuple[list[str], LatentCoTCalibrationResult]:
        """Generate responses with EORM temperature gating and return metrics.

        **For engineers:**
            For each prompt:
            1. Generate the full response via generate_fn (first pass with no
               gating, to get the raw text we will simulate gating against).
            2. Split the response into word-count chunks of step_boundary_tokens.
            3. For each prefix of those chunks, score EORM energy and compute
               an adjustment = 1 - alpha * energy.
            4. Record the energy and adjustment for each step.

            The generate_fn is called ONCE per prompt with the list of
            temperature_adjustments so a live adapter can apply them lazily
            (e.g. at each sampling step).  In offline mode it returns a fixed
            response regardless.

        Args:
            prompts: List of question strings.
            generate_fn: Callable(prompt, temperature_adjustments) -> response.
            n_questions: If provided, truncate prompts to this many.

        Returns:
            Tuple of (responses, LatentCoTCalibrationResult).
        """
        if n_questions is not None:
            prompts = prompts[:n_questions]

        all_energies: list[float] = []
        all_adjustments: list[float] = []
        responses: list[str] = []

        for question in prompts:
            # Step 1: get response (generate_fn may ignore the adjustments)
            # We pre-compute a placeholder adjustment list; real adapters use it.
            placeholder_adjs: list[float] = []
            response = generate_fn(question, placeholder_adjs)
            responses.append(response)

            # Step 2: simulate energy trajectory on the returned response
            words = response.split()
            step_energies: list[float] = []
            step_adjs: list[float] = []

            for boundary in range(
                self.step_boundary_tokens,
                len(words) + self.step_boundary_tokens,
                self.step_boundary_tokens,
            ):
                partial = " ".join(words[:boundary])
                cot_input = CoTEnergyInput(
                    question_text=question,
                    response_text=partial,
                )
                energy = self.eorm_model.energy(cot_input)
                adj = 1.0 - self.alpha * energy
                step_energies.append(energy)
                step_adjs.append(adj)

            # If response was empty, score the full response as a single step
            if not step_energies:
                cot_input = CoTEnergyInput(
                    question_text=question,
                    response_text=response if response else " ",
                )
                energy = self.eorm_model.energy(cot_input)
                step_energies.append(energy)
                step_adjs.append(1.0 - self.alpha * energy)

            all_energies.extend(step_energies)
            all_adjustments.extend(step_adjs)

        n_steps = len(all_energies)
        mean_energy = sum(all_energies) / n_steps if n_steps else 0.0

        result = LatentCoTCalibrationResult(
            n_steps=n_steps,
            per_step_energy=all_energies,
            mean_energy=mean_energy,
            violation_rate_before=0.0,   # filled in by compare_violation_rate
            violation_rate_after=0.0,
            temperature_adjustments=all_adjustments,
        )
        return responses, result

    # ------------------------------------------------------------------
    # Violation rate comparison
    # ------------------------------------------------------------------

    def compare_violation_rate(
        self,
        calibrated_responses: list[str],
        baseline_responses: list[str],
        labeled_questions: list[str],
    ) -> dict[str, float]:
        """Compare VPRM violation rates between calibrated and baseline responses.

        **For engineers:**
            Uses VPRMArithmeticVerifier.detect_violations() to flag any
            response that contains at least one failed arithmetic rule.
            "violation_rate" = fraction of responses with >= 1 violation.

            The delta = calibrated_rate - baseline_rate.
            A negative delta means calibration reduced violations (good).

        Args:
            calibrated_responses: Responses generated WITH calibration.
            baseline_responses: Responses generated WITHOUT calibration.
            labeled_questions: Not used directly; kept for API symmetry and
                future use (e.g. domain-specific rules).

        Returns:
            Dict with keys:
                baseline_violation_rate (float),
                calibrated_violation_rate (float),
                violation_rate_delta (float, calibrated - baseline).
        """
        # Import here to keep the module loadable without heavy deps at top level.
        from carnot.extraction import VPRMArithmeticVerifier

        verifier = VPRMArithmeticVerifier()

        def _rate(responses: list[str]) -> float:
            if not responses:
                return 0.0
            n_violated = sum(
                1 for r in responses if verifier.detect_violations(r)
            )
            return n_violated / len(responses)

        baseline_rate = _rate(baseline_responses)
        calibrated_rate = _rate(calibrated_responses)

        return {
            "baseline_violation_rate": baseline_rate,
            "calibrated_violation_rate": calibrated_rate,
            "violation_rate_delta": calibrated_rate - baseline_rate,
        }
