"""HermesV2StructuredLoop — forced COMPUTE: generation with live per-line verification.

**Why this module exists (RETRO-070, VR critical path #18):**

    Exp 641 (HermesV2LiveLoop) generates step-by-step and verifies at sentence
    boundaries, but still misses violations because instruction-tuned models rarely
    write explicit arithmetic — only ~12% of sentences contain a parseable operation.
    Exp 653 (StructuredEquationForcer) proved that injecting a system prompt addendum
    forces models to write 'COMPUTE: X op Y = Z' at every arithmetic step, raising
    detection_rate_on_forced to 1.0 on synthetic responses.

    This module combines both: force COMPUTE: format at generation time, then run
    SymCodeVerifier on each COMPUTE: line as it appears.  Since the model is forced to
    write every arithmetic step as a COMPUTE: line, the verifier can detect violations
    at every operation rather than only at the ~12% of sentences that happen to contain
    explicit arithmetic in free-form prose.

    Architectural improvement over all 17 prior attempts:
      - Exp 633 post-hoc baseline:  recall=0.12  (ceiling from prose arithmetic)
      - Exp 641 live loop:           recall=0.12  (same ceiling, different timing)
      - Exp 654 structured loop:     target recall >= 0.30  (COMPUTE: removes ceiling)

    The key insight: we cannot parse arithmetic that was never written.  Forcing the
    model to write it in a structured format is the only way to guarantee the verifier
    can see it.

Spec: REQ-VERIFY-147, REQ-VERIFY-148,
      SCENARIO-VERIFY-197, SCENARIO-VERIFY-198, SCENARIO-VERIFY-199
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from carnot.pipeline.structured_equation_forcer import StructuredEquationForcer
from carnot.pipeline.symcode_verifier import SymCodeVerifier


# ---------------------------------------------------------------------------
# HermesV2StructuredResult — result for one question in the structured loop
# ---------------------------------------------------------------------------


@dataclass
class HermesV2StructuredResult:
    """Result of running HermesV2StructuredLoop on one question.

    Each field records what the structured loop produced and what the verifier
    found.  This is the unit of measurement for the VR critical-path gate: a
    question "contributes to recall" iff the verifier detected at least one
    violation in the COMPUTE: lines (recall_contribution=True).

    Fields:
        question           — The original question text.
        full_response      — The model's complete response (or CI stub).
        compute_lines      — Bodies of all COMPUTE: lines found in the response
                             (each is the text after 'COMPUTE: ').
        n_compute_lines    — Convenience alias for len(compute_lines).
        n_violations       — Number of COMPUTE: lines where
                             SymCodeVerifier.detection_score > 0.0.
        n_hints            — Number of hints that would be injected (equals
                             n_violations in the current implementation; tracked
                             separately so future policies can suppress hints).
        recall_contribution — True iff at least one violation was detected
                              (n_violations > 0).  This is the recall signal
                              for RETRO-070: did we catch an error that the
                              post-hoc 12%-recall baseline would have missed?
    """

    question: str
    full_response: str
    compute_lines: list[str] = field(default_factory=list)
    n_compute_lines: int = 0
    n_violations: int = 0
    n_hints: int = 0
    recall_contribution: bool = False


# ---------------------------------------------------------------------------
# HermesV2StructuredLoop
# ---------------------------------------------------------------------------


class HermesV2StructuredLoop:
    """Generate with forced COMPUTE: format and live per-line SymCodeVerifier.

    This class is the key architectural improvement over Exp 641 (HermesV2LiveLoop).
    Instead of verifying at sentence boundaries (where only ~12% of sentences
    contain detectable arithmetic), it forces the model to emit explicit COMPUTE:
    lines for every arithmetic step, then runs SymCodeVerifier on each COMPUTE:
    line in sequence.

    The two-stage pipeline:
        1. StructuredEquationForcer.build_forced_prompt() injects the COMPUTE:
           instruction into the system prompt, so the model labels every
           arithmetic operation as 'COMPUTE: X op Y = Z' during generation.
        2. After generation, StructuredEquationForcer.extract_compute_lines()
           pulls out all COMPUTE: bodies, and SymCodeVerifier.detection_score()
           is run on each one.  Any line where detection_score > 0.0 is a
           violation, and a hint counter is incremented.

    In CI mode (llm_caller=None): generate_structured() uses the forcer's
    synthetic response ('We have 47 apples. COMPUTE: 47 + 28 = 76 So total is
    76.') which contains a deliberate arithmetic error (47+28=75, not 76) so
    the verifier detects a violation and sets recall_contribution=True.  This
    lets the entire pipeline be exercised without a GPU.

    Args:
        llm_caller    : callable(prompt: str, system: str) -> str for live mode.
                        Signature matches StructuredEquationForcer's llm_caller.
                        Pass None for CI / synthetic mode.
        verifier      : SymCodeVerifier instance for per-line violation checking.
        forcer        : StructuredEquationForcer instance providing the system
                        prompt and COMPUTE: extraction logic.
        max_sentences : Soft cap on generation length (not enforced in this
                        class directly; passed through for future extensions
                        that break generation into sentences).

    Spec: REQ-VERIFY-147, REQ-VERIFY-148
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str, str], str]],
        verifier: SymCodeVerifier,
        forcer: StructuredEquationForcer,
        max_sentences: int = 12,
    ) -> None:
        self.llm_caller = llm_caller
        self.verifier = verifier
        self.forcer = forcer
        self.max_sentences = max_sentences

    def generate_structured(self, question: str) -> HermesV2StructuredResult:
        """Generate with forced COMPUTE: format and live per-line verification.

        Steps:
        1. Build the COMPUTE:-forcing system prompt via forcer.build_forced_prompt().
        2. Call the LLM (or use the CI stub response if llm_caller is None).
        3. Extract all COMPUTE: lines from the response.
        4. Run SymCodeVerifier.detection_score('COMPUTE: ' + line) on each line.
           A score > 0.0 means the verifier detected an arithmetic violation on
           that line.  Each violation increments both n_violations and n_hints.
        5. Set recall_contribution=True if any violation was detected.

        Why prepend 'COMPUTE: ' before passing to detection_score: the verifier's
        CI regex mode uses _EXPR_RE to find 'N op M' patterns, and the COMPUTE:
        prefix helps segment_steps isolate the arithmetic expression correctly.

        Args:
            question: The question to answer with forced COMPUTE: format.

        Returns:
            HermesV2StructuredResult with all fields populated.
        """
        system_prompt, _ = self.forcer.build_forced_prompt(question)

        if self.llm_caller is None:
            # CI stub: deliberate wrong arithmetic (47+28=76 is wrong; correct=75)
            # so the verifier detects a violation and recall_contribution=True.
            forced_response = "We have 47 apples. COMPUTE: 47 + 28 = 76 So total is 76."
        else:
            full_prompt = system_prompt + "\n\nQuestion: " + question + "\nAnswer:"
            forced_response = self.llm_caller(full_prompt, "")

        compute_lines = self.forcer.extract_compute_lines(forced_response)

        n_violations = 0
        n_hints = 0
        for line in compute_lines:
            score = self.verifier.detection_score("COMPUTE: " + line)
            if score > 0.0:
                n_violations += 1
                n_hints += 1

        recall_contribution = n_violations > 0

        return HermesV2StructuredResult(
            question=question,
            full_response=forced_response,
            compute_lines=compute_lines,
            n_compute_lines=len(compute_lines),
            n_violations=n_violations,
            n_hints=n_hints,
            recall_contribution=recall_contribution,
        )
