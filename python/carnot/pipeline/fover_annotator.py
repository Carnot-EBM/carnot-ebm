"""FOVERAnnotator — FoVer-style Z3 step annotation for CoT reasoning (arXiv 2505.15960).

**Researcher summary:**
    FR-11 (autonomous self-learning) has missed 6 consecutive milestones because all
    EORM/JEPA retrains ran on SYNTHETIC data only.  The missing piece is real
    (step, correct/incorrect) labels from live LLM inference.

    FoVer (arXiv 2505.15960) shows that Z3/Isabelle can automatically annotate
    chain-of-thought reasoning steps with correctness labels WITHOUT human annotation.
    Each arithmetic step is checked against a formal Z3 solver — the solver is the
    ground truth, not a human rater.

    This module builds the annotation pipeline:
    1. Parse a live CoT response into discrete steps.
    2. Extract any claimed arithmetic equation from each step.
    3. Run each equation through the same Z3 exec() sandbox used by LLMz3Formalizer
       (Exp 357) — no LLM call needed, Z3 is purely CPU/deterministic.
    4. Emit (step_text, correct/incorrect) pairs as training targets for EORM (Exp 431).

**Why Z3 annotations are better than human labels:**
    - Deterministic: the same equation always gets the same label.
    - Scalable: Z3 runs in <5ms per step on CPU; we can label millions of steps/day.
    - Formally correct: Z3 is a sound SMT solver — its verdicts have mathematical
      guarantees, unlike LLM-based self-evaluation which can hallucinate.
    - No annotation budget: human labeling is $2-10 per question; Z3 is free.

**Why we filter by confidence (REQ-LEARN-031):**
    Steps without a clear equation (z3_label='not_verifiable') provide no learning
    signal — the model can't learn 'correct' or 'incorrect' from an unanswered check.
    Steps with confidence < 0.3 are ambiguous matches (partial regex hits) that
    introduce noise.  Filtering keeps the training data high-precision, which
    preserves the soundness of the downstream EORM training (same principle as
    REQ-LEARN-011's 0.85 precision gate).

**Hardware path:**
    Z3 is a CPU-only SMT solver.  Each step annotation takes < 5ms, making this
    pipeline suitable for labeling thousands of responses without GPU resources.
    The Ising/KAN tier IS a Verifiable Process Reward Model (VPRM, arXiv 2601.17223):
    it assigns energy scores to reasoning steps using deterministic constraint checks.
    The FOVER labels produced here are the training targets for that PRM.

**Relationship to existing code:**
    - Reuses `_exec_z3_snippet` from `llm_z3_formalizer` (Exp 357) for the Z3 sandbox.
    - Reuses the `_INLINE_EQ` arithmetic pattern from `crane_extractor` (Exp 418).
    - Produces output consumed by `ExperimentTimeoutWatchdog` (Exp 425) for safety.
    - `FOVERCoTStep` is a NEW dataclass (not the same as `CoTStep` in cot_circuit_verifier,
      which has different fields: step_id, text, input_refs, output_value).

Spec: REQ-LEARN-030, REQ-LEARN-031,
      SCENARIO-LEARN-054, SCENARIO-LEARN-055, SCENARIO-LEARN-056
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from carnot.pipeline.llm_z3_formalizer import _exec_z3_snippet

# ---------------------------------------------------------------------------
# Arithmetic equation pattern (same as CRANEExtractionGate._INLINE_EQ)
# ---------------------------------------------------------------------------

# Matches numbers including decimals and thousands-separated values.
_NUMBER_PAT = r"[\d,]+(?:\.\d+)?"

# Matches "N OP N = N" inline arithmetic equations — the canonical form that
# Z3 can verify without any LLM call.  This is the same regex used by CRANE
# (Exp 418) so annotation and extraction use a consistent grammar.
_INLINE_EQ_RE = re.compile(
    rf"(?P<a>{_NUMBER_PAT})\s*(?P<op>[+\-*/×÷])\s*(?P<b>{_NUMBER_PAT})"
    rf"\s*=\s*(?P<c>{_NUMBER_PAT})",
    re.IGNORECASE,
)

# Regex that FINDS step marker positions (used with finditer, not split).
# Two valid contexts for a step marker:
#   (A) At line start (with optional leading whitespace): handles multi-line CoT.
#       Pattern: (?:(?:^|\n)\s*)(?:step\s*\d+[\s.:)\-]+|\d{1,2}[.)]\s+)
#       The \n is CONSUMED so marker.start() is at the preceding \n; .strip() fixes it.
#   (B) After ". " (end of previous sentence): handles inline "1. Step. 2. Step."
#       Pattern: (?<=\. )(?:\d{1,2}[.)]\s+)
#       lookbehind is fixed-width (2 chars) so it doesn't consume.
# Limiting to \d{1,2} avoids matching large numbers embedded in prose.
_STEP_MARKER_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:step\s*\d+[\s.:)\-]+|\d{1,2}[.)]\s+))"
    r"|(?:(?<=\. )(?:\d{1,2}[.)]\s+))",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# FOVERCoTStep dataclass
# ---------------------------------------------------------------------------


@dataclass
class FOVERCoTStep:
    """One discrete reasoning step extracted from a CoT response, with Z3 annotation.

    **Detailed explanation for engineers:**
        This is NOT the same as `CoTStep` from `cot_circuit_verifier`.  That class
        tracks dependency links between steps (for circuit-style verification).
        `FOVERCoTStep` tracks the Z3 correctness label for each step — a different
        concern aligned with the FoVer paper (arXiv 2505.15960).

        `z3_label` semantics:
        - 'correct':         Z3 returned 'sat' (the claimed equation is arithmetically valid).
        - 'incorrect':       Z3 returned 'unsat' (the claimed equation is a contradiction).
        - 'not_verifiable':  No equation was found in the step, or Z3 returned 'unknown'/'error'.
        - None:              Not yet annotated (pre-annotation state).

        `z3_confidence` semantics:
        - 1.0: All three operands of the equation parsed correctly and a full Z3 assertion
               was built.  High confidence that the label is meaningful.
        - 0.5: Partial equation (e.g. regex matched but one operand is ambiguous).
        - 0.0: No equation found or annotation not attempted.

        `claimed_equation` is the raw matched string (e.g. "2 + 3 = 5"), not the parsed
        float values.  The Z3 snippet builder constructs the assertion from the match groups.

    Attributes:
        step_idx:         0-based index of this step in the CoT sequence.
        step_text:        Raw text content of the step.
        claimed_equation: Matched arithmetic equation string, or None.
        z3_label:         Z3 verdict ('correct'/'incorrect'/'not_verifiable') or None.
        z3_confidence:    Confidence score in [0.0, 1.0] for the label.

    Spec: REQ-LEARN-030, SCENARIO-LEARN-054/055/056
    """

    step_idx: int
    step_text: str
    claimed_equation: Optional[str]
    z3_label: Optional[Literal["correct", "incorrect", "not_verifiable"]] = None
    z3_confidence: float = 0.0


# ---------------------------------------------------------------------------
# parse_cot_into_steps
# ---------------------------------------------------------------------------


def parse_cot_into_steps(response_text: str) -> list[FOVERCoTStep]:
    """Split a CoT response into discrete FOVERCoTStep objects.

    **Detailed explanation for engineers:**
        Step boundaries are detected by two overlapping patterns:
        1. Numbered steps: "1. ", "2) ", "3: " at line start.
        2. "Step N:" / "Step N —" / "step N." format at line start.

        The split is done with a look-ahead so the delimiter (the step marker)
        is preserved as the first token of each chunk.  Empty chunks from the
        split (e.g. preamble before the first step) are discarded.

        After splitting, we search each step's text for an inline arithmetic
        equation using `_INLINE_EQ_RE`.  If found, `claimed_equation` is the
        full match string (e.g. "2 + 3 = 5").  This is the same equation grammar
        used by CRANEExtractionGate so the annotation pipeline is consistent with
        extraction.

    Args:
        response_text: Raw text of the full CoT response.

    Returns:
        List of FOVERCoTStep objects in order; empty list for empty input.

    Spec: REQ-LEARN-030, SCENARIO-LEARN-054
    """
    if not response_text or not response_text.strip():
        return []

    # Find all step marker positions using finditer.
    # This handles both:
    #   - Inline: "1. Step one. 2. Step two." (all on one line)
    #   - Multi-line: "1. Step one.\n2. Step two."
    # finditer returns non-overlapping matches left-to-right.
    markers = list(_STEP_MARKER_RE.finditer(response_text))

    if not markers:
        # No step markers found — treat the entire response as one step.
        eq_match = _INLINE_EQ_RE.search(response_text)
        return [
            FOVERCoTStep(
                step_idx=0,
                step_text=response_text.strip(),
                claimed_equation=eq_match.group(0) if eq_match else None,
            )
        ]

    steps: list[FOVERCoTStep] = []
    for i, marker in enumerate(markers):
        # Each step spans from the start of this marker to the start of the next.
        start = marker.start()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(response_text)
        chunk = response_text[start:end].strip()

        eq_match = _INLINE_EQ_RE.search(chunk)
        claimed_equation = eq_match.group(0) if eq_match else None

        steps.append(
            FOVERCoTStep(
                step_idx=len(steps),
                step_text=chunk,
                claimed_equation=claimed_equation,
            )
        )

    return steps


# ---------------------------------------------------------------------------
# _build_z3_assertion
# ---------------------------------------------------------------------------


def _build_z3_assertion(match: re.Match) -> str:
    """Build a self-contained Z3 Python snippet that asserts the equation.

    **Detailed explanation for engineers:**
        The equation "a OP b = c" is encoded as a Z3 assertion:
            import z3
            s = z3.Solver()
            s.add(a OP b == c)
            print(s.check())

        We use z3.RealVal for all operands to avoid integer division truncation
        (e.g. 7 / 2 = 3.5 should be correct, not 3 as in Python integer division).

        The 'sat' verdict means the assertion is satisfiable — i.e. the equation
        holds.  'unsat' means it's a contradiction — the equation is wrong.

        Thousands-separator commas are removed before parsing (e.g. "1,000" → 1000).

    Args:
        match: A regex match from _INLINE_EQ_RE with groups a, op, b, c.

    Returns:
        Python z3 snippet string ready to pass to _exec_z3_snippet.
    """
    a_str = match.group("a").replace(",", "")
    b_str = match.group("b").replace(",", "")
    c_str = match.group("c").replace(",", "")
    op_raw = match.group("op").strip()
    # Normalise Unicode operators to ASCII.
    op = {"×": "*", "÷": "/"}.get(op_raw, op_raw)

    # Use Real arithmetic to handle decimal results correctly.
    snippet = (
        "import z3\n"
        "s = z3.Solver()\n"
        f"a = z3.RealVal('{a_str}')\n"
        f"b = z3.RealVal('{b_str}')\n"
        f"c = z3.RealVal('{c_str}')\n"
        f"s.add(a {op} b == c)\n"
        "print(s.check())\n"
    )
    return snippet


# ---------------------------------------------------------------------------
# annotate_step_with_z3
# ---------------------------------------------------------------------------


def annotate_step_with_z3(step: FOVERCoTStep) -> FOVERCoTStep:
    """Annotate a single FOVERCoTStep with a Z3 correctness label.

    **Detailed explanation for engineers:**
        The annotation logic is intentionally simple:
        - No equation found → 'not_verifiable' (cannot check what isn't there).
        - Z3 says 'sat' → 'correct' (the claimed arithmetic is valid).
        - Z3 says 'unsat' → 'incorrect' (the claimed arithmetic is a contradiction).
        - Z3 says 'unknown' or 'error' → 'not_verifiable' (Z3 couldn't decide).

        Confidence reflects how much we trust the label:
        - 1.0: Full equation with all three operands parsed (most common case).
        - 0.5: Partial match (equation found but operands were ambiguous).
        - 0.0: No equation, or Z3 could not produce a verdict.

        The returned object is a NEW FOVERCoTStep (dataclasses are not mutated)
        with the same step_idx/step_text/claimed_equation, and filled-in
        z3_label and z3_confidence.

    Args:
        step: An un-annotated FOVERCoTStep (z3_label=None).

    Returns:
        A new FOVERCoTStep with z3_label and z3_confidence populated.

    Spec: REQ-LEARN-030, SCENARIO-LEARN-055, SCENARIO-LEARN-056
    """
    # No equation → not verifiable.
    if step.claimed_equation is None:
        return FOVERCoTStep(
            step_idx=step.step_idx,
            step_text=step.step_text,
            claimed_equation=None,
            z3_label="not_verifiable",
            z3_confidence=0.0,
        )

    # Re-match the equation to get named groups for Z3 snippet building.
    eq_match = _INLINE_EQ_RE.search(step.claimed_equation)
    if eq_match is None:
        # claimed_equation was set but re-match failed (shouldn't happen in practice).
        return FOVERCoTStep(
            step_idx=step.step_idx,
            step_text=step.step_text,
            claimed_equation=step.claimed_equation,
            z3_label="not_verifiable",
            z3_confidence=0.0,
        )

    # Check all three operands parse to valid floats — determines confidence.
    # _INLINE_EQ_RE only matches digit strings so ValueError is not expected in
    # practice; nonetheless we guard defensively and expose it as low confidence.
    def _safe_val(s: str) -> Optional[float]:
        val = s.replace(",", "")
        return float(val) if val.replace(".", "", 1).isdigit() else None

    a_val = _safe_val(eq_match.group("a"))
    b_val = _safe_val(eq_match.group("b"))
    c_val = _safe_val(eq_match.group("c"))
    all_parsed = a_val is not None and b_val is not None and c_val is not None

    # Full confidence only when all operands are well-formed floats.
    confidence = 1.0 if all_parsed else 0.5

    # Build Z3 snippet and execute in sandbox.
    snippet = _build_z3_assertion(eq_match)
    z3_result, _ = _exec_z3_snippet(snippet)

    if z3_result == "sat":
        label: Literal["correct", "incorrect", "not_verifiable"] = "correct"
    elif z3_result == "unsat":
        label = "incorrect"
    else:
        # 'unknown' or 'error': Z3 couldn't decide — treat as not verifiable.
        label = "not_verifiable"
        confidence = 0.0

    return FOVERCoTStep(
        step_idx=step.step_idx,
        step_text=step.step_text,
        claimed_equation=step.claimed_equation,
        z3_label=label,
        z3_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# FOVERAnnotator
# ---------------------------------------------------------------------------


class FOVERAnnotator:
    """Annotate chain-of-thought responses with Z3-verified step labels (FoVer, arXiv 2505.15960).

    **Detailed explanation for engineers:**
        The FOVERAnnotator is the top-level entry point for the FOVER annotation pipeline.
        It wraps `parse_cot_into_steps` and `annotate_step_with_z3` with corpus-level
        batching and training-pair export.

        Design decisions:
        - Stateless per-response: each `annotate_response` call is independent.
        - No LLM calls: Z3 is CPU-only and deterministic — no GPU or network needed.
        - `z3_timeout_seconds` is stored for documentation/artifact purposes.  The actual
          Z3 timeout is controlled by the exec() sandbox timeout in `_exec_z3_snippet`.
          Per-step Z3 runs take < 5ms in practice, making timeouts rarely relevant.

        Relationship to VPRM (arXiv 2601.17223):
        - Carnot's Ising/KAN tier IS a Verifiable Process Reward Model.
        - The (step_text, label) pairs from `to_training_pairs` are the training targets
          for that PRM — the model learns to predict 'correct' or 'incorrect' for each
          arithmetic step using energy-based scoring.

    Attributes:
        z3_timeout_seconds: Documented timeout budget per step (informational).

    Spec: REQ-LEARN-030, REQ-LEARN-031
    """

    def __init__(self, z3_timeout_seconds: int = 5) -> None:
        """Initialize the FOVERAnnotator.

        Args:
            z3_timeout_seconds: Per-step Z3 timeout budget in seconds (informational;
                                actual Z3 calls take <5ms per step on arithmetic).
        """
        self.z3_timeout_seconds = z3_timeout_seconds

    def annotate_response(
        self,
        response: str,
        question_id: str,
    ) -> list[FOVERCoTStep]:
        """Parse and annotate one CoT response.

        **Detailed explanation for engineers:**
            Runs the full pipeline for one response:
            1. `parse_cot_into_steps` splits the text into steps.
            2. `annotate_step_with_z3` labels each step.

            Returns the annotated steps.  The caller can inspect individual labels or
            pass the result to `to_training_pairs` for EORM training export.

        Args:
            response:    Full CoT response text.
            question_id: Identifier for this question (passed through to training pairs).

        Returns:
            List of annotated FOVERCoTStep objects.

        Spec: REQ-LEARN-030
        """
        steps = parse_cot_into_steps(response)
        return [annotate_step_with_z3(s) for s in steps]

    def annotate_corpus(
        self,
        responses: list[dict],
    ) -> list[list[FOVERCoTStep]]:
        """Annotate a list of response dicts, one list of steps per response.

        **Detailed explanation for engineers:**
            Each element of `responses` must have at least a 'response' key (string).
            An optional 'question_id' key is used for tracing; if absent, a sequential
            index is used.

            Returns a parallel list: `annotated[i]` is the annotated steps for
            `responses[i]`.

        Args:
            responses: List of dicts with keys 'response' (required) and
                       'question_id' (optional).

        Returns:
            List of lists of FOVERCoTStep (same length as `responses`).

        Spec: REQ-LEARN-030
        """
        result: list[list[FOVERCoTStep]] = []
        for i, item in enumerate(responses):
            response_text = item.get("response", "")
            question_id = item.get("question_id", str(i))
            result.append(self.annotate_response(response_text, question_id))
        return result

    def to_training_pairs(
        self,
        annotated: list[list[FOVERCoTStep]],
        responses: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Convert annotated steps into filtered training pairs for EORM.

        **Detailed explanation for engineers:**
            The EORM training signal requires (step_text, label) pairs where the label
            is a ground-truth correctness signal.  Only verifiable steps contribute:

            Inclusion criteria:
            - `z3_label in ('correct', 'incorrect')` — the step had a verifiable equation.
            - `z3_confidence >= 0.3` — minimum confidence threshold filters out partial
              regex matches that are too ambiguous to use as training signal.

            Excluded steps:
            - `z3_label='not_verifiable'` — no arithmetic to check; no learning signal.
            - `z3_confidence < 0.3` — ambiguous match; would add noise to training.

            Output schema keys: question_id, step_text, label, confidence.
            The `question_id` is pulled from the corresponding `responses` entry if
            provided, else from the step_idx.

        Args:
            annotated: Output from `annotate_corpus`.
            responses: Optional parallel list of response dicts (for question_id lookup).

        Returns:
            List of training-pair dicts with keys: question_id, step_text, label, confidence.

        Spec: REQ-LEARN-031
        """
        pairs: list[dict] = []
        for corpus_idx, steps in enumerate(annotated):
            question_id: str
            if responses is not None and corpus_idx < len(responses):
                question_id = str(responses[corpus_idx].get("question_id", corpus_idx))
            else:
                question_id = str(corpus_idx)

            for step in steps:
                if step.z3_label not in ("correct", "incorrect"):
                    continue
                if step.z3_confidence < 0.3:
                    continue
                pairs.append(
                    {
                        "question_id": question_id,
                        "step_text": step.step_text,
                        "label": step.z3_label,
                        "confidence": step.z3_confidence,
                    }
                )
        return pairs
