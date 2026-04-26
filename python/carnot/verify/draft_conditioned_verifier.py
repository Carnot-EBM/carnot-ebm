"""DraftConditionedVerifier — Tier 2.8 draft-conditioned constraint pre-conditioning.

**Why this module exists (arXiv 2603.03305, Draft-Conditioned Constrained Decoding):**

    When the Tier 2.7 CausalReasoningVerifier is uncertain, the expensive Ising
    sampler (Tier 3) must decide whether a response is correct.  Ising is powerful
    but blind to the *structure* of the response's likely answer: it samples from a
    generic constraint set that does not know whether the response was heading toward
    "42" or "1000" or a subtraction-heavy computation.

    The key insight from 2603.03305: generate a CHEAP draft response first using a
    small model at low temperature (greedy-ish), then condition the constraint set on
    the STRUCTURE of that draft — not its answer, but its structural markers
    (proposed value ranges, which arithmetic operations appear, how many steps).

    Draft-conditioning works because:
    1. The small model at temperature=0.1 is highly reproducible: its draft structure
       is a strong prior on what the final answer structure should look like.
    2. Injecting structural constraints BEFORE Ising runs narrows the search space —
       Ising samples spin configurations consistent with "the answer is in 0-100" AND
       "subtraction was used" rather than the full unconstrained space.
    3. The draft is CHEAP (50 tokens, Qwen3.5-0.8B on CPU in ~1s) relative to the
       cost of constraint violations reaching the user.

    This is Tier 2.8: positioned between Tier 2.7 (CausalReasoningVerifier) and
    Tier 3 (Ising), run ONLY when earlier tiers did not clear the response.

**Structural constraints extracted:**

    - answer_in_range_<lo>_to_<hi>: first numeric = X in draft → constrain [0, 2X]
    - arithmetic_op_<add|subtract|multiply|divide>: detected math ops
    - n_steps_<k>: number of sentences in draft (proxy for CoT depth)

    These are passed to ising_constraint_injector so Ising samples spin states
    consistent with the draft's structural expectations.

Spec: REQ-TIER2-010
SCENARIO-TIER2-010
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Utility: does draft structurally agree with response?
# ---------------------------------------------------------------------------


def draft_differs_from_response(draft: str, response: str) -> bool:
    """Return True when the draft and response lead to different structural conclusions.

    **Why structural, not textual:**
        We do not care whether the draft and response use the same wording.
        We care whether they agree on the NUMERIC CONCLUSION.  If the draft
        says "= 42" and the response says "= 99", the draft-conditioned
        constraints will fire on the WRONG range — that is the mismatch we
        need to flag so the downstream pipeline can adjust confidence.

    Two texts are considered structurally different when their final
    extracted numbers differ by more than 20%.  When no number is extractable
    from either text, they are treated as agreeing (no number to disagree on).

    Args:
        draft: Draft response text (from small model, low temperature).
        response: Full response text being verified.

    Returns:
        True when draft and response reach structurally different conclusions.

    Spec: REQ-TIER2-010
    """
    def _extract_last_number(text: str) -> float | None:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not nums:
            return None
        try:
            return float(nums[-1])
        except ValueError:
            return None

    draft_num = _extract_last_number(draft)
    resp_num = _extract_last_number(response)

    if draft_num is None or resp_num is None:
        return False  # no numeric conclusion to compare

    # Avoid division by zero; treat near-zero numbers as matching
    if abs(draft_num) < 1e-6 and abs(resp_num) < 1e-6:
        return False

    denom = max(abs(draft_num), abs(resp_num), 1.0)
    return abs(draft_num - resp_num) / denom > 0.20


# ---------------------------------------------------------------------------
# DraftConditionedVerifier
# ---------------------------------------------------------------------------


class DraftConditionedVerifier:
    """Generate a cheap draft and extract structural constraints for Ising pre-conditioning.

    **How to use (Tier 2.8 wiring in ThreeTierPipeline):**

        verifier = DraftConditionedVerifier()
        result = verifier.condition_and_verify(question, response)
        # Inject result["structural_constraints"] into Ising before Tier 3 runs.

    **CPU-only operation:**
        This verifier runs on CPU with Qwen3.5-0.8B loaded via the transformers
        pipeline API.  At temperature=0.1, 50 tokens takes ~1s on modern hardware.
        The model is loaded lazily on first call and cached for subsequent calls
        on the same instance.  In test/CI environments the model load is mocked
        and extract_structural_constraints() is exercised directly.

    Args:
        draft_model_name: HuggingFace model ID for the draft generator.
                          Default is Qwen/Qwen3.5-0.8B (CPU-runnable, small).
        draft_max_tokens: Maximum number of tokens in the draft.
                          Default 50 — just enough to see arithmetic structure.
        draft_temperature: Temperature for draft generation.
                           Default 0.1 (near-greedy) for reproducible structure.

    Spec: REQ-TIER2-010
    """

    def __init__(
        self,
        draft_model_name: str = "Qwen/Qwen3.5-0.8B",
        draft_max_tokens: int = 50,
        draft_temperature: float = 0.1,
    ) -> None:
        self.draft_model_name = draft_model_name
        self.draft_max_tokens = draft_max_tokens
        self.draft_temperature = draft_temperature
        self._pipeline: Any | None = None  # lazy-loaded transformers pipeline

    # ------------------------------------------------------------------
    # _load_pipeline() — lazy model loader
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Any:
        """Load the draft model via transformers.pipeline (CPU, text-generation).

        Why lazy: model loading takes ~2-3s on CPU.  Deferring to first call
        means experiments that do not exercise Tier 2.8 pay zero overhead.
        The pipeline is cached on self._pipeline so subsequent calls are free.

        Returns:
            Loaded transformers.pipeline instance.

        Raises:
            ImportError when transformers is not installed (CI-safe: tests
            should patch this method rather than require the real model).
        """
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline  # type: ignore[import]

        self._pipeline = pipeline(
            "text-generation",
            model=self.draft_model_name,
            device="cpu",
            trust_remote_code=False,
        )
        return self._pipeline

    # ------------------------------------------------------------------
    # generate_draft()
    # ------------------------------------------------------------------

    def generate_draft(self, question: str) -> str:
        """Generate a short draft response for the question using the small model.

        The draft is generated at low temperature (near-greedy) so its STRUCTURE
        is reproducible.  We only care about structure here — which arithmetic
        operations appear, how many steps, what numeric range the answer is in.
        The draft answer itself may be wrong; only the structure matters.

        Args:
            question: The question whose answer structure we want to probe.

        Returns:
            Draft response string, at most draft_max_tokens tokens.

        Spec: REQ-TIER2-010
        """
        pipe = self._load_pipeline()
        outputs = pipe(
            question,
            max_new_tokens=self.draft_max_tokens,
            temperature=self.draft_temperature,
            do_sample=self.draft_temperature > 0.0,
            return_full_text=False,
        )
        # transformers pipeline returns list of dicts: [{"generated_text": "..."}]
        if outputs and isinstance(outputs, list) and "generated_text" in outputs[0]:
            return str(outputs[0]["generated_text"]).strip()
        return ""

    # ------------------------------------------------------------------
    # extract_structural_constraints()
    # ------------------------------------------------------------------

    def extract_structural_constraints(self, draft: str) -> list[str]:
        """Parse a draft response for structural markers and return constraint strings.

        **What structural markers we look for and why:**

        1. Numeric answer range (= X patterns):
           If the draft contains "= 42", we know the answer SHOULD be in [0, 84].
           Ising can apply this as a range constraint: spins representing answers
           outside this range get penalized.  We use 2X as the upper bound to
           give the final model a ~2x tolerance.

        2. Arithmetic operation type:
           "47 + 28" → arithmetic_op_add
           "100 - 37" → arithmetic_op_subtract
           "3 * 4" → arithmetic_op_multiply
           "20 / 4" → arithmetic_op_divide
           Ising constraints can enforce that the arithmetic operators PRESENT
           in the solution match those signaled by the draft.

        3. Step count:
           The number of non-empty sentences is a proxy for CoT depth.
           A 3-step draft means the response should have ~3 reasoning steps.
           Ising can penalize responses that are significantly longer or shorter.

        All returned constraint strings use underscore-separated lowercase tokens
        so they are valid as Ising spin-variable names and can be logged verbatim.

        Args:
            draft: Draft response text from generate_draft().

        Returns:
            List of constraint strings, e.g.:
                ["answer_in_range_0_to_84", "arithmetic_op_subtract", "n_steps_3"]
            Empty list when no structural markers are found.

        Spec: REQ-TIER2-010
        """
        constraints: list[str] = []

        # 1. Numeric answer range: look for "= <number>" patterns
        eq_matches = re.findall(r"=\s*(-?\d+(?:\.\d+)?)", draft)
        if eq_matches:
            # Use the last "= X" as the proposed answer (most likely the final answer)
            try:
                proposed_answer = float(eq_matches[-1])
                lo = 0
                hi = int(abs(proposed_answer) * 2) + 1
                constraints.append(f"answer_in_range_{lo}_to_{hi}")
            except (ValueError, OverflowError):
                pass  # numeric conversion failed — skip range constraint

        # 2. Arithmetic operations
        # Check for subtraction BEFORE general minus to avoid false matches
        if re.search(r"\d\s*[-−]\s*\d", draft):
            constraints.append("arithmetic_op_subtract")
        if re.search(r"\d\s*[+]\s*\d", draft):
            constraints.append("arithmetic_op_add")
        if re.search(r"\d\s*[×*]\s*\d", draft):
            constraints.append("arithmetic_op_multiply")
        if re.search(r"\d\s*[÷/]\s*\d", draft):
            constraints.append("arithmetic_op_divide")

        # 3. Step count: number of non-empty sentences (split on ". " or newline)
        sentences = [s.strip() for s in re.split(r"[.\n]+", draft) if s.strip()]
        n_steps = len(sentences)
        if n_steps > 0:
            constraints.append(f"n_steps_{n_steps}")

        return constraints

    # ------------------------------------------------------------------
    # condition_and_verify()
    # ------------------------------------------------------------------

    def condition_and_verify(self, question: str, response: str) -> dict[str, Any]:
        """Generate draft, extract structural constraints, return advisory for Tier 3.

        This is the main entry point for Tier 2.8.  The pipeline calls this
        AFTER Tier 2.7 (CausalReasoningVerifier) when that tier is uncertain,
        and BEFORE Tier 3 (Ising) runs.

        The returned dict contains everything Tier 3 needs to pre-condition its
        constraint set.  The actual Ising run still happens in ThreeTierPipeline;
        this method just computes the constraints to inject.

        Args:
            question: The question being verified.
            response: The LLM response being verified (the expensive model's output).

        Returns:
            dict with keys:
                draft                      — raw draft text from small model
                structural_constraints     — list of constraint strings to inject
                draft_mismatch             — True when draft and response lead to
                                             structurally different conclusions
                tier28_advisory            — always "draft_conditioned"

        Spec: REQ-TIER2-010
        SCENARIO-TIER2-010
        """
        draft = self.generate_draft(question)
        structural_constraints = self.extract_structural_constraints(draft)
        mismatch = draft_differs_from_response(draft, response)

        return {
            "draft": draft,
            "structural_constraints": structural_constraints,
            "draft_mismatch": mismatch,
            "tier28_advisory": "draft_conditioned",
        }
