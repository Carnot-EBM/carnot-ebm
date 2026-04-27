"""ThinkPRMVerifier — generative step-level Process Reward Model for Carnot.

**Researcher summary:**
    ThinkPRM (arXiv 2504.16828) shows that having a verifier LLM generate a
    chain-of-thought (CoT) explanation *before* emitting CORRECT/INCORRECT
    dramatically outperforms heuristic rule-based PRMs and discriminative
    classifiers under equivalent token budgets. Key results:
      - +8% on GPQA-Diamond vs discriminative PRM
      - +4.5% on LiveCodeBench
      - +7.2% over LLM-as-a-Judge
    All using only 1% of the labels required by supervised discriminative baselines.

    Exp 924 (R-PRM Tier 2.9) produced AUC delta=0 because it used HEURISTIC
    rule-based explanations before scoring. ThinkPRMVerifier fixes this by
    requiring MODEL-GENERATED chain-of-thought for each step verdict.

**Why model-generated CoT is superior to heuristic explanations:**
    A heuristic rule like 'contains arithmetic error' classifies the SURFACE FORM
    of the step. It cannot detect errors that require semantic understanding, such
    as correct arithmetic applied to the wrong quantities. A generative verifier
    reasons over the meaning of the step: it extracts the claim, checks the
    computation, and produces an explanation grounded in the actual mathematics.
    The VERDICT then reflects that reasoning, not a surface pattern.

**Architecture position:**
    ThinkPRMVerifier operates at the step level (one reasoning step at a time),
    unlike CarnotThinkProbe which operates at the response level (full LLM output).
    They are complementary: ThinkProbe is a fast pre-filter on the whole response;
    ThinkPRMVerifier is a fine-grained scorer for Process Reward Model applications
    like beam search re-ranking and chain-of-thought filtering.

**CI-safety:**
    When llm_caller is None (default), verify_step() returns a deterministic
    'uncertain' stub. Callers that need deterministic arithmetic checking for tests
    can pass a stub llm_caller that implements simple regex arithmetic. See
    tests/python/test_thinkprm_verifier.py for the canonical stub pattern.

**Phase 3 path:**
    In Phase 3, ThinkPRMVerifier becomes a non-autoregressive energy minimiser:
    each reasoning step is a latent state, and the verifier settles into the
    correct verdict via gradient descent rather than token prediction. The
    ThinkPRMResult interface is designed to survive that transition intact.

Spec: REQ-VERIFY-098, SCENARIO-VERIFY-130
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# ThinkPRMResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ThinkPRMResult:
    """Result of ThinkPRMVerifier.verify_step() for a single reasoning step.

    **Detailed explanation for engineers:**
        Each field reflects one piece of the step-level verification decision:

        step_text:
            The original step string (e.g. "47 + 28 = 76"). Preserved verbatim
            for traceability — callers can correlate results back to the input
            corpus without maintaining a parallel index.

        verdict:
            One of 'correct', 'incorrect', or 'uncertain'. These mirror the
            ThinkPRM paper's three-class output. 'uncertain' is used when:
              - llm_caller is None (CI stub), OR
              - the LLM output did not contain a parseable VERDICT: line, OR
              - the step is genuinely ambiguous (e.g. "the result is approximately N").

        confidence:
            Float in [0.0, 1.0] representing how confident the verifier is in
            its verdict. Sources:
              - 0.95 for a clear VERDICT: CORRECT or VERDICT: INCORRECT from LLM.
              - 0.5 for 'uncertain' (maximum entropy / no information).
            In future work, this will be extracted from the LLM's logit distribution
            over the verdict tokens (log P(CORRECT) - log P(INCORRECT)).

        reasoning_steps:
            The raw chain-of-thought text produced by the LLM, or an empty string
            in CI stub mode. Preserved for debugging, auditing, and future
            fine-tuning label generation. Callers should NOT parse this string
            programmatically — it is free-form LLM output and its format may
            change across model versions.

        latency_ms:
            Wall-clock time from verify_step() entry to return, in milliseconds.
            Includes LLM inference time (GPU path) or ~0 ms (CI stub). Used to
            benchmark the per-step overhead of adding ThinkPRM to the pipeline.

    Spec: REQ-VERIFY-098
    """

    step_text: str
    verdict: Literal["correct", "incorrect", "uncertain"]
    confidence: float
    reasoning_steps: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_THINKPRM_STEP_TEMPLATE = """\
You are a mathematical step verifier. Your task is to verify whether the following \
reasoning step is mathematically correct.

Reasoning step to verify:
\"\"\"{step}\"\"\"

{context_block}\
Follow these steps exactly:

Step 1: Extract the arithmetic or logical claim being made in this step.
Step 2: Verify the claim by computing or reasoning carefully. Show your work.
Step 3: State your verdict.

At the end of Step 3, you MUST write exactly one of:
VERDICT: CORRECT
VERDICT: INCORRECT

Use VERDICT: CORRECT if the step is mathematically correct.
Use VERDICT: INCORRECT if the step contains a clear arithmetic or logical error.
Do not use VERDICT: UNCERTAIN — if genuinely ambiguous, pick the most likely verdict.
"""

_CONTEXT_BLOCK_TEMPLATE = """\
Context (preceding steps):
\"\"\"{context}\"\"\"

"""

_VERDICT_PATTERN = re.compile(r"VERDICT\s*:\s*(CORRECT|INCORRECT)", re.IGNORECASE)


def _build_step_prompt(step_text: str, context: str) -> str:
    """Build the ThinkPRM verification prompt for a single reasoning step.

    The prompt instructs the LLM to follow three explicit steps before emitting
    a parseable VERDICT: CORRECT or VERDICT: INCORRECT line. This mirrors the
    ThinkPRM template from arXiv 2504.16828 adapted for step-level granularity
    rather than full-response granularity.

    The triple-quoted step text prevents prompt injection where the step might
    contain newlines or 'Step N:' markers that could confuse the structure.

    Spec: REQ-VERIFY-098
    """
    context_block = _CONTEXT_BLOCK_TEMPLATE.format(context=context) if context else ""
    return _THINKPRM_STEP_TEMPLATE.format(step=step_text, context_block=context_block)


def _parse_step_output(
    output: str,
) -> tuple[Literal["correct", "incorrect", "uncertain"], float, str]:
    """Parse the LLM's CoT output into a (verdict, confidence, reasoning) tuple.

    Uses the LAST occurrence of VERDICT: CORRECT/INCORRECT so that if the model
    echoes the prompt (which includes the VERDICT format instruction), only the
    model's final decision counts.

    Falls back to ('uncertain', 0.5, output) when no VERDICT line is found —
    this means Ising or other downstream verifiers will still run as a safety net.
    We never fall back to 'incorrect' on a parse failure, because that would
    suppress downstream verification on an ambiguous input.

    Spec: REQ-VERIFY-098, SCENARIO-VERIFY-130
    """
    matches = _VERDICT_PATTERN.findall(output)
    if not matches:
        return "uncertain", 0.5, output

    raw = matches[-1].upper()
    verdict: Literal["correct", "incorrect", "uncertain"] = (
        "correct" if raw == "CORRECT" else "incorrect"
    )
    return verdict, 0.95, output


# ---------------------------------------------------------------------------
# ThinkPRMVerifier
# ---------------------------------------------------------------------------


@dataclass
class ThinkPRMVerifier:
    """Generative step-level Process Reward Model for Carnot verification pipeline.

    **Detailed explanation for engineers:**
        ThinkPRMVerifier scores individual reasoning steps from multi-step LLM
        solutions (e.g., GSM8K chains). For each step, it:
          1. Builds a 3-step CoT verification prompt (extract claim → check → verdict).
          2. Calls the LLM to generate the verification chain-of-thought.
          3. Parses VERDICT: CORRECT / VERDICT: INCORRECT from the output.
          4. Returns a ThinkPRMResult with the verdict and confidence score.

        The confidence score (P(CORRECT)) replaces Ising energy as the step-level
        reward signal in the Tier 2.9 R-PRM cascade. Higher confidence means the
        verifier considers the step more likely correct.

        CI-safety (REQ-VERIFY-098):
            When llm_caller is None (the default), verify_step() returns a
            deterministic 'uncertain' verdict with confidence=0.5. This allows
            the pipeline routing logic and AUROC evaluation code to be tested
            in CI without GPU hardware. Callers that need deterministic
            arithmetic-aware stubs for testing should pass a stub llm_caller;
            see tests/python/test_thinkprm_verifier.py for the pattern.

        Confidence threshold:
            confidence_threshold controls what the CALLER considers a 'definitive'
            verdict. ThinkPRMVerifier itself always emits the parsed verdict;
            the threshold is advisory information for callers that want to implement
            abstention (e.g., only accept verdicts with confidence > 0.8).
            Currently stored but not used internally — the caller decides policy.

    Attributes:
        llm_caller: Optional callable (prompt: str) -> str. If None, CI stub mode.
        confidence_threshold: Advisory threshold for callers. Default 0.8.

    Spec: REQ-VERIFY-098
    """

    llm_caller: Callable[[str], str] | None = None
    confidence_threshold: float = 0.8

    def verify_step(self, step_text: str, context: str = "") -> ThinkPRMResult:
        """Verify a single reasoning step via ThinkPRM-style generative CoT.

        **Detailed explanation for engineers:**
            CI stub path (llm_caller is None):
                Returns ThinkPRMResult with verdict='uncertain', confidence=0.5,
                reasoning_steps='', latency_ms~=0. No model loading occurs.
                This is safe for CI pipelines that have no GPU access.

            Live LLM path (llm_caller is set):
                1. Build the 3-step verification prompt for the step.
                2. Call llm_caller(prompt) to get the CoT output.
                3. Parse VERDICT: CORRECT / VERDICT: INCORRECT from the output.
                4. Return verdict='correct'/'incorrect'/'uncertain' with confidence.

            'uncertain' is returned when the LLM output contains no parseable
            VERDICT line (model failed to follow template). This preserves
            downstream verification as a safety net.

        Args:
            step_text: The reasoning step to verify (e.g. "47 + 28 = 76").
            context: Optional string of preceding steps for context window.

        Returns:
            ThinkPRMResult with verdict, confidence, reasoning, and latency.

        Spec: REQ-VERIFY-098, SCENARIO-VERIFY-130
        """
        t0 = time.perf_counter()

        if self.llm_caller is None:
            # CI stub: no GPU, no model — return 'uncertain' deterministically.
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ThinkPRMResult(
                step_text=step_text,
                verdict="uncertain",
                confidence=0.5,
                reasoning_steps="",
                latency_ms=latency_ms,
            )

        prompt = _build_step_prompt(step_text, context)
        raw_output = self.llm_caller(prompt)
        verdict, confidence, reasoning = _parse_step_output(raw_output)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ThinkPRMResult(
            step_text=step_text,
            verdict=verdict,
            confidence=confidence,
            reasoning_steps=reasoning,
            latency_ms=latency_ms,
        )

    def batch_verify(
        self, steps: list[str], contexts: list[str] | None = None
    ) -> list[ThinkPRMResult]:
        """Verify a list of reasoning steps, returning results in input order.

        **Detailed explanation for engineers:**
            Calls verify_step() for each step sequentially. The results list
            preserves the order of the input list so callers can zip(steps, results)
            or use index-based access without additional bookkeeping.

            Currently sequential (not parallel) because:
              1. LLM inference is typically the bottleneck; parallelism requires
                 either vLLM or multi-GPU setup which is environment-dependent.
              2. Sequential execution makes latency_ms per-step meaningful as a
                 wall-clock measurement rather than a queuing measurement.

            Future work: add optional async batching via asyncio when vLLM is
            available. The interface here (list in, list out) is designed to
            support that extension without API breakage.

        Args:
            steps: List of reasoning step strings to verify.
            contexts: Optional parallel list of context strings. If None or
                shorter than steps, missing contexts default to empty string.

        Returns:
            List of ThinkPRMResult, one per step, in the same order as input.

        Spec: REQ-VERIFY-098
        """
        contexts = contexts or []
        results: list[ThinkPRMResult] = []
        for i, step in enumerate(steps):
            ctx = contexts[i] if i < len(contexts) else ""
            results.append(self.verify_step(step, ctx))
        return results
