"""OracleCorpusBuilder — ORACLE-style step-level labeled corpus construction.

**Why this module exists (RETRO-066):**

    JEPA v13 ECE=0.207 (target <0.10).  The root cause: all training data came from
    synthetic violations or binary correct/incorrect response labels.  Neither source
    provides step-level constraint labels that match the distribution of live LLM output.

    ORACLE (arXiv 2603.21140, AAAI 2026) closes this gap by generating multi-step
    reasoning data where EACH STEP has a symbolic verification label.  The key insight:
    if you use SymCodeVerifier to label the SAME model's outputs that will later be
    verified, the training distribution matches the live distribution exactly.  There
    is no offline/live gap because the labels are derived from live responses.

    This module implements:
      1. StepLabel — one step's verification outcome (correct / violated / unknown).
      2. OracleChain — a full reasoning chain annotated with per-step labels.
      3. OracleCorpusBuilder — takes a list of live response dicts and produces
         OracleChain objects by running SymCodeVerifier over each response.

Spec: REQ-DATA-012, REQ-DATA-013,
      SCENARIO-DATA-019, SCENARIO-DATA-020
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from carnot.pipeline.symcode_verifier import SymCodeVerifier


@dataclass
class StepLabel:
    """Verification outcome for a single reasoning step within a CoT chain.

    Fields:
        step_index       — Zero-based position of this step in the parent response.
        step_text        — Raw text of the reasoning step (sentence or line).
        violation_detected — True iff SymCodeVerifier found a numeric mismatch
                            between the executable expression result and the
                            stated result in this step's text.
        executed_result  — Float result of evaluating the Python expression
                           extracted from this step.  None when no arithmetic
                           expression was detected or the expression could not
                           be evaluated.
        stated_result    — Last numeric value stated in the step text.  None
                           when no number could be extracted.
        label            — Human-readable disposition: 'violated' when
                           violation_detected is True, 'correct' otherwise.
                           'unknown' is reserved for future use (e.g., when
                           neither executed_result nor stated_result is
                           available and the step contains no verifiable claim).
    """

    step_index: int
    step_text: str
    violation_detected: bool
    executed_result: Optional[float]
    stated_result: Optional[float]
    label: str  # 'correct' | 'violated' | 'unknown'


@dataclass
class OracleChain:
    """A live LLM reasoning chain annotated with ORACLE-style step-level labels.

    An OracleChain wraps the original live-response fields from fover_corpus_v5
    or live_pairs_578 and adds step-level violation labels produced by
    SymCodeVerifier.  Because the labels derive from the same model that
    generated the response, the label distribution matches the live inference
    distribution — this is the ORACLE property that closes RETRO-066.

    Fields:
        question_id      — Unique identifier for the question (question_index as
                           str, or hash of question text when not provided).
        question         — Original natural-language question.
        model_response   — Full model response text (chain-of-thought + answer).
        is_correct       — True when the model's final answer is correct
                           (from the source corpus label).
        model_id         — Identifier of the model that produced the response
                           (e.g., 'Qwen/Qwen3.5-0.8B').
        inference_mode   — How the response was generated ('live_gpu', 'cpu',
                           'synthetic', etc.).
        step_labels      — Ordered list of StepLabel objects, one per
                           SymCodeVerifier-segmented reasoning step.
        has_violation    — True iff at least one step has violation_detected=True.
        n_violated_steps — Count of steps where violation_detected is True.
    """

    question_id: str
    question: str
    model_response: str
    is_correct: bool
    model_id: str
    inference_mode: str
    step_labels: list[StepLabel]
    has_violation: bool
    n_violated_steps: int


class OracleCorpusBuilder:
    """Build an ORACLE-style corpus by labeling each reasoning step in live responses.

    Given a list of live-response dicts (schema: question, response, is_correct,
    model / model_id, optionally question_id and inference_mode), this class uses
    SymCodeVerifier to label every reasoning step with a violation flag and a
    human-readable 'correct' / 'violated' label.

    The resulting OracleChain objects can be serialised with dataclasses.asdict()
    and written to JSON for use as JEPA v14 training data.

    Why SymCodeVerifier (not an LLM judge or DSVD probe): SymCodeVerifier is
    distribution-invariant — it converts arithmetic claims to executable Python
    and evaluates them.  The result is always correct-by-construction, regardless
    of which model generated the response.  This means labels produced here will
    remain valid even as the base model changes between training runs.

    Args:
        verifier : SymCodeVerifier instance.  Pass llm_caller=None for CI / regex
                   mode, or a live Qwen3.5-0.8B caller for higher extraction accuracy.
    """

    def __init__(self, verifier: SymCodeVerifier) -> None:
        self.verifier = verifier

    def label_chain(self, chain: dict) -> OracleChain:
        """Label each reasoning step in a live response using SymCodeVerifier.

        The source dict must have at minimum: 'question', 'response', 'is_correct'.
        Optional fields: 'question_id', 'model' or 'model_id', 'inference_mode'.

        If question_id is absent, we fall back to str(hash(question)) so every
        chain has a stable, reproducible identifier even from old corpus formats
        that pre-date the question_id field.

        Why step-level (not response-level): the goal of ORACLE labeling is to
        give the JEPA predictor a supervision signal at each step boundary, not
        just at the end of the chain.  Response-level is-correct labels average
        over all steps and hide which specific step caused the error.
        """
        response = chain["response"]
        step_results = self.verifier.verify_response(response)

        step_labels = [
            StepLabel(
                step_index=i,
                step_text=s.text,
                violation_detected=s.violation_detected,
                executed_result=s.executed_result,
                stated_result=s.stated_result,
                label="violated" if s.violation_detected else "correct",
            )
            for i, s in enumerate(step_results)
        ]

        model_id = chain.get("model_id") or chain.get("model") or "unknown"
        question_id = chain.get("question_id")
        if question_id is None:
            question_id = str(hash(chain["question"]))

        return OracleChain(
            question_id=question_id,
            question=chain["question"],
            model_response=response,
            is_correct=bool(chain.get("is_correct", False)),
            model_id=model_id,
            inference_mode=chain.get("inference_mode", "live_gpu"),
            step_labels=step_labels,
            has_violation=any(s.violation_detected for s in step_results),
            n_violated_steps=sum(1 for s in step_results if s.violation_detected),
        )

    def build_corpus(self, live_pairs: list[dict]) -> list[OracleChain]:
        """Label all chains in a list of live-response dicts.

        Each dict is passed to label_chain() independently.  There is no batching
        here because SymCodeVerifier is CPU-only and the bottleneck is Python
        execution (safe_eval), not memory bandwidth.

        Returns a list of OracleChain objects in the same order as live_pairs.
        """
        return [self.label_chain(p) for p in live_pairs]
