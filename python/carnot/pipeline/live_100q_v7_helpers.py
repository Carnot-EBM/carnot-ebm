"""Helpers for Exp 514 Live 100q Precision v7 benchmark.

**Why this module exists:**
    Exp 514 is the seventh attempt at closing RETRO-033 (live 100q benchmark).
    The blocking chain across six previous milestones was: zombie VRAM -> env propagation
    -> conductor VRAM -> VRAM budget -> runtime VRAM state.  Exp 513 (JITVRAMCheck)
    resolves the final blocker by querying VRAM immediately before each model.load().

    This module extracts three reusable helpers from the Exp 514 script so they can be
    unit-tested in isolation at 100% coverage:
    - ``load_jit_gated_model`` — JIT VRAM gate then load
    - ``run_100q_benchmark``   — full baseline+pipeline benchmark for one model
    - ``write_cot_pairs``      — write FOVER-format CoT pairs atomically

**FOVER format:**
    Each entry in the CoT pairs JSON file is a dict with keys:
        question  (str)  — original question text
        cot_text  (str)  — model's chain-of-thought response
        correct   (bool) — whether the answer was correct after pipeline pass
        model_id  (str)  — model identifier (e.g. 'Gemma4-INT4')

Spec: REQ-BENCH-014, REQ-BENCH-015,
      SCENARIO-BENCH-033, SCENARIO-BENCH-034
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from carnot.pipeline.jit_vram_check import JITVRAMCheck  # noqa: F401 — module-level for mockability

_log = logging.getLogger(__name__)

__all__ = [
    "PrecisionBenchmarkResult",
    "load_jit_gated_model",
    "run_100q_benchmark",
    "wilson_ci",
    "write_cot_pairs",
]

# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------


def wilson_ci(n_correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval for a proportion.

    Wilson CI is preferred over the naive Wald interval because it remains
    valid near 0 and 1 — situations that arise when a model gets everything
    right or everything wrong on small subsets.

    Parameters
    ----------
    n_correct : int
        Number of correct answers.
    n : int
        Total number of questions.
    z : float
        Z-score for the confidence level. Default 1.96 = 95%.

    Returns
    -------
    (lower, upper) : tuple of float
        Lower and upper bounds of the confidence interval, clamped to [0, 1].

    Spec: REQ-BENCH-014
    """
    if n == 0:
        return 0.0, 0.0
    p = n_correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# PrecisionBenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class PrecisionBenchmarkResult:
    """Result from run_100q_benchmark for one model.

    Contains baseline accuracy (no pipeline) alongside pipeline accuracy,
    Wilson 95% CI on the pipeline accuracy, and all derived signals needed
    to populate the Exp 514 artifact.

    Fields
    ------
    model_id : str
        Human-readable model name.
    n : int
        Number of questions answered.
    baseline_correct : int
        Number of baseline-correct answers (no pipeline).
    pipeline_correct : int
        Number of pipeline-correct answers.
    baseline_accuracy : float
        baseline_correct / n.
    pipeline_accuracy : float
        pipeline_correct / n.
    wilson_95ci_lower : float
        Wilson 95% CI lower bound on pipeline_accuracy.
    wilson_95ci_upper : float
        Wilson 95% CI upper bound on pipeline_accuracy.
    signed_improvement : float
        pipeline_accuracy - baseline_accuracy (unclamped, honest).
    is_positive : bool
        True iff signed_improvement > 0.
    cot_pairs : list of dict
        FOVER-format CoT pairs collected during the pipeline pass.

    Spec: REQ-BENCH-014
    """

    model_id: str
    n: int
    baseline_correct: int
    pipeline_correct: int
    baseline_accuracy: float
    pipeline_accuracy: float
    wilson_95ci_lower: float
    wilson_95ci_upper: float
    signed_improvement: float
    is_positive: bool
    cot_pairs: List[dict]

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary (excludes cot_pairs list for brevity)."""
        return {
            "model_id": self.model_id,
            "n": self.n,
            "baseline_correct": self.baseline_correct,
            "pipeline_correct": self.pipeline_correct,
            "baseline_accuracy": self.baseline_accuracy,
            "pipeline_accuracy": self.pipeline_accuracy,
            "wilson_95ci_lower": self.wilson_95ci_lower,
            "wilson_95ci_upper": self.wilson_95ci_upper,
            "signed_improvement": self.signed_improvement,
            "is_positive": self.is_positive,
        }


# ---------------------------------------------------------------------------
# load_jit_gated_model
# ---------------------------------------------------------------------------


def load_jit_gated_model(
    loader_factory: Callable[[], Any],
    model_id: str,
    required_gb: float,
    device: int,
) -> Optional[Any]:
    """Gate a model load through JITVRAMCheck and return the loaded model or None.

    This is the RETRO-051 fix applied per-model-load.  Planning-time VRAM
    forecasts (VRAMBudgetLedger) are computed once at script startup; by the
    time model.load() fires, the GPU state may have changed.  This function
    queries VRAM immediately before the load — in the same call frame — and
    aborts cleanly if VRAM is insufficient rather than crashing with CUDA OOM.

    Algorithm:
        1. Instantiate JITVRAMCheck(device_id=device).
        2. Call gate_model_load(model_id, required_gb).
        3. If is_cleared=False: log and return None (caller writes deferred artifact).
        4. Otherwise: call loader_factory() to create the loader, then .load(), return loader.

    Parameters
    ----------
    loader_factory : callable
        Zero-argument callable that returns a loader object with a ``.load()`` method.
        Called ONLY after the JIT VRAM gate clears.
    model_id : str
        Human-readable model identifier used in log messages and JITVRAMResult.
    required_gb : float
        Minimum free VRAM in GB required before the load is permitted.
    device : int
        Zero-based CUDA device index to query (0 for primary, 1 for secondary).

    Returns
    -------
    loader or None
        The loaded model object if VRAM is sufficient; None if the gate blocked the load.

    Spec: REQ-BENCH-014, SCENARIO-BENCH-033
    """
    checker = JITVRAMCheck(device_id=device)
    gate_result = checker.gate_model_load(model_id, required_gb=required_gb)

    if not gate_result.is_cleared:
        _log.warning(
            "load_jit_gated_model: JIT gate BLOCKED model=%r device=%d "
            "available=%.2f GB required=%.2f GB — load aborted",
            model_id, device, gate_result.available_gb, required_gb,
        )
        return None

    _log.info(
        "load_jit_gated_model: JIT gate CLEARED model=%r device=%d "
        "available=%.2f GB required=%.2f GB — proceeding with load",
        model_id, device, gate_result.available_gb, required_gb,
    )
    loader = loader_factory()
    loader.load()
    return loader


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_answer(text: str) -> Optional[str]:
    """Extract the numeric final answer from a GSM8K-style response.

    Looks for the official '#### N' GSM8K delimiter first, then falls back to
    the last number found in the response text.

    Parameters
    ----------
    text : str
        Raw model output.

    Returns
    -------
    str or None
        Normalized numeric string or None when no number is found.
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        raw = m.group(1)
        try:
            f = float(raw)
            if f == int(f):
                return str(int(f))
        except ValueError:
            pass
        return raw
    nums = _NUMBER_RE.findall(text)
    if not nums:
        return None
    last = nums[-1].replace(",", "")
    try:
        f = float(last)
        if f == int(f):
            return str(int(f))
        return last
    except ValueError:
        return last


def _is_correct(response: str, gold: Optional[str]) -> bool:
    """True when the extracted response answer matches the gold answer.

    Uses 0.501 tolerance to handle floating-point rounding in answers like
    "72.0" vs "72".

    Parameters
    ----------
    response : str
        Raw model output.
    gold : str or None
        Ground-truth answer string extracted from the GSM8K answer field.
    """
    if not gold or not response:
        return False
    extracted = _extract_answer(response)
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


# ---------------------------------------------------------------------------
# run_100q_benchmark
# ---------------------------------------------------------------------------


def run_100q_benchmark(
    inference_fn: Callable[[str], str],
    model_id: str,
    questions: List[dict],
    extractor: Any,
) -> PrecisionBenchmarkResult:
    """Run baseline + pipeline benchmark for one model on a question list.

    Two passes:
    1. BASELINE — raw model output, no pipeline.  Counts baseline_correct.
    2. PIPELINE — VeriCoT+VPRM extraction on each response; if violations are
       found, the model is prompted again with an explicit repair prompt.
       Counts pipeline_correct and accumulates FOVER CoT pairs.

    The signed_improvement is unclamped: a negative value signals regression and
    must not be hidden from the caller.

    Parameters
    ----------
    inference_fn : callable
        ``(prompt: str) -> str`` — generates one model response.
    model_id : str
        Human-readable name embedded in each CoT pair and the result.
    questions : list of dict
        Each entry must have 'question' (str) and 'answer' (str, GSM8K format
        with optional '#### N' delimiter).
    extractor : any
        Object with an ``.extract(text)`` method that returns a list of violations
        (empty list = no violations detected).  Pass an IntegratedExtractor instance.

    Returns
    -------
    PrecisionBenchmarkResult
        Full result including Wilson CI and FOVER CoT pairs.

    Spec: REQ-BENCH-014, REQ-BENCH-015
    """
    n = len(questions)

    # Pass 1: Baseline — no pipeline corrections
    baseline_correct = 0
    for q in questions:
        response = inference_fn(q["question"])
        gold = _extract_answer(q["answer"])
        if _is_correct(response, gold):
            baseline_correct += 1

    baseline_accuracy = baseline_correct / max(n, 1)
    _log.info(
        "run_100q_benchmark [%s] BASELINE: %d/%d = %.4f",
        model_id, baseline_correct, n, baseline_accuracy,
    )

    # Pass 2: Pipeline — VeriCoT+VPRM with repair on violations
    pipeline_correct = 0
    cot_pairs: List[dict] = []

    for q in questions:
        response = inference_fn(q["question"])
        violations = extractor.extract(response)

        if violations:
            repair_prompt = (
                f"Question: {q['question']}\n\n"
                "Your previous answer contained logical or arithmetic errors. "
                "Please solve step by step carefully and double-check every calculation."
            )
            response = inference_fn(repair_prompt)

        gold = _extract_answer(q["answer"])
        correct = _is_correct(response, gold)
        if correct:
            pipeline_correct += 1

        cot_pairs.append({
            "question": q["question"],
            "cot_text": response,
            "correct": correct,
            "model_id": model_id,
        })

    pipeline_accuracy = pipeline_correct / max(n, 1)
    signed_improvement = pipeline_accuracy - baseline_accuracy
    ci_lower, ci_upper = wilson_ci(pipeline_correct, n)

    _log.info(
        "run_100q_benchmark [%s] PIPELINE: %d/%d = %.4f delta=%.4f CI=[%.4f, %.4f]",
        model_id, pipeline_correct, n, pipeline_accuracy,
        signed_improvement, ci_lower, ci_upper,
    )

    return PrecisionBenchmarkResult(
        model_id=model_id,
        n=n,
        baseline_correct=baseline_correct,
        pipeline_correct=pipeline_correct,
        baseline_accuracy=baseline_accuracy,
        pipeline_accuracy=pipeline_accuracy,
        wilson_95ci_lower=ci_lower,
        wilson_95ci_upper=ci_upper,
        signed_improvement=signed_improvement,
        is_positive=signed_improvement > 0,
        cot_pairs=cot_pairs,
    )


# ---------------------------------------------------------------------------
# write_cot_pairs
# ---------------------------------------------------------------------------


def write_cot_pairs(
    cot_pairs: List[dict],
    path: str,
) -> int:
    """Write FOVER-format CoT pairs atomically to disk.

    Writes to a .tmp file first and renames atomically so a crash mid-write
    never leaves a partial JSON file.  Each entry must already conform to the
    FOVER schema (keys: question, cot_text, correct, model_id).

    Parameters
    ----------
    cot_pairs : list of dict
        Pairs to write.  Each dict must have: question (str), cot_text (str),
        correct (bool), model_id (str).
    path : str
        Output file path.  Parent directories are created if absent.

    Returns
    -------
    int
        Number of pairs written.

    Spec: REQ-BENCH-015, SCENARIO-BENCH-034
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(cot_pairs, indent=2))
    tmp.replace(out)
    _log.info("write_cot_pairs: wrote %d pairs to %s", len(cot_pairs), out)
    return len(cot_pairs)
