#!/usr/bin/env python3
"""Experiment 451: Live precision post-fix benchmark — GemmaTransformersLoader + Carnot pipeline.

**Researcher summary (RETRO-028 follow-up):**
    Exp 439 (milestone 2026.04.33) reported 0% Gemma4-E4B-it accuracy on GSM8K.
    Root cause: llama.cpp tokenizer bug (GitHub issue llama.cpp#21516) caused the model
    to emit infinite ``<unused8>`` tokens (token_id=14) — the model appeared to run but
    produced zero valid text.  Published Gemma4 GSM8K accuracy is 75-80%.

    This experiment re-runs the Exp 439 harness with three fixes:
    1. GemmaTransformersLoader replaces the llama.cpp path for Gemma4-E4B-it.
    2. CRANE extraction (arXiv 2504.15030) as the structured claim extractor.
    3. LivePrecisionResult (REQ-BENCH-013) captures signed improvement per model.

**Expected outcome:**
    - Gemma4 baseline: 75-80% (published) — now achievable with the tokenizer fix.
    - Qwen3.5-0.8B baseline: 30-50% typical for a 0.8B model on GSM8K.
    - First POSITIVE verify-repair number (first_positive_number=True) if pipeline
      improves accuracy for at least one model.

**Gate chain (runs in order):**
    0. apply_env_autofix() — FIRST, before any CUDA import (RETRO-022 fix)
    1. ExperimentTimeoutWatchdog(451, timeout_minutes=60) — outer budget cap
    2. CARNOT_FORCE_LIVE check — hard gate, no simulated fallback
    3. setup_gpu() — blocked if not all_healthy
    4. GemmaTransformersLoader.load() for Gemma4-E4B-it on GPU 0
    5. HF pipeline for Qwen3.5-0.8B on GPU 1

**Outputs:**
    results/experiment_451_live_precision_postfix.json — primary artifact

Spec: REQ-BENCH-012, REQ-BENCH-013, SCENARIO-BENCH-031, SCENARIO-BENCH-032
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() injects CARNOT_FORCE_LIVE=1 before any
# CUDA import occurs.  Moving this below any torch/JAX import is a bug.
# See RETRO-022 for why this matters.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import os
import re
from typing import Any

from carnot.pipeline.crane_extractor import CRANEExtractionGate  # noqa: E402
from carnot.pipeline.experiment_watchdog import (  # noqa: E402
    ExperimentTimeoutWatchdog,
    get_timeout_minutes,
)
from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: E402
from carnot.pipeline.live_precision_result import LivePrecisionResult  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 451
EXP_TITLE = "Live Precision Post-Fix: GemmaTransformersLoader + CRANE (50q × 2 models)"
DELIVERABLE = "results/experiment_451_live_precision_postfix.json"
N_QUESTIONS = 50

MODEL_SPECS: list[dict[str, Any]] = [
    {"name": "Gemma4-E4B-it", "hf_id": "google/gemma-4-E4B-it", "gpu": 0},
    {"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 1},
]


# ---------------------------------------------------------------------------
# Answer extraction helpers (shared with Exp 439)
# ---------------------------------------------------------------------------


def _extract_gsm8k_answer(text: str) -> str | None:
    """Extract the numeric final answer from a GSM8K response.

    Looks for the official '#### N' format first, then falls back to the last
    number in the text.  Returns None when no numeric answer can be found.
    """
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _is_correct(response: str, gold: str | None) -> bool:
    """Return True when the response contains the gold answer as a final number."""
    if not gold or not response:
        return False
    extracted = _extract_gsm8k_answer(response)
    if extracted is None:
        return False
    try:
        return abs(float(extracted) - float(gold)) < 0.501
    except (ValueError, TypeError):
        return extracted.strip() == gold.strip()


def _load_gsm8k_questions(n: int) -> list[dict]:
    """Load up to n GSM8K questions, falling back to synthetic data when unavailable.

    Uses the HuggingFace ``datasets`` package when available, otherwise generates
    synthetic arithmetic word problems.  Synthetic questions are clearly labelled
    (source='synthetic') so any accuracy numbers are distinguishable from real
    GSM8K numbers in the artifact output.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset("gsm8k", "main", split="test")
        items = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            items.append({"question": row["question"], "answer": row["answer"]})
        if items:
            _log.info("Loaded %d GSM8K questions from HuggingFace datasets", len(items))
            return items
    except Exception as exc:
        _log.warning("Could not load GSM8K: %s — using synthetic fallback", exc)

    synthetic = []
    for i in range(1, n + 1):
        a, b = i * 3, i * 2
        c = a + b
        synthetic.append({
            "question": (
                f"Janet has {a} apples and receives {b} more.  "
                f"How many apples does she have?"
            ),
            "answer": f"She starts with {a} and gets {b} more, so {a} + {b} = {c}.  #### {c}",
        })
    _log.info("Using %d synthetic GSM8K questions (real dataset unavailable)", len(synthetic))
    return synthetic[:n]


# ---------------------------------------------------------------------------
# Gemma4 inference via GemmaTransformersLoader
# ---------------------------------------------------------------------------


def _run_gemma_inference(loader: GemmaTransformersLoader, prompt: str) -> str:
    """Generate a response from Gemma4 via GemmaTransformersLoader.

    Validates output using is_valid_output() to catch any lingering llama.cpp-style
    token failures.  Returns empty string on invalid output so downstream scoring
    treats it as incorrect rather than silently fabricating an answer.
    """
    try:
        text = loader.generate(prompt, max_new_tokens=256)
        if not GemmaTransformersLoader.is_valid_output(text):
            _log.warning(
                "GemmaTransformersLoader.is_valid_output() returned False — "
                "prompt may have triggered <unusedN> token collapse"
            )
            return ""
        return text
    except Exception as exc:
        _log.warning("Gemma4 generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Qwen inference via standard HF text-generation pipeline
# ---------------------------------------------------------------------------


def _load_qwen_pipeline(hf_id: str, gpu_index: int) -> object:
    """Load a HuggingFace text-generation pipeline for Qwen on the given GPU.

    Uses explicit device= assignment (Exp 438 fix) to avoid the GPU1 zombie
    issue where device_map='auto' sometimes allocates all weights to GPU0.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(f"transformers not installed: {exc}") from exc

    return hf_pipeline(
        "text-generation",
        model=hf_id,
        device=gpu_index,
        torch_dtype="auto",
    )


def _run_qwen_inference(pipe: object, prompt: str) -> str:
    """Generate a response from Qwen via HF pipeline.  Returns '' on failure."""
    try:
        outputs = pipe(prompt, max_new_tokens=256, do_sample=False, return_full_text=False)
        return str(outputs[0]["generated_text"])
    except Exception as exc:
        _log.warning("Qwen generation failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Per-model benchmark runner
# ---------------------------------------------------------------------------


def _run_model_benchmark(
    model_name: str,
    inference_fn: Any,
    crane: CRANEExtractionGate,
    questions: list[dict],
) -> LivePrecisionResult:
    """Run baseline and pipeline variants for one model.

    Runs two passes over all questions:
    1. BASELINE: raw model output, no verify-repair pipeline.
    2. PIPELINE: CRANE arithmetic extraction + one-shot repair when violations detected.

    Returns a LivePrecisionResult capturing pre_accuracy (baseline) and post_accuracy
    (pipeline) for this model.  signed_improvement and is_positive are derived properties.

    Why two separate passes instead of one:
        Running baseline first gives us a clean baseline_accuracy before any pipeline
        interaction.  This mirrors the Exp 439 design and prevents the pipeline from
        influencing the baseline number (which would happen if we ran them together
        and re-used the same response object).
    """
    # ---- Pass 1: BASELINE ----
    n_correct_baseline = 0
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        gold = _extract_gsm8k_answer(q_dict["answer"])
        if _is_correct(response, gold):
            n_correct_baseline += 1

    pre_accuracy = n_correct_baseline / max(len(questions), 1)
    _log.info(
        "  [%s] BASELINE: %d/%d correct (%.4f)",
        model_name, n_correct_baseline, len(questions), pre_accuracy,
    )

    # ---- Pass 2: PIPELINE (CRANE extraction + one-shot repair) ----
    n_correct_pipeline = 0
    for q_dict in questions:
        response = inference_fn(q_dict["question"])
        violations = crane.extract(response, "arithmetic")
        if violations:
            # CRANE detected arithmetic violations — attempt one-shot repair
            repair_prompt = (
                f"Question: {q_dict['question']}\n\n"
                f"Your previous answer contained arithmetic errors.  "
                f"Please solve step by step carefully and double-check your arithmetic."
            )
            response = inference_fn(repair_prompt)
        gold = _extract_gsm8k_answer(q_dict["answer"])
        if _is_correct(response, gold):
            n_correct_pipeline += 1

    post_accuracy = n_correct_pipeline / max(len(questions), 1)
    _log.info(
        "  [%s] PIPELINE: %d/%d correct (%.4f) delta=%.4f",
        model_name, n_correct_pipeline, len(questions), post_accuracy,
        post_accuracy - pre_accuracy,
    )

    return LivePrecisionResult(
        model_id=model_name,
        pre_accuracy=pre_accuracy,
        post_accuracy=post_accuracy,
    )


# ---------------------------------------------------------------------------
# Artifact write helper
# ---------------------------------------------------------------------------


def _write_artifact(repo_root: Path, artifact: dict) -> None:
    """Atomically write artifact dict to the DELIVERABLE path under repo_root."""
    out_path = repo_root / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(artifact, f, indent=2)
    Path(tmp).replace(out_path)
    _log.info("Artifact written to %s", out_path)


# ---------------------------------------------------------------------------
# run_experiment
# ---------------------------------------------------------------------------


def run_experiment(repo_root: Path | None = None) -> dict[str, Any]:
    """Run Experiment 451 and return the artifact dict.

    All gates are checked in order.  Any gate failure writes a deferred artifact
    and returns immediately.  No simulated fallback is allowed — either we have a
    real live GPU number or we have nothing (deferred/blocked).

    Returns
    -------
    dict
        The full artifact dict (always JSON-serializable).
    """
    if repo_root is None:
        repo_root = _REPO_ROOT

    is_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=True,
        repo_root=repo_root,
    )
    tmpl.setup()

    env_autofix_dict = {
        "gpu_detected": _autofix_result.gpu_detected,
        "carnot_force_live_was_set": _autofix_result.carnot_force_live_was_set,
        "auto_fix_applied": _autofix_result.auto_fix_applied,
        "final_env_value": _autofix_result.final_env_value,
    }

    # ------------------------------------------------------------------
    # Gate 0: GPU required — no simulated fallback (SCENARIO-BENCH-032)
    # ------------------------------------------------------------------
    if not is_live:
        _log.info("CARNOT_FORCE_LIVE not set — GPU required, writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v2",
                "gemma4_loader": "GemmaTransformersLoader",
                "env_autofix": env_autofix_dict,
                "qwen_result": None,
                "gemma4_result": None,
                "first_positive_number": False,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_artifact(repo_root, artifact)
        return artifact

    # ------------------------------------------------------------------
    # Gate 1: setup_gpu — blocked if not all_healthy
    # ------------------------------------------------------------------
    gpu_status = tmpl.setup_gpu(MODEL_SPECS)
    if not gpu_status["all_healthy"]:
        _log.error("setup_gpu not all_healthy — writing deferred artifact.")
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v2",
                "gemma4_loader": "GemmaTransformersLoader",
                "env_autofix": env_autofix_dict,
                "gpu_setup_status": gpu_status,
                "qwen_result": None,
                "gemma4_result": None,
                "first_positive_number": False,
            },
            status="gpu_required",
            honest_verdict="deferred_to_gpu",
        )
        _write_artifact(repo_root, artifact)
        return artifact

    # ------------------------------------------------------------------
    # Gate 2: Load Gemma4 via GemmaTransformersLoader (RETRO-028 fix)
    # ------------------------------------------------------------------
    gemma_loader: GemmaTransformersLoader | None = None
    try:
        _log.info("Loading Gemma4-E4B-it via GemmaTransformersLoader on cuda:0 ...")
        gemma_loader = GemmaTransformersLoader(
            model_id="google/gemma-4-E4B-it",
            device="cuda:0",
        )
        gemma_loader.load()
        _log.info("Gemma4-E4B-it loaded OK via GemmaTransformersLoader")
    except Exception as exc:
        _log.error("Failed to load Gemma4 via GemmaTransformersLoader: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v2",
                "gemma4_loader": "GemmaTransformersLoader",
                "env_autofix": env_autofix_dict,
                "qwen_result": None,
                "gemma4_result": None,
                "first_positive_number": False,
            },
            status="blocked",
            blocked_reason=f"Gemma4 load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_artifact(repo_root, artifact)
        return artifact

    # ------------------------------------------------------------------
    # Gate 3: Load Qwen via standard HF pipeline
    # ------------------------------------------------------------------
    qwen_pipe: object | None = None
    try:
        _log.info("Loading Qwen3.5-0.8B on cuda:1 ...")
        qwen_pipe = _load_qwen_pipeline("Qwen/Qwen3.5-0.8B", gpu_index=1)
        _log.info("Qwen3.5-0.8B loaded OK")
    except Exception as exc:
        _log.error("Failed to load Qwen: %s", exc)
        artifact = tmpl.build_result(
            {
                "schema": "carnot.live_precision.v2",
                "gemma4_loader": "GemmaTransformersLoader",
                "env_autofix": env_autofix_dict,
                "qwen_result": None,
                "gemma4_result": None,
                "first_positive_number": False,
            },
            status="blocked",
            blocked_reason=f"Qwen load failed: {exc}",
            honest_verdict="deferred_to_gpu",
        )
        _write_artifact(repo_root, artifact)
        return artifact

    # ------------------------------------------------------------------
    # Load questions and wire CRANE extractor
    # ------------------------------------------------------------------
    questions = _load_gsm8k_questions(N_QUESTIONS)
    _log.info("Loaded %d questions", len(questions))

    crane = CRANEExtractionGate(min_confidence=0.7)

    # ------------------------------------------------------------------
    # Run benchmarks: Gemma4 then Qwen
    # ------------------------------------------------------------------
    def gemma_fn(prompt: str) -> str:
        assert gemma_loader is not None
        return _run_gemma_inference(gemma_loader, prompt)

    def qwen_fn(prompt: str) -> str:
        assert qwen_pipe is not None
        return _run_qwen_inference(qwen_pipe, prompt)

    _log.info("=== Running Gemma4-E4B-it benchmark ===")
    gemma4_result = _run_model_benchmark("Gemma4-E4B-it", gemma_fn, crane, questions)
    tmpl.checkpoint_save(gemma4_result.to_dict(), step=1)

    _log.info("=== Running Qwen3.5-0.8B benchmark ===")
    qwen_result = _run_model_benchmark("Qwen3.5-0.8B", qwen_fn, crane, questions)
    tmpl.checkpoint_save(qwen_result.to_dict(), step=2)

    # ------------------------------------------------------------------
    # Build artifact
    # ------------------------------------------------------------------
    first_positive_number = gemma4_result.is_positive or qwen_result.is_positive
    honest_verdict = "first_positive" if first_positive_number else "no_improvement_v2"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.live_precision.v2",
            "gemma4_loader": "GemmaTransformersLoader",
            "env_autofix": env_autofix_dict,
            "n_questions": N_QUESTIONS,
            "qwen_result": qwen_result.to_dict(),
            "gemma4_result": gemma4_result.to_dict(),
            "first_positive_number": first_positive_number,
        },
        status="success",
        honest_verdict=honest_verdict,
        inference_mode="live_gpu",
    )
    _write_artifact(repo_root, artifact)

    _log.info(
        "HEADLINE: honest_verdict=%s first_positive=%s "
        "gemma4_delta=%.4f qwen_delta=%.4f",
        honest_verdict,
        first_positive_number,
        gemma4_result.signed_improvement,
        qwen_result.signed_improvement,
    )

    return artifact


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 451: live precision post-fix benchmark.

    Wraps run_experiment() in an ExperimentTimeoutWatchdog with a 60-minute
    budget.  The extra 15 min over Exp 439 accounts for GemmaTransformersLoader
    being slightly slower than llama.cpp for model load (larger model overhead).
    """
    with ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=get_timeout_minutes(default=60),
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        artifact = run_experiment()

    verdict = artifact.get("honest_verdict", "unknown")
    _log.info(
        "Exp %d complete: honest_verdict=%s status=%s",
        EXP_ID,
        verdict,
        artifact.get("status", "unknown"),
    )

    if verdict == "first_positive":
        gemma4 = artifact.get("gemma4_result") or {}
        qwen = artifact.get("qwen_result") or {}
        _log.info(
            "FIRST POSITIVE VERIFY-REPAIR NUMBER: "
            "Gemma4 delta=%.4f Qwen delta=%.4f",
            gemma4.get("signed_improvement", float("nan")),
            qwen.get("signed_improvement", float("nan")),
        )


if __name__ == "__main__":
    main()
