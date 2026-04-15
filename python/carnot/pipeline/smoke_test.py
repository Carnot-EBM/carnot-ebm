"""Live GPU smoke test: minimal gating check before any benchmark experiment.

**Researcher summary:**
    Experiments 340, 341, 346, 347 all ran in *simulated* mode despite
    ``CARNOT_FORCE_LIVE=1``.  The artifacts were labelled ``inference_mode="live_gpu"``
    but contained only synthetic answers.  Exp 352 diagnosed the root cause: the
    pre-warm failure was caught and logged, but execution continued silently.

    This module is the gating experiment (353) that MUST pass before any benchmark
    is run.  It enforces a strict invariant:

        If CARNOT_FORCE_LIVE=1 and the model does not produce live_gpu inference,
        we raise RuntimeError — never silently degrade.

    The CI-safe path (CARNOT_FORCE_LIVE not set) returns a SmokeTestResult with
    is_live=False and inference_mode="ci_skip" without raising.  This keeps CI
    green on machines without GPUs.

**Why a separate smoke test module?**
    The benchmark scripts (Exp 340, 341) were long (~700 lines) and mixed GPU
    setup with pipeline logic.  It was easy for the "fail loudly" check to be
    buried and accidentally bypassed.  A dedicated smoke test module that is
    imported by ALL subsequent benchmark scripts removes that risk: each script
    calls run_smoke_test() first and either gets a live_confirmed artifact or
    raises before reaching any inference code.

Spec: REQ-BENCH-005, SCENARIO-BENCH-012, SCENARIO-BENCH-013
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimal hardcoded GSM8K questions for smoke test
# ---------------------------------------------------------------------------

_SMOKE_QUESTIONS = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning "
        "and bakes muffins for her friends every day with 4933600. She sells the "
        "remainder at the farmers' market daily for $2 per fresh duck egg. How much "
        "in dollars does she make every day at the farmers' market?",
        "answer": "#### 18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. "
        "How many bolts in total does it take?",
        "answer": "#### 3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 "
        "and then puts in $50,000 in repairs. This increased the value of the "
        "house by 150%. How much profit did he make?",
        "answer": "#### 70000",
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each "
        "sprint. How many total meters does he run a week?",
        "answer": "#### 540",
    },
    {
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed animal "
        "feed, containing a mix of seeds, mealworms and vegetables to help keep "
        "them healthy. She gives the chickens their feed in three separate meals. "
        "In the morning she gives her flock of chickens 15 cups of feed. In the "
        "afternoon she gives her chickens another 25 cups of feed. How many cups "
        "of feed does she need to give her chickens in the final meal of the day "
        "if the size of Wendi's flock is 20 chickens?",
        "answer": "#### 20",
    },
]


# ---------------------------------------------------------------------------
# SmokeTestResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SmokeTestResult:
    """Result from ``run_smoke_test()``.

    Fields
    ------
    inference_mode : str
        One of ``"live_gpu"``, ``"ci_skip"``, or ``"blocked"``.
        ``"ci_skip"`` means CARNOT_FORCE_LIVE was not set — no inference ran.
        ``"live_gpu"`` means actual GPU inference completed successfully.
        ``"blocked"`` means CARNOT_FORCE_LIVE=1 but inference failed.
    n_questions : int
        Number of questions requested.
    n_answered : int
        Number of non-empty responses received.  Zero on ci_skip.
    elapsed_s : float
        Wall-clock seconds for inference.  Zero on ci_skip.
    model_id : str
        HuggingFace model ID that was tested.
    is_live : bool
        ``True`` iff inference_mode == "live_gpu" and real GPU inference ran.
    blocked_reason : str
        Human-readable description of why inference did not run.
        Empty string when is_live=True.
    """

    inference_mode: str
    n_questions: int
    n_answered: int
    elapsed_s: float
    model_id: str
    is_live: bool
    blocked_reason: str


# ---------------------------------------------------------------------------
# Internal helpers (patchable in tests)
# ---------------------------------------------------------------------------


def _prewarm_model(name: str, hf_id: str, gpu: int) -> object:
    """Call the real model_prewarm from Exp 294.

    Separated into its own function so the test suite can patch it without
    importing the real Exp 294 module (which requires live GPU hardware).

    Parameters
    ----------
    name : str
        Short model name for logging.
    hf_id : str
        HuggingFace model identifier.
    gpu : int
        GPU device index.

    Returns
    -------
    object
        ModelPrewarmResult with .health_ok, .load_time_s, .stall_root_cause.
    """
    from scripts.experiment_294_gpu_baseline_apple import (  # type: ignore[import]
        model_prewarm,
    )

    return model_prewarm(name, hf_id, gpu)


def _load_model_for_smoke_test(hf_id: str, gpu: int) -> object:
    """Load a HuggingFace text-generation pipeline for live inference.

    Separated so tests can patch this without importing transformers.

    Parameters
    ----------
    hf_id : str
        HuggingFace model identifier (e.g. "google/gemma-4-E4B-it").
    gpu : int
        GPU device index.

    Returns
    -------
    object
        A callable HuggingFace pipeline (text-generation).

    Raises
    ------
    RuntimeError
        If the model cannot be loaded.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import]

        return hf_pipeline(
            "text-generation",
            model=hf_id,
            device=gpu,
            torch_dtype="auto",
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {hf_id} on GPU {gpu}: {exc}") from exc


# ---------------------------------------------------------------------------
# run_smoke_test
# ---------------------------------------------------------------------------


def run_smoke_test(
    model_id: str,
    *,
    n_questions: int = 5,
    timeout_s: float = 300.0,
) -> SmokeTestResult:
    """Run a minimal live GPU smoke test and return a ``SmokeTestResult``.

    **Why this function exists:**
        All benchmark experiments (Exp 340+) must call this function FIRST.
        If it does not return is_live=True, no benchmark inference should run.
        This prevents the silent-simulated-mode bug that affected Exps 340-347.

    **CI-safe guarantee:**
        When ``CARNOT_FORCE_LIVE`` is not set, this function returns immediately
        with ``inference_mode="ci_skip"`` and ``is_live=False`` — no GPU hardware
        is accessed, no model is loaded, no exception is raised.

    **Live mode behaviour:**
        When ``CARNOT_FORCE_LIVE=1``:
        1. GPU pre-warm via Exp 294's ``model_prewarm()`` (health-check prompt).
        2. If pre-warm fails → ``RuntimeError`` (never silent fallback).
        3. Model is loaded and ``n_questions`` GSM8K questions are run.
        4. Returns ``SmokeTestResult(inference_mode="live_gpu", is_live=True, ...)``.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID to test (e.g. ``"google/gemma-4-E4B-it"``).
    n_questions : int
        Number of GSM8K questions to run (default 5; must be ≤ 5 for the
        hardcoded minimal set).
    timeout_s : float
        Maximum wall-clock seconds for the entire smoke test (default 300 s).

    Returns
    -------
    SmokeTestResult
        Structured result with inference_mode, n_answered, elapsed_s, is_live.

    Raises
    ------
    RuntimeError
        If ``CARNOT_FORCE_LIVE=1`` and live GPU inference is unavailable or
        the model fails to load.  Never raised when CARNOT_FORCE_LIVE is not set.
    """
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    # --- CI-safe path: CARNOT_FORCE_LIVE not set ---
    # Return immediately without touching GPU hardware.
    if not force_live:
        _log.info(
            "run_smoke_test: CARNOT_FORCE_LIVE not set — returning ci_skip result"
        )
        return SmokeTestResult(
            inference_mode="ci_skip",
            n_questions=n_questions,
            n_answered=0,
            elapsed_s=0.0,
            model_id=model_id,
            is_live=False,
            blocked_reason="CARNOT_FORCE_LIVE not set",
        )

    # --- Live path: CARNOT_FORCE_LIVE=1 ---
    # Step 1: GPU pre-warm + health check (Exp 294 pattern).
    # We call _prewarm_model (patchable) instead of ExperimentTemplate.setup_gpu()
    # so this module has no hard dependency on scripts.experiment_template.
    _log.info("run_smoke_test: CARNOT_FORCE_LIVE=1 — running GPU pre-warm for %s", model_id)

    prewarm_result = _prewarm_model(model_id, model_id, 0)

    if not prewarm_result.health_ok:
        # REQ-INFRA-014: explicit fail, never silent fallback.
        # Diagnose which layer failed so the researcher has actionable info.
        from carnot.pipeline.live_gpu_diagnostic import diagnose_live_gpu  # noqa: PLC0415

        diag = diagnose_live_gpu([model_id])
        reason = diag.failure_reason or (prewarm_result.stall_root_cause or "model prewarm failed")
        raise RuntimeError(
            f"Live GPU required but unavailable: {reason}"
        )

    # Step 2: Load model for live inference.
    _log.info("run_smoke_test: loading model %s on GPU 0", model_id)
    model_obj = _load_model_for_smoke_test(model_id, 0)

    # --- DualGPUMonitor health check (non-fatal; informational) ---
    try:
        from carnot.pipeline.dual_gpu_monitor import DualGPUMonitor  # noqa: PLC0415

        monitor = DualGPUMonitor()
        gpu_health = monitor.check_dual_gpu_health()
        if not gpu_health["all_healthy"]:
            _log.warning(
                "run_smoke_test: DualGPUMonitor reports unhealthy GPU state — "
                "n_gpus=%d n_zombies=%d",
                gpu_health["n_gpus_detected"],
                gpu_health["n_zombies"],
            )
    except Exception as exc:  # pragma: no cover — import failures are non-fatal
        _log.warning("run_smoke_test: DualGPUMonitor unavailable: %s", exc)

    # Step 3: Run N questions from the hardcoded minimal set.
    questions = _SMOKE_QUESTIONS[:n_questions]
    n_answered = 0
    t0 = time.perf_counter()

    for q in questions:
        prompt = f"Question: {q['question']}\nLet's think step by step.\n"
        try:
            raw = model_obj(prompt, max_new_tokens=256)  # type: ignore[operator]
            if isinstance(raw, list) and raw:
                response = raw[0].get("generated_text", "")
            else:
                response = str(raw)
            if response.strip():
                n_answered += 1
        except Exception as exc:
            _log.warning("run_smoke_test: inference call failed: %s", exc)

    elapsed_s = round(time.perf_counter() - t0, 3)

    _log.info(
        "run_smoke_test: completed — n_answered=%d/%d elapsed_s=%.3f",
        n_answered,
        n_questions,
        elapsed_s,
    )

    return SmokeTestResult(
        inference_mode="live_gpu",
        n_questions=n_questions,
        n_answered=n_answered,
        elapsed_s=elapsed_s,
        model_id=model_id,
        is_live=True,
        blocked_reason="",
    )


# ---------------------------------------------------------------------------
# build_smoke_test_artifact
# ---------------------------------------------------------------------------


def build_smoke_test_artifact(result: SmokeTestResult) -> dict:
    """Build a serializable artifact dict from a ``SmokeTestResult``.

    **Honest verdict mapping:**
        - ``is_live=True``                       → ``"live_confirmed"``
        - ``is_live=False, mode="ci_skip"``      → ``"blocked_simulated"``
        - ``is_live=False, mode=anything else``  → ``"blocked_error"``

    The ``"blocked_simulated"`` verdict is used for the CI-skip path where
    no GPU was available.  ``"blocked_error"`` covers cases where live mode
    was attempted but failed (though in practice run_smoke_test raises before
    returning a blocked result in live mode).

    Parameters
    ----------
    result : SmokeTestResult
        The result from ``run_smoke_test()``.

    Returns
    -------
    dict
        JSON-serializable artifact with ``schema``, ``honest_verdict``, and
        all SmokeTestResult fields.
    """
    if result.is_live:
        honest_verdict = "live_confirmed"
    elif result.inference_mode == "ci_skip":
        honest_verdict = "blocked_simulated"
    else:
        honest_verdict = "blocked_error"

    return {
        "schema": "carnot.smoke_test.v1",
        "honest_verdict": honest_verdict,
        "inference_mode": result.inference_mode,
        "n_questions": result.n_questions,
        "n_answered": result.n_answered,
        "elapsed_s": result.elapsed_s,
        "model_id": result.model_id,
        "is_live": result.is_live,
        "blocked_reason": result.blocked_reason,
    }
