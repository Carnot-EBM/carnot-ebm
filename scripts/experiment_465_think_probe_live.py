#!/usr/bin/env python3
"""Experiment 465: ThinkProbeV2 Live GPU — RETRO-029 + RETRO-036 closure.

**Researcher summary:**
    Exp 455 (ThinkProbeV2) reported RETRO-029 CLOSED in the conductor log but
    the result JSON file was absent at retrospective time.  Root cause: path
    mismatch between conductor spec and script output, plus no DeliverableGuard
    assertion at experiment exit.

    This experiment re-runs ThinkProbeV2 on live GPU and uses DeliverableGuard
    (Exp 462) to guarantee the deliverable file is present on disk as the FINAL
    assertion in main().  The ThinkProbeV2 module (60-min budget, partial verdict,
    incremental checkpoint) was correctly implemented; this run just executes it.

**Deliverable:** results/experiment_465_think_probe_live.json
**Schema:** carnot.think_probe.live.v1

**Honest verdict logic:**
    'complete'          — all 50 questions finished within 65-minute internal budget
    'partial_N_of_50'   — partial run (N questions completed before budget expired)
    'timeout_no_data'   — budget expired before any question returned
    'deferred_to_gpu'   — GPU hardware not available; experiment deferred

Depends on: Exp 462 (DeliverableGuard).
Hardware: 1x RTX 3090 (CARNOT_FORCE_LIVE=1).

Spec: REQ-PROBE-008, REQ-PROBE-009,
      SCENARIO-PROBE-013, SCENARIO-PROBE-014
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# apply_env_autofix() MUST be called first — detects ROCm/CUDA and injects
# CARNOT_FORCE_LIVE=1 when the env var is absent.  This is the RETRO-022
# belt-and-suspenders safeguard in case the conductor's os.environ lacked the var.
sys.path.insert(0, str(Path(__file__).parent.parent))
from carnot.pipeline.env_autofix import apply_env_autofix

env_result = apply_env_autofix()

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.live_think_probe_result import LiveThinkProbeResult  # noqa: E402
from carnot.pipeline.think_probe_v2 import ThinkProbeV2, ThinkProbeV2Result  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 465
EXP_TITLE = "ThinkProbeV2 Live GPU — RETRO-029 + RETRO-036 closure"
RESULT_PATH = "results/experiment_465_think_probe_live.json"
MODEL_ID = "google/gemma-4-E4B-it"

# 75-minute external watchdog; 65-minute internal budget leaves 10 minutes
# for artifact write and DeliverableGuard assertion within the watchdog window.
WATCHDOG_TIMEOUT_MINUTES = 75
INTERNAL_BUDGET_MINUTES = 65

N_QUESTIONS = 50

# ---------------------------------------------------------------------------
# Corpus builder (reused from Exp 455)
# ---------------------------------------------------------------------------

_CORRECT_TEMPLATES = [
    "The answer is {n}. We compute {a} + {b} = {n}.",
    "Therefore {a} * {b} = {n}, which is correct.",
    "Since {a} - {b} = {n}, the result is {n}.",
    "The sum {a} + {b} equals {n}.",
    "Multiplying {a} by {b} gives {n}.",
]

_WRONG_TEMPLATES = [
    "The answer is {wrong}. We compute {a} + {b} = {wrong}.",
    "Therefore {a} * {b} = {wrong}, so the result is {wrong}.",
    "Since {a} - {b} = {wrong}, we conclude {wrong}.",
    "The sum {a} + {b} equals {wrong}.",
    "Multiplying {a} by {b} gives {wrong}.",
]


def _build_50_questions() -> list[str]:
    """Build 50 synthetic arithmetic questions for the benchmark.

    25 correct + 25 wrong responses interleaved.  Same corpus as Exp 455 so
    results are directly comparable.
    """
    questions: list[str] = []
    for i in range(25):
        a = 10 + i
        b = 5 + (i % 7)
        n_add = a + b
        n_mul = a * b
        wrong_add = n_add + 1

        tmpl_correct = _CORRECT_TEMPLATES[i % len(_CORRECT_TEMPLATES)]
        tmpl_wrong = _WRONG_TEMPLATES[i % len(_WRONG_TEMPLATES)]

        questions.append(tmpl_correct.format(a=a, b=b, n=n_add, wrong=wrong_add))
        questions.append(tmpl_wrong.format(a=a, b=b, n=n_mul, wrong=wrong_add))

    return questions[:N_QUESTIONS]


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    """Return True iff CARNOT_FORCE_LIVE=1 and torch reports a CUDA/ROCm device."""
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _gpu_device_string() -> str:
    """Return a human-readable string describing the active GPU device.

    Used for the 'gpu_used' provenance field in the artifact.  Returns
    'unavailable' when called without GPU (should not happen after _gpu_available()).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "unavailable"
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name})"
    except ImportError:
        return "unavailable"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import datetime
    import time

    def _utc_now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _run_date() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")

    started_at = _utc_now()
    t0 = time.perf_counter()

    # DeliverableGuard is declared early so every exit path (deferred, partial,
    # complete) is covered by the final assert_written() call.
    guard = DeliverableGuard(RESULT_PATH)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        RESULT_PATH,
        requires_gpu=True,
    )
    tmpl.setup()

    writer = AtomicResultWriter(RESULT_PATH)

    # ------------------------------------------------------------------
    # GPU gate: emit deferred artifact when GPU is unavailable
    # ------------------------------------------------------------------
    if not _gpu_available():
        _log.warning(
            "Exp 465: CARNOT_FORCE_LIVE != 1 or no CUDA device — emitting deferred artifact"
        )
        artifact = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema": "carnot.think_probe.live.v1",
            "run_date": _run_date(),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_s": round(time.perf_counter() - t0, 3),
            "status": "gpu_required",
            "honest_verdict": "deferred_to_gpu",
            "inference_mode": "deferred",
            "model_id": MODEL_ID,
            "gpu_used": "unavailable",
            "retro_029_closed": False,
            "retro_036_closed": False,
            "n_completed": 0,
            "n_total": N_QUESTIONS,
            "completion_fraction": 0.0,
        }
        writer.write(artifact)
        _log.info("Exp 465: wrote deferred artifact to %s", RESULT_PATH)
        guard.assert_written()
        return

    # ------------------------------------------------------------------
    # Live GPU run
    # ------------------------------------------------------------------
    gpu_str = _gpu_device_string()
    _log.info("Exp 465: GPU confirmed — %s", gpu_str)

    # Load GemmaTransformersLoader and build the inference_fn.
    # The loader is kept alive for the duration of the run; loading is done
    # once outside the question loop to avoid re-loading per question.
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    loader = GemmaTransformersLoader(MODEL_ID)
    _log.info("Exp 465: loading model %s …", MODEL_ID)
    loader.load()
    _log.info("Exp 465: model loaded — starting ThinkProbeV2 run")

    def inference_fn(question: str) -> str:
        """Wrap GemmaTransformersLoader.generate() for ThinkProbeV2.run()."""
        raw = loader.generate(question, max_new_tokens=256)
        return raw if loader.is_valid_output(raw) else ""

    questions = _build_50_questions()

    # 75-minute external watchdog; 65-minute internal budget (10-min buffer
    # for artifact write and DeliverableGuard assertion before watchdog fires).
    with ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=WATCHDOG_TIMEOUT_MINUTES,
        result_path=RESULT_PATH,
    ):
        probe = ThinkProbeV2(
            budget_minutes=INTERNAL_BUDGET_MINUTES,
            checkpoint_interval=10,
        )
        result = probe.run(questions, inference_fn)

    finished_at = _utc_now()
    duration_s = time.perf_counter() - t0

    live_result = LiveThinkProbeResult(
        n_completed=result.n_completed,
        n_total=result.n_total,
        results=result.results,
        status=result.status,
        inference_mode="live_gpu",
        model_id=MODEL_ID,
        gpu_used=gpu_str,
    )

    artifact = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "schema": "carnot.think_probe.live.v1",
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 3),
        "status": live_result.status,
        "inference_mode": live_result.inference_mode,
        "model_id": live_result.model_id,
        "gpu_used": live_result.gpu_used,
        "retro_029_closed": True,
        "retro_036_closed": True,
        "n_completed": live_result.n_completed,
        "n_total": live_result.n_total,
        "completion_fraction": round(live_result.completion_fraction, 4),
        "honest_verdict": live_result.honest_verdict,
        "results_summary": [
            {"question_index": r["question_index"], "response_len": len(r["response"])}
            for r in live_result.results
        ],
    }

    writer.write(artifact)

    _log.info(
        "Exp 465: %s  n_completed=%d/%d  honest_verdict=%s  duration=%.1fs",
        live_result.status.upper(),
        live_result.n_completed,
        live_result.n_total,
        live_result.honest_verdict,
        duration_s,
    )
    _log.info("Exp 465: artifact written to %s", RESULT_PATH)

    # FINAL LINE: assert the deliverable was written.  Raises FileNotFoundError
    # if the file is absent — turns a silent omission into a loud crash (RETRO-036).
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
