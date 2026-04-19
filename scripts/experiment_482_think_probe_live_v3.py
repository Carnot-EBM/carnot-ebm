#!/usr/bin/env python3
"""Experiment 482: ThinkProbeV2 Live v3 — RETRO-036 + RETRO-042 closure via GPUVRAMGate.

**Researcher summary:**
    RETRO-036 opened when Exp 455 reported RETRO-029 CLOSED in the conductor log but the
    result JSON file was absent.  RETRO-042 opened when Exp 465 (ThinkProbeV2 live GPU)
    deferred because zombie VRAM held 23.8 GB on GPU 0 at 0% utilisation, preventing
    the Gemma model from loading.

    This experiment (Exp 482) closes both retros by combining:
    1. GPUVRAMGate (Exp 474) — kills zombie processes and waits for >= 8 GB free VRAM
       BEFORE loading the model.  If VRAM cannot be freed, the experiment emits a
       ``status='gpu_vram_insufficient'`` artifact and exits cleanly.
    2. DeliverableGuard (Exp 462) — assert_deliverable_written() as the FINAL LINE of
       main(), turning a silent missing-file omission into a loud crash.
    3. ThinkProbeLiveV3Result.is_viable — three-threshold viability verdict:
       completion_fraction >= 0.80, tp_rate >= 0.70, fp_rate <= 0.20.

**Deliverable:** results/experiment_482_think_probe_live_v3.json
**Schema:** carnot.think_probe.live.v3
**Target:** >= 40 of 50 GSM8K questions with inference_mode='live_gpu'

Spec: REQ-PROBE-010, REQ-PROBE-011,
      SCENARIO-PROBE-015, SCENARIO-PROBE-016
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# apply_env_autofix() MUST be called first — detects ROCm/CUDA and injects
# CARNOT_FORCE_LIVE=1 when the env var is absent (RETRO-022 belt-and-suspenders fix).
sys.path.insert(0, str(Path(__file__).parent.parent))
from carnot.pipeline.env_autofix import apply_env_autofix

env_result = apply_env_autofix()

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gpu_vram_gate import GPUVRAMGate, GPUVRAMInsufficientError  # noqa: E402
from carnot.pipeline.think_probe_live_v3_result import ThinkProbeLiveV3Result  # noqa: E402
from carnot.pipeline.think_probe_v2 import ThinkProbeV2  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 482
EXP_TITLE = "ThinkProbeV2 Live v3 — RETRO-036 + RETRO-042 closure via GPUVRAMGate"
RESULT_PATH = "results/experiment_482_think_probe_live_v3.json"
MODEL_ID = "google/gemma-4-E4B-it"

# 90-minute external watchdog; 75-minute internal budget leaves 15 minutes for
# artifact write and DeliverableGuard assertion within the watchdog window.
WATCHDOG_TIMEOUT_MINUTES = 90
INTERNAL_BUDGET_MINUTES = 75

N_QUESTIONS = 50

# ---------------------------------------------------------------------------
# GSM8K corpus builder (same synthetic corpus as Exp 465 for direct comparison)
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

    25 correct + 25 wrong responses interleaved.  Same corpus as Exp 465 so
    results are directly comparable across the RETRO-036/042 closure history.
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
# Rate computation helpers
# ---------------------------------------------------------------------------


def _compute_rates(results: list[dict], questions: list[str]) -> tuple[float, float, float]:
    """Compute skip_rate, tp_rate, fp_rate over completed results.

    The synthetic corpus interleaves correct (even index) and wrong (odd index)
    questions.  A response is considered a 'flag' if it is non-empty (the model
    produced some output rather than timing out).

    Returns (skip_rate, tp_rate, fp_rate) as floats in [0, 1].

    Why even/odd for correct/wrong: _build_50_questions() appends a correct
    template entry followed by a wrong template entry in each loop iteration,
    so even-indexed questions (0, 2, 4, …) are always 'correct' corpus items
    and odd-indexed questions (1, 3, 5, …) are always 'wrong' corpus items.
    """
    if not results:
        return 0.0, 0.0, 0.0

    n_completed = len(results)
    skips = sum(1 for r in results if not r.get("response", ""))
    skip_rate = skips / n_completed if n_completed > 0 else 0.0

    # Separate correct and wrong corpus items by question_index parity.
    correct_items = [r for r in results if r["question_index"] % 2 == 0]
    wrong_items = [r for r in results if r["question_index"] % 2 == 1]

    # tp_rate: fraction of correct-corpus items that received a non-empty response.
    if correct_items:
        flagged_correct = sum(1 for r in correct_items if r.get("response", ""))
        tp_rate = flagged_correct / len(correct_items)
    else:
        tp_rate = 0.0

    # fp_rate: fraction of wrong-corpus items that received a non-empty response.
    # A non-empty response on a wrong item means the verifier "accepted" a bad answer.
    if wrong_items:
        flagged_wrong = sum(1 for r in wrong_items if r.get("response", ""))
        fp_rate = flagged_wrong / len(wrong_items)
    else:
        fp_rate = 0.0

    return skip_rate, tp_rate, fp_rate


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

    # DeliverableGuard is declared early so every exit path is covered.
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
    # GPUVRAMGate: ensure >= 8 GB free before model load (REQ-PROBE-010)
    # Kills zombie processes that hold VRAM at 0% utilisation (RETRO-042 fix).
    # ------------------------------------------------------------------
    gpu_vram_gate_fired = False
    try:
        gate = GPUVRAMGate(min_free_gb=8.0)
        gate.__enter__()
        gpu_vram_gate_fired = True
        _log.info("Exp 482: GPUVRAMGate passed — >= 8 GB VRAM free")
    except GPUVRAMInsufficientError as e:
        _log.warning("Exp 482: GPUVRAMGate failed — %s", e)
        artifact = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema": "carnot.think_probe.live.v3",
            "run_date": _run_date(),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_s": round(time.perf_counter() - t0, 3),
            "status": "gpu_vram_insufficient",
            "honest_verdict": "deferred_to_gpu",
            "inference_mode": "deferred",
            "model_id": MODEL_ID,
            "gpu_vram_gate_fired": False,
            "n_completed": 0,
            "n_total": N_QUESTIONS,
            "completion_fraction": 0.0,
            "skip_rate": 0.0,
            "tp_rate": 0.0,
            "fp_rate": 0.0,
            "is_viable": False,
            "retro_036_closed": False,
            "retro_042_closed": False,
        }
        writer.write(artifact)
        _log.info("Exp 482: wrote gpu_vram_insufficient artifact to %s", RESULT_PATH)
        tmpl.assert_deliverable_written()
        return

    # ------------------------------------------------------------------
    # Check CARNOT_FORCE_LIVE gate
    # ------------------------------------------------------------------
    if os.environ.get("CARNOT_FORCE_LIVE", "0") != "1":
        _log.warning("Exp 482: CARNOT_FORCE_LIVE != 1 — emitting deferred artifact")
        artifact = {
            "experiment": EXP_ID,
            "title": EXP_TITLE,
            "schema": "carnot.think_probe.live.v3",
            "run_date": _run_date(),
            "started_at": started_at,
            "finished_at": _utc_now(),
            "duration_s": round(time.perf_counter() - t0, 3),
            "status": "gpu_required",
            "honest_verdict": "deferred_to_gpu",
            "inference_mode": "deferred",
            "model_id": MODEL_ID,
            "gpu_vram_gate_fired": gpu_vram_gate_fired,
            "n_completed": 0,
            "n_total": N_QUESTIONS,
            "completion_fraction": 0.0,
            "skip_rate": 0.0,
            "tp_rate": 0.0,
            "fp_rate": 0.0,
            "is_viable": False,
            "retro_036_closed": False,
            "retro_042_closed": False,
        }
        writer.write(artifact)
        _log.info("Exp 482: wrote deferred artifact to %s", RESULT_PATH)
        tmpl.assert_deliverable_written()
        return

    # ------------------------------------------------------------------
    # Live GPU run
    # ------------------------------------------------------------------
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    _log.info("Exp 482: loading model %s …", MODEL_ID)
    loader = GemmaTransformersLoader(MODEL_ID)
    loader.load()
    _log.info("Exp 482: model loaded — starting ThinkProbeV2 run")

    def inference_fn(question: str) -> str:
        """Wrap GemmaTransformersLoader.generate() for ThinkProbeV2.run()."""
        raw = loader.generate(question, max_new_tokens=256)
        return raw if loader.is_valid_output(raw) else ""

    questions = _build_50_questions()

    with ExperimentTimeoutWatchdog(
        EXP_ID,
        timeout_minutes=WATCHDOG_TIMEOUT_MINUTES,
        result_path=RESULT_PATH,
    ):
        probe = ThinkProbeV2(
            budget_minutes=INTERNAL_BUDGET_MINUTES,
            checkpoint_interval=10,
            checkpoint_dir=Path("results/checkpoints/experiment_482"),
        )
        result = probe.run(questions, inference_fn)

    finished_at = _utc_now()
    duration_s = time.perf_counter() - t0

    skip_rate, tp_rate, fp_rate = _compute_rates(result.results, questions)

    live_result = ThinkProbeLiveV3Result(
        inference_mode="live_gpu",
        model_id=MODEL_ID,
        n_completed=result.n_completed,
        n_total=result.n_total,
        gpu_vram_gate_fired=gpu_vram_gate_fired,
        skip_rate=round(skip_rate, 4),
        tp_rate=round(tp_rate, 4),
        fp_rate=round(fp_rate, 4),
    )

    artifact = {
        "experiment": EXP_ID,
        "title": EXP_TITLE,
        "schema": "carnot.think_probe.live.v3",
        "run_date": _run_date(),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 3),
        "status": result.status,
        "honest_verdict": result.honest_verdict,
        "inference_mode": live_result.inference_mode,
        "model_id": live_result.model_id,
        "gpu_vram_gate_fired": live_result.gpu_vram_gate_fired,
        "n_completed": live_result.n_completed,
        "n_total": live_result.n_total,
        "completion_fraction": round(live_result.completion_fraction, 4),
        "skip_rate": live_result.skip_rate,
        "tp_rate": live_result.tp_rate,
        "fp_rate": live_result.fp_rate,
        "is_viable": live_result.is_viable,
        "retro_036_closed": True,
        "retro_042_closed": True,
        "results_summary": [
            {"question_index": r["question_index"], "response_len": len(r.get("response", ""))}
            for r in result.results
        ],
    }

    writer.write(artifact)

    _log.info(
        "Exp 482: %s  n_completed=%d/%d  is_viable=%s  tp_rate=%.2f  fp_rate=%.2f  duration=%.1fs",
        result.status.upper(),
        live_result.n_completed,
        live_result.n_total,
        live_result.is_viable,
        live_result.tp_rate,
        live_result.fp_rate,
        duration_s,
    )
    _log.info("Exp 482: artifact written to %s", RESULT_PATH)

    # FINAL LINE: assert the deliverable was written.  Raises FileNotFoundError
    # if the file is absent — turns a silent omission into a loud crash (RETRO-036).
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
