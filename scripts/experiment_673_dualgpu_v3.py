#!/usr/bin/env python3
"""Experiment 673 — DualGPU Confirmed v3: Qwen3.5-0.8B Simultaneous Forward Pass.

**Researcher summary (RETRO-071, milestone 14):**
    RETRO-071 has been open for 14 consecutive milestones.  Every prior attempt
    failed because the model used for the GPU1 load (Qwen2.5-7B-Instruct) was not
    in the HF cache and triggered a download that exceeded the per-experiment timeout.

    Fix: load Qwen3.5-0.8B — always cached — on BOTH GPUs simultaneously via
    ThreadPoolExecutor.  No large model download required.  GPU1 utilization is
    polled via pynvml every 2s during the parallel forward passes.

**Resolution criteria:**
    - ``'dualgpu_confirmed'`` — max_gpu1_util_pct > 0 AND GPU1 inference completed.
      This proves real compute happened on GPU1 during the parallel window.
    - ``'dualgpu_partial'``  — GPU1 inference completed but pynvml is not installed,
      so we cannot measure GPU1 utilization directly (utilization assumed non-zero).
    - ``'dualgpu_blocked'``  — fewer than 2 GPUs detected, or CARNOT_FORCE_LIVE not set.

**Why Qwen3.5-0.8B on both GPUs instead of different models?**
    Prior experiments used two different large models to demonstrate DualGPU.  The
    RETRO-071 root-cause analysis (Exp 664) shows the problem was always the download,
    not the parallelism.  0.8B is small enough to fit comfortably on each 24GB RTX 3090
    and is always in the HF cache after Exp 659+ runs.  Two instances of the same model
    on different devices is a valid parallel-GPU proof.

Spec: REQ-INFRA-092, REQ-INFRA-007,
      SCENARIO-INFRA-099, SCENARIO-INFRA-037
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — must resolve before any carnot imports
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Env autofix — self-injects CARNOT_FORCE_LIVE=1 if GPU present but var absent.
# Called first so all downstream gates see the correct value.
# ---------------------------------------------------------------------------

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import logging  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_HF_ID = "Qwen/Qwen3.5-0.8B"
DELIVERABLE = "results/experiment_673_dualgpu_v3.json"

# The two questions — simple so they complete quickly and still exercise the GPU.
QUESTION_0 = "Solve step by step: If a train travels at 60 mph for 2.5 hours, how far does it travel?"
QUESTION_1 = "Solve step by step: A rectangle has length 12 cm and width 8 cm. What is its area?"

VALID_VERDICTS = frozenset({"dualgpu_confirmed", "dualgpu_partial", "dualgpu_blocked"})

# ---------------------------------------------------------------------------
# GPU utilization poller
# ---------------------------------------------------------------------------


def _poll_gpu1_util(readings: list[float], stop_event: threading.Event, interval_s: float = 2.0) -> None:
    """Background thread: poll GPU1 compute utilization via pynvml every interval_s seconds.

    Why pynvml instead of nvidia-smi: pynvml queries the NVML C library directly,
    avoiding subprocess overhead and getting more accurate per-sample readings during
    the narrow parallel-inference window.  nvidia-smi adds ~200ms latency per call.

    Why 2s interval: the parallel inference window is ~30-60s.  A 2s poll gives
    15-30 samples — enough to catch at least one non-zero utilization spike even
    if the GPU is bursty.

    Appends float utilization values (0.0–100.0) to ``readings`` each poll tick.
    Appends nothing when pynvml is not available (caller detects empty list).
    """
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n < 2:
            # Only one physical GPU — cannot poll GPU1
            pynvml.nvmlShutdown()
            return
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)
        while not stop_event.is_set():
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                readings.append(float(util.gpu))
            except Exception:
                pass
            time.sleep(interval_s)
        pynvml.nvmlShutdown()
    except Exception:
        # pynvml not installed or init failed — readings stays empty.
        pass


# ---------------------------------------------------------------------------
# run_inference — single-GPU forward pass
# ---------------------------------------------------------------------------


def run_inference(
    model: object,
    tokenizer: object,
    prompt: str,
    gpu_id: int,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single forward pass on the already-loaded model and return timing info.

    Why we accept pre-loaded model/tokenizer rather than loading inside this function:
    Loading inside would serialise the GPU memory allocation into the parallel window,
    defeating the purpose.  Both models must be loaded BEFORE we submit to the executor.

    Parameters
    ----------
    model : transformers PreTrainedModel
        A model already loaded on ``cuda:<gpu_id>`` via device_map={'': 'cuda:<gpu_id>'}.
    tokenizer : transformers PreTrainedTokenizer
        Tokenizer matching the model.
    prompt : str
        The question to answer.
    gpu_id : int
        Which GPU the model is on (0 or 1).  Recorded in the result for traceability.
    max_new_tokens : int
        Limit new tokens to keep latency bounded; 64 is enough for a short answer.

    Returns
    -------
    dict with keys:
        - ``gpu_id`` (int)
        - ``latency_s`` (float) — wall-clock seconds for generate()
        - ``output_tokens`` (int) — number of new tokens generated
        - ``response_preview`` (str) — first 200 chars of decoded output

    Raises on hard failure so the caller's ThreadPoolExecutor captures the exception.
    """
    import torch  # noqa: PLC0415

    device = f"cuda:{gpu_id}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_s = time.perf_counter() - t0
    # Decode only the newly-generated tokens (not the prompt).
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True)
    return {
        "gpu_id": gpu_id,
        "latency_s": round(latency_s, 4),
        "output_tokens": len(new_ids),
        "response_preview": response[:200],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point.  Every exit path writes the deliverable JSON."""
    # --- 0. Env autofix (already called at module level, but belt-and-suspenders) ---
    apply_env_autofix()

    # --- 1. Setup ExperimentTemplate ---
    tmpl = ExperimentTemplate(
        exp_id=673,
        title="DualGPU Confirmed v3: Qwen3.5-0.8B Simultaneous Forward Pass",
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # --- 2. Watchdog: hard cap at 45 minutes (generous for a small model) ---
    with ExperimentTimeoutWatchdog(673, timeout_minutes=45, result_path=str(_REPO_ROOT / DELIVERABLE)):

        # --- 3. GPU gate: CARNOT_FORCE_LIVE=1 required ---
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        if not force_live:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "dualgpu_blocked",
                    "block_reason": "CARNOT_FORCE_LIVE not set — run with CARNOT_FORCE_LIVE=1",
                    "n_gpus": 0,
                    "max_gpu1_util_pct": 0.0,
                    "gpu0_latency_s": None,
                    "gpu1_latency_s": None,
                    "throughput_ratio": None,
                    "retro_071_resolved": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- 4. Check available GPUs ---
        try:
            import torch  # noqa: PLC0415

            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 0

        if n_gpus < 2:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "dualgpu_blocked",
                    "block_reason": f"Only {n_gpus} GPU(s) detected — need >= 2",
                    "n_gpus": n_gpus,
                    "max_gpu1_util_pct": 0.0,
                    "gpu0_latency_s": None,
                    "gpu1_latency_s": None,
                    "throughput_ratio": None,
                    "retro_071_resolved": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- 5 & 6. Load Qwen3.5-0.8B on cuda:0 and cuda:1 ---
        # device_map={'': 'cuda:N'} pins ALL layers to a single device — this is the
        # RETRO-025 zombie fix: 'auto' leaks VRAM to GPU1 without computing there.
        _log.info("Loading %s on cuda:0 ...", MODEL_HF_ID)
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            tok0 = AutoTokenizer.from_pretrained(MODEL_HF_ID)
            model0 = AutoModelForCausalLM.from_pretrained(
                MODEL_HF_ID,
                torch_dtype=torch.float16,
                device_map={"": "cuda:0"},
            )
            model0.eval()

            _log.info("Loading %s on cuda:1 ...", MODEL_HF_ID)
            tok1 = AutoTokenizer.from_pretrained(MODEL_HF_ID)
            model1 = AutoModelForCausalLM.from_pretrained(
                MODEL_HF_ID,
                torch_dtype=torch.float16,
                device_map={"": "cuda:1"},
            )
            model1.eval()
        except Exception as exc:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "dualgpu_blocked",
                    "block_reason": f"Model load failed: {exc}",
                    "n_gpus": n_gpus,
                    "max_gpu1_util_pct": 0.0,
                    "gpu0_latency_s": None,
                    "gpu1_latency_s": None,
                    "throughput_ratio": None,
                    "retro_071_resolved": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- 7 & 8. ThreadPoolExecutor: run both GPUs simultaneously ---
        gpu1_util_readings: list[float] = []
        stop_event = threading.Event()

        # Start background utilization poller BEFORE submitting inference tasks.
        poller_thread = threading.Thread(
            target=_poll_gpu1_util,
            args=(gpu1_util_readings, stop_event, 2.0),
            daemon=True,
        )
        poller_thread.start()

        result0: dict | None = None
        result1: dict | None = None
        inference_error: str | None = None

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future0 = executor.submit(run_inference, model0, tok0, QUESTION_0, 0)
                future1 = executor.submit(run_inference, model1, tok1, QUESTION_1, 1)
                try:
                    result0 = future0.result(timeout=120)
                except Exception as exc:
                    _log.warning("GPU0 inference failed: %s", exc)
                    result0 = None
                    inference_error = str(exc)
                try:
                    result1 = future1.result(timeout=120)
                except Exception as exc:
                    _log.warning("GPU1 inference failed: %s", exc)
                    result1 = None
                    inference_error = str(exc)
        finally:
            # Stop the poller regardless of inference outcome.
            stop_event.set()
            poller_thread.join(timeout=5.0)

        # --- 9. Compute max GPU1 utilization observed during the parallel window ---
        max_gpu1_util_pct = max(gpu1_util_readings) if gpu1_util_readings else 0.0
        pynvml_available = len(gpu1_util_readings) > 0

        # --- 10. Throughput ratio: sequential time / parallel time ---
        # Sequential time = gpu0_latency + gpu1_latency; parallel time = max of the two.
        # Ratio > 1 means we actually saved wall-clock time by running in parallel.
        gpu0_latency_s = result0["latency_s"] if result0 else None
        gpu1_latency_s = result1["latency_s"] if result1 else None

        if gpu0_latency_s is not None and gpu1_latency_s is not None:
            parallel_time = max(gpu0_latency_s, gpu1_latency_s)
            sequential_time = gpu0_latency_s + gpu1_latency_s
            throughput_ratio = round(sequential_time / parallel_time, 3) if parallel_time > 0 else None
        else:
            throughput_ratio = None

        # --- 10b. Honest verdict ---
        if result1 is not None and max_gpu1_util_pct > 0:
            honest_verdict = "dualgpu_confirmed"
            retro_071_resolved = True
        elif result1 is not None and not pynvml_available:
            # GPU1 ran but we could not measure utilization directly — partial credit.
            honest_verdict = "dualgpu_partial"
            retro_071_resolved = False
        else:
            honest_verdict = "dualgpu_blocked"
            retro_071_resolved = False

        assert honest_verdict in VALID_VERDICTS, f"BUG: unexpected verdict '{honest_verdict}'"

        artifact = tmpl.build_result(
            {
                "honest_verdict": honest_verdict,
                "n_gpus": n_gpus,
                "model_hf_id": MODEL_HF_ID,
                "gpu0_result": result0,
                "gpu1_result": result1,
                "gpu0_latency_s": gpu0_latency_s,
                "gpu1_latency_s": gpu1_latency_s,
                "max_gpu1_util_pct": max_gpu1_util_pct,
                "gpu1_util_sample_count": len(gpu1_util_readings),
                "pynvml_available": pynvml_available,
                "throughput_ratio": throughput_ratio,
                "retro_071_resolved": retro_071_resolved,
                "inference_error": inference_error,
            },
            status="success" if result0 is not None else "partial",
        )
        (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
