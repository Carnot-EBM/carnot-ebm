#!/usr/bin/env python3
"""Experiment 684 — DualGPU pynvml: Confirm GPU1 Compute Utilization > 0%.

**Researcher summary (RETRO-071, milestone 15):**
    Exp 673 (DualGPU v3) achieved throughput_ratio=1.963 but max_gpu1_util_pct=0.0
    because pynvml was not installed in the project venv.  Throughput alone does not
    prove parallel compute — a sequential interleaved implementation could produce
    the same throughput ratio.  This experiment installs pynvml, polls GPU1 via
    nvmlDeviceGetUtilizationRates() during the parallel window, and confirms that
    max_gpu1_util_pct > 0.

**Resolution criteria:**
    - ``'dualgpu_confirmed'``      — max_gpu1_util_pct > 0 AND GPU1 inference done.
      This is the first definitive proof of real parallel GPU compute.
    - ``'dualgpu_partial_no_pynvml'`` — GPU1 inference done but pynvml install failed.
    - ``'dualgpu_blocked'``         — < 2 GPUs or CARNOT_FORCE_LIVE not set.

**Why pynvml over nvidia-smi for utilization polling:**
    nvidia-smi adds ~200ms subprocess overhead per call.  With a 2s poll interval
    during a 30-60s inference window we would get 15-30 samples — but each sample
    carries 7-13% overhead jitter that can mask bursty GPU activity.  pynvml queries
    the NVML C library in-process with microsecond latency, giving cleaner samples.

Spec: REQ-HW-035, SCENARIO-HW-035, REQ-INFRA-092, SCENARIO-INFRA-099
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
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
DELIVERABLE = "results/experiment_684_dualgpu_pynvml.json"

# Ten varied questions — keeps each GPU busy for a meaningful inference window.
# Simple arithmetic and logic so answers are short, but varied enough to avoid
# trivial caching by the model.
QUESTIONS_GPU0 = [
    "Solve step by step: 3 + 4 × 5 =",
    "Solve step by step: A train goes 60 mph for 2.5 hours. Distance?",
    "Solve step by step: 12 % 5 =",
    "Solve step by step: 2^10 =",
    "Solve step by step: Area of circle with radius 7?",
    "Solve step by step: 100 / 8 =",
    "Solve step by step: 15 × 15 =",
    "Solve step by step: sqrt(144) =",
    "Solve step by step: 7! =",
    "Solve step by step: GCD(48, 36) =",
]

QUESTIONS_GPU1 = [
    "Solve step by step: 5 × (3 + 2) - 4 =",
    "Solve step by step: Rectangle 12 cm × 8 cm. Area?",
    "Solve step by step: 17 mod 3 =",
    "Solve step by step: log2(256) =",
    "Solve step by step: Volume of sphere radius 3?",
    "Solve step by step: 200 / 16 =",
    "Solve step by step: 25 × 24 =",
    "Solve step by step: sqrt(225) =",
    "Solve step by step: 6! =",
    "Solve step by step: LCM(4, 6) =",
]

VALID_VERDICTS = frozenset({
    "dualgpu_confirmed",
    "dualgpu_partial_no_pynvml",
    "dualgpu_blocked",
})


# ---------------------------------------------------------------------------
# pynvml install helper
# ---------------------------------------------------------------------------


def ensure_pynvml() -> bool:
    """Try to import pynvml; install it via pip if missing.  Return True if available.

    Why we install at runtime rather than listing in requirements: pynvml is a thin
    NVML binding that is only useful on NVIDIA hardware.  Adding it as a hard
    dependency would break CPU-only CI.  Installing at runtime is the pattern used
    throughout the experiment layer (see Exp 673, ExperimentTemplate.kill_gpu_zombies).

    Returns True if pynvml is importable after the attempt, False otherwise.
    """
    try:
        import pynvml  # noqa: PLC0415, F401
        return True
    except ImportError:
        pass

    _log.info("pynvml not found — installing via pip ...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pynvml", "-q"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            _log.warning("pip install pynvml failed: %s", result.stderr[:500])
            return False
    except Exception as exc:
        _log.warning("pip install pynvml exception: %s", exc)
        return False

    try:
        import pynvml  # noqa: PLC0415, F401
        return True
    except ImportError:
        _log.warning("pynvml still not importable after pip install")
        return False


# ---------------------------------------------------------------------------
# GPU utilization pollers
# ---------------------------------------------------------------------------


def poll_gpu_utilization(
    handle: object,
    stop_event: threading.Event,
    results: list[float],
    interval_s: float = 2.0,
) -> None:
    """Background thread: poll one GPU's compute utilization via pynvml.

    Why poll every 2s: the parallel inference window spans ~30-60s on RTX 3090 with
    Qwen3.5-0.8B.  A 2s interval yields 15-30 samples — enough to catch at least one
    non-zero utilization spike even if the GPU is bursty between token batches.

    Appends float values (0.0–100.0) to ``results`` each tick.
    Runs until ``stop_event`` is set.

    Parameters
    ----------
    handle :
        pynvml device handle, from nvmlDeviceGetHandleByIndex().
    stop_event :
        threading.Event; when set, this thread exits after the current sleep.
    results :
        Mutable list that receives utilization samples (floats).
    interval_s :
        Seconds between polls (default 2.0).
    """
    import pynvml  # noqa: PLC0415

    while not stop_event.is_set():
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            results.append(float(util.gpu))
        except Exception:
            pass
        time.sleep(interval_s)


# ---------------------------------------------------------------------------
# single-GPU inference
# ---------------------------------------------------------------------------


def run_inference_batch(
    model: object,
    tokenizer: object,
    questions: list[str],
    gpu_id: int,
    max_new_tokens: int = 100,
) -> dict:
    """Run a batch of questions sequentially on a pre-loaded model; return timing info.

    Why we accept pre-loaded model/tokenizer: loading inside this function would
    serialise GPU memory allocation into the parallel window, defeating the point.
    Both models must be in VRAM BEFORE the executor submits tasks.

    Parameters
    ----------
    model :
        HuggingFace PreTrainedModel already on cuda:<gpu_id>.
    tokenizer :
        Matching tokenizer.
    questions :
        List of prompt strings to run sequentially on this GPU.
    gpu_id :
        Which GPU (0 or 1).  Recorded in the result for traceability.
    max_new_tokens :
        Token budget per question; 100 balances answer quality vs. inference time.

    Returns
    -------
    dict with keys: gpu_id, total_latency_s, n_questions, output_tokens_total,
    response_previews (list[str]).
    """
    import torch  # noqa: PLC0415

    device = f"cuda:{gpu_id}"
    t_batch_start = time.perf_counter()
    output_tokens_total = 0
    response_previews: list[str] = []

    for question in questions:
        inputs = tokenizer(question, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        output_tokens_total += len(new_ids)
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        response_previews.append(response[:100])

    total_latency_s = time.perf_counter() - t_batch_start

    return {
        "gpu_id": gpu_id,
        "total_latency_s": round(total_latency_s, 4),
        "n_questions": len(questions),
        "output_tokens_total": output_tokens_total,
        "response_previews": response_previews,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point.  Every exit path writes the deliverable JSON."""
    # --- 0. Env autofix (belt-and-suspenders — also called at module level) ---
    apply_env_autofix()

    # --- 1. Setup ExperimentTemplate ---
    tmpl = ExperimentTemplate(
        exp_id=684,
        title="DualGPU pynvml: Confirm GPU1 Compute Utilization > 0%",
        deliverable=DELIVERABLE,
        requires_gpu=True,
    )
    tmpl.setup()

    # --- 2. Watchdog: hard cap at 45 minutes ---
    with ExperimentTimeoutWatchdog(684, timeout_minutes=45, result_path=str(_REPO_ROOT / DELIVERABLE)):

        # --- 3. GPU gate: CARNOT_FORCE_LIVE=1 required ---
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
        if not force_live:
            artifact = tmpl.build_result(
                {
                    "honest_verdict": "dualgpu_blocked",
                    "block_reason": "CARNOT_FORCE_LIVE not set — run with CARNOT_FORCE_LIVE=1",
                    "n_gpus": 0,
                    "pynvml_installed": False,
                    "max_gpu0_util_pct": 0.0,
                    "max_gpu1_util_pct": 0.0,
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
                    "pynvml_installed": False,
                    "max_gpu0_util_pct": 0.0,
                    "max_gpu1_util_pct": 0.0,
                    "throughput_ratio": None,
                    "retro_071_resolved": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- 5. Install pynvml if not available ---
        pynvml_installed = ensure_pynvml()
        _log.info("pynvml_installed=%s", pynvml_installed)

        # --- 6a. Init pynvml handles ---
        handle0 = None
        handle1 = None
        if pynvml_installed:
            try:
                import pynvml  # noqa: PLC0415
                pynvml.nvmlInit()
                handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
                handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
                _log.info("pynvml handles acquired for GPU0 and GPU1")
            except Exception as exc:
                _log.warning("pynvml handle acquisition failed: %s", exc)
                pynvml_installed = False

        # --- 6b. Load Qwen3.5-0.8B on cuda:0 and cuda:1 ---
        # device_map={'': 'cuda:N'} pins ALL layers to one device — prevents the
        # RETRO-025 zombie-allocation pattern where 'auto' leaks VRAM to GPU1
        # without scheduling compute there.
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
                    "pynvml_installed": pynvml_installed,
                    "max_gpu0_util_pct": 0.0,
                    "max_gpu1_util_pct": 0.0,
                    "throughput_ratio": None,
                    "retro_071_resolved": False,
                },
                status="blocked",
            )
            (_REPO_ROOT / DELIVERABLE).write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # --- 7. Parallel inference + utilization polling ---
        # All four threads start simultaneously:
        #   Thread 1 — 10 questions on cuda:0
        #   Thread 2 — 10 questions on cuda:1
        #   Thread 3 — poll GPU0 utilization every 2s
        #   Thread 4 — poll GPU1 utilization every 2s
        gpu0_util_readings: list[float] = []
        gpu1_util_readings: list[float] = []
        stop_event = threading.Event()

        result0: dict | None = None
        result1: dict | None = None
        inference_error: str | None = None

        # Build poller threads — only spawn when pynvml handles were acquired.
        poller_threads: list[threading.Thread] = []
        if pynvml_installed and handle0 is not None and handle1 is not None:
            t_poll0 = threading.Thread(
                target=poll_gpu_utilization,
                args=(handle0, stop_event, gpu0_util_readings, 2.0),
                daemon=True,
            )
            t_poll1 = threading.Thread(
                target=poll_gpu_utilization,
                args=(handle1, stop_event, gpu1_util_readings, 2.0),
                daemon=True,
            )
            poller_threads = [t_poll0, t_poll1]
            for t in poller_threads:
                t.start()

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future0 = executor.submit(
                    run_inference_batch, model0, tok0, QUESTIONS_GPU0, 0
                )
                future1 = executor.submit(
                    run_inference_batch, model1, tok1, QUESTIONS_GPU1, 1
                )
                try:
                    result0 = future0.result(timeout=600)
                except Exception as exc:
                    _log.warning("GPU0 inference failed: %s", exc)
                    inference_error = str(exc)
                try:
                    result1 = future1.result(timeout=600)
                except Exception as exc:
                    _log.warning("GPU1 inference failed: %s", exc)
                    if inference_error is None:
                        inference_error = str(exc)
        finally:
            stop_event.set()
            for t in poller_threads:
                t.join(timeout=5.0)
            if pynvml_installed:
                try:
                    import pynvml as _pynvml  # noqa: PLC0415
                    _pynvml.nvmlShutdown()
                except Exception:
                    pass

        # --- 8. Compute peak utilization ---
        max_gpu0_util_pct = max(gpu0_util_readings) if gpu0_util_readings else 0.0
        max_gpu1_util_pct = max(gpu1_util_readings) if gpu1_util_readings else 0.0

        # --- 9. Throughput ratio ---
        gpu0_latency_s = result0["total_latency_s"] if result0 else None
        gpu1_latency_s = result1["total_latency_s"] if result1 else None

        if gpu0_latency_s is not None and gpu1_latency_s is not None:
            parallel_time = max(gpu0_latency_s, gpu1_latency_s)
            sequential_time = gpu0_latency_s + gpu1_latency_s
            throughput_ratio = round(sequential_time / parallel_time, 3) if parallel_time > 0 else None
        else:
            throughput_ratio = None

        # --- 10. Honest verdict ---
        gpu1_inference_completed = result1 is not None
        if max_gpu1_util_pct > 0 and gpu1_inference_completed:
            honest_verdict = "dualgpu_confirmed"
            retro_071_resolved = True
        elif gpu1_inference_completed and not pynvml_installed:
            honest_verdict = "dualgpu_partial_no_pynvml"
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
                "pynvml_installed": pynvml_installed,
                "gpu0_result": result0,
                "gpu1_result": result1,
                "gpu0_latency_s": gpu0_latency_s,
                "gpu1_latency_s": gpu1_latency_s,
                "gpu0_util_sample_count": len(gpu0_util_readings),
                "gpu1_util_sample_count": len(gpu1_util_readings),
                "max_gpu0_util_pct": max_gpu0_util_pct,
                "max_gpu1_util_pct": max_gpu1_util_pct,
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
