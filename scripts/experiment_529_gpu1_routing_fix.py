#!/usr/bin/env python3
"""Experiment 529 — GPU1 Routing Fix (RETRO-052 resolution attempt).

**What this experiment validates:**
    RETRO-052: Exp 517 (controlled DualGPU test) confirmed gpu1_compute_pct=0.0
    even in live_gpu mode with explicit device_map={'': 'cuda:1'}.
    Hypothesis: transformers allocates weights on cuda:1 at load time but the
    forward pass dispatches back to cuda:0 via a backend-level override that
    happens after device_map is applied.

    This experiment applies a triple-layer cuda:1 constraint:
      Layer 1: device_map={'': 'cuda:1'} in from_pretrained()
      Layer 2: model = model.to('cuda:1') after load (PyTorch belt-and-suspenders)
      Layer 3: verify_model_on_device(model, 1) asserts parameters are on cuda:1

    It then samples nvmlDeviceGetUtilizationRates() for GPU 1 every 0.25 s while
    running 20 short inference passes.  If mean GPU 1 compute > 10%, RETRO-052 is
    closed.  If still 0%, the result documents the exact transformers path.

Spec: REQ-INFRA-071, REQ-INFRA-072, SCENARIO-INFRA-081, SCENARIO-INFRA-082
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

# apply_env_autofix() MUST be called before any GPU-touching import.
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).parent))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.gpu1_routing_fix import (  # noqa: E402
    GPU1RoutingResult,
    force_cuda1_device_map,
    verify_model_on_device,
)
from carnot.pipeline.jit_vram_check import JITVRAMCheck  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)

DELIVERABLE = "results/experiment_529_gpu1_routing_fix.json"
EXP_ID = 529
TITLE = "GPU1 Routing Fix"
QWEN_HF_ID = "Qwen/Qwen2.5-0.5B"

# 20 short arithmetic prompts used for the inference load test.
_INFERENCE_PROMPTS = [
    f"Answer: {i} + {i} ="
    for i in range(20)
]


def _write_result(repo_root: Path, artifact: dict) -> None:
    """Atomically write *artifact* to the deliverable path."""
    out = repo_root / DELIVERABLE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(artifact, indent=2))
    tmp.rename(out)
    _log.info("Deliverable written: %s", out)


def _sample_gpu1_utilization(
    stop_event: threading.Event,
    results: list[float],
    interval_s: float = 0.25,
) -> None:
    """Poll nvmlDeviceGetUtilizationRates() for GPU 1 until stop_event is set.

    Appends the compute utilization percentage (0-100) to *results* on each
    poll.  Silently stops if pynvml is unavailable or GPU 1 does not exist.

    Why a stop_event instead of a fixed n_samples?
        We need the sampling window to overlap the active inference period.
        With a fixed count the thread might finish before inference starts,
        or inference might finish before the thread completes.  An event-based
        stop ties the window directly to the inference period.
    """
    try:
        import pynvml  # noqa: PLC0415

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # GPU 1
        while not stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            results.append(float(util.gpu))
            time.sleep(interval_s)
        pynvml.nvmlShutdown()
    except Exception as exc:
        _log.warning("nvml sampling failed: %s — no GPU 1 utilization data", exc)


def main() -> None:
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):
        tmpl = ExperimentTemplate(
            EXP_ID,
            TITLE,
            DELIVERABLE,
            requires_gpu=True,
        )
        tmpl.setup()

        repo_root = tmpl._repo_root
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

        # --- Gate: live GPU required for utilization measurement ---
        if not force_live:
            _log.info(
                "CARNOT_FORCE_LIVE not set — emitting gpu_required artifact. "
                "Re-run with CARNOT_FORCE_LIVE=1 on a dual-GPU host."
            )
            result = GPU1RoutingResult(
                device_used="unknown",
                gpu1_compute_pct_during_inference=0.0,
                routing_verified=False,
                honest_verdict="gpu_required",
            )
            artifact = tmpl.build_result(
                {
                    "device_used": result.device_used,
                    "gpu1_compute_pct_during_inference": result.gpu1_compute_pct_during_inference,
                    "routing_verified": result.routing_verified,
                    "model_on_device": False,
                    "retro_052_closed": False,
                    "honest_verdict": result.honest_verdict,
                    "inference_mode": "gpu_required",
                },
                status="gpu_required",
                schema="carnot.gpu1_routing_fix.v1",
            )
            _write_result(repo_root, artifact)
            tmpl.assert_deliverable_written()
            return

        # --- Live GPU path ---

        # Step 1: JIT VRAM gate for GPU 1 (Qwen3.5-0.8B fits in ~1.5 GiB)
        gate1 = JITVRAMCheck(device_id=1)
        vram1 = gate1.gate_model_load("qwen3.5-0.8b", required_gb=1.5)
        if not vram1.is_cleared:
            _log.warning(
                "GPU 1 VRAM insufficient (%.1f GB free, need 1.5 GB) — aborting",
                vram1.available_gb,
            )
            result = GPU1RoutingResult(
                device_used="unknown",
                gpu1_compute_pct_during_inference=0.0,
                routing_verified=False,
                honest_verdict="gpu_required",
            )
            artifact = tmpl.build_result(
                {
                    "device_used": result.device_used,
                    "gpu1_compute_pct_during_inference": result.gpu1_compute_pct_during_inference,
                    "routing_verified": result.routing_verified,
                    "model_on_device": False,
                    "retro_052_closed": False,
                    "honest_verdict": result.honest_verdict,
                    "inference_mode": "gpu_required",
                    "gpu1_vram_available_gb": vram1.available_gb,
                },
                status="blocked",
                schema="carnot.gpu1_routing_fix.v1",
            )
            _write_result(repo_root, artifact)
            tmpl.assert_deliverable_written()
            return

        # Step 2: Load Qwen3.5-0.8B with THREE explicit cuda:1 constraints.
        # See module docstring for why each layer matters.
        try:
            import torch  # noqa: PLC0415
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            # Layer 1: force_cuda1_device_map() returns {'': 'cuda:1', '_model_id': ...}
            # Transformers ignores unknown keys so _model_id is safe to include.
            dm = force_cuda1_device_map(QWEN_HF_ID)
            _log.info("Loading %s with device_map=%s", QWEN_HF_ID, dm)
            qwen_tok = AutoTokenizer.from_pretrained(QWEN_HF_ID)
            qwen_model = AutoModelForCausalLM.from_pretrained(
                QWEN_HF_ID,
                device_map=dm,
                torch_dtype=torch.float16,
            )

            # Layer 2: belt-and-suspenders model.to() reassigns all parameters at the
            # PyTorch level, overriding any backend dispatch that ignores device_map.
            _log.info("Applying model.to('cuda:1') belt-and-suspenders")
            qwen_model = qwen_model.to("cuda:1")

            # Layer 3: verify the model's first parameter is actually on cuda:1.
            model_on_device = verify_model_on_device(qwen_model, expected_device_id=1)
            if not model_on_device:
                _log.warning(
                    "verify_model_on_device returned False — first parameter is NOT on cuda:1 "
                    "despite device_map and model.to().  Forward passes may run on wrong GPU."
                )
            else:
                _log.info("verify_model_on_device: confirmed model parameters on cuda:1")

            def _qwen_infer(prompt: str) -> str:
                """Run one inference pass on cuda:1, return decoded output."""
                inputs = qwen_tok(prompt, return_tensors="pt").to("cuda:1")
                with torch.no_grad():
                    out = qwen_model.generate(**inputs, max_new_tokens=20, do_sample=False)
                return qwen_tok.decode(out[0], skip_special_tokens=True)

            load_ok = True
            device_used = "cuda:1"

        except Exception as exc:
            _log.warning("Qwen load failed: %s — cannot verify GPU 1 routing", exc)
            load_ok = False
            model_on_device = False
            device_used = "unknown"

            def _qwen_infer(prompt: str) -> str:  # type: ignore[misc]
                return "load_failed"

        if not load_ok:
            result = GPU1RoutingResult(
                device_used=device_used,
                gpu1_compute_pct_during_inference=0.0,
                routing_verified=False,
                honest_verdict="gpu1_still_idle",
            )
            artifact = tmpl.build_result(
                {
                    "device_used": result.device_used,
                    "gpu1_compute_pct_during_inference": result.gpu1_compute_pct_during_inference,
                    "routing_verified": result.routing_verified,
                    "model_on_device": model_on_device,
                    "retro_052_closed": False,
                    "honest_verdict": result.honest_verdict,
                    "inference_mode": "live_gpu",
                    "load_failed": True,
                },
                status="blocked",
                schema="carnot.gpu1_routing_fix.v1",
            )
            _write_result(repo_root, artifact)
            tmpl.assert_deliverable_written()
            return

        # Step 3: Start nvml sampling thread, then run 20 inference passes.
        stop_event = threading.Event()
        util_samples: list[float] = []
        sampler_thread = threading.Thread(
            target=_sample_gpu1_utilization,
            args=(stop_event, util_samples, 0.25),
            daemon=True,
        )
        sampler_thread.start()

        _log.info("Running 20 inference passes on cuda:1...")
        responses: list[str] = []
        for prompt in _INFERENCE_PROMPTS:
            try:
                resp = _qwen_infer(prompt)
                responses.append(resp)
            except Exception as exc:
                _log.warning("Inference pass failed: %s", exc)
                responses.append("error")

        # Stop the sampler and collect results.
        stop_event.set()
        sampler_thread.join(timeout=5.0)

        gpu1_pct = float(sum(util_samples) / len(util_samples)) if util_samples else 0.0
        _log.info(
            "GPU 1 mean compute pct: %.1f%% (over %d samples)", gpu1_pct, len(util_samples)
        )

        routing_verified = gpu1_pct > 10.0
        if routing_verified:
            honest_verdict = "gpu1_active"
            _log.info("RETRO-052 CLOSED — GPU 1 compute confirmed (%.1f%%)", gpu1_pct)
        else:
            honest_verdict = "gpu1_still_idle"
            _log.warning(
                "GPU 1 still idle (%.1f%%) despite triple cuda:1 constraint. "
                "Next step: investigate nvml sampling window timing.",
                gpu1_pct,
            )

        artifact = tmpl.build_result(
            {
                "device_used": device_used,
                "gpu1_compute_pct_during_inference": gpu1_pct,
                "routing_verified": routing_verified,
                "model_on_device": model_on_device,
                "retro_052_closed": routing_verified,
                "honest_verdict": honest_verdict,
                "inference_mode": "live_gpu",
                "n_inference_passes": len(responses),
                "n_util_samples": len(util_samples),
                "responses_sample": responses[:3],
            },
            status="success",
            schema="carnot.gpu1_routing_fix.v1",
        )

        _write_result(repo_root, artifact)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
