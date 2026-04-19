#!/usr/bin/env python3
"""Experiment 511: AMD XDNA NPU Probe — per-token entropy on NPU via ONNX/VitisAI EP.

**Research question:**
    Can the AMD XDNA NPU on this machine run per-token softmax entropy computation
    at <5ms/token latency, enabling zero-overhead Tier 0c hallucination filtering
    that pipelines with LLM generation?

**Background (arXiv 2504.03083):**
    April 2025 paper demonstrates LLM fine-tuning on Ryzen AI XDNA NPU via the IRON
    tool-flow.  The NPU's 2D spatial AI Engine array is optimised for streaming
    arithmetic reductions — exactly what softmax + H(p) requires.

    Per-token entropy is O(vocab_size=50k) ops per token.  If the NPU achieves
    <5ms/token AND the LLM generates at 5-50ms/token (20-100 tokens/sec), the
    entropy probe can run ahead of generation → Tier 0c = zero-overhead filter.

**What this experiment does:**
    1. Export the softmax + entropy ONNX graph.
    2. Attempt to load it via VitisAI EP.
    3. Benchmark latency: NPU (if available) vs CPU baseline.
    4. Emit honest_verdict:
       - 'npu_viable'                if NPU available and speedup >= 2x
       - 'npu_measured_no_speedup'   if NPU available but speedup < 2x
       - 'npu_not_available'         if VitisAI EP not installed

**Why CPU-only fallback is correct:**
    If VitisAI EP is absent, we still measure the CPU baseline and emit setup
    instructions.  The experiment must NOT fail silently (REQ-INFRA-063).

Spec: REQ-INFRA-061, REQ-INFRA-062, REQ-INFRA-063,
      SCENARIO-INFRA-070, SCENARIO-INFRA-071, SCENARIO-INFRA-072
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.npu_entropy_probe import NPUEntropyProbe
from experiment_template import ExperimentTemplate

_DELIVERABLE = "results/experiment_511_amd_npu_probe.json"
_ONNX_PATH = "results/npu_entropy_probe.onnx"

_VITISAI_SETUP_INSTRUCTIONS = (
    "VitisAI execution provider not found. To enable AMD XDNA NPU inference:\n"
    "  1. pip install onnxruntime-vitisai\n"
    "  2. Install Vitis AI runtime from https://github.com/amd/RyzenAI-SW\n"
    "     (follow the NPU EP quickstart for your OS/kernel version)\n"
    "  3. Set XLNX_VART_FIRMWARE to the NPU firmware binary path.\n"
    "  4. Verify: python -c \"import onnxruntime; "
    "print(onnxruntime.get_available_providers())\"\n"
    "     Expected: [..., 'VitisAIExecutionProvider', ...]"
)


def main() -> None:
    # Step 1: environment autofix FIRST (belt-and-suspenders for CARNOT_FORCE_LIVE)
    apply_env_autofix()

    tmpl = ExperimentTemplate(
        511,
        "AMD XDNA NPU Probe",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    guard = DeliverableGuard(_REPO_ROOT / _DELIVERABLE)

    with ExperimentTimeoutWatchdog(511, timeout_minutes=25):
        probe = NPUEntropyProbe(seq_len=64, vocab_size=50000)

        onnx_path = str(_REPO_ROOT / _ONNX_PATH)
        probe.export_onnx(onnx_path)
        probe.load_vitisai(onnx_path)

        result = probe.benchmark(n_trials=100)

        if result.npu_viable:
            honest_verdict = "npu_viable"
        elif result.npu_available:
            honest_verdict = "npu_measured_no_speedup"
        else:
            honest_verdict = "npu_not_available"

        setup_instructions = (
            _VITISAI_SETUP_INSTRUCTIONS if not result.npu_available else None
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.npu_entropy_probe.v1",
                "npu_available": result.npu_available,
                "npu_latency_ms": result.npu_latency_ms,
                "cpu_latency_ms": result.cpu_latency_ms,
                "speedup_ratio": result.speedup_ratio,
                "npu_viable": result.npu_viable,
                "setup_instructions": setup_instructions,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        out_path = _REPO_ROOT / _DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
