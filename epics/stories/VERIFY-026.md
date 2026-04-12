# Epic: VERIFY-026 - TensorRT-LLM Backend For Warm Inference

**Status:** Complete
**Goal:** Add an optional TensorRT-LLM backend that can build and cache
engines for the warm inference path, prefer those engines in
`python/carnot/inference/model_server.py`, and benchmark HuggingFace versus
TensorRT execution on the existing deterministic 50-question harness.
**Rationale:** Exp 224a removed repeated cold loads and enabled real batching,
but the wall time is still dominated by `model.generate()` itself. TensorRT-LLM
promises a further 2-4x speedup through fused kernels, quantization, and KV
cache optimization, while the repository still needs a clean HuggingFace
fallback for CI and machines where TensorRT is unavailable.

## Stories
- [x] Add `REQ-VERIFY-039` and `REQ-VERIFY-040` plus
  `SCENARIO-VERIFY-039` through `SCENARIO-VERIFY-041` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for TensorRT availability detection, engine-cache
  reuse, build failure fallback, ModelServer backend preference, and the
  deterministic 50-question HF-vs-TRT benchmark helper
- [x] Implement `python/carnot/inference/tensorrt_backend.py`
- [x] Integrate TensorRT preference into
  `python/carnot/inference/model_server.py` without breaking the existing
  warm-server or `model_loader` contracts
- [x] Attempt real engine build and HF-vs-TRT benchmarking for
  `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it`, or record the concrete
  local prerequisite that blocks the live step
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
