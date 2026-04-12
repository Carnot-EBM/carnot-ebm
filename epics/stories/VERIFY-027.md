# Epic: VERIFY-027 - Dual-GPU Parallel Runner For Exp 218

**Status:** Complete
**Goal:** Add `python/carnot/inference/dual_gpu.py`, extend
`python/carnot/inference/model_loader.py`, and update
`scripts/experiment_218_live_dual_model_suite.py` so the paired live harness
can place Qwen on `cuda:0`, Gemma on `cuda:1`, fall back to
`device_map="auto"` for `7B+` models, and benchmark sequential versus
parallel wall time on a deterministic 10-question slice.
**Rationale:** Exp 219 / 220 / 221 currently run the paired small models
sequentially even on a machine with **2x RTX 3090**. A dedicated dual-GPU
runner should reduce wall time where the models fit independently, but the
repository needs an honest measured benchmark plus a safe sharded fallback for
larger models.

## Stories
- [x] Add `REQ-VERIFY-041` and `SCENARIO-VERIFY-042` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for the dual-GPU runner, explicit `cuda:N`
  `model_loader` support, `device_map="auto"` fallback, and the Exp 218
  `--parallel` dispatch path
- [x] Implement `python/carnot/inference/dual_gpu.py`
- [x] Integrate explicit device and `device_map` loading into
  `python/carnot/inference/model_loader.py` without breaking existing callers
- [x] Add `--parallel` to
  `scripts/experiment_218_live_dual_model_suite.py` while preserving ordered
  paired artifact output
- [x] Run the requested sequential-versus-parallel 10-question benchmark and
  record the honest result artifact
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
