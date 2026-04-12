# Epic: VERIFY-025 - Warm Batched Model Server For Live Benchmarks

**Status:** In Progress
**Goal:** Add `python/carnot/inference/model_server.py` and integrate
`carnot.inference.model_loader` so repeated live benchmark calls can reuse
warm models, batch prompt inference, and measure the cold-load versus
warm-server speedup on a deterministic 50-question benchmark.
**Rationale:** Exp 219 currently pays repeated cold-load cost and processes one
question per forward pass even when the target GPUs can batch multiple
questions. A warm server removes repeated model initialization overhead,
enables request coalescing, and gives the benchmark harness a reusable
latency/throughput primitive without changing `scripts/research_conductor.py`.

## Stories
- [ ] Add `REQ-VERIFY-036` through `REQ-VERIFY-038` and
  `SCENARIO-VERIFY-036` through `SCENARIO-VERIFY-038` to the
  `verifiable-reasoning` spec before implementation changes
- [ ] Write tests first for server lifecycle, request batching, health and
  shutdown reporting, `model_loader` server-backed handles, and the
  deterministic 50-question benchmark helper
- [ ] Implement `python/carnot/inference/model_server.py`
- [ ] Integrate the warm-server option into
  `python/carnot/inference/model_loader.py` without breaking existing callers
- [ ] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [ ] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
