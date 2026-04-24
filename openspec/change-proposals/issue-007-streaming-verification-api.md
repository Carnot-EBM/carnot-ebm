# Streaming verification API (`verify_stream`) for large candidate pools

**Status:** Draft change proposal.
**Origin:** [GitHub issue #7](https://github.com/Carnot-EBM/carnot-ebm/issues/7) (2026-04-24).
**Target milestone:** 2026.04.63, pairs well with issue #2 (budget_ms) and
  #3 (structured VerdictRecord) — all three generalize the public pipeline
  API surface.
**Priority:** Medium-high. Unlocks Exp 817-style multi-agent-arbiter use
  cases at scale.
**Depends on:** `VerifyRepairPipeline.verify()` (exists),
  `VerdictRecord` dataclass from issue #3 (proposed),
  `budget_ms` from issue #2 (proposed).

## Summary

Milestone .62 Exp 817 (Multi-Agent Arbiter MCP) ranks agent responses by EBM
energy. Same primitive generalizes to streaming use cases:

- Large-N candidate ranking (top-k from hundreds, partial useful, full-pool
  wasteful).
- Interactive IDE/CLI flows (show top verdict ASAP, refine with more).
- Budget-aware consumers (stop when the margin between 1st and (k+1)th
  tightens).
- Backpressure-friendly pipelines (MCP consumer throttles; producer pauses
  rather than buffering).

See issue #7 for the full use-case breakdown.

## Proposed experiments

### Exp A — `verify_stream` MCP tool + Python async iterator

**Deliverable:** `python/carnot/pipeline/verify_stream.py` +
MCP tool registration in `tools/verify-mcp/server.py` +
`results/experiment_<N>_verify_stream_primitive.json`.

**API sketch (from issue #7):**

```python
async def verify_stream(
    candidates: list[dict],  # [{"id": str, "question": str, "answer": str}, ...]
    budget_ms_per_candidate: int | None = None,
    budget_ms_total: int | None = None,
    top_k: int | None = None,
    early_stop_margin: float | None = None,
) -> AsyncIterator[VerdictRecord]:
    """Yields verdicts as they complete, in energy-sorted order."""
```

**Semantics:**

- Backed by a priority queue over partial verdicts — as tier 0/1 fast-paths
  clear, the queue surfaces the current top-k candidates with calibrated
  confidence.
- `top_k=None` → streams all candidates.
- `top_k=K` + `early_stop_margin=ε` → stop when the margin between rank K and
  K+1 stabilizes above ε across N consecutive polls.
- Yields `VerdictRecord` objects (from issue #3) enriched with `rank` and
  `is_final` fields.
- MCP tool wraps the async iterator with a backpressure-aware stream.

**Acceptance gates:**

1. On a 500-candidate test pool, `verify_stream` emits the top-10 within
   30% of the wall time of 500 sequential `verify()` calls — parallelism
   on tier 0/1 must actually save time.
2. Early-stop logic: on a top-5 query with `early_stop_margin=0.1`, the
   API terminates before processing 100% of candidates in at least 80% of
   runs on well-separable pools.
3. Honest-verdict enum: `stream_ships`, `stream_no_parallelism_speedup`,
   `stream_early_stop_broken`, `blocked_on_verdict_record_missing`.

### Exp B — Backpressure + cancellation

Streaming MCP consumer pauses; producer must pause verification after the
in-flight candidate completes, not mid-tier. Cancellation from consumer
must propagate to underlying verification subprocesses cleanly.

### Exp C — Benchmark vs naive top-k

Compare `verify_stream(top_k=5)` vs a naive loop that calls `verify()`
500 times and sorts. Publish latency-reduction number. Feeds the
generative-time-safety-gate use case (where the cascade needs to rank
safety verdicts across an N-way safety-classifier stack).

## Risks

- **Depends on issues #2 + #3 landing first.** `verify_stream` returning
  `VerdictRecord` with per-candidate `budget_ms` needs both. Schedule
  #2 and #3 in the same or earlier milestone.
- **Parallelism requires thread-safe tier implementations.** EORM is
  thread-safe; Ising sampler via `FpgaBackend` is not (one kernel at a
  time). Streaming will fan out to the parallelizable tiers but serialize
  on Ising. Acknowledge this in docs.
- **Memory pressure at large N.** Priority queue of VerdictRecords with
  full `evidence` dicts scales linearly. Mitigation: `evidence` is
  populated lazily on `is_final=True` only.
