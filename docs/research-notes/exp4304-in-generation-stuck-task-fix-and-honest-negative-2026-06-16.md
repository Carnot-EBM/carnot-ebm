# exp4304 (PHASE B1, .398 in-generation moat): stuck-task fix + honest negative

**Outer-loop intervention, 2026-06-16 ~22:00 UTC (operator-authorized: "kill it and
fix the problem and then restart it").**

## What was stuck

`PHASE B1 (exp4304, DiffusionGemma in-generation)` doomed-looped. Two codex attempts
both FAILed:

- **Attempt 1 (FAIL 21:54):** codex hit the conductor's 80-min HARD wall-clock cap. It
  had spun the **entire ~37,565-test suite** through a per-test memory-profiling harness
  (`tests/python/conftest.py` `PytestMemoryWatchdog`, 8 GB `RLIMIT_AS`). Subprocess
  workers inherited the 8 GB virtual-memory cap, OOM'd (several went defunct/zombie), and
  the parent pytest dead-locked at `futex_wait`. This is the codex over-engineering
  pattern (running the full suite instead of the conductor's ~10 s smart-subset gate).
- **Attempt 2 (FAIL 22:07 `artifact_not_updated_past_bootstrap`):** codex again ran the
  full suite (slowly; it hit a **Z3 segmentation fault in `z3_math_verifier`** plus other
  unrelated failures, reached ~25%), then declared a deliverable the conductor rejected as
  bootstrap-only.

Neither failure was a defect in the exp4304 experiment itself.

## The fix: the experiment runs fine — codex just never ran it

Preconditions are all met (DiffusionGemma PR binary `llama-diffusion-gemma-eval` + the
`diffusiongemma-26B-A4B-it-Q4_K_M.gguf` cached; the exp4292 partial-state scorer artifact
present). Running the finished 936-line experiment directly:

```
.venv/bin/python -m carnot.experiment_4304_diffusiongemma_in_generation_engaged_controls \
  --max-tasks 30 --minimum-duration-s 60
```

completed in **210 s** (`live_llm_inference`, the diffusion binary is CPU-bound at ~100%),
producing the real artifact. So the 80-min "hang" was entirely codex's full-suite tangent,
not the experiment.

## The honest result: in-generation moat NOT established (engaged control collapsed)

`results/experiment_4304_diffusiongemma_in_generation_engaged_controls.json`:

- `honest_verdict: partial: diffusiongemma_guidance_prior_eval_incomplete`
- `diffusiongemma_guidance_moat: False`, `controls_differentiated: False`
- `condition_accuracy: {carnot 0.4231, entrgi 0.3077, unguided 0.3077}`,
  `scorer_leak_recheck_passed: True`, `verifier_is_oracle: False`

carnot beats the controls (+0.115), BUT **`entrgi == unguided` bit-identical (0.307692)** —
the "engaged" EntRGi control (`entrgi = unguided + ENTRGI_GAMMA·entropy_gate·
candidate_prior`) collapses to the unguided no-op because the guidance term is ≈0 on this
corpus. So the experiment **correctly refuses the moat** (`controls_differentiated=False`)
rather than claiming "carnot beats a no-op" — the exact circularity .397's exp4293 was
flagged for. `adversarial_verify` flags it CRITICAL **TAUTOLOGY** (the two identical deltas)
— a true-positive degenerate-control signature, so the artifact is correctly
**quarantine-eligible / not headline-eligible**.

This is the SAME honest open result as .397, now confirmed *with* an engaged-control
attempt: **the in-generation moat remains unestablished.** It was NOT faked to a `complete:`
verdict — re-running the same code reproduces the collapse, and gaming the prefix to dodge
the conductor's acceptance check would violate the verdict / circularity disciplines.

## What a real fix needs (next-milestone research, NOT a watchdog edit)

To genuinely test the in-generation moat, the engaged control must actually engage:

1. Make EntRGi non-degenerate — tune `ENTRGI_GAMMA` / fix the entropy-gate so
   `entrgi != unguided` (a mechanical no-op guard already rejects ties; the arm just needs
   to produce distinct selections), OR
2. Use a 2nd-checkpoint RFG control (a strictly-weaker reference DiffusionGemma vs the
   enhanced generator at γ>0, the provably-engaged regime), per the exp4304 task design.

Until then the in-generation axis stays open — the hardest, longest-unproven §5 axis.

## Process notes

- Cleaned up leaked orphan `python -c exec` pool-workers (reparented to `systemd --user`
  from the first dead codex) that were pegging CPU; killed the second attempt's frozen
  full-suite pytest. All by exact PID (never `pkill -f`, which self-matches the outer-loop
  shell → exit 144). The conductor (PID 935238) was never touched.
- Latent infra bug surfaced: the full suite hits a **Z3 segfault in `z3_math_verifier`** and
  the `PytestMemoryWatchdog` 8 GB `RLIMIT_AS` dead-locks subprocess-spawning tests. The
  conductor's normal smart-subset gate avoids both (cached-green), so this only bites when
  an agent runs the *whole* suite. Worth a dedicated fix so codex full-suite runs can't
  doom-loop.
- exp4304 will 3-fail-skip (2/3 used) and the conductor proceeds to PHASE C1; this honest
  artifact is preserved for the .398 capstone (which must SKIP it as flagged, per the
  fabrication gate — never aggregate the degenerate numbers into a headline).
