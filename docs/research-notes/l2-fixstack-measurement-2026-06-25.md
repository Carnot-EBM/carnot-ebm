# Measuring the L2 fix stack: it does NOT bank lp85 L2 — and the wall is upstream of goal quality

**Date:** 2026-06-25 · **Author:** outer-loop (operator: "run the measurement").
**Artifact:** `results/proto_l2_measure_fixstack.json`. **Probe:** `scripts/experiments/proto_l2_measure_fixstack.py`.

## What was measured

After shipping the default-ON truncation fix + the goal-repair loop (commits c25e11211, b4c865d42),
the open question was: do they actually let lp85 bank a REPRODUCED L2? A focused single-game,
single-arm, reproduction-gated probe (lp85 binary, budget 400, warm Qwen :8920) — NOT the 4-arm A/B
that hung on sc25.

**Prediction (going in):** the fixes unblock planning (satisfiable goal via repair) but lp85 L2 still
won't reproduce, because the loose nonzero-count fallback isn't lp85's real win (shape-alignment) →
the wall is goal QUALITY.

## Result — lp85 L2 did NOT bank, but the failure mode REFUTES the prediction

`max_depth_reached=1`, `l2_offline_reproduced=False`, `goal_predicate_satisfiable_any=False`,
`goal_repair_fired=False`. Induction summary:

```
induction[0]: goal_level=1  skipped=proposer_failed_or_missing_root  (the online/L1 path)
induction[1]: goal_level=2  skipped=proposer_failed  exemplar_injected=True  (the L2 level-up reinduction)
```

The L2 level-up reinduction **`proposer_failed`** — the LLM induction produced no valid parseable
`engine`+`is_level_complete` after its 3 internal retries. Because that is UPSTREAM of the goal
satisfiability check, the goal-repair loop never engaged (it only fires after a successful induction
whose goal is degenerate). So the predicted "loose goal doesn't reproduce" mechanism was never
reached — the wall this run is **L2 reinduction reliability**, not goal quality.

## Why this is surprising — and what it bisects

- The agent DOES pass the exemplar (`arc_competition_agent.py:2415`) and `_call_induce` routes
  through `proposer.induce` → the code-only path IS applied to the L2 reinduction. No wiring gap.
- The online (non-level-up) induction SUCCEEDED this run — `results/arc_e3/lp85/world_model.py` was
  written at 13:23 mid-run with a REAL general predicate (`is_level_complete: no cell == 3 → True`),
  not a degenerate `return False`. So the model CAN write code on lp85.
- The code-only induce is **3/3 reliable on the SYNTHETIC lp85 L2 prompt** (re-tested this session).
- Yet the **REAL lp85 L2 reinduction prompt failed code-gen** (proposer_failed, 3 retries).

So the truncation fix transfers to the synthetic prompt but NOT (this run) to the real lp85 L2
reinduction prompt. The bisected wall, in order:

1. **L2 reinduction reliability (the immediate blocker):** the real lp85 L2 reinduction prompt
   reliably fails to yield valid code. The synthetic prompt does not. The difference is prompt
   content — real click-action transitions (x/y data), larger/real grids, and the real win exemplar.
   Hypotheses to check by CAPTURING the real prompt + the model's raw output: (a) prose leakage
   despite the directive; (b) the `stop=["```"]` cutting a long hardcoded win grid mid-code; (c) an
   oversized prompt degrading code adherence; (d) is_level_complete missing / syntax error.
2. **Goal quality (the structural-alignment lever):** only reachable once (1) is fixed and the
   induction reliably emits code + a satisfiable goal. The pre-staged perception-grounded
   structural-alignment L2 goal (`ops/known-issues.md` MANDATORY-NEXT-MILESTONE) remains the lever
   for the goal-quality half — but it is downstream of fixing reinduction reliability.

## Honest caveats

- **n=1 run.** "3/3 fail on the real prompt" is the single live run's internal retries, not a
  controlled repeat. The next diagnostic must CAPTURE the real lp85 L2 reinduction prompt and the
  model's raw completion (the same method used for the original truncation investigation) and test
  the valid-rate over several samples to distinguish a real prompt-specific failure from variance.
- The goal-repair loop is still correct and tested; it simply was not exercised because induction
  failed first. The default-ON truncation fix is still a strict improvement on the prompts where it
  applies (synthetic 3/3; the online induction succeeded).
- `inference_substrate=live_llm_inference`; `solve_provenance=development_proxy`;
  `verifier_is_oracle=false`. No submission; no landing-page edits.

## Recommendation

Before the goal-quality lever, fix L2 reinduction reliability: capture the real lp85 L2 reinduction
prompt + raw output, find why code-gen fails on it (vs the synthetic prompt), and harden the
code-only path for the real prompt (likely a stop-sequence / prose-leakage robustness fix). Then the
structural-alignment goal addresses the goal-quality half. The measurement was worth running: it
moved the diagnosis from a guessed "goal quality" wall to a concrete, upstream "real-L2-prompt
code-gen reliability" wall, with a clear next diagnostic.
