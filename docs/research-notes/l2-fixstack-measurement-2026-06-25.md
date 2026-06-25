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

## CAPTURE DIAGNOSTIC (2026-06-25, follow-up) — root cause + the tested fix

`scripts/experiments/proto_l2_capture.py` tee'd every `/completion` call while the lp85 agent
played, and captured the EXACT real L2 reinduction call (`results/l2_capture.jsonl`). Then
`scripts/experiments/proto_l2_fix_finder.py` tested candidate fixes on that exact prompt
(`results/proto_l2_fix_finder.json`).

**Root cause (definitive, live-representative — the live agent uses the same `max_tokens=2560`,
`arc_competition_agent.py:2200`):** the real lp85 L2 reinduction prompt is `9290` chars (vs the
synthetic ~2200) because it carries real click-action transitions + grids. The model **ignores the
directive's "not even as comments" and rambles its analysis into code COMMENTS inside `engine()`**
("The game seems to toggle...", "Since we cannot deduce the exact rule...", "as a placeholder, but
this is not ideal..."), filling all 2560 tokens (`stop_type=limit`) on a verbose `engine()` and
**never reaching `def is_level_complete`** -> missing def -> `proposer_failed`. The synthetic prompt
is simple enough that the model writes concise code that fits; the real one is not. This is NOT
variance: it is a deterministic, prompt-complexity-driven failure.

**The fix (tested on the captured prompt):**

| candidate | result |
|---|---|
| **B — raise budget 2560→4096** | **FAILS** (`stop=limit`, 4096 tokens, still no `is_level_complete`, 369s). The model rambles MORE to fill the bigger budget. **Do NOT just bump `max_tokens`.** |
| **C — goal-first ordering** (write `is_level_complete` before `engine`) | works (`stop=word`, 185 tokens, both defs parse, 72s) — the short goal is written before the engine ramble can starve it. Cheap one-call fix, but n=1 (may be variance-sensitive to whether the engine then also stays concise). |
| **D — separate focused goal call** (`is_level_complete`-only, win exemplar front-and-centre) | **works best** (`stop=word`, **17 tokens, 3.5s**, valid). Structural: the engine ramble cannot starve a goal induced in its OWN call. |

**Recommended fix: D (separate focused goal induction).** Induce `is_level_complete` in its own
focused call (reliable, 3.5s) rather than relying on the model to write both functions in one budget.
Minimal-blast-radius implementation: keep the combined induce for `engine`, and when the combined
output has a valid engine but is missing/degenerate `is_level_complete` (the captured failure mode),
do a focused goal-only call to obtain it, then combine. This unblocks the L2 reinduction so the
goal-repair loop + the eventual structural-alignment goal can actually run. (Goal-first ordering C is
a cheaper alternative but less robust than the separate call.)

This conclusively answers the bisected wall #1 (L2 reinduction reliability): the fix is a separate
focused goal call, NOT a budget increase. Wall #2 (goal quality / structural-alignment goal) is then
next, and now reachable.

## END-TO-END MEASUREMENT after shipping the split-fallback (2026-06-25, v2) — HONEST NEGATIVE

Re-ran the live lp85 probe (`results/proto_l2_measure_fixstack.json`, 940s) WITH the split-fallback
fix (commit 06b527ea1) in the tree. **lp85 L2 still does NOT bank** (`max_depth=1`,
`l2_offline_reproduced=False`, `goal_satisfiable_any=False`). The induction breakdown is the honest
story, and it REFUTES the optimistic read that the split alone unblocks L2:

```
[0] goal_level=1  skipped=world_model_accuracy_below_threshold  heldout_acc=0.12   (L1 engine only 12% accurate)
[1] goal_level=2  skipped=proposer_failed  exemplar_injected=True                  (L2 induction STILL fails)
```

**Two walls, both rooted in MODEL CAPABILITY on the real prompts — not prompt engineering:**

1. **L1 engine accuracy = 0.12.** The combined L1 induction DID succeed (wrote engine +
   is_level_complete), but the induced ENGINE reproduces only 12% of observed transitions -> rejected
   by the held-out verifier. The model writes a verbose placeholder engine (2639 chars of
   `# Copy the grid... # If action is 6...` comments) that does not capture lp85's real click dynamics.
2. **L2 induction still `proposer_failed`** despite the split fix being correct in isolation
   (unit tests + the isolated real-prompt test where engine-only=35s and goal-only=8.6s both
   produced valid code). In the FULL live run it still fails. The most likely mechanism: the
   combined call rambles to the 2560 limit (`stop_type=limit`, ~230s at ~11 tok/s) and the focused
   engine-only call ALSO rambles on the real multi-transition prompt; with the proposer timeout at
   300s (`LocalGGUFProposer.timeout=300`) and likely iGPU contention from the concurrent conductor
   milestone, the rambling calls exceed the timeout and the split's calls fail too. (Confirming
   exactly whether the split fired vs timed out needs a tee'd-HTTP capture of the L2 induction's
   call count — not yet run.)

**The honest conclusion.** The truncation fix + goal-repair + split-fallback are each correct and
necessary, but the end-to-end measurement shows they are NOT sufficient: the binding wall is that
**Qwen3.5-9B produces low-quality, rambling, placeholder code on the REAL lp85 prompts** (12%-accurate
engine; `return False` goal), where the SYNTHETIC prompts were simple enough to mask it. This is a
model-capability / induction-quality wall.

**Revised recommendation (evidence-based, supersedes the optimistic "now reachable"):**
- The lever is to reduce what the model must induce from free-form code, OR to use a stronger
  inducer: (a) the **perception-grounded structural-alignment goal** (pre-staged) gives the model a
  structured predicate over detected objects instead of free-form is_level_complete -- directly
  attacks the placeholder-goal problem; (b) the **engine/dynamics accuracy** (12%) is its own wall --
  a structured action-effect model (the existing CNN/program-synthesis action-effect work) may beat
  free-form LLM engine induction; (c) a stronger induction model (the frozen ARC sprint stack is
  Qwen3.5-9B for the live generator, but the offline induction could use a larger model).
- A budget bump and the split are both confirmed insufficient on their own; do not pursue further
  prompt-only tweaks as the primary lever.
- The split-fallback remains shipped (correct, tested, fabrication-safe) -- it is a real robustness
  improvement that helps when the focused calls DON'T ramble; it just does not, by itself, clear the
  model-capability wall on lp85's real prompts.
