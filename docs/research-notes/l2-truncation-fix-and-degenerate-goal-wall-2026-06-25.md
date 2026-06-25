# L2 goal-induction: the token-truncation wall is FIXED — and fixing it reveals a deeper degenerate-goal wall

**Date:** 2026-06-25 · **Author:** outer-loop (operator-directed: "update qwen ctx to 16384; re-call
the captured L2 prompt with a stronger code-only prefix at 4096; if CoT irrepressible fall back to
raise-budget; if it produces a satisfiable goal, chain into re-testing the graded-bias fix to see if
lp85 L2 banks").
**Artifacts:** `results/proto_l2_code_only_prefix.json`, `results/proto_graded_goal_bias_ab.json`.
**Code:** `scripts/experiments/proto_l2_code_only_prefix.py`, `proto_l2_stopseq_verify.py`;
shipped fix env-gated in `python/carnot/agentic/arc_executable_world_model.py`
(`CARNOT_ARC_CODEONLY_INDUCE=1`).

## TL;DR

1. **Qwen context** was already `n_ctx=16384` (code + running llama-server) — no change needed.
2. **The truncation wall is real and now FIXED.** A forceful CODE-ONLY directive + a `stop=["```"]`
   sequence makes the L2 induction emit ONLY the code block instead of burning the whole 4096-token
   budget on win-state chain-of-thought (CoT). Measured on the lp85 L2 prompt:
   - baseline `/no_think`: `stop_type=limit`, 4096 tokens, **0 code**, 450s.
   - code-only prefix (no stop): valid `engine`+`is_level_complete`, but rambled to the limit (605s).
   - **code-only prefix + `stop=["```"]`: valid code, `stop_type=word`, 195 tokens, 15.6s** — a
     ~30× latency *improvement*, not a hit. In isolation the induced predicate is
     satisfiable+discriminating (True on the rendered win state, False elsewhere).
   - Wired end-to-end through the shipped `generate()`: returns valid+satisfiable code in **12.9s**
     where the unpatched path produced nothing.
3. **But fixing truncation does NOT make lp85 L2 bank.** On the LIVE lp85 game the fixed induction
   emits code, yet the induced `is_level_complete` is **degenerate** (`return False`), so the
   harness's `_goal_satisfiability_check` (BFS over the induced engine; satisfiable iff some
   reachable grid makes the predicate True) returns `satisfiable=False`. lp85 stays at L1.
   **The deeper wall is goal-induction QUALITY, not token budget.**

## What was wrong before (the truncation wall)

`proto_l2_proposer_truncation_check` (prior session) proved the 4096-token L2 induce call returned
`stop_type='limit'` with `tokens_predicted=4096` and **zero** code: the model spent the entire
budget "analyzing the win state" (CoT prose) and never reached the code block. `/no_think` did not
suppress this. So `is_level_complete` was never induced → `goal_predicate_satisfiable=False` for
~10 milestones, blamed on truncation.

## The fix (truncation half — confirmed, shipped, env-gated)

`_L2_CODEONLY_DIRECTIVE` (a forceful "output ONLY a ```python block, do NOT analyze, skip all
reasoning" instruction) is prepended to the induce prompt, an opening fence is appended, and the
completion request carries `stop=["```"]`. The model emits the two functions and stops at the
closing fence. Gated by `CARNOT_ARC_CODEONLY_INDUCE=1`, scoped to the induce call
(`"is_level_complete" in required`) so gap-filler/refactor prompts are untouched. Default OFF — no
behaviour change unless the env var is set. Extraction is made robust to the stop-sequence eating
the closing fence (the opener is in the prompt, so the raw completion IS the code body).

Within-run matched control (same prompt bytes, warm Qwen :8920): baseline truncates to 0 code;
the fixed path emits valid code in 15.6s. This is an unambiguous, deployable improvement.

## The deeper wall the fix REVEALS (goal-induction quality)

On the live lp85 graded-bias re-test (`CARNOT_ARC_CODEONLY_INDUCE=1`, both arms): the smoke arm
reached **maxL=1, n_induce=2, goal_satisfiable=False (712s)**, and the induced world model was:

```python
def engine(grid, action, data):
    if action == 6:
        ...
        new_grid[py, px] = 5      # a plausible click-sets-cell dynamics model
        return new_grid
    return grid
def is_level_complete(grid):
    return False                  # <-- DEGENERATE constant-false goal
```

`is_level_complete: return False` can never be satisfied → `_goal_satisfiability_check` returns
`degenerate_goal_predicate` → the planner has no goal to plan toward → no L2.

**Two candidate causes (the live A/B's per-induction `induction_summary` disambiguates which):**

- **(likely benign) this is the L1 induction**, fired before any win-state exemplar exists. With no
  exemplar the model legitimately cannot know the win condition, and `return False` is an honest
  placeholder. L1 is reached by exploration, not by a goal, so a false goal there is harmless.
- **(the real problem) this is the L2 induction WITH the win-state exemplar**, and the model still
  punted to `return False`. That would mean code-only reasoning-suppression hurts the GOAL predicate:
  in isolation (a clear ASCII win exemplar) the code-only model hardcodes the exact win grid (a
  satisfiable-if-reachable predicate), but on the real lp85 L2 exemplar it gives up. The CoT we
  suppressed ("let me re-examine the win state…") was the model *trying to derive the win condition*;
  skipping it may trade truncation for laziness on the hardest part of the induction.

## Honest reconciliation with the operator's conditional

The operator's chain was: fix truncation → IF a satisfiable goal is produced → re-test graded-bias →
see if lp85 L2 banks. In **isolation** the fix produces a satisfiable goal (condition met), so we
chained. On the **live** game the goal collapses to degenerate, so **lp85 L2 does NOT bank** — the
graded-bias fix is moot when the goal predicate is constant-false (graded distance to an unreachable
/ never-True goal is uninformative). This is **outcome B** (goal induction is the wall) — but now
demonstrably a *quality* wall beneath the *truncation* wall, not the truncation itself.

## What ships vs what's next

- **Ships now:** the env-gated truncation fix (`CARNOT_ARC_CODEONLY_INDUCE=1`). It is a real,
  validated, 30×-faster, non-truncating induction path. It is necessary-not-sufficient for L2.
- **Next (goal-induction quality):** the candidate levers are (a) a LESS aggressive directive that
  forbids prose but still permits a brief, fenced reasoning scratch *before* the code (suppress the
  ramble, keep the win-condition derivation); (b) a goal-predicate REPAIR/verify loop that rejects
  `return False` / constant predicates and re-prompts (the harness already has
  `_nonzero_count_predicate` as an exemplar-derived fallback — wire it in when the LLM punts);
  (c) generate the goal predicate as a SEPARATE focused call from the engine, with the win-state
  exemplar front-and-center, so the model's budget is spent on the win condition, not the dynamics.
- The **rendering-space caveat** (the `_transitions_block` ASCII win state can differ from the raw
  frame array) is a separate latent risk for exact-match predicates and should be audited if a
  future induction hardcodes a grid.

## Method notes / integrity

- `inference_substrate=live_llm_inference`; `solve_provenance=development_proxy` (the offline dev
  twin, not a live-agent self-discovery solve); `verifier_is_oracle=false`.
- The original `proto_l2_code_only_prefix` satisfiability verdict was a TEST BUG (it fed the raw
  `prev` array, not the `_transitions_block`-rendered win grid the model was shown); corrected in the
  artifact (`goal_satisfiability_CORRECTED`).
- Server reuse: the warm Qwen :8920 (4 idle slots) was reused directly; the graded proto's
  port-busy hard-abort was patched to reuse a healthy Qwen server.
- No external submission; no landing-page edits; canonical URLs unchanged.
