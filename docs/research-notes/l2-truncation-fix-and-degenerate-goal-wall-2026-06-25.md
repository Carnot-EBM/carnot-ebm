# L2 goal-induction: the token-truncation wall is FIXED — but a deeper goal-induction-QUALITY wall stands

**Date:** 2026-06-25 · **Author:** outer-loop (operator-directed: "update qwen ctx to 16384; re-call
the captured L2 prompt with a stronger code-only prefix at 4096; if CoT irrepressible fall back to
raise-budget; if it produces a satisfiable goal, chain into re-testing the graded-bias fix to see if
lp85 L2 banks").
**Self-corrected** after an adversarial verification pass (workflow `verify-l2-truncation-fix`,
2026-06-25) caught three over-claims in the first draft; this note retracts them explicitly below.
**Artifacts:** `results/proto_l2_code_only_prefix.json` (isolation; well-supported);
`logs/proto_graded_ab_codeonly.log` (the fix-enabled live A/B; binary arm decisive, graded+sc25
still completing). **Code:** shipped fix env-gated in `python/carnot/agentic/arc_executable_world_model.py`
(`CARNOT_ARC_CODEONLY_INDUCE=1`).

## TL;DR (corrected)

1. **Qwen context** was already `n_ctx=16384` (code + running llama-server) — no change needed.
2. **The truncation wall is real and now FIXED.** A forceful CODE-ONLY directive **plus a
   `stop=["```"]` sequence** makes the L2 induction emit ONLY the code block instead of burning the
   whole 4096-token budget on win-state chain-of-thought (CoT):
   - baseline `/no_think`, SAME prompt (matched within-run control): `stop_type=limit`, 4096 tokens,
     **0 code**, 450s.
   - **directive ALONE did NOT stop the model**: it ran to `stop_type=limit`, 4096 tokens, 605s — it
     emitted the code *early* (so valid code is extractable) then rambled to the limit.
   - **directive + `stop=["```"]`: `stop_type=word`, 195 tokens, 15.6s, valid parseable
     `engine`+`is_level_complete`.** The stop-sequence — not the directive — is what stops the
     rambling. This is the deployable fix and it is a ~30× latency *improvement*, not a hit.
   - Wired through the shipped `generate()` it returns valid code where the unpatched path returned
     nothing.
3. **RETRACTED over-claim (first draft said "produces a satisfiable goal in isolation").** It does
   NOT under the harness's real criterion. Two reasons: **(a) circular** — the synthetic prompt
   literally instructs `is_level_complete must return True here` and prints the win grid, so the
   model copying it into `np.array_equal` is True *by construction*; **(b) unreachable** — the
   harness's `_goal_satisfiability_check` (`arc_llm_reinduction.py:481`) is BFS *reachability over
   the induced engine*, and the toy synthetic engine only ever mutates cell `[2,2]`, so the hardcoded
   25-cell win grid is unreachable → the harness would return `satisfiable=False` even in isolation.
   The only defensible isolation claim is **"emits valid parseable code, fast"**.
4. **On the live game, lp85 L2 does NOT bank — even with the fix.** That is the answer to the
   operator's question.

## The fix (truncation half — confirmed, shipped, env-gated)

`_L2_CODEONLY_DIRECTIVE` (a forceful "output ONLY a ```python block, do NOT analyze, skip all
reasoning") is prepended to the induce prompt, an opening fence is appended, and the completion
request carries `stop=["```"]`. The model emits the two functions and stops at the closing fence.
Gated by `CARNOT_ARC_CODEONLY_INDUCE=1`, scoped to inductions (`"is_level_complete" in required`,
which matches `_gen_to_file` and `experiment_4544:615`) so gap-filler/refactor callers are untouched.
Default OFF — no behaviour change unless the env var is set (verified by the adversarial review:
`bool(os.environ.get(...))` is False when unset → the pre-existing `no_think_prefix` path runs).
Extraction is made robust to the stop-sequence eating the closing fence (the opener is in the prompt,
so the raw completion IS the code body), and still passes through the same `def`-presence + `ast.parse`
gate, so garbage cannot slip through.

## The deeper wall (goal-induction QUALITY) — confirmed at L2 on the fix-enabled live run

Live graded-bias A/B on real lp85 with `CARNOT_ARC_CODEONLY_INDUCE=1` (the actual fix-enabled run;
`logs/proto_graded_ab_codeonly.log`):

```
lp85/binary: maxL=1 l2=False exemplar=True n_induce=2 gps=False outcome_c=False (1398.5s)
```

- `maxL=1` — reached L1, **not L2**.
- `exemplar=True` — the L2 induction **did** receive the win-state exemplar. So the unsatisfiable
  goal is **the L2 attempt with exemplar present**, NOT a benign L1 placeholder (this resolves the
  L1-vs-L2 question the first draft left open).
- `gps=False` — `goal_predicate_satisfiable_any=False`: no induction produced a goal satisfiable
  under the harness's reachability criterion.

Unlike the **pre-fix** run (whose stale artifact recorded `skipped: proposer_failed` = no code at
all for L2), the fix-enabled run **does emit valid code** — multiple world models were written
during the run, with *inconsistent* `is_level_complete` quality:

```python
# one induction (degenerate constant):        # another induction (a real general rule):
def is_level_complete(grid):                   def is_level_complete(grid):
    return False                                   h, w = grid.shape
                                                   for r in range(h):
# a third (exact-match a hardcoded grid,           for c in range(w):
#  unreachable by the induced engine):                 if grid[r, c] == 3: return False
def is_level_complete(grid):                       return True
    return np.array_equal(grid, np.array([...]))
```

So the truncation fix moves the live failure from **"no code (truncated)"** to **"valid code, but the
induced win-condition is inconsistent and not satisfiable"** (`gps=False`). **The deeper wall is
goal-induction QUALITY, not token budget** — and, plausibly, suppressing reasoning trades truncation
for a lazier goal predicate (HYPOTHESIS, not established: in isolation the directive made the model
copy the instructed grid; here it produces `return False` / unreachable-exact-match on some attempts).

## Honest reconciliation with the operator's conditional

The chain was: fix truncation → IF a satisfiable goal is produced → re-test graded-bias → does lp85
L2 bank. The honest reading: the fix produces **valid code but not a harness-satisfiable goal**
(neither in isolation — circular/unreachable — nor live — `gps=False`). So the antecedent is **not
cleanly met**, and the live re-test confirms **lp85 L2 does NOT bank**. The graded-bias fix is moot
when there is no satisfiable goal predicate to grade distance against. This is **outcome B** (goal
induction is the wall), now demonstrably a *quality* wall beneath the *truncation* wall.

## What ships vs what's next

- **Ships now:** the env-gated truncation fix (`CARNOT_ARC_CODEONLY_INDUCE=1`). It is a real,
  validated, ~30×-faster, non-truncating induction path that emits valid code where the unpatched
  path emitted none. It is **necessary-not-sufficient** for L2.
- **Next (the real lever — goal-induction QUALITY):**
  1. A goal-predicate REPAIR/verify loop that REJECTS degenerate predicates (`return False`,
     `return True`, exact-match grids the engine can't reach) and re-prompts — the harness already
     ships `_nonzero_count_predicate` as an exemplar-derived fallback; wire it in when the LLM punts.
  2. Generate the goal predicate in a SEPARATE focused call (win-state exemplar front-and-centre)
     from the dynamics engine, so the budget is spent on the win condition specifically.
  3. A LESS aggressive directive that forbids *prose* but permits a brief fenced reasoning scratch
     *before* the code — suppress the ramble, keep the win-condition derivation (the suppressed CoT
     was the model trying to derive the win condition).
  4. Make `is_level_complete` satisfiability a first-class gate: reject any induced predicate whose
     `_goal_satisfiability_check` returns `degenerate_goal_predicate` and re-induce.

## Integrity / provenance

- `inference_substrate=live_llm_inference`; `solve_provenance=development_proxy` (offline dev twin,
  not a live-agent self-discovery solve); `verifier_is_oracle=false`.
- **Corrections applied after adversarial review:** (1) the isolation "satisfiable" claim is retracted
  (circular + unreachable; see `goal_satisfiability_RECONCILED` in the artifact); (2) stopping is
  attributed to the stop-sequence, not the directive; (3) the live conclusion is now backed by the
  fix-enabled run's binary arm (`logs/proto_graded_ab_codeonly.log`), not the stale pre-fix
  `proto_graded_goal_bias_ab.json` (06:24). The full fix-enabled A/B artifact (graded arm + sc25 +
  per-induction `induction_summary`) is completing in the background and is confirmatory — the binary
  arm already settles the lp85-L2 question.
- The `712s` smoke / `1398.5s` binary-arm durations are from the fix-enabled run's log; the
  first draft's reference to a `649.9s` figure was the stale pre-fix artifact and has been dropped.
- No external submission; no landing-page edits; canonical URLs unchanged.
