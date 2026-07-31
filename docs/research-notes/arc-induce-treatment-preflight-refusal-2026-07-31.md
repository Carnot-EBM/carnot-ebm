# Phase 3 pre-flight: the wired induce treatment changes the engine on every game and the agent's actions on none

**Date:** 2026-07-31 · **HEAD at run:** `c39f4bde6` · **Artifact:**
`results/outer_loop_arc_phase3_preflight_wired_induce_treatment_20260731.json`

## Verdict

**REFUSE the 12-cell banked-levels grid.** Not "too few cells perturb" — **zero** do.
Four of four comparable A/B pairs produced **byte-identical action traces**. A pair whose two
arms emit the same actions cannot differ on any endpoint computed downstream of those actions,
so no outcome of that grid could reach `alpha=0.05`.

Phase 2 refused the same grid on an *analytic bound* (`p_reach <= 0.238`) and left open which of
two channels was real, noting that "which channel is real is what a live pre-flight buys, and it
is worth buying — it picks 16 cells over 68." **It bought something better than a cell count: the
answer is neither channel.** The treatment does not reach the endpoint at all, so no grid size
licenses it. 16 live cells, ~7243 s of GPU wall clock.

## What was compared

Both arms are the **same commit, same repo path, same assets, same interpreter, same
llama-server process**. They differ only in two environment variables read inside `induce()`:

| arm | `CARNOT_ARC_INDUCE_REPEAT_PENALTY` | `CARNOT_ARC_INDUCE_DEFECT_REASKS` |
|---|---|---|
| `ctl` / `ctlb` | `1.0` (restores the pre-wiring payload byte-for-byte) | `0` |
| `trt` / `trtb` | *unset* — the shipped default, 1.1 | *unset* — the shipped default, 1 |

`trt` leaves the knobs unset on purpose: an arm that re-specified `1.1` explicitly would still
pass if the wiring were reverted, which is exactly what these cells must be able to notice.

Both A/A replicates are measured, not argued. A single floor witnesses only one of the two arms,
and the treatment **changes the sampler** — precisely the kind of change that can alter how
repeatable an arm is.

## The evidence

| game | ctl | trt | A/B | ctl heldout | trt heldout |
|---|---|---|---|---|---|
| ft09 | 27 actions, 0 plans | 27 actions, 0 plans | **IDENTICAL** | 0.5 | 0.5 |
| tn36 | 31 actions, 0 plans | 31 actions, 0 plans | **IDENTICAL** | **0.0** | **1.0** |
| vc33 | 18 actions, 0 plans | 18 actions, 0 plans | **IDENTICAL** | 1.0 | 1.0 |
| sc25 | 27 actions, 0 plans | 27 actions, 0 plans | **IDENTICAL** | 0.875 | 0.875 |
| tu93 | 27 actions, 0 plans | 27 actions, 0 plans | *excluded — crossed a server process* | 0.0 | 0.0 |
| lp85 | truncated @1500 s | truncated @1500 s | MISSING | — | — |

Both `tu93` cells ran to completion, but its control landed on server PID `1532116` and its
treatment on `1562595` (the original server was replaced mid-session by a concurrent agent), so
the pair is confounded by sampler variance and the guard excludes it. Reported for completeness
and **counted nowhere in the verdict**: that pair was *also* 27/27 byte-identical. It points the
same way as the four scored cells, but a confounded pair is not evidence and is not treated as
any.

### The treatment was active — proven three ways, so this is not "the flag didn't apply"

1. `induce_repeat_penalty_effective` reads back **1.1** on every treatment cell and **1.0** on
   every control cell, through the shipped accessor `e3._induce_repeat_penalty()` — not echoed
   off `os.environ`.
2. **16 cells produced 16 distinct engines.** No two `world_model.py` files match.
3. On **tn36** the treatment's engine scored held-out accuracy **1.0** where the control's
   scored **0.0** — and both arms took byte-identically the same **31 actions**.

That third row is the whole result in one line: a maximal induce-quality improvement, and the
agent does not do one thing differently.

## Where the causal chain actually breaks

`n_plans_found` is **0 in every completed cell but one**. The single cell whose action trace
diverged — ft09, `ctl` against its own replicate `ctlb` — is the single cell where
`n_plans_found` differed (0 vs 1).

So the action stream moves **when and only when a plan is found**; the treatment never changed
whether a plan was found; and run-to-run noise changed it once, i.e. **more than the treatment
did**. Improving induce *quality* cannot move banked levels while the planner converts ~0 of
those engines into a plan.

**The next target is `plan_in_model` / the induce→plan conversion, not further induce-payload
tuning.**

## The re-ask half was never exercised

`induce_defect_reasks_observed == 0` in all 16 cells. The gate re-asks only on an engine that
fails static validation, and every live induction here passed. Phase 1's 22-of-36
defective-accept rate **did not reproduce on the live path**.

This is *"the tier never fired"*, not *"the tier did not help"*. The re-ask arm is **untested**
by this probe; its only support remains Phase 1's 2 of 13 paired wins.

## Instrument findings

**The seed does not make induce deterministic.** `ctl` and `ctlb` ran at an identical sampler
seed, identical config, identical code, and the *same server process* — and still produced
substantively different engine source (different generated logic and commentary, not a
timestamp), and on ft09 a different plan outcome. `CARNOT_ARC_GENERATOR_SEED` is necessary but
not sufficient. This corroborates the prior session's n=2 observation with more cells.

**A server-process guard was added to the scorer.** The sampler seed does not hold across
llama-server processes, so a pair whose arms crossed one is confounded by sampler variance in
*either* direction. This was not hypothetical: both original servers were replaced mid-session
by a concurrent agent. Every scored pair was audited as intra-process; `tu93` was excluded by
this guard rather than scored.

**A bug in this harness was found and fixed mid-run.** `cell.py`'s `ARM_ENV` had no `trtb`
entry, so `vc33/trtb` exited `rc=1` with no record. Fixed and backfilled. Recorded because a
gap left by the harness's own defect is not a datum.

## Known limits — do not over-read

- **4 comparable cells, not 6.** Both `tu93` cells ran but on different server processes;
  `lp85` truncated on both arms. Both are missing observations, never zeros. They were
  *attempted*, not skipped.
- **0 of 4–5 does not mean "never".** The one-sided 95% upper bound on the perturbation rate is
  **0.451**. The refusal rests on the **mechanism** (`n_plans_found` 0 across arms) at least as
  much as on the count — the count alone would be thin.
- The two omitted games are the two **slowest**, i.e. the deepest runs where a treatment acting
  through the engine would have had the most room to express itself.
- A REFUSE means only that *this grid* cannot produce a finding. It does **not** retract Phase
  1's induce-level effect, which was real and stands.

## What was not done

No flag was changed on the basis of this run. No banked level was claimed. No scored or online
ARC game was played and nothing was submitted — every cell ran against the offline arcade via
`arc_solver_kit.offline_arcade()`.
