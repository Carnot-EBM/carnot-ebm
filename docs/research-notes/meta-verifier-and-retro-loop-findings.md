# Meta-verifier over our own process (#5) + the retro-loop recommendation

**2026-06-07, outer-loop session.** Answers the operator's retrospective question —
*"should we add a retrospective to the end of every milestone?"* — with data from a
discriminator run over the full 688-milestone / 7,896-task history
(`scripts/experiments/meta_verifier_experiment_selection_draft.py`).

## The short answer to the retro question

**We already run one every milestone** — 307 `results/operational_retro_*.json`
files, fired by the conductor at milestone close. The problem is not that it's
missing; it's that it is **write-only**: process metrics + a prose `meta_reflection`
that **nothing mechanically consumes**. A retrospective that never changes the next
plan is meta-churn. The fix is to **close the loop**, not add another retro.

## What the meta-verifier found (v1 -> v3)

Joining the task list to each artifact's `honest_verdict`:

- **v1/v2 — 32% of tasks carry a blocked/skipped/flagged verdict; 44% of titles are
  breadth-churn scope** (capstone / archive / matrix / vN+1 re-measurement). Per-scope
  wasted-rate lift vs base: `hardware 1.5x`, `infra/harness 1.4x`, `churn-aggregate
  1.3x`, `build/train 0.82x`, **`science-probe 0.71x`** (depth science wastes the
  *least*). A title-keywords classifier predicts "wasted" at only AUROC ~0.57 — weak.
- **v2 — the waste TREND over the project's history** (milestone quartiles):
  `18.6% -> 39.4% -> 41.2% -> 35.8%`. Waste **rose** as hardware/harness/cascade
  complexity grew and only modestly recovered; the accreting discipline rules slowed
  but did not reverse it to early levels.
- **v3 — but most "waste" is NOT preventable.** Decomposing the wasted verdicts:

  | category | % of wasted | preventable by a cheap pre-check? |
  |---|---|---|
  | honest_gate_negative (ran/blocked on an acceptance gate) | 41.0% | **No — it's a RESULT** |
  | other_blocked (genuine experiment failures) | 47.6% | mostly no |
  | preventable_precondition (board/cache/cred/cuda) | 9.7% | **yes — a live pre-flight** |
  | preventable_cascade_skip (poison-test / 3-fail) | 1.4% | yes |
  | preventable_doomed_rerun | 0.3% | yes |

  **Only ~11% of wasted verdicts (= 2.5% of ALL tasks) are cheaply preventable.**
  The "32% wasted" over-counts: 41% are honest negative *results*, not avoidable burn.

## The honest conclusion

There is **no silver-bullet meta-verifier**. A live precondition oracle would reclaim
only ~2.5% of task wall-clock (real, not transformative), concentrated in hardware/
infra tasks (reachability/readiness failures). The **bigger lever is allocation**: 44%
of tasks are breadth-churn re-measurement, which the Depth-Over-Breadth rule already
targets — and depth science is the *safest* class (0.71x waste). The meta-verifier's
value is **steering allocation toward depth**, not pre-filtering experiments (most run
and fail honestly).

## The retro-loop recommendation (operator decision — touches `research_conductor.py`)

Wire the retro's **structured** output into the planner as required reading, the same
way Overdue-Priority / Depth-Over-Breadth already force consumption. Two cheap signals
+ one thin field:

1. **Breadth-churn fraction** of the just-closed milestone (from the title-scope
   buckets). If it exceeds a threshold, the planner must reallocate toward depth.
2. **Per-scope preventable-precondition rate** (hardware/infra). Gate those task
   classes on a **live pre-flight check** (board reachable? upstream harness READY?
   model cached?) *before* scheduling — the ~2.5% reclaim, concentrated where it lives.
3. **A thin "thesis-movement" field** on the retro: *did this milestone advance the
   headline claim, close a G-gate, or test a load-bearing-unproven link? (yes/which)*
   — the north-star §1 test, currently absent from the operational (process-only) retro.

None of this is transformative on its own; together they convert a write-only retro
into a steering signal. The precondition pre-check is the one concrete mechanical
win; the rest is allocation discipline the rules already encode but don't measure.

## Status

`#5 v1/v2/v3` are CPU-only analysis drafts (observe-only; not conductor-wired). The
artifact (`results/experiment_meta_verifier_selection_draft.json`) carries the full
breakdown. Wiring any of the three signals into the conductor is an operator decision.
