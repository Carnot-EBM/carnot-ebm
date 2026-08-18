---
type: Claim
title: Seeded repair converts about half of catastrophic-recall cells
description: 2 of 4 seeded catastrophic cells convert outright to recall 1.0; the other 2 return their seed unchanged.
tags: [arc-agi-3, repair, engine-quality, shipped]
status: stable
resource: /python/carnot/agentic/arc_recall_gated_resample.py
sources:
  - id: wave
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/repair_widen
    author: repair-extend
  - id: enable
    resource: ec6f09b71a
    author: outer-loop/claude-opus-5
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: repair-extend, at: 2026-08-18T01:45:00Z }
---

# The claim

With the nudge fixed, seeded tool-loop repair on catastrophic-recall cells (n=4) converts 2 of 4
outright to recall 1.0 / accuracy 1.0, flipping the trust layer from reject to accept. The other
2 return their seed unchanged.

- `sb26 t0` 0.0 -> 1.0, tail 1.0, memorisation scan clean -- fully out-of-sample verified
- `sp80 t1` 0.294 -> 1.0, scan clean, **no scoreable tail exists** on that window (its single
  tail row IS the excluded level-up row -- checked against the instrument, not assumed)
- `tu93 t0` +0.066, accuracy still 0.0
- `tr87 t0` +0.0 exactly

Reported as a conversion rate on purpose. The arithmetic mean of +0.44 describes no cell.

# Cost of a miss, accepted

~9-11 minutes and one of two per-game slots, because the counter increments when the repair
FIRES, before the outcome exists. Bounded by the turn cap, which shipped at 8.

# Safety, confirmed empirically at 7 of 7

The loop is seeded with the failed engine and cannot return a candidate with more visible
mismatches, so no run regressed below its seed. A trust-ACCEPTED engine is never re-rolled --
`decide_resample` returns `downstream_accepted_engine` first -- so this can only act on engines
the pipeline was going to discard.

# Scope: engine quality, NOT levels

Per [the goal-predicate claim](/okf/claims/goal-predicate-never-fires.md), levels are won by
exploration, not planning. Expect better engines. Do not expect the score to move on this alone.

# Shipped

Enabled on the scored path at the 0.6 gate with turn cap 8,[^enable] set in the submission
kernel rather than as a code default: the evidence is scored-path evidence, and the conductor's
trade-off at ~10 minutes per miss is unmeasured.

[^enable]: Commit `ec6f09b71a`.
