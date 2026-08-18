---
type: Retracted Claim
title: Seeded repair gains nothing in the 0.6-0.95 marginal band
description: Withdrawn 2026-08-17 -- every confirming measurement shared one broken submission nudge.
tags: [arc-agi-3, repair, retracted, confounded]
status: deprecated
superseded_by: /okf/claims/seeded-repair-converts.md
sources:
  - id: nudge-fix
    resource: /python/carnot/agentic/arc_induction_tool_loop.py
    author: repair-extend
  - id: attribution
    resource: 51df5d093f
    author: outer-loop/claude-opus-5
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: repair-extend, at: 2026-08-17T22:00:00Z }
---

# The claim, as it stood

Seeded tool-loop repair produced no gain on cells whose pre-repair recall was between 0.6 and
0.95. Three seeded marginal-band repairs gained exactly zero. This was reported as
**independently confirmed three times** and used to justify the shipped `<0.6` recall gate.

# Why it was withdrawn

Every one of those measurements ran on an arm with a broken submission nudge.[^nudge-fix] With
the nudge fixed, the same band gives 4 of 4 repairs to zero visible mismatches; the broken arm
had produced 0 of 3.

The lesson generalises past this claim: **a repeated finding is not a replicated one when every
repeat shares a defect.** Three confirmations of a confounded arm are one confounded result.

# What replaces it

The gate's `<0.6` threshold still stands, but on different and narrower grounds — reachability,
not absence of gain. Of the three marginal cells that did gain once the nudge was fixed, only
one is reachable live: `sk48 t2` is trust-ACCEPTED so repair never fires on it, and `vc33 t0`
fails the evidence floor at `n_changing` 2.

# Do not

Re-derive this claim from the pre-fix artifacts. They are still on disk and still say zero.
Check the commit range of any arm before treating its result as evidence.[^attribution]

[^nudge-fix]: Tool-loop nudge fix, `arc_induction_tool_loop.py`.
[^attribution]: Commit `51df5d093f`, which records the fix's attribution.
