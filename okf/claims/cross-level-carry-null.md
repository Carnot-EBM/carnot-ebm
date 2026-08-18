---
type: Retracted Claim
title: Carrying a verified engine across a level boundary buys nothing
description: Withdrawn 2026-08-17 -- the lever never fired, so the A/B measured nothing.
tags: [arc-agi-3, cross-level-carry, retracted, vacuous-null]
status: deprecated
superseded_by: /okf/claims/carried-engines-do-not-transfer.md
sources:
  - id: carry-ab
    resource: /results/arc_cross_level_retention_20260817/carry_ab_full.json
    author: level-retention
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: levelwin-ideas2, at: 2026-08-18T00:50:00Z }
---

# The claim, as it stood

A paired A/B of cross-level engine carry measured a mean of -0.6 actions out of ~1450 (0.04%)
with zero cells reaching a different level, and was reported as a clean null: carrying a
verified dynamics engine across a level boundary buys nothing.

# Why it was withdrawn

`carry_fires` is **0 in all 9 paired cells**.[^carry-ab] The lever engaged and never once fired.
The -0.6 action delta is RNG-consumption noise, not an effect. The A/B did not measure whether
carrying an engine helps; it measured a guard that never opened.

This is the **vacuous null** class: a headline number that looks like evidence of absence and is
actually absence of evidence. Nothing in the artifact's summary fields distinguishes the two.

# What the attempt rows actually show

The ON-arm rows do carry a real finding, which the retracted claim obscured: the guard requires
`min_accuracy_bar` 1.0 on the new level, carried engines scored `verify_accuracy` 0.25-0.5, and
each attempt logged `carried_engine_failed_new_level_verification` after three deferrals for
`insufficient_new_level_evidence`.

# Do not

Lower the 1.0 accuracy bar. Any defensible lower value still rejects a 0.25-accuracy engine, and
admitting one would be worse than carrying nothing. Transfer is the blocker, not the bar.

[^carry-ab]: `carry_ab_full.json`, field `carry_fires`, all 9 cells.
