---
type: Claim
title: Carried engines do not transfer across a level boundary
description: Carried engines score 0.25-0.5 accuracy on the next level; the boundary is not where the loss is.
tags: [arc-agi-3, cross-level-carry, transfer]
status: stable
sources:
  - id: carry-ab
    resource: /results/arc_cross_level_retention_20260817/carry_ab_full.json
    author: level-retention
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: levelwin-ideas2, at: 2026-08-18T00:50:00Z }
---

# The claim

Engines carried across a level boundary score `verify_accuracy` 0.25-0.5 on the new level,
against a guard bar of 1.0.[^carry-ab] They do not transfer.

This is what the carry A/B actually established -- **not** that carrying knowledge is worthless,
which is the [vacuous null that was withdrawn](/okf/claims/cross-level-carry-null.md).

# Do not tune the bar

A perfection-level bar looks like the obvious culprit. It is not. Any defensible lower value
still rejects a 0.25-accuracy engine, and admitting one is worse than carrying nothing.

# What else might survive a boundary

Weak, on current evidence. The HUD mask already persists per-game. The hazard model is real but
small (-88 actions on `tu93`). Verified sub-paths and the go-explore archive die to the
no-teleport constraint: the live environment cannot restore archived states, so every navigation
costs charged actions. The recommendation on 2026-08-18 was not to spend effort here -- the
boundary is not where the loss is.

[^carry-ab]: `carry_ab_full.json`, ON-arm `carry_attempt_rows`.
