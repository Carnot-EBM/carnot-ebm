---
type: Retracted Claim
title: A near-noise seed anchors induction to a dead approach
description: Withdrawn 2026-08-17 -- refuted 6/6 paired; its counterexample was in the same table that proposed it.
tags: [arc-agi-3, repair, retracted, single-cell-anomaly]
status: deprecated
superseded_by: /okf/claims/seeded-repair-converts.md
sources:
  - id: skip-wave
    resource: 86a9ef38f2
    author: repair-extend
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: repair-extend, at: 2026-08-18T00:05:00Z }
---

# The claim, as it stood

The seed helps when it is nearly right and hurts when it is nearly noise. Motivating evidence:
`tu93 t0` reached recall 1.0 / accuracy 0.727 unseeded, and 0.0656 seeded on the identical cell.
A proposed lever followed: skip the seed when its visible mismatches equal the visible row count.

# Why it was withdrawn

Two independent reasons, and the first should have been caught before the claim was made.

**The counterexample was already in the table that proposed it.** `sb26 t0`'s seed also had
recall 0.0, and it converted to 1.0. The record on near-noise seeds is 1 of 2, not a mechanism.

**Six paired cells then refuted it outright.**[^skip-wave] Seeded to skip: `tu93 t0`
0.0656 -> 1.0 (skip), `sp80 t1` tie, `sb26 t0` 1.0 -> 0.9906 (seed), `ar25 t2` **1.0 -> 0.0**
(seed, catastrophically), `vc33 t0` 1.0 -> 0.9549 (seed), `tr87 t0` marginal. The broad trigger
would forfeit `ar25 t2` entirely -- and that seed is 93% cell-right while wrong somewhere on
every row, so "mismatches equal visible rows" was never a signal for noise at all.

# What survives

`tu93 t0` is recorded as a **single-cell anomaly**, not a lever. Its skip win is real and fully
out-of-sample -- tail recall 1.0, tail accuracy 1.0, memorisation scan clean. It is a genuine
engine that no statable trigger can reach.

A secondary finding outlived the claim and strengthens the shipped default: seeding is also
**2.6x cheaper** -- 52.5 minutes against 134.1 across the same six cells, because blank-page runs
grind the turn cap where seeded runs converge by turn seven.

[^skip-wave]: Commit `86a9ef38f2`, six paired cells, same code state and pinned seeds.
