---
type: Retracted Claim
title: Tool-assisted induction repair is worth +0.054
description: Superseded 2026-08-17 -- true on its own single cell, too narrow to describe the lever.
tags: [arc-agi-3, repair, superseded, scope-error]
status: deprecated
superseded_by: /okf/claims/seeded-repair-converts.md
sources:
  - id: original
    resource: 651f157fcc
    author: outer-loop/claude-opus-5
  - id: seeded-wave
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/repair_widen
    author: repair-extend
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: repair-extend, at: 2026-08-18T01:45:00Z }
---

# The claim, as it stood

Tool-assisted induction works as repair rather than replacement, worth +0.054 for one extra
pass.[^original]

# Why it was superseded

The number was measured UNSEEDED while the shipped path SEEDS -- `arc_competition_agent.py`
passes `seed_engine_code=old_code`, the failed engine the recall gate just measured. So the
figure did not describe the configuration that ships.

Note the distinction: this is **not a retraction**. The number survives on its own corpus -- it
was one cell, `sp80 t1`, and the seeded path reproduces that cell's conversion exactly. It is
withdrawn for being too narrow to characterise the lever, not for being wrong.

# What replaces it

A conversion rate, not a mean. See [seeded repair converts catastrophic cells](/okf/claims/seeded-repair-converts.md).
The arithmetic mean across the four seeded catastrophic cells is about +0.44 and **describes no
cell in the set** -- the distribution is bimodal, two convert outright and two move essentially
not at all. Reporting the mean would have been the more flattering and less true option.
