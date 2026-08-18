---
type: Claim
title: Arm was confounded with GPU lane in the paired A/B design
description: GPU 1 is 8-11% faster than GPU 0 on this workload; a dead lever is what exposed it.
tags: [arc-agi-3, methodology, confound, paired-ab]
status: stable
sources:
  - id: goal-ab
    resource: /home/ianblenke/.claude/jobs/ad0c053d/tmp/repair_widen/goalab
    author: repair-extend
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T12:40:00Z }
verified:
  - { by: repair-extend, at: 2026-08-18T12:30:00Z }
---

# The finding

The paired A/B ran control on GPU 0 and treatment on GPU 1 for the whole corpus. Treatment was
faster on 10 of 11 completed pairs, mean ratio 0.922, median 0.893. **GPU 1 is 8-11% faster than
GPU 0 on this workload**, consistently across all four games. Arm and lane were confounded, and
nothing in the design would have separated them.

# Why it was caught, which is the interesting part

Only because the lever was DEAD. The two arms produced byte-identical engines and zero re-asks
fired, so no treatment effect could exist -- which meant the 8-11% timing difference had to come
from somewhere else. Had the lever been live, that same asymmetry would have looked exactly like
a plausible mechanism (the flags letting the model abandon a dead goal early instead of grinding),
and it would likely have been reported as one.

So an invalid run exposed a defect that a valid run would have hidden.

# The dependent artifact

Control's `tr87 t1` hit the 3600s induce timeout while treatment's identical-seed run completed
in 2875.8s. At the lane penalty control was heading for 3100-3300s and its retry ladder pushed it
past the cliff. That is a lane artifact, not a dropout and not a treatment effect.

# Required of any re-run

Alternate or swap GPU assignment mid-corpus so arm and lane are not confounded. This requirement
did not exist in the original design.
