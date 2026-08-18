---
type: Claim
title: The induced goal predicate has never fired on a real win state
description: 0 of 31 win states across 49 engine evaluations; levels are won by exploration, not planning.
tags: [arc-agi-3, goal-predicate, binding-constraint]
status: stable
resource: /python/carnot/agentic/arc_executable_world_model.py
sources:
  - id: captures
    resource: /results/arc_cross_level_retention_20260817/capture_b4000_part1.json
    author: level-retention
  - id: gate-anatomy
    resource: /results/outer_loop_arc_induce_gate_anatomy_20260802.json
  - id: trust-gate-hole
    resource: /results/experiment_6012_hidden_state_trust_gate_hole.json
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: outer-loop/claude-opus-5, at: 2026-08-18T00:45:00Z }
  - { by: repair-extend, at: 2026-08-18T04:14:00Z }
---

# The claim

The induced goal predicate has never fired on a real win state: **0 of 31 across 49 engine
evaluations**.[^captures] Corroborated by `n_planned` 0 of 136 induction attempts[^gate-anatomy]
and 33 of 33 on-disk goal predicates being constant.[^trust-gate-hole]

`plan_in_model` terminates ONLY on `is_level_complete`, so a constant-False predicate guarantees
`None` at any budget.

# Why this null is interpretable

CLAUDE.md names an `n_win_states=0` corpus artifact where a zero denominator makes a null
meaningless. **This is not that**: the captures contain 31 genuine level-up transitions. State
the positive-example count before interpreting any ARC null.

Note also that `goal_fired_on_win_states` is a STRING of the form `"0/1"`. An integer cast reads
0/0 silently and produces a spurious clean result.

# Two scope corrections, both load-bearing

**Live path only.** The 0-of-136 characterises the LIVE default configuration. The offline
seeded-window harness plans on roughly half its rows and occasionally reaches real level-ups.
"The planner never returns a plan" is false as a general statement.

**A plan is not a recognised win.** In the offline corpus, 4 of 7 plans terminate on states the
predicate wrongly accepts, and `sp80 t0` reached a REAL level-up with `levelup_positive_recall`
0.0. So `n_planned > 0` alone can be satisfied by a hallucinated win.

# Consequence

Engine-quality levers are not levels levers. The measured exception is
`CARNOT_ARC_STRUCTURED_NAV=1` (opt-in), which took `tu93` L0 -> L1 with a correct-by-construction
model -- the only configuration measured to make the planner return a plan.

[^captures]: `capture_b4000_part{1,2}.json`, field `goal_fired_on_win_states`.
[^gate-anatomy]: `outer_loop_arc_induce_gate_anatomy_20260802.json`, `n_planned`.
[^trust-gate-hole]: `experiment_6012_hidden_state_trust_gate_hole.json`.
