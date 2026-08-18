---
type: Retracted Claim
title: Pooling the action budget across games recovers ~56% of wasted actions
description: Withdrawn 2026-08-18 -- actions are per-game and non-transferable; there is no pool.
tags: [arc-agi-3, budget, retracted, impossible]
status: deprecated
superseded_by: /okf/claims/goal-predicate-never-fires.md
sources:
  - id: agent-counter
    resource: /home/ianblenke/arc-sota-refs/ARC-AGI-3-Agents/agents/agent.py
    author: carryover-plan
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
verified:
  - { by: outer-loop/claude-opus-5, at: 2026-08-18T04:05:00Z }
---

# The claim, as it stood

14 of 25 games complete zero levels while consuming ~56% of all actions, because the grace
early-stop only arms after a level-up. Pooling that budget -- strictly additively, so no game is
cut short -- was ranked the single highest-value lever available.

# Why it was withdrawn

**There is no pool to reclaim.** `action_counter` is a per-instance attribute on `Agent`
(`agents/agent.py:25`) and the swarm constructs one instance per game.[^agent-counter] Actions
are per-game and non-transferable. Reclaiming what the zero-level games spend gives it to
nobody; cutting `tu93`'s 1,244 reset-replay actions returns them to `tu93`, which already levels.

Worse, actions after the last level-up are charged to an incomplete level and score 0.0
regardless, so freed actions only pay when they sit inside a level that is later completed.

# What survives

A stagnation stop is still worth building, but for **wall clock and RAM only**. Selling it as a
score fix would repeat this error. Wall clock is the currency that is actually short: roughly 100
induce calls at ~1053s each against an 11.5h cap, where a fired timeout degrades that game to
LLM-off silently.

[^agent-counter]: `agents/agent.py:25`, `action_counter: int = 0` as a class attribute.
