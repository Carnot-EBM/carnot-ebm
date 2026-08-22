# Story 6610: Fix the zero-world-model induction failure on the live ARC path

## Context

A supervisor A/B baseline (2026-08-21, five public games, budget 2000,
Qwen3.8-27B) produced ZERO world models. Every induce call hit the
4096-token output cap and failed. The cap default was validated for the
retired 9B generator and never moved through two generator swaps. The
skip record conflated two causes and discarded the induce failure note.
The model label named the retired 9B while the loaded weights were the
27B.

## Spec Traces

- REQ-ARC-WMTE-6610 / SCENARIO-ARC-WMTE-6610-1, -2
- REQ-ARC-WMTE-6620 / SCENARIO-ARC-WMTE-6620-1, -2, -3
- REQ-ARC-WMTE-6630 / SCENARIO-ARC-WMTE-6630-1, -2

## Acceptance Criteria

- [x] Skip record splits `proposer_failed` from `missing_plan_start_grid`
      and carries the induce note.
- [x] Induce budget/timeout defaults derive from the live generator pin
      and match the scored kernel's env pins.
- [x] Per-request budget clamps to the running server's observed pool.
- [x] Model label derives from the weights actually loaded.
- [ ] Live re-run on ar25/tr87/tu93/sp80/re86 produces world models
      (before: 0).

## Status

In progress (2026-08-21).
