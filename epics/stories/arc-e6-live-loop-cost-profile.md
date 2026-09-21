# ARC E6 live-loop cost profile

**Status:** Complete
**Requirement:** REQ-ARC-WMTE-7490
**Experiment:** 7490

## Goal

Reduce existing ARC cost evidence. Do not run a model or game.

## Scope

- Inventory provenance, induction, request, phase, and backend-usage artifacts.
- Exclude adversarial artifacts and non-current models from numeric support.
- Reconcile exclusive phase time and backend token use.
- Report cold and warm coverage.
- Publish numeric shares only after 30 complete episodes across 10 games.

## Result

The existing clean current-model evidence has 26 complete episodes across 10
games. The episode gate is short by four. Candidate selection, induction and
generation, and supervisor time are observed. World-model verification,
planner, and environment time are not separate. The artifact is coverage only.
It makes no speed claim.

## Verification

The synthetic control recovers one injected delay and one injected token count
exactly. Focused tests use `tmp_path`. The real entrypoint is a CPU-only replay
of existing files.
