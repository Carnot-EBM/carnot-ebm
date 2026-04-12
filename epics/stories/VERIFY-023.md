# Epic: VERIFY-023 - Exp 223 Held-Out Self-Learning Replay

**Status:** Completed
**Goal:** Build a deterministic Exp 223 replay workflow that reconstructs the
checked-in live Exp 219 / 220 / 221 traces, evaluates held-out slices in
chronological order, compares `no_learning`, `tracker_only`, and
`tracker_plus_memory`, and writes `results/experiment_223_results.json` with
task metrics, false-positive budget accounting, retrieval hit rates, and
per-model transfer effects.
**Rationale:** Exp 222 proved that live traces can be normalized into memory,
but its replay precision was only 12.6% and it did not show a held-out benefit.
Exp 223 needs an honest replay benchmark where all gains come from prior live
updates only, so Carnot can test whether the current tracker/memory path helps
future live cases instead of merely summarizing the same corpus.

## Stories
- [x] Add `REQ-VERIFY-033` through `REQ-VERIFY-035` and
  `SCENARIO-VERIFY-033` through `SCENARIO-VERIFY-035` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for deterministic held-out slicing, chronological
  replay, tracker gating, memory reuse, transfer accounting, and the Exp 223
  script/artifact contract
- [x] Implement the Exp 223 replay module and
  `scripts/experiment_223_self_learning_replay.py`
- [x] Execute the workflow and write `results/experiment_223_results.json`
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and any required metrics metadata
