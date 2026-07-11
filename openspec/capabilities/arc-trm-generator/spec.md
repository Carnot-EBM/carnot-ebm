# ARC PTRM Action-Sequence Generator Capability Specification

**Capability:** arc-trm-generator
**Version:** 0.1.0
**Status:** Implemented

## Overview

Defines the bounded Stage-1 PTRM-style ARC-AGI-3 action-sequence generator.
Stage 1 is an offline development proxy over public human-winning replay
trajectories. It validates a training substrate, held-out protocol, checkpoint,
and verifier-selected stochastic trajectories. It is not a hidden-game solve.

## Implementation Status

- 2026-07-11: Implemented in
  `python/carnot/agentic/arc_ptrm_stage1_generator.py` with focused tests in
  `tests/python/test_arc_ptrm_stage1_generator.py`. Scoped module coverage is
  100% (`464` statements, `0` missing). The implementation preserves the
  Stage-1 proxy claim boundary and does not modify `scripts/research_conductor.py`.

## Requirements

### REQ-ARC-PTRM-5574-1: Stage-1 Preconditions and Blocked Artifact

Experiment 5574 SHALL check CUDA 3090-class availability through Torch, replay
corpus hashes, disk budget, non-overlapping held-out games, and the
`results/trm_runs/DO_NOT_RELAUNCH` sentinel scope before training. If any
precondition is unavailable, the experiment SHALL write
`results/experiment_5574_ptrm_stochastic_generator_stage1.json` with a
terminal `honest_verdict` beginning `blocked_`, populated `preconditions`, and
no CPU or toy fallback.

### REQ-ARC-PTRM-5574-2: Human-Winning K-Window Dataset Contract

The repository SHALL build Stage-1 examples from the staged ARC Public Demo
human replay shards by grouping rows by `(env, guid)`, keeping only sessions
whose `level_progress` reaches 1.0, and producing K=8 or longer target action
windows. The split SHALL keep held-out games completely absent from training.
Each example SHALL condition on recent frame features, recent actions, and an
explicit intent/state embedding.

### REQ-ARC-PTRM-5574-3: PTRM Stochastic Recursion and Dynamic Halting

The Stage-1 generator SHALL implement stochastic recursive refinement by
injecting seeded Gaussian noise at each recursion step. It SHALL generate
multiple trajectories per input, expose ACT-style dynamic halting, and report
accuracy and energy by recursion depth so overthinking can be audited.

### REQ-ARC-PTRM-5574-4: Oracle-Distinct Verifier Selection and Controls

Trajectory selection SHALL use a Carnot verifier score that is independent of
the held-out target label and SHALL record `verifier_is_oracle=false`. The
experiment SHALL compare the PTRM arm with a matched non-recursive control and
a deterministic fixed-depth control, and SHALL run a positive-control overfit
or synthetic task before interpreting held-out results.

### REQ-ARC-PTRM-5574-5: Artifact, Checkpoint, and Claim Boundary

Experiment 5574 SHALL emit
`results/experiment_5574_ptrm_stochastic_generator_stage1.json` and a
checkpoint. The artifact SHALL include field principles for every headline or
gate field; `track=arc-trm-generator`; dataset hashes; held-out games;
leakage count; model architecture; parameter count; stochastic noise schedule;
trajectories per input; history/intent and dynamic-halting booleans;
recursion-depth metrics; overthinking curve; controls; verifier-selection
method; checkpoint path and sha256; resource receipt; `stage1_training_complete`;
`loo_verdict_reached`; `heldout_generalization_signal`;
`retire_trm_generator_line`; `no_level_solve_claim=true`;
`solve_provenance=development_proxy`; and
`inference_substrate=trained_ptrm_offline_development_proxy`.

### SCENARIO-ARC-PTRM-5574-DATASET: K-Window Split Has No Held-Out Leakage

Given staged replay rows containing won and non-won sessions, the Stage-1
dataset builder returns only won-session windows, all with K target actions,
and reports `leakage_count=0` when the configured held-out game set is absent
from training.

### SCENARIO-ARC-PTRM-5574-STOCHASTIC: Seeded Recursion Produces Selectable Diversity

Given the same input example and different trajectory seeds, the PTRM
recursion produces more than one candidate sequence, records halting depths,
and the verifier selector returns the highest verifier-scored trajectory
without reading the target.

### SCENARIO-ARC-PTRM-5574-ARTIFACT: Stage-1 JSON Is Honest and Complete

Given a completed bounded Stage-1 run, artifact validation confirms every
required field is present, all headline/gate fields have principles, the
checkpoint hash matches the written file, `verifier_is_oracle=false`, and
`no_level_solve_claim=true`.
