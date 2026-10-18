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
- 2026-07-13: **Wiring bug found and fixed** (REQ-ARC-PTRM-5600-1).
  `generate_trajectories` seeded every trajectory from `_base_action_logits`
  (an untrained action-frequency heuristic) regardless of whether a trained
  `PTRMActionSequenceGenerator` was available. `_train_proxy_model` genuinely
  trained the model via real backprop and a real checkpoint, but the trained
  weights were never consulted at generation time -- Stage 1's own
  `recursion_depth_metrics`/`overthinking_curve` (and therefore any held-out
  generalization signal computed from them) reflected an untrained heuristic,
  not the trained model. Fixed by threading an optional `model` parameter
  through `generate_trajectories` -> `_recursion_metrics` ->
  `run_experiment_5574`, seeding trajectory generation from the model's own
  forward pass (mean-pooled over the K-step horizon) when supplied, with the
  original heuristic preserved as the fallback when no model is given.
- 2026-07-13: **exp5574 artifact/code mismatch found, not yet re-run.** The
  checked-in `results/experiment_5574_ptrm_stochastic_generator_stage1.json`
  (committed alongside the source file in the single commit `70c857a69`,
  whose own commit message is unrelated -- "SGE anti-stagnation diversity
  controller and live-path precheck", task `exp5575-sge-anti-stagnation-live-
  precheck") contains fields
  (`recursion_depth_metrics.*.exact_window_accuracy`,
  `recursion_depth_metrics.*.per_action_accuracy`,
  `verifier_selection_method.selection_eval.verifier_selection_uplift`,
  `gpu_device_receipt.device_count`/`nvidia_smi_returncode`,
  `controls.non_recursive.halting_distribution`) that the committed
  `_recursion_metrics`/`_gpu_device_receipt`/`build_stage1_artifact` code in
  that same commit does not compute or accept -- the artifact could not have
  been produced by the code it was committed with. `scripts/
  adversarial_verify.py` did not flag this (its checks do not include
  artifact-schema-vs-source-code consistency). The original artifact is
  preserved unmodified per this project's never-prune discipline; its
  specific numbers should not be cited pending investigation into how it was
  produced. REQ-ARC-PTRM-5600-2's properly-powered multi-seed run
  supersedes it as the trustworthy reference going forward.
- 2026-07-13: Multi-seed (10 seeds/game), pre-registered leave-one-game-out
  gate implemented in `python/carnot/experiment_5600_ptrm_loo_gate.py`
  (REQ-ARC-PTRM-5600-2), reusing the now-fixed generation path. **Real run
  completed: the gate FAILS.** Only `ft09` (1 of 5 held-out games) clears
  both bars (p=0.0020, PTRM mean 0.1459 vs majority-baseline 0.1367);
  `cd82` and `vc33` beat the majority baseline but not significantly vs the
  non-recursive control; `m0r0` and `sk48` clear neither. 1 of 5 is below
  the required majority (>= 3). Per `ops/known-issues.md` task 8's
  `retire_if_same_verdict: true`, the TRM-as-generator line for ARC-AGI-3 is
  retired: `results/experiment_5600_ptrm_loo_gate.json`.

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

### REQ-ARC-PTRM-5600-1: Trajectory Generation Consumes Trained Model Weights

`generate_trajectories` SHALL accept an optional trained
`PTRMActionSequenceGenerator`. When supplied, the per-input starting logits
SHALL be derived from the model's own forward pass (mean-pooled over the
K-step target horizon), not from the untrained `_base_action_logits`
frequency heuristic. When no model is supplied, the original heuristic-only
behavior SHALL be preserved unchanged. `_recursion_metrics` and
`run_experiment_5574` SHALL forward the trained model so that Stage-1's
`recursion_depth_metrics`/`overthinking_curve` reflect the trained model's
weights, and `model_architecture.trajectory_generation_uses_trained_model`
SHALL record `true` when this path is exercised.

### REQ-ARC-PTRM-5600-2: Multi-Seed Pre-Registered Leave-One-Game-Out Gate

Following the same rigor established by the prior standalone-reimplementation
pilots (`docs/research-notes/trm-leave-one-game-out-pilot-results-2026-07-05.md`
v3), Experiment 5600 SHALL evaluate the (now wiring-fixed) PTRM Stage-1
pipeline against a fixed, pre-registered set of held-out games
(`ft09`, `m0r0`, `vc33`, `sk48`, `cd82` -- the same set v3 used, for direct
comparability and to avoid post-hoc held-out-game selection) across multiple
independent seeds per game. For each (game, seed), the experiment SHALL train
on all other games and measure, on the held-out game: (a) the wiring-fixed
PTRM arm (trained-model-seeded stochastic recursion + Carnot-verifier
selection), (b) a non-recursive control (the trained model's own single-shot
argmax prediction, no recursion or verifier selection), and (c) a
majority-class baseline fit on training targets only. The experiment SHALL
run a paired Wilcoxon signed-rank test (matched by seed) between the PTRM arm
and the non-recursive control per held-out game, and SHALL report a
pre-registered falsifiable gate: PTRM is supported only if, in a majority
(>= 3 of 5) of held-out games, it both (i) beats the non-recursive control
with p < 0.05 and (ii) has a higher mean per-action accuracy than the
majority-class baseline. The experiment SHALL disclose the corpus's
`level_progress >= 1.0` won-session proxy as an inherited methodology caveat
(per the v4 pilot's own finding that this proxy can mean "reached this
session's own highest recorded checkpoint," not "won the whole game") without
attempting to fix it in this experiment's scope. If the gate fails, the
artifact SHALL set `retire_trm_generator_line` per the `retire_if_same_verdict`
condition in `ops/known-issues.md` task 8.

### SCENARIO-ARC-PTRM-5600-WIRING-FIX: Different Model Weights Produce Different Trajectories

Given two `PTRMActionSequenceGenerator` instances with materially different
weights, the same batch, seed, and zero noise, `generate_trajectories` called
with each model produces different depth-1 energy and different sampled
action sequences -- proving the model's weights are load-bearing in
generation, not orphaned as they were before the fix.

### SCENARIO-ARC-PTRM-5600-LOO-GATE: Pre-Registered Gate Computed Honestly

Given the multi-seed sweep across the five held-out games, the artifact
reports the Wilcoxon p-value and win count for every (game) combination, the
majority-class baseline as a flat non-seed-varying reference, and a single
`loo_verdict_reached=true` with an honest `heldout_generalization_signal`
that matches the pre-registered gate definition -- never a verdict computed
after seeing results that contradicts the gate as defined in
REQ-ARC-PTRM-5600-2.
