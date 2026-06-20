# ARC Human Replay Frame-Change Predictor Capability Specification

**Capability:** arc-human-replay-frame-change
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines the ARC-AGI-3 human replay frame-change predictor used to rank live
exploration candidates without reading private environment internals. The
predictor consumes rendered frames only, learns or loads an action-effect
scorer when replay data is available, and falls back to deterministic legacy
candidate order when the corpus or weights are absent.

## Requirements

### REQ-ARC-FCP-4490: Frame-Only Replay Feature Contract

The repository SHALL provide a frame-change predictor module for experiment
4490 that recomputes model inputs from raw rendered replay frames, not from
mirror-supplied opaque feature vectors or `env._game` internals. The module
SHALL expose a small torch CNN with a click heatmap head and a directional
action head, plus helpers that normalize ARC frames into fixed-size tensors.

The terminal artifact at
`results/experiment_4490_human_replay_frame_change_predictor.json` SHALL record
whether the ARC Public Demo replay corpus was locally available, whether any
weights were trained or bundled, and the exact preconditions checked.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `trained_on_human_corpus`: bare bool: true only when raw replay frames were locally available and used for training.
- `weights_bundled`: bare bool: true only after official CC0/MIT-0-compatible licensing is verified for bundled weights.
- `heldout_median_actions_before`: baseline median actions-to-first-level-up on held-out frame-only evaluation.
- `heldout_median_actions_after`: predictor-ranked median actions-to-first-level-up on the same held-out frame-only evaluation.
- `implied_efficiency_delta`: score-relevant delta in min(human/agent,1)^2 efficiency, never inferred when held-out data is missing.
- `positive_control`: synthetic clickability sanity check proving the ranking harness can detect a known win.
- `solve_rate_dropped`: guardrail bool: efficiency wins must not come from reducing solve rate.

### REQ-ARC-FCP-4491: Behavior Prior and Candidate Ranking

The repository SHALL expose a behavior-cloning action prior that combines
marginal action frequencies with optional state-conditioned click cells. The
ARC graph explorer SHALL use the prior and/or learned frame-change scorer to
rank `rich_action_candidates` while preserving the previous salience/raster
order as the stable tie-break and as the default when no scorer is supplied.

### REQ-ARC-FCP-4492: Honest Efficiency Artifact

Experiment 4490 SHALL report held-out median actions-to-first-level-up before
and after predictor ranking when a real human replay corpus is present. If the
corpus is not locally present, the artifact SHALL use a terminal-prefixed
`honest_verdict` that records the blocked resource instead of fabricating
training, weights, or efficiency deltas. The artifact SHALL include
`inference_substrate=verifier_ensemble_against_cached_candidates`,
`preconditions_checked`, and field principles for all required artifact fields.

### REQ-ARC-FCP-4493: Energy-Augmentation LOO Gate

Experiment 4492 SHALL write
`results/experiment_4492_energy_augmentation_loo_gate.json` from the
discriminative `cross_game_features_v3` leave-one-game-out verifier run. The
artifact SHALL compare `v3_loo_auroc` against the `0.503` v2 baseline and the
`0.600` deployment gate, include feature-class leave-one-game-out AUROCs and
material movement labels, and use
`inference_substrate=verifier_ensemble_against_cached_candidates`.

If `v3_loo_auroc > 0.600`, the frame-change ranking surface SHALL expose a
structural-feature energy term scored as `P(change) * (-delta_E)` while
preserving the previous stable tie-break ordering. If the gate does not pass,
the artifact SHALL honestly report which v3 feature classes moved the AUROC and
which did not, without enabling the structural energy ranking hook.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

### REQ-ARC-FCP-4495: Human Replay Corpus Staging

Experiment 4495 SHALL stage ARC Public Demo replay training shards under a
gitignored data directory so A1 loaders can reuse them without repeating a
cold upstream download. Each staged example SHALL be derived from replay
frames and actions and expose the local training contract
`frame`, `action`, `frame_delta`, and `level_progress`. The staging artifact at
`results/experiment_4495_human_replay_corpus_staging.json` SHALL record the
upstream URL, reachable mirror URL, source checksum metadata, shard checksums,
license status, attribution text when a CC BY mirror is used, and whether any
weights were committed.

If an official ARC source with CC0/MIT-0-compatible terms is reachable, the
artifact MAY record `official_license_verified=true`. If only the CC BY mirror
or an otherwise non-CC0/MIT-0 mirror is reachable, the artifact SHALL still
stage the format and attribution but SHALL record `weights_committed=false`,
`official_license_verified=false`, and a terminal-prefixed honest verdict that
does not imply bundled weights.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

### REQ-ARC-FCP-4496: Submitted-Agent Scoreboard Tracks Headline Signals

Experiment 4496 SHALL write
`results/experiment_4496_submitted_agent_scoreboard.json` as a milestone
scoreboard for the exact submitted-default ARC agent. Each scoreboard row SHALL
record the `SUBMITTED_AGENT_CONFIG` snapshot, the frame-only held-out generic
solve-rate with `env._game` blocked, and the variant-transfer rate. The artifact
SHALL keep `reproducible_total_levels` only as context and SHALL explicitly mark
it as not the headline signal.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

The artifact SHALL declare
`inference_substrate=verifier_ensemble_against_cached_candidates`, record the
precondition smoke checks for `arc_solver_kit.offline_arcade()` and Torch, and
reference the submitted-agent parity test so the reported generic solve-rate
cannot silently drift away from what ships.

### REQ-ARC-FCP-4505: Submitted-Agent Scoreboard Refresh Tracks Real Leaderboard Signal

Experiment 4505 SHALL write
`results/experiment_4505_submitted_agent_scoreboard.json` as the refreshed
.415 B2 submitted-default ARC scoreboard. The artifact SHALL report the current
`SUBMITTED_AGENT_CONFIG` snapshot, the frame-only held-out generic solve-rate
with `env._game` blocked, the variant-transfer solve-rate, and the A1
value-weight verdict. The artifact SHALL keep `reproducible_total_levels` only
as context and SHALL explicitly mark it as not the leaderboard headline.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_.
- `inference_substrate`: explicit substrate so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

The artifact SHALL declare
`inference_substrate=verifier_ensemble_against_cached_candidates`, record the
precondition smoke checks for `arc_solver_kit.offline_arcade()` and Torch,
include a focused parity gate for
`tests/python/test_arc_submitted_agent_parity.py` with `value_weight==0.0`, and
make the real leaderboard signal the pair
`submitted_default_heldout_generic_solve_rate` plus `variant_transfer_rate`
rather than `reproducible_total_levels`.

### REQ-ARC-FCP-4501: Frame-Only Predictor Rerun From Staged Replay Shards

Experiment 4501 SHALL write
`results/experiment_4501_frame_change_predictor_rerun.json` from the locally
staged ARC Public Demo human replay corpus. The rerun SHALL consume only
raw-frame shard fields (`frame`, normalized `action_id`, optional click
coordinates, `frame_delta`, and `level_progress`) and SHALL NOT consume mirror
`feature_keys`, opaque state vectors, bundled third-party weights, or
`env._game` internals.

The rerun SHALL train or smoke-train the repository's small torch CNN action
effect model and emit a behavior-cloning action prior that can be passed to
`rich_action_candidates`. It SHALL record how many local examples were actually
loaded versus the 14,672-example `action_effect_dict.npz` target. If the exact
14,672-example NPZ corpus is absent but staged frame-only shards are available,
the artifact SHALL keep the run complete but mark the corpus shortfall and use
the FALSE_NEGATIVE_RISK null guard instead of fabricating a headline efficiency
win.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `expected_action_effect_examples`: the external action-effect corpus target count, fixed at 14,672 for this rerun.
- `corpus_examples_loaded`: the exact number of frame-only shard examples loaded locally for this run.
- `feature_source`: proves features were recomputed from raw frames, not mirror feature vectors.
- `behavior_prior_emitted`: bare bool showing the behavior-cloning prior was built and can rank candidates.
- `heldout_median_actions_before`: baseline median actions-to-first-level-up on held-out frame-only evaluation.
- `heldout_median_actions_after`: predictor/prior-ranked median actions-to-first-level-up on the same held-out frame-only evaluation.
- `implied_efficiency_delta`: score-relevant delta in min(human/agent,1)^2 efficiency.
- `solve_rate_dropped`: guardrail bool; efficiency wins must not come from reducing solve rate.
- `false_negative_risk_guard`: records whether the null is interpretable because the positive control passed.

### REQ-ARC-FCP-4502: Energy-Augmented Candidate Ranking Measurement

Experiment 4502 SHALL write
`results/experiment_4502_energy_augmented_ranking.json` after the structural
energy gate has passed. The experiment SHALL combine the A2 frame-change
predictor score with an objective candidate energy over the v3 structural
feature classes and SHALL rank held-out cached candidates by
`P(frame_change) * (-delta_E)`. The predictor-only baseline SHALL rank the
same cached candidates by `P(frame_change)` alone, using the previous stable
candidate order as the tie-break.

The measurement SHALL report held-out solve-rate and action-efficiency for
both arms on the same candidate groups. If the energy-augmented arm does not
improve over predictor-only, the artifact SHALL report an honest null without
fabricating a gain. The artifact SHALL identify the v3 feature classes used for
energy, including the frame-delta and object-relational classes that moved the
4492 leave-one-game-out AUROC.

Required field principles:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit verifier_ensemble_against_cached_candidates declaration so adversarial_verify applies the cached-candidate duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.
- `predictor_only_solve_rate`: held-out solve-rate for the same cached candidates ranked by P(frame_change) alone.
- `energy_augmented_solve_rate`: held-out solve-rate for the same cached candidates ranked by P(frame_change) * (-delta_E).
- `predictor_only_median_actions`: median actions-to-first-heldout-solve under predictor-only ranking.
- `energy_augmented_median_actions`: median actions-to-first-heldout-solve under energy-augmented ranking.
- `efficiency_delta_vs_predictor_only`: difference in min(human/agent,1)^2 efficiency versus predictor-only, never inferred from mismatched candidate groups.
- `energy_term_added_value`: bare bool indicating whether the energy term improved solve-rate or efficiency without reducing solve-rate.

## Scenarios

### SCENARIO-ARC-FCP-4490: Positive-Control Candidate Ranking

Given a frame and candidate actions where only one click cell is known to
change the frame
When the behavior prior or scorer ranks the candidates
Then the changing click is ordered ahead of no-op candidates
And the legacy candidate order remains the tie-break for equal scores.

### SCENARIO-ARC-FCP-4491: Missing Corpus Does Not Fabricate Results

Given the ARC Public Demo replay corpus and replay-derived weights are absent
When experiment 4490 writes its terminal artifact
Then the artifact is valid JSON, starts `honest_verdict` with a terminal
prefix, records the missing corpus in `preconditions_checked`, leaves
`weights_bundled=false`, and reports no held-out efficiency win.

### SCENARIO-ARC-FCP-4492: Passing Structural Energy Gate Wires Ranking

Given the discriminative v3 verifier reports leave-one-game-out AUROC above
0.600 versus the 0.503 baseline
When experiment 4492 writes its terminal artifact
Then `honest_verdict` is terminal-prefixed, `loo_gate_passed=true`, the feature
classes that materially moved AUROC are recorded, and frame-change ranking can
rank candidates by `P(change) * (-delta_E)` with stable ties.

### SCENARIO-ARC-FCP-4495: Attributed Mirror Shards Load Without Cold Download

Given the official ARC Public Demo short link is not directly reachable from
the staging machine but an attributed mirror is reachable
When experiment 4495 writes replay-derived training shards under the gitignored
data directory
Then the loader reads rows containing `frame`, `action`, `frame_delta`, and
`level_progress`, the artifact records provenance and attribution, and
`weights_committed=false`.

### SCENARIO-ARC-FCP-4496: Scoreboard Separates Headline From Banked Levels

Given a cached submitted-default benchmark artifact and the current
variant-transfer signal
When experiment 4496 writes its scoreboard artifact
Then the row reports the exact submitted-default held-out generic solve-rate and
variant-transfer rate as headline metrics, preserves source provenance for both
measurements, keeps `reproducible_total_levels` in a context-only field, and
records that `test_arc_submitted_agent_parity.py` is the focused parity gate.

### SCENARIO-ARC-FCP-4505: Refreshed Scoreboard Pins Submitted Defaults

Given the refreshed submitted-default held-out benchmark, the current
variant-transfer signal, and the value-weight remeasurement verdict
When experiment 4505 writes its scoreboard artifact
Then the artifact reports the current submitted-default config with
`value_weight==0.0`, records the parity test as green, reports the generic
solve-rate and variant-transfer rate as headline metrics, and leaves
`reproducible_total_levels` in context-only metadata.

### SCENARIO-ARC-FCP-4501: Staged Frame-Only Rerun Has Interpretable Null Guard

Given locally staged replay shards but no bundled 14,672-example
`action_effect_dict.npz`
When experiment 4501 trains or smoke-trains the frame-only predictor and writes
its terminal artifact
Then the artifact is valid JSON with terminal-prefixed `honest_verdict`,
records the staged corpus count and the missing NPZ target, emits a behavior
prior, keeps `solve_rate_dropped=false`, reports before/after held-out action
metrics from the same candidate-order harness, and marks
`false_negative_risk_guard=positive_control_passed_null_interpretable` when the
positive control detects a known ranking win.

### SCENARIO-ARC-FCP-4502: Energy-Augmented Ranking Is Measured Against Predictor-Only

Given the 4492 structural energy gate passed and the 4501 frame-change
predictor/prior can rank held-out cached candidates
When experiment 4502 scores each candidate with both `P(frame_change)` and the
v3 structural `delta_E`
Then the artifact compares predictor-only and energy-augmented held-out
solve-rate and action-efficiency on identical candidate groups, records
`inference_substrate=verifier_ensemble_against_cached_candidates`, and uses a
terminal-prefixed honest verdict that reports either an added-value win or an
honest null over predictor-only.
