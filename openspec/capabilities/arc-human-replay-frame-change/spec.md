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

### REQ-ARC-FCP-4511: Self-Supervised No-Op Pruning From Offline Arcade Transitions

Experiment 4511 SHALL write
`results/experiment_4511_frame_change_prune_predictor.json` from a
self-supervised offline-arcade corpus. The corpus SHALL be collected from
locally executed `(frame, action, next_frame)` transitions across the 25 public
offline games, deriving `changed` and `magnitude` only from rendered frame
differences. The optional human-replay corpus MAY bootstrap counts, but the
experiment SHALL remain complete without network or external quota.

The frame-change predictor SHALL train a small torch action-effect model and
wire it into `rich_action_candidates` as a PRUNING gate. Candidates with
`P(frame_change)` below the swept threshold SHALL be dropped before explorer
expansion, while default behavior without a threshold SHALL preserve the legacy
candidate set. The experiment SHALL compare the fixed local submission-gate
baseline of 7760 median actions against the pruned measurement and SHALL report
an honest null if pruning does not reduce median actions or reduces solve-rate.

Required field principles:

- `honest_verdict`: principle "terminal prefix complete:/success:/passed:/shipped_; e.g. success: frame_change_prune_median_actions_<n>_below_7760 OR complete: prune_no_action_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search + small-predictor scoring, NO GGUF/LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control so the delta is auditable, not a moving baseline."
- `median_actions_with_prune`: principle "the headline -- pruning's whole point is to cut this number."
- `solve_rate_baseline`: principle "pruning MUST NOT drop solve-rate; a faster agent that solves less is not a win."
- `solve_rate_with_prune`: principle "the no-regression check on solve-rate."
- `heldout_noop_precision`: principle "the predictor must generalize cross-game (pooled training) -- the StochasticGoose persist-across-games idea, measured held-out."
- `positive_control_passed`: principle "proves the harness can detect a real reduction (guards against a silently-broken metric)."
- `false_negative_risk_checked`: principle "a null result is only valid if a positive control passed -- per CLAUDE.md FALSE_NEGATIVE_RISK."
- `random_seed`: principle "determinism is the precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records WHICH resources were verified; pre-empts silent-missing-resource fabrication."

### REQ-ARC-FCP-4512: Imitation Action-Type and Sequence Prior

Experiment 4512 SHALL write
`results/experiment_4512_imitation_action_prior.json` from a behavior-cloning
prior over action types and action sequences. The prior SHALL estimate
`P(action | frame-class)` from the locally staged human replay corpus when it is
loadable. If the human replay corpus is absent or empty, the experiment SHALL
fall back to a self-supervised marginal derived from offline-arcade
`(frame, action, next_frame)` transitions, so the prior cannot null merely
because the replay corpus is unavailable.

The explorer SHALL accept the action prior as an expansion-ordering signal and
an opt-in low-likelihood pruning gate. Candidate expansion SHALL preserve the
legacy candidate order as the stable tie-break, SHALL drop only the bottom
quantile requested by the prior-prune setting, and SHALL retain at least one
candidate. Experiment 4512 SHALL compare the fixed 8-game local submission-gate
baseline of 7760 median actions against the prior-guided measurement on the
same gate games, report per-game rows, solve-rate, and median
actions-to-first-level-up, and use an honest null if the prior does not reduce
actions or reduces solve-rate.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: imitation_prior_median_actions_<n>_below_7760 OR complete: imitation_prior_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control, fixed."
- `median_actions_with_prior`: principle "the headline -- did the human prior cut exploration."
- `solve_rate_baseline`: principle "no-regression reference."
- `solve_rate_with_prior`: principle "the prior must not drop solve-rate."
- `prior_source`: principle "honest declaration of whether the human replays or the self-supervised fallback supplied the prior (no silent corpus dependency)."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4513: ACT-Style Adaptive Per-Step Explorer Budget

Experiment 4513 SHALL write
`results/experiment_4513_adaptive_per_step_budget.json` from an ACT-style
adaptive budget gate for the live `StepwiseExplorer`. The gate SHALL use only
already-computed, frame-local signals: value-head/candidate margin when
available, predicted no-op fraction from an already-wired frame-change or
induced-model scorer when available, and frame novelty from the rendered-state
hash. It SHALL NOT train a new model, load an LLM, submit to the leaderboard, or
claim to implement LoopWM; it SHALL frame the method as an ACT/PonderNet-style
budget controller applied to the existing explorer.

The explorer SHALL preserve legacy behavior when no adaptive threshold is
configured. When configured, each newly materialized frame SHALL compute an
ambiguity score from those signal components; scores below the swept threshold
SHALL commit the single top-ranked candidate as budget 1, while scores at or
above threshold SHALL retain the normal candidate width. The gate SHALL retain
at least one candidate and SHALL record diagnostics sufficient to audit how many
frames committed versus expanded.

Experiment 4513 SHALL sweep thresholds on the fixed 8-game local submission
gate, compare against the fixed 7760 median-actions baseline, report per-game
rows, solve-rate, and median actions-to-first-level-up, and reproduce solved
first-level-up action segments through `arc_solver_kit.reproduce`. If no swept
threshold reduces median actions without dropping solve-rate, the artifact SHALL
report an honest null. A positive control SHALL prove the harness can detect a
known reduction before a null may be interpreted.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: adaptive_budget_median_actions_<n>_below_7760 OR complete: adaptive_budget_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control, fixed."
- `median_actions_with_adaptive`: principle "the headline -- did skipping expansion on easy frames cut actions."
- `solve_rate_baseline`: principle "no-regression reference."
- `solve_rate_with_adaptive`: principle "the gate must not drop solve-rate."
- `ambiguity_signal_components`: principle "names the already-computed signals used (no new model/training) -- the zero-cost claim is auditable."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4514: Lazy Best-First Value-Weight Remeasure

Experiment 4514 SHALL write
`results/experiment_4514_lazy_best_first_value_weight.json` by sweeping the
submitted-default E3 explorer with `search_mode="best_first"` and lazy
frame-hash-cached value-head scoring at `value_weight` in `{0.0, 0.5, 1.0,
2.0}`. The sweep SHALL keep `value_weight=0.0` as the explicit control and
SHALL NOT rerun the known-regressed `value_weight=5.0` arm.

The live `StepwiseExplorer` SHALL preserve the full frontier: lazy top-K
selection MAY decide which candidate nodes pay the expensive value-head cost,
but SHALL NOT drop unscored tail nodes. Frontier ordering SHALL remain
depth-primary using the existing `depth + value_weight * value, depth,
-on_path` blend for nodes with value scores, while unscored tail nodes retain
their cheap priority and remain expandable. Value scores SHALL be cached by
frame hash.

The experiment SHALL measure the local submission-gate game set, report
held-out solve-rate, median actions-to-first-level-up on the fixed core
`{lp85,m0r0,sp80,vc33}`, and median per-game wall seconds for every swept
weight. A positive weight SHALL be selected only when it preserves every core
solve, reduces median core actions versus the `0.0` control, and keeps median
per-game wall time within the approximately 390-second budget. Otherwise the
artifact SHALL report an honest null and keep `SUBMITTED_VALUE_WEIGHT=0.0`.
Solved first-level-up segments SHALL be checked through
`arc_solver_kit.reproduce()`.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: lazy_value_weight_<w>_beats_0 OR complete: lazy_value_weight_null_keep_0."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade + value-head scoring, no GGUF load (1s floor)."
- `per_weight_results`: principle "the full {weight -> (solve_rate, median_actions, median_wall_s) table so the decision is auditable, not asserted."
- `control_value_weight_0`: principle "the explicit baseline -- a weight only wins if it BEATS 0 (guards the FALSE_NEGATIVE_RISK null)."
- `chosen_submitted_value_weight`: principle "the new SUBMITTED_VALUE_WEIGHT (0 if null); must keep test_arc_submitted_agent_parity.py consistent."
- `lazy_eval_speedup_confirmed`: principle "confirms the cheap-eval cost regime that distinguishes this from the .416 full-cost null."
- `false_negative_risk_checked`: principle "a null is valid only with the value_weight=0 control present."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4516: Submitted Integration Gate And Forward Navigation Loop

Experiment 4516 SHALL write
`results/experiment_4516_integration_8game_gate.json` by reading the landed
A1-A4 artifacts (`experiment_4511` through `experiment_4514`) and integrating
only non-flagged levers that pass the CORE set-containment gate: every baseline
core solve in `{lp85,m0r0,sp80,vc33}` SHALL be preserved and
`median_actions_on_core` SHALL be lower than the no-lever/value-weight-0
control. Any artifact carrying `flagged_adversarial: true` SHALL be skipped.
If no A1-A4 lever passes, the submitted configuration SHALL keep the bare
explorer and report an honest null for those levers.

The live `StepwiseExplorer` SHALL expose forward-navigation diagnostics that
count frontier navigation attempts, exact `_shortest_path` hits, partial
forward-walk hits, RESET replay fallbacks, recorded forward edges, and the
resulting `forward_walk_hit_rate`. When exact forward navigation to a selected
frontier node is unavailable, the explorer MAY walk forward to the deepest
known reachable ancestor of that node and replay only the suffix; if no
ancestor is reachable, it SHALL retain the existing RESET-replay fallback. This
navigation change SHALL NOT alter which frontier nodes are reachable and SHALL
NOT corrupt the stored `node["path"]` reconstruction. The artifact SHALL
separately call out the deepest known reachable ancestor behavior in its
nav-loop finding when that partial walk engages.

Experiment 4516 SHALL remeasure the fixed 8-game local submission gate
end-to-end on the submitted configuration after the accepted levers and
navigation fix are wired. The artifact SHALL report median actions, solve-rate,
held-out solve-rate, and an explicit finding that explains why the prior
forward-edge fix did or did not move actions. An honest null is valid only when
the same 7760 baseline is present and measured on the same gate.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: integrated_median_actions_<n>_below_7760 OR complete: no_lever_beats_7760_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end, no LLM load (1s floor)."
- `median_actions_baseline`: principle "the 7760 control."
- `median_actions_integrated`: principle "the HEADLINE -- the SUBMITTED-config median after wiring the winners + the nav fix."
- `levers_integrated`: principle "names which of A1-A4 (and the nav fix) were wired -- traceable to their measured deltas."
- `solve_rate_integrated`: principle "integration must not drop solve-rate (and ideally keeps >13 reproducible levels for the submission gate)."
- `heldout_solve_rate`: principle "the real transfer signal (was 0.143); integration should not regress it."
- `nav_loop_finding`: principle "the answer to why the .416 nav-edge fix did not move actions (closes candidate 5)."
- `false_negative_risk_checked`: principle "an honest null only valid with the 7760 baseline measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4518: Canonical Local Submission Metric Harness

Experiment 4518 SHALL write
`results/experiment_4518_metric_harness_canonical.json` and make
`scripts/kaggle/arc_local_submission_gate.py` the canonical A1-A4/A6 local
metric dashboard. The gate SHALL pin the fixed eight-game set
`{lp85,m0r0,sp80,vc33,cd82,ft09,su15,ls20}`, derive CORE from the verified
baseline solves `{lp85,m0r0,sp80,vc33}`, and guard the baseline median of
`7760.0` against silent movement. The canonical action metric SHALL be total
`actions` on solved CORE games for both baseline and treatment; a treatment
that reports a different action field than the baseline SHALL fail the guard
instead of mixing `actions_to_first_levelup` with total actions.

The gate SHALL expose `--lever <name>` and include a per-lever attribution row
with `median_actions_on_core`, `core_solves_preserved`, `bonus_solves`, and
the uniform delta versus the fixed baseline. The CORE set-containment verdict
SHALL be retained: every baseline CORE solve must be preserved, fringe bonus
solves SHALL be reported but never netted against a CORE loss, and the old raw
solved-count verdict SHALL NOT return. The fixture suite SHALL keep the A1
lost-CORE failure, A2 CORE-for-fringe failure, positive-control improvement,
neutral non-inferior pass, bonus reporting, and legacy baseline fallback.

The harness SHALL measure the headroom budget `B*` as the smallest candidate
budget in `{8000,12000,16000,24000}` whose baseline solved set equals the
baseline solved set at `1.5B`. The CLI default budget SHALL remain `8000`
until that measurement is available, and after measurement SHALL use `B*` as
the canonical default. The experiment artifact SHALL record the measured
headroom table and the selected default so cap changes cannot hide a slowdown.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. shipped: metric_harness_canonical_ci_guarded OR complete: metric_harness_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
- `canonical_game_set`: principle "the fixed 8 games -- pins the metric so no A-task can cherry-pick an easier subset."
- `canonical_baseline`: principle "7760, guarded against silent movement (raising-a-cap-to-hide-drift is forbidden)."
- `positive_control_passed`: principle "proves the harness can detect a real reduction (guards a silently-broken metric)."
- `tests_added_pass`: principle "Tests Must Run and Assert."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4523: Forward-Walk Navigation Cost And Frontier Batch Sweep

Experiment 4523 SHALL write
`results/experiment_4523_forward_walk_navigation.json` by sweeping the live
`StepwiseExplorer` over frontier batch sizes `{1,3,8,all}` and the
navigation-cost frontier tie-break `{false,true}` on the fixed eight-game local
submission gate. The control SHALL be exactly `k=1` with no navigation-cost
tie-break and SHALL measure the same canonical total `actions` field as every
treatment.

The live `StepwiseExplorer` SHALL expose opt-in controls for both levers while
preserving the submitted default when those controls are left at their control
values. Frontier ordering SHALL keep depth as the primary priority; navigation
cost MAY only break ties among equal-depth eligible frontier nodes and SHALL
score exact forward reachability by `_shortest_path` length before falling back
to root replay path length. When the explorer has already paid navigation to a
frontier node, the frontier batch lever MAY queue up to `k` of that node's
untested salient actions before global frontier selection is allowed to move
elsewhere; `k=1` SHALL remain byte-for-byte equivalent to the previous single
probe behavior.

Experiment 4523 SHALL report, for every swept config,
`median_actions_on_core`, `core_solves_preserved`, `reset_replay_fallbacks`,
`reset_replay_steps`, and `forward_walk_hit_rate`. The submitted config SHALL
be wired only if the gate verdict tag is `IMPROVED`, every baseline CORE solve
in `{lp85,m0r0,sp80,vc33}` is preserved, and the treatment median actions on
CORE is strictly lower than the `k=1`/no-tie-break control. Otherwise the
artifact SHALL report an honest null and leave the submitted config unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; e.g. success: forward_walk_median_actions_on_core_<n>_below_<control> OR complete: forward_walk_no_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search, no GGUF/LLM load (1s floor)."
- `median_actions_on_core_control`: principle "the k=1/no-tie-break baseline measured the SAME way -- the apples-to-apples control (A3 false-win guard)."
- `median_actions_on_core_best`: principle "the headline -- did amortizing the replay cut TOTAL actions on the CORE games."
- `core_solves_preserved`: principle "HARD empirical gate on {lp85,m0r0,sp80,vc33} -- under the fixed explore_budget, batch/reorder CAN drop a knife-edge solve (the .417 m0r0 mechanism); a dropped CORE solve FAILS the lever regardless of action savings."
- `nav_diagnostics_before_after`: principle "reset_replay_fallbacks + reset_replay_steps + forward_walk_hit_rate WITH vs WITHOUT -- the causal mechanism witness (did replay actually drop, or is any action change incidental)."
- `action_field_used`: principle "names the SINGLE action field both conditions were measured on (total actions on solved) -- the A3 metric-mismatch guard."
- `config_sweep`: principle "the full {k, tie_break -> (median_actions_on_core, core_solves_preserved, reset_replay_steps)} table so the decision is auditable, not asserted."
- `chosen_submitted_config`: principle "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); must keep test_arc_submitted_agent_parity.py consistent."
- `positive_control_passed`: principle "proves the harness can detect a real replay reduction (guards a silently-broken metric)."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4527: Nav Metric First-Class Submission Gate Guard

Experiment 4527 SHALL write
`results/experiment_4527_nav_metric_harness.json` after validating that the
local submission gate records the authoritative per-game score levers and the
secondary navigation diagnostics as first-class fields. The gate SHALL keep
CORE set-containment and per-level efficiency as the verdict metric; it SHALL
track each game's deepest level reached and per-level efficiency, because
solving more or deeper levels is the real score lever. The gate SHALL also
track each game's `reset_replay_steps` and `forward_walk_hit_rate` from the
live navigation diagnostics, but these diagnostics SHALL only produce a
wall-clock warning and SHALL NOT replace or demote the per-level score verdict.

The `--update-baseline` path SHALL validate the candidate baseline with
`validate_canonical_baseline` before persisting it. An efficiency-bearing
baseline SHALL include per-level efficiency for every CORE game and SHALL keep
the `lp85` per-level efficiency at or above the canonical floor, so a deflated
baseline cannot later disarm the SF-3 per-game efficiency guard.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: nav_metric_first_class_ci_guarded OR complete: nav_metric_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the offline gate, no LLM load (1s floor)."
- `nav_metric_added`: principle "names the per-game nav fields the gate now tracks -- the nav-regression early warning."
- `tests_added_pass`: principle "Tests Must Run and Assert."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4528: .417 B-Track Infra Carryforward Audit

Experiment 4528 SHALL write
`results/experiment_4528_infra_carryforward.json` by reading the upstream
`.417 B2` artifact
`results/experiment_4518_metric_harness_canonical.json` and reconciling the
canonical local submission-gate state without rerunning the live headroom
measurement unless the upstream artifact is missing required evidence. The
audit SHALL confirm whether the CORE set-containment verdict is canonical,
whether the four-or-more verdict fixtures are CI-guarded, and whether the
headroom-budget table measured `B*` as the smallest candidate whose baseline
solved set equals the baseline solved set at `1.5B`. If no candidate plateau is
present in the upstream table, the audit SHALL record `b_star_measured=false`
and keep the default budget unchanged instead of raising it blindly.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: infra_carryforward_complete OR complete: infra_audit_already_done."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (reads the .417 B2 artifact) unless it measures B* live (then verifier_ensemble_against_cached_candidates)."
- `b_track_status`: principle "what landed in .417 B2 vs what this task completed -- the audit trail."
- `cited_upstream_artifacts`: principle "traceability of the .417 B2 numbers this reconciles."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4524: Stop After Scored Target Level-Up

Experiment 4524 SHALL write
`results/experiment_4524_stop_after_levelup.json` by measuring the fixed
eight-game local submission gate under a run-to-completion control and a
stop-at-scored-target treatment. The control SHALL use the submitted
run-to-completion target, log actions to reach each observed level-up for every
gate game, and report the total run actions on the same `actions` field used by
the local submission gate. The treatment SHALL stop through `is_done` once the
measured scored target is reached; it SHALL NOT compare
`actions_to_first_levelup` from one arm against total `actions` from another.

The treatment SHALL be accepted only when every baseline CORE solve in
`{lp85,m0r0,sp80,vc33}` is preserved, every gate game's banked level depth is
preserved (`best_level` before versus after), and median CORE `actions` is
strictly lower than the measured control. If the treatment drops a CORE solve,
sheds any game's banked level depth, or shows no total-action overrun, the
artifact SHALL report an honest null and leave the submitted target unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: stop_after_levelup_core_actions_<n>_below_control OR complete: no_overrun_or_drops_solve_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade, no LLM load (1s floor)."
- `core_solves_preserved`: principle "stopping earlier MUST NOT drop a CORE solve (set-containment over {lp85,m0r0,sp80,vc33})."
- `levels_per_game_preserved`: principle "HARD gate: per-game best_level before vs after -- stopping early MUST NOT shed any game's banked level depth (the CORE gate checks the game set only; the competition scores total LEVELS, so a level-depth regression that the gate would PASS must be caught here)."
- `median_actions_on_core_control`: principle "the run-to-completion baseline, same action field."
- `median_actions_on_core_best`: principle "the headline -- did stopping at the scored target cut total actions."
- `action_field_used`: principle "single action field both conditions measured on (A3 metric-mismatch guard)."
- `positive_control_passed`: principle "proves the harness detects a real reduction."
- `false_negative_risk_checked`: principle "a null is valid only with the control present."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

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

### SCENARIO-ARC-FCP-4524: Stop Policy Preserves CORE Level Depth

Given the fixed eight-game local submission gate has a measured
run-to-completion control
When experiment 4524 compares that control with a stop-at-scored-target run
Then the artifact records actions-to-reach-each-level versus total actions per
game, uses `action_field_used="actions"` for both arms, preserves every CORE
solve, preserves every game's before/after `best_level`, and only reports
success when the treatment's median CORE total actions is strictly lower than
the measured control.

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

### SCENARIO-ARC-FCP-4511: Predicted No-Ops Are Pruned Before Expansion

Given a frame, legacy candidate actions, and a frame-change scorer that assigns
low probability to no-op candidates
When `rich_action_candidates` is called with a pruning threshold
Then candidates below the threshold are removed before ranking or expansion,
at least one candidate is retained, and the experiment 4511 artifact records
the baseline 7760 control, pruned median action measurement, solve-rate guard,
held-out no-op precision, positive control status, random seed, and
reproducibility checksum.

### SCENARIO-ARC-FCP-4512: Imitation Prior Orders and Prunes Expansion

Given a frame, a legacy candidate list, and an imitation prior that assigns
higher likelihood to the human-like action sequence for that frame class
When the explorer builds candidates with an action prior and bottom-quantile
prune setting
Then the high-likelihood action is expanded before low-likelihood candidates,
the bottom quantile is pruned while retaining at least one candidate, and the
experiment 4512 artifact records the 7760 baseline, prior-guided median action
measurement, solve-rate guard, prior source, positive control status, false
negative risk check, random seed, and reproducibility checksum.

### SCENARIO-ARC-FCP-4513: Easy Frames Commit One Candidate, Ambiguous Frames Expand

Given a frame, a legacy ranked candidate list, value-margin/no-op/novelty
signals, and an adaptive budget threshold
When the ambiguity score is below the threshold
Then the explorer retains only the top candidate for that frame and records a
budget-1 commit diagnostic.
When the ambiguity score is at or above the threshold
Then the explorer retains the normal candidate width, records an expanded
diagnostic, and the experiment 4513 artifact records the 7760 baseline,
adaptive median action measurement, threshold sweep, per-game solve-rate guard,
ambiguity signal components, positive control status, false negative risk check,
random seed, and reproducibility checksum.

### SCENARIO-ARC-FCP-4514: Lazy Value Scores Reorder Without Filtering

Given a live `StepwiseExplorer` frontier with more expandable nodes than the
lazy top-K value-scoring budget
When the explorer chooses a best-first frontier node with `value_weight>0`
Then only the top-K cheap-priority nodes pay the value-head cost, repeated frame
hashes reuse cached scores, unscored tail nodes remain in the frontier, and the
experiment 4514 artifact reports every swept weight, the explicit
`value_weight=0.0` control, the chosen submitted value weight, lazy-eval speedup
confirmation, false-negative-risk check, random seed, and reproducibility
checksum.

### SCENARIO-ARC-FCP-4516: Integrated Submission Gate Reports Null Or Win Honestly

Given the A1-A4 artifacts, the submitted agent config, and the fixed 8-game
local submission gate
When experiment 4516 selects integration levers
Then flagged artifacts are skipped, non-flagged levers are accepted only when
they preserve the core solve set and reduce core median actions versus the
no-lever control, and the submitted config keeps `value_weight=0.0` when no
positive value weight wins.

Given a live `StepwiseExplorer` with known forward edges from the current node
to an ancestor of a selected frontier node but no exact path to the frontier
When the explorer serves navigation for that frontier
Then it walks forward to the deepest reachable ancestor, replays only the
suffix before probing the frontier action, records navigation diagnostics, and
falls back to RESET replay only when no forward ancestor is reachable.

Given the integrated submitted config is measured on the fixed 8-game gate
When the artifact is written
Then it records the 7760 baseline, integrated median actions, integrated
solve-rate, held-out solve-rate, nav-loop finding, false-negative-risk check,
random seed, reproducibility checksum, and the exact levers integrated.

### SCENARIO-ARC-FCP-4518: Canonical Gate Guards The Fixed Metric

Given the verified 7760 baseline and the fixed eight-game local submission
gate
When the gate builds a dashboard row for a named lever
Then the row uses total `actions` for both baseline and treatment, preserves
the CORE set-containment verdict, reports bonus solves separately, and fails a
treatment that reports a different action field than the baseline.

Given the gate's regression fixtures
When the focused gate tests run
Then the A1 and A2 CORE-loss fixtures fail, the positive-control and neutral
fixtures pass, bonus solves are reported, the legacy baseline fallback remains
covered, and the fixed game set plus 7760 baseline guard cannot move silently.

Given candidate budgets `{8000,12000,16000,24000}`
When experiment 4518 measures baseline headroom
Then it selects the smallest `B*` whose solved set matches the solved set at
`1.5B`, keeps `8000` only if the measurement is unavailable, and writes the
terminal artifact with the required field principles.

### SCENARIO-ARC-FCP-4523: Batch And Navigation Tie-Break Are Swept Against CORE

Given a live `StepwiseExplorer` with multiple equal-depth frontier nodes
When navigation-cost tie-break is enabled
Then the selected frontier still comes from the shallowest eligible depth, but
equal-depth ties prefer exact forward walks from the current node before
shorter RESET replay paths.

Given a selected frontier node with several untested salient actions
When the explorer navigates to that node with frontier batch size greater than
one
Then up to `k` actions from that node are queued before global frontier
selection moves elsewhere, while `k=1` preserves the prior single-probe
control behavior.

Given experiment 4523 measures the fixed eight-game gate
When it compares the sweep against the `k=1`/no-tie-break control
Then every row uses the same total `actions` field, reports CORE preservation
and navigation diagnostics, passes the positive-control reduction guard, and
wires `SUBMITTED_AGENT_CONFIG` only for a strict `IMPROVED` CORE median action
reduction.

### SCENARIO-ARC-FCP-4527: Nav Metrics Are CI-Guarded But Secondary

Given the local submission gate measures the fixed eight-game set with the
authoritative per-level scorecard
When a measured config preserves CORE solves and per-level efficiency but has
more `reset_replay_steps` at equal total actions
Then the verdict remains non-inferior, the dashboard emits a navigation
regression warning, and the per-game rows expose `deepest_level_reached`,
`per_level_efficiency`, `reset_replay_steps`, and `forward_walk_hit_rate`.

Given the gate is invoked with `--update-baseline`
When the measured candidate baseline is missing CORE efficiency or drops `lp85`
below the canonical per-level efficiency floor
Then the baseline is not persisted and the command reports the canonical
baseline guard failure.

### SCENARIO-ARC-FCP-4528: .417 B-Track Audit Records No Blind Budget Raise

Given the upstream `.417 B2` canonical metric harness artifact
When experiment 4528 audits the artifact
Then it records the CORE containment gate, fixture CI guard, and measured
headroom table in `b_track_status`, cites the upstream artifact fields it
imports, and reports a complete audit if the headroom table has no stable `B*`
candidate instead of raising the local submission-gate default blindly.
