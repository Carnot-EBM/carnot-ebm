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

### REQ-ARC-FCP-4547: Cached Human-Replay CNN Ranker Measurement

Experiment 4547 SHALL write
`results/experiment_4547_frame_change_predictor.json` after training the
repository's small frame-only CNN on the locally cached ARC Public Demo human
replay `(frame, action, frame_delta)` corpus. The training and scoring path
SHALL use CPU or integrated-device execution only, recompute features from
rendered frames, and SHALL NOT consume mirror `feature_keys`, bundled third
party weights, `env._game` internals, network resources, or 3090-class GPUs.

The trained CNN SHALL be exposed through `FrameChangeScorer` and passed into
`rich_action_candidates`/`StepwiseExplorer` as an ordering signal that ranks
candidate clicks and ACTION1-5 by predicted frame change while preserving the
legacy blind-BFS/salience order as the stable fallback tie-break. Experiment
4547 SHALL compare the CNN-ranked arm against the matched blind-BFS control on
the same held-out games or cached candidate groups, reporting median
actions-to-first-level-up and solve-rate for both arms. A success verdict is
allowed only when the CNN median is strictly lower than the blind median and
solve-rate is not dropped; otherwise the artifact SHALL report an honest null.

The experiment SHALL also compute a held-out transition-delta AUROC for the CNN
against the trivial `0.5` baseline. A median-actions null SHALL be considered
interpretable only when this positive control passes; otherwise the artifact
SHALL mark the false-negative risk as unchecked instead of claiming that the
ranker was fairly ruled out. If no action reduction is found after the positive
control passes, the artifact SHALL include the secondary recommended input:
hidden-field probing in the state hash for the `ka59`/`ar25` L2 stall.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: frame_change_cnn_median_actions_reduced_<n> OR complete: frame_change_cnn_no_action_reduction_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- CNN trains on cached transitions + scores offline candidates, no headline LLM load."
- `median_actions_to_first_levelup_cnn`: principle "the HEADLINE -- held-out median actions-to-first-levelup with the CNN ranker (the score-metric lever)."
- `median_actions_to_first_levelup_blind`: principle "the matched blind-BFS control measured the SAME way -- the apples-to-apples comparison."
- `solve_rate_preserved`: principle "HARD gate -- the action-efficiency win must NOT drop solve-rate (a faster agent that solves fewer games is worse)."
- `cnn_held_out_delta_auroc`: principle "the POSITIVE CONTROL -- the CNN predicts held-out transition deltas above a trivial baseline; guards a silently-broken predictor."
- `positive_control_passed`: principle "the CNN learned the action-effect signal; a median-actions null is valid only if this passed."
- `false_negative_risk_checked`: principle "a null is valid only with the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, human-replay corpus cached); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4568: Pooled Clickability Action-Effect Predictor

Experiment 4568 SHALL write
`results/experiment_4568_clickability_action_effect_predictor.json` after
loading the local ARC Public Demo human replay shards and the local
`data/arc_transition_corpus/*.npz` self-captured transitions into a pooled
frame/action/frame-change corpus. The training path SHALL use the small
frame-only CNN action-effect model, CPU or integrated-device inference, rendered
frames only, and SHALL NOT depend on mirror feature vectors, third-party
weights, live LLM calls, or `env._game` internals.

The experiment SHALL expose the trained model through `FrameChangeScorer` and
wire that scorer into `rich_action_candidates` so candidate clicks and
ACTION1-5 are ordered by predicted frame-change probability with the existing
salience order as the no-regression tie-break. It SHALL compare the
predictor-ranked arm against the matched blind-BFS order on held-out games or
cached candidate groups, report median actions-to-first-level-up for both arms,
emit `actions_delta = baseline - with_predictor`, and gate any success verdict
on a strictly positive action delta, a bootstrap CI excluding zero, and no
solve-rate drop.

The artifact SHALL include the generic-transfer measurement shape, the
leaderboard efficiency term `min(human/agent,1)^2` from replay-derived human
action counts, a positive-control clickability check, `verifier_is_oracle=false`,
`offline_reproduced`, `chosen_submitted_config`, a `null_delta_methodology_note`
when the action delta is exactly zero, and a checksum over corpus/model/metric
inputs.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: clickability_predictor_actions_to_levelup_<n>_below_blind OR complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- the CNN trains/scores against cached transitions (no LLM load, 1s floor); fast CPU forward pass declared."
- `verifier_is_oracle`: principle "MUST be false -- learned action-effect model is oracle-DISTINCT from executable win-check."
- `median_actions_to_first_levelup_with_predictor`: principle "the HEADLINE -- held-out median actions-to-first-levelup with the predictor-ranked explorer."
- `median_actions_to_first_levelup_baseline`: principle "the blind-BFS baseline, measured the SAME way."
- `actions_delta`: principle "baseline - with_predictor; positive = fewer actions."
- `actions_delta_ci`: principle "bootstrap CI on the actions delta; efficiency claim requires the CI to exclude zero."
- `efficiency_score_min_human_agent_sq`: principle "min(human/agent,1)^2 with the human baseline from replay corpus."
- `generic_transfer_rate_with_predictor`: principle "held-out variant transfer WITH the predictor vs the 0.04 baseline."
- `solve_rate_preserved`: principle "HARD gate -- efficiency win must NOT drop solve-rate."
- `positive_control_passed`: principle "learnable-clickability control where predictor-ranking must beat blind."
- `false_negative_risk_checked`: principle "a no-value null is valid only if the positive control passed."
- `null_delta_methodology_note`: principle "present when actions_delta==0.0 -- honest no-gain null, not a measurement bug."
- `chosen_submitted_config`: principle "recommend enable predictor-ranker when successful; unchanged if null."
- `missing_verifier_gaps`: principle "if no gain, record residual generation/ranking gap."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, torch, LOCAL corpora)."

### REQ-ARC-FCP-4575: Learned-CNN Substrate Guard

The adversarial artifact reader SHALL recognize a learned CNN action-effect
artifact that declares `inference_substrate=verifier_ensemble_against_cached_candidates`
and references CNN or torch markers but no LLM/GGUF invocation as an offline
cached-candidate scoring artifact. Such an artifact SHALL use the one-second
verifier-scoring duration floor, not the sixty-second live-model floor, so a
fast CPU or integrated-device forward pass is not quarantined as
`DURATION_TOO_SHORT`.

The guard SHALL remain strict for real live-model claims: an artifact that
declares `inference_substrate=live_llm_inference` or names a GGUF/live LLM
model and reports `duration_s < 60` SHALL still emit a critical
`DURATION_TOO_SHORT` flag. The summary reader SHALL surface the substrate floor
it applied so reviewers can see why a CNN artifact used the offline floor and
why a fake live-LLM artifact used the live floor.

Experiment 4575 SHALL write
`results/experiment_4575_learned_cnn_substrate_guard.json` with required
principle-annotated fields for `honest_verdict`, `inference_substrate`,
`guard_mechanism`, `cnn_artifact_not_flagged`, `fake_llm_still_flagged`,
`tests_added_pass`, and `preconditions_checked`.

Required field principles:

- `honest_verdict`: principle "terminal prefix; shipped: learned_cnn_substrate_guard_added OR complete: learned_cnn_substrate_guard_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the guard against fixtures, no model load (1s floor)."
- `guard_mechanism`: principle "names the recognized substrate/floor + where it fires -- the fix that stops a fast-but-real CNN being quarantined."
- `cnn_artifact_not_flagged`: principle "a fast learned-CNN action-model fixture is NOT DURATION_TOO_SHORT-flagged -- the .422 A1 headline protection."
- `fake_llm_still_flagged`: principle "a live_llm_inference fixture at <60s IS still flagged -- guards against weakening the real fabrication check."
- `tests_added_pass`: principle "Tests Must Run and Assert -- both the not-flagged-on-CNN and still-flagged-on-fake-LLM cases."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4629: Graduate Action-Effect Predictor Into The Scored Live Explorer

Experiment 4629 SHALL write
`results/experiment_4629_graduate_action_effect_predictor_live.json` after
assembling the self-supervised small CNN action-effect predictor and the
PersistentAEM cross-game action-effect memory from the local
`(frame, action, next_frame)` transition corpus. The graduated scorer SHALL be
reachable from the SCORED `E3AgentPolicy` path and consumed by
`arc_graph_explore.rich_action_candidates` as a frame-only action-effect ranker,
with the bare explorer order retained as the matched control and stable
tie-break.

When both a candidate router and the action-effect ranker are supplied, the
candidate router MAY provide a stable tie-break order, but the action-effect
ranker SHALL be the final ordering pass so predicted no-op actions are
deprioritized on the submitted live path. The scorer SHALL use only rendered
frames, action ids, click coordinates, and locally cached transition effects;
it SHALL NOT inspect `env._game`, use an executable win-check as a scorer, call
live LLMs, or require 3090-class hardware.

Experiment 4629 SHALL compare the predictor-ranked live candidate ordering
against the bare explorer order on the same held-out cached public-game
candidate groups, reporting median actions-to-first-levelup, the
`min(human/agent,1)^2` efficiency term, first-win-rate, solve-rate, and a
bootstrap confidence interval on `actions_delta = bare - predictor`. A success
verdict is allowed only when the action delta is positive with a CI excluding
the bare baseline and solve-rate is preserved. If the matched bare control
passes but the live efficiency delta is zero, the artifact SHALL report the
honest null and leave `chosen_submitted_config` unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: action_effect_predictor_graduated_live_efficiency_up_<n> OR complete: action_effect_predictor_graduated_no_live_efficiency_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the CNN is a small conv net (CPU/iGPU), declared so a fast forward-pass is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the action-effect predictor is a learned action-pruner, oracle-DISTINCT from the executable win-check (north-star §5 action-pruner role)."
- `solve_provenance`: principle "live_agent_self_discovery -- this improves the SCORED live agent's OWN action selection (arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the predictor module is imported by arc_graph_explore (rich_action_candidates) AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT orphaned)."
- `median_actions_to_first_levelup_predictor`: principle "the HEADLINE -- LIVE median actions-to-first-levelup WITH the action-effect predictor (lower = the score-term win)."
- `median_actions_to_first_levelup_bare`: principle "the matched bare-explorer actions on the SAME variants (today's no-op-burning baseline)."
- `actions_delta`: principle "bare - predictor (positive = fewer actions), emitted explicitly so a null (0) is annotated."
- `efficiency_score_term`: principle "the min(human/agent,1)^2 leaderboard efficiency term WITH the predictor (the score metric we have NONE of)."
- `actions_delta_ci`: principle "bootstrap CI on the actions delta; an efficiency claim requires the CI to exclude the bare baseline."
- `first_win_rate_delta`: principle "predictor - bare first-win-rate; emitted explicitly so a null is annotated (efficiency must not cost solves)."
- `solve_rate_preserved`: principle "HARD gate -- ranking candidates by predicted frame-change must NOT drop solve-rate vs bare."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the bare explorer ran on the SAME variants; an efficiency null is valid only then."
- `false_negative_risk_checked`: principle "true with the bare control run -- a no-efficiency null is valid only then."
- `null_delta_methodology_note`: principle "present when actions_delta==0 -- states the equality is an honest no-value null, not a bug."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (predictor on, ranking mode) -- the A6 input; 'unchanged' if null."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + rich_action_candidates importable); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4641: Action-Effect Expansion Prior For Live Search

Experiment 4641 SHALL write
`results/experiment_4641_action_effect_expansion_prior_live.json` after
graduating the proven `persistent_aem_plus_optional_cnn` action-effect predictor
from a candidate ranker into a live search expansion prior. The implementation
SHALL be imported by `arc_graph_explore`, SHALL be reachable from the scored
`E3AgentPolicy` / `StepwiseExplorer` path, and SHALL bias frontier branch
expansion by predicted frame-change for each frontier node's remaining
candidate actions. The existing 4629 ranker-only behavior SHALL remain
available as the matched control.

The expansion prior SHALL treat the action-effect predictor as a learned
action-pruner, not an oracle: it may use rendered frames, candidate action ids,
click coordinates, local PersistentAEM evidence, and the optional small CNN
forward pass, but it SHALL NOT inspect `env._game`, call a win-check to score
branches, require live LLM inference, or require 3090-class hardware. Broken or
missing scorer calls SHALL fail closed to the ranker-only ordering.

Experiment 4641 SHALL compare the expansion-prior live search against the
ranker-only baseline on the same held-out public-game cached transition groups,
reporting live solve-rate, depth-of-live-solve, median actions-to-win,
first-win-rate, `solve_rate_delta`, `depth_of_live_solve_delta`,
`first_win_rate_delta`, and bootstrap confidence intervals for the solve-rate
and depth deltas. A deeper-solve success verdict is allowed only when solve-rate
or live depth improves over the ranker-only baseline with the relevant CI
excluding zero, first-win-rate does not regress, parity stays green, and
`scripts/arc_orphan_solver_lint.py` passes. If the matched ranker-only control
passes but deltas are zero, the artifact SHALL report an honest null with
`null_delta_methodology_note` and leave `chosen_submitted_config` unchanged.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: action_effect_expansion_prior_live_deeper_solve_<n> OR complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the predictor is a small conv net (CPU/iGPU), declared so a fast forward-pass is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the action-effect expansion prior is a learned action-pruner, oracle-DISTINCT from the executable win-check (north-star section 5 action-pruner role)."
- `solve_provenance`: principle "live_agent_self_discovery -- this improves the SCORED live agent's OWN search expansion (arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the expansion-prior module is imported by arc_graph_explore (graph_explore_solve_v2) AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT orphaned)."
- `live_solve_rate_expansion`: principle "the HEADLINE -- LIVE solve-rate WITH the action-effect EXPANSION PRIOR on the SCORED agent."
- `live_solve_rate_ranker_baseline`: principle "the matched .427 ranker-only baseline solve-rate on the SAME variants (the no-regression control)."
- `solve_rate_delta`: principle "expansion - ranker_baseline (positive = the expansion prior deepened the live solve), emitted explicitly so a null (0) is annotated."
- `depth_of_live_solve_delta`: principle "max live level reached: expansion - ranker_baseline (the direct measure of converting first-win into a deeper solve -- the 2nd-level-up the wall sits at)."
- `first_win_rate_delta`: principle "expansion - ranker_baseline first-win-rate; emitted explicitly so a null is annotated (deepening must not cost first-wins)."
- `solve_rate_delta_ci`: principle "bootstrap CI on the solve-rate / depth delta; a deeper-solve claim requires the CI to exclude the ranker-only baseline."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the .427 ranker-only baseline ran on the SAME variants; a no-deeper-solve null is valid only then."
- `false_negative_risk_checked`: principle "true with the ranker-only baseline run + reachable-headroom confirmed -- a no-deeper-solve null is valid only then."
- `null_delta_methodology_note`: principle "present when a delta==0 -- states the equality is an honest no-value null, not a bug."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (expansion-prior mode) -- the A6 input; 'unchanged' if null."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + graph_explore_solve_v2 importable, .427 predictor artifact present); pre-empts missing-resource fabrication."

### REQ-ARC-FCP-4715: Goal-Free Online Action Learning Driver Corrected Build

Experiment 4715 SHALL write
`results/experiment_4715_online_action_learning_driver_corrected.json` after
installing the corrected goal-free online action-learning driver into the live
cascade. The driver SHALL remain additive to the existing `cascade=True`
goal-induction path: it may propose and rank actions through the scored
`E3AgentPolicy` / `StepwiseExplorer` import closure, but it SHALL NOT remove
the existing LLM goal-induction planner or banked-solution replay path.

The online action-effect driver SHALL use self-supervised binary frame-change
labels from the agent's own transitions, perform one CPU-safe Adam/BCE update
on approximately every five observed actions, use hash-deduped examples, and
expose a coordinate head that proposes top-k ACTION6 click coordinates. Plain
dict candidate rows and ArcAction-like objects SHALL both receive the CNN term
so the coordinate head is not silently bypassed. On any observed level increase,
the driver SHALL reset per-level buffer state, optimizer state, and trainable
CNN weights to the scorer's initial cross-game prior snapshot; this reset-to-
prior behavior SHALL be measured separately from a scratch/random arm.

The corrected build SHALL also flip the cheap online dynamics floor by default:
`gated_engine_from_transitions` SHALL default to `trust_metric="cell_recall"`
while preserving explicit callers that request `"exact"`. The experiment SHALL
record the live preconditions from the operator prompt: CUDA available for
offline arms, Qwen3.5-9B-MTP GGUF cached, offline arcade and the bug-fixed
Go-Explore archive importable, and `/props` verification that the proposer
port serves Qwen rather than Gemma.

Experiment 4715 SHALL compare the frozen, online-scratch, and online-warm
arms on the experiment 4605 held-out harness or a content-addressed reuse of
the corresponding completed arm artifacts. A success verdict is allowed only
when online-warm first-win rate exceeds frozen by at least +0.05 and a
goal-free lp85/sc25 multi-level probe reaches L2 and offline-reproduces via
`arc_solver_kit.reproduce`. If the arm delta is flat and no goal-free L2 is
reproduced, the artifact SHALL report a terminal `complete:` null, name the
residual cause, keep `verifier_is_oracle=false`, and leave the submitted config
unchanged except for explicitly safe additive floors.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: online_warm_beats_frozen_<delta>_l2_<game> OR complete: online_action_learning_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference for the live arm that loads the Qwen GGUF, or verifier_ensemble_against_cached_candidates for the offline held-out harness arm; model_specs name the GGUF."
- `online_warm_first_win`: principle "the +0.05 online-warm-over-frozen gate is the whole bet; the warm arm isolates online adaptation from scratch initialization."
- `online_scratch_first_win`: principle "the online-from-random arm isolates whether the win is online learning or warm start."
- `frozen_first_win`: principle "the frozen-prior baseline is the current submitted behavior and the no-online control."
- `online_warm_vs_frozen_delta`: principle "online_warm_first_win - frozen_first_win; >=+0.05 is the gate; emitted explicitly so a null is annotated."
- `cpu_train_step_ms`: principle "the Kaggle path is CPU under a 12h/600-RPM cap; an online step too slow to run every five actions makes the loop infeasible regardless of offline gains."
- `goal_free_l2_reached`: principle "a goal-free L2 deepening proves the wall is crossed by demoting goal-induction, not fixing it."
- `offline_reproduced`: principle "a goal-free L2 counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer level the goal-free driver reached on the multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic goal-free L2; development_proxy if an adapter or cached arm artifact was needed."
- `verifier_is_oracle`: principle "MUST be false -- the online frame-change CNN is oracle-distinct and does not run the win-check."
- `live_path_reachable`: principle "HARD gate -- the changed E3AgentPolicy/StepwiseExplorer path is in the scored agent import closure and arc_orphan_solver_lint passes."
- `bare_control_passed`: principle "the positive control proves the held-out harness has reachable first-win headroom before accepting a flat null."
- `false_negative_risk_checked`: principle "true only when the three arms ran or were content-addressed and reachable headroom was confirmed."
- `null_methodology_note`: principle "present when online_warm_vs_frozen_delta is approximately zero; states the equality is an honest no-lift null, not a measurement bug."
- `chosen_submitted_config`: principle "the recommended submitted-agent config change: online driver on, reset-to-prior, cell_recall un-gate; unchanged if null."
- `proposer_served_model`: principle "the model served by /props; MUST be Qwen3.5-9B-MTP to guard the port-8919 Gemma confound."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records CUDA, Qwen cache, offline arcade, bug-fixed archive, and /props served Qwen."

### REQ-ARC-FCP-4726: Valid Online Driver Non-Degeneracy Gate

Experiment 4726 SHALL write
`results/experiment_4726_online_action_learning_driver_valid_test.json` before
interpreting any held-out lift from the goal-free online action-learning
driver. The experiment SHALL first verify the hard resource preconditions:
CUDA for offline training arms, cached Qwen3.5-9B-MTP GGUF weights, the offline
ARC arcade plus the bug-fixed Go-Explore archive, and a `/props`-verified Qwen
proposer on a non-8919 port. If any precondition is absent, the artifact SHALL
record `preconditions_checked` and a terminal blocked verdict instead of
fabricating an A/B measurement.

The non-degeneracy gate SHALL compare the frozen, online-scratch, and
online-warm arms before measuring lift. The gate SHALL prove that the three arms
produce distinct per-arm action distributions, that online Adam steps actually
execute with positive gradient norms, and that the online coordinate head
proposes click cells that differ from the frozen prior's click head. If this
gate fails, the artifact SHALL report
`complete: online_driver_arms_degenerate_confirmed_harness_bug` with diagnostic
evidence and SHALL NOT treat a flat first-win delta as a capability null.

When the gate passes, Experiment 4726 SHALL aggregate the held-out first-win
rates for `{frozen, online-scratch, online-warm}` on the experiment 4605
color-permuted harness, measure one CPU online train-step latency, run the
live-path orphan lint and submitted-agent parity checks, and include the
bounded lp85/sc25 goal-free L2 probe. A success verdict requires
`online_warm_vs_frozen_delta >= 0.05` or an offline-reproduced goal-free L2.
Otherwise the artifact SHALL report an honest no-lift residual only when the
non-degeneracy gate, parity, and reachable-headroom positive control pass.

Required field principles:

- `honest_verdict`: principle "terminal prefix; success: online_warm_beats_frozen_<delta>_or_l2_<game> OR complete: online_driver_arms_degenerate_confirmed_harness_bug OR complete: online_action_learning_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference precondition for the Qwen GGUF plus verifier_ensemble_against_cached_candidates for the offline held-out arm artifacts."
- `arms_non_degenerate`: principle "first gate; true only when action distributions differ, online train steps have positive gradient norms, and online coordinate proposals differ from the frozen prior."
- `per_arm_action_distribution_distinct`: principle "explicit evidence the arm action histograms are not byte-identical."
- `online_train_steps_executed`: principle "positive-gradient Adam steps actually run; proves the online CNN trained."
- `online_warm_first_win`: principle "the +0.05 online-warm-over-frozen gate is the bet."
- `online_scratch_first_win`: principle "online-from-random arm isolates online learning from warm start."
- `frozen_first_win`: principle "frozen-prior baseline, the no-online control."
- `online_warm_vs_frozen_delta`: principle "online_warm_first_win - frozen_first_win; >=+0.05 is the first-win gate."
- `cpu_train_step_ms`: principle "CPU wall-clock for one online Adam/BCE step after about five actions."
- `goal_free_l2_reached`: principle "a goal-free L2 deepening proves the wall is crossed by demoting goal-induction."
- `offline_reproduced`: principle "a goal-free L2 counts only if offline-reproduced."
- `reproduced_levels`: principle "integer level reached by the goal-free multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic goal-free L2; development_proxy otherwise."
- `verifier_is_oracle`: principle "MUST be false; the online frame-change CNN does not run the win-check."
- `live_path_reachable`: principle "the changed E3AgentPolicy/StepwiseExplorer code is in the scored agent import closure."
- `bare_control_passed`: principle "positive control; held-out harness has reachable first-win headroom."
- `false_negative_risk_checked`: principle "true only with non-degenerate arms and reachable headroom."
- `null_delta_methodology_note`: principle "present when a flat non-degenerate delta is an honest no-lift null."
- `positive_control_passed`: principle "bool(parity_test_green AND arms_non_degenerate); gates the TAUTOLOGY null-delta exemption."
- `chosen_submitted_config`: principle "recommended submitted-agent config; unchanged for honest null."
- `proposer_served_model`: principle "the `/props`-reported model; MUST be Qwen3.5-9B-MTP."
- `parity_test_green`: principle "test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records CUDA, Qwen cache, offline arcade, Go-Explore import, and `/props` verification."

### REQ-ARC-FCP-5360: Live-Reachable Color-Blob Salience Level-Up Attempt

Experiment 5360 SHALL write
`results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json`
after a registry-prechecked ARC live-path attempt. The workflow SHALL audit the
live frame-diff/perception path for self-consistent but wrong reads, expose a
classical single-color connected-component/color-blob salience tier through the
live `E3AgentPolicy`/`StepwiseExplorer` candidate ordering path, and run a
bounded first-contact next-level attempt on a rotated shallow target such as
`re86`, `sb26`, `bp35`, or `lf52`.

The result artifact SHALL include `experiment_id`, `milestone`, `status`,
`honest_verdict`, `inference_substrate=live_arc_agent_policy`,
`solve_provenance=live_agent_self_discovery`,
`registry_precheck_completed`, `target_game`, `target_level_before`,
`perception_audit_completed`, `salience_policy_live_reachable`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`actions_to_first_levelup`, `perception_error_classes`, `outer_loop_re_used`,
`registry_updated`, and `tests_run`. A credited level-up SHALL require
`offline_reproduced=true`, `reproduced_levels>=1`, and
`outer_loop_re_used=false`; otherwise the artifact SHALL report an honest null
with exact blockers.

Required field principles:

- `experiment_id`: principle "Stable id ties the artifact to this roadmap task."
- `milestone`: principle "Satisfies the per-milestone ARC standing floor for `.488`."
- `status`: principle "Lets capstone distinguish banked level, honest null, or blocked live path."
- `honest_verdict`: principle "Terminal prefix `complete:` or `blocked_` prevents ambiguous ARC progress."
- `inference_substrate`: principle "Expected value is live_arc_agent_policy for credited progress."
- `solve_provenance`: principle "Must be live_agent_self_discovery for a credited ARC solve path."
- `registry_precheck_completed`: principle "Bare boolean prevents duplicate re-solving of already banked levels."
- `target_game`: principle "Names the rotated target game for coverage auditing."
- `target_level_before`: principle "Bare integer records pre-attempt reproduced depth."
- `perception_audit_completed`: principle "Bare boolean proves the known-issues priority was exercised."
- `salience_policy_live_reachable`: principle "Bare boolean proves the live agent can reach the new mechanism."
- `offline_reproduced`: principle "Bare boolean is the ARC level-up lint and registry gate."
- `reproduced_levels`: principle "Bare integer records new live-path level count; success gate includes reproduced_levels>=1."
- `new_level_banked`: principle "Bare boolean separates real progress from diagnostics."
- `actions_to_first_levelup`: principle "Bare integer or null measures action efficiency."
- `perception_error_classes`: principle "Lists observed perception failure modes."
- `outer_loop_re_used`: principle "Bare boolean must be false for credited live-path progress."
- `registry_updated`: principle "Bare boolean records whether solve registry changed."
- `tests_run`: principle "Lists live-path, registry, and salience-policy checks."

### REQ-ARC-FCP-5373: Live-Reachable Salience Repair Re86 Level-Up Attempt

Experiment 5373 SHALL write
`results/experiment_5373_arc_salience_re86_levelup_v489.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL select `re86` L3
when the registry still records only two reproduced `re86` levels; if that depth
is already banked, it SHALL choose another non-duplicate live-path target and
record the reason. The repair SHALL be reachable from the live
`E3AgentPolicy`/`StepwiseExplorer` path and SHALL improve the Exp5360 salience
error classes by deprioritizing status bars and large flat blobs, ranking
button-like blobs ahead of flat distractors, and gating frame-diff action-effect
scores until at least one observed transition validates the scorer against real
frame change.

The result artifact SHALL include `status`, `solve_provenance`,
`registry_precheck_done`, `target_game`, `target_level_before`,
`attempted_level`, `salience_repair_live_reachable`,
`status_bar_deprioritization_enabled`,
`frame_diff_ground_truth_validated`, `button_like_blob_rank_delta`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`registry_total_before`, `registry_total_after`, `live_attempt_count`,
`perception_error_classes`, `no_outer_loop_re`, `no_duplicate_solve`, and
`honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true`,
`reproduced_levels>=1`, `new_level_banked=true`, `no_outer_loop_re=true`, and
`no_duplicate_solve=true`; otherwise the artifact SHALL report an honest null
with residual perception/salience error classes.

Required field principles:

- `status`: principle "complete or honest_null; never claim a solve without registry-compatible evidence."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited solves."
- `registry_precheck_done`: principle "must be true before target selection or attempts."
- `target_game`: principle "target game id after registry precheck."
- `target_level_before`: principle "reproduced level count before the attempt."
- `attempted_level`: principle "level attempted for +1 deeper progress."
- `salience_repair_live_reachable`: principle "true only if the repair is in the live agent path."
- `status_bar_deprioritization_enabled`: principle "whether the repair addresses the .488 status-bar error class."
- `frame_diff_ground_truth_validated`: principle "whether frame-diff salience is validated before committing probes."
- `button_like_blob_rank_delta`: principle "measured ranking change for button-like blobs if available."
- `offline_reproduced`: principle "true only when a new level is banked by the accepted registry/evidence path; include this exact field for ARC lint."
- `reproduced_levels`: principle "number of newly reproduced levels; success requires reproduced_levels>=1."
- `new_level_banked`: principle "true only if registry-compatible evidence banks a level not already present before this task."
- `registry_total_before`: principle "total reproduced levels before the attempt."
- `registry_total_after`: principle "total reproduced levels after the attempt."
- `live_attempt_count`: principle "number of live attempts made."
- `perception_error_classes`: principle "residual perception/salience errors observed."
- `no_outer_loop_re`: principle "must be true for credited solve."
- `no_duplicate_solve`: principle "must be true."
- `honest_verdict`: principle "one-line banked/no-bank verdict."

### REQ-ARC-FCP-5385: Geometric Salience Live-Path Level-Up Attempt

Experiment 5385 SHALL write
`results/experiment_5385_arc_geometric_salience_live_path_v490.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL prefer `re86` L3
when the registry still records fewer than three reproduced `re86` levels; if
that depth is already reproducible, it SHALL choose a non-duplicate live-path
target or emit a duplicate-blocked artifact. The salience mechanism SHALL be a
live-reachable geometric, hyperbolic, or geodesic ranking signal over the
agent-observed frame and its own observed transition stream, and SHALL NOT use
offline ground-truth BFS, public-game source inspection, per-game adapters, or
outer-loop reverse engineering for a credited solve.

The result artifact SHALL include `status`, `solve_provenance`,
`registry_precheck_done`, `target_game`, `target_level_before`,
`attempted_level`, `geometric_salience_live_reachable`,
`hyperbolic_or_geodesic_ranking_enabled`, `live_attempt_count`,
`offline_reproduced`, `no_outer_loop_re`, `no_per_game_adapter`,
`no_duplicate_solve`, `reproduced_levels`, `new_level_banked`,
`failure_mode`, and `honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true` from
replaying a live-agent-discovered solution, `no_outer_loop_re=true`,
`no_per_game_adapter=true`, `no_duplicate_solve=true`, and
`reproduced_levels >= target_level_before + 1`; otherwise the artifact SHALL
report an honest null or duplicate block with the live-path blocker.

Required field principles:

- `status`: principle "complete, honest_null, or duplicate_blocked with evidence."
- `solve_provenance`: principle "must equal live_agent_self_discovery for any credited solve."
- `registry_precheck_done`: principle "must be true."
- `target_game`: principle "selected game id."
- `target_level_before`: principle "reproduced level count before this task."
- `attempted_level`: principle "target level attempted."
- `geometric_salience_live_reachable`: principle "true only if the live agent can use the salience signal without outer-loop help."
- `hyperbolic_or_geodesic_ranking_enabled`: principle "whether the GeoWorld-inspired ranking was active."
- `live_attempt_count`: principle "number of live attempts."
- `offline_reproduced`: principle "true only if a live-agent-discovered solve was replayed/reproduced for banking; must not mean offline BFS or outer-loop reverse engineering."
- `no_outer_loop_re`: principle "must be true."
- `no_per_game_adapter`: principle "must be true."
- `no_duplicate_solve`: principle "must be true for credited deliverables."
- `reproduced_levels`: principle "level count after this task; for a credited first-contact solve this must satisfy reproduced_levels>=1, and for a deeper solve it must be at least target_level_before+1."
- `new_level_banked`: principle "true only if the live agent self-discovered a new reproducible level."
- `failure_mode`: principle "if no bank, concrete live-path blocker."
- `honest_verdict`: principle "one-line ARC outcome."

### REQ-ARC-FCP-5397: Blob Salience Generation-Stage Live-Path Level-Up Attempt

Experiment 5397 SHALL write
`results/experiment_5397_arc_blob_salience_live_path_v491.json` after a
registry-prechecked ARC live-path attempt. The workflow SHALL prefer `re86` L3
when the registry still records fewer than three reproduced `re86` levels; if
that depth is already reproducible, it SHALL choose the next unsolved reachable
target and SHALL NOT duplicate a prior solved level. The salience mechanism
SHALL segment rendered frames into single-color connected components, mask or
deprioritize status-bar regions, classify blobs into tiers from button
likelihood, salient color, size, and non-status evidence, and SHALL apply those
tiers in the live `E3AgentPolicy`/`StepwiseExplorer` generation stage before the
click candidate cap. It SHALL NOT use a per-game adapter, offline BFS,
outer-loop reverse engineering, or a hand-coded game model for a credited solve.

The result artifact SHALL include `status`, `milestone`, `target_game`,
`attempted_level`, `registry_precheck_done`, `duplicate_solve_avoided`,
`solve_provenance`, `live_agent_policy_modified`,
`connected_component_salience_enabled`, `salience_tiers_emitted`,
`per_game_adapter_used`, `offline_bfs_used`, `outer_loop_re_used`,
`live_attempt_count`, `offline_reproduced`, `reproduced_levels`,
`new_level_banked`, `failure_mode`, and `honest_verdict`. A credited level-up
SHALL require `status=complete`, `solve_provenance=live_agent_self_discovery`,
`offline_reproduced=true`, `reproduced_levels>=1`, `new_level_banked=true`,
`per_game_adapter_used=false`, `offline_bfs_used=false`, and
`outer_loop_re_used=false`; otherwise the artifact SHALL report `honest_null`
with a concise failure mode, or `blocked` only when harness access is missing.

Required field principles:

- `status`: principle "complete for a banked +1 level, honest_null for a real no-bank attempt, or blocked for missing harness access."
- `milestone`: principle "must equal 2026.07.491."
- `target_game`: principle "game selected after registry precheck."
- `attempted_level`: principle "level attempted after registry precheck."
- `registry_precheck_done`: principle "must be true."
- `duplicate_solve_avoided`: principle "must be true."
- `solve_provenance`: principle "must be live_agent_self_discovery for a credited solve."
- `live_agent_policy_modified`: principle "true only if E3AgentPolicy generation-stage action prioritization was changed."
- `connected_component_salience_enabled`: principle "true if the blob salience mechanism was active."
- `salience_tiers_emitted`: principle "true if action tiers were logged."
- `per_game_adapter_used`: principle "must be false."
- `offline_bfs_used`: principle "must be false."
- `outer_loop_re_used`: principle "must be false."
- `live_attempt_count`: principle "count of live harness attempts."
- `offline_reproduced`: principle "true only if the live-discovered new level is reproduced."
- `reproduced_levels`: principle "number of newly reproduced levels, success requires reproduced_levels>=1."
- `new_level_banked`: principle "true only for a +1 reproducible level."
- `failure_mode`: principle "null on success or concise no-bank reason."
- `honest_verdict`: principle "one-line summary starting with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5410: Live Trajectory Frontier Self-Discovery Level-Up Attempt

Experiment 5410 SHALL write
`results/experiment_5410_arc_live_trajectory_frontier_levelup_v492.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `re86` L3 when that level is not already live/reproduction reached; if
the registry already records the target level, it SHALL choose the next eligible
target or emit a blocked duplicate-solve artifact without replaying the solved
level. The mechanism SHALL reuse Exp5397's connected-component blob/color
salience only as a live perception route, SHALL generate action prefixes from
the live agent's own attempted actions and observed runtime transitions, and
SHALL reject or cautiously withhold low-support inferred dynamics through an
uncertainty gate. It SHALL NOT use offline BFS, game source inspection,
per-game adapters, hidden-source probes, or outer-loop reverse engineering for
credited progress.

The result artifact SHALL include `registry_precheck_done`, `target_game`,
`target_level`, `solve_provenance`, `offline_reproduced`, `attempt_count`,
`frontier_expansion_count`, `salience_routes_used`,
`uncertainty_rejections`, `reproduced_levels`, `arc_new_level_banked`,
`duplicate_solve_avoided`, `no_offline_bfs`, `no_per_game_adapter`,
`inference_substrate`, and `honest_verdict`. A credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`arc_new_level_banked=true`, `duplicate_solve_avoided=true`,
`no_offline_bfs=true`, and `no_per_game_adapter=true`; otherwise the artifact
SHALL report `honest_null:` with bounded live-attempt evidence or `blocked:`
when the duplicate/precondition gate prevents a valid attempt.

Required field principles:

- `registry_precheck_done`: principle "bare bool proving no duplicate solve attempt starts before reading the registry."
- `target_game`: principle "selected game id for reproducibility."
- `target_level`: principle "selected target level label for reproducibility."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited progress."
- `offline_reproduced`: principle "legacy ARC lint field; true only for a live-agent self-discovered registry-compatible new level."
- `attempt_count`: principle "bounded live-agent effort count."
- `frontier_expansion_count`: principle "number of trajectory/frontier prefixes emitted by the new mechanism."
- `salience_routes_used`: principle "auditable blob/color salience routes used by prefix generation."
- `uncertainty_rejections`: principle "low-support inferred dynamics rejected by the gate."
- `reproduced_levels`: principle "registry-compatible new level count."
- `arc_new_level_banked`: principle "standing ARC floor success flag."
- `duplicate_solve_avoided`: principle "registry discipline prevents duplicate solved-level credit."
- `no_offline_bfs`: principle "must be true; forbidden solve path was not used."
- `no_per_game_adapter`: principle "must be true; no hand per-game shortcut was used."
- `inference_substrate`: principle "must be offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

## Scenarios

### SCENARIO-ARC-FCP-4490: Positive-Control Candidate Ranking

Given a frame and candidate actions where only one click cell is known to
change the frame
When the behavior prior or scorer ranks the candidates
Then the changing click is ordered ahead of no-op candidates
And the legacy candidate order remains the tie-break for equal scores.

### SCENARIO-ARC-FCP-4726: Online Driver Arms Must Be Non-Degenerate Before Lift

Given the frozen, online-scratch, and online-warm goal-free driver arms are
available and the Qwen/arcade/CUDA preconditions pass
When experiment 4726 runs the valid-driver gate before aggregating held-out
first-win rates
Then it records distinct per-arm action distributions, positive-gradient
online Adam train steps, and online coordinate-head click proposals that differ
from the frozen prior before setting `arms_non_degenerate=true`.

Given any of those non-degeneracy witnesses is missing
When experiment 4726 writes its artifact
Then it reports
`complete: online_driver_arms_degenerate_confirmed_harness_bug`, keeps
`chosen_submitted_config` unchanged, and does not treat a flat first-win delta
as a capability null.

### SCENARIO-ARC-FCP-4629: Submitted E3 Uses Live-Reachable Action-Effect Ranking

Given local cached transition effects and a small trained CNN action-effect
checkpoint are available
When the submitted `E3AgentPolicy` builds its `StepwiseExplorer`
Then the explorer receives a live action-effect scorer by default
And `rich_action_candidates` applies that scorer as the final ordering pass
after any candidate-router tie-break
And experiment 4629 writes a terminal artifact with the matched bare control,
bootstrap action-delta CI, parity-test result, orphan-lint result,
`verifier_is_oracle=false`, and `solve_provenance=live_agent_self_discovery`.

### SCENARIO-ARC-FCP-4641: Submitted E3 Uses Live-Reachable Expansion Priority

Given the 4629 live action-effect scorer is available
When `graph_explore_solve_v2` or the submitted `E3AgentPolicy` has multiple
frontier branches with remaining untested actions
Then the branch whose remaining actions have higher predicted frame-change is
expanded before predicted no-op branches
And the ranker-only mode remains available as the matched control
And experiment 4641 writes a terminal artifact with live solve-rate,
depth-of-live-solve, median actions-to-win, first-win-rate deltas, bootstrap
CIs, parity-test result, orphan-lint result, `verifier_is_oracle=false`, and
`solve_provenance=live_agent_self_discovery`.

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

### SCENARIO-ARC-FCP-4547: CNN Ranker Beats Only With Positive Control

Given the cached ARC Public Demo human replay shards are present and
`arc_solver_kit.offline_arcade()` imports successfully
When experiment 4547 trains the small CNN on frame-only action-effect rows and
measures CNN-ranked candidates against the matched blind-BFS order
Then the artifact records the required median action, solve-rate, AUROC,
positive-control, random-seed, checksum, and precondition fields, and a success
verdict is emitted only when the CNN median actions-to-first-level-up is
strictly lower than blind BFS while preserving solve-rate.

Given the CNN does not reduce held-out median actions
When the held-out transition-delta AUROC is above the trivial baseline
Then the artifact reports `complete: frame_change_cnn_no_action_reduction_honest_null`,
sets `false_negative_risk_checked=true`, and records hidden-field probing for
the `ka59`/`ar25` L2 stall as the secondary input.

### SCENARIO-ARC-FCP-4568: Pooled Clickability Predictor Emits Honest Efficiency Delta

Given the local human replay shards and local transition corpus are present
When experiment 4568 trains the small frame-only action-effect CNN and measures
predictor-ranked candidates against the matched blind order
Then the artifact records the required median actions, explicit action delta,
bootstrap CI, efficiency score, generic-transfer fields, non-oracle verifier
flag, positive control, null note when needed, checksum, and preconditions, and
emits a success verdict only when the action delta is positive with no
solve-rate drop.

Given the predictor does not reduce held-out median actions
When the positive control passes
Then the artifact reports
`complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened`,
sets `false_negative_risk_checked=true`, keeps `chosen_submitted_config` as
`unchanged`, and records the residual missing-verifier gaps.

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

### SCENARIO-ARC-FCP-4575: Learned-CNN Duration Floor

Given a fixture artifact with a CNN/torch marker, `duration_s` around five
seconds, no LLM/GGUF marker, and
`inference_substrate=verifier_ensemble_against_cached_candidates`
When adversarial verification and the summary reader process the artifact
Then the applied floor is the one-second verifier-scoring floor and no
`DURATION_TOO_SHORT` flag is emitted.

Given a fixture artifact with `inference_substrate=live_llm_inference`, a GGUF
marker, and `duration_s < 60`
When adversarial verification processes the artifact
Then it emits a critical `DURATION_TOO_SHORT` flag.

### SCENARIO-ARC-FCP-5360: Color-Blob Salience Is Live-Reachable

Given a rendered ARC frame with single-color connected components that include
a button-like salient-color blob, a larger dull blob, and a status-bar-colored
blob
When the color-blob salience prior scores live `rich_action_candidates`
Then the button-like blob is ranked ahead of the dull and status-bar blobs.

### SCENARIO-ARC-FCP-5373: Salience Repair Is Validated Before Probing

Given a rendered ARC frame with a button-like blob, a large flat distractor, and
a status-bar-like strip
When the live salience repair scores `rich_action_candidates`
Then the button-like blob is ranked ahead of both flat/status distractors and
the artifact records the measured rank delta.

Given a frame-diff scorer assigns high confidence before any observed transition
has confirmed that high scores predict real frame change
When the submitted `E3AgentPolicy` asks for live candidate ordering
Then the frame-diff score is gated out until an observed transition validates it
against the actual before/after frame pixels.

Given the submitted `E3AgentPolicy` constructor builds its `StepwiseExplorer`
without an explicit action prior
When the live policy is instantiated
Then the explorer receives the classical color-blob salience prior and the
submitted config declares the mechanism live-reachable.

Given the registry records `re86`, `sb26`, `bp35`, and `lf52` at shallow depth
2
When experiment 5360 performs its registry precheck
Then it selects the next unbanked target level rather than duplicating an
already reproduced L1/L2 solve, runs the perception audit, and writes an
artifact whose credited progress fields only pass when offline reproduction
banks at least one new live-path level.

### SCENARIO-ARC-FCP-5385: Geometric Salience Stays On The Live Path

Given a rendered ARC frame with two same-tier button-like blobs and a later
agent-observed transition changing cells near only one of those blobs
When the geometric salience prior ranks live `rich_action_candidates`
Then the geodesic transition anchor moves the nearer blob ahead of the equal
base-salience distractor and the prior reports hyperbolic/geodesic ranking as
enabled.

Given the submitted `StepwiseExplorer` has an action prior that can observe
transitions
When the live policy ingests its own before/action/after observation
Then the action prior receives that observation through the same live path that
feeds the frame-change scorer.

Given `ops/arc_solve_registry.yaml` records `re86` below L3
When Experiment 5385 performs its registry precheck
Then it selects `re86` L3, marks duplicate credit disallowed, and only sets
`new_level_banked=true` if the live-agent-discovered action labels reproduce
beyond the prior level.

### SCENARIO-ARC-FCP-5397: Blob Tiers Shape Live Candidate Generation

Given a rendered ARC frame containing a status strip, a large flat colored
region, and a compact salient button-like component
When the live `StepwiseExplorer` asks `rich_action_candidates` for click
candidates with the connected-component salience prior enabled
Then the compact salient button-like component is generated before the click cap
can drop it, status-bar components are pushed to the lowest tier, and the live
explorer records emitted salience tiers.

Given `ops/arc_solve_registry.yaml` records `re86` below L3
When Experiment 5397 performs its registry precheck
Then it selects `re86` L3, marks duplicate solve avoidance complete, runs a
bounded live-agent attempt, and only emits `complete:` when a
live-agent-discovered new level is reproduced offline.

### SCENARIO-ARC-FCP-5410: Trajectory Prefixes Are Live-Observed And Uncertainty-Gated

Given the live agent observes repeated frame-changing button-like blob clicks
from its own attempts
When the trajectory frontier generator is asked for a multi-action prefix
Then it emits only prefixes supported by the observed transition stream,
records the blob/color salience route, and increments frontier expansion
evidence.

Given a single low-support or conflicting inferred dynamic
When the same generator is asked to promote that dynamic into a prefix
Then the uncertainty gate rejects it, records an uncertainty rejection, and
keeps `offline_reproduced=false` unless a live self-discovered solution later
passes the reproduction gate.
