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
`tests/python/test_arc_submitted_agent_parity.py` that records the CURRENT
`SUBMITTED_AGENT_CONFIG["value_weight"]` (originally frozen as
`value_weight==0.0` at the .415 B2 milestone; commit `0fad75f38`, PHASE A1 /
REQ-LEARN-4652, later deliberately moved it to a tiny bounded-positive value
once a compute-cost fix made that route affordable -- a legitimate policy
evolution the scoreboard now tracks dynamically rather than re-asserting the
stale literal), and make the real leaderboard signal the pair
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

### REQ-ARC-FCP-5423: CoEx Landmark Live Self-Discovery Level-Up Attempt

Experiment 5423 SHALL write
`results/experiment_5423_arc_coex_landmark_levelup_v493.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `lf52` L3 when the registry records fewer than three reproduced `lf52`
levels; otherwise it SHALL choose the nearest eligible unbanked frontier from
`ops/arc_solve_registry.yaml` or emit a blocked duplicate-solve artifact. The
credited mechanism SHALL run through the live ARC agent runtime by feeding a
CoEx-style persistent frontier generator into the `E3AgentPolicy` action-prior
and QD sequence hooks. The generator SHALL learn only from the agent's own
runtime transitions, persist frontier states across resets, decompose observed
progress into hierarchical landmarks, cluster action histories, and emit
measurement-access receipts for the observations used to promote any prefix.
It SHALL NOT inspect hidden game source, run offline ground-truth BFS as the
credited solve path, or create a per-game adapter/calibration solver as the
headline path.

The result artifact SHALL include `registry_precheck`, `target_game`,
`target_level`, `duplicate_solve_avoided`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `arc_new_level_banked`,
`attempt_count`, `frontier_expansion_count`, `landmark_count`,
`action_sequence_receipts`, `no_offline_bfs`, `no_per_game_adapter`,
`arc_levelup_lint_passed`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require `solve_provenance=live_agent_self_discovery`,
`inference_substrate=live_arc_agent_runtime`, `offline_reproduced=true`,
`reproduced_levels>=1`, `arc_new_level_banked=true`,
`duplicate_solve_avoided=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, at least one frontier expansion, at least one
landmark or explicit no-landmark receipt, and at least one replayable action
sequence receipt. Otherwise the artifact SHALL report `honest_null:` with
bounded live-attempt evidence or `blocked:` when the duplicate/precondition gate
prevents a valid attempt.

Required field principles:

- `registry_precheck`: principle "bare bool proving duplicate-solve avoidance ran before any live attempt."
- `target_game`: principle "selected game id with registry provenance."
- `target_level`: principle "selected target level label with registry provenance."
- `duplicate_solve_avoided`: principle "true only when the target level was not already banked."
- `solve_provenance`: principle "must be live_agent_self_discovery for credited progress."
- `offline_reproduced`: principle "true only after the live-discovered sequence passes the offline reproduction gate."
- `reproduced_levels`: principle "new reproduced level count; complete requires at least one."
- `arc_new_level_banked`: principle "north-star metric flag for a newly banked level."
- `attempt_count`: principle "bounded live-agent action count."
- `frontier_expansion_count`: principle "CoEx persistent frontier prefixes emitted by the live mechanism."
- `landmark_count`: principle "hierarchical landmark count discovered from runtime observations."
- `action_sequence_receipts`: principle "replayable live action sequences and measurement receipts used for reproduction."
- `no_offline_bfs`: principle "must be true; forbidden offline ground-truth BFS was not used."
- `no_per_game_adapter`: principle "must be true; no hand per-game adapter/calibration solver was used."
- `arc_levelup_lint_passed`: principle "roadmap guarantee lint result recorded when practical."
- `inference_substrate`: principle "must be live_arc_agent_runtime."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5437: Registry-Guided Live Reinduction Level-Up Attempt

Experiment 5437 SHALL write
`results/experiment_5437_arc_live_reinduction_levelup_v494.json` after a
registry-prechecked bounded ARC live-agent level-up attempt. The workflow SHALL
prefer `cn04` L4 when the registry records exactly three reproduced `cn04`
levels and SHALL prefer `vc33` L3 when `cn04` is not eligible and the registry
records exactly two reproduced `vc33` levels. If neither preferred frontier is
eligible, it SHALL choose the nearest eligible unbanked next-level frontier from
`ops/arc_solve_registry.yaml` or emit a blocked duplicate-solve artifact before
spending live-agent budget. The credited mechanism SHALL run through the live
ARC agent runtime and use registry-guided per-level reinduction evidence,
runtime observation clustering, generic verifier routing, and
measurement-access receipts from the agent's own attempts. It SHALL NOT inspect
hidden game source, run offline ground-truth BFS as the credited solve path, or
create a per-game adapter/calibration solver as the headline path.

The result artifact SHALL include `registry_precheck`, `target_game`,
`target_level`, `duplicate_solve_avoided`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `arc_new_level_banked`,
`attempt_count`, `frontier_expansion_count`, `runtime_predicate_count`,
`action_sequence_receipts`, `no_offline_bfs`, `no_per_game_adapter`,
`arc_levelup_lint_passed`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require `solve_provenance=live_agent_self_discovery`,
`inference_substrate=live_arc_agent_runtime`, `offline_reproduced=true`,
`reproduced_levels>=1`, `arc_new_level_banked=true`,
`duplicate_solve_avoided=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, at least one runtime predicate or frontier
transition from the live attempt, and at least one replayable action sequence
receipt. Otherwise the artifact SHALL report `honest_null:` with bounded
live-attempt evidence or `blocked:` when the duplicate/precondition gate
prevents a valid attempt.

Required field principles:

- `registry_precheck`: principle "duplicate-solve avoidance"
- `target_game`: principle "target provenance"
- `target_level`: principle "target provenance"
- `duplicate_solve_avoided`: principle "no already-banked headline"
- `solve_provenance`: principle "credited path"
- `offline_reproduced`: principle "reproducible solve gate"
- `reproduced_levels`: principle "level-up gate; must be >=1 for a banked solve"
- `arc_new_level_banked`: principle "north-star metric"
- `attempt_count`: principle "effort accounting"
- `frontier_expansion_count`: principle "mechanism evidence"
- `runtime_predicate_count`: principle "reinduction evidence"
- `action_sequence_receipts`: principle "reproducibility"
- `no_offline_bfs`: principle "live-path discipline"
- `no_per_game_adapter`: principle "live-path discipline"
- `arc_levelup_lint_passed`: principle "roadmap guarantee evidence"
- `inference_substrate`: principle "actual live agent"
- `honest_verdict`: principle "terminal status; start with complete: or honest_null: or blocked:"

### REQ-ARC-FCP-5450: Measurement-Access Live Level-Up Target Rotation

Experiment 5450 SHALL write
`results/experiment_5450_arc_measurement_access_live_levelup_v495.json` after a
registry-prechecked bounded ARC live-path +1 level-up attempt. The workflow
SHALL precheck all public-game frontier depths from
`ops/arc_solve_registry.yaml` and the current `results/arc_loop_solve*.json`
artifacts before selecting a target. It SHALL rotate away from stale recent
no-bank targets including `cn04` L4 and repeated `re86` L3 salience attempts
unless the selected live mechanism explicitly addresses their recorded residual
gap. The credited mechanism SHALL use only the live agent's own runtime
measurement access: frame-level measurements, action-effect observations,
state-change summaries, and verifier-routed predicates induced from attempted
transitions. It SHALL NOT read hidden game source, run offline ground-truth BFS
as the credited path, or credit a hand per-game `GameAdapter` as the solve path.

The result artifact SHALL include `solve_provenance`,
`registry_precheck_total_levels`, `selected_game`, `selected_target_level`,
`target_rotation_reason`, `live_attempt_count`,
`runtime_predicates_induced`, `offline_reproduced`, `reproduced_levels`,
`new_levels_banked`, `new_level_reproduced`, `no_offline_bfs`,
`no_source_reading`, `no_per_game_adapter_credited`,
`arc_new_level_banked`, `inference_substrate`, and `honest_verdict`. A
credited level-up SHALL require
`solve_provenance=live_agent_self_discovery`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`new_levels_banked>=1`, `new_level_reproduced=true`,
`arc_new_level_banked=true`, `no_offline_bfs=true`,
`no_source_reading=true`, `no_per_game_adapter_credited=true`, a selected
target level deeper than the registry precheck, and at least one runtime
predicate induced from the live attempt. Otherwise the artifact SHALL report
`honest_null:` with frontier expansions, attempted predicates, and the residual
wall, or `blocked:` when preconditions prevent a valid attempt.

Required field principles:

- `solve_provenance`: principle "live_agent_self_discovery -- credited path is the live agent's own attempts."
- `registry_precheck_total_levels`: principle "duplicate-solve prevention."
- `selected_game`: principle "target audit."
- `selected_target_level`: principle "target audit."
- `target_rotation_reason`: principle "no stale rerun."
- `live_attempt_count`: principle "live effort evidence."
- `runtime_predicates_induced`: principle "credited mechanism evidence."
- `offline_reproduced`: principle "official reproduction gate."
- `reproduced_levels`: principle "level-up acceptance."
- `new_levels_banked`: principle "north-star movement."
- `new_level_reproduced`: principle "lint-readable solve gate."
- `no_offline_bfs`: principle "no outer-loop solve."
- `no_source_reading`: principle "no hidden source path."
- `no_per_game_adapter_credited`: principle "live mechanism only."
- `arc_new_level_banked`: principle "capstone field."
- `inference_substrate`: principle "explicit runtime path."
- `honest_verdict`: principle "terminal status; start with complete: or honest_null: or blocked:."

### REQ-ARC-FCP-5464: ARC Metric-Integrity And Perception Precheck

Experiment 5464 SHALL write
`results/experiment_5464_arc_metric_integrity_perception_precheck_v496.json`
as a live-path precheck that does not claim a level solve. The workflow SHALL
read `ops/arc_solve_registry.yaml`, current reproduced public-game levels, and
recent ARC no-bank artifacts before selecting Exp5465 targets. It SHALL reject
duplicate-depth solve credit, reject off-path or source-derived provenance,
check whether one-step null-coordinate/no-coordinate trajectories contaminate
banked reproduced levels, and emit a target shortlist that avoids already
reached levels and recent duplicate no-bank lanes unless explicitly justified.

The workflow SHALL also exercise live-agent reachable perception diagnostics
for connected components, color blobs, changed pixels, salience tiers, and
action-effect observations through the submitted
`E3AgentPolicy`/`StepwiseExplorer` salience path. The perception receipts SHALL
be written as JSON and referenced by the main artifact. This precheck SHALL
not modify `ops/arc_solve_registry.yaml` and SHALL use
`inference_substrate=live_path_precheck_no_solve_claim`.

The result artifact SHALL include `registry_precheck_performed`,
`reproduced_total_levels_before`, `duplicate_solve_rejected`,
`off_path_solve_rejected`, `null_coordinate_exploit_valid`,
`perception_feature_receipts_path`, `target_shortlist`,
`recent_no_bank_targets_avoided_or_justified`,
`arc_metric_integrity_ready`, `inference_substrate`, and
`honest_verdict`. The artifact SHALL set
`arc_metric_integrity_ready=true` only when the registry precheck ran,
duplicate and off-path probe claims were rejected, no null-coordinate exploit
is valid for the reproduced-level metric, the perception receipts file exists,
and at least one shortlist target remains after recent no-bank avoidance.

Required field principles:

- `registry_precheck_performed`: principle "bare bool proving registry/public-game precheck ran before Exp5465 target selection."
- `reproduced_total_levels_before`: principle "authoritative registry total before this no-solve precheck."
- `duplicate_solve_rejected`: principle "bare bool: duplicate-depth solve claims fail closed and do not increment metrics."
- `off_path_solve_rejected`: principle "bare bool: source-derived, replay-only, or outer-loop provenance cannot receive live solve credit."
- `null_coordinate_exploit_valid`: principle "bare bool: true only if a banked reproduced level is validly explained by the null-coordinate exploit; metric-ready requires false."
- `perception_feature_receipts_path`: principle "path to JSON receipts for connected components, color blobs, changed pixels, salience tiers, and action-effect observations."
- `target_shortlist`: principle "Exp5465 candidates selected after avoiding already reached levels and recent duplicate no-bank lanes."
- `recent_no_bank_targets_avoided_or_justified`: principle "auditable list of recent no-bank targets avoided or any explicit justification for including one."
- `arc_metric_integrity_ready`: principle "true only when duplicate, provenance, null-coordinate, perception, and target-rotation gates are all clean."
- `inference_substrate`: principle "must equal live_path_precheck_no_solve_claim."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked: with no level-solve claim."

### REQ-ARC-FCP-5465: Gated Connected-Component Salience Level-Up Attempt

Experiment 5465 SHALL write
`results/experiment_5465_gated_arc_connected_component_salience_levelup_v496.json`
only after the Exp5464 metric-integrity precheck reports
`arc_metric_integrity_ready=true`. The workflow SHALL re-run the registry
precheck, select one target from Exp5464's `target_shortlist`, exercise
live-agent reachable connected-component, color-blob, changed-pixel,
salience-tier, and action-effect features, and run a bounded live/offline
agent attempt using the submitted `E3AgentPolicy`/`StepwiseExplorer` salience
path. It SHALL NOT read hidden game source, run offline ground-truth BFS, or
credit a hand-built per-game adapter as the solve path.

If the live attempt reaches a candidate new level, the workflow SHALL reproduce
that candidate through `arc_solver_kit.reproduce` or the official
`scripts/arc_loop_solve.py` reproduction gate before claiming progress. Success
SHALL require `offline_reproduced=true` and `reproduced_levels` strictly
greater than the selected target's registry-precheck depth. Otherwise the
artifact SHALL report an honest null while preserving the live-attempt feature
receipts and prohibited-input flags.

The result artifact SHALL include `solve_provenance`,
`registry_precheck_performed`, `target_game`, `target_level_before`,
`target_level_attempted`, `live_attempt_count`, `perception_features_used`,
`source_reading_used`, `offline_bfs_used`, `hand_adapter_credited`,
`offline_reproduced`, `reproduced_levels`, `new_level_banked`,
`arc_registry_update_required`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `solve_provenance`: principle "live_agent_self_discovery; credited path is the live agent's own attempts plus runtime reverse engineering."
- `registry_precheck_performed`: principle "bare bool proving the registry was re-read before selecting a target from Exp5464's shortlist."
- `target_game`: principle "game selected from Exp5464 target_shortlist after the rerun precheck."
- `target_level_before`: principle "registry-precheck reproduced depth for the selected target before this attempt."
- `target_level_attempted`: principle "one deeper level attempted by the live/offline agent."
- `live_attempt_count`: principle "bounded count of live-agent attempts actually executed."
- `perception_features_used`: principle "auditable list containing connected_component, color_blob, changed_pixel, salience_tier, and action_effect when exercised."
- `source_reading_used`: principle "must be false; hidden/public game source is not credited in this live self-discovery path."
- `offline_bfs_used`: principle "must be false; exhaustive offline ground-truth BFS is not the credited solve path."
- `hand_adapter_credited`: principle "must be false; a hand GameAdapter may not receive live-agent self-discovery credit."
- `offline_reproduced`: principle "true only when the live-agent candidate reproduces through the official reproduction gate beyond the precheck depth."
- `reproduced_levels`: principle "absolute reproduced level count after the gate; success requires this to exceed target_level_before."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels > target_level_before."
- `arc_registry_update_required`: principle "true only when a newly banked level should update ops/arc_solve_registry.yaml."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5479: Target Rotation Live-Path Precheck Without Solve Claim

Experiment 5479 SHALL write
`results/experiment_5479_arc_target_rotation_precheck_v497.json` as a
no-solve ARC target-rotation precheck before any level-up attempt. The workflow
SHALL read `ops/arc_solve_registry.yaml`, the Exp5464 precheck shortlist, the
Exp5465 no-bank artifact, and `ops/known-issues.md`; reject target levels that
the live mechanism already reproduces; avoid recent no-bank targets `bp35:L3`,
`ka59:L2`, and `cn04:L4` unless a new mechanism and target level are recorded;
and prefer a rotated Exp5464 shortlist target such as `sb26:L3`, `g50t:L3`,
`dc22:L3`, or `sp80:L3`.

The workflow SHALL verify live-path eligibility through frame-only
connected-component/color-blob salience diagnostics without reading hidden game
source, running offline ground-truth BFS, or crediting a hand per-game adapter.
The bounded dry check SHALL report connected components, color blobs, changed
cells, target-region candidates, and known blockers. The artifact SHALL make no
solve claim and SHALL use
`inference_substrate=arc_live_path_precheck_no_solve`.

The result artifact SHALL include `selected_game`,
`selected_target_level`, `registry_reproducible_total_levels_before`,
`duplicate_target_rejected`, `recent_no_bank_targets_avoided`,
`live_path_reachable`, `hidden_source_reading`, `offline_bfs_used`,
`hand_adapter_used`, `salience_feature_summary`,
`arc_target_rotation_ready`, `solve_claimed`, `inference_substrate`,
`random_seed`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "rotated non-duplicate Exp5464 shortlist game selected before any level-up attempt."
- `selected_target_level`: principle "selected next level, strictly greater than the registry reproduced depth."
- `registry_reproducible_total_levels_before`: principle "authoritative registry total before this no-solve precheck."
- `duplicate_target_rejected`: principle "bare bool proving already reproduced target levels are rejected before selection."
- `recent_no_bank_targets_avoided`: principle "auditable list containing bp35:L3, ka59:L2, and cn04:L4 unless a new mechanism justifies inclusion."
- `live_path_reachable`: principle "true only when submitted live salience/perception code emits the dry-check features."
- `hidden_source_reading`: principle "must be false; hidden/public source is not read in this precheck."
- `offline_bfs_used`: principle "must be false; no offline ground-truth BFS is used."
- `hand_adapter_used`: principle "must be false; target eligibility is not credited to a hand per-game adapter."
- `salience_feature_summary`: principle "dict summarizing connected components, color blobs, changed cells, target-region candidates, and known blockers."
- `arc_target_rotation_ready`: principle "true only when non-duplicate target rotation and live-path salience eligibility both pass."
- `solve_claimed`: principle "must be false; this artifact is a precheck, not a level solve."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_solve."
- `random_seed`: principle "deterministic seed for reproducible target ordering and dry-check fixtures."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked: with no solve claim."

### REQ-ARC-FCP-5480: Rotated Live Salience Level-Up Attempt

Experiment 5480 SHALL write
`results/experiment_5480_arc_live_salience_levelup_v497.json` only after
loading `selected_game` and `selected_target_level` from
`results/experiment_5479_arc_target_rotation_precheck_v497.json`. The workflow
SHALL abort with a blocked artifact when the Exp5479 target is missing or when
`ops/arc_solve_registry.yaml` already records the selected target level as
reproduced. Otherwise it SHALL run exactly one bounded live-agent
self-discovery attempt on that rotated target, using the submitted live
connected-component/color-blob/change-cell salience path as the prioritization
substrate.

The credited path SHALL be the agent's own observed frames, action effects, and
runtime reverse engineering only. It SHALL NOT read hidden game source, run an
offline ground-truth BFS, credit a hand per-game adapter, or rely on an
outer-loop reverse-engineered adapter. Any candidate new level SHALL be
reproduced through the registry-approved live reproduction mechanism before the
registry is updated. Success SHALL require `offline_reproduced=true` and
`reproduced_levels >= 1` beyond the selected target's precheck depth; otherwise
the artifact SHALL report an honest null and SHALL NOT modify
`ops/arc_solve_registry.yaml`.

The result artifact SHALL include `game`, `target_level`,
`solve_provenance`, `hidden_source_reading`, `offline_bfs_used`,
`hand_adapter_used`, `outer_loop_re_used`, `action_count`,
`explored_state_count`, `failed_hypotheses`, `offline_reproduced`,
`reproduced_levels`, `new_level_banked`, `reproduced_levels_before`,
`reproduced_levels_after`, `registry_updated`, `first_win_trace_path`,
`inference_substrate`, `random_seed`, and `honest_verdict`.

Required field principles:

- `game`: principle "selected Exp5479 game, or none when the target precondition blocks the attempt."
- `target_level`: principle "selected Exp5479 target level; success must reproduce at least this level."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `hidden_source_reading`: principle "must be false; hidden/public game source is not part of the credited path."
- `offline_bfs_used`: principle "must be false; exhaustive offline ground-truth BFS is not the credited path."
- `hand_adapter_used`: principle "must be false; a hand per-game adapter is not credited."
- `outer_loop_re_used`: principle "must be false; outer-loop reverse engineering is not credited."
- `action_count`: principle "bare integer count of bounded live-agent actions actually executed."
- `explored_state_count`: principle "bare integer count of live-agent states observed or tracked during the attempt."
- `failed_hypotheses`: principle "list of rejected salience/runtime hypotheses when no target level is banked."
- `offline_reproduced`: principle "true only when the live-agent candidate reproduces beyond the precheck depth."
- `reproduced_levels`: principle "new reproduced levels banked beyond the precheck depth; success requires >=1."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels>=1."
- `reproduced_levels_before`: principle "registry reproduced depth for the selected game before Exp5480."
- `reproduced_levels_after`: principle "registry reproduced depth after Exp5480; unchanged on honest null."
- `registry_updated`: principle "true only when ops/arc_solve_registry.yaml was updated for a newly reproduced level."
- `first_win_trace_path`: principle "relative path to the first reproduced winning trace, or empty string when none exists."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `random_seed`: principle "deterministic seed for the bounded live attempt."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5493: Registry-Only Trajectory Target Precheck

Experiment 5493 SHALL write
`results/experiment_5493_arc_trajectory_target_precheck_v498.json` as a
no-solve ARC registry and live-path precheck for the follow-on Exp5494
trajectory/option-induction attempt. The workflow SHALL read
`ops/arc_solve_registry.yaml`, `ops/exclusion_manifest.yaml`,
`ops/known-issues.md`, and the recent Exp5479/Exp5480 target artifacts before
selecting any target. It SHALL report each candidate game's current
`levels_reproduced`, reject any already reproduced target level, and avoid
recent no-bank targets `sb26:L3`, `bp35:L3`, `ka59:L2`, `cn04:L4`, and
`re86:L3` unless the registry proves a different target level and mechanism.

The workflow SHALL reject retired generic exploration-signal scopes from
Exp5154, including novelty-only, curiosity-only, energy-as-fitness
quality-diversity, and archive-granularity reruns. It SHALL prefer a target
whose registry row records live-path mechanism hooks and enough visible
runtime-observation structure for the live agent to form trajectory or option
hypotheses from its own observations. It SHALL NOT read game source, run
offline ground-truth BFS, or build or credit a per-game hand adapter. When no
eligible target exists, it SHALL emit a clean blocked no-target artifact rather
than selecting a stale duplicate.

The result artifact SHALL include `registry_path`,
`excluded_recent_no_bank_targets`, `duplicate_solve_avoided`,
`selected_game`, `selected_target_level`, `prior_levels_reproduced`,
`proposed_live_mechanism`, `trajectory_induction_preconditions`,
`offline_bfs_used`, `per_game_adapter_used`,
`arc_trajectory_precheck_ready`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `registry_path`: principle "must equal ops/arc_solve_registry.yaml."
- `excluded_recent_no_bank_targets`: principle "auditable list containing sb26:L3, bp35:L3, ka59:L2, cn04:L4, and re86:L3 unless the selected target proves a different level and mechanism."
- `duplicate_solve_avoided`: principle "bare bool proving the selected target is strictly deeper than the registry depth."
- `selected_game`: principle "selected game id, or empty string when no eligible target exists."
- `selected_target_level`: principle "selected next target level as a bare int, or 0 when blocked."
- `prior_levels_reproduced`: principle "authoritative registry depth for the selected game before Exp5493."
- `proposed_live_mechanism`: principle "one-line live-path mechanism to hand to Exp5494, not a retired exploration-signal rerun."
- `trajectory_induction_preconditions`: principle "list of live-observation prerequisites that must hold before Exp5494 attempts the target."
- `offline_bfs_used`: principle "must be false; this precheck is registry-only and no offline ground-truth BFS is run."
- `per_game_adapter_used`: principle "must be false; target eligibility is not credited to a hand adapter."
- `arc_trajectory_precheck_ready`: principle "true only when a non-duplicate, non-recent-no-bank, non-retired-scope target is selected."
- `inference_substrate`: principle "must equal registry_precheck_no_solve."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without a solve claim."

### REQ-ARC-FCP-5494: Exp5493-Selected Live Trajectory-Induction Level-Up Attempt

Experiment 5494 SHALL write
`results/experiment_5494_arc_live_trajectory_levelup_v498.json` after exactly
one bounded live ARC trajectory/option-induction attempt on the target selected
by Exp5493. The workflow SHALL read the Exp5493 precheck artifact and
`ops/arc_solve_registry.yaml` before spending live-agent budget; it SHALL block
without attempting when the target is already reproduced, appears in the recent
same-mechanism no-bank set, or lacks the live trajectory-induction
preconditions recorded by Exp5493. The credited path SHALL be
`live_agent_self_discovery`: the live agent's own attempted actions, observed
frame deltas, runtime hypotheses, verifier checks, and rejected option
sequences. It SHALL NOT read game source, run offline ground-truth BFS, or build
or credit a per-game hand adapter.

The workflow SHALL run through the live `E3AgentPolicy` path with
`LiveCoExLandmarkFrontierGenerator` as the trajectory/option-induction
generator when the preconditions pass. It SHALL record hypothesized action
sequences, observation deltas, verifier checks, and rejection reasons even when
no level is banked. If an LLM generator is invoked, the artifact's
`model_specs_if_llm_used` SHALL include
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`; if those model specs are unavailable for the
live path, the workflow SHALL emit a blocked artifact instead of substituting a
legacy small model as a headline generator. The registry SHALL be updated only
when the standard live/offline reproduction gate reports
`offline_reproduced=true` and at least one new level beyond the Exp5493 prior
depth is reproduced.

The result artifact SHALL include `selected_game`, `target_level`,
`prior_levels_reproduced`, `post_levels_reproduced`, `solve_provenance`,
`offline_bfs_used`, `per_game_adapter_used`, `game_source_read`,
`trajectory_hypothesis_count`, `live_attempt_count`, `offline_reproduced`,
`reproduced_levels`, `new_level_banked`, `registry_updated`,
`model_specs_if_llm_used`, `failure_mode`, `inference_substrate`,
`random_seed`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5493-selected game id, or empty string only when the attempt blocks before target selection."
- `target_level`: principle "Exp5493-selected target level as a bare int."
- `prior_levels_reproduced`: principle "registry depth before Exp5494; success must be strictly deeper."
- `post_levels_reproduced`: principle "registry depth after Exp5494; unchanged on honest null."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `offline_bfs_used`: principle "must be false; no offline ground-truth BFS is credited."
- `per_game_adapter_used`: principle "must be false; no hand per-game adapter is credited."
- `game_source_read`: principle "must be false; source reading is outside the credited live path."
- `trajectory_hypothesis_count`: principle "bare int count of hypothesized action sequences induced from runtime observations."
- `live_attempt_count`: principle "bare int count of live actions actually executed."
- `offline_reproduced`: principle "true only after the live-discovered candidate passes the reproduction gate."
- `reproduced_levels`: principle "new reproduced levels beyond the prior depth; complete requires >=1."
- `new_level_banked`: principle "true only when offline_reproduced=true and reproduced_levels>=1."
- `registry_updated`: principle "true only when a newly reproduced level is written to ops/arc_solve_registry.yaml."
- `model_specs_if_llm_used`: principle "empty when no LLM was invoked; otherwise contains the three mandated headline GGUF model specs."
- `failure_mode`: principle "empty on success or concise blocked/no-bank reason."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `random_seed`: principle "deterministic seed for the bounded attempt."
- `honest_verdict`: principle "terminal status starts with complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5507: Null-Coordinate And Perception-Grounding Target Precheck

Experiment 5507 SHALL write
`results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json`
as a no-solve aggregation artifact before Exp5508 spends any live-agent
budget. The workflow SHALL read `ops/arc_solve_registry.yaml`,
`ops/known-issues.md`, `docs/research-notes/arc-agi3-levers-tried-x-verdict-2026-06-25.md`,
`results/experiment_5493_arc_trajectory_target_precheck_v498.json`, and
`results/experiment_5494_arc_live_trajectory_levelup_v498.json`. It MAY also
read prior ARC salience/precheck artifacts as upstream evidence, but it SHALL
NOT read game source, run offline ground-truth BFS, build a hand per-game
adapter, or claim a solve.

The workflow SHALL reject any requested target level that the registry already
records as reproduced. It SHALL audit recent no-bank targets and reject exact
same-target/same-mechanism reruns unless the selected mechanism is materially
changed. It SHALL audit prior null-coordinate or no-op behavior as either valid
game actions with zero frame effect or metric artifacts, and SHALL record
whether any banked level appears contaminated by a null-coordinate exploit. It
SHALL inspect upstream live observations for perception-grounding findings that
a classical connected-component/color-blob segmentation pass can expose.

The result artifact SHALL include `registry_path`,
`reproducible_total_levels_before`, `duplicate_targets_rejected`,
`recent_no_bank_targets_rejected`, `null_coordinate_audit`,
`perception_grounding_findings`, `selected_game`, `selected_level`,
`selected_mechanism`, `levelup_attempt_ready`, `solve_claimed`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `registry_path`: principle "must equal ops/arc_solve_registry.yaml."
- `reproducible_total_levels_before`: principle "authoritative registry total before this no-solve target precheck."
- `duplicate_targets_rejected`: principle "auditable list of requested target levels rejected because the registry already reproduces them."
- `recent_no_bank_targets_rejected`: principle "auditable list of recent same-target/same-mechanism no-bank reruns rejected before selection."
- `null_coordinate_audit`: principle "dict classifying null/missing/no-op coordinate evidence as valid game action, metric artifact, or contamination."
- `perception_grounding_findings`: principle "list of upstream live-observation findings exposed by connected components, color blobs, salience tiers, changed pixels, or action-effect asymmetry."
- `selected_game`: principle "selected game id, or empty string when blocked."
- `selected_level`: principle "selected level label such as L3, or empty string when blocked."
- `selected_mechanism`: principle "changed live-path perception-generation mechanism, not a same-mechanism rerun."
- `levelup_attempt_ready`: principle "bare bool true only when the selected level is non-duplicate, same-mechanism reruns are rejected, null-coordinate audit is clean, and perception findings are present."
- `solve_claimed`: principle "must be false; this artifact is a precheck and cannot bank a level."
- `inference_substrate`: principle "must equal aggregation_from_upstream_artifacts."
- `honest_verdict`: principle "terminal status starts with complete: or blocked: and makes no solve claim."

### REQ-ARC-FCP-5508: Live Perception-Generation Level-Up Attempt

Experiment 5508 SHALL write
`results/experiment_5508_arc_live_perception_generation_levelup_v499.json`
after re-reading `ops/arc_solve_registry.yaml` and the Exp5507 precheck
artifact immediately before any live-agent attempt. The workflow SHALL abort
with a blocked artifact when the Exp5507 target is absent, not ready, or already
reproducible in the re-read registry. Otherwise it SHALL run one bounded live
ARC attempt on the selected target using `E3AgentPolicy` with a reusable
classical perception-generation pass that extracts connected components, color
blobs, sprite overlays, salient motion, and action affordances from runtime
frames only.

The credited path SHALL be `live_agent_self_discovery`: the live agent's own
runtime observations, action effects, perception-generated candidates, and
standard reproduction gate. It SHALL NOT read game source, run offline
ground-truth BFS, or use a hand-built per-game adapter. Any candidate level-up
SHALL be reproduced through the standard live/offline reproduction gate before
the registry is updated. Success SHALL require `offline_reproduced=true` and
`reproduced_levels>=1` for a strictly new level beyond the selected game's
registry depth; otherwise the artifact SHALL report an honest null and SHALL
NOT modify `ops/arc_solve_registry.yaml`.

The workflow SHALL record per-step trajectory-taxonomy counts inspired by
Trajel for factual, referential, logical, procedural, and scope-based failures
when applicable. Honest-null artifacts SHALL still record enough live-attempt
duration, candidate-generation, prohibited-input, and methodology fields to
distinguish a bounded runtime null from a missing-methodology artifact.

The result artifact SHALL include `selected_game`, `selected_level`,
`registry_before_levels`, `registry_after_levels`, `arc_registry_delta`,
`offline_reproduced`, `reproduced_levels`, `solve_provenance`,
`live_agent_attempts`, `runtime_observation_steps`,
`perception_features_enabled`, `trajectory_taxonomy_counts`,
`offline_bfs_used`, `game_source_read`,
`hand_built_per_game_adapter_used`, `methodology_receipt`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5507-selected game id, or empty string only when the precheck blocks before a target can be attempted."
- `selected_level`: principle "Exp5507-selected level label such as L3; success must be strictly deeper than the re-read registry depth."
- `registry_before_levels`: principle "authoritative `ops/arc_solve_registry.yaml` total immediately before the live attempt."
- `registry_after_levels`: principle "authoritative registry total after the attempt; unchanged on honest null or blocked runs."
- `arc_registry_delta`: principle "bare int delta between after and before totals; success requires this to equal the newly reproduced levels."
- `offline_reproduced`: principle "true only when the live-discovered candidate passes the standard reproduction gate for a new level."
- `reproduced_levels`: principle "new reproduced levels banked beyond the selected game's pre-run depth; success requires >=1."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `live_agent_attempts`: principle "bare int count of runtime actions actually executed by the live agent."
- `runtime_observation_steps`: principle "bare int count of runtime frame/action observations available to perception generation."
- `perception_features_enabled`: principle "list containing connected_components, color_blobs, sprite_overlays, salient_motion, and action_affordances when the pass is active."
- `trajectory_taxonomy_counts`: principle "dict with factual, referential, logical, procedural, and scope_based failure counts."
- `offline_bfs_used`: principle "must be false; offline ground-truth BFS is not part of the credited path."
- `game_source_read`: principle "must be false; game source reading is outside live self-discovery credit."
- `hand_built_per_game_adapter_used`: principle "must be false; no hand per-game adapter is credited."
- `methodology_receipt`: principle "string receipt naming the bounded live runtime, candidate-generation mechanism, reproduction gate, and prohibited-input flags."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5520: Action-Diversity Target Precheck

Experiment 5520 SHALL write
`results/experiment_5520_arc_action_diversity_target_precheck.json` before the
next live ARC level-up attempt spends credit-bearing budget. The workflow SHALL
read `ops/arc_solve_registry.yaml` and the Exp5508 live perception-generation
artifact, reject any candidate whose target level is already reproduced by the
registry, and reject the Exp5508 repeated coordinate/action pattern as a target
or mechanism reuse.

The workflow SHALL configure a live-path candidate-generator change based on
connected-component/color-blob salience plus repeated-coordinate suppression,
target rotation, action entropy, or salience coverage. It SHALL measure that
changed path through a dry-run or limited no-credit probe only; the probe SHALL
NOT run offline ground-truth BFS, use a hand per-game adapter, or claim solve
credit. A target SHALL be selected only when the probe shows a meaningfully
different action-diversity path from Exp5508.

The result artifact SHALL include `registry_precheck_done`, `selected_game`,
`selected_level`, `already_reproduced`, `exp5508_pattern_reused`,
`candidate_generator_changes`, `action_entropy`,
`repeated_coordinate_rate`, `salience_coverage_rate`,
`no_credit_probe_attempts`, `arc_levelup_candidate_ready`,
`solve_provenance`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `registry_precheck_done`: principle "bare bool proving ops/arc_solve_registry.yaml was checked before target selection."
- `selected_game`: principle "one registry-safe game id selected for the next live level-up attempt, or empty string when blocked."
- `selected_level`: principle "next unreproduced target level as a string or bare int; it must be strictly deeper than the registry depth."
- `already_reproduced`: principle "must be false for any ready artifact."
- `exp5508_pattern_reused`: principle "must be false; Exp5508's repeated ACTION6 coordinate loop cannot be reused."
- `candidate_generator_changes`: principle "non-empty list naming live-path generation changes such as repeated-coordinate suppression, target rotation, action entropy gating, or salience coverage."
- `action_entropy`: principle "Shannon entropy over dry-run action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of dry-run consecutive coordinate choices that repeat a prior coordinate, as a bare float."
- `salience_coverage_rate`: principle "fraction of dry-run choices covering distinct salience candidates, as a bare float."
- `no_credit_probe_attempts`: principle "bare int count of no-credit dry-run choices measured before the live attempt."
- `arc_levelup_candidate_ready`: principle "bare bool true only when registry and Exp5508-pattern gates pass and the diversity metrics meet threshold."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `inference_substrate`: principle "must equal arc_live_precheck."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5521: Live Action-Diverse Level-Up Attempt

Experiment 5521 SHALL write
`results/experiment_5521_arc_live_action_diverse_levelup.json` after the
Exp5520 action-diversity precheck is ready. The attempt SHALL re-read
`ops/arc_solve_registry.yaml`, verify that Exp5520 selected a registry-safe
target with `arc_levelup_candidate_ready=true`, and run one bounded live-agent
self-discovery attempt against that target using the action-diverse
connected-component/color-blob generator. The live attempt MAY replay a
live-discovered trajectory through the standard offline reproduction gate, but
it SHALL NOT use offline ground-truth BFS, read game source, or credit a
hand-built per-game adapter.

The result artifact SHALL include `selected_game`, `selected_level`,
`offline_reproduced`, `reproduced_levels`, `banking_gate`, `registry_delta`,
`solve_provenance`, `live_attempts`, `action_entropy`,
`repeated_coordinate_rate`, `salience_coverage_rate`, `trajectory_log_path`,
`reproduction_command`, `arc_live_levelup_ready`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5520-selected registry-safe game id; empty only when the readiness gate blocks before live runtime."
- `selected_level`: principle "Exp5520-selected unreproduced level label or int; success must be strictly deeper than the registry depth."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires >=1."
- `banking_gate`: principle "bare bool equal to offline_reproduced=true and reproduced_levels>=1 for solve_provenance=live_agent_self_discovery."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the banking gate is true."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `live_attempts`: principle "bare int count of runtime actions executed by the live agent."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed live coordinate choices that repeated a prior coordinate, as a bare float."
- `salience_coverage_rate`: principle "fraction of executed live coordinate choices covering proposed salience coordinates, as a bare float."
- `trajectory_log_path`: principle "path to the detailed trajectory log containing observations, proposed actions, verifier feedback, and diversity metrics."
- `reproduction_command`: principle "exact replay command when a live trajectory was reproduced, else null."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5520 and the registry reread allowed the live attempt."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5533: Strategy-Routing Precheck Before Live Level-Up Attempt

Experiment 5533 SHALL write
`results/experiment_5533_arc_strategy_routing_precheck.json` before the next
credit-bearing ARC live level-up attempt. The workflow SHALL read
`ops/arc_solve_registry.yaml`, Exp5520, and Exp5521; select one adjacent
frontier target whose target level is not already reproduced; and rotate away
from the stale Exp5521 target when Exp5521 records a no-bank live attempt with
repeated-coordinate collapse.

The workflow SHALL configure a bounded, deterministic strategy portfolio with
at least three live-path-compatible strategies, including salience-first,
action-effect memory, verifier-router candidate ranking, or conservative
reset/reinduction. The precheck SHALL verify that strategy routing is reachable
through the live candidate-router hook and that repeated-coordinate suppression
changes the candidate action selected before metrics are computed. It SHALL NOT
inspect hidden game source, run exhaustive offline ground-truth BFS, build a
hand per-game adapter, or claim a solve.

The result artifact SHALL include `selected_game`, `selected_level`,
`already_reproduced`, `registry_precheck_passed`, `strategy_portfolio`,
`strategy_routing_live_path_reachable`,
`repeated_coordinate_suppression_enabled`,
`repeated_coordinate_rate_precheck`, `action_entropy_precheck`,
`salience_coverage_rate_precheck`, `model_specs`,
`llm_strategy_proposer_used`, `solve_provenance`,
`arc_sge_candidate_ready`, `tests_added_or_reused`, `field_principles`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id selected for the next strategy-guided live attempt, or empty string when blocked."
- `selected_level`: principle "adjacent unreproduced frontier level label; it must be deeper than the registry depth."
- `already_reproduced`: principle "must be false for any ready artifact because duplicate live levels cannot satisfy the standing floor."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `strategy_portfolio`: principle "list of at least three bounded live-path-compatible strategy descriptors used before the attempt."
- `strategy_routing_live_path_reachable`: principle "bare bool proving the router object reaches the live candidate-router hook used by E3AgentPolicy and graph exploration."
- `repeated_coordinate_suppression_enabled`: principle "bare bool true only when repeated-coordinate suppression changes candidate selection before metrics."
- `repeated_coordinate_rate_precheck`: principle "fraction of routed precheck coordinate choices repeating earlier coordinates after suppression."
- `action_entropy_precheck`: principle "Shannon entropy over routed precheck action/coordinate choices as a bare float."
- `salience_coverage_rate_precheck`: principle "fraction of salience candidate coordinates covered by routed precheck choices."
- `model_specs`: principle "allowed local-GGUF proposer specs recorded for audit; no model is invoked when llm_strategy_proposer_used=false."
- `llm_strategy_proposer_used`: principle "bare bool; false means deterministic strategy templates were used and no GGUF tokenizer/model path was loaded."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `arc_sge_candidate_ready`: principle "bare bool true only when target, strategy routing, suppression, and metric gates pass."
- `tests_added_or_reused`: principle "list of focused tests that cover the Exp5533 schema, target rotation, live routing hook, and suppression evidence."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_solve_claim."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5534: Strategy-Routed Live Level-Up Attempt

Experiment 5534 SHALL write
`results/experiment_5534_arc_strategy_routed_levelup.json` from the target
selected by Exp5533. The workflow SHALL load Exp5533's `selected_game` and
`selected_level`, reread `ops/arc_solve_registry.yaml`, and block as
`blocked_duplicate_target` without choosing a replacement if the registry now
already reproduces the selected level.

When not duplicate-blocked, the workflow SHALL run the live ARC agent with the
Exp5533 bounded strategy portfolio and repeated-coordinate suppression enabled.
It SHALL record action attempts, strategy choices, verifier route evidence,
suppression events, and level counter changes in a trajectory artifact. It
SHALL keep `solve_provenance=live_agent_self_discovery` and SHALL NOT inspect
hidden game source, run exhaustive offline ground-truth BFS, perform outer-loop
reverse engineering, or build a hand per-game adapter.

A new-level claim SHALL require `offline_reproduced=true` and
`reproduced_levels>=1` through the standard live-path reproduction gate. The
ARC solve registry SHALL be updated only when that gate passes. Otherwise the
artifact SHALL use an `honest_null:` verdict and preserve the trajectory
evidence.

The result artifact SHALL include `selected_game`, `selected_level`,
`solve_provenance`, `strategy_portfolio_used`, `strategy_switch_count`,
`attempts`, `action_entropy`, `repeated_coordinate_rate`,
`repeated_coordinate_suppression_events`, `salience_coverage_rate`,
`offline_reproduced`, `reproduced_levels`, `registry_delta`,
`trajectory_path`, `model_specs`, `llm_strategy_proposer_used`,
`arc_live_levelup_ready`, `tests_added_or_reused`, `field_principles`,
`inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "Exp5533-selected registry-safe game id; empty only when Exp5533 blocks before runtime."
- `selected_level`: principle "Exp5533-selected adjacent frontier level label; duplicate targets block rather than rotate."
- `solve_provenance`: principle "must equal live_agent_self_discovery."
- `strategy_portfolio_used`: principle "bounded live-path-compatible strategy descriptors actually installed on the candidate router."
- `strategy_switch_count`: principle "integer count of changes between executed strategy labels after repeated-coordinate suppression."
- `attempts`: principle "bare int count of runtime actions executed by the live agent."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed live coordinate choices that repeated an earlier executed coordinate."
- `repeated_coordinate_suppression_events`: principle "bare int count of candidate-router repeated-coordinate suppressions recorded during live selection."
- `salience_coverage_rate`: principle "fraction of executed live coordinate choices covering proposed salience coordinates."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires >=1."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the live reproduction gate passes."
- `trajectory_path`: principle "path to the detailed trajectory log containing attempts, strategies, verifier routes, suppression events, and level changes."
- `model_specs`: principle "allowed local-GGUF proposer specs recorded for audit; no model is invoked when llm_strategy_proposer_used=false."
- `llm_strategy_proposer_used`: principle "bare bool; false means deterministic strategy templates were used and no GGUF tokenizer/model path was loaded."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5533 and registry reread allowed live runtime."
- `tests_added_or_reused`: principle "list of focused tests that cover the Exp5534 schema, duplicate block, live routing trace, and registry gate."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5547: No-LLM ARC Substrate Clean Precheck

Experiment 5547 SHALL write
`results/experiment_5547_arc_no_llm_substrate_precheck.json` as a clean
no-credit ARC live-path precheck before any level-up attempt that follows the
flagged Exp5533/Exp5534 artifacts. The workflow SHALL read
`ops/arc_solve_registry.yaml`, run a duplicate registry precheck, and choose one
adjacent target level that is not already banked by the live mechanism. It SHALL
use only the live agent path, SHALL NOT invoke an LLM strategy proposer, and
SHALL NOT include `model_specs` or `target_model` because the declared
substrate is the offline arcade live-agent runtime with self-discovery and no
LLM.

The result artifact SHALL include `selected_game`, `selected_level`,
`registry_precheck_passed`, `already_reproduced`,
`llm_strategy_proposer_used`, `no_model_specs_required`, `random_seed`,
`reproducibility_checksum`, `strategy_routing_live_path_reachable`,
`repeated_coordinate_suppression_enabled`, `action_entropy_precheck`,
`solve_provenance`, `arc_clean_precheck_ready`, `tests_added_or_reused`,
`field_principles`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id selected for the next clean no-LLM live-path attempt."
- `selected_level`: principle "adjacent unreproduced frontier level label selected after the duplicate registry precheck."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `already_reproduced`: principle "must remain false because duplicate live levels cannot satisfy the ARC standing progress floor."
- `llm_strategy_proposer_used`: principle "bare bool false proving this precheck did not load or invoke an LLM strategy proposer."
- `no_model_specs_required`: principle "bare bool true because the no-LLM substrate has no model invocation to name."
- `random_seed`: principle "deterministic seed required for third-party reruns of the target choice and checksum."
- `reproducibility_checksum`: principle "content-addressed hash over registry target, seed, substrate, and routing gates to catch silent drift."
- `strategy_routing_live_path_reachable`: principle "bare bool proving the bounded candidate router is reachable from the live candidate-router hook."
- `repeated_coordinate_suppression_enabled`: principle "bare bool proving repeated-coordinate suppression is active before action entropy is trusted."
- `action_entropy_precheck`: principle "bare float expectation for routed action/coordinate diversity before the live attempt."
- `solve_provenance`: principle "must equal live_agent_self_discovery even though this artifact claims no solve."
- `arc_clean_precheck_ready`: principle "bare bool true only when registry, no-LLM substrate, provenance, seed/checksum, live path, and suppression gates pass."
- `tests_added_or_reused`: principle "list of focused tests covering duplicate blocking, no-model metadata, checksum determinism, and schema gates."
- `field_principles`: principle "mapping of one-line principle annotations for every headline and gate field."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5548: Clean No-LLM Live Level-Up Attempt

Experiment 5548 SHALL write
`results/experiment_5548_arc_clean_live_levelup.json` after one gated live ARC
attempt using the clean no-LLM substrate precheck from Exp5547. The workflow
SHALL first confirm that Exp5547 reports `arc_clean_precheck_ready=true`, then
SHALL use Exp5547's `selected_game`, `selected_level`, `random_seed`, and
repeated-coordinate suppression setting unless a fresh registry reread shows
that target level is already banked. If the selected target is already banked,
the workflow SHALL rotate to a fresh adjacent frontier target and record the
rotation reason.

The live attempt SHALL use the live agent self-discovery path with a disabled
LLM strategy proposer, no model load, no `model_specs`, no hidden-game source
inspection, no exhaustive offline ground-truth BFS, and no hand-built per-game
adapter. It SHALL record the trajectory path, action entropy,
repeated-coordinate rate, attempt budget, terminal state, and exact replay or
banking result. A level-up claim SHALL be accepted only when
`offline_reproduced=true` and `reproduced_levels>=1` with
`solve_provenance=live_agent_self_discovery`; otherwise the artifact SHALL emit
an honest null and leave `registry_delta=0`.

The result artifact SHALL include `selected_game`, `selected_level`,
`solve_provenance`, `llm_strategy_proposer_used`, `no_model_specs_required`,
`random_seed`, `reproducibility_checksum`, `attempts`, `trajectory_path`,
`action_entropy`, `repeated_coordinate_rate`, `offline_reproduced`,
`reproduced_levels`, `registry_delta`, `arc_live_levelup_ready`,
`tests_added_or_reused`, `field_principles`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `selected_game`: principle "registry-safe game id used for the clean live attempt after Exp5547 and duplicate checks."
- `selected_level`: principle "adjacent unreproduced frontier level label attempted by the live agent."
- `solve_provenance`: principle "must equal live_agent_self_discovery so only the live runtime's own attempt can receive credit."
- `llm_strategy_proposer_used`: principle "bare bool false proving no LLM strategy proposer or model path was invoked."
- `no_model_specs_required`: principle "bare bool true because the declared no-LLM substrate has no model invocation to name."
- `random_seed`: principle "Exp5547-recorded deterministic seed reused for target rotation, trajectory gating, and checksum replay."
- `reproducibility_checksum`: principle "content-addressed hash over target, seed, trajectory metrics, and banking gate to catch silent drift."
- `attempts`: principle "bare int count of runtime actions executed during the live attempt."
- `trajectory_path`: principle "path to the detailed trajectory log containing actions, route evidence, suppression events, terminal state, and replay gate."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed coordinate actions that repeated an earlier executed coordinate."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the standard offline replay gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires at least one."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the accepted reproduction gate passes."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5547, registry reread, no-LLM metadata, and live harness preconditions allowed runtime."
- `tests_added_or_reused`: principle "list of focused tests covering clean schema, target rotation, trajectory metrics, checksum, and banking gate."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, or blocked:."

### REQ-ARC-FCP-5561: FSM Target Rotation Precheck

Experiment 5561 SHALL write
`results/experiment_5561_arc_fsm_target_rotation_precheck.json` before the
next credit-bearing ARC live attempt after Exp5548's honest no-bank null. The
workflow SHALL read `ops/arc_solve_registry.yaml` before selecting a target,
SHALL reject target levels already reproduced by the live mechanism, and SHALL
avoid the recent Exp5548 no-bank target unless a registry or ops-document reason
explicitly justifies retrying it.

The workflow SHALL build or reuse a simple finite-state action abstraction that
is reachable through the live candidate-router hook, SHALL enable
repeated-coordinate suppression before computing action entropy, and SHALL NOT
inspect hidden game source, run exhaustive offline ground-truth BFS, or build a
hand per-game calibration solver. The precheck is no-credit: it SHALL carry
`solve_provenance=live_agent_self_discovery`, SHALL invoke no LLM, and SHALL
claim readiness only when registry, target-rotation, FSM abstraction,
suppression, and entropy gates all pass.

The result artifact SHALL include `llm_invoked`,
`no_model_specs_required`, `selected_game`, `selected_level`,
`registry_precheck_passed`, `already_reproduced`,
`recent_no_bank_targets_avoided`, `fsm_action_abstraction_ready`,
`repeated_coordinate_suppression_enabled`, `action_entropy_precheck`,
`solve_provenance`, `arc_fsm_precheck_ready`, `tests_added_or_reused`,
`field_principles`, `inference_substrate`, and `honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving the FSM target rotation precheck did not invoke any LLM."
- `no_model_specs_required`: principle "bare bool true because the declared no-LLM precheck substrate has no model invocation to name."
- `selected_game`: principle "registry-safe game id selected for the next FSM-guided live attempt, or empty string when blocked."
- `selected_level`: principle "adjacent unreproduced frontier level label selected after registry and recent no-bank target checks."
- `registry_precheck_passed`: principle "bare bool proving the registry was read and the selected level is not already reproduced."
- `already_reproduced`: principle "must remain false because duplicate live levels cannot satisfy the ARC standing progress floor."
- `recent_no_bank_targets_avoided`: principle "list of recent no-bank target markers rejected before selection unless an explicit retry reason exists."
- `fsm_action_abstraction_ready`: principle "bare bool proving the FSM action abstraction is live-router reachable and emits bounded action phases."
- `repeated_coordinate_suppression_enabled`: principle "bare bool proving repeated-coordinate suppression is active before action entropy is trusted."
- `action_entropy_precheck`: principle "bare float Shannon entropy over suppressed FSM action/coordinate choices before the live attempt."
- `solve_provenance`: principle "must equal live_agent_self_discovery even though this artifact claims no solve."
- `arc_fsm_precheck_ready`: principle "bare bool true only when registry, rotation, FSM abstraction, suppression, entropy, and no-LLM gates pass."
- `tests_added_or_reused`: principle "list of focused tests covering the Exp5561 schema, target rotation, FSM reachability, suppression, and artifact write."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal arc_live_path_precheck_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-FCP-5562: Gated FSM Live Level-Up Attempt

Experiment 5562 SHALL write
`results/experiment_5562_arc_fsm_live_levelup.json` after the Exp5561-selected
FSM ARC target is rechecked against the registry. The workflow SHALL run a live
level-up attempt only when Exp5561 reports `arc_fsm_precheck_ready=true`, the
selected target is still adjacent and unreproduced, and the FSM action
abstraction remains live-router reachable. If the registry now records the
selected target level as already reproduced, the workflow SHALL emit a
duplicate-prevented artifact, SHALL NOT run the live attempt, and SHALL leave
`registry_delta=0`.

The credited attempt SHALL use only the live agent's own runtime attempts and
runtime-reachable FSM induction/routing. It SHALL NOT inspect hidden game
source, run an offline solver or exhaustive offline ground-truth BFS as solve
evidence, use an LLM strategy proposer, read hidden-source mechanics, or treat
LLM proposer text as the credited solver path. If the live agent reaches a new
candidate level, post-solve banking SHALL run only after that live path has
found the solution. A level-up claim SHALL be accepted only when
`solve_provenance=live_agent_self_discovery`, `offline_reproduced=true`, and
`reproduced_levels>=1`; otherwise the artifact SHALL emit an honest null and
leave `registry_delta=0`.

The result artifact SHALL include `llm_invoked`,
`no_model_specs_required`, `upstream_arc_precheck`, `solve_provenance`,
`llm_strategy_proposer_used`, `random_seed`, `reproducibility_checksum`,
`selected_game`, `selected_level`, `attempts`, `trajectory_path`,
`action_entropy`, `repeated_coordinate_rate`, `offline_reproduced`,
`reproduced_levels`, `registry_delta`, `arc_live_levelup_ready`,
`tests_added_or_reused`, `field_principles`, `inference_substrate`, and
`honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving the credited FSM live attempt did not invoke any LLM."
- `no_model_specs_required`: principle "bare bool true because this no-LLM live-agent substrate has no model invocation to name."
- `upstream_arc_precheck`: principle "path to the Exp5561 gate that selected the target and proved FSM live-path reachability before runtime."
- `solve_provenance`: principle "must equal live_agent_self_discovery so only the live runtime's own attempts and runtime reverse engineering receive solve credit."
- `llm_strategy_proposer_used`: principle "bare bool false proving no LLM strategy proposer text contributed to the credited solver path."
- `random_seed`: principle "deterministic seed used for target recheck, bounded live routing, trajectory logging, and checksum replay."
- `reproducibility_checksum`: principle "content-addressed hash over target, seed, trajectory metrics, duplicate gate, and banking result to catch silent drift."
- `selected_game`: principle "Exp5561-selected registry-safe game id rechecked immediately before the live attempt."
- `selected_level`: principle "Exp5561-selected adjacent unreproduced frontier level label, or the duplicate-prevented target label."
- `attempts`: principle "bare int count of live-agent runtime actions executed; zero means the duplicate or readiness gate prevented runtime."
- `trajectory_path`: principle "path to the detailed live trajectory or duplicate-prevented trajectory receipt."
- `action_entropy`: principle "Shannon entropy over executed live action/coordinate choices as a bare float."
- `repeated_coordinate_rate`: principle "fraction of executed coordinate actions that repeated an earlier executed coordinate."
- `offline_reproduced`: principle "true only when the live-discovered trajectory passes the post-solve offline reproduction gate."
- `reproduced_levels`: principle "integer new levels banked from the live-discovered trajectory; success requires at least one."
- `registry_delta`: principle "bare int registry total delta; nonzero only when the accepted reproduction gate passes after live discovery."
- `arc_live_levelup_ready`: principle "bare bool proving Exp5561, registry reread, live-reachability, no-LLM metadata, and duplicate gates allowed runtime."
- `tests_added_or_reused`: principle "list of focused tests covering the Exp5562 schema, duplicate prevention, live trajectory metrics, checksum, and banking gate."
- `field_principles`: principle "mapping of one-line principle annotations for each headline and gate field."
- `inference_substrate`: principle "must equal arc_live_agent_self_discovery_no_llm."
- `honest_verdict`: principle "one-line verdict starting complete:, honest_null:, duplicate_prevented:, or blocked:."

### REQ-ARC-FCP-5575: SGE Anti-Stagnation Live-Path Precheck

Experiment 5575 SHALL write
`results/experiment_5575_sge_anti_stagnation_live_precheck.json` before any
follow-up SGE live inference spend. The workflow SHALL read the recorded SGE
trace at `results/outer_loop_sge_smoke_test.json`, define fixed collapse
thresholds before evaluation, and detect collapse using repeated normalized
strategy text, low pairwise strategy distance, repeated action proposals, and
consecutive null outcomes. It SHALL implement the anti-stagnation diversity
controller inside the existing `LLMStrategyProposer`/`SGECandidateRouter`
candidate-router path consumed by `E3AgentPolicy`, not as a standalone router.

When collapse is detected, the live-path router SHALL force a bounded portfolio
spanning observation, active coordinate probe, action-type probe, mechanic
falsification, and recovery/reset hypotheses. The controller SHALL apply an
outcome-conditioned taboo set against recently failed normalized strategies and
SHALL remain stable when the LLM proposer fails or returns malformed output. It
SHALL NOT use win-check, level-completion, hidden-source, scorecard, or oracle
signals for strategy generation, ranking, or readiness.

The precheck workflow SHALL read `ops/arc_solve_registry.yaml`, reject targets
already reproduced at the requested level, reject recently exhausted targets
unless the anti-stagnation controller is the new mechanism that justifies a
retry, select one unsolved adjacent frontier target and action budget, and emit
`live_path_ready=true` only when tests pass, the submitted E3 import path reaches
the controller, the controller activates on the recorded SGE trace, no oracle
leakage exists, and `target_unsolved=true`.

The result artifact SHALL include `field_principles`, `llm_invoked`,
`prior_trace_path`, `collapse_definition`,
`collapse_detected_on_prior_trace`, `diversity_metrics_before_after`,
`forced_portfolio`, `taboo_policy`, `e3_import_path`,
`live_path_reachable`, `verifier_is_oracle`,
`win_check_used_for_ranking`, `registry_precheck`, `target_game`,
`target_level`, `prior_levels_reproduced`, `action_budget`,
`tests_run`, `positive_control_passed`, `solve_provenance`,
`inference_substrate`, `target_unsolved`, `live_path_ready`, and
`honest_verdict`.

Required field principles:

- `llm_invoked`: principle "bare bool false proving Exp5575 is a deterministic precheck and spends no live LLM inference."
- `prior_trace_path`: principle "path to the recorded SGE trace used only for collapse diagnosis, not solve credit."
- `collapse_definition`: principle "fixed thresholds declared before evaluating the prior trace."
- `collapse_detected_on_prior_trace`: principle "bare bool true only when the fixed repeated-strategy, pairwise-distance, repeated-action, and null-outcome signals trip the collapse gate."
- `diversity_metrics_before_after`: principle "records strategy/action diversity before and after forced anti-stagnation routing."
- `forced_portfolio`: principle "bounded live-path-compatible strategy categories spanning observation, coordinate probe, action-type probe, mechanic falsification, and recovery/reset."
- `taboo_policy`: principle "outcome-conditioned taboo set derived only from recently failed normalized strategies."
- `e3_import_path`: principle "exact submitted path proving E3AgentPolicy reaches SGECandidateRouter through candidate_router.rank."
- `live_path_reachable`: principle "bare bool proving the controller is reachable through the scored E3 candidate-router hook."
- `verifier_is_oracle`: principle "must be false because precheck readiness cannot use oracle or hidden-source verifiers."
- `win_check_used_for_ranking`: principle "must be false because strategy ranking cannot inspect win, level-completion, scorecard, or source signals."
- `registry_precheck`: principle "structured evidence that registry depth was read and duplicate or exhausted targets were rejected before target selection."
- `target_game`: principle "registry-safe game id selected for the next SGE live attempt."
- `target_level`: principle "adjacent unreproduced frontier level selected after duplicate and exhaustion checks."
- `prior_levels_reproduced`: principle "registry depth before the attempted target; target_level must be greater."
- `action_budget`: principle "bounded action budget chosen before live inference."
- `tests_run`: principle "exact test commands proving fake-completer behavior and full Python suite status."
- `positive_control_passed`: principle "bare bool true only when fake-completer collapse activation and E3 reachability tests pass."
- `solve_provenance`: principle "must equal live_agent_self_discovery because the follow-up live attempt must be the scored runtime's own discovery path."
- `inference_substrate`: principle "must equal deterministic_live_path_precheck_no_llm."
- `target_unsolved`: principle "a target already reproduced at the requested level cannot receive duplicate solve credit."
- `live_path_ready`: principle "live inference requires tested E3 reachability, a new mechanism, and no oracle leakage."
- `honest_verdict`: principle "one-line verdict starting complete: or blocked: without claiming a solve."

### REQ-ARC-WMTE-5716: Game-Agnostic Exploration-Playbook Primitives

The repository SHALL provide a mined exploration playbook at
`docs/research-notes/arc-exploration-playbook-20260717.md` that distills
CROSS-GAME-TRANSFERABLE exploration methodology from `ops/arc_solve_registry.yaml`,
and SHALL encode the most mechanizable of those patterns as reusable primitives in
`python/carnot/agentic/arc_solver_kit.py`. Every primitive MUST be game-agnostic
(no per-game color / coordinate / mechanic hardcoded) and MUST read only rendered
frames or caller-injected callables, never `env._game` internals, so it is usable
both offline and on the live scored path.

The module SHALL provide at least: `probe_action_semantics` (measure each candidate
action's effect this level from a fresh env instead of assuming carryover),
`read_absolute_trajectory` (recover a sprite's absolute motion across the multi-layer
animation array the camera-relative settled grid hides),
`find_unexplained_glyphs` (surface uncatalogued objects to interrogate before
committing a route), `bounded_reachability_search` (a pluggable-state-hash graph
search), and `bisect_death_prefix` (isolate the minimal death-causing prefix).

`bounded_reachability_search` SHALL distinguish a PROVEN-exhaustive negative
(frontier emptied with no cut branches) from a merely SEARCH-CAPPED negative
(node/depth budget hit), and SHALL set `proven_unreachable=true` only for the
former. Each primitive SHALL be covered by unit tests exercising every branch, with
no skipped tests, and SHALL pass ruff and mypy.

Required field principles:

- `proven_unreachable`: principle "bare bool true ONLY when the frontier emptied with no depth/node cut; conflating capped-with-proven is the corpus error this primitive prevents (wa30 settled-dead-end overturned)."
- `status`: principle "one of goal/exhausted/capped_nodes/capped_depth so a caller can tell WHY a search returned negative before acting on it."
- `changed_cells`: principle "None when the grid shape changed (degenerate/terminal frame), which the caller must treat as not-inert rather than as zero change."
- `direction`: principle "absolute sprite motion (up/down/left/right/none) recovered from the animation layer stack, independent of any camera recentering."
- `fatal_prefix_len`: principle "the minimal action count that still dies (or None), separating a real hazard from an action-budget/timer death."
- `verifier_is_oracle`: principle "must be false; these are exploration heuristics, not win oracles -- only the offline reproduction gate counts a level."

### REQ-ARC-WMTE-5717: Dev-Gated Playbook-Exemplar Stall Re-Induction Injection

`arc_executable_world_model.induce_prompt` SHALL accept an opt-in
`include_playbook_exemplars` boolean defaulting to False, and when False SHALL
return the exact pre-existing prompt byte-identically. When True it SHALL prepend a
SMALL, game-AGNOSTIC few-shot of exploration-method PATTERN statements (from the
REQ-ARC-WMTE-5716 playbook) that biases the tier-3 proposer's priors without asking
it to reason at length. `LocalGGUFProposer` and `CodexProposer` SHALL expose an
`include_playbook_exemplars` field defaulting to False that passes through to
`induce_prompt`.

`E3AgentPolicy` SHALL arm the injection ONLY on the STALL / first-contact
re-induction path (never the level-up reinduction path), gated by
`SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED` OR the `CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED`
runtime env override, both defaulting to OFF. The agent SHALL reset the cached
proposer's flag to False at the start of each induction so a prior stall's injection
never leaks into a level-up or program-synthesis-filter induction. This change SHALL
NOT alter the frozen scored-submission default (the flag ships False) and SHALL NOT
modify `scripts/kaggle/submission_kernel/`.

Experiment 5717 SHALL run a MATCHED-BUDGET offline A/B (control = exemplars off,
treatment = exemplars on) on the frozen live-stack proposer over a roster of games
solved to L1, scoring each induced engine's reproduction accuracy against the full
winning trajectory, and SHALL write
`results/experiment_5717_playbook_exemplars_stall_induction_ab.json`. It SHALL
declare the proposer's stochasticity and small sample explicitly and SHALL report
the direction (improved / hurt / inconclusive) honestly, never fabricating a
positive result. If the GGUF or llama-server precondition is missing it SHALL emit a
`blocked_*` verdict rather than proceed.

Required field principles:

- `honest_verdict`: principle "MUST start with a terminal prefix complete:/complete_/success:/passed:/shipped: (Verdict Terminal-Prefix Discipline); an inconclusive or negative A/B is a valid terminal outcome."
- `inference_substrate`: principle "live_llm_inference -- each arm loads and runs the real Qwen3.5-9B-MTP GGUF, so the 60s generative-inference duration floor applies."
- `playbook_exemplars_delta_accuracy`: principle "treatment-minus-control mean reproduction accuracy; the primary induction-quality signal, reported with its small-sample caveat, never inflated."
- `preconditions_checked`: principle "records that the GGUF weights + llama-server binary were verified before any inference; pre-empts silent-missing-resource fabrication."
- `random_seed`: principle "harness determinism precondition; the LLM sampling remains stochastic and that gap is disclosed in methodology_note."
- `reproducibility_checksum`: principle "content hash over config + per-arm rows catches silent drift on replay."
- `verifier_is_oracle`: principle "must be false; reproduction accuracy is an oracle-distinct induction-quality metric, not a win-check."

### REQ-ARC-WMTE-5718: Retrieval-Based (Graph-RAG) Playbook Injection

The repository SHALL provide a RETRIEVAL variant of the stall re-induction injection that
replaces the fixed exemplar block with the top-K playbook patterns relevant to the CURRENT
stuck situation. `python/carnot/agentic/arc_playbook_patterns.py` SHALL hold the patterns as
structured records (game-agnostic `statement`, `mechanic_tags` from a fixed taxonomy,
`source_games`, `citation`) -- the graph edges pattern -> mechanic-tags -> corpus citations.
Experiment 5718 SHALL build a static asset under `models/arc_playbook_index/` (index.json +
embeddings.npy + a compact AST-derived `kit_reference.json` of the Phase-2 primitives),
embedding each pattern `statement` with the SAME Qwen3.5-9B GGUF the live agent uses so the
vector space matches; substrate `live_llm_embedding_extraction`, `blocked_*` if the GGUF is
absent.

`arc_playbook_retrieval.py` SHALL load the static index and, given a query embedding, return
the top-K patterns by cosine similarity plus a mechanic-tag relevance boost, where an
on-mechanic pattern (sharing a non-universal query tag) outranks a merely-universal pattern
(a small baseline) which outranks an off-mechanic pattern (zero). `induce_prompt` and the
proposers SHALL accept `include_playbook_exemplars` as `bool | str`: False/"" -> byte-identical
prompt, True -> the static block, a non-empty string -> that retrieved block. `E3AgentPolicy`
SHALL, on the stall path only, prefer the retrieved block when `SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED`
OR `CARNOT_ARC_PLAYBOOK_RETRIEVAL` is set (default OFF), falling back to the static block then to
nothing; the query embedder failure path SHALL never crash the live path. This SHALL NOT change
the frozen scored-submission default (both flags ship False) and SHALL NOT modify
`scripts/kaggle/submission_kernel/`.

Experiment 5719 SHALL run a 3-arm matched-budget A/B (none / static / retrieval), score each
induced engine's graded cell-recall on the full winning trajectory, and report -- floor-aware
and leave-one-out variance-aware -- whether retrieval beats static (does the retrieval step earn
its complexity) and whether either beats none, never fabricating a positive result.

Required field principles:

- `honest_verdict`: principle "terminal-prefixed; a floored / no-reliable-signal 3-arm result is a valid terminal outcome."
- `retrieval_vs_static_delta`: principle "the core question -- does retrieval earn its complexity over a fixed block -- reported honestly even when null."
- `metric_floored`: principle "true when graded cell-recall is at its floor for all arms, so the offline metric cannot detect any effect."
- `inference_substrate`: principle "live_llm_inference for inductions plus a bounded live_llm_embedding_extraction step for the retrieval queries."
- `verifier_is_oracle`: principle "must be false; cell-recall is an oracle-distinct induction-quality metric."

## Scenarios

### SCENARIO-ARC-WMTE-5716: Reachability Search Reports Proven Versus Capped

Given a bounded reachability search over an abstract transition function whose goal
is unreachable and whose frontier empties with no depth or node cut
When `bounded_reachability_search` runs to completion
Then it returns `status=exhausted`, `reached=false`, and `proven_unreachable=true`.

Given the same search but with the node or depth budget hit before the frontier empties
When the search stops at the cap
Then it returns `status=capped_nodes` or `status=capped_depth`, `reached=false`, and
`proven_unreachable=false`, so no caller mistakes a capped search for a proof.

Given a multi-layer animation frame in which a sprite moves while the camera recenters
When `read_absolute_trajectory` reads the layer stack
Then it reports the sprite's ABSOLUTE per-layer centroids and a net direction that
the camera-relative settled grid alone would hide.

### SCENARIO-ARC-WMTE-5718: Retrieval Surfaces On-Mechanic Patterns And Stays Off By Default

Given a query embedding for a navigation-mechanic stuck situation with query tags including
`navigation`
When `retrieve` ranks the indexed patterns
Then patterns tagged `navigation` outrank merely-`universal` patterns, which outrank
off-mechanic patterns, so the injected top-K is specific to the situation rather than a
fixed set.

Given neither `SUBMITTED_PLAYBOOK_RETRIEVAL_ENABLED` nor `CARNOT_ARC_PLAYBOOK_RETRIEVAL` is set
When the live agent enters a stall re-induction
Then `_playbook_retrieval_gate_on()` is False and no retrieved block is injected, and
`induce_prompt` with `include_playbook_exemplars=""` returns the byte-identical prompt.

Given the retrieval gate is on but the query embedder or index is unavailable
When the agent tries to build a retrieved block
Then `_retrieve_playbook_block` returns None and the agent falls back to the static block (if
its gate is on) or to no injection, never crashing the live path.

Given the 3-arm A/B scores graded cell-recall for none / static / retrieval
When every arm's mean recall is at the floor
Then the verdict is `metric_floored` and no arm is declared better, and otherwise pairwise
directions are reported leave-one-out-variance-aware.

### SCENARIO-ARC-WMTE-5717: Exemplar Injection Is Off By Default And Stall-Scoped

Given `include_playbook_exemplars` is False (the shipped default)
When `induce_prompt` builds a world-model induction prompt
Then the prompt is byte-identical to the pre-existing prompt and contains no
exploration-principles block.

Given a proposer whose `include_playbook_exemplars` is True
When it induces a world model
Then the induction prompt is prepended with the game-agnostic exploration-principles
few-shot, and that few-shot contains no per-game color, coordinate, or mechanic name.

Given the live agent enters induction with neither `SUBMITTED_PLAYBOOK_EXEMPLARS_ENABLED`
nor `CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED` set
When a stall or level-up re-induction runs
Then `_playbook_exemplars_gate_on()` is False, the cached proposer's flag is reset to
False, and no exemplars are injected on any path.

Given the gate is on via the module flag or the env override
When a STALL / first-contact re-induction runs (not a level-up reinduction)
Then the agent arms the cached proposer's `include_playbook_exemplars` for that
induction only and records `playbook_exemplars_injected=true` in the attempt.

### SCENARIO-ARC-FCP-5575: SGE Anti-Stagnation Controller Is E3-Reachable

Given the prior SGE trace repeatedly proposes passive waiting strategies with
null outcomes
When the fixed Exp5575 collapse thresholds are evaluated
Then repeated normalized strategy text, low pairwise strategy distance, repeated
action proposals, and consecutive null outcomes are reported, and
`collapse_detected_on_prior_trace=true`.

Given collapse has been detected inside `SGECandidateRouter`
When the live candidate-router hook ranks candidates for `E3AgentPolicy`
Then no live LLM completion is required for the collapsed step, the forced
portfolio emits observation, active coordinate probe, action-type probe,
mechanic falsification, and recovery/reset hypotheses, and diversity metrics
increase relative to the collapsed history.

Given recent failed strategies exist in router history
When the controller builds its taboo set
Then taboo entries are normalized from outcome-conditioned failed strategies
only, malformed or failed LLM completions degrade to deterministic fallback
ranking, and prompts/ranking diagnostics contain no win-check, level-completion,
hidden-source, scorecard, or oracle signal.

Given the ARC registry records the candidate target depth
When Exp5575 performs the registry precheck
Then it selects only a deeper unreproduced adjacent frontier target, records
`target_unsolved=true`, and sets `live_path_ready=true` only if tests pass, the
E3 import path reaches the controller, collapse controls activate on the prior
trace, and oracle leakage gates remain false.

### SCENARIO-ARC-FCP-5562: FSM Live Attempt Banks Only Reproduced Self-Discovery

Given Exp5561 reports `arc_fsm_precheck_ready=true` and selects `r11l:L3`
And the registry still records `r11l` below L3
When Exp5562 runs one bounded FSM live attempt with no LLM proposer
Then the artifact records `upstream_arc_precheck` pointing to Exp5561,
`solve_provenance=live_agent_self_discovery`,
`llm_invoked=false`, `llm_strategy_proposer_used=false`,
`no_model_specs_required=true`,
`inference_substrate=arc_live_agent_self_discovery_no_llm`, trajectory metrics,
and no `model_specs` or `target_model`.

Given the live runtime discovers a candidate level-up trajectory
When the post-solve offline reproduction gate verifies at least one new level
Then Exp5562 updates the registry according to repository practice and emits
`complete:`, `offline_reproduced=true`, `reproduced_levels>=1`, and matching
`registry_delta`.

Given the live runtime does not discover a reproducible new level
When the bounded live budget is exhausted
Then Exp5562 emits `honest_null:`, records the same trajectory and metric fields,
sets `offline_reproduced=false`, `reproduced_levels=0`, and leaves
`registry_delta=0`.

Given the registry now records the Exp5561-selected target level as already
reproduced
When Exp5562 performs the immediate duplicate recheck
Then it emits a duplicate-prevented artifact with `attempts=0`,
`arc_live_levelup_ready=false`, a reproducible trajectory receipt, and does not
rerun the solve.

### SCENARIO-ARC-FCP-5561: FSM Target Rotation Avoids Recent No-Bank Target

Given Exp5548 selected `g50t:L3`, banked no reproduced levels, and recorded
`registry_delta=0`
And the registry records multiple adjacent unreproduced frontier targets
When experiment 5561 runs the FSM target rotation precheck
Then it rejects `g50t:L3` as a recent no-bank target, selects a different
registry-safe adjacent frontier target, records
`recent_no_bank_targets_avoided=["g50t:L3"]`, and writes
`results/experiment_5561_arc_fsm_target_rotation_precheck.json` without a
solve claim.

Given the FSM action abstraction receives repeated click coordinates from
multiple finite-state phases
When experiment 5561 applies repeated-coordinate suppression through the live
candidate-router hook
Then selected FSM actions cover multiple coordinates, `action_entropy_precheck`
is a float above the readiness threshold,
`repeated_coordinate_suppression_enabled=true`, and
`arc_fsm_precheck_ready=true` only when no LLM was invoked and all registry and
provenance gates pass.

### SCENARIO-ARC-FCP-5548: Clean Live Attempt Banks Only Reproduced Self-Discovery

Given Exp5547 reports `arc_clean_precheck_ready=true` and the registry still
has the selected target as an adjacent unreproduced frontier level
When Exp5548 runs one bounded live attempt with the recorded seed and
repeated-coordinate suppression setting
Then the artifact uses `solve_provenance=live_agent_self_discovery`,
`llm_strategy_proposer_used=false`,
`no_model_specs_required=true`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
omits `model_specs` and `target_model`, and records trajectory metrics plus an
exact replay or no-bank result.

Given the registry has already banked Exp5547's selected target
When Exp5548 prepares the live attempt
Then it rotates to a fresh adjacent frontier target, records the rotation
reason, and still applies the same clean no-LLM metadata gates.

Given a live trajectory reaches a candidate level
When the standard offline replay gate does not reproduce at least one new level
Then Exp5548 emits `honest_null:`, `offline_reproduced=false`,
`reproduced_levels=0`, and `registry_delta=0`.

### SCENARIO-ARC-FCP-5547: Clean No-LLM Precheck Blocks Duplicates And Omits Model Specs

Given the ARC solve registry has at least one reproduced game with an
unbanked adjacent frontier level
When Exp5547 builds the clean substrate precheck
Then it selects a target whose requested level is deeper than the registry
depth, records `already_reproduced=false`,
`llm_strategy_proposer_used=false`, `no_model_specs_required=true`,
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`solve_provenance=live_agent_self_discovery`, a deterministic `random_seed`,
and a non-empty `reproducibility_checksum`
And the artifact contains no `model_specs` or `target_model` field.

Given the registry already banks a proposed target level
When Exp5547 runs the duplicate precheck
Then that target is rejected before readiness and the artifact only becomes
ready after selecting a non-duplicate adjacent frontier target.

### SCENARIO-ARC-FCP-5508: Classical Perception Generation Is Runtime-Grounded

Given the Exp5507 target precheck is ready and the registry re-read shows the
selected level is not already reproducible
When Exp5508 runs one bounded live `E3AgentPolicy` attempt with the classical
perception-generation pass
Then candidate actions are generated from connected components, color blobs,
sprite overlays, salient motion, and action affordances observed at runtime
And the artifact records trajectory-taxonomy counts, prohibited-input flags,
registry before/after totals, and an honest null unless the standard
reproduction gate confirms at least one new level.

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

### SCENARIO-ARC-FCP-5423: CoEx Landmarks Persist Live Runtime Frontier Evidence

Given the live agent observes repeated frame-changing actions from its own
attempts
When the CoEx landmark frontier is asked for a multi-action prefix
Then it promotes only live-observed action clusters, persists the prefix across
resets, records a frontier transition, emits a measurement-access receipt, and
reports the discovered landmark count.

Given `lf52` L3 is not already banked in the registry
When experiment 5423 performs its registry precheck
Then it selects `target_game=lf52`, `target_level=L3`, and
`duplicate_solve_avoided=true` before running any bounded live attempt.

Given the bounded live attempt does not reproduce a new level
When the experiment writes its artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`arc_new_level_banked=false`, and `honest_verdict` starts with
`honest_null:` while preserving attempt counts, reset counts, frontier
transitions, landmarks, runtime observations, action-sequence receipts,
`no_offline_bfs=true`, and `no_per_game_adapter=true`.

### SCENARIO-ARC-FCP-5437: Registry-Guided Reinduction Attempts New Frontier

Given `cn04` has three reproduced levels and no reproduced L4 in the ARC solve
registry
When experiment 5437 performs its registry precheck
Then it selects `target_game=cn04`, `target_level=L4`, records the target as an
eligible unbanked frontier, and sets `duplicate_solve_avoided=true` before any
bounded live attempt runs.

Given the live agent gathers frame-changing runtime observations from its own
attempts
When experiment 5437 summarizes the attempt
Then it records runtime predicates, frontier transitions,
measurement-access action sequence receipts, reset count, and action count
without crediting any level unless the live-discovered sequence reproduces
beyond the registry depth.

Given the bounded reinduction attempt does not reproduce a new level
When the artifact is validated
Then `offline_reproduced=false`, `reproduced_levels=0`,
`arc_new_level_banked=false`, and `honest_verdict` starts with `honest_null:`
while preserving `registry_precheck=true`, `no_offline_bfs=true`,
`no_per_game_adapter=true`, and `inference_substrate=live_arc_agent_runtime`.

### SCENARIO-ARC-FCP-5450: Measurement-Access Rotation Avoids Stale Frontiers

Given the registry records `cn04` L3 and `re86` L2, and recent artifacts record
no-bank `cn04` L4 and repeated `re86` L3 salience attempts
When experiment 5450 performs its target precheck
Then it selects a different eligible next-level frontier, records
`registry_precheck_total_levels`, `selected_game`, `selected_target_level`, and
`target_rotation_reason`, and runs the bounded attempt only after duplicate
solve avoidance.

Given the live attempt gathers frame measurements, action-effect observations,
state-change summaries, or verifier-routed predicates from its own transitions
When the artifact is built
Then `runtime_predicates_induced` records those predicates and
`live_attempt_count` records the bounded live effort without reading hidden
source, running offline ground-truth BFS, or crediting a per-game adapter.

Given the bounded measurement-access attempt does not reproduce a level deeper
than the registry precheck
When the artifact is validated
Then `offline_reproduced=false`, `new_level_reproduced=false`,
`new_levels_banked=0`, `arc_new_level_banked=false`, and `honest_verdict`
starts with `honest_null:` while preserving the target rotation reason,
frontier evidence, attempted predicates, and residual wall.

### SCENARIO-ARC-FCP-5464: Metric Precheck Rejects Duplicate And Off-Path Credit

Given the ARC registry records a reproduced depth for a public game
When experiment 5464 audits a claimed solve whose target level is less than or
equal to that depth
Then the claim is rejected as duplicate, `duplicate_solve_rejected=true`, and
the reproduced total is unchanged.

Given a claimed solve uses `outer_loop_re`, source reading, offline
ground-truth BFS, a replay-only artifact, or a hand per-game adapter as the
credited path
When experiment 5464 audits provenance
Then the claim is rejected before solve credit, `off_path_solve_rejected=true`,
and the artifact remains a no-solve precheck.

Given reproduced loop artifacts and registry rows are available
When experiment 5464 audits null-coordinate exploit validity
Then one-step ACTION6 solves with null or missing click coordinates are treated
as metric contamination, while normal multi-step or coordinate-bearing replays
leave `null_coordinate_exploit_valid=false`.

### SCENARIO-ARC-FCP-5464: Perception Receipts And Target Shortlist Are Live-Path Ready

Given the live `E3AgentPolicy` can reach the connected-component salience prior
When experiment 5464 runs perception diagnostics
Then the receipts JSON contains connected component rows, color-blob rows,
changed-pixel rows, salience-tier rows, and action-effect observation rows
from live-path reachable code.

Given recent no-bank targets include `re86` L3, `lf52` L3, `cn04` L4, and
`ka59` L2
When experiment 5464 builds the Exp5465 shortlist
Then shortlisted targets avoid already reached levels and those recent no-bank
lanes unless a row records an explicit justification.

### SCENARIO-ARC-FCP-5465: Gated Salience Attempt Uses Live Features And Reproduction Credit

Given Exp5464 reports `arc_metric_integrity_ready=true` and a non-empty
`target_shortlist`
When experiment 5465 re-runs the registry precheck
Then it selects a target from that shortlist, records the precheck depth as
`target_level_before`, and attempts `target_level_before + 1`.

Given the live `E3AgentPolicy` salience path is available
When experiment 5465 builds its live attempt receipts
Then `perception_features_used` contains connected-component, color-blob,
changed-pixel, salience-tier, and action-effect features from live-path
reachable code.

Given a bounded live/offline attempt does not reproduce deeper than the selected
precheck depth
When experiment 5465 validates its artifact
Then `offline_reproduced=false`, `reproduced_levels` remains at
`target_level_before`, `new_level_banked=false`,
`arc_registry_update_required=false`, `source_reading_used=false`,
`offline_bfs_used=false`, `hand_adapter_credited=false`, and
`honest_verdict` starts with `honest_null:`.

Given a candidate reaches a deeper level through the live-agent route
When the official reproduction gate confirms the candidate beyond the selected
precheck depth
Then and only then `offline_reproduced=true`, `new_level_banked=true`,
`arc_registry_update_required=true`, and `honest_verdict` starts with
`complete:`.

### SCENARIO-ARC-FCP-5479: Rotated Target Precheck Avoids Duplicate And Recent No-Bank Lanes

Given Exp5464 shortlisted `bp35:L3`, `sb26:L3`, `g50t:L3`, `dc22:L3`, and
`sp80:L3`, Exp5465 no-banked `bp35:L3`, and the registry records those games at
L2
When experiment 5479 selects a target before any level-up attempt
Then it rejects already reproduced duplicate target probes, avoids
`bp35:L3`, `ka59:L2`, and `cn04:L4`, selects the first eligible rotated target
from `sb26:L3`, `g50t:L3`, `dc22:L3`, or `sp80:L3`, and records the registry
total before the attempt.

Given the submitted live salience path is importable
When experiment 5479 runs the bounded dry check
Then the salience summary reports connected components, color blobs, changed
cells, target-region candidates, and known blockers without hidden source
reading, offline BFS, or hand-adapter credit.

Given the artifact is validated
When experiment 5479 writes
`results/experiment_5479_arc_target_rotation_precheck_v497.json`
Then `arc_target_rotation_ready=true`, `solve_claimed=false`,
`inference_substrate=arc_live_path_precheck_no_solve`, and `honest_verdict`
contains no level-solve claim.

### SCENARIO-ARC-FCP-5480: Rotated Salience Attempt Banks Only Reproduced New Levels

Given Exp5479 selected `sb26:L3` and the registry records `sb26` at L2
When experiment 5480 loads the target before attempting a level-up
Then it records `game=sb26`, `target_level=3`,
`reproduced_levels_before=2`, and proceeds only because the target is not
already reproduced.

Given the registry already records the selected target level or the Exp5479
target fields are missing
When experiment 5480 runs
Then it emits a `blocked:` artifact with `action_count=0`,
`offline_reproduced=false`, `new_level_banked=false`, and
`registry_updated=false`.

Given the bounded live salience attempt does not reproduce a level beyond the
precheck depth
When experiment 5480 validates its artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`reproduced_levels_after=reproduced_levels_before`,
`new_level_banked=false`, `registry_updated=false`,
`hidden_source_reading=false`, `offline_bfs_used=false`,
`hand_adapter_used=false`, `outer_loop_re_used=false`, and
`honest_verdict` starts with `honest_null:`.

Given a candidate reaches the selected target level through the live-agent
self-discovery path
When the registry-approved reproduction gate confirms it beyond the precheck
depth
Then and only then `offline_reproduced=true`, `reproduced_levels>=1`,
`new_level_banked=true`, `registry_updated=true`, `first_win_trace_path` is
non-empty, and `honest_verdict` starts with `complete:`.

### SCENARIO-ARC-FCP-5493: Trajectory Target Precheck Avoids Stale And Retired Lanes

Given the registry records `sb26`, `bp35`, and `re86` at L2, `ka59` at L1,
`cn04` at L3, and `dc22` at L2
When experiment 5493 evaluates trajectory/option-induction candidates
Then it excludes `sb26:L3`, `bp35:L3`, `ka59:L2`, `cn04:L4`, and `re86:L3`,
rejects any candidate target whose level is already reproduced, rejects
novelty-only, curiosity-only, energy-as-fitness quality-diversity, and
archive-granularity reruns from the Exp5154 retired scope, and selects the
first remaining target with live-observation trajectory hooks.

Given `dc22:L3` survives those filters
When experiment 5493 writes
`results/experiment_5493_arc_trajectory_target_precheck_v498.json`
Then `registry_path=ops/arc_solve_registry.yaml`,
`duplicate_solve_avoided=true`, `selected_game=dc22`,
`selected_target_level=3`, `prior_levels_reproduced=2`,
`offline_bfs_used=false`, `per_game_adapter_used=false`,
`arc_trajectory_precheck_ready=true`,
`inference_substrate=registry_precheck_no_solve`, and `honest_verdict`
starts with `complete:`.

Given no candidate survives duplicate, recent no-bank, and retired-scope
filters
When experiment 5493 writes its artifact
Then `selected_game` is an empty string, `selected_target_level=0`,
`arc_trajectory_precheck_ready=false`, `offline_bfs_used=false`,
`per_game_adapter_used=false`, and `honest_verdict` starts with `blocked:`.

### SCENARIO-ARC-FCP-5494: Exp5493 Target Attempts Through Live Trajectory Induction

Given Exp5493 selected `dc22:L3`, the registry records `dc22` at L2, and the
Exp5493 trajectory preconditions include runtime action-effect observations,
visible toggle/navigation state changes, level-counter deltas, and frontier
prefix grouping
When experiment 5494 performs its pre-attempt gate
Then it selects `selected_game=dc22`, `target_level=3`,
`prior_levels_reproduced=2`, `solve_provenance=live_agent_self_discovery`,
`offline_bfs_used=false`, `per_game_adapter_used=false`,
`game_source_read=false`, and proceeds only because the target is not already
reproduced and is not a recent same-mechanism no-bank duplicate.

Given the live agent executes a bounded `E3AgentPolicy` attempt with
`LiveCoExLandmarkFrontierGenerator`
When the attempt is summarized
Then the artifact records `trajectory_hypothesis_count`,
`live_attempt_count`, hypothesized action sequences, observation deltas,
verifier checks, rejection reasons, and `model_specs_if_llm_used=[]` when no
LLM generator was invoked.

Given the bounded live trajectory-induction attempt does not reproduce a new
level beyond the prior registry depth
When experiment 5494 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`new_level_banked=false`, `registry_updated=false`,
`post_levels_reproduced=prior_levels_reproduced`, and `honest_verdict` starts
with `honest_null:`.

Given a live self-discovered trajectory candidate reaches the target level
When the standard reproduction gate confirms it beyond the prior depth
Then and only then `offline_reproduced=true`, `reproduced_levels>=1`,
`new_level_banked=true`, `registry_updated=true`,
`post_levels_reproduced>prior_levels_reproduced`, and `honest_verdict` starts
with `complete:`.

### SCENARIO-ARC-FCP-5507: Null-Coordinate Perception Precheck Selects Changed Mechanism

Given Exp5494 records a `dc22:L3` no-bank trajectory attempt, the registry
records `dc22` at L2, and prior no-bank artifacts record `bp35:L3` and
`sb26:L3` salience-only nulls
When experiment 5507 aggregates upstream evidence
Then it rejects the already reproduced duplicate probes, rejects recent
same-target/same-mechanism no-bank reruns, audits Exp5494's zero-change click
receipts as valid recorded ACTION6 coordinates rather than missing/null
coordinates, records perception-grounding findings from connected components,
color blobs, salience tiers, changed pixels, and action-effect asymmetry, and
selects `dc22` `L3` only with a materially changed perception-generation
mechanism.

Given experiment 5507 writes
`results/experiment_5507_arc_null_coordinate_perception_precheck_v499.json`
When the artifact is validated
Then `registry_path=ops/arc_solve_registry.yaml`,
`reproducible_total_levels_before` is a bare int,
`duplicate_targets_rejected` and `recent_no_bank_targets_rejected` are lists,
`null_coordinate_audit` is a dict, `perception_grounding_findings` is a list,
`selected_game` and `selected_level` are strings,
`levelup_attempt_ready=true`, `solve_claimed=false`,
`inference_substrate=aggregation_from_upstream_artifacts`, and
`honest_verdict` starts with `complete:`.

Given every candidate is either already reproduced, a same-mechanism no-bank
rerun, or lacks perception evidence
When experiment 5507 builds its artifact
Then it leaves `selected_game`, `selected_level`, and `selected_mechanism`
empty, sets `levelup_attempt_ready=false`, keeps `solve_claimed=false`, and
uses a `blocked:` honest verdict with exact reasons.

### SCENARIO-ARC-FCP-5520: Action-Diversity Precheck Selects Changed Target

Given Exp5508 repeatedly chose ACTION6 at the same small coordinate set and did
not bank a new level
And the registry records the Exp5508 target depth but has a different candidate
game with an unreproduced next level
When experiment 5520 runs the no-credit action-diversity precheck
Then it rejects the Exp5508 target/pattern, rotates to one registry-safe target,
measures action entropy, repeated-coordinate rate, and salience coverage from
the changed live-path candidate generator, and writes
`results/experiment_5520_arc_action_diversity_target_precheck.json` without a
solve claim.

Given the registry already reproduces the candidate level or the dry-run probe
collapses to the Exp5508 repeated-coordinate pattern
When experiment 5520 validates the artifact
Then `arc_levelup_candidate_ready=false`, `already_reproduced` or
`exp5508_pattern_reused` records the blocker, and `honest_verdict` starts with
`blocked:`.

### SCENARIO-ARC-FCP-5521: Live Action-Diverse Attempt Is Reproduction-Gated

Given Exp5520 selected `sb26:L3` with a changed action-diverse generator and
the registry still records only `sb26` depth 2
When experiment 5521 runs the bounded live self-discovery attempt
Then the attempt records observations, proposed actions, verifier feedback,
action entropy, repeated-coordinate rate, and salience coverage in a trajectory
log
And the result writes
`results/experiment_5521_arc_live_action_diverse_levelup.json` with
`solve_provenance=live_agent_self_discovery` and
`inference_substrate=arc_live_agent_self_discovery`.

Given the live attempt does not reproduce a new level through the standard
offline replay gate
When experiment 5521 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`banking_gate=false`, `registry_delta=0`, `reproduction_command=null`, and
`honest_verdict` starts with `honest_null:` while preserving enough trajectory
detail to distinguish the attempt from Exp5508.

Given the live attempt reaches a new level and the replay gate reproduces it
When experiment 5521 banks the result
Then `offline_reproduced=true`, `reproduced_levels>=1`,
`banking_gate=true`, `registry_delta` equals `reproduced_levels`, and the
artifact records the exact reproduction command used for the live-discovered
trajectory.

### SCENARIO-ARC-FCP-5533: Strategy-Routing Precheck Selects A Non-Stale Live Target

Given Exp5521 selected `sb26:L3` but banked no level and recorded repeated
coordinate collapse
And the registry records multiple adjacent unreproduced frontier targets
When experiment 5533 runs the strategy-routing precheck
Then it rejects the stale Exp5521 target, selects one non-duplicate adjacent
frontier target, records a bounded strategy portfolio with at least three
strategies, proves the strategy router is reachable through the live candidate
router hook, and writes
`results/experiment_5533_arc_strategy_routing_precheck.json` without a solve
claim.

Given repeated candidate coordinates would otherwise be selected by multiple
strategies
When experiment 5533 applies repeated-coordinate suppression
Then the selected action sequence changes before diversity metrics are
computed, `repeated_coordinate_suppression_enabled=true`,
`repeated_coordinate_rate_precheck` is lower than the unsuppressed route, and
`arc_sge_candidate_ready=true` only when target, routing, suppression, and
metric gates pass.

### SCENARIO-ARC-FCP-5534: Strategy-Routed Live Attempt Is Reproduction-Gated

Given Exp5533 selected `g50t:L3` and `arc_sge_candidate_ready=true`
And the registry still records only `g50t` depth 2
When experiment 5534 runs the bounded live self-discovery attempt
Then the attempt installs the Exp5533 strategy portfolio on the live candidate
router, enables repeated-coordinate suppression, records strategy choices,
verifier routes, suppression events, and level counter changes, and writes
`results/experiment_5534_arc_strategy_routed_levelup.json` with
`solve_provenance=live_agent_self_discovery` and
`inference_substrate=arc_live_agent_self_discovery`.

Given the registry already reproduces the Exp5533 target level
When experiment 5534 starts
Then it writes a `blocked_duplicate_target` artifact, keeps
`registry_delta=0`, chooses no replacement target, and does not run the live
agent.

Given the live attempt does not reproduce a new level through the standard
offline replay gate
When experiment 5534 validates the artifact
Then `offline_reproduced=false`, `reproduced_levels=0`,
`registry_delta=0`, and `honest_verdict` starts with `honest_null:` while
preserving the trajectory evidence.

Given the live attempt reaches a new level and the replay gate reproduces it
When experiment 5534 banks the result
Then `offline_reproduced=true`, `reproduced_levels>=1`, `registry_delta`
equals `reproduced_levels`, and the registry update records the Exp5534
live-path reproduction evidence.

### REQ-ARC-FCP-5591: Translation-Invariant Object Identity + Containment/Adjacency Topology

`ops/known-issues.md`'s 2026-07-11 task 10 entry (folded into task 2, not a
separate experiment) identified two sub-components a real top-3 ARC-AGI-3
competitor's open-sourced classical connected-component segmentation adds
beyond `ColorBlobSaliencePrior`'s existing size/color salience tiers: a
translation-invariant object-identity signature (so the same shape+color
object hashes identically across frames regardless of position -- attacking
the GAP-4891 / `project_arc_live_agent_learning_gaps` binding constraint that
frame-only, position-only features sit at LOO=chance), and a containment tree
+ adjacency graph over the blob list (which objects sit inside which, and
which touch).

`python/carnot/agentic/arc_color_blob_salience.py` SHALL expose
`object_hash(blob: ColorBlob) -> str`, a sha1 signature of the blob's color
plus its cell-shape pattern normalized to its own bounding box's top-left
corner as origin, and `blob_topology(frame) -> dict`, computing the FULL
(unfiltered, `min_pixels=1, max_component_fraction=1.0`) connected-component
partition of a frame via the existing `connected_color_blobs` (unmodified)
and returning `blobs` (list, index is blob id), `object_hashes` (id ->
`object_hash` value), `children` (id -> sorted list of directly-enclosed
blob ids, computed via complement flood-fill from the frame border per blob),
and `adjacency_list` (sorted `[i, j]` id pairs for 4-connected-edge-sharing
blobs). Both additions SHALL be pure functions over existing primitives with
no change to `ColorBlob`'s fields or `connected_color_blobs`'/
`ColorBlobSaliencePrior`'s existing signatures or outputs -- this is an
additive perception-data extension, not a scoring or ranking change.

Required field principles:

- `object_hash`: principle "two blobs with the same shape and color hash identically regardless of position, giving the agent a position-invariant object-identity feature the existing bbox/centroid/tier fields do not provide."
- `blob_topology.children`: principle "innermost-encloser parent assignment yields a clean nesting tree (an object's parent is not every ancestor, just the tightest enclosing blob)."
- `blob_topology.adjacency_list`: principle "any two blobs sharing a 4-connected edge, including parent/child pairs since they physically touch -- lets a consumer reason about spatial relationships beyond raw bbox overlap."

#### SCENARIO-ARC-FCP-5591-TRANSLATION-INVARIANT-IDENTITY

Given a frame containing two same-color, same-shape blobs at different
positions (e.g. two 3x3 squares of the same color, one elsewhere on the
grid)
When `object_hash` is computed for each blob
Then both blobs receive the identical hash value

Given a frame containing two blobs of the same color but a DIFFERENT shape,
or the same shape but a DIFFERENT color
When `object_hash` is computed for each
Then the two blobs receive different hash values

#### SCENARIO-ARC-FCP-5591-CONTAINMENT-AND-ADJACENCY

Given a frame with a ring-shaped blob of one color fully enclosing a smaller
solid blob of a different color
When `blob_topology` runs
Then the inner blob's id appears in `children[ring_id]`, and the pair
appears in `adjacency_list` (the ring and its contents share a 4-connected
edge)

Given a frame with two same-shape, same-color blobs where one is enclosed by
a third blob and the other sits freely on the background
When `blob_topology` runs
Then only the enclosed blob's id appears under its true parent's `children`
entry, the freely-sitting blob has no parent, and `object_hashes` reports
the SAME hash for both despite their different topological position

### REQ-ARC-FCP-5591-2: Object-Hash Change-History Bonus (Task 10's Deferred Live-Wiring Step)

REQ-ARC-FCP-5591's DONE note deferred its own suggested live-consuming
mechanism -- "preferring an object whose hash was seen to change in a prior
frame" -- as "a distinct, separately-scoped design + empirical-validation
step per the Phase Prototype + Empirical Validation discipline." This
requirement closes that gap. `ObjectHistorySaliencePrior`
(`python/carnot/agentic/arc_object_history_salience.py`) SHALL wrap
`ColorBlobSaliencePrior` (mirroring the existing
`arc_geometric_salience.GeometricSaliencePrior` precedent for composing a
mutable history-tracking layer on top of the frozen base prior) with a
per-`object_hash` `(obs, changed)` tally, updated via `observe_transition`
from real observed click transitions and consumed via `score` as an additive
bonus proportional to the observed change rate, gated behind a
`min_observations` evidence floor (mirroring `InertClickSigPruner`'s
trust+specificity discipline, inverted polarity: boost instead of prune,
identity-hash-keyed instead of structural-signature-keyed). An
under-observed hash SHALL receive zero bonus, never a penalty.

`coerce_object_history_salience_prior` SHALL provide the standard
`None`/`False` -> unchanged base, instance -> passthrough, `True` -> wrap
coercion contract, threaded through `E3AgentPolicy.__init__` as
`object_history_salience`, gated OFF by default
(`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG`). Because `action_prior` is already a generic,
externally-composable slot that `_ingest`'s existing `hasattr`-gated
`observe_transition`/`reset` hooks and `_candidates`' existing
`action_prior.score` consumption dispatch to generically, wiring this in
requires NO new hook call sites in `arc_competition_agent.py` -- unlike
REQ-ARC-FCP-5595's `InertClickSigPruner`, which needed a brand-new
`rank_candidates` call site.

### SCENARIO-ARC-FCP-5591-2-CHANGE-RATE-BONUS: Reliable Change History Outscores Reliably-Inert History

Given the SAME object (identical `base_prior` score), observed `>=
min_observations` times either always changing the frame or never changing
it when clicked
When `score` is called for a click candidate on that object
Then the reliably-changing history's score is strictly higher than the
reliably-inert history's score, and the reliably-inert history's score
equals the unmodified base score (zero bonus, not a penalty)

### SCENARIO-ARC-FCP-5591-2-EVIDENCE-FLOOR: Under-Observed Hashes Get Zero Bonus

Given an object observed fewer than `min_observations` times, regardless of
how many of those observations changed the frame
When `score` is called for a click candidate on that object
Then the score equals the unmodified base score -- no premature boost from
sparse evidence

### SCENARIO-ARC-FCP-5591-2-NOT-DEGENERATE: Adversarial Check Against Base-Tier Redundancy

Given two click candidates whose `base_prior` scores are IDENTICAL (same
color, size, and shape, hence the same tier and button-likelihood features)
before any observed history
When one of the shared-hash objects accumulates a reliable change-history
tally and `score` is called for both candidates
Then both candidates' final scores are boosted identically above the shared
base score (the mechanism is hash-identity-based, not position-based) --
confirming the bonus is genuinely new information, not a re-derivation of
the base tier features under a different name

### SCENARIO-ARC-FCP-5591-2-DEFAULT-OFF-PARITY: Unwrapped By Default

Given the SUBMITTED default configuration
When a `StepwiseExplorer` or `E3AgentPolicy` is constructed with no explicit
`object_history_salience` argument
Then `action_prior` is a plain, unwrapped `ColorBlobSaliencePrior`
(`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["object_history_salience_enabled"]`) -- byte-identical
to the agent's behavior before this requirement, until a matched-budget
offline A/B validates flipping it on, per the `solve_rate_dropped` guardrail

### SCENARIO-ARC-FCP-5591-2-REAL-GAME-NON-DEGENERATE-SIGNAL: Real-Game Empirical Validation

Given real click transitions collected from a real `E3AgentPolicy`
exploration run on a click-heavy game (`m0r0`, per REQ-ARC-FCP-5595's own
confirmed roster)
When those transitions are fed through `ObjectHistorySaliencePrior.observe_
transition`
Then the artifact reports, honestly, whether any `object_hash` cleared both
`min_observations` and shows a nonzero change rate (a genuine, non-degenerate
signal for the mechanism to act on) -- zero is an honest, valid outcome when
evidence is sparse within the measured budget, not an error; and an
adversarial degeneracy check reports whether any two real click candidates
sharing an identical `base_prior` score were differentiated by history alone

### REQ-ARC-FCP-5591-3: Matched-Budget A/B -- The Flip-Decision Measurement

The `solve_rate_dropped` guardrail names a matched-budget A/B as the
precondition for ever flipping `SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED` to
`True`. Unlike REQ-ARC-FCP-5595-2's `InertClickSigPruner` A/B,
`ObjectHistorySaliencePrior` cannot be measured through `OfflineSolver`
(`action_prior` is not a `move_pruner`; `OfflineSolver` has no `action_prior=`
concept), and no `states_expanded`-equivalent counter exists on the live
`StepwiseExplorer`/`E3AgentPolicy` path. This requirement therefore measures
TRAJECTORY DIVERGENCE -- whether enabling the mechanism actually changes
which candidates a real `E3AgentPolicy` exploration run selects, compared
against an identical-budget baseline with the mechanism off -- as the
honestly-available substitute, since the tested game/policy does not reach a
level-up within the tested budget (no `actions_to_first_levelup` available).
The experiment SHALL run three arms on the SAME game/budget: baseline
(disabled), the real default `change_bonus_weight`, and a diagnostic arm with
`change_bonus_weight` rescaled to match `ColorBlobSaliencePrior`'s own
tier-score magnitude -- isolating whether the mechanism CAN influence
behavior at an appropriate scale, separate from whether the current default
is well-tuned. Any hypothesis formed from an informal check without a real
baseline comparison SHALL be re-verified against the formal three-arm result
before being reported, per this project's own "cross-check surprising
results" discipline.

### SCENARIO-ARC-FCP-5591-3-DEFAULT-WEIGHT-NO-OP: Identical Trajectories At Default Weight Is An Honest Null

Given the baseline (`object_history_salience=False`) and default-weight
treatment (`object_history_salience=True`, `change_bonus_weight=10.0`) arms
run on the same game and matched budget
When their action sequences are compared
Then an identical trajectory is reported as an honest, valid null (the
mechanism tracks real evidence but the bonus magnitude never changes
candidate ranking at this weight), not a failure requiring escalation

### SCENARIO-ARC-FCP-5591-3-RESCALED-WEIGHT-STILL-NO-OP: A Rescaled Bonus Weight Isolates Scale From Structure

Given a diagnostic arm with `change_bonus_weight` rescaled to match
`ColorBlobSaliencePrior`'s real tier-score magnitude, run on the same game
When compared against the baseline trajectory
Then the artifact reports honestly whether the rescaled weight changes
behavior (confirming the mechanism CAN matter at an appropriate scale) or
remains identical to baseline (an open question for follow-up, not assumed
to be an over-exploitation risk without a real baseline-comparison check --
a repeated-coordinate pattern that appears IDENTICALLY in the baseline
disproves that it is bonus-induced)

### REQ-ARC-FCP-5590: Dict-Candidate CNN Fix Matched-Budget A/B

`docs/research-notes/arc-perception-grounding-audit-2026-07-13.md` found that
`FrameChangeScorer.candidate_score`'s `getattr(candidate, "action_id")`
raised `AttributeError` on dict-shaped candidates, silently zeroing the CNN
term of the live default frame-change scorer on every
`ActionEffectExpansionPrior.frontier_priority` call (the always-on default
frontier-priority computation,
`SUBMITTED_ACTION_EFFECT_EXPANSION_PRIOR_ENABLED = True`). The fix
(`arc_frame_change_predictor._as_action_like`, normalizing a dict candidate
to an attribute-bearing shim before the CNN call) is low-risk by
construction (the pre-fix behavior was "silently contributes zero," so a
correct fix can only add signal, never regress the unaffected
`PersistentAEM` memory term), but per CLAUDE.md's Phase Prototype +
Empirical Validation + Adversarial Check Discipline a fix that stops a
silent bug is not automatically a capability win and must be measured before
being trusted as one.

`python/carnot/experiment_5590_frame_change_cnn_dict_candidate_fix_ab.py`
SHALL run a matched-budget A/B across the full `CLAIMED` 11-game roster,
CONTROL reproducing the PRE-FIX behavior via a scoped monkeypatch of
`arc_frame_change_predictor._as_action_like` to the identity function (the
same shipped `E3AgentPolicy` construction otherwise, restored after each
game so concurrent TREATMENT runs are unaffected) and TREATMENT using the
real, unmodified fixed default. Tier-3 LLM induction SHALL be disabled
(`CARNOT_ARC_DISABLE_INDUCTION=1`) so the measurement isolates the
search/frontier-priority effect from induction noise and wall-clock.

The artifact SHALL report `levels_gained_control_total`,
`levels_gained_treatment_total`, `per_game_levels_delta`,
`states_expanded_control_total`, `states_expanded_treatment_total` (each
arm's total distinct explored-state count, `len(policy.explorer.graph)`, a
search-behavior proxy independent of whether any level was actually
reached), and `levels_gained_headroom_present` (CLAUDE.md
FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at
least one arm reached a nonzero level somewhere on the roster).

**RESOLUTION (2026-07-13).** Ran cleanly on the full 11-game roster,
budget=200/game. `levels_gained_headroom_present: true` (1 level reached in
both arms, `lp85`) -- the null is interpretable, not a degenerate
zero-headroom test. `per_game_levels_delta`: zero on every single game.
`states_expanded_control_total` and `states_expanded_treatment_total` were
IDENTICAL, and per-game `states_expanded` matched EXACTLY (not just in
aggregate) for all 11 games -- a much stronger and more informative null
than "levels didn't change": the fix produced literally zero measurable
difference in which states the explorer visited, in either direction.
Consistent with the audit's own honest bound (the CNN term's blend weight is
already small, `cnn_weight=0.05` vs `memory_weight=1.0`) and with the
project's prior LOO=chance finding on frame-only features
(`project_arc_live_agent_learning_gaps`): the underlying signal the bug was
blocking was itself not load-bearing at this weight and on this roster, not
that the bug was somehow inert. The fix remains correct code hygiene (a
scorer that silently discards part of its own blended signal on a
type-shape mismatch is a latent hazard regardless of today's measured
weight), but does not currently move live-agent capability.

Required field principles:

- `levels_gained_headroom_present`: principle "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster, else the null may just mean neither arm had headroom."
- `states_expanded_control_total` / `states_expanded_treatment_total`: principle "a search-behavior proxy independent of whether any level was reached -- a per-game exact match is a stronger null signal than an aggregate levels-only comparison."
- `control_results`: principle "PRE-FIX behavior reproduced via a scoped monkeypatch of _as_action_like to identity -- the SAME shipped construction with one function swapped for the duration of the run, not a different code path."

#### SCENARIO-ARC-FCP-5590-MATCHED-BUDGET-DELTA

Given the real `E3AgentPolicy` cascade run on the same game under CONTROL
(pre-fix `_as_action_like` identity monkeypatch) and TREATMENT (the real
fixed default) with the same budget and induction disabled
When both arms complete
Then `per_game_levels_delta` and any `states_expanded` difference for that
game reflect ONLY the dict-candidate CNN-term fix, with no other
construction difference between the two arms

### REQ-ARC-FCP-5592: Candidate-Scoring Stack Vs Bare-Control Matched-Budget A/B

`docs/research-notes/arc-agi3-milestone1-winners-sota-ingestion-2026-07-11.md`
(O2) found the 3rd-place ARC-AGI-3 team ("forge") had an architecturally
equivalent candidate-selection "arbiter" slot to our own
`candidate_router` + DAgger value head + goal-energy candidate guidance +
navigation-cost-tiebreak stack, but disabled their arbiter (a second LLM
call) in their winning config for cost, keeping only a hand-tuned static
fallback. `SUBMITTED_AGENT_CONFIG["bare_control_config"]` already documents
the exact on/off toggle matching forge's own ablation, but no experiment had
ever run it against the real full-stack default and reported the delta --
this is the missing measurement, not a new scorer build.

`python/carnot/experiment_5592_candidate_scoring_stack_bare_control_ab.py`
SHALL run a matched-budget A/B across the full `CLAIMED` 11-game roster,
FULL STACK being the real, unmodified live `E3AgentPolicy` default, and
BARE CONTROL being `E3AgentPolicy` constructed with
`SUBMITTED_AGENT_CONFIG["bare_control_config"]`'s knobs mapped to their real
constructor kwargs (`target_levels=1`, `value_weight=0.0`,
`candidate_router=None`, `navigation_cost_tiebreak=False`,
`action_effect_expansion_prior=False`, `goal_bias=None`,
`goal_candidate_guidance=False`). Tier-3 LLM induction SHALL be disabled
(`CARNOT_ARC_DISABLE_INDUCTION=1`) so the measurement isolates the
candidate-SELECTION axis forge's ablation targeted.

The artifact SHALL report `levels_gained_full_stack_total`,
`levels_gained_bare_control_total`, `per_game_levels_delta`,
`efficiency_full_stack_total`/`efficiency_bare_control_total` (summed
per-game `arc_agi.scorecard.EnvironmentScoreCalculator` efficiency, the
action-efficiency half of forge's own reported metric), and
`levels_gained_headroom_present` (CLAUDE.md FALSE_NEGATIVE_RISK discipline).

**RESOLUTION (2026-07-13).** Ran cleanly on the full 11-game roster,
budget=200/game. `levels_gained_headroom_present: true` (`lp85` reached L1
in both arms) -- interpretable, not a degenerate zero-headroom test.
`per_game_levels_delta`: zero on every single game;
`efficiency_full_stack_total == efficiency_bare_control_total` (2.7778,
matching to 4 decimal places). Verified this is a real result, not a
construction bug: the two arms' `lp85` rows show genuinely DIFFERENT search
behavior (`actions_to_first_levelup`: 7 for full stack vs 5 for bare
control; `actions` total 198 vs 5, since `bare_control_config`'s
`target_levels=1` correctly stops bare control immediately after L1 while
the full stack continues toward its default target of 3) -- the ablation
demonstrably took effect. The efficiency SCORE nonetheless matched exactly
because `arc_agi.scorecard`'s per-level score is `min((human/agent)^2*100,
115)`: both 5 and 7 actions are already well under `lp85` L1's human
baseline, so both arms saturate at the same capped per-level score
regardless of the 2-action difference between them -- a real property of
the (capped) metric, not evidence the ablation was a no-op.

**Honest conclusion: on this roster/budget, the richer candidate-scoring
stack (candidate_router + DAgger value head + goal-energy candidate
guidance + navigation-cost tiebreak) produces NO measured level-up or
action-efficiency advantage over the bare control forge's own ablation
methodology would compare it against.** This does not establish the stack
is worthless (it may earn its keep at a different budget, on a different
roster, or via a mechanism `EnvironmentScoreCalculator`'s capped-per-level
formula does not reward, e.g. reducing variance or avoiding worse-case
failures elsewhere), but it means the claim "our scoring stack is the
arbiter forge wanted but couldn't afford" is NOT YET empirically supported
on this measurement and should not be cited as a moat without a
follow-up at a different budget/roster or metric that is genuinely
sensitive to the ablation.

Required field principles:

- `levels_gained_headroom_present`: principle "CLAUDE.md FALSE_NEGATIVE_RISK discipline -- a null delta is only interpretable if at least one arm shows nonzero levels_gained somewhere on the roster."
- `bare_control_kwargs`: principle "the real E3AgentPolicy constructor kwargs mapped from SUBMITTED_AGENT_CONFIG['bare_control_config'] -- documents exactly what was ablated, matching forge's own on/off toggle."
- `efficiency_full_stack_total` / `efficiency_bare_control_total`: principle "sum of the leaderboard harness's own per-game efficiency score -- the action-efficiency half of forge's own reported metric, not just level count; a capped-per-level formula can saturate identically for two genuinely different search behaviors on a shallow level, which is a real property of the metric, not evidence of a construction bug."

#### SCENARIO-ARC-FCP-5592-STACK-VS-BARE-DELTA

Given the real `E3AgentPolicy` cascade run on the same game under FULL
STACK (the real unmodified default) and BARE CONTROL
(`SUBMITTED_AGENT_CONFIG["bare_control_config"]`'s knobs applied) with the
same budget and induction disabled
When both arms complete
Then `per_game_levels_delta` and any `efficiency` difference for that game
reflect ONLY the candidate-selection-stack ablation, with no other
construction difference between the two arms, and a per-game row inspection
(actions taken, actions to first level-up) can distinguish a genuine null
from a construction bug that silently made both arms identical

### REQ-ARC-FCP-5701-HEADROOM-RESCOPE: Broader-Headroom Re-Run of the Candidate-Scoring-Stack Ablation

REQ-ARC-FCP-5592's own resolution flagged its limitation honestly: "this
roster's near-total lack of headroom -- only 1/11 games reached any level at
all -- bounds how informative this specific run can be." Diagnosing that
limitation further: exp5592's 11-game roster mixed adaptered and
UN-adaptered games (`wa30`, `sc25` have no registered `GameAdapter` in
`arc_game_adapters.py` and are structurally unreachable by the generic
policy at any budget within a reasonable range -- confirmed by a
same-session calibration probe that also found `wa30`/`sc25` still scored
zero at `budget=600`), so exp5592's near-total floor was partly a roster
artifact, not purely a stack-doesn't-matter finding.

`python/carnot/experiment_5701_candidate_scoring_stack_bare_control_ab_headroom.py`
SHALL re-run the IDENTICAL ablation methodology (same `BARE_CONTROL_KWARGS`,
same `CARNOT_ARC_DISABLE_INDUCTION=1` isolation, same reported fields) on
the full `arc_game_adapters.adaptered_games()` roster (every game the live
path has a registered `GameAdapter` for) at `budget=500` -- calibrated by
the same probe finding that raising budget 200->600 lifted the level>=1 hit
rate on adaptered games from ~9% (1/11) to ~50% (3/6 tested); 500 is a
margin-preserving midpoint. The artifact SHALL additionally report
`n_games_with_headroom` (count of games where EITHER arm reached level>=1,
not just a single boolean) and `prior_attempt` (the CLAUDE.md
Failed-Experiment Rerun Discipline block naming exp5592, its root cause, and
what is different here), so this re-scope is a documented root-cause fix,
not a doomed re-run.

**RESOLUTION (2026-07-14).** Ran cleanly on the full 22-game adaptered
roster, `budget=500` (`duration_s=1255.5`). **`n_games_with_headroom=5`**
(`lp85`, `sp80`, `su15`, `tu93`, `vc33`) -- a 5x improvement over exp5592's
single game, giving the arm comparison genuine statistical footing. Per-game
result: `lp85` tied (1-1, identical efficiency 2.7778 both arms, matching
exp5592's own finding on this game exactly -- a direct cross-check that
nothing else drifted between the two runs); `su15` tied on levels (1-1) but
full stack more efficient (0.0061 vs 0.0051); `vc33` tied on levels (1-1)
but full stack markedly more efficient (1.75 vs 0.039, ~45x); `tu93` full
stack WON (level 1 vs 0); `sp80` bare control WON (level 1 vs 0, the ONE
game where the richer stack's exploration lost ground it could have banked
cheaply). Totals: `levels_gained_full_stack_total=4`,
`levels_gained_bare_control_total=4` (tied), `efficiency_full_stack_total=
4.5384` vs `efficiency_bare_control_total=2.862` (full stack ahead, driven
mainly by `vc33`).

**Honest conclusion: on this broader, genuinely-headroom-bearing
roster/budget, the candidate-scoring stack does not out-level bare control
in aggregate (a tie, with one win and one loss cancelling), but it IS
measurably more action-efficient in aggregate, and the per-game spread shows
the ablation has real, game-dependent effects in both directions -- not a
uniform advantage and not a no-op.** This is a genuinely more informative
null-tending result than exp5592's (a single tied game gives no information
about direction; five games with a mixed win/loss/efficiency pattern gives a
real, if modest and non-uniform, signal). Still not sufficient grounds to
cite the stack as an unqualified moat (`sp80`'s loss is a real
counter-example), but the efficiency edge across `su15`/`vc33` and the
`tu93` level win are real, citable, non-floor-effect findings.
`adversarial_verify.py` clean.

Required field principles (in addition to REQ-ARC-FCP-5592's):

- `n_games_with_headroom`: principle "count of games where EITHER arm reached level>=1 -- the direct fix for exp5592's single-game floor effect; more non-zero cells means the arm comparison rests on genuine statistical footing, not one game's tie."
- `roster_source`: principle "documents WHY this roster differs from exp5592's -- every game here has a registered GameAdapter, so the generic policy has a structural chance to progress, unlike exp5592's un-adaptered games which were floor-effect noise, not signal."
- `prior_attempt`: principle "CLAUDE.md Failed-Experiment Rerun Discipline -- names the exp5592 floor-effect finding and what is different here (roster restricted to adaptered games, budget raised 200->500) so this is a documented root-cause fix, not a doomed re-run."

#### SCENARIO-ARC-FCP-5701-BROADER-HEADROOM

Given the same candidate-scoring-stack ablation as REQ-ARC-FCP-5592 but run
on the full adaptered-game roster at a calibrated higher budget
When both arms complete across all games
Then `n_games_with_headroom` is materially greater than exp5592's single
game, so the resulting `honest_verdict` (tie, win, loss, or efficiency-only
edge) rests on multiple independent games' worth of real signal rather than
one game's floor-effect tie

### REQ-ARC-FCP-5703: Mechanism-Level Diagnosis of the sp80 Candidate-Scoring-Stack Regression

exp5701 found `sp80` was the one game (of 5 with measured headroom) where
`bare_control` beat `full_stack` by a level. Restating the number is not a
diagnosis -- `python/carnot/experiment_5703_sp80_candidate_stack_mechanism_trace.py`
SHALL directly instrument all three mechanisms that differ between the arms
(`candidate_router`, `goal_bias`, `goal_candidate_guidance`) during a real
sp80 replay of both arms, to determine whether any of them ACTIVELY steered
`full_stack` toward a worse choice, or whether the regression's cause lies
elsewhere in the stack.

The artifact SHALL report, per mechanism: whether it was present, how many
times it was genuinely invoked, and whether it ever changed the outcome it
influences (`candidate_router_changed_order_count` for candidate reordering;
`goal_bias_score_variance` across every real frontier-node scoring call for
the goal-energy bias). `inert_mechanisms` SHALL list any mechanism confirmed
structurally incapable of having caused the regression this run.

**RESOLUTION (2026-07-14).** Replayed sp80 under both arms, budget=500,
offline (no LLM). Regression reproduced (`full_stack.levels_gained=0` vs
`bare_control.levels_gained=1`, `bare_control` reaching L1 in 425 actions).
**All three learned mechanisms were confirmed structurally inert this run:**
`goal_bias` (`arc_goal_energy_live.GoalSatisfactionEnergy`, source
`exp4020_graded_goal_satisfaction_energy`) scored EXACTLY `1.0` on all 771
real frontier-node invocations (`goal_bias_score_variance=0.0`) -- a
mathematical proof it could not have influenced the A*-style frontier
ordering, since a constant score maps to a constant sort key
(`_goal_bias_key`). `goal_candidate_guidance` (the same energy source
applied to the immediate candidate pool) also scored uniformly and
correctly self-detected its own degeneracy (`arms_non_degenerate=False`)
via its existing audit, falling back to the unranked candidate order by
design. `candidate_router` was genuinely invoked 33 times but never once
changed the candidate ordering it was given
(`candidate_router_changed_order_count=0`).

**Honest conclusion: the sp80 regression is NOT caused by a bad learned
signal actively misleading search -- it is structurally impossible for
these three mechanisms to be the cause here, since two contribute a
provable no-op and the third self-audited its own uselessness and
correctly disengaged.** The real cause must trace to one of the other
differing knobs (`value_weight`/DAgger value head, `navigation_cost_
tiebreak`, `action_effect_expansion_prior`) -- not further isolated in
this investigation's scope. Separately, this run surfaced a genuine,
independently-useful finding: `GoalSatisfactionEnergy`'s frame-state
extraction is structurally blind on sp80 (falls to its constant-1.0
default because `visible_state()` cannot parse sp80's placement mechanic
into a target-fraction), corroborating `ops/verifier_gaps.md` GAP-4891's
independent finding (via a completely different code path -- the offline
self-induction operator, not the live goal-bias stack) that sp80's goal is
spatial/placement and not discriminable by the count/generic-fraction
features these mechanisms use. Logged as GAP-5703 (`ops/verifier_gaps.md`)
per the Missing-Verifier Gap Logging discipline, with a concrete,
general-purpose fix recommendation: give `goal_bias`'s frontier scoring the
same degenerate-score self-audit `goal_candidate_guidance` already has.

Required field principles:

- `goal_bias_score_variance`: principle "zero variance across every real invocation is direct, mechanical proof the goal-energy source could not have influenced frontier ordering on this game, independent of any post-hoc narrative."
- `candidate_router_changed_order_count`: principle "counts how many of the router's real invocations actually altered the candidate ordering it was given -- distinguishes 'present and consulted' from 'present and load-bearing'."
- `prior_result`: principle "CLAUDE.md Failed-Experiment Rerun Discipline analog for a diagnostic task -- names the exp5701 finding this investigates so the connection is traceable."

#### SCENARIO-ARC-FCP-5703-MECHANISM-INERT-OR-IMPLICATED

Given a real replay of the sp80 regression with all three learned
candidate-scoring mechanisms instrumented for actual influence (not just
presence)
When the regression reproduces
Then `inert_mechanisms` correctly distinguishes mechanisms that were merely
present from mechanisms that measurably changed a decision, so the
diagnosis names what did NOT cause the regression with the same rigor as
what might have

### REQ-ARC-FCP-5703-2: `goal_bias` Degenerate-Score Self-Audit

GAP-5703's candidate design (a): `StepwiseExplorer.goal_bias_diagnostics()`
SHALL surface the same class of degenerate-score self-audit
`GoalEnergyCandidateGuidance` already has (`arms_non_degenerate`), so a
future investigator can see directly from the diagnostics dict that a
goal-energy source is contributing zero real signal on the current game --
without needing exp5703's manual instrumentation to find out.

Implementation: `StepwiseExplorer` tracks a STREAMING (not stored-list, to
avoid unbounded memory on long episodes) running sum/sum-of-squares/min/max
of every real `goal_bias(frame)` invocation in `_goal_bias_score`.
`goal_bias_diagnostics()` computes `score_variance` from the running
sum/sum-of-squares and reports `degenerate: bool` --
`nodes_scored >= 20 AND score_variance <= 1e-12` (the same variance floor
`GoalEnergyCandidateGuidance` uses, plus a minimum-sample floor to avoid a
false-positive read on a short episode that has not scored enough nodes
yet to say anything).

This is OBSERVABILITY ONLY: `degenerate=True` does NOT disable `goal_bias`
mid-episode or change search behavior. A constant score's `_goal_bias_key`
is already mathematically a no-op on ordering (a constant added to every
candidate's combined score does not change relative rank) -- confirmed
directly by exp5703's `candidate_router_changed_order_count=0` /
`goal_candidate_guidance`'s own `arms_non_degenerate=False` no-op finding
for the sibling mechanisms. Auto-disabling mid-episode would be a separate,
larger behavioral change (risk: a signal that is degenerate early in an
episode but would become informative later) that this fix deliberately
does not attempt.

**RESOLUTION (2026-07-14).** Shipped in `arc_competition_agent.py`
(`StepwiseExplorer.__init__`, `_goal_bias_score`, `goal_bias_diagnostics`).
Verified directly against the real sp80 case that motivated it: after 4
nodes scored, `degenerate=False` (correctly below the 20-sample floor,
guards against a false-positive read); after a full 500-budget episode (938
real nodes scored), `degenerate=True`, `score_variance=0.0`,
`score_min=score_max=1.0` -- exactly reproducing exp5703's manually-found
result, now available from `goal_bias_diagnostics()` directly without
custom instrumentation. Existing test (`test_experiment_4534_energy_trust_
next_level_routing.py`, which only asserted `lower_is_better`) still
passes unmodified.

#### SCENARIO-ARC-FCP-5703-2-DEGENERATE-SELF-AUDIT

Given a `StepwiseExplorer` with a `goal_bias` that returns a constant score
regardless of the frame it is given
When enough real nodes have been scored to clear the minimum-sample floor
Then `goal_bias_diagnostics()["degenerate"]` is `True` and
`score_variance` is at or below the variance floor, WITHOUT any change to
which candidate the explorer actually selects

### REQ-ARC-WMTE-5593-4: Real-World Pass-Rate Survey of the `min_heldout_accuracy=1.0` Dynamics Gate

REQ-ARC-WMTE-5593-3's live-integration follow-up found both arms failing at
the pre-existing dynamics gate before the goal-consistency veto was ever
reached in its first real attempt, and disclosed this as a
`dynamics_gate_finding` rather than investigating further.
`python/carnot/experiment_5702_dynamics_gate_pass_rate_survey.py` SHALL
follow up: aggregate every real `heldout_accuracy` value recorded across the
checked-in corpus of `inference_substrate == "live_llm_inference"` result
artifacts (excluding exp5700, which deliberately bypassed the gate for an
unrelated isolation test) to estimate how often a real induction round
actually clears the live call site's own threshold of `1.0`.

The artifact SHALL report `pass_rate_at_live_threshold`, `exact_zero_rate`,
`mean`/`median`, a `threshold_sweep` across common alternative bars, and
`cited_upstream_artifacts` (CLAUDE.md Inference-Substrate Declaration
Discipline audit trail -- every row must trace back to the real artifact
that measured it).

**RESOLUTION (2026-07-14).** Aggregated 95 real round-level
`heldout_accuracy` values across 12 real `live_llm_inference` artifacts.
**`pass_rate_at_live_threshold=0.1263`** (12.6% of real induction rounds
ever reach the exact `1.0` bar the live pipeline enforces).
`exact_zero_rate=0.4737` (47.4% of real rounds score a complete `0.0`).
`mean=0.3069`, `median=0.12` -- a strongly right-skewed, mostly-poor
distribution, not a narrow miss clustered just below the bar.

**Honest conclusion and limitation.** This measures the PER-ROW pass rate,
not the bounded 3-round retry loop's eventual within-budget success rate --
the checked-in corpus does not contain enough same-attempt multi-round
traces to reconstruct that distinct statistic (most artifacts in this
corpus measure first-shot induction quality, not a full bounded-retry
trace). The per-row rate is still the direct, real answer to "how strict is
`1.0` in practice": across a large, diverse, real historical corpus, the
overwhelming majority of individual real induction attempts miss the live
threshold entirely. This corroborates, with corpus-scale evidence, task 8's
single-attempt observation that the dynamics gate is "frequently the
dominant blocker" -- and raises a genuine calibration question (not
resolved here) of whether a graduated-trust tier, mirroring
`GoalEnergyCandidateGuidance`'s own degenerate-score self-audit pattern
(REQ-ARC-FCP-5703 above), could safely make use of a "good but imperfect"
induced model instead of an all-or-nothing accept/reject at `1.0`.

Required field principles:

- `pass_rate_at_live_threshold`: principle "the direct answer to task 11's question -- how often a real induction round clears the exact threshold (1.0) the live call site enforces."
- `excluded_artifacts`: principle "exp5700 deliberately set min_heldout_accuracy=0.0 to isolate an unrelated veto test; including its rows would understate how strict the REAL live threshold is by mixing in rows collected under a different, lower bar."

#### SCENARIO-ARC-WMTE-5593-4-CORPUS-PASS-RATE

Given the full checked-in corpus of real `live_llm_inference` artifacts,
excluding any artifact that deliberately used a non-default gate threshold
When every real round-level `heldout_accuracy` value is aggregated
Then the resulting `pass_rate_at_live_threshold` reflects only genuine
real-world attempts at the live pipeline's actual configured threshold, with
a full audit trail (`cited_upstream_artifacts`) back to the source artifacts

### REQ-ARC-WMTE-5593-5: Live A/B -- Does Relaxing `min_heldout_accuracy` Unlock Usable Plans?

REQ-ARC-WMTE-5593-4 raised, but did not resolve, a calibration question: does
a graduated-trust tier let the pipeline safely use a "good but imperfect"
induced model instead of discarding it outright at `1.0`?
`python/carnot/experiment_5704_dynamics_gate_relaxed_threshold_ab.py` SHALL
test this directly and LIVE: collect real transitions on a game with an
observed real level-up, then make N independent fresh real induction
attempts (`execute_bounded_llm_reinduction`, `max_rounds=1`, a fresh
proposer per attempt, `min_heldout_accuracy=0.0` so every attempt's real
`heldout_accuracy` is observed regardless of outcome) against the SAME real
transitions. For each attempt, record whether the STRICT threshold (`1.0`)
and a RELAXED threshold (`0.7`) would each accept it, and -- for any attempt
the relaxed threshold accepts but the strict one rejects -- whether the
resulting plan actually reaches the goal under in-model verification
(`plan_reaches_goal`).

This experiment does NOT presuppose relaxing the threshold helps. It SHALL
report one of three DISTINCT, non-interchangeable outcomes: (a) relaxing
unlocks one or more genuinely good plans the strict gate discards, (b)
relaxing accepts attempts but none produce a good plan (the strict gate is
correctly protective), or (c) no attempt lands in the relaxed-only band at
all (inconclusive -- the small live sample did not happen to sample the
interesting middle ground).

**RESOLUTION (2026-07-14).** Collected real transitions on `lp85`
(47 collected, 1 real level-up), then ran 3 independent fresh real
induction attempts (real GPU-backed `Qwen3.5-9B-MTP`,
`duration_s` per attempt: 794.3s / 94.2s / 89.8s -- the first attempt's
outlier duration is consistent with GPU/model cold-start overhead, not a
hang). **All 3 attempts scored `heldout_accuracy=0.0`** -- a complete
dynamics-check failure on every single real attempt, landing NONE in the
`[0.7, 1.0)` relaxed-only band this experiment was built to characterize.
`n_relaxed_only_accepts=0`, honest verdict
`complete: inconclusive_no_attempt_in_relaxed_only_band`.

**Honest conclusion: inconclusive by construction, and this null result is
itself consistent with REQ-ARC-WMTE-5593-4's corpus survey** -- `0.0` was
found to be the single most common real-world outcome there
(`exact_zero_rate=0.4737`, 47.4% of the historical corpus), so a 3-attempt
live sample landing entirely in that bucket, missing the much narrower
`[0.7, 1.0)` band (historically ~6.3 percentage points of the corpus:
`threshold_sweep["0.7"]=0.189` minus `pass_rate_at_live_threshold=0.1263`),
is unsurprising rather than anomalous. This experiment does not resolve the
graduated-trust calibration question either way; a larger N (costly -- each
real attempt is a genuine GPU-backed induction call, ~90-800s observed) or
a game/corpus selection biased toward the interesting band would be needed
for a conclusive answer, and is not pursued further in this task's scope.

**Orthogonal observation, RESOLVED via code analysis (2026-07-14, no further
live compute needed).** Despite `heldout_accuracy=0.0` on every attempt,
`planned=True` and `plan_reaches_goal=True` on all 3. Tracing
`_plan_reaches_goal` (`arc_llm_reinduction.py:452-501`) explains this fully:
it re-simulates the candidate plan by repeatedly calling the SAME induced
`engine` the held-out check just scored `0.0` (`arc_world_model_trust_
energy.py:select_trusted_world_model` -> `_score_accuracy` ->
`WorldModelVerifier(heldout).score(engine).accuracy`, an exact per-transition
grid match over a genuine 16-transition held-out split on lp85's 47
collected transitions, per `_split_prefix_heldout`'s 1/3 fraction) --
then checks the SAME induced `goal` predicate against that engine's own
simulated final grid. **This is a purely self-referential, internally-
consistent check with zero grounding against real transitions or the real
environment.** An engine that is completely wrong about real dynamics
(0/16 held-out transitions correct) can still "prove" its own plan correct
by simulating entirely within its own incorrect beliefs and checking an
equally ungrounded goal predicate against the result.

**This is NOT a sign the held-out check and planning usefulness are
misaligned metrics -- it is a direct demonstration of the EXACT failure
mode `min_heldout_accuracy=1.0` exists to prevent.** A plan that "reaches
the goal" only inside a self-consistent-but-wrong simulation carries no
evidence it would work against the real game; if executed for real against
an engine this wrong about held-out dynamics, the plan would have no
principled reason to succeed. This closes the open question this task
originally flagged: the finding is not a counter-argument for relaxing the
strict threshold, it is corroborating evidence FOR keeping it -- a low
`heldout_accuracy` and a passing `plan_reaches_goal` can coexist precisely
because the latter provides no independent check when the former is bad.

Required field principles:

- `n_relaxed_only_accepts`: principle "count of real attempts that would be accepted under the relaxed bar but rejected under the strict bar -- the direct evidence for whether the strict threshold discards recoverable attempts."
- `relaxed_only_accepts_with_good_plan`: principle "of the relaxed-only accepts, how many produced a plan that ALSO passed plan_reaches_goal (in-model verification) -- an accepted-but-useless model would not support loosening the gate."

#### SCENARIO-ARC-WMTE-5593-5-RELAXED-THRESHOLD-AB

Given N independent fresh real induction attempts against the SAME real
transitions, with the dynamics gate bypassed so every attempt's real
`heldout_accuracy` is observed
When the strict (1.0) and relaxed (0.7) thresholds are each applied to the
same attempts
Then the artifact reports exactly one of "relaxing unlocks a good plan",
"relaxing accepts but no good plan resulted", or "inconclusive -- no
attempt in the relaxed-only band", never conflating the three, and never
presupposing which outcome will occur before the real attempts run

### REQ-ARC-WMTE-5593: Goal-Predicate Consistency Against Real Observed Level-Progress

`ops/known-issues.md`'s 2026-07-11 task 11 entry (hallucination-consistency
checks) found two independent top-3 ARC-AGI-3 teams each carry an
unexploited self-report-vs-ground-truth gap: Reki's `board_change_assessment`
(what it thinks changed) is never cross-checked against `changed_pixels`
(the real pixel diff); Duck's free-text Goal/Action hypothesis is
regenerated each turn but never checked against observed level-up/no-change
transitions. Investigating this project's own architecture found no direct
analog to Reki's exact natural-language self-report, but found the DYNAMICS
half of the same gap-class was already closed
(`WorldModelVerifier.score(engine)` checks the induced `engine()`'s
predicted next-grid against the real observed next-grid) while the GOAL
half was genuinely open: nothing validated `is_level_complete` (the induced
code's formalized goal hypothesis, installed by
`execute_bounded_llm_reinduction` as a search termination condition) against
real observed level-progress ground truth.

`python/carnot/agentic/arc_executable_world_model.py` SHALL expose
`score_goal_predicate_consistency(is_level_complete, transitions) ->
GoalPredicateConsistency`, the goal-hypothesis sibling of
`WorldModelVerifier`/`VerifyResult`: for each transition, the real ground
truth is `level_after > level_before` (a genuine level-up occurred); the
claim under test is `is_level_complete(next_grid)`. Agreement is a cheap,
deterministic sign check -- no second LLM call, matching forge's own
competitive-pressure finding (docs/research-notes/arc-agi3-milestone1-
winners-sota-ingestion-2026-07-11.md, O3) that an expensive LLM judge was
not worth the cost while a deterministic filter was kept. The function
SHALL treat a raising predicate as a `claimed=False` miss rather than
propagating the exception, and SHALL return a well-formed zero result (no
`ZeroDivisionError`) for an empty transition list. The CALLER CONTRACT
(documented in the function's own docstring) is to pass transitions from a
SINGLE level boundary, since `is_level_complete` is a per-boundary predicate
in the real pipeline (freshly re-induced after every level-up).

This is a NEW, purely additive verification primitive -- it does not modify
`WorldModelVerifier`, `VerifyResult`, or any existing induction/planning
behavior, and is NOT yet wired into any live decision (e.g. vetoing a goal
predicate before planning); that is a distinct, separately-scoped design +
empirical-validation step, consistent with how the color-blob salience
topology extension (REQ-ARC-FCP-5591) was left additive-only pending its
own validation.

**RESOLUTION (2026-07-13).** The function itself is validated by 5 direct
unit tests on realistic (non-toy) synthetic data
(`tests/python/test_arc_goal_predicate_consistency.py`): a perfect
predictor scores 1.0; a predictor that never claims a win is CAUGHT missing
a real level-up (not silently trusted); a predictor that claims every state
is a win is CAUGHT false-positiving a no-op (both miscalibration
directions detected, not just one); a raising predicate is handled
gracefully; an empty transition list returns a well-formed zero result.

The offline-sim prototype (`experiment_5593_goal_predicate_consistency_
offline_sim_prototype.py`) attempted a REAL end-to-end test against `lp85`
(the only game with any measured headroom across the full 11-game roster in
this session's exp5590/exp5592 A/Bs) using the real default Qwen3.5-9B-MTP
proposer, and found a genuine, precisely-diagnosed limitation of the
EXISTING (pre-dating this task) `induce_prompt`/`LocalGGUFProposer`
machinery: `lp85`'s logical grid is 64x64 (much larger than typical), and
`induce_prompt`'s fixed "render the full initial grid" overhead alone
produces a prompt whose token count is already close to the induction
pipeline's available budget (`n_ctx=16384` minus `max_tokens=2560` reserved
for the completion = 13,824 tokens) -- confirmed by direct debugging
(bypassing `LocalGGUFProposer.induce()`'s own truncated error-repr handling
to read the llama-server's actual JSON error body): an 8-transition window
measured 18,355 prompt tokens (`exceed_context_size_error`), and even a
MINIMAL single-transition window (the level-up transition alone) measured
~13,400+ tokens -- already at the edge of the budget regardless of how few
transitions are used. This is a real, useful finding about the shared
induction pipeline's scalability on large-grid games, not a flaw in
`score_goal_predicate_consistency` itself (which never got the chance to
run against a real induced predicate on `lp85`, since induction itself
never produced one). Fixing `induce_prompt`'s large-grid scalability is
out of scope for this task; the checked-in artifact honestly records
`goal_predicate_accuracy: null` and
`induction_failure_detail` (the truncated exception repr
`LocalGGUFProposer.generate()` currently captures -- the fuller JSON error
body is not preserved by the existing error handling, a minor, separately
fixable gap not addressed here).

Required field principles:

- `real_levelup_present_in_sample`: principle "the check is only interpretable if the collected transitions include at least one genuine level-up -- otherwise the accuracy figure reflects only no-op agreement, which any always-False predictor would also score perfectly (CLAUDE.md FALSE_NEGATIVE_RISK discipline, applied to this new consistency check)."
- `goal_predicate_mismatches`: principle "the specific transitions where the induced is_level_complete disagreed with real observed level-progress, for honest post-hoc inspection."

#### SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR

Given a goal predicate whose sign correctly matches every real observed
level-up and no-op transition in a single level boundary's transition
window
When `score_goal_predicate_consistency` scores it
Then `accuracy` is 1.0 and `mismatches` is empty

#### SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT

Given a goal predicate that never claims a win (or, in the opposite
direction, claims every state is a win)
When `score_goal_predicate_consistency` scores it against a transition
window containing at least one real level-up and one real no-op
Then the predicate's disagreement with the real observed transitions is
recorded in `mismatches`, not silently scored as a correct prediction

### REQ-ARC-WMTE-5593-2: `induce_prompt` Large-Grid Scalability Fix -- Real Positive-Control Demo

REQ-ARC-WMTE-5593's own RESOLUTION left the `induce_prompt` large-grid-scalability limitation
explicitly out of scope, and `ops/known-issues.md` task 11's DONE note named it as the natural
prerequisite for a real positive-control demonstration of `score_goal_predicate_consistency` on
`lp85` specifically. This requirement closes that gap.

**Root cause, precisely measured.** `_transitions_block`'s two full-grid renders (the INITIAL
grid and, when a level-up occurred in the window, the WIN STATE grid) used `to_ascii` --
one raw character per cell. On `lp85`'s 64x64 grid this is exactly correct at the character
level (~4,160 chars/grid including newlines) but catastrophically token-inefficient: a
single-transition window (up to 2 full grids) already measured ~13,400+ tokens against the
13,824-token available budget, and an 8-transition window measured 18,355 tokens
(`exceed_context_size_error`, 32.8% over budget).

**Fix, two parts, both real-data-measured (not estimated):**

1. `_rle_grid(g) -> str` -- a NEW function, run-length-encodes a FULL grid one line per row,
   `r<row>:<v0>x<n0>,<v1>x<n1>,...`, with the starting column of each run left IMPLICIT (the
   running sum of prior counts in that row) since every cell in a row is covered with no gaps.
   An earlier attempt that spelled out an explicit column per run (matching `_rle_delta`'s own
   style) measured WORSE than `to_ascii` on `lp85`'s real grid for its WIN-state render (5,164
   vs 4,159 chars) -- the per-run column overhead dominated for `lp85`'s actual run-length
   distribution (avg run length ~11.6 cells, 352 runs on the measured INITIAL grid). Dropping the
   column entirely removed that overhead: measured on the SAME real `lp85` grid, INITIAL
   4,159->1,857 chars (2.24x), WIN 4,159->2,368 chars (1.76x), both verified lossless
   round-trip.
2. `_rle_delta_compact(g0, g1) -> str` -- a NEW function (kept separate from the existing
   `_rle_delta`, which has its own round-trip tests and another caller and is NOT modified here).
   After fix (1) above, the per-transition DELTAS became the new dominant cost on `lp85`'s real
   transitions (measured: 8 deltas via `_rle_delta` = 9,308 tokens, still over budget even after
   the full-grid fix). `_rle_delta_compact` sub-compresses each changed run's NEW values as
   `<value>x<count>` pairs instead of listing one value per changed cell -- large changes are
   often a single-color object moving or appearing, which the raw comma-per-cell format cannot
   exploit. Measured on the SAME 8 real deltas: 9,308 -> 5,992 tokens (a 3,316-token additional
   saving), verified lossless round-trip against 200 random synthetic diffs.

`induce_prompt`'s own explanatory text (the part of the prompt teaching the model how to decode
the compact forms) SHALL be updated to describe both new formats precisely enough for the model
to reconstruct a grid/delta from them -- this is a prompt-CONTRACT change, not just an internal
encoding change, since the model must correctly parse the new notation to reason about the game.

**RESOLUTION (2026-07-14).** Real, tokenizer-measured result on `lp85`'s actual 8-transition
window (via `llama_cpp.Llama(vocab_only=True)` against the real `Qwen3.5-9B-MTP` GGUF, the
exact model `induce_prompt` targets): the SAME window that measured 18,355 tokens before this fix
now measures 11,167 tokens against the 13,824-token budget -- comfortably under budget with
~2,657 tokens of real headroom (a ~39% total token reduction). Re-running
`experiment_5593_goal_predicate_consistency_offline_sim_prototype.py` end-to-end (real GPU
inference against the real port-8920 `Qwen3.5-9B-MTP` server, `duration_s=33.452`,
`inference_substrate=live_llm_inference`) now produces the real positive-control demo REQ-ARC-
WMTE-5593 could not: `induction_ok=true` (no context overflow), `induce_transition_count=8` (the
exact target window size), `real_levelup_present_in_sample=true` (interpretable per
FALSE_NEGATIVE_RISK), and `score_goal_predicate_consistency` scores the REAL induced
`is_level_complete` against the 8 real transitions: `goal_predicate_accuracy=0.75` (6/8 correct,
2 false-negative mismatches at transition indices 6 and 7 -- both real level-ups the induced
predicate missed). `honest_verdict:
"complete: goal_predicate_consistency_prototype_induced_predicate_miscalibrated"` -- the
CHECK works correctly on real data (that is this requirement's job); the induced predicate
itself being imperfect is a separate, honest finding about induction QUALITY on `lp85`
specifically, not a defect in the check or the scalability fix.

Existing tests unaffected (`test_rle_delta_lossless.py`'s 3 tests, `test_arc_goal_predicate_
consistency.py`'s 5 tests, `test_experiment_5593_...py`'s 7 tests all still pass unmodified --
`_rle_delta` itself was never touched). New tests:
`tests/python/test_arc_induce_prompt_large_grid_scalability.py`.

Required field principles: none new (this requirement modifies internal encoding functions and
`induce_prompt`'s prompt text; no new artifact schema fields).

#### SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP

Given a full grid or a changed-cell delta encoded via `_rle_grid` / `_rle_delta_compact`
When the encoding is decoded back into a grid using the exact reconstruction rule stated in
`induce_prompt`'s own explanatory text (implicit column for full grids; explicit run-start
column + implicit sub-run column for deltas)
Then the reconstructed grid is byte-identical to the original for both real `lp85` grids and
randomized synthetic grids/diffs across multiple colors (0-15) and shapes

#### SCENARIO-ARC-WMTE-5593-2-REAL-BUDGET-FIT

Given `lp85`'s real 64x64 grid and a real 8-transition induction window selected the same way
`experiment_5593_...py` selects it (through the first real level-up plus one)
When `induce_prompt` renders the window and the result is tokenized with the real
`Qwen3.5-9B-MTP` tokenizer
Then the token count is below the 13,824-token available budget (`n_ctx=16384` minus
`max_tokens=2560`), where it previously measured 18,355 tokens and overflowed with
`exceed_context_size_error`

### REQ-ARC-WMTE-5593-3: Wire `score_goal_predicate_consistency` Into a Live Veto

REQ-ARC-WMTE-5593 built `score_goal_predicate_consistency` and left it explicitly
additive-only: "NOT yet wired into any live decision (e.g. vetoing a goal predicate
before planning); that is a distinct, separately-scoped design + empirical-validation
step." Its own docstring names the exact gap: `execute_bounded_llm_reinduction`
"installs `outcome.goal_predicate` as a search termination condition on the strength of
the proposer's own code, unchecked against any observed transition." This requirement
closes that gap.

**Prerequisite bug found and fixed first.** `score_goal_predicate_consistency`'s core
logic (`real_levelup = t.level_after > t.level_before`) depends on `Transition.
level_before` being correct. Investigation found the LIVE path's `Transition`
construction sites in `E3AgentPolicy` (`arc_competition_agent.py`, `next_move` and
`_remember_active_probe_origin`) hardcoded `level_before=0` unconditionally -- wiring
the consistency check against raw live `transitions` without fixing this would have
judged every predicate against systematically wrong ground truth once the real level
ever exceeded 0. Fixed by adding `self._prev_level` (mirroring the
`previous_best_level` pattern already used correctly by `AutonomousExplorer._ingest`):
captured alongside every real `self._prev` assignment, consumed as the `Transition`'s
`level_before` instead of the hardcoded `0`. Verified safe via a ~280-test regression
sweep across every `E3AgentPolicy`-touching test file; the one hardcoded-registry-
snapshot test that failed (`test_experiment_5176_...`) was confirmed pre-existing
registry drift unrelated to this change (`ops/arc_solve_registry.yaml`'s `cd82` has
legitimately advanced past the test's stale hardcoded expectation).

**Safety precondition confirmed before wiring.** `is_level_complete` is used ONLY as
`plan_in_model`'s in-model BFS terminal condition -- a pure simulation-internal search
hint. The live agent's REAL win-recognition (`E3AgentPolicy._current_goal_reached()`,
comparing `self.explorer.best_level` against the real environment's own level signal
via `_level_of`/`_levels_completed`) is entirely independent of the induced predicate.
A veto is therefore low-risk: worst case, `plan_in_model` fails to find a plan that
round and the phase machine falls back to `self.explorer.next_move(...)` (real
exploration) -- the agent's actual level-progress detection is untouched either way.

**Design, mirroring the existing dynamics-veto pattern rather than inventing a new
one.** `execute_bounded_llm_reinduction` gains a new `min_goal_predicate_consistency:
float = 0.0` parameter (default OFF, matching `min_heldout_accuracy`'s own default).
When `> 0.0`, right after `goal_check`/`_repair_degenerate_goal` confirms the predicate
is SATISFIABLE (not degenerate) and BEFORE `plan_in_model` is called,
`score_goal_predicate_consistency(selected_goal, transitions)` scores it against the
SAME `transitions` the dynamics check already uses. The veto fires ONLY when BOTH (a)
`accuracy < threshold` AND (b) `n_real_levelups >= 1` (CLAUDE.md FALSE_NEGATIVE_RISK
discipline -- a window with zero real level-ups makes any predicate, including a
constant-False stub, trivially score 1.0, so judging on that data would be
uninformative, not merely lenient). On veto, a `goal_predicate_consistency_failed`
counterexample (accuracy, threshold, and per-transition mismatches) is attached the
same way `heldout_transition_verification_failed` attaches dynamics mismatches, the
round is marked `skipped`, and the loop `continue`s to the next round -- `refactor()`
receives the counterexample via the EXISTING generic `_counterexample_result` fallback
path (not the DYNAMICS-specific BEFORE/PREDICTED/OBSERVED shape, since a goal mismatch
isn't a grid-prediction mismatch; `_counterexample_result`'s own docstring names this
exact fallback "safe for any caller that has not attached real evidence").

**RESOLUTION (2026-07-14).** The live call site (`arc_competition_agent.py`'s
`execute_bounded_llm_reinduction(...)` invocation, `reason == "level_up_reinduction"`)
now passes `min_goal_predicate_consistency=1.0` -- mirroring that SAME call site's own
existing `min_heldout_accuracy=1.0`, the established risk tolerance for this specific
gate (strict acceptance, trust the bounded 3-round refinement loop to repair a failure,
safe fallback to real exploration if it cannot). This is a GENUINELY LIVE wire, not
inert plumbing: a goal predicate that disagrees with a real observed level-up during
live re-induction will now be rejected and trigger a refactor attempt, rather than
being installed unchecked. 4 new tests
(`tests/python/test_arc_goal_predicate_live_veto.py`) cover the veto firing on a real
mismatch, the opt-in default (disabled, backward-compatible with every existing
caller), and the false-negative-risk guard. Full ~280-test regression sweep across
every `E3AgentPolicy`/`execute_bounded_llm_reinduction`-touching test file passed
(one pre-existing, unrelated `Memory leak` teardown-watchdog false-positive and one
pre-existing, unrelated registry-drift failure, both confirmed present on the
pre-change tree too).

#### SCENARIO-ARC-WMTE-5593-3-VETO-FIRES

Given a re-induced goal predicate that is satisfiable (not degenerate) in the induced
model, but disagrees with a real observed level-up in `transitions` (claims the
post-transition grid is not level-complete when a real level-up occurred there)
When `execute_bounded_llm_reinduction` runs with `min_goal_predicate_consistency` set
above the predicate's real accuracy
Then the round is skipped with `goal_predicate_consistency_failed` BEFORE
`plan_in_model` is ever called, the mismatch is attached as a counterexample fed to the
next round's `refactor()` call, and `planned` is `False` if no later round produces a
consistent, satisfiable, plan-reaching predicate within the round budget

#### SCENARIO-ARC-WMTE-5593-3-VETO-OPT-IN

Given the exact same mismatching goal predicate and transitions as above
When `min_goal_predicate_consistency` is left at its default (`0.0`)
Then the veto never fires, no `goal_predicate_consistency_*` fields appear on the round,
and planning proceeds exactly as it did before this requirement existed -- every
existing caller of `execute_bounded_llm_reinduction` that does not pass the new kwarg
is unaffected

#### SCENARIO-ARC-WMTE-5593-3-FALSE-NEGATIVE-RISK-GUARD

Given a `transitions` window containing ZERO real level-ups (every transition has
`level_after == level_before`) and a strict `min_goal_predicate_consistency` threshold
When `execute_bounded_llm_reinduction` scores the goal predicate against this window
Then the veto does NOT fire regardless of the predicate's raw accuracy score, because
`n_real_levelups == 0` makes the score structurally uninformative (any predicate,
including a constant-False stub, would score a trivial 1.0 on an all-no-op window) --
per CLAUDE.md's FALSE_NEGATIVE_RISK discipline, a null/negative judgment requires a
positive control the data does not provide here

**LIVE-INTEGRATION EMPIRICAL FOLLOW-UP (2026-07-14, outer-loop, operator-directed).** The
above was verified with fake proposers/engines only. This follow-up ran the SAME real code
path with a real proposer against real `lp85` transitions (real `E3AgentPolicy` episode,
budget=50, real GPU-backed `LocalGGUFProposer`, `Qwen3.5-9B-MTP`), to check whether
`min_goal_predicate_consistency=1.0` at the live call site's own risk tolerance helps or
hurts in practice.

*First real attempt (the live-configured strict dynamics gate).* Both a veto-on and a
veto-off arm, called directly with the live call site's own `min_heldout_accuracy=1.0`,
failed at ROUND 1 on the PRE-EXISTING dynamics gate (`heldout_transition_verification_
failed`, `heldout_accuracy` 0.6875 and 0.0 respectively -- neither reached the required
1.0) before the goal-consistency veto was ever reached. Honest finding: in practice, on
real first-shot LLM induction, the already-strict `min_heldout_accuracy=1.0` dynamics
gate is frequently the dominant blocker, so this new veto's real-world marginal impact is
SUBORDINATE to that pre-existing gate more often than not -- it is checked LAST, after
goal-satisfiability, which is itself after dynamics acceptance.

*Second, isolated attempt (dynamics gate bypassed to test the veto specifically).* With
`min_heldout_accuracy=0.0` (bypassing the dynamics gate so the goal-consistency veto gets
a genuine chance to run) and a fresh proposer per arm (avoiding a real, separately-noted
LLM-proposer-connection-reuse hiccup found reusing one proposer object across sequential
`execute_bounded_llm_reinduction` calls -- an infra quirk worth a future look, not
investigated further here), on the SAME real transitions (47 collected, 1 real level-up):
the veto-ON arm's real induced predicate scored `goal_predicate_consistency_accuracy=
0.021277` (correct on only 1 of 47 real transitions) and was correctly rejected
(`skipped=goal_predicate_consistency_failed`, `planned=False`). The veto-OFF arm's
(independently induced, comparably poor `heldout_accuracy=0.0`) predicate was accepted
unchecked and produced `planned=True` (`plan_reaches_goal=True` -- but that check is
IN-MODEL, verified against the induced engine/goal's own simulation, not against real
environment ground truth). This is a direct, real confirmation of the exact failure mode
this requirement exists to catch: a badly-miscalibrated goal predicate installed as a
search-termination condition, believed successful by the agent's own internal check,
while actually almost entirely wrong about what a real win looks like.

**Honest caveat.** The two arms used SEPARATE, independent real induction calls (not the
literal same candidate under test twice), since isolating the goal-consistency check
required bypassing the dynamics gate cleanly per-arm -- not a perfectly matched pair. Both
arms' `heldout_accuracy=0.0` suggests comparable induction difficulty on this specific
transition window, which is why the comparison is still informative, but this is
disclosed rather than overclaimed as a controlled trial.

**Conclusion: no threshold adjustment.** This evidence supports KEEPING
`min_goal_predicate_consistency=1.0` at the live call site as originally set (mirroring
`min_heldout_accuracy=1.0`'s own established risk tolerance) -- a real, badly-wrong
predicate was directly observed being caught, and no instance of the veto rejecting a
predicate that was ACTUALLY accurate was observed in any test (unit or live). The
dynamics-gate-dominance finding above is noted as a real property of the live pipeline's
layered gates, not a reason to loosen this specific threshold.

### REQ-ARC-WMTE-5594: `/think` vs `/no_think` Induction Quality A/B on the Frozen Live Generator

`ops/known-issues.md`'s 2026-07-11 task 7 entry (cheap, dev-side-only) was
filed after ARC Prize's public GPT-5.6 results showed reasoning-effort
scaling ARC-AGI-3 solve rate ~26x (Low->Max) versus only ~1.3x on
ARC-AGI-1 for the same model -- a domain-specific signal that induction
tasks (the thing GPT-5.6 gains the most from extra reasoning on) benefit
disproportionately more from think-mode than the field's typical single-
grid puzzle tasks do. The frozen live-submission generator
(`project_arc_live_generator` memory: Qwen3.5-9B-MTP, MTP + q8 KV +
`n_predict>=2048` + `/no_think`) was decided under June sprint time
pressure and never re-measured against this specific finding.

**Precondition (a) mechanism finding.** Before any A/B could run, this
task discovered that `LocalGGUFProposer`'s `no_think_prefix` instance
attribute (set to `"/no_think\n"` in the live default config) has **no
effect on real induction calls today**: `CARNOT_ARC_CODEONLY_INDUCE`
defaults ON (2026-06-25 operator directive), and codeonly mode's own
`_L2_CODEONLY_DIRECTIVE` module-level constant in
`arc_executable_world_model.py` hardcodes a literal `"/no_think\n"` as its
own first line, which wins in `generate()`'s
`if _codeonly: ... elif self.no_think_prefix: ...` branching (codeonly
always applies when `codeonly_eligible=True`, which every `induce()` call
passes). Testing `/think` fairly therefore requires a scoped module-level
monkeypatch of `_L2_CODEONLY_DIRECTIVE` (swap the leading `/no_think\n`
for `/think\n`), restored in a `finally` block after each induction call
-- not the dead `no_think_prefix` attribute. This is a real, previously
undocumented fact about the induction pipeline, independent of this task's
headline A/B result.

**Precondition (a) compatibility-check bug, found and fixed in-session.**
The first automated compatibility probe (`check_think_mode_compatibility`)
falsely reported `think_mode_compatible_with_mtp: False`, directly
contradicting two separate manual verifications (direct `/completion`
calls against the live port-8920 server) that `/think` mode produces
genuinely different, clearly-reasoning-shaped output. Root cause: the
check's tag test was a bare `"<think>" in think_content` substring check,
but the model's actual opening tag varies across calls -- one probe
emitted `<thinking>` (observed verbatim:
`'<thinking>\nThe user wants me to write a Python world-model
engine...'`), and the literal characters `<think>` are not a substring of
`<thinking>` (the closing `>` does not align). The check's length-ratio
fallback (`len(think) > 1.5 * len(no_think)`) was also too strict for a
short `n_predict=120` probe budget: one real probe measured 549 vs 403
chars (a genuine 36% delta), below the 1.5x (604.5-char) threshold. Fixed
by checking a tuple of known reasoning-tag prefixes
(`("<think>", "<thinking>", "<reasoning>")`) via `str.startswith` and
lowering the length-ratio fallback to 1.15x. Re-verified standalone after
the fix: `compatible=True`, `"think content starts with a reasoning tag
(527 vs 460 chars) -- compatible"` -- consistent with the manual
verification. This is a lesson for any future probe of LLM-emitted
reasoning-tag content: match a set of known tag spellings, not one
literal string, and do not rely solely on a length-ratio heuristic at a
short completion budget.

**RESOLUTION (2026-07-13).** With the compatibility check fixed, the real
4-attempt measurement (2 roster games -- `m0r0`, `sk48` -- x 2 arms) ran
against the live default Qwen3.5-9B-MTP proposer (161.6s total,
`inference_substrate: live_llm_inference`). Both arms induced successfully
on both games (4/4 induction_ok). Per-game `heldout_accuracy` (via the
existing `WorldModelVerifier`, reused unmodified from REQ-ARC-WMTE-4494):
`m0r0` no_think=0.5 vs think=0.0 (no_think better); `sk48` no_think=1.0 vs
think=1.0 (tie). Neither roster game's 10-transition window contained a
real level-up (`real_levelup_present_in_sample: false` on all four
attempts), so `score_goal_predicate_consistency` (REQ-ARC-WMTE-5593) was
never triggered and goal-predicate accuracy is not part of this result --
an honest scope limitation, not a gap papered over. `honest_verdict:
"complete: think_mode_ab_equal_success_no_think_higher_accuracy"`: equal
induction-success count (2 vs 2), but no_think's mean heldout accuracy
(0.75) exceeds think's (0.5) on this 2-game roster.

**What this does and does not show.** This is a real, informative,
NEGATIVE result for switching the frozen live stack to `/think` on
induction quality specifically -- on this small roster, `/think` mode
never wins outright and loses once. It does NOT settle the broader
GPT-5.6-style reasoning-effort-scaling question for the ARC-AGI-3 domain
in general: (1) the roster is 2 games, well below the CLAUDE.md sample-
size floor for any percentage-point claim; (2) `heldout_accuracy` scores
world-model INDUCTION quality only, not the fuller "actions-to-first-win
across a real solve" metric GPT-5.6's own comparison used, which this
task's scope explicitly excluded (reusing existing verifiers, not
extending `lb.run_game` into a full solve loop); (3) neither game's
sample included a real level-up, so the goal-predicate half of induction
quality is entirely unmeasured here. Per the task's own explicit
instruction and the ARC-AGI-3 Submission Sprint / November-Floor
disciplines' "the live stack is frozen" rule, this result does NOT change
the frozen live stack's `/no_think` setting -- it is reported as an
offline dev measurement requiring an explicit operator decision, and on
these numbers the honest recommendation is "no evidence yet to justify
unfreezing the stack for `/think`."

Required field principles:

- `think_mode_compatible_with_mtp`: principle "task 7 precondition (a),
  checked here rather than assumed -- if False, this experiment stops per
  the task's `blocked_think_mode_incompatible_with_mtp` instruction rather
  than proceeding on an untested assumption."
- `think_max_tokens`: principle "materially larger than
  `no_think_max_tokens` by design -- think mode needs completion budget
  for reasoning tokens before code; comparing truncated-mid-thought
  output to quick code would not be a fair test, and this asymmetry is
  disclosed, not hidden."

#### SCENARIO-ARC-WMTE-5594-INCOMPATIBLE-BLOCKS-CLEANLY

Given the live-server compatibility probe finds no reasoning-tag prefix
and no material length delta between `/think` and `/no_think` output
When `build_artifact` runs
Then `honest_verdict` is
`"complete: blocked_think_mode_incompatible_with_mtp"` and no induction
attempt is made on any roster game

#### SCENARIO-ARC-WMTE-5594-TAG-VARIANT-RECOGNIZED

Given the live-server compatibility probe's `/think`-arm response begins
with `<thinking>` rather than the literal `<think>`
When `check_think_mode_compatibility` evaluates the probe responses
Then it recognizes the `<thinking>` prefix as a reasoning-tag match and
reports `compatible=True`, not a false `no observable difference`
negative

### REQ-ARC-FCP-5595: InertClickSigPruner -- Dead-Signature Click Pruning

`ops/known-issues.md`'s 2026-07-11 task 9 entry ("New 2026-07-11, cheap,
reuses an existing code shape") asked for Reki's dead-signature click-pruning
mechanism -- track a clicked component's structural signature `(color, size,
is_rect, twin_count)`; if a click on that signature never changes the frame,
suppress future clicks on components sharing it -- built by extending the
`arc_hazard_pruner.HazardMovePruner` trust+specificity gating pattern rather
than Reki's own greedy K=2 threshold, which the audit that surfaced this
(`docs/research-notes/arc-perception-grounding-audit-2026-07-13.md`) flagged
as over-aggressive: two observations is a thin evidence floor, and a strict
"first effective observation ever = permanently sacred, otherwise K=2 kills
it" rule has no tolerance for a signature that is mostly inert but
occasionally does something.

**The implementation (`python/carnot/agentic/arc_inert_click_pruner.py`,
`InertClickSigPruner`).** Per structural signature, the pruner accumulates
observed `(obs, inert, leveled)` counts from the search's own clicks -- no
offline ground truth, transferring to any game the same way both sibling
pruners (`HazardMovePruner`, `RelationalMaskMovePruner`) do. It differs from
Reki's rule in two ways: the evidence floor is raised from 2 to
`min_observations` (default 4, matching `RelationalMaskMovePruner`'s own
explicitly-not-K=2 default), and a `min_specificity` threshold (default 0.9)
replaces literal-zero-tolerance -- a signature is pruned once its OBSERVED
inert fraction clears the bar, not the instant a single effective click is
seen. A signature that has ever produced a real level-up is permanently
sacred (never pruned, regardless of specificity), mirroring both sibling
pruners' hard binary veto for level-ups specifically.

The clicked component's signature is computed via `click_signature(blob,
blobs)`, a new free function built on `connected_color_blobs`/`blob_at_click`
(`arc_color_blob_salience`, REQ-ARC-FCP-5591). `blob_at_click` (REQ-ARC-FCP-
5595) is the free-function promotion of `ColorBlobSaliencePrior`'s existing
private `_blob_for_click` lookup, added purely additively so callers outside
`ColorBlobSaliencePrior` can reuse the exact same click-to-blob resolution.
`twin_count` is the number of OTHER blobs in the same frame sharing a blob's
`(color, pixel_count, is_rect)` triple, so evidence about one component
transfers to its structural twins even when they sit at different frame
positions -- confirmed directly by `test_click_signature_twin_count_matches_
shared_shape_only` and by the offline-sim prototype's cross-twin evidence
transfer in `test_prunes_signature_confidently_inert_after_evidence_floor`.

The pruner implements the SAME `should_prune(frame, label) -> bool` /
`observe(frame_before, label, frame_after, leveled_up)` protocol as both
sibling pruners, so it composes through the existing `arc_relational_mask_
pruner.CompositeMovePruner` and is consumable by `OfflineSolver` via the
identical `move_pruner=` constructor parameter both siblings already use --
this makes it live-path-reachable per the ARC Live-Path Reachability
Discipline without any new wiring code (`OfflineSolver` is one of the two
recognized live entrypoints). A separate `rank_candidates(frame, rows) ->
rows` method implements the identical gating logic in the filter-protocol
shape `StepwiseExplorer._candidates` (`arc_competition_agent.py`) already
composes with (matching `program_synthesis_filter`/`goal_candidate_
guidance`'s contract) -- at the time this task was originally scoped, tested
and ready but deliberately NOT wired into that live composition chain,
consistent with how the color-blob salience front-end (REQ-ARC-FCP-5591) was
left additive-only pending its own live-wiring decision (`ops/known-issues.md`
task #97, still open).

**WIRING FOLLOW-ON (2026-07-13, same day).** The gap above is closed.
`coerce_inert_click_pruner` (new, `arc_inert_click_pruner.py`, matching
`coerce_program_synthesis_filter`'s `None`/`False` -> no pruner, instance ->
passthrough, `True` -> construct-default shape) plugs an `InertClickSigPruner`
into both `StepwiseExplorer` (a new `inert_click_pruner` constructor param,
threaded through a `rank_candidates` call inside `_candidates` alongside
`program_synthesis_filter`/`goal_candidate_guidance`) and `E3AgentPolicy`
(same param name, passed through). The pruner also gets a real `observe()`
call from `_ingest`'s existing per-transition OBSERVE hook -- the same site
that already feeds `dense_curiosity`/`controllable_novelty_policy`/
`object_centric_proposal_policy`/`action_prior` -- so it accumulates evidence
from the search's OWN live clicks, matching `HazardMovePruner`'s own online
discipline; without this half, `rank_candidates` would be wired but
permanently a no-op (every signature stays "unproven" forever). Gated OFF by
default (`SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["inert_click_pruner_enabled"]`), matching every other
freshly-wired-but-unvalidated component in that file -- per the
`solve_rate_dropped` guardrail, flipping the default for the SCORED agent
needs its own matched-budget offline A/B (states/actions-expanded reduction,
zero regression in reproduced levels) first, not assumed safe just because
it is reachable. Verified with 8 new focused tests
(`tests/python/test_arc_inert_click_pruner_live_wiring.py`): the coercion
function's full branch set, `_candidates` calling and propagating
`rank_candidates`'s filtered result, a no-pruner no-op, a raising-pruner
non-fatal fallback (matching every sibling hook's try/except discipline),
`_ingest` feeding `observe()` the realized transition, a no-pruner no-op for
that hook too, default-off parity against `SUBMITTED_AGENT_CONFIG`, and
`E3AgentPolicy` opt-in threading. Ruff and mypy clean; 46 pre-existing
`arc_competition_agent.py`-adjacent tests (submitted-parity, program-synthesis
filter, E3 fidelity/named-tail gates, HUD mask, competition-agent adapter)
still pass unchanged.

**RESOLUTION (2026-07-13).** The pruner itself is validated by 7 direct unit
tests on realistic synthetic grids
(`tests/python/test_arc_inert_click_pruner.py`): confidently-inert signatures
are pruned only after clearing both the evidence floor and specificity bar;
evidence transfers between structural twins; below-floor and below-specificity
cases are conservatively NOT pruned; a signature that ever leveled up is
permanently sacred even after many subsequent inert observations; keyboard
actions and undecodable labels are safely ignored; `rank_candidates` drops
only the confidently-inert click rows from a mixed candidate list.

The offline-sim prototype (`experiment_5595_inert_click_sig_pruner_offline_
sim_prototype.py`) ran the pruner against REAL transitions collected from a
REAL `E3AgentPolicy`/`lb.run_game` exploration of `m0r0` (confirmed
click-heavy by direct probe before selection: 21 of 22 transitions were
action=6). GAME-SELECTION NOTE (found investigating, not assumed): a first
attempt to probe click-action prevalence via a bare `E3AgentPolicy(game,
explore_budget=6)` with no explicit `proposer=` stalled twice with near-zero
CPU growth over many minutes -- some default-constructed component the
exploration loop depends on appears to block. The script therefore always
constructs `E3AgentPolicy` with an explicit `LocalGGUFProposer`, matching
exp5594's proven-reliable pattern.

The real run (37 transitions collected, 32 clicks, 19.3s and 32.8s on two
runs) produced an honest, informative NULL: 12 distinct signatures were
tracked but ZERO cleared BOTH the `min_observations=4` floor and the
`min_specificity=0.9` bar (`honest_verdict:
"complete: inert_click_sig_pruner_prototype_ran_but_no_signature_cleared_
evidence_floor"`). This is not a flaw in the mechanism -- with 32 clicks
spread across 12 distinct signatures (an average of under 3 observations per
signature), most signatures simply did not accumulate enough repeated
evidence within this budget to clear the conservative gate. The gate is
DESIGNED to fail closed under sparse evidence (per the trust+specificity
discipline this task explicitly asked for, replacing Reki's more aggressive
K=2), so an honest null here is the expected behavior at this budget, not a
bug. A larger `total_budget` (more actions per game) or a roster with more
same-signature repeat clicks (e.g. a game with many decorative identical
sprites) is the natural follow-on if a positive pruning demonstration is
wanted; this task's "cheap" framing scoped the prototype to confirming the
mechanism runs correctly end-to-end on real data, which it does.

`inference_substrate` was corrected mid-task from an initial conservative
`live_llm_inference` guess to `offline_arcade_live_agent_runtime_self_
discovery_no_llm`: the real measured duration (19.3s) was far under the 60s
`live_llm_inference` floor, and `adversarial_verify.py` correctly flagged the
mismatch (`DURATION_TOO_SHORT`) before this artifact was accepted. A real
`LocalGGUFProposer` IS constructed and wired into `E3AgentPolicy`, but this
script never calls `induce()`/`generate()` on it, and the measured duration
confirms the exploration loop itself never invokes the LLM either -- the
`model_specs` GGUF entry is vestigial, matching the documented substrate's own
disclosed pattern for vestigial model strings.

Required field principles:

- `total_signatures_confidently_inert`: principle "count of distinct
  signatures that cleared BOTH the evidence floor AND the specificity bar
  with zero level-ups -- the load-bearing claim behind building the pruner
  at all, measured against real click data rather than synthetic grids."
- `inference_substrate`: principle "offline_arcade_live_agent_runtime_
  self_discovery_no_llm -- confirmed empirically (measured duration far
  under the live_llm_inference floor), not assumed."

#### SCENARIO-ARC-FCP-5595-SIGNATURE-CLASSIFIED-ON-REAL-DATA

Given real (frame_before, label, frame_after, leveled_up) transitions
collected from a real offline-arcade exploration run on a click-heavy game
When those transitions are fed through `InertClickSigPruner.observe`
Then the pruner's `stats()` reports a well-formed count of tracked
signatures and confidently-inert signatures (zero is an honest, valid
outcome when no signature clears the evidence floor within the budget, not
an error)

#### SCENARIO-ARC-FCP-5595-RANK-CANDIDATES-SANITY-CHECK

Given the same real transitions replayed as a candidate-row list against a
real collected frame
When `InertClickSigPruner.rank_candidates` filters that list
Then it runs without error and `rows_kept + rows_dropped == rows_in`, with
only rows whose click signature is confidently inert removed

#### SCENARIO-ARC-FCP-5595-LIVE-WIRING-CANDIDATES

Given a `StepwiseExplorer` with an `inert_click_pruner` configured
When `_candidates` builds the candidate-row list for a frame
Then it calls `inert_click_pruner.rank_candidates(frame, rows)` and returns
its (filtered) result, exactly like the `program_synthesis_filter`/
`goal_candidate_guidance` hooks it composes alongside; a raising pruner is
caught and never breaks candidate generation; no pruner configured is a
clean no-op

#### SCENARIO-ARC-FCP-5595-LIVE-WIRING-OBSERVE

Given a `StepwiseExplorer` with an `inert_click_pruner` configured and a
pending action recorded in `self.awaiting`
When `_ingest` processes the resulting frame
Then it calls `inert_click_pruner.observe(frame_before, label, frame_after,
leveled_up)` with the realized transition, from the same per-transition
OBSERVE hook that feeds `dense_curiosity`/`controllable_novelty_policy`/
`object_centric_proposal_policy`/`action_prior` -- without this, the pruner's
tally never accumulates evidence from live play and `rank_candidates` would
be wired but permanently inert

#### SCENARIO-ARC-FCP-5595-DEFAULT-OFF-PARITY

Given the SUBMITTED default configuration
When a `StepwiseExplorer` or `E3AgentPolicy` is constructed with no explicit
`inert_click_pruner` argument
Then `explorer.inert_click_pruner` is `None` (tracking
`SUBMITTED_INERT_CLICK_PRUNER_ENABLED = False`, mirrored in
`SUBMITTED_AGENT_CONFIG["inert_click_pruner_enabled"]`) -- the SCORED agent's
behavior is unchanged until a matched-budget offline A/B validates flipping
it on, per the `solve_rate_dropped` guardrail

### REQ-ARC-FCP-5595-2: Matched-Budget A/B -- The Flip-Decision Measurement

The `solve_rate_dropped` guardrail (REQ-ARC-FCP-5595-LIVE-WIRING-*'s own
default-off framing) names a matched-budget offline A/B (states/actions-
expanded reduction, zero regression in reproduced levels) as the precondition
for ever flipping `SUBMITTED_INERT_CLICK_PRUNER_ENABLED` to `True`. This
requirement runs that measurement, mirroring `HazardMovePruner`'s own tu93 A/B
precedent exactly: `scripts/arc_loop_solve.solve_adaptered` (extended with a
new `inert_click_prune: bool = False` parameter, composing
`InertClickSigPruner` into the same `move_pruner` construction `hazard_prune`/
`mask_prune` already use) SHALL be run twice at the SAME `--target-level` with
`hazard_prune`/`mask_prune` held fixed at `False` in both arms, varying ONLY
`inert_click_prune`, using the pruner's real, already-validated default
parameters. `OfflineSolver.last_states_expanded` SHALL be the efficiency
metric; `arc_solver_kit.reproduce` (the offline reproduction gate) SHALL be
the correctness backstop for both arms -- a states_expanded reduction only
counts as a genuine win if both arms reach the same target level and both
pass `offline_reproduced=True`.

Because `OfflineSolver`'s directed, verifier-guided search may never exercise
repeated inert clicks the way broad live exploration does, this requirement
ALSO runs an independent live-wired supplementary check: a real
`E3AgentPolicy` construction with `inert_click_pruner=True` (matching
exp5595's own real-game construction), reporting the pruner's own `stats()`
after a real exploration run -- confirming (or refuting) engagement
independent of the `OfflineSolver` harness.

### SCENARIO-ARC-FCP-5595-2-MATCHED-BUDGET-AB: Honest Verdict Taxonomy

Given the baseline (`inert_click_prune=False`) and treatment
(`inert_click_prune=True`) `OfflineSolver` runs plus the live-wired
supplementary check
When both arms pass the reproduction gate and reach the same target level
Then the verdict reports either a genuine `states_expanded` reduction (if
either the offline treatment arm or the live-wired check actually pruned
anything) or an honest no-op (if neither did) -- a zero-reduction result is
valid and informative at the tested budget, not a failure requiring
escalation; a failed reproduction gate or a lower reached level on the
treatment arm is classified as a regression regardless of any states_expanded
number, overriding the efficiency signal with the correctness backstop

### REQ-ARC-FCP-5609: Reachability-Controlled ARC Filter Intermediate-Invariance A/B

Experiment 5609 SHALL write
`results/experiment_5609_arc_filter_intermediate_invariance_ab.json` as the
decision-grade matched-budget A/B for the already-wired inert-click filter
(REQ-ARC-FCP-5595) and object-history salience filter (REQ-ARC-FCP-5591-2).
The experiment SHALL run a registry-prechecked roster of at least three
click/change-diverse games, excluding duplicate solve targets, without reading
game source, running exhaustive offline BFS, creating a per-game adapter, or
claiming a new solve. The measurement substrate is the existing offline
arcade live-agent runtime with no new LLM calls:
`inference_substrate=offline_arcade_live_agent_runtime_filters_no_new_llm`.

Before the outcome A/B, the experiment SHALL run fixed, non-source-aware
runtime reachability controls. The inert-click control SHALL demonstrate at
least one click signature clearing the shipped evidence floor
(`min_observations=4`, `min_specificity=0.9`). The object-history control
SHALL demonstrate at least one same-base candidate pair whose ordering changes
only after object-history evidence is observed. If either control is null, the
artifact SHALL report that mechanism as unreachable and SHALL NOT tune
thresholds on outcome data.

The outcome A/B SHALL run baseline, inert-only, history-only, and combined
arms on identical games, seeds, action budgets, proposer availability, target
levels, and stopping rules while keeping the frozen live generator unchanged.
Each arm SHALL report proposed candidates, pruned/reranked candidates,
environment actions, distinct states, nodes expanded, level gains,
actions-to-level, wall time, and exact offline reproduction receipts.
Candidate-count reduction alone SHALL NOT promote a mechanism.

Promotion SHALL require same-or-better reproduced level in every treatment, no
safety regression, and a preregistered improvement in at least one downstream
intermediate with paired uncertainty. If a reachable mechanism again produces
the same no-op on downstream live-path work, the artifact SHALL retire that
mechanism separately rather than aggregate it into a combined verdict.

Required field principles:

- `field_principles`: principle "principle annotations are carried in the artifact so the verifier can audit why every required 5609 field exists."
- `registry_precheck`: principle "duplicate solve targets are excluded and roster selection is auditable from registry/public environment metadata, not from game source."
- `roster`: principle "scope is auditable; at least three click/change-diverse games are measured under identical arms."
- `mechanism_reachability_controls`: principle "a null is interpretable because each mechanism first proves its own shipped hook can fire on runtime frames."
- `arm_configs`: principle "variables are isolated: only inert_click_pruner and object_history_salience vary across the four arms."
- `matched_budget_receipt`: principle "comparisons are fair: games, seeds, action budgets, proposer availability, target levels, and stopping rules are identical."
- `candidate_counts_by_arm`: principle "direct filter action is visible before downstream metrics are interpreted."
- `environment_actions_by_arm`: principle "filters must affect the live path, not only an internal candidate list."
- `distinct_states_by_arm`: principle "cosmetic candidate collapse is separated from real state-space change."
- `nodes_expanded_by_arm`: principle "search work is measured independently of candidate counts."
- `levels_gained_by_arm`: principle "the north-star outcome remains visible even though this is a development proxy."
- `wall_time_by_arm`: principle "runtime overhead cannot be hidden by reporting only search counts."
- `paired_effects_and_intervals`: principle "uncertainty controls promotion; no single aggregate delta can promote a filter."
- `filter_promotion_decisions`: principle "each mechanism is decided separately so a combined arm cannot hide one mechanism's repeat no-op."
- `solve_provenance`: principle "development_proxy -- public-game measurement receives no new-level credit."
- `offline_reproduced`: principle "known-level safety is exact and never inferred from level counters alone."
- `inference_substrate`: principle "offline_arcade_live_agent_runtime_filters_no_new_llm -- current reachable code is measured with no new LLM calls."
- `honest_verdict`: principle "repeat reachable no-op retires the corresponding mechanism instead of re-running unconstrained prototypes."

#### SCENARIO-ARC-FCP-5609-REACHABILITY-GATES-BLOCK-OUTCOME-TUNING

Given the fixed inert-click and object-history runtime reachability controls
When either control fails to demonstrate its required shipped mechanism signal
Then the artifact reports the failed mechanism as unreachable, does not tune
thresholds on outcome data, and records `filter_promotion_decisions` as
blocked/retired for that mechanism instead of promoting from candidate-count
movement alone.

#### SCENARIO-ARC-FCP-5609-MATCHED-BUDGET-ARM-ISOLATION

Given the baseline, inert-only, history-only, and combined arms
When the A/B runner builds `arm_configs` and `matched_budget_receipt`
Then every arm uses the same roster, seeds, action budget, target levels,
proposer availability, and stopping rule, and differs only in
`inert_click_pruner` and `object_history_salience`.

#### SCENARIO-ARC-FCP-5609-DOWNSTREAM-PROMOTION-GATE

Given paired per-game metrics for a reachable mechanism
When candidate counts improve but environment actions, distinct states, nodes
expanded, levels gained, and offline reproduction safety do not improve
Then the mechanism is not promoted; if this repeats the prior no-op after a
reachable control, the artifact retires that mechanism with an honest terminal
verdict.

### REQ-ARC-FCP-5610: Unconditional Live Self-Discovery Level-Up Attempt

Experiment 5610 SHALL write
`results/experiment_5610_arc_live_self_discovery_levelup_v506.json` after one
unconditional live-agent self-discovery attempt against a non-duplicate ARC
frontier level. The experiment SHALL first run a registry precheck across all
public offline-arcade games, excluding target levels already present in
`scripts/arc_loop_solve.py` outputs, `ops/arc_solve_registry.yaml`, the
previous milestone ARC artifact, and any already-recorded current-milestone
attempt. The selected target SHALL rotate toward a game with authenticated
public environment headroom and an adjacent next level beyond the registry
depth.

The live attempt SHALL use the live ARC runtime's own observations, attempted
actions, state transitions, and runtime reverse-engineering signals. It SHALL
NOT inspect game source, run exhaustive offline ground-truth BFS, inject a
hand-built per-game adapter, or replay a hidden solution recipe. Exp5609 is
advisory only: if it promoted a filter without safety regression, Exp5610 MAY
enable only that promoted configuration; otherwise Exp5610 SHALL run the
unchanged current no-LLM live-agent baseline. Exp5609 SHALL NOT gate or skip the
attempt.

The action budget, seed, target, stopping rule, filter configuration, and frozen
generator choice SHALL be fixed before the attempt. If the live path invokes no
LLM, the artifact SHALL declare
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`llm_invoked=false`, and `no_model_specs_required=true` instead of inventing a
model receipt. If a candidate level at or beyond the selected target is reached,
the experiment SHALL reproduce that exact live-discovered trace through the
generic offline reproduction path before banking. A level SHALL count only when
`offline_reproduced=true`, the reached level exceeds the registry precheck, and
the action trace checksum replays exactly.

Required field principles:

- `field_principles`: principle "principle annotations are carried in the artifact so every required 5610 field is auditable."
- `registry_precheck`: principle "duplicate levels receive no credit; all public games, registry depths, arc_loop_solve depths, previous milestone targets, and current milestone attempts are checked before target selection."
- `target_selection_receipt`: principle "rotation and authenticated public-game headroom are explicit, so the selected next level is not a duplicate."
- `live_attempt_executed`: principle "bare bool true proves the ARC standing floor was a real runtime attempt, not an advisory precheck."
- `filter_configuration`: principle "Exp5609 promotion use is auditable and cannot gate or skip the level-up attempt."
- `action_budget`: principle "search cost is bounded before runtime begins."
- `attempt_trace_path`: principle "discovery evidence is replayable from a durable trace."
- `levels_before`: principle "authoritative registry total before the attempt; the north-star delta is exact."
- `levels_after`: principle "authoritative registry total after accepted banking; unchanged on honest nulls."
- `new_reproducible_levels`: principle "only newly reproduced levels beyond the precheck depth count."
- `offline_reproduced`: principle "a live reach needs independent replay; duplicate or unreplayed reaches do not bank."
- `registry_updated`: principle "successful evidence becomes durable, while null attempts leave the registry unchanged."
- `solve_provenance`: principle "must equal live_agent_self_discovery for any credited path."
- `source_files_read`: principle "must be false; outer-loop source reverse engineering is excluded."
- `per_game_adapter_used`: principle "must be false; hidden per-game solvers are not smuggled into live self-discovery credit."
- `inference_substrate`: principle "offline_arcade_live_agent_runtime_self_discovery_no_llm when no LLM call is made."
- `honest_verdict`: principle "no-new-level is terminal; a failed Exp5609 filter A/B is not permission to skip the attempt."

#### SCENARIO-ARC-FCP-5610-PRECHECK-ROTATES-NON-DUPLICATE-HEADROOM

Given public environment metadata, registry depths, arc_loop_solve outputs, and
the previous milestone ARC artifact
When Exp5610 runs its registry precheck
Then it selects the first rotated target whose next level is beyond both the
registry and arc_loop_solve depths, has authenticated public headroom, and is
not the previous or current milestone's duplicate target.

#### SCENARIO-ARC-FCP-5610-FILTER-ADVISORY-NOT-GATING

Given Exp5609 either retired its filters, blocked a mechanism, or promoted one
safe configuration
When Exp5610 builds the live attempt configuration
Then the live attempt still executes, enabling only safe promoted filters when
present and otherwise using the unchanged current live-agent baseline.

#### SCENARIO-ARC-FCP-5610-REPRODUCTION-GATE-BANKS-ONLY-NEW-LEVELS

Given a bounded live-agent trace
When the trace does not reach and independently reproduce the selected target
Then the artifact records `live_attempt_executed=true`,
`offline_reproduced=false`, `new_reproducible_levels=[]`, `registry_updated=false`,
and a terminal honest null. When a trace reaches the selected target and
independent replay matches the trace checksum, then and only then the artifact
records a new reproducible level and increments `levels_after`.

### REQ-ARC-FCP-5621: V507 Unconditional Live Self-Discovery Level-Up Attempt

Experiment 5621 SHALL write
`results/experiment_5621_arc_live_self_discovery_levelup_v507.json` after one
unconditional live-agent self-discovery attempt against a non-duplicate ARC
frontier level. The experiment SHALL registry-precheck all public offline-arcade
games before target selection. The precheck SHALL exclude levels already
reproduced by `scripts/arc_loop_solve.py`, levels already recorded in
`ops/arc_solve_registry.yaml`, prior artifact targets, Exp5610's attempted
target, and any same-v507 attempted target.

Exp5620 is advisory only. If Exp5620 emits `live_branch_promotion_score=1.0`
with no safety regression, Exp5621 SHALL enable exactly that emitted live branch
configuration. Otherwise Exp5621 SHALL run the unchanged no-new-LLM live
baseline. A blocked or negative Exp5620 result SHALL NOT gate, skip, or replace
the live attempt with a development proxy.

The live attempt SHALL discover from its own runtime observations, actions,
state transitions, and runtime reverse-engineering signals. It SHALL NOT inspect
game source, run exhaustive offline ground-truth BFS, inject a hand-built
per-game adapter, or replay a hidden solution recipe. The action budget, seed,
target, stopping rule, and mechanism configuration SHALL be fixed before the
attempt. If no LLM is invoked, the artifact SHALL declare
`inference_substrate=offline_arcade_live_agent_runtime_self_discovery_no_llm`,
`llm_invoked=false`, and `model_specs=[]`. A candidate level SHALL count only
when it exceeds the registry precheck, the action trace checksum replays exactly,
and `offline_reproduced=true` through the generic offline reproduction path.

Required field principles:

- `field_principles`: principle "principle annotations are carried in the artifact so every required 5621 field is auditable."
- `registry_precheck`: principle "duplicate levels receive no credit; all public games, registry depths, arc_loop_solve depths, prior artifact targets, Exp5610's attempted target, and same-v507 attempts are checked before target selection."
- `target_selection_receipt`: principle "rotation and authenticated public-game headroom are explicit, so the selected next level is not a duplicate."
- `live_attempt_executed`: principle "bare bool true proves the ARC standing floor was a real runtime attempt, not an advisory precheck."
- `live_branch_configuration`: principle "Exp5620 promotion use is auditable; blocked or unsafe Exp5620 receipts leave the no-new-LLM baseline unchanged and cannot skip the attempt."
- `action_budget`: principle "search cost is bounded before runtime begins."
- `attempt_trace_path`: principle "discovery evidence is replayable from a durable trace."
- `levels_before`: principle "authoritative registry total before the attempt; the north-star delta is exact."
- `levels_after`: principle "authoritative registry total after accepted banking; unchanged on honest nulls."
- `new_reproducible_levels`: principle "only newly reproduced levels beyond the precheck depth count."
- `offline_reproduced`: principle "a live reach needs independent replay; duplicate or unreplayed reaches do not bank."
- `registry_updated`: principle "successful evidence becomes durable, while null attempts leave the registry unchanged."
- `solve_provenance`: principle "must equal live_agent_self_discovery for any credited path."
- `source_files_read`: principle "must be false; outer-loop source reverse engineering is excluded."
- `per_game_adapter_used`: principle "must be false; hidden per-game solvers are not smuggled into live self-discovery credit."
- `model_specs`: principle "empty only when no LLM is invoked; otherwise it contains a mandated cached V507 model with invocation receipt."
- `inference_substrate`: principle "offline_arcade_live_agent_runtime_self_discovery_no_llm when no LLM fires; otherwise the authenticated local GGUF substrate."
- `random_seeds`: principle "deterministic seeds make the attempt replayable and auditable."
- `reproducibility_checksum`: principle "content-addressed artifact checksum catches silent target, trace, or branch-configuration drift."
- `honest_verdict`: principle "no-new-level is terminal; a blocked or negative Exp5620 A/B is not permission to skip the attempt."

#### SCENARIO-ARC-FCP-5621-PRECHECK-ROTATES-PAST-EXP5610

Given public environment metadata, registry depths, arc_loop_solve outputs,
prior artifacts, Exp5610's attempted target, and same-v507 attempts
When Exp5621 runs its registry precheck
Then it selects the first rotated target whose next level is beyond both the
registry and arc_loop_solve depths, has authenticated public headroom, and is
not Exp5610's attempted target or a same-v507 duplicate.

#### SCENARIO-ARC-FCP-5621-BRANCH-ADVISORY-NOT-GATING

Given Exp5620 is blocked, negative, unsafe, or safely promoted
When Exp5621 builds its live branch configuration
Then the live attempt still executes, enabling only an exactly emitted
non-regressing promotion when present and otherwise using the unchanged
no-new-LLM live baseline.

#### SCENARIO-ARC-FCP-5621-REPRODUCTION-GATE-BANKS-ONLY-NEW-LEVELS

Given a bounded live-agent trace
When the trace does not reach and independently reproduce the selected target
Then the artifact records `live_attempt_executed=true`, `model_specs=[]`,
`offline_reproduced=false`, `new_reproducible_levels=[]`,
`registry_updated=false`, and a terminal honest null. When a trace reaches the
selected target and independent replay matches the trace checksum, then and only
then the artifact records a new reproducible level and increments
`levels_after`.

### REQ-ARC-FCP-5632: V508 Registry-Rotated Live Self-Discovery Level-Up Attempt

Experiment 5632 SHALL write
`results/experiment_5632_arc_live_self_discovery_levelup_v508.json` after one
bounded live-agent self-discovery attempt against exactly one unreproduced ARC
frontier level. The registry precheck SHALL run at execution time across every
public offline-arcade game, SHALL exclude `bp35` level 9, `sk48` level 8, and
every level already reproduced in `ops/arc_solve_registry.yaml`, and SHALL record
the selected target and target-selection hash before any live observation or
action is taken. A selected target SHALL have authenticated public-game headroom
for the next level beyond its registry depth.

Exp5631 is advisory only. Experiment 5632 SHALL use Exp5631's epistemic policy
only when that artifact has `live_epistemic_policy_ready=true`,
`unsafe_model_accept_count=0`, and `known_level_regression_count=0`. Otherwise
the experiment SHALL run the unchanged no-new-LLM live E3 baseline. The selected
target and policy source SHALL NOT change after any live outcome is observed.

The live attempt SHALL use only the agent's own runtime observations, actions,
memory, and runtime reverse-engineering signals. It SHALL NOT inspect game
source, run exhaustive offline ground-truth BFS, use a hand `GameAdapter`, import
an outer-loop recipe, or precompute the solution. Seeds, wall time, model calls,
environment actions, retries, and checkpoints SHALL be bounded before execution.
If no LLM is invoked, `model_specs` SHALL be `[]`; if an LLM is invoked, the
artifact SHALL name a mandated cached SOTA GGUF model receipt rather than a legacy
smoke-only model.

A level SHALL count only when the selected target level is reached by the live
environment, the same trace independently reproduces through the generic live
path from a clean state, `offline_reproduced=true`, and the trace checksum
matches. The registry delta SHALL be exactly `0` for a bounded null or `1` for
one newly banked target level.

Required field principles:

- `field_principles`: principle "principle annotations are carried in the artifact so every required 5632 field is auditable."
- `registry_count_before`: principle "authoritative reproduced-level total before target selection; the level-up baseline is explicit."
- `registry_precheck_receipt`: principle "execution-time precheck proves the selected target is unreproduced before any live observation."
- `excluded_targets`: principle "bp35 L9, sk48 L8, and registry-duplicate levels stay closed and cannot receive duplicate credit."
- `selected_game`: principle "the game scope is fixed before live observation, preventing outcome-driven target switching."
- `selected_level`: principle "the level scope is fixed before live observation, preventing outcome-driven target switching."
- `target_selection_hash`: principle "content hash of the pre-outcome target receipt proves the target was not changed after seeing results."
- `policy_source`: principle "records whether Exp5631 promoted cleanly or the unchanged baseline ran."
- `model_specs`: principle "empty for a no-LLM run; otherwise names the mandated cached SOTA GGUF receipt exactly."
- `budget_receipt`: principle "seeds, wall time, model calls, environment actions, retries, and checkpoints are bounded before execution."
- `live_trace_path`: principle "complete live observation/action evidence is durable and replayable."
- `live_path_reachability_counters`: principle "the scored live mechanism that generated actions is identified by runtime counters."
- `solve_provenance`: principle "must equal live_agent_self_discovery; only the credited path can solve."
- `level_reached`: principle "terminal environment level fact is explicit and separate from reproduction credit."
- `reproduced_levels`: principle "newly reproduced target levels; solve credit requires at least one."
- `offline_reproduced`: principle "exactly true is mandatory for solve credit after independent generic reproduction."
- `registry_count_after`: principle "authoritative reproduced-level total after accepted banking; unchanged on honest nulls."
- `registry_delta`: principle "exactly 0 or 1 so the banked-level delta is auditable."
- `source_read`: principle "must be false; game source is excluded from live self-discovery credit."
- `game_adapter_used`: principle "must be false; no per-game adapter can be smuggled into the live path."
- `outer_loop_re_used`: principle "must be false; off-path recipes are excluded from live self-discovery credit."
- `inference_substrate`: principle "live_agent_environment_interaction -- environment observations/actions are the authority, not an offline solver."
- `random_seeds`: principle "deterministic seeds make the bounded attempt replayable and auditable."
- `reproducibility_checksum`: principle "content-addressed target, trace, budget, and banking decision catch silent drift."
- `honest_verdict`: principle "a bounded no-level result is terminal and must not be upgraded without reproduction."

#### SCENARIO-ARC-FCP-5632-PRECHECK-EXCLUDES-RECENT-AND-REGISTRY-DUPLICATES

Given the current registry, authenticated public-game headroom, and the explicit
`bp35` L9 and `sk48` L8 exclusions
When Experiment 5632 runs its execution-time registry precheck
Then it closes every reproduced registry level, excludes the two recent unbanked
targets, selects one unreproduced next-level target with authenticated headroom,
and records a target-selection hash before live execution.

#### SCENARIO-ARC-FCP-5632-EPISTEMIC-POLICY-ADVISORY-NOT-GATING

Given Exp5631 is blocked, missing, unsafe, or regresses a known level
When Experiment 5632 builds its live policy receipt
Then the attempt still executes using the unchanged no-new-LLM baseline. Given
Exp5631 is ready, has zero unsafe accepts, and has zero known-level regressions
Then the promoted epistemic policy source is recorded without changing the
preselected target after outcomes.

#### SCENARIO-ARC-FCP-5632-REPRODUCTION-GATE-BANKS-AT-MOST-ONE-LEVEL

Given a bounded live-agent trace
When the trace does not reach and independently reproduce the selected target
Then `offline_reproduced=false`, `reproduced_levels=0`, `registry_delta=0`, and
the verdict is a terminal honest null. When the trace reaches and independently
reproduces the selected target from a clean state
Then `offline_reproduced=true`, `reproduced_levels=1`, `registry_delta=1`, and
only the selected target level is banked even if the terminal level counter is
deeper.

### REQ-ARC-FCP-5643: V509 Registry-Rotated Live Self-Discovery Level-Up Attempt

Experiment 5643 SHALL write
`results/experiment_5643_arc_live_self_discovery_levelup_v509.json` after one
bounded live-agent self-discovery attempt against exactly one authenticated ARC
target fixed before interaction. The registry precheck SHALL run at execution
time across every public offline-arcade game, SHALL exclude every level already
reproduced in `ops/arc_solve_registry.yaml`, SHALL exclude `bp35` level 9,
`sk48` level 8, `lf52` level 7, and every recent failed ARC target recorded by
the V509 transition receipt, and SHALL record the selected target plus a
target-selection hash before any live observation or action is taken. If the
immediate next level is closed by a recent failed-target receipt but the same
game still has authenticated public-game headroom, the selector MAY rotate to
the next non-closed unreproduced level, but the artifact SHALL record the closed
intermediate level explicitly and SHALL NOT count any registry delta unless the
generic clean-state reproduction gate proves the selected target.

Exp5642 is advisory only. Experiment 5643 SHALL use Exp5642's executable-model
policy only when that artifact has `live_executable_model_ready_score=1.0`,
`unsafe_model_accept_count=0`, and `known_level_regression_count=0`. Otherwise
the experiment SHALL run the unchanged no-new-LLM live E3 baseline. The selected
target, policy source, thresholds, and budget SHALL NOT change after any live
outcome is observed.

The live attempt SHALL use only the agent's own runtime observations, actions,
memory, and runtime reverse-engineering signals. It SHALL NOT inspect game
source, run exhaustive offline ground-truth BFS, use a hand `GameAdapter`, import
an outer-loop recipe, or precompute the solution. Seeds, wall time, model calls,
environment actions, retries, checkpoint cadence, and terminal conditions SHALL
be bounded before execution. If no LLM is invoked, `model_specs` SHALL be `[]`;
if an LLM is invoked, the artifact SHALL name a mandated cached SOTA GGUF model
receipt rather than a legacy smoke-only model.

A level SHALL count only when the selected target level is reached by the live
environment, the same trace independently reproduces through the generic live
path from a clean state, `offline_reproduced=true`, and the trace checksum
matches. The registry delta SHALL be exactly `0` for a bounded null or `1` for
one newly banked target level.

Required field principles:

- `field_principles`: principle "principle annotations are carried in the artifact so every required 5643 field is auditable."
- `registry_count_before`: principle "authoritative reproduced-level total before target selection; the level-up baseline is explicit."
- `registry_precheck_receipt`: principle "execution-time precheck proves the selected target is unreproduced before any live observation."
- `excluded_targets`: principle "bp35 L9, sk48 L8, lf52 L7, transition-receipt failures, and registry-duplicate levels stay closed and cannot receive duplicate credit."
- `selected_game`: principle "the game scope is fixed before live observation, preventing outcome-driven target switching."
- `selected_level`: principle "the level scope is fixed before live observation, preventing outcome-driven target switching."
- `target_selection_hash`: principle "content hash of the pre-outcome target receipt proves the target was not changed after seeing results."
- `policy_source`: principle "records whether Exp5642 promoted cleanly or the unchanged baseline ran."
- `methodology_receipt`: principle "records target precheck, policy freeze, budget freeze, no-source/no-adapter/no-outer-loop provenance, and reproduction criteria so the run is not a short opaque artifact."
- `model_specs`: principle "empty for a no-LLM run; otherwise names the mandated cached SOTA GGUF receipt exactly."
- `budget_receipt`: principle "seeds, wall time, model calls, environment actions, retries, checkpoint cadence, and terminal conditions are bounded before execution."
- `live_trace_path`: principle "complete live observation/action evidence is durable and replayable."
- `live_path_reachability_counters`: principle "the scored live mechanism that generated actions is identified by runtime counters."
- `solve_provenance`: principle "must equal live_agent_self_discovery; only the credited path can solve."
- `level_reached`: principle "terminal environment level fact is explicit and separate from reproduction credit."
- `reproduced_levels`: principle "newly reproduced target levels; solve credit requires at least one."
- `offline_reproduced`: principle "exactly true is mandatory for solve credit after independent generic reproduction."
- `registry_count_after`: principle "authoritative reproduced-level total after accepted banking; unchanged on honest nulls."
- `registry_delta`: principle "exactly 0 or 1 so the banked-level delta is auditable."
- `source_read`: principle "must be false; game source is excluded from live self-discovery credit."
- `game_adapter_used`: principle "must be false; no per-game adapter can be smuggled into the live path."
- `outer_loop_re_used`: principle "must be false; off-path recipes are excluded from live self-discovery credit."
- `inference_substrate`: principle "live_agent_environment_interaction -- environment observations/actions are the authority, not an offline solver."
- `random_seeds`: principle "deterministic seeds make the bounded attempt replayable and auditable."
- `reproducibility_checksum`: principle "content-addressed target, trace, budget, methodology, and banking decision catch silent drift."
- `honest_verdict`: principle "a bounded no-level result is terminal and must not be upgraded without reproduction."

#### SCENARIO-ARC-FCP-5643-PRECHECK-EXCLUDES-TRANSITION-FAILURES

Given the current registry, authenticated public-game headroom, the explicit
`bp35` L9, `sk48` L8, and `lf52` L7 exclusions, and a V509 transition receipt
that records `lf52` L7 as a failed live-level credit target
When Experiment 5643 runs its execution-time registry precheck
Then it closes every reproduced registry level, excludes all explicit and
transition-receipt failed targets, rotates only among unreproduced authenticated
targets, and records a target-selection hash before live execution.

#### SCENARIO-ARC-FCP-5643-EXECUTABLE-POLICY-ADVISORY-NOT-GATING

Given Exp5642 is blocked, missing, unsafe, or regresses a known level
When Experiment 5643 builds its live policy receipt
Then the attempt still executes using the unchanged no-new-LLM baseline. Given
Exp5642 has ready score 1.0, zero unsafe accepts, and zero known-level
regressions
Then the promoted executable-model policy source is recorded without changing
the preselected target after outcomes.

#### SCENARIO-ARC-FCP-5643-METHODOLOGY-AND-REPRODUCTION-GATE

Given a bounded live-agent trace
When the trace does not reach and independently reproduce the selected target
Then `offline_reproduced=false`, `reproduced_levels=0`, `registry_delta=0`,
`methodology_receipt` records the no-source/no-adapter/no-outer-loop path, and
the verdict is a terminal honest null. When the trace reaches and independently
reproduces the selected target from a clean state
Then `offline_reproduced=true`, `reproduced_levels=1`, `registry_delta=1`, and
only the selected target level is banked even if the terminal level counter is
deeper.

### REQ-ARC-FCP-5699: SGE Anti-Stagnation Controller -- Genuine Live Collapse-Escape Re-Test

`ops/known-issues.md` task 6's own "NEXT STEP" named the fix: detect repeated
null-outcome LLM strategies and force diversity instead of converging on a
passive "wait" strategy. The conductor built `AntiStagnationDiversityController`
(now `SGECandidateRouter`'s default `anti_stagnation_controller`) and ran a
DETERMINISTIC precheck (exp5575) replaying the original recorded collapse
trace -- real evidence the controller LOGIC is correct, but not yet evidence
a genuine LIVE run (fresh strategies from the model, not a replayed trace)
actually escapes a collapse it would otherwise fall into. exp5575's own
follow-on live attempt (exp5576) never ran -- it GATE_BLOCKed on unrelated
project-wide gates (a pre-existing 16-failure full-suite pytest run and a
1262-test spec-coverage backlog), not on anything about the SGE mechanism
itself.

This requirement runs the genuine live re-test. The ORIGINAL null target
(g50t L3) is no longer usable as a registry-frontier attempt -- it was
independently fully cleared (`levels_reproduced=7, full_game_clear=true`)
one day after exp5575's precheck by an unrelated hand-derived mechanism. Per
"(or the original null target)", this requirement structurally replicates the
original scenario instead: a fresh-episode g50t session
(`prior_levels=0, target_level=1`, no registry credit claimed) reproduces the
early-exploration regime the original collapse occurred in, run TWICE (n=2
independent episodes) per CLAUDE.md's cross-check-surprising-results
discipline. A secondary, bonus real attempt at the current shallowest
not-fully-cleared registry frontier (read live, not hardcoded) tests whether
the fix also helps bank new territory.

### SCENARIO-ARC-FCP-5699-LIVE-COLLAPSE-ESCAPE: Real Live Collapse, Real Live Escape

Given a fresh `E3AgentPolicy` session on g50t with `SGECandidateRouter`'s
default anti-stagnation controller active and real GPU-backed local-GGUF LLM
inference (not a fake-completer, not a replayed trace)
When the live strategy proposer genuinely converges on a repeated
"observe/wait" strategy across several steps -- the exact failure mode task 6
was filed against
Then the controller detects the live collapse (`collapse_detected_live=True`,
with a real `collapse_trigger_step`), switches ranking away from the LLM
proposer to the deterministic forced diverse portfolio
(`forced_portfolio_activated_live=True`, `post_collapse_strategy_diversity>1`),
and this behavior reproduces independently across both replication episodes;
the escape is reported honestly including its own limitation (the forced
portfolio may itself settle into a smaller, static repetition on a frozen
game state) rather than being overclaimed as full resolution; a genuine
no-collapse observation on the secondary frontier pass (e.g. because the
perception layer yields empty candidate lists at a deep, already-explored
frontier) is reported as an honest, separately-flagged null, not silently
folded into the headline collapse-escape claim

### REQ-ARC-FCP-5699-2: Forced-Portfolio Rotation -- Closing the Partial-Escape Gap

REQ-ARC-FCP-5699's own RESOLUTION disclosed a limitation, not hidden: the forced
diverse portfolio genuinely escapes the LLM's repeated strategy TEXT, but
`rank_forced_portfolio`'s `ranked_pool` sort was fully deterministic given an unchanging
candidate list -- on a frozen game state, THREE of the five forced-portfolio categories
(`observation`, `action_type_probe`, `recovery_reset`) intentionally tolerate a past
OUTCOME failure (`allow_failed_signature=True`, a deliberate design choice) but nothing
prevented them from re-selecting the exact same top-ranked candidate every subsequent
call. A real 90-budget live run on g50t measured the IDENTICAL 5-candidate
`forced_portfolio_selected` set on every one of 44 consecutive forced-portfolio calls
(steps 47-90) -- a genuine escape from the LLM's own repetition, immediately followed by
a NEW static repetition of the controller's own making.

`AntiStagnationDiversityController` SHALL track `_recently_forced_signatures(history,
window=5)` -- every candidate signature ANY forced-portfolio category selected in the
last `window` forced-portfolio history rows, regardless of recorded outcome (a
forced-portfolio row is appended with `outcome="pending"` and typically never gets a
real outcome before the NEXT call, one step later, so outcome-gating alone cannot drive
this). `rank_forced_portfolio`'s `add_from_pool` SHALL make two passes per category: pass
1 excludes candidates whose signature is in `_recently_forced_signatures` (rotate to the
next-best candidate in that category's ranked pool); pass 2, entered only when pass 1
finds nothing in an otherwise non-empty pool, re-selects without that exclusion (the
category-fill guarantee is not weakened -- a category with a truly-exhausted pool still
gets filled, it simply can no longer avoid repeating). The router's own history-append
SHALL record a NEW `forced_signatures` list field (every category's picks for that call,
not just the first/observation one, which is all the pre-existing singular
`chosen_signature` field ever captured) so the rotation tracker can see the FULL set of
recently-forced candidates across all categories, not only the one the pre-existing field
happened to record.

**RESOLUTION (2026-07-14).** Verified two ways, deliberately not conflated:

1. **Unit-level, mechanism correctness.** `test_anti_stagnation_forced_portfolio_rotates_across_repeated_calls`
   uses a realistically-sized pool (20 candidates vs. `SGECandidateRouter`'s own
   `max_candidates=8` default) and shows the `observation` category picks a genuinely
   DIFFERENT signature on 2 consecutive calls (both `!=` assertions pass) --
   the exact behavior the pre-fix code could not produce (a direct trace with the same
   pool and the OLD code shows the identical top-ranked pick every call).
   `test_anti_stagnation_forced_portfolio_rotation_falls_back_when_pool_exhausted`
   confirms a single-candidate pool still fills the category (re-selecting the only
   option) and correctly reports `rotation_exhausted_categories`, so the category-fill
   guarantee holds.
2. **Live-level, an honest non-bug finding.** Re-running the EXACT g50t scenario that
   exposed the original bug, AFTER this fix, still measured static repetition. Direct
   investigation (no GPU needed -- candidate generation does not invoke the LLM) found
   the real, structural cause: `rich_action_candidates` on g50t's spawn frame returns
   EXACTLY 5 candidates (`ACTION1`-`ACTION5`; `_available_action_ids` returns `[1,2,3,4,5]`
   -- action 6, click, is not available at all, so there is zero coordinate diversity) --
   matching the anti-stagnation controller's 5 forced-portfolio categories exactly. Every
   category's pool is exhausted the FIRST time it fills; there is no alternative candidate
   for ANY selection algorithm to rotate to on this specific game/state. This is a genuine
   limit of g50t's candidate space at this frame, not evidence the fix does not work --
   the unit tests are the correct place to verify the rotation MECHANISM, since they can
   construct a pool with real headroom that this particular frozen g50t frame does not
   have. `sk48` (45 candidates at spawn, action 6 available) was tried as a
   richer-candidate live target but did not trigger collapse within a 90-step budget in
   this session's attempt -- an inconclusive, not a negative, result (sk48's real
   candidates likely give the LLM strategy proposer enough genuine variety that it does
   not degenerate into the repeated-"wait" pattern this whole mechanism targets).

#### SCENARIO-ARC-FCP-5699-2-FORCED-PORTFOLIO-ROTATES

Given a frozen candidate pool (unchanging across consecutive `rank_forced_portfolio`
calls, matching a stalled game state) with genuine headroom (more distinct candidates
than the portfolio consumes per call)
When forced-portfolio mode is invoked on consecutive calls with accumulating history
Then a category that tolerates re-selecting a past-failed signature (`observation`,
`action_type_probe`, `recovery_reset`) selects a DIFFERENT candidate than it selected on
the immediately preceding call, for as many consecutive calls as the pool has
unconsumed alternatives; once the pool is genuinely exhausted, the category still gets
filled (re-selecting a recently-forced signature) rather than being left empty, and this
is reported via `rotation_exhausted_categories`

### REQ-ARC-FCP-5699-3: Reflect()-Prompt Anti-Stagnation Nudge -- Softer Signal, Inside the LLM Call

REQ-ARC-FCP-5699/5699-2 (above) built a HARD deterministic escape: once
`AntiStagnationDiversityController.assess()` detects collapse (>= 3 of 4 signals across an
8-step window, including `consecutive_null_outcomes >= 4`), `SGECandidateRouter.rank()`
bypasses the LLM entirely and forces a hand-coded diverse portfolio. This closes the failure
mode, but only AFTER a fairly strict multi-signal threshold fires. The mechanism's own
2026-07-10 outer-loop investigation documented a softer, earlier version of the same
problem: a full-budget live g50t run showed the LLM's own `reflect()` calls -- which DO run
before collapse is detected, on the plain `_REFLECT_INSTRUCTIONS` prompt ("state ONE short
sentence on what to try differently") -- converged on a repetitive "wait for the system to
process the pending interaction" strategy across MULTIPLE reflection cycles, never
escalating to more assertive probing (`ops/known-issues.md` task 6, "NEXT STEP" note). The
generic reflection framing was not enough to break the pattern on its own; nothing INSIDE
the prompt the model actually reads named the repetition explicitly.

`LLMStrategyProposer.reflect()` SHALL accept an optional keyword-only `taboo_strategies:
Sequence[str] = ()` and SHALL compute `_consecutive_null_outcomes(history)` internally
(reusing the existing REQ-ARC-FCP-5699 helper, not a new detector). When EITHER
`taboo_strategies` is non-empty OR the null-outcome streak reaches `_REFLECT_NUDGE_NULL_STREAK`
(= 2, deliberately lower than the hard gate's `consecutive_null_outcomes = 4`, so the
prompt-level nudge fires earlier and more gently than the deterministic override), an
`ANTI-STAGNATION WARNING` sentence SHALL be spliced into the prompt BEFORE the history
section, naming the taboo strategies verbatim when given (so the model sees exactly what
NOT to repeat) and explicitly demanding a genuinely different action category (a different
action type, a different grid area, or an active/committal action instead of a
passive/waiting one) rather than another minor variation. `reflect()`'s return dict SHALL
report `nudge_fired: bool` and `consecutive_null_outcomes: int` for auditability (matching
this module's existing convention of recording every anti-stagnation decision -- `taboo_set`,
`taboo_policy`, `rotation_exhausted_categories`, etc. -- rather than making it silent).
`SGECandidateRouter.rank()` SHALL feed `anti_stagnation_controller.taboo_set(reflect_window)`
into every scheduled `reflect()` call (when a controller is configured; `()` otherwise, which
preserves the exact pre-existing plain-prompt behavior for any caller/test that constructs
`LLMStrategyProposer` directly) and SHALL record `reflection_nudge_fired` in
`last_diagnostics`, complementing (not replacing) the harder deterministic override this
requirement's own gate still handles once collapse is fully detected.

**Empty-history and completer-failure paths are unaffected.** `reflect()`'s existing
empty-history short-circuit (`if not history: return {...}`, 3 keys, unchanged) fires before
any nudge computation -- there is no history to detect a streak in. The completer-failure
path now also reports `nudge_fired`/`consecutive_null_outcomes` (computed before the
completer call, since the prompt was already built with or without the nudge by the time the
completer is invoked), matching `propose_one()`'s existing convention of reporting
`temperature`/`completer_ok` on every return path including failures.

#### SCENARIO-ARC-FCP-5699-3-REFLECT-PROMPT-NAMES-THE-STAGNATION

Given `reflect()` is called with a history window containing at least one strategy that led
to a null outcome (or the caller supplies `taboo_strategies` derived from
`AntiStagnationDiversityController.taboo_set`)
When the null-outcome streak reaches `_REFLECT_NUDGE_NULL_STREAK` OR `taboo_strategies` is
non-empty
Then the prompt sent to the completer contains an explicit `ANTI-STAGNATION WARNING` naming
the specific repeated strategy text (when known) and demanding a genuinely different action
category, rather than relying solely on the model to infer from raw (strategy, outcome) pairs
that its recent choices have not been working -- the documented g50t failure showed that
inference does not reliably happen on its own across multiple reflection cycles

### REQ-ARC-FCP-5699-4: Early-Trigger the Soft Nudge -- Racing the Hard Collapse Gate, Not Nested Inside It

REQ-ARC-FCP-5699-3 shipped the reflect()-prompt nudge (above) but left it gated ENTIRELY by
`SGECandidateRouter.rank()`'s pre-existing periodic schedule (`self._step % reflect_every ==
0`, default every 6th call) -- `reflect()` was only ever invoked from that one call site. A
2026-07-15 real-GPU 3-game re-test (`scripts/outer_loop_sge_smoke_test.py`, extended per
operator request "can we also add more games to the sample?" from g50t-only to g50t + sk48 +
cd82) found the soft nudge NEVER fired in any of the 3 games. Root cause: 2 of 3 games (g50t,
sk48) instead hit `AntiStagnationDiversityController`'s HARD collapse gate first --
`assess()` is checked at the TOP of every `rank()` call (not gated by any schedule), and once
`collapse_detected` is true, `rank()` returns early via the `rank_forced_portfolio` branch,
which structurally never reaches the propose/reflect code path again for the rest of that
game's run. The soft signal (checked only every `reflect_every`th call) and the hard signal
(checked every call) were racing on unequal terms by construction -- the hard gate wins
essentially every time its own thresholds are anywhere close to firing, because the soft gate
simply isn't looking yet.

`SGECandidateRouter.rank()` SHALL check the SAME soft signal `reflect()` itself uses
(`_consecutive_null_outcomes` over the `reflect_window`, OR a non-empty
`anti_stagnation_controller.taboo_set(reflect_window)`) on EVERY call in the non-collapsed
branch -- not just when `self._step % reflect_every == 0`. When the soft signal is present on
a call where the periodic schedule has NOT yet arrived, `reflect()` SHALL fire immediately
anyway (an "early" trigger), racing the hard gate on genuinely equal terms: both signals are
now evaluated every call, so whichever threshold the accumulating history crosses first is the
one that acts -- and since the soft threshold (`_REFLECT_NUDGE_NULL_STREAK = 2`, OR any single
null-outcome entry populating a non-empty taboo set) is structurally easier to satisfy than the
hard gate's (`consecutive_null_outcomes >= 4` AND at least 2 more of 3 other rolling-window
signals, `min_triggered_signals = 3` of 4), the soft path should now usually get a genuine turn
before the hard gate can pre-empt it. This is a strictly additive OR-condition on the
pre-existing schedule (`scheduled_reflect or nudge_would_fire_early`) -- the periodic cadence
is unchanged for any call where no stagnation signal is present, so a healthy (non-stagnating)
run's LLM-call budget is unaffected; only calls that were ALREADY heading toward the hard
collapse gate anyway get an extra, cheap (no LLM call to decide) early check.

`last_diagnostics` SHALL record `reflection_trigger: "scheduled" | "early_stagnation_signal" |
None`, distinguishing which condition caused (or didn't cause) a reflect() call, for the same
auditability reasons as every other anti-stagnation decision this module records.

**Not a full fix.** This closes the "soft nudge is nested inside the periodic-only cadence"
gap specifically. It does NOT change what happens ONCE collapse has already triggered on an
EARLIER call (the forced-portfolio branch still bypasses propose/reflect entirely for the rest
of that game) -- it only widens the WINDOW in which the soft path can act BEFORE collapse ever
triggers in the first place. Whether this actually changes real outcomes (more games leveling
up, or the nudge actually parsing successfully more often) is an open empirical question for a
follow-up real-GPU run, not established by this requirement alone.

#### SCENARIO-ARC-FCP-5699-4-EARLY-TRIGGER-RACES-HARD-COLLAPSE

Given a game whose candidate/outcome dynamics are on a trajectory toward
`AntiStagnationDiversityController`'s hard collapse gate (checked every `rank()` call)
When the same underlying history ALSO crosses the softer reflect()-nudge signal
(`_consecutive_null_outcomes` or a non-empty taboo set) on a call that does not fall on the
periodic `reflect_every` schedule
Then `reflect()` fires immediately on that call (`reflection_trigger =
"early_stagnation_signal"`) rather than waiting for the next scheduled boundary or losing the
race entirely to the hard gate's own every-call check -- a healthy run with no stagnation
signal present is unaffected, still reflecting only on the periodic schedule
(`reflection_trigger = "scheduled"`)

### REQ-ARC-FCP-5699-5: Honest Level Tracking in the SGE Smoke-Test Harness -- Corrigendum

Discovered 2026-07-15 while checking whether cd82's reflection advice (REQ-ARC-FCP-5699-3/5699-4)
actually changed subsequent `propose_many()` behavior (operator: "do that"). Direct inspection of
`action_log` in every artifact `scripts/outer_loop_sge_smoke_test.py` had ever produced (the
original 2026-07-10 g50t run through this session's 3-game REQ-ARC-FCP-5699-4 re-test, 7 real-GPU
runs total) found `level_before`/`level_after` = 0 on every single logged action, in every run,
on all 3 games -- the real environment level never left 0, not once.

**Root cause.** `run_game()` initialized `max_level = prior_levels` and folded the real observed
level into that SAME variable (`max_level = max(max_level, after_level)`). This harness has NEVER
seeded the env at `prior_levels` -- there is no `GameAdapter`, no banked-trajectory replay, just a
bare `env.reset()` before the exploration loop; every game therefore starts at whatever level a
true cold reset lands on (observed to be 0 in every run). Because `prior_levels` (1 or 2, taken
from `ops/arc_solve_registry.yaml`'s per-game shallow-frontier labels -- what OTHER solve methods,
GameAdapters and banked trajectories, have reached for that game) was always >= the real level
ever actually observed, `max(max_level, after_level)` silently reported the ASSUMED starting point
forever, independent of what the run genuinely achieved. Every prior write-up of this smoke test's
results (chat summaries and `ops/known-issues.md` task 6 entries alike) reported this artifact
(e.g. "g50t stayed at L2") as if it were a real measurement; the actual, more informative fact was
"g50t (and every other game tested) never left level 0 at all."

`run_game()` SHALL track the real observed level trajectory (`real_initial_level`, set from the
FIRST post-reset frame; `real_max_level_observed`, the max over every subsequent `level_after`)
independently of the `prior_levels`/`target_level` parameters, which SHALL be documented
explicitly as informational-only labels (never applied as an env seed) via a `methodology_note`
field on the artifact. `leveled_up` SHALL be computed as `real_max_level_observed >
real_initial_level`, never blended with `prior_levels`. `max_level_reached` (the pre-existing
field name, kept for backward-compat readability) SHALL report `real_max_level_observed`, not the
unenforced floor.

**RESOLUTION (2026-07-15).** Fixed in `scripts/outer_loop_sge_smoke_test.py`. Every existing
artifact this harness had ever produced (`results/outer_loop_sge_smoke_test*.json`, 9 files
including both REQ-ARC-FCP-5699-3/5699-4 baseline snapshots) was retroactively patched with
`real_initial_level`/`real_max_level_observed`/`leveled_up` computed directly from each file's own
already-recorded `action_log`, plus a `corrigendum_2026_07_15` field explaining the fix -- the
original (misleading) `max_level_reached`/`prior_levels_reproduced` fields were preserved
unmodified alongside the correction, per this project's adversarial-artifact-verification
corrigendum convention (never silently overwrite a wrong number; disclose the correction next to
it). Result: `real_initial_level=0, real_max_level_observed=0, leveled_up=false` for every one of
the 7 runs. **This does NOT invalidate the REQ-ARC-FCP-5699-3/5699-4 mechanism findings** (the
nudge firing, parse-rate improvements, cd82's strategy-text language genuinely shifting toward
"active-commitment" advice after each reflection) -- those are facts about the router's internal
behavior verified directly from `diagnostics_log`, independent of this level-tracking bug. What
changes is the INTERPRETATION of "0/3 leveled up": never "0/3 escaped their assumed L1/L2 starting
point," always "0/3 escaped level 0 at all" -- a starker, more honest null.

#### SCENARIO-ARC-FCP-5699-5-LEVEL-TRACKING-NEVER-BLENDS-WITH-UNVERIFIED-PRIOR

Given a smoke-test harness explores a game from a bare `env.reset()` with no game-specific seeding
mechanism (no `GameAdapter`, no banked-trajectory replay)
When the harness also carries an INFORMATIONAL `prior_levels` label describing what a DIFFERENT
solve method has reached for that game (from a registry, not from this run)
Then the harness's own tracked "level reached this run" variable is NEVER initialized from or
blended with that informational label -- it is computed strictly from the real observed
`level_before`/`level_after` trajectory of THIS run, so a `leveled_up` claim always reflects a
genuine measured transition and never an unverified assumption carried through a `max()` call

### REQ-ARC-FCP-5699-6: Deterministic-Router Control -- Was It Ever SGE Specifically?

REQ-ARC-FCP-5699-5's corrigendum established that `real_max_level_observed=0` on all 3 games in
every run of this investigation (7 real-GPU runs, `SGECandidateRouter` every time). No run in the
whole REQ-ARC-FCP-5699 through 5699-5 chain ever used a DIFFERENT candidate router under the same
stripped-down `E3AgentPolicy` config (`proposer=_NoOpInductionProposer()`,
`frame_change_scorer=None`, `action_effect_expansion_prior=False`, `goal_bias=None`,
`goal_candidate_guidance=False`, `active_probe_controller=False`, `go_explore_archive=False` --
every OTHER production exploration feature deliberately disabled to isolate the candidate router
under test). Without that comparison, "0/3 leveled up with SGE" carries no information about
whether SGE specifically is the bottleneck, versus the stripped-down harness itself being
incapable of a first level-up on these 3 games regardless of exploration strategy.

`run_game()` SHALL accept a `router_mode: str = "sge"` parameter. `router_mode="baseline"` SHALL
construct `BoundedStrategyCandidateRouter` (exp5534's deterministic, non-LLM router --
`arc_bounded_strategy_router.py`, the SAME class the REQ-ARC-FCP-5699 docstring's very first
sentence contrasts SGE against) in place of `SGECandidateRouter`, under the IDENTICAL policy
config, budget, and game set. `main()` SHALL expose this via a `--baseline` CLI flag, writing to
non-colliding output paths (`outer_loop_sge_smoke_test_baseline_<game>.json` /
`outer_loop_sge_smoke_test_baseline_suite.json`) so the SGE-mode artifacts (including g50t's
backward-compat unsuffixed path) are never overwritten by a control run.

**RESOLUTION (2026-07-15, operator: "run it").** Ran `--baseline` against all 3 games
(`results/outer_loop_sge_smoke_test_baseline_{g50t,sk48,cd82,suite}.json`). **The control ALSO
never leaves level 0 on any of the 3 games** (`real_initial_level=0, real_max_level_observed=0,
leveled_up=false`, confirmed directly against each artifact's raw `action_log`, not just the
summary field, per this project's own Reading-Results Discipline) -- attempts=44-45 per game,
matching the SGE runs' budget exhaustion, but completing in ~2s per game (vs 20-50s for SGE,
since the deterministic router invokes no LLM at all).

**This resolves the open question REQ-ARC-FCP-5699-5 left unanswered: the stripped-down harness
itself -- not SGE specifically -- is what caps every run in this investigation at level 0.** A
completely different candidate-ranking strategy (deterministic template scoring vs. LLM-sampled
natural-language strategies), with zero shared code path other than the `rank()` interface, lands
on the exact same result. The other disabled production features (`_NoOpInductionProposer`,
no frame-change scorer, no goal-bias, no go-explore archive) are apparently load-bearing for even
a FIRST level-up on g50t/sk48/cd82 within a ~45-action budget -- independent of which router
selects among the candidates those disabled/degraded systems still manage to generate. **Every
REQ-ARC-FCP-5699-3/5699-4 "0/3 leveled up" finding should be read in this light: it was never
evidence that SGE fails to add value over a simpler router on these games; it was evidence that
THIS stripped-down test configuration cannot reach level 1 on these 3 games at all, with any
router tried so far.** The router-internal findings (nudge firing correctly, parse-rate
improvements, cd82's strategy-text language shifting toward the advised action type) remain true
and are unaffected by this control -- they describe SGE's own behavior, not its comparative
value against a baseline, which this control now shows cannot yet be assessed on these 3 games
in this harness.

**Open follow-up, not done here:** whether a MUCH longer budget (200-500+ actions) lets either
router eventually escape level 0 on any of these games, and whether re-enabling the other
disabled production features (induction, frame-change scoring, goal-bias) is what's actually
required to reach level 1 here at all -- in which case this specific 3-game/46-budget/stripped-
config harness may not be a useful SGE-vs-baseline comparison ground regardless of budget, and a
different game selection or a less-stripped-down config would be needed to isolate the router's
marginal contribution.

#### SCENARIO-ARC-FCP-5699-6-CONTROL-ISOLATES-THE-ROUTER-UNDER-TEST

Given a smoke-test harness deliberately strips every OTHER exploration feature to isolate ONE
candidate router's contribution
When the harness is run with the router under test AND with a structurally-different control
router, under the otherwise-identical config, budget, and game set
Then a null result (no level-up) with the router under test is NOT attributed to that router
specifically unless the control router also fails to reproduce the null result under the same
control -- if the control ALSO nulls, the null is evidence about the stripped-down harness
configuration itself, not about the router under test, and must be reported as such rather than
as a finding against the router

### REQ-ARC-FCP-5699-7: Budget Was Not The Limiting Factor Either

REQ-ARC-FCP-5699-6's control left an explicit open follow-up: "whether a MUCH longer budget
(200-500+ actions) lets either router eventually escape level 0." Every prior run in this
investigation (SGE and the REQ-ARC-FCP-5699-6 baseline alike) used `budget=46`.

`main()` SHALL accept a `--budget N` flag overriding every selected game's default budget,
writing to a `_budgetN`-suffixed output path so a longer run never overwrites the 46-budget
artifacts it needs to be compared against (`run_game()` already accepted `budget` as a plain
parameter; this is a CLI-level convenience, not a new code path).

**RESOLUTION (2026-07-15, operator: "run it with a longer budget").** Ran BOTH router modes at
`budget=250` (~5.4x the original) against all 3 games:
`results/outer_loop_sge_smoke_test_baseline_{g50t,sk48,cd82,suite}_budget250.json` and
`results/outer_loop_sge_smoke_test_{g50t,sk48,cd82,suite}_budget250.json`. **All 6 runs (3 games x
2 router modes) again show `real_max_level_observed=0`**, confirmed against each artifact's raw
`action_log` (not just the summary field), with 239-248 real attempts per game (near-full budget
consumption, not an early stop) -- SGE completing in 141-486s per game (real LLM calls) and the
baseline in 2.4-5.4s per game (no LLM), matching the ~5.4x wall-clock scaling expected from a
~5.4x action-count increase.

**This closes the "maybe it just needs more budget" hypothesis for this specific harness
configuration.** Neither exploration strategy escapes level 0 on g50t, sk48, or cd82 within 250
actions, any more than within 46. Combined with REQ-ARC-FCP-5699-6's router-independence finding,
the two most obvious explanations for the REQ-ARC-FCP-5699-3/5699-4/5699-6 null results (SGE is
worse than a simpler router; the harness just needed more time) are both now ruled out for these 3
games in this configuration. The remaining, not-yet-tested hypothesis is REQ-ARC-FCP-5699-6's
other open thread: whether one of the OTHER deliberately-disabled production features
(`_NoOpInductionProposer`, no frame-change scorer, no goal-bias, no go-explore archive) is what's
actually load-bearing for a first level-up on these specific games, independent of both router
choice and budget.

#### SCENARIO-ARC-FCP-5699-7-BUDGET-INCREASE-ALONE-DOES-NOT-CHANGE-A-STRUCTURAL-NULL

Given two router modes both null (no level-up) on the same 3 games at the harness's default budget
When both router modes are re-run at a substantially larger budget (>=5x) under the otherwise
identical stripped-down configuration
Then a null that persists unchanged at the larger budget, with near-full budget consumption
confirmed via attempt counts (not an early termination), rules out "insufficient budget" as the
explanation and narrows the remaining hypothesis space to the other deliberately-disabled
components of the configuration, not to the exploration strategy or the time allotted

### REQ-ARC-FCP-5699-8: Re-Enabling Induction -- Blocked By A Different Gate, Not The Harness Disable

REQ-ARC-FCP-5699-6/5699-7 narrowed the remaining hypothesis to "one of the other deliberately-
disabled production features (induction, frame-change scorer, goal-bias, go-explore archive) is
what's actually load-bearing." `run_game()` SHALL accept `induction_enabled: bool = False`;
`True` constructs a real `LocalGGUFProposer` (the SAME defaults `E3AgentPolicy._proposer()`
would lazily build in production -- `Qwen3.5-9B-MTP`, MTP, q8 KV, `/no_think`) on a dedicated
port (8930, distinct from the SGE router's own `gemma-4-12B-it` server on 8929) instead of
`_NoOpInductionProposer`. `main()` SHALL expose `--induction`. Because
`CARNOT_ARC_DISABLE_INDUCTION=1` is set at module scope BEFORE argv parsing normally happens (a
production-safe escape hatch read at call time inside `E3AgentPolicy`, per its own docstring
comment), the `--induction` check SHALL happen via a plain `sys.argv` membership test before that
env-var line, not deferred into `main()`.

**RESOLUTION (2026-07-15, operator: "re-enable induction and run it").** Ran `--induction`
(SGE router, default `budget=46`) against all 3 games --
`results/outer_loop_sge_smoke_test_{g50t,sk48,cd82,suite}_induction.json`. **Still
`real_max_level_observed=0` on all 3 games** -- but `induction_attempts_not_skipped=0` on all 3
too, and direct inspection of each artifact's `induction_attempts` list (E3AgentPolicy's own
real-time induction log, not inferred) shows the LLM induction call was skipped every single time
with `"skipped": "hidden_state_trust_below_threshold"`. This is a DIFFERENT skip reason than the
one this whole investigation had been avoiding (`"disabled_by_env"`, the harness's own escape
hatch) -- confirming `--induction` genuinely bypassed that hatch, and inducton hit a real,
separate, pre-existing production gate instead.

**The gate, traced to source (`arc_competition_agent.py:3601-3617`):** all 3 games (`g50t`,
`sk48`, `cd82`) are members of `HIDDEN_STATE_GAME_IDS`
(`arc_world_model_trust_energy.py:22-32`) -- coincidentally, not by deliberate selection; g50t
was exp5534's original scope and sk48/cd82 were added purely for candidate-space diversity
(REQ-ARC-FCP-5699 extension). For a hidden-state game, `select_trusted_world_model` fits a CNN
dynamics prior from observed transitions and computes a `TrustScore` BEFORE any LLM call is
attempted; `trust_pass` requires `heldout_change_consistency >= threshold` (in addition to a
non-degeneracy check) per `arc_world_model_trust_energy.py:388`. All 3 games' real
`induction_attempts[0]` entries show `heldout_change_consistency` at or near zero (g50t: 0.0,
sk48: 0.0, cd82: 0.0165) after only 25 observed transitions from a cold `budget=46` start --
`trust_pass` fails on the consistency term regardless of the OTHER reported sub-metrics (sk48
even shows `binary_gate_pass: true` yet is still skipped, since `trust_pass` is the stricter,
compound condition the code actually branches on, not `binary_gate_pass` alone).

**This narrows the open hypothesis further, rather than closing it.** "Re-enabling induction"
alone does not change the headline result on THESE 3 games in THIS harness -- but not because
induction was tried and failed to help; because a pre-existing, unrelated production safety gate
(designed to avoid inducing from an untrustworthy dynamics prior) never lets the LLM call happen
at all with only ~25 cold-start transitions. Two distinct, not-yet-tested follow-ups this
surfaces: (1) whether a larger budget specifically increases `transition_count` enough for
`heldout_change_consistency` to clear the threshold naturally (REQ-ARC-FCP-5699-7 already showed
budget alone doesn't change the LEVEL outcome, but that run had induction disabled throughout --
this is a genuinely different question: does budget change whether induction EVER FIRES); (2)
testing on a NON-hidden-state game, where this gate does not apply at all and a real LLM
induction call would actually be attempted from the very first stall.

#### SCENARIO-ARC-FCP-5699-8-INDUCTION-RE-ENABLED-STILL-GATED-BY-TRUST-CHECK

Given a game classified in `HIDDEN_STATE_GAME_IDS` and a harness that re-enables the LLM
induction proposer (bypassing the harness's own disable flag)
When the CNN-fitted dynamics prior's held-out change-consistency has not yet cleared
`trust_pass`'s threshold (typically true very early in a cold-start run with few observed
transitions)
Then the LLM induction call is skipped with an honest, distinct reason
(`hidden_state_trust_below_threshold`) rather than silently defaulting to
`disabled_by_env` -- a harness genuinely re-enabling induction must distinguish "induction never
had a chance to run" (env-disabled) from "induction was allowed to run but a real production
safety gate declined to invoke the LLM this time" (trust-gated), since conflating the two would
misattribute a null result to the wrong cause

### REQ-ARC-FCP-5699-9: Non-Hidden-State Game -- A Different Gate, A Documented Lever That Still Doesn't Clear It

REQ-ARC-FCP-5699-8 found g50t/sk48/cd82 are ALL coincidentally members of `HIDDEN_STATE_GAME_IDS`,
so induction had never been tested where that SPECIFIC gate does not apply. `sp80` (registry:
"Exp4535 ... reached_level=2, banked +1 over the current L1 registry row", the same L1->L2
shallow-frontier framing as sk48/cd82) is NOT in `HIDDEN_STATE_GAME_IDS` and was added to `GAMES`
as the fourth suite member for this test.

**Correctness fix found first (while running sp80 alone).** A single-game `main()` invocation
(`... sp80`, no `--baseline`/`--budget`) wrote its one-game summary to the SAME
`outer_loop_sge_smoke_test_suite.json` path the full 3-game default run uses, silently
overwriting the committed g50t/sk48/cd82 summary with a 1-game summary. Restored via `git
checkout` (uncommitted at the time, safely recoverable) and fixed: the summary filename now
includes a `_<game1>_<game2>...` suffix whenever `requested` (the explicit game-id CLI args) is
non-empty, so a subset run never collides with the full-suite summary path again.

**RESOLUTION (2026-07-15, operator: "do that").** Ran sp80 three ways: (1) baseline (no
`--induction`, matching every other game's default config) -- `real_max_level_observed=0`,
consistent with g50t/sk48/cd82. (2) `--induction` -- still `real_max_level_observed=0`, and
`induction_attempts` shows a DIFFERENT skip reason than REQ-ARC-FCP-5699-8's finding:
`"world_model_accuracy_below_threshold"`, NOT `"hidden_state_trust_below_threshold"` --
confirming sp80 genuinely takes the non-hidden-state code branch
(`arc_competition_agent.py:3620-3636`, `WorldModelVerifier(...).score(engine)` gated at
`< 0.5` on the `CARNOT_ARC_TRUST_METRIC` env var's chosen metric, default `"exact"`). That
branch's own source comment explicitly names itself "the coordinated-redesign lever for the 0.08
wall: exact-match reads ~0 for an imperfect-but-useful induced model and gates it out." (3) Same
`--induction` run WITH `CARNOT_ARC_TRUST_METRIC=cell_recall` set (the documented lever) -- STILL
`real_max_level_observed=0`, STILL skipped with the same reason, and critically `verify_cell_recall:
0.0` in the raw attempt (not just `verify_accuracy: 0.0` under the stricter default metric). **The
documented lever does not apply here**: it exists to rescue an "imperfect-but-useful" induced
model that the strict exact-match metric would unfairly zero out; sp80's candidate engine scores
genuinely 0.0 on the LENIENT graded metric too, meaning the underlying candidate is producing zero
correct held-out predictions regardless of which metric grades it -- a harder floor than metric
strictness, not fixed by switching metrics.

**Net effect on the open question.** Induction still never reaches the actual LLM call on ANY of
the 4 games tested so far in this harness, for THREE distinct, now-documented reasons across two
code branches (`hidden_state_trust_below_threshold` for hidden-state games;
`world_model_accuracy_below_threshold`, both under the default AND the documented alternative
metric, for sp80). Every gate traces to the same root shape: a cheap non-LLM candidate engine
(the CNN prior / DSL-induced baseline) is checked for trustworthiness BEFORE the expensive LLM
induction call is attempted, and with only ~25 transitions from a cold `budget=46` start, none of
the 4 tested games' cheap candidates clear their respective bar. This is consistent with (not
additional evidence against) REQ-ARC-FCP-5699-7's still-open follow-up: whether a substantially
larger budget gives these cheap-candidate pre-checks enough observed transitions to pass, letting
the actual LLM induction call fire for the first time in this whole investigation.

#### SCENARIO-ARC-FCP-5699-9-DOCUMENTED-LEVER-DOES-NOT-RESCUE-A-GENUINELY-ZERO-SCORING-CANDIDATE

Given a pre-LLM-induction trust gate offers an alternative, more lenient scoring metric
specifically to rescue "imperfect-but-useful" candidates that a stricter default metric would
unfairly zero out
When the candidate's score under the LENIENT metric is ALSO genuinely zero (not merely
suppressed by the stricter metric's exactness requirement)
Then switching to the lenient metric does not change the gate's verdict -- the two cases (metric
too strict vs. candidate genuinely worthless) must be distinguished by checking the lenient
metric's own raw value, not assumed equivalent just because a documented lever exists for the
metric-strictness case

### REQ-ARC-FCP-5699-10: Budget Cannot Help Induction Fire -- The Trigger Is Exploration Exhaustion, Not Action Count

REQ-ARC-FCP-5699-9 closed with the one remaining untested combination: does a much larger budget
give the pre-LLM-induction trust-check gates enough observed transitions to clear, letting
induction fire for the first time in this whole investigation? This requirement answers it --
decisively, and by a different mechanism than the question assumed.

**RESOLUTION (2026-07-15, operator: "run it").** Ran `--induction --budget 250` on `sp80` (the
one game/config combination not yet tried at the larger budget) --
`results/outer_loop_sge_smoke_test_sp80_budget250_induction.json`. Still
`real_max_level_observed=0`. But the decisive finding is in `induction_attempts`: **exactly ONE
attempt, at `transition_count=25`** -- byte-identical to the budget=46 run's single attempt --
despite the run consuming 242 real actions (near-full budget=250 consumption, not an early stop).
**Induction was never even given a SECOND chance to try, let alone more transitions to learn
from.**

**Traced to source (`arc_competition_agent.py:3261-3293`):** the induction-attempt path is gated
behind `self.phase == "explore"`, and `_should_enter_induction`'s `stalled` condition is `len(
self.transitions) >= self.explore_budget OR self.explorer.explored_out`. `explored_out` (set at
`arc_competition_agent.py:2254`, `StepwiseExplorer._frontier() is None`) means the underlying
graph-explorer's frontier of UNTESTED candidate states is genuinely EMPTY -- there is nothing new
left to try. This is a property of the game's REACHABLE-STATE GRAPH SIZE from the generic,
domain-blind explorer used in this harness, not of the total action budget allotted. sp80's
reachable frontier from a cold `ActionDiverseLiveGenerator`-driven start apparently exhausts at
~25 transitions regardless of whether 46 or 250 total actions are available -- so the SAME single
stall-triggered induction attempt, at the SAME transition count, fires either way, and the
remaining ~217 actions in the budget=250 run are spent doing something else (post-stall fallback
behavior, not further exploration feeding fresh transitions to the trust gate). Recorded going
forward via a new `explorer_explored_out` field in the artifact (`run_game()`,
`policy.explorer.explored_out`), so future runs can check this directly instead of re-deriving it
from `induction_attempts`' transition count alone.

**This closes the budget hypothesis conclusively, not just empirically.** REQ-ARC-FCP-5699-7
already showed budget doesn't change the LEVEL outcome (induction disabled throughout, so that
result said nothing about induction specifically). This requirement shows budget CANNOT help
induction fire more often either, by construction: the trigger is exploration exhaustion, which
is capped by the explorer's own reachable-state graph size on this game, not by the budget ceiling.
More actions past that exhaustion point are not spent gathering more transitions for the trust
gate to reconsider -- they are spent in whatever non-induction fallback the policy falls back to
after the single stall-triggered attempt is skipped. Every reasonably-cheap lever this
investigation has tried on these games -- router choice (REQ-ARC-FCP-5699-6), budget
(REQ-ARC-FCP-5699-7), induction re-enablement (REQ-ARC-FCP-5699-8/9/10), and the codebase's own
documented trust-metric override (REQ-ARC-FCP-5699-9) -- has now been tried and found not to move
the headline result on this specific 4-game/stripped-config harness. The remaining, genuinely
untested levers are structural, not parametric: re-enabling one of the OTHER still-disabled
production features this harness deliberately strips (the frame-change scorer, goal-bias,
go-explore archive), or accepting that this harness's generic domain-blind explorer simply does
not generate enough distinct transitions on these specific games for ANY trust-gated mechanism to
engage, independent of budget, router, or induction settings.

#### SCENARIO-ARC-FCP-5699-10-EXPLORATION-EXHAUSTION-NOT-BUDGET-GATES-INDUCTION-RETRY

Given an induction-attempt trigger is defined as EITHER a hard budget-exhaustion condition OR the
underlying explorer's frontier being genuinely empty (no untested candidate states remain)
When a game's reachable-state graph, from a generic domain-blind explorer, is exhausted well
before the action budget is
Then increasing the budget alone produces NO additional induction attempts and NO additional
observed transitions feeding the trust gate -- the single stall-triggered attempt occurs at the
SAME transition count regardless of budget, so "give it more budget" cannot be a lever for this
specific failure mode; only reducing the explorer's own exhaustion point (a richer domain-aware
exploration strategy) or accepting the ceiling can change the outcome

### REQ-ARC-FCP-5699-11: Wire SGE Into The Live Path -- Reachable and Selectable, Not the Default

REQ-ARC-FCP-5699-3 through REQ-ARC-FCP-5699-10 built and thoroughly diagnosed the LLM
Strategy-Guided Exploration mechanism entirely inside `scripts/outer_loop_sge_smoke_test.py`, an
OFFLINE diagnostic harness. `arc_llm_strategy_proposer.py` (`SGECandidateRouter`,
`LLMStrategyProposer`) was imported ONLY by that harness and its own experiment/test files --
never by `arc_competition_agent.py`, the live scored-agent entrypoint. Per this project's ARC
Live-Path Reachability Discipline, a solver mechanism the live agent cannot reach produces no
live capability by construction, regardless of how thoroughly its offline behavior is understood.
The live agent's actual `candidate_router` (`_load_submitted_candidate_router()` ->
`CrossGameDiscriminativeCandidateRouter`, `SUBMITTED_AGENT_CONFIG["candidate_router"] ==
"cross_game_discriminative_v3_tiebreaker"`) is a DIFFERENT module entirely.

`_load_submitted_candidate_router(game_id: str = "unknown_game")` SHALL gain a
`SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED` flag (default `False`, matching the
`SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED` precedent -- built, reachable, gated off pending a real
win). When `True`, a new `_load_sge_candidate_router(game_id)` helper SHALL construct
`SGECandidateRouter` wired to a `LocalGGUFProposer` configured IDENTICALLY to `_proposer()`'s own
lazy default (`repo_substr="Qwen3.5-9B-MTP"`, `mtp`, `kv_quant="q8_0"`,
`no_think_prefix="/no_think\n"`, the SAME `CARNOT_ARC_GGUF_PATH`/`CARNOT_ARC_MTP`/`CARNOT_ARC_NGL`
env vars, and no explicit `port=` override -- the class default, `8919`, is the SAME port
`_proposer()`'s own `LocalGGUFProposer` uses). `LocalGGUFProposer._ensure_server()`'s existing
port-based server-reuse (documented elsewhere in this codebase: "reuses ANY already-healthy
server on the configured port regardless of which build backs it") means this and the induction
proposer share ONE warm llama-server automatically, whichever call constructs it first -- NEVER a
second model load, which would risk the Kaggle 16GB VRAM budget the frozen-generator config is
built around. If SGE construction raises for any reason,
`_load_submitted_candidate_router()` SHALL fall through to the existing discriminative-router
path (never propagate the exception, never return `None` just because SGE failed) -- matching
the pre-existing `except Exception: return None` safety pattern for the discriminative router
itself. `game_id` SHALL be threaded from the constructing `E3AgentPolicy`'s `self.short` (the
actual current game), not a placeholder, so SGE's `_context()` prompt names the real game.
`SUBMITTED_AGENT_CONFIG` SHALL gain `"sge_candidate_router_wired": True` and
`"sge_candidate_router_enabled": SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED` for the same
audit/parity-test pattern every other wired-but-optional feature already follows.

**RESOLUTION (2026-07-15, operator: "wire SGE into the live path").** Implemented exactly as
above. Verified: (1) `scripts/arc_orphan_solver_lint.py` now passes clean ("52 modules in the
live closure", up from 51 -- `arc_llm_strategy_proposer.py` is no longer orphaned). (2) All 20
pre-existing + 5 new tests in `tests/python/test_arc_submitted_agent_parity.py` pass, including
the pre-existing `test_shipped_explorer_config_matches_single_source_of_truth` and
`test_req_capstone_4605_live_stack_integrates_only_non_regression_levers` (proving the DEFAULT
live-path behavior is byte-for-byte unchanged -- flag stays `False`, `E3AgentPolicy` constructed
with no override still yields the discriminative router, never `SGECandidateRouter`). (3) The 5
new tests directly verify: SGE disabled by default; `_load_sge_candidate_router()` builds a
correctly-configured `LocalGGUFProposer` (frozen-generator fields, default port 8919); the flag
being flipped on genuinely returns an `SGECandidateRouter` with the right `game_id`; and a
simulated SGE construction failure falls through to the discriminative router rather than
breaking the live path. **This is integration, not validation** -- the flag stays `False`.
Whether SGE actually helps on the real, non-stripped live path (with induction, the frame-change
scorer, and goal-bias all live together, unlike the deliberately-isolated
`outer_loop_sge_smoke_test.py` harness) is a SEPARATE, not-yet-run experiment: a real matched-
budget A/B on the local submission gate or the live scored path, per the flag's own docstring
("Re-enable only after a real matched-budget A/B on the ACTUAL live path shows a win").

#### SCENARIO-ARC-FCP-5699-11-SGE-REACHABLE-BUT-NOT-DEFAULT

Given a candidate-router mechanism was built and thoroughly diagnosed entirely inside an offline
diagnostic harness, never imported by the live agent's actual entrypoint
When the mechanism is wired into the live entrypoint's candidate-router loader behind a
default-`False` flag, reusing the already-loaded frozen generator rather than requiring a second
model
Then the live agent's DEFAULT behavior (flag unset) is provably unchanged (verified by the
pre-existing parity-test suite passing without modification), the mechanism becomes genuinely
reachable (satisfying the live-path-reachability lint), AND it remains gated off until a real
matched-budget comparison on the actual live path -- not the offline diagnostic harness that
built it -- demonstrates a capability win, so integration and validation are never conflated

### REQ-ARC-FCP-5699-12: Real Live-Path A/B -- No Capability Win, Real Cost

REQ-ARC-FCP-5699-11 wired SGE in but left the actual matched-budget A/B on the real (non-stripped)
live path unrun. `_load_submitted_candidate_router()` gains an env-var escape hatch
(`_sge_candidate_router_requested()`: `SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED or
CARNOT_ARC_SGE_CANDIDATE_ROUTER == "1"`) so a subprocess-based measurement can opt into SGE for a
single run without touching the committed module default (subprocess isolation means an in-
process monkeypatch of the module attribute can't reach a spawned measurement process -- the same
reason `CARNOT_ARC_DISABLE_INDUCTION` exists as an env var rather than a code edit).

A new script, `scripts/arc_sge_live_path_ab.py`, runs the actual comparison: both arms construct
genuinely full-production `E3AgentPolicy(game, proposer=None)` -- the SAME `_proposer()`
lazy-induction default and every other constructor default (`frame_change_scorer`, `goal_bias`,
`action_effect_expansion_prior`, etc.) `make_carnot_agent` ships, NOT
`outer_loop_sge_smoke_test.py`'s deliberately-stripped config. The ONLY difference between arms is
`candidate_router` (default discriminative router vs. a hand-built `SGECandidateRouter` pinned to
a port distinct from 8919, the conductor's own concurrent induction proposer at the time this
ran, avoiding request-queuing/contention with that legitimate concurrent process). Scored via
`arc_leaderboard_eval.py`'s own `run_game()` -- the real leaderboard scorer
(`arc_agi.scorecard.EnvironmentScoreCalculator`), zero reimplementation.

**RESOLUTION (2026-07-15, operator: "run the A/B").** Ran on `sp80`, `budget=250` --
`results/arc_sge_live_path_ab_sp80.json`. **Both arms: `levels=0, reached=L0, actions=241,
efficiency=0.0`, byte-identical outcome.** The gap log is identical too:
`{"stuck_at_level": 0, "signature": "no_level_up_within_budget"}` for both. The only measured
difference is cost: `duration_s` 42.9 (discriminative router) vs 165.4 (SGE) -- **SGE is ~3.9x
slower for zero capability difference on this game/budget, on the real production stack.**

**This is a clean, decisive null, consistent with (not contradicted by) every offline finding in
this investigation.** Notably, even the SHIPPED DEFAULT (discriminative router, full production
config, real induction included) never leaves level 0 on sp80 at `budget=250` -- matching
REQ-ARC-FCP-5699-6's offline finding that this specific game's wall is router-independent, now
confirmed on the real live path with the real scorer, not just the stripped diagnostic harness.
Combined with REQ-ARC-FCP-5699-7's exploration-exhaustion finding (the graph-explorer's frontier
exhausts at ~25 transitions regardless of budget) and REQ-ARC-FCP-5699-8/9/10's trust-gate
findings (induction never actually fires on this game either, gated by the same exhaustion), the
full picture is now: sp80's L0 wall is not attributable to router choice, budget, or induction
enablement, on EITHER the stripped offline harness or the real production stack. **Per
`SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED`'s own docstring ("Re-enable only after a real matched-
budget A/B on the ACTUAL live path shows a win"), this result does NOT meet that bar -- the flag
stays `False`.** This closes the REQ-ARC-FCP-5699 chain's central open question (does SGE add
live capability) with an honest, real-path-verified no, on the one game tested; a broader claim
across more games would need more A/B runs, not assumed from this single result.

#### SCENARIO-ARC-FCP-5699-12-REAL-LIVE-PATH-AB-CONFIRMS-NO-WIN-AT-REAL-COST

Given a candidate-router alternative was integrated (REQ-ARC-FCP-5699-11) but never compared
against the shipped default under genuinely full-production policy defaults
When both arms are run matched-budget on the real scorer, differing ONLY in candidate_router
Then an identical outcome (same levels, same reached level, same gap signature) across both arms,
combined with a real wall-clock cost delta, is sufficient grounds to keep the alternative's
enable flag at its default-off value -- integration alone, however clean, never substitutes for
this comparison, and a null result here is exactly as actionable as a positive one (it closes the
question rather than leaving it open)

### REQ-ARC-FCP-5699-13: CORRIGENDUM -- The "Exploration Exhaustion" Finding Was Scoped To The Smoke-Test's Narrow Generator, Not Production

Operator: "continue there" (state-hashing/dedup investigation, following the recommendation to
look at whether `StepwiseExplorer`'s state identity could be aliasing distinct states). Reading
`StepwiseExplorer._hash()` found state identity is `frame_hash(grid_of(frame))` -- a documented,
pre-existing architectural limitation (`ops/verifier_gaps.md` GAP-ARCH-GRID-ONLY-STATE;
`HIDDEN_STATE_GAME_IDS`/`select_trusted_world_model` is the partial mitigation), not a new bug.
sp80 is NOT in `HIDDEN_STATE_GAME_IDS`, making this an unlikely primary explanation for its
specific wall. Investigating further (comparing `StepwiseExplorer._candidates()` -- the actual
live click-candidate generator, `arc_graph_explore.rich_action_candidates()`, `max_click=48`,
salience-sorted, its own docstring documenting a HISTORICAL fix of exactly a naive 12-click cap
-- against `scripts/outer_loop_sge_smoke_test.py`'s harness config) found something more
consequential than the original question.

**The REQ-ARC-FCP-5699-7/8/9/10 "explorer frontier exhausts at ~25 transitions regardless of
budget" finding was measured entirely through `outer_loop_sge_smoke_test.py`, which passes
`action_prior=generator` AND `qd_generator=generator` where `generator =
ActionDiverseLiveGenerator(max_candidates=8)` -- an explicit, hard 8-candidate cap forced onto
EVERY step via the `qd_generator.generate_candidate_pool(...)` override in `_candidates()`.**
Production's `SUBMITTED_QD_GENERATION_ENABLED = False` means `qd_generator` defaults to `None`
(`coerce_qd_generator(None) -> None`) on the real live path, so `_candidates()` never overrides
`rich_action_candidates()`'s own output -- production genuinely uses up to 48 salience-sorted
candidates per frame, not 8. **The REQ-ARC-FCP-5699-12 real live-path A/B (this same session, run
on the identical game, sp80) directly confirms the two stacks behave differently**: its
`navigation_diagnostics` show `reset_replay_steps=6` and `forward_walk_hit_rate=~0.54` across
`actions=241` (near-full `budget=250` consumption) for BOTH arms -- a small reset count relative
to 241 actions is the signature of an explorer that found and rode ONE-ish long, mostly-novel
branch for most of the budget, not one that hit a fast, repeated frontier-exhaustion wall the way
the 8-candidate-capped smoke-test harness did at transition_count=25.

**What this narrows, and what it does NOT narrow.** REQ-ARC-FCP-5699-12's own headline conclusion
(SGE vs. the discriminative router: byte-identical outcome, SGE ~3.9x slower) is UNAFFECTED --
that comparison ran on the real, unstripped production stack directly, so router-choice-doesn't-
matter still holds as measured. What DOES need re-scoping: REQ-ARC-FCP-5699-7's "budget cannot
help" and REQ-ARC-FCP-5699-8/9/10's "induction is trust-gated by exploration exhaustion" findings
were established entirely on the smoke-test's artificially-narrow 8-candidate generator. Whether
the SAME trust-gate/induction-never-fires pattern holds on the real, ~48-candidate production
generator is an OPEN, not-yet-directly-tested question -- REQ-ARC-FCP-5699-12's A/B did not
capture `induction_attempts`/`explorer.explored_out` (unlike the smoke-test script), so this
corrigendum identifies the gap without yet closing it.

**Honest bottom line.** Production's own explorer, with its real (much richer) candidate
generator, still never leveled up sp80 within 250 actions (REQ-ARC-FCP-5699-12) -- so the
headline "sp80 doesn't level up in this budget" finding stands on the real stack too, independent
of the smoke-test's narrower generator. What's now uncertain is WHY, specifically: the smoke-
test's "explorer exhausts its frontier at ~25 transitions" mechanism is confirmed specific to that
harness's 8-candidate cap, not shown (yet) to be what happens on the real 48-candidate stack. A
genuine next step, if this thread continues, is capturing `policy.explorer.explored_out` and
`policy.induction_attempts` from a REAL production run (extending `arc_sge_live_path_ab.py` or a
sibling script) to see whether the real generator's richer candidate pool changes the induction-
trigger picture, rather than assuming the smoke-test's mechanism transfers unmodified.

#### SCENARIO-ARC-FCP-5699-13-DIAGNOSTIC-HARNESS-FINDINGS-DO-NOT-AUTOMATICALLY-TRANSFER-TO-PRODUCTION

Given a diagnostic harness was built to isolate ONE variable (candidate-router choice) by
stripping several OTHER production features to fixed, simplified stand-ins (here: a hard
8-candidate generator replacing production's ~48-candidate salience-sorted one)
When a finding from that harness (frontier exhaustion at a fixed transition count, independent of
budget) is used to explain behavior on the REAL production stack
Then the finding must be re-verified directly on production before being treated as an
explanation for production's behavior -- a mechanism specific to the harness's simplification
(here, an artificially narrow candidate pool) can produce a superficially similar symptom (no
level-up) for a DIFFERENT underlying reason on the real stack, and conflating the two
misattributes the real stack's failure to a mechanism that was never actually exercised there

### REQ-ARC-FCP-5699-14: Closes The 5699-13 Gap -- Real Generator Does Not Exhaust; The Wall Is Downstream, In Plan-Finding And Trust-Verify, Not Exploration

REQ-ARC-FCP-5699-13 closed with an explicit open gap: capture `policy.explorer.explored_out` and
`policy.induction_attempts` from a real production run rather than assuming the smoke-test's
exhaustion mechanism transfers unmodified. `scripts/arc_sge_live_path_ab.py` was extended to
record both fields for each arm immediately after `run_game()` returns (the same `policy` object
mutated during the run, so both attributes are still valid to read post-hoc), then re-run on sp80,
`budget=250`, matching REQ-ARC-FCP-5699-12's configuration exactly so the two runs are directly
comparable (`results/arc_sge_live_path_ab_sp80.json`, both arms overwritten with the instrumented
re-run; `duration_s` 63.18s baseline / 251.65s SGE -- both similar to the original 5699-12 timings,
confirming the instrumentation itself added negligible overhead).

**Finding 1 -- the gap is closed, and the corrigendum's suspicion is confirmed.**
`explorer_explored_out=False` for BOTH arms. The real ~48-candidate `rich_action_candidates()`
generator does NOT hit the frontier-exhaustion wall the smoke-test's 8-candidate-capped
`ActionDiverseLiveGenerator` hit at `transition_count=25`. REQ-ARC-FCP-5699-7/8/9/10's "induction
is trust-gated by exploration exhaustion" framing is now confirmed to be an artifact of that
harness's narrow generator, not a property of the real live-agent stack, closing the open question
5699-13 left unresolved.

**Finding 2 -- induction DOES fire, but for a different, previously-uncharacterized reason than
"never triggers."** Both arms recorded exactly one induction attempt
(`policy.induction_attempts`), `reason="stall"` at `transition_count=25` (matching
`_should_enter_induction`'s `stalled = len(self.transitions) >= self.explore_budget` branch --
i.e. the trigger IS action-count-based stall, same as always documented, and it DOES fire on the
real stack). Reading `arc_competition_agent.py`'s induction handler (~line 3443-3709) alongside
the recorded attempt shows the single attempt actually threads through TWO distinct tiers, and
BOTH decline to produce a plan, for two DIFFERENT reasons:

- **Tier 1 (CNN-dynamics-prior warm-start, `gated_engine_from_transitions` in `arc_live_ttt.py`)
  PASSES its own held-out trust gate** -- `ttt_prior_engine.gate: "PASS"`,
  `heldout_cell_recall=1.0` (baseline) / `0.9794` (sge), both well above the `trust_threshold=0.5`
  bar that gate uses (`trust_metric="cell_recall"` internally, per REQ-ARC-FCP-4715's own docstring
  rationale for using the lenient graded metric here). Per `gated_engine_from_transitions`'s
  return contract, a `"PASS"` gate always returns a non-None `engine` -- so `_eng is not None`
  held, and `e3.plan_in_model` was actually invoked against a TRUSTED engine. It returned no plan
  (`attempt` shows no `engine_source: "ttt_prior_warmstarted"` and `planned: false`), so execution
  fell through past tier 1 without using it.
- **Tier 2 (the DSL/LLM-induced engine gated by `e3.WorldModelVerifier`, since sp80 is not in
  `HIDDEN_STATE_GAME_IDS`) FAILS its trust gate on BOTH metrics it records** --
  `verify_accuracy=0.0` (exact-match, the active `CARNOT_ARC_TRUST_METRIC` default) AND
  `verify_cell_recall=0.0012` (baseline) / `0.0` (sge) -- both near zero, both below the `0.5`
  gate. `attempt["skipped"] = "world_model_accuracy_below_threshold"` is the recorded outcome.
  Notably, switching `CARNOT_ARC_TRUST_METRIC=cell_recall` (the escape hatch REQ-ARC-FCP-4715
  added specifically because "the online CNN can be useful at changed-cell granularity even when
  exact full-grid accuracy is near zero") would NOT have rescued this specific attempt -- tier 2's
  `verify_cell_recall` is also near-zero here, unlike the imperfect-but-useful case that env var
  was designed for.

**What this narrows.** The wall on sp80 is not "induction never gets a chance" (it does, exactly
once per stall, on the real stack) and not "the trust gate always rejects" (tier 1's gate PASSES).
The previously-uncharacterized failure mode is: a dynamics model can pass its own held-out trust
gate and still have `plan_in_model` find no executable plan against it -- a planner-level gap
distinct from every trust-gating explanation REQ-ARC-FCP-5699-7 through -13 considered. Tier 2's
failure is the already-understood trust-gate story, and confirmed genuinely inapplicable to
`cell_recall` rescue here (unlike the case that metric was built for).

**Honest scope limit -- do not over-generalize from this.** This is n=1 game (sp80), n=1 induction
attempt per arm (the budget only reaches one stall trigger). It establishes that "the trusted-tier-1
engine yields no plan" CAN happen on the real stack; it does not establish how often, on how many
games, or whether it is the dominant contributor to the wall generally. Per Sample-Size Rigor
discipline, this is a diagnostic lead, not a headline capability claim.

**Concrete next step if this thread continues.** Investigate why `e3.plan_in_model` finds no plan
against a tier-1 engine whose own held-out gate passed -- e.g. instrument
`_call_plan_in_model`/`plan_in_model` to record WHY it returns empty (search exhausted vs. goal
predicate never satisfied vs. some other structural cause) on this same sp80 trace, and/or repeat
this measurement on 1-2 more games from `ops/arc_solve_registry.yaml`'s unsolved set to see if the
tier-1-passes-but-no-plan pattern recurs.

#### SCENARIO-ARC-FCP-5699-14-EXPLORATION-EXHAUSTION-RULED-OUT-ON-REAL-GENERATOR

Given REQ-ARC-FCP-5699-13 left open whether the real ~48-candidate generator ever hits the same
frontier-exhaustion wall the smoke test's 8-candidate generator hit
When `arc_sge_live_path_ab.py` is extended to record `policy.explorer.explored_out` and
`policy.induction_attempts` and re-run on the real production `E3AgentPolicy` stack (both the
baseline discriminative-router arm and the SGE arm)
Then `explorer_explored_out` is `False` for both arms, closing the gap: exploration exhaustion is
confirmed specific to the smoke-test harness's narrow generator and does not explain production's
sp80 wall

#### SCENARIO-ARC-FCP-5699-14-TIER-1-ENGINE-PASSES-TRUST-GATE-BUT-YIELDS-NO-PLAN

Given the real stack's single stall-triggered induction attempt on sp80 runs the CNN-dynamics-prior
warm-start tier (`gated_engine_from_transitions`) before falling through to the DSL/LLM tier
When the warm-start tier's own held-out cell-recall trust gate passes (`gate: "PASS"`,
`heldout_cell_recall` >= 0.5 threshold, confirmed 0.98-1.0 on this trace)
Then `e3.plan_in_model` may still return no executable plan against that trusted engine, and
execution falls through to the second (DSL/LLM) tier rather than using the passed-gate engine --
a planner-level gap distinct from any trust-gating explanation the prior REQ-ARC-FCP-5699-N chain
considered, and not yet root-caused

### REQ-ARC-FCP-5699-15: Root-Causes The 5699-14 Lead -- The Trusted Engine's Search Runs Out Of Node Budget, Not A Missing Goal Predicate

REQ-ARC-FCP-5699-14 left one question genuinely open: WHY does `e3.plan_in_model` return no plan
against a tier-1 (CNN-dynamics-prior warm-start) engine whose own held-out trust gate just passed?
Its own concrete next step named the fix directly: instrument `plan_in_model` to record why an
empty return happened, rather than guessing.

**Instrumentation added (purely additive, zero behavior change by default).** `plan_in_model()`
(`arc_executable_world_model.py`) gained an optional `diagnostics: Optional[dict] = None` kwarg;
when a caller passes a dict, both the best-first (`goal_energy`-guided) and blind-BFS code paths
populate it with `is_level_complete_was_none` (bool), `nodes_expanded` (int), and
`termination_reason` (`"is_level_complete_none"` / `"plan_found"` / `"max_nodes_reached"` /
`"queue_exhausted"`) immediately before returning. `_call_plan_in_model` in
`arc_competition_agent.py` mirrors the existing `_planner_accepts_goal_energy` pattern with a new
`_planner_accepts_diagnostics` signature check, so passing `diagnostics=None` (every pre-existing
call site) is byte-identical to before. The two production induction tiers (tier 1's
`gated_engine_from_transitions` warm-start at ~line 3455, tier 2's `WorldModelVerifier`-gated
DSL/LLM engine at ~line 3713) now thread a fresh `{}` through and record the result onto
`attempt["ttt_prior_engine_plan_diagnostics"]` / `attempt["plan_diagnostics"]` respectively, so the
diagnostics flow all the way into `policy.induction_attempts` -- already captured by
`arc_sge_live_path_ab.py` since REQ-ARC-FCP-5699-14, so no script changes were needed to observe
this session's answer.

**Re-ran sp80, `budget=250`, identical config to 5699-12/5699-14.** Both arms' single
stall-triggered induction attempt now report, for the tier-1 engine:

```
baseline_discriminative: is_level_complete_was_none=false, nodes_expanded=20008, termination_reason=max_nodes_reached
sge:                     is_level_complete_was_none=false, nodes_expanded=20002, termination_reason=max_nodes_reached
```

**This directly rules out both hypotheses REQ-ARC-FCP-5699-14 raised and answers the question
precisely.** It is NOT `is_level_complete is None` (the goal predicate the CNN-prior tier derived
IS a real, callable function -- `is_level_complete_was_none=false` on both arms). It is also NOT
`"queue_exhausted"` (the search space was not fully enumerated and found empty of goals) -- it is
`"max_nodes_reached"`: the BFS/best-first search consumed its entire `max_nodes=20000` budget
(overshooting slightly to 20002/20008, expected since the `nodes < max_nodes` check is evaluated
once per outer-loop iteration, not per node) with a non-empty frontier still remaining, and never
reached a state satisfying the induced goal predicate within that budget. Both arms landing on
essentially the same node count independently (20008 vs 20002, both from separately-induced CNN
priors with slightly different `heldout_cell_recall`, 1.0 vs 0.9794) is a reproducibility signal,
not noise -- this is a real, budget-bound search-space limit, not a one-off fluke.

**What this narrows.** The tier-1 engine passing its trust gate means the induced one-step
dynamics ARE locally accurate (high held-out changed-cell recall). But `plan_in_model`'s search
operates by repeatedly composing that one-step model forward -- and a model that is locally
accurate per-transition does not guarantee its multi-step rollout stays close enough to reality
(or dense enough near the goal) for a 20,000-node budget to discover a path to the induced goal
predicate. This is a genuinely different failure class than anything the REQ-ARC-FCP-5699-N chain
had characterized before: not exploration exhaustion (5699-13, ruled out), not router choice
(5699-12, ruled out), not a missing/broken goal predicate (5699-15, ruled out this session), but a
search-budget-vs-compounding-model-error limit in the planner itself.

**Honest scope limit.** n=1 game (sp80), n=1 induction attempt per arm, and this session did not
test whether raising `max_nodes` past 20000 would find a plan (that budget could be genuinely too
small, or the induced model's rollout could diverge from reality regardless of budget -- these are
distinguishable but untested hypotheses). Per Sample-Size Rigor discipline, this is a precisely
root-caused lead on one trace, not a generalized claim about the planner's capability ceiling.

**Concrete next step if this thread continues.** Two distinguishable follow-ups, cheapest first:
(a) re-run with `max_nodes` raised well past 20000 (e.g. 100000) on the SAME sp80 trace/tier-1
engine to see if a plan is found given more budget -- if yes, this is a tunable-parameter fix, not
an architecture problem; if the search still exhausts without finding a goal, that points toward
the induced model's multi-step rollout genuinely diverging from reality near the goal region,
which the cheap held-out cell-recall gate (a 1-step metric) cannot detect. (b) repeat this same
diagnostic on 1-2 more unsolved games to see whether `max_nodes_reached` is the dominant
termination reason generally, or specific to sp80's induced dynamics.

#### SCENARIO-ARC-FCP-5699-15-TRUSTED-ENGINE-SEARCH-EXHAUSTS-NODE-BUDGET-NOT-MISSING-GOAL

Given `plan_in_model` gains an optional `diagnostics` dict that records
`is_level_complete_was_none`/`nodes_expanded`/`termination_reason` on every return path, threaded
through `_call_plan_in_model`'s two production induction-tier call sites without changing default
behavior (`diagnostics=None` preserves byte-identical prior calls)
When the real production stack's tier-1 CNN-dynamics-prior engine (already confirmed to pass its
own held-out trust gate) is re-measured on sp80 with this instrumentation, for both the baseline
discriminative-router arm and the SGE arm
Then `is_level_complete_was_none` is `false` (the goal predicate is real and callable) and
`termination_reason` is `"max_nodes_reached"` at `nodes_expanded` ~20000 (baseline: 20008, sge:
20002) for both arms independently -- ruling out a missing/broken goal predicate and ruling out a
fully-enumerated-empty search space, and identifying the actual mechanism as the planner's
node-budget being insufficient (or the induced model's multi-step rollout diverging from reality
before the budget is exhausted) to reach the goal predicate from within the induced model

### REQ-ARC-FCP-5699-16: 5x Budget Does Not Rescue The 5699-15 Wall -- Not A Tunable-Parameter Fix

REQ-ARC-FCP-5699-15 named the cheapest distinguishing follow-up: re-run with `max_nodes` raised
well past 20000 on the same sp80 trace. If a plan is found, the wall is a tunable-parameter limit.
If the search still exhausts, that points toward the induced model's multi-step rollout genuinely
diverging from reality (or the goal region being unreachable within the model as induced), which
the 1-step held-out cell-recall gate cannot detect.

**Implementation.** `_call_plan_in_model` (`arc_competition_agent.py`) gained a DEV-ONLY
`CARNOT_ARC_PLAN_MAX_NODES` environment-variable override, unset in production (byte-identical
default behavior), guarded the same way as the `goal_energy`/`diagnostics` kwargs (a
`_planner_accepts_max_nodes` signature check so a `plan_in_model`-shaped test double without a
`max_nodes` parameter is never broken by the override). When set, it overrides `plan_in_model`'s
`max_nodes=20000` default for both production induction tiers.

**Re-ran sp80, `budget=250`, `CARNOT_ARC_PLAN_MAX_NODES=100000` (5x the default).** Both arms'
tier-1 diagnostics:

```
baseline_discriminative: is_level_complete_was_none=false, nodes_expanded=100015, termination_reason=max_nodes_reached
sge:                     is_level_complete_was_none=false, nodes_expanded=100001, termination_reason=max_nodes_reached
```

`planned=false`, `engine_source=None` for both -- the tier-1 engine still produced no plan, even
at 5x the search budget, again with near-identical overshoot-adjusted node counts across two
independently-induced models (100015 vs 100001, the same reproducibility signature as the 20000
case).

**This rules out the tunable-parameter hypothesis.** Raising `max_nodes` 5x did not find a plan.
Combined with 5699-15's finding that the goal predicate IS real (`is_level_complete_was_none=
false`), the remaining, sharper explanation is that the induced CNN-prior model's multi-step
rollout does not represent a path to its own induced goal predicate that is discoverable within a
budget this large -- i.e. either the model's forward predictions diverge from a self-consistent
trajectory quickly enough that no bounded BFS/best-first search over it reaches the goal (the
model is locally accurate per-step but not globally coherent over the many steps a real solve
requires), or the induced goal predicate itself does not correspond to any state the model's own
transition function can actually produce from `root_grid` (a self-consistency gap between the
learned dynamics and the learned goal condition, not a search problem at all). This session did
not further distinguish between those two variants -- doing so would require inspecting the
model's own predicted trajectories directly (e.g., sampling a rollout and checking how quickly
predicted grids diverge from plausible real dynamics), which is a materially deeper, more
instrumentation-heavy investigation than the last three REQs in this chain and a natural point to
check in with the operator before continuing, rather than open-endedly deepening further on a
single n=1 game.

**Honest scope limit, unchanged from 5699-15: n=1 game (sp80), n=1 induction attempt per arm.**
This is a negative result on the specific "is it just a budget knob" question, decisively answered
for sp80's tier-1 engine -- it does not establish that model-rollout-divergence is the general
explanation for the wall across other games or other induction attempts.

**Concrete next step if this thread continues.** Either (a) inspect the tier-1 model's own
predicted rollout directly -- from `root_grid`, greedily follow the induced `engine`'s own
transitions for a bounded number of steps and check for structural implausibility (repeated
no-op-looking transitions, grids that diverge from any observed real transition's statistics,
etc.) to distinguish "model predicts a coherent-but-wrong world" from "model predicts noise
quickly"; or (b) the cheaper breadth check named in 5699-14/5699-15 -- repeat the
diagnostics-instrumented measurement on 1-2 more unsolved games to see whether `max_nodes_reached`
recurs as the dominant termination reason there too, which would suggest this is a systemic
property of the tier-1 CNN-prior-warm-start path rather than sp80-specific.

#### SCENARIO-ARC-FCP-5699-16-RAISED-BUDGET-DOES-NOT-RESCUE-THE-WALL

Given REQ-ARC-FCP-5699-15 found the tier-1 engine's search exhausts its 20000-node budget without
finding a plan, and named "raise max_nodes and re-test" as the cheapest way to distinguish a
tunable-parameter limit from a deeper model-rollout-divergence issue
When `CARNOT_ARC_PLAN_MAX_NODES=100000` (5x the default) is set and sp80 is re-measured on the same
production stack, both arms
Then `termination_reason` is still `"max_nodes_reached"` (nodes_expanded ~100000 for both arms
independently, `planned=false`) -- ruling out the tunable-parameter-fix hypothesis and sharpening
the remaining explanation toward the induced model's multi-step rollout not representing a
discoverable path to its own goal predicate, a materially different and deeper question than any
prior REQ-ARC-FCP-5699-N step addressed

### REQ-ARC-FCP-5699-17: Breadth Check -- The Tier-1 max_nodes_reached Wall Recurs Identically On Two More Games

REQ-ARC-FCP-5699-15/16 named the cheaper of two remaining follow-ups: repeat the diagnostics-
instrumented measurement on 1-2 more games to see whether `max_nodes_reached` is a systemic
property of the tier-1 CNN-prior-warm-start path, or specific to sp80. No code changes were
needed -- the diagnostics wiring from 5699-15/16 already flows into `policy.induction_attempts`
for any game `arc_sge_live_path_ab.py` is pointed at.

**Ran the identical diagnostic (default `max_nodes=20000`, `budget=250`) on `cd82` and `g50t`** --
both drawn from this REQ chain's original 4-game sample (REQ-ARC-FCP-5699-6 through -10), for
direct comparability. Both are in `HIDDEN_STATE_GAME_IDS`, unlike sp80, so their single induction
attempt routes through the OTHER second-tier gate (`hidden_state_trust_below_threshold` at
~line 3689, not sp80's `world_model_accuracy_below_threshold` at ~line 3708) -- but tier 1's
`gated_engine_from_transitions` warm-start attempt runs FIRST regardless of hidden-state routing
(the check happens earlier, ~line 3448-3468, before the `HIDDEN_STATE_GAME_IDS` branch), so the
same tier-1 diagnostics apply uniformly across both game classes.

```
cd82 baseline: explored_out=False, is_level_complete_was_none=False, nodes_expanded=20014, termination_reason=max_nodes_reached, planned=False
cd82 sge:      explored_out=False, is_level_complete_was_none=False, nodes_expanded=20012, termination_reason=max_nodes_reached, planned=False
g50t baseline: explored_out=False, is_level_complete_was_none=False, nodes_expanded=20005, termination_reason=max_nodes_reached, planned=False
g50t sge:      explored_out=False, is_level_complete_was_none=False, nodes_expanded=20034, termination_reason=max_nodes_reached, planned=False
```

**The pattern recurs identically, on all six arm-measurements across three games now (sp80's two
arms at both the 20000 and 100000 budgets, cd82's two arms, g50t's two arms).** Every single one:
`explorer_explored_out=False` (the real ~48-candidate generator never exhausts -- 5699-13/14
confirmed, still holding), exactly one induction attempt (`reason="stall"` at
`transition_count=25`, the same trigger every time), tier 1's goal predicate genuinely real
(`is_level_complete_was_none=False`), and tier 1's search always ends in `"max_nodes_reached"`
with `planned=False` -- never `"queue_exhausted"`, never a successful plan. The node counts
cluster tightly around the `max_nodes` setting in effect for that run (20005-20034 at the default,
100001-100015 at 5x), consistent with independently-induced-but-similarly-behaved CNN priors
across different games, not a fluke specific to one game's induced dynamics.

**What this narrows.** The tier-1 `max_nodes_reached` wall is now confirmed SYSTEMIC across the
n=3 games tested, not an sp80 idiosyncrasy -- it recurs identically on a second AND third game,
including one structurally different code path (`HIDDEN_STATE_GAME_IDS` routing). This strengthens
REQ-ARC-FCP-5699-15/16's standing hypothesis (the induced model's multi-step rollout does not
represent a discoverable path to its own goal predicate within any tested budget) from a single
n=1 lead to a reproduced, multi-game pattern. What does NOT generalize from this measurement:
WHICH second-tier gate fires after tier 1 fails (game-class-dependent, per `HIDDEN_STATE_GAME_IDS`
membership) -- only tier 1's own exhaustion behavior is shown to be uniform.

**Honest scope limit.** n=3 games out of the 25-game registry, all three drawn from the SAME prior
sample this REQ chain already used (5699-6 through -10) -- not a random or exhaustive sample of
the registry. It is now reasonable to treat `max_nodes_reached` as the LIKELY dominant tier-1
termination reason across the broader corpus, but "likely" is not "confirmed for all 25 games" --
per Sample-Size Rigor discipline, a claim about the full registry would need a wider sample.

**Concrete next step if this thread continues.** The cheap breadth-check avenue is now
well-exercised (3/3 recur); the higher-value remaining avenue is REQ-ARC-FCP-5699-16's other named
follow-up -- inspect the tier-1 model's own predicted rollout directly (greedily follow the
induced `engine`'s transitions from `root_grid` for a bounded number of steps and check for
structural implausibility against real observed transitions) to distinguish "the model predicts a
coherent-but-wrong world that a search can't reach the goal in" from "the model's predictions
diverge into structural noise quickly" -- the deeper, more instrumentation-heavy investigation
5699-16 flagged as a natural checkpoint before continuing.

#### SCENARIO-ARC-FCP-5699-17-TIER-1-WALL-RECURS-ON-TWO-MORE-GAMES

Given REQ-ARC-FCP-5699-15/16 established the tier-1 `max_nodes_reached` wall on sp80 alone (n=1),
and named repeating the diagnostic on more games as the cheap way to check whether it generalizes
When the same diagnostics-instrumented measurement (default `max_nodes=20000`, `budget=250`) is
run on `cd82` and `g50t` -- both `HIDDEN_STATE_GAME_IDS` members, a structurally different
second-tier gate path than sp80
Then all four new arm-measurements (cd82 baseline/sge, g50t baseline/sge) show the identical
tier-1 signature -- `is_level_complete_was_none=False`, `termination_reason="max_nodes_reached"`
at `nodes_expanded` clustered near 20000, `planned=False` -- confirming the wall is systemic
across n=3 games tested (not sp80-specific), while which SECOND-tier gate fires afterward remains
game-class-dependent

### REQ-ARC-FCP-5699-18: Root Cause Found -- Goal-Energy Is Unconditionally Binary (Zero Gradient) For Any Never-Yet-Completed Level, By Construction

REQ-ARC-FCP-5699-17 closed the breadth check; the remaining, higher-value avenue it named was
inspecting the tier-1 model's own predicted rollout directly, to distinguish "coherent but wrong"
from "diverges to noise fast." This is the operator-requested follow-through on that avenue.

**Instrumentation added.** `plan_in_model`'s best-first (`goal_energy`-guided) branch -- the
branch that actually fires in production, since `SUBMITTED_GOAL_GUIDANCE_LAMBDA = 1.0` makes
`goal_energy` non-None by default whenever `is_level_complete` is callable -- already computes
`_h(g)` (the goal-energy value) on every expanded node as the heap priority. Tracking the running
minimum across the whole search is nearly free: `plan_in_model` gained
`initial_goal_energy`/`min_goal_energy_observed` (floats) and `used_goal_energy_search` (bool) in
its `diagnostics` dict (purely additive, same backward-compatible pattern as 5699-15/16). The
signal answers directly: did the search's own heuristic ever consider ANY visited state closer to
the goal than the start?

**Re-ran sp80, `budget=250`, default `max_nodes=20000`.** Both arms:

```
baseline: initial_goal_energy=1.0, min_goal_energy_observed=1.0, nodes_expanded=20008
sge:      initial_goal_energy=1.0, min_goal_energy_observed=1.0, nodes_expanded=20002
```

**`min_goal_energy_observed` exactly equals `initial_goal_energy` -- the search never found a
single state, across 20000+ expanded nodes each, that its own heuristic considered closer to the
goal than the starting grid.** This rules out "coherent but ran out of budget" (which would show
`min_goal_energy_observed` meaningfully below `initial_goal_energy`, even without reaching exactly
0) and instead points at the heuristic itself providing no usable gradient at all.

**Root cause, confirmed by reading `_goal_energy_for_plan` (`arc_competition_agent.py` ~line
3108-3136) and its exemplar source, not guessed from the numbers alone.** The function computes a
GRADED distance (`scale * mean(grid != exemplar)`, a real [0,1] gradient) only when `use_graded =
os.environ.get("CARNOT_ARC_GRADED_GOAL_BIAS") == "1" and exemplar is not None`, where `exemplar =
self._previous_level_complete_grid`. Confirmed via `grep`: that attribute is initialized `None` in
the constructor (~line 2793) and is ONLY ever assigned a real grid at ~line 3020, inside the
level-up-detection handler that captures the just-COMPLETED level's final grid. **For a game that
has never completed even its first level -- exactly sp80/cd82/g50t's situation in every run this
REQ chain has measured -- `_previous_level_complete_grid` is unconditionally `None`, so
`use_graded` is `False` regardless of the `CARNOT_ARC_GRADED_GOAL_BIAS` env var.** With
`exemplar_arr is None`, `_energy(grid)` collapses to the binary fallback: `0.0` if `is_done(grid)`
else `scale` (== `goal_guidance_lambda`, i.e. exactly `1.0`) -- IDENTICAL for every non-goal state.
This exactly reproduces the empirical observation: every one of the 20000+ visited states ties at
energy `1.0`, so the heap's priority ordering provides no goal-directed signal at all; `heapq`
breaks ties by insertion order (the `counter` field), making the "best-first" search functionally
equivalent to a blind traversal for this regime -- the `SUBMITTED_GOAL_GUIDANCE_LAMBDA = 1.0`
"guidance" is silently inert exactly where guidance would matter most.

**This is not the same bug as the 2026-06-25 `proto_graded_goal_bias_ab.json` finding** (which
found the EXPLORER's graded goal bias failed to fire even with the env var set AND an exemplar
present, for lp85's L1-to-L2 transition -- a live bug in an already-eligible case). This REQ's
finding is a level PRIOR to that one: for a first-contact, never-completed level, no exemplar can
possibly exist yet, so graded guidance is structurally inapplicable regardless of whether that
other bug is ever fixed. The two are complementary, not duplicate, findings.

**Why this matters more than a normal parameter tuning gap.** Per the ARC-AGI-3 IS a Live
Hidden-Game Discovery Agent framing (CLAUDE.md), the scored agent's job on every hidden game is
precisely first-contact discovery from a state with zero prior completions -- the exact regime
where this analysis shows goal-energy guidance provides no signal. This is not a corner case of
the live agent's operation; it is close to the modal case for a genuinely novel hidden game.

**Honest scope limit.** This root-causes WHY the search doesn't get closer (confirmed via direct
code reading, not just correlation with the numbers) for the specific `is_done`/tier-1-engine
combination measured on sp80. It does not by itself prove that a graded first-level energy (were
one to be designed -- there is no existing per-level exemplar to fall back to, so this is a
genuinely open design question, not a simple env-var flip) would find a plan; it only establishes
that the CURRENT mechanism provides zero information in this regime, which is a necessary (not
sufficient) condition for the search's failure.

**Concrete next step if this thread continues.** Design and test a first-level-applicable goal
signal that doesn't depend on a completion exemplar -- candidates include: (a) a self-supervised
novelty/coverage energy (reward states not yet seen, cheap and exemplar-free, though it doesn't
target the goal specifically); (b) using the LIVE agent's own explorer-side signals (frame-change
magnitude, score/HUD deltas if any) as a proxy goal-energy inside `plan_in_model`, if such signals
exist for this env; (c) confirming whether fixing the multi-level graded-bias bug from
2026-06-25 (getting SOME level completed at least once) would let subsequent levels benefit from
graded guidance even though level 1 itself cannot.

#### SCENARIO-ARC-FCP-5699-18-GOAL-ENERGY-IS-BINARY-WITH-ZERO-GRADIENT-FOR-FIRST-CONTACT-LEVELS

Given `plan_in_model` gains `initial_goal_energy`/`min_goal_energy_observed`/
`used_goal_energy_search` diagnostics tracking the goal-energy heuristic's value across every
visited state in a failed search
When sp80 (a game that has never completed level 0 in any measurement this REQ chain has run) is
re-measured with this instrumentation, for both arms
Then `min_goal_energy_observed` exactly equals `initial_goal_energy` (both `1.0`) -- the search
never found any state its own heuristic considered closer to the goal than the start -- and
reading `_goal_energy_for_plan`'s source confirms why: its graded-distance branch requires
`self._previous_level_complete_grid`, which is unconditionally `None` until a level has been
completed at least once, so for any first-contact level the function structurally collapses to a
binary 0.0-at-goal/1.0-elsewhere energy that provides the best-first search with zero gradient,
regardless of the `CARNOT_ARC_GRADED_GOAL_BIAS` env var

### REQ-ARC-FCP-5699-19: Novelty Goal-Energy Fallback For First-Contact Levels -- Real Gradient, Still No Plan; Plus A Self-Caught Test-Fixture-Realism Bug

REQ-ARC-FCP-5699-18 named a genuinely open design question: first-contact levels have no
completion exemplar, so a graded goal-energy is structurally inapplicable there -- what, if
anything, could give the search SOME gradient without one? This REQ implements and empirically
tests candidate (a) from that REQ's next-step list: a self-supervised novelty/coverage energy.

**Design.** `E3AgentPolicy` gained `_novelty_observed_stack()`: a stack of every real (before,
after) grid observed so far this episode (`self._active_transitions()`), shape-filtered for
consistency. `_goal_energy_for_plan` gained a THIRD branch, tried only when the graded-exemplar
branch is inapplicable (`use_graded` False): if `CARNOT_ARC_NOVELTY_GOAL_BIAS=1` is set (DEV-ONLY,
unset in production -- opt-in pending empirical validation, matching this REQ chain's established
discipline for every prior env-var-gated addition) and observed grids exist, the energy for a
candidate grid becomes `scale * (1.0 - min_diff_to_observed)`, where `min_diff_to_observed` is the
normalized Hamming distance to the NEAREST already-observed real grid. States identical to
something already concretely seen get the SAME flat energy as the pre-existing binary fallback
(never worse); states far from everything seen get low energy (attractive to the min-heap search)
-- a go-explore-flavored, execution-grounded proxy for "unexplored territory is more likely to
contain progress," not a claim about the actual unknown goal. The returned closure's
`energy_source` attribute (`"graded_exemplar"` / `"novelty"` / `"binary"`) is threaded into
`_call_plan_in_model`'s `diagnostics["goal_energy_source"]` so any A/B run can confirm which
branch actually fired.

**Self-caught bug (a genuine test-fixture-realism finding, not glossed over).** The first live
validation attempt found `goal_energy_source` stayed `"binary"` even with
`CARNOT_ARC_NOVELTY_GOAL_BIAS=1` set. Root cause: `_novelty_observed_stack()` used index access
(`t[0]`, `t[3]`) against real `Transition` objects, but `Transition`
(`arc_executable_world_model.py`) is a `@dataclass`, NOT a namedtuple -- index access silently
raises `TypeError`, caught by a broad `except Exception: continue`, leaving `grids=[]` on every
real transition. The 8 unit tests written alongside the implementation all passed despite this,
because they constructed `pol.transitions` from plain tuples (`(before, action, data, after)`)
rather than real `Transition` objects -- an unrealistic fixture that happened to support index
access where the real dataclass doesn't, masking the bug from the test suite entirely. **Fixed
both the implementation** (`.grid`/`.next_grid`, the real dataclass field names, confirmed against
existing usage at `arc_competition_agent.py:3296-3298`) **and the test fixtures** (a `_transition()`
helper builds real `Transition` objects everywhere). Verified the corrected test suite actually
detects the class of bug it's meant to catch: temporarily reverted the implementation fix, ran
`test_req_arc_fcp_5699_19_novelty_fires_when_enabled_and_no_exemplar`, confirmed it fails with
`AssertionError: assert 'binary' == 'novelty'`, then re-applied the fix and reconfirmed all 32
tests pass. This is the exact QA-Layer-Authenticity-Discipline pattern (CLAUDE.md) applied to a
NEW check rather than an existing one: don't trust a green suite without confirming it would have
gone red on the bug it claims to guard against.

**Live A/B re-run, sp80, `budget=250`, `CARNOT_ARC_NOVELTY_GOAL_BIAS=1`, WITH the fix applied:**

```
baseline: goal_energy_source=novelty, initial_goal_energy=1.0, min_goal_energy_observed=0.8875, nodes_expanded=20016, termination_reason=max_nodes_reached, planned=False, duration_s=429.5
sge:      goal_energy_source=novelty, initial_goal_energy=1.0, min_goal_energy_observed=0.6765, nodes_expanded=20016, termination_reason=max_nodes_reached, planned=False, duration_s=400.1
```

**Honest read -- a genuine partial validation, not a full one.** `goal_energy_source=novelty`
confirms the fix works on the real live path (the first live-run's confirmation that unit tests
alone were insufficient). `min_goal_energy_observed` is now MEANINGFULLY BELOW `initial_goal_energy`
for both arms (0.8875 and 0.6765, vs the binary case's exactly-flat 1.0/1.0 measured in
REQ-ARC-FCP-5699-18 on the same game) -- the search genuinely found real gradient this time, states
it considered closer to a novel/unexplored region than the start. **This does not translate into a
plan or a level-up**: `planned=False` and `termination_reason=max_nodes_reached` are unchanged from
the binary case, `levels=0`/`reached=0` for both arms. Even the MOST novel state found among
20000+ visited states (min_diff ~0.68-0.89 from `1.0`, i.e. energy 0.68-0.89, not close to `0.0`)
did not happen to satisfy `is_level_complete`. **Real, measurable cost**: wall-clock roughly
tripled to quadrupled versus the binary baseline's prior timings on this same game (429.5s/400.1s
here vs 63-168s/101-251s previously) -- the per-candidate numpy distance computation against the
observed-grid stack is real overhead, not free.

**What this narrows.** Novelty energy closes the "zero gradient" gap REQ-ARC-FCP-5699-18
identified -- the search is no longer flying blind by its own heuristic's measure. It does NOT
close the "no plan found" gap: providing SOME gradient is a necessary-feeling but empirically
NOT sufficient condition for this search to find an executable path to an unknown goal within a
20000-node budget, at least on this one game/trial. The baseline and sge arms' differing
min-energy values (0.8875 vs 0.6765) likely reflect their independently-collected observed-grid
sets (different candidate routers, different real transitions before the stall trigger) rather
than a meaningful router effect -- n=1 per arm, not a claim about router quality.

**Honest scope limit.** n=1 game (sp80), n=1 induction attempt per arm, opt-in DEV-ONLY flag
(production unaffected by default). Whether novelty energy helps on OTHER games, whether a
cheaper/vectorized implementation would change the cost/benefit tradeoff, and whether COMBINING
novelty energy with a larger `max_nodes` budget (REQ-ARC-FCP-5699-16 already showed 5x budget
alone doesn't help under binary energy) would together find a plan, are all untested.

**Concrete next step if this thread continues.** (a) Cheapest: combine novelty energy with a
raised `max_nodes` (via `CARNOT_ARC_PLAN_MAX_NODES`, already implemented) on the same sp80 trace --
now that real gradient exists, more budget might behave differently than it did against a flat
heuristic. (b) Vectorize/cache the novelty computation (e.g. precompute once per unique candidate
rather than per heap-pop) to address the 3-4x wall-clock cost before considering wider use. (c) The
breadth check named in 5699-17's pattern -- repeat on cd82/g50t to see if the partial-gradient,
no-plan result recurs.

#### SCENARIO-ARC-FCP-5699-19-NOVELTY-ENERGY-PROVIDES-REAL-GRADIENT-BUT-NOT-A-PLAN

Given REQ-ARC-FCP-5699-18 found goal-energy is unconditionally flat (zero gradient) for any
first-contact level, and named a self-supervised novelty energy as a candidate fallback
When `_goal_energy_for_plan` gains a novelty branch (opt-in via `CARNOT_ARC_NOVELTY_GOAL_BIAS=1`,
scoring candidates by distance to the nearest already-observed real grid) and sp80 is re-measured
with it enabled, for both arms
Then `goal_energy_source=novelty` confirms the branch fired, and `min_goal_energy_observed` is
meaningfully below `initial_goal_energy` (0.8875 and 0.6765 vs the flat 1.0/1.0 the binary
fallback produced on the same game) -- real gradient now exists -- but `planned` stays `False` and
`termination_reason` stays `max_nodes_reached` for both arms: providing gradient did not, on this
trial, translate into finding an executable plan, and wall-clock cost roughly tripled to
quadrupled versus the binary baseline

#### SCENARIO-ARC-FCP-5699-19-UNREALISTIC-TEST-FIXTURES-MASKED-A-REAL-DATACLASS-INDEXING-BUG

Given `_novelty_observed_stack()` was implemented using index access (`t[0]`/`t[3]`) against
`self._active_transitions()`'s elements, and 8 unit tests were written using plain-tuple fixtures
that happen to support index access
When the implementation is exercised on the REAL live path, where `_active_transitions()` returns
real `Transition` `@dataclass` objects (which do NOT support index access -- only attribute
access)
Then every transition's index access silently raises `TypeError`, caught by a broad
`except Exception: continue`, leaving zero observed grids and silently falling back to the binary
energy despite `CARNOT_ARC_NOVELTY_GOAL_BIAS=1` being set -- a bug the unit tests could not have
caught because their fixtures did not match production's actual data shape, only caught by
validating against the real live path per this REQ chain's established discipline of never
trusting a diagnosis without a live confirmation run

### REQ-ARC-FCP-5699-20: Combining Novelty Energy With 5x Budget Does Not Help Further -- The Search Plateaus

REQ-ARC-FCP-5699-19's own next-step list named the cheapest remaining combination test: now that
novelty energy gives the search real gradient (unlike the flat binary case), does adding
REQ-ARC-FCP-5699-16's already-implemented `CARNOT_ARC_PLAN_MAX_NODES` override on top change the
outcome? Neither env var needed new code -- both were already independently wired through
`_call_plan_in_model`.

**Re-ran sp80, `budget=250`, `CARNOT_ARC_NOVELTY_GOAL_BIAS=1` AND `CARNOT_ARC_PLAN_MAX_NODES=100000`
together:**

```
baseline: goal_energy_source=novelty, min_goal_energy_observed=0.88671875, nodes_expanded=100035, termination_reason=max_nodes_reached, planned=False, duration_s=124.0
sge:      goal_energy_source=novelty, min_goal_energy_observed=0.671142578125, nodes_expanded=100014, termination_reason=max_nodes_reached, planned=False, duration_s=218.6
```

**Decisive negative result: 5x more search budget barely moved the minimum energy found at all.**
Compared to the novelty-only, 20000-node run on the same game (REQ-ARC-FCP-5699-19:
`min_goal_energy_observed` 0.8875 baseline / 0.6765 sge), the 100000-node run found
0.88671875 / 0.671142578125 -- a change of -0.0008 and -0.0054 respectively, both far smaller
than noise-worthy. `nodes_expanded` scaled correctly to the new budget (~100000 vs ~20000),
confirming the override took effect; `termination_reason` stayed `max_nodes_reached` (not
`queue_exhausted` -- the heap still had unexplored frontier when the budget ran out), `planned`
stayed `False`, `levels`/`reached` stayed `0` for both arms.

**What this means.** The search's achievable minimum novelty-energy under this induced model
appears to have effectively PLATEAUED somewhere in the first ~20000 nodes -- the additional 80000
nodes of search found almost nothing meaningfully more novel than what the first 20000 already
found. This rules out "just needed more search to find a genuinely novel/winning region" as an
explanation: the ceiling is not a budget problem, it looks like a property of the reachable state
space under THIS specific induced dynamics model and novelty scoring, at least on this trace. This
combines with REQ-ARC-FCP-5699-16's earlier, independent finding (5x budget alone, under the
BINARY energy, also didn't help) into a consistent picture: more search budget is not the lever
that unlocks a plan here, under either energy function tested so far.

**Incidental observation (reported honestly, not chased further).** Wall-clock for this combined
run (124.0s/218.6s) was LOWER than the novelty-only 20000-node run (429.5s/400.1s,
REQ-ARC-FCP-5699-19) despite visiting 5x more nodes. This is very likely GPU/system load variance
between runs (concurrent conductor activity on the shared GPU throughout this session), not a
property of the node-budget increase itself -- noted for completeness, not investigated further,
since it doesn't bear on the plan-finding question this REQ set out to answer.

**Honest scope limit.** Still n=1 game (sp80), n=1 induction attempt per arm. The plateau
observation is specific to this trace's induced dynamics model and its particular set of
already-observed real grids; it does not establish that budget is universally unhelpful across
all games or all induced models.

**Concrete next step if this thread continues.** The two cheap levers named across
REQ-ARC-FCP-5699-15 through -20 (raise budget, add gradient) have now both been tried alone and
combined, without finding a plan on sp80. The remaining avenues from REQ-ARC-FCP-5699-16's
original list are the more invasive ones: inspect the tier-1 model's own predicted rollout for
structural plausibility against real transitions (does the induced dynamics model diverge from
anything resembling reality as it's followed forward?), or step back and question whether tier 1's
CNN-dynamics-prior-warm-start mechanism itself is well-suited to first-contact levels at all,
versus falling through faster to a different induction tier.

#### SCENARIO-ARC-FCP-5699-20-BUDGET-PLUS-GRADIENT-COMBINED-STILL-PLATEAUS

Given REQ-ARC-FCP-5699-19 found novelty energy gives the search real gradient (`min_goal_energy_
observed` below `initial_goal_energy`) but no plan at the default 20000-node budget, and
REQ-ARC-FCP-5699-16 separately found 5x budget alone (under binary energy) didn't help either
When both levers are combined -- `CARNOT_ARC_NOVELTY_GOAL_BIAS=1` AND
`CARNOT_ARC_PLAN_MAX_NODES=100000` -- and sp80 is re-measured, both arms
Then `min_goal_energy_observed` changes by less than 0.006 from the 20000-node novelty-only value
despite `nodes_expanded` scaling 5x (to ~100000) -- the search's achievable minimum has
effectively plateaued -- and `planned` stays `False` for both arms, establishing that neither lever
alone nor combined finds a plan on this trace

### REQ-ARC-FCP-5699-21: Tier 2's LLM-Synthesized Dynamics Model Has Never Succeeded, Not Marginally -- Complete Failure Every Time Measured

Every prior REQ in this chain (5699-14 through -20) diagnosed tier 1 (the CNN-dynamics-prior
warm-start). Tier 2 -- the DSL/LLM induction path, gated by `WorldModelVerifier` (non-hidden-state
games) or `select_trusted_world_model` (`HIDDEN_STATE_GAME_IDS` games) -- was treated as a black
box that "also fails" without characterizing HOW. Reading the code between the two tiers
(`arc_competition_agent.py` ~line 3660-3712) clarifies tier 2 is NOT skipped or unreached when
tier 1 fails: in the SAME induction attempt, `self._proposer().induce(self.short,
active_transitions, self.cell)` makes a REAL call to the live LLM proposer (the same
`LocalGGUFProposer`/Qwen3.5-9B-MTP the `arc_sge_live_path_ab.py` A/B script's `proposer=None`
default lazily constructs) to SYNTHESIZE Python code implementing the game's dynamics + win
predicate, then `engine, is_done = e3.load_engine(self.short)` loads it, before the trust gate
scores it. So tier 2 genuinely runs an LLM call and gets genuine synthesized code every attempt --
the open question is how well that code actually predicts reality, which this REQ answers directly
from artifacts THIS SESSION ALREADY COLLECTED (no new live run needed).

**Pulled every measurement across the whole REQ chain's artifacts** (`arc_sge_live_path_ab_sp80/
cd82/g50t.json`, all baseline + SGE arms, all 4 sp80 runs -- binary, novelty-only, novelty+100k
budget):

```
sp80  baseline/sge (all 4 sp80 runs): verify_accuracy=0.0, verify_cell_recall=0.0
cd82  baseline: heldout_change_consistency=0.0, heldout_accuracy=0.0, correct_changed_cells=0, binary_gate_pass=False
cd82  sge:      heldout_change_consistency=0.0, heldout_accuracy=0.125, correct_changed_cells=0, binary_gate_pass=False
g50t  baseline: heldout_change_consistency=0.0, heldout_accuracy=0.0, correct_changed_cells=0, binary_gate_pass=False
g50t  sge:      heldout_change_consistency=0.0, heldout_accuracy=0.125, correct_changed_cells=0, binary_gate_pass=False
```

**Tier 2's LLM-synthesized dynamics model has never succeeded -- and it is not a marginal, near-
miss failure. It is a COMPLETE failure by the most direct metric.** `correct_changed_cells=0` in
ALL FOUR hidden-state measurements (cd82/g50t, both arms) -- the synthesized code predicts ZERO
correctly-changed cells across every held-out transition tested, not "some but not enough."
`verify_cell_recall=0.0` in both sp80 measurements -- same complete failure by the graded metric.
`heldout_change_consistency=0.0` across the board. The one non-zero number, `heldout_accuracy=
0.125` on the SGE arm for cd82/g50t (1 of 8 held-out transitions scored "correct"), is very likely
a NO-OP held-out transition (agent action that didn't change the grid) coincidentally matching a
degenerate always-predict-no-change synthesized function -- consistent with `correct_changed_cells`
staying `0` even there (a genuinely correct model would get SOME changed cells right if it scored
above chance on anything).

**What this narrows.** This session's entire diagnostic effort (5699-14 through -20) characterized
tier 1's search-mechanics failure in detail. Tier 2's failure is a DIFFERENT, more fundamental
problem: the LLM is not producing dynamics code that predicts these games' mechanics AT ALL, on
first contact, on any of the 3 games measured. This is not a search-budget problem, not a
gradient problem, and not (necessarily) a search-mechanism problem at all -- it's a code-synthesis
correctness problem, upstream of anything this REQ chain has touched so far.

**Honest scope limit.** n=3 games, and this REQ characterizes tier 2's failure MAGNITUDE from
already-recorded numeric trust metrics -- it does not yet inspect the actual LLM PROMPT or
GENERATED CODE to diagnose WHY the synthesis fails so completely (bad game understanding from too
few transitions, a systematic code-generation bug, a prompt/context issue, or the games genuinely
being hard to infer dynamics for from 25 transitions). That is qualitatively different work
(reading LLM I/O, not measuring search statistics) from everything else in this REQ chain.

**Concrete next step if this thread continues.** Capture and read the ACTUAL prompt +
LLM-generated code for one attempt (e.g. add a diagnostics field logging the raw proposer output,
or intercept `self._proposer().induce(...)`'s return in a dedicated diagnostic script) to see
whether the synthesized code is plausible-but-wrong (a genuine game-understanding miss) or
structurally broken (a code bug unrelated to game understanding) -- this determines whether the
fix is "give the LLM better/more transitions" or "fix a proposer bug," which are very different
follow-ups.

#### SCENARIO-ARC-FCP-5699-21-TIER-2-LLM-SYNTHESIS-COMPLETELY-FAILS-EVERY-MEASUREMENT

Given tier 2 is exercised (a real LLM call synthesizes dynamics code) in every induction attempt
this REQ chain has measured across sp80, cd82, and g50t, gated by `WorldModelVerifier` or
`select_trusted_world_model`
When every already-collected artifact's trust metrics are read together (not just each game's
own headline `skipped` reason, which this REQ chain had treated as a single opaque gate failure)
Then `correct_changed_cells=0` (hidden-state games, all 4 measurements) and `verify_cell_recall=
0.0` (sp80, both measurements) show the synthesized dynamics model predicts ZERO correctly-changed
cells in every single measurement -- a complete failure, not a marginal near-miss -- establishing
that tier 2's problem is upstream code-synthesis correctness, a qualitatively different failure
class from tier 1's search-mechanics wall this REQ chain spent 5699-14 through -20 characterizing

### REQ-ARC-FCP-5699-22: Reading The Actual Generated Code -- Not An Execution Bug; Two Precise, Verified Root Causes, One Of Them Shared With Tier 1

REQ-ARC-FCP-5699-21's own next step, and the operator's direct request, was to read the actual
LLM-generated code rather than reason from aggregate trust metrics alone. Going in, the leading
hypothesis (stated at the top of this REQ, based on the SUSPICIOUSLY uniform zero-everywhere
failure pattern) was that this looked more like an execution/plumbing bug than genuine LLM
incapability. **Reading the real generated code overturns that hypothesis.**

**The generated code is real, syntactically valid, and (mostly) non-crashing.** The induced
`world_model.py` for each game (`results/arc_e3/{sp80,cd82,g50t}/world_model.py`, all written by
real LLM calls during this session's own runs) was read directly:

- **sp80**: a full, plausible-LOOKING hypothesis -- cardinal + one diagonal movement action, a
  click-to-clear action, win = "all cells empty." Syntactically correct, would execute cleanly on
  any grid. Simply wrong against held-out transitions (hence `verify_cell_recall=0.0`).
- **cd82**: `def is_level_complete(grid): return False` -- an UNCONDITIONAL, literal `False`. This
  predicate can never be satisfied by construction, regardless of the grid.
- **g50t**: `engine()` hardcodes LITERAL, ABSOLUTE observed coordinates per action (e.g. `if
  action == 1: grid[63, 62] = 1`) instead of inferring a relative/general movement rule -- clear
  memorization of specific instances, not generalization, despite the prompt's own explicit
  instruction ("Prefer SIMPLE GENERAL rules over per-frame special cases").

None of these are crashes or plumbing failures. All three are genuine (if bad) reasoning
artifacts. **Reading `induce_prompt`/`_transitions_block` (`arc_executable_world_model.py`
~line 1054-1169) explains BOTH pathologies precisely, with root causes verified from the source,
not inferred from symptoms:**

1. **The dynamics half is starved by a hard `k=8` transition cap.** `induce_prompt` calls
   `_transitions_block(trans, previous_level_complete_grid=...)` with NO `k` override, so the
   DEFAULT `k=8` applies -- `sample = changed[:k-2] + noop[:2]` shows the LLM at most **6
   grid-changing transitions**, out of the full 25 collected before the stall trigger. Spread
   across ARC-AGI-3's typical 6-8 distinct action types, that is roughly ONE example per action
   type on average -- genuinely insufficient to distinguish "this action moves the player by a
   RELATIVE offset" from "this action sets THIS SPECIFIC ABSOLUTE cell," which requires observing
   the SAME action fire from at least two DIFFERENT starting positions. g50t's hardcoded-coordinate
   engine is exactly the signature this data-starvation would produce: with one example per
   action, memorizing the literal observed cell IS the maximum-likelihood hypothesis a reasoner
   could produce, general-rule instruction notwithstanding.

2. **The goal-predicate half is starved of ANY positive example on first-contact levels -- the
   SAME structural gap REQ-ARC-FCP-5699-18 found for tier 1, now shown to independently cripple
   tier 2 too.** `_transitions_block` only includes a WIN STATE block via one of two paths: `win =
   next((t for t in trans if t.level_after > t.level_before), None)` (a transition that actually
   leveled up -- impossible by construction for a first-contact level, since no level-up has ever
   happened yet) OR `previous_level_complete_grid` (REQ-ARC-FCP-5699-18: unconditionally `None`
   until a level has completed at least once). **For sp80/cd82/g50t, BOTH are always `None` --
   the LLM's prompt contains ZERO positive win-state information.** Given zero positive examples,
   `is_level_complete: return False` (cd82's actual output) is close to the epistemically-honest
   answer: with no evidence of what winning looks like, a reasoner with no positive signal cannot
   construct a non-trivial predicate that isn't itself a guess. This is the SAME `_previous_level_
   complete_grid is None` root cause REQ-ARC-FCP-5699-18 diagnosed for tier 1's flat goal-energy --
   confirmed here to independently starve a SECOND, structurally different component (tier 2's
   goal-predicate induction, not tier 1's search heuristic) via the same upstream first-contact
   data gap.

**What this narrows.** The original hypothesis going into this REQ (execution/plumbing bug) is
REFUTED by direct evidence -- explicitly correcting course rather than defending the prior
framing. The real picture is sharper and more useful: first-contact levels structurally lack ANY
win-state example, and this single upstream gap independently cripples TWO different downstream
mechanisms across BOTH induction tiers (tier 1's goal-energy gradient, REQ-ARC-FCP-5699-18; tier
2's goal-predicate induction, this REQ). Tier 2's dynamics (engine) failure is a SEPARATE, milder
problem (data thinness from the `k=8` cap, not a first-contact-specific structural gap -- more
transitions would exist even for a first-contact level, they're just not all shown).

**Honest scope limit.** n=3 games' generated code read directly (real, verified evidence, not
inferred). The dynamics-half root cause (the `k=8` cap) is verified from source and matches g50t's
symptom precisely; it has NOT been empirically tested (would raising `k` actually produce more
general rules, or does the underlying game genuinely need more real transitions collected before
attempting induction at all -- a budget/exploration question, not a prompt one). The goal-predicate
starvation explanation for cd82's `return False` is strongly evidenced (the prompt path is proven
to supply zero win information) but not proven to be the SOLE cause (an LLM could in principle
produce a degenerate predicate even with a positive example).

**Concrete next step if this thread continues.** (a) Cheapest: raise `_transitions_block`'s `k`
for the induce-prompt call (currently uncapped-default 8) and re-measure whether the dynamics half
stops memorizing literal coordinates -- directly testable, matches the identified root cause. (b)
Harder, and likely the more fundamental fix: give tier 2 SOME positive goal signal for
first-contact levels analogous to what REQ-ARC-FCP-5699-19's novelty energy did for tier 1 --
e.g., prompt the LLM to propose a CANDIDATE goal predicate from structural/visual regularities
(a common color/shape becoming absent, a counter reaching zero) rather than leaving it with
literally nothing, since "no positive example" is unfixable by more transitions alone (transitions
before the first win are, by definition, all non-win states).

#### SCENARIO-ARC-FCP-5699-22-GENERATED-CODE-IS-REAL-NOT-A-CRASH-BUT-TWO-VERIFIED-DATA-STARVATION-ROOT-CAUSES

Given the operator asked to read the actual LLM-generated dynamics code (rather than continue
reasoning from aggregate trust metrics) to distinguish a genuine reasoning failure from a
structural/execution bug
When `results/arc_e3/{sp80,cd82,g50t}/world_model.py` (real code from this session's own live
induction attempts) is read directly, alongside `induce_prompt`/`_transitions_block`'s source
Then the code is syntactically valid and non-crashing in every case (refuting the leading
execution-bug hypothesis), and two precise, source-verified root causes explain the observed
failures: (1) `_transitions_block`'s uncapped-default `k=8` shows the LLM at most 6 grid-changing
transitions of the 25 collected, roughly one per action type -- explaining g50t's hardcoded-
literal-coordinate memorization directly; (2) the win-state block in the prompt requires either a
level-up transition (impossible on a first-contact level by construction) or `previous_level_
complete_grid` (REQ-ARC-FCP-5699-18: unconditionally `None` until a level completes once) --
so first-contact levels supply ZERO positive win-state information, explaining cd82's `return
False` predicate as close to the epistemically honest answer given no evidence -- the SAME
upstream first-contact data gap REQ-ARC-FCP-5699-18 found for tier 1, now shown to independently
cripple tier 2's goal-predicate induction too

### REQ-ARC-FCP-5699-23: Testing The Cheap Fix -- Raise The Transitions-Shown Cap

REQ-ARC-FCP-5699-22's cheapest concrete next step: raise `_transitions_block`'s `k` (the number of
grid-changing transitions shown to the LLM in the induce prompt, uncapped-default 8 -> ~6 shown of
25 collected) and re-measure whether the dynamics half stops memorizing literal coordinates.

**Implementation.** `induce_prompt` gained an optional `k: int = 8` kwarg (matching
`_transitions_block`'s own pre-existing default -- every call site that doesn't pass `k` is
byte-identical to before, verified by a dedicated regression test asserting
`induce_prompt(..., k=8) == induce_prompt(...)` with no `k` arg). A new `_induce_transitions_k()`
helper reads a DEV-ONLY `CARNOT_ARC_INDUCE_TRANSITIONS_K` env var (unset -> returns 8, unchanged
production default), threaded into both `CodexProposer.induce()` and `LocalGGUFProposer.induce()`
(the two real proposer call sites; `induce_programmatic_experts`'s own, unrelated
`_transitions_block` call was left untouched -- out of scope, a different induction path). 4 new
unit tests (`test_arc_induce_prompt_large_grid_scalability.py`) verify the env-var default/
override, the byte-identical-when-unset regression guard, and that raising `k` genuinely surfaces
more transitions in the rendered prompt (synthetic 25-transition pool, `k=8` renders 6 changed
transitions, `k=20` renders 18 -- confirms the cap is the binding constraint being tested, not an
artifact of too-small a synthetic pool).

**Live A/B re-run, g50t, `budget=250`, `CARNOT_ARC_INDUCE_TRANSITIONS_K=20` (vs the k=8 baseline
from REQ-ARC-FCP-5699-21/22's own g50t measurement).** Result is genuinely MIXED, not a clean win:

```
baseline: heldout_change_consistency=0.114187, heldout_accuracy=0.0, correct_changed_cells=33, trust_energy=13.808703
sge:      heldout_change_consistency=0.0,      heldout_accuracy=0.0, correct_changed_cells=0,  trust_energy=inf
```

**The baseline arm improved substantially.** `correct_changed_cells` went from 0 (k=8) to 33
(k=20) -- a real, non-trivial jump, and `heldout_change_consistency` moved from a flat 0.0 to
0.114. This is direct, positive evidence FOR the diagnosed root cause: showing more per-action
examples let this particular LLM generation correctly predict a meaningful number of changed
cells it previously got zero of. `binary_gate_pass` still reads `False` (0.114 is well below the
0.5 trust threshold), so tier 2 still does not succeed outright -- but the underlying dynamics
model genuinely got BETTER, not just differently wrong.

**The SGE arm's independent generation attempt got WORSE, and for a NEW reason: a real code bug,
not just a still-wrong hypothesis.** Reading the actual generated code left on disk after this run
(`results/arc_e3/g50t/world_model.py`, the SGE arm's code -- it runs second in the same script
invocation and overwrites the baseline arm's file, which was NOT separately captured; a real
limitation of this specific re-run, noted below) shows:

```python
if action == 6:
    px, py = data['x'], data['y']
    grid[py, px] = 1
    return grid
if action in [1, 2, 3, 4, 5]:
    if action == 1:
        grid = grid.copy()
        grid[py, px] = 1     # px, py are UNDEFINED here -- only assigned inside the action==6
        return grid          # branch above, which already returned. This raises NameError.
    ...  # same undefined-variable bug repeated for actions 2-5
```

`px`/`py` are referenced in the `action in [1,2,3,4,5]` branch but only ever assigned inside the
`action == 6` branch, which already `return`s before that code is reached -- a genuine
`NameError`/`UnboundLocalError` at runtime for every action 1-5 call. This is consistent with
`trust_energy=inf` (the verifier's scoring path evidently assigns an infinite/maximal-violation
energy when the engine raises rather than returning a wrong-but-valid grid) and
`correct_changed_cells=0`. **This is a WORSE failure mode than k=8's hardcoded-but-executing
coordinates** -- k=8's code was wrong but never crashed; this k=20 generation is syntactically
valid Python that crashes at call time.

**What this narrows.** Raising `k` is not a strictly-reliable fix -- it can genuinely help (this
run's baseline arm: real, substantial improvement) or genuinely hurt (this run's sge arm: a novel
execution-time bug from writing more complex, longer code under the larger prompt) in the SAME
diagnostic session, on the SAME game. LLM code-synthesis reliability here is evidently
sensitive to sampling/generation variance, not a deterministic function of how much data is shown
-- more context can unlock a better hypothesis OR invite a new class of bug (undefined-variable
control-flow mistakes that a shorter, simpler generation didn't have room to make). Neither arm
reached `binary_gate_pass=True`; tier 2 still does not succeed on g50t even in the improved case.

**Honest scope limit.** n=1 game (g50t), n=1 pair of measurements (one improved, one regressed).
The baseline arm's actual improved CODE was not preserved (overwritten by the sge arm's run before
being read) -- a real gap in this specific re-run's methodology, worth fixing (capture per-arm
code snapshots, not just the final state) before drawing a stronger conclusion. This result should
be read as "raising k is a live, real lever with visible effect in both directions," not as either
a confirmed fix or a refutation.

**Concrete next step if this thread continues.** (a) Fix the methodology gap directly: capture
`world_model.py` after EACH arm (not just the final overwritten state) so both outcomes are fully
inspectable. (b) Repeat with multiple seeds/generations at a fixed `k` to characterize the
variance directly (is a 33-correct-cells result typical or a lucky outlier at k=20?) rather than
drawing conclusions from n=1 per arm. (c) If variance is confirmed high, consider a repair-loop
approach instead of (or alongside) raising k: the codebase already has a "refactor" path
(`CodexProposer.refactor`/`LocalGGUFProposer`'s equivalent) that feeds back mismatches for a
second pass -- worth checking whether that path is exercised for tier 2's stall-triggered
first-contact induction, since a self-correcting second pass could catch exactly this class of
undefined-variable bug that raising k alone cannot prevent.

#### SCENARIO-ARC-FCP-5699-23-RAISING-K-HELPS-ONE-GENERATION-BREAKS-ANOTHER

Given REQ-ARC-FCP-5699-22 diagnosed the `k=8` default as starving the LLM to ~1 example per
action type, producing hardcoded-literal-coordinate memorization on g50t
When `CARNOT_ARC_INDUCE_TRANSITIONS_K=20` is set and g50t is re-measured, both arms, against the
prior k=8 baseline (`correct_changed_cells=0` for both arms)
Then the baseline arm's `correct_changed_cells` rises to 33 (`heldout_change_consistency`
0.0->0.114) -- genuine improvement supporting the diagnosed root cause -- while the sge arm's
independent generation attempt produces code with an undefined-variable bug (`px`/`py` referenced
outside the branch that defines them) that crashes at call time (`trust_energy=inf`,
`correct_changed_cells=0`) -- a NEW, worse failure mode than k=8's non-crashing-but-wrong
hypothesis -- establishing that raising k has a real but HIGH-VARIANCE effect, not a reliable fix,
and neither arm reaches `binary_gate_pass=True`

### REQ-ARC-FCP-5699-24: The Codebase's Entire Repair Infrastructure Is ALSO Gated By The Same First-Contact Exemplar Requirement

REQ-ARC-FCP-5699-23's own next step named checking whether the codebase's existing repair-loop
(`proposer.refactor`, feeding mismatches back for a second pass) is exercised for tier 2's
stall-triggered first-contact induction. Reading the code (no new run needed) answers this
precisely, and surfaces a second, even more complete finding beyond the original question.

**The repair loop exists, is real, tested, working code -- and is structurally unreachable for
first-contact induction.** `execute_bounded_llm_reinduction`
(`arc_llm_reinduction.py:654`, up to `MAX_REFINEMENT_ROUNDS=3` rounds: round 0 = `.induce()`,
rounds 1+ = `.refactor(game, counterexample_result)`, feeding back real per-transition mismatch
evidence) has exactly ONE call site in the whole agent
(`arc_competition_agent.py:3610`), and that call site is entirely INSIDE the `if attempt["reason"]
== "level_up_reinduction" or next_level_episode:` branch -- the branch every sp80/cd82/g50t attempt
this REQ chain has measured NEVER takes (`reason` is always `"stall"`, and `next_level_episode`
requires `_previous_level_complete_grid is not None`, per REQ-ARC-FCP-5699-18, always `None` for
these games). **First-contact induction gets exactly ONE shot with the plain, single-round
`self._proposer().induce(...)` + `e3.load_engine(...)` path (`arc_competition_agent.py:3715-3719`)
-- zero refinement rounds, no counterexample feedback, ever.**

**A second, even sharper finding: the codebase ALREADY BUILT a specific rescue for cd82's exact
pathology, and it is ALSO neutralized by the same gap.** `_repair_degenerate_goal`
(`arc_llm_reinduction.py:606-646`, 2026-06-25 operator directive) exists precisely to rescue "a
constant `return False`, or an exact-match against a hardcoded win grid the induced engine can
never reach" -- REQ-ARC-FCP-5699-22's cd82 finding, verbatim, in the function's own docstring,
written a month before this REQ chain independently rediscovered the same failure mode. Its
mechanism: substitute an exemplar-derived `_nonzero_count_predicate` fallback ("complete once the
grid has at least as many filled cells as the L1-completion exemplar") and re-check
satisfiability. But: `if previous_level_complete_grid is None: return None` -- line 636, the very
first line of the function body. **The goal-repair mechanism requires the exact same exemplar that
REQ-ARC-FCP-5699-18 established cannot exist for a first-contact level.** So even if the refactor
loop WERE wired into the stall path, its goal-repair fallback would still do nothing for
first-contact levels -- only the DYNAMICS-side refactor rounds (which operate on transition
mismatches, not the goal predicate, and genuinely don't need an exemplar) could plausibly help.

**What this narrows -- the complete picture across REQ-ARC-FCP-5699-18, -22, and this REQ.** Every
mechanism this codebase has for handling a bad first-contact induction shares one root dependency:
`_previous_level_complete_grid` (or an equivalent win exemplar). Tier 1's goal-energy gradient
(5699-18), tier 2's goal-predicate induction (5699-22), AND the codebase's own goal-repair rescue
(this REQ) are ALL gated by the same `is None` check. This is not three unrelated bugs in three
components -- it is ONE structural gap (no positive win example exists before the first win, by
definition) surfacing identically in every place the architecture assumed one would be available.
The DYNAMICS-side refactor loop is the one exception: it operates on transition mismatches (which
DO exist pre-first-win) rather than the goal predicate, so it remains a genuinely untested,
plausible lever for tier 2's dynamics-half failures (REQ-ARC-FCP-5699-21's `correct_changed_cells`
metric) specifically -- just never wired into the path that would need it.

**Honest scope limit.** This REQ is a code-reading finding (real, verified from source, no new run
executed) -- it establishes that the mechanism is UNREACHABLE and WHY the existing goal-repair
half wouldn't help even if reached, but does NOT yet establish whether wiring the DYNAMICS-side
refactor rounds into the stall path would actually improve `correct_changed_cells` in a live
measurement. That is the natural next empirical test.

**Concrete next step if this thread continues.** Wire a DEV-ONLY opt-in path that routes
stall-triggered first-contact induction through `execute_bounded_llm_reinduction` (with
`previous_level_complete_grid=None`, `structural_goal_provider=None` -- both already handled
gracefully by the function without crashing, verified from source) instead of the current
single-shot `induce()`+`load_engine()` call, gated behind a new env var so production is
unaffected by default. Test in ISOLATION from REQ-ARC-FCP-5699-23's `k` change (i.e. at the k=8
default) to avoid confounding two simultaneous variables, then live A/B on the same g50t trace for
direct comparability.

#### SCENARIO-ARC-FCP-5699-24-REPAIR-INFRASTRUCTURE-SHARES-THE-SAME-FIRST-CONTACT-GAP

Given REQ-ARC-FCP-5699-23 named checking whether the codebase's existing refactor/repair-loop path
is exercised for tier 2's first-contact induction
When `execute_bounded_llm_reinduction`'s only call site and `_repair_degenerate_goal`'s guard
clause are read directly from source
Then the refactor loop is confirmed structurally unreachable for stall-triggered (first-contact)
induction attempts (its only call site requires `reason == "level_up_reinduction"` or
`next_level_episode`, both requiring a prior level completion), AND the goal-repair fallback built
specifically to rescue constant-false predicates like cd82's is ALSO gated by
`previous_level_complete_grid is None` -- establishing that tier 1's goal-energy gap
(REQ-ARC-FCP-5699-18), tier 2's goal-predicate starvation (REQ-ARC-FCP-5699-22), and the
codebase's own repair infrastructure (this REQ) are three surfacings of the SAME single structural
gap, not three independent problems

### REQ-ARC-FCP-5699-25: Wiring The Dynamics-Side Refactor Loop Into The Stall Path

REQ-ARC-FCP-5699-24's concrete next step: wire a DEV-ONLY opt-in path routing stall-triggered
first-contact induction through `execute_bounded_llm_reinduction` instead of the current
single-shot `induce()`+`load_engine()` call, tested in isolation from REQ-ARC-FCP-5699-23's `k`
change.

**Implementation.** `_induce_and_plan()` (`arc_competition_agent.py`) gained a new branch, checked
right after `self._fit_dsl_model()` and before the (pre-existing, unrelated) active-probe block:
when `CARNOT_ARC_STALL_REFACTOR_LOOP=1` is set, calls `execute_bounded_llm_reinduction` with the
SAME parameters the pre-existing `level_up_reinduction` branch uses (`max_rounds=
MAX_REFINEMENT_ROUNDS=3`, `min_heldout_accuracy=1.0`, `min_goal_predicate_consistency=1.0`), except
`previous_level_complete_grid=self._previous_level_complete_grid` (which will be `None` for
first-contact levels -- REQ-ARC-FCP-5699-24 confirmed this is handled gracefully, not a crash) and
`structural_goal_provider=None` (also confirmed graceful). The outcome is recorded onto `attempt`
via the same field-mapping pattern as the level_up_reinduction branch, plus a new
`stall_refactor_loop_used: True` marker so artifacts self-document which path produced them. When
the env var is unset (production default), execution falls through UNCHANGED to the pre-existing
plain single-shot path -- verified by a dedicated test that monkeypatches
`execute_bounded_llm_reinduction` to raise if called, confirming it is never invoked by default.
A second test confirms the enabled path threads `previous_level_complete_grid=None` and
`structural_goal_provider=None` through correctly and records the outcome fields.

**Live re-run, g50t, `budget=250`, `CARNOT_ARC_STALL_REFACTOR_LOOP=1`, at the k=8 DEFAULT
(isolated from REQ-ARC-FCP-5699-23's `k=20` change -- one variable at a time).** Both arms:
`stall_refactor_loop_used=True` (confirms the wiring fires), `refinement_rounds_used=2`,
`planned=False`.

```
baseline round 1 (induce):  heldout_accuracy=0.125, real_n_correct=5/25, accepted_by_heldout_verifier=False (0.125 < 1.0 threshold)
baseline round 2 (refactor): proposer_ok=False, "local model code unusable after 3 tries (missing ('engine','is_level_complete') in output)", skipped=proposer_failed
sge round 1 (induce):       heldout_accuracy=0.0,   real_n_correct=0/25, accepted_by_heldout_verifier=False
sge round 2 (refactor):     proposer_ok=False, same "unusable after 3 tries" message, skipped=proposer_failed
```

**The mechanism is wired and fires correctly -- but its core value-add (repair via counterexample
feedback) never actually executes: round 2's `.refactor()` call fails to produce parseable code,
for BOTH arms, every time measured.** The refactor-loop machinery correctly detects round 1's
failure (`0.125`/`0.0` well below the strict `min_heldout_accuracy=1.0` bar) and correctly attempts
a `.refactor()` call with real counterexample evidence -- but the LOCAL LLM (Qwen3.5-9B-MTP) cannot
reliably close out valid `engine`/`is_level_complete` code when given the counterexample-feedback
prompt (which embeds substantial per-mismatch cell-level diff data -- the `real_mismatches` payload
above is representative: dozens of individual cell changes per mismatch, several mismatches per
counterexample). This is a NEW, previously-unmeasured bottleneck in this REQ chain: the repair
mechanism's prompt is apparently long/complex enough to push this local model past reliable
structured-code generation, similar in spirit to the documented L2 rambling-truncation issue the
existing `_L2_CODEONLY_DIRECTIVE` was built to fix (2026-06-25) -- but that directive is NOT applied
to the plain `refactor_prompt()` path.

**A genuinely interesting side-observation: round 1's induced code (`results/arc_e3/g50t/
world_model.py`, preserved from the baseline arm's induce call since round 2 never overwrote it)
is qualitatively the BEST-REASONED generation this whole REQ chain has observed** -- it finds the
player DYNAMICALLY by color (`np.argwhere(new_grid == 5)`, not a hardcoded coordinate), applies a
genuinely GENERAL relative-offset movement rule (`dr, dc` per action, applied to the found
position), and writes a non-degenerate, testable win predicate ("player reaches (0,0)") instead of
a constant `False`. This is a materially different, better-reasoned hypothesis than either the k=8
baseline (hardcoded per-action coordinates) or the k=20 test's two outcomes (a genuine improvement
and a crashing bug). **This improvement is attributable to LLM sampling variance on the SAME
single-shot `induce()` call the plain path already makes -- NOT to anything the refactor loop
itself contributed, since round 2 (the actual repair round) failed outright.** It still didn't
reach a passing score (`real_n_correct=5/25`), so it's not a full win either -- but it strengthens,
rather than refutes, REQ-ARC-FCP-5699-23's high-variance finding: the underlying model IS capable
of well-reasoned, general hypotheses on this game, at least sometimes; NEITHER raising `k` NOR
adding a refactor loop reliably elicits that capability on demand.

**What this narrows.** In this specific measurement, wiring the refactor loop into the stall path
produced NO net benefit -- neither arm improved on the k=8 no-refactor baseline's
`correct_changed_cells=0`, because the loop's repair round never successfully executed. The
mechanism is real, correctly gated, and confirmed non-crashing (matching REQ-ARC-FCP-5699-24's
prediction that `previous_level_complete_grid=None`/`structural_goal_provider=None` would be
handled gracefully) -- but its practical value on this local model is currently bottlenecked by
generation reliability on the counterexample-feedback prompt specifically, a distinct failure mode
from anything diagnosed so far.

**Honest scope limit.** n=1 game (g50t), n=1 pair of measurements. Whether the refactor prompt's
length/complexity is really the cause of round 2's failure (versus some other generation issue) is
inferred from the error message and the L2-rambling precedent, not directly instrumented here.

**Concrete next step if this thread continues.** (a) Apply an analogous code-only/token-budget
directive to `refactor_prompt()`'s call (mirroring `_L2_CODEONLY_DIRECTIVE`'s fix for the same
rambling-truncation failure mode in the goal-induction path) and re-measure whether round 2 then
succeeds. (b) Truncate/summarize the counterexample payload shown to `.refactor()` (fewer
mismatches, or only the changed-cell COUNT rather than every coordinate) to reduce prompt length
directly. (c) Given the demonstrated high sampling variance (this REQ's round 1, REQ-ARC-FCP-
5699-23's k=20 baseline arm), consider a best-of-N sampling strategy (run `.induce()` N times,
keep the highest-`heldout_accuracy` candidate) as an alternative to iterative refinement, since
variance has now been observed to sometimes produce a genuinely strong result on the FIRST try.

#### SCENARIO-ARC-FCP-5699-25-REFACTOR-LOOP-FIRES-BUT-REPAIR-ROUND-FAILS-TO-GENERATE

Given `CARNOT_ARC_STALL_REFACTOR_LOOP=1` routes stall-triggered induction through
`execute_bounded_llm_reinduction` (previously reachable only from the level_up_reinduction branch)
When g50t is re-measured at the k=8 default, both arms
Then `stall_refactor_loop_used=True` and `refinement_rounds_used=2` confirm the mechanism fires
correctly, round 1's `.induce()` call succeeds with partial signal (`real_n_correct` 5/25 and
0/25) but is correctly rejected by the strict `min_heldout_accuracy=1.0` gate, and round 2's
`.refactor()` call FAILS to produce parseable code for both arms
("local model code unusable after 3 tries") -- so the mechanism's core repair value-add never
executes, `planned` stays `False` for both arms, and the net effect of wiring in the refactor loop
is null in this measurement, though round 1's induced code is independently the best-reasoned
generation observed in this REQ chain, attributable to sampling variance rather than the refactor
wiring itself

### REQ-ARC-FCP-5699-26: Fix The Real Bug -- `refactor_prompt`'s Mismatch Payload Was Invalid JSON, Not A Reasoning-Task Limitation

The operator's recommended cheapest fix for REQ-ARC-FCP-5699-25's round-2-generation-failure was
to apply an `_L2_CODEONLY_DIRECTIVE`-style forcing prefix to `refactor_prompt()`'s call, mirroring
the fix that already solves the same class of failure for the induce path. Reading
`_gen_to_file`'s own code comments (`arc_executable_world_model.py` ~line 1458-1464) BEFORE
implementing that found it is explicitly, deliberately CONTRAINDICATED: `codeonly_eligible` is
scoped to induce-only "because [refactor] is a REASONING task (debug BEFORE/PREDICTED/OBSERVED
mismatches) that must NOT be told to 'skip all reasoning'" -- a documented 2026-06-25 design
decision, not an oversight. Applying the recommended fix as originally framed would have silently
reverted a considered architectural choice without engaging with why it was made.

**Investigating further (rather than blindly executing the operator's literal instruction) found
the REAL bug.** `refactor_prompt()`'s mismatch encoding was `json.dumps(vr.mismatches[:5],
indent=1)[:4000]` -- a raw CHARACTER-COUNT slice applied AFTER JSON-encoding. Verified directly
against a real g50t counterexample from REQ-ARC-FCP-5699-25's own run: 5 real mismatches serialize
to **12,212 characters**, and `[:4000]` cuts the string at `'...   "true_chang'` -- literally
mid-field-name, producing **genuinely invalid JSON** that was being shown directly to the LLM as
the failing-cases evidence it's asked to debug from. The model was never given a reasoning-capacity
problem to solve; it was given a malformed, truncated data blob and asked to fix code based on it.
This fully explains round 2's "code unusable after 3 tries" failure without needing to suppress
the model's reasoning at all.

**The fix respects the deliberate reasoning-task design.** `_bounded_mismatches()` caps each
mismatch's `true_change`/`your_prediction_was_wrong_at` cell-diff lists to a fixed count (8) BEFORE
JSON-encoding, adding a companion `_omitted_count` field when cells are cut -- every mismatch
entry stays structurally COMPLETE and the overall JSON stays VALID regardless of how large the
underlying diffs are, while still showing a representative sample of cells per mismatch (up to 5
mismatches, matching the prior cap). The blind post-encoding character slice is removed entirely
now that the input is bounded at the source. This is an unconditional bug fix, not an opt-in
env-var-gated hypothesis -- invalid JSON has no legitimate defense, and `refactor_prompt` is shared
by BOTH the pre-existing, already-shipped `level_up_reinduction` path AND REQ-ARC-FCP-5699-25's new
stall path, so this fix improves PRODUCTION behavior, not just the dev diagnostic being built here.
3 new unit tests verify: large cell lists get capped with an honest omitted-count, small lists pass
through untouched, and -- the direct regression test -- a payload sized like the real g50t
counterexample (which is FIRST verified to reproduce the old truncation-into-invalid-JSON bug, so
the test would have failed before this fix) now parses as valid JSON.

**Live re-run, g50t, `budget=250`, `CARNOT_ARC_STALL_REFACTOR_LOOP=1`, k=8 default, WITH the JSON
fix -- same isolated config as REQ-ARC-FCP-5699-25 for direct comparability.**

```
baseline round 2 (refactor): proposer_ok=False, "local model code unusable after 3 tries (missing ('engine', 'is_level_complete') in output)" -- SAME message as before the fix
sge round 2 (refactor):      proposer_ok=False, "local gguf (GPU server) failed: TimeoutError('timed out')" -- a NEW, different failure mode
```

**Honest result: the JSON fix did NOT resolve round 2's failure.** The baseline arm produces the
EXACT SAME "code unusable after 3 tries" message as the pre-fix run, even though its input is now
verifiably valid JSON -- this directly refutes the hypothesis that malformed JSON was the (sole,
or even primary) cause of round 2's generation failure. The sge arm failed differently this time --
an outright `TimeoutError` on the underlying HTTP request to the GPU server (the class default
`timeout=300` seconds was exceeded) -- a genuinely different failure class, not a code-quality
issue at all. **The JSON fix is still correct and worth keeping** (it's a real, independently
verifiable bug -- invalid JSON has no legitimate defense regardless of whether it explains THIS
specific symptom), but it is not sufficient by itself to unblock the refactor loop.

**What this narrows, and sets up REQ-ARC-FCP-5699-27.** Two live, real data points now point AWAY
from "the input was malformed" and TOWARD "the refactor call's generation is taking too
long/producing too much text" -- the sge arm's direct 300s timeout is unambiguous evidence of
this, independent of any new instrumentation. This run predates REQ-ARC-FCP-5699-27's truncation-
detection mechanism, so it's not yet known whether the baseline arm's 3 failed tries were hitting
`stop_type == "limit"` (the model ran out of its 2560-token generation budget before finishing) --
that is the concrete next measurement, now with the tooling in place to answer it directly rather
than inferring from symptoms.

#### SCENARIO-ARC-FCP-5699-26-JSON-FIX-ALONE-DOES-NOT-UNBLOCK-ROUND-2

Given `refactor_prompt`'s mismatch payload is fixed to always produce valid JSON (REQ-ARC-FCP-
5699-26's `_bounded_mismatches`)
When g50t is re-measured at the k=8 default with `CARNOT_ARC_STALL_REFACTOR_LOOP=1`, both arms
Then the baseline arm's round 2 fails with the IDENTICAL "code unusable after 3 tries" message as
the pre-fix run (refuting invalid JSON as the sole cause), and the sge arm's round 2 fails with a
direct `TimeoutError` on the underlying HTTP request -- establishing that the bottleneck is more
likely generation TIME/LENGTH than input validity, and motivating REQ-ARC-FCP-5699-27's
truncation-detection instrumentation as the next concrete diagnostic step

### REQ-ARC-FCP-5699-27: MANDATORY Truncation Detection -- Every Completion Call Must Surface Why Generation Stopped

Operator directive, prompted directly by REQ-ARC-FCP-5699-25/26's investigation: "should we
increase max context? we must always detect reaching max context truncation." Before this REQ, the
answer to "should we raise `max_tokens`" was UNANSWERABLE from the codebase's own signals --
`generate()`/`complete_text()` (`arc_executable_world_model.py`, `LocalGGUFProposer`) read ONLY
`response.get("content", "")` from llama.cpp's `/completion` endpoint, silently discarding two
fields the server already provides: `stop_type` (`"limit"` means generation was cut off by hitting
`n_predict` -- `self.max_tokens` -- before finishing naturally; confirmed directly via a live probe
against the warm port-8930 server) and `truncated` (whether the INPUT PROMPT itself exceeded the
server's context window, `n_ctx` -- a different, upstream failure mode from an output-length cap).
Every generation failure collapsed into the same generic "missing (...) in output" message,
indistinguishable from "the model finished and produced garbage."

**Implementation.** `LocalGGUFProposer` gained two new dataclass fields, `last_stop_type: str = ""`
and `last_prompt_truncated: bool = False`, plus a `_record_completion_diagnostics(response: dict)`
helper that sets both from any completion response. Wired into BOTH completion paths: `generate()`
(the code-extraction path used by induce/refactor/gap-fillers) and `complete_text()` (the raw-text
path used more broadly -- SGE's `LLMStrategyProposer`, the cell selector, and several standalone
experiment scripts). Neither method's RETURN CONTRACT changes (`generate()` still returns `(bool,
str)`; `complete_text()` still always returns `(True, str)` per its documented contract) -- the
diagnostics are ALWAYS recorded as instance attributes regardless of success/failure, so any caller
that cares can check `proposer.last_stop_type`/`proposer.last_prompt_truncated` after a call without
every caller needing a signature change. `generate()`'s own failure-path message additionally names
the cause explicitly when detected (`"[HIT n_predict=N OUTPUT LIMIT before completing]"` /
`"[PROMPT TRUNCATED -- exceeded server context window]"`), so the diagnosis flows into
`refinement_rounds[i]['message']` -- a field every caller in this REQ chain already captures --
with zero schema changes needed elsewhere. 5 new unit tests (`test_codeonly_induce_scoping.py`,
mocking the `/completion` response) verify: diagnostics are recorded on the happy path too (not
just failures), the two failure messages name their specific cause, and `complete_text()` records
diagnostics despite always returning success.

**Answering "should we increase max context" -- not yet fully resolved, but now answerable.** This
REQ ships the INSTRUMENTATION; REQ-ARC-FCP-5699-26's live run (which predates this code) already
supplies one direct, instrumentation-independent data point supporting the "generation is taking
too long" theory (the sge arm's raw 300-second `TimeoutError`), but does not yet show whether the
baseline arm's failures were `stop_type == "limit"` (an output-token-budget problem, where raising
`max_tokens` is the right lever) versus a `n_ctx`-related `truncated` prompt overflow (unlikely
given the bounded ~3-4K-character prompts REQ-ARC-FCP-5699-23/26 already produce) versus something
else the model itself is doing. **Concrete next step, already queued:** re-run the SAME isolated
g50t configuration WITH this detection code active, and read `refinement_rounds[i]['message']`
directly for the `stop_type`/`truncated` markers -- that measurement will answer the question with
real evidence instead of inference, and if `stop_type == "limit"` is confirmed, raising
`max_tokens` for the refactor call specifically (not necessarily `n_ctx`, which appears
under-pressured given current prompt sizes) becomes a directly-justified next fix.

#### SCENARIO-ARC-FCP-5699-27-STOP-TYPE-AND-TRUNCATED-ARE-NO-LONGER-SILENTLY-DISCARDED

Given `generate()`/`complete_text()` previously read only `response.get("content", "")` from every
llama.cpp completion response, discarding `stop_type`/`truncated`
When a completion response reports `stop_type="limit"` (the model was cut off by its output-token
budget) or `truncated=True` (the prompt itself overflowed the context window)
Then `LocalGGUFProposer.last_stop_type`/`.last_prompt_truncated` are set from the REAL response on
every call (success or failure), and `generate()`'s own failure message names the specific cause
when either condition is detected -- verified live against the warm production llama-server (a
direct probe with `n_predict=5` returns `stop_type: "limit"`, confirming the field genuinely
exists and behaves as documented, not just in mocked tests) -- so a future measurement can
distinguish "ran out of output budget" from "prompt too long for context" from "the model produced
complete-but-wrong output," three previously-indistinguishable failure modes

### REQ-ARC-FCP-5699-28: The Definitive Answer -- Yes, `max_tokens` Was A Real Bottleneck (Plus A Self-Caught Stale-Read Mistake)

REQ-ARC-FCP-5699-27's own queued next step: re-run g50t with the truncation-detection code active
and read the real `stop_type`/`truncated` markers.

**A self-caught mistake worth recording plainly, not glossed over.** Mid-run, after the baseline
arm alone had finished, the results file was read and showed round 2's failure message with NO
truncation marker -- and that was reported to the operator as the (implied) answer: "no, raising
max context would not help." **This was wrong, and the error was foreseeable**: `arc_sge_live_
path_ab.py` writes `results/arc_sge_live_path_ab_g50t.json` exactly ONCE, after BOTH arms
complete -- so reading it before the sge arm finished returned the PREVIOUS run's leftover file,
not this run's real baseline result. This is precisely the pitfall this project's own
Reading-Results Discipline exists to prevent, and it recurred here despite having been flagged
multiple times earlier in this exact session's transcript. Caught and corrected once the sge arm
completed and the file was re-read for real.

**The corrected, definitive answer, from the REAL final data:**

```
baseline round 2 (refactor): FAILS -- "local model code unusable after 3 tries (missing ('engine', 'is_level_complete') in output [HIT n_predict=2560 OUTPUT LIMIT before completing])"
sge round 2 (refactor):      FAILS -- "local gguf (GPU server) failed: TimeoutError('timed out')"
```

**Yes -- the baseline arm's round-2 failure IS confirmed to be a genuine `n_predict` (max_tokens)
truncation.** The model was cut off at its 2560-token output budget before completing valid
`engine`/`is_level_complete` code, on the specific (3rd, final) sub-try whose diagnostics are
captured in the reported message. This is a DIRECT, real-evidence-backed justification for raising
`max_tokens` on the refactor call -- not a guess. **The sge arm's failure is a SEPARATE, different
bottleneck**: an outright 300-second HTTP request `TimeoutError`, which bypasses the new
diagnostic code entirely (a bare network exception is caught before `stop_type`/`truncated` are
ever read) -- this is a wall-clock request-timeout problem, not a token-count problem, and is a
DIFFERENT axis: raising `max_tokens` alone could plausibly make this WORSE (a longer permitted
generation takes longer wall-clock time), not better, unless `timeout` is raised in tandem.

**Direct answer to "should we increase max context."** `n_ctx` (the context WINDOW) has no
evidence of being the bottleneck in either arm -- `truncated` was never observed `True` across any
measurement in this REQ chain, consistent with REQ-ARC-FCP-5699-26's now-bounded ~3-4K-character
prompts. `max_tokens` (the per-request OUTPUT budget, currently 2560) DOES have direct, confirmed
evidence of being a real bottleneck for at least one failure. The honest, complete answer is: raise
`max_tokens` for the refactor call (a directly justified, evidence-backed fix), AND raise `timeout`
in proportion (since a larger `max_tokens` permits longer generations, and the sge arm's failure
shows the current 300s timeout is already sometimes insufficient even at the CURRENT budget) --
neither alone is quite the right isolated fix; they need to move together.

**Honest scope limit.** n=1 game (g50t), n=1 pair of measurements. Only ONE of the 3 sub-tries per
round is captured in the final message (the LAST one before the loop exhausts `tries`) -- earlier
sub-tries' diagnostics are not preserved, so it's unknown whether ALL 3 tries hit the same failure
mode or whether earlier tries failed differently.

**Concrete next step if this thread continues.** Raise both `max_tokens` (e.g. to 4096, the
`LocalGGUFProposer` class default already used elsewhere in this codebase, vs the 2560 this
specific live-submission config uses) and `timeout` proportionally for the refactor call
specifically (or globally, since both completion paths share the same instance), then re-measure
on the same isolated g50t config to see whether round 2 actually succeeds this time. If it still
fails, capture the RAW completion text on a failed try (not just whether it parsed) to see what
the model is actually producing when it exhausts its budget.

#### SCENARIO-ARC-FCP-5699-28-N-PREDICT-LIMIT-CONFIRMED-REAL-AFTER-A-SELF-CAUGHT-STALE-READ

Given a mid-run read of `results/arc_sge_live_path_ab_g50t.json` (before the sge arm had finished,
so the file was still the PREVIOUS run's leftover state) was mistakenly reported as this run's
real result, incorrectly answering "no" to whether raising max context would help
When the file is re-read after BOTH arms genuinely complete (the file is written exactly once, at
the very end)
Then the baseline arm's round-2 failure message DOES contain `[HIT n_predict=2560 OUTPUT LIMIT
before completing]` -- confirming max_tokens truncation IS a real, evidence-backed bottleneck,
reversing the premature stale-read conclusion -- while the sge arm fails via a separate 300s
`TimeoutError` that bypasses the detection code entirely, establishing that `max_tokens` and
`timeout` are two distinct axes that both need raising together, not either one alone

### REQ-ARC-FCP-5699-29: Raising Both Levers Does Not Fix It Either -- The Bottleneck Is Not Resource Budget

REQ-ARC-FCP-5699-28's own next step: raise both `max_tokens` and `timeout` together and re-measure.

**Implementation.** `_proposer()`'s lazy default (`arc_competition_agent.py` ~line 3318-3340, the
construction site used by BOTH production and every A/B script in this REQ chain via
`proposer=None`) gained two DEV-ONLY env overrides: `CARNOT_ARC_INDUCE_MAX_TOKENS` (unset -> 2560,
byte-identical) and `CARNOT_ARC_INDUCE_TIMEOUT` (unset -> 300, byte-identical -- previously
implicit via the `LocalGGUFProposer` class default, now explicit). 2 new tests verify the
unset-default and the override both work correctly.

**Live re-run, g50t, `budget=250`, `CARNOT_ARC_STALL_REFACTOR_LOOP=1`, k=8 default,
`CARNOT_ARC_INDUCE_MAX_TOKENS=4096` (the `LocalGGUFProposer` class default used elsewhere in this
codebase, 1.6x the prior 2560), `CARNOT_ARC_INDUCE_TIMEOUT=600` (2x the prior 300s).** Read only
after confirmed-genuine completion this time (`ps aux` showed the process had exited AND the
task's own log showed the `"wrote results/..."` line, per the discipline the REQ-ARC-FCP-5699-28
stale-read mistake established as mandatory going forward):

```
baseline round 2 (refactor): FAILS -- "local model code unusable after 3 tries (missing ('engine', 'is_level_complete') in output)" -- NO n_predict/truncated marker
sge round 2 (refactor):      FAILS -- IDENTICAL message -- NO n_predict/truncated marker, NO TimeoutError this time
```

**Decisive: raising both levers does NOT fix it, and this time NEITHER failure mode from the prior
two runs recurred.** No `[HIT n_predict=...]` marker on either arm (the model completed within its
now-4096-token budget) and no `TimeoutError` (the sge arm's prior 300s timeout, now raised to 600s,
was not hit either). Both arms simply produced output that, once again, does not contain the
required `def engine`/`def is_level_complete` after 3 tries -- the SAME generic failure as every
prior measurement in this REQ chain, just now with confirmed non-truncated, non-timed-out
generations. **This closes the resource-budget hypothesis definitively**: REQ-ARC-FCP-5699-28's
`n_predict` hit was real (confirmed on that specific measurement) but was NOT the dominant or sole
cause -- with ample room and time, the model still fails to produce parseable required code when
responding to the counterexample-heavy refactor prompt. Duration also confirms more room was
genuinely used: baseline round-1 took long enough that the FULL script run reached 983.56s (vs
648-770s at the smaller budget), consistent with longer generations actually happening, not the
override silently failing to apply.

**What this narrows -- the complete picture across REQ-ARC-FCP-5699-23 through -29.** Every
"cheap, well-evidenced" lever this chain identified has now been tried and correctly ruled out on
g50t's specific refactor-loop failure: more per-action training examples (5699-23, high-variance,
not reliable), the refactor loop's own repair mechanism (5699-25, fires correctly but the repair
round itself fails), a genuine JSON-encoding bug in the counterexample payload (5699-26, fixed but
insufficient alone), and now output-token/wall-clock budget (this REQ, ruled out definitively).
None of these were wasted effort -- each is either a real, kept bug fix (5699-26), real
infrastructure (5699-27's detection, 5699-28/29's budget levers, now correctly justified as NOT
needed further here), or a real negative result narrowing the hypothesis space. What remains
untested is qualitatively different: actually reading the model's RAW completion text on a failed
try (not just whether it parsed) to see what it's actually producing -- the one diagnostic
REQ-ARC-FCP-5699-25 named at the start of this sub-thread and never yet executed.

**Honest scope limit.** n=1 game (g50t), n=1 measurement per configuration across this whole
sub-thread. None of REQ-ARC-FCP-5699-23 through -29's fixes have been tested on sp80 or cd82 --
whether any of them (individually or combined) would behave differently on a game with a different
induced-model/counterexample profile is completely open.

**Concrete next step if this thread continues, per the operator's own instruction to name this
choice explicitly rather than default to another sub-fix.** Two genuinely different directions,
not more of the same: (a) DEPTH -- capture and read the actual raw completion text from a failed
refactor try (the one diagnostic never yet run in this whole sub-thread) to see WHY the model
doesn't emit the required functions, which could reveal something structurally different from
every resource-budget hypothesis tried so far; (b) BREADTH -- test whether ANY of REQ-ARC-FCP-
5699-23 through -29's fixes (the JSON bug fix alone, at minimum, since it's a real, kept,
unconditional improvement) change the picture on sp80/cd82, since all 7 REQs in this sub-thread
have measured ONLY g50t.

#### SCENARIO-ARC-FCP-5699-29-RAISED-BUDGETS-DO-NOT-FIX-IT-RESOURCE-HYPOTHESIS-CLOSED

Given REQ-ARC-FCP-5699-28 confirmed `max_tokens` truncation was real on one specific measurement
and named raising both `max_tokens` and `timeout` together as the direct next test
When g50t is re-measured with `CARNOT_ARC_INDUCE_MAX_TOKENS=4096` and `CARNOT_ARC_INDUCE_
TIMEOUT=600`, both arms, read only after confirmed-genuine completion (learning directly from the
prior stale-read mistake)
Then NEITHER arm shows an n_predict-limit marker NOR a TimeoutError this time -- both generations
completed within their (larger) budget and time allowance -- yet round 2 still fails with the
same generic "missing required functions" message on both arms, closing the resource-budget
hypothesis definitively: the bottleneck is not token count or wall-clock time, and the remaining
untested diagnostic is reading the model's actual raw output directly rather than trying another
budget-shaped fix

### REQ-ARC-FCP-5699-30: Reading The Actual Raw Output -- The Model Abandons The Required Interface, Not A Resource Problem At All

REQ-ARC-FCP-5699-29's own remaining untested diagnostic, and the operator's direct request: capture
and read the model's raw completion text on a failed refactor try -- the one thing this entire
7-REQ sub-thread (5699-23 through -29) had diagnosed everything AROUND but never actually looked at
directly.

**Implementation.** `LocalGGUFProposer` gained `last_raw_completion: str = ""`, set inside the
existing `_record_completion_diagnostics()` helper (REQ-ARC-FCP-5699-27) alongside
`last_stop_type`/`last_prompt_truncated` -- no new wiring needed in `generate()`/`complete_text()`
since both already call that helper on every response. 1 new unit test verifies the raw text is
captured even when `generate()` returns `ok=False` (the short failure message alone never showed
what was actually produced).

**Diagnostic method: replay the EXACT failing call with REAL data, not a fresh live run.** Rather
than spend another ~15-30 minute full A/B run, a standalone script
(`read_raw_refactor_output.py`) loaded the REAL 8-mismatch counterexample data already sitting in
`results/arc_sge_live_path_ab_g50t.json` (from REQ-ARC-FCP-5699-29's own run) and called
`LocalGGUFProposer.refactor("g50t", vr)` directly against the warm production server, at the SAME
`max_tokens=4096`/`timeout=600` REQ-ARC-FCP-5699-29 confirmed the model completes within (no
truncation) -- reproducing the exact failing conditions, once, cheaply.

**The raw output, read directly, reveals the actual root cause.** `last_stop_type='eos'`
(confirms REQ-ARC-FCP-5699-29 again: natural completion, not a token-budget cutoff) and
`last_raw_completion` (1761 characters, well under the 4096-token ceiling) contains:

```python
class WorldModel:
    def __init__(self):
        self.grid_size = 25
        self.grid = np.zeros((25, 25, 5), dtype=int)
        self.grid[12, 12, 0] = 1  # Start position
        self.grid[12, 12, 1] = 1  # Goal position
    ...
    def engine(self, state, action, data):
        ...
        elif action == 0:   # duplicate of the FIRST action==0 branch above -- unreachable
            # Move left
        ...
    ...
    def reset(self):
        self.grid_size = 25
        self.grid = np.zeros((25, 25, 5), dtype=int)
        self.grid[12, 12, 0] = 1  # Start position
        self.grid[12,           # <- cuts off mid-statement, no closing, no is_level_complete anywhere
```

**This is NOT a resource, truncation, or timeout problem -- it is the model abandoning the
required task interface entirely.** Instead of patching the top-level `def engine(grid, action,
data)` / `def is_level_complete(grid)` functions the prompt requires, the model invented a
self-contained `class WorldModel` with `self`-bound methods, a completely fictional 25x25x5
one-hot internal grid representation that matches NOTHING about g50t's real state (every prior
generation this REQ chain has read used the real 2D color-indexed grid), and NEVER writes
anything resembling a win-condition check -- `is_level_complete` does not appear anywhere in the
1761 characters. The response also contains an internal bug even on its own fictional terms
(`elif action == 0:` duplicated, making the intended "move left" branch permanently unreachable
behind the first `action == 0` check), and simply stops mid-statement without the model appearing
to notice anything is wrong. This looks like the model drifting into a generic, memorized
gridworld/RL-environment template under a complex debugging prompt, rather than genuinely
engaging with the specific counterexamples it was shown.

**What this narrows -- the complete answer across the whole 5699-23 through -30 sub-thread.**
Every resource-shaped hypothesis (thin training data, no repair loop, invalid JSON, token budget,
wall-clock timeout) has now been tried, and each was either a real, worthwhile, KEPT fix (the JSON
bug, the detection instrumentation) or a clean negative result. None of them touched the actual
cause, because the actual cause isn't resource-shaped at all: it's an instruction-adherence
failure under a complex, counterexample-heavy reasoning prompt, on this specific 9B local model.
This is a genuinely different problem class from anything this REQ chain could have found by
tuning parameters -- it required reading the model's own words.

**Honest scope limit.** n=1 game (g50t), n=1 raw-output sample. One failed generation showing
this specific pathology (class-wrapped, fictional-representation drift) does not establish this
is the ONLY or even the DOMINANT failure mode across all of round 2's failed tries in this REQ
chain -- REQ-ARC-FCP-5699-25's original failure and REQ-ARC-FCP-5699-28's `n_predict`-truncated
failure could plausibly be different manifestations (or the same one, cut off at different
points). A single sample is a real, informative data point, not a complete characterization.

**Concrete next step if this thread continues.** (a) Cheapest: strengthen `refactor_prompt`'s
own instructions with an explicit, repeated reminder of the EXACT required top-level signature
immediately before/after the mismatches block (e.g. "Your response MUST define exactly these two
TOP-LEVEL functions: `def engine(grid, action, data):` and `def is_level_complete(grid):` -- do
NOT wrap them in a class, do NOT rename the parameters"), directly targeting the observed
class-wrapping drift. (b) Capture MORE raw-output samples (repeat this diagnostic 3-5 times, or
across cd82/sp80 too) to see whether this specific pathology (class-wrapping, fictional grid
representation) recurs, or whether other failed tries show different symptoms entirely. (c) Given
how far this sub-thread has gone (8 REQs now) without a level-up, this is also a natural point to
weigh continuing to iterate on prompt-engineering fixes for this ONE model's reliability against
testing generalization elsewhere or stepping back to a different priority.

#### SCENARIO-ARC-FCP-5699-30-RAW-OUTPUT-REVEALS-INTERFACE-ABANDONMENT-NOT-A-RESOURCE-PROBLEM

Given `LocalGGUFProposer` gains a `last_raw_completion` capture (alongside the existing
`last_stop_type`/`last_prompt_truncated` from REQ-ARC-FCP-5699-27), and the exact failing
`refactor()` call from REQ-ARC-FCP-5699-29's own run is replayed with the same real
counterexample data and the same confirmed-non-truncating `max_tokens=4096`/`timeout=600`
When the raw completion text is read directly rather than only checking whether it parsed
Then `last_stop_type='eos'` (natural completion, not truncated) and the 1761-character response
shows the model wrote a `class WorldModel` with `self`-bound methods and a fictional 25x25x5
one-hot grid representation instead of the required top-level `engine`/`is_level_complete`
functions, never mentions `is_level_complete` at all, contains an internal bug (a duplicated,
unreachable `elif action == 0:` branch), and stops mid-statement -- establishing that round 2's
failure across this entire REQ chain is an instruction-adherence/task-drift problem, not a
resource, truncation, or timeout problem, and that no further budget-shaped fix could have found
this without reading the model's actual output

### REQ-ARC-FCP-5699-31: The Targeted Prompt Fix -- Structural Reminder Directly Countering The Observed Pathology

REQ-ARC-FCP-5699-30's cheapest named next step, and the operator's direct pick: strengthen
`refactor_prompt` with an explicit reminder of the exact required signature, directly targeting
the class-wrapping/invented-representation drift the raw output revealed.

**Implementation.** `refactor_prompt` gained a DEV-ONLY `CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=1`
env override (unset -> exact pre-existing prompt, verified byte-identical by a dedicated test).
When enabled, two blocks are spliced in: `_REFACTOR_STRUCTURE_REMINDER_BEFORE` (primacy position,
right after the task description, before the mismatches) states the exact required top-level
signature and explicitly forbids the THREE specific things REQ-ARC-FCP-5699-30's raw sample did
wrong -- wrapping in a class, using `self`, and inventing a new grid representation instead of
reusing the real one -- and `_REFACTOR_STRUCTURE_REMINDER_AFTER` (recency position, immediately
before the model starts generating) restates the required-functions-only constraint briefly. This
is a CONTENT/structure reminder, not the `codeonly` "skip all reasoning" directive REQ-ARC-FCP-
5699-26 confirmed is deliberately excluded from refactor() -- the model is still told to reason
about the mismatches, just reminded what shape the final answer must take. 2 new tests verify the
unset-default is byte-identical and the enabled reminder names the specific forbidden structures.

**Live test: replay the SAME real g50t counterexample data used in REQ-ARC-FCP-5699-30's
diagnostic, with the reminder enabled, at the same confirmed-non-truncating
`max_tokens=4096`/`timeout=600` -- same cheap single-call method, not a full A/B, for a direct,
apples-to-apples comparison against 5699-30's exact sample.**

**Result: `ok=True` -- the structural pathology is fixed.** The response starts DIRECTLY (no
preamble) with a proper top-level `def engine(grid, action, data):` / `def is_level_complete
(grid):` pair -- no class, no `self` -- and `_extract_python` (which takes everything up to the
first bare closing ` ``` `) grabbed exactly that first, complete, syntactically valid block and
wrote it to `results/arc_e3/g50t/world_model.py`. This directly reverses REQ-ARC-FCP-5699-30's
sample: `is_level_complete` is present this time (checks whether all entities are cleared from
the grid), and there is no class/`self` wrapping anywhere. `refactor()` returned `(True, "local
gguf (GPU server) wrote world_model.py")` -- the SAME success path a genuinely useful refactor
round takes.

**Two honest caveats, not glossed over.** (1) `last_stop_type='limit'` this time (unlike 5699-30's
`'eos'`) -- the model DOES hit the 4096-token cap, but only AFTER the first valid block was
already complete: past that point, the response spirals into a REPETITIVE SELF-CORRECTION LOOP
("Wait, I need to check the logic again... Let me rewrite the code" repeated near-verbatim 5-6
times, revising the same `_move_entities` helper with only cosmetic differences each pass) until
truncated mid-loop. This did not hurt THIS specific outcome (`_extract_python` only cares about
the first fenced/pre-fence block), but is a new, previously-unobserved behavior -- a different
sample could plausibly have the rambling start BEFORE the first valid block completes. (2)
Structural correctness is fixed, but CONTENT correctness is still unverified: the code invents a
`[row, col, entity_id, state]` per-cell list-of-lists representation, which does not match g50t's
real simple 2D color-indexed grid (every prior successful induction this REQ chain read used the
real format) -- so while this response would be ACCEPTED as syntactically valid code by
`generate()`'s check, it is not established whether it would pass the strict
`min_heldout_accuracy=1.0` downstream verification gate against the real transitions.

**What this narrows.** The targeted reminder is a real, measurable, single-sample improvement on
the EXACT failure REQ-ARC-FCP-5699-30 found (class-wrapping, missing `is_level_complete`) -- this
is not a null result. Whether it reliably produces a plan-worthy (not just syntactically-valid)
engine, and whether the new repetitive-looping behavior is a stable side-effect or a one-off, are
both genuinely open after n=1.

**Honest scope limit.** n=1 sample, same as REQ-ARC-FCP-5699-30's baseline -- this compares one
reminder-enabled generation against one non-reminder generation, both real but both single draws
from a demonstrably high-variance process (REQ-ARC-FCP-5699-23's k=20 test already showed one
generation succeed and an independent one crash on the identical config). A full live A/B (both
arms, multiple induction attempts across a real budget=250 run) is the natural validation this
single-sample result invites, but was NOT launched automatically here given how much of this
sub-thread's time budget (now 9 REQs on one game) has already gone to g50t specifically -- left as
an explicit choice for the operator rather than defaulted into.

#### SCENARIO-ARC-FCP-5699-31-TARGETED-REMINDER-FIXES-THE-STRUCTURAL-PATHOLOGY-CONTENT-STILL-OPEN

Given the SAME real g50t counterexample data REQ-ARC-FCP-5699-30 used (which produced a
class-wrapped, `is_level_complete`-free, syntactically-incomplete response) is replayed with
`CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=1` enabled
When the raw completion is read directly, as before
Then the response begins with a valid top-level `engine`/`is_level_complete` pair (no class, no
`self`), `generate()` returns `ok=True`, and the code is successfully written to disk -- directly
reversing the specific structural failure the reminder targeted -- while ALSO revealing a new
repetitive self-correction loop that consumes the remaining token budget after the first valid
block (harmless here, but previously unobserved) and leaving the code's CONTENT correctness (an
invented grid representation inconsistent with g50t's real format) as a separate, still-open
question that only a full live A/B could answer

### REQ-ARC-FCP-5699-32: The Full Live A/B -- Structural Fix Generalizes Completely; No Level-Up, Content Accuracy Is The Real Remaining Wall

The natural validation REQ-ARC-FCP-5699-31 named but didn't launch automatically, given nine REQs
already spent on one game: a full live A/B on g50t, `budget=250`, combining every fix this
sub-thread validated -- `CARNOT_ARC_STALL_REFACTOR_LOOP=1` (5699-25), `CARNOT_ARC_INDUCE_
MAX_TOKENS=4096`/`CARNOT_ARC_INDUCE_TIMEOUT=600` (5699-29), and `CARNOT_ARC_REFACTOR_
STRUCTURE_REMINDER=1` (5699-31).

**Result, read only after confirmed genuine completion (`ps aux` showed the process had exited
AND the task log showed `"wrote results/..."`, per the discipline established after this
session's own earlier stale-read mistake):**

```
baseline: refinement_rounds_used=3, skipped=no_reachable_plan_after_refinement, planned=False
  round 1 (induce):   ok=True, heldout_accuracy=0.125, accepted=False
  round 2 (refactor): ok=True, heldout_accuracy=0.0,   accepted=False
  round 3 (refactor): ok=True, heldout_accuracy=0.0,   accepted=False
sge: refinement_rounds_used=3, skipped=no_reachable_plan_after_refinement, planned=False
  round 1 (induce):   ok=True, heldout_accuracy=0.0, accepted=False
  round 2 (refactor): ok=True, heldout_accuracy=0.0, accepted=False
  round 3 (refactor): ok=True, heldout_accuracy=0.0, accepted=False
```

**The structural fix generalizes completely -- this is not n=1 luck.** All 6 generation attempts
across this full run (3 rounds x 2 arms) return `proposer_ok=True` -- every single one produces
syntactically valid code containing both required functions. This is a total reversal from every
prior measurement in this REQ chain: `skipped` is now `"no_reachable_plan_after_refinement"` (the
loop genuinely ran all 3 rounds and never accepted a candidate) rather than `"proposer_failed"`
(the loop couldn't even get valid code out of the model). The class-wrapping/missing-function
pathology REQ-ARC-FCP-5699-30 found is gone, consistently, across every round of a real
full-budget run -- not just the single sample REQ-ARC-FCP-5699-31 checked.

**Honest bottom line: NO LEVEL-UP.** `levels=0`/`reached=0` for both arms, `planned=False`
throughout. `heldout_accuracy` never rises above 0.125 across all 6 rounds -- the induced dynamics
are syntactically valid but functionally WRONG, and three rounds of genuine counterexample-driven
refinement (which now actually execute, unlike before) do not converge toward the real g50t
mechanics. **This confirms REQ-ARC-FCP-5699-31's own caveat exactly**: fixing the interface
problem (structure) does not fix the harder, still-unsolved problem (content) -- the model can now
reliably WRITE code in the right shape, but still cannot correctly INFER g50t's actual game rules
from 25 real transitions within 3 refinement rounds. Per CLAUDE.md's ARC Solve Reproducibility
discipline, this is explicitly NOT a level-up to bank in `ops/arc_solve_registry.yaml` or headline
in `ops/status.md` -- there is no level-up to report.

**What this closes out.** This is the definitive end-to-end answer for the entire REQ-ARC-FCP-
5699-23 through -32 sub-thread (10 REQs on one game): the resource-budget hypotheses (data
amount, repair rounds, JSON validity, token/time budget) were all real diagnoses but not the
actual blocker; the interface/structure problem (REQ-ARC-FCP-5699-30) WAS a real blocker and IS
now fixed, reliably, at zero cost to structural correctness; and what remains is squarely a model-
capability problem -- correctly inferring a novel game's dynamics from a small transition sample
-- that no amount of prompt engineering around output FORMAT can address, because the problem is
in what the model UNDERSTANDS, not how it's asked to phrase the answer.

**Honest scope limit.** n=1 game (g50t), n=1 full-budget run at this exact combined
configuration. The structural fix's reliability is now well-evidenced (6/6 in this run, plus the
1/1 prior sample); the content/accuracy ceiling has only been measured on this one game's specific
mechanics and this one real transition sample.

**Concrete next step if this thread continues.** The structural fix (REQ-ARC-FCP-5699-31) is worth
keeping and could reasonably graduate toward becoming a default, given its now-repeated success --
but the underlying capability gap (content accuracy) is a fundamentally different, harder problem
than anything a prompt tweak can solve. Plausible directions: (a) accept this 9B local model has a
real capability ceiling on first-contact game-dynamics induction and treat that as a boundary
condition rather than a bug to keep chasing; (b) test whether a LARGER model (this project has
`Qwen3.6-27B-MTP`/`gemma-4-31B-it` queued for other purposes) closes the accuracy gap on the SAME
counterexample data, isolating whether this is a scale-of-model problem; (c) step back from this
specific sub-thread entirely -- ten REQs deep on one game's tier-2 induction is a natural
stopping point regardless of which of (a)/(b) is chosen next.

#### SCENARIO-ARC-FCP-5699-32-STRUCTURAL-FIX-GENERALIZES-CONTENT-ACCURACY-REMAINS-THE-WALL

Given every validated fix from this sub-thread (stall-refactor-loop wiring, raised max_tokens/
timeout, the structural reminder) is combined in a single full-budget live A/B on g50t
When the run is read only after confirmed genuine completion, both arms
Then all 6 generation attempts (3 rounds x 2 arms) return `proposer_ok=True` -- the structural
fix generalizes completely, not just the single prior sample -- `skipped` changes from
`proposer_failed` (couldn't generate valid code) to `no_reachable_plan_after_refinement` (valid
code every round, never accurate enough to accept) -- yet `heldout_accuracy` never exceeds 0.125
across any round and `planned` stays `False` for both arms: NO level-up, confirming the remaining
blocker is a genuine model-capability gap in inferring correct game dynamics from a small
transition sample, not anything a prompt-format fix can address

### REQ-ARC-FCP-5699-33: Does A Larger Model Close g50t's Accuracy Ceiling? -- Parity On Round 1, A New Truncation Failure On Round 2

REQ-ARC-FCP-5699-32's own concrete next-step option (b): test whether a LARGER model closes the
content/accuracy gap on g50t specifically, isolating whether the wall is model-scale or something
deeper. Reused the real production path (`E3AgentPolicy(game, proposer=<explicit override>)` --
the frozen `_proposer()` default in `arc_competition_agent.py` is completely untouched, since an
explicit `proposer=` argument bypasses the lazy default entirely) with the SAME combined-fix
config as REQ-ARC-FCP-5699-32 (`CARNOT_ARC_STALL_REFACTOR_LOOP=1`, raised max_tokens/timeout,
`CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=1`), swapping only the proposer's model to
`Qwen3.6-27B-MTP-GGUF` (~3x the frozen live generator's parameter count).

**Honest scope departure from "the same counterexample data."** Unlike REQ-ARC-FCP-5699-30/31's
`read_raw_refactor_output.py` precedent (which replayed a saved counterexample byte-for-byte),
this ran a FRESH live episode end-to-end (same game, same combined-fix config, a NEW explorer run
collecting its own real transitions) rather than literally reloading REQ-ARC-FCP-5699-32's saved
counterexample dict. This is a defensible design choice (a fresh full-loop run is more realistic
than a synthetic single-call replay) but is NOT literally "the same counterexample data" and is
recorded here as an honest scope note, not glossed over.

**A real, independently-rediscovered VRAM bug (already known, not novel).** The first launch
attempt used `mtp=True` (matching the live generator's config) and OOM'd:
`ggml_backend_cuda_buffer_type_alloc_buffer: allocating 15621.78 MiB on device 0: cudaMalloc
failed: out of memory` -- MTP's self-draft mechanism loads the GGUF file TWICE (main + draft), so
a 16GB Q4_K_M file needs ~32GB, exceeding a single 24GB RTX 3090. This is the EXACT bug
`REQ-ARC-WMTE-5596` (below) already found and fixed in its own experiment script
(`python/carnot/experiment_5596_generator_size_ab_gemma31b_vs_current.py`,
`_candidate_mtp_self_draft_fits_vram`) one milestone earlier -- this session's diagnostic script
was built standalone and hit the same wall independently before this cross-reference was found.
Fixed the same way: `mtp=False` for this model (costs decode throughput, not capability).

**Result, read only after confirmed genuine completion (`ps aux` showed no matching process AND
the task log showed `"wrote results/..."`):**

```
larger_model_qwen3.6_27b_mtp: refinement_rounds_used=2, skipped=proposer_failed, planned=False
  round 1 (induce):   ok=True,  heldout_accuracy=0.125, accepted=False
  round 2 (refactor): ok=False, heldout_accuracy=None,  skipped=proposer_failed
    message: "local model code unusable after 3 tries (missing ('engine', 'is_level_complete')
              in output [HIT n_predict=4096 OUTPUT LIMIT before completing])"
levels=0, reached=L0, actions=248, duration_s=388.9
```

**Round 1: parity, not improvement.** The 27B model's induce-round `heldout_accuracy=0.125` is
IDENTICAL to the ceiling every round of REQ-ARC-FCP-5699-32's fully-fixed 9B config also hit (0.0
to 0.125 across all 6 of that run's rounds) -- no accuracy gain on this game, in contrast to
`REQ-ARC-WMTE-5596`'s own finding that this same 27B candidate showed MATERIALLY HIGHER
`heldout_accuracy` than the 9B generator on two OTHER games (m0r0: 0.0->0.5; sk48: 0.2->1.0).
g50t's content-accuracy wall does not respond to model scale the way m0r0/sk48's did.

**Round 2: a NEW failure mode, not previously seen at this budget.** The 27B model's refactor call
hit the SAME truncation pathology REQ-ARC-FCP-5699-27/28/30 spent three REQs diagnosing and fixing
for the 9B model -- but REQ-ARC-FCP-5699-32 confirmed the 9B model's refactor calls complete
cleanly within 4096 tokens (all 6/6 rounds `proposer_ok=True`) once the structural reminder was
added. The 27B model's refactored code apparently needs MORE than 4096 tokens to complete --
plausibly because a larger model produces more verbose/detailed code, or reasons longer before
writing it, even with the same `/no_think` prefix and structural reminder. This budget was tuned
(REQ-ARC-FCP-5699-29) against the 9B model only; it was never re-validated against a 3x larger
model, and this result shows that assumption does not carry over silently.

**No level-up.** `levels=0`, consistent with every other measurement in this sub-thread. Per
CLAUDE.md's ARC Solve Reproducibility discipline, nothing here is a countable/reproducible
registry level -- there is no level-up to report in `ops/arc_solve_registry.yaml` or
`ops/status.md`.

**Honest scope limit (Sample-Size Rigor).** n=1 run, n=1 game, n=1 model swap -- this alone does
NOT prove model scale is irrelevant to g50t's wall, only that it did not help on this one sample,
while the SAME model scale-up DID help on two other games in a separate, earlier experiment. The
two findings are not in conflict -- they show the accuracy ceiling is evidently game-dependent,
not a uniform model-capability floor. Concrete next steps if this is worth pursuing further: (a)
re-run with `CARNOT_ARC_INDUCE_MAX_TOKENS` raised specifically for the 27B model (e.g. 8192) to
see whether round 2 completes and produces a materially different accuracy than round 1's 0.125;
(b) accept g50t specifically as a capability boundary (independent of model scale) and redirect
effort to games more like m0r0/sk48 where scale demonstrably helps; (c) step back from per-game
induction-accuracy chasing entirely, per REQ-ARC-FCP-5699-32's option (c).

Required field principles:

- `inference_substrate: live_llm_inference`: principle "a real GGUF model genuinely loaded and
  generated on a live llama-server -- distinguishes this from an aggregation/replay artifact."
- `model_specs`: principle "names the exact GGUF and its mtp setting (False, and why) so a reader
  does not assume this used the same decoding mechanism as the frozen live generator."

#### SCENARIO-ARC-FCP-5699-33-LARGER-MODEL-PARITY-NOT-IMPROVEMENT-ON-G50T

Given the same combined-fix config validated in REQ-ARC-FCP-5699-32, with only the proposer's
model swapped to a ~3x larger candidate (`Qwen3.6-27B-MTP-GGUF`, `mtp=False` after an OOM
confirmed self-draft does not fit alongside the base weights on a single 24GB GPU)
When a fresh live episode runs on g50t and reaches the stall-refactor-loop
Then round 1 (induce) succeeds with `heldout_accuracy=0.125` -- parity with, not improvement over,
the 9B model's own ceiling on this game -- while round 2 (refactor) fails on prompt-budget
truncation the 9B model does not hit at the same token budget, and no level-up occurs; the
accuracy ceiling on g50t is therefore evidenced as game-specific rather than resolved by model
scale, in contrast to `REQ-ARC-WMTE-5596`'s positive scale-up finding on two other games

### REQ-ARC-FCP-5699-34: Raising max_tokens To 8192 For The Larger Model -- A Self-Caught Confound, Then A Clean Structural Fix

Direct operator follow-up to REQ-ARC-FCP-5699-33: raise `max_tokens` to 8192 specifically for the
27B model and re-run round 2 (the refactor call that hit `[HIT n_predict=4096 OUTPUT LIMIT before
completing]`). Rather than a full fresh episode (which would also draw a new, different round-1
counterexample), this replayed ONLY round 2 -- a targeted single `refactor()` call reusing the
REAL round-1 counterexample already saved in `results/arc_larger_model_g50t_ab.json` (`real_n=25`,
`real_n_correct=5`, `real_accuracy=0.2`, 8 capped mismatches), matching the
REQ-ARC-FCP-5699-30 `read_raw_refactor_output.py` replay pattern.

**First attempt: a self-caught confound, not a finding about token budget alone.** The first
replay script raised `max_tokens` to 8192 but forgot to set
`CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=1` -- REQ-ARC-FCP-5699-31's fix, which is read fresh at
call-time inside `refactor_prompt()` via `os.environ.get(...)`, not baked into any object state.
Without it, the model reverted to inventing its own function (`def solve(grid, action,
data=None)`) instead of the required `engine`/`is_level_complete` -- exactly the pathology the
reminder exists to prevent. `last_stop_type='eos'` (natural completion, NOT truncated) proved 8192
tokens was enough room, but `ok=False` (missing required functions) proved the reminder is what
was actually load-bearing at round 2, not budget alone. This was caught and corrected before
drawing any conclusion from it -- flagged here explicitly as a self-introduced test defect, not
buried.

**Corrected, clean, timed re-run (`results/arc_larger_model_g50t_round2_replay_8192.json`,
`duration_s=195.95`, read only after confirmed genuine completion):** with BOTH fixes applied
(`max_tokens=8192` AND `CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER=1`), round 2 succeeds structurally:
`proposer_ok=True`, `last_stop_type='eos'`, `last_prompt_truncated=False`, both required functions
present.

**But: the induced code is a hardcoded fit to the shown examples, not a general rule.** Manual
read of the raw completion shows `engine()` hardcodes literal row/col ranges lifted directly from
the 8 shown counterexample mismatches (e.g. `for r in range(8, 10): for c in range(14, 19): if
grid[r, c] == 9: grid[r, c] = 5`) rather than deriving a spatial/color rule that would generalize
to unseen cells. This is consistent with memorizing the shown failing examples rather than
inferring g50t's actual dynamics. `heldout_accuracy` was NOT independently re-measured in this
targeted replay (would require the real 25-transition corpus, not just the capped mismatch
summary this replay reused) -- this overfitting read is an informed expectation based on the
code's structure, not a re-measured result, and is labelled as such in the artifact rather than
presented as a finding.

**What this settles and doesn't.** Settles: raising `max_tokens` to 8192 for the 27B model DOES
fix round 2's structural completion failure, but ONLY together with the structural reminder --
neither fix alone was sufficient (this session tested both orderings). Does NOT settle: whether
the now-completing round 2 candidate would actually raise `heldout_accuracy` above the 0.125
ceiling REQ-ARC-FCP-5699-33's round 1 measured -- the code's hardcoded-coordinate structure makes
a further increase implausible, but this is a read of the code, not a measurement.

Required field principles:

- `overfit_risk_note`: principle "distinguishes an informed expectation (a code read) from a
  measured result (a scored heldout run); conflating the two would overclaim a finding this
  replay did not actually measure."
- `duration_s`: principle "a genuinely time-measured artifact, not a wall-clock estimate from
  task-notification timing -- this was written specifically because the two exploratory replay
  runs that preceded it were not separately artifact-backed."

#### SCENARIO-ARC-FCP-5699-34-BUDGET-ALONE-INSUFFICIENT-REMINDER-ALSO-REQUIRED

Given round 2's real counterexample from REQ-ARC-FCP-5699-33, replayed with `max_tokens` raised to
8192
When the structural reminder env var is OMITTED, the model completes naturally (not truncated) but
invents its own function name instead of the required interface
And when the SAME replay is re-run with the structural reminder correctly set
Then round 2 completes both naturally AND structurally correctly -- proving raising the token
budget and applying the structural reminder are BOTH independently necessary for this larger
model, neither sufficient alone, and the resulting code's hardcoded example-coordinate structure
means this structural fix should not be read as having also closed the accuracy gap

### REQ-ARC-FCP-5699-35: Graduate REQ-25/29/31 To The Live Agent Default -- With A Real Robustness Fix Found Along The Way

The 25-game public roster is near-saturated for further per-game deepening (REQ-ARC-FCP-5699-33's
own registry audit: only lf52/sk48/bp35 aren't `full_game_clear`, and all three already have a
recent, dedicated deepening attempt on record that found nothing -- re-attempting without a new
angle would be a doomed rerun). Operator-directed pivot: instead of chasing one more banked level,
graduate the three validated, zero-regression fixes from this sub-thread out of dev-only env-var
gating and into the actual production default -- improving the SCORED hidden-game agent's
induction robustness across ALL games, not one.

**The three graduations (all keep an explicit opt-out, matching the existing `CARNOT_ARC_MTP=0`
pattern used elsewhere in the same files):**

1. `_proposer()`'s `max_tokens`/`timeout` defaults (`arc_competition_agent.py`): `2560`/`300` ->
   `4096`/`600`. REQ-ARC-FCP-5699-32 measured 4096/600 reaches `proposer_ok=True` on 6/6 rounds of
   a real full-budget live run for THIS (9B) model -- the 8192 requirement REQ-ARC-FCP-5699-34
   found was specific to a 3x larger, non-live candidate, not this one. Both env vars remain
   overridable.
2. `refactor_prompt`'s structural reminder (`arc_executable_world_model.py`): default-on
   (`CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER` now `!= "0"` instead of `== "1"`). REQ-31/32 found
   zero regressions across a full real run.
3. The stall-triggered bounded-refinement loop (`arc_competition_agent.py`): default-on
   (`CARNOT_ARC_STALL_REFACTOR_LOOP` now `!= "0"` instead of `== "1"`) -- routes first-contact
   induction through `execute_bounded_llm_reinduction`'s counterexample-driven refinement instead
   of one single-shot attempt.

**A real, self-caught bug found DURING graduation, not before it.** The dev-only version of (3)
unconditionally `return`ed after the refinement-loop block regardless of whether it actually
reached a plan. That was safe while opt-in (an operator explicitly traded the pre-existing
`active_probe_controller`/plain single-shot fallback path away for the duration of a diagnostic
run) but would have been a REAL regression once graduated to the always-on default: REQ-32's own
measurement showed the strict `min_heldout_accuracy=1.0` gate this loop uses is rarely met (0/6
rounds across a full g50t run), so EVERY non-planned stall would have silently discarded a
separate, valuable, pre-existing mechanism -- a trust-energy-based world-model selector that does
NOT require a perfect candidate, does NOT require `.refactor()` to exist, and was reachable for
every prior stall event. Fixed by changing the unconditional `return` to a conditional one: return
only when `stall_outcome.planned` is `True`; otherwise fall through to the unchanged
`active_probe_controller` / plain single-shot path below, exactly as before this REQ.

**A second, deeper bug found via the test suite, not by inspection.** Even with the fallthrough
fix, a REAL regression surfaced in `tests/python/test_arc_world_model_trust_energy.py::
test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate`: `execute_bounded_llm_reinduction`
CAN raise mid-call (e.g. its round-2 `proposer.refactor(...)` call, if the round-1 candidate
doesn't clear the strict gate) -- and an exception raised INSIDE the call propagates straight past
my conditional-return fix (which only handles the CLEAN `planned=False` return case) to this
method's OUTERMOST exception handler, aborting the entire `_induce_and_plan()` attempt and
skipping the fallback path just the same, just via a different mechanism (an exception instead of
a clean return). Root-caused via a temporary debug `traceback.print_exc()` in the outer handler
(added, used, then reverted -- never left in the committed diff): `AttributeError: '_FakeProposer'
object has no attribute 'refactor'` in the test's case, but any real exception during refinement
(model timeout, malformed response, etc.) would trigger the identical failure mode in production.
Fixed with a SECOND, LOCAL `try/except` around just the `execute_bounded_llm_reinduction` call
itself: any exception now falls through to the plain path exactly like a clean `planned=False`
outcome, recording `stall_refactor_loop_error` on the attempt (matching the existing non-fatal
`program_synthesis_filter_error`/`ttt_prior_engine_error` pattern already used elsewhere in this
same method). This is a genuine production robustness improvement, not merely a test fix -- the
bug would have fired on ANY refinement-round exception in the live agent, not just this test's
specific stub gap.

**A third, independent drift bug found by the test suite's own "configured IDENTICALLY" contract.**
`_load_sge_candidate_router()` builds its OWN separate `LocalGGUFProposer` and its own test
(`test_req_arc_fcp_5699_11_load_sge_candidate_router_reuses_frozen_generator_config`) asserts this
must match `_proposer()`'s config "IDENTICALLY" (so both share one warm llama-server, never a
second model load). It turned out to hardcode `max_tokens=2560` as a bare literal and never set
`timeout` at all -- so when `_proposer()`'s default moved to 4096/600, this function silently
drifted out of sync, exactly the failure mode its own docstring warns against. Fixed to read the
SAME env-var-overridable defaults `_proposer()` now does.

**Verification (thorough, given this touches the scored hidden-game agent):**

1. The 5 original test files (78 tests): 5 pre-existing tests needed updating because their own
   premise ("unset == old/disabled behavior") is now intentionally false -- each was rewritten as
   a graduated-default test plus a companion explicit-opt-out test preserving the old behavior
   byte-for-byte, rather than deleted. All 78 pass.
2. A repo-wide ARC/submission-package sweep (`tests/python/ -k "arc_ or arc3 or submission_package
   or 4544 or 4744"`, ~3880 tests) found 38 failures before the local try/except fix; the fix
   resolved 9 of them (all sharing the same root cause -- the propagating-exception bug above) with
   ZERO new failures introduced. The remaining 29 were cross-checked against a clean `git worktree`
   at the pre-graduation commit: 27 fail identically on the unmodified baseline (confirmed
   pre-existing, unrelated drift -- e.g. a stale committed artifact's schema, a growing
   generic-operator registry's selection ordering) and the other 2
   (`test_off_arc_execution_verifier_transfer_accumulate.py`) pass cleanly in isolation on BOTH the
   clean baseline and the modified code, confirming they were parallel-batch test-pollution
   artifacts, not a real regression.
3. A direct memory-growth probe (4 repeated calls to `_induce_and_plan()` through the graduated
   path) found the pytest memory-watchdog's one CI-relevant hit
   (`test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate`'s teardown, +582MB) is a
   ONE-TIME import-warming cost (heavy candidate-scoring/DSL-fitting modules imported for the first
   time), not a real per-call leak: RSS grew +810MB on the first call, then +43MB, +2MB, +0MB on
   calls 2-4 -- completely flat. Not a production risk for a long-running Kaggle session; the cost
   is paid once, not accumulated.

Required field principles:

- `stall_refactor_loop_error`: principle "records WHY the bounded-refinement attempt was abandoned
  when it fails via exception rather than a clean planned=False, so a reviewer can distinguish
  'the model tried and failed the strict gate' from 'the mechanism itself broke' without needing to
  reproduce the failure."

#### SCENARIO-ARC-FCP-5699-35-GRADUATION-PRESERVES-THE-PRE-EXISTING-FALLBACK

Given the three REQ-25/29/31 fixes graduated from dev-only opt-in to the production default
When a stall-triggered induction attempt's bounded-refinement loop either (a) completes cleanly
without reaching a plan, or (b) raises an exception mid-round (e.g. a proposer that cannot complete
a refactor round)
Then in BOTH cases execution falls through to the pre-existing `active_probe_controller` / plain
single-shot trust-energy-selector path exactly as it did before this REQ -- the graduation is
strictly additive, never silently discarding a working fallback mechanism, verified by both a
targeted regression test and a clean-worktree-diffed repo-wide sweep showing zero new failures

### REQ-ARC-FCP-5699-37: Color-Blob Salience Re-Validation -- Performance Fixed, Stays Disabled For A Different, More Precise Reason

Operator-selected item 2 from `ops/known-issues.md`'s CURRENT ACTIVE PRIORITIES ("the main new
lever"): re-validate `SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED` (disabled since 2026-07-14 after a
near-hang on the local submission gate) after its first fix, and decide re-enable / keep disabled
/ needs further optimization.

**A second, separate O(candidates x grid_cells) recomputation bug, missed by the 2026-07-14 fix.**
Profiling a real lp85 budget=500 episode (`cProfile`, 43.3s total) found `connected_color_blobs()`
called **8176 times for only 500 actions** -- not once per frame as the 2026-07-14 fix intended,
but once per CANDIDATE. Root cause: `arc_frame_change_predictor.rank_arc_actions` ->
`_prior_value` calls `ColorBlobSaliencePrior.score(frame, candidate)` via the generic two-arg
protocol shared across action-prior classes (no `blobs`/`color_counts` cache args) -- a call path
`action_tier_rows()`'s own internal caching never touched. This single call site accounted for
23.1s of the 43.3s total profiled runtime.

**Two independent fixes, both in `arc_color_blob_salience.py`:**

1. **Vectorized `connected_color_blobs()`** via `scipy.ndimage.label` (one call per distinct
   color, 4-connectivity structure matching the original flood-fill's neighbor rule) replacing the
   pure-Python per-cell BFS. Verified field-for-field equivalent to the original across 300+
   randomized grids (varying size 1x1 to 64x64, 1-16 colors, min_pixels, max_component_fraction)
   plus realistic structured-frame cases, using a FRESH independent oracle re-implementation (not
   a copy of either implementation under test) -- `tests/python/test_arc_color_blob_salience_
   vectorized.py`. Honest tradeoff, not hidden: ~5.9x faster on a realistic structured ARC-like
   frame (16 blobs: 2.9ms -> 0.5ms), but SLOWER on a pathological near-uniform-random grid
   (thousands of near-singleton blobs) due to per-blob Python object construction overhead --
   real ARC games render structured puzzle grids, not per-pixel noise, so the realistic-frame
   speedup is what governs actual live-agent performance.
2. **A bounded module-level per-frame cache** (`_cached_blobs_and_counts`, keyed on grid shape +
   content bytes + min_pixels + max_component_fraction, max 8 entries) so the generic two-arg
   `score()` callers stop recomputing from scratch per candidate. `ColorBlobSaliencePrior` is a
   frozen dataclass shared across multiple caller modules, so the cache lives at module level
   rather than requiring frozen-dataclass mutation.

**Result: the wall-clock/timeout problem is genuinely solved.** lp85 budget=500 went
**68s -> 8.4s** (baseline: 6.1s) -- vectorization alone got to 29.7s, the frame cache closed the
rest of the gap. The full local submission gate (`arc_local_submission_gate.py --check`, all 8
canonical games, budget=8000, 115s cap each) now runs with **`timed_out: 0`** on every game with
the flag temporarily enabled for this measurement, where it previously timed out on 7/8.

**But re-enabling reveals a DIFFERENT, more precise reason to keep it disabled: a genuine
BEHAVIORAL regression, previously masked by the timeout.** With the confounding timeout gone, the
gate's verdict flips from "TIMED OUT" to a clean `FAIL: REGRESSION: lost CORE solves ['m0r0']`
(current: solved 6 vs baseline: solved 7; `core_solves_preserved: False`). The salience-reordered
exploration priority causes the agent to lose a game (m0r0) it otherwise reliably solves. This is
NOT a performance problem -- `median_actions_on_core` and the bonus solves (`cd82`, `ft09`,
`su15`) are essentially unchanged from the disabled-baseline run -- it is a genuine capability
cost from reordering which actions get tried first.

**Decision: keep `SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED = False`.** The speed fixes are kept and
shipped regardless (pure implementation-detail rewrites, zero behavioral risk on their own,
verified by 300+ equivalence tests plus the full existing salience test suite -- 43/43 passing on
the vectorization+cache tests, and the pre-existing 12 unrelated failures in the broader salience
test sweep confirmed present on a clean pre-fix worktree, not caused by this change) -- they make
any FUTURE re-validation of this feature cheap to re-run, and benefit every other caller of
`connected_color_blobs`/`ColorBlobSaliencePrior.score` regardless of whether this specific flag is
ever flipped. Re-enabling the feature itself now requires investigating and fixing the m0r0
regression specifically, not another performance pass.

Required field principles: none new -- this REQ ships pure internal performance fixes with an
unchanged public API (`connected_color_blobs`, `ColorBlobSaliencePrior.score` keep their exact
signatures and return types); no artifact schema changes.

#### SCENARIO-ARC-FCP-5699-37-SPEED-FIXED-CAPABILITY-REGRESSION-NEWLY-VISIBLE

Given `connected_color_blobs()` vectorized via `scipy.ndimage.label` and a bounded per-frame cache
added for the generic two-arg `score()` callers, both verified equivalent to the pre-fix behavior
When the full local submission gate re-runs all 8 canonical games with
`SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED` temporarily set `True`
Then every game completes within its 115s cap (`timed_out: 0`, down from 7/8 timed-out
pre-fix) -- the performance problem is genuinely solved -- but the gate's verdict is still a FAIL,
now for a newly-visible, different reason: `REGRESSION: lost CORE solves ['m0r0']`; the flag stays
disabled, and the follow-up work required is a behavioral investigation of the m0r0 regression,
not further optimization

### REQ-ARC-FCP-5699-38: Post-Submission Regression Investigation -- One Confirmed, Unit-Tested Bug Fixed; The Original Hypothesis Refuted

Kernel v9 (REQ-35's graduated fixes) scored **0.08**, down from the prior day's **0.12** -- a real
regression, investigated directly rather than assumed. The local submission gate that passed before
submitting could not have caught this: it sets `CARNOT_ARC_DISABLE_INDUCTION=1`, never exercising
the induction/LLM path REQ-35 changed at all.

**Hypothesis 1 (wall-clock cost of the graduated stall-refactor-loop): REFUTED by direct
measurement.** A real A/B (`E3AgentPolicy(game, proposer=None)` -- the actual production
lazy-default, matching what ships -- `CARNOT_ARC_STALL_REFACTOR_LOOP=1` current default vs `=0` the
pre-REQ-35 behavior, same game `sc25`, budget=250) found the CURRENT default was **544.75s FASTER**
(636.32s vs 1181.08s), not slower, with identical outcomes (`levels=0`, `actions=245` both arms).
The scored live agent never uses per-game `GameAdapter`s (offline-dev-twin-only mechanism per the
ARC Live-Path Reachability Discipline), so a public game under `proposer=None` is a representative
simulation of genuine first-contact induction on a hidden game -- this measurement directly bears on
the hypothesis and refutes it. `results/arc_stall_refactor_loop_cost_ab_sc25.json`.

**Hypothesis 2 (the stall-refactor-loop installs a goal bias from an untrustworthy induced
predicate): a REAL bug, found, unit-tested, and fixed -- but not what fired in the live repro.**
`_induce_and_plan()`'s stall-refactor-loop and `level_up_reinduction` call sites both installed
`stall_outcome.goal_predicate` / `outcome.goal_predicate` as a frontier bias whenever it was not
`None`, completely independent of `goal_predicate_satisfiable` (already computed by
`execute_bounded_llm_reinduction`) or `planned`. A real live repro on `sc25` confirmed this firing:
`goal_predicate_satisfiable=False`, `planned=False`, `skipped="hidden_state_trust_below_threshold"`,
yet the bias was replaced. Fixed at BOTH call sites (`outcome.goal_predicate is not None and
outcome.goal_predicate_satisfiable` / the stall-path twin) -- there is no principled reason an
unsatisfiable predicate is safe to install after a level-up but not on first contact. Proven via a
targeted unit test that FAILS on pre-fix code (confirmed directly: temporarily reverted, the test
fails with the exact bias-replacement assertion, restored, passes) and a corrected live re-repro
using object-IDENTITY comparison, not a None-check -- `E3AgentPolicy` always installs a DEFAULT
`RelationalGoalEnergy` bias at construction (`SUBMITTED_AGENT_CONFIG["goal_energy_enabled"]=True`),
so `goal_bias is not None` is ALWAYS true and was a flawed initial test signal, corrected before
trusting it.

**But the corrected live re-repro revealed the SAME `sc25` scenario's bias replacement came from a
THIRD, different, pre-existing call site** (`_install_goal_bias(is_done)` in the plain single-shot
path, gated only on world-model DYNAMICS-accuracy trust via `select_trusted_world_model`/
`WorldModelVerifier` -- never on GOAL satisfiability at all) -- NOT from the two now-fixed call
sites. This path predates REQ-35 entirely and would behave identically whether or not the
stall-refactor-loop ever ran, so it cannot by itself explain a REQ-35-correlated regression, though
it is the same class of defect and worth fixing regardless.

**A fix for the third call site was attempted, found to have a real design flaw via the test suite
itself, and shipped dev-only rather than unconditionally.** Reusing `_goal_satisfiability_check`
(the same bounded-BFS the LLM-induction paths already use) at the plain-path call site broke
`test_req_arc_wmte_4494_live_policy_uses_trust_energy_candidate`: not via an exception, but because
the bounded BFS, run against that test's deliberately-simplified mock engine, correctly-per-its-own-
logic determined the goal unreachable within the search bound -- when the goal WAS trivially
reachable in the test's simplified world. This exposes a genuine risk the first two fixes do not
share: an imperfect-but-still-useful REAL induced engine could analogously fail to prove a
genuinely-achievable goal reachable within `max_nodes`/`max_depth`, rejecting a good goal, not just
a bad one -- a false-negative risk requiring real matched-budget validation before it is safe as an
unconditional default, per this project's own standing REQ-25->32->35 dev-gate->validate->graduate
discipline. Shipped behind `CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK` (default unset,
byte-identical to before), not graduated.

**Honest bottom line.** One real, confirmed, unit-tested bug is fixed and shipped as an
unconditional default (the two `goal_predicate_satisfiable`-gated fixes -- pure hardening, zero
behavioral risk since they only ADD a condition on an already-computed, already-validated signal).
The original wall-clock hypothesis for the score regression is refuted. The regression's actual
cause remains genuinely unexplained after this investigation -- this is reported honestly rather
than retrofitting either finding into a confident causal story the evidence does not support.

Required field principles:

- `goal_predicate_satisfiable`-gated bias install: principle "an induced goal predicate that is
  diagnosably unsatisfiable must never bias exploration toward it -- pursuing a provably-unreachable
  goal is strictly worse than no bias at all, regardless of which code path induced the predicate."
- `CARNOT_ARC_PLAIN_PATH_GOAL_SATISFIABILITY_CHECK`: principle "the plain path's version of the same
  defect is real but its naive fix has a demonstrated false-negative risk against imperfect engines;
  dev-only gating prevents shipping an unvalidated behavior change on the strength of one
  investigation session, per this project's standing dev-gate->validate->graduate discipline."

#### SCENARIO-ARC-FCP-5699-38-WALL-CLOCK-REFUTED-GOAL-BIAS-BUG-CONFIRMED-THIRD-PATH-NEEDS-VALIDATION

Given kernel v9's score regression (0.12 -> 0.08) and the local gate's blind spot (induction
disabled, never exercising what REQ-35 changed)
When a real production-config A/B measures stall-refactor-loop wall-clock cost, and a corrected
(identity-based) live repro checks whether an unsatisfiable induced goal predicate gets installed as
a bias
Then the wall-clock hypothesis is refuted (current default is FASTER, not slower) and a real,
unit-proven bug is found and fixed at the two call sites reachable from the LLM-induction paths, but
the exact scenario that exposed it turns out to route through an unrelated, pre-existing THIRD call
site whose naive fix has its own demonstrated false-negative risk against imperfect engines -- ship
the proven fix as default, gate the risky one dev-only, and report the regression's cause as still
genuinely unexplained rather than overclaiming either finding as the answer

### REQ-ARC-FCP-5699-39: A Fourth Call Site Found -- The Actual Smoking Gun For The `sc25` Repro

Checking the raw `sc25` artifact's own `ttt_prior_engine`/`ttt_prior_engine_plan_diagnostics` fields
(already on disk from REQ-38, no re-run needed) found the REAL source of that repro's bias
replacement: `gated_engine_from_transitions` (tier-1, the CNN-warm-started prior, 2026-06-21, runs
FIRST on every `_induce_and_plan()` call, untouched by REQ-35) reported `gate: "PASS"` (dynamics
cell_recall=1.0), meaning `_install_goal_bias(_isdone)` fired unconditionally at that call site --
but the SAME attempt's `plan_in_model` diagnostics show `nodes_expanded=20009`,
`termination_reason="max_nodes_reached"`, `initial_goal_energy=1.0`,
`min_goal_energy_observed=1.0` -- the goal energy never improved once across a genuine 20,009-node
search. A degenerate goal predicate, installed as a bias, from the FIRST-tried, most-frequently-
reached call site in the whole method.

**Fixed cheaply, without the plain-path fix's false-negative risk.** Rather than a SEPARATE bounded-
BFS satisfiability probe (the plain-path fix's approach, which the test suite caught rejecting a
genuinely-reachable goal on an imperfect mock engine), this fix reuses evidence `plan_in_model`
ALREADY computes: move `_install_goal_bias` to AFTER the `_call_plan_in_model` call, and only skip
installing when the diagnostics show `used_goal_energy_search=True` AND the energy never improved
(`min_goal_energy_observed >= initial_goal_energy`). Missing/absent diagnostic fields (a planner that
doesn't support goal-energy search) default to installing, preserving old behavior when the signal
isn't available -- only positive evidence of a flat search suppresses the install. Shipped as an
unconditional default (unlike the plain-path fix): it doesn't run a second, independently-bounded
search that could disagree with the real planning attempt, so it doesn't share that fix's
demonstrated false-negative mechanism.

**Verification.** The target 87-test scope passes cleanly. A broad ~3900-test sweep initially showed
8 "new" failures against the last snapshot; every one was confirmed, via direct inspection AND a
clean-HEAD (fix fully reverted) re-run producing the IDENTICAL 8 failures, to be unrelated conductor
drift accumulated during the ~1+ hour between snapshots (a growing generic-operator registry
reordering an unrelated selection test; a `disable_induction` kwarg added to
`arc_local_submission_gate.py` -- a file untouched by this session -- desyncing an old test mock)
-- not caused by this fix.

**Honest bottom line, unchanged from REQ-38.** This is now the FOURTH confirmed instance of the same
bug class (install a goal bias without checking whether the goal is achievable), and the one most
directly implicated by the `sc25` repro specifically. But like the plain path, this tier-1 mechanism
predates REQ-35 by weeks and is untouched by it -- it would have degraded exploration identically
yesterday, when the score was 0.12. It remains a real, worthwhile fix regardless; it still does not,
by itself, explain the REQ-35-correlated regression.

Required field principles:

- `ttt_prior_goal_bias_installed`: principle "records whether the tier-1 warm-started engine's bias
  install fired or was suppressed by the energy-improvement gate, so a reviewer can distinguish a
  genuinely degenerate tier-1 goal from one this fix correctly still trusts."

#### SCENARIO-ARC-FCP-5699-39-CHEAP-REUSE-OF-EXISTING-SEARCH-EVIDENCE-NO-SECOND-BOUNDED-SEARCH

Given `plan_in_model`'s own diagnostics already report whether a real bounded search's goal energy
ever improved
When the tier-1 warm-started-engine call site's goal-bias install is moved AFTER planning and gated
on that existing evidence instead of a new, separately-bounded satisfiability probe
Then a degenerate goal predicate (energy flat across a genuine 20,009-node search) is correctly
never installed, missing/unavailable diagnostics default to the old always-install behavior, and the
fix ships as an unconditional default because it reuses real planning evidence rather than
introducing the plain-path fix's own demonstrated false-negative mechanism

### REQ-ARC-WMTE-5596: Generator-Size A/B -- Qwen3.6-27B-MTP vs the Frozen Live Generator

`ops/known-issues.md` task 13 (2026-07-12, HIGH PRIORITY) queued a re-verification of the Kaggle
VRAM budget and an offline A/B of a larger generator against the frozen live Qwen3.5-9B-MTP,
after the operator asked: "are we still using qwen-3.5-9B when the leaders are using the larger
and newer qwen-3.6-27B and Gemma-4-31b models?"

**13(a) -- Kaggle backend hardware re-verification.** A fresh, first-party check (fetched
2026-07-13, not the stale May-2026 staff-post evidence the original investigation relied on)
found `docs.arcprize.org/arc-prize-2026`'s starter kit explicitly names an `rtx6000` accelerator
option mapped to `Nvidia RTX 6000 (g4-standard-48)`, labelled "Heavy ML; ARC-AGI-3 exclusive."
This session's own clone of the ARC-AGI-3 Milestone-1 winners' code
(`external/arc-m1-3rd-forge/kernel-metadata.json`, 3rd place, LB 0.86) confirms this is not
theoretical: forge's real, scored submission requests `"machine_shape":
"NvidiaRtxPro6000"` directly and ran `gemma-4-31b-it`. This corroborates (does not replace) the
project's existing accelerator investigation
(`docs/research-notes/arc-kaggle-accelerator-upgrade-2026-06-21.md`, updated 2026-07-13 with
this finding). Also clarified: `results/kaggle_env_probe.json`'s P100 finding is from an
unrelated auxiliary dev/build-verify kernel with no `machine_shape` field (a known, previously
flagged gap, not evidence the SCORED submission kernel's own `NvidiaL4` request -- a deliberate
2026-06-21 quota-cost tradeoff, unchanged here -- is broken).

**13(c) -- MTP compatibility at the larger size (checked, not assumed).** Direct GGUF metadata
inspection (`gguf_dump.py --no-tensors`) found the FIRST candidate considered,
`unsloth/gemma-4-31B-it-GGUF`, declares NO `nextn_predict_layers` key at all -- no native MTP
support. The operator surfaced two better-fitting official candidates mid-task:
`unsloth/Qwen3.6-27B-MTP-GGUF` and `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, both genuinely MTP-capable
(`qwen35.nextn_predict_layers = 1`, confirmed by the same metadata inspection) at the exact
"Qwen3.6-27B-class" size the operator's original question named. Downloaded and verified the
dense 27B variant (16.3GB Q4_K_M, full download, `unsloth/Qwen3.6-27B-MTP-GGUF`).

**A second, deeper compatibility gap found via a real launch attempt.** Even with genuine
architectural MTP support, a manual CUDA launch with `--spec-type draft-mtp --model-draft <same
path>` OOM'd on a single 24GB RTX 3090 (verbatim from the real launch log: `ggml_backend_cuda_
buffer_type_alloc_buffer: allocating 15621.78 MiB on device 0: cudaMalloc failed: out of
memory`): llama.cpp's SELF-draft MTP loads the SAME GGUF file TWICE (main + draft model), so the
real VRAM footprint is roughly 2x the file size (~32.6GB for this 16.3GB file) plus KV
cache/CUDA overhead -- exceeding a single 3090's capacity entirely.
This is exactly the kind of gap task 13(c) asked to rule out before assuming a speedup carries
over, and it generalizes past "does the metadata say MTP" to "does self-draft actually fit."
Fixed by adding `_candidate_mtp_self_draft_fits_vram()` (a real free-VRAM-vs-2x-filesize
feasibility check, +2GB margin) alongside the existing metadata check
(`_candidate_declares_mtp_metadata()`); the experiment's actual runtime `mtp` flag
(`candidate_mtp_used`) is the AND of both, so the candidate arm cleanly falls back to non-MTP on
this hardware instead of crash-looping (the first real run, before this fix, silently fell back
to a much slower AMD iGPU each time the OOM'd process died -- see the second bug below).

**A separate, structural GPU-pinning bug found and fixed.** The experiment's first working draft
never called `LocalGGUFProposer.stop()` between arms. Because both arms are pinned to the SAME
GPU (GPU 1, the outer loop's own allocated 3090), the "current" arm's still-resident ~11GB server
left only ~13GB free on the 24GB card -- right at
`_generator_server_and_env()`'s `_GENERATOR_CUDA_MIN_FREE_MB=13000` guard floor -- so the
candidate's launch attempt tripped the guard and silently fell back to the slow iGPU (`build-hip`
binary), defeating the "both arms on the same hardware tier" design goal with no error or
warning. Fixed by wrapping each arm's body in `try/finally: proposer.stop();
_wait_for_port_down(proposer.port)`, a new helper that polls the stopped server's `/health`
endpoint (bounded, 30s timeout) so the NEXT arm's free-VRAM guard sees genuinely reclaimed
memory, not a still-warm process.

**RESOLUTION (2026-07-13).** With both bugs fixed, the real 4-attempt measurement (m0r0 + sk48,
both arms, GPU 1, both arms confirmed via wall-clock plausibility to have run on the CUDA build
not the iGPU) produced a real, informative result: both arms 2/2 induction success;
`heldout_accuracy` -- m0r0 current=0.0 vs candidate=0.5; sk48 current=0.2 vs candidate=1.0. The
candidate (Qwen3.6-27B, running WITHOUT MTP per the VRAM-infeasibility finding) shows materially
HIGHER induction quality on both tested games. `honest_verdict:
"complete: generator_size_ab_equal_success_candidate_higher_accuracy"`.

**What this does and does not show.** This is a real, positive signal for the candidate's
induction QUALITY specifically, on a 2-game roster (well below the CLAUDE.md sample-size floor
for any percentage-point claim), measuring `WorldModelVerifier.heldout_accuracy` (not a full
actions-to-first-win live-solve sweep). It does NOT show the candidate would be faster or more
efficient on the actual Kaggle hardware: MTP is confirmed infeasible for this model on a single
24GB RTX 3090 (this dev box), but the ACTUAL scored hardware may be the 96GB RTX Pro 6000 per
13(a)'s finding, where self-draft MTP (2x ~16.3GB = ~32.6GB) would fit comfortably alongside KV
cache -- a materially different VRAM regime this experiment did not test. Per the task's own
guardrail and CLAUDE.md's frozen-live-stack discipline (mirroring REQ-ARC-WMTE-5594's task-7
precedent), this result does NOT change the frozen live generator -- it is reported as an offline
dev measurement requiring an explicit operator decision, and the honest recommendation is: this
result is promising enough to justify a deeper look (a larger roster, the 35B MoE variant, and/or
a real VRAM-matched test if 96GB hardware becomes available), not an automatic swap.

Required field principles:

- `candidate_mtp_self_draft_fits_vram`: principle "self-draft MTP loads the SAME GGUF file twice
  (main + draft); found mid-task that this can OOM even when the metadata declares support, so
  this is checked separately before attempting a real launch, not assumed from metadata alone."
- `candidate_mtp_used`: principle "the actual runtime decision (metadata support AND VRAM
  feasibility); this is what determines whether any wall-clock delta between arms is confounded
  by an MTP-vs-no-MTP asymmetry, not the metadata declaration alone."

#### SCENARIO-ARC-WMTE-5596-MTP-SUPPORT-VERIFIED

Given a candidate GGUF whose metadata declares an MTP self-draft head
When `_candidate_mtp_self_draft_fits_vram` estimates the self-draft footprint (2x file size plus
a margin) against real free VRAM on the target GPU
Then a candidate whose self-draft footprint exceeds available VRAM is correctly identified as
MTP-infeasible on this hardware, and the experiment runs that arm without MTP instead of
attempting a doomed launch

#### SCENARIO-ARC-WMTE-5596-INDUCTION-QUALITY-DELTA

Given real induction attempts from both the current frozen generator and the candidate generator,
run sequentially on the SAME GPU with each arm's server fully stopped before the next starts
When `build_artifact` compares `heldout_accuracy` across arms
Then the comparison reflects a genuine same-hardware-tier measurement (not confounded by one arm
silently falling back to a slower device), and the verdict honestly reports which arm has higher
accuracy without flipping the frozen live-submission generator

### REQ-ARC-WMTE-5597: Generator-Size A/B -- Qwen3.6-35B-A3B-MTP (MoE) vs the Frozen Live Generator

Follow-on to REQ-ARC-WMTE-5596. exp5596's dense `Qwen3.6-27B-MTP` candidate showed materially
higher `heldout_accuracy` than the current generator on both tested games, and its own spec entry
flagged the 35B MoE variant (`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) as "a natural follow-on if this
result is promising." This experiment is that follow-on, reusing exp5596's exact two-step MTP
feasibility check (`_candidate_declares_mtp_metadata` + `_candidate_mtp_self_draft_fits_vram`,
duplicated per-experiment per this session's established one-file-per-experiment convention) and
its GPU-pinning/stop-and-wait fix unmodified.

**MoE VRAM confirmation.** The candidate's Q4_K_M quant is 21.6GB (larger than the dense 27B
candidate's 16.3GB, despite the 35B MoE architecture activating only ~3B params per token --
self-draft MTP stores the FULL weight file regardless of expert-routing sparsity, so MoE does not
relax the self-draft VRAM problem). GGUF metadata confirms genuine MTP support
(`qwen35moe.nextn_predict_layers = 1`), but the self-draft feasibility check correctly found it
infeasible on a single 24GB RTX 3090 (self-draft estimate 43227MB vs 24120MB free), so the
candidate ran without MTP -- as expected, the same outcome as exp5596. A direct manual single-load
sanity check (no MTP) confirmed the model DOES fit non-MTP, but only barely: 21.9GB used of 24GB,
leaving 2.2GB free -- confirmed BEFORE the real run to avoid another crash-loop.

**RESOLUTION (2026-07-13).** The real 4-attempt measurement (m0r0 + sk48, both arms, GPU 1, both
arms confirmed via the same stop-and-wait GPU-pinning fix) produced a DIFFERENT result from
exp5596's dense-27B finding: both arms 2/2 induction success; `heldout_accuracy` -- m0r0
current=0.5 vs candidate=0.3 (current wins); sk48 current=1.0 vs candidate=1.0 (tie). Mean:
current=0.75, candidate=0.65. `honest_verdict:
"complete: generator_size_ab_equal_success_current_higher_accuracy"` -- the MoE candidate
performed WORSE than the current frozen 9B generator on this roster, in direct contrast to the
dense 27B candidate's positive result on the SAME two games. `induce_duration_s` for the
candidate was also notably higher (67.3s and 14.4s vs the current generator's 1.5s and 0.7s) --
the larger MoE model is substantially slower even without the MTP self-draft doubling, since its
weights are ~4x the current generator's file size.

**Note on cross-run comparability.** exp5596's OWN `current` arm scores on this same roster
(m0r0=0.0, sk48=0.2) differ from THIS run's `current` arm scores (m0r0=0.5, sk48=1.0) despite
identical model, game, and budget configuration -- real sampling variance in LLM-driven
exploration + induction (temperature > 0, no fixed decoding seed control at this layer), not a
bug. This means neither run's `current`-arm baseline should be treated as a fixed reference point;
the WITHIN-RUN candidate-vs-current comparison is the only sound reading, and even that is a
single-draw comparison on a 2-game roster -- well below the CLAUDE.md sample-size floor for any
percentage-point claim. Combined with exp5596's contradictory-direction result, the honest
reading across both experiments is: **neither the dense-27B nor the MoE-35B candidate has
demonstrated a reliable induction-quality edge over the current generator on this hardware and
roster** -- exp5596's positive signal and exp5597's negative signal could both be sampling noise
at n=2 games per arm. A larger roster (5+ games) and/or multiple seeds per (game, arm) pair would
be needed before either direction is trustworthy enough to inform an operator decision.

Required field principles: identical to REQ-ARC-WMTE-5596 (same field set, same rationale).

#### SCENARIO-ARC-WMTE-5597-MOE-MTP-FEASIBILITY-CHECKED

Given a MoE candidate GGUF whose metadata declares an MTP self-draft head and whose file is
LARGER than a previously-tested dense candidate's file
When `_candidate_mtp_self_draft_fits_vram` estimates the self-draft footprint
Then MoE sparsity does NOT relax the self-draft VRAM requirement (the full file is still loaded
twice), and a candidate whose self-draft footprint exceeds available VRAM is correctly identified
as MTP-infeasible regardless of active-parameter count

#### SCENARIO-ARC-WMTE-5597-INDUCTION-QUALITY-DELTA

Given real induction attempts from both the current frozen generator and the MoE candidate
generator, run sequentially on the SAME GPU with each arm's server fully stopped before the next
starts
When `build_artifact` compares `heldout_accuracy` across arms
Then the comparison honestly reports whichever arm has higher accuracy -- including a result
where the LARGER candidate performs WORSE than the current generator, without any pressure to
report a positive-sounding outcome

### REQ-ARC-WMTE-5598: Properly-Powered Multiseed Generator-Size A/B -- Resolving the exp5596/5597 Contradiction

exp5596 (dense Qwen3.6-27B-MTP, 2 games, n=1 draw/arm) found the candidate BEAT the current
generator. exp5597 (MoE Qwen3.6-35B-A3B-MTP, same 2 games, n=1) found the candidate LOST. Both
spec entries flagged this as likely sampling noise at n=1 -- this experiment resolves that by
testing all THREE arms (current, both candidates) together on a widened 4-game roster (m0r0,
sk48, cd82, sp80) with 3 independent repeated draws per (arm, game) cell (n=12 draws/arm total),
batching the loop by arm (one server per arm, reused across all its draws, stopped once before
the next arm starts) to avoid the per-attempt server-restart overhead exp5596/5597's interleaved
loop would have incurred at this scale.

**A genuine hardware fault interrupted the first attempt.** Mid-run, during the candidate_35b_moe
arm, GPU 1 (the outer loop's dedicated RTX 3090, hosted in an external eGPU enclosure) dropped
off the PCI bus entirely -- `nvidia-smi -q -i 1` returned "No devices were found," and even the
purpose-built `nvidia-smi --gpu-reset -i 1` recovery tool could not reach it. This is a hardware/
driver-level fault, not a script bug: the existing `_generator_server_and_env()` GPU-pinning
guard (by design, per its own docstring, "never fight the conductor for the 3090s") silently fell
back to the slow AMD iGPU when the free-VRAM check on GPU 1 started returning -1 (unreadable),
which would have contaminated the arm's later draws with a different, inconsistent hardware tier
partway through -- exactly the kind of confound this experiment was designed to eliminate. The
operator physically power-cycled the machine to recover the eGPU; both RTX 3090s came back
healthy afterward (confirmed via `nvidia-smi`).

**Hardening added before the retry.** (1) A per-arm `n_ctx` override (`candidate_35b_moe`
reduced from 16384 to 10240) intended to relax the arm's tight VRAM margin -- a direct manual
comparison found this saved only ~80MB (21.9GB used either way; the footprint is dominated by
model WEIGHTS, not KV cache, so this change alone would not have prevented the fault, and is kept
as a modest, honest, non-load-bearing precaution rather than oversold as a fix). (2) A genuine
fix: `_gpu1_free_mb() < 0` is now checked BEFORE every draw (not just once per arm); if GPU 1
becomes unreachable mid-run, the experiment immediately stops collecting further draws and
returns a distinct `generator_size_multiseed_ab_blocked_gpu1_lost_mid_run_partial_ranked_*`
verdict rather than silently continuing on degraded hardware -- fail closed, not silently degrade.
This guard was built defensively; it did NOT trigger on the successful retry (confirmed: no
`gpu1_unreachable_mid_run_aborting_remaining_draws` row in the checked-in artifact).

**RESOLUTION (2026-07-13).** The retried run completed cleanly on GPU 1 for all three arms
(3061.6s total, 35 of 36 draws succeeded -- one candidate_35b_moe draw on sp80 hit a normal,
honest induction failure: "syntax error line 159: expected an indented block," a real generated-
code defect after 3 retries, unrelated to the earlier hardware fault). Mean `heldout_accuracy`:
current=0.100 (std 0.289), candidate_27b=0.525 (std 0.352), candidate_35b_moe=0.391 (std 0.437).
Paired win/loss/tie against current (per (game, repeat) cell): candidate_27b **10 wins / 0
losses / 2 ties**; candidate_35b_moe **5 wins / 1 loss / 5 ties**. `honest_verdict:
"complete: generator_size_multiseed_ab_ranked_candidate_27b_gt_candidate_35b_moe_gt_current"`.

**This resolves the exp5596-vs-exp5597 contradiction: at real statistical power, BOTH candidates
beat the current generator, with the dense 27B model the clearer, more decisive winner.**
candidate_27b's 10-0-2 paired record is a near-unanimous signal (a naive sign test on the 10
decisive draws: P(10/10 or better under a fair-coin null) = 0.5^10 ~ 0.001) -- exp5596's original
positive finding replicates and strengthens with more data. candidate_35b_moe's 5-1-5 record is
net positive but much weaker and noisier (sign test on 6 decisive draws: P(>=5/6) ~ 0.11, not
strong evidence on its own) -- consistent with exp5597's negative n=1 draw having been an
unlucky single sample from a real but modest, noisy positive distribution, not a genuine
candidate-loses-to-current effect. Both candidates ran WITHOUT MTP (self-draft infeasible on a
single 24GB card for both, reusing exp5596/5597's feasibility check unmodified) -- this remains
an open axis: performance AND speed on the actual Kaggle hardware (a 96GB RTX Pro 6000, per
REQ-ARC-WMTE-5596's task-13(a) finding) where self-draft would fit is untested.

**What this does and does not show.** Still a 4-game, offline dev-quality-only measurement (not a
full actions-to-first-win live-solve sweep, and n=12/arm is still below the CLAUDE.md N>=30 floor
for a firm percentage-point claim on the absolute accuracy VALUES) -- but the PAIRED, WITHIN-RUN,
same-hardware-tier, multi-draw comparison is now genuinely more trustworthy than either single-
draw prior result, and candidate_27b's near-unanimous win record is a real, replicated, promising
signal specifically for the dense 27B candidate. Per the task's own guardrail and CLAUDE.md's
frozen-live-stack discipline, this result does NOT change the frozen live generator -- it is
reported as an offline dev measurement requiring an explicit operator decision. The honest
recommendation: candidate_27b's induction-quality edge is now well-supported enough to justify a
genuine cost/benefit evaluation for switching (weighed against Kaggle quota, actual scored-
hardware VRAM/MTP behavior, and a fuller live-solve test) -- it is no longer just a single-draw
curiosity.

Required field principles: identical to REQ-ARC-WMTE-5596/5597's `candidate_declares_mtp_
metadata`/`candidate_mtp_self_draft_fits_vram`/`candidate_mtp_used` (per-arm here), plus:

- `per_draw_results`: principle "every individual (arm, game, repeat) draw is recorded, not just
  aggregates -- exp5596/5597's contradiction came from single draws, so preserving the full draw
  list is what lets a reader assess variance directly rather than trusting a summary alone."
- `paired_vs_current`: principle "per-(game, repeat) win/loss/tie counts against the current arm,
  the higher-power paired comparison (controls for per-game/per-draw difficulty) vs comparing
  unpaired means across arms."

#### SCENARIO-ARC-WMTE-5598-MULTISEED-PAIRED-COMPARISON

Given multiple independent draws per (arm, game) cell across a widened roster
When `build_artifact` computes `per_arm_summary` and `paired_vs_current`
Then the paired win/loss/tie counts against the current arm are reported per candidate, giving a
higher-power signal than any single-draw comparison, without asserting formal statistical
significance beyond what the sample size supports

#### SCENARIO-ARC-WMTE-5598-ARM-BATCHED-SERVER-LIFECYCLE

Given three arms sharing one GPU, each needing multiple draws across multiple games
When `build_artifact` runs the game/repeat loops
Then exactly one proposer/server is constructed per arm (reused across all its draws) and
explicitly stopped before the next arm starts, and a mid-run GPU-health check aborts the run
honestly (rather than silently falling back to different hardware) if the GPU becomes unreachable
between draws

### REQ-ARC-WMTE-5599: Real Reinduction-Path A/B -- candidate_27b Reverses on the Actual Scored Code Path

Follow-on to REQ-ARC-WMTE-5598, prompted by the operator's cost/benefit question. Investigating
`E3AgentPolicy._induce_and_plan()` (the method the SCORED live agent actually calls) found its
LLM tier (`execute_bounded_llm_reinduction`, `arc_llm_reinduction.py`) is ONLY invoked when the
induction reason is `"level_up_reinduction"` -- a genuine level-up just happened. For the
initial `"stall"` reason (exploration exhausted without ever winning), the agent tries a
zero-LLM TTT-prior model, then classical DSL/active-probe tiers -- the LLM is never invoked.
Confirmed empirically: a direct `lb.run_game` measurement on `m0r0` (never-leveled, like every
roster game in exp5596/5597/5598) completed `_induce_and_plan()` in 17.6s -- far too fast for
real LLM inference -- and `level_induction_events` stayed empty. **This means exp5596/5597/5598's
induction-quality measurements, while real and valid as a measure of induction quality via the
`LocalGGUFProposer.induce()` wrapper, could not have exercised the real live-agent reinduction
code path on their own roster** -- that roster structurally cannot trigger it.

This experiment instead calls `execute_bounded_llm_reinduction` directly -- the exact function
the scored agent invokes -- on real, reproducible post-level-up transitions from `lp85` (the one
game with a session-confirmed level-up, `first_levelup_index` around 6). A widened
`n_ctx=22000` (up from the class default 16384) was required and verified first via a manual
pre-check: lp85's 64x64 grid previously overflowed the induction prompt at 16384 (exp5593's real
HTTP 400 `exceed_context_size_error`); 22000 resolved it (a real 101.6s call completed cleanly).
3 independent repeats per arm (lp85's exploration is stochastic; fresh transitions collected
every draw), current vs `candidate_27b` (exp5598's clear induction-quality winner), same
GPU-1/stop-between-arms/GPU-health-guard discipline as exp5598.

**DISCLOSED METHODOLOGY GAP (found reviewing the real result, not hidden).** `_induce_and_plan()`
calls `execute_bounded_llm_reinduction` with `min_heldout_accuracy=1.0` plus several other
policy-configured kwargs (`enable_subgoal_search`, `subgoal_budget`, `value_head`,
`enable_factored_planner`, `factored_trust_threshold`, `structural_goal_provider`). This
experiment's call used the function's own bare defaults for all of these (`min_heldout_accuracy
=0.0` etc.) rather than replicating the exact policy-configured values -- a real fidelity gap
against the true scored call site, not a bit-identical replication. This is flagged explicitly
because `current`'s one "planned" draw (repeat 2, below) had `heldout_accuracy=0.0`, which would
almost certainly have been REJECTED under the real agent's strict 1.0 threshold -- so even
`current`'s single apparent success may not survive under the real, stricter gating. This
experiment answers "does the generator work through the reinduction machinery under standard/
default gating," a real and useful question, but not a bit-identical replication of the scored
call site; a follow-on with exact parameter matching is the natural next step if this result is
acted on.

**RESOLUTION (2026-07-13) -- REVERSES exp5598's induction-quality-only finding.** All 6 draws
reached the real level-up transition set (37 total transitions, level-up at action 7, 8-transition
induce window, consistent across every draw). `current`: plan_rate_given_levelup = **1/3 (33%)**,
`reinduce_duration_s` = 67.0s / 78.1s / 20.7s (mean ~55s), mean `heldout_accuracy` = 0.0.
`candidate_27b`: plan_rate_given_levelup = **0/3 (0%)** -- WORSE, not better -- and
`reinduce_duration_s` = 294.9s / 476.4s / 431.2s (mean ~401s, i.e. **~7x slower** than current),
mean `heldout_accuracy` = 0.222 (nominally higher than current's 0.0, but never actually
producing a usable plan in any of the 3 draws). `honest_verdict:
"complete: reinduction_ab_current_plans_more_reliably"`.

**What this changes for the cost/benefit recommendation.** exp5598's induction-quality signal
(candidate_27b wins 10-0-2 on a held-out-accuracy proxy, n=12/arm) was real and well-powered, but
this experiment shows that signal does NOT carry over to the real live-agent reinduction code
path: candidate_27b never once produced a usable plan here, while current did once (with the
above-noted caveat about whether that one success would survive stricter gating), AND
candidate_27b is dramatically slower per attempt (~7x). **This reverses the tentative
recommendation from the cost/benefit discussion following exp5598** -- candidate_27b's induction-
quality edge does not translate into better live-agent planning reliability on this test, and its
severe speed cost (already flagged as a theoretical risk in that discussion) is now empirically
confirmed as large. The honest conclusion: **do not switch the frozen live-submission generator
based on the evidence gathered so far** -- exp5598's quality-only signal was measuring the wrong
thing for this decision, and the real-reinduction-path test (despite its own disclosed parameter-
fidelity gap) points the opposite direction, on top of a severe throughput cost.

**Scope limits, honestly stated.** n=3 repeats per arm on a SINGLE game (lp85) is a small sample;
`plan_rate 1/3 vs 0/3` is not independently statistically decisive on its own (a single flip on
either side would change the ranking). The speed finding (~7x slower) is the more robust,
sample-size-independent part of this result. The disclosed parameter-fidelity gap (bare function
defaults vs the real policy's stricter configured values) means this is not the final word on
the planning-reliability question either. Taken together with exp5598's contradicted signal, the
overall picture is: no candidate has yet shown a reliable, real advantage on the metric that
matters (live planning success), and the cost (speed) is real and severe.

Required field principles: identical structure to REQ-ARC-WMTE-5596/5597/5598.

#### SCENARIO-ARC-WMTE-5599-REAL-REINDUCTION-PATH

Given real post-level-up transitions and a candidate generator with a strong induction-quality
signal from a prior, narrower experiment
When `execute_bounded_llm_reinduction` (the actual function the scored live agent calls) is
invoked directly with that candidate as the proposer
Then the plan-success rate and wall-clock cost are measured on the real code path, which may
disagree with the narrower induction-quality proxy -- and the experiment reports whichever
direction the real data shows, without assuming the narrower result carries over

#### SCENARIO-ARC-WMTE-5599-CONTEXT-BUDGET-FIX-VERIFIED

Given a large-grid game whose induction prompt previously overflowed the default context window
When the reinduction call is made with a widened `n_ctx`
Then the call completes without a context-size error, isolating any subsequent
planning-quality result from the context-budget confound

### REQ-ARC-WMTE-5599-2: Apples-to-Apples Precision Isolation -- the Model Forge Actually Used, at the Highest Precision Tractable on This Hardware

REQ-ARC-WMTE-5599 found `Qwen3.6-27B-MTP` (Q4_K_M) planned LESS reliably (0/3) than the current
9B (1/3) and took ~7x longer on the real reinduction path -- but that comparison left open
whether 4-bit quantization itself was the cause, since forge (3rd-place ARC-AGI-3) ran a
comparably-sized model (Gemma-4-31B-it) at FULL precision via vLLM on 96GB of VRAM. The operator
directed a genuine apples-to-apples attempt on this project's own hardware: "we have plenty of
VRAM on our AMD iGPU if we want to try the full model weights instead of 4bit quants and/or full
kv-cache key size."

`python/carnot/experiment_5705_full_precision_27b_vs_4bit_quant_ab.py` SHALL isolate precision
(and, of necessity, hardware and serving stack) as cleanly as this hardware allows, reusing
REQ-ARC-WMTE-5599's exact per-draw methodology (real post-level-up `lp85` transitions,
`execute_bounded_llm_reinduction`, the same widened `n_ctx`).

**RESOLUTION (2026-07-14) -- three pivots, all disclosed, ending in a real measured result.**

1. **vLLM ruled out.** The PyPI `vllm` wheel is CUDA-only; vLLM's ROCm support has no PyPI
   distribution and has historically targeted MI-series datacenter cards, not this consumer
   gfx1150 iGPU. A from-source ROCm build for an unsupported architecture was judged too large
   an undertaking for this task.
2. **First attempt (`unsloth/Qwen3.6-27B`, full BF16, precision-only isolation against the SAME
   model REQ-ARC-WMTE-5599 tested) -- ABANDONED after three real, reproducible load failures**
   on the project's HIP-built llama.cpp binary: the default `-fit` auto-memory-fit heuristic
   hard-hung (zero `/proc/PID/io` read progress for 12+ minutes); `-fit off` hard-stalled again
   at a later step (zero RSS/IO progress for 5+ minutes, one thread pinned near 100% CPU);
   `-fit off --parallel 1` crawled at ~11MB/s (over an hour to finish loading). No backtrace
   tooling is available on this box (`ptrace` restricted, no `perf`, no root) to diagnose the
   exact stuck call; the pattern is consistent with this specific build mishandling Qwen3.6's
   unusual hybrid `Qwen3_5ForConditionalGeneration` architecture (mixed
   "linear_attention"/"full_attention" layers).
3. **Operator-directed pivot to `google/gemma-4-31B-it`** (the model forge ACTUALLY used, and a
   conventional sliding-window + full-attention architecture, the same mature pattern Gemma 2/3
   already used) -- **this ALSO failed at full BF16 precision**: the initial page-cache-warm
   bulk read completed in 20s (RSS 0->36.1GB), but the final loading stage then crawled to a
   near-stall (36.1->37.6->38.2GB over ~9 minutes, a rate that would have taken hours more) --
   the SAME failure class as Qwen3.6, on a structurally different, more conventional
   architecture. This points to a broader HIP/ROCm large-BF16-model-loading issue on this
   specific llama.cpp build, not something specific to either model's attention mechanism.
4. **Operator-directed second pivot: Q8_0 quantization, not full BF16.** Q8_0 (8-bit,
   near-lossless -- NOT literally full/lossless precision) is a well-tested quantization level
   this HIP build handles routinely elsewhere in this project. `google/gemma-4-31B-it` was
   converted via `convert_hf_to_gguf.py --outtype q8_0` (32.6GB) and loaded CLEANLY -- healthy
   within ~20s, confirmed via a real `/completion` call ("The capital of France is Paris.") --
   a sharp, informative contrast with both BF16 failures. Full (non-quantized) F16 KV cache was
   used throughout (`kv_quant=None`), addressing the "full kv-cache key size" half of the
   operator's request even though the WEIGHTS ended up at Q8_0, not full precision.

**The real measurement (n=1, reduced from the planned n=3 -- disclosed, not hidden).** The
first repeat of a 3-repeat run took ~40 minutes (real, slow token generation at ~2.4 tok/s on
this iGPU, ~5x slower than the frozen 9B's ~13 tok/s) and the script has no incremental
checkpointing (writes its artifact only once, at the end) -- continuing to a full n=3 risked
losing ALL data, including the already-completed first repeat, to a timeout kill. The run was
stopped and re-launched at `n_repeats=1` with a matched timeout, mirroring REQ-ARC-WMTE-5593-5's
same n-reduction pattern for the same reason (real GPU-bound cost, not a shortcut).

The single real draw reached a real level-up (`actions_to_levelup=7`, 8 induction transitions),
then genuinely FAILED to induce: `skipped="proposer_failed"` after `reinduce_duration_s=2408.163`
(~40 minutes -- consistent with 3 near-full-budget generation retries, `generate()`'s default
`tries=3`, at this model's slow token rate) -- never even reaching the held-out-accuracy scoring
stage (`heldout_accuracy=null`). This is a genuine induction failure at the code-generation
step, not a quality-threshold rejection. `plan_rate_given_levelup=0.0` (0/1).

**Honest conclusion.** Comparing against the current-9B historical baseline (the CLEAN
comparison this experiment supports, same task/methodology):
`gemma_q8_0_plans_less_reliably_than_current_9b` -- the current 9B's historical plan rate
(1/3 = 0.333) beat the near-lossless Q8_0 Gemma-4-31B-it (0/1) on this single real draw, and
took roughly 44x longer per attempt.
Comparing (as disclosed CONTEXT ONLY, not a controlled isolation -- model family AND
quantization level both differ) against REQ-ARC-WMTE-5599's Q4 Qwen candidate:
`gemma_q8_0_ties_qwen_q4_context_only_different_model_and_precision` -- both larger candidates
scored 0/N on this specific reinduction task, a real convergence across two different model
families and two different quantization levels. **This is now the THIRD independent measurement
(Q4 Qwen, Q8_0 Gemma, both vs the current 9B baseline) pointing the same direction: on Carnot's
own real induction PROMPT and task, larger 27-31B-class models have not yet demonstrated an
advantage over the frozen 9B, regardless of quantization level or model family.** The n=1
sample for this specific run is a real, disclosed limitation -- not conclusive on its own -- but
it is consistent with, not contradictory to, the prior Q4 finding, strengthening rather than
reversing REQ-ARC-WMTE-5599's original cost/benefit conclusion (do NOT switch the frozen
live-submission generator).

**Follow-up (2026-07-14, same day) -- timeout-margin investigation, v2.** The operator pushed
back on the v1 conclusion with a specific, well-founded question: "did we give it enough
kv-cache for context and wait long enough?" This was investigated directly rather than
defended against:

- **Context was ruled out cleanly.** `REINDUCTION_N_CTX=22000` vs the real Gemma-tokenizer-
  measured induce prompt of 11207 tokens (via `llama_cpp.Llama(vocab_only=True).tokenize()`,
  not a char-based estimate) -- comfortably sufficient, not the bottleneck.
- **Timeout margin was genuinely tight.** A real `/completion` call with `n_predict=1` isolated
  prefill cost: 203.16s for the 11207-token prompt (55.19 tok/s -- fast; an earlier 6-token
  smoke test's 6.669 tok/s figure was a fixed-overhead artifact, not representative). Combined
  with the ~2.4 tok/s decode rate observed in v1, a full `max_tokens=2560` generation could take
  up to ~1067s, for a worst-case total of ~1270s -- which DOES exceed the v1 `timeout=1200`
  used both as the load-wait budget and the per-HTTP-call timeout in
  `LocalGGUFProposer.generate()`. The v1 `proposer_failed` result was plausibly a timeout
  artifact, not a genuine model-quality failure.
- **Retry with `timeout=3600` (3x the estimated worst case), wrapped in a 7200s (2-hour) outer
  budget: killed by the outer wrapper with ZERO output.** The result file after the retry is
  byte-identical to the pre-retry backup; the run's own log never advanced past initial game/
  scorecard setup (7 lines, nothing appended across the full 2-hour window). This was NOT a
  hang: `rocm-smi` showed the iGPU pinned at 100% utilization and the llama-server process held
  `R` (running) state with steadily climbing CPU-time and RSS across every check performed
  during the run. It was genuinely, continuously computing -- and still did not produce one
  complete reinduction attempt inside a 3x-generous per-call timeout AND a 2-hour outer budget.

**This closes the timeout-margin question, and strengthens rather than reverses the v1
conclusion.** The hypothesis that v1 failed merely because 1200s was a tight-but-plausible
margin is falsified: giving the same candidate 3x more time per call and 6x more wall-clock
budget overall still produced no result. The v1 artifact (`proposer_failed`,
`reinduce_duration_s=2408.163`) remains the checked-in measurement -- it was not overwritten,
since the retry produced nothing to overwrite it with. Per standing discipline, a third retry
was not launched without checking with the operator first. The decisive, hardware-grounded
finding is now: on this iGPU, at Q8_0 precision, `google/gemma-4-31B-it` cannot complete the
live reinduction task within a bounded, practically-usable window -- independent of whatever
timeout value is chosen, and independent of any hypothetical quality advantage the larger model
might have in principle.

Required field principles:

- `weight_precision`: principle "honestly discloses that the WEIGHTS served are Q8_0 (near-lossless 8-bit), NOT the originally-planned full/lossless BF16 -- full precision was tried and abandoned for two different models on this hardware; reporting this run as 'full precision' would misrepresent what was actually measured."
- `serving_hardware`: principle "honestly discloses the NECESSARY hardware confound: even at Q8_0 (~31GB), this model does not fit the single 24GB RTX 3090 REQ-ARC-WMTE-5599 used, so this arm runs on the AMD iGPU instead -- serving stack (llama.cpp) is held constant, hardware is not."
- `qwen_q4_context_comparison`: principle "explicitly labeled CONTEXT ONLY, never the primary verdict -- comparing Gemma-4-31B-it (Q8_0) against the Q4 Qwen candidate conflates model family and precision; reporting it as a clean isolation would be dishonest."

#### SCENARIO-ARC-WMTE-5599-2-PRECISION-ISOLATION-WITH-DISCLOSED-PIVOTS

Given full (lossless) precision proves impractical to load on this hardware for the originally-
planned candidate AND its operator-directed replacement
When the experiment pivots to the highest precision level that DOES load and run cleanly (Q8_0)
Then every pivot (model swap, precision fallback, sample-size reduction) is disclosed in the
artifact and spec with its concrete cause, and the resulting verdict is scoped to what was
ACTUALLY measured (Q8_0, not full precision; n=1, not n=3) rather than overclaiming the
originally-planned comparison

### REQ-ARC-WMTE-5599-3: Third-Party Ternary Quantization on a Real Discrete GPU (exp5709)

Same-day operator follow-up: "I would like to try
https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf on CUDA." Ternary Bonsai is a ~1.71
bits/weight ternary ({-1,0,+1}) quantization of Qwen3.6-27B (GGUF `Q2_0_g128` packing),
requiring a bespoke third-party fork (`github.com/PrismML-Eng/llama.cpp`, branch `prism`) --
standard llama.cpp cannot load its tensor type.

**Pre-integration audit (before building or running anything).** The fork was cloned and
inspected: normal llama.cpp fork layout, no curl-pipe-to-shell or remote-code-eval patterns,
217 stars, released same-day. One genuine concern was raised: grepping `ggml-cuda`/`ggml-hip`/
`ggml-metal` for the ternary type names (`Q2_0_g128`, `TQ1_0`, `TQ2_0`, `PQ2_0`) found no
dedicated CUDA kernel files, only generic CPU-side hits -- raising the possibility of a silent
CPU-fallback or a load failure on GPU. `src/models/dspark.cpp` (the file HF's own API
auto-parser mis-summarized as the whole model's "architecture," inflating confusion about a
3.65B-param mismatch against the "27B" branding) turned out to be the separate EAGLE-style
speculative-decoding drafter, not the ternary trunk -- confirming genuine, non-trivial custom
C++ work in the fork, not a thin wrapper.

**This audit-stage concern did NOT materialize empirically.** Built via
`cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` (this project's own RTX 3090
compute capability), the fork compiled cleanly (`ggml-cuda` built, `llama-server`/`llama-cli`
linked, ~8 minutes wall-clock on 24 cores). Loading `Ternary-Bonsai-27B-Q2_0.gguf` (7.17GB, the
file the model card's own quickstart names) on GPU 1 (`CUDA_VISIBLE_DEVICES=1` -- GPU 0 is
reserved for the conductor per the standing hardware-allocation rule) used 22.5GB of real GPU
memory and a real `/completion` smoke call ("The capital of France is" -> "Paris. Paris is the
largest city in France...") returned coherent, correct text at **67.5 tok/s decode** -- genuinely
fast, real GPU-accelerated ternary inference. The kernel implementation exists somewhere this
audit's naming-based grep missed; it is not a functional gap.

**The real measurement.** `python/carnot/experiment_5709_ternary_bonsai_cuda_reinduction_ab.py`
reused the exact same methodology as exp5599/exp5705 (real post-level-up `lp85` transitions,
`execute_bounded_llm_reinduction`). The single real draw reached a real level-up
(`actions_to_levelup=7`) and completed the full reinduction attempt in **212.948s (~3.5 min)** --
**11x faster than exp5705's Q8_0/iGPU run (2408.163s)** and closer to (though still ~4x slower
than) the frozen 9B's ~55s. This time the attempt got FURTHER than exp5705's: round 1
(`action=induce`) actually produced valid, parseable code (`proposer_ok=true`) but was rejected
as `degenerate_goal_predicate` -- the induced world model's goal condition did not meaningfully
discriminate win states, a semantic failure rather than a syntax failure. Round 2
(`action=refactor`) then failed to produce usable code after 3 retries
(`missing ('engine', 'is_level_complete') in output`), landing on the same terminal
`skipped=proposer_failed` / `heldout_accuracy=0.0` outcome as exp5705, just via a different,
more informative failure path.

**Honest verdict:** `complete: ternary_bonsai_plans_less_reliably_than_current_9b` -- 0/1 vs the
frozen 9B's historical 1/3. **This is now the FOURTH independent measurement (Q4 Qwen, Q8_0
Gemma, ternary-Q2_0 Bonsai, all vs the current 9B baseline) pointing the same direction**, across
three different quantization schemes, two different base model families, and now two
structurally different serving stacks (this project's own HIP/CUDA build vs a real third-party
CUDA fork) and two different hardware classes (iGPU vs discrete 3090). The one variable NOT yet
disproven as a confound is base-model-family-independent code-induction reliability itself --
every 27B-class candidate tested so far, regardless of precision or serving stack, has failed
this project's specific reinduction task while the frozen 9B has not. Frozen live-submission
generator remains UNCHANGED.

**Confound disclosure (not hidden).** Unlike exp5705 (same serving stack as every other
experiment, different hardware only), this run changes model family, quantization scheme,
serving stack (`serving_stack_provenance`), AND hardware simultaneously versus the 9B baseline --
several confounds move together, so this is informative but not a controlled isolation of any
single variable. It IS a controlled comparison against exp5705 on one dimension (both are 27-31B
candidates that failed the SAME task), strengthening the cross-candidate pattern even though it
does not isolate why.

Required field principles: see `FIELD_PRINCIPLES` in
`python/carnot/experiment_5709_ternary_bonsai_cuda_reinduction_ab.py` (`weight_precision`,
`serving_hardware`, `serving_stack_provenance` -- the third-party-fork disclosure -- and the two
historical-reference fields, each principle-annotated per CLAUDE.md's Principle-Annotated
Artifact Fields discipline).

**Follow-up (2026-07-14, same day) -- sample-size fairness, n=1 upgraded to n=3.** The operator
asked a sharp, correct methodological question: "If the final was 0/1 plan rate vs the 9B's 1/3,
does that mean that the 9B was allowed 3 plans? Should we allow this model the same?" The n=1
result above was real but statistically uninformative on its own: with a single draw, observing
0/1 is unsurprising (67% likely) even if Ternary Bonsai's TRUE underlying success rate matched
the 9B's exactly (1/3 ~= 0.33) -- a single data point cannot distinguish "worse than the 9B" from
"as good as the 9B, unlucky draw." The "0/1 vs 1/3" framing invited an implicit rate comparison
the sample size did not support.

Since Ternary Bonsai runs fast on the real 3090 (~213s/attempt, unlike exp5705's ~40min/attempt
on the iGPU, where a full n=3 genuinely risked losing all data to a timeout), re-running at
`n_repeats=3` to match exp5599's 9B baseline sample size exactly was cheap and the honest fix --
not a caveat, an actual re-measurement. **Real n=3 result:** all three independent draws reached
a real level-up (`actions_to_levelup=7` on every draw -- the exploration path is evidently stable
for this game/policy), and all three replicated the SAME failure shape seen in the n=1 run: round
1 (`induce`) produced valid, parseable code every time (`proposer_ok=true`), rejected as
`degenerate_goal_predicate` every time; round 2 (`refactor`) then failed after 3 retries every
time. `mean_reinduce_duration_s=168.805` (156.4s / 165.9s / 184.2s per draw -- consistent, not
one outlier). **`arm_summary.plan_rate_given_levelup = 0/3 = 0.0`.**

**This is now the real, apples-to-apples comparison the operator asked for: Ternary Bonsai 0/3
vs the frozen 9B's 1/3 -- same sample size, same task, same methodology.** The n=3 result also
sharpens the finding beyond what n=1 could show: the failure is not a one-off unlucky draw but a
REPRODUCIBLE failure mode -- the SAME induced world model bug (a goal predicate that does not
discriminate win states) recurs identically across three independent exploration-and-induction
attempts. That determinism is itself informative: it points at a systematic gap in this
candidate's ability to produce a semantically correct goal condition for this game, not
stochastic noise that a larger n might average away. Honest verdict unchanged in direction
(`ternary_bonsai_plans_less_reliably_than_current_9b`), now backed by a fair-sample-size
comparison rather than a single provisional draw. The n=1 result remains in this spec as
disclosed provisional context (per "never remove existing content"), superseded by this n=3
measurement as the citable comparison.

#### SCENARIO-ARC-WMTE-5599-3-THIRD-PARTY-TERNARY-ON-REAL-GPU

Given a third-party llama.cpp fork is required to load a novel ternary quantization format, and
an audit of that fork's CUDA kernel coverage cannot conclusively confirm GPU support from source
inspection alone
When the fork is built and the model is actually loaded and exercised on a real GPU
Then the empirical result (real GPU memory used, real tok/s, real task outcome) is trusted over
the audit's inconclusive static finding, and the serving-stack provenance (third-party, not this
project's own build) is disclosed as a confound in the artifact and spec rather than treated as
equivalent to every other experiment's serving stack

### REQ-ARC-WMTE-5599-4: One-Last-Time Qwen3.6-27B Q4 Check -- Already Measured, One Variable Left

Same-day operator follow-up: "We should try Qwen3.6-27B 4bit quant one last time with a Q8
kv-cache and see how well it does."

**Pre-check found the requested config already measured, cleanly, at n=3.** exp5599
(`results/experiment_5599_reinduction_ab_lp85_levelup.json`) already ran EXACTLY Q4_K_M
`Qwen3.6-27B-MTP-GGUF` with `kv_quant="q8_0"` on the real reinduction path, n=3, not a loading
failure or a degenerate n=1: `plan_rate_given_levelup=0/3`, `mean_reinduce_duration_s=401.0`
(294.9s/476.4s/431.2s), `heldout_accuracy` 0.333/0.0/0.333 (real signal, never crossing the 1.0
acceptance threshold). Re-running that identical configuration would be a doomed rerun per
CLAUDE.md's Failed-Experiment Rerun Discipline -- surfaced to the operator via
`AskUserQuestion` rather than silently re-run or silently refused; operator confirmed pivoting to
the one genuinely untested variable.

**The one untested variable: `mtp=False` was set for exp5599's `candidate_27b` arm with no
recorded rationale**, despite the model being named `Qwen3.6-27B-MTP-GGUF` (the 9B baseline arm
used `mtp=True`). `python/carnot/experiment_5713_qwen27b_q4_mtp_enabled_ab.py` isolates exactly
that one variable: same weights (Q4_K_M), same KV-cache precision (Q8_0), same hardware (GPU 1,
RTX 3090, this project's own CUDA build), same task/methodology -- ONLY `mtp` flips
`False -> True`. Launched at `n_repeats=3` from the start, applying the sample-size-fairness
lesson from the SAME day's exp5709 n=1->n=3 upgrade before it could recur.

**RESOLUTION: the first attempt found a real, hard OOM -- not a quality question.** The
background `n_repeats=3` run stalled: the driver dropped to near-zero CPU and the launched
`llama-server` subprocess was found `<defunct>` (crashed, zombie) within the first health-check
poll window. `LocalGGUFProposer._ensure_server()` redirects the subprocess's stdout/stderr to
`DEVNULL`, so the crash reason was invisible from the automated run alone, and the driver would
have polled a dead server for up to `load_wait_attempts` (600 x 2s = 20 minutes) per repeat --
60 minutes total -- for no new information. Killed and diagnosed directly instead: a manual
launch with visible output showed the target model (Q4_K_M, ~15.9GiB on disk) loaded fine, but
loading the DRAFT model -- self-speculative MTP loads the SAME GGUF file a SECOND time as a
separate CUDA buffer, even though target and draft are literally the same weights -- failed:
`cudaMalloc failed: out of memory` trying to allocate ~15.6GiB on top of the already-loaded
target. Total demand (~32.6GB) exceeds the single RTX 3090's 24GB outright. This almost
certainly is the root cause exp5599 hit too (undocumented at the time, hence `mtp=False` with no
recorded rationale) -- and explains why MTP works for the 9B arm (9B x 2 copies comfortably fits
24GB) but not for a 27B-class model on a single card.

**The precondition check now computes this directly** (`2 x on-disk file size vs free VRAM`,
not a magic number) so the experiment blocks in well under a second with the concrete numbers
(`mtp_dual_load_estimated_mb=32628.6` vs `gpu1_free_mb=24120.0`) instead of burning up to an hour
confirming a deterministic, instantly-reproducible failure a second and third time. Honest
verdict: `complete: blocked_gpu1_free_vram_sufficient_for_mtp_dual_load` -- a real, fast,
non-fabricated precondition block, with the manual diagnostic's crash log excerpt embedded
verbatim in the artifact (`manual_diagnostic_crash_confirmation`) as concrete evidence, not just
an arithmetic inference.

**Answering the operator's original question directly.** The Q4_K_M + Q8-KV-cache config for
Qwen3.6-27B has now been tried as thoroughly as this hardware allows: with MTP off (exp5599,
0/3, clean) and with MTP on (this experiment, structurally cannot run -- OOM). Neither path
beats the frozen 9B. The frozen live-submission generator remains UNCHANGED.

**Sibling fix, same incident: `scripts/adversarial_verify.py`'s `_is_precondition_check_only_blocked`
only recognized a BARE `blocked_` prefix**, missing the `complete: blocked_<resource>` form
CLAUDE.md's own Verdict Terminal-Prefix Discipline mandates every terminal verdict use (a clean
precondition-block IS a terminal state). This experiment's first (correctly-formed, terminal-
prefixed) blocked artifact was false-flagged `DURATION_TOO_SHORT` before the fix -- exp5705's
and exp5709's blocked branches use the identical `complete: blocked_{miss}` pattern and would
have hit the same false positive had their blocked paths ever become the checked-in artifact.
Fixed via `_strip_verdict_terminal_prefix` (mirrors `research_conductor.py:_verdict_is_untrustworthy`'s
terminal-prefix list), with a 5-test regression suite added to
`tests/python/test_adversarial_verify_blocked_verdict_duration_exemption.py`.

Required field principles: see `FIELD_PRINCIPLES` in
`python/carnot/experiment_5713_qwen27b_q4_mtp_enabled_ab.py`.

#### SCENARIO-ARC-WMTE-5599-4-MTP-DUAL-LOAD-OOM-NOT-QUALITY

Given a self-speculative MTP configuration requires loading the same GGUF file twice (target and
draft), and the requested weights/KV-cache combination was already cleanly measured with MTP
disabled
When the one remaining variable (MTP enabled) is isolated and attempted on a single 24GB GPU
Then a real, computed precondition check (2x on-disk file size vs free VRAM) blocks the attempt
honestly and fast, backed by a manual diagnostic's crash log as concrete evidence, rather than
either fabricating a performance result or burning up to an hour polling a subprocess that
already crashed

### REQ-ARC-FCP-5591-3: ColorBlobSaliencePrior Per-Frame Caching (Submission-Prep Pre-Flight Incident)

Discovered 2026-07-14 during ARC-AGI-3 submission-prep pre-flight (operator: "let's prepare for
a proper submission"). Running `scripts/kaggle/arc_local_submission_gate.py --check` against the
current live scored config (`SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED = True`, flipped on 7/7,
after the last staged submission on 6/30) produced a catastrophic result: 0/8 core solves vs the
verified baseline's 4/8, with 7 of 8 canonical games timing out at the gate's 115s cap.

**Root cause, found via `faulthandler` stack trace (not guesswork).** A direct manual
reproduction of `lp85` (a 64x64-grid game) with a 60-180s timeout hung with ZERO progress --
no per-step log line ever printed. `PYTHONFAULTHANDLER=1` + `SIGABRT` produced a live stack
trace pinpointing the hang inside
`ColorBlobSaliencePrior.score() -> connected_color_blobs()` (`arc_color_blob_salience.py`).
`action_tier_rows()` (the caller inside `_record_action_salience_diagnostics` ->
`_candidates()` -> `_ingest()` -> `next_move()`, i.e. the live per-step hot path) already
computed the frame's blob decomposition ONCE at its own top -- but then called
`self.score(frame, candidate)` once per candidate action (up to one per grid cell on a
click-heavy game), and `score()` INDEPENDENTLY recomputed the SAME full-grid flood-fill from
scratch on every call, ignoring the decomposition its own caller had just computed. Net cost:
O(candidates x grid_cells) per `next_move()` call -- on a 64x64 grid with thousands of click
candidates, this is tens of millions of redundant cell-visits per step, in pure Python
(list/tuple/dict + numpy-scalar-indexing, not vectorized) -- a de facto hang, not a true
infinite loop, but indistinguishable from one at any realistic time budget.

**Fix.** `score()` now accepts optional keyword-only `blobs`/`color_counts` cache arguments
(default `None`, preserving the exact two-positional-argument `score(frame, candidate)`
protocol shared by every other action-prior class in this codebase --
`arc_frame_change_predictor.py`, `arc_geometric_salience.py`, `arc_discriminative_router.py`,
`arc_perception_generation.py`, `arc_object_history_salience.py` all call `.score(frame,
candidate)` generically and are unaffected). `action_tier_rows()` now computes `color_counts`
once alongside its existing once-computed `blobs`, and passes both through to every
`self.score(...)` call within the same invocation, eliminating the redundant recomputation.

**Verified.** A direct `lp85` run (budget=500, `CARNOT_ARC_DISABLE_INDUCTION=1`) that previously
hung indefinitely (180s+, zero actions taken) now completes in 25-68s (496 actions, reaches
L1, `eff=2.0069` -- an exact match to `arc_local_submission_gate.py`'s own
`CANONICAL_LP85_PER_LEVEL_EFFICIENCY_FLOOR` constant, strong evidence the fix restores the
documented baseline behavior). Existing coverage
(`tests/python/test_arc_color_blob_salience_object_topology.py`, 5 tests) still passes
unchanged.

**Even fixed, the flag stays disabled for now.** Post-fix, the feature is still measurably
slower per action than the pre-color-blob-salience baseline (a full 8-game/8000-action gate run
still could not complete within the local gate's 115s/game cap, though this specific
measurement is confounded by heavy concurrent system load -- load average 33.93 at measurement
time, from the concurrently-running research conductor plus an unrelated pytest invocation).
Combined with the fact that three follow-on live-path level-up attempts using this feature
(same day, per `ops/known-issues.md`) all returned `honest_null` -- zero measured benefit --
`SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED` is set back to `False` pending a fresh matched-budget
A/B under a quiet system state that shows a real win justifying the residual per-step cost.

#### SCENARIO-ARC-FCP-5591-3-PER-FRAME-CACHE-NOT-PER-CANDIDATE

Given a frame-level quantity (a full-grid connected-component decomposition) is expensive
(O(grid_cells)) and is needed once per next_move() call
When multiple candidate actions in the same call each need to consult that quantity
Then it is computed ONCE per frame and threaded through to every consultation, not recomputed
independently inside each per-candidate scoring call -- a redundant per-candidate recomputation
of a per-frame quantity turns an O(grid_cells) cost into an O(candidates x grid_cells) one, which
at realistic candidate counts on a large grid is indistinguishable from a hang

### REQ-CAPSTONE-4556-2: CrossGameDiscriminativeCandidateRouter Per-Frame Feature Caching

Second incident of the same submission-prep pre-flight session (2026-07-15, immediately
following `REQ-ARC-FCP-5591-3`). After fixing and disabling `ColorBlobSaliencePrior`, a re-run
of `scripts/kaggle/arc_local_submission_gate.py --check` improved (1/8 solved, up from 0/8,
`vc33` recovered) but still lost `lp85`/`m0r0`/`sp80` to the verified baseline.

**Root cause, again found via a live `faulthandler` stack trace on a real slow `lp85` run (not
guesswork):** the stack bottomed out in `arc_value_learner.py:_component_stats_from_grid`, via
`arc_discriminative_router.py:score()`/`rank()` -- the SAME anti-pattern as
`REQ-ARC-FCP-5591-3`, in a different module.
`CrossGameDiscriminativeCandidateRouter.rank()` (the live `discriminative_candidate_router`,
`SUBMITTED_AGENT_CONFIG["discriminative_candidate_router_enabled"] = True`, unchanged since
before 6/30) calls `score()` once per candidate action, and `score()` calls
`cross_game_features_v3(frame, previous_frame, action_id, goal_frame)` fresh every time. Of
that function's four feature groups (`cross_game_features_v2`, `_object_relational_features`,
`_frame_delta_features`, `_predicate_distance_features`, plus `_action_features`), only the
last (`_action_features(action_id)`, a cheap 7-element one-hot) actually depends on the
per-candidate `action_id` -- the other four depend ONLY on `(frame, previous_frame,
goal_frame)`, identical across every candidate in a single `rank()` call, yet were being fully
recomputed per candidate. `_object_relational_features` in particular runs an O(components^2)
greedy frame-matching loop -- a cost this project's own 2026-06-30 investigation
(`arc_value_learner.py`'s docstring, commit `3c721292d`) had already identified as "the real
per-node cost" without recognizing it was being paid redundantly per CANDIDATE on top of per
NODE.

**Why the 6/30 baseline didn't trip this (best available explanation, not fully certain):** the
flag has been `True` since before 6/30 and the code path is structurally unchanged in this
window per `git log`, so this is not a NEW regression in the sense of new code -- it is a
pre-existing O(candidates x components^2) cost that the 6/30 baseline measurement happened to
clear within its time budget, and that today's heavier system load and/or slightly different
candidate counts pushed over the local gate's 115s cap. Unlike `REQ-ARC-FCP-5591-3` (a
genuinely NEW flag flip since 6/30), this fix is a legitimate general performance improvement
to a long-standing code path, not a revert of a recent regression.

**Fix.** `arc_value_learner.py` gains `CrossGameFrameContextV3` (a `NamedTuple` of the four
frame-only feature groups) and `cross_game_frame_context_v3(frame, previous_frame, goal_frame)`
(computes it once). `cross_game_features_v3()` gains an optional keyword-only `frame_context`
parameter -- when given, skips recomputing the four frame-only groups and splices in the fresh
per-candidate `_action_features(action_id)`; when omitted, behaves exactly as before (every
other existing caller -- `arc_world_model_trust_energy.py`, `arc_controllable_novelty.py`, two
`experiment_47xx_structural_energy_*` scripts -- is unaffected).
`CrossGameDiscriminativeCandidateRouter.rank()` now computes the `frame_context` once and
passes it to every `score()` call within that `rank()` invocation.

**Verified.** A real `lp85` run (budget=8000, `CARNOT_ARC_DISABLE_INDUCTION=1`, no induction)
that was previously timing out at the gate's 115s cap now completes in 54s with
`actions=7792` -- an EXACT match to `arc_local_submission_gate.py`'s own
`CANONICAL_BASELINE_ACTIONS_BY_GAME["lp85"]` constant -- and `eff=2.0069`, again matching the
documented baseline floor. `m0r0`/`sp80`/`vc33` (the other three CORE games) all completed
individually within budget too (29s/49s/24s), with `vc33` reaching `actions=7777`, close to its
own `CANONICAL_BASELINE_ACTIONS_BY_GAME` entry (7731). A subsequent FULL 8-game gate run (all
games in parallel via `ThreadPoolExecutor(max_workers=8)`) still failed to complete cleanly --
but system load at that exact moment was 61.40 on a 24-core box (vs 5.97 five minutes earlier),
consistent with resource contention from running 8 CPU-heavy game-evals simultaneously
alongside the continuously-running research conductor, not a remaining code defect. The
per-game ISOLATED measurements (each run alone, no self-contention) are the cleaner signal and
all support the fix. Existing coverage (60 tests across
`test_experiment_4556_verifier_router_generic_transfer.py`,
`test_arc_verifier_variant_augmentation.py`, `test_arc_submitted_agent_parity.py`, and the
`structural_energy_s0/s0prime/s1` + `value_routing_cost_fix_live` + `live_integration_scored_agent`
suites) all still pass unchanged.

Required field principles: not applicable (this is a performance-only fix; the output feature
vector is byte-identical between the cached and uncached paths, verified directly).

#### SCENARIO-CAPSTONE-4556-2-PER-FRAME-CONTEXT-NOT-PER-CANDIDATE

Given `cross_game_features_v3`'s output decomposes into a frame-only part (independent of the
per-candidate action_id) and a cheap action-only part
When a router scores N candidate actions against the SAME frame in a single ranking call
Then the frame-only part is computed ONCE and reused across all N candidates -- computing it
fresh per candidate turns an O(components^2) per-frame cost into an O(candidates x
components^2) per-call one, which at realistic candidate counts is a severe, hang-adjacent
slowdown even though the underlying flag/code path predates the incident that surfaced it

### REQ-ARC-LIVESUBMIT-4679-2: sc25 WARMUP_GAMES Step Missing From Live-Submit Replay

Discovered 2026-07-15 while chasing down a `sc25 claimed L5 -> LIVE L-1 MISMATCH` result that
occurred IDENTICALLY in both a VALIDATE-mode and a subsequent `--submit`-mode run of
`scripts/arc3_live_submit.py` (same failure point both times -- reproducible, not a fluke). This
was NOT a new regression from the 2026-07-14/15 submission-prep session's other fixes above; it
is a pre-existing gap between two sibling replay functions that has been present since sc25's
102-action/L5 banked trajectory was first wired into the live driver.

**Root cause, found via direct log inspection (not guesswork).** The submit-run log
(`/tmp/arc3_live_submit_submit.log`) showed:
```
ERROR | Failed to perform action ACTION4 for game sc25-635fd71a: 400 Client Error: Bad Request
    for url: https://three.arcprize.org/api/cmd/ACTION4
    sc25  claimed L5 -> LIVE L-1  MISMATCH  [13s]
```
i.e. the LIVE ARC Prize API rejected `ACTION4` mid-replay (the 22nd action in the 102-action
banked trajectory, `results/arc3_live_banked_trajectories/sc25.json`, sourced from
`results/experiment_4468_bank_sc25_provisional_levels.json`), which made `env.step()` raise and
`replay_live()`'s loop break early with the `-1` sentinel.

Three independent places in this codebase already know sc25 needs special first-step handling:
`python/carnot/agentic/arc_solver_kit.py`'s module docstring ("4. The FIRST `env.step` after
`env.reset()` is CONSUMED (no-op) in at least sc25.") and its `reproduce()` function (THE
REPRODUCTION GATE that offline-certified sc25's L5 claim, per the current submission package's
`env_match_basis: "offline_reproduction_gated_package_refresh_4679"`), which accepts a
`warmup_label` parameter for exactly this; and
`scripts/arc3_replay_scorecard_metaharness.py`, which defines `WARMUP_GAMES = {"sc25"}` and its
`replay_game()` function explicitly prepends a throwaway warmup `env.step()` (repeating
`actions[0]`) before replaying the real sequence from index 0. `scripts/arc3_live_submit.py`'s
`replay_live()` -- the function that actually drives BOTH the VALIDATE and the `--submit` live
replay -- had no such handling; it iterated `actions` directly with no warmup step. Because
sc25's win condition involves state-dependent tank-controls (per `arc_solver_kit.py`'s docstring:
"6. Some games have STATE-DEPENDENT controls (sc25 tank-controls) ..."), the resulting one-step
phase drift compounds across the trajectory until a specific action becomes illegal for the
live game's actual (drifted) state -- consistent with the observed 400 landing at action 22, not
action 1.

**Fix.** `replay_live()` now checks `short in mh.WARMUP_GAMES` immediately after `env.reset()`
and, if the frame is live and there is at least one action, applies one throwaway
`env.step(ACTION<actions[0].action>, data=actions[0].data, reasoning={"policy": "warmup"})`
before entering the main replay loop -- an exact mirror of `replay_game()`'s existing
WARMUP_GAMES handling. Every other (non-`WARMUP_GAMES`) game's replay is untouched.

**Verified.** Unit-level: `tests/python/test_arc3_live_submit_warmup_games.py` (3 tests, fake
env/arcade/metaharness) proves (a) a `WARMUP_GAMES` game gets exactly one extra `env.step` call
repeating `actions[0]` with `reasoning={"policy": "warmup"}` before the real replay loop begins,
(b) a non-`WARMUP_GAMES` game's replay is byte-for-byte unchanged (no extra step), and (c) an
empty action list or a `None` reset never crashes the new warmup-step guard. Live re-validation
of the fix itself (a fresh non-destructive VALIDATE run confirming sc25 now env-matches) is the
natural follow-up before any future `--submit`, per the project's established validate-before-
submit discipline (Operator-Only External Publication) -- the already-closed 2026-07-15 submit
scorecard is unaffected by this fix (submission is irreversible; this fix only benefits future
runs).

Required field principles: not applicable (this is a live-replay correctness fix; no new
artifact fields).

#### SCENARIO-ARC-LIVESUBMIT-4679-2-WARMUP-STEP-BEFORE-REAL-REPLAY

Given a game is listed in `WARMUP_GAMES` because its live/offline env silently consumes the
first `env.step()` after `env.reset()` as a no-op
When the live-submit driver replays that game's banked trajectory
Then it must first apply one throwaway warmup step (consuming the no-op slot) before starting
the real action sequence from index 0 -- omitting this step shifts every subsequent action one
position out of phase against the actual game state, which for a state-dependent-controls game
compounds into an eventually-illegal action and a live API rejection, masquerading as an
"env-mismatch" even though the underlying banked solution was legitimately offline-reproduced

### REQ-ARC-WMTE-5714: Rescoped `/think` vs `/no_think` Induction-Quality A/B (re-test of REQ-ARC-WMTE-5594)

This re-tests the previously-negative REQ-ARC-WMTE-5594 result under the
Failed-Experiment Rerun Discipline. The rerun is legitimate ONLY because it
names the prior failure, the diagnosed root cause, what is genuinely different,
and a falsifiable acceptance gate -- all below.

**Prior failure (named).** REQ-ARC-WMTE-5594
(`results/experiment_5594_think_mode_induction_quality_ab.json`), verdict
`complete: think_mode_ab_equal_success_no_think_higher_accuracy`. On a 2-game
roster (`m0r0`, `sk48`) both arms induced successfully (4/4) but mean
`heldout_accuracy` favored `/no_think` (0.75) over `/think` (0.5). The prior
itself disclosed three limits: (1) N=2 is below this project's sample-size floor
for any percentage-point claim; (2) `heldout_accuracy` scores world-model
INDUCTION dynamics only, not a win-recognition / actions-to-first-win metric;
(3) NEITHER game's window contained a real level-up, so
`score_goal_predicate_consistency` (REQ-ARC-WMTE-5593, the goal-hypothesis half
of induction quality) was NEVER exercised.

**Diagnosed root cause.** The prior measured the wrong-for-the-question metric
(dynamics-only heldout) on windows that structurally could not exercise the
goal-predicate half of induction quality, at an N too small to separate signal
from noise, with each arm running its OWN `run_game` (arm-dependent windows =
a confound). Separately, the induction pipeline itself has a mechanism that the
prior found but under-weighted: with `CARNOT_ARC_CODEONLY_INDUCE` ON (the frozen
live default), codeonly mode's `_L2_CODEONLY_DIRECTIVE` hardcodes `/no_think\n`
AND a body that says "Do NOT reason ... Skip all reasoning" -- so the
operationally-relevant "flip the toggle" is not even a real reasoning-effort
change.

**What is different (the four deltas).**
1. **Live-agent code churn.** A dozen+ behavioral changes landed in
   `arc_competition_agent.py` since 2026-07-13 (the `REQ-ARC-FCP-5699-*` series;
   most relevantly -38/-39, which fixed FOUR real bugs installing a goal-energy
   bias without checking goal achievability -- a plausible asymmetric confound of
   planning quality between the two arms in the prior run). The prior test
   predates that fix and is stale.
2. **MTP re-enabled on Kaggle-L4-matched hardware.** The live submission kernel
   hardcodes `CARNOT_ARC_MTP=0` (a June decision under a presumed 16GB ceiling);
   Kaggle's scored hardware has been `NvidiaL4` (24GB/card) for weeks. This
   experiment runs against a FRESH MTP-ON `llama-server` pinned to ONE local RTX
   3090 (24GB, VRAM-matched to a single L4 card) via
   `CARNOT_ARC_GENERATOR_CUDA_GPU`. The prior test's proposer passed `mtp=True`
   but CONNECTED to a pre-warmed port-8920 server of UNKNOWN launch config, so its
   true MTP state is not knowable from the artifact -- this run removes the ambiguity.
3. **Larger, level-up-targeted roster.** Adaptered games are solved to L1 offline
   and the winning labels replayed to build, per game, ONE SHARED window that
   STRADDLES the real L0->L1 level-up. This (a) raises N well above 2, and (b)
   makes `score_goal_predicate_consistency` fire on every game -- the prior's
   single biggest gap. The SHARED window (fed to both arms of a comparison)
   also removes the prior's arm-dependent-window confound.
4. **A fuller, more decision-relevant metric.** Beyond dynamics `heldout_accuracy`,
   each arm's induced `is_level_complete` is scored for whether it RECOGNIZES the
   real win at the actual level-up transition (`levelup_positive_recall`) -- the
   induction-quality signal that most determines whether a discovered hypothesis
   leads to real progress, i.e. the thing GPT-5.6's ~26x Low->Max reasoning-effort
   scaling on ARC-AGI-3 actually improved.

**Two comparisons (both on the same shared level-up window; only the think axis
differs within a comparison).**
- **Comparison A -- FROZEN-STACK TOGGLE (codeonly ON):** `A1 no_think` vs
  `A2 think_toggle` (the module constant's `/no_think`->`/think`). This is the
  operator's LITERAL decision.
- **Comparison B -- GENUINE REASONING (codeonly OFF):** `B1 no_think_plain` vs
  `B2 think_plain`. Tests the GPT-5.6 hypothesis (does REAL reasoning improve
  induction quality?), a pipeline change larger than a toggle flip -- reported as
  a secondary, smaller-N signal because B2's full-reasoning calls are ~10-20x
  slower (they reason for thousands of tokens before emitting code).

**Falsifiable acceptance gate / retirement.** The rerun's gate is win-recognition
head-to-head on Comparison A: `/think` is credited only if `A2` beats `A1` on
`levelup_positive_recall` across more games than it loses. Given the RE-CONFIRMED
mechanism finding (the codeonly toggle emits NO `<think>` trace -- verified: the
toggle output opens with `import numpy`, no reasoning tag), the expected and
honest outcome is `think_toggle_inert_under_codeonly` -- the toggle is a no-op,
so flipping it in the frozen stack is not justified. This is NOT a re-run of a
doomed approach: it answers a DIFFERENT, sharper question (win-recognition under
a shared level-up window at larger N, plus a genuine-reasoning arm the prior
never ran) and reaches a mechanistically-grounded conclusion rather than the
prior's under-powered heldout comparison. `retire_if_same_verdict: false` because
the verdict SHAPE differs (mechanistic inertness vs an accuracy delta).

**GUARDRAIL.** OFFLINE DEV MEASUREMENT ONLY. This never edits the frozen
submission stack; it reports a delta + an explicit recommendation and the operator
decides. `solve_provenance: development_proxy`.

Required field principles:

- `codeonly_toggle_inert`: principle "the load-bearing mechanistic finding,
  measured not assumed -- True iff no `A2` (think_toggle) arm produced any
  reasoning trace; if True, the operator's literal toggle flip is a no-op."
- `levelup_positive_recall`: principle "delta #4 -- does the induced
  is_level_complete RECOGNIZE the real win at the actual level-up transition? The
  win-recognition signal reasoning-effort is supposed to improve, which the prior
  never measured (no level-up in any prior window)."
- `mtp_enabled`: principle "delta #2 -- a FRESH MTP-ON server on L4-VRAM-matched
  hardware, removing the prior's unknown-MTP ambiguity."

#### SCENARIO-ARC-WMTE-5714-CODEONLY-TOGGLE-INERT

Given the frozen codeonly induction path (`CARNOT_ARC_CODEONLY_INDUCE=1`) with
`_L2_CODEONLY_DIRECTIVE` swapped `/no_think\n`->`/think\n`
When an `A2` (think_toggle) arm induces on a shared level-up window
Then the raw completion contains NO reasoning tag (`<think>`/`<thinking>`/
`<reasoning>`), `reason_engaged` is False, and if this holds for every `A2` arm
the artifact reports `codeonly_toggle_inert: true` and the verdict
`complete: think_toggle_inert_under_codeonly_no_operator_action_justified`

#### SCENARIO-ARC-WMTE-5714-GENUINE-THINK-ENGAGES

Given the non-codeonly path (`CARNOT_ARC_CODEONLY_INDUCE=0`) with
`no_think_prefix="/think\n"`
When a `B2` (think_plain) arm induces
Then the raw completion of at least one `generate()` call begins with / contains
a reasoning tag and `reason_engaged` is True -- confirming genuine reasoning is
engaged only OUTSIDE codeonly mode, not by the toggle alone

#### SCENARIO-ARC-WMTE-5714-BLOCKS-CLEANLY

Given a missing precondition (no cached GGUF, no CUDA llama-server, or neither an
already-running MTP server nor >=13000 MB free on the pinned 3090)
When `build_artifact` runs
Then `honest_verdict` starts with `complete: blocked_` and no induction attempt
is made on any roster game

**Implementation Status (2026-07-16): DONE.** Experiment script
`python/carnot/experiment_5714_think_mode_rescoped_ab.py`; tests
`tests/python/test_experiment_5714_think_mode_rescoped_ab.py` (7 tests); artifact
`results/experiment_5714_think_mode_rescoped_ab.json` (live MTP-on Qwen3.5-9B-MTP
on one RTX 3090, 1964.9s, adversarial-verify clean).
`honest_verdict: complete: think_toggle_inert_under_codeonly_but_genuine_reasoning_no_winrecognition_delta`.

**Measured result (N=10 adaptered games; dc22 + ka59 skipped — no reproducible
L0->L1 level-up window that run):**

1. **Frozen-stack toggle is INERT (Comparison A).** `codeonly_toggle_inert=true`:
   all 10 `A2` (think-toggle) arms engaged NO reasoning (`reason_engaged=false`,
   10/10; output opens with `import numpy`), because `_L2_CODEONLY_DIRECTIVE`'s
   body ("Do NOT reason ... Skip all reasoning") overrides the toggle. Both A arms
   induced 10/10; `mean levelup_positive_recall` no_think 0.0 vs think 0.1;
   per-game win-recognition head-to-head think/tie/no_think = 1/9/0. The single A2
   "win" (vc33) is STOCHASTIC NOISE, not a reasoning effect — that A2 arm's
   `reason_engaged` is still false (temperature-0.2 variance in the codeonly
   induction, not `/think` doing anything). So flipping the live toggle is a
   measured no-op.
2. **Genuine reasoning fails to induce (Comparison A1-vs-B2 / B-control).** With
   codeonly OFF and the induce fence removed, `/think` genuinely engages
   (`reason_engaged=true` on 9/10 `B2` arms) but OVERRAN the 8192-token budget on
   ALL 10 games (reasoned ~7000+ tokens, `HIT n_predict OUTPUT LIMIT`, never
   emitted the required functions -> `induction_ok=false` 10/10). So the
   genuine-reasoning-vs-frozen comparison has `n_games_both_induced=0` — genuine
   reasoning produced NO usable world model on any roster game. (An isolated probe
   on dc22 — a game NOT in the final roster — did once succeed with recall 1.0, so
   concise-reasoning success is possible but did NOT replicate on the 10-game
   roster.)
3. **The code fence is load-bearing (B1 control).** `B1` (no-fence `/no_think`)
   also induced 0/10 (`HIT n_predict` limit) — removing the fence breaks even
   no_think code extraction, so any B2 gain would be confounded by the fence
   change, not attributable to reasoning alone.
4. **heldout_accuracy is floor-0** on the level-up windows for every arm
   (exact-grid prediction near a boundary is too hard), so win-recognition is the
   discriminating axis; MTP re-enablement is a throughput feature (same model
   self-drafts) with no quality delta.

**Recommendation (operator decision; NOT changed here per the frozen-live-stack +
Operator-Only disciplines):** leave the frozen live stack's `/no_think` UNCHANGED
(flipping the toggle is a measured no-op) and treat `CARNOT_ARC_MTP=0` as a
separate throughput call. Genuine reasoning is NOT a safe drop-in for this 9B
induction path (it overruns the budget and depends on removing the load-bearing
fence), so GPT-5.6's ~26x reasoning-effort scaling does NOT transfer here as a
config flip. This is a stronger, better-powered NEGATIVE than the prior's N=2
heldout comparison, not a re-run of a doomed approach. `retire_if_same_verdict:
false` (the verdict SHAPE — mechanistic inertness + overrun — differs from the
prior's accuracy-delta framing).

### REQ-ARC-LIVESUBMIT-LF52-INBOUNDS: Live-Submittable Replays Must Use Only In-Bounds ACTION6 Coordinates (lf52 L9 Camera-Crane Fix)

A banked ARC-AGI-3 replay sequence intended for LIVE submission SHALL emit only
ACTION6 click coordinates within the visible frame (pixel x,y each in 0..63).
The OFFLINE `arc_agi` arcade permissively accepts and routes ACTION6 clicks with
coordinates OUTSIDE the 64x64 frame (addressing off-screen objects directly by
their source-grid position), but the LIVE API validates coordinates strictly and
rejects anything outside 0..63 with HTTP 400. A sequence that reproduces a level
OFFLINE via out-of-bounds clicks is therefore NOT live-submittable to that depth;
its live-submittable depth is capped at the last level reached before the first
out-of-bounds click.

**Origin (2026-07-17):** lf52 was solved offline 10/10 (commit `c588364d7`) but a
live VALIDATE-mode replay of the banked 1009-action sequence
(`results/outer_loop_live_revalidation_20260717.json`) reached only L8 before
failing at action 849 with HTTP 400 on `ACTION6 {x:132,y:18}` — one of 22
out-of-bounds clicks (x up to 132 / down to -8) that walked off-screen blue relay
pegs (source grid x~11-21) in the L9 tail. All 22 were in the L9 segment; the
L1-L8 prefix and the L10 suffix were already fully in-bounds.

**Reference fix (development_proxy, offline dev-twin restructuring — NOT a new
solve; lf52 was already 10/10):** the L1-L8 prefix and the fresh-init L10 suffix
are kept UNCHANGED; only the L9 completion is restructured so every ACTION6 target
renders within 0..63. The repository's understanding of the lf52 L9 camera model
SHALL be treated as ground truth for any future lf52 live work:

- Render pixel of a grid cell `(gx,gy)` = `(6*gx + world_offset_x, 6*gy +
  world_offset_y)`; `world_offset` starts `(-15,5)` on L9 (after the scripted
  first pan).
- A SCRIPTED `-20px` world pan fires whenever ANY peg LANDS at cell `(6,5)` (every
  time, not only once).
- A carrier ACTION1-4 move pans the world by `(-dx*6, -dy*6)` ONLY when a GREEN
  peg (`fozwvlovdui`, not `fozwvlovdui_blue`) rides the carrier cell; an empty or
  blue-only carrier does NOT pan. The carrier rail is row `y=5`, `x=6..20`.
- Consequently the carried green is a controllable "camera crane": keeping a green
  on the carrier and shuttling it along the rail brings any east-chamber peg into
  the visible window before it is clicked. Blue pegs are indestructible relays;
  only a green-over-green same-name jump reduces the win count.

An in-bounds L9 completion SHALL be found by a search that constrains every
ACTION6 target to be in-bounds at the current camera offset, over a model that is
cross-checked against the real engine (the reference fix used a constrained
best-first search over a faithful pure-Python L9 model, validated op-by-op against
the real offline engine).

**Acceptance (all MUST hold for the fix to count):**

- `out_of_bounds_action6_clicks == 0` across the whole assembled sequence.
- A fresh offline-arcade replay reaches `levels_completed == 10` and
  `GameState.WIN`.
- `arc_solver_kit.reproduce(claimed_level=10)` returns `reproduced == True`, and 5
  additional fresh replays all reach L10 (registry robustness).
- A LIVE VALIDATE-mode replay (scorecard opened but NEVER closed; no `--submit`, no
  leaderboard record) reaches live L10, replaying all actions past the original
  action-849 failure point.

The reference artifact is
`results/outer_loop_lf52_l9_inbounds_live_parity_20260717.json` (1033 actions,
sha256 `2b03a9d5fbe7b79e1cd46ef6ff981eef447e41b0a152db6c05e0a2da8e31ebd5`,
live replay `https://arcprize.org/replay/6a0876aa-5c8b-44c3-959d-c4be272b6e38`).

Required field principles for a live-parity route-fix artifact:

- `solve_provenance`: MUST be `development_proxy` (offline dev-twin restructuring of
  an already-banked route; not a live self-discovery and not a new bank).
  `new_levels_banked` MUST be 0 when the game is already at that depth.
- `out_of_bounds_action6_clicks`: bare int; the necessary-condition gate for live
  submittability (MUST be 0).
- `inference_substrate`: `live_scored_env_real_network_calls_no_llm` (no LLM; the
  headline evidence is the real live replay).
- `honest_verdict`: MUST start with a terminal prefix
  (`complete:`/`complete_`/`success:`/... per Verdict Terminal-Prefix Discipline)
  and name both the offline reproduction AND the live-parity outcome.
- `reproducibility_checksum`: sha256 over the concatenated compact JSON of every
  concrete action, so a third party can confirm the exact submitted sequence.

**Generalization:** other banked games' deep tails SHALL be audited for the same
off-screen-click pattern before their live-submittable depth is trusted; the
camera-crane restructuring is the reusable remedy where a game exposes an
equivalent carried-piece camera-follow mechanic.

### REQ-ARC-WMTE-5720: Actions-to-Progress Bounded-Live-Solve Metric Harness + Re-Test of the `/think` vs Frozen A/B

This defines a reusable **actions-to-progress** metric harness and uses it to
re-resolve the two capability A/Bs that both landed inconclusive in 2026-07:
REQ-ARC-WMTE-5714 (`/think` vs `/no_think`) and REQ-ARC-WMTE-5716/5717/5718 /
exp5719 (playbook-exemplar injection). BOTH prior results explicitly named the
SAME root cause -- their metric (single-shot world-model induction quality:
graded cell-recall / goal-predicate accuracy on ONE fixed transition window) was
floored and high-variance, and it did not measure the thing that matters
(CLAUDE.md "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent"): whether a
capability change helps the agent make REAL PROGRESS in a live-style solve, not
predict one grid transition. exp5719's own agent wrote: "A live-submission
`levels_gained` A/B remains the only definitive test."

The harness (`python/carnot/agentic/arc_actions_to_progress.py`) drives the
ACTUAL scored `E3AgentPolicy` cascade (the same `next_move`/`is_done` loop
`scripts/arc_leaderboard_eval.py:run_game` and `make_carnot_agent.choose_action`
run in production -- one of the two live entrypoints per the ARC Live-Path
Reachability Discipline, so this is the live mechanism, not a parallel solver) on
the offline arcade for a bounded solve, per arm. Each arm is activated through
its REAL live env-var wiring (`CARNOT_ARC_CODEONLY_INDUCE`,
`proposer.no_think_prefix`, `CARNOT_ARC_PLAYBOOK_RETRIEVAL` /
`_EXEMPLARS_ENABLED`), so a result here is directly relevant to a graduation
decision.

**Three floor-busting signals (a ladder from decisive-but-sparse to dense-but-indirect).**
1. `levels_gained` / `actions_to_first_solve` -- the RHAE-relevant ground truth
   (a real level-up = `frame.levels_completed` advanced; the win oracle is the
   level counter, NEVER a heuristic, so `verifier_is_oracle=False`). Sparse on
   the live frame-only path (most public games do not level up live, consistent
   with the ~0.08 live score), so confirmatory, not primary.
2. `hv_progress` -- a DENSE goal-distance-reduction proxy: the per-game
   `GameAdapter.hand_verifier` is a hand-written distance-to-goal (LOWER = closer;
   the OfflineSolver best-first-searches on it). We track its MINIMUM over the run
   and report `(start_hv - best_hv) / max(start_hv, 1)`. It changes whenever the
   board gets closer to a win, so it discriminates even when no arm achieves a
   full level-up -- the exact floor the single-shot proxy could not clear. It is a
   MEASUREMENT (oracle-distinct from both the win gate and the agent's own routing
   verifier), not a win oracle.
3. Induction-quality aggregated over the run's MULTIPLE induction attempts:
   `plan_found_rate` (fraction of attempts that produced a reachable plan) and
   `mean_heldout_accuracy` (how close the induced model got to the live planning
   gate). These are affected by induction CONTENT (reasoning / retrieval) even
   when the strict live gate blocks every plan, and averaging over the attempts
   the agent makes inside ONE solve on its OWN collected transitions is
   lower-variance than the prior single-shot induction.

**Roster selection (why these 6 games).** A fast induction-disabled probe
selected `ls20 tr87 lp85 g50t m0r0 ft09`: adaptered games that (a) reach the
stall->induce phase within budget and (b) have a NON-DEGENERATE `hand_verifier`
the agent can actually REDUCE (`best_hv < start_hv`). Games whose `hand_verifier`
was flat (`cn04 sp80 sk48 ka59`) or monotonically increasing (`ar25 dc22 cd82`)
were excluded because the dense proxy cannot discriminate on them. `lp85`
additionally solves L0->L1 under the pure explorer, giving a live level-up signal.

**Tractability tradeoff (stated explicitly).** A full live solve per (game, arm,
seed) is expensive; the dominant cost is LLM induction (~30-100s on a warm CUDA
server; a genuine `/think` attempt is ~2-3 min). Each run is bounded three ways:
an action `budget` (the real eval constraint), a `max_inductions` cap (the
PRIMARY cost/fairness bound -- each arm gets the SAME number of induction
attempts, so a slower arm is not penalized by wall time, which the live eval does
not limit), and a `wall_s` safety cap (recorded honestly as `timed_out`).
`explore_budget=24` (`SUBMITTED_ROUTED_EXPLORE_BUDGET`, a real shipped value) is
set so every bounded run reliably reaches the induce->plan->execute phase the arms
differ in -- without it, the graph-explore route's default 80-transition budget
means a short run never induces and all arms are identical. Runs are PAIRED
within (game, seed) so shared exploration luck cancels in the delta, and reported
with paired win/tie/loss + exact sign-test p + an outlier-fragility flag (does
dropping the single largest-magnitude pair flip the sign), NOT a bare mean.

**Substrate / provenance.** `inference_substrate=live_llm_inference` (the real
Qwen3.5-9B-MTP GGUF induces on a CUDA `llama-server`; MTP speculative decoding is
content-neutral, so the arms differ in the request, not the MTP flag; the CUDA
3090 substrate gives dev throughput and produces induction CONTENT identical to
the iGPU frozen live stack -- wall-clock is NOT Kaggle-representative, only the
CONTENT questions are). `solve_provenance=development_proxy` on PUBLIC games (NOT
a hidden-game self-discovery solve); `read_game_source=False` (the dense proxy
reads the live runtime game object via the adapter's public callable,
`used_env_source=True`, never a game's `.py` source). The harness NEVER flips the
frozen live default and NEVER submits.

**Question (this REQ).** Does removing the codeonly fence + genuine `/think`
reasoning (`CARNOT_ARC_CODEONLY_INDUCE=0`, `no_think_prefix=/think`, 8192
n_predict) help actions-to-progress vs the frozen codeonly + `/no_think` default
(4096 n_predict)? Prior (REQ-ARC-WMTE-5714): genuine reasoning DID engage on
9/10 games but overran the token budget and produced NO usable single-shot
induction; the narrower win-recognition metric showed no delta.

**Falsifiable gate / retirement.** Graduate genuine reasoning to the frozen
default ONLY on a significant (`sign_test_p < 0.05`), NOT-outlier-fragile
positive paired delta on the primary metric (`hv_progress`), corroborated by a
non-negative `plan_found_rate` / `mean_heldout_accuracy` delta. Any other outcome
(no reliable signal, negative, or fragile) recommends KEEPING the frozen default
and is a terminal honest result (NOT retried) -- an honest "the solve-loop metric
also cannot separate the arms, here is why" is a valid, citable outcome.

**Result artifact:** `results/experiment_5720_actions_to_progress_reason_ab.json`.

### REQ-ARC-WMTE-5721: Actions-to-Progress Re-Test of Playbook-Exemplar Retrieval Injection (re-test of REQ-ARC-WMTE-5716/5717/5718)

Uses the REQ-ARC-WMTE-5720 harness to re-resolve the retrieval question exp5719
left open. exp5719 found retrieval-based injection beat static injection but
neither beat no-injection on graded cell-recall (N=8/arm,
`no_reliable_signal_high_variance`), and its agent recommended a live
`levels_gained` A/B as the only definitive test. This runs the retrieval feature
through its REAL live wiring (`CARNOT_ARC_PLAYBOOK_RETRIEVAL=1`, which makes
`E3AgentPolicy._induce_and_plan` embed the stuck situation with the same GGUF,
retrieve the top-K playbook patterns, and inject them into the induction prompt)
inside the bounded live e3 solve, and measures the same actions-to-progress
signal ladder.

**Three contrasts, all paired within (game, seed):** `retrieval_vs_none`
(retrieval vs the frozen no-injection control -- the primary), `static_vs_none`
(the fixed exemplar block vs control), and `retrieval_vs_static` (does the
retrieval step earn its complexity over a fixed block -- the core exp5719
question). All arms hold the frozen codeonly + `/no_think` induction path
constant; only the injected prefix differs (none / static block / retrieved
top-K), so the contrast isolates injection, mirroring exp5719's design but with
the actions-to-progress metric replacing single-shot cell-recall.

**Falsifiable gate / retirement.** Graduate retrieval injection to the frozen
default ONLY on a significant, not-outlier-fragile positive `retrieval_vs_none`
paired delta on `hv_progress`, corroborated by `plan_found_rate` /
`mean_heldout_accuracy`. Otherwise KEEP the dev-gated-off default; an honest
no-reliable-signal outcome is terminal.

**Result artifact:**
`results/experiment_5721_actions_to_progress_retrieval_ab.json` (shares the
frozen/none control runs with REQ-ARC-WMTE-5720 via the common shard
`results/exp5720_atp_shard.jsonl`).

#### SCENARIO-ARC-WMTE-5720-HARNESS-DRIVES-LIVE-E3
- **WHEN** `arc_actions_to_progress.run_bounded_progress(game, arm, proposer=..., seed=..., budget, max_inductions, wall_s, explore_budget)` is called
- **THEN** it SHALL drive the real `E3AgentPolicy` (`next_move`/`is_done`) on the offline arcade, configure the arm ONLY through the live env-var / proposer wiring, and return a `ProgressResult` with `levels_gained`, `actions_to_first_solve`, `hv_progress`, `plan_found_rate`, `mean_heldout_accuracy`, and honest `timed_out` / `hit_induction_cap` / `error` flags, restoring all mutated env vars and proposer fields on exit.

#### SCENARIO-ARC-WMTE-5720-PAIRED-HONEST-STATS
- **WHEN** `paired_summary(results, treat, base, metric=...)` compares two arms
- **THEN** it SHALL pair strictly within `(game, seed, variant)`, drop pairs with a `None` metric, and report `mean_delta`, `wins/ties/losses`, an exact two-sided sign-test p-value, and an `outlier_fragile` flag (whether dropping the single largest-magnitude pair flips the sign) -- never a bare mean presented as a conclusion.

#### SCENARIO-ARC-WMTE-5721-RETRIEVAL-THROUGH-LIVE-WIRING
- **WHEN** the `retrieval` arm runs with `CARNOT_ARC_PLAYBOOK_RETRIEVAL=1`
- **THEN** the induced-attempt diagnostics SHALL record `playbook_injection_mode == "retrieval"` (proving the real live retrieval path fired), and the `retrieval_vs_none` / `retrieval_vs_static` contrasts SHALL be reported with the same paired-honesty guards as SCENARIO-ARC-WMTE-5720-PAIRED-HONEST-STATS.

### REQ-ARC-WMTE-5722: Generator-Swap Induction-Floor Test -- Gemma-4-31B-it vs the Frozen Qwen3.5-9B-MTP (extends REQ-ARC-WMTE-5720/5721)

The REQ-ARC-WMTE-5714 (`/think` vs `/no_think`) and REQ-ARC-WMTE-5716/5717/5718
(playbook retrieval) A/Bs, re-run on the REQ-ARC-WMTE-5720/5721 actions-to-progress
harness, BOTH found the frozen live generator (Qwen3.5-9B-MTP) at a hard
`heldout_accuracy` floor (0.0 on every game/trial; no real level-up) that neither
reasoning-mode nor retrieval-augmentation moved. Both prior levers change the
induction *prompt*; this REQ tests the remaining hypothesis those A/Bs could not
address -- that the binding constraint is model *capacity*, not prompting. It
substitutes a stronger open-weight SOTA generator (`unsloth/gemma-4-31B-it-GGUF`,
Q4_K_M, ~18.3GB) for Qwen3.5-9B-MTP in the induce call ONLY, holding the harness,
roster, seeded level-up windows, planner, verifier, and execution identical, so any
difference is attributable to the generator swap alone.

**Clean-swap design (why it is confound-free).** The swap uses the SAME `frozen`
arm the prior baselines used (`CARNOT_ARC_CODEONLY_INDUCE=1`, 4096 n_predict). In
that arm `LocalGGUFProposer.generate` takes the codeonly branch, which prepends the
model-agnostic codeonly directive and NEVER prepends the Qwen-specific
`no_think_prefix` -- so the induce prompt is byte-identical in structure across
generators and the `/no_think` vs `/think` toggle (a separate, already-tested lever)
is NOT a confound here. The `reason`/`retrieval`/`static` arms are deliberately NOT
re-run per-generator: `/think`/`/no_think` are Qwen tokens (meaningless to Gemma)
and reasoning/retrieval were independently ruled out as levers, so mixing them in
would conflate the generator question with prompt questions already answered.

**Substrate / provenance (unchanged from REQ-ARC-WMTE-5720).**
`inference_substrate=live_llm_inference` -- the real Gemma-4-31B-it GGUF induces on a
CUDA `llama-server` (GPU-offload verified: `llama_supports_gpu_offload()==True`, VRAM
loads on the outer-loop's GPU 1 via `CARNOT_ARC_GENERATOR_CUDA_GPU=1`, real
GPU-speed generation). MTP is OFF for Gemma (the dense GGUF has no MTP self-draft
heads; MTP is speed-only/content-neutral, so this does not change induction
content). The Qwen3.5-9B-MTP baseline is re-measured FRESH in the same harness
invocation (same session/server-warmth) for a clean matched comparison, not reused
from the REQ-ARC-WMTE-5720 shard, to remove any stale-baseline objection; it is
expected to reproduce the 0.0 `heldout_accuracy` floor as a sanity check.
`solve_provenance=development_proxy` on PUBLIC games; `read_game_source=False`,
`used_env_source=True`, `verifier_is_oracle=False`. The wall-clock is NOT
Kaggle-representative (a 24GB 3090 dev card, not the ~16GB eval GPU); this tests
the CONTENT question (does more capacity break the floor), and a positive result
would only recommend a stronger generator PENDING a separate real-VRAM/latency
feasibility check on the actual Kaggle hardware -- it does NOT itself flip the
frozen live default (an operator-only graduation decision, and a bigger change than
the reasoning/retrieval toggles).

**Question (this REQ).** On the identical 6-game roster (`ls20 tr87 lp85 g50t m0r0
ft09`), 2 trials/game, does the stronger Gemma-4-31B-it generator move the induction
floor on the same signal ladder (`reached_levelup` / `hv_progress` / `plan_found` /
`heldout_accuracy` / `cell_recall` / `goal_predicate_accuracy` /
`levelup_positive_recall`), paired by game vs the fresh Qwen3.5-9B-MTP baseline?

**Falsifiable gate / interpretation.** A stronger generator is worth pursuing (as a
candidate live-stack replacement, pending real-hardware feasibility) ONLY on a
positive paired delta on a real-progress metric (`reached_levelup` first, else
`heldout_accuracy`/`hv_progress`) that is directionally consistent across games
(small-N: report win/tie/loss + exact sign-test p + outlier-fragility, never a bare
mean; with <=6 game pairs the sign test cannot reach p<0.05 unless every game
agrees). If the floor persists at 0.0 even with the much larger model, that is an
equally valuable terminal result pointing AWAY from "use a bigger model" and TOWARD
prompt/architecture/gate as the binding constraint -- reported plainly, not spun.

**Result artifact:** `results/experiment_5722_generator_swap_gemma31_ab.json`.

**Implementation Status (measured 2026-07-18): DONE -- clean NULL.** All 36 cells re-run
under the attribution fix (SCENARIO-ARC-WMTE-5722-INDUCE-ATTRIBUTION-GUARD). `heldout_accuracy`
is floored at 0.0 for BOTH Gemma-4-31B-it and the Qwen3.5-9B-MTP baseline (paired delta 0.0, 6
ties); ZERO real level-ups on any cell for any generator. Gemma-31B induces reliably (11/12
cells) and shows only a modest, NON-significant edge on the partial `cell_recall` proxy (0.0995
vs 0.0034, 3W/2T/1L, sign-test p=0.625) -- not exact-match, not a level-up. A stronger generator
does NOT move the induction floor on this codeonly world-model-induction path. Recommendation:
KEEP the frozen live default (operator-only decision; this experiment does not flip it). NB: an
EARLIER buggy-harness run had shown Gemma-31B at heldout 0.25 on 2 cells; that did NOT replicate
under clean attribution (small-N stochastic noise) -- see the artifact `attribution_integrity`.

### REQ-ARC-WMTE-5723: Generator-Swap Size-vs-Capability Datum -- Gemma-4-12B-it (secondary to REQ-ARC-WMTE-5722)

A lighter secondary arm on the SAME REQ-ARC-WMTE-5722 harness/roster/metric:
substitute `unsloth/gemma-4-12B-it-GGUF` (Q4_K_M, ~7.1GB) for the generator and pair
it by game against BOTH the fresh Qwen3.5-9B-MTP baseline and the Gemma-4-31B-it arm.
This gives a size-vs-capability data point (does induction quality scale with model
size at all on this task) and, because the 12B is close to the ~16GB Kaggle VRAM
envelope, a more deployment-relevant capacity probe than the 31B. Same falsifiable
gate and small-N honesty guards as REQ-ARC-WMTE-5722; the 31B is the primary ask and
this must not displace it. **Result artifact:**
`results/experiment_5723_generator_swap_gemma12_ab.json`.

**Implementation Status (measured 2026-07-18): DONE -- clean NULL, and a distinct failure mode.**
Under the attribution fix, Gemma-4-12B-it FAILS to induce a valid world model on 11/12 cells
(`induce_ok=False` -- it cannot reliably emit valid `engine`+`is_level_complete` code under the
codeonly path); its 1 successful induction scored `heldout_accuracy` 0.0. So the mid-size 12B is
WORSE than the 9B here, NOT "~ 31B": induction quality does not scale smoothly with size on this
task. (The `gemma31_vs_gemma12` `cell_recall` contrast favors 31B, +0.4618 on the single
comparable game, reflecting that 31B induces where 12B fails.) The pre-fix run had shown 12B
"moving the floor" on 4 cells -- those were 100% stale re-reads of Gemma-31B's engines and are
RETRACTED. Recommendation: KEEP the frozen default; a smaller-but-SOTA generator is not a
shortcut.

#### SCENARIO-ARC-WMTE-5722-GENERATOR-IS-ONLY-VARIABLE
- **WHEN** `experiment_5722_generator_swap_ab.py` runs a `(generator, game, trial)` cell
- **THEN** it SHALL call `arc_actions_to_progress.run_seeded_progress(game, "frozen", proposer=<generator's LocalGGUFProposer>, ...)` with the SAME seeded `build_progress_window` input, planner (`plan_in_model`), verifier (`WorldModelVerifier`), and offline execution for every generator, so that the LLM identity used in `proposer.induce` is the ONLY variable, and SHALL record the exact resolved `.gguf` path + `llama_supports_gpu_offload()` in `preconditions_checked`.

#### SCENARIO-ARC-WMTE-5722-FRESH-BASELINE-AND-PAIRED-HONESTY
- **WHEN** the driver emits the `gemma31_vs_qwen9b` (and `gemma12_vs_qwen9b`, `gemma31_vs_gemma12`) comparisons
- **THEN** the Qwen3.5-9B-MTP baseline SHALL be re-measured fresh in the same run (not reused from the REQ-ARC-WMTE-5720 shard), and each comparison SHALL be reported via `paired_by_game` with `mean_delta`, `wins/ties/losses`, an exact sign-test p-value, and an `outlier_fragile` flag -- with the artifact stating plainly whether the floor moved, never presenting a bare mean as a conclusion.

#### SCENARIO-ARC-WMTE-5722-INDUCE-ATTRIBUTION-GUARD
- **WHEN** `arc_actions_to_progress.run_seeded_progress` runs a cell whose `proposer.induce` FAILS (returns a falsy success flag and does not overwrite the per-game `world_model.py`)
- **THEN** the harness SHALL (a) have DELETED the prior `world_model.py` before inducing and (b) gate `induction_ok` through `_attribution_ok(induce_ok, engine, is_done)` so the cell reports `induction_ok == False` and `heldout_accuracy is None` (never the score of an earlier generator's leftover engine), and SHALL record `induce_ok` on the row so an auditor can confirm every scored engine belongs to that cell's own generator. This closes the stale-engine cross-contamination bug the exp5722 generator-swap adversarial review found (a failed induce was previously scored on the prior generator's engine, e.g. Gemma-12B's apparent floor-move cells were stale re-reads of Gemma-31B's engines).

### REQ-ARC-WMTE-5724: Token-Efficient Reasoning Test -- ThinkingCap-Qwen3.6-27B vs the Genuine-Reasoning Budget-Overrun Wall (extends REQ-ARC-WMTE-5714; sibling to REQ-ARC-WMTE-5722/5723)

REQ-ARC-WMTE-5714's genuine-reasoning arm (B2: codeonly fence REMOVED, `/think`, 8192
n_predict, single combined call via `_induce_no_fence`) found a specific, distinct
failure mode: on the frozen Qwen3.5-9B-MTP generator, `/think` reasoning ACTUALLY
ENGAGED (a real `<think>` trace on ~9/10 games) but OVERRAN the 8192-token completion
budget on ALL 10 games -- the model reasoned past the budget without ever emitting the
required `engine` + `is_level_complete` code, so `induction_ok = False` on every cell
(`0/10` successful inductions). REQ-ARC-WMTE-5720's `reason` arm independently
reproduced this (`0/12` honest `induce_ok` on the 6-game roster; the pre-fix
`induction_ok=12` were stale re-reads, retracted). The finding is NOT that the
reasoning was wrong -- it is that genuine reasoning never FINISHED within budget.
This REQ tests a lever the prior A/Bs could not: whether a model specifically RL-tuned
for **token-efficient reasoning** (`bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF`, Q4_K_M,
~16GB -- fine-tuned to use ~50% fewer thinking tokens on average while preserving
answer quality) completes reasoning + emits valid code within that SAME 8192-token
budget more often than vanilla Qwen did (`0/N`). Because ThinkingCap is a Qwen3.6-family
model, `/think`/`/no_think` are valid control tokens for it (unlike the Gemma generators
of REQ-ARC-WMTE-5722/5723, where the reasoning toggle is meaningless) -- so this is the
one generator swap for which re-running the genuine-reasoning arm is well-posed.

**Design (faithful to the REQ-ARC-WMTE-5714 B2 failure mode).** The induce step reuses
exp5714's `_induce_no_fence` mechanism EXACTLY -- codeonly OFF, `no_think_prefix=/think`,
`max_tokens=8192`, `tries=1`, and CRUCIALLY no pre-opened ```` ```python ```` fence
(the real `induce()` appends that fence-opener, which suppresses the `<think>` trace;
`run_seeded_progress`'s built-in `reason` arm would therefore NOT reproduce B2 and is
deliberately not used for the induce step). Only the generator LLM changes; the
downstream measurement is the REQ-ARC-WMTE-5720 actions-to-progress ladder
(`load_engine` -> `plan_in_model` -> execute against the real offline env ->
`WorldModelVerifier`/`score_goal_predicate_consistency`), identical to the sibling
generator-swap experiments. A fresh vanilla Qwen3.5-9B-MTP `reason` baseline is measured
in the same invocation (matched control), and REQ-ARC-WMTE-5714/5720 are cited as the
two prior independent confirmations of the `0/N` floor.

**Substrate / provenance.** `inference_substrate=live_llm_inference` -- ThinkingCap-27B
really induces on a CUDA `llama-server` on the outer-loop's GPU 1
(`CARNOT_ARC_GENERATOR_CUDA_GPU=1`; GPU-offload verified `llama_supports_gpu_offload()
==True` + a VRAM jump on load + a `/think` `/completion` round-trip confirming the
Qwen3.6 prompt plumbing). Server config: `n_ctx=22000` (lp85's 64x64 grid overflows the
class default, matching exp5599/exp5705), `-fit off` (the Qwen3.6-27B hybrid
linear/full-attention arch hard-hangs on the default `-fit` heuristic; learned in the
exp5705 diagnostic loop), `q8_0` KV, MTP off (this GGUF has no self-draft heads; MTP is
speed-only/content-neutral). `solve_provenance=development_proxy` on PUBLIC games;
`read_game_source=False`, `used_env_source=True`, `verifier_is_oracle=False`. The
24GB-3090 dev card is NOT Kaggle-representative (the eval GPU is ~16GB and ThinkingCap-27B
Q4 is a tight fit there); this tests the CONTENT question only and NEVER flips the frozen
live default (operator-only) and NEVER submits.

**Question / primary metric (this REQ).** On the identical 6-game roster (`ls20 tr87
lp85 g50t m0r0 ft09`), 2 trials/game, does token-efficient ThinkingCap-27B COMPLETE
genuine-reasoning inductions within the 8192-token budget more often than vanilla
Qwen's `0/N`? The PRIMARY metric is the induce COMPLETION RATE (`induce_ok`: did the
cell finish reasoning and emit parseable `engine`+`is_level_complete` code before hitting
n_predict), reported alongside `reason_engaged` (confirming reasoning actually engaged,
via the exp5714 `<think>` tag detection) and the overrun signal (`last_stop_type=="limit"`
/ `max_raw_completion_len`). The full signal ladder (`reached_levelup` / `hv_progress` /
`plan_found` / `heldout_accuracy` / `cell_recall` / `goal_predicate_accuracy` /
`levelup_positive_recall`) is measured and paired by game as secondary color, but the
completion-rate comparison against Qwen's `0/N` is the direct test of the
token-efficiency claim itself, independent of whether it also improves quality.

**Falsifiable gate / interpretation.** (a) Token-efficient reasoning is a viable path to
make `/think` usable ONLY if ThinkingCap completes MATERIALLY more inductions within the
same 8192 budget than vanilla Qwen's `0/N` -- a real, actionable completion-rate delta
(and better still if those completions translate into a real-progress signal on the
ladder). (b) If ThinkingCap STILL overruns on most/all cells, the conclusion flips to
"8192 tokens is simply too tight a budget regardless of efficiency tuning" -- pointing
toward RAISING the budget (or splitting the induce) as the lever rather than swapping in
a more efficient model. Both outcomes are equally reportable; a clean negative is stated
plainly, never spun. Small-N honesty guards apply as in REQ-ARC-WMTE-5722 (<=6 game
pairs, stochastic proposer: report win/tie/loss + exact sign-test p + outlier-fragility,
never a bare mean; a positive is a DIRECTION to investigate, not a significance claim).
This REQ does not itself graduate any generator or budget change to the live stack.

**Result artifact:** `results/experiment_5724_thinkingcap_token_efficient_reason_ab.json`.

**Implementation Status:** Run 2026-07-18 (GPU 1, after the REQ-ARC-WMTE-5722/5723
generator-swap run freed the card). Driver
`python/carnot/experiment_5724_thinkingcap_token_efficient_reason_ab.py`; artifact
`results/experiment_5724_thinkingcap_token_efficient_reason_ab.json` (adversarial-verify clean).
**Result: INCONCLUSIVE for the token-efficiency claim** — on 10/12 ThinkingCap-27B reason cells the
model emitted an immediate end-of-sequence with ~0 output on the no-fence raw-`/completion` `/think`
path (a prompt-format artifact, NOT a genuine reasoning attempt); only 2/12 genuinely reasoned and
0/2 completed (1 overran the 8192 budget, 1 finished reasoning but emitted unparseable code). The
SECONDARY `thinkingcap27_frozen` codeonly arm induced cleanly on 12/12 cells (model, server, GPU
offload and prompt plumbing all healthy — verified: VRAM 4->17176 MiB on load, `/think` smoke
round-trip OK), isolating the failure to the raw-completion genuine-reasoning path. The fresh
Qwen3.5-9B-MTP reason baseline completed 1/12 (12/12 genuine attempts, 12 overran) — reproducing the
~0/N budget wall of REQ-ARC-WMTE-5714/5720. Recommended follow-up: re-test ThinkingCap through its
proper Qwen3.6 chat template before any budget-wall conclusion. Does NOT flip the frozen live default.

#### SCENARIO-ARC-WMTE-5724-GENUINE-REASONING-INDUCE-FAITHFUL-TO-5714
- **WHEN** `experiment_5724_thinkingcap_token_efficient_reason_ab.py` runs a genuine-reasoning `(generator, game, trial)` cell
- **THEN** it SHALL induce via the exp5714 `_induce_no_fence` path (codeonly OFF, `no_think_prefix=/think`, `max_tokens=8192`, `tries=1`, NO pre-opened `python` fence) so `/think` genuinely engages, SHALL NOT route the induce step through `run_seeded_progress`'s built-in `reason` arm (whose real `induce()` fence-opener suppresses the `<think>` trace), and SHALL measure the downstream signal ladder with the SAME REQ-ARC-WMTE-5720 machinery (`load_engine` -> `plan_in_model` -> execute -> `WorldModelVerifier`) used by the sibling generator-swap experiments.

#### SCENARIO-ARC-WMTE-5724-COMPLETION-RATE-IS-PRIMARY
- **WHEN** the driver emits the ThinkingCap-27B vs fresh-Qwen3.5-9B-MTP comparison
- **THEN** the artifact SHALL report per-generator induce COMPLETION RATE (`n_induce_ok / n_cells`) as the primary metric, annotate each cell with `reason_engaged` and the overrun signal (`last_stop_type`/`max_raw_completion_len`), re-measure the Qwen baseline FRESH in the same invocation (citing REQ-ARC-WMTE-5714 `0/10` and REQ-ARC-WMTE-5720 `0/12` as prior confirmations), and state plainly whether token-efficient reasoning cleared the budget wall or the overrun persists -- never presenting a bare mean or an induction-quality proxy as the headline in place of the completion-rate result.

#### SCENARIO-ARC-WMTE-5724-THINKINGCAP-PRECONDITIONS
- **WHEN** the driver launches the ThinkingCap-27B `llama-server`
- **THEN** it SHALL verify (and record in `preconditions_checked`) that `llama_supports_gpu_offload()==True`, the exact resolved `.gguf` path, that the server was launched with `-fit off` + `n_ctx=22000` + `q8_0` KV on `CARNOT_ARC_GENERATOR_CUDA_GPU=1`, and that a `/think` `/completion` round-trip succeeded -- writing `honest_verdict` `complete_blocked_*` and exiting WITHOUT fabricating numbers if any precondition fails.

### REQ-ARC-WMTE-5725: ThinkingCap-Qwen3.6-27B Genuine-Reasoning Retest Through Its Proper Chat Template (harness correction of REQ-ARC-WMTE-5724; ultimately targets REQ-ARC-WMTE-5714)

REQ-ARC-WMTE-5724 tried to answer whether a model RL-tuned for token-efficient
reasoning (`bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF`, Q4_K_M) completes genuine `/think`
induction within the 8192-token budget that vanilla Qwen3.5-9B-MTP overran on `0/N`
(REQ-ARC-WMTE-5714's finding), but its result was **INCONCLUSIVE for a harness reason,
not a model reason.** The exp5714 `_induce_no_fence` mechanism POSTs a hand-built prompt
string to llama.cpp's RAW `/completion` endpoint with NO chat template applied (deliberately,
so a `/think` prefix reasons before emitting the code fence). That works for Qwen3.5-9B, but
Qwen3.6-family models (ThinkingCap-27B) REQUIRE their embedded chat template's turn structure
(system/user/assistant delimiters) to know a turn has started. Without it, ThinkingCap emitted
an immediate end-of-sequence with ~0 output on `10/12` reason cells (`stop_type=='eos'`, no
`<think>` trace) -- those cells never genuinely tested the budget hypothesis. The 2 genuine
attempts (`0/2` completed) and the healthy `12/12` codeonly frozen control proved the
model/server/GPU path itself is fine; only the raw-completion-without-template combination was
broken for this model family. REQ-ARC-WMTE-5724's own recommended follow-up was to re-test
through the proper Qwen3.6 chat template before drawing any budget-wall conclusion. This REQ
is that retest.

**The fix.** A new opt-in `LocalGGUFProposer.use_chat_template` field routes `generate()` and
`complete_text()` through the OpenAI-compatible `/v1/chat/completions` endpoint (a single user
turn) instead of raw `/completion`, so `llama-server` applies the GGUF's OWN embedded Qwen3.6
chat template automatically -- no external HuggingFace template needed (the `unsloth`/`bottlecap`
`-GGUF` repos ship no tokenizer files, but the template lives in the `.gguf` metadata). The
OpenAI-shaped reply is normalized back into llama.cpp's native `{content, stop_type, truncated}`
shape (`finish_reason=='length'` -> `stop_type='limit'`; a split-out `reasoning_content` is
folded back into `content` wrapped in `<think>` tags), so `_record_completion_diagnostics`,
`reason_engaged` detection, `max_raw_completion_len`, and every existing caller work unchanged.
The field DEFAULTS False, keeping the FROZEN live-generator path (Qwen3.5-9B raw `/completion`)
byte-identical -- this is a decentralization/frozen-stack-safe additive change, not a live-stack
edit.

**Genuine-reasoning semantics preserved (the load-bearing care point).** The retest must not
let the chat template's own scaffolding silently re-suppress the `<think>` trace the way the
codeonly directive originally did (the exact failure REQ-ARC-WMTE-5714's B2 arm was built to
avoid). The induce stays codeonly OFF with NO pre-opened ```` ```python ```` fence and the
`/think` soft-switch in the user turn, so reasoning genuinely engages and the model emits its
```` ```python ```` block afterward. This was **smoke-tested BEFORE the full run**: 4/4
induction-shaped `/v1/chat/completions` calls engaged a real `<think>` trace with non-empty
output (vs the `10/12` immediate-EOS the raw path produced in 5724), confirming the template
supplies the assistant-turn structure reliably (not just once) before any budget-wall numbers
were trusted.

**Design -- byte-identical to REQ-ARC-WMTE-5724 EXCEPT the chat-template route.** The
ThinkingCap reason cell reuses `exp5724.run_reason_cell` VERBATIM (imported, not copied):
codeonly OFF, `no_think_prefix=/think`, `max_tokens=8192`, `tries=1`, `_induce_no_fence`, the
exp5722 stale-engine unlink guard, and the full REQ-ARC-WMTE-5720 actions-to-progress ladder
(`load_engine` -> `plan_in_model` -> execute -> `WorldModelVerifier`/`score_goal_predicate_consistency`).
The ONLY change is the proposer's `use_chat_template=True`. The two BASELINE arms are NOT
re-measured -- they were already healthy in REQ-ARC-WMTE-5724 on the raw path each tolerates:
the Qwen3.5-9B raw-`/think` reason baseline (`1/12` `induce_ok`) and the ThinkingCap codeonly
frozen control (`12/12`, proving model health) are CITED from 5724's artifact and their per-run
rows folded in (tagged `reused_from=REQ-ARC-WMTE-5724`) so the paired-by-game comparison against
Qwen is real. This keeps compute on the one arm the harness bug actually corrupted.

**Substrate / provenance (unchanged from REQ-ARC-WMTE-5724).**
`inference_substrate=live_llm_inference` -- ThinkingCap-27B really induces on a CUDA
`llama-server` on GPU 1 (`CARNOT_ARC_GENERATOR_CUDA_GPU=1`; GPU-offload verified
`llama_supports_gpu_offload()==True` + a VRAM jump on load + a `/think` chat-completions
round-trip). Server config `n_ctx=22000`, `-fit off`, `q8_0` KV, MTP off, distinct port
(8959) so a stale 5724 server is never reused. `solve_provenance=development_proxy` on PUBLIC
games; `read_game_source=False`, `used_env_source=True`, `verifier_is_oracle=False`. The
24GB-3090 dev card is NOT Kaggle-representative (the eval GPU is ~16GB and ThinkingCap-27B Q4
is a tight fit there); this tests the CONTENT question only and NEVER flips the frozen live
default (operator-only) and NEVER submits.

**Question / primary metric.** On the identical 6-game roster (`ls20 tr87 lp85 g50t m0r0
ft09`), 2 trials/game, does token-efficient ThinkingCap-27B -- now genuinely reasoning via its
proper chat template -- COMPLETE genuine-reasoning inductions within the 8192-token budget more
often than vanilla Qwen's `~0/N`? PRIMARY metric = induce COMPLETION RATE (`n_induce_ok /
n_cells`) for the freshly-measured chat-template ThinkingCap reason arm vs Qwen's cited baseline,
annotated with `reason_engaged` (confirming the template fixed the immediate-EOS degeneracy) and
the overrun signal (`last_stop_type=='limit'` / `max_raw_completion_len`). The full signal ladder
is measured and paired by game as secondary color.

**Falsifiable gate / interpretation.** (a) The chat template must FIRST demonstrably fix the
degeneracy (most cells now genuine attempts, not immediate-EOS) -- otherwise the fix did not take
and the result is flagged for investigation rather than reported as a budget finding. (b) Given
genuine reasoning, token-efficient reasoning is a viable path to make `/think` usable ONLY if
ThinkingCap completes MATERIALLY more inductions within the same 8192 budget than Qwen's `~0/N`.
(c) If ThinkingCap STILL overruns on most/all cells even with the correct template, the
conclusion is "8192 tokens is simply too tight regardless of efficiency tuning" -- pointing
toward RAISING the budget (or splitting the induce) as the lever rather than swapping in a more
efficient model. All outcomes are equally reportable; a clean negative is stated plainly, never
spun. Small-N honesty guards apply (<=6 game pairs, stochastic proposer: report win/tie/loss +
exact sign-test p + outlier-fragility, never a bare mean; a positive is a DIRECTION to
investigate, not a significance claim). This REQ does not itself graduate any generator or budget
change to the live stack.

**Result artifact:** `results/experiment_5725_thinkingcap_chat_template_reason_ab.json`.

**Implementation Status:** Run 2026-07-18 (GPU 1, ThinkingCap-27B server VRAM 4->17176 MiB on
load, `/think` chat-completions smoke round-trip OK). Driver
`python/carnot/experiment_5725_thinkingcap_chat_template_reason_ab.py`; artifact
`results/experiment_5725_thinkingcap_chat_template_reason_ab.json` (adversarial-verify clean,
`summarize_artifact` clean, no adversarial flags). **The harness fix WORKED, and the result is a
modest, NON-SIGNIFICANT completion gain — not a solved budget wall.** Pre-run smoke: 4/4
induction-shaped `/v1/chat/completions` calls engaged a genuine `<think>` trace with non-empty
output (the raw `/completion` path gave 10/12 immediate-EOS in 5724). Full run: with the chat
template applied, ThinkingCap-27B genuinely reasoned on **12/12** cells (`reason_engaged`;
`n_degenerate_empty_eos=0` vs 5724's 10/12) — so REQ-ARC-WMTE-5724's inconclusive result is
resolved: it was a HARNESS artifact (raw completion without the Qwen3.6 turn structure), NOT a
model limitation. On the now-valid measurement, ThinkingCap completed **3/12** inductions within
the 8192 budget vs the cited Qwen3.5-9B baseline's **1/12** (paired by game: +0.167 mean delta,
3 wins / 2 ties / 1 loss over 6 game pairs, **sign-test p=0.625 — NOT significant**,
`outlier_fragile=False`), and overran fewer cells (9/12 vs Qwen 12/12; delta -0.25, p=0.25). BUT
the completions did NOT translate to progress: **0/12 level-ups** (both arms 0) and only **1/12**
`plan_found` (tr87 trial 1, `outlier_fragile=True`). The ThinkingCap frozen control (12/12) and
Qwen reason baseline (1/12) are CITED from REQ-ARC-WMTE-5724 (rows tagged
`reused_from=REQ-ARC-WMTE-5724`), not re-measured. **Honest read:** token-efficient reasoning
does clear the 8192 budget somewhat more often than vanilla Qwen and overruns less, but the effect
is small, not statistically significant at n=6, still leaves the majority (9/12) of cells
overrunning, and yields no downstream solve progress — a real DIRECTION, not evidence sufficient
to prefer a more-efficient model over the simpler "raise the budget / split the induce" lever.
Does NOT flip the frozen live default (operator-only); a real-VRAM/latency feasibility check on
the ~16GB Kaggle eval GPU (27B Q4 is a tight fit) remains pending before any deployment
consideration.

#### SCENARIO-ARC-WMTE-5725-CHAT-TEMPLATE-ROUTE-IS-THE-ONLY-CHANGE
- **WHEN** `experiment_5725_thinkingcap_chat_template_reason_ab.py` runs a ThinkingCap-27B genuine-reasoning `(game, trial)` cell
- **THEN** it SHALL reuse `exp5724.run_reason_cell` verbatim (codeonly OFF, `no_think_prefix=/think`, `max_tokens=8192`, `tries=1`, `_induce_no_fence`, NO pre-opened `python` fence, the exp5722 stale-engine unlink guard, and the REQ-ARC-WMTE-5720 ladder), the ONLY difference from REQ-ARC-WMTE-5724 being the proposer's `use_chat_template=True` which routes induce through `/v1/chat/completions` so the GGUF's embedded Qwen3.6 chat template applies -- and SHALL NOT re-measure the Qwen3.5-9B reason baseline or the ThinkingCap frozen control (both CITED from 5724 with per-run rows tagged `reused_from=REQ-ARC-WMTE-5724`).

#### SCENARIO-ARC-WMTE-5725-SMOKE-BEFORE-BUDGET-NUMBERS
- **WHEN** the driver (or its pre-run smoke) exercises the chat-completion path on ThinkingCap-27B
- **THEN** genuine `<think>` reasoning SHALL be confirmed to engage RELIABLY (multiple induction-shaped calls, not just one) with non-empty output BEFORE any full-run completion-rate numbers are trusted, and the artifact SHALL record (in `measurement_validity`) both the smoke result and whether the chat template fixed the immediate-EOS degeneracy (`n_degenerate_empty_eos <= n_genuine_attempt`) -- so a still-degenerate result is flagged for investigation, never silently reported as a false-negative budget finding.

#### SCENARIO-ARC-WMTE-5725-NORMALIZED-DIAGNOSTICS-AND-FROZEN-STACK-SAFETY
- **WHEN** `LocalGGUFProposer` runs with `use_chat_template=True`
- **THEN** the `/v1/chat/completions` reply SHALL be normalized back into llama.cpp's native `{content, stop_type, truncated}` shape (`finish_reason=='length'` -> `stop_type='limit'`; a split-out `reasoning_content` folded into `content` wrapped in `<think>` tags) so `_record_completion_diagnostics` / `reason_engaged` / `overran` / `max_raw_completion_len` stay faithful to everything the model emitted, AND the field SHALL default `False` so the frozen live-generator path (Qwen3.5-9B raw `/completion`) stays byte-identical (a decentralization/frozen-stack-safe additive change, never a live-stack edit).

### REQ-ARC-WMTE-5727: Perception/Representation Adequacy for Action-Effect (Dynamics) Prediction -- the Frame Representation Is at Chance; the Above-Chance Number Is a Trivial Action Base Rate (re-tests the "order-1 at chance" finding on the dynamics half of induction)

**Motivation.** A full session of generator-side experiments (reasoning depth,
retrieval augmentation, three generator sizes/architectures for the induction step
-- REQ-ARC-WMTE-5714/5720/5721/5722/5723/5724/5725 and siblings) produced ZERO real
level-ups. That convergence raises a diagnostic question upstream of generation: is
the world-model induction task being posed on top of a frame representation that
does not carry the signal induction needs -- in which case no generator swap could
help, and the lever is a representation upgrade, not a generator? This REQ tests one
concrete, load-bearing half of that question.

**Prior context this extends.** The standing project finding
(`project_arc_live_agent_learning_gaps` memory; exp4545
`experiment_4545_cross_game_discrimination_v3`, REQ-LEARN-4476) is that FRAME-ONLY
ORDER-1 features test at CHANCE in leave-one-GAME-out (LOO-AUROC ~0.49), while the
FULL live featurizer (`cross_game_features_v3`) clears chance (~0.67). But exp4545
measured that only on the VALUE/GOAL half of induction: POS = states ON a winning
path, NEG = states one wrong move OFF it. It never measured the DYNAMICS half --
"given this state and this action, will the action change the frame?" -- which the
live perception layer (`arc_frame_change_predictor`, `ActionEffectExpansionPrior`)
exists to answer and which dynamics induction depends on.

**Method (exp5727, deliberately identical to exp4545's LOO methodology so the two
halves are directly comparable).** Replay all banked solve trajectories on the
offline `Arcade` env (18 banked games; 14 usable after dropping single-class games).
At sampled on-path states, try candidate actions {gold} u {ACTION1..5}, step the REAL
env, and label `y = 1` iff the raw grid changed (`count_nonzero(before != after) > 0`
-- the SAME ground-truth definition the live `GroundTruthValidatedFrameChangeScorer`
uses). Feature = `cross_game_features_v3(before, previous_frame, action_id)` -- never
the after-frame, so no label leakage. Ablate the SAME feature classes as exp4545 and
train the SAME `DiscriminativeVerifier` with leave-one-GAME-out AUROC + bootstrap CI.
Substrate `verifier_ensemble_against_cached_candidates` (offline env-stepping + a
linear classifier; no LLM, no GPU; 32.3s).

**Result (honest null on perception).** The full-vector headline LOO-AUROC is 0.844
(CI [0.704, 0.940], excludes chance) -- but an adversarial control shows this is NOT
the frame representation carrying signal:

| representation | LOO-AUROC |
|---|---|
| v2 (frame-only order-1) | **0.536** (chance; in-sample 0.797, so the harness works) |
| v2 + frame_delta | 0.490 |
| v2 + object_relational | 0.529 |
| v2 + predicate_distance | 0.536 |
| v2 + action_conditioned | 0.837 |
| v3_full | 0.844 |
| **action-id ALONE (control)** | **0.883** |

The entire lift comes from the `action_conditioned` family, which is a bare one-hot
of the action id. Action-id ALONE (no frame features at all) scores 0.883 -- HIGHER
than v3_full -- and adding frame features to the action id changes LOO by -0.039 (it
slightly hurts). The mechanism is a global per-action base rate that transfers across
games because action semantics are shared: ACTION5 changes the frame only 15.8% of
the time, ACTION6 (click) 100%, directional ACTION1-4 ~72%. That base rate is trivial
prior knowledge every agent already has; it is not perception.

**The precise finding.** On BOTH halves of induction the FRAME-ONLY ORDER-1
representation sits at chance cross-game (exp4545 value target 0.494; exp5727 dynamics
target 0.536 -- a fresh-corpus, fresh-target REPLICATION of the standing finding). On
the value half the richer v3 frame features (frame_delta, object_relational) DO lift
transfer to 0.674. On the dynamics half NO frame feature family lifts it: the current
representation cannot predict, from the frame, WHICH state-specific actions are
effective (which click target is live vs dead, which move direction is wall-blocked)
beyond the action's global base rate. This is structural: `cross_game_features_v3`'s
action family is a bare action-id one-hot with NO action x frame interaction term, so
it literally cannot express state-grounded action-effect. That missing interaction
feature -- exactly what the `SmallFrameChangeCNN` in `arc_frame_change_predictor`
computes -- is the representational gap; and per the 2026-07-13 perception-grounding
audit that CNN term is silently zero on the live default frontier-priority path (a
dict-shape bug) and weight 0.05 even when alive.

**Recommendation (diagnose-and-recommend only; no frozen default flipped).** (1) Do
NOT conclude "perception is the wall, the generator work was wasted" -- that is too
strong; the richer v3 features DO carry cross-game VALUE signal (exp4545, 0.674), so
the value representation is adequate yet the offline->live bridge still fails (the
unresolved 2026-06-23 issue). (2) Do NOT "add more global frame features" -- exp4545
showed that is done for value and exp5727 shows it cannot help dynamics. (3) The
actionable dynamics lever is a genuine action x frame-LOCALIZED perception feature
(clickability of the click target; blocked-ness of the move direction) -- which the
frame-change CNN already computes but is under-wired (fix its live frontier-priority
call site + weight) rather than a generator swap or another global featurizer.

#### SCENARIO-ARC-WMTE-5727-GATE
- **WHEN** the exp5727 harness scores `cross_game_features_v3` on the action-effect (grid-change) target with leave-one-GAME-out AUROC over the banked-game corpus
- **THEN** it SHALL report the per-feature-class LOO-AUROC table AND the action-id-alone control, and SHALL classify the result as a frame-representation signal ONLY IF the frame features beat the action-id-alone control by >= 0.05 AUROC (else the honest verdict is `action_effect_above_chance_but_driven_by_action_base_rate_not_frame_representation_honest_null_on_perception`), so an above-chance headline driven purely by the action base rate is never reported as evidence the representation is adequate.

#### SCENARIO-ARC-WMTE-5727-POSITIVE-CONTROL-AND-NO-LEAKAGE
- **WHEN** the harness computes the LOO null for the frame-only (v2) representation
- **THEN** the in-sample (train==test) AUROC SHALL be reported as a positive control and SHALL exceed 0.5 (measured 0.797) so the null is informative rather than a broken harness, AND every feature vector SHALL be computed from `(before_frame, real_previous_frame, action_id)` only -- never the after-frame -- so the ground-truth grid-change label cannot leak into the features.

#### SCENARIO-ARC-WMTE-5727-LIVE-PATH-FIDELITY
- **WHEN** the "current representation" under test is chosen
- **THEN** it SHALL be the live-path featurizer `cross_game_features_v3` (the one `arc_competition_agent.py`'s value head loads; its shipped subset is `v2_plus_frame_delta`), NOT a hypothetical richer representation, per the ARC Live-Path Reachability Discipline -- so the finding describes what the shipped agent can actually encode, and the recommended lever (an action x frame-localized feature such as the `SmallFrameChangeCNN` term) names a real live-path module rather than a strawman.

### REQ-ARC-WMTE-5726: ThinkingCap-Qwen3.6-27B 16384-Budget, Larger-N, Dual-GPU Retest (properly-scaled extension of REQ-ARC-WMTE-5725)

REQ-ARC-WMTE-5725 established (via the chat-template fix) that ThinkingCap-27B genuinely
reasons and completed `3/12` genuine-reasoning inductions within the 8192-token budget vs
vanilla Qwen3.5-9B-MTP's `1/12` -- a **modest, NON-significant** edge at only `n=6` game pairs,
with `9/12` ThinkingCap cells still OVERRUNNING the budget and `0/12` level-ups. Three
operator-requested changes properly scale that test:

1. **Doubled completion budget: 8192 -> 16384 tokens.** The 5725 failure mode was OVERRUN
   (the model reasoned past 8192 without emitting the required `engine`+`is_level_complete`
   code). This REQ tests whether a 2x budget lets materially more cells finish.
2. **Larger N: 6 -> the maximum honestly-usable game set.** Of the 25 public games in
   `ops/arc_solve_registry.yaml`, exactly **20** supply a valid seeded level-up window via
   `build_progress_window` (5 fail: `wa30`/`sc25`/`tn36` have no `hand_verifier` adapter,
   `ka59` hits a coordinate-parse `ValueError`, `dc22` returns `None`). **N>=30 UNIQUE games
   is MATHEMATICALLY UNREACHABLE** (only 25 public games exist) -- disclosed, not silently
   under-reported. Both generators are re-measured FRESH at 16384 (the 5725 8192 Qwen baseline
   is NOT reused; the budget change makes it an unfair comparator).
3. **Dual-GPU llama-server topology for the 16GB ThinkingCap-27B, empirically selected.** A
   pre-run smoke test (coherence + tok/s) compares single-GPU vs a dual-GPU FFN+MTP-draft split
   (FFN tensors `blk.N.ffn_{down,gate,up}` -> CUDA0 via `-ot ffn=CUDA0`, main attention + SSM +
   KV cache -> CUDA1 via `-ts 0,1`, the nextn MTP self-draft -> CUDA0 via `-otd nextn=CUDA0` when
   MTP on). This requires STOPPING the autonomous conductor to free GPU 0 (operator-authorized).
   The winning topology is picked by measurement, never assumed.

**Design -- generator is the ONLY scientific variable.** The induce cell is the
`exp5724.run_reason_cell` mechanism (codeonly OFF, `/think`, no pre-opened fence, `tries=1`, the
exp5722 stale-engine unlink guard, the REQ-ARC-WMTE-5720 actions-to-progress ladder) with ONLY
the token budget changed 8192->16384. ThinkingCap routes through `/v1/chat/completions`
(`use_chat_template=True`, the 5725 fix); Qwen3.5-9B routes through the raw `/completion` path it
tolerates (the frozen live-generator path). This per-model routing asymmetry is intrinsic (each
model uses the path it needs) and unchanged from 5724/5725. Servers run SEQUENTIALLY (one torn
down before the next) so the two models never contend and never race on the shared
`results/arc_e3/<game>/world_model.py`.

**Substrate / provenance.** `inference_substrate=live_llm_inference` on 2x RTX 3090.
`solve_provenance=development_proxy` on PUBLIC games; `read_game_source=False`,
`used_env_source=True`, `verifier_is_oracle=False`. The dual-GPU topology is a DEV-RIG
approximation of Kaggle L4x4, NOT the submission-kernel config (a separate operator decision).
NEVER flips the frozen live default (operator-only) and NEVER submits.

**Question / primary metric.** At the doubled 16384 budget and larger N, does ThinkingCap-27B
(genuine `/think` via its chat template) COMPLETE the seeded induction more often than vanilla
Qwen3.5-9B-MTP, and is any edge statistically distinguishable? PRIMARY metric = induce COMPLETION
RATE (`n_induce_ok / n_cells`); PRIMARY test = **Wilcoxon signed-rank on the per-game deltas** (more
powered than the binary sign test at this N), with the sign test + outlier-fragility reported
alongside. Real LEVEL-UP count is the decisive downstream signal, reported regardless of the proxy
metrics.

**Falsifiable gate / interpretation.** (a) The dual-GPU topology must FIRST smoke-test COHERENT
(a cross-GPU split that loads can still silently corrupt output) AND its speed measured honestly
(single-GPU vs split, tok/s) BEFORE the full run -- the fastest COHERENT config is deployed. (b)
The completion-rate result is reported with its Wilcoxon p; a positive delta at this N is a
DIRECTION, not a significance claim, unless the test says so. (c) A clean negative (still
small/non-significant even at larger N, or the dual-GPU split adds overhead rather than speeding
up) is exactly as reportable as a positive. This REQ does not graduate any generator, budget, or
GPU-topology change to the live stack.

**Result artifact:** `results/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.json`.

**Implementation Status:** Run 2026-07-19 on 2x RTX 3090. Driver
`python/carnot/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.py`; artifact
`results/experiment_5726_thinkingcap_16k_dualgpu_reason_ab.json` (adversarial-verify clean,
`summarize_artifact` clean, 0 flags). **Deployed config: single-GPU, MTP off** (ThinkingCap on
GPU0, VRAM 17586 MiB; Qwen on GPU1, VRAM 12186 MiB; budget 16384, n_ctx 32768, q8 KV), N=20 (the
full usable ceiling) x 2 trials = 80 cells. **The completion advantage is now LARGE and
STATISTICALLY SIGNIFICANT at proper N, but it does NOT translate to progress.** ThinkingCap
genuinely reasoned on 40/40 cells and completed 31/40 (77.5%) genuine-reasoning inductions within
the 16384 budget vs vanilla Qwen3.5-9B-MTP's 6/40 (15%) -- paired by game (N=20): +0.625 mean
delta, 16 wins / 2 ties / 2 losses, sign-test p=0.0013, **Wilcoxon signed-rank p=0.0005
(two-sided), p=0.0002 (greater)** -- a well-powered win, resolving 5725's non-significant n=6 edge
(there ThinkingCap 3/12=25% vs Qwen 1/12; here 77.5% vs 15%, so the doubled budget roughly TRIPLED
ThinkingCap's completion rate). ThinkingCap overran only 9/40 cells vs Qwen's 37/40.
**BUT: ZERO level-ups in BOTH arms across all 20 game pairs (reached_levelup all-ties at 0), and
plan_found only 9/40 (ThinkingCap) / 0/40 (Qwen).** Honest read: at the doubled budget the
completion wall is decisively cleared for the token-efficient model, but completion is NOT the
level-up bottleneck -- 31 completed inductions across 20 distinct games still produced no game
progress, consistent with the entire REQ-ARC-WMTE-5714->5725 lineage (the wall is downstream, in
plan generation/execution quality, not induce completion). Per the operator's stated priority (a
single genuine real level-up matters more than a significant proxy-metric delta), the DECISIVE
signal remains NULL. **Dual-GPU topology finding (empirical, honest negative):** the FFN-split
(FFN tensors -> CUDA0 via `-ot ffn=CUDA0`, attention+SSM+KV -> CUDA1 via `-ts 0,1`) smoke-tested
COHERENT (VRAM split cleanly GPU0=10476/GPU1=7540 MiB, real reasoning+code) but ran ~24% SLOWER
than single-GPU (30.2 vs 39.5 tok/s) -- a 16GB Q4 model already fits one 24GB card, so intra-model
tensor split only adds cross-GPU PCIe + sequential-GPU-execution overhead; no speed benefit for
this model size. MTP self-draft is INFEASIBLE on the 27B in this llama.cpp build (b9606): it loads
`--model-draft` as a full second 16GB copy (2x16=32GB > 24GB -> OOM), even though the GGUF ships
nextn heads; Qwen3.5-9B's MTP works only because 2x5.5GB fits. So the run deployed the fastest
COHERENT config (single-GPU). This is a DEV-RIG approximation of Kaggle L4x4, NOT the
submission-kernel config; it does NOT flip the frozen live default (operator-only) and NEVER
submits. **Recommendation:** do NOT adopt ThinkingCap-27B into the live stack on the strength of
the completion win -- the completion advantage, though large and significant, buys no level-up;
the lever that matters (plan generation/execution) is untouched. Keep single-GPU serving for a
16GB-class model. A real-VRAM/latency feasibility check on the ~16GB Kaggle eval GPU remains a
prerequisite before any deployment consideration of the 27B.

#### SCENARIO-ARC-WMTE-5726-GENERATOR-IS-THE-ONLY-VARIABLE-AT-16384
- **WHEN** `experiment_5726_thinkingcap_16k_dualgpu_reason_ab.py` runs a genuine-reasoning `(generator, game, trial)` cell
- **THEN** it SHALL use the `exp5724.run_reason_cell` induce mechanism with ONLY the token budget changed (8192 -> 16384), re-measure BOTH generators FRESH at 16384 (NOT reusing 5725's 8192 numbers), hold harness/planner/verifier constant so ONLY the generator LLM (and its intrinsic chat-template-vs-raw routing) varies, and run the two servers SEQUENTIALLY so they never contend or race on `results/arc_e3/<game>/world_model.py`.

#### SCENARIO-ARC-WMTE-5726-DUAL-GPU-TOPOLOGY-SMOKE-BEFORE-FULL-RUN
- **WHEN** the dual-GPU ThinkingCap topology is a candidate for the full run
- **THEN** a pre-run smoke test SHALL confirm OUTPUT COHERENCE (real reasoning + real code, not garbled cross-GPU-corrupted output) AND measure per-config tok/s (single-GPU vs the FFN+MTP-draft split) BEFORE any full-run numbers are trusted, the fastest COHERENT config SHALL be deployed, and the artifact SHALL record every tested config's coherence + speed in `dual_gpu_topology_smoke` -- reporting honestly whether the split sped the run up or added overhead.

#### SCENARIO-ARC-WMTE-5726-SAMPLE-SIZE-CEILING-DISCLOSED
- **WHEN** the artifact reports the completion-rate comparison
- **THEN** it SHALL disclose that N>=30 UNIQUE games is mathematically unreachable (only 25 public games exist, of which 20 supply a valid seeded window), state the exact N used and whether it was further bounded by conductor downtime (a downtime bound, NOT a fresh statistical ceiling), use the Wilcoxon signed-rank on per-game deltas as the primary paired test (with the sign test + fragility alongside), and disclose that trials-per-game add per-game stability but are NOT independent game-level degrees of freedom.

### REQ-ARC-WMTE-5760: CEGIS Refinement-Loop Induction A/B (ThinkingCap-27B + Qwen3.5-9B)

`docs/research-notes/arc-induction-quality-improvement-design-2026-07-20.md` found the live agent's
CEGIS-style refinement loop (`execute_bounded_llm_reinduction`, `arc_llm_reinduction.py:654`, wired
into BOTH live call sites `arc_competition_agent.py:3885,4005`) was never exercised by
REQ-ARC-WMTE-5726's single-shot diagnosis, which hardcodes `n_refinement_rounds=0`
(`arc_actions_to_progress.py:764`) -- so the 29/37-near-zero `heldout_accuracy` finding is a
SINGLE-SHOT number, and whether the live agent's actual repair loop fixes it was unmeasured.

`python/carnot/experiment_5760_cegis_refinement_induction_ab.py` routes induction through
`execute_bounded_llm_reinduction` (`min_heldout_accuracy=1.0` so refactor rounds actually fire,
`candidate_provider`/`load_engine`/`plan_in_model` from the live modules) instead of single-shot
`proposer.induce()`, capturing the FULL per-round `heldout_accuracy` trajectory (round 0 = baseline,
rounds 1-2 = counterexample-guided refactor) on BOTH ThinkingCap-Qwen3.6-27B and Qwen3.5-9B-MTP, a
13-game pre-registered roster (12 fastest-to-induce of the 17 REQ-ARC-WMTE-5726 games + `ft09`, fixed
before the run), 3 trials/game. Primary metric `delta_heldout = heldout(best refined round) -
heldout(round 0)`, pooled + per-game, bootstrap 95% CI. Also tracks the structural window-memorization
detector (before/after refinement) and the refactor-round code-emission rate (attribution guard against
an emission-budget confound). Three pre-registered branches (POSITIVE / HONEST-NEGATIVE /
EMISSION-CONFOUND) per the design doc's exact thresholds; a `degradation_guard` on the two games already
at round-0 `heldout=1.0` (`sp80`, `ft09`) requires refinement NOT to corrupt an already-correct model.

**RESOLUTION (2026-07-21).** Ran the full pre-registered roster (78 cells, both models, GPU 0, launched
fully detached per the established pattern). **Pooled `delta_heldout = -0.0128`** (bootstrap CI
`[-0.032, 0.0]`, includes 0), `positive_game_frac = 0.0`, `wins = 0` across all 13 games -- ZERO games
improved on EITHER model. `degradation_guard` HOLDS (no corruption on `sp80`/`ft09`). BUT the result does
NOT cleanly satisfy the pre-registered HONEST-NEGATIVE branch, for two disclosed, real reasons: (1)
ThinkingCap-27B's window-memorization rate dropped substantially (`0.7297 -> 0.3243`, a genuine ~40-point
swing) even though `heldout_accuracy` did not improve -- refinement measurably changes the induced code's
SURFACE structure (stops hardcoding literal window coordinates) without fixing its DYNAMICS correctness;
(2) Qwen-9B's refactor-round emission rate (`0.55`) fell just under the `0.6` healthy threshold (vs
ThinkingCap's clean `0.9863`), so Qwen's own null is partially confounded by budget/emission, while
ThinkingCap's is cleanly attributable. `gate_branch: partial_inconclusive` -- the artifact reports this
honestly rather than force-fitting a pre-registered label the data does not cleanly support. Verdict:
`complete_cegis_refinement_partial_pooled_delta_-0.0128_..._does_not_cleanly_meet_a_preregistered_branch`.
adversarial_verify: 0 flags. Independently re-verified (bootstrap CI, per-model breakdown, memorization
detector, emission attribution all cross-checked against the artifact's raw fields).

Required field principles:

- `primary_result`: principle "pooled delta_heldout + per-game + per-model breakdown + bootstrap CI is the load-bearing quantity for whether refinement helps; reporting only a point estimate would hide whether 0 is inside or outside the confidence interval."
- `memorization_detector`: principle "a before/after DELTA on window-coordinate-literal counts, gated to cancel the dense-window chance-match confound -- distinguishes 'refinement changed surface code structure' from 'refinement fixed dynamics correctness', which the primary delta_heldout metric alone cannot separate."
- `emission_attribution`: principle "per-model refactor-round code-emission rate lets a reader distinguish a genuine no-improvement null from a budget-overrun mechanical artifact -- without it, a low-emission model's null would be indistinguishable from a real negative result."
- `degradation_guard`: principle "an already-correct model (round0 heldout=1.0) being CORRUPTED by refinement would be the single worst possible outcome (actively harmful, not merely unhelpful) -- this field makes that check explicit and auditable rather than buried in an aggregate mean."

#### SCENARIO-ARC-WMTE-5760-WITHIN-LOOP-DELTA-IS-THE-FAIR-COMPARISON
- **WHEN** the artifact reports whether CEGIS refinement improved induction quality
- **THEN** the PRIMARY metric SHALL be the WITHIN-LOOP delta (round 0 vs best-refined round, same induction call, same model, same game) rather than a cross-experiment comparison against a differently-invoked single-shot baseline, so the result isolates the refinement mechanism's own effect and is not confounded by an unrelated methodology difference in how round 0 itself was invoked.

#### SCENARIO-ARC-WMTE-5760-MEMORIZATION-CHANGE-DOES-NOT-IMPLY-CORRECTNESS-CHANGE
- **WHEN** the window-memorization detector shows a substantial before/after drop for a model whose `delta_heldout` is flat or negative
- **THEN** the artifact SHALL report both quantities separately and SHALL NOT infer or claim a correctness improvement from a memorization-rate change alone -- refinement measurably altering the induced code's surface structure (removing hardcoded literals) is a DISTINCT finding from whether the code's DYNAMICS predictions became more accurate, and conflating the two would overstate what refinement achieved.

#### SCENARIO-ARC-WMTE-5760-DEGRADATION-GUARD-BLOCKS-A-FALSE-POSITIVE-RECOMMENDATION
- **WHEN** a game already at round-0 `heldout_accuracy=1.0` (a genuinely correct induced model) goes through refactor rounds
- **THEN** the artifact SHALL explicitly check and report whether the refined heldout ever drops below the round-0 value on that game, and a violation SHALL block any POSITIVE gate_branch recommendation regardless of the pooled delta, since a mechanism that improves the average while corrupting an already-working model is not safe to recommend for the live path.

### REQ-ARC-WMTE-5766: CEGIS Refinement-Loop Induction A/B (gemma-4-31B-it, GPU1, parallel to REQ-ARC-WMTE-5760)

REQ-ARC-WMTE-5764 (single-shot induction, same 13-game roster) found gemma-4-31B-it -- a genuinely
different, dense, more-per-token-compute model family than ThinkingCap-27B or the 35B-A3B MoE --
substantially outperforms ThinkingCap-27B's single-shot `heldout_accuracy` (pooled `0.378487` vs
`0.187604`, 12/13 nonzero games vs 6/13; both numbers independently re-verified against the raw
39-row shard). This REQ completes the missing cell of the model x mechanism 2x2 design (operator
directive: "swap models and try both again at the same time to see if either result changes"): does
verifier-grounded CEGIS refinement COMPOUND gemma's already-better single-shot starting point, or was
that edge already near a ceiling refinement cannot improve on? Ran in PARALLEL with REQ-ARC-WMTE-5760
on GPU 1 (`CARNOT_5726_QW_CUDA=0` set on REQ-ARC-WMTE-5760's restart specifically to keep both arms of
that job on GPU 0, permanently eliminating a real, identified GPU-1 contention risk between the two
concurrent jobs rather than racing a timing estimate).

`python/carnot/experiment_5766_gemma31b_cegis_refinement_ab.py` reuses REQ-ARC-WMTE-5760's exact CEGIS
mechanism and gate thresholds, substituting gemma-4-31B-it-GGUF as the model, on the IDENTICAL
13-game/3-trial roster (imported from the sibling module to guarantee an exact match).

**RESOLUTION (2026-07-21).** Ran the full pre-registered roster (39 cells, GPU 1, launched fully
detached, ran independently of REQ-ARC-WMTE-5760's concurrent GPU-0 job with zero contention).
**Pooled `delta_heldout = -0.0598`** (bootstrap CI `[-0.145, 0.0]`, includes 0), `positive_game_frac =
0.0`, `wins = 0` across all 13 games -- ZERO games improved. `degradation_guard` HOLDS. Emission rate
`0.9872` (healthy). Memorization rate dropped `0.2564` absolute. Same `gate_branch:
partial_inconclusive` outcome as REQ-ARC-WMTE-5760, for the analogous reason (memorization changed
substantially without a matching correctness gain). **Cross-experiment comparison caveat, disclosed
explicitly and NOT to be read at face value:** the CEGIS loop's own round-0 pooled heldout
(`0.059829`) is far below REQ-ARC-WMTE-5764's standalone single-shot pooled heldout (`0.378487`) on
the SAME roster -- but this gap is NOT clean evidence that refinement is destructive; the CEGIS loop's
round-0 call uses the live path's STANDARD `induce()` invocation, a disclosed, real methodological
divergence from REQ-ARC-WMTE-5764's dedicated `_induce_no_fence` single-shot script
(`methodology_note`, both artifacts). The FAIR, apples-to-apples quantity is the within-loop `-0.0598`
delta above (same round-0 invocation method used for both the "before" and "after" comparison), which
is directly comparable to REQ-ARC-WMTE-5760's `-0.0128` and shows the same qualitative result on a
substantially more capable model: refinement does not compound a strong single-shot starting point,
either. adversarial_verify: 0 flags. Independently re-verified.

Required field principles:

- `comparison_to_gemma31b_singleshot_baseline`: principle "reports the cross-experiment gap explicitly WITH its methodology caveat, so a reader cannot mistake a round-0-invocation difference for evidence that refinement is actively harmful relative to not refining at all -- the within-loop delta is the metric that isolates the refinement mechanism's own effect."
- `delta_heldout_by_game`: principle "identical schema and gate thresholds to REQ-ARC-WMTE-5760's `primary_result`, by design, so the two runs are directly comparable without re-deriving a matched methodology after the fact."

#### SCENARIO-ARC-WMTE-5766-BIGGER-MODEL-DOES-NOT-MAKE-REFINEMENT-WORK
- **WHEN** gemma-4-31B's own single-shot baseline (REQ-ARC-WMTE-5764) is substantially higher than ThinkingCap-27B's, and the SAME CEGIS mechanism is applied to gemma
- **THEN** the artifact SHALL report the within-loop `delta_heldout` (not merely the cross-experiment gap to the single-shot baseline) as the primary test of whether refinement compounds the stronger starting point, and a result matching REQ-ARC-WMTE-5760's qualitative outcome (zero games improved) on a model with a 2x higher single-shot baseline SHALL be read as evidence that the refinement mechanism's ineffectiveness is not explained by insufficient model capacity.

#### SCENARIO-ARC-WMTE-5766-PARALLEL-EXECUTION-GPU-ISOLATION
- **WHEN** REQ-ARC-WMTE-5760 and REQ-ARC-WMTE-5766 run concurrently on separate GPUs
- **THEN** each job's model-serving process SHALL be pinned to a single, exclusive GPU index for its ENTIRE run (including any later arm/model transition within the same job), verified via an explicit env-var override where the default GPU assignment would otherwise create a future collision, so neither job's GPU memory or compute contends with the other at any point in either run.

### REQ-ARC-FCP-5728: CNN-Blend-Weight Sweep, Matched-Budget A/B

`arc_frame_change_predictor.LiveActionEffectScorer` combines two signals
additively for live frontier-priority scoring: `memory_weight (=1.0) *
PersistentAEM.candidate_score(...)` plus `cnn_weight (=0.05) *
SmallFrameChangeCNN`-backed `FrameChangeScorer.candidate_score(...)`.
REQ-ARC-FCP-5590 fixed a dict-candidate crash that silently zeroed the CNN
term, then found a clean null (byte-identical control/treatment across the
11-game roster) -- the fix alone did not change search behavior. The
hypothesis this REQ tests: is that null explained by `cnn_weight` being 20x
smaller than `memory_weight`, so even a now-correct CNN signal is too small
to ever flip a frontier tie-break?

`python/carnot/experiment_5728_cnn_weight_sweep.py` SHALL sweep `cnn_weight`
across at least 5 values (0.05 baseline, 0.25, 0.5, 1.0, 2.0 -- a 40x range
over the shipped default) with `memory_weight` held fixed at `1.0`, on the
SAME 11-game roster + `budget=200` + `CARNOT_ARC_DISABLE_INDUCTION=1`
no-LLM harness as exp5590, so results are directly comparable. For each
weight the artifact SHALL report `levels_gained_total`,
`states_expanded_total`, `per_game_levels_delta_vs_baseline`, and a
`cnn_calibration_summary` distinguishing "the CNN's raw outputs are
non-discriminating" from "the CNN's raw outputs vary but are never
consulted" -- specifically `ranking_scorer_nonzero_returns_total` (how many
times the wrapping `GroundTruthValidatedFrameChangeScorer.candidate_score`
returned nonzero, the ONLY moments `cnn_weight` can influence frontier
order) versus `cnn_raw_nonzero_total`/`baseline_cnn_raw_variance_mean_over_games`
(whether the underlying CNN forward pass itself produces varied,
non-degenerate output). A `prior_work_extended` block SHALL cite exp5590
by id + verdict + what is different here (the weight, not the bug).

**RESOLUTION (2026-07-19).** Ran cleanly on the full 11-game roster,
budget=200/game, 5 weights x 11 games. No tested weight (0.05..2.0) banks
more levels than the 0.05 baseline (`baseline_levels_total: 1`,
`best_weight: 0.05`, `any_weight_beats_baseline_levels: false`) --
`levels_gained_headroom_present: true` (interpretable null: 1 level, `lp85`
L1, in every arm). `cnn_calibration_summary` localized WHY: 7/11 games
(`cd82, cn04, lp85, m0r0, sk48, su15, wa30`) never reach `validated=True`
in `GroundTruthValidatedFrameChangeScorer` by end of run, so its
`candidate_score` returns `0.0` UPSTREAM of the `cnn_weight` multiply on
those games regardless of magnitude -- `n_games_gtv_never_validated: 7`. On
2 of those (`lp85`, `su15`) `ranking_scorer_nonzero_returns_total` was
`0` for the entire run: the blended scorer was consulted zero times, so no
weight could possibly have mattered. Meanwhile `baseline_cnn_raw_variance_
mean_over_games: 0.0115` confirms the CNN's raw forward-pass outputs ARE
non-trivially varied -- this is a GATE problem, not a dead/non-discriminating-
CNN problem. Verdict:
`complete: cnn_weight_sweep_headroom_present_weight_change_yields_same_levels`.
Recommendation (operator-only): do NOT change the live default `cnn_weight`
on a capability basis; the next lever is the `GroundTruthValidatedFrameChange
Scorer` validation gate itself (root-caused and tested in REQ-ARC-FCP-5729).

Required field principles:

- `weight_sweep_results`: principle "per-weight levels/states/efficiency -- levels is the capability answer; without it a weight change could look meaningful on a proxy metric while banking nothing real."
- `cnn_calibration_summary`: principle "distinguishes a non-discriminating CNN from a gated-off CNN -- conflating them would misdirect the next experiment at the wrong layer (retraining the model vs fixing the gate)."
- `any_weight_beats_baseline_levels`: principle "the load-bearing recommendation boolean -- a weight change only justifies touching the live default if it beats baseline on the decisive metric, not a proxy."

#### SCENARIO-ARC-FCP-5728-WEIGHT-INERT-WHEN-GATE-UNVALIDATED
- **WHEN** a game's `GroundTruthValidatedFrameChangeScorer` never reaches `validated=True` during a run (e.g. `lp85`, `su15`: zero nonzero ranking-scorer consults across the entire episode)
- **THEN** every tested `cnn_weight` value SHALL produce byte-identical `states_expanded` and `levels_gained` on that game, because the wrapping gate returns `0.0` before the weight multiply is ever applied -- the artifact SHALL report this as a gate-localization finding, not attribute the null to the CNN's own discriminative quality.

#### SCENARIO-ARC-FCP-5728-RAW-CNN-OUTPUT-VARIED-DESPITE-NULL
- **WHEN** the sweep's headline result shows no weight beats baseline levels
- **THEN** the artifact SHALL separately report the CNN's raw (pre-gate) output variance across the roster, so a reader cannot mistake "the weight sweep found no capability delta" for "the CNN produces degenerate/constant output" -- these are different failure modes requiring different fixes.

### REQ-ARC-FCP-5729: GroundTruth-Validation-Gate Tolerance Matched-Budget A/B

`arc_frame_change_predictor.GroundTruthValidatedFrameChangeScorer` gates its
`candidate_score` to return `0.0` whenever `not self.validated`, where
`validated` requires `agreement_count >= required_agreements (=1)` AND
`contradiction_count == 0` -- zero contradictions FOREVER, and its `reset()`
forwards to the inner scorer but never clears its OWN agreement/contradiction/
observed counters. Because the dominant `PersistentAEM` memory term has an
observed floor `>= 0.94` (exp5728 per-game `memory_term_stats.min`), the gate's
`predicted_changed` is nearly always True, so any ordinary no-op action (a
blocked move, an inert click -- normal ARC play) trips one contradiction and
PERMANENTLY disqualifies validation for the rest of the game run. This is
stricter than the gate's own originating spec (REQ-ARC-FCP-5373: "at least one
observed transition validates it"); the zero-tolerance-forever requirement is
an implementation overreach. exp5728's `cnn_weight` sweep localized this gate
(not the blend weight) as the next lever: 7/11 games never validated post-run,
so the scorer was returned `0.0` upstream of every downstream multiply and was
never exercised.

`python/carnot/experiment_5729_gtv_gate_fix_ab.py` SHALL run a matched-budget
A/B across the full 11-game roster (same roster + `budget=200` +
`CARNOT_ARC_DISABLE_INDUCTION=1` no-LLM harness as exp5590/exp5728,
single-threaded for determinism) comparing the CURRENT shipped gate (baseline
control) against at least: (a) a RATE-tolerance criterion
(`validated := observed_count >= min_observations AND agreement_count/
observed_count >= threshold`, threshold swept at 0.6/0.7/0.8, `min_observations`
justified), and (b) a RESET-ON-LEVEL-UP criterion (keep zero-tolerance but clear
the wrapper's own counters when `reset(reset_to_prior=True)` fires, the live
level-up path at `arc_competition_agent.py:1798`). Configs SHALL be injected by
in-place `__class__` reassignment onto the single shared gate object
(`policy.explorer.frame_change_scorer`, identity-preserving so the ranking,
frontier-priority, observe, and reset paths all see the change) WITHOUT editing
`arc_frame_change_predictor.py` -- this is a measurement task; whether to change
the live default is reported as an operator-only recommendation. The chosen
tolerance MUST still gate OFF a genuinely miscalibrated scorer (the gate is not
deleted wholesale). The artifact SHALL report `gate_config_results` (per-config
levels/states/efficiency + `n_games_validated_post_run`),
`n_games_validated_post_run_by_config`, `safety_regression_check` (per-config
`states_expanded` vs baseline, flagging any material blow-up),
`levels_gained_headroom_present`, `any_config_beats_baseline_levels`,
`any_config_beats_baseline_without_safety_regression`, and a
`prior_work_extended` block citing exp5590 AND exp5728.

**RESOLUTION (2026-07-19).** Ran cleanly on the full 11-game roster,
budget=200/game, 300.9s, 5 configs x 11 games. `levels_gained_headroom_present:
true` (1 level, `lp85` L1, reached in every config -- interpretable null, not a
zero-headroom test). The rate-tolerance gate WORKED MECHANICALLY: games-
validated-post-run rose from baseline's `3/11` to `9/11` (threshold 0.7/0.8) and
`10/11` (threshold 0.6), and ranking-scorer nonzero returns nearly tripled
(18,691 -> ~49-55k) -- the gate loosening turned the scorer ON as designed.
Protection was preserved: `sk48` (agreement rate 0.056) stayed gated OFF in
every config, and `cn04` (0.526) passed only at the most-permissive 0.6 -- a
genuinely miscalibrated scorer is still blocked. BUT no capability gain: every
config banked exactly `1` level (`any_config_beats_baseline_levels: false`). On
`lp85` the gate flipped to validated and the scorer was consulted ~27,000 times,
yet `states_expanded` (181), `levels` (1), and `actions_to_first_levelup` (11)
were IDENTICAL to baseline -- because the memory-dominated blended signal is
near-uniform across candidates (exp5728 `memory_term_stats.variance` ~= 3e-4 on
`lp85`), so admitting it does not reorder the search. No safety regression:
`states_expanded` DROPPED 6-8% under the rate configs (854-876 vs baseline 931,
`any_config_states_regression: false`), so the loosened gate did not make search
noisier. `reset_on_levelup` was byte-identical to baseline on all 11 games (only
`lp85` levels up, and it never reaches L2 within budget, so the single reset had
no 2nd-level window to act on -- the mechanism is correct but untestable-as-
beneficial on this roster). Verdict:
`complete: gtv_gate_loosening_turns_scorer_on_3_to_10_of_11_games_validated_but_no_level_gain_scorer_signal_is_the_blocker_not_the_gate`.
This confirms exp5728's retire condition: the gate WAS a real blocker (it kept
the scorer off), loosening it is SAFE and turns the scorer on, but the blocker
is now DOWNSTREAM -- the scorer's own discriminative quality (CNN/memory
perception), not the gate that admits it. Recommendation (operator-only): do NOT
change the live default gate on a capability basis (no level gain); the right
next lever per Missing-Verifier Gap Logging is the scorer's signal quality. A
gate-criterion change remains defensible on its own merits (it is safe and
correctly admits well-calibrated scorers) IF a future stronger scorer makes the
admitted signal load-bearing.

Required field principles:

- `gate_config_results`: principle "per-config levels/states/efficiency + n_games_validated_post_run -- n_games_validated is the mechanism this experiment directly manipulates, levels is the capability answer."
- `safety_regression_check`: principle "a loosened gate that lets a miscalibrated scorer steer search can make search WORSE/noisier even when it banks a level -- states_expanded vs baseline flags any blow-up so a level win is never reported without its search cost."
- `n_games_validated_post_run_by_config`: principle "the direct mechanism readout -- a gate change that does not move validation cannot have changed the search; one that moves validation a lot but banks no levels localizes the failure downstream of the gate."
- `any_config_beats_baseline_without_safety_regression`: principle "the load-bearing recommendation boolean -- a level win only justifies changing the live default if it does not come with a search-cost blow-up."

#### SCENARIO-ARC-FCP-5729-RATE-TOLERANCE-GATE
- **WHEN** the rate-tolerance gate config runs a game whose end-of-run agreement rate exceeds the threshold despite a nonzero contradiction count (e.g. `lp85`: agreement 184, contradiction 10, rate 0.948)
- **THEN** the gate SHALL validate (unlike the zero-tolerance baseline which stays permanently disqualified on the first contradiction) and admit the blended scorer to the ranking + frontier-priority paths, while a genuinely miscalibrated scorer (e.g. `sk48`, agreement rate 0.056) SHALL remain gated OFF -- and the artifact SHALL report whether turning the scorer on changed banked levels AND whether `states_expanded` regressed.

#### SCENARIO-ARC-FCP-5729-RESET-ON-LEVELUP
- **WHEN** the reset-on-level-up gate config runs a game that banks a level, triggering `reset(reset_to_prior=True)`
- **THEN** the gate SHALL clear its own agreement/contradiction/observed counters (giving the next level a fresh validation window) while an ordinary `reset()` without `reset_to_prior` SHALL NOT clear them, and the artifact SHALL report that on a roster where no game reaches a 2nd level this config is byte-identical to baseline (the mechanism is correct but has no lever to pull), never fabricating a difference.

### REQ-ARC-FCP-5730: CNN Held-Out AUROC Action-ID-Base-Rate + Seed-Robustness Audit

exp5729 localized the live-search inertness of the `SmallFrameChangeCNN`-backed
action-effect scorer to the scorer's OWN discriminative quality (loosening the
`GroundTruthValidatedFrameChangeScorer` gate turned it on ~27,000 times on
`lp85` yet search was byte-identical). `SmallFrameChangeCNN` has a prior
held-out evaluation in exp4547 (`cnn_held_out_delta_auroc = 0.7092`), but that
number is compared only against a HARDCODED chance constant (0.5) on a ~95%-
"changed"-skewed held-out set -- never against an action-id-only base-rate
control. exp5727 found that exactly this control refuted the apparent signal of
a DIFFERENT (linear `cross_game_features_v3`) classifier on the same
action-effect target (`frame_adds_over_action_id = -0.039`: the "signal" was a
per-action-type base rate, not frame content). This REQ applies exp5727's
control to the CNN.

`python/carnot/experiment_5730_cnn_baserate_audit.py` SHALL IMPORT exp4547's
corpus/split/training building blocks verbatim (not re-implement them), and
SHALL (a) re-measure `cnn_held_out_delta_auroc` on the current corpus (the
human-replay corpus was regenerated ~11.8x larger since exp4547, so exp4547's
0.7092 is CITED as the prior-corpus number, not reproduced), (b) add an
ACTION-ID-ONLY baseline (per-action-id empirical `changed` rate from TRAIN,
Laplace-smoothed, applied to HELDOUT) scored through the SAME `binary_auroc`
helper and report `frame_adds_over_action_id = cnn_auroc - action_id_only_auroc`
(exp5727's naming/sign convention), and (c) probe WITHIN-ACTION-ID
discrimination for the `click_head` (action_id==6 fixed, per-pixel) and the
`directional_head` (each action id 1..5 fixed, per-frame) where the action-id
base rate is constant and cannot explain any lift. Because a single CNN training
run is stochastic, every within-action-id number SHALL be reported ACROSS >=5
seeds (mean/min/max/std) against a per-seed UNTRAINED-CNN structural baseline; a
head SHALL be declared to carry a real learned signal ONLY if its WORST seed
clears the 0.55 floor AND beats its untrained baseline. A `prior_work_extended`
block SHALL cite exp4547, exp4568, exp5727, exp5728, and exp5729 by id + verdict.

**RESOLUTION (2026-07-19).** Ran cleanly across 5 seeds on the current corpus
(165,542 examples, 31,687 held-out trainable). `positive_control_passed: true`
(training reduced loss; the AUROC machinery reaches 0.918 when structure aligns).
The CNN carries NO robust, non-base-rate signal:

- **Base-rate mirage.** `action_id_only_auroc = 0.549` is HIGHER than the CNN's
  mean `cnn_held_out_delta_auroc = 0.539` (per-seed 0.485..0.576), so
  `frame_adds_over_action_id = -0.010` mean (range -0.064..+0.027, std 0.032) --
  near-zero/negative, the SAME finding as exp5727's linear classifier (-0.039).
  The held-out AUROC is the trivial per-action-type base rate, not frame content.
- **The apparent click_head win was seed luck.** During scoping, seed 4547 alone
  gave a click_head within-action AUROC of 0.918 (N=2973; 2553 changed / 420
  no-op) -- in isolation a strong positive. Across 5 seeds it is `min 0.444,
  max 0.918, mean 0.570`, and the UNTRAINED (random-init) structural baseline is
  actually HIGHER on average (`mean 0.580`): training does not reliably add click
  discrimination over random weights. `click_head_robust_learned_signal: false`.
- **The directional_head is at/below chance.** Trained within-action
  `mean 0.498, min 0.402, max 0.616`; `directional_head_robust_learned_signal:
  false`.

Verdict: `complete: cnn_held_out_auroc_is_action_id_base_rate_and_seed_luck_
mirage_frame_adds_-0.010_no_robust_within_action_signal_matching_exp5727_null`.
This closes the exp5590->5728->5729 chain: the CNN is redundant with the memory
term (`PersistentAEM` already encodes per-action-id base rates), which is
mechanically why loosening the blend weight (exp5728) and the gate (exp5729)
banked no levels. Recommendation (operator-only): do NOT retrain
`SmallFrameChangeCNN` on the same frame-only representation, nor re-tune its
blend weight/gate -- the ceiling is the representation, not the model, weight, or
gate. The next lever per Missing-Verifier Gap Logging is a richer state-grounded
representation (the exp5727 gap), not another pass on this scorer.

Required field principles:

- `frame_adds_over_action_id`: principle "the headline -- mean cnn_auroc minus the action-id base rate; near-zero/negative means the held-out AUROC is a per-action-type base rate, not frame-content discrimination (guards the exp4547 0.709-vs-chance overclaim)."
- `seed_robustness`: principle "a within-action AUROC is only a real learned signal if the WORST seed clears the floor AND beats the untrained structural baseline; a lone high seed is seed luck, not signal (the exact trap the Adversarial Verification cross-check rule guards)."
- `click_head_within_action_discrimination`: principle "the one signal a per-action-type base rate structurally cannot explain (action_id fixed) -- reported across seeds vs an untrained control so a stochastic high value is not mistaken for a learned discriminator."
- `positive_control_passed`: principle "the AUROC harness is functional (loss drops; machinery reaches above-chance when structure aligns) -- a null is informative only if the harness could have detected signal."

#### SCENARIO-ARC-FCP-5730-BASE-RATE-CONTROL
- **WHEN** the CNN's held-out `changed`-vs-`no-op` AUROC is measured on a held-out set that is ~91% "changed" and an action-id-only baseline (per-action-id empirical change rate from TRAIN) is scored on the SAME held-out examples through the SAME `binary_auroc` helper
- **THEN** the artifact SHALL report `action_id_only_auroc` and `frame_adds_over_action_id = cnn_auroc - action_id_only_auroc`, and SHALL NOT claim the CNN carries frame-content signal when `frame_adds_over_action_id` is near-zero/negative (here -0.010 mean, action-id baseline 0.549 exceeding the CNN's 0.539) -- the held-out AUROC is attributed to the trivial per-action base rate, matching exp5727's linear-classifier finding.

#### SCENARIO-ARC-FCP-5730-WITHIN-ACTION-ID-SEED-ROBUST
- **WHEN** a within-action-id head (e.g. `click_head` with action_id fixed at 6) produces a high AUROC on one training seed (0.918 on seed 4547) but is re-measured across >=5 seeds against a per-seed untrained-CNN structural baseline
- **THEN** the artifact SHALL report the trained AUROC distribution (mean/min/max) AND the untrained baseline distribution, and SHALL declare `..._robust_learned_signal: false` when the worst seed does not clear the 0.55 floor or does not beat the untrained baseline (here trained min 0.444 vs untrained mean 0.580) -- never headlining the lone lucky-seed value as evidence of a learned discriminator.

### REQ-ARC-FCP-5740: ObjectHistorySaliencePrior Properly-Powered 11-Game Live A/B + Blob-Cache Perf Fix

`ObjectHistorySaliencePrior` (`arc_object_history_salience.py`) is an
already-live-wired `action_prior` (gated OFF by default,
`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED = False`) that wraps
`ColorBlobSaliencePrior` with a per-`object_hash` change-history bonus: click
candidates on objects observed to change the frame when clicked are preferred.
A prior A/B (exp5603) found `object_history_salience_ab_no_op_at_either_weight`
but tested exactly ONE game (`m0r0`, 37 transitions) with the ONLY available
metric being trajectory divergence (it self-documented "no OfflineSolver-
equivalent states_expanded metric exists for action_prior"). exp5732 then
measured a real ONLINE prefix-causal `object_hash`-keyed memory signal (AUROC
0.844 vs click-bucket 0.711, +0.133) that its own coverage probe showed was
heavily `lp85`-dominated -- so the `m0r0` null was plausibly game-specific, not
mechanism-wide. This re-test is NOT a doomed rerun (Failed-Experiment Rerun
Discipline): it changes the roster (1 game -> the full 11-game exp5729/exp5732
roster) AND the metric (trajectory-only -> real `levels`/`states_expanded`),
NOT the weight (exp5603 already settled both tested weights are null).

A per-candidate perf bug is fixed FIRST (independently motivated):
`ObjectHistorySaliencePrior.score()` called `connected_color_blobs` RAW per
candidate, bypassing the module-level per-frame LRU cache
(`_cached_blobs_and_counts`, REQ-ARC-FCP-5699 item-2, 2026-07-16) that its own
`base_prior` (`ColorBlobSaliencePrior.score`) already uses with identical
params -- reintroducing the exact per-candidate flood-fill the 2026-07-16 fix
eliminated elsewhere. It is routed through the same cache (the `base_prior.score`
call one line above already warms it for this exact grid key), behavior-
preserving (identical blobs -> identical scores).

`python/carnot/experiment_5740_object_history_salience_11game_ab.py` SHALL run a
matched-budget A/B across the full 11-game roster (same roster + `budget=200` +
`CARNOT_ARC_DISABLE_INDUCTION=1` no-LLM harness as exp5729, single-threaded,
per-game fixed seed) with FOUR arms: (1) `baseline` -- shipped default
(`action_prior=None`, since `SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED=False`); (2)
`blob_only` -- ISOLATION control (`action_prior=ColorBlobSaliencePrior()`, no
object-history bonus), REQUIRED because the shipped default has no action_prior
so the faithful live flip adds BOTH the ColorBlob prior AND the bonus; (3)
`treatment_default` -- the faithful live flip (`object_history_salience=True`,
default weight 10.0); (4) `treatment_rescaled` -- exp5603 continuity (weight
2000.0). The artifact SHALL report `levels_gained_total_by_arm`,
`states_expanded_total_by_arm`, per-game breakdown, `safety_regression_check`
(states vs baseline, exp5729 discipline), a THREE-way `trajectories_diverge_per_game`
(the LOAD-BEARING `object_history_bonus_marginal_vs_blob_only` isolation, the
CONFOUNDED `full_live_flip_vs_baseline`, and `color_blob_prior_alone_vs_baseline`
attribution), `blob_cache_perf_fix` (before/after per-candidate timing +
behavior-preservation check), and a `prior_work_extended` block citing exp5603
(the null this extends), exp5732 (the motivating online signal), and exp5729
(the harness precedent). The verdict SHALL key on the ISOLATED bonus effect
(treatment vs `blob_only`), never the confounded treatment-vs-baseline.

**RESOLUTION (2026-07-20).** Ran cleanly on the full 11-game roster,
budget=200/game, 4 arms, 270.3s, adversarial_verify 0 flags. NO capability gain:
every arm banked exactly `1` level (`lp85` L1, a pre-existing registry solve
reached in every arm), `any_config_beats_baseline_levels: false`,
`levels_gained_headroom_present: true` (interpretable null). The ISOLATION was
decisive: the object-history bonus's TRUE marginal effect over the ColorBlob
prior (treatment vs `blob_only`) changed search on only 4 games at default
weight (`cd82, lp85, sp80, su15`) and 5 at the rescaled weight (`+m0r0`) -- 9
arm-game divergences -- while the ColorBlob prior ALONE (`blob_only` vs baseline)
changed search on 8 games and the CONFOUNDED treatment-vs-baseline showed 16.
So the bulk of the "`object_history_salience=True` changes search" effect is the
ColorBlob base prior, NOT the object-history bonus -- exactly the confound the
`blob_only` arm was built to expose, and the reason a treatment-vs-baseline-only
comparison (exp5603's implicit frame) would have over-credited the bonus.
`m0r0` is NOT in the DEFAULT-weight isolation set, reconciling with exp5603's
`m0r0` default-weight no-op; the full-roster breadth shows the bonus is not
mechanism-wide inert (4-5 games diverge) but its effect is small and banks no
level. No safety regression: the action_prior arms REDUCED `states_expanded`
(baseline 931 -> `blob_only` 813 -> treatment 812, `any_config_states_regression:
false`). The perf fix removed a real ~4x per-candidate flood-fill cost (raw
2807us/cand -> fixed 703us/cand, `fixed_scores_identical_to_raw: true`); the
residual fixed-vs-baseline overhead (~47x per candidate in isolation) is
`object_hash(blob)`, inherent to the mechanism, and does NOT confound the
capability A/B (run_game's budget is action-count, not wall-clock; the
no-induction search path has no wall-clock cutoff). Verdict:
`complete: object_history_bonus_marginal_over_colorblob_prior_changes_search_on_9_arm_games_but_no_level_gain_over_baseline_1_colorblob_prior_drives_the_larger_behavioral_change`.
Recommendation (operator-only): do NOT flip
`SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED` on a capability basis (no level
gain); the object-history bonus is not the live-path lever. Per the retire
condition this is NOT a same-verdict rerun of exp5603 (single-game trajectory-
only vs full roster with real metrics + isolation), so the lineage is not
auto-retired, but the marginal signal does not justify a live flip. The
object-identity signal exp5732 found is real ONLINE but does not reach the live
search's frontier ordering here -- logged as a verifier gap.

Required field principles:

- `object_history_bonus_marginal_vs_blob_only`: principle "THE isolated object-history-bonus effect -- both arms carry the ColorBlob prior so only the bonus differs; the verdict keys on this, never the treatment-vs-baseline comparison confounded by the ColorBlob prior."
- `blob_cache_perf_fix`: principle "documents the independently-motivated uncached-blob bug, the minimal cache-routing fix, the before/after per-candidate timing, and WHY the residual object_hash cost does not confound an action-budget A/B."
- `safety_regression_check`: principle "a config that banks a level while materially inflating states_expanded may be luck-under-noise -- states vs baseline flags any blow-up so a level win is never reported without its search cost."
- `any_config_beats_baseline_levels`: principle "the load-bearing capability boolean -- an action_prior only justifies a live-default flip if it beats baseline on banked levels, not on a proxy or a confounded divergence count."

#### SCENARIO-ARC-FCP-5740-BONUS-MARGINAL-ISOLATED-FROM-COLORBLOB-PRIOR
- **WHEN** the shipped default has no `action_prior` (`SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED=False`), so enabling `object_history_salience=True` wraps a freshly-constructed `ColorBlobSaliencePrior` and thus adds BOTH the ColorBlob base prior AND the object-history change bonus
- **THEN** the experiment SHALL include a `blob_only` isolation arm (`action_prior=ColorBlobSaliencePrior()`, no bonus) and report the object-history bonus's marginal behavioral effect as treatment-vs-`blob_only` (here 4 games at default weight, 5 at rescaled), NOT treatment-vs-baseline (here 16, dominated by the ColorBlob prior's own 8-game effect), and the verdict SHALL attribute the larger behavioral change to the ColorBlob prior -- never over-crediting the object-history bonus with a confounded divergence count.

#### SCENARIO-ARC-FCP-5740-BLOB-CACHE-PERF-FIX-BEHAVIOR-PRESERVING
- **WHEN** `ObjectHistorySaliencePrior.score()` is routed through `_cached_blobs_and_counts` instead of calling `connected_color_blobs` raw per candidate
- **THEN** the fix SHALL be behavior-preserving (identical blobs -> `fixed_scores_identical_to_raw: true`, verified against a pre-fix raw replica on a real frame) and SHALL remove the per-candidate flood-fill cost (raw ~2807us/cand -> fixed ~703us/cand), and the artifact SHALL document that the residual overhead over the blob-only baseline is `object_hash(blob)` (inherent to the mechanism) which does NOT confound the capability metrics because `run_game`'s budget is action-count, not wall-clock, and the no-induction search path has no wall-clock cutoff.

### REQ-ARC-FCP-5756: InertClickSigPruner Properly-Powered 11-Game Live A/B + Cache Hygiene

`InertClickSigPruner` (`arc_inert_click_pruner.py`) is an already-live-wired
move-pruner (gated OFF by default, `SUBMITTED_INERT_CLICK_PRUNER_ENABLED =
False`) that drops click candidates whose structural signature `(color, size,
is_rect, twin_count)` has repeatedly produced zero frame change, gated by
`HazardMovePruner`'s trust+specificity discipline (`min_observations=4`,
`min_specificity=0.9`, any ever-leveled signature permanently sacred). It is
wired into the LIVE `E3AgentPolicy`/`StepwiseExplorer` path as a
`rank_candidates` drop-filter in `_candidates` + a real `observe()` from
`_ingest`'s per-transition OBSERVE hook. Its own docstring states flipping it on
for the SCORED agent "needs its own matched-budget offline A/B first, per the
`solve_rate_dropped` guardrail." Two prior artifacts touched this mechanism,
BOTH single-game (`m0r0`) nulls at a low budget (~37 transitions): exp5595 (the
offline prototype, `0 confidently inert`) and exp5602 (the matched-budget A/B,
`no_op_offline_and_live_at_this_budget_on_m0r0`, whose OfflineSolver arm's
pruner `observed=0` -- the directed search never even exercised it -- and whose
live E3 arm observed 32 clicks but pruned 0). The shared root cause: a single
game at a low budget never accumulates `min_observations=4` inert observations
of one signature to fire. This re-test is NOT a doomed rerun (Failed-Experiment
Rerun Discipline): it changes the roster (1 game -> the full 11-game
exp5729/exp5732/exp5740 roster), the budget (~37 -> 200), AND uses real
capability metrics (`levels`/`states_expanded`) on the ACTUAL live-wired E3 path,
NOT exp5602's OfflineSolver arm where the pruner was never exercised.

A cache-hygiene change is made FIRST (NOT exp5740's per-candidate flood):
`InertClickSigPruner`'s live path already decomposes ONCE per `rank_candidates`
call (then loops candidates via the cheap `blob_at_click` lookup) and ONCE per
`observe()` transition -- NOT once per candidate -- so it does NOT have exp5740's
severe O(candidates x grid_cells) flood. What it DID do was call
`connected_color_blobs` RAW, bypassing the shared `_cached_blobs_and_counts`
per-frame LRU cache (the same key `ColorBlobSaliencePrior.score` uses with
`min_pixels=1`/`max_component_fraction=1.0`), so repeated same-frame
decomposition recomputed the flood from scratch. A new `_frame_blobs(grid)`
helper routes through the cache (int16-normalized to match the shared key),
behavior-preserving (identical signatures, verified). Because `run_game`'s budget
is action-count, not wall-clock, this per-frame timing change cannot confound the
capability read either way -- it is hygiene, not a capability-confound fix.

`python/carnot/experiment_5756_inert_click_pruner_11game_ab.py` SHALL run a
matched-budget A/B across the full 11-game roster (same roster + `budget=200` +
`CARNOT_ARC_DISABLE_INDUCTION=1` no-LLM harness as exp5740, single-threaded,
per-game fixed seed) with THREE arms: (1) `baseline` -- shipped default
(`inert_click_pruner` OFF); (2) `treatment_default` -- the faithful live flip
(`inert_click_pruner=True`, shipped params `min_observations=4`,
`min_specificity=0.9`); (3) `treatment_aggressive` -- `min_observations=2` (Reki's
original K=2 floor) to test whether the shipped default's higher evidence floor is
WHY the pruner never fired in exp5595/exp5602. The artifact SHALL report
`levels_gained_total_by_arm`, `states_expanded_total_by_arm`, per-game breakdown,
`n_signatures_reaching_evidence_floor_per_game` + `n_candidates_pruned_per_game`
(the engagement diagnostics distinguishing a never-fired guard from a real null),
`trajectories_diverge_per_game`, `blob_cache_perf_fix` (before/after timing +
behavior-preservation), a `safety_regression_check` that includes an EMPIRICAL
missed-win check (`suppressed_a_winnable_click` = any game where a treatment arm
banks FEWER levels than baseline), and a `prior_work_extended` block citing
exp5595 + exp5602 (the m0r0 nulls this extends) and exp5740 (the harness
precedent). The verdict SHALL distinguish: pruner never fires; fires but banks
zero prunes; fires + holds levels + REDUCES states (efficiency win); fires + holds
levels but states did NOT decrease (no efficiency benefit); or suppressed a
winnable click (safety failure).

**RESOLUTION (2026-07-20).** Ran cleanly on the full 11-game roster,
budget=200/game, 3 arms, 183.3s, adversarial_verify 0 flags. NEW FINDING (unlike
exp5595/exp5602's never-fired null): at budget=200 the pruner FIRES. Signatures
crossed the evidence floor on 8/11 games (`ls20`/`tu93`/`wa30` had 0 click
transitions), and the pruner actually dropped candidates on ONE game, `sk48`
(1122 candidates at default `min_observations=4`, 1521 across both treatment arms
= 2643 total; `treatment_aggressive` at `min_observations=2` crossed the floor on
83 signatures vs 48 at default). So the larger budget DID cross the evidence floor
the single-game low-budget priors never reached. Capability: every arm banked
exactly `1` level (`lp85` L1, a pre-existing registry solve reached in every arm),
`any_config_beats_baseline_levels: false`, `levels_gained_headroom_present: true`
(interpretable null). SAFETY: `any_config_suppressed_a_winnable_click: false` --
NO game lost a level under pruning; the trust+specificity+sacred gate correctly
avoided suppressing any winnable click (verified empirically via per-game level
deltas, not assumed). EFFICIENCY (the load-bearing result): pruning did NOT reduce
search cost -- `states_expanded` INCREASED, baseline 931 -> `treatment_default`
953 (+22, all from `sk48` 26 -> 48) -> `treatment_aggressive` 960 (+29). On the
one game where it actually pruned, dropping frame-inert click candidates reshaped
the search frontier so the search expanded MORE states for the same banked levels
-- the OPPOSITE of `HazardMovePruner`'s tu93 efficiency win (2947 -> 2859). Below
the 20% `states_expanded_regression` threshold (+2.4%), so not flagged as a
material regression, but a real deterministic (per-game fixed seed) increase, not
noise. The cache-hygiene fix removed a real ~205x per-decomposition cost on
repeated same-frame calls (raw 3018us/call -> cached 15us/call,
`fixed_signatures_identical_to_raw: true`). Verdict:
`complete: inert_click_pruner_fires_at_budget200_prunes_2643_candidates_holds_levels_at_1_no_missed_win_but_states_expanded_did_not_decrease_delta_+22_no_efficiency_benefit`.
Recommendation (operator-only): do NOT flip `SUBMITTED_INERT_CLICK_PRUNER_ENABLED`
-- at this budget the inert-click pruner buys no capability AND no efficiency (a
frame-inert click can still be a necessary traversal step the search re-routes
around). Per the retire condition, another plain budget/roster A/B is not warranted
without a NEW mechanism that makes pruning reduce rather than reshape search cost;
logged as a verifier gap.

Required field principles:

- `n_signatures_reaching_evidence_floor_per_game`: principle "the load-bearing engagement diagnostic -- distinguishes 'inert-click pruning does not help' from 'the evidence floor was never reached' (exp5595/exp5602's ambiguity at budget~37); a null is only interpretable (FALSE_NEGATIVE_RISK) if some signature crossed the floor somewhere."
- `n_candidates_pruned_per_game`: principle "0 everywhere means treatment is byte-identical to baseline by construction (a never-fired guard); a positive count means the capability/efficiency delta is a REAL pruning effect."
- `safety_regression_check.suppressed_a_winnable_click`: principle "the pruner-specific safety property -- a treatment arm banking FEWER levels than baseline on any game means the pruner dropped a click that led to progress, the exact failure the trust+specificity+sacred gate exists to prevent; measured empirically via per-game level deltas, not assumed."
- `blob_cache_perf_fix`: principle "documents the cache-hygiene change and WHY it is NOT exp5740's per-candidate flood (the pruner already decomposes once per frame), plus the behavior-preservation assertion and the action-budget invariance argument."

#### SCENARIO-ARC-FCP-5756-ELEVEN-GAME-CAPABILITY-AB
- **WHEN** the shipped default has `inert_click_pruner` OFF and the faithful live flip (`inert_click_pruner=True`) is A/B'd against baseline across the full 11-game roster at `budget=200` with real `levels`/`states_expanded` metrics, extending exp5595/exp5602's single-game (`m0r0`) low-budget (~37 transitions) nulls where no signature ever crossed the evidence floor
- **THEN** the experiment SHALL report per-arm `levels_gained_total_by_arm`, `states_expanded_total_by_arm`, per-game `n_signatures_reaching_evidence_floor_per_game` + `n_candidates_pruned_per_game` (here: the pruner FIRES at budget=200 -- 2643 candidates pruned, all on `sk48` -- unlike the priors' zero), an EMPIRICAL missed-win safety check (here `any_config_suppressed_a_winnable_click: false`, no game lost a level), and a verdict that distinguishes an efficiency win (states reduced) from no efficiency benefit (here states INCREASED 931 -> 953, so `pruner_reduced_states_expanded: false`) -- and the recommendation SHALL be operator-only, never self-authorizing a live-default flip.

#### SCENARIO-ARC-FCP-5756-INERT-CLICK-CACHE-HYGIENE
- **WHEN** `InertClickSigPruner._signature_for_click` and `.rank_candidates` are routed through `_cached_blobs_and_counts` (via a new `_frame_blobs` helper) instead of calling `connected_color_blobs` raw, sharing the per-frame LRU cache key `ColorBlobSaliencePrior.score` uses
- **THEN** the change SHALL be behavior-preserving (`fixed_signatures_identical_to_raw: true`, verified against a raw replica on a real frame) and the artifact SHALL document that this is NOT exp5740's per-candidate flood (the pruner already decomposes once per `rank_candidates` call and once per `observe`, not once per candidate) and that the per-frame timing change (raw ~3018us/call -> cached ~15us/call on repeated same-frame decomposition) cannot confound the capability metrics because `run_game`'s budget is action-count, not wall-clock.

### REQ-ARC-FCP-5757: Candidate-Coverage Attribution (Generation vs Selection vs Planning), Offline & LLM-Free

Tonight's seven consecutive live-path A/Bs (REQ-ARC-FCP-5590/5728/5729/5730/5732/
5740/5756) each MODIFIED a component and measured a level-gain DELTA on a
near-zero-headroom 11-game roster, and every one returned a clean honest null. A
delta on a near-zero-headroom corpus cannot distinguish "component useless" from "no
headroom to move" (the `FALSE_NEGATIVE_RISK` failure mode). REQ-ARC-WMTE-5720/5724
hit the same wall from the induction side (induce-completion 0/12). Per the
2026-07-20 top-project search-architecture audit (`docs/research-notes/
arc-top-project-search-architecture-audit-2026-07-20.md` §4), this REQ runs a
STRUCTURAL ATTRIBUTION with no delta to be null on: it classifies WHERE every
known-progress action sits relative to Carnot's own live candidate generator, so the
result localizes the gap and dictates a build in every branch.

`python/carnot/experiment_5757_candidate_coverage_attribution.py` SHALL, for every
action on the offline known-winning path (`solve_adaptered(game, 1)`) of each
pre-registered STALLED game (0 new levels in the exp5756 baseline AND an adapter that
reaches L1 offline), classify that action relative to `rich_action_candidates(frame)`
(`arc_graph_explore.py:117`) into exactly one of: (a) NOT in the candidate set
(perception/generation miss); (b) in-set but NOT frame-changing in isolation while on
a known level-up path (multi-step/lookahead signal); (c) in-set AND frame-changing but
ranked >= a pre-registered low-rank threshold (selection/ranking miss); or (d) in-set,
frame-changing, top-ranked (already handled, not a gap). Membership SHALL be reported
both coordinate-exact and radius-tolerant. The artifact SHALL carry
`inference_substrate: verifier_ensemble_against_cached_candidates`,
`verifier_is_oracle: false`, a `diagnostic_offline_attribution` `solve_provenance`,
`offline_reproduced: true`, `random_seed`, `reproducibility_checksum`, a
`prior_work_extended` block citing the seven nulls + REQ-ARC-WMTE-5720/5724 by id +
verdict + what is different (a structural attribution, not another delta A/B), and a
pre-registration (`n_pre_registered` + `games_pre_registered` + thresholds) written
BEFORE the buckets are computed.

#### SCENARIO-ARC-FCP-5757-PRE-REGISTRATION-BEFORE-BUCKETS
- **WHEN** the attribution is computed
- **THEN** the artifact SHALL contain a `pre_registration` block (declared `before bucket_computation`) fixing the stalled-game list, the minimum total-progress-action floor `n_pre_registered`, the `low_rank_threshold`, the `tolerance_radius`, and the bucket definitions -- none of which is a threshold on the gate's a/b/c FRACTIONS -- so N cannot be retro-fit to hit a branch (Failed-Experiment Rerun Discipline); and if the measured total progress actions fall below `n_pre_registered` OR fewer than 3 games reproduce, the pre-registered RETIRE condition SHALL fire instead of forcing a bucket assignment.

#### SCENARIO-ARC-FCP-5757-BUCKET-B-REQUIRES-KNOWN-PROGRESS-PATH
- **WHEN** an in-set action is not frame-changing in isolation and is therefore a bucket-(b) candidate
- **THEN** it SHALL be counted as (b) only because it lies on a real level-up trajectory (every classified action is on an adapter winning path to L1, by construction), the artifact SHALL report the verified-vs-unverified (b) count and a `b_well_powered` flag (>= `b_power_floor` across >= 2 games), and when (b) is under-powered the search/lookahead hypothesis SHALL be reported as UNTESTABLE-HERE -- never as evidence against search (design 4a point 2). (Observed: `b_count == 0`, so the search lever gets zero support.)

#### SCENARIO-ARC-FCP-5757-GATE-AND-RETIRE-DECISION-MATRIX
- **WHEN** the pooled bucket fractions are known (with a bootstrap 95% CI)
- **THEN** the gate SHALL fire `a_perception_generation` if fraction(a) > 0.5, `b_lookahead_reopened` if fraction(b) > 0.3 AND (b) is well-powered, `c_selection_ranking` if fraction(c) > 0.3, else report the dominant GAP bucket among a/b/c via the gap-only conditional distribution when bucket-(d) handled dilutes all three below their thresholds; the recommendation SHALL name the specific live-path consumer the dominant bucket points at (segmentation/click-generation for a, `plan_in_model`/induction for b, `candidate_router`/value-head + click-cap for c) and SHALL be operator-only. (Observed: 98.9% exact / 100% tolerant coverage, `b == 0`, all gap actions bucket-(c) object-click ranking misses -> the SELECTION/RANKING branch, which also empirically down-weights `GAP-ARCH-NO-HIERARCHICAL-SEARCH`.)

### REQ-ARC-FCP-5758: Small-Object-First Click-Reorder -- Live A/B Against the exp5757 Selection/Ranking Gap

REQ-ARC-FCP-5757 localized the ONLY residual single-action gap to SELECTION/RANKING of
object clicks (bucket c: 6 action-6 clicks in-set + frame-changing but rank >= 12; r11l
x1, su15 x4). This REQ attacks exactly that gap. The root-cause diagnosis (reproducing the
gap frames and dumping per-object salience features) is that the low-ranked winning clicks
are consistently VERY SMALL objects (r11l's rank-22 winner is a single pixel area=1; su15's
repeatedly-clicked winner is area=1; r11l's others are area 4 and 12), and the shipped
default orders object clicks by `area * (1 + 1/(1 + global_color_pixels))`, which is
AREA-DOMINATED so a 1-pixel target always loses to large decorative regions. The HONEST
nuance: for the single-pixel-field cases the winner is ALSO outranked by many EQUAL-size,
rarer-coloured pixels, so no monotonic-in-area formula can surface it -- part of the gap is
a discriminating-SIGNAL absence (a Missing-Verifier Gap), not a formula bug.

`python/carnot/agentic/arc_graph_explore.py` SHALL provide an opt-in, gated-OFF-by-default
reorder (`CARNOT_ARC_SMALL_OBJECT_FIRST=1` -> `_small_object_first_click_points`) that
reorders the SAME object-click candidate set into two bands -- a SMALL band (object area
<= 8) ordered by colour-rarity (rarest first) tried FIRST, then all remaining objects in
the proven salience order -- adding/dropping no click and being byte-identical when the flag
is off. `python/carnot/experiment_5758_click_ranking_fix_ab.py` SHALL run a matched-budget
(200-action) live A/B on the same 11-game roster as REQ-ARC-FCP-5740, per-game fixed seed,
`CARNOT_ARC_DISABLE_INDUCTION=1` (no-LLM), two byte-identical `E3AgentPolicy(game)` arms
differing ONLY in the flag, reporting real capability metrics (`levels_gained_total_by_arm`,
`states_expanded_total_by_arm`), an offline `rank_shift_offline` diagnostic (winning-click
rank under default vs the reorder on the exact r11l/su15 gap frames), an
`r11l_su15_specifically_fixed` decisive boolean, a `safety_regression_check` (>20% states
growth flagged), and a `prior_work_extended` block naming REQ-ARC-FCP-5757 (the motivating
gap), exp4556 (the learned-router null on colour-variant first-contact), and the
`CARNOT_ARC_TIER_SCHEDULE` null (`proto_tier_ab.json`) by id + verdict + what is different.
The artifact SHALL carry `inference_substrate:
offline_arcade_live_agent_runtime_self_discovery_no_llm`, `verifier_is_oracle: false`,
`solve_provenance: development_proxy`, `random_seed`, and `reproducibility_checksum`, and
the recommendation to flip the live default SHALL be operator-only.

**Implementation Status (measured 2026-07-20): DONE -- clean NULL, reframed as a
Missing-Verifier Gap.** Both arms bank 1 level (lp85 L1, incidental -- headroom present, so
the null is interpretable, not a no-headroom artifact); r11l and su15 stay at 0 in BOTH
arms (`r11l_su15_specifically_fixed: false` -- the decisive negative); the reorder changes
the search trajectory on 7 of 11 games yet banks no additional level; and it actually
LOWERS search cost (`states_expanded` 931 -> 807, -13.3%, no regression). The offline
rank-shift is a WASH (0 of 2 decisive games reduce their low-rank count): the reorder
reshuffles the single-pixel field without picking THE winning pixel, confirming the
signal-absence diagnosis. `adversarial_verify.py`: 0 flags. The flag stays OFF; the gap is
logged as a Missing-Verifier Gap for a learned / goal-conditioned click discriminator.

#### SCENARIO-ARC-FCP-5758-ELEVEN-GAME-CAPABILITY-AB
- **WHEN** the live A/B runs the two arms (`baseline` flag-off vs `small_object_first` flag-on) over the 11-game roster at a matched 200-action budget with per-game fixed seeds and no LLM
- **THEN** the artifact SHALL report `levels_gained_total_by_arm`, `states_expanded_total_by_arm`, per-game level/states rows, `any_config_beats_baseline_levels`, `levels_gained_headroom_present`, and a `safety_regression_check` flagging any arm whose `states_expanded` grows > 20% vs baseline; and a level win SHALL NOT be reported without its search cost. (Observed: baseline 1 / treatment 1 level, states 931 -> 807, no regression, `any_config_beats_baseline_levels: false`, headroom present via lp85.)

#### SCENARIO-ARC-FCP-5758-R11L-SU15-DECISIVE
- **WHEN** the decisive test is evaluated on the two games exp5757 localized the click-ranking gap to
- **THEN** the artifact SHALL carry `r11l_su15_specifically_fixed` = true iff r11l and/or su15 completes an ADDITIONAL level under the reorder within budget that baseline does not, with a per-game `baseline_levels` / `treatment_levels` / `gained_additional_level` breakdown. (Observed: false -- both games 0 levels in both arms; the reorder does not convert to a level on the target games.)

#### SCENARIO-ARC-FCP-5758-RANK-SHIFT-OFFLINE-IS-A-WASH
- **WHEN** the offline `rank_shift_offline` diagnostic recomputes each winning object-click's rank at the exact r11l/su15 gap frame under default vs the reorder
- **THEN** the artifact SHALL report per-game the winning-click ranks and the low-rank (rank >= 12) count under each ordering, and summarise `games_low_rank_reduced` vs `games_low_rank_worsened`; a reduction of 0 SHALL be interpreted as the reorder NOT surfacing the bucket-c winners (they are buried under a field of equal-size rarer pixels), confirming the discriminating-signal-absence (Missing-Verifier) framing rather than a formula bug the reorder can close. (Observed: 0 of 2 games reduced -- a wash; the flag stays OFF and the gap is logged for a learned/goal-conditioned click discriminator.)

### REQ-ARC-FCP-5904: Coordinate-Aware Online Click-Target Discrimination (the exp5758 Missing-Verifier Gap)

REQ-ARC-FCP-5758 closed with a clean NULL and an explicit forward pointer: "the gap is
logged as a Missing-Verifier Gap for a learned / goal-conditioned click discriminator."
This REQ builds the discriminating SIGNAL that entry names, and first documents WHY no
learned router could have closed the gap: the live candidate router is
COORDINATE-BLIND. `arc_discriminative_router._action_id` returns the action TYPE integer
(6 for every click), and `cross_game_features_v3` consumes it only through
`_action_features()`, a 7-dimensional one-hot in which coordinates are structurally
unrepresentable. Measured on the offline arcade (lp85, `max_click=48`): 37 distinct click
targets, 37 distinct `candidate_action_key` values, ONE distinct value reaching the
router, ONE distinct score, and `rank()` preserving the input click order exactly. The
collapse reproduces on all 19 click-capable public games.

`python/carnot/agentic/arc_click_target_features.py` SHALL provide a coordinate-aware
click-target featurization split explicitly into a per-FRAME context
(`click_target_frame_context`, computed once per ranking pass, content-cached) and a
per-CANDIDATE featurizer (`click_target_features`, O(1)-ish given the context), exporting a
fixed-order `CLICK_TARGET_FEATURE_NAMES` of length `CLICK_TARGET_FEATURE_DIM` = 21. It SHALL
NOT call `blob_topology()` at either granularity (measured 322 ms on one bp35 frame, ~100x
the rest of the context combined). It SHALL also provide `OnlineClickTargetDiscriminator`, a
numpy logistic head that standardizes with a variance FLOOR (not `std + 1e-8`) plus a
z-clip, because the unfloored form measurably saturates to exactly 1.0 on a near-constant
in-episode column -- collapsing the ranking back into a tie and emitting
IMPLAUSIBLE_PERFECT-shaped values.

`python/carnot/agentic/arc_discriminative_router.py` SHALL provide `OnlineClickTargetRouter`,
a wrapper that blends `weight * (P(progress) - 0.5)` onto the incumbent v3
score for click candidates only, DEFAULT OFF via the module-level constant
`SUBMITTED_ONLINE_CLICK_TARGET_ROUTER_ENABLED`, with cold start (sample gate unmet) a
bit-identical no-op and online state keyed on the frame's own `(game_id, guid)` so nothing
transfers across games or episodes. It SHALL NOT change `_action_id`,
`candidate_action_key`, `cross_game_features_v3`, or
`CrossGameDiscriminativeCandidateRouter`'s behaviour: the shared 79-feature contract is
load-bearing for `models/arc_discriminative_verifier_v3.json` (79 weights + bias, no shape
validation on load), and appending features would raise inside `proba_features` where
`score`'s blanket `except Exception: return 0.5` would silently re-create the very
constant-score bug being fixed.

**CORRECTION 2026-07-24 (review): "purely additive" was NOT sufficient for "clicks only", and
`rank` SHALL stable-partition.** An additive contribution of exactly 0.0 pins a non-click
candidate's SCORE but not its RANK: because the incumbent base score is a CONSTANT across
candidates on one frame (that is the defect), a full re-sort of the blended scores slides
clicks past the stationary keyboard actions. Measured with a fitted head at `weight=0.25`,
input `[A1, A6(3,3), A6(10,10), A6(20,6), A6(5,20), A6(12,20), A4]` returned
`[A6(20,6), A6(12,20), A6(3,3), A1, A4, A6(5,20), A6(10,10)]` -- `A4` last-to-5th and `A1`
1st-to-4th with both scores unchanged at 0.5. Since `arc_graph_explore` feeds this order into
`lst.pop(0)`, the live A/B would have silently varied NON-CLICK selection too. `rank` SHALL
therefore take the base router's ordering as the frame of reference, permute only the
candidates occupying CLICK slots, and write them back into exactly those slots, so every
non-click candidate provably keeps its base index; click ties SHALL break on base rank so a
zero `weight` (or an unfitted head) reproduces the base ordering exactly. The word "purely
additive" is struck above accordingly.

The fit SHALL be ONLINE and WITHIN-GAME ONLY -- fitted from the agent's own in-episode
transitions and discarded at episode end. Cross-game value transfer is retired by
`ops/exclusion_manifest.yaml` id `cross_game_value_transfer_retired_exp4342_v401`
(`operator_reopen_required: true`), and a hidden-game agent has no prior exposure to
transfer from in any case.

`python/carnot/experiment_5904_click_target_discrimination.py` SHALL run the OFFLINE
STAGE-1 diagnostic only: harvest a labelled (frame, click, progress) corpus by ACTUALLY
STEPPING each generated click candidate against the offline env (the label is the observed
post-click outcome, causally downstream of the click by construction), then compare four
arms -- BLIND (the incumbent coordinate-blind featurization; a NEGATIVE control that CANNOT
beat chance on a click-dependent label), COORD (the 21 features, fit online within-game on a
temporal split), RANDOM (`RandomCandidateRouter`, the already-shipped coordinate-aware but
uninformative positive control), and STEP_INDEX (a zero-perception control, the exp5835
lesson as a gate). The artifact SHALL carry `inference_substrate:
offline_arcade_live_agent_runtime_self_discovery_no_llm`, `verifier_is_oracle: false`,
`solve_provenance: development_proxy`, `random_seed`, `reproducibility_checksum`,
`duration_s`, `preconditions_checked`, `online_within_game_only: true`,
`cross_game_checkpoint_loaded: false`, per-arm AUROC with n / n_pos / n_neg, per-game rows,
`label_definition`, and an explicit statement that offline AUROC licenses NOTHING -- the
terminal gate is a LIVE A/B on banked levels (exp4545's 0.725-AUROC discriminator REGRESSED
live search, which is why the agent's `SUBMITTED_VALUE_WEIGHT` is pinned at 1e-12). It SHALL
NOT emit `offline_reproduced` or any game-solve claim.

**Implementation Status (2026-07-24): module + router + 61 tests + experiment SHIPPED, flag
OFF, STAGE-1 SMOKE ONLY (2 games requested, 1 contributing / 4 states / 182 labelled rows).**
Measured smoke, WITHIN-STATE AUROC (the metric of record, because the live router only ever
ranks within one frame): blind 0.500 exactly, coord 0.980, static-salience 0.936,
shipped-RandomCandidateRouter control 0.631, zero-perception step-index control 0.500.
`label_validity_check: valid`; ~~pre-registered gate PASSED~~ (struck 2026-07-24 -- see the
CORRECTION below); `adversarial_verify.py` 0 flags; `arc_orphan_solver_lint.py` clean.

**CORRECTION 2026-07-24 (review): the smoke run establishes NO gain over the incumbent
ordering, and the artifact now says so.** Three measured facts forced this.
(a) The originally pre-registered expression is a single threshold in disguise (within-state
`blind` is a per-state constant, hence exactly 0.5, so the second conjunct collapses into the
first) and it is CLEARED BY THE UNINFORMATIVE ARMS: evaluating it against this run's own
numbers, the shipped `RandomCandidateRouter` (0.6308) passes and so does the incumbent
`static_salience` sort (0.9360). Its pass region therefore CONTAINS REGRESSIONS -- a coord of
0.70 would have reported "passed" while ordering clicks worse than the sort it replaces. The
key `pre_registered_gate` is GONE from the artifact (a downstream capstone would have read
`passed: true` as a positive result); the expression is retained verbatim, renamed
`coordinate_blindness_repair_check`, carrying a measured `also_passed_by_arms`.
(b) The decision-relevant number, `coord - static = +0.0436`, is NOT significant: paired
within-state bootstrap (4000 replicates, rows resampled within each scored state) gives
CI95 [-0.0302, +0.1245] with P(delta<=0) = 0.133, against a measured noise floor of sd 0.0717
for the uninformative random arm's own within-state AUROC over 200 seeds on this corpus -- the
delta is roughly HALF the free noise. This is exp3540's shape (paired p=0.135, retro verdict
"advantage was small-sample artifact"). The new `informativeness_gate` (coord >= 0.60 AND
bootstrap CI95 lower bound > 0 per game) is the GATE OF RECORD, is labelled
`added_post_hoc_after_review: true` rather than backdated, and FAILS on the smoke corpus.
(c) The hard-negatives arm was STRUCTURALLY DEAD and now runs. It filtered rows on `changed`
while reusing the pooled label `(changed OR levels_up)`, so every hard row was a positive and
`auroc` returned None by construction (shipped as `coord_hard_only: null`,
`pooled_coord_hard_only_auroc.n_games: 0`); its source comment also cited "pooled 0.8747
collapses to 0.5544 hard-only, 84.6% trivial no-op negatives", figures that label could not
produce, now deleted. With the label corrected to `levels_up` conditioned on `changed`
(n=20, 5 positives): coord 0.5067 -- CHANCE -- against static 0.6600 and shipped-random 0.7067.
So the 0.98 headline is carried by the trivial "does this click do anything at all" component,
and on the non-trivial slice the head is at chance and behind the incumbent (at an n where no
direction is powered). Verdict token:
`complete_coordinate_blindness_repaired_but_gain_over_static_baseline_unestablished`.

**Read the limits with the numbers.** (1) The honest comparator is `coord - static = +0.044`
WITH ITS INTERVAL (null, per the CORRECTION above), NOT the +0.48 delta against the constant
blind arm -- beating a constant is trivial. (2) The smoke label is MEASURED IDENTICAL to
frame-change: 182/182 rows have `label == changed` and `n_levels_up_without_change == 0`, so
the 7 level-ups supply no positives of their own and `n_levels_up` must not be read as if they
did; the static sort already reaches 0.936 on that easy component. (2b) This repo ALREADY ships
frame-change predictors for that exact target (`arc_frame_change_predictor`'s
`FrameChangeScorer`, `LiveActionEffectScorer`, `GroundTruthValidatedFrameChangeScorer`,
`BehaviorActionPrior`, `ActionEffectExpansionPrior`); they are NOT run as arms
(`FrameChangeScorer` needs a trained `SmallFrameChangeCNN` and `models/` contains no
frame-change checkpoint -- checked at runtime; the ground-truth-validated scorer is already
DOWNSTREAM of this router, not a comparable offline arm) and the artifact says so in
`baselines_not_run_and_why`. Consequently "the incumbent carries ZERO signal" is scoped to the
V3 ROUTER, never to this project's frame-change capability. (3) Offline AUROC licenses NOTHING
(exp4545). (4) `observe_click_outcome` has no live caller yet -- that site is in
`arc_competition_agent.py`. (5) Even flag-on, the router fully owns click order only where
`GroundTruthValidatedFrameChangeScorer` is unvalidated (measured: bp35/lp85/su15 yes;
tn36/r11l no). (6) Only 1 of the 2 requested smoke games contributes any state (vc33's
frame-change base rate is 1.000, so every state is excluded for having no negative), so every
`pooled_*` value is a SINGLE game's number -- now flagged `single_game_not_pooled: true`.

A MEASURED CORRECTION this experiment made to its own design: the incumbent's per-state CONSTANT
score reaches an ACROSS-state AUROC of 0.3105 from cross-state base-rate variation alone while
being provably 0.500 within-state, so an across-state band check on the blind arm would have
rejected a valid run. Both the metric of record and the label-validity check are therefore
within-state.

NOT DONE (operator follow-ups): the full multi-game corpus run, the live A/B on banked levels
stratified by `frame_diff_ground_truth_validated`, and the one-line agent-side observation hook.

#### SCENARIO-ARC-FCP-5904-COORDINATE-BLINDNESS-IS-REPAIRED
- **WHEN** >= 20 distinct click candidates on the same frame are scored by the router
- **THEN** with the flag OFF the router SHALL produce exactly ONE distinct click score and SHALL return the input order unchanged (locking in live parity and catching an accidental default flip); and with the flag ON and the online head fitted, the router SHALL produce more than one distinct click score and SHALL reorder the clicks.

#### SCENARIO-ARC-FCP-5904-COLD-START-IS-A-NO-OP
- **WHEN** the flag is ON but the online head has not met its sample gate (>= 3 positives, >= 3 negatives, >= 8 total)
- **THEN** every candidate's score SHALL equal its flag-OFF score exactly, `proba` SHALL be exactly 0.5, and the ranking SHALL be the wrapped router's own ordering.

#### SCENARIO-ARC-FCP-5904-EPISODE-ISOLATION
- **WHEN** one router instance is reused across two frames carrying different `(game_id, guid)` pairs (the `scripts/arc_leaderboard_eval.py` shape, which module-caches ONE router for a whole sweep) and outcomes are observed only on the first
- **THEN** the second frame's discriminator SHALL be unfitted and SHALL return exactly 0.5, and the online-state store SHALL retain at most `max_episodes` keys -- so no value transfers across games or episodes.

#### SCENARIO-ARC-FCP-5904-PER-FRAME-NOT-PER-CANDIDATE
- **WHEN** a single `rank()` call scores N click candidates with the flag ON and the head fitted
- **THEN** the number of `connected_color_blobs` invocations SHALL NOT scale with N, and `blob_topology` SHALL NOT appear in the featurization module's source at all.

#### SCENARIO-ARC-FCP-5904-SATURATION-GUARD
- **WHEN** the online head is fitted on a near-constant feature column (e.g. 19 zeros and one 1e-6) and then scores an out-of-range value for that column
- **THEN** the returned probability SHALL be strictly between 0 and 1 (NOT exactly 1.0, the measured failure of the `std + 1e-8` form) and the clip SHALL be counted in `stats()['saturation_clips']`.

#### SCENARIO-ARC-FCP-5904-BLIND-ARM-IS-A-LABEL-VALIDITY-CHECK
- **WHEN** the stage-1 diagnostic reports the BLIND arm's AUROC
- **THEN** a BLIND AUROC outside [0.45, 0.55] SHALL be reported as `label_validity_check: invalid` (a coordinate-blind scorer cannot beat chance on a click-dependent label, so an out-of-band value means the label leaks) and in that case NO arm SHALL be interpreted as a pass or a fail; ~~the pre-registered treatment gate SHALL be COORD AUROC >= 0.60 AND COORD - BLIND >= 0.10~~ (struck 2026-07-24: retained as the `coordinate_blindness_repair_check`, NOT as a gate -- see SCENARIO-ARC-FCP-5904-INFORMATIVENESS-GATE), containing no conjunct that asserts an assumption about another arm's own value.

#### SCENARIO-ARC-FCP-5904-INFORMATIVENESS-GATE
- **WHEN** the stage-1 diagnostic evaluates whether the coordinate-aware head is worth advancing to a live A/B
- **THEN** the artifact SHALL NOT emit a `pre_registered_gate` key; it SHALL emit `coordinate_blindness_repair_check` (the original expression verbatim, with an `also_passed_by_arms` list computed by applying that same expression to EVERY arm, so the fact that the shipped uninformative `RandomCandidateRouter` and the incumbent `static_salience` sort also clear it travels with the number) and `informativeness_gate` (`within_state_coord_auroc >= 0.60` AND the lower bound of a paired within-state bootstrap CI95 on `coord - static_salience` strictly above 0, for every contributing game), marked `added_post_hoc_after_review: true`; `gate_of_record` SHALL name `informativeness_gate`, and the `honest_verdict` SHALL name BOTH halves when the repair check passes while the gate of record fails (defect repaired, gain over the incumbent unestablished).

#### SCENARIO-ARC-FCP-5904-HEADLINE-DELTA-CARRIES-UNCERTAINTY
- **WHEN** the artifact reports `honest_comparator` (`coord - static_salience`, the only decision-relevant figure)
- **THEN** it SHALL carry a paired within-state bootstrap `delta_ci95` and `fraction_replicates_le_zero` (resampled WITHIN each scored state, both arms on the same resample) alongside a measured noise floor -- the spread of the shipped uninformative `RandomCandidateRouter`'s own within-state AUROC over `RANDOM_NOISE_SEEDS` seeds on the same corpus -- and a bare point estimate SHALL NOT be reported as the headline. Any `pooled_*` value derived from a single contributing game SHALL be flagged `single_game_not_pooled: true` with `ci95: null`.

#### SCENARIO-ARC-FCP-5904-HARD-SLICE-IS-OPERATIVE
- **WHEN** the stage-1 diagnostic reports the hard slice (the arm whose purpose is to strip out the trivial "did anything happen at all" component)
- **THEN** the slice SHALL be `levels_up` conditioned on `changed` -- NOT the pooled `(changed OR levels_up)` label, which makes every row in the slice a positive and the AUROC undefined by construction -- and it SHALL report `n_hard_pos` and `n_hard_neg` both non-zero together with coord, static-salience and shipped-random values, whether or not they flatter the treatment. The artifact SHALL also report whether the pooled label is MEASURED identical to frame-change (`n_levels_up_without_change`, `label_is_measured_identical_to_frame_change`) and SHALL name the frame-change baselines it did not run, with the reason (`baselines_not_run_and_why`).

#### SCENARIO-ARC-FCP-5904-ONLY-CLICK-POSITIONS-CHANGE
- **WHEN** the flag is ON, the head is fitted, and the candidate list interleaves non-click actions among clicks
- **THEN** every non-click candidate SHALL occupy exactly the index it held in the wrapped router's own ordering (a stable partition over the click slots), the clicks SHALL be permuted only among those slots, and a `weight` of 0.0 SHALL reproduce the base ordering exactly -- so a live A/B varies click order and nothing else.

### REQ-ARC-FCP-5927: Powered Coordinate-Router Progress Qualification

Experiment 5927 SHALL qualify the coordinate-aware click router on a frozen
multi-game candidate corpus before any live default-on A/B. The experiment SHALL
NOT select a level-solve target, SHALL NOT update `ops/arc_solve_registry.yaml`,
and SHALL NOT claim level credit. Its `solve_provenance` SHALL be
`development_proxy`, because this is an offline qualification over public games.

Before fitting, the experiment SHALL run a registry precheck for every selected
game, hash Exp5904, Exp5758, source frames/actions, feature/router code, the
registry, protected files, and the output path, and record deterministic seeds,
disk/RAM receipts, `cross_game_checkpoint_loaded=false`,
`online_within_game_only=true`, and an atomic checkpoint/resume receipt. The
corpus SHALL freeze its row and label contract before scoring, separating raw
frame change, UI animation-only change, state novelty, and verifier-validated
progress. The primary hard/progress slice SHALL be
`validated_progress` conditioned on raw frame change, and it SHALL require at
least 30 positive rows; otherwise the artifact SHALL stop as
`complete_underpowered:` without promotion.

The measurement SHALL compare five controls on the same rows: coordinate-aware
online features, static salience, coordinate-blind action id, step index, and a
seeded deterministic random control. Metrics SHALL include within-state and
leave-state-out AUROC receipts, a paired interval for coordinate minus static
salience on the powered hard/progress slice, random-control sanity, leakage
checks, and old-path regression receipts. The experiment SHALL never load a
cross-game value checkpoint and SHALL never use future outcome labels as current
features.

`OnlineClickTargetRouter` SHALL expose a default-off `observe_click_outcome`
update hook that records only the later validated outcome of a committed click
action into bounded within-game state. It SHALL support reset, rollback,
delayed outcome, duplicate outcome, missing outcome, replay, and cross-game
isolation checks, and the submitted router loader SHALL make this hook reachable
while preserving default-off old-path behavior.

Experiment 5927 SHALL write
`results/experiment_5927_coordinate_router_progress_qualification.json` with
bare top-level fields `status`, `preconditions_checked`,
`registry_precheck_receipt`, `games_states_rows_and_label_manifest`,
`solve_provenance`, `no_level_solve_or_registry_update`,
`cross_game_checkpoint_loaded`, `online_within_game_only`,
`coordinate_static_blind_step_and_random_controls`,
`frame_change_vs_validated_progress_receipts`,
`hard_progress_positive_count_and_power_gate`,
`within_state_and_leave_state_out_metrics`,
`coordinate_over_static_delta_and_interval`, `random_control_sanity`,
`observe_click_outcome_contract_and_tests`,
`reset_rollback_delay_duplicate_missing_and_replay_matrix`,
`cross_game_isolation_and_leakage_checks`, `default_enabled`,
`old_path_regression_receipt`, `protected_files_unchanged`,
`coordinate_router_progress_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Required field principles:

- `solve_provenance`: principle "use `development_proxy`; this offline qualification receives no level credit."
- `cross_game_checkpoint_loaded`: principle "must be bare false."
- `online_within_game_only`: principle "the router may learn only from the current game's committed outcomes."
- `hard_progress_positive_count_and_power_gate`: principle "at least 30 validated-progress positives are required before interpreting coordinate-vs-static deltas."
- `coordinate_over_static_delta_and_interval`: principle "promotion requires an interval-separated gain over the incumbent static salience sort, not just over a blind constant."
- `default_enabled`: principle "must be bare false unless the full preregistered promotion gate passes, and the task still may not mutate the default."
- `coordinate_router_progress_ready_score`: principle "emit bare 1.0 only for at least 30 hard positives, interval-separated coordinate gain over static salience, no leakage, random sanity, hook correctness, and no old-path regression."
- `inference_substrate`: principle "use `deterministic_offline_arc_development_proxy_no_llm`."
- `verifier_is_oracle`: principle "true only for environment progress, action legality, and exact replay labels."
- `honest_verdict`: principle "use `complete_ready:`, `complete_underpowered:`, `retired:`, or `blocked:`."

#### SCENARIO-ARC-FCP-5927-POWERED-PROGRESS-CORPUS
- **WHEN** Exp5927 freezes the candidate corpus
- **THEN** the manifest SHALL report selected games, state count, row count,
  raw frame-change count, UI-animation-only count, state-novelty count,
  verifier-validated progress count, and the hard/progress positive count; if
  that hard positive count is below 30, the artifact SHALL be
  `complete_underpowered:` with ready score `0.0`.

#### SCENARIO-ARC-FCP-5927-CONTROLS-AND-LEAKAGE
- **WHEN** Exp5927 scores the frozen rows
- **THEN** coordinate, static-salience, blind, step-index, and seeded-random
  controls SHALL be evaluated on within-state and leave-state-out splits, the
  coordinate-minus-static interval SHALL be paired on the same rows, random
  control scores SHALL be distinct and near chance, no cross-game checkpoint
  SHALL be loaded, and future outcomes SHALL NOT appear in current-row features.

#### SCENARIO-ARC-FCP-5927-COMMITTED-OUTCOME-HOOK
- **WHEN** committed click outcomes are delayed, duplicated, missing, rolled
  back, replayed, reset, or crossed between games
- **THEN** only the first matching later outcome for the same committed
  click and episode SHALL update the online discriminator; reset and rollback
  SHALL remove pending state; missing, duplicate, replayed, and cross-game
  outcomes SHALL be rejected without samples.

#### SCENARIO-ARC-FCP-5927-NO-PROMOTION-WITHOUT-GATE
- **WHEN** any promotion precondition is absent -- fewer than 30 hard positives,
  a coordinate-over-static interval that includes zero, label leakage, random
  control failure, hook failure, old-path regression, or default-on mutation
- **THEN** `default_enabled` SHALL remain bare `false`,
  `coordinate_router_progress_ready_score` SHALL be `0.0`, and the honest
  verdict SHALL not make a level/solve claim.

### REQ-ARC-FCP-5699-41: The Induce Sampler SHALL Send A Repetition Penalty, And A Mechanically Defective Engine SHALL Be Re-Asked Rather Than Accepted

**Written 2026-07-31, AFTER the code shipped.** This requirement id was cited by
`python/carnot/agentic/arc_executable_world_model.py` and by
`tests/python/test_arc_induce_repeat_penalty_and_reask_2026_07_31.py` from commit `d20533033`,
but no such requirement existed anywhere in `openspec/` — the highest id in this chain was
`-39`. Shipped code and its test carried traceability to a requirement that was never written,
which the 2026-07-31 adversarial review caught. The id is written here rather than renumbered
in the source so the two existing citations stay valid; `-40` is unused and stays unused.

**WHAT FAILED.** The shipped induce path sent NO repetition control, so `llama-server` applied
its default of `1.0` — i.e. none, read back from a running server's own `/props`. The dominant
induce failure is a decode-level repetition loop: the model reaches an impasse part-way through
`engine()`, emits the same comment line until `n_predict` is exhausted, and never writes a
`return`. ft09's live engine is 1112 of 1144 lines of duplicated comment. Separately, the shipped
`generate()` ACCEPTED a mechanically defective candidate in **22 of 36** attempts — it parses and
defines the required symbols, so parse-level acceptance passes it through to the semantic gate,
which then records "predicted wrong" for what is actually broken code.

**THE REQUIREMENT.** On an ENGINE code-only induce call the system SHALL:

1. Send `repeat_penalty` and `repeat_last_n` on the `/completion` payload, both env-overridable,
   with `repeat_penalty = 1.0` restoring the pre-2026-07-31 payload BYTE-FOR-BYTE (llama.cpp
   treats 1.0 as the identity, so the off switch must omit the fields entirely rather than send
   a small value).
2. Check the candidate for MECHANICAL defects through
   `arc_engine_static_validation.validate_engine_code` before accepting it, and on a defect
   re-ask ONCE with a block that names NOTHING about what went wrong.
3. Never convert an accept into a hard failure. The re-ask SHALL be spent only while an attempt
   remains; on the last try the defective candidate SHALL be accepted exactly as before.
4. Keep the two interventions INDEPENDENTLY disableable, because they carry very different
   weight — the penalty carries 11 of the 13 paired wins and the re-ask 2.

**SCOPE — the shipped condition SHALL equal the measured condition.** Both interventions are
armed on `codeonly_eligible AND "engine" in required`, and on nothing else. `refactor()` is a
reasoning task on a different prompt shape. The focused goal-only call in `_split_induce`
(`required=("is_level_complete",)`) is ALSO `codeonly_eligible`, and as first wired it inherited
the penalty although Phase 1 sent ENGINE prompts only; that was narrowed 2026-07-31 after the
adversarial review, in the conservative direction — with no measurement in either direction the
goal-only call's correct state is the long-standing no-penalty baseline every banked result was
produced under.

**WHAT IS CLAIMED: VALIDITY AND COST.** Attempts producing a mechanically-usable engine
**13/36 → 22/36**; wall clock **47.2 s per attempt against 100.3 s** (65.2 s charging the
re-ask); calls hitting the 4096-token cap **20/36 → 2/36**; `missing_return` **13 → 2**.

**WHAT IS NOT CLAIMED: QUALITY.** Restricted to the games whose held-out set contains a CHANGING
transition, strict out-of-sample quality is **4/36 in both arms — identical**. The whole
strict-funnel gain (2/6 games → 3/6) is ft09, which has zero held-out changing rows and passes
only on the weaker no-op fallback. On tu93 and sc25 NO arm ever produced a correct engine at any
budget. Nothing here fixes induction.

**THE INFERENCE SHALL BE REPORTED CLUSTERED BY GAME.** The 36 pairs are 6 games × 6 resamples of
ONE prompt each; resamples of one prompt are not independent draws from the population the claim
generalizes to, which is NEW GAMES. Reporting the within-prompt p alone is pseudo-replication.

#### SCENARIO-ARC-FCP-5699-41-1: The penalty reaches the ENGINE induce payload and nothing else
- **WHEN** `generate()` is called with `codeonly_eligible=True` and `engine` in `required`
- **THEN** the payload SHALL carry `repeat_penalty` and `repeat_last_n`; and WHEN it is called
  with `codeonly_eligible=False`, or with `codeonly_eligible=True` but `engine` absent from
  `required` (the goal-only call), the payload SHALL carry NEITHER field.

#### SCENARIO-ARC-FCP-5699-41-2: A defective engine is re-asked, not accepted
- **WHEN** a candidate parses and defines the required symbols but carries a mechanical defect,
  and an attempt remains
- **THEN** it SHALL be rejected and re-asked exactly once, with a block naming nothing about the
  defect; a CLEAN candidate SHALL never trigger a re-ask.

#### SCENARIO-ARC-FCP-5699-41-3: The gate can never break the path it guards
- **WHEN** the defect budget is exhausted, or the validator itself raises for any reason
- **THEN** `generate()` SHALL degrade to its pre-2026-07-31 accept-everything behaviour and
  return the candidate, NEVER a hard failure and never an exception. Removing the
  attempt-remains guard is a mutation this requirement's tests SHALL kill.

#### SCENARIO-ARC-FCP-5699-41-4: The off switches are exact
- **WHEN** `CARNOT_ARC_INDUCE_REPEAT_PENALTY=1.0` is set
- **THEN** the payload SHALL be byte-identical to the pre-2026-07-31 one (fields omitted, not set
  to 1.0); and WHEN `CARNOT_ARC_INDUCE_DEFECT_REASKS=0` is set, the defect re-ask SHALL be
  disabled WITHOUT disabling the penalty. A malformed value SHALL fall back rather than raise.

#### SCENARIO-ARC-FCP-5699-41-5: A quality claim SHALL NOT be made from this change
- **WHEN** any artifact, capstone or note cites this requirement's measurement
- **THEN** it SHALL report the game-clustered inference as primary (p = 0.125, 4 games better,
  0 worse, 2 tied) with the within-prompt p = 0.049 labelled as such; SHALL state the
  change-gradable-only strict counts (4/36 vs 4/36); and SHALL NOT claim the change improves
  engine quality, the funnel of games reaching the policy, or banked levels.

## Implementation Status (REQ-ARC-FCP-5699-41)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-ARC-FCP-5699-41 | `python/carnot/agentic/arc_executable_world_model.py` — `_INDUCE_REPEAT_PENALTY` (1.1) / `_INDUCE_REPEAT_LAST_N` (256) / `_INDUCE_DEFECT_REASKS` (1) / `_INDUCE_PLAIN_REASK_BLOCK`, applied in `LocalGGUFProposer.generate()` under `_engine_induce_call` (`codeonly_eligible AND "engine" in required` — the same condition as `_defect_check_on`, named separately so narrowing one cannot move the other); defect check via `LocalGGUFProposer._engine_defects` → `arc_engine_static_validation.validate_engine_code` (see REQ-ARC-WMTE-6052 CORRECTION #2, which records that this wiring superseded that module's standalone status). `repair_prompt_block()` deliberately UNWIRED: defect-naming text measured p = 1.000 against the neutral block. Env: `CARNOT_ARC_INDUCE_REPEAT_PENALTY` (1.0 = byte-identical off), `CARNOT_ARC_INDUCE_DEFECT_REASKS` (0 = off, independent of the penalty). | `tests/python/test_arc_induce_repeat_penalty_and_reask_2026_07_31.py` — 15 tests. **Mutation-proven:** the original 13 killed 3 mutations including removal of the `attempt < tries - 1` safety guard (which the first draft got WRONG — the `continue` fell through to the content-failure return, caught by the mutation check and not by review); the 2 added 2026-07-31 pin the narrowed scope from BOTH sides and kill "widen the penalty back to `codeonly_eligible`" and "drop the penalty entirely". **Measurement:** `results/arc_induce_confirm_20260731/confirm_scored.json` — 36 attempt-matched pairs, 6 games, gemma-4-31B-it Q4_K_M, CUDA build proven from `/proc/<pid>/exe` + 21416 MiB per-PID VRAM on distinct bus IDs, one server process per GPU shared by both arms, scored OUT-OF-SAMPLE against a proven held-out split (`split.json`). Usable 13/36 → 22/36 (within-prompt p = 0.049; **game-clustered p = 0.125**, permutation p = 0.125, 4 better / 0 worse / 2 tied, min reachable clustered p 0.031); strict quality on change-gradable games 4/36 both arms; cost 100.3 s → 47.2 s per attempt. **Config caveat:** measured at `CARNOT_ARC_INDUCE_N_CTX=32768`, while the shipped `_default_induce_n_ctx()` still returns 81920 — the COST half is completion-budget sensitive and is conditioned on 32768 until the default is fixed or the treatment re-measured at it. **Live pre-flight (`results/arc_phase3_preflight_20260731/`) REFUSED a banked-levels grid:** the wired change produced 17 distinct engines across 17 cells and byte-identical action traces in 4 of 4 comparable A/B pairs, so no endpoint downstream of the actions could reach alpha (minimum reachable p = 1.0). The re-ask half fired 0 times in all 17 live cells — Phase 1's 22-of-36 defective-accept rate did not reproduce live, which is "the tier never fired", not "the tier did not help". |
