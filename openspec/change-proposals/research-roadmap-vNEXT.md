# Research Roadmap vNEXT: V475 Hidden-State Verifier On Confirmed Headroom, MAP Landmark Pre-Stage For The Enumeration Wall, DiffusionGemma Unblock Attempt

**Milestone:** `2026.07.475`
**Status:** Pre-staged next milestone (hand-authored by an outer-loop Claude planning session, per the
Pre-Staged Roadmap Convention)
**Prepared:** 2026-07-03
**Predecessor:** `2026.07.474`
**Execution manifest:** `research-roadmap-next.yaml`

## Why this plan was hand-authored instead of left to the automated planner

The automated conductor's own attempt to plan and activate `.475` stalled for approximately 2.5 hours
(`ops/conductor-log.md`, 2026-07-03 03:04-05:25 UTC): ~65 consecutive `Activation REFUSED` entries, all
`exclusion-manifest: 1 HARD violation(s); first: SCOPE_MATCHED_PRIOR_FAILURE`, followed at 06:14 UTC by a
`Plan next milestone` failure (`Codex CLI error: Error: Reached max turns (50)`). The most probable root
cause, based on direct inspection of `ops/exclusion_manifest.yaml` and the Exclusion-Manifest Cross-Check
discipline: a draft task's scope legitimately matched a `.473`/`.474` prior-failure artifact (most likely
the trajectory-enumeration-wall lineage -- `exp5175`/`exp5176`, both null/blocked -- or the DiffusionGemma
lineage -- `exp5173`, blocked) without a `prior_failures:` block carrying all four mandatory sub-fields
(`experiment_id`, `verdict`, `addressed_by`, `retire_if_same_verdict`), which the Layer-2
`exclusion_manifest_lint.py` HARD-refuses at activation. This has happened before (milestones `.222`-`.227`
per the roadmap-authoring instructions) and is a known, avoidable failure mode.

This plan was therefore authored with two disciplines applied uniformly and defensively: **(1)** every task
that continues, extends, or is thematically adjacent to a `.473`/`.474` non-clean-success verdict carries an
explicit `prior_failures:` block with all four sub-fields, even where the connection is arguable, because the
cost of an unnecessary block is trivial and the cost of another multi-hour activation stall is not; **(2)**
no task title or prompt text reproduces any `blocked_patterns:` string from `ops/exclusion_manifest.yaml`
verbatim (the full retired-scope list was read in full before drafting a single task).

Both directories that carry this repository (`/home/ianblenke/github.com/ianblenke/carnot`, where this plan
was authored, and `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`, which `carnot-conductor.service` reads
from) were at identical HEAD (`75bc15756`) when this was written. This plan's files live in the former; they
need to reach the latter (push/pull, or an equivalent sync) before the live conductor will see them -- that
is an explicit operator decision, not performed automatically here.

## Executive Summary

`.474` closed three open threads and opened one new one. It **hardened** the ARC oracle-distinct
Set-Encoder-vs-vote win past the CLT sample-size floor (`exp5171`, n=30, gate passed, `gate_passed: true`),
**retired** the seven-milestone-null PHASE D external-text-scorer program (LoRA-EBM/uPRM/EBRM vs.
self-consistency on off-ARC corpora -- `exp5170`, 28 artifacts, `phase_d_external_text_scorer_retired_exp5163_v474`
in the exclusion manifest) while explicitly preserving hidden-state/internal-representation verifiers as an
exempted, architecturally distinct mechanism class, and **diagnosed but did not close** the ARC
trajectory-enumeration wall (`exp5175`: a relational-mask pruner correctly prunes edges but
`states_expanded` is unchanged and zero levels bank -- pruning a fixed frontier does not help when the
winning trajectory is never enumerated into that frontier in the first place). It also left the
`hidden_state_verifier_pilot` (`exp5178`) as an honest negative: on a small (n=6 questions) pilot, even on a
headroom-present slice (oracle=1.0 vs. SC=0.333), a naive trained-centroid hidden-state probe lost to tuned
self-consistency, and it left `exp5173`'s DiffusionGemma energy-guided-diffusion pilot **blocked before any
measurement** by a two-GPU device-placement bug.

V475 makes three decisive moves, each directly informed by fresh literature (independently converged on
twice -- see `research-references.md`'s two separate `V475` sections, one from `exp5172`'s in-milestone
sweep, one from this session's three-agent sweep):

1. **Retry the hidden-state verifier with a different design, on a different (headroom-confirmed) corpus,
   not just a bigger version of the same pilot.** `exp5178`'s failure has two plausible causes conflated:
   too small a sample (n=6 questions) and too weak a probe (a naive trained centroid, not a proper trained
   classifier). PHSV (arXiv:2504.05419) supplies a concrete, reproducible probe recipe (last-token/last-layer
   hidden state, chunk-split reasoning, small trained MLP, AUROC 0.75-0.91 on local open-weight models).
   Critically, per arXiv:2512.02304's headroom formalism (independently confirming what `.474`'s own PHASE D
   retrospective found empirically), a verifier can only beat self-consistency where real headroom exists --
   so V475 targets the MMLU-Pro pool that `ops/known-issues.md`'s 2026-07-01 headroom check already
   confirmed has real headroom (`oracle_at_k=0.350` vs. `sc_vote=0.075`-`0.269`), not `exp5178`'s
   near-ceiling MuSR slice. Two free baselines (self-certainty, arXiv:2502.18581; CLUE non-parametric
   clustering, arXiv:2510.01591) are mandatory control arms, so a trained probe's extraction overhead has to
   earn its keep against zero-training alternatives.

2. **Attack the diagnosed wall with a structurally different mechanism, not a bigger pruner.** `exp5172`'s
   own MAP deep-read (arXiv:2605.13037) already specified the exact falsifiable gate: on CD82/SK48/SP80, run
   pruner-only, map-only, and map-plus-pruner under the same expansion budget and reproduction gate; promote
   MAP only if map-only or map-plus-pruner banks a level that pruner-only does not. A pre-search landmark
   stage changes WHAT gets proposed into the search frontier; a frontier pruner (already tried) only reorders
   or filters a frontier that structurally excludes the winner. These are complementary, not redundant, per
   the literature's own comparison, and the exact same recommendation was independently reached THREE times
   now (`exp5172`'s sweep, the `.474` design doc's own Phase B contingency note, and this session's fresh
   three-agent sweep).

3. **Diagnose before retrying DiffusionGemma, with a hard time-box and an honest bail-out.** The blocking bug
   (`ValueError: Some modules are dispatched on the CPU or the disk before forward`, then the underlying
   `Tensor.item() cannot be called on meta tensors` forward error) has now resisted two attempts across
   `.474` (4 sub-attempts: wrong auto class, two `device_map=auto` retries with different model classes, one
   with explicit `max_memory={0:24GiB,1:24GiB}` -- all failed identically). V475 tries specific, genuinely new
   mitigations (explicit non-`auto` device maps, `_no_split_modules` inspection, non-quantized/8-bit
   fallback) inside a fixed diagnostic budget, and if still blocked, writes an honest
   `blocked_diffusiongemma_meta_tensor_bug_unresolved_v475` rather than burning the pilot task's turns on a
   load that has already failed five times.

PHASE D's retirement is respected throughout: nothing in this milestone re-proposes an external-text-scorer
construction on an off-ARC corpus. The one overdue MANDATORY priority (`ops/known-issues.md`: wiring
`scripts/retro_timing_fallback.py` into the conductor, pending 4+ milestones -- `.469`, `.473`, `.474`, now
`.475`) is picked up as a **patch-prep** task rather than a live edit, because every task this roadmap emits
ends with the standard "do not modify `scripts/research_conductor.py`" instruction, and that file is exactly
what needs the two-line change; this plan produces a reviewed, tested, ready-to-apply patch instead, and
flags the actual `git apply` + wiring step as an operator/outer-loop action outside the roadmap-task sandbox.

`_bmad/architecture.md` was last reconciled 2026-05-16 (48 days before this plan) -- past the 30-day
freshness threshold CLAUDE.md flags. It predates the entire ARC-AGI-3 pivot, the PHASE D lifecycle
(commit through retirement), and the verifier-tier changes since May. This is flagged to the operator
directly and picked up as one of the two reserved infrastructure slots.

## What V474 Proved (verified against primary artifacts, not paraphrased)

| Task | Verdict (verbatim key) | Key numbers |
|---|---|---|
| `exp5168` (archive .473->.474) | `complete_archive_473_closed_474_active_runtime_clean_exp5161_unquarantined` | 1 real win, 2 nulls, 1 blocked gate carried forward accurately |
| `exp5169` (adversarial-verify QD false-positive fix) | `complete: exp5156_resolves_clean_qd_citation_scope_fixed_warn_only_not_quarantine` | 4 archive artifacts un-flagged, 2 QD artifacts newly flagged in backfill |
| `exp5170` (PHASE D retirement) | `complete: phase_d_external_text_scorer_scope_retired_and_hidden_state_exception_preserved` | 28 artifacts examined; best point estimate exp5031 delta=+0.08 CI=[0.0,0.165]; terminal exp5163 delta=+0.025 CI=[-0.125,0.175] |
| `exp5171` (Set-Encoder hardening, n=24->n=30) | `success_arc_set_encoder_cross_corpus_gate_passed_n30` | delta=0.5, CI95=[0.333,0.667], identical across 5 seeds, `gate_passed: true` |
| `exp5172` (SOTA ingestion) | `complete: map_deep_read_recommends_map_pre_stage_if_phase_b_pruner_stalls` | MAP (2605.13037), Theoria, AutoMem, Unified Energy, Prism identified |
| `exp5173` (DiffusionGemma pilot) | `blocked_diffusiongemma_meta_tensor_bug_unresolved` | Never launched; `arm_rows=[]`; 4 distinct load-path sub-attempts, all failed |
| `exp5174` (GAP-LIVE-INTEGRATION reconciliation) | `complete: ... provenance audit finds 4/24 ... live-self-discovery vs 20/24 development-proxy` | Stale claims corrected; router/DSL now imported, `target_levels=3` |
| `exp5175` (relational-mask-pruner A/B) | `complete_relational_mask_pruner_prunes_edges_but_states_expanded_unchanged...` | Edges pruned {cd82:358, sk48:22807, sp80:0}; `states_expanded` unchanged on all 4 games; 0 levels banked |
| `exp5176` (level-up attempt) | `complete_blocked_no_validated_lever_from_b1_b2_zero_levels_banked` | 0 levels banked |
| `exp5177` (GAP-4 scale-up) | `complete_gap4_scaleup_v474_n62_of_target180_floor_not_crossed_scale_up_recommended` | n=60->62; still not significant vs. min-6-discordant-wins floor |
| `exp5178` (hidden-state verifier pilot) | `complete_hidden_state_verifier_ties_tuned_sc_accuracy_point_lower_efficiency_loses_to_sc...` | hidden_verifier accuracy 0.0 vs tuned_sc 0.333, delta=-0.333, CI=[-0.667,0.0]; n=6 questions, 48 candidates |
| `exp5179` (hardware continuity) | `complete_hardware_continuity_board_timing_kv260:reachable_gatemate:blocked...polarfire:reachable...` | 2/3 boards reachable; GateMate blocked on DirtyJTAG IDCODE 0x20000001 |
| `exp5180` (capstone) | `complete: v474 reconciled with no flagged headline artifacts after live verification...` | `reproducible_total_levels=69/24` (flat, 3rd+ consecutive milestone) |

No CRITICAL `flagged_adversarial` artifacts in `.474`'s own output. `exp5163` (inherited from `.473`) still
carries a live CRITICAL TAUTOLOGY flag and must not be cited as a clean number -- moot for V475 since PHASE D
(the scope `exp5163` belongs to) is retired and none of V475's tasks cite its numbers.

## Current registry / gate state (read directly, not inferred)

- `ops/arc_solve_registry.yaml`: `reproducible_total_levels=69`, `reproducible_total_games=24`, flat for 3+
  consecutive milestones. The wall is diagnosed (enumeration, not selection) but not yet closed.
- `ops/verifier_gaps.md`: GAP-4891 ("goal-induction REPRESENTATION beyond object/colour COUNTS") status
  `building` -- floor addressed (single-positive goal-energy), but "pruning alone does not close GAP-4891's
  enumeration wall; the next lever must generate/structure the [candidate pool] differently." GAP-4 (the
  same-shape rule-application discriminator, separately the `exp5161`/`exp5177` forward-protocol pilot) is
  `scale_up_recommended`, n=62 of a ~180 target floor.
- `ops/exclusion_manifest.yaml`: PHASE D external-text-scorer construction (LoRA-EBM/uPRM/EBRM style vs. SC
  on off-ARC corpora) is terminally retired as of `.474` (`phase_d_external_text_scorer_retired_exp5163_v474`)
  with hidden-state/internal-representation verifiers, ARC oracle-distinct verifier work, and the FoVer
  production ensemble explicitly named as **outside** the retired scope. ARC first-contact
  candidate-generation exploration-signal tweaks (novelty/program-synthesis/energy-as-fitness QD) are
  separately retired (`generation_axis_exploration_signal_retired_exp5154_v473`) -- not touched by this plan,
  since MAP is a pre-search landmark stage on already-partially-solved games' deepening, not a first-contact
  exploration-signal tweak.
- `ops/north-star.md`: ARC-AGI-3 (accuracy + efficiency) remains the destination; the FoVer headline
  (AUROC 0.9131) and the G1-G4 publication gate remain fixed and MET (`paper_ready: true` per
  `scripts/publication_gate.py`, 2026-06-12). Neither is touched by V475 except a numeric-only
  technical-report sync task.
- `_bmad/architecture.md`: **Last Reconciled 2026-05-16 -- 48 days stale, past the 30-day threshold.**
  Flagged to operator; picked up as a reserved infrastructure task (`exp5189`).

## Phase design

### Phase 0 -- Transition
`exp5181`: routine `.474`->`.475` archive/activation. Codex, `operator_override` per the standing 2026-05-29
routine-transition authorization.

### Phase A -- DiffusionGemma unblock (diagnostic-first, time-boxed) and GAP-4 continuation
- `exp5182`: root-cause the two-GPU device-placement failure with genuinely new mitigations (not a repeat of
  `.474`'s four `device_map=auto` variants). Routed to `model: opus` (dual-GPU / device-placement debugging
  matches the pre-emptive Opus-routing criteria in the roadmap-authoring instructions) with a hard turn
  budget and an honest bail-out.
- `exp5183`: **gated on `exp5182`'s `diffusiongemma_loadable` field being `true`.** If unblocked, runs the
  actual energy-guided-diffusion pilot using the EDLM recipe (arXiv:2410.21357, windowed importance
  resampling) with an intrinsic-confidence-only control arm (VFScale, arXiv:2502.01989) to satisfy the
  Circularity/Oracle-Distinctness Discipline, and commit-position telemetry (arXiv:2606.14620) as a
  precondition for any guidance-helped claim.
- `exp5184`: continue the GAP-4 forward-protocol scale-up from n=62 toward the ~180-sample significance
  floor (`exp5161`->`exp5177` lineage), as far as the turn/wall-time budget allows honestly.

### Phase B -- Trajectory-enumeration wall: MAP landmark pre-stage
- `exp5185`: prototype the MAP-style map-then-act pre-stage on CD82/SK48/SP80, using the exact 3-arm
  falsifiable gate `exp5172` already specified (pruner-only vs. map-only vs. map-plus-pruner, same expansion
  budget, reproduction-gated). This is a structurally different mechanism from the pruner (builds landmarks
  BEFORE search changes what gets proposed, vs. the pruner which filters an already-fixed frontier after the
  fact) -- the literature explicitly frames them as complementary, not redundant attempts at the same fix.
- `exp5186`: **gated on `exp5185` validating a lever** (i.e., map-only or map-plus-pruner banking a level
  pruner-only did not). The ARC Level-Up Attempt Guarantee's mandatory >=1 attempt-per-roadmap floor.

### Phase C -- Hidden-state verifier v2
- `exp5187`: PHSV-style trained probe (last-token/last-layer, chunk-split, small MLP) on the MMLU-Pro
  headroom-confirmed pool (`oracle_at_k=0.350` vs. `sc_vote=0.075`-`0.269`), with self-certainty and CLUE
  non-parametric clustering as mandatory free-baseline control arms, and a layer-sweep (FEPoID,
  arXiv:2605.26366) if the extraction path supports it without excessive engineering cost. This satisfies
  research-program.md's continuous-self-learning mandate (a probe trained on the model's own accumulated
  correct/incorrect experience) as well as the open, non-retired hidden-state-verifier thread.

### Reserved infrastructure slots (per the Overdue-Priority Forcing Function's >=2-slot reservation)
- `exp5189`: `_bmad/architecture.md` reconciliation -- bring the 48-day-stale document up to date with the
  ARC-AGI-3 pivot, PHASE D's full lifecycle, the hidden-state-verifier program, and current hardware state.
- `exp5190`: `retro_timing_fallback.py` wiring **patch-prep** (the MANDATORY overdue priority, pending 4+
  milestones). Produces a reviewed, tested, ready-to-`git apply` patch plus a regression test that fails
  today and would pass once applied -- without touching `scripts/research_conductor.py` directly, per every
  task's standing constraint. The actual application is flagged as the next operator/outer-loop action.

### Hardware continuity (mandatory, 1 combined task covering all 3 attached boards)
- `exp5188`: KV260 (SSH+hash-verified workload, the near-terminal focus board per `ops/north-star.md` §3),
  PolarFire (SSH+hash-verified workload), GateMate (attempt to actually resolve the DirtyJTAG IDCODE miss
  that has now persisted across `exp5166` and `exp5179`, not just re-run the same `--detect` call). Routed
  to `model: opus` (hardware integration / dual-board debugging).

### Docs
- `exp5191`: numeric-only sync of `docs/technical-report.md` results tables (Set-Encoder n=30 hardening,
  any new V475 numbers) -- prose stays operator-curated per Public Documentation Discipline.

### Phase Z -- Capstone
- `exp5192`: milestone capstone, reconciling all of the above honestly (including any Phase A/B/C tasks that
  come back blocked -- this plan does not assume every lever lands).

## Dependency graph

```
exp5181 (archive/activate)
   |
   +-- exp5182 (DiffusionGemma diagnose+fix) --gated_on(diffusiongemma_loadable==true)--> exp5183 (pilot)
   |
   +-- exp5184 (GAP-4 scale-up)                                            [independent]
   |
   +-- exp5185 (MAP landmark pre-stage A/B/C) --gated_on(lever validated)--> exp5186 (level-up attempt)
   |
   +-- exp5187 (hidden-state verifier v2)                                  [independent]
   |
   +-- exp5188 (hardware continuity)                                       [independent]
   +-- exp5189 (architecture.md reconciliation)                            [independent]
   +-- exp5190 (retro-timing patch-prep)                                   [independent]
   +-- exp5191 (technical-report numeric sync)                             [independent]
   |
   +-- exp5192 (capstone, reads all of the above)
```

## Hardware requirements

| Task | Hardware | Notes |
|---|---|---|
| `exp5182`/`exp5183` | 2x RTX 3090 (CUDA) | DiffusionGemma's confirmed load path splits a 26B/4B-active MoE across both GPUs at 4-bit NF4; GPU 1 must be idle (checked via `nvidia-smi`) before attempting |
| `exp5184` | ARC live-submission stack (frozen: Qwen3.5-9B-MTP on iGPU) or cached candidate pool, per whatever `exp5161`/`exp5177` already used | Continue established methodology, do not re-derive |
| `exp5185`/`exp5186` | CPU (offline ARC arcade simulation) | No GPU required; `arc_solver_kit` reproduction gate runs on CPU |
| `exp5187` | 1x RTX 3090 or iGPU, GGUF-cached `gemma-4-26B-A4B-it-GGUF` | Matches `exp5178`'s target model for continuity/comparability |
| `exp5188` | KV260 (SSH), PolarFire (SSH), GateMate (USB DirtyJTAG) | No new hardware; continuity only |
| `exp5189`, `exp5190`, `exp5191`, `exp5192` | None (CPU, aggregation/doc work) | |

## Risk notes

- **DiffusionGemma may remain blocked a third time.** `exp5183` is gated specifically so this does not
  cascade -- if `exp5182` fails, `exp5183` is mechanically skipped (no wasted Sonnet/codex call), and nothing
  else in this roadmap depends on the DiffusionGemma thread.
- **MAP may not close the enumeration wall either.** `exp5172`'s own citation frames this honestly as
  "should be prototyped next," not "will work." The 3-arm falsifiable gate is designed so a null result
  (pruner-only wins, or nothing banks a level) is just as informative and reportable as a positive one, and
  `exp5186` is gated so a null `exp5185` does not force a doomed level-up attempt.
- **The hidden-state verifier v2 may also lose to self-consistency.** If a properly-trained probe on a
  confirmed-headroom corpus with free-baseline controls still loses, that is a strong, well-controlled
  negative worth reporting (and would suggest the mechanism-class hypothesis itself, not just the prior
  pilot's small N, needs revisiting) -- this is a legitimate, publication-relevant outcome either way.
- **The retro-timing patch-prep does not close the MANDATORY priority.** It reduces the remaining work to a
  reviewed `git apply` + smoke test, but the actual wiring still needs an operator or a differently-scoped
  outer-loop session with permission to edit `scripts/research_conductor.py`. Flagged explicitly rather than
  silently deferred again.

## Cross-references

- `ops/exclusion_manifest.yaml` -- the full retired-scope list this plan was cross-checked against
- `ops/conductor-log.md` (2026-07-03 03:04-06:14 UTC) -- the stall this plan works around
- `research-references.md` §"V475 Planner References" (from `exp5172`) and §"V475 Outer-Loop Planner
  References -- Session 2" (this session) -- the two independently-converged literature sweeps
- `ops/verifier_gaps.md` GAP-4891, GAP-4 -- the open gaps `exp5185`/`exp5184` address
- `ops/known-issues.md` -- the MANDATORY retro-timing priority (`exp5190`) and the MMLU-Pro headroom check
  (`exp5187`'s target corpus)
- `ops/north-star.md` §0, §3, §5 -- ARC-AGI-3 destination framing, hardware focus, verifier-moat reframe
- CLAUDE.md "Exclusion-Manifest Cross-Check Before Planning", "Failed-Experiment Rerun Discipline",
  "Circularity / Oracle-Distinctness Discipline", "ARC Live-Path Reachability Discipline", "Inference-Substrate
  Declaration Discipline", "Architecture Freshness Check"
