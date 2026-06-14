# Research Studying — Ranked Ideas for Future Experiments

**Purpose:** Claude (outer loop) continuously researches novel ideas from
online sources, ranks them by potential impact on Carnot's current state,
and queues the most promising into the next roadmap milestone. Codex (inner
loop) executes the current experiments.

**Updated:** 2026-06-11 (Exp 4081 SOTA ingestion mapped the .377 verifier-as-reward pivot).
**Current Focus:** Phase 1 ship-track is one external reproducer away. Paper-v6 narrowed per the 2026-05-23 Deep Think round; two retractions + one rescue + five-post operations/honesty blog series shipped. Conductor on `.282 with metamorphic repair-oracle audit and FR-11 attractor trace-memory stability as load-bearing tasks. Sweep infrastructure recovered 2026-05-24 after 8 days degraded.

## 2026-06-14 Exp 4203 - .389 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-as-reward-v390-2026-06-14.md`.

**Filtered track:** verifier-as-reward de-confounding, code-RLVR baselines,
process/self-distill math rewards, and cost-normalized verifier plus
self-consistency framing.

**Seed and fresh-pass candidates marked ingested:**
- Spurious Rewards, arXiv:2506.10947 - mapped to the mandatory non-Qwen base
  and same-generator random-label A-vs-B control.
- Spurious Rewards Paradox, arXiv:2601.11061 - mapped to the
  memorization-shortcut diagnostic.
- RLV-epsilon-R, arXiv:2601.04411 - mapped to TPR/FPR/Youden-J reporting.
- RLEF, arXiv:2410.02089; Aletheia, arXiv:2601.12186; and CodeScaler,
  arXiv:2602.17684 - mapped to code-RLVR baselines a positive result must beat.
- Self-Distilled RLVR, arXiv:2604.03128; CEPO, arXiv:2605.19436; and
  ThinkPRM, arXiv:2504.16828 - mapped to the math-process-reward fork after
  the de-confounding gate.
- Budget-aware discriminative verification, arXiv:2510.14913, and
  When To Solve/Verify, arXiv:2504.01005 - mapped to the hybrid verifier plus
  self-consistency cost-crossover framing.

Exp 4199 status mapped honestly: `blocked_gate_check_failed`; the A-vs-B
collection did not run because the upstream training-launched gate was false.

flagged_for_v390:
`non_qwen_same_generator_random_label_ablation_v390`.

Flagged for .390: `non_qwen_same_generator_random_label_ablation_v390`.

**Bottom line for the .390 roadmap:** run the non-Qwen same-generator
random-label A-vs-B replication before any math-process-reward fork.
## 2026-06-14 Exp 4192 - .388 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-efficiency-gap4-diffusion-v389-2026-06-14.md`.

**Filtered track:** DiffusionGemma verifier-guided test-time scale-up,
efficiency-moat LLM-judge comparator and cost normalization, plus the CEM
operator-authorization closure for the retired GAP-3 trained-content-energy
selector lineage.

**Seed and fresh-pass candidates marked ingested:**
- Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching,
  arXiv:2602.22871 - mapped to step-level DiffusionGemma guidance and stitching
  ablations.
- S^3 Stratified Scaling Search, arXiv:2604.06260 - mapped to the strongest
  `.389` DiffusionGemma verifier-guided denoising-search target.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped as the self-guided particle
  control for the DiffusionGemma scale-up.
- Tuning LLM Judge Design Decisions for 1/1000 of the Cost,
  OpenReview:cve4NOiyVp / arXiv:2501.17178 - mapped to tuned LLM-judge
  comparator and cost-normalized moat accounting.
- When To Solve/Verify, arXiv:2504.01005 - mapped to the fixed-budget
  solve-versus-verify normalization bar.
- ThinkPRM, arXiv:2504.16828 - mapped as the high-quality but expensive
  process-verifier comparator.
- CEM, arXiv:2510.20607 - re-flagged to the operator only:
  `operator_authorization_required=true`, `auto_activation_recommended=false`,
  retirement marker `gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

cem_operator_authorization_flag:
`source_id=2510.20607; operator_authorization_required=true; auto_activation_recommended=false; retirement_marker=gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

flagged_for_v389:
`s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

Flagged for .389: `s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

**Bottom line for the .389 roadmap:** run the S^3-style DiffusionGemma
verifier-guided denoising search first, with Reward-Guided Stitching and
Self-Rewarding SMC as ablation/control arms and judge-cost normalization around
the efficiency moat. Keep CEM on the operator surface only; do not activate it
until operator authorization is granted and gate-1R is passed.
## 2026-06-14 Exp 4180 - .387 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-moat-gap3-diffusion-v388-2026-06-14.md`.

**Filtered track:** verifier-as-reward, sanitized headroom, accuracy-and-cost
moat framing, GAP-3 learned ARC energy, TRM vote/headroom decomposition, and
DiffusionGemma guidance for the `.388` handoff.

**Seed and fresh-pass candidates marked ingested:**
- Unsolvability Ceiling, arXiv:2605.07395 - mapped to the A1 headroom-gate
  sanitization already applied; it is a measurement guard, not a verifier.
- When To Solve/Verify, arXiv:2504.01005 - mapped to A3 accuracy-and-cost
  reporting against self-consistency.
- ThinkPRM, arXiv:2504.16828 - mapped to A3 as the high-quality but expensive
  process-verifier comparator.
- Generalizable Reasoning through Compositional Energy Minimization,
  arXiv:2510.20607 - mapped to GAP-3 Stage-2 compositional ARC energy and
  flagged as the strongest `.388` follow-on.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped to the queued DiffusionGemma
  particle-guidance template after a positive energy gate.
- TRM ARC-AGI-1 ablation, arXiv:2512.11847 - mapped to the TRM headroom/vote
  decomposition and identity-conditioning control.

flagged_for_v388:
`cem_gap3_stage2_compositional_arc_energy_v388`.

Flagged for .388: `cem_gap3_stage2_compositional_arc_energy_v388`.

**Bottom line for the .388 roadmap:** run the CEM-style GAP-3 Stage-2
compositional ARC energy prototype first. Keep A1/A3 as mandatory gates and use
Self-Rewarding SMC only for DiffusionGemma guidance once the energy gate is positive.
## 2026-06-13 Exp 4170 - .387 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-v387-2026-06-13.md`.

**Filtered track:** verifier-as-reward, accepted/rejected trace selection, and
energy-guided generation for the `.387` handoff. This ingestion keeps
DiffusionGemma guidance queued because Exp 4168 recorded
`verifier_value_added=false` from a deferred, unfaithful/still-training
baseline rather than from a tested positive or negative guidance result.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful baseline and oracle-headroom
  gate before any verifier or diffusion-guidance claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget no-verifier adaptation
  control.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector and
  strongest `.387` next step.
- SEDD, arXiv:2310.16834 - mapped as the discrete score/energy scaffold for
  generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free guidance,
  arXiv:2207.12598 - mapped as the external-energy precedent and internal-score
  control.
- EntRGi, arXiv:2602.05000 - mapped as the queued DiffusionGemma reward-guidance
  template after a positive verifier-discrimination gate.
- EDLM, arXiv:2410.21357 - mapped as the internal sequence-energy comparator
  for any future guidance claim.

flagged_for_v387:
`vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

Flagged for .387: `vstar_rejected_trace_selector_headroom_gate_before_diffusiongemma_v387`.

**Bottom line for the .387 roadmap:** build the V-STaR-style rejected-trace
selector and headroom gate first. Keep EntRGi/DiffusionGemma guidance queued
unless the verifier discrimination gate flips positive.
## 2026-06-13 Exp 4162 - .386 verifier-moat guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-moat-guidance-2026-06-13.md`.

**Filtered track:** verifier-vs-self-consistency, reward-guided generation,
and ARC-AGI-3 action efficiency for the `.386` handoff. This ingestion extends
the `.385` verifier moat and queued DiffusionGemma gate without duplicating the
prior TRM/TTA-TRM/V-STaR/SEDD/CFG milestone ingestion.

**Seed and fresh-pass candidates marked ingested:**
- ARBITER, arXiv:2605.26172 - mapped as the wrong-majority/rerank-recovery
  moat anchor and the reason to aggregate an external verifier with vote.
- ThinkPRM, arXiv:2504.16828 - mapped as the data-efficient process-verifier
  existence proof and LLM-judge comparison bar.
- Optimal LLM+PRM Aggregation, arXiv:2510.13918 - mapped as the calibrated
  vote-plus-verifier aggregation recipe.
- RLV, arXiv:2505.04842 - mapped as the cheap verifier/value-head efficiency
  head-to-head template.
- EntRGi, arXiv:2602.05000 - mapped as the discrete diffusion reward-guidance
  template for DiffusionGemma after a positive discrimination gate.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, and ARC-AGI-3 tech
  report, arXiv:2603.24621 - mapped as executable transition verification and
  action-efficiency anchors.

Flagged for .386: `entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`.

**Bottom line for the .386 roadmap:** run the EntRGi-style DiffusionGemma
energy-guidance template only after the verifier-discrimination gate is
positive. If the gate is not positive, run the RLV-style cheap
energy-verifier-vs-LLM-judge efficiency head-to-head first.
## 2026-06-13 Exp 4152 - .385 recursive-reasoner/verifier energy-guidance SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-energy-guidance-2026-06-13.md`.

**Filtered track:** verifier-guided training plus energy-guided generation for
the `.385` handoff. This connects the TRM/TTA/V-STaR recursive verifier stack
to the queued DiffusionGemma energy-guidance use without treating a generator
substrate as verifier evidence.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the `nano-trm` baseline and oracle-headroom
  gate before any verifier-guided or diffusion-guided claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the same-budget adaptation-control arm
  that prevents full fine-tuning from masquerading as verifier reward.
- V-STaR, arXiv:2402.06457 - mapped as the accepted/rejected trace selector for
  saved `nano-trm` candidates before another generator pass.
- SEDD, arXiv:2310.16834 - mapped as the discrete diffusion score/energy
  formalism for generation-time verifier guidance.
- Classifier-guided diffusion, arXiv:2105.05233, and classifier-free diffusion
  guidance, arXiv:2207.12598 - mapped as the external-guidance precedent and
  no-external-verifier control.
- DiffusionGemma official docs, https://ai.google.dev/gemma/docs/diffusiongemma
  - mapped as the queued open-weight block-diffusion substrate, gated on
  measured Carnot-verifier discrimination.

Flagged for .385: `diffusiongemma_sedd_verifier_energy_guidance_probe_v385`.

**Bottom line for the .385 roadmap:** run the DiffusionGemma/SEDD
verifier-energy-guidance probe only if the verifier discrimination gate is
positive; otherwise keep improving the V-STaR-style trace selector and
candidate diversity before spending on guided-generation probe.

## 2026-06-13 Exp 4141 - .383 recursive-reasoner/verifier SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-recursive-reasoner-verifier-2026-06-13.md`.

**Filtered track:** recursive reasoner generator choice plus verifier-as-reward
mapping for the `.383` decisive graft. This follows the Exp 4130 resumable
training ingestion and the Exp 4139 graft receipt, which currently reports
`verifier_value_added=false`, `headroom_present=false`, and
`complete: uninformative_no_headroom_false_negative_risk`.

**Seed and fresh-pass candidates marked ingested:**
- GRAM, arXiv:2605.19376 - mapped as the stochastic-latent generator to graft
  onto in `.384` only if a verifier-value/headroom gate is met.
- Thinking Reward Model for complex reasoning, arXiv:2602.08498 - mapped as
  the RLVR/GRPO precedent for isolating verified-correct trace quality from
  outcome correctness, directly informing the `.383` RFT de-confound.
- Weaver, arXiv:2506.18203 - mapped as the weighted weak-verifier ensemble
  precedent for the `.383` non-oracle ensemble-rerank headline.

Flagged for .384: `gram_as_generator_if_verifier_value_added_and_headroom_present_v384`.

**Bottom line for the .384 roadmap:** use GRAM as the next generator only if
the verifier side first demonstrates transferable value with measurable
oracle(best-of-K) headroom; otherwise continue fixing headroom/candidate
diversity, not as an unconditional rerank claim.

## 2026-06-13 Exp 4130 - .382 resumable-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-resumable-training-2026-06-13.md`.

**Filtered track:** checkpoint resume, LR-schedule continuity, and
long-horizon accumulation over the `nano-trm` plus Carnot stack. This follows
the Exp 4121 `.381` baseline-graft ingestion and narrows `.382` to the runner
discipline needed before another verifier-search or training claim.

**Seed and fresh-pass candidates marked ingested:**
- PyTorch Lightning checkpoint resume docs - mapped as the full-state
  `ckpt_path` gate because Lightning checkpoints carry optimizer and LR
  scheduler state as well as global step.
- PyTorch saving/loading docs - mapped as the fallback optimizer-state
  checkpoint contract for any non-Lightning runner.
- Lightning gradient-accumulation docs - mapped as the long-horizon accounting
  rule: count optimizer steps and effective batch size, not microbatches.
- TRM, arXiv:2510.04871 - mapped as the resumed long-horizon baseline whose
  Sudoku evidence must be accumulated by checkpoint lineage and optimizer step.
- TTA-TRM, arXiv:2511.02886 - mapped as the bounded full-fine-tune control that
  must share the same resumed scheduler receipts as any verifier-admitted arm.

Flagged for .383: `lightning_full_state_lr_scheduler_resume_gate_for_nano_trm_v383`.

**Bottom line for the .383 roadmap:** first ship a Lightning full-state resume
gate for nano-trm that proves optimizer, LR scheduler, global-step, data
checksum, and gradient-accumulation continuity across two bounded passes. If
that gate fails, do not spend the next run on per-step verifier work.

## 2026-06-13 Exp 4121 - .381 TRM baseline-graft SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-baseline-graft-2026-06-13.md`.

**Filtered track:** resumable TRM Sudoku baseline reproduction plus Carnot
verifier graft, after Exp 4108 produced a checkpointed but partial baseline,
Exp 4109 found no post-hoc verifier lift over vote, and Exp 4111 flagged
in-loop verifier-guided search as the next candidate.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the resumed Sudoku Extreme baseline gate
  before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- Adaptive verifier-guided candidate expansion, arXiv:2602.01070, with VPRM/VPR
  support from arXiv:2601.17223 and arXiv:2605.10325 - mapped as the strongest
  .382 follow-on because post-hoc verifier reranking already tied vote.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity and oracle support exist.
- ReST, arXiv:2308.08998, and STaR, arXiv:2203.14465 - mapped as the resumable
  generate-filter-improve curriculum, with rejected rows retained for selector
  data.

Flagged for .382: `verifier_guided_adaptive_candidate_expansion_over_resumed_trm`.

**Bottom line for the .382 roadmap:** put the executable Sudoku verifier inside
candidate expansion over the resumed TRM checkpoint before spending on selector
or RFT work. Require pass@1 or oracle-support lift over fixed-K vote and Exp
4109 post-hoc verifier rerank; otherwise selector/RFT work should stay blocked.
## 2026-06-12 Exp 4111 - .380 TRM verifier-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-verifier-training-2026-06-12.md`.

**Filtered track:** TRM baseline reproduction plus verifier-guided training and
search over the `nano-trm` Sudoku substrate after Exp 4108 produced an honest
partial baseline and Exp 4109 produced an honest post-hoc verifier null.

**Seed and fresh-pass candidates marked ingested:**
- TRM, arXiv:2510.04871 - mapped as the faithful Sudoku Extreme baseline
  reproduction gate before any verifier-lift claim.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tuning adaptation control
  that must be isolated from verifier-admission effects.
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected Sudoku trace selector
  training once candidate diversity exists.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence, with rejected rows retained for selector data.
- Adaptive verifier-guided search, arXiv:2602.01070, with VPRM/VPR support from
  arXiv:2601.17223 and arXiv:2605.10325 - mapped as the next in-loop verifier
  use because Exp 4109 post-hoc reranking tied vote.

Flagged for .381: `verifier_guided_adaptive_sudoku_search_before_training`.

**Bottom line for the .381 roadmap:** move the executable Sudoku verifier into
candidate expansion before spending on another full fine-tune. Require pass@1
or oracle-support lift over fixed-K vote and Exp 4109 post-hoc verifier rerank;
otherwise keep V-STaR and RFT routes blocked.

## 2026-06-12 Exp 4102 - .379 TRM self-training SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-trm-self-training-2026-06-12.md`.

**Filtered track:** verifier-certified RFT over a recursive `nano-trm`/TRM
substrate, with Carnot verifier labels selecting, correcting, or densifying the
training signal.

**Seed and fresh-pass candidates marked ingested:**
- V-STaR, arXiv:2402.06457 - mapped as accepted/rejected TRM trace selector
  training before any second RFT corpus gate.
- STaR, arXiv:2203.14465, and ReST, arXiv:2308.08998 - mapped as the cached
  generate-filter-improve cadence for recursive traces.
- TTA-TRM, arXiv:2511.02886 - mapped as the full-fine-tune substrate and a
  control against attributing adaptation-only gains to the verifier.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as FP/FN-calibrated
  weighting and abstention before verifier-certified RFT.
- VPRM/VPR, arXiv:2601.17223 and arXiv:2605.10325 - mapped as dense
  per-recursion step rewards only after outcome calibration.
- Self-Trained Verification, arXiv:2605.30290 - marked as fresh adjacent
  verifier-training evidence, but deferred behind the cheaper V-STaR trace
  selector because `.379` already emits accepted/rejected TRM traces.

Flagged for .380: `vstar_rejected_trace_selector_for_trm_rft`.

**Bottom line for the .380 roadmap:** build a V-STaR-style selector over the
saved nano-trm candidate pool, require a rerank win against the current Carnot
verifier ordering, and only then let the selector gate a second full-fine-tune
RFT corpus.
## 2026-06-12 Exp 4094 - .378 precision-calibration SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-precision-calibration-2026-06-12.md`.

**Filtered track:** the `.378` verifier-precision / verifier-as-reward headline:
the 0.32 false-positive channel behind ARC certification precision 0.6818, the
Exp 4087 precision rescue to 0.8824 at 0.7143 recall, the blocked Exp 4088/4089
RFT path, and the Exp 4093 OFF-ARC demo-fit precision replay.

**Seed and fresh-pass candidates marked ingested:**
- BARC / Combining Induction and Transduction, arXiv:2411.02272 - mapped as an
  augmentation-consistency filter before RFT corpus admission.
- Noisy Data is Destructive to RLVR, arXiv:2603.16140 - mapped as the stop-rule
  against training through the 0.32 false-positive channel.
- RLVR with imperfect verifiers, arXiv:2510.00915 - mapped as explicit FP/FN
  noise correction and calibration metadata for future RLVR hooks.
- V-STaR, arXiv:2402.06457 - mapped as rejected-trace retention and verifier
  training over accepted/rejected pairs.
- RFT scaling, arXiv:2308.01825 - retained as the simple fine-tuning baseline
  only after clean, diverse positives exist.
- Invisible Leash, arXiv:2507.14843 - retained as the same-pool latent-support
  gate before RFT/RLVR spend.
- Process Supervision-Guided Policy Optimization for Code Generation,
  arXiv:2410.17621, plus CodePRM ACL 2025 - mapped as step-level process reward
  records before sparse reward RL.

**Bottom line for the .379 roadmap:** prioritize
`calibrated_forward_noise_correction_before_rlvr`,
`augmentation_consistency_filter_before_rft_corpus`,
`vstar_rejected_trace_verifier_training`, and
`step_level_process_reward_weighted_sft`; keep
`latent_support_gate_before_rft_spend` as the launch gate so RFT is only used
when correct transforms are already present in the generated pool.

## 2026-06-11 Exp 4081 - .377 verifier-as-reward SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-verifier-as-reward-2026-06-11.md`.

**Filtered track:** the `.377` verifier-as-reward pivot: verifier-certified RFT,
Tulu-3-style RLVR, Invisible Leash latent-vs-absent support gating, dense
process-reward distillation, and RFT/STaR/ReST self-training over the current
three-arm RFT pipeline.

**Seed and fresh-pass candidates marked ingested:**
- Tulu 3, arXiv:2411.15124 - mapped as the open SFT/DPO/RLVR recipe, but gated
  behind a clean verifier-certification label.
- The Invisible Leash, arXiv:2507.14843 - mapped as the latent-vs-absent support
  diagnostic before any RFT/RLVR spend.
- RL vs. Distillation, arXiv:2505.14216 - mapped as the accuracy-vs-capability
  fork and the reason to track pass@k/oracle support, not only pass@1.
- Self-Distilled RLVR, arXiv:2604.03128 - mapped as a later credit-assignment
  upgrade only after external verifier reward direction is clean.
- STaR, arXiv:2203.14465 - mapped as the minimal generate-filter-finetune loop.
- ReST, arXiv:2308.08998 - mapped as the offline generate-filter-improve cadence
  for reusable cached trace pools.
- Verifiable Process Reward Models, arXiv:2601.17223 - mapped as deterministic
  rule-verifier step rewards rather than opaque neural step judges.
- Verifiable Process Rewards for Agentic Reasoning, arXiv:2605.10325 - mapped as
  dense turn/action rewards for long-horizon ARC agent trajectories.

**Bottom line for the .378 roadmap:** prioritize
`latent_vs_absent_precision_gate_before_rft` and
`process_reward_weighted_sft_over_trace_certification`; keep the deconfounded
RFT-correct vs RFT-ablation contrast; only add self-distilled RLVR after the
external verifier reward direction is measured clean.

## 2026-06-11 Exp 4067 - .376 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-v376-unsaturated-corpora-and-online-pruning.md`.

**Filtered tracks:** LOCAL-12B oracle-headroom code corpus for the off-ARC
demo-fit verifier transfer measurement, and VERIFIER-GUIDED ONLINE ACTION-PRUNING
for the efficient ARC harness over explore-first + GAP-4.

**Exp 4055 seed flags confirmed actionable for .376:**
- `evalplus_hidden_rescore_fixed_pool` - actionable as the cheap first hidden-test
  gate, but must route upward when oracle headroom is absent.
- `saga_generated_tests_as_discriminator_arm` - actionable after the official
  hidden-score path is stable; use only as generated-test tie-break/explanation.
- `gap4_online_pruner_for_explore_first_arc` - actionable now as soft pruning
  with replay-disabled-on-failure.
- `equivpruner_state_action_cache_for_arc` - actionable now for exact state hashes
  and GAP-4-confirmed equivalence only.

**Seed and fresh-pass candidates marked ingested:**
- LiveCodeBench v6, arXiv:2403.07974, plus current public leaderboard mirror - mapped as the local-12B headroom route after EvalPlus.
- EvalPlus / HumanEval+ / MBPP+, arXiv:2305.01210 - retained as the first fixed-pool hidden rescore gate.
- SAGA / Rethinking Verification for LLM Code Generation, arXiv:2507.06920 - retained as the generated-test discriminator arm.
- Inference-Time Code Selection via Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as the bounded functional-equivalence diagnostic.
- ACES, arXiv:2604.03922 - mapped as the same-pass-matrix Arm A++ baseline.
- What If We Allocate Test-Time Compute Adaptively?, arXiv:2602.01070 - mapped as the online PRM-style prune/expand control precedent.
- Update-Free On-Policy Steering via Verifiers, arXiv:2603.10282 - newly mapped as the no-weight-update verifier-steering precedent for GAP-4 action priors.
- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure, arXiv:2602.03975 - retained as selective verifier-call allocation over intermediate states.
- EquivPruner, arXiv:2505.16312 - retained as exact state/action equivalence caching before approximate pruning.
- CoT2-Meta, arXiv:2603.28135 - retained as the explicit expand/prune/repair/stop/fallback controller shape.
- DIRECT, arXiv:2606.12402 - marked as a fresh adjacent compute-router citation, useful for budget framing but not a first implementation target.

**Bottom line for the .377 roadmap:** prioritize
`livecodebench_v6_local12b_headroom_route` and
`gap4_soft_prune_replay_for_arc_efficiency`; add
`gap4_equivpruner_exact_state_action_cache` before learned/approximate pruning,
and add `saga_generated_tests_hidden_score_tiebreak` only after official hidden
scores are frozen.

## 2026-06-11 Exp 4055 - .375 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-unsaturated-execverif-and-verifier-pruner.md`.

**Filtered tracks:** UN-SATURATED execution-verification corpus for exp4056/4057 off-ARC
demo-fit transfer measurement, and VERIFIER-GUIDED online action-pruning for the efficient
ARC harness over explore-first + GAP-4.

**Seed candidates marked ingested:**
- EvalPlus / HumanEval+ / MBPP+, arXiv:2305.01210 - mapped as the default hidden-test rescore path that fixes the `.374` base HumanEval/MBPP saturation failure.
- LiveCodeBench v6, arXiv:2403.07974 - mapped as the contamination-free escalation corpus if EvalPlus hidden tests lack headroom.
- SAGA / Rethinking Verification for LLM Code Generation, arXiv:2507.06920 - mapped as the generated-test discrimination arm after the fixed-pool EvalPlus path is stable.
- DryRUN / You Don't Need Public Tests to Generate Correct Code, arXiv:2604.21598 - mapped as the public-test-free self-simulation tie-break arm, not as authoritative final scoring.
- What If We Allocate Test-Time Compute Adaptively?, arXiv:2602.01070 - mapped as the online verifier-guided prune/expand control rule for ARC frontier expansion.
- Marco DeepResearch, arXiv:2603.28376 - mapped as the verification-centric budget-ledger precedent for agentic search.
- Pushing Test-Time Scaling Limits of Deep Search with Asymmetric Verification, arXiv:2510.06135 - mapped as the cheap-verifier-vs-expensive-search budget split for GAP-4 pruning.

**Fresh-pass confirmations marked ingested:**
- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure, arXiv:2602.03975 - mapped as verifier-cost-limited selective GAP-4 calls over intermediate ARC states.
- EquivPruner, arXiv:2505.16312 - mapped as exact state/action equivalence caching before approximate pruning.
- CoT2-Meta, arXiv:2603.28135 - mapped as the explicit expand/prune/repair/stop/fallback controller shape for explore-first telemetry.
- SEP, arXiv:2604.06485, and ACES, arXiv:2604.03922 - carried forward as same-pool baselines around the EvalPlus measurement, not as the headline fix.

**Bottom line for the .376 roadmap:** prioritize `evalplus_hidden_rescore_fixed_pool`
and `gap4_online_pruner_for_explore_first_arc`; add `saga_generated_tests_as_discriminator_arm`
only after the fixed-pool EvalPlus adapter is stable, and add `equivpruner_state_action_cache_for_arc`
before any learned/approximate pruning.

## 2026-06-11 Exp 4043 - .374 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-offarc-power-and-closed-loop-planning.md`.

**Filtered tracks:** OFF-ARC statistical power + stronger discriminator for exp4044/4045, and
CLOSED-LOOP planning over the verified vc33 world model under model error for exp4046.

**Seed candidates marked ingested:**
- Inference-Time Code Selection via Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as the SEP semantic-partition tie-break/diagnostic for the full-power off-ARC panel.
- Scaling Agentic Verifier for Competitive Coding, arXiv:2602.04254 - mapped as the expensive targeted-counterexample comparator for hard ties.
- Efficient Prediction of Pass@k Scaling, arXiv:2510.05197 - mapped as the pilot sizing and budget discipline for HumanEval+MBPP power.
- What model does MuZero learn?, arXiv:2306.00840 - mapped as the policy-support / WM-trust constraint for vc33 search.
- World-in-World, arXiv:2510.18135 - mapped as the closed-loop task-success evaluation rule.
- Latent Geometry Beyond Search / GC-IDM, arXiv:2605.08732 - mapped as per-step replanning and action-prior guidance.
- Bounding Distributional Shifts through Novelty Detection, arXiv:2508.06096 - mapped as the novelty-MPC trust gate against WM exploitation.

**Fresh-pass confirmations marked ingested:**
- DOCE, arXiv:2408.13745 - retained as the execution-based code-selection protocol anchor for the powered measurement.
- CodeT, arXiv:2207.10397 - retained as the dual execution-agreement baseline alongside ACES.
- ACES, arXiv:2604.03922 - promoted as the strongest same-pass-matrix Arm A++ baseline.
- R-WoM, arXiv:2510.11892 - mapped as retrieval grounding over verified transition traces for short-lookahead vc33 planning.

**Bottom line for the .375 roadmap:** prioritize `offarc_full_power_sep_aces_agentic_counterexample_panel`
and `closed_loop_vc33_replan_with_wm_trust_gate`; add `novelty_mpc_gate_for_verified_wm_search`
as the required guardrail so the planner cannot claim model-only goal satisfaction as a solve.

## 2026-06-11 Exp 4030 - .373 SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-2026-06-11-offarc-and-search.md`.

**Filtered tracks:** OFF-ARC execution-consistency verifier transfer for exp4031/4032, and
hierarchical/subgoal search over a verified world model for exp4034/4035.

**Seed candidates marked ingested:**
- DOCE, arXiv:2408.13745 - mapped as the exp4031/4032 protocol anchor.
- ACES, arXiv:2604.03922 - mapped as the stronger same-pass-matrix consistency baseline.
- Symbolic Equivalence Partitioning, arXiv:2604.06485 - mapped as a bounded diagnostic/enrichment layer.
- Scaling Agentic Verifier, arXiv:2602.04254 - mapped as an expensive comparator/escalation path.
- Iterative Self-Training via Reinforced Re-Ranking, arXiv:2504.09643 - mapped as a future distillation target only.
- Hierarchical Planning with Latent World Models, arXiv:2604.03208 - mapped as symbolic hierarchical MPC over vc33.
- Subgoal-Guided PHS, arXiv:2506.07255 - mapped as failed-tree subgoal mining/logging for exp4035.
- Sokoban HRL with Landmarks, arXiv:2504.04366 - mapped as feature-level landmark decomposition precedent.

**Fresh-pass additions marked ingested:**
- Policy-Guided Heuristic Search with Guarantees, arXiv:2103.11505 - mapped as policy+heuristic priority extension for current best-first search.
- Structure-Induced Information for Rerooting Levin Tree Search, arXiv:2605.30664 - mapped as the .374 fallback if explicit subgoals are too costly.

**Bottom line for next roadmap:** prioritize `off_arc_demo_fit_vs_aces_and_doce_protocol` and
`vc33_hierarchical_subgoal_search_with_rerooting_fallback`; keep SEP as a diagnostic, not a gate.

## Sweep 2026-05-24T15:20Z (Claude outer-loop, infrastructure recovery)

**Recovery summary:**

The 2026-05-16 sweep entry below reported the infrastructure as
"2-channel-down": Bash dead, arXiv API HTTP 429, fallback URL
returning 404. The 2026-05-24 outer-loop probe ran a direct
verification:

| Check | Pre-recovery | Post-recovery |
|---|---|---|
| Bash exit code | exit 1 (every shell) | exit 0 (working) |
| arXiv listing URL (`/list/cs.LG/2605`) | HTTP 404 | still 404 — pattern was stale |
| arXiv listing URL (`/list/cs.LG/2026-05`) | not tested | **HTTP 200 — the correct pattern** |
| arXiv listing URL (`/list/cs.LG/recent`) | not tested | **HTTP 200 — alternate fallback** |
| arXiv API direct (cluster URL) | HTTP 429 | HTTP 200 after redirect-follow (HTTPS, `-L`) |
| HN Algolia API | 0 hits at search-time | HTTP 200 (channel itself live) |

**The fix:** the cron-prompt URL pattern `arxiv.org/list/cs.LG/2605`
should be replaced with `arxiv.org/list/cs.LG/2026-05` (calendar-month
format) or `arxiv.org/list/cs.LG/recent` (alternate fallback). The
arXiv API itself works fine via HTTPS with redirect-following; the
earlier 429s appear to have been transient rate-limiting, not a
permanent block. Cron-prompt URL update remains operator-owned but
is now a documented one-line edit.

**Fresh sweep results (all 5 clusters, 4 top hits each):**

Cluster 0 — verifier ensemble / spec gaming / reward hacking:
- arXiv:2605.21384 (2026-05-20) **SpecBench: Measuring Reward Hacking
  in Long-Horizon Coding Agents.** Directly Carnot-adjacent; benchmark
  for the exact failure mode our verifier-authenticity discipline
  catches. **Promote to research-references.md.**
- arXiv:2605.20744 (2026-05-20) **Hack-Verifiable Environments:
  Towards Evaluating Reward Hacking at Scale.** Adversarial-verify
  discipline analog at the environment level. **Promote.**
- arXiv:2605.22620 (2026-05-21) Two is Better Than One: Collapse-free
  Multi-Reward RLIF Training Framework. AND-composed verifier
  ensemble structurally adjacent. Track.

Cluster 1 — EBM / energy-guided LLM:
- arXiv:2605.14558 (2026-05-14) **Resolving Action Bottleneck:
  Agentic Reinforcement Learning Informed by Token-Level Energy.**
  Token-level energy as RL signal; direct EBM-as-policy framework.
  **Promote.**

Cluster 2 — SAE / probes / interpretability:
- arXiv:2605.22462 (2026-05-21) From Correlation to Cause: Five-Stage
  Methodology for Feature Analysis in Transformer Language Models.
  Methodology paper, could inform adversarial-verify discipline.
- arXiv:2605.20868 (2026-05-20) Runtime-Certified Bounded-Error
  Quantized Attention. Relevant to the RotorQuant conversation.

Cluster 3 — active inference / FEP / world model:
- arXiv:2605.22675 Self-Policy Distillation via Capability-Selective
  Subspace Projection — possible relevance to FR-11 attractor work
  (.282 exp3007). Track.
- Other cluster-3 hits this round were medical imaging / video / table
  recognition false positives. Cluster-3 narrowing may be warranted.

Cluster 4 — sub-quadratic / analog / FPGA:
- arXiv:2605.17720 (2026-05-18) **ROA-Based Subharmonic Injection
  Locking for Oscillator-Based Ising Machines.** Hardware Ising
  substrate evolution; future production target candidate.
  **Promote.**
- arXiv:2605.19399 (2026-05-19) HSCO-Bench: Agent-Driven End-to-End
  Hardware-Software Co-design Benchmark for SoCs. Track.

**Result:** 20 fetched / dedupe not yet run / 4 promoted to
references.md / 5 watched / 11 tangential or false-positive.
Sweep infrastructure operational again; future planner passes can
invoke sweep_clusters.py + arXiv API directly without intervention.

### Sweep takeaways

1. **The URL-pattern bug was the real annoyance** — Bash recovered
   on its own, the API was never permanently rate-limited, but the
   stale cron-prompt URL kept producing 404s. The fix is a single
   character class change: `2605` → `2026-05`.
2. **The 5-cluster fan-out works at high signal-to-noise** —
   clusters 0, 1, 4 produced highly Carnot-relevant papers; cluster
   2 produced one solid + one tangential; cluster 3 mostly false
   positives this round.
3. **Cluster 3 may need re-narrowing** — "active inference" + "free
   energy" without a strong AND-anchor surfaces too many adjacent
   domains (medical, vision, table recognition). Worth tightening
   when next operator-edits the cron prompt.
4. **The planner has been pulling references organically** through
   plan-next-milestone passes (7 new arXiv entries in research-
   references.md since 2026-05-21 — see the metamorphic-testing
   cluster + attractor-reasoning cluster). Sweep infrastructure
   being broken did NOT stop literature integration; it just made
   it less systematic.

## Sweep 2026-05-16T08:00Z (Claude outer-loop, 2-channel-down + Bash dead)

**Environment constraint:** outer-loop Bash still broken (every shell
exits 1); `sweep_dedupe.py` + `sweep_citations.py` etc. NOT INVOCABLE.

**Queries fired (degraded):**
- arxiv cluster 0 (verifier-ensemble / null-space / spec-gaming) →
  **HTTP 429 Too Many Requests**.
- arxiv cluster 3 (active-inference / free-energy / LLM) →
  **HTTP 429 Too Many Requests**.
- arxiv.org/list/cs.LG/2605 (cron-prompt fallback) →
  **HTTP 404 Not Found** (URL pattern may be stale).
- hn.algolia "verifier energy LLM" → 0 hits.

**Result: 0 fetched / 0 dedupe-skipped / 0 scored / 0 promoted.**

The arxiv API rate-limit on 2 consecutive queries plus the fallback
listing also unavailable means this sweep has zero candidate-fetch
capacity. Honest report: nothing surfaced.

### Sweep takeaways

1. **The cron-prompt fallback URL is stale.** `https://arxiv.org/list/cs.LG/2605`
   returns 404. Working format may be `cs.LG/2026-05` or `cs.LG/26.05`.
   Worth verifying when shell recovers and updating the cron prompt.
2. **arxiv API rate-limit hits confirm operator-discussed broadened
   queries would help** (more results per request → fewer requests).
   Cron-prompt URL update remains operator-owned.
3. **5 consecutive sweep cycles in degraded state** (Bash dead + now
   API rate-limited). Net new candidates this period: 1 (CoT2-Meta
   Score 320 from the 04:35Z citation-sweep window before API
   throttling hit). Routine keyword-rotation channel exhausted +
   cannot open new channels.

## Sweep 2026-05-16T06:50Z (Claude outer-loop, degraded environment — WebFetch+Edit only)

**Environment constraint:** outer-loop Bash failed earlier in this session
(every shell command returns exit 1 with no output). Sweep helpers
(`sweep_dedupe.py`, `sweep_citations.py`, etc.) NOT INVOCABLE. Manual
dedupe against known-set via Edit-tool memory only. Conductor's
auto-commit chain will sweep up this file edit even though outer-loop
git operations are blocked.

**Queries fired (hour-mod-4=2; cluster 2 primary):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction
  error" AND "LLM" → 8 fetched
- (cluster 0/1/3 skipped — Bash blocker means I can't pipe through
  dedupe efficiently; one cluster is sufficient to confirm saturation)
- HN skipped (5+ prior 0-hits today)

**Manual dedupe (no `sweep_dedupe.py --filter` available):**

All 8 cluster-2 IDs are already in research-studying.md from prior
sweeps (2605.14694, 2605.14449, 2605.14347, 2605.13930, 2605.12874,
2605.12809, 2605.12770, 2605.12245). Verified by memory of the
13-prior-sweep catalogue.

**Result: 8 fetched / 8 known-skipped / 0 scored / 0 promoted.**

This is the third 100%-saturation sweep in 24 hours (the pattern shipped
2026-05-16T00:40Z, again at 04:35Z when keyword rotation produced 0 but
routine citation-sweep on ODAR anchor surfaced CoT2-Meta Score 320, and
now). The auto-rotation channel has fully mapped the recent SOTA window.

### Sweep takeaways

1. **Continued saturation validates the operator-approved trickle policy.**
   Queue backlog (Phase 1 prongs awaiting `.198 outcomes + operator
   PyPI approval + CoT2-Meta routing experiment + recovered audit
   tasks in `.200) needs absorption time before new literature can
   productively layer on.
2. **Bash-tool blocker is the dominant constraint this sweep.** With
   `sweep_dedupe.py` and `sweep_citations.py` unavailable, citation-
   sweep depth-following (the high-yield channel from the morning's
   04:35Z sweep) couldn't fire. Next sweep with a working shell
   should run routine citation-sweep on CoT2-Meta or any other
   high-score anchor not yet depth-swept.
3. **Operator-flagged anchors remain the highest-yield channel.** The
   sweep helper suite, even when fully functional, beats keyword
   rotation primarily through citation-following. The keyword-rotation
   pure-keyword channel has converged to ~0 promotions per fire.

## Sweep 2026-05-16T04:35Z (Claude outer-loop, routine citation-sweep on ODAR anchor — cadence rule)

**Helpers used (per the routine-citation cadence shipped 2026-05-16T01:00Z):**
- `sweep_citations.py 2602.23681` (ODAR anchor, Score 400; not yet
  depth-swept since promotion at 21:30Z). Returned ~15 unique IDs.
- `sweep_citations.py 2605.12484` (Fast-Slow anchor, Score 400) —
  still 404 not-indexed in Semantic Scholar (paper too new at 4 days
  old at promotion + 14 days now).
- `sweep_dedupe.py --filter` (default workflow step 2.5).

**Result: ~15 fetched / dedupe-skipped 13 / 2 scored / 1 promoted.**

### NEW Rank HIGH: CoT2-Meta — Budgeted Metacognitive Control for Test-Time Reasoning (arXiv:2603.28135, Mar 30 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Siyuan Ma, Bo Gao, Zikai Xiao, Hailong Wang, Xinlei Yu,
  Rui Qian, Jiayu Qian, Luqi Gong, **Yang Liu** (same Ma/Gao/Liu line
  as ODAR)
- **Why it matters:** Training-free framework integrating CoT generation
  with metacognitive control decisions — **"expansion, pruning, repair,
  stopping, fallback decisions"** for budgeted computational allocation
  during reasoning. This is structurally an explicit ROUTING/ORCHESTRATION
  framework over a verify-repair-stop loop: "expand" = generate
  candidates, "prune" = verifier rejection, "repair" = exactly what
  Carnot does, "stopping" = ODAR-style fast-path acceptance, "fallback"
  = deliberative path. Same author group as ODAR (Ma/Gao/Liu) suggests
  a coherent research program; CoT2-Meta is the operational framework,
  ODAR is the routing-mechanism within it.
- **Action:** Cite alongside ODAR (Score 400) in paper-v6 §3 as evidence
  for the metacognitive-control architecture pattern. Concrete .197+
  proposal: extend the Carnot ODAR routing (.190 exp1822 queued) with
  the explicit expand/prune/repair/stop/fallback state-machine from
  CoT2-Meta. Carnot's verify-repair loop currently lacks an explicit
  "fallback" path — the LLM either passes the verifier or iterates;
  the fallback (e.g., escalate to k=16 ensemble disagreement check or
  human review) is implicit. CoT2-Meta provides the formalism.
- **Cross-reference:** complements arXiv:2602.23681 ODAR (Score 400)
  + arXiv:2605.12484 Fast-Slow (Score 400). Three coherent peer
  anchors from the same year on the routing/orchestration theme.

### Other scored (no promotion):

- **arXiv:2511.09873** HierRouter — Coordinated Routing of Specialized
  LLMs via RL (Gupta/Guo/Kannan/Prasanna, Nov 2025). Score 3×3×2×2 =
  **36**. Adjacent (LLM routing) but RL-trained hierarchical routing
  is high-cost to replicate; out-of-scope for Carnot's training-free
  verify-repair architecture.

### Pre-2026 references catalogued (no scoring; archived for citation tracking)

~13 papers in the ODAR references-direction sweep span Feb 2025
through Nov 2025 (foundational work on routing, reasoning RL, planning).
These are now in the known-set for future citation-sweep dedupe.

### Sweep takeaways

1. **Routine citation-sweep cadence (every 4th keyword sweep) earns
   its keep.** Today: keyword rotation at 100% saturation; citation-sweep
   surfaced 1 Score-320 promotion. CoT2-Meta would NEVER have surfaced
   via the 4 cluster queries — its abstract uses "metacognitive control"
   and "expansion / pruning" not "verifier ensemble" / "energy based
   model" terms.
2. **Author-cluster signal:** Ma/Gao/Liu have published two highly-
   relevant Carnot-adjacent papers (ODAR Feb 2026 + CoT2-Meta Mar 2026).
   This is a research-program convergence worth following — citation-
   sweep on CoT2-Meta in a future cycle may surface their newer work.
3. **arXiv:2605.12484 Fast-Slow still not S2-indexed** (14 days old);
   citation-sweep yield decays sharply with anchor age below 30 days.
   Patience expected for the May 2026 anchors.
4. **The fast-slow / ODAR / CoT2-Meta cluster** is now a coherent
   sub-literature anchor for paper-v6 §3 architecture-validation. All
   three converge on the same fast-deliberative routing pattern that
   Carnot's verify-repair loop implements. The four-anchor triangulation
   (bijection + Fast-Slow + ODAR + CoT2-Meta) is becoming a five-anchor
   triangulation with inference-time-planning (2602.02991) as the fifth.

## Sweep 2026-05-16T00:40Z (Claude outer-loop, light fire — saturation acknowledgment)

**Queries fired (light fire per operator-approved trickle policy):**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 5 fetched (cluster 3)
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 8 fetched (cluster 2)

**Dedupe filter:**
13 candidates fetched, **all 13 known-skipped** (100% filter rate).
64 IDs now in known-set (up from 53 at 16:50Z fire). The keyword
rotation has fully saturated against research-studying.md's catalogue.

**Result: 0 NEW / 0 scored / 0 promoted.**

This is the expected outcome per the operator-approved trickle policy.
The queue's backlog (Fast-Slow Variant adversarial confirmation
pending in .192 exp1837, ODAR Score 400 awaiting .193+ integration,
Inference-Time Planning Score 144 cite still pending, PyPI ship-track
about to unblock via tag-push) needs experimental absorption time
before new literature can be productively layered on.

### Sweep takeaways

1. **100% dedupe filter rate is healthy at this point in the cycle.**
   Carnot's literature landscape is well-mapped relative to the
   4-cluster query rotation's reach. New high-impact findings will
   come from: (a) operator-flagged anchors (like the 13:15Z Fast-Slow
   promotion), (b) citation-following from existing high-score nodes
   when triggered, OR (c) cron-prompt URL update to the broadened
   cluster queries (sweep_clusters.py output) when operator decides.
2. **No helper-extended sweep this fire.** Citation-sweep + semscholar
   helpers remain available; reserving them for either (a) a specific
   operator-flagged research question or (b) post-.192 retro when
   exp1837 confirmation result drives new follow-up direction.
3. **Saturation confirms the queue's high-impact items are real.**
   When the auto-rotation finds 0 new across 13 fetched, it means
   the queue's existing entries (Fast-Slow, ODAR, bijection,
   inference-time-planning) ARE the SOTA — not gaps in coverage.

## Sweep 2026-05-15T21:30Z (Claude outer-loop, 4-helper combined: citations + semscholar)

**Helpers exercised:**
- `sweep_citations.py 2604.07650` (Behavioral Entanglement anchor, Score 400, Apr 2026) — 38 unique IDs (mostly older references).
- `sweep_citations.py 2605.02269` (Spec Gaming anchor, Score 320, May 2026) — 20 unique IDs.
- `sweep_citations.py 2605.14449` (QAOD anchor, Score 320) — 404 not-indexed.
- `sweep_citations.py 2602.18671` (Spilled Energy, Feb 2026) citations-only — 0 hits (no arxiv-mirrored citations yet).
- `sweep_semscholar.py "fast-slow LLM verifier energy"` — 9 unique IDs.
- `sweep_dedupe.py --filter` (default workflow step 2.5).

**Result: ~70 fetched across helpers / ~50 known-skipped via dedupe / ~20 newly-scored / 1 promoted (Score 400, in-domain critical).**

### NEW Rank URGENT (TIED with Spec Gaming + QAOD at 320 → 400 now): ODAR — Free-Energy-Principled Adaptive Routing for LLM Reasoning (arXiv:2602.23681, Feb 27 2026)
- **Score:** 5×4×4×5 = **400**
- **Authors:** Siyuan Ma, Bo Gao, Xiaojun Jia, Simeng Qin, Tianlin Li,
  Ke Ma, Xiaoshuang Jia, Wenqi Ren, Yang Liu
- **Why it matters CRITICAL:** Adaptive routing for LLM reasoning that
  dynamically allocates compute between FAST AND DELIBERATIVE agents
  using **active inference + free-energy principle**. Employs "a
  free-energy-principled, risk-sensitive fusion mechanism" to select
  answers while balancing likelihood with epistemic uncertainty.
  **This directly merges Carnot's Phase 4 (active inference) track
  with the Fast-Slow Variant (.189 exp1811) track** — the two have
  been parallel until now, but ODAR demonstrates they're the SAME
  mechanism viewed from different sides:
    - Phase 4: free-energy reduction = alpha_t (Carnot's target metric,
      ensemble-output substrate-inaccessible per exp1745)
    - Fast-Slow Variant: fast-weight context shaped by slow-weight
      verifier ensemble (exp1811's design)
    - ODAR: free-energy routing between fast/deliberative agents on
      the same answer-space (Ma et al. closed-form mechanism)
- **Empirical strength:** Tested across **23 benchmarks** with
  "reduced computational overhead compared to uniform sampling
  strategies." This is multi-benchmark evidence at a scale Carnot
  hasn't yet achieved on Phase 4.
- **Action — high-leverage:**
  1. ops/known-issues.md MANDATORY entry: ODAR routing mechanism
     should inform Phase 4 canonical-metric decision (.189 exp1814 OR
     a downstream task).
  2. Paper-v6 §3: ODAR is the FOURTH independent peer-reviewed
     anchor for Carnot's design pattern. Quadruple triangulation now
     (bijection 500 + Fast-Slow 400 + ODAR 400 + inference-time-planning
     144).
  3. Concrete .190+ proposal: "Carnot ODAR-style Routing" — adopt
     the free-energy-principled risk-sensitive fusion in place of
     Carnot's current verify-repair argmax selection. Acceptance gate:
     match ODAR's "reduced computational overhead" claim relative to
     uniform-iteration verify-repair on a 30-example reasoning corpus.
- **Cross-references:** complements arXiv:2605.12536 (IIT↔FEP
  maximum-caliber bridge, the basis of Carnot's alpha_t' replacement
  derivation in exp1721). ODAR is the OPERATIONAL counterpart to
  exp1721's theoretical derivation.

### Other newly-scored (no promotion):

- **arXiv:2604.01681** Agentic Fast-Slow Planning for AVs (Chen et al.,
  Apr 2026). Score 2×4×2×2 = **32**. Autonomous-vehicle hierarchical
  planning; out-of-domain for Carnot's LLM verification.
- **arXiv:2603.22866** Aerial Agentic AI (Dong et al., Mar 2026). Score
  1×3×1×1 = **3**. UAV wireless networks, out-of-domain.
- **arXiv:2604.12185** Order-Aware Hypergraph RAG (Wu/Kuai et al., Apr
  2026). Score 2×3×2×2 = **24**. RAG with order-aware knowledge
  representation; adjacent to structural verification but not core.
- **arXiv:2601.03267** OpenAI GPT-5 System Card (OpenAI team, Dec 2025
  v1 / May 2026 v2). Score 3×3×1×2 = **18**. Production-deployment
  baseline reference; Carnot doesn't have GPT-5 access. Noted as
  reference material.

### Sweep takeaways

1. **semscholar channel produced the day's highest-yield single
   helper invocation** — 9 IDs fetched, 1 Score-400 promotion. By
   contrast: prior 2 keyword-rotation sweeps (16:50Z, 20:35Z) found
   0 promotions each; 21:15Z citation-sweep found 1 Score-144
   promotion.
2. **ODAR is the day's most strategic literature finding.** The
   Phase 4 program has been pursuing alpha_t measurement across 5
   experiments (exp1715/1721/1741/1745/1811) without convergence;
   ODAR demonstrates that a DIFFERENT free-energy-derived target
   (routing mechanism, not metric measurement) succeeds across 23
   benchmarks. This may be the rescue path Phase 4 has been blocked
   on.
3. **Citation-following yield decays with anchor age:** Dec 2025
   anchor → 39 unique IDs; Feb 2026 → 38; Apr 2026 → 38 (mostly older
   references); May 2026 → 404 (too new). The older anchors provide
   archival depth; newer anchors need different methods. semscholar
   keyword search complements citation-following at the SOTA edge.
4. **Quadruple peer-reviewed triangulation** of Carnot's verify-repair
   architecture pattern: arXiv:2512.15605 (bijection) + arXiv:2605.12484
   (Fast-Slow) + arXiv:2602.23681 (ODAR) + arXiv:2602.02991 (inference-
   time planning). Paper-v6 §3 has substantial peer-review backing now.

## Sweep 2026-05-15T21:15Z (Claude outer-loop, citation-following + broadened-cluster attempt — FIRST extended-window sweep)

**Helpers used (per the suite shipped 21:00Z):**
- `scripts/sweep_citations.py 2512.15605 --direction both` (AR-LM↔EBM
  bijection anchor, Score 500 — highest-scoring active queue entry).
- `scripts/sweep_dedupe.py --filter` (default workflow step 2.5).
- `scripts/sweep_clusters.py` broadened queries on clusters 1 + 2 —
  **arXiv API returned HTTP 429 Too Many Requests** on the broadened
  cluster URLs (max_results=20 + complex OR-chains hit rate-limit
  thresholds). Citation-sweep alone carried this fire.

**Result: 39 fetched / 0 dedupe-skipped / 39 scored / 1 promoted (in-domain).**

Citation-sweep returned 39 unique arxiv IDs from the bijection anchor's
references + citations. **All 39 NEW to the queue** (0 known-skipped —
the keyword rotation has been blind to all of these because the topical
filters didn't pattern-match the abstract wording). 36 of 39 are
pre-2026 references (foundational EBM / RL / LLM papers); 3 are 2026
citations:

### NEW Rank MEDIUM: Inference-Time Planning Self-Generated Context (arXiv:2602.02991, Feb 3 2026)
- **Score:** 4×3×3×4 = **144**
- **Authors:** Haijiang Yan, Jian-Qiao Zhu, Adam Sanborn
- **Why it matters:** Bayesian framework explaining LLM planning
  dynamics: "self-generated context accumulation drives planning
  behavior shifts at inference time." **This is exactly the mechanism
  Fast-Slow Variant exploits** — the verifier-output-summary IS the
  self-generated context that accumulates across verify-repair
  iterations. Provides independent peer evidence that the
  fast-weight-context approach is mechanistically grounded, not just
  empirically motivated by arXiv:2605.12484. Useful paper-v6 §3 cite
  alongside the AR-LM↔EBM bijection and Fast-Slow papers.
- **Action:** cite in paper-v6 §3 (architecture validation). If .189
  exp1811 succeeds, this is a third independent literature anchor for
  the design pattern (Score 500 bijection + Score 400 Fast-Slow + Score
  144 inference-time-planning = ~triangulated theory base).
- **Caveat:** the paper is non-EBM-native (Bayesian framing, not energy);
  Carnot's specific verifier-energy mechanism is still novel relative
  to this work.

### Skipped (low score):
- **arXiv:2603.23398** Graph Energy Matching (Score 8) — molecular EBM,
  out-of-domain.
- **arXiv:2604.00555** Ontology-Constrained Neural Reasoning (Score 36) —
  enterprise agentic neurosymbolic, adjacent but not core.

### Pre-2026 references catalogued (no scoring; archived for citation tracking)

36 papers in the references-direction sweep span 2010 ("1004.2027" —
early relevant work) through 2025. Notable buckets include foundational
EBM papers (2010s), RL-from-feedback work (2017-2022), reasoning
benchmarks (2021-2024), and ICLR/NeurIPS 2024-2025 reasoning-model
papers. These are now in the known-set; future citation-sweeps from
other anchors will dedupe against them.

### Sweep takeaways

1. **Citation-following dwarfs keyword rotation in yield.** Zero promotions
   from the prior 2 keyword sweeps (16:50Z, 20:35Z); 1 in-domain
   promotion + 38 archived references from this single citation-sweep.
   Validates the operator-confirmed "operator-flagged anchors are the
   highest-yield channel" finding empirically.
2. **The bijection anchor (Score 500) was published Dec 2025; 5 months
   of citations means a meaningful citation graph already exists.** The
   Fast-Slow anchor (Score 400, May 2026) is too new to have meaningful
   citations yet (sweep_citations.py 404'd it earlier today). Citation-
   following yield scales with anchor age.
3. **Broadened-cluster sweeps need rate-limit care.** arXiv API 429'd
   on `max_results=20` with complex OR-chains; should drop back to
   `max_results=8` for the broadened cluster URLs (operator can paste
   them into the cron prompt or invoke via `sweep_clusters.py` with
   `--max-results 8`).
4. **arXiv:2602.02991 strengthens paper-v6 §3 architecture validation.**
   Three independent peer-reviewed mechanisms now point at the same
   design: AR-LM↔EBM bijection (2512.15605), Fast-Slow Training
   (2605.12484), and self-generated-context-driven planning shifts
   (2602.02991). Carnot's verify-repair loop sits at the intersection
   of all three.

## Sweep 2026-05-15T20:35Z (Claude outer-loop, hour-mod-4=0; clusters 1 EBM + 0 verifier-ensembles)

**Queries fired (rotated to clusters with productive history):**
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 8 fetched (5 known + 1 new + 2 out-of-domain)
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 fetched (8 known)
- HN skipped (4 prior 0-hits today; no broader-query authorization)

**Dedupe filter (2nd deployment of scripts/sweep_dedupe.py):**
16 candidates fetched, 13 already-known filtered at ingest, 3 truly-new
candidates surfaced. 53 IDs now in known-set (up from 49 at 16:50Z fire).

**Result: 0 NEW promotions.** All 3 newly-surfaced candidates marginal
or out-of-domain:

- **arXiv:2604.14733** "Differentiable Object Pose Connectivity Metrics
  for Regrasp Sequence Optimization" (Qin/Wan/Harada, Apr 2026).
  Score 1×3×1×1 = **3**. Robotic manipulation EBM, not LLM domain.
  Skipped.
- **arXiv:2602.03640** "Tutorial on Reasoning for IR & IR for Reasoning"
  (Hoveyda et al., Feb 2026). Score 3×2×2×2 = **24**. IR-context
  survey; mildly cites EBM approaches but no novel methodology for
  Carnot. Skipped.
- **arXiv:2601.02594** "Annealed Langevin Posterior Sampling (ALPS)"
  (Chand/Jacob, Jan 2026). Score 3×3×2×3 = **54**. Multiscale EBM
  for IMAGE inverse problems; out-of-domain BUT the annealed-Langevin
  sampling primitive is potentially adaptable to Carnot's THRML
  near-critical sampler failure (.175 exp1709 — fundamental limit at
  beta=1.05 unfixed in 54-cell ablation). Marginally relevant; note
  but don't promote.

### Sweep takeaways

1. **Saturation confirmed across 2 successive fires.** 16:50Z dedupe
   filter rate: 81% (17/21). 20:35Z dedupe filter rate: 81% (13/16).
   The 4 fixed-cluster queries have fully mapped the recent-window
   arxiv state. Future productive sweeps require either (a) the
   operator-discussed cluster-URL broadening (process-reward-model,
   token-energy, transcoder, predictive-coding terms), (b) extension
   to broader arxiv categories beyond cs.LG, OR (c) shift to a
   different signal channel (PaperWithCode, OpenReview venue tracking).
2. **arXiv:2601.02594 ALPS annealed-Langevin is the closest hit to
   exp1709's open question.** The near-critical sampler limit at
   beta=1.05 (no intervention closed the gap in 54-cell burn-in ×
   h_schedule ablation) is exactly the kind of failure mode that
   annealing schedules attack. NOT promoting to active queue
   (Score 54 too low) but flagging the cross-cite potential — if
   .190+ revisits exp1709 with ALPS-style multiscale annealing,
   this paper becomes the methodology reference.
3. **Operator-flagged additions remain the highest-yield channel.**
   The auto-rotation surfaced 0 promotions across 2 successive sweeps;
   meanwhile operator-flagged arXiv:2605.12484 (Fast-Slow, Score 400)
   from 13:15Z remains the single most impactful literature input of
   the day. The signal: routine arxiv rotation does not surface novel
   directions at the current state-of-the-art window; targeted
   operator review is where new ideas come from.

## Sweep 2026-05-15T16:50Z (Claude outer-loop, hour-mod-4=0; clusters 2/0/3 — FIRST with dedupe filter)

**Queries fired (rotated to clusters with stalest coverage):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 8 fetched
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 fetched
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 5 fetched
- HN skipped (3 prior 0-hits in 24h; no broader query yet authorized)

**Dedupe filter (NEW workflow step 2.5, first deployment):**
21 candidates fetched, 17 already-known filtered at ingest via
`python3 scripts/sweep_dedupe.py --filter`, 4 truly-new candidates
surfaced for scoring.

**Result: 0 NEW promotions.** All 4 newly-surfaced candidates either
out-of-domain or low-score:

- **arXiv:2602.19160** "LLM Reasoning from General Game Playing"
  (Świechowski et al., Feb 2026). Score 3x3x2x2 = **36**. LLM logical-
  error taxonomy in GGP environments; mildly applicable for adversarial
  corpus design but not critical path. Skipped.
- **arXiv:2602.18082** "AndroWasm" — Android malware obfuscation,
  out-of-domain false positive on the "specification gaming" filter
  (the paper uses the phrase in security context). Skipped.
- **arXiv:2601.23206** "Game content via small LMs" — game content
  generation, out-of-domain. Skipped.
- **arXiv:2605.12784** "ToolMol" drug discovery agentic framework —
  molecular agentic, out-of-domain. Skipped.

**Dedupe protocol validated.** Pre-dedupe, this sweep would have
re-scored 17 papers we've already ranked across the prior 7 sweeps
(2604.07650, 2605.02269, 2604.12500, 2603.28063, 2605.12874, 2605.14694,
2605.14449, 2605.14347, 2605.13930, 2605.12809, 2605.12770, 2605.12245,
2605.07639, 2605.12536, 2605.12495, 2605.11638, 2603.08806). Post-dedupe,
only 4 new candidates scored — saving ~70% of the prose budget per the
operator-confirmed efficiency win 2026-05-15.

### Sweep takeaways

1. **Dedupe filter works as designed** (17/21 = 81% filter rate). The
   protocol is now stable; future sweeps will report in the
   "N fetched / M known-skipped / P scored / Q promoted" compact format.
2. **The 3 out-of-domain false positives** (AndroWasm, game content,
   ToolMol) confirm the operator-precedence bug noted in the
   12:48Z sweep entry — narrow `abs:"phrase"` matches hit unrelated
   papers when the phrase has alternate meanings. Adding explicit
   AND-grouping (per the operator-discussed cron-URL upgrade) would
   filter these at fetch.
3. **0 promotions ≠ low-yield**. The 81% dedupe rate means the queue
   is well-mapped and stable. Operator-flagged manual additions
   (e.g., arXiv:2605.12484 Fast-Slow at Score 400) remain the
   highest-yield input channel for surfacing genuinely-new ideas
   the auto-rotation misses.

## Sweep 2026-05-15T12:48Z (Claude outer-loop, hour-mod-4=0; clusters 1 EBM + 3 active inference)

**Queries fired (rotated to clusters not covered in 08:45Z fire):**
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results (ALL re-hits from prior sweeps)
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 4 results (3 re-hits, 1 UAV-domain rejected)
- hn.algolia "energy EBM verifier" → 0 hits

**Result: 9 candidates; 0 NEW promotions; 8 re-hits + 1 rejected. Cluster saturation confirmed across 4 sweep rotations in 24h.**

### Rejected candidates (this sweep):

- **arXiv:2604.27935v1** — "Flying by Inference: Active Inference World Models for Adaptive UAV Swarms" (Arshid et al., Apr 30 2026). Score 2×4×2×2 = **32**. UAV swarms domain; hierarchical world-model active inference is mildly applicable to Carnot's verifier-as-free-energy framing but the domain gap is too large for replication value. **Worth noting**: the "hierarchical probabilistic inference + online KL minimization" structure echoes the alpha_t / alpha_t' computation Carnot is currently rescuing in .182 exp1745 — IF the .182 per-step disaggregation succeeds, this paper becomes a candidate cross-cite for paper-v6 §3.

### Saturation pattern (all 4 sweeps 2026-05-15)

The 4 cluster queries have been hit ~3 times each in 24h. Re-hit rate:
- Cluster 0 (verifier ensembles / spec gaming): 4 sweeps, 0 new since 04:42Z
- Cluster 1 (EBM + LLM): 3 sweeps, 0 new since 04:45Z
- Cluster 2 (SAE / white-box probe): 2 sweeps, last new 08:45Z (QAOD, exemplars, rate-distortion)
- Cluster 3 (active inference): 2 sweeps, 0 new since 04:45Z

The fixed `max_results=8` + the recent-paper bias of `sortBy=submittedDate` means each rotation re-fetches the same 5-8 papers until enough time passes for arXiv's listing to refresh.

### Recommended next-rotation broadening (do NOT modify CLAUDE.md per cron constraint, but the next-fire prompt could consider):

- Cluster 0 expansion: add `OR abs:"process reward model"` OR `abs:"deliberative alignment"`
- Cluster 1 expansion: add `OR abs:"token energy"` OR `abs:"energy guided decoding"`
- Cluster 2 expansion: add `OR abs:"feature attribution"` OR `abs:"transcoder"`
- Cluster 3 expansion: add `OR abs:"predictive coding"` OR `abs:"world model"`

Saturation is a healthy sign that Carnot's literature landscape is well-mapped at the current state-of-the-art window. Re-hits are NOT wasted; they confirm priority stability.

### Sweep takeaways

1. **Re-hit-only fire validates queue stability** — the active queue's top 5 (2512.15605, 2605.02269, 2605.14449, 2605.12536, 2605.14558) are not being displaced by new arrivals.
2. **Carnot's .182 per-step alpha disaggregation (exp1745) has no published peer methodology in this sweep window** — the arXiv:2604.27935 hierarchical-active-inference angle is the closest hit but UAV-domain. Carnot is operating in a literature-gap zone for this specific question.
3. **Next rotation should consider broader queries** per the suggestions above. Marking this as a sweep-mechanism observation, not a CLAUDE.md change.

## Sweep 2026-05-15T08:45Z (Claude outer-loop, hour-mod-4 rotation; cluster 2 SAE primary + cluster 0 verifier-ensembles)

**Queries fired (clusters 1/3 just covered in prior fire; rotating to cluster 2 SAE primary and cluster 0):**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 5 results
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 3 results (all already in queue)
- (cluster 1 EBM + cluster 3 active inference skipped — covered in 00:42Z + 04:45Z sweeps respectively)

**Result: 8 candidates scored; 3 NEW promotions; 5 re-hits acknowledged. Top score 320 (no >400 this fire).**

### NEW Rank HIGH: QAOD White-Box Hallucination Detection (arXiv:2605.14449v1, May 14 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Siyang Yao, Erhu Feng, Yubin Xia
- **Why it matters:** White-box probing framework using ORTHOGONAL
  decomposition of answer representations against question context.
  Reports "up to 21% improvement on BioASQ" for cross-domain
  hallucination detection. **Direct adversarial test against Carnot's
  NLA-via-SAE methodology** (exp1694/1720): orthogonal-decomposition
  may outperform the SAE-based NLA probe shipped as verifier #16 in
  .178. Worth a head-to-head comparison.
- **Action for .180+:** propose a "QAOD vs NLA-SAE probe head-to-head"
  experiment on the same gemma-4-26B-A4B-it-GGUF substrate Carnot
  already has loaded. If QAOD outperforms NLA-SAE by >5pp on the
  same 60-example test set used in exp1716, propose adding QAOD as
  verifier #17 (NOT as an NLA replacement — k=16 stays shipped).
- **Cross-reference:** complements exp1716 eval-awareness test
  (delta_tpr=-0.042 SAFE) — QAOD provides an alternative probe
  family that may have different eval-awareness characteristics.

### NEW Rank MEDIUM-HIGH: Exemplar Partitioning for Mechanistic Interpretability (arXiv:2605.14347v1, May 14 2026)
- **Score:** 4×5×3×4 = **240**
- **Authors:** Jessica Rumbelow (independent)
- **Why it matters:** Voronoi partitions of activation space as
  unsupervised alternative to SAE training, achieving "comparable
  interpretability with ~10³× fewer tokens." If the 1000× efficiency
  claim transfers, Carnot could replace the exp1694-trained SAE (1k
  calibration corpus) with a Voronoi-partition probe trained on ~10
  examples — drastically cheaper for the kind of small-corpus
  domain-specific verification Carnot does.
- **Action:** treat as the cheaper-substrate alternative to the
  current NLA-SAE. If the .180+ QAOD comparison shows SAE is the
  weaker probe, Voronoi-partition is the next candidate replacement
  rather than re-training a larger SAE.
- **Cross-reference:** orthogonal to QAOD; both are "skip the SAE"
  alternatives.

### NEW Rank MEDIUM: Rate-Distortion-Polysemanticity Tradeoff in SAEs (arXiv:2605.14694v1, May 14 2026)
- **Score:** 4×4×3×3 = **144**
- **Authors:** Tommaso Mencattini, Francesco Montagna, Francesco Locatello
- **Why it matters:** Formal rate-distortion analysis of the
  polysemanticity tradeoff in SAEs. Shows enforcing interpretability
  necessarily increases both rate AND distortion; polysemanticity
  is driven by training-data distribution characteristics. Carnot's
  NLA-SAE probe has 704 active features (exp1694); this paper's
  tradeoff curve tells us where 704 sits on the
  reconstruction-vs-monosemanticity spectrum and whether scaling up
  is even productive.
- **Action:** cite in paper-v6 §3 NLA-probe methodology section.
  Not a near-term experiment but informs SAE sizing for future
  Carnot probes.

### Re-hits of papers already in queue (no action):

- arXiv:2605.12874 (Descriptive Collision in SAE Auto-Interpretability) — promoted in 2026-05-14T04:15Z sweep (URGENT)
- arXiv:2604.07650 (Behavioral Entanglement Verifier Ensembles) — promoted .144
- arXiv:2605.02269 (Spec Gaming in Reasoning Models) — promoted 00:42Z, Score 320
- arXiv:2604.12500 (Safety Training under On-Policy RL) — Score 36, skipped 3x
- arXiv:2605.13930 (SAE on EEG Foundation Models) — out-of-domain (EEG, not LLM)

### Sweep takeaways

1. **NLA-probe landscape has 3 alternative families** now visible:
   (a) Carnot's current SAE-based NLA (exp1694 shipped, k=16
   production), (b) QAOD orthogonal-decomposition (this sweep,
   Score 320), (c) Voronoi-partition exemplars (this sweep, Score 240).
   The head-to-head comparison is a clear .180+ task and would
   strengthen paper-v6 §3.
2. **No score>400 this fire** — converging toward operational
   refinement (head-to-head probes; rate-distortion analysis) rather
   than fundamental new directions. This is healthy: it means
   Carnot's high-level architecture stays competitive with the
   literature; what's left is engineering refinement.
3. **The verifier-ensemble cluster has saturated** in the recent
   window — all 3 results were repeats from prior sweeps. Next
   rotation may need to broaden the query (add "process reward
   model" or "deliberative alignment" as adjacent terms).

## Sweep 2026-05-15T04:45Z (Claude outer-loop, hour-mod-4 rotation, clusters 3/0/1)

**Queries fired (cluster 3 active-inference skipped last fire — picked up this fire):**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 3 results
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 3 results (all 3 already in queue from prior sweeps)
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results (4 already in queue; 1 new from 2026-05-14)
- hn.algolia "verifier energy LLM" → 0 hits

**Result: 11 candidates scored; 2 NEW promotions (both Score 192); 9 re-hits acknowledged. No score>400 this fire.**

### NEW Rank MEDIUM-HIGH: Token-Level Energy for Agentic RL (arXiv:2605.14558v1, May 14 2026 — yesterday)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Langzhou He, Junyou Zhu, Yue Zhou, Zhengyao Gu, Junhua Liu,
  Wei-Chieh Huang, Henry Peng Zou, David Wipf, Philip S. Yu, Qitian Wu
- **Why it matters:** Token-level energy-based credit assignment in agentic
  RL reveals that training signals concentrate on action tokens despite
  their scarcity. Proposes the **ActFocus** reweighting mechanism, reporting
  a 65.2pp gain over PPO with no computational overhead. Directly relevant
  to Carnot's FR-11 (verifier-as-reward RL) work which has been an open
  retro question across .96-.150+. The token-level energy framing is
  compatible with Carnot's verifier-output-as-energy interpretation; if the
  ActFocus reweighting transfers to verifier-driven RL, it could be a
  near-term lift.
- **Caveat:** the 65.2pp PPO gain is a SUBSTANTIAL claim; should be treated
  as adversarial-verify-worthy if Carnot replicates. Replication budget:
  one Carnot agentic-RL experiment with + without ActFocus reweighting on
  the same FR-11-style verifier signal.
- **Action:** queue for .177+ as a candidate FR-11 follow-up experiment.

### NEW Rank MEDIUM-HIGH: IIT ↔ FEP Maximum-Caliber Bridge (arXiv:2605.12536v1, May 3 2026)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Alexander Kearney
- **Why it matters:** Establishes mathematical connection between the Free
  Energy Principle (Phase 4 substrate) and Integrated Information Theory
  through maximum-caliber variational principles. Demonstrates that
  information emerges from prediction error under predictive coding.
  Directly relevant to Carnot's Phase 4 active-inference framing — the
  alpha_t metric needs theoretical grounding (the exp1693 suspicious
  invariance + the AR-LM↔EBM bijection paper suggest alpha_t may be
  bijection-invariant by construction). This paper's maximum-caliber
  framing may provide an alternative derivation of alpha_t that is NOT
  bijection-invariant — worth investigating before .176 exp1715 retries
  the alpha_t audit.
- **Action:** cite in paper-v6 §3 (Phase 4 theoretical framing) alongside
  arXiv:2512.15605. If exp1715 confirms bijection-invariance artifact,
  the maximum-caliber derivation in this paper is the replacement candidate.

### Re-hits of papers already in queue (no action):

- arXiv:2604.07650 (Behavioral Entanglement) — promoted .144
- arXiv:2605.02269 (Spec Gaming in Reasoning Models) — promoted 04:42Z sweep
- arXiv:2604.12500 (Safety Training under On-Policy RL) — Score 36, skipped twice
- arXiv:2512.15605v3 (AR-LMs are Secretly EBMs) — promoted 00:42Z sweep, Score 500
- arXiv:2512.18730v1 (Theoretical Lens RL-Tuned LLMs) — promoted, Score 192
- arXiv:2601.21064v3 (Textual Equilibrium Propagation) — noted, Score 144
- arXiv:2602.18671v4 (Spilled Energy) — already integrated as `verify_spilled_energy`

### Sweep takeaways

1. **No score>400 this fire** — the queue is converging on the AR-LM↔EBM
   bijection (2512.15605) as the top theoretical anchor. Two new Score-192
   candidates (token-level energy for agentic RL; IIT↔FEP bridge) are
   complementary: ActFocus is a near-term operational lift; IIT↔FEP is
   theoretical framing for Phase 4 alpha_t derivation.
2. **Active inference cluster has thinned out** — only 3 results in the
   most-recent listing, of which 2 are weakly LLM-related. The cluster
   may need a broader query (e.g., add "predictive coding", "perception-
   action loop", "world model").
3. **exp1709 finding from .175 ALREADY beats the literature on its
   specific question** — analytic Curie-Weiss ground-truth comparison at
   n=128 with 10k samples isn't matched in any of the sampling papers
   surveyed this fire. The .176 exp1714 codification has at least one
   independently-novel contribution.

## Sweep 2026-05-15T00:42Z (Claude outer-loop, hour-mod-4 rotation, clusters 0/1/2)

**Queries fired (3 of 4 cluster rotation; cluster 3 active-inference skipped this fire):**
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 4 results
- arxiv abs:"energy based model" AND ("reasoning" OR "verification" OR "LLM") → 5 results
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM" → 5 results
- hn.algolia "verifier energy LLM" → 0 results (no HN front-page activity this window)

**Result: 14 candidates scored; 4 NEW promotions (top score 500 — score>400 noted to known-issues per protocol).**

### NEW Rank URGENT: Autoregressive LMs are Secretly EBMs (arXiv:2512.15605v3, Dec 2025; v3 update May 2026)
- **Score:** 5×5×4×5 = **500**
- **Authors:** Mathieu Blondel, Michael E. Sander, Germain Vivier-Ardisson, Tianlin Liu, Vincent Roulet (Google DeepMind, INRIA, EPFL collaboration)
- **Why it matters URGENT:** Establishes an explicit BIJECTION between
  autoregressive language models and energy-based models, and connects
  both to maximum-entropy RL. Provides theoretical error bounds for
  DISTILLING an EBM into an AR-LM. Directly relevant to Carnot's
  Phase-3 endgame ("evolve into a foundation model based on hardware-
  acceleratable EBM/EBT"). The bijection means our work translating
  between LLM outputs and EBM energy IS a well-defined map — not a
  bolt-on. The distillation error bounds may give Phase 3 a clean
  acceptance gate (distillation gap small → AR-LM and EBM are
  operationally equivalent).
- **Action for paper-v6:** cite as §3 peer methodology AND §6 theoretical
  framing. The bijection is exactly the formal scaffolding Carnot needed.
- **Action for Phase 3:** add a milestone task that re-derives Carnot's
  verifier-as-free-energy interpretation through the AR-LM↔EBM bijection
  in this paper. The Phase 4 active-inference suspicious-invariance
  finding from exp1693 may be a corollary of this bijection (alpha_t
  computed in a way that's invariant to substrate size because both
  representations are operationally equivalent under the bijection).
- **Score-gate cross-reference:** also added to ops/known-issues.md under
  RESEARCH-STUDYING CANDIDATES per the score>400 protocol.

### NEW Rank HIGH: Specification Gaming in Reasoning Models (arXiv:2605.02269v1, May 4 2026)
- **Score:** 5×4×4×4 = **320**
- **Authors:** Kei Nishimura-Gasparian, Robert McCarthy, David Lindner (Lindner is at Anthropic)
- **Why it matters:** Open-source evaluation suite demonstrating "all
  tested models exploit their specifications at non-negligible rates"
  across diverse settings. RL reasoning training INCREASES exploitation
  rates; test-time mitigations only partially reduce. Directly tests
  whether Carnot's k=15 verifier ensemble's null-space resilience holds
  up against deliberate spec-gaming (vs unwitting hallucinations).
- **Action:** when Phase-3 substrate is ready, run Carnot's k=6/k=15
  verifier ensemble against this suite. The k=6→k=15 lift is the
  empirical handle on null-space-mimicry defence (cf. memory entry
  project_null_space_mimicry_attack.md).
- **Cross-reference:** complements arXiv:2603.28063 (next entry) which
  proves the theoretical inevitability — together they bracket the
  empirical-vs-theoretical sides of spec gaming.

### NEW Rank MEDIUM-HIGH: Theoretical Lens for RL-Tuned LLMs via EBMs (arXiv:2512.18730v1, Dec 2025)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Zhiquan Tan, Yinrong Hong
- **Why it matters:** Theoretical analysis of KL-regularized RL for LLMs
  using CLOSED-FORM EBM structures, analyzing "verifiable rewards"
  through the lens of optimal reasoning distributions. Provides
  convergence properties for instruction-tuned models. Carnot's
  FR-11 (verifier-as-reward) work has empirical retros from
  .96-.150+; this paper's closed-form EBM framework is the
  theoretical scaffolding those retros were missing.
- **Action:** cite in paper-v6 §3 (FR-11 methodology) AND §6 (theoretical
  underpinning for verifiable-reward RL convergence claims).

### NEW Rank MEDIUM-HIGH: Orthogonal Latent Spaces SAE for Token Influence (arXiv:2605.12809v1, May 12 2026)
- **Score:** 4×4×3×4 = **192**
- **Authors:** Shixing Yu, Promit Ghosal, Kyra Gan
- **Why it matters:** SAE-based token-influence attribution with EXPLICIT
  orthogonality constraint on latent features + Jacobian-vector products
  for "non-decomposable" latent influence. Directly informs Carnot's
  NLA 16th verifier 4-task chain — exp1694 (.171) shipped TPR=0.73 with
  704 active SAE features but did NOT enforce orthogonality. Adding the
  orthogonal-latent constraint may improve eval-awareness robustness
  (which exp1700 will measure in .172).
- **Action:** if exp1700 (.172) NLA eval-awareness test detects a gap >
  5pp, the .173 NLA prototype v4 should adopt the orthogonal-latent
  constraint from this paper. The Jacobian-vector products methodology
  may also reduce per-example latency below exp1694's 150ms p50.

### Additional candidates noted (score 80-150; not promoted to active queue):

- **arXiv:2601.21064v3** — Textual Equilibrium Propagation (Chen, Deng, Zou, Yu, Li; Jan 2026 v3). Score 144. Equilibrium-prop inspired LLM workflow optimization; relevant to EBT path but replication non-trivial.
- **arXiv:2603.28063v1** — Reward Hacking as Equilibrium (Wang, Huang; Mar 2026). Score 144. Proves spec gaming is a "structural equilibrium, not a correctable bug" — bracket-citation with arXiv:2605.02269 above.
- **arXiv:2511.21882v1** — Closed-Loop / Equilibrium Transformers (Anbar Jafari; Nov 2025). Score 144. Iterative latent refinement via energy minimization; parallel to Carnot's EBT direction.
- **arXiv:2605.12055v1** — Linguistic Constraint Violations via SAE (Hardy, Padó; May 12 2026). Score 81. Negative result — limited evidence for unified violation detectors. Informs NLA: should NOT expect one feature per verifier class.

### Re-hits of papers already in queue (no action):

- arXiv:2604.07650 (Behavioral Entanglement) — already promoted in 2026-05-14 sweep #4
- arXiv:2602.18671 (Spilled Energy v4 update) — already partially integrated as `verify_spilled_energy` method; v4 = newer revision but no new claims
- arXiv:2605.12874 (Descriptive Collision in SAE) — already promoted in 2026-05-14T04:15Z sweep #2

### Sweep takeaways

1. **2512.15605 is the highest-score sweep result in Carnot's literature
   record to date** (500 > prior top scores of 400). The AR-LM↔EBM
   bijection is the theoretical scaffolding Phase 3 was missing and
   may explain the exp1693 alpha_t suspicious-invariance finding.
2. **Spec-gaming corpus is converging on "structural equilibrium" framing**
   (2605.02269 empirical + 2603.28063 theoretical). Carnot's null-space-
   mimicry defence is the right thing to test against this corpus.
3. **SAE methodology is maturing fast** (3 May 2026 SAE papers in this
   sweep alone). The NLA 4-task chain should explicitly track this
   sub-literature; .173 prototype v4 should adopt orthogonality
   constraints if .172 eval-awareness detects a gap.

## Sweep 2026-05-14T20:45Z (Claude outer-loop /loop job 875c06b4 fire #6)

**Queries fired:**
- arxiv abs:"active inference" OR "free energy" AND "LLM" → 8 results (this hour the API responded)
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming" → 8 results

**Result: 1 NEW candidate promoted; 2 re-hits of papers already in queue.**

### NEW Rank MEDIUM: AlphaGRPO Decompositional Verifiable Reward (arXiv:2605.12495, May 12 2026)
- **Score:** 4×4×3×3 = **144**
- **Authors:** Runhui Huang, Jie Wu, Rui Yang
- **Why it matters:** Introduces "Decompositional Verifiable Reward" that
  decomposes requests into verifiable semantic queries during GRPO training.
  Structurally similar to Carnot's NSVIF constraint extraction (DSL →
  PySAT/Z3 verifiable constraints). Worth investigating whether the
  decomposition primitive transfers to Carnot's pipeline, OR whether the
  reverse — Carnot's NSVIF-style constraint extraction — could enhance
  AlphaGRPO's reward decomposition. For paper-v6 §3 peer mention.

### Repeat hits (already in queue from prior sweeps, no action needed)
- arXiv:2604.07650 Behavioral Entanglement (already Score 400)
- arXiv:2605.02269 Specification Gaming in Reasoning (already Score 300)
- arXiv:2605.11638 U-Statistics with Active Inference (already Score 36, not promoted)
- arXiv:2605.07639 Tacit Knowledge Extraction (already not promoted)
- arXiv:2605.12536 Maximum-Caliber Deviation (already Score 48)

### Sweep-#6 takeaways
- arxiv API is responsive again this hour (vs the 429/timeout streak earlier today)
- Yield is low because we've already harvested the high-relevance recent
  submissions in prior sweeps. The corpus refreshes weekly-ish on arxiv;
  expect sweep-#7+ to be similarly thin until the next batch of relevant
  preprints lands.

---

## Sweep 2026-05-14T16:55Z (Claude outer-loop /loop job 875c06b4 fire #5)

**Queries fired:**
- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" → **HTTP 429**
- arxiv abs:"formal verification" AND "LLM" → **HTTP 429**
- HN search: `energy verifier hallucination` → 0 hits
- HN search: `energy based model` → same 10 results as sweep #4 (no churn)
- Semantic Scholar API → **HTTP 429**

**Result: low-yield fire.** Both arxiv and Semantic Scholar API rate-limited this hour. HN hadn't churned since sweep #4. No new candidates promoted.

**Operational observation:** the cron is firing every 4 hours but arxiv's API has been 429-throttled on 2 of the last 3 fires. This is consistent with the rest-of-the-world also hammering it. Recommendation for cron-prompt revision: stagger by day_of_year mod 4 AND add a 60-90s delay between WebFetches to spread the request load.

---

## Sweep 2026-05-14T12:40Z (Claude outer-loop /loop job 875c06b4 fire #4)

**Queries fired:**
- arxiv abs:"probabilistic computing" OR "Ising machine" OR "stochastic circuit" OR "p-bit" → **timeout 60s**
- arxiv cat:cs.LG AND abs:"Ising" AND abs:"sampling" → **HTTP 503 Service Unavailable**
- HN search: `energy based model` (timestamp > 1746500000)

**arxiv API is degraded this fire.** Both queries failed (one timeout, one 503).
HN was the only successful source.

**Candidates surfaced (10 HN stories, 1 promoted):**

### NEW Rank LOW-MEDIUM: Kona EBM Sudoku Benchmark (logicalintelligence.com, Feb 2026)
- **Score:** 4×3×4×3 = **144**
- **Source:** HN (2 points, low signal, but content is high-signal)
- **URL:** logicalintelligence.com/blog/energy-based-model-sudoku-demo
- **Why it matters:** Headline claim **96% vs 2% on Sudoku** — Carnot's Phase-3
  parity target (per CLAUDE.md Project Vision: "Functional parity with Kona")
  has new published benchmark numbers. Worth checking:
  1. Whether Carnot's existing sudoku verifier (per `python/carnot/verify/sudoku.py`)
     can run the same benchmark format
  2. The "96%" methodology: is it solve-rate, verification-accuracy, or
     constraint-satisfaction? Kona is closed-weight; Carnot adaptation TBD.
- **Status:** NOT pre-stage material (Phase-3 parity is a long-horizon target;
  current focus is Phase 1 ship + adversarial-verify rigor).

### Additional HN candidates (not promoted; score < 100):
- ebmsovereign.com Energy-Guard OS — 88.7% leak detection EBM (security domain,
  off-topic for Carnot's reasoning verification mission)
- 2024 iopscience.iop.org "Introduction to latent variable EBMs" — foundational,
  already implicitly used in Phase-3 DBAE-EBM design (memory)
- "Logical Intelligence" startup page (LeCun-linked, EBM-based) — already cited
  in CLAUDE.md
- Other HN hits: self-promotional / off-topic (Sudoku demos, anomaly detection
  blog posts, YouTube content)

### Sweep-#4 takeaways

- **arxiv API health is unreliable** at this hour. Two queries failed in a
  row. Future fires should be defensive: try one arxiv query, fall back to
  HN/openreview/semantic scholar promptly.
- **HN-only fires have lower yield**. arxiv is the load-bearing source for
  serious literature.
- Worth augmenting the cron prompt with a semantic-scholar fallback URL:
  `https://api.semanticscholar.org/graph/v1/paper/search?query=...&limit=10&fields=title,abstract,year,citationCount`

---

## Operator-flagged 2026-05-14T11:00Z: Iron Layer (github.com/bwahacker/iron-layer)

**Source:** Operator-flagged during conversation, not via sweep. Worth recording as peer methodology.

### Iron Layer — Prompt-Injection Honeypot Labeler (bwahacker, ~Apr 2026)
- **Score:** 4×3×4×3 = **144** (moderate; not pre-stage material but worth tracking)
- **Repo:** github.com/bwahacker/iron-layer
- **What it is:** Detonates untrusted text inside an isolated sandbox with a canary LLM (Claude Haiku 4.5) wired to a wildcard MCP server that **fakes** dangerous tool execution. Records which dangerous-intent buckets (`filesystem-read`, `code-exec`, `network-egress`, `exfil-email`, `secret-access`, etc.) the injected text coaxed out. Output: JSONL pairs `(raw_input_text → tickled_signals)` for downstream classifier training (Featrix).
- **Why it matters to Carnot (5 angles):**
  1. **Adversarial corpus generator for exp2102 NLA probe v2.** Pre-staged
     `.165 task explicitly needs n>=30 adversarial examples spanning factual /
     logical / arithmetic classes. Iron Layer's JSONL output format is
     directly suitable as adversarial eval data. Candidate corpus source.
  2. **Signal normalization pattern.** Iron Layer's clean bucket taxonomy
     (`filesystem-read`, `code-exec`, etc.) could inspire a similar
     normalization layer for Carnot's verifier-ensemble output schema in
     paper-v6 §3.
  3. **Sandbox-canary pattern.** Structurally similar to Carnot's
     `CARNOT_USE_SANDBOX=1` gvisor pattern. When Carnot eventually runs
     adversarial outputs through the verifier ensemble, an Iron-Layer-style
     canary sandbox would let us "execute" suspect outputs without
     real-world side effects.
  4. **Deterministic lures via hashing.** Same principle as Carnot's
     `reproducibility_checksum`. Prior art for hash-deterministic adversarial
     inputs — worth citing in the Adversarial Artifact Verification CLAUDE.md
     rule.
  5. **Specification gaming connection.** Iron Layer is the operational
     artifact of arXiv:2605.02269 ("Specification Gaming in Reasoning
     Models") — the injection text gamings the canary's apparently-helpful
     behavior. Reinforces the load-bearing nature of that paper.
- **Decentralization concern:** uses closed-weight Claude Haiku 4.5 for the
  canary. Per CLAUDE.md decentralization Rule 1 (local-first using open
  models), Carnot adaptation would need a local model (Qwen3.5-0.8B as the
  canary). Adapting is straightforward — the methodology is model-agnostic.
- **License unknown** from README excerpt; need to verify before integration.
- **Status:** NOT pre-stage material — Phase 1 ship, Phase 4 active inference,
  THRML parity v2, and NLA probe v2 are higher-priority. Recorded here for
  reference if exp2102 NLA probe v2 needs more adversarial data than we can
  synthesize manually, OR if a future "verifier-ensemble taxonomy paper" milestone
  draws on it as peer methodology.

---

## Sweep 2026-05-14T08:20Z (Claude outer-loop /loop job 875c06b4 fire #3)

**Queries fired:**
- arxiv abs:"test time compute" AND (verification OR reasoning OR sampling) → **HTTP 429 rate-limited**
- HN search: `LLM verifier` (timestamp filter created_at_i > 1746000000)
- arxiv abs page: 2512.02080 (deep-dive on top HN hit)
- arxiv listing /list/cs.LG/2605 → HTTP 404 (URL format issue; skipped)

**Candidates surfaced (10 HN stories, 1 deep-dive, 2 promoted):**

### NEW Rank URGENT: The 4/δ Bound — Predictable LLM-Verifier Convergence (arXiv:2512.02080, Dec 2025)
- **Score:** 5×5×4×4 = **400**
- **Authors:** Pierre Dantas, Lucas Cordeiro, Youcheng Sun, Waldir Junior
- **Surfaced via:** Hacker News (59 points, 13 comments)
- **Why it matters:** This is a **theoretical convergence bound for verifier-loop
  systems with formal guarantees** — exactly the kind of architectural
  justification paper-v6 §3 needs. Models the LLM-verifier loop as an
  absorbing Markov chain with 4 stages (CodeGen → Compilation →
  InvariantSynth → SMTSolving) and proves: (1) termination for any δ > 0
  success rate per stage, (2) expected latency E[n] ≤ 4/δ iterations.
  Validated over 90,000 trials. **Carnot's verify-repair pipeline is
  structurally this exact architecture** (different specific stages, but
  same absorbing-Markov-chain shape). Citing this paper grounds Carnot's
  pipeline in published convergence theory rather than empirical hand-wave.
- **Action items:**
  1. Cite in paper-v6 §3 architecture lineage discussion
  2. Compute Carnot's empirical δ (stage success rate) from recent
     verify-repair runs and validate against the 4/δ prediction
  3. The "three operational zones (marginal, practical, high-performance)"
     calibration strategy is directly applicable to Carnot's tier system

### NEW Rank HIGH: BEAVER — Efficient Deterministic LLM Verifier (arXiv:2512.05439, Dec 2025)
- **Score:** 5×4×5×3 = **300**
- **Surfaced via:** Hacker News
- **Why it matters:** Carnot already has a "BEAVER-lite" task in `.147 (exp1879
  Deterministic Bounds for Validators ran OK). This is the **source paper**.
  Confirms our existing implementation is literature-grounded; should be
  cited in paper-v6 alongside Spera Theorem 9.2 + the 4/δ Bound. Worth
  reading the full paper to identify any features we're missing in the
  exp1879 implementation.

### Additional HN candidates (not promoted; below score 200):
- **Aura-State** (GitHub, Mar 2026, 23 pts) — Formally verified LLM state
  machine compiler using Z3 + CTL model checking. Z3 is already in Carnot;
  conceptual sibling. Score ~48.
- **Terminal-Bench-RL** (GitHub, July 2025, 125 pts) — "Hybrid reward
  signal of unit test verifiers & a behavioural LLM judge." Adjacent
  verifier-design pattern. Score ~36.
- **VR.dev** (Show HN, Mar 2026, 3 pts) — HARD/SOFT/AGENTIC verifier
  taxonomy; "deterministic probes against databases." Adjacent.
- **Sigma Guard** (Show HN, May 2026, 3 pts) — Cellular sheaf cohomology
  for consistency verification. Mathematically interesting, far from Carnot's
  current path.
- 5 others (Pencil Puzzle Bench, PupiBot1.0 triple-agent, Maestro orchestrator,
  Probus AI vuln scanner) — off-topic or low-signal.

### Cron-prompt bug status

Fire #3 confirmed the rotation issue: `hour mod 4` at fire-times :13 every 4h
always equals 0. Manual cluster selection used each fire instead. Should fix
the prompt formula to `day_of_year mod 4` OR `fire_counter mod 4` in a future
cron-prompt revision. Flagged but not fixed this fire.

---

## Sweep 2026-05-14T04:15Z (Claude outer-loop /loop job 875c06b4 fire #2)

**Queries fired** (rotated to clusters 2+3 since the rotation formula `hour mod 4`
in the cron always lands on the same residue when fires are every 4h — flagged
as a cron-prompt bug to fix later):

- arxiv abs:"sparse autoencoder" OR "white box probe" OR "reconstruction error" AND "LLM"
- arxiv abs:"active inference" OR "free energy" AND "LLM"

**Candidates surfaced (13 raw, 3 promoted to ranked queue below):**

### NEW Rank URGENT: Descriptive Collision in SAE Auto-Interpretability (arXiv:2605.12874, May 13 2026)
- **Score:** 5×5×4×4 = **400**
- **Author:** Jordan F. McCann
- **Why it matters:** **Direct adversarial-verify critique of SAE-based
  interpretability** — the foundational technique for Carnot's NLA-class 16th
  verifier (per `feedback_nla_class_16th_verifier_committed.md`). McCann shows
  that distinct SAE features receive identical text-descriptions, inflating
  reported interpretability by ~⅓ of feature identity bits. **This means the
  exp1851 NLA probe v2 must include a description-collision check before
  claiming the SAE features actually discriminate adversarial outputs.** If
  the 16-features-with-1-description pattern shows up in our SAE, the TPR
  lift claim is artificial (the probe is detecting feature-class identity,
  not output-distinguishing signal).
- **Action items:**
  1. Add a `feature_description_collision_rate` check to the planned exp2102
     NLA probe v2 artifact schema
  2. Cite McCann in paper-v6 §6 as a methodology limitation for white-box
     verifiers
  3. Sanity-check whether existing carnot SAE code (if any) already audits
     for this

### NEW Rank MEDIUM: Domain Restriction via Multi SAE Layer Transitions (arXiv:2605.11920, May 12 2026)
- **Score:** 4×4×4×3 = **192**
- **Authors:** Elias Shaheen, Avi Mendelson
- **Why it matters:** OOD detection via cross-layer SAE activation analysis.
  Carnot's verifier ensemble is structurally an OOD-detection problem for
  LLM outputs (verifier ensemble = "is this output in the support of valid
  outputs?"). Cross-layer SAE signal as an OOD verifier is a natural fit for
  the 16th verifier methodology. Lower priority than McCann because it's a
  technique to adopt rather than a methodology critique to defend against.

### NEW Rank MEDIUM: Do LMs Encode Linguistic Constraint Violations? (arXiv:2605.12055, May 12 2026)
- **Score:** 4×3×3×3 = **108**
- **Authors:** Hardy, Sebastian Padó
- **Why it matters:** "Employs sparse autoencoders to investigate whether
  LLMs encode grammatical violation detection through monosemantic feature
  activation patterns." Direct conceptual sibling to Carnot's verifier
  ensemble — constraint-violation detection via white-box SAE features. If
  the answer is yes, that's a baseline Carnot can adopt for the linguistic-
  constraint class of verifiers. If no, that's a known limitation.

### Additional candidates (not promoted; rank below 100):
- arXiv:2605.12809 Correcting Influence: Unboxing LLM Outputs (Yu, Ghosal, Gan) — SAE-based training-data attribution; tangential to verification
- arXiv:2605.12770 WriteSAE: SAEs for Recurrent State (Young) — SSM-targeted; relevant if Phase-3 substrate becomes recurrent
- arXiv:2605.12245 SOAR: Scale Optimization for NVFP4 Quantization — efficient deployment, not verification
- arXiv:2605.12225 Mechanistic Interpretability of ASR — audio domain
- arXiv:2605.12122 Disentangled Sparse Representations for Diffusion Unlearning — diffusion concept suppression
- arXiv:2605.11638 Learning U-Statistics with Active Inference — statistical estimation, not LLM
- arXiv:2605.07639 Tacit Knowledge Extraction via Logic Augmented Generation + Active Inference — adjacent to NSVIF
- arXiv:2605.12536 Information as Maximum-Caliber Deviation (Kearney) — FEP/IIT bridge, theoretical only
- arXiv:2605.01290 How Light Reshapes the Mind — cognition modeling, off-topic
- arXiv:2604.27935 Flying by Inference: UAV Swarms — robotics, off-topic

---

## Sweep 2026-05-14T00:42Z (Claude outer-loop, manual seed for /loop job 875c06b4)

**Queries fired:**
- arxiv abs:"verifier ensemble" OR "null space attack" OR "specification gaming"
- arxiv abs:"energy based model" AND (reasoning OR verification OR LLM)

**Candidates surfaced (10 raw, 4 promoted to ranked queue below):**

### NEW Rank URGENT: Behavioral Entanglement + Reweighting Verifier Ensembles (arXiv:2604.07650, Apr 2026)
- **Score:** 5×4×4×5 = **400**
- **Authors:** Kuai, Jiang, Zhu et al.
- **Why it matters:** Directly addresses Carnot's k=15 AND-composition null-space concern (per Spera Theorem 9.2 memory). Demonstrates that "correlated reasoning patterns and synchronized failures undermine ensemble verification" — i.e., the joint-null-space attack is empirically observable, not just theoretically possible. Reports up to **4.5% accuracy lift** from de-entangled reweighting of verifier ensembles. This is a load-bearing peer paper for paper-v6's Phase-3 architecture justification.
- **Carnot integration path:** Phase-3 verifier ensemble could adopt reweighting; current uniform-weight AND-composition is what the paper shows is suboptimal. Worth a dedicated milestone task to replicate the reweighting algorithm on Carnot's k=15 setup and measure the lift on a held-out adversarial corpus.

### NEW Rank URGENT: Spilled Energy in LLMs (arXiv:2602.18671, Feb 2026)
- **Score:** 5×4×5×4 = **400**
- **Authors:** Minut, Dewidar, Masi
- **Why it matters:** "Reinterprets LLM softmax classifiers as EBMs to detect hallucinations using training-free metrics derived from output logits without requiring probe classifiers." This is structurally identical to Carnot's verifier-energy philosophy — energy as verification, no labels required. Strong methodological peer. May provide a baseline to compare Carnot's verifier ensemble against on standard hallucination benchmarks. **Already partially used in Carnot** (verify_spilled_energy is a method in VerifyRepairPipeline per the conductor's AST signatures); confirm coverage + cite in paper-v6 §3 as a peer methodology.

### NEW Rank HIGH: Autoregressive LMs are Secretly EBMs (arXiv:2512.15605, Dec 2025)
- **Score:** 5×5×5×3 = **375**
- **Authors:** Blondel, Sander, Vivier-Ardisson
- **Why it matters:** Theoretical foundation — "Mathematical equivalence between autoregressive models and EBMs, revealing lookahead capabilities in next-token prediction." Supports Carnot's premise that any LLM output admits an energy interpretation, and therefore can be verified via energy. Cite in paper-v6 §3 architecture lineage; Phase-3 substrate justification.

### NEW Rank HIGH: Specification Gaming in Reasoning Models (arXiv:2605.02269, May 2026)
- **Score:** 5×4×3×5 = **300**
- **Authors:** Nishimura-Gasparian, McCarthy, Lindner
- **Why it matters:** "RL reasoning training increases exploitation rates of model specifications." Specification gaming is precisely the failure mode Carnot's adversarial-verify caught on exp1851 (3.4s wall time with TPR=1.0). This paper formalizes that the SOTA RL-trained models (Qwen3.6 GRPO etc.) are MORE prone to gaming the verifier signal — i.e., Carnot's verifier needs to be MORE adversarially robust on the current SOTA than on prior generations. Cite in paper-v6 §6 limitations + adversarial-verify CLAUDE.md rule.

### Additional candidates (not promoted; rank below 300):
- arXiv:2603.28063 Reward Hacking as Equilibrium (Wang, Huang) — theoretical unification of specification gaming; relevant to paper-v6 §6 but not actionable
- arXiv:2511.21882 Closed-Loop Transformers (Anbar Jafari) — iterative energy refinement; Phase-3 EBT track relevance, but the abstract is thin on numerical results
- arXiv:2601.21064 Textual Equilibrium Propagation (Chen, Deng, Zou) — workflow optimization not verification; adjacent
- arXiv:2604.12500 Safety Training Modulates Misalignment (Eshuijs et al.) — environment-design effects; less direct
- arXiv:2512.18730 RL-Tuned LLMs via EBMs (Tan, Hong) — theoretical, less actionable
- arXiv:2603.08806 Test-Driven AI Agent Definition (Rehan) — agent design, off-topic

---


## How This Works

1. Claude searches arxiv, OpenReview, GitHub, Extropic, Semantic Scholar, HN
2. Each finding is ranked by: relevance × novelty × feasibility × urgency
3. Top ideas are promoted to `research-roadmap-next.yaml` when a slot opens
4. Lower-ranked ideas stay here for future consideration
5. Ideas that prove irrelevant are moved to "Archived"

## Ranking Criteria

- **Relevance (1-5):** How directly does this apply to Carnot's current gaps?
- **Novelty (1-5):** Is this a new approach we haven't tried?
- **Feasibility (1-5):** Can we implement this in 1-2 experiments?
- **Urgency (1-5):** Does our current research depend on this?
- **Score = R × N × F × U** (max 625)

## Active Research Queue (Ranked)

### Rank 0a-prime: Fast-Slow Training (FST) — Carnot's Verify-Repair Loop Validated as Dual-Timescale Architecture (NEW 2026-05-15T13:15Z, operator-flagged)
- **Score:** 5×4×4×5 = **400**
- **Source:** arXiv:2605.12484 (May 2026) — "Learning, Fast and Slow:
  Towards LLMs That Adapt Continually"
- **Idea:** Treat LLM training as two timescales. "Slow weights" = model
  parameters (RL updates); "fast weights" = optimized context (in-context
  learning). Combining both yields 3× sample efficiency over RL-only,
  70% less KL divergence from base, less catastrophic forgetting, and
  successful continual learning where parameter-only RL stalls.
- **Direct mapping onto Carnot's architecture:** Slow weights = k=16
  verifier ensemble + base LLM (frozen at inference). Fast weights =
  the verifier-output-summary that re-prompts the LLM on the next
  verify-repair iteration. Carnot's value proposition has been
  "second-pair-of-eyes verification at inference time" — this paper
  provides peer-validated theoretical scaffolding for that exact
  architecture pattern.
- **Phase 4 rescue hypothesis:** the .181 exp1741 + .182 exp1745
  finding that alpha_t and alpha_t' are BOTH bijection-invariant at
  the ensemble-output level may be because we're measuring at the
  wrong scale. The fast-slow framing suggests measuring
  free-energy reduction on FAST WEIGHTS (context shaped by verifier),
  not slow weights (base model). If exp1745 confirms ensemble-level
  invariance, switching the measurement target to fast-weight context
  is the cleaner rescue.
- **FR-11 rethink:** Paper's central empirical finding is that
  parameter-only RL is strictly worse than fast-slow split on sample
  efficiency, drift, AND continual learning. FR-11 (verifier-as-reward
  RL) has stalled across .96-.150+ retros — possibly because it routes
  the verifier signal into slow weights (RL gradients) when the right
  destination is fast weights (context optimization).
- **Continual self-learning angle:** Carnot's CSL experiments (.177
  exp1779-1780, .180 exp1791) had mixed results. The paper provides a
  concrete mechanism: CSL works when the fast-slow split is in place —
  new tasks land in fast weights; slow weights only update slowly.
  Worth re-auditing CSL artifacts through this lens.
- **Where to land:** paper-v6 §3 architecture validation cite + concrete
  .183+ experiment (Carnot Fast-Slow Variant; see ops/known-issues.md
  RESEARCH-STUDYING CANDIDATES).
- **Cross-references:** complements but is structurally different from
  arXiv:2512.15605 (AR-LM↔EBM bijection theory) and arXiv:2605.14558
  (Token-Level Energy ActFocus). The bijection paper says "AR-LMs ARE
  EBMs"; this paper says "you should layer fast-weight context on top
  of slow-weight LMs/EBMs and train them at different timescales."

### Rank 0b: Token-Level Energy for Agentic RL — ActFocus Reweighting (NEW 2026-05-15T04:45Z)
- **Score:** 4×4×3×4 = **192**
- **Source:** arXiv:2605.14558v1 (He, Zhu, Zhou, Gu, Liu, Huang, Zou, Wipf, Yu, Wu; May 14 2026)
- **Idea:** Token-level energy-based credit assignment in agentic RL.
  Training signals concentrate on action tokens despite their scarcity.
  ActFocus reweighting reports 65.2pp gain over PPO with no compute
  overhead.
- **Hypothesis to investigate:** Does ActFocus reweighting transfer to
  Carnot's FR-11 (verifier-as-reward RL) flow? If yes, near-term
  high-leverage operational lift on the same .96-.150+ retros that
  have run on FR-11 without breakthrough.
- **Caveat:** 65.2pp gain is a substantial claim — replicate
  adversarial-verify-aware; treat as IMPLAUSIBLE_PERFECT-adjacent if
  Carnot replication shows gains > 30pp without methodology disclosure.
- **Where to land:** queue for .177+ FR-11 follow-up experiment.

### Rank 0c: IIT ↔ FEP Maximum-Caliber Bridge — Alternative alpha_t Derivation (NEW 2026-05-15T04:45Z)
- **Score:** 4×4×3×4 = **192**
- **Source:** arXiv:2605.12536v1 (Alexander Kearney; May 3 2026)
- **Idea:** Maximum-caliber variational principle bridges FEP and IIT;
  shows information emerges from prediction error under predictive
  coding.
- **Hypothesis to investigate:** alpha_t may have a maximum-caliber
  derivation that is NOT bijection-invariant under arXiv:2512.15605.
  If true, the .176 exp1715 audit's invariance finding (if confirmed)
  would NOT invalidate Phase 4 — it would just be using the wrong
  variational principle.
- **Where to land:** paper-v6 §3 Phase 4 theoretical framing,
  cited alongside arXiv:2512.15605.
- **Cross-reference:** depends on .176 exp1715 outcome — if
  artifact_detected=true, this is the replacement candidate framing.

### URGENT Rank 0a: AR-LMs are Secretly EBMs — Theoretical Scaffolding for Phase 3 (NEW 2026-05-15)
- **Score:** 5×5×4×5 = **500**
- **Source:** arXiv:2512.15605v3 (Blondel, Sander, Vivier-Ardisson, Liu, Roulet)
- **Idea:** Explicit bijection between autoregressive LMs and EBMs, plus
  distillation error bounds. The bijection is the formal scaffolding
  Carnot's Phase 3 ("foundation model based on hardware-acceleratable
  EBM/EBT") was missing — every architectural decision in Phase 3 can
  now be cross-checked against an existence proof that the AR-LM↔EBM
  map is well-defined.
- **Hypothesis to investigate:** The exp1693 (.171) Phase 4
  alpha_t = 0.15054 invariance across n=8/16/32/64 may be a corollary
  of this bijection — alpha_t is bijection-invariant, hence n-invariant.
  exp1699 (.172) random-verifier-injection audit will partially test this.
- **Where to land:** paper-v6 §3 (peer methodology) + §6 (theoretical
  framing); Phase 3 milestone task to re-derive verifier-as-free-energy
  through the bijection.
- **Why #0a (alongside the live-vs-simulated finding):** highest sweep
  score in Carnot's literature record (500 vs prior top 400); the
  bijection result is load-bearing for the Phase 3 endgame.

### URGENT Rank 0: Live vs Simulated Inference Validation
- **Score:** 5×5×5×5 = **625** (MAXIMUM)
- **Source:** Internal finding — ALL positive results were simulated inference
- **Crisis:** Exp 184 is the FIRST live GPU experiment and shows -2% standard,
  -12% adversarial on 3B model. But ALL previous positive results (Exp 91,
  120, 121, 161, 162) used SIMULATED inference. We cannot distinguish whether
  the negative result is model-size (precision ceiling) or inference-mode
  (simulation was unrealistically favorable).
- **MUST DO IMMEDIATELY:** Run 0.8B Qwen3.5 with LIVE GPU inference on the
  SAME GSM8K questions. If 0.8B live shows +10-14%, precision ceiling is real
  and we fix it with Z3/confidence. If 0.8B live shows ~0%, our ENTIRE results
  narrative is based on simulation artifacts and we have a fundamental problem.
- **Status:** INVESTIGATED — result is CONFIRMED ARTIFACT
- **Finding:** Live 0.8B inference on GSM8K produces identical wrong answers
  as the checkpoint (Q0=182, Q1=3, Q2=120000). The model scores ~25% on
  GSM8K — the simulated inference assumed ~65-70% (instruction-tuned level).
  ALL positive improvement numbers were measured against fake baselines.
- **Root cause:** Simulated inference was calibrated to published benchmarks
  for instruction-tuned models, but we loaded the BASE model (Qwen3.5-0.8B,
  not an instruct variant). The base model's actual GSM8K score is ~25%.
- **Impact:** The core +10-28% improvement claim is based on simulation
  artifacts. Real live inference shows 0% improvement at both 0.8B and 3B.
- **Path forward:** Either (a) use instruction-tuned models, (b) improve
  prompt engineering for base models, or (c) acknowledge constraint
  verification helps simulated/ideal scenarios but not raw base model outputs.
- **Exp 316 update (2026-04-14):** Full-scale benchmark ran in simulated mode
  (no live GPU). Schema and CI tests pass (28/28). Simulated results show no
  mode-to-mode improvement as expected — simulation is not live inference.
  Live GPU run still required to resolve the open question.
- **Why #0:** This is the most important finding of the entire project.

### Rank 1: Confidence-Calibrated Constraint Verification
- **Score:** 5×4×5×5 = 500
- **Source:** Internal finding (Exp 184: 3B model -2% regression)
- **Idea:** Weight constraint violations by confidence level. High-confidence
  violations (exact arithmetic mismatch) get repaired; low-confidence
  (approximate values, intermediate steps) get logged but not repaired.
  This directly addresses the precision ceiling where FP > TP on larger models.
- **Status:** Already in roadmap as Exp 202. Highest priority.
- **Why #1:** Without this, Carnot's value proposition shrinks as models improve.

### Rank 2: Semantic Constraint Verification via Chain-of-Thought Decomposition
- **Score:** 5×5×3×5 = 375
- **Source:** Exp 184 error analysis — larger models make semantic errors, not arithmetic
- **Idea:** Decompose chain-of-thought into logical steps, verify each step's
  LOGIC (not just arithmetic). "If A then B, A is true, therefore B" can be
  checked structurally. Apply the global consistency checker (Exp 172, 100%
  detection) to single-response multi-step reasoning.
- **Status:** Noted in research-program.md, not yet in roadmap
- **Why #2:** Addresses the 67% of errors that are currently uncatchable

### Rank 3: Speculative Decoding with Constraint Pre-Filtering
- **Score:** 4×5×3×4 = 240
- **Source:** Speculative decoding literature + our guided decoding (0.006ms)
- **Idea:** Use a small draft model to generate candidate tokens, then
  verify each candidate's constraint energy BEFORE the large model commits.
  Like speculative decoding but with constraint energy as the accept/reject
  criterion instead of probability matching.
- **Status:** Not in roadmap. Needs research.
- **Why #3:** Combines two proven techniques (spec decoding + constraint energy)

### Rank 4: Contrastive Constraint Learning from Model Errors
- **Score:** 4×4×4×4 = 256
- **Source:** Exp 184 data — we now have (correct, incorrect) pairs from a 3B model
- **Idea:** Train constraint extractors on the SPECIFIC error patterns of each
  model size. Instead of one-size-fits-all ArithmeticExtractor, learn what
  the 3B model gets wrong vs right and build model-specific constraints.
  The self-learning tracker (Exp 132) already accumulates this data.
- **Status:** Partially addressed by Exp 201 (precision curve)
- **Why #4:** Makes the constraint system model-adaptive

### Rank 5: FPGA Ising Sampler with Real-Time Coupling Updates
- **Score:** 3×5×3×3 = 135
- **Source:** Kria KV260 arriving in 4 days + research-hardware-wishlist.md
- **Idea:** Implement a 4K p-bit Ising sampler in Verilog with AXI-Lite
  interface for real-time coupling updates. The coupling matrix is
  reprogrammed for each constraint verification, not fixed at synthesis.
  This enables dynamic constraint checking at hardware speed.
- **Status:** Hardware ordered. Needs Verilog implementation.
- **Why #5:** Validates the TSU hardware path

### Rank 6: Energy-Aware Beam Search
- **Score:** 4×4×3×3 = 144
- **Source:** Guided decoding (Exp 110) + beam search literature
- **Idea:** Modify beam search to include constraint energy in the beam score.
  Standard beam search: score = log_prob. Energy beam search:
  score = log_prob - alpha * constraint_energy. This naturally steers
  generation toward constraint-satisfying sequences without post-hoc repair.
- **Status:** Not in roadmap
- **Why #6:** Principled integration of energy into generation

### Rank 7: Hierarchical Constraint Composition for Complex Reasoning
- **Score:** 3×4×3×4 = 144
- **Source:** Exp 63 (hierarchical Ising) + Exp 172 (global consistency)
- **Idea:** Compose constraints hierarchically: word-level (arithmetic),
  sentence-level (logic), paragraph-level (consistency), document-level
  (factual). Each level feeds violations to the next. This mirrors how
  human reasoning catches errors at multiple scales.
- **Status:** Partially explored (Exp 63, 172, 176)
- **Why #7:** Framework for scaling verification to complex documents

### Rank 8: Differentiable Constraint Compilation to Hardware
- **Score:** 3×5×2×3 = 90
- **Source:** Exp 66 (differentiable constraints) + FPGA path
- **Idea:** Compile differentiable KAN constraints directly to FPGA lookup
  tables. The spline knots become LUT entries. Training updates the LUT
  contents without FPGA resynthesis. This is the bridge between Tier 4
  adaptive structure and hardware acceleration.
- **Status:** Long-term, needs FPGA first
- **Why #8:** The eventual production architecture

## New Findings from Study Run (2026-04-11)

### NSVIF: Neuro-Symbolic Verification via First-Order Logic (HIGH RELEVANCE)
- **Source:** [arxiv 2601.17789](https://arxiv.org/html/2601.17789v1)
- **What:** Formalizes instruction verification as a CSP — extracts constraints
  from instructions, converts to first-order logic, solves with Z3 SMT solver.
- **Relevance to precision ceiling:** This is EXACTLY what we need for larger
  models. Instead of pattern-matching arithmetic (ArithmeticExtractor), formalize
  the constraints as FOL and use an SMT solver. FOL constraints have NO false
  positives — they're either satisfied or not. This could eliminate the FP
  problem on 3B+ models entirely.
- **Score:** 5×5×4×5 = **500** — ties with Rank 1
- **Action:** Promote to roadmap. Replace ArithmeticExtractor's regex with
  Z3 SMT solving for arithmetic constraints. Keep regex as fast path, Z3
  as verification backend.

### ConstraintLLM: Neuro-Symbolic for Industrial Scheduling
- **Source:** [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.809.pdf)
- **What:** Neuro-symbolic framework combining LLMs with constraint solvers
  for industrial scheduling. LLM generates constraint specifications, solver
  verifies feasibility.
- **Relevance:** Directly applicable to our scheduling domain (Exp 44, LagONN).
  Could improve scheduling constraint extraction.
- **Score:** 4×4×4×3 = 192

### FPGA P-Bit Cluster: 6400 Spins, 64 Billion Flips/Second
- **Source:** [arxiv 2512.24558](https://arxiv.org/html/2512.24558) + 
  [Nature Electronics](https://www.nature.com/articles/s41928-024-01182-4)
- **What:** Multi-FPGA cluster implementing sparse Boltzmann machines with
  p-bits. Achieved 6400 spins (80×80 Ising) on FPGA, 50-64 billion
  probabilistic flips/second. CD training with up to n=10M sweeps.
- **Relevance:** Our KV260 (arriving in 4 days) has 256K LUTs — enough for
  ~4K p-bits. This paper provides the implementation reference: sparse
  connectivity, local parallel updates, low-precision arithmetic.
  Key detail: they use CD-n with n=10M sweeps per update, far more than
  our CD-1 or CD-5. Worth testing higher-n CD on our learned Ising models.
- **Score:** 4×4×5×3 = 240 — promotes above energy-aware beam search
- **Action:** Use as implementation reference for KV260 Ising sampler.
  Add high-n CD experiment to roadmap.

### Speculative Speculative Decoding (ICLR 2026)
- **Source:** [ICLR 2026](https://openreview.net/pdf?id=aL1Wnml9Ef)
- **What:** Meta-speculation — speculate the NEXT round during current
  verification. Amortizes verification cost across rounds.
- **Relevance:** If we combine with constraint energy, the draft model
  generates candidates, constraint energy pre-filters, and the target
  model verifies. Three-level pipeline. But complex to implement.
- **Score:** 3×5×2×3 = 90

### KAN Computing-in-Memory (Nature Communications 2026)
- **Source:** [Nature Comms](https://www.nature.com/articles/s41467-026-69592-w)
- **What:** Hardware implementation of KAN using tunable Gaussian-like
  memory cells. Spline activations implemented as analog memory lookups.
- **Relevance:** Validates our Tier 4 vision (KAN → hardware). Not directly
  actionable until we have the right hardware, but confirms the path.
- **Score:** 3×5×2×2 = 60

### Agentic Confidence Calibration (2026)
- **Source:** [arxiv 2601.15778](https://arxiv.org/html/2601.15778v1)
- **What:** Holistic Trajectory Calibration — extracts process-level features
  across an agent's entire trajectory to calibrate confidence.
- **Relevance:** Directly applicable to our multi-turn agentic verification.
  Instead of per-step constraint checking, calibrate confidence across the
  whole reasoning trajectory. Could improve the global consistency checker.
- **Score:** 4×4×3×4 = 192

## Updated Rankings After Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | NSVIF: FOL + Z3 SMT constraint verification | **500** | NEW — promote to roadmap |
| 1 | Confidence-calibrated constraints | 500 | In roadmap (Exp 202) |
| 3 | Semantic constraint via CoT decomposition | 375 | Noted |
| 4 | Contrastive constraint learning | 256 | Partially in Exp 201 |
| 5 | FPGA p-bit cluster (implementation ref) | **240** | NEW — use for KV260 |
| 6 | Speculative decoding with constraints | 240 | Needs research |
| 7 | ConstraintLLM industrial scheduling | **192** | NEW |
| 7 | Agentic confidence calibration | **192** | NEW |
| 9 | Energy-aware beam search | 144 | Noted |
| 9 | Hierarchical constraint composition | 144 | Partially explored |
| 11 | FPGA Ising real-time updates | 135 | KV260 arriving |
| 12 | Speculative speculative decoding | **90** | NEW — complex |
| 12 | Differentiable constraint compilation | 90 | Long-term |
| 14 | KAN computing-in-memory | **60** | NEW — validates path |

### Kona 1.0 Architecture Details (STRATEGIC INTELLIGENCE)
- **Source:** [logicalintelligence.com](https://logicalintelligence.com/kona-ebms-energy-based-models),
  [BusinessWire Jan 2026](https://www.businesswire.com/news/home/20260120751310)
- **What:** Kona 1.0 is now in pilot programs. Key architectural details:
  - **Non-autoregressive at trace level** — generates complete reasoning traces
    simultaneously (not token-by-token)
  - **Continuous latent space** — outputs dense vector tokens, not discrete
  - **Self-correcting** — learns by recognizing and correcting own mistakes
  - **96.2% Sudoku** in 313ms (vs LLMs at 2%)
  - Yann LeCun added to leadership (validates EBM direction)
  - Pilot sectors: energy, manufacturing, semiconductors
- **Relevance:** This is our North Star competitor. Key differences from Carnot:
  - Kona generates reasoning; Carnot verifies LLM reasoning
  - Kona is non-autoregressive; Carnot works with autoregressive LLMs
  - Kona operates in continuous latent space; we're bridging to it (Exp 64-66)
  - The self-correcting aspect is what our verify-repair loop does externally
- **Implications for our precision ceiling:** Kona's continuous latent space
  may not have the FP problem because it doesn't use discrete constraint
  matching. Our Z3 SMT approach (NSVIF) is the bridge.
- **Score:** Strategic intelligence, not directly actionable. Monitor.

### Extropic Z1 Timeline Update
- **Source:** [extropic.ai/hardware](https://extropic.ai/hardware)
- **What:** Z1 chip (hundreds of thousands of p-bits per chip, millions per
  card) scheduled for early access 2026. XTR-0 testing platform was Q3 2025.
  Mass-manufacturable using standard CMOS.
- **Relevance:** Our KV260 FPGA (arriving in 4 days) is the bridge. If Z1
  early access opens, we have the SamplerBackend abstraction (Exp 71) ready
  to plug in. Our FPGA work validates the architecture before Z1 ships.
- **Score:** 3×3×2×3 = 54 — monitor, hardware path validated

### "Hallucination is Inevitable" (HuggingFace trending)
- **Source:** [huggingface.co/papers/2401.11817](https://huggingface.co/papers/2401.11817)
- **What:** Formal proof that LLMs inherently hallucinate — cannot learn all
  computable functions. Hallucination is a mathematical inevitability.
- **Relevance:** VALIDATES our entire approach. If hallucination can't be
  eliminated from INSIDE the model, external verification (Carnot) is the
  only path. This is the theoretical justification for our product.
- **Score:** 5×1×1×5 = 25 — not actionable but validates our thesis

## Libraries of Reference (Consulted During Study Runs)

Study runs check ALL of these sources:
1. **arxiv.org** — primary research papers
2. **OpenReview.net** — NeurIPS/ICML/ICLR submissions
3. **extropic.ai/writing** — TSU hardware updates
4. **Semantic Scholar** — citation tracking for key papers
5. **HuggingFace papers** (huggingface.co/papers) — daily ML paper feed
6. **GitHub trending** — new repos (ising-model, energy-based-model topics)
7. **logicalintelligence.com** — Kona architecture updates
8. **FPGA conferences** (FCCM, FPL, DAC) — Ising machine implementations
9. **AMD developer forums** — NPU/XDNA updates
10. **Nature Electronics/Communications** — hardware implementations
11. **ACL Anthology** — NLP constraint/verification papers

## Needs Investigation (Unranked)

- LagONN + guided decoding combination (oscillatory escape + energy steering)
- Multi-agent constraint verification (one agent generates, another verifies)
- Retrieval-augmented constraints (look up facts before verifying)
- Constraint transfer learning (train on one domain, apply to another)
- Grammar-constrained decoding as constraint substitute (ACL 2025 finding)
- Block verification for speculative decoding (5-8% speedup, OpenReview)
- Physics-informed KAN with augmented Lagrangian (Nature 2025)

## Revalidation Sweep — Approaches That Deserve Live Re-Testing

**Context:** The simulation artifact discovery (Exp 203-209) led us to remove
unverified numbers from reporting. But some earlier approaches may genuinely
work — they were tested with bad experimental methodology, not bad ideas.
This sweep re-runs the most promising old experiments with live GPU inference
to either confirm or definitively rule them out.

**STATUS: COMPLETED 2026-04-14** — Exp 271-279 executed and classified.
Full results: `results/revalidation_sweep_271_279_summary.json`.

### High Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 172, 176 | **Exp 271** | ✅ **CONFIRMED** | 100% detection, 0% FP, 1.91 ms/call — logic-based, inference-mode-independent |
| 134 (Tier 1) | **Exp 272** | ⚠️ INCONCLUSIVE | 86% FP reduction confirmed (7→1); task-success rate flat 32.7% — FP win is real, primary objective not met |
| 126-127 | **Exp 273** | ✅ **CONFIRMED** | 100% rollback success + 100% violation detection (canned outputs; deterministic logic) |
| 158 | **Exp 274** | ✅ **CONFIRMED** | 45% coverage ≥ 40% target; 100% accuracy ≥ 75% target on IT model responses |
| 175 | **Exp 275** | ✅ **CONFIRMED** | KAN AUROC 0.991 on live traces; AMR pruned 17 params, 0.0 AUROC gain |

### Medium Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 91-92 | **Exp 276** | ✅ **CONFIRMED** | Z3+LLM: 80% detection, 0% FP; semantic: 0% detection, 20% FP for arithmetic |
| 142 | **Exp 277** | ⚠️ INCONCLUSIVE | 3068 tests pass; results JSON absent — needs re-run for quantitative classification |
| 149 | — | Not revalidated | TruthfulQA factual coverage deferred to future milestone |
| 136 | **Exp 278** | ✅ **CONFIRMED** | 100% warm hit rate, 0% FP unseen, session boundary preserved, avg score 95.67 |

### Low Priority — Results

| Original Exp | Revalidation Exp | Classification | Outcome |
|-------------|-----------------|----------------|---------|
| 161, 163 | — | Superseded | Covered by Exp 219/235 (200-question GSM8K live runs) |
| 178 | **Exp 279** | ✅ **CONFIRMED** | Stale detection 100%, fresh-wrong 0%, FP 20%, lift +40pp — semantic grounding targets quantity-mismatch specifically |

### Definitively Ruled Out (evidence-based, not provenance-based)

These are NOT candidates for revalidation — they were disproven by experimental evidence:
- **Activation-based EBMs** (Exp 1-38): 14 principles prove they detect confidence, not correctness. No provenance issue — the approach is fundamentally flawed.
- **LNN adaptive couplings** (Exp 116): -90% vs static Ising. Worse in every metric.
- **Precision-based constraint reweighting** (Exp 134 original): 0% improvement on the specific reweighting approach (though the self-learning architecture was validated by Exp 223).

### D-Wave Quantum Annealing (ACTIONABLE — Add Now)
- **D-Wave Advantage**: 5,000+ qubits (Pegasus topology, 15-way connectivity).
  Advantage2: 7,000+ target (Zephyr, 20-way). Solves Ising/QUBO natively.
- **Ocean SDK**: Apache 2.0, `pip install dwave-ocean-sdk`. `dimod` for BQM,
  `neal` for local simulated annealing, `dwave-system` for real QPU.
- **Carnot fit**: Perfect — D-Wave literally solves Ising problems. Our
  SamplerBackend abstraction + IsingEBM coupling matrix maps 1:1 to dimod BQM.
- **Local simulation**: `neal.SimulatedAnnealingSampler()` runs locally, same
  API as hardware. Prove the approach works without QPU access.
- **Free tier**: 1 min QPU/month via D-Wave Leap (enough for ~1000 problems).
- **Score**: 5x5x5x4 = **500** — high relevance, high feasibility, proven technology
- **Action**: Add `dwave-ocean-sdk` as optional dep, create `DWaveSampler`
  implementing `SamplerBackend`, benchmark local sim vs CPU Ising sampler.

### Intel Loihi 2 Neuromorphic (Track — Cloud Access Available)
- **Intel Loihi 2**: 1M spiking neurons, on-chip learning. Free academic access
  via Intel Neuromorphic Research Community (INRC). Natively implements
  energy-minimization via spiking dynamics. Demonstrated Ising solving via
  neural annealing (Intel labs 2023-2024). Relevant to Boltzmann tier sampling.
- **Action:** Apply for INRC access. Could implement a `LoihiSampler` backend.

### Oscillator-Based Ising Machines (Track — CMOS Scalable)
- **Purdue/Cornell coupled CMOS oscillator networks**: Phase-encoded spins via
  injection-locked LC oscillators. 240-spin chip demonstrated (2024). CMOS-native
  means it could scale to millions of spins on standard foundries.
- **Purdue p-bit MRAM** (Camsari group): MRAM-based stochastic magnetic tunnel
  junctions. 8-p-bit ASIC demonstrated, 50K designs published. Same p-bit
  abstraction as Extropic but magnetic rather than thermodynamic.
- **Action:** Monitor for ASIC availability (2025-2027). Our SamplerBackend
  abstraction is ready for both approaches.

### NTT Coherent Ising Machine (Track — Largest Demonstrated)
- **NTT/Stanford CIM**: Optical parametric oscillator pulses, 100,000+ spins
  demonstrated. Time-multiplexed (vs SPIM's spatial). NTT offers cloud access
  for research collaborations. Largest Ising machine demonstrated to date.
- **Action:** Explore NTT research collaboration for cloud CIM access.

### Analog In-Memory Computing (Monitor)
- **Mythic M1076**: Analog matrix-multiply in flash memory, 25 TOPS. Dev kit
  ~$500. Energy function evaluation (W*s products) maps to analog MAC.
  Relevant for KAN/Boltzmann forward passes, not Ising sampling.

### EBM Safety Classifier (Distilled from gpt-oss-safeguard) — HIGH PRIORITY
- **Concept:** Train Carnot's KAN tier as a lightweight safety classifier using
  gpt-oss-safeguard (Apache 2.0, 20B/120B) as teacher. The KAN model (2.3K params,
  0.994 AUROC) could classify inputs as safe/unsafe at a fraction of the compute.
- **How it works:**
  1. Run gpt-oss-safeguard-20b on a corpus of safe + unsafe prompts
  2. Collect (input, safety_label, reasoning) pairs
  3. Train KAN energy model: low energy = safe, high energy = unsafe
  4. Deploy as a pre-filter in VerifyRepairPipeline for input sanitization
- **Advantages over gpt-oss-safeguard alone:**
  - 2.3K params vs 5.1B active params (2000x smaller)
  - Runs on CPU in <1ms (vs GPU inference for the teacher)
  - Integrates natively with Carnot's energy pipeline
  - Hardware-acceleratable (Ising/FPGA/D-Wave for the safety energy landscape)
- **Score:** 5x5x4x5 = **500** — high impact, feasible, proven teacher model
- **Action:** Add to next milestone. Requires downloading gpt-oss-safeguard-20b
  weights from HuggingFace and running distillation pipeline.

### Mythos System Card Insights (Applied — From Anthropic's 244-page safety evaluation)
- **Source:** Anthropic Claude Mythos Preview System Card (April 7, 2026)
- **Key findings applicable to Carnot:**

1. **Verification gap validated:** Even Mythos (93.9% SWE-bench) produces factual
   errors that are only caught when users explicitly request re-derivation. The model
   "could reach the right answer once asked but did not verify claims before writing
   them." This validates Carnot's external verification thesis.

2. **Reward hacking in self-learning:** Mythos discovered novel reward hacks (moving
   computation outside timing calls, using test data to train). Our self-learning
   loop (Exp 223/241) needs guards against energy function gaming.

3. **Behavioral monitoring for autonomous systems:** Anthropic uses automated offline
   monitoring, behavioral audits, and interpretability analysis for alignment. Our
   conductor runs autonomously for hours — we should apply similar monitoring.

4. **Constitutional alignment for autoresearch:** Define explicit rules for what the
   conductor can/cannot do without human approval. Prevent autonomous systems from
   taking irreversible actions.

- **Proposed experiments:**
  - Reward hacking detection in self-learning energy function
  - Conductor behavioral audit log with anomaly detection
  - Conductor constitution defining allowed/forbidden autonomous actions
  - Verification-before-publication gate (extend Exp 209 provenance audit)

### Vulkan Compute Backend for Universal GPU Support (Phase 2 — Plan Now)
- **Why:** CUDA locks us to NVIDIA. ROCm is unstable (broke on our iGPU).
  Vulkan works on every modern GPU: NVIDIA, AMD, Intel, mobile.
- **What to build:** Vulkan compute shaders for energy function evaluation
  (E = -0.5 x^T J x), Ising sampling (parallel spin flips), and KAN
  forward pass (B-spline evaluation).
- **Tools:** `vulkano` (Rust, our production language), `kompute` (Python bridge),
  or `wgpu` (Rust, WebGPU API over Vulkan/Metal/DX12).
- **Architecture:** Vulkan for energy computation, CUDA/ROCm for LLM inference
  (PyTorch/JAX still need vendor backends for model loading).
- **When:** Phase 2 — after core verification pipeline is stable. The Rust
  crates (`carnot-ising`, `carnot-kan`) are the natural place to add Vulkan.
- **Score:** 4x4x3x4 = 192 — important for portability, medium effort
- **Action:** Add Vulkan compute experiment to Phase 2 milestone. Start with
  Ising energy evaluation (simplest kernel), then KAN forward pass.

### NVIDIA "Ising" — NAMING COLLISION, NOT an Ising optimization solver (Noted, Low Relevance)
- **Source:** [nvidia.com/en-us/solutions/quantum-computing/ising/](https://www.nvidia.com/en-us/solutions/quantum-computing/ising/) (2026 release, exact date TBD)
- **What NVIDIA's "Ising" actually is:** A family of Apache-2.0-ish AI models for
  quantum computing workflows. NOT a classical Ising-model optimization solver.
  Two members:
  - **Ising Calibration** — 35B-parameter Vision-Language Model that automates
    quantum processor (QPU) tuning by inferring calibration actions from QPU
    experimental data.
  - **Ising Decoding** — Two 3D CNN models (0.9M / 1.8M parameters) for quantum
    error correction. Claimed 2.5x faster, 3x more accurate than prior methods.
- **Why the name collision matters for us:** Carnot's "Ising" means the discrete
  spin-glass optimizer (carnot-ising crate, Ising tier in the four-tier model
  hierarchy). NVIDIA's "Ising" means "AI models for operating qubit hardware".
  Future contributors reading "NVIDIA Ising" may incorrectly assume it's directly
  applicable to Carnot's Ising verifier. It is not.
- **Direct applicability to Carnot:** Very low.
  - We don't operate qubit hardware; we use Ising *as a math formulation* for
    constraint satisfaction on classical hardware.
  - The D-Wave sampler we integrated (Exp 320) IS quantum-hardware-adjacent,
    but uses D-Wave's quantum annealing, not NVIDIA's QPU calibration flow.
- **Indirect applicability:** The 3D CNN for quantum error correction is an EBM-
  like discriminator architecture — it learns to assign low "energy" to valid
  error syndromes vs. invalid ones. Pattern-level similarity to our CIKANEnergy
  and EORM models. Worth a skim of the arXiv write-up when available, but not
  worth an experiment.
- **Score:** 2x3x3x1 = 18 (mostly "name disambiguation" value, not research value).
- **Action:** No experiment planned. This entry exists so future sessions don't
  confuse NVIDIA Ising with our Ising work.

### CUDA Megakernel Fusion Techniques (Study — Transferable Optimizations)
- **Source:** [luce-megakernel](https://github.com/Luce-Org/luce-megakernel) (MIT)
- **What it is:** A single-dispatch CUDA megakernel that runs the entire Qwen 3.5-0.8B
  forward pass (24 layers) without returning to CPU. Eliminates ~100 kernel launches
  per token, achieving 413 tok/s on RTX 3090 at 1.87 tok/J.
- **Why study:** The kernel fusion pattern — running an entire compute graph in one
  dispatch with cooperative grid sync between stages — is transferable beyond LLM
  inference. Relevant techniques for Carnot:
  - **Fused Ising sampling:** Multiple spin-flip rounds + energy evaluation in one
    kernel launch, avoiding host round-trips between sampling iterations.
  - **Fused verify-repair pipeline:** Chain constraint extraction, energy evaluation,
    and repair candidate scoring in a single GPU dispatch.
  - **Register-resident state:** Keep Ising spin vectors in registers across iterations
    instead of writing to global memory — directly applicable to our RTX 3090 setup.
  - **DVFS power tuning:** Their 1.87 tok/J efficiency comes partly from GPU clock
    management — useful for extended conductor runs to avoid thermal throttle (GPU 0
    already hitting 82C per RETRO-025).
- **Limitations:** CUDA-only (no Vulkan path), batch-1 only, single-model specific.
  The code itself is not reusable, but the patterns are.
- **Score:** 3x4x3x3 = 108 — novel fusion patterns, medium relevance, medium effort
  to study and adapt, not urgent.
- **Action:** Study cooperative grid sync and register-resident state patterns. Consider
  applying to Ising sampling kernel if we write custom CUDA (before Vulkan port).

### Photonic Computing (Monitor — Not Actionable Yet)
- **Q.ANT NPU 2.0** — commercial photonic matmul accelerator (30x energy efficiency).
  Not directly useful for sampling. Commercial-only, no cloud access.
- **Photonic Ising Machines (SPIMs)** — encode spins as phase patterns, compute
  Hamiltonians optically in a single pass. Currently ~32 spins experimentally.
  - [arxiv 2508.17440](https://arxiv.org/abs/2508.17440) — k-local Ising + optical KANs on same platform (maps to Carnot tiers!)
  - [arxiv 2502.18918](https://arxiv.org/abs/2502.18918) — parallel SPIM via spatial multiplexing
  - [arxiv 2410.10689](https://arxiv.org/abs/2410.10689) — fully programmable SPIM
- **When to act:** When SPIMs scale past ~100 spins or Q.ANT opens cloud access.
  Our SamplerBackend abstraction is ready for a photonic adapter.

## Archived (Investigated, Not Promising)

- LNN adaptive couplings within chains: -90% vs static Ising (Exp 116)
- Precision-based constraint reweighting: 0% improvement (Exp 134)
- Activation-based EBMs: detect confidence not correctness (14 principles)

<!-- EXP210_STUDYING_START -->
## Study Run 2026-04-12 - Constraint Extraction for Instruction-Tuned Models

### Ranking update
| Rank | Idea | Score | Why it matters |
|------|------|-------|----------------|
| 1 | Prompt-to-constraint intermediate representation with solver fallback | 625 | NSVIF, DeCRIM, and ConstraintLLM all point to the same fix: extract atomic constraints from the instruction before verifying the answer. |
| 2 | Benchmark-first extraction workbench | 500 | FollowBench, CFBench, RealInstruct, and VIFBench provide the missing datasets needed to measure extraction recall and false positives directly. |
| 3 | Dual-path verification: prompt-answer first, CoT second | 500 | CoT verification is promising, but monitorability papers say Carnot should never depend on raw CoT alone. |
| 4 | Typed step-graph verification for arithmetic and logic traces | 375 | VeriCoT, PCRLLM, Deductive Verification, and Typed CoT all support moving from free-form traces to explicit premises and rules. |
| 5 | Constraint-programming route for scheduling and resource tasks | 240 | ConstraintLLM plus IndusCP is the best external path for Carnot's scheduling extractor gap. |
| 6 | CoT monitorability score and fallback policy | 240 | Recent monitorability work implies Carnot needs a gate deciding when CoT evidence is safe to trust. |

### Key takeaways
- The strongest direct fit is prompt-side instruction verification: convert instructions into atomic constraints first, then verify the answer against them.
- Step-level CoT verification is now technically credible, but only when reasoning traces are reformatted into explicit premises, rules, and typed steps.
- Benchmark coverage for fine-grained instruction constraints is finally good enough to evaluate extraction quality directly instead of using answer accuracy as a proxy.
- Recent monitorability papers make raw chain-of-thought an unsafe sole source of truth; Carnot needs a fallback path that does not trust CoT by default.

### Proposed experiments for 2026-04-15
- **EXP-211 - Instruction-to-Constraint IR Benchmark**
  Goal: Build a gold benchmark of atomic prompt constraints from FollowBench, RealInstruct, CFBench, and VIFBench, then measure extraction recall and false positives on instruction-tuned models.
  Hypothesis: Prompt-side decomposition will reduce false positives more than answer-only regex extraction because the verifier will know exactly which constraints matter before inspecting the response.
  Success criteria: Atomic constraint recall >= 0.85 on the curated benchmark, satisfied-constraint false-positive rate <= 0.05, and measurable improvement over the current regex plus Z3 promptless path.
- **EXP-212 - Dual-Path CoT Verifier with Typed Step Graphs**
  Goal: Implement a step-level verifier for arithmetic and logic traces using premise-rule-conclusion records inspired by VeriCoT, PCRLLM, Deductive Verification, and Typed CoT.
  Hypothesis: A typed step graph will catch errors that answer-only checking misses, but only when combined with prompt-derived constraints and a fallback to answer-level verification.
  Success criteria: On a live instruction-tuned cohort, catch >= 25% of wrong answers missed by prompt-only verification while adding < 2% extra false positives on correct answers.
- **EXP-213 - CoT Monitorability Audit and Fallback Policy**
  Goal: Measure whether Qwen and Gemma instruction-tuned models expose enough faithful reasoning to justify CoT-based extraction, using recent faithfulness and pathology metrics.
  Hypothesis: Monitorability differs by model family and task, so Carnot should gate CoT extraction behind a measured trust score rather than assuming traces are faithful.
  Success criteria: Produce a per-model monitorability score, a pathology breakdown, and a simple policy that predicts when to trust CoT extraction versus prompt-answer-only verification.
<!-- EXP210_STUDYING_END -->

## Study Run 2026-04-12 — Post-Milestone 2026.04.14 + Early 2026.04.15

**Updated:** 2026-04-12
**Current Focus:** Semantic grounding gap (0/9 wrong answers detected on live GSM8K)

### New Findings

#### Property-Generated Solver (HIGH IMPACT — code verification)
- **Source:** [arxiv 2506.18315](https://arxiv.org/abs/2506.18315)
- **What:** Uses property-based testing to validate LLM-generated code. Properties
  are simpler to define than exhaustive test oracles. **23-37% pass@1 improvement.**
- **Relevance:** Directly applicable to Exp 217 (property code verifier) and our
  HumanEval pipeline. Could multiply the +3.3pp we got in Exp 208.
- **Score:** 5×5×5×5 = **625** — MAXIMUM. Implement immediately.
- **Action:** Integrate PBT into CodeExtractor for Exp 217/220.

#### Eidoku: Neuro-Symbolic Verification Gate
- **Source:** [arxiv 2512.20664](https://arxiv.org/pdf/2512.20664)
- **What:** Deterministic rejection gate for LLM reasoning hallucinations.
  Neuro-symbolic sanity check that gates generative output.
- **Relevance:** Exactly what our verify-repair pipeline does. Validate our
  architecture against their design patterns.
- **Score:** 5×4×4×4 = 320

#### Neuro-Symbolic Compliance (LLM + SMT for Finance)
- **Source:** [arxiv 2601.06181](https://arxiv.org/html/2601.06181v1)
- **What:** LLM interprets regulations → generates SMT constraints → solver
  enforces consistency. 86.2% SMT code gen accuracy, 100x reasoning speedup.
- **Relevance:** Same pattern as our Z3 extractor but for legal/financial domain.
  Validates LLM-as-SMT-generator approach.
- **Score:** 4×4×4×3 = 192

#### SCoRe: Multi-Turn RL Self-Correction (ICLR 2025)
- **Source:** ICLR 2025 SuperCorrect
- **What:** Multi-turn RL teaches LLMs to self-correct. +15.6% MATH, +9.1% HumanEval.
- **Relevance:** Our verify-repair loop is external self-correction. SCoRe shows
  internal self-correction can complement it. Could inform repair prompting.
- **Score:** 4×4×3×4 = 192

#### Learning to Self-Verify (CRITICAL INSIGHT)
- **Source:** [arxiv 2602.07594](https://arxiv.org/html/2602.07594v1)
- **What:** Self-verification doesn't improve with model scale. Needs explicit
  training. Generation and verification are asymmetric capabilities.
- **Relevance:** Validates Carnot's external verification approach. LLMs can't
  self-verify — they need us.
- **Score:** 5×3×1×5 = 75 — not actionable but validates thesis

#### Thought Anchors (NeurIPS 2025 Workshop)
- **Source:** [OpenReview](https://openreview.net/forum?id=VnSlfeRCaU)
- **What:** Identifies which CoT reasoning steps have outsized impact on final
  answers. Some steps are "anchors" that determine the trajectory.
- **Relevance:** Could improve our CoT monitorability audit (Exp 213) — focus
  verification on anchor steps, not all steps.
- **Score:** 4×5×3×4 = 240

#### Scientific Knowledge-Driven Decoding Constraints
- **Source:** [arxiv 2604.06603](https://arxiv.org/html/2604.06603)
- **What:** Hard constraints combined with LLM distributions during decoding
  without interfering with normal reasoning.
- **Relevance:** Directly applicable to our guided decoding (Exp 110). Better
  constraint integration method.
- **Score:** 4×4×3×3 = 144

### Updated Rankings After 2026-04-12 Study Run

| Rank | Idea | Score | Status |
|------|------|-------|--------|
| 1 | **Property-Based Testing for code verification** | **625** | NEW — integrate into Exp 217/220 |
| 1 | Prompt-to-constraint IR with solver fallback | 625 | In progress (Exp 211-212) |
| 3 | Confidence-calibrated constraints | 500 | Deferred |
| 4 | Semantic constraint via CoT decomposition | 375 | In progress (Exp 215-216) |
| 5 | Eidoku verification gate pattern | **320** | NEW — architecture validation |
| 6 | Contrastive constraint learning | 256 | Partially explored |
| 7 | Thought Anchors for CoT focus | **240** | NEW — improve Exp 213 |
| 8 | FPGA p-bit cluster | 240 | KV260 arriving soon |
| 9 | Neuro-Symbolic Compliance (SMT) | **192** | NEW — validates Z3 approach |
| 9 | SCoRe self-correction | **192** | NEW — inform repair prompting |
| 11 | ConstraintLLM scheduling | 192 | Noted |
| 12 | Energy-aware beam search | 144 | Noted |
| 12 | Scientific decoding constraints | **144** | NEW — guided decoding |
| 14 | FPGA Ising real-time updates | 135 | KV260 arriving |

#### 1024-Neuron FPGA Ising Accelerator (FPGA REFERENCE)
- **Source:** [arxiv 2505.20250](https://arxiv.org/abs/2505.20250)
- **What:** All-to-all connected probabilistic Ising machine on FPGA with
  ~10,000x speedup over GPU heuristics. 1024 neurons.
- **Relevance:** Direct implementation reference for our KV260 Ising sampler
  (Exp 228). Our target is 4K spins — this shows 1K is proven.
- **Score:** 5×4×5×4 = 400

#### VCoT-Bench: Z3 Proofs → Verus Rust Verification
- **Source:** [arxiv 2603.18334](https://arxiv.org/html/2603.18334)
- **What:** Benchmarks LLMs on transforming Z3 proofs into Verus-level
  Rust verification steps. Bridges formal proofs to systems code.
- **Relevance:** Could connect our Z3 constraint verification to our Rust
  crates — formal proofs that compile to verified Rust code.
- **Score:** 4×5×2×3 = 120

#### Solver-Aided Policy Compliance for LLM Agents
- **Source:** [arxiv 2603.20449](https://arxiv.org/html/2603.20449)
- **What:** Translates NL tool-use policies into Z3 constraints, checks
  planned tool calls before execution.
- **Relevance:** Directly applicable to our agentic verification — could
  verify conductor/agent actions before they execute.
- **Score:** 5×4×4×3 = 240

#### SemLoc: Structured Grounding of LLM Reasoning
- **Source:** [arxiv 2603.29109](https://arxiv.org/abs/2603.29109)
- **What:** Binds each inferred property to a typed program anchor for
  runtime checking. 42.8% Top-1 fault localization accuracy.
- **Relevance:** Typed grounding of reasoning steps — aligns with Exp 212
  typed reasoning IR and Exp 215 semantic grounding.
- **Score:** 4×5×3×4 = 240

#### Graph of Verification: DAG-Based Multi-Granular Verification
- **Source:** [arxiv 2506.12509](https://arxiv.org/abs/2506.12509)
- **What:** Adaptive multi-granular verification using DAG structure over
  reasoning steps. Complements our step-by-step approach.
- **Score:** 4×4×3×3 = 144

#### Continuous Self-Improvement via Learned Verifier
- **Source:** [arxiv 2505.19475](https://arxiv.org/abs/2505.19475)
- **What:** Learned verifier scores candidates for test-time self-training
  loop. Matches Carnot's autonomous self-learning vision.
- **Score:** 4×4×3×4 = 192

### Implications for Milestone 2026.04.16

The Property-Generated Solver finding is transformative for code verification.
Our HumanEval result (+3.3pp) used only basic execution testing. PBT showed
23-37% improvement on similar benchmarks — we should expect a much larger delta
if we integrate property-based testing into our CodeExtractor + repair loop.

**Proposed milestone 2026.04.16 theme: "Scale What Works"**
1. Scale code verification with PBT (our strongest live result)
2. FPGA Ising prototype (KV260 should have arrived)
3. Full 164-problem HumanEval with PBT + repair (publishable result)
4. Multi-model code verification (Qwen + Gemma + larger models)
5. Self-learning from code verification traces (Tier 1-2)
6. Bridge to production: package the code verification pipeline

### Proposed Milestone: "Security Hardening" (after revalidation)

**Theme:** Harden the autoresearch pipeline against adversarial inputs,
supply chain attacks, and untrusted code execution.

**Experiments:**
1. **gvisor sandbox validation** — verify sandboxed_exec_function works
   end-to-end on full HumanEval, measure overhead vs in-process exec
2. **gpt-oss-safeguard-20b integration** — deploy as local content scanner
   for arxiv/web ingestion, measure false positive rate on research papers
3. **Model supply chain audit** — pin all HuggingFace model hashes, verify
   no trust_remote_code=True calls exist, add pre-download hash check
4. **Semgrep/Bandit for generated code** — scan LLM-generated code before
   execution, integrate into the verify-repair loop
5. **Conductor isolation** — run conductor in Firecracker microVM with
   limited filesystem access, network filtering
6. **Prompt injection detection** — add Rebuff or similar to detect injection
   attempts in web-fetched content used by the study run

**Dependencies:** gvisor already installed (runsc), Docker running,
sandbox.py module created
**Expected outcome:** Code execution fully sandboxed, external content
scanned before ingestion, model supply chain verified

### ~~Proposed~~ Completed Milestone: "Revalidation Sweep" (2026-04-14)

**Theme:** Re-run the 10 most promising pre-provenance experiments with
live GPU inference and modern extractors. Either confirm they work (and
add to the live results portfolio) or definitively rule them out with
evidence, not just missing metadata.

**Actual experiments:** 9 (Exp 271-279)
**Outcome:** 6 CONFIRMED, 2 INCONCLUSIVE (Exp 272 FP-only win; Exp 277 missing JSON), 0 definitively ruled out.
**Credible results added:** GlobalConsistencyChecker, agent rollback, factual KB extraction,
KAN verification, Z3+LLM on GSM8K arithmetic, cross-session memory, adversarial semantic grounding.
**Remaining:** Exp 277 (combined signals) needs re-run with explicit JSON output; TruthfulQA deferred.

### Sweep 2026-05-16T12:00Z
- **Anchor**: arXiv:2603.28135
- **New IDs**: 49
- **Promotions**:
  - arXiv:2601.17223 (Score 400)
  - arXiv:2602.14189 (Score 320)
  - arXiv:2604.16753 (Score 320)

## 2026-06-07 Exp 3932 - Agentic Verification Efficiency Positioning

**Candidate:** Verification-efficiency positioning for the next convergence
milestone.

**Score: 5 x 5 x 4 x 4 = 400** - high alignment with north-star section 5, high
experiment leverage, medium implementation risk, and high convergence value.

**Position:** Carnot belongs in the cheap discriminative verifier lane: a
classifier-first energy layer screens all steps/actions, while competent
GenRM/ThinkPRM judges handle hard cases. The local Exp 3926/3928 artifacts are
blocked, so the claim is positioned as a near-term convergence target rather
than a landed parity result; Exp 3929 supplies the synthetic ARC-AGI-3 action-
efficiency bridge.

**Next experiments:** ProcessBench full-benchmark head-to-head: run Carnot energy scores versus a competent GenRM/ThinkPRM-style judge on the full held-out benchmark so the efficiency claim is tested against a credible comparator; ARC-AGI-3 real-benchmark agentic run: replace the synthetic grid step with the official interactive harness and report action efficiency without claiming a leaderboard score.

## 2026-06-08 Exp 3943 - Verifier Efficiency Landscape Positioning

**Candidate:** .365 convergence steer after the verifier-efficiency proof.

**Score: 5 x 5 x 5 x 4 = 500** - maximum north-star alignment, maximum
experiment leverage, maximum public-positioning value, and medium execution
risk because real benchmark access and full ProcessBench throughput can still
block.

**Position:** Carnot now belongs in the cheap discriminative verifier lane:
energy verification screens every candidate cheaply, while GenRM/ThinkPRM-style
judges handle close or high-value cases. The .364 result should be framed as a
cost-normalized verifier proof, not as a claim that energy scoring replaces
generative reasoning.

**Next experiments:** ProcessBench full-benchmark head-to-head: run the landed cheap-energy verifier and the competent GenRM/ThinkPRM-style judge on the full held-out benchmark with cost-normalized parity/Pareto reporting; ARC-AGI-3 real agentic run / real ARC-AGI-3 agentic run: move from synthetic action-pruning to an official interactive harness run, reporting action efficiency only under the benchmark protocol.
