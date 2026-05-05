# Research Roadmap vNEXT: Milestone 2026.04.107

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.106 Tag-First Certificate Prefix Injection + Eidoku CSP + KAN Formal Verification
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .106 Proved

| Track | Evidence | Finding |
|---|---|---|
| .105 handoff audit | `exp1364` | `thinking_mode_blocker_confirmed=true`, `terminal_certificate_required=true`, `prior_certificate_parse_rate=0.0` — confirmed the structural tag emission failure as terminal negative evidence. |
| Eidoku CSP probe | `exp1365` | `eidoku_csp_viable=true`, `csp_feasibility_rate=0.740`, `eidoku_auroc_proxy=0.614` — grammar-free verification alternative established as a viable fallback path. |
| Certificate v8 tag-first | `exp1366` | `certificate_parse_rate=1.0`, `parse_rate_delta_over_exp1353=1.0`, `prefix_injection_supported=true`, `headline_result_allowed=true` — **TERMINAL POSITIVE EVIDENCE**. Tag-first prefix injection + CRANE alternating pattern fully resolved the thinking-mode blocker. |
| DiffuTruth probe | `exp1367` | `detection_auroc_proxy=0.867`, `ising_correlation=0.699`, `kan_correlation=0.961`, `viable_as_complement=true` — non-equilibrium energy signal strongly correlates with KAN energy. |
| FOVER-aligned skill audit | `exp1368` | `fover_aligned_logicskills_audit_no_skill_gap` — no dominant skill gap in certificate failures; all three LogicSkills categories passing with tag-first generation. |
| Semantic validator v2 | `exp1369` | `validator_execution_pass_rate=1.0`, `z3_constraint_pass_rate=1.0`, `semantic_validator_claim_allowed=true` — full semantic validation chain operational. |
| VERGE MCS repair | `exp1370` | `repair_hint_precision=1.0`, `accepted_violation_delta=-2`, `repair_claim_allowed=true` — MCS repair localization working with zero false hints. |
| Margin-aware scheduler | `exp1371` | `false_acceptance_rate=0.0`, `full_verifier_call_reduction=0.25`, `triage_claim_allowed=true` — 25% verifier call reduction with zero false acceptance. |
| KAN PWA formal | `exp1372` | `formal_property_verified=true`, `milp_verification_result=verified`, `kan_formal_claim_allowed=true` — first formal certified energy bound for GS-KAN tier. |
| Ising inertia | `exp1373` | `inertia_convergence_speedup=0.795`, `best_inertia_alpha=0.0` — inertia term provides no CPU speedup; checkerboard baseline is optimal for dense float computation. FPGA claim not made. |
| Continuous self-learning v3 | `exp1374` | `path_used=primary_semantic_verified`, `fresh_verified_sample_count=4`, `self_learning_delta_overall=1.596429`, `dvi_ready=true`, `headline_result_allowed=true` — **FIRST HEADLINE SELF-LEARNING RESULT**. Primary semantic verified path used; DVI declared ready. |
| Publication hold v15 | `exp1375` | **MISSING** — SKIPped 3× due to pre-test cascade failure starting at 18:14 UTC. |
| Milestone retro | `exp1376` | **MISSING** — SKIPped 3× due to same pre-test cascade failure. |

**Key operational finding.** Two final tasks were exhausted by the pre-test cascade.
Root cause: `tests/python/phase5/test_intermediate_scale_v3.py` has an unresolvable
`ModuleNotFoundError: No module named 'carnot.phase5.intermediate_scale_v3'` because
the module was never implemented (exp1238 was incomplete). The pre-test suite fails
at collection time for this file, and the conductor's self-heal loop cannot fix it.
The `.107` milestone must address this as its first task.

## Research Signals Added Before Planning

The post-.106 sweep added the following 2025–2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2604.09624`, SECL: test-time discriminative distillation for
  self-calibrating LLMs. Reduces ECE by 56–78%. Maps directly to DVI:
  Carnot's verifier asks the discriminative question ("Is this step correct?"),
  and SECL provides the training recipe for using those discriminative signals
  to improve calibration.
- `arXiv:2601.17223`, VPRMs: Verifiable Process Reward Models with rule-based
  formal verifiers at each step, 20% F1 improvement vs outcome-only. Validates
  Carnot's GRPO-VPS architecture using Z3/semantic verifiers as step rewards.
- `arXiv:2604.25419`, JURY-RL: majority-vote proposal + formal Lean-proof
  reward for label-free RLVR. ResZero fallback for inconclusive verification.
  Pass@1 parity with supervised training. Maps to Carnot: score_candidates
  (jury) + Z3/semantic certificate (proof) + UNKNOWN preservation (ResZero).
- `arXiv:2511.07124`, EBM-CoT: contrastive hinge loss + consistency
  regularization for implicit CoT calibration. 82.73% accuracy with
  self-consistency. Directly applicable to Carnot's KAN energy tier training.
- `arXiv:2504.13134`, EBRM: post-hoc conflict-aware contrastive refinement
  for reward models. 5.97% safety alignment improvement, no retraining.
  Applicable to Carnot's SC-Energy and KAN energy tiers.
- `arXiv:2501.04971`, Self-Adaptive Ising Machines: Lagrange-relaxation-based
  dynamic energy landscape shaping without prior penalty tuning. 7,500× fewer
  samples than Digital Annealer on 300-variable knapsack. Implements Tier 4
  adaptive energy landscape from research-program.md.
- `arXiv:2601.09037`, 2D Parallel Tempering FPGA: 15 replicas, 128-node
  system, 1,920 p-bits on-chip, 4.7ms end-to-end, >10× convergence speedup.
  Directly applicable to KV260 constraint verification planning.
- `arXiv:2503.01177`, Scalable Connectivity: copy-node sparsification for
  dense Ising machines, constant frequency scaling. Theoretical foundation
  for KV260 v4 RTL sparse design (arXiv:2604.17109 inertia + this sparsification
  = v4 RTL target).

## Three Biggest Gaps

1. **Pre-test suite broken; publication hold and retro missing.** The
   `tests/python/phase5/test_intermediate_scale_v3.py` import error blocks
   all pre-test-gated tasks. The publication hold (exp1375) and retro
   (exp1376) were both exhausted without producing artifacts. With the full
   certificate chain now proven (parse_rate=1.0, semantic repair complete,
   headline self-learning achieved), the publication hold likely can be
   lifted — but we cannot assess it until the pre-test suite is fixed and
   exp1375 runs.

2. **DVI training loop never executed.** `dvi_ready=True` has appeared in
   exp1374 (`.106`), exp1358 (`.105`), and exp1344 (`.104`) — three consecutive
   milestones. The `fresh_verified_sample_count=4` from exp1374 provides the
   first actual DVI training signal. Yet the discriminative verifier update
   itself has never run. This is the core gap in FR-11: the infrastructure
   is complete, verified samples exist, but the learning step is missing.

3. **arXiv submission still pending; GRPO v7 never trained.** The paper has
   been "ready to submit pending orthogonality audit + semantic chain" since
   milestone .96. Both blockers are now resolved (`.97` orthogonality audit
   complete; `.106` semantic chain complete). With DiffuTruth, EBM-CoT, KAN
   formal verification, and JURY-RL all available as new results to include,
   the paper needs a targeted audit pass and submission. In parallel, GRPO v7
   has a confirmed working GPU path (exp1366 ran SOTA GGUF on GPU successfully)
   and the JURY-RL recipe provides formal verifier rewards without labeled data.

## Architecture (4 Phases)

```
.107 Milestone Architecture
════════════════════════════════════════════════════════════════

Phase 0 — Close .106 Missing Artifacts (MANDATORY, both unconditional)
  exp1377: Pre-test fix + .106 retro closeout ─────────────────┐
  exp1378: Publication hold v16 + claim boundary ──────────────┤
                                                               ↓
Phase 1 — Publication Sprint                                  (after Phase 0)
  exp1379: Paper integrity audit v2 + main.tex update ─────────┐
  exp1380: arXiv bundle v11 + submission ──────────────────────┤
            gated on exp1379.arxiv_submission_ready=true       ↓
                                                               ↓
Phase 2 — DVI Training + Full-Scale Evaluation               (parallel)
  exp1381: DVI discriminative verifier training v1 ────────────┐
  exp1382: Full-scale certificate + semantic repair (100+ cases)┤
            exp1382 gated on exp1381.dvi_deployed=true         ↓
                                                               ↓
Phase 3 — GRPO + New Research                                (parallel)
  exp1383: GRPO v7 JURY-RL formal verifier rewards ────────────┐ (DualGPU)
  exp1384: EBM-CoT energy calibration probe ───────────────────┤
  exp1385: Self-adaptive Ising machine probe ──────────────────┤
  exp1386: SECL discriminative self-calibration ───────────────┤
  exp1387: 2D parallel tempering KV260 FPGA estimate ──────────┤
                                                               ↓
Phase 4 — FR-11 Self-Learning + Retro                        (closes milestone)
  exp1388: FR-11 self-learning v4 (DVI + GRPO integration) ────┐
            gated on exp1381.dvi_deployed=true                 ↓
  exp1389: Milestone .107 retro ───────────────────────────────┘
            (skip_pre_test: true, mandatory)
```

## Dependency Graph

```
exp1377 ──────────────────────────────────── (required reading for all Phase 1+)
exp1378 ──────────────────────────────────── (required reading for exp1379)
exp1379 ←─────── reads exp1377, exp1378 ─── gates exp1380
exp1380 ←─────── gated_on exp1379.arxiv_submission_ready
exp1381 ←─────── reads exp1378, exp1374 ─── gates exp1382, exp1388
exp1382 ←─────── gated_on exp1381.dvi_deployed
exp1383 ─────────────────────────────────── unconditional (DualGPU)
exp1384, 1385, 1386, 1387 ──────────────── unconditional (CPU-only)
exp1388 ←─────── gated_on exp1381.dvi_deployed
exp1389 ─────────────────────────────────── unconditional (skip_pre_test: true)
```

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|-------------|-------|
| exp1377–1378 | No | Documentation + diagnosis |
| exp1379–1380 | No | Paper editing |
| exp1381 | Yes (1× RTX 3090) | DVI inference for verifier signals |
| exp1382 | Yes (1× RTX 3090) | SOTA GGUF certificate generation at scale |
| exp1383 | Yes (2× RTX 3090, DualGPU) | GRPO training |
| exp1384–1387 | No | CPU-only research probes |
| exp1388 | No | Replay + DVI-updated memory |
| exp1389 | No | Documentation |

Both RTX 3090s were idle at 0% utilization at .106 closeout. GPU path confirmed
working via exp1366 (parse_rate=1.0 on SOTA GGUF with GPU). `.107` should
actively schedule GPU work via DualGPURunner.

## Success Criteria (14)

1. `pre_test_suite_fixed` — exp1377: `ModuleNotFoundError` in `test_intermediate_scale_v3.py` resolved
2. `retro_106_complete` — exp1377: `.106` retro artifact written with all 13 experiment statuses
3. `publication_hold_reviewed` — exp1378: publication hold state assessed with `.106` full evidence
4. `paper_v7_audit_complete` — exp1379: paper updated with certificate v8, semantic repair, headline self-learning
5. `arxiv_submitted` — exp1380: arXiv bundle v11 submitted (or blocked with specific reason)
6. `dvi_training_complete` — exp1381: first discriminative verifier training run completed
7. `full_scale_repair_100_cases` — exp1382: certificate→validate→repair pipeline run at 100+ cases
8. `grpo_v7_formal_reward_measured` — exp1383: GRPO v7 with JURY-RL formal rewards measured
9. `ebm_cot_probe_complete` — exp1384: EBM-CoT contrastive hinge loss applied to KAN tier
10. `self_adaptive_ising_probe_complete` — exp1385: Lagrange-relaxation self-adaptive Ising measured
11. `secl_discriminative_calibration_complete` — exp1386: SECL calibration improvement measured
12. `parallel_tempering_kv260_estimate_complete` — exp1387: 2D PT CPU sim + KV260 LUT estimate
13. `fr11_self_learning_v4_complete` — exp1388: DVI + GRPO integrated into self-learning
14. `retro_107_complete` — exp1389: milestone retrospective written

## Decentralization Implications

All experiments in this milestone satisfy the CLAUDE.md decentralization rules:

- **Rule 1 (local-first):** GRPO v7 uses local GGUF models (SOTA unsloth models);
  DVI uses Carnot's own verifiers; no closed-weight API required for core experiments.
- **Rule 2 (closed-weight optional):** Paper audit may reference closed-weight
  comparison models for context only; no critical code path depends on them.
- **Rule 3 (distribution mirroring):** arXiv submission uses the existing
  Carnot-EBM HuggingFace + gitea mirror infrastructure.
- **Rule 4 (integration surfaces):** No new integration surface is proposed;
  existing Python API, CLI, MCP, and REST paths maintained.
- **Rule 5 (hardware portability):** KV260 FPGA estimate, self-adaptive Ising,
  and 2D parallel tempering all contribute to the FPGA/hardware sovereignty path.
- **Rule 6 (data minimization):** No closed-weight calls proposed in core verifier.
- **Rule 7 (no vendor abstractions in core):** DVI and SECL work in
  `python/carnot/verify/` using abstract protocols; no vendor-specific imports.

## Notes on Key Design Decisions

**Why pre-test fix is FIRST:** The conductor's 3-attempt retry loop was
exhausted on exp1375 and exp1376 because pre-tests failed at collection time.
Without fixing `test_intermediate_scale_v3.py`, every subsequent task with
pre-test validation enabled will also fail. This is load-bearing infrastructure.

**Why DVI training is now viable:** `dvi_ready=True` since `.104`, four fresh
semantically-verified samples from exp1374, and the SECL paper provides a
concrete training recipe. The only reason DVI hasn't run is that the certificate
chain was not producing verified samples until .106. Now it is.

**Why GRPO v7 uses JURY-RL rewards:** Previous GRPO attempts used GRPO-VPS
with step-level supervision but no formal verifier as reward signal. JURY-RL's
"votes propose, proofs dispose" architecture maps cleanly: Carnot's
score_candidates provides the jury voting, and Carnot's Z3/semantic verifiers
provide the formal proof reward. ResZero handles UNKNOWN cases, preserving
the UNKNOWN discipline from the certificate chain.

**Why inertia Ising is NOT retried on CPU:** exp1373 showed `best_inertia_alpha=0.0`
(no inertia is optimal) for dense float computation on CPU. The inertia benefit
is specific to FPGA digital fixed-point arithmetic (paper's 20-35× gain). The
2D parallel tempering route (exp1387) is the more promising FPGA path.
