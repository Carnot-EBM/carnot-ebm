# Research Roadmap vNEXT: Milestone 2026.04.108

Planned: 2026-05-05
Status: Draft for conductor execution
Predecessor: 2026.04.107 Pre-test Fix + arXiv Submission Sprint + DVI Training + GRPO v7 JURY-RL
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .107 Proved

| Track | Evidence | Finding |
|---|---|---|
| Pre-test fix | `exp1377` | `pre_test_fix_applied=true`, `tests_passing_after_fix=5/5` — phase5 intermediate_scale_v3 import blocker resolved; `full tests/python` still has unrelated collection debt (separate task). |
| Publication hold | `exp1378` | `publication_hold_state=lift_recommended`, `all_primary_blockers_resolved=true` — all primary blockers resolved; external parity claims remain disallowed. |
| Paper integrity audit | `exp1379` | `critical_issues_after_audit=0`, `arxiv_submission_ready=true` — paper compiled with tectonic; v7 ready to submit. |
| arXiv bundle | `exp1380` | `bundle_created=true`, `submission_attempted=false`, `submission_result=not_attempted_arxiv_upload_cli_missing` — bundle at results/arxiv_bundle_v11.tar.gz; arxiv_upload CLI missing blocked actual submission. ONLY criterion NOT MET. |
| DVI training v1 | `exp1381` | `dvi_deployed=true`, `dvi_auroc_delta=0.003486`, `dvi_trained_auroc=0.394526` — only 4 fresh cases used; tiny delta. 59 fresh cases now available from exp1388. |
| Full-scale pipeline | `exp1382` | `certificate_parse_rate=1.0`, `semantic_validation_pass_rate=0.59`, `full_pipeline_pass_rate=0.29` — headline certificate extraction proven; 41% semantic validation failure rate is the primary quality gap. |
| GRPO v7 JURY-RL | `exp1383` | `grpo_v7_improvement_pp=0.0`, `formal_reward_pass_rate=0.0`, `all_rollouts_unknown=true` — all rollouts returned UNKNOWN from semantic verifier; zero-gradient collapse. Root cause: JURY-RL requires at least one positive confirmation to compute advantage; ResZero produced zero-mean redistribution without positive signal. NGRPO (arXiv:2509.18851) Advantage Calibration is the fix. |
| EBM-CoT energy calibration | `exp1384` | `calibration_auroc_delta=0.185`, `implicit_cot_energy_viable=true`, `consistency_regularization_effect=worsened_paraphrase_variance` — hinge loss alone is strong (+0.185 AUROC); consistency regularization WORSENED variance. |
| Self-adaptive Ising | `exp1385` | `adaptive_ising_viable=true`, `convergence_speedup=1.38` — Lagrange relaxation improves convergence 1.38x without penalty tuning. |
| SECL self-calibration | `exp1386` | `ece_reduction_pct=34.96`, `secl_viable_for_dvi=true` — 35% ECE reduction; discriminative signal confirmed applicable to DVI. |
| 2D parallel tempering KV260 | `exp1387` | `convergence_speedup_2d_pt=1.08`, `15_replicas_lut_estimate=540K`, `kv260_budget=117K`, `hardware_claim_allowed=false` — 15 replicas requires 540K LUTs vs 117K budget; max 3 replicas fit. Discrete SB (arXiv:2510.12407) is the BRAM-limited alternative. |
| FR-11 self-learning v4 | `exp1388` | `fresh_verified_sample_count=59`, `grpo_cases_integrated=0`, `headline_result_allowed=true` — 59 fresh cases via DVI-only replay (GRPO contributed 0 because exp1383 had no improvement); DVI checkpoint active. |
| Milestone retro | `exp1389` | `criteria_met=13/14`, `arxiv_submitted=NOT_MET` — single failure; all others met. skip_pre_test + STEP 0 discipline worked. |

**Key operational finding.** The operational retro (results/operational_retro_2026_04_107.json)
found the arXiv Bundle v10 gate (Exp 1269) consumed 62.5% of wall time for the SEVENTH
consecutive milestone. Three DOOMED_RERUN_BLOCK slots consumed 37.5%. Both RTX 3090s were
idle throughout. The conductor's activation-time gate preflight is still not mechanically
enforced — the same structural defect has generated the same retro finding for seven consecutive
milestones. Milestone .108 MUST resolve arXiv submission (eliminating the gate) and MUST NOT
propose SOTA GGUF cache tasks without a substantive prior_failures entry.

## Research Signals Added Before Planning

The post-.107 sweep added the following 2025–2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2509.18851`, NGRPO: Negative-enhanced GRPO with Advantage Calibration via virtual
  max-reward sample. Directly solves exp1383's zero-gradient collapse when all rollouts are
  UNKNOWN. Virtual max-reward sample breaks the zero-mean ResZero symmetry, producing non-trivial
  advantage even when no rollout is formally confirmed.
- `arXiv:2508.01682`, BiPRM: Bidirectional Process Reward Model with R2L retrospective stream.
  Forward PRM scores left-to-right; retrospective stream scores from answer backward, identifying
  pivotal reasoning steps. Directly applicable to Carnot's step-level verification chain.
- `arXiv:2510.12407`, Discrete SB on KV260: Discrete Simulated Bifurcation implemented on
  Xilinx KV260. BRAM-limited (not LUT-limited), 256-variable problems fit. Alternative to 2D PT
  (exp1387 was LUT-limited to 3 replicas).
- `arXiv:2602.04200`, Restoring Sparsity in Potts Machines: Mean-field sparsification for
  FPGA-class Potts machines (q=3). Reduces connectivity without sacrificing solution quality.
  Applicable to KV260 Potts extension motivated by exp534.
- `arXiv:2603.28204`, ERPO: Token-level entropy-regulated policy optimization. Prevents reward
  hacking via per-token entropy constraints; orthogonal complement to NGRPO advantage calibration.
- `arXiv:2509.21880`, RL-ZVP: Zero-variance prompt exploitation via entropy shaping. Identifies
  and exploits prompts with low-variance model outputs for more stable RL training signal.
- `arXiv:2510.01069`, Typed CoT Curry-Howard: ICLR 2026. Type-theoretic interpretation of CoT
  steps as proof terms; type errors = reasoning errors. Provides formal semantics for
  Carnot's certificate step verification (complementary to Z3 arithmetic and AST structural checks).

## Three Biggest Gaps

1. **arXiv submission still not attempted.** Paper is audit-complete (exp1379,
   `critical_issues_after_audit=0`, `arxiv_submission_ready=true`). Bundle exists
   at `results/arxiv_bundle_v11.tar.gz`. The sole blocker is the arXiv upload
   mechanism: the `arxiv_upload` CLI was missing. Exp1390 installs the arXiv
   SWORD/APP Deposit API path or produces step-by-step manual upload instructions
   with the ready bundle. This is publication-blocking; it gates further paper
   iteration.

2. **Full pipeline quality insufficient for publication (0.59 / 0.29).** Exp1382
   achieved `certificate_parse_rate=1.0` (excellent) but only 59% semantic validation
   pass rate and 29% full pipeline pass rate. Before scaling to 200+ cases, we need
   to understand WHY 41% of validated certificates fail semantic validation. Exp1391
   runs a structured failure diagnosis on the exp1382 failures. Exp1396 implements
   fixes. Exp1397 re-runs at 200 cases with fixes applied.

3. **GRPO has produced 0.0pp improvement across all attempts.** Exp1383 was the
   seventh GRPO experiment. Root cause: all rollouts returned UNKNOWN from Carnot's
   semantic verifier, causing ResZero to produce zero-mean advantage — no gradient.
   NGRPO (arXiv:2509.18851) solves this via a virtual max-reward sample injected
   into the group to break symmetry. Exp1393 applies NGRPO directly to exp1383's
   failure mode.

## Architecture (5 Phases)

```
.108 Milestone Architecture
══════════════════════════════════════════════════════════════════

Phase 0 — Close .107 Missing Artifact (MANDATORY, unconditional)
  exp1390: arXiv submission via SWORD API ──────────────────────┐
            (bundle ready at results/arxiv_bundle_v11.tar.gz)   │
  exp1391: Full-scale pipeline failure diagnosis ───────────────┤
            (classify 41% semantic validation failures)         │
                                                                ↓
Phase 1 — Infrastructure (parallel)
  exp1392: Test suite hygiene v2 ──────────────────────────────┐
            (fix unrelated collection debt from exp1377)        ↓

Phase 2 — GRPO Fix + DVI Improvement (parallel, GPU-gated)     ↓
  exp1393: GRPO v8 NGRPO zero-reward fix ──────────────────────┐  (DualGPU MANDATORY)
  exp1394: DVI v2 + SECL combined (59 fresh cases) ────────────┤
  exp1395: FR-11 v5 ────────────────────────────────────────────┤
            gated on exp1394.dvi_v2_deployed==true              ↓

Phase 3 — Pipeline Quality + New Research (parallel)            ↓
  exp1396: Semantic validation pass rate fix v1 ───────────────┐
            gated on exp1391.failure_analysis_complete==true    │
  exp1397: Full-scale pipeline v2 (200 cases) ─────────────────┤  (GPU, gated on exp1396)
  exp1398: NGRPO theory probe (CPU) ───────────────────────────┤
  exp1399: Discrete SB KV260 CPU simulation ───────────────────┤
  exp1400: BiPRM retrospective verification probe ─────────────┤
  exp1401: EBM-CoT v2 hinge-only ─────────────────────────────┤
                                                                ↓
Phase 4 — Retro
  exp1402: Milestone .108 retro ──────────────────────────────────
            (skip_pre_test: true, mandatory)
```

## Dependency Graph

```
exp1390 ─── unconditional ──────────────────────────────
exp1391 ─── unconditional ──────────────────────────────
exp1392 ─── unconditional ──────────────────────────────

exp1393 ─── unconditional (DualGPU) ────────────────────
exp1394 ─── unconditional ──────────────────────────────
exp1395 ─── gated on exp1394.dvi_v2_deployed==true ─────

exp1396 ─── gated on exp1391.failure_analysis_complete==true
exp1397 ─── gated on exp1396.semantic_validation_improvement_measured==true (GPU)
exp1398 ─── unconditional ──────────────────────────────
exp1399 ─── unconditional ──────────────────────────────
exp1400 ─── unconditional ──────────────────────────────
exp1401 ─── unconditional ──────────────────────────────

exp1402 ─── all other experiments complete ─────────────
```

## Experiment Summaries

| ID | Title | Phase | GPU | Pri | Gate |
|---|---|---|---|---|---|
| exp1390 | arXiv Submission via SWORD API | 0 | No | critical | unconditional |
| exp1391 | Full-Scale Pipeline Failure Diagnosis | 0 | No | critical | unconditional |
| exp1392 | Test Suite Hygiene v2 | 1 | No | medium | unconditional |
| exp1393 | GRPO v8 NGRPO Zero-Reward Fix | 2 | Yes (dual) | critical | unconditional |
| exp1394 | DVI v2 + SECL Combined | 2 | No | high | unconditional |
| exp1395 | FR-11 Self-Learning v5 | 2 | No | high | exp1394.dvi_v2_deployed |
| exp1396 | Semantic Validation Pass Rate Fix v1 | 3 | No | high | exp1391.failure_analysis_complete |
| exp1397 | Full-Scale Pipeline v2 (200 cases) | 3 | Yes | high | exp1396.semantic_validation_improvement_measured |
| exp1398 | NGRPO Theory Probe | 3 | No | medium | unconditional |
| exp1399 | Discrete SB KV260 CPU Simulation | 3 | No | medium | unconditional |
| exp1400 | BiPRM Retrospective Verification Probe | 3 | No | medium | unconditional |
| exp1401 | EBM-CoT v2 Hinge-Only | 3 | No | medium | unconditional |
| exp1402 | Milestone .108 Retro | 4 | No | critical | unconditional |

## Hardware Requirements

- **DualGPU**: exp1393 (GRPO v8 NGRPO) — MANDATORY, both RTX 3090s required
- **Single GPU**: exp1397 (full-scale pipeline v2, 200 cases SOTA GGUF inference)
- **CPU-only**: exp1390, exp1391, exp1392, exp1394, exp1395, exp1396, exp1398, exp1399, exp1400, exp1401, exp1402

**GPU scheduling note**: exp1393 and exp1397 must not run concurrently if only one
GPU rig is available. Conductor should schedule exp1393 first (higher priority),
then exp1397 after exp1393 completes.

## Success Criteria

The .108 milestone succeeds if ALL of the following are met:

| Criterion | Source | Target |
|---|---|---|
| arxiv_submitted | exp1390 | submission_attempted=true AND submission_result!="not_attempted" |
| failure_diagnosis_complete | exp1391 | failure_analysis_complete=true AND failure_root_causes identified |
| test_suite_collection_clean | exp1392 | collection_errors_remaining=0 |
| grpo_ngrpo_measured | exp1393 | grpo_v8_improvement_pp > 0 OR retire_if_same_verdict=true applied |
| dvi_v2_deployed | exp1394 | dvi_v2_deployed=true AND dvi_v2_auroc_delta > 0.003486 |
| fr11_v5_fresh_count_growing | exp1395 | fresh_verified_sample_count > 59 OR gate_blocked_artifact_emitted |
| semantic_validation_fix_measured | exp1396 | semantic_validation_improvement_measured=true AND fix_applied |
| full_pipeline_v2_at_scale | exp1397 | cases_evaluated >= 200 OR gate_blocked_artifact_emitted |
| ngrpo_theory_confirmed | exp1398 | ngrpo_advantage_calibration_tested=true |
| discrete_sb_kv260_estimated | exp1399 | bram_budget_feasible determined AND hardware_claim_set |
| biprm_verified | exp1400 | retrospective_verification_viable determined |
| ebm_cot_hinge_only_measured | exp1401 | calibration_auroc_delta > 0 (hinge-only, no consistency reg) |
| retro_108_complete | exp1402 | criteria_met/criteria_total written |

## Prior Failure Summary (Carry-Forwards)

| Experiment | .107 Verdict | Root Cause | .108 Fix |
|---|---|---|---|
| exp1380 | submission_attempted=false | arxiv_upload CLI missing | exp1390: SWORD API or manual upload |
| exp1381 | dvi_auroc_delta=0.003486 | Only 4 fresh cases | exp1394: 59 fresh cases (exp1388) |
| exp1382 | semantic_validation_pass_rate=0.59 | Unknown failure modes | exp1391 diagnosis → exp1396 fix |
| exp1383 | grpo_v7_improvement_pp=0.0 | All rollouts UNKNOWN → zero gradient | exp1393: NGRPO virtual max-reward |
| exp1384 | consistency reg worsened variance | Consistency term counterproductive | exp1401: hinge loss only |
| exp1387 | 15 replicas = 540K LUTs (over budget) | LUT-limited, not BRAM | exp1399: Discrete SB (BRAM-limited) |
| exp1388 | grpo_cases_integrated=0 | exp1383 had no positive improvement | exp1395: DVI v2 + NGRPO integration |

## Decentralization Implications

- exp1390 (arXiv): paper submission mechanism is publisher-side; content is open. No sovereignty risk.
- exp1393 (GRPO v8): uses local GGUF models from unsloth on HuggingFace; no closed-weight dependency.
- exp1394 (DVI v2): trains on existing FoVer corpus; no new cloud dependency.
- exp1395-exp1397: local inference and verification; no closed-weight dependency.
- All experiments use local open-weight models as primary; closed-weight paths behind optional flags.
- MUST confirm exp1393 MODEL_SPECS uses cached_sota_pair() with at least one of:
  - unsloth/Qwen3.6-35B-A3B-GGUF
  - unsloth/gemma-4-31B-it-GGUF
  - unsloth/gemma-4-26B-A4B-it-GGUF

All 7 decentralization rules (CLAUDE.md) are satisfied: local-first, closed-weight optional,
distribution mirroring N/A (no new weights published this milestone), multiple integration
surfaces unchanged, hardware portability preserved.
