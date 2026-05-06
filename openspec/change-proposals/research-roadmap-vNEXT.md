# Research Roadmap vNEXT: Milestone 2026.04.109

Planned: 2026-05-06
Status: Draft for conductor execution
Predecessor: 2026.04.108 arXiv Submission + Pipeline Quality Fix + GRPO v8 NGRPO + DVI v2 + Discrete SB KV260
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .108 Proved

| Track | Evidence | Finding |
|---|---|---|
| arXiv submission | `exp1390` | `submission_attempted=false`, `manual_checklist_generated=true` — SWORD API credentials absent; full manual checklist written at `docs/arxiv-manual-submission-checklist.md`; bundle at `results/arxiv_bundle_v11.tar.gz`; operator action required. |
| Pipeline failure diagnosis | `exp1391` | `failure_analysis_complete=true`, `top_failure_category=DVI_BOUNDARY_FAILURES` — DVI boundary failures dominated the 41% semantic validation failure rate; exp1396 targeted fixes. |
| Test suite hygiene v2 | `exp1392` | `collection_errors_after=0`, `test_suite_collection_clean=true` — collection errors resolved; execution-time failures and spec-coverage debt remain as separate carry-forward. |
| GRPO v8 NGRPO | `exp1393` | `grpo_v8_improvement_pp=0.0`, `unknown_rollout_rate=1.0`, `retire_if_same_verdict=true` triggered — NGRPO still produced zero gradient; FoVer corpus fundamentally incompatible with JURY-RL. GRPO permanently retired from future milestones. |
| DVI v2 + SECL | `exp1394` | `dvi_v2_auroc_delta=0.011458`, `secl_ece_reduction_pct=45.35`, `dvi_v2_deployed=true` — positive improvement on 59 cases; DVI v3 on 1508 cases expected to deliver substantially larger delta. |
| FR-11 self-learning v5 | `exp1395` | `fresh_verified_sample_count=1508`, `self_learning_delta_overall=1449`, `headline_result_allowed=true` — major breakthrough; 1508 vs 59 fresh cases. DVI v2 + SECL active verifier. |
| Semantic validation fix | `exp1396` | `semantic_validation_improvement_measured=true`, `30/30 exp1382 failures recovered` — fixes applied. |
| Full-scale pipeline v2 | `exp1397` | `certificate_parse_rate=1.0`, `semantic_validation_pass_rate=1.0`, `full_pipeline_pass_rate=0.305` — semantic validation is now PERFECT (1.0) but full pipeline below 0.40 headline gate. ROOT CAUSE: VERGE MCS localizes correctly (`mcs_repair_localization_rate=1.0`, `repair_hint_precision=1.0`) but repair step generates `REPAIR_HINT` labels — LLM execution of the repair is NOT implemented. `semantic_equivalence_passed=False` in all repair rows. |
| NGRPO theory probe | `exp1398` | `ngrpo_advantage_calibration_verified=true`, `theory_supports_exp1393=true` — theory correct; practice still failed because corpus produces no positive rewards. |
| Discrete SB KV260 | `exp1399` | `bram_budget_feasible=true`, `hardware_claim_allowed=true` (estimate only), `convergence_speedup_discrete_sb=1.38` — BRAM and LUT budget both fit; hardware execution not performed; synthesis and board validation required for real claim. |
| BiPRM R2L probe | `exp1400` | `pivot_precision_delta=-0.013`, `retrospective_verification_viable=false` — RETIRED; proxy-only negative delta. |
| EBM-CoT v2 hinge-only | `exp1401` | `calibration_auroc_delta=+0.186`, `variance_worsened=true`, `implicit_cot_energy_viable=true` — strong positive AUROC but paraphrase energy variance still worse. Temperature calibration needed (see arXiv:2604.07172). |
| Milestone retro | `exp1402` | `criteria_met=12/13`, `arxiv_submitted=NOT_MET` — single failure; GRPO retired; pipeline semantic fixed; full pipeline below headline; EBM-CoT variance outstanding. |

**Critical insight from exp1397.** The full pipeline bottleneck is NOT semantic validation (now 1.0)
and NOT MCS localization (precision=1.0). It is repair EXECUTION: the VERGE MCS engine emits
`REPAIR_HINT` actions that correctly identify the broken certificate step, but no LLM call is
made to actually produce a corrected certificate. `semantic_equivalence_passed=False` in every
repair row. The 30.5% that pass the full pipeline are cases that required NO repair. The remaining
69.5% need LLM-guided repair execution. If repair success rate reaches 50%, full pipeline
rises to 0.305 + 0.5*0.695 = 0.65.

## Research Signals Added Before Planning

The post-.108 sweep added the following 2025–2026 sources to
`research-references.md` before this roadmap was designed:

- `arXiv:2604.07172`, Temperature Scaling for Semantic Uncertainty: Temperature-optimized
  calibration reduces output variance while preserving discriminative power (AUROC).
  Direct fix for EBM-CoT v2's `variance_worsened=True` finding from exp1401.
- `arXiv:2501.07301`, Process Reward Models in Mathematical Reasoning: Error-aware step-level
  PRM training on 1,000–5,000 labeled traces. 1508 verified cases from exp1395 are directly
  in this range. Enables step-level discriminative PRM on FoVer.
- `arXiv:2504.16828`, Process Reward Models That Think: Generative PRM approach (critique
  then score) matches discriminative PRM at 1% label cost. Validates Carnot's certificate
  extraction as the "thinking" step in a generative PRM architecture.
- `arXiv:2510.00977`, GRPO is Secretly DPO: Every GRPO gradient update equals a DPO update
  on an implicit pair. DPO is a direct substitute for retired GRPO when 1508 verified
  preference pairs are available.
- `arXiv:2504.11343`, Minimalist RL Reasoning: Rejection sampling fine-tuning (RAFT) matches
  GRPO at lower cost; sidesteps all-UNKNOWN collapse. RAFT alternative to DPO if DPO unstable.

## Three Biggest Gaps

1. **Certificate repair execution not implemented.** VERGE MCS localizes incorrectly-reasoned
   steps with 100% precision but only emits `REPAIR_HINT`. No LLM call executes the repair.
   `semantic_equivalence_passed=False` in all 69.5% of cases requiring repair. Implementing
   LLM-guided repair execution (with SOTA GGUF model) is the single highest-leverage action
   for raising `full_pipeline_pass_rate` from 0.305 toward 0.50+. Gate: if repair success
   rate >= 50% on repaired cases, `full_pipeline_pass_rate` rises to approximately 0.65.

2. **arXiv submission outstanding (9 consecutive milestones).** Bundle v11 is complete and
   audit-verified. Manual checklist is at `docs/arxiv-manual-submission-checklist.md`.
   The operator must log into arxiv.org and follow the checklist. This milestone's
   submission task (exp1403) produces a minimal terse action sheet that can be acted on
   in under 5 minutes — no experiment credentials needed.

3. **EBM-CoT variance outstanding.** `calibration_auroc_delta=+0.186` confirmed positive
   from exp1401. `variance_worsened=True` prevents publication claim. Temperature scaling
   (arXiv:2604.07172) applied after hinge training is the targeted fix — it optimizes T*
   to reduce paraphrase energy variance without sacrificing AUROC.

## Architecture (4 Phases)

```
.109 Milestone Architecture
══════════════════════════════════════════════════════════════════

Phase 0 — Close .108 Outstanding (MANDATORY, unconditional, fast)
  exp1403: arXiv operator action sheet v3 ───────────────────────┐
            (verify bundle + produce terse SUBMIT NOW doc)       │
  exp1404: Certificate repair execution diagnosis ───────────────┤
            (classify 69.5% repair-hint cases; repairability)    │
                                                                  ↓
Phase 1 — Major Capability Improvements (parallel)
  exp1405: Certificate LLM repair executor v1 ──────────────────┐
            (GPU: SOTA model repairs REPAIR_HINT cases)          │
  exp1406: DVI v3 on 1508 fresh cases ───────────────────────────┤
            (CPU: expect AUROC delta >> 0.011 from 59-case DVI)  │
  exp1407: EBM-CoT v3 temperature calibration ──────────────────┤
            (CPU: T* scaling to fix variance_worsened from 1401) ↓

Phase 2 — Pipeline Scale + Self-Learning (gated)
  exp1408: FR-11 v6 (gated on exp1406 DVI v3 deployed) ─────────┐
            (CPU: measure improvement vs 1508-case baseline)      │
  exp1409: Full-scale pipeline v3 200+ cases ────────────────────┤
            (GPU: with LLM repair executor from exp1405)         ↓

Phase 3 — Research Probes (parallel, unconditional)
  exp1410: DPO training on 1508 verified pairs ──────────────────┐
            (GPU DualGPU: GRPO alternative with verified pairs)  │
  exp1411: Test suite execution debt v1 ─────────────────────────┤
            (CPU: fix execution-time failures from exp1392)      │
  exp1412: Discrete SB KV260 RTL specification ──────────────────┤
            (CPU: Verilog sketch + synthesis estimate)           │
  exp1413: Process reward model v1 on FoVer 1508 pairs ──────────┤
            (CPU: step-level PRM on 1508 verified traces)        ↓

Phase 4 — Retro (MANDATORY, closes milestone)
  exp1414: Milestone .109 retrospective ─────────────────────────┘
```

## Dependency Graph

```
exp1403 ──────────────────────────────────────────── (unconditional)
exp1404 ──────────────────────────────────────────── (unconditional)

exp1405 ──────────────────────────────────────────── (unconditional, GPU)
exp1406 ──────────────────────────────────────────── (unconditional, CPU)
exp1407 ──────────────────────────────────────────── (unconditional, CPU)

exp1408 ──── GATE: exp1406.dvi_v3_deployed == true ── (CPU)
exp1409 ──── GATE: exp1405.repair_executor_deployed == true ── (GPU)

exp1410 ──────────────────────────────────────────── (unconditional, GPU DualGPU)
exp1411 ──────────────────────────────────────────── (unconditional, CPU)
exp1412 ──────────────────────────────────────────── (unconditional, CPU)
exp1413 ──────────────────────────────────────────── (unconditional, CPU)

exp1414 ──────────────────────────────────────────── (retro, skip_pre_test)
```

## Hardware Requirements

| Experiment | GPU | Notes |
|---|---|---|
| exp1403 | No | Fast verification + doc generation |
| exp1404 | No | Read-only diagnosis from exp1397 artifact |
| exp1405 | Yes (1 RTX 3090) | SOTA GGUF inference for repair execution |
| exp1406 | No | DVI training on 1508 cases is CPU-only |
| exp1407 | No | Temperature optimization is CPU-only |
| exp1408 | No | FR-11 replay is CPU-only |
| exp1409 | Yes (1 RTX 3090) | SOTA GGUF for 200-case full pipeline |
| exp1410 | Yes (DualGPU) | DPO training on 1508 pairs, tensor_split=[0.5,0.5] |
| exp1411 | No | Test suite execution, CPU |
| exp1412 | No | RTL specification, CPU |
| exp1413 | No | PRM training on 1508 cases, CPU |
| exp1414 | No | Retro artifact |

## Success Criteria

| # | Criterion | Target | Source |
|---|---|---|---|
| 1 | arXiv operator action sheet complete | submission_ready_for_operator=true | exp1403 |
| 2 | Repair execution diagnosis complete | repair_execution_diagnosis_complete=true | exp1404 |
| 3 | LLM repair executor deployed | repair_executor_deployed=true | exp1405 |
| 4 | DVI v3 AUROC delta > DVI v2 (0.011) | dvi_v3_auroc_delta > 0.011 | exp1406 |
| 5 | EBM-CoT variance not worsened | variance_worsened=false after T* scaling | exp1407 |
| 6 | FR-11 v6 headline allowed | headline_result_allowed=true, fresh > 1508 | exp1408 |
| 7 | Full pipeline pass rate > 0.40 | full_pipeline_pass_rate >= 0.40 | exp1409 |
| 8 | DPO improvement measured | dpo_improvement_pp determined (pos or neg) | exp1410 |
| 9 | Test suite execution clean | execution_failures_fixed=true | exp1411 |
| 10 | Discrete SB RTL estimate complete | rtl_spec_complete=true | exp1412 |
| 11 | PRM v1 trained and measured | prmv1_auroc determined | exp1413 |
| 12 | Retro complete | criteria_met >= 9/12 | exp1414 |

## Prior Failure Summary

| Experiment | Prior Failure | Verdict | Addressed By |
|---|---|---|---|
| exp1403 | exp1390 | credentials_missing_manual_checklist_generated | New task: operator action sheet (no credentials needed) |
| exp1405 | exp1397 | full_pipeline_below_0.40_repair_hint_only | Root cause: LLM repair execution not implemented; this task implements it |
| exp1406 | exp1394 | dvi_v2_auroc_delta=0.011_only_59_cases | 1508 fresh cases now available from exp1395 |
| exp1408 | exp1395 | grpo_v8_cases_integrated=0 | GRPO retired; FR-11 v6 uses DVI v3 checkpoint only |
| exp1409 | exp1397 | full_pipeline_below_0.40 | exp1405 implements repair executor; exp1409 re-runs with it |
| exp1410 | exp1393 (GRPO retired) | grpo_v8_ngrpo_no_improvement_all_unknown_retired | DPO replaces GRPO; 1508 verified pairs available |
| exp1412 | exp1399 | hardware_execution_performed=false | RTL specification moves from CPU simulation toward synthesis |

## Decentralization Implications

All experiments in this milestone respect the decentralization rules in CLAUDE.md:
1. **Local-first**: repair executor uses `cached_sota_pair()` with GGUF models; DPO training uses same local models
2. **GRPO retired**: DPO is simpler, more data-efficient, and works with local models
3. **No closed-weight calls**: all LLM calls through llama.cpp local inference path
4. **Multi-surface**: pipeline improvements serve CLI, MCP, HTTP REST, Python API surfaces equally
5. **Hardware portability**: discrete SB RTL specification targets KV260 (open FPGA) not proprietary accelerator
6. **Data minimization**: verified pairs are locally generated; no external API calls for training data

## Codex Default Audit

All 12 experiments use `agent_type: codex`, `model: gpt-5.5`. No `requires_claude: true` tasks.
Reasoning: exp1405 (repair executor) involves multi-file edits but follows a well-established
repair pattern; exp1406 (DVI training) is a deterministic training loop; exp1410 (DPO) is a
well-documented training recipe. None require open-ended cross-context judgment beyond codex's
capability envelope.
