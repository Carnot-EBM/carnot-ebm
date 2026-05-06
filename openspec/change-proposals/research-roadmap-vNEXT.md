# Research Roadmap vNEXT: Milestone 2026.04.109

Planned: 2026-05-06
Status: Draft for conductor execution
Predecessor: 2026.04.108 arXiv Submission + Pipeline Quality Fix + GRPO v8 NGRPO + DVI v2 + Discrete SB KV260
Roadmap YAML: `research-roadmap-next.yaml`

## ID Allocation Note

This milestone starts at `exp1412` because post-.108 issue work already used
`exp1403`, `exp1408`, and `exp1411` for non-roadmap deliverables. Avoiding those
IDs prevents artifact collisions in `results/`.

## What Milestone .108 Proved

| Track | Evidence | Finding |
|---|---|---|
| arXiv submission | `exp1390` | `submission_attempted=false`, `manual_checklist_generated=true`; SWORD credentials absent; bundle verified at `results/arxiv_bundle_v11.tar.gz`; operator action still required. |
| Pipeline failure diagnosis | `exp1391` | `failure_analysis_complete=true`; `top_failure_category=CORPUS_SPECIFIC` / DVI boundary failures; estimated semantic pass after fixes reached 1.0. |
| Test suite hygiene | `exp1392` | `collection_errors_after=0`, `test_suite_collection_clean=true`; collection fixed, execution-time debt remains. |
| GRPO v8 NGRPO | `exp1393` | `grpo_v8_improvement_pp=0.0`, `unknown_rollout_rate=1.0`; NGRPO still zero-gradient; GRPO is retired for future milestones. |
| DVI v2 + SECL | `exp1394` | `dvi_v2_auroc_delta=0.011458`, `secl_ece_reduction_pct=45.35`, `dvi_v2_deployed=true`; positive but trained on only 59 cases. |
| FR-11 self-learning v5 | `exp1395` | `fresh_verified_sample_count=1508`, `self_learning_delta_overall=1449`, `headline_result_allowed=true`; major self-learning breakthrough. |
| Semantic validation fix | `exp1396` | `semantic_validation_improvement_measured=true`, `30/30 exp1382 failures recovered`; semantic validator bug class closed. |
| Full-scale pipeline v2 | `exp1397` | `certificate_parse_rate=1.0`, `semantic_validation_pass_rate=1.0`, `full_pipeline_pass_rate=0.305`; semantic validation and MCS localization are fixed, but repair execution is absent. |
| NGRPO theory probe | `exp1398` | `ngrpo_advantage_calibration_verified=true`; theory correct, practice failed because FoVer produced no positive reward rollouts. |
| Discrete SB KV260 | `exp1399` | `bram_budget_feasible=true`, `hardware_claim_allowed=true` as estimate only, `convergence_speedup_discrete_sb=1.38`; RTL/synthesis still required for a hardware claim. |
| BiPRM R2L probe | `exp1400` | `pivot_precision_delta=-0.013`, `retrospective_verification_viable=false`; retired. |
| EBM-CoT v2 hinge-only | `exp1401` | `calibration_auroc_delta=+0.186`, `variance_worsened=true`; positive calibration signal but no publication claim until variance is fixed. |
| Milestone retro | `exp1402` | `criteria_met=12/13`, `arxiv_submitted=NOT_MET`; only arXiv remained unmet, but full pipeline still missed headline pass-rate gate. |

**Critical insight from `exp1397`:** the full pipeline bottleneck is no longer
semantic validation or MCS localization. VERGE MCS emits correct `REPAIR_HINT`
actions with localization precision 1.0, but no local LLM call executes those
hints into corrected certificates. The 30.5% full-pipeline pass rate is mostly
the no-repair subset. If repair execution succeeds on half of the remaining
69.5%, pass rate should rise to about `0.305 + 0.5 * 0.695 = 0.6525`.

## Research Signals Added Before Planning

The post-.108 sweep updated `research-references.md` before this roadmap was
finalized. The five near-term signals are:

- `arXiv:2604.07172`: token-level temperature scaling improves semantic
  uncertainty calibration. This is the direct fix for `exp1401`
  `variance_worsened=true`.
- `arXiv:2501.07301`: PRM development lessons emphasize step-level and
  response-level evaluation. Carnot now has 1508 verified traces, which is
  enough for a small FoVer PRM.
- `arXiv:2504.16828` / OpenReview ThinkPRM: generative process reward models
  can verify steps with far less supervision, supporting Carnot's certificate
  extraction architecture.
- `arXiv:2510.00977` and HuggingFace TRL's paper index: GRPO can be treated as
  an implicit DPO-style pairwise objective; DPO is the natural replacement now
  that GRPO is retired and verified pairs exist.
- `arXiv:2603.28248`: EBRM structured latent trajectory planning reports the
  exact failure Carnot must guard against in Phase 3: lower latent energy can
  hurt decoded accuracy if planner outputs drift off the decoder's training
  support.

Extropic and Kona checks remain strategic context rather than local
dependencies. Extropic still lists XTR-0 as the research platform and Z1 as
early access 2026; Logical Intelligence continues to position Kona 1.0 as an
energy-based reasoning layer for certifiable systems. The .109 plan keeps all
work local-first and hardware-portable.

## Three Biggest Gaps

1. **Certificate repair execution is missing.** Semantic validation and MCS
   localization are fixed, but `REPAIR_HINT` is not executed by an LLM. This is
   the highest-leverage path to move full-pipeline pass rate from 0.305 toward
   0.50+.

2. **FR-11 self-learning is data-rich but not yet integrated into the strongest
   verifier/training loops.** `exp1395` produced 1508 fresh verified samples,
   while `exp1394` DVI v2 used only 59. DVI v3, PRM v1, and DPO should all use
   the 1508-case corpus.

3. **Phase-3 continuous reasoning lacks adversarial drift instrumentation.**
   EBRM-style latent energy minimization is a direct path toward the PRD/Kona
   vision, but `arXiv:2603.28248` shows lower energy can be a false friend if
   latent trajectories leave decoder support. Carnot needs a small drift smoke
   test before any scale-up.

## Architecture (5 Phases)

```
.109 Milestone Architecture
========================================================================

Phase 0 - Close .108 Outstanding (fast, unconditional)
  exp1412: arXiv operator action sheet v3 ----------------------------.
  exp1413: Certificate repair execution diagnosis --------------------+--> Phase 1

Phase 1 - Major Capability Improvements (parallel)
  exp1414: Certificate LLM repair executor v1 (GPU, SOTA GGUF) -------.
  exp1415: DVI v3 on 1508 fresh cases --------------------------------+--> Phase 2
  exp1416: EBM-CoT v3 temperature calibration ------------------------'
  exp1417: EBRM latent-trajectory drift smoke ------------------------'

Phase 2 - Pipeline Scale + Self-Learning (gated)
  exp1418: FR-11 v6, gated on exp1415.dvi_v3_deployed ----------------.
  exp1419: Full-scale pipeline v3, gated on exp1414 repair executor --+--> Phase 3

Phase 3 - Research and Infrastructure Probes (parallel)
  exp1420: DPO training on 1508 verified pairs (DualGPU, SOTA GGUF) --.
  exp1421: Test suite execution debt v1 ------------------------------+
  exp1422: Discrete SB KV260 RTL specification -----------------------+
  exp1423: Process reward model v1 on FoVer 1508 pairs ---------------'

Phase 4 - Retro
  exp1424: Milestone .109 retrospective
```

## Phase Descriptions

**Phase 0 - close .108 outstanding.** `exp1412` converts the arXiv checklist
into a terse operator action sheet; it does not attempt another credentialed
submission path. `exp1413` reads `exp1397` and classifies repair-hint rows so
the LLM executor targets real failure modes instead of guessing.

**Phase 1 - capability improvements.** `exp1414` implements the missing local
LLM repair executor using mandated SOTA GGUF models. `exp1415` retrains or
updates DVI on the 1508 fresh verified cases. `exp1416` applies temperature
scaling to EBM-CoT. `exp1417` adds a bounded latent-drift smoke test for
EBRM-style energy minimization.

**Phase 2 - gated integration.** `exp1418` runs FR-11 v6 only if DVI v3 is
deployed. `exp1419` reruns the 200-case full pipeline only if the repair
executor exists.

**Phase 3 - parallel research and hygiene.** `exp1420` replaces retired GRPO
with DPO on verified pairs. `exp1421` fixes execution-time test debt left after
collection was repaired. `exp1422` moves Discrete SB from CPU feasibility to RTL
specification. `exp1423` trains a step-level PRM on the 1508 FoVer traces.

**Phase 4 - retro.** `exp1424` closes the milestone and records carry-forward
rules, with `skip_pre_test=true` to avoid retro bootstrap wedges.

## Dependency Graph

```
exp1412 ---------------------------------------------------- (unconditional)
exp1413 ---------------------------------------------------- (unconditional)

exp1414 ---------------------------------------------------- (unconditional, GPU)
exp1415 ---------------------------------------------------- (unconditional, CPU)
exp1416 ---------------------------------------------------- (unconditional, CPU)
exp1417 ---------------------------------------------------- (unconditional, CPU)

exp1418 ---- GATE: exp1415.dvi_v3_deployed == true -------- (CPU)
exp1419 ---- GATE: exp1414.repair_executor_deployed == true (GPU)

exp1420 ---------------------------------------------------- (unconditional, DualGPU)
exp1421 ---------------------------------------------------- (unconditional, CPU)
exp1422 ---------------------------------------------------- (unconditional, CPU)
exp1423 ---------------------------------------------------- (unconditional, CPU)

exp1424 ---------------------------------------------------- (retro, skip_pre_test)
```

## Hardware Requirements

| Experiment | GPU | Notes |
|---|---|---|
| `exp1412` | No | Fast doc/action sheet generation |
| `exp1413` | No | Read-only diagnosis from `exp1397` |
| `exp1414` | Yes, 1 RTX 3090 | llama.cpp GGUF repair executor; must include mandated SOTA model specs |
| `exp1415` | No | DVI update on 1508 cases is CPU-scoped unless existing code requires torch |
| `exp1416` | No | Temperature optimization is CPU |
| `exp1417` | No | Tiny latent planning smoke test |
| `exp1418` | No | FR-11 replay/update |
| `exp1419` | Yes, 1 RTX 3090 | 200-case local SOTA GGUF pipeline |
| `exp1420` | Yes, DualGPU | DPO adapter training on 1508 verified pairs |
| `exp1421` | No | Test suite execution debt |
| `exp1422` | No | RTL/spec/synthesis-estimate only; no hardware execution claim |
| `exp1423` | No | Lightweight PRM training |
| `exp1424` | No | Retro artifact |

## Success Criteria

| # | Criterion | Target | Source |
|---|---|---|---|
| 1 | arXiv action sheet complete | `submission_ready_for_operator=true` | `exp1412` |
| 2 | repair diagnosis complete | `repair_execution_diagnosis_complete=true` | `exp1413` |
| 3 | LLM repair executor deployed | `repair_executor_deployed=true` | `exp1414` |
| 4 | DVI v3 improves on v2 | `dvi_v3_auroc_delta > 0.011458` | `exp1415` |
| 5 | EBM-CoT variance fixed | `variance_worsened=false` with AUROC preserved | `exp1416` |
| 6 | latent drift smoke test complete | `latent_drift_smoke_complete=true` | `exp1417` |
| 7 | FR-11 v6 headline allowed | `headline_result_allowed=true`, fresh cases > 1508 | `exp1418` |
| 8 | full pipeline clears headline gate | `full_pipeline_pass_rate >= 0.40` | `exp1419` |
| 9 | DPO measured | `dpo_improvement_pp` determined | `exp1420` |
| 10 | execution debt fixed | `execution_failures_fixed=true` | `exp1421` |
| 11 | Discrete SB RTL spec complete | `rtl_spec_complete=true` | `exp1422` |
| 12 | PRM v1 measured | `prmv1_auroc` determined | `exp1423` |
| 13 | retro complete | `criteria_met >= 10/13` | `exp1424` |

## Prior Failure Summary

| Experiment | Prior Failure | Verdict | Addressed By |
|---|---|---|---|
| `exp1412` | `exp1390` | credentials missing; manual checklist generated | New operator action sheet; no credentialed submission attempt required |
| `exp1413` | `exp1397` | full pipeline below 0.40 due repair-hint-only path | Diagnosis isolates which hints are executable |
| `exp1414` | `exp1397` | repair execution not implemented | Implements local SOTA GGUF repair executor |
| `exp1415` | `exp1394` | DVI v2 positive but only 59 cases | Uses 1508 fresh verified cases from `exp1395` |
| `exp1416` | `exp1401` | AUROC positive but variance worsened | Applies temperature scaling from `arXiv:2604.07172` |
| `exp1418` | `exp1395` | GRPO contributed zero cases | Uses DVI v3 only; GRPO retired |
| `exp1419` | `exp1397` | full pipeline below 0.40 | Reruns after repair executor exists |
| `exp1420` | `exp1393` | GRPO retired after all-UNKNOWN collapse | DPO replaces GRPO using verified preference pairs |
| `exp1421` | `exp1392` | collection fixed, execution failures remain | Targets runtime failures only |
| `exp1422` | `exp1399` | CPU estimate only; no hardware execution | Writes RTL spec and synthesis estimate without board claim |

## Decentralization Implications

All experiments preserve the CLAUDE.md decentralization rules:

- Local-first LLM tasks use mandated GGUF models through llama.cpp and
  `cached_sota_pair()`.
- No closed-weight model is required for any result.
- Pipeline work improves Python API, CLI, MCP, and future HTTP surfaces rather
  than a vendor-specific adapter.
- DVI/PRM/DPO train from locally generated verified data.
- Discrete SB remains KV260/open-FPGA scoped and does not depend on Extropic Z1
  availability.
- arXiv submission docs do not transmit private data to third-party LLM APIs.

## Codex Default Audit

All 13 tasks use `agent_type: codex`, `model: gpt-5.5`. No
`requires_claude: true` tasks are warranted. The work is either formulaic code,
bounded diagnostics, deterministic training, or doc/action-sheet generation with
clear artifact gates.
