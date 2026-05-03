# Milestone 2026.04.95 — Design Document

**Milestone:** 2026.04.95
**Title:** Pre-Commit Safety + Paper Related Work + GRPO-VPS Training + Phase-5 Derisking A-C + Reward Gaming Defense
**Estimated wall time:** ~480 min
**Experiments:** exp1216–exp1228 (13 total)
**Planned:** 2026-05-03

---

## What Milestone 2026.04.94 Proved

**13/13 criteria met — clean sweep.** Key results:

| Experiment | Verdict | Finding |
|---|---|---|
| exp1203 | pre_test_fixed | Pre-test suite broken since .92 — RLIMIT_AS/watchdog issue fixed |
| exp1204 | template_updated | STEP 0 skeleton pattern documented; RESOLVED in known-issues |
| exp1205 | all_5_critical_resolved | All 5 paper integrity critical issues fixed; arXiv bundle unblocked |
| exp1206 | bundle_ready_for_review | carnot-arxiv-v8.tar.gz built; publication hold still active (operator gate) |
| exp1207 | gpu_offload_verified | llama.cpp CUDA GPU offload verified; ≥50 tok/s throughput |
| exp1208 | improvement_below_v4 | **REGRESSION: GRPO v5 DualGPU = -35pp vs v4 baseline (+10pp)** |
| exp1209 | step_supervision_improves_over_outcome | **BREAKTHROUGH: GRPO-VPS +24pp delta with step-level verification** |
| exp1210 | phase4_advantage_on_intractable | Phase 4 Blocked Gibbs beats BFS on ≥50% intractable 15x15 puzzles |
| exp1211 | fover_expanded_k5_improved | FoVer corpus expanded, k=5 AUROC improved with hard negatives |
| exp1212 | constraint_addition_improves_precision | Tier 1 constraint ADDITION beats reweighting baseline |
| exp1213 | insufficient_logprob_coverage | SDPO failed: llama.cpp lacks per-token logprob access for KL |
| exp1214 | nonogram_shipped_e0_at_solution | Nonogram (Picross) WOPR cartridge shipped, E=0 at solution |
| exp1215 | milestone_94_clean_sweep_13_of_13 | 13/13 criteria met |

**Three dominant findings from .94:**

1. **GRPO v5 TinyV v2 regressed badly (-35pp).** DualGPU training with TinyV abstention produced
   significantly worse results than the v4 structural warm-up (+10pp). Root cause unknown —
   TinyV abstention may suppress too much signal, or DualGPU tensor split introduces instability.
   Must diagnose before proposing GRPO v6.

2. **GRPO-VPS step supervision gives +24pp delta — the biggest GRPO result yet.** Step-level
   supervision via CausalReasoningVerifier + Z3MathVerifier produces dramatically better credit
   attribution than outcome-only rewards. Must validate in a full training run.

3. **SDPO needs a fundamentally different implementation.** llama.cpp does not expose per-token
   logprobs cleanly enough for the teacher-student KL formulation. FSPO (arXiv 2505.24630)
   provides an alternative using per-step factuality scores — compatible with existing verifiers.

**Outstanding MANDATORY infrastructure items from known-issues.md:**

1. **Pre-commit `staged_files_only` data loss (HIGHEST PRIORITY)** — 5+ files lost during .94
   session. Stash-and-restore failure causes permanent loss of working-tree changes.

2. **Auto-populate `prior_failures` from failure-ledger (MANDATORY operations)** — 7+
   DOOMED_RERUN_BLOCK false-positives per milestone. Each requires 5 min operator intervention.

3. **Paper v6 Related Work Overhaul (MANDATORY for publication)** — Deep Research dive found
   5 papers reviewers will flag as missing. Novelty boundary discipline required.

---

## Three Biggest Gaps Entering .95

### Gap 1: GRPO regression (-35pp) must be diagnosed before v6 design

The v4→v5 regression is anomalous: TinyV abstention should improve signal quality, not hurt it.
-35pp regression means either high abstention rate suppresses training signal, DualGPU tensor
split introduces training instability, or threshold misconfiguration causes near-zero gradient.
Without diagnosing the root cause, GRPO v6 design is guesswork.

### Gap 2: GRPO-VPS +24pp needs full training validation

exp1209 measured the step supervision delta on 50 GSM8K questions (evaluation mode, not training).
The +24pp might not transfer to an actual training run — step rewards need compatibility with
GRPO's advantage normalization, and variance may differ from the evaluation measurement.
A full GRPO training run using step-level supervision is required to confirm stability.

### Gap 3: Phase-5 derisking track is committed and overdue (.94 was supposed to start it)

The in-situ training phase-5 derisking (exp_NEXT_A, B, C) was committed to start in .94 per
the Phase-5 commitment (feedback_phase5_derisking_committed.md). Deferred due to infrastructure
rescue in .94. Must ship in .95 — experiments take ~3 weeks total.

---

## Architecture Diagram

```
2026.04.95 Execution Flow
═══════════════════════════════════════════════════════════════════

Phase 0 — Infrastructure (unconditional, MANDATORY)
┌─────────────────────────────────────────────────────────────────┐
│ exp1216: Pre-commit staged_files_only fix + batching-check      │
│   agent: claude/opus  |  ~40 turns  |  CPU only                 │
│                                                                  │
│ exp1217: Auto-populate prior_failures from failure-ledger       │
│   agent: codex/gpt-5.5  |  ~30 turns  |  CPU only             │
└─────────────────────────────────────────────────────────────────┘
         ↓ (both unconditional)

Phase 1 — Paper (MANDATORY)
┌─────────────────────────────────────────────────────────────────┐
│ exp1218: Paper v6 Related Work Overhaul                         │
│   5 citations + novelty boundaries + thesis sentence            │
│   agent: claude/sonnet  |  ~40 turns  |  CPU only             │
└─────────────────────────────────────────────────────────────────┘
         ↓

Phase 2 — GRPO Diagnosis + Training (sequential dependency chain)
┌─────────────────────────────────────────────────────────────────┐
│ exp1219: GRPO v5 Regression Diagnosis                          │
│   Read exp1208 artifact + run diagnostics; diagnose -35pp      │
│   agent: claude/opus  |  ~50 turns  |  CPU + optional GPU     │
│                                                                  │
│ exp1220: GRPO-VPS Full Training Run [gated on 1219]            │
│   Full GRPO training with CausalReasoning + Z3Math step rewards │
│   agent: claude/opus  |  ~60 turns  |  GPU MANDATORY          │
│                                                                  │
│ exp1221: GRPO v6 FSPO+VPS Combined [gated on 1220 >0pp]       │
│   FSPO-style per-token factuality weighting + VPS step rewards │
│   agent: claude/sonnet  |  ~50 turns  |  GPU optional         │
└─────────────────────────────────────────────────────────────────┘
         ↓

Phase 3 — Phase-5 Derisking (sequential, committed track)
┌─────────────────────────────────────────────────────────────────┐
│ exp1222: Phase 5-A — Minimal in-situ substrate prototype       │
│   50K param encoder + energy MLP + decoder; 100 5x5 puzzles   │
│   agent: claude/opus  |  ~60 turns  |  GPU 4-6h               │
│                                                                  │
│ exp1223: Phase 5-B — In-situ training loop [gated on 1222]    │
│   PCD with k=3 verifiers, 1000 queries, 5-failure-mode check  │
│   agent: claude/opus  |  ~60 turns  |  GPU 8-12h              │
│                                                                  │
│ exp1224: Phase 5-C — Adversarial probe [gated on 1223]        │
│   Single-verifier gaming + pairwise correlation + joint null   │
│   agent: claude/sonnet  |  ~40 turns  |  CPU                  │
└─────────────────────────────────────────────────────────────────┘
         ↓

Phase 4 — New Research (unconditional)
┌─────────────────────────────────────────────────────────────────┐
│ exp1225: LLMs Gaming Verifiers + Composite Rewards Defense     │
│   Single-verifier RLVR gaming vs k=5 AND-compose defense      │
│   agent: claude/sonnet  |  ~40 turns  |  CPU                  │
│                                                                  │
│ exp1226: Boltzmann-GPT Phase-3 Seed Integration                │
│   BoltzmannGPTLayer in continuous_ebm.py; AUROC vs NRGPT      │
│   agent: claude/sonnet  |  ~40 turns  |  CPU                  │
└─────────────────────────────────────────────────────────────────┘
         ↓

Phase 5 — WOPR + Retro
┌─────────────────────────────────────────────────────────────────┐
│ exp1227: WOPR Futoshiki Cartridge                              │
│   Inequality grid puzzle Ising EBM; E=0 at valid solution     │
│   agent: codex/gpt-5.5  |  ~30 turns  |  CPU                 │
│                                                                  │
│ exp1228: Milestone Retro [STEP 0 + claude/opus + 100 turns]   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase Descriptions

### Phase 0 — Infrastructure (exp1216–1217)

Two MANDATORY unconditional infrastructure fixes from known-issues.md (2026-05-03 entries).

**exp1216: Pre-commit staged_files_only + batching-check fix** (CRITICAL data-loss)

The `staged_files_only` plugin stashes unstaged changes, runs hooks, then restores the stash.
When the stash patch fails to apply (base files shifted), unstaged changes are permanently lost.
Fix: (a) disable `staged_files_only` or configure hooks to run on dirty tree; (b) add
`# batching-check: exempt-{reason}` marker to GRPO scripts so they pass the hook.

**exp1217: Auto-populate prior_failures from failure-ledger at plan time**

Write `scripts/conductor_priors_autofill.py`: reads `research-roadmap-next.yaml`, queries the
failure-ledger's matching_priors for each task, generates `prior_failures` stubs, writes back.
Prevents 7+ DOOMED_RERUN_BLOCK false-positives per milestone from tasks building on successful
upstreams that the ledger misclassifies as failures.

### Phase 1 — Paper (exp1218)

**exp1218: Paper v6 Related Work Overhaul** (MANDATORY for publication hold lift)

Add 5 bibliography entries, adopt new thesis sentence ("Open-source EXTERNALLY-GROUNDED EBM that
solves multimodal text collapse"), apply novelty-boundary discipline, insert positioning paragraph,
align with EBM-as-System-2 industry consensus. Source material at:
- `docs/research-notes/energy-based-llm-alternatives-deep-research-results.md`
- `docs/research-notes/paper-v5-decentralization-section-draft.md`

### Phase 2 — GRPO Diagnosis + Training (exp1219–1221)

**exp1219: GRPO v5 Regression Diagnosis** — Read exp1208 artifact, identify root cause of -35pp.
**exp1220: GRPO-VPS Full Training Run** — Validate +24pp from exp1209 in actual parameter updates.
**exp1221: GRPO v6 FSPO+VPS Combined** — Add FSPO per-token factuality weighting to VPS.

### Phase 3 — Phase-5 Derisking (exp1222–1224)

Per `openspec/change-proposals/in-situ-training-phase5-derisking.md`:
**exp1222: Phase 5-A** — 50K param prototype end-to-end on 100 5×5 puzzles.
**exp1223: Phase 5-B** — 1000-query in-situ training with k=3 verifier, 5 failure-mode checks.
**exp1224: Phase 5-C** — Adversarial probe: 3 attack classes per Q9 spec.

### Phase 4 — New Research (exp1225–1226)

**exp1225: LLMs Gaming Verifiers Defense** — Verify k=5 AND-composition resists verifier gaming
(arXiv 2604.15149 + arXiv 2509.15557). Compare single-verifier vs ensemble gaming resistance.

**exp1226: Boltzmann-GPT Phase-3 Seed** — Add BoltzmannGPTLayer to continuous_ebm.py,
compare energy ordering on FoVer traces vs NRGPT baseline (arXiv 2601.17094).

### Phase 5 — WOPR + Retro (exp1227–1228)

**exp1227: WOPR Futoshiki Cartridge** — Inequality constraint Ising EBM; E=0 at valid solution.
**exp1228: Milestone Retro** — STEP 0 + claude/opus + 100 turns; evaluates 13 criteria.

---

## Dependency Graph

```
exp1216 ──────────────────────────────────────────────────────►
exp1217 ──────────────────────────────────────────────────────►
exp1218 ──────────────────────────────────────────────────────►
exp1219 ──────────────────────────────────────────────────────► exp1220 ──► exp1221
exp1222 ──────────────────────────────────────────────────────► exp1223 ──► exp1224
exp1225 ──────────────────────────────────────────────────────►
exp1226 ──────────────────────────────────────────────────────►
exp1227 ──────────────────────────────────────────────────────►
exp1228 (runs last — retro)
```

---

## Hardware Requirements

| Experiment | GPU Required | Estimated GPU-hours |
|---|---|---|
| exp1216–1219 | No / Optional | 0–1 |
| exp1220 | Yes (MANDATORY DualGPU) | 4–6 |
| exp1221 | Optional (eval only) | 0–1 |
| exp1222 | Yes (4–6 GPU-hours) | 4–6 |
| exp1223 | Yes (8–12 GPU-hours) | 8–12 |
| exp1224–1228 | No | 0 |

Total GPU: ~16–26 GPU-hours on 2× RTX 3090.

---

## New arxiv Findings (2026-05-03 Scan)

Added to research-references.md:

1. **FSPO** (arXiv:2505.24630) — Factuality-Aware Step-wise Policy Optimization; per-token
   advantage weighting via step-wise factuality. Complements GRPO-VPS for exp1221.
2. **LLMs Gaming Verifiers** (arXiv:2604.15149) — RLVR models game rule-induction verifiers.
   Carnot k=5 AND-composition is proposed defense. Tested in exp1225.
3. **Verifiable Composite Rewards** (arXiv:2509.15557) — Composite reward with hacking penalties.
   Carnot Tier-0 cascade is natural composite structure. Combined with exp1225.
4. **Boltzmann-GPT** (arXiv:2601.17094) — EBM world model + GPT language generation bridge.
   Phase-3 seed candidate. Implemented in exp1226.
5. **Eidoku** (arXiv:2512.20664) — Neuro-symbolic CSP verification gate. Filed for .96+.
6. **Gradient Fingerprints** (arXiv:2604.16242) — Reward hacking detection. Filed for .96+.

---

## Success Criteria (13)

| # | Criterion | Measured by |
|---|---|---|
| 1 | pre_commit_data_loss_fixed | exp1216.precommit_fail_forward_enabled |
| 2 | prior_failures_autofill_script_shipped | exp1217.autofill_script_exists |
| 3 | paper_v6_related_work_complete | exp1218.all_5_citations_added |
| 4 | grpo_v5_regression_diagnosed | exp1219.diagnosis_complete |
| 5 | grpo_vps_training_result | exp1220 artifact exists (any honest verdict) |
| 6 | grpo_v6_fspo_delta_measured | exp1221.grpo_v6_fspo_delta_measured |
| 7 | phase5a_prototype_ready | exp1222.phase5a_prototype_ready |
| 8 | phase5b_stability_confirmed | exp1223.phase5b_stability_confirmed |
| 9 | phase5c_adversarial_probe_complete | exp1224.adversarial_probe_complete |
| 10 | gaming_verifiers_defense_measured | exp1225.gaming_defense_measured |
| 11 | boltzmann_gpt_auroc_measured | exp1226.boltzmann_gpt_auroc_measured |
| 12 | futoshiki_cartridge_shipped | exp1227.futoshiki_cartridge_shipped |
| 13 | retro_complete | exp1228.retro_complete |
