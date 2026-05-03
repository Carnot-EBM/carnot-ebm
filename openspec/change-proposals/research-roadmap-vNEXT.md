# Milestone 2026.04.93 — Design Document

**Milestone:** 2026.04.93  
**Title:** Paper Critical Fixes + GRPO v5 GPU Unblock + Phase 4 Real Puzzles + GRPO-VPS  
**Estimated wall time:** ~380 min  
**Experiments:** exp1191–exp1202 (12 total)  
**Planned:** 2026-05-03

---

## What Milestone 2026.04.92 Proved

**10/13 criteria met.** Three failed:

| Experiment | Verdict | Root cause |
|---|---|---|
| exp1179 | MISSING (no artifact) | llama.cpp GPU offload install timed out at 60 min |
| exp1180 | MISSING (no artifact) | Critical paper fix timed out at 55 turns |
| exp1183 | audit_failures_remain | Gated on exp1180; never ran cleanly |

**Key retirements from .92:**
- **k=6 AND-compose retired** — even after regularization, k6 AUROC=0.9027 vs k5=0.9240; structural ceiling confirmed
- **DoT EBM-diffusion retired** — AUROC=0.5 at all temperatures; per-token energy gradient is flat
- **Latent-GRPO masking** — no_delta on current FoVer corpus (too few trivially-easy samples)
- **Phase 4 vs synthetic BFS** — phase4_tied_with_bfs; all energy traces [0.0,...] (trivial puzzles)

**Key successes from .92:**
- Pytest memory watchdog active (exp1178)
- SC-Energy overfit resolved; v2 model has holdout AUROC (regularization working)
- Paper high+medium/low integrity issues resolved (13/18 done; 5 critical remain)
- Hex WOPR cartridge operational (random_vs_gibbs_win_rate=0.9)
- Phase 4 BFS baseline established (honest negative)

---

## Three Biggest Gaps Entering .93

### Gap 1: Publication hold still active
Five critical issues (ISSUE-1 through ISSUE-5) remain unresolved after exp1180 timed out. The arXiv hold cannot be lifted until all 18 issues are addressed and operator approval is given.

### Gap 2: GRPO v5 has never run on real GPU hardware
Four consecutive attempts blocked by toolchain issues (routing bug, llama.cpp CPU-only). Self-learning trajectory stalled at v4's +10pp. GRPO v5 with TinyV false-negative correction has never been empirically tested.

### Gap 3: Phase 4 puzzles are trivially solvable
exp1189: all 20 synthetic puzzles were BFS-tractable, energy traces all [0.0,...] (puzzles pre-solved, Blocked Gibbs had nothing to do). Phase 4 cannot claim a contribution until it beats BFS on genuinely hard problems where BFS hits the 100k-state intractability cap.

---

## Architecture Diagram

```
Phase 0 (Infrastructure)
  exp1191: prlimit memory cap [conftest.py resource.setrlimit]
  exp1192: llama.cpp GPU offload v2 [CUDA wheel + verify throughput]
      |
      v gates exp1195

Phase 1 (Paper Integrity)
  exp1193: Critical ISSUE-1 to ISSUE-5 [opus, retry of exp1180]
      |
      v gates exp1194
  exp1194: Paper v6 bundle v7 [gated on exp1193 + exp1181 done]

Phase 2 (Self-Learning MANDATORY)
  exp1195: GRPO v5 + TinyV v2 [DualGPU MANDATORY, gated on exp1192]
  exp1196: GRPO-VPS step-level [step verifiers as per-segment reward]

Phase 3 (Research)
  exp1197: Phase 4 harder puzzles [BFS-intractable 15x15+, prior_failures:exp1189]
  exp1198: FoVer expansion v7 [hard negatives, SOTA GGUF]
  exp1199: KANtize SOS-KAN 4-bit [edge deployment prep, arXiv 2603.17230]

Phase 4 (Self-Learning Tier 1)
  exp1200: Online constraint reweighting v2 [constraint ADDITION from memory]

Phase 5 (WOPR + Retro)
  exp1201: WOPR Nonogram cartridge
  exp1202: Milestone retro
```

---

## Phase Descriptions

### Phase 0 — Infrastructure (2 tasks, unconditional, MANDATORY FIRST)

**exp1191: prlimit/resource.setrlimit memory cap**
The exp1178 watchdog detects leaks post-test but cannot prevent a single-test catastrophic load.
Six occurrences in 5 hours on 2026-05-02 (17:18, 19:33, 20:04, 21:18, 21:33, 22:48).
Fix: `resource.setrlimit(RLIMIT_AS, 8GB)` in conftest.py — caps any single pytest process.
MANDATORY per known-issues.md (NEW 2026-05-02 22:50Z).

**exp1192: llama.cpp GPU offload fix v2**
exp1179 produced a MISSING artifact — pip install timed out at 60 min. This retry uses
the pre-built binary wheel from abetlen GitHub releases (faster) and falls back to source.
Once verified >=50 tok/s with n_gpu_layers active, emits `llama_cpp_gpu_offload_verified=true`
which unblocks exp1195 (GRPO v5).

### Phase 1 — Paper Integrity (2 tasks)

**exp1193: Paper v5 critical ISSUE-1 to ISSUE-5 (retry of exp1180)**
exp1180 produced MISSING artifact at 55 turns. This retry gets 60 turns and starts with
the minimal fix for ISSUE-1 (drop fig3 if honest speedup <2x per exp1094 data) rather
than full figure rewrite. Issues 2-5 are text changes (<5 turns each).
Ships `scripts/figure_integrity_audit.py`.

**exp1194: Paper v6 recompile + arXiv bundle v7**
Gated on exp1193 (critical_issues_fixed=5) AND exp1181 (high_severity_fixed=5, done in .92).
Runs tectonic, runs both audit scripts, builds `carnot-arxiv-v7.tar.gz`.
Publication hold remains active until operator lifts it explicitly.

### Phase 2 — Self-Learning MANDATORY (2 tasks)

**exp1195: GRPO v5 + TinyV v2 DualGPU**
DualGPU MANDATORY. Gated on exp1192.llama_cpp_gpu_offload_verified=true.
Four prior failures addressed: (1) exp1184 blocked on GPU offload, fixed by exp1192;
(2) exp1173 routing bug to codex; (3) exp1159 v4 baseline (+10pp floor);
(4) exp1146 v3 baseline (+2.86pp).
First clean GRPO v5 attempt with: TinyV confidence abstention (thresh_low=0.3, thresh_high=0.7),
structural warm-up (300s pure r_reflect), tensor_split=[0.5, 0.5] across both RTX 3090s.

**exp1196: GRPO-VPS step-level process supervision**
arXiv 2604.20659: compute per-step reward as change in model's belief at each CoT step boundary.
Wire CausalReasoningVerifier + Z3MathVerifier as segment-level GRPO rewards.
No GPU needed for scoring — verifiers run on CPU. SOTA GGUF for generation only.
Up to 2.6pp improvement + 13.7% shorter reasoning chains in the original paper.
Orthogonal to TinyV (addresses credit assignment, not false-negative correction).

### Phase 3 — Research (3 tasks)

**exp1197: Phase 4 harder ARC-AGI-3 puzzles**
Prior failure exp1189: phase4_tied_with_bfs on trivial synthetic puzzles.
Root cause: energy traces [0.0,...] — puzzles pre-solved, Gibbs sampler idle.
This retry generates 15x15+ grids where BFS must explore >100k states before failing.
Phase 4 energy minimization's advantage is exactly this intractable-search regime.
prior_failures: exp1189 (verdict=phase4_tied_with_bfs).

**exp1198: FoVer expansion v7 hard negatives**
Standard FoVer corpus expansion using Qwen3.6-35B-A3B-GGUF + gemma-4-31B-it-GGUF.
Focus on hard negatives (samples near energy decision boundary, confidence 0.4-0.6)
to improve verifier ensemble discriminative power.
Target: >=500 new labeled CoT pairs. Supports k=5 re-evaluation with expanded holdout.

**exp1199: KANtize SOS-KAN 4-bit quantization**
arXiv 2603.17230. Apply 4-bit and 8-bit quantization to SOS-KAN verifier (AUROC=0.9902).
Target: AUROC >0.97 at 4-bit (< 2% regression).
If successful: ~2.2MB model — deployable on AMD XDNA NPU, Intel AI Boost, Apple Neural Engine.
This is the sovereignty anchor: constraint verification on consumer hardware.

### Phase 4 — Self-Learning Tier 1 (1 task)

**exp1200: Online constraint reweighting v2 with constraint ADDITION**
From research-program.md Tier 1. Key insight missed in Exp 134: reweighting doesn't improve
accuracy because the constraint set is fixed. Fix: ADD new constraints from memory patterns.
When memory tracker detects "arithmetic carry errors on 60% of wrong responses," ADD a carry-check
constraint to the active set — don't just upweight existing arithmetic constraints.
Uses ConstraintStateMachine from Exp 125 as the memory layer.

### Phase 5 — WOPR + Retro (2 tasks)

**exp1201: WOPR Nonogram cartridge**
Nonogram (Picross) puzzle: binary grid, row/column count constraints.
Each row/column specifies length and order of runs of filled cells.
Energy = Σ_row (constraint_violation²) + Σ_col (constraint_violation²). E=0 at valid solution.

**exp1202: Milestone 2026.04.93 retrospective**
Standard operational retrospective. Criteria pass/fail, slowest-5, DualGPU utilization,
publication hold status update.

---

## Dependency Graph

```
exp1191 ─── (unconditional)
exp1192 ─── (unconditional) ──→ gates exp1195
exp1193 ─── (unconditional) ──→ gates exp1194
exp1194 ─── gated on exp1193.critical_issues_fixed==5
exp1195 ─── gated on exp1192.llama_cpp_gpu_offload_verified==true
exp1196 ─── (standalone)
exp1197 ─── (standalone, prior_failures:exp1189)
exp1198 ─── (standalone)
exp1199 ─── (standalone)
exp1200 ─── (standalone)
exp1201 ─── (standalone)
exp1202 ─── (runs last)
```

---

## Hardware Requirements

- **DualGPU MANDATORY:** exp1195 (35B model requires tensor_split across both RTX 3090s)
- **GPU optional:** exp1198 (GGUF inference; single GPU acceptable)
- **CPU only:** exp1191, exp1193, exp1194, exp1196, exp1199, exp1200, exp1201, exp1202
- **GPU probe:** exp1192 (must verify >=50 tok/s throughput with n_gpu_layers active)

---

## Success Criteria (12)

| # | Criterion | Experiment |
|---|---|---|
| 1 | prlimit_memory_cap_active | exp1191 |
| 2 | llama_cpp_gpu_offload_verified | exp1192 — gates exp1195 |
| 3 | critical_issues_fixed_5_of_5 | exp1193 — gates exp1194 |
| 4 | arxiv_bundle_v7_ready | exp1194 |
| 5 | grpo_v5_honest_result | exp1195 — DualGPU MANDATORY |
| 6 | grpo_vps_step_delta_measured | exp1196 |
| 7 | phase4_bfs_intractable_fraction_above_50pct | exp1197 |
| 8 | fover_v7_pairs_above_500 | exp1198 |
| 9 | kantize_auroc_maintained_above_0p97 | exp1199 |
| 10 | tier1_online_addition_honest_verdict | exp1200 |
| 11 | nonogram_cartridge_shipped | exp1201 |
| 12 | retro_complete | exp1202 |

---

## Key Architectural Decisions

- **exp1192 gates exp1195:** No GRPO v5 without confirmed GPU offload. Prevents a fifth wasted attempt.
- **exp1193 uses opus with max_turns:60:** exp1180 failed at 55 turns. 5 extra turns + minimal-first approach (drop fig3 if honest speedup <2x) should clear the critical blocker.
- **k=6 AND-compose officially retired:** exp1185 confirmed regularization resolves overfit but k=6 AUROC (0.903) < k=5 (0.924). No further k=6 experiments.
- **DoT retired:** exp1186 confirmed. No further DoT experiments without LaDiR-style latent redesign (arXiv 2510.04573).
- **No gemini agent_type:** Still 429-rate-limited since milestone .84.
- **Latent-GRPO deprioritized:** exp1187 no_delta. Will not reappear until invalid-sample rate >10% confirmed.
- **Spurious rewards context (arXiv 2506.10947):** 73% of GRPO gain is structural. GRPO v5 must beat v4 by >3pp to confirm energy reward adds real signal beyond structure.

---

## Prior Milestone Quick Reference

| Milestone | Criteria | Key win | Key miss |
|---|---|---|---|
| .90 | 12/13 | Phase 3/4 sampler chain complete | arXiv on hold |
| .91 | 11/13 | GRPO v4 +10pp, Phase 4 pilot 74.7% | GRPO v5 routing bug |
| .92 | 10/13 | Hex operational, 13/18 paper issues | exp1179/1180 MISSING |
