# Research Roadmap — Milestone 2026.04.79
## Pre-Test Surgery + Cascade Validation + Self-Learning Deployment + FoVer Expansion

**CalVer:** 2026.04.79
**Date:** 2026-04-28
**Experiments:** 1013–1025 (13 experiments)
**Planned wall time:** ~525 min

---

## What Milestone .78 Proved

Milestone 2026.04.78 completed with 3/12 criteria met. Two successes were genuine:
**SC-Energy deployed as production Tier 2** (Exp 1001, tier2_wired=True) and
**gate schema repaired** (Exp 1000, gate_schema_repaired=True). But a **critical new
failure mode** appeared: "comparison failed" pre-test errors blocked 4 experiments that
never ran (Exp 1004, 1009, 1010, 1012 — including the retro). Additionally, 4 experiments
(Exps 1006, 1007, 1008, 1011) were DOOMED_RERUN_BLOCKed because the planning agent forgot
to add `prior_failures` YAML fields for experiments with matching historical scope.

### Bright Spots from .78
- **Exp 1001 (SC-Energy Tier 2 v4):** test_failures_after=0, tier2_wired=True. SC-Energy
  is now production Tier 2. Three pre-existing test failures (from Exp 969/.75) diagnosed
  and fixed.
- **Exp 1002 (DualGPU v5):** wired synthetically — DualGPU path now exists in
  VerifyRepairPipeline. Live activation not yet confirmed (13th consecutive idle milestone).
- **Exp 1003 (SpilledEnergy live):** Pipeline reached live GPU. AUROC=0.5 (below threshold),
  NUP probe AUROC=0.611. 9 violations collected (gate needed 10). Root cause: SOTA models
  rarely violate on standard GSM8K — the probe needs harder problems or a synthetic injection
  approach to gather enough violations.

### Root Causes to Address
1. **"Comparison failed" pre-test failure** — Unknown root cause, blocked 4 experiments in .78.
   Exp 1001 fixed SC-Energy test failures; this may have changed pipeline behavior such that
   an existing test's expected output no longer matches the new SC-Energy Tier 2 output. The
   preflight must find and fix the specific failing test(s) before any other experiment runs.
2. **DOOMED_RERUN_BLOCK for new .78 experiments** — Exps 1006/1007/1008/1011 lacked
   `prior_failures` fields in the YAML. The conductor correctly blocked them (found 4-7
   prior matching-scope experiments with failures). For .79, every task with historical
   scope overlap MUST include `prior_failures` entries.
3. **SpilledEnergy violation shortage** — 9 live violations < 10 gate threshold. The FoVer
   corpus expansion (Z3 auto-labeling) provides 500+ labeled pairs including many violations,
   making the 10-violation gate easily passable for PPSEBM.
4. **Legacy carryovers still running** — Exps 786, 641, 906 appeared for the 19th consecutive
   milestone. CLAUDE.md mandatory gate requires retirement. Preflight must add these to the
   exclusion manifest.

### What Was Carried Forward (unfinished from .78)
- Triple Integration E2E (Exp 1004 scope) — pre-test comparison failures
- Energy-Selection SSD (Exp 1006 scope) — DOOMED_RERUN_BLOCK (missing prior_failures)
- ThinkPRM Step Verifier (Exp 1007 scope) — DOOMED_RERUN_BLOCK (missing prior_failures)
- VPRM Rule-Based v3 (Exp 1008 scope) — DOOMED_RERUN_BLOCK (missing prior_failures)
- GS-KAN Energy Tier (Exp 1009 scope) — pre-test comparison failures
- KV260 First Light (Exp 1010 scope) — pre-test comparison failures
- ALMC-ODE Annealed Sampler (Exp 1011 scope) — DOOMED_RERUN_BLOCK (missing prior_failures)
- PPSEBM Live Relay (Exp 1005 scope) — gate failure (9 violations < 10)
- .78 Milestone Retro (Exp 1012) — never ran (pre-test comparison failures)

---

## Architecture Diagram

```
Live GPU Inference
        |
        v
[ThinkPRM Tier 0a]  [SpilledEnergy Tier 0b]  [NUP Probe Tier 0c]
        |                      |                      |
        +----------------------+----------------------+
                               |
                  [SC-Energy Tier 2] (deployed, Exp 1001)
                               |                  |
                    [Ising Sampler Tier 3]    [DualGPU Runner]
                         |            |
                  [KAEMEnergy]   [GS-KAN Energy] <-- .79 adds RM/BOP/NABS metrics
                                     |
                           [KV260 FPGA] <-- .79 first light
                                     |
                      [ALMC-ODE Annealed Sampler] <-- .79 adds Mpemba init
                                     |
               [PPSEBM Cross-Session Memory] (FR-11 relay)
                                     |
                  [Energy-Selection SSD] <-- .79 mandatory FR-11 closure
                                     |
           [FoVer Corpus 500+ pairs] <-- .79 expands from 57 pairs
```

---

## Phase Descriptions

### Phase 0 — Pre-Test Surgery and Manifest Retirement (1 exp)

**Exp 1013: Preflight v29 — Pre-Test Surgery + Manifest Retirement + .78 Retro**

This preflight has three mandated tasks:
1. Find and fix the "comparison failed" pre-test failure that blocked 4 experiments in .78.
   Run `pytest tests/python/ -x -v` to locate the specific failing assertion, identify the
   root cause (likely a pipeline-output snapshot test broken by SC-Energy Tier 2 wiring in
   Exp 1001), and fix it.
2. Retire Exps 786, 641, and 906 to the exclusion manifest (`ops/exclusion_manifest.yaml` and
   `scripts/conductor_exclusion_manifest.json`). These have run 19, 9, and 6 consecutive
   milestones respectively — all past CLAUDE.md mandatory gate thresholds.
3. Write the .78 retro summary (Exp 1012 scope was never run) — capture verdict distributions,
   successes, and biggest_gaps_79 in the preflight artifact.

### Phase 1 — Cascade Validation (2 exps)

**Exp 1014: Triple Integration E2E v5**

The full 3-tier cascade (ThinkPRM Tier 0a → SpilledEnergy Tier 0b → SC-Energy Tier 2 →
Ising) has never been validated end-to-end. SC-Energy is deployed. The pre-test failure
that blocked .78 will be fixed by Exp 1013. This experiment validates that the cascade
works on 50 synthetic questions and that skip rates are non-zero for all tiers.

**Exp 1015: Energy-Selection SSD v2 (MANDATORY FR-11 self-learning)**

This is the mandatory self-learning experiment for milestone .79. Energy-Selection SSD
implements FR-11 Tier 2 (cross-session learning) by using Carnot's energy function as the
selection filter for self-distillation: keep outputs where energy < threshold (high-confidence
correct), train on them. Inspired by both SSD (arXiv 2604.01193) and Self-Distilled Reasoner
(arXiv 2601.18734). The comparison baseline uses temperature-based filtering (pure SSD).
The Carnot variant uses energy-threshold filtering. A combined variant uses both energy
AND FoVer step labels for conditioning. Measures which filter gives best self-learning signal
on the FoVer corpus.

### Phase 2 — FoVer Corpus Expansion (1 exp)

**Exp 1016: FoVer Corpus Expansion — Z3 Auto-Label 500+ CoT Pairs**

The FoVer corpus has 57 labeled pairs (from Exp 442, live GPU). All probe training
(ThinkPRM, VPRM, GS-KAN) uses this tiny corpus. This experiment uses Z3-based VeriCoT
labeling (already implemented in Exp 453/564) to automatically generate and label 500+
CoT steps from synthetic GSM8K-style arithmetic problems. Per arXiv 2505.15960, Z3 labels
outperform human labels for PRM generalization. The expanded corpus enables better probe
training in Exps 1017-1019. Also provides the 50+ violation pairs needed to unblock PPSEBM.

### Phase 3 — Probe Training on Expanded Corpus (3 exps)

**Exp 1017: ThinkPRM Step Verifier v3**

Train ThinkPRMProbe (implemented in .77 Exp 945) on the 500-pair expanded FoVer corpus
(gated on Exp 1016). Split 400 train / 100 test. Compare AUC to zero-shot CarnotThinkProbe
baseline and to the 57-pair trained version. Hypothesis: 10x more data gives 10+ AUC points.

**Exp 1018: VPRM Rule-Based Step Verifier v3**

Extend VPRMArithmeticVerifier to 10+ rule families (arithmetic + logical entailment +
comparative transitivity + order-of-magnitude + unit consistency + 4 new). Evaluate step-level
F1 on the expanded 500-pair corpus. Compare to outcome-level F1. Hypothesis: step-level F1
exceeds outcome-level by 15+ points on diverse violations.

**Exp 1019: GS-KAN Energy Tier v3 + Hardware FPGA Analysis**

Implement GS-KAN (arXiv 2512.09084) with proper `prior_failures` documentation. Add
hardware complexity analysis using RM/BOP/NABS metrics from arXiv 2604.03345. Train on
500-pair corpus. Target: AUROC ≥ standard KAN baseline - 0.02, RM < 20% of KAEMEnergy RM,
estimated LUT < 60K for KV260 budget fit.

### Phase 4 — Sampler + Training Improvements (2 exps)

**Exp 1020: ALMC-ODE Annealed Sampler v2 + Mpemba Initialization**

Implement ALMCODESampler (arXiv 2604.20052) with Mpemba-optimized initialization from
arXiv 2603.24183. Mpemba init: start from the mean-field equilibrium of the energy function
rather than random spin states. Compare convergence speed vs random-init ALMC vs LSB baseline
(Exp 983). Target: multimodal escape rate > LSB, convergence steps < 50% of random-init.

**Exp 1021: NK-KAEMEnergy + Multilevel KAN Training**

Implement Newton-Kaczmarz optimizer (arXiv 2512.18921) with multilevel spline grid training
(arXiv 2603.04827) for KAEMEnergy.fit(). Grid schedule: G=4 → G=8 → G=16, three levels.
Compare training wall time and AUC vs Adam single-level baseline (Exp 936 baseline). Target:
3x convergence speedup with no AUC regression on 500-pair corpus.

### Phase 5 — Hardware and Infrastructure (3 exps)

**Exp 1022: KV260 First Light v5 — SD Re-Flash + USB Serial Approach**

New approach targeting the SSH blockade. Prior approaches tried SSH directly; this experiment:
1. Investigates USB serial console path (minicom to board UART)
2. Investigates SD card re-flash (download Kria Ubuntu 22.04 image, flash to SD,
   boot with default-on SSH and known credentials)
3. If all automated paths fail, writes a precise 5-command human action guide for
   enabling SSH in one session

**Exp 1023: DualGPU Live Inference v2 — Activate Live GPU Path**

DualGPU has been wired synthetically for 13 consecutive milestones. This experiment
forces live GPU activation: runs a 10-question benchmark with CARNOT_FORCE_LIVE=1 and
two real models loaded (one per GPU). Measures actual throughput_ratio on live inference,
not synthetic validation. Gate: requires CARNOT_FORCE_LIVE=1 and torch.cuda.device_count()>=2.
If GPUs are unavailable, writes honest blocked verdict with clear diagnostic.

**Exp 1024: PPSEBM Live Relay v4 — FR-11 Relay on Expanded FoVer Violations**

PPSEBM blocked in .78 (9 violations < 10 gate). Exp 1016 provides 500+ labeled pairs
including 100+ violations. Exp 1024 runs PPSEBM on the expanded violation set. Gate:
n_fover_violations >= 50 (from Exp 1016), not on live GPU violations. This decouples
the FR-11 relay from the live GPU availability problem that plagued .75-.78.

### Phase 6 — Retrospective (1 exp)

**Exp 1025: Milestone 2026.04.79 Retrospective**

Covers BOTH .78 (Exp 1012 never ran) and .79 results. Evaluates all 13 success criteria.
Updates ops/status.md and ops/changelog.md.

---

## Success Criteria

| # | Criterion | Target Experiment | Gate |
|---|-----------|-------------------|------|
| 1 | Pre-test comparison failure fixed | Exp 1013 | pytest 0 failures |
| 2 | Exps 786/641/906 retired to manifest | Exp 1013 | manifest entries confirmed |
| 3 | .78 retro coverage written | Exp 1013 | 9 .78 exps evaluated |
| 4 | Triple Integration cascade validated E2E | Exp 1014 | cascade_validated=True |
| 5 | Energy-Selection SSD self-learning loop closed | Exp 1015 | self_learning_closed=True |
| 6 | FoVer corpus expanded to 500+ pairs | Exp 1016 | n_labeled_pairs >= 500 |
| 7 | GS-KAN FPGA hardware budget confirmed | Exp 1019 | kv260_budget_fit=True |
| 8 | ALMC-ODE multimodal escape > LSB | Exp 1020 | bimodal_escape_rate_almc > lsb |
| 9 | KV260 first light achieved OR guide written | Exp 1022 | hardware_working OR human_action_required |
| 10 | PPSEBM FR-11 relay on real violations | Exp 1024 | live_relay_confirmed=True |
| 11 | Retrospective written, ops/ updated | Exp 1025 | retro artifact present |

---

## Dependency Graph

```
Exp 1013 (preflight — pre-test fix + manifest retirement)
    |
    +---> Exp 1014 (triple integration v5)
    |
    +---> Exp 1015 (energy-selection SSD v2) [mandatory self-learning]
    |
    +---> Exp 1016 (fover corpus expansion)
              |
              +---> Exp 1017 (thinkprm v3, gated on 500+ pairs)
              |
              +---> Exp 1018 (vprm v3, gated on 500+ pairs)
              |
              +---> Exp 1019 (gskan v3, uses expanded corpus)
              |
              +---> Exp 1024 (ppsebm v4, gated on n_violations >= 50)

Exp 1020 (almc-ode v2) [independent]
Exp 1021 (nk-kaem + multilevel) [independent]
Exp 1022 (kv260 first light v5) [independent]
Exp 1023 (dualgpu live v2) [independent]

Exp 1025 (retro) [depends on all]
```

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|-------------|-------|
| Exp 1023 | Yes (dual GPU) | CARNOT_FORCE_LIVE=1, sg render -c |
| All others | No (CPU) | Synthetic or saved corpus |
| Exp 1022 | No (FPGA target) | SCP to 192.168.51.98 or USB serial |

---

## New Papers Incorporated

| Paper | ArXiv | Incorporated In |
|-------|-------|----------------|
| Multilevel KAN Training | 2603.04827 | Exp 1021 |
| Hardware-Oriented KAN Complexity | 2604.03345 | Exp 1019 |
| Mpemba-Optimized Thermodynamic Init | 2603.24183 | Exp 1020 |
| Self-Distilled Reasoner | 2601.18734 | Exp 1015 |
| KAN Universality Theorem | 2604.23765 | Background reference |
| LagONN Lagrange Constraint Satisfaction | 2505.07179 | Filed for .80 |
