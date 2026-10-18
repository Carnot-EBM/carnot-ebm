# Research Roadmap — Milestone 2026.04.78
## Gate Schema Repair + .77 Carry-Forward Recovery + ThinkPRM + VPRM + ALMC-ODE

**CalVer:** 2026.04.78
**Date:** 2026-04-28
**Experiments:** 1000–1012 (13 experiments)
**Planned wall time:** ~550 min

---

## What Milestone .77 Proved

Milestone 2026.04.77 completed with 3/10 criteria met. The dominant failure mode was the
**third consecutive gate schema cascade block**: Exp 987 (EnvPropagationGuard) succeeded
functionally — state file written, subprocess propagation confirmed, RETRO-015 closed — but
the artifact field was named `subprocess_propagation_ok` while downstream gates checked
`env_propagation_persistent`. All 7 dependent experiments (criteria 2–6, 9, plus Exp 994)
received `None` from the gate check and were blocked.

### Bright Spots from .77
- **Exp 992 (KAN MILP fix):** 11 violations (7 monotonicity + 4 boundary) eliminated via
  isotonic projection in `enforce_monotonicity()`. 1.89x speedup after fix. This experiment
  had no upstream gates — it completed because it was fully independent.
- **Exp 993 (KV260 IP):** Board discovered at 192.168.51.98. Bitstream confirmed ready at
  `output/carnot_ising_v4_bd/carnot_ising.bit`. SSH/SCP from conductor host blocked — human
  key exchange required for first light.
- **Exp 995 (PCIB Tier 0f):** Clean negative. Text-statistical PCIB proxy AUROC=0.532 vs
  NUP probe AUROC=0.964. PCIB needs LLM logit access to be competitive. Research question
  closed.

### Root Causes to Address
1. **Gate schema contract** — upstream artifact field names must match gate config exactly.
   Fix: preflight patches Exp 987 artifact to add `env_propagation_persistent: true` since
   the experiment actually succeeded.
2. **KV260 SSH** — human must exchange SSH keys. Document exact steps in Exp 1010.
3. **Planner prior_failures discipline** — Exp 996 (GS-KAN) and Exp 997 (NK optimizer)
   blocked because `prior_failures` fields were missing. Fixed in this roadmap.

### What Was Carried Forward (unfinished business)
- SC-Energy Tier 2 deployment (Exp 988 scope) — gate schema fix required first
- DualGPU fresh timestamp validation (Exp 989 scope)
- Triple Integration E2E (Exp 990 scope)
- SpilledEnergy live GPU AUROC >= 0.70 (Exp 991 scope)
- PPSEBM live relay — FR-11 44 milestones without real-data relay (Exp 994 scope)

---

## Architecture Diagram

```
Live GPU Inference
        |
        v
[SpilledEnergy Tier 0b]  [NUP Probe Tier 0c]   [ThinkPRM Tier 0a]
        |                        |                       |
        +------------------------+-----------------------+
                                 |
                    [SC-Energy Tier 2] (NEW — .78 deploys)
                                 |
                      [Ising Sampler Tier 3]
                         |            |
                  [KAEM Energy]  [GS-KAN Energy]
                                 |
                        [KV260 FPGA] (pending SSH)
                                 |
                     [ALMC-ODE Annealed Sampler]
                                 |
              [PPSEBM Cross-Session Memory] (FR-11 relay)
                                 |
                    [Energy-Selection SSD] (FR-11 self-learning)
```

---

## Phase Descriptions

### Phase 0 — Gate Schema Repair Preflight (1 exp)

**Exp 1000: Preflight v28 — Gate Schema Repair + Manifest Check**

The core task: patch `results/experiment_987_env_propagation_guard_v2.json` to add
`env_propagation_persistent: true`. This is justified because Exp 987 succeeded functionally
— the state file was written, subprocess propagation confirmed, RETRO-015 closed. The wrong
field name was a documentation error, not a functional failure.

Also: verify exclusion manifest entries for legacy carryovers (786/627/603/641).

**Deliverable:** `results/experiment_1000_preflight_v28.json`
with `gate_schema_repaired: true`

---

### Phase 1 — .77 Carry-Forward Recovery (4 exps, all gated on Phase 0)

All four experiments were designed in .77, succeeded in design, and failed only because
of the gate cascade. They are replayed here with identical scope but gated on Exp 1000.

**Exp 1001: SC-Energy Tier 2 v4 — Fix 3 Test Failures + Wire Production**
- Fix the 3 pre-existing test failures blocking SC-Energy deployment (Exp 969 root cause)
- Wire SC-Energy as Tier 2 in `VerifyRepairPipeline`
- Gated on: `exp1000.gate_schema_repaired == true`

**Exp 1002: DualGPU Pipeline v5 — Fresh Timestamp (13th milestone, 720 min foregone)**
- Wire DualGPURunner into VerifyRepairPipeline
- Produce fresh `run_date` timestamp — "deliverable already exists" not acceptable
- Gated on: `exp1000.gate_schema_repaired == true`

**Exp 1003: SpilledEnergy + NUP Probe Live GPU v4**
- Validate SpilledEnergy (AUROC=1.0 on CPU synthetic) on real live GPU CoT
- Collect violations for Exp 1005 PPSEBM relay
- Requires GPU: `CARNOT_FORCE_LIVE=1`, `sg render -c '...'`
- Gated on: `exp1000.gate_schema_repaired == true`

**Exp 1004: Triple Integration E2E v4 — SC-Energy + SpilledEnergy + ThinkPRM Cascade**
- Validate full 4-tier cascade end-to-end after SC-Energy is wired (Exp 1001)
- Gated on: `exp1001.tier2_wired == true`

---

### Phase 2 — Self-Learning / FR-11 (2 exps)

**Exp 1005: PPSEBM Live GPU Relay v3 — Cross-Session Memory on Real Violations (FR-11)**
- Uses live violations collected by Exp 1003
- FR-11 relay: last confirmed in Milestone .33 (44 milestones ago — JEPA AUC 0.457→0.571)
- Target: sessions_with_new_templates >= 3 on real data
- Gated on: `exp1003.n_live_violations_collected >= 10`

**Exp 1006: Energy-Selection SSD — FR-11 Continuous Self-Improvement (MANDATORY)**
- Implements the "FR-11 + Energy-Selection SSD opportunity" from research-references.md
- SSD (Apple, arXiv 2604.01193): self-distillation using energy function as selection filter
  instead of temperature/truncation. Carnot's energy marks high-confidence-correct outputs;
  use those as training signal for generator self-improvement.
- This is the mandatory self-learning experiment for the milestone.
- CPU-only, no gate.

---

### Phase 3 — New Research (3 exps)

**Exp 1007: ThinkPRM Step Verifier (arXiv 2504.16828)**
- Train a CoT-reasoning step verifier on the 57-pair FoVer corpus using ThinkPRM's
  generation-based verification approach.
- Compare to CarnotThinkProbe zero-shot baseline (Tier 0a).
- Headline result to aim for: trained ThinkPRM probe beats zero-shot on FoVer AUC.
- CPU-only, no gate.

**Exp 1008: VPRM Rule-Based Step Verifier (arXiv 2601.17223)**
- Extend VPRMArithmeticVerifier (Exp 454, F1=1.0) to 6+ rule families.
- Evaluate on 57-pair FoVer per-step labels.
- Measure step-level F1 vs outcome-only verification.
- CPU-only, no gate.

**Exp 1009: GS-KAN Energy Tier v2 — Parameter-Efficient KAN for KV260 Budget**
- Re-attempt Exp 996 (GS-KAN) with proper `prior_failures` field.
- Implement shared-basis KAN (arXiv 2512.09084) for KV260 BRAM budget.
- Target: param_reduction > 50%, AUROC >= standard KAN - 0.02.
- CPU-only, no gate. Has `prior_failures` entry.

---

### Phase 4 — Hardware + Sampling Infrastructure (2 exps)

**Exp 1010: KV260 First Light v4 — Alternative SSH Discovery + Human Action Guide**
- Attempt alternative SSH paths: USB UART (`/dev/ttyUSB*`), manual IP (192.168.51.98)
- If SSH still fails: write exact human action guide for key exchange
- CPU/hardware, no gate.

**Exp 1011: ALMC-ODE Annealed Sampler (arXiv 2604.20052)**
- Add annealed temperature schedule to LSB sampler (Exp 983 default)
- Compare convergence on multi-modal constraint problems using 57-pair FoVer corpus
- Target: AUC improvement vs flat-temperature LSB
- CPU-only, no gate.

---

### Phase 5 — Retrospective (1 exp)

**Exp 1012: Milestone Retro 2026.04.78**
- Standard milestone retrospective
- Update ops/status.md and ops/changelog.md
- Identify 3 biggest gaps for .79

---

## Success Criteria

| # | Criterion | Experiment | Target |
|---|-----------|-----------|--------|
| 1 | Gate schema repaired (env_propagation_persistent written) | 1000 | gate_schema_repaired=True |
| 2 | SC-Energy test failures fixed (0 remaining) | 1001 | test_failures_after == 0 |
| 3 | SC-Energy wired as production Tier 2 | 1001 | tier2_wired=True |
| 4 | DualGPU wiring confirmed with fresh .78-era timestamp | 1002 | run_date matches 2026-04-28+ |
| 5 | Triple Integration cascade validated E2E | 1004 | cascade_validated=True |
| 6 | SpilledEnergy AUROC >= 0.70 on live GPU | 1003 | spilled_energy_live_auroc >= 0.70 |
| 7 | PPSEBM live relay confirmed (FR-11) | 1005 | live_relay_confirmed=True |
| 8 | Energy-Selection SSD self-learning loop closes | 1006 | self_learning_closed=True |
| 9 | ThinkPRM probe beats zero-shot on FoVer AUC | 1007 | thinkprm_auroc > baseline_auroc |
| 10 | Retrospective written, ops/ updated | 1012 | honest_verdict includes criteria count |

---

## Dependency Graph

```
[Exp 1000 — Preflight]
    |
    +---> [Exp 1001 — SC-Energy Tier 2]
    |           |
    |           +---> [Exp 1004 — Triple Integration]
    |
    +---> [Exp 1002 — DualGPU]
    |
    +---> [Exp 1003 — SpilledEnergy Live GPU]
                |
                +---> [Exp 1005 — PPSEBM Live Relay]

[Exp 1006 — Energy-Selection SSD]  (independent)
[Exp 1007 — ThinkPRM]              (independent)
[Exp 1008 — VPRM]                  (independent)
[Exp 1009 — GS-KAN]                (independent)
[Exp 1010 — KV260]                 (independent)
[Exp 1011 — ALMC-ODE]              (independent)
[Exp 1012 — Retro]                 (independent)
```

---

## Hardware Requirements

| Experiment | GPU | FPGA | Notes |
|-----------|-----|------|-------|
| 1000–1002 | No | No | CPU-only |
| 1003 | Yes | No | CARNOT_FORCE_LIVE=1, sg render -c |
| 1004–1009 | No | No | CPU-only |
| 1010 | No | Yes | KV260 board at 192.168.51.98, SSH blocked |
| 1011–1012 | No | No | CPU-only |

---

## New Research Papers Added to research-references.md

1. **arXiv 2504.16828** — ThinkPRM: "Process Reward Models That Think" — trains CoT-generating
   step verifiers using 1% of standard PRM training data. +8% GPQA-Diamond, +4.5% LiveCodeBench.
2. **arXiv 2601.17223** — VPRMs: "Beyond Outcome Verification: Verifiable PRMs" — deterministic
   rule-based step verifiers +20% F1 vs SOTA in medical reasoning domain.

---

## Planner Prior_Failures Discipline

All tasks in this roadmap with prior failed attempts include `prior_failures` YAML fields
per CLAUDE.md requirements. Specifically:
- Exp 1001 (SC-Energy): cites exp988, exp976, exp969 failures
- Exp 1002 (DualGPU): cites exp989, exp977, exp966 failures
- Exp 1003 (SpilledEnergy live): cites exp991, exp979, exp964 failures
- Exp 1004 (Triple Integration): cites exp990, exp978, exp965 failures
- Exp 1005 (PPSEBM relay): cites exp994, exp981 failures
- Exp 1009 (GS-KAN): cites exp996 failure with prior_failures_missing root cause

---

## Decentralization Review

All experiments in this milestone satisfy Carnot's decentralization rules (CLAUDE.md):
- Rules 1–2: All GPU experiments use locally-hosted SOTA GGUFs (no closed-weight LLM required)
- Rule 3: No new model weights published in this milestone (carry-forward from .72 HF publish)
- Rule 5: KV260 FPGA track continues sovereignty hardware path
- Rule 7: No vendor-specific imports in core verifier stack (SC-Energy, GS-KAN use abstract protocols)
