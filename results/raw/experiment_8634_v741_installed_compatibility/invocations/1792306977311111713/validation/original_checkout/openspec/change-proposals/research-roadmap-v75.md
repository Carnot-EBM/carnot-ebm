# Research Roadmap — Milestone 2026.04.75

## Title
Prior-Failures Recovery + DualGPU Wire + SC-Energy OOD + KV260 Board Programming

## What Milestone 2026.04.74 Proved

Milestone .74 ran 10 experiments (952-961), but only 3 completed (952, 957, 958). Seven were
blocked (953, 955, 956, 959, 960, 961) or never wrote a result file (954).

**Successes:**
- Exp 952 (Preflight v23): manifest updated — added exp786 and exp641; SOTA models pre-downloaded
  (all 3: Gemma4-31B, Qwen3.6-35B, Gemma4-26B); exp627/exp603 were already in manifest.
- Exp 957 (JEPA v23): OOD AUC=0.2812 — JEPA OFFICIALLY RETIRED. Seven consecutive attempts
  (v17-v23) all below 0.75 gate. SC-Energy AUROC=0.9017 (Exp 944) is the designated replacement.
- Exp 958 (KV260 v4 RTL): rtl_synthesizes_within_budget — 27136 LUTs (25% under 36,250 spec
  estimate), 4 simulation checks pass, E-MVL Sparse Ising sampler fully implemented in Verilog.

**Root cause of 7 blocked experiments:**
The .74 YAML lacked `prior_failures` fields on 7 tasks that overlapped with previously-attempted
experiment domains. The conductor gate-checker correctly blocked all of them. The planner error
was architectural: the .74 YAML was generated without consulting the failure record. .75 MUST
supply prior_failures for every task whose scope overlaps prior attempts — this is the load-
bearing fix.

**Experiment 954 (Fast-Path Probe Live GPU Validation): MISSING RESULT.**
No results/experiment_954_*.json was created. The experiment appears to have never been
launched by the conductor, possibly because it was preceded by the blocked Exp 953 in
the task queue and the queue runner did not continue after consecutive blocks.

**JEPA track permanently closed:**
Exp 957 verdict = jepa_retired. Alternative OOD detector = SC-Energy (AUROC=0.9017).
No further JEPA experiments should be proposed. SC-Energy must be wired as production Tier 2.

**Open RETROs entering .75:**
- RETRO-PROBE-SYNTHETIC-ONLY: SpilledEnergy AUROC=1.0, ThinkPRM AUROC=0.99, Symbolic-KAN
  AUC=1.0 — all CPU/synthetic; need live GPU validation (Exp 954 never ran).
- RETRO-DUALGPU-IDLE: 9 consecutive milestones since Exp 932 benchmarked at 1.96x; 540 min
  cumulative foregone savings; Exp 956 blocked in .74 due to missing prior_failures.
- RETRO-MATH-REPAIR-MODEL-CEILING: Exp 953 blocked in .74; needs retry with prior_failures.
- RETRO-ITERATIVEREPAIR-SCALE: Exp 959 blocked in .74; needs retry with prior_failures.
- RETRO-SYMBOLIC-KAN-UNDEPLOYED: Exp 960 blocked in .74; needs retry with prior_failures.

## Three Biggest Gaps vs PRD Vision

### Gap 1: Seven .74 Tasks Blocked — Prior-Failures Planner Discipline Failure

The primary blocker for .74 was not experimental — it was process. Seven experiments were
blocked because the planning YAML lacked `prior_failures` fields. This milestone recovers
those seven experiments with complete prior_failures documentation.

Each carry-forward task in .75 names every prior experiment in its domain, states the exact
verdict, and explains how the new attempt differs. The planner checklist: for every task,
read research-complete.yaml and ops/conductor-log.md before finalizing the task YAML.

Gap 1 fix: Exp 962 (Preflight v25) audits the .74 failure record and diagnoses Exp 906 root
cause before any .75 GPU experiments run. All six carry-forward tasks include complete
prior_failures entries.

### Gap 2: SC-Energy Must Replace JEPA as Production OOD Detector

JEPA is retired (Exp 957, jepa_retired). The pipeline's Tier 2 slot — originally occupied
by EORM (Exp 340-360) then VJEPA (Exp 884, ood_auc=0.9211) — is now vacant. SC-Energy
(Exp 944, AUROC=0.9017) is the designated alternative but has not been wired into
ThreeTierPipeline. FR-11 (autonomous self-learning loop) requires an OOD detector to
determine when verification is needed; without Tier 2, the pipeline falls through to always
running the expensive Ising Tier 3.

Gap 2 fix: Exp 969 (SC-Energy OOD Production) wires SCEnergyModel as the Tier 2 predictor in
ThreeTierPipeline, with a short-circuit: if SC-Energy coherence score > threshold, skip Tier 3.
Hardware path: SC-Energy is a small MLP — NPU-compatible (AMD XDNA).

### Gap 3: KV260 RTL Done but Board Not Programmed

Exp 958 succeeded: 27136 LUTs, within budget, simulation passes. The KV260 board has been
physically present since 2026-04-20. The full Vivado 2025.2.1 toolchain is installed (per
ops/status.md: "BD wrapper flow end-to-end"). The next step is generating a proper bitstream
from the v4 RTL and programming it to the actual board, then measuring hardware latency.
This is the first opportunity to measure real FPGA Ising sampling latency vs CPU baseline.

Gap 3 fix: Exp 971 (KV260 Board Programming) synthesizes the v4 RTL with Vivado, generates
the bitfile, SCPs it to the board, programs it via dfx-mgr-client, and measures hardware vs
CPU sampling latency. Target: hardware_latency_us < 100ms (any improvement over CPU is a win).

## Architecture After .75

```
ThreeTierPipeline (production deployment target)
  Tier 0a: CarnotThinkProbe (CoT fast path)               [deployed]
  Tier 0b: SpilledEnergy (logit chain-rule energy)         [CPU-proven; live GPU pending]
  Tier 0c: NUPProbeV4 (bigram contrastive energy)         [deployed]
  Tier 0d: HallucinationBasinDetector                      [deployed]
  Tier 0e: HalluField (thermodynamic instability)          [deployed]
  Tier 1: SinkProbe (attention sink concentration)         [deployed]
  Tier 2: SC-Energy (OOD set-consistency, AUROC=0.9017)   [REPLACING JEPA -- new in .75]
  Tier 2.5: SymCodeVerifier (arithmetic execution)         [deployed]
  Tier 2.7: CausalReasoningVerifier (causal entailment)   [deployed]
  Tier 2.9: ThinkPRM (generative CoT verification)         [CPU-proven; live GPU pending]
  Tier 3: Symbolic-KAN (AUC=1.0)                          [needs deploy + HF in .75]

Hardware Acceleration Path
  CPU: Parallel Ising Gibbs (183x vs thrml)               [deployed]
  KV260 FPGA: v4 RTL synthesized (27136 LUT)              [bitstream + board program in .75]
  DualGPU: 1.96x validated (Exp 932)                       [production wiring in .75]

Self-Learning Pipeline
  Tier 1: Online constraint weight updates                  [operational, single-session]
  Tier 2: Cross-session memory (PPSEBM fix for plateau)    [PPSEBM new in .75]
  Tier 3: SC-Energy as predictive OOD (replaces JEPA)     [new in .75]
  Tier 4: KAN adaptive structure                           [seeded, not wired]
```

## Experiments Summary

| Exp | Title | GPU | max_turns | Priority |
|-----|-------|-----|-----------|----------|
| 962 | Preflight v25 — Exp 906 Diagnosis + .74 Audit | No | 20 | critical |
| 963 | Math Repair SOTA v3 — Retry with prior_failures | Yes | 50 | critical |
| 964 | Fast-Path Probe Live GPU Validation | Yes | 30 | high |
| 965 | Triple Integration SC-Energy + SpilledEnergy + ThinkPRM | No | 50 | high |
| 966 | DualGPU VerifyRepairPipeline Wiring | Yes | 50 | high |
| 967 | IterativeSelfRepair 100q + DebugRepair Hypothesis | Yes | 50 | medium |
| 968 | Symbolic-KAN v2 Deploy + HuggingFace Publish | No | 30 | medium |
| 969 | SC-Energy Production OOD Detector (replacing JEPA) | No | 30 | high |
| 970 | PPSEBM Tier 2 Cross-Session Memory (self-learning) | No | 50 | medium |
| 971 | KV260 v4 FPGA Board Programming | No | 50 | medium |
| 972 | KAN Formal Verification via MILP (arXiv 2602.06737) | No | 50 | medium |
| 973 | Milestone 2026.04.75 Retrospective | No | 20 | mandatory |

## Dependency Graph

```
Exp 962 (preflight)
  ├──► Exp 963 (math repair, gated on sota_models_ready)
  ├──► Exp 964 (live GPU probes, gated on sota_models_ready)
  └──► Exp 966 (DualGPU, gated on sota_models_ready)

Exp 957 (JEPA retired, .74)  ──► Exp 969 (SC-Energy OOD)
Exp 958 (KV260 RTL, .74)     ──► Exp 971 (Board programming)
Exp 944 (SC-Energy, .73)     ──► Exp 965 (Triple integration) ──► Exp 969

Exp 965 (Triple integration) ──► standalone, no downstream
Exp 967 (100q repair)        ──► standalone (different domain from Exp 963)
Exp 968 (Symbolic-KAN)       ──► Exp 972 (KAN MILP verification needs model)
Exp 970 (PPSEBM)             ──► standalone
Exp 963-972 all              ──► Exp 973 (retrospective)
```

## Success Criteria

Milestone 2026.04.75 succeeds if 8 of 10 criteria are met:

1. Zero experiments blocked by missing prior_failures (planner discipline — verified in Exp 973)
2. Math Repair SOTA v3 writes a result file with any verdict (Exp 963 — result_file_written)
3. SpilledEnergy AUROC >= 0.70 on live GPU with real model responses (Exp 964)
4. ThinkPRM AUROC >= 0.80 on live GPU with real model responses (Exp 964)
5. Triple Integration wired and E2E cascade test passes (Exp 965)
6. DualGPU production wiring validated with pipeline throughput >= 1.5x (Exp 966)
7. SC-Energy wired as Tier 2 production OOD detector in ThreeTierPipeline (Exp 969)
8. KV260 v4 bitstream generated (Vivado synthesis passes) OR board programmed (Exp 971)
9. PPSEBM Tier 2 cross-session: non-zero templates added in sessions 3+ (Exp 970)
10. Retrospective written and ops/status.md updated (Exp 973)

## Governance Actions (Mandatory in Exp 962)

1. Read results/experiment_906_*.json and diagnose Exp 906 root cause (IterativeSelfRepair 50q
   at 35 min, 2 consecutive slow appearances). Apply CARNOT_EXPERIMENT_TIMEOUT_MINUTES=40 cap
   or retire to exclusion_manifest.yaml.

2. Confirm ops/exclusion_manifest.yaml contains: exp786, exp627, exp603, exp641 (added in Exp 952).

3. Verify all 3 SOTA GGUFs still cached:
   - unsloth/gemma-4-31B-it-GGUF
   - unsloth/Qwen3.6-35B-A3B-GGUF
   - unsloth/gemma-4-26B-A4B-it-GGUF

4. Audit Exp 954 missing result: determine if it was never launched, or launched and failed
   silently. This determines whether Exp 964 is a retry or a fresh experiment.

## RETRO Closures Expected

| RETRO ID | Closes Via | Condition |
|----------|-----------|-----------|
| RETRO-PROBE-SYNTHETIC-ONLY | Exp 964 | live GPU AUROC confirmed |
| RETRO-DUALGPU-IDLE | Exp 966 | production wiring validated >= 1.5x |
| RETRO-MATH-REPAIR-MODEL-CEILING | Exp 963 | result file written (any verdict) |
| RETRO-ITERATIVEREPAIR-SCALE | Exp 967 | result file written (any verdict) |
| RETRO-SYMBOLIC-KAN-UNDEPLOYED | Exp 968 | HF push succeeds + integration test passes |

## Decentralization Implications

All .75 experiments preserve rules 1-7 (research-program.md):
- SOTA GGUFs are local unsloth models (rule 1: local-first)
- KV260 FPGA uses OSS-CAD-Suite open toolchain (rule 5: hardware portability)
- HuggingFace + IPFS dual-channel established (rule 3: distribution mirroring)
- SC-Energy, Symbolic-KAN, PPSEBM are all open models (rule 2: no closed-weight requirement)
- ThreeTierPipeline exposed via Python API + CLI + MCP + HTTP (rule 4: multiple surfaces)

## arxiv References Filed This Planning Cycle

Three new papers added to research-references.md before milestone design:

1. **arXiv 2602.06737** — Optimal Abstractions for Verifying KAN Properties via MILP:
   PWA abstraction + MILP for formal KAN property verification; Exp 972.
2. **arXiv 2512.15658** — PPSEBM: Progressive Parameter Selection EBM for Continual Learning:
   EBM + progressive params prevents cross-session saturation; Exp 970.
3. **arXiv 2601.03600** — ALERT: Zero-Shot Jailbreak Detection via Internal Discrepancy:
   Training-free jailbreak detection; deferred to .76 (Tier B safety product).
