# Research Roadmap — Milestone 2026.04.74

## Title
Governance Surgery + Live Probe Validation + Math Repair SOTA v3 + DualGPU Wiring

## What Milestone 2026.04.73 Proved

Milestone .73 ran 12 experiments (941-951), achieving 10/12 criteria:

**Successes:**
- Symbolic-KAN on real FoVer data (Exp 948): AUC=1.0 on 57 real violation pairs — best
  discriminative result in project history, surpassing prior synthetic AUC=0.9344
- SpilledEnergy Tier 0 (Exp 949): AUROC=1.0 training-free hallucination detector — logit
  spill computed directly from chain-rule probability, no model training required
- ThinkPRM Tier 2.9 (Exp 945): AUROC=0.99 vs heuristic 0.85 — generative CoT step
  verification closes RETRO-HEURISTIC-RPRM-FLAT-SIGNAL after 3 milestones
- SC-Energy v2 (Exp 944): AUROC=0.9017 set-level consistency energy network — closes
  RETRO-SC-ENERGY-GATE-DISCIPLINE after 2 milestones of gate failures
- Tier 2.8 DraftConditioned live GPU validated (Exp 946): GPU path confirmed working
- DRIFTProbe v3 depth-recurrent (Exp 947): probe_auc=0.5807 — incremental improvement;
  depth-recurrent attention pooling is architecturally correct per arXiv 2604.17121
- E-MVL Sparsified Ising (Exp 950): emvl_comparable at K=16 (1.25x speedup, 36,250 LUTs
  within KV260 117K budget); v4 RTL spec written at hardware/kv260/ising_sampler_v4_spec.md

**Failures:**
- Exp 942 (SOTA math repair): result file MISSING — experiment never wrote output before
  exit. Root cause: likely SOTA GGUF model not cached, download failed silently, exit
  before result write. Open RETRO: RETRO-MATH-REPAIR-MODEL-CEILING (third consecutive).
- Exp 941 (preflight v22): slow-5 carryovers 786/627/603 appeared for 14th consecutive
  milestone; Exp 641 appeared for 4th consecutive past mandatory gate; no manifest update.

**Open RETROs entering .74:**
- RETRO-MATH-REPAIR-MODEL-CEILING: Exp 942 never wrote result; SOTA model pre-download
  required. Exp 930 also failed (12% baseline model ceiling). Exp 919 blocked by gate.
  Three attempts over three milestones — .74 is the last attempt before retirement.
- RETRO-DUALGPU-IDLE: 8 consecutive milestones since Exp 932 benchmarked DualGPU at 1.96x.
  Pipeline never wired. 480 min cumulative foregone savings.
- RETRO-PROBE-SYNTHETIC-ONLY: SpilledEnergy AUROC=1.0 (Exp 949), ThinkPRM AUROC=0.99
  (Exp 945), Symbolic-KAN AUC=1.0 (Exp 948) all validated on CPU/synthetic. Need live GPU
  validation on real model responses before claiming headline results.
- RETRO-MANIFEST-FULL-SCOPE: ops/exclusion_manifest.yaml still missing Exp 786, 627, 603,
  641 despite 14 consecutive milestone appearances with RETRO flags.

## Three Biggest Gaps vs PRD Vision

### Gap 1: Governance Debt Compounding (HIGHEST PRIORITY)

Four experiments (786, 627, 603, 641) have appeared in consecutive milestones for 4-14
iterations with no progress, burning an estimated 172+ min/milestone in conductor idle time.
The exclusion manifest (ops/exclusion_manifest.yaml) exists specifically for this purpose
but has not been updated. Additionally, SOTA GGUF model pre-download is required BEFORE
any GPU experiment runs — the Exp 942 failure proves that absent pre-download, the model
download takes 30+ min and often times out before the result file is written.

Gap 1 fix: Exp 952 (Preflight v23) adds 786/627/603/641 to exclusion manifest AND runs
explicit SOTA model pre-download for all models needed in .74 GPU experiments.

### Gap 2: Three AUC=1.0 Results Are CPU/Synthetic — Not Credible Headlines Without Live GPU

SpilledEnergy AUROC=1.0 (Exp 949), ThinkPRM AUROC=0.99 (Exp 945), and Symbolic-KAN AUC=1.0
(Exp 948) are the strongest results in project history. All were run on CPU with synthetic
or limited real data. Per CLAUDE.md: "All headline results must have live GPU provenance."
These results cannot be cited in publications or the project README until live GPU validation
runs real model responses through the full pipeline.

Gap 2 fix: Exp 954 (Fast-Path Probe Live GPU Validation) runs SpilledEnergy + ThinkPRM +
DRIFTProbe v3 against real Gemma4-31B responses on the ROCm GPU, gated on proven CPU results.

### Gap 3: DualGPU Deployed but Unused (8 Milestones)

Exp 932 benchmarked DualGPU at 1.96x throughput on realistic workloads in milestone .72.
Eight milestones later (approximately 480 cumulative minutes of foregone savings per
milestone estimate), DualGPU is still not wired into VerifyRepairPipeline. Every GPU
experiment since then has run single-GPU. The benchmark result is valid; the problem is
the wiring was never implemented.

Gap 3 fix: Exp 956 (DualGPU VerifyRepairPipeline Wiring) implements the production wiring,
with an explicit gate that retires DualGPU from the roadmap permanently if the result is
not production-validated by end of .74.

## Additional Research-Driven Experiments

### Math Repair SOTA v3 (RETRO-MATH-REPAIR-MODEL-CEILING Closure)

Exp 942 never ran (no result file). Exp 930 proved E4B model ceiling (12% baseline). The
fix is explicit pre-downloaded SOTA GGUF (Gemma4-31B or Qwen3.6-35B, ~75% GSM8K baseline)
plus a mandatory result-write guard that writes a partial artifact on any exception, making
diagnosis possible even if the experiment crashes. Per CLAUDE.md failed-experiment discipline:
if Exp 953 produces the same verdict as 930 (zero repair improvement), the entire
SOTA-math-repair experiment line is retired and RETRO-MATH-REPAIR-MODEL-CEILING is closed
as "model ceiling even at SOTA — technique deprecated."

### Three-Tier Pipeline Integration (SC-Energy + SpilledEnergy + ThinkPRM)

SC-Energy (Exp 944), SpilledEnergy (Exp 949), and ThinkPRM (Exp 945) are all proven on
CPU. None are wired into ThreeTierPipeline. Exp 955 integrates all three, establishing
the full Tier 0 (SpilledEnergy) → Tier 2.9 (ThinkPRM) → Tier 3 (SC-Energy) pipeline.

### FR-11 JEPA v23 with SC-Energy Coherence Labels

JEPA OOD AUC has stalled below 0.75 across v17-v22 (6 consecutive milestones). The
diagnosed root cause: self-supervised training signal from forward-world-model loss is
not discriminative enough to separate in-distribution from out-of-distribution. SC-Energy
(Exp 944, AUROC=0.9017) produces exactly the kind of set-level coherence signal needed
as a training label. JEPA v23 uses SC-Energy coherence scores as auxiliary training
supervision, which provides direct gradient signal toward the OOD detection objective.

### KV260 v4 RTL Implementation

E-MVL v4 RTL spec is written (hardware/kv260/ising_sampler_v4_spec.md from Exp 950).
KV260 arrived (2026-04-20), OSS-CAD-Suite installed. This is the first opportunity to
implement and simulate the sparse Ising sampler in RTL. Exp 958 targets: synthesizable
VHDL/Verilog, yosys synthesis passing, simulation producing correct Ising distributions.

### IterativeSelfRepair Scale-Up + DebugRepair Hypothesis Step

Exp 905 proved IterativeSelfRepair works at 25q (code domain, 4%→72%+68pp). Exp 959
scales to 100q and adds the DebugRepair hypothesis step from arXiv 2604.19305 (+8.2pp
HumanEval vs standard self-repair in the paper). Gated on Exp 952 confirming SOTA model
pre-downloaded to avoid Exp 906's 30+ min download overhead within timed experiment.

### Symbolic-KAN v2 Pipeline Deploy + HuggingFace Publish

Exp 948 proved Symbolic-KAN AUC=1.0 on 57 real FoVer pairs. The result has never been
deployed to the production pipeline or published to HuggingFace. Exp 960 deploys the
model as a production Tier 3 verifier and publishes to huggingface.co/Carnot-EBM per
CLAUDE.md distribution-mirroring rule (HF + IPFS dual-channel established in Exps 933/934).

## Experiments Summary

| Exp | Title | GPU | max_turns | Gates |
|-----|-------|-----|-----------|-------|
| 952 | Preflight v23 — Manifest + SOTA Pre-Download | No | 20 | none |
| 953 | Math Repair SOTA v3 | Yes | 50 | exp952.sota_downloaded |
| 954 | Fast-Path Probe Live GPU Validation | Yes | 30 | exp949.auroc>0.60 AND exp945.auroc>0.60 |
| 955 | Triple Integration (SC-Energy + SpilledEnergy + ThinkPRM) | No | 50 | exp944.sc_energy_viable |
| 956 | DualGPU VerifyRepairPipeline Wiring | Yes | 50 | exp932.dualgpu_benchmark_valid |
| 957 | FR-11 JEPA v23 SC-Energy Coherence Labels | No | 50 | exp944.sc_energy_viable |
| 958 | KV260 v4 RTL Sparse Ising Implementation | No | 50 | exp950.emvl_comparable |
| 959 | IterativeSelfRepair 100q + DebugRepair Hypothesis | Yes | 50 | exp952.sota_downloaded |
| 960 | Symbolic-KAN v2 Pipeline Deploy + HF Publish | No | 30 | exp948.auc>=0.90 |
| 961 | Milestone 2026.04.74 Retrospective | No | 20 | none |

## Success Criteria

Milestone 2026.04.74 succeeds if 8 of 10 criteria are met:

1. Exclusion manifest updated: 786, 627, 603, 641 all added (Exp 952)
2. SOTA GGUF models pre-downloaded before any GPU experiment (Exp 952)
3. Math repair SOTA v3 runs to completion AND writes result file (Exp 953) —
   any verdict acceptable; result file absent = failure
4. SpilledEnergy AUROC >= 0.80 on live GPU with real Gemma4-31B responses (Exp 954)
5. ThinkPRM AUROC >= 0.85 on live GPU with real Gemma4-31B responses (Exp 954)
6. Three-tier pipeline integration verified E2E (SC-Energy + SpilledEnergy + ThinkPRM) (Exp 955)
7. DualGPU production wiring validated with real pipeline throughput >= 1.5x (Exp 956)
8. JEPA v23 OOD AUC > 0.75 OR FR-11 officially retired with documented root cause (Exp 957)
9. KV260 v4 RTL synthesizes without errors under yosys (Exp 958)
10. Symbolic-KAN model published to HuggingFace AND deployed to production Tier 3 (Exp 960)

## Governance Actions (Mandatory in Exp 952)

The following entries MUST be added to `ops/exclusion_manifest.yaml` in Exp 952:

```yaml
- experiment_id: exp786-scope-carryover
  reason: "14 consecutive milestone appearances; no result in 14 runs; technique plateau"
  retire_if_same_verdict: true
  added_milestone: "2026.04.74"

- experiment_id: exp627-scope-carryover
  reason: "14 consecutive milestone appearances; no result in 14 runs; technique plateau"
  retire_if_same_verdict: true
  added_milestone: "2026.04.74"

- experiment_id: exp603-scope-carryover
  reason: "14 consecutive milestone appearances; no result in 14 runs; technique plateau"
  retire_if_same_verdict: true
  added_milestone: "2026.04.74"

- experiment_id: exp641-dualgpu-gate-cascade
  reason: "4 consecutive milestones past mandatory gate; gate conditions will never be met
           without explicit wiring experiment (Exp 956)"
  retire_if_same_verdict: true
  added_milestone: "2026.04.74"
```

## RETRO Closures Expected

| RETRO ID | Closes Via | Condition |
|----------|-----------|-----------|
| RETRO-MANIFEST-FULL-SCOPE | Exp 952 | manifest updated with 786/627/603/641 |
| RETRO-MATH-REPAIR-MODEL-CEILING | Exp 953 | result file written (any verdict) |
| RETRO-PROBE-SYNTHETIC-ONLY | Exp 954 | live GPU AUROC confirmed |
| RETRO-DUALGPU-IDLE | Exp 956 | production wiring validated |

## arxiv References Filed This Planning Cycle

Three new papers added to research-references.md before milestone design:

1. **arXiv 2502.11250** — Uncertainty-Aware Step Verification via CoT Entropy:
   CoT-entropy UQ for generative PRMs; deferred to .75 (ThinkPRM live baseline first)
2. **arXiv 2604.20701** — Divide-and-Conquer NN Surrogates for Sparse Ising:
   20x MCMC speedup on sparse Ising; deferred to .75 (KV260 v4 RTL baseline first)
3. **arXiv 2602.17750** — iCKANs: Inelastic Constitutive KANs with Symbolic Regression:
   Physical constraint regularization for KANs; deferred to .75 (Symbolic-KAN deploy first)
