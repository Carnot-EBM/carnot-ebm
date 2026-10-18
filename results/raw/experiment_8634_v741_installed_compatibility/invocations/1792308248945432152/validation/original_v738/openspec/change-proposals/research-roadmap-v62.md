# Research Roadmap — Milestone 2026.04.62

**Title:** JEPA v22 CPMI Rewire + Gemma4 Fix v5 + Constraint Live Delta + OSS-CAD FPGA

**CalVer:** 2026.04.62 (sequence increment from 2026.04.61)
**Planned Experiments:** Exps 806-818 (13 experiments)
**Date Designed:** 2026-04-24
**Prerequisite:** Milestone 2026.04.61 retro complete (Exp 805)

---

## What Milestone 2026.04.61 Proved

Milestone .61 (Exps 793-805) hit 3 of 10 success criteria and produced one critical diagnostic:

**Wins:**
- JEPA v21 data collection: n_labeled=300 (3.75x the 80-pair target) — data starvation solved
- CPMI augmentation: augmentation_ratio=2.5 — hard negative mining pipeline works
- EmbeddingConstraintStore: retrieval AUC=0.92 — SPO-format constraint encoding correct
- HuggingFace publish (bonus): model READMEs updated and pushed (RETRO-HF-AUTH resolved)

**Critical finding — JEPA v21 wiring miss:**
Exp 799 (JEPA v21 retrain) ran with augmentation_ratio=1.0 (the CPMI triples from Exp 798
were not loaded). JEPA v21 OOD AUC=0.2444 is an all-time project low — caused entirely by
this implementation handoff error, not an algorithmic failure. The CPMI hypothesis is still
untested. A single assert augmentation_ratio > 1.0 at script startup would have caught this.

**Persistent failures (direct carry into .62):**
- RETRO-028: Gemma4 OOM — 4th+ consecutive milestone without closure (partial_success only)
- RETRO-CONSTRAINT-ZERO-DELTA: delta=0.0 — EmbeddingConstraintStore AUC=0.92 but NOT wired
  into IsingEBM coupling matrix (synthetic_cpu stub is deterministic, delta always 0)
- RETRO-KV260-TOOLS-UNAVAILABLE: yosys/nextpnr-ice40/icepack absent — 3rd consecutive milestone
- FR-11 Tier 1: tier1_plateau_persists — blocked upstream by constraint injection wiring miss
- SOTA code repair: never ran — gated on RETRO-028 closure

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: JEPA OOD Generalization — 9 Consecutive Failures, 1 Pure Wiring Bug

JEPA has failed to reach OOD AUC >= 0.75 in 9 consecutive retrains (v13-v21). The v21
result (0.2444) is uniquely diagnostic: it is NOT an algorithmic regression — it is the
identical model architecture run WITHOUT the CPMI data it was supposed to have. The correct
interpretation is: "we do not know if CPMI works because the experiment ran without it."

**Fix for .62:**
- Exp 808 adds the missing data loader merge: load fover_labeled_steps_v21_multi.json
  (300 pairs) AND experiment_798_cpmi_pairs_triples.json (750 triples). Assert augmentation_ratio > 1.0
  at startup before any training begins.
- If JEPA v22 still fails OOD after the correct wiring: apply RA-PRM (arXiv 2502.14361) —
  retrieval-augmented soft supervision from similar FoVer-labeled steps (Exp 809).

### Gap 2: Constraint Injection Stub — EmbeddingConstraintStore Retrieval Works But Delta Is Zero

EmbeddingConstraintStore (Exp 800) achieves retrieval AUC=0.92 across 5 constraint types.
The constraint vectors are good. But Exps 801 and 802 both show delta=0.0 because the
inference mode uses a deterministic synthetic_cpu placeholder — IsingEBM.infer() ignores
the retrieved constraint features entirely. The EmbeddingConstraintStore output is never
passed to IsingEBM as penalty inputs.

**Fix for .62:**
- Exp 812 wires EmbeddingConstraintStore output into IsingEBM as soft penalty inputs on the
  spin coupling matrix. Retrieved constraint embeddings are projected onto the Ising coupling
  space and added as bias terms. This is a 4-6 hour implementation (identified in .61 retro).
- Exp 813 then runs the live validation: 30 GSM8K questions x 3 sessions with live GPU.
  If retrieval AUC=0.92 translates correctly, delta should be measurable on real inference.

### Gap 3: FPGA Toolchain Blocked Three Consecutive Milestones — Wrong Installation Method

Three milestones of `sudo pacman -S yosys nextpnr icestorm` have all failed because the
conductor environment cannot run pacman with sudo. OSS-CAD-Suite (YosysHQ/oss-cad-suite-build)
is the correct alternative: a pre-built binary tarball downloaded from GitHub, no package
manager or sudo required. A single `curl` + `tar` command provides all three tools.

**Fix for .62:**
- Exp 807 downloads OSS-CAD-Suite release tarball from GitHub API (no sudo required),
  extracts to ~/tools/oss-cad-suite/, and verifies all three tools.
- Exp 816 synthesizes ising_sampler_v3.v with the installed tools (gated on Exp 807).

---

## Architecture Diagram

```
Query
  |
  v
[Tier 0a] CarnotThinkProbe (generative CoT verdict)
  |  fast-path on "incorrect" verdict
  v
[Tier 0b] SpilledEnergyDetector (logit-discrepancy, arXiv 2602.18671)
  |
  v
[Tier 0c] NUP Probe v4 (contrastive energy, AUC=1.0, Exp 523)
  |
  v
[Tier 0d] HallucinationBasinDetector (latent basin depth, arXiv 2604.04743)
  |
  v
[Tier 0e] HalluField (thermodynamic instability, advisory, arXiv 2509.10753)
  |
  v
[Tier 0h] JailbreakDetectionKAN (safety gate, AUC=1.0, Exp 775)
  |  returns SAFETY_GATE if jailbreak detected
  v
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
  |
  v
[Tier 2]  EORM (CoT energy reward model, 55M params)
  |
  v
[Tier 2.1] JEPAReasonerProbe (latent-space reasoning, AUC=0.993, Exp 726)
  |
  v
[Tier 2.5] SymCodeVerifier (executable arithmetic, AUC=0.804 live)
  |
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback loop)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment across steps)
  |
  v
[Tier 3]  IsingEBM (full constraint verification, 0.006 ms/check)
  |        + EmbeddingConstraintStore → IsingEBM coupling injection [FIX IN .62]
  v
[Tier 3.5] JEPA v22 (OOD predictor, TARGET: AUC >= 0.75) [v21: 0.2444 wiring miss]

Self-Learning Loop (FR-11):
  Tier 1: EmbeddingConstraintStore → IsingEBM coupling matrix [NEW IN .62]
  Tier 2: Cross-session constraint memory (persist EmbeddingConstraintStore via JSON)
  Tier 3: JEPA v22 trained on multi-source + CPMI corpus (correctly wired)

VG-Search Scheduling [NEW IN .62]:
  At each tier: track energy variance → skip expensive tier if variance low
  Granularity adapts to question difficulty (arXiv 2505.11730)

Multi-Agent Arbiter [NEW IN .62]:
  score_agent_outputs MCP tool: N agent responses → energy-ranked selection
```

---

## Phase Descriptions

### Phase 0: Governance + FPGA Unblock (Exps 806-807) — CPU only

**Goal:** Implement MILESTONE_PREREQS.md process gate and unblock the FPGA toolchain.

**Exp 806 — MILESTONE_PREREQS.md Gate Implementation**
The .61 retro explicitly called for a "one-page checklist file (MILESTONE_PREREQS.md) listing
all IMMEDIATE-class retro actions from the prior retro" to be verified before any milestone
experiments run. This experiment implements that file AND the spec/test infrastructure for
it. The gate is also a pre-run assertion enforcement system: add check_augmentation_ratio()
to JEPA experiment scripts so wiring misses are caught at startup (the single fix that would
have prevented JEPA v21 all-time low).

**Exp 807 — OSS-CAD-Suite FPGA Toolchain Install** (RETRO-KV260-TOOLS-UNAVAILABLE, attempt v3)
Download OSS-CAD-Suite pre-built tarball from GitHub (YosysHQ/oss-cad-suite-build releases
API). No sudo required. Extract to ~/tools/oss-cad-suite/. Add to PATH and verify:
`yosys --version`, `nextpnr-ice40 --version`, `icepack --help`. Run minimal 2-spin Ising
Verilog test synthesis as proof. This gates Exp 816 (KV260 synthesis v2).

### Phase 1: JEPA v22 Wiring Fix + OOD Hardening (Exps 808-809) — CPU only

**Goal:** Evaluate whether CPMI hard negatives actually fix JEPA OOD (untested in v21).

**Exp 808 — JEPA v22 CPMI Corpus Rewire + Retrain** (RETRO-JEPA-OOD attempt #10, CPU)
The only change from Exp 799: the training data loader now merges BOTH
fover_labeled_steps_v21_multi.json (300 pairs from Exp 797) AND
experiment_798_cpmi_pairs_triples.json (750 CPMI triples from Exp 798).
Assert augmentation_ratio > 1.0 at startup (would have caught the v21 wiring miss).
Expected: ood_auc >= 0.75 if CPMI hard negatives are the correct fix.
If still < 0.75: write domain-level AUC breakdown to diagnose which failure mode persists.

**Exp 809 — JEPA v22 RA-PRM OOD Enhancement** (arXiv 2502.14361, fallback if Exp 808 fails)
If Exp 808 ood_auc < 0.75: apply retrieval-augmented PRM. For each training step, retrieve
3 similar FoVer-labeled steps via EmbeddingConstraintStore (AUC=0.92 from Exp 800).
Use retrieved labels as soft targets (0.4 weight each vs 1.0 for ground-truth FoVer label).
This directly addresses step-OOD: similar steps from the training distribution anchor the
prediction for novel domain steps. Run without GPU (CPU training, same architecture as v22).
If Exp 808 ood_auc >= 0.75: Exp 809 evaluates on additional held-out benchmark (ARC, SVAMP).

### Phase 2: GPU Unblock (Exps 810-811) — GPU required

**Goal:** Close RETRO-028 (4th+ consecutive) and run SOTA code repair.

**Exp 810 — Gemma4 OOM Fix v5** (RETRO-028, CARNOT_FORCE_LIVE=1)
New isolation protocol: nvidia-smi verification LOOP (poll every 5s, 3 retries, 10s sleep
between retries). If VRAM does not drop below 500MB after 3 retries: abort with diagnostic
artifact (do NOT attempt model load). Use GPU 1 (38C, 10C cooler than GPU 0).
Add checkpoint_save() after each verification step so partial results survive timeout.
honest_verdict=retro_028_closed when n_valid_responses >= 8 on live GPU.

**Exp 811 — SOTA GGUF Code Repair v4** (GATED on Exp 810 retro_028_closed, GPU)
50 HumanEval problems in 10 batches of 5. ExperimentTimeoutWatchdog at 90 min.
Checkpoint per batch (atomic JSON write). Qwen3.6-35B-A3B Q4_K_M GGUF on GPU 1.
Apply MARS margin gate (arXiv 2601.15498): skip test oracle for high-margin outputs.
honest_verdict=code_repair_positive if signed_improvement > 0 on >= 10 problems live GPU.

### Phase 3: Constraint Injection Live Fix (Exps 812-813) — CPU + GPU

**Goal:** Wire EmbeddingConstraintStore into IsingEBM coupling matrix (the identified .61 fix).

**Exp 812 — IsingEBM Constraint Feature Injection** (RETRO-CONSTRAINT-ZERO-DELTA root fix, CPU)
Implement the fix identified in the .61 retro: wire EmbeddingConstraintStore retrieved
constraint embeddings into IsingEBM.infer() as soft penalty inputs on the spin coupling matrix.
Mechanism: retrieved constraint vector is projected to spin-space via a learned linear map;
projection is added as a bias term on the coupling matrix J before Ising sampling.
This makes the energy function constraint-aware: violations flagged by memory produce higher
energy on the spins encoding the violating claim. CPU-only; no GPU needed.
Test: energy increases when retrieved constraint fires vs when no constraint retrieved.

**Exp 813 — Constraint Addition Live Validation** (GPU required, GATED on Exp 812)
With IsingEBM coupling injection wired: run 30 GSM8K questions x 3 sessions on live GPU.
Compare: baseline VerifyRepairPipeline vs pipeline with EmbeddingConstraintStore active.
Measure constraint_addition_delta on real inference (NOT deterministic synthetic_cpu stub).
honest_verdict=constraint_addition_works if delta_overall > 0 across all 3 sessions.
This closes RETRO-CONSTRAINT-ZERO-DELTA if delta > 0 on live data.

### Phase 4: Self-Learning + Compute Efficiency (Exps 814-815) — GPU + CPU

**Goal:** Close FR-11 Tier 1 on live data and reduce unnecessary energy computation.

**Exp 814 — FR-11 Tier 1 Live Relay** (FR-11 mandatory, GPU, GATED on Exp 813 delta>0)
Run 5-session relay on live GPU with IsingEBM coupling injection active.
Each session: 20 GSM8K questions (real inference). Track precision per session.
Update EmbeddingConstraintStore after each session with violations found.
Tier 1 criterion: precision non-decreasing AND delta > 0 by session 3.
Apply capacity-constrained update (arXiv 2507.21479): update only high-variance constraints,
freeze well-calibrated ones. This should prevent the plateau seen in Exps 802.

**Exp 815 — VG-Search Energy Scheduling** (arXiv 2505.11730, CPU)
Implement Variable Granularity energy scheduling in ThreeTierPipeline.
At each tier, track rolling energy variance (last 3 checks on the current session).
If variance < threshold: extend interval before next tier check (MARS-style skip).
If variance > threshold: check at full frequency.
Test on 50 synthetic GSM8K responses. Measure: constraint_calls_reduced, accuracy_delta.
Target: 30% fewer Ising calls at <= 1% accuracy degradation.
Export: VGSearchScheduler from carnot.pipeline.

### Phase 5: Hardware + New Capability (Exps 816-817) — CPU + GPU

**Goal:** Flash KV260 with open-source synthesis and ship Multi-Agent Arbiter.

**Exp 816 — KV260 Open-Source Synthesis v2** (GATED on Exp 807 tools_installed, CPU)
With OSS-CAD-Suite installed: synthesize hardware/kv260/ising_sampler_v3.v at N=32
using yosys with synth_ice40 target. Then attempt N=64 if LUT budget allows.
Parse yosys output for LUT count, timing, and errors.
honest_verdict=synthesis_clean_n32 if yosys completes without errors AND lut_count < 5000.
Update research-hardware-wishlist.md with synthesis result and LUT count.

**Exp 817 — Multi-Agent Arbiter MCP Tool** (Tier B product, CPU)
Implement score_agent_outputs MCP tool: given N agent responses to the same query,
return the response with lowest EBM energy (most constraint-consistent).
Architecture: run each response through VerifyRepairPipeline.verify(), collect energy scores,
return ranked list with energies. Export as MCP tool scored via IsingEBM.
Test: 3 synthetic agents with different math answers to same problem.
Agent A: wrong answer (high energy). Agent B: correct answer (low energy). Agent C: wrong.
Verify arbiter returns Agent B. honest_verdict=arbiter_correct if winner_is_correct.
Export: score_agent_outputs from carnot.mcp and carnot.pipeline.

### Phase 6: Retrospective (Exp 818)

**Exp 818 — Milestone 2026.04.62 Operational Retrospective**
Standard retro format (schema=carnot.operational_retro.v37).
Success criteria: 9 gates evaluated.
Write improvements_suggested for .63 based on blockers found.

---

## Dependency Graph

```
Exp 806 (MILESTONE_PREREQS.md)    → no dependency (CPU)
Exp 807 (OSS-CAD Install)         → no dependency (CPU)
  ↓
Exp 808 (JEPA v22 Rewire)         → no dependency (CPU, uses Exps 797/798 artifacts)
Exp 809 (JEPA v22 RA-PRM)         → GATED on Exp 808 ood_auc result (CPU)
  ↓
Exp 810 (Gemma4 Fix v5)           → CARNOT_FORCE_LIVE=1 (GPU)
Exp 811 (SOTA Code Repair v4)     → GATED on Exp 810 retro_028_closed (GPU)
  ↓
Exp 812 (Ising Coupling Inject)   → no dependency (CPU)
Exp 813 (Constraint Live Validation) → GATED on Exp 812 (GPU)
  ↓
Exp 814 (FR-11 Tier 1 Live Relay)  → GATED on Exp 813 delta>0 (GPU)
Exp 815 (VG-Search Scheduling)     → no dependency (CPU)
  ↓
Exp 816 (KV260 Synthesis v2)       → GATED on Exp 807 tools_installed (CPU)
Exp 817 (Multi-Agent Arbiter)      → no dependency (CPU)
  ↓
Exp 818 (Retro)                    → all experiments complete
```

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| prereqs_gate_implemented | Exp 806 | MILESTONE_PREREQS.md written + assertion infra deployed |
| fpga_tools_installed | Exp 807 | yosys + nextpnr-ice40 + icepack verified on PATH |
| jepa_v22_ood_viable | Exp 808 | ood_auc >= 0.75 (cascade deploy gate) |
| retro_028_closed | Exp 810 | honest_verdict=retro_028_closed (n_valid_responses>=8) |
| sota_code_repair_positive | Exp 811 | signed_improvement > 0 live GPU |
| constraint_injection_wired | Exp 812 | energy increases when constraint fires |
| constraint_addition_live | Exp 813 | delta_overall > 0 on live GPU inference |
| tier1_relay_live | Exp 814 | precision non-decreasing + delta > 0 by session 3 |
| kv260_synthesis_clean | Exp 816 | lut_count < 5000 + no synthesis errors |

---

## New Research Papers Incorporated

| Paper | ArXiv | Incorporated In |
|-------|-------|----------------|
| Retrieval-Augmented PRM for OOD | 2502.14361 | Exp 809 (fallback if v22 fails OOD) |
| Variable Granularity Search | 2505.11730 | Exp 815 (VG-Search scheduling) |
| Capacity-Constrained CL | 2507.21479 | Exp 814 (selective constraint update) |
| OSS-CAD-Suite (GitHub) | — | Exp 807 (FPGA toolchain) |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|---------|-------|
| Exps 810, 811, 813, 814 | GPU (CARNOT_FORCE_LIVE=1) | RTX 3090; GPU 1 for Gemma4 (thermal) |
| All others | CPU | JAX_PLATFORMS=cpu |
| Exp 816 | CPU + OSS-CAD | No GPU; yosys synthesis is CPU-bound |

---

## Key Invariants for .62 Experiments

1. **CPMI wiring assertion:** Every JEPA retrain script MUST have `assert augmentation_ratio > 1.0`
   before training begins. This is a pre-condition, not a post-condition.

2. **nvidia-smi loop gate:** Gemma4 experiments MUST use the loop verification (3 retries, 10s
   sleep) before attempting model load. No model load until VRAM < 500MB confirmed.

3. **Live vs synthetic delta:** Constraint addition experiments MUST report inference_mode.
   delta results from synthetic_cpu mode are explicitly labeled "synthetic_only" and do NOT
   count toward RETRO-CONSTRAINT-ZERO-DELTA closure.

4. **OSS-CAD path:** All FPGA experiments use ~/tools/oss-cad-suite/bin — never pacman.
