# Research Roadmap — Milestone 2026.04.61

**Title:** JEPA v21 Multi-Source Real Data + Embedding Constraint Retrieval + Gemma4 Unblocked

**CalVer:** 2026.04.61 (sequence increment from 2026.04.60)
**Planned Experiments:** Exps 793-805 (13 experiments)
**Date Designed:** 2026-04-24
**Prerequisite:** Milestone 2026.04.60 retro complete (Exp 792)

---

## What Milestone 2026.04.60 Proved

Milestone .60 (Exps 780-792) showed efficiency progress but failed on six of twelve success criteria:

**Wins:**
- GPU Zombie Killer deployed (Exp 780): kill_gpu_zombies() now mandatory before every model load
- EDU-PRM step selection validated (Exp 782): 31.6% uncertainty-selected steps exceed 0.30 gate
- S* energy prefilter (Exp 787): 75% test oracle call reduction with energy ranking
- EBM calibration (Exp 789): ECE reduced 67.6% via isotonic regression on Ising energy scores
- NPU mlir-aie toolchain installed (Exp 790): Xilinx NPU path unblocked

**Failures (direct carry into .61):**
- RETRO-JEPA-V20-NO-DATA: Exp 781 collected n_labeled=0 because CARNOT_FORCE_LIVE not set
- RETRO-JEPA-OOD: JEPA v20 OOD AUC=0.4467 — regression vs v19 (0.5667), 8 consecutive failures
- RETRO-028: Gemma4 OOM — RETRO-028 unresolved for 3rd milestone (4-step isolation not applied)
- RETRO-SOTA-GGUF-TIMEOUT: SOTA GGUF code repair timed out again at 90 min (2nd consecutive)
- RETRO-CONSTRAINT-ZERO-DELTA: Constraint addition delta=0.0 — scalar encoding doesn't work
- RETRO-KV260-TOOLS-UNAVAILABLE: yosys/nextpnr-ice40/icepack absent from PATH

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: FR-11 Tier 3 Self-Learning Still Not Deployed (8 Consecutive JEPA Failures)

JEPA OOD AUC has failed to reach 0.75 in 8 consecutive retrains (v13-v20). Three root causes
identified:

1. **Data starvation:** v20 trained on only 18 EDU-PRM selected pairs (n_labeled=0 from live GPU
   because CARNOT_FORCE_LIVE was unset). Single-source (Qwen3.5-0.8B, GSM8K) data doesn't
   generalize to different models or domains.

2. **Insufficiently contrastive training pairs:** Standard correct/incorrect FoVer pairs are too
   easy — the predictor learns to distinguish clearly-right from clearly-wrong. It fails OOD
   because it never sees subtly-wrong steps. CPMI (arXiv 2604.10660) provides hard negatives:
   plausible-but-wrong steps with CPMI score in the 0.2-0.6 range.

3. **Single-distribution training data:** All 57 real pairs from Exp 442 come from GSM8K q1-300
   with Qwen3.5-0.8B. OOD evaluation uses different questions and possibly different models.
   Fix: multi-source corpus (GSM8K + MATH-500 + HumanEval) × 2 models (Qwen3.5-0.8B + Gemma4).

**Fix plan:** Exp 797 collects 80+ real pairs from 3 diverse sources with CARNOT_FORCE_LIVE=1.
Exp 798 augments with CPMI hard negatives. Exp 799 retrains JEPA v21 with LambdaRank +
PROGRS outcome-conditioned centering.

### Gap 2: Constraint Addition Zero-Delta (RETRO-CONSTRAINT-ZERO-DELTA)

Exp 788 showed dynamic (memory-augmented) IsingEBM equals static baseline (delta=0.0).
Root cause: scalar keyword-count encoding of error patterns causes semantic interference
(arXiv 2601.15313). When "carry_errors=3" and "sign_errors=2" are encoded as scalars, they
cannot be meaningfully distinguished in the Ising coupling space.

**Fix plan:** Exp 800 implements EmbeddingConstraintStore using sentence-transformer embeddings
in SPO (Subject-Predicate-Object) format: {subject: "arithmetic_step_N", predicate: "violates",
object: "carry_propagation_rule"}. Orthogonality regularization prevents semantic collapse.
Exp 801 benchmarks the delta improvement over static baseline.

### Gap 3: No SOTA Code Repair Headline (2 Consecutive Timeouts + Gemma4 OOM)

The strongest credible result Carnot can publish is code repair via execution verification
(+3.0pp HumanEval from Exp 226). But SOTA GGUF code repair has timed out twice (Exp 769 at
120 min, Exp 785 at 90 min) because:
- GPU zombies occupy VRAM before model load (RETRO-028 root cause)
- Qwen3.6-35B-A3B Q4_K_M requires ~20 GiB VRAM; GPU 0 had ~15 GiB occupied
- 50 problems × 3.6 min budget each is too tight; batching at 10q allows per-batch checkpoints

**Fix plan:** Exp 795 applies the four-step GPU isolation protocol (kill_gpu_zombies + explicit
VRAM eviction + <500MB verification + GPU 1 routing). Exp 796 runs SOTA code repair with 25
problems (not 50) in 10q batches with per-batch checkpoint saves.

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
  |        + EmbeddingConstraintStore (SPO memory → dynamic constraints) [NEW .61]
  v
[Tier 3.5] JEPA v21 (OOD predictor, TARGET: AUC >= 0.75) [BLOCKED: v20 AUC=0.4467]

Self-Learning Loop (FR-11):
  Verification results → FOVER annotation → JEPA v21 training corpus
  Repair outcomes → EmbeddingConstraintStore (session memory → active constraints)
  EmbeddingConstraintStore → VerifyRepairPipeline (dynamic constraint injection)
```

---

## Phase Descriptions

### Phase 0: Governance (Exps 793-794) — CPU only

**Goal:** Address process failures from .60 before running any GPU experiments.

**Exp 793 — Manifest Full-Scope Enforcement Audit**
Verify and document all dequeue call sites in scripts/research_conductor.py that bypass manifest
exclusion. Write a structured patch spec (not the patch itself — we cannot modify the conductor
script) as a JSON artifact mapping each dequeue site to the required exclusion check.
RETRO-MANIFEST-FULL-SCOPE: provides the precise action items for human enforcement.

**Exp 794 — FPGA Toolchain Install + Verify**
Install yosys + nextpnr-ice40 + icestorm via pacman (RETRO-KV260-TOOLS-UNAVAILABLE).
Verify installation: `yosys --version`, `nextpnr-ice40 --version`, `icepack --version`.
If installed, attempt minimal synthesis of a 4-spin Ising test circuit as proof-of-concept.
This gates the KV260 synthesis in Phase 5 (Exp 804).

### Phase 1: GPU Unblock (Exps 795-796) — GPU required

**Goal:** Close RETRO-028 and RETRO-SOTA-GGUF-TIMEOUT in sequence.

**Exp 795 — Gemma4 OOM Fix v4** (RETRO-028, CARNOT_FORCE_LIVE=1)
Apply the four-step GPU isolation protocol from the .60 retro resolution path:
1. kill_gpu_zombies() (deployed in Exp 780)
2. Explicit VRAM eviction: pkill loop until nvidia-smi shows <500MB on target GPU
3. Verify <500MB VRAM used before any model load attempt
4. Load Gemma4-E4B-it on GPU 1 (cooler GPU, 8-10C thermal headroom)
Run 10 GSM8K questions with live GPU inference to confirm model produces valid outputs.
honest_verdict=retro_028_closed when n_valid_responses >= 8.

**Exp 796 — SOTA GGUF Code Repair v3** (RETRO-SOTA-GGUF-TIMEOUT, GATED on Exp 795)
Batch 25 HumanEval problems into 5 batches of 5. ExperimentTimeoutWatchdog at 90 min.
Checkpoint per batch (atomic JSON write). Use Qwen3.6-35B-A3B Q4_K_M GGUF.
Apply MARS margin gate (arXiv 2601.15498): skip test execution for high-margin outputs.
honest_verdict=code_repair_positive if pass@1_repair > pass@1_baseline on >=10 problems.

### Phase 2: JEPA v21 Real Data + Retrain (Exps 797-799) — GPU + CPU

**Goal:** Break the 8-consecutive-failure streak on JEPA OOD generalization.

**Exp 797 — JEPA v21 Multi-Source Data Collection** (RETRO-JEPA-V20-NO-DATA, CARNOT_FORCE_LIVE=1 mandatory)
Collect real CoT steps from 3 diverse sources:
- Source A: 60 GSM8K questions with Qwen3.5-0.8B (existing model, new question range q301-360)
- Source B: 30 MATH-500 questions with Qwen3.5-0.8B (different difficulty distribution)
- Source C: 30 HumanEval problems with Gemma4-E4B-it (code domain, different model)
FOVER Z3 annotator labels each step. Target: n_labeled >= 80 pairs across 3 sources.
Write to results/fover_labeled_steps_v21_multi.json.

**Exp 798 — CPMI Contrastive Pair Builder** (arXiv 2604.10660)
For each positive (step, correct) pair from Exp 797, generate hard negative alternatives:
- Sample 5 response variants for the same prefix using model temperature=0.9
- Select the variant with CPMI score in [0.15, 0.6] range (plausible but wrong)
- Write (prefix, positive_step, hard_negative_step) triples to training corpus
CPU-only. Target: 3x corpus augmentation ratio (80 pairs → 240 triples).

**Exp 799 — JEPA v21 Retrain + Cascade Deploy Gate** (FR-11 mandatory, GATED on Exp 797)
Retrain on multi-source corpus (Exp 797) + CPMI triples (Exp 798).
Loss: LambdaRank listwise (v18 approach) + PROGRS outcome-conditioned centering.
Evaluate: in-distribution AUC + OOD AUC (Exp 442 held-out set).
Gate: if OOD AUC >= 0.75 → wire into ThreeTierPipeline as Tier 3.5.
If gate fails: write failure analysis (which sources contributed most to OOD variance).

### Phase 3: Embedding Constraint Retrieval (Exps 800-801) — CPU only

**Goal:** Fix RETRO-CONSTRAINT-ZERO-DELTA with embedding-based retrieval (arXiv 2601.15313).

**Exp 800 — EmbeddingConstraintStore** (RETRO-CONSTRAINT-ZERO-DELTA, arXiv 2601.15313)
Implement SPO-format constraint encoding:
- Constraint as {subject: step_context, predicate: violation_type, object: rule_name}
- Encode each constraint via sentence-transformer (all-MiniLM-L6-v2, CPU-native)
- Orthogonality regularization during storage (project out components along existing constraint vectors)
- Retrieval by cosine similarity (not keyword match)
Test: retrieval AUC across 5 constraint types (carry/sign/unit/comparison/causal).
Export: EmbeddingConstraintStore, ConstraintSPOTuple from carnot.pipeline.

**Exp 801 — Embedding Constraint Addition Benchmark** (RETRO-CONSTRAINT-ZERO-DELTA follow-up)
Wire EmbeddingConstraintStore into VerifyRepairPipeline as the Tier 1 self-learning mechanism.
Run 200 GSM8K questions (synthetic_cpu mode) across 5 sessions.
Compare: static Ising baseline vs embedding-retrieved constraint addition.
Target: constraint_addition_delta > 0.0 (vs Exp 788 delta=0.0).
Self-learning (FR-11 Tier 1) criterion: precision increases across sessions.

### Phase 4: Self-Learning Relay + Publishing (Exps 802-803)

**Goal:** Close the FR-11 Tier 1 real data relay and resolve RETRO-HF-AUTH.

**Exp 802 — FR-11 Tier 1 Real Relay with Embedding Constraints** (self-learning mandatory)
Run 10-session test (50q each, synthetic_cpu) with EmbeddingConstraintStore active.
Track: constraints_added_per_session, precision_improvement, constraint_recall.
Tier 1 criterion (research-program.md): precision must increase monotonically across sessions.
honest_verdict=tier1_relay_works if precision non-decreasing AND delta>0 by session 5.

**Exp 803 — HuggingFace Publish v2** (RETRO-HF-AUTH)
Generate SOPS-encrypted HF_TOKEN configuration spec: SOPS key file + secrets structure.
Write models/hf_upload_commands.sh (huggingface-cli login + push for each model card).
If HF_TOKEN available: update 3 model READMEs (one per tier) with .60 results.
honest_verdict=hf_auth_documented (if only docs) or hf_models_published (if authenticated).

### Phase 5: Hardware + Retro (Exps 804-805)

**Exp 804 — KV260 Open-Source Synthesis Attempt** (RETRO-KV260-TOOLS-UNAVAILABLE, GATED on Exp 794)
If Exp 794 confirmed yosys+nextpnr-ice40+icepack installed:
Synthesize hardware/kv260/ising_sampler_v3.v using yosys (not Vivado).
Timing: N=32 (fits iCE40 HX8K), then N=64 if LUT budget allows.
Target: LUT utilization report + timing analysis.
honest_verdict=synthesis_clean if yosys completes without errors.

**Exp 805 — Milestone 2026.04.61 Operational Retrospective**
Standard retro format. Success criteria: 8 RETRO closures, JEPA v21 OOD>=0.75,
constraint_addition_delta>0, Gemma4 working, SOTA code repair positive, FPGA synthesis clean.

---

## Dependency Graph

```
Exp 793 (Governance)    → no dependency
Exp 794 (FPGA Install)  → no dependency
  ↓
Exp 795 (Gemma4 Fix)    → requires CARNOT_FORCE_LIVE=1
Exp 796 (SOTA Repair)   → GATED on Exp 795 honest_verdict=retro_028_closed
  ↓
Exp 797 (JEPA Data)     → requires CARNOT_FORCE_LIVE=1
Exp 798 (CPMI Pairs)    → can run in parallel with Exp 797, feeds Exp 799
Exp 799 (JEPA v21)      → GATED on Exp 797 n_labeled>=80
  ↓
Exp 800 (Emb Store)     → no dependency (CPU-only)
Exp 801 (Emb Bench)     → GATED on Exp 800 retrieval_auc>0.7
  ↓
Exp 802 (FR-11 Relay)   → GATED on Exp 801 delta>0
Exp 803 (HF Publish)    → no dependency
  ↓
Exp 804 (KV260 Synth)   → GATED on Exp 794 tools_installed
Exp 805 (Retro)         → all experiments complete
```

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| retro_028_closed | Exp 795 | honest_verdict=retro_028_closed (n_valid>=8) |
| sota_code_repair_positive | Exp 796 | pass@1_repair > pass@1_baseline |
| jepa_v21_data_adequate | Exp 797 | n_labeled >= 80 across >=2 sources |
| cpmi_augmentation_works | Exp 798 | augmentation_ratio >= 2.0x |
| jepa_v21_ood_viable | Exp 799 | ood_auc >= 0.75 (cascade deploy gate) |
| embedding_retrieval_works | Exp 800 | retrieval_auc > 0.70 across 5 constraint types |
| constraint_addition_positive | Exp 801 | constraint_addition_delta > 0.0 |
| tier1_relay_works | Exp 802 | precision non-decreasing + delta > 0 by session 5 |
| fpga_tools_installed | Exp 794 | yosys + nextpnr-ice40 + icepack on PATH |
| kv260_synthesis_attempted | Exp 804 | honest_verdict != blocked (gated on Exp 794) |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|---------|-------|
| Exps 795, 797 | GPU (CARNOT_FORCE_LIVE=1) | RTX 3090 preferred; GPU 1 for Gemma4 (thermal) |
| Exp 796 | GPU (CARNOT_FORCE_LIVE=1) | Qwen3.6-35B requires ~20 GiB; GPU 1 isolated |
| Exps 793, 794, 798-803, 805 | CPU only | JAX_PLATFORMS=cpu |
| Exp 804 | CPU (yosys) | No GPU needed; uses open-source FPGA tools |

---

## Open RETROs Addressed

| RETRO | Addressed By | Expected Status |
|-------|-------------|-----------------|
| RETRO-028 | Exp 795 | CLOSED if n_valid >= 8 |
| RETRO-JEPA-V20-NO-DATA | Exp 797 | CLOSED if n_labeled >= 80 |
| RETRO-JEPA-OOD | Exp 799 | CLOSED if ood_auc >= 0.75 |
| RETRO-SOTA-GGUF-TIMEOUT | Exp 796 | CLOSED if code_repair_positive |
| RETRO-CONSTRAINT-ZERO-DELTA | Exp 800-801 | CLOSED if delta > 0 |
| RETRO-KV260-TOOLS-UNAVAILABLE | Exp 794 | CLOSED if tools installed |
| RETRO-HF-AUTH | Exp 803 | PARTIALLY: documented; CLOSED if authenticated |
| RETRO-MANIFEST-FULL-SCOPE | Exp 793 | DOCUMENTED: patch spec written for human application |

---

## New Papers to Incorporate

| Paper | arXiv ID | Used In |
|-------|---------|---------|
| Semantic Interference / SPO constraint encoding | 2601.15313 | Exp 800 |
| CPMI contrastive MI for process rewards | 2604.10660 | Exp 798 |
| ExecVerify execution-trace code constraints | 2603.11226 | Exp 801 |
| PROGRS outcome-conditioned centering | 2604.02341 | Exp 799 |
| MARS margin-aware speculative verification | 2601.15498 | Exp 796 |
