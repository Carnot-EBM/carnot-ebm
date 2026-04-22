# Research Roadmap — Milestone 2026.04.56

**Title:** Tier 2.1 Production Deploy + FR-11 Relay + Privacy Safety Integration

**CalVer:** 2026.04.56 (sequence increment from 2026.04.55)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.55 — "JEPA v18 LambdaRank + Pre-flight Optimization + VR 200q Credibility Run"

---

## Executive Summary

Milestone .55 delivered three strategic breakthroughs that unblock a cascade of
previously-gated work:

1. **JEPAReasonerProbe (Exp 726)** — AUC=1.0, latency p99=0.0248ms — first viable
   pre-generative verifier. Qualifies as Tier 2.1. This supersedes the blocked JEPA v18
   cascade path (v18 achieved AUC=0.5115, above random but below 0.75 gate). The Tier 2.1
   probe operates on question-end hidden states BEFORE generation, enabling a genuinely
   novel verification path.

2. **KAN Distillation v3 (Exp 724)** — AUROC=0.9078, gate passed. First KAN model ready
   for production Tier 0b deployment as a prompt-injection pre-filter.

3. **VarGran Gate (Exp 727)** — 60% Ising skip rate, FN delta=-0.042. Tier 3 Ising runs
   can be selectively bypassed for high-confidence responses, reducing compute without
   accuracy loss.

Additionally, RETRO-033 (VR never positive) is closed. The known-issues.md entry confirms
VR has been removed from the active roadmap pending a larger model (>= 7B) or different
architecture. The PSV root cause remains unknown after pool exhaustion was ruled out
(condition B slope=+0.007, WORSE than condition A).

Milestone .56 capitalizes on these breakthroughs: deploy Tier 2.1 in cascade, wire FR-11
relay through Tier 2.1, integrate KAN Tier 0b, diagnose PSV with alternative hypothesis,
deliver the user-pinned privacy safety experiments, and implement Tier 2 cross-session
constraint memory.

---

## What Milestone .55 Proved

| Experiment | Result | Implication |
|------------|--------|-------------|
| Exp 716 — Pre-flight v7 | incremental_mode=True, 0/554 tests selected | Incremental selection operational |
| Exp 717 — JEPA v18 LambdaRank | OOD AUC=0.5115 (above random, first time) | v18 above random but cascade gate (0.75) not met |
| Exp 718 — JEPA v18 Cascade | cascade_deploy_auc_fail | JEPA v18 not cascade-ready; Tier 2.1 is the path |
| Exp 719 — FR-11 Relay | gated_blocked | Blocked by Exp 718 failure; unlocked by Tier 2.1 |
| Exp 720 — VR 200q | vr_marginal (+0.51pp live GPU) | RETRO-033 closed; VR removed from roadmap |
| Exp 722 — PSV pool exhaustion | pool_exhaustion_not_confirmed (B slope=+0.007) | Root cause remains unknown |
| Exp 724 — KAN Distill v3 | AUROC=0.9078, gate passed | Ready for Tier 0b production |
| Exp 725 — SC-Energy v2 | AUC=0.606 (below 0.75 gate) | SC-Energy not yet cascade-ready |
| Exp 726 — JEPAReasonerProbe | AUC=1.0, latency p99=0.025ms | BREAKTHROUGH — Tier 2.1 candidate |
| Exp 727 — VarGran Gate | skip_rate=0.60, fn_delta=-0.042 | Tier 3 can skip 60% without harm |

---

## Three Biggest Gaps (PRD vs. Current State)

### Gap 1: FR-11 Autonomous Self-Learning Loop Not Operational (HIGHEST PRIORITY)

PRD FR-11 requires: Tier 2 detects violation → event bus → Tier 1 updates weights →
feedback improves future generations. This has been blocked across 15+ milestones by JEPA
never reaching AUC >= 0.75. JEPAReasonerProbe (Tier 2.1) now provides the violation signal
with AUC=1.0. The path: deploy Tier 2.1 (Exp 733) → wire FR-11 relay (Exp 734) → wire
Tier 2 cross-session memory (Exp 738). All three are sequential.

**New architecture for FR-11:** Tier 2.1 probe fires violation event → FR11EventBus →
Tier 1 PerModelFPTracker updates per-constraint weights → Tier 2 SessionMemory caches
violation type + question domain → ConstraintTemplateLibrary adds template after 5+
cached violations of same type. This is the first complete FR-11 relay path.

### Gap 2: Constraint Verification Has No Production-Ready Top-Tier Filter

The cascade still runs expensive Ising (Tier 3) on every query that passes Tier 2. KAN
Distill v3 (AUROC=0.9078) is ready for Tier 0b deployment as a prompt-injection pre-filter.
VarGran (60% Ising skip) is wired but not yet using Tier 0b pre-filter. Together, Tier 0b
+ VarGran should reduce end-to-end cascade cost significantly.

**New architecture for efficiency:** Tier 0b KAN (prompt-injection check) → if score > 0.5,
route to safety pipeline instead of verification cascade. This is orthogonal to constraint
verification and provides the "safety layer" listed in the PRD product roadmap.

### Gap 3: PSV Self-Play Degradation Root Cause Unknown

PSV (Propose-Solve-Verify) is the core mechanism for autonomous self-improvement. It has
been degrading for 3 consecutive milestones. Pool exhaustion theory was ruled out (Exp 722
showed rotating pool was WORSE). Without knowing why PSV degrades, the self-play loop
cannot be reliably improved. Two alternative hypotheses need testing:
- **Constraint specialization**: verifier overfits to arithmetic errors, loses generality
- **Gradient interference**: opposing constraint types produce conflicting weight updates

---

## Architecture Diagram (After Milestone .56)

```
Query Input
    │
    ▼
Tier 0a — CarnotThinkProbe (generative CoT verdict, skip=incorrect)
    │
    ▼
Tier 0b — KAN Prompt-Injection Classifier (AUROC=0.908, NEW in .56)
    │       If high-energy: route to safety pipeline, skip verification cascade
    ▼
Tier 0c — NUP Probe v4 (contrastive energy probe, AUC=1.0)
    │
    ▼
Tier 0d — HallucinationBasinDetector (latent basin depth)
    │
    ▼
Tier 0e — HalluField (thermodynamic instability, advisory only)
    │
    ▼
Tier 1 — SinkProbe (attention sink concentration)
    │
    ▼
Tier 2 — EORM (CoT energy reward model, 55M params)
    │
    ▼
Tier 2.1 — JEPAReasonerProbe (pre-gen hidden-state probe, 0.025ms, NEW in .56)
    │       Violation event → FR11EventBus → Tier 1 weight updater (FR-11 relay)
    │       Violation type → SessionMemory → ConstraintTemplateLibrary (Tier 2 memory)
    ▼
Tier 2.5 — SymCodeVerifier (executable arithmetic, AUC=0.804)
    │
    ▼
Tier 2.6 — HermesVerifierAdapter (step-boundary feedback)
    │
    ▼
Tier 2.7 — CausalReasoningVerifier (carry-forward entailment)
    │
    ▼
Tier 3 — Ising VerifyRepairPipeline (VarGran: skip 60% via EORM confidence gate)
```

---

## Dependency Graph

```
Exp 731 (Zombie Kill + Preflight v8)          — no dependencies (MANDATORY FIRST)
Exp 732 (Probe 5-Fold Cross-Validation)       — no dependencies, GPU required
Exp 733 (Tier 2.1 Cascade Integration)        — GATED on Exp 732 gate file
Exp 734 (FR-11 Tier 2.1 Relay)                — GATED on Exp 733 (FR-11 MANDATORY)
Exp 735 (KAN Distill v3 Tier 0b Integration)  — no dependencies
Exp 736 (PSV Specialization Diagnosis)        — no dependencies, CPU
Exp 737 (PSV Domain-Diverse Recovery)         — GATED on Exp 736 diagnosis
Exp 729 (PrivacyFilter KAN v1 Distillation)   — no dependencies, USER-PINNED
Exp 730 (PrivacyFilter KAN v1 Gate)           — GATED on Exp 729, USER-PINNED
Exp 738 (Tier 2 Cross-Session Memory)         — GATED on Exp 734 (FR-11 relay)
Exp 739 (Operational Retrospective)           — all other experiments complete
```

---

## Phase 0: Infrastructure Pre-flight

### Exp 731: GPU Zombie Kill + Pre-flight v8 + Manifest Enforcement Audit

**Goal:** Clean up .55 session state. Kill GPU 1 zombie (PID 368449, 24GB VRAM), validate
incremental test selection, audit manifest enforcement gap.

**Context:** .55 was the first dirty GPU close in 14 milestones. PID 368449 holds 24082MB
on GPU 1 at 0% utilization. Until killed, dual-GPU experiments cannot use GPU 1.
The conductor projection vs actual gap was 787 min in .55 — root cause: exclusion manifest
written at text level but never enforced at dispatch. This experiment cannot fix
research_conductor.py but can document exactly what code change is needed (file path, line
number, diff) and build a validate_manifest_at_dequeue() function for the conductor to call.

**Implementation:**
- Kill zombie: os.kill(368449, signal.SIGKILL) + verify GPU 1 VRAM < 100MB
- Validate incremental test selection: re-run Exp 716 logic, confirm 0 tests on clean diff
- Manifest enforcement gap: scan conductor dequeue logic (without modifying), document
  the exact insertion point for manifest filtering. Write results/manifest_fix_patch.txt
  with the required diff as a human-action item.
- Write conductor_manifest_validator.py: validate_manifest_at_dequeue(task_id) that
  checks conductor_exclusion_manifest.json before dispatching any task.

**Success criteria:**
- GPU 1 VRAM < 100MB after kill
- Incremental test selection confirmed working
- manifest_fix_patch.txt written with specific diff

**Deliverable:** `results/experiment_731_zombie_kill_preflight_v8.json`

**Hardware:** CPU only

---

## Phase 1: Tier 2.1 Validation and Deployment

### Exp 732: JEPAReasonerProbe 5-Fold Cross-Validation

**Goal:** Validate that JEPAReasonerProbe AUC=1.0 from Exp 726 is robust and not overfit.
The probe was trained on 800 samples, tested on 200 OOD. AUC=1.0 is suspiciously perfect.
5-fold stratified cross-validation with different data splits will confirm or refute.

**Method:**
- Split FoVer v2 + GSM8K 500-699 into 5 equal folds, stratified by step_correct label
- Train 5 probe instances (fold i held out, rest train), test on held-out fold
- Compute mean OOD AUC ± std across all 5 folds
- Also test: same probe on a completely different benchmark (MATH-500, 50 questions)
  to check domain transfer

**Gate logic:**
- PASS: mean_auc >= 0.75 AND std_auc < 0.15 → write results/tier21_gate.json {"gate": "pass"}
- FAIL: either condition fails → write {"gate": "fail", "reason": ...}

**Why AUC=1.0 might be real:** Qwen3.5-0.8B layer-16 hidden states at question-end encode
"constraint complexity" as a nearly linear subspace (arXiv 2512.19171 shows this for similar
models). The probe extracts this directly — no ambiguity about whether the constraint is
satisfied because the model already "knows" whether it will make an error.

**Deliverable:** `results/experiment_732_probe_xval.json`

**Hardware:** GPU (hidden state extraction via Qwen3.5-0.8B)

---

### Exp 733: Tier 2.1 Cascade Integration

**Goal:** Wire JEPAReasonerProbe as Tier 2.1 between EORM (Tier 2) and SymCodeVerifier
(Tier 2.5) in the production ThreeTierPipeline. Measure cascade latency and skip rate.

**Integration design:**
- Tier 2.1 fires AFTER EORM, BEFORE SymCodeVerifier
- Threshold: 5th percentile of FoVer v2 correct-step scores (calibrated to < 5% FP rate)
- If probe score > threshold: fire ViolationEvent to FR11EventBus (Exp 734 wires this)
- If probe score <= threshold: mark as "likely_correct", skip SymCode + HERMES + Causal
  (early-exit path — saves ~100-500ms per query)
- Latency benchmark: measure cascade with/without Tier 2.1 on 200q GSM8K

**Success criteria:**
- Tier 2.1 integrated and wired to FR11EventBus stub
- skip_rate_symcode >= 0.40 (40% of correct responses skip downstream tiers)
- FN delta < 0.05 (not missing violations by skipping)
- Probe latency p99 < 1ms (confirmed from Exp 726: 0.025ms)

**Gate for Exp 734:** writes results/tier21_cascade_gate.json

**Deliverable:** `results/experiment_733_tier21_cascade.json`

**Hardware:** GPU (Qwen3.5-0.8B forward pass for hidden state extraction)

---

## Phase 2: FR-11 Self-Learning Relay (MANDATORY)

### Exp 734: FR-11 Tier 2.1 Relay

**Goal:** Wire the FR-11 autonomous self-learning relay through Tier 2.1. When the probe
detects a violation, the event flows through FR11EventBus to the Tier 1 weight updater
(PerModelFPTracker) and to the Tier 2 cross-session memory cache (SessionMemory).

**This is the FIRST time FR-11 has a viable trigger.** All prior relay attempts were blocked
because the violation detector never reached the relay threshold. Tier 2.1 with AUC >= 0.75
(confirmed by Exp 732-733) provides a reliable violation signal.

**Implementation:**
- Implement FR11EventBus if not already present (from Exp 719 design)
- ViolationEvent: (query_id, step_index, energy_score, probe_confidence, timestamp)
- Tier 1 subscriber: PerModelFPTracker.on_violation() — increment constraint weight
- Tier 2 subscriber: SessionMemory.cache_violation(violation_type, question_domain)
  After 5+ cached violations of same type: call ConstraintTemplateLibrary.observe_pattern()
- Throttle: max 1 Tier 1 update per 10 queries (prevents weight thrash)
- Run 50q validation: measure relay_events_published, relay_events_acked, latency_p99

**Success criteria (FR-11 MANDATORY):**
- relay_events_acked >= 1 (at least one violation detected and acked)
- relay_latency_ms_p99 < 200ms
- fr11_relay_operational = True
- Tier 1 weight update confirmed (fp_rate_delta measured before/after)

**Deliverable:** `results/experiment_734_fr11_tier21_relay.json`

**Hardware:** GPU (Tier 2.1 requires Qwen3.5-0.8B forward pass)

---

## Phase 3: KAN Tier 0b Production Integration

### Exp 735: KAN Distill v3 Tier 0b Integration

**Goal:** Deploy models/kan_distill_v3_tier0b.safetensors as Tier 0b (prompt-injection
pre-filter) in the production cascade. This is the first production-ready safety classifier
derived from teacher distillation.

**Integration:**
- Wire KANTier0bClassifier before CarnotThinkProbe (Tier 0a) — catches prompt injection
  before any expensive operations
- If score > 0.5: route to safety pipeline (not verification cascade), return
  safety_violation verdict immediately
- FP gate: must have FP rate < 5% on benign GSM8K prompts (1000 questions)
- Latency: < 5ms CPU inference (from Exp 726: KAN inference is 2-3ms)

**Benchmark:**
- Condition A: cascade without Tier 0b (baseline)
- Condition B: cascade with Tier 0b
- Measure: safety_skip_rate (how many prompts caught early), FP rate on benign GSM8K

**Success criteria:**
- Tier 0b wired and functional in cascade
- FP rate on benign prompts < 0.05
- No regression in verification cascade AUC

**Deliverable:** `results/experiment_735_kan_tier0b_integration.json`

**Hardware:** CPU (KAN inference is CPU-native)

---

## Phase 4: PSV Root Cause Investigation

### Exp 736: PSV Constraint Specialization Diagnosis

**Goal:** Test the constraint specialization hypothesis for PSV degradation: the verifier
overfits to arithmetic error patterns in GSM8K, losing the ability to detect violations in
other domains. This produces degradation when evaluated on held-out questions from different
distributions.

**Controlled experiment:**
- Condition A: PSV 20 iterations on GSM8K only (arithmetic domain) — baseline
- Condition B: PSV 20 iterations with rotating domain pool: GSM8K (10q) + MATH-Algebra (5q)
  + ARC-Challenge (5q) per iteration
- Condition C: PSV 20 iterations with GSM8K BUT using constraint verifier trained on ALL
  domains (not GSM8K-specialized)
- Measure: fp_rate_trend_slope per condition on HELD-OUT GSM8K questions (100-199)

**Gate logic:**
- If condition B slope < condition A slope: domain diversity helps → write gate "pass"
- If condition C slope < condition A slope: verifier generalization helps → write gate "pass_verifier"
- If both B and C worse: specialization not the root cause → write gate "fail"

**Deliverable:** `results/experiment_736_psv_specialization.json`

**Hardware:** CPU only

---

### Exp 737: PSV Domain-Diverse Recovery (GATED on Exp 736)

**Goal:** If Exp 736 confirms specialization as the root cause, implement the fix.
Use domain-diverse question pool AND per-domain verifier ensemble.

**Implementation (only if Exp 736 gate passes):**
- Question pool: 100 questions stratified across 3 domains (GSM8K, MATH, ARC-Challenge)
- Per-domain constraint verifier: separate ConstraintTemplateLibrary instances per domain
- Router: assign each question to its domain verifier before PSV self-play
- Run 30 iterations, measure fp_rate_trend_slope

**Success criteria:**
- fp_rate_trend_slope < 0 (improving) OR abs(slope) < 0.0001 (stable)

**Deliverable:** `results/experiment_737_psv_domain_diverse.json`

**Hardware:** GPU (Qwen3.5-0.8B for PSV generation)

---

## Phase 5: Privacy Safety Integration (USER-PINNED)

### Exp 729: PrivacyFilter KAN v1 — True Teacher Distillation

**USER-PINNED for milestone 2026.04.56.** Do not evict or modify without flagging to user.

Distil openai/privacy-filter into a KAN student using the same pattern as Exps 690/710.
Full prompt is in research-roadmap.yaml. Key requirements:
- REQ-SAFE-011 invariant AUTO-ENFORCED (teacher_inference_duration_s >= corpus_size * 0.5)
- AUROC target >= 0.85 (stretch 0.90)
- Latency < 5ms CPU

**Deliverable:** `results/experiment_729_privacy_filter_kan_true_distillation.json`

**Hardware:** GPU (teacher inference)

---

### Exp 730: PrivacyFilter KAN v1 — Cross-Dataset Gate

**USER-PINNED for milestone 2026.04.56.** Do not evict or modify without flagging to user.

Strict cross-dataset gate: AUROC >= 0.90 AND per-dataset min_tp >= 1 AND precision >= 0.80
AND FN-rate on credit cards <= 0.02 AND OOD AUROC not > training by 0.05.

Full prompt is in research-roadmap.yaml.

**Deliverable:** `results/experiment_730_privacy_filter_kan_cross_dataset.json`

**Hardware:** CPU (scoring only)

---

## Phase 6: Step-Level Latent Probe + Tier 2 Cross-Session Memory (arxiv + Self-Learning)

### Exp 738: Step-Level Latent Probe + Tier 2 Cross-Session Memory (arXiv 2511.06209)

**Goal:** Two complementary improvements in one experiment.

**Part 1 — Step-Level Probe (arXiv 2511.06209 "Efficient Test-Time Scaling via Probing
Internal States"):**
- Current Tier 2.1: query-level probe, extracts ONE hidden state at question-end token
- Step-level upgrade: extract hidden states at each CoT step boundary, pool with
  max-pooling across steps, train probe on pooled features
- arXiv 2511.06209 shows <10M-param probes with step-level features achieve parity with
  much larger PRMs on math/planning/QA. Query-level probes miss individual step failures.
- Comparison: step-level AUC vs. query-level AUC (JEPAReasonerProbe baseline) on FoVer v2 OOD

**Part 2 — Tier 2 Cross-Session Memory (research-program.md Tier 2):**
- Wire SessionMemory.on_session_end() to consolidate violation_type → template_key mappings
- Wire SessionMemory.on_session_start() to replay cached templates into ConstraintTemplateLibrary
- Test: 3-session simulation: S1 (baseline), S2 (cache from S1), S3 (cache from S1+S2)
  Measure: precision_s1, precision_s2, precision_s3 — expect monotonic improvement
- Gate: fr11_tier2_relay_functional = True if any template from S1 fires in S2

**Hardware path:** GPU for step extraction (Qwen3.5-0.8B), CPU for memory relay.
Research-program.md Tier 2: "CPU + system memory for storage; FPGA for fast pattern
matching" — this experiment implements the CPU tier.

**Success criteria:**
- step_auc >= query_auc (0.75 baseline from Tier 2.1 probe) OR both >= 0.75
- fr11_tier2_relay_functional = True
- template_reuse_rate > 0 across sessions

**Deliverable:** `results/experiment_738_step_probe_tier2_memory.json`

**Hardware:** GPU (step hidden state extraction), CPU (memory relay)

---

## Phase 7: Retrospective

### Exp 739: Milestone 2026.04.56 Operational Retrospective

**Goal:** Measure wall time, strategic outcomes, feed improvements back into process.

**5 key questions:**
1. **FR-11**: Is the full Tier 2.1 → FR-11EventBus → Tier 1 → Tier 2 memory relay operational?
2. **Tier 2.1**: Did 5-fold CV confirm AUC >= 0.75 (not overfit)?
3. **KAN Tier 0b**: Is the prompt-injection filter deployed with FP rate < 5%?
4. **PSV**: Is the root cause identified and slope improving?
5. **Privacy Filter**: Is privacy_filter_v1 publication-ready?

**Deliverable:** `results/operational_retro_2026_04_56.json`

**Hardware:** CPU

---

## Open RETROs Addressed in .56

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-CRITICAL (JEPA cascade) | Primary path pivots to Tier 2.1 | Exp 732-733 |
| RETRO-072/073 (FPGA synthesis) | Human action: install Vivado/yosys | Not in .56 (human action) |
| RETRO-033 (VR never positive) | CLOSED in .55 | — |
| PSV degradation (unnamed) | Alternative hypothesis testing | Exp 736-737 |
| FR-11 relay gated | Unblocked by Tier 2.1 | Exp 733-734 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|---------|-------|
| Exp 731 | CPU | nvidia-smi + kill |
| Exp 732 | GPU 0 | Qwen3.5-0.8B hidden states |
| Exp 733 | GPU 0 | Same |
| Exp 734 | GPU 0 | Same |
| Exp 735 | CPU | KAN inference |
| Exp 736 | CPU | PSV simulation |
| Exp 737 | GPU 0 | Qwen3.5-0.8B PSV generation |
| Exp 729 | GPU 0 | Teacher inference |
| Exp 730 | CPU | Scoring only |
| Exp 738 | GPU 0 | Session inference |
| Exp 739 | CPU | Analysis only |

**NOTE on GPU 1:** After zombie kill in Exp 731, GPU 1 (RTX 3090) becomes available.
Experiments that benefit from DualGPU can use it: Exp 737 (Qwen on GPU 0 + verifier on GPU 1)
and Exp 729 (teacher on GPU 0, student training on GPU 1).

**NOTE on FPGA:** KV260 hardware is present. Vivado not installed. RETRO-072/073 remain
open pending human install of Vivado or yosys. The .56 milestone does not attempt FPGA
synthesis; that is a dedicated human action.

---

## Success Criteria Summary

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| FR-11 relay operational | relay_events_acked >= 1 | Exp 734 |
| Tier 2.1 validated | 5-fold mean AUC >= 0.75 | Exp 732 |
| Tier 2.1 cascade | skip_rate_symcode >= 0.40 | Exp 733 |
| KAN Tier 0b deployed | FP rate < 0.05 on benign | Exp 735 |
| PSV root cause found | condition B or C slope < A | Exp 736 |
| Privacy filter distilled | AUROC >= 0.85 | Exp 729 |
| Privacy gate passed | 5 conditions met | Exp 730 |
| Tier 2 memory functional | template_reuse_rate > 0 | Exp 738 |
| FR-11 Tier 2 relay | fr11_tier2_relay_functional=True | Exp 738 |
