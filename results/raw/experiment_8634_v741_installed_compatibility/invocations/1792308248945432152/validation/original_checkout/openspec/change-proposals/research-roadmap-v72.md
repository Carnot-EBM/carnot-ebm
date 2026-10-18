# Research Roadmap v72: Gate-Check Recovery + Math Repair + Symbolic-KAN + SC-Energy

**Milestone:** 2026.04.72
**Planned:** 2026-04-26
**Predecessor:** 2026.04.71 (2/12 criteria met, 3902 min total)
**Experiments:** Exps 929–940 (12 experiments)

---

## What Milestone .71 Proved

Milestone .71 met only 2/12 criteria — a structural planning failure, not an algorithm failure.

**Root Cause:** 7 of 12 experiments were immediately blocked at the conductor pre-gate because
their YAML tasks lacked `prior_failures` fields required by the rerun-discipline enforcement
system. The conductor's gate-checker now enforces no-doomed-rerun discipline across the ENTIRE
research history. Any task touching a domain with prior experiments (code repair, DualGPU
benchmarking, HF publishing, KAN training, FR-11, pipeline integration) is rejected unless its
YAML includes `prior_failures` entries naming the prior experiments, their verdicts, and what
is specifically different about this attempt.

**What .71 proved (genuine results):**

| Result | Experiment | Finding |
|--------|------------|---------|
| Lagrange multi-constraint works | Exp 918 | marginal_improvement: entropy delta +0.018 |
| DriftProbe ensemble no improvement | Exp 923 | OOD AUC 0.5625 vs 0.565 baseline — uniform weights HURT |
| R-PRM Tier 2.9 heuristic flat | Exp 924 | AUC delta = 0.0 — needs live model inference, not heuristics |
| Milestone retro complete | Exp 928 | milestone_complete |

**What .71 didn't accomplish (7 experiments blocked by planning error):**
- Exp 917: Pre-flight (no prior_failures for chain of 6 prior preflight experiments)
- Exp 919: Math iterative self-repair (no prior_failures for Exps 744/759/905/919)
- Exp 920: Combined pipeline (cascaded from Exp 919 block)
- Exp 921: DualGPU throughput (no prior_failures for 15 prior benchmark experiments)
- Exp 922: HF publish v4 (no prior_failures for Exps 803/829/915)
- Exp 925: KAN Tier 4 real data (no prior_failures for Exp 485)
- Exp 926: FR-11 Tier 2 code domain (no prior_failures for Exps 713/814/864)
- Exp 927: DraftConditioned integration (no prior_failures for Exps 467/912)

**Critical Lesson for .72:** Every task that touches a domain with ANY prior experiment history
MUST include `prior_failures` entries. The conductor's gate-checker scans the full research
history. Missing prior_failures = immediate block regardless of whether the experiment is
genuinely different from prior attempts.

---

## Architecture Diagram (Verification Pipeline)

No architectural changes from .71. Tiers 2.8 and 2.9 still pending full integration:

```
User prompt → LLM → raw response
                          ↓
           [Tier 0a] CarnotThinkProbe (fast CoT verdict)
                          ↓ (uncertain → continue)
           [Tier 0b] SpilledEnergyDetector (logit discrepancy)
                          ↓ (high spill → continue)
           [Tier 0c] NUPProbeV4 (bigram contrastive, AUC=1.0)
                          ↓ (low score → continue)
           [Tier 0d] HallucinationBasinDetector (basin depth)
                          ↓ (shallow basin → continue)
           [Tier 0e] HalluField (thermodynamic instability, advisory)
                          ↓
           [Tier 1] SinkProbe (attention sink concentration)
                          ↓ (uncertain → continue)
           [Tier 2] VJEPA v2 (CoT violation prediction, OOD AUC=0.9211)
                          ↓ (energy > threshold → continue)
           [Tier 2.5] SymCodeVerifier (arithmetic execution)
                          ↓
           [Tier 2.6] HermesVerifierAdapter (step-boundary feedback)
                          ↓
           [Tier 2.7] CausalReasoningVerifier (causal entailment)
                          ↓
    PENDING [Tier 2.8] DraftConditionedVerifier ← Exp 938 integrates this
    PENDING [Tier 2.9] R-PRM Step Reward (live model needed) ← future milestone
                          ↓
           [Tier 3] Ising VerifyRepairPipeline (full constraint verify)
                          ↓
           [Repair] IterativeSelfRepair (execute-feedback-retry)
                          ↓
          certificate + repaired response

Self-learning loop (FR-11):
Tier 1 violations → LagrangeMultiplierUpdater (forgetting curve confirmed by Exp 918)
                 → ConstraintTemplateLibrary (code domain: Exp 935 adds this)
                 → FoVer → VJEPA v2 retraining
                 → KAN Tier 4 adaptive structure (Exp 936 uses real FoVer data)
```

---

## Phase Descriptions

### Phase 0: Governance Audit (Exp 929)

Pre-flight v21 audits .71 outcomes, formally closes RETRO-LAGRANGE-ENTROPY-DEGENERATE (Exp 918
confirmed marginal improvement), documents the gate-check discipline lesson, and sets .72 gates.
**THIS EXPERIMENT MUST include `prior_failures` for all prior preflight experiments or it will
be blocked exactly as Exp 917 was.**

### Phase 1: Math Iterative Self-Repair (Exps 930-931)

Exp 905 proved IterativeSelfRepair works for HumanEval code: 4% baseline → 72% repair rate
(+68pp), live GPU with Gemma4-E4B-it. This phase extends the same execute-feedback-retry
approach to GSM8K math. The mechanism differs: instead of running Python code and checking test
cases, we parse the numeric answer from the LLM's response and compare to the ground truth.
Carnot's energy function ranks multiple repair attempts and selects the lowest-energy one.

Exp 931 (combined math+estimation pipeline) is gated on Exp 930 signed_improvement > 0.

### Phase 2: Infrastructure (Exps 932-934)

**Exp 932 (DualGPU):** Exp 913 wired DualGPU into ThreeTierPipeline and measured 1.4x speedup
on tiny synthetic workloads. Exp 921 was blocked by missing prior_failures. This experiment
runs a realistic parallel inference load (50 GSM8K questions) with CARNOT_DUAL_GPU=1 to confirm
the speedup on real workloads.

**Exp 933 (HF Publish v4):** Exp 915 confirmed model cards ready, gitea mirror confirmed, but
HF authentication blocked (hf_authenticated=false). This experiment injects SOPS-encrypted
credentials and executes the actual HuggingFace CLI upload for VJEPA v2 and EstimationVerifier.

**Exp 934 (IPFS Mirror):** ops/known-issues.md documents that VJEPA v2 published weights have
no IPFS mirror. CLAUDE.md rule 3 requires all published weights to have at least two independent
distribution channels. This experiment installs IPFS and establishes the mirror.

### Phase 3: Self-Learning + KAN (Exps 935-936)

**Exp 935 (FR-11 Tier 2 Code Domain):** Mandatory per research-program.md (every milestone
must include a self-learning experiment). Extends Tier 2 constraint memory from math/arithmetic
patterns to code repair patterns. Uses real code repair data accumulated by Exp 905. Exp 864
confirmed FR-11 Tier 2 relay functional for math; this extends to code.

**Exp 936 (KAN Tier 4 Real Data):** Exp 910 (AutoKnots seed) validated adaptive spline
refinement on synthetic data. This experiment applies the same approach to real FoVer-labeled
violation pairs from the live corpus, testing whether real data produces better constraint
structure than synthetic.

### Phase 4: Research New Ground (Exps 937-939)

**Exp 937 (Symbolic-KAN):** Based on arXiv 2603.23854 (Symbolic-KAN, April 2026). Implements
a constraint verifier using KAN with discrete symbolic structure: each spline node maps to a
symbolic operation (ADD, MUL, CMP, EQ) enabling interpretable constraint checking. Compared
to standard spline KAN, symbolic structure should improve AUC on structured math reasoning
verification where the constraint types are known.

**Exp 938 (DraftConditioned Tier 2.8 Integration):** Exp 912 proved DraftConditioned Tier 2.8
is viable standalone (AUC 0.42 → 0.48, signed_energy_improvement positive). This experiment
wires it into ThreeTierPipeline between Tier 2.7 and Tier 3, so that the draft scaffold is
automatically applied when Tier 2.7 flags causal uncertainty.

**Exp 939 (SC-Energy Set Consistency):** Based on arXiv 2503.10695 (Set Consistency Energy
Networks, March 2026). Implements set-level coherence checking: given a set of statements
{s1, ..., sn} from a response, the energy function scores whether the set is internally
consistent vs contradictory. Contrastive training on FoVer pairs. This is complementary to
Carnot's single-step verification — it catches contradictions that span multiple statements.

### Phase 5: Retrospective (Exp 940)

Milestone retro documenting criteria met, lessons learned, and open RETROs for .73.

---

## Success Criteria

| # | Criterion | Experiment | Target |
|---|-----------|------------|--------|
| 1 | `preflight_complete` | Exp 929 | honest_verdict = 'preflight_complete' |
| 2 | `math_repair_working` | Exp 930 | signed_improvement > 0 |
| 3 | `combined_pipeline_viable` | Exp 931 | combined_accuracy > baseline (if gate open) |
| 4 | `dualgpu_throughput_confirmed` | Exp 932 | speedup >= 1.4x (Exp 913 baseline) |
| 5 | `hf_published` | Exp 933 | hf_authenticated = True, upload confirmed |
| 6 | `ipfs_mirror_established` | Exp 934 | ipfs_cid != None |
| 7 | `tier2_code_memory_works` | Exp 935 | honest_verdict in ('tier2_code_memory_works', 'partial') |
| 8 | `kan_tier4_real_data` | Exp 936 | honest_verdict != 'blocked' |
| 9 | `symbolic_kan_viable` | Exp 937 | auc > 0.70 |
| 10 | `tier28_wired` | Exp 938 | honest_verdict = 'tier28_wired' |
| 11 | `sc_energy_viable` | Exp 939 | auc > 0.70 |
| 12 | `retro_complete` | Exp 940 | honest_verdict = 'milestone_complete' |

---

## Hardware Requirements

| Experiment | Hardware | Required |
|------------|----------|---------|
| Exp 930 (Math Repair) | RTX 3090, GemmaTransformersLoader | CARNOT_FORCE_LIVE=1 |
| Exp 932 (DualGPU) | Both RTX 3090s | CARNOT_DUAL_GPU=1 |
| All others | CPU only | JAX_PLATFORMS=cpu |

---

## Dependency Graph

```
Exp 929 (preflight v21)
├── Exp 930 (math repair, GPU, CARNOT_FORCE_LIVE=1)
│   └── Exp 931 (combined math pipeline, CPU, GATED on Exp 930 signed_improvement > 0)
├── Exp 932 (DualGPU throughput, GPU, CARNOT_DUAL_GPU=1)
├── Exp 933 (HF publish v4, CPU)
│   └── Exp 934 (IPFS mirror, CPU, after Exp 933)
├── Exp 935 (FR-11 Tier 2 code, CPU) — mandatory self-learning
├── Exp 936 (KAN Tier 4 real data, CPU)
├── Exp 937 (Symbolic-KAN, CPU) — new research
├── Exp 938 (DraftConditioned integration, CPU)
└── Exp 939 (SC-Energy set consistency, CPU) — new research
└── Exp 940 (milestone retro)
```

---

## New Research References

### Symbolic-KAN: Discrete Symbolic Structure for KAN Interpretability
- **Paper:** arXiv 2603.23854 (April 2026)
- **What:** Augments KAN splines with discrete symbolic node labels (ADD, MUL, CMP, EQ) drawn from
  a predefined vocabulary. Each activation function is constrained to follow its symbolic label's
  behavior, making the learned constraint function interpretable. Achieves 94% symbolic accuracy on
  arithmetic benchmark tasks.
- **Relevance:** Carnot's KAN tier currently uses pure splines with no symbolic structure. Symbolic
  labels would make the energy function interpretable: "this node checks addition constraints."
  Interpretability is key for debugging false positives and building user trust.
- **Experiment:** Exp 937 (Symbolic-KAN Constraint Verifier)

### SC-Energy: Set Consistency Energy Networks
- **Paper:** arXiv 2503.10695 (March 2026)
- **What:** Energy function that scores whether a SET of statements {s1,...,sn} is internally
  consistent. Contrastive loss: E(coherent_set) << E(contradictory_set). Achieves 0.89 AUROC on
  multi-statement consistency detection. Requires no explicit logical parsing — consistency is
  learned from energy landscape.
- **Relevance:** Carnot's global consistency checker (100% detection in Exp 172) uses explicit
  logical rules. SC-Energy provides a learned alternative: train on FoVer pairs, let the energy
  function learn what "consistent" means for math reasoning. Complement to per-step verification.
- **Experiment:** Exp 939 (SC-Energy Set Consistency Verifier)

---

## Arxiv Scan Results (2026-04-26 Planning Session)

Papers filed to research-references.md:
- arXiv 2603.23854 — Symbolic-KAN (April 2026): Symbolic discrete structure for KAN interpretability
- arXiv 2503.10695 — SC-Energy (March 2026): Set consistency energy networks via contrastive loss
- arXiv 2604.20659 — GRPO-VPS: Verifiable process supervision for step-level reasoning
- arXiv 2505.14999 — EORM: Energy Outcome Reward Model, lightweight post-hoc verifier
- arXiv 2604.19305 — DebugRepair: Self-directed debugging for program repair (complements Exp 905)

---

## Decentralization Implications

All experiments in this milestone satisfy CLAUDE.md decentralization rules 1-7:
- Models: Gemma4-E4B-it (open weight, local via GemmaTransformersLoader)
- Hardware: local RTX 3090s (no cloud compute required)
- Distribution: HF publish + IPFS mirror (Exp 934) establishes second channel
- Core verifier stack: no vendor SDK imports
- Closed-weight models: not used in any experiment

---

## Open RETROs Entering .72

| RETRO | Status | Action |
|-------|--------|--------|
| RETRO-MANIFEST-FULL-SCOPE | HUMAN_REQUIRED | Requires modifying research_conductor.py |
| RETRO-XILINX-TOOLS-UNAVAILABLE | HUMAN_REQUIRED | Requires Vivado install |
| RETRO-LAGRANGE-ENTROPY-DEGENERATE | **CLOSED** by Exp 918 (marginal improvement confirmed) |

---

## Predecessor Context

- **Milestone .70:** 11/12 criteria in 36.9 min — project's best criteria density ever
  - IterativeSelfRepair v1 (code): 4% → 72% pass rate (+68pp) [Exp 905]
  - EstimationVerifier SVAMP AUC: 0.125 → 0.90 [Exp 906]
  - DraftConditioned Tier 2.8 viable: AUC 0.42 → 0.48 [Exp 912]
  - DualGPU wired: ThreeTierPipeline.wire_dual_gpu_runner() + 1.4x speedup [Exp 913]
  - KAN Tier 4 seed: AutoKnots adaptive refinement validated (synthetic) [Exp 910]

- **Milestone .71:** 2/12 criteria — 7 experiments blocked by missing prior_failures fields
  - Lagrange forgetting multi-constraint: entropy delta +0.018 (marginal, algorithm confirmed) [Exp 918]
  - DriftProbe ensemble: WORSE than single probe (uniform weights suboptimal) [Exp 923]
  - R-PRM heuristic: zero improvement (needs live model inference) [Exp 924]
