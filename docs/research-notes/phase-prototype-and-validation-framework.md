# Carnot Phase Prototype + Validation Framework

**Date:** 2026-04-30
**Trigger:** user observation after Phase-3 architecture
blind-spot audit caught 5 fatal flaws three rigorous theoretical
rounds missed: *"unless we have adversarial checks to ensure a
solid foundation for each phase we are building a house of cards
that cannot function in the end."*

This document operationalizes that lesson. For each Carnot phase
and sub-phase, define:

1. **Software prototype** — what code do we build? (concrete artifact)
2. **Empirical validation criteria** — what measurable data tells
   us pass/fail? (numbers, not opinions)
3. **Adversarial check** — what hostile-reviewer / red-team /
   null-space-attack process catches blind spots BEFORE we commit
   to scale?

The current state is **architecture-heavy / prototype-light /
adversarial-check-rare**. This document identifies the gaps and
proposes a remediation cadence.

---

## Phase 1 — Verification (current, shipping)

### Phase 1a — CPU verifier on (input, output) pairs

**Prototype:** Already shipping. `python/carnot/verify/`. Energy
function E(input, output) returning a scalar. Hand-coded constraint
verifiers (Z3 SAT, AST, type-check, etc.).

**Empirical validation criteria:**
- AUROC > 0.72 on FoVer-class reasoning corpus (per CLAUDE.md target)
- exp1077 / exp1079 measured this; 1077 reported AUROC numbers
- HumanEval / GSM8K downstream task pass-rate improvement when
  Carnot verifies-and-rejects vs no-verifier baseline

**Adversarial check status: WEAK.**
- We know about null-space mimicry attack
  (`project_null_space_mimicry_attack.md`) and
  orthogonality stall (`project_orthogonality_stall.md`)
  theoretically.
- We have NOT empirically tested current Phase 1a verifiers against
  adversarial inputs designed to fool them.
- **Gap:** an experiment that takes the current verifier suite,
  generates adversarial outputs (e.g., "look like good code" but
  fail on hidden constraint), measures false-pass rate. If
  false-pass > X%, the verifier ships with a known blind spot.

**Recommended .85+ task:** *Adversarial Verifier Robustness Audit*.
Take the current k verifiers, run 1000 GPT-generated adversarial
outputs through them, measure false-pass rate. Threshold for
shipping: < 5% false-pass on canonical attack patterns.

### Phase 1b — GPU-accelerated verifier

**Prototype:** Same energy function, GPU-accelerated inference path.
exp259 (onnxruntime CUDA EP) covers this.

**Empirical validation criteria:**
- GPU latency vs CPU latency at batch sizes 1, 32, 256
- Throughput (verifications/sec) on canonical workload
- Numerical agreement with CPU implementation (max divergence < ε)

**Adversarial check status: COVERED** by Phase 1a (same energy
function, just hardware-accelerated).

### Phase 1c — Multi-verifier AND-composition (k=15+)

**Prototype:** **NOT YET BUILT.** Round 9 architecture said k=15+
verifiers AND-composed via energy summation. We have 4-6 individual
verifiers shipped but no AND-composition layer.

**Empirical validation criteria:**
- Joint pass rate on adversarial inputs designed to defeat any
  single verifier
- Orthogonality measurement: rank of the verifier ensemble's
  combined energy gradient
- α_t survival on combined energy (per
  `project_zenil_alpha_grounding.md`)

**Adversarial check status: ABSENT.**
- The orthogonality stall theorem (`project_orthogonality_stall.md`)
  predicts a single verifier has a compute-immune ceiling. k=15+
  is the proposed solution.
- We have NOT measured whether 15 verifiers ACTUALLY have orthogonal
  null spaces, or whether they share pathological joint null spaces
  (per `project_pathological_joint_null_space.md`).
- **Gap:** explicit experiment measuring `dim(∩_i ker E_i)` for the
  shipping verifiers. If joint-null-space dimensionality > 0,
  AND-composition has structural blind spots regardless of k.

**Recommended .85+ task:** *Verifier Joint Null-Space Measurement*.
For the existing 4-6 verifiers + 9 more from Phase-3 candidates,
generate corruption corpus, measure which corruptions evade ALL k
verifiers. Output: empirical joint-null-space dimensionality
estimate. Acceptance: dim < 5% of input space.

---

## Phase 2 — Hardware acceleration

### Phase 2a — KV260 FPGA proof-of-concept (in progress)

**Prototype:** Shipped in exp1041 (first-light), exp1068
(smoke-test), exp1081 (scale benchmark). KV260 board, sampler running
synchronous parallel Glauber on quadratic Ising.

**Empirical validation criteria — REVISED 2026-04-30 per audit:**
- Latency vs CPU (exp1081 done, with caveat)
- Latency vs GPU (PENDING — was the original .85 task,
  now superseded by sampler-correctness audit)
- **Sampler correctness:** KL(FPGA_samples ‖ correct_Gibbs_samples)
  on a small-N test problem with frustrated J. If KL > threshold,
  Finding #2 is empirically confirmed and exp1081's headline numbers
  carry a "different distribution" caveat.

**Adversarial check status: AUDIT DONE 2026-04-30 — 5 FATAL findings.**
- Finding #1: Dimensionality guillotine
- Finding #2: Synchronous Glauber limit-cycle collapse (BONUS,
  unprompted)
- Finding #3: Hopfield capacity mode collapse (~18 modes max)
- Finding #4: Taylor-induced spurious corners
- Finding #5: Higher-order logic eradication

**Recommended .85+ task:** *Phase-2 Sampler Correctness Audit*. As
described in the revised `ops/known-issues.md` entry. Specifically
measure KL divergence between FPGA samples and correct Gibbs
samples on a deliberately frustrated J matrix. Documents Finding
#2 empirically.

### Phase 2b — Extropic Z1 production hardware (future)

**Prototype:** TBD pending Z1 hardware availability and SDK.

**Empirical validation criteria:**
- Same metrics as Phase 2a (latency, sampler correctness, throughput)
- Plus: does Z1's primitive escape the 5 FATAL findings of
  KV260? (synchronous-parallel-Glauber, pairwise-only,
  Hopfield-capacity, Q8.8-quantization, dense-Hessian-fitting)

**Adversarial check status: NOT YET RUN.**
- The audit methodology that worked for KV260 transfers: when Z1
  specs are public, run a dedicated hostile-reviewer round on the
  Z1 architecture before committing prototype budget.
- **Gap:** waiting for Z1 specs to be public.

### Phase 2c — Photonic (research-grade, long horizon)

Prototype TBD, no near-term action.

---

## Phase 3 — Foundation model substrate

### Phase 3a — Small-scale prototype (NEXT IMMEDIATE WORK)

**Prototype: NOT YET BUILT.** Round 1-2-3 + audit specified the
recipe but no code exists yet:

- 100M params, RoBERTa-base init
- DBAE-EBM: encoder → bounded `z ∈ [-1,1]^d` → decoder
- Hard tanh STE on encoder pre-activations
- Deep neural energy `E(z)` — small NN (2-3 layers, hidden 1024)
- Mirrored Langevin Dynamics sampling
- 3-stage training: AE warmup → EBM warmup → asymmetric finetune
  with stop-gradient on z_fake → encoder
- VICReg/Barlow Twins regularization (variance + decorrelation)
- Masked-token reconstruction (30-50%)
- Denoising-AE training (5-10% Gaussian noise on z_data before decoder)
- Parallel Tempering at training (β ∈ {0.1, 0.4, 0.7, 1.0})
- PCD with 10K-state replay buffer + 5% noise injection
- Latent dim sweep d ∈ {256, 512}
- FoVer 6,500-pair corpus for training
- 10K training steps

**Empirical validation criteria (Round 1 + 2 + 3 acceptance gate):**

| # | Test | Threshold | Source |
|---|---|---|---|
| 1 | α_t survival | `inf_t α_t > 0.1` across 100 MLD steps | Round 1 + Zenil grounding |
| 2 | Bottleneck integrity | Decoder >85% joint-constraint pass rate after 100 MLD steps | Round 1 |
| 3 | Manifold dead-zone | Decoder output stays meaningful after 500-1000 MLD steps | Round 3 (revised) |
| 4 | VICReg saturation | `var(z) ≈ 1.0` (Rademacher saturation) | Round 3 |
| 5 | Verifier signal | Each verifier `E_i` independently ranks held-out test set with AUROC > 0.65 | NEW |
| 6 | AND-composition | Joint `Σ E_i(z)` ranks better than any single `E_i` (composition gain > 5%) | NEW |

**Adversarial check status: AUDIT DONE 2026-04-30.** The Round 1-2-3
architecture survived the audit (the FATAL findings were all about
the FPGA distillation step, which is now deferred). The remaining
DEGRADING findings (#6, #7, #8, #9) and COSMETIC #10 transfer to
the prototype as known caveats.

**Recommended pre-prototype adversarial check:** before committing
to even the small-scale FoVer prototype, run ONE dedicated
adversarial round on the *prototype-implementation level* (not
architecture level). Specifically: "find ways the prototype could
silently produce passing acceptance-gate numbers without actually
working." Examples: trivial encoder identities, decoder LM-prior
overpowering bottleneck, EBM converging to constants.

### Phase 3b — 1B-token-scale validation

**Prototype:** scale Phase 3a successful prototype to 1B tokens / 1B
parameters.

**Empirical validation criteria:**
- All Phase 3a gates STILL hold at 1B scale
- Plus: emergent reasoning benchmark performance (GSM8K, MATH,
  HumanEval) at competitive vs Kona at similar parameter count
- α_t survives across longer reasoning chains (1000+ MLD steps)

**Adversarial check status: NOT YET COMMISSIONED.**
- Need a dedicated adversarial round BEFORE committing 1B-scale
  compute spend. The prototype's success at 100M doesn't guarantee
  1B success — scale-frontier failures (see SSD-distillation +
  α_t-collapse interaction in `project_ssd_self_distillation.md`)
  may surface only at scale.

### Phase 3c — Full Kona-parity foundation model

**Prototype:** production-scale, with full k=15+ verifier
AND-composition, full continuous reasoning chain, deployable
inference.

**Empirical validation criteria:**
- Matches Kona on benchmark tasks at lower compute cost OR matches
  Kona compute at higher accuracy
- Verifier-grounded outputs reduce hallucination rate vs unverified
  baseline by measurable margin
- Sovereignty/decentralization: runs on consumer hardware (RTX 3090
  class) without cloud dependency

**Adversarial check status: NOT YET COMMISSIONED.** Pre-publication
adversarial review by external research community is the canonical
adversarial check at this phase.

---

## The strategic gap pattern

| Phase | Prototype exists? | Empirical criteria? | Adversarial check done? |
|---|---|---|---|
| 1a | ✅ (shipping) | ⚠️ (partial) | ❌ |
| 1b | ✅ | ✅ | ❌ |
| **1c** | **❌** | **partial** | **❌** |
| 2a | ✅ (POC) | ⚠️ (caveated) | ✅ (audit done) |
| 2b | ❌ (no Z1 yet) | speccable | ❌ |
| 2c | ❌ | ❌ | ❌ |
| **3a** | **❌ (next step)** | **✅** | **partial** |
| 3b | ❌ | ✅ | ❌ |
| 3c | ❌ | ✅ | ❌ |

Three categories of gap:
- **Empirical-criteria gaps**: we have theoretical pass/fail but no
  measurement instrumentation. Phase 1a is the most embarrassing —
  Carnot is shipping a verifier framework but we don't measure its
  adversarial robustness empirically.
- **Adversarial-check gaps**: only Phase 2a has had its adversarial
  audit. Phase 1a, 1c, 3a, 3b, 3c are wide open.
- **Prototype gaps**: Phase 1c (multi-verifier AND), Phase 3a (the
  thing we're about to build) — neither exists in code yet.

## The new discipline

**Before scaling any phase, run a dedicated adversarial check on
that phase's prototype-implementation, not just its architecture.**

The architecture-level adversarial check we did 2026-04-30 caught 5
fatal flaws THEORETICALLY. But those flaws would also have surfaced
if we'd just built the prototype and run it — Finding #2
(synchronous Glauber non-equilibrium) shows up as KL-divergence
measurements; Finding #3 (Hopfield capacity) shows up as decoded-
text repetition; Finding #4 (spurious corners) shows up as gibberish
output.

**Empirical instrumentation IS adversarial check at scale.** A
prototype that emits the right diagnostics will surface most
architecture-level flaws automatically. A prototype that doesn't
emit the right diagnostics will let architecture-level flaws ship
into production.

So the operational rule is:

1. **Every phase prototype must include diagnostic instrumentation**
   for ALL theoretical concerns the phase rests on. (α_t tracking,
   joint null-space measurement, decoded-text diversity, sampler
   KL divergence, etc.)
2. **Every phase prototype must have a hostile-reviewer round**
   before scaling.
3. **Every phase artifact must produce empirical pass/fail data
   visible to downstream phases** — so a Phase 3 prototype that
   relies on Phase 1c's k=15 AND-composition can VERIFY at
   integration time that the dependency holds.

## Immediate action items

The audit-driven re-scope identified these as the highest-priority
empirical / adversarial gaps:

1. **Phase 1a Adversarial Verifier Robustness Audit** — measure
   false-pass rate on adversarial inputs designed to fool the
   shipping verifier. .85 candidate.
2. **Phase 1c Verifier Joint Null-Space Measurement** — measure
   `dim(∩_i ker E_i)` empirically for the shipping verifier suite.
   .85 candidate.
3. **Phase 2a Sampler Correctness Audit** — KL divergence between
   KV260 FPGA samples and correct Gibbs samples on frustrated J.
   Already in revised `ops/known-issues.md` for .85.
4. **Phase 3a Pre-Prototype Adversarial Round** — find ways the
   prototype could silently pass acceptance-gate numbers without
   working. Before committing prototype implementation budget.
5. **Diagnostic instrumentation library** — single shared module
   for α_t tracking, joint-null-space estimation, KL-divergence
   measurement, decoded-text diversity scoring. Cross-phase utility
   beats one-off hacks. .85 infrastructure-class candidate.

## Cross-references

- `project_orthogonality_stall.md`
- `project_null_space_mimicry_attack.md`
- `project_pathological_joint_null_space.md`
- `project_zenil_alpha_grounding.md`
- `project_phase3_architecture_complete.md`
- `project_dbae_ebm_phase3.md`
- `project_ssd_self_distillation.md`
- `feedback_fpga_rescope_extropic_pivot.md`
- `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
