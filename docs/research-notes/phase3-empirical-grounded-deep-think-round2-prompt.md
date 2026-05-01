# Deep Think Round 2 Prompt — Adversarial Follow-up to Phase-3 Empirical-Grounded Round 1

**Status:** Ready to send. Round 1 returned a strong "BLOCK Phase-3"
recommendation with specific prescriptions (k=5-7 ceiling,
2-stage training, 0.8B model tier, Block-and-Resolve workflow).
Per the project's documented Deep-Think prediction pattern
("qualitative survival claims well-calibrated, specific prescriptions
systematically wrong" — `feedback_carnot_prediction_pattern.md`),
those prescriptions deserve adversarial cross-validation before
becoming policy.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session (NOT a continuation
of Round 1). Paste the section labelled `## Prompt to send (verbatim)`.
Do NOT include Round 1's prior reasoning — let Round 2 reason
independently.

---

## Prompt to send (verbatim)

### Background

A prior Deep Think round (Round 1, dated 2026-05-01) produced the
following five specific prescriptions for the Carnot project's
Phase-3 DBAE-EBM (Deterministic Bounded Autoencoder + Latent EBM)
prototype. Each prescription is provided here AS A STARTING-POINT
HYPOTHESIS that we want to stress-test, NOT as established fact.

**Prescription P1: AND-composition feasibility ceiling is k=5-7,
NOT k=15.** Rationale: the LLM output manifold has intrinsic
dimensionality D_int ≈ 5 for "correctness failure modes" (syntax,
logic, semantic reality, execution trace). By pigeonhole, k > 7
verifiers must become linear combinations of existing ones,
forcing pairwise correlation r → 1.

**Prescription P2: AND-composition over k=5 with r<0.3 is
mathematically superior to k=15 with r=0.55** because joint
volume shrinks as ~O(cos^k(θ_F)) and r ≈ cos(θ_F).

**Prescription P3: Phase-3 training schedule must be REVISED from
3-stage to 2-stage** (DBAE pre-train → Joint Verifier-Guided
Contrastive Fine-Tune, instead of DBAE pre-train → EBM training →
joint fine-tune). Rationale: isolated EBM training cannot penalize
Finding-#4 spurious decoder corners.

**Prescription P4: Phase-3 prototype must validate at 0.8B
parameter tier (Qwen3.5-0.8B), NOT the SOTA 35B tier
(Qwen3.6-35B-A3B-GGUF or Gemma-4-31B-it-GGUF).** Rationale:
isolating architectural validation from LLM memorization
confounders.

**Prescription P5: Phase-3 prototype construction must be BLOCKED
until Phase-1d (Verifier Diversity Expansion) ships a k=5 suite
with r<0.3.** Sequential workflow; no parallel Phase-1d / Phase-3
work allowed.

### Project-wide design constraints (do not contradict)

These are mandatory project rules from CLAUDE.md. Round 2 must
respect them:

- **MANDATORY SOTA model tier:** "New experiments that need an LLM
  must include at least one of these three state-of-the-art
  GGUF-quantized local models: unsloth/Qwen3.6-35B-A3B-GGUF,
  unsloth/gemma-4-31B-it-GGUF, unsloth/gemma-4-26B-A4B-it-GGUF.
  Keep Qwen3.5-0.8B / Gemma4-E4B only for cheap CPU smoke-tests or
  reproduction runs; they are not acceptable as headline-result
  models." This contradicts Prescription P4.

- **Decentralization rule 1:** local-first using open models always.
  All open-weight model tiers above qualify.

- **Phase Prototype + Empirical Validation + Adversarial Check
  Discipline:** every phase prototype must include software
  prototype + measurable pass/fail tests + hostile-reviewer round
  before scaling.

### Established empirical evidence (use as facts)

- **exp1093** (2026-05-01): pairwise r-correlation = 0.66 across 3
  text probes (NUP / SpilledEnergy / KAN-based SOSKANEnergyV3)
  on 200-pair text corpus; joint null-space fraction = 0.0.
- **exp1094** (2026-05-01): KL(P_parallel_glauber || P_correct_gibbs)
  = 3.07 nats on 12-spin frustrated antiferromagnetic ring.
- **exp1100** (2026-05-01): cascade validates SOTA outputs but
  Pareto-suboptimal cost structure (`cascade_validated_sota_inefficient`).
- **exp1099** (2026-05-01): RLVR+SSD on pre-filtered corpus
  produced `no_improvement_honest_negative` because all energy
  scores were degenerate.
- **exp1081** (2026-04-30): FPGA-vs-CPU at N=64 measured 13,061×
  speedup with the now-empirically-falsified parallel Glauber.

### The Round 2 questions

Round 2 should reason INDEPENDENTLY (do not build on Round 1's
chain). Treat each prescription as a hypothesis to confirm,
refute, or modify.

#### Q1 (P1, P2 stress-test): k-ceiling and Friedrichs-angle scaling

**Q1a.** The "intrinsic dimensionality D_int ≈ 5 for correctness
failure modes" is a heuristic claim. Derive a tight upper bound
on k_max using a measured proxy for D_int (e.g., the rank of the
verifier-output covariance matrix on a held-out corpus). For
exp1093's data (3 verifiers, 200-pair corpus, r=0.66), what does
this give?

**Q1b.** The claim that "joint null-space volume shrinks as
O(cos^k(θ_F))" assumes uniform pairwise correlation. In reality,
if 3 of k=15 verifiers are tightly correlated (r=0.7) and 12 are
weakly correlated (r=0.2), what is the actual joint-null-space
shrinkage? Is the geometric-mean approximation cos^k(θ_F) within
2× of the truth? Within 10×?

**Q1c.** Adversarial: is there a verifier-design strategy that
breaks the D_int=5 ceiling? E.g., adding a verifier that depends
on EXTERNAL state (web search, code execution, theorem prover),
making D_int effectively unbounded. Does this rescue k=15, or
does the LLM-output-manifold still bottleneck?

**Q1d.** Empirical experiment specification: design a single
Carnot experiment (~1 milestone) that produces an empirical lower
bound on k_max with high confidence. Specify: corpus, verifier
candidates, metric, sample size, statistical test.

#### Q2 (P3 stress-test): training schedule

**Q2a.** The claim that "isolated EBM training cannot penalize
Finding-#4 spurious decoder corners" is theoretical. What is the
specific empirical test that distinguishes:

1. 3-stage with adversarial-corner regularization in stage-3 (works)
2. 3-stage without such regularization (fails per Round 1)
3. 2-stage joint training (works, claims Round 1)

If 3-stage with stage-3 regularization is empirically equivalent
to 2-stage joint, the prescription P3 is over-constrained.

**Q2b.** The 2-stage joint training requires the verifier cascade
to generate live negatives during DBAE fine-tuning. If the cascade
itself is the k=5 r<0.3 suite (per P5), the negatives are
generated by Z3-AST + runtime-execution + combinatorial-encoding
+ 2 others. What's the per-step cost? Does it make the joint
fine-tune wall-time-prohibitive for the 0.8B tier?

**Q2c.** Adversarial: is there a 4-stage training that's safer than
either? E.g., DBAE pre-train → EBM train (warm-up) → joint
fine-tune (decoder-frozen) → joint fine-tune (full).

#### Q3 (P4 stress-test): model tier validation

**Q3a.** Round 1 recommended 0.8B for "isolating architectural
validation from LLM memorization." But the project's CLAUDE.md
explicitly prohibits 0.8B as headline-result tier (only valid for
CPU smoke-tests). Is there a third path: validate at 0.8B for
methodology (smoke tier) AND 35B for headline results? What's
the per-experiment cost ratio?

**Q3b.** Is there an architectural reason 0.8B WILL produce
non-degenerate Phase-3 prototype behavior? Or is "small model
captures the architecture" itself a hypothesis that needs
empirical validation?

**Q3c.** Adversarial: if 0.8B produces successful Phase-3 prototype
(final_energy=0, AUROC>0.95), is that meaningful evidence the
architecture works at scale? Or is it a model-class confound that
masks scale-specific failures?

#### Q4 (P5 stress-test): Block-and-Resolve workflow

**Q4a.** The Block-and-Resolve workflow forces Phase-1d strictly
before Phase-3 prototype work. Is there a parallel-track workflow
where Phase-3 prototype scaffolding (DBAE encoder/decoder + EBM
schema + training-loop infrastructure) is built in parallel with
Phase-1d (verifier diversity expansion), with the actual EBM
training deferred until Phase-1d ships? This would compress
calendar time without violating the empirical-precedence principle.

**Q4b.** What is the smallest-scope Phase-1d that would unblock
Phase-3? E.g., is k=5 sufficient or must k=7? Must r<0.3 be
worst-case-pairwise or average-pairwise? Maximum permissible
correlation matrix structure?

**Q4c.** Adversarial: if Phase-1d takes 3 milestones to ship k=5
with r<0.3, and the publication target is 2026-05-15 (2 weeks
out), should we ship Phase-3-with-r=0.66-cautioned NOW with the
correlation finding documented as a known limitation, OR wait
3 milestones for the corrected suite? The honest paper-submission
calculus matters.

#### Q5 (Risk register stress-test)

Round 1 produced 5 silent-failure risks (dimensionality guillotine,
spurious corner exploitation, hardware thermodynamic drift,
sub-space mode collapse, contrastive hard-negative washout).

**Q5a.** Are there silent failures Round 1 missed? Specifically
consider:

- **Verifier capture by training:** the EBM during contrastive
  fine-tune learns to satisfy the k=5 verifiers exploitatively
  (Goodhart's Law on the cascade itself).
- **Adversarial-hard-negatives compositional bias:** the adversarial
  negatives in the 1:1:1 corpus mix are themselves model-generated
  → systematic distribution shift the EBM optimizes for instead
  of the real adversarial distribution.
- **Verifier monotonicity violation:** at training time, scaled-up
  verifiers (k=5 deployed) produce slightly different verdicts
  than slimmer training-time proxies → drift between train and
  inference.

**Q5b.** For each silent-failure risk (Round 1's 5 + your Round 2
additions), specify the simplest test that DETECTS the failure
within 1 hour of GPU compute on an 0.8B (or 35B) model. Cheap
diagnostics > expensive ones.

### What Round 2 should NOT do

- Do NOT defer to Round 1's reasoning. Reason independently.
- Do NOT propose new theoretical defense layers.
- Do NOT recommend "more research is needed" without specifying
  the experiment that would resolve the question.
- Do NOT contradict CLAUDE.md design rules (specifically the
  SOTA-model-tier mandate, decentralization rules, phase-validation
  discipline).

### Output format

1. **Executive summary** — 1 paragraph naming which Round 1
   prescriptions you confirm, modify, or refute, and the
   confidence level for each.
2. **Q1 answer** with k-ceiling derivation and Friedrichs-angle
   correctness check.
3. **Q2 answer** with training schedule comparison.
4. **Q3 answer** with model tier reconciliation against CLAUDE.md.
5. **Q4 answer** with parallel-track feasibility analysis.
6. **Q5 answer** with new silent-failure risks + cheap diagnostics
   for each.
7. **Recommended Round-3 question** — what's the highest-leverage
   open question after Round 2?

### Honesty requirement

Round 1 noted the project's documented pattern: Deep Think's
qualitative survival claims are well-calibrated but specific
prescriptions are systematically wrong. Round 2's job is to
identify which Round 1 prescriptions are the wrong-prescription
class. If a prescription survives Round 2's adversarial review,
that's confirming evidence; if it doesn't, document the
modification or refutation precisely.
