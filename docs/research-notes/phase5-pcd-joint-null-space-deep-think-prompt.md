# Deep Think Q12 — Phase-5 PCD + Verifier Joint Null Space Co-Evolution

**Status:** PROMPT DRAFT — for paste into Gemini chat UI Deep Think
**Drafted:** 2026-05-04 ~01:25Z
**Strategic role:** Predict whether in-situ training is self-stabilizing or requires explicit anti-gaming regularization at production scale (informs exp1238 Phase-5-D acceptance gates)
**Predecessors:** Q9 (in-situ adversarial robustness), Q11 (verifier orthogonality design), Spera Theorem 9.2 empirical confirmation in exp1224
**Dependency:** Q11 result strengthens Q12 by giving the joint null space a principled measurement basis

---

## Paste boundaries

```
START:  "## The Deep Think question" (line ~16)
END:    end of "## Output format requested" section
SKIP:   this header, "## Why this question now"
```

---

## The Deep Think question

I am building Carnot, an open-source energy-based-model framework with
an in-situ training mechanism: model parameters update DURING inference
when the verifier ensemble accepts an output. The training rule is
Persistent Contrastive Divergence (PCD) with the verifier-AND-composition
serving as the supervision signal — accepted samples become positive
PCD examples; rejected samples are negatives.

Phase-5-A and Phase-5-B (small-scale) confirmed the architecture works:
all 5 detectable-at-toy-scale failure modes were absent. Phase-5-C
adversarial probe found a verifier-correlation attack (P(V_i|V_j) =
1.000) — the empirical realization of Spera Theorem 9.2 (arXiv:2603.15973)
on a k=3 ensemble. Phase-5-D will run intermediate-scale (~100-300M
params, d=128, k=5+ ensemble) to detect 3 production-scale-only failure
modes: mode collapse, MCMC mixing paralysis, and substrate shift.

Before Phase-5-D commits 30-60 GPU-hours, I need a theoretical
prediction of how the substrate's training distribution and the
verifier ensemble's joint null space CO-EVOLVE during in-situ PCD
updates. Specifically: does PCD systematically expand, shrink, or
rotate the joint null space?

### The core question

Define:

- **Substrate M(θ_t):** an energy-based model with parameters θ_t at
  in-situ training step t. Encoder + energy network + decoder, mapping
  inputs x → continuous latent z ∈ [-1,1]^d → sign(z) → discrete
  action a.
- **Verifier ensemble V = {V_1, ..., V_k}:** fixed (NOT co-trained
  with M). Each V_i: A → {0,1}.
- **Joint null space N_t = {a ∈ A : ∀i, V_i(a) = 1, but O(a) = 0}**
  where O is the ground-truth oracle. This is the set of outputs that
  pass all k verifiers but are wrong by oracle.
- **Substrate output distribution P_t(a) = ∫ p_θ_t(z) δ(a = decode(z)) dz**.
- **PCD update rule:** at each accepted output, perform one
  contrastive-divergence step on θ; positive sample is accepted output,
  negative sample is from persistent chain of model fantasies.

The question: as t → ∞ under PCD updates with fixed V, what is the
asymptotic relationship between P_t and N_t?

### Three competing hypotheses to evaluate

**Hypothesis A (Self-correction / verifier-shaped distribution).**
PCD pulls P_t toward the verifier-acceptance region (P_t concentrates
on a ∈ A satisfying ∀i, V_i(a) = 1). Over time, P_t becomes more
verifier-aligned, but if the verifiers are well-calibrated against
oracle (small N_t), the substrate becomes oracle-aligned too. The
joint null space N_t stays small and bounded.

**Hypothesis B (Adversarial substrate gaming / null-space excavation).**
PCD pulls P_t toward the EASIEST verifier-acceptance region — which
includes the joint null space (vacuous truths, formally-valid-but-
semantically-empty outputs). The substrate learns to produce N_t
samples preferentially because they're cheap to generate and pass all
verifiers. This is Carnot's exp1108 vacuous-dead-code finding scaled
up: dead code passes Z3 + AST + liveness, so the substrate learns to
produce dead code efficiently. N_t expands relative to oracle-correct
acceptance regions.

**Hypothesis C (Stationary rotation / null-space drift without
expansion).** PCD's stationary distribution is invariant in volume
(the verifier-acceptance region's measure under P_∞ is bounded by the
verifier ensemble's structure), but the *direction* of joint null
space within that region rotates as P_t shifts. The substrate doesn't
preferentially produce N_t but the GEOMETRY of where N_t sits in the
acceptance region changes — making post-hoc detection (which relies on
fixed-direction probing) unreliable.

### Specific sub-questions to engage

**Q12.1 — Stationary distribution of PCD under verifier-AND
supervision.** Derive (or prove non-existence of) the stationary
distribution P_∞ of the substrate M(θ) under PCD updates conditioned
on verifier-AND acceptance. Express P_∞ in terms of: substrate's prior
energy E_θ, verifier acceptance indicator V_AND(a), and (if relevant)
the persistent-chain mixing properties.

**Q12.2 — Joint-null-space evolution under stationary distribution.**
Given P_∞, what is the volume of the joint null space N_∞ relative to
the oracle-correct acceptance region? Specifically: is
|N_∞| / |V_AND ∩ O_correct| asymptotically bounded, growing, or shrinking?

**Q12.3 — Connection to Q11's verifier-design result.** If Q11's
constructive procedure (synthesizing verifiers with bounded joint null
space at design time) succeeds, does that bound carry through PCD
training, or does PCD require a separate analysis? Specifically: is
Q11's pairwise-orthogonality bound preserved under in-situ training, or
can PCD updates rotate the substrate output distribution in a way that
re-introduces correlation in the verifier-accepted region even when
verifiers are pairwise-disjoint at design time?

**Q12.4 — Anti-gaming regularization.** If hypothesis B (substrate
gaming) is correct, what regularization on θ updates would prevent
null-space excavation? Candidates: (a) entropy maximization on accepted
outputs (force the substrate to keep producing diverse correct outputs,
not just easy null-space outputs); (b) negative-sampling from oracle-
disagreement set (when ground-truth-oracle is available periodically,
use disagreements as explicit negative examples); (c) null-space-
distance regularization (penalize θ updates that move accepted outputs
closer to known null-space anchors per Q9 mode 4 instrumentation).
Derive the formal correctness condition for each candidate.

**Q12.5 — Detectability at intermediate scale.** Phase-5-D will run
~100-300M params with k=5+ verifier ensemble for 10K queries against
ARC-AGI tasks. Which of hypotheses A/B/C is empirically distinguishable
at this scale, and which require production scale (1B+ params)?
Specifically: derive the minimum number of queries N_min such that the
PCD trajectory's projection onto the joint null space N_t is
statistically separable from no-evolution baseline. If N_min > 10K,
Phase-5-D cannot disambiguate hypotheses and the design must be
revised.

### Output format requested

Please structure as:

1. **Executive summary (3-5 paragraphs).** State which hypothesis
   (A/B/C) the analysis supports, under what assumptions, and the
   strongest empirical signature.

2. **Per sub-question (Q12.1 – Q12.5).** Derivation or proof sketch,
   with explicit assumptions and the resulting bound or counterexample.

3. **Honest framing.** Where does the analysis depend on unverified
   assumptions about Carnot's substrate (encoder/decoder details, PCD
   chain length, verifier sharpness)? Spell out what could change the
   conclusion.

4. **Recommended exp1238 acceptance gates.** Given the analysis, what
   numerical thresholds should Phase-5-D's gates use to detect each
   hypothesis empirically? Map each hypothesis to a measurable
   signature (e.g., "if hypothesis B holds, expect E(z_accepted) to
   plateau at some threshold E* > 0 corresponding to null-space
   energy; if hypothesis A holds, expect E(z_accepted) to monotonically
   decrease toward oracle-correct minimum").

5. **Open empirical follow-up.** What measurements should exp_NEXT_E
   instrument that Phase-5-A/B did NOT? Specifically: what observable
   distinguishes A from B from C at intermediate scale?

6. **Connection to active inference (Phase-4 commitment).** Phase-4
   committed to the active-inference-as-free-energy hypothesis. Does
   the PCD trajectory have a free-energy reformulation that makes
   hypothesis selection more transparent? E.g., is one hypothesis
   equivalent to "verifier ensemble fails to provide enough free-
   energy gradient to escape null-space attractors"?

### Format constraints

- Use formal notation: stationary distributions, KL divergences,
  measure theory on N_∞, ergodicity assumptions.
- Cite related work explicitly: Spera 2026 (arXiv:2603.15973), Apple
  SSD self-distillation (the relevant precedent for self-improvement
  without verifier), Du & Mordatch 2019 PCD foundations, Q9 result
  on 8 failure modes.
- Distinguish "stationary distribution exists and is non-degenerate"
  from "stationary distribution exists and is what we want" — these
  are different theoretical results.
- For results requiring strong assumptions (ergodicity, verifier
  sharpness, prior diversity), state the assumption explicitly and
  flag fragility.
- Speculative steps should be flagged with "Speculative:" and
  alternative interpretations provided.

---

## Why this question now (decision-leverage)

exp1238 (Phase-5-D intermediate scale) is on the .96/.97 roadmap and
will run when Phase-5-C revision (exp1233) ships and the orthogonality
audit (exp1232) passes. exp1238 commits 30-60 GPU-hours to detect 3
production-scale-only failure modes that Phase-5-A/B couldn't reach.

But Phase-5-A/B/C couldn't predict exp1224's verifier-correlation
finding either. That was a Q9 prediction (mode 5: correlated
evaluator blind spots), confirmed empirically. Q12 is the analogous
prediction for Phase-5-D: derive A/B/C hypotheses theoretically, then
let Phase-5-D's empirical data discriminate.

Without Q12, exp1238's acceptance gates are designed against generic
"failure-mode catalog" thresholds (Q9 mode 1/2/3 at production scale).
With Q12, gates are designed against specific hypotheses with
predicted observable signatures — making exp1238's results
diagnostically more powerful.

Cost asymmetry: ~30 minutes of operator paste time + Gemini Ultra
Deep Think compute (off Carnot's quota) vs running exp1238 with
ambiguous gates and getting "verdict: research_finding" with no clear
disambiguation. Not catastrophic but expensive — 30-60 GPU-hours of
ambiguity.

## Cross-references

- exp1224 artifact: `results/experiment_1224_phase5c_adversarial_probe.json`
- Q9 results: `docs/research-notes/in-situ-training-adversarial-robustness-deep-think-results.md`
- Q11 prompt (verifier orthogonality design): `docs/research-notes/verifier-orthogonality-design-deep-think-prompt.md`
- Phase-5 derisking proposal: `openspec/change-proposals/in-situ-training-phase5-derisking.md`
- Spera Theorem 9.2 memory: `memory/reference_spera_theorem_92.md`
- Phase-4 active-inference commitment: `memory/feedback_active_inference_phase4_committed.md`
- Apple SSD self-distillation precedent: `memory/project_ssd_self_distillation.md`
