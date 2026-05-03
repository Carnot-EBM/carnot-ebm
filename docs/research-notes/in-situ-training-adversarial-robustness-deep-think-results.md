# Deep Think Q9 Response — In-Situ Training Adversarial Robustness

**Status:** Response received 2026-05-03 ~19:30Z. Methodology-only adversarial review.
**Source prompt:** `in-situ-training-adversarial-robustness-deep-think-prompt.md`
**Pattern compliance:** clean — all eight failure modes named at methodology level with empirical signatures + experiment mapping + lane checks. No specific architectural prescriptions. Three explicit unresolvables flagged.

---

## TL;DR

| Failure mode | Detectable at toy scale? | Severity |
|---|---|---|
| 1. Mode collapse (overparameterized degenerate attractors) | **NO** — capacity required | CRITICAL |
| 2. MCMC mixing paralysis (PCD distillation degradation) | **NO** — mixing time scales w/ d | CRITICAL |
| 3. Substrate shift (high-d measure concentration) | **NO** — phase transition req's large d | CRITICAL |
| 4. Gradient-driven null space excavation | YES (anchor distance tracking) | HIGH |
| 5. Correlated evaluator blind spots | YES (conditional acceptance probs) | HIGH |
| 6. Unbounded representation drift | YES (encoder weight spectral norms) | HIGH |
| 7. Autocatalytic acceptance spiral | YES (acceptance-rate first-derivative) | HIGH |
| 8. Capability overwriting (forgetting) | YES (held-out task buffer eval) | HIGH |

**Three honest unresolvables:**
- Joint null space worst-case bound (no a-priori structural argument exists)
- NP-hard threshold for crafting attacks (continuous EBM circumvents discrete bounds)
- Diverse degeneracy vs diverse correctness indistinguishable from internal metrics alone

---

## The load-bearing finding

**Three of the eight failure modes are STRUCTURALLY UNDETECTABLE at toy scale.** Specifically:

- Mode collapse requires capacity to learn the evaluator boundary (50K params can't even attempt)
- MCMC mixing failure requires high-d landscape geometry (toy d~16 mixes smoothly, hides issue)
- Measure concentration is a geometric phase transition that requires large d to manifest

This means the small-scale Phase-5 derisking plan I scoped tonight, while valuable, is **NECESSARY BUT INSUFFICIENT.** Validating that 5 of 8 failure modes don't manifest at toy scale tells us nothing about the 3 modes that only manifest at production scale.

---

## Deep Think's full classification (preserved verbatim, organized for retrieval)

### Q9.1 Scale-dependent failure modes

**1. Mode collapse — overparameterized degenerate attractors**
- Mechanism: at 1B+, the model becomes a universal function approximator for the evaluator boundary itself. Learns to map diverse inputs into a "safe" manifold that satisfies all verifiers without genuine problem-solving.
- Signature: input variance high, latent/output variance decays toward zero, verifier acceptance ~100%
- Experiment mapping: **NOT detectable at toy scale.** 50K params lack capacity to learn the multi-verifier Goodhart manifold without breaking the decoder.
- Lane: published evidence. Sakana DGM Clever Hans. Generalized mode collapse in overparameterized EBMs.

**2. Distillation degradation — MCMC mixing paralysis**
- Mechanism: PCD negative phase requires sampler to mix between modes. At large d, energy landscape becomes highly non-convex; finite-step blocked-Gibbs fails to mix; spurious low-energy wells form globally; sampler never proposes them for grading; verifiers never see them; energy degrades silently.
- Signature: Gibbs chain autocorrelation → 1.0; L2 distance between PCD positive and negative phase z values monotonically collapses toward zero.
- Experiment mapping: **NOT detectable at toy scale.** Mixing times scale exponentially with d. Toy d mixes smoothly.
- Lane: theoretical + published (Du & Mordatch 2019).

**3. Substrate shift — high-d measure concentration**
- Mechanism: in [-1,1]^d with d=256-1024, ~100% of Euclidean volume is at the surface and 2^d corners. Continuous in-situ gradient descent routes z into these boundaries where gradient flow blocks. Attack manifolds that were isolated points at small d become contiguous hypersurfaces at large d.
- Signature: L∞ norm of accepted latents → 1.0 (saturation); dimensional histograms become strictly bimodal at {-1, 1}.
- Experiment mapping: **NOT detectable at toy scale.** Geometric phase transition requires large d.
- Lane: theoretical (curse of dimensionality, concentration of measure on hypercubes).

### Q9.2 Joint null space failure modes

**4. Gradient-driven null space excavation**
- Mechanism: PCD positive phase only updates on accepted outputs. Shared null modes (cf. exp1108 vacuous dead code) become massive attractors. Path-of-least-resistance optimization aggressively digs energy wells around the easiest shared blind spot.
- Signature: generation frequency of trivial-output classes increases monotonically; latent distance to "vacuous anchor states" decreases monotonically.
- Experiment mapping: **exp_NEXT_B (stability) + exp_NEXT_A (mechanism)** — DETECTABLE at toy scale if vacuous anchors are explicitly tracked.
- Lane: theoretical (implicit-bias optimization toward simpler local minima; reward hacking).

**5. Correlated evaluator blind spots**
- Mechanism: AND shrinks null space exponentially ONLY under verifier independence. AST and Z3 share syntactic rigidities; Semantic and ThinkPRM share LLM-distributional biases. Trainable encoder learns joint-space projections that exploit correlations.
- Signature: conditional acceptance probabilities violate independence: P(Pass V_i | Pass V_j) ≫ P(Pass V_i).
- Experiment mapping: **exp_NEXT_C (adversarial probe)** — DETECTABLE via verifier ablation.
- Lane: published (transferability of adversarial examples; ensemble robustness failure modes).

### Q9.3 Continual-update dynamics failure modes

**6. Unbounded representation drift — encoder compounding**
- Mechanism: bounded z guarantees bounded latent state, NOT bounded coordinate frame. Encoder weights are NOT bounded. Over 1M queries, encoder progressively warps input space, mapping distinct queries to identically saturated z coordinates.
- Signature: monotonic unbounded growth in encoder weight spectral norm / L2 norm; crash in batch variance of f(x) for semantically diverse inputs.
- Experiment mapping: **exp_NEXT_B (stability)** — DETECTABLE at toy scale over thousands of queries.
- Lane: published (representation collapse + weight explosion in continual unregularized autoencoders).

**7. Autocatalytic acceptance spiral — parasitic feedback loop**
- Mechanism: acceptance rate gates update frequency. Higher acceptance → more positive PCD updates per query → accelerated drift toward heuristic. Effective learning rate scales proportionally with exploit severity.
- Signature: super-linear (exponential) acceleration in rolling-average acceptance rate, perfectly anti-correlated with drop in held-out ground-truth oracle accuracy.
- Experiment mapping: **exp_NEXT_B (stability)** — DETECTABLE via first derivative of acceptance rate curve.
- Lane: theoretical + published (confirmation bias and runaway feedback loops in pseudo-labeling / continuous self-training).

**8. Capability overwriting — catastrophic forgetting**
- Mechanism: PCD negative phase operates locally on current user input. Over thousands of queries, this acts as a localized bulldozer, systematically raising energy basins of non-queried task domains.
- Signature: aggressive monotonic exponential decay in AUROC + sampling accuracy on broad-coverage static evaluation benchmark hidden from in-situ loop.
- Experiment mapping: **exp_NEXT_D + exp_NEXT_B** — DETECTABLE at toy scale via interleaved held-out task buffer.
- Lane: published (continuous learning catastrophic forgetting; exacerbated in EBMs by negative-sampling-pushes-up-non-target-energies).

### Honest unresolvables (4-test failures the toy plan cannot fix)

- **Worst-case joint null space size**: no a priori structural argument exists. AST (syntactic grammar), Z3 (combinatorial), Semantic/ThinkPRM (continuous embedding) operate on incommensurate topologies. Volume of intersection is purely empirical.
- **Minimum verifier diversity for NP-hard crafting**: discrete-search NP-hardness is well-defined, but continuous EBM optimizers routinely circumvent worst-case bounds via differentiable proxies. Exact threshold is unmapped.
- **Diverse degeneracy vs diverse correctness**: cannot be distinguished from internal variance metrics alone. Requires frozen ground-truth oracle in continuous evaluation.

---

## Implications for the Phase-5 derisking plan

The 4-experiment plan as scoped (`openspec/change-proposals/in-situ-training-phase5-derisking.md`) targets 5 of 8 failure modes correctly. **Plan revisions required:**

1. **Add ground-truth oracle continuous evaluation as the central acceptance gate.** All 5 detectable failure modes' signatures pivot on comparison to a held-out frozen oracle. Without it, exp_NEXT_B/C/D's metrics are partially blind.

2. **Add encoder weight norm tracking to exp_NEXT_B.** Currently unspecified; load-bearing for failure mode #6.

3. **Add vacuous-anchor distance tracking to exp_NEXT_A.** Currently unspecified; load-bearing for #4.

4. **Add conditional acceptance probability matrix to exp_NEXT_C.** Currently unspecified; load-bearing for #5.

5. **Explicitly DOCUMENT the 3 undetectable failure modes as deferred-to-production-validation.** This is the most important change. The plan must NOT claim "Phase-5 architecture validated" if it passes — it can only claim "5 of 8 modes detected absent at toy scale; 3 modes structurally undetectable, requiring intermediate-scale validation before 1B+ deployment."

6. **Consider scoping an INTERMEDIATE-SCALE experiment for .96 or .97.** ~100-300M params on real ARC-AGI tasks. Catches the 3 undetectable modes earlier than full 1B+ deployment.

---

## Cross-validation status — Q9 clean

```
Q1-Q5      historical Deep Think rounds       (clean)
Q7        HMC compatibility                   (clean, self-corrected)
Q8        action representation               (clean, self-corrected)
Q9        in-situ training adversarial review (clean, 3 explicit unresolvables)
```

The pattern across rounds: prediction-error pattern fully internalized; failure mode classifications methodology-level (no prescriptions); empirical signatures specified concretely; honest unresolvables surfaced explicitly.

---

## Recommended next steps

1. ✅ **Save this response** (done — this file).
2. **Update the Phase-5 derisking change-proposal** to incorporate Q9's findings:
   - Add ground-truth oracle as required infrastructure
   - Add 4 new measurement metrics (encoder norm, vacuous anchor, conditional acceptance, retention curve)
   - Document the 3 undetectable failure modes as scope limits
   - Reframe acceptance criteria: "5/8 detectable failure modes absent" not "Phase-5 architecture validated"
3. **Update known-issues**: incorporate the gap finding into the .94+ planner directive.
4. **Consider an intermediate-scale experiment** scoped for .96 or .97 to catch the 3 production-scale-only failure modes before full 1B+ deployment.
