# Deep Think Q9 — In-Situ Training Adversarial Robustness Architecture

**Status:** PROMPT DRAFT — not yet sent to Deep Think
**Drafted:** 2026-05-03 ~18:55Z
**Strategic role:** De-risk Phase-5 derisking experiments BEFORE running them
**Predecessors:** Q1-Q5, Q7 (HMC compatibility), Q8 (action representation) — all completed
**Subject:** architectural failure modes for externally-grounded verifier ensemble defense against reward hacking in in-situ training

---

## Context

Carnot is an open-source energy-based-model framework. Phase-5 (long-horizon vision) ships in-situ training: the model updates DURING inference using the verifier ensemble as the supervision signal. This is positioned as a candidate solution to the open problem reported by Zhang et al.\ (Sakana Darwin-Gödel Machine, arXiv:2505.22954, 2025): self-improving agents game their own evaluators.

Concretely, in Sakana's documented failure mode: when the agent was tasked with fixing tool-use hallucination, it "removed the markers we use in the reward function to detect hallucination, hacking our hallucination detection function to report false successes." The Sakana team identifies aligned self-improvement as the central open problem for self-improving AI: "no guarantee that improvements maintain alignment with human intentions."

Carnot's claim: this problem is tractable for ENERGY-BASED substrates with EXTERNALLY-GROUNDED verifier ensembles, in a way that's not tractable for autoregressive LLMs trained against their own evaluators. The architectural commitments that support this claim are spelled out in Section "Architectural commitments under test" below. We have scoped a 4-experiment small-scale derisking plan (Phase 5 prep, .94-.95 milestones) but want adversarial architectural review BEFORE running the experiments, so the experimental design can target the failure modes Deep Think identifies rather than just the ones we've thought of.

---

## Carnot's architecture (concrete commitments under test)

These are the commitments. Question is whether they together provide the claimed defense.

### Substrate (Phase 3 architecture)

- **DBAE-EBM**: Deterministic Bounded Autoencoder + Latent EBM. Bounded latent z ∈ [-1, 1]^d (d small at toy scale, d ~ 256-1024 at 1B-7B scale).
- **Encoder**: input → latent z. Trainable. Same agent that updates the energy function during in-situ training.
- **Energy network E(z, input) → ℝ**: scalar energy assignment over (latent, input) pairs. Trainable. The "model" being updated.
- **Decoder**: latent → output. Could be frozen or trainable; design choice.

### Verifier ensemble (the external grounding)

The ensemble is a k-AND-composition of formally distinct verifiers:

1. **Z3 SAT/SMT verifier** — formal correctness of constraint encodings
2. **AST structural verifier** — syntactic/structural validity of code outputs
3. **Semantic embedding probe** — distance-in-embedding to natural-language description
4. **ThinkPRM probe** — process reward model checking step-by-step reasoning quality
5. **JSON schema validator** — output well-formedness
6. **SC-Energy (Set-Consistency Energy Network)** — set-level consistency check; recently added (.91, exp1168)

Verifiers are NOT trained jointly with the model. They run as external graders. Their parameters do not update during in-situ training. The model can be updated; the graders cannot.

The AND-composition: a sample is "accepted" only if ALL k verifiers grade it as valid. Per CLAUDE.md, the architectural intent is "AND shrinks kernels exponentially" — the joint null space of disagreement points should be much smaller than any single verifier's null space.

### In-situ training mechanism (Phase 5)

Per query:

1. User submits input
2. Encoder maps input → latent space
3. Phase-4 blocked-Gibbs sampler explores energy landscape on latent z, conditioned on input
4. Decoder maps candidate z values to candidate outputs
5. Verifier ensemble grades each candidate; AND-composition selects best
6. Selected output returned to user
7. **IF verifier-AND PASSES (all k verifiers accept)**:
   - Positive phase: the accepted z
   - Negative phase: current sampler chain state
   - PCD gradient step on E with η ≈ 1e-5
   - Updated E persists for the next query

The verifier ensemble is the supervision signal. The reward-hacking attack would require: producing an output that satisfies all k formal verifiers while violating ground truth.

### Existing empirical findings

- exp1108 (.86): the joint null space of {Z3-AST, AST-structural, liveness} contains "vacuous dead code" — code that all three verifiers accept but ground truth rejects. Topological distinctness of verifiers does NOT guarantee disjoint null spaces.
- exp1121 (.87): SOSKANEnergyV3 verifier reaches AUROC=0.9545 on FoVer corpus but AUROC=0.3333 on a production corpus (worse than random) — OOD energy inversion. Verifier calibration is non-stationary across distributions.
- exp1163 (.90): NRGPT energy-recurrence prototype reports AUROC=0.92 with n_iters_monotone=False. The energy iteration that's the whole point is non-monotone; an interpretation gap remains.
- The 4-experiment small-scale derisking plan (Phase-5 prep) tests mechanism + stability + adversarial probe at toy scale (~50K params, 5×5 ARC-AGI puzzles, k=3 verifier subset).

---

## The question

**Are there architectural failure modes that compromise this defense at scale, such that the toy-scale derisking experiments would not detect them?**

Three sub-questions, in priority order:

### Q9.1 — Scale-dependent failure modes

The toy-scale derisking experiments use a ~50K-param model, k=3 verifier subset, 5×5 ARC-AGI puzzles, ~1000 queries. Phase-5 production deployment would be 1B-7B params, full k=6+ ensemble, real-task domains, millions of queries.

What failure modes are CHARACTERISTIC of the scale-up gap? Specifically:

- **Mode collapse**: at small scale, the energy landscape is small enough that the model's expressivity is limited. At 1B+ scale, can the model learn an energy function that is FORMALLY-CORRECT for verifiers (every output passes formal checks) while concentrating probability mass on a degenerate subset (e.g., single canonical answer for all queries, ignoring nuance)? What empirical signature would distinguish "diverse correct outputs" from "concentration on one formally-correct fixed point"?

- **Distillation degradation**: PCD updates in continual training have known degradation patterns at scale (cf. self-distillation collapse literature). Is there a regime where the in-situ training pattern enters a degenerate cycle the verifier ensemble cannot detect because every cycle-state passes formal checks?

- **Substrate shift**: at small scale, the bounded latent z ∈ [-1,1]^d is tightly constrained. At larger d, the bounded geometry may admit attack manifolds that don't exist at small d. Is there a topological argument that bounds attack manifolds across scale, or do new manifolds emerge with d?

### Q9.2 — Joint null space failure modes

exp1108 already established that formally-distinct verifiers can share a null space (vacuous dead code). The architectural intent is that AND-composition shrinks the joint null space exponentially, but this is empirical aspiration, not theorem.

- For k=6 verifiers each with non-empty null space N_i, what is the size of the joint null space ∩_i N_i in the worst case? Is there a structural argument that bounds it (e.g., based on the diversity of verification disciplines), or is it an empirical question?

- exp1108's vacuous-dead-code example is a SHARED null mode that's structural (verifiers all accept code that doesn't run). At scale, does the in-situ training pattern preferentially DRIVE the model toward such shared null modes (because they're locally low-energy and verifier-passing)? What empirical signature would catch this drift?

- Is there a class of attacks that specifically EXPLOITS joint null spaces — adversarial inputs that are crafted to land in the intersection of verifier acceptance regions while violating ground truth? What's the minimum verifier-diversity required to make such crafting NP-hard?

### Q9.3 — Continual-update dynamics failure modes

The in-situ pattern is fundamentally a CONTINUAL learning problem. One-shot evaluation may pass while continual dynamics fail.

- The PCD update is η ≈ 1e-5 per query. Over 1M queries, integrated drift = ~10. Does the bounded latent z ∈ [-1,1]^d guarantee bounded total drift, or can drift compound through the encoder (which IS being updated)?

- The verifier ensemble's accept rate is itself a non-stationary signal: as the model learns to satisfy verifiers, the ACCEPT rate goes up, which means more updates per query, which compounds. Is there a self-reinforcing loop where the model's drift co-evolves with the verifier's acceptance rate, even though the verifier itself isn't being updated?

- Catastrophic forgetting: PCD updates can erase prior knowledge. EWC regularization is mentioned in the proposal but not specified. Without EWC, what's the predicted retention curve for an arbitrary held-out distribution (i.e., how fast does the model forget non-task-related capabilities)?

---

## Format constraints (methodology level only — no implementations)

Per the project's Deep Think prediction-error pattern (`memory/feedback_carnot_prediction_pattern.md`):

> "Cross-validate every architectural prescription with Deep Think; qualitative survival claims well-calibrated, specific prescriptions systematically wrong."

Therefore Deep Think should provide:

- **Failure mode classification**: name each candidate failure mode at the methodology level. Don't propose specific architectural revisions; just NAME the failure mode and describe its mechanism.
- **Empirical signature per failure mode**: for each mode, specify what would be observed in measurements that would distinguish "this mode is happening" from "this mode is not happening." Empirical signatures should be measurable in the 4-experiment derisking plan with minimal modification.
- **Failure-mode-to-experiment mapping**: indicate which sub-experiment (exp_NEXT_A through exp_NEXT_D) would BEST detect each failure mode. If a failure mode CANNOT be detected at toy scale, say so explicitly — that's a critical finding.
- **Cross-validation lane check**: explicit self-check on whether each named failure mode is supported by published evidence, theoretical argument, or speculation. Distinguish these.
- **Honest unresolvable uncertainty**: if there are failure modes the toy-scale experiments cannot detect AND the architectural prediction is genuinely undetermined, say so. The point of asking is to find these.

What we DO NOT want:

- Specific architectural prescriptions (e.g., "you should add modulo-3 verifier"). The previous Deep Think rounds have shown specific prescriptions are systematically wrong; methodology-level findings have been calibrated.
- Speculative attack constructions without grounding in published evidence or theoretical argument.
- "Three options exhaustive" claims without explicit self-correction (cf. Q8's self-corrections were valued).

---

## Cross-references for Deep Think

If Deep Think can read these references in context, they may help calibrate:

- `CLAUDE.md` Project Vision section (Phase 1-3 commitments)
- `CLAUDE.md` Decentralization-Respecting Design Constraints (sovereignty + verifier-ensemble grounding)
- `_bmad/architecture.md` (if available; otherwise CLAUDE.md captures the substrate)
- `openspec/change-proposals/in-situ-training-phase5-derisking.md` (the 4-experiment plan to be calibrated)
- `memory/reference_sakana_dgm.md` (the Sakana DGM open problem in detail)
- `memory/reference_pnas_evolvable_ai.md` (Müller et al. PNAS 2026 Breeder Scenario framing)
- `docs/research-notes/paper-v5-decentralization-section-draft.md` (paper-v6 positioning that depends on this defense)
- exp1108 result artifact (joint null space empirical finding)
- exp1121 result artifact (SOSKANEnergyV3 OOD inversion)
- exp1163 result artifact (NRGPT non-monotonicity)

---

## Why this prompt before the experiments run

The 4-experiment Phase-5 derisking plan is queued for .94-.95 (scoped tonight in known-issues + change-proposal). exp_NEXT_D in particular is an ADVERSARIAL probe — and we want that probe to test the failure modes Deep Think identifies, not just the ones we've imagined. The cost asymmetry is decisive:

- Running Deep Think now: ~30 min, refines the experimental design
- Running the experiments without Deep Think calibration: 2-3 weeks, may miss failure modes

This is the same pattern as Q7 (HMC compatibility) and Q8 (action representation): architectural cross-validation BEFORE empirical implementation. Both Q7 and Q8 surfaced findings that would have been expensive to discover post-implementation.
