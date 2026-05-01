# Deep Think Prompt — Phase-3 Prototype Pre-Flight Adversarial Gates

**Status:** Ready to send. Most strategic of the three pending Deep
Think questions: this defines the **abort conditions** for the .88
DBAE-EBM 4-stage prototype kickoff. Without these, the prototype will
either burn a milestone before discovering an architecture flaw, or
ship at scale before its assumptions are empirically validated.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-3 architecture has been formalized over
**6 Deep Think rounds** between 2026-04-25 and 2026-05-01. The
synthesis is documented at
`memory/reference_phase3_deep_think_synthesis.md`. Headline
architecture (load-bearing claims that survived all rounds):

#### Substrate

- **DBAE-EBM**: Deterministic Bounded Autoencoder + Latent EBM.
  Encoder maps text x → bounded latent z ∈ [-1,1]^d. Decoder
  reconstructs x from z. Latent EBM operates on **sgn(z) → Ising**
  (i.e., the binary thresholding of z is the Ising spin
  configuration that the EBM scores).

#### Verifier ensemble

- **6 base verifiers** spanning topologically distinct mechanism
  families: Z3-AST formal, gVisor runtime execution, semantic
  embedding, ThinkPRM step-level, JSON schema, Potts combinatorial.
- **AND-composition** at **k_max ≤ 8** by Welch-Rankin Simplex
  bound. Empirical α² = 0.66 (homogeneous text probes) limits
  k* ≤ 3 in collapse regime; cross-mechanism diversity at α² ≈
  0.4 admits k_max ≈ 7-8.
- **Empirically validated at k=5** (exp1108): max pairwise r =
  0.462 < 0.5 architectural threshold. k=6 has max r = 0.506 >
  threshold (one rogue pair: ThinkPRM × Z3MathVerifier).

#### Training pipeline (4-stage)

- **Stage 1**: Pretrain DBAE on FoVer 6,500-pair corpus (next
  expansion: exp1119's 7K+ corpus from SOTA outputs).
- **Stage 2**: Train latent EBM on sgn(z) labels with energy
  reward from AND-composed k=5 verifier ensemble.
- **Stage 3**: GRPO with energy reward (validated by exp1118:
  `positive_improvement` verdict).
- **Stage 4**: Joint fine-tune with **SP-IWPER** (Stratified
  Pipeline with Importance-Weighted Prioritized Energy Replay)
  buffer.

#### Asynchronous replay constraint

- **DDE Hopf bound** on async replay buffer:
  `N_max < π / (2·sqrt(η_d·η_e·ρ(H_φθ·H_θφ)))`
  where η_d, η_e are decoder / encoder learning rates and ρ is
  the spectral radius of the cross-Hessian. Below this N, the
  buffer's stale data is safe; above, the joint fine-tune
  oscillates.

### Phase-validation discipline (CLAUDE.md mandatory)

The 2026-04-30 Phase-3 architecture blind-spot audit caught **5
FATAL findings** that three rounds of rigorous theoretical Deep
Think missed. The lesson:

> *Unless we have adversarial checks at each phase boundary, we
> are building a house of cards that cannot function in the end.*

Per CLAUDE.md (mandatory), every Carnot phase prototype must
satisfy three requirements before any scaling decision:

1. **Software prototype** — concrete code, runnable end-to-end at
   small scale.
2. **Empirical validation criteria** — measurable pass/fail tests
   with explicit thresholds.
3. **Adversarial check** — hostile-reviewer round explicitly
   commissioned to find ways the prototype could pass acceptance
   gates without actually working.

The CLAUDE.md framework also says:

> *Empirical instrumentation IS adversarial check at scale. A
> prototype that emits the right diagnostics surfaces architecture-
> level flaws automatically. A prototype that doesn't will let
> flaws ship.*

### The gap we want Deep Think to close

The Phase-3 prototype kickoff is provisionally targeted for
milestone .88 (estimated 2026-05-08 to 2026-05-15). Right now we
have:

- An architecture document
- 6 Deep Think rounds of theoretical analysis
- Empirical validation at the *substrate* level (k=5 ensemble
  works, GRPO+energy reward shows positive_improvement, KV260
  sequential sampler validates KL = 0.025 nats)

What we **DO NOT have**:

- A list of specific **adversarial attacks** the prototype must
  survive in the first 1000 training steps
- The **diagnostic instrumentation** required to surface those
  attacks automatically
- The **abort thresholds** that would prevent a flawed prototype
  from scaling

### Specific question

**For the DBAE-EBM 4-stage prototype defined above, list 5–7
hostile-reviewer attacks the prototype must survive in the first
1000 training steps to justify continuing to scale (.88 → .89
+).**

For each attack:

1. **Attack name** — short identifier.
2. **Failure mode probed** — which architectural assumption it
   stress-tests. (e.g., "DBAE encoder degenerates to constant",
   "EBM converges to single low-energy point", "verifier
   ensemble shares pathological joint null space",
   "GRPO advantages collapse to constant", "decoder ignores
   bottleneck and uses LM prior alone".)
3. **Diagnostic to instrument** — specific quantity to log per
   step (e.g., "encoder output variance per dimension",
   "EBM energy histogram width", "joint Σ determinant on
   verifier output covariance"). Specify the quantity precisely
   enough that we can implement the logger without ambiguity.
4. **Abort threshold** — value(s) of the diagnostic at which we
   abort the prototype rather than continue scaling. Specify the
   direction (above / below) and the rationale (why this value
   means the architecture is broken, not just slow-to-converge).
5. **Confidence calibration** — what would make this attack
   *less* informative as a gate? (e.g., "very small batch sizes
   make this metric noisy", "the threshold depends on
   batch_size × learning_rate product".)

### Constraints on output

- **NO parameter prescriptions** at the architecture level. Don't
  recommend specific model dimensions, learning rates, or batch
  sizes — those are .88 implementation details, not phase
  validation criteria.
- **DO provide diagnostic-quantity definitions** that are precise
  enough to implement (formulas, pseudocode, or log-line
  examples).
- **DO calibrate thresholds in *direction* and *order of
  magnitude***, not specific decimal values. e.g.,
  "decoder reconstruction loss must DECREASE at least 10x from
  step 1 to step 1000" is fine; "decoder reconstruction loss
  must reach 0.247 by step 873" is over-prescribed.
- **DO link each attack to a Phase-3 architectural assumption**
  named in the synthesis above. Attacks that don't trace to a
  specific assumption are speculative.

### Output format request

```
ATTACK 1: <name>
  Failure mode: <which architectural assumption breaks>
  Architectural assumption: <quote from synthesis above>
  Diagnostic: <quantity, with formula or pseudocode>
  Abort threshold: <direction + order of magnitude + rationale>
  Confidence calibration: <what makes this less reliable>

ATTACK 2: <name>
  ...

ATTACK 7: <name>
  ...

DECISION TREE:
  If 0 attacks fire: continue to scaling (.89+).
  If 1 attack fires: <recommended response — fix vs. abort>
  If 2+ attacks fire: <recommended response>
  If a specific combination of attacks fires: <recommended response>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think
rounds have qualitative survival claims well-calibrated, but
specific numerical prescriptions systematically wrong. This
question is framed in the methodology / diagnostic-design lane;
keep the answer there.

Also: per the 2026-04-30 blind-spot audit, three rounds of
rigorous Phase-3 theoretical Deep Think missed 5 FATAL findings
that an explicitly-adversarial round caught. **Be hostile in
designing these attacks.** Imagine a reviewer at NeurIPS who
*wants* to reject the paper. What would they probe to make the
prototype's results look like specification-gaming or null-result-
masquerading-as-success? Those are the attacks we need.

---

## End of prompt
