# Phase-3 Architecture: Pivot to DAE-DEBM (Discrete AE + Discrete EBM)

**Date:** 2026-04-30
**Status:** **DECIDED — pivot taken** based on Round-2 Deep Think bonus
flag + hardware verification

## Decision

The Phase-3 Kona-parity foundation model architecture is now:

```
text → encoder (real-valued NN logits, e.g., RoBERTa init)
     → straight-through estimator: z = sgn(logits) ∈ {-1, +1}^d
     → EBM(z) trained via exact Gibbs/Glauber sampling
     → z' (sampled by Glauber, matches FPGA hardware exactly)
     → decoder
     → text'
```

This **supersedes** the Round-1 DBAE-EBM (continuous bounded latent
in `[-1, 1]^d` with Mirrored Langevin Dynamics) result.

## Why the pivot — verified hardware primitive

Round 2 Deep Think raised the conditional flag: *"if Phase-2 FPGA
specifically executes discrete Glauber spin-flips on a ±1 lattice,
a continuous EBM is structurally mismatched."*

Verification 2026-04-30 against `hardware/kv260/ising_sampler_v2.v`
(623 lines, used in exp1081 FPGA scale benchmark) confirmed
the condition holds. Smoking-gun line:

```verilog
if (lfsr[spin_i] < {flip_prob[spin_i][7:0], 8'b0}) begin
    state_ram[spin_i >> 5][spin_i[4:0]] <= 1'b1;   // spin = +1
end else begin
    state_ram[spin_i >> 5][spin_i[4:0]] <= 1'b0;   // spin = -1
end
```

Hardware characteristics that force the pivot:

| Property | KV260 reality |
|---|---|
| Spin state representation | 1 bit per spin (packed 32 spins / 32-bit word) |
| Update primitive | Synchronous parallel Glauber (checkerboard even/odd) |
| Probability source | Sigmoid LUT of `β · h_eff[i]` + LFSR |
| Continuous spin variables | None (no DAC, no analog spin reg) |
| Gradient hardware | None |
| Annealing | β-schedule on discrete temperature |

A continuous EBM would learn gradient paths through `[-1, 1]^d` that
the hardware cannot execute. The mismatch surfaces at deployment as
non-zero `sgn(z')` transpilation degradation — exactly the failure
mode Round 2's acceptance gate flagged as the Phase-2 go/no-go.

By choosing discrete latent + discrete EBM, **the transpilation gap
collapses to zero by construction** — the encoder output IS the
hardware spin state.

## How Round 2's recipe transfers to discrete

Most of Round 2's answers transfer verbatim. The mapping:

| Round 2 (continuous DBAE) | Updated for discrete DAE |
|---|---|
| Tanh-bounded latent `[-1, 1]^d` | Binary latent `{-1, +1}^d` via STE |
| `var(z) ≤ 1.0` penalty (Q1) | `var(±1) = 1.0` automatic; **decorrelation only** |
| Masked-token reconstruction (Q1) | **Same — still essential** |
| 3-stage warmup (Q2) | **Same** — but EBM in stage 2/3 is discrete Gibbs |
| Mirrored Langevin Dynamics (Q3) | **Replaced** by exact Glauber |
| Stop-gradient on `z_fake` (Q2) | **Same** — STE already provides gradient bottleneck |
| `β` start at ~0.01·∇L_recon (Q2) | **Same** |
| 100M params, FoVer 6500-pair, 10K steps (Q4) | **Same** |
| Transpilation Gap test (Q4) | **Trivially passes by construction** |
| Manifold Dead-Zone test (Q4) | **Still essential** — discrete may have sparser semantics |

## Open questions for Round 3

The pivot creates new unknowns Round 3 should answer:

1. **STE variant.** Which straight-through estimator? Identity STE
   (Bengio 2013), Hinton's saturation-aware STE, or Gumbel-Softmax-
   then-sgn? Different gradient noise characteristics.

2. **Discrete EBM training algorithm.** PCD (persistent contrastive
   divergence), NCE (noise contrastive estimation), score matching
   adapted for discrete (e.g., concrete score matching), or pure
   Gibbs maximum likelihood? Each has different stability properties
   on language data; literature is sparse for discrete EBM at LLM
   scale.

3. **Latent dimension `d` selection.** A continuous EBM has the
   tanh activation as a soft regularizer on `d`. Discrete `{-1,+1}^d`
   doesn't. Too small → expressivity ceiling; too large → exponential
   state space and slow Glauber mixing. Need a principled choice or
   sweep range.

4. **Glauber temperature schedule.** The hardware uses β-annealing.
   Should training use the same schedule, or static `β = 1`? Does the
   verifier-grounded gradient still survive a hot-to-cold trajectory
   during training?

5. **Manifold dead-zones at discrete granularity.** With only
   `2^d` possible latents, semantic interpolation is impossible at
   the latent level. Does this mean reasoning chains must be
   *long* (many Glauber steps) to traverse intermediate
   semantically-meaningful states? If so, what's the minimum
   chain length for a 100M-parameter prototype to demonstrate
   anything?

## Position paper update

The Phase-3 section of any future position paper draft (after
exp1078-v2) must:
- Cite **DAE-DEBM** as the chosen architecture, not DBAE-EBM
- Cite the Round 1 + Round 2 + Verification chain
- Highlight the architectural elegance: **the transpilation gap is
  closed by construction**, not by engineering
- Cite `hardware/kv260/ising_sampler_v2.v` as the hardware reality
  that constrained the choice

## Implementation seed (.85 or .86 candidate task)

Title: "DAE-DEBM Phase-3 Prototype on FoVer 6500 — Manifold Validity
Test"

Spec:
- 100M-param encoder/decoder (RoBERTa-base init), STE on
  `z ∈ {-1,+1}^d` with `d = 256` (sweep 128/256/512 if budget allows)
- Discrete EBM trained via PCD; chain length 100 Glauber steps
- 10K training steps on FoVer corpus (exp1055 deliverable)
- Two go/no-go tests:
  - **Manifold Validity**: 50 Glauber steps from valid `z`, decoder
    output stays semantically meaningful
  - **AND-composition smoke test**: k=3 simple verifiers (lengths,
    syntax, no-banned-tokens) AND-composed in z-space; verify
    constraint satisfaction in decoded output >70%
- Honest verdict tokens:
  - `dae_debm_proto_validates` — both tests pass, scale to .86
  - `dae_debm_proto_partial_dead_zones` — manifold test fails;
    investigate corpus size or `d`
  - `dae_debm_proto_constraint_violation` — AND-composition fails;
    energy summation in z-space doesn't compose as theorized

## Cross-references

- Round 1 result: `phase3-encoder-architecture-deep-think-results.md`
- Round 2 result: `phase3-encoder-architecture-deep-think-followup-results.md`
- Hardware verification: `hardware/kv260/ising_sampler_v2.v`
- Original Phase-3 architecture: `project_phase3_architecture_complete.md`
- α_t grounding: `project_zenil_alpha_grounding.md`
