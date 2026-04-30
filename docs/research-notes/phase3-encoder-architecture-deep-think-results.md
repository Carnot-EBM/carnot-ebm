# Phase-3 Encoder Architecture — Deep Think Result

**Date:** 2026-04-30
**Prompt:** `phase3-encoder-architecture-deep-think-prompt.md`

## Verdict — Candidate D: Deterministic Bounded Autoencoder + Latent EBM

```
text → deterministic encoder → bounded z (tanh into [-1, 1]^d) → EBM(z) MCMC → z' → deterministic decoder → text'
```

Deep Think's key insight: introduce a **fourth option** that wasn't in
the prompt. Drop the variational stochasticity of VAE entirely; keep
only the encoder-decoder pair, deterministic, with a **bounded
continuous latent** in `[-1, 1]^d` (tanh-squashed).

### Why D dominates the three originals

1. **No posterior collapse.** Eliminating the Gaussian prior + ELBO
   removes the competing uninformative-language-prior pull. The
   decoder cannot ignore `z` because there's no alternative
   information path. Fixes the Bowman 2015 / OPTIMUS 2020 failure
   mode by construction.

2. **Continuous-Boolean Transpilation Gap solved by construction.**
   `[-1, 1]^d` thresholds via `sgn(z)` to binary spin states. The FPGA
   Ising array (Phase 2, KV260, exp1068/1081) operates on exactly
   those binary spins. No separate transpilation layer needed.

3. **AND-composition over k≥15 verifiers is clean.** Energy in
   z-space sums: `E_total(z) = Σ_i E_i(z)`. Each verifier contributes
   a continuous energy manifold; the joint minimum is the AND of the
   constraints. Phase-3 Round 9 architecture (k=15+ verifiers per
   `project_phase3_architecture_complete.md`) maps directly.

4. **Verifier gradient survives bottleneck.** Deterministic bridge =
   no stochastic dilution; the verifier's energy gradient
   backpropagates from text → z → encoder weights cleanly during
   training.

### Fallback 1 — Candidate B (Diffusion + EBM as score prior)

If DBAE's deterministic bottleneck lacks semantic density at 1B-scale:

- Parameterize the score network analytically as the negative
  gradient of the verifier energy: `s_θ(x_t, t) = −∇_{x_t} E_θ(x_t, t)`.
  Score-matching loss then explicitly sculpts the true energy
  landscape. The verifier signal survives the noise schedule.
- α_t grounding survives by anchoring α_t to forward SNR decay with
  an explicit floor. Bound the scheduled noise variance to guarantee
  `inf_t α_t > 0`. Logical constraints survive every timestep.
- **Why fallback, not pick:** mapping iterative continuous noise
  schedules over vocabulary embeddings to Phase 2 fixed-function
  spatial hardware is "mathematically hostile" compared to the
  static bounded `z`-space of D.

### Fallback 2 — Candidate A (VAE-EBM, but only with major changes)

If probabilistic decoding diversity is mandatory:

- **OPTIMUS-style two-stage training is insufficient.** Posterior
  collapse always wins on long reasoning chains.
- Replace the Gaussian KL with **MMD-VAE / WAE** OR an explicit
  **Energy-Based Prior** (EBP) — pull `z` toward constraint
  satisfaction, not zero-information noise.

### Definitively discarded — Candidate C (Direct EBM on token embeddings)

- Balanced Barker proposals and gradient-guided discrete Langevin
  exist but no one has shipped them at LLM scale.
- Mixing times are glacially slow.
- No continuous trajectory structure → cannot track temporal α_t.

## Falsifiable Acceptance Gate — Zero-Shot Boolean Composition

The 1B-token-scale test that proves DBAE-EBM bridges discrete language
and continuous-latent EBM without losing the verifier signal:

1. Train encoder/decoder + base EBM jointly on the corpus.
2. Freeze all weights.
3. Instantiate **zero-shot AND-composition of k=15 orthogonal Phase-1
   verifiers**: `E_total(z) = Σ_{i=1}^{15} E_i(z)`.
4. On a held-out set of corrupted prompts, run **100 continuous Langevin
   MCMC steps** in z-space.

Architecture passes IFF all three:

| Condition | Threshold |
|---|---|
| **α_t survival** | `inf_t α_t > 0.1` across the entire 100-step trajectory |
| **Bottleneck integrity** | Decoder maps minimized `z'` to text with **>85% joint-constraint pass rate**, no catastrophic perplexity degradation |
| **Transpilation readiness** | Binarizing `z'` via `sgn(z')` before decoding degrades the k=15 pass rate by **<5% absolute** |

The third condition is the load-bearing one for Phase 2 — it proves
the topology is hardware-ready for FPGA/ASIC fixed-function
compilation without retraining.

## Implications for Carnot

### Position paper (Phase 3 section)

The architectural choice is now load-bearing. Update the Phase 3
section of `exp1078`'s position-paper-v2 (when it lands) to cite
DBAE-EBM as the Phase 3 substrate, with B and A as fallbacks. Cite
this Deep Think round.

### Phase 2 hardware story

DBAE makes the FPGA story tighter: the same `[-1, 1]^d` latent that
the EBM operates in *is* the spin lattice. No separate transpilation
step. exp1081 (KV260 Ising at N=64..1024) is the prerequisite that
proves the lattice can run; DBAE-EBM is the architecture that uses it.

### α_t grounding (Round 7 / `project_zenil_alpha_grounding.md`)

The acceptance gate's first condition (`inf_t α_t > 0.1`) wires the
α_t self-distillation theorem directly into Phase 3's go/no-go.
This is exactly the "track α_t as first-class metric" mandate from
the Round 7 memory.

### Continuous-Boolean Transpilation Gap

One of the three irreducible Phase-3→Phase-8 open problems
(`project_phase3_architecture_complete.md`). DBAE's `sgn(z)`
construction *closes* it. Verify in the acceptance gate's third
condition.

## Next experiment seeds

1. **Small-scale DBAE prototype.** Train a deterministic
   encoder/decoder pair + EBM on a small reasoning corpus (10K-100K
   tokens). Verify α_t > 0 on holdout. ~1 milestone slot.
2. **MMD-EBM ablation.** Same corpus, swap to MMD-VAE encoder.
   Compare α_t survival + reasoning accuracy. Tells us whether
   stochasticity adds value or just collapse risk.
3. **Boolean transpilation test.** On the small DBAE, measure the
   `sgn(z)` degradation. If <5% pre-Phase-2, FPGA compilation is on
   track.

These are .85+ candidates after the FPGA-vs-GPU baseline lands.
