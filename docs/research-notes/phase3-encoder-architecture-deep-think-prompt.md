# Phase-3 Encoder Architecture — Deep Think Prompt

**Status:** Drafted 2026-04-30 in response to user question "Does it
make sense to use a VAE model to translate to/from our Kona parity
EBM model?". Ready for paste into Deep Think when quota resets.

**Decision needed:** which encoder/decoder architecture should bridge
discrete language and the continuous-latent EBM in Carnot's Phase 3
Kona-parity foundation model?

---

## Prompt to paste

You are advising the Carnot research project on a Phase 3
architectural decision. Carnot's three-phase vision:

- **Phase 1 (current, shipping):** verifier-only EBM that scores LLM
  outputs against constraint-grounded energy. Operates on discrete
  text/code outputs. No latent space, no generation.
- **Phase 2 (in progress):** hardware-accelerated energy evaluation
  via FPGA (KV260, ASIC), photonics, etc. Still operates on outputs.
- **Phase 3 (the question):** open-source foundation model with
  *Kona-style continuous latent reasoning* — non-autoregressive,
  iterative, self-correcting. Replaces Kona's diffusion-based
  reasoning with energy-based reasoning while preserving the
  continuous-latent + iterative properties.

Phase 3 needs an encoder/decoder bridge between discrete tokens and
the continuous space the EBM samples in. Three candidates:

### Candidate A — VAE encoder + EBM in latent

```
text → VAE encoder → z (continuous, ~Gaussian-prior) → EBM(z) MCMC → z' → VAE decoder → text'
```

- Pros: clean separation of concerns; well-studied; latent EBM has
  smooth gradient for Langevin/Gibbs sampling.
- Cons: VAEs on language hit posterior collapse (Bowman 2015,
  OPTIMUS 2020); the encoder ignores `z` and the decoder becomes a
  language prior; reasoning chains lose information at the
  bottleneck.
- Prior work: Pang et al. NeurIPS 2020 "Latent Space Energy-Based
  Prior", Vahdat et al. LION 2022 (3D shapes, not language).

### Candidate B — Diffusion encoder + EBM as score prior

```
text → embedding → noise schedule (forward diffusion) → reverse via EBM-as-score → embedding → text'
```

- Pros: matches Kona's actual architecture; no information bottleneck
  (iterative refinement preserves content); the EBM training signal
  can be wired through score matching.
- Cons: harder to combine an explicit *energy* function (Carnot's
  ground truth) with score-based sampling; the EBM's α_t grounding
  signal (Zenil-style, see project_zenil_alpha_grounding.md) needs to
  survive the noise schedule.
- Prior work: DiffuLM, Plaid, Diffusion-LM, VDM.

### Candidate C — Direct EBM on token embeddings

```
tokens → embedding lookup → EBM(emb) MCMC → emb' → nearest-neighbor decode → text'
```

- Pros: simplest; no extra encoder/decoder to train; the EBM IS the
  whole architecture.
- Cons: discrete-ish embedding space makes MCMC hard; nearest-
  neighbor decode is lossy and discontinuous.
- Prior work: not much, because it's hard.

## Key constraints from Carnot's specifics

1. **Verifier-grounded training signal.** Carnot's energy function is
   the ground truth — the architecture must let the energy gradient
   reach all the way through encoder + EBM + decoder during training.
   In a VAE+EBM hybrid, the ELBO/KL loss competes with the energy
   loss. In a diffusion+EBM hybrid, score matching has to be
   reconciled with the explicit energy function.

2. **α_t grounding is non-negotiable.** Per Round 7 Deep Think
   (project_zenil_alpha_grounding.md), self-distillation collapses
   without `inf_t α_t > 0`. Whichever architecture wins must track
   α_t as a first-class metric. VAE bottleneck would crush α_t;
   diffusion preserves it across timesteps; direct-EBM has no α_t
   structure to begin with.

3. **AND-composition over hundreds of verifiers** (Phase 3
   architecture per project_phase3_architecture_complete.md, k=15
   verifiers minimum). Whichever encoder is chosen has to support
   composing energy functions from many verifiers, each with its own
   learned head. VAE latent supports this naturally (multiple
   energy terms summed in z-space). Diffusion supports it less
   directly. Direct-EBM is the most flexible but has the worst MCMC
   properties.

4. **Hardware portability** (REQ-KONA-006, FPGA/ASIC/photonics path
   per Phase 2). The architecture has to compile to fixed-function
   energy hardware. Continuous-latent EBM with bounded `z` dimension
   maps cleanly to Ising/Potts arrays. Token-embedding EBM doesn't.
   Diffusion mapping is unclear.

5. **Continuous-Boolean Transpilation Gap** (project memory: one of
   the three irreducible Phase-3 → Phase-8 open problems). The
   architecture choice may interact with how this gap closes —
   continuous z is naturally Booleanizable via threshold, diffusion
   embeddings are noisier.

## Question to Deep Think

Given those constraints, which of A/B/C is the right Phase 3
architecture for Carnot? Or is there a fourth option I'm missing
(e.g., flow-based encoder, normalizing-flow + EBM hybrid, structured
latent like a discrete VQ-VAE)?

If A (VAE-EBM): how do we mitigate posterior collapse on long
reasoning chains? Is the OPTIMUS-style two-stage training enough?

If B (diffusion-EBM): how do we wire Carnot's explicit energy
function into score matching without losing the verifier-grounded
training signal? Is α_t grounding compatible with noise schedules?

If C (direct EBM): what MCMC method works on discrete-ish embedding
space? Has anyone shipped this at LLM scale?

If a fourth option: name it and argue it.

Output format: 4-paragraph response with explicit pick + 2 fallbacks,
then a falsifiable acceptance gate that would tell us whether the
chosen architecture is working at the scale Carnot needs (millions of
parameters, ≥1B-token training corpus equivalent).
