# Phase-3 DAE-DEBM Round 3 — Deep Think Prompt

**Status:** Drafted 2026-04-30 after the architectural pivot from
continuous DBAE-EBM to discrete DAE-DEBM. Ready for paste when
quota resets.

**Why a third round:** the discrete pivot creates new implementation
unknowns that Round 2's continuous-EBM answers don't cover.
Specifically the choice of straight-through estimator, discrete-EBM
training algorithm, latent dimension, β-schedule, and dead-zone
navigation in `{-1,+1}^d`.

---

## Prompt to paste

You previously chose Candidate D (Deterministic Bounded Autoencoder
+ Latent EBM, DBAE-EBM) for Carnot's Phase-3 Kona-parity foundation
model, and in Round 2 you flagged a critical architectural pivot:
*"if Phase-2 FPGA specifically executes discrete Glauber spin-flips
on a ±1 lattice, a continuous EBM is structurally mismatched."*

Hardware verification has now confirmed the condition. Carnot's
KV260 `ising_sampler_v2.v` is unambiguously a synchronous parallel
Glauber sampler on a ±1 lattice (1 bit per spin, sigmoid-LUT +
LFSR comparison, β-annealed discrete temperature, no continuous
spin variables, no gradient hardware). The pivot is taken.

The new architecture is **DAE-DEBM**:
```
text → encoder logits → STE: z = sgn(logits) ∈ {-1, +1}^d
     → EBM(z) trained via exact Gibbs/Glauber sampling (matching the FPGA)
     → z' → decoder → text'
```

Most of your Round 2 recipe transfers verbatim — masked-token
reconstruction, the 3-stage training (manifold warmup → EBM warmup →
asymmetric finetune with stop-gradient on `z_fake`), the β=0.01
gradient-magnitude ratio, the 100M-param prototype on the 6,500-pair
FoVer corpus. The transpilation gap collapses to zero by
construction (the encoder output IS the hardware spin state). Round
2's variance penalty becomes redundant because `var(±1) = 1.0`
exactly.

Five new implementation questions remain. Each blocks the prototype.

### Question 1 — Straight-through estimator variant

Which STE for `z = sgn(logits)`?

- **Identity STE (Bengio 2013)**: `∂z/∂logits = 1`. Simplest. Ignores
  the discontinuity; gradient flows through `sgn()` as if it were
  identity.
- **Hinton's saturation-aware STE** (e.g., `1[|logits| < 1]`): pass
  gradient only when activations haven't saturated; clip elsewhere.
  Reduces gradient noise on saturated cells.
- **Hard tanh STE** (`∂z/∂logits = 1[|logits| < 1]`): equivalent to
  Hinton in practice.
- **Gumbel-Softmax → sgn** (relaxed during training, hard at test
  time): smooth proxy with temperature annealing, then sgn() at
  deployment. Higher gradient quality, but introduces a
  training-test gap.
- **Concrete distribution / Categorical reparameterization**: full
  variational treatment, ditches the deterministic AE.

Which is right for Carnot's setup (encoder is RoBERTa-init,
decoder must reconstruct natural language, EBM must compose k=15+
verifier energies)? If the answer depends on `d` (latent dimension)
or training-set size, say so.

### Question 2 — Discrete EBM training algorithm

Discrete EBMs at LLM scale are sparsely-explored territory. Which
training method is most stable on language data?

- **PCD (Persistent Contrastive Divergence)** with chain-length
  100 Glauber steps. Standard for discrete EBMs but mode collapse
  is common.
- **NCE (Noise Contrastive Estimation)**: trains a binary classifier
  to distinguish data from a known noise distribution. Avoids MCMC
  entirely. Quality depends on noise distribution.
- **Concrete Score Matching** (Meng et al., adapts score matching
  to discrete domains via the concrete distribution).
- **Maximum Likelihood with exact Gibbs** (only feasible for small
  `d`; combinatorial blow-up beyond `d ≈ 20`).
- **Hybrid: PCD with NCE warmup** (use NCE to initialize the EBM,
  then PCD for fine-tuning).

Which has the best published track record for **language-modelling-
adjacent** discrete state spaces? Specifically: has anyone shipped a
discrete EBM at >10M parameters on text data?

### Question 3 — Latent dimension `d`

Continuous EBMs have soft regularization on `d` (over-large `d`
costs MCMC mixing time but doesn't break the model). Discrete EBM
on `{-1,+1}^d`:

- **Too small** (`d ≤ 64`): expressivity ceiling. The 2^d state
  space is too small to encode reasoning chains.
- **Too large** (`d ≥ 1024`): Glauber mixing on 2^1024 states
  becomes intractable; the trajectory length needed to traverse
  semantically-distant states grows exponentially.

What's a principled `d` for a Phase-3 Kona-parity model trained on
1B tokens with k=15 verifier AND-composition? Or is the right answer
"sweep `d ∈ {128, 256, 512}` and pick by validation perplexity +
manifold-validity gate"?

If the answer is "depends on AND-composition rank":
- The k=15 verifiers each contribute a constraint manifold in
  `{-1,+1}^d`. Their AND is the intersection.
- For the intersection to be non-empty AND non-trivial, `d` must be
  ≥ some function of k. What's that function?

### Question 4 — Glauber β-schedule during training

The KV260 hardware uses β-annealing (hot to cold) at inference time
to find low-energy spin configurations. During *training*, should
the EBM update rule use:

- **Static β = 1** (standard): treat training as a single-temperature
  problem.
- **Match the hardware schedule** at training time: anneal β across
  PCD steps so the model learns the actual deployment trajectory.
- **β-curriculum**: start high temperature (β small, near-uniform
  state), gradually cool over many epochs. Avoids early collapse to
  a single low-energy mode.
- **Multi-temperature training** (parallel tempering): maintain
  multiple chains at different β simultaneously, swap states
  between chains.

Does the verifier-grounded gradient (`E_total = Σ E_i(z)`) survive a
hot-to-cold trajectory during training? Specifically, does
`inf_t α_t > 0.1` (the Round 1 acceptance gate) measure differently
under variable-β vs static-β training?

### Question 5 — Discrete manifold dead-zone navigation

In continuous `[-1, 1]^d`, MCMC can interpolate smoothly between
isolated low-energy modes — the trajectory passes through
intermediate (possibly meaningless) `z` values. The Round 2
"Manifold Dead-Zone" test checks whether the decoder gracefully
handles this.

In discrete `{-1, +1}^d`, there are no intermediate states. Every
Glauber step flips one or more spins, jumping directly between
discrete configurations. Two failure modes:

(a) **Sparse semantic structure**: only a tiny fraction of the
   2^d possible latents are decodable to meaningful text. The
   Glauber sampler will spend most of its time in
   "meaningless-text" regions.

(b) **Long-chain requirement**: to traverse from one
   meaningfully-decoded `z` to another (e.g., one valid reasoning
   step to the next), a long Glauber chain is needed. The chain
   must pass through "dead zone" intermediate states without
   getting trapped.

What's the minimum Glauber chain length expected for a 100M-param
prototype to navigate dead zones reliably? Is the Manifold Dead-Zone
test pass criterion (decoder output meaningful after 50 steps)
realistic for discrete latent, or should it be 500 or 5000?

### Bonus question — Prior art

Has anyone shipped a discrete autoencoder + discrete EBM combination
at any meaningful scale on text data? Specifically:

- VQ-VAE + EBM hybrids: known territory, but VQ-VAE uses learned
  codebook clusters, not `{-1, +1}^d` raw spins. Are the lessons
  transferable?
- BigGAN-discrete + EBM: image-only.
- Anything else?

If the answer is "no, this is novel territory," that's important
context for the prototype — the implementation risk is higher than
adapting known work.

## Output format

Five numbered answers + the bonus, each ≤ 250 words, with concrete
recommendations we can implement in code. If any question doesn't
have a clean answer, say so and identify what we'd need to know to
answer it.

If any answer suggests the DAE-DEBM pivot itself was premature
(e.g., "discrete EBMs at 100M parameters fundamentally don't train
stably; reconsider continuous-with-accepted-transpilation-loss"),
flag it explicitly. We'd rather take that regret now.
