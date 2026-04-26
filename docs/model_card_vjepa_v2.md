# Model Card: VJEPA v2 (Variational JEPA Predictor v2)

## Model Description

VJEPA v2 is Carnot's second-generation Variational Joint-Embedding Predictive
Architecture verifier. It operates as a post-hoc constraint checker: given a
completed chain-of-thought (CoT) reasoning trace, it produces a violation
probability indicating whether the trace contains an arithmetic or logical
constraint error.

**Model ID:** Carnot-EBM/vjepa-v2
**Architecture:** Variational encoder + GRU prior + linear classifier
**Framework:** JAX / Flax
**Serialization:** safetensors
**Experiment lineage:** Exp 877 (V-JEPA architecture) -> Exp 883 (cascade) -> Exp 884 (v2 deploy)

## Architecture Details

The model consists of three components:

**Encoder q(z_t | x_t):** A 2-layer MLP that maps TF-IDF bag-of-words features
(vocab_size=50) to a Gaussian posterior. Instead of a single point estimate, the
encoder produces (mu, log_var) pairs — this forces the model to represent
uncertainty explicitly. On out-of-distribution (OOD) inputs the posterior variance
grows, signalling that the model is uncertain rather than silently producing a
confident wrong answer.

    Input (50-dim TF-IDF) -> Linear(128) -> ReLU -> Linear(64) -> ReLU
                           -> mu head: Linear(32)
                           -> log_var head: Linear(32)

**Prior p(z_t | c_{t-1}):** A GRU cell that predicts the expected latent
distribution from the prior context vector. This allows the model to integrate
information across multiple verification steps in a pipeline.

    Context (context_dim) -> GRU(64) -> mu head: Linear(32), log_var head: Linear(32)

**Classifier:** A single linear layer with sigmoid activation that maps a sampled
latent vector z to a violation probability in [0, 1].

    z (32-dim) -> Linear(1) -> sigmoid -> violation_prob

**Loss function (Variational ELBO):**

    L = BCE(classifier(z), label) + 0.1 * KL[q(z|x) || p(z|c)]

The KL weight of 0.1 follows the beta-VAE convention, balancing reconstruction
quality against latent-space regularisation. The KL term prevents posterior
collapse: even on OOD inputs the model distributes probability mass across the
latent space rather than concentrating at a single point.

## Training

- **Training corpus:** GSM8K (grade-school math) and MATH (competition math)
  correct/incorrect CoT trace pairs
- **Training pairs:** 146 labeled (correct trace, incorrect trace) pairs
- **Epochs:** 200
- **Optimizer:** Adam via Optax
- **Hardware:** CPU-only training (no GPU required for this model tier)
- **Training duration:** ~10 seconds on a single CPU core

## Evaluation

| Metric | Value |
|--------|-------|
| OOD AUC (held-out ARC + SVAMP) | 0.9211 |
| In-distribution AUC (Exp 883) | 0.664 (JEPA v1 baseline) |
| Heldout seed | 999 |
| Heldout ARC examples | 10 |
| Heldout SVAMP examples | 10 |

The OOD evaluation uses domains (ARC science questions, SVAMP word problems) that
were not present in the GSM8K/MATH training corpus. An AUC of 0.9211 on OOD data
indicates strong generalisation to novel arithmetic constraint patterns.

## Cascade Integration

VJEPA v2 serves as a Tier 2 gate in Carnot's verification cascade:

1. Tier 1 (fast): EstimationVerifier — range plausibility check (SVAMP AUC=0.90)
2. Tier 2 (medium): VJEPA v2 — variational post-hoc CoT violation detection (OOD AUC=0.9211)
3. Tier 3 (expensive): Full energy-based model sampling

A trace passes the cascade only if it clears all tiers. The streaming variant
(VJEPAStreamingLogitsProcessor, REQ-VERIFY-177) applies soft logit penalties
during generation rather than post-hoc rejection.

## Limitations

- **CPU-only training:** The training corpus is small (146 pairs), making CPU
  training fast enough that GPU is not needed. Larger corpora will require GPU.
- **TF-IDF features only:** The model operates on bag-of-words token frequency
  vectors (vocab_size=50, hash-bucketed), not on full token embeddings. This
  loses word order and semantic content but keeps inference extremely fast.
- **GSM8K/MATH training bias:** The model was trained on grade-school and
  competition math traces. Performance on non-arithmetic domains (code, logic,
  science) is not measured and may be lower.
- **Small held-out set:** The OOD evaluation used 10 ARC + 10 SVAMP examples.
  The AUC estimate has high variance; a larger evaluation is warranted before
  claiming production-grade reliability.
- **No calibration:** The violation_prob output is not calibrated against a
  held-out frequency distribution. Use it as a ranking signal, not an absolute
  probability.

## Intended Use

VJEPA v2 is intended for use in Carnot's verification pipeline as a constraint
checker for arithmetic and mathematical reasoning traces. It is not intended for:

- Production safety-critical decisions without a human in the loop
- Non-arithmetic domains without further evaluation and fine-tuning
- Use as a replacement for formal verification or symbolic solvers

## Hardware Portability

The architecture uses only GEMM + ReLU operations, making it compatible with
NPU accelerators, FPGAs (iCE40 / ECP5 / Xilinx KV260), and future Extropic
XTR-0 hardware. This is a deliberate design choice: Phase 2 of the Carnot
roadmap targets hardware-accelerated energy computation on open FPGA platforms.

## Decentralization

Per Carnot's decentralization policy (CLAUDE.md rule 3), this model is published
to at least two independent distribution channels:

- HuggingFace Hub: https://huggingface.co/Carnot-EBM/vjepa-v2
- Gitea mirror: ssh://git@gitea.noblehunt.org:2222/ianblenke/carnot.git

The model weights are stored in safetensors format with no vendor lock-in. The
training code and inference code are Apache 2.0 licensed and depend only on
open-source libraries (JAX, Optax, NumPy).

## Specifications

- REQ-VERIFY-175 (variational encoder)
- REQ-VERIFY-176 (KL regularisation)
- REQ-VERIFY-145 (cascade deployment)
- SCENARIO-VERIFY-233, SCENARIO-VERIFY-234 (OOD evaluation)
