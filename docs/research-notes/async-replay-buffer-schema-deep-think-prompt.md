# Deep Think Round 3 Prompt — Async Replay Buffer Schema for Phase-3 Joint Fine-Tune

**Status:** Ready to send. This is the Round 3 question recommended
by Phase-3 Round 2's Q7. Engineering-bottleneck question with
mathematically well-defined optimization criteria.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-3 prototype training pipeline integrates
a Deterministic Bounded Autoencoder (DBAE) decoder with a Latent
Energy-Based Model (EBM) and a verifier cascade (k≈8 verifiers
across topologically distinct mechanism families: Z3-AST formal
verification, gVisor runtime execution, semantic embedding,
ThinkPRM step-level, JSON schema, Potts combinatorial, NUP, and
SOSKANEnergyV3).

A prior Deep Think round (Phase-3 Round 2, dated 2026-05-01)
identified a **fatal GPU-starvation bottleneck** in the originally-
proposed 2-stage live joint training:

- DBAE forward/backward pass: ~15 ms/step on 0.8B GPU, ~50-100 ms
  on 35B GPU
- Z3-AST verifier evaluation: 0.5-3.0 seconds per output
- gVisor runtime execution: 1-10 seconds per output
- Semantic embedding: 50-200 ms per output

**With synchronous live verifier calls, the GPU idles >98% of
training time, expanding a 10-hour training loop into weeks.**
Phase-3 Round 2 prescribed a 4-stage schedule:

1. DBAE pre-train (standard)
2. EBM warm-up (offline contrastive on static cached negatives)
3. Joint fine-tune with FROZEN decoder + async replay buffer
4. Full joint fine-tune (unfrozen, very low learning rate)

**Stages 3 and 4 require an async replay-buffer architecture that
decouples verifier latency from gradient updates while preventing
the DBAE decoder from Goodharting stale EBM energy states.**

### The question

**What is the mathematically optimal asynchronous replay-buffer
schema for stages 3-4 of Phase-3 training?**

The load-bearing concerns:

1. **Decoupling verifier latency from gradient updates** — verifier
   evaluations of 0.5-10 s cannot block 15-100 ms model steps.
2. **Goodhart resistance** — if the decoder produces output X at
   training step t, gets EBM energy evaluated at step t, but the
   verifier verdict on X arrives at step t+N (N ~ 100-1000 steps),
   the EBM may have evolved to no longer correspond to the
   decoder's intent. The decoder can exploit this temporal
   mismatch.
3. **Sample efficiency** — with verifier costs of 1-10 s per
   output, the buffer cannot afford to discard samples; importance
   weighting matters.
4. **Stationarity vs adaptivity** — too-stale samples teach the
   wrong thing; too-fresh sampling re-introduces the GPU-starvation
   problem.
5. **Memory footprint** — at training scales of 10⁵-10⁶ verified
   samples, naive replay buffers exceed GPU memory; eviction
   policy matters.

### Sub-questions to address

#### Q1. Optimal sampling distribution

Given a replay buffer with samples {(x_i, E_i, V_i, t_i)} where
x_i is decoder output, E_i is EBM energy at sample time, V_i is
verifier verdict, t_i is sample-time index, what is the optimal
sampling distribution for the next gradient update? Specifically:

- **Uniform sampling** baseline: equal probability for all buffer
  entries.
- **Prioritized experience replay** (PER, Schaul et al. 2016):
  P(i) ∝ |TD error|^α; what's the equivalent of "TD error" for
  EBM contrastive training? Is it |E_i_now − E_i_at_sample_time|
  (energy drift)? Or |E_i − target_energy(V_i)| (current TD-style
  error)?
- **Importance-weighted policy gradient** (PPO-style):
  P(i) ∝ π_now(x_i) / π_at_sample(x_i), with importance ratio
  clipping. Does this apply to EBM training where the "policy" is
  the implicit decoder distribution?
- **Recency-decay weighting**: P(i) ∝ exp(−λ(t_now − t_i)).
  Decays stale samples but loses sample-efficiency.

Derive the optimal schema mathematically. State assumptions clearly.

#### Q2. Goodhart resistance bound

Define Goodhart-distance for stage 3: the decoder produces output
x_t at step t with energy E_t. The verifier returns verdict V_t
at step t+N. The EBM at step t+N has parameters θ_{t+N} ≠ θ_t.
The decoder optimizes against the EBM signal at step t+N, but the
verifier verdict was on x_t which was sampled from energy
landscape θ_t.

**What is the maximum N (delay window) for which the decoder cannot
exploit the temporal mismatch?**

Specifically:

- Bound N as a function of EBM gradient norm ‖∇θ E‖ and decoder
  learning rate η_decoder.
- Derive the spectral radius of the gradient-update operator
  during stage 3.
- Show conditions under which Lyapunov stability holds.

#### Q3. Buffer eviction policy

When the buffer is full and a new sample arrives, what gets
evicted?

- **FIFO**: oldest first. Loses long-tail rare-event coverage.
- **Energy-stratified**: maintain k buckets by E_i; evict from
  most-populated bucket. Preserves coverage of hard negatives.
- **Verifier-stratified**: maintain k buckets by V_i; evict from
  most-populated. Preserves cascade-diversity.
- **Reservoir sampling**: random; theoretically uniform asymptotic.
- **Adversarial-priority eviction**: keep samples where decoder's
  current output most differs from sample-time output; these
  identify drift.

Which is mathematically optimal under the criteria of (a) sample
efficiency, (b) Goodhart resistance, (c) memory footprint?

#### Q4. Asynchronous architecture topology

Multiple architectural patterns are plausible:

**Pattern A (single-buffer, multi-producer):** decoder produces
samples, K parallel verifier workers consume from a queue,
buffer absorbs verified samples, training loop samples from buffer.

**Pattern B (multi-stage pipeline):** decoder → verifier-pass-1
queue → verifier-pass-2 queue → … → verifier-pass-K queue →
buffer → training. Each verifier pass is its own worker pool.

**Pattern C (decoupled-EBM):** EBM trains on its own time-axis from
buffer samples, decoder trains on its own time-axis from
EBM-evaluation queue, with explicit synchronization barriers
every M steps.

For each, give:
- Throughput in samples/sec under realistic Carnot verifier latencies
- Goodhart vulnerability (does decoder outpace EBM updates?)
- Engineering complexity (lines of code, distributed-systems risk)

#### Q5. Empirical validation experiment

Specify a single 1-2 milestone experiment that empirically
validates the recommended schema on the 0.8B model tier (smoke
tier). Include:

- Corpus (size, source, balance)
- Verifier suite (which subset of k=8)
- Buffer parameters (size, sampling rule, eviction rule)
- Convergence metric (training loss + held-out AUROC + held-out
  Goodhart-test pass rate)
- Acceptance criterion (a clear-cut threshold that distinguishes
  "buffer works" from "buffer fails")

### Constraints from CLAUDE.md (do not violate)

- **Decentralization rule 1 — local-first:** the schema must be
  implementable using PyTorch + standard Python multiprocessing,
  no proprietary distributed-training framework.
- **Decentralization rule 2 — closed-frontier-model integration
  optional:** Z3, gVisor, etc. are open-source and acceptable as
  verifier dependencies.
- **Phase Prototype + Empirical Validation + Adversarial Check
  Discipline:** the schema must include diagnostic instrumentation
  for every stated theoretical concern.
- **MANDATORY SOTA model tier:** validation experiment may use
  0.8B for smoke but headline results must be 35B.

### Reference literature to consider

- Schaul et al. 2016 (Prioritized Experience Replay)
- Lin 1992 (Self-Improving Reactive Agents — replay buffer origin)
- Mnih et al. 2015 (DQN)
- Schulman et al. 2017 (PPO importance sampling)
- Kuznetsov et al. 2020 (Truncated Quantile Critics — buffer
  uncertainty)
- Du & Mordatch 2019 (EBM training with replay buffer +
  short-run MCMC)

The Du & Mordatch reference is particularly relevant — they
demonstrate replay-buffer-based EBM training works in practice;
the question is the OPTIMAL schema for verifier-latency-bounded
fine-tuning.

### What I am NOT asking

- I am NOT asking for a new theoretical defense layer beyond the
  existing Phase-3 → Phase-7 architecture.
- I am NOT asking to revisit the 4-stage training schedule from
  Phase-3 Round 2 (it's the load-bearing baseline; refine within
  it).
- I am NOT asking for hyperparameter sweeps; that's empirical work.
- I am NOT asking for distributed-systems engineering details
  beyond schema-level guidance.

### Output format

1. **Executive summary** (1-2 paragraphs naming the recommended
   schema)
2. **Q1 answer** (sampling distribution with derivation)
3. **Q2 answer** (Goodhart resistance bound with stability proof)
4. **Q3 answer** (eviction policy ranked across criteria)
5. **Q4 answer** (architecture topology comparison + recommendation)
6. **Q5 answer** (empirical validation experiment specification)
7. **Risk register** — top 3 ways the recommended schema can
   silently fail (analogous to Round 1 risk registers), each with
   the cheap diagnostic that would detect it.

### Honesty requirement

If the recommended schema cannot prevent decoder Goodharting under
realistic Carnot verifier latencies (0.5-10 s), say so explicitly
and recommend a stage-3 redesign (e.g., decoder-frozen-throughout
+ shorter EBM-only fine-tune). The honest "the schema doesn't fully
solve the bottleneck, here's the residual constraint" answer is
more valuable than a confident-but-wrong prescription.

Per the project's documented Deep-Think prediction pattern
(qualitative claims well-calibrated, specific prescriptions
systematically wrong), prefer to derive bounds rigorously rather
than prescribe specific buffer-size numbers without empirical
basis.
