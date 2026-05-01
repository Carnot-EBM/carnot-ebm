# Deep Think Results — Phase-3 Architecture Path Given Mixed Empirical Evidence

**Status:** Returned 2026-05-01.
**Paired with:** `phase3-empirical-grounded-deep-think-prompt.md`
**Headline:** Phase-3 DBAE-EBM prototype is **NOT YET READY** for
construction. **Block-and-Resolve pivot mandated.**

---

## 1. Executive Summary

**Honesty Declaration:** The Phase-3 DBAE-EBM prototype is NOT YET
READY for construction or empirical validation.

Proceeding with the original Phase-3 specification given the
2026-04-30 empirical data mathematically guarantees a silent
prototype failure. The empirically confirmed KL=3.07 parallel-Glauber
divergence means the hardware cannot sample the required Boltzmann
distribution, poisoning any EBM training objective. Furthermore, the
empirical correlation of r=0.66 across the Tier-0 verifiers
fundamentally invalidates the architectural assumption that
AND-composition over k=15 verifiers will exponentially shrink the
joint null space.

To salvage the Phase-3 prototype, we must execute a mandatory
**Block-and-Resolve pivot**:

1. Construction must be explicitly BLOCKED until Phase-1d (Verifier
   Diversity Expansion) establishes a topologically distinct k=5
   verifier suite with r < 0.3.
2. The fatally flawed KV260 arbitrary-J parallel Glauber must be
   deprecated to a sequential proof-of-concept.
3. The production hardware path must pivot entirely to the Extropic
   Z1.
4. The prototype must then be trained via contrastive divergence on
   the 0.8B model tier using a strictly CPU/Simulator-based sampler
   to prevent hardware noise from destroying the energy gradients.

## 2. Q1 Answer: AND-composition feasibility ceiling for k

**Theoretical bound and feasibility ceiling: realistic k_max = 5-7
(NOT 15).**

The assumption that k=15 verifiers can maintain low mutual
information violates the information bottleneck principle. The
generative output manifold of an LLM has a finite intrinsic
dimensionality of "correctness" failure modes (syntax, logic,
semantic reality, execution trace). By the pigeonhole principle for
feature extraction, extracting k > 7 independent features from a
space with D_int ≈ 5 forces the marginal verifiers to become linear
combinations of the existing basis. Thus, correlation r unavoidably
approaches 1.0 as k scales, providing zero marginal reduction in
the joint null space.

### Verifier mechanism families for r < 0.5

To break the r=0.66 correlation floor against NUP/SpilledEnergy/KAN
text probes, abandon text-surface heuristics entirely.

**Orthogonal candidates (r < 0.3):**
- **Z3-AST formal verification** — evaluates structural logic
  satisfiability
- **Runtime-execution probe** — evaluates physical empirical state
- **Combinatorial-encoding probe** — evaluates topological graph
  constraints via Ising/Potts

**Highly correlated candidates (REJECT, r > 0.5):**
- Semantic-embedding probe
- ThinkPRM step-level probe
- Both share the exact underlying neural representation and
  linguistic token distributions as the Tier-0 probes.

**Marginal candidates (r ≈ 0.3-0.4):**
- Output-format probe — orthogonal but easily spoofed/shallow.

### Friedrichs-angle / Joint Null-Space tradeoff

**AND-composition over k=5 with r<0.3 is mathematically and
geometrically superior to k=15 with r=0.55.**

Let θ_F be the Friedrichs angle between the verifiers' sub-spaces.
The pairwise correlation r closely approximates cos(θ_F). A high
correlation (r=0.55) implies the sub-spaces are nearly parallel
(small angle); their intersection (the joint null space) decays
sub-linearly and is highly vulnerable to manifold-walking. At
r < 0.3, the sub-spaces are highly orthogonal. The joint volume
shrinks exponentially as ~O(cos^k(θ_F)), granting stronger
adversarial bounding with a fraction of the Pareto-suboptimal
cascade rejection costs identified in exp1100.

## 3. Q2 Answer: FPGA Glauber sampler architectural rescue

Given the KV260 is purely a POC and Extropic Z1 is the production
hardware, the priority is to rescue exact detailed balance without
wasting budget on dead-end FPGA DSP overhead.

### Rank 1: Abandon FPGA Glauber, redirect Phase-2 budget to Extropic Z1

- **Evaluation:** Best strategic path. Z1 natively utilizes analog
  continuous-time thermodynamic noise, resolving detailed balance
  without discrete algorithmic violations. It bypasses Finding #2
  entirely.
- **Pass/Fail Criterion:** Z1 SDK Digital Twin matches exact CPU
  Gibbs with KL < 0.05 on the exp1041 12-spin graph.
- **Instrumentation:** `KLDivergenceEstimator`, new
  `ExtropicThermodynamicAPIProfiler`.
- **Cost:** 1.5-2 milestones.
- **High-priority vendor tasks:** secure Python SDK bit-exact
  simulator; define a continuous energy landscape API avoiding
  Extropic's LLM-specific wrappers.

### Rank 2: Sequential single-spin Glauber

- **Evaluation:** Fastest KV260 POC rescue. Mathematically
  guarantees detailed balance on arbitrary symmetric J. Reuses
  existing bitstreams with a trivial state-machine loop rewrite.
  Hardware speedup vs CPU is negligible or negative due to O(d)
  sequential AXI-Lite bus latency, but it unblocks the engineering
  POC requirement.
- **Pass/Fail:** KL < 0.05 against CPU baseline on hardware.
- **Instrumentation:** `KLDivergenceEstimator`, new
  `FPGALatencyTracer`.
- **Cost:** 0.5 milestones.

### Rank 3: Bipartite-block parallel Glauber

- **Evaluation:** Retains hardware parallelism but fundamentally
  restricts the encoder. AST and Numeric SAT can be forced into
  bipartite RBM topology, but requires massive numbers of hidden
  auxiliary spins. This drastically inflates spin count d, directly
  compounding Finding #1 (dimensionality guillotine) and Finding #3
  (capacity collapse).
- **Pass/Fail:** Successful bipartite mapping of all 6 constraint
  families without exceeding KV260 limits, KL < 0.05.
- **Instrumentation:** `BipartiteAuxiliaryOverheadTracker` (new).
- **Cost:** 2-3 milestones.

### Rank 4: Metropolis-Hastings

- **Evaluation:** Correct for arbitrary J, but consumes massive
  DSP slices for PRNGs and exponential ratio comparators. Severely
  limits max spin capacity.
- **Pass/Fail:** Logic utilization < 80% on KV260; KL < 0.05.
- **Instrumentation:** `FPGAUtilizationTracker`.
- **Cost:** 4 milestones.

### Rank 5: HMC on relaxed boolean — **DISQUALIFIED**

Relaxing the boolean constraints directly worsens Finding #1 and
invalidates the combinatorial logic projection (Finding #5).

### Rank 6: Heat-bath continuous-time chain — **DISQUALIFIED**

Architecturally incompatible with KV260 synchronous AXI-Lite
registers.

## 4. Q3 Answer: Phase-3 prototype training path

### Training Corpus

Must utilize Contrastive Divergence (CD). Relying on the exp1099
pre-filtered corpus caused a degenerate gradient (∇E = 0) because
the EBM lacked boundary conditions. The corpus must contain a 1:1:1
mix of:

1. Verifier-passed ground truths
2. Pre-filtered raw LLM generations
3. **Adversarial hard-negatives** — model outputs that look valid
   but fail 1 or more specific orthogonal verifiers — to push the
   EBM energy up at the boundaries.

### Verifier Suite at Training Time

**Phase-1d (Verifier Diversity Expansion) MUST strictly precede
Phase-3 training.** Do not train on the correlated r=0.66 suite;
the latent EBM will overfit to linguistic principal components and
suffer mode collapse. Deploy the achievable k=5 suite first.

### Sampler Choice at Training Time

Train entirely on **CPU Exact Gibbs OR Z1 SDK Simulator**.
Hardware-in-the-loop (FPGA or physical Z1 silicon) introduces
non-differentiable thermal/timing noise that destroys contrastive
gradients. **Migrate to hardware only for inference.**

### The 3-stage training schedule — REVISED to 2-stage

The original schedule does not survive the blind-spot audit due to
Finding #4 (Taylor spurious corners). If EBM training is isolated,
the autoencoder's continuous decoder will hide smooth "corners"
that bypass the Ising energy.

**Revised Schedule:**
1. DBAE pre-train
2. **Joint Verifier-Guided Contrastive Fine-Tune** — the EBM is
   dynamically fine-tuned simultaneously with the decoder, using
   the k=5 cascade to generate live negative samples, forcing the
   EBM to actively penalize the spurious corners.

## 5. Q4 Answer: Path to Phase-3 published claim

### Minimum Empirical State

`final_energy = 0` on a held-out test corpus + `KL < 0.05` is
**strictly necessary but insufficient**. Mode collapse easily
satisfies this. Acceptance requires appending:

1. **AUROC > 0.95** on a held-out dataset of adversarial logic
   failures.
2. **Generative Entropy / Diversity Thresholding** — verifying the
   Shannon entropy of continuous decoded outputs matches the
   baseline model's valid diversity.

### Operationalizing Parity

Token-level perplexity is an **invalid metric** for latent EBMs
because EBMs intentionally warp probability mass to enforce
constraints. ARM-EBT parity operationalizes as:

**Sample-quality exact-match on combinatorial generative tasks**
— e.g., Pass@1 on formal Z3-verified program synthesis or
mathematical invariant generation, proving the EBM natively samples
valid trajectories without cascade rejection costs.

### Smallest Valid Model Size

The claim must be demonstrated at the **0.8B parameter tier
(Qwen3.5-0.8B)**. Achieving non-degenerate thermodynamic sampling
on an open 0.8B model adheres to decentralization constraints and
isolates the architectural validation from the confounding variable
of massive LLM memorization. Scaling to the SOTA 35B tier before
fixing the theoretical mechanisms burns compute unscientifically.

## 6. Risk Register — top 5 silent-failure modes

If the revised Phase-3 prototype proceeds, these are the top 5 ways
it could silently fail empirically:

### Risk 1: Dimensionality Guillotine Collapse (Finding #1 manifests)

- **Failure:** The discrete latent space lacks the continuous
  capacity to reconstruct the generative state, forcing the DBAE
  decoder to map valid E=0 states into semantically garbled text
  that technically passes strict logic checks but is gibberish.
- **Diagnostic:** `LatentReconstructionMSE` (plotting EBM
  constraint satisfaction against autoencoder reconstruction
  entropy).

### Risk 2: Decoder Spurious Corner Exploitation (Finding #4 manifests)

- **Failure:** The smooth decoder learns to map specifically
  hallucinated tokens into valid zero-energy latent states, allowing
  adversarial prompts to bypass the Ising penalties completely.
- **Diagnostic:** `ContinuousBoundaryAdversaryProbe` (using PGD
  directly on the continuous decoder inputs to hunt for unpenalized
  semantic shifts).

### Risk 3: Hardware Thermodynamic Drift

- **Failure:** Physical noise on the Extropic Z1 analog silicon
  drifts from the theoretical Boltzmann expectation used in the SDK
  simulator, breaking detailed balance silently during production
  inference.
- **Diagnostic:** `SiliconThermodynamicDriftMonitor` (A/B testing
  the API hardware trajectory moments against exact CPU mathematical
  probability mass functions).

### Risk 4: Sub-Space Mode Collapse (Tautology Exploit)

- **Failure:** The generator learns to satisfy all k=5 orthogonal
  verifiers by exploiting a degenerate intersection, reliably
  generating empty code blocks or logical tautologies (e.g.,
  "True=True").
- **Diagnostic:** `GenerativeDiversityEstimator` (measuring textual
  n-gram variance on constrained output against the unconstrained
  frontier model).

### Risk 5: Contrastive Hard-Negative Washout

- **Failure:** The negative samples generated for the CD corpus are
  too lexically malformed, causing the EBM to learn a massive, flat
  energy basin that rejects raw noise but fails to penalize subtle,
  realistic logical errors.
- **Diagnostic:** `OODEnergyMarginAnalyzer` (measuring the ΔE scalar
  specifically between ground-truth valid samples and single-
  character logic inversion errors).

---

## Cross-reference correction (added 2026-05-01)

**Round 1's specific prescription on k was REFUTED by the
k-ceiling Round 2 response** (`and-composition-k-ceiling-deep-think-results.md`).

| Round 1 (this document) said | k-ceiling Round 2 corrected to |
|---|---|
| k_max = 5-7 (heuristic, D_int ≈ 5 via pigeonhole) | **k_max ≈ 7-8** via Welch/Rankin bound derivation |
| k=5 r=0.3 is mathematically superior to k=15 r=0.55 | **k=10 r=0.4 is OPTIMAL** per exact Gaussian copula tail integral |
| Train Phase-3 on k=5 once Phase-1d ships | Train Phase-3 on **k ≈ 7-8 with strict mechanism orthogonality**; k=10 r=0.4 sweet spot if achievable |

**Welch bound math (Round 2):** k* ≤ ⌊(1−r_max)/(α²−r_max)⌋. For
α²=0.66, r_max=0.5: k* ≤ 3.125 within the homogeneous text-probe
cluster. To exceed this requires crossing the "mechanism gap" to
formal-verification (Z3-AST), runtime-execution (gVisor), or
combinatorial-encoding (Ising/Potts) families.

**Joint evasion probability (Round 2 exact integral):**
- k=5 r=0.30: 4.12×10⁻⁶
- **k=10 r=0.40: 1.20×10⁻⁶ (OPTIMAL)**
- k=15 r=0.55: 8.36×10⁻⁶

**Implication for the Phase-3 Block-and-Resolve workflow (Round 1's
P5):** Phase-1d's exit gate must be empirical k_max ≤ 8 with
strict pairwise r ≤ 0.45 across mechanism-gap-spanning probes —
NOT "k=5 with r<0.3" as Round 1 specified. exp1104 is the gate
experiment, defined in the k-ceiling Round 2 results document.

**The Round 2 adversarial follow-up to this document
(`phase3-empirical-grounded-deep-think-round2-prompt.md`) remains
valid and pending** — its Q3 (0.8B vs SOTA tier reconciliation),
Q4 (parallel-track vs sequential Block-and-Resolve), and Q5 (silent
failure modes) cover concerns the k-ceiling Round 2 did not
address.
