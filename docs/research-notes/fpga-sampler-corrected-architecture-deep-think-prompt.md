# Deep Think Prompt — FPGA Sampler Corrected Architecture (Post-Finding-#2)

**Status:** Ready to send. Asks Deep Think to rank-order sampler
architectures that preserve detailed balance on the KV260 FPGA
while retaining the 13,061× CPU-vs-FPGA speedup measured in
exp1081, given that the synchronous parallel Glauber sampler is
empirically wrong (KL=3.07 vs threshold 0.05, exp1094).
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-2 hardware-acceleration story rested on
the KV260 FPGA running a synchronous parallel Glauber sampler over
arbitrary symmetric J, claimed to produce samples from the model's
intended Boltzmann distribution `exp(-E(z)/T)` at ~13,000× CPU
speed.

**Today (2026-05-01) experiment 1094 falsified this claim
empirically.** A 12-spin frustrated antiferromagnetic ring
benchmark produced:

- KL(P_parallel_glauber || P_correct_gibbs) = **3.07 nats**
- Acceptance threshold was **0.05 nats**
- Verdict: `fpga_sampler_distribution_mismatch_confirmed`
- Detailed-balance closed-form check verified on 2-spin J;
  parallel-vs-sequential observability instrumented in
  `python/carnot/eval/diagnostics.py:KLDivergenceEstimator`

**Why this matters:** the audit's Finding #2 ("synchronous parallel
Glauber on arbitrary J does not preserve detailed balance") is now
empirical fact, not theoretical concern. The deployed bitstream's
parallel Glauber update produces samples from a distribution ~3
nats away from `exp(-E/T)` — the model is using a sampler that is
not the sampler the architecture assumes. Phase-2 hardware claims
must be reconstructed.

**The user has clarified Phase-2 hardware policy** (2026-04-30
directive):

> "I am less interested in FPGA with the future looking more
> Extropic Z1 or photonic. Option C + D sounds like the most
> realistic to me." [where C = re-scope FPGA to POC tier; D =
> pivot production hardware bet to Extropic Z1 + photonic]

So:
- KV260 FPGA is now POC-tier, not production-load-bearing
- Existing bitstreams (exp1041 first-light, exp1068 smoke,
  exp1081 scale benchmark, exp1098 Potts) preserved as engineering
  proof-points with documented sampler-correctness caveat
- Future production hardware = Extropic Z1 + photonic
- 2x RTX 3090 CUDA + Strix Point ROCm + NPU sovereignty anchor
  remain primary GPU + edge paths

### The question

**Given Phase-2 is now POC-tier on FPGA, rank-order the following
sampler architectures on (a) detailed-balance correctness, (b) FPGA
hardware speedup vs CPU (preserving the 13,061× headline if
possible), (c) problem-class restriction (which J matrices it can
sample correctly), (d) fit with the existing exp1098 Potts-machine
bitstream architecture, (e) wall-clock cost in milestones to ship
a corrected bitstream:**

1. **Sequential single-spin Glauber** — correct on arbitrary J,
   loses parallelism. What's realistic FPGA latency?
2. **Bipartite-block (red-black) parallel Glauber** — correct on
   bipartite J only. Restricts the encoder's expressible J matrix
   space. Empirically compatible with the 6 existing constraint
   families (constraint store, code AST, numeric SAT, etc.) the
   project's verifier suite imposes?
3. **Metropolis-Hastings** — correct on arbitrary J via accept/
   reject. Requires hardware random number generator and
   accept-ratio comparator. FPGA gate-count overhead?
4. **Heat-bath continuous-time chain** — correct in continuous
   time, asynchronous hardware. Compatible with KV260's AXI-Lite
   synchronous register interface, or requires bitstream redesign?
5. **Hamiltonian Monte Carlo (HMC) on relaxed boolean** — train the
   latent space to be smooth/continuous, use HMC. Re-introduces the
   dimensionality concern from audit Finding #1 (boolean latent
   cannot represent continuum). Does that self-cancel or compound?
6. **Tempered transitions / parallel tempering** — multiple
   chains at different temperatures, swap proposals. Correct on
   arbitrary J. FPGA implementation complexity?
7. **Walker's alias-method exact sampler** — for low-dimensional
   z (small d), precompute the Boltzmann distribution directly,
   sample exactly. Memory cost grows as 2^d; only viable for
   small problems. What's the d-ceiling on KV260?
8. **Abandon FPGA Glauber, redirect Phase-2 budget to Extropic Z1
   access** — what is the production-hardware path's risk profile
   (Z1 silicon availability, SDK maturity, vendor relationship)?
   What are the highest-impact Z1-vendor-relationship tasks?

For each of (1)-(8), provide:

- **Detailed-balance correctness status:** does it preserve
  `exp(-E/T)` exactly, in the limit, or only on a restricted J class?
- **Hardware speedup vs CPU:** rough order-of-magnitude estimate;
  does it preserve the 13,061× headline from exp1081 or degrade?
- **FPGA gate-count overhead:** comparable to existing Ising
  bitstream, or substantial redesign?
- **Sampler-class restriction:** can it run on the Carnot project's
  6 existing constraint families (constraint store / Z3 SAT / AST /
  liveness / type / numeric)? On the future DBAE-EBM latent space
  (k=15 verifier joint constraints)?
- **Empirical pass/fail criterion** the prototype must meet (e.g.,
  KL < 0.05 on the same 12-spin frustrated ring exp1094 used)
- **Wall-clock cost in milestones** at the project's current
  ~2-week cadence to ship + bitstream load + smoke-tested artifact

### Decision framework

The recommended sampler should optimize:

1. **Correctness first:** detailed-balance must hold for a
   well-defined J class.
2. **Speedup preservation:** the published headline result is
   13,061× CPU-vs-FPGA at N=64 from exp1081. Maintaining this
   number (or achieving a defensible new number) on a corrected
   sampler is the strongest paper claim.
3. **Problem-class compatibility:** the corrected sampler must
   run on at least the 6 existing constraint families. If
   restricted to bipartite J, the project must demonstrate which
   constraint families admit bipartite J.
4. **Future-proof:** the corrected sampler should map cleanly to
   Extropic Z1 (when SDK ships) and photonic (when accessible)
   without architecture rework. Z1 is thermodynamic-computing-
   based; photonic is ground-state-search-based. Both produce
   correct Boltzmann samples natively. The question is whether
   the FPGA POC tier should anticipate that handoff.

### Specific tactical questions

**ST1.** How fast can the existing exp1098 Potts-machine bitstream
be retrofitted to use sequential single-spin updates while
preserving ≥100× CPU speedup? Is the bottleneck the bipartite
clock domain or the single-spin update logic?

**ST2.** Which of the 6 existing constraint families have J
matrices that are bipartite (or can be re-encoded as bipartite)?
- Constraint store: usually NOT bipartite (constraints span
  multiple variables)
- Z3 SAT: usually NOT bipartite (clauses span 3+ variables)
- AST: bipartite if node-pair only
- Liveness: bipartite if variable-vs-state
- Type: bipartite (variable-vs-type)
- Numeric: usually NOT bipartite

**ST3.** Is there a hybrid — bipartite-block update on the
bipartite-decomposable subgraph + sequential update on the rest —
that preserves correctness with partial parallelism?

**ST4.** For the future Extropic Z1 path: what is the minimum
viable Z1 SDK / vendor relationship the Phase-2 milestone should
target? Z1 is reportedly a thermodynamic-computing ASIC that
samples natively from `exp(-E/T)` with hardware noise. If the SDK
is ready, the FPGA work becomes proof-of-concept signaling rather
than load-bearing infrastructure.

### What I am NOT asking

- I am NOT asking to abandon the FPGA POC tier. exp1041 / exp1068
  / exp1081 / exp1098 stay as engineering proof-points.
- I am NOT asking for new theoretical defense layers. The Phase-3
  through Phase-7 architecture is complete.
- I am NOT asking to revisit the Phase-2 production-pivot
  decision (Z1 + photonic). The question is whether the POC FPGA
  bitstream needs a sampler-correctness redesign and at what cost.

### Output format

1. **Executive summary** — 1 paragraph naming the recommended
   corrected sampler architecture and rough effort estimate
2. **Ranked table** — all 8 architectures evaluated against the
   5 criteria
3. **Tactical answers** — ST1-ST4
4. **Recommended Phase-2 milestone scope** — what does the next
   FPGA experiment ship? What's the acceptance gate? What's the
   honest_verdict if it falls short?
5. **Risk register** — top 3 ways the corrected sampler could
   fail empirically (analogous to Finding #2 itself)

### Honesty requirement

If the answer is "no FPGA sampler architecture preserves the
13,061× speedup while being correct on arbitrary J", say so. The
honest answer "FPGA speedup at N=64 is reduced to ~30× when the
sampler is correct, and the position paper should re-frame the
Phase-2 claim accordingly" is more valuable than maintaining the
original headline. The paper hasn't been finalized yet; the claim
can be recalibrated honestly.
