# Deep Think Results — FPGA Sampler Corrected Architecture (Post-Finding-#2)

**Status:** Returned 2026-05-01.
**Paired with:** `fpga-sampler-corrected-architecture-deep-think-prompt.md`
**Headline:** Recommends **Option 8 (Z1 pivot) + ST3 (Chromatic
Glauber hybrid)** for the FPGA POC tier. ~3,200× speedup at k=4
chromatic colors; gracefully degrades on dense J.
**Caveat:** Treat the ~3,200× speedup claim as PROVISIONAL — depends
on the unmeasured chromatic number of Carnot's actual J matrices.
Round 4 adversarial follow-up recommended before Phase-2 commits.

---

## 1. Executive Summary

The empirical falsification in experiment 1094 forces a candid
recalibration: the 13,061× CPU speedup headline cannot be maintained
for arbitrary, dense J matrices if detailed balance is required.
Synchronous parallel updates on interacting spins inherently induce
oscillatory artifacts and distribution collapse in frustrated
systems, which is exactly why the KL divergence hit 3.07 nats.

The recommended path forward is a combination of **Option 8
(Redirect to Extropic Z1)** for production, paired with the **ST3
Hybrid (Chromatic/Block-Sequential Glauber)** to gracefully close
out the FPGA POC tier. A Chromatic sampler uses graph coloring to
update non-interacting spins in parallel. It strictly preserves
detailed balance for arbitrary J and will retain an O(N/k) speedup
(yielding a highly defensible ~1,000× to ~3,000× speedup depending
on graph density). This requires roughly 2 milestones to ship,
permanently fixes the FPGA correctness caveat, and cleanly tees up
the thermodynamic principles required for the Z1 transition.

## 2. Sampler Architectures Ranked

Note: the 13,061× baseline represents O(N) parallel unrolling at
N=64.

| Rank | Architecture | (a) Correctness | (b) Speedup | (c) J Restriction / Constraint Fit | (d) Gate / Architecture Overhead | (e) Wall-clock |
|------|--------------|-----------------|-------------|-------------------------------------|----------------------------------|----------------|
| **1** | **Option 8: Abandon FPGA / Z1 Pivot** | Exact (continuous/native) | N/A (hardware-specific) | None — natively suited for DBAE-EBM | N/A (vendor silicon) | 1 milestone (API integration) |
| **2** | **ST3: Chromatic (graph-colored) Hybrid** | **Exact (arbitrary J)** | **~1,000–3,000×** (drops by factor of k colors) | None (handles all 6 families; dense J slows it) | Low — reuse exp1098 + add coloring scheduler | 2 milestones |
| 3 | Sequential Glauber | Exact (arbitrary J) | ~30–100× (13k headline is dead) | None — handles all 6 families + future k=15 | Low — serialize AXI-stream / state machine | 1 milestone |
| 4 | Metropolis-Hastings | Exact (arbitrary J) | ~100× (sequential proposals) | None — handles all 6 families | High — RNG/comparator logic overhead | 3 milestones |
| 5 | Bipartite-block (Red-Black) | Exact (bipartite J ONLY) | ~6,500× (half parallelism) | **Severe** — fails generic Constraint Store, Z3, Numeric | Very Low — mask exp1098 updates | 1 milestone |
| 6 | Tempered Transitions | Exact (arbitrary J) | ~50–100× (spatial layout for replicas) | None — highly robust on frustrated J | Very High — replica exchange + multi-temperature | 4+ milestones |
| 7 | Walker's Alias | Exact immediately (arbitrary J) | Massive but bounded to N ≤ 20 | **Severe** — BRAM ceiling at 2^20 states; fails N=64 | High — BRAM routing overhaul | 2 milestones |
| 8 | HMC on relaxed boolean | **DISQUALIFIED** (Finding #1 compounds) | N/A | Relaxed continuous only | Extreme — gradient logic on FPGA | 5+ milestones |
| 9 | Heat-bath continuous | Continuous time only | N/A | Asynchronous mismatch | Extreme — clashes with synchronous KV260 AXI clock | 4+ milestones |

## 3. Tactical Answers (ST1 - ST4)

### ST1. Retrofitting exp1098 to Sequential Glauber

The bottleneck is **NOT** the bipartite clock domain; it is the
**data dependency latency** in the single-spin update logic. In a
sequential update, spin i+1 cannot accurately calculate its ΔE
until spin i has completed its read-modify-write cycle to the
BRAM/register constraint store. If your FPGA clock is 250 MHz, a
4-cycle pipeline stall for dependency resolution means each spin
takes 16ns. For N=64, one sweep takes ~1 µs. A highly optimized
C++ CPU implementation easily achieves sub-microsecond sweeps for
small N. **Preserving a ≥100× speedup using purely sequential
updates on a 250 MHz FPGA vs a 4 GHz CPU is exceptionally
difficult.**

### ST2. Bipartite nature of constraint families

Your assessment is spot-on. Bipartite topologies (where nodes can
be partitioned into two sets with edges only between sets) are
**fundamentally incompatible** with generic SAT and numeric
constraints.

- **Constraint Store & Z3 SAT:** the moment you have a 3-SAT
  clause, or any triangular logical dependency (e.g., A → B,
  B → C, C → ¬A), you create frustrated odd-cycles. Odd-cycles
  cannot be bipartite colored.
- **Verdict:** you cannot restrict the solver to bipartite J
  without crippling the Carnot verifier. **Option 2 is a dead end
  for the general problem class.**

### ST3. The Hybrid: Chromatic (Block-Sequential) Glauber

**This is the silver bullet for the FPGA POC tier.** It is standard
graph coloring. You pre-process arbitrary J into k independent
sets (colors). No two spins of the same color share a non-zero
J_ij coupling.

- **Execution:** update all spins of Color 1 in parallel. Then
  Color 2 in parallel. Then Color k.
- **Correctness:** strict detailed balance is mathematically
  guaranteed because no two interacting spins are updated
  simultaneously.
- **Speedup:** if your graph is sparse enough to be 4-colored
  (k=4), you maintain 1/4 of your maximum parallelism. The
  13,061× headline recalibrates to **~3,200×**. If J is dense
  (all-to-all), k=N, and it gracefully degrades to purely
  sequential.

### ST4. Extropic Z1 path minimum viable milestone

Since Extropic Z1 hardware relies on thermodynamic noise to sample
exp(-E/T) natively, the MVM is software-layer emulation and API
plumbing.

- **Goal:** prove the Carnot architecture can generate J matrices,
  ship them to the Z1 API, and digest the returned samples back
  into the verifier constraint pipeline.
- **Action:** build against Extropic's Python simulator/SDK today.
  The milestone should demonstrate the system resolving a 64-spin
  constraint problem using the Z1 SDK mocked endpoint. If the SDK
  is mature enough to ingest arbitrary J and return boolean
  states, the FPGA becomes purely a fallback sovereignty anchor.

## 4. Recommended Phase-2 Milestone Scope

**Objective:** ship "Carnot Exp 1100: Chromatic Glauber POC & Z1
Pivot".

**Scope:**
1. Implement an off-chip (CPU) greedy graph-coloring algorithm to
   assign the N=64 spins into k color batches based on J.
2. Modify the exp1098 bitstream to accept a k-step scheduling
   sequence, executing parallel updates per color block.
3. Finalize the Python wrapper for the Extropic Z1 SDK simulator.

**Acceptance Gate:**
- Empirical pass on the 12-spin frustrated ring:
  KL(P_chromatic || P_gibbs) < 0.05 nats.
- Maintain >1,000× CPU speedup on average k ≤ 8 graph topologies.

**Honest Verdict (if it falls short):** "Hardware acceleration of
exact MCMC sampling on dense graphs is severely bounded by
data-dependency serialization. Our architecture pivot to Extropic
Z1 (continuous thermodynamic sampling) resolves this, while our
FPGA POC serves as an edge-tier sequential fallback." Update the
position paper accordingly.

## 5. Risk Register (Top 3 Empirical Failure Modes)

### Risk 1: Vendor / Paradigm Vaporware (Z1 Risk)

Thermodynamic computing at scale is still bleeding-edge. The Z1
silicon might suffer from calibration issues (analog noise not
perfectly mapping to the target T), leading to effective sampling
from exp(-E/T_effective) where T_effective drifts over time or
space.

### Risk 2: Dense J Degeneration (Chromatic Risk)

If the future k=15 DBAE-EBM latent spaces generate fully-connected,
all-to-all J matrices, the chromatic number k equals N. The
Chromatic sampler will silently degrade to pure sequential
execution, evaporating the FPGA speed advantage in production edge
deployments.

### Risk 3: LUT Precision Truncation (Hardware Risk)

Even with mathematically sound chromatic scheduling, if the FPGA's
lookup table (LUT) for calculating exp(-ΔE/T) lacks floating-point
precision at the tails (high energy barriers), the transition
probabilities will skew. This will manifest as a lingering,
irreducible KL divergence of ~0.1–0.5 nats, failing the 0.05 nat
threshold despite topological correctness.

---

## Operator Notes (added 2026-05-01)

This response is provisional. Per the project's documented
Deep-Think prediction pattern (`feedback_carnot_prediction_pattern.md`):
qualitative survival claims tend to be well-calibrated, but
specific prescriptions (especially numeric performance claims) are
systematically wrong. Three Round 3 prescriptions need adversarial
cross-validation:

1. **The ~3,200× speedup at k=4 chromatic colors** assumes the
   chromatic number of Carnot's actual J matrices is ≤ 4. Round 3
   does not measure this empirically — it is an extrapolation. If
   the typical Z3 SAT graph requires k=10–15 colors, the realized
   speedup drops to ~870–1,300×, still substantial but less
   impressive. **Empirical chromatic-number measurement is a
   prerequisite for the speedup claim.**

2. **ST2 says generic Z3/Constraint-Store/Numeric is NOT
   bipartite-colorable** (correct, due to odd cycles). ST3 then
   says Chromatic with k=4 works. This is internally consistent
   ONLY IF k=4 colors is achievable on the actual graphs — which
   ST2's odd-cycle argument doesn't establish. Empirical chromatic
   number distribution must be measured on a representative sample
   of Carnot's deployed J matrices.

3. **The "1 milestone Z1 API integration" estimate** assumes the
   Extropic SDK exists, is documented, and accepts arbitrary J +
   returns boolean states. None of this is verified by Round 3 —
   it is a reasonable extrapolation but requires Extropic vendor
   contact.

A Round 4 adversarial follow-up is recommended before Phase-2
commits to Chromatic Glauber as the corrected architecture. The
core question for Round 4: **what is the empirical chromatic
number distribution of Carnot's actual deployed J matrices, and
does that distribution support the ~1,000–3,000× speedup claim?**
