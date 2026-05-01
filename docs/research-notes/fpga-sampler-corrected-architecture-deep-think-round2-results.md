# Deep Think Round 2 Results — FPGA Sampler Adversarial Follow-up

**Status:** Returned 2026-05-01.
**Paired with:** `fpga-sampler-corrected-architecture-deep-think-round2-prompt.md`
**Headline:** **REFUTES** Round 3's ~3,200× speedup claim AND the
"pivot production to Extropic Z1" recommendation. **MODIFIES** the
2-milestone shipping plan to a 1-milestone χ≤4 Fast-Path fallback.
**CONFIRMS** chromatic Glauber preserves detailed balance in theory.
**Confidence:** HIGH on the math + Z1 vendor-claim skepticism.

---

## 1. Executive Summary

We **CONFIRM P1** (Chromatic Glauber preserves detailed balance in
theory, High Confidence), but note it is highly vulnerable to
hardware-level quantization and PRNG correlations in practice.

We **REFUTE P2** (~3,200× speedup, High Confidence): Round 1
hallucinated away sequential pipeline stalls, and the empirical
chromatic number (χ) of Carnot matrices is fundamentally
unmeasured.

We **MODIFY P3** (2-milestone shipping, Medium Confidence) to
propose a 1-milestone χ ≤ 4 "Fast-Path" fallback to guarantee
shippability and bypass complex Vivado routing constraints.

We **REFUTE P4** (Extropic Z1 integration, High Confidence):
betting production on unverified vendor silicon without independent
exact-Gibbs benchmarks is an existential risk.

We **CONFIRM P5**.

**Per the honesty requirement, all speedup claims must be formally
retracted until χ is empirically measured.**

## 2. Q1 Answer: Empirical chromatic number distribution

### Q1a. Empirical χ distribution

The true χ of deployed Carnot matrices is **unknown**. Any speedup
claim published without it is scientifically baseless.

- **3-SAT:** for c clauses on n variables with c ≈ 10n, average
  variable degree Δ ≈ 60. Sparse random graph theory sets expected
  χ ≈ Δ / (2 ln Δ) ≈ 7.3, but local constraint clustering pushes
  typical χ ∈ **[8, 15]**.
- **Constraint-store:** highly variable; dictated entirely by the
  maximum bag-of-constraints overlap.
- **Numeric SAT:** arithmetic constraints form fully connected
  subgraphs. **An 8-variable arithmetic equation inherently forms
  an 8-clique (K_8), mathematically guaranteeing a global χ ≥ 8.**

### Q1b. Speedup recalibration

Speedup degrades proportionally with k and pipeline stalls. If
high-degree numeric nodes push χ ≥ 15, spatial parallelism
collapses. At N=64 and k=20, the FPGA effectively degrades to a
pseudo-sequential machine clocked at 250 MHz versus a 4 GHz CPU.
**The architectural claim of being "meaningfully faster" for
single-chain latency fails.**

### Q1c. Adversarial bit-packing

Packing 8 spins into a single register to update them simultaneously
**mathematically reintroduces the exact detailed-balance violation
found in exp1094.** The only valid way to update interacting spins
concurrently is a Block-Gibbs sampler. Computing exact joint
conditionals for 2^8 = 256 states per cycle would instantaneously
obliterate the KV260's LUT budget.

### Q1d. Empirical experiment specification

- **Target:** 100 randomly sampled deployed Carnot J matrices
  (stratified by SAT, Numeric, and Constraint-store).
- **Algorithm:** DSatur (Degree of Saturation) via CPU for fast,
  near-optimal coloring.
- **Metric:** Median and 95th percentile χ.
- **Power:** N=100 is statistically sufficient to capture 95th
  percentile outliers with high confidence.

## 3. Q2 Answer: 2-milestone shipping estimate

### Q2a. Vivado synthesis & headroom

The exp1098 build likely uses < 20% of KV260 LUTs/DSPs, so physical
area is abundant. Vivado synthesis takes roughly 2-4 hours.
**The silent risk is Block RAM (BRAM) routing congestion.**
Implementing an arbitrary k-step AXI-Lite state machine requires
dynamically multiplexing variable-length graph read/write masks
across the MAC arrays, risking critical timing-closure failures at
250 MHz.

### Q2b. Speedup inflection vs CPU

**Round 1's speedup math is catastrophically flawed.** A single
color-batch pipeline flush requires 4 cycles. At k=4, a sweep takes
16 cycles = 64 ns. Against an optimized 1 µs CPU baseline, the
latency speedup is **~15.6×, not 3,200×**.

The 13,061× headline from exp1081 conflated fully parallel
(1-cycle) execution with an unoptimized Python CPU baseline
(~52 µs). To cross the >100× speedup inflection point against a
C++ CPU, k must equal 0.6 (impossible), or N must scale to ≥ 1024
so massive parallelism amortizes the fixed pipeline stalls.

### Q2c. Adversarial 1-milestone variant — the SHIPPABLE recommendation

**Sacrifice completeness for shippability.** Ship a bitstream
strictly hardcoded for a χ ≤ 4 fast-path. The CPU pre-processor
runs DSatur; if χ ≤ 4, it dispatches to the FPGA. If χ > 4, it
bypasses the FPGA and executes sequential exact-Gibbs on the CPU.

This bounds RTL complexity, guarantees detailed balance, and lets
you ship a mathematically verified **"Sparse-Constraint Accelerator"
in one milestone.**

## 4. Q3 Answer: Extropic Z1 SDK reality

### Q3a. Z1 SDK reality

As of May 2026, there is **zero independent, peer-reviewed evidence**
that physical Z1 silicon samples exactly from P(x) ∝ exp(-E/T)
across arbitrary frustrated non-planar topologies with KL ≤ 0.05.
Public SDK access generally provides vendor-idealized boolean
sampling from closed-beta cloud APIs, lacking transparent metrics
on minor-embedding overhead and analog precision loss.

### Q3b. Realistic Phase-2 scope

Pursue **Option A** (build a CPU exact-Gibbs "Z1 emulator") internally
to unblock the Extropic API integration for the Carnot software
team. Concurrently pursue **Option B** (defer physical Z1
integration) for the paper. Frame the Z1 as a **"future target
architecture."** We cannot block a rigorous cryptographic verifier
on an unverified vendor timeline.

### Q3c. Vendor claim adversarial base

Analog thermodynamic hardware natively optimizes (finds energy
minima) but **notoriously fails rigorous equilibrium sampling** due
to analog noise floors, local freezing, and calibration drift.
**Round 1's recommendation to pivot production entirely to Z1 is
built on vendor marketing, not empirical science.**

## 5. Q4 Answer: Practical FPGA deviations

### Q4a. Practical FPGA deviations

- **Quantization:** 8-bit fixed-point ΔE truncates to zero in flat
  landscapes (|ΔE| < T/255). Transition probabilities snap to 1 or 0,
  severely breaking ergodicity. **Diagnostic:** Run
  `KLDivergenceEstimator` at extremely low temperatures on
  near-degenerate J matrices to map the truncation limit.
- **PRNG Quality:** LFSRs exhibit linear dependencies. If seeds
  correlate across sequential colors, it pseudo-synchronizes spins
  and creates artificial traps. **Diagnostic:** Extract raw traces
  and compute cross-color spatial autocorrelation ⟨S_i S_j⟩ using
  the Dieharder suite.
- **Color-boundary timing:** Multi-AXI transactions risk Color k+1
  reading stale states before Color k writes commit. **Diagnostic:**
  Use the Vivado Integrated Logic Analyzer (ILA) to trigger and
  trace read/write valid handshakes precisely at color boundaries.

### Q4b. Pipeline stall verification

As explicitly calculated, **Round 1 ignored pipeline physics.**
Updating sequential color batches requires flushing the MAC
pipeline to resolve structural dependencies. 4 colors × 4
cycles/stall = 16 cycles per sweep. **The ~3,200× speedup is a
mathematical artifact of ignoring inter-color stalls and dividing
a slow Python baseline.**

## 6. Q5 Answer: Additional silent failures

### Additional silent failures Round 1 missed

- **CPU Pre-processor Bottleneck (Amdahl's Law):** Greedy/DSatur
  graph coloring scales at O(N²). At N=1024, coloring takes
  milliseconds. **If Carnot's verifier dynamically mutates J
  mid-run, CPU recoloring latency will entirely dwarf the 64 ns
  FPGA execution time, starving the hardware and nullifying the
  speedup.**
- **Color-batch Ordering Effects:** While any valid color schedule
  preserves detailed balance asymptotically, deterministic
  ordering (e.g., Color 1 → 2 → 3 → 4) alters the Markov transition
  matrix. Poor schedules against frustrated cliques can drastically
  lower the spectral gap, **causing exponentially slower mixing
  times.**
- **Chromatic Number Drift:** Dynamic J matrices mean χ is
  non-stationary. **If χ drifts above the hardware's physical
  scheduling limit during training, the system must detect the
  violation, halt the FPGA pipeline, and trigger a CPU fallback.**

## 7. Recommended Phase-2 Milestone .86 Task List

**Do not publish any "Chromatic Glauber speedup" multipliers in the
position paper until these four experiments are completed:**

### Task .86.1 — The Honesty Benchmark
Execute CPU DSatur on 100 held-out Carnot J matrices. Measure
median and 95th percentile χ. Formally recalculate theoretical
FPGA latency using this denominator.

### Task .86.2 — Stall Recalibration
Use a Vivado cycle-accurate ILA trace to empirically measure the
exact AXI-Lite pipeline drain/fill latency between color
transitions. Finalize the true single-sweep physical latency limit.

### Task .86.3 — Amdahl's Law Audit
Benchmark CPU wall-clock DSatur execution vs exact FPGA sweep time.
Calculate the minimum number of samples needed per J matrix to
prevent the CPU pre-processor from acting as the system bottleneck.

### Task .86.4 — Fast-Path Prototyping
Draft the 1-milestone χ ≤ 4 bitstream fallback specification to
guarantee an immediate, mathematically verified shippable POC.

---

## Operator Notes (added 2026-05-01)

**This Round 2 is RIGOROUSLY GROUNDED and DEMOLISHES specific
Round 1 prescriptions:**

| Round 1 (FPGA) said | Round 2 verdict | Confidence |
|---|---|---|
| ~3,200× speedup at k=4 chromatic | **REFUTED — actual ~15.6× vs optimized C++ CPU** | HIGH |
| Pivot production to Extropic Z1 | **REFUTED — Z1 has no independent peer-reviewed exact-Gibbs benchmark** | HIGH |
| 2-milestone shipping plan | **MODIFIED — 1-milestone χ≤4 Fast-Path is the real shippable** | MEDIUM |
| Chromatic preserves detailed balance | **CONFIRMED — but vulnerable to hardware quantization + PRNG quality** | HIGH |
| Bipartite-J restriction is dead | **CONFIRMED** | HIGH |

**THE 13,061× FPGA HEADLINE FROM EXP1081 IS NOW SUSPECT.** Round 2
explicitly notes that exp1081's CPU baseline was unoptimized
Python (~52 µs); against optimized C++ CPU the actual multiplier
is much smaller. **The position paper draft v2 (exp1091) cites this
13,061× number; before publication (deadline 2026-05-15), the
number must be either retracted, recalibrated against optimized
C++, or contextualized.**

**Updated provisional Phase-2 architecture:**

1. **Ship the 1-milestone χ≤4 Fast-Path Sparse-Constraint
   Accelerator** (CPU fallback for χ>4). Round 2's Q2c.
2. **Run the 4 .86 tasks (.86.1-.86.4) BEFORE publishing speedup
   claims.** Round 2's Section 7.
3. **DO NOT bet production on Extropic Z1 yet.** Build a CPU
   exact-Gibbs Z1 emulator for software-team API plumbing; defer
   physical hardware integration.
4. **Position paper v3 framing:** "FPGA POC tier with empirically-
   measured χ-distribution-dependent speedup; production target
   architecture (Z1, photonic) deferred to future work pending
   independent silicon benchmarking."

**Cross-reference correction for Round 1 (FPGA Round 3 results):**
the operator notes section of `fpga-sampler-corrected-architecture-deep-think-results.md`
should be updated to reflect Round 2's refutation of the ~3,200×
and Z1 prescriptions. (This file does that update.)

**Cross-reference correction for Round 1 (Phase-3 meta):** the
"Q2 sampler architecture" answer in
`phase3-empirical-grounded-deep-think-results.md` ranked Z1 pivot
as Rank 1. Round 2 (FPGA) refutes this. The meta document's
sampler ranking should be amended: Sequential single-spin Glauber
(Round 1's Rank 2) is now the realistic engineering rescue,
and Z1 becomes "future architecture pending independent
benchmarking" rather than "1-milestone integration."

**Pending: the Phase-3 meta Round 2 adversarial follow-up
(`phase3-empirical-grounded-deep-think-round2-prompt.md`) remains
unanswered.** Its Q2 (training schedule), Q3 (0.8B vs SOTA), Q4
(parallel-track Block-and-Resolve), Q5 (silent failures) are still
open.
