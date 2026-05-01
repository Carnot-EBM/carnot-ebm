# Deep Think Round 2 Prompt — Adversarial Follow-up to FPGA Sampler Round 1

**Status:** Ready to send. Round 1 returned a strong "Z1 + Chromatic
Glauber" recommendation with specific numerical prescriptions
(~3,200× speedup at k=4 chromatic colors, 2-milestone shipping
estimate, 1-milestone Z1 API). Per the project's documented
Deep-Think prediction pattern (qualitative survival claims
well-calibrated, specific prescriptions systematically wrong),
those numbers need adversarial cross-validation.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session (NOT a continuation
of Round 1). Paste the section labelled `## Prompt to send (verbatim)`.
Reason independently of Round 1.

---

## Prompt to send (verbatim)

### Background

A prior Deep Think round (Round 1, dated 2026-05-01) recommended
the Chromatic/graph-colored Glauber sampler as the FPGA POC tier
rescue path after Finding #2 (synchronous parallel Glauber on
arbitrary symmetric J does not preserve detailed balance) was
empirically confirmed by exp1094 (KL=3.07 vs threshold 0.05).

Round 1's specific prescriptions to validate:

**Prescription P1: Chromatic Glauber preserves detailed balance
on arbitrary J via graph coloring.** Mathematical claim. Believed
correct.

**Prescription P2: Chromatic Glauber on Carnot's J matrices yields
~1,000× to ~3,000× speedup vs CPU.** Specifically ~3,200× at k=4
chromatic colors. **The chromatic number of Carnot's actual J
matrices is not measured.**

**Prescription P3: 2 milestones to ship a corrected bitstream.**
Includes greedy graph-coloring CPU pre-processor + bitstream
modification to accept k-step scheduling sequence.

**Prescription P4: 1 milestone to integrate Extropic Z1 SDK API.**
Assumes Extropic SDK exists, is documented, ingests arbitrary J,
returns boolean states.

**Prescription P5: ST2 verdict — bipartite-colorable J is
incompatible with generic SAT/Numeric/ConstraintStore due to
odd-cycles.** Correct as far as it goes. **But ST2 does not say
what chromatic number IS achievable for typical Carnot J matrices.**

### Established empirical evidence (use as facts)

- **exp1094 (2026-05-01):** parallel Glauber on 12-spin frustrated
  antiferromagnetic ring: KL(P_parallel || P_correct_gibbs) = 3.07
  nats. Detailed-balance failure confirmed.
- **exp1081 (2026-04-30):** 13,061× CPU-vs-FPGA speedup at N=64
  measured with the now-empirically-falsified parallel Glauber.
  This is the "headline" number Round 1 recalibrates to ~3,200×.
- **exp1098 (2026-05-01):** Potts machine bitstream produced
  successful POC artifacts (`potts_sim_and_rtl_complete`) — but
  on a synthetic test rather than on Carnot's verifier-derived J.
- **exp1041 / exp1068:** Ising bitstream first-light + smoke; J
  matrices used were small and synthetic.

### Project-wide design constraints (do not contradict)

- KV260 FPGA is now **POC-tier**, not production-load-bearing.
- Production hardware = Extropic Z1 + photonic.
- 2x RTX 3090 CUDA + Strix Point ROCm + NPU sovereignty anchor
  remain primary GPU + edge paths.
- Decentralization rules: local-first, multi-mirror.

### The Round 2 questions

Round 2 should reason INDEPENDENTLY (do not build on Round 1's
chain). Each prescription is a hypothesis to confirm, refute, or
modify.

#### Q1 (P2 stress-test): empirical chromatic number distribution

**Q1a.** What is the realistic chromatic number distribution of
J matrices produced by the Carnot verifier suite? Specifically:

- For **3-SAT clauses**: each clause introduces a triangle on its
  3 variables. A graph of c clauses on n variables has graph density
  ~3c/n^2. For typical Z3 SAT problems (c ≈ 10n), what is the
  expected chromatic number? Brooks' theorem says χ ≤ Δ where Δ
  is max degree, so χ ≤ ~3·10 = 30. Real number probably much
  smaller, but how small?
- For **constraint-store** with bag-of-constraints (typical case):
  what is the J matrix sparsity pattern? Is there a way to
  empirically measure χ on a held-out corpus of 50-100 deployed
  Carnot constraints?
- For **numeric SAT**: numeric constraints often have wide
  arithmetic dependencies (a + b + c + d = 0 ⇒ all 4 spins
  interact). What is χ on a typical numeric constraint?

**Q1b.** If empirical χ is, e.g., k=8 instead of k=4, the speedup
recalibrates from ~3,200× to ~1,600×. Is the architectural claim
"FPGA still meaningfully faster than CPU" preserved at k=8? At
k=15? At k=20?

**Q1c.** Adversarial: are there encoding strategies that REDUCE
the chromatic number of a given constraint? E.g., "bit-pack 8 spins
into 1 hardware register so they share a single update cycle and
their pairwise interactions don't bloat χ" — is that mathematically
sound or does it violate detailed balance?

**Q1d.** Empirical experiment specification: design a single
1-milestone Carnot experiment that produces an empirical chromatic
number histogram on 50+ of Carnot's deployed J matrices. Specify:
which J matrices to sample, the coloring algorithm (greedy /
DSatur / branch-and-bound), the metric (mean / median / 95th
percentile χ), the sample size for statistical power.

#### Q2 (P3 stress-test): 2-milestone shipping estimate

**Q2a.** The "2 milestones" estimate breaks down as:
- Milestone 1: Off-chip greedy graph-coloring algorithm + CPU-side
  J → color-batch pre-processor.
- Milestone 1.5: Bitstream modification accepting k-step scheduling
  sequence + AXI-Lite register layout for color-batch metadata.
- Milestone 2: Integration test on real J matrices + KL divergence
  validation.

What is the Vivado synthesis time for the modified bitstream? On
the existing exp1098 build, what fraction of LUTs / DSP blocks are
already used? Is there headroom for the color-batch scheduler logic?

**Q2b.** ST1 said "preserving ≥100× speedup using purely sequential
updates on a 250 MHz FPGA vs a 4 GHz CPU is exceptionally
difficult." Does this concern propagate to chromatic? At k=N (dense
J), chromatic IS sequential. At k=4 (sparse J), it's 4-batch
parallel. What's the inflection point — at what k does the
speedup cross below 100×?

**Q2c.** Adversarial: is there a 1-milestone version that
sacrifices completeness for shippability? E.g., implement chromatic
ONLY for J matrices with χ ≤ 4 (use sequential fallback for χ > 4),
and document the speedup as "1,000× on χ ≤ 4 problems, sequential
otherwise." This trades coverage for time-to-paper-claim.

#### Q3 (P4 stress-test): Extropic Z1 SDK plausibility

**Q3a.** As of 2026-05-01, what is publicly known about Extropic
Z1's SDK availability? Specifically:

- Is there a Python simulator publicly accessible (pip-installable
  or via API key)?
- Does the simulator accept arbitrary symmetric J as input?
- Does it return boolean spin states (samples) or probability
  distributions?
- What is the documented latency / cost per sample?
- Are there known accuracy claims (KL divergence vs theoretical
  Gibbs)?

**Q3b.** If the SDK doesn't exist or is closed-beta, what's the
realistic Phase-2 milestone scope under the constraint that we
cannot ship a working Z1 integration in 1 milestone?
- Option A: build a "Z1 emulator" that locally simulates Z1's
  expected behavior using CPU exact Gibbs.
- Option B: defer Z1 entirely; ship Chromatic FPGA + position-paper
  framing of "future hardware = Z1 (when SDK ships)".
- Option C: pursue vendor-relationship task — pay for Z1 silicon
  access if available, develop relationship even if SDK is
  closed-beta.

**Q3c.** Adversarial: Extropic has been publicly advertising Z1
since ~2024 but full silicon details remain proprietary. Is there
an evidence base that the Z1 actually samples from exp(-E/T)
correctly at scale? Has any independent third party benchmarked
it? If not, Round 1's recommendation to "pivot production to Z1"
is built on vendor claims, not empirical data.

#### Q4 (P1 stress-test): chromatic correctness in practice

**Q4a.** P1 (mathematical claim that chromatic preserves detailed
balance) is theoretically sound. But the FPGA implementation
introduces real-world deviations:

- **Quantization:** the energy difference ΔE for the spin update
  is computed in fixed-point. If precision is 8-bit, the smallest
  detectable energy difference is ~T/255. For very flat energy
  landscapes, this floors the transition probability and breaks
  detailed balance.
- **PRNG quality:** the on-chip RNG affects sample quality.
  Linear-feedback shift registers (LFSR) have known biases; if
  the same LFSR seed produces correlated samples across colors,
  detailed balance is preserved-in-theory but violated-in-practice.
- **Color-boundary timing:** between updating Color k and Color
  k+1, must all Color-k writes complete before Color-k+1 reads
  begin. AXI-Lite synchronous behavior should guarantee this, but
  edge cases (interrupt-driven coloring updates, multi-AXI
  transactions) can race.

For each, what is the diagnostic that detects the failure? Are
they observable via the existing KLDivergenceEstimator + new
instruments?

**Q4b.** ST1 mentioned "4-cycle pipeline stall for dependency
resolution" in sequential. In chromatic, the dependency is between
COLORS, not spins. Is there a 4-cycle stall between color batches?
At k=4, that's 16 cycles per sweep step, or ~1/4 of sequential
cost. The recalculation: at 250 MHz, 4 colors × 4 cycles / step
= 16 cycles/step = 64ns/sweep at N=64. CPU: ~1µs/sweep at N=64.
Speedup: ~15×, not 3,200×. Is Round 1's speedup math missing
this overhead?

#### Q5 (Risk register stress-test)

Round 1 produced 3 risks (Z1 vaporware, dense J degeneration, LUT
precision). Are there more? Specifically:

- **Color-batch ordering effects:** does the order of color
  updates affect mixing time? E.g., if the J graph has dense
  cliques of color 1 and sparse clique of color 2, updating color
  1 first vs color 2 first may differ in ergodicity.
- **Chromatic number drift:** as the encoder/decoder learn new J
  matrices during training, χ may drift upward. Does the
  pre-processor re-color on every J change, or only periodically?
- **CPU pre-processor as bottleneck:** if greedy graph-coloring
  takes Θ(N²) time on the CPU and the FPGA finishes a sample in
  ~1µs, the CPU pre-processor BECOMES the bottleneck. At what N
  does this kick in?

### What Round 2 should NOT do

- Do NOT defer to Round 1's reasoning. Reason independently.
- Do NOT propose new theoretical sampler architectures beyond
  the 9 from Round 1.
- Do NOT recommend "vendor-only research" — Round 2 must produce
  experimentally-testable claims.
- Do NOT exceed 1500 words in the response (concise, like Round 1).

### Output format

1. **Executive summary** — 1 paragraph: which Round 1 prescriptions
   you confirm, modify, or refute, with confidence level for each.
2. **Q1 answer** — empirical chromatic number distribution + speedup
   recalibration if χ is higher than 4.
3. **Q2 answer** — 2-milestone shipping estimate's robustness.
4. **Q3 answer** — Extropic Z1 SDK reality check.
5. **Q4 answer** — practical FPGA implementation deviations.
6. **Q5 answer** — additional silent failures.
7. **Recommended Phase-2 milestone .86 task list** — concrete
   experiments that should run BEFORE the position-paper-v2 update
   commits to a Chromatic Glauber claim.

### Honesty requirement

If empirical chromatic number distribution is unknown (and Round 2
cannot derive it from publicly available data), the prescription
"~3,200× speedup" is unfounded. Recommend the empirical
measurement BEFORE the speedup claim ships in any paper. The
honest "we don't know yet, here's the experiment that resolves it"
answer is more valuable than confirming a number.
