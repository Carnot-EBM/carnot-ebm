# Phase-3 Empirical-Readiness Adversarial Audit — Deep Think Prompt

**Status:** Drafted 2026-05-23, ready to paste when operator chooses
to spend the Deep Think budget.

**Why this audit now.** The 2026-04-30 architecture blind-spot audit
(`phase3-architecture-blindspot-audit-prompt.md`) produced 5 FATAL
findings against a *pre-prototype* Phase-3 design — including the
famous unprompted finding #2 (synchronous Glauber limit-cycle
collapse) that nothing in the eight enumerated attack surfaces would
have caught. Three weeks of Phase-2 hardware work and Phase-4 active-
inference validation later, **we now have empirical anchors that did
not exist in April**. The right successor question is no longer "does
the design survive theoretical scrutiny?" — that audit produced its
five fatal findings and the architecture has been adjusted. The
question now is "**does the post-pivot architecture survive the
measurements we just made?**" — a different question with a different
set of attack surfaces.

**Why this is high-value-per-token.** Same methodology that produced
the 2026-04-30 catch (theoretical reasoning + adversarial framing +
"anything outside these N categories" slot). Each prior Phase-3 Deep
Think round has either redirected the design materially or surfaced a
fatal flaw before publication. Wall-clock cost ~10–15 minutes of
operator-side prompt-shepherding. Counterfactual cost of skipping:
arXiv submission with a deployment-time pathology nobody saw.

---

## Prompt to paste

You are a hostile adversarial reviewer at NeurIPS / ICML / ICLR
trying to **torch** Carnot's Phase-3 architectural decision at
submission time. You have read the project's paper draft, the
architecture documents, and the empirical artifacts that anchor
its claims. Your goal is to find every failure mode the designers
haven't anticipated **given the measurements that now exist**. Be
ruthless. Do not soft-pedal.

This is the THIRD adversarial round of this kind. The 2026-04-30
round (against a pre-prototype design) produced 5 FATAL findings,
including one — synchronous Glauber limit-cycle collapse on
bipartite-but-frustrated graphs — that was outside the audit's eight
enumerated attack surfaces. The architecture has been adjusted to
address those findings (deep-EBM-on-FPGA retired, KV260 now POC tier,
production hardware target shifted to Extropic Z1 + photonic). The
current question is whether the *adjusted* design survives the new
empirical anchors.

## The architecture under audit (post-pivot, with empirical anchors)

### Training-time architecture (unchanged in spirit from the post-pivot DAE-DEBM)

- **Discrete Autoencoder (DAE):**
  `text → encoder (real-valued NN logits) → straight-through
  estimator z = sgn(logits) ∈ {-1, +1}^d → decoder → text`
- **Discrete EBM `E(z)`** trained via exact Gibbs / Glauber sampling
  in `{-1, +1}^d` (matches FPGA hardware exactly — no
  continuous-to-discrete distillation step, by design)
- **AND-composed verifier ensemble** of k=15 base verifiers
  contributing the energy term: `E_total(z) = E_DAE(z) + Σ_i E_v_i(z)`
- **Latent dimension** d ∈ {128, 256} (re-scoped down from
  {256, 512} after the 2026-04-30 dimensionality-guillotine finding)
- **STE gradient regularization, VICReg/Barlow-Twins
  regularization, masked-token reconstruction, denoising-AE
  training, persistent contrastive divergence (PCD)** with replay
  buffer
- **Phase-4 active inference:** the variational free energy of the
  verifier ensemble IS the optimization target, validated by four
  Phase-4 artifacts (exp2550 JEPA fast-path, exp2748 FEP factor
  graph, exp2753 FEP aggregator v2, exp2766 LOO-CV adversarial
  recheck)

### Phase-3 deployment-time architecture

- **GPU deployment (RTX 3090):** run the discrete EBM via exact
  Glauber on `{-1, +1}^d`. Verified live 2026-05-22 via exp2862 SOTA
  runtime cache-offload resolver (`usable_response=true`).
- **KV260 deployment (POC tier):** synchronous parallel Glauber on
  `{-1, +1}^N` with N up to 4096 (currently exercised at N=64).
  Verified 2026-05-22 via exp2898 first board-level latency
  measurement: **24.0 µs per sample on n=64 Ising, 3 seeds × 10k
  samples, bitstream sha256 anchored, p95 within 1.5% of median**.
  Same-basis CPU comparison via exp2912/exp2913 confirms speedup
  claim eligibility.
- **Future production hardware:** Extropic Z1 / TSU / photonic
  Ising machines. Not yet accessible; deferred.

### Empirical anchors (NEW since 2026-04-30 audit)

**Anchor A — KV260 per-sample wall-clock measured.**
exp2898 produced 24.0 µs/sample on n=64 Ising. Per-spin update rate
≈ 0.375 µs. At full board capacity (~4096 spins) extrapolated
latency stays roughly constant in n. CPU crossover point estimated
at n ≈ 240 spins.

**Anchor B — CPU same-basis baseline measured.**
exp2912 measured CPU Gibbs at the same problem instance. exp2913
confirms KV260 speedup vs CPU is claim-eligible on the same-basis
comparison. (Specific factor not stated in this prompt; the
adversary can derive it from the artifacts.)

**Anchor C — Verifier ensemble's actual capability on code corpora.**
exp2910 measured `pass@1 = 0.0750`, `pass@k=8 = 0.1750` on a
bounded-budget SOTA code generation task. The ~7.5% absolute
pass-rate is significantly lower than the verifier ensemble's
~89-91% FoVer AUROC; the gap is on code corpora specifically.

**Anchor D — Cross-corpus matrix v9 produced.**
exp2921 built the v9 cross-corpus matrix with **6 headline-eligible
rows** (up from 2 in v6 a week earlier). The matrix anchors the
verifier ensemble's transfer properties across FoVer, HaluEval,
FEVER, code corpora, and constraint corpora.

**Anchor E — Dual-condition (production vs architecture-only)
AUROC discipline shipped.**
The FoVer headline was repinned from 0.9857 → 0.9131 (production)
with architecture-only at 0.8947 (5-seed dual-condition,
adversarially defensible). Delta of +0.0185 between conditions —
self-learning's contribution to FoVer is small, NOT a confound.

**Anchor F — Five consecutive `paper_ready=true` capstones**
(`.271 → `.272 → `.273 → `.274 → `.275). The conductor's autonomous
discipline machinery (adversarial-verify, dual-condition,
preconditions, substrate-declaration) has converged to a state
where milestone-close artifacts pass internal checks reliably.

### What the 2026-04-30 audit found and how the architecture changed

1. **FATAL #1 Dimensionality & Sparsity Guillotine:** Hessian doesn't
   fit FPGA. Architecture response: latent dimension capped to
   d ∈ {128, 256}; deep-EBM-on-FPGA bet retired; future production
   hardware shifts to Extropic / photonic.
2. **FATAL #2 (unprompted) Synchronous Glauber Limit-Cycle Collapse:**
   the FPGA's bipartite checkerboard schedule violates detailed
   balance on frustrated graphs. Architecture response: the deep-EBM-
   on-FPGA bet was retired anyway, so the synchronous Glauber failure
   mode is now contained to the n=64 POC measurement, not load-bearing
   for paper claims. **But the per-sample 24µs anchor in exp2898 IS
   produced by synchronous Glauber.** Whether that anchor's energies
   correspond to true Boltzmann samples is an open question.
3. **FATAL #3 Hopfield Capacity Mode Collapse:** pairwise Ising
   stores ~0.14·N modes. Architecture response: not load-bearing
   anymore since deep-EBM-on-FPGA retired; but still relevant for
   KV260 POC interpretability.
4. **FATAL #4 Taylor-Induced Spurious Black Holes:** non-convex deep
   EBM Hessian produces hypercube-corner pathology. Architecture
   response: Taylor distillation retired; the DAE-DEBM trains
   directly on discrete states.
5. **FATAL #5 Higher-Order Logic Eradication:** pairwise Ising can't
   represent XOR / parity without hidden spins. Architecture
   response: same as FATAL #3 — not load-bearing for current paper
   claims; verifier ensemble does the higher-order logic at training
   time, FPGA only runs the energy function.

## Your task

Find the ways this architecture fails that we haven't anticipated
**given the new empirical anchors**. Specifically attack the
seams between the measurements we just took and the claims they
support. That's where real systems break.

### Required attack surfaces

1. **The synchronous-Glauber-anchor problem.** exp2898's 24 µs/sample
   number was produced by the same synchronous Glauber sampler that
   the 2026-04-30 audit flagged as violating detailed balance on
   frustrated graphs. The paper's hardware validation section will
   cite the 24 µs number. Find the failure mode where:
   - The 24 µs latency is honest (the FPGA really completes a sweep
     in 24 µs), AND
   - The energies it reports are NOT samples from the target
     Boltzmann distribution, AND
   - Any downstream claim that depends on the energies being
     Boltzmann (e.g., AUROC on energy, free-energy interpretation)
     is therefore quietly wrong.

   Quantify how the audit could distinguish "anchored to wall-clock"
   from "anchored to a physically-meaningful sample." If the
   distinction can't be made on present evidence, that is itself a
   FATAL finding.

2. **CPU comparator hidden mismatch.** exp2912 ran a CPU Gibbs
   baseline on the same Ising problem. exp2913 declared speedup-
   claim eligibility. Find the scenario where:
   - The CPU comparator's update schedule (sequential Gibbs?
     asynchronous?) is detailed-balance-preserving, AND
   - The KV260 update schedule (synchronous parallel Glauber on a
     bipartite checkerboard) is NOT, AND
   - The "speedup" therefore measures CPU-correct-sampler vs
     FPGA-non-equilibrium-blinker — i.e., the comparison is
     apples-to-oranges and the speedup is not what the paper will
     claim it is.

   What evidence would the paper need to defuse this? Is that
   evidence in the current artifacts, or does it need a new
   experiment?

3. **Verifier-ensemble capability gap on code corpora.** exp2910
   measured `pass@1 = 0.0750` on a bounded-budget SOTA code
   generation task. The verifier ensemble's headline FoVer AUROC
   is 0.9131. **The verifier is 90+% on FoVer but the base model
   is 7.5% on code.** Find the failure mode where:
   - The verifier ensemble produces high AUROC on the
     small-fraction-of-correct examples it sees during training,
     AND
   - The base model's actual error rate at production time is so
     high that the verifier sees mostly-error inputs at test time,
     AND
   - The verifier's behavior on the mostly-error regime is
     uncalibrated (FoVer-style training doesn't teach the verifier
     how to handle a corpus where 92.5% of candidates are wrong).

   Quantify: at what pass@1 floor does the verifier ensemble's
   AUROC claim cease to be defensible? Is 7.5% above that floor or
   below it?

4. **AND-composition vs the 6 headline-eligible matrix rows.**
   exp2921 produced cross-corpus matrix v9 with 6 headline-eligible
   rows. The verifier ensemble's energy is the AND-composition of
   k=15 base verifiers. Find the failure mode where:
   - The 6 corpus rows individually pass headline-eligibility, AND
   - The 15-verifier joint null space (Spera Theorem 9.2: detection
     coNP-complete) contains a subspace that's specifically large
     on some new corpus the paper does NOT have a row for, AND
   - The paper's "the verifier ensemble generalizes" claim is
     therefore narrowly true on the 6 measured corpora but
     systematically false on corpora outside the measurement set.

   What corpus would you, the reviewer, demand to see before
   accepting the generalization claim?

5. **Phase-4 free-energy validation vs the Boltzmann-anchoring
   gap.** The four Phase-4 artifacts (exp2550 / 2748 / 2753 / 2766)
   validate the variational free-energy framework — but they were
   validated on continuous-sampler MCMC. The KV260 anchor is
   synchronous Glauber. Find the failure mode where:
   - The Phase-4 free-energy claim is theoretically correct on the
     continuous sampler, AND
   - The same claim is mathematically false on the FPGA sampler
     (because the latter is not detailed-balance-preserving), AND
   - The paper conflates the two by citing the Phase-4 artifacts
     in defense of FPGA-side claims.

   Where in the paper draft does this conflation actually occur, or
   would it occur if the hardware section cites the Phase-4
   section?

6. **The 1.5% p95-vs-median margin.** exp2898's per-sample p95
   latency was within 1.5% of median across 10k samples per seed.
   That's remarkably tight. Reviewers expect this for purely
   compute-bound hardware loops. But Glauber-type samplers
   typically have variable per-sweep cost on inputs that are
   *easy* vs *hard* to thermalize. Find the scenario where:
   - The 1.5% margin is real, AND
   - It's real because the FPGA is doing a fixed number of
     synchronous sweeps regardless of mixing time, AND
   - The reported "final energy" is therefore not a sample from
     the Boltzmann distribution but a snapshot at a fixed schedule.

   How does this interact with finding #1?

7. **The five-paper_ready streak as paper risk.** Five consecutive
   `paper_ready=true` capstones is the project's longest streak.
   Find the reviewer's framing: "the project's autonomous loop
   declared itself paper-ready five times running; what would a
   reviewer infer about the strictness of the paper-ready criterion?"
   Is the streak evidence of convergence to a stable artifact, or
   evidence that the criterion is too loose to be informative?

   Specifically: which of the five `paper_ready=true` flags would
   fail to clear a NeurIPS reviewer's "show me the experiment"
   request, and why?

8. **Hardware sovereignty claim vs operator-only submission.**
   The paper will claim Carnot's verifier ensemble runs on
   commodity FPGA without cloud dependency (hardware-sovereignty
   framing). But the actual KV260 measurement required: a Vivado-
   produced bitstream, a Xilinx Ubuntu image, the conductor's
   SSH-attached workflow, and the operator's specific hardware
   investment. Find the reviewer's framing: "the paper claims
   sovereignty, but the path from 'I pip-install carnot-ebm' to
   '24 µs/sample on my own board' is N concrete steps and M of
   them require non-public artifacts." What is N and M? Does the
   paper acknowledge them?

### Severity classification

Classify each finding as:

- **🚨 FATAL** — claim cannot be made as drafted; requires
  redesign or claim-narrowing. Pre-publication catch is critical.
- **⚠️ DEGRADING** — claim is technically defensible but the paper
  will under-deliver on what reviewers expect. Worth documenting as
  a known limitation in the paper itself.
- **📝 COSMETIC** — minor issue, won't reviewer-torch the paper.

For each FATAL finding, propose:

- The minimum change that could rescue the claim, if any —
  scope-narrowing, additional experiment, retraction, etc.
- Whether the change is purely textual (narrow the paper's claim)
  or requires another empirical measurement (and which one).

For each DEGRADING finding, identify what the paper should say to
defuse reviewer expectations BEFORE the reviewer fires the
objection.

### Output format

A numbered list of findings, each tagged 🚨/⚠️/📝, with a 100-word
explanation per finding. **Bonus points for findings that don't fit
into any of the eight attack surfaces I enumerated** — those would
indicate the audit found something I'd have completely missed. The
2026-04-30 audit's most impactful finding (synchronous Glauber
limit-cycle collapse) was an unprompted finding outside the eight
attack surfaces. Aim for the same.

Be specific. "The hardware claim might be problematic" is useless.
"The 24 µs latency was measured under a fixed-sweep schedule that
doesn't track mixing time, so the final-energy distribution at
seed=42 differs from the target Boltzmann's by [specific concrete
mechanism]" is useful.

Take the regret now. Better to narrow the paper before the reviewer
torches it than to have it torched on arXiv after submission.

### What I am NOT asking you to find

- Failure modes that the 2026-04-30 audit already caught (dimensionality
  guillotine, Hopfield capacity, Taylor black holes, higher-order
  logic eradication, synchronous Glauber limit cycle). The
  architecture has been adjusted; those findings are documented.
  I want NEW failure modes that the new measurements expose.
- Failure modes that depend on Phase-3 production scale (we are
  POC-tier on KV260, NOT claiming Phase-3 production deployment).
  Stay focused on what the current paper draft will actually claim.
- Generic ML-paper objections that don't engage with Carnot's
  specifics. "The model could overfit" without naming what
  specifically would overfit and to what is not useful.
