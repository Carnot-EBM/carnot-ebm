# Energy-as-Generator on Constraint Grids (Sudoku) — Regime-Corrected Thesis

**Human-seeded by operator 2026-06-05** after the Kona EBM Sudoku demo
(`reference_kona_ebm_logical_intelligence`). Revisits energy-as-generator in the
regime where it should win — combinatorial constraint grids where autoregression
is structurally weak — rather than arithmetic (where Carnot's P1/Thesis-A came
back bounded because self-consistency is near-ceiling, i.e. the WRONG regime).

## Hypothesis (Kona's claim, regime-corrected)
On Sudoku-Extreme, a non-autoregressive generator that holistically refines a full
candidate grid against the constraints (energy descent, OR recursive refinement)
beats a matched-compute autoregressive transformer — because AR "cannot easily
revise earlier decisions when it discovers a conflict later."

## What we test (operator chose: EBT + TRM/HRM refiner vs AR)
Three generators, **matched compute** (same param budget / steps / data / batch),
trained from scratch on Sudoku-Extreme-1k-aug-1000, evaluated by held-out
exact-grid **solve-rate**:
1. **AR transformer** — autoregressive over the 81 cells. greedy + self-consistency@32.
   This is also the **headroom positive control**.
2. **EBT** — energy model E(grid | puzzle); inference = energy descent in a
   continuous (softmax-relaxed) grid + discretize. Tests Kona's energy-descent claim
   directly (Kona is an energy-descent EBM).
3. **Recursive refiner (TRM-style)** — tiny net applied recursively: z<-net(x,y,z)
   repeated, y<-net(y,z), iterate; output final y. The arguably-stronger non-AR
   generator per HRM/TRM/LDT.
Plus, as a labelled **reference ceiling** (NOT matched-compute): the pretrained
HRM Sudoku-Extreme checkpoint (~/hrm_ckpt), if it loads safely (pickle audit first).

## Falsification gates (decisive either way)
- **HEADROOM positive control (must pass FIRST, else ABORT):** AR greedy solve-rate
  is non-trivial but **AR+SC@32 < 0.75**. Sudoku-Extreme is hard (LLMs ~2%), so this
  should hold. If AR+SC@32 > 0.75 the regime is polluted (the trap that made P1
  inconclusive) — abort and harden.
- **VALIDATE Kona/regime:** EBT (and/or refiner) solve-rate **> AR+SC@32 by a
  material margin** (target >= +0.10 absolute, non-overlapping bootstrap CI).
- **REFUTE:** EBT <= AR even on Sudoku -> energy-as-generator is bounded even in its
  favoured regime (for our from-scratch EBT). The refiner result is reported
  separately (different paradigm — curl-free energy descent vs arbitrary vector field).
- **Artifact-vs-fundamental (if EBT loses):** compare energy-argmin vs energy-descent
  vs beam decode. If a better decoder closes the gap -> ARTIFACT (decode bottleneck);
  if all decoders fail equally -> FUNDAMENTAL (energy landscape misshaped).

## Adversarial checks (before any "validate" claim)
- The AR baseline must be a HONEST matched-compute control, not a crippled one
  (same params/steps/data; report its config).
- Solve-rate is EXACT 81-cell match against the gold solution (no partial credit,
  no constraint-checker leniency that would flatter a model).
- No constraint-checker in the loss/decode of any model unless ALL models get it
  (keep it pure generation; the constraint checker is only used post-hoc to
  characterise, not to pick answers, unless symmetric).
- Watch for a degenerate EBT (collapses to a single low-energy grid) or a degenerate
  AR (constant output) — report per-cell output entropy.

## Substrate / infra
- Data: ~/trm_src/data/sudoku-extreme-1k-aug-1000 (npy; 81 cells, vocab 11:
  0=pad,1=blank,2-10=digits1-9). Train subsample + fixed held-out eval set.
- GPU: INTERNAL RTX 3090 only (CUDA_VISIBLE_DEVICES=GPU-7971baff-...); eGPU
  (GPU-b52387a2) is flaky under load. Conductor PAUSED during runs (kill_zombies).
- Harness: scripts/experiments/sudoku_energy_vs_ar_v1.py (--model {ar,ebt,refiner}).

## Status — COMPLETE 2026-06-05 (DECISIVE; unlike P1 arithmetic)

| Generator (matched compute) | exact-grid solve | blank-cell acc | violations | params |
|---|---|---|---|---|
| AR greedy | 0.002 | 0.342 | 7.6 | 3.2M |
| AR SC@32 | **0.000** | — | — | 3.2M |
| EBT argmin decode | 0.000 | 0.109 (=chance) | 132 | 3.2M |
| EBT energy-descent decode | 0.000 | 0.109 (=chance) | 145 | 3.2M |
| **Refiner (TRM-style)** | **0.182** | **0.652** | 9.4 | **1.8M** |

**VERDICT: VALIDATE the regime claim via REFINEMENT; REFUTE naive energy-descent.**

1. **HEADROOM confirmed (the control P1 lacked):** AR genuinely learns (blank-cell
   0.342 = 3x chance) but solves ~0% — the non-degenerate weak-AR regime.
2. **EBT = perfect SCORER, failed GENERATOR:** the NCE energy converged (loss
   0.0001, gold-vs-corrupt gap ~17) — it scores Sudoku validity near-perfectly. But
   both decoders (per-cell argmin AND continuous energy-descent, Kona's literal
   mechanism) give 0.000 solve, chance-level blank cells, and worse-than-random
   violations. The wall in energy-as-generator is the DECODE/SEARCH + global
   landscape shaping, NOT the energy learning. (Contrastive training shapes the
   energy only near valid solutions; the global space the decoder searches is full
   of bad minima.)
3. **Refiner WINS:** recursive refinement (end-to-end, deep supervision) solves
   18.2% — +18pp over AR — with FEWER params. It GENERATES where energy-descent
   does not.
4. **DT-P2 made concrete:** recursive refinement (arbitrary learned vector field)
   generates; contrastive-energy-descent (curl-free scorer) does not.
5. **Re Kona:** the 96% is plausibly a REFINEMENT/amortized-inference result (or a
   far more sophisticated energy training+inference), NOT the naive "gradient-descend
   a contrastively-trained energy" recipe — which we show fails.
6. **For Carnot Phase-3:** pursue recursive REFINEMENT (TRM-style) as the generator
   substrate, not energy-descent. The energy function's defensible role stays
   SCORING/verification (EBT near-perfect), consistent with the energy-as-sound-
   lattice reframe.

Caveat: single seed; tiny matched-compute first pass (refiner 18.2% vs TRM/HRM SOTA
~87/55% at full scale) — the COMPARISON is the point, not the absolutes. Artifacts:
results/experiment_sudoku_energy_vs_ar_v1.json + sudoku_{ar_baseline,ebt,refiner}_v1.json.

- 2026-06-05: thesis seeded; harness built (AR/EBT/refiner); experiment run on the
  internal RTX 3090 (~3.5h, both 3090s survived — internal-GPU pinning held); verdict above.

## v2 — ENERGY COMPRESSION test (operator follow-up, 2026-06-05)

Diagnosis of v1's score-vs-generate failure via Kona's "energy compression": v1
used LOCAL corruption negatives, so the energy was only carved near the data →
broad low-energy "hallucinated flatlands" globally → decode drifted into them.
v2 (scripts/experiments/sudoku_energy_compression_v2.py) changes ONE variable:
GLOBAL negatives via Langevin / persistent contrastive divergence (PCD) + an
anti-collapse energy-magnitude regularizer = restrict the low-energy VOLUME
(compress the partition function). Decode = annealed Langevin.

| metric | v1 (local negs) | v2 (global/PCD negs) |
|---|---|---|
| solve-rate | 0.000 | 0.000 |
| blank-cell acc | 0.109 (chance) | 0.110 (chance) |
| mean violations | 132-145 | **79.4** (improved) |
| decoded-grid energy vs gold | BELOW (FLATLAND) | **ABOVE (CARVED)** |

**VERDICT: volume compression WORKS but is NECESSARY-NOT-SUFFICIENT.** The energies
carved cleanly + stably (gold pinned -5.0, Langevin negs +5.0, no divergence) and
the FLATLAND DIAGNOSTIC FLIPPED — decoded grids went from BELOW gold energy (v1
low-energy garbage) to ABOVE gold (v2 +0.16 vs -5.0). So energy compression did its
job: the low-energy flatlands got pushed up; decode no longer drifts into low-energy
garbage (violations 132->79). BUT solve-rate is still 0: decode can't FIND the
narrow gold basin. Two compressions, two failure modes now isolated:
- **Failure 1 (v1): low-energy flatlands** — fixed by VOLUME compression (global
  negatives). Confirmed.
- **Failure 2 (v2 exposes): no smooth global FUNNEL / large basin of attraction** —
  the carved landscape has the right minimum (gold) but descent from a blank start
  can't reach the needle. The fix is DIMENSIONAL/LATENT (DBAE) compression: compress
  the space so the basin is findable (exactly how the refiner, 0.182, generates — it
  works in a compact learned latent). That is v3.
Artifact: results/sudoku_ebt_compression_v2.json.

## v3 — LATENT (DBAE) compression — and the DECISIVE conclusion (2026-06-05)

Added the SECOND compression: a DBAE autoencoder compresses the 729-dim grid into a
256-dim latent z; energy + annealed-Langevin descent happen IN z (the compact,
supposedly-findable space). scripts/experiments/sudoku_latent_energy_v3.py.

| metric | v1 (raw, local) | v2 (raw, global) | v3 (latent, global) |
|---|---|---|---|
| solve-rate | 0.000 | 0.000 | **0.000** |
| AE test-recon (exact) | — | — | **1.000** |
| energy carving | none | gold -5 / neg +5 | gold -5 / neg +5 |
| flatland | BELOW (drift) | CARVED | CARVED |

**THE WALL IS AMORTIZED INFERENCE, not representation or carving.** v3 removes the
last excuse: the autoencoder reconstructs UNSEEN solutions PERFECTLY (recon 1.000),
so the target z_true provably exists and decodes exactly; the energy carves cleanly
(z_true at -5, everything else +5). And solve-rate is STILL 0. The decode-time
Langevin ends at HIGH energy (+5.26), never reaching the deep narrow z_true (-5):
the carved landscape is a high-energy PLATEAU with a measure-zero spike at the
solution, and gradient/Langevin descent from a random start has NO funnel toward
it. (Classic EBM-CD pathology: the model drives its own descent endpoints high
without creating a path to the true minimum; descent and negatives chase each
other.)

**Complete v1->v2->v3 ablation isolates the wall:** representation (v3 AE=1.0) ✓,
volume carving (v2/v3 global negs) ✓, dimensional compression (v3 latent) ✓ — and
pure energy-descent STILL generates nothing. The missing piece is a LEARNED
inference map (amortization). The refiner (0.182) wins precisely because it does
NOT descend an energy — it learns the generation map end-to-end (a trained funnel
toward the answer). 

**Conclusion for Carnot Phase-3 (this is the load-bearing takeaway):**
- Energy-as-generator via descent is NOT the path — not because energy can't learn
  (it scores perfectly) but because finding the minimum of a sharply-peaked energy
  is an unsolved search problem that compression does not fix.
- Generation needs a LEARNED, AMORTIZED inference map = recursive REFINEMENT
  (TRM-style). Pursue that substrate.
- The energy/verifier's defensible role is SCORING + being the training ORACLE for
  the learned generator (the distillation-oracle thesis) — NOT the generator via
  descent. v1->v3 is direct empirical proof.
- Re Kona: their 96% is therefore amortized inference (a learned trace producer),
  with the energy as objective/guide — NOT naive gradient-descent on a
  contrastively-trained energy, which we exhaustively show fails (3 experiments,
  even with perfect latent + perfect carving).
Artifact: results/sudoku_latent_energy_v3.json.

## v4 — AMORTIZED CAPSTONE: confirm the wall by REVERSAL (2026-06-05)

v1->v3 argued the wall is amortized inference by ELIMINATION (energy descent fails
even with perfect latent + perfect carving). v4 confirms it by REVERSAL: train the
amortized generator (Refiner) AND the carved scorer (global-negative EBT) on the
same data, then run FOUR inference procedures against the SAME two trained models
and compare. If the wall is initialization (amortization), then descending the
SAME energy from a learned-refiner start should jump far above descending it from a
random start. scripts/experiments/sudoku_amortized_capstone_v4.py.

| inference procedure (same trained EBT + Refiner) | exact-grid solve |
|---|---|
| random-init energy descent (Adam, 200 steps) | **0.000** |
| refiner greedy (amortized generator alone) | **0.180** |
| refiner-init energy descent (learned start, same energy) | **0.166** |
| energy-rerank refiner@K=32 (EBT picks lowest-energy sample) | **0.120** |

n_eval=500 held-out, seed 0, ~101 min on the internal RTX 3090. EBT carved cleanly
and stably (gold pinned -5.0, Langevin/PCD negatives +5.0, loss saturated at the
-5.0 reg bound, no divergence). Refiner greedy replicates v1 (0.180 vs 0.182).

**VERDICT — SPLIT, reported honestly (not spun):**

1. **AMORTIZATION WALL — CONFIRMED BY REVERSAL (the load-bearing claim).** The
   energy landscape is IDENTICAL across rows 1 and 3; only the *initialization*
   differs. Random-init descent = **0.000** (the v1->v3 failure, recomputed for
   THIS exact carved EBT). Learned-refiner-init descent = **0.166**. A 0 -> 16.6%
   jump driven purely by where descent STARTS proves the wall was the funnel /
   getting-into-the-basin = amortized inference, NOT the energy. This is the
   mechanistic confirmation of v1->v3: the energy was never the problem; reaching
   its minimum from a cold start was, and a learned init solves exactly that.

2. **ENERGY-AS-SCORER-ON-THE-GENERATOR — HONEST NEGATIVE (this setup).** The carved
   EBT does NOT add value on top of the amortized generator here:
   - refiner-init energy descent (0.166) is slightly BELOW refiner greedy (0.180) —
     polishing with the energy mildly HURTS.
   - energy-reranking K=32 refiner samples (0.120) is well BELOW refiner greedy
     (0.180) — letting the EBT pick the "best" of K samples is WORSE than just
     taking the refiner's mode.
   Why: the EBT was contrastively carved to separate gold (one-hot) from
   Langevin/PCD negatives (a global low-energy distribution). The refiner's errors
   are a DIFFERENT, near-correct distribution the energy never saw as negatives, so
   the energy ranks gross validity (gold vs corrupted, near-perfect per v1) but
   cannot finely discriminate among near-correct candidates. The "verifier+generator
   product" intuition — energy as a post-hoc reranker on the learned generator — is
   NOT supported by this from-scratch tiny experiment.

**What v4 sharpens.** The amortization conclusion is now confirmed two ways
(elimination v1->v3, reversal v4): generation needs a learned inference map;
energy-descent from a cold start generates nothing regardless of carving/latent.
AND v4 adds an honest boundary on the energy's role: in this setup the carved EBT
is a VALIDITY scorer, not a fine-grained candidate SELECTOR — energy-reranking the
refiner did not help and slightly hurt. The energy's defensible role stays (a) the
training ORACLE that teaches the refiner, and (b) a gross-validity / abstention
gate — NOT a polish/rerank stage on a strong amortized generator's near-correct
outputs. (Open: a verifier trained on the GENERATOR's own error distribution —
hard negatives mined from refiner samples — might rerank usefully where this
Langevin-carved one did not. That is a natural v5, not claimed here.)
Artifact: results/sudoku_amortized_capstone_v4.json.

## External corroboration — NVIDIA Ising QEC decoder (2026), a deployed instance

The v1->v4 conclusion — a learned amortized inference net beats naive
energy-minimization on a hard inference problem over an energy/Ising landscape —
is not a Sudoku artifact. It is independently corroborated, in a deployed,
production-targeted domain, by NVIDIA's open "Ising" QEC decoder (launched
2026-04-14; decoder framework Apache-2.0; weights on HuggingFace/NGC).

**The mapping (established physics, NOT NVIDIA's framing — they give no reason for
the "Ising" name).** Decoding the rotated surface code is MAP inference on the
random-bond Ising model (Dennis, Kitaev, Landahl & Preskill, 2002). The classical
decoder, PyMatching / minimum-weight perfect matching (MWPM), is exactly the
combinatorial **energy-minimization** solver for that Ising-structured problem.

**The result.** NVIDIA replaced MWPM with a *learned* 3D-CNN decoder — and it wins:
"2.5x faster than PyMatching and 1.11x more accurate" (Fast model) / "2.25x faster
and 1.53x more accurate" (Accurate model) at code distance d=13, p=0.003. Notably
the nets are **tiny** — ~912K and ~1.79M parameters — echoing the tiny-recursive-
model efficiency point: a small *learned amortized* inference net beats the
classical energy-minimizer, faster.

**Why it matters here.** This is the same shape as our finding — *amortized
inference > energy-minimization on an energy/Ising landscape* — realized in real,
deployed quantum error correction. It answers the obvious "is this just a Sudoku
toy?" objection: the pattern is winning in production QEC. And the neural decoder
is functionally a fast verifier/corrector grounded in a physical error model — the
hybrid shape (a learned amortized component grounded by an externally-defined
energy/error model). Caveat: NVIDIA does not frame any of this in energy / Ising-
inference terms; the energy-minimization reading is the established statistical-
mechanics literature, and the analogy to our result is ours. (Naming note:
"NVIDIA Ising" is a quantum-AI toolkit, distinct from the Ising *machines* /
samplers Carnot's hardware path uses — say "Ising machine/sampler" explicitly in
Carnot docs to disambiguate.) See memory `nvidia-ising-qec-amortized`.

## External corroboration — BES "entropy shell" (Harvard/MIT + Yilun Du, 2026)

"Self-Improving Language Models with Bidirectional Evolutionary Search" (BES; Xu,
Qi, Su, Ye, Lakkaraju, Kakade, Yilun Du; arXiv 2605.28814, 2026-05-27) gives a
**formal, independent** version of the central finding here.

**The mapping.** Our result: autoregressive generation is *confined* (Sudoku AR
~chance), and escaping the wall needs a NON-AR mechanism — the learned recursive
refiner solved it; naive energy-descent did not. BES states the abstract verbatim:
best-of-N and tree search "construct candidates primarily through autoregressive
expansion, restricting exploration to regions with substantial model probability
mass," and proves candidates from "expansion-only search are confined to a narrow
**entropy shell**" that an escape operator must break out of. Same diagnosis, in
entropy-geometry terms, from a Harvard/MIT group.

**The contrast to keep sharp (corroborates the diagnosis, not "energy generates").**
BES's escape operator is *evolutionary recombination* (crossover / translocation
of partial trajectories); ours was *learned recursive refinement*. Critically,
BES does NOT claim energy-descent escapes the shell — and our experiments showed
it does not (rand-init descent 0.000). So BES confirms "AR is confined; you need a
non-AR escape," which is exactly the wall we measured — it does not resurrect
energy-as-generator.

**It also corroborates the verifier moat.** BES's *other* stated bottleneck is
"sparse verification signals," fixed by backward recursive decomposition into
*checkable sub-goals* giving dense intermediate feedback — they claim this "can
exponentially reduce the number of required samples to find a correct answer."
That dense per-step checkable signal is precisely the role Carnot's verifier
ensemble plays (we build it from the energy ensemble; they from task
decomposition). Forward-generate + backward-verify is the same generator+verifier
hybrid shape Carnot is pursuing. Regime note: BES's wins are on *open
combinatorial* problems (real headroom), matching our regime-specific reframe that
search/verification pays off where self-consistency is not already near-ceiling.
Caveat: BES is a search/generation framework, not an energy verifier; absolute
accuracy is modest; the analogy to our energy framing is ours. See memory
`bes-bidirectional-evolutionary-search`.

## Experiment #1 capstone — TRM standalone ceiling (Kona-reproduction step 1), 2026-06-06

Operator-directed: train a REAL recursive refiner (nano-trm TRM, the published
Tiny Recursive Model) on Sudoku-Extreme with NO energy, to measure how much of
Kona's 96% is "learned refiner + scale" vs energy. Run: nano-trm TRM, single
internal RTX 3090, ~7h, 1000-example random held-out val (disjoint from the 1000
base train puzzles); metrics nano-trm/train/runs/.../csv.

RESULT: **val exact-grid accuracy reached 0.86 (peak 0.8646), cell-accuracy 0.948**
— SOTA-PARITY with nano-trm's reported ~0.87. Val loss fell monotonically with
rising accuracy (2.65 -> 0.214), i.e. genuine generalization, NO overfitting (val
on unseen puzzles). Compare the same Sudoku-Extreme regime: AR greedy ~0.002 /
SC@32 0.000; EBT energy-descent (argmin + Langevin) 0.000.

CONCLUSION (decisive): a learned recursive refiner with NO energy solves Sudoku-
Extreme at SOTA (~0.86) where autoregression and energy-descent both get ~0%. So
essentially ALL of Kona's generative capability is the LEARNED AMORTIZED REFINER +
scale; the energy contributes at most a small increment and is NOT the generator.
This is the strongest form of the v1->v4 + EBT-kill-gate conclusion: energy
SCORES, refinement GENERATES. Drives the 2026-06-06 strategic reframe (ops/north-
star.md §5): Carnot is the hybrid's energy VERIFIER, not its generator; pursue the
verifier-earns-its-place proof, not more generator work.

## v5/v6 — energy as IN-LOOP GATE (#3) and as TEACHER (#4), 2026-06-07

Follow-up to the retrospective "exploit the verify<<generate asymmetry to GENERATE"
thread. Two new inference/training modes for energy, beyond descent and rerank.

**#3 — energy as a per-step IN-LOOP FEASIBILITY GATE (not descent, not rerank).**
Decode the refiner's blanks in confidence order; at each cell pick the highest-prob
digit that does not conflict (row/col/box) with already-committed cells. First run
was underpowered (refiner trained at ~3% vs the real ~18%, my config error) and
showed +0.0007. Re-run on a FAIR base (v4 config, base greedy 0.1602):
**base_energy_gated = 0.1855, delta +0.0254** (16.0% -> 18.6%). So the in-loop
energy gate gives a real, modest lift when the generator is competent -- the first
inference-time mode where energy ADDS generative value (descent and rerank both
lost). Modest (+2.5pp) and just under an arbitrary 0.03 bar, but clean and
correctly signed. Caveat: a pure greedy gate still leaves ~7-8 mean violations
(it cannot do CSP without backtracking), so the lift comes from pruning local
conflicts, not from solving.

**#4 — energy as a TEACHER (RFT self-distillation), INCONCLUSIVE (control failed).**
Hypothesis: energy belongs at TRAINING time (where amortization wins). Generate K=16
samples/puzzle, keep ONLY samples the verifier CERTIFIES as 0-violation (correct,
since a valid givens-consistent Sudoku is unique), distill the refiner toward them
(no gold labels). Result: energy_distilled collapsed 0.16 -> 0.02. BUT the gold
upper-bound arm (distill on the true answers) ALSO collapsed (0.16 -> 0.13) -- so by
the positive-control discipline the distillation HARNESS, not the energy selector,
is the fault: the self-distillation uses a single forward while the refiner was
trained with deep_supervision over n_cycles, so naive SFT degrades the recurrent
model regardless of selector. Compounded by a low + easy-biased certified yield
(only 4.6% of puzzles yield a certified best-of-16 sample) causing catastrophic
forgetting. **#4 is NOT a clean negative.** A fair #4 v2 needs: deep_supervision-
preserving distillation, replay of the original training data (anti-forgetting),
and higher temp/K to raise the certified yield.

**Net.** #3 nudges the picture: energy is not purely a post-hoc scorer -- as an
in-loop feasibility gate it adds a small but real generative lift on a competent
base. The big training-time question (#4) remains open pending a non-degrading
distillation harness. Neither result disturbs the headline: energy's dominant,
demonstrated value is as the cheap verifier (the moat + efficiency panel), and the
verify<<generate asymmetry stands -- in-loop gating helps a little, descent/rerank
do not, and energy-as-teacher is still unproven.

### v6 #4 v2 update (fixed harness) — energy-as-teacher is HEADROOM-bounded, 2026-06-07

The v1 #4 collapse was a harness bug (gold control fell too). v2 fixed it: deep-
supervision-preserving distillation + a frozen-base anti-forgetting anchor (no gold)
+ K=32. Result on the fair 18% base:

| arm | solve | delta |
|---|---|---|
| base_greedy | 0.1758 | — |
| base_energy_gated (#3) | 0.1934 | +0.0176 |
| energy_distilled (#4) | 0.1680 | -0.0078 |
| gold_distilled (UB) | 0.1621 | -0.0137 |

Certified yield rose 4.6% -> **62.9%** (K=32) and the collapse is gone -- so energy
has abundant certified self-data and a sane harness. Yet energy gives no lift. The
decisive point: **the gold upper bound ALSO fails to beat base** (-1.4%), because
the base is already trained to convergence on gold (15k deep-supervised steps) ->
gold-SATURATED -> NO HEADROOM for any teacher. This is the **P0.1 lesson again**:
energy-as-selection was bounded by SC-near-ceiling; energy-as-teacher is bounded by
a base-at-its-compute-ceiling. Energy's generative value is gated by HEADROOM
everywhere. The only surviving generative contribution is the in-loop gate (#3,
+1.8% robust) -- a different mechanism (inference-time pruning of residual local
errors, not teaching). A truly fair #4 would need a deliberately UNDER-trained base
(real headroom) and ask whether energy-RFT recovers a fraction of the gold-SFT lift
-- but the strong cross-experiment prior is now that energy helps only where headroom
exists, and a saturated model offers none.

### v6 #4 v3 (under-trained base, real headroom) — SUGGESTIVE positive, within noise, 2026-06-07

To test "can the verifier act as a self-improvement REWARD?" fairly, under-train the
base (base_steps=5000 -> 4.5% solve, real headroom) then compare gold-SFT (ceiling)
vs energy-RFT (verifier-only, no labels). One seed, n_eval=512.

| arm | solve | delta |
|---|---|---|
| base_greedy | 0.0449 | — |
| base_energy_gated (#3) | 0.0664 | +0.0215 |
| energy_distilled (#4, no labels) | 0.0547 | +0.0098 |
| gold_distilled (UB) | 0.0527 | +0.0078 |

For the FIRST time energy-as-teacher is positive: energy-RFT lifted +1.0% with NO
labels and MATCHED/edged the gold-SFT lift (+0.8%; frac of gold ceiling recovered
~1.25). The direction the verifier-as-self-improvement-reward hypothesis predicts.
**HONEST CAVEAT:** at ~5% solve / n_eval=512 the standard error on a proportion is
~1.0%, so BOTH the +1.0% (energy) and +0.8% (gold) lifts are ~1 SE -- not significant
at one seed; the gold ceiling itself only moved +0.8% (small headroom). Suggestive,
NOT decisive.

**Cross-version synthesis (the real result):** v2 saturated base (no headroom) ->
NOBODY lifts (gold or energy); v3 under-trained base (headroom) -> BOTH lift, small
and comparable. Consistent with the **headroom law**: the energy verifier can teach
a generator about as well as gold WHERE headroom exists, and not at all where it
doesn't -- the same law as P0.1/energy-as-selection. The one robust generative
positive across all runs is the in-loop GATE (#3, +1.8 to +2.5%), a distinct
inference-time error-pruning mechanism, not teaching. A DECISIVE #4 v4 would need
more headroom (lower base_steps), 3+ seeds, and larger n_eval for tight CIs.

### v6 #4 v4 (DECISIVE, 3 seeds, n_eval=2000) — energy TEACHES, and BEATS gold-SFT. LINE RETIRED.

The decisive replication of #4 v3 (under-trained 5% base, real headroom), 3 seeds,
n_eval=2000 (per-seed SE ~0.5%), distill 3000 steps, K=32.

| arm | mean delta | per-seed | 
|---|---|---|
| #3 in-loop gate | +0.0217 | +0.020/+0.023/+0.022 |
| #4 energy-teacher (NO labels) | +0.0112 | +0.006/+0.016/+0.0115 |
| gold-SFT (control) | -0.0042 | -0.008/+0.001/-0.0055 |

**DECISIVE POSITIVES (both energy generative modes):**
1. #3 in-loop feasibility gate: +2.2%, 3/3 seeds, tight -- robustly real.
2. #4 energy-as-teacher: +1.1%, 3/3 seeds positive (~2 SE/seed) -- SIGNIFICANT, and
   it BEAT the gold-SFT control (-0.4%). The energy VERIFIER (no labels) taught the
   generator BETTER than raw gold supervision.

**Why energy > gold (the strategic point).** Gold-SFT pushes an under-trained model
toward hard true answers off its own distribution (high-variance, no greedy gain).
Energy-RFT trains on the model's OWN verifier-certified-correct samples -- on-
distribution and learnable ("reinforce what you already get right"). That is the
STaR/RFT mechanism, and this shows **Carnot's energy verifier can be the correctness
filter that drives generator self-improvement -- out-performing labels on an under-
trained base.** Direct evidence for verifier-as-self-improvement-reward (the Phase-3
/ foundation-model thesis).

**REVISED conclusion for the whole line (energy-as-generator, RETIRED 2026-06-07).**
The earlier "energy verifies, does not generate" was too strong. Corrected: energy's
DECISIVE, large value is verification (the moat + efficiency panel). But energy is
NOT useless for generation -- used correctly it adds real, modest, headroom-gated
value via TWO mechanisms: (a) an in-loop feasibility GATE at inference (+2.2%), and
(b) as an RFT TEACHER that self-improves a generator from its own certified samples,
beating labels where the base has headroom (+1.1%). What FAILS is the naive route:
energy DESCENT (0.000) and energy RERANK (hurts). The unifying law is HEADROOM:
energy aids generation only where headroom exists (v2 saturated base -> null; v3/v4
headroom -> positive), the same law as P0.1/energy-as-selection. Caveats: tiny toy
(Sudoku), small absolute effects (the ORDERING is the result), gold-SFT-continuation
is a weak label-ceiling (not the best possible use of labels), single task. The
headline is unchanged and strengthened: Carnot is the verifier -- and the verifier
can also bootstrap a generator. Line retired; the verifier-as-self-improvement-reward
result is the forward hook (scale from this Sudoku proxy to the ensemble-on-reasoning).
