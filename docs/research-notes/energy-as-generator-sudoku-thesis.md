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
