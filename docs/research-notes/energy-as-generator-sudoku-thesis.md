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
