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

## Status
- 2026-06-05: thesis seeded; harness build started (AR + headroom gate first).
