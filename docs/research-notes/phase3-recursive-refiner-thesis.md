# Phase-3 Candidate Thesis — Recursive Latent Refiner + Carnot-Verifier-as-Q-Head

**Status:** STAGED CANDIDATE (2026-06-04). Operator-seeded from the TRM/GRAM/PTRM/LDT
technical review. NOT activated. Awaiting operator/Deep-Think review before any milestone
wiring — same gate the EBT (Thesis-A) seed went through.

**Provenance:** operator surfaced a comparative review of small recursive-latent-refinement
reasoners (TRM and three claimed extensions GRAM/PTRM/LDT). Analysis captured in memory
`tiny-recursive-reasoning-models`. This note stages the experiment that analysis implies.

---

## The reframe that motivates this

We closed **energy-as-selector (P0.1)** and **energy-as-generator (Thesis-A / EBT)** as
**BOUNDED** — but both were tested in the **NL-math / arithmetic regime, where self-consistency
is near-ceiling** (no headroom; "can't beat an integral with a mode"). TRM-class models
demonstrate the recursive-latent-refinement paradigm **wins** — but in a *different* regime:
**combinatorial grids with a clean verifiable answer and real headroom** (Sudoku, ARC-AGI,
mazes), where giant autoregressive LLMs are known to be weak.

So our bounded conclusion may be **regime-specific, not paradigm-fatal**. And the *winning
recipe* — tiny recursive latent refiner + internal verifier-halt + noise injection — is
**adjacent to but not identical to** EBT energy-descent. The candidate bet: **EBT energy-
descent was the wrong specific instantiation; a TRM-style recursive refiner is the better
generator substrate, and Carnot's verifier slots in as the internal Q-head / lattice-abstain
component.** Carnot does not have to win the generator race — it owns the verifier these
architectures need internally (banked-product-as-a-component).

## Hypothesis (falsifiable)

A tiny (~7-50M param) recursive latent refiner, with **Carnot's verifier as the internal
halt/Q-head**, beats a matched-compute autoregressive baseline on a **headroom-bearing
combinatorial-grid task** (where AR is demonstrably weak and an oracle/optimal materially
exceeds the AR baseline) — at equal inference compute.

## Why this is NOT a doomed rerun of Thesis-A

| Axis | Thesis-A (EBT, BOUNDED) | This candidate |
|---|---|---|
| Task regime | NL-math / arithmetic (SC near-ceiling, **no headroom**) | combinatorial grid (ARC/Sudoku/maze) with **confirmed oracle > AR headroom** |
| Generator mechanism | global energy-descent (Langevin over E(y₁…y_T)) | TRM-style **recursive latent refinement** (inner/outer loop), noise-injected |
| Verifier role | none internal (energy was both gen + score) | **Carnot verifier as the internal Q-head / halt criterion** (the load-bearing novelty) |
| Decode | greedy energy-argmin / learned decoder (flatlined ~uniform) | iterative refine-in-place of a discrete grid (no off-manifold void) |

The decisive difference is the **regime has measurable headroom** (the P0.1/FALSE_NEGATIVE_RISK
lesson: never test a method where the baseline is already near-optimal) **and** the verifier is
an *internal* component, not a separate scorer.

## Mandatory preconditions (per CLAUDE.md Pre-Launch + Adversarial-Verify discipline)

1. **Positive-control / headroom gate FIRST.** Pick a grid corpus where an oracle (or optimal
   solver) materially exceeds a matched-compute AR baseline (oracle − AR ≥ some margin, AR not
   already near-ceiling). If no headroom, STOP — that's the P0.1 trap.
2. **Matched-COMPUTE, not matched-params** (the P0.1 trap): the refiner runs K refinement
   steps; give AR an equal-FLOP best-of-N / self-consistency budget. A win counts only at equal
   total inference compute.
3. **Verify the claimed papers exist** (GRAM/PTRM/LDT) with real benchmarks before adopting any
   specific number; build on TRM (confirmed real) as the anchor.
4. Tiny-model stability kill-gate (the Thesis-A part-a pattern): can the refiner train stably
   on the rig? If it diverges/NaNs within a bounded budget → bounded at small scale, STOP.

## Kill-gate / acceptance

- **PASS:** refiner-with-Carnot-verifier-Q-head > matched-compute AR by a statistically
  significant margin on a confirmed-headroom grid task, 3+ seeds.
- **BOUNDED (honest negative):** ≤ AR at matched compute even with headroom → recursive-latent-
  refinement + Carnot-verifier does not beat AR at this scale; record and stop (do NOT re-grind).
- **INCONCLUSIVE:** positive control (AR) fails to learn or oracle ≤ AR (no headroom) →
  regime invalid, fix the corpus before claiming anything (the v1/v2 P1 lesson).

## What stays banked regardless

paper_ready TRUE, FoVer 0.9131 frozen, energy-selection bounded, the verifier product. This is
a venture bet with a kill-gate, parallel to (not replacing) the banked deliverable. If it
PASSES, the strategic payoff is "Carnot's verifier is the component inside the winning
small-reasoner class" — the strongest version of the moat per `deep-think-post-bounded-2026-06`.

## Caveats (do not over-claim)

- Narrow fixed-grid combinatorial wins do NOT imply general reasoning (the
  `verifier-domain-bound-math-only` + `mai-thinking-1-hill-climbing` lesson).
- TRM ≠ EBM: this abandons the energy-descent mechanism in favor of recursive refinement; it is
  a NEW substrate, not a continuation of the bounded EBT.
- "Crushes frontier LLMs / fraction of a fraction of cost" is over-claim phrasing
  (`carnot-prediction-error-pattern`); cross-check before citing.

**Next step:** operator/Deep-Think review of this framing → if green-lit, stage as a pre-staged
roadmap milestone with the headroom-gate as task 1 (the EBT seed precedent).

---

## Deep Think reconciliation (2026-06-04) — VERDICT: FIX-FIRST + pivot to distillation-oracle

DT validated the thesis pivot as **mathematically sound** but found a **physical execution
blocker** in the Carnot-specific bet (P3). The thesis is NOT dead — it is REWRITTEN. Net:
**Carnot's role is the offline DISTILLATION ORACLE that trains the continuous Q-head, NOT the
runtime internal Q-head.** Full P1-P5 with confidences in
`phase3-recursive-refiner-deep-think-prompt.md` response (pasted into session 2026-06-04).

**P1 (90%) — regime vs paradigm: BOTH.** Our EBM negative was regime-polluted (SC near-ceiling
= zero headroom), BUT the EBM failure on NL is ALSO paradigm-deep: unconstrained Langevin in
continuous space drifts into off-manifold voids because a global scalar gradient lacks the
causal/discrete inductive bias for valid syntax. TRM *escapes the energy void* (learned forward
passes h_{t+1}=f_θ(h_t), not open-ended gradient descent) — BUT likely hits a NEW
"sequence-compositionality wall" on text: TRM's refine-in-place success leans on the fixed
spatial topology of GRIDS (local changes don't shatter global structure); text needs
variable-length capacity + strict left-to-right causality. **Falsification (cheap):** vanilla
TRM vs matched-compute AR on a strictly-1D variable-length task with AR headroom (algorithmic
sequence correction / multi-step parity). TRM fails there → the paradigm is grid-bound.

**P2 (95%) — TRM ≠ EBT, profoundly (this is the strongest defense of the pivot).** EBT's
inference update field is mathematically restricted to be CONSERVATIVE (curl-free; the gradient
of a scalar energy, symmetric Jacobian). TRM learns a GENERIC discrete-time dynamical system —
an ARBITRARY (asymmetric) vector field — so it can express conditional logic ("if conflict A,
shift B"), asymmetric attractors, and algorithmic limit-cycles that EBT is mathematically
INCAPABLE of expressing. **So our bounded EBT result does NOT cover TRM.** Falsification: fit a
scalar E to a trained TRM's latent updates Δh s.t. Δh≈−∇E(h); the curl-free constraint
guarantees you fail (if you succeed, TRM is secretly energy descent).

**P3 (99%, HIGHEST STAKES) — literal "verifier-as-runtime-Q-head" is a FATAL CATEGORY ERROR.**
External verifiers (SAT/Z3/AST/discrete-energy) require discrete, well-formed, COMPLETE syntax;
TRM's intermediate recursive latents are continuous, half-baked, and decode to gibberish until
the final steps. Decode-in-the-loop → the solver throws syntax errors / returns a flat 0 →
destroys the halting signal and gradient. An end-to-end LEARNED Q-head structurally dominates
(natively reads continuous latents, per-step granular, differentiable). **THE ONLY VIABLE
INTEGRATION PATH IS OFFLINE DISTILLATION:** use the Carnot verifier to score unrolled FINAL
trajectories → train the continuous Q-head via BCE/MSE. **REWRITE THE PITCH:** from "we own the
verifier they need internally" → **"we own the ORACLE required to DISTILL the internal
continuous Q-heads they need."** (Note: this is arguably a BETTER fit for the banked verifier —
used OFFLINE, where its discreteness + ~ms latency don't matter.)

**P4 (85%) — transfer is suspect.** Grids = fixed dimensionality (N×N) + absolute verifiable
completed states. Open-ended reasoning = dynamically expanding length + long-range causality +
abstract semantics. Fixed-latent iterative refinement historically face-plants on open-ended
text (cf. non-autoregressive machine translation). Falsification: TRM on a task whose required
trace must dynamically grow >> input length; if accuracy collapses on unseen lengths (where AR
trivially scratchpads), it cannot reach the foundation-model mission.

### Revised plan (supersedes the task order above where they conflict)

**TASK 0 — the "Latent-Symbol Bridge" kill-criterion ($0 scaling compute, DO THIS FIRST).** Take
an OFF-THE-SHELF TRM that already beats AR on a toy grid. Try to replace its internal Q-head
with an idealized external programmatic verifier: force-decode the intermediate latents to
discrete board states at EVERY step t, query the verifier, measure (a) step-latency overhead and
(b) the false-negative / unparseable rate of intermediate states. If latency balloons and/or
intermediate states crash the verifier (flat signal, no monotonic correlation with step T), the
"verifier slots right in at runtime" thesis is **definitively falsified for $0**, and the
distillation-oracle pivot is confirmed as the only path. Build the distillation pipeline first.

**TASK 0.5 — the P1/P2 cheap falsifications:** (a) vanilla TRM vs matched-AR on a 1D
variable-length headroom task (does TRM escape grids?); (b) energy-fit to TRM's Δh (confirm
TRM ≠ EBT). Both cheap, both decide whether to proceed.

**MANDATORY positive-control / headroom gate (tighter than before, per DT):** curate a regime
where AR greedy ≈ 20% AND **AR + Self-Consistency@32 plateaus FIRMLY < 50%.** **ABORT
immediately if AR+SC@32 > 75%** — that is the regime-polluted ceiling trap that already produced
two expensive false negatives (P0.1, Thesis-A). This gate is non-negotiable before any new
generator training.

**TASK 1+ (only if 0/0.5 pass + headroom confirmed):** train a TRM-style recursive refiner with
a continuous Q-head DISTILLED from the Carnot-verifier oracle (verifier scores final
trajectories → BCE/MSE on the Q-head), matched-COMPUTE vs AR, on the headroom corpus.

**Strategic bottom line:** the pivot survives and may be STRONGER — Carnot as the
verification-oracle that distills the Q-heads the winning small-reasoner class needs, used
offline where the verifier's discreteness is a feature not a bug. But Task 0 must run first; it
can kill the naive integration for free.

---

## TRM CHECKPOINT NOW BEING TRAINED (2026-06-04, operator option 1) — load recipe for the bridge test

No off-the-shelf TRM exists (repo archived, no released checkpoints, smallest reference run
~18h/L40S/48GB). Operator authorized training one. A background agent set it up + launched a
REDUCED Sudoku-Extreme run on cuda:0 (24GB-fit). The exp3819/3821/3823 blocks were correct —
this artifact unblocks the whole TRM thread. **Operational details the bridge test (exp3819-
class re-run) needs to LOAD the checkpoint:**

- **Repo:** `/home/ianblenke/trm_src` (github.com/SamsungSAILMontreal/TinyRecursiveModels).
- **Venv:** `/home/ianblenke/trm_venv` (python3.12, **torch 2.9.1 cu130** — NOT the pinned 2.7
  cu126; host CUDA is 13.2, so cu130 was required + adam-atan2 built with --no-build-isolation).
  Do NOT use the carnot-ebm .venv for TRM.
- **REQUIRED env var to load/run the checkpoint (else libnvrtc-builtins.so.13.0 JIT crash):**
  `LD_LIBRARY_PATH=/home/ianblenke/trm_venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH`
- **Checkpoint dir:** `/home/ianblenke/trm_src/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/trm_sudoku_24gb/`
  — `step_<N>` files (raw `model.state_dict()`), one per eval interval (first ~step 5200, 5 total).
  Arch config to reconstruct the TRM: `all_config.yaml` in that dir (arch=trm, mlp_t=True,
  pos_encodings=none, L_layers=2, H_cycles=3, L_cycles=6, hidden_size 512).
- **Dataset:** `/home/ianblenke/trm_src/data/sudoku-extreme-1k-aug-1000` (HF sapientinc/sudoku-extreme).
- **Launch cmd + log:** `/home/ianblenke/trm_train.log`; training PID 612874; ETA ~2h45m
  (26,041 steps @ ~2.6 it/s); GPU0 only, GPU1 untouched.
- **The grid VERIFIER** for the bridge test = a Sudoku validity checker (rows/cols/boxes contain
  1-9 once) applied to the decoded intermediate board states.

---

## LEAPFROG: official HRM Sudoku checkpoint DOWNLOADED (2026-06-04, post-eGPU-crash)

The from-scratch TRM training was lost to the eGPU drop. Instead of re-training (~2h45m + eGPU
risk), downloaded a TRUSTED off-the-shelf recursive-latent refiner — answers the operator's
"borrow existing parts to save training time". The bridge test (exp3819-class) can probe THIS
instead of a self-trained TRM.

- **HRM Sudoku checkpoint:** `/home/ianblenke/hrm_ckpt/sudoku-extreme/` — `checkpoint` (109MB
  state_dict, ~27M-param HRM) + `all_config.yaml` (arch config). Source:
  `sapientinc/HRM-checkpoint-sudoku-extreme` (OFFICIAL, Sapient — the lab that made HRM, TRM's
  direct parent; trusted provenance, preferred over the unaudited community `Sanjin2024/...TRM`
  per audit-before-integration). HRM is the same recursive-latent-refinement family — equally
  valid for the bridge hypothesis (can an external verifier read its intermediate latents?).
- **To LOAD it:** needs the HRM repo (github.com/sapientinc/HRM — reachable, cloneable to a
  scratch dir OUTSIDE the carnot repos, like the TRM clone, to avoid the gitlink-breaks-Pages
  bug). `torch.load` the `checkpoint` into the HRM arch built from `all_config.yaml`.
- **CAUTION:** `torch.load` of a 3rd-party checkpoint can execute pickle code — it's the OFFICIAL
  Sapient checkpoint (low risk) but load with care / `weights_only=True` if the format allows.
- Alternatives if HRM integration is heavy: the exact-match community TRM checkpoint
  `Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-mlp` loads into our existing ~/trm_src TRM code
  (but unaudited — pickle risk), or re-train TRM on the INTERNAL GPU only (no eGPU, ~2h45m).
- **GPU lesson (cost us a crash + reboot):** run sustained GPU jobs on the INTERNAL 3090
  (GPU-7971baff, currently cuda:1) ONLY; the eGPU (GPU-b52387a2, cuda:0) drops off the USB4 bus
  under sustained load and corrupts CUDA. Pin with `CUDA_VISIBLE_DEVICES=GPU-7971baff-...`. NEVER
  run two heavy GPU jobs in parallel across both 3090s.

---

## LITERATURE DEEP-READ (2026-06-04): LDT + Huginn — REFRAMES the bridge test

### LDT (arXiv:2605.08605) — Carnot's transpilation+abstention thesis, ALREADY BUILT AND WORKING

LDT is, mechanically, the thing Carnot's `project_dbae_ebm_phase3` + `project_continuous_ising_rank`
+ the abstention-calibrated verifier have been describing — instantiated and SOTA:
- **The lattice IS a Cousot abstract-interpretation lattice.** Abstract domain = per-cell candidate
  sets (cell → subset of {1..9}), ordered pointwise by inclusion; lower=more informative, ⊤=all
  candidates, ⊥=conflict. Formal methods, not a metaphor.
- **Projection = thresholded discretization of the latent each step.** Latent = multi-hot (9
  sigmoids/cell); σ<θ_elim ⇒ eliminate ⇒ discretize into the lattice. This IS the sign(z)→discrete
  bottleneck (Q11), done rigorously and per-step.
- **It passes the DISCRETE LATTICE between steps, NOT a continuous embedding** (the key arch
  difference vs TRM/HRM, which pass opaque latents). So every intermediate state is a valid,
  interpretable, verifiable partial solution.
- **Soundness = abstention.** "returns a correct answer or abstains; it is never incorrect."
  Two heads at opposite ends of soundness/completeness: a candidate-elimination head (must be
  SOUND — asymmetric BCE w⁺/w⁻=8 penalizing false eliminations) + a CLS conflict head (must be
  COMPLETE — detect ⊥). This is exactly the abstention-calibrated-verifier + verified-or-abstain
  pattern. NO Q-head; termination = singleton solution or conflict.
- **On-policy training mirrors a search-based constraint solver** (a pool of partial lattice
  states → forward+threshold+branch; same Solve loop at train and inference → no distribution
  shift, unlike TRM/HRM).
- **Results: 100% Sudoku-Extreme (TRM 87.4%), 99.9% Maze-Hard (TRM 85.3%), trained in 15 min on
  1×B200 (TRM: 36h on 4×L40S).** Crushes TRM on accuracy AND ~100× cheaper to train.

**WHY THIS REFRAMES OUR BRIDGE TEST (the strategy shift).** Our Latent-Symbol Bridge question was
"can an external verifier read a recursive refiner's CONTINUOUS intermediate latents, or is offline
distillation the only path?" LDT shows there is a THIRD, BETTER path that DISSOLVES the question:
**architect the refiner's STATE to BE the discrete verifiable structure each step** (don't read
opaque latents; don't distill — transpile the state itself). The three options are now:
  (a) runtime external verifier reads continuous intermediate latents → DEAD (DT-P3; + Huginn
      confirms latents are non-decodable);
  (b) offline DISTILLATION oracle (Carnot verifier → train a continuous Q-head) → viable;
  (c) **state-IS-the-lattice (LDT / Carnot transpilation)** → WORKS, sound, SOTA. STRONGEST.
LDT validates that (c) — Carnot's transpilation+abstention thesis — is the right architecture.
**But LDT is partly a SCOOP:** it built (c) for CLOSED grid domains with a HAND-BUILT per-cell
lattice. **The gap LDT leaves = Carnot's differentiated angle:** a GENERAL, multi-domain VERIFIER
ENSEMBLE (k=15: SAT/Z3/AST/energy/…) as the abstraction lattice for domains where the lattice
CANNOT be hand-built (open-ended NL reasoning, code, proofs). That is the novel, defensible
experiment — NOT re-probing HRM's continuous latents (which we now know returns the DT-P3 negative).

### Huginn / Recurrent-Depth (arXiv:2502.05171) — recursion DOES transfer to TEXT, but latents stay opaque

- 3.5B params, recurrent block R repeated r times (input injected every step), test-time r
  arbitrary; effective depth 132 layers at r=32.
- **Halting = UNSUPERVISED KL-convergence** (KL(p_i,p_{i+1})<5e-4), hand-set, NO learned Q-head/
  verifier. Exits early on easy tokens, late on hard (~3.5 more steps on moral-scenario tokens).
- **Recursion transfers to TEXT reasoning:** GSM8K 42% vs a non-recurrent baseline's 2.2% (5×+),
  HellaSwag 65%, HumanEval 23%, ARC-Challenge 38%. → counters the HRM "grid-bound/task-specific"
  worry; partially answers DT-P4: the paradigm is NOT grid-only.
- **Intermediate latents are NOT decodable to reasoning steps** — hard tokens enter ORBITS (not
  fixed points), "slider" drifts (iteration counting), path-independent. → empirically CONFIRMS
  DT-P3: an external verifier cannot read continuous intermediate latents. So verifier-integration
  must be (b) or (c), never (a).
- Caveat: undertrained (800B tokens), lags OLMo on memorization; "less capacity to memorize facts,
  more to reason about context."

### NET PLAN UPDATE
1. **Down-weight / reframe exp3819 (Latent-Symbol Bridge on HRM):** Huginn already gives the DT-P3
   answer (continuous latents opaque). Running it on HRM mostly re-confirms a known negative.
2. **The high-value experiment is now the LDT-gap test:** can Carnot's verifier ensemble act as the
   per-step abstraction lattice / soundness oracle for a recursive refiner on a domain LDT did NOT
   cover (multi-domain or open-ended), where the lattice is NOT hand-buildable? That is the novel,
   defensible contribution — Carnot's k=15 ensemble as a GENERAL abstraction operator vs LDT's
   single hand-built Sudoku lattice.
3. **Huginn = the text-transfer existence proof** for whether to pursue text at all (recursion helps
   on GSM8K); pair with a verifier-guided halt (smarter than blind KL-convergence) as a concrete
   Carnot angle.
4. Read in full next if pursued: PTRM (2605.19943, the Q-head selector), GRAM (2605.19376), Coconut
   (2412.06769), Encode-Think-Decode (2510.07358).
