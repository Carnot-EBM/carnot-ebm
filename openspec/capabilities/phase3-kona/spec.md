# Phase 3 — Kona Parity Capability Specification

**Capability:** phase3-kona
**Version:** 0.1.0 (exploratory)
**Status:** Draft — primitives only, not yet a shipped capability
**Traces to:** PRD Phase 3 vision (see `_bmad/prd.md`), CLAUDE.md "functional parity with Kona"

## Overview

This capability specifies what it means for Carnot to reach **functional parity with
Logical Intelligence's Kona** — an open-source foundation model whose operating
principles align with Carnot's energy-based lineage. Parity here is defined by four
observable properties, not by parameter count or training compute. An implementation
reaches parity when it can demonstrate all four on a benchmark task, even at small
scale.

The four properties:

1. **Continuous latent reasoning.** The model operates in a continuous latent space
   during its reasoning phase, not in discrete token space. Token decoding happens
   only at the final output step.
2. **Non-autoregressive generation.** Output sequences are produced by iterative
   refinement of the full answer in parallel, not by sampling one token at a time
   from left to right. The inference-time primitive is *energy minimisation over the
   full answer*, not *sample next token given previous tokens*.
3. **Self-correction inside the forward pass.** The verify-and-repair loop that
   Carnot currently implements as an external wrapper (Phase 1
   `VerifyRepairPipeline`) is internalised — the model emits an answer only after
   its own energy has converged below a threshold, or after a bounded number of
   refinement steps.
4. **Hardware portability.** The refinement step maps onto Carnot's existing
   sampler-backend abstraction (`CpuBackend`, `FpgaBackend`, `TsuBackend`,
   `DWaveBackend`). Parity does not require any specific hardware, only that the
   architecture does not pin the model to general-purpose CPUs/GPUs.

This spec is deliberately **capability-level, not implementation-level**. Specific
architectural choices (transformer depth, continuous-latent dimension, halting
mechanism) go in `design.md`. The spec below is the set of observable properties a
concrete implementation has to satisfy.

## Phase 3 dependencies

This capability has **hard upstream dependencies** that are not yet resolved:

- **Phase 1 maturity.** The verify-repair loop must be reliable, well-calibrated,
  and stable enough to serve as the training target for self-correction. Phase 1 is
  currently mature (milestone 2026.04.60 closed), but audit-surfaced retractions
  continue to land in the research record. Phase 3 training on an unreliable
  Phase 1 target would teach the model to reproduce Phase 1's blind spots.
- **Phase 2 hardware scale-up.** Phase 3 training requires fast energy evaluation
  in the refinement loop. At the current 32-spin KV260 prototype scale (live as of
  2026-04-22), the hardware can hold a demonstration problem but cannot serve
  production-scale refinement for a model of interesting size. Either an XCZU-scale
  FPGA, a photonic Ising machine, or an Extropic TSU is the unlock.
- **Dataset and compute.** Kona-style training is unlikely to fit on the current
  dual-RTX-3090 setup. This capability assumes access to cluster-scale compute
  (cloud GPU rental or a research grant) for Stage 3 onward.

Absent these dependencies, Stages 1 and 2 are still valuable — they produce the
architectural primitives and a small-scale demonstration. Stage 3+ is blocked on
the upstream unlocks above.

## Stages

Parity is reached through four stages, each with its own acceptance gate. Earlier
stages are prerequisites for later ones; a stage is not considered started until
the previous stage's gate is passed.

- **Stage 1 — Architecture primitives** (no training).
  Implement the Recurrent-Depth Transformer (RDT) scaffolding in
  `python/carnot/phase3/rdt/`, with LTI-constrained injection, loop-index
  positional embeddings, and an adaptive halting head. Unit tests validate the
  fixed-point convergence property on a synthetic energy landscape.
- **Stage 2 — Tiny-scale end-to-end demonstration** (training on a toy task).
  Train a sub-100M-parameter RDT on a synthetic continuous-latent reasoning task
  (e.g., arithmetic in latent space, or small-graph constraint satisfaction).
  Demonstrate that the model reaches an answer via iterative refinement rather
  than autoregression, and that the number of refinement steps adapts to problem
  difficulty.
- **Stage 3 — Internalised verify-repair** (training on real data).
  Extend Stage 2 with a training signal derived from Phase 1's verify-repair
  pipeline. The refinement step should reduce both the prediction loss and the
  Phase 1 energy score simultaneously. Demonstrate that the trained model's
  single forward pass is competitive with an autoregressive baseline + Phase 1
  verify-repair wrapper at the task.
- **Stage 4 — Hardware-accelerated refinement**.
  Bind the refinement step to one of the Phase 2 sampler backends. Demonstrate
  that the same model checkpoint runs on CPU, GPU, and at least one accelerator
  (KV260 FPGA initially, TSU / photonic when available) with the same observable
  behaviour, only different wall-clock per refinement step.

## Requirements

### REQ-KONA-001: Continuous Latent Refinement

A Phase 3 model MUST maintain its reasoning state as a continuous tensor
(`jnp.ndarray` of float32 or float16) throughout the refinement loop. Token
embeddings enter the continuous state at the prelude stage; token decoding occurs
only at the coda stage. The refinement block MUST NOT sample tokens.

**Rationale:** this is the distinguishing property of Kona-style reasoning.
Discrete-token iteration is autoregressive generation and does not count as Phase 3
parity even if it is iterated.

**Acceptance criteria:**

- `RDTModel.refine_step(state)` has signature `state -> state` where both sides are
  `jnp.ndarray`, not `TokenSequence`.
- A unit test asserts that no `jax.random.categorical` or equivalent token-sampling
  call occurs inside `refine_step`.

### REQ-KONA-002: Bounded-Depth Iterative Refinement

The refinement block MUST be applied a bounded number of times per input (between
`min_steps` and `max_steps`, both configurable), with a halting head that can stop
early when energy convergence is detected.

**Rationale:** unbounded iteration is not a production-viable primitive. The
bound gives the runtime a worst-case guarantee, and the halting head gives the
model agency to stop early on easy inputs.

**Acceptance criteria:**

- `RDTModel.generate(input, min_steps=1, max_steps=64)` returns in at most
  `max_steps` iterations for any input.
- The halting head is trained to predict the energy gradient magnitude; a unit
  test asserts that on a problem with an analytic fixed point, the halting head
  fires within 20% of the true fixed-point step count.

### REQ-KONA-003: LTI Stability Constraint

The injection matrix used to reinject the encoded input at each refinement step
MUST be LTI-constrained — its spectral radius MUST stay strictly below 1.0
during training. This is enforced by spectral-norm regularisation or an equivalent
constraint.

**Rationale:** without this constraint, iterative refinement can diverge. The
spectral-radius bound is the standard stability proof for linear dynamical
systems and is cheap to enforce with a regularisation term.

**Acceptance criteria:**

- `LTIInjectionLayer.effective_spectral_radius()` returns a float < 1.0 at every
  training step.
- A unit test asserts that on a synthetic diverging initialisation, the
  constraint training loop brings the spectral radius below 1.0 within 100 steps.

### REQ-KONA-004: Energy-Convergence Halting Criterion

The halting decision MUST be triggered either by the adaptive halting head (learned
signal) or by a direct energy-convergence check (analytic signal): the refinement
loop stops when `|E(state_t) - E(state_{t-1})| < tolerance` for a configurable
tolerance.

**Rationale:** a learned halting head is a useful training signal but not a
reliability guarantee. The analytic energy-convergence check is the principled
stopping condition and MUST be available as a fallback.

**Acceptance criteria:**

- `RDTModel.generate(..., use_learned_halting=False)` stops on the energy
  criterion alone.
- Both modes agree on the stopping step within 10% on a validation set.

### REQ-KONA-005: Internalised Verify-Repair Signal

A Stage 3+ Phase 3 model MUST include a training loss term derived from Carnot's
Phase 1 verify-repair pipeline. The refinement step is rewarded for reducing
Phase 1's violation energy alongside the primary prediction loss.

**Rationale:** this is what moves the verify-repair loop from "external wrapper"
to "internal behaviour". Without it, Stage 3 produces a pretty RDT that still
needs Phase 1 around it — which is not parity.

**Acceptance criteria:**

- The training loss includes a `phase1_energy_weight * phase1_violation_energy`
  term with `phase1_energy_weight > 0`.
- An ablation unit test confirms that training with this term produces a model
  whose single forward pass has strictly lower Phase 1 energy than training
  without it, on a held-out set.

### REQ-KONA-006: Sampler-Backend Compatibility

The refinement step MUST be implementable as a call to any of the
`SamplerBackend` implementations in `python/carnot/samplers/`. Swapping backends
MUST NOT change the model's observable output distribution (within numerical
tolerance), only the wall-clock per step.

**Rationale:** hardware portability is one of the four parity properties. It also
makes the capability honest — if Phase 3 is irreversibly wedded to CPU/GPU, it
has not actually used Carnot's hardware lineage.

**Acceptance criteria:**

- A test runs the same checkpoint through `CpuBackend` and at least one other
  backend on a small validation set and asserts output KL-divergence < 0.01.
- FpgaBackend support is exercised when the KV260 is available.

### REQ-KONA-007: Honest-Verdict Emission

Phase 3 experiments MUST populate the `honest_verdict` schema field with one of:
`stage1_primitives_only`, `stage2_toy_converged`, `stage2_toy_diverged`,
`stage3_verify_repair_internalised`, `stage3_verify_repair_regressed`,
`stage4_backend_swap_verified`, `stage4_backend_swap_failed`,
`option_a_viable_above_95pct`, `option_a_marginal_90_to_95pct`,
`option_a_failed_below_90pct`, `phase3_continuous_ebm_not_found`, or
`regime_A_hmc_viable`, `regime_B_preconditioning_needed`,
`regime_C_hmc_inappropriate`, `sampler_kl_below_05_viable`,
`sampler_kl_above_05_needs_tuning`, `regime_c_langevin_deployed`,
`phase4_better_than_baseline`, `phase4_tied_with_baseline`,
`phase4_worse_than_baseline`, `prototype_only_no_convergence`,
`diagnostics_partial_pipeline_not_found`, `pipeline_not_found_blocked`, or
`blocked_*` for dependency failures.

**Rationale:** Phase 3 is where optimistic claims are most tempting. The
`honest_verdict` discipline that caught the Phase 1 retractions (the "+64 pp VR",
the cross-dataset 0.96 AUROC, the 1.0 JEPA OOD AUC) must extend here.

**Acceptance criteria:**

- Any experiment script under `scripts/experiment_*phase3*.py` or
  `scripts/experiment_*kona*.py` asserts at exit that its artifact contains a
  non-empty `honest_verdict` matching the enum above.

### REQ-KONA-008: Latent-to-Validity Snap Sweep

Before Phase 4 commits to Option A (continuous relaxation plus nearest-neighbor
snap), Carnot MUST run a snap-validity diagnostic against the Phase 3 bounded
continuous latent space `z in [-1, 1]^d`. The diagnostic samples 10,000
continuous states uniformly from the latent hypercube, snaps each state to the
nearest discrete action, checks whether the snapped action is legally executable,
and emits `snap_validity_rate = n_legal_snaps / 10000`.

When a deterministic ARC-AGI-3 rule engine is not available in the repository,
the diagnostic MUST use a synthetic proxy: a deterministic set of legal action
points drawn from the 0.1-spaced grid in `[-1, 1]^d`, with nearest-neighbor
Euclidean snap and `proxy_used=True` in the artifact. The proxy action space MUST
be capped to at most 1,000 actions, matching the Q8 assumption that per-turn
action spaces are small enough for cheap snapping.

**Rationale:** Q8 identifies the snap sweep as the cheapest pre-prototype check
for whether Option A is viable. If fewer than 95% of continuous states snap to
legal actions, Phase 4 should pivot before investing in the HMC sampler.

**Acceptance criteria:**

- The artifact includes `latent_dim`, `n_states_sampled`, `n_legal_snaps`,
  `snap_validity_rate`, `snap_validity_gate_passed`,
  `phase4_option_a_viable`, `proxy_used`, `action_space_description`, and
  `honest_verdict`.
- `snap_validity_gate_passed` and `phase4_option_a_viable` are both true iff
  `snap_validity_rate >= 0.95`.
- `honest_verdict` is `option_a_viable_above_95pct` at or above 95%,
  `option_a_marginal_90_to_95pct` from 90% inclusive to below 95%, and
  `option_a_failed_below_90pct` below 90%.
- If the Phase 3 `ContinuousEBM` cannot be loaded, the artifact emits
  `honest_verdict='phase3_continuous_ebm_not_found'`.

### REQ-KONA-009: HMC Compatibility Diagnostics

Before Phase 4 commits to Hamiltonian Monte Carlo over the continuous latent
space, Carnot MUST run the Deep Think Q7 diagnostics against the current k=5
AND-composed verifier energy bridge. The diagnostic samples random latent
vectors with dimension `latent_dim` taken from the Exp 1154 snap-validity
artifact, evaluates D1 symplectic reversibility, D2 Hamiltonian energy
conservation, D3 cross-component gradient norm disparity, and D4 continuous
subspace recovery, and emits a regime classification in `{A, B, C}`.

When the k=5 verifier bridge includes text, symbolic, or otherwise
non-autodifferentiable components, the diagnostic MUST use central finite
differences for gradients and record `gradient_method="numerical_fd"` in the
artifact. The artifact MUST include the required D1-D4 scalar fields, per-
diagnostic regime signals, `hmc_regime_classified=True`, `hmc_regime`,
`recommended_sampler`, and `honest_verdict`.

**Rationale:** Q7 identifies these diagnostics as the cheap pre-prototype check
that prevents spending Phase 4 implementation time on HMC when the verifier
gradient is discontinuous, badly conditioned, or only usable after
preconditioning.

**Acceptance criteria:**

- `scripts/experiment_1155_hmc_compatibility_diagnostics.py` reads
  `results/experiment_1154_snap_validity_sweep.json` for `latent_dim` and
  writes `results/experiment_1155_hmc_compatibility_diagnostics.json`.
- D1, D2, and D3 regime signals use the thresholds specified in the Exp 1155
  task prompt: D1 A/B/C at `<0.01`, `<0.1`, otherwise C; D2 A/B/C at `<0.1`,
  `<1.0`, otherwise C; D3 A/B/C at `<10`, `<100`, otherwise C.
- The final HMC regime is the worst of the D1-D3 signals, with sampler
  recommendations `hmc`, `preconditioned_hmc`, or a Regime C fallback selected
  from D4.
- D4 reports whether low continuous-subspace Hamiltonian variance with high
  full-ensemble variance identifies discrete verifier components as the
  bottleneck.

### REQ-KONA-010: Regime-Conditional Phase 4 Sampler

Phase 4 MUST deploy the sampler recommended by the Exp 1155 HMC compatibility
artifact. Regime A uses NumPyro NUTS, Regime B uses preconditioned HMC with a
diagonal SoftAbs-style mass rescaling, Regime C with a D4 discrete-component
bottleneck uses blocked Gibbs for discrete verifier coordinates plus Langevin
updates for continuous verifier coordinates, and general Regime C uses adaptive
SGLD.

The implementation MUST expose `python/carnot/samplers/phase4_sampler.py` with
`Phase4Sampler`, which satisfies the `SamplerBackend` structural protocol and
provides `sample(energy_fn, init_state, n_steps, **kwargs) -> samples` for
continuous-latent energy functions. The experiment artifact for Exp 1156 MUST
include `hmc_regime_used`, `sampler_algorithm`, `sampler_module`,
`sampler_written`, `n_validation_examples`, `acceptance_rate`,
`kl_divergence_vs_boltzmann`, `n_tests_passing`,
`active_inference_sampler_ready`, `hmc_sampler_honest_result`, and
`honest_verdict`.

**Rationale:** Exp 1155 showed that direct HMC is inappropriate for the current
k=5 verifier bridge because symbolic verifier coordinates create a discrete
bottleneck. Phase 4 should therefore follow the diagnostic result rather than
ship a sampler whose assumptions are already known to be false.

**Acceptance criteria:**

- `Phase4Sampler.from_exp1155(...)` selects `blocked_gibbs` when the artifact has
  `hmc_regime="C"` and `d4_discrete_components_bottleneck=true`.
- Regime C blocked Gibbs samples remain finite and bounded in the latent cube,
  update at least one discrete coordinate by Gibbs/MH flips, and return a chain
  with shape `(n_steps, latent_dim)`.
- `scripts/experiment_1156_hmc_sampler_conditional.py` validates 100 synthetic
  latent examples for 1,000 sampler steps each, estimates KL divergence against a
  Boltzmann reference at `T=1`, and writes the required Exp 1156 JSON fields.

### REQ-KONA-011: NRGPT Energy Recurrence Seed

A Phase 3 architecture-seed experiment MUST evaluate whether an NRGPT-style
energy recurrence improves FoVer binary classification over the current
ContinuousEBM-shaped baseline. The recurrence takes a fixed FoVer embedding
`z`, applies bounded gradient descent on a learned 3-layer MLP energy
`E(z)`, and trains a linear classification head on the refined state.

The experiment MUST evaluate exactly 5,000 training examples and 500 held-out
examples when enough FoVer labels are available, MUST compare `n_iters=1` and
`n_iters=3` under the same learned energy function, and MUST emit the required
Exp 1163 artifact fields:
`n_training_pairs`, `n_eval_pairs`, `baseline_auroc`, `nrgpt_auroc_n1`,
`nrgpt_auroc_n3`, `nrgpt_above_baseline`, `n_iters_monotone`,
`energy_block_module_written`, `fover_data_source`,
`nrgpt_phase3_prototype_honest_result`, and `honest_verdict`.

**Rationale:** NRGPT claims that a GPT-style hidden state can be retrofitted
with an energy-based iterative refinement step. Carnot needs a small, honest
Phase 3 seed before committing that architecture to the foundation-model path.

**Acceptance criteria:**

- `python/carnot/phase3/nrgpt_energy.py` exposes `NRGPTEnergyBlock`, whose
  forward pass preserves embedding shape and applies `n_iters` bounded
  gradient-descent updates on a learned 3-layer MLP scalar energy.
- `scripts/experiment_1163_nrgpt_energy_native_prototype.py` writes
  `results/experiment_1163_nrgpt_energy_native_prototype.json` with every
  required field and an honest verdict from the Exp 1163 enum.
- The artifact reports whether `n_iters=3` beats the baseline and whether
  `n_iters=3 >= n_iters=1`; it MUST NOT coerce a negative or tied result into
  a positive verdict.

### REQ-KONA-012: Phase 4 Active Inference Pilot

Phase 4 MUST include a small active-inference pilot that composes a k=5 verifier
ensemble into a scalar variational free energy over the bounded latent cube,
minimises that energy with the Regime C Phase 4 sampler, snaps the minimised
latent to a legal action, and compares the selected actions against a random
legal-action baseline on at least ten deterministic ARC-AGI-3-like synthetic
5x5 puzzles.

The pilot artifact MUST include `prototype_operational`, `n_puzzles_evaluated`,
`phase4_mean_action_count`, `baseline_mean_action_count`,
`action_count_ratio`, `phase4_solved_rate`, `baseline_solved_rate`,
`energy_trace_monotone_fraction`, `free_energy_values`,
`comparison_to_seed_iq`, `blocked_gibbs_params`, and `honest_verdict`.

**Rationale:** Exp 1154 established that bounded continuous latents can snap to
legal action representatives, Exp 1155 selected Regime C, and Exp 1156 deployed
blocked Gibbs. The next check is whether minimising the AND-composed verifier
free energy produces measurably more efficient action selection than an
uninformed legal-action baseline.

**Acceptance criteria:**

- `ActiveInferencePilot.minimize_free_energy(...)` returns a bounded latent and
  an energy trace whose last value is no greater than the initial value when the
  sampler finds an improving state.
- `ARC3PuzzleEnv` exposes ten named 5x5 synthetic puzzles with 3-5 legal actions
  per step and finite 3-10 step solution traces.
- `scripts/experiment_1165_phase4_active_inference_pilot_v1.py` runs five
  episodes per puzzle per method, writes the required artifact fields, and sets
  `honest_verdict` from the measured Phase 4/baseline action-count ratio.

### REQ-KONA-013: ARC-AGI-3 Leaderboard Positioning and Themesis Outreach

Phase 4 MUST include a positioning artifact that cross-references the Exp 1165
active-inference pilot against the public ARC-AGI-3 leaderboard context and
drafts operator-reviewed Themesis outreach without sending mail automatically.

The artifact MUST include `seed_iq_score_confirmed`, `seed_iq_score`,
`seed_iq_action_efficiency`, `carnot_phase4_action_count_ratio`,
`leaderboard_comparison_table`, `themesis_email_drafted`,
`themesis_email_text`, and `honest_verdict`.

**Rationale:** Exp 1165 measured Carnot's free-energy minimisation prototype on
synthetic ARC-AGI-3-like puzzles. The next step is to document the honest
relationship between that pilot, Seed IQ's active-inference result, and weak
autoregressive ARC-AGI-3 baselines before any collaboration outreach is sent.

**Acceptance criteria:**

- `scripts/experiment_1166_arc_agi3_leaderboard_themesis_outreach.py` fetches
  the ARC Prize leaderboard context when available and falls back to the
  documented Seed IQ values from `ops/known-issues.md` when a Seed IQ row is not
  independently exposed.
- The comparison table includes Seed IQ, Carnot Phase 4, and frontier
  autoregressive LLM rows, with Carnot fields derived from Exp 1165 rather than
  hard-coded independently.
- The Themesis email draft names Denise Holt / Denis O., uses
  `ian@blenke.com`, stays under 300 words, and frames Carnot as Apache 2.0,
  decentralization-respecting, multi-vendor, and complementary to Seed IQ.

### REQ-KONA-014: NRGPT Per-Token Energy Inference

The NRGPT Phase 3 seed MUST expose per-token energy evaluation over FoVer
responses. `NRGPTEnergyModel.energy_per_token(response_tokens)` MUST process
tokens sequentially through a recurrent hidden state, apply a linear energy
readout at each token boundary, and return one scalar energy per input token.
`NRGPTEnergyModel.batch_energy(response)` MUST preserve the batch-level contract
by returning the sum of the per-token energies for the tokenized response.

The Exp 1172 runner MUST compare per-token FoVer localization against the
Exp 1163 batch AUROC baseline and write:
`per_token_auroc`, `batch_auroc_baseline`, `per_token_above_batch`,
`nrgpt_per_token_energy_above_batch`, `energy_spike_localization_rate`, and
`honest_verdict`.

**Rationale:** Exp 1163 showed that batch-level NRGPT energy recurrence can beat
the ContinuousEBM-shaped FoVer baseline. DoT masking and per-token GRPO reward
need a finer signal that identifies where response energy spikes rather than
only whether the whole response is risky.

**Acceptance criteria:**

- `python/carnot/phase3/nrgpt_energy.py` exposes `NRGPTEnergyModel` with
  `energy_per_token(response_tokens: list[str]) -> list[float]` and
  `batch_energy(response: str) -> float`.
- Correct FoVer-style arithmetic responses produce low, stable per-token energy;
  incorrect arithmetic responses produce their maximum energy within two tokens
  of the detected error token when such a token can be located.
- `scripts/experiment_1172_nrgpt_per_token_energy_inference.py` writes
  `results/experiment_1172_nrgpt_per_token_energy_inference.json` with every
  required Exp 1172 field and an honest verdict from
  `per_token_improves_auroc`, `per_token_tied_with_batch`, or
  `per_token_worse_than_batch`.

### REQ-KONA-015: Phase 4 Stronger BFS Baseline at 5x5 and 10x10

The Phase 4 active-inference pilot MUST be stress-tested against a
non-trivial BFS-to-goal baseline at both 5x5 and 10x10 grid sizes.
BFS is the gold standard for shortest-path finding on deterministic
puzzles; if Phase 4 cannot match or beat BFS on harder grids, Phase 4
adds no value over classical tree search and the project must report
that honest finding. If BFS hits exponential branching on 10x10
puzzles and becomes intractable while Phase 4 still solves, the
result is publishable evidence that variational free-energy
minimization beats brute-force search at scale.

`ARC3PuzzleEnv` MUST accept a `grid_size` parameter (default 5, also
support 10). At grid_size=10 the environment MUST expose ten 10x10
puzzles with 5-8 legal actions per step and step counts in the
4-10 range. Wrong actions on 10x10 MUST mutate the grid in a
deterministic way so the BFS state space genuinely branches; on 5x5
the existing wrong-action-stays-put behavior is preserved for
backward compatibility.

A `BFSBaseline` class MUST implement breadth-first search over the
puzzle state space, tracking `(step_index, grid)` as the dedup key
and exiting early with `BFS_intractable` when more than 100,000
states have been popped without finding a solved state.

The Exp 1189 artifact MUST include `bfs_baseline_implemented`,
`grid_sizes_tested`, `n_5x5_puzzles`, `n_10x10_puzzles`,
`phase4_5x5_action_ratio`, `phase4_10x10_action_ratio`,
`phase4_better_than_bfs_5x5`, `phase4_better_than_bfs_10x10`,
`bfs_intractable_10x10`, `free_energy_values_all_puzzles`,
`stronger_baseline_implemented`, `paper_narrative`, and
`honest_verdict`.

**Rationale:** Paper ISSUE-9 (paper-v5-integrity-remediation.md)
flagged that the random-legal-action baseline used by Exp 1165
solved 98% of the puzzles, demonstrating they were too easy for the
baseline to be a meaningful comparison. BFS-to-goal removes that
ambiguity: BFS finds the optimum on tractable puzzles, and Phase 4
must beat it on intractable ones to claim any advantage.

**Acceptance criteria:**

- `ARC3PuzzleEnv(grid_size=10)` exposes ten 10x10 puzzles whose
  initial grids have shape `(10, 10)` and whose `legal_actions`
  count is in `[5, 8]` at every step.
- `BFSBaseline.bfs_solve(env, puzzle_id)` returns either
  `(action_sequence, n_states_explored)` or `(None, n_explored)`
  when the 100,000-state cap is exceeded.
- `scripts/experiment_1189_phase4_stronger_baseline_10x10.py` runs
  Phase 4 and BFS on the same ten 5x5 and ten 10x10 puzzles, captures
  the full free-energy trace for every Phase 4 episode (closing
  ISSUE-9), and writes
  `results/experiment_1189_phase4_stronger_baseline_10x10.json` with
  every required field and an honest verdict from
  `phase4_beats_bfs_on_hard_puzzles`, `phase4_tied_with_bfs`,
  `phase4_loses_to_bfs_all_sizes`, or `bfs_mostly_intractable`.

### REQ-KONA-016: BFS-Intractable Scrambled-Grid Puzzles for Phase 4

The Phase 4 active-inference pilot MUST be evaluated on a puzzle
generator that produces initial states with strictly positive
energy. Exp 1189 (REQ-KONA-015) showed that the existing
`ARC3PuzzleEnv` always starts in a state where the legal-action set
contains the correct action and the default verifier returns 0 for
that action, so every Phase 4 free-energy trace was identically zero
and BFS was trivially tractable on all 20 puzzles. That comparison
told us nothing about whether Phase 4 has any advantage over BFS at
scale.

A new `ScrambledGridEnv` MUST therefore generate state-traversal
puzzles by applying `n_scramble_steps >= 50` random valid actions in
reverse from a known goal grid. The energy MUST be the Hamming
distance between the current grid and the goal grid (count of
differing cells), so the *initial* energy is strictly greater than
zero for every generated puzzle whenever scrambling actually
produced a state that differs from the goal. The action set MUST
have branching at least large enough that BFS-to-goal exceeds the
100,000-state cap on grid sizes >= 15x15 within roughly three depth
levels; cell-flip actions on a 15x15 mod-2 grid (225 actions per
state) are the canonical realisation.

The Exp 1210 artifact MUST include the fields enumerated in the
acceptance criteria so a reviewer can verify that:

1. The puzzle generator actually produces nonzero initial energy
   for ALL puzzles (not just on average).
2. BFS hits the 100,000-state cap on a *majority* of the generated
   puzzles (otherwise the benchmark is no harder than the Exp 1189
   benchmark and tells us nothing new).
3. Phase 4's free-energy traces start strictly above zero on every
   episode, because that is the only configuration in which a
   monotone-decreasing trace is empirical evidence that Blocked
   Gibbs free-energy minimization is doing useful work.

**Acceptance criteria:**

- `results/experiment_1210_phase4_bfs_intractable_puzzles_v2.json`
  exists with fields:
  - `n_puzzles_total: int` (>= 15)
  - `grid_size: int` (>= 15)
  - `n_scramble_steps: int` (>= 50)
  - `initial_energy_nonzero_fraction: float` (must be 1.0)
  - `bfs_intractable_count: int`
  - `bfs_intractable_fraction: float` (>= 0.5 for the experiment
    to support the headline claim)
  - `phase4_solved_on_intractable: int`
  - `phase4_energy_traces_all_nonzero_initial: bool`
  - `phase4_bfs_intractable_fraction_above_50pct: bool`
  - `honest_verdict` in `{"phase4_advantage_on_intractable",
    "phase4_tied_with_bfs_again",
    "puzzle_generator_fixed_but_bfs_still_tractable",
    "blocked"}`
- For every per-puzzle row, `initial_energy > 0` and the BFS result
  records both the explored-state count and the intractable flag.

### REQ-KONA-017: Phase 5-B In-Situ Training Loop With Verifier-Ensemble Grounding

The Phase 5-B in-situ training loop builds on the Phase 5-A prototype
(REQ-KONA-008 + REQ-KONA-012 reuse) by adding a Contrastive Divergence
(CD-1) update step to the encoder + energy MLP whenever a k=3 verifier
ensemble (Z3-math stub + causal-reasoning stub + ThinkPRM v2 stub)
AND-passes on the proposed action sequence. The 1000-query trajectory
MUST monitor the five Q9-detectable failure modes from
`openspec/change-proposals/in-situ-training-phase5-derisking.md`
(failure modes 4-8 in the Q9 catalog) so that any single mode firing
during training is auditable from the artifact alone, without re-running
the experiment.

The acceptance test for Phase 5-B is FIVE simultaneous gates, all of
which must pass for `phase5b_stability_confirmed = true`. A partial-pass
result is documented honestly under the corresponding gate failure
verdict — the experiment is not "blocked" simply because a stability
gate fired.

**Acceptance criteria:**

- `results/experiment_1223_phase5b_insitu_training_loop.json` exists
  with fields:
  - `n_queries_run: int` (1000)
  - `n_accepted_by_verifier: int`
  - `acceptance_rate: float`
  - `energy_decrease_pct: float`
  - `spectral_norm_growth_rate: float`
  - `acceptance_rate_sublinear: bool`
  - `mean_anchor_distance: float`
  - `oracle_accuracy_initial: float`
  - `oracle_accuracy_final: float`
  - `oracle_accuracy_drop_pp: float`
  - `gate1_energy_decrease_30pct: bool`
  - `gate2_no_representation_drift: bool`
  - `gate3_no_autocatalytic_spiral: bool`
  - `gate4_no_null_space_excavation: bool`
  - `gate5_no_catastrophic_forgetting: bool`
  - `gates_passed: int` in [0, 5]
  - `phase5b_stability_confirmed: bool` (gates_passed == 5)
  - `honest_verdict` in `{"all_5_gates_pass", "partial_gates",
    "gate_failure_diagnosed", "blocked"}`

### REQ-KONA-018: Phase 5-C Adversarial Probe — Three Attack Classes

The Phase 5-C hostile-reviewer round runs three adversarial attack classes
against the Phase 5-A+B prototype (REQ-KONA-017) BEFORE any Phase 5 scaling
decision is committed. This enforces the CLAUDE.md "Phase Prototype + Empirical
Validation + Adversarial Check Discipline" requirement. Phase 5-B stability
(`phase5b_stability_confirmed = true`) is a prerequisite.

**Attack 1 — Single-verifier gaming (budget: 200 queries):**
Detect inputs where V0 (Z3-verifier analog) passes alone (score > 0.9) but
the AND-composed k=3 composite rejects (score < 0.5). Defense passes if
`gaming_rate_attack1 < 0.10`.

**Attack 2 — Pairwise correlation exploitation (budget: 200 queries):**
Compute the empirical N×N conditional-acceptance probability matrix
P(V_i passes | V_j passes). Defense passes if
`pairwise_max_correlation < 0.70` (no correlated blind spots).

**Attack 3 — Joint null-space gradient attack (budget: 20 starts × 50 steps):**
Start from invalid puzzle states; gradient-ascend the energy MLP (continuous
proxy) for 50 steps; check if proxy > 0.8 while actual verifiers still reject.
Defense passes if `joint_gaming_rate < 0.05`.

**Acceptance criteria:**

- `results/experiment_1224_phase5c_adversarial_probe.json` exists with fields:
  - `gaming_rate_attack1: float` in [0, 1]
  - `pairwise_max_correlation: float` in [0, 1]
  - `joint_gaming_rate: float` in [0, 1]
  - `attack1_blocked: bool`
  - `attack2_blocked: bool`
  - `attack3_blocked: bool`
  - `all_attacks_blocked: bool`
  - `failure_modes_discovered: list[str]` (empty if all_attacks_blocked)
  - `architectural_revision_if_needed: str` ("none" if all blocked)
  - `adversarial_probe_complete: bool` (True)
  - `honest_verdict` in `{"all_attacks_blocked_architecture_validated",
    "partial_attack_success_revision_needed",
    "full_gaming_found_architecture_invalid", "blocked"}`

### REQ-KONA-020: Phase 5-D Intermediate-Scale In-Situ Derisking

The Phase 5-D intermediate-scale run bridges the validated toy-scale
Phase 5-A/B/C substrates and later 1B+ substrate training. It MUST expose a
100M+ parameter-count encoder/refiner architecture at latent dimension 128,
run a four-arm in-situ training comparison, and measure all eight Q9/Q12
failure-mode gates in one auditable artifact. The production-scale-only
gates are measurement gates: the artifact MUST report the measurements and
honestly classify whether a production failure mode was found.

The four arms are:

- Arm A: standard PCD control with no anti-gaming regularization.
- Arm B: Q12 entropy regularization for substrate-gaming resistance.
- Arm C: DR-3 upstream variance compensation before the sign bottleneck.
- Arm D: combined Q12 entropy regularization and DR-3 compensation.

The run MUST include PPSEBM-style early replay by keeping a 100-sample replay
buffer and mixing 10% replay into the training stream. It MUST report the
oracle-accuracy delta with replay versus without replay.

**Acceptance criteria:**

- `python/carnot/phase5/intermediate_scale.py` exposes the Phase 5-D
  substrate/gate helpers and artifact builder.
- `results/experiment_1238_phase5d_intermediate_scale.json` exists with fields:
  - `encoder_param_count: int` (>= 100,000,000)
  - `latent_dim: int` (128)
  - `n_queries_run: int` (1000)
  - `gate1_energy_decrease_30pct: bool`
  - `gate2_no_representation_drift: bool`
  - `gate3_no_autocatalytic_spiral: bool`
  - `gate4_no_null_space_excavation: bool`
  - `gate5_no_catastrophic_forgetting: bool`
  - `gate6_mode_collapse_entropy: float`
  - `gate6_mode_collapse_absent: bool`
  - `gate7_mcmc_mixing_acf1: float`
  - `gate7_mcmc_mixing_ok: bool`
  - `gate8_substrate_shift_saturation_frac: float`
  - `gate8_substrate_shift_absent: bool`
  - `mcmc_chain_distance_trajectory: list[float]`
  - `latent_centroid_cosine_drift: list[float]`
  - `trace_cov_trajectory: list[float]`
  - `per_pair_conditional_trajectory: dict[str, list[float]]`
  - `energy_on_accepted_trajectory: list[float]`
  - `replay_buffer_effectiveness: float`
  - `gates_measured: int` (8)
  - `phase5d_all_8_gates_measured: bool` (true)
  - `honest_verdict` in `{"all_8_gates_pass",
    "toy_gates_pass_production_modes_detected",
    "production_failure_mode_found", "prototype_failed_initialization",
    "blocked"}`

### REQ-KONA-023: Phase 5-D Intermediate-Scale v2 Four-Gate Measurement

The Phase 5-D v2 run is a CPU-feasible intermediate derisking pass after the
Phase 5-B 50K-parameter loop passed all five toy gates. It MUST build a
d_model=32, four-layer in-situ substrate at approximately one-million
parameters, verify the Phase 5-B prerequisite artifact, and measure at least
four of the eight Phase 5-D failure-mode gates in one auditable artifact.
Unmeasured gates MUST be represented explicitly as `null`/`None` rather than
omitted or silently treated as failures.

The measured v2 gates are:

- Gate 1: accepted-sample energy decreases by at least 20% over the trajectory.
- Gate 2: output diversity remains non-collapsed; the mean per-dimension output
  standard deviation is greater than 0.01.
- Gate 3: MCMC mixing avoids paralysis; absolute lag-1 autocorrelation is less
  than 0.90.
- Gate 4: catastrophic forgetting is absent; late held-out accuracy drops by
  less than 20 percentage points relative to early held-out accuracy.

**Acceptance criteria:**

- `python/carnot/phase5/intermediate_scale_v2.py` exposes the Phase 5-D v2
  substrate, gate helpers, and artifact builder.
- `results/experiment_1250_phase5d_intermediate_scale_v2.json` exists with
  fields:
  - `phase5d_gates: dict` with exactly eight gate keys and `null` for
    unmeasured gates.
  - `phase5d_gates_passed: int` (>= 4 for success).
  - `phase5d_gates_measured: int` (>= 4).
  - `model_scale: str` containing `1M_params_d32_4layers`.
  - `ppsebm_replay_buffer: str` describing the anti-forgetting replay approach.
  - `honest_verdict: str` formatted as
    `phase5d_N_gates_passed_M_measured`.
- The artifact MUST also preserve the Phase 5-B prerequisite status and the
  measured numerical values used to derive the four gate booleans.

### REQ-KONA-019: Boltzmann-GPT Contrastive Training on FoVer

The Exp 1237 Boltzmann-GPT training run MUST train a visible-hidden
Boltzmann-GPT energy layer on FoVer correct/incorrect traces with contrastive
divergence loss `E_correct.mean() - E_incorrect.mean()`. The run MUST use a
deterministic 80/20 train/test split, train for 10 epochs with Adam
(`lr=1e-3`, `batch_size=16`), save a PyTorch checkpoint, and evaluate held-out
AUROC with `roc_auc_score(labels, -energies)` so lower energy means a trace is
more likely correct.

The artifact MUST include `n_training_epochs`, `n_train_samples`,
`n_test_samples`, `boltzmann_gpt_contrastive_auroc`,
`nrgpt_auroc_baseline`, `boltzmann_gpt_beats_seed`,
`boltzmann_gpt_above_0p80`, `checkpoint_path`, and `honest_verdict`.

**Rationale:** Exp 1226 showed that random Boltzmann-GPT weights have
non-degenerate FoVer signal but remain below NRGPT. Exp 1237 checks whether the
energy-gap training signal is enough to move the architecture above the 0.80
AUROC threshold.

**Acceptance criteria:**

- `python/carnot/phase3/boltzmann_gpt.py` exposes a trainable
  `BoltzmannGPTLayer` and contrastive training helpers that lower correct-trace
  energy relative to incorrect-trace energy on separable FoVer rows.
- `python/carnot/data/fover.py` exposes `FoVerDataset`, whose labels are binary
  with `1` for correct and `0` for incorrect traces.
- Running Exp 1237 writes
  `results/experiment_1237_boltzmann_gpt_contrastive_training.json` and
  `python/carnot/phase3/boltzmann_gpt_cd_v1.pt`, with the honest verdict
  derived only from the measured held-out AUROC.

### REQ-KONA-021: NRGPT Frozen-Prefix Monotonicity Diagnostic

The Exp 1239 NRGPT frozen-prefix diagnostic MUST evaluate ten FoVer response
sequences by comparing the existing NRGPT energy recurrence trace on each full
sequence with the recurrence trace on only that sequence's first token. A trace
is monotonic iff every subsequent recurrence energy is less than or equal to
the previous energy. The first-token run MUST remove all prefix context by
embedding only the first token under the same trained recurrence block and train
normalization statistics used for the full sequence run.

The artifact MUST include `n_sequences_tested`,
`n_monotonic_full_sequence`, `n_monotonic_first_token_only`, `regime`,
`paper_v6_framing_recommendation`, `frozen_prefix_regime_classified`, and
`honest_verdict`. The regime MUST be `b_causal_context_shift` when at least
eight first-token traces are monotonic, `c_non_conservative_preconditioner`
when at most two first-token traces are monotonic, and `mixed_ambiguous`
otherwise.

**Rationale:** Exp 1163 showed that the NRGPT iteration comparison is not
monotone. The next paper-v6 framing decision depends on whether that behavior
comes from causal-context energy shifts or from a learned non-conservative
preconditioner that remains non-monotone even without a prefix.

**Acceptance criteria:**

- `python/carnot/phase3/nrgpt_energy.py` exposes a frozen-prefix evaluator that
  writes `results/experiment_1239_nrgpt_frozen_prefix_evaluation.json` with the
  required fields and an honest verdict derived only from the first-token
  monotonicity count.
- `python/carnot/phase3/nrgpt.py` exposes an `NRGPT` compatibility wrapper so
  existing diagnostic imports can reach the energy recurrence trace without
  moving the underlying implementation out of `nrgpt_energy.py`.

### REQ-KONA-022: Boltzmann-GPT CD Training v2 Artifact

The Exp 1248 Boltzmann-GPT CD training v2 run MUST load
`results/fover_corpus_v5.json`, build a deterministic balanced correct/incorrect
training set from the available labeled rows, run a finite forward pass through
the trainable Boltzmann-GPT energy layer, and apply 100 contrastive-divergence
optimization steps that minimize `E_correct.mean() - E_incorrect.mean()`.

The artifact MUST be written to
`results/experiment_1248_boltzmann_gpt_cd_training_v2.json` and include
`forward_pass_ok`, `pre_cd_auroc`, `post_cd_auroc`, `n_cd_steps`, and
`honest_verdict`. `pre_cd_auroc` MUST preserve the Exp 1226 random-weight
baseline of `0.65`. `post_cd_auroc` MUST be measured from trained
Boltzmann-GPT energies, with lower energy treated as the correct-trace score.
`honest_verdict` MUST be formatted as `boltzmann_gpt_cd_auroc_X.XX` using the
measured post-CD AUROC rounded to two decimal places.

**Acceptance criteria:**

- `python/carnot/phase3/boltzmann_gpt.py` exposes a reusable Exp 1248 runner
  that reports the required schema fields without hard-coded post-training
  AUROC.
- The runner records the actual balanced FoVer v5 class counts used for the
  100-step CD pass.
- The generated artifact is complete only when the forward pass succeeds and
  the post-CD AUROC is finite.

## Scenarios

### SCENARIO-KONA-001: Stage 1 Primitive — RDT Fixed-Point Convergence

**Given** a trained `RDTModel` on a synthetic energy landscape with a known global
minimum
**When** `RDTModel.refine(state_0, max_steps=100)` is called from an arbitrary
initialisation
**Then** the returned state is within `tolerance` of the global minimum for all
initialisations in a validation cohort

**Spec traces:** REQ-KONA-001, REQ-KONA-002

### SCENARIO-KONA-002: LTI Constraint Holds During Training

**Given** an `RDTModel` with an `LTIInjectionLayer`
**When** 1,000 training steps are executed on a synthetic task
**Then** `effective_spectral_radius()` is strictly less than 1.0 at every
checkpoint

**Spec traces:** REQ-KONA-003

### SCENARIO-KONA-003: Learned and Analytic Halting Agree

**Given** a trained Stage 2 `RDTModel` with both halting modes available
**When** `generate(input, use_learned_halting=True)` and
`generate(input, use_learned_halting=False)` are called on 100 validation inputs
**Then** the two modes' stopping-step counts agree within 10% mean absolute error

**Spec traces:** REQ-KONA-004

### SCENARIO-KONA-004: Verify-Repair Loss Improves Phase 1 Score

**Given** two Stage 3 checkpoints trained with and without the
`phase1_energy_weight` term, all other hyperparameters equal
**When** both are evaluated on a held-out verify-repair benchmark
**Then** the model trained with the term has a lower mean Phase 1 violation
energy on single-forward-pass outputs

**Spec traces:** REQ-KONA-005

### SCENARIO-KONA-005: Backend Swap Preserves Output Distribution

**Given** a Stage 4 `RDTModel` checkpoint
**When** the same input batch is processed through `CpuBackend` and a second
backend (GPU or FpgaBackend when available)
**Then** the per-sample output distributions have KL divergence < 0.01

**Spec traces:** REQ-KONA-006

### SCENARIO-KONA-006: Non-Parity Attempt Emits Honest Verdict

**Given** a Stage 2 experiment where `refine_step` secretly calls
`jax.random.categorical`
**When** the acceptance test for REQ-KONA-001 runs
**Then** the test fails, the experiment artifact records
`honest_verdict='stage2_toy_diverged'`, and the run is not considered passing

**Spec traces:** REQ-KONA-001, REQ-KONA-007

### SCENARIO-KONA-007: Q8 Snap Validity Sweep Gates Option A

**Given** the Phase 3 `ContinuousEBM` latent dimension and no checked-in
ARC-AGI-3 rule engine
**When** `scripts/experiment_1154_snap_validity_sweep.py` samples 10,000 states
from `[-1, 1]^d` and snaps them to the synthetic legal action grid
**Then** it writes `results/experiment_1154_snap_validity_sweep.json` with
`n_states_sampled=10000`, the required REQ-KONA-008 fields, `proxy_used=True`,
and `phase4_option_a_viable == snap_validity_gate_passed`

**Spec traces:** REQ-KONA-008

### SCENARIO-KONA-008: Q7 HMC Diagnostics Classify The Verifier Gradient

**Given** the Exp 1154 snap-validity artifact contains `latent_dim`
**When** `scripts/experiment_1155_hmc_compatibility_diagnostics.py` runs the
D1-D4 diagnostics with finite-difference gradients on the k=5 verifier energy
bridge
**Then** it writes `results/experiment_1155_hmc_compatibility_diagnostics.json`
with all required REQ-KONA-009 fields
**And** the artifact records a classified regime, a sampler recommendation, and
`gradient_method="numerical_fd"` when symbolic components are present.

**Spec traces:** REQ-KONA-009

### SCENARIO-KONA-009: Exp 1156 Deploys The Regime-Conditional Sampler

**Given** `results/experiment_1155_hmc_compatibility_diagnostics.json` classifies
the HMC regime and recommends a sampler
**When** `scripts/experiment_1156_hmc_sampler_conditional.py` runs
**Then** it instantiates the matching `Phase4Sampler`, validates 100 synthetic
latent examples, estimates KL divergence against the `ContinuousEBM` Boltzmann
reference, and writes `results/experiment_1156_hmc_sampler_conditional.json`
with the required schema fields.

**Spec traces:** REQ-KONA-010

### SCENARIO-KONA-010: Exp 1163 Writes Honest NRGPT Comparison

**Given** FoVer labels are available from `data/fover_dataset.jsonl` or the
pipeline-generated FoVer corpus
**When** `scripts/experiment_1163_nrgpt_energy_native_prototype.py` runs
**Then** it trains the ContinuousEBM-shaped baseline and NRGPT recurrence on
5,000 examples, evaluates both on 500 held-out examples, and writes the required
REQ-KONA-011 artifact fields.

**Spec traces:** REQ-KONA-011

### SCENARIO-KONA-011: NRGPT Iteration Count Is Reported Without Bias

**Given** a learned `NRGPTEnergyBlock` and the same FoVer split
**When** classifiers are evaluated after `n_iters=1` and `n_iters=3`
**Then** the artifact sets `n_iters_monotone` to true iff
`nrgpt_auroc_n3 >= nrgpt_auroc_n1`, regardless of whether either NRGPT variant
beats the baseline.

**Spec traces:** REQ-KONA-011

### SCENARIO-KONA-012: Exp 1165 Measures Phase 4 Active Inference

**Given** Exp 1154 snap validity passed, Exp 1155 recommended blocked Gibbs, and
Exp 1156 reported `sampler_kl_below_05_viable`
**When** `scripts/experiment_1165_phase4_active_inference_pilot_v1.py` evaluates
the active-inference pilot and random legal-action baseline on ten synthetic
ARC-AGI-3-like puzzles
**Then** it writes `results/experiment_1165_phase4_active_inference_pilot_v1.json`
with all REQ-KONA-012 fields and an honest verdict derived from the measured
action-count ratio.

**Spec traces:** REQ-KONA-012

### SCENARIO-KONA-013: Exp 1166 Documents Leaderboard Context and Outreach

**Given** Exp 1165 has written `action_count_ratio` and `phase4_solved_rate`
**When** `scripts/experiment_1166_arc_agi3_leaderboard_themesis_outreach.py`
runs
**Then** it writes
`results/experiment_1166_arc_agi3_leaderboard_themesis_outreach.json` with all
REQ-KONA-013 fields, an honest verdict that distinguishes current leaderboard
confirmation from fallback documentation, and a ready-for-review outreach email.

**Spec traces:** REQ-KONA-013

### SCENARIO-KONA-014: Exp 1172 Localizes NRGPT Energy Spikes

**Given** FoVer responses and the Exp 1163 batch AUROC baseline artifact are
available
**When** `scripts/experiment_1172_nrgpt_per_token_energy_inference.py` runs
**Then** it evaluates per-token energy spikes against located arithmetic error
tokens, compares the per-token AUROC with the Exp 1163 batch baseline, and
writes the required REQ-KONA-014 artifact fields without converting a tied or
negative result into a positive verdict.

**Spec traces:** REQ-KONA-014

### SCENARIO-KONA-015: Exp 1189 Compares Phase 4 to BFS at 5x5 and 10x10

**Given** `ARC3PuzzleEnv` supports both 5x5 (default) and 10x10 grid sizes,
`BFSBaseline` is implemented with a 100,000-state intractability cap, and
the Phase 4 active-inference pilot is operational
**When** `scripts/experiment_1189_phase4_stronger_baseline_10x10.py` runs
the same ten 5x5 and ten 10x10 puzzles through both Phase 4 and BFS
**Then** it writes `results/experiment_1189_phase4_stronger_baseline_10x10.json`
with all REQ-KONA-015 fields, a full free-energy trace for every Phase 4
episode, and an honest verdict drawn from `phase4_beats_bfs_on_hard_puzzles`,
`phase4_tied_with_bfs`, `phase4_loses_to_bfs_all_sizes`, or
`bfs_mostly_intractable` based strictly on the measured action-count ratios
and BFS intractability counts.

**Spec traces:** REQ-KONA-015

### SCENARIO-KONA-016: Exp 1210 Phase 4 vs BFS on Scrambled-Grid Puzzles With Nonzero Initial Energy

**Given** `ScrambledGridEnv` generates a 15-puzzle batch on a 15x15 mod-2
grid by applying 50 random cell-flip actions to a known goal grid,
energy is the Hamming distance between the current grid and the goal,
the BFS baseline shares the 100,000-state cap from REQ-KONA-015, and
the Phase 4 Blocked Gibbs free-energy minimization is operational
**When** `scripts/experiment_1210_phase4_bfs_intractable_puzzles_v2.py`
runs both BFS and Phase 4 on each of the 15 generated puzzles
**Then** it writes
`results/experiment_1210_phase4_bfs_intractable_puzzles_v2.json` with
`initial_energy_nonzero_fraction == 1.0`,
`phase4_energy_traces_all_nonzero_initial == True`,
`bfs_intractable_fraction >= 0.5`, every per-puzzle row's recorded
initial energy strictly greater than zero, and an honest verdict
drawn from `phase4_advantage_on_intractable`,
`phase4_tied_with_bfs_again`,
`puzzle_generator_fixed_but_bfs_still_tractable`, or `blocked` based
strictly on the measured Phase-4-vs-BFS solve counts and the BFS
intractability fraction.

**Spec traces:** REQ-KONA-016

### SCENARIO-KONA-017: Exp 1223 In-Situ Training Loop Trajectory Passes Five Q9 Stability Gates

**Given** the Phase 5-A prototype is operational
(`results/experiment_1222_phase5a_insitu_prototype.json` has
`phase5a_prototype_ready == True`), the Phase 5-B module exposes
`run_phase5b_training_loop`, and a 20-puzzle frozen oracle set is
available
**When** `scripts/experiment_1223_phase5b_insitu_training_loop.py`
runs a 1000-query in-situ training trajectory with η=1e-5, k=3
verifier ensemble (Z3-math stub + causal-reasoning stub + ThinkPRM v2
soft-accept stub), and CD-1 PCD updates on the encoder + energy MLP
on every accepted query
**Then** it writes
`results/experiment_1223_phase5b_insitu_training_loop.json` with
all REQ-KONA-017 fields, every individual gate boolean derived
strictly from the corresponding measurement
(energy drop ≥ 30%, encoder spectral-norm growth rate < 0.01/query,
acceptance-rate first derivative sub-linear,
mean anchor distance > 0.5,
oracle-accuracy drop ≤ 5pp), and an honest verdict drawn from
`all_5_gates_pass`, `partial_gates`,
`gate_failure_diagnosed`, or `blocked` based strictly on the gate
boolean count.

**Spec traces:** REQ-KONA-017

### SCENARIO-KONA-020: Exp 1238 Measures All Eight Phase 5-D Gates

**Given** the Phase 5-A/B/C artifacts exist, Q12 anti-gaming instrumentation
is required, and an intermediate-scale Phase 5-D substrate exposes latent
dimension 128 with an encoder parameter count of at least 100,000,000
**When** Exp 1238 runs its four-arm in-situ trajectory with PPSEBM-style
replay and evaluates the five toy-detectable gates plus the three
production-scale-only gates
**Then** it writes
`results/experiment_1238_phase5d_intermediate_scale.json` with all
REQ-KONA-020 fields, `gates_measured == 8`,
`phase5d_all_8_gates_measured == True`, and an honest verdict drawn
strictly from the measured gate booleans.

**Spec traces:** REQ-KONA-020

### SCENARIO-KONA-023: Exp 1250 Writes Phase 5-D v2 Four-Gate Artifact

**Given** the Phase 5-B artifact reports `phase5b_stability_confirmed == True`
and a CPU-feasible d_model=32, four-layer substrate is available
**When** Exp 1250 measures the Phase 5-D v2 energy-decrease, mode-collapse,
MCMC-mixing, and catastrophic-forgetting gates
**Then** it writes
`results/experiment_1250_phase5d_intermediate_scale_v2.json` with all
REQ-KONA-023 fields, exactly four measured gate booleans, exactly four
unmeasured `null` gates, and an honest verdict string derived strictly from
the measured pass and measured gate counts.

**Spec traces:** REQ-KONA-023

### SCENARIO-KONA-019: Exp 1237 Writes Boltzmann-GPT CD Artifact

**Given** FoVer correct and incorrect traces are available from the checked-in
FoVer corpus or an explicit dataset path
**When** Exp 1237 trains the Boltzmann-GPT layer with contrastive divergence for
10 epochs and evaluates the held-out 20% split
**Then** it writes the required REQ-KONA-019 artifact fields, saves the
checkpoint, and sets `honest_verdict` to
`contrastive_auroc_above_0p80`,
`contrastive_auroc_improved_below_threshold`, `training_diverged`, or
`blocked` according to the measured result.

**Spec traces:** REQ-KONA-019

### SCENARIO-KONA-021: Exp 1239 Classifies Frozen-Prefix NRGPT Regime

**Given** FoVer responses are available from the checked-in corpus or a
synthetic fallback
**When** Exp 1239 evaluates ten full-sequence and first-token-only NRGPT energy
recurrence traces
**Then** it writes
`results/experiment_1239_nrgpt_frozen_prefix_evaluation.json`, reports both
monotonicity counts, classifies the regime according to REQ-KONA-021, and
includes a one- or two-sentence paper-v6 Section 4 framing recommendation.

**Spec traces:** REQ-KONA-021

### SCENARIO-KONA-022: Exp 1248 Writes Boltzmann-GPT CD v2 Artifact

**Given** the checked-in FoVer v5 corpus has labeled correct and incorrect
responses
**When** Exp 1248 runs 100 deterministic Boltzmann-GPT CD optimization steps
over a balanced class slice
**Then** it writes
`results/experiment_1248_boltzmann_gpt_cd_training_v2.json` with
`forward_pass_ok == True`, `pre_cd_auroc == 0.65`, `n_cd_steps == 100`, finite
`post_cd_auroc`, and an `honest_verdict` derived from the measured post-CD
AUROC.

**Spec traces:** REQ-KONA-022

## Out of scope

The following are deliberately **not** required by this capability:

- **Parameter-count parity with Kona's published size.** Parity is
  architecture-level, not scale-level. A 100M-parameter Phase 3 model that
  exhibits all four properties counts; a 35B-parameter autoregressive transformer
  does not.
- **Public-benchmark leadership.** Demonstrating Phase 3 parity on a toy task is
  sufficient for this capability; surpassing published benchmarks is a Phase 4
  ambition not captured here.
- **Tokenizer innovation.** Kona-style reasoning happens in continuous latent
  space after tokenization; the tokenizer itself is orthogonal.
- **Training-data innovation.** Whatever dataset trains Phase 1's verify-repair
  well enough to serve as the Stage 3 target is sufficient.

## Implementation status

- **Stage 1 primitives:** partial.
  `python/carnot/phase3/continuous_ebm.py` exists (Exp 435a) with
  `ContinuousEBMMinimiser`, Langevin and energy-matching samplers (Exp 446).
  The RDT scaffolding is **not yet written**. The LTI constraint layer is **not
  yet written**.
- **Stage 2 demo:** not started.
- **Stage 3 internalisation:** not started (Phase 1 maturity dependency).
- **Stage 4 hardware binding:** gated on Phase 2 scale-up.
- **Phase 4 action-representation preflight:** snap-validity sweep specified by
  REQ-KONA-008; implementation lives in
  `python/carnot/phase3/snap_validity.py` and
  `scripts/experiment_1154_snap_validity_sweep.py`. HMC compatibility
  diagnostics are specified by REQ-KONA-009 and implemented in
  `python/carnot/phase3/hmc_compatibility.py` and
  `scripts/experiment_1155_hmc_compatibility_diagnostics.py`. The
  regime-conditional sampler is specified by REQ-KONA-010.
- **NRGPT architecture seed:** specified by REQ-KONA-011; implementation lives
  in `python/carnot/phase3/nrgpt_energy.py` and
  `scripts/experiment_1163_nrgpt_energy_native_prototype.py`.
- **Phase 4 active-inference pilot:** specified by REQ-KONA-012; implementation
  lives in `python/carnot/phase3/active_inference_pilot.py` and
  `scripts/experiment_1165_phase4_active_inference_pilot_v1.py`.
- **ARC-AGI-3 positioning and Themesis outreach:** specified by REQ-KONA-013;
  implementation lives in
  `scripts/experiment_1166_arc_agi3_leaderboard_themesis_outreach.py`.
- **NRGPT per-token energy inference:** specified by REQ-KONA-014; implementation
  lives in `python/carnot/phase3/nrgpt_energy.py` and
  `scripts/experiment_1172_nrgpt_per_token_energy_inference.py`.
- **Boltzmann-GPT contrastive training:** specified by REQ-KONA-019;
  implementation lives in `python/carnot/phase3/boltzmann_gpt.py`.
- **Phase 5-D v2 intermediate-scale derisking:** specified by REQ-KONA-023;
  implementation lives in `python/carnot/phase5/intermediate_scale_v2.py`.
- **NRGPT frozen-prefix monotonicity diagnostic:** specified by REQ-KONA-021;
  implementation lives in `python/carnot/phase3/nrgpt_energy.py` and
  `scripts/experiment_1239_nrgpt_frozen_prefix_evaluation.py`.

First concrete next experiment:
`experiment_XXX_rdt_primitive_convergence.py` — implement the RDT scaffold and
the LTI-constrained injection, verify SCENARIO-KONA-001 (fixed-point convergence
on synthetic landscape) and SCENARIO-KONA-002 (LTI constraint holds). Expected
to be a 1-week effort; would not require GPU beyond what's already available.
