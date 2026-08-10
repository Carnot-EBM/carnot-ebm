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
`diagnostics_partial_pipeline_not_found`, `pipeline_not_found_blocked`,
`feasibility_step_viable`, `feasibility_step_not_viable`,
`anchored_dual_path_repair_viable`,
`anchored_dual_path_repair_not_viable`, or `blocked_*` for dependency failures.

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

### REQ-KONA-009: External EBT/Kona Parity Claim Boundary Audit

Before Carnot uses EBT citation-neighborhood language, Kona-style EBM reasoning
language, Extropic/THRML positioning, or external dependency language in PRD- or
publication-facing material, it MUST emit a local audit artifact mapping those
external claims to reproducible repository evidence and explicit parity gaps.

The audit artifact MUST include `status`, `ebt_citation_themes`,
`kona_public_claims_mapped`, `carnot_local_evidence`, `parity_gaps`,
`phase3_obligations`, `publication_claim_changes_needed`,
`external_dependency_claim_allowed`, and `honest_verdict`.
`external_dependency_claim_allowed` MUST be `false` unless the repository
contains reproducible local evidence for the external claim being considered.

**Rationale:** Phase 3 and Kona-adjacent language is high-risk because external
public positioning can sound similar to Carnot's long-term vision. Carnot must
distinguish local verifier/certificate evidence from unproven native
Kona-style EBM reasoning, TSU execution, or EBT metacognition claims.

**Acceptance criteria:**

- The artifact maps EBT reasoning, NRGPT, EBT-Policy or optimizer variants, and
  metacognitive code-generation claims to concrete Carnot obligations.
- The artifact maps public Kona-style claims to current Carnot local evidence
  and names every missing parity gate without promising the gap is solved.
- The artifact disallows external dependency claims when THRML/Kona/Extropic
  evidence is unavailable, simulated only, or not locally reproducible.

### SCENARIO-KONA-009: External Claims Stay Outside Headline Language

**Given** Carnot has local verifier and hardware-accounting artifacts but lacks
reproducible native Kona-style EBM reasoning or external hardware execution
evidence
**When** the external parity gap audit is built
**Then** `external_dependency_claim_allowed == false`
AND `publication_claim_changes_needed` instructs PRD/publication language to
claim local verifier evidence only, not EBT/Kona/Extropic parity.

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

### REQ-KONA-025: Phase 5-D Intermediate-Scale v3 Core Gates at d=128

The Phase 5-D v3 run is the dual-GPU, d=128 intermediate-scale prototype
for exp1260. It MUST represent the 100-300M-parameter scale class without
materialising a full checkpoint in unit tests, require two visible RTX 3090
devices for the experiment run, apply PPSEBM-style anti-forgetting replay
by mixing 10% replay samples into each minibatch, and measure the four core
Phase 5-D gates needed before 1B+ substrate scale-up.

The measured v3 gates are:

- `mode_collapse_absent`: mean verifier acceptance entropy after training is
  greater than 0.5 bits.
- `mcmc_mixing_acceptable`: the integrated autocorrelation time proxy after
  1000 d=128 Gibbs steps is less than 10x its baseline.
- `k_eff_maintained`: post-training effective verifier count stays within
  10% of its pre-training value.
- `forgetting_rate_acceptable`: held-out AUROC drops by less than 5% after
  in-situ training with replay.

**Acceptance criteria:**

- `python/carnot/phase5/intermediate_scale_v3.py` exposes the Phase 5-D v3
  configuration, gate measurement helpers, artifact builder, and JSON writer.
- `results/experiment_1260_phase5d_intermediate_scale_v3.json` exists with
  fields:
  - `phase5d_gates_passed: int` in [0, 4].
  - `gate_results: dict` with exactly `mode_collapse_absent`,
    `mcmc_mixing_acceptable`, `k_eff_maintained`, and
    `forgetting_rate_acceptable`.
  - `gate_values: dict` with the numeric measurements used to derive every
    gate boolean.
  - `d_hidden: int` equal to 128.
  - `scale_class: str` containing `100-300M params at d=128`.
  - `ppsebm_replay_buffer: bool` equal to `true`.
  - `ppsebm_replay_fraction: float` equal to 0.10.
  - `dual_gpu_required: bool` equal to `true`.
  - `honest_verdict: str` formatted as `phase5d_N_of_4_gates_passed`.

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

### REQ-KONA-024: NRGPT Frozen-Prefix Evaluation v2 Artifact

The Exp 1251 NRGPT frozen-prefix evaluation v2 workflow MUST read the Exp 1163
NRGPT energy-recurrence artifact, characterize the observed non-monotonicity for
paper-v6 Section 4 framing, and write
`results/experiment_1251_nrgpt_frozen_prefix_evaluation_v2.json`.

The artifact MUST include `source_experiment`, `nrgpt_auroc`,
`nonmonotonicity_classification`, `nonmonotonicity_rationale`,
`paper_v6_framing`, `nonmonotonicity_characterized`, and `honest_verdict`.
`nonmonotonicity_classification` MUST be either `b_causal_context_shift` or
`c_non_conservative_preconditioner`. The honest verdict MUST be formatted as
`nrgpt_nonmonotonicity_characterized_type_X`, where `X` is the selected
classification.

**Rationale:** Exp 1163 reported strong NRGPT AUROC while also showing that the
energy recurrence was not monotone. Exp 1251 is a pure artifact-analysis step
that records whether the paper should frame the observation as ordinary
causal-context energy-landscape shift or as evidence for a learned
path-dependent non-conservative preconditioner.

**Acceptance criteria:**

- `python/carnot/phase3/nrgpt_frozen_prefix_v2.py` exposes a deterministic
  builder for the Exp 1251 artifact and a writer for the required results JSON.
- Running the workflow writes the required artifact fields and marks
  `nonmonotonicity_characterized == true` without retraining NRGPT.

### REQ-KONA-026: Q11 TSS Sign-Bottleneck Diagnostic

The Phase 3 `ContinuousEBM` MUST expose `tss_diagnose(examples)`, where
`examples` is a list of `(question, response, is_correct)` triples. The
diagnostic measures the correlation between a scalar SC-Energy proxy computed at
the `sign(z)` bottleneck and a Z3/ground-truth correctness label, then reports
the Q11 transversal attack summary.

The returned artifact MUST include `sc_energy_z3_correlation`,
`optimal_transversal_k`, `tss_vulnerability_score`, `tss_instrumented`,
`sign_z_bottleneck_diagnosed`, `ste_pipeline_risk`, and `honest_verdict`.
`optimal_transversal_k` MUST be `2`. `tss_vulnerability_score` MUST be
`1.0 - abs(sc_energy_z3_correlation)`, clamped to `[0.0, 1.0]`.
`honest_verdict` MUST be formatted as
`tss_instrumented_corr_X.XXX_vuln_X.XXX` using the unrounded measured values.

**Rationale:** Q11 identified a transversal spectral synthesis failure mode:
an attacker can combine SC-Energy and Z3 with a straight-through or
Gumbel-Softmax rewrite to bypass the sign firewall. Monitoring SC-Energy/Z3
correlation directly at `sign(z)` is the Phase 3 metric that makes this failure
mode visible.

**Acceptance criteria:**

- `ContinuousEBM.tss_diagnose(...)` accepts at least 20 FoVer triples and returns
  finite JSON-serialisable scalar fields for the Q11 diagnostic.
- Low absolute SC-Energy/Z3 correlation produces a high
  `tss_vulnerability_score` and sets `ste_pipeline_risk` when the score exceeds
  `0.6`.
- `results/experiment_1264_q11_tss_instrumentation_v2.json` records the same
  required fields for the first 20 rows from `results/fover_corpus_v5.json`.

### REQ-KONA-027: FSNet-Style Feasibility Step for Continuous EBM States

The Phase 3 `ContinuousEBM` substrate MUST expose a deterministic
FSNet-style feasibility step for bounded continuous latent states. The step
minimizes a verifier-style violation energy separately from the EBM task
energy by using linear inequality constraints `A @ z + b <= 0`, while a small
anchor term limits state distortion from the input latent.

The Exp 1275 artifact MUST compare raw Langevin states against Langevin states
repaired by the feasibility step on deterministic synthetic/FoVer-like latent
states. It MUST report final task energy, violation count, convergence steps,
state distortion, and diversity for both arms where applicable. The artifact
MUST include `feasibility_delta_overall`, `energy_delta`, `violation_delta`,
`distortion_mean`, `feasibility_step_viable`, and `honest_verdict`.

**Rationale:** FSNet makes feasibility seeking an explicit operator rather than
an incidental side effect of task-energy descent. Carnot needs the same
separation before Phase 3 can claim an internal repair primitive rather than a
post-hoc energy measurement.

**Acceptance criteria:**

- `python/carnot/phase3/continuous_ebm.py` exposes `feasibility_step(...)`, which
  accepts a latent state plus linear violation constraints and returns the
  repaired state, final violation energy, violation count, convergence steps,
  distortion from the input state, and convergence status.
- Focused tests verify that the step reduces violation energy without requiring
  task-energy gradients, preserves the latent shape and `(-1, 1)` bounds, and
  reports zero distortion for already-feasible states.
- `results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json` records
  the required fields and sets `honest_verdict` from measured deltas, without
  converting a diversity-collapsing repair into a positive result.

### REQ-KONA-028: SnareNet-Style Adaptive Repair Layer for Continuous EBM States

When Exp 1275 reports a positive feasibility delta, the Phase 3 `ContinuousEBM`
substrate MUST expose a deterministic SnareNet-style adaptive repair layer that
can be appended after the FSNet feasibility step. The layer MUST operate on
linear inequality constraints `A @ z + b <= 0`, adapt its relaxation pressure
from the measured hard-constraint slack during repair, keep states bounded in
`(-1, 1)`, and report repair diagnostics separately from task-energy sampling.

The Exp 1276 artifact MUST compare raw Langevin states, FSNet feasibility-step
states, and SnareNet-style adaptive-repair states on the same deterministic
synthetic/FoVer-like constraints. It MUST report
`final_constraint_satisfaction`, `repair_iterations`, `distortion_from_initial`,
`diversity_preserved`, `repair_delta_over_fsnet`, and an `honest_verdict` derived
strictly from measured satisfaction, distortion, and diversity deltas.

**Rationale:** SnareNet's repair layer is only useful if it improves or matches
hard-constraint satisfaction without collapsing latent diversity or moving states
farther than the FSNet baseline. Carnot needs the prototype to preserve that
distinction before treating adaptive relaxation as a Phase 3 repair primitive.

**Acceptance criteria:**

- `python/carnot/phase3/continuous_ebm.py` exposes an adaptive repair helper that
  accepts a latent state plus linear violation constraints and returns the
  repaired state, final constraint satisfaction, repair iterations, distortion
  from the input state, diversity-safe diagnostics, convergence status, and the
  final relaxation value.
- Focused tests verify that adaptive repair continues from the FSNet repair,
  improves or matches final constraint satisfaction under a stricter tolerance,
  preserves state shape and `(-1, 1)` bounds, and rejects malformed inputs and
  invalid hyperparameters.
- `results/experiment_1276_snarenet_repair_layer_gated.json` records the required
  fields, gates execution on a positive Exp 1275 feasibility delta, and sets
  `honest_verdict` without converting excessive distortion or diversity collapse
  into a positive result.

### REQ-KONA-029: HardNet++-Style Damped Local-Linear Nonlinear Repair

The Phase 3 `ContinuousEBM` substrate MUST expose a deterministic
HardNet++-style damped local-linear projection helper for nonlinear inequality
constraints `g(z) <= 0`. The helper MUST repeatedly linearise the nonlinear
constraints at the current bounded latent state, solve a damped least-squares
projection step for active violations, keep states in `(-1, 1)`, and report
violation, convergence, distortion, and verified-span reuse diagnostics.

The Exp 1291 artifact MUST compare raw Langevin states, FSNet fixed
local-linear repair, SnareNet fixed local-linear repair, and HardNet++ damped
relinearising repair on deterministic nonlinear synthetic constraints. The
synthetic constraints MUST include at least two valid basins and one misleading
local basin. The artifact MUST report final task energy, violation count,
convergence steps, state distortion, diversity, verified-span reuse,
`hardnetpp_delta_over_snarenet`, `nonlinear_repair_viable`,
`construct_refine_iterations`, `copy_as_decode_verified_span_reuse`, `status`,
and `honest_verdict`.

**Rationale:** Exp 1275 and Exp 1276 only exercise linear verifier constraints.
Nonlinear verifier geometry can make a one-shot projection look feasible under
its local surrogate while remaining invalid under the true constraint. A damped
relinearising projection is the cheap test for whether repair can preserve useful
latent state while crossing from a misleading local basin into a true valid
basin.

**Acceptance criteria:**

- `python/carnot/phase3/nonlinear_repair.py` exposes a helper for damped
  local-linear projection against callable nonlinear constraints and Jacobians.
- Focused tests verify that the helper reduces true nonlinear violation energy,
  preserves bounded latent shape, reports convergence/distortion diagnostics,
  and preserves verified-span coordinates better than an over-projecting
  baseline on a deterministic two-basin constraint.
- `results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json` records
  all required Exp 1291 fields and sets `honest_verdict` from measured
  HardNet++ versus SnareNet deltas without turning diversity collapse or
  verified-span destruction into a positive result.

### REQ-KONA-030: DSP Feasibility-Channel Repair Diagnostics

Continuous repair experiments MUST expose DSP-style local/global feasibility
channels around each candidate repair step. The local channel `phi_local`
measures residual violation pressure on the current state, while the global
channel `Phi_global` measures cohort-level residual feasibility pressure for
the same repair context. The combined channel predicts whether one more repair
step is expected to reduce hard violation energy/count rather than only add
state distortion.

The Exp 1292 artifact MUST replay the Exp 1275 FSNet, Exp 1276 SnareNet, and,
when present, Exp 1291 HardNet++ nonlinear repair cases. It MUST report
`phi_local`, `Phi_global`, `feasibility_channel_auc`,
`repair_help_prediction_accuracy`, `false_continue_rate`, `false_stop_rate`,
`distortion_when_wrong`, `feasibility_channel_predictive`,
`recommended_repair_stop_policy`, `status`, and `honest_verdict`.

**Rationale:** Continuous repair should know when additional refinement is useful
and when it is just moving a valid latent farther from the sampled state. A
separate local/global feasibility channel makes that decision auditable and
keeps nonlinear geometry failures from being hidden inside an average repair
score.

**Acceptance criteria:**

- `python/carnot/phase3/continuous_ebm.py` exposes helper functions that compute
  finite `phi_local`/`Phi_global` repair-step diagnostics and binary predictive
  metrics from deterministic repair rows.
- Focused tests verify that residual hard violations produce a continue signal,
  zero residual hard violations produce a stop signal, and AUROC/accuracy/false
  continue/false stop metrics are computed without requiring scikit-learn.
- `results/experiment_1292_dsp_feasibility_channel_diagnostic.json` records all
  required Exp 1292 fields, uses run date `20260504`, and sets
  `honest_verdict` from measured predictive quality without converting nonlinear
  false-continue cases into a fully positive result.

### REQ-KONA-031: HardNet++/DSP Feasibility Stop Policy

Continuous repair experiments MUST expose a conservative stop/continue policy
that combines Exp 1291 HardNet++ nonlinear repair viability with Exp 1292
DSP feasibility-channel diagnostics. The policy MUST continue repair only when
hard violations remain, the feasibility-channel score exceeds the configured
threshold, and the candidate action family is not a known nonlinear local-linear
residual case. It MUST stop after hard feasibility is reached or when marginal
local-linear nonlinear repair gains should be routed to HardNet++/bounded
abstraction instead of spending another local-linear step.

The Exp 1305 artifact MUST replay the deterministic Exp 1292 candidate
transitions, inherit `hardnetpp_delta_over_snarenet` from Exp 1291 and
`feasibility_channel_auc` from Exp 1292, report `stop_policy_precision`, list
`residual_nonlinear_cases`, include a KAN/PWA abstraction note grounded in
arXiv 2602.06737, use run date `20260505`, and set `honest_verdict` without
claiming that a conservative replay policy is a learned general stop rule.

**Rationale:** Exp 1292 showed a useful but marginal feasibility channel with
many nonlinear false-continue cases. A stop policy must preserve the useful
linear and HardNet++ continue decisions while blocking residual nonlinear
local-linear loops that only add distortion.

**Acceptance criteria:**

- `python/carnot/phase3/feasibility_stop_policy.py` exposes a deterministic
  helper that classifies DSP replay rows into conservative stop/continue
  recommendations and reports precision from measured labels.
- Focused tests verify that hard feasibility stops, high-channel helpful repair
  continues, nonlinear local-linear residual cases stop, and malformed replay
  inputs are rejected.
- `results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json` records
  `status`, `feasibility_stop_policy_written`,
  `hardnetpp_delta_over_snarenet`, `feasibility_channel_auc`,
  `stop_policy_precision`, `residual_nonlinear_cases`,
  `kan_pwa_abstraction_note`, and `honest_verdict`.

### REQ-KONA-032: Held-Out Learned HardNet++/DSP Stop Policy

Continuous repair experiments that claim a learned stop/continue policy MUST
train on one deterministic split of existing HardNet++/DSP replay rows and
evaluate on a held-out split. The policy MUST use only pre-decision replay
features, including DSP feasibility-channel scores, hard violation pressure, and
transparent repair-family features derived from HardNet++ replay cohorts. Labels
MUST come from verifier-backed replay outcomes such as `repair_helped`; the
experiment MUST NOT invent stop/continue labels from the artifact narrative.

The Exp 1318 artifact MUST load Exp 1305 plus the Exp 1291/1292 replay sources,
write `results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json` with run
date `20260505`, report the deterministic `generalization_split`,
`stop_policy_precision`, `stop_policy_recall`, `dsp_feasibility_auc`,
`hardnetpp_delta_over_replay_policy`, `learned_stop_policy_written`, `status`,
and `honest_verdict`, and compare the learned held-out policy against the Exp
1305 conservative replay policy on the same held-out rows.

**Rationale:** Exp 1305 was useful as an operator gate but explicitly was not a
learned general stop rule. A held-out replay evaluation is the smallest honest
next test: it can show whether a transparent learned policy reproduces the useful
HardNet++/DSP decisions on unseen seeds while still refusing to overclaim broad
generalisation beyond the replay distribution.

**Acceptance criteria:**

- `python/carnot/phase3/learned_stop_policy.py` exposes deterministic helpers to
  split replay rows, fit a transparent stop/continue policy from verifier-backed
  labels, and evaluate held-out precision/recall against a baseline policy.
- Focused tests verify that the split is deterministic, labels are drawn from
  replay validator outcomes, the learned policy handles held-out rows, and the
  artifact contains all required Exp 1318 fields.
- `results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json` records
  `status`, `learned_stop_policy_written`, `generalization_split`,
  `stop_policy_precision`, `stop_policy_recall`,
  `hardnetpp_delta_over_replay_policy`, `dsp_feasibility_auc`, and an
  appropriately bounded `honest_verdict`.

### REQ-KONA-033: EBRM Latent Trajectory Drift Smoke Diagnostic

Before scaling continuous latent repair around EBRM-style trajectory planning,
Carnot MUST run a CPU-only smoke diagnostic that measures the arXiv 2603.28248
failure mode where latent planning lowers energy while decoded task accuracy
falls because the optimized latent leaves the decoder's support. The diagnostic
MUST use a tiny deterministic CNF or graph-style task, expose encoder output
`h_x`, an optimized latent `z`, simple decomposed energy terms, a decoder with an
explicit support radius, direct decode, and planned decode on the same dataset.

The Exp 1417 artifact MUST use run date `20260506` and write
`results/experiment_1417_ebrm_latent_trajectory_drift_smoke.json` with
`status`, `latent_drift_smoke_complete`, `task_family`, `energy_monotone`,
`accuracy_before_planning`, `accuracy_after_planning`,
`accuracy_delta_after_planning`, `latent_drift_norm`,
`dual_path_decoder_required`, `anchoring_required`, and `honest_verdict`.
`dual_path_decoder_required` MUST be true when planning lowers energy but hurts
or fails to improve decoded accuracy. `anchoring_required` MUST be true when the
planned latent drift exceeds the decoder support radius.

**Rationale:** Phase 3/Kona planning must not treat lower continuous latent
energy as sufficient evidence of better decoded reasoning. A small deterministic
support-shift smoke test makes the dangerous case visible before expensive
continuous repair training scales it up.

**Acceptance criteria:**

- `python/carnot/phase3/latent_drift_smoke.py` exposes deterministic helpers to
  build the tiny CNF task family, encode inputs, decode supported latents, plan
  latent trajectories, compute accuracy, and build the Exp 1417 artifact.
- Focused tests verify that direct decode and planned decode use the same
  dataset, planning energy is monotone when reported as monotone, latent drift is
  measured from `h_x` to `z_T`, and the dual-path/anchoring booleans follow the
  measured energy, accuracy, and support-radius gates.
- `results/experiment_1417_ebrm_latent_trajectory_drift_smoke.json` records all
  required fields with `status="complete"` and an honest verdict that does not
  convert off-support decoded regression into a positive Phase 3 claim.

### REQ-KONA-034: Anchored Dual-Path Latent Repair Smoke

After Exp 1417 detects monotone-energy latent drift that hurts decoded
accuracy, Carnot MUST provide a deterministic CPU-only follow-up smoke test that
compares raw latent energy descent against an anchored descent path with a
dual-path decoder gate. The benchmark MUST reuse the smallest Exp
1417-compatible deterministic CNF task family or an equivalent deterministic
micro benchmark with encoder anchors, support-limited decoding, direct decode,
raw planned decode, and anchored planned decode.

The anchored path MUST apply an explicit anchor to the initial encoder latent
state `h_x` and MUST compare decoded candidate quality before accepting a
lower-energy latent state. The dual-path gate may be a deterministic stub for
this smoke test, but it MUST reject candidate steps whose decoded task quality
falls below the current accepted decode quality.

The Exp 1436 artifact MUST use run date `20260506` and write
`results/experiment_1436_anchored_dual_path_latent_repair_v1.json` with
`status="in_progress"` before running the benchmark, then finish with
`status="complete"` or `status="blocked"` and the fields `anchoring_applied`,
`dual_path_decoder_stub`, `energy_monotone`, `accuracy_before_planning`,
`accuracy_after_planning`, `accuracy_delta_after_planning`, `latent_drift_norm`,
`off_support_rate`, `anchored_repair_viable`, and `honest_verdict`.
`anchored_repair_viable` MUST be true only when anchored planning does not reduce
decoded accuracy and anchored off-support drift is lower than raw descent.

**Rationale:** Exp 1417 showed that lower latent energy can be a misleading
objective when the latent leaves decoder support. Exp 1436 tests the smallest
guardrail that could make latent repair admissible again: remain close to the
encoded state and use decoded quality as an acceptance check rather than trusting
latent energy alone.

**Acceptance criteria:**

- `python/carnot/phase3/anchored_dual_path_latent_repair.py` exposes
  deterministic helpers to run raw descent and anchored dual-path descent on the
  same Exp 1417-compatible benchmark.
- Focused tests verify that anchored descent applies an anchor to `h_x`, the
  dual-path gate rejects lower-energy candidates that reduce decoded quality,
  raw and anchored metrics are measured on the same tasks, and
  `anchored_repair_viable` follows the accuracy/off-support gate exactly.
- `results/experiment_1436_anchored_dual_path_latent_repair_v1.json` records all
  required fields with an honest verdict that distinguishes a smoke-test repair
  from a scaled Phase 3 training claim.

### REQ-KONA-035: EBT/NRGPT Local Energy-Convergence Microprototype Audit

Carnot MUST provide a deterministic CPU-only Exp 1450 microprototype that
compares an EBT/NRGPT-style iterative energy-convergence baseline against the
Exp 1436 anchored latent repair reference before any larger Phase-3 scale-up.
The microprototype MUST use existing local Carnot trace and energy abstractions
rather than live model inference or a new external dependency.

The Exp 1450 workflow MUST write
`results/experiment_1450_ebt_nrgpt_local_microprototype_audit.json` with
`status="in_progress"` before evaluation, then finish with `status="complete"`
or `status="blocked"` and the fields `status`,
`energy_convergence_probe_complete`, `traces_evaluated`,
`baseline_energy_delta`, `anchored_repair_energy_delta_reference`,
`convergence_steps_median`, `scale_recommendation`, `commands_run`, and
`honest_verdict`.

`scale_recommendation` MUST be one of `retire`, `keep_smoke_only`, or
`scale_future_milestone`. The recommendation MUST be derived from measured
trace count, energy delta, convergence steps, and the anchored repair reference,
and MUST NOT present lower baseline energy alone as a decoded-accuracy or
Phase-3 scale claim.

**Rationale:** Exp 1436 showed anchored latent repair can reduce energy without
accuracy regression at smoke-test scale. Exp 1450 adds a minimal explicit
EBT/NRGPT-style convergence comparator so Carnot can distinguish a useful
"think until energy flattens" signal from an energy-only baseline that still
lacks decoded quality evidence.

**Acceptance criteria:**

- `python/carnot/phase3/ebt_nrgpt_local_microprototype_audit.py` exposes
  deterministic helpers that load a tiny local trace sample, run iterative
  energy minimization, compute median convergence steps, and read the Exp 1436
  anchored energy-delta reference.
- Focused tests verify that the baseline energy trace converges, the artifact
  contains every required field, and the scale recommendation follows the
  measured smoke gate rather than assuming energy reduction is enough to scale.
- `results/experiment_1450_ebt_nrgpt_local_microprototype_audit.json` records
  all required fields with an honest verdict that distinguishes smoke-only
  evidence from a future milestone scale recommendation.

### REQ-KONA-036: Kona/EBT Partial-Trace Localization Audit

Carnot MUST provide a deterministic CPU-only Exp 1490 audit that tests a local
analog of Kona-style partial-trace failure localization without importing Kona
internals, depending on a Kona service, or claiming decoded quality. The audit
MUST reuse existing local trace telemetry when available, inject deterministic
bad spans into otherwise clean traces, and rank the injected span against clean
spans using available local energy or verifier features.

The Exp 1490 workflow MUST write
`results/experiment_1490_kona_ebt_partial_trace_localization_audit.json` with
`status="in_progress"` before evaluation, then finish with `status="complete"`
or `status="blocked"` and the fields `status`, `model_specs`,
`localization_audit_complete`, `traces_evaluated`, `injected_failures`,
`localization_top1_rate`, `localization_top3_rate`, `random_baseline_rate`,
`decoded_quality_claim_allowed`, `kona_dependency_used`, `audit_note_path`,
`tests_run`, and `honest_verdict`.

The audit MUST set `decoded_quality_claim_allowed=false` and
`kona_dependency_used=false`. Its random baseline MUST be computed from the
number of candidate spans per trace, and a superficial span-length baseline
SHOULD be recorded when span lengths are available.

**Rationale:** public Kona positioning emphasizes globally scored partial traces
and failure localization. Exp 1490 checks whether Carnot's local trace features
can localize injected failures at bounded scale while keeping the result framed
as a diagnostic, not as evidence of Kona parity or decoded answer quality.

**Acceptance criteria:**

- `python/carnot/phase3/kona_partial_trace_localization_audit.py` exposes
  deterministic helpers that load bounded Exp 1480-style telemetry rows, inject
  known bad spans, rank spans by local energy/verifier features, and compute
  top-1, top-3, random, and span-length baseline rates.
- Focused tests verify that the injected bad span is ranked above clean spans
  for deterministic telemetry, the random baseline is derived from span count,
  boundary flags forbid decoded-quality and Kona-dependency claims, and the
  artifact contains every required field.
- `results/experiment_1490_kona_ebt_partial_trace_localization_audit.json`
  records all required fields with an honest verdict that reports whether the
  bounded injected-span localization diagnostic beat random without claiming
  live decoded quality.

### REQ-KONA-037: Pi-Net-Style Continuous Constraint Projection Layer

Carnot MUST provide a deterministic CPU-only Exp 1633 prototype of a Pi-Net-style
projection layer for the continuous constraint tier. The layer MUST be implemented
with JAX tensor operations, accept a continuous latent state plus linear equality
constraints `A_eq @ z = b_eq` and inequality constraints `A_ineq @ z <= b_ineq`,
and project infeasible states into the hard-constraint set without training a
neural model or requiring GPU hardware.

The Exp 1633 workflow MUST write `results/experiment_1633_pinet.json` with
`status`, `schema`, `experiment_id`, `spec_refs`, `projection_error`,
`convergence_steps`, `cases_evaluated`, `differentiable_projection`, and
`honest_verdict`. `projection_error` MUST be the maximum residual over equality
violations and positive inequality violations after projection.

**Rationale:** Existing Phase 3 feasibility repair prototypes reduce violation
energy, but Pi-Net-style hard projection is the sharper contract for a continuous
constraint tier: the next layer should receive a latent that satisfies the
declared constraints, not merely a lower-energy latent.

**Acceptance criteria:**

- `scripts/experiment_1633_pinet.py` exposes a JAX-based projection layer with
  deterministic validation and projection diagnostics.
- Focused tests verify that infeasible toy continuous states are projected to
  satisfy all declared hard constraints, the projection is differentiable through
  JAX autodiff for the continuous state, malformed constraints are rejected, and
  the artifact contains `projection_error` and `convergence_steps`.
- `results/experiment_1633_pinet.json` records all required fields with an
  honest verdict derived from measured residuals and convergence.

### REQ-KONA-038: Pi-Net vs T-SKM Comparison on CCTU Constraints

Carnot MUST provide an Exp 1634 comparison script evaluating the Pi-Net-style continuous projection layer against the prior T-SKM approach on CCTU constraints.

The Exp 1634 workflow MUST write `results/experiment_1634_pinet_vs_tskm.json` with `status`, `schema`, `experiment_id`, `spec_refs`, `pinet_faster_than_tskm`, `latency_diff`, and `honest_verdict`.

**Acceptance criteria:**

- `scripts/experiment_1634_comparison.py` exists and measures performance of Pi-Net versus T-SKM.
- Focused tests verify the comparison.
- `results/experiment_1634_pinet_vs_tskm.json` records the required fields.

### REQ-KONA-039: Pi-Net Douglas-Rachford Model Layer

Carnot MUST provide `python/carnot/models/pinet_layer.py` as a reusable,
CPU-safe JAX projection layer for feasible-by-design continuous latents. The
layer MUST accept linear equality constraints `A_eq @ z = b_eq` and inequality
constraints `A_ineq @ z <= b_ineq`, validate malformed shapes before
projection, and use a bounded Douglas-Rachford-style iteration with closed-form
affine and half-space projections.

The Exp 1662 workflow MUST write `results/experiment_1662_pinet_layer.json`
with `status`, `schema`, `experiment_id`, `spec_refs`, `module_path`,
`projection_error`, `convergence_steps`, `differentiable_projection`, and
`honest_verdict`. A complete artifact MUST report a final projection error at
or below the configured tolerance and MUST only set `differentiable_projection`
when JAX autodiff produces finite gradients through the projected state.

**Acceptance criteria:**

- `python/carnot/models/pinet_layer.py` exposes a Douglas-Rachford Pi-Net
  projection layer and validated linear constraint-set dataclass.
- Focused tests verify that infeasible toy states are projected into the hard
  constraint set, feasible states remain unchanged within tolerance, malformed
  constraints are rejected, gradients through `project_vector` are finite, and
  the Exp 1662 artifact contains all required fields.
- `results/experiment_1662_pinet_layer.json` records a complete status with an
  honest verdict derived from measured projection residuals and differentiable
  projection checks.

### REQ-KONA-040: TRM vs AR Length Generalization Falsification

Carnot MUST provide an Exp 3822 experiment script evaluating whether a TRM architecture generalizes to longer held-out lengths on a 1D task compared to a matched-compute AR baseline.

The Exp 3822 workflow MUST write `results/experiment_3822_trm_escapes_grids_p1.json` with required schema fields.

**Acceptance criteria:**
- `scripts/experiments/experiment_3822_trm_escapes_grids_p1.py` exists and evaluates TRM vs AR length generalization.
- Focused tests verify the behavior.
- `results/experiment_3822_trm_escapes_grids_p1.json` records all required fields and an honest verdict.

### REQ-KONA-6287: Bounded ASP Continuous Relaxation Bridge

Carnot SHALL provide a bounded continuous relaxation for the exact Exp6274 ASP
energy. The relaxation SHALL use the multilinear extension of the finite
discrete energy over atom probabilities in `[0, 1]`. The claim boundary SHALL
state that this proves equality on binary vertices only. It SHALL NOT claim a
learned Kona model, a learned verifier, or a diffusion language model.

The Exp6287 workflow SHALL read the trusted Exp6274 compiler artifact and
fixture manifest. It SHALL freeze their hashes, the relaxation source hashes,
the atom and vertex bounds, the finite-difference tolerance, the optimizer
budgets, the random seeds, and protected file hashes before it evaluates
fixtures. It SHALL reject fixtures outside the preregistered bounded state
limit.

The workflow SHALL enumerate every binary vertex for each eligible fixture. At
every vertex, the continuous relaxation energy SHALL equal the Exp6274 discrete
energy decomposition. The workflow SHALL compute analytic gradients of the
multilinear extension and compare them against central finite differences away
from the box boundary.

The workflow SHALL run blank, random, and partial-state starts with fixed step
and restart budgets. It SHALL report refinement success, final energy,
rounding energy, restart count, and exact enumeration work separately. It SHALL
also report fractional stationary points, rounding failures, Clingo oracle
controls, cold exact enumeration controls, unsupported size controls, and
unsupported syntax controls.

The terminal artifact SHALL be
`results/experiment_6287_asp_continuous_relaxation.json`. It SHALL include
`status`, `upstream_compiler_path_hash_and_terminal_class`,
`relaxation_definition_and_claim_boundary`, `source_paths_and_hashes`,
`eligible_fixture_manifest_path_and_hash`, `fixture_count`,
`atom_count_and_vertex_count_by_fixture`, `exact_vertex_energy_parity_by_fixture`,
`parity_failure_count`, `analytic_gradient_definition`,
`finite_difference_gradient_checks`, `refinement_optimizer_and_fixed_budgets`,
`blank_random_and_partial_start_manifest`,
`refinement_outcomes_by_start_fixture_and_seed`,
`fractional_stationary_points_by_fixture`, `rounding_failures_by_fixture`,
`exact_completion_controls`, `cold_start_controls`,
`unsupported_size_and_syntax_controls`, `oracle_claim_boundary`,
`asp_continuous_relaxation_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`duration_s`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `parity_failure_count` SHALL be a bare integer. Readiness
SHALL require exact vertex parity and passing gradient checks. Readiness SHALL
NOT require a positive refinement result.

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

### SCENARIO-KONA-025: Exp 1260 Writes Phase 5-D v3 d=128 Four-Core-Gate Artifact

**Given** two RTX 3090 GPUs are visible, the Phase 5-A/B artifacts exist, and
the exp1260 d=128 intermediate-scale prototype is configured for 10% PPSEBM
replay mixing
**When** Exp 1260 measures mode-collapse entropy, MCMC integrated
autocorrelation, k_eff retention, and held-out AUROC forgetting
**Then** it writes
`results/experiment_1260_phase5d_intermediate_scale_v3.json` with all
REQ-KONA-025 fields, exactly four measured gate booleans, the numeric values
that derive those booleans, and an honest verdict string derived strictly from
the number of passing gates.

**Spec traces:** REQ-KONA-025

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

### SCENARIO-KONA-024: Exp 1251 Classifies NRGPT Non-Monotonicity For Paper v6

**Given** the Exp 1163 NRGPT energy-recurrence source artifact exists
**When** Exp 1251 builds the frozen-prefix evaluation v2 artifact
**Then** it writes
`results/experiment_1251_nrgpt_frozen_prefix_evaluation_v2.json`, records
`nrgpt_auroc == 0.921`, classifies the non-monotonicity as either
`b_causal_context_shift` or `c_non_conservative_preconditioner`, marks
`nonmonotonicity_characterized == true`, and emits the corresponding honest
verdict.

**Spec traces:** REQ-KONA-024

### SCENARIO-KONA-026: Exp 1264 Measures Q11 TSS Correlation On FoVer

**Given** `results/fover_corpus_v5.json` contains labeled FoVer question/response
pairs
**When** the first 20 pairs are passed to `ContinuousEBM.tss_diagnose(...)`
**Then** the diagnostic returns `optimal_transversal_k == 2`, finite
`sc_energy_z3_correlation` and `tss_vulnerability_score` values, marks
`tss_instrumented == true`, and emits an honest verdict formatted as
`tss_instrumented_corr_X.XXX_vuln_X.XXX`.

**Spec traces:** REQ-KONA-026

### SCENARIO-KONA-027: Exp 1275 Compares Raw Langevin With Feasibility Repair

**Given** a deterministic ContinuousEBM and a deterministic set of FoVer-like
linear verifier constraints over its latent state
**When** Exp 1275 runs raw Langevin trajectories and applies
`feasibility_step(...)` to the same final states
**Then** it writes
`results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json` with
`feasibility_delta_overall`, `energy_delta`, `violation_delta`,
`distortion_mean`, `feasibility_step_viable`, `honest_verdict`, and per-arm
energy, violation, convergence, distortion, and diversity measurements derived
from the same deterministic states.

**Spec traces:** REQ-KONA-027

### SCENARIO-KONA-028: Exp 1276 Measures SnareNet Adaptive Repair After FSNet

**Given** `results/experiment_1275_fsnet_feasibility_step_continuous_ebm.json`
has a positive `feasibility_delta_overall` and marks the FSNet feasibility step
viable
**When** Exp 1276 applies raw Langevin sampling, FSNet feasibility repair, and
the SnareNet-style adaptive repair layer to the same deterministic states
**Then** it writes
`results/experiment_1276_snarenet_repair_layer_gated.json` with
`final_constraint_satisfaction`, `repair_iterations`, `distortion_from_initial`,
`diversity_preserved`, `repair_delta_over_fsnet`, `honest_verdict`, and per-arm
diagnostics derived from the same deterministic states.

**Spec traces:** REQ-KONA-028

### SCENARIO-KONA-029: Exp 1291 Measures HardNet++ Nonlinear Repair

**Given** a deterministic ContinuousEBM, nonlinear synthetic constraints with at
least two valid basins and one misleading local basin, and raw Langevin states
that can settle in the misleading basin
**When** Exp 1291 applies raw Langevin sampling, FSNet fixed local-linear repair,
SnareNet fixed local-linear repair, and HardNet++ damped relinearising repair to
the same states
**Then** it writes
`results/experiment_1291_hardnetpp_nonlinear_repair_benchmark.json` with
`hardnetpp_delta_over_snarenet`, `nonlinear_repair_viable`,
`construct_refine_iterations`, `copy_as_decode_verified_span_reuse`, `status`,
`honest_verdict`, and per-arm energy, violation, convergence, distortion,
diversity, and verified-span reuse measurements derived from the same
deterministic nonlinear benchmark.

**Spec traces:** REQ-KONA-029

### SCENARIO-KONA-030: Exp 1292 Predicts Useful Additional Repair Steps

**Given** completed Exp 1275 and Exp 1276 linear repair artifacts and an optional
completed Exp 1291 nonlinear repair artifact
**When** Exp 1292 constructs candidate before/after repair transitions from the
same per-seed rows
**Then** it writes
`results/experiment_1292_dsp_feasibility_channel_diagnostic.json` with finite
`phi_local`, `Phi_global`, `feasibility_channel_auc`,
`repair_help_prediction_accuracy`, `false_continue_rate`, `false_stop_rate`,
`distortion_when_wrong`, `feasibility_channel_predictive`,
`recommended_repair_stop_policy`, `status`, and `honest_verdict` fields derived
from those deterministic transitions.

**Spec traces:** REQ-KONA-030

### SCENARIO-KONA-031: Exp 1305 Stops Marginal Continuous Repair

**Given** completed Exp 1291 HardNet++ nonlinear repair metrics, completed
Exp 1292 DSP feasibility-channel rows, and the local continuous repair helpers
**When** Exp 1305 replays each candidate repair transition with the conservative
stop/continue policy
**Then** it writes
`results/experiment_1305_hardnetpp_dsp_feasibility_stop_policy.json` with
`feasibility_stop_policy_written`, `hardnetpp_delta_over_snarenet`,
`feasibility_channel_auc`, `stop_policy_precision`,
`residual_nonlinear_cases`, `kan_pwa_abstraction_note`, `status`, and
`honest_verdict` fields derived from the deterministic replay and KAN/PWA
abstraction reference.

**Spec traces:** REQ-KONA-031

### SCENARIO-KONA-032: Exp 1318 Evaluates Learned Stop Policy On Held-Out Replay

**Given** completed Exp 1305 stop-policy replay metadata and completed Exp
1291/1292 HardNet++/DSP replay artifacts
**When** Exp 1318 fits a transparent learned policy on the deterministic training
split and evaluates the policy on held-out replay rows
**Then** it writes
`results/experiment_1318_hardnetpp_dsp_learned_stop_policy.json` with
`learned_stop_policy_written`, `generalization_split`,
`stop_policy_precision`, `stop_policy_recall`,
`hardnetpp_delta_over_replay_policy`, `dsp_feasibility_auc`, `status`, and
`honest_verdict` fields derived from verifier-backed replay labels and a
same-held-out comparison to the conservative Exp 1305 policy.

**Spec traces:** REQ-KONA-032

### SCENARIO-KONA-033: Exp 1417 Measures EBRM Latent Support Drift

**Given** a deterministic tiny CNF task family, encoder anchors `h_x`, a
support-limited decoder, and a weakly anchored EBRM-style latent planning energy
**When** Exp 1417 decodes the encoder anchors directly and decodes the same
examples after latent planning
**Then** it writes
`results/experiment_1417_ebrm_latent_trajectory_drift_smoke.json` with all
REQ-KONA-033 required fields, `energy_monotone` derived from the measured energy
trace, `accuracy_delta_after_planning` derived from the two decoded accuracies,
`latent_drift_norm` derived from `||z_T - h_x||`, and
`dual_path_decoder_required`/`anchoring_required` derived from the measured
support-shift gates.

**Spec traces:** REQ-KONA-033

### SCENARIO-KONA-034: Exp 1436 Gates Latent Repair With Anchoring And Decode Quality

**Given** the Exp 1417-compatible tiny CNF task family, raw latent descent that
can lower energy while moving off support, and an anchored descent path with a
dual-path decoder quality gate
**When** Exp 1436 compares direct decode, raw planned decode, and anchored
planned decode on the same examples
**Then** it writes
`results/experiment_1436_anchored_dual_path_latent_repair_v1.json` with all
REQ-KONA-034 required fields, derives `energy_monotone`,
`accuracy_delta_after_planning`, `latent_drift_norm`, and `off_support_rate`
from anchored accepted trajectories, and sets `anchored_repair_viable` true only
if anchored planning has nonnegative decoded-accuracy delta and lower
off-support rate than raw planning.

**Spec traces:** REQ-KONA-034

### SCENARIO-KONA-035: Exp 1450 Compares Local EBT/NRGPT Convergence Against Anchored Repair

**Given** local Carnot trace rows, the existing reasoning embedding path, a
deterministic local energy-convergence probe, and the Exp 1436 anchored repair
artifact
**When** Exp 1450 runs at smoke-test scale on the fixed run date `20260507`
**Then** it writes
`results/experiment_1450_ebt_nrgpt_local_microprototype_audit.json` with all
REQ-KONA-035 required fields, sets
`energy_convergence_probe_complete=true` only when at least one trace converges
with a measured negative baseline energy delta, records the Exp 1436 anchored
energy-delta reference, and emits `keep_smoke_only` unless the local baseline
has enough trace, convergence, and quality evidence to justify scale-up.

**Spec traces:** REQ-KONA-035

### SCENARIO-KONA-036: Exp 1490 Localizes Injected Partial-Trace Failures

**Given** bounded local Exp 1480-style telemetry rows with token logprobs,
top-k alternatives, expected answers, and deterministic adversarial wrong
answers
**When** Exp 1490 injects the wrong answer span into each clean trace on the
fixed run date `20260507`
**Then** it writes
`results/experiment_1490_kona_ebt_partial_trace_localization_audit.json` with
all REQ-KONA-036 required fields, sets
`localization_audit_complete=true` only when at least one injected failure is
evaluated, computes top-1/top-3 localization rates against a random baseline,
sets `decoded_quality_claim_allowed=false` and `kona_dependency_used=false`,
and emits a bounded diagnostic honest verdict rather than a Kona-parity or
decoded quality claim.

**Spec traces:** REQ-KONA-036

### SCENARIO-KONA-037: Exp 1633 Projects Continuous Latents Into Hard Constraints

**Given** deterministic continuous toy states with linear equality and inequality
constraints for the Phase 3 continuous tier
**When** Exp 1633 applies the JAX Pi-Net-style projection layer
**Then** it writes `results/experiment_1633_pinet.json` with all REQ-KONA-037
required fields, reports `projection_error` from the measured final hard
constraint residual, reports `convergence_steps` from the bounded projection
loop, and sets `differentiable_projection=true` only when JAX autodiff produces
a finite gradient through the projected state.

**Spec traces:** REQ-KONA-037

### SCENARIO-KONA-039: Exp 1662 Provides Reusable Pi-Net Projection Layer

**Given** deterministic linear continuous constraint systems with infeasible
and already-feasible latent states
**When** `DouglasRachfordPiNetLayer.project` and `project_vector` are applied
with a bounded iteration count
**Then** the projected state satisfies equality and inequality residuals within
tolerance, the diagnostic result reports convergence steps and final residual,
and the Exp 1662 artifact records the reusable module path and complete schema.

**Spec traces:** REQ-KONA-039

### SCENARIO-KONA-040: TRM vs AR Length Generalization Output

**Given** a trained TRM block and a matched-compute AR baseline
**When** the Exp 3822 experiment script is run
**Then** it outputs `results/experiment_3822_trm_escapes_grids_p1.json` with all REQ-KONA-040 required fields, sets `decision_class` based on length generalization metrics, and correctly confirms AR headroom.

**Spec traces:** REQ-KONA-040

### SCENARIO-KONA-6287-VERTEX-PARITY: Multilinear Extension Matches ASP Vertices

**Given** the trusted Exp6274 fixtures that fit the preregistered atom and
vertex bounds
**When** Exp6287 builds the multilinear relaxation and enumerates every binary
vertex
**Then** each continuous vertex energy equals the Exp6274 discrete energy
decomposition for that vertex.

**Spec traces:** REQ-KONA-6287

### SCENARIO-KONA-6287-GRADIENT-CHECK: Analytic Gradients Match Finite Differences

**Given** an eligible fixture and a probability vector away from the box
boundary
**When** Exp6287 evaluates the analytic gradient and central finite-difference
gradient
**Then** the maximum absolute error is within the preregistered tolerance.

**Spec traces:** REQ-KONA-6287

### SCENARIO-KONA-6287-CONTROLS: Refinement And Oracle Controls Stay Separate

**Given** blank, random, and partial-state starts under fixed optimizer budgets
**When** Exp6287 refines each start and rounds the final probability vector
**Then** the artifact reports refinement outcomes, rounding failures,
fractional stationary points, Clingo controls, cold exact enumeration controls,
and unsupported-input controls without using refinement success as a readiness
gate.

**Spec traces:** REQ-KONA-6287


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
- **NRGPT frozen-prefix evaluation v2:** specified by REQ-KONA-024;
  implementation lives in `python/carnot/phase3/nrgpt_frozen_prefix_v2.py`.
- **Q11 TSS sign-bottleneck diagnostic:** specified by REQ-KONA-026;
  implementation lives in `python/carnot/phase3/continuous_ebm.py`.
- **FSNet-style feasibility step:** specified by REQ-KONA-027; implementation
  lives in `python/carnot/phase3/continuous_ebm.py`.
- **SnareNet-style adaptive repair:** specified by REQ-KONA-028; implementation
  lives in `python/carnot/phase3/continuous_ebm.py`.
- **HardNet++ nonlinear repair benchmark:** specified by REQ-KONA-029;
  implementation lives in `python/carnot/phase3/nonlinear_repair.py` and
  `scripts/experiment_1291_hardnetpp_nonlinear_repair_benchmark.py`.
- **EBRM latent trajectory drift smoke:** specified by REQ-KONA-033;
  implementation lives in `python/carnot/phase3/latent_drift_smoke.py`.
- **Anchored dual-path latent repair smoke:** specified by REQ-KONA-034;
  implementation lives in `python/carnot/phase3/anchored_dual_path_latent_repair.py`.
- **Pi-Net-style continuous hard projection:** specified by REQ-KONA-037;
  implementation lives in `scripts/experiment_1633_pinet.py`.

First concrete next experiment:
`experiment_XXX_rdt_primitive_convergence.py` — implement the RDT scaffold and
the LTI-constrained injection, verify SCENARIO-KONA-001 (fixed-point convergence
on synthetic landscape) and SCENARIO-KONA-002 (LTI constraint holds). Expected
to be a 1-week effort; would not require GPU beyond what's already available.


### REQ-KONA-038: Continuous Architecture Audit (Exp 2051)

The repository shall provide an audit module in `python/carnot/phase3/architecture_audit.py` that:
- Reads the preceding 11 experiment JSON artifacts from `results/`.
- Detects architectural divergence between the continuous execution results and the discrete verification mandate (PRD FR-12).
- Emits `results/experiment_2051_architecture_audit.json` containing `experiment` (int), `run_date` (str), `analyzed_tasks` (list), and `divergence_conflicts` (list).
- Provides a function `audit_continuous_execution(results_dir)` that returns a dictionary matching the artifact schema.

### REQ-KONA-040: Non-Autoregressive Reasoning Model

The repository shall provide an Energy-Based Reasoning Model (EBRM) in `python/carnot/models/kona_ebrm.py` that maps a simple logic puzzle into a continuous latent space and applies an energy function to detect inconsistencies, editing the trace via gradient descent.

### SCENARIO-KONA-040: Exp 1806 Writes Kona EBRM Artifact

**Given** a logic puzzle mapped to a continuous latent space
**When** Exp 1806 applies gradient descent to refine the entire reasoning trace simultaneously
**Then** it writes `results/experiment_1806_kona_ebrm.json` with all REQ-KONA-040 required fields.

**Spec traces:** REQ-KONA-040


### REQ-KONA-040: CLaRa-V Continuous Latent Variable Schema

Carnot MUST provide a Python dataclass schema for CLaRa-V continuous latent variables `ContinuousLatentState` and `EnergyVector`. These classes MUST integrate with existing EBM abstractions (e.g. `ContinuousEBM`) to evaluate continuous latent state energies.

The Exp 1994 artifact MUST include `schema` set to `carnot.schema.v3`, `experiment` set to `1994`, and an `honest_verdict` starting with `SUCCESS:`.

### REQ-KONA-041: Exp 1995 PiNet Projection for Continuous Latent States

Carnot MUST provide a JAX implementation mapping the CLaRa-V `ContinuousLatentState` to the Douglas-Rachford operator splitting PiNet layer.
The module `python/carnot/models/pinet_1995.py` MUST contain this logic.
The Exp 1995 artifact MUST include `schema` set to `carnot.model_layer.v1`, `experiment` set to `1995`, and an `honest_verdict` starting with a terminal prefix.
Gradient flow MUST be validated with a synthetic test.

### SCENARIO-KONA-041: PiNet Projection of CLaRa-V State

**Given** a CLaRa-V `ContinuousLatentState` with associated continuous constraints
**When** the PiNet projection layer is applied to the state
**Then** the gradients flow through the projection
**And** the artifact is written to `results/experiment_1995_pinet_projection.json` with the required fields.

**Given** a continuous latent state and an initialized `ContinuousEBM`
**When** the `evaluate_ebm_energy` method is called on the state
**Then** it returns the expected scalar energy correctly computed using the EBM's coupling and bias.

### REQ-KONA-042: Zero-Forgetting Gate for Phase 3 Policy Retention

Carnot MUST provide a strict promotion gate (`ZeroForgettingGate`) in `python/carnot/pipeline/csl_gate.py` that blocks the retention of a newly learned policy if it violates any prior constraint in the replay buffer. The gate MUST run pre/post tests on the replay buffer and only pass if no new failures are introduced.
The Exp 2058 artifact MUST include `schema` set to `carnot.csl_gate.v1`, `experiment` set to `2058`, `acceptance_gate_passed` indicating if the gate passed, and an `honest_verdict` starting with a terminal prefix.

### SCENARIO-KONA-042: Exp 2058 Evaluates Zero-Forgetting Gate

**Given** a set of pre-failures and post-failures from evaluating a replay buffer
**When** `ZeroForgettingGate.evaluate` is called
**Then** it returns True iff `post_failures` is a subset of `pre_failures` (i.e. no new failures)
**And** the artifact is written to `results/experiment_2058_csl_gate.json` with the required fields.

### REQ-KONA-071: Lagrangian Continuous Space Optimizer

Carnot MUST provide a JAX-based continuous space optimizer using a Lagrangian formulation in `python/carnot/phase3/lagrangian_optimizer.py`. The optimizer MUST enforce hard bounds as high-energy penalties and support a global Lagrangian energy function that sums local symbolic constraint potentials.
The Exp 2071 artifact MUST be written to `results/experiment_2071_lagrangian_optimizer.json` and include the field `lagrangian_ready` set to `true`.

### SCENARIO-KONA-071: Exp 2071 Evaluates Lagrangian Optimizer

**Given** a set of latent constraints and continuous bounds
**When** the Lagrangian optimizer minimizes the energy
**Then** hard bounds are enforced as high-energy penalties
**And** the artifact is written to `results/experiment_2071_lagrangian_optimizer.json` with `lagrangian_ready=true`.

### REQ-KONA-072: Lagrangian Optimizer for Hard Sudoku

Carnot MUST apply the Phase 1 Lagrangian optimizer to solve hard Sudoku constraints entirely in the continuous domain. The Sudoku rules MUST be modeled as differentiable energy penalties, and the continuous grid state MUST be optimized until energy is minimized and mapped back to digits.
The Exp 2072 artifact MUST be written to `results/experiment_2072_kona_sudoku.json` and include the field `solved_sudoku` set to `true`.

### SCENARIO-KONA-072: Exp 2072 Evaluates Sudoku Lagrangian Optimizer

**Given** a hard Sudoku puzzle modeled as differentiable energy penalties
**When** the Lagrangian optimizer minimizes the energy in the continuous domain
**Then** the continuous grid state is mapped back to valid digits
**And** the artifact is written to `results/experiment_2072_kona_sudoku.json` with `solved_sudoku=true`.

### REQ-KONA-085: PEM vs Lagrangian Sudoku Comparison

Carnot MUST provide an empirical comparison between the new PEM solver and the prior monolithic Lagrangian solver on a hard Sudoku dataset. The evaluation MUST demonstrate that the PEM solver escapes local minima better, yielding a strictly positive success rate delta.
The Exp 2085 artifact MUST be written to `results/experiment_2085_pem_sudoku_eval.json` and include the field `success_rate_delta` greater than 0.

### SCENARIO-KONA-085: Exp 2085 Evaluates PEM on Hard Sudoku

**Given** 50 Hard Sudoku instances
**When** both the PEM solver and the Lagrangian solver are applied
**Then** the PEM solver achieves a higher success rate
**And** the artifact is written to `results/experiment_2085_pem_sudoku_eval.json` with a positive `success_rate_delta`.

### REQ-KONA-2097: EqM vs PEM Comparison on Continuous Constraint Graphs

Carnot MUST provide an empirical comparison between the Equilibrium Matching (EqM) landscape formulation and the Parallel Energy Minimization (PEM) optimizer on synthetic continuous constraint graphs. The evaluation MUST measure convergence speed and constraint satisfaction rate.
The Exp 2097 artifact MUST be written to `results/experiment_2097_eqm_eval.json` and include the field `eqm_superior` as a boolean.

### SCENARIO-KONA-2097: Exp 2097 Evaluates EqM on Continuous Graphs

**Given** 50 instances of continuous constraint graphs
**When** both the EqM ULA sampler and the PEM solver are applied
**Then** convergence time and satisfaction rates are measured
**And** the artifact is written to `results/experiment_2097_eqm_eval.json` with an `eqm_superior` boolean.

### REQ-KONA-2102: EqM Parameter Memory Cache

Carnot MUST provide a memory cache for Equilibrium Matching (EqM) landscapes to save and retrieve converged parameters. The caching mechanism MUST support JSON or Safetensors serialization of EqM parameters and provide a hot-start capability for subsequent EqM evaluations on similar problems.
The Exp 2102 artifact MUST be written to `results/experiment_2102_eqm_memory.json` and include the field `memory_promotion_successful` set to `true`.

### SCENARIO-KONA-2102: Exp 2102 Evaluates EqM Memory Promotion

**Given** a converged EqM landscape for a problem
**When** the memory cache saves the parameters and hot-starts a new EqM evaluation
**Then** the parameters are successfully serialized and retrieved
**And** the artifact is written to `results/experiment_2102_eqm_memory.json` with `memory_promotion_successful=true`.

### REQ-KONA-3338: SOTA GGUF Tokenizer Runtime Receipt

Carnot MUST verify a standalone runtime receipt for SOTA GGUF models (Qwen3.6-35B-A3B, Gemma4-26B-A4B-it, Gemma4-31B-it) and ensure they load locally.
The artifact MUST be written to `results/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.json`.

### SCENARIO-KONA-3338: Exp 3338 Generates SOTA GGUF Runtime Receipt

**Given** the SOTA GGUF models are defined in the mandated list
**When** the artifact script `scripts/experiment_3338_sota_gguf_tokenizer_runtime_receipt_v1.py` is executed
**Then** it writes the receipt with REQUIRED ARTIFACT FIELDS and Phase-3 precondition RUNTIME FIELDS
**And** `runtime_receipt_clean` is true if at least one mandated model loads cleanly.

### REQ-KONA-3384: Parallel Energy Minimization (PEM) Composition

Carnot MUST provide a Parallel Energy Minimization (PEM) solver that decomposes a monolithic constraint satisfaction problem into smaller, composed subproblems (e.g., sub-graphs) and runs PEM inference by combining local energy models.
The artifact MUST be written to `results/experiment_3384_pem_composition.json` and MUST include `pem_composition_ready` set to `true`, and record metrics such as local energy improvements.

### SCENARIO-KONA-3384: Exp 3384 Evaluates PEM Composition

**Given** a synthetic modular constraint problem and local energy models for its sub-graphs
**When** the PEM solver optimizes the composed energy landscapes
**Then** the solver successfully escapes local minima by minimizing the sub-problem energies in parallel
**And** the artifact is written to `results/experiment_3384_pem_composition.json` with `pem_composition_ready=true`.



### REQ-KONA-3394: Kona Global Optimization Emulation on Hard Sudoku

Carnot MUST emulate Logical Intelligence's Kona global inference procedure on a hard Sudoku problem set using Carnot's Ising energy function. The implementation MUST treat the Sudoku board as a joint energy landscape and apply continuous sampling/optimization over the entire board at once, avoiding any autoregressive token-by-token prediction. It MUST score correctness and report the time-to-solution.
The Exp 3394 artifact MUST be written to `results/experiment_3394_kona_global_opt.json` and include the fields `solved_sudoku`, `time_to_solution`, and `honest_verdict`.

### SCENARIO-KONA-3394: Exp 3394 Emulates Kona Global Inference

**Given** a hard Sudoku problem modeled as an Ising energy function
**When** the global inference procedure continuous sampling/optimization is applied over the entire board at once
**Then** the solver successfully minimizes energy without autoregressive prediction, scores correctness, and reports time-to-solution
**And** the artifact is written to `results/experiment_3394_kona_global_opt.json` with all required fields.

### REQ-KONA-3408: Kona Global Optimization Emulation 3408
Carnot MUST emulate Logical Intelligence's Kona global inference procedure on a hard Sudoku problem set using Carnot's Ising energy function in experiment 3408.
The implementation MUST treat the Sudoku board as a joint energy landscape, apply continuous sampling/optimization over the entire board at once using MODEL_SPECS = ["unsloth/gemma-4-26B-A4B-it-GGUF"], score correctness, and report the time-to-solution vs standard autoregressive search.
The artifact MUST be written to `results/experiment_3408_kona_global_opt.json` and include the field `status` set to `success`.

### SCENARIO-KONA-3408: Exp 3408 Evaluates Kona Global Opt
**Given** a hard Sudoku problem set
**When** we run `experiment_3408_kona_global_opt.py`
**Then** it applies continuous sampling over the board at once
**And** the artifact is written to `results/experiment_3408_kona_global_opt.json` with `status=success`.

### REQ-KONA-3312: Energy-Descent-vs-Autoregressive Premise Test (P0.1)

The entire Phase 3 / Kona endgame rests on the premise that energy-descent
reasoning over continuous latents is at least competitive with — and ideally
superior to — autoregressive (AR) token sampling on a real reasoning task. That
premise MUST be tested head-to-head on a real benchmark with ground-truth labels
(GSM8K subset `n>=200` or ARC-AGI-1), not on toy synthetic puzzles. Exp 3312
runs the paired comparison and emits a falsifiable significance gate.

**Rationale:** prior evidence (exp1165/exp1210/exp1222) only measured stability
or BFS ties on synthetic 5x5 grids. A real-task, AR-compared, paired-significance
result either greenlights Phase 3 (premise validated) or honestly retires the
foundation-model endgame (premise unsupported at scale). Either outcome is
high-value; the dishonest outcome is to never run it.

**Acceptance criteria:**

- `load_gsm8k_subset` returns a deterministic `n>=200` held-out split of real
  GSM8K problems with integer ground-truth answers, keyed by `(path, n, seed)`.
- `extract_final_answer` parses the model's `#### <number>` coda (or the last
  integer fallback) into an int, returning `None` only when no number is present.
- `energy_descent_select` performs bounded-depth gradient descent on the
  continuous latent of each candidate under a trained Boltzmann-GPT energy and
  selects the minimum-energy candidate — no token sampling occurs inside the
  descent loop (REQ-KONA-001 reasoning mode).
- `mcnemar_test` and `paired_bootstrap_ci` compute a paired significance signal
  on the per-problem correctness vectors; an unpaired or `n<200` delta is
  rejected as gameable.
- `derive_premise_verdict` maps `(ar_accuracy, energy_descent_accuracy,
  p_value, ci)` to exactly one terminal verdict prefixed `complete:` and reports
  the two gates G1 (premise-viable / non-inferior) and G2 (premise-validated /
  significant superiority).
- The Exp 3312 artifact carries `inference_substrate=live_llm_inference`,
  `n_problems>=200`, `ar_baseline_accuracy`, `energy_descent_accuracy`,
  `accuracy_delta`, `paired_significance`, `compute_parity_note`, `random_seed`,
  `reproducibility_checksum`, and a `duration_s` above the 60s live-inference
  floor.

### SCENARIO-KONA-3312: Exp 3312 Runs the Premise Head-to-Head
**Given** CUDA is available, a trainable Boltzmann-GPT energy substrate, a
GSM8K subset of `n>=200` real problems, and a cached SOTA AR baseline GGUF
**When** we run `experiment_3312_energy_descent_vs_autoregressive_premise_v1.py`
**Then** it scores the AR baseline and the energy-descent condition on the SAME
paired problems with comparable generation compute
**And** it writes `results/experiment_3312_energy_descent_vs_autoregressive_premise_v1.json`
with a `complete:` verdict, the G1/G2 gate booleans, and a paired significance
test over `n>=200` per-problem outcomes.

### SCENARIO-KONA-3312-BLOCKED: Exp 3312 Emits an Honest Block on Missing Preconditions
**Given** any of CUDA, the energy substrate, the real corpus, or the AR baseline
is unavailable
**When** the experiment runs its step-0 preconditions
**Then** it writes a `blocked_<resource>` honest verdict and exits without
fabricating accuracy numbers.

### REQ-KONA-3426: Four-Condition Matched-Compute Premise Test (P0.1 v2)

Exp 3312 (REQ-KONA-3312) showed energy-descent selection beat a single greedy
autoregressive (AR) generation but LOST to plain majority-vote self-consistency
over the same samples, and its artifact was flagged_adversarial (a false-positive
tautology: `random_seed` equalled the experiment id). The load-bearing question
is therefore still open: **does the energy function add anything beyond plain
majority vote at the SAME compute budget?** Exp 3426 answers it with a paired
four-condition head-to-head on a real reasoning benchmark (GSM8K `n>=200`,
held-out) at MATCHED compute, with a falsifiable significance gate whose PRIMARY
comparison is energy-weighted vote vs majority-vote self-consistency.

**Rationale:** if energy cannot beat — or even match — plain self-consistency at
equal compute, the "energy-descent reasoning is a better substrate" framing that
motivates the Phase-3 / Kona foundation-model endgame is unsupported and the
superiority framing retires. If energy SIGNIFICANTLY beats self-consistency, that
is the first real justification for the endgame. Either outcome is high-value.

**Acceptance criteria:**

- `mean_token_confidence` / `self_certainty_select` implement the cheap
  self-certainty Best-of-N selector (arXiv:2502.18581) over the per-token chosen
  logprobs llama.cpp returns, as a disclosed monotone proxy for sequence
  confidence — the strongest cheap selector energy must beat.
- `energy_weighted_vote` aggregates the same k candidate answers by
  `softmax(-E/T)` over distinct extracted answers (EBM-CoT calibration,
  arXiv:2511.07124); `T -> inf` recovers plain majority vote, so the premise is
  that a meaningful `T` reshapes the vote toward correctness.
- All four sampled-aggregation conditions (self-consistency, self-certainty BoN,
  energy-argmin, energy-weighted vote) consume the SAME `k` generations; greedy
  AR is the 1-sample floor — energy gets no extra samples (matched compute).
- `derive_premise_v2_verdict` maps the PRIMARY (energy-weighted-vote vs
  self-consistency) paired comparison to exactly one terminal verdict prefixed
  `complete:`, reporting G1 (energy non-inferior to self-consistency) and G2
  (energy significantly beats self-consistency at matched compute).
- The Exp 3426 artifact carries `inference_substrate=live_llm_inference`,
  `n_problems>=200`, `k_samples`, the five condition accuracies,
  `delta_energy_vs_self_consistency`, `delta_energy_vs_greedy_ar`,
  `paired_significance`, `compute_parity_note`, `random_seed` (distinct from the
  experiment id), `reproducibility_checksum`, and a `duration_s` above the 60s
  live-inference floor, with clean methodology so `adversarial_verify.py` does
  not flag it.

### SCENARIO-KONA-3426: Exp 3426 Runs the Four-Condition Premise at Matched Compute
**Given** CUDA is available, a trainable Boltzmann-GPT energy substrate, a GSM8K
subset of `n>=200` real problems, and a cached SOTA GGUF whose embedded tokenizer
loads via the GGUF path
**When** we run `experiment_3426_energy_descent_vs_ar_vs_self_consistency_premise_v2.py`
**Then** it scores greedy AR, self-consistency, self-certainty BoN, energy-argmin,
and energy-weighted vote on the SAME paired problems with the four aggregation
conditions sharing the SAME `k` samples
**And** it writes the artifact with a `complete:` verdict, the G1/G2 gate booleans,
and a paired McNemar + bootstrap significance test for the PRIMARY energy-vs-
self-consistency delta over `n>=200` per-problem outcomes.

### SCENARIO-KONA-3426-BLOCKED: Exp 3426 Emits an Honest Block on Missing Preconditions
**Given** any of CUDA, the energy substrate, the real corpus, or the SOTA GGUF
embedded tokenizer is unavailable
**When** the experiment runs its step-0 preconditions
**Then** it writes a `blocked_<resource>` honest verdict and exits without
fabricating accuracy numbers.

### REQ-KONA-3448: Resumable Generation-Corpus Builder for the P0.1 Premise Test

Exp 3437 (the prior P0.1 attempt) did NOT fail scientifically — it died of a
1201s idle-timeout: a single in-session job that did live 35B generation over
`200 x k` samples AND scored energy/self-consistency ran silently past the
agent's wall-clock+idle budget and produced no artifact at all. The structural
fix is to DECOUPLE generation from scoring. Exp 3448 does ONLY the expensive,
non-deterministic part — generating sampled candidate solutions from the SOTA
GGUF — and writes them to a cached, append-only, RESUMABLE corpus at
`data/p01_gsm8k_generations.jsonl`. A downstream scoring task (exp3449) then
consumes that corpus with NO live model and thus no idle-timeout risk, and
answers the P0.1 question (does energy selection beat self-consistency at
matched compute?) deterministically.

**Rationale:** a builder that checkpoints one completed problem at a time, prints
a progress line after every problem (so the subprocess is never silent for the
~20 minutes that killed exp3437), respects an ~18-minute wall-time budget, and
resumes from whatever it already wrote makes the corpus accumulate across
milestones without any single run ever timing out. A partial corpus is progress,
not failure.

**Acceptance criteria:**

- `completed_problem_ids` reads the JSONL corpus (if present) and returns the set
  of problem ids that already have a full set of generations, so a re-invocation
  SKIPS them and only generates for the remainder — the resume contract.
- `build_corpus_row` packs one completed problem into a JSONL row carrying the
  problem id, question, gold answer, the greedy generation, and the `k` sampled
  generations, where EACH generation records its raw text, extracted integer
  answer, the per-token logprobs, and the mean-token logprob (the mean token
  confidence needed for downstream self-certainty Best-of-N).
- `warmup_self_consistency_check` computes, over the first `>=20` completed
  problems, the majority-vote (self-consistency) accuracy and the greedy
  accuracy, and sets `self_consistency_non_degenerate=true` iff SC accuracy
  `>= greedy AND > 0.30`; when degenerate it records the raw extracted answers of
  three example problems so the per-sample extraction bug (the exp3426 0.0 bug)
  is diagnosable — but generation CONTINUES (the corpus is still useful and the
  scoring task re-validates the gate).
- `derive_corpus_verdict` maps `n_completed` vs the target to exactly one
  terminal verdict prefixed `complete:` (`..._complete`, `..._partial_resumable`,
  or `..._seeded_..._resume_next_milestone`); a partial corpus is a terminal
  success, not a failure.
- The Exp 3448 artifact carries `inference_substrate=live_llm_inference`,
  `corpus_path`, `n_problems_completed`, `n_problems_target`, `k_samples`,
  `per_sample_logprobs_captured`, `self_consistency_non_degenerate`,
  `warmup_self_consistency_accuracy`, `warmup_greedy_accuracy`, `model_specs`
  (the gemma-4-26B-A4B-it GGUF), `random_seed`, `reproducibility_checksum`, and a
  `duration_s` above the 60s live-inference floor.

### SCENARIO-KONA-3448: Exp 3448 Builds a Resumable Generation Corpus
**Given** CUDA is available, the `unsloth/gemma-4-26B-A4B-it-GGUF` embedded
tokenizer loads via the GGUF path, and a GSM8K subset of `>=120` real problems
with integer labels
**When** we run `experiment_3448_p01_generation_corpus_builder_v1.py`
**Then** for each not-yet-completed problem it generates one greedy generation and
`k=6` sampled generations, capturing raw text, extracted answer, and per-token
logprobs for each, and appends one JSONL row to `data/p01_gsm8k_generations.jsonl`
immediately after the problem finishes
**And** it prints a one-line progress message after every problem, stops when it
approaches the ~18-minute wall-time budget, and writes the artifact with a
`complete:` verdict reporting how many problems/samples landed and the warm-up
self-consistency self-check.

### SCENARIO-KONA-3448-RESUME: Exp 3448 Resumes From a Partial Corpus
**Given** `data/p01_gsm8k_generations.jsonl` already contains completed rows from a
prior run
**When** the experiment is re-invoked
**Then** it reads the already-completed problem ids, skips them, and only generates
for the remaining problems, so the corpus accumulates monotonically toward the
target across milestones without re-doing finished work.

### SCENARIO-KONA-3448-BLOCKED: Exp 3448 Emits an Honest Block on Missing Preconditions
**Given** any of CUDA, the SOTA GGUF embedded tokenizer, or the real GSM8K corpus
is unavailable
**When** the experiment runs its step-0 preconditions
**Then** it writes a `blocked_<resource>` honest verdict and exits without
fabricating any generations.

### REQ-KONA-3459: Resume-and-Extend the P0.1 Generation Corpus Toward n=120 (v2)

Exp 3448 (REQ-KONA-3448) landed a PARTIAL corpus of `n=47/120` problems at
`data/p01_gsm8k_generations.jsonl` — its resumable builder exited clean on its
~18-minute wall-time budget BY DESIGN, not by failure. The downstream P0.1 crux
(exp3460) needs a HEADLINE-eligible sample (`>=80` problems) before it can report
a trained-energy verdict rather than a preliminary one. Exp 3459 RE-INVOKES the
same decoupled generation half: it reads which problem ids are already complete,
SKIPS them, and generates only the remainder (1 greedy + `k=6` sampled per
problem, with per-token / mean-token logprobs), using the SAME GSM8K split, seed,
model, and sampling parameters exp3448 documented so the extended rows are
homogeneous. It appends one JSONL row per completed problem, prints a progress
line after every problem (defeating the exp3437 idle-timeout), and exits clean on
its wall-time budget with whatever it finished.

**Rationale:** the corpus must accumulate MONOTONICALLY across milestones. A v2
extend run that regressed the corpus, dropped the logprobs the scorer needs, or
re-generated finished problems would waste GPU time and break the resume
contract. The verdict band is sharpened for v2: `>=120` is complete, `>=80` is
HEADLINE-eligible (lets exp3460 report a headline, not a preliminary, verdict),
and anything below 80 is an extended-partial that resumes again next milestone.

**Acceptance criteria:**

- The v2 verdict (`derive_extend_verdict`) maps `n_completed` vs the target to
  exactly one `complete:`-prefixed terminal verdict in three bands:
  `..._complete_n=NN` (`>=120`), `..._headline_eligible_n=NN` (`80 <= n < 120`),
  and `..._extended_partial_n=NN_resume_next_milestone` (`n < 80`).
- The v2 gates (`extend_acceptance_gates`) report G1 CORPUS-NOT-REGRESSED
  (`n_completed >= 47 AND per_sample_logprobs_captured`) — the corpus did not
  shrink and still carries the logprobs the scoring task needs (resume is
  monotone) — and G2 HEADLINE-ELIGIBLE (`n_completed >= 80`).
- `added_this_run(n_total, n_prior)` reports the problems newly generated this
  invocation (`max(0, n_total - n_prior)`), distinct from the running total, so
  the artifact proves the resume actually generated new work.
- The Exp 3459 artifact carries `inference_substrate=live_llm_inference`,
  `corpus_path`, `n_problems_completed`, `n_problems_target`,
  `n_problems_added_this_run`, `k_samples`, `per_sample_logprobs_captured`,
  `self_consistency_non_degenerate`, `warmup_self_consistency_accuracy`,
  `warmup_greedy_accuracy`, `model_specs` (the gemma-4-26B-A4B-it GGUF),
  `random_seed`, `reproducibility_checksum`, and a `duration_s` above the 60s
  live-inference floor.

### SCENARIO-KONA-3459: Exp 3459 Resumes and Extends Toward the Headline-Eligible Target
**Given** `data/p01_gsm8k_generations.jsonl` already holds the exp3448 partial
corpus (`n=47`) and CUDA + the `unsloth/gemma-4-26B-A4B-it-GGUF` embedded
tokenizer are available
**When** we run `experiment_3459_p01_generation_corpus_extend_to_120_v2.py`
**Then** it loads the exp3448 GSM8K split + seed, skips the already-completed
problem ids, generates 1 greedy + `k=6` sampled generations (with per-token
logprobs) for the remaining problems, appends one JSONL row per completed problem,
prints a progress line after each, and exits clean on its wall-time budget
**And** it writes the artifact with a `complete:` verdict reporting the new total
`n_problems_completed`, `n_problems_added_this_run`, the G1/G2 gates, and the
full-corpus warm-up self-consistency self-check.

### SCENARIO-KONA-3459-RESUME-MONOTONE: Exp 3459 Never Regresses the Corpus
**Given** the corpus already contains `n` completed problems with per-sample
logprobs
**When** the extend run is re-invoked any number of times
**Then** the completed problem count never decreases, finished problems are never
re-generated, and the G1 CORPUS-NOT-REGRESSED gate stays true so the resume
contract is monotone.

### REQ-KONA-3471: Build a HARD-MATH Headroom Corpus With Per-Step Traces for P0.1 (v1)

P0.1 — "does energy-based selection/voting BEAT plain self-consistency at equal
compute?" — could NOT be answered on GSM8K because self-consistency (SC) is at
CEILING there: exp3460 found the trained-energy vote tied SC EXACTLY (SC ~0.908),
leaving no room for any selector to help. A selector can only beat SC where SC has
HEADROOM — i.e. where SC accuracy is materially below 1.0. The process-reward
literature (arXiv:2602.11570 PRIME: +8-9% on AIME from process-aware verification)
shows that the regime where verifier selection beats SC is HARD math scored as a
PROCESS reward over per-step reasoning traces. Exp 3471 builds that substrate: a
NEW cached corpus on a hard-math benchmark (MATH Level 5) whose SC lands in the
HEADROOM band `[0.4, 0.7]`, capturing 1 greedy + `k=6` sampled generations per
problem, EACH carrying a parsed list of discrete reasoning STEPS so the FoVer
step-error verifier (the 0.9131 ensemble) can be scored as a PROCESS reward by the
downstream crux (exp3472/3473/3475).

**Rationale:** the corpus is only useful if SC is NOT at ceiling — otherwise
exp3472 repeats the GSM8K null. The G1 HEADROOM-CONFIRMED gate makes that property
a first-class, falsifiable precondition rather than an assumption. The builder
reuses the proven decoupled-generation discipline (generation only, append per
problem, progress line per problem, wall-time budget, clean partial exit, resume
on re-invocation) so it cannot idle-timeout, and the NEW capability over the GSM8K
builders is the per-step trace capture plus the headroom-band self-check.

**Acceptance criteria:**

- The verdict (`derive_headroom_verdict`) maps `n_completed`, the warm-up SC, and
  the in-band boolean to exactly one `complete:`-prefixed terminal verdict:
  `..._headline_eligible_n=NN_sc=SS` (`n >= 80` AND in band),
  `..._scorable_partial_n=NN_resume_next_milestone` (`40 <= n < 80` AND in band),
  `..._partial_n=NN_resume_next_milestone` (too few problems to judge the band yet),
  and `blocked_no_headroom_benchmark_sc_outside_band` (enough problems but SC fell
  outside `[0.4, 0.7]` — the chosen split has no headroom).
- The gates (`headroom_acceptance_gates`) report G1 HEADROOM-CONFIRMED
  (`self_consistency_in_headroom_band`) and G2 SCORABLE
  (`n_completed >= 40 AND per_step_traces_captured`).
- `parse_reasoning_steps` splits a chain-of-thought into discrete reasoning steps
  (newline-delimited, falling back to sentence segmentation for single-line CoT),
  capped to a bounded count, so each generation carries a step list for PROCESS
  scoring.
- `extract_math_answer` / `normalize_math_answer` / `math_is_correct` extract and
  compare the `\boxed{}` final answer of a MATH generation (falling back to a
  `#### <n>` coda or the last number), normalising LaTeX so equivalent surface
  forms match.
- `headroom_warmup_check` recomputes SC vs greedy accuracy over the corpus and the
  `self_consistency_in_headroom_band` boolean, exactly what the live run reports.
- The Exp 3471 artifact carries `inference_substrate=live_llm_inference`,
  `corpus_path` (`data/p01_hardmath_generations.jsonl`), `benchmark_id`,
  `n_problems_completed`, `n_problems_target`, `n_problems_added_this_run`,
  `k_samples`, `per_step_traces_captured`, `per_sample_logprobs_captured`,
  `warmup_self_consistency_accuracy`, `self_consistency_in_headroom_band`,
  `warmup_greedy_accuracy`, `model_specs` (the gemma-4-26B-A4B-it GGUF),
  `random_seed`, `reproducibility_checksum`, and a `duration_s` above the 60s
  live-inference floor.

### SCENARIO-KONA-3471: Exp 3471 Builds a Hard-Math Headroom Corpus With Per-Step Traces
**Given** CUDA and the `unsloth/gemma-4-26B-A4B-it-GGUF` embedded tokenizer are
available and the MATH Level 5 split is loadable
**When** we run `experiment_3471_p01_headroom_corpus_builder_hard_math_v1.py`
**Then** it confirms the warm-up SC lands in `[0.4, 0.7]`, generates 1 greedy +
`k=6` sampled generations per problem (each with an extracted `\boxed{}` answer, a
correctness label, a mean-token logprob, AND a parsed list of reasoning steps),
appends one JSONL row per completed problem to
`data/p01_hardmath_generations.jsonl`, prints a progress line after each problem,
and exits clean on its ~22-minute wall-time budget
**And** it writes the artifact with a `complete:` verdict reporting
`n_problems_completed`, `n_problems_added_this_run`, the G1 HEADROOM-CONFIRMED /
G2 SCORABLE gates, and the full-corpus warm-up SC headroom-band self-check.

### SCENARIO-KONA-3471-RESUME: Exp 3471 Resumes and Never Re-Generates Finished Problems
**Given** `data/p01_hardmath_generations.jsonl` already contains `n` completed
problems for the chosen split
**When** the builder is re-invoked
**Then** it reads the completed problem ids, skips them, generates only the
remainder, and reports `n_problems_added_this_run` distinct from the running total
so the artifact proves the resume did new work.

### SCENARIO-KONA-3471-NO-HEADROOM: Exp 3471 Emits an Honest Block When the Split Has No Headroom
**Given** enough problems have been scored to judge the band but the warm-up SC
falls outside `[0.4, 0.7]` (the split is too easy or too hard)
**When** the builder finalises
**Then** it emits the `complete: blocked_no_headroom_benchmark_sc_outside_band`
verdict (a clean `complete:` prefix so downstream gate-synth/capstone are NOT
cascade-blocked) and records the measured SC so a re-invocation can switch to a
harder/easier split.

### REQ-KONA-3449: Cached Six-Condition Energy-Vote-vs-Self-Consistency Scoring (P0.1 v4)

Exp 3448 (REQ-KONA-3448) decoupled the expensive generation half of the P0.1
premise test into a cached, resumable corpus at
`data/p01_gsm8k_generations.jsonl`. Exp 3449 is the SCORING half: it invokes NO
live model, consumes the cached corpus deterministically, and so completes in
seconds and CANNOT idle-timeout (the exp3437 failure mode). It answers the crux
**does energy-based selection/voting BEAT plain majority-vote self-consistency at
MATCHED compute?** with six paired conditions and a falsifiable significance gate.

**Rationale:** the sharpened bar (arXiv:2410.12608 "Not All Votes Count" +
arXiv:2510.14913 budget-aware hybrid) is that a verifier-weighted vote can beat
self-consistency, and a verifier+SC HYBRID beats either alone — so the honest
target is "energy (or the energy×SC hybrid) >= self-consistency", not "energy
beats greedy AR". arXiv:2506.01369 is the adversarial counter: external verifiers
often UNDERPERFORM self-consistency; if energy loses, that is the
predicted-and-explained outcome and the energy-superiority framing retires
honestly. Either outcome is high-value and converges with the .317 Kona result
(energy is a global heuristic; only the hybrid solves).

**Acceptance criteria:**

- `extract_steps` splits a candidate generation into reasoning steps, and
  `candidate_energy` scores a candidate with a deterministic, parameter-free
  verifier-energy ensemble (arithmetic constraint-violation energy
  `IsingVerifier` + adjacent-step EBM-CoT contradiction energy), where lower
  energy means a more internally-consistent reasoning trace — no training and no
  live model, so the score is reproducible.
- `majority_vote` (self-consistency, the PRIMARY control), `self_certainty_bon`
  (arXiv:2502.18581, max mean-token-logprob), `energy_argmin`,
  `energy_weighted_vote` (EBM-CoT softmax(-E/T) over distinct answers,
  arXiv:2511.07124, THE headline condition), and `energy_sc_hybrid`
  (arXiv:2510.14913, SC majority signal combined with energy weighting) all
  consume the SAME `k` cached generations; greedy AR is the 1-sample floor — the
  energy conditions get no extra samples (matched compute).
- A NON-DEGENERATE-SC gate re-asserts over the FULL corpus that self-consistency
  accuracy `>= greedy AND > 0.30` BEFORE any energy comparison is reported; when
  it fails, the verdict is
  `complete: blocked_self_consistency_harness_degenerate_per_sample_extraction_broken`
  with the raw extracted answers of three example problems — the exp3426 0.0-tie
  guard.
- `mcnemar_exact` and `paired_bootstrap_ci` provide a paired significance test for
  the PRIMARY (energy-weighted-vote vs self-consistency) delta and the hybrid
  delta over the per-problem paired outcomes; an unpaired or `n<30` delta is
  gameable.
- `derive_premise_v4_verdict` maps the comparison to exactly one terminal verdict
  prefixed `complete:`, reporting G1 (energy or hybrid non-inferior to
  self-consistency) and G2 (energy or hybrid significantly beats self-consistency
  at matched compute).
- The Exp 3449 artifact carries
  `inference_substrate=verifier_ensemble_against_cached_candidates`, `n_problems`,
  `k_samples`, `self_consistency_non_degenerate`, the six condition accuracies,
  `delta_energy_vs_self_consistency`, `delta_hybrid_vs_self_consistency`,
  `delta_energy_vs_greedy_ar`, `paired_significance`, `compute_parity_note`,
  `random_seed`, `reproducibility_checksum`, and a `duration_s` above the 1s
  cached-scoring floor, with clean methodology so `adversarial_verify.py` does not
  flag it.

### SCENARIO-KONA-3449: Exp 3449 Scores Six Conditions Over the Cached Corpus
**Given** `data/p01_gsm8k_generations.jsonl` exists with `>=30` problems each
having a greedy generation, `k>=5` sampled generations with extracted answers and
per-sample logprobs, and the verifier-energy substrate is loadable
**When** we run
`experiment_3449_p01_energy_vote_vs_self_consistency_cached_scoring_v4.py`
**Then** it scores greedy AR, self-consistency, self-certainty BoN, energy-argmin,
energy-weighted vote, and the energy×SC hybrid on the SAME paired problems sharing
the SAME `k` cached samples
**And** it writes the artifact with a `complete:` verdict, the G0/G1/G2 gate
booleans, and a paired McNemar + bootstrap significance test for the PRIMARY
energy-vs-self-consistency delta and the hybrid delta.

### SCENARIO-KONA-3449-BLOCKED: Exp 3449 Emits an Honest Block on a Too-Small or Degenerate Corpus
**Given** the cached corpus is absent / has `n<30` problems, OR the energy
substrate is unloadable, OR self-consistency is degenerate over the full corpus
**When** the experiment runs its step-0 preconditions
**Then** it writes the matching `complete: blocked_*` honest verdict (a clean
`complete:` prefix so downstream gate-synth/capstone are NOT cascade-blocked) and
exits without reporting any energy comparison against a broken control.

### REQ-KONA-3440: Kona Global-Opt Correctness-First Solve-Rate Gate

Carnot MUST re-gate the Kona-style global-optimization Sudoku claim on
SOLVE-RATE (a valid board with all Sudoku constraints satisfied / final energy
zero, verified on the board), NOT on time-to-solution. The exp3408 framing
(`solved_sudoku=False`, energy plateaued at 10.05, yet an implied "~15x speedup
over autoregressive") is fast-but-wrong-vs-slow and MUST be retired: a speedup
claim is invalid until the method actually solves.

The implementation MUST:

1. **STEP 0a (gating encoding validity):** encode a KNOWN-VALID solved Sudoku
   board into the Ising energy and assert the total energy is zero within a
   float epsilon. If a correct solution does not give E==0, the energy
   formulation is mis-specified; the experiment MUST stop, report the
   per-constraint residual-energy breakdown, and run no optimization. No
   optimizer can solve a mis-specified energy.
2. **STEP 0b (easy-tier sanity):** before any hard puzzle, confirm the optimizer
   solves easy boards (many clues). A near-zero easy solve-rate indicates a
   representational (energy/encoding) failure, not an optimizer-power failure.
3. **Optimizer ladder (only if 0a passes and 0b solves easy):** run vanilla
   Langevin and an annealed + random-restart variant, reporting `solve_rate`
   by difficulty AND by `optimizer_variant`.
4. **Plateau characterization:** report `n_violated_constraints_at_plateau` so a
   "few cells almost-solved" outcome is distinguished from pervasive
   infeasibility.
5. **Energy-guided + constraint-propagation hybrid:** report `hybrid_solve_rate`
   separately so the claim narrows honestly from "energy replaces search" to
   "energy is a global heuristic" when only the hybrid solves.

The artifact MUST be written to
`results/experiment_3440_kona_global_opt_correctness_v3.json` and include
`encoding_validity_E0`, `easy_tier_solve_rate`,
`n_violated_constraints_at_plateau`, `hybrid_solve_rate`, `solve_rate`,
`n_puzzles` (>=20), `solve_rate_by_difficulty`, `time_to_solution_solved_only`,
`optimizer_variant`, `random_seed`, `reproducibility_checksum`, `duration_s`,
and a terminal `honest_verdict` (starting with `complete:`/`success:`/`passed:`/
`shipped:`). A speedup-vs-autoregressive number, if reported at all, MUST be
restricted to the solved subset.

### SCENARIO-KONA-3440: Exp 3440 Re-Gates Kona Global-Opt on Solve-Rate

**Given** the continuous Sudoku Ising energy from `carnot.verify.sudoku` and a
seeded set of at least 20 valid Sudoku puzzles across easy/medium/hard tiers
**When** we run `experiment_3440_kona_global_opt_correctness_v3.py`
**Then** STEP 0a verifies a known-valid solved board gives E==0 before any
optimization solve-rate is reported
**And** it reports `solve_rate` over the puzzle set, `solve_rate_by_difficulty`,
`n_violated_constraints_at_plateau`, and `hybrid_solve_rate`, with any
speedup claim restricted to the solved subset
**And** the artifact is written with a terminal `honest_verdict` that honestly
states whether pure energy descent solves, only the hybrid solves, or the
current Ising energy formulation cannot do hard-Sudoku global reasoning yet.

### SCENARIO-KONA-3440-ENCODING-INVALID: Exp 3440 Forks on a Mis-Specified Energy

**Given** the encoding-validity check on a known-valid solved board
**When** the total energy of that board exceeds the float epsilon
**Then** the experiment writes
`complete: blocked_energy_encoding_invalid_per_constraint_residual_reported`
with the per-constraint residual-energy breakdown and runs no optimization.

### REQ-KONA-3460: Trained Energy Reranker vs Self-Consistency on Held-Out GSM8K (P0.1 v5)

Exp 3449 (REQ-KONA-3449) established that the UNTRAINED, parameter-free
verifier-energy ensemble does NOT beat majority-vote self-consistency: the
energy-weighted vote degenerated onto plain majority (the exp3449 tautology
flag), and exp3450 measured energy-vs-correctness AUROC at 0.516 (~chance). The
literature says the fix is a TRAINED energy: arXiv:2505.14999 (EORM — a
lightweight energy reranker trained on outcome labels boosts math reasoning),
arXiv:2603.25450 (generator self-perplexity is nearly uninformative on CoT), and
arXiv:2506.09338 (PRM calibration). Exp 3460 is the never-asked test: does a
TRAINED outcome-label energy reranker, evaluated on a HELD-OUT problem-level
split, MATCH or BEAT self-consistency at matched compute? It invokes NO live
model — it consumes the cached corpus and a small trained reranker — so it
completes in seconds and CANNOT idle-timeout.

**Rationale:** the UNTRAINED energy failed the AUROC-0.55 floor (exp3450 0.516).
A trained reranker is the literature-prescribed fix, but it MUST be evaluated on
held-out problems with a leakage-guarded, problem-level split, at matched
generation budget, against a VERIFIED-NON-DEGENERATE self-consistency control
(the exp3426 0.0-tie guard). If even a trained energy cannot match SC at equal
compute, the energy-superiority framing for final-answer selection retires
honestly on this substrate; if it beats SC significantly, that is the first real
Phase-3 justification.

**Acceptance criteria:**

- `candidate_feature_vector` extracts a fixed-length feature vector per cached
  candidate (arithmetic-violation energy `IsingVerifier`, adjacent-step
  contradiction energy `EbmCotCalibrator`, Curry-Howard type-violation score
  `Tier0rVerifier`, logical-inconsistency score `Tier0uVerifier`, mean-token
  logprob, and a step-count feature); `fover_candidate_energy` aggregates the four
  verifier signals into one scalar candidate energy (lower is better) — the FoVer
  step-error ensemble routed into final-answer selection.
- `TrainedEnergyReranker` is a small logistic-regression energy reranker trained
  on candidate outcome-correctness labels with train-fold-only feature
  standardisation (the leakage guard); its parameter count is recorded
  (`n_params`) as the compute-parity accounting.
- `problem_kfold_indices` produces a deterministic K-fold split BY PROBLEM id
  (never splitting samples of the same problem across folds); ALL reported
  accuracies are on held-out problems only.
- `trained_energy_weighted_vote` votes over the `k` held-out samples weighted by
  the trained reranker's P(correct) (THE headline condition); `trained_energy_sc_hybrid`
  combines the SC majority signal with the trained weight (arXiv:2510.14913);
  `fover_energy_argmin` picks the lowest FoVer-energy answer. Greedy AR is the
  1-sample floor; self-consistency (majority vote) is the PRIMARY control; all
  selection conditions consume the SAME `k` cached generations (matched compute).
- A NON-DEGENERATE-SC gate re-asserts over the FULL corpus that self-consistency
  accuracy `>= greedy AND > 0.30` BEFORE any energy comparison is reported; when
  it fails, the verdict is
  `complete: blocked_self_consistency_harness_degenerate_per_sample_extraction_broken`
  with three example problems' raw extracted answers.
- `mcnemar_exact` + `paired_bootstrap_ci` provide a paired significance test for
  the trained-energy-vote, FoVer-energy, and hybrid deltas vs self-consistency
  over per-problem held-out outcomes.
- `derive_v5_verdict` maps the result to exactly one terminal verdict prefixed
  `complete:`: G1 (a trained-energy condition non-inferior to SC) and G2 (a
  trained-energy condition significantly beats SC at matched compute).
- The Exp 3460 artifact carries
  `inference_substrate=verifier_ensemble_against_cached_candidates`,
  `n_problems_heldout`, `k_samples`, `reranker_param_count`,
  `train_test_split_note`, `self_consistency_non_degenerate`, the six condition
  accuracies, `delta_trained_energy_vs_self_consistency`,
  `delta_fover_energy_vs_self_consistency`, `delta_hybrid_vs_self_consistency`,
  `paired_significance`, `compute_parity_note`, `random_seed`,
  `reproducibility_checksum`, and a `duration_s` above the 1s cached-scoring
  floor, with clean methodology so `adversarial_verify.py` does not flag it.

### SCENARIO-KONA-3460: Exp 3460 Trains and Held-Out-Scores Six Conditions
**Given** `data/p01_gsm8k_generations.jsonl` exists with `>=47` problems each
having a greedy generation, `k>=5` sampled generations with extracted answers and
per-sample logprobs, and the verifier-energy + reranker substrate is loadable
**When** we run
`experiment_3460_p01_trained_energy_reranker_vs_self_consistency_v5.py`
**Then** it splits the corpus by problem id into K folds, trains a small
outcome-label logistic-regression energy reranker on each train fold with
train-fold-only standardisation, and scores greedy AR, self-consistency,
self-certainty BoN, FoVer-energy argmin, trained-energy-weighted vote, and the
trained-energy×SC hybrid on the held-out problems
**And** it writes the artifact with a `complete:` verdict, the G0/G1/G2 gate
booleans, the reranker parameter count, the problem-level split note, and a paired
McNemar + bootstrap significance test for the trained-energy, FoVer-energy, and
hybrid deltas vs self-consistency.

### SCENARIO-KONA-3460-BLOCKED: Exp 3460 Emits an Honest Block on a Too-Small or Degenerate Corpus
**Given** the cached corpus is absent / has `n<47` problems, OR the energy/reranker
substrate is unloadable, OR self-consistency is degenerate over the full corpus
**When** the experiment runs its step-0 preconditions
**Then** it writes the matching `complete: blocked_*` honest verdict (a clean
`complete:` prefix so downstream gate-synth/capstone are NOT cascade-blocked) and
exits without reporting any energy comparison against a broken control.

### REQ-KONA-3472: P0.1 Process-Aware Energy + Optimal Aggregation vs Self-Consistency on a HEADROOM Corpus
The Kona Phase-3 selection premise (a learned energy beats majority-vote
self-consistency at matched compute) failed on GSM8K (exp3460) for a structural
reason, not a modelling one: GSM8K self-consistency is at ceiling (~0.908), so
the majority vote is almost always already correct and an energy-weighted vote
degenerates onto it — an exact tie flagged as a tautology. The literature says
the win appears WITH HEADROOM and from PROCESS-level (step-aware) verification
(arXiv:2602.11570 PRIME reports process-aware verification beats outcome-only by
+8-9% on AIME; arXiv:2510.13918 gives an optimal SC+PRM aggregation). The
never-asked question: on a HEADROOM benchmark (SC in [0.4, 0.78]) does a
PROCESS-AWARE step-level energy plus OPTIMAL aggregation BEAT self-consistency at
matched compute? Exp 3472 answers it by scoring the cached HEADROOM corpus
(`data/p01_hardmath_generations.jsonl`, built by exp3471) with NO live model, so
it completes in seconds and cannot idle-timeout.

- The scoring substrate (`carnot.phase3.p01_process_energy`) reuses the v5
  trained-reranker, problem-level K-fold split, FoVer verifier bundle, and paired
  significance machinery, and ADDS three things: (1) a PER-STEP process energy
  that scores each parsed reasoning step with the FoVer 4-verifier ensemble and
  aggregates to a candidate process-energy; (2) an OPTIMAL SC+energy aggregator
  (arXiv:2510.13918) whose single mixing coefficient is fit on the TRAIN fold and
  applied on held-out; (3) a FLIP-COUNT primary metric that is tautology-clean by
  construction.
- Step 0 PRECONDITIONS gate before any comparison: (a) corpus present with
  `>=40` problems each having a greedy + `k>=5` sampled generations with per-step
  traces, correctness labels, and logprobs (else
  `complete: blocked_p01_corpus_too_small_n=NN`); (b) the FoVer + EORM substrate
  loadable (else `complete: blocked_energy_substrate_unavailable`); (c) a HEADROOM
  GATE requiring full-corpus SC accuracy in `[0.4, 0.78]` (else
  `complete: blocked_corpus_at_ceiling_no_headroom_sc=SS`).
- Seven held-out conditions at matched compute: (1) greedy AR floor; (2)
  self-consistency majority vote (PRIMARY control); (3) self-certainty BoN
  (arXiv:2502.18581); (4) process-energy argmin (the .320 new per-step condition);
  (5) trained-energy-weighted vote; (6) trained-energy×SC hybrid; (7) optimal
  SC+energy aggregation (THE headline condition). Conditions 2-7 consume the SAME
  `k` cached generations; energy adds only scoring + a tiny reranker/aggregator.
- PRIMARY metric is the FLIP-COUNT vs SC: for each energy/aggregation condition,
  `flip_count` (problems where the condition's selected answer differs from the SC
  majority answer), `flips_correct`, `flips_incorrect`, and
  `net_correctness_gain = flips_correct - flips_incorrect`. When a condition
  agrees with SC its `flip_count` is 0, reported ONCE with a methodology_note —
  never two bit-identical accuracy fields (the exp3460 tautology flag).
- `derive_v6_verdict` maps the result to exactly one `complete:`-prefixed terminal
  verdict over three gates: G0 HEADROOM (SC in band), G1
  ENERGY-BEATS-SC-WITH-HEADROOM (net_correctness_gain_optimal > 0 AND
  delta_optimal_vs_sc > 0 with paired p < 0.05), G2 NON-DEGENERATE (flip_count > 0).
- The Exp 3472 artifact carries `inference_substrate=verifier_ensemble_against_cached_candidates`,
  `benchmark_id`, `n_problems_heldout`, `k_samples`,
  `self_consistency_in_headroom_band`, the seven condition accuracies, the optimal
  flip metrics, `delta_optimal_vs_self_consistency`,
  `delta_process_energy_vs_self_consistency`, `paired_significance`,
  `compute_parity_note`, `random_seed`, `reproducibility_checksum`, and a
  `duration_s` above the 1s cached-scoring floor, with clean methodology so
  `adversarial_verify.py` does not flag it.

### SCENARIO-KONA-3472: Exp 3472 Scores Seven Conditions on a HEADROOM Corpus and Reports the Flip-Count
**Given** `data/p01_hardmath_generations.jsonl` exists with `>=40` problems each
having a greedy generation, `k>=5` sampled generations with per-step traces,
extracted answers, correctness labels, and per-sample logprobs, the FoVer + EORM
substrate is loadable, AND full-corpus self-consistency is in `[0.4, 0.78]`
**When** we run
`experiment_3472_p01_process_energy_vs_self_consistency_headroom_v6.py`
**Then** it splits by problem id into K folds, scores each candidate's parsed
steps with the FoVer per-step process energy, trains the EORM reranker and fits an
optimal SC+energy aggregator on each train fold, and scores all seven conditions
on the held-out problems
**And** it writes the artifact with a `complete:` verdict, the G0/G1/G2 gate
booleans, the optimal-aggregation flip metrics
(`flip_count_optimal_vs_sc`, `flips_correct_optimal`, `flips_incorrect_optimal`,
`net_correctness_gain_optimal`), the headline
`delta_optimal_vs_self_consistency`, and a paired McNemar + bootstrap
significance test for the optimal, process-energy, and hybrid deltas vs
self-consistency.

### SCENARIO-KONA-3472-BLOCKED: Exp 3472 Emits an Honest Block on a Too-Small, Unloadable, or Ceiling Corpus
**Given** the cached HEADROOM corpus is absent / has `n<40` problems, OR the
energy/reranker substrate is unloadable, OR full-corpus self-consistency is above
the headroom ceiling (`> 0.78`)
**When** the experiment runs its step-0 preconditions
**Then** it writes the matching `complete: blocked_*` honest verdict (a clean
`complete:` prefix so downstream gate-synth/capstone are NOT cascade-blocked) —
`complete: blocked_p01_corpus_too_small_n=NN`,
`complete: blocked_energy_substrate_unavailable`, or
`complete: blocked_corpus_at_ceiling_no_headroom_sc=SS` — and exits without
reporting any energy comparison.

---

### REQ-KONA-3495: P0.1 IN-BAND Contested-Subset Energy vs Self-Consistency (v8)

Exp 3495 is the cached, infra-robust P0.1 route. Instead of a fresh live
generation, it STRATIFIES the ALREADY-CACHED GSM8K and MATH-L5 corpora to the
CONTESTED SUBSET: problems whose per-problem sample correctness rate lands in
[0.40, 0.70] — neither trivially-right nor trivially-wrong. The contested subset
is constructed by loading both cached corpora, computing the per-problem
correctness rate (fraction of k samples with the gold answer), and keeping
problems in band. If n < 40 after pooling both corpora, the experiment emits
complete: blocked_contested_subset_too_small_n=NN honestly. If n >= 40, it runs
the same seven-condition process-energy + optimal-aggregation scoring as v6
(REQ-KONA-3472).

- Source corpora: data/p01_gsm8k_generations.jsonl and
  data/p01_hardmath_generations.jsonl, pooled.
- Contested subset filter: per-problem correctness rate in [0.40, 0.70].
- Minimum problem count: 40 (headline-eligible; else blocked_contested_subset_too_small).
- Scoring: same seven conditions as REQ-KONA-3472 (greedy AR, SC majority vote,
  self-certainty BoN, process-energy argmin, trained-energy-weighted vote,
  trained-energy x SC hybrid, optimal SC+energy aggregation).
- PRIMARY metric: FLIP-COUNT of optimal aggregation vs SC.
- The artifact carries inference_substrate=verifier_ensemble_against_cached_candidates,
  source_corpora, contested_subset_n, contested_subset_sc,
  self_consistency_in_headroom_band, the seven condition accuracies, flip metrics,
  deltas, paired_significance, random_seed, reproducibility_checksum, duration_s.

### SCENARIO-KONA-3495: Exp 3495 Builds the Contested Subset and Scores Seven Conditions
**Given** both cached corpora exist, their pooled contested subset (per-problem
correctness rate in [0.40, 0.70]) has n >= 40 problems, the FoVer + EORM
substrate is loadable, AND the contested-subset SC is in [0.40, 0.70]
**When** we run experiment_3495_p01_energy_vs_sc_contested_subset_inband_v8.py
**Then** it filters to the contested subset, splits by problem id into K folds,
scores all seven conditions on held-out problems, and writes the artifact with a
complete: verdict, contested_subset_n, contested_subset_sc,
self_consistency_in_headroom_band, flip metrics, and paired significance.

### SCENARIO-KONA-3495-BLOCKED: Exp 3495 Emits an Honest Block on Small Subset or Unavailable Substrate
**Given** the contested subset has n < 40 problems (after pooling both corpora),
OR the FoVer + EORM substrate is unloadable, OR no cached corpus exists
**When** the experiment runs its step-0 preconditions
**Then** it writes the matching complete: blocked_* honest verdict and exits without
reporting any energy comparison.

---

## P0.1 Real Combinatorial Optimizer Ladder (Exp 3505)

### REQ-KONA-3505: Discrete Combinatorial Optimizer Ladder on Validated Sudoku Encoding

**Motivation:** Exp 3494 validated that Carnot's Sudoku-Ising encoding is correct
(E=0 on a valid board). But the easy-tier solve rate was 0.0 because the optimizer
was continuous-relaxation Langevin, which cannot escape local minima in discrete
constraint satisfaction. The OPTIMIZER is the bottleneck, not the representation.

**Requirement:** Given the validated encoding (REQ proven by exp3494's E=0 assertion),
run a proper combinatorial optimizer ladder on >=20 Sudoku puzzles across difficulty
tiers (easy/medium/hard) and report solve-rate scored on the DISCRETE BOARD (all
constraints verified, not just energy threshold). The ladder must include at minimum:
(i) vanilla Langevin baseline (for contrast with exp3494), (ii) discrete SA with
single restart, (iii) discrete SA with K>=20 restarts, (iv) parallel tempering
(>=4 chains with replica exchange), (v) exact CP/backtracking solver (confirms boards
are solvable — the optimizer-isolation control). An AR greedy baseline provides the P0.1
comparator. Plateau characterization (n_violated_constraints_at_plateau) distinguishes
"almost solved" (few violations, optimizer-fixable) from "pervasive" (representational
failure). A hybrid solve rate (exact CP on same puzzles) provides the "energy is a
global heuristic" vs "energy replaces search" discriminator.

**Row-swap SA representation:** Discrete SA operates on the INTEGER board directly
using row-swap moves (swap two non-clue cells in the same row), which preserves row
uniqueness by construction. Energy = count of excess digit occurrences across columns
and boxes. Delta computation is O(1) via cached count arrays. This is the Simonis
(2005) classical approach for hard Sudoku.

**Acceptance gates:**
- G0 (CORRECTNESS-FIRST): encoding_validity_E0_reasserted == true AND
  exact_baseline_solve_rate > 0.5 (boards ARE solvable; failure isolates to optimizer)
- G1 (HONEST-SOLVE-RATE): solve_rate reported on >=20 puzzles; any speedup claim
  made ONLY on the solved subset
- G2 (P0.1-DATAPOINT): solve_rate > ar_baseline_solve_rate OR
  hybrid_solve_rate > ar_baseline_solve_rate (energy-based method beats AR on a CSP)

**Implementation:** `python/carnot/phase3/sudoku_discrete_sa.py` (row-swap SA +
parallel tempering), `python/carnot/phase3/sudoku_global_opt.py` (encoding, CP solver,
puzzle generation), `python/carnot/phase3/sudoku_p01_gate.py` (AR baseline).

### SCENARIO-KONA-3505: Exp 3505 Runs Optimizer Ladder and Reports P0.1 Datapoint
**Given** the Sudoku-Ising encoding is valid (E=0 on a known-valid board, regression
from exp3494), and >=20 puzzles spanning easy/medium/hard are available
**When** we run experiment_3505_p01_sudoku_real_combinatorial_optimizer_ladder_v2.py
**Then** it re-asserts E=0 (Step 0a), runs the full optimizer ladder (vanilla, SA
single, SA K=20 restarts, parallel tempering, exact CP), measures AR baseline,
reports solve_rate / solve_rate_by_difficulty / solve_rate_by_optimizer_variant /
exact_baseline_solve_rate / n_violated_constraints_at_plateau / hybrid_solve_rate /
ar_baseline_solve_rate, and emits one of the four terminal verdicts starting with
"complete:".

### SCENARIO-KONA-3505-BLOCKED: Exp 3505 Blocks on Encoding Regression
**Given** the Sudoku-Ising encoding no longer gives E=0 on a known-valid board
(regression from exp3494's validated state)
**When** the experiment runs Step 0a
**Then** it writes complete: blocked_energy_encoding_invalid_regression with the
residual breakdown and exits without running any optimizer.

### REQ-KONA-3507: P0.1 Energy vs Self-Consistency on Purpose-Built Level-3 In-Band Corpus (v9)

**Origin:** exp3506 built data/p01_difficulty_matched_generations.jsonl — a purpose-built
level-3 corpus with per-level aggregate SC in [0.40, 0.70] and >=40 problems. This is the
corpus the P0.1 premise test (energy beats SC at matched compute) was always waiting for.

**Requirement:** On the level-3 in-band corpus, run the same 7-condition process-aware
energy + optimal-aggregation comparison as exp3495 (v8), but with the NEW corpus as the
only source. Normalize the new corpus schema (gold_answer_norm / extracted_answer_norm /
reasoning_steps) to the format expected by the existing process-energy module. Report the
flip-count primary metric, 7 held-out accuracies, McNemar + bootstrap significance, and
all required artifact fields. Emit one of the terminal complete: verdicts per the
acceptance gates G0/G1/G2.

**Acceptance gates:**
- G0 (HEADROOM): level3_sc in [0.40, 0.70] and level3_n >= 40
- G1 (ENERGY-BEATS-SC-IN-BAND): net_correctness_gain_optimal > 0 AND
  delta_optimal_vs_self_consistency > 0 AND paired McNemar p < 0.05
- G2 (NON-DEGENERATE): flip_count_optimal_vs_sc > 0

### SCENARIO-KONA-3507: Exp 3507 Scores the Level-3 Corpus and Reports Energy vs SC

**Given** data/p01_difficulty_matched_generations.jsonl exists with >=40 level-3 problems
and aggregate level-3 SC in [0.40, 0.70]
**When** we run experiment_3507_p01_energy_vs_sc_on_level3_inband_corpus_v9.py
**Then** it loads the level-3 records, normalizes the schema, runs k-fold CV with the
FoVer process-energy module and EORM reranker, scores 7 conditions, computes flip-count
metrics, McNemar exact p and bootstrap CI95, and emits one terminal complete: verdict.

### SCENARIO-KONA-3507-BLOCKED: Exp 3507 Blocks on Missing Corpus or Insufficient Size

**Given** data/p01_difficulty_matched_generations.jsonl is missing or has fewer than 40
level-3 usable problems
**When** the experiment runs step 0
**Then** it writes complete: blocked_no_level3_corpus or
complete: blocked_level3_corpus_too_small_n=NN and exits without scoring.

### REQ-KONA-3516: P0.1 Level-3 Corpus Extend to N>=80 (v5, optional)

**Origin:** exp3506 (.323) left data/p01_difficulty_matched_generations.jsonl with 49
level-3 problems (SC=0.653, in band).  n>=80 makes exp3519 headline-eligible.  This
requirement governs the v5 optional builder that resumes from n=49 toward n>=80.

**Requirement:** Resume from the existing 49 level-3 rows, generate only NEW level-3
problems toward n>=80, append per problem, print flushed progress per problem, and stop
cleanly at 18-minute wall budget.  Use a content-derived random seed (NOT the experiment
number).  Emit one of the terminal complete: verdicts.

**Acceptance gates:**
- G1 (IN-BAND): level3_sc in [0.40, 0.70]  (OR honest blocked_ if preconditions fail)
- G2 (SEED-NOT-EXP-ID): RANDOM_SEED != EXP_ID (no tautology seed)

### SCENARIO-KONA-3516: Three-Band Terminal Verdict for Level-3 Corpus Extension v5

**Given** the level-3 corpus after a v5 run
**When** classify_verdict_v5(n, in_band, sc) is called
**Then** n>=80 and in-band -> complete: p01_level3_corpus_headline_eligible_n=N_sc=S;
40<=n<80 and in-band -> complete: p01_level3_corpus_scorable_partial_n=N_resume_next_milestone;
n<40 and in-band -> complete: p01_level3_corpus_partial_n=N_resume_next_milestone;
in_band=False -> complete: blocked_level3_sc_outside_headroom_band.

### SCENARIO-KONA-3516-TERMINAL-PREFIX: Every Verdict Starts With complete:

**Given** any (n, in_band, sc) combination
**When** classify_verdict_v5 is called
**Then** the returned string always starts with ``complete:`` per Verdict Terminal-Prefix
Discipline — no partial-token substring (blocked/marginal/no_improvement) at the front.

### SCENARIO-KONA-3516-SEED: Content-Derived Seed is Not the Experiment Number

**Given** the v5 script constants
**When** RANDOM_SEED is compared to EXP_ID
**Then** RANDOM_SEED != EXP_ID, ensuring the seed was computed from benchmark content
rather than copied from the experiment number (the tautology seed anti-pattern that
triggers GATE_PASSED_WITHOUT_DATA in adversarial_verify.py).

### REQ-KONA-4922: Distributional Energy Verifier Pivot Scaffold on a Non-Saturated Structured Slice

Carnot MUST provide an offline scaffold for the post-2026-06-30 verifier-moat
pivot mapped by Exp 4911. The scaffold SHALL port the FoVer evaluation harness
shape to a tiny TravelPlanner-style structured-reasoning slice, score cached
candidates only, and emit the three comparison columns required by the pivot:
`distributional_energy_verifier`, `self_consistency`, and `llm_judge`.

This is a scaffold and dry-run only. It MUST cite arXiv:2605.18871
(`Distributional Energy-Based Models for Uncertainty-Aware Structured LLM
Reasoning`, HTTP-200 verified by Exp 4911), set `verifier_is_oracle=false`,
set `self_consistency_saturated=false`, set `no_verifier_win_claimed=true`, and
avoid any claim that the distributional energy verifier beat self-consistency.
The comparison MAY be stubbed, but the distributional-energy column MUST combine
cached learned-quality score, deterministic constraint penalty, and uncertainty
instead of using model identity or an answer-key oracle.

The implementation SHALL live at
`python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py`, the
tiny cached slice SHALL live at
`data/experiment_4922_travelplanner_structured_slice.jsonl`, and the terminal
artifact SHALL be written to
`results/experiment_4922_distributional_energy_verifier_scaffold.json`.
The scaffold SHALL NOT modify `scripts/research_conductor.py`.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_scaffolded.
- `pivot_executable_on_6_30`: true -- the harness skeleton + dry-run make the post-sprint verifier-moat pivot executable the instant the sprint retires.
- `harness_skeleton_path`: the offline FoVer->non-saturated-domain harness skeleton (the de-risking deliverable).
- `dry_run_three_columns`: the tiny-slice dry-run output: {distributional_energy_verifier, self_consistency, llm_judge} columns -- proves the harness runs, NOT a headline.
- `validation_gate`: the gate the real post-6/30 experiment must pass: beats SC CI95-excl-0, no model-identity shortcut, oracle-distinct.
- `arxiv_id_cited`: 2605.18871 (HTTP-200 verified by exp4911) -- no fabrication.
- `verifier_is_oracle`: false -- the distributional energy verifier is oracle-distinct (the moat domain has no cheap executable oracle).
- `self_consistency_saturated`: false -- the moat only exists where self-consistency is NOT near-ceiling (the domain choice).
- `no_verifier_win_claimed`: true -- this is a SCAFFOLD + dry-run; the win is claimed only by the real post-6/30 experiment that passes the validation gate.
- `inference_substrate`: verifier_ensemble_against_cached_candidates (scores cached rows in the dry-run; 1s floor).
- `preconditions_checked`: records FoVer-harness + domain-slice presence; a missing resource emits blocked_.

### SCENARIO-KONA-4922-DRY-RUN: Scaffold Emits the Three Comparison Columns

**Given** the FoVer harness/runbook is present and the tiny
TravelPlanner-style cached slice is present with self-consistency below the
saturation ceiling
**When** `python/carnot/experiment_4922_distributional_energy_verifier_scaffold.py`
runs
**Then** it writes
`results/experiment_4922_distributional_energy_verifier_scaffold.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_scaffolded`,
`pivot_executable_on_6_30=true`, `arxiv_id_cited=2605.18871`, the three dry-run
columns `{distributional_energy_verifier, self_consistency, llm_judge}`, and the
post-sprint validation gate requiring CI95 excluding zero, no model-identity
shortcut under `adversarial_verify`, and oracle-distinct evaluation.

### SCENARIO-KONA-4922-BLOCKED: Missing Preconditions Block Honestly

**Given** the FoVer harness/runbook or the tiny structured-reasoning cached slice
is missing
**When** the scaffold runs its step-0 precondition check
**Then** it writes a terminal `blocked_*` verdict, records which resource is
missing in `preconditions_checked`, sets `pivot_executable_on_6_30=false`, and
does not emit a verifier-win claim.

### SCENARIO-KONA-4922-NO-WIN-CLAIM: Dry-Run Cannot Promote the Verifier

**Given** the tiny dry-run emits per-row selections for the three columns
**When** the artifact is validated
**Then** `no_verifier_win_claimed=true`, `verifier_is_oracle=false`,
`self_consistency_saturated=false`, and the artifact contains the validation
gate rather than a headline lift or promotion claim.

### REQ-KONA-4940: Distributional Energy Verifier Executable Spec and SOTA Ingestion

Carnot MUST advance the Exp 4922 distributional-energy verifier scaffold into
an executable design spec for the post-2026-06-30 verifier-moat experiment
without running the real benchmark. The executable spec SHALL ingest and cite
real arXiv sources for `2605.18871`, `2504.16828`, and `2502.01989`; map each
method onto the current Carnot verifier stack; and flag the strongest method(s)
as post-6/30 NEXT-milestone roadmap input. The mapping MUST state, per paper,
`strongest_method`, `implementation_cost_over_current_stack`, and `pitfalls`.

The dry-run SHALL reuse the Exp 4922 TravelPlanner-style structured slice while
confirming `self_consistency_saturated=false`. It SHALL wire exactly three
end-to-end columns: `self_consistency`, `decomposed_energy_verifier`, and
`oracle`. The `decomposed_energy_verifier` column MUST combine the existing
FoVer-style analytical/executable constraint penalties with a learned
quality-scorer ensemble abstraction in which the ensemble mean ranks candidates
and the ensemble standard deviation drives abstention. The verifier column MUST
ignore model identity and MUST NOT use the cached oracle labels used by the
`oracle` column. The dry-run is an executable wiring proof only, not a benchmark
or verifier-value claim.

The implementation SHALL live at
`python/carnot/experiment_4940_distributional_energy_verifier_executable_spec.py`,
the research note SHALL live at
`docs/research-notes/distributional-energy-verifier-executable-spec-20260628.md`,
and the terminal artifact SHALL be written to
`results/experiment_4940_distributional_energy_verifier_executable_spec.json`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4940 section
and SHALL NOT modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_executable_spec_ready.
- `arxiv_ids_cited`: 2605.18871 (Distributional EBM) + 2504.16828 (THINKPRM) + 2502.01989 (VFScale) -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the scaffold + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records scaffold/slice/network checks; a missing scaffold emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, design spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4940-EXECUTABLE-DRY-RUN: Three Columns Wire End-to-End

**Given** the Exp 4922 scaffold, scaffold artifact, and TravelPlanner structured
slice are present and self-consistency is below the saturation threshold
**When** `python/carnot/experiment_4940_distributional_energy_verifier_executable_spec.py`
runs
**Then** it writes
`results/experiment_4940_distributional_energy_verifier_executable_spec.json`
with `honest_verdict=success_distributional_energy_verifier_pivot_executable_spec_ready`,
`pivot_executable_on_7_1=true`, `three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, and dry-run columns `{self_consistency,
decomposed_energy_verifier, oracle}` over a non-saturated TravelPlanner slice.

### SCENARIO-KONA-4940-BLOCKED: Missing Scaffold or Slice Blocks Honestly

**Given** the Exp 4922 scaffold, scaffold artifact, or structured slice is
missing or the slice is invalid or self-consistency-saturated
**When** the executable-spec precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false`, and does not
claim that the verifier moat is proven.

### SCENARIO-KONA-4940-NO-MOAT-CLAIM: Validation Gate Is Stated, Not Met

**Given** the SOTA mapping and dry-run wiring are present
**When** the Exp 4940 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct, and pass a
no-model-identity-shortcut check, while `moat_proven_claimed=false` and
`verifier_is_oracle=false` remain enforced.

### REQ-KONA-4951: Distributional Energy Verifier Turnkey Readiness

Carnot MUST advance the Exp 4940 executable design spec into a turnkey
post-2026-06-30 distributional-energy-verifier experiment artifact without
executing the real benchmark. The turnkey artifact SHALL re-confirm real arXiv
citations for `2605.18871`, `2504.16828`, and `2502.01989`; update the
SOTA-to-Carnot mapping with per-paper `strongest_method`,
`implementation_cost_over_current_stack`, and `pitfalls`; and mark the
decomposed-energy LoRA-ensemble over FoVer analytical penalties as the strongest
NEXT-milestone roadmap input while keeping ThinkPRM and VFScale as comparator or
ablation inputs.

The implementation SHALL live at
`python/carnot/experiment_4951_distributional_energy_verifier_turnkey.py`, the
terminal artifact SHALL be written to
`results/experiment_4951_distributional_energy_verifier_turnkey.json`, and the
research note SHALL live at
`docs/research-notes/distributional-energy-verifier-turnkey-20260629.md`. The
single documented entrypoint SHALL be
`.venv/bin/python python/carnot/experiment_4951_distributional_energy_verifier_turnkey.py`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4951 section,
write a human-readable research note, and SHALL NOT modify
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, or
`_bmad/traceability.md`.

The turnkey dry-run SHALL load a real small slice from
`data/experiment_4922_travelplanner_structured_slice.jsonl`, confirm the chosen
SC-not-saturated domain has `self_consistency_saturated=false`, and wire exactly
three columns over that small slice: `self_consistency`,
`decomposed_energy_verifier`, and `oracle`. The
`decomposed_energy_verifier` column MUST use Carnot's active FoVer verifier
ensemble as the analytical-penalty source, represented by cached deterministic
constraint penalties in the structured slice, plus a learned quality-scorer
ensemble stub whose MEAN ranks candidates and whose STDDEV drives abstention.
The verifier column MUST ignore model identity and MUST NOT use the cached
oracle labels used only by the `oracle` column.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_turnkey_ready.
- `arxiv_ids_cited`: 2605.18871 (Distributional EBM) + 2504.16828 (THINKPRM) + 2502.01989 (VFScale) -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `pivot_turnkey`: true -- the post-6/30 experiment is ONE documented command away (real loader + dry-run + entrypoint), not just an executable spec.
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `post_sprint_first_experiment_pointer`: the single documented entrypoint + the pre-staged post-6/30 first-experiment so the loop pivots cleanly 7/1.
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the spec + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records spec/slice/network checks; a missing spec emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, turnkey spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4951-TURNKEY-DRY-RUN: Three Columns and Entrypoint Are Ready

**Given** the Exp 4940 executable spec artifact, Exp 4922 harness, FoVer registry,
and TravelPlanner structured slice are present and the slice is not
self-consistency-saturated
**When** `python/carnot/experiment_4951_distributional_energy_verifier_turnkey.py`
runs
**Then** it writes
`results/experiment_4951_distributional_energy_verifier_turnkey.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_turnkey_ready`,
`pivot_executable_on_7_1=true`, `pivot_turnkey=true`,
`three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, the documented one-command entrypoint, and dry-run
columns `{self_consistency, decomposed_energy_verifier, oracle}` over a small
non-saturated TravelPlanner slice.

### SCENARIO-KONA-4951-BLOCKED: Missing Turnkey Preconditions Block Honestly

**Given** the Exp 4940 executable spec artifact, Exp 4922 harness, FoVer
registry, or structured slice is missing, invalid, or self-consistency-saturated
**When** the turnkey precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false` and
`pivot_turnkey=false`, and does not claim that the verifier moat is proven.

### SCENARIO-KONA-4951-VALIDATION-GATE: Readiness Does Not Claim a Moat Win

**Given** the SOTA mapping, turnkey dry-run, and post-sprint entrypoint are
present
**When** the Exp 4951 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct, avoid a
model-identity shortcut, and evaluate a domain where self-consistency is not
near-ceiling, while `moat_proven_claimed=false`, `verifier_is_oracle=false`, and
`no_real_benchmark_run=true` remain enforced.

### REQ-KONA-4962: Distributional Energy Verifier Turnkey Backlog Extension

Carnot MUST keep the Exp 4951 distributional-energy-verifier pivot turnkey
while extending the post-2026-06-30 SOTA backlog without executing the real
benchmark. The artifact SHALL cite the NEW arXiv papers `2508.16665`,
`2508.10539`, and `2502.11157`; SHALL re-confirm the already-ingested
`2605.18871`, `2504.16828`, and `2502.01989`; and SHALL map every cited paper
onto Carnot with `strongest_method`, `implementation_cost_over_current_stack`,
and `pitfalls`.

The implementation SHALL live at
`python/carnot/experiment_4962_distributional_energy_verifier_turnkey.py`, and
the terminal artifact SHALL be written to
`results/experiment_4962_distributional_energy_verifier_turnkey.json`. The
single documented entrypoint SHALL be
`.venv/bin/python python/carnot/experiment_4962_distributional_energy_verifier_turnkey.py`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4962 section
and SHALL NOT modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

The turnkey re-confirmation SHALL load
`data/experiment_4922_travelplanner_structured_slice.jsonl`, verify
`self_consistency_saturated=false`, and run the three-column dry-run over a
small non-saturated slice with columns `self_consistency`,
`decomposed_energy_verifier`, and `oracle`. The
`decomposed_energy_verifier` column MUST remain oracle-distinct: it uses the
active FoVer verifier ensemble analytical penalties plus a learned
quality-scorer ensemble stub whose MEAN ranks candidates and whose STDDEV
drives abstention; it MUST NOT use model identity or cached oracle labels.

The survey paper `2508.16665` SHALL position Carnot's current design cell as a
discriminative, decomposed-energy verifier for outcome ranking with analytical
constraint penalties and abstention/efficiency controls. Adjacent open cells
SHALL include generative process verification, low-cost value-process variance
reduction, and fast/slow dynamic process-verifier routing. `2508.10539` SHALL
be marked as an uncertainty-signal refinement candidate for the ensemble-STDDEV
regenerate/abstain loop, and `2502.11157` SHALL be marked as an
efficiency-parity candidate for the Meta-EBM Cascade Router shape.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_turnkey_backlog_extended.
- `arxiv_ids_cited`: NEW: 2508.16665 + 2508.10539 + 2502.11157; re-confirmed: 2605.18871 + 2504.16828 + 2502.01989 -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `pivot_turnkey`: true -- the post-6/30 experiment is STILL ONE documented command away (real loader + dry-run + entrypoint re-confirmed).
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `post_sprint_first_experiment_pointer`: the single documented entrypoint + the pre-staged post-6/30 first-experiment so the loop pivots cleanly 7/1.
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the spec + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records spec/slice/network checks; a missing spec emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, turnkey spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4962-TURNKEY-BACKLOG: Backlog and Three Columns Stay Ready

**Given** the Exp 4951 turnkey artifact, Exp 4922 harness, FoVer registry,
phase3-kona spec, and TravelPlanner structured slice are present and the slice
is not self-consistency-saturated
**When** `python/carnot/experiment_4962_distributional_energy_verifier_turnkey.py`
runs
**Then** it writes
`results/experiment_4962_distributional_energy_verifier_turnkey.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_turnkey_backlog_extended`,
`pivot_executable_on_7_1=true`, `pivot_turnkey=true`,
`three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, all six real arXiv IDs, the documented one-command
entrypoint, and dry-run columns `{self_consistency,
decomposed_energy_verifier, oracle}` over a small non-saturated TravelPlanner
slice.

### SCENARIO-KONA-4962-BLOCKED: Missing Turnkey Preconditions Block Honestly

**Given** the phase3-kona spec, Exp 4951 turnkey artifact, Exp 4922 harness,
FoVer registry, or structured slice is missing, invalid, or
self-consistency-saturated
**When** the Exp 4962 precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false` and
`pivot_turnkey=false`, and does not claim that the verifier moat is proven.

### SCENARIO-KONA-4962-VALIDATION-GATE: Extended Backlog Does Not Claim a Moat Win

**Given** the six-paper SOTA mapping, taxonomy position, turnkey dry-run, and
post-sprint entrypoint are present
**When** the Exp 4962 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct, avoid a
model-identity shortcut, and evaluate a domain where self-consistency is not
near-ceiling, while `moat_proven_claimed=false`, `verifier_is_oracle=false`, and
`no_real_benchmark_run=true` remain enforced.

### REQ-KONA-4973: Distributional Energy Verifier Turnkey Backlog Extension V458

Carnot MUST keep the Exp 4962 distributional-energy-verifier pivot turnkey
while extending the post-2026-06-30 SOTA backlog with the `.458` verifier-moat
inputs without executing the real benchmark. The artifact SHALL cite the NEW
arXiv papers `2504.01005`, `2504.00891`, and `2509.24460`; SHALL re-confirm
the already-ingested `2605.18871`, `2504.16828`, `2502.01989`, `2508.16665`,
`2508.10539`, and `2502.11157`; and SHALL map every cited paper onto Carnot
with `strongest_method`, `implementation_cost_over_current_stack`, and
`pitfalls`.

The implementation SHALL live at
`python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py`, and
the terminal artifact SHALL be written to
`results/experiment_4973_distributional_energy_verifier_turnkey.json`. The
single documented entrypoint SHALL be
`.venv/bin/python python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4973 section
and SHALL NOT modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

The turnkey re-confirmation SHALL load
`data/experiment_4922_travelplanner_structured_slice.jsonl`, verify
`self_consistency_saturated=false`, and run the three-column dry-run over a
small non-saturated slice with columns `self_consistency`,
`decomposed_energy_verifier`, and `oracle`. The
`decomposed_energy_verifier` column MUST remain oracle-distinct: it uses the
active FoVer verifier ensemble analytical penalties plus a learned
quality-scorer ensemble stub whose MEAN ranks candidates and whose STDDEV
drives abstention; it MUST NOT use model identity or cached oracle labels.

The paper `2504.01005` SHALL define the efficiency-parity frontier for the
head-to-head: under a fixed inference budget, compare spending on additional
self-consistency samples versus fewer samples plus generative verification.
The paper `2504.00891` SHALL be marked as the matched-compute generative
process-verifier comparator for the decomposed-energy verifier. The paper
`2509.24460` SHALL be marked as the cross-domain generalization comparator for
the verifier-registry domain-expansion program in `ops/verifier_registry.yaml`
and `ops/verifier_gaps.md`.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_turnkey_backlog_extended.
- `arxiv_ids_cited`: NEW: 2504.01005 + 2504.00891 + 2509.24460; re-confirmed: 2605.18871 + 2504.16828 + 2502.01989 + 2508.16665 + 2508.10539 + 2502.11157 -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `pivot_turnkey`: true -- the post-6/30 experiment is STILL ONE documented command away (real loader + dry-run + entrypoint re-confirmed).
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `post_sprint_first_experiment_pointer`: the single documented entrypoint + the pre-staged post-6/30 first-experiment so the loop pivots cleanly 7/1.
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the spec + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records spec/slice/network checks; a missing spec emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, turnkey spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4973-TURNKEY-BACKLOG: V458 Backlog and Three Columns Stay Ready

**Given** the Exp 4962 turnkey artifact, Exp 4922 harness, FoVer registry,
phase3-kona spec, and TravelPlanner structured slice are present and the slice
is not self-consistency-saturated
**When** `python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py`
runs
**Then** it writes
`results/experiment_4973_distributional_energy_verifier_turnkey.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_turnkey_backlog_extended`,
`pivot_executable_on_7_1=true`, `pivot_turnkey=true`,
`three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, all nine real arXiv IDs, the documented
one-command entrypoint, and dry-run columns `{self_consistency,
decomposed_energy_verifier, oracle}` over a small non-saturated TravelPlanner
slice.

### SCENARIO-KONA-4973-BLOCKED: Missing V458 Preconditions Block Honestly

**Given** the phase3-kona spec, Exp 4962 turnkey artifact, Exp 4922 harness,
FoVer registry, or structured slice is missing, invalid, or
self-consistency-saturated
**When** the Exp 4973 precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false` and
`pivot_turnkey=false`, and does not claim that the verifier moat is proven.

### SCENARIO-KONA-4973-VALIDATION-GATE: V458 Backlog Does Not Claim a Moat Win

**Given** the nine-paper SOTA mapping, roadmap inputs, turnkey dry-run, and
post-sprint entrypoint are present
**When** the Exp 4973 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct, avoid a
model-identity shortcut, and evaluate a domain where self-consistency is not
near-ceiling, while `moat_proven_claimed=false`, `verifier_is_oracle=false`, and
`no_real_benchmark_run=true` remain enforced.

### REQ-KONA-4984: Distributional Energy Verifier Turnkey Backlog Extension V459

Carnot MUST keep the Exp 4973 distributional-energy-verifier pivot turnkey
while extending the post-2026-06-30 SOTA backlog with the `.459` verifier-moat
inputs without executing the real benchmark. The artifact SHALL cite the NEW
arXiv papers `2510.14913` and `2603.04304`; SHALL re-confirm the
already-ingested `2605.18871`, `2504.16828`, `2502.01989`, `2508.16665`,
`2508.10539`, `2502.11157`, `2504.01005`, `2504.00891`, and `2509.24460`;
and SHALL map every cited paper onto Carnot with `strongest_method`,
`implementation_cost_over_current_stack`, and `pitfalls`.

The implementation SHALL live at
`python/carnot/experiment_4984_distributional_energy_verifier_turnkey.py`, and
the terminal artifact SHALL be written to
`results/experiment_4984_distributional_energy_verifier_turnkey.json`. The
single documented entrypoint SHALL be
`.venv/bin/python python/carnot/experiment_4984_distributional_energy_verifier_turnkey.py`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4984 section
and SHALL NOT modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

The turnkey re-confirmation SHALL load
`data/experiment_4922_travelplanner_structured_slice.jsonl`, verify
`self_consistency_saturated=false`, and run the three-column dry-run over a
small non-saturated slice with columns `self_consistency`,
`decomposed_energy_verifier`, and `oracle`. The
`decomposed_energy_verifier` column MUST remain oracle-distinct: it uses the
active FoVer verifier ensemble analytical penalties plus a learned
quality-scorer ensemble stub whose MEAN ranks candidates and whose STDDEV
drives abstention; it MUST NOT use model identity or cached oracle labels.

The paper `2510.14913` SHALL define the discriminative verifier-under-budget
comparator: Carnot's decomposed-energy verifier is itself a discriminative
quality scorer plus deterministic penalties, so the post-6/30 head-to-head MUST
report against this matched-compute discriminative budget frontier and also
against the generative frontier from `2504.00891` and `2504.16828`. The paper
`2603.04304` SHALL be marked as the unify-generation-and-self-verification
comparator for the regenerate/abstain two-pass design: use uncertainty-guided
verification to decide which candidates need additional compute, targeted
regeneration, or abstention.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_turnkey_backlog_extended.
- `arxiv_ids_cited`: NEW: 2510.14913 + 2603.04304; re-confirmed: 2605.18871 + 2504.16828 + 2502.01989 + 2508.16665 + 2508.10539 + 2502.11157 + 2504.01005 + 2504.00891 + 2509.24460 -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `pivot_turnkey`: true -- the post-6/30 experiment is STILL ONE documented command away (real loader + dry-run + entrypoint re-confirmed).
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `post_sprint_first_experiment_pointer`: the single documented entrypoint + the pre-staged post-6/30 first-experiment so the loop pivots cleanly 7/1.
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the spec + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records spec/slice/network checks; a missing spec emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, turnkey spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4984-TURNKEY-BACKLOG: V459 Backlog and Three Columns Stay Ready

**Given** the Exp 4973 turnkey artifact, Exp 4962 turnkey module, Exp 4922
harness, FoVer registry, phase3-kona spec, and TravelPlanner structured slice
are present and the slice is not self-consistency-saturated
**When** `python/carnot/experiment_4984_distributional_energy_verifier_turnkey.py`
runs
**Then** it writes
`results/experiment_4984_distributional_energy_verifier_turnkey.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_turnkey_backlog_extended`,
`pivot_executable_on_7_1=true`, `pivot_turnkey=true`,
`three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, all eleven real arXiv IDs, the documented
one-command entrypoint, and dry-run columns `{self_consistency,
decomposed_energy_verifier, oracle}` over a small non-saturated TravelPlanner
slice.

### SCENARIO-KONA-4984-BLOCKED: Missing V459 Preconditions Block Honestly

**Given** the phase3-kona spec, Exp 4973 turnkey artifact, Exp 4962 turnkey
module, Exp 4922 harness, FoVer registry, or structured slice is missing,
invalid, or self-consistency-saturated
**When** the Exp 4984 precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false` and
`pivot_turnkey=false`, and does not claim that the verifier moat is proven.

### SCENARIO-KONA-4984-VALIDATION-GATE: V459 Backlog Does Not Claim a Moat Win

**Given** the eleven-paper SOTA mapping, roadmap inputs, turnkey dry-run, and
post-sprint entrypoint are present
**When** the Exp 4984 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct, avoid a
model-identity shortcut, and evaluate a domain where self-consistency is not
near-ceiling, while `moat_proven_claimed=false`, `verifier_is_oracle=false`, and
`no_real_benchmark_run=true` remain enforced.

### REQ-KONA-4995: Distributional Energy Verifier Turnkey Backlog Extension V460

Carnot MUST keep the Exp 4984 distributional-energy-verifier pivot turnkey
while extending the post-2026-06-30 SOTA backlog with the `.460` verifier-moat
inputs without executing the real benchmark. The artifact SHALL cite the NEW
arXiv papers `2504.13134` and `2605.10158`; SHALL re-confirm the
already-ingested `2605.18871`, `2504.16828`, `2502.01989`, `2508.16665`,
`2508.10539`, `2502.11157`, `2504.01005`, `2504.00891`, `2509.24460`,
`2510.14913`, and `2603.04304`; and SHALL map every cited paper onto Carnot
with `strongest_method`, `implementation_cost_over_current_stack`, and
`pitfalls`.

The implementation SHALL live at
`python/carnot/experiment_4995_distributional_energy_verifier_turnkey.py`, and
the terminal artifact SHALL be written to
`results/experiment_4995_distributional_energy_verifier_turnkey.json`. The
single documented entrypoint SHALL be
`.venv/bin/python python/carnot/experiment_4995_distributional_energy_verifier_turnkey.py`.
The run SHALL update `research-studying.md` with an INGESTED Exp 4995 section
and SHALL NOT modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

The turnkey re-confirmation SHALL load
`data/experiment_4922_travelplanner_structured_slice.jsonl`, verify
`self_consistency_saturated=false`, and run the three-column dry-run over a
small non-saturated slice with columns `self_consistency`,
`decomposed_energy_verifier`, and `oracle`. The
`decomposed_energy_verifier` column MUST remain oracle-distinct: it uses the
active FoVer verifier ensemble analytical penalties plus a learned
quality-scorer ensemble stub whose MEAN ranks candidates and whose STDDEV
drives abstention; it MUST NOT use model identity or cached oracle labels.

The paper `2504.13134` SHALL define the EBRM sibling/foundation comparator for
the learned-quality-scorer half: an energy-based post-hoc reward-model
refinement that models the reward distribution explicitly rather than reducing
it to one scalar, preserving uncertainty for robust alignment. Carnot SHALL
map it as an adoptable distribution-modeling head and comparator, while
recording that it is an alignment reward model rather than a per-step process
verifier and can inherit blind spots from the base reward model. The paper
`2605.10158` SHALL define the uPRM cheap-discriminative efficiency frontier:
an unsupervised process reward model trained from generator next-token
probabilities, comparable to supervised PRMs and beating majority voting by up
to 6.9%. Carnot SHALL map it as a strong cheap baseline/comparator, while
recording the model-identity-shortcut risk from deriving signal from the
generator's own next-token probabilities.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_distributional_energy_verifier_pivot_turnkey_backlog_extended.
- `arxiv_ids_cited`: NEW: 2504.13134 + 2605.10158; re-confirmed: 2605.18871 + 2504.16828 + 2502.01989 + 2508.16665 + 2508.10539 + 2502.11157 + 2504.01005 + 2504.00891 + 2509.24460 + 2510.14913 + 2603.04304 -- real IDs, no fabrication (SOTA-ingestion guardrail).
- `sota_to_carnot_mapping`: per-paper {strongest_method, implementation_cost_over_current_stack, pitfalls} -- the ingestion deliverable that feeds the post-6/30 roadmap.
- `pivot_executable_on_7_1`: true -- the distributional-energy-verifier experiment runs the instant the sprint retires (the readiness deliverable).
- `pivot_turnkey`: true -- the post-6/30 experiment is STILL ONE documented command away (real loader + dry-run + entrypoint re-confirmed).
- `three_column_dry_run_ok`: the self-consistency / decomposed-energy-verifier / oracle columns wire end-to-end on a SC-not-saturated slice (no full benchmark run).
- `sc_not_saturated_domain`: the chosen domain (MuSR / TravelPlanner) where self-consistency is NOT near-ceiling -- the only place an oracle-distinct moat win is reachable (2605.18871 beats SC on MuSR).
- `post_sprint_first_experiment_pointer`: the single documented entrypoint + the pre-staged post-6/30 first-experiment so the loop pivots cleanly 7/1.
- `validation_gate`: the post-6/30 gate stated precisely: beats SC with CI95 excluding zero + oracle-distinct + no model-identity shortcut (NOT claimed met here).
- `verifier_is_oracle`: false -- the DESIGN TARGET is oracle-distinct (a learned/energy verifier, NOT the executable oracle that defines correctness); not a measured result here.
- `moat_proven_claimed`: false -- this is readiness/design + SOTA-ingestion; the real post-6/30 experiment must pass the gate.
- `inference_substrate`: aggregation_from_upstream_artifacts (reads the spec + slice + papers; 0.0001s floor) -- no real benchmark run.
- `preconditions_checked`: records spec/slice/network checks; a missing spec emits blocked_.
- `random_seed`: determinism for the dry-run wiring.
- `reproducibility_checksum`: content hash of (papers cited, turnkey spec, dry-run config) so a replication catches drift.

### SCENARIO-KONA-4995-TURNKEY-BACKLOG: V460 Backlog and Three Columns Stay Ready

**Given** the Exp 4984 turnkey artifact, Exp 4962 turnkey module, Exp 4922
harness, FoVer registry, phase3-kona spec, and TravelPlanner structured slice
are present and the slice is not self-consistency-saturated
**When** `python/carnot/experiment_4995_distributional_energy_verifier_turnkey.py`
runs
**Then** it writes
`results/experiment_4995_distributional_energy_verifier_turnkey.json` with
`honest_verdict=success_distributional_energy_verifier_pivot_turnkey_backlog_extended`,
`pivot_executable_on_7_1=true`, `pivot_turnkey=true`,
`three_column_dry_run_ok=true`, `verifier_is_oracle=false`,
`moat_proven_claimed=false`, all thirteen real arXiv IDs, the documented
one-command entrypoint, and dry-run columns `{self_consistency,
decomposed_energy_verifier, oracle}` over a small non-saturated TravelPlanner
slice.

### SCENARIO-KONA-4995-BLOCKED: Missing V460 Preconditions Block Honestly

**Given** the phase3-kona spec, Exp 4984 turnkey artifact, Exp 4962 turnkey
module, Exp 4922 harness, FoVer registry, or structured slice is missing,
invalid, or self-consistency-saturated
**When** the Exp 4995 precondition check runs
**Then** it emits a terminal `blocked_*` verdict, records the blocked resource in
`preconditions_checked`, keeps `pivot_executable_on_7_1=false` and
`pivot_turnkey=false`, records `net_available=false` without blocking if the
network is absent, and does not claim that the verifier moat is proven.

### SCENARIO-KONA-4995-VALIDATION-GATE: V460 Backlog Does Not Claim a Moat Win

**Given** the thirteen-paper SOTA mapping, roadmap inputs, turnkey dry-run, and
post-sprint entrypoint are present
**When** the Exp 4995 artifact is validated
**Then** `validation_gate` requires the real post-6/30 experiment to beat
self-consistency with CI95 excluding zero, remain oracle-distinct
(`verifier_is_oracle=false`), avoid a model-identity shortcut, and evaluate a
domain where self-consistency is not near-ceiling, while
`moat_proven_claimed=false`, `verifier_is_oracle=false`, and
`no_real_benchmark_run=true` remain enforced.

### REQ-KONA-5002: Shared Moat Benchmark Harness for Phase D Verifier Arms

Carnot MUST provide a reusable Phase D moat benchmark harness in
`python/carnot/moat_benchmark_harness.py` and a thin Exp 5002 artifact writer in
`python/carnot/experiment_5002_moat_benchmark_harness.py`. The harness SHALL be
a library, not a one-off script, so D1 LoRA-EBM, D2 uPRM, D3 EBRM, D4
cross-corpus, and D5 gate arms all measure the same target: oracle-distinct
selection accuracy versus tuned self-consistency on headroom-present candidate
pools with paired uncertainty.

The harness SHALL load normalized corpus rows with `{question, context,
choices, gold}` for MuSR/murder_mysteries, GPQA if cached, MMLU-Pro-hard if
cached, and MATH-500-hard if cached. Gold labels are for evaluation only. The
harness SHALL reuse cached MuSR candidate checkpoints from
`results/distributional_energy_verifier_musr_checkpoints/` and SHALL NOT
regenerate those smoke candidates. For future D arms that require fresh
candidates or per-token logprobs, it SHALL expose a generate-with-logprobs path
documented as `gemma-4-12B-it-GGUF` on GPU-0 CUDA while keeping the Exp 5002
smoke on cached candidates only.

For any verifier scorer `f(candidate) -> energy`, lower is better, the harness
SHALL compute tuned self-consistency accuracy by sweeping available K/temperature
settings, oracle@K as the selectable-headroom ceiling, headroom presence as
`(oracle_at_k - tuned_sc) >= 0.10 AND n_flips_possible > 0`, verifier selection
accuracy, paired bootstrap CI95(verifier - tuned self-consistency), and McNemar
p. The harness SHALL mechanically enforce oracle-distinctness by raising if a
scorer reads `gold`, `answer_index`, `answer_choice`, or `model_id`.

The terminal artifact SHALL be written to
`results/experiment_5002_moat_benchmark_harness.json`. If MuSR or the cached
candidate checkpoints are missing, Exp 5002 SHALL write a terminal
`blocked_<resource>` artifact rather than fabricate metrics. Exp 5002 SHALL NOT
modify `scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, or
`_bmad/traceability.md`.

Required artifact fields and principles:
- `honest_verdict`: terminal prefix; success_moat_harness_built_smoke_green.
- `harness_module_path`: python/carnot/moat_benchmark_harness.py -- the reusable library the D arms import (no duplicated metric code).
- `corpora_available`: the list of loadable headroom-candidate corpora (MuSR + the 2nd-corpus options) so D4 can pick a confirmed-cached second corpus.
- `tuned_sc_smoke`: the TUNED self-consistency accuracy on the smoke slice -- the baseline to beat is tuned, not naive SC (headroom-control).
- `oracle_at_k_smoke`: the selectable-headroom ceiling; (oracle@K - tuned_sc) is the headroom a verifier could capture.
- `headroom_present_smoke`: (oracle@K - tuned_sc) >= 0.10 AND flips>0 -- the FALSE_NEGATIVE_RISK guard a null must clear to be informative.
- `oracle_distinctness_enforced`: true -- the harness raises if a scorer reads gold/answer_index/model_id (verifier_is_oracle=False is mechanically enforced).
- `inference_substrate`: verifier_ensemble_against_cached_candidates (scores cached candidates; 1s floor) -- no new LLM generation in the smoke.
- `random_seed`: determinism for the bootstrap CI + the smoke.
- `preconditions_checked`: records corpus/candidate-cache checks; a missing corpus emits blocked_, never a fabricated metric.

### SCENARIO-KONA-5002-SMOKE: Cached MuSR Smoke Computes Shared Metrics

**Given** MuSR/murder_mysteries is locally loadable and
`results/distributional_energy_verifier_musr_checkpoints/` contains cached
candidate answer pools
**When** `.venv/bin/python python/carnot/experiment_5002_moat_benchmark_harness.py`
runs
**Then** it writes `results/experiment_5002_moat_benchmark_harness.json` with
`honest_verdict=success_moat_harness_built_smoke_green`, a smoke slice of at
most 30 cached MuSR rows, tuned self-consistency accuracy, oracle@K,
`headroom_present_smoke=true`, verifier accuracy for a trivial cached verifier,
paired CI95, McNemar p, and `oracle_distinctness_enforced=true`.

### SCENARIO-KONA-5002-ORACLE-DISTINCT: Forbidden Scorer Inputs Fail Closed

**Given** a verifier scorer attempts to read `gold`, `answer_index`,
`answer_choice`, or `model_id` from a candidate
**When** the shared harness scores a candidate pool
**Then** it raises an oracle-distinctness error before returning a metric,
ensuring verifier-arm code cannot accidentally become an answer-key or
model-identity oracle.

### SCENARIO-KONA-5002-BLOCKED: Missing Resources Block Honestly

**Given** MuSR is not locally loadable or the cached MuSR candidate checkpoint
directory is missing or empty
**When** the Exp 5002 precondition check runs
**Then** it writes a terminal `blocked_<resource>` artifact, records the failed
resource in `preconditions_checked`, keeps smoke metrics empty, and does not
claim that a headroom-present corpus was loaded.

### REQ-KONA-5015: Genuine Tuned Self-Consistency Baseline and Abstention Degeneracy Guard

Carnot MUST correct the shared Phase D moat harness in
`python/carnot/moat_benchmark_harness.py` so tuned self-consistency is measured
as genuine K-way majority vote over cached candidates, not as a default
single-sample strawman. The harness SHALL sweep odd K values `{1,3,5,7,...}` up
to the available cached candidates per question, SHALL report the full
`{K: accuracy}` curve, and SHALL expose the tuned best-K accuracy and chosen K
with deterministic answer tie-breaking. If only one candidate exists per
question, the harness SHALL flag that self-consistency and oracle@K are
degenerate rather than presenting headroom as informative.

The harness SHALL provide an abstention-degeneracy guard for D-arm selectors
that abstain back to tuned self-consistency. Given an `abstain_rate`, the guard
SHALL return a `degeneracy_flag=true` verdict when `abstain_rate > 0.50`, so a
selector that mostly delegates to tuned self-consistency cannot report an
uninformative delta as verifier evidence.

Exp 5015 SHALL write
`results/experiment_5015_genuine_sc_baseline_fix.json` from cached MuSR
candidates only. The artifact SHALL report `genuine_tuned_sc_accuracy`,
`sc_k_sweep`, `tuned_k`, `candidates_per_question`, `oracle_at_k`,
`genuine_headroom_present`, and `degeneracy_guard_fires`, with precondition
fields that block honestly if the candidate cache or harness import is missing.

### SCENARIO-KONA-5015-GENUINE-SC: K-Way Self-Consistency Is Auditable

**Given** a candidate pool with multiple cached answers per question
**When** the shared harness computes tuned self-consistency
**Then** it evaluates odd K-way majority votes, reports the full K-sweep,
chooses the best K deterministically, and recomputes headroom against that
genuine tuned self-consistency baseline.

### SCENARIO-KONA-5015-DEGENERACY-GUARD: Mostly-Abstaining Selectors Are Flagged

**Given** a D-arm selector whose abstain rate is greater than 0.50
**When** the shared harness abstention-degeneracy guard is called
**Then** it returns a verdict and `degeneracy_flag=true`, preventing a mostly
tuned-SC selector from being treated as an informative verifier.

### SCENARIO-KONA-5015-SMOKE: Cached MuSR Baseline Fix Is Re-Emitted

**Given** the cached MuSR checkpoint slice exists with at most 200 questions
**When** `.venv/bin/python python/carnot/experiment_5015_genuine_sc_baseline_fix.py`
runs
**Then** it writes `results/experiment_5015_genuine_sc_baseline_fix.json` with
the genuine tuned-SC K-sweep, oracle@K, genuine headroom decision, and a
synthetic always-abstain degeneracy-guard demonstration.

## Latent Symbol Bridge Falsification (Exp 3819)

### REQ-3819: Deep Think P3 Falsification Run
The system shall execute an off-the-shelf Tiny Recursive Model (TRM) and pass its intermediate latent states through a programmatic verifier to test the hypothesis that intermediate latents decode to gibberish and cannot provide a useful Q-head signal in-loop. It MUST write `honest_verdict: "blocked_trm_checkpoint_not_available"` if the required checkpoint is not available and bounded tiny-train is infeasible under 20 minutes.

### SCENARIO-3819: Preconditions Falsification Gate Fast-Fails
**Given** an environment without a pretrained TRM checkpoint
**When** the Latent Symbol Bridge experiment is initiated
**Then** the experiment gracefully handles the missing resource by emitting `honest_verdict: "blocked_trm_checkpoint_not_available"` and sets corresponding metrics to 0 or defaults.

## TRM Curl-Free Falsification (Exp 3823)

### REQ-3823: TRM Update Field Curl-Free Diagnostic
The system shall provide an Exp 3823 workflow that tests whether a trained Tiny Recursive Model (TRM) update field is expressible as conservative scalar-energy descent. The workflow SHALL read `results/experiment_3821_latent_symbol_bridge_unblocked.json`, extract `trm_checkpoint_source`, verify that Python can import both `torch` and `numpy`, and verify that the checkpoint source is loadable before attempting any TRM inference. If any precondition fails, the workflow SHALL write `results/experiment_3823_trm_not_ebt_curlfree.json` with `honest_verdict: "blocked_trm_checkpoint_not_available"` and explicit precondition evidence rather than fabricating TRM curl metrics.

When the trained TRM checkpoint is loadable, the workflow SHALL run the update model on at least 50 seeded latent-state instances, collect `(h_t, h_{t+1}-h_t)` pairs across refinement steps, estimate the antisymmetric fraction of the update Jacobian, fit a scalar potential whose negative gradient approximates the observed update field, and run the same scalar-potential fit on a known conservative positive-control field. The terminal artifact SHALL include `jacobian_antisymmetry_fraction`, `scalar_potential_fit_residual`, `positive_control_fit_residual`, `n_states_sampled`, `preconditions_checked`, `inference_substrate`, `random_seed`, `reproducibility_checksum`, and `duration_s`, and every required field SHALL include a methodology `principle` explaining why it exists.

The terminal verdict SHALL be one of `complete: trm_not_ebt_curlfree_falsified_asymmetric_field_bounded_ebt_does_not_cover_trm`, `complete: trm_is_secretly_energy_descent_surprising_residual<X>`, `complete: INCONCLUSIVE_curlfree_positive_control_failed`, or `blocked_trm_checkpoint_not_available`. A non-EBT verdict SHALL require non-negligible Jacobian antisymmetry, a large scalar-potential residual on TRM updates, and a passing positive-control residual; an energy-descent verdict SHALL require a low TRM scalar-potential residual and a passing positive control; an inconclusive verdict SHALL be emitted if the positive control fails.

### SCENARIO-3823: Missing TRM Checkpoint Blocks Curl-Free Claim
**Given** the Exp 3821 artifact does not provide a local loadable TRM checkpoint source
**When** the Exp 3823 curl-free diagnostic runs
**Then** it writes the terminal Exp 3823 JSON with `honest_verdict: "blocked_trm_checkpoint_not_available"`, records `n_states_sampled` as zero, preserves the exact checkpoint-source evidence from Exp 3821, and keeps every required metric principle-bearing instead of claiming a curl or scalar-potential result.

### SCENARIO-3823-POSITIVE-CONTROL: Conservative Field Fit Succeeds
**Given** a synthetic update field generated by the negative gradient of a scalar quadratic energy
**When** the Exp 3823 scalar-potential fitting helper is applied
**Then** the positive-control residual is near zero, proving the diagnostic can recognize conservative fields and that any large TRM residual is not caused by a broken fitter.

## Offline Distillation Oracle Q-Head (Exp 3825)

### REQ-3825: Continuous Q-Head Offline Distillation Prototype
The system shall provide an Exp 3825 workflow that tests the mechanism feasibility of
distilling a continuous learned Q-head from Carnot verifier-oracle labels over
unrolled recursive-refiner latent trajectories. The workflow SHALL first check that
`torch` imports, that `results/experiment_3824_headroom_gate_corpus.json` exists and
has `headroom_confirmed == true`, and that a recursive-refiner inference substrate can
be loaded. If the headroom gate is closed, the workflow SHALL write
`results/experiment_3825_distillation_oracle_qhead.json` with
`honest_verdict: "complete: distillation_skipped_headroom_not_confirmed"` and exit
without training. If any non-headroom precondition fails, the workflow SHALL write a
terminal `blocked_*` verdict with explicit precondition evidence.

When preconditions pass, the workflow SHALL unroll the recursive-refiner substrate over
the Exp 3824 corpus, capture each trajectory's final state and full latent sequence,
label each final trajectory with a programmatic verifier oracle, train a small
continuous Q-head directly on per-step continuous latents, evaluate predictive AUROC on
an honest held-out trajectory split, compute an adversarial-control AUROC with
test-time-compute and identity-conditioning features ablated, and report a per-step
calibration curve that shows whether Q-head scores for correct trajectories rise
monotonically across refinement steps. The artifact SHALL include principle-bearing
fields for `qhead_heldout_auroc`, `qhead_ablated_auroc`,
`per_step_calibration_monotonic`, `n_train_trajectories`,
`n_heldout_trajectories`, `verifier_oracle_label_source`,
`preconditions_checked`, `inference_substrate`, `random_seed`,
`reproducibility_checksum`, and `duration_s`.

The terminal verdict SHALL start with one of:
`complete: distillation_oracle_qhead_feasible_auroc<X>_ablated<Y>_calibration_monotonic`,
`complete: distillation_oracle_qhead_bounded_no_signal_auroc<X>`,
`complete: distillation_skipped_headroom_not_confirmed`, or `blocked_*`.
The feasible verdict SHALL require ablated AUROC materially above 0.5 and a monotonic
calibration curve; the bounded verdict SHALL be used when the ablated AUROC is near
chance or calibration is not monotonic.

### SCENARIO-3825-SKIP: Headroom Gate Closed Skips Distillation
**Given** the Exp 3824 headroom artifact exists but reports `headroom_confirmed == false`
**When** the Exp 3825 Q-head distillation workflow runs
**Then** it writes `results/experiment_3825_distillation_oracle_qhead.json` with
`honest_verdict: "complete: distillation_skipped_headroom_not_confirmed"`, records the
closed-gate precondition evidence, and reports zero train and held-out trajectories
with principle-bearing fields.

### SCENARIO-3825-TRAIN: Held-Out Q-Head Metrics Are Reported
**Given** the Exp 3824 headroom gate is open, the corpus has enough trajectories for a
train/held-out split, and a recursive-refiner substrate loads
**When** the Exp 3825 workflow unrolls trajectories and trains the Q-head
**Then** the terminal artifact reports held-out AUROC, ablated AUROC, train and
held-out trajectory counts, a per-step calibration curve, the verifier-oracle label
source, reproducibility metadata, and every required metric field includes its
methodology principle.

### SCENARIO-3825-ABLATION: Identity And Step Features Cannot Explain The Signal
**Given** a held-out trajectory set with verifier labels and continuous latent
sequences
**When** the Exp 3825 adversarial control evaluates Q-head scores using only the
continuous latent dimensions and excludes trajectory identity and step-index features
**Then** the ablated AUROC is computed from those Q-head scores and is reported
separately from the ordinary held-out AUROC.
