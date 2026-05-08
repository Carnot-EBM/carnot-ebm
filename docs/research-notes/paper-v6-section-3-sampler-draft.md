# Paper v6 Section 3 sampler/verifier draft after .120

**Status:** focused draft for Section 3 integration.
**Ready for exp1579 OT framework integration: yes**, with the caveat that
exp1579 still has to supply the final OT notation, Youden/J calibration, and
attack-compute formalism. This note is intended as sampler/verifier prose to
be integrated with that framework, not as a wholesale rewrite of the paper.

## THRML vendored sampler

Draft text:

Carnot's inference sampler is no longer a local reimplementation of the
reference process. After the .120 sampler audit, the implementation target is
Extropic THRML 0.1.3, vendored under Apache-2.0, with Carnot invoking the
THRML block-Gibbs transition operator rather than approximating it with
single-site Metropolis-Hastings or a differentiable MCMC layer. The evidence
record is `results/experiment_1564_thrml_vendored_block_gibbs_replacement.json`:
`thrml_vendoring_complete = true`, `thrml_version = "0.1.3"`,
`thrml_license = "Apache-2.0"`, and `kl_to_thrml_after_vendoring = 0.0`.
This resolves the prior finite-K sampler-mismatch problem by making the
reference sampler and Carnot sampler the same software transition.

The paper should keep the claim narrow. This is a simulator/software
alignment claim only: `simulator_only = true`, `no_tsu_hardware_claim = true`,
and the result does not claim Extropic hardware execution. It also does not
claim THRML security parity; that question is handled separately in the
kinetic-security caveat below.

Suggested paper wording:

> We use the vendored THRML 0.1.3 block-Gibbs sampler as Carnot's reference
> inference sampler. This makes the finite-K sampler target constructive:
> Carnot follows THRML's transition operator directly, rather than replacing it
> with a distinct MH kernel whose transient distribution can differ at the
> latency budgets used in deployment. This is a software-simulator alignment
> claim, not a claim of Extropic hardware execution.

## Candidate warm-start

Draft text:

The inference chain should initialize at the candidate being verified, not at a
uniform cold start and not from a cached state from a previous prompt. The
candidate is already prompt-conditioned and structurally localized, so it is
the only warm-start state compatible with Carnot's stateless verifier API.

The .120 benchmark in
`results/experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json`
supports this design choice: `candidate_warm_start_validated = true`,
`recommended_deployment_policy = "candidate_warm_start"`, and
`cached_state_worse_than_cold_start = true`. At K=100, candidate warm-start
accuracy was 1.0, cold-start accuracy was 0.465, and cached-state warm-start
accuracy was 0.45; the recorded `cold_start_accuracy_drop_percent_at_k100` is
51.052632.

Suggested paper wording:

> Carnot initializes the THRML block-Gibbs chain at the candidate under
> evaluation. This exploits the verifier API contract: the input candidate is
> already a prompt-conditioned proxy for the local target mode. In the .120
> benchmark, candidate warm-start reached 1.0 accuracy at K=100 while uniform
> cold-start and cached cross-prompt state reached only 0.465 and 0.45,
> respectively. Cached state is therefore rejected as a deployment pattern,
> because a state from a different prompt behaves like an adversarial
> initialization rather than a useful persistent chain.

## Soft-Gibbs Residual

Draft text:

Paper-v6 should not import the OT framework's hard rejection sampler as-is.
For Carnot's AND-composed verifier stack, hard conditioning on the exact
intersection can become operationally empty even when the algebraic residual
identity remains valid. The .120 implementation in
`results/experiment_1565_soft_gibbs_residual_implementation.json` found
`hard_brs_acceptance_rate = 0.0` on the contradictory-verifier fixture while
`soft_gibbs_residual_implemented = true` and `soft_brs_decay_confirmed = true`.

The replacement residual is:

```text
V(y) = number of failed verifiers
mu_res^beta(y) proportional to mu(y) * exp(-beta * V(y))
Z_beta = E_mu[exp(-beta * V(y))]
```

This keeps rejection-style sampling implementable while degrading smoothly to
minimum-violation states when the strict verifier intersection is empty.
The coverage follow-up in
`results/experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json`
records `jensen_bound_holds_for_all_beta = true`,
`optimal_beta_for_deployment = 0.1`, and an empirical acceptance rate of
0.9267369803014833 at beta=0.1 versus the Jensen lower-bound objective
0.9183592117771301. Paper-v6 should disambiguate this residual beta from any
OT coverage beta used by exp1579.

Suggested paper wording:

> Carnot replaces hard residual rejection with a Soft-Gibbs Residual,
> `mu_res^beta(y) proportional to mu(y) exp(-beta V(y))`, where `V(y)` counts
> failed verifiers. This preserves constant-cost rejection-style sampling while
> avoiding infinite rejection loops under contradictory or near-empty
> AND-composed verifier intersections. Empirically, hard BRS had zero
> acceptance on the .120 contradictory-verifier fixture, while the soft
> residual decayed as predicted and satisfied the Jensen coverage bound for
> every tested beta.

## Kinetic-security caveat

Draft text:

The sampler-security story must be revised downward. Earlier prose treated
Gibbs-class plateau friction as a potential kinetic defense-in-depth against
joint-null-space search. Exp 1561 directly falsified security parity for the
vendored THRML block-Gibbs path on the zero-coupling predicate:
`results/experiment_1561_kinetic_defense_zero_coupling_test.json` reports
`kinetic_defense_in_depth_validated = false`,
`thrml_security_parity_with_single_site_gibbs = false`,
`thrml_hits_at_mh_class_rate = true`, and honest verdict
`complete_thrml_block_gibbs_falsifies_kinetic_security_parity`.

The measured hitting-time fields are the important numbers:
`thrml_block_gibbs_hitting_time_steps_per_block = 14.9182`,
`single_site_gibbs_hitting_time_steps_per_block = 28.093`, and
`mh_hitting_time_steps_per_block = 21.3348`. On this fixture, THRML reaches the
planted null space faster than single-site Gibbs and at MH-class rate. The
paper therefore does not claim THRML security parity. Kinetic defense-in-depth remains an unresolved sampler-security question that needs a
separate throttle/audit story before it can be promoted from caveat to
defense.

Suggested paper wording:

> Vendoring THRML fixes the reference-sampler mismatch, but it does not by
> itself provide the kinetic security property hypothesized for slower
> single-site Gibbs. In the .120 zero-coupling test, THRML block-Gibbs reached
> the planted null space faster than single-site Gibbs and at an MH-class rate.
> We therefore treat kinetic defense-in-depth as an open security caveat:
> THRML is adopted for sampler fidelity, while null-space exposure under
> block-parallel scheduling remains subject to explicit throttling and audit.

## SpecAnn rejection

Draft text:

SpecAnn remains rejected for Phase 3. The architecture record
`results/experiment_1563_specann_rejection_architecture_record.json` is
complete and records that SpecAnn is rejected for Phase 3 inference-time
argmin while Carnot retains a Gibbs-heuristic path on unreduced HUBO energy.
This matches the Deep Think composition finding: Carnot's AND-composed
verifier energy is naturally higher-order, while SpecAnn requires a QUBO
reduction. The required gadget variables and large consistency penalties
distort the spectrum, shrink eigengaps, and create spurious gadget-satisfying
minima. The level-crossing brittleness also conflicts with a training regime
whose energy landscape can change sharply.

Suggested paper wording:

> Spectral Annealing is not used for Phase 3 inference-time argmin. Applying it
> to Carnot's verifier energy would first require reducing the unreduced HUBO
> objective to QUBO with auxiliary gadgets and large penalties. The .120
> architecture record treats that reduction as a structural failure mode, not a
> harmless implementation detail, because it changes the optimization
> landscape the verifier stack is meant to expose.

## BRAIN expressivity vs training-dynamics open question

Draft text:

The paper should not preserve the earlier BRAIN+Linear-AR rescue claim as a
Phase 3 adoption claim. The extended k-sweep in
`results/experiment_1562_brain_linear_ar_k_sweep_extended.json` records
`brain_linear_ar_rescue_validated = false`,
`phase_3_recommendation = "brain_dropped"`, `made_required_at_k15 = false`,
`best_parameterization_kl_at_k15 = 0.001336`, and
`factorized_vs_ar_ratio_at_k15 = 1.000749`. In this evidence, Linear-AR did
not widen the gap enough to justify BRAIN as the Phase 3 distribution learner,
and BRAIN-as-published is rejected for Phase 3.

There is still a narrower training-dynamics question, but it is not a Phase 3
sampler claim. `results/experiment_1571_step_wise_baseline_AR_REINFORCE.json`
records `step_wise_baseline_implemented = true`,
`gradient_variance_reduction_factor = 10.454576`, and
`convergence_rate_noisy_to_clean_ratio = 0.995218`. That supports the
step-wise baseline as an AR-REINFORCE variance-control mechanism, not the
broader claim that BRAIN should train the Phase 3 substrate. The training-dynamics question remains open for any future noisy-hardware or
separate-generator setting: can AR-REINFORCE with step-wise baselines improve a
real deployment objective without reintroducing the expressivity and
null-space problems that caused Phase 3 to drop BRAIN?

Suggested paper wording:

> BRAIN is not adopted as Carnot's Phase 3 distribution-learning mechanism.
> The .120 k-sweep falsified the need for the Linear-AR rescue at k=15 and
> records `phase_3_recommendation = brain_dropped`. A separate AR-REINFORCE
> training-dynamics result remains useful as a variance-control baseline, but
> it is not evidence that BRAIN has Phase 3 validation or analog
> hardware execution.

## Paper-v6 integration checklist

- Requested source note: `docs/papers/paper-v6/main.tex is not present` in
  this repository. The active local TeX source is `docs/arxiv-paper/main.tex`.
- Primary insertion point: add a new sampler/verifier subsection inside
  `docs/arxiv-paper/main.tex:756`, immediately after the section header
  `\section{Hardware Acceleration \& Sampling Limits}` and before unrelated
  hardware-speed claims take over the section.
- Replace or extend the legacy sampler audit bridge at
  `docs/arxiv-paper/main.tex:777`, currently
  `\subsection{The detailed-balance audit (exp1094)}`. The new text should
  preserve exp1094 as historical motivation, then state that .120 supersedes
  the sampler implementation path with THRML vendoring plus candidate
  warm-start.
- Keep the new subsection before `docs/arxiv-paper/main.tex:804`, currently
  `\subsection{Same-basis CPU-vs-FPGA timing remains open}`, so simulator
  fidelity, sampler initialization, and verifier-residual conditioning are
  separated from hardware-timing caveats.
- Add a short cross-reference from the framework section near
  `docs/arxiv-paper/main.tex:352` if Section 3 needs an architecture summary:
  THRML block-Gibbs at candidate warm-start, Soft-Gibbs Residual conditioning,
  no THRML security-parity claim, no Extropic hardware execution claim.
- Defer final OT notation, `J(C)`, and SubOpt integration to exp1579. This
  draft is ready as the sampler/verifier substrate input for that task.
- Do not perform a wholesale rewrite. This note supplies a focused insertion
  block and claim boundaries only.

## Evidence ledger

- `results/experiment_1561_kinetic_defense_zero_coupling_test.json`: THRML
  block-Gibbs falsified kinetic-security parity with single-site Gibbs.
- `results/experiment_1562_brain_linear_ar_k_sweep_extended.json`: BRAIN
  Linear-AR rescue not validated at k=15; Phase 3 recommendation is dropped.
- `results/experiment_1563_specann_rejection_architecture_record.json`:
  SpecAnn rejection architecture record complete.
- `results/experiment_1564_thrml_vendored_block_gibbs_replacement.json`:
  THRML vendored under Apache-2.0, simulator-only, no TSU hardware claim.
- `results/experiment_1565_soft_gibbs_residual_implementation.json`:
  Soft-Gibbs Residual implemented; hard BRS empty-intersection failure
  reproduced.
- `results/experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json`:
  candidate warm-start validated; cold and cached state rejected.
- `results/experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json`:
  Jensen bound verified and beta=0.1 selected for deployment objective.
- `results/experiment_1571_step_wise_baseline_AR_REINFORCE.json`: step-wise
  AR-REINFORCE baseline passes its variance and convergence proxy gates.
