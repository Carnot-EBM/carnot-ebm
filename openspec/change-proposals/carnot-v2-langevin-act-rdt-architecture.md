# Carnot v2.0: Langevin-on-latent + ACT-gated RDT + RLM orchestration

**Status:** Draft change proposal.
**Origin:** External architecture review (Google Deep Think, 2026-04-25)
  synthesised against the existing project state. The review consolidated
  three threads we already had drafted (Phase 3 Kona-parity, RLM
  alignment, OpenMythos as study reference) and proposed three
  *operational* primitives that bridge them. This proposal captures the
  three primitives at the right scope for the project — large enough to
  be Phase 2.5 work, small enough that each experiment has a falsifiable
  acceptance gate.
**Target milestone:** 2026.05.NN+ — after the .65/.66 layered diagnostic
  chain (constraint retrieval embedding-layer fix, arbiter root-cause
  resolution, JEPA cross-domain corpus expansion) lands. Skipping these
  layers to chase architecture would repeat the .60–.64
  documentation-without-application pattern; they have to land first.
**Priority:** High *as a Phase 2.5 anchor*, low *as a near-term
  scheduler input*. The three primitives compound across multiple
  milestones once introduced.
**Depends on:**
  - Existing IsingEBM (`python/carnot/models/ising.py`) — for Exp A.
  - Existing JEPA probe (`python/carnot/samplers/jepa_reasoner_probe.py`,
    plus the v23/v24/v24b iterations from .63–.65) — for Exp B.
  - SOTA GGUF cache (`python/carnot/pipeline/gguf_cache.py` shipped in
    .65 Exp 849) and `cached_sota_pair()` helper in
    `scripts/experiment_template.py` — for Exp C.
  - At least one of the .65/.66 layered fixes producing a positive delta
    (so we know the underlying energy function discriminates before we
    build a generation loop on top of it).

## What this proposal IS

Three operational primitives that translate the existing Phase 3
spec (`openspec/capabilities/phase3-kona/spec.md`) and the RLM proposal
(`openspec/change-proposals/recursive-extractor-and-verify-stream-alignment.md`)
from "design vision" into "buildable code paths":

1. **Langevin dynamics on the residual stream**, applied *before* the
   language-modelling head projects to vocabulary. The update rule is
   $h_{new} = h_{old} - \alpha \nabla_h E_{carnot}(h_{old}) + \text{noise}$
   where the noise term is calibrated so the dynamics implement
   stochastic gradient Langevin sampling from $\exp(-E/T)$. The
   gradient $\nabla_h E_{carnot}$ requires the energy function to be
   differentiable in the residual-stream domain (not the token domain),
   which our IsingEBM already is (its forward pass is JAX-traced).

2. **Adaptive Computation Time (ACT) halting** with the Carnot energy
   as the stopping criterion. Instead of OpenMythos's hardcoded
   $T=16$ recurrent loop, the loop body re-applies the same
   transformer block until $E(h_t) < \tau$ for a tuned threshold
   $\tau$. The maximum-loop bound is set conservatively (e.g. 32) so
   the worst case is bounded; in practice the energy drops within a
   few iterations on satisfiable inputs.

3. **Energy-guided steering inside the loop**:
   $h_{t+1} = \text{TransformerBlock}(h_t) - \lambda \nabla_h E(h_t)$.
   Combines the recurrent pass with the Langevin gradient so each
   loop step both refines the representation *and* descends the
   energy landscape. $\lambda$ is the steering coefficient, separately
   tuned from the Langevin step size $\alpha$ in primitive (1).

## What this proposal IS NOT

- **Not** a Kona reimplementation. Kona's PutnamBench numbers
  (99.4%) are marketing-stated and closed-source-unverifiable; we
  cannot use them as targets without reproducing the eval ourselves.
  The architecture *pattern* (continuous latent reasoning,
  energy-as-validator) is the part we adopt.
- **Not** an OpenMythos dependency. Kye Gomez's reimplementation has
  quality concerns (single contributor, no validation gates, partial
  reconstruction). We borrow the **RDT pattern** (recurrent block +
  prompt re-injection at each step) but implement it ourselves with
  Carnot's no-slop standards.
- **Not** for the generative-time safety gate. Langevin steps and
  RDT loops both blow hard latency budgets. The gate (issue #2
  `budget_ms`) stays on logprob trajectories per Cognometry. This
  proposal is for *offline verification* and *Phase 3 generation*.
- **Not** a replacement for the dogfood-safeguard track. Three
  proposals there (`conductor-self-protection-safeguard`,
  `generative-time-safety-gate`, `garak-red-team-integration`) remain
  separately scheduled.

## Proposed experiments

### Exp A — Langevin update on a frozen IsingEBM (cheap, no LLM hooks)

**Deliverable:** `python/carnot/phase3/langevin_update.py` +
`scripts/experiment_<N>_langevin_ising.py` +
`tests/python/test_langevin_ising.py` +
`results/experiment_<N>_langevin_ising.json`.

**What it does:**

1. Implement `langevin_step(model, h, alpha, temperature, key) -> h_new`
   for a frozen `IsingModel`. Uses JAX `jax.grad` against the model's
   `energy()` method.
2. Sample 100 random initial states $h_0 \in \{-1, +1\}^N$ on a
   pre-trained IsingEBM (the .65 Exp 819 external-field-fix variant).
3. Run 50 Langevin steps per initial state. Record per-step energy.

**Acceptance gates:**

1. **Median energy decrease**: across the 100 starts, median
   $E(h_{50}) - E(h_0) \le -0.5 \cdot \text{median}(E(h_0))$.
   Verifies the gradient field flows the right direction.
2. **No-divergence**: zero starts produce $|h|_\infty > 10$ over the
   50 steps (the dynamics shouldn't blow up under the calibrated
   step size).
3. **Temperature sensitivity**: at $T = 0$ the dynamics are
   deterministic gradient descent and 100% of starts reach a local
   minimum within 50 steps; at $T = 1.0$ the dynamics explore
   multiple basins (final-state distribution measurably differs from
   the $T = 0$ distribution by KL divergence $\ge 0.1$).
4. **Honest-verdict enum**: `langevin_descends_cleanly`,
   `langevin_diverges_below_gate`,
   `langevin_no_temperature_sensitivity`,
   `langevin_implementation_buggy`.

This experiment is contained: no LLM, no GGUF, no GPU required (CPU
JAX is fine for $N \le 256$). Builds the primitive in isolation
before wiring it into anything else.

### Exp B — ACT-gated halting on the existing JEPA probe

**Deliverable:**
`python/carnot/phase3/act_halting.py` +
`scripts/experiment_<N>_act_halting_jepa.py` +
`tests/python/test_act_halting_jepa.py` +
`results/experiment_<N>_act_halting_jepa.json`.

**What it does:**

1. Wrap the latest JEPA probe (v23 or v24b depending on whether the
   .66 corpus expansion lands) in an ACT loop. The loop applies the
   probe's last residual block $T$ times with the prompt embedding
   re-injected at each step, halting when the JEPA energy
   $E_{\text{JEPA}}(h_t) < \tau$.
2. Sweep $\tau \in \{0.1, 0.3, 0.5, 0.7\}$ on a held-out set
   (multi-domain, including the SVAMP cases that .65 Exp 844 isolated
   as embedding-layer-bound).
3. For each $\tau$, report (a) mean iteration count $\bar{T}$,
   (b) accuracy on the held-out set, (c) latency per query.

**Acceptance gates:**

1. **Pareto frontier**: at least one $\tau$ produces strictly higher
   accuracy than the no-loop baseline ($T = 1$) without exceeding
   $\bar{T} = 8$. If the frontier is dominated by the baseline, ACT
   adds no value here and the primitive shouldn't ship — verdict
   `act_no_pareto_improvement`.
2. **Halting termination**: 100% of queries halt within the bounded
   maximum-loop count ($T_{\max} = 32$). Zero hung loops.
3. **No accuracy regression at $\tau$ = max**: setting $\tau = \infty$
   (always-halt-immediately, $\bar{T} = 1$) reproduces the no-loop
   baseline accuracy within $\pm 0.5\%$. Sanity check on the wrapper.
4. **Honest-verdict enum**: `act_halting_pareto_improvement`,
   `act_no_pareto_improvement`,
   `act_halting_unbounded_loop`,
   `act_baseline_regression_at_tau_max`.

### Exp C — Open-weight RDT integration with energy-guided steering (gated on A and B)

**Deliverable:**
`python/carnot/phase3/rdt_steering.py` +
`scripts/experiment_<N>_rdt_steering.py` +
`tests/python/test_rdt_steering.py` +
`results/experiment_<N>_rdt_steering.json`.

**What it does:**

1. Hook the residual stream of an open-weight Qwen3.6-35B-A3B-GGUF or
   gemma-4-31B-it-GGUF (per the SOTA models in CLAUDE.md) at the
   last-layer pre-LM-head position. Requires the cache module from
   .65 Exp 849 + `cached_sota_pair()` to be loadable.
2. Apply the energy-guided steering update
   $h_{t+1} = \text{LastBlock}(h_t) - \lambda \nabla_h E(h_t)$
   for $T$ iterations (or until ACT halts per Exp B).
3. Project the final $h_T$ back through the LM head to produce
   tokens. Compare against the baseline (no steering) on a small
   held-out reasoning set (50 GSM8K + 50 HumanEval + 50 SVAMP).

**Acceptance gates:**

1. **Energy decrease**: the steering produces a strictly lower mean
   final energy than the no-steering baseline. Validates the
   plumbing.
2. **Output-quality non-regression**: accuracy on the held-out set
   is no worse than $-1.0\%$ vs the unsteered baseline. The
   steering shouldn't make outputs worse even when it doesn't make
   them better — that's the no-rubber-stamp gate.
3. **Output-quality improvement**: accuracy on the held-out set is
   strictly higher than the baseline by $\ge +1.0\%$. Without this,
   the primitive ships as "plumbing works but no win" rather than
   "production-ready."
4. **Honest-verdict enum**: `rdt_steering_energy_and_quality_both_up`,
   `rdt_steering_energy_up_quality_flat`,
   `rdt_steering_energy_up_quality_regressed`,
   `rdt_steering_energy_did_not_decrease`,
   `rdt_steering_unhookable_residual_stream`.

This experiment is **gated on Exps A and B**. If the Langevin update
diverges or the ACT halting gives no Pareto improvement, the
combined primitive in C has no foundation and shouldn't ship.

## Risks and honest concerns

- **Langevin cost vs latency**: each gradient step is one full
  forward pass through the EBM. For a 35B-parameter wrapped LM with
  $T = 8$ ACT iterations, that's 8x the latency of a single forward
  pass. Acceptable for offline / verification settings; immediately
  excluded from the generative-time safety gate (issue #2
  `budget_ms`).
- **Open-weight residual-stream hooking is non-trivial**. The
  current `cached_sota_pair()` path uses `Gemma4QuantizedLoader`
  (llama.cpp-backed). llama.cpp does not expose intermediate
  residual streams natively; we would need either a
  Transformers-based path with the full safetensors weights (large
  disk, large VRAM) or a llama.cpp patch. Exp C may surface this as
  `rdt_steering_unhookable_residual_stream` and that is a real risk.
- **The phase boundary**: Phase 1 EBM is inference-time scoring;
  this proposal is the bridge to Phase 3 generation via energy
  minimisation. The bridge requires open-weight access, which
  forces a Transformers-based path even though our .65 work
  prefers GGUF for memory. Worth budgeting that — likely a
  separate Exp B' that picks the right loader path before Exp C
  gets started.
- **Skipping the .65/.66 layered diagnostics**: the strongest
  argument against scheduling this proposal early is that the
  underlying energy function still has unresolved discriminative
  weakness on short arithmetic (Exp 844 embedding-layer collapse),
  partial constraint retrieval (Exp 847 → ?), and arbiter
  calibration in progress (Exp 846 → ?). Building a generation loop
  on top of an energy function with known weak points propagates
  those weaknesses into generation. The .65/.66 fixes have to land
  first.
- **Kona-parity claims have to be reproducible to publish**. Our
  honest-record discipline says we don't cite Kona's PutnamBench
  numbers as comparison targets. If Exp C produces a quality
  improvement, the comparison is against our own pre-steering
  baseline, never against unverifiable closed-source numbers.
- **OpenMythos as a dependency is rejected** for code-quality
  reasons. We adopt the RDT *pattern* (recurrent block, prompt
  re-injection) and implement it ourselves in
  `python/carnot/phase3/`. Exp B's wrapper is built clean.

## Tie-ins to other drafted proposals

- **Issue #4 (ECP5/Nexus open-FPGA port)** — once Exp A's Langevin
  primitive is solid, the same gradient-flow can run on ECP5 with
  the open PnR toolchain (the .64/.65 iCE40 misses pushed this up
  in priority).
- **Issue #6 (ManipulableSignalDependency)** — the centrality of
  load-bearing factors in a reasoning graph is exactly the kind of
  thing the Langevin energy landscape can score and steer away
  from. After Exp A ships, issue #6's anchor-detection becomes a
  natural follow-up using the same primitive.
- **Recursive-extractor-and-verify-stream-alignment** (drafted
  643a21ad) — this proposal pairs cleanly with that one. Deep
  Think's "RLM-as-orchestrator, Carnot-as-engine" division of
  labour is what verify_stream + the Langevin/ACT/RDT engine
  *together* look like. Not duplicative; complementary.
- **Phase 3 capability spec
  (`openspec/capabilities/phase3-kona/spec.md`)**: the three
  primitives in this proposal directly satisfy three of the
  REQ-KONA-* requirements. Implementation status table in the spec
  should be updated when any of A/B/C ships.
