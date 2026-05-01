# Deep Think Prompt — GRPO + SP-IWPER Async Replay Composition

**Status:** Ready to send. Phase-3 prototype design dependency: if
GRPO and SP-IWPER are incompatible, the .88 prototype needs a
different RL algorithm or a different replay design BEFORE training
starts, not after.
**Date drafted:** 2026-05-01
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project's Phase-3 4-stage training pipeline integrates
two recent design decisions that have NOT been jointly
validated:

#### Recent decision 1: GRPO with energy reward as Stage 3

GRPO (Group Relative Policy Optimization) is the .87
**replacement** for RLVR + SSD (which had degenerate-corpus
issues, see exp1099). GRPO ditches the value/critic network
entirely and computes advantages from a *group* of sampled
trajectories using their relative reward. For Carnot, the
reward is the energy score from the AND-composed k=5 verifier
ensemble.

**Empirical status:** exp1118 (today, 2026-05-01) trained
GRPO with the ThinkPRM v2 energy reward and reported
`honest_verdict: positive_improvement`. This validates GRPO
as a **standalone** Stage 3.

#### Recent decision 2: SP-IWPER buffer for Stage 4

SP-IWPER = **Stratified Pipeline with Importance-Weighted
Prioritized Energy Replay**. It is the Phase-3 Round 3 Deep
Think solution to a fatal GPU-starvation bottleneck in the
naive synchronous joint fine-tune of DBAE + EBM + verifier.

Key SP-IWPER design properties:

- **Stratified** — replay buffer partitions samples by
  energy bin (high / medium / low). Each training batch
  pulls from all strata to prevent the EBM from collapsing
  onto a single energy mode.
- **Importance-weighted** — sample probability is inversely
  proportional to current energy estimate. Hard examples
  (high energy under current EBM) get sampled more often.
- **Prioritized** — replay priority decays with sample age,
  so newer samples are favored.
- **Async** — buffer is filled by a separate inference
  process running in parallel with training. Training reads
  from the buffer as fast as available; inference fills it
  at its own pace. **This is what creates staleness.**

The DDE Hopf bound on async replay is:
`N_max < π / (2·sqrt(η_d·η_e·ρ(H_φθ·H_θφ)))`

where N_max is the **maximum staleness** (number of training
steps between when a sample was generated and when it's used)
that the joint fine-tune can tolerate without bifurcating
into oscillation.

### The composition concern

GRPO is traditionally an **on-policy** algorithm. The group
baseline is computed from *currently-sampled* trajectories,
and the advantage estimate assumes the trajectories reflect
the current policy distribution.

SP-IWPER provides **stale data by design**. A replay buffer
sample generated under policy π_t (current) but used at
policy π_{t+N} (current + N steps later) has reward signal
calibrated to π_t, not π_{t+N}.

The interaction question is:

**Can GRPO's advantage estimation safely use SP-IWPER replay
data?** Or does the staleness make the group baseline
inconsistent — i.e., the group is a snapshot of an old
policy, but the gradient is applied to the new policy, so
the resulting parameter update is biased in some
non-trivial way?

### Three plausible outcomes

#### Outcome A: They compose safely

GRPO's group-relative formulation cancels the absolute reward
calibration; only the *relative* ranking within a group
matters. Stale data is fine as long as the *relative*
ranking doesn't change much between when the group was
sampled and when it's used. The DDE Hopf bound governs the
joint fine-tune dynamics regardless of GRPO; SP-IWPER works
without modification.

#### Outcome B: They compose with a staleness budget

GRPO + SP-IWPER work together, but the staleness budget is
**tighter** than the DDE Hopf bound. The relative ranking
within a group degrades faster than the joint fine-tune
oscillation threshold, so we need a *separate, tighter*
staleness cap on the GRPO-specific reads.

#### Outcome C: They are incompatible

Stale GRPO data systematically biases the gradient estimate
in a way that the standard variance-reduction techniques
can't fix. The .88 prototype needs to either:
- Replace GRPO with PPO + critic (so the value baseline is
  re-computed at training time, not sampling time)
- Or rebuild SP-IWPER as on-policy (no replay, just batch
  sampling) — which re-introduces the GPU-starvation
  bottleneck SP-IWPER was designed to solve

### Specific questions

1. **Which outcome (A / B / C) is supported by the
   theoretical analysis?** Walk through the GRPO advantage
   estimator equation and show whether stale data introduces
   a fixed bias, a noise-amplification, or no problem.

2. **If Outcome B is correct, what determines the staleness
   budget for GRPO specifically?** Is it the policy KL
   divergence between sampling-time and training-time
   policies? The Lipschitz constant of the energy reward?
   Some other quantity?

3. **What would empirically distinguish A from B from C** in
   the first 1000 training steps of the Phase-3 prototype?
   List specific diagnostics we can log per step (e.g.,
   "compare GRPO advantage estimates from current-batch vs.
   buffer-batch on the same trajectories", "monitor the
   variance of group-relative advantages over time").

4. **If Outcome C is correct, is there a hybrid design that
   preserves the GPU-throughput benefit of SP-IWPER without
   compromising GRPO's on-policy assumption?** (e.g.,
   "sample fresh on-policy data for GRPO advantage, but
   use SP-IWPER replay only for the EBM loss term".)

5. **Does the answer change** if the GRPO group size is
   small (N=4) vs. large (N=64)? Larger groups should give
   tighter advantage estimates but require more on-policy
   samples per gradient step.

### Constraints on output

- **NO parameter prescriptions.** Don't tell us "use
  staleness_max = 312" or "GRPO group size = 17". Stick to
  *which staleness regime is safe*, *which diagnostic
  measures it*, and *what the trade-offs are*.
- **DO provide compositional analysis.** The point of asking
  Deep Think (versus running an experiment) is to get a
  derivation that anchors the empirical work, not the other
  way around.
- **DO acknowledge uncertainty** — if the answer depends on
  problem-specific factors we haven't measured, name them
  and recommend specific empirical pre-tests we should run
  before committing to a design.

### Output format request

```
COMPOSITIONAL ANALYSIS:
  GRPO advantage estimator equation: <write it out>
  Staleness perturbation: how does stale data change the estimator
  Bias term: <closed form if available, "non-trivial" if not>
  Variance term: <closed form or qualitative>
  Conclusion: A / B / C with confidence

EMPIRICAL DIAGNOSTICS (for first 1000 steps):
  Diagnostic 1: <name, formula, what value supports A vs B vs C>
  Diagnostic 2: ...
  ...

HYBRID DESIGN (if Outcome C):
  Proposed architecture:
  Trade-off vs. pure SP-IWPER:
  Trade-off vs. pure GRPO:

GROUP SIZE DEPENDENCY:
  Small (N=4-8): <does answer change>
  Large (N=64+): <does answer change>
  Tradeoff: <Pareto frontier qualitative>
```

### Cross-validation reminder

Per `feedback_carnot_prediction_pattern.md`: prior Deep Think
rounds have qualitative survival claims well-calibrated, but
specific numerical prescriptions systematically wrong. The
question here is framed as compositional analysis (which
outcome / which diagnostic / which trade-off), not parameter
prescription. If your answer drifts toward specific numerical
recommendations, please flag the drift explicitly and provide
the qualitative answer alongside.

Also note: exp1118 (today) showed GRPO with energy reward as
*standalone* working (`positive_improvement`). The next
empirical data point will come from the .88 Phase-3 prototype
joint fine-tune, where GRPO + SP-IWPER are stacked. The
purpose of this Deep Think round is to set up the .88
prototype's instrumentation correctly so that whichever
outcome (A / B / C) emerges is *immediately* recognizable
from the diagnostics, not discovered after a milestone of
training.

---

## End of prompt
