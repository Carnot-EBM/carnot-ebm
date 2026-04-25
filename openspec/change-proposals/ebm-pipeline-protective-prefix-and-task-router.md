# EBM Pipeline: protective-layer prefix + routed task experts

**Status:** Draft change proposal.
**Origin:** Conversation 2026-04-25 — surfaced when discussing whether
  Mixture-of-Experts is the right pattern for Carnot's safety
  components. The answer: not flat MoE. Safety is a *protective layer
  that always fires*, not an alternative path to route to. The right
  architecture has two distinct contracts: `SafetyPrefix` (always-on,
  veto on threshold) and `TaskExpert` (routed by claim type, sparse
  activation), composed by an `EBMPipeline` chain. This proposal
  formalises that composition layer.
**Target milestone:** 2026.05.NN — alongside or just after the v2
  open-weight (`a1d0a338`) and closed-weight integration (`fc7ee798`)
  proposals. Together they form the project's full verification-stack
  architecture.
**Priority:** High. The three drafted dogfood-safeguard proposals
  (`conductor-self-protection-safeguard`,
  `generative-time-safety-gate`, `garak-red-team-integration`) all
  *behave* as protective layers but lack a unifying composition
  contract. Without `EBMPipeline`, each one ships as bespoke wiring;
  with it, they slot into the prefix layer cleanly and the
  per-claim-type EBMs slot into the task-router layer.
**Depends on:**
  - Existing dogfood-safeguard proposals (3 drafted) — provide the
    initial SafetyPrefix instantiations.
  - Existing per-claim-type verifiers
    (`python/carnot/verify/{constraint,property_test,sat,sc_energy}.py`)
    — provide the initial TaskExpert instantiations.
  - `explicit-factor-graph-abstraction.md` (drafted, `b95c27d1`) —
    each pipeline layer maps onto factors in the underlying graph.

## Why not flat MoE

A flat Mixture-of-Experts pattern routes each input to one (or a few)
experts based on a learned gate, and aggregates only the selected
experts' outputs. That is correct for *task specialisation* —
arithmetic vs code vs logic — but **wrong for safety**. Safety
is not "an alternative to task-correctness"; it is "always evaluate,
always have veto power." A safety-EBM that only fires when you
already suspect a threat is too late: by the time you suspect, the
input may have done damage.

The right pattern is a **filter chain** with two distinct contracts:
- **`SafetyPrefix`** — every input, fast, vetoes downstream on
  threshold. Like iptables packet filtering chains, like Kubernetes
  admission controllers, like TLS termination. *Not* routed.
- **`TaskExpert`** — routed by claim type, sparse activation. Like
  MoE for the *task* portion of the pipeline.

Both contracts compose under a single `EBMPipeline` interface that
runs the prefix chain first, short-circuits on veto, then routes to
task experts.

## Two contracts

### `SafetyPrefix` contract

```python
class SafetyPrefix(Protocol):
    """A stateless, always-on filter. Never routed.

    Returns an energy, an optional veto flag, and a human-readable
    reason for the veto. Latency target: under 10ms per call.
    Calibration objective: false-positive rate <= 2% on a benign
    held-out set (rejecting legitimate work is expensive).
    """
    def evaluate(self, input_: PipelineInput) -> SafetyResult:
        ...

@dataclass
class SafetyResult:
    energy: float          # higher = more dangerous
    veto: bool             # if True, downstream is short-circuited
    reason: str            # diagnostic for the artifact
    layer_id: str          # which prefix in the chain this came from
```

Initial instantiations (from the dogfood-safeguard proposals):
- `InjectionPrefix` — wraps the prompt-injection KAN v3
  (existing AUROC 0.9078, with the cross-vendor caveat from
  Cognometry to be tightened by Exp A.3 of the closed-weight
  integration proposal).
- `JailbreakPrefix` — wraps the existing privacy-filter v2
  (AUROC 1.0 on training distribution).
- `StructuralAnomalyPrefix` — Cognometry-style logprob-trajectory
  scoring; black-box-accessible signal for closed-weight upstream
  generators.
- `ManipulableSignalPrefix` — wraps the issue-006 anchor-centrality
  detector once it ships.

Aggregation: **max-pool**. Any layer's veto short-circuits the rest
of the prefix and the entire downstream task layer. Final prefix
energy is `max(layer_energies)` across the chain.

### `TaskExpert` contract

```python
class TaskExpert(Protocol):
    """A routed expert for a specific claim type.

    Returns an energy and the evidence that produced it.
    Calibration objective: false-negative rate <= 5% on flawed
    held-out claims (missing real flaws is the failure mode).
    """
    def evaluate(self, claim: ClassifiedClaim) -> TaskResult:
        ...
    def claim_types(self) -> set[ClaimType]:
        """Which claim types this expert can handle."""
        ...

@dataclass
class TaskResult:
    energy: float
    claim_type: ClaimType
    evidence: dict          # for the artifact's diagnostic field
    expert_id: str
```

Initial instantiations (from existing `python/carnot/verify/`):
- `ArithmeticExpert` — wraps `verify/sat.py` with Z3 backing.
- `CodeExpert` — wraps `verify/property_test.py` with Hypothesis
  backing.
- `LogicalSatisfiabilityExpert` — wraps `verify/constraint.py`.
- `CrossStepConsistencyExpert` — wraps the EBM energy probe used
  for inter-step consistency (the .65 Exp 836 layer-2 work).

Aggregation: **weighted sum**. The pipeline computes total task
energy as `Σ(weight_i * task_energy_i)` over the routed experts.
Weights are learned per claim type from a calibration set.

### `EBMPipeline` chain

```python
class EBMPipeline:
    def __init__(
        self,
        safety: list[SafetyPrefix],
        router: TaskRouter,
        experts: dict[ClaimType, TaskExpert],
        aggregator: EnergyAggregator,
    ): ...

    def evaluate(self, input_: PipelineInput) -> PipelineVerdict:
        # 1. Run prefix chain (always-on)
        for layer in self.safety:
            r = layer.evaluate(input_)
            if r.veto:
                return PipelineVerdict.short_circuit(reason=r.reason, layer=r.layer_id)
        # 2. Route to task experts (sparse)
        claims = self.router.classify(input_)
        task_results = [self.experts[c.claim_type].evaluate(c) for c in claims]
        # 3. Aggregate
        return self.aggregator.combine(safety_results, task_results)
```

The `EnergyAggregator` is the cross-layer calibration object — the
hard problem (see Risks). Final pipeline energy is
`weighted_combine(max(safety_energies), Σ(task_energies))` with
safety weighted high so that `safety_energy = 1.0` dominates any
finite `task_energy`.

## Proposed experiments

### Exp A — `SafetyPrefix` contract + first three prefix layers

**Deliverable:** `python/carnot/pipeline/safety_prefix.py` +
`scripts/experiment_<N>_safety_prefix_chain.py` +
`tests/python/test_safety_prefix.py` +
`results/experiment_<N>_safety_prefix.json`.

**What it does:** ship the `SafetyPrefix` Protocol + three
implementations wrapping existing safety models:

1. `InjectionPrefix` — wraps `python/carnot/inference/injection_kan.py`.
2. `JailbreakPrefix` — wraps the existing privacy-filter classifier.
3. `StructuralAnomalyPrefix` — new, scores logprob trajectories
   (Cognometry-style; works on closed-weight generators since
   logprobs are API-accessible).

Run on a labelled corpus of 500 reasoning traces (300 benign,
200 known-flawed across the three threat classes) collected from
GPT-4o, Claude, Gemini, and our local SOTA models.

**Acceptance gates:**
1. **Per-layer in-distribution AUROC ≥ 0.85** on the threat class
   each layer targets (injection layer on injection traces,
   jailbreak layer on jailbreak traces, structural-anomaly layer
   on hallucinated traces).
2. **Aggregate false-positive rate ≤ 2%** on the 300 benign traces.
   The chain max-pools layers; total FP rate must stay within
   budget. If higher, layer thresholds need tightening.
3. **Aggregate latency p95 ≤ 30ms** end-to-end across the three
   layers on a representative input.
4. **Honest-verdict enum**: `safety_prefix_all_three_clear`,
   `safety_prefix_aggregate_fp_above_budget`,
   `safety_prefix_individual_layer_below_auroc_gate`,
   `safety_prefix_latency_above_budget`.

### Exp B — `TaskExpert` contract + initial four expert wrappers

**Deliverable:** `python/carnot/pipeline/task_expert.py` +
`scripts/experiment_<N>_task_expert_router.py` +
`tests/python/test_task_expert.py` +
`results/experiment_<N>_task_expert.json`.

**What it does:** ship the `TaskExpert` Protocol + four wrappers
around existing verifiers, plus a `TaskRouter` that classifies
claims and dispatches:

1. `ArithmeticExpert` (Z3-backed).
2. `CodeExpert` (Hypothesis-backed).
3. `LogicalSatisfiabilityExpert` (constraint solver).
4. `CrossStepConsistencyExpert` (EBM energy across reasoning steps).

The router uses the existing `LLMConstraintExtractor` claim-type
classification plus a fallback that routes ambiguous claims to
all eligible experts (fan-out, not exactly-one).

**Acceptance gates:**
1. **Router classification accuracy ≥ 95%** on a labelled set of
   500 claims with known types. Sub-95% means too many claims
   land on the wrong expert.
2. **Each expert's per-claim AUROC ≥ 0.90** on its own claim type.
   This is mostly already true for Z3 / Hypothesis (they're
   formal verifiers); the EBM-based experts are the harder cases.
3. **Fan-out budget**: if the router falls back to fan-out, no
   more than 2.5 experts evaluate per ambiguous claim on average
   (cost cap).
4. **Honest-verdict enum**: `task_expert_router_all_above_gate`,
   `task_router_classification_below_gate`,
   `task_expert_individual_auroc_below_gate`,
   `task_router_fanout_above_budget`.

### Exp C — `EBMPipeline` assembly + cross-layer energy calibration (gated on A and B)

**Deliverable:**
`python/carnot/pipeline/ebm_pipeline.py` +
`scripts/experiment_<N>_ebm_pipeline_e2e.py` +
`tests/python/test_ebm_pipeline.py` +
`results/experiment_<N>_ebm_pipeline.json`.

**What it does:** compose the prefix chain from Exp A and the
expert router from Exp B into a single `EBMPipeline`. Implement
`EnergyAggregator` with three options to test:

1. **Calibrated weighted sum** — Platt scaling per layer, then
   weighted sum. Standard binary-calibration approach.
2. **Calibrated isotonic regression** — non-parametric per-layer
   calibration, then sum. Handles non-linear miscalibration but
   needs more data.
3. **Bayesian-product** — treat each layer's energy as a
   log-likelihood and sum (multiply probabilities). Theoretically
   principled but assumes layer independence.

Run end-to-end on a held-out 1000-trace corpus and pick the
aggregator that gives the best detection rate at a 2% FP cap.

**Acceptance gates:**
1. **End-to-end detection rate ≥ 90%** at the 2% FP budget on the
   held-out corpus, using the best of the three aggregators.
   This is the headline number — the pipeline as a whole.
2. **No-aggregation-collapse**: the pipeline's detection rate is
   strictly higher than the best single layer's detection rate.
   If aggregation gives no lift over the strongest individual
   layer, the assembly isn't earning its complexity.
3. **End-to-end latency p95 ≤ 200ms** on the held-out corpus.
   Stacked layers add up; cap the total budget.
4. **Honest-verdict enum**: `ebm_pipeline_lift_above_gate`,
   `ebm_pipeline_no_aggregation_lift`,
   `ebm_pipeline_latency_above_budget`,
   `ebm_pipeline_aggregator_uncalibrated_at_threshold`.

## Risks and honest concerns

- **Cross-layer energy calibration is the hard problem.** Safety
  layers produce energies on residual-stream / logprob features.
  Task experts produce energies from Z3 satisfiability or
  Hypothesis property tests. These aren't on the same scale.
  Exp C tests three aggregators because we don't know in advance
  which one works on this data shape; if all three fail the
  no-aggregation-collapse gate, the right answer is to keep the
  layers as *independent* judges (each producing its own verdict)
  rather than fused into a single pipeline energy.
- **Safety layer false-positive rate compounds across layers.** A
  3-layer max-pool with 1% per-layer FP rate produces ~3% aggregate
  FP. The 2% aggregate cap in Exp A.2 means each individual layer
  must run at < 0.7% FP. That requires careful threshold tuning,
  and may not be achievable on all three threat classes
  simultaneously.
- **Router misclassification puts claims at the wrong expert.**
  Exp B.1's 95% classification accuracy gate is a hard floor;
  below that, the wrong expert sees the claim and the verdict is
  unreliable. The fan-out fallback for ambiguous claims is the
  mitigation but adds cost.
- **The `SafetyPrefix` contract assumes prefix layers are
  independent.** If injection and jailbreak failure modes
  correlate (likely, since both are forms of prompt manipulation),
  the max-pool aggregation can mask real differences. Exp A
  measures cross-layer correlation as a side output.
- **Garak-style poisoning extends to safety prefixes.** If a
  prefix layer's training corpus is contaminated by adversarial
  inputs labelled as benign, the layer trains a blind spot.
  Inheriting the dogfood-safeguard track's
  `training_eligible:false` quarantine discipline is a strict
  prerequisite.
- **The pipeline is configurable per use-case.** The next
  proposal in this thread (multi-LLM verification pipeline)
  formalises that — different deployments enable different
  layers based on cost / latency / stakes. This proposal
  delivers the assembly primitive; the configuration DSL is
  separate.

## Tie-ins to other drafted proposals

- **Dogfood-safeguard track** (3 proposals, queued for 4
  milestones now): the three SafetyPrefix instances in Exp A are
  exactly those three proposals' verification surfaces. Once
  this proposal lands, the dogfood-safeguard work has a clear
  composition target.
- **Factor-graph proposal** (`b95c27d1`): each `SafetyPrefix` and
  `TaskExpert` becomes one factor in the underlying factor graph.
  The pipeline's veto semantics are an unusual factor type
  (short-circuit) that the factor-graph abstraction needs to
  handle.
- **v2 open-weight architecture** (`a1d0a338`): the Langevin /
  ACT-RDT engine processes a single hidden state. The pipeline
  evaluates the *output* of that engine. They compose: pipeline
  is the verifier of a v2-engine generation.
- **Closed-weight integration** (`fc7ee798`): the teacher-forced
  proxy in Exp A of that proposal is one form of safety prefix.
  When ready, the closed-weight track's experiments fill in
  additional `SafetyPrefix` instances.
- **Multi-LLM verification pipeline** (not yet drafted, proposed
  in conversation): the configuration-DSL layer that sits
  *above* this proposal — selects which `SafetyPrefix` and
  `TaskExpert` instances are active per deployment use-case.
  This proposal is the assembly primitive; that one is the
  configuration primitive.

## What this proposal explicitly does not deliver

- **Not a configuration DSL.** Per-use-case toggling of layers
  (chatbot vs production-codegen vs offline-audit) is the
  multi-LLM verification proposal's job, not this one's.
- **Not LLM-as-judge layers.** This proposal is EBM and formal
  verifier components only. LLMs as components in the pipeline
  belong in the multi-LLM proposal (which builds on top of this
  one).
- **Not a learned router.** `TaskRouter` in Exp B uses the
  existing `LLMConstraintExtractor` plus deterministic
  fallback. Learning the router from data is a future
  experiment, not this one.
