# Explicit `FactorGraph` abstraction for the Carnot EBM

**Status:** Draft change proposal.
**Origin:** Conversation 2026-04-25 — surfaced when investigating the .63
  Layer-2 constraint-zero-delta finding (Exp 821) and the .64 plan to
  diagnose its root cause (Exp 833). The diagnosis is structurally
  difficult without explicit per-factor energy tracking.
**Target milestone:** 2026.04.65 — small, additive, sized to fit alongside
  the dogfood-safeguard track that didn't land in .64.
**Priority:** Medium-high. Three other drafted proposals have a cleaner
  landing surface if a `FactorGraph` exists first (issue #5 SessionMemory
  schema, issue #6 ManipulableSignalDependency, the XTR-0 hardware path).
  Not on the critical path for any single milestone, but compounds.
**Depends on:** existing `IsingModel` (`python/carnot/models/ising.py`),
  `BoltzmannModel`, `GibbsModel`. No external dependencies.

## Summary

The Carnot EBM has factor-graph structure pervasively but implicitly:

- `IsingModel.energy(x) = -0.5 x @ J @ x - bias @ x` decomposes naturally
  into pairwise factors (one per non-zero $J_{ij}$) and unary factors
  (one per $h_i$). The decomposition is mathematical, not structural —
  the code stores a single matrix $J$ and computes total energy in one
  shot.
- The `LLMConstraintExtractor` produces a "claim graph" — claims are
  variables, the relations between them act as factors. Already a
  factor graph in everything but name.
- The Boltzmann and Gibbs tiers are MRFs (Markov Random Fields), a
  special case of factor graphs.
- The Phase 2 hardware target (Extropic XTR-0 / X0) consumes
  factor-graph specifications natively. So does THRML.

What we don't have:

- A first-class `FactorGraph` data structure with separate variable and
  factor nodes.
- Per-factor energy attribution. We compute total energy, not which
  factor contributed how much. The .63 Exp 821 zero-delta finding is
  exactly the kind of bug an explicit per-factor view would expose:
  if a new constraint adds tiny perturbations across many entries of
  the global $J$, its signal averages out and the total energy delta
  reads as zero, even though *some* factors might be carrying real
  weight.
- Structured inference (sum-product / loopy belief propagation). We
  use Gibbs sampling and Langevin instead. Mathematically valid but
  doesn't exploit factor-locality, and does not map onto XTR-0's
  message-passing primitives.
- A serialisation format for individual factors, which would slot
  cleanly into issue #5's portable SessionMemory schema (each
  constraint memory entry could be exactly one factor).

This proposal adds a thin `FactorGraph` abstraction that wraps
existing tiers without breaking them, exposes per-factor energy
contributions, and gives downstream proposals (issues #5 and #6,
the XTR-0 hardware path, and the .64 Exp 833 root-cause diagnosis)
a clean target.

## Proposed experiments

### Exp A — `FactorGraph` data structure + Ising adapter

**Deliverable:** `python/carnot/factor_graph/__init__.py` +
`python/carnot/factor_graph/core.py` (Variable, Factor, FactorGraph)
+ `python/carnot/factor_graph/ising_adapter.py` (build a FactorGraph
from an existing IsingModel) +
`tests/python/test_factor_graph.py` +
`results/experiment_<N>_factor_graph_primitive.json`.

**Core data classes:**

```python
@dataclass(frozen=True)
class Variable:
    """A node in the factor graph. Identified by a string id; carries
    the domain (e.g. "binary" for spins, "continuous" for relaxations)."""
    id: str
    domain: str
    metadata: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Factor:
    """A factor in the graph. `scope` is the tuple of Variable ids the
    factor connects to. `energy_fn` takes an assignment dict
    {var_id: value} and returns a scalar energy contribution."""
    id: str
    scope: tuple[str, ...]
    energy_fn: Callable[[dict[str, Any]], float]
    metadata: dict = field(default_factory=dict)

class FactorGraph:
    """A factor graph. Holds variables and factors, supports add/remove,
    per-factor energy queries, total-energy aggregation, and JSON
    serialisation of the graph structure (without the energy_fn
    closures, which serialise via a registered template id)."""
```

**Adapter from existing Ising:**

```python
def from_ising(model: IsingModel) -> FactorGraph:
    """Decompose an IsingModel into its factor-graph view.

    Each non-zero coupling J_ij becomes one pairwise factor; each
    non-zero bias h_i becomes one unary factor. Symmetry of J is
    used to halve the factor count (J_ij and J_ji are the same factor).
    The resulting FactorGraph computes the same total energy as the
    underlying IsingModel.energy() — equivalence is verified by an
    end-to-end test on randomly-initialised models."""
```

**Acceptance gates:**

1. **Total-energy equivalence**: on 10 random `IsingModel(input_dim=64)`
   instances and 100 random input vectors per instance, the
   `FactorGraph(...).total_energy(x)` matches `IsingModel.energy(x)`
   to within 1e-6 absolute tolerance. Verifies the adapter's
   decomposition is mathematically correct.
2. **Per-factor query**: `FactorGraph.factor_energies(x)` returns a
   dict `{factor_id: energy_contribution}` whose values sum to the
   total energy (within 1e-6 absolute tolerance). This is the
   capability the .63 Layer-2 finding needed.
3. **Round-trip serialisation**: `FactorGraph.to_json()` / `from_json()`
   on the structure (variables + factor metadata + factor template
   ids) — the energy functions themselves serialise as registered
   template names, not closures. Round-trip on a 64-variable Ising
   factor graph reconstructs an equivalent FactorGraph (verified by
   total-energy equivalence on 100 random inputs).
4. **Honest-verdict enum**: `factor_graph_primitive_ships`,
   `ising_adapter_total_energy_mismatch`,
   `per_factor_energy_does_not_sum_to_total`,
   `serialisation_roundtrip_breaks`.

### Exp B — Per-factor energy attribution on the .63 Layer-2 case

**Deliverable:**
`scripts/experiment_<N>_layer2_factor_attribution.py` +
`results/experiment_<N>_layer2_factor_attribution.json`.

**What it does:** Reproduce the Exp 821 setup (constraint addition
across 3 sessions, observed `delta_overall=0.0`). Re-run via the new
`FactorGraph` representation. Compute per-factor energy contributions
*before* and *after* each constraint is added. Identify whether:

- **(a)** the new constraint factors carry zero energy individually
  (constraints are vacuous — extraction or template problem);
- **(b)** they carry non-zero energy individually but average out
  across the existing global $J$ (encoding problem — fix is to keep
  factors separate rather than merging into the coupling matrix);
- **(c)** they carry non-zero energy but the *sign* of their
  contribution is negative on the same problems where existing
  factors are negative (signal collision — different fix entirely).

The previous experiments (821, 836 in the .64 plan) measured a single
total-energy delta. They cannot distinguish these three failure modes.
The factor-graph view can, in one experiment.

**Acceptance gates:**

1. The experiment produces a clear classification: (a), (b), or (c).
   Honest-verdict reflects which mode the data shows.
2. If (b) is the answer, propose a fix: the .65 follow-up keeps
   constraint factors separate from the global coupling matrix and
   re-runs Exp 821-shape evaluation.
3. If (a) is the answer, propose a fix in the extractor / constraint
   template, not the EBM — directs work toward the right layer.
4. Honest-verdict enum: `layer2_mode_a_constraints_vacuous`,
   `layer2_mode_b_signal_averaged_in_global_J`,
   `layer2_mode_c_sign_collision`,
   `layer2_inconclusive_below_signal_floor`.

### Exp C — Boltzmann and Gibbs adapters

Same shape as Exp A's Ising adapter, but for the Boltzmann and Gibbs
tiers. Establishes that the `FactorGraph` representation works
uniformly across all four tiers (with the KAN tier being the only
non-graphical-model exception — KAN is a continuous functional
approximator, not a factor graph; the adapter for KAN returns
`NotImplementedError` with a clear message).

**Acceptance gates:** total-energy equivalence on 10 random instances
of each tier; honest-verdict lists which tiers have working adapters.

## Tie-ins to other drafted proposals (compounding wins)

This is what makes the proposal worth doing — it unblocks more than
itself:

- **Issue #5 (portable SessionMemory JSON schema)**: with `Factor`
  serialisable as a single self-contained JSON object, each
  constraint memory entry becomes exactly one factor. The schema in
  issue #5 uses the `Factor` shape directly; no separate format
  needed.
- **Issue #6 (ManipulableSignalDependency)**: directly graph-structural.
  Issue #6's "anchor centrality" and "load-bearing single-source
  signal" are operations on a factor graph. With `FactorGraph` in
  hand, issue #6 becomes a single-pass graph algorithm rather than
  custom plumbing.
- **XTR-0 hardware backend** (Phase 2 horizon): XTR-0 consumes a
  factor-graph specification at the wire protocol. Today our
  `SamplerBackend` protocol passes flat coupling matrices; an XTR
  adapter would have to re-decompose them. With `FactorGraph` as a
  first-class type, the XTR backend just consumes it directly.

## Risks

- **Scope creep into a full graphical-models library**. We do not need
  belief-propagation inference, junction trees, or general
  message-passing in the first cut. Scope this proposal at the data
  structure + per-factor energy queries + serialisation; defer
  inference algorithms to a separate proposal once we have a
  use-case (likely the XTR-0 path).
- **Performance regression on `IsingModel.energy`**. The
  factor-graph view has higher constant overhead per evaluation
  (Python iteration over factors instead of one matrix multiply).
  Mitigation: keep the existing `IsingModel.energy` as the fast path
  for sampling loops; only call `FactorGraph.factor_energies` from
  diagnostic / debugging / proposal experiments. Acceptance gate
  on Exp A measures this — if the fast path regresses by more than
  5%, we add a `__c__` cython/numpy hot loop.
- **Adapter correctness across tiers**. The Boltzmann/Gibbs adapters
  in Exp C are non-trivial — these tiers have different parameter
  shapes than Ising. Mitigation: each tier's adapter must pass the
  same total-energy-equivalence gate that Ising's does, with a
  separate honest-verdict per tier.
- **The .64 Exp 833 (constraint-delta root-cause diagnosis) might
  conclude before this proposal lands**, in which case its
  conclusion drives the priority of Exp B here. If 833 finds (a),
  this proposal's Exp B becomes verification only. If 833 cannot
  distinguish (a) / (b) / (c) — which is likely without
  per-factor attribution — Exp B is the diagnosis, on its own
  schedule.
