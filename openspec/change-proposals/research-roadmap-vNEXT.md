# Research Roadmap vNEXT: Milestone 2026.05.282

**Title:** Claim Repair + Metamorphic Validation + Attractor Memory + GateMate IO

**Planned:** 2026-05-24

**Previous milestone:** 2026.05.281

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.281 Proved

Milestone `.281` completed through
`results/experiment_2999_capstone_v281.json`. The capstone verdict is
`complete: capstone_ready=true; paper_ready=false; clean=34; flagged=20;
blocked=8; missing=1; gated_skipped=0`.

- **Mandated SOTA local GGUF inference is usable.** `exp2989` proved that at
  least one mandated headline local GGUF model can produce a live transcript
  with provenance. `.282` should reuse that cache path, not broaden the model
  claim.
- **The repair signal is real but not claim-safe.** `exp2991` produced a live
  headline hard-set repair delta (`pass_at_1_delta=0.4167`), but the row was
  flagged because methodology, tautology, and false-accept provenance did not
  clear promotion gates.
- **Solver feedback is the cleanest technical result.** `exp2992` reproduced
  solver-feedback formalization gains with stricter Z3 provenance
  (`formalization_clean=true`, `feedback_repair_delta=1.0`).
- **Prompt-to-validator structure is ready.** `exp2994` produced a deterministic
  exact-check dialogue protocol. It is a schema/protocol result, not proof that
  LLM judgments are verifiers.
- **FR-11 has a narrow ready boundary.** `exp2995` produced verifier-grounded
  trace memory while preserving the independent self-learning boundary.
- **AquaForte/BEAVER still needs durable live retry provenance.** `exp2993`
  separated live retry and enumerator fallback, but the matrix/capstone still
  require clean model checksums and duration provenance before promotion.
- **GateMate remains blocked at host-visible evidence.** `exp2996` detected the
  board but flash failed, readback was unsupported, and no smoke vector passed.
  `exp2997` was gate-blocked/missing because SSQA cannot advance on flash-only
  or failed-flash evidence.
- **The next milestone should be claim repair, not publication.** The matrix
  and capstone identify exact repair targets: hard-set methodology, substrate
  provenance, hardware IO/readback, missing SSQA artifact, and bounded FR-11
  carry-forward.

## Three Biggest Gaps

### Gap 1: Repair Evidence Has a Promotion Gap

The PRD requires verifiable reasoning, but a positive pass-rate delta is not
enough if the benchmark, oracle, or false-accept path is ambiguous. Recent
metamorphic-testing work (LLMORPH, MR-Coupler, and the 2026 metamorphic testing
survey note) suggests the next repair attempt should add relation-preserving
variants, test-amplification provenance, and separate false-accept accounting
before re-running live local GGUF repair.

### Gap 2: Continuous Self-Learning Needs Stability Evidence

`.281` made FR-11 trace memory ready in a narrow sense, but the long-term PRD
vision needs memories that converge, resist drift, reject negative controls,
and remain useful on held-out verifier metrics. New attractor/fixed-point work
(Equilibrium Reasoners and Solve the Loop) gives a bounded diagnostic target:
measure convergence and drift over existing solver/validator traces before
claiming any native attractor-model capability.

### Gap 3: Hardware Still Lacks a Host-Visible Output Contract

GateMate detection without flash/readback/smoke output cannot support SSQA,
sampler, or acceleration claims. `.282` must build or conclusively diagnose a
minimal host-visible IO path, then gate SSQA RTL/PnR/resource reporting on that
boundary. Extropic/THRML and Kona remain strategic context only.

## New Research Integrated

The 2026-05-24 post-`.281` sweep appended these sources to
`research-references.md` before this design was written:

- **LLMORPH (arXiv:2603.23611):** metamorphic relations can expose LLM
  inconsistencies without a full label set. `.282` uses this as methodology
  repair for hard-set repair.
- **From Untestable to Testable (arXiv:2603.24774):** metamorphic testing turns
  relations across executions into executable oracles. Carnot uses it as a
  supplement to Z3/runtime checks, not as a replacement.
- **MR-Coupler (arXiv:2604.10126):** functional coupling can seed metamorphic
  test cases. `.282` applies the idea to solver/validator-coupled repair
  stressors.
- **LAVE (arXiv:2602.00612):** partial outputs should remain extendable to a
  valid grammar. `.282` applies the idea to validator-tree viability checks.
- **Equilibrium Reasoners (arXiv:2605.21488):** iterative reasoning can be
  analyzed as convergence to task-conditioned attractors. `.282` adds a
  fixed-point diagnostic over existing traces.
- **Solve the Loop (arXiv:2605.12466):** attractor modules support adaptive
  fixed-point refinement. `.282` uses this as inspiration for FR-11 memory drift
  and convergence tests.
- **Universal Verifier (arXiv:2604.06240):** process/outcome separation and
  controllable/uncontrollable failure accounting inform `.282` matrix and
  capstone claim boundaries.
- **Extropic and Logical Intelligence public pages:** both remain relevant
  architecture context, but no `.282` claim may depend on TSU/Z1/Kona access.

## Architecture Snapshot

```text
                  exp2999 capstone v281 / matrix v15
                                  |
                                  v
                    exp3000 archive .281 and activate .282
                                  |
                                  v
                  exp3001 SOTA GGUF cache carry-forward gate
                         /              |                 \
                        /               |                  \
                       v                v                   v
      exp3002 metamorphic        exp3004 AquaForte/     exp3005 validator
      repair-oracle audit        BEAVER live retry      tree expansion
             |                         |                   |
             v                         |                   v
      exp3003 SOTA repair              |           exp3006 fixed-point
      rerun with metamorphic           |           energy diagnostic
      false-accept evidence            |                   |
             \                         |                   v
              \                        |           exp3007 FR-11 attractor
               \                       |           trace memory stability
                \                      |                   |
                 ---------------> exp3010 matrix v16 <-----+

        Hardware side branch:
          exp3008 GateMate host-visible IO transport
              |
              v
          exp3009 SSQA dual-BRAM RTL/PnR/resource report
              |
              v
          exp3010 matrix v16
              |
              v
          exp3011 capstone v282
```

## Phase Structure

### Phase A: Activation and Evidence Gates

1. **exp3000 - Archive `.281` and activate `.282`.**
   Archive `.281`, carry forward unresolved rows explicitly, and make the
   staged roadmap active.

2. **exp3001 - SOTA GGUF cache carry-forward and checksum refresh.**
   Refresh the `.281` SOTA local GGUF evidence and provide the upstream gate for
   live LLM tasks. At least one mandated headline model must run for
   `sota_headline_ready=true`.

3. **exp3002 - Metamorphic repair-oracle audit.**
   Build deterministic relation-preserving repair variants and false-accept
   accounting around the flagged `exp2991` hard set. No live LLM repair happens
   here.

### Phase B: Claim Repair and Validator Expansion

4. **exp3003 - Gated SOTA repair rerun with metamorphic checks.**
   Rerun repair only if the SOTA and metamorphic-oracle gates pass. Promotion
   requires positive deltas plus clean false-accept and tautology evidence.

5. **exp3004 - AquaForte/BEAVER live retry provenance v2.**
   Rerun only the live retry substrate with durable model checksum, transcript,
   and duration evidence; enumerator fallback remains separate.

6. **exp3005 - Solver-to-validator tree expansion.**
   Expand the clean solver/protocol line into a larger deterministic validator
   tree corpus with prompt-to-validator viability checks.

7. **exp3006 - EqR fixed-point energy diagnostic.**
   Measure convergence, basin sensitivity, and fixed-point stability over cached
   solver/validator trajectories. This is diagnostic only, not a native EqR
   implementation claim.

### Phase C: Continuous Self-Learning and Hardware Grounding

8. **exp3007 - FR-11 attractor trace-memory stability.**
   Continuous self-learning task. Stress verifier-grounded trace memory with
   drift, convergence, negative-control, and forgetting checks inspired by
   attractor/fixed-point work.

9. **exp3008 - GateMate host-visible IO transport v2.**
   Add or conclusively diagnose a minimal host-visible IO path. No sampler or
   speedup claim is allowed.

10. **exp3009 - SSQA dual-BRAM RTL/PnR/resource report v2.**
    Produce the missing SSQA artifact only when the IO boundary supports it;
    otherwise gate-skip cleanly with no Sonnet call wasted.

### Phase D: Synthesis and Go/No-Go

11. **exp3010 - Cross-corpus matrix v16.**
    Aggregate `.282` against the PRD, paper-v6 claim boundaries, OpenSpec,
    hardware, and FR-11 requirements. The matrix must run even if upstream tasks
    are blocked or flagged.

12. **exp3011 - Milestone `.282` capstone.**
    Decide whether claim repair succeeded, list blocked/flagged rows, and
    recommend the next exact milestone. No publication action.

## Dependency Graph

```text
exp3000
  -> exp3001
       -> exp3003
       -> exp3004
  -> exp3002 -> exp3003
  -> exp3005 -> exp3006 -> exp3007
exp3008 -> exp3009
exp3003 -> exp3010
exp3004 -> exp3010
exp3007 -> exp3010
exp3009 -> exp3010
exp3010 -> exp3011
```

Structured conductor gates are included for:

- `exp3003`: requires `exp3001.sota_headline_ready == true` and
  `exp3002.metamorphic_oracle_ready == true`.
- `exp3004`: requires `exp3001.sota_headline_ready == true`.
- `exp3006`: requires `exp3005.validator_tree_expanded == true`.
- `exp3007`: requires `exp3006.fixed_point_diagnostic_ready == true`.
- `exp3009`: requires `exp3008.host_visible_io_ready == true`.
- `exp3011`: requires `exp3010.matrix_v16_ready == true`.

`exp3010` intentionally has no hard gate. It is the aggregation task that must
still run when upstream rows are blocked, flagged, or gate-skipped.

## Hardware Requirements

- **Dual RTX 3090 CUDA host:** required for live local SOTA GGUF cache refresh,
  repair rerun, and AquaForte/BEAVER live retry. Legacy small models are allowed
  only as CPU smoke tests and cannot support headline fields.
- **CPU-only path:** acceptable for archive, deterministic metamorphic-audit,
  validator expansion, fixed-point diagnostics over cached traces, matrix, and
  capstone tasks.
- **GateMate A1:** required for `exp3008`. Advancement requires host-visible
  output, readback, or a precise blocked diagnosis. Flash-only or failed-flash
  evidence remains non-promotable.
- **SSQA/GateMate RTL/PnR:** `exp3009` may produce RTL, PnR, resource reports,
  and testbench/smoke hooks. It must not claim sampler speed, Boltzmann
  correctness, thermodynamic behavior, or FPGA acceleration without observable
  sample/timing evidence.
- **Extropic/THRML and Kona/Aleph:** public context only. No task depends on
  authenticated TSU/Z1/XTR-0 or Kona internals.

## Acceptance Criteria

- `research-references.md` contains the post-`.281` sweep before the roadmap
  design.
- `research-roadmap-next.yaml` declares milestone `2026.05.282` and leaves
  `research-roadmap.yaml` unchanged.
- Every live LLM task includes the mandated SOTA GGUF models in `MODEL_SPECS`
  and records model/cache/provenance fields.
- The hard-set repair row cannot be promoted unless metamorphic variants,
  tautology checks, and false-accept accounting are clean.
- AquaForte/BEAVER cannot be promoted unless live retry duration, transcript,
  model checksum, and enumerator separation are all durable.
- FR-11 includes at least one continuous self-learning experiment with
  independent held-out metrics, negative controls, drift checks, and forgetting
  checks.
- GateMate/SSQA claims remain bounded to host-visible IO, RTL/PnR/resource, and
  smoke-hook evidence.
- Matrix and capstone preserve the `.281` claim narrowing: no KV260/GateMate
  speedup, no Extropic/Z1 hardware claim, no Kona parity claim, no broad
  self-learning claim beyond verifier-grounded trace memory stability.

## Failed-Experiment Rerun Compliance

Carry-forward tasks include `prior_failures` entries with mandatory
`retire_if_same_verdict: true` for the relevant blocked/flagged lineages:

- `exp2991` flagged repair methodology.
- `exp2993` AquaForte/BEAVER substrate corrigendum requiring cleaner live retry
  provenance.
- `exp2996` GateMate blocked flash/readback/smoke evidence.
- `exp2997` SSQA gate-blocked/missing artifact.
- `exp2998` and `exp2999` synthesis/capstone rows that ended with
  `paper_ready=false`.

The new tasks do not depend on retired upstream IDs from
`ops/exclusion_manifest.yaml`.

## Out of Scope

- Publishing or broadening paper-v6 claims.
- New WOPR/game cartridges, GRPO/VPRM reruns, SpecAnn revival, PIMI revival, or
  retired THRML/host-SD/OTV scopes.
- Claiming live Extropic/TSU/Z1, Kona, photonic, or quantum hardware access.
- Claiming GateMate/KV260 acceleration or thermalization without board-visible
  sample/timing evidence.
- Treating LLM judges, metamorphic consistency, or prompt schemas as substitutes
  for executable verifiers.
