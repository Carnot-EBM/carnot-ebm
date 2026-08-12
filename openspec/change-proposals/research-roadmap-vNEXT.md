# Research Roadmap V546: Certified Constraint Evolution and Causal Influence

**Milestone:** `2026.08.546`
**Date:** 2026-08-12
**Status:** Proposed
**Task range:** Exp6337-Exp6349
**Execution file:** `research-roadmap-next.yaml`

## Milestone thesis

Carnot has an exact constraint substrate and a positive factor-local learning
result. It does not yet have a useful constrained generator, an anytime-valid
release process, or evidence that its ARC route changes live actions. V546
tests these three boundaries.

The milestone gives constrained generation one final, changed-technique
attempt. The attempt acts during decoding. It uses parser state and SMT prefix
feasibility. A semantic-diversity canary must pass before a held utility run.
The branch retires if it repeats the V545 null.

The main scientific line is continuous self-learning. V546 adds an e-value
release ledger and evidence-carrying factor lifecycle. It then runs one sealed,
chronological learning trial. The base GGUF weights stay frozen. Exact factors
keep release authority.

The ARC line stays narrow. It measures whether a target-licensed route changes
the live agent's own action order. It does not solve a game for the agent. It
makes no game or level solve claim.

## What V545 proved

| Evidence | Result | V546 consequence |
|---|---|---|
| Exp6324 post-marker source freeze | `complete_null`; no accepted source delta in its window | Start a new dated source window. Do not widen scope from old product claims. |
| Exp6325 GateMate detect | `blocked_detect_failed`; one command failed and no retry ran | Run no unchanged GateMate task. Require a new physical receipt before any future board command. |
| Exp6326 restricted policy compiler | Ready score `1.0`; bounded semantics and verified fallbacks passed | Reuse the DSL, exact compiler, and fallback. Do not replace them. |
| Exp6327 guarded synthesis | `complete_null`; exact-factor search did not add value | Close post-hoc energy search. Test only an incremental decoding method with a fail-fast canary. |
| Exp6328 blind integrity audit | `complete_ready`; safety passed but utility promotion stayed zero | Keep the blind checker. Do not count exact-oracle safety as model utility. |
| Exp6329 held-family A/B | `complete_null`; every search delta over guard-only was zero | Require a new causal mechanism and retire it if the held result is again null. |
| Exp6318 and Exp6319 carry-forward | Positive factor-local learning and feedback-directed search | Add release statistics, lifecycle control, and prospective chronology. |
| Exp6320 carry-forward | Safety audit passed while utility promotion stayed blocked | Reuse rollback and protected-validation attacks. Add e-process and compaction attacks. |
| Exp6321 carry-forward | Target-licensed live ARC shadow reached the live route without solve credit | Test action influence only. Keep the route default-off. |

Exp6323 did not produce an artifact after three hard wall-clock failures. The
transition work is therefore split and bounded in Exp6337. Exp6330-Exp6336
were proposal-only identifiers in the old design. They did not enter the V545
conductor queue. V546 starts at Exp6337 so those identities are not reused.

## The three largest gaps

### Gap 1: Exact guards do not yet improve generation

The exact policy compiler is sound. The V545 search method did not improve a
single held cell. Carnot still needs evidence that constraints can change the
generator before a complete candidate exists.

V546 builds incremental parser state and JIT SMT prefix checks. It first asks
whether they create more unique, valid program semantics at a matched budget.
Only a positive canary opens the held utility trial.

### Gap 2: Positive self-learning lacks a release process and lifecycle

The V544 learner can update factor-local state. It lacks an anytime-valid
certificate under repeated looks. It also lacks a safe method to merge or
delete stale factors. Without lifecycle control, constraint memory can grow
without bound and preserve obsolete rules.

V546 binds each factor to evidence, rationale, exact replay, and lineage. An
e-process controls release under optional stopping. A prospective trial then
tests useful, safe evolution over chronological events.

### Gap 3: ARC routing is reachable but not causally useful

The target-licensed route can run on live-agent evidence. Carnot has not shown
that the route changes which action the live agent takes next. Reachability is
not influence.

V546 first uses counterfactual replay to pre-register eligible windows. It then
runs a default-off A/B on fresh live attempts. The endpoint is action-order and
exact transition quality. Solve credit remains out of scope.

## Research findings used in V546

The full dated search record is in `research-references.md` under the V546
planner marker.

| Finding | Experiment use |
|---|---|
| Parser-state bias correction, arXiv:2608.10137 | Exp6339 and Exp6340 use incremental parser state. |
| LeJIT, HotNets 2025 | Exp6339 adds JIT SMT prefix feasibility. |
| NxN E-valuation, arXiv:2608.06621 | Exp6342 builds the anytime release ledger. |
| Catastrophic Remembering, arXiv:2608.11095 | Exp6343 adds evidence and rationale to factor lifecycle decisions. |
| Verification-cost evaluation, arXiv:2608.08709 | Exp6340, Exp6341, Exp6345, and Exp6348 report exact-check cost. |

## Architecture

```text
                       frozen local GGUF model
                      /         |          \
          Qwen3.6-35B-A3B   Gemma-4-31B   Gemma-4-26B-A4B
                      \         |          /
                       token / action proposals
                                  |
                 +----------------+----------------+
                 |                                 |
       restricted policy path               live ARC path
                 |                                 |
       parser and lexer state               own attempt history
                 |                                 |
       JIT SMT prefix feasibility          target-licensed route
                 |                                 |
       exact policy contract guard         counterfactual action order
                 |                                 |
       verified fallback                    default-off action A/B
                 |                                 |
                 +---------------+-----------------+
                                 |
                         exact outcome evidence
                                 |
                     minimized counterexample
                                 |
              evidence-carrying factor proposal
                                 |
           versioned factor graph and rollback state
                                 |
              anytime e-value release certificate
                                 |
                 release / retain / merge / delete
```

The exact compiler, exact outcome checker, and ARC transition checker are
oracles. They can prove contract results. They do not count as learned
verification. The GGUF models can propose programs, factors, and actions. They
cannot approve their own output.

## Phase 0: Evidence boundary and transition

### Exp6337 - Bounded V545 terminal handoff

Classify the seven queued V545 tasks and their terminal records. Preserve the
missing Exp6323 artifact as a failure receipt. Record that Exp6330-Exp6336 were
never queued. Validate the 13 V546 task identities, gates, and deliverables.

**Deliverable:** `results/experiment_6337_v546_bounded_terminal_handoff.json`

### Exp6338 - New source-window scope freeze

Search only after the V546 marker. Record accepted, duplicate, watch-only,
inaccessible, and excluded sources. Freeze the three scientific lanes and the
no-hardware boundary.

**Deliverable:** `results/experiment_6338_v546_post_marker_source_scope_freeze.json`

## Phase 1: Prefix-constrained generation

### Exp6339 - Incremental prefix enforcement substrate

Extend the restricted policy DSL with deterministic parser states. Add a JIT
SMT prefix-feasibility interface. Prove prefix soundness and completion
parity against the existing exact compiler. This task makes no LLM call.

**Deliverable:** `results/experiment_6339_incremental_prefix_enforcement_substrate.json`

### Exp6340 - Parser and JIT semantic-diversity canary

Use all three mandatory local GGUF models. Compare unconstrained decoding,
grammar masking, parser-state correction, and JIT prefix enforcement. Freeze
the token, call, wall-time, and checker budgets. The endpoint is the count of
unique valid normalized semantics. Utility is secondary and cannot open the
next task by itself.

**Deliverable:** `results/experiment_6340_parser_jit_semantic_diversity_canary.json`

### Exp6341 - Prospective held-family prefix utility A/B

Run only if Exp6340 reports `semantic_diversity_gain_score == 1.0`. Seal new
held families before generation. Compare exact guard plus fallback against
the best pre-declared prefix method. Require positive fallback-adjusted utility
with zero accepted violations. Retire this constrained-generation scope if it
repeats the V545 null.

**Deliverable:** `results/experiment_6341_prospective_prefix_utility_ab.json`

## Phase 2: Anytime-certified continuous self-learning

### Exp6342 - Anytime e-value release ledger

Implement an immutable e-process for repeated factor-update decisions. Test
the null, alternatives, optional stopping, restarts, duplicate evidence, and
selection after observation. Keep the exact outcome checker as the oracle.
This task makes no LLM call.

**Deliverable:** `results/experiment_6342_anytime_evalue_release_ledger.json`

### Exp6343 - Evidence-carrying factor lifecycle

Give each learned factor a rationale, exact counterexample, replay witness,
lineage, and retention set. Permit merge or deletion only when exact replay,
protected retention, and rollback checks pass. Bound memory growth. This is a
Tier 4 structural self-learning task. It makes no LLM call.

**Deliverable:** `results/experiment_6343_evidence_carrying_factor_lifecycle.json`

### Exp6344 - Counterexample-to-factor proposal calibration

Use all three mandatory local GGUF models. Give each model only the changed
factor, minimized counterexample, and permitted edit schema. Compare random
valid edits, repeated sampling, stability-regularized proposals, and
counterexample-directed proposals. The exact checker supplies all labels.

**Deliverable:** `results/experiment_6344_counterexample_factor_proposal_calibration.json`

### Exp6345 - Prospective chronological certified evolution trial

Run only if the e-value ledger, lifecycle, and proposal calibration are ready.
Seal the event order and protected validation once. Compare a frozen champion,
the V544 fixed lifecycle, and certified evidence-carrying evolution. Report
future exact yield, rollback, factor growth, verification cost, and
catastrophic-remembering events.

**Deliverable:** `results/experiment_6345_prospective_certified_factor_evolution_ab.json`

### Exp6346 - Independent certificate and lifecycle safety audit

Attack optional stopping, e-value reset, duplicated evidence, selected nulls,
rationale laundering, witness swaps, unsafe factor merge, harmful deletion,
protected-set reuse, restart faults, and rollback failure. Safety success
cannot promote utility. This task makes no LLM call.

**Deliverable:** `results/experiment_6346_certified_factor_evolution_safety_audit.json`

## Phase 3: ARC causal influence and reconciliation

### Exp6347 - ARC counterfactual action-influence preflight

Replay the live agent's own Exp6321 attempts. Compare action rankings with the
target-licensed route on and off. Pre-register only windows where the route can
change a legal next action without hidden game source, offline BFS, or a
per-game adapter. Make no solve claim.

**Deliverable:** `results/experiment_6347_arc_action_influence_preflight.json`

### Exp6348 - Default-off live ARC action-influence A/B

Run only if Exp6347 reports `arc_action_influence_eligible_score == 1.0`. Use
fresh live attempts from mandatory local GGUF models. Compare route off and
route on at matched budgets. Measure legal action-order changes, exact
transition quality, and verification cost. Do not update the solve registry.

**Deliverable:** `results/experiment_6348_arc_default_off_action_influence_ab.json`

### Exp6349 - V546 adversarial capstone

Recompute every dependency and structured gate from terminal artifacts. Check
model receipts, oracle boundaries, prior-failure retirement, ARC provenance,
hardware non-use, tests, spec coverage, and operations documents. Reconcile
the roadmap, archive, status, changelog, and traceability record.

**Deliverable:** `results/experiment_6349_v546_adversarial_capstone.json`

## Dependency graph

```text
Exp6337 bounded handoff
   |
Exp6338 source freeze
   |--------------------------|--------------------------|
   v                          v                          v
Exp6339 prefix substrate   Exp6342 e-value ledger    Exp6347 ARC preflight
   |                          |                          |
Exp6340 diversity canary   Exp6343 factor lifecycle     | gate = 1.0
   | gate = 1.0                |                          v
Exp6341 held utility A/B   Exp6344 proposal test     Exp6348 live A/B
                              |
                    all three readiness gates = 1.0
                              |
                           Exp6345
                              |
                           Exp6346
                              |
        Exp6341 ----------- Exp6349 <----------- Exp6348
```

Exp6349 consumes terminal or structured skip artifacts. A failed gate is an
expected scientific result. It must not trigger an ungated replacement task.

## Pre-registered gates and retirement rules

| Downstream task | Gate | Meaning |
|---|---|---|
| Exp6340 | Exp6339 `prefix_enforcement_substrate_ready_score == 1.0` | Prefix checks are sound and complete before model use. |
| Exp6341 | Exp6340 `semantic_diversity_gain_score == 1.0` | A prefix method improved unique valid semantics at matched cost. |
| Exp6345 | Exp6342, Exp6343, and Exp6344 readiness scores all equal `1.0` | Release statistics, lifecycle, and proposals are independently ready. |
| Exp6348 | Exp6347 `arc_action_influence_eligible_score == 1.0` | The live route can affect a legal action before fresh A/B work. |

Exp6341 declares Exp6327 and Exp6329 as prior nulls. If Exp6341 reaches the
same terminal null, the scope must enter `ops/exclusion_manifest.yaml`. Future
work must change the model substrate or receive an operator override.

Exp6337 declares Exp6323 as a failed transition. If the bounded replacement
again ends without an artifact, retire this transition shape and move terminal
classification into the capstone only.

## Model policy

Every LLM experiment uses the canonical local llama.cpp route and each GGUF's
embedded tokenizer. `MODEL_SPECS` must use `cached_sota_pair(gpu_indices=(0,
1))` or the canonical helper that supersedes it. No task may use
`AutoTokenizer` for these GGUFs.

The required model set is:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6340, Exp6341, Exp6344, and Exp6345 use all three. Exp6348 uses at least
Qwen3.6-35B-A3B and Gemma-4-31B. Legacy small models may run CPU smoke tests
only. They cannot support a headline claim.

## Hardware requirements

### Required local compute

- Two RTX 3090 GPUs with 24 GB VRAM each.
- Working CUDA llama.cpp offload.
- Cached files for all three mandatory GGUF families.
- Enough local disk for raw generation rows, sealed manifests, and model
  hashes.
- CPU and RAM for exact enumeration, Z3, e-process replay, and tests.

GPU tasks must load one placement at a time unless a precondition receipt
proves a safe dual placement. Each task must record VRAM before load, after
load, and after release. A failed model cell remains visible. CPU replay cannot
stand in for a missing headline model.

### Explicitly not required

- GateMate: Exp6325 exhausted the dated receipt with one failed detect.
- KV260: the current line is terminal and has no V546 workload delta.
- Extropic TSU: no authenticated local access exists.
- Kona: no public local weights or API exist.
- AMD eGPU, NPU, and other FPGA boards: none has a new receipt tied to a V546
  question.

No V546 task may claim hardware speed, power, or sampling results.

## Acceptance boundary

V546 succeeds as a research milestone if all 13 tasks reach honest terminal or
structured skipped artifacts and the capstone reconciles the record. The
scientific branches can still end null.

The strongest possible claim is narrow:

1. prefix enforcement improves held exact utility after a positive diversity
   canary, or the constrained-generation branch retires;
2. evidence-carrying factor updates pass an anytime-valid chronological trial
   and independent safety audit, or certified self-learning remains blocked;
3. the target-licensed ARC route causally changes live action quality, or it
   remains a reachable shadow only.

V546 does not claim AGI, broad hallucination elimination, learned-verifier
soundness, ARC game solves, or hardware acceleration.
