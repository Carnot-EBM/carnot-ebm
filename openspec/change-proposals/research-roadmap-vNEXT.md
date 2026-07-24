# Carnot Research Roadmap vNEXT — Milestone 2026.07.524

**Milestone:** 2026.07.524  
**Title:** Grounded Constraint IR, Shortcut-Safe Self-Learning, and Structured Live Memory  
**Status:** Proposed  
**Task range:** Exp5890-Exp5903 (14 experiments)  
**Execution file:** `research-roadmap-next.yaml`  
**Date planned:** 2026-07-24

---

## What Milestone 2026.07.523 Actually Proved

Milestone `.523` reached a terminal conductor state with six activated tasks, not
the 13 tasks described in its planning document. Proposed Exp5883-Exp5889 were
never activated and are not evidence.

| Area | Terminal evidence | Consequence for `.524` |
|---|---|---|
| Boundary | Exp5877 archived the seven activated `.522` tasks and preserved absent, blocked, retired, ready, and unactivated identities. | Exp5890 must archive exactly the six activated `.523` tasks and must reserve Exp5890-Exp5903 without treating Exp5883-Exp5889 as completed. |
| Source currency | Exp5878 completed a post-marker sweep with zero accepted additions. Its artifact also exposed a methodology warning: external primary-source synthesis must not be mislabeled as live-model inference. | Exp5891 searches only after `V524-PLANNER-REFRESH-20260724-END` and keeps `aggregation_from_external_primary_sources` distinct from LLM inference. |
| Hardness surface | Exp5879 preserved the immutable Exp5868 rows, separated non-oracle nuisance controls from oracle-derived telemetry, found non-oracle maximum AUROC `0.583333`, and emitted `hardness_surface_headroom_ready_score=1.0`. | The scientific admission conditions are met, but the conductor retired the task after three `artifact_not_updated_past_bootstrap` failures because its artifact remained `blocked: science_ready_but_unrelated_global_suite_debt`. Exp5892 creates a changed evidence-escrow admission path; no task may depend on retired Exp5879 directly. |
| Shortcut science | Exp5880 was pre-emptively skipped because Exp5879 was retired. Exp5881 emitted only a conductor gate-block artifact because Exp5880 was absent. Exp5882 was pre-emptively skipped because Exp5881 was retired. | `.523` did not test the ICLP shortcut-grounding hypothesis or continuous self-learning. `.524` may run it only behind a new non-retired admission artifact. |
| Current-model internal energy | The `.523` layer-dynamic tasks were not activated. Planning-time inspection of installed `llama-cpp-python==0.3.33` found public final embeddings and logits but no public intermediate-layer state API. | Do not spend a task on a surface already known to be unavailable, do not patch private llama.cpp internals, and do not fall back to the disqualified Exp5853 final-embedding route. |
| ARC | No ARC task was activated in `.523`. Prior compact-ledger, leave-one-game-out interaction, and active-observation experiments remain clean nulls. | The mandatory ARC generalization slot must test a materially different memory architecture on the live E3 path, not re-solve any of the 25 cleared public games. |

The exact lesson is operational as well as scientific: a valid scalar inside a
retired upstream artifact is not a safe conductor dependency. `.524` first
creates a clean, immutable evidence-admission boundary, then fans out into
independent grounded-reasoning and live-memory branches.

---

## The Three Largest Gaps to the PRD Vision

### Gap 1 — Natural language still does not reliably become executable constraints

The PRD requires real constraint extraction, not a hand-authored fixture or a
model-written answer that is scored afterward. Carnot has exact validators,
typed semantic-grounding utilities, and rule-specific extractors, but no
engine-neutral constraint IR benchmark that measures translation, compilation,
proof-trace inspection, and repair across current local SOTA models.

**Milestone response:** Exp5896 builds an exact typed-IR fixture. Exp5897 compares
single-pass extraction, exact-trace-guided repair, and matched-compute no-trace
controls on all three mandated GGUF families. Exp5898 tests bounded
constraint-wise recursive improvement. Exp5899 performs portability, leakage,
and camouflage audits.

### Gap 2 — Positive self-learning is not yet shortcut-resistant

Exp5858 showed that bounded external structural memory can preserve a prospective
lift with only 1.67% of the full exact-query budget and zero unsafe accepts. It
did not establish that learned structure represents the intended semantics
rather than a satisfying shortcut, biased grounding, or nuisance surface.

**Milestone response:** Exp5893 builds paired exact shortcut rows, Exp5894 tests
one-to-one logical-atom grounding against soft/distributed alternatives, and
Exp5895 runs the accepted mechanism prospectively with exact future-event
validation, an AREX-inspired unresolved-constraint state, quarantine, rollback,
protected-prefix retention, and a backend-neutral hardware mapping.

### Gap 3 — The live ARC agent lacks a task-relevant persistent evidence architecture

The scored `E3AgentPolicy` has a compact ledger and optional observation/action
mechanisms, but prior experiments showed no held-out lift. PRO-LONG suggests
lossless programmatic access; arXiv:2607.21571 sharpens the hypothesis by showing
that persistence alone is insufficient when memory omits task-relevant semantic
evidence.

**Milestone response:** Exp5900 defines an append-only agent-owned event tape and
a structured evidence index over identical bytes. Exp5901 requires exact
retrieval, deletion utility, stale/shuffled controls, and bounded query cost.
Exp5902 runs a no-memory/raw-tape/structured-index live E3 A/B on held-out
measurement episodes. No public level is targeted or re-solved.

---

## Research Findings Incorporated

The dated source ledger is
`research-references.md#v524-planner-refresh---20260724`.

| Finding | Milestone use | Authority boundary |
|---|---|---|
| Differentiable Logic Programming to Mitigate Reasoning Shortcuts (`2607.21185`) | Exp5893-Exp5895: constraint-satisfaction/cognition shortcuts, one-to-one atom grounding, prospective shortcut-safe learning | Exact semantic and solver labels own acceptance; learned energy/rankers never certify themselves. |
| Euclid-MCP (`2607.21412`) | Exp5896-Exp5899: engine-neutral typed IR and translate-run-inspect-repair | Parser, compiler, exact solver, and replay traces are authoritative; LLM prose and self-scores are not. |
| AREX (`2607.21461`) | Exp5895 and Exp5898: bounded unresolved-constraint state and targeted recursive improvement | Only exact-validated state updates persist; learned compression, weight mutation, and model-authored labels are excluded. |
| PRO-LONG (`2607.20064v2`) | Exp5900-Exp5902: programmatic access to a complete agent-owned interaction tape | Import the memory mechanism, not public ARC scores, closed APIs, Docker harnesses, or prior-game leakage. |
| Beyond Episodic Evaluation (`2607.21571`) | Exp5900-Exp5902: compare structured task-relevant evidence with raw tape and no memory | Live E3 receipts and agent-owned observations/actions only; no source inspection, offline BFS, or per-game adapter. |
| Pipelined p-computer (`2607.21077`) and Extropic Z1 availability page | Hardware mapping and future ABI context only | No board or TSU execution claim without an authenticated local route and same-input reference receipt. |

OpenReview, Hugging Face Papers, Semantic Scholar, GitHub Trending, Extropic,
and Logical Intelligence yielded no other source that supersedes exact
validators or reopens final embeddings, generated-answer repair, KAN mutation,
unchanged board probes, or public ARC solves.

---

## Target Architecture

```text
       natural-language requirement / chronological exact event
                              |
                              v
   Qwen3.6-35B-A3B | Gemma-4-31B | Gemma-4-26B-A4B
             immutable local GGUF proposal models
                              |
                    typed ConstraintIR proposal
                              |
           +------------------+------------------+
           |                                     |
           v                                     v
   exact parser/compiler                learned external-state
   Z3/Python/optional Prolog            energy/ranker (advisory)
           |                                     |
           v                                     |
   proof/violation trace ------------------------+
           |
   unresolved-constraint queue
           |
   bounded inspect/repair loop
           |
           v
   exact semantic-equivalence gate
           |
     accept / quarantine / rollback
           |
           +--> protected bounded constraint memory
                         |
                         v
                  future-event evaluation

   E3AgentPolicy live action/observation stream
                         |
             append-only agent-owned tape
                    /             \
              raw queries     structured evidence index
                    \             /
             budget-matched policy context
                         |
                  submitted live E3 path
```

The learned components rank proposals or decide which exact check to request.
They never replace the exact parser, compiler, solver, semantic replay, or live
ARC environment.

---

## Phase 0 — Boundary and Evidence Admission (Exp5890-Exp5892)

### Exp5890 — Exact `.523` transition and collision-free allocation

Archive exactly Exp5877-Exp5882. Preserve Exp5879's operational failure and
scientific-ready scalar simultaneously; preserve Exp5880/Exp5882 as missing
pre-emptive skips and Exp5881 as a gate-block artifact. Record Exp5883-Exp5889
as unactivated proposal IDs. Prove Exp5890-Exp5903 collision-free.

**Deliverable:** `results/experiment_5890_transition_v524.json`

### Exp5891 — Post-V524 source-delta receipt

Search only after `V524-PLANNER-REFRESH-20260724-END`. Zero accepted findings is
a valid complete result. A new item may sharpen controls inside existing `.524`
tasks but may not change task identities, gates, model policy, or retired-scope
boundaries.

**Deliverable:** `results/experiment_5891_v524_source_delta_ingestion.json`

### Exp5892 — Immutable headroom evidence escrow

Do not rerun or mutate Exp5868, Exp5869, or Exp5879. Re-adjudicate their hashes
and scientific gate receipts through a new evidence-escrow contract. Freshly
run the owned checks and current relevant verification/spec checks. Global
failures may be classified as unrelated only with exact failing node IDs,
ownership paths, and proof that they cannot alter the fixture, audit, schemas, or
gate fields.

**Deliverable:** `results/experiment_5892_headroom_evidence_escrow.json`  
**Gate:** `headroom_admission_ready_score == 1.0`.

---

## Phase 1 — Grounded Constraint Learning and Exact IR (Exp5893-Exp5899)

### Exp5893 — Exact grounding-shortcut fixture

Extend the immutable hardness surface with canonical, constraint-satisfaction
shortcut, cognition-shortcut, one-to-one, soft/distributed, shuffled,
frequency-balanced, label-permuted, and no-information grounding pairs. Separate
`intended_semantics` from `encoded_constraint_outcome` and certify both exactly.

**Deliverables:**

- `results/experiment_5893_grounding_shortcut_fixture.json`
- `results/experiment_5893_grounding_shortcut_fixture.rows.jsonl`

**Gate:** `grounding_shortcut_fixture_ready_score == 1.0`.

### Exp5894 — One-to-one atom-grounding structural-acquisition A/B

Compare one-to-one logical-atom grounding with soft/distributed, shuffled,
frequency-matched, and parameter-matched controls. Use chronology-safe
train/admit/future splits, held grounding families, and exact semantic labels.
The learner remains bounded external state.

**Deliverable:** `results/experiment_5894_one_to_one_grounding_ab.json`  
**Gate:** `one_to_one_grounding_ready_score == 1.0` with positive held-family
lower bounds and zero unsafe accepts.

### Exp5895 — Prospective shortcut-resistant continuous self-learning

Run the accepted grounding mechanism through exact future-event validation. Add
an AREX-inspired state containing verified evidence plus unresolved constraints;
compare it with fixed validated memory and no-memory controls. Require bounded
state, quarantine, rollback, rejected-update non-propagation, protected-prefix
retention, held-family lift, and no weight mutation.

**Deliverable:** `results/experiment_5895_shortcut_safe_continuous_self_learning.json`  
**Gate:** `shortcut_resistant_csl_ready_score == 1.0`.

### Exp5896 — Engine-neutral typed ConstraintIR fixture

Define a minimal typed IR for facts, Horn implications, finite-domain predicates,
arithmetic relations, negation, and query goals. Compile it to existing exact
backends. Build exact natural-language/IR/certificate rows with held templates,
paraphrases, symbol renamings, invalid IR, unsatisfiable IR, and
semantic-non-equivalence controls.

**Deliverables:**

- `results/experiment_5896_typed_constraint_ir_fixture.json`
- `results/experiment_5896_typed_constraint_ir_fixture.rows.jsonl`

**Gate:** `typed_constraint_ir_fixture_ready_score == 1.0`.

### Exp5897 — Three-family translate-run-inspect-repair A/B

On all three mandated GGUF families, compare:

1. single-pass typed-IR extraction;
2. one exact parser/solver-trace-guided repair;
3. a matched two-call no-trace control.

Measure parse validity, compilation, exact semantic equivalence, unsafe accepted
constraints, latency, tokens, and GPU receipts on group-held templates and
domains. This is constraint extraction, not generated-answer repair.

**Deliverable:** `results/experiment_5897_sota_constraint_ir_repair_ab.json`  
**Gate:** `trace_repair_mechanism_ready_score == 1.0`.

### Exp5898 — Constraint-wise recursive improvement

Use at most three exact-check rounds. Compare an unresolved-constraint scheduler
with first-error, random-error, and matched-compute controls. The model receives
only parser/compiler/solver trace information that would be available in a real
run; it never sees hidden answer labels. Model weights stay immutable.

**Deliverable:** `results/experiment_5898_recursive_constraint_improvement.json`  
**Gate:** `recursive_constraint_improvement_ready_score == 1.0`.

### Exp5899 — Portability, leakage, and camouflage audit

Audit the accepted recursive path without fitting a new model. Hold out model
family, constraint family, templates, names, and render. Swap exact evaluators,
shuffle or delete traces, equalize length/token budgets, and test whether
surface, model identity, or direct label proxies explain the apparent gain.

**Deliverable:** `results/experiment_5899_constraint_repair_portability_audit.json`  
**Gate:** `constraint_repair_portability_ready_score == 1.0`.

---

## Phase 2 — Structured Live ARC Memory (Exp5900-Exp5902)

### Exp5900 — Agent-owned tape and structured evidence-index contract

Add a default-off append-only tape to the scored `E3AgentPolicy` boundary. The
raw and structured arms must contain identical agent-owned events. The
structured index may expose time, object/glyph, spatial relation, action effect,
uncertainty, and evidence provenance, but it may not inspect game source, load
prior-game logs, invoke offline BFS, or encode per-game rules.

**Deliverable:** `results/experiment_5900_arc_structured_evidence_memory_contract.json`  
**Gate:** `structured_evidence_memory_contract_ready_score == 1.0`.

### Exp5901 — Retrieval fidelity and causal necessity

Use deterministic agent-owned traces to test exact retrieval fidelity, shuffled
index, stale evidence, index deletion, evidence deletion, irrelevant growth,
bounded query count, byte budget, and latency. Compare no memory, raw tape, and
structured index over identical bytes. Promotion requires a causal advantage,
not merely more context.

**Deliverable:** `results/experiment_5901_arc_structured_memory_causal_audit.json`  
**Gate:** `structured_memory_causal_ready_score == 1.0`.

### Exp5902 — Adapter-disabled live E3 memory A/B

Registry-precheck first: all 25 public games are already cleared, so the task
must not target or headline a level solve. On preregistered held-out measurement
episodes, run no memory, raw tape, and structured index with identical action,
token, wall-clock, and query budgets. Use at least Qwen3.6-35B-A3B and
Gemma-4-26B-A4B. Any incidental solve is credited only if independently new and
reproduced by the submitted live mechanism.

**Deliverable:** `results/experiment_5902_arc_structured_memory_live_ab.json`  
**Gate:** `structured_memory_live_ready_score == 1.0` only for a positive
preregistered lower bound with no safety or budget regression.

---

## Phase 3 — Reconciliation (Exp5903)

### Exp5903 — Milestone reconciliation

Aggregate every activated task by exact task ID and declared deliverable.
Classify each branch independently as ready, null, blocked, disqualified,
retired, gate-blocked, or missing. Re-run fresh adversarial verification on
present artifacts, append `.524` exactly once to completion history, reconcile
specs/traceability/status/changelog, and produce the next three falsifiable
recommendations. Never convert an upstream scalar inside a retired artifact into
a completed downstream experiment.

**Deliverable:** `results/experiment_5903_v524_capstone_reconciliation.json`

---

## Dependency Graph

```text
Exp5890 transition ──────────────────────────────────────────────────────┐
Exp5891 source receipt ──────────────────────────────────────────────────┤
                                                                         │
Exp5868 + Exp5869 + Exp5879 immutable evidence                           │
                    └──> Exp5892 evidence escrow                         │
                              └──> Exp5893 shortcut fixture              │
                                         └──> Exp5894 grounding A/B     │
                                                   └──> Exp5895 CSL      │
                                                                         │
existing exact extractors ──> Exp5896 typed IR fixture                   │
                                    └──> Exp5897 SOTA repair A/B         │
                                              └──> Exp5898 recursion     │
                                                        └──> Exp5899 audit
                                                                         ├──> Exp5903
Exp5726/5766/5860 null evidence                                           │
                    └──> Exp5900 structured memory contract              │
                              └──> Exp5901 causal audit                   │
                                        └──> Exp5902 live E3 A/B         │
                                                                         │
all terminal task/deliverable identities ────────────────────────────────┘
```

The IR and ARC branches are independent of Exp5892 so a headroom-admission
failure cannot erase the whole milestone. Exp5903 is deliberately ungated and
must reconcile blocked branches.

---

## Model Policy

Every task that invokes an LLM must define `MODEL_SPECS` with at least one
mandated local GGUF. Headline tasks use:

| Experiment | Required headline models | Purpose |
|---|---|---|
| Exp5897 | Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it | Single-pass versus exact-trace repair |
| Exp5898 | the same three families | Bounded recursive constraint improvement |
| Exp5902 | Qwen3.6-35B-A3B and Gemma-4-26B-A4B-it; Gemma-4-31B confirmatory if budget permits | Live E3 memory A/B |

`cached_sota_pair()` is the required cache-resolution pattern. Qwen3.5-0.8B and
Gemma-4-E4B-it may run CPU smoke tests only and may not support a headline
verdict. All GGUF weights remain immutable.

---

## Hardware Requirements

| Resource | Tasks | Requirement and claim boundary |
|---|---|---|
| CPU/RAM/disk | Exp5890-Exp5896, Exp5899-Exp5901, Exp5903 | Existing host, deterministic solvers, atomic JSON/JSONL writes, exact replay, bounded external state |
| Dual RTX 3090, 24 GB each | Exp5897, Exp5898, Exp5902 | Healthy CUDA, cache resolution, real GPU offload/utilization/VRAM receipts, no protected concurrent workload |
| KV260 / PolarFire / GateMate | none | No unchanged continuity probe. Exp5865 retired the available adaptive-kernel chain and Exp5861 found no changed authenticated route. |
| Extropic XTR-0/Z1 | none | Public availability context only; no authenticated local route |

Exp5895 must emit a backend-neutral state-operation map for insert, quarantine,
lookup, supersede, rollback, and sparse energy/ranking. This is an interface
contract, not a hardware speedup claim. A future hardware task may open only
after a ready bounded kernel and an authenticated changed route both exist.

---

## Failed-Experiment Discipline

The YAML carries four-field `prior_failures` entries where scope overlaps a
failed or retired experiment:

- Exp5892 addresses Exp5869 and Exp5879 by separating immutable scientific
  admission from producer-task retirement and by proving any remaining suite
  debt unrelated at exact node/path granularity.
- Exp5894 addresses Exp5749, Exp5773, and gate-blocked Exp5881 with a non-KAN,
  exact-fixture, one-to-one grounding mechanism.
- Exp5895 addresses Exp5750, Exp5787, and Exp5867 with a qualified grounding
  upstream and deterministic bounded external state rather than the retired
  adaptive-kernel chain.
- Exp5900-Exp5902 address Exp5726, Exp5766, and Exp5860 with identical-byte raw
  versus structured lossless memory, rather than compact cross-game summaries,
  component interactions, or active observation selection.
- Exp5903 addresses blocked Exp5862 through branch-independent exact
  reconciliation that cannot be cascade-skipped.

Every entry sets `retire_if_same_verdict: true`. No gate or dependency names a
retired upstream experiment.

---

## Explicit Scope Exclusions

This milestone does not authorize:

- rerunning the retired PHASE D generated-text/logprob verifier;
- grammar, stop-token, or parser tuning for the retired finite-ID answer channel;
- final-embedding verification after Exp5853;
- private llama.cpp hooks or a custom intermediate-layer fork;
- KAN rendering, knot mutation, or adaptive-kernel requalification;
- model-weight writes, GRPO, or model-authored training labels;
- generated-answer repair disguised as constraint extraction;
- source inspection, offline BFS, per-game adapters, or public ARC re-solves;
- unchanged board probes, RTL redesign, TSU execution, or hardware speedup claims;
- Kona/Aleph execution without public weights or an authenticated endpoint;
- external publication, push, or modification of `scripts/research_conductor.py`.

---

## Milestone Success Criteria

The milestone succeeds operationally when all 14 activated identities reach an
honestly classified terminal state and Exp5903 reconciles them. Scientific
promotion remains branch-specific:

1. **Grounding/CSL:** positive held-family lower bounds, zero unsafe accepts or
   transfers, exact retention, quarantine, rollback, and bounded state.
2. **Constraint extraction:** exact semantic-equivalence lift over both
   single-pass and matched-compute no-trace controls, no label leakage, and
   survival under held-model/family/render audits.
3. **ARC memory:** exact retrieval plus deletion utility and a positive
   preregistered live-path lower bound for structured indexing over both raw
   tape and no memory, with no registry laundering.
4. **Evidence discipline:** zero history amplification, zero retired dependency,
   zero protected-file mutation, exact field provenance, and fresh adversarial
   verification of every present artifact.

Null results are complete results. A blocked precondition is not a scientific
null, and a scalar inside a retired artifact is not a completed downstream
experiment.
