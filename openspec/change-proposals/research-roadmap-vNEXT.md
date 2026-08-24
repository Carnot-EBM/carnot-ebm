# Carnot Research Roadmap vNEXT: Auditable Proof Graphs and Poison-Resistant Self-Learning

**Milestone:** 2026.08.571  
**Created:** 2026-08-24  
**Status:** Proposed  
**Supersedes:** milestone 2026.08.570 after its four-task execution  
**Experiment range:** Exp 6575-6586  
**Primary references:** `research-program.md`, `_bmad/prd.md`,
`research-references.md`, and the terminal V570 artifacts

## Purpose

V570 removed the model-runtime blocker. Content-derived metadata identified all
three hash-only GGUF blobs. Actual sequential llama.cpp execution then admitted
Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B-it. The milestone also
froze the source-span joint-sufficiency method.

The raw result is useful, but the evidence chain is not yet eligible for
downstream science. Exp6571, Exp6572, and Exp6573 received
`DURATION_TOO_SHORT` findings. Exp6573 completed all three model receipts in
52.068 seconds, below the 60-second floor for its declared live-inference
substrate. A positive honest verdict does not override a structural finding.
V571 therefore creates fresh qualifying receipts. It does not edit, relabel, or
silently import the flagged V570 aggregate artifacts.

After that repair, V571 executes the science that V570 prepared. It creates one
immutable all-family source stream, tests semantic-block joint proof graphs,
audits grounds/norms/authority counterfactuals, and runs prospective continuous
self-learning with protected-core and bounded-memory controls. It then makes
bounded Rust, ARC, and hardware decisions.

The milestone has four phases and 12 experiments. No task may modify
`research-roadmap.yaml` or `scripts/research_conductor.py`.

## What V570 Proved

| Evidence | Result | V571 consequence |
|---|---|---|
| Exp6571 evidence root | Correctly classified V569 blocked and missing work. Froze model, gate, retirement, ARC, hardware, and protected-file rules. The artifact later received a duration-floor finding. | Reuse its contract as context only. Recompute eligibility from fresh V571 receipts. |
| Exp6572 GGUF metadata | All three hash-only flagship blobs passed bounded content inspection and independent cache provenance. Negative fixtures failed closed. The artifact later received a duration-floor finding. | Keep the resolver implementation. Re-run its focused checks inside a fresh, complete evidence workload. |
| Exp6573 flagship admission | All three mandated families produced authentic sequential runtime receipts with no blocked family. The artifact later received a duration-floor finding at 52.068 seconds. | Run a fresh, preregistered evidence-qualification workload. Do not cite the flagged aggregate as model admission authority. |
| Exp6574 joint-sufficiency contract | Source-byte nodes, dependency edges, exact reducers, splits, arms, fixtures, attacks, and retirement rules were frozen. This artifact was clean. | Use the frozen method. Add only the new semantic-block and counterfactual controls from the V571 research refresh. |

V570 proved that the local runtime can reach every flagship family and that the
joint-proof method is executable. It did not produce an eligible source stream,
a proof-extraction result, or a continuous-learning result.

## The Three Biggest Gaps to the PRD Vision

### Gap 1: no clean flagship evidence stream

PRD FR11 and FR12 require real local inference and exact verification. The
runtime works, but its V570 aggregate evidence is structurally flagged. Carnot
still lacks a clean all-family stream with immutable prompts, sources, raw
responses, process receipts, lineage, and recomputable cost.

### Gap 2: source text does not yet become auditable joint proof

Carnot has a frozen method contract, not a live extraction result. It must show
that source-bound atomic obligations form a well-formed dependency graph, that
all required hops are covered, and that each release traces through source
bytes, block-owned rules, exact checks, and one frozen reducer. It must also
show that controlled changes to grounds, norms, or authority cause the expected
decision change.

### Gap 3: continuous self-learning has no flagship, prospective safety result

FR11 remains open at the current SOTA substrate. Existing transactional memory
and exact replay primitives are not enough. The system needs a chronological
run with within-query freezing, exact commits, retention, held-future support,
restart, rollback, false-memory attacks, and source-occupancy controls. It must
separate current gain from support collapse and verifier circularity.

Rust speed, ARC supervisor influence, and physical hardware remain important.
They are bounded decisions, not substitutes for these three gaps.

## Research Refresh Incorporated

The dated V571 refresh was added to `research-references.md` before this design.
Five findings change the experiments:

1. **AI with Authority** (`arXiv:2608.21356`) motivates a kernel-like,
   link-by-link evidence chain from raw receipt to release claim.
2. **Semantic-Block Model** (`arXiv:2608.19475`) adds acyclicity, single
   ownership, constraint domination, and totality-or-ambiguity-stop checks to
   the joint proof graph.
3. **No Judgment Without a Reason** (`arXiv:2608.20938`) adds independent
   grounds, norms, and authority counterfactual receipts.
4. **SPARCL** (`arXiv:2608.21307`) adds a protected trusted core and
   residual-only updates to the graph-Potts learning control.
5. **Utility Under Attack** (`arXiv:2608.21230`) adds false-memory attacks and
   bounded source/family occupancy instead of relying on additive provenance.

CellFill (`arXiv:2608.20873`) is promising for reversible in-cell weight
updates, but it is not executable through Carnot's reviewed llama.cpp GGUF
path. V571 keeps generator weights frozen. It does not patch cached GGUFs.

The refresh found no authenticated Extropic TSU route and no public Kona
weights or local runner. KAN training remains closed because compact function
approximation is not the current bottleneck. The retired external-text scorer,
generated ConstraintIR, answer-ID transport, and schema-reprompt lanes remain
closed.

## V571 Architecture

```text
       flagged V570 aggregates                    clean Exp6574 method
                |                                      contract
                | context only                            |
                v                                         v
    fresh all-family runtime receipts ------------> V571 evidence root
                |                                      |
                v                                      v
      immutable source + raw response rows --> independent stream audit
                |                                      |
                +------------------+-------------------+
                                   v
             source-bound semantic blocks and dependency edges
                    |       |       |       |
                    |       |       |       +--> totality or abstain
                    |       |       +----------> constraint domination
                    |       +------------------> single rule ownership
                    +--------------------------> acyclicity
                                   |
                                   v
                      exact joint release reducer
                                   |
                 +-----------------+------------------+
                 |                                    |
                 v                                    v
      grounds/norms/authority receipts       chronological exact outcomes
                 |                                    |
                 v                 +------------------+------------------+
        independent proof audit    |          |          |               |
                                   v          v          v               v
                                frozen     uniform    graph-Potts   protected-core
                                memory      replay                 + occupancy caps
                                   \          |          |               /
                                    +---------+----------+--------------+
                                                      |
                                                      v
                                     exact transactional memory commit
                                    retention | future support | attacks
                                        restart | rollback | cost
                                                      |
                         +----------------------------+-------------------+
                         v                            v                   v
                fused Rust decision         live ARC receipts    hardware receipts
                         \___________________________|___________________/
                                                      v
                                      adversarial V571 capstone
```

Exact executable checks remain release authority. A learned extractor may
propose graph nodes and spans. A graph-Potts estimator may select a challenge.
Neither may certify its own output or commit unverified memory.

## Phase 0: Clean Evidence and Immutable Flagship Stream

### Exp6575 - clean evidence and flagship qualification replay

Build a new V571 evidence root from fresh work. Re-run bounded GGUF metadata
checks, negative fixtures, and actual one-model-at-a-time generation for all
three flagship families. Record process and GPU samples, unload and recovery,
raw receipt hashes, evidence-link rows, live verifier results, and monotonic
duration. Do not modify or relabel V570 artifacts.

**Gate:** `v571_flagship_evidence_ready_score=1.0` only if all three fresh
family receipts pass, the clean Exp6574 method contract replays, every evidence
link resolves, protected files are unchanged, and the new terminal artifact is
not structurally flagged.

**Deliverable:**
`results/experiment_6575_v571_clean_evidence_and_flagship_qualification.json`

### Exp6576 - immutable all-family source-span stream

Run the three qualified local families on one content-pinned corpus. Freeze
prompts, sources, seeds, order, stop rules, and budgets before inference. Store
raw response bytes and failure rows before any extraction. Do not discard
malformed or claim-free responses.

**Gate:** `immutable_claim_stream_ready_score=1.0` requires qualified rows from
all three families, complete lineage, nonzero claim-bearing rows, and
recomputable charged costs. It does not require good proof extraction.

**Deliverable:**
`results/experiment_6576_immutable_flagship_source_span_stream_v3.json`

### Exp6577 - independent source-stream audit

Recompute source, prompt, response, model, process, token, order, and cost
hashes without trusting Exp6576 aggregates. Check family coverage, duplicate
rows, hidden filtering, post-outcome prompt changes, and row-to-headline
consistency. Always run, even if the upstream task blocked.

**Gate emitted:** `claim_stream_audit_ready_score=1.0` only if every eligible
raw row is independently reproducible and no family or failure class vanished.

**Deliverable:**
`results/experiment_6577_flagship_source_stream_independent_audit.json`

## Phase 1: Semantic-Block Joint Proof Graphs

### Exp6578 - semantic-block joint proof extractor

Compare no filter, atomic-span-only, and hop-conditioned joint-proof arms on
the same immutable rows. The graph arm must enforce acyclicity, single rule
ownership, constraint domination, and totality or ambiguity-stop. Every
released composed claim must trace from source bytes through obligation nodes,
dependency edges, exact checks, and the frozen reducer.

**Acceptance:** improve exact-certified composed-claim coverage over the
atomic-only arm without lower precision, unsafe release, lineage loss, or an
unbounded charged-cost increase. Oracle-defined success uses
`verdict_class=circular_positive`.

**Deliverable:**
`results/experiment_6578_semantic_block_joint_proof_extractor_v3.json`

### Exp6579 - counterfactual joint-proof audit

Always run. Recompute the proof rows and independently perturb one ground
(bound span), one norm (block-owned exact rule), or one authority (named exact
checker) at a time. Require the expected decision transition and a minimal
changed-link receipt. Detect graph cycles, duplicate ownership, undominated
constraints, hidden open questions, and release despite missing evidence.

**Gate emitted:** `joint_proof_audit_ready_score=1.0` only if proof rows and
counterfactual receipts replay and the upstream claim is not disqualified.

**Deliverable:**
`results/experiment_6579_counterfactual_joint_proof_independent_audit.json`

## Phase 2: Protected-Core Continuous Self-Learning

### Exp6580 - graph-Potts and protected-core conformance

Implement or harden the graph-Potts/Beta-Binomial estimator from the frozen
method. Add a protected trusted core whose statistics are invariant while
residual difficulty state updates. Test exact small cases, mean-field
convergence, clamps, cold start, disconnected graphs, occupancy caps,
chronological no-lookahead, restart equality, and rollback equality.

**Gate:** both `graph_potts_runtime_ready_score=1.0` and
`protected_core_runtime_ready_score=1.0` require all hand-computed and replay
fixtures to match. This is infrastructure evidence, not utility.

**Deliverable:**
`results/experiment_6580_graph_potts_protected_core_conformance.json`

### Exp6581 - prospective poison-resistant continuous self-learning

This is the required continuous-self-learning experiment. Compare frozen
memory, uniform verified replay, recent-failure, ordinary graph-Potts, and
protected-core graph-Potts with per-source and per-family occupancy caps. Use
one chronological flagship stream. Match dose, capacity, candidate pool,
write opportunities, seeds, and evaluation points. Keep model weights frozen.
Commit memory only after independent exact verification.

Inject preregistered false-memory and source-concentration attacks. Measure
current benefit, retention, held-future support, exact safety, occupancy,
charged cost, restart, and rollback after every transition.

**Acceptance:** the protected-core arm must improve a preregistered current or
future-support metric over matched uniform replay, preserve retention and exact
safety, resist the bounded attacks, and remain within the charged-cost bound.
Oracle-defined success is circular positive, not positive.

**Deliverable:**
`results/experiment_6581_prospective_poison_resistant_continuous_self_learning.json`

### Exp6582 - independent continuous-learning audit

Always run. Recompute arm equality, chronological visibility, selector state,
protected-core invariance, occupancy, memory hashes, exact certificates,
attacks, benefit, retention, future support, cost, restart, and rollback from
raw rows. Detect model-identity leakage, future-label leakage, same-query
mutation, and aggregate contradictions.

**Gate emitted:** `csl_audit_ready_score=1.0` only if the disposition is
independently reproducible and not disqualified.

**Deliverable:**
`results/experiment_6582_poison_resistant_csl_independent_audit.json`

## Phase 3: Production, North-Star Continuity, and Closeout

### Exp6583 - fused Rust joint-proof NFR01 final decision

Test one materially changed workload: a single PyO3 boundary for batched
semantic-block canonicalization, graph validation, exact relation dispatch,
and release reduction. Compare the current Python path and the fused Rust path
on identical proof rows. Require exact parity, disabled-path identity,
fallback, restart, rollback, p50, p95, p99, throughput, and end-to-end charged
latency.

**Decision:** promote only if safety passes and NFR01's 10x gate is met. A
repeated no-benefit or NFR01 miss retires this acceleration lane.

**Deliverable:**
`results/experiment_6583_fused_rust_joint_proof_nfr01_final.json`

### Exp6584 - prospective ARC supervisor receipt decision

Inspect only live trajectory-supervisor receipts newer than Exp6558. Compare
eligible pre-action redirects with matched unredirected live attempts. Do not
solve a public game, read game source, run offline ground-truth search, or
credit a development proxy. If no newer outcome-bearing receipt exists, block
with the exact missing check and run no replacement replay.

**Deliverable:**
`results/experiment_6584_arc_live_supervisor_prospective_receipts_v2.json`

### Exp6585 - hardware and external-substrate continuity

Audit KV260, PolarFire, GateMate, Extropic, and Kona receipts. Issue no board
command without a newer operator-authored state receipt. KV260 remains SSH-only.
GateMate remains physical-receipt-gated. PolarFire may run one bounded command
only if a new receipt opens the terminal workload. Extropic and Kona remain
non-local unless authenticated access exists.

**Deliverable:**
`results/experiment_6585_hardware_and_external_substrate_continuity.json`

### Exp6586 - V571 adversarial capstone and reconciliation

Always run. Recompute every gate and headline from row-level artifacts. Record
complete, circular-positive, null, partial, blocked, disqualified, retired, and
unrun tasks separately. Reconcile the research record, OpenSpec, traceability,
architecture freshness, status, changelog, and known issues. Do not convert a
blocked task into a null scientific result.

**Deliverable:**
`results/experiment_6586_v571_adversarial_capstone.json`

## Dependency Graph

```text
Exp6575 clean evidence qualification
   |
   +--> Exp6576 immutable source stream --> Exp6577 stream audit
   |                                           |
   |                                           v
   |                                 Exp6578 joint proof extractor
   |                                           |
   |                                           v
   |                                 Exp6579 counterfactual audit
   |                                           |
   +------------------> Exp6580 conformance ---+
                                               |
                                               v
                                  Exp6581 prospective self-learning
                                               |
                                               v
                                  Exp6582 independent CSL audit

Exp6579 --> Exp6583 fused Rust final decision
Exp6584 ARC receipts ----------- independent, always run
Exp6585 hardware continuity ---- independent, always run

Exp6575-Exp6585 --> Exp6586 capstone (logical fan-in; capstone always runs)
```

Structured conductor gates are only used when a task cannot produce useful
evidence without an upstream field. Independent audits and the capstone always
run so that a missing or blocked artifact receives a terminal diagnosis.

## Model and Inference Policy

All LLM-bearing tasks use local llama.cpp GGUF inference and declare their
models in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6575 and Exp6576 use all three families. Exp6578 and Exp6581 also include
all three so that no legacy smoke model carries a headline result. Legacy
Qwen3.5-0.8B and gemma-4-E4B-it may test CPU plumbing only. They cannot satisfy
admission, family coverage, extraction, or learning gates.

Each live task resolves a local `.gguf`, uses the embedded tokenizer, records
repository/revision/hash provenance, confirms CUDA offload, freezes seeds and
budgets, and writes process/GPU receipts. No task calls `AutoTokenizer` on a
GGUF repository.

## Hardware Requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Dual RTX 3090 | Exp6575, Exp6576, Exp6578, Exp6581 | Run one large GGUF at a time. Preserve unrelated GPU work. Confirm `llama_supports_gpu_offload()` and record sampled utilization and memory. |
| CPU and RAM | All tasks | Exact replay, graph reduction, Z3 or existing exact checkers, artifact audits, and model unload recovery. |
| Local model cache | Exp6575, Exp6576, Exp6578, Exp6581 | Content-derived GGUF identity and immutable repository/revision/hash receipts for all three mandated families. |
| Rust/PyO3 toolchain | Exp6583 | Existing workspace only. One fused boundary. No broad redesign or new runtime dependency. |
| KV260 | Exp6585 only | SSH-only continuity. No host storage probing. One bounded command only after a changed operator receipt. |
| PolarFire SoC Icicle | Exp6585 only | One bounded terminal-workload command only after a newer receipt explicitly opens it. |
| GateMate A1 | Exp6585 only | No command without a new dated physical-state receipt. DirtyJTAG remains receipt-gated. |
| Extropic TSU | None | No authenticated local device or API. Simulation cannot support hardware speed, power, or availability claims. |
| Kona | None | No public weights or documented local runner. Product comparison only. |

No new hardware purchase is required for V571.

## Prior Failures and Retirement Rules

| V571 task | Prior scope | Material change | Repeat rule |
|---|---|---|---|
| Exp6575 | Exp6571-Exp6573 duration-flagged evidence | Fresh receipts, a preregistered complete workload, link-by-link provenance, and live structural verification. No V570 artifact relabeling. | Retire this qualification ID if the same structural disposition repeats. |
| Exp6576 | Exp6568 gate-blocked source stream | Clean V571 family evidence exists before inference; raw rows are written before extraction. | Retire if it again blocks for the same upstream-evidence reason. |
| Exp6578 | Exp6569 produced no artifact | New semantic-block and joint-sufficiency mechanism over a clean stream; no prior honest verdict exists to list as `prior_failures`. | Missing support abstains; retired generated-ConstraintIR methods remain closed. |
| Exp6579 | Exp6570 blocked audit | V571 supplies immutable stream and graph rows plus grounds/norms/authority controls. | Retire if the same missing-input verdict repeats. |
| Exp6581 | Exp6553 blocked prospective CSL | Flagships are qualified; proof rows are audited; protected-core and occupancy mechanisms are new. | Retire if the same GPU/evidence precondition block repeats. |
| Exp6582 | Exp6554 blocked CSL audit | The audit targets emitted V571 transitions, attacks, and protected-core rows. | Retire if the same missing-live-evidence verdict repeats. |
| Exp6583 | Exp6563-Exp6564 production nulls | One fused semantic-block-to-release crossing replaces routing-only and per-call Rust paths. | A repeated no-benefit or NFR01 miss permanently retires the lane. |
| Exp6584 | Exp6558 no-policy-change result | Only newer prospective live receipts can reopen the decision. | Retire this ID if no new supported policy change is again found. |
| Exp6585 | Exp6559 missing GateMate receipt | Newer receipt search spans all active boards and external access; commands remain changed-state-only. | Retire this ID if the same missing-receipt verdict repeats. |
| Exp6586 | Prior milestone capstones | Standing milestone-transition authorization; V571-only evidence and reconciliation. | No science promotion from capstone prose. |

## Success, Null, Block, and Stop Conditions

### Success

- Fresh clean receipts qualify all three mandated model families.
- The immutable source stream contains all families and survives independent
  lineage and cost replay.
- Joint proof graphs improve exact-certified composed-claim coverage without a
  precision or safety loss.
- Counterfactual grounds, norms, and authority changes produce the expected
  minimal changed-link receipts.
- Protected-core graph-Potts learning improves current or future support over
  matched uniform replay while preserving retention, exact safety, attack
  resistance, restart, rollback, and cost bounds.

### Honest null

- A clean, powered, adequately sampled comparison shows no proof-graph gain.
- Protected-core graph-Potts is no better than matched uniform replay after all
  safety and retention checks pass.
- The fused Rust path passes parity but misses NFR01.
- New ARC receipts exist but support no policy change.

### Blocked

- A required upstream artifact or exact field is absent.
- A mandated GGUF cannot load or unload safely.
- Raw source, model, proof, or learning rows are not replayable.
- No newer ARC or physical-hardware receipt exists.

Every `blocked_*` verdict must include `gate_check_summary` with the failed
check and observed value.

### Disqualified

- A task imports a structurally flagged V570 aggregate as valid evidence.
- An LLM judge, extractor, selector, or self-report becomes release authority.
- Rows are filtered after outcomes, family identity leaks into a selector, or
  future labels enter an online decision.
- A release lacks source, graph, exact-check, or reducer provenance.
- A task uses a legacy smoke model for a headline claim.

### Stop and retire

- Do not repair V570 artifacts in place.
- Do not reopen generated ConstraintIR, schema reprompting, finite answer-ID
  transport, external-text scoring, or CPU-only GGUF offload.
- Retire the fused Rust lane after another measured no-benefit or NFR01 miss.
- Do not re-solve a public ARC game.
- Issue no unchanged physical-hardware command.

## Architecture Freshness

`_bmad/architecture.md` was last reconciled on 2026-07-03 and is older than 30
days at planning time. Each implementation task must cross-check current code,
terminal artifacts, and capability specs rather than treating the architecture
document as fresh authority. Exp6586 must either reconcile the changed proof
graph and continuous-learning surfaces or record a precise architecture-drift
item in `ops/known-issues.md`.

## Out of Scope

- Training a new foundation model, EBT, KAN, or CellFill GGUF.
- Editing cached model weights or quantization metadata.
- Treating source-span overlap, schema validity, or an LLM verdict as semantic
  correctness.
- Reproducing Extropic, Kona, SPADE, or distributed RL results.
- A new FPGA architecture or speed claim without changed physical evidence.
- Public ARC level solves, offline ground-truth search, or hand-built per-game
  adapters.
- Changes to `research-roadmap.yaml` or `scripts/research_conductor.py`.

## Completion Checklist

- [ ] Exp6575-Exp6586 each write the declared terminal JSON artifact.
- [ ] Every artifact declares `verdict_class` from the closed enum.
- [ ] Every blocked verdict includes `gate_check_summary`.
- [ ] Every comparative task emits per-unit rows and aggregate recomputation.
- [ ] Every live compute task records preconditions, model specs, seeds,
      hashes, process/GPU receipts, substrate, duration, and tests.
- [ ] Every gate field is declared identically in the upstream task.
- [ ] No task ID or dependency chain violates the exclusion manifest.
- [ ] Relevant unit, lint, spec-coverage, row-consistency, adversarial, and E2E
      checks pass or are recorded honestly.
- [ ] `research-complete.yaml`, OpenSpec, traceability, architecture freshness,
      `ops/status.md`, `ops/changelog.md`, and `ops/known-issues.md` agree.
- [ ] `research-roadmap.yaml` and `scripts/research_conductor.py` remain
      unchanged.
- [ ] Nothing is pushed.
