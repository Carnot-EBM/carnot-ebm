# Carnot Research Roadmap vNEXT: Joint Proof Graphs and Prospective Self-Learning

**Milestone:** 2026.08.570  
**Created:** 2026-08-23  
**Status:** Proposed  
**Supersedes:** milestone 2026.08.569 after its six-task execution  
**Experiment range:** Exp 6571-6584  
**Primary references:** `research-program.md`, `_bmad/prd.md`,
`research-references.md`, and the terminal V569 artifacts

## Purpose

V569 did not test its two headline methods. It stopped at a model-admission
contract defect. The three mandated GGUF repositories and large cached blobs
were present. Embedded tokenizers, CUDA llama.cpp support, and sequential GPU
headroom were also present. Admission still failed because a hash-only
Hugging Face cache path did not look like a named `.gguf` file. The preflight
therefore set `language_model_file=false` and `quantization_known=false`
without reading the GGUF header or attempting inference.

V570 fixes that exact root cause first. It then runs the unexecuted science on
an immutable prospective stream. It adds one method correction from the dated
literature refresh: independently supported spans are not enough for a
multi-hop claim. The extractor must prove hop-conditioned obligations and
joint sufficiency over their dependency graph.

The milestone has four phases and 14 experiments. It reserves infrastructure,
SOTA-ingestion, continuous-self-learning, independent-audit, production,
ARC, hardware-continuity, and capstone slots. No task may modify
`research-roadmap.yaml` or `scripts/research_conductor.py`.

## What V569 Proved

| Evidence | Result | V570 consequence |
|---|---|---|
| Exp 6565 evidence contract | V568 imports were content-addressed. Exp 6562 was disqualified. Exp 6563 and Exp 6564 were clean nulls. | Reuse the immutable evidence boundary. Do not revive the saturation headline or default-on production routing. |
| Exp 6566 method contract | Source-bound proof obligations and graph-Potts online difficulty estimation received executable local contracts. | Extend the proof method with joint sufficiency. Reuse the graph-Potts equations after a fresh conformance check. |
| Exp 6567 model admission | All model files resolved. Tokenizers, CUDA llama.cpp, and memory conditions passed. Hash-only cache paths failed filename-derived model and quantization tests. No model was invoked. | Inspect GGUF content metadata, then require real one-at-a-time generation receipts. Do not repeat path-shape inference. |
| Exp 6568 source stream | Structured gate saw `all_mandated_models_loaded_score=0.0`. No source stream ran. Its blocked artifact also carried the wrong `verdict_class`. | Retry only after authentic admission. Emit `verdict_class=blocked` for a gate block. |
| Exp 6569 extractor | The conductor skipped the task after its upstream was retired. No terminal artifact exists. | Treat proof extraction as unrun science. The new joint-sufficiency mechanism also separates V570 from retired full-ConstraintIR retries. |
| Exp 6570 independent audit | Correctly blocked because no recomputable stream, spans, compiler rows, release rows, harm rows, or cost rows existed. | Re-run as an always-run audit against immutable V570 rows. |

V569's scientific result is therefore **not null**. It is **not run because of
a prerequisite contract defect**. V570 must not count a repaired admission
receipt as evidence that extraction or self-learning works.

## The Three Biggest Gaps to the PRD Vision

### Gap 1: no authentic flagship evidence stream

PRD FR11 and FR12 require real local reasoning and verification. Current
headline candidates cannot begin while the three mandated families fail
admission before inference. The first gap is operational but load-bearing:
content-derived GGUF identity, actual generation, sampled GPU use, clean
unload, and immutable raw rows.

### Gap 2: source text does not yet become jointly sufficient exact evidence

The project can bind an atomic claim to a source span and compile an exact
obligation in fixtures. It has not shown that this works on live flagship
output. It also lacks a release rule for composed claims. A set of locally
supported spans can still omit a required hop. V570 introduces an obligation
dependency graph and a joint-sufficiency check.

### Gap 3: continuous self-learning lacks prospective, audited utility

Carnot has transactional memory and exact replay primitives, but its latest
prospective flagship learning task never loaded a model. FR11 remains open at
the current flagship substrate. V570 tests graph-Potts challenge selection on
one chronological stream with frozen weights, matched dose, exact commits,
retention, future support, restart, rollback, and an independent audit.

Production speed, ARC supervisor influence, and hardware continuity remain
important, but they cannot substitute for these three gaps. They are bounded
Phase 3 decisions after the evidence path is measured.

## Research Refresh Incorporated

The dated refresh is recorded in `research-references.md` before this roadmap.
The selected new source is arXiv:2608.00585, *Verification Without
Sufficiency*. Its result motivates a direct control:

1. bind each atomic claim to immutable source bytes;
2. decompose a composed claim into hop-conditioned obligations;
3. compile each obligation into a whitelisted exact check;
4. require the dependency graph to cover every needed hop;
5. abstain if any node, edge, span, or exact check is missing.

This mechanism does not generate full ConstraintIR. It does not use an LLM
judge as release authority. It does not reopen the retired schema-reprompt or
finite-ID answer-transport lanes.

The same refresh found no access change for Extropic Z1 or Kona. There is no
authenticated TSU device or API, and no public Kona weights or local runner.
They remain comparison targets, not execution substrates.

## V570 Architecture

```text
                  immutable V569 evidence and retirement root
                                      |
                                      v
 hash-only HF blob --> GGUF header reader --> actual sequential admission
                                               |  all 3 families
                                               v
                                  immutable raw flagship responses
                                               |
                                               v
 source bytes --> hop decomposition --> typed atomic obligations
      |                  |                       |
      |                  v                       v
      +----------> dependency graph ------> exact compiler/checker
                                               |
                   +---------------------------+------------------+
                   |                                              |
                   v                                              v
          joint-sufficiency gate                         independent audit
                   |                                              |
                   +---------------------+------------------------+
                                         v
                         chronological exact outcome stream
                                         |
                 +-----------------------+-----------------------+
                 |           |            |          |           |
                 v           v            v          v           v
             frozen       uniform      recent     bandit    graph-Potts
             memory        replay       failure               selector
                 |           |            |          |           |
                 +-----------+------------+----------+-----------+
                                         v
                       exact transactional memory commit
                       retention | future support | cost
                       restart | rollback | independent audit
                                         |
                 +-----------------------+----------------------+
                 v                       v                      v
        fused Rust decision       live ARC supervisor    hardware receipt
        and retirement            evidence only          continuity only
                 \_______________________|______________________/
                                         v
                         adversarial capstone reconciliation
```

Release authority remains exact and executable. The graph-Potts estimator is
a scheduler. It may decide which verified challenge to present next. It may
not certify claims or commit unsafe memory.

## Phase 0: Evidence and Authentic Runtime Admission

### Exp 6571 - V570 evidence, gate, and retirement root

Create one additive root over Exp 6565-6570. Re-run live artifact checks. Freeze
task IDs, deliverables, exact gate names, model rules, prior failures,
retirement rules, ARC provenance boundaries, and zero-unchanged-hardware-command
rules. It emits `v570_evidence_contract_ready_score` and carries forward
`rust_fusion_reopen_ready_score` only for the materially changed fused workload.

**Gate:** ready only if all imported evidence is hash-bound, V569 blocked and
missing artifacts are classified honestly, every downstream gate names a V570
field, and protected files are unchanged.

**Deliverable:** `results/experiment_6571_v570_evidence_gate_and_retirement_root.json`

### Exp 6572 - content-derived GGUF metadata resolver

Implement and test a bounded reader for GGUF magic, version, architecture,
quantization/file type, tokenizer metadata, shard identity, and repository
provenance. It must work when the cache path is a hash with no extension. It
must reject a non-GGUF blob, a truncated header, a tokenizer-only GGUF, and a
repository mismatch.

**Gate:** `gguf_blob_metadata_ready_score=1.0` only when all three cached
flagship blobs are identified from content and every negative fixture fails
closed.

**Deliverable:** `results/experiment_6572_content_derived_gguf_metadata_resolver.json`

### Exp 6573 - sequential flagship admission v2

Load exactly one model at a time. Generate a bounded token receipt. Sample
process and GPU memory while the model is live. Exit cleanly. Prove unload
before the next family. Use the three mandated repositories. Legacy small
models may only smoke-test CPU plumbing.

**Gate:** `all_mandated_models_loaded_score=1.0` requires authentic receipts
for Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B-it. Prediction or
filename inference cannot satisfy it.

**Deliverable:** `results/experiment_6573_sequential_flagship_gguf_admission_v2.json`

## Phase 1: Immutable Source Stream and Joint Proof Graphs

### Exp 6574 - SOTA joint-sufficiency method contract

Preregister hop decomposition, obligation-node and dependency-edge schemas,
source-byte binding, compiler ownership, exact releases, abstention, attacks,
frozen splits, acceptance gates, and retirement. Add small hand-checkable
single-hop, valid multi-hop, missing-hop, wrong-span, and cyclic-graph fixtures.

**Gate:** `joint_sufficiency_method_ready_score=1.0` only if all conformance
fixtures and source receipts replay and the release reducer is frozen before
live model results.

**Deliverable:** `results/experiment_6574_joint_sufficiency_method_contract.json`

### Exp 6575 - immutable flagship source-span claim stream v2

Run the three admitted families on a content-pinned corpus. Store prompt,
source bytes, raw response bytes, seeds, launch receipts, token counts,
latency, cost, parser diagnostics, and exact unit hashes before extraction.
Do not tune prompts or discard failures after seeing results.

**Gate:** `immutable_claim_stream_ready_score=1.0` requires qualified rows from
every family, complete lineage, nonzero claim-bearing rows, and recomputable
charged costs. It does not require good extraction accuracy.

**Deliverable:** `results/experiment_6575_immutable_source_span_claim_stream_v2.json`

### Exp 6576 - source-span joint proof extractor v2

Compare no-filter, atomic-span-only, and hop-conditioned joint-sufficiency
arms on the same immutable rows. The compiler, not a model, owns exact
obligations. Every release must trace through source bytes, obligation nodes,
dependency edges, exact checks, and the frozen reducer. Missing support means
abstention.

**Acceptance:** improve exact-certified composed-claim coverage over the
atomic-only arm without lower precision, unsafe release, lineage loss, or
unbounded charged cost. A circular exact result uses
`verdict_class=circular_positive`, never `positive`.

**Deliverable:** `results/experiment_6576_source_span_joint_proof_extractor_v2.json`

### Exp 6577 - independent joint-proof audit

Always run. Recompute raw hashes, spans, graph coverage, obligation hashes,
exact results, release decisions, harm, and cost without trusting Exp 6576
aggregates. It may confirm, narrow, disqualify, or block the extraction claim.

**Gate emitted for learning:** `joint_proof_audit_ready_score=1.0` only if all
rows are independently recomputable and the upstream disposition is not
disqualified.

**Deliverable:** `results/experiment_6577_joint_proof_independent_audit.json`

## Phase 2: Continuous Self-Learning

### Exp 6578 - graph-Potts estimator conformance and replay engine

Implement or harden the graph-Potts/Beta-Binomial online estimator from the
frozen method contract. Test exact small cases, mean-field convergence,
clamps, cold start, disconnected graphs, restart equality, rollback equality,
and chronological no-lookahead behavior. The estimator ranks verified
challenges only.

**Gate:** `graph_potts_runtime_ready_score=1.0` requires all hand-computed and
restart/rollback rows to match. This is infrastructure evidence, not utility.

**Deliverable:** `results/experiment_6578_graph_potts_estimator_conformance.json`

### Exp 6579 - prospective graph-Potts continuous self-learning

This is the milestone's required continuous-self-learning experiment. Compare
frozen memory, uniform verified replay, recent-failure, exact contextual
bandit, and graph-Potts selection on one chronological stream. Match dose,
capacity, seed, candidate pool, write opportunity, and evaluation points.
Keep model weights frozen. Commit memory only after exact verification. Measure
current performance, retention, held-future support, safety, cost, restart,
and rollback after every transition.

**Acceptance:** graph-Potts must improve a preregistered current or held-future
metric over matched uniform replay, preserve retention and exact safety, and
remain noninferior on charged cost. No result may be positive if its verifier
is the oracle; use `circular_positive` where appropriate.

**Deliverable:** `results/experiment_6579_prospective_graph_potts_continuous_self_learning.json`

### Exp 6580 - independent continuous-learning audit

Always run. Recompute arm equality, graph features, chronological order,
decisions, memory hashes, exact certificates, safety, current benefit,
retention, future support, cost, restart, and rollback from raw rows. Detect
model-identity leakage, future-label leakage, same-query mutation, aggregate
contradictions, and post-outcome threshold changes.

**Gate emitted for downstream work:** `csl_audit_ready_score=1.0` only if the
learning disposition is independently recomputable and not disqualified.

**Deliverable:** `results/experiment_6580_continuous_self_learning_independent_audit_v2.json`

## Phase 3: Production, North-Star Continuity, and Closeout

### Exp 6581 - fused Rust exact workload final decision

Test one materially changed production workload: a single PyO3 boundary for
batched obligation-node canonicalization, graph validation, exact relation
dispatch, and release reduction. Compare it with the current Python path on
identical rows. Require exact parity, disabled-path identity, fallback,
restart, rollback, p50, p95, p99, throughput, and end-to-end charged latency.

**Decision:** promote only if safety passes and NFR01's 10x gate is met on the
frozen workload. A repeated no-benefit or NFR01 miss permanently retires this
Safety-Net acceleration lane.

**Deliverable:** `results/experiment_6581_fused_rust_joint_proof_nfr01.json`

### Exp 6582 - prospective ARC supervisor receipt decision

Inspect new live trajectory-supervisor receipts only. Do not solve a public
game, read game source, run offline ground-truth BFS, or credit a development
proxy. Compare eligible pre-action redirects with matched unredirected live
attempts. Decide whether any supported supervisor policy change exists. If no
new outcome-bearing receipt exists, block with a precise summary and run no
replacement replay.

**Deliverable:** `results/experiment_6582_arc_live_supervisor_prospective_receipts.json`

### Exp 6583 - physical hardware and external substrate continuity

Audit KV260, PolarFire, GateMate, Extropic, and Kona access receipts. Issue no
board command unless an operator-authored, dated, board-specific changed-state
receipt is newer than the last attempt. A changed receipt permits only the
documented next command. Extropic and Kona remain no-execution comparators
without authenticated access.

**Deliverable:** `results/experiment_6583_hardware_external_substrate_continuity.json`

### Exp 6584 - adversarial milestone capstone

Always run. Recompute every claim from per-unit rows. Run live adversarial,
row-consistency, convention, authenticity, exclusion, gate, spec-coverage,
and applicable E2E checks. Reconcile OpenSpec, traceability, status, changelog,
the completed-experiment archive, and the next transition only where V570
shipped implementation or evidence. State what advanced the north star and
what retired.

**Deliverable:** `results/experiment_6584_v570_adversarial_capstone.json`

## Dependency Graph

```text
Exp6571 evidence root ----------------------------+-------------------+
    |                                             |                   |
    +--> Exp6572 GGUF metadata --> Exp6573 admission                 |
    |                                |                               |
    +--> Exp6574 joint method --------+                               |
                                     v                               |
                             Exp6575 source stream                    |
                                     |                               |
                                     v                               |
                             Exp6576 extractor                       |
                                     |                               |
                                     v                               |
                             Exp6577 audit                            |
                                     |                               |
                      +--------------+--------------+                 |
                      |                             |                 |
                      v                             v                 |
              Exp6578 Potts runtime -------> Exp6579 CSL             |
                                                   |                 |
                                                   v                 |
                                             Exp6580 audit           |
                                                   |                 |
                    +------------------------------+                 |
                    |                                                |
                    v                                                v
              Exp6581 Rust                                     Exp6582 ARC

Exp6571 -----------------------------------------------> Exp6583 hardware

Exp6571 + all terminal artifacts ----------------------> Exp6584 capstone
```

Exp 6577, Exp 6580, and Exp 6584 are always-run audits. They have no structured
gate. They must still write a terminal artifact when upstream evidence is
missing. Exp 6582 and Exp 6583 are receipt-driven and may honestly block.

## Model Policy

| Experiment | LLM use | Required headline models |
|---|---|---|
| 6573 | Runtime admission | All three mandated GGUF families |
| 6575 | Prospective source-stream generation | All three mandated GGUF families |
| 6576 | No new generation; consumes frozen Exp 6575 rows | Preserve all three family IDs in every comparison row |
| 6579 | No weight update; consumes the frozen chronological stream | Preserve all three family IDs and report family-stratified rows |
| 6582 | Reads new live receipts only | Any new LLM generation must include at least one mandated family and cannot use a legacy smoke model for a headline |

Mandated families:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

`Qwen3.5-0.8B` and `gemma-4-E4B-it` may only provide fast CPU smoke tests.
They cannot satisfy admission, source-stream, extraction, learning, or ARC
headline gates.

## Hardware Requirements

| Experiments | Compute | Memory or access | Expected time | Boundary |
|---|---|---|---|---|
| 6571, 6574, 6577, 6580, 6584 | CPU | 8-16 GB RAM | 20-90 min each | Audit and contract work. No LLM or board command. |
| 6572 | CPU, local disk | Access to cached GGUF blobs; bounded header reads | 30-60 min | Never copy or hash entire multi-GB blobs when existing trusted hashes suffice. Fail closed on content mismatch. |
| 6573 | NVIDIA GPU | One flagship model at a time; prefer the idle 24 GB device and preserve the busy device | 2-4 hours | Actual execution is fit authority. Record GPU/process receipts and unload between families. |
| 6575 | NVIDIA GPU | Sequential flagship inference; adequate disk for immutable raw rows | 4-8 hours | All three families. Stop cleanly on resource loss. No legacy-model substitution. |
| 6576 | CPU, optional GPU for existing exact modules | 16 GB RAM | 2-4 hours | Reuse immutable model output. Exact compiler/checker owns release. |
| 6578-6579 | CPU, optional GPU for vectorized software | 16 GB RAM | 2-5 hours each | Software graph-Potts only. No FPGA or TSU speed claim. |
| 6581 | CPU and Rust toolchain | PyO3 build; 16 GB RAM | 2-4 hours | Compare identical frozen rows. Promotion requires exact parity and 10x. |
| 6582 | Existing live ARC receipts | No external submission | 30-90 min | No game solve credit and no off-path solver. |
| 6583 | Receipt-dependent | KV260, PolarFire, or GateMate only after changed receipt | 20-60 min without receipt | No unchanged command. No Extropic or Kona execution claim. |

The architecture document was last reconciled on 2026-07-03, more than 30
days before this plan. Treat its diagrams as historical. V570 tasks must
cross-check current code, specs, status, artifacts, and hardware receipts
before implementation or execution.

## Prior-Failure and Retirement Discipline

- Exp 6571 addresses Exp 6565's earlier adversarial-duration inconsistency by
  recording the current clean live replay and rebuilding all V570 gates.
- Exp 6572 and Exp 6573 address Exp 6567 with a changed mechanism:
  content-derived GGUF metadata followed by actual inference.
- Exp 6575 addresses Exp 6568 and Exp 6562 with authentic admitted models,
  immutable raw rows, and no inherited saturation claim.
- Exp 6576 addresses Exps 5909, 5910, 5923, and 6569. It uses source-byte
  spans, compiler-owned atomic obligations, and a joint-sufficiency graph. It
  does not generate full ConstraintIR or do schema-only repair.
- Exp 6577 addresses Exp 6570 by auditing present immutable rows rather than
  assuming missing evidence.
- Exp 6579 addresses Exps 6496 and 6553 with a prospective flagship stream,
  matched controls, and a conformed graph-Potts runtime.
- Exp 6580 addresses Exp 6554 with complete upstream raw rows and an always-run
  audit contract.
- Exp 6581 addresses Exps 6563 and 6564 with a fused end-to-end workload. The
  same no-benefit verdict retires the lane.
- Exp 6582 addresses Exps 6524 and 6558 only through new prospective live
  receipts. It does not rerun an offline supervisor proxy.
- Exp 6583 addresses Exp 6559 only if a new physical receipt exists. The same
  no-receipt result preserves the block and issues zero hardware commands.

Every YAML entry carries all four required prior-failure fields where scope
matches. Each repeat uses `retire_if_same_verdict: true`.

## Success, Null, and Stop Conditions

V570 succeeds as a research milestone if it closes decisions honestly. It
does not require positive results.

- A repaired GGUF resolver is useful only if real admission receipts follow.
- A ready source stream is an evidence prerequisite, not an extraction win.
- An exact-authority extraction gain is circular. Class it
  `circular_positive`, not `positive`.
- A graph-Potts learning gain must survive independent recomputation,
  retention, future-support, safety, cost, restart, and rollback checks.
- A repeated Rust no-benefit or NFR01 miss retires the fused acceleration lane.
- Missing ARC or hardware receipts produce precise blocked artifacts and no
  substitute experiment.
- Any aggregate contradicted by its per-unit rows is disqualified.

## Explicitly Out of Scope

- generated full ConstraintIR or schema-only reprompting;
- finite-ID answer transport, parser tuning, or stop-token retries;
- an external-text EBM or LLM judge as release authority;
- model-weight updates during continuous self-learning;
- public-game ARC replay, offline ground-truth BFS, hand GameAdapters, or game
  solve credit;
- GateMate, KV260, or PolarFire commands without changed-state receipts;
- Extropic latency, power, availability, or execution claims;
- Kona reproduction without public weights and a documented local runner;
- new KAN training while the evidence and learning path is still blocked.

## Completion Checklist

- [ ] `research-roadmap-next.yaml` passes schema, gate, exclusion, and
      prior-failure validation.
- [ ] All 14 task prompts name unique deliverables and exact run commands.
- [ ] Every comparative task emits per-unit rows and recomputable aggregates.
- [ ] Every task declares `verdict_class`; every blocked result records
      `gate_check_summary`.
- [ ] Every structured gate names an identically spelled upstream required
      artifact field in this roadmap.
- [ ] Every LLM task follows the mandated flagship model policy.
- [ ] Continuous self-learning is prospective, chronological, exact-verified,
      transactional, reversible, and independently audited.
- [ ] Relevant unit, lint, spec-coverage, and E2E checks pass before closeout.
- [ ] OpenSpec, traceability, status, changelog, and completed research records
      reconcile with work that actually shipped.
- [ ] Protected files remain unchanged and nothing is pushed.
