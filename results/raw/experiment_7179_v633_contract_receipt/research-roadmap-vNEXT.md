# Carnot Research Roadmap vNEXT: Evidence Energy, Self-Evolving Constraint Memory, Live ARC Generalization, and Conserved-Sector Sampling

**Created:** 2026-09-09  
**Milestone:** `2026.09.632`  
**Status:** Planned; activates after milestone `2026.09.631` completes  
**Supersedes:** the V631 design contract  
**Task contract:** exactly 13 tasks, `exp7166` through `exp7178`, in the order listed below  
**Execution authority:** `research-roadmap-next.yaml`; this document and that YAML must agree exactly

## Outcome

V632 turns the exact entity/evidence fixture into a real Qwen3.8 measurement,
tests continuous constraint learning with held-out commit and rollback rules,
restores the mandatory live ARC generalization slot, and adds an exact
fixed-magnetization sampler path. The roadmap is deliberately not one chain:
an unchanged GPU conflict can block the claim/evidence capture and ARC tasks
without preventing the CPU self-learning or sampler branches from running.

## What V631 Proved

V631 completed all seven tasks that actually reached its activated YAML, but it
did not complete the fourteen-task contract described by the stale Markdown.
The evidence is operationally useful even though the science branches did not
run:

| Result | Evidence | Consequence for V632 |
|---|---|---|
| The Markdown/YAML contract was inconsistent. | Exp7159 observed 14 design rows and 7 active YAML rows and returned `complete_disqualified_v631_markdown_yaml_contract_mismatch`. | V632 emits both complete files with one literal 13-row contract and starts with an advisory independent audit. |
| The mandated model exists locally, but the runtime was unavailable. | Exp7160 found `unsloth/Qwen3.8-27B-GGUF` and a CUDA runner, but PID 233772 held memory on both RTX 3090s without a task-owned lease. | Each Qwen task performs its own non-destructive preflight and terminates blocked if the conflict remains; CPU work is not gated on it. |
| The gate machinery behaved honestly. | Exp7161 failed its readiness gate and retired after three identical blocks; exp7162-exp7165 were pre-emptively skipped. | Keep structured gates for true data dependencies, but never put unrelated branches behind a shared resource gate. |
| The exact fixture remains the strongest unfinished verification input. | Exp7158 already sealed 648 entity/evidence counterfactual rows with exact labels, source spans, pair IDs, and frozen energy terms. | Capture noisy Qwen structures once, then replay comparisons and audits on CPU. |
| A 26-second canary can be honest only as bounded generation. | Exp7153 was quarantined after declaring `model_full_generation`; the work was actually bounded generation. | V632 uses `model_full_generation` only for real multi-row generation and never disguises a canary as a scientific run. |

## Three Largest Gaps to the PRD Vision

### 1. Natural-language constraints still lack external-authority validation

Carnot has exact verifiers and a strong controlled fixture, but no completed
Qwen3.8 result showing that a candidate claim/evidence representation improves
support discrimination under source interventions. This is the direct gap to
PRD FR-1, FR-2, FR-4, and FR-12. V632 separates expensive structure capture
from CPU energy scoring and independent causal recomputation.

### 2. Continuous self-learning has no supersession-safe commit protocol

Earlier work showed that procedural memory can help, but V631 never exercised
its noisy-feedback stream or memory update. The remaining gap to FR-11 is a
durable learner that sees events chronologically, cannot read future labels,
commits only held-out-safe changes, keeps rejected edits, handles stale
constraints under a fixed inspection budget, and rolls back poison.

### 3. Live generalization and conserved-sector sampling are not established

The ARC live path still lacks an adapter-knowledge-withheld measurement, and
the sampler layer lacks an exact fixed-magnetization kernel and Rust parity.
These gaps block credible generalization claims and hardware-shaped sampling.
V632 addresses both without re-solving a public game, reading a game source,
or claiming execution on hardware that Carnot does not possess.

## 2025-2026 Research Inputs

The dated source sweep is recorded before this plan in
`research-references.md` under “V632 Planner Refresh”. The sources that change
the experiments are:

| Source | Mechanism adopted | Boundary retained |
|---|---|---|
| ReCite, arXiv:2609.09156 | Decouple claim localization, evidence structure, and reflective support verification. | Exact fixture labels remain the authority; no model judges itself. |
| Procedural Graphs, arXiv:2609.09153 | Contrast failures and successes; commit graph edits only when held-out evidence does not regress; retain rejected edits. | The local experiment does not reproduce the paper's LLM refiner. |
| Stale Constraints, arXiv:2608.25553 | Compare target-blind limiting-constraint priority with recency/uniform allocation under one fixed budget. | Forced-critical selection is oracle upper bound only. |
| Growing and Elastic Networks, arXiv:2608.01475 | Bound growth and prune dead memory nodes through explicit support tests. | No assumption that growth alone prevents forgetting. |
| Transformers as In-Context Samplers, arXiv:2609.08981 | Motivates tracking energy and sampling geometry. | Deferred: established GGUF serving does not expose the required layerwise state clouds. |
| SymStep, arXiv:2607.23055 | Keep exact feasibility separate from learned ranking. | No substitution of an LLM verifier for exact execution. |
| Fixed-magnetization Ising sampling, arXiv:2609.08873 | Use pair-swap proposals that preserve the conserved sector. | Local parity, stationarity, and mixing must be measured independently. |
| Extropic Z1T, 2026-09-04 | Measure fixed-degree graph placement and digital orchestration categories. | No Z1 access; no Z1 latency, power, energy, or parity claim. |

## V632 Architecture

```text
                       exact authorities already present
                exp7158 fixture             exact ARC runtime
                       |                            |
          +------------+------------+               |
          |                         |               |
          v                         v               v
  Qwen3.8 claim/evidence      chronological     live scored E3 policy
  trace capture (7167)        supersession      with adapter knowledge
          |                   stream (7170)     denied by PATH (7174)
          v                         |               |
  frozen structural energy         +---------+-----+
  comparison (7168)                |         |
          |                         v         v
          v                 held-out-safe   fixed-budget stale
  causal/leakage audit       procedural      memory allocation
  (7169)                     graph (7171)    (7172)
                                    \         /
                                     v       v
                                cold retention and
                                rollback audit (7173)

  exact finite Ising laws -> pair-swap sampler (7175)
                                      |
                                      v
                               Rust parity (7176)
                                      |
                                      v
                         cross-substrate placement audit
                                (7177; no board claim)

  contract audit (7166) and capstone (7178) observe all branches but gate none.
```

The design has four independent roots after the advisory contract check:
claim/evidence capture, the CPU supersession stream, live ARC, and the CPU
sampler benchmark. The largest transitive blast radius is three downstream
tasks, not the five-task GPU cascade from V631.

## Phase 1: Claim/Evidence Energy With Exact Authority

### Exp7166 — V632 exact Markdown and YAML task-contract preflight

Independently parse the design and activated YAML, compare all thirteen rows,
and validate IDs, order, titles, deliverables, gates, producer fields,
prior-failure records, model declarations, substrate classes, progress clauses,
and prompt tails. The result is advisory infrastructure evidence and does not
gate science.

### Exp7167 — Qwen3.8 claim/evidence trace capture

Run real full generation with the sole mandated model,
`unsloth/Qwen3.8-27B-GGUF` / `Qwen3.8-27B-Q4_K_M.gguf`, over a prospectively
sealed 48-row subset of exp7158. Preserve raw structured claim/evidence output,
direct support decisions, source spans, tokens, timing, GPU identity, and
task-owned process receipts. No energy-value conclusion is made here.

### Exp7168 — Frozen claim/evidence energy comparison

Replay exp7167 on CPU and compare a pre-registered structural energy with the
same model's direct decision, lexical overlap, missing-field ablations, and
seeded shuffled-evidence controls. Exact fixture labels authorize correctness.
The gate is completion, not a positive result.

### Exp7169 — Independent claim/evidence causal audit

Recompute the comparison from raw rows in a fresh process, audit label leakage,
swap evidence within matched pairs, mutate each energy term, and verify every
headline from per-unit rows. It can classify a complete comparison as positive,
null, circular, or disqualified without rewriting exp7168.

## Phase 2: Continuous Self-Learning Under Supersession

### Exp7170 — Supersession-aware chronological constraint stream

Build an immutable CPU stream from exact historical artifacts and exp7158
families. Each constraint is introduced, supported, contradicted, withdrawn,
or restored at a known time; feedback can be delayed or noisy. Seal the train,
online-validation, future-retention, and poison windows before any learner
runs. Authority-only labels stay outside learner-visible rows.

### Exp7171 — Held-out-safe procedural-graph constraint memory

Compare frozen memory, additive memory, and a bounded grow/prune procedural
graph. Candidate edits are derived only from past successes and failures,
accepted only when online held-out energy improves without future-support loss,
and otherwise retained in a rejection ledger. This is the milestone's required
continuous self-learning experiment.

### Exp7172 — Budgeted stale-memory allocation comparison

With exactly two memory inspections per decision, compare FIFO/recency,
uniform, target-blind limiting-constraint priority, and forced-critical oracle
upper bound. Measure stale-consistent errors, provenance-path inspection,
current exact decisions, and future-support retention on the same events.

### Exp7173 — Cold retention, poison, and rollback audit

In a fresh process, replay immutable stream events and memory checkpoints,
recompute the two Phase-2 comparisons, inject sealed poison, and prove that
rollback restores hashes and decision behavior. External upstream blocks are
`blocked`, never retryable `partial`.

## Phase 3: Live Generalization and Conserved-Sector Sampling

### Exp7174 — Live ARC adapter-path-withheld A/B

Satisfy the ARC generalization floor with the real scored
`E3AgentPolicy`/`make_carnot_agent` path. Use one registry-eligible historical
target only as a mechanism probe, not as a new solve claim. Run matched control
and adapter-withheld arms with an OS/path allowlist that prevents the scored
process from reading game source, per-game adapters, solve registries, prior
solve artifacts, or outer-loop ground truth. Both arms use the sole mandated
Qwen3.8 GGUF and identical budgets. Report level progress and live
self-discovery provenance; headline no duplicate solve.

### Exp7175 — Fixed-magnetization pair-swap sampler benchmark

Implement a pair-swap Metropolis kernel that preserves magnetization exactly.
Compare it with exact enumeration on small systems and with admissible software
controls on larger fixed sectors. Report distribution error, detailed balance,
sector violations, acceptance, autocorrelation, ESS/s, and wall time per seed.

### Exp7176 — Rust fixed-magnetization parity and throughput

Port only the completed exp7175 kernel through the existing Rust/PyO3 sampler
surface. Require bit-for-bit energy deltas on fixed proposals, distributional
parity across seeds, zero sector violations, and measured throughput. A speed
claim requires parity first.

## Phase 4: Hardware-Shaped Placement and Synthesis

### Exp7177 — Cross-substrate sampler placement audit

Measure the accepted sampler's graph degree, coupling quantization, state,
proposal, reduction, and host-orchestration costs against existing CPU, CUDA,
FPGA-interface, and public Z1T constraints. This is a placement audit, not a
board run. KV260 and PolarFire are already graduated. GateMate remains excluded
until the operator provides a newer `operator_authored: true` physical-state
attestation or a dated retirement override.

### Exp7178 — V632 independent capstone

Read every expected artifact directly, rerun current validation and adversarial
checks, reconstruct claims from per-unit rows, preserve blocked/null/partial and
quarantined evidence, and produce the next handoff. It is intentionally
ungated so a blocked branch cannot erase the milestone synthesis.

## Exact Task Contract

| Order | Task ID | Exact title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | `exp7166-v632-exact-contract-preflight` | V632 exact Markdown and YAML task-contract preflight | `results/experiment_7166_v632_contract_preflight.json` | none |
| 2 | `exp7167-qwen38-claim-evidence-trace-capture` | Qwen3.8 claim/evidence structured trace capture | `results/experiment_7167_v632_claim_evidence_trace_capture.json` | none |
| 3 | `exp7168-claim-evidence-energy-comparison` | Frozen claim/evidence structural-energy comparison | `results/experiment_7168_v632_claim_evidence_energy_comparison.json` | `exp7167-qwen38-claim-evidence-trace-capture.claim_evidence_trace_ready_score == 1` |
| 4 | `exp7169-claim-evidence-causal-audit` | Independent claim/evidence causal and leakage audit | `results/experiment_7169_v632_claim_evidence_causal_audit.json` | `exp7168-claim-evidence-energy-comparison.claim_evidence_comparison_complete_score == 1` |
| 5 | `exp7170-supersession-aware-constraint-stream` | Immutable supersession-aware chronological constraint stream | `results/experiment_7170_v632_supersession_constraint_stream.json` | none |
| 6 | `exp7171-procedural-graph-constraint-memory-csl` | Held-out-safe procedural-graph constraint-memory self-learning | `results/experiment_7171_v632_procedural_graph_memory_csl.json` | `exp7170-supersession-aware-constraint-stream.supersession_stream_ready_score == 1` |
| 7 | `exp7172-budgeted-stale-memory-allocation-ab` | Budgeted stale-constraint memory-allocation comparison | `results/experiment_7172_v632_stale_memory_allocation_ab.json` | `exp7170-supersession-aware-constraint-stream.supersession_stream_ready_score == 1` |
| 8 | `exp7173-cold-retention-rollback-audit` | Fresh-process self-learning retention and rollback audit | `results/experiment_7173_v632_cold_retention_rollback_audit.json` | `exp7171-procedural-graph-constraint-memory-csl.procedural_memory_comparison_complete_score == 1` AND `exp7172-budgeted-stale-memory-allocation-ab.stale_memory_allocation_complete_score == 1` |
| 9 | `exp7174-live-arc-adapter-path-withheld-ab` | Live ARC adapter-knowledge path-withheld generalization A/B | `results/experiment_7174_v632_live_arc_path_withheld_ab.json` | none |
| 10 | `exp7175-fixed-magnetization-pair-swap-benchmark` | Fixed-magnetization pair-swap sampler exact benchmark | `results/experiment_7175_v632_fixed_magnetization_sampler.json` | none |
| 11 | `exp7176-rust-fixed-magnetization-parity` | Rust fixed-magnetization sampler parity and throughput | `results/experiment_7176_v632_rust_fixed_magnetization_parity.json` | `exp7175-fixed-magnetization-pair-swap-benchmark.fixed_magnetization_benchmark_complete_score == 1` |
| 12 | `exp7177-sampler-hardware-placement-audit` | Cross-substrate fixed-magnetization sampler placement audit | `results/experiment_7177_v632_sampler_hardware_placement.json` | `exp7176-rust-fixed-magnetization-parity.rust_fixed_magnetization_comparison_complete_score == 1` |
| 13 | `exp7178-v632-independent-capstone` | V632 independent evidence capstone and next handoff | `results/experiment_7178_v632_capstone.json` | none |

## Dependency Graph

```text
7166  (advisory; no downstream gate)

7167 -> 7168 -> 7169

7170 -> 7171 --+
  |             +-> 7173
  +----> 7172 --+

7174  (independent live ARC)

7175 -> 7176 -> 7177

7178  (ungated synthesis of all expected states)
```

Structured gates require only producer completion/readiness fields, never a
positive scientific verdict. This lets a complete null remain useful. A stable
external block is terminal `blocked`, with the failed check recorded under the
exact `gate_check_summary` field.

## Model and Substrate Contract

Only two tasks invoke an LLM:

| Task | MODEL_SPECS | Work | Substrate class |
|---|---|---|---|
| Exp7167 | sole entry `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf`, Q4_K_M | 48-row structured generation | `model_full_generation` |
| Exp7174 | sole entry `unsloth/Qwen3.8-27B-GGUF`, `Qwen3.8-27B-Q4_K_M.gguf`, Q4_K_M | matched live ARC generative runs | `model_full_generation` |

Both use the tokenizer and chat template embedded in the GGUF through
llama.cpp. They download nothing and never substitute Qwen3.5-0.8B or
gemma-4-E4B-it for headline evidence. Those legacy models remain permissible
only in explicitly non-headline CPU smoke tests, and none is needed here.

CPU replays declare `no_model_load`, exact enumeration and samplers declare
`cpu_exact_solver_or_simulator`, contract/capstone tasks declare `aggregation`,
and pre-invocation blocks declare `blocked_no_run`. No bounded canary is
misdeclared as full generation.

## Hardware Requirements and Honest Limits

| Tasks | Required resources | Expected use | Stop condition |
|---|---|---|---|
| 7167 | One idle RTX 3090, cached 16 GB Qwen3.8 Q4_K_M GGUF, CUDA llama.cpp, writable raw storage | One task-owned full-generation process with teardown | Any unowned/conflicting process, cache mismatch, CPU-only runner, or missing lease -> terminal blocked artifact |
| 7168-7173 | CPU, 8 GB RAM, local artifacts | Deterministic replay, online learning, audits | Missing or hash-mismatched authority input -> blocked; never synthesize replacement rows |
| 7174 | One idle RTX 3090, same Qwen3.8 cache/runner, live ARC entrypoint, OS/path isolation | Matched full-generation live-policy arms | Knowledge path not provably withheld or model/GPU unavailable -> blocked/disqualified; no solve credit |
| 7175 | CPU, 8 GB RAM | Exact enumeration plus seeded MCMC | Exact law unavailable for requested small case -> shrink preregistered case before running, not after seeing results |
| 7176 | Rust toolchain, CPU, PyO3/maturin surface | Parity and throughput | Parity failure forbids speed headline |
| 7177 | CPU and checked-in backend/interface descriptions | Static and measured placement accounting only | No authenticated device receipt -> no device claim |
| 7166, 7178 | CPU, repository documents and artifacts | Contract validation and aggregation | Missing inputs are named as blocked rows; capstone still writes |

Board continuity is handled at the current evidence boundary: KV260 graduated
at exp2742, PolarFire graduated at exp3867, and the GateMate lineage has no new
operator-authored physical receipt after exp6559. The current known-issues
ledger explicitly says that another unchanged GateMate experiment would only
repeat the same block. V632 therefore records GateMate as an operator decision,
not an experiment, and makes no claim that a board ran.

Extropic Z1 is also unavailable locally. Exp7177 may use published Z1T degree,
quantization, and partition constraints as a compatibility profile, but it may
not report Z1 execution, power, latency, energy efficiency, or parity.

## Acceptance and Evidence Rules

- Every task writes a schema-complete terminal artifact at its declared path.
- Every comparative task emits a `rows` list with one row per unit and YAML
  declares `per_unit_rows: true`.
- Every artifact declares `verdict_class` from `positive | circular_positive |
  null | blocked | disqualified | partial`; only locally unfinished work is
  `partial`.
- Every blocked result uses `gate_check_summary` with the exact failed check,
  expected value, and observed value.
- Every gate field is named verbatim in its producer's required artifact
  fields and the producer appears earlier in this roadmap.
- Every required artifact field in the YAML prompt carries a `principle:`
  annotation.
- Every long operation has flushed before/after lines; long loops emit flushed
  heartbeats at least every 300 seconds and keep every output gap under 600
  seconds.
- Verifier tasks declare `verifier_is_oracle`; model self-judgment never
  authorizes correctness.
- ARC rows preserve path provenance. No game source, exhaustive offline BFS,
  hand adapter, registry solution, or prior solve artifact is credited as live
  self-discovery.
- The capstone excludes artifacts carrying current or stamped critical
  adversarial flags from positive aggregation while preserving them as
  quarantined evidence rows.

## Explicitly Deferred or Excluded

- Killing or adopting PID 233772 without task-owned lease evidence or operator
  stop authority.
- Reopening retired Spilled Energy, external-text scorer, PWA-KAN, generation-
  axis, cross-game value-transfer, or per-game ARC-solver lineages.
- Training an EBT/Kona-scale model or treating Kona as an executable baseline.
- The layerwise hidden-state experiment suggested by arXiv:2609.08981 until the
  established Qwen3.8 runtime exposes the required tensors with provenance.
- Any new public ARC game-level solve, offline ground-truth BFS, or adapter-built
  solve.
- Another unchanged GateMate physical task before operator-authored changed
  state or dated retirement direction.
- Any TSU/Z1 hardware performance claim before authenticated local access.
- Public submission, release, deployment, or push.

## Deliverable

The execution roadmap is `research-roadmap-next.yaml`. It contains exactly the
thirteen task IDs, titles, deliverables, and structured gates in the Exact Task
Contract table, in that order. Any difference is a contract failure and must be
fixed in both files before activation.
