# Research Roadmap vNEXT — Causal Factors and Safe Persistent Learning

**Milestone:** 2026.08.554  
**Planning date:** 2026-08-14  
**Status:** Proposed  
**Predecessor:** 2026.08.553  
**Execution queue:** `research-roadmap-next.yaml`  
**Task count:** 12 experiments in four phases

## 1. Decision

Milestone 2026.08.554 will move from factor correlation to factor causality. It
will test whether exact-admitted factors change future exact outcomes under
active, ablated, tombstoned, and placebo controls. It will then use only the
surviving factor policy in a fresh continuous self-learning stream.

The milestone also repairs two V553 evidence failures. It reruns the
verification-cost measurement with real monotonic work and enough independent
rows. It recovers the zero-byte ARC reachability artifact with bounded CPU
shards and checkpoints.

Exact validators remain the only release authority. Models, factors, energy
signals, memories, and routers may propose, rank, abstain, or direct work. They
may not override an exact rejection.

No board experiment is scheduled. The local FPGA and TSU state did not change.
V554 uses the two RTX 3090 GPUs for authenticated GGUF generation and uses CPU
execution for exact replay, cost measurement, audits, and ARC reachability.

## 2. What V553 proved

V553 established a narrow factor result and repaired the recurring gate
contract. It did not establish the broader cost, self-learning, or ARC claims.

| Evidence | What V553 proved | Boundary carried into V554 |
|---|---|---|
| Exp6424–6426 | The queue preflight, recurring-gate diagnostics, and task-scoped runtime receipt are usable. A local GGUF task can bind process, model, raw bytes, timing, and GPU evidence. | V554 extends this receipt across generation, factor compilation, checker transport, and final verdict. |
| Exp6427 | A clean 144-row factor corpus covers three model families and three factor families. Of 144 rows, 64 are exactly evaluable; joint exact success is 15/64. | The corpus proves availability and exact binding. It does not prove that a stored factor caused a future outcome. |
| Exp6428 | Exact write-time admission improves future exact yield by 0.08333 with zero contamination and no retention harm. | This is the narrow eligible factor result. A write-everything arm has the same yield but 0.3542 contamination, so exact admission remains required. |
| Exp6429 | Selective verification matched always-refine accuracy and used fewer checker calls in the reported rows. | The artifact is claim-ineligible. `DURATION_TOO_SHORT` and underpowered cost cells remain open. |
| Exp6430 | A development write-once memory frontier reports future exact yield rising from 0 at capacity 0 to 0.75 at capacities 16 and 32. | This is development evidence. It does not repair the held-stream authenticity failure. |
| Exp6431 | Authority-aware memory reports 0.60 future yield against 0.12 for the baseline. | All 400 cells are underpowered. The result is not enough for a safety promotion. |
| Exp6432 | Held restart rows report a large positive effect. | `DURATION_TOO_SHORT` makes the result ineligible. Fresh task-scoped generation and powered timing are required. |
| Exp6433 | Independent row recomputation found no aggregate mismatch. | Prospective CSL remains `complete_null` because the held prerequisite is flagged. |
| Exp6434 | No scientific evidence was produced. | The deliverable is zero bytes after a hard wall-clock failure. Reachability remains unknown. |
| Exp6435 | The narrow factor branch is eligible. Verification cost, prospective CSL, ARC reachability, public ARC, and hardware claims remain blocked. | V554 must preserve these separate determinations. |

## 3. The three largest PRD gaps

### Gap 1 — Stored factors have utility but no causal influence proof

FR-12 requires deterministic constraint verification. V553 shows that exact
admission prevents contamination and can improve future yield. It does not
show that the factor itself changed behavior. A factor may be dead prompt text,
or an execution wrapper may introduce the observed difference. Verification
cost also lacks an eligible powered measurement.

V554 response:

- bind immutable bytes at every generation-to-verdict boundary;
- rerun verification cost with real monotonic work and at least five distinct
  source rows per cost cell;
- compare active, ablated, tombstoned, and length-matched placebo factors; and
- replicate causal influence under binding shifts and constraint revocation.

### Gap 2 — FR-11 continuous self-learning lacks fresh held safety evidence

The PRD requires autonomous improvement with immutable validation, rollback,
and bounded forgetting. V553 has a development memory frontier. Its held run
is duration-flagged, and its interference cells are underpowered. It also does
not test whether a successful but unsafe exposure becomes persistent policy.

V554 response:

- compare full-event replay with a target-bound factor contract on a fresh
  chronological stream;
- measure authoring, retrieval, execution, quarantine, rollback, and
  fresh-session harm after malicious-but-successful exposures;
- replicate the frozen policy on new held generations after process restarts;
  and
- recompute every CSL headline from immutable per-unit rows.

This phase is the milestone's required continuous self-learning work.

### Gap 3 — Live ARC policy cannot prove generic reachability

The live ARC route can change legal policy, but the latest state-key
reachability experiment produced no artifact. The PRD vision needs a general
agent that can explore hidden tasks through the canonical live path. It does
not need another offline solver or per-game adapter.

V554 response:

- retain the canonical adapter-bypassed live path;
- detect observation-history collisions generically;
- compare the baseline key with the smallest collision-certified suffix;
- run bounded per-roster CPU shards with atomic checkpoints; and
- make no game, level, registry, or public-score solve claim.

Hardware acceleration remains a fourth gap. It is not one of this milestone's
three active questions because no authenticated device state changed.

## 4. Research refresh and experiment hooks

The dated source ledger is in `research-references.md` under the V554 planner
marker.

| Source | Carnot use in V554 | Boundary |
|---|---|---|
| Dead text or binding clause?, arXiv:2608.12599 | Compile active factor ledgers, preserve tombstones, and use sequential clause ablation to measure incremental exact effect. | Exact checkers remain authoritative. Prompt influence is not correctness. |
| QuoteBench, arXiv:2608.13547 | Replay immutable bytes across raw generation, factor compilation, checker transport, and final exact validation. | Attribute model and wrapper failures separately. |
| Practice Makes Unsafe, arXiv:2608.12851 | Add malicious-success exposure and lifecycle metrics for authoring, retrieval, execution, quarantine, rollback, and later harm. | A local task success cannot override a protected exact invariant. |
| Beyond Retrieval, arXiv:2608.12847 | Compare full trajectories with query-conditioned factor contracts that carry procedure, bindings, applicability, and verification requirements. | Hold retrieval candidates, models, budgets, and target rows fixed. |
| Agent Behavioral Contracts II, arXiv:2608.12895 | Measure joint model-factor-checker co-failure and an assumption-free finite-sample certificate. | Do not multiply marginal reliabilities or treat the certificate as release authority. |
| Constraint activation bottlenecks, arXiv:2608.12321 | Distinguish factor availability from factor use with black-box active-versus-ablated exact outcomes. | Do not reopen the retired hidden-state scorer lane. |

OpenReview and Hugging Face results do not provide a stronger executable local
authority than Carnot's exact checkers. Semantic Scholar was rate-limited
during this refresh; the V553 EBT and ARM-EBM counts remain the last
authenticated snapshot. Current GitHub results add no required dependency.
Extropic still provides no authenticated Carnot-local TSU route. Kona still
provides no public weights or reproducible local API. Recent KAN, Ising, and
FPGA work does not close the selected evidence gaps with available hardware.

## 5. Target architecture

```text
 mandated local GGUFs
 Qwen MoE | Gemma dense | Gemma MoE
                |
                v
       immutable raw generation
                |
                v
      exact-admitted factor ledger
  active | ablated | tombstoned | placebo
                |
                v
       compiled target-bound support
                |
                v
       exact checker transport input
                |
                v
         exact checker output
                |
                v
          final release veto
                |
       generation-to-verdict receipt
   hashes | PIDs | model | timing | replay
                |
       +--------+---------+
       |                  |
       v                  v
 causal factor A/B     verification cost A/B
       |                  |
       +--------+---------+
                |
                v
   query-conditioned persistent memory
 full event | target-bound factor | frozen
                |
       +--------+---------+
       |                  |
       v                  v
 malicious success    held process restart
 lifecycle controls   fresh model generations
       |                  |
       +--------+---------+
                |
                v
      independent row recomputation
                |
                v
      exact release veto remains final

 canonical live ARC observation
                |
                v
 generic collision certificate
                |
                v
 bounded state-key suffix A/B
 CPU shards | checkpoints | no solve credit
```

Five rules hold across the diagram:

1. Every comparative claim has immutable `per_unit_rows`.
2. Every aggregate is a deterministic reduction of those rows.
3. Raw bytes are hashed at each command-path boundary.
4. Joint reliability comes from joint outcomes, not multiplied marginals.
5. A model, score, factor, memory, or router cannot override exact rejection.

## 6. Phase design

### Phase 0 — Handoff and path attribution

#### Exp6436 — V553 terminal handoff and V554 queue preflight

Question: Is the V554 queue anchored to the real V553 terminal evidence and
free of schema, gate, exclusion, model, and prior-failure defects?

The task records the clean factor boundary, flagged cost and held results,
underpowered interference result, zero-byte ARC artifact, and capstone
determinations. It validates every V554 task and its exact gate field before
research starts.

Deliverable:
`results/experiment_6436_v554_terminal_handoff_and_queue_preflight.json`

#### Exp6437 — Generation-to-verdict receipt and replay contract

Question: Can Carnot locate a failure at the raw-generation, factor-
compilation, checker-transport, checker-output, or final-verdict boundary?

The task extends the V553 runtime receipt without changing the conductor. It
uses identity, injected-wrapper, and restored-wrapper replays over immutable
bytes. Exact final-state validation must expose which boundary changed.

Deliverable:
`results/experiment_6437_generation_to_verdict_receipt_replay_contract.json`

### Phase 1 — Cost and causal factor influence

#### Exp6438 — Powered verification-cost repair

Question: Can selective exact verification preserve always-refine accuracy
while reducing checker work in an eligible, powered measurement?

This is a changed rerun of Exp6429. It uses real monotonic checker work, at
least five distinct source rows in every reported cost cell, and the Exp6437
path receipt. It compares never-refine, always-refine, and exact-triggered
selective arms. `duration_s` must measure the real task and pass current
adversarial checks.

Promotion requires accuracy parity with always-refine, no false-accept
increase, fewer checker calls or lower measured exact-verification time, full
row recomputation, and zero current critical flag.

Deliverable:
`results/experiment_6438_powered_verification_cost_repair_ab.json`

#### Exp6439 — Development clause-influence A/B

Question: Does an exact-admitted factor change a future exact outcome, or is it
dead support text?

Fresh matched generations from all three mandated model families compare an
active factor, sequential clause ablation, a tombstoned factor, no factor, and
a length-matched placebo. Every clause has an executable checker. The same
task, model, decoding budget, and final checker are held fixed.

Promotion requires a positive row-derived active-over-ablated exact effect,
no false-accept increase, a valid placebo control, full command-path receipts,
and zero current critical flag.

Deliverable:
`results/experiment_6439_factor_clause_influence_ab.json`

#### Exp6440 — Held revocation and binding-shift replication

Question: Does the active factor effect survive new bindings, and do
tombstones stop stale or revoked factors from causing behavioral relapse?

The task seals held units before generation. It compares compiled active
ledgers, raw factor text, tombstoned ledgers, and no-memory controls across all
three mandated model families. Each reported cell needs at least 12 distinct
held units. This powers the authority-interference question that Exp6431 left
underpowered.

Deliverable:
`results/experiment_6440_held_factor_revocation_binding_shift_ab.json`

### Phase 2 — Continuous self-learning and lifecycle safety

#### Exp6441 — Prospective query-conditioned factor reuse

Question: Does exact-governed, target-bound factor memory improve later exact
outcomes over both frozen memory and full-event replay?

This is the main continuous self-learning experiment. It creates a new
chronological stream with the three mandated model families. It compares
frozen memory, full-event replay, and query-conditioned factor contracts with
procedure, recovered bindings, applicability, and verification requirements.
Every event has fresh raw bytes. The future partition is sealed before any
memory write.

Promotion requires positive future exact yield over frozen memory, no increase
in contamination or protected-case failure, bounded memory growth, full row
recomputation, and zero critical path-attribution failure.

Deliverable:
`results/experiment_6441_prospective_query_conditioned_factor_reuse.json`

#### Exp6442 — Malicious-success skill-misevolution safety A/B

Question: Can a locally successful but globally unsafe exposure persist into
later factor state, and do exact quarantine and rollback stop it?

The task compares frozen memory, ungoverned factor evolution, and exact-
governed quarantine plus rollback. It measures authoring, retrieval,
execution, quarantine, rollback, benign utility, and fresh-session harm. One
mandatory SOTA GGUF operating point is sufficient for this safety study.

Any unsafe factor that reaches release authority, survives quarantine, or
reappears after rollback blocks promotion.

Deliverable:
`results/experiment_6442_skill_misevolution_quarantine_rollback_ab.json`

#### Exp6443 — Fresh held-shift process-restart CSL replication

Question: Does the frozen exact-governed policy retain positive value on new
held generations after real process restarts?

This is a changed rerun of Exp6432. It uses all three mandated model families,
new prompts, new raw outputs, task-scoped path receipts, sealed held manifests,
and process restarts between sessions. Development bytes may not appear in any
held row.

Deliverable:
`results/experiment_6443_fresh_held_restart_csl_replication.json`

#### Exp6444 — Independent CSL lifecycle recomputation audit

Question: Do Exp6441 through Exp6443 support a prospective self-learning claim
when an independent implementation recomputes every metric and replays every
safety attack?

The audit is ungated. Missing, skipped, blocked, null, flagged, or underpowered
upstream evidence must remain visible. It cannot import upstream aggregate
functions. A repeated `complete_null` CSL determination retires this exact
claim scope.

Deliverable:
`results/experiment_6444_csl_lifecycle_recomputation_audit.json`

### Phase 3 — ARC recovery and adversarial close

#### Exp6445 — Sharded ARC state-key reachability recovery

Question: Does a generic collision-certified state suffix prevent premature
frontier collapse without causing full-roster regressions?

This is a changed recovery of Exp6434. It runs bounded per-roster CPU shards,
writes atomic checkpoints, and resumes without repeating completed cells. It
uses only the canonical adapter-bypassed live path. The outcome is reachability,
collision reduction, legal-action coverage, and action cost.

The task makes no game or level solve claim. It may not read game source, use
offline ground-truth BFS, add a per-game adapter, or update the solve registry.

Deliverable:
`results/experiment_6445_arc_state_key_reachability_sharded_ab.json`

#### Exp6446 — Joint pathway dependence audit

Question: Do shared model, factor, and checker components co-fail strongly
enough that marginal reliability overstates pipeline reliability?

The audit reads immutable joint rows from Exp6438 through Exp6445. It reports
co-failure moments for same-model and heterogeneous cells and computes an
assumption-free finite-sample interval where the data support it. It does not
fit or multiply an independence model and does not change release authority.

Deliverable:
`results/experiment_6446_joint_pathway_dependence_audit.json`

#### Exp6447 — V554 adversarial capstone and reconciliation

Question: Which V554 claims remain eligible after row recomputation,
dependency review, current adversarial checks, and determination preservation?

The capstone audits all 12 tasks. It preserves separate eligibility for
verification cost, causal factor influence, prospective continuous self-
learning, narrow ARC reachability, public ARC, and hardware. It cannot average
away missing, flagged, null, or underpowered evidence.

Deliverable:
`results/experiment_6447_v554_adversarial_capstone.json`

## 7. Dependency graph

```text
Exp6436  terminal handoff and queue preflight
   |
   v
Exp6437  generation-to-verdict receipt
   | \
   |  +-----------------------> Exp6438 verification-cost repair
   v
Exp6439  development clause influence
   |
   v
Exp6440  held revocation and binding shift
   |
   v
Exp6441  prospective query-conditioned factor reuse
   | \
   |  +-----------------------> Exp6442 malicious-success safety
   |                                  |
   +----------------------------------+
   |                                  |
   v                                  v
Exp6443  held restart replication <---+
   |
   v
Exp6444  independent CSL audit

Exp6445  sharded ARC reachability recovery

Exp6438 --+
Exp6439 --+
Exp6440 --+
Exp6441 --+--> Exp6446 joint dependence audit
Exp6442 --+
Exp6443 --+
Exp6445 --+

Exp6436 through Exp6446 ----------------> Exp6447 capstone
```

Structured gates exist only where a failed prerequisite makes downstream work
scientifically meaningless. Exp6444, Exp6446, and Exp6447 stay ungated so they
can report missing or blocked evidence.

## 8. Models and inference policy

Experiments that execute an LLM are Exp6439 through Exp6443. Each has an
explicit `MODEL_SPECS` contract and uses `cached_sota_pair()` with embedded
GGUF tokenizers. None may call `AutoTokenizer`.

Exp6439, Exp6440, Exp6441, and Exp6443 use all three headline families:

- `unsloth/Qwen3.6-35B-A3B-GGUF`;
- `unsloth/gemma-4-31B-it-GGUF`; and
- `unsloth/gemma-4-26B-A4B-it-GGUF`.

Exp6442 may use only `unsloth/gemma-4-26B-A4B-it-GGUF` as its declared
headline operating point to control cost. It still needs authenticated local
CUDA execution and complete receipts.

`Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear only in CPU smoke tests. They
cannot support a headline cell.

All V554 tasks use the repository's Codex-default route:
`agent_type: codex` and `model: gpt-5.5`.

## 9. Hardware requirements

| Resource | Tasks | Requirement and boundary |
|---|---|---|
| Dual RTX 3090, 48 GB total VRAM | Exp6439–Exp6443 | Run mandatory GGUF generations with CUDA offload. Bind PIDs, GPU samples, runner, model bytes, raw output, and phase timing to each task. CPU fallback blocks headline eligibility. |
| CPU and system RAM | Exp6436–Exp6438, Exp6444–Exp6447 | Run exact replay, checker-cost controls, row recomputation, dependence certificates, ARC search, and audits. Record the substrate honestly. |
| Persistent local disk | Exp6437–Exp6446 | Store immutable raw bytes, path-stage hashes, per-unit JSONL, sealed manifests, memory heads, restart receipts, ARC checkpoints, and source hashes. |
| ARC live environment | Exp6445 | Use the canonical adapter-bypassed public roster and live interface. No source reads, exhaustive ground-truth search, or per-game adapter. |
| KV260, GateMate, PolarFire | None | No state change justifies a probe. Existing proof-of-concept limits remain. |
| Extropic XTR-0 or Z1 | None | No authenticated local route exists. No execution, power, latency, speed, or availability claim. |

The milestone does not claim hardware acceleration.

## 10. Measurement contract

Every comparative artifact must include `per_unit_rows`. Each row must carry a
stable unit ID, arm, partition, model or substrate, source hashes, relevant
path-stage hashes, exact outcome, work, timing, and exclusion reason. Aggregate
fields must name their row filter and deterministic reduction.

Every task must include:

- `blocked_reason` and `gate_check_summary`;
- `preconditions_checked` and `inference_substrate`;
- `field_principles` for every required field and acceptance gate;
- `field_provenance` classified as measured, derived, constant, or upstream;
- `random_seed`, `duration_s`, `tests_run`, and `reproducibility_checksum`;
- current adversarial findings for each scientific claim; and
- one terminal `honest_verdict`.

If `honest_verdict` starts with `blocked_`, `gate_check_summary` must name the
failed check, field, expected condition, observed value, and evidence path.

The claim ladder is:

```text
authenticated execution
  -> immutable path-stage bytes
  -> exact per-unit outcomes
  -> causal active-versus-control effect
  -> positive fresh held effect
  -> lifecycle attacks fail closed
  -> independent recomputation
  -> narrow claim eligibility
```

Failure at one rung blocks every higher rung. It does not erase valid lower-
rung engineering evidence.

## 11. Success and retirement rules

V554 succeeds scientifically when it closes questions honestly. A clean null
is acceptable. A fabricated, aggregate-only, or path-ambiguous positive is not.

Causal factor eligibility requires:

- clean Exp6437 path attribution;
- positive exact active-over-ablated effect in Exp6439;
- held binding-shift effect and tombstone safety in Exp6440;
- no false-accept increase; and
- eligible Exp6446 and Exp6447 determinations.

Prospective CSL eligibility requires:

- fresh write-once outputs in Exp6441 and Exp6443;
- positive row-derived development and held future exact value;
- bounded memory growth and protected retention;
- no surviving malicious-success, quarantine, or rollback attack;
- eligible Exp6444, Exp6446, and Exp6447 determinations.

ARC eligibility is limited to generic reachability. Exp6445 cannot create solve
credit or support a hidden-game or public-score claim.

Mechanical retirement applies to these changed reruns:

| New task | Prior task | Retire if the same verdict returns |
|---|---|---|
| Exp6438 | Exp6429 | The repaired cost study repeats the same duration-flagged or underpowered evidence outcome. |
| Exp6440 | Exp6431 | The held authority study again reports a positive headline with underpowered cells. |
| Exp6441 | Exp6433 | The changed query-conditioned CSL chain repeats the same ineligible terminal determination. |
| Exp6443 | Exp6432 | The held restart study repeats the same duration-flagged positive outcome. |
| Exp6444 | Exp6433 | The independent audit repeats the same `complete_null` CSL determination. |
| Exp6445 | Exp6434 | The recovery again produces no usable artifact or equivalent terminal result. |

Every entry in `research-roadmap-next.yaml` includes
`retire_if_same_verdict: true`. No retired experiment ID is reused. No task
depends on a retired upstream ID.

## 12. Reconciliation obligations

Before V554 is complete:

1. add or update relevant `REQ-*` and `SCENARIO-*` entries before code;
2. write focused tests before implementation changes;
3. run applicable E2E checks from `ops/e2e-test-plan.md`;
4. run roadmap schema, prior-failure, gate, exclusion, spec-coverage,
   adversarial, row-consistency, determination-preservation, and root-clutter
   checks;
5. preserve unrelated dirty audit and ARC files present at planning time;
6. reconcile `openspec/`, `_bmad/traceability.md`, `ops/status.md`,
   `ops/changelog.md`, `ops/known-issues.md`, and claim records; and
7. write one honest next research question from the surviving evidence.

The milestone must not modify `research-roadmap.yaml` during planning. It must
not modify `scripts/research_conductor.py`. It must not push.
