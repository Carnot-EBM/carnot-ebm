# Carnot Research Roadmap vNEXT: Correct Transport, Exact Energy, and Causal Learning

**Created:** 2026-09-06
**Status:** Proposed
**Milestone:** `2026.09.621`
**Supersedes:** milestone `2026.09.620` after its terminal conductor run
**Task contract:** exactly 12 tasks, `exp7084` through `exp7095`, in the order below
**Informed by:** V620 artifacts, the V621 planner refresh in `research-references.md`, the PRD, and the active architecture

## What V620 Proved

V620 closed all eight tasks that were present in its execution YAML. It did not
complete the entrance-energy science branch.

| Area | V620 evidence | Result |
|---|---|---|
| Roadmap contract | Exp7076 compared the Markdown plan with the eight-row execution YAML | Disqualified. The Markdown promised 13 tasks while the YAML held eight. |
| Current sources | Exp7077 froze the V620 paper, product, and code evidence | Ready. The source sweep itself was not the blocker. |
| GPU lease safety | Exp7078 migrated old released journals and Exp7079 repeated the checks in fresh processes | Ready on both local RTX 3090 devices. |
| SOTA entrance generation | Exp7080 ran the Gemma 4 31B shard | Incomplete after three attempts. Raw completion bypassed the GGUF chat template. Of 192 rows, 90 were empty, 64 had zero tokens, and 21 leaked a control token. |
| Entrance sufficiency and energy | Exp7081 and Exp7082 depended on the incomplete bank | Pre-emptively blocked. No value claim exists. |
| QUBO and Ising parity | Exp7083 depended on the missing energy equation | Pre-emptively blocked. No V620 parity claim exists. |

The failure is narrow and actionable. The models, CUDA runner, and lease path
exist. The next run must prove instruction transport on every model family
before it spends the full proposal budget.

## Three Largest Gaps to the PRD Vision

1. **Reliable live constraint transport.** Carnot cannot claim real local
   extraction while an instruction-tuned GGUF is called through raw completion.
   A small all-family transport gate must precede the large bank.
2. **Oracle-distinct global energy.** Carnot has exact entrance labels and an
   explicit energy design, but no completed held-source comparison, cold
   abstention audit, or exact QUBO-to-Ising distribution receipt for this path.
3. **Continuous learning with causal credit.** The current memory loop either
   had too few chronological events or produced a null result. It did not
   distinguish certified causal progress from a correct guess, and the live ARC
   supervisor did not prove single-credit causal effects.

## Research Findings That Change This Milestone

- arXiv:2609.04063 warns that bounded answers can assign high advantage to a
  correct guess. V621 requires witness-bearing success, guess-poison attacks,
  and verifier-signed updates.
- arXiv:2609.04194 shows that stated step importance can differ from causal
  importance. V621 tests ARC supervisor credit by removal or replay effects.
- arXiv:2609.03241 retains, reverses, or disables self-guidance from verifier
  group advantage. V621 adapts that principle to an external constraint memory
  with frozen LLM weights.
- arXiv:2609.03900 shows that a continual-learning winner can change across
  time and capacity. V621 reports full trajectories under equal memory budgets.
- OpenReview record `PF9lhBseXQ` uses final energy as an abstention signal. V621
  tests this idea on held entrance source groups while exact labels remain the
  authority.
- The NeurIPS 2025 workshop paper "The Energy to Say No" compares energy
  abstention with softmax, k-nearest-neighbor, ODIN, and Mahalanobis controls
  under several out-of-distribution splits. V621 uses matched risk-coverage
  comparisons and distinct held shifts. It does not transfer the medical RAG
  result as evidence for Carnot.
- arXiv:2608.00220 shows that verifier-scored training can improve pass-at-one
  while reducing best-of-k support. V621 keeps model weights frozen and
  reports proposal support separately from selected accuracy.
- Extropic's Z1T report fixes the software target at degree 16. V621 checks
  placement, finite distributions, sampling schedules, and autocorrelation. It
  makes no physical Z1 claim.

## V621 Architecture

```text
              Exact entrance fixture (Exp7064)
                            |
                            v
              +---------------------------+
              | all-family chat canary    | Exp7085
              | embedded GGUF templates   |
              +-------------+-------------+
                            | transport ready
                            v
              +---------------------------+
              | three-family SOTA bank    | Exp7086
              | raw-first, exact labels   |
              +-------------+-------------+
                            | complete bank
                            v
              +---------------------------+
              | cold set sufficiency      | Exp7087
              | authenticity + headroom   |
              +-------------+-------------+
                            | two exact readiness fields
                            v
              +---------------------------+
              | bounded entrance energy   | Exp7088
              | strong matched controls   |
              +------------+--------------+
                           / \
                          v   v
             +---------------+ +----------------------+
             | cold abstain  | | QUBO -> Ising       |
             | and OOD audit | | degree-16 receipt   |
             | Exp7089       | | Exp7090             |
             +---------------+ +----------------------+

  Exact fixture -----------------------> sealed 144+ event stream (Exp7091)
                                                   |
                                                   v
                                      verifier-signed memory A/B (Exp7092)
                                                   |
                                                   v
                                      cold drift and rollback (Exp7093)

  Live ARC receipts ------------------> causal single-credit replay (Exp7094)

  All terminal artifacts -------------> ungated evidence matrix (Exp7095)

  Markdown/YAML -----------------------> advisory exact contract (Exp7084)
```

The exact verifier is final authority. The LLM proposes. Learned energy ranks
or abstains. External memory changes only after a sealed outcome. Model weights
stay frozen. The ARC task uses only the live agent path and saved runtime
evidence. It does not read game source or claim a level solve.

## Phase 0: Contract and Correct Local Transport

Phase 0 removes the failure mode that invalidated V620. Exp7084 is advisory so
a documentation defect cannot cascade-block science. Exp7085 is a small fail-
fast run over all three mandated GGUF families. Exp7086 spends the full GPU
budget only after the canary proves chat-template, stop, parse, and cleanup
behavior.

### Exp7084 - V621 Markdown and YAML task-contract preflight

**Purpose:** Independently parse the active Markdown and YAML and compare all 12
task rows, titles, deliverables, gates, producer fields, model rules, prior
failures, and prompt tails.

**Method:** Use two source parsers. Require a five-column Markdown contract
table. Treat a mismatch as `disqualified`, not `partial`. Keep this task
advisory and keep Exp7095 ungated.

**Acceptance:** `v621_task_contract_conforms_score=1` only when the exact ordered
IDs are `exp7084` through `exp7095` and every task field agrees.

**Gate:** None.

**Prior failures:**

- `exp7050-v618-active-contract-preflight` — V618 had a Markdown/YAML mismatch.
- `exp7076-v620-contract-preflight` — V620 again had a Markdown/YAML mismatch.

### Exp7085 - Three-family GGUF chat-template transport canary

**Purpose:** Prove that each required instruction-tuned model receives its
embedded chat template and can return a short parseable entrance proposal.

**Method:** Run eight sealed fixture units per model with one matched seed,
`create_chat_completion` or an equivalent embedded-template call, a 192-token
budget, explicit stop checks, raw-first writes, and GPU lease cleanup.

**Acceptance:** `chat_transport_ready_score=1` requires all three real models,
zero empty and zero leaked-control-token outputs, at least 95 percent parseable
outputs per family, correct template receipts, and clean VRAM release.

**Gate:** None. It consumes the already-ready Exp7064 and Exp7079 artifacts as
fixed inputs rather than as new roadmap gates.

**Prior failures:**

- `exp6200-three-family-raw-code-transport-canary` — raw transport covered its cells but no family became ready.
- `exp7080-recovered-three-family-entrance-bank` — raw completion caused empty, zero-token, and control-token outputs.

### Exp7086 - Chat-correct three-family SOTA entrance bank

**Purpose:** Produce the complete proposal bank that V619 and V620 did not
produce.

**Method:** Run the same frozen units, four or more proposal seeds, and forced-
prefix panel on Qwen 3.6 35B-A3B, Gemma 4 31B, and Gemma 4 26B-A4B. Use the
canary-approved chat path. Preserve raw outputs and token scores before exact
post-generation labeling.

**Acceptance:** `entrance_proposal_bank_complete_score=1` requires all model,
unit, seed, source, checkpoint, telemetry, runner, identity, and cleanup cells.
Empty output must stay below 5 percent per family and at least 90 percent of
rows must parse per family.

**Gate:**

- `exp7085-three-family-chat-transport-canary.chat_transport_ready_score == 1`

**Prior failures:**

- `exp7065-three-family-entrance-proposal-bank` — malformed legacy lease journals blocked generation.
- `exp7080-recovered-three-family-entrance-bank` — incorrect instruction transport left the bank incomplete.

## Phase 1: Exact Energy, Abstention, and Ising Portability

Phase 1 first verifies the bank as a set. It then compares a bounded pairwise
energy with structural, likelihood, frequency, uniform, shuffled, and same-
family controls. The cold audit tests abstention under held shifts. The Ising
task proves exact algebra and software distribution behavior even if learned
selection value is null.

### Exp7087 - Cold entrance-bank sufficiency and headroom audit

**Purpose:** Recompute raw outputs, exact labels, family coverage, conflicts,
and selector headroom in a fresh process.

**Method:** Check every model, seed, source group, and entrance family. Run
family deletion, source swap, byte mutation, duplication, conflict, and label-
leakage attacks. Do not fit a selector.

**Acceptance:** `entrance_support_audit_ready_score=1` requires authentic and
complete support. `entrance_selector_headroom_ready_score=1` requires at least
30 held units with both reachable and unreachable proposals and no perfect
non-oracle base arm.

**Gate:**

- `exp7086-chat-correct-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1`

**Prior failures:**

- `exp7066-entrance-bank-independent-audit` — the upstream proposal bank was blocked.
- `exp7081-entrance-bank-set-sufficiency-audit` — the upstream V620 bank retired incomplete.

### Exp7088 - Entrance energy versus strong matched controls

**Purpose:** Test whether an explicit bounded entrance energy adds held-source
selection value beyond model likelihood and structural controls.

**Method:** Fit only on calibration source groups. Compare learned energy,
MRV, target log probability, average log likelihood, summed log likelihood,
proposal frequency, uniform legal choice, a same-family scorer, and shuffled
energy. Use exact outcomes only after held selection.

**Acceptance:** `entrance_energy_comparison_complete_score=1` means every arm
finished and can accompany an honest null. `entrance_energy_value_ready_score=1`
also requires a five-point gain over the strongest non-oracle control, a paired
95 percent lower bound above zero, and no required-family regression.

**Gates:**

- `exp7087-cold-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1`
- `exp7087-cold-entrance-bank-sufficiency-audit.entrance_selector_headroom_ready_score == 1`

**Prior failures:**

- `exp1006-energy-selection-ssd` — the prior fixed-bank energy branch did not reach its gate.
- `exp7067-hopfield-entrance-energy-selection` — the V619 energy task lost its upstream bank.
- `exp7082-entrance-energy-likelihood-controls` — the V620 energy task was cascade-blocked.

### Exp7089 - Cold entrance-energy abstention and shift audit

**Purpose:** Test whether the final energy supports calibrated abstention under
held source shifts without becoming an oracle.

**Method:** In a fresh process, freeze thresholds on calibration groups. Test
normal held groups, source swaps, family deletion, conflict injection, and
counterfactual numeric shifts. Compare energy abstention with likelihood,
structural, random-budget, and no-abstention controls.

**Acceptance:** `entrance_abstention_audit_complete_score=1` requires all fixed
arms and attacks. `entrance_abstention_value_ready_score=1` requires improved
selective risk at matched coverage, no hidden exact-label access, and stable
direction across all three model families.

**Gate:**

- `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1`

**Prior failures:**

- `exp533-cold-decoding-energy-guidance` — verdict
  `no_violation_reduction`. Exp533 steered token decoding. Exp7089 instead
  freezes thresholds on a completed entrance selector, evaluates held shift
  classes, and compares selective risk at matched coverage. If this changed
  attempt returns the same verdict, retire this scope.

### Exp7090 - Entrance QUBO, Ising, and degree-16 sampling receipt

**Purpose:** Translate the frozen entrance energy to QUBO and Ising forms and
prove exact finite behavior on a degree-16 software target.

**Method:** Exhaustively check affine energy and rank parity. Compare exact
Boltzmann probabilities with CPU Gibbs and parallel tempering at three or more
temperatures. Report schedules, autocorrelation, effective sample size, and
degree-16 placement overhead.

**Acceptance:** `entrance_ising_parity_ready_score=1` requires energy, rank,
tie, and finite-distribution checks. `degree16_software_mapping_ready_score=1`
requires explicit placement overhead and reconstruction parity. The artifact
must set `hardware_execution_claim=false`.

**Gate:**

- `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1`

**Prior failures:**

- `exp7073-entrance-energy-ising-parity` — the V619 translation task lost its upstream energy.
- `exp7074-degree16-placement-sampler-audit` — the V619 placement task lost its upstream translation.
- `exp7083-entrance-ising-degree16-parity` — the V620 combined task was cascade-blocked.

## Phase 2: Prospective Continuous Self-Learning

Phase 2 runs independently of the GPU bank. Exp7091 solves the prior evidence-
volume defect with a sealed exact stream. Exp7092 compares external-memory
updates prospectively and keeps model weights frozen. Exp7093 repeats the
result from clean processes and attacks drift, poison, crash, and rollback.

### Exp7091 - Sealed causal entrance outcome stream

**Purpose:** Create enough chronological exact evidence for a real prospective
continuous-learning comparison.

**Method:** Derive at least 144 ordered events over all 12 fixture source
groups. Freeze time-zero features before opening each later exact outcome.
Label whether success has a reachable witness or only an answer-level match.
Hash-chain the stream and reserve protected retention slices.

**Acceptance:** `causal_entrance_stream_ready_score=1` requires at least 144
unique events, all 12 groups, no future-field leakage, explicit guess rows,
protected retention groups, and byte-stable replay.

**Gate:** None.

**Prior failures:**

- `exp7070-bcit-prospective-self-learning` — its frozen input had only eight events over five groups.

### Exp7092 - Verifier-signed prospective constraint-memory learning

**Purpose:** Satisfy the milestone's continuous self-learning requirement with
causal, verifier-grounded external memory.

**Method:** On the sealed stream, compare context-bound verifier-signed balanced
updates, context-bound raw advantage, flat reuse, validate-all, and no reuse.
Use equal memory and validation budgets. Retain, reverse, or disable proposed
updates from exact group evidence. Never update LLM weights.

**Acceptance:** `constraint_memory_comparison_complete_score=1` means all arms
and the full trajectory finished. `constraint_memory_value_ready_score=1`
requires a positive time-averaged gain over the strongest control, no protected-
slice regression, fewer harmful transfers, and stable ordering across declared
memory capacities.

**Gate:**

- `exp7091-sealed-causal-entrance-stream.causal_entrance_stream_ready_score == 1`

**Prior failures:**

- `exp6978-transactional-constraint-self-learning` — the earlier transactional comparison was null.
- `exp7070-bcit-prospective-self-learning` — the later comparison was blocked by an insufficient stream.

### Exp7093 - Cold drift, guess-poison, and rollback audit

**Purpose:** Verify that the learned external memory is durable, bounded,
causal, and recoverable outside the producer process.

**Method:** Recompute every trajectory from immutable inputs. Attack context
drift, answer-only guess poisoning, reordered outcomes, partial writes, stale
parents, capacity pressure, crash recovery, and rollback. Compare several
evaluation times and memory capacities.

**Acceptance:** `constraint_memory_cold_audit_ready_score=1` requires byte-
stable replay, atomic transactions, zero protected regression, rejected guess
poison, successful rollback, and agreement between rows and headlines.

**Gate:**

- `exp7092-verifier-signed-constraint-memory-learning.constraint_memory_comparison_complete_score == 1`

**Prior failures:**

- `exp6979-self-learning-cold-audit` — the earlier cold audit was null.
- `exp7071-bcit-drift-rollback-audit` — the V619 audit lost its upstream comparison.

## Phase 3: Live-Path Causal Credit and Integration

Phase 3 improves one live ARC mechanism without claiming a solve. The capstone
then reports the complete milestone, including null, blocked, and disqualified
branches.

### Exp7094 - Live ARC causal single-credit supervisor replay

**Purpose:** Replace textual or duplicate supervisor credit with causal credit
from saved live-agent transitions.

**Method:** Use only live E3 agent receipts and runtime replay. For each eligible
intervention, compare the saved action path with one deterministic removal or
redirect replay. Deduplicate identities and award at most one credit event per
causal transition. Never read game source, run offline ground-truth BFS, or add
a per-game adapter.

**Acceptance:** `arc_causal_single_credit_ready_score=1` requires at least 12
eligible replay pairs, causal-effect and assigned-credit agreement, zero double
credit, identity safety, and no outer-loop evidence. Set
`game_level_solve_claim=false`.

**Gate:** None. The task must report a terminal null or blocked result if live
runtime evidence cannot support the comparison.

**Prior failures:**

- `exp6524-arc-supervisor-redirect-generalization` — outcome-bearing live supervisor receipts were absent.
- `exp6921-arc-dynamic-supervisor-banked-credit` — banked progress existed but missed the frozen evidence floor.

### Exp7095 - V621 independent evidence matrix and branch disposition

**Purpose:** Recompute milestone status from primary artifacts and state what
may proceed, what is null, and what must retire.

**Method:** Hash and validate every available artifact. Recompute task gates,
row aggregates, model compliance, substrate classes, prior-failure retirement,
hardware claim boundaries, continuous-learning evidence, and ARC provenance.
Do not convert missing or blocked evidence into success.

**Acceptance:** `v621_capstone_complete_score=1` means all 12 task slots have a
terminal disposition and all available primary rows were independently checked.
It does not mean every science gate passed.

**Gate:** None. This task must run even when every other branch is blocked.

**Prior failures:** None. V619 showed that an ungated capstone can preserve an
honest milestone disposition.

## Exact Task Contract

The table is the Markdown side of the execution contract. The YAML side is
`research-roadmap-next.yaml`. Titles, deliverables, order, and structured gates
must match byte-for-byte after YAML scalar parsing.

| Order | Task ID | Title | Deliverable | Structured gates |
|---:|---|---|---|---|
| 1 | `exp7084-v621-contract-preflight` | V621 Markdown and YAML task-contract preflight | `results/experiment_7084_v621_contract_preflight.json` | None |
| 2 | `exp7085-three-family-chat-transport-canary` | Three-family GGUF chat-template transport canary | `results/experiment_7085_v621_chat_transport_canary.json` | None |
| 3 | `exp7086-chat-correct-three-family-entrance-bank` | Chat-correct three-family SOTA entrance bank | `results/experiment_7086_v621_three_family_entrance_bank.json` | `exp7085-three-family-chat-transport-canary.chat_transport_ready_score == 1` |
| 4 | `exp7087-cold-entrance-bank-sufficiency-audit` | Cold entrance-bank sufficiency and headroom audit | `results/experiment_7087_v621_entrance_bank_sufficiency_audit.json` | `exp7086-chat-correct-three-family-entrance-bank.entrance_proposal_bank_complete_score == 1` |
| 5 | `exp7088-entrance-energy-strong-controls` | Entrance energy versus strong matched controls | `results/experiment_7088_v621_entrance_energy_controls.json` | `exp7087-cold-entrance-bank-sufficiency-audit.entrance_support_audit_ready_score == 1 AND exp7087-cold-entrance-bank-sufficiency-audit.entrance_selector_headroom_ready_score == 1` |
| 6 | `exp7089-cold-entrance-energy-abstention-audit` | Cold entrance-energy abstention and shift audit | `results/experiment_7089_v621_entrance_energy_abstention_audit.json` | `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1` |
| 7 | `exp7090-entrance-ising-degree16-sampling-receipt` | Entrance QUBO, Ising, and degree-16 sampling receipt | `results/experiment_7090_v621_entrance_ising_degree16_receipt.json` | `exp7088-entrance-energy-strong-controls.entrance_energy_comparison_complete_score == 1` |
| 8 | `exp7091-sealed-causal-entrance-stream` | Sealed causal entrance outcome stream | `results/experiment_7091_v621_causal_entrance_stream.json` | None |
| 9 | `exp7092-verifier-signed-constraint-memory-learning` | Verifier-signed prospective constraint-memory learning | `results/experiment_7092_v621_constraint_memory_learning.json` | `exp7091-sealed-causal-entrance-stream.causal_entrance_stream_ready_score == 1` |
| 10 | `exp7093-constraint-memory-cold-audit` | Cold drift, guess-poison, and rollback audit | `results/experiment_7093_v621_constraint_memory_cold_audit.json` | `exp7092-verifier-signed-constraint-memory-learning.constraint_memory_comparison_complete_score == 1` |
| 11 | `exp7094-live-arc-causal-single-credit-replay` | Live ARC causal single-credit supervisor replay | `results/experiment_7094_v621_arc_causal_single_credit.json` | None |
| 12 | `exp7095-v621-capstone` | V621 independent evidence matrix and branch disposition | `results/experiment_7095_v621_capstone.json` | None |

## Dependency Graph

```text
Exp7084  advisory contract check

Exp7085 -> Exp7086 -> Exp7087 -> Exp7088 -> Exp7089
                                      `-------> Exp7090

Exp7091 -> Exp7092 -> Exp7093

Exp7094  independent live-path causal credit

Exp7095  ungated capstone over every terminal slot
```

There are six structured task dependencies and seven gate clauses. Every gate
producer is earlier in this roadmap. Each producer field appears with the same
spelling in that producer's required artifact fields in the YAML prompt.

## Hardware Requirements

| Resource | Tasks | Requirement |
|---|---|---|
| Two local RTX 3090 GPUs | Exp7085 and Exp7086 | Required. Use owned leases, CUDA llama.cpp, per-stage telemetry, raw-first checkpoints, and clean release. |
| Cached SOTA GGUFs | Exp7085 and Exp7086 | Required: Qwen 3.6 35B-A3B, Gemma 4 31B, and Gemma 4 26B-A4B. No legacy-small headline fallback. |
| Host CPU and RAM | Exp7084 and Exp7087-Exp7095 | Required for exact enumeration, training, audits, replay, and aggregation. |
| Z1 or other TSU | Exp7090 | Not attached and not required. The task emits a software receipt only. |
| KV260 and PolarFire | None | Existing terminal receipts are sufficient. Do not reopen them. |
| GateMate | None | Physical access is still blocked. It is outside the milestone. |

If an unattributed process owns a GPU, the GPU task blocks and sends no signal.
No hardware task may claim physical power, latency, throughput, or execution
without a physical device receipt.

## Shared Evidence Rules

Every task must emit `field_principles`, `preconditions_checked`, `duration_s`,
`source_artifact_hashes`, per-unit rows, `random_seed`,
`reproducibility_checksum`, `gate_check_summary`, `verifier_is_oracle`, the free-
text `honest_verdict`, and the closed `verdict_class`. Every task also emits a
prose `inference_substrate` and a closed `inference_substrate_class`.

The allowed class values are `aggregation`, `no_model_load`,
`model_load_no_generation`, `model_bounded_generation`,
`model_full_generation`, `hardware_board`, and `blocked_no_run`. A task that
cannot run because of an unchanged external condition uses `blocked`, not
`partial`. A comparative task emits one row for every unit, arm, seed, game, or
condition used in its headline.

## Milestone Completion Rule

V621 is complete when all 12 YAML tasks have terminal dispositions and Exp7095
has independently described every slot. Science success remains branch-local:

- a complete bank does not prove energy value;
- a complete comparison may honestly be null;
- a software Ising receipt is not hardware execution;
- an external-memory gain is not model-weight learning;
- causal ARC supervisor progress is not a game-level solve.

The roadmap must not be shortened without changing this document and
`research-roadmap-next.yaml` together.
