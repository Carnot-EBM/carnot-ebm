# Carnot Research Roadmap vNEXT: Causal Verification and Reversible Trace Learning

**Created:** 2026-09-03  
**Milestone:** 2026.09.608  
**Status:** Planned after activation refusal repair  
**Supersedes:** milestone 2026.09.607, experiments 6927-6933  
**Task contract:** exactly 12 tasks, Exp6941 through Exp6952, in the order below  
**Execution manifest:** research-roadmap-next.yaml

## What Milestone 2026.09.607 Proved

| Result | Experiment | Finding |
|---|---:|---|
| Execution-time literature delta | 6927 | The source audit completed. It added one exact metadata correction for the KAN verification paper. |
| SOTA runtime receipt | 6928 | The artifact claimed receipt readiness. Adversarial review quarantined it because 57 seconds was too short for its three-model claim. It cannot support a runtime headline. |
| Live span acquisition | 6929 | The task reached one hard wall-time cap and two agent-network failures. It produced no terminal artifact. |
| Relation qualification | 6930 | The conductor blocked this task after Exp6929 retired. No qualification result exists. |
| Exact strategy fixture | 6931 | Three agent-network failures retired the task before a terminal artifact existed. |
| Episodic self-learning and cold audit | 6932-6933 | Both tasks blocked on the retired fixture chain. V607 produced no continuous self-learning result. |

The V607 design document promised 14 tasks. The active YAML contained seven.
V608 does not preserve that mismatch. This document and the execution manifest
contain the same 12 tasks, titles, deliverables, order, and structured gates.

## Three Largest Gaps to the PRD

### Gap 1: Exact verification coverage is not tied to useful decisions

Carnot has exact outcome checks and several learned signals. It has not measured
when step labels cover enough of a causal trace for first-error credit to help.
It also lacks a non-saturated SOTA candidate bank for this test. This gap blocks
FR-12 and weakens the verification part of FR-11.

### Gap 2: Internal signals have not shown causal selection value

Earlier hidden-state work often blocked on missing inputs. Other work measured
decodability without proving a better decision. Carnot needs replayable local
GGUF state surfaces and matched random-direction controls. A positive claim must
improve exact held-out top-1 selection, not only AUROC.

### Gap 3: Continuous learning and ARC energy lack safe causal credit

Recent memory tasks found harmful writes, weak abstention, or no prospective
utility. The live ARC agent also lacks a branch-discriminative energy result from
its own attempts. V608 tests both gaps with exact receipts. The memory task keeps
weights frozen. The ARC task stays shadow-only and makes no solve claim.

## Research Basis Added for V608

- **When Decodability Is Not Enough**, arXiv:2609.02438, separates hidden-state
  decodability from causal behavior. It motivates random-direction and
  score-only controls.
- **Discriminative World Models**, arXiv:2609.02885, motivates learning from
  authentic alternative-action successor states.
- **Trace-as-State**, arXiv:2609.02702, motivates placing an admitted prior trace
  before fresh context instead of appending it after the context.
- **Learning from Feedback**, arXiv:2609.02859, supports exact external outcomes
  as memory-write authority. A model judge cannot authorize its own update.
- **Coverage, Not Targeting**, arXiv:2609.02417, requires a verifier-density
  measurement before targeted credit.
- **Cliff**, arXiv:2609.02817, supplies the first-error credit shape. V608 uses
  this shape only with exact first-invalid-step labels.
- **Budgeted Verification**, arXiv:2609.02783, reinforces fixed proposal budgets
  and explicit stopping costs.

The dated source checks, code-state limits, and hardware boundaries are in
research-references.md under the V608 planner refresh.

## V608 Architecture

~~~text
                     advisory evidence
               ┌──────────────────────┐
               │ source delta Exp6941 │
               └──────────────────────┘

               ┌──────────────────────┐
               │ contract Exp6942     │
               └───────┬────────┬─────┘
                       │        │
                       │        ▼
                       │  ┌────────────────────┐
                       │  │ ARC branches 6948  │
                       │  └─────────┬──────────┘
                       │            ▼
                       │  ┌────────────────────┐
                       │  │ branch energy 6949 │
                       │  └────────────────────┘
                       ▼
               ┌──────────────────────┐
               │ exact prefix corpus  │
               │ Exp6943              │
               └───────┬────────┬─────┘
                       │        │
                       │        ▼
                       │  ┌────────────────────┐
                       │  │ trace memory 6950  │
                       │  └─────────┬──────────┘
                       │            ▼
                       │  ┌────────────────────┐
                       │  │ cold audit 6951    │
                       │  └────────────────────┘
                       ▼
               ┌──────────────────────┐
               │ SOTA prefix bank     │
               │ Exp6944              │
               └────────┬─────────┬───┘
                        │         │
                        ▼         ▼
             ┌───────────────┐  ┌────────────────────┐
             │ prefix energy │  │ GGUF state 6946    │
             │ Exp6945       │  └─────────┬──────────┘
             └───────────────┘            ▼
                                  ┌────────────────────┐
                                  │ causal select 6947 │
                                  └────────────────────┘

        all available receipts ──▶ capstone Exp6952 ──▶ V609
~~~

Exact solvers define fixture labels and external outcomes. Learned energies may
rank candidates. They may not certify their own labels. Outcome labels stay out
of model prompts and memory reads.

## Phase A: Evidence, Contract, and Exact Prefix Data

### Exp6941: V608 post-marker source delta and compatibility audit

Recheck the dated V608 papers and named secondary sources. Record metadata,
implementation state, local compatibility, citation edges, and explicit
no-update rows. This task is advisory. No science task gates on it.

### Exp6942: V608 executable contract and bounded-shard preflight

Compare this document with the active YAML. Require exactly 12 tasks and exact
parity for IDs, titles, deliverables, and gates. Validate model contracts,
prompt endings, prior-failure blocks, exclusion rules, and bounded work units.
This task is the root gate for the exact-prefix and ARC corpus branches.

### Exp6943: Verifier-density prefix corpus with exact first-error labels

Build at least 144 deterministic trace pairs across arithmetic updates, graph
reachability, and small scheduling problems. Emit exact prefix-validity vectors,
first-invalid-step labels, density bands, split hashes, and fresh-process replay.
This fixture can receive only a circular-positive conformance verdict.

The activation refusal matched this task to four older corpus-shaped failures.
The YAML now records all four with exact verdicts, changed mechanisms, and
`retire_if_same_verdict: true`. V608 uses exact deterministic semantics. It does
not reuse a safety classifier, adaptive injection ensemble, ARC RFT corpus, or
LoRA checkpoint path.

### Exp6944: Three-family bounded non-saturated prefix reasoning bank

Run only after Exp6943 emits `prefix_corpus_ready_score == 1`. Generate exactly
144 attempts across the three mandated GGUF families. Checkpoint each output
before the next call. Freeze raw text before exact checking. Keep malformed,
timed-out, and low-quality rows in the denominator. Readiness does not require
headroom or accuracy.

## Phase B: Structural and Internal Causal Verification

### Exp6945: Exact-prefix credit geometry and structural-energy canary

Run only after Exp6944 emits `reasoning_bank_complete_score == 1`. Compare
outcome-only, uniform reward-to-go, exact first-error, and shuffled
matched-concentration credit. Use one matched architecture and reward mass.
Require held-out top-1 selection gain. AUROC alone is not positive evidence.

### Exp6946: GGUF causal-state surface and replay receipt

Run only after Exp6944 emits `reasoning_bank_complete_score == 1`. Probe tokens,
logits, embeddings, and intermediate layers at exact trace-line boundaries.
Record unsupported runner surfaces directly. Require replayable non-text state
from at least one Qwen family and one Gemma family before downstream use.

### Exp6947: Causal hidden-state selection with random-direction controls

Run only after Exp6946 emits `causal_state_surface_ready_score == 1`. Compare
compact state rankers with at least 20 norm-matched random directions,
likelihood, final embeddings, structural energy, fixed order, and shuffled
labels. Require paired exact top-1 gain with a confidence interval above zero.

## Phase C: ARC Branch-Discriminative Energy

### Exp6948: ARC live-attempt branching corpus audit

Run only after Exp6942 emits `v608_execution_contract_ready_score == 1`. Scan
the live agent's own attempt receipts for normalized states with at least two
executed actions and exact successor observations. Exclude source inspection,
hand adapters, offline ground-truth search, and synthetic counterfactuals. The
task makes no game-level or level-level solve claim.

### Exp6949: Within-game branch-discriminative world-state energy

Run only after Exp6948 emits `arc_branch_corpus_ready_score == 1`. Compare
predicted-state matching energy, supervised next-state regression, action-only
ranking, and shuffled labels. Train and test chronologically within each game.
Keep the result shadow-only and default-off. Do not change the solve registry.

## Phase D: Reversible Self-Learning and Synthesis

### Exp6950: Prospective Trace-as-State continuous self-learning

Run only after Exp6943 emits `prefix_corpus_ready_score == 1`. Compare no
memory, append-after memory, and trace-before memory in separate processes.
Only a prior exact-success receipt can authorize a write. Match trace bytes and
token budgets between memory arms. Keep model weights frozen. Test retention,
poison, contradiction, restart, tombstone, and rollback behavior. This is the
required continuous self-learning experiment for FR-11.

### Exp6951: Fresh-process trace-memory causal audit

Run only after Exp6950 emits `trace_state_run_complete_score == 1`. Recompute
all comparative metrics from raw rows and saved stores. Verify time order,
receipt authority, equal budgets, model immutability, retention, and rollback.
The audit may not upgrade an upstream null result.

### Exp6952: V608 independent capstone and V609 handoff

Aggregate all available receipts without gating on a positive branch. Recompute
the task contract, branch headlines, controls, model coverage, ARC boundaries,
memory safety, and exclusion implications. Preserve missing, blocked, circular,
null, partial, disqualified, and adversarially flagged states.

## Exact Task Contract

| Order | Task ID | Title | Deliverable | Structured gate |
|---:|---|---|---|---|
| 1 | exp6941-v608-source-delta | V608 post-marker source delta and compatibility audit | results/experiment_6941_v608_source_delta.json | none |
| 2 | exp6942-v608-contract-preflight | V608 executable contract and bounded-shard preflight | results/experiment_6942_v608_contract_preflight.json | none |
| 3 | exp6943-verifier-density-prefix-corpus | Verifier-density prefix corpus with exact first-error labels | results/experiment_6943_verifier_density_prefix_corpus.json | exp6942-v608-contract-preflight.v608_execution_contract_ready_score == 1 |
| 4 | exp6944-three-family-prefix-bank | Three-family bounded non-saturated prefix reasoning bank | results/experiment_6944_three_family_prefix_bank.json | exp6943-verifier-density-prefix-corpus.prefix_corpus_ready_score == 1 |
| 5 | exp6945-prefix-credit-energy-canary | Exact-prefix credit geometry and structural-energy canary | results/experiment_6945_prefix_credit_energy_canary.json | exp6944-three-family-prefix-bank.reasoning_bank_complete_score == 1 |
| 6 | exp6946-gguf-causal-state-surface | GGUF causal-state surface and replay receipt | results/experiment_6946_gguf_causal_state_surface.json | exp6944-three-family-prefix-bank.reasoning_bank_complete_score == 1 |
| 7 | exp6947-causal-hidden-selection | Causal hidden-state selection with random-direction controls | results/experiment_6947_causal_hidden_selection.json | exp6946-gguf-causal-state-surface.causal_state_surface_ready_score == 1 |
| 8 | exp6948-arc-branch-corpus | ARC live-attempt branching corpus audit | results/experiment_6948_arc_branch_corpus.json | exp6942-v608-contract-preflight.v608_execution_contract_ready_score == 1 |
| 9 | exp6949-arc-branch-energy | Within-game branch-discriminative world-state energy | results/experiment_6949_arc_branch_energy.json | exp6948-arc-branch-corpus.arc_branch_corpus_ready_score == 1 |
| 10 | exp6950-trace-state-self-learning | Prospective Trace-as-State continuous self-learning | results/experiment_6950_trace_state_self_learning.json | exp6943-verifier-density-prefix-corpus.prefix_corpus_ready_score == 1 |
| 11 | exp6951-trace-memory-cold-audit | Fresh-process trace-memory causal audit | results/experiment_6951_trace_memory_cold_audit.json | exp6950-trace-state-self-learning.trace_state_run_complete_score == 1 |
| 12 | exp6952-v608-capstone | V608 independent capstone and V609 handoff | results/experiment_6952_v608_capstone.json | none |

## Dependency Graph

~~~text
Exp6941                                      advisory root
Exp6942 ──contract-ready──▶ Exp6943 ──corpus-ready──▶ Exp6944
Exp6944 ──bank-complete───▶ Exp6945
Exp6944 ──bank-complete───▶ Exp6946 ──state-ready──▶ Exp6947
Exp6942 ──contract-ready──▶ Exp6948 ──corpus-ready─▶ Exp6949
Exp6943 ──corpus-ready────▶ Exp6950 ──run-complete▶ Exp6951
Exp6952                                      ungated aggregate
~~~

No science task gates on Exp6941 or Exp6952. Each blocked producer must still
write its declared gate field and `gate_check_summary`. The conductor can then
skip only direct dependents.

## Model Contract

Every task that invokes an LLM declares `MODEL_SPECS` through the current local
GGUF resolver.

| Task | Required headline model set |
|---|---|
| Exp6944 | Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B-it |
| Exp6946 | Qwen3.6-35B-A3B, Gemma-4-31B-it, and Gemma-4-26B-A4B-it |
| Exp6950 | Qwen3.6-35B-A3B and Gemma-4-26B-A4B-it |

Legacy small models may run CPU smoke tests. They cannot support a headline.
GGUF tokenizers come from the GGUF file or runner. No task calls
`AutoTokenizer.from_pretrained()` on a GGUF repository ID.

## Acceptance and Reporting Rules

- Every task writes its artifact when blocked.
- Every artifact declares the closed `verdict_class` enum.
- Every blocked verdict names the failed check, expected value, and observed
  value in `gate_check_summary`.
- Every comparative task emits per-unit rows for each item, arm, seed, family,
  and condition used by a headline.
- Every model task records model, quantization, runner, device, offload, cache,
  duration, task-owned receipt, and teardown.
- Every learned result records split hashes and label-isolation checks.
- Exact outcomes remain external authority. A learned verifier cannot certify
  its own labels.
- Exp6948 and Exp6949 record `solve_claimed=false`. They do not need
  `solve_provenance` because neither task claims a game-level solve.
- No task modifies `scripts/research_conductor.py`.

## Hardware Requirements

| Tasks | Hardware | Budget and boundary |
|---|---|---|
| 6941-6943, 6945, 6947-6949, 6951-6952 | CPU and network where stated | Bounded audits, exact replay, or small-model training. No accelerator claim. |
| 6944, 6946, 6950 | Dual RTX 3090 CUDA | Run local GGUF models sequentially. Checkpoint each output or event. Record GPU UUIDs, offload, peak VRAM, and teardown. |
| all tasks | Disk | Reuse cached GGUF files. Put large states and checkpoints under results/checkpoints/. |
| none | KV260, GateMate, PolarFire, XTR-0, or Z1 | These devices remain outside the dependency graph. Make no speed, power, or availability claim. |

## Decentralization Implications

V608 remains local-first. All model work uses open local GGUF weights. Exact
solvers and stored receipts remain the authority. No closed model, hosted API,
or unavailable hardware sits on a science dependency path.

## Estimated Execution Budget

| Phase | Tasks | Expected wall time |
|---|---|---:|
| A | 6941-6944 | 12-17 hours |
| B | 6945-6947 | 16 hours |
| C | 6948-6949 | 7 hours |
| D | 6950-6952 | 19 hours |
| Total | 12 tasks | 54-59 hours |

The conductor executes tasks serially. GPU tasks use bounded row counts and
per-unit checkpoints. A failed unit remains visible and cannot trigger an
unbounded retry loop.

## Explicitly Deferred

- Weight updates and LoRA continual learning remain deferred. V608 tests
  reversible external trace state with frozen weights.
- A full EBT or Kona training run remains deferred. Kona is an architecture
  comparator without public weights or a local runner.
- Generated-text and log-probability reward scorers remain retired.
- New ARC game or level solves remain outside V608. The ARC branch is a
  provenance-qualified shadow experiment only.
- FPGA and TSU performance work remains outside this milestone.

## Completion Contract

V608 is complete when these 12 task IDs have terminal artifacts or
conductor-written blocked artifacts in this order. Exp6952 must preserve every
non-positive evidence class. Before activation, validate this document against
research-roadmap-next.yaml for exact task count, ID order, titles,
deliverables, and gates. The exclusion-manifest lint must report no hard
violations.
