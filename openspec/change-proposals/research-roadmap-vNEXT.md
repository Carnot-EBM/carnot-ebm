# Research Roadmap vNEXT: Entrance Energy and Context-Bound Learning

**Milestone:** 2026.09.618  
**Status:** Proposed  
**Date:** 2026-09-05  
**Scope:** Exact first-branch support, lossless branch controls, and safe
conditional experience reuse  
**Task contract:** 13 tasks, `exp7050` through `exp7062`, in the exact order
defined below

## What Milestone 2026.09.617 Proved

Milestone `.617` planned 12 tasks. Four tasks reached the conductor. Eight
tasks were cascade-skipped.

| Evidence | Result | Consequence for `.618` |
|---|---|---|
| The V617 design document was absent. | Exp7038 wrote a terminal blocked artifact. | Create this document and the YAML together. Keep the contract preflight advisory. |
| The official Qwen llama.cpp report was captured. | Exp7039 recorded the selected snapshot, launch argument, raw `/props`, resolved blob, and one-token probe. | Reuse the measured report shape. Do not repeat report discovery. |
| The report artifact claimed positive evidence after 46.97 seconds. | Adversarial verification quarantined it because live model evidence requires at least 60 seconds. Its checksum also failed recomputation. | Re-run only the evidence capture with a 75-second minimum and the canonical final-artifact checksum path. |
| The typed identity bridge checked upstream validity. | Exp7040 correctly blocked on the invalid Exp7039 artifact. | Repair the evidence first. Then combine the typed bridge and cold attack audit into one bounded task. |
| The cold audit gated on a zero readiness field. | Exp7041 wrote `blocked_gate_check_failed` three times and retired. | Do not depend on Exp7041. Declare it as a prior failure and use a new upstream chain. |
| The belief-value and self-learning tasks did not execute. | No V617 evidence changed the V615 null belief-utility result. | Do not restart that long ARC chain in this milestone. Test a smaller exact generation boundary and a new learning mechanism. |

The milestone therefore proved an evidence-contract failure. It did not prove
that typed identity, belief guidance, or selective learning is ineffective.
The next plan repairs the invalid evidence in one branch and moves the main
science to independent exact fixtures.

## The Three Biggest Gaps to the PRD Vision

### Gap 1: Live evidence is not yet release-grade

The PRD requires local, auditable inference and deterministic verification.
Carnot can capture the official model path and live server report, but the
latest positive artifact failed the duration and checksum contract. A
downstream verifier cannot trust evidence that its own verifier quarantines.

V618 closes this gap with one evidence requalification and one combined typed
identity attack audit. The branch takes no ARC action and claims no game solve.

### Gap 2: Energy has not changed a useful generation decision

Carnot has exact candidate banks, constraint labels, and several offline
rankers. Recent rankers either lacked class separation, depended on retired
certificate chains, or changed no live action. The PRD calls for generation,
verification, and repair under one energy interface. Carnot still lacks a
clean causal result at the point where a model commits to a reasoning path.

The new literature localizes solution-space contraction to the first branch.
V618 therefore builds an exact entrance-family panel. It measures all three
mandated GGUF families. It then changes only the first branch and preserves
the continuation budget and exact executor.

### Gap 3: Safe memory has not produced useful, portable self-learning

Carnot has chronological stores, rollback, poison tests, and default-off
selectors. V615 found zero held-future belief benefit. Earlier continuous
learning tasks showed safety but often failed utility, support, or audit
gates. The PRD needs an objective loop that learns continuously without
forgetting or reusing stale evidence outside its valid context.

Boundary-Calibrated Intervention Transfer suggests a concrete change. V618
binds each experience to its source context and current parent policy. It
chooses `use`, `validate`, or `reject` before an update. A small software Ising
receipt then tests whether any useful entrance energy has a sparse,
hardware-facing representation. It makes no hardware speed or power claim.

## Research Findings Used in This Plan

The source record is in `research-references.md`, section "V618 Planner
Refresh".

- arXiv:2608.29188 motivates an exact first-branch support panel. It separates
  entrance access from downstream execution.
- arXiv:2601.05724 supplies a lossless hierarchical branch-verification
  control. Distribution fidelity remains separate from correctness.
- arXiv:2608.26730 supplies the context-bound `use` / `validate` / `reject`
  authorization policy for continuous self-learning.
- Extropic's Z1T update motivates degree-bounded sparse placement. Carnot has
  no Z1 access, so this milestone stops at software parity and placement
  feasibility.
- Current KAN and Kona sources add no public local checkpoint or exact
  authority. V618 does not reopen the PWA-KAN lineage.

## Architecture

```text
                    advisory contract check
                     Exp7050 (no gates)
                              |
                              v
                   ungated independent capstone
                          Exp7062

  LIVE EVIDENCE BRANCH
  owned Qwen server
         |
         v
  Exp7051 report evidence requalification
         |
         v
  Exp7052 typed identity bridge + cold attack audit
  (no ARC action, no solve claim)

  ENTRANCE ENERGY BRANCH
  exact Countdown enumeration
         |
         v
  Exp7053 entrance-family fixture
         |
         v
  Exp7054 Qwen + dense Gemma + MoE Gemma proposal bank
         |
         v
  Exp7055 independent support/leakage audit
       /   \
      v     v
  Exp7056   Exp7057
  energy    lossless hierarchical
  causal A/B branch-fidelity control
      |
      v
  Exp7061 sparse Ising energy and ranking parity

  CONTINUOUS SELF-LEARNING BRANCH
  sealed chronological exact outcomes
         |
         v
  Exp7058 context-bound authorization contract
         |
         v
  Exp7059 prospective BCIT self-learning A/B
         |
         v
  Exp7060 cold drift, poison, and rollback audit
```

The contract task and capstone are ungated. The evidence branch cannot block
the entrance or self-learning branches. The entrance branch gates on
completion and exact audit fields before it gates on scientific headroom. The
self-learning branch gates on completion, not on a positive result. This
structure limits cascade loss while preserving real prerequisites.

## Phase 0: Contract and Evidence Recovery

### Exp7050 - V618 active-roadmap and design-document contract preflight

Build an independent parser for this Markdown file and the activated YAML.
Check exact task count, order, IDs, titles, deliverables, gates, upstream
fields, prior-failure blocks, model policy, prompt tails, and the ungated
capstone. The task is advisory. No science task gates on it.

This task addresses Exp7038. The method differs because this design document
exists before the staged YAML is written, and the parser checks 13 independent
rows in both sources.

**Deliverable:**
`results/experiment_7050_v618_active_contract_preflight.json`

### Exp7051 - Official-live model report evidence requalification

Re-run the already measured Qwen report capture. Use the repository model
resolver and an owned CUDA llama.cpp process. Hold the live process for at
least 75 seconds. Make a fixed diagnostic request. Preserve the raw `/props`
payload and the final resolved evidence. Compute the checksum only after the
artifact has reached its terminal form.

This task addresses Exp7039. It changes the two failed evidence conditions. It
does not rediscover the report schema, open an ARC game, or take an ARC action.

**Deliverable:**
`results/experiment_7051_v618_model_report_requalification.json`

### Exp7052 - Typed model identity bridge and fresh-process attack audit

Consume only a valid Exp7051 artifact. Implement the raw-observation to
canonical-resolution identity obligations that Exp7040 could not run. Then
launch a fresh subprocess that attacks alias, hub, revision, hash, broken-link,
hard-link, conflicting-field, and unknown-evidence cases. Unknown fails closed.

This task addresses Exp7040 and retired Exp7041. It combines the bridge and
its independent audit so the new evidence contract cannot create another
retired intermediate gate.

**Gate:**
`exp7051-model-report-evidence-requalification.model_report_evidence_ready_score == 1`

**Deliverable:**
`results/experiment_7052_v618_typed_identity_attack_audit.json`

## Phase 1: Exact Entrance Support and Energy Guidance

### Exp7053 - Exact entrance-family constraint fixture

Create a source-grouped Countdown fixture with deterministic exhaustive
enumeration. An entrance family is the unordered first operand pair plus the
first arithmetic operator. Record every legal first branch, whether it can
reach the target, and one exact continuation witness when one exists. Freeze
calibration and held-out source groups before any model output is visible.

The fixture must contain both reachable and unreachable legal entrances for
each retained unit. It must also contain at least two reachable entrance
families for the diversity subset. The exact solver is label authority only.
It must not appear in model prompts or learned inference features.

**Deliverable:**
`results/experiment_7053_v618_exact_entrance_fixture.json`

### Exp7054 - Three-family SOTA entrance proposal and continuation bank

Run the three mandated local GGUF families with matched prompts, seeds,
sampling settings, context, and completion budgets:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

For each held unit, collect short first-branch proposals. For a fixed subset,
also supply a reachable but unselected entrance prefix and measure downstream
completion. Store raw model text before exact labeling. The task acquires
data. It does not train or select an energy model.

**Gate:**
`exp7053-exact-entrance-family-fixture.entrance_fixture_ready_score == 1`

**Deliverable:**
`results/experiment_7054_v618_three_family_entrance_bank.json`

### Exp7055 - Independent entrance support and leakage audit

Re-enumerate the entrance labels in a fresh process. Verify source-group
isolation, model identity, prompt parity, seed coverage, raw-output hashes,
first-branch parsing, continuation accounting, and the absence of solver fields
from model-visible inputs. Report support and headroom by model and source
group.

Set `entrance_audit_ready_score=1` only when the bank is authentic and
recomputable. Set `entrance_headroom_ready_score=1` only when the held panel
contains nontrivial selector headroom. Keep the two fields separate.

**Gate:**
`exp7054-three-family-entrance-proposal-bank.entrance_bank_complete_score == 1`

**Deliverable:**
`results/experiment_7055_v618_entrance_bank_cold_audit.json`

### Exp7056 - Causal entrance-energy selection comparison

Fit a small additive, Ising-compatible structural energy on calibration
groups only. The energy may use the prompt-visible numbers, operation type,
target residual features, and model proposal frequency. It may not call the
exact solver or read held labels at selection time.

Compare energy selection with model-frequency, uniform-legal,
target-log-probability, and shuffled-energy controls on held source groups.
Use the exact solver only after selection. Measure unreachable-entrance rate,
reachable-family coverage, forced-prefix continuation success, abstention,
and budget. Keep an exact-oracle upper bound in diagnostic rows only.

**Gates:**

- `exp7055-independent-entrance-support-audit.entrance_audit_ready_score == 1`
- `exp7055-independent-entrance-support-audit.entrance_headroom_ready_score == 1`

**Deliverable:**
`results/experiment_7056_v618_entrance_energy_selection.json`

### Exp7057 - Lossless hierarchical branch verification control

Implement a bounded categorical version of hierarchical speculative
verification. Use the frozen proposal distributions from Exp7054. Compare
token-wise, block-wise, and hierarchical verification by total variation from
the target distribution, accepted branches per verification step, and wall
time. Include degenerate, missing-mass, zero-probability, and branch-order
mutations.

This experiment is a distribution-fidelity control. It does not claim better
constraint correctness. It stays independent of the Exp7056 energy result.

**Gate:**
`exp7055-independent-entrance-support-audit.entrance_audit_ready_score == 1`

**Deliverable:**
`results/experiment_7057_v618_hierarchical_branch_control.json`

## Phase 2: Context-Bound Continuous Self-Learning

### Exp7058 - Context-bound experience authorization contract

Create a typed experience record and authorization state machine from the
sealed chronological constraint stream. Each record binds an observed effect
to its parent policy, source group, constraint schema, support interval,
retention result, and named conflicts. The authorization result is `use`,
`validate`, or `reject`.

The contract must hide the current and future exact outcomes from the policy.
It must require a bounded current-state validation event when prior evidence
is related but not directly applicable. It must preserve no-op as a valid
decision.

**Deliverable:**
`results/experiment_7058_v618_bcit_authorization_contract.json`

### Exp7059 - Prospective BCIT continuous self-learning comparison

Run a chronological, read-only-then-commit comparison with four matched arms:
context-bound authorization, flat reuse, validate-all, and no reuse. Each event
may use only earlier observed evidence. Exact current outcomes arrive after the
decision and control admission. Use immutable held-out source groups, bounded
capacity, transaction journals, rollback, and protected retention cases.

Measure harmful update rate, useful update rate, validation cost, final exact
quality, retention, abstention, and equal-budget quality. A no-preference
result selects no-op. This is the milestone's required continuous
self-learning experiment.

This task addresses the null prospective belief utility from Exp7021. The
method changes from context-free belief reuse to context-bound experience
authorization with bounded validation.

**Gate:**
`exp7058-context-bound-experience-authorization.bcit_contract_ready_score == 1`

**Deliverable:**
`results/experiment_7059_v618_bcit_continuous_self_learning.json`

### Exp7060 - Fresh-process BCIT drift, poison, and rollback audit

Replay the Exp7059 journals in a fresh process. Recompute every authorization
and aggregate from event rows. Attack parent-policy drift, source-label drift,
schema drift, stale support, forged positive effects, duplicate events,
reordered events, checksum changes, interrupted commits, rollback, and store
capacity. Confirm that external gate blocks are terminal `blocked`, not
retryable `partial`.

This task addresses Exp6979's flagged cold-audit shape. It uses a deterministic
substrate class and a realistic short-task duration contract. It contains no
live-model marker.

**Gate:**
`exp7059-bcit-prospective-self-learning.bcit_stream_complete_score == 1`

**Deliverable:**
`results/experiment_7060_v618_bcit_drift_cold_audit.json`

## Phase 3: Sparse Energy Parity and Decision Handoff

### Exp7061 - Entrance energy to sparse Ising parity receipt

Compile the frozen Exp7056 additive energy into a bounded QUBO and Ising
representation. Verify exact energy equality up to one declared affine
constant. Verify candidate ranking parity by exhaustive enumeration. Compare
exact sampling statistics with the existing CPU Ising sampler on small graphs.
Then emit a degree-16 placement-feasibility receipt for a Z1-like graph and a
resource estimate for local FPGA simulation.

This is software-only. Set `hardware_used=false`. Do not claim Z1, FPGA,
thermodynamic, power, latency, or speed evidence. Do not build a bitstream.

**Gate:**
`exp7056-causal-entrance-energy-selection.entrance_energy_comparison_complete_score == 1`

**Deliverable:**
`results/experiment_7061_v618_entrance_ising_parity.json`

### Exp7062 - V618 independent capstone and V619 handoff

Read the active roadmap, this document, conductor log, and every available
V618 artifact. Recompute task count, ID order, gate status, artifact validity,
comparative headlines, and branch conclusions. Separate complete null,
blocked, disqualified, circular, and positive results. Release or retire each
branch by evidence. The capstone is ungated so it still runs after a branch
failure.

**Deliverable:**
`results/experiment_7062_v618_capstone.json`

## Exact Task Contract

The staged conductor YAML must contain exactly these 13 tasks, in this order.
Titles and deliverables must match byte-for-byte after YAML parsing.

| Order | Task ID | Title | Deliverable |
|---:|---|---|---|
| 1 | `exp7050-v618-active-contract-preflight` | V618 active-roadmap and design-document contract preflight | `results/experiment_7050_v618_active_contract_preflight.json` |
| 2 | `exp7051-model-report-evidence-requalification` | Official-live model report evidence requalification | `results/experiment_7051_v618_model_report_requalification.json` |
| 3 | `exp7052-typed-identity-bridge-attack-audit` | Typed model identity bridge and fresh-process attack audit | `results/experiment_7052_v618_typed_identity_attack_audit.json` |
| 4 | `exp7053-exact-entrance-family-fixture` | Exact entrance-family constraint fixture | `results/experiment_7053_v618_exact_entrance_fixture.json` |
| 5 | `exp7054-three-family-entrance-proposal-bank` | Three-family SOTA entrance proposal and continuation bank | `results/experiment_7054_v618_three_family_entrance_bank.json` |
| 6 | `exp7055-independent-entrance-support-audit` | Independent entrance support and leakage audit | `results/experiment_7055_v618_entrance_bank_cold_audit.json` |
| 7 | `exp7056-causal-entrance-energy-selection` | Causal entrance-energy selection comparison | `results/experiment_7056_v618_entrance_energy_selection.json` |
| 8 | `exp7057-lossless-hierarchical-branch-control` | Lossless hierarchical branch verification control | `results/experiment_7057_v618_hierarchical_branch_control.json` |
| 9 | `exp7058-context-bound-experience-authorization` | Context-bound experience authorization contract | `results/experiment_7058_v618_bcit_authorization_contract.json` |
| 10 | `exp7059-bcit-prospective-self-learning` | Prospective BCIT continuous self-learning comparison | `results/experiment_7059_v618_bcit_continuous_self_learning.json` |
| 11 | `exp7060-bcit-drift-cold-audit` | Fresh-process BCIT drift, poison, and rollback audit | `results/experiment_7060_v618_bcit_drift_cold_audit.json` |
| 12 | `exp7061-entrance-energy-ising-parity` | Entrance energy to sparse Ising parity receipt | `results/experiment_7061_v618_entrance_ising_parity.json` |
| 13 | `exp7062-v618-capstone` | V618 independent capstone and V619 handoff | `results/experiment_7062_v618_capstone.json` |

## Dependency Graph

```text
exp7050                                      [advisory]

exp7051 -> exp7052                           [live evidence]

exp7053 -> exp7054 -> exp7055 -> exp7056 -> exp7061
                             `-> exp7057     [entrance science]

exp7058 -> exp7059 -> exp7060                [continuous learning]

exp7062                                      [ungated capstone]
```

No task uses `requires:`. Every structured gate names an earlier task in this
milestone. Every gate field appears as a bare top-level field in the upstream
task's required artifact fields. The capstone reads available evidence and
does not gate on branch success.

## Hardware Requirements

| Task group | Requirement | Boundary |
|---|---|---|
| Exp7051 | One idle RTX 3090, owned GPU and port leases, CUDA llama.cpp, cached Qwen3.6 GGUF, at least 75 seconds of owned live runtime | Block if ownership or identity is unclear. Do not kill an unattributed server. No CPU or legacy-model substitute. |
| Exp7054 | Two local RTX 3090 GPUs when available, or one leased GPU with sequential model shards; all three cached mandated GGUF families | Record GPU UUID, model hash, runner, CUDA offload, phase clocks, retries, and cleanup for each shard. A missing family blocks the bank. |
| Exp7053, Exp7055-Exp7060, Exp7062 | CPU, local storage, and existing deterministic solver stack | No remote API model and no hidden answer source. |
| Exp7061 | CPU Ising sampler and software placement model | `hardware_used=false`. No FPGA bitstream and no Z1 execution. |
| All tasks | Writable `results/`, `tests/python/`, task modules, capability specs, and checkpoints as needed | Never write a scratch script in the repository root. |

The KV260 and PolarFire boards are not required. GateMate remains physically
blocked. Extropic hardware remains unavailable. Hardware absence must not
block the main scientific questions.

## Model Policy

Exp7051 uses `unsloth/Qwen3.6-35B-A3B-GGUF`. Exp7054 uses all three mandated
families. Both tasks must resolve models through the
`cached_sota_pair()` pattern in `scripts/experiment_template.py` and record
exact file hashes. The legacy small models may appear only in explicit CPU
smoke rows. They cannot supply a headline result or replace a missing mandated
model.

All other experiments consume frozen model evidence or use deterministic
local computation. They do not need a runtime `MODEL_SPECS` declaration.

## Claim Boundaries

- The exact Countdown solver labels reachability and final correctness. It is
  not a model-visible feature and not a learned energy.
- `verifier_is_oracle=true` forces `verdict_class=circular_positive` for a
  result whose claimed success comes from that same oracle. V618 avoids such a
  headline by comparing pre-verification selectors and reporting oracle bounds
  as diagnostic rows.
- HSD distribution fidelity is not constraint correctness.
- A safe self-learning result is not a useful self-learning result. Exp7059
  reports both separately.
- Software Ising parity is not hardware acceleration.
- No task claims an ARC game-level solve. `solve_provenance` is therefore not
  required in this milestone.
- A failed external prerequisite yields `verdict_class=blocked`. It never uses
  `partial`.

## Milestone Exit Criteria

The milestone is scientifically complete when the capstone can answer all
four questions from valid per-unit evidence:

1. Can Carnot produce a verifier-clean official-live model identity artifact?
2. Do mandated local GGUF models lose reachable solution support at the first
   branch, and can a non-oracle energy improve that decision under matched
   budgets?
3. Does hierarchical verification preserve the captured branch distribution
   while reducing verification work?
4. Does context-bound experience authorization reduce harmful updates or
   improve equal-budget held-future quality without retention loss?

A null answer is valid when headroom, controls, rows, and power are adequate.
A blocked answer names the exact missing evidence. A repeated verdict on a
declared prior failure retires that scope through `retire_if_same_verdict`.

## Operational Verification

Before activation and before completion, run at least:

```bash
.venv/bin/python -c "import yaml; yaml.safe_load(open('research-roadmap-next.yaml'))"
.venv/bin/python scripts/roadmap_schema.py research-roadmap-next.yaml
.venv/bin/python scripts/audit_roadmap_gates.py research-roadmap-next.yaml
.venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap-next.yaml
.venv/bin/python scripts/harness_fit_lint.py research-roadmap-next.yaml
.venv/bin/python scripts/root_clutter_sweep.py --check
```

Each experiment must add its `REQ-*` contract before implementation, add RED
tests first, run focused tests, validate its artifact, run adversarial
verification and row-consistency checks, run scoped spec coverage, and run the
applicable checks from `ops/e2e-test-plan.md`.
