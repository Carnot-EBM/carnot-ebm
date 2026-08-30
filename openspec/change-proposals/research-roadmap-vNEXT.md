# Carnot Research Roadmap vNEXT: Causal Replay and Real-Output Fixed Points

**Created:** 2026-08-30  
**Milestone:** 2026.08.593  
**Status:** Planned  
**Supersedes:** milestone 2026.08.592  
**Experiments:** Exp6796-Exp6807 (12 tasks, four phases)  
**Research refresh:** `research-references.md`, `V593 Planner Refresh - 2026-08-30`

## What Milestone 2026.08.592 Proved

Milestone `.592` completed all twelve conductor tasks. Its scientific branches had different
outcomes, so their metrics must remain separate.

| Branch | Terminal evidence | What `.592` proved | Remaining boundary |
|---|---|---|---|
| Infrastructure | Exp6785 preserved 9 prefix rows and resumed 15 rows exactly; Exp6784 blocked only because `research-roadmap-next.yaml` did not yet exist | Parent-owned atomic checkpoints and fresh-process resume work | The dispatch matrix still needs requalification against a present next-roadmap manifest |
| Soft fixed point | Exp6788 grouped minus flat exact-valid delta was `+0.1021`; its paired 95% lower bound was `+0.0760`; Exp6789 reproduced the effect under cold destructive audit | Declarative group topology improves a bounded synthetic proposal task without exact-oracle feedback | Every graph was a synthetic implication chain, star, or cycle; transfer to real SOTA-model outputs is unknown |
| Continuous self-learning | Exp6791 recorded 1,063 online writes, 3,132 later reads, 721 changed actions, and a five-order held-future lower bound of `+0.34375` versus frozen | A prospective component-level route learner has useful action headroom and a positive source comparison | Exp6792 found 3,189 commits but zero stored parent/new canonical byte snapshots, so causal, restart, rollback, poison, and retention credit remains withheld |
| Temporal Ising | Exp6793 and Exp6794 completed matched-update, exact-law simulation | The tested temporal exchange schedule is measurable and reproducible | Its efficiency gate and one target-law stratum failed; the unchanged schedule is retired and no hardware work follows from it |
| Capstone | Exp6795 read every branch and emitted a partial disposition without pooling metrics | Null and blocked branches remained visible | FR11 causal credit and real-output FR12 transfer remain open |

The `.592` result is therefore a bounded bridge, not a production architecture claim. The exact
checker remains the only validity authority. A learned fixed point proposes. A transactional memory
routes. Neither can certify itself.

## The Three Biggest Gaps to the PRD Vision

### Gap 1: FR11 self-learning lacks byte-replayable causal evidence

The online route learner changed later actions and improved held-future utility, but its artifact
stored hashes without the canonical parent and new state bytes. An independent process cannot prove
that each hash names the state that was read, committed, restarted, or restored. The smallest next
step is not another learning method. It is a snapshot-complete deterministic replay followed by the
already specified causal and safety audit.

### Gap 2: FR12 fixed-point evidence is synthetic, not model-output evidence

The grouped operator beat a parameter-matched flat control on three hand-generated topology
families. Carnot has a completed, frozen corpus from the mandated Qwen3.6 and Gemma-4 GGUF families,
but the grouped operator has never seen constraint graphs derived from those outputs. The next test
must freeze the `.592` recipe, create formally calibrated refinement and restructuring probes, and
measure zero-shot transfer with exact post-proposal checking.

### Gap 3: The live ARC supervisor and selfparse tool path lack causal progress evidence

The production selfparse transport works on its covered call shape, but the supervisor default
window gives too few firings, no live `tool_gap_events` rows prove that an unknown capability reaches
the generator, and the treatment-only evidence cannot establish an actions-to-progress effect. The
next live chain must use a window of 120, task-owned SOTA GGUF execution, durable cell checkpoints,
an actually unset control environment, and live-agent solve provenance. It must not solve games
offline or claim a duplicate level.

## Research Leads Promoted into the Design

| Source | Mechanism used in `.593` | Evidence boundary |
|---|---|---|
| X-RAY, arXiv:2603.05290 | Paired formal probes that separate constraint refinement from solution-space restructuring | The exact solution set is generated and checked independently; source model identity is a stratum, not a feature |
| Package-hallucination defenses, arXiv:2608.22652 | Adversarial structural rewrites plus utility/support preservation | A decoding score or lower soft energy cannot certify a candidate |
| Compositional Online Learning, arXiv:2608.27244 | Component-attributed writes, reads, and route actions | `.593` repairs causal bytes; it does not change the successful learning mechanism before audit |
| Thermalizing Stochastic Programs, arXiv:2608.01615 | Typed-factor representation and short-trajectory error accounting for the audited operator | Local simulator/compiler evidence only; no Extropic device claim |
| SymDiag, arXiv:2608.08786 | Translation-versus-reasoning failure separation in real-output probes | Exact symbolic encodings remain authority; no LLM judge |

The OpenReview, Hugging Face Papers, Semantic Scholar, GitHub, Extropic, KAN, and Kona checks found
no public checkpoint, exact verifier, or hardware interface that should replace the current local
stack. Extropic now targets Z1 systems for 2027 early access. Kona still exposes no reproducible
runner. These are watch items, not milestone dependencies.

## vNEXT Architecture

```text
                         FAIL-CLOSED HANDOFF
             research-roadmap-next.yaml + dispatch audit
                                  │
              ┌───────────────────┼────────────────────┐
              │                   │                    │
              ▼                   ▼                    ▼
       CSL deterministic     Frozen real GGUF      Task-owned live ARC
       transaction replay    output corpus         Qwen3.6-35B runtime
              │                   │                    │
       parent/new bytes      formal graph pairs    window-120 shadow
       + hash receipts       ┌─────┴──────┐         supervisor receipts
              │              │refinement  │               │
              ▼              │restructure │         live tool-gap event
       cold causal/safety    └─────┬──────┘               │
       restart/rollback            │                 paired selfparse
       poison/retention            ▼                 actions-to-progress
                             frozen .592 recipe            │
                       ┌────────────┴────────────┐          ▼
                       │ grouped fixed point    │     cold live-path audit
                       │ matched flat control   │
                       └────────────┬────────────┘
                                    ▼
                             exact post-proposal
                             checker + cold audit
                                    │
                                    ▼
                         typed-factor portability
                         on local simulator only
              │                   │                    │
              └───────────────────┴────────────────────┘
                                  ▼
                       ungated branch disposition
```

The real-output branch does not invoke an LLM. It consumes the frozen, completed GGUF corpus and
preserves the exact source model IDs and hashes. The ARC branch does invoke an LLM and pins
`MODEL_SPECS` to `unsloth/Qwen3.6-35B-A3B-GGUF`. It must resolve the local GGUF through the shared
SOTA resolver, prove llama.cpp CUDA offload, own the lease and server, and stop with a diagnostic
blocked artifact if the exact model cannot run. No legacy small model, remote API, or CPU substitute
may supply headline rows.

## Phase 1: Evidence-Contract Closure (Exp6796-Exp6798)

### Exp6796: Dispatch-contract requalification

Re-run the existing fail-closed compatibility implementation after
`research-roadmap-next.yaml` exists. Audit every `.593` agent/model pair, reproduce the Exp6781
cross-vendor rejection, and mutate a temporary manifest to prove failure occurs before dispatch.
This task may repair a narrow audit defect but may not modify the conductor.

**Prototype:** a static compatibility and manifest audit with one row per pair.  
**Empirical criterion:** all real `.593` pairs pass; the known bad pair and unknown namespaces fail
with stable reason codes.  
**Adversarial check:** remove, cross-wire, and corrupt agent/model fields in temporary manifests.

### Exp6797: Canonical transaction-byte replay fixture

Re-execute the deterministic Exp6791 comparison from the frozen Exp6790 stream without changing its
arms, order, thresholds, or action policy. Preserve every prior row outcome. For each committed
transaction, store canonical parent and new state bytes beside their hashes. Do not edit historical
artifacts or infer bytes from hashes.

**Prototype:** a snapshot-complete 4-arm, 5-order replay with atomic checkpoint/resume.  
**Empirical criterion:** all 4,800 cells reproduce, all 3,189 commits carry parent and new bytes, and
every recomputed hash matches.  
**Adversarial check:** flip bytes, reorder receipts, cross-wire arm stores, interrupt, restart, and
attempt a manifest-mismatched resume.

### Exp6798: Independent CSL causal and safety audit

Run the Exp6792 audit in a fresh module against Exp6797. Recompute all aggregates from rows and raw
bytes. Disable each credited update or retrieval, replay poison and capacity attacks, restart from
committed bytes, and force rollback on preregistered harm.

**Prototype:** cold transaction replay with causal and safety counterfactuals.  
**Empirical criterion:** every credited factor has a changed-action and utility witness; restart and
rollback are byte-identical; poison never enters or influences active memory; retention and hard-case
gates pass. A null or disqualified result is acceptable.  
**Adversarial check:** stale parents, valid-hash/wrong-arm bytes, reordered commits, poisoned receipts,
and rollback to a non-parent state.

This phase contains the milestone's mandatory continuous self-learning experiment. It tests the
positive learner prospectively and independently. It does not award FR11 credit from source
aggregates alone.

## Phase 2: Real-Output Fixed-Point Transfer and Hardware Portability (Exp6799-Exp6802)

### Exp6799: Formally calibrated model-output constraint probes

Transform frozen rows from Exp6745 and Exp6768 into exact-enumerable graph pairs. Each source case
gets a constraint-refinement variant and a solution-space-restructuring variant. Match variable
count, solution count band, exact-check cost, and surface length as closely as possible. Preserve
source model, family, case, output, and artifact hashes. Do not generate new model text.

**Prototype:** an X-RAY-style paired probe fixture derived from the three mandated GGUF families.  
**Empirical criterion:** at least 96 complete paired probe groups span all source models and required
families, with disjoint development and held-case splits and exact cold replay.  
**Adversarial check:** solution-preserving renames, parser disagreements, source-model label shuffles,
and restructuring operations that accidentally reduce to simple refinement.

### Exp6800: Frozen grouped fixed point versus matched flat control

Reconstruct the `.592` grouped and flat arms from the frozen Exp6786 training split, seeds, and
hyperparameters. First reproduce their `.592` candidate hashes. Then freeze both arms and evaluate
them zero-shot on Exp6799. No model-derived probe may enter fitting, early stopping, threshold
selection, or decoding. The exact checker runs only after each proposal.

**Prototype:** a paired zero-shot transfer comparison over model, family, case, transformation,
seed, and arm.  
**Positive criterion:** the grouped-minus-flat exact-valid lower confidence bound is above zero on
restructuring probes, with no preregistered refinement, support, convergence, or work-budget harm.
Completion does not depend on a positive effect.  
**Adversarial check:** dependency-edge removal, group-ID permutation, source-model identity removal,
surface relabeling, and an identical-arm sentinel.

### Exp6801: Cold real-output authority audit

Recompute the probe semantics, candidate validity, clustered intervals, work matching, and verdict in
a fresh process without importing the Exp6800 producer. Verify that training remained confined to
Exp6786 and that neither model ID nor exact labels entered proposal features.

**Prototype:** row-derived independent audit with destructive controls.  
**Empirical criterion:** all planned rows and source hashes reproduce; destructive controls behave as
preregistered; any source shortcut or oracle leak disqualifies the claim.  
**Adversarial check:** duplicate cases, swapped transformation labels, forged model IDs, hidden exact
features, and aggregate/row disagreement.

### Exp6802: Typed-factor portability of the audited operator

Map the grouped and flat update kernels from the audited real-output branch into the existing typed
stochastic factor IR. Use exact-enumerable small graphs and short trajectories. Report per-factor
conditional KL, trajectory total variation, precision, topology, and compiler provenance. Exact
target fitting is structurally circular and must be labeled as such.

**Prototype:** a local CPU/Torx-compatible compiler-fidelity study for the new operator class.  
**Empirical criterion:** the full factor/trajectory grid completes and independent exact enumeration
reproduces every reported error. This is representability evidence, not a reasoning or hardware win.  
**Adversarial check:** factor-order permutation, precision reduction, context mismatch, accumulated
trajectory error, and target/objective circularity classification.

## Phase 3: Live ARC Supervisor and Tool Causality (Exp6803-Exp6806)

### Exp6803: Window-120 shadow-supervisor accrual

Run the production ARC path with the supervisor in shadow mode and window 120. Use task-owned
Qwen3.6-35B-A3B CUDA inference, atomic cell checkpoints, frozen seeds, and a registry precheck.
Collect enough independent firings for both arms before any downstream usefulness claim. Shadow
mode may observe and redirect on paper but may not apply a world-model mutation.

**Prototype:** owned live cells with supervisor receipt transport.  
**Empirical criterion:** at least ten eligible firings per arm, complete receipt linkage, no applied
shadow mutation, and clean lease/server teardown.  
**Adversarial check:** default-window regression, duplicate firing, stale receipt, restart, orphaned
server, source access, per-game adapter, and duplicate-level target.

### Exp6804: Live selfparse tool-gap transport

On the ready shadow substrate, enable the production selfparse loop and capture a genuine unknown
capability demand as a `tool_gap_events` row. Prove that the event reaches the tool-gap generator,
creates a bounded candidate request, and remains subject to the exact tool schema and safety limits.

**Prototype:** one end-to-end live gap event plus bounded negative controls.  
**Empirical criterion:** nonzero live events traverse detection, serialization, generation request,
validation, and receipt linkage with the exact Qwen model provenance.  
**Adversarial check:** fabricated tool names, malformed XML, oversized arguments, stale events,
server-lifted versus selfparse transport confusion, and missing schema propagation.

### Exp6805: Selfparse actions-to-progress A/B

Run paired held live cells with identical model, games, seeds, prompts, action budgets, supervisor
window, and wall limits. The treatment enables `CARNOT_ARC_INDUCE_TOOL_LOOP=selfparse`. The control
must have that environment variable absent, not restored to a treatment value by `apply_arm`. Measure
actions to verified progress, tool use, firings, harmful redirects, level progress, and cost from
live rows.

**Prototype:** durable paired treatment/control live-agent evaluation.  
**Positive criterion:** the paired lower confidence bound for treatment minus control progress is
above zero with no action-budget, safety, or retention harm. Completion remains valid when null.  
**Adversarial check:** treatment contamination of control, unmatched checkpoints, already-reproduced
levels, outer-loop reverse engineering, hand adapters, source access, and offline ground-truth search.

### Exp6806: Independent ARC live-path audit

Audit Exp6803-Exp6805 from raw cells and process receipts. Recompute firings, tool-gap transport,
actions-to-progress, level provenance, resource ownership, and paired effects. The audit may adopt,
hold, retire, or disqualify each capability separately.

**Prototype:** cold receipt and row audit with no LLM invocation.  
**Empirical criterion:** every headline is row-derived, every solve row is
`live_agent_self_discovery`, the control environment is absent, and resource teardown is clean.  
**Adversarial check:** duplicate rows, missing headroom, null fields, registry duplication, outer-loop
artifacts, model substitution, and aggregate-only claims.

## Phase 4: Branch Disposition (Exp6807)

### Exp6807: V593 branch disposition and PRD reconciliation

Read every `.593` artifact, including blocked, null, circular, disqualified, and partial outcomes.
Recompute branch summaries from rows and independent audits. Update the research ledger, traceability,
status, and changelog with exact follow-up or retirement conditions. Do not pool dispatch readiness,
CSL causality, fixed-point transfer, simulator portability, or ARC progress into one score.

**Prototype:** receipt-only milestone synthesis.  
**Empirical criterion:** all twelve task IDs have a terminal disposition and every PRD gap names the
smallest changed next step or retirement.  
**Adversarial check:** missing artifacts, source/audit disagreement, positive source with failed audit,
circular hardware evidence, and duplicate ARC solve credit.

## Dependency Graph and Conductor Order

```text
Exp6796  dispatch requalification ───────────────────────────────────────┐
                                                                         │
Exp6797  canonical byte replay ──► Exp6798 cold CSL audit ───────────────┤
                                                                         │
Exp6799  formal real-output probes ──► Exp6800 frozen fixed-point A/B     │
                                               │                         │
                                               ▼                         │
                                      Exp6801 cold authority audit       │
                                               │                         │
                                               ▼                         │
                                      Exp6802 typed-factor portability ──┤
                                                                         │
Exp6803  window-120 shadow accrual                                        │
    │                                                                    │
    ▼                                                                    │
Exp6804  live tool-gap transport                                          │
    │                                                                    │
    ▼                                                                    │
Exp6805  actions-to-progress A/B                                          │
    │                                                                    │
    ▼                                                                    │
Exp6806  independent ARC audit ──────────────────────────────────────────┤
                                                                         ▼
                                                                  Exp6807 capstone
```

Structured gates consume completion/readiness fields spelled exactly in the upstream task's required
artifact fields. They do not require a positive scientific effect. Exp6807 is ungated so a resource
block cannot suppress the milestone disposition.

## Hardware Requirements

| Tasks | Substrate | Expected memory | Expected time | Claim boundary |
|---|---|---:|---:|---|
| Exp6796 | CPU, local YAML and Python | 2-4 GB RAM | 30-90 min | Dispatch readiness only |
| Exp6797-Exp6801 | CPU; local checkpoints; CPU PyTorch/NumPy | 4-12 GB RAM | 1-4 h each | Deterministic CSL replay and real-output proposal evidence |
| Exp6802 | CPU exact enumeration and local typed-factor simulator | 4-8 GB RAM | 2-4 h | Compiler portability only |
| Exp6803-Exp6805 | One task-owned RTX 3090, exact cached Qwen3.6 GGUF, CPU/RAM for ARC agent | At least 22,610 MiB free VRAM before each load; 32-64 GB system RAM | 4-15 h with durable resume | Live-agent evidence only; no fallback model |
| Exp6806-Exp6807 | CPU, artifact readback | 4-8 GB RAM | 1-3 h | Independent audit and synthesis |

- **RTX 3090 pair:** only one card is required at a time. The task must own the selected physical
  UUID, port, child process, and teardown. It may wait within its declared lease deadline but may not
  kill unrelated work.
- **Mandated GGUF:** live ARC tasks pin `unsloth/Qwen3.6-35B-A3B-GGUF`. The artifact records the hub
  ID, resolved path hash, quantization, llama.cpp build, physical UUID, and GPU-offload proof.
- **Legacy small models:** permitted only in unit smoke fixtures. They cannot emit headline rows or
  unblock a missing flagship runtime.
- **KV260 and GateMate:** prior terminal receipts stand. No duplicate bitstream or board task.
- **PolarFire:** opportunistic only and absent from the dependency graph.
- **Extropic Z1/X0:** no authenticated device is available. Exp6802 uses the local typed-factor
  compiler/simulator and cannot claim device speed, power, sampling fidelity, or availability.
- **Strix NPU and ROCm:** not substitutes for the pinned live CUDA path.

## Milestone Acceptance and Stop Rules

1. All twelve tasks write their declared artifact or a terminal diagnostic blocked artifact.
2. Every comparative claim emits per-unit rows. Every blocked result names its failed check and
   observed value in `gate_check_summary`.
3. Every artifact declares `verdict_class` from the closed enum and a terminal-prefix
   `honest_verdict`.
4. Exp6797 reproduces the prior learning actions before Exp6798 can grant any causal credit. Missing
   raw bytes block; hashes alone do not pass.
5. The real-output fixed-point branch fits only on the frozen Exp6786 training split. Exact labels,
   source model identity, and Exp6799 held probes stay outside fitting and proposal features.
6. Live ARC rows require the exact declared SOTA GGUF, task-owned CUDA offload, durable checkpoints,
   registry precheck, and `solve_provenance=live_agent_self_discovery` for any level progress.
7. No blocked branch is replaced by a smaller model, remote API, reduced cohort, offline ARC solver,
   hand GameAdapter, outer-loop reverse engineering, or changed endpoint metric.
8. A repeated prior verdict activates the task's `retire_if_same_verdict: true` signal. The capstone
   records retirement instead of proposing an unchanged rerun.

## Explicit Non-Goals

- No new proof generation or repeat of the three-model runtime-admission chain.
- No generated-text external energy scorer and no learned-verifier release authority.
- No online base-LLM weight training; `.593` self-learning is transactional constraint routing.
- No KAN scaling branch before byte-replayable causal memory evidence closes.
- No temporal-exchange rerun without a new invariant kernel and proof.
- No duplicate ARC level solve, offline ground-truth BFS, source read, or per-game adapter.
- No physical FPGA, NPU, or TSU performance claim.
- No modification to `scripts/research_conductor.py`, no push, and no autonomous publication change.
