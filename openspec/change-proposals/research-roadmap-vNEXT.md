# Research Roadmap vNEXT — Milestone 2026.07.515

**Milestone:** 2026.07.515  
**Title:** Prospective Constraint Drift, Accredited World Models, and Verified Online Adaptation  
**Status:** Proposed  
**Task range:** Exp5769-Exp5781 (13 experiments)  
**Execution file:** `research-roadmap-next.yaml`  
**Date planned:** 2026-07-21

## Thesis

Milestone `.514` found the first credible Tier-1 continuous-learning mechanism in the
current program: solver-certified query-driven constraint acquisition recovered exact
constraints with rollback and retention receipts. It also closed three tempting but
unproductive routes: local GGUF proposal ordering did not produce a positive lower bound,
the allocation-free one-axis PyO3 path did not prove the PRD's consecutive-size 10x rule,
and leave-one-game-out composition of existing ARC components produced no held-out gain.

The next milestone therefore moves the positive constraint-acquisition result out of its
synthetic fixture and into a prospective multi-turn stream from the three mandated local
SOTA GGUF families. In parallel, it follows the separate outer-loop ARC finding that model
capacity matters for single-shot world-model induction while iterative CEGIS refinement is
harmful. The ARC branch will accredit, compare, and select independent single-shot world
models before allowing any learned model to influence the live E3 path.

The milestone makes two claims falsifiable:

1. A solver-certified sidecar can learn constraints online from a chronological SOTA stream,
   reduce satisfiable answer drift, preserve protected facts and old-prefix behavior, and
   transfer across model families without changing GGUF weights.
2. Multiple independently induced ARC world models can be admitted and selected using only
   agent-owned transitions, improving held-out live E3 utility without game identity, source,
   adapters, banked trajectories, or duplicate public-level credit.

## What milestone 2026.07.514 proved

| Branch | Terminal evidence | Consequence for `.515` |
|---|---|---|
| Evidence gates | Exp5757/5758 built lossless scalar bridges, but Exp5755 itself blocked because three `.514` IDs collided with separately produced outer-loop artifacts. | Add an exact-deliverable evidence index and refuse glob/mtime artifact selection before new science. |
| Exact proposal utility | Exp5759 measured `delta_overall=+0.003416`, but its 95% lower bound was `-0.04529`, flagship non-regression failed, and the panel was not ready for promotion. Exp5760 gate-skipped. | Retire SOTA proposal ordering and selective proposal feedback; do not spend another milestone on local score/rank tuning. |
| Constraint acquisition | Exp5761 produced a ready exact CA benchmark. Exp5762 credited query-driven constraint learning with exact behavioral accuracy and lifecycle receipts. Exp5763 credited dependent-task transfer and reported `old_task_retention_delta=+0.12`. | Advance from synthetic exact fixtures to sealed prospective SOTA streams, satisfiable-drift taxonomy, family transfer, and disabled-by-default shadow integration. |
| Rust NFR-01 | Exp5764 produced a ready profiled hot path. Exp5765 still failed the strict consecutive larger-size 10x lower-bound rule and retired this allocation-free one-axis PyO3 technique. | Preserve parity code and the honest retirement. Do not relabel a kernel or hardware smoke number as the PRD production claim. |
| ARC generalization | Exp5766 found `loo_generalization_delta=0`, lower bound `0`, and no causal component interactions. Exp5767 correctly gate-blocked. | Do not retry component composition. Move to the independently identified world-model induction bottleneck. |
| Capstone | Exp5768 reconciled proposal utility not ready, CA credited, Rust 10x retired, and ARC composition blocked. | The only scientific promotions are exact CA and the already-shipped one-axis sampler parity path. |

Separate from the conductor queue, three collision-disclosed outer-loop artifacts answer the
next ARC mechanism question:

- `experiment_5760_cegis_refinement_induction_ab.json`: ThinkingCap CEGIS pooled delta
  `-0.0128`, CI `[-0.032051, 0]`.
- `experiment_5764_gemma31b_singleshot_induction_ab.json`: Gemma 4 31B single-shot pooled
  held-out accuracy `0.378487` versus `0.187604`, delta `+0.190883`, with 12/13 games
  moving off the floor.
- `experiment_5766_gemma31b_cegis_refinement_ab.json`: Gemma 31B CEGIS worsened the
  single-shot result by `-0.318658`; the pooled refinement delta was `-0.0598`, CI
  `[-0.145299, 0]`.

These are development-proxy measurements, not live solves. They justify a matched
single-shot family and hypothesis-selection experiment, not another CEGIS loop.

## The three biggest gaps to the PRD vision

### Gap 1 — Verifiable reasoning is exact but not yet prospective

FR-12 requires verifiable reasoning, yet the strongest current evidence is either a sealed
synthetic CA fixture or a negative SOTA proposal-ranking result. Carnot has not shown that
exact verification plus learned constraints improves a chronological, multi-turn stream of
real flagship-model answers. The missing boundary is not another learned judge: it is a
failure taxonomy and exact ledger that separately measures parser failure, contradiction,
satisfiable drift, protected-fact distortion, and exact answer error.

### Gap 2 — Continuous self-learning has not crossed models or runtime boundaries

FR-11 requires autonomous improvement while preserving prior knowledge. Exp5762/5763
proved a typed constraint lifecycle in a controlled benchmark, but not prospective online
adaptation, leave-one-model-family-out transfer, restart equivalence, or a disabled production
shadow path. `.515` must demonstrate positive new-suffix utility with old-prefix retention,
zero unsafe propagation, exact rollback, and immutable GGUF weights.

### Gap 3 — ARC's live agent lacks an accredited adaptive world model

The public registry is complete, and generic component composition is a held-out null.
Agent-owned evidence now points upstream: many induced executable transition models have
zero held-out accuracy, while a larger dense model materially improves single-shot induction.
Carnot still lacks a principled rule for deciding when an induced model is trustworthy, which
hypothesis to select, and whether the selected model improves a game-blind live E3 policy on
held-out games and actions.

NFR-01's end-to-end 10x Rust target remains open, but the only current software technique is
terminally retired. This milestone preserves that boundary rather than spending another slot
on an unchanged hot path. A future speed claim needs a genuinely different ABI, workload, or
authenticated accelerator precondition.

## 2025-2026 research update and experiment hooks

The full dated search ledger is in `research-references.md` under
`V515-PLANNER-REFRESH-20260721-END`.

| Finding | Carnot implication | Experiment hook |
|---|---|---|
| [Validate the Dream Before You Trust Its Verdict](https://arxiv.org/abs/2607.07196) | A learned world model must pass an admissibility ladder before its simulated verdict can count. Visual or syntactic fidelity does not establish action-following robustness. | Exp5776 defines L0-L4 for executable ARC models; Exp5777/5778 must pass held-out action and calibration rungs before Exp5779. |
| [Residual Drift Dominates Contradiction](https://arxiv.org/abs/2605.23940) | Solver consistency is insufficient: returned answers can violate a satisfiable maintained ledger. | Exp5772 labels contradiction and satisfiable drift separately; Exp5773 validates every answer against the learned ledger. |
| [Distortion Instead of Hallucination](https://arxiv.org/abs/2601.01490) | Better constraint compliance can hide protected-fact distortion. | Exp5772 seals immutable facts separately from mutable constraints; Exp5773 treats any protected-fact change as unsafe propagation. |
| [Bridging the Agent-World Gap](https://arxiv.org/abs/2606.09032) | World-model construction, evaluation, planning, verification, and adaptation must be separate stages. | The ARC branch is staged as contract → family panel → hypothesis selector → live E3 A/B. |
| Extropic XTR-0/Z1 public updates | TSUs remain relevant architecture context, but no authenticated Carnot-local device exists. | No TSU execution, power, latency, or speedup claim. |
| Logical Intelligence Kona/Aleph updates | Learned energy is explicitly an imperfect verifier beneath machine-checkable proof. | Preserve exact validators as authority; Kona remains non-executable context. |

## Target architecture

```text
       mandated local SOTA GGUFs
      Qwen3.6-35B-A3B / Gemma4-31B / Gemma4-26B-A4B
                         |
                         v
       sealed multi-turn finite-choice answer stream
       + immutable facts + mutable constraint ledger
                         |
             +-----------+-----------+
             |                       |
             v                       v
   exact failure taxonomy      solver-certified CA sidecar
   parse / contradiction /     propose -> query -> admit /
   satisfiable drift / fact    refine / quarantine / rollback
   distortion / exact error             |
             |                          v
             +--------------> chronological retention
                                + cross-family transfer
                                + disabled shadow adapter

 agent-owned ARC observations/actions
                         |
                         v
       independent single-shot executable world models
                         |
                         v
       L0 syntax -> L1 seen action -> L2 unseen action
       -> L3 rollout calibration -> L4 closed-loop utility
                         |
                  calibration-only selector
                         |
                         v
       shared live E3AgentPolicy / StepwiseExplorer A/B
       (game blind; no source, adapter, BFS, or banked plan)
                         |
                  real environment authority
```

The two science branches share evidence discipline but not a scientific gate. Constraint-stream
failure cannot suppress ARC evaluation, and ARC model failure cannot suppress FR-11 evaluation.

## Phase 1 — Evidence integrity and prospective stream (Exp5769-Exp5772)

### Exp5769 — Transition `.514` with collision-disclosed archival

Archive `.514` exactly once using declared deliverable paths and conductor outcomes. Preserve the
three same-number outer-loop ARC artifacts as separate evidence, repair no history by deletion, and
allocate collision-free Exp5769-Exp5781 IDs.

**Deliverable:** `results/experiment_5769_transition_v515.json`

### Exp5770 — Post-V515 source-delta ingestion

Repeat the bounded primary/secondary search only for work newer than the V515 planner marker.
Zero accepted findings is a valid complete result; no execution-time finding may silently rewrite
IDs, gates, or headline claims.

**Deliverable:** `results/experiment_5770_v515_source_delta_ingestion.json`

### Exp5771 — Exact-deliverable evidence index and collision preflight

Build a read-only canonical projection from roadmap task ID to declared deliverable, artifact hash,
conductor outcome, and collision aliases. Refuse ambiguous glob/mtime selection, expose duplicate
history blocks without deleting them, and leave `scripts/research_conductor.py` unchanged.

**Deliverable:** `results/experiment_5771_evidence_index_collision_preflight.json`

### Exp5772 — Prospective three-family constraint-drift stream

Using all three mandated SOTA GGUF families, produce a sealed chronological multi-turn stream over
Z3-checkable scheduling, finite-domain, and logic-grid problems. Reuse the already-qualified
finite-choice answer boundary so the retired parse-failure path is not repeated. Seal protected
facts and mutable constraints separately, preserve raw answers, and label parser failure,
contradiction, satisfiable drift, protected-fact distortion, and exact correctness independently.

**Deliverables:**

- `results/experiment_5772_sota_constraint_drift_stream.json`
- `results/experiment_5772_sota_constraint_drift_stream.rows.jsonl`

## Phase 2 — Solver-certified continuous self-learning (Exp5773-Exp5775)

### Exp5773 — Prospective constraint-acquisition A/B

Compare frozen ledger, MUS/contradiction-only feedback, and query-driven constraint acquisition on
the sealed chronological stream. Exact validators are feedback and release authority; the learner
may change only a typed sidecar. Credit continuous self-learning only when satisfiable drift falls
with a positive paired lower bound, protected facts never change, old-prefix retention passes, and
rollback restores exact state hashes.

**Deliverable:** `results/experiment_5773_prospective_constraint_acquisition_ab.json`

### Exp5774 — Leave-one-family-out transfer and forgetting audit

Train the sidecar lifecycle on two generator families and evaluate once on disjoint sessions from
the held-out family, rotating all three families. Separate task-template transfer from model-identity
shortcuts, report negative transfer and dynamic regret, and require chronological prefix retention.

**Deliverable:** `results/experiment_5774_constraint_transfer_forgetting_audit.json`

### Exp5775 — Disabled online shadow integration

Wire the credited sidecar behind a disabled-by-default verification shadow adapter. Replay a fresh
sealed suffix with interruption/resume, state-size caps, quarantine, rollback, and exact answer-ledger
validation. No production answer may be changed in this task.

**Deliverable:** `results/experiment_5775_constraint_sidecar_shadow_integration.json`

## Phase 3 — Accredited ARC world-model generalization (Exp5776-Exp5779)

### Exp5776 — ARC executable-world-model admissibility contract

Implement and test an embodiment-appropriate L0-L4 ladder over agent-owned transition traces:
syntax/compile, seen-action fidelity, held-out/unseen-action fidelity, rollout calibration, and
closed-loop policy utility. Re-score retained single-shot evidence without reading game source or
using a per-game adapter. This task validates models; it does not induce or deploy one.

**Deliverable:** `results/experiment_5776_arc_world_model_admissibility_contract.json`

### Exp5777 — Matched mandated-SOTA single-shot inducer panel

Hash-import the existing Gemma 31B single-shot rows as a disclosed fixed anchor, then run the
missing Qwen3.6-35B-A3B and Gemma4-26B-A4B cells under the identical agent-owned split and
single-shot mechanism. Re-score all families through Exp5776. No refinement, source, adapter,
banked plan, or public solve claim is allowed.

**Deliverable:** `results/experiment_5777_arc_sota_singleshot_inducer_panel.json`

### Exp5778 — Calibration-selected independent world-model hypotheses

Generate independent single-shot hypotheses from the best admissible mandated family. Select using
only calibration transitions, complexity, and stability; evaluate once on future test transitions.
Compare first sample, random selection, and a non-deployable oracle upper bound. This is hypothesis
selection, not iterative CEGIS repair.

**Deliverable:** `results/experiment_5778_arc_calibrated_world_model_selector.json`

### Exp5779 — Held-out live E3 generalization A/B

Only after positive selector and admissibility gates, run baseline versus selected-world-model arms
through the shared live E3 entrypoint on held-out games/actions with identical budgets and seeds.
The task measures live generalization utility, not public solves. Any incidental level progress must
come from the live agent's own attempts and remains registry-neutral.

**Deliverable:** `results/experiment_5779_arc_live_world_model_generalization_ab.json`

## Phase 4 — Hardware boundary and capstone (Exp5780-Exp5781)

### Exp5780 — Multi-board terminal-state and operator-action receipt

Reconcile KV260, PolarFire, and GateMate terminal/continuity evidence without repeating an unchanged
probe. Run a bounded non-destructive board check only when the device/reachability precondition hash
has changed. Preserve PolarFire thermal disclosure, never touch KV260 host `/dev/mmcblk*`, and emit
an exact operator-action packet for any remaining physical/JTAG blocker. No speedup claim.

**Deliverable:** `results/experiment_5780_hardware_terminal_state_receipt.json`

### Exp5781 — `.515` capstone reconciliation

Reconcile task outcomes, gates, negative results, retirements, specs, traceability, references,
status, changelog, and completed research. Keep missing/gate-blocked work distinct from scientific
nulls and never publish or push.

**Deliverable:** `results/experiment_5781_v515_capstone_reconciliation.json`

## Dependency graph

```text
Exp5769 transition
  +--> Exp5770 source ingestion ------------------------------+
  +--> Exp5771 evidence index                                 |
  |      +--> Exp5772 SOTA drift stream                       |
  |              +--> Exp5773 prospective CA A/B              |
  |                      +--> Exp5774 family transfer          |
  |                              +--> Exp5775 shadow adapter   |
  |                                                             +--> Exp5781 capstone
  +--> Exp5776 ARC admissibility contract                     |
  |      +--> Exp5777 SOTA single-shot panel                  |
  |              +--> Exp5778 hypothesis selector             |
  |                      +--> Exp5779 live E3 A/B              |
  +--> Exp5780 hardware boundary -----------------------------+
```

All dependency gates are conjunctive and declared structurally in
`research-roadmap-next.yaml`. Exp5781 always runs and reconciles skipped tasks.

## Model requirements

Every experiment that performs local LLM inference must declare `MODEL_SPECS`, resolve models via
`cached_sota_pair()`, record exact hub IDs and GGUF paths, and include at least one of:

- `unsloth/Qwen3.6-35B-A3B-GGUF` — flagship MoE
- `unsloth/gemma-4-31B-it-GGUF` — flagship dense
- `unsloth/gemma-4-26B-A4B-it-GGUF` — middle MoE

Exp5772 and Exp5777 use all three families. Exp5778 uses the best admissible mandated family and
declares all eligible SOTA specifications. Qwen3.5-0.8B and Gemma4-E4B are smoke-only fallbacks;
their output cannot satisfy a headline-result gate.

## Hardware requirements and boundaries

| Resource | Tasks | Requirement / boundary |
|---|---|---|
| 2x RTX 3090 24 GB | Exp5772, Exp5777, Exp5778 | Real CUDA llama-server offload, VRAM before/after receipt, one large GGUF per card or sequential cells. CPU fallback is blocked for headline evidence. |
| CPU/RAM/disk | All; especially Exp5771, Exp5773-Exp5776 | At least 64 GB free RAM preferred, 100 GB free disk for GGUF outputs/checkpoints, Z3 and exact validators available. Precondition failures emit terminal blocked artifacts. |
| ARC local environment | Exp5776-Exp5779 | Agent-owned observations/actions only. No game source, GameAdapter, offline ground-truth BFS, registry recipe, or banked action plan. |
| KV260 | Exp5780 only | SSH-only if reachability state changed; never access host `/dev/mmcblk*`; no speedup claim. |
| PolarFire SoC | Exp5780 only | Bounded authenticated workload/hash receipt if reachable; disclose passive cooling and duration/temperature; no sustained load. |
| GateMate A1 | Exp5780 only | Do not repeat unchanged detect/flash. Probe only after device/JTAG precondition changes; otherwise emit operator-action receipt. |
| Extropic TSU / Kona | None | No local authenticated execution surface. Architecture context only. |

## Promotion and retirement rules

### Constraint branch

- Promote prospective CA only if `drift_reduction_lcb > 0`, old-prefix retention lower bound is
  non-negative, protected-fact distortion and unsafe propagation are zero, rollback hashes match,
  and GGUF weights remain immutable.
- Promote cross-family transfer only if the leave-one-family-out macro lower bound is positive and
  every family has non-negative retention within its preregistered margin.
- Keep the shadow adapter disabled unless both gates pass. A repeat of the Exp5708 parse-failure or
  Exp5709 gate-block verdict retires this prospective stream shape.

### ARC branch

- A world model may enter selection only at admissibility L2 or higher with zero source/game-ID
  leaks and positive held-out action-following evidence.
- Promote the selector only if its test-set lower bound is positive versus first-sample selection
  and it does not regress unseen-action fidelity.
- Promote live wiring only if the paired live lower bound is positive, validity does not regress,
  and all progress is `live_agent_self_discovery`. Otherwise leave defaults off and retire this
  selector-to-live intervention.
- Iterative CEGIS reinduction remains retired. Exp5778 cannot edit a rejected hypothesis; it may
  only choose among independent single-shot hypotheses.

### Closed branches

- Do not reopen PHASE-D external generated-text/logprob scoring.
- Do not rerun exact local GGUF proposal ordering or selective feedback after Exp5759/5760.
- Do not scale the negative KAN residual.
- Do not retry allocation-free one-axis PyO3 10x optimization without a new technique.
- Do not re-solve any of the 25 complete public ARC games or claim registry credit.
- Do not claim FPGA/TSU acceleration from continuity receipts.

## Required milestone outputs

By capstone, `.515` must leave:

1. a collision-safe evidence index and clean archival boundary for `.514`;
2. a sealed three-family prospective constraint-drift corpus;
3. an honest continuous-self-learning verdict with drift, retention, rollback, and safety receipts;
4. a cross-family transfer verdict and disabled shadow integration receipt;
5. an ARC world-model admissibility contract, matched SOTA inducer panel, and selector verdict;
6. at least one held-out/live-path ARC generalization attempt, even if gate-blocked after a negative
   upstream science result;
7. a no-speedup hardware terminal/action receipt; and
8. reconciled `openspec/`, `_bmad/traceability.md`, `research-complete.yaml`, `ops/status.md`,
   `ops/changelog.md`, and `research-references.md` surfaces.

No task may push, publish, submit, deploy, modify `scripts/research_conductor.py`, or mutate
`research-roadmap.yaml`.
