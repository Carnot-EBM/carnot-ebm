# Research Roadmap vNEXT - Milestone 2026.08.578

**Milestone:** 2026.08.578  
**Title:** Execution-Qualified Verification and Support-Preserving Self-Learning  
**Experiments:** Exp6619-Exp6632  
**Phases:** four  
**Planning date:** 2026-08-26

## What milestone 2026.08.577 proved

Milestone 2026.08.577 completed all three tasks that reached the conductor. Its
result is an execution-contract finding, not a scientific result.

1. Exp6616 found that the staged roadmap document promised Exp6616-Exp6628, but
   `research-roadmap.yaml` contained only Exp6616-Exp6618. It honestly returned
   `blocked_roadmap_contract_incomplete` and set
   `execution_contract_ready_score=0`.
2. Exp6617 was correctly blocked by the failed Exp6616 gate. Repeated gate
   checks did not implement the GPU lease. Exp6618 was then skipped because its
   upstream task retired.
3. The conductor gate failed closed. No task silently treated missing model,
   process, GPU, phase, or science evidence as success.
4. No mandated GGUF family ran in V577. No constrained-decoding, spectral,
   memory-actionability, or continuous-learning claim was tested.
5. The failure exposed a missing planning invariant: a roadmap document and its
   execution YAML must be validated as one complete activation unit before the
   first task starts.

V576 remains the last scientific evidence base. It established an exact
two-level plan corpus, a null live invariant projection, promising but blocked
software-only spectral rows, a conformant memory lifecycle without utility, and
zero prospective self-learning benefit. V578 must preserve those verdicts.

## The three biggest gaps to the PRD vision

### Gap 1: the execution substrate cannot yet support a flagship claim

The PRD requires verifiable local inference. Carnot has exact corpora and model
loaders, but V576 and V577 did not produce one complete flagship direct-baseline
artifact with task-scoped GPU ownership, process identity, model hash, phase
receipts, raw rows, and unload evidence. Until this exists, constrained decoding
has no eligible baseline and model-family comparisons have no scientific
meaning.

**V578 response:** validate the complete 14-task activation manifest, implement
the lease and receipt substrate, run independent canaries for all three mandated
families, then qualify one bounded Qwen3.6 baseline through an independent
reducer.

### Gap 2: learned verification can alter support or its own decision threshold

Carnot separates learned proposal scores from exact release authority, but it
does not yet measure two 2026 failure modes. Prior audit-repair context can shift
an LLM verifier's criterion. Verifier-guided optimization can improve pass-at-one
while reducing fixed-budget recoverable support for later objectives. A valid
output rate alone cannot reveal either failure.

**V578 response:** compare direct, syntax-only, and delayed semantic constraint
arms only after direct headroom exists. Run a separate cold-context verifier
experiment with length-matched controls. Audit exact false accepts, semantic
support, and best-at-k across candidate budgets.

### Gap 3: persistent memory is safe but behaviorally inert

The PRD requires autonomous continuous self-learning from verifier feedback.
V576 proved lifecycle conformance but not useful learning. Its live projection
had no effect, and the prospective treatment had zero benefit. Memory must be
state-grounded, invoked by the live policy, error-independent from the model
that produced the candidate, and useful on held-future events.

**V578 response:** establish live-policy influence before patching memory. Admit
only component-scoped patches that repair a source failure without reducing
anchors or recoverable support. Compare memory against frozen and context-only
controls across seeds and task orders. Keep model weights and the base policy
immutable.

## Research inputs added before planning

The dated V578 section of `research-references.md` records the external scan.
The following findings directly shape this milestone:

- arXiv:2608.16003 requires fresh-context and length-matched controls for learned
  verifier signals after audit-repair histories.
- arXiv:2608.00017 requires error-independence checks and forbids same-model
  self-grades from admitting memory updates.
- arXiv:2608.00220 requires fixed-budget recoverable-support measurements before
  and after a verifier-governed update.
- arXiv:2608.03874 requires a context-only control and proof that retrieved state
  influenced the live action.
- arXiv:2608.12700 supports independent execution, exactness, evidence, and
  protection gates rather than one pooled success flag.
- Extropic's summer 2026 update makes Torx and THRML useful portability targets.
  Carnot still has no authenticated TSU runner, so V578 makes no TSU claim.

## Milestone thesis

Carnot can make scientific progress only after the execution chain is
replayable. Once one flagship family has complete direct rows and real headroom,
factored constraints can improve exact success without silently collapsing
semantic support. Separately, a state-grounded memory patch can improve held
future behavior only when its admission signal is exact, cold-context-safe, and
independent of the error process it judges.

The milestone has four release boundaries:

1. **Execution boundary:** no model science without identity, lease, process,
   phase, accelerator, and unload receipts.
2. **Verification boundary:** only exact executable checks release outputs.
   Learned signals are diagnostic or routing evidence.
3. **Learning boundary:** no memory credit without live influence, source repair,
   held-anchor retention, prospective benefit, and support preservation.
4. **Hardware boundary:** RTX 3090 measurements are local accelerator evidence.
   Rust, Torx simulation, and THRML compatibility are not TSU hardware evidence.

## Target architecture

```text
            complete V578 roadmap document + YAML
                           |
                 [Exp6619 activation contract]
                           |
                 [Exp6620 lease + phase journal]
                           |
         +-----------------+--------------------+
         |                                      |
 [Exp6621 model canaries]                CPU evidence branch
         |                               [Exp6627 integrity]
 [Exp6622 Qwen direct rows]                      |
         |                               [Exp6628 CPU/GPU replay]
 [Exp6623 independent reducer]
         |
         +----------------------+-----------------------+
         |                      |                       |
 [Exp6624 delayed        [Exp6625 cold-context   [Exp6629 live memory
  two-level decoding]     verifier control]       actionability]
         |                                              |
 [Exp6626 exact safety]                         [Exp6630 patch gate]
                                                        |
                              [Exp6631 prospective continuous learning]
                                                        |
                           [Exp6632 independent capstone]

Learned path:  propose -> route -> constrain -> patch -> abstain
Exact path:    parse -> execute -> check -> admit/reject -> release
State path:    observe -> retrieve -> influence -> exact feedback -> candidate patch
Hardware path: CPU reference -> Rust parity -> RTX GPU replay -> future TSU portability
```

The exact path never consumes a learned verifier verdict as release authority.
The state path never mutates model weights or the immutable base policy. The
capstone runs even when earlier gates block and reports the dependency cut.

## Phase I: activation truth and bounded flagship headroom

### Exp6619 - complete V578 activation and gate-ownership contract

Validate that the document and YAML describe the same Exp6619-Exp6632 task set.
Check IDs, deliverables, milestone values, prior failures, gate owners, required
artifact fields, model policy, protected files, and prompt endings. This differs
from Exp6616 because the complete YAML exists before the task starts.

**Deliverable:** `results/experiment_6619_v578_activation_contract.json`  
**Gate field:** `activation_contract_ready_score`

### Exp6620 - task-scoped GPU lease and phase receipts

Implement the previously blocked lease substrate. Bind task, token, device UUID,
PID start time, model hash, VRAM, phase, heartbeat, terminal reason, and unload
evidence. Prove race, stale-owner, PID-reuse, tamper, timeout, and restart
behavior with process fixtures before loading a model.

**Deliverable:** `results/experiment_6620_gpu_lease_phase_receipts.json`  
**Gate field:** `gpu_lease_scheduler_ready_score`

### Exp6621 - independent mandated-model admission canaries

Run short fresh-process CUDA canaries for Qwen3.6-35B-A3B, Gemma-4-31B-it, and
Gemma-4-26B-A4B-it. Emit readiness per family. One failure must not erase another
family's valid receipt. Legacy small models cannot satisfy readiness.

**Deliverable:** `results/experiment_6621_mandated_model_admission.json`  
**Gate fields:** `qwen_admission_ready_score`, `gemma31_admission_ready_score`,
`gemma26_admission_ready_score`

### Exp6622 - bounded Qwen3.6 direct-headroom requalification

Use the exact V576 corpus and one frozen Qwen3.6 configuration. Run direct
generation only. Preserve every attempt, exact result, failure category, timing,
model identity, phase receipt, and candidate-budget row. Do not add a treatment.

**Deliverable:** `results/experiment_6622_qwen36_direct_headroom.json`  
**Gate field:** `baseline_rows_complete_score`

### Exp6623 - independent headroom and support reducer

Replay Exp6622 from raw rows. Freeze eligibility, exact success, syntax failure,
semantic failure, best-at-k, and preregistered headroom. The reducer must not
load an LLM. It opens Phase II only when the direct arm is complete and has
nontrivial repair headroom.

**Deliverable:** `results/experiment_6623_headroom_support_reducer.json`  
**Gate field:** `constrained_decoding_ready_score`

## Phase II: constrained decoding and verifier-context safety

### Exp6624 - delayed two-level constrained decoding

Compare direct, syntax-only, and delayed syntax-plus-semantic constraints on the
same prompts, seeds, budgets, model, and exact checker. Preserve free-form
reasoning until the structured-output trigger. Report exact validity, semantic
success, best-at-k, support, latency, and constraint activation per unit.

**Deliverable:** `results/experiment_6624_delayed_two_level_decoding.json`  
**Gate field:** `decoding_rows_ready_score`

### Exp6625 - cold-context verifier criterion control

On byte-identical labeled traces, compare learned verifier output in fresh,
prior audit-repair, and length-matched neutral contexts. Use Qwen3.6 and Gemma
26B as independent measurement families. Exact labels remain authority. Report
criterion, discrimination, false positives, false negatives, and per-row logits
or stable scores.

**Deliverable:** `results/experiment_6625_cold_context_verifier_control.json`  
**Gate field:** `cold_context_verifier_ready_score`

### Exp6626 - exact constraint authority and support audit

Independently replay Exp6624. Attack incomplete semantic automata, malformed
plans, contradictory constraints, budget inflation, duplicate candidates, and
learned-verifier disagreement. Measure false accepts and recoverable support at
each budget. Only the exact checker can accept.

**Deliverable:** `results/experiment_6626_constraint_authority_support_audit.json`

## Phase III: spectral sampler evidence and local acceleration

### Exp6627 - spectral integrity and reference repair

Repair the V576 spectral artifact's failing parity, cost, test, and protection
receipts. Compare the Rust k-block kernel against exact enumeration where
tractable and a sequential Gibbs reference above that range. Separate setup,
transition, effective-sample-size, autocorrelation, and wall time.

**Deliverable:** `results/experiment_6627_spectral_integrity_repair.json`  
**Gate field:** `sampler_integrity_ready_score`

### Exp6628 - independent CPU/GPU spectral replay

Replay the qualified kernel in a fresh process. Compare sequential CPU, Rust
k-block CPU, and batched RTX GPU execution on identical Ising instances and
seeds. Report setup and transfer costs. Retain a sparse 16-neighbor portability
record for future Torx, THRML, or Z1 work, but make no TSU claim.

**Deliverable:** `results/experiment_6628_spectral_cpu_gpu_replay.json`

## Phase IV: live actionability and continuous self-learning

### Exp6629 - held-out live memory actionability canary

Route typed invariant memory through the real `E3AgentPolicy` decision path on a
held archive. Record state, retrieval, candidate action before and after memory,
influence, exact outcome, and abstention. Use game-agnostic interfaces. Make no
ARC game or level solve claim.

**Deliverable:** `results/experiment_6629_live_memory_actionability.json`  
**Gate field:** `live_memory_activation_ready_score`

### Exp6630 - error-independent component patch gate

Localize a failed trajectory to one typed component: stored record, working
state, invocation rule, or exact checker binding. Compare exact feedback with a
same-model self-grade only as a contamination diagnostic. Admit no patch unless
the exact signal is error-independent, the source failure repairs, anchors
retain, recoverable support does not fall, restart replays, and rollback works.

**Deliverable:** `results/experiment_6630_error_independent_memory_patch_gate.json`  
**Gate field:** `memory_patch_contract_ready_score`

### Exp6631 - prospective multi-order continuous self-learning

Compare frozen, context-only, and verifier-governed memory arms across multiple
seeds and chronological, reverse, and shuffled orders. Apply only past accepted
patches to later events. Keep weights and base policy frozen. Report activation,
future exact success, best-at-k, support, retention, poison, restart, rollback,
and variance per event.

**Deliverable:** `results/experiment_6631_prospective_support_preserving_csl.json`

### Exp6632 - independent capstone and architecture reconciliation

Replay every available V578 artifact, including blocked artifacts. Recompute
gates and headline claims from per-unit rows. Report the strongest supported
execution, decoding, verifier-context, sampler, and self-learning statements.
Update capability specs, traceability, architecture, status, and changelog only
to match evidence.

**Deliverable:** `results/experiment_6632_v578_independent_capstone.json`

## Dependency graph

```text
Exp6619 activation contract
  -> Exp6620 GPU lease
       -> Exp6621 model admission
            -> Exp6622 direct Qwen rows
                 -> Exp6623 headroom reducer
                      -> Exp6624 delayed constraints
                           -> Exp6626 exact authority audit
            -> Exp6625 cold-context verifier control
            -> Exp6629 live memory actionability
                 -> Exp6630 component patch gate
                      -> Exp6631 prospective CSL

Exp6619 activation contract
  -> Exp6627 spectral integrity
       -> Exp6628 CPU/GPU spectral replay

All available artifacts -> Exp6632 capstone (ungated)
```

Structured gates:

| Downstream | Upstream field | Condition |
|---|---|---|
| Exp6620 | `exp6619.activation_contract_ready_score` | `== 1.0` |
| Exp6621 | `exp6620.gpu_lease_scheduler_ready_score` | `== 1.0` |
| Exp6622 | `exp6621.qwen_admission_ready_score` | `== 1.0` |
| Exp6623 | `exp6622.baseline_rows_complete_score` | `== 1.0` |
| Exp6624 | `exp6623.constrained_decoding_ready_score` | `== 1.0` |
| Exp6625 | `exp6621.qwen_admission_ready_score` | `== 1.0` |
| Exp6626 | `exp6624.decoding_rows_ready_score` | `== 1.0` |
| Exp6627 | `exp6619.activation_contract_ready_score` | `== 1.0` |
| Exp6628 | `exp6627.sampler_integrity_ready_score` | `== 1.0` |
| Exp6629 | `exp6621.qwen_admission_ready_score` | `== 1.0` |
| Exp6630 | `exp6629.live_memory_activation_ready_score` | `== 1.0` |
| Exp6631 | `exp6621.qwen_admission_ready_score` and `exp6630.memory_patch_contract_ready_score` | both `== 1.0` |

Exp6632 has no structured gate. It must diagnose a broken chain rather than
disappear behind it.

## Hardware requirements and boundaries

- **Primary inference hardware:** two local RTX 3090 GPUs with task-scoped
  leases. Record UUID, PID start, model hash, free and resident VRAM, offload,
  heartbeat, exit, unload, and residual VRAM.
- **CPU and RAM:** required for exact checkers, independent reducers, Rust
  reference kernels, and sampler parity. Report CPU model, thread count, RAM,
  and affinity when measured.
- **Local acceleration:** Exp6628 may use an RTX 3090 for batched Ising updates.
  It must report host-device transfer, setup, transition, and wall costs.
- **Attached FPGA boards:** KV260, GateMate, and PolarFire are out of scope
  without a new changed-state receipt. Do not probe them merely because they are
  attached.
- **Thermodynamic hardware:** no authenticated XTR-0 or Z1 runner is available.
  Torx, Thermalizers, THRML, and Z1 sparse topology are interface references.
  Simulator evidence cannot support a TSU latency, power, or hardware claim.
- **Disk:** preserve raw rows, logs, process receipts, model hashes, and atomic
  artifacts. Stop before inference when the preregistered evidence budget is not
  available.

## Model policy

Every task that invokes an LLM binds at least one of these exact repo IDs in
`MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Exp6621 uses all three. Exp6622, Exp6624, Exp6629, and Exp6631 use Qwen3.6.
Exp6625 uses Qwen3.6 and Gemma 26B. Each uses the cached SOTA pattern from
`scripts/experiment_template.py`, records exact model and quant hashes, derives
tokenizer and chat-template behavior from GGUF metadata, and refuses silent
fallback. Qwen3.5-0.8B and Gemma-4-E4B-it may run CPU smoke tests only. Their
rows cannot satisfy readiness or headline fields.

## Claim, ARC, and safety boundaries

- No task claims that a learned verifier is an oracle.
- No task accepts a candidate without an exact executable checker.
- `circular_positive` is required if a task accidentally uses its own tested
  mechanism as authority. Such a result cannot become a headline.
- Comparative tasks retain one row per prompt, seed, budget, instance, event,
  or condition. Aggregates are secondary.
- Every blocked artifact uses `gate_check_summary` and names the failed check and
  observed value.
- Exp6629-Exp6631 improve only a game-agnostic live-policy path. They make no ARC
  game-level or level-level solve claim and therefore receive no solve credit.
- No offline ground-truth BFS, game-source reading, hand GameAdapter, or
  per-game calibration may enter the live path.
- Memory updates are typed, local, reviewable, immutable until admitted, and
  reversible. The generator weights and base policy remain frozen.
- `research-roadmap.yaml` and `scripts/research_conductor.py` are protected in
  every task.

## Expected milestone outcomes

A positive milestone has all of the following:

1. A complete activation contract and task-scoped execution receipts.
2. At least one mandated family with replayable direct rows and honest headroom.
3. An exact constrained-decoding result that reports semantic support as well as
   validity, or an honest null/block showing why it cannot proceed.
4. A cold-context verifier result that separates criterion shift from
   discrimination.
5. A repaired sampler result with independent CPU/GPU cost accounting, or an
   honest failure that retires the unchanged scope.
6. A prospective self-learning result with live influence, context-only control,
   exact error-independent admission, held-future benefit, and support
   preservation, or an honest null that prevents another passive-memory rerun.

The capstone may still be partial. It must preserve every negative result and
identify the smallest next dependency cut.
