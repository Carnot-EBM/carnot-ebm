# Research Roadmap vNEXT: Provenance-Valid ARC Belief Decisions and Selective Self-Learning

**Milestone:** `2026.09.616`

**Status:** Proposed

**Task contract:** exactly 10 tasks, `exp7028` through `exp7037`, in the exact order in this document

**Staged executable contract:** `research-roadmap-next.yaml`

**Execution-time contract:** `research-roadmap.yaml` after activation

**North star:** improve the live ARC-AGI-3 discovery agent's accuracy or action efficiency without game source, offline ground-truth search, or per-game adapters

## 1. Executive Summary

Milestone `2026.09.615` built the intended belief-memory architecture but did
not establish value. The chronological stream, counterexample ledger, bounded
query API, and default-off E3 selector all completed. The deterministic
prospective comparison was a controlled null, and the cold audit found the
ledger safe but not promotable.

The live experiment did not answer the scientific question. Exp7025 passed
every cache, GPU, CUDA-offload, lease, port, model-server, context, and official
ARC access check. It then failed because llama.cpp reported the resolved
Hugging Face blob path while the ARC provenance contract required a requested
`.gguf` filename. Exp7026 was mechanically gate-blocked. Exp7027 prescribed a
bounded V616 handoff: repair that identity defect, then rerun only the live
shadow and held-mechanic cells.

V616 follows that handoff and adds one literature-motivated continuation. It
first proves a hash-bound snapshot-to-blob identity bridge. It then runs the
previously blocked uniform belief A/B once. A fresh audit decides whether the
result is positive, a real null, or disqualified for lack of headroom. Finally,
a selective belief-use learner, inspired by SEVRA and FlowBalance, trains only
on completed intervention outcomes and is tested prospectively on a disjoint
live roster. Repeating the null retires the relevant belief-value scope.

The milestone contains exactly 10 tasks. The contract preflight and source
audit are independent roots. Neither gates the science chain, so a reporting
defect cannot erase the live experiment.

## 2. What V615 Proved

V615 proved the following implementation and evidence facts.

1. The belief stream is chronological, game-blind, and separates pre-action
   information from later outcomes.
2. The belief ledger supports contradiction clustering, tombstones, bounded
   capacity, restart, rollback, and protected-case retention.
3. On the frozen deterministic fixture, the belief arm did not satisfy its
   paired lower-bound gate. The result was
   `complete_null_prospective_belief_utility_not_demonstrated`.
4. The independent audit reproduced safety but not value promotion:
   `belief_shadow_safe_score=1`, `belief_promotion_ready_score=0`.
5. The bounded query API and belief-aware E3 selector are wired behind a
   default-off path.
6. The live substrate itself was available. The shadow run reached an owned
   CUDA llama.cpp server with Qwen3.6-35B-A3B and failed only at the model
   identity/provenance boundary.
7. No live belief comparison completed. V615 therefore did not prove either a
   positive or a live null.

V615 also exposed an execution-time contract mistake. Exp7016 attempted to
read `research-roadmap-next.yaml` after activation, when the active contract
was `research-roadmap.yaml`. V616's preflight reads the active file and keeps
the staged filename only as a planning-time receipt.

## 3. The Three Biggest Gaps to the PRD Vision

### Gap 1: The live hidden-game agent still lacks a provenance-valid value result

The PRD calls for useful constraint-guided reasoning, not merely a safe helper
API. Carnot has no completed live result showing that belief evidence improves
progress, actions-to-progress, or cost against matched base and simulation
controls. This is the immediate north-star gap.

### Gap 2: Continuous self-learning has lifecycle safety but no prospective utility

FR-11 requires directed self-learning from experience. The project can append,
rollback, restart, quarantine, and retain state, but V615's first prospective
belief test was null. A useful learner must decide when an intervention helps,
abstain when exact outcomes show no preference, and retain that decision on a
later disjoint stream.

### Gap 3: Carnot is still a hybrid verifier framework, not the PRD's native continuous latent EBM

The production architecture combines local autoregressive generators, exact
validators, learned sidecars, and sampler backends. It does not yet provide a
trainable foundation-scale continuous reasoning state refined by one native
global energy. EBT, ARM-EBM, and Kona remain architectural evidence, but no
matching open local checkpoint or reproducible training path changes that
fact. V616 does not disguise this long-term gap as a provenance patch. It asks
whether the current hybrid architecture produces measurable live value while
keeping the native-EBM direction as a future model-access/training milestone.

## 4. Research Inputs

The planning-time source record is in `research-references.md`, section
"V616 Planner Refresh - 2026-09-05". The experiment-changing inputs are:

- Self-Reports Are Not Verification (`arXiv:2609.00652`): use exact later
  environment outcomes, never selector confidence or rationale, as authority.
- SEVRA (`arXiv:2606.19808`): treat belief use as a selective intervention and
  compare always-off, always-on, and extra-base-compute controls.
- FlowBalance (`arXiv:2609.03241`): retain guidance only under positive exact
  group advantage and disable it under no preference.
- Online Learnability of CoT Verifiers (`arXiv:2603.03538v4`): report false
  acceptance and false rejection costs separately under distribution shift.
- EBT (`ICLR 2026`) and ARM-EBM (`arXiv:2512.15605`): preserve the distinction
  between whole-configuration energy and next-step policy scores; neither is a
  runnable V616 dependency.

Extropic Z1T and Logical Intelligence Kona remain architecture comparators.
Carnot has no authenticated Z1/TSU route and no public Kona checkpoint. V616
makes no hardware latency, power, execution, or availability claim.

## 5. Target Architecture

```text
cached_sota_pair()
       |
       v
requested snapshot path ---- filename / hub id / revision ----+
       |                                                       |
       +---- content hash -------------------------------+      |
                                                         v      v
llama.cpp /props --> observed resolved blob path --> hash-bound identity bridge
                                                         |
                                                         v
                                         ARC provenance receipt + live row
                                                         |
                  +-------------------+------------------+----------------+
                  |                   |                                   |
                  v                   v                                   v
             base policy       simulation policy                 belief query
                                                                      |
                                                                      v
                                                               E3 action rank
                                                                      |
                                                                      v
                                                         exact later observation
                                                                      |
                                     +--------------------------------+------+
                                     |                                       |
                                     v                                       v
                          immutable intervention row             no self-report authority
                                     |
                         calibration roster only
                                     |
                                     v
                         bounded selective-use learner
                         help / harm / abstain counts
                                     |
                           freeze before held roster
                                     |
                                     v
                  always-off vs always-on vs selective vs extra-base control
                                     |
                                     v
                              release or retire
```

The identity bridge never infers identity from a basename alone. It carries
the requested snapshot `.gguf` path, the observed resolved blob path, hub ID,
revision, requested filename, and one content hash. A different hash, wrong
repository, unresolved path, or ambiguous requested file is a hard rejection.

The learning boundary is chronological. The selective policy sees only fields
available before an action. Exact later observations may update the next
decision, but they cannot score the current action at selection time.

## 6. Experimental Design

### Primary hypotheses

1. A strict snapshot-to-blob identity bridge can accept the exact model that
   llama.cpp loaded without weakening rejection of aliases or different files.
2. Uniform belief access either improves held-mechanic live decisions under a
   fixed budget or earns a terminal retirement-quality null.
3. A bounded selective-use learner can preserve helpful interventions and
   abstain on harmful or no-preference contexts better than always-on belief.

### Outcomes

- Progress per 100 actions and actions to next progress event.
- Helpful, harmful, missed-helpful, and abstained belief interventions.
- Invalid or contradicted actions, model calls, generated tokens, wall time,
  and task-linked GPU receipts.
- Identity acceptance/rejection by exact path, revision, repository, filename,
  and content hash.
- Protected-case retention, bounded state size, restart, rollback, and poison
  behavior for the selective learner.

### Controls and sample rules

- Uniform live comparison: base, simulation-only, belief-only, and combined.
- Selective live comparison: always-off, always-on, selective, and an
  extra-base-compute control.
- Qwen3.6-35B-A3B is the primary live family. Gemma-4-26B-A4B is the uniform
  comparison canary. Gemma-4-31B is the selective comparison canary.
- At least 30 held primary-family decision units are required for a
  percentage-point claim. Canary rows are family-bound and not pooled into
  the headline.
- A null is scientific only if the intervention changes decisions and the
  oracle-best logged arm has headroom. Otherwise the result is disqualified,
  not called negative.
- All comparative claims are recomputable from per-unit rows. No pooled number
  can override a contradictory row audit.

### Claim boundary

The ARC environment is an evaluation authority available only after an action;
it is not a policy input. Positive policy claims therefore require
`verifier_is_oracle=false`. Selector self-reports are telemetry. This milestone
does not propose a game-level solve. Any incidental solve may be recorded only
with `solve_provenance=live_agent_self_discovery` after a registry precheck and
must not become the milestone headline.

## 7. Phases and Exact Task Contract

The following table is the complete task contract. It contains exactly the 10
tasks in `research-roadmap-next.yaml`, with identical IDs, titles,
deliverables, order, and structured prerequisites.

| Order | Experiment | Title | Deliverable | Structured prerequisites |
|---:|---|---|---|---|
| 1 | `exp7028-v616-active-contract-preflight` | V616 active-roadmap and design-document contract preflight | `results/experiment_7028_v616_active_contract_preflight.json` | None |
| 2 | `exp7029-v616-sota-scope-audit` | V616 post-marker source delta and experiment-scope audit | `results/experiment_7029_v616_sota_scope_audit.json` | None |
| 3 | `exp7030-arc-gguf-model-identity-bridge` | ARC GGUF snapshot-to-blob model identity bridge | `results/experiment_7030_arc_gguf_model_identity_bridge.json` | None |
| 4 | `exp7031-arc-model-identity-cold-audit` | Fresh-process ARC model identity and alias-confusion audit | `results/experiment_7031_arc_model_identity_cold_audit.json` | Exp7030 `arc_model_identity_bridge_ready_score == 1` |
| 5 | `exp7032-repaired-belief-shadow-live-trace` | Repaired provenance-complete live belief shadow trace | `results/experiment_7032_repaired_belief_shadow_live_trace.json` | Exp7030 `arc_model_identity_bridge_ready_score == 1`; Exp7031 `arc_model_identity_audit_ready_score == 1` |
| 6 | `exp7033-uniform-belief-live-ab` | Held-mechanic uniform belief live A/B and retirement test | `results/experiment_7033_uniform_belief_live_ab.json` | Exp7032 `belief_shadow_trace_ready_score == 1` |
| 7 | `exp7034-live-belief-cold-audit` | Independent live belief value and provenance cold audit | `results/experiment_7034_live_belief_cold_audit.json` | Exp7033 `belief_live_comparison_complete_score == 1` |
| 8 | `exp7035-selective-belief-csl` | Prospective selective belief-use continuous self-learning A/B | `results/experiment_7035_selective_belief_csl.json` | Exp7034 `belief_live_value_audit_complete_score == 1` |
| 9 | `exp7036-belief-release-or-retire` | ARC belief policy release-or-retire lifecycle closure | `results/experiment_7036_belief_release_or_retire.json` | Exp7034 `belief_live_value_audit_complete_score == 1`; Exp7035 `selective_belief_policy_complete_score == 1` |
| 10 | `exp7037-v616-capstone` | V616 independent evidence capstone and V617 handoff | `results/experiment_7037_v616_capstone.json` | None; structurally ungated |

### Phase I: Contract and current-source boundary (`exp7028`-`exp7029`)

Exp7028 reads the active `research-roadmap.yaml`, not the staging file that
disappears during activation. It independently checks exact Markdown/YAML
parity, gates, producer fields, model rules, failure disclosures, and prompt
tails. It reports defects but does not gate science.

Exp7029 performs the reserved post-marker source audit. It records arXiv,
OpenReview, Hugging Face, Semantic Scholar, GitHub, Extropic, and Logical
Intelligence access results. It may narrow a task for a verified new safety
control, but it may not expand the 10-task contract or reopen a retired scope.

### Phase II: Model identity repair (`exp7030`-`exp7031`)

Exp7030 implements the dual-path identity receipt in the shared ARC provenance
path. Tests begin with the exact Exp7025 snapshot-symlink/resolved-blob shape.
The implementation accepts only a content-hash and declared-model match; it
does not relax the `.gguf` filename requirement or accept arbitrary blobs.

Exp7031 audits the bridge from a fresh process with mutation fixtures: wrong
hash, wrong revision, wrong repository, misleading basename, broken symlink,
hard-link ambiguity, missing requested filename, and stale server path. The
live trace remains blocked until every negative control is rejected.

### Phase III: Repaired uniform live decision (`exp7032`-`exp7034`)

Exp7032 reruns only the failed V615 shadow cell with
`unsloth/Qwen3.6-35B-A3B-GGUF`. Belief can be queried but cannot change the
selected action. The row must prove action parity, full identity, CUDA offload,
owned process/lease/port, context, cleanup, and live-agent provenance.

Exp7033 runs the previously blocked held-mechanic comparison. It uses Qwen3.6
as the primary family and Gemma-4-26B-A4B as a family canary. Budgets, roster,
mechanic grouping, and success rules are frozen before outcomes. A repeated
belief null activates the failed-experiment retirement rule.

Exp7034 independently replays hashes, gates, row arithmetic, family
separation, chronology, solve registry, task-linked compute receipts, and
positive-control headroom. It may mark the uniform mechanism promotable,
null, blocked, or disqualified. It cannot turn a safe ledger into a value
claim.

### Phase IV: Selective self-learning and terminal disposition (`exp7035`-`exp7037`)

Exp7035 is the milestone's continuous self-learning experiment. It learns a
small bounded belief-use policy from Exp7033's completed intervention rows,
using only serving-visible pre-action features. Exact outcomes update the next
decision. The policy is frozen before a new disjoint live roster is opened.
Qwen3.6 is primary and Gemma-4-31B is a canary. No-preference groups abstain.

Exp7036 applies a deterministic release-or-retire table. A clean selective
positive becomes a default-off canary only. A repeated null retires the
matching belief-value scope. A disqualified test stays held without a value
claim. The ordinary production path remains unchanged in every branch.

Exp7037 is ungated. It reads every V616 artifact, recomputes the task contract,
preserves blocked/null/disqualified classes, summarizes exact claims, and
names one bounded V617 handoff. It never publishes or submits externally.

## 8. Dependency Graph

```text
exp7028 contract preflight (independent, advisory) -------------------+
                                                                     |
exp7029 source/scope audit (independent, advisory) -------------------+
                                                                     |
exp7030 identity bridge                                               |
   -> exp7031 cold identity audit                                     |
      -> exp7032 repaired live shadow                                 |
         -> exp7033 uniform live A/B                                  |
            -> exp7034 independent live-value audit                   |
               -> exp7035 selective continuous-learning live A/B      |
                  -> exp7036 release-or-retire closure                |
                                                                     v
                                                        exp7037 ungated capstone
```

Only same-roadmap producers appear in structured gates, and every producer
field is declared verbatim in the corresponding task's required artifact
fields. The capstone is intentionally ungated so external blocks are reported
once rather than retried as partial work.

## 9. Hardware and Runtime Requirements

### Required local compute

- Two local RTX 3090 GPUs with 24 GB VRAM each are the supported live host.
- Exp7032 needs one idle supported GPU for Qwen3.6-35B-A3B.
- Exp7033 resolves Qwen3.6-35B-A3B and Gemma-4-26B-A4B with
  `cached_sota_pair()` and runs the frozen primary/canary schedule without a
  tiny-model headline fallback.
- Exp7035 resolves Qwen3.6-35B-A3B and Gemma-4-31B. Dense-model execution may
  use the existing dual-GPU llama.cpp split only after an explicit admission
  and runner receipt.
- Live tasks require owned GPU and port leases, an owned llama.cpp server,
  observed CUDA layer offload, sufficient free VRAM, writable checkpoints,
  cleanup receipts, and task-linked phase/GPU/model telemetry.
- The model cache must contain the selected Q4_K_M GGUF files. Missing files,
  unsupported CUDA, busy devices, or unavailable live ARC access produce a
  terminal `blocked` artifact with `gate_check_summary`; there is no CPU or
  legacy-small-model headline fallback.

### Non-GPU work

Contract, literature, identity fixtures, cold audits, disposition, and
capstone tasks run on CPU and read existing artifacts. They use one of the six
current legal `inference_substrate` values exactly; they do not mint another
free-text alias.

### External and attached hardware

No KV260, GateMate, PolarFire, Extropic Z1, or Kona task is on the dependency
graph. Current board receipts are terminal or opportunistic, and no changed
authenticated board state is needed for the V616 question. Z1T's public
software may be cited, but no TSU hardware metric is eligible.

## 10. Promotion, Retirement, and Stop Rules

Uniform or selective belief value is positive only when all preregistered
quality, headroom, chronology, provenance, protected-case, and matched-budget
gates pass on per-unit rows. `verdict_class=positive` is forbidden when a
failed acceptance gate exists. A circular selection result uses
`circular_positive`, never `positive`.

If the uniform live test reproduces the prior belief-utility null, its task
uses the prior honest verdict and `retire_if_same_verdict: true`. The selective
learner is the one changed technique allowed by the literature. If it also
reproduces the null, the belief-value scope is retired and V617 must not
propose another ledger, KAN compressor, or threshold sweep without an operator
override and a genuinely changed evidence source.

An external block is terminal `blocked`, not `partial`. The artifact names the
failed prerequisite and observed value in `gate_check_summary`. The capstone
may summarize the block but cannot promote absent evidence.

## 11. Deliverable Boundary

This milestone creates experiment code, tests, specs, artifacts, and the
conditional lifecycle receipt described in its task prompts. It does not
modify `scripts/research_conductor.py`, does not modify the active roadmap at
planning time, does not push, and does not perform external publication or
leaderboard submission.
