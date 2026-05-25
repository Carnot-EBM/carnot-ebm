# Research Roadmap vNEXT - Milestone 2026.05.288

**Title:** Abstention-Calibrated Verifier Recovery + Exact Fixtures + FR-11 Completeness Repair
**Created:** 2026-05-25
**Status:** Planned
**Supersedes:** 2026.05.287 "Verifier-Gain Recovery + Soundness-Bounded FR-11 + Blocker Reconciliation"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.287 Proved

Milestone `.287` completed its runnable work, but its authority artifacts still
do not support a paper-ready claim. The capstone authority is
`results/experiment_3080_capstone_v287.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=42`
- `verifier_gain_status=flagged_or_gated_verifier_gain_recovery_incomplete`
- `repair_claim_status=bounded_and_gated_skipped`
- `fr11_self_learning_status=flagged_controller_only_budget_exceeded`
- `gatemate_status=blocked_no_rerun_operator_actions_required`
- `ssqa_status=gated_skipped_host_visible_smoke_missing`
- `ebt_arm_status=projection_only_feasible_no_implementation`

The useful result is that `.287` made the failure modes explicit. The
first-token abstention panel used a mandated local SOTA GGUF and measured a
positive abstention-adjusted delta, but `abstention_precision=0.5` failed the
`>=0.7` gate. The formal correction pilot preserved exact solver fallback but
matched solver-only success, so it did not establish lift. The repair branch
correctly stayed blocked by gates. FR-11 showed controller-side rollback and
zero soundness mistakes, but one completeness mistake exceeded the budget. The
EBT/ARM audit stayed projection-only, and GateMate/SSQA remained bounded by
missing operator evidence.

| Area | `.287` result | `.288` consequence |
| --- | --- | --- |
| Verifier gain | Abstention precision was too low to unlock calibration | Build a larger exact fixture bank, add I-CALM/task-abstention prompts, and retest |
| Formal feedback | Solver fallback preserved but no measured lift | Add Dafny/Z3 structural feedback and vacuity checks before calibration |
| Repair | Structured repair protocol ready; micro-panel gate blocked | Add XGrammar/LLGuidance-style emitter preflight before a gated repair run |
| FR-11 | Soundness passed, completeness budget failed | Target completeness repair with ReSyn-style fixtures and KAN-CL anchoring |
| EBM bridge | EBT/ARM path feasible only as projection | Prototype sidecar schema and replay scorer without weight-update claims |
| Hardware | GateMate/SSQA still operator-evidence blocked | Ingest only host-visible evidence if it exists; no hardware rerun or speedup claim |
| Matrix/capstone | Matrix v21 ready but 42 blockers remain | Build a blocker-reduction ledger, matrix v22, and capstone with strict claim boundaries |

## Three Biggest Gaps To PRD Vision

1. **Verifiable reasoning evidence gap.** The PRD requires measured
   verification and repair gains under exact authority. Current evidence is
   negative or gated: low abstention precision, no formal-feedback lift, and
   skipped repair. `.288` must widen exact labels and separate solving,
   verifying, abstaining, and repairing.

2. **Continuous self-learning completeness gap.** FR-11 now has a controller
   loop with rollback, but it still failed a zero-completeness-mistake budget.
   `.288` must test whether synthetic exact families plus lightweight
   KAN-style anchoring improve retention and family holdout without soundness
   regressions.

3. **Architecture-claim bridge gap.** Carnot's long-term architecture points
   toward EBT/ARM-style energy refinement and hardware-accelerated sampling,
   but local code and hardware evidence do not justify those claims yet. `.288`
   must create adapter schemas and evidence-ingestion contracts while keeping
   implementation, timing, and speedup claims bounded.

## New Research Integrated

The post-`.287` planning sweep was appended to `research-references.md` before
this milestone was designed. Findings that shape `.288`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| I-CALM and task-abstention work separate confidence, humility, and abstention behavior in code generation | arXiv:2604.03904 and arXiv:2605.17029 | Redesign the abstention panel around calibrated reject/accept boundaries |
| LLM verification can be harder than solving | OpenReview `4jnJjSgQC1` | Add an autopsy protocol before trusting verifier gain |
| ReSyn-style synthetic environments can scale exact generators and verifiers | arXiv:2602.20117 / OpenReview `YcrOuJRVGh` | Build an exact fixture bank with solver/execution labels |
| Self-verification training helps reasoning only when verification is grounded | arXiv:2602.07594 | Keep self-verification under exact labels and perturbation tests |
| Dafny feedback and uDebug-style vacuity checks improve verified-code loops | arXiv:2604.22601 | Add structural formal feedback and vacuity guards |
| XGrammar-2 improves structured generation throughput and custom grammars | arXiv:2601.04426 / `mlc-ai/xgrammar` | Preflight structured repair emission before a repair micro-panel |
| KAN-CL and COOL-KAN show continual/on-device KAN adaptation paths | arXiv:2605.12306 / Springer 2026 | Try KAN-style anchors for FR-11 completeness and retention |
| Extropic THRML/XTR-0 and Logical Intelligence Aleph/Kona remain architecture signals | Extropic writing/GitHub and logicalintelligence.com | Keep hardware and Kona claims as bounded architecture context until local evidence exists |

## Architecture Snapshot

```text
                         +----------------------------------+
                         | Mandated local SOTA GGUF models  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B,   |
                         | Gemma-4-26B-A4B                 |
                         +----------------+-----------------+
                                          |
                                          v
+---------------------+      +----------------------------+      +----------------------+
| exp3083 verifier    | ---> | exp3084 ReSyn exact        | ---> | exp3085 I-CALM /    |
| hardness autopsy    |      | fixture bank               |      | task-abstention     |
+----------+----------+      +-------------+--------------+      +----------+-----------+
           |                               |                                |
           |                               v                                v
           |                 +----------------------------+      +----------------------+
           |                 | exp3086 Dafny/Z3 formal    | ---> | exp3087 gated       |
           |                 | feedback and vacuity guard |      | verifier calib v3   |
           |                 +----------------------------+      +----------+-----------+
           |                                                              |
           v                                                              v
+---------------------+      +----------------------------+      +----------------------+
| exp3088 XGrammar /  | ---> | exp3089 gated structured   | <--- | positive verifier   |
| LLGuidance emitter  |      | repair micro-panel         |      | gain required       |
+---------------------+      +----------------------------+      +----------------------+

+---------------------+      +----------------------------+      +----------------------+
| exp3084 exact       | ---> | exp3090 FR-11 ReSyn +      | ---> | controller-only or  |
| fixture families    |      | KAN-CL completeness repair |      | bounded promotion   |
+---------------------+      +----------------------------+      +----------------------+

+---------------------+      +----------------------------+      +----------------------+
| exp3091 EBT/ARM     | ---> | sidecar schema + replay    | ---> | no weight-update    |
| adapter prototype   |      | scorer tests               |      | implementation claim|
+---------------------+      +----------------------------+      +----------------------+

+---------------------+      +----------------------------+      +----------------------+
| exp3092 GateMate /  | ---> | operator evidence ledger   | ---> | no rerun unless     |
| SSQA evidence       |      | and allowed next actions   |      | evidence exists     |
+---------------------+      +----------------------------+      +----------------------+

                                          |
                                          v
                         +----------------------------------+
                         | exp3093 matrix v22               |
                         | exp3094 capstone .288            |
                         +----------------------------------+
```

## Phase Plan

### Phase A - Archive, Blockers, and Verifier-Hardness Grounding

Tasks: `exp3081`-`exp3083`

- Archive `.287` without modifying the active roadmap.
- Convert matrix v21's 42 blockers into a reduction ledger with explicit
  research, artifact-hygiene, bounded, gated, and operator-evidence buckets.
- Apply the "verification is harder than solving" warning to Carnot's local
  verifier failures before another calibration or repair attempt.

Exit condition: `.288` starts from an auditable blocker ledger and a stricter
verifier/abstention protocol.

### Phase B - Exact Fixtures, Abstention Calibration, and Formal Feedback

Tasks: `exp3084`-`exp3087`

- Build a ReSyn-inspired exact fixture bank with solver/execution labels across
  SAT/SMT/code-style constraints.
- Run an I-CALM/task-abstention local SOTA panel on at least one mandated GGUF.
- Add Dafny/Z3 feedback and vacuity guards, falling back to Z3-only only when
  Dafny is unavailable and recording that boundary.
- Run gated verifier calibration v3 only if abstention and formal-feedback
  gates pass.

Exit condition: Carnot either has positive, exact-grounded verifier gain or the
repair branch remains mechanically blocked.

### Phase C - Structured Repair and Continuous Self-Learning

Tasks: `exp3088`-`exp3090`

- Preflight a structured repair emitter using XGrammar/LLGuidance design
  patterns without claiming grammar decoding if local libraries are missing.
- Run a tiny structured repair panel only after verifier calibration is
  positive.
- Run the mandatory continuous self-learning experiment with ReSyn families,
  KAN-CL-inspired anchors, exact delayed-regression checks, and rollback.

Exit condition: repair is either cleanly improved under gates or skipped, and
FR-11 has an updated soundness/completeness budget with explicit retention.

### Phase D - Adapter/Hardware Boundaries, Matrix v22, and Capstone

Tasks: `exp3091`-`exp3094`

- Prototype an EBT/ARM sidecar adapter schema and replay scorer with tests,
  without claiming model integration or weight updates.
- Ingest GateMate/SSQA operator evidence only if host-visible artifacts exist;
  otherwise keep the no-rerun ledger.
- Build matrix v22 and capstone `.288` from the new artifacts.

Exit condition: matrix v22 is the authority for paper readiness, and the
capstone preserves every remaining blocker without inflated wording.

## Dependency Graph

```text
exp3081 archive .287
  |
  v
exp3082 publication blocker ledger
  |
  v
exp3083 verifier-hardness autopsy

exp3083
  |
  v
exp3084 ReSyn exact fixture bank
  |      |
  |      +--> exp3090 FR-11 ReSyn/KAN-CL completeness repair
  |
  v
exp3085 I-CALM/task-abstention SOTA panel
  |
  +------------------------+
                           v
exp3086 Dafny/Z3 feedback ---> exp3087 gated verifier calibration v3
                                      |
                                      v
exp3088 structured emitter ------> exp3089 gated structured repair

exp3091 EBT/ARM sidecar adapter schema prototype
exp3092 GateMate/SSQA operator evidence ingestion

exp3093 cross-corpus matrix v22
  ^
  | depends on all completed or gated-skipped .288 artifacts
  v
exp3094 capstone .288
```

## Hardware Requirements

- **Local SOTA GGUF inference:** `exp3085`, `exp3086`, `exp3087`, and
  `exp3089` need at least one mandated local GGUF available:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, or
  `unsloth/gemma-4-26B-A4B-it-GGUF`.
- **Exact solvers:** `exp3084`, `exp3086`, `exp3087`, `exp3089`, and `exp3090`
  need local exact labeling through existing validators, Z3, execution tests,
  or documented blocked status.
- **Dafny/Z3:** `exp3086` should use Dafny when installed; if Dafny is absent
  but Z3 exists, it may run a Z3-only pilot and set the boundary fields.
- **GPU:** live SOTA GGUF tasks should declare CUDA/GPU/model-cache
  preconditions. Legacy tiny models are smoke tests only.
- **GateMate/SSQA:** `exp3092` must not flash boards, rerun timing, or claim
  speedups. It may only ingest operator-provided pinout, command, transcript,
  and safety evidence already present in the repo.

## Required SOTA Model Policy

Any `.288` task that uses an LLM must include a `MODEL_SPECS` section naming at
least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as Qwen3.5-0.8B and gemma-4-E4B-it may appear only as
CPU smoke tests with `legacy_smoke_only_used=true`; they cannot satisfy
headline-result fields.

## Failed-Experiment Rerun Compliance

The roadmap explicitly includes `prior_failures` for tasks that continue scopes
from `.287` and earlier, including verifier calibration, formal feedback,
repair, FR-11 self-learning, EBT/ARM bridging, hardware evidence, matrix, and
capstone. Every prior entry includes `retire_if_same_verdict: true` so a repeated
negative result can be mechanically retired instead of carried forward.

No task reuses a retired experiment id. No task depends on a retired upstream
through `requires:`. KV260/GateMate/SSQA work avoids retired host-SD-card
preconditions and uses evidence-ingestion only.

## Acceptance Criteria

- `research-roadmap-next.yaml` validates with the roadmap schema, prior-failure
  validator, exclusion-manifest linter, and gate audit.
- Every live LLM task includes mandated `MODEL_SPECS`, `model_specs` artifact
  fields, precondition checks, prompt hashes, and `legacy_smoke_only_used`.
- Every gated task has a structured `gated_on` block matching upstream required
  artifact fields.
- At least one experiment targets continuous self-learning: `exp3090`.
- No task modifies `research-roadmap.yaml` or `scripts/research_conductor.py`.
- Capstone `.288` keeps `paper_ready=false` unless matrix v22 has no blocker
  rows and all headline claims have exact-grounded evidence.

## Out Of Scope

- No broad WOPR/gallery work.
- No GRPO/VPRM revival.
- No GateMate/SSQA flashing, timing, or speedup rerun without operator evidence.
- No EBT/ARM model integration or weight update claim before a tested adapter
  schema and replay scorer exist.
- No repair headline claim unless verifier calibration v3 passes its positive
  gain gate.
