# Research Roadmap vNEXT - Milestone 2026.07.493

**Milestone title:** Risk-Calibrated Structured Verification, Evidence-Stable Self-Learning, ARC Level-Up, and Hardware Timing Receipts

**Planner date:** 2026-07-08
**Previous milestone:** 2026.07.492
**Task range:** Exp 5415-5427
**Pre-staged roadmap:** `research-roadmap-next.yaml`

## Inputs Read

Required repository inputs were read before planning:

1. `research-program.md`
2. `_bmad/prd.md`
3. `_bmad/architecture.md`
4. `ops/status.md`
5. `ops/changelog.md`
6. `research-complete.yaml`
7. `research-roadmap.yaml`
8. `openspec/change-proposals/`
9. `ops/conductor-log.md`
10. `research-references.md`
11. `research-hardware-wishlist.md`

Additional guardrails checked before writing the roadmap:

- `CLAUDE.md`
- `CODEX.md`
- `ops/exclusion_manifest.yaml`
- `ops/arc_solve_registry.yaml`
- `ops/known-issues.md`
- `scripts/experiment_template.py`
- `scripts/roadmap_schema.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `.492` result artifacts under `results/`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview,
Extropic writing, Semantic Scholar citation routes, HuggingFace Papers,
GitHub, and Logical Intelligence public pages before designing experiments.
Actionable findings were appended to `research-references.md` under
`### V493 Planner Refresh - 2026-07-08`.

Promoted sources and planning consequences:

- **Constrained Flow Matching via Lagrangian Dual Flows** (`arXiv:2607.04513`):
  motivates dual-residual and nonlinear-constraint diagnostics for active
  constraints and p-bit transfer, while keeping Carnot's deterministic solver
  as final authority.
- **Neuro-Symbolic Safety Guidance for VLA Models via Constrained Flow
  Matching** (`arXiv:2607.01378`): motivates predictive prefix/tool-action
  safety checks before final-state verification, without treating learned
  guidance as an oracle.
- **Uncertainty-Aware Abstention in LLMs with Provable Alignment Guarantees**
  (`arXiv:2607.04430`): motivates confidence-interval abstention and accepted
  risk accounting for structured local SOTA panels and learned-memory
  promotion.
- **Weave of Formal Thought** (`arXiv:2606.25987`): motivates prefix-valid
  grammar/structure constraints and latent-symbol traces, but only as a
  verifier-facing diagnostic because Carnot does not yet have authenticated
  logits/hidden-state receipts.
- **Empirical Study for Structured Output Control in LLMs for Software
  Engineering** (`arXiv:2606.09395`): warns that schema/syntax validity is not
  enough. `.493` therefore measures semantic, policy, reachability, and
  abstention errors separately.
- **Measurement-Access Risk Frontiers for Autonomous Scientific Control**
  (`arXiv:2607.05696`): motivates hardware timing receipts and measurement
  access checks. Missing board records cannot be repaired by compute alone.
- **Hidden Forgetting in Continual Multimodal Learning** (`arXiv:2607.02020`):
  motivates evidence-reliance drift tests for continuous self-learning, not
  just task-score retention.
- **Beyond the Leaderboard: A Taxonomy of Failure Modes in LLM Tool-Use,
  Planning, and Reasoning** (`arXiv:2607.05775`): motivates the `.493`
  PRD-gap/failure table taxonomy.

Secondary-source status:

- OpenReview and HuggingFace Papers reinforced constrained decoding,
  interactive verification, and verifier-first evaluation; no secondary source
  added a stronger local baseline than the arXiv items above.
- Semantic Scholar citation routes for EBT `2507.02092` and ARM-EBM
  `2512.15605` were checked; the API route was rate-limited during planning,
  so no new citation-derived experiment is promoted.
- GitHub search surfaced useful watchlist repos for constrained decoding and
  ARC exploration, but no repo was adopted as a milestone dependency.
- Extropic TSU/XTR and Logical Intelligence Kona/Aleph remain architecture
  context only. Carnot has no executable local TSU, Kona, or Aleph baseline for
  `.493`.

## What 2026.07.492 Proved

The `.492` result artifacts show several lanes are now usable, while live
evidence remains the limiting factor:

- **Formal corrigendum cleaned the prior tautology failure.** Exp5404 produced
  `formal_encoding_corrigendum_clean=true` with row-level checksums,
  deterministic policy authority, local SOTA GGUF inference, and GPU/offload
  receipts.
- **Structured safety/action verification is headline-ready at bounded scale.**
  Exp5405 produced `structured_safety_action_panel_ready=true` with the
  mandated SOTA GGUF model specs and deterministic schema, semantic, policy,
  and reachability authority.
- **Active-constraint guidance now has a safe solver-authority interface.**
  Exp5406 showed a small but real solver-work reduction under deterministic
  overwrite authority.
- **P-bit/QUBO stress remains CPU-only but valid.** Exp5407 showed p-bit
  diagnostics can match exact enumeration and stress active-constraint hints,
  but no hardware speedup or hardware transfer claim is available.
- **Continuous self-learning has resource and uncertainty controls.** Exp5408
  and Exp5409 produced resource-accounted routing, raw episode retention,
  rollback, no model-weight mutation, and uncertainty-gated promotion.
- **ARC still did not bank a new level.** Exp5410 was an honest no-bank on
  `re86` L3 through live-agent self-discovery. It avoided offline BFS and
  per-game adapters, but did not increase reproducible levels.
- **Hardware repeatability improved but speedup evidence is absent.** Exp5411
  restored repeated same-workload PolarFire receipts and kept KV260/GateMate
  limitations explicit. It made no speedup claim.
- **KAN certificates remain bounded but useful.** Exp5412 emitted a clean
  active-constraint certificate family with false-property rejection and
  true-property preservation.
- **Evidence table and capstone are explicit about closed, partial, and blocked
  lanes.** Exp5413/Exp5414 classify formal/structured, resource-CSL, and
  uncertainty-gated promotion as headline-ready; active-constraint, p-bit, KAN,
  ARC, hardware, and token/internal evidence remain bounded or blocked.

## Three Biggest Gaps

1. **Structured verification is not yet risk-calibrated.** `.492` proved clean
   row-level structured verification, but PRD FR-12 needs decisions that know
   when to abstain. `.493` adds confidence-interval accepted-risk accounting,
   prefix/tool-action safety, and semantic-error separation so schema success
   is not mistaken for verified reasoning.

2. **Continuous self-learning is not yet evidence-stable.** `.492` learned
   resource-aware routing and uncertainty gates without weight mutation, but
   FR-11 needs durable learning that does not silently shift evidence
   reliance. `.493` adds hidden-forgetting/evidence-reliance drift probes and
   promotion gates that must preserve grounding, rollback, and raw episodes.

3. **Live evidence still lags the PRD vision.** ARC has repeated no-bank
   attempts, hardware lacks comparable CPU-vs-board timing receipts, and
   token/internal feature lanes remain closed. `.493` targets a different ARC
   live-path mechanism, authenticated hardware timing, and bounded certificate
   expansion without claiming unsupported hardware speedup or hidden-state
   access.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via GGUF runtime  |
                         +-------------------+------------------+
                                             |
                             structured candidates + risk receipts
                                             |
        +------------------------------------v-----------------------------------+
        | Risk-calibrated structured verifier layer                              |
        | schema, semantic, policy, reachability, prefix safety, abstention      |
        | deterministic verifier is final authority                              |
        +---------------+------------------------------+-------------------------+
                        |                              |
        active constraints + dual residuals            | resource and uncertainty gates
                        |                              |
        +---------------v--------------+   +-----------v------------------------+
        | Solver and p-bit diagnostics |   | Evidence-stable CSL controller     |
        | solver accepts/rejects/       |   | raw episodes, influence shares,    |
        | overwrites hints; CPU first   |   | reliance drift, rollback           |
        +---------------+--------------+   +-----------+------------------------+
                        |                              |
                        +--------------+---------------+
                                       |
        +------------------------------v----------------------------------------+
        | Live evidence surfaces                                                 |
        | ARC self-discovery, board timing receipts, KAN measurement-access      |
        | certificates; token/internal lanes stay closed                         |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition, Source Delta, and Risk-Calibrated Verification

- **Exp5415:** archive `.492`, stage `.493`, and record the exact closed,
  partial, and blocked lanes from `.492` artifacts.
- **Exp5416:** run an execution-time source delta so the roadmap can absorb
  any source changes after this planner refresh without reopening retired
  scopes.
- **Exp5417:** scale the clean `.492` structured safety/action panel into a
  risk-calibrated local SOTA panel with confidence intervals, accepted-risk
  accounting, and abstention.
- **Exp5418:** if Exp5417 is ready, add predictive prefix/tool-action safety
  checks inspired by constrained flow matching and neuro-symbolic safety
  guidance.

### Phase 1 - Active-Constraint Scale and P-Bit Transfer Preconditions

- **Exp5419:** scale active-constraint guidance to larger solver panels with
  LNS-style subproblem hints and dual-residual diagnostics under solver
  authority.
- **Exp5420:** if Exp5419 is clean, run a p-bit hardware-transfer preflight
  that preserves exact-enumeration validity and refuses speedup claims without
  comparable board timing receipts.

### Phase 2 - Evidence-Stable Continuous Self-Learning

- **Exp5421:** implement the required continuous self-learning experiment for
  `.493`: evidence-reliance drift and hidden-forgetting diagnostics under the
  existing no-weight-mutation controller.
- **Exp5422:** if Exp5421 is ready, expand uncertainty-gated promotion so
  learned fragments must preserve risk, grounding, resource, rollback, and
  evidence-reliance thresholds before influencing routing.

### Phase 3 - Live Evidence, Certificates, and Synthesis

- **Exp5423:** attempt an ARC live-path level-up with a different mechanism
  from repeated `re86` salience attempts: CoEx-style frontier persistence,
  hierarchical landmarks, measurement-access receipts, and live-agent
  self-discovery only.
- **Exp5424:** collect comparable same-workload CPU and PolarFire timing
  receipts, use KV260 SSH-only diagnostics, and keep GateMate diagnostic-only
  unless physical/JTAG evidence returns.
- **Exp5425:** expand the bounded KAN certificate family around measurement
  access and active-constraint false-property controls.
- **Exp5426:** aggregate `.493` evidence against PRD gaps using the tool-use,
  planning, and reasoning failure taxonomy.
- **Exp5427:** emit the `.493` capstone truth table and recommendations for
  the next milestone.

## Dependency Graph

```text
exp5415 transition
  -> exp5416 source delta
  -> exp5417 risk-calibrated SOTA structured panel
      -> exp5418 predictive prefix/action safety

Exp5406/Exp5407 prior solver evidence
  -> exp5419 active-constraint LNS scale
      -> exp5420 p-bit hardware-transfer preflight

Exp5408/Exp5409 prior CSL evidence
  -> exp5421 evidence-reliance CSL
      -> exp5422 gated CSL promotion scale

Exp5410 no-bank ARC attempt + arc_solve_registry
  -> exp5423 live ARC CoEx/landmark level-up attempt

Exp5411 hardware repeatability
  -> exp5424 comparable timing receipts

Exp5412 KAN active-constraint certificate
  -> exp5425 measurement-access KAN certificate

exp5417, exp5418, exp5419, exp5420, exp5421, exp5422, exp5423, exp5424, exp5425
  -> exp5426 PRD gap/failure table
  -> exp5427 capstone
```

## Hardware Requirements

- **Local SOTA GGUF inference:** Exp5417 and Exp5418 require authenticated
  local GGUF runtime evidence for at least one mandated SOTA model in headline
  rows and must include all three model specs in artifacts:
  `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy small models are smoke-test only.
- **CUDA/offload:** Any SOTA GGUF headline row must record runtime, command,
  model path, offload settings, memory/offload evidence, duration, and failure
  fallback. CPU-only smoke rows cannot support headline model claims.
- **ARC:** Exp5423 must use live-agent self-discovery from runtime attempts and
  runtime reverse engineering. It must not use offline BFS, source inspection,
  or per-game adapters as the credited solve path.
- **PolarFire:** Exp5420 and Exp5424 may use the currently reachable PolarFire
  path. Exp5424 must compare the same workload against CPU with hashes, repeat
  counts, and timing receipts before any speedup-adjacent interpretation.
- **KV260:** KV260 checks are SSH-only using `ssh -o ConnectTimeout=5 -o
  BatchMode=yes kria true`. Do not use host SD-card or block-device probes.
- **GateMate:** GateMate remains diagnostic-only unless the physical/JTAG path
  is restored during the experiment. No task may assume it is available.
- **Extropic TSU, Kona, Aleph:** watch-only architecture references for
  `.493`; no local executable baseline exists.

## No-Go Boundaries

- Do not reopen the retired external generated-text scorer, token-internal,
  LoRA/uPRM/EBRM, or hidden-feature lanes without a new authenticated backend
  artifact.
- Do not claim hardware speedup from CPU-only p-bit, unmatched board receipts,
  or one-off timings.
- Do not re-solve already-banked ARC levels as headline evidence. The ARC task
  must target a new reproducible level and report `solve_provenance:
  live_agent_self_discovery`.
- Do not modify `research-roadmap.yaml` or `scripts/research_conductor.py`
  during planning.
- Do not treat schema validity as semantic correctness. `.493` must separate
  syntax/schema, semantic, policy, reachability, uncertainty, and abstention
  outcomes.

## Expected Evidence at Milestone End

The milestone should end with one of two honest outcomes:

- **Headline-positive path:** risk-calibrated structured verification remains
  clean, prefix/action safety improves over final-only checks, evidence-stable
  CSL preserves grounding while reducing waste, ARC banks at least one new live
  level, and hardware emits comparable timing receipts without overstated
  speedup.
- **Bounded/null path:** any blocked GPU, ARC, or hardware lane emits a
  precondition-checked artifact with no headline claim; the capstone records
  exactly which PRD gaps remain and why.
