# Research Roadmap vNEXT: Contract-Guarded Energy and Certified Self-Learning

**Milestone:** `2026.08.545`
**Date:** 2026-08-12
**Experiments:** Exp6323-Exp6336
**Phases:** 4
**Primary requirements:** FR7, FR11, FR12
**North-star constraint:** use exact observable contracts to guard real local-model work, then improve a versioned factor policy online with a certificate and rollback

## Milestone thesis

V544 closed the model-local representation branch. All three required GGUF
models failed the causal surface controls. Carnot must not retry hidden-state,
activation, prefix-state, or pooled representation scoring. V545 moves the
energy boundary to an observable restricted policy program. An exact compiler
turns each policy and contract into local factors. The energy is the number
and weight of unsatisfied contract clauses. A verified fallback handles every
rejected or invalid policy. The exact guard is an oracle and is reported as
such. The research question is whether bounded energy-guided candidate search
adds utility over raw generation, reject-only filtering, and fallback alone.

V544 also produced a positive same-domain learner. It versions factor-local
state, proposes updates from exact feedback, and rolls back safely. V545 adds
two missing release controls. Exact minimized counterexamples become candidate
factor changes. An anytime-valid certificate controls optional stopping,
restarts, and repeated release decisions. The GGUF weights remain frozen. The
protected final partition is opened once.

The ARC branch remains a no-solve branch. V544 proved that a target-licensed
route can run as a live shadow. It did not show that the route changes a legal
action. V545 first measures counterfactual influence on the agent's own
candidates. It then permits one default-off A/B test that can reorder only
candidates already produced by the live E3 policy. It cannot inspect game
source, hidden state, an offline breadth-first search, or a per-game adapter.
It cannot claim or register a level solve.

One hardware task is allowed. A dated 2026-08-11 receipt records a physical
GateMate power cycle. That receipt permits exactly one non-destructive
`openFPGALoader -c dirtyJtag --detect` command. The milestone permits no flash,
synthesis, place and route, or timing task. KV260 is terminal. PolarFire stays
opportunistic. No other hardware task is scheduled.

## What V544 proved

| Branch | V544 evidence | V545 consequence |
|---|---|---|
| Milestone boundary | Exp6322 closed the declared scientific branches. It also recorded a broad `pytest` exit 3 and determination-lint exit 1. | Exp6323 preserves both the scientific terminal states and the command failures. It does not rewrite them as success. |
| Source refresh | Exp6311 ended `complete_null` with zero accepted post-marker sources. | Exp6324 scans only after the V545 planner marker. A zero accepted delta is valid. |
| Model-local representation | Exp6312 set readiness to 0.0. Every required model failed at least one causal or shortcut control. Exp6314, Exp6315, and Exp6317 did not run. | Close the entire hidden/model-local state lane. No hidden states, activations, embeddings, prefix trajectories, text-energy scorers, or pooled rescue appear in V545. |
| Exact fixture | Exp6313 shipped an exact code-safety fixture and exact oracle. | Reuse its exactness patterns for a new bounded policy DSL and contract compiler. Do not treat an exact oracle as a learned verifier. |
| Continuous self-learning | Exp6318 and Exp6319 ended positive. They shipped versioned factor-local state, feedback-directed search, protected validation, and exact rollback. | Add exact counterexample proposals and anytime-valid release certificates on a fresh stream. Stay same-domain and keep model weights frozen. |
| Learning safety | Exp6320 passed safety attacks. It made no utility claim. | Preserve restart, poison, reversal, lineage, and rollback attacks. Add alpha-spending and certificate-reset attacks. |
| ARC live path | Exp6321 set readiness to 1.0 for a target-licensed live shadow. It used live-agent self-discovery, claimed no solve, and made no registry update. | Test causal action influence in a default-off sandbox. Do not solve a public game or reuse the retired provenance patch. |

## The three largest gaps to the PRD vision

### Gap 1: no useful live correctness boundary after the state lane closed

FR7 and FR12 require deterministic verification around real model behavior.
The learned model-state route is now closed. Carnot needs an observable object
that a model can propose and an exact verifier can check. V545 uses a bounded,
typed policy DSL over finite domains. The exact factor energy gives a clear
reason for rejection. A hash-pinned verified fallback keeps the system safe.

### Gap 2: the positive learner lacks a sequential release certificate

FR11 requires continuous improvement without silent regression. Exp6318 and
Exp6319 proved a bounded same-domain update mechanism. They did not prove that
repeated peeking, optional stopping, restarts, or many candidate releases keep
the declared error rate. V545 adds an anytime-valid release ledger, alpha
spending, immutable predecision receipts, retention tests, and byte-exact
rollback.

### Gap 3: the ARC shadow is reachable but has no measured action influence

The live ARC agent can compute the target-licensed route, but the route has not
caused a legal action difference. V545 measures influence before it measures
utility. A clean null result is valuable. It tells us that the route is present
but behaviorally inert. Any A/B test stays default-off and uses only the live
agent's own attempts and candidates.

## Research delta used by this roadmap

- Self-Evolving Agents with Anytime-Valid Certificates (`arXiv:2607.00871`)
  motivates a versioned learner, an explicit error budget, and release gates
  that remain valid under optional stopping. V545 keeps the foundation model
  frozen and treats the paper's limited empirical scale as a caveat.
- SEVerA (`arXiv:2603.25111`) motivates first-order output contracts,
  rejection sampling, and a verified fallback. V545 uses a smaller bounded DSL
  and exact local contracts. It does not copy a paper result into a Carnot
  claim.
- VASO (`arXiv:2606.05395`) motivates converting exact model-checker
  counterexamples into update proposals while the foundation model stays
  frozen. V545 accepts only minimized exact counterexamples.
- MARCH (`arXiv:2603.24579`) motivates information asymmetry between a solver
  and a blind checker. The checker receives the canonical contract, normalized
  candidate semantics, and exact evidence. It never receives the solver's
  rationale or claimed verdict.
- Loss Smoothing for Continual Adaptation (OpenReview `pUqcOkV69j`) motivates a
  stability control. It is a baseline, not the release authority.
- Optimal KAN abstractions, energy-guided text sampling, pairwise text
  verifiers, and external learned energy remain deferred. They match retired
  KAN, masked-model, best-of-N text scorer, or external-scorer lanes.

The full source disposition is in `research-references.md` under the V545
planner marker.

## Target architecture

```text
finite task + canonical contract
              │
              ▼
     restricted typed policy DSL ◄──── local GGUF candidate generator
              │                         Qwen3.6 / Gemma-31B / Gemma-26B
              ▼
      parser + normalized semantics
              │
       ┌──────┴───────────────────┐
       ▼                          ▼
 exact clause factors       blind integrity checker
 E = weighted violations    contract + semantics + evidence only
       │                          │
       └──────────┬───────────────┘
                  ▼
        accept policy or invoke
        hash-pinned verified fallback
                  │
        held-family prospective A/B

continuous self-learning:
fresh event ─► immutable predecision ─► exact outcome/counterexample
    │                                         │
    │                              minimized factor proposal
    │                                         │
    └──── frozen champion ◄── candidate version + parent hash
                                  │
                       anytime-valid release ledger
                       alpha spend + retention gate
                                  │
                         next-boundary release
                                  │
                        monitor ─► exact rollback

ARC no-solve branch:
live E3 own attempts ─► existing candidate set ─► default-off route reorder
        │                                              │
        └──── control action unchanged ─► legal influence A/B ─► no solve
```

## Phase 0: exact transition, source freeze, and hardware continuity

### Exp6323: V544-to-V545 terminal transition

Consume Exp6322 and every V544 artifact. Preserve the failed broad validation
commands. Validate 14 task IDs, deliverables, dependencies, structured gates,
prior-failure entries, Codex routing, model policy, prompt endings, and
protected files. Do not activate the staged roadmap.

**Deliverable:** `results/experiment_6323_v545_terminal_transition.json`

### Exp6324: post-marker source and scope freeze

Search only after the V545 reference marker. Freeze the restricted-policy,
exact-contract, blind-checker, anytime-certificate, counterexample-update, ARC
no-solve, and hardware contracts. A zero accepted source delta is terminal.

**Deliverable:** `results/experiment_6324_v545_post_marker_source_scope_freeze.json`

### Exp6325: GateMate dated-receipt single detect

Validate the 2026-08-11 physical power-cycle receipt. Run exactly one
non-destructive DirtyJTAG detect. Record stdout, stderr, exit code, USB/JTAG
identity, and before/after state. Stop after the detect for every outcome.

**Deliverable:** `results/experiment_6325_gatemate_dated_receipt_single_detect.json`

## Phase 1: exact contract-guarded policy energy

### Exp6326: restricted policy DSL and exact contract compiler

Build a bounded typed policy language for finite state-action tasks. Normalize
every program. Compile the canonical contract into exact local factors. Check
the full finite domain with enumeration or Z3. Ship a hash-pinned verified
fallback and adversarial fixtures for vacuous contracts, parser defaults,
fallback laundering, validator mutation, and hash swaps.

**Deliverable:** `results/experiment_6326_restricted_policy_contract_compiler.json`

### Exp6327: three-family guarded policy synthesis, gated on Exp6326 readiness=1

Use all three required local GGUF models to propose restricted policies. Keep
calls and tokens matched. Compare raw single generation, reject-only filtering,
guard plus fallback, and bounded exact-factor-energy candidate search. The
exact guard is an oracle. Utility is held exact task reward after all fallback
costs.

**Deliverable:** `results/experiment_6327_three_family_guarded_policy_synthesis.json`

### Exp6328: blind-obligation integrity audit

Rebuild contract and policy evidence independently. The checker receives no
solver narrative or claimed label. Attack vacuous specifications, parser
defaults, fallback laundering, spec or validator mutation, test deletion,
hash swaps, label swaps, pair swaps, evaluator swaps, and budget mismatches.
Safety-only success cannot promote utility.

**Deliverable:** `results/experiment_6328_blind_guard_integrity_audit.json`

### Exp6329: prospective held-family guarded-policy A/B, gated on Exp6328 integrity=1

Seal new contract families before generation. Run all three local GGUF models.
Compare the four preregistered arms with matched budgets. Report every model
and family separately. Headline value requires higher held exact utility than
guard plus fallback, no contract violation, and no failed required command.

**Deliverable:** `results/experiment_6329_prospective_held_family_guarded_policy_ab.json`

## Phase 2: certified continuous self-learning

### Exp6330: anytime-valid release certificate engine

Add an independent sequential release ledger to the versioned factor policy.
Test null streams, optional stopping, repeated candidates, alpha spending,
restart identity, retention, degradation, and exact rollback. This task does
not need an LLM and can run even if Phase 1 is null.

**Deliverable:** `results/experiment_6330_anytime_valid_release_certificate_engine.json`

### Exp6331: exact counterexample-to-factor update calibration

Use all three local GGUF models on a development stream. For rejected bounded
policies, minimize exact counterexamples and convert them into candidate factor
changes. Compare counterexample proposals with repeated sampling and a smoothed
update control under matched budgets. Do not expose protected validation.

**Deliverable:** `results/experiment_6331_counterexample_factor_update_calibration.json`

### Exp6332: prospective certified continuous self-learning A/B, gated on Exp6330 and Exp6331 readiness=1

Run a fresh sealed chronological stream. Compare a frozen champion, the fixed
Exp6318/Exp6319 learner, and the counterexample-guided learner with the
anytime-valid release certificate. Persist predictions before outcomes. Open
the final protected partition once. Measure future utility, retention, release
rate, false release, cost, and rollback. Keep GGUF weights frozen.

**Deliverable:** `results/experiment_6332_prospective_certified_continuous_learning_ab.json`

### Exp6333: independent certificate and learning safety audit

Audit the certificate engine and the calibrated proposer. Inspect Exp6332 if
it ran. Attack optional stopping, alpha reset, restart, duplicate evidence,
future leakage, protected-set reuse, counterexample fabrication, factor
misattribution, lineage swaps, poison, reversal, forgetting, and rollback.
This is a safety audit and cannot promote utility.

**Deliverable:** `results/experiment_6333_certified_learning_safety_audit.json`

## Phase 3: live ARC influence and capstone

### Exp6334: ARC counterfactual action-influence preflight

Start from Exp6321's target-licensed live shadow. Use only the live agent's own
attempts and already-generated candidates. Measure whether a legal route score
can cause a nontrivial candidate ordering change above the A/A noise floor.
Make no shipped action change. Claim no solve and update no registry entry.

**Deliverable:** `results/experiment_6334_arc_counterfactual_action_influence_preflight.json`

### Exp6335: default-off live E3 causal-influence A/B, gated on Exp6334 eligibility=1

Run a prospective control/shadow A/B on fresh live-agent windows. The shadow
may reorder only existing E3 candidates. It cannot create actions, inspect
hidden state or game source, run an offline solver, or use a per-game adapter.
Measure legal action support and first-useful-action efficiency. Do not claim a
level solve. Do not update the solve registry.

**Deliverable:** `results/experiment_6335_arc_default_off_live_causal_influence_ab.json`

### Exp6336: V545 adversarial capstone and reconciliation

Classify every branch exactly. Recheck retired-scope discipline, contract and
certificate hashes, model receipts, ARC provenance, hardware command count,
protected files, tests, and documentation. Update OpenSpec, traceability,
status, changelog, conductor log, and the next architecture state. Do not hide
null, skipped, blocked, flagged, or failed commands.

**Deliverable:** `results/experiment_6336_v545_adversarial_capstone.json`

## Dependency graph

```text
Exp6323 transition
  ├── Exp6324 source/scope freeze
  │     ├── Exp6326 contract compiler
  │     │     └── Exp6327 guarded synthesis [6326 ready]
  │     │             └── Exp6328 blind integrity audit
  │     │                     └── Exp6329 prospective A/B [6328 ready]
  │     ├── Exp6330 anytime certificate
  │     ├── Exp6331 counterexample calibration
  │     │     └── Exp6332 certified CSL A/B [6330 ready AND 6331 ready]
  │     └── Exp6334 ARC influence preflight
  │             └── Exp6335 live influence A/B [6334 eligible]
  ├── Exp6325 GateMate single detect
  └── Exp6336 capstone depends on every declared task

Exp6333 safety audit depends on Exp6330 and Exp6331.
It inspects Exp6332 if the gated task ran.
```

## Hardware and model requirements

| Resource | Requirement | Fail-closed behavior |
|---|---|---|
| Local models | `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and `unsloth/gemma-4-26B-A4B-it-GGUF` in every LLM experiment | A missing required model blocks that declared model cell. No legacy model may replace a headline cell. |
| Tokenizer and runtime | Canonical llama.cpp GGUF path with each file's embedded tokenizer | Do not use Hugging Face `AutoTokenizer`. Record file, revision, quantization, tokenizer, placement, and memory receipts. |
| GPUs | Two local NVIDIA GPUs for bounded sequential model cells | Load one declared model placement at a time when needed. Prove memory release. CPU tiny-model runs are smoke tests only. |
| GateMate | Dated 2026-08-11 power-cycle receipt and one DirtyJTAG detect | Run exactly one `openFPGALoader -c dirtyJtag --detect`. Stop after the command. No flash, synthesis, place and route, or timing. |
| KV260 | None | Terminal. Do not schedule a probe or bring-up task. |
| PolarFire | None | Opportunistic only. Do not make it a milestone dependency. |
| Extropic or other thermodynamic hardware | None | No authenticated local substrate exists. Software results cannot become hardware claims. |

## Promotion rules

- The exact contract guard can promote safety only. It is an oracle.
- Guarded synthesis promotes utility only on fresh held contract families after
  all fallback costs, with every required model cell reported separately.
- No model or family mean can rescue a failed disaggregated cell.
- The continuous learner promotes only with immutable chronology, a passing
  anytime-valid certificate, future utility, retention, and exact rollback.
- The protected final partition is opened once. It never guides an update.
- ARC promotion means legal causal action influence only. It does not mean a
  game or level solve.
- A missing artifact, failed required command, provenance violation, protected
  file change, unapproved hardware action, or retired-scope recurrence blocks
  promotion.

## Explicit exclusions

- No hidden-state, activation, embedding, prefix-state, shared-bus, or pooled
  model-local representation retry.
- No external generated-text scorer, masked-model energy, best-of-N verifier,
  or teacher-label side channel.
- No KAN experiment.
- No natural-language ConstraintIR reprompt or finite-ID answer transport.
- No cross-family, cross-domain, or cross-game transfer.
- No GGUF weight update.
- No public ARC game re-solve, hidden source access, exhaustive offline BFS,
  hand-built game adapter, outer-loop reverse engineering, or solve-registry
  mutation.
- No hardware command beyond the one authorized GateMate detect.

## Expected milestone outcomes

V545 can end positively in parts. The contract compiler and certificate engine
are valuable exact infrastructure even if model utility is null. A clean null
guarded-policy result closes one observable candidate-search strategy. A clean
null ARC influence result proves that the shadow is reachable but inert. The
milestone succeeds operationally when every branch is classified honestly and
the next roadmap preserves those boundaries.
