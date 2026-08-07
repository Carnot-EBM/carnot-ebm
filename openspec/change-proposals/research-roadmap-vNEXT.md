# Research Roadmap vNEXT — Transport-Qualified Verification and Prospective Memory

**Created:** 2026-08-07

**Target milestone:** `2026.08.537`

**Status:** Planned after terminal milestone `2026.08.536`

**Experiment range:** Exp6197-Exp6210 (14 tasks, four phases)

**Primary question:** Can Carnot turn V536's two zero-correctness local-SOTA
streams into transport-qualified executable evidence, finally run the
oracle-distinct code-selector and prospective self-learning tests, and carry
the two genuine V536 positives into safe runtime/generalization boundaries?

**Informed by:** `research-program.md`, `_bmad/prd.md`,
`_bmad/architecture.md` (last reconciled 2026-07-03; stale by the project's
30-day freshness rule), `ops/status.md`, `ops/changelog.md`,
`research-complete.yaml`, `research-roadmap.yaml`, all prior roadmap proposals,
`ops/conductor-log.md`, `research-references.md` through the
`V537-PLANNER-REFRESH-20260807-END` marker,
`research-hardware-wishlist.md`, `ops/known-issues.md`,
`ops/exclusion_manifest.yaml`, and the exact Exp6183-Exp6196 artifacts.

## What milestone 2026.08.536 proved

| Branch | Terminal evidence | Consequence for V537 |
|---|---|---|
| Transition and closure | The conductor recorded Exp6183 and Exp6196 as completed, but their declared artifacts remain `blocked: bootstrap only` with `duration_s=0.0`; neither performed its promised reconciliation. | Add a fail-closed terminal-state classifier before another transition/capstone. A conductor receipt cannot promote `running`, `running_bootstrap`, or bootstrap-only artifacts. |
| Frozen code bank | Exp6186 froze 120 unique cached LiveCodeBench tasks across calibration, held, CSL-seed, and CSL-prospective roles with a private-test boundary. | Reuse this immutable bank. Do not reselect tasks or expose private tests. |
| Authentic code generation | Exp6187 retained 576 raw Gemma-4-31B samples over 72 tasks, but used an answer budget of 128 tokens: 510 rows had no code block, 513 were truncated, no candidate was correct, `pool_integrity_ready_score=0`, and the artifact was quarantined by `DURATION_TOO_SHORT`. | The next scientific action is a small transport canary with adequate bounded budgets and real timing, not another full pool. Only a passing canary may launch a new K=8 collection. |
| Phase-D selector | Exp6188 was gate-blocked and Exp6189-Exp6191 never produced scientific selector evidence. | The hidden-state hypothesis remains untested on executable code. Preserve the pool -> headroom -> surface -> calibration freeze -> one-shot held chain. |
| Continuous self-learning | Exp6192 sealed 108 live Qwen3.6/Gemma-4-26B strategy generations, but outcomes were 86 syntax, 3 compile, 19 runtime, and zero passes; `seed_stream_ready_score=0`. Exp6193 was therefore gate-blocked. | Reuse the transport canary for both families, rebuild the seed only if the raw-code envelope works, then run the chronological A/B. Do not call zero-pass memory a learning result. |
| Stochastic substrate | Exp6194 achieved exact short-chain and distributional Python/Rust/PyO3 parity with `mode_jump_rust_pyo3_ready_score=1.0`, while making no hardware or speed claim. | Integrate the fixed kernel behind the existing runtime sampler boundary, default off, with matched quality and regression receipts. Do not rerun parity as research. |
| ARC generalization | Exp6195 collected 48 fresh live-agent-owned transitions. Its frozen task-aware policy beat the global policy by `0.208333` decision accuracy with a positive interval, without solve or registry credit. | Spend exactly one ARC slot on leave-one-game-out shadow measurement with adapters disabled. Do not reopen the closed induction/refinement axis or target a level solve. |

V536 therefore supplied two real positives (mode-jump parity and fresh ARC
policy generalization), one reusable bank, and two independently reproduced
generation-transport failures. It did **not** test an internal-state selector
or continuous live improvement.

## The three largest gaps to the PRD vision

### Gap 1 — evidence terminality is not mechanically trustworthy

The PRD's autonomous-research loop assumes completed tasks leave auditable,
terminal evidence. V536 demonstrates a direct mismatch: the conductor's
completion cache accepted two bootstrap-only artifacts as done. Until a shared
terminal classifier rejects nonterminal status/verdict combinations, future
capstones can launder work that never ran. This is an operational prerequisite
for every scientific branch, not housekeeping.

### Gap 2 — Carnot still lacks an oracle-distinct verifier win on competent local outputs

FR-12 requires a verifier that improves selection, not merely an exact oracle
that labels failures. V536's pool failed before competence or headroom could be
measured because the generation budget truncated nearly every response. The
missing evidence is now precise: a transport-qualified local flagship K=8 code
pool, both correctness classes, genuine oracle headroom over a tuned label-free
equivalence baseline, then a task-disjoint held internal-state selection test.

### Gap 3 — autonomous self-learning has no positive live event stream

FR-11 and `research-program.md` require verified experience to change future
decisions without forgetting. V536 exercised the transaction machinery but
produced zero correct live events, so the prospective A/B never ran. Carnot
needs a stream with real positive and negative outcomes, immutable predecision
memory snapshots, post-outcome commits, procedural-memory retrieval, explicit
negative-transfer measurement, and unchanged model weights.

The V536 positives expose a smaller integration gap: the sampler parity result
is not yet in the runtime boundary, and the ARC policy has not been tested
leave-one-game-out. Those receive one bounded task each; they do not displace
the Phase-D majority.

## Research findings incorporated

| 2025-2026 source | Finding used | V537 response |
|---|---|---|
| *On LLMs' Internal Representation of Code Correctness* (arXiv:2512.07404) and *Code Correctness Is Linearly Decodable...* (arXiv:2606.14530 v3) | Same-task correct/incorrect code can carry linearly decodable internal signal, but length/surface leakage and task splits matter. | Run prompt-final and code-final features only after real executable headroom; residualize length/surface controls and freeze on calibration before the held-label join. |
| WybeCoder (arXiv:2603.29088) | Code, invariants, and proofs can co-evolve under hybrid SMT/Lean feedback, and verified imperative tasks remain difficult. | Preserve exact execution as the current label oracle; defer proof/invariant co-generation until the simpler raw-code transport and selector are viable. |
| RepoZero (arXiv:2605.07122) | Sandboxed black-box equivalence yields scalable executable labels; iterative test generation supports later test-time scaling. | Keep private tests hidden and fixed for V537. Generated-test actions are deferred so they cannot leak into selector features or change the held oracle. |
| AgentCL (arXiv:2606.02461) | Controlled compositional streams distinguish memory transfer from naive replay, while held settings expose memory-induced degradation. | Use chronological task-family blocks, measure forward transfer and negative transfer separately, and retain a fixed no-memory arm. |
| *When Continual Learning Moves to Memory* (arXiv:2604.27003) | Procedural memories transfer better than raw trajectories; retrieval can relocate rather than solve forgetting. | Store bounded procedural summaries plus exact outcome provenance, compare against no memory, and report hard-case/family retention. |
| MemoPilot (arXiv:2606.08656, ICML 2026) and Memoir (arXiv:2607.20792) | Memory updates benefit from explicit multi-turn credit, but same-pass writes can hurt finite-budget learning. | Keep decision-time snapshots read-only and commit only after both arms are labeled. No weight update or same-pass write is allowed. |
| LLM-as-a-Verifier (arXiv:2607.05391) and SEVRA (arXiv:2606.19808) | Verification is an allocation axis; continuous scoring and selective intervention need tuned cost and harmful-flip controls. | Compare the internal selector with tuned label-free baselines and report harmful selections/headroom recovered, not accuracy alone. Do not reopen external logit/text scorers. |
| Thermalizing Stochastic Programs (arXiv:2608.01615) and *Scaling Up Thermodynamic AI Models* (arXiv:2607.00170) | Sampling deployments need explicit factor, autocorrelation, schedule, and error accounting. | Runtime mode-jump integration retains quality, ESS/autocorrelation, serialization, and fallback receipts; no TSU/FPGA speed claim. |
| Extropic, Kona, OpenReview, Hugging Face, GitHub, and EBT/ARM-EBM citation checks | No new authenticated TSU, reproducible Kona route, KAN replacement, or citation-trail method removes the current local transport prerequisite. | Keep TSU/Kona/KAN execution deferred. Use the one nonterminal board slot for a cached GateMate action audit only. |

The full dated discovery and duplicate-suppression record is in
`research-references.md` under the V537 planner marker.

## Target architecture

```text
                   fail-closed terminal artifact classifier
                                   │
              ┌────────────────────┼─────────────────────┐
              │                    │                     │
       Phase-D code path      continuous FR-11      positive-mechanism carry
              │                    │                     │
   immutable Exp6186 bank      same transport        Exp6194 fixed sampler
              │                 canary gate           + Exp6195 frozen policy
   3-family raw-code envelope       │                     │
   canary: finish/extract/compile   │              runtime default-OFF adapter
   sample-run; no private labels    │              + ARC LOO shadow measurement
              │                    │
    Gemma-4-31B K=8 pool      two-family strategy seed
    raw bytes before labels    positive + negative events
              │                    │
 exact private execution labels    │
              │              immutable snapshot
 competence + headroom         choose -> generate
              │              -> verify -> delayed commit
 matching Gemma-4-31B base          │
 hidden-state surface         procedural memory vs no-memory
              │
 calibration-only selector freeze
              │
 one-shot held selection

 GateMate cached receipt/action audit (no JTAG without changed-state receipt)

 All terminal branches ───────────────────────────────→ exact-path capstone
```

The canary deliberately chooses an output envelope using only serving-visible
transport and public sample-run signals: finish reason, raw bytes, code
extraction, compilation, and public sample execution. Private-test correctness
must not select token budget, prompt envelope, model, or retry policy. The
full-pool task freezes the selected configuration before generation and stores
every response before extraction or labeling.

## Phase 0 — Evidence, dated ingress, and hardware continuity (Exp6197-Exp6199)

### Exp6197: fail-closed terminal-artifact contract

Add one shared classifier outside `scripts/research_conductor.py` that accepts
only final status/verdict combinations and rejects `running`,
`running_bootstrap`, bootstrap-only, missing, and contradictory artifacts.
Replay it against Exp6183/Exp6196 plus known valid complete, blocked, skipped,
and retired fixtures. The deliverable must prove both V536 bootstrap artifacts
are nonterminal even though conductor receipts say completed. This is the first
reserved infrastructure slot.

### Exp6198: post-marker source delta and scope audit

Search only evidence dated after the V537 planner marker, record all named
source-channel receipts, append only reproducible deltas, and emit a null if
there are none. In the same deterministic artifact, lint the staged roadmap
against the exclusion manifest, protected paths, SOTA model rules, two-infra
reservation, Phase-D majority, exact one-slot ARC floor, and prompt endings.
This fills the second infrastructure slot and the SOTA-ingestion slot without
pretending planning-time sources are runtime discoveries.

### Exp6199: GateMate unchanged-state terminal-action audit

Satisfy hardware continuity by hashing the canonical Exp6121 and hardware-spec
receipts. If there is no dated physical-route change, run no JTAG, IDCODE,
programming, timing, or power command; emit the exact operator action packet and
terminal blocked/no-change classification. A genuine changed-state receipt may
permit only the already-specified non-destructive detect path. No terminal,
speed, energy, power, TSU, or Kona claim is allowed.

## Phase 1 — Phase-D transport, headroom, and held internal verification (Exp6200-Exp6205)

### Exp6200: three-family raw-code transport canary

On a fixed calibration-only subset of Exp6186, test bounded 512/1024/1536-token
raw-code envelopes for the mandated Gemma-4-31B dense, Qwen3.6-35B-A3B MoE,
and Gemma-4-26B-A4B MoE GGUFs. Persist raw bytes, finish reasons, token counts,
extraction, compile, and public sample-run receipts. Freeze one per-family
configuration without private-test access. Phase-D and CSL readiness are
separate fields so one dead family cannot be hidden by an aggregate.

### Exp6201: authentic Gemma-4-31B executable K=8 pool

Gated on the dense canary. Re-run the 72 immutable selector tasks with exactly
eight independent samples each, the frozen envelope, no correctness retry, and
raw-before-label checkpoints. Execute private tests only after the pool is
sealed. Record real process duration, llama.cpp CUDA offload, both-GPU samples,
finish reasons, code hashes, and restricted executor receipts.

### Exp6202: code competence and selectable-headroom audit

Gated on pool integrity. Measure extraction/runnable coverage, per-candidate
accuracy, correct/incorrect support, oracle@8, tuned label-free code
equivalence/self-consistency, discordant tasks, harmful selections, and
calibration/held strata. Hidden-state work proceeds only when both 36-task
splits have genuine oracle headroom and both outcome classes; otherwise retire
this exact pool and stop the selector chain.

### Exp6203: matching-base hidden-state surface

Gated on headroom. Replay exact calibration rows through the cached
`google/gemma-4-31B-it` revision with `output_hidden_states=True`, using dual
GPU plus explicit CPU offload if necessary. Qualify prompt-final and code-final
layers with exact model/tokenizer/prompt/row/token alignment, precision,
device-map, and quantization-boundary receipts. Do not train a selector or read
held labels.

### Exp6204: calibration-only selector freeze

Gated on surface readiness. Compare CLUE, residualized linear probes,
likelihood/entropy, length/surface/difficulty, and shuffled/random controls
under nested task-level calibration. Materialize held features label-blind,
freeze one selector/layer/threshold/recipe and all hashes, then make further
tuning mechanically impossible.

### Exp6205: one-shot held code selection

Gated on selector freeze. Join held labels once; compare the locked selector
against tuned label-free equivalence/self-consistency, CLUE, likelihood, and
random controls. Report task-level paired bootstrap intervals, oracle headroom
recovered, harmful selections, per-stratum effects, and shortcut audits.
Promotion requires a positive lower interval and no leakage; the same clean
null retires this family/feature construction.

## Phase 2 — Continuous self-learning from verified live code (Exp6206-Exp6207)

### Exp6206: transport-qualified two-family strategy seed

Gated on CSL canary readiness. Rebuild the 18 immutable seed tasks with three
fixed procedural strategies across Qwen3.6-35B-A3B and Gemma-4-26B-A4B, using
their frozen envelopes. Persist raw outputs before exact labeling. Readiness
requires both positive and negative executable events per family and strategy
coverage; a second zero-pass stream retires this exact live-code strategy
construction.

### Exp6207: prospective procedural-memory continuous-learning A/B

Gated on seed readiness. Process the untouched 30-task stream chronologically
for both families. The treatment retrieves bounded procedural memories from an
immutable predecision snapshot; the control uses the frozen no-memory seed
policy. Both decide and generate before labels; only then may verified events
commit. Report forward transfer, negative transfer, regret, hard-case/family
retention, state bytes, eviction, duplicate/reorder/restart/rollback behavior,
and poison propagation. Weights remain immutable.

## Phase 3 — Runtime integration, ARC floor, and reconciliation (Exp6208-Exp6210)

### Exp6208: mode-jump runtime integration from qualified Exp6194 parity

Wire the already-qualified Rust/PyO3 kernel into the existing runtime sampler
selection behind a default-off flag with exact fallback. Test identical seeded
quality, distribution, ESS/autocorrelation, serialization, cancellation,
unsupported-shape fallback, and task-owned integration paths. Timing is
diagnostic only. Do not claim FPGA, TSU, power, energy, or speedup.

### Exp6209: single ARC slot — leave-one-game-out shadow generalization

Registry-precheck first. Select already-cleared games only as evaluation
fixtures, disable each per-game adapter in turn, run the canonical live scored
path, and compare the frozen task-aware/global policies in shadow on fresh
agent-owned transitions. The policy must not alter actions and may not access
source, BFS, per-game adapters, prior-game logs, hidden state, or hidden labels.
This is no-solve generalization measurement: `solve_claimed=false`, registry
hash unchanged, level-credit delta zero. It must not touch the closed local
single-shot induction/refinement axis.

### Exp6210: V537 adversarial capstone

Resolve Exp6197-Exp6209 by exact deliverable path and shared terminal
classification, adversarial-verify every present artifact, and preserve
missing/blocked/skipped/null/retired/flagged/nonterminal classes. Reconcile
specs, traceability, status, changelog, references, exclusions, and hardware
notes only where terminal evidence changed. Headline eligibility is reported
separately for Phase D, FR-11, sampler integration, and ARC generalization.

## Dependency graph and fail-closed gates

```text
Exp6197 terminal contract ───────────────┬──────────────────────────────┐
                                         │                              │
Exp6198 source/scope audit               │                              │
Exp6199 GateMate audit                   │                              │
                                         │                              │
Exp6200 transport canary                 │                              │
  ├─ [phase_d_transport_ready==1] -> Exp6201 K8 pool                   │
  │                                      │                              │
  │                           [pool_integrity==1]                        │
  │                                      v                              │
  │                              Exp6202 headroom                        │
  │                                      │ [headroom==1]                 │
  │                                      v                              │
  │                              Exp6203 surface                         │
  │                                      │ [surface==1]                  │
  │                                      v                              │
  │                              Exp6204 freeze                          │
  │                                      │ [selector==1]                 │
  │                                      v                              │
  │                              Exp6205 held                            │
  │
  └─ [csl_transport_ready==1] -> Exp6206 seed [ready==1] -> Exp6207 CSL

Exp6194 prior parity (verified inside Exp6208) -> Exp6208 runtime integration
Exp6195 frozen policy evidence  -> Exp6209 ARC LOO shadow

Exp6197-Exp6209 terminal classes ────────────────────────────> Exp6210
```

Every natural-language gate in a title is duplicated as structured
`gated_on` YAML. The selector branch and CSL branch share only the transport
canary; either may fail without suppressing sampler integration, ARC, hardware,
source audit, or capstone execution.

## Allocation and roadmap-rule compliance

| Allocation | Experiments | Count |
|---|---|---:|
| Phase-D executable-code/internal-state science | Exp6200-Exp6205 | 6 |
| Continuous self-learning science | Exp6206-Exp6207 | 2 |
| Runtime stochastic integration | Exp6208 | 1 |
| ARC generalization floor | Exp6209 | 1 |
| Infrastructure/SOTA/hardware/capstone | Exp6197-Exp6199, Exp6210 | 4 |
| **Total** | Exp6197-Exp6210 | **14** |

Exp6197 and Exp6198 are the two reserved infrastructure slots. Exp6198 also
fills the focused SOTA-ingestion slot. Exp6199 maintains the only attached
nonterminal board without repeating an unchanged probe. Phase D owns six of
ten non-foundation slots and therefore remains the scientific majority. There
is exactly one ARC task, and Exp6207 is the mandatory continuous self-learning
experiment.

## Hardware and model requirements

| Resource | Experiments | Requirement and boundary |
|---|---|---|
| Dual RTX 3090 host | Exp6200-Exp6201, Exp6203, Exp6206-Exp6207 | Record CUDA/offload and both-device utilization/memory at real intervals. Fail closed if the requested cached local path is unavailable. |
| Flagship dense GGUF | Exp6200-Exp6201 | `unsloth/gemma-4-31B-it-GGUF`; headline Phase-D pool generator. |
| Flagship MoE GGUFs | Exp6200, Exp6206-Exp6207 | `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF`; both required for CSL headline rows. |
| Matching dense base | Exp6203-Exp6204 | Cached `google/gemma-4-31B-it`, exact revision `518276fb...`, with dual-GPU/CPU-offload receipts and no network download. |
| CPU/RAM/local disk | All | Restricted code execution, cached LiveCodeBench, raw checkpoints, hidden-state shards, bootstrap analysis, Rust builds, and atomic artifacts. |
| Rust/PyO3 toolchain | Exp6208 | Existing sampler crates and extension boundary; default-off integration and exact fallback. |
| GateMate A1 | Exp6199 | Cached receipt audit by default. No physical command unless a new dated state hash authorizes the specified non-destructive path. Passive cooling remains disclosed. |
| KV260 / PolarFire | None | Both have clean terminal receipts and have graduated from mandatory continuity. |
| TSU / Kona | None | No authenticated local hardware, public weights, reproducible architecture, or local API. |
| Network | Exp6198 only | Low-concurrency primary/first-party source refresh. Compute experiments remain local-first and must not download models at runtime. |

Every experiment that invokes an LLM includes at least one user-mandated SOTA
GGUF in `MODEL_SPECS` and records exact hub ID, file hash, quantization, prompt
template, context, token budget, GPU placement, and llama.cpp receipt. Legacy
Qwen3.5-0.8B or Gemma-4-E4B may smoke a harness only and cannot supply a
headline row. GGUF directories are never passed to a Hugging Face tokenizer.

## Promotion and retirement rules

- **Terminal contract:** both V536 bootstrap artifacts must classify
  nonterminal; known complete/blocked/skipped/retired fixtures must preserve
  their terminal classes; no protected result may be rewritten.
- **Transport:** the dense Phase-D family and both CSL families report separate
  readiness. Raw bytes must precede extraction, no private test may select an
  envelope, and a second same-class truncation/syntax verdict retires that
  exact construction.
- **Pool:** integrity requires all 576 raw-before-label samples, exact `K=8`,
  zero correctness retries, deterministic resume, valid restricted-executor
  receipts, and plausible live duration.
- **Headroom:** both 36-task splits require at least 30 gradeable tasks, both
  outcome classes, non-saturated competence, and at least 0.10 recoverable
  oracle headroom over tuned label-free equivalence/self-consistency with
  nontrivial discordance. Failure retires this pool before hidden-state spend.
- **Internal selector:** promotion requires a positive held task-level delta
  whose paired bootstrap lower bound exceeds zero, no private-test/label
  leakage, shuffled/random controls at chance, and stable length/surface
  residuals. `verifier_is_oracle=false` is mandatory.
- **Continuous learning:** promotion requires live outputs from both mandated
  MoE families, positive and negative seed events, predecision read-only
  snapshots, delayed verified commits, positive lower intervals by family,
  no negative-transfer/retention safety failure, bounded state, immutable
  weights, and zero poison propagation.
- **Sampler integration:** promotion means exact runtime quality/fallback and
  task-owned test readiness only. Speed, power, energy, FPGA, and TSU claims
  remain forbidden.
- **ARC:** this is shadow leave-one-game-out measurement, not a solve. The
  registry is immutable, level credit is zero, and every forbidden access
  count is bare zero. No `solve_provenance` field is needed because no level
  solve is claimed.
- **GateMate:** unchanged state means no hardware command. A terminal hardware
  claim remains forbidden while the historical flash artifact is
  adversarial-flagged and no new authorized route exists.
- **Capstone:** nonterminal, missing, gated, null, retired, and flagged evidence
  stays excluded; conductor completion receipts cannot override artifact state.

## Explicitly deferred

- Another full code pool or live strategy stream before Exp6200 qualifies its
  exact family; any finite-ID, grammar, stop-token, parser-only, CCTU, or
  external generated-text/logprob scorer retry.
- Proof/invariant co-generation, RepoZero-style generated-test evolution,
  pairwise verifier co-training, CodeCircuit attribution graphs, NRGPT/EBT
  training, or learned distributional energy before the linear/CLUE selector
  has a competent held test.
- Weight-updating, GRPO, MemoPilot training, KAN retraining, or unbounded memory
  before the frozen-weight procedural-memory A/B establishes live value.
- ARC local single-shot induction, refinement, prompt/budget/repetition work,
  first-contact search signals, per-game adapters, offline BFS, or any level
  solve/registry-credit target.
- GateMate programming/timing without a dated changed-state receipt; KV260 or
  PolarFire continuity after terminal graduation; all TSU/Kona, speed, power,
  energy, or hardware-acceleration claims.
- Public documentation, publication, model upload, plugin installation,
  external messages, active-roadmap replacement, conductor modification, or
  push operations.
