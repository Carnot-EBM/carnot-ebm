# Research Roadmap vNEXT — Causal ARC Levers and Runtime Recovery

**Created:** 2026-08-07

**Target milestone:** `2026.08.538`

**Status:** Planned after terminal milestone `2026.08.537`

**Experiment range:** Exp6211-Exp6224 (14 tasks, four phases)

**Primary question:** Which operator-authorized ARC mechanisms improve the
canonical live path under matched controls, and can Carnot restore the local
flagship runtime while advancing safe continuous learning and executable-code
verification without repeating retired work?

**Informed by:** `research-program.md`, `_bmad/prd.md`,
`_bmad/architecture.md` (last reconciled 2026-07-03; stale by the 30-day
freshness rule), `ops/status.md`, `ops/changelog.md`,
`research-complete.yaml`, milestone `2026.08.537`, all prior roadmap
proposals, `ops/conductor-log.md`, `research-references.md` through the
`V538-PLANNER-REFRESH-20260807-END` marker,
`research-hardware-wishlist.md`, `ops/known-issues.md`,
`ops/exclusion_manifest.yaml`, and the exact Exp6197-Exp6210 artifacts.

## What milestone 2026.08.537 proved

| Branch | Terminal evidence | Consequence for V538 |
|---|---|---|
| Artifact integrity | Exp6197 shipped a fail-closed terminal-artifact classifier. It rejects bootstrap-only and contradictory artifacts. Exp6210 then used exact artifact state rather than conductor receipts. | Reuse the classifier. Do not spend another slot on terminality unless a new failure occurs. |
| Source and scope audit | Exp6198 found no reproducible post-V537-marker source delta. Its stored adversarial stamp was a stale duration heuristic, while a live recheck was clean. | Start from the new V538 planning marker. A runtime source audit may honestly be null. |
| GateMate | Exp6199 found no dated cable, power, port, board, or DirtyJTAG change and ran zero hardware commands. | Do not repeat the cached audit. Preserve the blocked state until a new physical receipt exists. |
| Local flagship runtime | Exp6200 attempted all 18 planned model/task/budget cells, but every load ended in `backend_exception`; no family emitted one token and both transport readiness scores were zero. | Diagnose model-file integrity, loader selection, process ownership, VRAM admission, CUDA offload, and server lifecycle before any new generation task. Do not rerun the 18-cell canary unchanged. |
| Phase D and live code memory | Exp6201-Exp6207 were blocked, skipped, or missing because Exp6200 qualified no model family. No new code pool, headroom, internal-state selector, seed stream, or prospective memory result exists. | After runtime recovery, rerun only a small transport canary, then one immutable pool and one headroom audit. Hidden-state work remains deferred until genuine headroom exists. |
| Stochastic substrate | Exp6208 integrated the qualified mode-jump Rust/PyO3 kernel behind a default-off fallback boundary. Seeded parity passed, but several repository-wide checks exited nonzero and no runtime quality or speed claim was made. | Run a bounded matched runtime A/B with exact quality, ESS, autocorrelation, fallback, and test classification. Keep the feature default off. |
| ARC generalization | Exp6209 scored 48 fresh canonical live-path transitions over six leave-one-game-out cells. The frozen task-aware policy beat the global policy by `0.208333`, with zero losing games, zero live action influence, and no solve or registry credit. | Preserve this as shadow evidence. The newer operator six-lever directive authorizes causal A/Bs of the shipped live mechanisms; it does not authorize solve farming or another single-shot refinement rerun. |

V537 therefore fixed evidence terminality, produced one positive ARC shadow
result, and shipped one default-off sampler integration. It also localized the
Phase-D blocker below generation: the three mandated GGUF families did not
load. The milestone did **not** measure any of the newly shipped ARC levers,
produce a competent code pool, or run prospective live-code self-learning.

## The three largest gaps to the PRD vision

### Gap 1 — the live ARC architecture has mechanisms but no causal ranking

The PRD's north star is an adaptive live agent on unseen interactive games.
The 2026-08-07 operator directive caused six levers to land: larger live
budgets, the one authorized Exp6091 fix, object-centric input, object-relative
trajectory transfer, budget-aware search, and a Gemma-4-31B thinking toggle.
Exp6091 is closed again on complete data. The remaining mechanisms are either
partial or default off and have no matched live-path A/B. Carnot must measure
each mechanism independently before testing a portfolio or changing defaults.

### Gap 2 — the local flagship runtime blocks the oracle-distinct verifier path

FR-12 still lacks a held oracle-distinct selection win on competent local
outputs. V537 moved the bottleneck from token-envelope speculation to a lower
runtime failure: every exact cached SOTA GGUF load failed. Until Carnot records
file integrity, compatible loader/build, real CUDA layer placement, owned
process lifetime, and at least one generated token for each family, another
large pool is waste. Runtime recovery must precede a changed canary, immutable
Gemma-4-31B K=8 pool, and executable-headroom audit.

### Gap 3 — continuous learning and stochastic inference lack safe temporal use

FR-11 requires future decisions to improve from verified experience without
forgetting. Existing experiments disagree on update timing: Exp5968 found its
write-through control strongest, while Memoir reports that writing during
pondering can hurt finite-budget learning. Carnot needs a clean distinction
between immutable predecision reads, immediate **post-outcome** commits, and
block-end consolidation under known constraint drift. In parallel, the
mode-jump sampler is integrated but not qualified as a runtime choice. Both
branches need matched, rollback-safe evaluation rather than new components.

## Research findings incorporated

| 2025-2026 source | Finding used | V538 response |
|---|---|---|
| ARCANA (arXiv:2607.09059) | Object-centric scene graphs, executable hypotheses, and failure-driven reflection form a useful reasoning decomposition. | Complete generic per-transition object deltas, translation-invariant identity tracking, and HUD-strip rejection. Measure only through `make_carnot_agent`/`E3AgentPolicy`; import no task solution or adapter. |
| *Cost-Effective Agent Harnesses for Abstract Reasoning and Generalization on ARC-AGI-1* (arXiv:2607.06764) | Separating exploration from executable definition is effective, and a think-tool ablation has measurable value under fixed budgets. | Pre-register `/think` versus `/no_think` for the local Gemma-4-31B inducer with identical prompts, samples, and budgets; report quality, harmful regressions, tokens, and wall time. |
| *Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?* (arXiv:2607.15439) | Nested ablations are needed to attribute gains among executable modeling, simplification, and replay verification. | Give every ARC A/B matched seeds, identical budgets, exact replay, fire counts, A/A controls, and no-solve provenance. Test a portfolio only after independent measurements exist. |
| Hyper-SET (arXiv:2502.11646; ICLR 2026) | Recurrent-depth computation can arise from explicit energy minimization and extrapolate to more test-time iterations. | Keep energy monotonicity and train-short/test-long curves as Phase-3 comparators; do not open a new training track in this milestone. |
| Audited Skill-Graph Self-Improvement (arXiv:2512.23760), Memoir (arXiv:2607.20792), and AgentCL (arXiv:2606.02461) | Continual improvement needs verifier-backed promotion, reconstructible rewards, delayed writes, retention, and explicit negative-transfer controls. | Compare post-outcome and block-end commits on a sealed exact drift stream. Keep decision snapshots read-only, weights immutable, and rollback atomic. |
| Code-correctness internal-state papers (arXiv:2512.07404 and 2606.14530), WybeCoder (2603.29088), and RepoZero (2605.07122) | Executable code supplies exact labels and potential internal signal, but only after authentic competence and headroom exist. | Restore runtime, qualify transport, seal a raw pool before private labels, and measure headroom. Defer hidden-state fitting and proof/test evolution. |
| Extropic writing/THRML, Kona first-party pages, KAN search, Hugging Face Papers, GitHub, and EBT/ARM-EBM citation trails | No new authenticated TSU, reproducible Kona API, KAN replacement, or repository removes the current local prerequisites. | Keep physical hardware and new architecture training deferred. Run only the already-integrated software sampler A/B. |

The dated discovery and duplicate-suppression record is in
`research-references.md` under the V538 planner marker.

## Target architecture

```text
                           V538 source/scope receipt
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
          ARC live-path branch   Local GGUF branch   Temporal substrate
                 │                    │                    │
      object-delta wiring        3-family runtime      exact constraint
                 │               recovery/preflight      drift stream
       ┌─────────┼─────────┐          │                    │
       │         │         │    changed raw-code      no memory / immediate
 object A/B  trajectory  budget       canary          post-outcome / block-end
             transfer A/B  A/B         │               commit + rollback
       └─────────┼─────────┘     Gemma-4-31B K=8            │
                 │                    │               FR-11 decision
       Gemma-4-31B think A/B     exact labels              │
                 │                    │         mode-jump runtime A/B
       eligible-lever portfolio  competence/headroom       │
       on held-out games              │                    │
                 └────────────────────┼────────────────────┘
                                      │
                         exact-path adversarial capstone

 GateMate: preserve Exp6199 blocked state; no command without a new dated
 physical-state receipt. No task in V538 claims a new ARC level solve.
```

The ARC branch has one rule: a mechanism must first fire, then improve a
pre-registered quality or efficiency metric without a safety regression.
Neither activation alone nor an aggregate that hides a losing game permits a
default flip. The portfolio task may use only independently admissible levers
and remains shadow/default off.

## Phase 0 — Dated ingress, runtime recovery, and object representation (Exp6211-Exp6213)

### Exp6211: post-marker SOTA delta and causal-scope preregistration

Search only evidence strictly later than the V538 planning marker. Record all
named source channels and append only reproducible deltas. Independently lint
the staged roadmap against model, retirement, gate, prompt, ARC, CSL, and
hardware rules. Freeze the ARC A/B outcome vocabulary and registry nonmutation
contract before any measurement.

### Exp6212: three-family GGUF runtime recovery

Reproduce one failed Exp6200 load per model family under a read-only diagnostic
preflight. Verify the exact file, hash, size, revision, quantization, embedded
template, llama.cpp build, compatible loader path, owned PID, GPU ownership,
VRAM admission, CUDA layers, stderr, exit reason, and process lifetime. Make
the smallest task-owned fix outside the conductor. A family is ready only when
it produces and records at least one deterministic canary token with real CUDA
offload. Do not kill unrelated processes or rewrite GGUF files.

### Exp6213: transition-aware object representation wiring

Finish the partial object lever under a new default-off flag. Add per-transition
connected-component deltas, translation-invariant identity matching, and
HUD-strip rejection to the generic object table. Wire it into the canonical
induction input with unit, mutation, and live-closure tests. This task proves
wiring and invariant preservation only; it does not measure an ARC gain.

## Phase 1 — Independent ARC lever measurements (Exp6214-Exp6218)

### Exp6214: held-out object-delta live-path A/B

On a pre-registered game/seed matrix, compare the shipped object table with and
without the Exp6213 transition additions. Use the same Gemma-4-31B generator,
prompt apart from the object section, action budget, and replay verifier. Report
per-game change fidelity, executable-engine yield, action efficiency, fire
counts, and harmful regressions. This is the milestone's explicit held-out ARC
generalization test and claims no solve.

### Exp6215: object-relative trajectory-transfer A/B

Compare the default-off within-game, level-to-level transfer stage with an
identical baseline on replayable, already-cleared games. Measure transfer fire
rate, verifier acceptance, displacement correctness, avoided LLM induction
calls, actions, score, and per-game losses. This is not cross-game value
transfer and cannot use source, BFS, adapters, registry trajectories, or hidden
state as live inputs.

### Exp6216: budget-aware search A/B

Compare `CARNOT_ARC_BUDGET_AWARE_SEARCH=0/1` with matched seeds and budgets.
Require the estimator and frontier weight to fire on admitted HUD evidence;
report deadline misses, path cost, states expanded, navigation actions, score,
and levels observed without claiming new solve credit. A non-firing treatment
is an instrument failure, not a null effect.

### Exp6217: Gemma-4-31B `/think` A/B

After Exp6212 proves the dense runtime, compare native `/think` and `/no_think`
inside the canonical world-model induction call. Hold prompt content, already
expanded live budget, sampling, tasks, and replay verifier fixed. This task is
authorized by the 2026-08-07 directive but may not revisit Exp6091, tune
`n_predict`, or treat longer output as success. Primary evidence is executable
transition fidelity and goal/change accuracy; tokens, time, completion, and
harmful regressions are co-primary cost/safety receipts.

### Exp6218: independently admissible lever portfolio on held-out games

Read the four A/B artifacts without trusting prose. If fewer than two levers
are complete, fire, and satisfy their pre-registered safety gate, emit a
structured skip. Otherwise freeze at most the two strongest independently
admissible levers before opening a new held-out game/seed matrix. Compare
baseline, each lever, and their pair. Report interactions and every per-game
loss. Keep the portfolio default off and leave the solve registry unchanged.

## Phase 2 — Continuous learning and software sampling (Exp6219-Exp6220)

### Exp6219: two-timescale continuous constraint learning under drift

Use the clean exact Exp6145 nonstationary constraint stream, not flagged
decision artifacts. Freeze chronological family blocks and future evaluation
IDs. Compare no memory, immediate post-outcome commit, block-end consolidation,
and shuffled-memory controls. Every decision reads an immutable snapshot;
verified constraints become procedural records only after outcome disclosure.
Measure forward transfer, retained accuracy, negative transfer, update utility,
quarantine, poison rejection, and rollback. GGUF weights remain immutable and
no LLM runs. This is the mandatory continuous self-learning experiment.

### Exp6220: mode-jump sampler runtime A/B

Exercise the Exp6208 default-off runtime boundary on fixed multimodal and
unimodal energy fixtures. Compare the existing fallback and mode-jump backend
under matched seeds, samples, and schedules. Report exact support validity,
energy/observable error, ESS, autocorrelation, transition counts, serialization,
fallback behavior, and measured CPU wall time. A speed claim is forbidden
unless quality gates pass and uncertainty excludes parity; no FPGA or TSU is
involved.

## Phase 3 — Transport-qualified code evidence and closure (Exp6221-Exp6224)

### Exp6221: changed three-family raw-code canary

Gated on all three Exp6212 runtime receipts. Reuse the immutable Exp6186
calibration tasks and the preregistered 512/1024/1536 token grid, but now record
owned server lifecycle and real token output. Store raw bytes before extraction
and use only finish, extraction, compile, and public sample-run evidence to
freeze an envelope. Private tests cannot select configuration. If the same
zero-ready verdict returns after runtime recovery, retire this canary lineage.

### Exp6222: authentic Gemma-4-31B K=8 code pool

Gated on dense transport readiness. Generate exactly eight independent samples
for each of the 72 immutable selector tasks. Seal every raw row before opening
private tests, then label all rows through the restricted executor. No repair,
replacement, correctness retry, or held-task reselection is permitted.

### Exp6223: executable-code competence and headroom audit

Gated on pool integrity. Measure runnable coverage, per-candidate accuracy,
both-class support, oracle@8, tuned label-free code equivalence/self-consistency,
discordant tasks, harmful selections, and calibration/held strata. Hidden-state
work proceeds in a later milestone only if both splits contain genuine
non-combinatorial headroom. Otherwise retire this exact pool and record why.

### Exp6224: V538 exact-path adversarial capstone

Classify every declared artifact with the Exp6197 terminal contract. Preserve
blocked, skipped, null, retired, and adversarial states independently. Re-run
adversarial checks, verify ARC registry nonmutation, record the GateMate blocked
boundary without a board command, and emit branch-specific eligibility for
ARC levers, continuous learning, sampler runtime, and Phase D. Reconcile
`openspec/`, `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
only to measured facts.

## Dependency graph

```text
Exp6211 source/scope prereg
  ├── Exp6212 GGUF runtime recovery
  │     ├── Exp6217 Gemma-4-31B think A/B ───────────────┐
  │     └── Exp6221 code canary -> Exp6222 pool -> Exp6223 headroom
  ├── Exp6213 object-delta wiring -> Exp6214 held-out A/B ─┐
  ├── Exp6215 trajectory-transfer A/B ────────────────────┤
  └── Exp6216 budget-aware-search A/B ────────────────────┤
                                                          v
                                                    Exp6218 portfolio

Exp6211 -> Exp6219 two-timescale CSL
Exp6211 -> Exp6220 mode-jump runtime A/B

Exp6211-Exp6223 -> Exp6224 exact-path capstone
```

No task may require a retired experiment ID. Exp6218 uses only current A/B
artifacts and runs a new held-out matrix; it does not depend on retired ARC
composition or cross-game value-transfer lines.

## Hardware and model requirements

| Resource | Use | Fail-closed preflight |
|---|---|---|
| RTX 3090 GPU 0 | Dense Gemma-4-31B runtime, ARC induction A/Bs, code generation | Record owner/PID, free VRAM, CUDA build, exact offloaded layers, model interval, and wall time. Never kill an unrelated process. |
| RTX 3090 GPU 1 | Qwen3.6-35B-A3B and Gemma-4-26B-A4B runtime canaries; explicit offload only when task-owned | Same receipts. If occupied, block or use a pre-registered safe split; do not fabricate readiness. |
| CPU/Rust/PyO3 | Object representation, CSL, mode-jump sampler, audits, restricted code execution | Record threads, seeds, schedule, exact backend, fallback, and wall time. Software evidence carries no board claim. |
| GateMate | No V538 command planned | Exp6199 remains authoritative. A new dated physical-state receipt is required before any detect/program action. |
| KV260 / PolarFire | No active experiment | Their prior terminal receipts remain authoritative. Do not reopen parity or speed work. |
| Extropic TSU / Kona | Not available | Software/docs context only. No authenticated device/API means no runtime, latency, power, or speed claim. |

LLM tasks must use these cached headline models:

- `unsloth/Qwen3.6-35B-A3B-GGUF` — flagship MoE.
- `unsloth/gemma-4-31B-it-GGUF` — flagship dense and sole ARC inducer.
- `unsloth/gemma-4-26B-A4B-it-GGUF` — middle MoE.

Legacy Qwen3.5-0.8B and Gemma-4-E4B may appear only in CPU smoke tests and
contribute zero headline rows. Every live task records exact cached path,
revision, hash, quantization, embedded chat template, context, sampling,
llama.cpp build, device placement, and process interval.

## Promotion, stopping, and retirement rules

1. **ARC:** one lever must fire and improve its pre-registered metric without a
   safety regression. No task claims a solve, updates the registry, imports
   source/BFS/adapter truth, or treats a development proxy as live evidence.
2. **Portfolio:** fewer than two independently admissible levers produces a
   structured skip. No exploratory combination fishing is allowed.
3. **Runtime:** readiness requires a real owned process, CUDA offload, and
   persisted output token. File presence or a conductor completion receipt is
   insufficient.
4. **Continuous learning:** promotion requires nonnegative protected retention,
   positive forward utility, zero decision-time writes, exact post-outcome
   provenance, poison/quarantine checks, and successful rollback. A repeated
   Exp5895-style nonpromotion retires this exact two-timescale construction.
5. **Sampler:** quality precedes timing. Default remains off unless support,
   observable, ESS/autocorrelation, fallback, and task-owned tests pass.
6. **Phase D:** no canary readiness means no pool; no pool integrity means no
   headroom audit; no genuine headroom means no later hidden-state task.
7. **Hardware:** unchanged GateMate state causes no command. No V538 result may
   claim FPGA, TSU, Kona, power, or hardware speedup.
8. **Evidence:** every terminal artifact carries principle annotations,
   `inference_substrate`, `verifier_is_oracle`, real `duration_s`, checksum,
   test commands/exit codes, and a terminal-prefixed `honest_verdict`.

## Deferred beyond V538

- Hidden-state surface extraction, CLUE fitting, and held code selection until
  Exp6223 proves both competent candidates and real headroom.
- Any ARC level solve, registry update, public-game adapter, offline BFS, or
  single-shot refinement rerun.
- Default flips for ARC levers or the mode-jump sampler before independent
  safety and held-out evidence.
- Weight-changing continual learning, live LoRA, or same-pass memory writes.
- Hyper-SET/EBT/KAN training, proof/invariant co-generation, generated-test
  evolution, TSU/Kona execution, and physical GateMate work.
