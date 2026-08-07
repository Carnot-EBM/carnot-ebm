# Research Roadmap vNEXT — Executable Code Verification and Prospective Learning

**Created:** 2026-08-07

**Target milestone:** `2026.08.536`

**Status:** Planned after terminal milestone `2026.08.535`

**Experiment range:** Exp6183-Exp6196 (14 tasks, four phases)

**Primary question:** Can Carnot replace the retired CCTU transport with an
authentic, executable local-SOTA code pool, demonstrate oracle-distinct value
from matching-model internal states, and turn deterministic replay into
prospective retention-safe learning while preserving evidence and live-path
boundaries?

**Informed by:** `research-program.md`, `_bmad/prd.md`,
`_bmad/architecture.md` (stale baseline, reconciled against newer artifacts),
`ops/status.md`, `ops/changelog.md`, `research-complete.yaml`, the full prior
roadmap corpus, `ops/conductor-log.md`, `research-references.md` through the
V536 marker, `research-hardware-wishlist.md`, `ops/known-issues.md`, and
`ops/exclusion_manifest.yaml`.

## What milestone 2026.08.535 proved

| Branch | Terminal evidence | Consequence for V536 |
|---|---|---|
| Transition/evidence | Exp6169 exhausted three hard wall-clock attempts and left its exact transition artifact missing. Exp6170 qualified only a task-scoped isolation surface with readiness `0`, and Exp6172's current-rule companion was itself flagged by duration/methodology heuristics. | Start with a minimal exact-boundary receipt and a narrower V536 canary. Do not claim repository-wide isolation or reconstruct the missing Exp6169 artifact. |
| Phase D pool | Exp6173 froze a valid 120-case bank. Exp6174 collected 960 authentic Gemma-4-31B samples, but all 960 failed the structured parse. Exp6175 therefore retired the CCTU pool for failed parseability, competence, headroom, minority, and family-support gates; Exp6176-Exp6178 were correctly skipped or blocked. | Change domain and transport. Use raw Python code plus a restricted execution oracle, not another CCTU JSON/schema/grammar/parser retry. Preserve the same bank-first, pool-second, headroom-before-hidden-state discipline. |
| Continuous self-learning | Exp6179 found a positive bounded-replay strategy-memory result with retention, rollback, and poison controls, but no live model generation executed. | Acquire fresh flagship-generated, executable events and run a chronological prospective A/B with read-only decision snapshots and post-outcome commits. Do not treat deterministic replay as continuous live learning. |
| Stochastic substrate | Exp6180 reproduced Exp6166's mode-jumping CNCE improvement from immutable software evidence while preserving the original blocked status and making no hardware claim. | Move the fixed algorithm across the Rust/PyO3 production boundary and close parity/tests. Do not rerun THRML scaling or claim TSU/board speedup. |
| ARC | Exp6181 found no task-logo shortcut in the fixed Exp6167 task-aware policy, using 144 live-agent-owned transitions, with no solve or registry delta. | Use the single ARC slot for prospective fresh-transition replay of the already-frozen policy. Add no induction prompt, search lever, per-game adapter, or solve target. |
| Capstone | Exp6182 preserved missing, null, flagged, blocked, retired, skipped, software-only, and no-solve classes without laundering them. | Keep branches independent and reconcile only exact declared paths. A failed code-selector gate must not suppress CSL, sampler parity, ARC, or the capstone. |

## The three largest gaps to the PRD vision

### Gap 1 — no oracle-distinct verifier has selected better local-SOTA outputs

PRD FR-12 requires verifiable reasoning that improves outcomes rather than
merely detecting errors. Carnot has exact code validators and internal-state
infrastructure, but its latest Phase-D domain never yielded one parseable
candidate. Earlier MMLU-Pro hidden-state probes did not beat tuned
self-consistency, and the code oracle-distinct replication was corpus-specific.
The missing evidence is a fixed, competent, unsaturated, authentic `K>=8` pool
with an executable oracle used only for labels/evaluation, followed by a
task-disjoint one-shot selector test.

### Gap 2 — continuous learning is still replay-only at the live frontier

PRD FR-11 and the research program require systems that improve from verified
experience while they run. Exp6179 proved useful transaction mechanics on a
deterministic event table, not on new flagship generations. Carnot still needs
a chronological live A/B in which a policy chooses before inference, cannot
see the test oracle, reads an immutable memory snapshot, and commits the
verified outcome only after both arms decide.

### Gap 3 — positive software mechanisms are not yet portable or prospectively general

The PRD calls for a Python/Rust architecture and eventual hardware-neutral
sampling. Exp6180's mode-jump result is Python/JAX software evidence rather than
a production Rust/PyO3 contract. The ARC task-aware policy has positive
retrospective evidence but has only been checked on its existing transition
corpus. V536 must test cross-runtime parity and fresh live-agent-owned
generalization without converting either result into a hardware or solve claim.

An operational sub-gap cuts across all three: transition history and tracked
result isolation remain partial. V536 reserves two foundation slots for an
exact transition and a task-scoped evidence canary, but deliberately avoids an
open-ended repository-wide migration.

## Research findings incorporated

| 2025-2026 source | Finding used | V536 response |
|---|---|---|
| *On LLMs' Internal Representation of Code Correctness* (arXiv:2512.07404, ICSE 2026) | Correct and incorrect code from the same task can be separated using model internals and used for sample selection. | Build a new executable-code internal-state branch rather than retrying CCTU or external-text scorers. |
| *Code Correctness Is Linearly Decodable from LLM Hidden States Before Generation* (arXiv:2606.14530 v3) | Leakage-controlled prompt-final probes retain signal after prompt-length residualization; repair evidence was too sparse. | Test prompt-final and code-final features, residualize length, use nested task splits, and make no repair claim. |
| *Solver-Hard Is Not Model-Hard* (arXiv:2607.17047) | Classical solver difficulty and surface form can dissociate from model difficulty. | Freeze difficulty/platform strata, but gate on observed flagship competence/headroom and run surface/length controls. |
| *Distributional Energy-Based Models for Uncertainty-Aware Structured LLM Reasoning* (arXiv:2605.18871) | Deterministic penalties and learned energy are complementary; code scorers can learn model identity shortcuts. | Keep exact tests out of selector inputs and report model/family, length, likelihood, and uncertainty controls separately. Do not reopen the retired external-text scorer. |
| *Memoir: Should a Model Write to Its Memory While It Thinks?* (arXiv:2607.20792) | Same-pass fast-memory writes slowed finite-budget learning relative to read-only pondering. | Keep decision-time memory read-only and commit only verified post-outcome events. |
| *Thermalizing Stochastic Programs* (arXiv:2608.01615) and *Scaling Up Thermodynamic AI Models* (arXiv:2607.00170) | Factor-boundary compilation requires explicit error accumulation, schedule, autocorrelation, and effective-sample accounting. | Carry Exp6166's fixed mode-jump mechanism into Rust/PyO3 with exact transition/distribution parity and sampling-quality receipts; make no THRML scaling or hardware claim. |
| VeRA (arXiv:2602.13217) | Executable task specifications enable fresh verified variants and hardening. | Freeze executable task and validator manifests before model access; leave generated variants to a later milestone after the fixed held result. |
| OpenReview/Hugging Face/GitHub ecosystem checks | NRGPT, V_1, CodeCircuit, and THRML are relevant comparators, but none removes the need for local pool viability and cheap baselines first. | Run CLUE/linear baselines before attribution graphs, pairwise co-training, or new EBM architectures. |
| Extropic and Logical Intelligence first-party checks | THRML is usable in software, but Carnot has no authenticated TSU route; Kona still has no reproducible local weights/API. | No TSU, Kona, latency, power, or speedup experiment. |

The full dated source record and guarded interpretations are in
`research-references.md` under
`V536-PLANNER-REFRESH-20260807-END`.

## Target architecture

```text
                 exact transition + V536 task-scoped evidence canary
                                      │
                 ┌────────────────────┼─────────────────────┐
                 │                    │                     │
        executable code Phase D   prospective FR-11    portability/generalization
                 │                    │                     │
 cached LiveCodeBench snapshot   frozen CSL splits    Exp6166 immutable evidence
         ┌───────┴────────┐           │                     │
  frozen cal/held IDs   exact     live seed strategy     Rust core + PyO3 ABI
  + private-test vault   tests     events, two SOTA      exact transition/
         │                 │       model families         distribution parity
 Gemma-4-31B GGUF K=8     │           │
 raw code before labels   │     read-only snapshot
         └───────┬─────────┘     choose → generate → verify
                 │                    │
 competence + runnable +          post-outcome bounded commit
 oracle-headroom audit               │
                 │              prospective memory vs no-memory
 matching HF Gemma-4-31B
 prompt/code hidden states
                 │
 CLUE + residualized linear probe
 calibration-only freeze
                 │
 one-shot held selection vs tuned SC

 submitted ARC kernel → fresh agent-owned transitions → fixed Exp6167 replay
            (source/BFS/adapters/prior logs/hidden state disabled; no solve)

All branches ───────────────────────────────→ exact-path capstone
```

The GGUF generation and matching Hugging Face hidden-state surfaces are
distinct substrates. Joining their rows requires model/revision/hash receipts,
prompt formatting receipts, token-alignment checks, and an explicit
quantization-boundary caveat. Exact private tests are the candidate outcome
oracle; they may label calibration rows and evaluate held rows, but their
inputs, outputs, failure messages, and derived features may not enter a
selector before its decision.

## Phase 0 — Exact evidence and dated ingress (Exp6183-Exp6185)

### Exp6183: minimal exact `.535` to `.536` transition

Write a blocked/ready receipt first, preserve Exp6169 as missing, verify that
the `.535` completion record is not duplicated, collision-check Exp6183-
Exp6196, and activate only the exact staged V536 roadmap. This task is
pre-routed to Opus because the previous Codex transition exhausted three hard
wall-clock windows.

### Exp6184: V536 evidence-isolation and history-multiplicity canary

Qualify only the writers/tests introduced by Exp6183-Exp6196. Separate an
expected intercepted negative-control attempt from a real isolation mutation,
prove tracked sentinel/quarantine hashes are unchanged, and report current
`research-complete.yaml` milestone multiplicity without attempting a global
history rewrite. Repository-wide closure remains false.

### Exp6185: post-marker source delta

Search primary and named secondary sources strictly after the V536 planner
marker. Append only genuinely new reliable findings, record null honestly, and
do not reinsert planning-time sources as runtime deltas.

## Phase A — Executable-code Phase D substrate (Exp6186-Exp6189)

### Exp6186: frozen LiveCodeBench bank and preregistration

From the cached dataset revision, freeze 120 task IDs before model access:
36 calibration, 36 held selector evaluation, 18 CSL seed, and 30 CSL
prospective. Stratify by platform, difficulty, date, and input mode; hash every
prompt/test payload; keep private tests in an oracle vault; and pre-register
pool, competence, runnable, headroom, split, selector, and CSL gates.

### Exp6187: authentic local-SOTA `K=8` code pool

Generate exactly eight raw completions for each of the 72 selector tasks with
cached `unsloth/gemma-4-31B-it-GGUF` on CUDA. Persist raw text before extraction
or execution, use bounded restricted execution for labels, checkpoint by
content hash, and record dual-GPU utilization intervals. Do not retry for
correctness or expose private tests to the model.

### Exp6188: runnable competence, unsaturation, and headroom audit

Using only the frozen pool, assess extraction/runnable coverage, correct and
incorrect class support, observed competence, oracle@8, tuned label-free code
self-consistency/equivalence selection, discordant task counts, and all
pre-registered strata. Downstream rows become eligible only when both
calibration and held splits have at least 30 tasks, both outcome classes, and
nonzero selectable headroom. A failure retires this exact pool rather than
launching hidden-state extraction.

### Exp6189: matching-base hidden-state surface qualification

On a fixed calibration canary only, load the cached matching
`google/gemma-4-31B-it` checkpoint across the dual RTX 3090 host, replay exact
prompt/code rows, and qualify prompt-final plus code-final layer features.
Require exact revision/hash, tokenizer/prompt, row, shape, token-alignment,
precision, device-map, and quantization-boundary receipts. This experiment does
not train a selector or inspect held labels.

## Phase B — Internal verification and prospective FR-11 (Exp6190-Exp6193)

### Exp6190: calibration-only CLUE and residualized linear selector freeze

Materialize calibration hidden states and label-blind held features. Compare
CLUE nearest-centroid deltas, residualized linear probes, prompt-final and
code-final layer summaries, likelihood/entropy, length, difficulty, and random
controls using nested task-level calibration splits. Freeze one selector,
threshold, layer/feature recipe, and held-feature checksum before any held
label join.

### Exp6191: one-shot held internal-state code selection

Join the frozen held labels exactly once and compare the locked selector with
tuned label-free self-consistency/equivalence selection, CLUE, likelihood, and
random controls. Report task-level paired bootstrap intervals, oracle headroom
recovered, per-stratum and length-residual results, and shortcut audits.
Promotion requires positive held gain with a positive lower interval and no
oracle leakage; a clean null retires this code-family/feature construction.

### Exp6192: live two-family strategy seed stream

On the 18 frozen CSL seed tasks, run three fixed code-generation strategies
with cached `unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Persist raw outputs before exact execution,
freeze the no-memory baseline strategy using only seed outcomes, and create a
bounded verified experience store. This task acquires seed experience; it does
not claim prospective improvement.

### Exp6193: prospective retention-safe continuous strategy-learning A/B

Process the 30 untouched CSL prospective tasks chronologically for both model
families. The memory arm reads an immutable snapshot and chooses a strategy
before inference; the no-memory arm uses the frozen seed winner. Both arms
generate live outputs, and only after both decisions do exact validators commit
outcomes. Measure utility, learning speed, regret, family retention, abstention,
state bytes, duplicate/reorder/restart/rollback/eviction safety, and poison
propagation. Model weights remain immutable.

## Phase C — Portable sampling, ARC freshness, and closure (Exp6194-Exp6196)

### Exp6194: mode-jump Rust/PyO3 parity

Port the fixed Exp6166/Exp6180 mode-jumping proposal and CNCE accounting into
the existing Rust sampler crate and PyO3 boundary. Compare Python, Rust, and
PyO3 on exact transition probabilities, seeded traces, distributions, KL,
autocorrelation, effective sample size, serialization, and error budgets. The
task is correctness/portability only: no reopened THRML scaling, two-axis
tempering, TSU, board, or speedup claim.

### Exp6195: single ARC slot — prospective fresh-transition policy replay

Run the submitted live kernel with all escape hatches disabled to collect a
fresh, disjoint set of agent-owned transitions. Replay the already-frozen
Exp6167 task-aware and global policies on identical transitions; do not alter
the agent's actions. Audit task-label aliases again, report generalization and
safety intervals, and preserve `solve_claimed=false`, `level_credit_delta=0`,
and registry immutability. This adds no induction or search mechanism.

### Exp6196: branch-independent `.536` capstone

Resolve every task by exact declared path and conductor receipt, adversarial-
verify present artifacts, preserve all missing/blocked/skipped/null/retired/
flagged classes, and reconcile specs, traceability, status, changelog,
references, exclusions, and hardware notes only where evidence changed.
Report whether Phase D, prospective FR-11, Rust/PyO3 parity, and the ARC fresh-
transition result are separately headline-eligible.

## Implementation priority

| Order | Experiments | Reason |
|---|---|---|
| 1 | Exp6183-Exp6185 | Establish exact history/evidence and ingest only post-plan source deltas. |
| 2 | Exp6186 and Exp6192's prerequisite split | Freeze every code task and oracle payload before any model access. |
| 3 | Exp6187-Exp6188 | Pay for the authentic pool, then decide whether hidden-state work is admissible. |
| 4 | Exp6189-Exp6191 | Qualify the matching-base surface, freeze on calibration, evaluate held once. |
| 5 | Exp6192-Exp6193 | Run live seed acquisition and the mandatory prospective self-learning A/B independently of selector success. |
| 6 | Exp6194-Exp6195 | Close cross-runtime and fresh-live-path generalization without hardware or solve claims. |
| 7 | Exp6196 | Reconcile exact evidence after every branch reaches a terminal class. |

## Dependency graph and fail-closed gates

```text
Exp6183 transition ─────────────┬───────────────┬──────────────────────────┐
                               │               │                          │
Exp6184 evidence canary ────────┼──────┐        │                          │
                               │      │        │                          │
Exp6185 source delta ───────────┘      │        │                          │
                                      │        │                          │
Exp6186 frozen bank [ready==1] ────────┼──→ Exp6192 seed [ready==1] ─→ Exp6193 CSL
       │                              │
       └→ Exp6187 K8 pool [integrity==1]
                    │
                    └→ Exp6188 headroom [ready==1]
                                  │
                                  └→ Exp6189 surface [ready==1]
                                                    │
                                                    └→ Exp6190 freeze [ready==1]
                                                                      │
                                                                      └→ Exp6191 held

Exp6184 ─→ Exp6194 Rust/PyO3 parity
Exp6184 ─→ Exp6195 single ARC fresh-transition slot

Exp6183-Exp6195 terminal states ───────────────────────────────→ Exp6196 capstone
```

Every title-level dependency with an artifact condition is encoded as
structured `gated_on` YAML. A failed bank/pool/headroom/surface/selector/seed
gate skips its downstream agent call. Exp6193, Exp6194, Exp6195, and Exp6196 do
not depend on Phase-D selector success.

## Allocation and roadmap-rule compliance

| Allocation | Experiments | Count |
|---|---|---:|
| Phase-D executable-code/internal-state science | Exp6186-Exp6191 | 6 |
| Continuous self-learning science | Exp6192-Exp6193 | 2 |
| Cross-runtime stochastic science | Exp6194 | 1 |
| ARC live-path measurement | Exp6195 | 1 |
| Transition/evidence/source/capstone | Exp6183-Exp6185, Exp6196 | 4 |
| **Total** | Exp6183-Exp6196 | **14** |

Phase D owns six of ten scientific slots, satisfying the standing majority
rule after infrastructure, CSL, sampler, and ARC reservations. There is exactly
one ARC slot. The milestone contains the required continuous self-learning
experiment and no public-facing documentation or publication task.

## Hardware and model requirements

| Resource | Experiments | Requirement and boundary |
|---|---|---|
| Dual RTX 3090 host | Exp6187, Exp6189, Exp6190, Exp6192, Exp6193 | Record both-device utilization/memory intervals. Fail closed before model load if the requested local CUDA/offload or matching-base device map is unavailable. |
| Flagship dense GGUF | Exp6187 | `unsloth/gemma-4-31B-it-GGUF`; this is the headline code-pool generator. Legacy small models may smoke the harness only. |
| Matching dense base checkpoint | Exp6189-Exp6190 | Cached `google/gemma-4-31B-it`, joined to the GGUF family only after revision/token/prompt/row alignment and quantization caveats. |
| Flagship MoE GGUFs | Exp6192-Exp6193 | `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-26B-A4B-it-GGUF`; both are live CSL model families. |
| CPU/RAM/local disk | Exp6186-Exp6196 | Cached LiveCodeBench snapshot, restricted code execution, content-addressed checkpoints, hidden-feature stores, bootstrap analysis, Rust build, and exact artifacts. |
| Network | Exp6185 only | Reliable post-marker source checks; compute tasks are local-first and do not download models at runtime. |
| Rust/PyO3 toolchain | Exp6194 | Existing workspace crates and local extension build; correctness/parity only. |
| FPGA/TSU boards | None | KV260 and PolarFire have terminal receipts; GateMate has no dated physical-state change; Carnot has no authenticated TSU. No board task or hardware claim is eligible. |

Every runtime experiment that invokes an LLM includes at least one mandated
SOTA GGUF in `MODEL_SPECS`, resolves it through the existing cache pattern,
records hash/revision/quantization, and never passes a GGUF path to
`AutoTokenizer.from_pretrained()`.

## Promotion and retirement rules

- **Executable bank:** readiness requires 120 unique frozen IDs, four disjoint
  splits, exact dataset/test hashes, zero candidate/model access, and a private-
  test vault that cannot enter prompts or features.
- **Authentic pool:** integrity requires `K=8` raw-before-label coverage for all
  72 selector tasks, exact local model/CUDA receipts, restricted execution, and
  deterministic content-addressed resume. Correctness retries are forbidden.
- **Headroom:** hidden-state work is eligible only with both correct and
  incorrect candidates, non-saturated competence, at least 30 tasks per
  selector split, and nonzero oracle recovery beyond tuned label-free
  self-consistency/equivalence selection. If this fails, retire the exact code
  pool and skip Exp6189-Exp6191.
- **Internal selector:** promotion requires positive held task-level gain with
  a positive bootstrap lower bound, no label/test leakage, and stable
  length/surface/stratum controls. A clean repeat of the prior no-gain verdict
  retires this code-family/feature path.
- **Continuous self-learning:** promotion requires live local flagship
  generation, predecision read-only snapshots, post-outcome commits, positive
  lower intervals for both model families, retained prior-family utility,
  bounded state, immutable weights, and zero poison/rollback safety regression.
  The same null/blocked verdict triggers retirement rather than another replay.
- **Rust/PyO3:** readiness means exact transition/distribution/serialization
  parity and zero task-owned test failures. Timing may be diagnostic only and
  cannot become a TSU, FPGA, energy, or speedup claim.
- **ARC:** the fixed policy may be reported only as fresh-transition
  generalization evidence. `solve_claimed=false`, no registry delta, no source,
  BFS, adapter, prior-game, hidden-state, or per-game calibration access.
- **Evidence:** any adversarial flag, missing required field, nonterminal
  verdict, unclassified nonzero command, tracked-result mutation, or failed
  structured gate excludes the artifact from headline aggregation.

## Explicitly deferred

- Any CCTU, finite-ID answer transport, grammar/parser retry, schema-supported
  ConstraintIR reprompt, or external generated-text/logprob scorer rerun.
- Pairwise verifier co-training, CodeCircuit attribution graphs, full EBT/
  NRGPT training, KAN retraining, and weight-updating continual learning before
  the cheaper V536 tests qualify their prerequisites.
- ARC single-shot induction prompt/budget/refinement work, first-contact search
  signals, inert-click pruning, per-game adapters, offline BFS, or level-solving
  targets.
- THRML parity/scaling sweeps, two-axis tempering, FPGA/TSU latency, power,
  energy, or speedup claims without a new authenticated physical route.
- Kona benchmarking without reproducible local weights/API.
- Public documentation, publication, model upload, plugin install, remote
  message, or push operations.
