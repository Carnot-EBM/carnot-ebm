# Carnot Research Roadmap vNEXT: Durable SOTA Inference, ARC Depth, and Live Constraint Learning

**Created:** 2026-08-09
**Milestone:** 2026.08.539
**Status:** Planned; activates only after terminal archival of 2026.08.538
**Supersedes:** the V538 plan for milestone 2026.08.538, experiments Exp6211-Exp6224
**Experiment range:** Exp6225-Exp6238
**Informed by:** terminal V538 artifacts, the 2026-08-09 ARC follow-up record,
the V539 source refresh, and the current exclusion manifest

## What V538 Proved

| Track | Evidence | Finding |
|---|---|---|
| Flagship runtime | Exp6212 | All three mandated GGUF files produced short native-server token receipts, but no family had an accepted CUDA-layer receipt. `three_family_runtime_ready_score=0` and dense readiness stayed 0. The remaining failure class is process lifecycle and receipt ownership, not model availability. |
| ARC object input | Exp6213-Exp6214 | Transition-aware object input reached the canonical policy and fired on four games. Its clustered change-fidelity interval was `[-0.006, 0.01]`; promotion readiness was only 0.8. It did not earn solve credit. |
| ARC efficiency | Exp6215-Exp6216 | Object-relative trajectory transfer avoided one induction call per game with no score loss. Budget-aware search changed the synthetic paired score by +1.0 and actions by -7. Both are now default on, but neither proved deeper live progress. |
| ARC portfolio | Exp6217-Exp6218 | The long Gemma think arm was runtime-gated. The portfolio skipped with fewer than two admissible levers. Later outer-loop work fixed think routing and built prompt enrichment, but its live A/B remains incomplete. |
| Continuous self-learning | Exp6219 | Two-timescale external memory was poison-safe, rollback-exact, and free of decision-time writes. The result used deterministic Exp6145 replay and no live LLM, so it does not yet prove system learning from fresh model events. |
| Sampler | Exp6220 | The mode-jump integration remained default off. The scientific A/B did not activate all arms and was blocked by a recursive repository-wide test path. It made no quality, timing, or hardware claim. |
| Executable verification | Exp6221-Exp6223 | The raw-code canary, pool, and headroom chain gate-skipped because shared runtime readiness was 0. No new code-selection or hidden-state claim is available. |
| Reconciliation | Exp6224 | V538 closed with 2 missing, 3 nonterminal, 2 blocked, 3 skipped, and 2 flagged tasks. ARC credit and hardware claims remained zero. |

The result is useful but narrow: Carnot has safe components and several local
positive controls, yet the live flagship path is not durable enough to feed
the research loops that matter.

## The Three Largest Gaps to the PRD

1. **The local SOTA inference service is not an owned, durable substrate.**
   Short token canaries work, but long ARC jobs still lose `llama-server`, and
   the caller can wait without a bound. This blocks flagship ARC, code, and
   fresh self-learning evidence.
2. **ARC work improves first-level efficiency more than depth.** The corrected
   world-model wall remains 0 accepted engines in 69 eligible units. Think
   routing is fixed and prompt enrichment is built, but neither has a clean
   terminal determination on the current runtime. No V538 task earned a level.
3. **Continuous learning is safe but not live.** External procedural memory
   has retention, poison, and rollback controls, but Exp6219 consumed an old
   deterministic stream. The PRD requires verified adaptation from new events
   and a reachable consumer, not only replay mechanics.

Executable verification cuts across gaps 1 and 3. A small content-margin test
will determine whether code correction changes semantics or only repairs
format before Carnot pays for another large candidate pool.

## V539 Architecture: Owned Runtime to Verified Adaptation

```text
                         PHASE 0: OWN THE SUBSTRATE
┌─────────────────────────────────────────────────────────────────────┐
│ process tree + signal trace -> bounded supervisor -> endurance gate │
│                                         │                           │
│            Qwen3.6-35B-A3B  Gemma-4-31B  Gemma-4-26B-A4B          │
└─────────────────────────────────────────┬───────────────────────────┘
                                          │ owned CUDA + token receipts
                  ┌───────────────────────┴───────────────────────┐
                  │                                               │
          PHASE 1: ARC DEPTH                         PHASE 2: VERIFIED LEARNING
┌────────────────────────────────────┐   ┌──────────────────────────────────┐
│ de-confounded think determination  │   │ code correction:                │
│ prompt-enrichment held A/B         │   │   parse margin != content margin│
│ bounded re-induction A/B           │   │ fresh exact constraint events   │
│ admissible live portfolio          │   │ two-timescale procedural memory │
│                                    │   │ default-off shadow consumer      │
│ frames + own attempts only         │   │ exact verifier commits only      │
└───────────────────┬────────────────┘   └─────────────────┬────────────────┘
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       ▼
                         PHASE 3: EBM METHOD + CAPSTONE
                    ┌──────────────────────────────────┐
                    │ activated mode-jump paired test  │
                    │ equivalence-aware null handling  │
                    │ exact-path adversarial reconcile │
                    └──────────────────────────────────┘
```

The runtime gate is shared, but the ARC, verification, self-learning, and
sampler branches remain independently terminal. A blocked GPU branch cannot
turn CPU evidence into a success claim.

## Phase 0: Evidence and Durable Runtime (Exp6225-Exp6228)

### Exp6225: Exact V538-to-V539 transition

Archive the terminal V538 task identities and re-check roadmap schema,
exclusions, ID collisions, and protected files. The task records the stale
architecture date and does not pretend to reconcile that document.

### Exp6226: Post-marker evidence and scope freeze

Search only after the V539 planning marker. Record every required source
channel, including null results. Freeze model, ARC provenance, content-margin,
continual-memory, sampler-activation, and no-hardware-claim contracts.

### Exp6227: Llama-server reaper sender and wait-bound diagnostic

Instrument a deliberate, short reproduction with process-tree snapshots and
audit/eBPF signal receipts when available. No privilege escalation is allowed.
Whether the sender is identified or remains unlocalized, specify a bounded
wait and owned-process recovery contract. Do not build recovery before the
diagnostic distinguishes server death, caller hang, and external termination.

### Exp6228: Supervised three-family runtime endurance

Implement the smallest task-owned supervisor outside the conductor. Qualify
each mandated family with a real CUDA-offload receipt, repeated tokens, a
controlled owned-child termination, bounded recovery, and an endurance
window. This differs from Exp6212's 12-14 second canaries. It must never kill
an unrelated PID and must keep GGUF files read-only.

## Phase 1: ARC Induction and Depth (Exp6229-Exp6232)

Every ARC task uses only the canonical live policy, agent-visible frames, the
agent's own actions, and runtime reverse engineering. No game source, hidden
state, exhaustive offline BFS, per-game adapter, or registry mutation is
allowed. Any observed level completion must declare
`solve_provenance=live_agent_self_discovery`.

### Exp6229: De-confounded Gemma think determination

Reuse the already-launched expanded-roster run when its receipts are sound.
Resume only missing pre-registered cells under the bounded supervisor. The
primary outcome is held-out exact-admission rate with game clustering and an
exact sign test. The task freezes think on/off for downstream work even when
the result is null or underpowered.

### Exp6230: Induce-prompt enrichment held A/B

Measure the already-built default-off enrichment: semantic action names,
explicit changed-cell counts, and cross-frame object identity/topology. This
is not Exp6214's component-delta table. Use a leave-one-game-out matrix,
treatment-fire preflight, A/A control, HUD-masked symmetric change fidelity,
and live admission rate.

### Exp6231: Bounded re-induction A/B

After the think configuration is frozen, compare the current one-attempt latch
with the existing bounded re-induction flag. Use identical action budgets and
stall criteria. The primary outcome is admission or level-depth gain, not
lower action count on an already completed level.

### Exp6232: Admissible ARC depth portfolio or honest skip

Recompute eligibility from current artifacts without changing the shared
terminal classifier or erasing corrigenda. If at least two independently
admissible levers exist, run a prospective matched portfolio. Otherwise emit
a terminal skip with the exact missing evidence. This is a new prospective
test; it does not re-label Exp6218.

## Phase 2: Verification and Continuous Self-Learning (Exp6233-Exp6236)

### Exp6233: Three-family code-correction content margin

On a small frozen executable-code bank, compare no revision, format-only
normalization, and exact-diagnostic revision. Record raw generation bytes,
parse, compile, run, public-test, and sealed hidden-test outcomes. The primary
claim is the hidden-test content margin among already-parseable pairs. A gain
caused only by parse recovery is useful engineering but not verified reasoning.
This experiment applies the control from arXiv:2608.04355 without reopening
finite-ID grammar retries or a large K=8 pool.

### Exp6234: Fresh flagship exact-constraint event stream

Create a chronological, family-shifted stream from at least two mandated local
GGUF families. Each event contains the predecision snapshot, raw candidate,
exact verifier result, post-outcome commit eligibility, and immutable hashes.
Use the established exact constraint schema rather than the failed raw-code
strategy seed.

### Exp6235: Prospective two-timescale continuous learning

Run no-memory, shuffled-memory, immediate verified post-outcome commit, and
slow block-end consolidation arms on the fresh stream. Decision snapshots are
read-only. Model weights stay frozen. Require forward transfer, retention,
negative-transfer, memory-cost, poison, duplicate/reorder/stale-event,
quarantine, and exact rollback receipts.

### Exp6236: Default-off online constraint-memory shadow consumer

Only after Exp6235 is promotion-ready, wire the governed memory into the real
constraint decision path behind a default-off flag. Replay the fresh stream,
prove fail-closed fallback and mutation sensitivity, and keep all writes after
the verified outcome boundary. This is the milestone's FR11 reachability step.

## Phase 3: Activated Sampling and Reconciliation (Exp6237-Exp6238)

### Exp6237: Activated mode-jump quality and efficiency A/B

Replace Exp6220's recursive full-suite experiment path with bounded focused
science and separate repository validation. Require nonzero jump proposals and
acceptances, a synthetic multimodal positive control, matched seeded fallback,
quality metrics, ESS/autocorrelation, and wall cost. Following
arXiv:2608.05025, report positive, negative, equivalence-supported, or
inconclusive. An inactive treatment is an instrument failure.

### Exp6238: V539 exact-path adversarial capstone

Classify every task by its exact declared deliverable and current adversarial
rules. Preserve blocked, skipped, null, partial, flagged, and retired states.
Reconcile specs, traceability, status, changelog, known issues, exclusions, and
the hardware boundary. Do not create an ARC solve, hardware speedup, sampler,
code-verification, or self-learning claim that upstream evidence does not
support.

## Dependency Graph

```text
Exp6225 transition
   └── Exp6226 source and scope freeze
         ├── Exp6227 runtime diagnostic
         │      └── Exp6228 three-family endurance
         │             ├── Exp6229 think determination
         │             │      └── Exp6231 bounded re-induction
         │             ├── Exp6230 prompt-enrichment A/B
         │             │      └──────────────┐
         │             ├── Exp6233 code content margin
         │             └── Exp6234 fresh constraint stream
         │                    └── Exp6235 prospective CSL
         │                           └── Exp6236 shadow consumer
         └── Exp6237 activated mode-jump A/B

Exp6229 + Exp6230 + Exp6231 ──> Exp6232 portfolio or skip

Exp6225-Exp6237 ─────────────> Exp6238 capstone
```

Structured conductor gates are used for runtime, stream, CSL, and shadow
readiness. The portfolio task intentionally performs its dynamic eligibility
count inside the experiment because its rule is “at least two,” which cannot
be expressed as a conjunction of the current gate operators.

## Hardware Requirements

| Tasks | Hardware | Admission and evidence |
|---|---|---|
| Exp6227-Exp6230, Exp6231-Exp6234 | Two local RTX 3090 GPUs | Record both GPU UUIDs, PID ownership, free VRAM, CUDA layers, process lifetime, llama.cpp build, exact GGUF hash/revision/quantization, and raw output receipts. Never evict an unrelated process. |
| Exp6235-Exp6238 | CPU; GPU only for replay already declared by the task | No GPU claim without task-linked engagement receipts. Mode-jump science is software-only. |
| GateMate A1, KV260, PolarFire | No scheduled execution | GateMate has no new dated physical-state receipt. KV260 and PolarFire are terminal continuity assets. Repeating cached probes is excluded. |
| Extropic XTR-0/Z1 | No authenticated route | Z1 is taped out and early access is planned for 2027. The simulator API is early access. No TSU/Z1 execution, power, energy, latency, or speedup claim is allowed. |

## Model Policy

Every task that calls an LLM uses at least one mandated local GGUF. Exp6228
qualifies all three. ARC uses `unsloth/gemma-4-31B-it-GGUF`. The code
content-margin task uses all three. The fresh constraint stream uses
`unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`, with the dense family as a runtime control
when budget permits. Legacy Qwen3.5-0.8B and gemma-4-E4B-it are smoke-only and
cannot contribute headline rows.

All GGUF loads use exact cached files and embedded templates through the local
llama.cpp path. No task passes a GGUF repository directory to
`AutoTokenizer`.

## Prior-Failure and Retirement Boundaries

- Exp6227-Exp6228 name Exp6212 and change the mechanism from short canaries to
  sender tracing, bounded supervision, forced owned-child recovery, and
  endurance.
- Exp6229 names the gate-blocked Exp6217 and runs only after dense runtime is
  qualified.
- Exp6230 names Exp6214 and changes the representation from component deltas to
  semantic actions, counts, and identity/topology cross-references.
- Exp6232 names Exp6218 and uses new prospective evidence. It does not clear or
  rewrite historical corrigenda.
- Exp6233 names Exp6200 and Exp6221. It uses a small content-margin design, not
  another unchanged raw-code envelope or K=8 pool.
- Exp6234-Exp6236 name the blocked Exp6206, Exp6207, Exp6164, and Exp6165
  lineage. The new chain uses exact constraint events, a fresh qualified
  runtime, and the already-positive Exp6219 two-timescale mechanism.
- Exp6237 names Exp6166 and Exp6220. If the same blocked verdict recurs, retire
  the mode-jump runtime A/B scope.

Every listed entry sets `retire_if_same_verdict: true` in the YAML. No task
requires a retired upstream ID.

## Explicitly Deferred

- Hidden-state code selectors and another K=8 pool, until Exp6233 proves
  semantic headroom and complete executable transport.
- Prompt-phrasing, sequential refinement, gate-threshold tuning, finite-ID
  generated answers, external-text Phase D, cross-game ARC transfer, source
  reading, offline ground-truth BFS, and per-game adapters.
- Mutable GGUF weights, decision-time memory writes, and online distillation.
- New KAN training, Hyper-SET/EBT training, Kona reproduction, or a claimed
  TSU comparator.
- GateMate, KV260, PolarFire, or Z1 execution without a new admissible receipt.

## Exit Criteria

V539 succeeds as an evidence milestone when:

1. the runtime either passes owned three-family endurance or has a precise,
   reproducible blocked verdict with no dependent model claims;
2. each ARC lever has a terminal causal determination or an honest gate skip,
   with live-agent provenance and no registry mutation;
3. code correction reports format and semantic margins separately;
4. fresh flagship events feed a terminal two-timescale CSL A/B, and the shadow
   consumer runs only if its gate passes;
5. sampler evidence proves treatment activation before any comparison; and
6. the capstone preserves every non-positive state and aligns the spec and ops
   record without modifying `research-roadmap.yaml` or the conductor.
