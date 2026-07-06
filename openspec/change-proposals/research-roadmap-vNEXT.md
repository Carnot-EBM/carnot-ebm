# Research Roadmap vNEXT: 2026.07.486

Created: 2026-07-06
Milestone: 2026.07.486
Status: Planned
Milestone title: Runtime-Stable Local SOTA, Rewrite-Certified Claims, and Context-Lifecycle Self-Learning

## Inputs Read

- `CODEX.md`
- `CLAUDE.md`
- `research-program.md`
- `_bmad/prd.md`
- `_bmad/architecture.md`
- `ops/status.md`
- `ops/changelog.md`
- `research-complete.yaml`
- `research-roadmap.yaml`
- `openspec/change-proposals/`
- `ops/conductor-log.md`
- `research-references.md`
- `research-hardware-wishlist.md`
- `ops/exclusion_manifest.yaml`

## What 2026.07.485 Proved

The completed `.485` milestone closed several verifier-first loops, but it did not unlock local SOTA quality measurement.

- **Local SOTA runtime is still blocked.** Exp5309 found authenticated CUDA/offload evidence for mandated GGUF models, but no mandated model completed load, first-token, and 8-token generation. The most useful finding was concrete: the native `llama-cli` path used an unsupported conversation flag, and the Gemma MoE path hit a batch/context assertion. The next attempt must be a backend and flag bisect, not another quality run.
- **Deterministic claim and memory fixtures are clean positives.** Exp5310 proved paraphrase label preservation, contradiction detection, and invalid-premise handling on deterministic fixtures. Exp5312 proved transition-memory coverage, preservation, faithfulness, unsafe rejection, and rollback fields.
- **Adaptive memory preserved quality but did not improve final quality.** Exp5313 avoided three full verifier calls, rejected unsafe commits, and preserved quality relative to always-full verification, but final quality delta stayed at zero. The next step should measure context-object lifecycle and process failures, not only answer quality.
- **Solver guidance is useful only under symbolic authority.** Exp5314 and Exp5315 kept SAT/CDCL authoritative, preserved aggregate conflict savings on eligible cases, and blocked misleading instance classes. This remains a bounded diagnostic, not a hardware or general reasoning result.
- **KAN abstraction tightened diagnostics but not certification.** Exp5316 improved the abstraction envelope while certificate success delta stayed zero. KAN work should now target false-property sensitivity and counterexample localization rather than repeat certificate-success attempts.
- **EBT telemetry was methodologically repaired, but quality and hardware claims remain quarantined.** Exp5317 cleared the deterministic audit flag, but future energy-descent, SOTA-quality, and hardware-speedup claims are still ineligible without fresh gates.
- **SMT hint protocol needs a corrigendum.** Exp5318 showed promising deterministic acceptance/rejection fields, but the adversarial audit flagged duration and compute-bound marker confusion. It must be re-emitted cleanly before any LLM-guided SMT work builds on it.
- **Hardware remains reachability-only.** Exp5319 could not authenticate a KV260 workload, saw PolarFire status-only access, and left GateMate unchanged. No local board speedup claim exists.

## Three Biggest Gaps To PRD Vision

1. **Reliable local SOTA reasoning substrate.** The PRD requires verification against modern local open-weight models. Carnot now has mandated GGUF model names and some GPU-offload evidence, but not a stable generation receipt or quality measurement path.
2. **Continuous self-learning with certified state change.** The project has transition-memory verification and adaptive verifier dosing. It still lacks context-object lifecycle accounting, self-learning policy promotion, no-op controls, and certificate-gated rollback discipline.
3. **Certificate-bearing reasoning stack across claims, constraints, and abstractions.** Paraphrase, SMT, solver, KAN, and internal-energy work are still separate bounded diagnostics. The next milestone should connect them through typed rewrite states, solver-authoritative validation, runtime-gated internal receipts, and explicit no-claim boundaries.

## Research Incorporated

The `.486` planning refresh appended to `research-references.md` promotes these sources into this milestone:

- **Self-Evolving Agents with Anytime-Valid Certificates** (arXiv:2607.00871): use certificate gates, no-op controls, and frozen-model policy promotion for continuous self-learning.
- **Theoria** (arXiv:2607.01223): cast claim and document rewrites as typed state transitions with acceptability and complete-change checks.
- **VISTA / LLM Agents Are Latent Context Managers** (arXiv:2606.30005), **Self-GC** (arXiv:2607.00692), **A-TMA** (arXiv:2607.01935), and **AutoMem** (arXiv:2607.01224): evaluate context lifecycle, stale-state, retrieval, and answer-time failures separately from final quality.
- **Frequency-Aware Attention**, **Semantic Energy**, and **Spilled Energy** (arXiv:2602.18145, arXiv:2508.14496, arXiv:2602.18671): probe local logits/attention/internal-signal availability only after runtime is stable; do not reopen retired external generated-text scoring.
- **p-bit CDCL and FPGA p-bit annealer work** (arXiv:2605.04033, arXiv:2602.16143), plus Extropic and Logical Intelligence public writing: useful for architecture and hardware-boundary context, but not local speedup baselines.

## Target Architecture

```text
                  task / claim / memory event / constraint instance
                                      |
                                      v
        +--------------------------------------------------------+
        | Local SOTA GGUF runtime gate                          |
        | Qwen3.6-35B-A3B / Gemma-4-31B-it / Gemma-4-26B-A4B-it |
        | native llama.cpp backend + receipt matrix             |
        +--------------------------------------------------------+
                | stable text/logit/trace receipts only if gated
                v
 +-----------------------------+     +------------------------------+
 | typed rewrite verifier      |     | solver-authoritative layer   |
 | Theoria-style transitions   |<--->| Z3/CDCL/SMT/LNS/Ising hints |
 | paraphrase + claim changes  |     +------------------------------+
 +-----------------------------+                    |
                |                                   v
                |                    +------------------------------+
                |                    | KAN abstraction diagnostics  |
                |                    | false-property rejection +   |
                |                    | counterexample localization  |
                |                    +------------------------------+
                v
 +---------------------------------------------------------------+
 | context-object lifecycle and self-learning loop               |
 | Self-GC/VISTA object IDs, A-TMA current/history/transition    |
 | labels, transition verifier, SEA certificate gate, rollback,  |
 | no-op controls, policy registry, no weight mutation           |
 +---------------------------------------------------------------+
                |
                v
 +---------------------------------------------------------------+
 | artifact registry, conductor gates, exclusion discipline,     |
 | hardware reachability receipts, no unsupported quality or     |
 | speedup claims                                                |
 +---------------------------------------------------------------+
```

The central `.486` loop is context lifecycle self-learning: object-level context decisions are proposed, verified, certificate-gated, and either promoted or rolled back. Model weights remain frozen.

## Phase Plan

### Phase A: Transition, Source Refresh, And Runtime Repair

Experiments: Exp5321, Exp5322, Exp5323, Exp5324

Archive `.485`, refresh current sources, then repair the mandated local SOTA runtime path by bisecting native llama.cpp binaries, command flags, batch/context settings, and generation modes. Exp5323 is allowed to fail with a precise root cause. Exp5324 runs only if Exp5323 identifies a backend candidate and must prove repeatable first-token and 8-token receipts before any SOTA quality task can run.

### Phase B: Rewrite-Certified Claims And SMT Corrigendum

Experiments: Exp5325, Exp5326, Exp5327

Build a deterministic Theoria-style rewrite-state fixture over paraphrase and claim-verification examples. Only after both the rewrite fixture and runtime stability gate pass may Exp5326 spend local SOTA cycles on a tiny paraphrase/rewrite smoke. Exp5327 re-emits the SMT hint protocol with clean methodology fields and no compute-bound marker confusion.

### Phase C: Continuous Self-Learning Through Context Lifecycle

Experiments: Exp5328, Exp5329, Exp5330

Promote `.485` transition-memory work into context-object lifecycle self-learning. Exp5328 creates stable object IDs, current/historical/transition labels, recoverable sidecars, lifecycle actions, and preconditioned safe commits. Exp5329 measures lifecycle policy rollout against always-full verification and transition-only verification. Exp5330 adds SEA-style anytime-valid certificates, rollback, and no-op controls for policy promotion with no weight mutation.

### Phase D: Internal Signals, KAN Diagnostics, Hardware Receipts, And Capstone

Experiments: Exp5331, Exp5332, Exp5333, Exp5334

If runtime stability exists, Exp5331 checks whether the local backend can expose logits, attention, or related internal receipts for future energy diagnostics without reopening retired text-scorer work. Exp5332 moves KAN work to counterexample localization and false-property sensitivity. Exp5333 records hardware continuity with no speedup claims. Exp5334 closes the milestone with an explicit gate table and next-step decision.

## Dependency Graph

```text
exp5321 archive/activate
  -> exp5322 source refresh
  -> exp5334 capstone

exp5323 native GGUF backend/flag bisect
  -> exp5324 gated runtime receipt stabilization
      -> exp5326 gated SOTA paraphrase/rewrite smoke
      -> exp5331 gated internal-signal receipt harness

exp5325 Theoria rewrite-state fixture
  -> exp5326 gated SOTA paraphrase/rewrite smoke

exp5327 SMT hint corrigendum
  -> exp5334 capstone

exp5328 context-object lifecycle fixture
  -> exp5329 gated memory/context policy rollout
  -> exp5330 SEA anytime certificate gate

exp5332 KAN counterexample localization
  -> exp5334 capstone

exp5333 hardware continuity
  -> exp5334 capstone
```

Structured gates in `research-roadmap-next.yaml`:

- Exp5324 requires `exp5323.sota_backend_candidate_ready == true`.
- Exp5326 requires `exp5324.sota_runtime_unblocked_stable == true`.
- Exp5326 requires `exp5325.rewrite_state_fixture_ready == true`.
- Exp5329 requires `exp5328.context_lifecycle_fixture_ready == true`.
- Exp5330 requires `exp5328.context_lifecycle_fixture_ready == true`.
- Exp5331 requires `exp5324.sota_runtime_unblocked_stable == true`.

## Model And Inference Requirements

Every `.486` experiment that needs an LLM must include `MODEL_SPECS` with at least one mandated local SOTA GGUF model, and the headline runtime/quality/internal-signal tasks include all three:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as Qwen3.5-0.8B and Gemma-4-E4B-it may appear only as fast CPU smoke tests. They cannot be headline-result models. GGUF repositories must be run through llama.cpp-compatible tooling or the project cached SOTA helper path, never Hugging Face `AutoTokenizer` loading.

## Hardware Requirements

- Dual RTX 3090 CUDA host for Exp5323, Exp5324, Exp5326, and Exp5331 if they run.
- Runtime receipts must include preconditions checked, backend/binary, exact command, model spec, context and batch settings, GPU memory before/after, layer/offload evidence, timing, timeout class, and whether downstream quality claims are permitted.
- KV260 checks use `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Host `/dev/mmcblk*` checks are not valid board-state evidence.
- PolarFire checks are limited to authenticated status or workload receipts actually reachable in the environment.
- GateMate remains blocked unless the task captures physical/JTAG/toolchain evidence.
- No TSU, Kona, FPGA, or board speedup claim may be made from papers, public writing, or status-only access.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired Phase D external generated-text/logprob scorer work.
- Do not rerun the retired CPU-only llama-cpp-python GGUF offload path.
- Do not propose ARC level solves in this milestone; no ARC solve provenance is needed.
- Do not make SOTA quality claims unless the structured runtime gate passes.
- Do not make SMT, internal-energy, KAN, or hardware readiness claims from blocked, gated, or methodology-flagged artifacts.

## Expected End State

`.486` succeeds if it produces:

- A precise answer on whether native llama.cpp can run at least one mandated local SOTA GGUF model repeatably enough for quality work.
- A deterministic typed rewrite-state verifier and, only if gated, a tiny local SOTA paraphrase/rewrite smoke.
- A clean SMT hint protocol corrigendum.
- A context-object lifecycle self-learning fixture with certificate-gated policy promotion, rollback, and no-op controls.
- A runtime-gated internal-signal receipt decision that either opens or closes future energy diagnostics.
- KAN counterexample-localization evidence and hardware continuity receipts that preserve no-claim discipline.
- A capstone that makes the `.487` choice explicit: runtime quality, self-learning scale-up, or another substrate repair.
