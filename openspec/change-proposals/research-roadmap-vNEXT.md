# Research Roadmap vNEXT: 2026.07.485

Created: 2026-07-06
Milestone: 2026.07.485
Status: Planned
Milestone title: Memory-Transition Verification, Paraphrase-Stable Claims, and Ising/KAN Certificate Gates

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

## What 2026.07.484 Proved

The completed `.484` milestone closed the immediate post-SOTA migration loop, but did not unlock SOTA quality measurement.

- **Local SOTA runtime remains the critical blocker.** Exp5297 changed substrate to native `llama.cpp` and showed CUDA/offload memory deltas for the mandated GGUF families, but all mandated models still timed out. Exp5298 correctly gate-blocked the coherence/trace smoke instead of spending a quality run on an unproven runtime.
- **Adaptive memory is the strongest positive result.** Exp5302 and Exp5303 showed a governed adaptive memory policy can preserve quality while avoiding full verifier calls under clean and stress conditions, with zero unsafe false accepts in the fixture.
- **Solver guidance works when gated by instance class.** Exp5299 made the LNS solver fixture usable, and Exp5300 preserved conflict savings by blocking misleading p-bit/CDCL instance classes. Symbolic fallback stayed authoritative.
- **KAN abstraction improved diagnostics but not final certificate success.** Exp5304 improved dynamic spot-check hit rate and reduced envelope gaps, but certificate success did not improve.
- **EBT telemetry is useful but must be audited.** Exp5301 produced an EBT spectral step-control diagnostic, but the conductor flagged the artifact for duration/methodology. `.485` should repair the telemetry record before building on it.
- **Hardware remains reachability-only.** Exp5305 found KV260 unreachable over SSH, PolarFire status-only reachable, and GateMate still physically/JTAG blocked. No speedup claim exists.

## Three Biggest Gaps To PRD Vision

1. **Reliable local SOTA reasoning path.** The PRD needs verifiable reasoning against modern open-weight LLMs. Carnot has mandated local GGUF model names and offload evidence, but no completed flagship inference/quality run. The next milestone must separate runtime root cause from downstream quality gates.
2. **Continuous self-learning beyond outcome scoring.** `.484` proved adaptive memory dosing can be safe on small fixtures. The missing PRD step is transition-level learning: every memory write/revise/delete needs coverage, preservation, and faithfulness checks before persistent state changes.
3. **Certificate-bearing energy/constraint stack.** Solver, p-bit, EBT, and KAN components are still bounded diagnostics. The next milestone should tighten them with smooth Ising relaxation baselines, solver-authoritative hint validation, optimal KAN abstraction budgets, and audited EBT workload telemetry.

## Research Incorporated

The planning refresh appended to `research-references.md` promotes these sources into `.485`:

- TrustMem, arXiv:2606.25161: memory transition verifier for coverage, preservation, and faithfulness.
- Constrained Paraphrase Consistency, arXiv:2606.08158: paraphrase-invariance and label-preservation constraints for hallucination detection.
- Optimal KAN abstractions, arXiv:2602.06737: dynamic programming plus knapsack allocation for PWA abstraction budgets.
- Local-minima-preserving Ising relaxation, arXiv:2606.30333: smooth one-flip-minima-preserving relaxation for tiny CPU Ising baselines.
- LLM-guided quantified SMT and inductive-constraint work, arXiv:2601.04675 and arXiv:2603.03668: LLM hints must remain solver-validated, with overwrite/fallback telemetry.
- p-bit dual-BRAM FPGA annealer, arXiv:2602.16143, plus Extropic TSU public writing: useful for hardware boundary metadata only, not local speedup claims.
- Spilled Energy, Semantic Energy, and CRV OpenReview work: internal/logit/circuit verifier directions gated behind a local runtime that exposes logits or hidden-state traces; they do not reopen retired Phase D external generated-text scoring.

## Target Architecture

```text
                   user task / claim / constraint instance
                                  |
                                  v
              +------------------------------------------+
              | SOTA local GGUF runtime gate             |
              | Qwen3.6-35B-A3B / Gemma-4-31B-it /      |
              | Gemma-4-26B-A4B-it                       |
              +------------------------------------------+
                 | text/logits/trace only if runtime unblocks
                 v
      +-----------------------------+       +-----------------------------+
      | paraphrase-stable claim     |       | solver-authoritative         |
      | and coherence verifier      |<----->| constraint layer             |
      +-----------------------------+       | Z3/CDCL/LNS/Ising hints      |
                 |                          +-----------------------------+
                 |                                        |
                 v                                        v
      +-----------------------------+       +-----------------------------+
      | memory transition verifier  |       | KAN/PWA certificate budget   |
      | coverage/preservation/      |       | DP + knapsack + MILP checks  |
      | faithfulness before commit  |       +-----------------------------+
      +-----------------------------+                     |
                 |                                        |
                 v                                        v
      +---------------------------------------------------------------+
      | artifact registry, conductor gates, exclusion discipline,     |
      | hardware reachability receipts, no-speedup/no-quality claims  |
      +---------------------------------------------------------------+
```

The loop that matters for FR-11 is the memory verifier: verified memory transitions update the agent state, and future retrieval/verifier-dose decisions are scored against process-level memory labels, not only final task answers.

## Phase Plan

### Phase A: Transition, Source Refresh, Runtime Triage

Experiments: Exp5307, Exp5308, Exp5309

Archive `.484`, activate `.485`, refresh SOTA references, then do a runtime root-cause matrix for the mandated GGUF families. Exp5309 is not a headline quality task. It must distinguish load, prompt ingestion, first-token, token-throughput, context length, and timeout class. If it cannot prove a fresh runtime-unblocked receipt, downstream LLM quality tasks remain gate-skipped.

### Phase B: Paraphrase-Stable Claims And Gated SOTA Quality

Experiments: Exp5310, Exp5311

Build a deterministic paraphrase-consistency fixture first, using solver/verifier labels and no LLM headline claim. Only after the runtime gate and fixture gate pass may Exp5311 run the mandated local SOTA models on a tiny coherence/trace smoke. This avoids repeating `.484`'s blocked quality run.

### Phase C: Continuous Self-Learning Through Memory Transitions

Experiments: Exp5312, Exp5313

Promote `.484` adaptive memory from call-saving policy to transition-level self-learning. Exp5312 builds the verifier and labels omission, corruption, hallucinated update, stale retention, and rollback. Exp5313 gates on that verifier and measures whether adaptive memory keeps quality while improving process-level transition scores and avoiding unsafe commits.

### Phase D: Solver, Energy, KAN, And SMT Certificate Work

Experiments: Exp5314, Exp5315, Exp5316, Exp5317, Exp5318

Add the smooth Ising relaxation baseline next to p-bit/CDCL, then run a gated solver ablation that keeps CDCL authoritative. Advance KAN verification from dynamic spot-checking to optimal PWA piece allocation. Re-emit EBT telemetry with methodology and workload counters before reusing it. Add a solver-authoritative SMT hint protocol fixture so future LLM-guided conjectures are judged by validity/usefulness/fallback, not by raw model confidence.

### Phase E: Hardware Continuity And Capstone

Experiments: Exp5319, Exp5320

Continue hardware receipts without speedup claims. KV260 checks use SSH only, PolarFire remains status-only unless an authenticated workload exists, GateMate remains blocked without physical/JTAG evidence, and Extropic/TSU references remain roadmap context. The capstone reconciles all gates and explicitly records whether SOTA quality, memory transition learning, solver/KAN certificates, EBT telemetry, and hardware reachability are ready for `.486`.

## Dependency Graph

```text
exp5307 archive/activate
  -> exp5308 source refresh
  -> exp5320 capstone

exp5309 SOTA runtime root-cause
  -> exp5311 gated SOTA paraphrase/coherence smoke

exp5310 paraphrase fixture
  -> exp5311 gated SOTA paraphrase/coherence smoke

exp5312 memory transition verifier
  -> exp5313 gated adaptive memory rollout

exp5314 Ising smooth relaxation
  -> exp5315 gated solver guidance ablation

exp5316 KAN optimal abstraction
  -> exp5320 capstone

exp5317 EBT telemetry audit
  -> exp5320 capstone

exp5318 SMT hint protocol
  -> exp5320 capstone

exp5319 hardware continuity
  -> exp5320 capstone
```

Structured gates in `research-roadmap-next.yaml`:

- Exp5311 requires `exp5309.sota_runtime_unblocked == true`.
- Exp5311 requires `exp5310.paraphrase_fixture_ready == true`.
- Exp5313 requires `exp5312.memory_transition_verifier_ready == true`.
- Exp5315 requires `exp5314.smooth_relaxation_ready == true`.

## Model And Inference Requirements

Any `.485` experiment that needs an LLM must include `MODEL_SPECS` with at least one of the mandated local SOTA GGUF models, and the headline tasks include all three:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as Qwen3.5-0.8B and Gemma-4-E4B-it may appear only as fast CPU smoke tests. They cannot be headline-result models. GGUF repositories must be run through llama.cpp-compatible tooling or the project cached SOTA helper path, not Hugging Face `AutoTokenizer` loading.

## Hardware Requirements

- Dual RTX 3090 CUDA host for Exp5309 and any gated SOTA quality run.
- Runtime receipts must include preconditions checked, command form, backend, layer/offload evidence, memory delta, timing, timeout class, and whether quality claims are permitted.
- KV260 board checks use `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Host `/dev/mmcblk*` checks are not valid board-state evidence.
- PolarFire checks are limited to authenticated status or workload receipts actually reachable in the environment.
- GateMate remains blocked unless the task captures physical/JTAG/toolchain evidence.
- No TSU, Kona, or board speedup claim may be made from papers or public writing alone.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired Phase D external generated-text/logprob scorer work.
- Do not rerun the retired CPU-only llama-cpp-python GGUF offload path without a changed runtime substrate and fresh GPU-offload receipt.
- Do not propose ARC level solves in this milestone; no ARC solve provenance is needed.
- Do not make hardware, SOTA quality, or EBT readiness claims from blocked, gated, or methodology-flagged artifacts.

## Expected End State

`.485` succeeds if it produces:

- A clean decision on whether the local SOTA runtime is unblocked enough for `.486` quality experiments.
- A deterministic paraphrase-consistency verifier fixture and, if gated, a tiny SOTA smoke result.
- A transition-level continuous self-learning memory verifier with rollout evidence.
- Solver/KAN/EBT artifacts that tighten certificates or telemetry without weakening symbolic fallback.
- Hardware continuity receipts that preserve no-speedup discipline.
- A capstone that makes the `.486` choice obvious instead of broadening the search space.
