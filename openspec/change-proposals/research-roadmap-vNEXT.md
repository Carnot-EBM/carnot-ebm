# Research Roadmap vNEXT: 2026.07.487

Created: 2026-07-07
Milestone: 2026.07.487
Status: Planned
Milestone title: Utility-Governed Context Self-Learning and Structured Local SOTA Verification

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

## What 2026.07.486 Proved

The completed `.486` milestone changed the local-SOTA branch from "runtime blocked" to "runtime usable but output protocol not yet trustworthy," while giving the self-learning branch the clearest clean positive evidence.

- **Local native SOTA runtime is finally stable for one mandated model.** Exp5323 and Exp5324 found and repeated a native `llama-cli` path for `unsloth/gemma-4-31B-it-GGUF`, with three bounded load/first-token/8-token receipts and authenticated GPU memory/offload evidence. Exp5323 remains adversarial-flagged for duration/methodology, so `.487` must clean the receipt before using it as a headline substrate.
- **SOTA structured-output quality was measured and failed.** Exp5326 ran bounded local SOTA paraphrase/rewrite prompts, but `paraphrase_label_preservation_rate=0.0` and `rewrite_acceptability_rate=0.0`. The failure mode was parse/protocol: the model stayed in a thinking transcript and did not emit compact JSON within the token budget. The next quality task must repair structured-output protocol before expanding sample count.
- **Deterministic rewrite and SMT gates are clean.** Exp5325 produced a six-case typed rewrite-state fixture with full acceptability, complete-change, and unsafe-rewrite rejection rates. Exp5327 re-emitted the SMT hint protocol with deterministic solver substrate, actual timing, and no compute-bound marker confusion.
- **Continuous self-learning has the strongest scale-up path.** Exp5328 built a context-object lifecycle fixture with 1.0 detection rates for bank/retrieval/answer failures and rollback. Exp5329 matched always-full quality while avoiding three verifier calls. Exp5330 promoted one lifecycle policy, rejected three, deferred one, kept unsafe promotions at zero, and preserved frozen-model discipline.
- **Internal-signal work is open but flagged.** Exp5331 found token probabilities, token timing, and raw receipts, but no logits/attention/hidden-state proxies. It was adversarial-flagged for short duration/methodology, so `.487` may clean the receipt and run only a tiny diagnostic, not a text-scorer rerun.
- **KAN localization is useful but bounded.** Exp5332 rejected deterministic false-property perturbations and localized counterexample regions with 1.0 accuracy while preserving true-property certificates. Certificate success did not improve, so the next step is counterexample-to-constraint injection, not a repeat certificate-success attempt.
- **Hardware remains no-speedup continuity.** Exp5333 found KV260 SSH unreachable, PolarFire SSH status-only reachable, and GateMate unchanged at the physical/JTAG level. No authenticated workload or speedup claim exists.

## Three Biggest Gaps To PRD Vision

1. **Modern local model verification is not yet semantically useful.** Carnot can now run a mandated local GGUF model, but it cannot yet reliably extract structured verified outputs from that model. The PRD's verifiable-reasoning vision needs a stable "generate structured candidate -> deterministic checker -> repair or reject" path.
2. **Self-learning is still fixture-scale and policy-only.** The clean `.486` self-learning win avoided verifier calls and safely promoted one policy, but it has not learned utility values across episodes, compressed bounded state, or handled drift/poisoning across sessions.
3. **Certificates remain isolated diagnostics.** Rewrite-state checks, SMT, solver guidance, KAN localization, token-probability energy, and hardware receipts are still separate. The PRD needs a certificate-bearing stack where counterexamples, provenance, temporal/spatial constraints, and energy signals feed actionable verification decisions without overclaiming.

## Research Incorporated

The `.487` planning refresh appended to `research-references.md` promotes these ideas into the milestone:

- **MemRL** (arXiv:2601.03192): non-parametric runtime reinforcement learning on episodic memory; use utility-weighted retrieval without model-weight mutation.
- **Agent Cognitive Compressor** (arXiv:2601.11653): separate artifact recall from persistent state commitment; test bounded state and memory-drift anomalies.
- **Scaling the Harness** (arXiv:2605.26112): report harness-level metrics such as memory hygiene, verifier cost, context efficiency, and safe evolution, not only final answer quality.
- **Field-Theoretic Memory** (arXiv:2602.21220): use decay/coupling as a bounded simulator diagnostic for pruning policies, not as a replacement for hash-chained context identity.
- **QSTRBench** (arXiv:2605.18380): add qualitative temporal/spatial constraint fixtures with solver-authoritative Allen/RCC-style checks.
- **OpenViking**: treat filesystem-style context objects as an implementation contrast for IDs and hierarchical context loading; do not add a dependency.

Previously recorded but still active constraints include G-RRM, ProvenanceGuard, CiteTracer, BEAVER, In-Writing, CEM, Spilled Energy, Semantic Energy, Extropic TSU writing, and Logical Intelligence Kona/Aleph public updates. Semantic Scholar EBT/ARM-EBM API checks returned HTTP 429 on 2026-07-07, so no citation-count delta is claimed.

## Target Architecture

```text
           local task, claim, memory event, or constraint case
                                  |
                                  v
 +----------------------------------------------------------------+
 | Clean local SOTA GGUF runtime receipt                          |
 | Gemma-4-31B-it stable path; Qwen/Gemma MoE probed if feasible  |
 | duration/methodology clean before downstream quality           |
 +----------------------------------------------------------------+
                  |                            |
                  v                            v
 +--------------------------------+   +--------------------------------+
 | Structured output protocol     |   | Token-probability energy       |
 | post-think JSON extraction,    |   | receipt-clean Spilled/Semantic |
 | parse gates, no headline claim |   | energy diagnostic, no scorer   |
 +--------------------------------+   +--------------------------------+
                  |
                  v
 +----------------------------------------------------------------+
 | Deterministic certificate layer                                 |
 | rewrite-state fixture + SMT + QSTR temporal/spatial checker +   |
 | solver-authoritative G-RRM-style overwrite/fallback telemetry   |
 +----------------------------------------------------------------+
                  |
                  v
 +----------------------------------------------------------------+
 | Utility-governed context self-learning                          |
 | ContextNest identity/hash/audit rows, MemRL utility values,     |
 | ACC bounded state, drift/poisoning monitors, certificate gate,  |
 | rollback, no-op controls, no model-weight mutation              |
 +----------------------------------------------------------------+
                  |
                  v
 +----------------------------------------------------------------+
 | KAN/Ising counterexample-to-constraint bridge + hardware        |
 | continuity receipts with no speedup claim                       |
 +----------------------------------------------------------------+
```

The `.487` center of gravity is continuous self-learning scale-up. Local SOTA work is necessary, but it is protocol repair and bounded measurement, not a headline benchmark.

## Phase Plan

### Phase A: Transition, Source Refresh, Runtime Cleanup, Structured Output

Experiments: Exp5335, Exp5336, Exp5337, Exp5338, Exp5339

Archive `.486`, refresh sources, then clean the Exp5323/5324 runtime receipts so the stable Gemma 4 31B path is not adversarial-flagged. Exp5338 repairs structured-output protocol by comparing prompt/flag/parse variants that extract final compact JSON after Gemma thinking. Exp5339 reruns a small paraphrase/rewrite/citation panel only after the protocol gate passes, with no headline quality claim.

### Phase B: Continuous Self-Learning Scale-Up

Experiments: Exp5340, Exp5341, Exp5342

Turn `.486` lifecycle policy promotion into utility-governed self-learning. Exp5340 adds MemRL-style utility values over memory/context operations. Exp5341 adds ACC-style bounded state and drift/poisoning monitors. Exp5342 combines both under provenance-bound, hash-chained, cross-session context governance with no-op controls, rollback, and frozen model discipline.

### Phase C: Deterministic Constraints, Solver Guidance, Internal Energy, KAN Bridge

Experiments: Exp5343, Exp5344, Exp5345, Exp5346

Add QSTRBench-style qualitative temporal/spatial constraints with an authoritative checker. Revisit solver guidance only as overwrite/fallback telemetry, following G-RRM's lesson that the symbolic solver remains the authority. Clean the token-probability internal-energy receipt and run a tiny non-headline Spilled/Semantic Energy diagnostic if the receipt is valid. Convert KAN counterexample localization into concrete constraint cuts against bounded fixtures.

### Phase D: Hardware Continuity And Capstone

Experiments: Exp5347, Exp5348

Record hardware status without speedup claims. KV260 remains SSH-only, PolarFire may run a hash-verified workload only if reachable, and GateMate needs fresh physical/JTAG evidence before another detect loop matters. Exp5348 closes `.487` with gate tables for structured SOTA, self-learning, constraints, internal energy, KAN bridge, and hardware.

## Dependency Graph

```text
exp5335 archive/activate
  -> exp5336 source refresh
  -> exp5348 capstone

exp5337 clean local SOTA runtime receipt
  -> exp5338 structured-output protocol calibration
      -> exp5339 bounded SOTA claim/rewrite panel
  -> exp5345 token-probability energy corrigendum

exp5340 utility-weighted context memory
  -> exp5342 provenance-bound self-learning scale-up

exp5341 bounded compressor and drift monitor
  -> exp5342 provenance-bound self-learning scale-up

exp5343 QSTR temporal/spatial fixture
  -> exp5344 solver-guidance overwrite/fallback telemetry
  -> exp5346 KAN/Ising counterexample-to-constraint bridge

exp5347 hardware continuity
  -> exp5348 capstone
```

Structured gates in `research-roadmap-next.yaml`:

- Exp5338 requires `exp5337.sota_runtime_clean_receipt_ready == true`.
- Exp5339 requires `exp5338.structured_output_protocol_ready == true`.
- Exp5342 requires `exp5340.utility_memory_ready == true`.
- Exp5342 requires `exp5341.compressor_drift_fixture_ready == true`.
- Exp5344 requires `exp5343.qstr_fixture_ready == true`.
- Exp5345 requires `exp5337.sota_runtime_clean_receipt_ready == true`.
- Exp5346 requires `exp5343.qstr_fixture_ready == true`.

## Model And Inference Requirements

Every `.487` experiment that invokes an LLM must include `MODEL_SPECS` with the mandated local SOTA GGUF models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The stable `.486` path is `unsloth/gemma-4-31B-it-GGUF` via native `llama-cli`. Exp5337 may attempt Qwen/Gemma MoE variants only as runtime-expansion receipts. Legacy small models may appear only as CPU smoke tests and cannot be headline-result models. GGUF repositories must be loaded through llama.cpp-compatible paths or the cached SOTA helper; never use Hugging Face `AutoTokenizer` on GGUF-only repos.

## Hardware Requirements

- Dual RTX 3090 CUDA host for Exp5337, Exp5338, Exp5339, and Exp5345 if their gates pass.
- Runtime artifacts must record exact command, model path, context/batch/ubatch, GPU layers, split mode, token budget, duration, first-token latency, token count, GPU memory before/during/after, offload evidence, and whether quality claims are permitted.
- KV260 checks use only `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; host block devices are invalid evidence.
- PolarFire may record SSH status and a hash-verified board-local workload if reachable; no speedup comparison is allowed.
- GateMate work is opportunistic unless the physical/JTAG setup changed; repeating the same unchanged detect failure is not useful.
- Extropic/TSU and Logical/Kona public material remains architecture context only without local authenticated execution.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired Phase D external generated-text/logprob scorer work.
- Do not rerun the retired CPU-only llama-cpp-python GGUF offload path.
- Do not claim headline SOTA quality from the bounded structured-output panel.
- Do not claim token-probability energy, KAN, solver-guidance, or hardware readiness from blocked or flagged artifacts.
- Do not propose ARC level solves in this milestone.

## Expected End State

`.487` succeeds if it produces:

- A clean local SOTA runtime receipt and a structured-output protocol that can extract parseable final JSON from the stable mandated GGUF path.
- A bounded SOTA claim/rewrite panel that reports actual fixture quality without headline claims.
- A utility-weighted, bounded-state, provenance-bound context self-learning scale-up with no model-weight mutation, no-op controls, rollback, and unsafe-promotion zero.
- Deterministic qualitative temporal/spatial constraints and solver-guidance telemetry that keep symbolic checkers authoritative.
- A cleaned token-probability energy receipt or a precise blocked reason, without reopening retired text-scorer work.
- A KAN/Ising counterexample-to-constraint bridge and hardware continuity artifact that preserve no-claim discipline.
- A capstone decision for `.488`: structured SOTA quality scale-up, self-learning utility scale-up, or internal-energy/hardware cleanup.
