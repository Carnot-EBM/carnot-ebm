# Research Roadmap vNEXT: Milestone 2026.05.271

**Title:** Runtime Repair + Manifest Reconciliation + Offline Self-Learning

**Planned:** 2026-05-22

**Previous milestone:** 2026.05.270

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.270 Proved

Milestone `.270` completed, but it did not create a new paper-ready multi-corpus
headline. Its value is diagnostic: it narrowed the critical path to four concrete
blockers that can be attacked without inventing a new research direction.

The authoritative capstone is `results/experiment_2860_capstone_v270.json`:

- Local dataset materialization works. `exp2849` wrote dated local manifests for
  MBPP, HumanEval, TruthfulQA, HaluEval, and FEVER with checksums and non-synthetic
  row counts.
- The downstream HaluEval/FEVER runner did not consume those manifests because it
  looked for plain `data/eval_manifests/halueval.jsonl` and `fever.jsonl` while
  the materializer wrote `halueval_20260522.jsonl` and `fever_20260522.jsonl`.
- FoVer produced a clean dataset-only dual-condition row:
  `production_auroc_mean=0.9131336`, `architecture_only_auroc_mean=0.8946624`,
  and `learning_contribution=+0.0184712`, with no live-model provenance claim.
- SOTA local runtime remains blocked. CUDA torch sees two RTX 3090s, but
  `llama_cpp.llama_supports_gpu_offload()` returned false and `cached_sota_pair()`
  returned no two-model mandated local SOTA pair.
- Cross-corpus evidence is still absent: the `.270` matrix was honest but not built
  because only FoVer was clean.
- FR-11 continuous self-learning is still blocked at the recurrence backend layer:
  `exp2856` never produced an artifact, so `exp2857` could not run.
- BEAVER/EPR produced a clean bounded-prefix proxy on FoVer labels, but it remains
  a proxy, not exact BEAVER frontier evidence.

The lesson is not "rerun every blocked benchmark." The next milestone should first
repair the local runtime/cache and manifest contracts, then generate at least one
clean non-FoVer row and one offline FR-11 recurrence result that do not depend on
live SOTA inference.

## Three Biggest Gaps

### Gap 1: Local SOTA Inference Is Still Not Operational

The PRD depends on decentralized, local open-model evaluation. `.270` proved the
host has CUDA and two RTX 3090s, but the GGUF loader/cache path is not usable:
`llama_cpp` lacks GPU offload support and `cached_sota_pair()` resolves no required
pair. Until this is fixed, every live LLM benchmark must be structurally gated.

### Gap 2: Non-FoVer Evidence Is Blocked by Contracts, Not Concepts

The data is present. The failure is path/schema coordination between materializer
and consumers. The next milestone must create a stable manifest resolver/contract,
rerun HaluEval/FEVER against actual dated manifests, and rebuild the matrix from
clean rows only.

### Gap 3: FR-11 Needs a Replayable Recurrence Backend Before Live Loops

Continuous self-learning remains central to the PRD, but `.270` showed that tying
the recurrence backend to live SOTA runtime creates a cascade failure. `.271`
therefore splits recurrence into an offline replay adapter and a gated FR-11 replay
experiment. Live LoopUS-style recurrence can resume after the runtime path is clean.

## New Research Integrated

The 2026-05-22 post-`.270` sweep added these planning signals to
`research-references.md`:

- **Spilled Energy** (arXiv:2602.18671; ICLR 2026) and **First Token Knows**
  (arXiv:2605.05166): low-cost logit/energy hallucination baselines should precede
  expensive multi-sample self-consistency.
- **Error Verifiability** (arXiv:2604.04418): report whether verifier outputs make
  errors easier to localize, not just whether AUROC moves.
- **ChopChop** (arXiv:2509.00360) and **RWOPD for NL-to-SVA**
  (arXiv:2605.13501): semantic constrained decoding and verifier-rewarded
  distillation are now concrete precedents for verifier-in-the-loop code and
  hardware-property work.
- **KAN PWA/MILP verification** (arXiv:2602.06737): KAN work should become a tiny
  exact abstraction verifier, not another black-box classifier.
- **Ising/FPGA decomposition**, **REASON**, and **Extropic THRML**:
  hardware acceleration supports the long-term substrate thesis, but `.271` should
  stay software-only because KV260, GateMate, and PolarFire tracks are terminal.
- **Semantic Scholar citation watch** for EBT and ARM-as-EBM: follow-ons emphasize
  recurrence, output-side constraints, and energy-scored trajectory repair.
- **Logical Intelligence Kona**: useful architecture framing for globally scored,
  partial-trace energy evaluation; no local Kona access or latency claim is made.

## Architecture Snapshot

```text
                         +---------------------------------------+
                         | Phase A: root contracts              |
                         |                                       |
                         | exp2861 archive/activate             |
                         | exp2862 SOTA runtime/cache resolver  |
                         | exp2863 manifest resolver contract   |
                         +-------------------+-------------------+
                                             |
                +----------------------------+-----------------------------+
                |                                                          |
                v                                                          v
     +---------------------------+                         +---------------------------+
     | Phase B: clean evidence   |                         | Phase C: replay learning |
     |                           |                         |                           |
     | exp2864 HaluEval/FEVER    |                         | exp2868 recurrence backend|
     | exp2865 matrix v5         |                         | exp2869 FR-11 replay      |
     | exp2866 exact BEAVER tiny |                         +-------------+-------------+
     | exp2867 drift/MUS v2      |                                       |
     +-------------+-------------+                                       |
                   |                                                     |
                   +---------------------------+-------------------------+
                                               |
                                               v
                          +----------------------------------------+
                          | Phase D: gated live + formal checks    |
                          |                                        |
                          | exp2870 SOTA energy micro-panel        |
                          | exp2871 KAN PWA/MILP verifier          |
                          | exp2872 capstone                       |
                          +----------------------------------------+
```

## Phase Structure

### Phase A: Archive, Runtime, and Manifest Contracts

- `exp2861` archives `.270` and activates `.271`.
- `exp2862` replaces the retired `.270` SOTA runtime attempt with a concrete
  cache/offload resolver. It may fix code or environment documentation, but
  `sota_runtime_ready_v3=true` only if a mandated GGUF actually produces usable
  GPU-backed output.
- `exp2863` reconciles dated manifest paths with downstream consumers and emits
  a stable manifest contract for MBPP, HumanEval, TruthfulQA, HaluEval, and FEVER.

### Phase B: Clean Non-FoVer Rows and Failure Diagnostics

- `exp2864` reruns HaluEval/FEVER full calibration against the manifest contract.
- `exp2865` rebuilds the cross-corpus matrix from FoVer plus any clean non-FoVer
  rows. Missing rows remain null; no metrics are inferred.
- `exp2866` converts the BEAVER/EPR proxy into a tiny exact-frontier feasibility
  check where possible, or writes a blocked solver/dependency verdict.
- `exp2867` reruns the residual-drift plus MUS prioritizer only if the matrix is
  actually built.

### Phase C: Offline Continuous Self-Learning

- `exp2868` implements/selects an offline replay recurrence backend that can
  consume existing FoVer/HaluEval verifier traces without live LLM inference.
- `exp2869` is the mandatory FR-11 continuous self-learning task. It runs a
  bounded replay loop, records memory hashes before/after, and proves whether
  energy/correctness improves without model-weight mutation.

### Phase D: Runtime-Gated Live Evidence and Formal Abstraction

- `exp2870` runs a small SOTA GGUF micro-panel only if `exp2862` proves runtime
  readiness. It reports first-token confidence and spilled-energy-style baselines
  before any heavier sampling.
- `exp2871` implements a tiny KAN piecewise-affine/MILP verifier or writes an
  honest blocked dependency artifact.
- `exp2872` synthesizes `.271`, classifies clean/blocked/flagged artifacts, and
  decides whether paper-v6 Section 5 has enough non-FoVer evidence to regenerate.

## Dependency Graph

```text
exp2861
  -> exp2862
       -> exp2870

exp2861
  -> exp2863
       -> exp2864
            -> exp2865
                 -> exp2867
                 -> exp2869

exp2868
  -> exp2869

exp2866 and exp2871 are independent after exp2861.

all artifacts, including blocked states
  -> exp2872
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2864` gates on `exp2863.manifest_contract_ready == true`.
- `exp2865` gates on `exp2864.halueval_fever_ready == true`.
- `exp2867` gates on `exp2865.cross_corpus_matrix_built == true`.
- `exp2869` gates on `exp2868.offline_recurrence_backend_ready == true` and
  `exp2865.cross_corpus_matrix_built == true`.
- `exp2870` gates on `exp2862.sota_runtime_ready_v3 == true`.
- `exp2872` is intentionally ungated.

## Hardware Requirements

Required for runtime-gated live tasks only:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` built with GPU offload support.
- At least one loadable mandated SOTA GGUF, with `cached_sota_pair()` returning a
  two-model pair when possible:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`

Required for all other tasks:

- Local Python environment and existing repository data/artifacts.
- Local manifests under `data/eval_manifests/`.
- Existing FoVer and FR-11 state files.

Not required:

- KV260 board execution, Vivado synthesis, GateMate, PolarFire, AMD NPU, D-Wave,
  photonic hardware, Extropic TSU/Z1/XTR-0 access, or Logical Intelligence Kona
  access.

## Agent Routing

- `codex/gpt-5.5`: formulaic code, manifest reconciliation, dataset pipelines,
  cross-corpus synthesis, BEAVER/KAN prototypes, and diagnostics.
- `claude/opus`: SOTA runtime/cache resolver and capstone synthesis, because the
  runtime path mixes environment state, GPU evidence, and artifact discipline.
- `gemini` is not used because `ops/known-issues.md` keeps Gemini routing paused
  due upstream 429/rate-limit failures.

## Acceptance Criteria

1. `exp2862` either records a real mandated GGUF GPU-backed usable response or
   emits a specific `blocked_*` verdict with all preconditions checked.
2. `exp2863` writes a manifest contract consumed by HaluEval/FEVER and future
   corpus tasks, eliminating plain-vs-dated path drift.
3. `exp2864` creates at least one clean non-FoVer artifact or an honest blocked
   artifact that names the exact missing resource.
4. `exp2865` builds a cross-corpus matrix only from clean upstream rows and leaves
   unavailable metrics null.
5. `exp2869` sets `continuous_self_learning_task=true`, records memory hashes, and
   measures energy/correctness deltas without mutating model weights.
6. Every LLM-bearing task includes at least one mandated SOTA GGUF in `MODEL_SPECS`.
7. Legacy small models appear only as CPU smoke-test fallbacks and never as headline
   models.
8. No artifact claims GGUF/CUDA/live-model provenance unless it actually invokes a
   model and records seed/checksum/methodology evidence.
9. `exp2872` reports `paper_ready` strictly from clean artifacts and lists any
   residual blocked or missing rows.
