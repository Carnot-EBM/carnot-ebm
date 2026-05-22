# Research Roadmap vNEXT: Milestone 2026.05.270

**Title:** Evidence Integrity + Dataset Materialization + Continuous Recurrence

**Planned:** 2026-05-22

**Previous milestone:** 2026.05.269

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.269 Proved

Milestone `.269` completed all scheduled tasks, but it did not produce a new paper-ready
multi-corpus headline. Its main contribution was an honest boundary around what is currently
credible.

The authoritative capstone is `results/experiment_2846_capstone_v269.json`:

- `sota_runtime_ready=true`, but the runtime artifact was adversarially flagged because the
  GGUF/CUDA evidence completed in 21 seconds and lacked complete methodology fields.
- FoVer dual-condition scoring produced `production_auroc_mean=0.9131`,
  `architecture_only_auroc_mean=0.8947`, and `learning_contribution=+0.0185`, but the artifact
  was flagged for duration and missing `random_seed`, so it is not headline-eligible.
- MBPP, HumanEval, and TruthfulQA all emitted honest `blocked_*` verdicts because the local
  datasets/splits were not materialized.
- HaluEval/FEVER reached a 50-example pilot only; the result is useful readiness evidence but not a
  full benchmark.
- BEAVER/EPR produced a bounded-prefix proxy only, not exact BEAVER soundness, and the artifact was
  flagged by the same compute-bound provenance issue.
- LoopUS/FR-11 self-learning blocked on a missing `live_recurrence_backend`.
- Cross-corpus matrix and paper table artifacts were missing or gate-blocked in the authoritative
  capstone, so `paper_ready=false`.

The lesson is specific: the next milestone should not add another speculative verifier before the
evidence chain is clean. It should first clear provenance flags, materialize local data, rerun the
blocked corpus rows, and then run one bounded continuous self-learning loop.

## Three Biggest Gaps

### Gap 1: Evidence Integrity Is Blocking Every Headline Claim

The runtime and FoVer artifacts contain useful signals, but they are currently excluded by
adversarial verification. `.270` must separate genuine live-model evidence from fast dataset-only
scoring, avoid GGUF/CUDA claims in non-live artifacts, and include seed/checksum/methodology fields.

### Gap 2: Local Benchmark Manifests Are Missing

The PRD vision requires verifiable reasoning beyond FoVer. `.269` showed that MBPP, HumanEval, and
TruthfulQA cannot even start until local manifests exist with counts, checksums, and split metadata.
Materialization is therefore a first-class Phase A task, not something buried inside each corpus
runner.

### Gap 3: Continuous Self-Learning Still Has No Live Recurrence Backend

FR-11 is the center of the PRD, but `.269` blocked before running a single recurrence example. The
next milestone splits this into a backend adapter and then a LoopUS-style pilot. That keeps the
failure mode diagnosable: backend unavailable, recurrence energy not improving, or self-learning
helping/hurting.

## New Research Integrated

The 2026-05-22 post-`.269` sweep added the following planning signals to `research-references.md`:

- **ConstraintBench** (arXiv:2602.22465): feasibility is the main bottleneck in direct constrained
  optimization. `.270` therefore reports per-constraint readiness and dataset checksums before
  aggregate accuracy.
- **Residual Drift / DriftBench** (OpenReview ICLR 2026 Workshop; arXiv:2604.28031): models can keep
  a satisfiable ledger while violating prior commitments in their assignment. `.270` adds a
  residual-drift/MUS conflict diagnostic after the matrix is rebuilt.
- **HGNN-MUSE** (arXiv:2604.09001; AISTATS 2026): hypergraph structure can reduce MUS enumeration
  cost. `.270` starts with a cheap HGNN-inspired prioritizer, not a full RL agent.
- **LoopUS** (arXiv:2605.11011): latent recurrence and adaptive early exit motivate an external
  recurrence loop for FR-11, split into adapter and pilot.
- **EBT / ARM-as-EBM / CEM** (arXiv:2507.02092, 2512.15605v3, 2605.07588): theory anchors only for
  `.270`; clean local evidence remains the bottleneck.
- **Extropic TSU and Logical Intelligence Kona**: support the long-term hardware thesis, but no
  `.270` task claims TSU/Kona hardware access or latency.

## Architecture Snapshot

```text
                         +------------------------------------+
                         |  Phase A: evidence and data gates  |
                         |                                    |
                         |  exp2848 SOTA runtime evidence v2  |
                         |  exp2849 local dataset manifests   |
                         +-----------------+------------------+
                                           |
                 +-------------------------+-------------------------+
                 |                                                   |
                 v                                                   v
     +---------------------------+                       +---------------------------+
     | Phase B: clean corpora    |                       | Phase C: recurrence       |
     |                           |                       |                           |
     | exp2850 FoVer integrity   |                       | exp2856 LoopUS backend    |
     | exp2851 MBPP              |                       | exp2857 FR-11 pilot       |
     | exp2852 HumanEval         |                       +-------------+-------------+
     | exp2853 TruthfulQA        |                                     |
     | exp2854 HaluEval/FEVER    |                                     |
     +-------------+-------------+                                     |
                   |                                                   |
                   v                                                   v
        +-----------------------+                         +-------------------------+
        | exp2855 matrix v4     |                         | exp2858 BEAVER proxy    |
        | clean rows only       |                         | exp2859 Drift/MUS       |
        +-----------+-----------+                         +------------+------------+
                    |                                                  |
                    +--------------------------+-----------------------+
                                               |
                                               v
                              +--------------------------------+
                              | exp2860 capstone v270          |
                              | claim boundary + next actions  |
                              +--------------------------------+
```

## Phase Structure

### Phase A: Archive, Runtime Evidence, and Dataset Materialization

- `exp2847` archives `.269` and activates `.270`.
- `exp2848` reruns SOTA runtime evidence with real wall-clock inference or an honest blocked verdict.
- `exp2849` creates local manifests for MBPP, HumanEval, TruthfulQA, HaluEval, and FEVER.

This phase fixes the two root causes behind most `.269` failures: provenance flags and missing local
datasets.

### Phase B: Clean Corpus Measurements

- `exp2850` reruns FoVer dual-condition scoring without live-model overclaiming.
- `exp2851` reruns MBPP dual-condition generation and verification, gated on runtime and dataset
  readiness.
- `exp2852` reruns HumanEval full dual-condition generation and verification, gated the same way.
- `exp2853` reruns TruthfulQA generation-split dual-condition scoring, gated the same way.
- `exp2854` scales HaluEval/FEVER from pilot to full dataset-only calibration.
- `exp2855` rebuilds the cross-corpus matrix from available clean rows only.

Headline eligibility requires no adversarial flags, non-null corpus rows, meaningful sample sizes,
and explicit SOTA GGUF model specs for LLM-bearing tasks.

### Phase C: Continuous Recurrence and New Diagnostics

- `exp2856` implements or selects a live recurrence backend adapter.
- `exp2857` runs the mandatory continuous self-learning task: a LoopUS-style external recurrence
  pilot with energy/correctness deltas and no model-weight mutation.
- `exp2858` reruns the BEAVER/EPR bounded-prefix proxy with honest labeling and no live-model
  provenance overclaim.
- `exp2859` builds a residual-drift plus MUS conflict-prioritization diagnostic from the `.270`
  matrix and ledger rows.

This phase connects the literature sweep to concrete artifacts while preserving the claim boundary:
BEAVER remains a proxy unless exact frontier bounds are implemented, and HGNN-MUSE remains a
prioritization heuristic unless a trained HGNN policy is actually added.

### Phase D: Capstone and Claim Boundary

- `exp2860` synthesizes `.270`, classifies clean/blocked/flagged artifacts, decides whether paper-v6
  Section 5 can be regenerated, and writes the next action list.

The capstone is ungated so the milestone always produces an honest terminal summary even when Phase A
or B blocks.

## Dependency Graph

```text
exp2847
  -> exp2848
       -> exp2851
       -> exp2852
       -> exp2853
       -> exp2856
            -> exp2857

exp2847
  -> exp2849
       -> exp2851
       -> exp2852
       -> exp2853
       -> exp2854

exp2850 + exp2851 + exp2852 + exp2853 + exp2854
  -> exp2855
       -> exp2859

exp2858 is independent after exp2847.

all artifacts, including blocked states
  -> exp2860
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2851` gates on `exp2848.sota_runtime_ready_v2 == true` and `exp2849.mbpp_ready == true`.
- `exp2852` gates on `exp2848.sota_runtime_ready_v2 == true` and `exp2849.humaneval_ready == true`.
- `exp2853` gates on `exp2848.sota_runtime_ready_v2 == true` and `exp2849.truthfulqa_ready == true`.
- `exp2854` gates on `exp2849.halueval_ready == true` and `exp2849.fever_ready == true`.
- `exp2856` gates on `exp2848.sota_runtime_ready_v2 == true`.
- `exp2857` gates on `exp2856.live_recurrence_backend_ready == true`.
- `exp2859` gates on `exp2855.cross_corpus_matrix_built == true`.
- `exp2860` is intentionally ungated.

## Hardware Requirements

Required:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` loader with GPU offload evidence.
- At least one mandated local SOTA GGUF cached and loadable:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- Local storage for MBPP, HumanEval, TruthfulQA, HaluEval, and FEVER manifests plus generated rows.

Not required:

- KV260 board execution, Vivado synthesis, GateMate, PolarFire, AMD NPU, D-Wave, photonic hardware,
  or Extropic TSU/Z1/XTR-0 access.
- Any TSU/Kona latency claim. Hardware references in `.270` remain strategic context only.

## Agent Routing

- `codex/gpt-5.5`: formulaic code, dataset materialization, SOTA preflight, benchmark runners, and
  diagnostics.
- `claude/opus`: capstone synthesis only.
- `gemini` is not used in this roadmap because the local audit rejects `agent_type: gemini`.

## Acceptance Criteria

1. `exp2848` records real runtime evidence for at least one mandated SOTA GGUF, or emits an honest
   `blocked_*` verdict before any downstream live-model task runs.
2. `exp2849` writes local manifest paths, counts, and checksums for every target corpus, with explicit
   booleans consumed by gates.
3. Every LLM-bearing task includes at least one mandated SOTA GGUF in `MODEL_SPECS`.
4. Legacy small models appear only as CPU smoke-test fallbacks and never as headline models.
5. No artifact claims GGUF/CUDA/live-model provenance unless it actually invokes the model and records
   seed/checksum/methodology evidence.
6. FoVer, MBPP, HumanEval, TruthfulQA, and HaluEval/FEVER either produce clean rows or honest
   `blocked_*` verdicts with `preconditions_checked`.
7. `exp2857` satisfies the milestone self-learning mandate by setting
   `continuous_self_learning_task=true`.
8. `prior_failures` are present on every scope-matched retry and every entry includes
   `retire_if_same_verdict: true`.
9. `exp2860` explicitly states whether any `.270` row is paper-ready; absent or flagged rows remain
   excluded.
