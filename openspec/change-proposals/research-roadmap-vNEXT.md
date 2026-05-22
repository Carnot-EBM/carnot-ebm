# Research Roadmap vNEXT: Milestone 2026.05.273

**Title:** Clean Telemetry + Fast/Slow Memory + Constraint Benchmark Expansion

**Planned:** 2026-05-22

**Previous milestone:** 2026.05.272

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.272 Proved

Milestone `.272` strengthened the paper-ready state but left two important
claims fenced off. The authoritative terminal artifact is
`results/experiment_2884_capstone_v272.json`.

- `paper_ready=true`, with 7 clean artifacts, 2 flagged artifacts, 1 blocked
  artifact, 1 pilot-only artifact, and 0 missing artifacts.
- The SOTA runtime corrigendum is clean for a single mandated local GGUF:
  `unsloth/gemma-4-26B-A4B-it-GGUF` generated 8 usable GPU-backed responses
  with complete provenance. Two-model `cached_sota_pair()` readiness remains
  false.
- The SOTA micro-panel produced 6 non-empty responses with logprobs, but the
  artifact is still flagged because duration/provenance were not citation-grade
  and no reproducibility checksum was recorded.
- The KAN PWA/MILP corrigendum cleared the `.271` tautology: local and global
  error bounds are distinct, Z3 returned `optimal`, and exact enumeration was not
  used as a fallback. The claim remains a tiny formal example, not a general KAN
  verifier.
- Exact frontier checking touched HaluEval/FEVER without overclaiming, but only
  8 of 1000 rows were exactly supported. HaluEval/FEVER remains a headline row
  only as a weak calibration/data point.
- MBPP and HumanEval now have deterministic manifest-only execution pilots, but
  they are explicitly pilot-only. TruthfulQA is still missing from matrix v6.
- FR-11 RecMem recurrence-triggered consolidation is clean as a trigger
  prototype. The scale-up artifact is flagged: it reports zero forgetting,
  zero correctness/AUROC delta, and very short runtime, so it cannot support a
  clean FR-11 scale-up claim.
- THRML sampler portability is blocked locally because THRML is unavailable; the
  fallback ran and made no hardware claim.

## Three Biggest Gaps

### Gap 1: Telemetry Claims Still Fail the Citation Bar

The local SOTA runtime is finally usable, but the micro-panel remains flagged.
The next milestone must either produce adversarial-clean telemetry with fixed
seeds, checksum, duration, raw responses, and token/logprob evidence, or
permanently downgrade the micro-panel to a non-benchmark note.

### Gap 2: Cross-Corpus Evidence Has Not Escaped Pilot Status

FoVer and HaluEval/FEVER are still the only headline rows. MBPP/HumanEval are
pilot-only, and TruthfulQA is absent. The next step is not a giant benchmark; it
is a bounded clean promotion path: materialize TruthfulQA as an error-taxonomy
manifest, run a small generated-code row only under clean local GGUF runtime, and
add structural-dependency verification before expanding matrix v7.

### Gap 3: FR-11 Needs Non-Tautological Fast/Slow Memory Evidence

RecMem's trigger is promising, but `.272` did not prove that recurrence-triggered
memory improves correctness or energy relative to eager replay. Recent
multi-timescale-memory work suggests a better comparator: fast episodic edges
plus slow consolidated edges. `.273` should test RecMem against an explicit
fast/slow baseline with real drift, duplicate, contradiction, energy, AUROC, and
forgetting checks.

## New Research Integrated

The 2026-05-22 post-`.272` sweep appended these items to
`research-references.md`:

- **EBT accepted as an ICLR 2026 oral** (OpenReview `ZBj3Qp1bYg`): strengthens
  Carnot's theory citation for energy-minimization as System-2 thinking.
- **NRGPT** (OpenReview `B3Muyi2zgo`): useful theory context for energy descent
  in language modeling, but not evidence for Carnot until local telemetry is
  clean.
- **CCTU** (arXiv:2603.15309): executable constraint validation for multi-turn
  tool use; a natural next benchmark pilot for Carnot's constraint verifier.
- **Structural Verification for EDA Code Generation** (arXiv:2604.18834):
  dependency-graph contracts before execution, directly relevant to
  MBPP/HumanEval promotion.
- **VeriCoT** (arXiv:2511.04662 / ICLR 2026): neuro-symbolic CoT validation with
  solver-backed logical consistency; useful for exact-frontier expansion.
- **InFi-Check** (arXiv:2601.06666): fine-grained fact-checking labels,
  evidence, justifications, and corrections; useful for TruthfulQA taxonomy
  materialization.
- **Memini** (arXiv:2605.05097): multi-timescale fast/slow external memory
  dynamics; direct comparator for the flagged RecMem scale-up.
- **KAN hardware complexity metrics** (arXiv:2604.03345) and **analog KAN
  hardware** (arXiv:2602.07518): next KAN step should be cost accounting, not
  hardware overclaim.
- **Extropic THRML / TSU status**: THRML remains the public software bridge, but
  `.273` should not rerun THRML parity unless the dependency is materialized.
- **llguidance**: optional local constrained-decoding engine for structured
  extractor outputs; keep it outside the core unless a local integration proves
  useful.
- **Logical Intelligence Aleph/Kona updates**: supports the formal verification
  direction, but Carnot must cite only local reproducible artifacts as evidence.

## Architecture Snapshot

```text
             +--------------------------------------------------+
             | Phase A: close flagged .272 evidence             |
             |                                                  |
             | exp2885 archive/activate                         |
             | exp2886 SOTA micro-panel clean telemetry v3      |
             | exp2887 FR-11 fast/slow memory corrigendum       |
             +----------------------+---------------------------+
                                    |
             +----------------------+---------------------------+
             |                                                  |
             v                                                  v
  +------------------------------+              +------------------------------+
  | Phase B: corpus promotion    |              | Phase C: formal constraints  |
  |                              |              |                              |
  | exp2888 TruthfulQA taxonomy  |              | exp2891 CCTU validator pilot |
  | exp2889 generated code row   |              | exp2892 VeriCoT frontier     |
  | exp2890 structural verifier  |              | exp2893 KAN cost accounting  |
  +---------------+--------------+              +---------------+--------------+
                  |                                             |
                  +---------------------+-----------------------+
                                        |
                                        v
                    +-------------------------------------------+
                    | Phase D: matrix + paper boundary + close  |
                    |                                           |
                    | exp2894 cross-corpus matrix v7            |
                    | exp2895 paper-v6 evidence table v4        |
                    | exp2896 capstone                          |
                    +-------------------------------------------+
```

## Phase Structure

### Phase A: Close Flagged `.272` Evidence

- `exp2885` archives `.272` and activates `.273`.
- `exp2886` reruns the SOTA energy/logprob micro-panel with an adversarial-clean
  checksum/duration/provenance requirement. It remains bounded and cannot claim a
  full benchmark.
- `exp2887` is the mandatory continuous self-learning task. It compares RecMem
  recurrence-triggered consolidation with a fast/slow Memini-style baseline and
  eager replay, requiring non-tautological metrics before any FR-11 scale-up
  claim.

### Phase B: Corpus Promotion

- `exp2888` materializes a TruthfulQA InFi-Check-style error taxonomy manifest
  from local labels/artifacts without remote LLM calls.
- `exp2889` attempts a small generated-code MBPP/HumanEval row using the clean
  local SOTA runtime and mandated GGUF model specs. If generation or sandboxing
  is not clean, it writes a blocked artifact rather than upgrading the pilot.
- `exp2890` builds a structural-dependency graph verifier for code tasks,
  inspired by the EDA structural-verification paper. It is a deterministic
  verifier layer, not another generated-code benchmark.

### Phase C: Formal Constraints and Hardware-Cost Accounting

- `exp2891` runs a tiny CCTU-style executable constraint validator pilot with
  no new LLM generation.
- `exp2892` expands exact frontier checking with a VeriCoT-style logical-step
  parser/prover on a bounded set of locally supported rows.
- `exp2893` adds KAN hardware-oriented complexity accounting (RM/BOP/NABS-style)
  to the clean tiny KAN PWA/MILP example, keeping analog/FPGA claims out of
  scope.

### Phase D: Matrix, Paper Boundary, and Capstone

- `exp2894` rebuilds cross-corpus matrix v7 from clean artifacts only, preserving
  pilot-only and missing-row boundaries.
- `exp2895` writes a paper-v6 evidence table / claim-boundary artifact gated on
  matrix v7.
- `exp2896` synthesizes `.273`, classifies clean/flagged/blocked artifacts, and
  decides whether the next milestone should scale evidence or keep correcting
  artifacts.

## Dependency Graph

```text
exp2885
  -> exp2886
  -> exp2887
  -> exp2888
       -> exp2894
  -> exp2889
       -> exp2894 (consumed opportunistically if clean)
  -> exp2890
       -> exp2894
  -> exp2891
       -> exp2894
  -> exp2892
       -> exp2894
  -> exp2893

exp2894
  -> exp2895

all clean/flagged/blocked side artifacts
  -> exp2896
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2894` gates on:
  - `exp2888.truthfulqa_taxonomy_ready == true`
  - `exp2890.structural_dependency_verifier_ready == true`
  - `exp2891.cctu_validator_ready == true`
  - `exp2892.vericot_frontier_ready == true`
- `exp2895` gates on `exp2894.cross_corpus_matrix_built == true`.
- `exp2896` is intentionally ungated so the milestone can close honestly even if
  one branch is blocked.

## Hardware Requirements

Required for live-model tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` with GPU offload support.
- At least one loadable mandated SOTA GGUF, with the `.272` clean single-model
  runtime artifact as the starting point:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`

Required for non-live tasks:

- Local Python environment, existing `.272` result artifacts, local eval
  manifests, Z3 where already used by Carnot, and repository test tooling.

Optional:

- `llguidance` for local schema/grammar constrained extraction experiments. If
  absent, tasks must use deterministic local fallbacks.

Not required:

- THRML installation, Extropic TSU/Z1/XTR-0 access, KV260 board execution,
  Vivado synthesis, GateMate, PolarFire, AMD NPU, D-Wave, photonic hardware, or
  Logical Intelligence Kona/Aleph access.

## Agent Routing

- `codex/gpt-5.5`: formulaic code, deterministic validators, manifest work,
  structural graph verifier, VeriCoT-style parser/prover scaffolding, KAN cost
  accounting, matrix/table synthesis, and archive bookkeeping.
- `claude/opus`: live local-GGUF telemetry and capstone synthesis, where
  environment evidence and artifact discipline dominate.
- `gemini` is not used because `ops/known-issues.md` still records Gemini
  routing as paused due upstream 429/rate-limit failures.

## Decentralization Implications

The milestone preserves local-first execution. Closed-weight providers are not
required for any experiment. Live-generation tasks use local mandated GGUFs;
benchmark and memory tasks use local artifacts/manifests; hardware work stays at
software cost-accounting and blocked-dependency reporting rather than remote
hardware claims.

## Acceptance Criteria

1. `exp2886` either clears the `.272` SOTA micro-panel flag with reproducibility
   checksum, duration, raw rows, token/logprob evidence, and no benchmark
   overclaim, or permanently downgrades it to a non-benchmark telemetry note.
2. `exp2887` sets `continuous_self_learning_task=true` and reports
   non-tautological eager-vs-RecMem-vs-fast/slow memory metrics.
3. `exp2888` creates a TruthfulQA taxonomy manifest without citing fabricated
   `exp2823` or synthesizing labels.
4. `exp2889` promotes MBPP/HumanEval only if local generated-code evidence,
   sandbox status, and labels/tests are clean; otherwise it remains blocked or
   pilot-only.
5. `exp2890` produces a deterministic structural-dependency verifier for code
   rows.
6. `exp2891` produces a CCTU-style executable constraint-validation pilot.
7. `exp2892` expands exact-frontier support without autoformalization overclaim.
8. `exp2893` reports KAN complexity metrics without analog/FPGA execution claims.
9. `exp2894` leaves unsupported rows null and preserves pilot-only boundaries.
10. Every LLM-bearing task includes at least one mandated SOTA GGUF in
    `MODEL_SPECS`.
11. Legacy small models are allowed only for CPU smoke tests and cannot become
    headline models.
12. `exp2896` reports `paper_ready` only from clean artifacts and names residual
    flagged or blocked evidence explicitly.
