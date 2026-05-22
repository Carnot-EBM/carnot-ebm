# Research Roadmap vNEXT: Milestone 2026.05.272

**Title:** Clean SOTA Corrigenda + RecMem Self-Learning + Evidence Expansion

**Planned:** 2026-05-22

**Previous milestone:** 2026.05.271

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.271 Proved

Milestone `.271` got Carnot back to a paper-ready state, but only by being
honest about the remaining weak links. The authoritative terminal artifact is
`results/experiment_2872_capstone_v271.json`.

- `paper_ready=true`, with 8 clean artifacts, 0 blocked artifacts, and 0 missing
  artifacts.
- FoVer and HaluEval/FEVER are headline-eligible clean rows. The HaluEval/FEVER
  row is real but weak: AUROC is low enough that it should be treated as a
  calibration/data point, not a win.
- The dated local eval manifest contract works. `.270`'s plain-vs-dated path
  drift is no longer the main blocker.
- Exact tiny Z3 arithmetic frontier, residual-drift/MUS prioritization, and
  offline recurrence backend are present as clean building blocks.
- FR-11 replay now has an offline result: energy decreased without forgetting on
  a tiny replay set, but there was no correctness lift and the sample size was
  too small for a durable self-learning claim.
- The SOTA runtime path is no longer totally blocked: one mandated GGUF
  (`unsloth/gemma-4-26B-A4B-it-GGUF`) produced usable GPU-backed output.
  However, the artifact was adversarially flagged because the two-model
  `cached_sota_pair()` contract was not satisfied and the clean provenance bar
  was not high enough for citation.
- The SOTA energy micro-panel invoked the live model, but empty responses and
  unavailable logprob fields prevented a full benchmark claim.
- The KAN PWA/MILP tiny verifier produced a useful prototype, but was flagged
  because the local and global error bounds were identical and the MILP path fell
  back to exact enumeration.

The next milestone should not open a new frontier before it cleans these three
flagged artifacts. It should turn `.271`'s paper-ready but fragile state into a
stronger multi-corpus, self-learning, and formal-verifier evidence base.

## Three Biggest Gaps

### Gap 1: SOTA Runtime Provenance Is Usable But Not Clean

The PRD requires local, open, decentralized evaluation. `.271` showed the host
can run a mandated SOTA GGUF on GPU, but the result is not clean enough to cite:
only one mandated model is cached, `cached_sota_pair()` still fails the two-model
contract, and token/logprob telemetry is incomplete. Live-model work must stay
gated on a clean runtime corrigendum.

### Gap 2: Cross-Corpus Evidence Is Still Too Narrow

The matrix is no longer empty, but it has only FoVer and HaluEval/FEVER. Code
rows, TruthfulQA, and more exact-verification evidence remain absent or weak. The
next milestone should add verifier-localization metrics, a small exact frontier
extension, and a manifest-only code row before any larger benchmark claim.

### Gap 3: FR-11 Self-Learning Needs Recurrence Control and Drift Guards

The PRD's autonomous self-learning loop is not satisfied by a tiny replay that
only lowers energy. The new RecMem result suggests the right next step:
consolidate only after sustained recurrence, measure token savings, and add
memory-drift/non-forgetting checks. The cautionary memory-corruption result makes
those drift guards mandatory.

## New Research Integrated

The 2026-05-22 post-`.271` sweep appended these items to
`research-references.md`:

- **Distributional EBMs for structured LLM reasoning** (arXiv:2605.18871):
  separates deterministic constraints, learned quality scores, and uncertainty.
  This supports `.272`'s claim boundary between structural verifiers and learned
  model priors.
- **RecMem** (arXiv:2605.16045): recurrence-triggered memory consolidation is
  the direct FR-11 implementation target.
- **Useful Memories Become Faulty When Continuously Updated by LLMs**
  (arXiv:2605.12978): self-learning must report drift and non-forgetting, not
  just token savings.
- **LaaB logical consistency bridge** (arXiv:2605.03971), **VERGE**
  (arXiv:2601.20055), **Energy-Based Constraint Networks** (OpenReview/TMLR
  2026), and **HalluGuard** (HF/arXiv:2601.18753): all point toward error
  localization and decomposed hallucination diagnostics.
- **Energy-guided decoding** (arXiv:2507.07731) and **ARM-as-EBM**
  (arXiv:2512.15605): useful theory for the SOTA energy micro-panel, but only
  after clean local telemetry exists.
- **KAN PWA/MILP verification** (arXiv:2602.06737): remains the formal KAN target,
  now with a stricter non-tautological artifact requirement.
- **Extropic THRML / TSU** and **Logical Intelligence Kona** updates: maintain the
  hardware and global-energy north stars, but `.272` should stay software-local
  unless THRML is already installed.

## Architecture Snapshot

```text
             +-----------------------------------------------+
             | Phase A: clean flagged .271 artifacts          |
             |                                               |
             | exp2873 archive/activate                      |
             | exp2874 SOTA runtime clean corrigendum        |
             | exp2875 SOTA micro-panel logprob corrigendum  |
             | exp2876 KAN PWA/MILP corrigendum              |
             +---------------------+-------------------------+
                                   |
             +---------------------+-------------------------+
             |                                               |
             v                                               v
  +-----------------------------+              +-----------------------------+
  | Phase B: evidence expansion |              | Phase C: FR-11 RecMem       |
  |                             |              |                             |
  | exp2877 exact frontier v2   |              | exp2881 recurrence trigger  |
  | exp2878 label consistency   |              | exp2882 replay scale-up     |
  | exp2879 code corpus pilot   |              +--------------+--------------+
  | exp2880 matrix v6           |                             |
  +-------------+---------------+                             |
                |                                             |
                +---------------------+-----------------------+
                                      |
                                      v
                  +-------------------------------------------+
                  | Phase D: sampler portability + capstone   |
                  |                                           |
                  | exp2883 THRML sampler portability smoke   |
                  | exp2884 capstone                          |
                  +-------------------------------------------+
```

## Phase Structure

### Phase A: Clean Flagged `.271` Artifacts

- `exp2873` archives `.271` and activates `.272`.
- `exp2874` reruns the SOTA runtime path as an adversarially clean corrigendum.
  It must prove GPU-backed mandated GGUF inference and record whether the two-model
  cache contract is satisfied.
- `exp2875` runs the SOTA energy micro-panel only if `exp2874` is clean. It fixes
  the empty-response/logprob gap or writes a clean `blocked_logprobs_unavailable`
  verdict without pretending it is a benchmark.
- `exp2876` repairs the KAN PWA/MILP artifact. Success requires distinct
  local/global error bounds and either a real MILP backend or an explicit blocked
  solver verdict.

### Phase B: Evidence Expansion

- `exp2877` extends exact frontier checking from FoVer-only arithmetic into a tiny
  HaluEval/FEVER safe-prefix or contradiction subset.
- `exp2878` adds a label-consistency/error-verifiability audit over HaluEval/FEVER,
  separating data-driven and reasoning-driven failure buckets where possible.
- `exp2879` creates a manifest-only MBPP/HumanEval pilot row using deterministic
  local tests and verifier metadata. It does not require SOTA generation.
- `exp2880` rebuilds the cross-corpus matrix v6 from clean rows only. Missing rows
  remain null and are named as residual gaps.

### Phase C: Continuous Self-Learning with RecMem

- `exp2881` implements a recurrence-triggered memory consolidation prototype for
  FR-11 using the offline recurrence backend from `.271`. This is the mandatory
  continuous self-learning experiment for the milestone.
- `exp2882` scales the replay evaluation only if `exp2881` proves the trigger is
  ready. It measures token reduction, AUROC/correctness deltas, energy deltas,
  memory-drift score, and forgetting regressions.

### Phase D: Hardware-Compatible Sampler Path and Capstone

- `exp2883` runs a software-local THRML or fallback sampler portability smoke for
  Carnot's Ising/PGM abstraction. It keeps the hardware path warm without claiming
  access to Extropic TSU hardware.
- `exp2884` synthesizes `.272`, classifies clean/flagged/blocked artifacts, and
  decides what can safely enter paper-v6 or the next milestone.

## Dependency Graph

```text
exp2873
  -> exp2874
       -> exp2875
  -> exp2876
  -> exp2877
       -> exp2880
  -> exp2878
       -> exp2880
  -> exp2879
       -> exp2880
  -> exp2881
       -> exp2882
  -> exp2883

exp2880 and exp2882, plus all clean/blocked side artifacts
  -> exp2884
```

Structured gates in `research-roadmap-next.yaml`:

- `exp2875` gates on `exp2874.sota_runtime_clean == true`.
- `exp2880` gates on:
  - `exp2877.frontier_expansion_ready == true`
  - `exp2878.error_verifiability_ready == true`
  - `exp2879.code_manifest_pilot_ready == true`
- `exp2882` gates on `exp2881.recmem_trigger_ready == true`.
- `exp2884` is intentionally ungated so the milestone can close honestly even if
  one branch is blocked.

## Hardware Requirements

Required for runtime-gated live tasks:

- Dual RTX 3090 CUDA host through `.venv/bin/python`.
- `llama_cpp` with GPU offload support.
- At least one loadable mandated SOTA GGUF for clean single-model evidence, and
  two loadable mandated SOTA GGUFs for a clean `cached_sota_pair()` claim:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`

Required for non-live tasks:

- Local Python environment, existing `.271` result artifacts, local eval manifests,
  Z3 where already used by Carnot, and repository test tooling.

Optional:

- `thrml` and JAX for `exp2883`. If unavailable, `exp2883` must write a clean
  blocked dependency artifact and a fallback Ising/PGM parity result if possible.

Not required:

- KV260 board execution, Vivado synthesis, GateMate, PolarFire, AMD NPU, D-Wave,
  photonic hardware, Extropic TSU/Z1/XTR-0 access, or Logical Intelligence Kona
  access.

## Agent Routing

- `codex/gpt-5.5`: formulaic code, verifier implementations, dataset pilots,
  KAN/MILP scaffolding, matrix synthesis, RecMem implementation, and THRML smoke
  code.
- `claude/opus`: GPU runtime corrigenda and capstone synthesis, where environment
  evidence and artifact discipline dominate.
- `gemini` is not used because `ops/known-issues.md` still records Gemini routing
  as paused due upstream 429/rate-limit failures.

## Acceptance Criteria

1. `exp2874` either emits a clean SOTA runtime artifact with real mandated GGUF
   GPU-backed output or a specific blocked verdict with every precondition
   checked.
2. `exp2875` does not claim a SOTA energy benchmark unless non-empty responses and
   token/logprob or explicit substitute telemetry are present.
3. `exp2876` clears the `.271` KAN tautology by proving non-identical error bounds
   and reporting the MILP backend status honestly.
4. `exp2880` builds only from clean upstream rows and leaves missing metrics null.
5. `exp2881` sets `continuous_self_learning_task=true` and implements
   recurrence-triggered consolidation with memory hashes and token-cost metrics.
6. `exp2882` reports non-forgetting and memory drift, not just energy improvement.
7. Every LLM-bearing task includes at least one mandated SOTA GGUF in
   `MODEL_SPECS`.
8. Legacy small GGUFs are allowed only for CPU smoke tests and cannot become
   headline models.
9. `exp2884` reports `paper_ready` only from clean artifacts and names any residual
   flagged or blocked evidence explicitly.
