# Research Roadmap vNEXT: 2026.07.466

## Milestone Title

PHASE D FINAL PROCESS-VERIFIER CHECK + GUARDED FR-11 MEMORY EVOLUTION

## Why This Milestone Exists

Milestone `2026.06.465` did not produce a defensible positive claim. Its capstone verdict was
`complete_capstone_v465_execution_incomplete_fr11_no_credible_positive_evidence_missing_sota`.

What it did prove:

1. **D1 is bounded, not open-ended.** The trained/SOTA MuSR verifier signal reached `+0.080`, but the
   paired CI touched zero and the result was flagged. The later outer-loop GPU-1 decisive D1 test in
   `ops/known-issues.md` tightened this further: `+0.015`, CI95 `[-0.060, +0.085]`, McNemar `p=0.78`,
   despite oracle@K headroom. Do not spend `.466` re-skeletoning D1.
2. **D4's old second-corpus evidence is retired.** `results/experiment_5060_second_corpus_audit_v2.json`
   reported `retired_d4_second_corpus_audit_failed_constraintbench_exact_v1_plus_0p370`; the apparent
   off-MuSR win came from a duplicate/unclean corpus path.
3. **D6 is an efficiency lead, not an accuracy moat yet.** Tool-first cascade parity at zero judge calls
   is valuable, but the accuracy CI still touches zero; it should be framed as Pareto/cost evidence.
4. **Guided decoding has a real but tiny signal.** `.465` found differentiated arms and a guided
   `+0.111` point estimate, but the sample was too small for a headline. The rerank-only arm was stronger
   than guided, so `.466` must scale the frontier rather than assume decoding-time guidance wins.
5. **FR-11 self-learning remains negative.** Skillgraph self-learning was correctly guarded and produced
   `complete_guarded_no_promote_minus_0p050`; the learning loop needs retrieval/memory governance and
   rollback, not another blind replay promotion attempt.
6. **SOTA hygiene regressed operationally.** The reserved SOTA ingestion slot failed because the task was
   routed to unavailable `gemini-3.1-pro-preview`. Some `.465` artifacts were also too fast for their
   claimed compute path. `.466` must repair agent routing and substrate truth before new claims.
7. **Hardware continuity improved but remains non-headline.** KV260 SSH overlay execution produced a
   parity/timing packet, but no speedup or scale claim is justified. GateMate and PolarFire remain visible
   targets needing precheck/terminal-state clarity.

The next milestone therefore has one scientific purpose: **decide whether the last distinct
process-verifier lever (uPRM/VPR with real SOTA GGUF telemetry) can produce a proper win, while moving
FR-11 from harmful replay toward guarded memory evolution.**

## Three Biggest Gaps Versus The PRD

### Gap 1: Verifier Moat Is Still Not Realized

PRD FR-12 requires verifiable reasoning that improves reliability. D1 and D4 are no longer credible
headline candidates. The only remaining distinct verifier-moat path is process-level supervision from
token logprobs or objective intermediate checks:

- uPRM (`arXiv:2605.10158`) over a real SOTA GGUF logprob cache.
- VPR (`arXiv:2605.10325`) only where intermediate verification is auditable and oracle boundaries are
  explicit.
- D6/tool-first cascade as an efficiency axis, not a hidden judge.
- DCCD/guided decoding as generation-time constraint control, measured against rerank-only.

### Gap 2: Runtime Provenance Is Not Reliable Enough For Claims

PRD-grade evidence needs live/cached provenance that survives adversarial review. `.465` exposed three
operational gaps: missing SOTA ingestion, DURATION_TOO_SHORT flags, and unavailable top-logprob/confidence
telemetry despite cached SOTA models. `.466` must stage model/cache/endpoint readiness before compute-bound
experiments and must avoid claiming `live_llm_inference` unless a real inference path ran.

### Gap 3: Continuous Self-Learning Is Not Yet Helpful

PRD FR-11 calls for an autonomous self-learning loop. The last two attempts produced negative held-out
deltas (`exp5051`, `exp5064`). Recent work (LifelongAgentBench, EvolveMem, MUSE-Autoskill) suggests the
natural next step: guarded group self-consistency, retrieval-config evolution, explicit rollback, and
non-forgetting. `.466` should produce either a small guarded positive or a concrete memory-gap ledger that
explains why promotion remains unsafe.

## Fresh Research Folded In

Added to `research-references.md` before this plan:

- `arXiv:2605.10158` - Unsupervised Process Reward Models.
- `arXiv:2605.10325` - Verifiable Process Rewards for Agentic Reasoning.
- `arXiv:2603.03305` plus `github.com/avinashreddydev/dccd` - Draft-Conditioned Constrained Decoding.
- `arXiv:2602.06737` - Optimal Abstractions for Verifying KANs.
- `arXiv:2505.11942` - LifelongAgentBench.
- Hugging Face Papers `2605.13941` / `github.com/aiming-lab/SimpleMem` - EvolveMem.
- Hugging Face Papers `2605.27366` - MUSE-Autoskill.
- `arXiv:2602.15985` - FPGA decomposition for large Ising problems.
- Extropic XTR-0/TSU public updates and Logical Intelligence Kona/Aleph public positioning.

Semantic Scholar citation checks for EBT/ARM were attempted, but the public API returned HTTP 429, so no
new citation-count claim is included.

## Architecture For .466

```text
                         research-references.md
                                  |
                                  v
                   +-------------------------------+
                   | Exp5069 archive/activate truth |
                   +-------------------------------+
                                  |
                                  v
      +-------------------+   +-----------------------------+
      | Exp5070 SOTA refs |-->| Exp5071 GGUF/logprob preflight |
      +-------------------+   +-----------------------------+
                                           |
                                           v
                         +-------------------------------+
                         | Exp5072 token logprob cache   |
                         +-------------------------------+
                                  |                 |
                                  v                 v
                  +------------------------+   +--------------------------+
                  | Exp5073 uPRM selector |   | Exp5074 VPR diagnostic  |
                  +------------------------+   +--------------------------+
                                  |                 |
                                  +--------+--------+
                                           |
                                           v
        +---------------------------+  +------------------------------+
        | Exp5075 DCCD/guided scale |  | Exp5076 D6 efficiency replay |
        +---------------------------+  +------------------------------+
                         |                         |
                         +-----------+-------------+
                                     |
                                     v
    +-------------------------------+     +------------------------------+
    | Exp5077 guarded FR-11 memory  |     | Exp5078 memory-gap ledger    |
    +-------------------------------+     +------------------------------+
                         |                         |
                         +-----------+-------------+
                                     |
                                     v
    +-------------------------------+     +------------------------------+
    | Exp5079 board continuity      |     | Exp5080 KAN PWA/MILP bridge |
    +-------------------------------+     +------------------------------+
                         |                         |
                         +-----------+-------------+
                                     |
                                     v
                     +--------------------------------+
                     | Exp5081 moat/FR-11 decision   |
                     +--------------------------------+
                                     |
                                     v
                     +--------------------------------+
                     | Exp5082 capstone              |
                     +--------------------------------+
```

## Phases

### Phase 0: Transition And Runtime Hygiene

Experiments: `exp5069`, `exp5070`, `exp5071`

Archive `.465` without inventing missing success, backfill the failed SOTA ingestion slot using Codex
routing, and produce a concrete SOTA GGUF/top-logprob readiness artifact. This phase must explicitly
distinguish cached-model readiness from endpoint/logprob readiness.

### Phase 1: Final Process-Verifier And Decoding Checks

Experiments: `exp5072`, `exp5073`, `exp5074`, `exp5075`, `exp5076`

Build the real token/step logprob substrate, test uPRM as the final D2 lever, test VPR only with
objective intermediate checks, scale DCCD/guided decoding, and replicate D6 as a cost/Pareto result.
The phase win condition is not "any positive point estimate." It is a proper paired win over tuned
self-consistency or a bounded-retirement result that says the moat is not currently there.

### Phase 2: FR-11, Formal KAN Bridge, And Hardware Continuity

Experiments: `exp5077`, `exp5078`, `exp5079`, `exp5080`

Run one guarded continuous self-learning attempt using group self-consistency/memory rollback; then write
a memory-gap ledger if promotion remains unsafe. Maintain hardware visibility across KV260, GateMate,
and PolarFire without claiming speedup. Add a small KAN/PWA/MILP verifier bridge so the KAN fast path has
a formal verification direction rather than only empirical calibration.

### Phase 3: Decision And Capstone

Experiments: `exp5081`, `exp5082`

Aggregate the verifier, decoding, self-learning, hardware, and KAN results. The milestone should end in
one of four states:

- `realized_process_verifier_moat`
- `efficiency_only_no_accuracy_moat`
- `bounded_retired_verifier_moat`
- `execution_incomplete`

No capstone may headline D1 or D4 as positive unless the artifact explicitly explains why the known
bounded/null and duplicate-failed evidence no longer applies.

## Dependency Graph

```text
exp5069
  -> exp5070
  -> exp5071
       -> exp5072
            -> exp5073
            -> exp5074
       -> exp5075
       -> exp5076
  -> exp5077
       -> exp5078
  -> exp5079
  -> exp5080

exp5073, exp5074, exp5075, exp5076, exp5077, exp5078, exp5079, exp5080
  -> exp5081
       -> exp5082
```

Structured `gated_on` entries are used for runtime skips where an upstream field is required:

- `exp5072` waits for `exp5071.logprob_endpoint_ready == true`.
- `exp5073` waits for `exp5072.logprob_cache_ready == true`.
- `exp5074` waits for `exp5072.step_cache_ready == true`.
- `exp5081` waits for the core phase artifacts to exist.
- `exp5082` waits for `exp5081.decision_ready == true`.

## Hardware Requirements

- **Dual RTX 3090:** required for GGUF SOTA inference and logprob cache construction. Use
  `cached_sota_pair()` / local `.gguf` paths; do not use `AutoTokenizer` on GGUF repo IDs.
- **Mandated local GGUFs for any LLM task:**
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **KV260:** use `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` and the existing overlay path.
  Do not use retired host block-device preconditions.
- **GateMate A1 / PolarFire:** run detect/precheck/terminal-state tasks only unless a known-good bitstream
  dispatch path exists. Do not flash or claim timing without transcript-backed evidence.
- **Extropic/TSU:** simulation and architecture only; no local TSU hardware target exists.

## Falsifiable Gates

1. uPRM is positive only if it beats genuine tuned self-consistency with paired statistics and clean
   token-logprob provenance.
2. VPR is positive only if intermediate rewards are objective and `verifier_is_oracle` is declared.
3. Guided/DCCD decoding is positive only if it beats both unguided and rerank-only at comparable or
   charged token/NFE budget.
4. D6 is an efficiency win only if accuracy parity holds within CI and judge/tool costs are charged.
5. FR-11 is positive only if held-out delta is non-negative, non-forgetting passes, contamination guard
   passes, and rollback prevents harmful promotion.
6. Hardware results may claim only the board/transport/timing evidence actually recorded in transcripts.
7. The capstone must prefer bounded retirement over a weak positive if every clean verifier arm ties or
   loses to tuned self-consistency.

## Expected Deliverables

- `research-roadmap-next.yaml` with 14 conductor tasks in execution order.
- `research-references.md` updated with `.466` source set.
- `results/experiment_5069_archive_465_activate_466.json` through
  `results/experiment_5082_capstone_v466.json` when the conductor runs.
- Updated ops docs after execution, including architecture freshness if capability work changes the
  architecture. `_bmad/architecture.md` is older than 30 days as of 2026-07-01 and should be reconciled
  by the next implementation-heavy milestone.
