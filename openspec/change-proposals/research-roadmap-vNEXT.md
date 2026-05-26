# Research Roadmap vNEXT - Milestone 2026.05.294

**Title:** Duration-Corrected Live Verifier Recovery + Repair Materialization + FR-11 Counterexample Closure
**Created:** 2026-05-26
**Status:** Planned
**Supersedes:** 2026.05.293 "Verifier Evidence Corrigendum + Repair Ladder Execution + FR-11 Ledger Closure"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.293 Proved

Milestone `.293` completed its scheduled non-gated work, but the headline path
regressed from 55 to 65 publication blockers. The authoritative closeout is
`results/experiment_3162_capstone_v293.json`:

- `capstone_ready=true`
- `paper_ready=false`
- `publication_blocker_count=65`
- `blocker_delta_from_v26=10`
- `next_top_gap=clean_live_verifier_corrigendum_repair_gate`
- `repair_gate_status=blocked_pending_clean_rerun_gate_failed`
- `repair_ladder_status=correctly_skipped_gate_blocked_no_live_repair_attempts`
- `kan_status=bounded_monitor_records_4_new_2_no_deployed_verifier_blockers_3`

The most important result was the evidence-quality correction, not a verifier
headline:

- `exp3150` produced `adversarial_corrigendum_v1_ready=true` and correctly set
  `live_verifier_evidence_trusted=false`.
- `exp3151` proved at least one mandated SOTA GGUF is locally callable:
  `unsloth/gemma-4-26B-A4B-it-GGUF` loaded through `llama_cpp` and returned a
  transcript hash, but the artifact honestly blocked because `duration_s` was
  about 10.6 s against a fixed 60 s plausibility floor.
- `exp3152`, `exp3154`, and `exp3155` were structurally gate-skipped or
  produced thin `blocked_gate_check_failed` artifacts, leaving matrix v27 with
  missing deliverable rows.
- `exp3156` and `exp3157` kept FR-11 honest: controller/residual memory exists,
  but `ledger_consistency_rate=0.857143`, so promotion remains blocked.
- `exp3158` showed a promising tiny EBCN diagnostic row
  (`scalar_energy_auc=1.0`, `exact_labeled_row_count=6`) but correctly denied a
  live integration claim.
- `exp3160` kept hardware/sampler claims bounded: no authenticated speedup, no
  board command transcript, and no local TSU/Kona execution.

The main lesson is that `.294` should not repeat `.293` with a larger duration
constant. It should repair the authenticity contract itself: measured work,
token counts, model-load evidence, transcript hashes, and controlled
invariance checks must replace a single arbitrary wall-clock floor.

| Area | `.293` result | `.294` consequence |
| --- | --- | --- |
| Live preflight | One real mandated GGUF call, blocked by fixed duration floor | Define token-scaled plausibility and replay the preflight |
| Clean verifier rerun | Gated skipped, no full deliverable | Always write a full artifact, even when internally gated |
| Repair gate | Thin gate-failed artifact | Always write an explicit decision with blockers |
| Repair ladder | Missing/gated artifacts persist | Always materialize gated-skip or repair metrics |
| FR-11 | Ledger consistency 0.857143 | Isolate failing counterexample family, then test nonforgetting |
| EBCN/KAN | Bounded diagnostics, no deployed verifier | Expand exact rows without promotion claim |
| Hardware | No authenticated speedup | Keep evidence boundaries and tooling probes only |

## Three Biggest Gaps To PRD Vision

1. **FR-12 verifier trust is still blocked by methodology, not model access.**
   `.293` showed local SOTA inference can run, but the contract rejected the run
   for a coarse duration rule. The next milestone must distinguish fake
   evidence from fast legitimate evidence without lowering the authenticity bar.

2. **The repair loop still lacks a complete negative artifact path.** The PRD
   needs verifiable repair, but the matrix is accumulating missing rows when
   structured gates skip whole tasks. `.294` makes repair tasks write complete
   blocked or gated-skip artifacts before any model call, so the publication
   ledger stops growing from absence.

3. **FR-11 continuous self-learning remains controller-only and inconsistent.**
   The self-learning loop has memory and replay, but promotion requires ledger
   consistency of 1.0 plus negative controls. `.294` targets the remaining
   counterexample family and tests nonforgetting under environment/variant
   separation.

## New Research Integrated

The post-`.293` sweep was appended to `research-references.md` before this
roadmap was designed. Findings shaping `.294`:

| Finding | Source | Milestone use |
| --- | --- | --- |
| Controlled-invariance hallucination sanity checks | arXiv:2605.08346 | `exp3166` adds Force/Remove/shuffled-trace style checks to verifier evidence |
| CoT trace performativity | arXiv:2605.11746 | `exp3166` treats visible reasoning traces as suspect unless exact evidence supports them |
| Token-level hallucination detection | arXiv:2605.12384 and HF 2605.05166 | `exp3166` records token/first-token suspicion as triage only |
| Constrained decoding alignment tax | arXiv:2604.06066 | `exp3169` and `exp3170` score semantic repair, not only schema validity |
| MCMC/AprAD constrained sampling | OpenReview NeurIPS 2025 | `exp3169` can propose distribution-aware repair candidates after gates pass |
| BEAVER and TraceFix | OpenReview + arXiv:2605.07935 | `exp3170` builds certificate-first counterexample repair evidence |
| Self-play/self-correction and KAN forgetting | arXiv:2510.27072, 2506.06923, 2511.12828 | `exp3171` and `exp3172` enforce nonforgetting before FR-11 promotion |
| THRML/XGrammar/Kona ecosystem | Extropic, GitHub, Logical Intelligence | `exp3174` records tooling boundaries without speedup claims |

## Architecture Direction

`.294` keeps exact authority at the center and adds two guardrails:

1. Live authenticity becomes a measured-work contract, not a fixed-duration
   heuristic.
2. Repair tasks always materialize a full artifact, even when they internally
   refuse to run a model.

```text
                 +---------------------------------------------+
                 | .293 capstone                               |
                 | paper_ready=false, blockers=65              |
                 | top gap: clean live verifier + repair gate  |
                 +----------------------+----------------------+
                                        |
                                        v
              +-------------------------+-------------------------+
              | exp3163 archive + exp3164 preflight contract v2  |
              +-------------------------+-------------------------+
                                        |
                                        v
              +-------------------------+-------------------------+
              | exp3165 authenticity replay v2                  |
              | measured work + transcript hashes + token budget |
              +-------------------------+-------------------------+
                                        |
                                        v
              +-------------------------+-------------------------+
              | exp3166 invariance + token-suspicion audit       |
              +-------------------------+-------------------------+
                                        |
                                        v
              +-------------------------+-------------------------+
              | exp3167 clean verifier rerun v9                 |
              | full clean artifact OR full gated-skip artifact  |
              +-------------------------+-------------------------+
                                        |
                                        v
              +-------------------------+-------------------------+
              | exp3168 repair gate decision v3                 |
              | always writes repair_gate_state                  |
              +-------------------------+-------------------------+
                                        |
                         unblocked only inside artifact logic
                                        |
                                        v
        +-------------------------------+-------------------------------+
        | exp3169 repair materializer + exp3170 counterexample certs   |
        +-------------------------------+-------------------------------+

        +--------------------------+   +-------------------------------+
        | exp3171 FR-11 ledger     |-->| exp3172 nonforgetting pilot   |
        | counterexample isolation |   | controller-only self-learning |
        +--------------------------+   +-------------------------------+

        +--------------------------+   +-------------------------------+
        | exp3173 EBCN/KAN bounded |   | exp3174 hardware/tooling      |
        | diagnostic expansion     |   | boundary and ecosystem probe  |
        +-------------+------------+   +---------------+---------------+
                      \                              /
                       v                            v
                  +----+----------------------------+----+
                  | exp3175 matrix v28 -> exp3176 capstone |
                  +----------------------------------------+
```

## Required SOTA Model Policy

Every `.294` experiment that invokes a local LLM must include `MODEL_SPECS`
and must attempt at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF` (flagship MoE)
- `unsloth/gemma-4-31B-it-GGUF` (flagship dense)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (middle MoE)

Legacy small models such as `Qwen3.5-0.8B` and `gemma-4-E4B-it` may appear
only as CPU smoke tests. They cannot headline verifier, repair, or
self-learning results. If a mandated model is not locally usable, the task must
write a complete blocked or diagnostic artifact with `live_call_count=0`,
`headline_claim_allowed=false`, and explicit precondition evidence.

## Milestone Phases

### Phase A - Evidence Contract Repair

**Goal:** fix the live-inference authenticity contract and make detector
evidence resistant to answer-artifact and trace-artifact shortcuts.

- `exp3163` archives `.293` exactly and stages `.294`.
- `exp3164` converts `.293`'s fixed 60 s duration rule into a measured-work
  contract: load evidence, prompt hashes, output token counts, repeated smoke
  calls, transcript hashes, GPU/CPU substrate, and reproducibility checksum.
- `exp3165` replays the local SOTA authenticity preflight under that contract.
- `exp3166` adds controlled-invariance and token-suspicion checks. Token-level
  signals may route exact checks but cannot authorize acceptance.

### Phase B - Verifier And Repair Materialization

**Goal:** produce clean verifier evidence if possible, and otherwise produce
full blocked artifacts that stop missing-row growth.

- `exp3167` runs the clean local SOTA verifier rerun v9 if preconditions pass;
  otherwise it writes a full gated-skip artifact.
- `exp3168` writes the repair gate decision v3 regardless of verifier outcome.
- `exp3169` writes the repair ladder materializer v4. It may call a model only
  when `repair_gate_state=unblocked`; otherwise it writes a full gated-skip
  artifact with no model calls.
- `exp3170` builds certificate-first counterexample repair evidence using exact
  rows, TLA+/Z3-style counterexamples, and BEAVER-style frontier bounds where
  available.

### Phase C - FR-11 Counterexample Closure

**Goal:** advance continuous self-learning only within controller/environment
memory boundaries.

- `exp3171` isolates the remaining FR-11 ledger counterexample family from the
  `.293` 0.857143 consistency result and defines an environment/variant split.
- `exp3172` runs a controller-only self-learning and nonforgetting pilot. It may
  recommend promotion only if ledger consistency reaches 1.0 and negative
  controls pass.
- `exp3173` expands EBCN/KAN bounded diagnostic rows against exact labels and
  known false-accept rows, without deploying a verifier.

### Phase D - Evidence Boundary And Closeout

**Goal:** keep hardware/tooling context current, then close the matrix.

- `exp3174` records THRML, XGrammar, llguidance, Extropic, Kona, CUDA, KV260,
  GateMate, and PolarFire boundaries without speedup or local hardware claims.
- `exp3175` writes cross-corpus matrix v28 and reconciles missing/gated rows.
- `exp3176` writes the `.294` capstone and next-gap recommendation.

## Dependency Graph

```text
exp3163
  -> exp3164
  -> exp3165
  -> exp3166
  -> exp3167
  -> exp3168
  -> exp3169
       -> exp3170

exp3171
  -> exp3172

exp3166
  -> exp3173

exp3174

exp3167, exp3168, exp3169, exp3170, exp3172, exp3173, exp3174
  -> exp3175
  -> exp3176
```

Structured conductor gates are intentionally sparse in `.294`. The previous
milestone showed that structural gates can save model calls but leave thin or
missing artifacts that worsen the publication ledger. The live verifier and
repair tasks therefore use internal preconditions and write full blocked or
gated-skip artifacts. Only the capstone structurally gates on matrix v28:

- `exp3176` gates on `exp3175.matrix_v28_ready == true`.

## Hardware Requirements

No new hardware claims are required for `.294`.

- **GPU/local GGUF:** `exp3165`, `exp3167`, and `exp3169` may use local SOTA
  inference. They must record model path, load evidence, transcript hashes,
  token counts, seed, checksum, and measured-work plausibility. Legacy small
  models are smoke tests only.
- **Dual RTX 3090 / CUDA:** usable only when detected and recorded. No speedup
  claim is allowed without matched baseline, command transcript, and artifact
  checksum.
- **KV260, GateMate, PolarFire:** evidence ingestion only unless the operator
  has already supplied authenticated logs. Do not convert wish-list status into
  measured sampler speedup.
- **Extropic THRML/TSU and Kona/Aleph:** public pages and local import probes
  are architecture references. They do not support local hardware acceleration
  claims.

## Success Criteria

`.294` succeeds if it produces complete evidence, not necessarily a positive
headline:

1. The authenticity preflight contract no longer depends on one arbitrary
   duration floor and can distinguish fast legitimate local calls from fake
   evidence.
2. The clean verifier rerun produces either clean live evidence or a complete
   gated-skip artifact with no missing row.
3. The repair gate and repair ladder materializer always write full artifacts,
   and no model repair call runs while the verifier gate is blocked.
4. FR-11 either reaches ledger consistency 1.0 with nonforgetting controls or
   remains explicitly controller-memory only with replayable counterexamples.
5. Matrix v28 and the capstone reduce missing artifacts and state whether the
   publication blocker count improved, regressed, or stayed blocked.
