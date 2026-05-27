# Research Roadmap vNEXT - Milestone 2026.05.295

**Title:** Receipt-Backed Live SOTA Clearance + Certificate Repair Gate + FR-11 Promotion Pack
**Created:** 2026-05-27
**Status:** Planned
**Supersedes:** 2026.05.294 "Duration-Corrected Live Verifier Recovery + Repair Materialization + FR-11 Counterexample Closure"
**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.294 Proved

Milestone `.294` completed all planned tasks and fixed the missing-artifact
behavior that made prior repair roadmaps hard to audit. The authoritative
closeout is `results/experiment_3176_capstone_v294.json`:

- `capstone_v294_ready=true`
- `paper_ready=false`
- `publication_blocker_count=73`
- `blocker_delta_from_v27=8`
- `missing_artifact_count=1`
- `verifier_status=gated_skip_preflight_failed_flagged_adversarial_exact_authority_only`
- `repair_gate_status=blocked_flagged_verifier`
- `repair_ladder_status=materialized_gated_skip_repair_gate_blocked_no_live_repair_attempts`
- `fr11_self_learning_status=controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update`
- `ebcn_kan_status=projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier`
- `hardware_sampler_status=blocked_no_authenticated_speedup_no_hardware_commands_no_speedup_claim_made`
- `next_top_gap=clean_live_verifier_adversarial_flag_clearance_repair_gate_unblock`

The most useful `.294` result is negative and operationally clean: every
downstream task wrote an honest artifact. `exp3164` retired the arbitrary fixed
60 second duration gate as a hard requirement and replaced it with a
measured-work contract, but `exp3165` still blocked because the mandated GGUF
replay substrate was unavailable. `exp3167` therefore wrote a full gated-skip
clean verifier artifact instead of pretending a live rerun occurred. `exp3168`
and `exp3169` correctly blocked repair until the verifier is clean.

`.294` also created two forward edges:

- `exp3172` produced a promotable controller-memory update for FR-11:
  before consistency `0.857143`, after consistency `1.0`, heldout consistency
  `1.0`, no negative-control regressions, and no model-weight update claim.
- `exp3173` found a promising but tiny EBCN/KAN diagnostic separation over
  exact rows and known false accepts. It remained projection-only and denied a
  deployed verifier claim.

| Area | `.294` result | `.295` consequence |
| --- | --- | --- |
| Live SOTA preflight | Contract fixed, but GPU/CUDA substrate unavailable | Classify model/cache/CUDA/CPU failure modes and produce proof-of-execution receipts |
| Clean verifier rerun | Full gated-skip artifact, still flagged adversarial | Re-run only after receipt-backed local SOTA smoke and controlled-invariance checks |
| Repair gate | Correctly blocked on flagged verifier | Expand certificate frontier first, then decide whether repair calls are allowed |
| Repair ladder | Materialized gated skip, no missing-row cascade | Run only if gate is unblocked; otherwise keep full blocked metrics |
| FR-11 | Controller-memory update promotable, no weights changed | Package promotion, rollback, and cross-environment drift replay |
| EBCN/KAN | Tiny diagnostic signal, no deployment | Test distributional/transport-aligned sidecars over exact rows |
| Hardware | THRML import available, no speedup claim | Add API/factor-graph boundary only, no performance claim |

## Three Biggest Gaps To PRD Vision

1. **FR-12 verifier trust is still blocked by executable evidence.** Carnot has
   exact authority rows and a corrected authenticity contract, but no current
   receipt-backed live SOTA run under the mandated GGUF policy. The next
   milestone must separate cache availability, CUDA availability, CPU fallback,
   transcript receipts, and headline eligibility.

2. **The repair loop is waiting on verifier clearance and certificate breadth.**
   The PRD requires verifiable repair, not repair attempted under a flagged
   verifier. `.295` expands counterexample certificates and bounded frontier
   records before any generation step, then lets an explicit repair gate decide.

3. **FR-11 self-learning is promotable but not operationally packaged.** `.294`
   proved controller-memory nonforgetting on the known ledger, but the PRD's
   autonomous self-learning vision needs promotion records, rollback policy,
   cross-environment drift replay, and clear language that no local model
   weights were updated.

## New Research Integrated

The post-`.294` sweep was appended to `research-references.md` before this
roadmap was designed. It adds these experimental inputs:

- **Distributional Energy-Based Models** (arXiv:2605.18871) for uncertainty
  sidecars that combine deterministic constraint penalties with learned quality
  scores.
- **Graph Energy Matching** (arXiv:2603.23398) for transport-aligned discrete
  energy sidecars over constraint graphs.
- **LoopUS** (arXiv:2605.11011) as a design hint for iterative refinement and
  early-exit controller memory, without claiming a model-weight update.
- **BEAVER** (OpenReview ICLR 2026 VerifAI-2) for deterministic bounded
  frontier records before repair generation.
- **Extropic THRML** and **Logical Intelligence Kona** as hardware/API boundary
  signals only; no local TSU/Kona execution or speedup is claimed.

## Architecture Direction

`.295` keeps exact solvers and canonical answers as authority. Local SOTA
models may propose or repair only after a receipt-backed preflight passes.
Diagnostic EBMs/KANs can score uncertainty and false-accept separation but
remain sidecars until coverage and live integration are proven.

```text
          Exact corpora + regression rows + counterexamples
                            |
                            v
                 Canonical exact authority layer
                            |
              +-------------+--------------+
              |                            |
              v                            v
  Receipt-backed local SOTA smoke   Distributional/GEM EBM sidecars
  - mandated GGUF cache check       - exact-row uncertainty
  - CUDA/CPU substrate class        - false-accept separation
  - transcript/proof receipt        - no deployed verifier claim
              |                            |
              v                            v
     Controlled-invariance executor + certificate frontier
              |                            |
              +-------------+--------------+
                            |
                            v
                   Clean verifier decision
                            |
                            v
                    Repair gate v4
                            |
             +--------------+---------------+
             |                              |
             v                              v
  Gated-skip repair artifact       Live repair ladder if clean
  with explicit blockers           with exact semantic scoring

FR-11 side lane:
  ledger counterexamples -> controller-memory promotion pack
  -> cross-environment drift replay -> rollback/nonforgetting policy

Hardware side lane:
  exact-row factor graph -> THRML API boundary smoke
  -> no speedup/performance claim without authenticated board transcript
```

## Required SOTA Model Policy

Every experiment that performs or may perform an LLM call must include at least
one mandated local SOTA GGUF in its `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as `Qwen3.5-0.8B` or `gemma-4-E4B-it` may appear only
as loud CPU smoke fallbacks and must not become headline-result models. The
planned LLM-touching tasks are `exp3179`, `exp3181`, and `exp3185`.

## Milestone Phases

### Phase 1 - Archive and Receipt Contract

- `exp3177` archives the `.294` closeout and activates `.295` planning without
  editing `research-roadmap.yaml`.
- `exp3178` defines receipt-contract v3: cache, loader, CUDA, CPU fallback,
  transcript hashes, token counts, wall-clock proof, throughput plausibility,
  and headline policy.
- `exp3179` performs the local SOTA receipt smoke if preconditions permit and
  otherwise writes a classified blocked artifact.

### Phase 2 - Verifier Clearance and Sidecar Expansion

- `exp3180` executes controlled-invariance checks over exact authority rows and
  any available receipt-backed transcripts.
- `exp3181` reruns the clean local SOTA verifier only if the receipt and control
  gates are satisfied; otherwise it writes a complete gated-skip artifact.
- `exp3182` tests a distributional/transport-aligned EBM sidecar over exact rows
  and known false accepts with no deployed verifier claim.

### Phase 3 - Certificate Repair Gate

- `exp3183` expands counterexample certificates and bounded frontier records.
- `exp3184` makes the repair-gate v4 decision from verifier, invariance, and
  certificate artifacts.
- `exp3185` materializes the repair ladder. If the gate is blocked it writes
  explicit blockers; if unblocked it runs a small local SOTA repair panel and
  scores semantic exactness.

### Phase 4 - FR-11 Promotion, Hardware Boundary, and Capstone

- `exp3186` packages the `.294` controller-memory update for promotion with
  rollback and no-weight-update language.
- `exp3187` replays the promoted controller update across heldout and
  cross-environment drift checks.
- `exp3188` creates a THRML factor-graph API boundary artifact without speedup
  or hardware-performance claims.
- `exp3189` updates cross-corpus matrix v29.
- `exp3190` writes the `.295` capstone and next-gap recommendation.

## Dependency Graph

```text
exp3177
  -> exp3178
       -> exp3179
            -> exp3180
                 -> exp3181
                      -> exp3184
                           -> exp3185

exp3180 -> exp3182
exp3183 -> exp3184

exp3172 (.294) -> exp3186 -> exp3187

exp3188 is independent hardware/API boundary work.

exp3181, exp3182, exp3183, exp3185, exp3187, exp3188
  -> exp3189
       -> exp3190
```

## Hardware Requirements

- **Dual RTX 3090 / CUDA:** Preferred for `exp3179`, `exp3181`, and `exp3185`,
  but not assumed. These tasks must first classify HF cache, loader, CUDA, and
  CPU fallback status. A CPU smoke can prove receipt wiring but cannot become a
  headline benchmark.
- **Mandated GGUF cache:** Required for headline local SOTA claims. If only
  `unsloth/gemma-4-26B-A4B-it-GGUF` is present, report that exactly.
- **KV260 / GateMate / PolarFire:** No board commands are required in `.295`.
  Do not revive retired host-SD-card checks or speedup claims.
- **THRML:** `exp3188` may import local `thrml` and translate tiny factor-graph
  structures. It must not report sampler speedup, TSU execution, or hardware
  acceleration without authenticated hardware evidence.

## Success Criteria

Minimum success for `.295`:

1. All 14 tasks write deliverable artifacts.
2. The live SOTA path ends in one of two honest states:
   - receipt-backed local SOTA smoke passes, controlled invariance is executed,
     and the clean verifier rerun reports exact-authority false-accept status; or
   - the artifact classifies cache/CUDA/CPU blockers without unblocking repair.
3. Repair gate v4 is explicit and repair ladder v5 materializes either semantic
   repair metrics or a complete gated-skip artifact.
4. FR-11 controller-memory promotion pack and cross-environment drift replay
   exist, with no model-weight update claim.
5. Distributional/GEM EBM sidecar and THRML boundary artifacts are diagnostic
   only unless they meet their stated coverage gates.
6. Matrix v29 and the capstone reconcile `ops/status.md`, `ops/changelog.md`,
   and the publication blocker count without reintroducing retracted claims.
