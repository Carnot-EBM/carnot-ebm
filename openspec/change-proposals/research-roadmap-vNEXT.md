# Research Roadmap vNEXT: Milestone 2026.05.308

**Title:** Phase-3 Path Recovery, Verifier Grounding, and FR-11 Nonforgetting

**Created:** 2026-05-29
**Status:** Proposed next milestone
**Supersedes:** 2026.05.307
**Milestone YAML:** `research-roadmap-next.yaml`
**Primary references:** `research-program.md`, `_bmad/prd.md`,
`_bmad/architecture.md`, `ops/status.md`, `ops/changelog.md`,
`ops/conductor-log.md`, `research-complete.yaml`, `research-roadmap.yaml`,
`research-references.md`, `research-hardware-wishlist.md`

## What Previous Milestone Proved

Milestone `2026.05.307` completed operationally, but it did not produce the two
scientific upstream artifacts it was designed to test:

- `exp3322-energy-descent-vs-autoregressive-premise-v1` failed repeatedly in the
  conductor with a Codex CLI error before writing an artifact.
- `exp3323-verifier-ensemble-diversity-audit-v1` failed the same way.
- `exp3324-capstone-v307` therefore reported a blocked Phase-3 path rather than
  a scientific verdict.

The key result is procedural but important: the next top gap is not a new
theory. It is a conductor-runnable recovery of the two Phase-3 link tests with
smaller bootstrap tasks, explicit preconditions, and structured artifacts.

Milestone `.306` remains the most recent source of clean quality-gate progress:
Garak is unblocked, the repair headline remains blocked by provenance/runtime
rules, and the prompt-injection KAN replacement path is retired as a headline
claim. That forces `.308` to keep claims narrow and exact-verifier anchored.

## Three Biggest Gaps To PRD Vision

1. **Phase-3/Kona premise is still unmeasured on real SOTA local inference.**
   The PRD wants verifiable energy-based reasoning and a path toward
   continuous-latent self-correction. The missing `.307` energy-descent versus
   autoregressive panel means Carnot still lacks evidence that an energy
   refinement loop improves verified outcomes over normal autoregressive
   generation.

2. **Verifier grounding and diversity are not quantified at the current stack.**
   The Phase-3 and publication arguments depend on independent-ish verifier
   signals. The missing `.307` lambda_min/effective-k audit means Carnot does not
   yet know whether the ensemble is a real committee or a correlated wrapper
   around one failure mode.

3. **FR-11 continuous self-learning is not yet nonforgetting-validated.**
   The PRD's autonomous self-learning loop requires updates that improve future
   behavior without human curation. Recent milestones have controller-memory and
   repair-audit work, but the next self-learning experiment must bind updates to
   exact verifier outcomes, measure negative transfer, and emit rollback logic.

Publication G2 remains a downstream readiness gate, not the core research gap.
However, `.308` includes a reproducer/evidence-matrix task so clean upstreams can
be handed to an independent runner without another planning cycle.

## External Research Integrated

The post-`.307` sweep added a new section to `research-references.md` before this
roadmap was designed. The actionable findings are:

- **Energy-Based Transformers (EBT, arXiv:2507.02092; `alexiglad/EBT`):**
  implementation-relevant sidecar for iterative energy minimization. `.308`
  tests a small adapter smoke only; it does not claim foundation-model parity.
- **Interwhen verifiable-reasoning framework:** concrete monitor pattern for
  scoring intermediate candidates and measuring monitor disagreement.
- **Hallucination as Commitment Failure and HalluScan:** live panels should log
  commitment/abstention/unsupported-claim telemetry, not only final accuracy.
- **Online learnability of CoT verifiers, KAN-CL, and KAN abstraction
  verification:** FR-11 should be framed as nonforgetting online verifier memory.
- **Energy-guided test-time scaling:** use energy for candidate proposal and
  ranking, while exact verifiers remain the authority.
- **Extropic TSU and Kona public updates:** useful long-term architecture
  targets, but no local speedup or parity claim is permitted without measured
  transcripts.

## SOTA Local GGUF Policy

Every `.308` task that performs live LLM inference must declare `MODEL_SPECS`
using at least one mandated local SOTA GGUF:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The expected implementation pattern is `cached_sota_pair(gpu_indices=(0, 1))`
from `scripts/experiment_template.py`. Legacy tiny models are allowed only as
explicit CPU smoke-test fallbacks and must not be used for headline metrics.
Live-inference artifacts must use `inference_substrate=live_llm_inference`,
record the exact `model_specs`, and satisfy the duration/provenance rules in
`CLAUDE.md`.

## Architecture

```text
                         Milestone 2026.05.308

   ops/conductor-log.md + .307 artifacts
                  |
                  v
   Phase 0: preflight and failure manifest
                  |
                  v
   +---------------------------+       +----------------------------+
   | SOTA GGUF candidate panel |       | Cached verifier corpus     |
   | Qwen3.6 / Gemma4 local    |       | exact labels + candidates  |
   +-------------+-------------+       +--------------+-------------+
                 |                                    |
                 v                                    v
   +---------------------------+       +----------------------------+
   | energy descent vs AR      |       | lambda_min / effective-k   |
   | commitment telemetry      |       | verifier diversity audit   |
   +-------------+-------------+       +--------------+-------------+
                 |                                    |
                 +----------------+-------------------+
                                  |
                                  v
       +--------------------------+--------------------------+
       | monitor-anchored proposal layer                    |
       | EBT sidecar smoke + Interwhen-style monitors +     |
       | energy-guided candidate ranking                    |
       +--------------------------+--------------------------+
                                  |
                                  v
       +--------------------------+--------------------------+
       | FR-11 online verifier memory                       |
       | nonforgetting gate + rollback + reproducibility     |
       +--------------------------+--------------------------+
                                  |
                                  v
                 evidence matrix v39 + capstone v308
```

## Phase Plan

### Phase 0 - Evidence Sanitation And Activation

Goal: convert `.307` from a missing-upstream state into a clean `.308` queue with
explicit failure context.

- `exp3325` archives `.307` honestly and activates `.308`.
- `exp3326` writes the preflight manifest that explains why `exp3322` and
  `exp3323` failed, checks required specs/resources, and defines the run-ready
  preconditions for the recovered tests.

### Phase 1 - Recover The Two Existential Link Tests

Goal: produce the missing Phase-3 evidence in smaller, conductor-runnable pieces.

- `exp3327` performs a live SOTA bootstrap smoke for the energy-descent substrate.
- `exp3328` runs the full SOTA GGUF energy-descent versus autoregressive panel,
  gated on the bootstrap.
- `exp3329` runs the cached verifier-ensemble diversity audit at sufficient
  sample size for covariance claims.
- `exp3330` writes a diversity-remediation plan only if the audit confirms
  low lambda_min.

### Phase 2 - Literature-Propelled Prototypes

Goal: integrate new external ideas without inflating claims.

- `exp3331` adds an EBT sidecar adapter smoke against exact verifier scores.
- `exp3332` pilots Interwhen-style monitors for intermediate candidate scoring.
- `exp3333` tests energy-guided test-time scaling as candidate proposal/ranking
  with exact verifier acceptance.

### Phase 3 - Continuous Self-Learning And Reproducibility

Goal: move FR-11 beyond static memory while packaging evidence for G2.

- `exp3334` runs online verifier-memory updates with nonforgetting and rollback
  gates.
- `exp3335` creates the independent reproducer pack and evidence matrix v39.
- `exp3336` writes the capstone, publication-gate status, and next-top-gap
  decision.

## Task Summary

| Exp | Title | Substrate | Main deliverable |
|---|---|---|---|
| 3325 | Archive `.307`, activate `.308` | aggregation | `results/experiment_3325_archive_v307_activate_v308.json` |
| 3326 | Phase-3 path preflight manifest | aggregation | `results/experiment_3326_phase3_path_preflight_manifest_v1.json` |
| 3327 | Energy-descent substrate bootstrap | live LLM | `results/experiment_3327_energy_descent_substrate_bootstrap_v1.json` |
| 3328 | Energy-descent vs AR SOTA panel | live LLM | `results/experiment_3328_energy_descent_vs_ar_sota_panel_v2.json` |
| 3329 | Verifier diversity audit v2 | verifier ensemble | `results/experiment_3329_verifier_ensemble_diversity_audit_v2.json` |
| 3330 | Diversity remediation plan | aggregation | `results/experiment_3330_verifier_diversity_remediation_plan_v1.json` |
| 3331 | EBT sidecar adapter smoke | cached verifier | `results/experiment_3331_ebt_sidecar_adapter_smoke_v2.json` |
| 3332 | Interwhen monitor pilot | cached verifier | `results/experiment_3332_interwhen_monitor_pilot_v1.json` |
| 3333 | Energy-guided TT scaling ablation | live LLM | `results/experiment_3333_energy_guided_ttscaling_sota_ablation_v1.json` |
| 3334 | FR-11 online memory nonforgetting | cached verifier | `results/experiment_3334_fr11_online_verifier_memory_nonforgetting_v4.json` |
| 3335 | Reproducer pack and matrix v39 | aggregation | `results/experiment_3335_reproducer_pack_and_evidence_matrix_v39.json` |
| 3336 | Capstone v308 | aggregation | `results/experiment_3336_capstone_v308.json` |

## Dependency Graph

```text
exp3325
  -> exp3326
      -> exp3327
          -> exp3328
      -> exp3329
          -> exp3330 (structured gate: lambda_min_sigma <= 0.1)

exp3331 reads exp3326/3329 when available but is not hard-gated.
exp3332 reads exp3329 when available but is not hard-gated.
exp3333 reads exp3328/3329 when available but is not hard-gated.
exp3334 reads any clean .308 upstreams plus stable .305-.306 cached evidence.
exp3335 aggregates all available .308 artifacts.
exp3336 aggregates all available .308 artifacts and records blocked states honestly.
```

Only `exp3328` and `exp3330` use conductor `gated_on` fields:

- `exp3328` runs only if `exp3327.energy_descent_bootstrap_ready == true`.
- `exp3330` runs only if `exp3329.lambda_min_sigma <= 0.1`.

The capstone is deliberately ungated so the milestone always emits a terminal
artifact even if an upstream task blocks.

## Hardware Requirements

| Requirement | Tasks | Status and rule |
|---|---|---|
| Dual RTX 3090 local GGUF runtime | 3327, 3328, 3333 | Required for live SOTA panels. Use `cached_sota_pair()` and write blocked artifacts if cache/GPU checks fail. |
| CPU/JAX verifier scoring | 3329, 3331, 3332, 3334 | Required. Claims must include sample size, random seed, checksum, and timing. |
| KV260 / GateMate / PolarFire | none as required | Mention only as future hardware path. No latency/speedup claim in `.308` without board transcripts. |
| Extropic TSU / Kona hardware | none | Literature/architecture reference only. No local hardware claim. |

## Evidence And Artifact Rules

Every task artifact must include:

- `honest_verdict` with a prefix such as `complete:`, `success:`,
  `blocked:`, or `failed:`.
- `inference_substrate`.
- `random_seed`.
- `reproducibility_checksum`.
- `duration_s`.
- Deliverable-specific readiness booleans, for example
  `energy_descent_bootstrap_ready`, `energy_descent_vs_ar_panel_v2_ready`,
  `verifier_diversity_audit_v2_ready`, and `fr11_nonforgetting_ready`.

Statistical claims must satisfy CLAUDE.md sample-size rigor. In particular,
covariance/eigenvalue claims should use at least 1000 scored cases or state that
they are diagnostic-only. Live accuracy deltas should avoid exact-1.0 language
unless the artifact records the exact methodology and confidence interval.

## Exit Criteria

Milestone `.308` is successful if it produces:

1. A conductor-runnable explanation and recovery of the missing `.307` upstreams.
2. Either a clean `exp3328` energy-descent versus AR verdict or a blocked artifact
   whose root cause is narrower than the `.307` CLI failure.
3. A clean `exp3329` verifier-diversity/lambda_min verdict or an actionable
   remediation plan.
4. At least one completed FR-11 continuous self-learning artifact with a
   nonforgetting/rollback gate.
5. A terminal capstone stating the publication-gate status and next top gap.

## Non-Goals

- No claim that Carnot has reached Kona parity.
- No claim that EBT sidecars are a production foundation-model replacement.
- No prompt-injection KAN replacement headline.
- No hardware latency or speedup claim without measured hardware transcripts.
- No modification to `research-roadmap.yaml` or `scripts/research_conductor.py`.
