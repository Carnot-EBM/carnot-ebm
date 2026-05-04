# Research Roadmap vNEXT: Milestone 2026.04.99

Planned: 2026-05-04
Status: Draft for conductor execution
Predecessor: 2026.04.98 Combined Retro + Orthogonality Audit v3 + arXiv Submit + GRPO v7 + Phase-5-D v3 + WOPR Completion
Roadmap YAML: `research-roadmap-next.yaml`

## What Milestone .98 Proved

Milestone 2026.04.98 completed the measurement-heavy parts that determine what can be claimed honestly:

- Verifier orthogonality is now measured from production-style data: max pairwise r = 0.462 for the k=5 ensemble, with k_eff = 1.76. This supports AND-composition, but also shows the ensemble is not close to five fully independent verifiers.
- Q11 Time-Series Skeptic instrumentation exists, but is only moderately useful: correlation 0.547 and vulnerability 0.453.
- DiffuTruth is a weak baseline on FoVer compared with Carnot semantic-energy scoring: DiffuTruth 0.082 vs Carnot 0.948.
- QuantKAN 3-bit LUT deployment preserved high AUROC: 0.9801 with 2.5x LUT speedup.
- The .98 retrospective found only 5 of 13 criteria met. The unfinished items are not ignorable: combined retro backfill, paper critical fixes, arXiv bundle, GRPO v7, Phase-5-D gates, WOPR Kakuro/Masyu, and gaming-defense measurement.

The next milestone treats those as carry-forward work, but does not simply rerun the failed prompts. Each carry-forward task has a narrowed deliverable and explicit prior-failure handling.

## Current Research Signals Added Before Planning

The 2026-05-04 literature scan added these candidate ideas to `research-references.md`:

- FSNet: feasibility-seeking neural updates for constrained optimization with guarantees.
- SnareNet: flexible repair layers for neural networks with hard constraints.
- Thinking Before Constraining: trigger structured decoding only when constraints are needed.
- PRIME: process-outcome alignment as a verifier-selection signal.
- CompassVerifier: unified lightweight verifier baseline for verifiable reasoning.
- Cactus: constrained acceptance speculative sampling.
- Extropic Z1/XTR-0 and Kona architecture status updates.

Milestone .99 turns FSNet, SnareNet, triggered structured certificates, PRIME-style verifier selection, and Cactus-style acceptance into local experiments. CompassVerifier remains a monitored baseline for a later slot unless triggered certificate extraction produces enough comparable outputs during .99.

## Three Biggest Gaps

1. **Publication credibility gap.** The strongest measurements now exist, but the paper/arXiv path remains blocked by unresolved critical fixes and stale release packaging. Carnot needs one clean publication bundle that cites measured artifacts and deletes unsupported claims.

2. **SOTA local extraction gap.** Many earlier extraction and verifier experiments used tiny smoke-test models or canned traces. The PRD requires local-first verification on current instruction-tuned models. Any new LLM experiment in .99 must use at least one mandated SOTA GGUF headline model:
   - `unsloth/Qwen3.6-35B-A3B-GGUF`
   - `unsloth/gemma-4-31B-it-GGUF`
   - `unsloth/gemma-4-26B-A4B-it-GGUF`

3. **Self-learning gap.** The architecture contains case memory, GRPO, VPRM, and continuous EBM components, but no recent milestone has closed the loop from verifier feedback to improved future decisions. .99 must produce at least one honest continuous self-learning artifact, even if the result is negative.

## Architecture Target

```text
Local SOTA GGUF models
  Qwen3.6-35B-A3B, Gemma-4-31B-it, Gemma-4-26B-A4B-it
          |
          v
Triggered certificate extraction
  constraint-aware tail only when uncertainty/energy trigger fires
          |
          v
Carnot verifier cascade
  Tier 0 probes -> formal claims -> k=5 AND ensemble -> Ising/energy terms
          |
          +----------------------------+
          |                            |
          v                            v
PRIME-style verifier selection      Cactus-style constrained acceptance
  process/outcome alignment          draft accepted only if energy/cert passes
          |
          v
Continuous self-learning
  verifier-weight memory -> GRPO/VPRM reward -> replayed case memory
          |
          v
Phase-3 continuous EBM repair
  FSNet feasibility step -> SnareNet adaptive repair layer
          |
          v
Publication + ops reconciliation
  arXiv bundle, WOPR hardening, gaming-defense measurement, milestone retro
```

## Phase 0: Publication and Stale-Retro Closeout

Goal: convert measured .98 results into a clean publication posture and close stale bookkeeping before new claims accumulate.

- `exp1268-retro-backfill-95-96-97-v2`: finalize the stale combined retro artifacts using pure JSON archaeology.
- `exp1269-paper-v6-critical-fixes-v2`: resolve the paper critical issues with measured citations to exp1256/1264/1265/1266.
- `exp1270-arxiv-bundle-v10-gated`: package the arXiv bundle only if exp1269 reports all critical issues fixed.

Success bar: paper bundle exists or is honestly blocked with a named missing field. No “in_progress” paper artifact is acceptable.

## Phase 1: SOTA Certificates and Self-Learning Reward Selection

Goal: produce verifiable SOTA local model traces and convert process/outcome signals into a learning signal.

- `exp1271-triggered-certificate-extraction-sota-gguf`: run triggered structured certificate extraction on SOTA GGUF outputs, using the `cached_sota_pair()` pattern and recording exact model IDs.
- `exp1272-prime-verifier-selection-audit`: rank verifiers by process/outcome alignment before attempting any GRPO update.
- `exp1273-grpo-v8-prime-vprm-smoke-gated`: run a bounded GRPO/VPRM smoke only if exp1272 writes a verifier-weight vector.
- `exp1274-online-self-learning-certificate-memory-v3`: update case memory from verified certificates and measure whether replay improves future decisions.

Success bar: at least one artifact reports a real self-learning delta (`self_learning_delta_overall` or `grpo_v8_delta_pp`) and records whether it is positive, zero, or negative.

## Phase 2: Continuous EBM Repair and Constrained Acceptance

Goal: test whether recent constrained-neural work maps to Carnot’s Phase-3/Kona substrate.

- `exp1275-fsnet-feasibility-step-continuous-ebm`: compare raw Langevin updates with an FSNet-style feasibility-seeking step on continuous EBM states.
- `exp1276-snarenet-repair-layer-gated`: add an adaptive repair-layer prototype only if the FSNet step improves feasibility.
- `exp1277-cactus-constrained-acceptance-sampling-gated`: try Cactus-style constrained acceptance only if triggered certificates are parseable enough to serve as a verifier.

Success bar: measure feasibility, energy, and violation deltas. Negative results are acceptable if they explain whether continuous repair should be retired, narrowed, or escalated to hardware.

## Phase 3: Hardening, WOPR Completion, and Retro

Goal: close the highest-value .98 carry-forwards without broadening the milestone.

- `exp1278-gaming-verifiers-defense-est-final`: finish the gaming-defense measurement using EST-style pure data analysis.
- `exp1279-wopr-kakuro-v4-minimal`: ship or honestly block the Kakuro cartridge with tests and spec alignment.
- `exp1280-wopr-masyu-v3-minimal`: ship or honestly block the Masyu cartridge with tests and spec alignment.
- `exp1281-milestone-retro-99`: evaluate .99 criteria and write carry-forwards for the next planner.

Success bar: no stale skeleton artifacts. A cartridge can be blocked, but the block must name the exact missing verifier, dataset, or API.

## Dependency Graph

```text
exp1268  exp1269
            |
            v
         exp1270

exp1271 ------------------------+
   |                            |
   v                            v
exp1277                      exp1274

exp1272
   |
   v
exp1273

exp1275
   |
   v
exp1276

exp1278   exp1279   exp1280
     \       |        /
      \      |       /
       v     v      v
          exp1281
```

Structured conductor gates:

- `exp1270` gates on `exp1269.critical_issues_fixed >= 5`.
- `exp1273` gates on `exp1272.verifier_weight_vector_written == true`.
- `exp1276` gates on `exp1275.feasibility_delta_overall > 0.0`.
- `exp1277` gates on `exp1271.certificate_parse_rate >= 0.8`.

## Hardware Requirements

Minimum:

- CPU-only path for publication closeout, retros, WOPR static tests, PRIME audit, FSNet/SnareNet smoke tests.
- Python environment with repo dependencies already used by prior Carnot experiments.

Preferred:

- Dual RTX 3090 or equivalent CUDA devices for SOTA GGUF LLM inference.
- Local Hugging Face cache containing at least one mandated SOTA GGUF, preferably the cached SOTA pair used by `scripts/experiment_template.py`.

Hardware deliberately not required in .99:

- KV260/Vivado FPGA bring-up. The current hardware wishlist still marks deeper FPGA work as blocked by tooling.
- AMD XDNA NPU unblocking. The missing dependency/wheel issue has already caused repeated blocked attempts.
- Extropic TSU access. .99 only updates the readiness map through references; no TSU experiment is planned until hardware/API availability changes.

## Milestone Success Criteria

1. `exp1268` closes the stale .95/.96/.97 retro gap or names the exact missing artifacts.
2. `exp1269` fixes all five critical paper issues or reports a non-ambiguous blocker.
3. `exp1270` writes a submission bundle only when gated prerequisites pass.
4. `exp1271` records exact SOTA GGUF model IDs and a certificate parse rate.
5. `exp1272` writes a verifier-weight vector for self-learning or rejects the premise with measured evidence.
6. `exp1273` reports an honest GRPO/VPRM delta if the gate passes.
7. `exp1274` measures continuous self-learning via case-memory replay.
8. `exp1275` measures FSNet-style feasibility improvement against raw Langevin.
9. `exp1276` tests adaptive repair only when the feasibility prerequisite is positive.
10. `exp1277` measures constrained acceptance only when certificates are parseable.
11. `exp1278` writes a final gaming-defense measurement.
12. `exp1279` ships or honestly blocks Kakuro.
13. `exp1280` ships or honestly blocks Masyu.
14. `exp1281` completes the .99 retrospective and proposes carry-forwards.

## Key Planning Decisions

- The milestone sequence increments from 2026.04.98 to 2026.04.99.
- All planned tasks use `agent_type: codex` and `model: gpt-5.5` unless a future conductor operator overrides them. This follows the current CLAUDE.md Codex-by-default policy for formulaic code and bounded research tasks.
- No task modifies `scripts/research_conductor.py`.
- No task modifies `research-roadmap.yaml`; the execution queue is written to `research-roadmap-next.yaml`.
- Legacy tiny LLMs are allowed only as loud CPU smoke-test fallbacks. They are not acceptable headline models for any .99 LLM result.
