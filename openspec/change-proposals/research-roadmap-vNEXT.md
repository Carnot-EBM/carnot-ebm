# Research Roadmap vNEXT - Milestone 2026.07.488

**Milestone title:** Constraint-Tax Structured SOTA, Dependency-Safe Self-Learning, and Solver-Projected Energy Certificates

**Planner date:** 2026-07-07
**Previous milestone:** 2026.07.487
**Task range:** Exp 5349-5362
**Pre-staged roadmap:** `research-roadmap-next.yaml`

## Inputs Read

Required repository inputs were read before planning:

1. `research-program.md`
2. `_bmad/prd.md`
3. `_bmad/architecture.md`
4. `ops/status.md`
5. `ops/changelog.md`
6. `research-complete.yaml`
7. `research-roadmap.yaml`
8. `openspec/change-proposals/`
9. `ops/conductor-log.md`
10. `research-references.md`
11. `research-hardware-wishlist.md`

Additional guardrails checked before writing the roadmap:

- `CLAUDE.md`
- `CODEX.md`
- `ops/exclusion_manifest.yaml`
- `ops/known-issues.md`
- `scripts/roadmap_schema.py`
- `scripts/exclusion_manifest_lint.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `results/experiment_5348_capstone_v487.json`

## Literature Refresh Incorporated

The planner performed a 2025-2026 source refresh and appended the actionable findings to `research-references.md` under `### V488 Planner Refresh - 2026-07-07` before designing experiments.

Promoted sources and planning consequences:

- **Thinking Before Constraining** (`arXiv:2601.07525`): repair structured local SOTA output with a trigger-token/free-reasoning-then-constrain protocol, not a one-pass JSON-only prompt.
- **The Constraint Tax** (`arXiv:2605.26128`) and **Tool Calling Suppression** (`arXiv:2606.25605`): separate syntactic validity, semantic correctness, wrong-valid-output rate, and tool/action reachability; do not collapse all failures into parse rate.
- **Memory-Induced Tool-Drift** (`arXiv:2605.24941`): add a self-learning lane that measures whether remembered context biases future verifier/tool/action selection.
- **ContextWeaver** (`arXiv:2604.23069`): repair the `.487` self-learning TAUTOLOGY by tracking dependency-edge provenance and execution feedback, not duplicate aggregate deltas.
- **NSGGM** (`arXiv:2602.16954`) and OpenReview hard-constrained graph generation with discrete-projection diffusion (`cbtykHVWX9`): test solver-authoritative projection/cuts with fallback completeness.
- **The Shape of Addition** (`arXiv:2606.03645`): watch residual/token features for arithmetic carry slippage only if the local backend exposes real token-probability rows.
- **p-bit schedule papers** (`arXiv:2601.15561`, `arXiv:2604.01564`, `arXiv:2604.17109`): run CPU schedule diagnostics before any hardware speedup claim.

Secondary-source status:

- Semantic Scholar API returned HTTP 429 for citation lookups on EBT `2507.02092` and ARM-EBM `2512.15605`; no citation delta is claimed.
- Extropic writing remains useful as TSU/probabilistic-hardware context only; there is no local TSU hardware path.
- Logical Intelligence/Kona public material is architecture context only; no reproducible local Kona baseline is claimed.
- GitHub/HuggingFace/OpenReview yielded watch items, but none supersedes local deterministic fixtures or the mandated GGUF runtime.

## What 2026.07.487 Proved

The `.487` capstone reports:

- `runtime_clean=true`: at least one mandated local SOTA GGUF runtime path is now usable.
- `structured_output_protocol_ready=false`: the structured protocol was flagged for duration/methodology and is not decision-grade.
- `bounded_sota_quality_usable=false`: the quality panel stayed blocked by the structured protocol.
- `utility_memory_ready=true` and `bounded_compressor_ready=true`: deterministic self-learning components are useful.
- `self_learning_scaleup_ready=false`: the scale-up artifact was quarantined for TAUTOLOGY, so no scaled self-learning claim stands.
- `qstr_fixture_ready=true`, `solver_guidance_ready=true`, and `kan_constraint_bridge_ready=true`: deterministic solver/KAN bridge components are ready for a projection/cuts follow-up.
- `internal_energy_corrigendum_clean=false`: token-probability energy remains blocked/flagged and must not be promoted.
- `hardware_speedup_claim=false`: hardware remains continuity-only; PolarFire produced a workload receipt without a speedup, KV260 was unreachable, and GateMate setup did not change.

## Three Biggest Gaps

1. **Local SOTA generation is runtime-clean but not protocol-clean.** The PRD needs verifiable reasoning outputs from local open models. `.487` proved runtime viability, but structured output, semantic correctness under constraints, and action/tool reachability are still not separated cleanly.

2. **Continuous self-learning has safe parts but no clean scaled loop.** The PRD's FR-11 vision requires durable self-learning under rollback and verifier control. `.487` utility/compressor pieces were ready, but the scale-up duplicated metrics and failed adversarial scrutiny.

3. **Energy/certificate substrates are not yet solver-authoritative end-to-end.** QSTR, solver telemetry, and KAN localization are promising, but token-probability energy is blocked and hardware has no speedup. The next step is solver projection/cuts and p-bit schedule diagnostics, not headline hardware or external text-scoring claims.

## Target Architecture

```text
                 +-------------------------------+
                 |  Mandatory Local SOTA GGUFs   |
                 |  Qwen3.6-35B-A3B / Gemma 31B  |
                 |  Gemma 26B-A4B                |
                 +---------------+---------------+
                                 |
                         trigger-then-constrain
                                 |
                 +---------------v---------------+
                 | Structured Protocol + Tax      |
                 | parse / semantic / wrong-valid |
                 | tool-action reachability       |
                 +---------------+---------------+
                                 |
              +------------------+------------------+
              |                                     |
 +------------v------------+           +------------v------------+
 | Dependency Self-Learning |           | Token/Internal Signals  |
 | utility memory           |           | receipt-clean only      |
 | bounded compressor       |           | no external scorer      |
 | memory-tool drift        |           +------------+------------+
 +------------+------------+                        |
              |                                     |
              +------------------+------------------+
                                 |
                 +---------------v---------------+
                 | Solver-Authoritative Energy    |
                 | QSTR + projection + cuts        |
                 | KAN counterexamples             |
                 | p-bit schedule diagnostics      |
                 +---------------+---------------+
                                 |
                 +---------------v---------------+
                 | Certificates, Hardware Context  |
                 | ARC live-path slot, board       |
                 | receipts, no speedup claim      |
                 +-------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Control

- **Exp 5349:** Archive `.487`, activate `.488`, and record the capstone truth without modifying the active roadmap or conductor.
- **Exp 5350:** Execution-time SOTA/source delta check. Append only genuinely new actionable findings.

### Phase 1 - Structured Local SOTA Under Constraint Tax

- **Exp 5351:** Repair structured output with trigger-then-constrain and a duration/methodology-clean local GGUF receipt.
- **Exp 5352:** If Exp 5351 passes, run a bounded constraint-tax panel that separates schema validity, semantic correctness, wrong-valid outputs, and tool/action reachability.
- **Exp 5353:** Clean the token-probability/internal-energy receipt without reopening the retired external text-scorer scope.
- **Exp 5354:** If Exp 5353 exposes real rows, test a tiny arithmetic carry/token-energy diagnostic.

### Phase 2 - Dependency-Safe Continuous Self-Learning

- **Exp 5355:** Build dependency-edge provenance and execution feedback for context self-learning, repairing the `.487` TAUTOLOGY.
- **Exp 5356:** Add memory-induced tool/action drift detection.
- **Exp 5357:** If both fixtures pass, run the gated self-learning scale-up with separate metrics and rollback.

### Phase 3 - Solver Projection, Schedules, ARC Floor, and Hardware

- **Exp 5358:** Add solver-authoritative projection/cuts and fallback completeness metrics.
- **Exp 5359:** Run CPU p-bit schedule diagnostics for partial deactivation, inertia, and cost landscapes.
- **Exp 5360:** Reserve the mandatory ARC-AGI-3 live-path slot: perception grounding plus classical color-blob salience, with a first-contact level-up attempt and registry discipline.
- **Exp 5361:** Run hardware continuity: KV260 SSH-only precondition, PolarFire workload receipt if reachable, GateMate unchanged blocker, no speedup claim.

### Phase 4 - Capstone

- **Exp 5362:** Synthesize the milestone, update ops docs if required, and select the next decisive gate.

## Dependency Graph

```text
exp5349 --> exp5362
exp5350 --> exp5362

exp5351 --> exp5352 --> exp5362
exp5353 --> exp5354 --> exp5362

exp5355 --> exp5357 --> exp5362
exp5356 --> exp5357 --> exp5362

exp5358 --> exp5362
exp5359 --> exp5362
exp5360 --> exp5362
exp5361 --> exp5362
```

Structured gates in `research-roadmap-next.yaml`:

- Exp 5352 runs only if `exp5351.structured_protocol_clean == true`.
- Exp 5354 runs only if `exp5353.tokenprob_feature_rows_ready == true`.
- Exp 5357 runs only if `exp5355.dependency_provenance_ready == true` and `exp5356.memory_tool_drift_ready == true`.

## Model and Inference Requirements

Every task that needs an LLM must define `MODEL_SPECS` containing the mandated local SOTA GGUF models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The legacy small models may appear only as CPU smoke tests, never as headline-result models. Tasks must not use HuggingFace `AutoTokenizer` on GGUF repositories; they must use llama.cpp/cached GGUF paths as established by `scripts/experiment_template.py`.

## Hardware Requirements

| Substrate | Required in .488 | Claim boundary |
|---|---:|---|
| Dual RTX 3090 CUDA | Required for local GGUF structured-output/token-prob tasks | Runtime/protocol receipts only unless quality gates pass |
| KV260 | SSH reachability check only: `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'` | No host `/dev/mmcblk*` evidence; no speedup claim |
| PolarFire | Board-local hash-verified workload only if reachable | Workload receipt only; no speedup claim |
| GateMate | Only record changed setup or unchanged blocker | No repeated unchanged detect loop |
| Extropic TSU / Kona | Public context only | No local execution or speedup claim |

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen the retired Phase D external generated-text/logprob scorer scope.
- Do not claim SOTA quality from parse-only protocol work.
- Do not claim token-probability or internal-energy discrimination unless real backend features are present and the receipt is methodology-clean.
- Do not claim hardware acceleration without authenticated local hardware workload and a valid comparison.
- Do not solve ARC with outer-loop reverse engineering, per-game adapters, or offline ground-truth BFS. The `.488` ARC task must be live-path reachable and must declare `solve_provenance`.

## Expected End State

By the end of `.488`, Carnot should know whether:

- trigger-then-constrain makes local SOTA structured outputs clean enough for constrained verification;
- constraint-tax metrics expose a real gap between valid-looking and semantically correct outputs;
- token-probability energy has sufficient local backend features to continue;
- continuous self-learning can scale with dependency provenance and memory-tool drift controls without TAUTOLOGY;
- solver projection/cuts produce safer certificates than neural hints alone;
- p-bit schedule variants merit hardware work;
- the ARC live path gains or honestly fails a perception/salience level-up attempt; and
- hardware remains continuity-only or has new authenticated board receipts.
