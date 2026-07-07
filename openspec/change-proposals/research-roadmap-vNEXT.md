# Research Roadmap vNEXT - Milestone 2026.07.489

**Milestone title:** Grammar-Budgeted SOTA, Budget-Curated Self-Learning, and Overwrite-Capable Solver Guidance

**Planner date:** 2026-07-07
**Previous milestone:** 2026.07.488
**Task range:** Exp 5363-5375
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
- `ops/arc_solve_registry.yaml`
- `scripts/roadmap_schema.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `results/experiment_5351_trigger_constrain_structured_protocol_v488.json`
- `results/experiment_5352_gated_constraint_tax_tool_action_panel_v488.json`
- `results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json`
- `results/experiment_5354_arithmetic_carry_token_energy_v488.json`
- `results/experiment_5355_dependency_provenance_self_learning_v488.json`
- `results/experiment_5356_memory_tool_drift_harness_v488.json`
- `results/experiment_5357_dependency_drift_self_learning_scaleup_v488.json`
- `results/experiment_5358_solver_projection_cut_bridge_v488.json`
- `results/experiment_5359_pbit_schedule_diagnostic_v488.json`
- `results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json`
- `results/experiment_5361_hardware_continuity_workload_v488.json`
- `results/experiment_5362_capstone_v488.json`

## Literature Refresh Incorporated

The planner performed a 2025-2026 source refresh and appended the actionable findings to `research-references.md` under `### V489 Planner Refresh - 2026-07-07` before designing experiments.

Promoted sources and planning consequences:

- **G-RRM** (`arXiv:2607.02491`): neural full-solution proposals help when the solver can overwrite bad hints. `.489` adds an overwrite-capable solver-guidance matrix rather than a forced-hint speedup claim.
- **Budget-Curated Memory** (`arXiv:2606.25115`): self-learning should score memory by value-minus-harm per byte under RAM/energy/uplink budgets. `.489` extends the clean `.488` dependency/drift artifacts into budgeted continuous self-learning.
- **ALMA** (`arXiv:2602.07755`): memory-design search is useful context but remains too open-ended for Carnot without deterministic safety gates. `.489` keeps memory governance bounded and verifier-scored.
- **CFGzip** (`arXiv:2605.29986`) and **TruncProof** (`arXiv:2605.13076`): structured local SOTA should compile grammar/token reachability and completion slack before live generation. `.489` gates the live protocol on this preflight.
- **FLaG** (`arXiv:2606.00301`) and **Thermodynamic Signatures of Reasoning** (`arXiv:2606.19404`): hidden-state/attention families are promising, but `.489` treats token/internal energy as a precondition/corrigendum lane unless the runtime exposes the needed tensors.
- **Programmable Probabilistic Computer with 1,000,000 p-bits** (`arXiv:2606.25313`): boundary exchange and communication-to-p-bit timing ratio are the next p-bit diagnostic, still CPU simulation-only.

Secondary-source status:

- Semantic Scholar rate-limited EBT `2507.02092`; ARM-EBM `2512.15605` had eight citations and two influential citations. Citation-trail ideas are watch items, not a reopened external text-scorer lane.
- OpenReview, HuggingFace Papers, and GitHub yielded implementation references for constrained decoding and solver guidance, but none replaces local GGUF, deterministic solvers, or live ARC requirements.
- Extropic TSU/XTR writing remains hardware architecture context only; no Carnot-accessible TSU execution path exists.
- Logical Intelligence Kona/Aleph pages support the architecture bias toward verifier/prover authority; no reproducible Kona baseline is claimed.

## What 2026.07.488 Proved

The `.488` capstone reported a mixed but useful milestone:

- `structured_protocol_clean=false`: live local SOTA JSON generation still failed schema cleanliness despite real GGUF execution. Exp5351 reached parse success on some prompts but only `schema_success_rate=0.25`.
- `constraint_tax_panel_ready=false`: the downstream panel was correctly gate-blocked because the structured protocol did not satisfy the clean gate.
- `tokenprob_feature_rows_ready=true` but token/internal-energy work remained adversarially flagged. Exp5353 had usable logprob row shape, while Exp5354 found no carry-token energy margin and was flagged for tautology/methodology.
- `dependency_provenance_ready=true`, `memory_tool_drift_ready=true`, and `self_learning_scaleup_ready=true`: the dependency/drift self-learning lane is now clean, with no weight mutation, rollback recovery, and measurable context/verifier savings.
- `solver_projection_ready=true`: the solver-authoritative projection/cuts bridge is clean and ready for overwrite-aware guidance.
- `pbit_schedule_signal_ready=true`: CPU p-bit schedule diagnostics produced useful class-level harm/signal data, but no hardware speedup.
- `arc_new_level_banked=false`: the required live ARC slot ran honestly but did not bank re86 L3; the salience/status-bar error classes are now concrete.
- `hardware_speedup_claim=false`: PolarFire produced a workload receipt, KV260 remained unreachable by SSH, GateMate remained physically/JTAG blocked, and no board speedup was claimed.

## Three Biggest Gaps

1. **Local SOTA is real but not yet structurally reliable.** The PRD requires verifiable reasoning outputs from local open models. `.488` proved live inference can run but not that JSON/schema/tool-action outputs are clean enough for downstream scoring.

2. **Continuous self-learning is clean but not budget-governed.** FR-11 needs a durable self-learning loop that can retain useful constraints while rejecting stale, poisoned, or over-expensive memory. `.488` cleaned dependency/drift mechanics; `.489` must add memory budget, provenance, and trust decisions.

3. **Energy guidance is promising but must stay solver-authoritative.** Solver projection and p-bit diagnostics are ready, but token/internal energy is not. `.489` should use overwrite-capable symbolic solvers and boundary-exchange diagnostics while treating neural signals as hints and preserving fallback completeness.

## Target Architecture

```text
                 +---------------------------------------+
                 | Mandatory Local SOTA GGUF Substrate   |
                 | Qwen3.6-35B-A3B / Gemma 31B /         |
                 | Gemma 26B-A4B via llama.cpp/GGUF      |
                 +-------------------+-------------------+
                                     |
                         grammar budget + JSON slack
                                     |
                 +-------------------v-------------------+
                 | Structured Protocol Repair             |
                 | schema reachability / token budget     |
                 | parse, schema, semantic, wrong-valid   |
                 | tool-action reachability               |
                 +-------------------+-------------------+
                                     |
              +----------------------+----------------------+
              |                                             |
 +------------v-------------+                 +-------------v------------+
 | Budget-Curated Memory    |                 | Internal Feature Ledger  |
 | dependency provenance     |                 | logprob/logit/hidden/    |
 | KEEP/SHARE/TRUST          |                 | attention availability   |
 | rollback + no mutation    |                 | no text-only energy      |
 +------------+-------------+                 +-------------+------------+
              |                                             |
              +----------------------+----------------------+
                                     |
                 +-------------------v-------------------+
                 | Solver-Authoritative Energy Guidance   |
                 | overwrite-capable hints / projection   |
                 | cuts / p-bit boundary timing           |
                 +-------------------+-------------------+
                                     |
                 +-------------------v-------------------+
                 | Live Verification Surface              |
                 | ARC self-discovery slot / board        |
                 | receipts / capstone gates              |
                 +---------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Fresh Source Delta

**Exp5363-Exp5364**

Archive the completed `.488` state into the `.489` execution context, then perform an execution-time source delta check so the conductor can catch anything discovered after the planner sweep.

Deliverables:

- `results/experiment_5363_transition_v489.json`
- `results/experiment_5364_sota_source_delta_v489.json`

### Phase 1 - Grammar-Budgeted Structured SOTA

**Exp5365-Exp5367**

Repair the failed structured local SOTA lane in two steps: first build a deterministic grammar/token-budget preflight from CFGzip/TruncProof ideas, then run a live mandated-GGUF protocol only if the preflight is ready. The constraint-tax panel remains gated on `structured_protocol_clean=true`.

Deliverables:

- `results/experiment_5365_grammar_budget_protocol_preflight_v489.json`
- `results/experiment_5366_live_grammar_budgeted_sota_protocol_v489.json`
- `results/experiment_5367_constraint_tax_tool_action_panel_v2_v489.json`

### Phase 2 - Budget-Curated Continuous Self-Learning

**Exp5368-Exp5369**

Extend the clean `.488` dependency-provenance and memory-drift work into budgeted self-learning: memory must prove value-minus-harm per byte, no weight mutation, rollback recovery, stale/poison deflection, and quality retention under larger traces.

Deliverables:

- `results/experiment_5368_budget_curated_memory_governance_v489.json`
- `results/experiment_5369_budgeted_continuous_self_learning_scaleup_v489.json`

### Phase 3 - Solver Guidance, Internal-Feature Preconditions, ARC, and Hardware

**Exp5370-Exp5374**

Convert the clean solver projection and p-bit schedule results into the next bounded tests: overwrite-capable solver guidance, boundary-exchange p-bit timing, and a token/internal-feature continuation gate that prevents text-only energy overclaims. The milestone also includes the required ARC live-path level-up slot and hardware continuity receipt.

Deliverables:

- `results/experiment_5370_overwrite_solver_guidance_matrix_v489.json`
- `results/experiment_5371_pbit_boundary_exchange_schedule_v489.json`
- `results/experiment_5372_token_feature_precondition_gate_v489.json`
- `results/experiment_5373_arc_salience_re86_levelup_v489.json`
- `results/experiment_5374_hardware_continuity_receipts_v489.json`

### Phase 4 - Capstone

**Exp5375**

Synthesize the phase results into milestone gates and next-step decisions.

Deliverable:

- `results/experiment_5375_capstone_v489.json`

## Dependency Graph

```text
Exp5363 transition
  |
  +--> Exp5364 source delta
  |
  +--> Exp5365 grammar-budget preflight
          |
          +-- gated_on grammar_budget_protocol_ready == true
              Exp5366 live grammar-budgeted SOTA protocol
                  |
                  +-- gated_on structured_protocol_clean == true
                      Exp5367 constraint-tax tool/action panel v2

Exp5368 budget-curated memory governance
  |
  +-- gated_on budget_curated_memory_ready == true
      Exp5369 budgeted continuous self-learning scale-up

Exp5358/Exp5359 prior clean artifacts
  |
  +--> Exp5370 overwrite-capable solver guidance
  +--> Exp5371 p-bit boundary-exchange schedule

Exp5353/Exp5354 flagged artifacts
  |
  +--> Exp5372 token/internal-feature precondition gate

Exp5360 ARC honest-null artifact
  |
  +--> Exp5373 live ARC salience repair and re86 +1 attempt

Exp5361 hardware receipt
  |
  +--> Exp5374 hardware continuity receipts

Exp5363-Exp5374
  |
  +--> Exp5375 capstone
```

## Model and Inference Requirements

Any task that calls an LLM must declare and use the mandated local GGUF SOTA model specs:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The legacy small models may be used only as CPU smoke tests and cannot supply headline results. GGUF repositories must run through llama.cpp-compatible GGUF loading paths; AutoTokenizer/AutoModel loading against a `-GGUF` repository is forbidden. Each live LLM artifact must record `MODEL_SPECS`, selected model, inference substrate, GPU/offload receipt, parser/schema metrics, unsafe false accepts, and no-autotokenizer evidence.

## Hardware Requirements

- **Dual RTX 3090 CUDA:** required for live SOTA protocol tasks unless the task writes an honest blocked artifact. CPU-only SOTA headline reruns are retired and must not be revived.
- **KV260:** check only by SSH, for example `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`. Do not inspect host `/dev/mmcblk*` devices as board evidence.
- **PolarFire:** if reachable, run the established workload receipt path and record output hashes/timing as receipts only.
- **GateMate:** perform current detect/toolchain checks only if the board/JTAG path is available; do not rerun unchanged blocked loops.
- **Extropic TSU / Kona:** architecture references only. No execution, speedup, or baseline claim is allowed.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired external text-scorer, CPU-only GGUF offload, or ARC candidate-exploration-signal scopes.
- Do not claim ARC solves from offline ground-truth BFS, hand-built per-game adapters, or outer-loop reverse engineering. The credited ARC path is live-agent self-discovery.
- Do not claim hardware speedup without authenticated board timing, baseline timing, workload hash, and repeatability evidence.
- Do not promote token/internal-energy signals from text-only completions or incomplete logprob rows.

## Expected End State

At the end of `.489`, Carnot should know whether the structured local SOTA lane can be repaired by grammar-budget preflight and completion slack, whether continuous self-learning remains clean under budget-curated memory governance, whether solver hints help only under overwrite-capable symbolic authority, and whether ARC/hardware continuity progressed without overclaiming. A clean milestone would unlock a later quality panel and broader solver-guided verification experiments; a blocked milestone should retire or narrow the failed lanes with explicit preconditions.
