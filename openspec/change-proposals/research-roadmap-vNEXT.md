# Research Roadmap vNEXT - Milestone 2026.07.490

**Milestone title:** Clean Structured SOTA Receipts, Audited Self-Learning, and Geometry-Guided Live Verification

**Planner date:** 2026-07-07
**Previous milestone:** 2026.07.489
**Task range:** Exp 5376-5388
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
- `scripts/experiment_template.py`
- `scripts/roadmap_schema.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `results/experiment_5375_capstone_v489.json`

## Literature Refresh Incorporated

The planner performed a 2025-2026 source refresh and appended the actionable findings to
`research-references.md` under `### V490 Planner Refresh - 2026-07-07` before designing
experiments.

Promoted sources and planning consequences:

- **DEX** (`arXiv:2606.29223`): multi-depth decoding with final-depth validation is promising, but
  Carnot should treat it as a backend precondition/watch lane until the local GGUF runtime exposes
  intermediate-layer or depth exits.
- **GeoWorld** (`arXiv:2602.23058`): energy-based predictive world models and hyperbolic/geodesic
  planning motivate a live ARC geometric salience diagnostic over the agent's own observations,
  without per-game adapters or offline BFS.
- **QCIVET** (`arXiv:2605.13109`): contract checks plus hash-chained audit traces motivate stronger
  board workload receipts for KV260, PolarFire, and GateMate.

Secondary-source status:

- Semantic Scholar citation checks for EBT `2507.02092` and ARM-EBM `2512.15605` were rate-limited;
  no fresh citation-count claim is made.
- OpenReview constrained-guidance entries were checked. DANCE-ST is watch-only because it was
  desk-rejected and is not a core experiment basis.
- HuggingFace Papers and GitHub did not surface a new reproducible local engine beyond already-filed
  hooks such as `llguidance`, CFGzip, TruncProof, G-RRM, and LongMemEval-V2.
- Extropic TSU/X0/XTR writing and Logical Intelligence Kona/Aleph pages remain architecture context,
  not executable baselines.

## What 2026.07.489 Proved

The `.489` capstone closed a milestone with useful positives and two important blocked gates:

- `grammar_budget_protocol_ready=true`: the structured-output preflight lane is ready.
- `structured_protocol_clean=false`: Exp5366 produced clean parse/schema/semantic/final-JSON rates
  and zero unsafe false accepts, but failed the clean gate because `methodology_duration_s=19.445366`
  was below the required 60-second receipt.
- `constraint_tax_panel_ready=false`: Exp5367 was correctly gate-blocked because the structured
  protocol did not satisfy `structured_protocol_clean=true`.
- `budget_curated_memory_ready=true` in the capstone, but the conductor log marked Exp5368 with an
  adversarial TAUTOLOGY flag. The result must be cleaned before it becomes a scaling prerequisite.
- `continuous_self_learning_budget_scaleup_ready=true`: Exp5369 showed positive context and verifier
  cost savings without weight mutation, but should now be tied to corrected memory governance.
- `overwrite_solver_guidance_ready=true`: Exp5370 confirmed solver-authoritative guidance is viable,
  with `overwrite_rate=0.5909`, `fallback_completeness_rate=1.0`, and no unsafe false accepts. The
  remaining issue is improving and explaining `post_projection_validity_rate=0.8571`.
- `boundary_exchange_schedule_ready=true`: Exp5371 produced a CPU-only p-bit boundary diagnostic.
  It remains simulation-only and cannot support a hardware speedup claim.
- `future_token_signal_allowed=false`: Exp5372 found the needed logits/hidden/attention features
  unavailable. Token/internal-feature energy remains closed until backend evidence changes.
- `arc_new_level_banked=false`: Exp5373 honestly attempted re86 L3 and did not bank a new live level.
- `hardware_speedup_claim=false`: Exp5374 found KV260 unreachable by SSH, PolarFire reachable with a
  workload receipt, GateMate still blocked physically/JTAG, and no repeatability evidence sufficient
  for speedup.

## Three Biggest Gaps

1. **The local SOTA structured lane has quality but not a valid live receipt.** The PRD needs
   verifiable reasoning outputs from local open models. `.489` showed the grammar path can work, but
   the evidence window was too short to unlock downstream constraint-tax scoring.

2. **Continuous self-learning is promising but one governance artifact is tainted.** FR-11 requires
   a durable learning loop with rollback, provenance, and stale/poison controls. `.489` produced a
   positive budget scale-up, but Exp5368 must be re-emitted from row-level evidence before the
   memory policy becomes a reliable dependency.

3. **Energy guidance, ARC, and hardware still lack live-reachable authenticated evidence.** Solver
   overwrite and p-bit schedules are promising, but token features are unavailable, ARC did not bank
   a new live level, and board receipts are not yet repeatable or hash-chained.

## Target Architecture

```text
                 +---------------------------------------+
                 | Mandatory Local SOTA GGUF Substrate   |
                 | Qwen3.6-35B-A3B / Gemma 31B /         |
                 | Gemma 26B-A4B via llama.cpp/GGUF      |
                 +-------------------+-------------------+
                                     |
                      >=60s live receipt + schema grammar
                                     |
                 +-------------------v-------------------+
                 | Clean Structured SOTA Protocol         |
                 | parse/schema/semantic/final JSON       |
                 | wrong-valid and tool/action evidence   |
                 +-------------------+-------------------+
                                     |
                       gate only if structured clean
                                     |
                 +-------------------v-------------------+
                 | Constraint-Tax Tool/Action Panel       |
                 | initial/final state and verifier rows  |
                 +-------------------+-------------------+
                                     |
           +-------------------------+-------------------------+
           |                                                   |
+----------v-----------+                         +-------------v------------+
| Audited Self-Learning|                         | Solver/P-bit Guidance    |
| row-level memory     |                         | solver overwrites hints  |
| budgets, trust,      |                         | p-bit boundary CPU diag  |
| rollback, no weights |                         | no hardware speedup      |
+----------+-----------+                         +-------------+------------+
           |                                                   |
           +-------------------------+-------------------------+
                                     |
                 +-------------------v-------------------+
                 | Live Verification Surfaces             |
                 | ARC geometric salience, backend feature |
                 | gates, hash-chained board receipts      |
                 +---------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Delta

Exp5376 archives the `.489` capstone, records the completed milestone state, and opens `.490`
without editing the active roadmap. Exp5377 performs the required execution-time literature/source
delta so the conductor has fresh citations and no duplicate retired-scope proposals.

### Phase 1 - Clean Local SOTA Structured Protocol

Exp5378 repairs the actual `.489` blocker: the live structured receipt must run long enough and
produce auditable GPU/offload evidence with a mandated local SOTA GGUF. Exp5379 is gated on that
preflight and reruns the live clean structured gate. Exp5380 is gated on Exp5379 and retries the
constraint-tax tool/action panel only if the structured protocol is clean.

### Phase 2 - Audited Continuous Self-Learning and Solver Energy

Exp5381 re-emits the budget-curated memory artifact from row-level evidence with anti-tautology
controls. Exp5382 is gated on that corrigendum and runs the required continuous self-learning
experiment on a real multi-session workflow, preserving no-weight-mutation and rollback. Exp5383
scales overwrite-capable solver guidance while explaining invalid post-projection cases. Exp5384
joins the p-bit boundary diagnostic with overwrite guidance in CPU simulation only.

### Phase 3 - Live Surfaces, Hardware Receipts, and Capstone

Exp5385 attempts a live ARC geometric salience improvement, with `solve_provenance` explicitly set
to `live_agent_self_discovery`. Exp5386 creates QCIVET-style hash-chained board receipts for KV260,
PolarFire, and GateMate without claiming speedup. Exp5387 decides whether token/internal-feature
energy can reopen; by default it remains closed unless backend feature receipts are present. Exp5388
aggregates the milestone and states exactly which lanes are ready, blocked, retired, or still
watch-only.

## Dependency Graph

```text
Exp5376 transition
  -> Exp5377 source delta
  -> Exp5378 structured methodology-duration receipt
       -> Exp5379 live structured clean rerun
            -> Exp5380 constraint-tax retry

Exp5376 transition
  -> Exp5381 memory-tautology corrigendum
       -> Exp5382 real-workflow continuous self-learning

Exp5376 transition
  -> Exp5383 overwrite guidance scale/validity
       -> Exp5384 p-bit boundary + overwrite diagnostic

Exp5376 transition
  -> Exp5385 ARC geometric salience live path
  -> Exp5386 hardware hash-chain receipts
  -> Exp5387 token/backend reopen gate

All phase outputs
  -> Exp5388 capstone
```

Structured conductor gates are used where a downstream task should be skipped before an agent call:

- Exp5379 requires `Exp5378.live_sota_receipt_ready == true` and
  `Exp5378.methodology_duration_s >= 60.0`.
- Exp5380 requires `Exp5379.structured_protocol_clean == true`.
- Exp5382 requires `Exp5381.budget_memory_corrigendum_clean == true`.

## Model and Inference Requirements

Any experiment that runs an LLM must use at least one mandated local GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models such as Qwen3.5-0.8B or gemma-4-E4B-it are allowed only as fast CPU smoke tests.
Headline results must come from the SOTA GGUF path using the repository's `cached_sota_pair()` style
and must not load `-GGUF` repositories through `AutoTokenizer` or `AutoModel`.

## Hardware Requirements

- **Dual RTX 3090 / local llama.cpp GGUF:** required for Exp5378-Exp5380 if live SOTA inference runs.
  CPU-only GGUF offload is not a headline path and must block or mark smoke-test-only.
- **KV260:** Exp5386 may check SSH reachability with
  `ssh -o ConnectTimeout=5 -o BatchMode=yes kria true`. It must not use host `/dev/mmcblk*` evidence
  or perform destructive flashing.
- **PolarFire:** Exp5386 should reuse the reachable workload-receipt path and add hash-chain fields.
- **GateMate / DirtyJTAG:** Exp5386 may record current physical/JTAG status and toolchain receipts,
  but must not repeat an unchanged blocked loop as a success.
- **Extropic TSU and Logical Kona:** architecture context only. No execution or speedup claim.

## No-Go Rules

- Do not modify `research-roadmap.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Do not reopen retired external generated-text scoring, broad CPU-only GGUF offload headline runs,
  KV260 host SD-card preconditions, TSU/Kona execution claims, or offline ARC solve paths.
- Do not claim token/internal-feature energy unless logits, hidden states, or attention are available
  with clean provenance.
- Do not claim hardware speedup without board timing, repeatability, and authenticated workload
  receipts.
- Do not count ARC solves from outer-loop reverse engineering, offline BFS, or per-game adapters.
