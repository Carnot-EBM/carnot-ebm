# Research Roadmap vNEXT - Milestone 2026.07.498

**Milestone title:** CSL Corrigendum, Preference-MaxSAT Verification, Concept Evidence Telemetry, Hardware Receipts, and ARC Trajectory Induction

**Planner date:** 2026-07-09
**Previous milestone:** 2026.07.497
**Task range:** Exp 5482-5495
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
- `ops/arc_solve_registry.yaml`
- `scripts/experiment_template.py`
- `scripts/roadmap_schema.py`
- `scripts/conductor_gates.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `ops/e2e-test-plan.md`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview,
Extropic writing, Semantic Scholar citation routes for EBT `2507.02092` and
ARM-EBM `2512.15605`, HuggingFace Papers, GitHub repositories, and Logical
Intelligence public pages before designing experiments. Actionable deltas were
appended to `research-references.md` under
`### V498 Planner Refresh - 20260709`.

New planning consequences:

- **Preference-Based Maximum Satisfiability** (`arXiv:2605.29687`) gives V498
  a clean way to compose hard exact constraints with soft user/preferences
  without reopening token steering. V498 adds a typed claim-state fixture with
  independently checked reference semantics.
- **Natural Language based Specification and Verification** (`arXiv:2605.11315`)
  motivates helper contracts written as natural-language specs only if they
  compile to executable predicates or exact tests. V498 applies this to
  deterministic witness rows, not broad code generation.
- **Concept-level EBM interpretability** (OpenReview `Uh0F0079Lh`) motivates
  advisory concept-energy attribution for local SOTA evidence telemetry. The
  concept energy is explanatory only; exact validators remain final authority.
- **LatentGym** (`arXiv:2606.15306`) becomes the CSL measurement reference
  after Exp5474 was adversarially flagged for metric tautology. V498 separates
  exploration, exploitation, memory support, and exact outcome metrics before
  another SOTA CSL headline.
- **Pitwall** (`arXiv:2607.06495`) remains the typed factual-claim reference
  for decomposing SOTA outputs into state-linked claims with support/fallback
  accounting.
- **CARS** (OpenReview `OhndOnT4Ih`) is promoted only as a complete-candidate
  rejection/abstention pattern. It does not lift guided-decoding quarantine and
  does not justify token-level steering.
- **Implicitly Parallel Neuromorphic Solver Design for CSPs**
  (`arXiv:2603.01150`) was already indexed; V498 folds its partition/update
  telemetry idea into active-constraint descriptors and hardware receipts.

Secondary-source status:

- Extropic TSU/XTR-0/Z1 writing remains architecture context only; Carnot has
  no local authenticated TSU execution path.
- Logical Intelligence Aleph/Kona pages reinforce EBRM-style global-state
  reasoning and proof-bearing code verification, but no local authenticated
  access exists.
- Semantic Scholar public routes for EBT and ARM-EBM surfaced adjacent items
  but no stronger executable dependency than the source EBT/ARM-EBM hooks and
  the V498 MaxSAT/concept-energy additions.
- HuggingFace Papers verification pages sharpen fixtures but do not replace
  exact validators or local GGUF receipt discipline.
- GitHub EBM/KAN/constraint repositories remain watch references; none
  superseded Carnot's local exact solvers, GGUF runtime, or hardware receipts.

## What 2026.07.497 Proved

The `.497` capstone, conductor log, and results artifacts are the immediate
planning source of truth:

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and source freshness | 5468, 5469 | `.497` activated cleanly and execution-time source deltas were reconciled. |
| Rewrite-state verification | 5470, 5471 | Deterministic rewrite-state, semantic guards, and guard composition are clean; hidden-premise and unlicensed-change fixtures are usable. |
| Local SOTA evidence telemetry | 5472 | A bounded local SOTA GGUF telemetry run succeeded with GPU offload and no guided decoding; exact validators remain authority. |
| Continuous self-learning | 5473, 5474, 5475 | KAN surrogate assurance and behavioral memory evidence were clean. Exp5474's artifact showed a large CSL scale delta, but the conductor flagged it `TAUTOLOGY`; V498 must resolve metric independence before using it as a headline. |
| Helper repair | 5476 | Helper-lemma witness repair reduced repeated failure classes without false accepts, with exact rechecks. |
| Active constraints and hardware | 5477, 5478 | p-bit/p-dit boundary exchange and board receipts are bounded. PolarFire receipts matched hashes; KV260 remained blocked; no speedup claim is supported. |
| ARC | 5479, 5480 | Target precheck selected `sb26 L3`; live salience attempt was an honest null/no-bank with no registry delta. |
| Synthesis | 5481 | `.497` closed verifiable reasoning and governed CSL as the strongest lanes, kept guided decoding quarantined, recorded ARC and hardware speedup as honest nulls, and recommended trajectory generation beyond salience. |

## Three Biggest Gaps

1. **CSL scale is promising but not headline-clean.** The PRD's FR-11 asks for
   autonomous directed self-learning, and `.497` produced strong KAN-assisted
   CSL numbers. The adversarial tautology flag means V498 must first prove
   metric independence, then rerun SOTA CSL on exploration/exploitation splits
   whose outcome metrics are independent of the policy score.

2. **Verification needs hard-plus-soft semantics.** Deterministic exact guards
   now catch unlicensed state changes, but practical reasoning also needs
   preference ranking among admissible repairs, concept attribution, and
   helper contracts that bridge natural language and exact predicates. V498
   adds Preference-MaxSAT, concept-energy telemetry, and NL helper-contract
   repair under independent exact validators.

3. **Grounding is still bounded in both hardware and ARC.** Hardware work has
   authenticated receipts but no speedup; ARC has a long no-bank tail after the
   69-level plateau. V498 should improve descriptors and trajectory induction
   rather than claiming speedup or repeating salience-only ARC attempts.

## Target Architecture

```text
                         +--------------------------------------+
                         | Local SOTA GGUF inference substrate  |
                         | Qwen3.6-35B-A3B, Gemma-4-31B-it,     |
                         | Gemma-4-26B-A4B-it via llama.cpp     |
                         +-------------------+------------------+
                                             |
                       complete candidates, receipts, optional logits
                                             |
        +------------------------------------v----------------------------------+
        | Typed claim-state and preference verification                         |
        | hard exact constraints, soft preferences, canonical MaxSAT refs,      |
        | typed factual claims, concept-energy attribution, abstention routing  |
        +-------------------+--------------------------+------------------------+
                            |                          |
              exact witness stream                     | governed experience stream
                            |                          |
        +-------------------v----------------+   +-----v------------------------+
        | Helper-contract repair             |   | Continuous self-learning      |
        | NL spec rows -> executable          |   | metric-independence audit,    |
        | predicates/tests, verifier failure  |   | latent exploration/replay,    |
        | signatures, exact rechecks          |   | KAN surrogate, frozen GGUF    |
        +-------------------+----------------+   +-----+------------------------+
                            |                          |
                            |                 descriptor / partition stream
                            |                          |
        +-------------------v--------------------------v------------------------+
        | Active constraints, hardware receipts, and ARC live path              |
        | p-bit/p-dit/MaxSAT descriptors, partition/update telemetry, matched    |
        | board hashes/timing, live ARC trajectory induction, registry gates     |
        +-----------------------------------------------------------------------+
```

## Phase Plan

### Phase 0 - Transition and Source Delta

- **Exp5482:** archive `.497` terminal evidence and stage `.498` execution
  context, including the Exp5474 tautology flag, guided-decoding quarantine,
  ARC no-bank, and hardware no-speedup facts.
- **Exp5483:** run execution-time source delta against the V498 planner refresh
  and append only non-duplicate actionable references.

### Phase 1 - Metric Corrigendum and Hard/Soft Verification

- **Exp5484:** resolve the Exp5474 CSL tautology finding with a metric
  independence graph and clean/bounded/null recommendation.
- **Exp5485:** build a deterministic Preference-MaxSAT typed claim-state
  fixture with hard exact constraints, soft preferences, canonical references,
  and false-accept accounting.
- **Exp5486:** if Exp5485 is clean, run a local SOTA GGUF concept-attributed
  evidence telemetry panel. This is not guided decoding and does not use token
  steering.
- **Exp5487:** extend helper-lemma witness repair into NL helper contracts that
  must compile to executable predicates or exact tests before being credited.

### Phase 2 - Continuous Self-Learning With Independent Metrics

- **Exp5488:** required CSL task. Build a LatentGym-style deterministic replay
  fixture that separates exploration, exploitation, memory support, and exact
  outcome metrics.
- **Exp5489:** if Exp5484 and Exp5488 are clean, rerun local SOTA GGUF CSL
  scale-up with independent metrics, frozen weights, and mandatory GGUF model
  receipts.
- **Exp5490:** map the governed CSL/KAN update path to hardware-compatible
  sparse fixed-point update ledgers and resource estimates without making a
  board speedup claim.

### Phase 3 - Boundary Exchange, Hardware Receipts, ARC, and Synthesis

- **Exp5491:** build active-constraint subproblem descriptors for p-bit/p-dit,
  Preference-MaxSAT, and exact fallback paths, including partition/update
  telemetry inspired by neuromorphic CSP work.
- **Exp5492:** if Exp5491 is ready, collect matched CPU/board receipts where
  reachable. This remains receipt-only; no speedup claim is allowed without
  authenticated matched timing.
- **Exp5493:** run ARC trajectory-target precheck that avoids duplicate levels,
  recent no-bank targets, offline BFS, and generic retired exploration signals.
- **Exp5494:** if Exp5493 is clean, run one live ARC trajectory-induction
  level-up attempt with `solve_provenance=live_agent_self_discovery`.
- **Exp5495:** emit the `.498` capstone with PRD gap table, failure taxonomy,
  headline/bounded/blocked truth table, and ops-doc alignment recommendations.

## Natural Next-Experiment Chain

```text
Exp5474 TAUTOLOGY flag
  -> Exp5484 CSL metric-independence corrigendum
      -> Exp5488 latent exploration/replay split
          -> Exp5489 SOTA GGUF CSL independent-metrics panel
              -> Exp5495 capstone headline decision

Exp5470/5471 deterministic guards + Exp5472 local SOTA telemetry
  -> Exp5485 Preference-MaxSAT hard/soft claim-state fixture
      -> Exp5486 concept-attributed SOTA evidence telemetry
      -> Exp5487 NL helper-contract repair

Exp5477 p-bit/p-dit boundary exchange + Exp5478 hardware receipts
  -> Exp5491 active-constraint subproblem descriptors
      -> Exp5492 matched hardware receipt continuation

Exp5480 sb26 L3 salience no-bank
  -> Exp5493 trajectory target precheck
      -> Exp5494 live trajectory-induction level-up attempt
```

## Dependency Graph

```text
5482 transition
  |
5483 source-delta

5484 CSL tautology corrigendum
  +--> 5488 CSL latent exploration replay
          +--> 5489 SOTA CSL independent metrics
                 |
                 v
              5495 capstone

5485 Preference-MaxSAT typed claims
  +--> 5486 SOTA concept evidence telemetry
  +--> 5487 NL helper-contract repair

5488 CSL replay
  +--> 5490 KAN fixed-point update ledger

5491 active-constraint descriptor
  +--> 5492 hardware receipts

5493 ARC target precheck
  +--> 5494 ARC live trajectory attempt
        |
        v
      5495 capstone
```

## Hardware Requirements

- **Dual RTX 3090 / CUDA:** required for Exp5486 and Exp5489 if any headline
  local SOTA GGUF inference runs. Use the repo SOTA cache resolver or
  `cached_sota_pair()` pattern from `scripts/experiment_template.py`.
- **Mandated local GGUF model specs:** every LLM experiment must include at
  least one of, and the prompts list all three:
  `unsloth/Qwen3.6-35B-A3B-GGUF`,
  `unsloth/gemma-4-31B-it-GGUF`, and
  `unsloth/gemma-4-26B-A4B-it-GGUF`. Legacy small models may only be used as
  CPU smoke tests and must never be headline results.
- **GGUF runtime discipline:** no `AutoTokenizer.from_pretrained` on GGUF repos;
  use CUDA-enabled `llama-cpp-python` or native `llama.cpp` with explicit GPU
  offload receipts, model paths, checksums, and VRAM/runtime data.
- **Exact solvers and deterministic fixtures:** Exp5484, Exp5485, Exp5487,
  Exp5488, Exp5490, Exp5491, and Exp5493 can run on CPU unless existing tests
  require otherwise.
- **Boards:** PolarFire, KV260, and GateMate are receipt-only. KV260 checks must
  use SSH/board identity, never host `/dev/mmcblk*`. GateMate physical/JTAG
  remains blocked unless local access changes. No hardware speedup claim is
  allowed without matched workload hashes, board identity, repeated timing, and
  an embedded/local baseline.
- **External hardware:** Extropic TSU, Logical Intelligence Kona/Aleph, and any
  non-local accelerator remain watch-only and cannot support execution claims.

## Experiment Summary

| Exp | Title | Gate | Deliverable |
|-----|-------|------|-------------|
| 5482 | Transition `.497` outcomes into `.498` context | none | `results/experiment_5482_transition_v498.json` |
| 5483 | Execution-time source delta for `.498` | none | `results/experiment_5483_source_delta_v498.json` |
| 5484 | CSL tautology corrigendum and metric independence | none | `results/experiment_5484_csl_tautology_corrigendum_v498.json` |
| 5485 | Preference-MaxSAT typed claim-state fixture | none | `results/experiment_5485_preference_maxsat_claim_fixture_v498.json` |
| 5486 | SOTA concept evidence telemetry | Exp5485 ready | `results/experiment_5486_sota_concept_evidence_panel_v498.json` |
| 5487 | NL helper-contract repair | none | `results/experiment_5487_helper_contract_nl_spec_repair_v498.json` |
| 5488 | CSL latent exploration replay | Exp5484 clean | `results/experiment_5488_csl_latent_exploration_replay_v498.json` |
| 5489 | SOTA CSL independent metrics | Exp5484 and Exp5488 clean | `results/experiment_5489_sota_csl_independent_metrics_v498.json` |
| 5490 | CSL/KAN fixed-point update ledger | Exp5488 ready | `results/experiment_5490_csl_kan_fixed_point_update_ledger_v498.json` |
| 5491 | Active-constraint subproblem descriptor | none | `results/experiment_5491_active_constraint_subproblem_descriptor_v498.json` |
| 5492 | Hardware receipt continuation | Exp5491 ready | `results/experiment_5492_hardware_receipts_v498.json` |
| 5493 | ARC trajectory target precheck | none | `results/experiment_5493_arc_trajectory_target_precheck_v498.json` |
| 5494 | ARC live trajectory level-up attempt | Exp5493 ready | `results/experiment_5494_arc_live_trajectory_levelup_v498.json` |
| 5495 | `.498` capstone | none | `results/experiment_5495_capstone_v498.json` |

## Rerun and Retirement Discipline

- Exp5484 and Exp5489 both cite Exp5474's adversarial `TAUTOLOGY` flag and
  include `prior_failures` with `retire_if_same_verdict: true`.
- Exp5486 cites the Exp5457 guided-decoding failure and explicitly avoids token
  steering. It uses complete-candidate evidence telemetry only.
- Exp5494 cites recent ARC no-bank attempts and the retired generic exploration
  scope. It changes the technique to trajectory/option induction from live
  runtime observations, uses target rotation, and keeps solve provenance live.
- No task references a retired upstream experiment in `requires`/`gated_on`.
- No task reuses a retired exp_id.

## Acceptance Criteria

The milestone is valid if:

- `research-roadmap-next.yaml` parses and has milestone `2026.07.498`.
- Every task has a unique deliverable under `results/*.json`.
- Every prompt includes `CONTEXT`, `EXISTING CODE TO READ FIRST`, `TASK`, and
  `CONCRETE STEPS`, and ends with the required non-push/conductor warning.
- All LLM tasks list the mandated SOTA GGUF model specs and block rather than
  producing CPU-only headline results.
- All structured gates in titles are represented in `gated_on`.
- ARC solve tasks include `solve_provenance: live_agent_self_discovery` in the
  required artifact fields.
- Relevant lint checks pass before activation.

## Expected Verification Commands

```bash
python -c "from pathlib import Path; import yaml; data=yaml.safe_load(Path('research-roadmap-next.yaml').read_text()); assert data['milestone']=='2026.07.498'; assert len(data['tasks'])==14"
python scripts/validate_prior_failures.py research-roadmap-next.yaml
python scripts/audit_roadmap_gates.py research-roadmap-next.yaml
python scripts/arc_levelup_guarantee_lint.py research-roadmap-next.yaml --min 1
python scripts/exclusion_manifest_lint.py research-roadmap-next.yaml
git diff --check -- research-references.md openspec/change-proposals/research-roadmap-vNEXT.md research-roadmap-next.yaml ops/status.md ops/changelog.md
```

## Planning Position

V498 should not chase a broad new capability surface. The clean move is to
convert `.497`'s strong but flagged CSL evidence into independently measured
self-learning, expand exact verification from hard constraints to
hard-plus-soft preference semantics, and keep the hardware and ARC lanes honest:
receipt-only hardware, live-only ARC provenance, and no duplicate solves.
