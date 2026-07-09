# Research Roadmap vNEXT - Milestone 2026.07.500

**Milestone title:** Structured SOTA Evidence, Non-Tautological CSL, Receipt-Only Hardware, and Action-Diverse ARC

**Planner date:** 2026-07-09
**Previous milestone:** 2026.07.499
**Task range:** Exp 5510-5522
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
- `scripts/conductor_gates.py`
- `scripts/audit_roadmap_gates.py`
- `scripts/arc_levelup_guarantee_lint.py`
- `scripts/exclusion_manifest_lint.py`
- `ops/e2e-test-plan.md`

## Literature Refresh Incorporated

The planner performed a 2025-2026 refresh across arXiv, OpenReview public pages where reachable,
Extropic writing, Semantic Scholar-style citation routes for EBT `2507.02092` and ARM-EBM
`2512.15605`, Hugging Face Papers, GitHub repository discovery, and Logical Intelligence public pages.
Actionable non-duplicates were appended to `research-references.md` under:

`## V500 Planner Refresh - 2026-07-09`

New planning consequences:

- **Distributional EBMs for Uncertainty-Aware Structured LLM Reasoning** (`arXiv:2605.18871`) motivates
  exact constraint penalties plus learned or empirical uncertainty and first-class abstentions. V500 uses
  this in the SOTA hard/soft panel rather than treating missing rows as evidence.
- **PCRLLM** (`arXiv:2511.08392`) motivates proof-carrying claim structure before runtime verification.
  V500 therefore adds a structured-output positive control before another SOTA GGUF panel.
- **Thinking Before Constraining**, **XGrammar-2**, and **llguidance** motivate reason-then-structure and
  local grammar-mask validation. V500 tests schema validity as an executable precondition before spending
  flagship-model runtime.
- **Spilled Energy** (`arXiv:2602.18671`) and **Semantic Energy** (`arXiv:2508.14496`) are useful
  diagnostics, but prior Carnot spilled-energy attempts were noise-floor. V500 only allows them as
  sidecars on already-parseable SOTA rows.
- **BloGDiT** (`arXiv:2605.25129`) motivates blockwise sparse repair for CSP/COP settings. V500 applies
  this to active-constraint descriptors with exact fallback and no speedup headline unless measured.
- **1M p-bit programmable probabilistic computer** (`arXiv:2606.25313`) reinforces communication/timing
  receipts for hardware experiments. V500 keeps hardware receipt-only.
- Recent agent-memory work (`arXiv:2606.10062`, `2606.24775`, `2602.05665`) reinforces graph/stream
  memory with independent outcomes and negative-transfer checks. V500 makes this the CSL gate before any
  SOTA memory panel.

## What 2026.07.499 Proved

The `.499` milestone recovered execution discipline after the `.498` skip cascade and produced useful
bounded results. Its main contribution was not headline science; it identified the remaining blockers
precisely enough to plan the next milestone without guessing.

| Lane | Experiments | Finding |
|------|-------------|---------|
| Transition and pretest health | Exp 5496, 5497 | Active-roadmap transition and pretest cascade were repaired. This is no longer the blocker for science tasks. |
| Hard/soft exact core | Exp 5499, 5501 | Preference-MaxSAT and helper-contract fixtures exist. The exact verifier substrate is usable. |
| SOTA GGUF evidence | Exp 5500 | The live SOTA panel produced abstentions/missing candidates and only `0.333333` measured accuracy from controls. Parser/format positive control is required before more model runtime. |
| Continuous self-learning | Exp 5502, 5503, 5504 | CSL evidence graph replay showed a promising heldout delta, but the metric-independence corrigendum blocked headline credit and a gate-field mismatch blocked the SOTA panel. |
| Active constraints | Exp 5505 | Descriptor generation worked, but the run fell back to exact solving and made no speedup claim. Sparse repair needs a real candidate mechanism. |
| Hardware | Exp 5506 | CPU, CUDA, and PolarFire were reachable; KV260 and GateMate were identity-blocked; matched timing was unavailable. Hardware remains receipt-only. |
| ARC | Exp 5507, 5508 | Registry precheck found eligible targets, but the live attempt repeated a small coordinate/action pattern and banked no new level. The next attempt must change action generation. |
| Capstone | Exp 5509 | The milestone closed honestly: hard/soft core bounded, SOTA panel abstained, CSL blocked, hardware speedup false, ARC delta zero. |

## Three Biggest Gaps To PRD Vision

1. **Verifiable reasoning still lacks reliable local SOTA evidence.** The PRD asks for trustworthy
   verifier-backed reasoning. Carnot now has exact hard/soft fixtures, but the SOTA GGUF panel failed at
   the candidate-output layer. V500 must prove parseable, structured, proof-like model rows before judging
   reasoning quality.

2. **Continuous self-learning is promising but not yet non-tautological.** FR-11 requires an agent that
   learns from experience without leaking the answer into the metric. `.499` found the exact fault line:
   graph replay can work, but independent labels and conductor-resolvable fields must be clean before
   SOTA models or memory claims are credited.

3. **Embodied/live-path evidence is underpowered.** The ARC live agent and hardware lanes both collect
   receipts, but neither generated a new operational win in `.499`. V500 must improve ARC live action
   diversity and hardware methodology without overclaiming.

## Architecture For V500

```text
          research-program / PRD / architecture / completed experiments
                                   |
                                   v
                    +-----------------------------+
                    | V500 Source Delta + Handoff |
                    +--------------+--------------+
                                   |
        +--------------------------+--------------------------+
        |                                                     |
        v                                                     v
+-------------------------+                         +-------------------------+
| Structured SOTA Control |                         | CSL Independence Repair |
| - proof/claim schema    |                         | - independent outcomes  |
| - GGUF local models     |                         | - graph memory hashes   |
| - exact hard/soft check |                         | - stale evidence checks |
+------------+------------+                         +------------+------------+
             |                                                   |
             v                                                   v
+-------------------------+                         +-------------------------+
| SOTA Evidence Panel     |                         | SOTA CSL Memory Panel   |
| - parseable rows        |                         | - gated on clean metric |
| - abstention measured   |                         | - negative transfer     |
| - energy sidecar only   |                         | - no tautology credit   |
+------------+------------+                         +------------+------------+
             |                                                   |
             +----------------------+----------------------------+
                                    |
                                    v
                     +------------------------------+
                     | Sparse Constraints + Receipts |
                     | - block repair descriptors    |
                     | - exact fallback              |
                     | - CPU/CUDA/FPGA receipts      |
                     +---------------+--------------+
                                     |
                                     v
                     +------------------------------+
                     | ARC Live Level-Up             |
                     | - registry precheck           |
                     | - salience/perception router  |
                     | - action entropy guard        |
                     | - live self-discovery only    |
                     +---------------+--------------+
                                     |
                                     v
                     +------------------------------+
                     | Capstone / Spec Reconciliation|
                     +------------------------------+
```

## Phase Plan

### Phase 0 - Transition And Source Freshness

**Goal:** Safely archive `.499`, activate `.500`, and lock in the new literature deltas.

- `exp5510-v500-roadmap-transition-activation` archives the active roadmap, installs the pre-staged
  roadmap, and preserves the conductor guardrails.
- `exp5511-v500-sota-source-delta-ingestion` verifies the new papers and references are represented in
  experiments before science work begins.

### Phase 1 - Structured SOTA Evidence

**Goal:** Convert `.499` missing-candidate failures into parseable, exact-checkable SOTA rows.

- `exp5512-structured-output-positive-control` proves the local schema/grammar path before flagship model
  runtime.
- `exp5513-sota-hard-soft-structured-panel` runs the mandated local GGUF models only after positive control
  passes.
- `exp5514-energy-spill-sidecar-diagnostic` attaches logits/energy diagnostics to parsed rows only; it
  cannot be a headline detector.

### Phase 2 - Continuous Self-Learning Without Metric Leakage

**Goal:** Repair CSL evidence so memory improvements can be credited without tautology.

- `exp5515-csl-independent-outcome-gate-repair` reruns graph replay with independent labels and stable gate
  fields.
- `exp5516-sota-csl-memory-panel` uses flagship GGUF models only if Exp 5515 proves clean.
- `exp5517-csl-memory-residue-stress` tests stale/negative transfer and residual memory contamination.

### Phase 3 - Sparse Constraints, Hardware Receipts, ARC Live Path, Capstone

**Goal:** Improve operational reach without claiming unearned speedups or offline solves.

- `exp5518-block-gibbs-sparse-repair-descriptors` tests sparse repair descriptors against exact fallback.
- `exp5519-hardware-continuity-methodology-receipts` records CPU/CUDA/PolarFire/KV260/GateMate receipt
  state and timing methodology, with no speedup headline unless matched timing exists.
- `exp5520-arc-action-diversity-target-precheck` selects a non-duplicate ARC target and verifies changed
  action generation before a solve attempt.
- `exp5521-arc-live-action-diverse-levelup` performs the milestone's required live ARC level-up attempt.
- `exp5522-v500-capstone-reconciliation` reconciles artifacts, specs, status, changelog, and claims.

## Dependency Graph

```text
exp5510 transition
  |
  v
exp5511 source delta
  |
  +--> exp5512 structured output positive control
  |       |
  |       v
  |     exp5513 SOTA hard/soft structured panel
  |       |
  |       v
  |     exp5514 energy-spill sidecar
  |
  +--> exp5515 CSL independent outcome gate repair
  |       |
  |       +--> exp5516 SOTA CSL memory panel
  |       |
  |       +--> exp5517 CSL memory residue stress
  |
  +--> exp5518 block Gibbs sparse repair descriptors
  |
  +--> exp5519 hardware continuity/methodology receipts
  |
  +--> exp5520 ARC action-diversity target precheck
          |
          v
        exp5521 ARC live action-diverse level-up

All terminal lanes feed exp5522 capstone reconciliation.
```

Structured gates in `research-roadmap-next.yaml`:

- Exp 5513 requires Exp 5512 `structured_output_positive_control_ready == true`.
- Exp 5514 requires Exp 5513 `sota_structured_panel_ready == true` and `sota_rows_emitted > 0`.
- Exp 5516 requires Exp 5515 `metric_independence_clean == true` and
  `csl_gate_fields_resolvable == true`.
- Exp 5517 requires Exp 5515 `csl_experience_graph_ready == true`.
- Exp 5521 requires Exp 5520 `arc_levelup_candidate_ready == true`.

## Hardware Requirements

V500 uses the hardware policy already established by `.499`:

- **Required for SOTA GGUF experiments:** local GGUF cache for at least one mandated model from
  `scripts/experiment_template.py`:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- **Preferred for SOTA GGUF experiments:** dual RTX 3090 CUDA through the existing cached model helper.
  CPU-only small models may be used for smoke tests but not headline evidence.
- **Hardware receipt lane:** CPU, CUDA, PolarFire SSH, KV260 SSH/xmutil/UIO if reachable, and GateMate
  identity/toolchain checks if reachable. KV260 host SD-card access through `/dev/mmcblk*` remains
  excluded by the manifest.
- **Watch-only:** Extropic TSU/XTR-0/Z1 and Logical Intelligence Kona because no local executable path is
  available.

## Guardrails And Retirements

- Do not modify `research-roadmap.yaml` during planning. `.500` is staged in `research-roadmap-next.yaml`.
- Do not modify `scripts/research_conductor.py`.
- Do not push.
- Include `prior_failures` on every task whose shape overlaps a bounded, failed, blocked, or retired prior
  experiment. Each entry includes `retire_if_same_verdict: true`.
- Do not re-run retired external generated-text scorer, CPU-only SOTA offload, or KV260 host SD-card
  scopes.
- ARC solve credit must come from `solve_provenance: live_agent_self_discovery`. The required
  `offline_reproduced=true` banking receipt is only a replay of the live-discovered trajectory. Offline
  BFS, per-game hand adapters, and outer-loop reverse engineering are not deliverables.
- Any LLM-bearing experiment must include one or more mandated local SOTA GGUF models in `MODEL_SPECS`.
  Legacy small models are smoke-test-only.
- Hardware speedup remains false unless authenticated matched timing exists.

## Expected Outcomes

V500 is successful if it produces one of the following clean branches:

1. **Structured SOTA branch:** parseable flagship GGUF rows with exact hard/soft labels, measured
   abstention, and optional energy sidecar diagnostics.
2. **CSL branch:** independently measured graph-memory improvement with conductor-visible gate fields and
   a bounded SOTA memory panel.
3. **ARC branch:** at least one live-agent ARC level improvement or a high-quality null showing a changed,
   non-repeated action generator and registry-safe target selection.
4. **Hardware branch:** trustworthy continuity receipts across local devices with methodology sufficient
   to support a future speedup run.

The milestone should not claim SOTA reasoning, CSL, hardware acceleration, or ARC solve progress unless
the corresponding artifacts support it directly.
