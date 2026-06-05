# Research Roadmap v356 Change Proposal — Re-issue the Verifier-MOAT Durability Adjudication (DT-P2), after the `.355 poison-test wipeout

**Milestone:** 2026.06.356
**Planned:** 2026-06-05 (Claude Opus 4.8, outer-loop planning agent)
**Milestone doc:** this file (alias: `research-roadmap-vNEXT.md`)

## Why this milestone exists: `.355 was a total wipeout

`.355 was designed to answer the single load-bearing forward question (Deep-Think P2):
**does Carnot's verifier independently catch the step-errors a strong reasoner's OWN
self-verification misses, or is it about to be subsumed?** It produced **zero usable
artifacts**. Two compounding blockers, both diagnosed and handled by the outer-loop this
session:

1. **A corrupted `research-complete.yaml` poisoned the conductor's pre-test gate.** During
   `.354's archive, the exp3833 verdict was appended unquoted:
   `result: complete: ldt_gap_LATTICE_VIABLE_...`. The bare `: ` makes YAML read the value
   as a nested mapping → `yaml.scanner.ScannerError: mapping values are not allowed here`.
   That broke `test_docs.py::test_public_docs_cover_latest_pbt_and_fpga_reporting`, which
   parses the YAML. The conductor's smart-subset pre-test gate runs `test_docs.py` on every
   task, so **one failing test (`1 failed, 80–86 passed`) SKIP-cascaded every experiment**
   from `.350 onward (intermittently) and the whole of `.355. **FIXED** this session by
   quoting the value (two lines, 35485 + 35569): `81 passed, 0 failed`; the full YAML now
   parses. This is the `incident_agent_shipped_test_cascade` failure mode, root-caused to a
   data-file corruption rather than a shipped test.
2. **The gemini-CLI crashed/stalled on the archive task** (`chunk-NBZI34` bundle crash; 429
   "Too Many Requests"), and the conductor ran it via gemini despite the YAML requesting
   claude. This is the `incident_333_gemini_quota_crash_wipeout` pattern. **ROUTED AROUND**
   by setting every `.356 task `agent_type: codex` + `requires_codex: true` — the proven
   anti-wipeout pattern from `.337/`.340 (which succeeded all-codex while holding
   `paper_ready=true`). Codex is the empirically-reliable backend in this conductor env.

The scientific spine is **unchanged from `.355** because the question was never answered. The
plan is re-issued verbatim in intent with fresh experiment IDs (exp3857–exp3868), all-codex
routing, and the gate now unblocked.

## What the previous milestones proved (the converged state)

- **The sole defensible headline is the FoVer 0.9131 math step-level error verifier** (G1–G4
  all pass, `paper_ready=TRUE`, G2 independently reproduced on a clean CI runner 2026-05-31).
  Frozen — this milestone adds a durability LENS, it does NOT re-measure the headline AUROC.
- **Both energy-foundation routes are BOUNDED** (P0.1 energy-as-selector; Thesis-A EBT
  energy-as-generator). Findings, not doomed-rerun ids.
- **The verifier is math-domain-bound** — earned-negative on facts (RAGTruth + NLI), weak on
  code. A *reasoning* verifier, not a general fact-checker.
- **The loop will NOT self-seed the next paradigm** (Verification Trap, Deep-Think P3). The
  operator seeds or freezes; the loop scaffolds and adjudicates.

## The ONE load-bearing question: does the moat survive at scale, with MEASURED independence?

The product is worth something only if Carnot *independently* catches the step-errors a
strong reasoner's self-verification MISSES. The only prior signal is exp3827's 100-item
subset (`residual_catch=0.9000`, `overlap=0.5000` — round numbers the Adversarial-Confirmation
Discipline forbids citing until replicated at scale). exp3844 tried it at scale and BLOCKED on
a missing balanced corpus (FoVer v4 is 98% correct-heavy → ~22 residual items).

### What the 2026 literature says about how to measure this

- **SUPPORTS — arXiv:2602.03485 (Self-Verification Dilemma):** strong reasoners'
  self-verification is "the vast majority confirmatory rather than corrective" → a residual
  error set demonstrably persists.
- **SUPPORTS — arXiv:2506.18203 (Weak-Verifier / Weaver, NEW this sweep):** a pool of cheap
  decorrelated weak verifiers closes most of the oracle gap — value from COMBINING, not member
  strength. The warrant for Carnot's cheap NON-LLM ensemble being complementary.
- **THE REFUTATION RISK — arXiv:2604.07650 (Behavioral Entanglement):** verifiers sharing an
  LLM lineage with the reasoner produce *correlated failure masquerading as independence*. A
  naive scissor can report a FAKE moat. **Independence must be MEASURED, not assumed.** Carnot's
  core verifiers are NON-LLM (Z3/AST/SAT/energy) → the prior is decorrelation, but PROVE it.
- **THE COMPLEMENTARITY BAR — arXiv:2504.16828 (ThinkPRM):** a single strong generative PRM is
  hard to out-AUROC. The moat must rest on *complementary catch*, not head-to-head AUROC.
- **THE CORPUS — arXiv:2501.03124 (PRMBench):** 6,216 problems / 83,456 step labels across 9
  fine-grained error axes — balanced, with axes enabling a per-error-axis independence
  decomposition.

## The three biggest gaps between current state and the PRD vision

1. **The moat is asserted from one 100-item run with the at-scale confirmation blocked.** Build
   the balanced corpus (PRMBench primary, FoVer v3 fallback), run the scissor properly (N≥1000,
   bootstrap CI95, two positive controls), AND audit verifier-vs-reasoner independence per
   2604.07650 so the moat is *measured*. (Phase 1 — the spine.)
2. **The verifier is earned-negative on FACTS; re-testing without a NEW architecture is
   forbidden** (memory `project_verifier_domain_bound`). Graph-RAG / KG-alignment grounding
   (2606.00610 / 2512.01659 / 2512.09148) is a mechanistically DIFFERENT verifier — the one
   route with a plausible mechanism. Prototype-first. (Phase 2.)
3. **Continuous self-learning (research-program.md MANDATE) has never operated on the moat
   itself.** 2604.07650's independence-reweighting (+4.5% over majority vote) is the exact
   Tier-1 online mechanism: upweight verifiers that catch the residual *others* miss. The
   mandated self-learning task and the moat thesis become the same task. (Phase 3.)

## Architecture (what `.356 touches)

```
  PRMBench (2501.03124, 9 axes) / FoVer v3 fallback ──► data/step_error_balanced_v2.json
                                                              │ (N≥1000, ≥100 gold-incorrect)
                       ┌──────────────────────────────────────┴───────────────┐
                       ▼                                                        │
   exp3859 MOAT SCISSOR @ SCALE (live Qwen3.6-35B self-verify + k=15 ensemble)  │
     residual_catch_rate + CI95, error_overlap, 2 positive controls            │
     persists per_item_error_masks ──────────────┬───────────────┐            │
                       ▼                          ▼               ▼            ▼
   exp3860 INDEPENDENCE AUDIT       exp3861 THINKPRM           (FR-11 reads per-verifier
     (2604.07650: phi/Matthews        COMPLEMENTARITY            independence — exp3864)
      reasoner-vs-ensemble,            (union-lift over a
      per-verifier, per-9-axis)        strong generative PRM)

  Phase 2 (facts, NEW architecture):  exp3862 graph-grounding prototype ─► exp3863 facts complementarity
  Phase 3 (self-learning + LDT):      exp3864 FR-11 v23 independence-reweighting; exp3865 LDT margin sharpening
  Phase 4 (hardware continuity):      exp3866 GateMate flash; exp3867 PolarFire SSH dispatch
  Phase 5:                            exp3868 capstone — moat verdict CONDITIONED on the independence audit
```

## Phases & tasks (12 tasks, exp3857–exp3868)

- **Phase 0 — Activation.** exp3857: archive `.355 (the wipeout, root cause + outer-loop fix
  recorded honestly), activate `.356; confirm `paper_ready=true` + frozen 0.9131.
- **Phase 1 — Verifier-moat durability (the spine).**
  - exp3858: build `data/step_error_balanced_v2.json` (PRMBench primary, FoVer v3 fallback;
    N≥1000, ≥100 gold-incorrect, 9-axis labels preserved). Emits `n_incorrect_steps` BARE (gate field).
  - exp3859: moat scissor at scale (live Qwen3.6-35B self-verify vs k=15 ensemble; residual_catch
    + bootstrap CI95; two positive controls; persists per-item error masks). Emits `n_residual_errors` BARE.
  - exp3860: verifier-vs-reasoner independence audit (2604.07650; phi/Matthews; per-verifier;
    per-9-axis). The measurement that makes the moat claim defensible vs the fake-independence trap.
  - exp3861: ThinkPRM complementarity (union-lift of the cheap NON-LLM ensemble over a strong
    generative PRM). Reuses exp3859's masks — CPU only.
- **Phase 2 — Facts via a NEW architecture (graph grounding).**
  - exp3862: graph-grounding fact-verifier PROTOTYPE on RAGTruth (KG-alignment; NEW mechanism).
    Emits `facts_catch_delta` BARE (gate field).
  - exp3863: facts-domain complementarity (graph verifier vs math ensemble; gated on delta>0).
- **Phase 3 — Self-learning (MANDATE) + LDT sharpening.**
  - exp3864: FR-11 v23 ONLINE independence-reweighting (2604.07650 method) on the live ensemble;
    invariant: must not drop the frozen 0.9131 below CI. Emits `auroc_in_frozen_ci` BARE bool.
  - exp3865: LDT-lattice margin sharpening (exp3833's 0.010 margin vs a SCORE-MATCHED control + CI).
- **Phase 4 — Hardware continuity (one task per non-terminal board).**
  - exp3866: GateMate A1-EVB-2M n≥16 Ising tile flash (himbaechel flow; prior_failures exp2899).
  - exp3867: PolarFire SoC SSH dispatch + hash-verify (prior_failures exp1680).
- **Phase 5 — Capstone.** exp3868: aggregate the moat verdict CONDITIONED on the independence
  audit; `paper_ready` stays TRUE; frozen 0.9131 unchanged; operator forward recommendation.

## Dependency graph

```
exp3857 (archive/activate)
exp3858 (corpus) ──► exp3859 (scissor) ──► exp3860 (independence audit)
                                      └──► exp3861 (ThinkPRM complementarity)
                                      └──► exp3864 (FR-11 reads per-verifier independence)
exp3862 (graph proto) ──► exp3863 (facts complementarity)
exp3865 (LDT margin)   exp3866 (GateMate)   exp3867 (PolarFire)
all ──► exp3868 (capstone)
```

`gated_on` (bare-scalar gates): exp3859⟵exp3858.n_incorrect_steps≥100;
exp3860⟵exp3859.n_residual_errors≥30; exp3861⟵exp3859.n_residual_errors≥30;
exp3863⟵exp3862.facts_catch_delta>0.

## Hardware requirements

- exp3859: one free RTX 3090 (≥10GB) for live Qwen3.6-35B-A3B-GGUF self-verification over N≥1000.
- exp3866: yosys 0.64+ (`synth_gatemate`), `nextpnr-himbaechel`, `gmpack`, GateMate board on DirtyJTAG.
- exp3867: PolarFire SoC reachable via `ssh polarfire`.
- All other tasks are CPU (verifier-scoring / aggregation substrates).

## Invariants (must hold at capstone)

- `paper_ready` stays TRUE (G1–G4). The frozen FoVer **0.9131** is NEVER silently substituted —
  this milestone adds a durability lens, not a new headline.
- KV260 is GRADUATED (terminal); GateMate + PolarFire are the two non-terminal boards covered.
- All tasks codex+`requires_codex` (anti-wipeout). No flagged-adversarial artifact is aggregated.

## Routing & discipline notes

- **All 12 tasks `agent_type: codex`, `model: gpt-5.5`, `requires_codex: true`** — the anti-wipeout
  routing. The conductor's claude path currently falls back to a crashing gemini-CLI; codex is the
  reliable backend (`.337/`.340 precedent).
- Every scope-matched legit continuation carries an `operator_override` (2026-06-05 standing
  directive: `.355 poison-wipeout re-issue / versioned lineage with stated forward difference).
  Genuinely-retired predecessors (exp2899 GateMate, exp1680 PolarFire) carry a full 4-subfield
  `prior_failures` block; the scissor carries `prior_failures` for the blocked exp3844.
- Gated fields (`n_incorrect_steps`, `n_residual_errors`, `facts_catch_delta`, `auroc_in_frozen_ci`)
  are emitted as BARE scalars (per `feedback_gated_fields_must_be_bare`).
- Inference-substrate hygiene: NO GGUF/CUDA markers on aggregation/scoring tasks; live floor only
  on exp3859 (and optionally exp3862 if it invokes a model).
