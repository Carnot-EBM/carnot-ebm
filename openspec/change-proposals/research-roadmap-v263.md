# Carnot Research Milestone v263
## Pre-Test Fix v2 + Verifier FoVer Redirect + Delta H2 Repair + Confidence Geometry + arXiv Package v5

**Milestone:** 2026.05.263
**Date:** 2026-05-21
**Experiment IDs:** exp2777–exp2788 (12 tasks)
**Status:** PROPOSED

---

## What Milestone .262 Proved

Milestone .262 (exp2764–exp2776) produced 5 usable artifacts (partial execution). Root causes of
incomplete execution: (a) Gemini "Too Many Requests" rate-limiting stalled exp2767 x3, (b) a pre-test
cascade from `tests/python/test_weak_strong_router.py` ImportError caused exp2770–exp2776 to all SKIP.

| Finding | Artifact | Status |
|---|---|---|
| Phase 4 FEP TAUTOLOGY resolved — held_out_auroc=0.9989 | exp2766 | CONFIRMED |
| Verifier energy=0.0 on all 5 live responses (5th consecutive) | exp2765 | CRITICAL UNRESOLVED |
| Delta H2 fix FAIL x3 — gemini "Too Many Requests" | exp2767 | CRITICAL UNRESOLVED |
| FR-11 production smoke test N=3 only | exp2768 | PARTIAL — needs full N=50 |
| Tier 0z auroc=0.5065 — barely above random | exp2769 | POOR — needs investigation |
| Repair forensic, NEXUS expansion, CP-Router | exp2770–exp2772 | ALL SKIP (pre-test cascade) |
| Paper v6 theory v5, HF model card, arXiv pkg v5 | exp2773–exp2775 | ALL SKIP (pre-test cascade) |
| Capstone v262 | exp2776 | SKIP (pre-test cascade) |

---

## Three Biggest Gaps for .263

### Gap 1 — CRITICAL: Pre-test cascade (test_weak_strong_router.py ImportError)

**Root cause (diagnosed this session):**
`tests/python/test_weak_strong_router.py` imports `WeakStrongRouter` and `RoutingDecision` from
`carnot.pipeline.verify_repair`, but neither class is exported from that module. `WeakStrongRouter`
does not exist anywhere in the codebase. This test was created by exp2758 (.261) as part of the
weak-strong policy fix, but the implementation was never added to the module.

`RoutingDecision` exists in `carnot.pipeline.odar_router`, not `carnot.pipeline.verify_repair`.

**Impact:** Every task after exp2769 (.262) was SKIP because the conductor's self-heal couldn't fix
this ImportError. This is the same pre-test cascade pattern that caused the 51-milestone stall
(exp2713 fixed it in .258) — but a new test created the same failure pattern.

**Fix for .263 (exp2778):**
Implement `WeakStrongRouter` dataclass + `RoutingDecision` dataclass in `carnot/pipeline/verify_repair.py`
(or a new `carnot/pipeline/weak_strong_router.py` module with re-export from verify_repair.py).
The class needs `t_low`, `t_high` constructor params and a `route(prompt, response, weak_score=None)`
method returning a `RoutingDecision` with `.path` and `.verifier` fields.

### Gap 2 — CRITICAL: Verifier produces zero energy on GGUF outputs (5th consecutive attempt)

**Root cause (diagnosed this session):**
The `VerifyRepairPipeline.verify()` always returns `energy=0.0` for live GGUF responses because:
1. `AutoExtractor` selects extractors based on content patterns.
2. `ArithmeticExtractor` looks for `"X op Y = Z"` regex patterns.
3. Instruction-tuned GGUF model outputs don't write equations in that format.
4. Result: 0 constraints extracted → energy = 0.0 by definition.

**Why previous experiments failed to catch this:**
All previous experiments (exp2727, exp2740, exp2752, exp2765) tried to extract constraints from
fresh GGUF model generations. The GGUF model generates natural language answers, not arithmetic
equations in the regex-matchable format.

**Fix for .263 (exp2779):**
Test the verifier on the FoVer corpus (known violation pairs, not fresh generations):
1. Load 20 FoVer violation pairs from `data/fover_corpus.jsonl`.
2. Feed each pair's `violated_response` to `pipeline.verify(question, response)`.
3. If energy > 0 for FoVer violations: ArithmeticExtractor DOES work on FoVer — the issue is only
   with fresh GGUF generations. Solution: add an LLM-as-extractor fallback for non-FoVer inputs.
4. If energy = 0 even for FoVer violations: the problem is deeper in the energy computation path.
   Solution: add debug logging to trace constraint extraction → energy computation → output.

This experiment produces a definitive diagnosis AND a fix plan. Gate: `fover_energy_nonzero: bool`.

### Gap 3 — HIGH: Delta H2 regression still unrepaired (exp2767 failed 3x, Gemini rate-limited)

**Root cause of exp2767 failure:**
Gemini CLI returned "Too Many Requests" 3 consecutive times. Each attempt stalled for 600s before
timeout. The git bisect scope requires many file reads across the repair pipeline history, which
exhausted the Gemini API quota window.

**Fix for .263 (exp2780):**
Route to Claude Opus (`requires_claude: true, model: opus`). The positive criterion is met:
- Gemini DEMONSTRABLY FAILED on this specific scope (x3 rate-limited in .262)
- The task requires multi-file tool choreography (git bisect + file inspection + pipeline fix)
- Multi-step reasoning: git bisect requires iterative hypothesis testing across commits

Gate: `empirical_delta > 0.10` on N=100 FoVer pairs after fix.

---

## Architecture Snapshot (entering .263)

```
Ensemble (k=19):
  Tier 0a–0h: logprob, AUROC, semantic, spilled energy, OTV(retired), TF-IDF, EORM, semantic-calib
  Tier 0r–0s: Curry-Howard, HalluGuard
  Tier 0u: Logical consistency
  Tier 0v: Set-Consistency Networks (arXiv:2503.10695)
  Tier 0w: Paraphrastic Consistency (arXiv:2602.11361) [corr=0.26 — best decorrelation]
  Tier 0x: Conformal Selective Acting (arXiv:2605.20270) [84% savings, anytime-valid]
  Tier 0y: Differentiable Conformal Calibration (arXiv:2604.20098) [ECE~5e-8]
  Tier 0z: Temporal/Causal (exp2769: auroc=0.5065 — POOR, under investigation)
  Candidate Tier 0aa: Confidence Geometry (arXiv:2605.16824, exp2784)

Routing:
  ODAR two-tier (K=3, 65% savings)
  Conformal routing (exp2757: 84% savings, anytime-valid, 1.2% FNR)
  Weak-strong policy (exp2758: 41% savings, 0% FNR) — BUT test_weak_strong_router.py broken

FR-11 ORCA-NEXUS:
  Tier 4 validated (exp2755: cycle3 AUROC=0.9275, 34 rules)
  Production integration (exp2768: smoke test N=3 — needs full benchmark)

Phase 4 FEP:
  Strategy2 validated (exp2766: held_out_auroc=0.9989, fep_viable=true) ✓

Verifier discriminativeness:
  All verifier results are CPU-only on FoVer corpus.
  No in-vivo validation on live GGUF outputs yet (energy=0 on all attempts).

Repair pipeline:
  empirical_delta=0.000 (H2 regression, repair fails on every attempt)
  Paper-v6 4/δ bound BLOCKED until delta > 0.10

Publication status:
  Phase 1 v0.1.0b1 shipped ✓ (exp2760)
  Paper-v6 28pp compiles ✓ (exp2761)
  arXiv package v3 ready, HOLDS operator ✓ (exp2762)
```

---

## Phase Structure

### Phase A — Admin (1 task)
- **exp2777**: Archive .262 + Activate .263

### Phase B — Critical Infrastructure (2 tasks)
- **exp2778**: Pre-test fix v2 — Implement WeakStrongRouter (Claude/Sonnet)
- **exp2779**: Verifier FoVer redirect v5 — Test on known violations, diagnose + fix zero-energy

### Phase C — Core Research Repair (2 tasks)
- **exp2780**: Delta H2 regression fix — Git bisect + repair pipeline fix (Claude Opus)
- **exp2781**: FR-11 full benchmark N=50 — Real production validation (gated on exp2778)

### Phase D — Research Advancement (4 tasks)
- **exp2782**: NEXUS constraint memory expansion 34→50+ rules (gated on exp2778)
- **exp2783**: CP-Router entropy routing (arXiv:2505.19970) — Tier 0 routing integration
- **exp2784**: Confidence Geometry Tier 0aa (arXiv:2605.16824) — New verifier candidate
- **exp2785**: Tier 0z investigation — why auroc=0.5? Fix or retire

### Phase E — Publication (2 tasks, gated)
- **exp2786**: Paper v6 theory v5 (gated on exp2779 fover_energy_nonzero + exp2780 delta_fixed)
- **exp2787**: arXiv package v5 + operator checklist (OPERATOR-ONLY submit; gated on exp2786)

### Phase F — Capstone (1 task)
- **exp2788**: Capstone v263 — Cross-artifact synthesis (Claude Opus)

---

## Dependency Graph

```
exp2777 (archive/activate)
   |
   +-- exp2778 (pre-test fix v2) ----+
   |                                 |
   +-- exp2779 (verifier FoVer)      +-- exp2781 (FR-11 N=50)
   |                                 |
   +-- exp2780 (delta H2 fix)        +-- exp2782 (NEXUS expansion)
   |                                 |
   +-- exp2783 (CP-Router)           +-- exp2786 (paper v6 theory v5)
   |                                      |
   +-- exp2784 (Confidence Geometry)      +-- exp2787 (arXiv pkg v5)
   |
   +-- exp2785 (Tier 0z investigation)
   |
exp2788 (capstone — reads all artifacts)
```

Gate conditions:
- exp2781: gated on exp2778 (pre_test_fixed=true)
- exp2782: gated on exp2778 (pre_test_fixed=true)
- exp2786: gated on exp2779 (fover_energy_nonzero OR diagnosis_complete) AND exp2780 (delta_fixed OR root_cause_identified)
- exp2787: gated on exp2786 (latex_compiles_v5=true)

---

## Acceptance Criteria (12 checks)

| # | Criterion | Exp | Gate |
|---|-----------|-----|------|
| 1 | pre_test_fixed=true (test collection clean) | exp2778 | WeakStrongRouter exists + test passes |
| 2 | fover_energy_nonzero=true OR energy_zero_root_cause_identified=true | exp2779 | definitive diagnosis |
| 3 | delta_fixed=true OR regression_commit_identified=true | exp2780 | empirical_delta > 0.10 or commit found |
| 4 | fr11_full_benchmark_validated=true (N≥50) | exp2781 | AUROC > 0.85, pool_test_overlap=0 |
| 5 | nexus_rules_expanded=true (≥50 rules) | exp2782 | n_rules ≥ 50 |
| 6 | cp_router_viable=true | exp2783 | savings ≥ 20%, coverage_guarantee=true |
| 7 | confidence_geometry_auroc > 0.70 | exp2784 | tier0aa candidate viable |
| 8 | tier0z_verdict != random (auroc ≠ 0.5±0.05) OR tier0z_retired=true | exp2785 | investigation complete |
| 9 | latex_compiles_v5=true | exp2786 | paper compiles |
| 10 | arxiv_package_v5_ready=true | exp2787 | operator checklist produced |
| 11 | fep_claim_included=true (held_out_auroc=0.9989 cited) | exp2786 | FEP in paper |
| 12 | capstone complete | exp2788 | all artifacts synthesized |

---

## Hardware Continuity

All 3 FPGA boards have reached terminal state. No mandatory hardware tasks this milestone:
- KV260: TERMINAL (.260 exp2742, kv260_terminal=true, 3.183μs latency)
- GateMate: TERMINAL (.247)
- PolarFire: TERMINAL (.241)

---

## Agent Routing

- 10 tasks: `agent_type: gemini`, `model: gemini-3.1-pro-preview`
- 2 tasks: `agent_type: claude`, `model: opus` (`requires_claude: true`)
  - exp2780: positive criterion met — Gemini demonstrably failed x3 (rate-limited), multi-file git bisect
  - exp2788: capstone — cross-artifact synthesis requires open-ended judgment under ambiguity

Routing split: 10/12 gemini (83.3%), 2/12 claude (16.7%). Within the 2/13 ceiling.

---

## New Research Papers (post-.262 sweep)

| arXiv ID | Title | Target Exp |
|---|---|---|
| arXiv:2605.16824 | Confidence Geometry for LLM Reasoning | exp2784 (Tier 0aa) |
| arXiv:2605.11334 | VERDI Confidence for Verification Judges | exp2784 context |
| arXiv:2605.13369 | QueST Query-Conditioned Test-Time Training | .264+ candidate |
| arXiv:2605.18871 | Distributional EBMs for LLM Reasoning | Background (already referenced) |

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` for scope matches:
- `grpo_vprm_v15_scope_closed`: no GRPO/VPRM tasks — 0 matches
- `wopr_puzzle_cartridge_research_scope_closed`: no puzzle tasks — 0 matches
- `hardnet_dsp_repair_stack_scope_closed`: no HardNet/DSP tasks — 0 matches
- `thrml_scaling_sweep_lineage_retired_after_vendoring`: no THRML parity tasks — 0 matches
- `kv260_host_sd_card_precondition_retired`: no KV260 tasks — 0 matches (boards terminal)
- `otv_kvcache_probe_retired`: no OTV tasks — 0 matches
- `diversity_maximizing_verifier_selection_retired`: Tier 0z investigation is about diagnosing low auroc, not greedy selection — 0 matches

**Total scope matches: 0**

---

## CLAUDE.md Compliance Checklist

- [x] Gemini-Default: 10/12 gemini (83.3%), 2 claude tasks meet ALL positive criteria
- [x] prior_failures: all rerun tasks (exp2779, exp2780) have mandatory 4-field structure
- [x] PRECONDITIONS step 0 on all compute-bound tasks (exp2779, exp2780, exp2781, exp2782, exp2783, exp2784, exp2785)
- [x] Principle-annotated artifact fields on all tasks
- [x] Terminal-prefix verdicts on all tasks (`complete:` / `success:` / `passed:` / `shipped:`)
- [x] FR-11 mandate: exp2781 (FR-11 full benchmark, continuous_self_learning_task=true)
- [x] Hardware continuity: no mandatory tasks (all 3 boards terminal)
- [x] KV260 SSH-Not-SD-Card: N/A (no KV260 tasks)
- [x] Exclusion Manifest: 0 scope matches
- [x] Operator-Only publication: exp2787 produces package + checklist; NEVER submits
- [x] Calendar-Month Prefix Rollover: 2026.05.263 (still May 2026)
- [x] Failed-Experiment Rerun Discipline: prior_failures on all rerun tasks
