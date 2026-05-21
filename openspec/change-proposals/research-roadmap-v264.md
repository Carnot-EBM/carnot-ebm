# Research Roadmap: Milestone 2026.05.264
## "Verifier FoVer Diagnosis + Delta H2 Fix + FR-11 N=50 + Tier 0aa/0bb + arXiv Package v5"

**Prepared:** 2026-05-21 (outer-loop planning session)
**Milestone:** 2026.05.264
**ID Range:** exp2789–exp2800 (12 tasks)
**Previous Milestone:** 2026.05.263 (0/12 artifacts — all SKIPPED, WeakStrongRouter ImportError)
**Pre-test Fix:** WeakStrongRouter + RoutingDecision implemented in
  `python/carnot/pipeline/verify_repair.py`, committed `b729ba788` before .264 planning.
  This outer-loop fix resolves the structural deadlock where exp2778 (the pre-test fix task)
  was itself blocked by the pre-test cascade it was meant to fix.

---

## What Milestone .262-.263 Proved

| Finding | Source | Status |
|---------|--------|--------|
| FEP TAUTOLOGY resolved — held_out_auroc=0.9989, fep_viable=true | exp2766 | VALIDATED |
| Verifier energy=0 on GGUF outputs (5th consecutive attempt) | exp2765 | UNRESOLVED |
| Tier 0z AUROC=0.5065 — barely above random | exp2769 | UNRESOLVED |
| FR-11 production integration smoke N=3, 8s (suspicious) | exp2768 | INSUFFICIENT |
| Delta H2 regression — gemini rate-limited 3x | exp2767 | UNRESOLVED |
| Pre-test cascade (WeakStrongRouter) blocked .263 entirely | conductor | FIXED (outer-loop) |
| Ensemble v13 viable (k=19 verifiers) | exp2769 | VALIDATED |
| Phase 1 v0.1.0b1 shipped | exp2760 | VALIDATED |
| arXiv package v3 ready (HOLDS operator) | exp2762 | READY FOR OPERATOR |

---

## Three Biggest Gaps

### Gap 1 — CRITICAL: Verifier still produces zero energy (5th consecutive attempt)

**Evidence:** exp2765 loaded gemma-4-26B-A4B-it-GGUF, generated N=5 real responses (10s load,
8s/response), but energy_values=[0,0,0,0,0]. Root cause confirmed: ArithmeticExtractor regex
finds 0 constraints in instruction-tuned model outputs. Five failed attempts (exp2727, exp2740,
exp2752, exp2765, exp2779-skipped) all tried fresh inference. The fix strategy for .264 is the
FoVer corpus redirect: test on labeled violation pairs to determine if the extractor works on
structured data but not fresh IT outputs. If CASE A (extractor works on FoVer), implement
LLM-as-extractor fallback. If CASE B (energy=0 even on FoVer), diagnose deeper.

**Experiment:** exp2790 — Verifier FoVer Redirect v5

### Gap 2 — CRITICAL: Delta H2 regression (repair pipeline broken)

**Evidence:** exp2754 confirmed empirical_delta=0.000 (0/60 successes in repair loop). exp2767
(git bisect + pipeline fix) stalled 3x due to Gemini API rate limiting. This scope requires
multi-file tool choreography (git log, read pipeline code, trace repair loop, implement fix)
that exceeded Gemini's capabilities under rate pressure. Claude Opus is the correct backend.

**Experiment:** exp2791 — Delta H2 Regression Fix (Claude Opus, requires_claude=true)

### Gap 3 — HIGH: Entire .263 research queue was SKIPPED

**Evidence:** 12/12 .263 tasks SKIP due to WeakStrongRouter ImportError pre-test cascade
(confirmed in conductor log 10:05-11:18 UTC). Fix committed before .264 planning.
8 experiments need re-runs: FR-11 N=50 benchmark, NEXUS expansion, CP-Router, Tier 0aa,
Tier 0z investigation, paper v6 theory, arXiv package, capstone.

**Experiments:** exp2792–exp2800 (carries-forward from .263)

---

## Architecture Snapshot

```
Carnot Verification Pipeline (as of .262)
==========================================

[Input: question + response]
        |
        v
[AutoExtractor] ──── ArithmeticExtractor (regex, 0 constraints on IT outputs — BUG)
        |         └── ConstraintExtractor (formal, FoVer-tested)
        |         └── LLM-as-extractor (planned in exp2790 CASE A fix)
        |
        v
[Ising/KAN Energy Computation] ──── k=19 verifier ensemble (ensemble v13)
        |                      └── Tier 0a-0y (conformal, semantic, causal, geometric)
        |                      └── Tier 0aa Confidence Geometry (exp2795, new)
        |                      └── Tier 0bb DiffuTruth (exp2797, new)
        |
        v
[Routing: ODAR + WeakStrongRouter] ──── fast-path (accept) | partial (Tier 0f) | full ensemble
        |
        v
[Repair Loop] ──── verify_and_repair() — delta H2 BROKEN (exp2791 fix)
        |
        v
[NEXUS Self-Learning] ──── 34 rules (expanding to 50+ in exp2793)
        |
        v
[FR-11 Tier 4] ──── ORCA-NEXUS loop — AUROC=0.9275 (exp2755)
                └── Production integration smoke (N=3, suspicious) — full N=50 in exp2792

Phase 4 FEP: held_out_auroc=0.9989 (validated exp2766)
arXiv Package: v3 assembled, holds until Phase 4 empirically validates
```

---

## Phase Structure

### Phase A — Archive + Activate (1 task)
- **exp2789**: Archive .263 + Activate .264

### Phase B — Critical Gap Fixes (3 tasks)
- **exp2790**: Verifier FoVer Redirect v5 — FoVer corpus redirect to isolate zero-energy root cause
- **exp2791**: Delta H2 Regression Fix — Claude Opus, multi-file git bisect (gemini rate-limited 3x)
- **exp2792**: FR-11 Tier 4 Full Benchmark N=50 — real cycle-to-cycle AUROC validation

### Phase C — Research Advancement (5 tasks)
- **exp2793**: NEXUS Constraint Memory Expansion — 34 → 50+ domain rules
- **exp2794**: CP-Router Entropy-Aware Conformal Routing — arXiv:2505.19970 implementation
- **exp2795**: Tier 0aa Confidence Geometry Verifier — arXiv:2605.16824 implementation
- **exp2796**: Tier 0z Investigation — diagnose AUROC=0.5065: fix or retire
- **exp2797**: Tier 0bb DiffuTruth Hallucination Energy — arXiv:2602.11364 (NEW)

### Phase D — Publication Track (2 tasks)
- **exp2798**: Paper v6 Theory v5 — FEP claim + verifier diagnosis + delta status
- **exp2799**: arXiv Package v5 — updated with .264 results, operator checklist

### Phase E — Capstone (1 task)
- **exp2800**: Capstone v264 — Claude Opus cross-artifact synthesis

---

## Dependency Graph

```
exp2789 (archive) → exp2790 (verifier FoVer) ─────────────┐
                  → exp2791 (delta H2, Opus) ──────────────┤
                  → exp2792 (FR-11 N=50)                   ↓
                  → exp2793 (NEXUS expand)         exp2798 (paper theory)
                  → exp2794 (CP-Router)                     │
                  → exp2795 (Tier 0aa)             exp2799 (arXiv pkg)
                  → exp2796 (Tier 0z fix/retire)            │
                  → exp2797 (DiffuTruth Tier 0bb)  exp2800 (capstone)
```

exp2798 gated on exp2790.fover_energy_nonzero AND exp2791.root_cause_identified
exp2799 gated on exp2798.latex_compiles_v5

---

## New Research Integration

From the post-.263 arxiv sweep (2026-05-21):

| Paper | arXiv ID | Target Experiment |
|-------|----------|-------------------|
| DiffuTruth — Energy of Falsehood | arXiv:2602.11364 | exp2797 (Tier 0bb) |
| Self-Improvement via Verifier TTT | arXiv:2505.19475 | FR-11 Tier 4 context (exp2792) |
| Incentivizing Self-Verify | arXiv:2506.01369 | Phase 4 FEP context (exp2798) |
| Stepwise Neuro-Symbolic Proof Search | arXiv:2603.19715 | exp2798 paper citation |
| Frequency-Aware Attention Hallucination | arXiv:2602.18145 | exp2795 comparator |

Also from prior sweeps (already in research-references.md, now targeting specific experiments):
| Paper | arXiv ID | Target Experiment |
|-------|----------|-------------------|
| Confidence Geometry | arXiv:2605.16824 | exp2795 (Tier 0aa) |
| VERDI Confidence | arXiv:2605.11334 | exp2795 context |
| QueST Test-Time Self-Training | arXiv:2605.13369 | .265+ candidate |

---

## Hardware Continuity

All 3 FPGA boards at terminal state — no mandatory hardware tasks:
- KV260: TERMINAL (.260 exp2742, 3.183μs latency)
- GateMate: TERMINAL (.247)
- PolarFire: TERMINAL (.241)

---

## Agent Routing

| Experiment | Agent | Justification |
|------------|-------|---------------|
| exp2789 (archive) | gemini | Admin task, gemini-default |
| exp2790 (verifier FoVer) | gemini | Research analysis, gemini-default |
| exp2791 (delta H2) | claude/opus | Gemini failed 3x rate-limited; multi-file tool choreography; iterative hypothesis testing across commit history requires cross-context reasoning |
| exp2792 (FR-11 N=50) | gemini | Benchmark run, deterministic structure |
| exp2793 (NEXUS expand) | gemini | Rule enumeration, gemini-default |
| exp2794 (CP-Router) | gemini | Implementation, gemini-default |
| exp2795 (Tier 0aa) | gemini | Implementation, gemini-default |
| exp2796 (Tier 0z) | gemini | Diagnosis, gemini-default |
| exp2797 (DiffuTruth) | gemini | Implementation, gemini-default |
| exp2798 (paper theory) | gemini | Table update, Public Docs Discipline |
| exp2799 (arXiv pkg) | gemini | Assembly, Operator-Only rule |
| exp2800 (capstone) | claude/opus | Synthesis + planning, requires deep cross-context reasoning |

**Routing summary:** 10 gemini/gemini-3.1-pro-preview (83.3%) + 2 claude/opus (16.7%)
Within 2/12 ceiling for claude tasks. Both claude tasks meet ALL 3 positive criteria.

---

## Acceptance Criteria (12 total)

1. `exp2789.archive_completed = true`
2. `exp2790.fover_energy_nonzero = true` OR `exp2790.diagnosis != null` (either outcome valuable)
3. `exp2791.root_cause_identified = true` OR `exp2791.delta_fixed = true`
4. `exp2792.fr11_full_benchmark_validated = true` (N≥50, AUROC>0.85, pool_test_overlap=0)
5. `exp2793.nexus_rules_expanded = true` (n_rules_after≥50)
6. `exp2794.cp_router_viable = true` (savings≥20%, coverage_guarantee=true)
7. `exp2795.tier0aa_viable = true` (confidence_geometry_auroc>0.70)
8. `exp2796.tier0z_resolved = true` (fixed with AUROC>0.65 OR retired in exclusion manifest)
9. `exp2797.diffutruth_tier0bb_auroc > 0.70`
10. `exp2798.latex_compiles_v5 = true`
11. `exp2799.arxiv_package_v5_ready = true`
12. `exp2800.n_criteria_met >= 8`

---

## CLAUDE.md Compliance Checklist

- [x] Gemini-Default: 10/12 gemini (83.3%) — within ceiling
- [x] prior_failures: all rerun tasks have 4-field structure
- [x] PRECONDITIONS step 0: all compute-bound tasks
- [x] principle-annotated artifact fields: all tasks
- [x] terminal-prefix verdicts: complete:/success:/passed:/shipped: on all tasks
- [x] FR-11 mandate: exp2792 (continuous_self_learning_task=true)
- [x] Hardware-Task Continuity: N/A (all boards terminal)
- [x] KV260 SSH-Not-SD-Card: N/A
- [x] Exclusion Manifest cross-check: 0 scope matches
- [x] Operator-Only publication: exp2799 produces package, never submits
- [x] Public Documentation Discipline: exp2798 updates results tables ONLY
- [x] Gemini model: gemini-3.1-pro-preview (pinned per CLAUDE.md)
