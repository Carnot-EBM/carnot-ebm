# Research Roadmap v237: AUROC Final Breach + Paper Capstone + Hardware Continuity

**Milestone:** 2026.05.237
**Status:** PROPOSED
**Date:** 2026-05-19
**Previous milestone:** 2026.05.236 (AUROC Ceiling Breach Attempt) — 10/13 tasks completed

---

## What .236 Proved

Milestone .236 achieved two major milestones and surfaced three critical gaps:

**Major wins:**
- **Phase 1 ship gate: FINALLY MET** (exp2441) — PyPI published, HF mirror up, external
  reproducer exists, MCP docs written, CLI docs written. All 5 criteria satisfied.
- **Conformal Ensemble AUROC: 0.9167** (exp2438, 7 verifiers fused) — gap to HIVE peer
  is now only **0.0069** (down from 0.034). ensemble_auroc_improved=true.

**Critical gaps:**
1. **Conformal JSON malformation** — exp2438 artifact had "Extra data: line 75 column 2
   (char 1999)" causing JSON parse failure. The gate `ensemble_auroc_improved==true`
   could not be read, so exp2445 (capstone) was blocked. The AUROC was achieved but
   the evidence chain was broken by a JSON bug.
2. **KV260 synthesis_errors=1** — exp2440 artifact is MISSING (task never ran).
   Prior exp2427 confirmed synthesis_errors=1 with 241s real run time, 18 RTL files,
   Yosys 0.64+149. The error is in RTL content, not infrastructure.
3. **NCO AUROC=0.500 tautology** — exp2444 was adversarially flagged (exact 0.5 AUROC).
   Likely cause: dataset with equal positive/negative counts and no information leakage,
   leading to chance-level performance. Needs corrigendum.

**Verifier results from .236:**
- DiffuTruth Tier 0k (exp2435): AUROC=0.588 (weak, but orthogonal: Pearson r=0.398)
- PCIB Tier 0l (exp2436): AUROC=0.802 (good standalone signal)
- LogCons Z3-True v3 (exp2437): AUROC=0.607 (WORSE than fallback 0.8896 — Z3 encoding
  via synthetic hierarchy degrades performance; hierarchy_violation_rate=0.0 means Z3
  found no violations to discriminate on)
- Conformal Ensemble v1 (exp2438): AUROC=0.9167 (7 verifiers fused) — JSON malformed
- FR-11 (exp2439): completed (non-standard verdict format, but learnability audit done)
- FST MCMC fix (exp2442): energy computation fixed
- Kinetic Langevin FST integration (exp2443): complete

---

## Architecture Diagram

```
Phase 0: Admin
┌─────────────────────┐
│ exp2447             │
│ Archive .236        │
│ activate .237       │
│ (codex, ungated)    │
└─────────────────────┘

Phase 1: AUROC Final Breach (fix JSON + add PCIB + HalluField + LaaB)
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ exp2448          │  │ exp2449          │  │ exp2450          │
│ Conformal        │  │ HalluField       │  │ LaaB ACL 2026    │
│ Ensemble v2      │  │ Tier 0m          │  │ Meta-Judgment    │
│ (fix JSON bug;   │  │ (arXiv:          │  │ Bridge v2        │
│  add PCIB;       │  │  2509.10753)     │  │ (arXiv:          │
│  7→8 verifiers)  │  │  (codex)         │  │  2605.03971)     │
│ (codex)          │  │                  │  │  (codex)         │
└────────┬─────────┘  └──────────────────┘  └──────────────────┘
         │ ensemble_auroc_improved==true
         │
Phase 2: FR-11 + Free-Energy Routing           │
┌────────────────────┐  ┌────────────────────┐ │
│ exp2451 (FR-11 v5) │  │ exp2455            │ │
│ Soundness/Complete │  │ ODAR Free-Energy   │ │
│ ness Tracking      │  │ Routing for        │ │
│ (arXiv:2603.03538) │  │ verify-repair      │ │
│ (codex, CSL)       │  │ (arXiv:2602.23681) │ │
└────────────────────┘  │ (codex)            │ │
                        └────────────────────┘ │
                                               │
Phase 3: Hardware (MANDATORY per CLAUDE.md)    │
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ exp2452          │  │ exp2453          │  │ exp2454          │
│ KV260 RTL        │  │ GateMate n=16    │  │ PolarFire SoC    │
│ Synthesis Fix v5 │  │ Ising Synthesis  │  │ Smoke v3         │
│ (debug 18 RTL    │  │ + Flash v2       │  │ (precondition-   │
│  files for 1     │  │ (OSS CAD Suite   │  │  gated, NOT      │
│  synthesis error)│  │  confirmed)      │  │  fabricated)     │
│ (claude,opus)    │  │ (codex)          │  │ (codex)          │
└──────────────────┘  └──────────────────┘  └──────────────────┘

Phase 4: Corrigendum + Synthesis
┌────────────────────┐  ┌────────────────────────────────────────────────┐
│ exp2456            │  │ exp2457 (Paper-v6 Capstone v237)               │
│ NCO Corrigendum    │  │ claude, model:opus, gated: exp2448 above       │
│ v2 (fix AUROC=0.5  │  │                                                │
│ tautology from     │  │ exp2458 (Milestone .237 Retro, codex, ungated) │
│ exp2444) (codex)   │  │                                                │
└────────────────────┘  └────────────────────────────────────────────────┘
```

---

## Three Biggest Gaps Entering .237

### Gap 1: AUROC Ceiling (gap=0.0069 to HIVE peer 0.9236)

**Evidence:** exp2438 achieved AUROC=0.9167 (ensemble_auroc_improved=true) but JSON was
malformed (Extra data: line 75 column 2 char 1999), blocking the downstream capstone gate.
The AUROC was real — conformal_ensemble_auroc=0.9167 is confirmed by regex extract from
the partial JSON. Gap is 0.9236 - 0.9167 = 0.0069.

**Fix plan for .237:**
- exp2448 (Conformal Ensemble v2): Rebuild with JSON-safe output, add PCIB scores
  (exp2436 produced tier0l_scores.json with 36 examples), fix calibration split.
  Target: 8 verifiers fused, AUROC ≥ 0.9236. Even if same 0.9167, valid JSON = capstone unblocked.
- exp2449 (HalluField Tier 0m): New signal axis (token-path energy distribution under
  temperature variation). If orthogonal to existing 7 verifiers, adding it may push ensemble AUROC.
- exp2450 (LaaB Meta-Judgment v2): Full ACL 2026 label constraint modeling bridge —
  update the existing laab_verifier.py from the initial implementation to the meta-judgment
  version with same/opposite label coherence constraints.

### Gap 2: KV260 RTL Synthesis (synthesis_errors=1)

**Evidence:** exp2427 (.235) ran 241s across 18 RTL files and got synthesis_errors=1.
exp2440 (.236) was MISSING (never produced an artifact). The infrastructure works
(Yosys 0.64+149 installed, 18 RTL files present); the error is in RTL content.

**Fix plan:** exp2452 (KV260 RTL Fix v5) reads the 18 RTL files + yosys log to identify
which file causes the synthesis error. Requires multi-file debugging (18 .v files).
Routed to claude+opus per CLAUDE.md hardware-integration pre-routing guidance and because
codex demonstrably failed twice (exp2413, exp2427) on the synthesis fix.

### Gap 3: Paper-v6 Capstone Blocked

**Evidence:** exp2445 was gate-blocked because exp2438 JSON was unreadable. With
phase1_ship_gate_met=true (exp2441) and AUROC=0.9167 (near-breach), the capstone
should finally run in .237 once exp2448 produces valid JSON with ensemble_auroc_improved=true.

**Fix plan:** exp2457 (Capstone v237) gated on exp2448.ensemble_auroc_improved==true.
The capstone synthesizes: phase1_ship_gate_met + best AUROC + sampler suite + hardware
progress + paper-v6 results table update.

---

## Hardware Track Status (MANDATORY per CLAUDE.md)

| Board | USB VID:PID | Last result | Next step | Terminal state |
|---|---|---|---|---|
| KV260 | internal programmer | exp2427: synthesis_errors=1 | Debug RTL content bug | kv260_synthesis_succeeded=true |
| GateMate A1-EVB-2M | 1209:c0ca (DirtyJTAG onboard) | exp2105: synthesis_completed=false (toolchain was missing; now confirmed installed) | yosys+nextpnr n=16 Ising synthesis + openFPGALoader flash | gatemate_bitstream_flashed=true |
| PolarFire SoC | 1514:2008 + ssh polarfire | exp1680: FABRICATED (run_duration_s=0, flagged_adversarial) | Precondition-gated SSH smoke, no fabrication | polarfire_workload_validated=true |

---

## Phase Structure

### Phase 0 — Admin (exp2447)
Archive .236 to research-complete.yaml, activate .237.

### Phase 1 — AUROC Final Breach (exp2448, exp2449, exp2450)
Three parallel experiments:
- exp2448: Fix conformal ensemble JSON + add PCIB scores
- exp2449: HalluField Tier 0m implementation + evaluation
- exp2450: LaaB full meta-judgment update

### Phase 2 — FR-11 + Free-Energy Routing (exp2451, exp2455)
- exp2451: FR-11 soundness/completeness tracking (MANDATORY per program)
- exp2455: ODAR-style free-energy routing for verify-repair iteration selection

### Phase 3 — Hardware Continuity (exp2452, exp2453, exp2454)
One task per attached board (mandatory):
- exp2452: KV260 RTL debug (claude+opus for multi-file hardware debugging)
- exp2453: GateMate synthesis + flash attempt v2
- exp2454: PolarFire proper smoke (no fabrication)

### Phase 4 — Corrigendum + Synthesis (exp2456, exp2457, exp2458)
- exp2456: Fix NCO tautology from exp2444
- exp2457: Paper-v6 capstone synthesis (gated on exp2448)
- exp2458: Milestone retro

---

## Dependency Graph

```
exp2447 → (ungated)
exp2448 → (ungated, Phase 1)
exp2449 → (ungated, Phase 1)
exp2450 → (ungated, Phase 1)
exp2451 → (ungated, FR-11 mandatory)
exp2452 → (ungated, hardware)
exp2453 → (ungated, hardware)
exp2454 → (ungated, hardware)
exp2455 → (ungated, ODAR routing)
exp2456 → (ungated, corrigendum)
exp2457 → gated: exp2448.ensemble_auroc_improved==true
exp2458 → (ungated, retro)
```

Critical path: exp2448 → exp2457 (2 hops, ~2 hours wall time if codex succeeds)

---

## FR-11 Mandate Fulfillment

FR-11 (continuous self-learning) is fulfilled by:
- exp2451 (FR-11 Soundness/Completeness Tracking v5): `continuous_self_learning_task: true`
  Mechanism: Online tracking of soundness errors (missed violations) vs completeness errors
  (false alarms) per arXiv:2603.03538, with Littlestone-dimension-bounded mistake budget.
  Gate: `soundness_tracking_enabled: true AND completeness_tracking_enabled: true AND
         n_violations_processed >= 10`

---

## Decentralization Check (Rules 1-7)

1. **Local-first**: All verifiers (conformal ensemble, HalluField, LaaB) run CPU-only.
   No closed-weight models required. ✓
2. **Closed-weight optional**: No closed-weight LLM dependencies in Phase 1-3 tasks. ✓
3. **Distribution mirroring**: Phase 1 ship gate already met (PyPI + HF mirror). ✓
4. **Multiple surfaces**: MCP + CLI + Python API all documented (exp2441 verified). ✓
5. **Hardware portability**: KV260 + GateMate + PolarFire tracks all use open toolchains. ✓
6. **Data minimization**: No closed-weight calls in this milestone. ✓
7. **No vendor abstractions in core**: Verifiers in python/carnot/verify/ only. ✓

---

## Exclusion Manifest Cross-Check

Retired experiments checked against all .237 task scopes:
- exp2091 (gemini CLI bail-out): NO gemini tasks in .237. ✓
- HalluSAEGeometricProbe: exp2449 is HalluField (field-theoretic), NOT SAE geometry. Distinct. ✓
- discriminative JEPA OOD (exp887/783/799/804/809/825): no JEPA in .237. ✓
- slowest-5 retirements (exp260/308/309/346/380-383/410/425/491/527/603/627): none re-proposed. ✓

No .237 task scope matches any retired experiment_id. ✓

---

## Agent Routing

| Exp | Agent | Model | Justification |
|-----|-------|-------|---------------|
| exp2447 | codex | gpt-5.5 | Admin/archive — formulaic |
| exp2448 | codex | gpt-5.5 | Fix known JSON bug + eval — formulaic with prior artifact |
| exp2449 | codex | gpt-5.5 | New verifier implementation — follows established pattern |
| exp2450 | codex | gpt-5.5 | Verifier update — established pattern |
| exp2451 | codex | gpt-5.5 | FR-11 extension — established NSVIF pattern |
| exp2452 | claude | opus | requires_claude: KV260 hardware + codex failed twice + 18-file debugging |
| exp2453 | codex | gpt-5.5 | GateMate hardware (toolchain confirmed; formulaic synthesis) |
| exp2454 | codex | gpt-5.5 | PolarFire SSH smoke (precondition-gated; formulaic) |
| exp2455 | codex | gpt-5.5 | ODAR routing implementation — follows FST/CASAL pattern |
| exp2456 | codex | gpt-5.5 | Corrigendum — single-file fix |
| exp2457 | claude | opus | requires_claude: capstone synthesis, 12+ artifacts, open-ended framing |
| exp2458 | codex | gpt-5.5 | Retro — templated |

**2/12 tasks claude (16.7%)** — both meet positive criterion (hardware+multi-file; capstone synthesis).

---

## Hardware Requirements

- KV260 FPGA: Vivado NOT required (Yosys-only synthesis). yosys must be installed.
- GateMate A1-EVB-2M: OSS CAD Suite at /opt/oss-cad-suite (confirmed 2026-05-14).
  openFPGALoader -c dirtyJtag -b olimex_gatemateevb (onboard programmer at 1209:c0ca).
- PolarFire SoC: `ssh polarfire` must succeed (IP: 192.168.51.197 / mpfs-disco-kit.local).
  Python 3.12.12 + pip3 available on PolarFire.
- RTX 3090 GPUs: NOT required for any .237 task (all CPU-feasible).

---

## Failed-Experiment Rerun Compliance

| Task | Prior failure | Root cause | What changed |
|------|---------------|-----------|--------------|
| exp2448 | exp2438 (JSON malformed) | Extra data at char 1999 in JSON output | Fix JSON generation (proper close, no trailing data) |
| exp2449 | exp571 (HalluField Tier 0e, succeeded) | N/A (prior succeeded, new scope) | exp571 was contextual check; exp2449 is standalone Tier 0m with 36-entry evaluation |
| exp2451 | exp2439 (non-standard verdict) | Verdict "Terminal-prefix required..." — malformed | Use correct terminal prefix + soundness/completeness tracking |
| exp2452 | exp2413, exp2427 (synthesis_errors=1) | RTL content bug in 18 files | Claude+opus explicitly debugs which file, with 18-file read choreography |
| exp2453 | exp2105 (synthesis_completed=false) | OSS CAD Suite not installed at time | Toolchain confirmed installed (yosys 0.64+149, openFPGALoader v1.1.1) |
| exp2454 | exp1680 (fabricated, run_duration_s=0) | Gemini agent fabricated result | Precondition-gated SSH check BEFORE any inference; codex agent |
| exp2456 | exp2444 (AUROC=0.500 tautology) | Data leakage / degenerate eval | Proper 5-fold CV with stratified split; confirm via adversarial_verify |
| exp2457 | exp2445 (gate-blocked by exp2438 JSON) | exp2438 JSON unreadable | exp2448 produces valid JSON with ensemble_auroc_improved=true |
